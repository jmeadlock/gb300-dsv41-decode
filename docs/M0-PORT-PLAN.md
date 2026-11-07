# M0 port plan — DeepSeek reference inference → standalone runtime on one GB300

Desk plan, 2026-11-06. Inputs: `reference/deepseek-inference/` (vendored verbatim, 1bae0e10), `docs/checkpoint-1017-inventory.json`, `PROJECT.md`, `w0/results/phase1/`. Op-level detail (shapes at 1 and 6 rows, cites): `docs/m0-forward-graph.json`. This desk run touched no GPU. The sm_103 kernel facts in sections 6 and 8 come from the orchestrator's receipt (`docs/ENVIRONMENT.md`), which landed during this run.

## Summary

1. **Base: yes.** At MP=1 the reference needs no `dist` changes: every collective is behind `world_size > 1` (model.py:169,175,273,323,559,903,1015), and `generate.py` initializes dist only when `WORLD_SIZE > 1` (generate.py:108-112). Three things stop it on one GPU, all outside the math: memory placement, long prefill, and speed.
2. **HBM fit: 325 hot experts/layer at most** (13,000 experts = 227.6 GiB). That leaves 59/layer = 41.3 GiB of experts in Grace, alongside 188.8 GiB of Engram. The charter's existing **295/89 map fits with 21.2 GiB slack**; Grace then pins 251.2 GiB.
3. **Change 1:** our own streaming loader instead of `convert.py`. At MP=1, convert.py holds the whole ~476 GiB checkpoint in host RAM.
4. **Change 2:** a host tier. Engram tables and cold experts go in `cudaHostRegister`ed Grace memory as CUDA alias tensors. Engram's `F.embedding` and the expert `fp4_gemm` both stay unchanged.
5. **Change 3:** an extend path (`start_pos > 0`, s > 1). 32K needs chunked prefill, because single-shot indexer scores need 64 GiB at 32K. M1's 6-row verify reuses the same path.
6. **Top risk: the speed gate.** Reference decode is eager. GEMMs are padded from M=1 to 32 rows, attention runs in one CTA, and a Python expert loop syncs to the host every layer (model.py:896). "≤ Pg61 + 10%" therefore needs graph capture plus borrowed kernels (M0b) inside the same six weeks.
7. **Decision D1.** The reference's own forward quantizes compressed KV (to an NVFP4 shape) and indexer K/Q (to MXFP4) (model.py:546,553,761). Pg61 runs `fp8_ds_mla`. M0 must pick one.
8. **Decisions D2 and D3.** D2: what "1-row-equivalent" means for the speed bar. D3: whether to grow the hot set past 295.
9. Vision and the MTP draft layers are not loaded at M0.
10. The W0 fixture (≤310 tokens) never exercises indexer selection. Long context needs its own test.

Corrections to the handoff:
- The 275.67 GiB of routed experts includes **3 MTP layers × 128 experts**, not one layer of 384 (config.json:6,11; 94,464 tensors / 6 = 15,744 = 40×384 + 3×128).
- Vision plus aligner is **≈0.90 GiB**, not 0.5.
- The inventory carries role totals, per-layer GiB and 6 examples per role, not every tensor. Per-tensor bytes below come from config shapes, checked against those role totals.

## 1. What the reference does per decode step (MP=1)

**TileLang** = kernel.py; everything else is torch. fp8_linear / fp4_linear = TileLang `act_quant` + `fp8_gemm` / `fp4_gemm` (model.py:181-207).

**One token**
1. **Engram hash ids**, computed once for both Engram layers on the GPU: compressed-token map, XOR-rolled multipliers, prime moduli, 3 n-gram sizes × 8 heads = 24 columns (model.py:1255; engram.py:159-184).
2. **Embedding**, expanded to 4 hc copies with an identity pre-mix (model.py:1256-1263).
3. **Per layer 0–39:**
   - a. **Engram (layers 1, 14)** runs before the block. `F.embedding` fetches 24 fp8 rows plus e8m0 scales, dequantizes them to bf16, and feeds the `wkv` fp8_linear (6144→25600). Then: per-copy normalized-dot gate, signed sqrt, sigmoid, `h += gate·value` (model.py:312-365, 1265-1266).
   - b. **mHC pre.** The attention mixes are an rms over 20,480 values plus an fp32 `F.linear` by `hc_attn_fn`, then **TileLang `hc_split_sinkhorn`** (20 iterations) for pre/post/comb. The stream is collapsed with the *previous* layer's `ffn_pre`, then RMSNorm (model.py:950-962, 984-986).
   - c. **Attention** (model.py:766-791):
     - q path: `wq_a` (5120→1280), q_norm, `wq_b` (→64×512), RoPE on the last 64 dims;
     - window KV: `wkv` (→512), norm, RoPE, **FP8 QDQ in place** (TileLang `act_quant`), written to ring slot `start_pos%128` (model.py:701-721);
     - compressed KV (ratio 2 on layers 2–19, ratio 1 on 20–39): only the KV sources 2/8/14/20 run the `Compressor` (fp32 gated softmax-pool over 2 tokens; plain bf16 projection at ratio 1, model.py:458-485). Each other layer reads its source's cache;
     - the **indexer** runs on 2/8/14/20/24/28/32/36 (model.py:527-581). Owners append index keys (bf16 `wk`, norm, RoPE, **FP4 QDQ**, E8M0/32). The query is fp8 `wq_b` plus FP4 QDQ; scoring is a **bf16** einsum, relu, then a `weights_proj`-weighted head sum. **Layer 20** keeps 2,048 candidate blocks of 8, and layers 24–36 mask to those blocks (model.py:570-576, 584-611). Selection is top-`min(512,T/r)`, sorted, +128;
     - other layers reuse the published indices (model.py:726-727);
     - the source latent gets RoPE, then **FP4 QDQ with E4M3/16 scales**, then is cached (model.py:752-762);
     - `torch.cat`(window, whole compressed cache) runs every layer, every step (model.py:776-779);
     - **TileLang `sparse_attn`** with a per-head sink (model.py:781);
     - output: inverse RoPE, FP8 QDQ per group, block-diagonal bf16 einsum `wo_a`, `wo_b` fp8_linear (model.py:782-790).
   - d. **mHC post**, then the FFN's mixes. `ffn_pre` goes to the next layer; this layer's FFN collapses with `attn_pre` (model.py:964-968, 988-993).
   - e. **MoE** (model.py:891-906):
     - gate: fp32 `F.linear` →384, `sqrt(softplus)`, top-6 on score+bias (`noaux_tc`: the bias selects, the raw score weights), normalize with `+1e-20`, ×1.5 (model.py:811-829);
     - `bincount().tolist()` host sync;
     - Python loop per active expert: w1 and w3 fp4_linear; up clamped to ±10, gate to ≤10; silu·up·weight; w2 fp4_linear; `y[idx] +=` in fp32, in expert-id order (model.py:843-853, 896-902);
     - plus the FP8 shared expert (model.py:889, 905).
   - f. mHC post (model.py:995).
4. **Head:** collapse, norm, **fp32** `F.linear` →129,280 (bf16 on disk, held fp32: model.py:1008), argmax at T=0 (model.py:1271-1273, 1291-1292).

**DSpark block.** `dspark_block_size` is 5 (config.json:7). `forward_spec` drafts 5 tokens, and the target verifies 6 rows.

- Draft side:
  - `main_proj` (15360→5120) on the hc-means of layers 37–39's attention *inputs* (model.py:1131-1138, 1267-1269);
  - the draft ids are the last token followed by 4 noise tokens (128799);
  - three `DSparkBlock`s (40–42). Each writes `main_x`'s KV into its window ring and attends 5 rows over the ring plus all 5 block rows (model.py:1022-1077). Their MoE is **128 experts, top-3** (config.json:11-12; model.py:142-149);
  - the head is the shared fp32 head, then a **sequential** 5-step Markov loop (bf16 embedding + fp32 head, both [129280,256]), then a confidence head (model.py:1080-1100, 1140-1159).
- **The reference implements neither the 6-row verify nor acceptance** (model.py:129-130). Its decode branches also assume s=1 (model.py:479-482, 542, 719, 757).

**TileLang vs torch:** TileLang covers `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn` and `hc_split_sinkhorn`. Torch covers every norm, RoPE, embedding, every bf16/fp32 Linear (gate, head, compressor, `wk`, `weights_proj`, `hc_*_fn`), both einsums, top-k, the expert loop and the Engram gate.

## 2. How `convert.py` lays weights out for MP=1

- **Renames:** strip `model.`; `self_attn`→`attn`; `mlp`→`ffn`; `weight_scale_inv`→`scale`; `e_score_correction_bias`→`bias` (convert.py:104-126). Its `mtp.*.embed/head` skip finds nothing: the `embed_head` role has n=4, none under `mtp`.
- **Experts keep checkpoint bytes.** `I8 [out, in/2]` is `.view`ed, zero-copy, as `float4_e2m1fn_x2`: two FP4 nibbles per byte along K, with E8M0 scales `[out, in/32]`, block 32 (convert.py:174-181; model.py:219-223). That is 18,800,640 B per expert.
- **Engram is read in place.** At MP=1 the slice is the whole table and nothing is padded (convert.py:135-141).
- **The only value transform is `wo_a`.** FP8×E8M0 becomes **bf16** (convert.py:157-173), +32 MiB per layer. It is exact, since e4m3 times a power of two always fits in bf16.
- **Output:** one file, `model0-mp1.safetensors`, ≈476.6 GiB (475.24 + 43 × 8192×4096 B).

**Recommendation: write our own loader; do not convert.** Three reasons:
- convert.py accumulates every tensor in `state_dicts[0]` before `save_file` (convert.py:83, 107, 146, 182). That is ~476 GiB resident on a 494 GiB, swapless Grace.
- The output would duplicate /models.
- `load_model` (generate.py:133) only fills parameters already allocated on `cuda` (generate.py:130), so it cannot place tiers.

Our loader streams each tensor from the HF shards into its final home (HBM, or a registered host buffer, via `readinto` + `POSIX_FADV_DONTNEED`, with no page-cache doubling). It applies convert.py's renames and the `wo_a` dequant, refuses on missing or unexpected keys, and proves bit-identity: the sha256 of the placed bytes must equal the shard bytes, and for `wo_a` it must equal convert.py's formula on CPU.

## 3. How the reference holds experts and Engram

- **Experts:** `MoE.__init__` materializes an `Expert` (3 Linears) for each of the 384 local experts on the GPU (model.py:875-887). The forward loops over **selected experts only**: host-side counts, then a `torch.where` gather and `y[idx] +=` per expert. Nothing is grouped or fused (model.py:896-902).
- **Engram:** a GPU parameter (`[N,256]` e4m3 + `[N,8]` e8m0), read with `F.embedding` and dequantized on the GPU (model.py:296-325). Hashing is also on the GPU (engram.py:140-184).
- **What breaks first:** `generate.py` builds everything under `torch.device("cuda")` (generate.py:130), and a `Block` allocates attention, then the FFN, then Engram (model.py:930-934). Cumulative allocation reaches ≈199.6 GiB before layer 14's Engram weight (embed 1.23 + ~6.92/layer + layer 1's 94.56). That `torch.empty` of 98,305,579,008 B (model.py:309) then **OOMs during construction**.

## 4. M0 memory plan (one GB300, text-only, 1 row, 32K)

Budget: 249.81 GiB CUDA total (lane measurement; 250.7 GiB visible). One expert is 18,800,640 B = 0.0175095 GiB.

| Role | Bytes (source / arithmetic) | Placement | Why |
|---|---|---|---|
| Routed experts, backbone | 40×384×18,800,640 B = 268.95 GiB | **Hot 295/layer: HBM 206.61 GiB** (fit max 325: 227.62). **Cold 89/layer: pinned Grace 62.33 GiB** (at 325: 59/layer, 41.32) | decision 13 static map (rowmap-static-v1, 295/89 nominal; per-layer counts read at boot) |
| Engram tables L1+L14 | 384,006,168×264 + 384,016,682×264 B = 188.83 GiB | Pinned Grace (registered, mapped) | 24 rows × 264 B = 6.3 KB/token/layer; lane-measured gather 0.04 ms |
| Engram `wkv` + q/k | 2×157,521,920 B = 0.29 GiB | HBM | GEMM every token |
| Embed | 129280×5120×2 B = 1.23 GiB | HBM | |
| Head | 1.23 GiB on disk → **2.47 GiB fp32** (model.py:1008) | HBM | reference-faithful; bf16 later ≈ +70 experts |
| Attention (backbone) | 4.84 GiB (`per_layer_GiB` sum) + wo_a bf16 growth 40×8192×4096 B = 1.25 GiB | HBM | |
| Shared experts (FP8) | 40×35,423,232 B = 1.32 GiB | HBM | |
| Router / indexer / hc_ffn+norms | 0.15 / 0.04 / 0.07 GiB | HBM | `bias_vl` not loaded |
| KV state @32K (bf16, reference form) | window 5 MiB + compressed 3×16384×512×2 + 32768×512×2 = 80 MiB + index K 20 MiB + RoPE tables 16 MiB + hash cache 0.25 MiB = **121 MiB** | HBM | buffer shapes model.py:520-525, 664-680; engram.py:155-157 |
| Cold-expert staging | 2 × 36 × 18,800,640 B = 1.26 GiB | HBM | sized for M1's 6 rows (M0 alone: 0.21) |
| Activations + CUDA graphs | 8.00 GiB | HBM | handoff constraint |
| Context / cuBLAS / TileLang workspaces / slack | 1.00 GiB (assumption; measure at T1) | HBM | |
| MTP layers (3×128 experts, attn, shared, main_proj, Markov heads) | 7.54 GiB | **Not loaded at M0**; HBM at M1 from slack | M0 is 1-row |
| Vision tower + aligner | 0.25 + 0.52 + 0.14 = 0.90 GiB | **Not loaded at M0** | text-only parity first |

Non-expert residents (11.77 GiB), staging, headroom and context total **22.04 GiB**, leaving 227.77 GiB for experts: **13,008 → 325/layer**. The 295 map leaves 21.16 GiB slack. M1's MTP takes 7.54 of it, and the remaining ~13.6 GiB could grow the hot set to ~314/layer once a decode-calibrated ranking exists (D3).

Grace pins 251.17 GiB at 295 (230.16 at 325) of 494 GiB. The "≤ ~250 GiB pinned" lane rule came from vLLM's 1.6× mmap peak. Our streaming loader avoids that peak, but boot still asserts `MemAvailable` before and after pinning.

## 5. Changes to run single-GPU with host-resident tiers (smallest first)

**(a) dist/MP: run at world size 1.** Nothing to delete for M0; the guards make the collectives dead code. The config overlay sets:
- `vision_n_layers=0` (no ViT, aligner or `bias_vl`; model.py:139-140, 809, 1219);
- `dspark_block_size=0` and `dspark_target_layer_ids=()` (model.py:1212, 1268);
- `max_batch_size=1` and `max_seq_len=32768` (model.py:51-52).

Also vendor the checkpoint's `encoding/` directory, which `generate.py` imports from `../encoding` and is missing here (generate.py:15-24).

**(b) Engram in pinned host memory.**
1. Allocate each table with plain host memory, **not** torch `pin_memory`: its caching host allocator rounds to powers of two, so 94 GiB would become 128 GiB.
2. Stream the bytes in.
3. `cudaHostRegister(Mapped|ReadOnly)`, then `cudaHostGetDevicePointer` (cuda-python).
4. Wrap the device pointer as a CUDA tensor (DLPack or `__cuda_array_interface__`) and assign it to `weight`/`scale`.

`forward` is untouched. `F.embedding` runs torch's index_select over C2C, and the dtype path is the reference's own (e4m3→fp32 × e8m0 → bf16, model.py:317-320). It is bit-identical by construction. Test: `torch.equal` against an HBM copy on 10⁶ random ids. If fp8 `index_select` is missing, gather `.view(torch.uint8)` instead. Boot asserts compressed vocab = 99,092 (engram.py:146), a free tokenizer check.

**(c) Experts split.** Hot experts live in per-layer HBM slot pools in reference layout (contiguous `[N,K/2]` + `[N,K/32]`, as `fp4_gemm` asserts, kernel.py:573-582). Cold experts live in one registered host pool with CUDA aliases. A device table `loc[40][384]` maps expert id to hot slot or cold row.

- **M0a (eager):** the loop already knows its active experts on the host (model.py:896). Each cold one is copied into staging with `copy_(non_blocking=True)` on the copy engine over C2C, then the unchanged `Expert.forward` runs on staging views. The math is exactly the reference's.
  - A free second arm: point cold `Expert`s straight at the aliases and let `fp4_gemm` read over C2C. With 18 CTAs at M=1 it is probably latency-bound. Measure both.
- **M0b (graph-capturable):**
  - device-side ids → `loc` → fixed-length cold list;
  - a deterministic gather kernel (vector loads, no atomics) fills the staging slots;
  - ids are remapped into one slot space, hot `[0,H)` plus staging `[H,H+S)`, so a borrowed grouped MXFP4 MoE sees an ordinary expert tensor;
  - **this is recipe A's self-fetch seed:** at M2 the copy moves into the GEMM producer, which loads straight from host-mapped addresses.

Verify the traffic, do not assume it. At a peer-measured ~4.8% decode cold share, ~11.5 cold experts per token ≈ 216 MB over C2C. Add a per-layer cold-hit counter on day one.

**(d) KV for 32K.** The buffers are 121 MiB in bf16. Decision 6's records are exact encodings of the reference's QDQ'd values (window 512 e4m3 + 16 e8m0 = 528 B; compressed 256 B FP4 + 32 E4M3 = 288 B; index K 68 B). Packing can wait until M2, behind a `torch.equal` round-trip test written now.

The real 32K blocker is prefill. Single-shot bf16 indexer scores `[S,32,S/r]` plus one same-size temporary (model.py:557-558) need 64 GiB each at S=32K, r=1. Single-shot therefore caps at about 7K tokens (2 × S·32·S·2 B ≤ 6 GiB). The fix is an **extend path**: `start_pos > 0` with s > 1. It generalizes model.py:422-424, 476-482, 542, 568, 719 and 757, with offset causal masks. Prefill then runs in fixed 2,048-token chunks; the chunk size becomes part of the numerics contract. M1's 6-row verify is the same path. The same work replaces the per-layer `torch.cat` (model.py:778) with a two-source `sparse_attn`, which gives identical values in identical order.

**(e) Greedy, 1-row parity mode.**
- `--temperature 0` → `argmax`, ties to the lowest id (generate.py:123; model.py:1291-1292).
- One prompt per call. Batch mode prefills only to the shortest prompt (generate.py:64-76).
- The driver adds teacher forcing (prefill the prompt, then force Pg61's tokens), dumps fp32 top-20 per position, and runs with `use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**(f) Profiling from day one.**
- NVTX ranges: `step` → `L{i}` → `{engram, attn, moe}`, then `head`.
- `torch.cuda.profiler.start/stop` around N steps, for `nsys --capture-range=cudaProfilerApi`.
- NVTX does not survive CUDA-graph replay, so M0b adds a **once-per-step marker kernel**. Segmenting on a single kernel was the W0 lesson.
- `w0/w0_gaps.py` and `step_profile_analyze.py` read the result unchanged.

## 6. Kernel map

| Reference kernel | Computes | 1 row | 6 rows | Borrowed later | Bit-identical risk |
|---|---|---|---|---|---|
| `act_quant` (kernel.py:40-124) | FP8 e4m3, pow2 E8M0/32, amax floor 1e-4; inplace = QDQ | M=1 in a 32-row tile; N 512–25600 | M=6 | fused MXFP8 GEMM prologue / FlashInfer `mxfp8_quantize` | **Medium**: floor + ceil-log2 rounding must match (kernel.py:76-78) |
| `fp4_act_quant` (127-204) | FP4 QDQ, E8M0/32 (indexer) or E4M3/16 (compressed KV) | q `[32,128]`; k/latent 1 row | 6× | vLLM indexer / `fp8_ds_mla` writers | **High under D1** |
| `fp8_gemm` (207-307) | fp8×fp8, 32×32 weight blocks, fp32 in-order K accumulate | M=1→32; N=32768 → 256 CTAs; N=512 → 4 | M=6 | FlashInfer MXFP8 GEMM; 32×32→1×32 scale broadcast (exact re-tile) | Medium: block-scaled MMA reorders accumulation |
| `fp4_gemm` (477-591) | fp8 act × fp4 weight (fp4→fp8 via fp32, exact) | per expert M=1; w1/w3 **18 CTAs**, w2 40 | ≤36 experts | FlashInfer MXFP4 routed MoE (lane C1 winner at M≈6) or DeepGEMM MegaMoE, via slot remap | **High**: fused sum order ≠ expert-id loop; keep bf16 rounding of w1/w3 outputs (kernel.py:583) |
| `sparse_attn` (310-403) | top-k gather, online softmax, per-head sink, -1 masked | **1 CTA**, 64 heads, topk 128+min(512,T/r) | 6 CTAs | FlashMLA sparse decode (mode 1), #63921 | Medium: split-KV reorders softmax |
| `hc_split_sinkhorn` (406-474) | pre/post/comb + 20 Sinkhorn iterations, fp32, in-thread | n=1, 2 calls/layer | n=6 | DeepGEMM mega_mhc / lane TileLang `mhc_*` | Low-medium: fusing moves the 20480-long dot's order |

**On sm_103, none is unusable.** All 7 smoke cases compile and run on the GB300, cap (10,3), under torch 2.13.0+cu130 and tilelang 0.1.12 (`docs/ENVIRONMENT.md`, `docs/kernel-smoke-gb300-20261106.json`, committed after my first draft). That includes my prior worry: `sparse_attn` at **h=64, d=512, topk=512**, rel_err 0.0021. MP=1 puts all 64 heads in one CTA (~200 KiB of fixed smem), but it fits.

What the smoke does **not** yet cover:
- `fp8_gemm` and `act_quant` were tested at block 128 with fp32 scales. The model uses **block 32 with E8M0** (model.py:27-30, 187-205).
- The inplace QDQ variants were not run (model.py:708, 787).
- Neither was the E4M3/16 FP4 path for compressed KV (model.py:761).
- `fp4_gemm` got a structure test only (random nibbles, M=64).
- There were no M=1 shapes.

T1's remainder covers these, with real expert bytes for `fp4_gemm`. Fallback for any failure, M0a only: a torch dequant path. M0 ships the reference kernels; the borrowed ones arrive in M0b, one swap at a time, each re-gated on parity. The borrowed ones (`deep_gemm`, `flash_mla`) sit inside vllm's tree in this image. Load their extension modules without importing the `vllm` package, which keeps decision 3 intact.

## 7. Parity instrument

**Fixture.** W0: 8 chat prompts, thinking off, greedy, 1,476 Pg61 positions with top-20 logprobs (`parity-A.jsonl`). The rendered prompt lengths must equal Pg61's `usage.prompt_tokens` (42/47/54/38/34/49/46/31). Pg61 rendered with Garner's `overlay-1017` encoder, so diff that encoder first.

**What Pg61's own data shows.**
- Logprob gaps below 1.0 are multiples of 0.125, so its logits are bf16-quantized.
- 18 positions are exact ties.
- A vs A2 matches on 7/8; `code` diverges at token 121, where A's margin is 0.25 and A2 has an exact tie.
- Shared-prefix chosen-token |Δlogprob|: p99 0.00044, max 0.12.
- Classes in A: confident (p₁ ≥ 0.9) 1,064, mid 341, near-tie (≤0.25) 53, tie 18.

**Three measurements:**
1. **Self-replay:** two runs in one process plus one relaunch. `torch.equal` on all logits and tokens; pass = 100% (decision 14).
2. **Teacher-forced vs Pg61-A:** M0's fp32 top-1 vs Pg61's top-1 at every position, counted by Pg61's margin class, ties credited 1/k. Also reported after rounding M0's logits to bf16. Pass = **0 confident flips**; the control has 0 inside its shared prefix. Mid and near-tie flips are reported only.
3. **Free-run vs A and A2:** pass = every first divergence lands where either side's TF margin is ≤ 0.25. A byte-identity count is reported, not gated, since Pg61 itself manages only 7/8.

**What this fixture cannot test.** At ≤310 tokens there are ≤155 compressed entries at ratio 2, so the top-512 and layer-20 candidate pruning (which needs more than 16,384 positions) never select. A long-context gate needs a 20–32K needle at several depths, self-replay equality, and a two-chunk-size greedy comparison (reported, not required bit-equal).

## 8. Risks and unknowns (ranked)

1. **Speed gate within six weeks.** Eager decode runs thousands of launches and 40 host syncs per token (model.py:896), with dynamic shapes (model.py:555, 579; window indices rebuilt every step, 409-426), M=1 GEMMs padded to 32, and one-CTA attention. My unmeasured estimate is tens of ms/token. Hitting "≤ Pg61 + 10%" needs all of M0b.
2. **D1 numerics.** The reference quantizes compressed KV and index K/Q to FP4 as part of the forward (model.py:546, 553, 761; kernel.py:161, "Training's compressed KV"). Pg61 runs `fp8_ds_mla` (`ds41-args.json`). James ruled out `nvfp4_ds_mla`/mxfp4 indexer as *cache* choices; here they are the model's own math. Whether vLLM applies the same QDQ before its FP8 store is unknown from the desk (T0 answers it). It matters even on the short fixture, because every compressed entry is attended.
3. **TileLang 0.1.12 on sm_103, narrowed.** All 7 smoke cases pass, including 64-head `sparse_attn` (ENVIRONMENT.md). What remains is the model's exact configs: block-32 E8M0 `fp8_gemm`, the inplace QDQs, E4M3/16 FP4, real-weight `fp4_gemm`, M=1. These are now a T1 item, not a structural unknown.
4. **Host memory:** 251 GiB pinned at 295/89 on a swapless box with two prior OOM deaths, plus the `pin_memory` rounding trap. `cudaHostRegister` time on 94 GiB is unknown.
5. **Prompt rendering:** `encoding/` is not vendored, and Pg61 rendered with Garner's overlay.
6. **The extend path is new code** with no reference ground truth; the chunk size becomes numerics.
7. **torch:** the image has 2.13.0+cu130 (≥ 2.10 required, requirements.txt:1), and the FP4/E8M0 dtypes work (smoke). Still untested: `F.embedding` (index_select) on e4m3 and e8m0 tensors, which Engram needs. The fallback is a uint8 view.
8. **`norm_eps` 1e-20** (config.json:23) is a no-op in fp32. Borrowed norm/mHC kernels that hardcode 1e-6 (or hold eps in fp16) change low-variance rows. Keep it distinct from `hc_eps` 1e-6 and the gate's literal `+1e-20` (model.py:827).
9. **MP>1 assumptions** are not in the math. They are all-on-device construction (generate.py:130) and convert.py's whole-checkpoint RAM. The 64-heads-per-CTA `sparse_attn` is cleared by the smoke.
10. **bf16 indexer scores** (model.py:556-558) make top-k ties common. A borrowed top-k must reproduce `torch.topk` tie order, or long-context parity drifts.
11. **Decision 8's stack has no TileLang.** It stays as the reference's dependency for M0 only.

## 9. Work breakdown (Nov 9 → Dec 18; ~28 working days, ~26 planned)

| # | Task | Days | Depends | Done when |
|---|---|---|---|---|
| T0 | Desk: read vLLM v0.31.0 DSV4.1 compressed-KV/indexer write path (read-only); vendor checkpoint `encoding/`; render 8 W0 prompts | 0.5 | — | D1 memo; counts = 42/47/54/38/34/49/46/31 — **2026-11-06: encoding vendored, counts match 8/8 (ENVIRONMENT.md); D1 memo in progress** |
| T1 | Env | 0.5 | — | **DONE 2026-11-06** (ENVIRONMENT.md): image/versions, 7 kernels on sm_103, block-32 E8M0 fp8_gemm + real-expert fp4_gemm at M=1/6 (rel_err 0.0017), inplace QDQs, fp8/e8m0 `F.embedding` all pass. Open: `python model.py` import smoke, context MiB |
| T2 | Streaming loader (renames, wo_a dequant, fp4 view, placement, refuse-on-mismatch) | 3 | T1 | placed-byte sha256 = shard bytes for 100% of tensors; host peak ≤ pinned + 8 GiB — **DONE 2026-11-06** (LEDGER) |
| T3 | Host tier: register + alias, Engram and cold pool | 1.5 | T1 | alias vs HBM `torch.equal` on 10⁶ ids; gather µs logged |
| T4 | Expert split M0a (staging and zero-copy arms), rowmap load | 2 | T2, T3 | layer-level `torch.equal` vs all-HBM experts; cold-hit counter live |
| T5 | Driver: greedy/TF/dumps/NVTX/profiler/determinism | 1.5 | — | runs on a 2-layer toy config |
| T6 | **First light:** 8 prompts, measurements 1–3 | 2 | T0, T2–T5 | parity report in `docs/`; self-replay 100% (target **Nov 24**) |
| T7 | Extend path + two-source `sparse_attn`; 32K | 4 | T6 | 32K prompt prefills and decodes 256 tokens; needles pass; self-replay equal |
| T8 | Graph-capturable step (device routing, fixed shapes, marker kernel) | 3 | T6 | full decode step from one CUDA graph; parity unchanged — **DONE 2026-11-06, 98.4 ms (T8-RESULT)** |
| T9 | Borrowed kernels one at a time: MoE → attention → MXFP8 GEMM → mHC | 6 | T8 | each swap re-passes parity; ms/step logged per swap — **T9a-1 DONE 2026-11-06, 44.6 ms (T9A1-RESULT)** |
| T10 | Trace + gap histogram + M0 verdict | 1.5 | T9 | nsys step table with named boundaries; histogram; verdict vs D2 bar |

There are two days of slack. If T6 has not passed by Dec 1, Milo flags a projected miss at the Dec 4 week-4 check (decision 17). A second worker can run T7 in parallel with T8.

**Decisions for James** (none blocks T0–T6):
- **D1:** compressed-KV / index-K numerics — reference FP4 QDQ or Pg61's `fp8_ds_mla`.
- **D2:** "1-row-equivalent". Proposed: M0's 1-row step ≤ 1.1 × 7.36 = 8.10 ms.
- **D3:** keep 295/89 through M1, or grow toward ~314 after MTP.

M0-PLAN: ready — the reference runs at MP=1 unchanged in its math; the work is placement, an extend path, and graph/kernel swaps, with three decisions for James (D1 numerics, D2 speed bar, D3 map) that do not block T0–T6.
