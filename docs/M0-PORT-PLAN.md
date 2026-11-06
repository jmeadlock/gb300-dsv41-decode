# M0 port plan — DeepSeek reference inference → standalone runtime on one GB300

Desk plan, 2026-11-06. Inputs: `reference/deepseek-inference/` (vendored verbatim, 1bae0e10), `docs/checkpoint-1017-inventory.json`, `PROJECT.md`, `w0/results/phase1/`. Machine-readable op list: `docs/m0-forward-graph.json`. Nothing here was run on a GPU.

## Summary

1. **Base: yes.** At MP=1 the reference needs no `dist` changes: every collective is behind `world_size > 1` (model.py:169,175,273,323,559,903,1015), and `generate.py` only calls `init_process_group` when `WORLD_SIZE > 1` (generate.py:108-112). It is DeepSeek's own forward. Three things stop it on one GPU, and all three sit outside the math: memory placement, long prefill, and speed.
2. **HBM fit: 325 hot experts/layer at most** (13,000 experts = 227.6 GiB). That is what is left after 11.77 GiB of non-expert weights+KV, 8 GiB headroom, 1.0 GiB context and 1.26 GiB staging. The Grace remainder at that fit is 59/layer = 41.3 GiB, on top of 188.8 GiB of Engram. The charter's existing map, **295/89, fits with 21.2 GiB of HBM slack**, and Grace then pins 251.2 GiB.
3. **Change 1:** our own streaming loader instead of `convert.py`. At MP=1, convert.py holds the whole ~476 GiB checkpoint in host RAM before it saves anything.
4. **Change 2:** a host tier. Engram tables and cold experts go in `cudaHostRegister`ed Grace memory, exposed to the GPU as CUDA alias tensors. Engram keeps its unchanged `F.embedding`. Cold experts are staged into HBM before the unchanged `fp4_gemm`.
5. **Change 3:** an *extend* path (`start_pos > 0`, more than one row). 32K needs chunked prefill, because single-shot indexer scores need 64 GiB at 32K. M1's 6-row verify reuses the same path.
6. **Top risk: the speed gate.** The reference decodes eagerly. It pads every GEMM from M=1 to 32 rows, runs sparse attention in one CTA, and loops over experts in Python with a host sync per layer (model.py:896). "≤ Pg61 + 10%" therefore depends on graph capture plus borrowed kernels landing inside the same six weeks (M0b).
7. **Decision for James (D1).** The reference *itself* runs compressed KV through FP4 (NVFP4-shaped) and indexer K/Q through MXFP4 (model.py:546,553,761). Pg61 runs `fp8_ds_mla`. Which numerics does M0 implement?
8. Two smaller decisions: what "1-row-equivalent" means (D2), and whether to keep the 295 map or go toward 325 (D3).
9. Vision is not loaded at M0, and neither are the MTP draft layers. The first is text-only parity; the second is M1.
10. The W0 fixture (prompts of 31–54 tokens, ≤310 total) never exercises indexer selection: every compressed entry fits inside top-512. Long-context correctness needs its own test.

Corrections to the handoff's facts, from the inventory:
- The 275.67 GiB of routed experts includes **3 MTP layers × 128 experts**, not one layer of 384 (config.json:6,11). The check is 94,464 tensors / 6 = 15,744 = 40×384 + 3×128.
- The vision tower plus aligner is **≈0.90 GiB**, not 0.5.
- The inventory holds per-role totals, per-layer GiB and 6 example tensors per role, not every tensor. Per-tensor bytes below come from config shapes, checked against those role totals.

## 1. What the reference does per decode step (MP=1)

Each op is marked as a **TileLang** kernel (kernel.py) or **torch**. "fp8_linear" means `act_quant` + `fp8_gemm`; "fp4_linear" means `act_quant` + `fp4_gemm` (model.py:181-207).

**One token (b=1, s=1)**

1. Engram hash ids for both Engram layers, computed once: torch, `[1,1,2,24]` int64 (model.py:1255; engram.py:159-184). Compressed-token map, XOR-rolled multipliers, prime moduli, 3 n-gram sizes × 8 heads = 24 columns.
2. Embedding, expanded to 4 hc copies, with an identity pre-mix (model.py:1256-1263).
3. For each layer 0–39:
   - a. **Engram (layers 1, 14)** runs *before* the block, on the hc stream (model.py:1265-1266). Steps:
     - torch `F.embedding` of 24 fp8 rows plus their e8m0 scales, dequantized to bf16 (model.py:317-320);
     - `wkv` fp8_linear, K=6144 → N=25600, split into 4 keys and 1 value;
     - per-copy normalized dot product, signed-sqrt sigmoid gate, `h += gate·value` (model.py:353-365).
   - b. DSpark capture (layers 37–39, only when DSpark is on): the hc-mean of the attention *input* (model.py:1267-1269).
   - c. **mHC pre**:
     - `hc_mixes` is an rms over the flattened 20,480 values plus an fp32 `F.linear` by `hc_attn_fn[24,20480]`, both torch (model.py:950-956);
     - then **TileLang `hc_split_sinkhorn`** gives pre/post/comb with 20 Sinkhorn iterations (kernel.py:406-474);
     - collapse with the *previous* layer's `ffn_pre` (model.py:984-985), then RMSNorm with eps 1e-20.
   - d. **Attention** (model.py:766-791):
     - q path: `wq_a` fp8_linear (5120→1280), q_norm, `wq_b` fp8_linear (1280→32768), RoPE on the last 64 of 512 dims;
     - window KV: `wkv` fp8_linear (→512), kv_norm, RoPE, **FP8 QDQ in place over all 512 dims** (TileLang `act_quant`, inplace), written to ring slot `start_pos%128`, attended oldest-first (model.py:701-721, 410-426);
     - compressed KV (ratio 2 on layers 2–19, ratio 1 on 20–39):
       - only the KV sources 2/8/14/20 run the `Compressor`: fp32 wkv/wgate, a two-slot state, and a softmax-pool every second step at ratio 2 (model.py:458-485);
       - each layer reads the cache of the last source at or before it.
     - **Indexer** on the index sources 2, 8, 14, 20, 24, 28, 32, 36 (model.py:527-581):
       - the owners 2/8/14/20 append index keys: bf16 `wk`, k_norm, RoPE, **FP4 QDQ** with E8M0 block 32;
       - query: `wq_b` fp8_linear (→4096), RoPE, FP4 QDQ;
       - bf16 `weights_proj`;
       - a **bf16** einsum score, relu, then a head-weighted sum;
       - **layer 20** keeps the top 2,048 eight-wide candidate blocks (model.py:570-573, 584-611), and layers 24–36 mask to those blocks (model.py:574-576);
       - top-`min(512,T/r)`, sorted, offset by 128.
     - Layers in between reuse the published `topk_idxs` (model.py:726-727).
     - The source's latent gets RoPE and then an **FP4 QDQ with E4M3 block-16 scales** before it is cached (model.py:752-762).
     - `torch.cat`(window, compressed) copies the whole compressed cache on every layer, every step (model.py:776-779).
     - **TileLang `sparse_attn`**, with a per-head sink (model.py:781; kernel.py:310-403).
     - Output: inverse RoPE, FP8 QDQ per group, block-diagonal bf16 einsum `wo_a`, then `wo_b` fp8_linear 8192→5120 (model.py:782-790).
   - e. mHC post (model.py:964-968). Then `hc_mixes` for the FFN, whose `ffn_pre` goes to the *next* layer; collapse with `attn_pre`; RMSNorm (model.py:990-993).
   - f. **MoE** (model.py:891-906):
     - **Gate:** fp32 `F.linear` →384, `sqrt(softplus)`, top-6 on score+bias (`noaux_tc`: the bias selects but does not weight), normalize with `+1e-20`, ×1.5 (model.py:811-829).
     - **Host sync:** `bincount(...).tolist()` (model.py:896).
     - **Experts:** a Python loop over the active ones. For each: `torch.where` gather; w1 and w3 fp4_linear; up clamped to ±10, gate to ≤10; silu·up·weight; w2 fp4_linear; `y[idx] +=` in fp32, in expert-id order (model.py:843-853, 897-902).
     - **Shared expert:** FP8, same clamped SwiGLU (model.py:889, 905).
   - g. mHC post (model.py:995).
4. Head: collapse with layer 39's `ffn_pre`, then norm, then an **fp32** `F.linear` →129,280 (the weight is held in fp32, bf16 on disk; model.py:1008), then argmax at T=0 (model.py:1271-1273, 1291-1292).

**DSpark block.** The checkpoint has `dspark_block_size` 5 (config.json:7). `forward_spec` drafts 5 tokens, and the target then verifies 6 rows: the last token plus 5 drafts.

- Draft side:
  - `main_proj` fp8_linear (15360→5120) on the layer 37/38/39 captures, then `main_norm` (model.py:1131-1138);
  - draft ids are the last token followed by 4 noise tokens (128799);
  - three `DSparkBlock`s (layers 40–42). Each writes `main_x`'s KV into its window ring and attends 5 draft rows over the ring plus all 5 block rows (model.py:1022-1031, 1034-1077). Their MoE has **128 experts, top-3** (config.json:11-12; model.py:142-149);
  - `forward_head`: the shared fp32 head gives `[1,5,129280]`, followed by a *sequential* 5-step Markov loop. Each step adds a bias from a bf16 embedding `[129280,256]` and an fp32 head `[129280,256]`, then samples. A confidence head (fp32, 5376→1) runs last (model.py:1080-1100, 1140-1159).
- **Not in the reference:** the 6-row target verify and acceptance ("nothing calls forward_spec", model.py:129-130). The decode branches also assume s=1 (model.py:479-482, 542, 719, 757). The 6-row target step is therefore ours to build, and the shapes in the kernel map assume it.

**TileLang vs torch.** Six ops are TileLang: `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn` and `hc_split_sinkhorn`. Everything else is torch: norms, RoPE (complex multiply), embeddings, every bf16/fp32 Linear (gate, head, compressor, `wk`, `weights_proj`, `hc_*_fn`), both einsums, top-k/sort, the expert loop, and the Engram gate.

## 2. How `convert.py` lays weights out for MP=1

- **Names.** Strip `model.`, `self_attn`→`attn`, `mlp`→`ffn` (vision excepted), `weight_scale_inv`→`scale`, `e_score_correction_bias`→`bias` (convert.py:104-126). It skips `mtp.*.embed/head`, but this checkpoint has none: the `embed_head` role has n=4 tensors, none of them under `mtp.*`.
- **Experts.** Kept as checkpoint bytes. `I8 [out, in/2]` is reinterpreted (`.view`, zero-copy) as `float4_e2m1fn_x2`: two FP4 nibbles per byte, packed along K, with E8M0 scales `[out, in/32]`, block 32 (convert.py:174-181; model.py:219-223). Per expert, 3×(5,898,240 + 368,640) = **18,800,640 B**. The `fp8` expert option (convert.py:18-53) is not ours.
- **Engram.** Read in place. At MP=1 the shard slice is the whole table and no padding happens (convert.py:135-141). Values and dtype are unchanged.
- **The only value transform is `wo_a`.** FP8 × E8M0 scale is dequantized to **bf16** (convert.py:157-173), which adds 32 MiB per layer. It is exact: e4m3 times a power of two always fits in bf16.
- **Output is one file per rank:** `model0-mp1.safetensors`, ≈476.6 GiB (475.24 + 43 × 8192×4096 B of wo_a growth).

**Recommendation: write our own loader, do not convert.** Four reasons:
- `convert.py` fills `state_dicts[0]` with every tensor (`get_tensor` → dict, convert.py:83, 107, 146) before `save_file` (convert.py:182). That is ~476 GiB resident on a 494 GiB, swapless Grace.
- The output would duplicate /models.
- `load_model` (generate.py:133) can only fill parameters that were allocated on `cuda` (generate.py:130), so it cannot place tiers.
- The loader needs placement logic anyway.

Our loader:
- streams each tensor from the HF shards straight into its final home, an HBM tensor or a registered host buffer, with no page-cache doubling (`readinto` at the safetensors offset, then `POSIX_FADV_DONTNEED`);
- applies convert.py's renames and the `wo_a` dequant;
- refuses on missing or unexpected keys;
- proves bit-identity: the sha256 of the placed bytes must equal the shard bytes, and for `wo_a` it must equal convert.py's formula run on CPU.

## 3. How the reference holds experts and Engram

- **Experts.** `MoE.__init__` builds an `Expert` (3 Linears) for every local expert, which is all 384 at MP=1, so every expert is materialized on the GPU (model.py:875-887). The forward is a dense Python loop over *selected* experts only: host-side counts, then a `torch.where` gather and `y[idx] +=` scatter per expert (model.py:896-902). Nothing is fused or grouped.
- **Engram.** `ParallelEngramEmbedding` is a GPU parameter `[N,256]` e4m3 plus `[N,8]` e8m0, indexed with `F.embedding` and dequantized on the GPU (model.py:296-325). Hashing also runs on the GPU from buffers built at init (engram.py:140-157). The host never touches a row.
- **What breaks first.** `generate.py` builds the model under `torch.device("cuda")` (generate.py:130), and `Block.__init__` allocates attention, then the FFN, then Engram (model.py:930-934). Summing allocation order (embed 1.23 GiB; ~6.92 GiB per layer; layer 1's Engram 94.56 GiB) gives ≈199.6 GiB just before layer 14's Engram weight. Its `torch.empty` of 98,305,579,008 B (91.6 GiB, model.py:309) then **OOMs at construction**. Weights alone are 268.95 GiB of experts plus 188.83 GiB of tables, against 249.81 GiB.

## 4. M0 memory plan (one GB300, text-only, 1 row, 32K)

Budget: 249.81 GiB CUDA total (lane measurement; 250.7 GiB visible). Experts are 18,800,640 B = 0.0175095 GiB each.

| Role | Bytes (source / arithmetic) | Placement | Why |
|---|---|---|---|
| Routed experts, backbone | 40×384×18,800,640 B = 268.95 GiB | **Hot 295/layer: HBM 206.61 GiB** (fit max 325: 227.62). **Cold 89/layer: pinned Grace 62.33 GiB** (at 325: 59/layer, 41.32) | Charter decision 13 static map (rowmap-static-v1, 295/89 nominal; per-layer counts read at boot) |
| Engram tables L1+L14 | 384,006,168×264 + 384,016,682×264 B = 188.83 GiB | Pinned Grace (registered, mapped) | 24 rows × 264 B = 6.3 KB/token/layer; latency-bound, measured 0.04 ms on the lane |
| Engram `wkv` + q/k | 2×157,521,920 B = 0.29 GiB | HBM | GEMM every token |
| Embed | 129280×5120×2 = 1.23 GiB | HBM | |
| Head | 1.23 GiB on disk → **2.47 GiB fp32** (model.py:1008) | HBM | reference-faithful; keeping bf16 is a later lever worth ~70 experts |
| Attention (backbone) | 4.84 GiB (sum of `per_layer_GiB`) + wo_a bf16 growth 40×8192×4096 B = 1.25 GiB | HBM | |
| Shared experts (FP8) | 40×35,423,232 B = 1.32 GiB | HBM | |
| Router, indexer, hc_ffn/norms | 0.15 + 0.04 + 0.07 GiB | HBM | `bias_vl` not loaded (vision off) |
| KV state @32K, bf16 reference form | window 40×128×512×2 = 5 MiB; compressed 3×16384×512×2 + 32768×512×2 = 80 MiB; index K 3×16384×128×2 + 32768×128×2 = 20 MiB; RoPE tables 2×32768×32×8 = 16 MiB; hash cache 0.25 MiB → **121 MiB** | HBM | shapes from model.py:664-680, 520-525; engram.py:155-157 |
| Cold-expert staging | 2 buffers × 36 experts × 18,800,640 B = 1.26 GiB | HBM | sized for M1's 6 rows (M0 alone needs 0.21) |
| Activations + CUDA graphs | 8.00 GiB | HBM | handoff constraint |
| CUDA context, cuBLAS/TileLang workspaces, allocator slack | 1.00 GiB (assumption; measure at T1) | HBM | |
| MTP draft layers (3×128 experts, attn, shared, main_proj, Markov heads) | 7.54 GiB (arithmetic in section 2's units) | **Not loaded at M0**; HBM at M1, out of the slack | M0 is 1-row |
| Vision tower + aligner | 0.25 + 0.52 + 0.14 = 0.90 GiB | **Not loaded at M0** (`vision_n_layers=0`) | text-only parity first |

The fit: non-expert residents 11.77, plus staging 1.26, headroom 8 and context 1, total **22.04 GiB**. That leaves 227.77 GiB for experts: **13,008 experts → 325/layer uniform**. At the 295 map the HBM slack is 21.16 GiB. M1's MTP takes 7.54 GiB of it, and about 13.6 GiB remain to grow the hot set to roughly 314/layer once a decode-calibrated ranking exists (D3).

The Grace side pins 251.17 GiB at 295 and 230.16 GiB at 325, out of 494 GiB. The lane rule "pinned set ≤ ~250 GiB" came from vLLM's 1.6× mmap peak. The streaming loader removes that peak, but boot still has to assert `MemAvailable` before and after pinning.

## 5. Changes to run single-GPU with host-resident tiers (smallest first)

**(a) dist/MP.** Run at world size 1. `python generate.py` without torchrun, or our driver. There is nothing to delete for M0: the guards already make the collectives dead code, and `ParallelEngramEmbedding`'s all-false mask is harmless (model.py:313-321). Our config overlay sets:
- `vision_n_layers=0`, which skips ViT, aligner and `bias_vl` (model.py:139-140, 809, 1219);
- `dspark_block_size=0` and `dspark_target_layer_ids=()`, so no MTP and no captures (model.py:1212, 1268);
- `max_batch_size=1` and `max_seq_len=32768` (model.py:51-52).

Also vendor the checkpoint's `encoding/` directory. `generate.py` imports `../encoding` (generate.py:15-24), and it is not in our repo.

**(b) Engram in pinned host memory.** Five steps:
1. Allocate each table (`[N,256]` u8/e4m3 and `[N,8]` e8m0) with plain host memory. **Not** torch `pin_memory`: its caching host allocator rounds to powers of two, so a 94 GiB table would cost 128 GiB.
2. Stream the shard bytes in.
3. `cudaHostRegister(ptr, size, Mapped|ReadOnly)` via cuda-python, then `cudaHostGetDevicePointer`.
4. Wrap the pointer as a CUDA tensor (DLPack or `__cuda_array_interface__`) and assign it as the module's `weight`/`scale`.
5. `forward` stays unchanged. `F.embedding(cuda_idx, alias)` runs torch's index_select kernel, which reads 24 rows over C2C, and the dtype path is the reference's own (e4m3→fp32 × e8m0→bf16, model.py:317-320). The result is **bit-identical by construction**: same kernel, same bytes, different address.

Test: `torch.equal` against an HBM copy for 10⁶ random ids on a 1 GiB slice. If torch lacks `index_select` for e4m3/e8m0, gather `.view(torch.uint8)` and view back, which is still bit-identical. Boot also checks the tokenizer: the compressed vocab must be 99,092 (engram.py:146; config.json:48).

**(c) Experts split HBM/Grace.** Hot experts live in per-layer HBM slot pools in reference layout (`[N,K/2]` fp4 + `[N,K/32]` e8m0, contiguous, as `fp4_gemm` asserts at kernel.py:573-582). Cold experts live in one registered host pool with CUDA aliases. A device table `loc[40][384]` maps expert id to hot slot or cold row.

- **M0a (eager, parity):**
  - the reference loop already knows the active experts on the host (model.py:896);
  - for each cold one, `staging[j].copy_(host_row, non_blocking=True)` for w1, w3 and w2 (copy engine over C2C), then the unchanged `Expert.forward` on staging views;
  - the math is the reference's exactly: same `fp4_gemm`, same bytes, same expert-id summation order;
  - a second arm costs nothing: point cold `Expert`s straight at the alias tensors and let `fp4_gemm` read over C2C, zero-copy (18 CTAs for w1 at M=1, so expect it latency-bound). Measure both.
- **M0b (graph-capturable):**
  - gate ids stay on the device, and `loc` lookup produces a fixed-length cold list (≤6 per row);
  - a deterministic gather kernel (16-B vector loads, no atomics) copies each cold expert's 18.8 MB into staging slot j;
  - ids are remapped into one virtual slot space, hot `[0,H)` plus staging `[H,H+S)`, so a borrowed grouped MXFP4 MoE sees an ordinary expert tensor (the slot-cache trick).
  - **This is the seed of recipe A's self-fetch:** at M2 the copy disappears into the GEMM producer, which loads tiles straight from host-mapped addresses.

Cost model to verify, not assume: at a peer-measured ~4.8% decode cold-route share, about 11.5 cold experts per token is about 216 MB of C2C per token. Instrument a per-layer cold-hit counter from day one.

**(d) KV for 32K.** The buffers are allocated in `__init__` and come to 121 MiB in bf16 (section 4). Decision 6's records are exact encodings of the reference's QDQ'd values: window 512 e4m3 + 16 e8m0 = 528 B; compressed 256 B fp4 + 32 e4m3 scales = 288 B; index K 64 + 4 = 68 B. Packing can therefore wait for M2, with a `torch.equal` round-trip test written in M0.

The real 32K blocker is **prefill**. Single-shot prefill computes bf16 indexer scores `[S,32,S/r]` plus one same-size temporary (model.py:557-558): 64 GiB each at S=32K, r=1. Within the 8 GiB headroom, single-shot works only up to about 7K tokens (2 × S·32·S·2 B ≤ 6 GiB). The fix is an **extend path**: `start_pos > 0` with s > 1, generalizing:
- the s=1 assumptions: Compressor decode (model.py:476-482), ring write (719), single-latent RoPE (542, 757), scalar `compress_lens` (568) and window indices (422-424);
- prefill causal masking at an offset.

Run prefill in fixed 2,048-token chunks; the chunk size is part of the numerics contract. M1's 6-row verify is the same code path. Also replace the per-layer `torch.cat` (model.py:778) with a two-source `sparse_attn`: the kernel reads window or compressed by index, so values and order are identical.

**(e) Greedy and 1-row parity mode.** `--temperature 0`, so `sample` is `argmax` (generate.py:123; model.py:1291-1292). Ties go to the lowest id, which must be documented. One prompt per process call: batch mode prefills only to the shortest prompt and feeds the rest through decode (generate.py:64-76). The driver adds:
- **teacher-forced mode:** prefill the prompt, then force Pg61's tokens step by step;
- **dumps:** fp32 logits top-20 and top-1/top-2 per position;
- **determinism:** `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, to list any nondeterministic op the reference reaches.

**(f) Profiling from day one.** NVTX ranges: `step` → `L{i}` → `{engram, attn, moe}`, then `head`. `torch.cuda.profiler.start()/stop()` brackets N steps for `nsys --capture-range=cudaProfilerApi --capture-range-end=stop`. NVTX does not survive CUDA-graph replay, so M0b also emits a **once-per-step marker kernel**. Steps are segmented on a single kernel, which is the W0 lesson. `w0/w0_gaps.py` and `step_profile_analyze.py` consume the trace unchanged.

## 6. Kernel map

| Reference kernel | Computes | 1 row | 6 rows | Borrowed later | Bit-identical risk |
|---|---|---|---|---|---|
| `act_quant` (kernel.py:40-124) | FP8 e4m3, pow2 E8M0 block-32 scales, amax floor 1e-4; inplace = QDQ | M=1 (one 32-row tile); N = 5120/1280/2304/6144/8192/512 | M=6 | fused into MXFP8 GEMM prologue / FlashInfer `mxfp8_quantize` | **Medium**: floor and ceil-log2 rounding must match (kernel.py:76-78) or scales move |
| `fp4_act_quant` (127-204) | FP4 QDQ: E8M0/32 (indexer q,k) or E4M3/16 (compressed KV) | q `[32,128]`, k/latent 1 row | 6× | vLLM indexer/`fp8_ds_mla` writers | **High if D1 picks the reference**: Pg61's store is `fp8_ds_mla` |
| `fp8_gemm` (207-307) | A fp8 × B fp8, 32×32 weight blocks, fp32 sequential K-block accumulate | M=1 padded to 32; e.g. N=32768 K=1280 → 256 CTAs; N=512 → 4 CTAs | M=6 | FlashInfer MXFP8 GEMM (lane `mxfp8_gemm`); weight scales broadcast 32×32→1×32 (exact re-tile) | Medium: tensor-core block-scaled MMA changes accumulation order |
| `fp4_gemm` (477-591) | fp8 act × fp4 weight (cast fp4→fp8 via fp32, exact), E8M0/32 | per expert M=1; w1/w3 N=2304 → **18 CTAs**, w2 N=5120 → 40 | ≤36 experts, M=routed rows | FlashInfer MXFP4 routed MoE (lane's C1 winner at M≈6) or DeepGEMM MegaMoE, via slot remap | **High**: fused sum order ≠ expert-id loop; bf16 rounding of w1/w3 outputs (kernel.py:583) must be kept |
| `sparse_attn` (310-403) | gather top-k KV, online softmax, per-head sink, -1 = masked | **1 CTA** (grid = rows × batch, kernel.py:335), 64 heads, topk 128+min(512,T/r) | 6 CTAs | FlashMLA sparse decode (mode 1), #63921 `decode_topk` | Medium: split-KV changes softmax order |
| `hc_split_sinkhorn` (406-474) | pre/post/comb + 20 Sinkhorn iterations, fp32, in-thread | n=1, 2 calls/layer | n=6 | DeepGEMM mega_mhc / lane TileLang `mhc_*` (fuses the 20480-long `hc_fn` dot) | Low-medium: fusing moves the projection's reduction order |

**Is any of them unusable on sm_103?** None is known to be, and none is proven. `pass_configs` disables TMA lowering and warp specialization (kernel.py:9-12), which favours portability. The only TileLang known to run on this box is the vLLM image's fp32 mHC kernels. The **most likely failure** is `sparse_attn` at MP=1. Unlike MP=8's 8 heads padded to 16 (kernel.py:395-397), one CTA holds all 64 heads. Its fixed smem is q 64 KiB + kv tile 64 KiB + o 64 KiB + `acc_s_cast` 8 KiB = 200 KiB before pipelining, against the ~227 KiB per-block limit of this class (check `MaxSharedMemoryPerBlockOptin` at T1). `acc_o` alone is 128 fp32 registers per thread. If it fails, **call it in 4 head-chunks of 16**: heads are independent, and that is the shape DeepSeek actually ran. Next most likely: `fp4_gemm`'s FP4 type codegen. Its fallback is a dequant-to-bf16 torch path, M0a only. M0 ships the reference kernels; borrowed kernels arrive in M0b, one swap at a time, each re-gated on parity.

## 7. Parity instrument

**Fixture.** The W0 fixture: 8 chat prompts with thinking off, greedy, 256 max tokens. Pg61 returned 1,476 completion positions, each with top-20 logprobs (`parity-A.jsonl`). Rendered prompt lengths must equal Pg61's `usage.prompt_tokens`: 42/47/54/38/34/49/46/31. If any count differs, the comparison is void. Pg61 mounts Garner's `overlay-1017` prompt files, so the encoder is diffed first.

**What Pg61's own data shows.**
- Its logprob gaps below 1.0 come in multiples of 0.125, so its logits are bf16-quantized.
- 18 positions are exact ties.
- A vs A2 matches on 7/8. `code` diverges at token 121, where A's margin is 0.25 and A2's is an exact tie.
- On the 1,341 shared-prefix positions, the chosen-token |Δlogprob| has p50 0, p99 0.00044 and max 0.12.
- Margin classes in A: confident (p₁ ≥ 0.9) 1,064, mid 341, near-tie (gap ≤ 0.25) 53, tie 18.

**Three measurements.**
1. **Self-replay:** two runs in one process plus one relaunch, `torch.equal` on every logit and token (decision 14). Pass = 100%.
2. **Teacher-forced vs Pg61-A:** at each of 1,476 positions, M0's fp32 top-1 vs Pg61's top-1. Flips are counted by Pg61's margin class, and ties get 1/k credit. M0's top-1 is also reported after rounding its logits to bf16. Pass = **0 confident flips overall**, the A-vs-A2 control having 0 inside its shared prefix. Mid and near-tie flips are reported, not gated. Top-20 KL is info only.
3. **Free-run vs A and A2:** first divergence per prompt. Pass = every divergence lands on a position whose TF margin (Pg61's or M0's) is ≤ 0.25. A byte-identity count is reported but not gated, because Pg61 itself manages only 7/8.

D1 decides how to read class 2 if it shows mid-margin drift.

**What this fixture cannot cover.** ≤310 tokens means ≤155 compressed entries at ratio 2. The indexer top-512 and the layer-20 candidate pruning (only above 16,384 positions) are therefore never exercised. A long-context gate needs separate tests:
- a 20–32K needle at several depths;
- self-replay `torch.equal`;
- chunk-invariance of greedy output across two chunk sizes, reported and not required bit-equal.

## 8. Risks and unknowns (ranked)

1. **Speed gate inside six weeks.** Eager reference decode means thousands of launches and 40 host syncs per token (model.py:896), dynamic shapes (model.py:579, 555; the `lru_cache` window indices are rebuilt every step, 409-426), GEMMs padded from M=1 to 32, and one-CTA attention. My estimate, not measured, is tens of ms/token. "≤ Pg61 + 10%" needs M0b's graph capture plus borrowed MoE, attention and GEMM kernels.
2. **D1, compressed KV and index K numerics.** The reference's forward quantizes them to FP4 (model.py:546, 553, 761; kernel.py:161, "Training's compressed KV"). Pg61 runs `fp8_ds_mla` (`ds41-args.json`). James ruled out `nvfp4_ds_mla` and the mxfp4 indexer as *cache* choices; here they are the model's own QDQ. Whether vLLM v0.31.0 applies the same QDQ before its FP8 store is unknown from the desk; reading its model source (T0) answers it. This affects parity even on the short fixture, because every compressed entry is attended.
3. **TileLang 0.1.12 on sm_103:** the `sparse_attn` smem/register load at 64 heads, FP4 codegen and FP8 `T.gemm` lowering. The self-test (`python model.py`, model.py:1298-1312) uses 16 heads and no Engram, so it does not cover the 64-head case. The T1 probe does.
4. **Host memory.** 251 GiB pinned at the 295 map, a swapless Grace and two prior OOM deaths. Trap: torch `pin_memory` power-of-two rounding. `cudaHostRegister` time on 94 GiB regions is unknown.
5. **Prompt rendering.** `encoding/` is not vendored. Garner's overlay files are what Pg61 rendered with.
6. **32K extend path is new code** with no reference ground truth, so chunk size becomes part of the numerics.
7. **torch ≥ 2.10** (requirements.txt:1), plus `float4_e2m1fn_x2`/`float8_e8m0fnu` and `F.embedding` on fp8 dtypes. The image's torch version is unverified.
8. **`norm_eps` 1e-20** (config.json:23). It is a no-op in fp32 for any nonzero row, but borrowed norm or mHC kernels that hardcode 1e-6 (or carry eps in fp16) change results on low-variance rows. It is separate from `hc_eps` 1e-6 and the gate's literal `+1e-20` (model.py:827).
9. **MP>1 assumptions.** None in the math. They sit in the 64-heads-in-one-CTA `sparse_attn` (risk 3), all-on-device construction (generate.py:130) and convert.py's whole-checkpoint RAM.
10. **bf16 indexer scores** (model.py:556-558) make top-k ties common. The skill already notes that determinism ends at more than 512 compressed entries. A borrowed top-k must reproduce `torch.topk` tie order or long-context parity drifts.
11. **Stack drift.** TileLang is not in decision 8's stack. It is kept as the reference's dependency for M0 only.

## 9. Work breakdown (Nov 9 → Dec 18; ~28 working days; ~26 planned)

| # | Task | Days | Depends | Done when |
|---|---|---|---|---|
| T0 | Desk: read vLLM v0.31.0 DSV4.1 compressed-KV/indexer write path (read-only); vendor checkpoint `encoding/`; render the 8 W0 prompts | 0.5 | — | D1 memo; token counts = 42/47/54/38/34/49/46/31 |
| T1 | Env on Station: torch ≥ 2.10 sm_103, tilelang 0.1.12; `python model.py`; 64-head d=512 `sparse_attn` probe; fp8/e8m0 `F.embedding` probe | 1 | — | all pass, or failing kernel named + fallback chosen; context MiB measured |
| T2 | Streaming loader (renames, wo_a dequant, fp4 view, placement, refuse-on-mismatch) | 3 | T1 | sha256 of placed bytes = shard bytes for 100% of tensors; host peak ≤ pinned + 8 GiB |
| T3 | Host tier: register + alias for Engram and cold pool | 1.5 | T1 | `torch.equal` alias vs HBM gather on 10⁶ ids; gather µs logged |
| T4 | Expert split M0a (staging + zero-copy arms), rowmap load | 2 | T2, T3 | layer-level `torch.equal` vs all-HBM experts on random inputs; cold-hit counter |
| T5 | Driver: greedy / TF / dumps / NVTX / profiler / determinism flags | 1.5 | — | runs on a 2-layer toy config |
| T6 | **First light:** 8 prompts, parity measurements 1–3 | 2 | T0, T2–T5 | report in `docs/`; self-replay 100% (target **Nov 24**) |
| T7 | Extend path + two-source `sparse_attn`; 32K | 4 | T6 | 32K prompt prefills and decodes 256 tokens without OOM; needles pass; self-replay equal |
| T8 | Graph-capturable step (device routing, fixed shapes, marker kernel) | 3 | T6 | whole decode step replays from one CUDA graph; parity unchanged |
| T9 | Borrowed kernels, one swap at a time (MoE → attention → MXFP8 GEMM → mHC) | 6 | T8 | each swap re-passes parity; ms/step logged per swap |
| T10 | Trace + gap histogram + M0 verdict | 1.5 | T9 | nsys step table with named boundaries; histogram; verdict vs D2 bar |

**Calendar.** Two days of slack. If T6 has not passed by Dec 1, Milo flags a projected miss at the Dec 4 week-4 check, as decision 17 requires. T7 can run in parallel with T8 if a second worker takes it.

**Decisions needed (none blocks T0–T6):**
- **D1** — compressed KV and index-K numerics: reference FP4 QDQ vs Pg61's `fp8_ds_mla`.
- **D2** — "1-row-equivalent". Proposed reading: M0's 1-row step ≤ 1.1 × 7.36 = 8.10 ms.
- **D3** — keep the 295/89 map through M1, or grow toward ~314 after MTP.

M0-PLAN: ready — the reference runs at MP=1 unchanged in its math; the plan's changes are placement, an extend path, and graph/kernel swaps, with three decisions for James (D1 numerics, D2 speed bar, D3 map) that do not block T0–T6.
