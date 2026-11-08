# M4-P1 — prefill anatomy and the fast-prefill design

2026-11-08. Worker: M4-P1 (Opus), main tree. This is a design and instrument round: no runtime change.
Handoff: `docs/HANDOFF-M4-P1-PREFILL-ANATOMY.md`.
Instrument: `bench/prefill_anatomy.py`, plus the prefill family map in `bench/step_anatomy.py` (`prefill_family_of`).
Tests: `tests/test_prefill_anatomy_cpu.py` (47).
Model receipt: `scratch/m4p1-model.{json,log}`, from `bench/prefill_anatomy.py model`.

Labels: [M] measured, [D] derived from a receipt, [E] modeled, [A] assumption. Every [E] below needs the box run in §3 before anyone builds on it.

## 0. Verdict

- **Dominant term, modeled [E]:** the reference `MoE.forward` loop runs one routed expert at a time (model.py:896-902).
  - A 42-token prefill makes about 4,400 expert calls (about 110 per layer).
  - Each call costs about 0.86 ms: roughly 54 eager launches issued from Python (~0.77 ms of host time), then a host sync in `torch.where(indices == i)`.
  - Underneath each call there is about 0.32 ms of GPU work (3 × the latency-bound 78 µs `fp4_gemm`, plus glue).
  - Total: about 3.8 s of the 3.99 s measured.
  - It is not C2C. The modeled cold traffic is 1.4 GB, which adds about 3 ms.
- **First build:** a layer-major prefill that keeps the reference's kernels. Only the MoE's per-expert loop is replaced, by a grouped variant of `kernel.py`'s own `fp4_gemm` text with row indirection, plus `fill_uninitialized_memory` off around the prefill. Modeled TTFT: **3.9 s → ~0.1 s [E]** at the fixture lengths, ~0.45 s at 4k, and ~3.2 s at 32k (beyond about 4.8k this needs a query-chunked indexer). About 2.5 days of work.
- **Bit-identical:** yes for that build, by construction (the same kernel text and the same per-row operations in the same order). It still has to be proven on the box with `torch.equal` per layer and the 8/8 stream gate.
  - Any path through our decode kernels is A3-class. That covers the chunked R-row verify step and a production-class DeepGEMM/FlashMLA prefill. Both change the KV, compressor and indexer state bytes that every later step reads.

## 1. What the prefill is, and what the receipts already say

The call chain is `DSparkLoop.start()` → `prefill_hidden()` → `reference_forward(model, tokens, 0)`. That is the reference `Transformer.forward` at start_pos 0, eager. The experts are the loader's pool views: hot experts in HBM, cold experts as UVA views of pinned Grace memory (`ExpertPool.bind`). Under `DSV41_POOL_SF=beside` the hot experts' K-major e8m0 scales are UVA views of Grace as well. After that come `seed_draft_rings` (forward_spec at start_pos 0) and the first draft (a graph replay, 2.9 ms [D]). The headline's `prefill_ms` covers all three.

The receipts say the prefill does not care where the bytes live [D]. Every `docs/headline*.json` from 2026-11-07 to 2026-11-08 has a mean prefill between 3,660 and 4,000 ms, through all of these changes:

- the hot map at 295, 301 and 304;
- the M3-S3 'beside' layout, which moved every hot expert's scales to Grace (3,845 → 3,856 ms);
- spare slots, residency, and PDL.

A GPU-bandwidth-bound or C2C-bound prefill would have moved. A host-bound one would not.

Per prompt [D] (`docs/headline-20261108-main-ss.json`, two reps each):

| prompt | story | shell | json | prose | long | code | tool | math |
|---|---|---|---|---|---|---|---|---|
| L | 31 | 34 | 38 | 42 | 46 | 47 | 49 | 54 |
| prefill ms | 3,507 | 3,496 | 3,675 | 3,988 | 3,931 | 4,063 | 4,169 | 4,142 |

The prefill runs with `fill_uninitialized_memory` ON (`set_determinism`, headline `determinism` block). The decode graphs are captured with fills off. The prefill is not.

## 2. Anatomy at 42 tokens — modeled [E], to be replaced by §3's measurement

### 2.1 Expert calls

The reference calls each routed expert once per layer, with all of its rows (model.py:897-902). So the number of expert calls is the number of distinct routed experts per layer.

The hard bound is L·k = 252 per layer, so "~all 384" is impossible at L = 42. To estimate the real number I used the independence model with the decode routing counts (`counts-post-20261024-020438.npz`, the counts the hot maps were fitted from) and the seed-301 map. It gives:

- **110.5 distinct experts per layer** (max 133), 4,420 calls in total;
- **1.9 cold per layer** (max 4.7), 75 cold calls in total;
- **1.41 GB over C2C.**

Uniform routing would give 186 per layer. The independence model is an upper bound, because a prompt's tokens route alike.

The handoff's 89 × 40 = 67 GB needs every cold expert to be touched. At 42 tokens about 2 per layer are. A ~0.26 s C2C term does not exist at the fixture lengths. It does appear at 4k: about 59 cold per layer, 44 GB.

The reference reads each touched expert once per 32-row M-tile (`fp4_gemm` block_M = 32), so once per layer at L ≤ ~200. At 4k an expert's ~68 rows mean three reads of the same weights, cold ones included.

### 2.2 The terms

Model inputs (all printed by `model`):

- From the 1-row eager M0 anatomy [D]: 20,550 launches per step; 404.9 ms with fills (357.1 ms without); 108.6 ms GPU-busy; MoE 14,480 launches for 240 expert calls and 84.4 ms GPU; `fp4_gemm` 78.1 µs per launch.
- From the C2C probe [D]: +11.5 µs per cold `fp4_gemm`.
- From me [A]: MoE fixed launches per layer (gate + shared expert + bookkeeping ≈ 40).

| term | modeled | how |
|---|---|---|
| MoE per-expert loop | **3.81 s (97 %)** | 4,420 calls × 0.862 ms. Per call: 53.7 launches × 14.4 µs host = 774 µs, plus the GPU tail (~88 µs) waited for at the next `where()` sync. GPU work per call is 321 µs (3 × 78 µs `fp4_gemm` at 18-40 CTAs + ~86 µs glue), so the loop is host-bound. |
| of which `fill_uninitialized_memory` | ~0.57 s | M0: fills-off host = 0.839 × fills-on host [D] |
| host syncs | ~4,460 | 40 `bincount().tolist()` + one `nonzero` per expert call (`torch.where`) |
| MoE fixed work (gate, shared) | 23 ms | 40 layers × 40 launches |
| everything else, eager (attention, indexer, compressor, mHC, engram, norms, head) | 88 ms | 152 launches per layer, host-bound (24 ms of GPU work at 1 row) |
| cold experts over C2C | +2.6 ms (1.4 GB) | 75 calls × 3 × 11.5 µs, inside latency-bound launches |
| seed_draft_rings + first draft | ~3.4 ms | forward_spec start_pos 0 (~0.5 ms [E]) + draft graph 2.9 ms [D] |
| **total** | **3.92 s** (measured 3.99) | |

Modeled vs measured across the fixture: 3.30 vs 3.51 s at L = 31, and 4.49 vs 4.14 s at L = 54. The independence model makes the call count grow too steeply with L, as expected for correlated tokens. That is why the census measurement in §3 replaces it.

GPU-busy would be about 1.6 s of the 3.9 s. Of that, ~1.0 s is `fp4_gemm` running one expert per launch at 18 CTAs (w1/w3) or 40 CTAs (w2), latency-bound at ~64 GB/s effective (C2C probe). **Even a zero-host-cost version of today's loop cannot go below ~1 s.** The per-expert launch itself has to go.

## 3. The instrument and the box commands

`bench/prefill_anatomy.py` has four subcommands.

**`run`** (box) boots like the headline (map seed-301, `load_mtp`, `set_determinism(0)`) and warms once. Then, per prompt:

- `--reps` untraced prefills with fills on (the headline state), timing `prefill_hidden` and `seed_draft_rings`, and recording the first token and the logits and main_hidden sha256.
- `--fills both`: the same reps with fills off, plus a bit-equality check against fills on.
- One hooked prefill (`PrefillProbe`, forward hooks only, removed after). It records host ms and GPU-timeline ms (CUDA events) per role: layer, engram, attn, compressor, indexer, moe (inclusive and exclusive), gate, shared, `expert.hot`, `expert.cold`, head.
  - It also takes the **census**: calls per layer = distinct experts; rows = L·k; hot/cold split by pool tier; cold ids; C2C bytes, including the hot scales under 'beside'.
  - `--profile` wraps this one prefill in `cudaProfilerStart/Stop` with NVTX ranges `prefill`, `seed_draft`, `L{i}`, and the roles above.
- `--probe-moe`: the captured MoE inputs of the listed layers are re-run through the reference `MoE.forward` and through `sliced_moe_forward`. The sliced version makes the same reference `Expert` calls with one sync per layer instead of one per expert. The probe reports `torch.equal` for each combine spelling and the synced wall ms, with fills on and off. This is the stage-B measurement.
- `--rows-probe 42,64`: `runtime/row_census.census(model, rows=R)`, i.e. is row r at R rows bit-equal to the same row alone, per kernel family. This is the decode path's 1-vs-6 invariance question, asked at 42 and 64.

**`analyze`** reads the nsys sqlite (stdlib only). For the prefill and seed_draft phases it reports:

- wall, GPU-busy (union of kernel intervals), sum of kernel time, and idle;
- kernels per **prefill family**, attributed to the innermost NVTX role range of each kernel's launch;
- per role: host wall, launches and GPU ms;
- per expert call, hot vs cold: wall, launches, GPU µs, and `fp4_gemm` µs per launch (cold − hot = the in-kernel UVA cost);
- expert ranges per layer (the census, from nsys alone);
- the runtime-API census: launch calls, sync calls, and host ms blocked in syncs;
- memcpy bytes by kind.

**`model`** (Mac) prints the table in §6 from the receipts and the counts. **`box`** prints the lines below.

Box lines (orchestrator; `python3 bench/prefill_anatomy.py box` prints them):

```
W=$HOME/dsv41/decode; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
docker ps --format '{{.Names}}' | grep -q 'ds41\|mk-' && echo refusing && exit 1
mkdir -p $W/scratch/prof
D="docker run --rm --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v $HOME/dsv41/w0/jit-cache/tilelang:/root/.tilelang -v $HOME/dsv41/w0/jit-cache/triton:/root/.triton -v $HOME/dsv41/w0/jit-cache/nv:/root/.nv -v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json"
# 1) timing, fills off + equality, census, stage-B MoE probe, rows census at 42/64 (~5-8 min incl. 63 s boot)
$D --name mk-prefill --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/prefill_anatomy.py run --model-dir /ref --only all --reps 3 --fills both --probe-moe 0,1,2,14,20,39 --rows-probe 42,64 --json /work/scratch/prefill-anatomy-all.json > $W/scratch/prefill-anatomy-all.log 2>&1
# 2) nsys: ONE hooked prefill of the 42-token prompt (traced = slower; take ms from run 1)
$D --name mk-prefill-nsys --cap-add SYS_ADMIN -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro --entrypoint /nsys/target-linux-sbsa-armv8/nsys vllm/vllm-openai:v0.31.0 profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o /work/scratch/prof/prefill-prose python3 bench/prefill_anatomy.py run --model-dir /ref --only prose --reps 0 --fills on --profile --json /work/scratch/prefill-anatomy-prose-nsys.json > $W/scratch/prefill-anatomy-prose-nsys.log 2>&1
/usr/local/bin/nsys export --type sqlite -o $W/scratch/prof/prefill-prose.sqlite $W/scratch/prof/prefill-prose.nsys-rep
python3 $W/bench/prefill_anatomy.py analyze $W/scratch/prof/prefill-prose.sqlite --out $W/scratch/prefill-anatomy-prose-nsys-analysis.json > $W/scratch/prefill-anatomy-prose-analysis.log
```

What the run decides:

- **Is the loop host-bound?** Compare (a) the per-expert wall in the census against the 0.86 ms model and (b) GPU-busy against the 1.6 s model.
- **Do fills off help, and are they bit-equal?** `fills_off_bit_equal` must be true. ~0.5 s [E].
- **Is stage B bit-equal, and what does it gain?** `moe_probe.equal_*` and the reference vs sliced ms.
- **Which kernel families are row-invariant at 42 and 64 rows?** This settles whether a token-chunked prefill could ever be bit-identical.

## 4. Design: a fast prefill on our decode machinery — what it would take

### 4.1 What the R-row static step supports today

The verify step runs at R = 6. The code allows R ≤ 8:

- `static_step.MAX_ROWS = 8`, rows even. The GEMV is M ≤ 8; `DSV41_GEMV_PIN_M1` pins the 1-row config for row-invariance.
- `moe_rows.DEDUP_MAX_PAIRS = 64` (R·k ≤ 64) for the one-warp dedup.
- Spare slots: 16 per layer, by compact cold index. An overflow gives wrong outputs, counted on the device.
- DeepGEMM contiguous layout at ALIGN 16 (`contiguous16`, so R ≤ 16).
- The 256-slot ring is safe for R ≤ 128.

### 4.2 What R = 32 or 64 would need

- **Dense projections:** a new M-tiled fp8 GEMM, because the GEMV stops at M = 8. That would be DeepGEMM fp8 or the reference `fp8_gemm`, and neither matches the GEMV's bytes.
- **Dedup** over 192-384 pairs: the matrix kernel, or a sort.
- **Cold slots.** My [E] numbers are smaller than the handoff's. With the measured routing a 64-token chunk touches ~3-5 cold experts per layer (max ~13 at 128 tokens), so 16 spare slots are enough. The "up to 89 per layer, 1.7 GB per layer" case is long-chunk only: about 59 per layer at 4k.
  - Per-layer spare slots cannot grow to 64. The cost would be 40 × 64 × 18.8 MB = 48 GB of HBM.
  - Long chunks need a shared staging region instead: one layer's cold set, double-buffered, ≤ 2 × 1.6 GB. The other option is VMM-aliasing one physical spare region behind every layer's pool tensor. That is a pool redesign and would also benefit decode.
- **DeepGEMM** at ALIGN ≥ the rows per expert (psum layout, or ALIGN 64). Masked needs max_m ≥ 128.

### 4.3 Chunked attention

FlashMLA's `sparse_prefill_fwd` **is** the prefill kernel. It is already loaded from the image's `.so` and used for decode at 6 rows (`runtime/attn_flashmla.py`).

- It takes any number of query rows, each with its own index list. A chunk's causal attention over [ring | chunk] or the full prompt KV is just index lists. No varlen kernel is needed.
- The reference path is `sparse_attn` (TileLang, one CTA per query) with `get_window_topk_idxs(start_pos=0)` causal windows plus the indexer's top-k.
- The reference cannot run a chunk at start_pos > 0 with seqlen > 1 as written (model.py:719, 476-482, 542, 568). `runtime/ref_multirow.py` patches exactly those lines for its 6-row oracle.

### 4.4 Why token-chunk-major is the wrong shape for long prompts

Every chunk re-streams the touched experts' weights. At 64-token chunks that is ~138 experts per layer per chunk (5,535 calls [E]; 243 per layer under uniform routing), all 40 layers, cold ones over C2C, 512 times for 32k. Modeled [E]:

- the 6-row verify step as a prefill: 32k ≈ 5,461 × 15.1 ms ≈ 82 s;
- 64-row chunks: ≥ ~10 s on weight streaming alone (≥ 20 ms per chunk: ~102 GB of HBM reads + 2.1 GB over C2C), before any compute.

A prefill has to be **layer-major**: all prompt tokens through layer l before layer l+1, sub-chunked only where activation memory demands it. That way each layer's experts, and its cold bytes, are read once per prompt. The reference forward is already layer-major. The only quadratic activation is the indexer score, [L, 32, L/r]. At the headline's 2.7 GiB free HBM a single shot caps near L ≈ 4.8k [E]. M0-PORT-PLAN's ~7k assumed ~6 GiB free.

## 5. The bit-identity question

The gate is that the stream after the prefill is byte-identical to today's. Today's stream is defined as the reference prefill numerics followed by our decode kernels. Here is everything the prefill writes that a later step reads:

| state the decode reads | producer in today's prefill | read by | C (grouped ref-kernel prefill) | our decode kernels / production-class (B-chunked, E) |
|---|---|---|---|---|
| first token (argmax of the last row's logits) | head: fp32 F.linear, cuBLAS, **M = 1** (model.py:1012-1014), on `norm(hc_pre(h))` | the stream | same ops → yes | fixed-order fp32 head + upstream differences → A3 |
| prefill logits bytes | same | the parity record only | yes | A3 |
| window KV, last ≤ 128 positions × 40 layers (bf16 after QDQ) | `wkv` fp8 Linear (TileLang `fp8_gemm` at M = L) → `kv_norm` → RoPE (torch complex) → `act_quant` QDQ (TileLang), model.py:706-708; copied into the step's 256-ring by `sync_ring` | every later step | yes (same ops; the residual stream is equal per row) | GEMV `wkv` (not bit-identical by design, the 32-wide block-sum order) + fused kv_path → A3 |
| compressed KV, all groups (layers 2/8/14 at ratio 2, layer 20 at ratio 1) | ratio 2: fp32 `wkv`/`wgate` F.linear (cuBLAS, M = L) → **prefill-spelling** pooling (model.py:473-475) → norm → RoPE → fp4 QDQ (e4m3 scales); ratio 1: bf16 F.linear → norm → RoPE → QDQ | every compressed layer, every step | yes | decode uses the T9b-8 fixed-order fp32 compressor and the **decode-spelling** pooling (482), which can differ from the prefill spelling by an fp32 ulp (static_step.py `pool_ratio2_rows`) → A3 |
| indexer K cache (owners 2/8/14/20) | `wk` bf16 F.linear → `k_norm` → RoPE → fp4 QDQ (model.py:544-547) | indexer layers, every step | yes | A3 |
| ratio-2 compressor `kv_state` / `score_state` (the odd tail) | fp32 F.linear rows at M = L (model.py:471-472) | the next decode pooling | yes | A3 (fixed-order fp32) |
| Engram hash cache | token ids (engram.py:167) | every step | exact in every design | exact |
| draft rings × 3 (128 slots) | forward_spec start_pos 0 from main_hidden (the hc means at layers 37-39) | drafts only → **accept counts, never emitted values** (design 1.4) | yes | changes tok/step only |
| hc / mHC state | none persists across positions | — | — | — |

### 5.1 Does row-invariance extend to 42 rows?

The decode path is row-invariant at 1 vs 6 rows (T9b-8: streams 8/8). It cannot simply be asked at 42, because the GEMV stops at 8 rows.

For the reference kernels the M1-T10 census at 6 rows already found every family row-invariant except the **cuBLAS fp32 head** and tuned-M GEMV. By construction, the following are expected to stay invariant at 42: `act_quant`, `fp8_gemm`, `fp4_gemm` (each output element is the same K-ordered sum regardless of M-tile placement), `sparse_attn` (one CTA per query), `hc_split_sinkhorn` (one CTA per token), and RMSNorm.

The M-dependent cuBLAS calls are unknown at 42 and 64: hc_mixes fp32, the gate fp32, compressor fp32/bf16, indexer `wk` / `weights_proj` bf16, and the einsums. `--rows-probe 42,64` measures them.

- If any of them is not row-invariant, a **token-chunked** prefill (any chunk ≠ L) cannot be bit-identical even on the reference kernels.
- A **layer-major, whole-prompt** prefill (design C) keeps every one of those calls at M = L and sidesteps the question.

### 5.2 Verdict

- Any fast prefill that runs our decode kernels (GEMV, DeepGEMM, FlashMLA, fixed fp32, decode-spelling pooling) or production's kernels is **A3-class**. It changes the KV, compressor and indexer bytes for every later step. James has not approved A3 for this.
- A prefill that keeps the reference kernels and only removes the per-expert host loop is **bit-identical by construction**, provided the grouped GEMM is the reference kernel text. The precedent is `runtime/kernels/sparse_attn_bounded.py` (T9b-3a), which modified the reference TileLang text and was proven bit-identical.

## 6. Alternatives, ranked

TTFT in ms, modeled [E]. Rows sorted by TTFT at the fixture lengths. Source: `scratch/m4p1-model.log`.

| alternative | TTFT @ 42 | @ 4,096 | @ 32,768 | days | numerics |
|---|---|---|---|---|---|
| E. dedicated production-class prefill (DeepGEMM m-grouped fp8×fp4, DeepGEMM fp8 dense, FlashMLA `sparse_prefill_fwd`, DeepGEMM MQA-logits indexer, cold prefetch on the copy engine one layer ahead, layer-major, graph-able) | **23** (Pg61 measures ~27/prompt [D]) | 158 | **576** | 6-7 | A3 |
| **C. layer-major reference-kernel prefill, grouped `fp4_gemm` text + row indirection, cold set staged per layer, fills off** | **92** | **453** | **3,209** (query-chunked indexer needed past ~4.8k) | **2.5** | **bit-identical** (box `torch.equal`) |
| D. chunked prefill through today's 6-row verify step (P0 = 0 support + driver) | 119 | 10,317 | 82,417 | 2 | A3 |
| B. reference loop with one sync per layer (stage B, `sliced_moe_forward`) + fills off | 2,524 | 7,996 | OOM | 0.5 | bit-identical |
| fills off only (scope `fill_uninitialized_memory=False` around `start()`) | 3,354 | 10,700 | OOM | 0.1 | bit-identical (T8 L2: fills on == off) |
| A. reference forward with cold experts pre-staged in bulk | 3,922 (saves 2.6 ms) | 12,432 | OOM | 1 | bit-identical |
| today | 3,924 (meas. 3,988) | 12,514 | **OOM** (single-shot indexer > 4.8k) | — | — |

### Why A loses

The modeled cold set at 42 tokens is 1.4 GB, read inside latency-bound launches that hide the extra 11.5 µs. Pre-staging attacks a term worth 0.07 % of the TTFT. A bulk pass over *all* cold experts (62 GB at 390 GB/s ≈ 0.16 s) would be slower than what it replaces.

### Why B is not enough

The Python dispatch per expert is the dominant term. B removes one sync and ~15 % of the ops, and the GPU's one-expert-per-launch `fp4_gemm` latency (~1 s at L = 42) stays underneath.

### Long context (32k in seconds)

- Only layer-major designs with grouped expert GEMMs get there: C at ~3 s, E at ~0.6 s.
- Today's reference prefill, A and B OOM in the indexer score beyond ~4.8k at the current HBM state. They would also cost ~13 s of host loop.
- D cannot get there.

### 32k items common to C and E

- the KV and state buffers: the loader overlay already sets `max_seq_len` 32768;
- the static step's `max_pos` (4095 today) and its compressed-cache width W_r. That is a decode-cost question for long contexts, separate from TTFT;
- a query-chunked indexer: the score in query chunks, then top-k per row. It equals the single shot when the cuBLAS bmm is row-invariant, and §5.1's probe answers that. Beyond ~4.8k there is no today's stream to match.

## 7. What I would build first, and why

**Build C**, in two stages behind one flag. It would be a patch to `MoE.forward` for start_pos 0 and multi-row input only, in `runtime/patches.py`, citing model.py:891-906. Default off.

**Stage 0 (hours): fills off for the prefill.** Scope `fill_uninitialized_memory=False` around the reference prefill. It is bit-identical by the T8 L2 parity row. Modeled saving ~0.57 s; `run --fills both` measures it and checks equality first.

**Stage 1 (0.5 day): one sync per layer.** This is the `sliced_moe_forward` structure:

- one stable sort and one `bincount().tolist()` per layer;
- per expert, contiguous row slices with no `where()`;
- `index_add_` combine, which is exact because the rows are unique within an expert.

It is bit-identical, and the box probe proves that on real activations before it lands. Modeled ~2.5 s. It is the scaffold for stage 2: the same sort, slices and combine order.

**Stage 2 (2 days): one grouped launch per projection per layer.** A grouped TileLang kernel made from `kernel.py:478-559`'s text. The only change is a tile → (expert slot, row base, row count) indirection, masked stores past the count, and B / scales taken from the layer's pool tensors (hot) or a per-layer cold staging buffer:

- the cold set is copied on the copy engine right after the gate;
- ~2 experts per layer at the fixture lengths;
- ≤ 1.6 GB double-buffered at 4k+.

The surrounding steps keep the reference's arithmetic:

- `act_quant` once over the gathered rows (row-invariant, same bytes as per expert);
- the reference SwiGLU, clamp and weights ops on all pair rows (elementwise);
- the per-row ascending-id fp32 combine: `0 + o_a + o_b + …` in the reference's order, as `moe_rows.posmat` already does for decode;
- the shared expert unchanged.

Host work becomes ~180 launches per layer for the whole forward. Modeled TTFT is **~0.09-0.13 s** at the fixture lengths, host-bound: 40 × 182 launches. A next step could remove the per-layer count sync with a device tile schedule (the bounded worst case is ≤ min(E, L·k) tiles) and graph the forward per length.

### Gates (box)

- per layer, `torch.equal` of the patched vs the reference MoE output on real prefill activations;
- prefill logits sha256 equal;
- window, compressed, indexer and state caches byte-equal;
- headline streams 8/8 byte-identical, with tokens and accept lengths equal to the current receipt;
- self-replay 8/8.

### Why C first rather than E

- It keeps today's stream byte-identical. E needs James's A3 approval, and he has not given it.
- It removes 97 % of the modeled TTFT at the lengths we serve today. E's extra gain at 42 tokens is ~70 ms.
- Its skeleton is E's skeleton: layer-major, grouped experts, a per-layer cold stage, a device-side tile schedule, and a query-chunked indexer for > 4.8k. If James later approves A3, E is a kernel swap behind flags, not a rewrite.

### Risks

- The grouped TileLang kernel is authored blind on the Mac and compiled on the box. The `sparse_attn_bounded` precedent says that is doable, but it costs one box window.
- The 2 TB/s weight-stream assumption for that kernel [A] is unmeasured. Even at 0.5 TB/s, C at 42 tokens is ~0.2 s (GPU-bound: 83 GB of touched hot weights).
- The independence model over-states calls at larger L. The census replaces it.

## 8. Open, decided by the box run in §3

1. The measured per-expert wall and GPU-busy (host-bound vs GPU-bound split), the launch and sync census, and cold vs hot `fp4_gemm` µs.
2. Whether `fills_off_bit_equal` holds, and the saving.
3. Whether stage B is bit-equal on real activations, and what it gains.
4. Row-invariance at 42 and 64 per family: whether a token-chunked prefill can ever be bit-identical.
5. The census: distinct experts and cold experts per layer per prompt. These replace §2's independence numbers, and the model table is re-run from them.

## 7.1 M4-P2 result (2026-11-08): what was built — Mac-verified, box pending

Worker: M4-P2 (Opus), worktree `gb300-dsv41-decode-wtp2`, branch `m4-p2`. Handoff `docs/HANDOFF-M4-P2-FAST-PREFILL.md`. Code: `runtime/prefill_fast.py` (new), `runtime/draft_step.py` `prefill_hidden` (call site), `runtime/dspark_loop.py` `start` (records `last_prefill`), `bench/prefill_bench.py` (new), `tests/gpu/test_prefill_fast_gpu.py`, `tests/test_prefill_fast_cpu.py`. Switches: `$DSV41_PREFILL=reference|grouped` (default **reference** until the box gate), `$DSV41_PREFILL_FILLS=off|on` (default **off**, now for both modes: box 8/8 `off==on True`, −0.55 s), `$DSV41_PREFILL_STREAMS` (default 4).

**No grouped entry exists in the reference.** `kernel.py` has one `fp4_gemm` (one expert, one launch) and no masked/grouped variant; DeepGEMM's masked grouped GEMM is row-invariant but is not the reference kernel text (A3). So this build is the handoff's "emulate the grouping" branch, not §7 stage 2's new TileLang kernel:

- per MoE layer: the reference `Gate`; one stable sort + ONE `bincount().tolist()` (the reference's own sync, 896; the per-expert `torch.where` syncs are gone); each routed expert gets a segment of a layer-wide buffer starting on a multiple of 32 rows and padded to 32; ONE `index_select` gathers x into it; ONE `act_quant` (linear()'s arguments) for w1/w3;
- per expert, ONLY the GEMM launches: the reference TileLang `fp4_gemm_kernel` object — the very object `kernel.fp4_gemm` builds, same arguments — called on views: A = the segment's c rows (M = c, the reference's M: same grid, same swizzle), B/scales = the expert's own pool views (hot HBM, cold UVA, unchanged), C = a view into the layer buffer. 3 launches per expert instead of 26;
- the reference SwiGLU ops in the reference order on all rows at once; one `act_quant`; the w2 launches; the combine as k fp32 adds of `out[pos[:, j]]` with pos = each row's segment positions ascending (= ascending expert id), i.e. the reference's `0 + o_e1 + o_e2 + …`; shared expert and `type_as` unchanged.
- the experts' launches can be spread over N CUDA streams (each writes only its own segment; the current stream waits before any read). Order cannot change a value.

Why the 32-row segments matter beyond tidiness: a segment's views start on a 64-byte boundary (TVM's assumed buffer alignment; asserted per launch) — with DSV4.1's inter 2304 the w2 activation scale row is 72 B, so packed rows would give 8-byte-aligned scale views; an unguarded tail-tile store stays inside the expert's own padding (concurrent launches never touch a neighbour); act_quant's 32-row tiles never straddle experts. Pad rows copy token row 0 with weight 0 and are never combined.

Everything else of the forward (attention, compressor, indexer, all KV caches, mHC, Engram, head) is the reference forward untouched: the scope installs the grouped function as the INSTANCE `forward` of each `model.layers[i].ffn` for one prefill and removes it on exit (the class-level static-MoE patch is not touched).

**Launches per prefill** (census from the M4-P1 box receipt [D]; after = this build, counted):

| prompt | L | expert calls | expert launches before (26/call) [D] | GEMM launches after (3/call) | + MoE glue after (40 × 42 ops) |
|---|---|---|---|---|---|
| story | 31 | 3,013 | 78,338 | 9,039 | ~1,680 |
| prose | 42 | 3,672 | 95,472 | 11,016 | ~1,680 |
| math | 54 | 3,652 | 94,952 | 10,956 | ~1,680 |

MoE host syncs per prefill: ~3,700 (a bincount per layer + a `where` per expert) → 40 (the bincounts). The kineto count in `prefill_bench run --kineto` replaces the glue estimate.

**Modeled TTFT [E]** (`python3 bench/prefill_bench.py model`; parameters printed with labels; the 20 µs host per direct TileLang launch is [A]): mean over the 8 prompts today 3,703 ms [D] → fills off 3,174 [D] → grouped 1 stream **~1,030** (GPU-bound: 3 × 3.5k latency-bound `fp4_gemm` × 79–88 µs [D] in series) → grouped 4–8 streams **~410** (host-bound: ~11k launches × 20 µs [A] + ~200 ms of unchanged non-MoE / gate / shared work). That is a 9× TTFT cut, not §0's ~0.1 s: the per-expert launch stays. The ~0.1 s needs §7 stage 2 (one grouped TileLang launch per projection per layer, the reference text with a tile → (expert, row base, count) indirection); this build is its scaffold (same routing, segments, gather, SwiGLU and combine) and its bit-identity oracle.

**Mac verification** (CPU kernel emulations, both sides of every comparison): `tests/test_prefill_fast_cpu.py` 16 passed — per-layer MoE `torch.equal` (mini4 top-4 at L = 1, 23, 37, 77; mini3), the direct-kernel path through a fake `kernel` module whose kernel also scribbles NaN over every tile's unguarded tail rows, prefill ids/logits/main_hidden and every layer buffer equal under reference vs grouped (also past the 128 window, L = 130), self-replay, the counters against the PrefillProbe census, the DSpark loop's tokens / accept lengths / per-position logits sha after a grouped vs a reference prefill, scope install/uninstall, the bench `run` end to end. Mutation check: reversing the combine order makes the per-layer test fail (the order is observed). Full suite `.venv/bin/python -m pytest tests/ -q --ignore=tests/gpu`: 1399 passed, 1 skipped (`scratch/m4p2-cpu-tests.log`).

## 8.1 M4-P2: open, decided by the box (`python3 bench/prefill_bench.py box`)

1. `tests/gpu/test_prefill_fast_gpu.py` on the mini with the real kernels: per-layer MoE equal at 1 and 4 streams, prefill + every buffer equal, loop streams equal in static and graph mode.
2. `prefill_bench.py run --streams 1,4,8 --kineto` on the real checkpoint, 8 prompts: first / logits / main_hidden / every layer buffer + draft rings equal to today's reference (fills on), replay, prefill and TTFT ms [M], kernels and `fp4_gemm` launches, GPU busy %, C2C bytes (cold weights are still read in-kernel over UVA, once per expert per M-tile, as today).
3. `headline.py` with `DSV41_PREFILL=grouped --residency-ref docs/headline-20261108-main-ss.json`: the 256-token streams and accept lengths per prompt equal to today's receipt (`all_tokens_equal`, `all_accepts_equal`), and the headline's prefill_ms.
4. Whether TileLang launches on the current stream (multi-stream speedup) and the real host µs per direct launch (the [A] above): both read off (2).
5. Only if 1–3 pass: flip `DEFAULT_MODE` to "grouped". Then the next build is §7 stage 2 (the grouped TileLang kernel, ~0.1 s [E]); the cold pre-stage (handoff second stage) is not built — at the fixture lengths the cold calls are 4–9 % of the GEMM launches and run concurrently with hot ones, so it is measured from (2)'s cold/hot split before anyone builds it.
