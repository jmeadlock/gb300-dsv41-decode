# T8 + T9a design: a graph-capturable step and the routed-MoE swap

Desk memo, 2026-11-06, written from source with no GPU, ssh, docker or vllm import. Inputs are the files named in `docs/HANDOFF-T8-T9A-DESIGN.md`. Two things landed during this run and are used to check the source prediction: the step profile (`docs/STEP-ANATOMY-M0-20261106.md`, `docs/step-anatomy-m0-20261106.json`) and the static-MoE spike (`bench/moe_static_spike.py`, `docs/moe-static-spike-20261106.json`). "est." means an estimate. Every other number cites its receipt.

## 0. Corrections to the handoff

1. **The MoE runs 6 routed experts plus 1 shared, not 7 routed.** `n_activated_experts=6` (`m0-forward-graph.json` config). The profile counts exactly 720 `fp4_gemm` = 6×3×40.
2. **Cold experts per layer at 1 row are 1.42, not 1.7.** First light saw 57 cold of 240 routed calls per step (`FIRST-LIGHT-20261106.md`), so 57/40 = 1.42.
3. **"~1,300 launches" is Pg61 v27's count.** The source predicts about 15k for our step (table 1), and the profile measured 20,550.
4. **There are 280 host syncs per step, not 40** (the plan's summary said 40). Per layer there is one `bincount().tolist()` (model.py:896) plus six `torch.where(indices == i)`, and each of those is a `nonzero` (model.py:901). The profile shows 240 `DeviceSelectSweepKernel` and 40 `kernelHistogram1D`.
5. **MegaMoE does round gate and up to bf16 before SwiGLU** (`sm100_fp8_fp4_mega_moe.cuh:1058-1059`) and clamps in bf16 (1062-1066), exactly like the reference (kernel.py:583; model.py:845-849). The bit-identity break is elsewhere; see §3.
6. **The masked grouped GEMM cannot run over the hot pool.** It offsets B by *group* index (`scheduler/gemm.cuh:175-177`), so the selected experts must first be gathered into `[G,N,K/2]`. The contiguous layout offsets B by the expert id read from `m_indices` (172-174) and skips `-1` blocks (327-331). Only contiguous reads `[H,N,K/2]` in place.
7. **vLLM's DSV4 MoE class is not vendored.** `DeepseekV4MoEBase`, its `finalize_mega_moe_weights`, the non-mega path and the x/x_sf preparation all live in `vllm/models/deepseek_v4/nvidia/model.py`. That file is imported at `deepseek_v41/nvidia/model.py:77-84` but is not in our tree. What vLLM does to the HF layout beyond `transform_weights_for_mega_moe` therefore cannot be read here. Also, vLLM's non-mega `deep_gemm` backend is "FP8 block-quantized only" (`config_kernel.py:248`); MXFP4 experts go through mega or FlashInfer.

## 1. Where the 405 ms goes, predicted from source vs measured

Here is the prediction from source. The launch count walks model.py op by op, with each torch op counted as one kernel and each `empty`/`new_empty` in the kernel.py wrappers as one NaN fill. GPU time uses the measured 91.3 µs for a w1 `fp4_gemm` at M=1 (`C2C-PROBE-20261106.md`). w2 is scaled by its K-loop length: 72 vs 160 iterations of 32, about 41 µs (est.).

**Table 1: 1 row, per step**

| role (×40 layers unless noted) | source: launches | source: GPU ms | measured launches | measured GPU ms |
|---|---|---|---|---|
| routed experts, 6 × {w1,w3,w2} `fp4_gemm` | 720 + 720 `act_quant` | 53.5 (est.: 6 × (91+91+41) µs) | 720 | **56.3** (78 µs avg) |
| shared expert, 3 `fp8_gemm` | 120 (+120 quant) | ~9 (est.) | fp8 total 290 (= 7×40 + 8 indexer `wq_b` + 2 Engram `wkv`, as predicted) | 15.9 for all 290 |
| attention GEMMs (`wq_a`, `wq_b`, `wkv`, `wo_b`; + indexer `wq_b` on 8 layers) | 168 | ~15 (est., same per-K-step latency as `fp4_gemm`) | (in the 290) | (in the 15.9: fp8 is faster per K step than the model assumed) |
| `act_quant` / `fp4_act_quant` | 1,090 / 13 | ~3 | **1,090 / 13** (exact) | 2.8 / 0.02 |
| `sparse_attn`, 1 CTA | 40 | 1.5–2.5 | 40 | 3.0 |
| `hc_split_sinkhorn` | 80 | 0.4 | 80 | 0.35 |
| MoE glue (gate, bincount, `where`, gather, clamp/silu/mul, scatter-add) | ~150/layer ≈ 6,000 | 6–10 | moe role 14,480 (GEMMs included) | moe role 84.4 |
| norms, mHC mixes, RoPE, cat, indexer glue | ~3,500 | ~5 | attn 3,053 + other 2,890 | attn 17.6, other 5.9 |
| Engram (L1, L14) + hash | ~110 | 0.5 | 124 | 0.38 |
| head, fp32 sgemv 129,280×5,120 | 1 | ~0.45 (2.65 GB at ~6 TB/s, est.) | 1 (`gemv2T`) | 0.42 |
| NaN fills from deterministic mode | ≥3,500 explicit empties | ~3.5 | inside 11,570 `vectorized_elementwise` | wall Δ **48 ms** (405 vs 357) |
| host syncs | 280 | — | 240 + 40 | — |
| **total** | **~15,000** | **~100** | **20,550** | **108.6 busy** |

The source model gets the GEMM structure exactly right (720, 290, 1,090, 40, 80, 13). It undercounts launches by ~27%: torch-internal allocations, the deterministic `index_put_` sort path, and `_assert_async` ×486 are not visible from Python. GPU time comes out within 10%.

**The two overheads, and the numbers that decide the order:**

- **(a) Python/launch gap.** The handoff guessed ~1,300 launches at 20–40 µs each. The measured step is about 300 ms of untraced idle over 20,550 launches, ≈ 14.6 µs per launch (derived), i.e. 73% of the step.
  - Decision rule: host idle above 50% of the step means T8 first; below it means T9a first.
  - **Measured 73%: T8 first.**
- **(b) `use_deterministic_algorithms` + `fill_uninitialized_memory`.** Measured 48 ms per step, 12% (405.0 vs 357.1, same tokens).
  - Decision rule: if this exceeds 10%, timed runs use fills off, and one fills-on replay is kept as an uninitialized-read detector.
  - Measured 12%, so that rule applies.

Fills recorded into a graph would replay as kernels. T8 therefore captures with fills off.

## 2. T8: one CUDA graph per decode step

### 2.1 Every data-dependent branch in the decode path, and its fixed-shape replacement

| # | where | what depends on data / `start_pos` | replacement (same arithmetic) | parity argument |
|---|---|---|---|---|
| 1 | runtime generate.py:363-366, 414 | H2D token tensor each step; D2H logits each step | static device `tok[1,1]`, `pos[1]`; the graph ends with `tok.copy_(argmax)` and `pos += 1`. Logits D2H only in parity mode, after replay | no math |
| 2 | engram.py:167, 169 | `cache[:, start_pos:...]` write; `arange(start_pos, …)` | `index_copy_` at device `pos`; `positions = pos` | integer ops: exact |
| 3 | model.py:768, 552 | `freqs_cis[start_pos:end_pos]` slice | `index_select(0, pos)` once per step per table (base and compressed rope) | value copy: exact |
| 4 | model.py:709 | `start_pos == 0` prefill branch | decode-only graph; prefill stays eager | — |
| 5 | model.py:719 | ring write at `start_pos % 128` | `index_copy_(1, pos % 128, kv)` | exact |
| 6 | model.py:721, 409-426 | window indices built on the host (lru_cache, keyed by `start_pos`) | device `idx = (pos%128 + 1 + arange(128)) % 128`, then `where(idx > pos, -1)` | **same order**. Order is irrelevant mathematically (model.py:415) but not for bits: `sparse_attn`'s online softmax accumulates in index order |
| 7 | model.py:745, 730 | `compress_len` is a host int; the `== 0` branch | decode with a prompt of ≥ 2 tokens never hits 0 at ratio 1/2 → assert at capture | — |
| 8 | model.py:764, 776-779 | `compress_kv[:, :T/r]` and `torch.cat` give dynamic length | **stopgap:** cat with the *full* compressed cache (max_seq_len/r rows), indices unchanged. **Final:** T7's two-source `sparse_attn` (plan §5(d)) | stopgap: same rows gathered by the same indices → bit-identical. Cost ~0.9 GB/step of copy at 32K (est. ~0.3 ms) until T7 |
| 9 | model.py:779, 579-581 | topk width `128 + min(512, T/r)`; `topk(k)` with dynamic k | width fixed at 128+512, k=512 over the full-length scores, positions ≥ T/r set to −inf before topk; existing `where(idxs < compress_lens, …, −1)` | T/r ≤ 512 (the whole W0 fixture): the reference selects every reachable position and so does k=512, the ascending sort gives the same order, and the tail is −1. Fully −1 blocks in `sparse_attn` are no-ops (`exp(m−m)=1`, adds of 0; kernel.py:360-376). **T/r > 512: ties in bf16 scores may resolve differently on a longer input. Test at long context; do not assume** |
| 10 | model.py:555, 564-568 | `index_k[:, :T/r]` length; `compress_lens` int | full-length buffer + mask; `compress_lens` on device | on the fixture selection = everything, so score bits cannot matter. At long context, bmm/reduce configs may change with shape (risk 6) |
| 11 | model.py:570-576, 597-611 | candidate blocks: `F.pad` width, `num_blocks`, `topk(min(2048, nb))` | fixed `nb = Tmax/r/8` (4,096 at 32K), −inf beyond reach, `last` from device `compress_lens` | ≤ 2,048 reachable blocks: all kept, identical. Beyond that: same tie caveat as #9 |
| 12 | model.py:477-484, 537, 747-748 | ratio-2 compressors (L2, L8, L14) pool only when `(pos+1) % 2 == 0`; latent `None` otherwise | **two graphs, `g_even` / `g_odd`**, chosen by the host from pos parity (all three flip together). Slot `pos%2` is a constant in each | each graph runs exactly the reference ops for its parity: no dead compute, no junk rows |
| 13 | model.py:542, 547, 757, 762 | `freqs_cis[pos+1-r]`; writes at `pos // r` | device gather; `index_copy_` at `pos // r` | exact |
| 14 | model.py:896-902 | `bincount().tolist()` and `torch.where` → 7 host syncs per layer; a data-dependent `Expert` choice | **the spike pattern:** device gate → `sort` ids ascending (the reference's accumulation order) → pointer-table Triton gather into static `[6,N,K/2]` staging → 6 fixed `Expert` evaluations → fp32 sum in slot order → + shared | **measured:** 32/32 `torch.equal` eager, 16/16 under graph replay, 1,828 µs/layer replay vs 5,389 eager reference (spike receipt) |
| 15 | kernel.py:111-112, 196-197, 299, 398, 469-471, 583 | wrapper allocations on every call | the graph's private pool gives stable addresses | fills off at capture (§1b) |

Everything else in the decode path has fixed shape: `Gate`'s topk(6), Engram masks, mHC, head `x[:, -1]` (model.py:1013), argmax (1292).

### 2.2 Scope

At 1 row, **the whole step (hash → embed → 40 layers → head → argmax) is one graph, in two parity variants.** Only these stay eager:

- prefill (s>1, `start_pos=0`);
- the host loop, which picks `g_even` or `g_odd` and digests logits in parity mode.

Cold-expert staging stays **inside** the graph, as the spike's SM gather reading UVA addresses from a device pointer table. A copy-engine memcpy cannot take a data-dependent source address inside a graph, so host-driven staging is excluded by construction.

### 2.3 Capture plan

1. Boot as today. Prefill a dummy prompt eagerly. This sets every state that warm-up touches; a later real prefill overwrites from position 0 (runtime generate.py:21-35 argues that nothing stale is ever read).
2. Allocate the static buffers: `tok`, `pos`, `logits[1,129280]` fp32, `ids[1]`.
3. Warm up 2× per parity on a side stream. This compiles every TileLang JIT kernel (2–13 s each, ENVIRONMENT.md:30) and the Triton gather, creates the cuBLAS handle/workspace (`CUBLAS_WORKSPACE_CONFIG` is set before torch import), and settles the allocator.
4. Capture `g_even` and `g_odd` with `torch.cuda.graph(g, pool=shared_pool)` at a representative pos of each parity.
5. KV writes use device indices derived from `pos` inside the graph (#2, #5, #13).
6. Add a once-per-step marker kernel, plus optional per-layer markers, because NVTX does not survive replay (plan §5(f)).
7. Keep `--graph off` as a flag that runs the same static code eagerly.

### 2.4 How bit-identity is proven (the first-light instrument: fp32-logit sha256 per step, 32 steps × 8 prompts)

- **L0, new code == reference, eager.**
  - Unit `torch.equal` on: device window indices vs `get_window_topk_idxs` for pos 1…4,096; padded vs dynamic topk idx; full-cat vs sliced `sparse_attn`; static MoE vs reference (already done, 32/32).
  - Then whole-step digests, static-eager vs reference-eager, over the 8 prompts.
- **L1, graph == static-eager.** Replay digests equal to static-eager digests, same 32 steps × 8.
- **L2, determinism.** Two replays plus one relaunch are equal. Run this with fills **on and off**: equality under both proves that no uninitialized byte is consumed.

**T8 gate.** All three levels pass, and the step is **≤ 110 ms at 1 row** (est. from measurements: 40 × 1.83 ms MoE replay = 73 ms, + 24 ms non-MoE GPU busy, + ~6 ms of graph node gaps). Above 130 ms, something is serializing: stop and trace with `--cuda-graph-trace=node`.

## 3. T9a: routed experts via DeepGEMM (no `import vllm`)

Loading: the image ships DeepGEMM as `vllm.third_party.deep_gemm` (utils_deep_gemm.py:204), and importing that runs `vllm/__init__`. Instead, put `…/vllm/third_party` on `sys.path` and `import deep_gemm`. Its `__init__` imports only torch and its own submodules (deep_gemm/__init__.py:1-111; there is no `vllm` string anywhere in `deep_gemm/*.py`). The DeepGEMM JIT compiles with NVCC at first call, so it needs a persistent `DG_JIT_CACHE_DIR` mounted like TileLang's, and a warm-up before capture.

### 3.1 Numerics, stage by stage

| stage | reference | (i) grouped GEMM (masked/contiguous) + reference glue | (ii) `fp8_fp4_mega_moe` |
|---|---|---|---|
| routing | fp32 gate, topk 6 (model.py:813-828) | same (ours) | same (ours) → we write `topk_idx` (int64, cuh:1399) / `topk_weights` (fp32) into the SymmBuffer views. **Do not use** `bf16_mega_gate` (bf16; utils_deep_gemm.py:489-535) |
| L1 input | `act_quant(x,32,ue8m0)`: e4m3 + pow2 E8M0/32, amax floor 1e-4 (kernel.py:76-78) | same bytes → SF re-laid out | same bytes into `sym_buffer.x/x_sf`. DeepGEMM's UE8M0 floors equal the reference's (math.cuh:125-127) |
| weights | fp4 `[N,K/2]` + E8M0 `[N,K/32]`; recipe (1,1,32) = per-row × 32-K for act and weight (mega/__init__.py:183) | bytes as-is; SF → packed int32, layout via `transform_sf_into_required_layout` | gate/up interleaved at granularity 8, SF interleaved + UTCCP row transpose within 128-row groups, packed int32 (mega/__init__.py:120-172, assert 139) |
| L1/L2 accumulate | per 32-K: fp8 MMA, then fp32 `acc += c·sa·sb` in K order (kernel.py:535-554) | block-scaled UMMA, scales inside the MMA | same as (i) |
| gate/up | bf16 output, then fp32, clamp (model.py:845-849) | bf16 output → reference torch glue | **bf16 round + bf16 clamp (cuh:1058-1066): equal** |
| SiLU·up·w | torch fp32 `silu(g)*u*w` (model.py:850-852) | reference ops | same order (cuh:1088-1089). `fast_math=True` default: `__expf` + `fast_rcp` (cuh:1070-1083) → **pass `fast_math=False`** |
| L2 input | `.to(bf16)` then `act_quant` (model.py:853) | reference ops | **fp32 → e4m3 directly, amax from fp32** (cuh:1092-1136): differs when bf16 rounding crosses an fp8 tie or a pow2 amax boundary |
| combine | `y` fp32, `+=` bf16 in **expert-id order**, + shared, → bf16 (model.py:895-906) | ours, in sorted-id order: equal | fp32 `add.rn.f32.bf16` (ptx/utils.cuh:65-69) in **top-k column order**, shared last (cuh:1397-1466) → **sort columns by expert id first: equal** |
| shared expert | fp8, 32×32 block SF (model.py:227-232) | reference `fp8_gemm` (or DG later) | fused (cuh:148); SF re-tiled 32×32 → 1×32 (exact), UMMA accumulation |

**What breaks bit-identity, by design:**

- the block-scaled UMMA accumulators, in both paths;
- in (ii) only, the L2-input quantization source.

**Expected drift class (est.):** ≤ 1 bf16 ulp on a fraction of GEMM outputs, propagating to mean |Δlogprob| in the A3 class (≤ 0.05 nats). That is smaller than one KV-numerics knob, which measured 0.004–0.055 ours-vs-ours (`FIRST-LIGHT-20261106.md`). Measure it before believing it: report the per-layer bf16 mismatch rate vs `ref`.

**The knob:** `--moe-impl {ref, dg-masked, dg-mega}`.

- `ref` is the T8 static MoE: reference kernels, bit-identical, used for every parity run.
- Each impl must pass self-replay `torch.equal`.

### 3.2 The two paths

**(i) Grouped GEMM.**

- *Masked over the hot pool is impossible* (§0.6).
- **Masked over the spike's `[6,N,K/2]` staging** is the smallest possible change: 3 launches instead of 18 `fp4_gemm`. The SF transform runs per call on the 6 gathered experts only, so pool layout is untouched and `ref` keeps working on the same boot.
- **Contiguous over `[H+S,N,K/2]`** reads hot experts in place, stages only cold ones, and lets the cold GEMM run as a second launch after the hot one, overlapping the copy (our combine makes the split numerically invisible). It costs a pool-wide SF permutation. A copy would not fit: SF is 1/17 of the pool, ≈12 GiB. `ref` then un-transforms inside its gather.
- `m_indices` must be padded to DeepGEMM's BLOCK_M. A mismatch gives an IMA under graph replay (utils_deep_gemm.py:398-413).

**(ii) MegaMoE.**

- *What it fuses:* top-k dispatch into per-expert pools (index atomics only, cuh:363, 379), L1 over interleaved gate|up, SwiGLU + clamp + top-k weight, fp8 requant, L2, bf16 out, combine, and the shared expert's L1/L2. All of it is one persistent launch with grid sync (cuh:1379).
- *World size 1:* the Python needs only `group.size()`, `group.barrier()` (mega/__init__.py:59-67) and `group.rank()` (196). At size 1 the buffer comes from `torch.empty` and the handle is a SimpleNamespace, so **a duck-typed stub group works, with no `dist.init_process_group` and no network.**
- *Tiered pools: no.* The weight TMA addresses `local_expert_idx * shape_n` in one tensor (cuh:775), so all experts share one base and one stride. A cold expert cannot be a separate UVA view. The options are:
  - staging slots appended per layer, with `num_experts = H+S` (≤ 512 per rank, layout/mega_moe.cuh:56). It is a template constant, so H+S must be equal across layers;
  - a VA range stitched with CUDA VMM (hot HBM + host-NUMA physical) so the kernel TMA-reads cold experts straight from Grace. This is speculative, but it is recipe A's thesis with no staging code.
- *Ordering:* the staging gather must be stream-ordered **before** the launch, so cold copies cannot overlap within the layer.

**Recommendation for M0b: (ii) MegaMoE is the end state**, reached through (i)-masked as a one-day step. Reasons:

1. (ii) is the kernel production measured on this box and model (~79 µs/layer for ~25 experts, Pg61). It removes almost all MoE glue launches, including the shared expert. And it is exactly the fused layer whose producer recipe A must teach to read Grace at M2.
2. (i)-masked lands about 80% of T9a's win (73 → ~19 ms of the 73 → ~7 ms total, table 3) with no layout change. It is also the cheapest way to learn the `_C` ABI and to get the first parity-graded DeepGEMM numbers while `ref` is untouched.
3. Contiguous (i) is the fallback if MegaMoE's ABI or SF layout resists, and the better design if cold copies must overlap at a static map.

### 3.3 Cold-staging design (grows into recipe A's self-fetch)

Per layer, inside the graph:

1. gate → topk;
2. sort by expert id (weights permuted along);
3. `slot, tier = loc[l, ids]` (`expert_pool.py:53-62`);
4. compact the cold entries into staging indices `j` with a 6-entry prefix sum, and remap ids: hot → slot, cold → H + j;
5. run the Triton pointer-table gather for the cold entries only (spike kernel, predicated);
6. one MegaMoE call over `pool[0:H+S]`.

Pool change: allocate each layer's per-projection pool as `[H+S, …]` instead of `[H, …]` (`expert_pool.py:70-72`), in the transformed layout. Store the cold pool in the same layout so that staging is a plain byte copy.

- **M0b (1 row):** S = 6, which covers the worst case with no overflow path. Cost 40 × 6 × 18,800,640 B = **4.20 GiB HBM**, inside the ~31 GiB free after boot (218.6 GiB used, `first-light-20261106.json`).
- **M1 (6 rows):** S = 36 per layer would be 25 GiB and does not fit. Use VMM aliasing instead: map one physical staging buffer (two, alternating even/odd layers) behind every layer's hot range. That is 2 × 36 experts = 1.26 GiB, the plan's §4 line.

**Table 3: expected MoE time per layer**

| case | cold experts per layer | cold copy (52.5 µs each, measured copy-engine rate; SM gather may be slower) | hot work | per layer | per step |
|---|---|---|---|---|---|
| T8 `ref` (spike) | 1.42 | inside the gather | 18 `fp4_gemm` | **1,828 µs measured** | 73 ms |
| T9a-1 masked over staging | 1.42 | inside the gather | 3 DG calls (est. ~45 µs) + ref shared 165 + glue | ~470 µs (est.) | ~19 ms |
| T9a-2 MegaMoE, static map, 1 row | 1.42 (measured) | 75 µs exposed (est. range 75–116 at 230–390 GB/s) | ≤ 79 µs (Pg61; 1-row value unknown) + ~20 glue | ~175 µs (est.) | ~7 ms |
| same, adaptive residency (≤ 0.41) | 0.41 (residency research) | 22 µs | ≤ 79 + 20 | ~120 µs | ~4.8 ms |
| MegaMoE, 6 rows, Pg61 rowmap | 2.06 (Pg61 `peer3_stats`) | 108 µs | 79 (Pg61) + 20 | ~207 µs | ~8.3 ms |
| MegaMoE, 6 rows, first-295 map | ~5.9 (est. 25 distinct × 0.237) | 310 µs | 79 + 20 | ~410 µs | ~16 ms |

The consequence for James: at any static map, cold reads cost **≥ 2.75 ms per step at 1 row** (57 × 18.8 MB at 390 GB/s, measured rates), and under MegaMoE they cannot hide within the layer. That is about a third of D2's 8.10 ms bar. Amendment A2 (adaptive residency) is therefore a prerequisite for M0's *speed* bar as well, not only for M2.

## 4. Order and gates

**T8 first**: 73% of the step is host gap, and T8 alone is worth ~3.7× (405 → ≤ 110 ms). T9a's MoE plumbing is the spike, which T8 needs anyway.

| task | after-task target | parity | self-replay |
|---|---|---|---|
| T8 | ≤ 110 ms/step, 1 row | **bit-identical** to reference (L0+L1) | L2, fills on and off |
| T9a-1 (`dg-masked`) | ≤ 55 ms/step (est. ~49 = 103 − 73 + 19) | A3: 0 margin-weighted flips, mean \|Δlogp\| ≤ 0.05 vs Pg61; per-layer mismatch rate vs `ref` reported | `torch.equal` with `deep_gemm.use_deterministic_algorithms(True)` |
| T9a-2 (`dg-mega`) | ≤ 45 ms/step (est. ~37 = 103 − 73 + 7) | same | same |

The profile settled what it was asked to settle (host share, fill share). Four bounded box experiments remain, in this order:

- **E1, before T8 (~15 min).** Capture one full layer (attention + static MoE) and compare replay time to the layer's measured 2.65 ms busy. This measures per-node graph overhead, and it checks that TileLang launches on the capturing stream rather than stream 0.
- **E2, before T9a.** Print `deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked.__doc__`, `transform_sf_into_required_layout.__doc__`, `fp8_fp4_mega_moe` (pybind signatures), and the shape/dtype/stride of every `SymmBuffer` view at (384, 1 token, topk 6, 5120, 2304, 1 shared). This turns the ABI from guessed to read.
- **E3.** For one layer-0 expert, compare DeepGEMM masked against the reference chain: the bf16 mismatch rate and max |Δ| of w1. Garbage means the SF or nibble layout is wrong.
- **E4 (speculative).** Run a DeepGEMM GEMM over a VMM-stitched HBM+host-NUMA weight tensor, measuring TMA bandwidth from Grace.

## 5. Risks, ranked

| # | risk | mitigation |
|---|---|---|
| 1 | **The `_C` ABI and semantics are guessed from the wrappers.** vLLM's callers are not vendored (§0.7). | Vendor `vllm/models/deepseek_v4/nvidia/model.py` read-only (desk, from the image); run E2 and E3 before any T9a code |
| 2 | **SF layout:** packed int32 UE8M0, MN-major/TMA-aligned for the GEMMs, K-major + UTCCP + interleave for mega. The pool SF cannot be duplicated (~12 GiB). | (i)-masked transforms per call; mega permutes in place; `ref` un-transforms in its gather; round-trip `torch.equal` test |
| 3 | **CUDA-graph capture across JIT kernels.** TileLang, Triton and DeepGEMM compile on first call; an implicit stream-0 launch breaks capture; DeepGEMM's `m_indices` padding must match BLOCK_M (IMA under replay). | Warm both parities; E1; mount JIT caches; pin BLOCK_M via `set_mk_alignment_for_contiguous_layout` |
| 4 | **MegaMoE under `--network none`.** The Python needs no process group at size 1 (stub). The kernel assumes a co-resident persistent grid (grid sync), so concurrent kernels delay it, and two grid-syncing persistent kernels can deadlock. | Stub group; no concurrent streams during the MoE; `set_num_sms` if a side stream is ever added |
| 5 | **`use_deterministic_algorithms` interactions.** `warn_only=True` hides nondeterministic ops; fills replay as kernels if captured on; DeepGEMM's own `use_deterministic_algorithms` semantics are unknown. | Fills off for capture; self-replay digests are the only real proof; turn DeepGEMM's flag on and check equality |
| 6 | **Long-context fixed shapes.** k=512 / 2,048 tie resolution, and shape-dependent bmm/reduce configs in the indexer; invisible on the ≤310-token fixture (plan §7). | 20–32K needle + self-replay before T7 closes; graph vs static-eager equality at long T |
| 7 | **Static-map cold exposure** (≥ 2.75 ms/step at 1 row) | A2 timing decision for James |

T8T9A: T8 first: graph the whole 1-row step with the spike's static MoE, in two parity graphs, at ≤ 110 ms/step and bit-identical to the reference. Then T9a with DeepGEMM: masked-over-staging as the 1-day ABI/numerics step (~50 ms/step), then world-size-1 MegaMoE over [hot pool ∪ cold staging] as the M0b kernel (~37 ms/step). Because the static map's ≥ 2.75 ms/step of cold reads cannot hide under MegaMoE, adaptive residency (A2) is needed for the D2 speed bar.
