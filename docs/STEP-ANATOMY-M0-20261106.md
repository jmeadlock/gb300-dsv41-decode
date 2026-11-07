# Step anatomy — M0 runtime, 2026-11-06 (`bench/step_anatomy.py` on `scratch/prof/step-m0.sqlite`; receipt `step-anatomy-m0-20261106.json`)

nsys `cudaProfilerApi` window, 64 decode steps after 32 warm, 1 row, prose prompt, NVTX per layer/role. Traced steps run 560 ms (tracing cost on 20k launches); untraced: **404.9 ms** with deterministic NaN-fill, **357.1 ms** with `--no-fill-uninit` (same tokens).

| per step | measured |
|---|---|
| kernels launched | **20,550** (Pg61 v27: 1,305) |
| GPU busy (union of kernel intervals) | **108.6 ms** |
| host/launch idle inside the step | 454 ms traced (≈ 300 ms untraced) |
| `fp4_gemm` (routed experts) | 720 launches × 78 µs = **56.3 ms** (6 experts × 3 GEMMs × 40 layers) |
| `fp8_gemm` (attention/shared/dense) | 290 × 55 µs = 15.9 ms |
| `vectorized_elementwise` glue | **11,570** × 1 µs = 11.7 ms |
| `sparse_attn` | 40 × 75 µs = 3.0 ms |
| `act_quant` | 1,090 × 2.6 µs = 2.8 ms |
| per layer GPU | 2.6–2.8 ms, flat across layers (Engram layers no heavier at 1 row) |
| by role (GPU) | moe 84.4 · attn 17.6 · other 5.9 · head 0.4 · engram 0.4 |

## Reading
1. **The step is 73 % host overhead.** 20,550 eager launches from Python at ~15–20 µs each is the 300 ms; the GPU would finish the same work in 109 ms. T8 (one CUDA graph per step) is the first task and is worth ~3.3× on its own — before any kernel changes.
2. **Of the 109 ms of GPU time, 56 ms is the reference `fp4_gemm` at 18 CTAs/launch** — latency-bound, one expert per launch. Pg61's grouped path does a whole layer's experts in ~79 µs; that is 56 → ~3 ms. T9a is the second task and is worth most of the remainder.
3. **11,570 one-microsecond elementwise kernels** per step is the reference's torch glue (routing, masks, dtype casts, QDQ scaffolding). Under a CUDA graph their launch cost vanishes but their 11.7 ms of GPU time does not; that is T9's "fused glue" tail, after the GEMMs.
4. Attention is 17.6 ms (fp8 GEMMs at 55 µs each — also latency-bound at M=1) + 3 ms `sparse_attn`; FlashMLA/fused projections are T9b.
5. **Projection after T8 + T9a, no other change:** ≈ 109 − 56 + 3 ≈ **56 ms/step at 1 row**, unfused. After T9b (attention) and glue fusion, the floor of this kernel set is ~15–20 ms; the rest of the way to 7.36 → 5.0 is the DSpark 6-row path (per-step amortization) and the fused chain. Consistent with the charter's thesis that fusion is the last 10–15 %, not the first.
6. NaN-fill under `use_deterministic_algorithms` costs 48 ms/step (405 vs 357). Self-replay was bit-identical with it on; T8 must re-check with it off (it is a debugging aid, not a numerics rule).

## Spike: graph-capturable MoE, bit-identical (`bench/moe_static_spike.py`, receipt `moe-static-spike-20261106.json`)

The reference `MoE.forward` (model.py:891–905) is why the step cannot be graphed as-is: `bincount().tolist()` is a host sync and the 384-iteration Python loop per layer is 15,360 Python iterations per step. Replacement tested on real layer-0 weights (368 experts in HBM, 16 deliberately in registered Grace memory):

routing on device → sort the 6 ids ascending (the reference's accumulation order) → **one Triton gather through a per-expert pointer table** into a static `[6, N, K/2]` staging buffer (the table holds HBM or UVA host addresses; the kernel does not care which) → 6 fixed-shape reference `Expert` evaluations on the staging slots, fp32 accumulate in slot order → + shared expert.

| | result |
|---|---|
| static vs reference, 32 random inputs (7 cold hits over C2C) | **32/32 `torch.equal`** |
| CUDA-graph replay of the static version vs reference, 16 inputs | **16/16 `torch.equal`** |
| reference MoE, one layer, eager | 5,389 µs |
| static MoE, eager | 4,382 µs |
| **static MoE, graph replay** | **1,828 µs** (= 18 reference `fp4_gemm` launches ≈ 1.4 ms + shared + gather) |

Projection: 40 × 1.83 ≈ 73 ms/step for MoE under a graph with the reference kernels, vs 84 ms GPU + its share of 300 ms host today. The remaining 1.4 ms/layer is the latency-bound `fp4_gemm` — T9a's grouped GEMM over the same staging buffer. The pointer-table gather is also recipe A's self-fetch seed: cold experts are read over C2C by the same kernel, no sidecar, no host decision.
