# Step anatomy after T9b-1 — the 32 ms step, fully attributed (2026-11-06 22:30 CST)

nsys on the **static-eager** path (`--mode static --moe-impl dg-masked --fp8-impl gemv`: the same kernels the graph replays, launched eagerly so CUPTI can see them), 32 steps, NVTX per role. Receipt `step-anatomy-t9b1-20261106.json`. Eager wall is 315 ms/step (launch gaps); **GPU busy is 32.71 ms/step — equal to the graphed step (32.16). The graph is GPU-bound; there is no launch overhead left to remove, only kernel time.** 7,739 kernels/step.

| kernel (per step) | ms | launches | µs each | what / next lever |
|---|---|---|---|---|
| elementwise + reduce glue (`unrolled_` 4.06, `vectorized_` 3.11, `elementwise_` 2.97, `reduce_` 1.64) | **11.8** | 5,144 | 1.3–3.6 | torch glue — the census counted launches right (~5,300) but at 1 µs; it is 2.3 µs mean → **glue is 12 ms, not 5**. T9b-2 R1 (running) removes ~2,150 launches ≈ 5 ms; R2 the rest |
| `sparse_attn_kernel` | **5.12** | 40 | 128 | the T9b-3 memo's correction confirmed: static step walks 640 padded indices serially (eager anatomy saw 75 µs at 2–5 blocks). **T9b-3a**: per-row valid-block bound, bit-identical → ~2–3 ms saved |
| `t8_gather_rows` (expert staging) | **4.20** | 240 | 17.5 | 113 MB/layer at ~1.1 TB/s. Levers: (a) gather kernel at HBM rate (~3 TB/s) → 1.5 ms, numerics "none"; (b) in-place hot MoE (DG contiguous over the pool) removes it for hot experts — mandatory at 6 rows (×3.7 unique experts → 15 ms otherwise) |
| `t9b_mxfp8_gemv` + `t9b_splitk_reduce` | 3.61 | 540 | 9.6 / 3.3 | as benched |
| DeepGEMM `sm100_fp8_fp4_gemm` | 1.82 | 120 | 15.2 | routed experts; ~fixed |
| `act_quant` | 1.22 | 450 | 2.7 | fuse into producers (R1 #2, R2 #1/#6) |
| `transpose_and_pack_fp32_into_ue8m0` (DG scale convert) | 1.01 | 240 | 4.2 | R1 cluster #9 removes |
| cuBLAS `gemv2T` (fp32 head, 2.65 GB) + `head` role | 0.84 | 3 | 210–419 | the reference widens the head to fp32; at HBM rate it is ~0.35 ms; leave |
| `nvjet` (aten mm, mHC/gate) 0.59 · `gatherTopK` 0.38 · sinkhorn 0.35 · bitonic sort 0.32 · `dot` 0.29 · Engram 0.27 | 2.2 | ~380 | | small; R2 clusters #4/#5 |

Role view: moe 14.2 ms (gather 4.2 + DG 1.8 + its glue/quant/convert ~8), everything else 17.8, head 0.4, engram 0.3.

## Consequences
1. **M1 memo §3's "14.8 ms unattributed" is resolved**: it is glue (11.8 total) + the padded `sparse_attn` (5.1) + the staging gather (4.2), minus what the census had already counted. All three are row-invariant or worse at 6 rows (the gather is ×3.7), so the memo's R ≈ 11–18 ms after T9b-2/3 stands, and in-place hot MoE moves from "nice" to required for M1.
2. Tonight's order by measured ms: T9b-2 R1 (glue, in flight) → **T9b-3a** (`sparse_attn` valid-block bound; bit-identical) + gather-kernel bandwidth tune (numerics none) as one coding round → T9b-2 R2 (medium-risk glue) → in-place MoE (A2 memo + T8-T9A §3.2).
3. Projection for the 1-row step after those four: 32.2 − 5 (R1) − 2.5 (3a) − 2.7 (gather) − 4 (R2) − 1 (#9) ≈ **17 ms**, consistent with the 15–20 ms kernel-set floor named earlier.
