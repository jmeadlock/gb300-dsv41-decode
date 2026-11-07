# 6-row step anatomy (2026-11-07 05:00 CST) — 21.8 ms GPU busy; the MoE is 12.5 of it

nsys on the static-eager 6-row teacher-forced path (dg-inplace + gemv + fused R-row glue + FlashMLA + seed map, S=16), 32 steps. Receipt `step-anatomy-rows6-20261107.json`. GPU busy **21.80 ms/step** vs graphed 24.10 (the gap is the per-step teacher-forced H2D/bookkeeping outside the kernels, worth checking in T12); **2,826 kernels/step** (1-row: ~2,000). Roles: **moe 12.53 ms / 920 launches**, everything else 9.28 / 1,906.

| kernel | ms/step | launches | µs | vs 1 row | lever |
|---|---|---|---|---|---|
| DeepGEMM `sm100_fp8_fp4_gemm` (routed experts, contiguous) | **5.79** | 120 | 48.3 | 1.64 / 13.7 µs | ~22 unique experts × 3 projections per layer at `M_total = groups × 128 alignment` — most rows are padding. The memo's L3 "MegaMoE-style" rung (one call per layer, row-packed) is the lever; or `m_grouped_…_masked` with the real row counts (masked_m ≤ 6) — probe both |
| `t9b_mxfp8_gemv` + `gemv2` + splitK | 3.15 | 345 | | 3.6 | GEMV at M=6 is fine (W3's rewrite still applies) |
| `t9a2_stage_inplace` | **1.72** | 40 | 43.0 | 0.37 / 9.2 | the staging kernel at 6 rows writes scales for all 36 pairs (`sf_store=step`); it should only move cold rows' bytes — rewrite as "pack SF once per unique expert, stage only cold" (−1.3) |
| `Kernel2` (unnamed Triton: likely a fused-glue cluster at R rows — identify) | 1.67 | 138 | 12.1 | — | identify by launch count (138 ≈ 3.45/layer) and fix its per-row serial loop |
| `m1_dedup_routes` | **1.64** | 40 | 40.9 | — | 41 µs for a 36-element sort/unique/cumsum: one program doing serial work; use a warp-level bitonic over 36 pairs (−1.5) |
| fp32 head `sgemm_largek_lds64` (M=6) | 0.67 | 1 | 669 | 0.42 (gemv2T) | cuBLAS picks a large-K sgemm at M=6 (not row-invariant either — W4); the bf16-stored fixed-order fp32 FFMA kernel fixes both (−0.4) |
| torch glue residual (`vectorized_` 0.75, `elementwise_` 0.57, `reduce_` 0.55) | 1.87 | 876 | ~2 | 1.4 | R-row glue leftovers |
| `t9b4_hc_norm` 0.65 · nvjet 0.57 · `gatherTopK` 0.42 · DG SF convert 0.41 · `norm_row` 0.36 · sinkhorn 0.34 · FlashMLA 0.43 | 3.2 | | | | as at 1 row (FlashMLA b=6 ≈ b=1 ✓) |

Reading: the 1-row → 6-row growth (11.3 → 24.1) is **MoE 4 → 12.5** (DeepGEMM padding 4.2, staging 1.3, dedup 1.6, SF convert) plus ~2 ms of R-row glue that is still per-row serial. Everything attention/GEMV/norm-side is row-invariant in time, as the memo predicted. Fixing the four MoE items (DG call shape, staging, dedup, SF pack) is worth ~7 ms → **~17 ms per 6-row step**, and that is before any M2 fusion; the memo's 13–15 assumed exactly this MoE cost.

Next coder (M1-T10-MOE): (1) DeepGEMM at 6 rows — probe `masked` with real `masked_m` vs `contiguous` with 128-row alignment vs a smaller alignment (`get_mk_alignment_for_contiguous_layout`), pick the fastest bit-equal path; (2) stage only cold rows, SF pack once per unique expert; (3) dedup as a warp-level kernel; (4) identify `Kernel2`.
