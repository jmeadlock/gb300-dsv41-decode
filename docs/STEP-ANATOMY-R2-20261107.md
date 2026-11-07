# Step anatomy at 13.9 ms (2026-11-07 01:15 CST) — the residual, measured

nsys on the static-eager path, full current configuration (`dg-inplace` + `gemv` + `fused` R1+R2 + `flashmla` + seed map), 32 steps. Receipt `step-anatomy-r2-20261107.json`. **GPU busy 13.95 ms/step = graphed 13.88**; 2,827 kernels/step (20,550 this morning).

| kernel | ms/step | launches | µs | lever |
|---|---|---|---|---|
| `t9b_mxfp8_gemv` + `t9b_splitk_reduce` | **3.58** | 290 + 250 | 9.6 / 3.2 | 290 fp8 projections. Bytes: ~5.9 GB/step of fp8 weights → ~0.75 ms at 8 TB/s; we are at ~1.7 TB/s because N≤2304 shapes are latency-bound at 9 µs. Levers: fold the split-K reduce into the consumer (−0.8); merge `wq_a‖wkv`, `w1‖w3`, indexer `wq_b‖wk` into one launch each (−~80 launches, −~0.7); persistent "GEMV megakernel" per layer running all its projections from one launch (M2 territory) |
| `t9b2r2_hc_norm` | **1.73** | 81 | 21.3 | R2's biggest kernel: hc_pre + RMSNorm + act_quant over 15360→5120 per row — 21 µs for ~100 KB of data is latency, not bandwidth (one program per row). Split the reduction across programs or fuse it into the GEMV prologue it feeds (−~1.2) |
| DeepGEMM routed | 1.64 | 120 | 13.7 | ~fixed (3 calls/layer); one call if w1‖w3 were interleaved in the pool (−0.5, a loader change) |
| torch glue residual (`vectorized_` 0.62, `unrolled_` 0.45, `elementwise_` 0.33) | 1.40 | 699 | 2 | the clusters R1/R2 left (gather, concat_free, sigmoid, sinkhorn surroundings, misc) |
| `nvjet` mm (mHC/gate) | 0.58 | 40 | 14.5 | aten mm on tiny shapes; a GEMV-class kernel would do it in ~4 µs (−0.4) |
| fp32 head `gemv2T` | 0.42 | 1 | 420 | 2.65 GB read → 0.33 ms at 8 TB/s; leave |
| `gatherTopK` 0.39 + `bitonicSort` 0.33 + `dot` 0.29 | 1.01 | 209 | | the router/indexer top-k family (tie order kept in torch) |
| `t9a2_stage_inplace` | 0.37 | 40 | 9.2 | cold staging + m_indices; could be skipped when no cold rows (device-predicated) |
| `sparse_attn_fwd` (FlashMLA) | 0.37 | 40 | 9.2 | |
| `hc_split_sinkhorn` | 0.35 | 80 | 4.4 | reference TileLang; fusable with its mixes |
| `transpose_and_pack_fp32_into_ue8m0` | 0.34 | 120 | 2.9 | DeepGEMM converts the *activation* SF per call (3/layer): pass pre-packed SF (−0.3) |
| `t9b2r2_norm_row` 0.27 · `kernel` 0.22 · rest | ~1.0 | | | |

Roles: moe 5.85 (1,320 launches), everything else 8.10.

**Reading.** At 13.9 ms the step is ~2,800 launches averaging 5 µs — the GPU is now launch-latency-bound *inside the graph*: most kernels move < 1 MB and take 3–20 µs regardless. Bandwidth floor for the bytes actually touched (~5.9 GB fp8 weights + 2.65 GB head + ~0.5 GB experts/KV ≈ 9 GB) is ~1.2 ms at 8 TB/s. The 1-row path has ~3–4 ms of straightforward wins left (gemv merges + reduce fold, hc_norm split, nvjet → gemv, SF pre-pack, cold-staging predicate) → ~10 ms; beyond that the structure must change — fewer, longer kernels (M2's fused chain) — or the rows must grow (M1), which amortizes the latency-bound kernels: a 6-row step costs about the same per kernel as a 1-row step everywhere except the MoE (more unique experts) and the head. That is why the M1-T10 R6/R1 measurement (running) is the most important number left.
