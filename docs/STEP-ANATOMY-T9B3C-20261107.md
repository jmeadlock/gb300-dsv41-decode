# Step anatomy at 19.9 ms (2026-11-07 00:05 CST) — what is left, measured

nsys on the static-eager path with the full current configuration (`dg-masked` + `gemv` + `fused` glue + `flashmla` + seed map), 32 steps. Receipt `step-anatomy-t9b3c-20261107.json`. **GPU busy 20.21 ms/step ≈ graphed 19.87**; 5,745 kernels/step (was 7,739 at 32 ms, 20,550 at 405 ms).

| kernel group | ms/step | launches | note / next lever |
|---|---|---|---|
| torch glue (`vectorized_` 2.20 + `unrolled_` 2.11 + `reduce_` 1.40 + `elementwise_` 0.80) | **6.5** | 3,230 | the medium-risk clusters (norms, router, mHC mixes, kv/q paths) → **T9b-2 R2** (coder running) |
| `t9b_mxfp8_gemv` + split-K reduce | 3.57 | 540 | 290 projections at ~bandwidth; the reduce (0.79) can fold into the next consumer; `wq_a‖wkv` and `w1‖w3` merges (audit #12) save ~80 launches |
| DeepGEMM routed GEMMs | 1.80 | 120 | fixed |
| `t9b2_gather_tables` | **1.48** | 40 | at the copy ceiling; removed by **T9a-2 in-place MoE** (coder running) |
| `transpose_and_pack_fp32_into_ue8m0` (DG scale convert) | **1.12** | 240 | R1's cluster #9 was meant to remove this — it is still here: DeepGEMM converts the *activation* SF internally per call (6 per layer). In-place MoE with resident fp32 scales, or passing pre-packed SF, removes it |
| `act_quant` | 0.79 | 290 | fold into GEMV prologues (R2 / #12) |
| aten `nvjet` mm (mHC/gate) 0.58 · `gatherTopK` 0.39 · sinkhorn 0.35 · bitonic sort 0.32 · `dot` 0.29 | 1.9 | ~330 | R2 #4/#5; the sort+topk pair is the router |
| fp32 head (cuBLAS `gemv2T`) + `head` role | 0.84 | 3 | reference widens the head to fp32 (2.65 GB) — leave for M0 |
| `sparse_attn_fwd` (FlashMLA) | **0.36** | 40 | was 5.12 |
| Engram | 0.27 | 86 | |

Projection: R2 (−3 to −4) + T9a-2 (−1.5 gather − 1.1 scale convert) → **≈ 13–14 ms at 1 row** with the model's arithmetic intact or 1-ulp class. Below that is the floor of this kernel set (GEMV 3.6 + DG 1.8 + head 0.8 + attention 0.4 + Engram 0.3 ≈ 7 ms of bandwidth/latency-bound essentials, plus whatever glue survives). The rest of the way is M1 (6 rows amortize the ~13 ms) and M2 (the fused chain).
