# CONTEXT — research round 2, 2026-11-07 (Sat, 03:35 CST) — "what gets a 6-row step from ~21 ms to 5.0 ms, and what did the first 36 hours miss?"

Project: **GB300 DSV4.1 Decode** (`~/hermes/gb300-dsv41-decode`, public at github.com/jmeadlock/gb300-dsv41-decode; charter `PROJECT.md`). Standalone decode runtime for **DeepSeek-V4.1-Flash-1017** (checkpoint `1bae0e10…`, stock weights: fp8 dense, e2m1 MXFP4 experts, 384 experts/40 MoE layers, k=6 routed + shared, mHC residual streams, Engram at L14, DSpark 3-stage MTP drafter; reference `model.py` is the arithmetic oracle). One DGX Station GB300 (sm_103, HBM 249.8 GiB CUDA-visible ≈ 8 TB/s; Grace 494 GiB pinned; NVLink-C2C measured 230 GB/s kernel-read, 390 GB/s copy engine). Image: vLLM 0.31.0 (used as a library source: DeepGEMM, FlashMLA `.so` loaded without importing vllm), torch 2.10 CUDA 13, Triton, TileLang 0.1.12, cutlass/CuTe DSL 4.7.1 (`docs/ENVIRONMENT.md`). Headline metric: **ms per decode step at 6 rows** (DSpark k=5 verify batch) — production vLLM "Pg61" does **7.36 ms**; bar **≤ 5.0 ms**; kill rule M0 by 2026-12-18.

## Numbers of record (all measured this session; receipts in `docs/`)
| what | value | receipt |
|---|---|---|
| 1-row step, eager reference as shipped | 404.9 ms, 20,550 kernels, GPU busy 109 ms | `first-light-20261106.json`, `STEP-ANATOMY-M0-20261106.md` |
| 1-row step now (graph + DeepGEMM in-place grouped MoE + Triton MXFP8 GEMV + FlashMLA + 18 fused glue clusters + seed map) | **10.52 ms**, ~2,800 kernels, GPU busy ≈ step (latency-bound inside the graph) | `first-light-t9b4-20261107.json`, `STEP-ANATOMY-R2-20261107.md` (at 13.9) |
| residual at 13.9 ms: GEMV 3.6 (290 launches, 9.6 µs each, ~1.7 TB/s on N≤2304 shapes), hc_norm 1.7, DeepGEMM 1.6 (120 × 13.7 µs), glue 1.4, nvjet mm 0.6, fp32 head 0.42 (one 420 µs cuBLAS gemv over 2.65 GB), topk family 1.0, FlashMLA 0.37 (40 × 9 µs) | `step-anatomy-r2-20261107.json` |
| bytes touched per 1-row step ≈ 9 GB → ~1.2 ms at 8 TB/s | arithmetic |
| 6-row teacher-forced step, bit-equal to the reference (11/11 gate) | R6/R1 = **2.008** on the slow MoE path (50.8 vs 25.3 ms); fast-path 6-row number being measured now (projection ~21 ms) | `M1-T10-RESULT-20261107.md`, `rows-gpu-test-r2-20261107.log` |
| row-invariance census | all kernels row-invariant except cuBLAS fp32 `F.linear` (head) and the GEMV's tuned M=6 configs | same |
| parity vs production | 2/8 prompts token-identical over 256 tok, 0 weighted top-1 flips, ≤0.05 nats — production-vs-itself with one KV knob scores the same | `FIRST-LIGHT-20261106.md` |
| FlashMLA sparse-prefill op on our bf16 caches | 125 → 9–15 µs/call, fp64 error 1.04× reference, q unquantized (A3 class) | `flashmla-probe-20261106.json` |
| cold experts | seed map (m3 v26ar) → cold share 0.9 % of routed calls; gather at 2.9 TB/s hot / 375 GB/s cold | `first-light-seed-20261106.json`, `gather-bench-20261106.json` |

## Closed levers (do not re-derive)
Host launch overhead (gone: one CUDA graph/step, bit-identical); staging gather (gone: DeepGEMM contiguous in place); glue launch count as such (18 clusters fused bit-identically with device-calibrated reduction order — the remaining ~700 glue launches are sub-2 µs kernels); KV-latent format as the parity-gap cause (falsified); FlashInfer sparse attention (API mismatch, FlashMLA adopted); fp32 scales in the pool (49.6 GiB, rejected — e8m0 kept, DG's packed SF = checkpoint bytes regrouped); per-role nsys of production (two stalled captures, parked).

## Standing policy
Stock weights, no requant, no KV quantization change, no expert pruning (REAP), all 384 experts; layout may change, values may not; bit-identical self-replay by design; every borrowed/written kernel graded against the reference (bit-identical or "1-ulp class" with the drift reported); nothing promoted on a tok/s headline alone. **Research only: no ssh to 192.168.1.*, no docker, no repo edits outside your report file; cite URLs with dates; say "unverified" rather than invent; mark every number measured / modeled / claimed-by-source.**

## The question for this round
At 10.5 ms (1 row) the step is ~2,800 kernels × ~3.7 µs: the floor is now **per-launch latency inside a CUDA graph**, not bytes. Six rows cost 2.0× one row. To reach 5.0 ms at 6 rows we need either far fewer launches (fusion — the M2 memo being written in parallel covers the *fusion ladder*; do not duplicate it) or a cheaper launch boundary, a cheaper per-row cost, and a better token yield per step. Workers answer the parts the fusion memo does not.

## Output contract (every worker)
Write ONE report to the named path, **create the file first and append sections as you go** (a rejected final message must not lose the work). Sections: verdict paragraph → findings with citations (URL + date) → numbers table (measured / modeled / claimed) → "what this means for our runtime" (concrete: which file/kernel, expected ms at 1 and 6 rows, numerics class bit-identical / 1-ulp / A3 / A1) → what must be measured on the box (a ≤ 30-min probe each) → sources. End your last message with a 10-line summary + the path.
