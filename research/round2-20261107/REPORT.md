# REPORT — research round 2 (2026-11-07, 03:35–04:30 CST): what gets a 6-row step toward 5.0 ms, and what we got wrong

## Verdict
**The 5.0 ms/step bar at 6 rows is a bytes bar the hardware cannot meet with stock weights.** A 6-row DSpark verify+draft step moves ~28.8 GB (16.9 GB of it the ~22 unique routed experts per layer × 40 layers); at the Station's measured HBM copy rate (6.3 TB/s) that is 4.6 ms of pure bytes, and at the best per-class rates achieved on this box 6.3 ms, before any latency-bound work (M2 memo §5.1). Production's 7.36 ms needs only 3.9 TB/s. **Token yield cannot rescue the bar**: production accepts by exact greedy match with the confidence head off and gets 2.6–2.9 tokens/step on prose; the block-5 drafter cannot usefully go past k=5 (W2). So the honest targets are: **beat 7.36 ms per 6-row step (reachable: ~9 ms by the M2 fusion ladder bottom-up, ~7.3 ms with the per-layer persistent stretch), and report ms/token** (today 8.3 at 21+2 ms; 7.36/2.8 = 2.6 for production). The 5.0 figure stays as the charter's stated bar until James re-states it; the project does not quietly change its own goalposts.

## What the round corrected in our own record
1. The 1-row step touches **15.9 GB**, not 9 (routed experts 4.5 GB + bf16 `wo_a` 2.7 GB were omitted) → bytes floor 2.5 ms, not 1.2 (M2 §0.2).
2. The 10.5 ms step is ≈ **2,000** kernels, not 2,800 (post-T9b-4 census; M2 §0.1). Nobody has an anatomy at 10.5 yet.
3. **No idle gap between graph nodes**: Σ kernel durations = step time; the ~5 µs/kernel is ramp + first-DRAM + tail *inside* each kernel; pure dispatch ≈ 0.4–0.55 µs/node ≈ 0.8–1.1 ms of the step (W1 §1.1, M2 §0.3). A removed boundary saves 1.5–3 µs, not 4.
4. The GEMV is **latency-bound, not bandwidth-bound**: 208/290 calls pay ~8 µs fixed on 2.6–12 MB weights (32 B/row/K-block loads, few bytes in flight, a second split-K launch) (W3).
5. R6/R1 = 2.0 and 3.57 both measured "how much of the step is still unfused at 6 rows", not hardware; a fully ported 6-row fast path models at 13–15 ms (M2 §0.4; measured 40.2 with 16 clusters on reference glue).
6. 6-row non-invariance is at **two** cuBLAS fp32 sites — the 80 mHC-mix calls and the head — not the head alone (W4).

## Levers, dispositioned
| lever | ms (6-row, modeled) | numerics | disposition |
|---|---|---|---|
| **M1 port** — fused glue at R rows, GEMV pair/fold at R rows, dedup glue, S=16 | 40 → ~14–16 | bit-identical per cluster (G rows) | **DO NOW** (coder running) |
| **M1-T11/T12** draft + loop → tokens/step, ms/token instrument | yield | bit-identical iff verify rows row-invariant (W2) | **DO NOW** (T11 on box; T12 next) |
| **Row-invariance**: fixed-order fp32 FFMA kernel for mHC mixes + head; or cuBLAS 13.8 experimental batch-invariant mode | 0 (head half −0.2) | makes 6-row emitted stream == 1-row greedy bit-for-bit | **ACCEPT** → T12 prerequisite for the bit-identical headline instrument |
| M2 fusion ladder L1–L7 (mixes, router+plan, MegaMoE-style grouped MoE, bf16-stored head, PDL, folds, cold overlap) | 16 → ~9 (bottom-up) | mostly bit-identical; MoE/attention A3 | **ACCEPT**, order per memo §6 |
| GEMV bytes-in-flight rewrite + same-input groups (W3) | −2.0 | bit-identical by construction | **ACCEPT** = M2 L6-adjacent |
| PDL on Triton + DG `set_pdl` (W1) | −0.5 to −1.5 | none | **ACCEPT WITH MOD**: after the node count falls (it is a rebate on remaining boundaries) |
| bf16 **weights** for the head/`wo_a` read, fp32 math kept (W3/W4) | −0.2 / −0.4 | 1-ulp class (not A1: values are the checkpoint's bf16) | ACCEPT WITH MOD — confirm the reference's own dequant path = our read |
| k > 5 drafts, confidence-gated skipping | — | — | **NEVER (this round)**: paper shows 7 rows worse; confidence head is computed post-draft (W2) |
| verify-depth truncation via `masked_m` | 1–8 % ms/token | bit-identical | LATER (after T12) |
| A2-2 adaptive residency | ~0 on fixture; shift robustness | none | DO NOW (merge after PORT; box probe) |
| per-layer persistent kernel (stretch) | 9 → ~7.3 | A3 | SPIKE → WATCH after L3 |
| 5.0 ms bar | — | — | **route to James**: physics says no at 6 rows/stock weights; options: (a) re-state as "beat 7.36 and publish ms/token", (b) keep 5.0 as aspirational with the bytes floor published, (c) widen scope (fewer rows per step + higher yield is not available; weights/KV changes are banned by charter) |

## Orchestrator verification (what I checked myself)
| claim | how | result |
|---|---|---|
| Σ kernel durations = graph step (no inter-node gap) | `step-anatomy-r2-20261107.json` top_kernels sum vs step cell | 13.91 vs 13.88 ✓ |
| 6-row 40.2 ms is glue, not hardware | `[glue]` line of the run: only `swiglu_shared,hc_post` fused at rows 6 | ✓ |
| overflow at S=8 | run output: 123/1018 steps | ✓ |
| vLLM accepts by exact greedy match, confidence off | W2 cites v0.31.0 source lines; not independently re-read this round | **worker claim** — verify when T12 is written (the T12 coder reads the same file) |
| 28.8 GB per 6-row step | M2 memo's shape arithmetic; experts 22.45 × 18.80 MB × 40 = 16.9 GB ✓; draft 3.4 GB not re-derived | partially verified |

## Box probes queued (≤ 30 min each)
1. anatomy at 10.5 ms (1 row) — the ladder's input (M2 T13 first job).
2. PDL probe: same graph, `launch_pdl` on our Triton kernels + DG `set_pdl` → step cell (W1 §4).
3. GEMV bytes-in-flight prototype on the 5 shape classes (W3 §P).
4. cuBLAS 13.8 batch-invariant mode on the two fp32 sites → row-invariance census (W4).
5. 6-row ported step cell (PORT gate) → the real R6/R1.
