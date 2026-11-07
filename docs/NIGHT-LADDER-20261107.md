# Night ladder, 2026-11-06 21:30 → 2026-11-07 00:45 CST — 32.16 → 13.88 ms/step (1 row)

Same instrument throughout: `bench/first_light.py`, cuda events over steps 32..512 of a 512-token greedy generation, 1 row, real checkpoint, self-replay 8/8 at every rung. Fixed inputs from the morning: 404.9 (eager) → 98.4 (T8 graph) → 44.6 (T9a-1) → 32.16 (T9b-1).

| rung | what | ms/step | numerics vs previous rung | receipt |
|---|---|---|---|---|
| A2-1 | m3 v26ar seed map replaces first-295 (cold share 23.5 % → 0.9 %) | 30.92 | byte-identical | `first-light-seed-20261106.json` |
| T9b-2 R1 | 7 zero/low-risk glue clusters fused (−2,080 launches) | 26.04 | bit-identical (G row 8/8) | `first-light-fused-20261106.json` |
| T9b-3a | `sparse_attn` bounded to valid blocks; 16-B gather at 2.9 TB/s | 23.80 | bit-identical (A row 8/8) | `first-light-bounded-20261106.json` |
| T9b-3c | FlashMLA sparse-prefill op on our bf16 caches | 19.87 | A3-class (0.0006–0.016 nats; kernel fp64 ratio 1.04) | `first-light-flashmla-20261106.json` |
| T9a-2 | in-place hot MoE: DeepGEMM contiguous over the pool, cold rows staged | 17.83 | bit-identical to masked (D row 8/8) | `first-light-inplace-20261107.json` |
| T9b-2 R2 | 7 medium-risk clusters (norm statistics, router, kv/q, mHC mix, engram, indexer) fused by device-calibrated reduction order; torch `topk` kept for tie order (−2,822 launches) | **13.88** | bit-identical (G row 8/8) | `first-light-r2-20261107.json` |

Across the night the only non-bit-identical rung is FlashMLA (and T9a-1/T9b-1 from the day), all in the A3 class; parity vs Pg61 is unchanged at every rung (2/8 identical, 0 weighted flips, same divergence positions). Final output `w0/results/first-light-cand-night-20261107.jsonl` is byte-identical to the T9a-2 run's.

Launch count per step: 20,550 → 7,739 (T9b-1) → 5,745 (T9b-3c) → ~2,900 now. GPU busy ≈ step time since T9b-1: the graph is GPU-bound; the remaining time is kernel time.

Also landed tonight: design memos `M1-DSPARK-DESIGN.md` (6-row speculative step), `T9B3-ATTN-DESIGN.md`, `A2-RESIDENCY-DESIGN.md`; FlashMLA probe; two anatomies (`STEP-ANATOMY-T9B1`, `STEP-ANATOMY-T9B3C`). M1-T10 (6-row teacher-forced verify graph + R6/R1 instrument) dispatched 00:45.

What is left at 1 row (from the 19.9 anatomy minus the two rungs since): GEMV 3.6, DG 1.8, head 0.84, attention 0.36, Engram 0.27, residual glue ≈ 3, DG scale convert (if still present) ≈ 1 → a 1-row anatomy at 13.9 is the next measurement; the floor of this kernel set is ~10–11 ms. From here the headline moves through rows (M1), not through 1-row kernel time.
