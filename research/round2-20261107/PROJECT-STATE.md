# PROJECT-STATE — GB300 DSV4.1 Decode (living ledger; re-plan v1, 2026-11-07 03:45 CST)

## Lane right now
- Station lane **DARK** (Pg61 not serving; no restore until James says). Production = 4× Spark SGLang lane. Box used in serialized windows by this project only.
- Runtime default path: `--mode graph --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --map docs/maps/hot-v26ar-seed-295.json`; exact path (`ref`/`bounded`) selectable at every switch.
- Repo `main` at the T9b-4 + m1-t10 + fix-2 merge; worktrees: `m1-t11` (draft graph, coder running), `a2-2` (residency, coder running); desk: M2 fused-chain memo (running); research round 2 W1–W4 (running).

## Numbers of record (measured; receipt in `docs/`)
| | |
|---|---|
| 1-row step | **10.52 ms** (95 tok/s), bit-identical ladder from 32.16 except FlashMLA/T9a-1/T9b-1 (1-ulp class) — `first-light-t9b4-20261107.json` |
| 1-row anatomy at 13.9 | 2,827 kernels × ~5 µs; GEMV 3.6, hc_norm 1.7, DG 1.6, glue 1.4, nvjet 0.6, head 0.42, FlashMLA 0.37 — `STEP-ANATOMY-R2-20261107.md` |
| 6-row teacher-forced step | **bit-equal to reference** (11/11); R6/R1 = **2.008** (slow MoE path); fast path: measuring (proj. ~21 ms) — `M1-T10-RESULT-20261107.md` |
| parity vs Pg61 | 2/8 identical / 0 flips / ≤0.05 nats, same as production-vs-itself — `FIRST-LIGHT-20261106.md` |
| production headline | 7.36 ms per 6-row step (C1, DSpark k=5); bar 5.0 |

## Closed levers (do not re-spend)
Host launch overhead (graph) · staging gather (in-place) · glue launch count (18 clusters fused) · KV-latent as parity cause · FlashInfer sparse attn · fp32 pool scales · Pg61 nsys anatomy (parked) · GEMV lever 5 "skip staging when no cold" (wrong under sf_store=step) · first-295 map.

## Open levers, ranked (ms at 6 rows, modeled unless marked)
| # | lever | ms (6-row) | numerics | state |
|---|---|---|---|---|
| 1 | **M1-T11/T12 DSpark loop** — turns 6 rows into ~2.6–2.9 tokens/step: the only lever that moves tokens/s ~3× | — (yield) | bit-identical if acceptance = exact greedy match | T11 coder running; T12 next |
| 2 | **Launch boundary (PDL / fewer graph nodes)** — 2,800 nodes × ~3.7 µs is the floor; PDL overlaps prologue with the previous tail | −2 to −5 (modeled, W1 to price) | none | W1 researching; box probe ≤ 30 min |
| 3 | **M2 fusion ladder** — one kernel per chain segment (~10/layer) | −5 to −8 (modeled, memo pending) | bit-identical where calibrated; A3 where tensor-core | M2 memo running |
| 4 | GEMV at bandwidth (grouped per layer, better small-N) | −1.5 to −2.5 | bit-identical by construction (per-row dots) | W3 researching |
| 5 | fp32 head → faster read / bf16 head | −0.2 / −0.3 | bf16 = **A1-class** (James) | W3 |
| 6 | A2-2 adaptive residency (shift robustness; cold share already 0.9 %) | ~0 on fixture; protects under shift | none (bytes identical) | coder running |
| 7 | T9b-4 lever 3 (nvjet mm → calibrated GEMV) | −0.4 | bit-identical only if calibration matches cuBLAS | probe later |
| 8 | k > 5 drafts (more rows ≈ free in a latency-bound step) | yield | same | W2 to price |

## Open correctness items
- A1 (KV wording), A2 (adaptive residency — A2-1 in use), A3 (parity bar: proposed; GEMV M=6 configs and fp32 head are the only non-row-invariant kernels; pin-vs-drift decision) — **James**.
- `story` prompt drift 0.073 nats over a 10-token prefix on the GEMV path (above 0.05; flagged, not gated).
- M1-T10 fast-path 6-row result + overflow count (running).

## Re-plan v1 (supersedes the M0-PORT-PLAN ordering; M0's deliverables are met except the parity-bar decision)
| slot (CST) | work | box dark? | status |
|---|---|---|---|
| 03:40–04:40 | M1-T10 fix-2 GPU tests + fast-path 6-row cell (the ~21 ms number) | box | running |
| 03:40–05:30 | desk: M2 memo, W1–W4 research, T11 draft graph, A2-2 residency | — | running |
| 04:45 | lock re-plan v2 from W1–W4 + M2 memo | — | pending |
| ~05:00 | box: T11 oracle test + draft bench (ms/draft, acceptance on fixture) | box | queued |
| ~05:45 | box: A2-2 determinism + shifted-workload probe | box | queued |
| ~06:30 | box: W1 PDL probe (same graph, PDL attributes on Triton launches) if W1 says it works in graphs on CUDA 13 | box | conditional |
| ≤ 10:00 | M1-T12 loop handoff (accept/commit/emit, async EOS) → first honest 6-row ms/step and tokens/step | — | after T11 lands |
| morning | blog update 4 (6-row result, R6/R1, the honest M1 projection); PROJECT.md ladder dates pulled in: M0 done 11/07 (pending A3), M1 target **mid-Nov**, M2 start on the memo | — | James awake |

## Session log (newest first)
- 03:45 re-plan v1 written; W1–W4 dispatched; 4 coders/desk running.
- 03:25 M1-T10 gate 11/11; fix-2 dispatched (spare-slot OOM).
- 02:13 R6/R1 = 2.008 measured; 01:50 T9b-4 10.52 ms; 00:45 R2 13.88; 00:05 T9a-2 17.83; 23:40 FlashMLA 19.87; 23:30 T9b-3a 23.80; 23:05 R1 26.04; 22:39 seed map 30.92.
