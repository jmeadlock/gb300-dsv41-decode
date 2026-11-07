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

## Re-plan v2 (locked 04:35 CST from W1–W4 + M2 memo + the 6-row measurements)
**Targets as the physics allows:** (1) first honest 6-row ms/step and ms/token (M1), (2) beat 7.36 ms per 6-row step (M2 ladder), (3) the 5.0 bar is a bytes bar (28.8 GB/step → 4.6 ms at measured HBM rate) → **James re-states or keeps it**; the project does not move its own goalposts.

| slot (CST) | work | box | status |
|---|---|---|---|
| 04:00– | M1-T10-PORT coder (row-generic fused glue, GEMV at R rows, dedup kernel, S=16) → gate ≤ 18 ms 6-row | after | running |
| 04:20– | box: T11 draft gate (real ckpt) + draft bench (ms/draft, acceptance) | box | running |
| ~05:15 | box: PORT GPU test + 6-row fast-path cell → real R6/R1 | box | queued |
| ~05:45 | merge a2-2; box: residency determinism + shifted-workload probe | box | queued |
| ~06:00 | **M1-T12 handoff**: loop (verify → accept by exact match → commit → draft, device-side emission, async EOS) + row-invariance prerequisite (fixed-order fp32 FFMA for mHC mixes + head, W4) → first honest 6-row ms/step + tokens/step + ms/token | — | next coder |
| ~06:30 | box probes: anatomy at 10.5; PDL probe; cuBLAS batch-invariant mode census | box | queued |
| 08:00–10:00 | M2 L1–L2 handoff (mixes branch, router+plan) from memo §6 if a coder slot is free before the reset | — | conditional |
| morning | blog update 4: 6-row results, R6/R1 story, the bytes floor and what it means for the bar (teaching voice, corrections box for the 9 GB / 2,800-kernel numbers) | — | after James |
| James | decisions: A1, A2, A3 (incl. pin GEMV M=1 config vs drift for the 6-row instrument), **the 5.0 bar**, PROJECT.md ladder dates (M0 done 11/07 pending A3; M1 mid-Nov; M2 start now) | — | — |

## Session log (newest first)
- 04:35 re-plan v2 locked; REPORT.md written. W1–W4 + M2 memo in; T11 and A2-2 landed (branches); T10-PORT coding; T11 on box.
- 04:00 6-row fast path 40.2 ms (glue unported), overflow at S=8.
- 03:38 M2 memo: 5.0 not reachable by bytes at 6 rows.
- 03:45 re-plan v1 written; W1–W4 dispatched; 4 coders/desk running.
- 03:25 M1-T10 gate 11/11; fix-2 dispatched (spare-slot OOM).
- 02:13 R6/R1 = 2.008 measured; 01:50 T9b-4 10.52 ms; 00:45 R2 13.88; 00:05 T9a-2 17.83; 23:40 FlashMLA 19.87; 23:30 T9b-3a 23.80; 23:05 R1 26.04; 22:39 seed map 30.92.
