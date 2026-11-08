# Overnight plan — 2026-11-07 → 08 (James asleep; approved: MegaMoE numerics probe = yes; comparability wording = Milo's pick)
Driver: `docs/GAP-LEDGER-OURS-20261107.md` — the step is a serial sum of kernels; the largest family is cold-expert staging over C2C (4.55 ms), then expert GEMM (4.45), glue (2.50), dense GEMV (2.01).
| track | worker | tree | box gate |
|---|---|---|---|
| M3-S1 staging overlap (copy engines under hot GEMM; bit-identical by construction) | Opus | -wts1 (m3-s1) | bench → 1-row byte-identical → headline |
| M3-N1 MegaMoE numerics (drift source, bf16-before-quant, A3 on substituted layers) | Opus | -wtl3 (m2-l3) | numerics bench on /ref |
| loose ends 2 (census fix, box lease, receipt stamping, A3 definition, totals) | Sonnet | main | census re-run |
| residency on, current main (`--residency on --pool-spare 24`) | — | box | headline vs 20.30 |
Box: strictly one window at a time via `flock ~/dsv41/box.lock`. Blog: §5 comparability caveat (Pg61's cold experts on a sidecar GPU). Pg61 matched re-run: NOT tonight (box time goes to rungs; the W0 numbers stand as the Pg61 column with their caveat).
