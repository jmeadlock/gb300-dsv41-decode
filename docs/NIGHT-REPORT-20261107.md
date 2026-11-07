# NIGHT REPORT — Friday 21:30 → Saturday 08:05 CST, 2026-11-06/07

## The number
| metric | Friday 21:30 | now | production (Pg61) | bar |
|---|---|---|---|---|
| 1-row step | 32.2 ms | **10.52 ms** (95 tok/s) | — | — |
| **6-row DSpark step** (the charter's metric; instrument built tonight) | — | **22.3 ms** (verify 19.4 + draft 2.9) | 7.36 | 5.0 |
| tokens accepted / step | — | 2.76–2.97 | ~2.8 | — |
| **ms / token** | — | **7.5** | ~2.6 | — |

Everything bit-identical on replay; the 6-row verify bit-equal to the reference model; the drafter bit-equal to DeepSeek's `forward_spec`; production's acceptance rule (exact greedy match, confidence head off) reproduced with production's yield.

## What tonight established
1. **M1 is done in substance** (charter said January): verify (T10), draft (T11), loop + headline instrument (T12), ports, 6-row MoE levers, fixed-order fp32. Blog §8 is live.
2. **The 5.0 ms bar is a bytes bar.** A 6-row step moves ~29 GB (17 GB of it experts: ~22 unique per layer × 40 × 18.8 MB); 4.6 ms at the measured HBM copy rate, 6.3 ms at the best any kernel reaches here; 5.0 needs 92 % of HBM for the whole step. Production's 7.36 needs 3.9 TB/s. No fusion design reaches 5.0 with stock weights at 6 rows. **Decision for you**: keep 5.0 as written, restate as "beat 7.36 and publish ms/token", or widen scope. The blog says it is your call and stays as written until you make it.
3. Row-invariance is solved: the three sites where 6 rows ≠ 1 row were two cuBLAS fp32 GEMMs (now fixed-order Triton, proven across M) and the Engram reductions (now per-row). The last 1-vs-6-row difference is the DeepGEMM expert path (1-row vs 16-row layout, A3-class).
4. Research round 2 (W1–W4 + M2 memo): no idle gap between graph nodes (cost is inside kernels); PDL is worth ~1 ms and works in our graphs; GEMV is latency-bound (−2 ms available); k>5 drafts never; six corrections to our own record published.

## Next (after your decisions)
- Match the DG expert path at 1 and 6 rows → headline stream == 1-row greedy (bit-identical headline instrument).
- bf16 head read (exact, −1.3 GB/step); GEMV bytes-in-flight rewrite (−2 ms); A2-2 residency box probe (branch landed, unmerged).
- M2 fusion ladder per `docs/M2-FUSED-CHAIN-DESIGN.md` §6 → models ~9 ms/step; persistent per-layer stretch ~7.3.

## Open / needs you
- A1, A2, A3 amendments; **the bar**; PROJECT.md ladder dates (M0 done 11/07 pending A3, M1 done in substance, M2 start now).
- Skill patches staged for approval: `5b24a3da` (+ tonight: the CPU-generator-under-cuda-default trap cost three box windows; `git pull` over an untracked receipt cost one).
- `test_loop_real` fixture (its own 1-row build asserts in `dg_moe._shared`; the runtime's identical path passes in the headline).

## Spend
Anthropic Opus workers: ~20 coder/desk/research dispatches since 21:30 on the subscription (resets 10:00). Nous/Grok: 0 tonight. Box: ~25 serialized windows, lane dark throughout.

Receipts: `docs/HEADLINE-20261107.md`, `docs/LEDGER.md`, `research/round2-20261107/REPORT.md`; blog https://al-engr.com/gb300-dsv41-decode.html (Updated 6:59 AM CST).
