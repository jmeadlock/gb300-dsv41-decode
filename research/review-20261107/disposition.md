# Astra review (Nous `openai/gpt-6-astra`, 2026-11-07 18:05 CST) — disposition

Verdict as given: **REQUEST CHANGES on the plan; APPROVE the day's work.** Reviewer saw the brief only (not code/receipts); its items are experiments, not verified findings. Cost $0.24. Hashes in `hashes.sha256`.

| # | Reviewer finding | Disposition | Action |
|---|---|---|---|
| 1 | Build a matched-workload gap ledger vs Pg61 (same prefixes/positions/shapes/residency; verify/draft/step GPU time, ms per emitted token, MoE/dense/attn/glue/unattributed, node+edge counts, measured overlap) before committing to a persistent kernel | **ACCEPT** | DO NOW: desk+box task. We have Pg61's nsys step (`w0/results/phase1/step-pg61.json`) and our static anatomy, not a matched decomposition. |
| 2 | "Launches, not bytes" is too strong: GEMV-v3 rejects one implementation, doesn't identify ramp/tail; three costs conflated (dispatch / device scheduling+overlap / memory-limited execution); bytes estimate is a bound, not a decomposition; separate HBM vs C2C | **ACCEPT WITH MOD** | Reword in ledger/blog: "above the bytes floor; per-kernel fixed costs dominate per the anatomy" and add the decomposition to the gap ledger. The PDL probe does directly measure the scheduling term, so the lever stands. |
| 3 | PDL: count eligible edges on the captured critical path; integrate as off / PDL-only / selective prefetch, edge families incrementally; audit dependency semantics (early-launch ≠ outputs ready); replays are evidence not proof | **ACCEPT** | Already the handoff's order (1)→(2) with the loads-before-wait rule; add: edge census on the real captured graph, families incrementally, three box variants. |
| 4 | L1: test fork on the draft graph only (A), selected boundaries (B), branch CTA count (C), priority last (D); priority ≠ reserved SMs | **ACCEPT** | Priority run already in flight (result pending; treat as secondary). Next: draft-only fork variant. |
| 5 | MegaMoE: 8× oracle distance is not disqualifying under A3; separate engineering gate / numerical diagnostic / acceptance gate; don't quietly swap gates; test bf16-before-quant variant; fresh-load experimental loader instead of in-place migration; audit strided-view assumptions | **ACCEPT WITH MOD** | Right: the oracle gate was MY operationalization in the handoff, not the approved A3 (which is vs production). Keep "no option A now" but allow a small, bounded probe: bf16-before-quant variant + representative layers + downstream-logit check on fixed prefixes, and a fresh-load layout experiment on a subset — LATER/small, not a pool rewrite. |
| 6 | Persistent per-layer kernel is premature; start with the largest profitable producer→consumer island (dense-projection/glue region, MoE dispatch/execution region) | **ACCEPT** | Plan order changed: islands before whole-layer. |
| 7 | "No acceptance lever left" too broad: adaptive/smaller k or other proposal trees exist — but a changed tree is a different workload: keep the fixed 6-row benchmark, report alternative policies separately in end-to-end ms/token | **ACCEPT WITH MOD** | Agreed on the reporting rule. k>5 and block-5 drafter were measured; adaptive k not measured → LATER research item, reported separately. |
| 8 | NEVER column (9 items) | **ACCEPT** | Recorded below; "never benchmark competing workers concurrently" was today's lesson. |
| 9 | Process: receipts carry commit + dirty-diff hash + extension cache hashes + config + checkpoint + fixture + driver + clocks + residency; machine-wide lease (fail closed); snapshot tree before launch; one independently owned acceptance harness | **ACCEPT** | DO NOW: box lease (flock) in `bench/box/`; receipt stamping in headline/first_light; acceptance = `bench/headline.py` + parity table owned by the orchestrator, not per-worker gates. |
| 10A | Compressor "+0.15 ms" is a net observed change (tok/step changed too), not an isolated cost → frozen-prefix A/B/A | **ACCEPT** | Ledger wording corrected; attribution via teacher-forced A/B/A pending. |
| 10B | 0.02 ms same-config spread = repeatability under those conditions; interleave baseline/candidate | **ACCEPT** | Box script will interleave A/B when comparing. |
| 10C | State aggregation: publish total time, emitted tokens, steps; ms/token = total time / total tokens, not mean of ratios | **ACCEPT** | Check `bench/headline.py` pooling (it reports "window steps ≥ 32 … pooled"); publish totals explicitly. |
| 10D | Define A3 precisely: raw disagreement counts, margins, denominator, prefix lengths, rounding; add teacher-forced fixed prefixes + held-out continuations | **ACCEPT** | Write `docs/A3-DEFINITION.md` from the parity code; add the teacher-forced arm. |
| — | nsys static timings are diagnostic, not an additive accounting of the captured graph | **ACCEPT** | Already labeled [M static]; keep the caveat in the anatomy docs. |

Reviewer factual errors: none found. Qualifier restored: the reviewer's "hot layer-0 MegaMoE vs routed path with cold staging" — the spike compared both hot on layer 0; the full-model extrapolation caveat still stands.

## NEVER (reviewer, accepted)
Never infer ms from removed launches alone · never extrapolate a hot microbenchmark across the model without a matched test · never adopt all-kernel PDL before auditing dependency semantics · never soften numerical gates silently because a kernel is fast · never fund the pool rewrite before numerical feasibility · never replace proven kernels with a monolith on a modeled 7.3 ms · never use a changed speculation shape to claim the fixed-shape target · never present shared-prefix parity as full continuation equivalence · never benchmark competing workers concurrently on the box.

## Revised order (reviewer's, adopted pending James)
DO NOW: gap ledger vs Pg61 (matched workload); bounded PDL integration (off / PDL-only / selective prefetch, census first); L1 draft-only fork; box lease + receipt stamping; A3 definition doc.
NEXT: folds ranked by measured critical-path savings (L6/L8/L9, L10 once fixed).
PARALLEL, SMALL: MegaMoE numerics probe (bf16-before-quant) + fresh-load layout experiment on a subset.
LATER: persistent *island* prototype, conditional on the ledger.
