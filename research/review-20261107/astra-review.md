REQUEST CHANGES on the plan; APPROVE the day’s work.

The correctness recovery is the strongest result: 2/8 → 8/8 agreement with your own greedy path, while roughly halving the original step time. Rejecting fast-looking regressions is also good work. But 20.30 ms remains far from 7.36; the evidence does not yet establish that launch/fixed costs explain that gap.

Scope: I reviewed the brief, not the code or JSON receipts. Recommendations below are experiments, not independently verified findings.

1. The one change I insist on

Before committing to the persistent-kernel end-game, build a matched-workload gap ledger against Pg61.

Use identical token prefixes, positions, verify/draft shapes, and controlled expert-residency conditions. Record separately:

  Verify, draft, and complete-step GPU time.
  End-to-end elapsed time per emitted token.
  MoE, dense projections, attention, glue, and unattributed time.
  Actual graph node/edge counts and measured overlap.

Production’s own nondeterminism does not prevent feeding both runtimes identical prefixes for a performance experiment.

You already know Pg61 uses a materially different MoE implementation. You do not yet know how much of its advantage comes from that, other kernels, residency, or scheduling. “Our persistent layer models at 7.3” is not a substitute for locating the gap—and offers essentially no margin against 7.36.

Revised order:

  DO NOW: gap ledger; bounded PDL integration; draft-only L1 experiment.
  NEXT: selective folds ranked by measured critical-path savings.
  PARALLEL, SMALL: MegaMoE numerical/layout feasibility probe.
  LATER: persistent sublayer prototype, conditional on the ledger.

2. “Launches, not bytes”: too strong

The GEMV-v3 result rejects that implementation as a sufficient solution. It does not uniquely identify ramp/tail as the cause, nor rule out memory-system limits.

Three different costs are being conflated:

  Host/graph dispatch.
  Device scheduling and insufficient overlap between kernels.
  Useful execution limited by memory traffic, dependencies, or utilization.

PDL mostly attacks the second. Fusion can affect all three. Node count alone predicts neither.

Your bytes estimate is a useful bound, not a decomposition. A copy benchmark is not the achieved bandwidth of quantized expert reads, scale loads, gathers, and Grace/UVA accesses. Separate HBM traffic from C2C traffic and account for whether they overlap. “Above the bytes floor” does not mean “the remainder is launches.”

Also: hot layer-0 MegaMoE versus a routed path involving cold staging is not enough to extrapolate a full-model saving. Match residency, expert count, and cache conditions.

PDL: worth testing; 700 × 5 µs is not yet a credible forecast.

The reported probe gain is substantial. But:

  Count eligible edges on the actual captured critical path, not merely owned→owned edges.
  Borrowed kernels interrupt opportunities and can change overlap.
  Prefetch can evict expert tiles or contend for HBM/C2C bandwidth.
  Savings along a chain are not necessarily additive after integration.
  A repeated probe can have very different cache behavior from the full model.

Use three integrated variants: off, PDL-only, PDL plus selective prefetch. Add edge families incrementally. Do not start with “all owned kernels.”

Audit dependency correctness separately: early launch permission is not proof that consumer-visible outputs are ready. Verify waits, writes, mixed-kernel boundaries, and replay state. Hundreds of identical replays are evidence, not a concurrency proof.

3. L1: test draft-only first

The 1-row win does not establish a 5-row draft win. Measure the actual draft graph.

My order:

  A. Verify unchanged; fork only the draft graph.
  B. Fork only selected low-contention boundaries.
  C. Reduce branch CTA count/residency.
  D. Stream priority as a secondary experiment.

Priority is not reserved SM capacity; it cannot guarantee the interference disappears.

Measure complete-step time and branch completion, not just the mixes’ duration. A delayed branch can move the stall downstream. If the draft-only variant wins and preserves exactness, it is a legitimate shape-specific implementation—not a compromise.

If B/C cannot recover a verify win, park verify-side L1.

4. MegaMoE: neither disqualify nor fund option A yet

No: 8× farther from fp64 is not automatically disqualifying under an A3 acceptance contract.

Yes: it is a strong warning. Quantization changes deserve more scrutiny than an alternative reduction tree. But oracle distance is a diagnostic; it is not the approved output-level acceptance criterion.

Separate:

  Engineering gate: layout, scales, routing, accumulation, and quantization are implemented as intended.
  Numerical diagnostic: relative-L2 and outliers across representative layers.
  Acceptance gate: the explicitly defined A3 test against production.

If the mini-checkpoint oracle gate was an agreed hard requirement, changing it requires an explicit decision. Do not quietly replace it with A3 after seeing a faster kernel.

Before any pool rewrite:

  Make one layer numerically understood.
  Test the proposed bf16-before-quantization variant.
  Measure representative routed layers and residency states.
  Use bounded scratch storage to assess downstream logits on fixed prefixes.

Cheaper layout routes worth checking:

  Pack directly into the final layout during a fresh load.
  Convert checkpoint shards offline, preserving weight and scale bits.
  Use a separate experimental loader rather than a live in-place migration.
  Investigate compatible strided views for existing readers.

An interleaved allocation may express each projection as a strided view; that does not mean existing kernels accept it. Audit contiguous-layout assumptions.

Avoid the 137 GiB duplicate allocation. But distinguish “building the layout is expensive” from “maintaining every existing reader is expensive.” A fresh-load experiment can answer the former without solving the latter.

5. Persistent layer: plausible research, premature end-game

A persistent per-layer kernel is not automatically the right boundary.

The layer includes routing, cross-CTA reductions, expert dispatch, attention, and heterogeneous memory access. Removing boundaries can introduce global synchronization, poor occupancy, load imbalance, or repeated reads. Preserving fixed reduction order adds constraints.

Start with the largest profitable producer→consumer island. A persistent dense-projection/glue region or MoE dispatch/execution region may be a better target than a whole layer.

Other levers remain:

  Better expert-path kernels and residency.
  Weight reuse across verification rows.
  Fewer intermediate materializations.
  Shape-specific draft execution.
  Avoiding work on rows/positions that provably need none.
  Speculation policy chosen for elapsed time per emitted token.

“No acceptance lever left” is too broad. Exact-match acceptance fixes the correctness rule, not the optimal proposal policy. A failed k>5 experiment does not rule out smaller/adaptive k or different proposal trees.

However, changing the tree changes the workload. It cannot honestly count as beating the original fixed 6-row step benchmark. Keep that benchmark, and report alternative decoding policies separately in end-to-end ms/token.

Also, 2.9 versus 2.8 tokens/step provides a small efficiency allowance, not zero allowance. It cannot rescue a 20 ms step, but “only via step time” is unnecessarily absolute.

6. NEVER column

  Never infer saved milliseconds from removed launches alone.
  Never extrapolate a hot-cache microbenchmark across the full model without a matched test.
  Never adopt all-kernel PDL before auditing dependency semantics.
  Never soften numerical gates silently because a kernel is fast.
  Never fund the full pool rewrite before numerical feasibility.
  Never replace proven kernels with a monolith on a modeled 7.3 ms promise.
  Never use a changed speculation shape to claim the fixed-shape step target.
  Never present shared-prefix parity as full continuation equivalence.
  Never benchmark competing workers concurrently on the box.

7. Measurement and orchestration changes

The worktree pattern is right. Enforce isolation mechanically.

Each receipt should identify commit, dirty-diff hash, loaded extension hashes, configuration, checkpoint, fixture, driver/runtime, clocks/power settings, and residency state. Separate worktrees do not isolate compiled-extension caches automatically.

Acquire a machine-wide lease before build/load/run; fail closed if occupied or stale. Snapshot the executable tree before launch. Have one independently owned acceptance harness—not worker-specific definitions of success.

Four reporting corrections matter:

  A. The compressor’s +0.15 ms is a net observed change, not yet its isolated cost. Tokens/step changed too, so trajectories and routed work may differ. Use frozen-prefix A/B/A trials for attribution.

  B. Repeated same-config runs within 0.02 ms establish repeatability under those conditions, not universal uncertainty. Interleave baseline/candidate trials.

  C. State aggregation explicitly. Publish total elapsed time, emitted tokens, steps, and the weighting behind headline means. Aggregate ms/token should be total time divided by total emitted tokens, not an unlabeled average of prompt ratios.

  D. Define A3 precisely. “0.000 margin-weighted flips” needs raw disagreement counts, margins, denominator, prefix lengths, and rounding. Shared-prefix tests can lose coverage exactly where trajectories diverge. Add teacher-forced fixed prefixes and held-out continuations.

Finally, static nsys timings are diagnostic—not an additive accounting of the captured graph. Your attribution must acknowledge that limitation.

Bottom line: keep the ladder, but replace the launch-count narrative with a matched critical-path ledger. PDL deserves the next experiment. Persistence has not yet earned the next major investment.
EXIT=0
