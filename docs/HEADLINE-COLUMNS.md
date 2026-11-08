# HEADLINE-COLUMNS — the two numbers the project publishes, what each one means, how to cite them

Decision (James, 11/08): the project publishes BOTH a **headline** column and a **steady state / long session** column.
`bench/headline.py` measures both in one run. Nothing about the headline column changed; the steady-state column is a
separate block next to it.

## The two columns

| | headline | steady state |
|---|---|---|
| JSON | `headline.two_graphs` (and `headline.ms_per_6row_step_dspark_k5`, `ms_per_token`, …) | top-level `steady_state` |
| What ran | the 8 fixture prompts (`w0/results/phase1/parity-A.jsonl`) × 256 new tokens, two graphs (verify \| draft), `--reps` fast + `--parity-reps` parity runs per prompt | **the same cell again**: same prompts, same reps, same two graphs |
| Residency policy at the start | **cold**: whatever the boot map gives; the policy has seen 0 steps | **warm**: carried over from every headline cell (two_graphs, then one_graph); no reset |
| What it answers | "what does a fresh server do on this workload" (the number every downstream reader keys off) | "what does the same server do after it has been serving for a while" (long session) |
| Exists when | always | residency on (the default); or `--steady-state` forced |

Both columns use the same instrument and the same window: CUDA events per step (start \| after verify graph \| after draft
graph), live steps with index ≥ `--stats-from` (32) of each sequence, **fast reps only** pooled over prompts (parity reps
are for numerics, not timing). Both report ms/step (+ verify graph + draft graph), tokens/step, and
**ms/token = total GPU ms / total emitted tokens** (sum of the window's step times over the sum of tokens emitted in
those steps, pooled — not a mean of per-prompt ratios; both totals are stored so the quotient re-derives from the JSON).

## What "carried over, no reset" means exactly

* Run order: boot → 1-row truth → `two_graphs` cell (the headline) → `one_graph` cell → residency report →
  **steady-state pass** → (`--residency-audit`) → JSON. `--graphs 2` skips the one_graph cell; the pass then follows the
  headline cell directly. The pass is therefore always the **last** thing measured and every older number is measured
  exactly as before.
* The residency manager is not touched: policy counters, EWMA, pending swaps, hot sets and the expert tables are as the
  headline cells left them. The only boundary action is the residency report's `flush()` (publish swaps whose copies were
  already in flight), which the old flow also did at the same point.
* If the last cell was the one-graph one the loop is recaptured with two graphs (`steady_state.recaptured`); that resets
  the loop's own counters, never the manager's. Overflow counts are read before the pass for the headline run
  (`overflow`) and as a delta for the pass (`steady_state.overflow`).
* **How warm is "steady"?** It is "after `policy_steps_before` manager steps" — recorded in the block
  (`steady_state.policy_steps_before`, also `swaps.swaps_before`, `swaps.drains_before`). With the default fixture that is
  roughly 2.4 thousand live steps (479 live steps per rep × 5 reps) plus the loop's few predicated no-op steps. It is a
  property of the run, not a constant: cite the number from the block. A different `--reps`, `--graphs` or fixture
  changes it.
* Tokens must not change: the policy moves experts, not numerics. The pass's tokens/accept lengths are compared with
  the headline cell's per prompt (`steady_state.prompts.<tag>.equal_headline_cell`, `vs_headline.streams_equal_headline_cell`);
  any difference is an entry in `errors` (stage `steady_state_vs_headline`) and the exit code is 2.

## Flags

* default: **ON when residency is on** (residency is on by default since 11/07), **OFF when `--residency off`**;
* `--no-steady-state`: headline columns only (saves the extra box time);
* `--steady-state`: force the pass; with residency off it is a **control** (no policy to warm) and should equal the
  headline within run-to-run noise (the receipt's `steady_state_condition` and the block's `why` say so).

## The `steady_state` block (JSON)

Top level of the report, next to `headline`, `residency`, `mode_gap`:

* disabled: `{"enabled": false, "why": "...", "definition": "..."}`;
* enabled: `enabled`, `why`, `cell` (`two_graphs`), `definition`, `order`, `ok`, `stats_from_step`, `reps`, `parity_reps`,
  and the same fields as the headline, with the same names: `ms_per_6row_step_dspark_k5`, `ms_verify_graph`,
  `ms_draft_graph`, `tokens_per_step`, `ms_per_token`, `total_gpu_ms`, `total_emitted_tokens`, `ms_per_token_definition`,
  `ms_per_step_incl_manager` (the number to compare with a residency-off run), `cold_share` (call-weighted over the pass's
  fast reps), `window` (the full `window()` dict: percentiles, all-steps numbers, manager ms);
* `swaps`: what the manager did during the pass — `swaps` issued, `swaps_per_step`, `manager_steps`, `steps_with_swaps`,
  `issued`/`published`/`policy_swaps`, `drains`, `drain_cold_share` (first/last/mean over the pass's drains) — and what
  had accumulated before it (`policy_steps_before`, `swaps_before`, `policy_swaps_before`, `drains_before`);
* `streams` (`exact` of `prompts`: the pass's emitted stream vs the 1-row greedy), `replay_ok` (self-replay for the pass:
  fast reps == parity reps, parity records byte-identical, logits digests equal), `prompts.<tag>` (stream record incl. the
  first divergence and the 1-row margin there, replay, cold share per rep, window), `overflow`;
* `vs_headline`: headline and steady ms/step, delta and delta %, ms/token, tokens/step, cold shares, streams equal;
* `recaptured`, `capture`, `error`, `start_wall`, `end_wall`, `wall_s`.

The summary printed at the end of a run has a `steady_state (...)` line, the `ms/token = total GPU ms / total emitted
tokens = A ms / N tokens = B` line for the pass, a `vs headline` line and a streams/self-replay line.

## Receipt

`bench/receipt.py` stamps the pass into the same receipt as the headline: `steady_state_condition` (a line known at
start: `on: two_graphs again after the headline cells, residency state carried over (no reset), reps …` /
`off (<why>)` / `n/a`) and `steady_state` (filled at `finish()` from the block: ms/step, verify, draft, tok/step,
ms/token with both totals, cold share, swaps, streams exact vs 1-row, self-replay, `policy_steps_before`, delta vs
headline, wall time and start/end; `null` when the pass did not run). The receipt's printed summary ends with
`steady-state two_graphs X ms/step`. Wall time of the whole run (`receipt.wall_s`) includes the pass.

## How to cite

Always name the column. Never put a steady-state number where a headline number belongs, or the reverse.

* Headline: "**X ms per 6-row step (DSpark k=5), headline column**: two graphs, 8 prompts × 256 new tokens, steps ≥ 32,
  residency policy cold at start" — from `headline.two_graphs.ms_per_step.mean`
  (= `headline.ms_per_6row_step_dspark_k5`); ms/token from `headline.two_graphs.ms_per_token`
  (= `total_gpu_ms / total_emitted_tokens`).
* Steady state: "**Y ms per 6-row step (DSpark k=5), steady-state column**: the same cell re-run after N policy steps with
  the residency state carried over" — from `steady_state.ms_per_6row_step_dspark_k5`; ms/token from
  `steady_state.ms_per_token`; N from `steady_state.policy_steps_before`.
* Together, in one sentence: "X ms/step cold (headline), Y ms/step warm (steady state, after N steps), same prompts."
  The gap `vs_headline.delta_ms_per_step` is a measured policy-warm-up effect (cold share
  `vs_headline.headline_cold_share` → `steady_cold_share`), not a mode effect.
* Residency on vs off: compare like with like. A residency-off run has no policy to warm, so its headline column is
  also its steady state; compare off-headline with on-headline (cold start) and off-headline with on-steady-state
  (warm), and say which. Use `ms_per_step_incl_manager` on the residency side (the manager's GPU time is outside the
  step events).
* one_graph vs two_graphs: `headline.one_graph` is measured **after** the two_graphs cell at a different policy state
  (`mode_gap`); it is neither the headline nor the steady-state column. Cite the graph-mode gap only from a run with
  residency off or frozen.
* A steady-state number is only citable with its checks: `steady_state.streams` and `replay_ok`
  (same as for the headline), `ok` true, no `steady_state_vs_headline` error, `overflow.steps` 0, and the receipt (git
  commit + dirty hash, switches, clocks) of the same JSON.
* Do not average the two columns, do not pool their steps, and do not quote the pass's `all_*` fields as the column.

## Cost

One more pass of the cell: 8 prompts × (`--reps` fast + `--parity-reps` parity) runs, each dominated by the reference
prefill (about 3.9 s) plus ~100 live steps, and a two-graph recapture when the one_graph cell ran last (~2.5 s warm
JIT cache). ESTIMATE from the 11/08 s3 receipts (the 16 fast runs of the headline cell total 84 s; the parity runs are
a little slower per step): **about 3–4 minutes per headline run**. The pass's measured `wall_s` is in the block —
replace this estimate with it after the first box run. `--no-steady-state` removes the cost.

## Decode wall columns (LOOP-OVERRUN)

The headline and steady-state columns above are GPU numbers (CUDA events around the live steps). What a client waits for
is the host wall of the whole request, and the loop used to waste part of it: it learned that a sequence had finished up to
15 steps late (poll every 8 steps, lookahead 1) and issued 9-15 predicated no-op graph replays (~18.6 ms each) after every
sequence — 97 steps ≈ 1.8 s over the 8 fixture prompts (`docs/headline-20261108-main-ss.json`: per prompt
`steps_issued − steps` = 9, 12, 14, 12, 10, 12, 15, 13), a fixed 0.2–0.3 s per request. Three new numbers per cell, per
prompt and pooled, next to (never replacing) the GPU columns:

| column | JSON key (under `decode_wall`) | definition |
|---|---|---|
| decode wall ms/token **excluding overrun** | `ms_per_token_wall_excl_overrun` | Σ(decode wall − GPU ms of the overrun steps) ÷ Σ decoded tokens |
| decode wall ms/token **including overrun** | `ms_per_token_wall_incl_overrun` | Σ decode wall ÷ Σ decoded tokens (what a client waits for) |
| GPU ms/token | `ms_per_token_gpu` | Σ live step ms (CUDA events) ÷ Σ decoded tokens |
| host gap | `host_gap_ms_per_token` | excl − gpu (gate: ≤ 0.1) |
| overrun steps | `overrun_steps` (per rep), `overrun_steps_total`, `overrun_steps_max` | `steps_issued − steps` (gate: ≤ 2 on every prompt) |
| overrun cost | `overrun_ms_total`, `overrun_ms_per_token` | GPU ms of the no-op steps (their own events); incl − excl |
| TTFT | `ttft_ms` (stats), headline `ttft_ms_mean` | = `prefill_ms`: reference prefill + draft-ring seeding + first draft, synced (≈ 3.5–4.2 s; the M4 work) |
| poll cost | `poll` (`polls`, `drains`, `enqueue_us`, `wait_ms`, `poll_every`, `lookahead`, `tail_sync`), `poll_enqueue_us_per_poll` | host µs spent in the status copy + event record; ms blocked waiting |

"Decoded tokens" = tokens after the prefill's first one (`len(token_ids) − 1`), pooled total ÷ total over the fast reps
like `ms_per_token`. Where it lives: `prompts.<tag>.decode_wall` (+ `overrun_steps`), `headline.<cell>.decode_wall`,
`headline.decode_wall` (the headline cell), `headline.ttft_ms_mean`, `steady_state.decode_wall`,
`steady_state.prompts.<tag>.decode_wall`; the per-run keys from `DSparkLoop.generate` are `overrun_steps`, `overrun_ms`,
`decode_wall_ms_excl_overrun`, `poll`. All additive; every existing key is unchanged. The printed summary has one
`decode wall ms/token: excl overrun … | incl overrun … | GPU … | host gap … | overrun steps … | TTFT … | poll …` line per
cell (and for the steady-state pass) and two new table columns (`ovr` = max overrun steps over the reps, `ms/tok wall` =
incl overrun).

What changed in the loop (`runtime/dspark_loop.py`, `PollPlan`): a poll after **every** step, one poll in flight
(`--poll-every 1 --lookahead 1`): the host learns of `done` after issuing one more step, so an EOS ending costs exactly
1 no-op step; plus a **tail sync** for max_new endings (a step emits ≤ 6 tokens, so from the last observed `n_out` the host
knows when a length ending is possible and waits for the exact state before issuing the step that could be a no-op):
length endings cost 0. The old schedule is `--poll-every 8 --no-tail-sync`. Arithmetic is untouched (the device already
turns every post-`done` step into a no-op; the host only picks when to stop issuing).

Expected (CPU simulation, real `LoopBuffers`, the fixture's own accept plans; `tests/test_dspark_loop_cpu.py
test_fixture_overrun_reproduced_and_fixed`): the old schedule reproduces the measured `steps_issued` of all 8 prompts
exactly (97 overrun steps); the new default gives **4** (one each for code, json, tool, story — the EOS endings; 0 for
prose, math, shell, long — the max_new endings), ≈ 1.7 s less wall over the fixture, ≈ 0.2 s per short request. A
length-ending sequence pays 1–4 tail drains (each a short GPU bubble: the host waits for the poll, wakes, launches).

Not measured here (no CUDA on the Mac): the host cost of a poll and the per-step bubble of waiting synchronously. The loop
records the first per run (`poll.enqueue_us`, `poll.wait_ms`); `--poll-probe` measures both without a model (below).
The trade it sizes: lookahead 0 (wait for the poll right after each step) would give overrun 0 for EOS endings too, at a
GPU bubble per step; it pays only while `steps × bubble < one step (18.6 ms)` — the probe reports the break-even step
count. A larger `--lookahead` (2) buys host-jitter slack at the price of overrun 2 (still within the gate).

### Box commands (orchestrator; after the merge, on main)

```
# 1. poll probe (no model, ~1 min; GPU lease as every box script): host µs per poll, per-step bubble per schedule
$D bench/headline.py --poll-probe --probe-step-ms 18.6 --json scratch/poll-probe.json

# 2. headline on main, new default schedule (also the steady-state pass), then the old schedule once for the A/B
$D bench/headline.py $H --json scratch/headline-lo-new.json
$D bench/headline.py $H --poll-every 8 --no-tail-sync --no-steady-state --json scratch/headline-lo-old.json
```
with `$D` = the docker-run prefix of `docs/A2-2-BOX-PLAN.md` (`bench/box/lease.sh` first; `--entrypoint python3 vllm/vllm-openai:v0.31.0`) and `$H` = the 11/08 main headline line (`--model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused
--attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency-drain 64 --max-new 256`). Gate (read
from `headline.decode_wall.gate` and the summary line): tokens/streams and accept plans byte-identical between the two runs
(`streams` 8/8, `replay_ok` 8, and per prompt `tokens_sha256` / `accepts` equal old vs new); `overrun_steps_max` ≤ 2 (expected
1); `host_gap_ms_per_token` ≤ 0.1; GPU ms/step unchanged within noise (the poll copy sits outside the step events; the residency
manager counts the no-op steps in `host_steps`, so with ~90 fewer of them per pass `policy_steps_before` and the drain
cadence shift slightly — cite them from the block, expect streams identical and ms/step within the usual cold-share noise).

## Device-side early stop: feasibility (not implemented)

A CUDA conditional graph node (IF, CUDA ≥ 12.4; the box has CUDA 13) around the step body, its condition set by a
one-thread kernel from `status[1]` (`done`), would make a post-`done` replay cost ~µs instead of ~18 ms. Byte-identical for
the stream (a no-op step already changes nothing that reaches it, `test_async_poll_streams_identical_…`). Not done because
(1) `torch.cuda.CUDAGraph` has no conditional-node capture that we know of (unverified on torch 2.13): it needs the driver
API through cuda-python and our own capture of the body graph (the current captures share torch's graph pool), (2) none of our
captured kernels (DeepGEMM, FlashMLA, TileLang, Triton, PDL edges) has been tried inside a conditional body — the driver
restricts what a conditional body may contain — so it needs a box probe first, (3) what it buys beyond the host poll is
≤ 1 step (≈ 18 ms) per request. Revisit only if the probe shows the host cannot keep lookahead 1 without GPU bubbles
(then lookahead would have to grow and the conditional node would pay for the extra overrun). Details: the
`runtime/dspark_loop.py` module docstring.
