# A2 — Cold-expert residency for recipe A (design memo)

Desk worker memo, 2026-11-06 ~22:45 CST. Inputs: the handoff and every file it names, m3 (`~/hermes/milo-moe-manager/`), and three more: `peer3_stats.py` and `t5_unique_vs_k.py` (field definitions only), plus `docs/STEP-ANATOMY-T9B1-20261106.md`, which landed mid-memo. No box work. Labels: **[M]** measured, **[E]** my estimate (method stated), **[A]** assumption to measure. §2's estimates are numpy over count snapshots already on disk (`~/.hermes/profiles/milo/cache/scratch/a2/est.py`).

## 0. Corrections to the inputs

1. **An expert is 18.8 MB, not 17.5 MB.** 18,800,640 B = 17.93 MiB (m3 golden (c), `README.md:113`; cold `HostBuffer` 62.334 GiB / 3,560 experts). Per expert over C2C that is **82 µs at 230 GB/s and 48 µs at 390**. The probe measured 52.5 µs per expert on the copy engine (`C2C-PROBE-20261106.md:9`).
2. **peer3_stats ran under Garner's `rowmap-mix-v1-h255.json`: 255 hot / 129 cold, frequency-built** (Garner `docs/ARCHITECTURE.md:36-37`; `dsv41-garner-port-20261029/PLAN.md:41`). It was not a 295/89 map.
3. **At T = 6 the file says 1.60 distinct cold experts per layer-call.** 2.13 is the mean over all T. So the C2C memo's "~80 cold reads ≈ 3.9 ms per 6-row step" (`C2C-PROBE:18`) should read **~64 reads ≈ 1.2 GB ≈ 3.1 ms at 390 GB/s, under a 129-cold map**. The conclusion stands.
4. **"At any static map ≥ 2.75 ms/step at 1 row"** (`T8-T9A-DESIGN.md:184`) is true of first-295 only. A frequency seed gets 1 row to about 0.1 ms (§2).

## 1. What first-295 costs today and at 6 rows

### 1.1 One row

| quantity | value | source |
|---|---|---|
| cold calls/step | **56.9 mean** (p50 57, p90 64, range 37–76) of 240 | `first-light{,-graph,-dg,-gemv}-20261106.json` `step_cell.cold`, n = 512 each [M] |
| cold share | 0.233–0.237 per cell, 0.233–0.239 per prompt | same [M] |
| a usage-blind map would give | 89/384 = 0.232 | arithmetic |
| `t8_gather_rows` in the 32 ms step | **4.20 ms/step**, 240 launches, 17.5 µs each | `STEP-ANATOMY-T9B1-20261106.md` (nsys, static-eager = graphed kernels) [M] |

First-295 carries **no usage information**.

**How the gather spends its time.** Per layer the step sorts the 6 ids, then gathers every selected expert's 6 tensors through a device pointer table into `[6, N, K/2]` staging:
- committed tree: `StaticMoE._gather` → 6 × `t8_gather_rows`;
- in-flight T9b-2: one `t9b2_gather_tables` over `ptab6 [6, E]`;
- each program copies 8 KB from `ptab[ids[j]]`, which is either an HBM slot or a UVA address in pinned Grace.

Two flows share the measured 4.20 ms:
- **hot staging:** 6 × 18.8 MB = 113 MB/layer copied HBM→HBM, so 4.5 GB/step;
- **cold reads:** 57 × 18.8 MB = 1.07 GB/step over C2C.

A launch ends when its last program ends, and cold programs are link-bound. Model: t_launch(c) ≈ max(6·b/B_hbm, c·b/B_c2c) + tail. The marginal cost of one cold expert is then about b/B_c2c − b/B_hbm = **42–76 µs [E]**.

**What the 4.20 ms constrains [E].** If the two flows add, the cold share lies between ~0 (cold fully hidden under a slow hot copy at the anatomy's ~1.1 TB/s) and **≈ 2.75 ms** (link-bound at 390 GB/s, hot at ~3.1 TB/s). 230 GB/s is excluded: it would need 4.7 ms for the cold reads alone.

This matters for tonight's plan. The anatomy's lever (a), "gather at HBM rate → 1.5 ms", assumes all 4.2 ms is copy inefficiency. Under first-295, up to 2.75 ms of it is C2C and no kernel tuning can remove it. Only the map can, so A2-1 and lever (a) belong in the same coding round.

The A2-1 measurement separates the two flows:
- (i) t_launch(c) for c = 0…6 with real pool addresses, at 8/32/64 KB per program;
- (ii) gather ms and step ms, first-295 vs seed. Both maps give the same bits (§3.4), so the difference is exactly the price of the cold reads.

### 1.2 Six rows (DSpark k = 5)

| quantity | value | source |
|---|---|---|
| distinct experts / layer / step, T = 6 | **22.45** (22.1–23.4 across classes); a lower bound, since rejected drafts route too | `unique_experts_vs_k.json` ALL/"5", 576 windows [M] |
| positional (first-N, 93 cold) share of that set | 0.246 | same, `grace_share_positional` [M] |
| first-295 cold / layer / step | **≈ 5.2** (22.45 × 89/384); 7.5 under m3's independence visit model on Card M2 counts | [E] |
| first-295 cold bytes / step | 3.9–5.6 GB → **10–14 ms at 390 GB/s** | [E] |
| usage-chosen cold set of the same size | 0.014 → 0.31 / layer → 0.24 GB → 0.6 ms | same file [M], in-sample (optimistic) |
| Pg61 h255, T = 6 | **1.60** distinct cold / layer-call; 73.4 % of calls touch ≥ 1 cold; 2.12 cold pairs → 64 / step → **3.1 ms** | `peer3_stats.json` `by_T["6"]`, 14,760 calls [M] |

At 6 rows first-295 is 2–3× the whole 5.0 ms bar in cold bytes alone.

Separately, "stage everything" grows ×3.7 at 6 rows: 16.6 GB/step, 5–15 ms at 3.1–1.1 TB/s [E; the anatomy says ~15 ms]. M1 therefore needs hot experts read in place, through a contiguous GEMM over `[H+S]` or MegaMoE over `pool[0:H+S]` (`T8-T9A-DESIGN.md:139, 147, 157-171`), with only cold experts staged. **This memo assumes that path.** With it in place, residency decides only which bytes cross C2C.

### 1.3 peer3 fields (`peer3_stats.py` header, `fold()`)

One record per peer-tier layer-call. `tag` encodes the layer and T = rows in the call.

| field | meaning |
|---|---|
| `window_s`, `calls`, `calls_with_cold`, `lost_records` | window; layer-calls (22,200 = 555 × 40); calls with ≥ 1 cold row; ring overruns (0) |
| `cold_frac` | share of calls with any cold work |
| `rows` / `pairs` | rows with ≥ 1 cold pick / (row, cold expert) pairs, mean per call |
| `cold_exp_mean`, `_p50`, `_p90`, `cold_exp_hist` | distinct cold experts per call (histogram index = count) |
| `hot_exp_mean`, `hot_unknown` | distinct hot experts; known for only 47 of 555 calls per layer, so unusable |
| `busy_us_*` | RTX busy time per call that has cold work |
| `gb_wait_us_*`, `gb_wait_over_5us` | GB300's exposed wait for the previous call's acknowledgement |
| `layers[l]` vs `by_T[T]` | folded per layer over all T, vs per T over all layers |

Only `by_T` gives per-step numbers. The per-layer means mix in T = 12–64 calls with 26–124 cold pairs each. In this window T = 6 is 66 % of calls and T = 8 is 22 %.

## 2. Seed map

**Reuse m3's v26 seed, `rowmap-mixed-ar-v1.json`.**
- Identity: sha256 `9b55d83a…`, the same file `out/emit-v26/residency.proposed.yaml:9` pins; copy in `cards/M2/m3-hook/`.
- Shape: 295 hot / 89 cold.
- Provenance: m3's row-stable refit of static-v1 on 128.7 M decode selections on the 1017 checkpoint. Golden (b) is 40/40 row-identical (`README.md:112`).
- Quality: 0.88 % cold in-sample. Static-v1, built on September counts, scores 18.2 % on the same data.

For held-out numbers I scored it on Card M2 run 2's per-(layer, expert) counts (`cards/M2/results-run2/arm-*/snap/counts-post-*.npz`). Same checkpoint, a different boot and day, 16 snapshots 18 s apart. Routing does not depend on placement, so these counts are a clean trace.

**Table 2** (training = run1 counts; scored on arm C; arms A and A2 within ±0.001):

| map | pick share, frozen phase¹ | pick share, shift phase² | cold/step, 1 row (shift … frozen) | cold/layer, 6 rows³ (shift … frozen) | 6-row cold bytes/step → at 390 GB/s |
|---|---|---|---|---|---|
| first-295 | 0.245 | 0.249 | 57 [M, our fixture] | 5.2–7.5 | 3.9–5.6 GB → 10–14 ms |
| h255 (Pg61) | 0.117 | 0.278⁴ | — | (1.60 measured, Pg61 fixture) | 1.2 GB → 3.1 ms |
| **seed v26ar, frozen** | **0.046–0.047** | **0.0045–0.0088** per interval | **1.1–2.1 … 11** | **0.16–0.31 … 1.6** | 0.12–0.23 … 1.2 GB → **0.3–0.6 … 3.1 ms** |
| seed + v26 policy (map at snapshot⁵) | — | 0.0030–0.0061 | 0.7–1.5 | 0.11–0.22 | 0.08–0.17 GB → 0.2–0.4 ms |
| top-327 from run1 (+32 slots/layer) | 0.020 | 0.0022 | 0.5 … 4.8 | 0.08 … 0.72 | 0.06 … 0.54 GB → **0.15 … 1.4 ms** |
| top-340 from run1 (+45 slots/layer) | 0.014 | 0.0012 | 0.3 … 3.3 | 0.04 … 0.48 | 0.03 … 0.36 GB → 0.08 … 0.9 ms |

¹ Card M2 frozen phase (parity, C1/C8 knee, agent fixture, gauntlet), boot → snapshot 01. The counter counts calls with T ≤ 256 (`BRIEF-M2.md:24`), so short prefills may be included; unverified.
² JP/DE → agent → JP/DE, deltas between snapshots.
³ m3's visit model, κ = 1, `v = 1 − (1 − π)^M` (`m3/cost.py:5`): an upper bound; ≈ pick share × 36 for rare experts.
⁴ Near a random 129-cold map (0.336) on this traffic, yet good on Pg61's own fixture: a static map is only as good as its calibration traffic.
⁵ Optimistic: the map at the end of an interval is scored on that same interval.

**Expected cold share with the seed:**
- 1 row: **0.5–0.9 % in domain, ~5 % on unfamiliar traffic**, against 23.5 % today.
- 6 rows: 0.16–1.6 cold experts per layer per step.

The seed fixes the in-domain case. Under a shift its frozen-phase row is as bad as Pg61's h255; that gap is the case for adaptivity.

**Cheapest win: boot a bigger hot set.**
- Headroom after the current placement: 249.81 − 218.61 = 31.2 GiB = 1,782 slots (44.5/layer).
- KV at 32K is only ~0.15 GB (`LEDGER.md` 14:17).
- Not yet measured: what capture adds (graph pool, workspaces, staging); the JSONs record only boot `cuda_allocated`.
- Planning reserve [A]: 4.2 GiB of MegaMoE cold staging (`T8-T9A:170`) plus ~4 GiB of workspaces. That leaves ~23 GiB → **H = 327 (+32/layer)**.
- Keep H uniform. MegaMoE's `num_experts` is a template constant (`T8-T9A:147`), and the literature puts non-uniform budgets at ≤ 3 % (ERM `REPORT.md:13`).

**Format and tool.**
- `HotMap.load` takes `{"<layer>": [ids]}` or `{"hot": {...}}` (`placement.py:111-116`). m3's rowmap-v0 is `{"layers": {"<l>": {"hot", "cold"}}}`.
- The conversion is `{l: v["hot"] for l, v in d["layers"].items()}`. A2-1 adds it as a 3-line branch in `HotMap.load` that logs the sha256. Row order does not matter because `HotMap` sorts ids.
- To build H = 327: `m3 plan` with `examples/dsv41-seed-mixed-ar-v1.yaml`, `banks.H.allowed_sizes: [327]` and no parent (new geometry), then `m3 export --format rowmap-v0`.
- Our own trace: dump `cnt` int64 `[40, 384]` (+ `host_steps`) to `.npz`. `m3 trace import` detects it as v26ar (`m3/trace.py:9-12`).

**m3: what we reuse, what we don't.**
- Engine-agnostic, pure numpy: geometry, trace, cost model, solver, `residency-map/v1`/rowmap-v0, and **`m3.control.policy`**. The policy is v26ar.2's decision core, 93,400 lockstep checks with 0 diffs (`README.md:122`).
- vLLM/FlashInfer-bound: all actuators (`flashinfer_split`, `adaptive.py`, `megamoe_peer`, `marlin_expertmap`), `shim/`, `emit/`, the gate parsers, the cost constants (fit on vLLM lanes), the "rows" bank layout, and the **1.9–2.4 ms per swap**, which is vLLM step structure, not physics (`REPORT.md:15`).

## 3. Adaptive residency in our runtime

### 3.1 Mechanism: a byte copy plus a pointer-table write; no kernel change

- **Pools.**
  - HBM: per layer, H hot slots plus r spares [A: r = 4]; `expert_pool.py`'s per-layer `[H, N, K/2]` pools become `[H + r, …]` at boot.
  - Grace: today's cold pool plus a demotion pool of D slots per layer, registered at boot [A: D = 64 → 45 GiB]. Hugepage registration ran 0.66 s for 251 GiB (`LEDGER` 14:15), and 197 GiB stays free after pinning (`t2-boot`).
  - A full Grace mirror does not fit: 189 + 269 = 458 GiB against 449.5 available.
- **Promote e_in:**
  - H2D into a spare slot on a copy stream: 6 `cudaMemcpyAsync`, ≈ 52.5 µs/expert [M], then record an event.
  - Between two replays, on the compute stream: `wait(event)`, then write `ptab6[:, e_in]` (48 B), `loc` and `cold_tab`.
- **Demote e_out:**
  - an expert promoted at runtime still has its Grace copy, so demotion is a pointer write;
  - a boot-hot expert first needs one D2H into the demotion pool. C2C D2H rate unmeasured [A] → measured in A2-1.
- **Reclaim:** an HBM slot rejoins the spare list only after an event behind the first replay that no longer references it. Replays never stall.
- Each pool is one `cudaHostRegister` chunk, so no copy crosses a chunk boundary (`LEDGER` 14:11).

### 3.2 Policy: m3's controller, host side, every `drain_n` steps

- **Counter.** `cnt` int32 `[40, 384]`, incremented by program `(j, 0, 0)` of `t9b2_gather_tables` (which already stores `ids[j]`) with `tl.atomic_add`.
  - v26ar's equivalent in-kernel tracer cost 2–6 µs/step (ERM `W5-system-design.md:34`) [M, vLLM].
  - The atomics touch counters only, never data (decision 14).
- **Trace ring.** `ids_ring[pos % R] ← StaticStep.ids_step`, one launch per step; 7.9 MB at R = 4096. Every routed id is logged at no extra kernel cost.
- **Tick.**
  1. Async D2H of `cnt` (61 KB) into pinned memory.
  2. Vendored, sha-pinned `m3.control.policy`: `drain` → `queue` → `apply_pending`.
  3. Issue the §3.1 copies.

  The host is idle during a replay, so the decode stream never waits on this.
- **Start values** = shipped v26 (`residency.proposed.yaml`): drain_n 64, swaps_per_drain 2, halflife 500, hyst 1.5, min_abs 8, swaps_per_barrier 8.
- **Why we can be more aggressive than vLLM.** A swap here is ~52 µs of copy, concurrent with compute (overlap probe: GEMM 90.6 → 90.6 µs [M]), against vLLM's 1.9–2.4 ms. The economic floor S/(Δc·H) in `Δv > max(z√(2v/τ), S/(Δc·H))` (ERM `W3:169`) therefore shrinks ~40×, and the statistical floor binds. A2-2 tests a 3-point grid around v26.
- **Budget.** The cold SM reads share the link with the copies. Cap at ≤ 4 swaps/step ≈ 0.2 ms of link per 5 ms step [A]. v26's average is ≤ 1.25/step.

**Device-side policy: rejected.** The copy engine cannot take a data-dependent source inside a graph (`T8-T9A:81`). SM copy kernels and a device free list would cost SM time to buy nothing while the host sits idle.

### 3.3 What the graph tolerates

`ptab6`, `ptabs`, `loc` and `cold_tab` are device tensors that the graph reads by address. Rewriting their **contents** between replays, stream-ordered, is legal; it is exactly how `tok` and `pos` are fed (`StaticStep.set_inputs`).

Two things are illegal:
- reallocating any of them;
- updating one copy of a table while a kernel reads another. `ptab6` stacks the six `ptabs`, so one owner object must update every table some kernel actually reads.

### 3.4 Determinism: proof obligation and tests

**Claim:** for every map M and every swap schedule σ, the logits are bitwise equal to those of first-295 with no swaps. It rests on four properties:
- **P1 bytes.** Every address a pointer can name holds the checkpoint bytes: copies of immutable weights, published only after their event, slots reused only after their last reader.
- **P2 no tier-dependent arithmetic.** Staging slot j is the j-th ascending id, and the GEMMs and the fp32 combine run in slot order. For the in-place paths (local index = slot, cold → H + j) the combine must reduce in id or top-k order, never in slot order. **Open check:** MegaMoE's combine order (`sm100_fp8_fp4_mega_moe.cuh`).
- **P3** Counters feed only the policy.
- **P4** The policy is deterministic. Copy timing is irrelevant, by P1.

**Tests.**
- **T-R1:** the 8-prompt fixture under first-295, seed, top-327, and seed with forced random swaps every step (including experts the same step routes to) → fp32 logit digests and token streams `torch.equal`. This extends A3's "tiered == all-HBM".
- **T-R2:** pool audit. sha256 every HBM slot and demotion slot against the shard bytes.
- **T-R3:** invariants at every tick: one published home per expert, no shared slot, spares ∩ published = ∅ (m3's I1–I2).
- **T-R4:** mutations. Publish before the event, corrupt one promoted byte, update `ptabs` but not `ptab6`. T-R1 or T-R2 must catch each one.

## 4. Recipe A thesis check

**What exists at M0.** The gather already self-fetches cold experts from Grace through UVA pointers inside the graph (`static_kernels.py` docstring): recipe A in its primitive form. It has two defects:
- it stages the hot experts too (§1.2);
- the cold fetch is stream-ordered before the GEMM, so it is fully exposed (`T8-T9A:149`).

**What M2 adds, defensibly.**
- **(a) Hot in place, cold staged only:** contiguous or MegaMoE over `[H+S]` (`T8-T9A:157-171`).
- **(b) Overlap inside the layer:** hot GEMM first, cold copies under it. Options: the contiguous two-launch split (`T8-T9A:139`), or a producer that streams cold tiles. The speculative option is a VMM-stitched VA, so MegaMoE's TMA reads Grace directly (`T8-T9A:148`). At 0.11–0.22 cold/layer (Table 2, adaptive, 6 rows) that is ~5–11 µs of copy per layer under ~80 µs of hot work [E]: hideable.
- **(c) Cross-step reuse:** per-token distinct experts fall from 6 to 3.74 at 6 tokens (T5). Residency already exploits that.

**Not defensible without data.**
- Prefetching layer L+1's experts: its routing depends on layer L's output.
- Using draft-row routing to predict the verify.
- "Pre-gating" (gate_{L+1}(h_L)) has an unmeasured hit rate on this model. Measure it offline in A2-3, and build it only if more than 0.3 ms/step of cold time is still exposed after (a)–(b).

**The measurement that proves or kills A.** Exposed cold = t_step(A: adaptive + overlap) − t_step(timing control), at 6 rows on the fixture plus the shift corpus. The timing control redirects every residual cold pointer to a fixed HBM slot. Its numerics are wrong, so it is timing-only and never graded.
- **Pass:** ≤ 0.3 ms/step, i.e. 6 % of the 5.0 ms bar.
- **Kill:** > 1.0 ms/step in domain after the overlap work. A then loses on bandwidth; publish it.

For honesty: production's sidecar already hides its cold work at T = 6 (exposed GB wait 0.5 µs per layer-call, `peer3_stats.json` `by_T["6"]`). A must win on the total step (no RTX, no handshake), not on hiding.

## 5. Tasks

| task | desk | box window | gates |
|---|---|---|---|
| **A2-1 seed map + measurement** | `HotMap.load` rowmap-v0 branch + sha256 log; top-327 via `m3 plan`/`export`; `cnt` + `ids_ring` in the gather, npz dump; `bench/gather_cold_sweep.py` (t_launch(c), c = 0…6, 8/32/64 KB; D2H rate). Pair with the anatomy's gather-bandwidth lever (a) in one round | **~45 min**: boot (61 s) + capture × 3 maps (first-295, seed, top-327); 512-step cells (graph + dg-masked + gemv); static-eager nsys of the gather per map; microbench; traces of the fixture + a JP/DE/agent corpus; `mem_get_info` after capture | G1 8/8 digests equal across maps; G2 seed cold/step ≤ 6 on the fixture (prediction 1–11 vs 57); G3 (decision) Δ gather ms between maps = cold price: ≥ 1.5 ms confirms the link-bound model (prediction ≤ 2.75); < 0.5 ms → cold already hidden, lever (a) is the lever; report t_launch(c), D2H GB/s, free HBM after capture |
| **A2-2 swap policy** | `ResidencyManager` (spares, demotion pool, copy stream, events, single owner of ptab6/ptabs/loc/cold_tab); vendored policy; CPU: lockstep vs m3 on the Card M2 snapshots, T-R3, T-R4 | **~60 min**: fixture + shift corpus, frozen vs adaptive vs 3-point grid; forced-random-swap run | G1 T-R1 + T-R2 pass; G2 shift-phase steady cold share ≤ 0.6 % and ≤ frozen seed; G3 step-time cost of swapping ≤ 1 %; G4 swaps/step within cap |
| **A2-3 prefetch/overlap** (M2) | hot-in-place + overlapped cold staging (contiguous two-launch, then MegaMoE/VMM); timing-control mode; offline pre-gate hit rate on our traces | **~90 min**, after M1's 6-row graphs | exposed cold ≤ 0.3 ms/step at 6 rows → A viable; > 1.0 ms after overlap → A loses on bandwidth, publish |

Order: A2-1 first. It adds no mechanism, it fits in one short window, and its G3 tells us whether §1.1's model is real before A2-2 builds machinery on it.

A2-DESIGN: replace first-295 with m3's v26 seed map now (static, bit-identical, enlarged to ~327 hot/layer if post-capture HBM allows), then add the host-side v26 swap policy as A2-2 for shift robustness — expected cold share 0.5–0.9 % in-domain (~5 % on unfamiliar traffic) at 1 row vs 23.5 % today, and ≈0.1–0.2 cold experts/layer/step (≤0.4 ms/step if exposed; ~0.15 ms frozen at H=327 in-domain) at 6 rows with adaptivity, vs ~5 (≈10–14 ms) under first-295.
