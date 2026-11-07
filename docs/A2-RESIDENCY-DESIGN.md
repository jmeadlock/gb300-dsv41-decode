# A2 — Cold-expert residency for recipe A (design memo)

Desk worker memo, 2026-11-06 22:30 CST. Inputs: `docs/HANDOFF-A2-RESIDENCY-DESIGN.md` and the files it names, the m3 project (`~/hermes/milo-moe-manager/`), and two provenance files I opened to read field definitions: Garner's `fastpath/peer3/py/peer3_stats.py` and the T5 script `t5_unique_vs_k.py`. I did no box work. Labels: **[M]** = measured (receipt cited), **[E]** = my estimate (method stated), **[A]** = an assumption that still has to be measured. The offline estimates in §2 came from a numpy pass over count snapshots already on disk (cited per row). Scratch script: `~/.hermes/profiles/milo/cache/scratch/a2/est.py`.

## 0. Corrections to the inputs (read first)

1. **The expert is 18.8 MB, not 17.5 MB.** It is 18,800,640 B = 17.93 MiB. Sources: m3 golden (c) (`README.md:113`); the cold `HostBuffer` 62.334 GiB / 3,560 experts (`first-light-20261106.json` boot); 6.724 GiB/384 = 17.93 MiB. Per-expert C2C time is therefore **82 µs at 230 GB/s, 51 µs at 366, 48 µs at 390**. The probe measured 52.5 µs per expert on the copy engine (`C2C-PROBE-20261106.md:9`).
2. **peer3_stats was recorded under Garner's `rowmap-mix-v1-h255.json` (255 hot / 129 cold), not a 295/89 map.** Garner `docs/ARCHITECTURE.md:36-37`, `dsv41-garner-port-20261029/PLAN.md:41`. h255 is frequency-built from Garner's own calibration mix (its `meta`).
3. **At T = 6 the file says 1.60 distinct cold experts per layer-call, not 2.06.** 2.13 is the call-weighted mean over all T (`docs/data/peer3_stats.json` `by_T`). The C2C memo's "~80 cold reads ≈ 3.9 ms per 6-row step" (`C2C-PROBE:18`) should read **~64 reads ≈ 1.2 GB ≈ 3.1 ms at 390 GB/s, under a 129-cold map**. The conclusion stands: a static map at production quality still costs more than half of the 5.0 ms bar.
4. **"At any static map, cold reads cost ≥ 2.75 ms/step at 1 row"** (`T8-T9A-DESIGN.md:184`) holds for first-295 only. A frequency seed gets 1 row to ~0.1 ms (§2).

## 1. What first-295 costs today and at 6 rows

### 1.1 One row (measured)

| quantity | value | source |
|---|---|---|
| cold calls/step | **56.9 mean** (p50 57, p90 64, range 37–76) of 240 | `first-light{,-graph,-dg,-gemv}-20261106.json` `step_cell.cold`, n = 512 each [M] |
| cold share | 0.233–0.237 per step cell; 0.233–0.239 per prompt | same [M] |
| cold share of a map that ignores usage | 89/384 = 0.232 | arithmetic |

First-295 therefore carries **no usage information**: expert ids are not ordered by frequency.

**The gather and its marginal cost.** Per layer, the T8 path sorts the 6 ids and gathers every selected expert's 6 tensors through a device pointer table into `[6, N, K/2]` staging. In the committed tree that is 6 launches (`static_step.py` `StaticMoE._gather`, `static_kernels.py` `t8_gather_rows`). The in-flight T9b-2 tree folds them into one, `t9b2_gather_tables`, over `ptab6 [6, E]`. Each program copies 8 KB from `ptab[ids[j]]`. Hot sources are HBM; cold sources are UVA addresses of pinned Grace memory. The kernel cannot tell them apart.

The kernel copies all 6 experts every step, hot and cold alike: 113 MB per layer, 4.5 GB per step HBM→HBM at 1 row. At ~3.15 TB/s read (half the probe's 6.3 TB/s r+w) that is ≈ 1.4 ms/step before any cold byte moves [E].

A launch ends when its last program ends. The programs of a cold expert are link-bound, so the model is

  t_launch(c) ≈ max(6·b/B_hbm, c·b/B_c2c) + tail,

where c = cold experts in the launch and b = bytes per expert-tensor. The marginal cost of a cold expert is about b/B_c2c − b/B_hbm = **42–76 µs per expert** for B_c2c between 390 and 230 GB/s [E].

What this model cannot tell apart is a **link-bound** gather (cost linear in c) from a **latency-bound** one (several cold experts overlap, so c = 2 costs less than 2× c = 1). At 57 cold per step the range is **2.4–4.3 ms/step [E]**, 7–13 % of today's 32.16 ms.

None of this has been timed. The "MoE gather + DG ≈ 4 ms" figure (`T9B1-RESULT-20261106.md:22`) is a census estimate, not an nsys number, and it sits awkwardly with the model unless the gather beats 390 GB/s. A2-1 measures two things:
- (i) t_launch(c) for c = 0…6 with real pool addresses, at 8/32/64 KB per program;
- (ii) the 1-row step time, first-295 vs seed map. Both maps produce the same bits (§3.4), so the difference is the price of the cold reads.

### 1.2 Six rows (DSpark k = 5)

| quantity | value | source |
|---|---|---|
| distinct experts / layer / step, T = 6 | **22.45** (22.1–23.4 across classes); lower bound, because rejected drafts route too | `unique_experts_vs_k.json` ALL/"5" (576 windows) [M] |
| positional (first-N) cold share of that set | 0.246 (93-cold tail) | same, `grace_share_positional` [M] |
| first-295 cold experts / layer / step | **≈ 5.2** (22.45 × 89/384); 7.5 under m3's independence visit model on Card M2 counts | [E] |
| first-295 cold bytes / step | 208–300 × 18.8 MB = **3.9–5.6 GB → 10–14 ms at 390 GB/s** | [E] |
| usage-chosen cold set of the same size (in-sample) | 0.014 → 0.31 / layer → 12.6 / step → 0.24 GB → **0.6 ms** | same file, `grace_share_usage_cold` [M]; in-sample, so optimistic |
| Pg61, h255 (129 cold, frequency map), T = 6 | **1.60** distinct cold / layer-call; 73.4 % of calls touch ≥ 1 cold; 2.12 cold pairs → 64 / step → **3.1 ms at 390 GB/s** | `peer3_stats.json` `by_T["6"]`, 14,760 calls [M] |

At 6 rows first-295 is 2–3× the whole 5.0 ms bar on cold bytes alone, so it cannot be recipe A's map.

A second cost belongs to M1, but the residency design has to assume its fix. "Stage everything" copies ~22 experts per layer: 414 MB per layer, 16.6 GB per step, ≈ 5 ms/step of HBM copies [E]. The 6-row path must therefore read hot experts in place, through a contiguous GEMM over `[H+S]` or MegaMoE over `pool[0:H+S]` (`T8-T9A-DESIGN.md:139, 147, 157-171`), and stage only cold experts. With that in place, residency decides exactly one thing: which experts' bytes cross C2C.

### 1.3 What the peer3 fields are (`peer3_stats.py` header and `fold()`)

One record per peer-tier MoE call. `tag` encodes the layer and T = rows in the call (`(tag & 63) + 1`).

| field | meaning |
|---|---|
| `window_s`, `calls`, `calls_with_cold`, `lost_records` | fold window; total layer-calls (22,200 = 555 per layer × 40); calls with ≥ 1 row routed cold; ring overruns (0) |
| per layer: `cold_frac` | share of calls with any cold work |
| `rows`, `pairs` | mean rows with ≥ 1 cold pick; mean (row, cold expert) pairs |
| `cold_exp_mean` / `_p90` | distinct cold experts per call |
| `hot_exp_mean`, `hot_unknown` | distinct hot experts, known for only 47 of 555 calls per layer, so unusable |
| `busy_us_*` | RTX busy time per call with cold work |
| `gb_wait_us_*`, `gb_wait_over_5us` | GB300's exposed wait for the previous call's acknowledgement |
| `by_T[T]` | the same folded over layers per T, plus `cold_exp_hist` (index = distinct cold experts) |

The per-layer means mix every T. Calls at T = 12–64 carry 26–124 cold pairs each, so only `by_T` gives the per-step numbers. In this window, T = 6 is 66 % of calls and T = 8 is 22 %.

## 2. Seed map

**Reuse m3's v26 seed: `rowmap-mixed-ar-v1.json`.**
- sha256 `9b55d83a…` = `out/emit-v26/residency.proposed.yaml:9`; copy at `cards/M2/m3-hook/`.
- 295 hot / 89 cold.
- Built by m3's row-stable refit of static-v1 on 128.7 M AR1+AR2 decode selections on the 1017 checkpoint. Golden (b): 40/40 row-identical (`README.md:112`).
- In-sample cold share: 0.88 %. Static-v1, built on September counts, scores 18.2 % on the same data, which shows maps rot across checkpoints and workloads.

I estimated held-out numbers myself from Card M2 run 2. Routing does not depend on placement, so these per-(layer, expert) cumulative counts are a clean trace: 1017 checkpoint, a different boot and day from the seed's source, 16 snapshots 18 s apart (`cards/M2/results-run2/arm-{A,C,A2}/snap/counts-post-*.npz`).

**Table 2 — cold share and cold reads by map** (training counts = run1; evaluated on arm C; arms A and A2 agree to ±0.001):

| map | pick share, frozen phase¹ | pick share, shift phase² | cold/step, 1 row (shift … frozen) | cold experts/layer, 6 rows³ (shift … frozen) | cold bytes/step, 6 rows → at 390 GB/s |
|---|---|---|---|---|---|
| first-295 | 0.245 | 0.249 | 57 [M, our fixture] | 5.2–7.5 | 3.9–5.6 GB → 10–14 ms |
| h255 (Pg61) | 0.117 | 0.278⁴ | — | (1.60 measured, Pg61 fixture) | 1.2 GB → 3.1 ms |
| **seed v26ar, frozen** | **0.046–0.047** | **0.0045–0.0088** (per interval) | **1.1–2.1 … 11** | **0.16–0.31 … 1.6** | 0.12–0.23 … 1.2 GB → **0.3–0.6 … 3.1 ms** |
| seed, adaptive (v26 policy, map at snapshot⁵) | — | 0.0030–0.0061 | 0.7–1.5 | 0.11–0.22 | 0.08–0.17 GB → 0.2–0.4 ms |
| top-327 from run1 (+32 slots/layer) | 0.020 | 0.0022 | 0.5 … 4.8 | 0.08 … 0.72 | 0.06 … 0.54 GB → **0.15 … 1.4 ms** |
| top-340 from run1 (+45 slots/layer) | 0.014 | 0.0012 | 0.3 … 3.3 | 0.04 … 0.48 | 0.03 … 0.36 GB → 0.08 … 0.9 ms |

Notes:
1. Card M2 frozen phase: parity, C1/C8 knee window, agent fixture, gauntlet, plus the first 15 s of the shift (boot → snapshot 01). The counter counts calls with T ≤ 256 (`BRIEF-M2.md:24`), so this phase may include short prefills; unverified.
2. JP/DE → agent → JP/DE, with snapshot-to-snapshot deltas from 01 to the final.
3. m3's visit model with κ = 1, `v = 1 − (1 − π)^M` (`m3/cost.py:5`). This is an upper bound on distinct experts. For rare experts it is ≈ pick share × 36.
4. h255 varies a lot by workload. On Pg61's own fixture it is far better than on Card M2's agent/JP-DE traffic.
5. Optimistic: the map at the end of an interval is scored on that same interval.

**Expected seed cold share:**
- 1 row: 0.5–0.9 % in domain, ~5 % on unfamiliar traffic, against first-295's 23.5 %.
- 6 rows: 0.2–1.6 cold experts per layer per step, so 0.4–3.1 ms per step if fully exposed.

The spread across workloads is the case for adaptivity. The seed alone fixes the in-domain case. It does not fix a shift: the frozen-phase row is as bad as Pg61's h255.

**The cheapest win is to boot a bigger hot set.** HBM headroom is 249.81 − 218.61 = 31.2 GiB (`placement.py` `--hbm-budget-gib`; `t2-boot`). That is 1,782 slots in total (44.5/layer) before anything else claims it. KV at 32K is only ~0.15 GB (`LEDGER.md` 14:17).

What has not been measured is the graph pool, workspaces and staging after capture: the JSONs record only boot `cuda_allocated` = 218.61. Planning reserve [A]: 4.2 GiB of MegaMoE cold staging (`T8-T9A:170`) plus ~4 GiB of workspaces. That leaves ~23 GiB → **32 extra slots per layer, H = 327**.

Keep H uniform across layers. MegaMoE fixes `num_experts` per launch (`T8-T9A:147`), and the literature puts non-uniform layer budgets at ≤ 3 % (ERM `REPORT.md:13`). L0 and L1 are the seed's worst layers (5.2 % and 4.9 % on arm C).

**Tool and format.**
- `runtime/placement.py` `HotMap.load` takes `{"<layer>": [ids]}` or `{"hot": {...}}` (`placement.py:111-116`).
- m3's rowmap-v0 is `{"layers": {"<l>": {"hot": [...], "cold": [...]}}}` (`m3/residency.py` docstring).
- Conversion is one comprehension, `{l: v["hot"] for l, v in d["layers"].items()}`. A2-1 adds it as a 3-line branch in `HotMap.load` that logs the file's sha256. Row order is irrelevant to us because `HotMap` sorts ids.
- For H ≠ 295, emit with `m3 plan` using `examples/dsv41-seed-mixed-ar-v1.yaml` with `banks.H.allowed_sizes: [327]` and no `parent_map`: a new geometry, so plain threshold placement. Then run `m3 export --format rowmap-v0`.
- Our own trace: dump `cnt` int64 `[40, 384]` (+ `host_steps`) to `.npz`. `m3 trace import` detects it as the v26ar format (`m3/trace.py:9-12`).

**What in m3 is engine-agnostic vs vLLM-bound.**
- Agnostic, pure numpy, reusable as-is: Sizer (`geometry.py`), trace import/stats, cost model, solver (threshold, row-stable refit, bank sizes), `residency-map/v1` + rowmap-v0, and `m3.control.policy`. The policy is the v26ar.2 decision core, lockstep-checked over 93,400 checks with 0 diffs (`README.md:122`).
- vLLM/FlashInfer-bound: every actuator (`flashinfer_split` = FlashInfer two-bank split + `execute_model` barrier; `adaptive.py` hook; `megamoe_peer` = el8 sidecar; `marlin_expertmap`), `shim/`, `emit/` (jm-recipes vLLM recipes), the gate parsers, the cost constants (fit on vLLM lanes), the bank "rows" concept, and the 1.9–2.4 ms per-swap cost (vLLM step structure: `REPORT.md:15`).

## 3. Adaptive residency inside our runtime

### 3.1 Mechanism

A swap is a byte copy plus a pointer-table write. Kernels do not change.

- **HBM:** per layer, H hot slots plus r spare slots [A: r = 4]. These come from boot allocation, which `expert_pool.py` already does per layer (`[H, N, K/2]` pools); it becomes `[H + r, …]`.
- **Grace:** today's cold pool, plus a **demotion pool** of D slots per layer, registered at boot [A: D = 64 → 45 GiB]. Registration with hugepages took 0.66 s for 251 GiB (`LEDGER` 14:15), and 197 GiB stays available after pinning (`t2-boot` mem_log).
- No full Grace mirror. Engram (189 GiB) plus all experts (269 GiB) would be 458 GiB against 449.5 available at boot.
- **Promote e_in** (Grace → spare slot s): H2D on a copy stream, 6 `cudaMemcpyAsync` calls (one per tensor), ≈ 52.5 µs per expert [M]. Record an event. Between two replays, on the compute stream: `wait(event)`, write `ptab6[:, e_in]` (6 int64), and update `loc[l, e_in]` and `cold_tab[l, e_in]`.
- **Demote e_out:** an expert promoted at runtime keeps its Grace copy, so demotion is only a pointer write. A boot-hot expert has no Grace copy, so it first needs one D2H into the demotion pool. The C2C D2H rate is not measured [A]; measure it in A2-1.
- **Reclaim a slot:** an HBM slot returns to the spare list only after an event recorded after the first replay that no longer references it. The swap never stalls a replay.
- Copies may not cross a `cudaHostRegister` chunk (`LEDGER` 14:11). Each pool is one chunk, so every copy stays inside one.

### 3.2 Policy: reuse m3's controller, host side, every N steps

- **Counter:** `cnt` int32 `[40, 384]` on the device. Program `(j, 0, 0)` of `t9b2_gather_tables` already stores `ids[j]`; it adds `tl.atomic_add(cnt + l·E + e, 1)`. The v26ar in-kernel relaxed-atomic tracer measured 2–6 µs/step (ERM `W5-system-design.md:34`) [M, vLLM lane]. The atomics touch counters only, never data, which fits decision 14.
- **Trace ring:** `ids_ring[pos % R]` ← `ids_step` (already filled per step: `StaticStep.ids_step`). At R = 4096 it is 7.9 MB. Every routed id per step is logged at one launch per step.
- **Tick:** every `drain_n` steps the host copies `cnt` (61 KB) D2H, asynchronously into pinned memory. It runs `m3.control.policy` (vendored, sha-pinned: `drain` → `queue` → `apply_pending`) and issues the copies of §3.1. The host has nothing else to do during a replay, so the decode stream never waits on the policy.
- **Start values** = shipped v26 (`residency.proposed.yaml`): drain_n 64, swaps_per_drain 2, halflife 500, hyst 1.5, min_abs 8, swaps_per_barrier 8.
- **What changes vs vLLM:** a swap costs about 52 µs of copy-engine time, fully concurrent with compute (overlap probe: GEMM 90.6 → 90.6 µs) [M], plus 48 B of pointer writes. vLLM paid 1.9–2.4 ms per applied swap. The economic floor S/(Δc·H) of the admission rule `Δv > max(z√(2v/τ), S/(Δc·H))` (ERM `W3:169`) shrinks ~40×, so the statistical floor binds. A cheaper, more aggressive policy may now pay where AR6 H2 found none. A2-2 tests a 3-point grid.
- **Budget:** at 390 GB/s and a 5 ms step the link could take ~100 swaps per step. The SM cold reads share it, so cap swaps at ≤ 4 per step, ≈ 0.2 ms of link time (4 %) [A]. v26's average rate is ≤ 2·40/64 = 1.25 swaps/step.

**Device-side policy: rejected for now.** The copy engine cannot be driven from a device-side decision inside a graph (`T8-T9A:81`). A device-side policy would need SM copy kernels and a device free list. Both cost SM time, and they buy nothing while the host is idle anyway.

### 3.3 What the graph tolerates

The pointer tables (`ptab6`, the per-key `ptabs`, `loc`, `cold_tab`) are device tensors the graph reads by address. Rewriting their contents between replays, stream-ordered on the replay stream, is legal. It is exactly how `tok` and `pos` are fed (`static_step.py` `set_inputs`). Two things are illegal:
- reallocating a captured tensor;
- writing a table that is a stale copy. `ptab6` is built by stacking the six tables, so the swap writer must update **every** table a kernel actually reads. That needs one owner object, and a test.

### 3.4 Determinism: proof obligation and tests

**Claim:** for every swap schedule σ and map M, the logits are bitwise equal to those of first-295 with no swaps.

The claim needs four things:
- **P1 bytes.** Every location a pointer may name holds the checkpoint bytes. Copies are byte copies of immutable weights. A pointer is published only after its copy's event, and a slot is reused only after the last replay that referenced it.
- **P2 no tier-dependent arithmetic.** Staging slot j is always the j-th ascending id. GEMMs and the fp32 combine run in slot order. For the in-place paths, contiguous and MegaMoE (local index = slot, cold → H + j), the combine must reduce in top-k or id order, never by slot index. Open check: MegaMoE's combine order (`sm100_fp8_fp4_mega_moe.cuh`).
- **P3** Counters and atomics feed only the policy.
- **P4** The policy is deterministic. m3's core is deterministic; when a copy happens to land does not matter, by P1.

**Tests:**
- **T-R1:** the 8-prompt fixture under first-295, seed, top-327, and seed with forced random swaps on every step (including experts the same step routes to) → fp32 logit digests and token streams `torch.equal`. This extends A3's "tiered == all-HBM bit-exact".
- **T-R2:** pool audit after a run. sha256 every HBM slot and demotion-pool slot against the shard bytes.
- **T-R3:** invariants at every tick: one published home per expert, no shared HBM slot, spare ∩ published = ∅. These are m3's I1–I2 for our layout.
- **T-R4:** mutations. Publish before the copy event, corrupt one promoted byte, update `ptabs` but not `ptab6` → T-R1/T-R2 must fail.

## 4. Recipe A thesis check

**What exists at M0.** The gather's programs load cold experts from Grace through UVA pointers inside the graph (`static_kernels.py` docstring). That is recipe A's self-fetch in its primitive form. It has two defects:
- it stages the hot experts too (§1.2);
- the cold fetch is stream-ordered before the GEMM, so it is fully exposed (`T8-T9A:149`).

**What M2 adds, defensibly:**
- **(a) Hot in place, cold staged only.** Contiguous or MegaMoE over `[H+S]` (`T8-T9A:157-171`).
- **(b) Overlap inside the layer.** Hot GEMM first, cold copies under it: the contiguous two-launch split (`T8-T9A:139`), or a producer that streams cold tiles while hot tiles compute. The speculative variant is a VMM-stitched VA so MegaMoE's TMA reads Grace directly (`T8-T9A:148`). The C2C memo's arithmetic: ~21 µs of copy per layer at ~0.4 cold against ~80 µs of hot MoE work. With Table 2's adaptive numbers (0.11–0.22 cold/layer at 6 rows) the copy is ~10 µs per layer, which the hot work can hide [E].
- **(c) Cross-step reuse.** Per-token distinct experts drop from 6 to 3.74 at 6 tokens (T5). That locality is what residency already exploits.

**What is not defensible without data:**
- Prefetching layer L+1's experts. Its routing depends on layer L's output.
- Using draft-row routing as a predictor for the verify.

"Pre-gating" (layer L+1's gate applied to h_L) is a predictor whose hit rate on this model is unmeasured. It is an offline measurement in A2-3, and only worth building if the exposed cold time left over after (a)–(b) exceeds 0.3 ms/step.

**The measurement that proves or kills the thesis.** Exposed cold time = t_step(recipe A: adaptive + overlap) − t_step(timing control). In the timing control every residual cold pointer is redirected to a fixed HBM slot. Its numerics are wrong, so it is timing-only and never graded. Run at 6 rows on the fixture plus the shift corpus.
- **Passes** at ≤ 0.3 ms/step: 6 % of the 5.0 ms bar.
- **Fails** above 1.0 ms/step in domain, even after the overlap work. Recipe A then loses on bandwidth; publish it.

For honesty: production's sidecar already hides almost all of its cold work at T = 6. Its exposed GB wait is 0.5 µs per layer-call ≈ 20 µs/step (`peer3_stats.json` `by_T["6"]`). Recipe A therefore wins on the total step — no RTX, no publish/handshake kernels — not on hiding.

## 5. Tasks

| task | desk | box window | gates |
|---|---|---|---|
| **A2-1 seed map + measurement** | `HotMap.load` rowmap-v0 branch (+ sha256 log); top-327 map via `m3 plan`/`export`; `cnt` counter and `ids_ring` in the gather + npz dump; `bench/gather_cold_sweep.py` (t_launch(c), c = 0…6, 8/32/64 KB blocks, real pool pointers; D2H rate) | **~45 min**: boot (61 s) + capture per map × 3 (first-295, seed, top-327); 512-step cells (graph + dg-masked + gemv); microbench; trace the fixture plus a JP/DE + agent corpus; `mem_get_info` after capture | G1 8/8 digests equal across the 3 maps; G2 cold/step seed ≤ 6 on the fixture (prediction 1–11 vs 57); G3 step saving ≥ 1.5 ms at 1 row (prediction 2.4–4.3; if < 1 ms, re-model the gather before A2-2); report t_launch(c), D2H GB/s, free HBM after capture |
| **A2-2 swap policy** | `ResidencyManager` (spares, demotion pool, copy stream, events, single owner of ptab6/ptabs/loc/cold_tab); vendored `m3.control.policy` (sha-pinned); CPU tests: lockstep vs m3 on the Card M2 snapshots, T-R3, T-R4 | **~60 min**: fixture + shift corpus, frozen vs adaptive vs 3-point policy grid; forced-random-swap determinism run | G1 T-R1 + T-R2 pass (adaptive and forced swaps); G2 steady shift-phase cold share ≤ 0.6 % and ≤ frozen seed; G3 step-time regression from swapping ≤ 1 %; G4 swaps/step within the cap |
| **A2-3 prefetch/overlap** (M2) | hot-first in-place path with overlapped cold staging (contiguous two-launch, then MegaMoE/VMM); timing-control mode; offline pre-gate hit-rate script on our traces | **~90 min**, after M1's 6-row graphs exist | exposed cold ≤ 0.3 ms/step at 6 rows → recipe A viable; > 1.0 ms after overlap → A loses on bandwidth, publish |

Order: A2-1 first. It needs no new mechanism, it uses the box for ~45 min, and its G3 tells us whether the cost model of §1.1 is real before A2-2 builds machinery on it.

A2-DESIGN: replace first-295 with m3's v26 seed map now (static, bit-identical, and the HBM headroom permitting, enlarged to ~327 hot/layer), then add the host-side v26 swap policy as A2-2 for shift robustness — expected cold share 0.5–0.9 % in-domain (~5 % on unfamiliar traffic) at 1 row vs 23.5 % today, ≈0.1–0.2 cold experts/layer/step (≤0.4 ms/step if exposed; ~0.15 ms frozen at H=327 in-domain) at 6 rows with adaptivity, vs ~5 (≈10–14 ms) under first-295.
