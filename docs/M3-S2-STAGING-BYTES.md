# M3-S2 — what the staging launch moves, and the 2.7 ms it still exposes

Handoff: `docs/HANDOFF-M3-S2-STAGING-BYTES.md`. Branch `m3-s1` (worktree `gb300-dsv41-decode-wts1`), 2026-11-07 late.
Tags: [M] measured, [D] derived from receipts by the arithmetic shown, [E] modeled, [A] assumed.
Instrument: `bench/staging_bytes_bench.py` (CPU: `census`, `pool-fit`, `exposure`, `hist`; box: `probe`).
CPU receipt: `scratch/staging-bytes-all-20261108.json` (+ `.log`).

## Bottom line

- In the headline condition (residency on, kernel-overlap) the bytes the staging moves per 6-row layer-call are
  dominated by **(c) the HOT experts' scale factors under sf_store "step": ~21 experts × 1.106 MB = 22.8–23.8 MB
  read + the same written (HBM, a transpose), every layer, every step** — vs 6.3–21 MB of cold weights over C2C.
- Time: the exposed staging (19.34 − 16.64 = 2.70 ms; 18.65 − 16.62 = 2.03 ms in the warm cell) splits into a
  per-cold-expert part (~20 µs per cold expert per layer-call) and a **c-independent ~37–43 µs/layer ≈ 1.5–1.7
  ms/step** that the M3-S1 probe (pool-style lists, no hot scales) does not have [D]. That c-independent part is the
  hot-SF repack + its contention — the term sf_store "pool" removes.
- "pool" needs 13.14 GiB (derived below) and 4.93 GiB is free: it cannot fit by any headroom value. Implemented
  the bit-identical part that does fit: **partial pool** (`$DSV41_SF_POOL_LAYERS=auto`): the layers that fit get
  their hot SF pre-packed, the rest stay "step". Default unchanged (`all` = today's all-or-nothing rule).
- Expected [E]: 7 layers at today's pool (headroom 2.0) −0.23..−0.26 ms → ~19.1; 16 layers at `--pool-spare 20`
  ~18.8–18.9; 24 layers at `--spare-slots 8 --pool-spare 16` ~18.5 (only if the cold histogram clears overflow).
  The whole 1.3–1.5 ms needs the 12.15 GiB K-major e8m0 duplicate out of HBM (next lever, a layout change).

## (1) Bytes census per layer-call at 6 rows

Real rows (`runtime/expert_pool.py` proj shapes): weights w1/w3 2304×2560 + w2 5120×1152 = **17,694,720 B/expert**;
e8m0 scales 2304×160 ×2 + 5120×72 = **1,105,920 B/expert** (the packed int32 MN-major copy is the same bytes).

Counted [M] through the runtime's own dedup (`moe_rows.dedup_routes` + `rows_layout`) and staging contracts
(`dg_inplace.stage_rows_emulated` = m1_stage_rows, `staging_overlap.stage_proj_emulated` ×3 = s1_stage_proj) on a
synthetic layer, 22 unique experts, c cold: rows read per layer-call

| c cold | step store | pool store |
|---|---|---|
| 0 | hot.scale 66 | — (work list empty) |
| 1 | cold.weight 3, cold.scale 3, hot.scale 63 | cold.weight 3, cold.scale 3 |
| 2 | 6 / 6 / hot.scale 60 | 6 / 6 |
| 4 | 12 / 12 / hot.scale 54 | 12 / 12 |

i.e. 3 rows per cold expert per kind, and under "step" 3 scale rows per USED HOT expert; no hot weight is ever
read. Step-style lists write exactly the bytes pool-style lists write on a pre-packed SF (the hot groups get the
same values again; `contract_check` [M]). (d) the rest: work list / tables / m_indices < 2 KB; the zero-work
launch 2.8 µs [M probe c=0].

Bytes at the receipts' counts [D] (U = 21.84 unique/layer-call from `docs/gap-tf-static-20261107.json`:
195,298 hot + 15,236 cold unique over 9,640 layer-calls; cold per call = cold share × U):

| condition | hot | cold | (a) cold weights MB (C2C) | (b) cold scales MB | (c) hot scales MB, step (HBM r+w each) |
|---|---|---|---|---|---|
| tf seed map, residency off (anatomy cell) | 20.26 | 1.58 | 27.97 | 1.75 | 22.40 |
| headline two_graphs (cold share 5.45 %) | 20.65 | 1.19 | 21.07 | 1.32 | 22.84 |
| headline one_graph (warm, 1.63 %) | 21.48 | 0.36 | 6.32 | 0.39 | 23.76 |

Which term dominates the 114 µs (serial `m1_stage_rows`, seed map) [D]: the M3-S1 probe measured the same kernel
with pool-style (cold-only) lists at real shapes: 2.83 µs at c=0, then 48.9 µs per cold expert (362 GB/s) +
3.6 µs. At the anatomy's mean 1.58: **80.7 µs cold-only; the remaining 33.1 µs is the hot-scale repack** (1.63 µs
per hot expert, 22.4 MB r + w ≈ 1.35 TB/s effective). Independent second reading: the probe's own serial model
(3.404 ms/step) vs the anatomy's 4.552 → 28.7 µs/layer excess. So at the seed map the cold weights dominate the
serial time (≈71 %); in the headline condition (fewer cold experts, same hot count) the hot scales dominate the
bytes, and — next section — the exposure.

## Exposure after M3-S1 (feeds (3))

`bench/staging_bytes_bench.py exposure` [D]: exposed = kernel-overlap − none-diag in the same cell
(`docs/headline-s1-shapes-20261107.json`, `docs/headline-s1-nonediag-20261107.json`); the probe's pool-style
kernel-overlap exposure (np48b8k: 3.2 / 17.4 / 48.6 / 95.3 / 141.6 µs at c = 0..4) at Poisson(cold/call) [A]:

| cell | exposed ms | cold/call | probe pool-style ms [E] | residual ms | residual µs/layer |
|---|---|---|---|---|---|
| two_graphs (the headline) | 2.705 | 1.19 | 1.24 | 1.47 | 36.6 |
| one_graph (warm) | 2.028 | 0.36 | 0.37 | 1.65 | 41.3 |

The 2-point linear fit, no probe: 20.3 µs per cold expert per layer-call, **43.5 µs/layer (1.74 ms/step) at zero
cold**. The probe has no hot scales and exposes 3.2 µs/layer at c = 0 (launch + fork/join). The c-independent
~37–43 µs/layer is therefore attributed to the step store's hot-SF repack on the side stream (its duration on the
w1 path, and its CTAs contending with the persistent DeepGEMM under w3/w2) — **a [D] attribution the box probe
tests directly** (`probe` B: `pooled_layer_saves_us` at c = 0; C: exposure vs hot slots listed).

## (2) sf_store "pool": why 13.14 GiB, what reduces it, what fits

Derivation: one pool group's packed SF = (2304×40 + 2304×40 + 5120×18) int32 words × 4 B = 1,105,920 B; a layer has
G = H + S = 295 + 24 = 319 groups; 40 × 319 × 1,105,920 B = 14.11e9 B = **13.142 GiB** (`pool_sf_layer_bytes`;
hot part 12.154, spare part 0.989). The step store's shared copy is one layer's worth, 0.329 GiB.

- Do the hot SFs already live in the pool? **Yes, as K-major e8m0** (`pool.hbm[l][p][1]`, [H, N, K/32], 12.15
  GiB) — the layout the reference experts and the residency copies use. DeepGEMM's sm100 kernel needs the
  MN-major packed transpose, so the pool copy is a transposed duplicate of bytes already in HBM. That duplicate is
  the real budget item.
- Only the staging + residency slots pooled? No: the GEMM reads packed SF for every group it touches; 295 hot
  groups per layer need it. Pooling only the 24 spare groups (0.99 GiB) is the step store with extra steps.
- Per-layer lazy allocation: already lazy (`pool_packed_sf` builds a layer on first use); every layer is touched
  in the first step, so the steady state is the full copy. No reduction.
- Lower `$DSV41_SF_POOL_HEADROOM_GIB`? Free at the check is 4.93 GiB [M, every headline log] < 13.14: no headroom
  value makes the whole copy fit. What is actually safe: the receipt build consumed **1.30 GiB after the check**
  (4.93 → 3.63 `hbm_after_build_GiB.free`; that includes the 0.33 GiB step SF, the KV at max_pos 4095 (0.84), the
  1-row step, the draft graph, workspaces and capture); max_allocated after build +0.03 GiB. Counting the step SF
  in the plan's need (partial pool does), **2.0 GiB headroom leaves ~1.0 GiB free after build** [E]; 1.5 → ~0.5.
- Trim `--pool-spare` 24 → 20 (residency 8 → 4 slots): frees 4 × 40 × (17.69 MB weights + 1.11 MB residency
  e8m0) = 2.80 GiB; still far from 13.14 (7.73 free) — it buys more partial-pool layers, not the whole copy.

What fits — `bench/staging_bytes_bench.py pool-fit`, decided by `dg_inplace.sf_pool_plan` (the build's rule):

| pool-spare | stage | resid. | headroom | layers pooled | need GiB | free after build [E] | saved ms [E] |
|---|---|---|---|---|---|---|---|
| 24 (today) | 16 | 8 | 6.0 (today) | 0 (step) | 0 | 3.63 | 0 |
| 24 | 16 | 8 | 2.5 | 6 | 2.30 | 1.66 | 0.20–0.22 |
| 24 | 16 | 8 | **2.0** | **7** | 2.63 | 1.33 | **0.23–0.26** |
| 20 | 16 | 4 | 2.0 | 16 | 5.52 | 1.25 | 0.53–0.59 |
| 17 | 16 | 1 | 2.0 | 23 | 7.71 | 1.15 | 0.76–0.84 |
| 16 | 8 | 8 | 2.0 | 24 | 8.01 | 1.23 | 0.79–0.88 |

saved = layers × [33.1 serial-derived, 36.6 overlap-residual] µs. Trading residency slots for pooled layers
[D]: residency off runs at cold share 6.39 % (`docs/headline-pdl-on-20261107.json`) = 1.40 cold/call; the 8 slots
take the warm cell to 0.36 (−1.04, 0.13/slot average, ~0.11 ms/step per slot = 0.15 ms/GiB at 20.3 µs/cold) but
the headline two_graphs cell only to 1.19 (−0.21: 0.02 ms/step per slot = 0.03 ms/GiB). A pooled layer is
0.033–0.037 ms per 0.329 GiB = 0.10–0.11 ms/GiB in either cell. So for the headline metric pool-spare 20 wins;
in the warm steady state it is about even (and the last 4 residency slots are worth less than the average).
Staging slots 16 → 8 costs no residency but an overflow (more than 8 cold unique in one layer-call = wrong logits,
counted) must never happen: gated on the histogram below.

## (3) What remains exposed with the hot-SF copies gone

If every layer were pooled, the remaining exposure is the probe's pool-style figure: **1.24 ms (two_graphs) /
0.37 ms (warm)** [E] — the cold weights' bytes (~20 µs per cold expert per layer-call at 1.19 / 0.36 cold) plus
~0.13 ms of launch + fork/join (3.2 µs/layer). So after the SF term the lever is the hot set (cold count), not the
fork's join latency. +N hot experts per layer: 0.70 GiB per slot over 40 layers; the measured average value of
a residency slot is above (0.02–0.11 ms/step per slot), the marginal one is lower and unmeasured — the
`$DSV41_STAGE_HIST` instrument (below) records per-(layer, expert) cold hits, from which
`staging_bytes_bench.py hist` reports the cold per layer-call the top-N cold experts of each layer would remove.
Against today's HBM (≤ 1–3 GiB spendable) more hot experts are not on the table; the SF bytes are the cheaper GiB.

## (4) Implemented (bit-identical by construction; current behaviour is the default)

- `runtime/dg_inplace.py` (sf_store pool sizing): `$DSV41_SF_POOL_LAYERS` = `all` (default: today's rule) |
  `auto` (as many layers as fit above the headroom; layers already packed first, then ascending) | `N` | `l0,l1,..`.
  `sf_pool_plan()` decides (pure function, tested); `pool_sf_layer_bytes()`; per-layer store `sf_layer`,
  `layer_store(t)` / `step_sf(t)` / `stage_n_work(t)`; a step layer of a partial pool reads the shared step SF
  through `StepSFView` — deliberately not a dict, so `runtime/residency.py attach()` registers promoted-expert SF
  writes for the pooled layers only (a residency write into the shared tensor would race another layer's GEMM).
  `describe()`: `sf_layers`, `sf_pool_layers`, `sf_pool_plan`; stderr `[dg-inplace] sf_store pool: partial pool: k/40 ...`.
- `runtime/moe_rows.py` (hook): the dedup's `step_sf` is per layer (`b.step_sf(t)`), so hot slots enter the work
  list only in step layers; `runtime/staging_overlap.py`: `n_work` per layer.
- Instrument `$DSV41_STAGE_HIST=<json>` (`StageOverlap.count`, called from `DGInplaceMoERows._stage`): per
  layer-call cold / hot unique histogram + per-(layer, expert) cold hits as device one-hot adds (capturable, no
  atomics, no sync), JSON at exit (per path the instance that counted most; others to `<path>.<i>`). Off by default (no tensors, no launches). It adds ~6 tiny kernels per layer:
  **never time a headline with it on**.
- Tests: `tests/test_staging_overlap_cpu.py` (+7: derivation, plan rules, partial == pool == step in kernel and
  kernel-overlap incl. per-layer work lists, partial + residency (forced + policy swaps in pooled and step layers)
  == residency-off step store with resident-SF registration only on pooled layers, hist == the step's routed
  counters and changes nothing, bench CPU subcommands); a negative control (wrong per-layer `step_sf`) breaks the
  partial run's logits. `tests/gpu/test_staging_overlap_gpu.py` (+2: partial / whole pool == step store eager and
  graph in kernel and kernel-overlap, hist in a captured graph; the bytes probe quick).
- Mac verification: `.venv/bin/python -m pytest tests/ -q` = 1277 passed, 196 skipped, **23 failed — the identical
  23 fail on an untouched `git archive HEAD` copy** (`scratch/pytest-m3s2-baseline-HEAD.log` vs
  `scratch/pytest-m3s2.log`: same set; combined-run order dependence + a stale residency-default assertion, all
  pass when their files run alone: test_static_rows / test_residency / test_t9b4 125/125 with this change).
  The touched modules' files (staging_overlap, moe_rows_t10, dg_inplace, residency_rows, router_plan_rows):
  149/149.

## Box commands (orchestrator; lease-serialized as `bench/box/m3s1b.sh`, same `dk` wrapper and image)

```
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 24 --residency-drain 64 --max-new 256"
# t  GPU gate (all PASS incl. test_m3s2_partial_pool_mini4, test_m3s2_bytes_probe_quick)
dk s2t --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py > $S/s2-gpu.log 2>&1
# p  bytes probe: read overlap[c]["pooled_layer_saves_us"] (hypothesis: ~33-41 at c=0) and hot_sweep
dk s2p --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/staging_bytes_bench.py probe --out scratch/staging-bytes-probe-$(date +%Y%m%d).json > $S/s2-probe.log 2>&1
# a  baseline, same window
dk s2a --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s2-A.json > $S/headline-s2-A.log 2>&1
# b  partial pool at today's spare (expect the log line "partial pool: 7/40")
dk s2b -e DSV41_SF_POOL_LAYERS=auto -e DSV41_SF_POOL_HEADROOM_GIB=2.0 --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s2-B.json > $S/headline-s2-B.log 2>&1
# h  cold histogram (TIMING DISCARDED: the instrument adds kernels); then on the host: bench/staging_bytes_bench.py hist --hist scratch/stage-hist-s2.json
dk s2h -e DSV41_STAGE_HIST=scratch/stage-hist-s2.json --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s2-hist.json > $S/headline-s2-hist.log 2>&1
# c  residency 4 slots, 16 layers pooled (HL with --pool-spare 20)
dk s2c -e DSV41_SF_POOL_LAYERS=auto -e DSV41_SF_POOL_HEADROOM_GIB=2.0 --entrypoint python3 vllm/vllm-openai:v0.31.0 ${HL/--pool-spare 24/--pool-spare 20} --json scratch/headline-s2-C.json > $S/headline-s2-C.log 2>&1
# d  ONLY if h shows cold_max <= 6 and p_cold_gt["8"] == 0: staging 8 + residency 8, 24 layers pooled
dk s2d -e DSV41_SF_POOL_LAYERS=auto -e DSV41_SF_POOL_HEADROOM_GIB=2.0 --entrypoint python3 vllm/vllm-openai:v0.31.0 ${HL/--spare-slots 16 --residency on --pool-spare 24/--spare-slots 8 --residency on --pool-spare 16} --json scratch/headline-s2-D.json > $S/headline-s2-D.log 2>&1
grep -h "\[dg-inplace\] sf_store" $S/headline-s2-*.log
```

Expected (two_graphs headline cell, base 19.34) [E]: **B 19.08–19.11**; C 18.75–18.89 (residency loss ≤ 0.08 in
this cell; the one_graph cell about even); D 18.46–18.55 if overflow stays 0. Adopt per arm only with streams 8/8,
self-replay 8/8, overflow 0, GPU gate PASS. If the probe's `pooled_layer_saves_us` at c = 0 is ≲ 5 µs the residual
is not the hot SF (then fork/join / graph structure: back to ordering) and B–D will not move the headline.

## Next lever (not done: an ExpertPool layout change, out of this round's ownership)

Move the hot experts' K-major e8m0 scales (12.15 GiB, never modified after boot except by residency promotions
into boot slots) to pinned Grace once the pool SF copy exists: the step reads only the packed copy; prefill's
reference experts read scales over C2C as the cold ones already do (bit-identical, +11.5 µs per call [M C2C
probe], ~+40 ms on a 4 s prefill [E]); residency copies source/target host bytes. Then the whole pool copy fits
(4.93 + 12.15 − 13.14 ≈ 3.9 GiB free at the check) → the full c-independent term, **−1.3..−1.5 ms/step [E] beyond
today (~17.9)**, and the step store's 0.33 GiB goes too.
