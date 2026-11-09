# HANDOFF — A3: adaptive residency on gracetier's fixed slots (card GT3)

You are the A3 coding worker. Read in full, in order: `docs/REPLAN-20261109.md`; `docs/LEDGER.md` tail (GT1 and A2
entries of 2026-11-09); `docs/P1-GRACETIER-DESIGN.md` §2.2 and §4; `port/garner/fastpath/fastpath_gracetier.py`
(all of it — you extend it); `runtime/gracetier_stage.py`, `runtime/gracetier_pool.py`, `runtime/gracetier_vmm.py`,
`runtime/host_tier.py`; `runtime/residency_policy.py` (the v26ar.2 decision core, vendored byte-identical: use it
unchanged) and `runtime/residency.py` (how our standalone runtime drives that policy: counters, drain, barrier,
`_copies`, demotion views — patterns to reuse, not the design to copy); in Garner's tree (read-only)
`reference/garner-8043a681/fastpath/fastpath_a12_retire.py` (it hooks `GPUModelRunner` and owns the SF banks'
compressed copies — you must not break it), `fastpath_boundary.py` (how `row_map` is read per call),
`fastpath_peer3.py`; `bench/box/card-gt1.sh` (you add a mode); `tests/test_gracetier_cpu.py`,
`tests/test_gracetier_port_cpu.py`, `tests/gpu/test_gracetier_gpu.py`.

## Where we are (measured, receipts in `docs/gt1/`)
gracetier serves DSV4.1-Flash on one GB300 with a static 255-hot map: **10.45 ms/step**. The nsys anatomy says the
whole single-GPU penalty is one kernel: `gt_stage` **2.60 ms/step** (40 × 65 µs); step minus `gt_stage` = 7.89 ms,
Pg61 (two GPUs) = 7.75. `gt_stage` costs ~6 µs per layer-call when nothing is cold and ~48 µs per cold expert. So the
lever is the number of cold picks, and the tool is the adaptive residency policy we already ship in the vLLM UVA
recipe (v26ar: cold share 17 % → 0.7 % on mixed traffic, T90 9–16 s).

## The design you build (decided; do not redesign — if a fact in the code makes it impossible, stop and say which)
Geometry stays exactly as it is: G = 384, groups 0..H−1 private HBM, groups H..383 on the shared scratch, Grace pool
slot c ↔ group H + c, SF banks resident for all 384 groups. **A swap is a true exchange** of hot expert `e_out`
(group g < H) with cold expert `e_in` (cold slot c):

1. D2D: HBM group g weights (L1, L2) → scratch group H + c (the scratch is free between steps; this is the staging row)
2. H2D: Grace slot c (L1, L2) → HBM group g
3. D2H: scratch group H + c → Grace slot c (the pool is registered host memory; chunk rule in `host_tier.py` applies)
4. SF: exchange rows g and H + c in both SF banks **that MegaMoE actually reads after `a12_retire`** (read
   `fastpath_a12_retire.py` and say in a comment which tensors those are and how you verified it)
5. tables: `row_map[e_in] = g`, `row_map[e_out] = H + c`, `g2e[g] = e_in`, `g2e[H + c] = e_out`

`src_tab` never changes (slot c's Grace address is fixed; only its contents change). No extra host or HBM memory.
All five steps are enqueued **on the model's own stream at a barrier between two `execute_model` calls** (the
v26ar shape: stream order makes the swap atomic with respect to every step; no events, no side stream, no
re-capture — every address is fixed, only contents change). Budget ≤ `swaps_per_barrier` (default 4) per barrier.

**Counters.** `gt_stage` already scans the call's groups. Add an order-independent per-expert counter: one device
tensor `cnt` int32 [L, E] (one row per layer, fixed address) and per layer `g2e` int32 [E]; one program of the launch
does `atomic_add(cnt_row + g2e[group], 1)` over the call's ids. Counting is on only for decode-sized calls
(P ≤ 256 × top-k; a host-side constant per launch, so each captured graph size bakes its own value). Keep
`stage_emulated` as the contract and extend it; keep the mini-test bench working (optional args, old behaviour when
they are absent).

**Policy glue** (`runtime/gracetier_res.py`, new, no vllm import, CPU-testable with emulated copies): mirror of
hot/cold per layer built from the all-resident rowmap → `residency_policy.State`; every `drain_n` (64) barriers an
async D2H of `cnt` into pinned memory, the next barrier that finds it complete runs `policy.drain`; every barrier runs
`take_barrier` / `is_stale` / `commit` and actuates. Shipped v26 values for the rest (halflife 500, hysteresis 1.5,
min_abs 8, swaps_per_drain 2). Runtime control through a JSON file polled at drain: `freeze`, `restore_map`,
`baseline`, `dump`, `audit` (xor/sha digest of N experts at their current home against digests taken at boot →
one telemetry line with mismatches). Telemetry JSONL per drain: host steps, cold share of counted picks since the last
drain, swaps done/vetoed/stale, hot_moved, queue length.

**Wiring** (`fastpath_gracetier.py`): build `g2e`/`cnt` at finalize; create the manager after `load_model`; barrier =
a wrap of `GPUModelRunner.execute_model` that runs before the original (coexist with `a12_retire`'s wrap of the same
class — check order and say what you found). Config is read from `/fastpath/gracetier-res.json` if present (the
container forwards no arbitrary env; the card writes this file at build): `{"mode": "adaptive"|"off", "freeze": bool,
policy overrides}`. Absent file or `mode: off` = today's behaviour byte for byte (assert it in a test). Control and
telemetry live under `/prof/gracetier-res/` (`ctrl.json`, `telemetry.jsonl`): `/prof` is the one rw mount serve.sh
already makes (host `$GT/home/state/prof`).

## Card GT3 (`GT1_RES=1` mode in `bench/box/card-gt1.sh`, after G1; plus `bench/gt_res_drive.py`)
Build step: copy `gracetier_res.py` and `residency_policy.py` into `gracetier_rt/runtime/`; write
`gracetier-res.json` (`adaptive`, `freeze: true` at boot). Cells, each to `scratch/gt1-res-*-$STAMP.*`:

| cell | what | reported |
|---|---|---|
| R0 | frozen at the boot map: `w0_drive.py step` (3 reps) | ms/step, tok/step = the same-boot static control |
| R1 | unfreeze; C1 traffic = the 8 fixture prompts × 256 tokens, looped 90 s | telemetry: cold share and swaps per drain; T90 of cold share |
| R2a | freeze (map now fitted to the fixture); step cell (3 reps) — the step prompt is NOT in the fixture | held-out ms/step |
| R2b | unfreeze; step cell × 5 | adaptive ms/step, last 3 reps |
| R3 | freeze; fixture parity capture twice | self-identical N/8; vs the R0-map capture N/8 (reported, not gated) |
| R4 | `audit` (all moved experts + 64 random per layer) then `restore_map`, audit again, step cell | mismatches (must be 0); ms/step back at R0 within 2 % |
| R5 | unfreeze; `bench/c32_smoke.py`; server alive | pass/fail |

Verdict line: `GT3: R0 <ms> | R2a <ms> | R2b <ms> | T90 <s> | cold share <a> → <b> | audit <n> mismatches | C32 <pass>`.
Adopt bar (pre-registered): R2b ≤ 0.90 × R0, T90 ≤ 20 s, audits 0, C32 alive, R4 within 2 % of R0.
Also in the card: redact the API-key line in the `gt1-up-*.log` receipt before the run ends
(`sed -E 's/(API key \(save it\): )[0-9a-f]+/\1<redacted>/'`) — a key was pushed to the public repo once already.
The nsys mode (`GT1_NSYS=1`) and the plain GT1 gates must keep working unchanged.

## Tests
`tests/test_gracetier_res_cpu.py`: after N random swaps on mini shapes with emulated copies — every expert's bytes at
its current home equal golden; `row_map` is a permutation and `g2e` its inverse; SF rows follow their expert;
`restore_map` returns the boot map and bytes; counters attribute picks to experts across swaps; `mode: off` leaves
`gt_stage`'s arguments and the component's tensors identical to the current build. Extend
`tests/gpu/test_gracetier_gpu.py` with a box-only case: layer 10, all-HBM G = 384 reference vs the pool after 32
random swaps, `torch.equal` on the MegaMoE output for random routings at T = 1/6/16/64 (expected equal — the combine
is in top-k column order; if it is not, report the class with rows-exact / max|Δ| the way M6 did, do not hide it).
The Mac has no GPU: `device="cpu"`, fp32 on every host-side constructor; `.venv/bin/python -m pytest tests -q`
(run with `env -i HOME=$HOME PATH=/usr/bin:/bin` if a foreign PYTHONPATH breaks the import); two pre-existing
failures in `tests/test_prefill_anatomy_cpu.py` are not yours.

## Rules
No ssh, no docker, no network, no git commit. Never import vllm on the Mac. `reference/` read-only. New files plus
edits to: `runtime/gracetier_stage.py`, `port/garner/fastpath/fastpath_gracetier.py`, `bench/box/card-gt1.sh`, the
gracetier tests. No invented numbers; where Garner's or vLLM's behaviour is inferred rather than read, say so in a
comment and add a boot-time check that raises. Finish with: files changed, pytest counts, every inference you had to
make, and the line `A3-BUILD: code ready; cpu tests <n> pass; card=GT1_RES=1 bench/box/card-gt1.sh; cells=R0..R5`.
