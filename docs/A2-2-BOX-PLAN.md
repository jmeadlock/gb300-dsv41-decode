# A2-2 BOX PLAN — adaptive residency on the rebased runtime (branch `a2-2`, 1 row and the 6-row DSpark loop)

For the orchestrator. Everything below is a command to run on the Station from a checkout of branch `a2-2` (call it
`$W`; the merge with main is in the working tree, uncommitted — commit it first). Receipts go to `scratch/` (never
`docs/`); copy the ones worth keeping into `docs/` afterwards. No Mac numbers are in this file: every figure quoted is
from an existing receipt, named.

## 0. What the rebase changed (why the old A2-2 commands are not enough)

main moved under the branch: the 6-row dg-inplace step now stages cold experts by a **compact cold index** into
`S = 16` spare slots per layer (`DEFAULT_SPARE_SLOTS_ROWS`, M1-T10-PORT), keeps the hot experts' packed scale factors in
a **pool-owned copy** (`sf_store "pool"`, 12.8 GiB, built once), counts overflow on the device, and the DSpark loop
(`runtime/dspark_loop.py`) replays the step graphs back to back with no host decision. The residency manager was
adapted, not rewritten:

| point | before | now |
|---|---|---|
| staging slots under the manager (`residency_stage`) | `topk * rows` (36 at 6 rows) | `S` = the step's compact budget **kept for cold staging: 8 at 6 rows, `topk` at 1 row** (A2-2 FIX; was 16 = all of the pool's spare slots); residency slots start at `H + S`, never inside `H..H+S-1` (the dedup kernel addresses those) |
| packed SF of a promoted expert | `step` store re-packs it every step; `resident` store written by the manager | also the **`pool` store** (6-row default): the copy stream writes the packed SF into the pool copy before the copy event, `resident_sf` now covers `pool` as well as `resident` |
| slot budget (A2-2 FIX, box r1 refused with "no spare HBM slot affordable") | free HBM − stage − KV − 8 GiB reserve → NEW slots only | **residency promotes into the pool's own free spare slots first** (pool 16 − staging 8 = 8 per layer, already in HBM: no allocation, no growth, no SF re-key, whatever the free HBM). Only a shortfall (explicit `--residency-spare` above that, or a pool with nothing free → `spare_cap` 4) is NEW HBM, allowed only while free HBM − 8 GiB reserve − KV (− the re-keyed packed-SF copy + 6 GiB headroom, when the pool grows) covers it; else it is capped with a warning, and refused only if zero slots remain. One new slot costs `expert_bytes + one SF group` per layer (0.74 GiB) |
| per-step hook | `Generator.step` / `step_rows` | also `DSparkLoop.step_once` → `rm.after_step(live=b.live)` after each step's replay (steps after `done` are predicated no-ops: the device `live` flag keeps them out of the hit counters); the hook runs after the step's end event, its GPU time is reported separately (`mgr_ms`) |
| `bench/headline.py` | no residency | `--residency on --residency-drain N …` (same knobs as `first_light`) plus `--residency-stage K` (staging slots kept, default `--spare-slots` else 8) and `--pool-spare N` (pre-grow the pool to ≥ N, default 16 with residency), `--residency-ref <off headline JSON>`, `--residency-audit`; report key `residency` |

**Event ordering at 6 rows (the "do not move an expert a captured graph reads in the same replay" rule).** The graphs read
pointer-table *contents* (`ptabs`, `ptab6`, `gtab`, the fused glue's stacked `_ptab6`, `cold_tab`, `pool.loc`) and the
pool slots / packed SF they name; the tables' *addresses* never change, so no recapture. The manager writes the tables
only between two replays, with scalar `fill_` kernels on the replay stream, and only after `cur_stream.wait_event(copy
event)` — so a replay enqueued later sees the new contents and one enqueued earlier is complete before a table
changes. A demoted expert's slot rejoins the free list carrying an event recorded right after its table writes, and the
next copy into it (on the copy stream) first waits on that event: every replay enqueued before the publish has finished
reading the old bytes by then. Promotion order: copy weights, scales, packed SF (pool/resident) → record event →
(later) publish. Nothing in the host loop syncs: the loop's 8-step lookahead and the manager's `Event.query` publish
rule keep the queue full. Gate: GPU test 5 below (swaps racing queued 6-row replays, n_spare 1, ~1 GiB of dummy H2D
ahead of each swap, plus the `publish_before_copy` negative control that must diverge).

## 1. Common docker line

```
W=<the a2-2 checkout on the Station>
REF=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
D="docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
  --cap-add IPC_LOCK -v $W:/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v $REF:/ref:ro \
  -v $HOME/dsv41/w0/jit-cache/tilelang:/root/.tilelang -v $HOME/dsv41/w0/jit-cache/triton:/root/.triton \
  -v $HOME/dsv41/w0/jit-cache/nv:/root/.nv -v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
  --entrypoint python3 vllm/vllm-openai:v0.31.0"
mkdir -p $W/scratch
```
(`--cap-add IPC_LOCK` + `memlock=-1` are required: the demotion pool is `cudaHostRegister`'d Grace memory, 40 layers ×
64 slots × 17.93 MiB = 44.8 GiB; `check_mem_available` keeps 16 GiB of MemAvailable free.) The same
fixtures as every earlier window: `w0/results/phase1/parity-A.jsonl` (8 prompts) inside `$W`.

## 2. GPU test (mini checkpoint; ~10 min incl. JIT)

```
$D tests/gpu/test_residency_gpu.py 2>&1 | tee $W/scratch/a22-gpu-test.log
```
Expect `12/12 not failed` (the six 1-row tests of the original A2-2 plus six 6-row tests:
`test_residency_rows6_on_equals_off_{ref,dg_inplace_pool_sf,dg_inplace_step_sf,dg_inplace_fused_glue}`,
`test_residency_rows6_swaps_while_replays_queued_event_ordering` and
`test_residency_rows6_promotes_into_the_pools_spare_slots_without_growing` (stage 8 + the pool's 8 free slots, no
growth; prints `overflow {...}` — if S = 8 overflowed on the mini model it says NOTE and skips only the equality
assert); the count printed on the last line is authoritative). r1 (`docs/residency-gpu-test-r1-20261107.log`) was 8/11:
the three `read target must be host memory, got cuda:0` failures were the audit's golden digest reading shards into a
tensor allocated under the cuda default device (fixed: `device="cpu"`). Gate: no `FAIL`; the event-ordering
tests print their negative control diverging (`negative control diverged at n/N`); the 6-row dg-inplace tests print
`sf store pool` / `sf store step` respectively and `audit … ok`. A `SKIP (deep_gemm unavailable)` on the dg-inplace
tests is NOT a pass — the library image has it; investigate.

## 3. 1-row first light: `--residency on` byte-identical to off

Flags F (the 10.52 ms path; default fp32 impl, as the earlier A2-2 windows):

```
F="--model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph --moe-impl dg-inplace --fp8-impl gemv \
   --glue-impl fused --attn-impl flashmla --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --max-new 256 \
   --reps 2 --step-cell 512 --no-fill-uninit"
$D bench/first_light.py $F --out scratch/fl-a22-off.jsonl --json scratch/first-light-a22-off.json 2>&1 | tee scratch/fl-a22-off.log
$D bench/first_light.py $F --residency on --residency-ref scratch/first-light-a22-off.json --residency-audit \
     --out scratch/fl-a22-on.jsonl --json scratch/first-light-a22-on.json 2>&1 | tee scratch/fl-a22-on.log
```
(Both boot the real model: ~60 s each + prompts.) Report from `scratch/first-light-a22-on.json`, key `residency`:
- **gate** `vs_off.rep1_files_byte_identical == true`, `vs_off.step_cell_tokens_equal == true`, `errors == []`;
- **gate** `audit.ok == true` (sha256 of every moved expert, HBM and Grace copies, vs the checkpoint shards);
- **gate** `swaps_per_step.max <= 4` (cap); report `swaps_per_step.mean` (design budget ≤ 1.25/step), `swaps`;
- **gate (G3)** `vs_off.step_ms_delta_pct <= 1 %` (step-cell ms/step, on vs off; the manager's kernels run between
  replays, outside the graph);
- report `cold_share_first_drain` → `cold_share_last4_drains`, `step_cell_cold_share_first64` → `last64`, `hbm`
  (`spare_GiB`), `n_spare`, `budget` (the headroom calc printed at start: `[residency] HBM free … affordable …`).
Seed-map reference for the cold share: 0.9 % on the step cell, 3–4 % on the fixture prompts (out of sample;
`docs/first-light-seed-20261106.json`, LEDGER 22:38).

## 4. Shifted-workload probe (the policy must *adapt*)

Same 8 prompts, a deliberately wrong map: `--hot-per-layer 295` with no `--map` = ids 0..294 (cold share 0.237/step,
LEDGER 14:59). Control = residency off on the same map; probe = residency on. A longer run gives the policy more
drains: `--reps 4` (4 × 8 × 256 tokens ≈ 8,200 steps ≈ 128 drains at drain_n 64).

```
S="--model-dir /ref --hot-per-layer 295 --mode graph --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused \
   --attn-impl flashmla --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --max-new 256 --reps 4 \
   --step-cell 512 --no-fill-uninit"
$D bench/first_light.py $S --out scratch/fl-a22-shift-off.jsonl --json scratch/first-light-a22-shift-off.json 2>&1 | tee scratch/fl-a22-shift-off.log
$D bench/first_light.py $S --residency on --residency-ref scratch/first-light-a22-shift-off.json --residency-audit \
     --out scratch/fl-a22-shift-on.jsonl --json scratch/first-light-a22-shift-on.json 2>&1 | tee scratch/fl-a22-shift-on.log
```
Report (`residency` in the on JSON): `cold_share_curve` ([host step, cold share] per drain — the curve over the run),
`cold_share_first_drain` → `cold_share_last4_drains`, `step_cell_cold_share_first64` → `last64`, `rep_cold_share`
(per prompt, per rep), `swaps_per_step`, `stats.d2h_demotions`. **Expected from the design, not a promise:** the cold
share starts near 0.237 (first drains see the wrong map) and falls toward the seed-map figure (0.9–4 %) as swaps
accumulate; a flat curve means the policy is not adapting (then check `policy.pending` / `stats.vetoed_*` in the JSON:
`vetoed_no_slot` ⇒ `--residency-spare` too small, `vetoed_no_demotion_slot` ⇒ `--residency-demote` too small).
`vs_off` byte-identity must hold here too (bytes are the checkpoint's wherever they live).

## 5. 6-row headline with residency on (the number that matters)

Baseline to beat: run 4 (`docs/headline-20261107d.json`): **22.32 ms/step** (verify 19.41 + draft 2.90), 2.971 tok/step,
**7.51 ms/token**; the bar is 7.36 ms/token ⇒ at that yield a step ≤ **21.87 ms** (7.36 × 2.971, arithmetic only; the
yield moves with the near-tie continuations). Cold share on the seed map at 6 rows in that run: 33,587 cold of 500,884
routed unique-expert calls = **6.7 %** (receipt `routed`; the 1-row figure is 0.9 %), so the 6-row step is where
residency has something to remove. Off then on, back to back (same boot conditions). **The A/B needs the same staging
size S in both arms**: an S-overflow gives wrong logits for that step, and residency changes how many experts are cold,
so arms with different S can diverge for that reason alone. Default pair (S = 8 both, pool 16 both — the on run uses the
8 free slots with no new allocation, which is what fits in the real build's 8.6 GiB free):

```
H="--model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed \
   --spare-slots 8 --pool-spare 16 --max-new 256"
$D bench/headline.py $H --json scratch/headline-a22-off.json 2>&1 | tee scratch/headline-a22-off.log
$D bench/headline.py $H --residency on --residency-drain 64 --residency-ref scratch/headline-a22-off.json \
     --residency-audit --json scratch/headline-a22-on.json 2>&1 | tee scratch/headline-a22-on.log
```
(`--map` defaults to the seed map. `--spare-slots 8` is the step's staging S in both arms (`--residency-stage 8` is the
same thing on the on arm); `--pool-spare 16` pre-grows the pool ONCE to 16 slots per layer before any build in both arms
(the pre-grow is the only growth: the manager's own `add_spare_slots` is then a no-op, so no layer block is re-allocated
while an older build still references it — that re-growth OOMed M1-T10-FIX round 3). The on run's log line must read
`pool holds 16 spare/layer, 8 kept for staging, 8 free in place + 0 new; n_spare 8` and `[pool]` must show no second
growth. **Overflow check:** with S = 8 the earlier M1-T10-PORT measurement was 123/1018 overflowing steps (before
residency, seed map); the report's `overflow` field (and the residency line, `overflow steps N`) says what this build
does. If the OFF arm overflows at S = 8, its logits are wrong on those steps and the A/B is void: rerun the pair with
`--spare-slots 12 --pool-spare 16` (residency gets 4) or `--residency-stage 12`; the old baseline (S = 16, no residency:
`--spare-slots 16`) stays the headline number to beat. Each run: boot ~60 s, 1-row truth per prompt, then the loop at
2 graphs and 1 graph.)
Report from `scratch/headline-a22-on.json`:
- `headline.two_graphs.ms_per_step.mean` (graph only, as every earlier headline), `ms_manager_per_step.mean` (the
  manager's GPU time between replays: count kernel + table writes), **`ms_per_step_incl_manager`** (the honest on-number
  to put next to 22.3), `ms_verify_graph`, `ms_draft_graph`, `tokens_per_step`, `ms_per_token`;
- `residency.swaps_per_step` (mean, max vs cap 4), `residency.cold_share_curve` (per drain) and
  `residency.cold_share_by_seq` (the step counters' cold share per sequence in run order: prompt × fast/parity rep),
  `residency.cold_share_first_drain` → `cold_share_last4_drains`, `residency.hbm.spare_GiB`, `residency.n_spare`,
  `hbm_after_build_GiB` (off run: 9.2 GiB free; each spare slot costs ≈ 0.74 GiB: 40 × (17.93 MiB + 1.05 MiB packed SF));
- **gates**: `overflow` `{"steps": 0, "experts": 0}` (an overflowed layer-step has wrong logits; residency only lowers the
  cold count, but check), `residency.vs_off.all_accepts_equal == true` and `all_tokens_equal == true` (the emitted
  stream and accept lengths are the checkpoint's regardless of residency — the on run must match the off run exactly,
  prompt by prompt), `replay_ok` 8/8 (self-replay incl. 1 graph == 2 graphs), `residency.audit.ok`, `errors == []`;
- **loud warning to look for** in the on log: `[residency] WARNING the step fell back to sf_store 'step'` (also
  `residency.sf_store.got` in the JSON). The pool SF store refuses to build with < 6 GiB free after its 12.8 GiB; the
  manager's spare slots eat headroom. If it fell back, the on/off A/B is confounded (hot scales re-packed every step) —
  rerun with `--residency-spare 2` (or 1) and/or `-e DSV41_SF_POOL_HEADROOM_GIB=4`, and report which.
- `RuntimeError: residency: no residency slot` at start ⇒ the pool holds nothing free beyond the staging (e.g.
  `--spare-slots 16` with `--residency-stage` unset, so S = 16 = the whole pool) AND no new slot is affordable. Use the
  default pair above (S = 8 of 16), or give it room: `--residency-stage 12` / `--spare-slots 12`. New HBM is only
  asked for by an explicit `--residency-spare` above the pool's free slots (`WARNING wants N new HBM slots/layer, only M
  affordable` — it then uses M); report the printed budget line.
- Optional 6-row shifted probe (only if §4 adapts): `H` with `--hot-per-layer 295` and no `--map`, off/on pair; expect
  more cold unique experts per layer-step, so watch `overflow` (S = 8) — an overflow there is a probe artefact, not a
  residency bug, and the run's logits are then not comparable.

## 6. Reading the result (what each outcome means)

| outcome | meaning |
|---|---|
| §3 byte-identical, swaps ≤ 4/step, step Δ ≤ 1 % | residency is free at 1 row on the seed map; the mechanism is sound on the real model |
| §4 cold share falls from ≈ 0.24 | the policy adapts to a workload shift — A2-2's reason to exist |
| §5 `ms_per_step_incl_manager` < off, cold share < 6.7 % | residency pays at 6 rows: report ms/step and **ms/token** (publish it either way) |
| §5 on > off | the manager's cost or the HBM it took (SF fallback?) outweighs the saved C2C bytes; report `ms_manager_per_step` and the fallback flag before concluding anything |
| §5 tokens differ on vs off | a bug — residency must never change an emitted token; stop and report `residency.vs_off.prompts` |

## 7. Files changed by the rebase (for the reviewer)

`runtime/residency.py` (`n_stage_for(spare_slots)`, `residency_stage`, `sf_store_for`, pool-SF write, budget,
`after_step(live)`, CLI helpers), `runtime/generate.py` (merge of `spare_slots` + `residency`; the step is built with
the staging the manager keeps), `runtime/expert_pool.py` (`loc_cpu` explicitly on the host),
`runtime/dspark_loop.py` (per-step hook, `mgr_ms`), `bench/headline.py` (flags, report, cold share per sequence),
`bench/first_light.py` (merge only), `runtime/README.md`, `tests/test_residency_rebase_cpu.py` (new),
`tests/test_residency_cpu.py` / `test_residency_rows_cpu.py` (static_kernels `stage_rows`, S), `tests/gpu/test_residency_gpu.py`
(6-row tests). A2-2 FIX (box r1): `golden_from_index` and `ExpertPool.loc_cpu` made host-explicit, the pool-first slot
budget, `--residency-stage` / `--pool-spare`, tests `test_residency_stage_rule`, `test_manager_promotes_into_the_pools_free_spare_slots_without_growing`,
`test_host_tensors_carry_device_cpu_explicitly` and the GPU no-growth test. Mac suite: `.venv/bin/python -m pytest tests/ -q`.
