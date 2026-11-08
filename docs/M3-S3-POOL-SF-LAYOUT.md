# M3-S3 — the scale factors beside their weights: what DeepGEMM allows, what was built, what it frees

Handoff: `docs/HANDOFF-M3-S3-POOL-SF-LAYOUT.md`. Branch `m3-s1` (worktree `gb300-dsv41-decode-wts1`), 2026-11-08 ~01:40 CST.
Tags: [M] measured, [D] derived from receipts by the arithmetic shown, [E] modeled, [A] assumed.
Instrument: `bench/pool_layout_bench.py` (CPU: `bytes`, `estimate`, `contract`; box: `probe`).
CPU receipt: `scratch/pool-layout-all-20261108.json` (+ `.log`).

## Bottom line

- **Physically adjacent weights+SF in one pool slot is not addressable by DeepGEMM's kernel without a repack.**
  `sm100_fp8_fp4_gemm_1d1d` reads B and SFB through two separate 2-D TMA maps (two base pointers); the group offset
  is folded into one coordinate of each map, so each tensor's group stride is implicit: B = N × B row pitch, SFB =
  K/128 × SFB row pitch. A slot = [weights | SF] would need both strides equal to the slot size, which forces either
  dead bytes in every row or K-major SF (the kernel wants MN-major). Evidence below.
- **What is feasible, and built (`$DSV41_POOL_SF=beside`, default `separate` = today):** the pool OWNS every slot's
  SF in DeepGEMM's packed MN-major layout (`pool.sfp[l][p]`, int32 [H+S, N, K/128]), indexed by the SAME group
  number as the weight pool. One `m_indices` entry addresses both; **no second pointer table, no per-call repack**.
  The K-major e8m0 duplicate of the hot scales (12.15 GiB) moves to Grace (registered host buffer; the reference
  prefill reads it over C2C, the decode step never does). Every layer is then "pool": **no hot SF is copied, ever**;
  the stage moves cold weights + cold SF only (n_work = the spare budget, 16, instead of R·k = 36).
- **HBM freed [D]: 5.18 GiB** vs tonight's adopted arm C (SF bytes in HBM 18.16 → 12.98 GiB at pool-spare 20).
  Enough to return to **8 residency slots (pool-spare 24) and still be 2.38 GiB better than C** (free after build
  ~3.9 GiB [E] vs C's 1.54 [M]); pool-spare 27 (11 residency slots) is the break-even.
- **Expected [E]: ~18.7 ms/step** (range 18.5–18.8) at pool-spare 20 or 24, from tonight's measured per-pooled-layer
  gain (0.013–0.023 ms) × the 23 layers still re-copying. M3-S2's per-layer model (33–37 µs) would give 18.2–18.3;
  B and C delivered ~40 % of that model per layer, so 18.2 is the ceiling, not the forecast.
- Values never change: every gate is torch.equal (CPU, 9/9 new tests) and the GPU gate file is ready for the box.

## (1) Who reads the pool's SF layout (enumerated first)

| reader | what it reads | under `beside` |
|---|---|---|
| `runtime/expert_pool.py` | owns `hbm[l][p] = (w, s)`, `hbm_full`, now `sfp`, `hot_sf_host` | **changed** (the switch, `pack_hot_sf`, `packed_sf`, growth of `sfp` in `add_spare_slots`, `bind` makes `s` a UVA view) |
| `runtime/dg_inplace.py` | `pool_packed_sf` (pool store), `layer_tables` (GEMM's B + SFB, the stage's 6 destinations), `check_layout` / `check_rowmatch` (layer-0 packed SF), `describe` | **changed**: store forced to "pool" for every layer, tables = `pool.sfp`, no HBM budget check (the bytes are the pool's), build checks read `pool.sfp`, `[dg-inplace] sf_store pool: pool SF layout beside ...` on stderr |
| `runtime/moe_rows.py` | dedup work list: hot slots listed iff `b.step_sf(t)` | unchanged code: `step_sf` is False on every layer → cold-only lists |
| `runtime/staging_overlap.py` | `n_work = b.stage_n_work(t)` per layer | unchanged code: n_work = spare budget on every layer |
| `runtime/static_kernels.py` (not owned) | stage kernels write `dsts` = 3 weight + 3 SF bases at slot g | unchanged: the destinations are `hbm_full` + `sfp`, same layout as the pool store's copy |
| `runtime/residency.py` | slot views (weights `hbm_full`, scales `hbm[l][p][1]` / its own `sspare`), promotion copies, packed SF of promoted slots (`resident_sf`), budget, digests | **changed**: `sspare` e8m0 in a registered host buffer, budget = weights + one packed group per slot (no rebuilt-copy headroom), `sf_store_for` → "pool", digests read host views (a `.cpu()` of a UVA alias fails on the box, `runtime/host_tier.py`) |
| `runtime/loader.py` | reads hot experts into `hbm_dst`, cold into the host pool, verify | **changed**: hot scales read straight into `hot_sf_host`, registered with the host tier, `pack_hot_sf()` after `bind`, verify hashes the host view, `pool_HOST_hot_scales` in `bytes` |
| `runtime/static_step.py` | pointer tables `ptabs` from the Experts' bound params (checked against `pool.hbm[l][p][1][slot]`) | hook only: `describe()["pool_sf"]`; the tables still name the e8m0 homes (now host addresses) |
| `runtime/dg_mega.py`, `dg_moe.py`, `fused_glue.py` gathers (not owned) | hot e8m0 through `ptabs` / `pool.hbm[l][p][1]` | work unchanged, read over C2C (dg-masked / ref / dg-mega are not the headline path) |
| `runtime/draft_step.py` | MTP expert stacks | not the pool: unaffected |
| `runtime/generate.py` (not owned) | `sf_store_for(moe_impl, rows)` | no change needed: `sf_store_for` falls back to `$DSV41_POOL_SF` |

## (2) The layout memo

### What DeepGEMM requires (read from the vendored device code, `reference/vllm-0.31.0/deep_gemm/include/deep_gemm`)

How the SF reaches the kernel today: `DGInplaceBuffers.gemm_inplace` calls
`m_grouped_fp8_fp4_gemm_nt_contiguous((xq, xs), (W [G, N, K/2] int8, SF [G, N, K/128] int32 MN-major), d, m_indices)`
— B and SFB are two tensors; the host builds one TMA descriptor per tensor (host csrc not vendored: [A] for the host
checks; the device addressing below is the source). `transform_sf_into_required_layout` is NOT used (the runtime
packs once: `pack_words` = the checkpoint's 4 e8m0 bytes as one little-endian int32, then MN-major).

- `impls/sm100_fp8_fp4_gemm_1d1d.cuh:234-238`: for K-major B in an M-grouped contiguous GEMM,
  `n_idx = get_global_idx<kBWithGroupMNOffset>(shape_n, BLOCK_N, n_block_idx, m_block_idx)`;
  `scheduler/gemm.cuh:172-174`: `offset = grouped_layout[m_block_idx * BLOCK_M]; return offset * shape_dim + ...`
  → B row coordinate = **group × N + n**, one 2-D map over [G·N rows, K/2 bytes] (`:286-291`, `tma::copy<..., kIsBatchedMM>`
  with `kIs3DTMA = false` for this GEMM type, `common/tma_copy.cuh:18-35`). B's group stride is therefore
  **N × row pitch**, implicit.
- `:270-274`: `sfb_n_idx = n_block_idx * BLOCK_N` (no group term), `sfb_k_idx = get_global_idx<true, SF_K>(shape_sfb_k,
  SF_BLOCK_K, ..., m_block_idx)` → SFB coordinate = (n, **group × K/128 + k-word**), one 2-D map with N contiguous
  (MN-major; `SF_BLOCK_N` = BLOCK_N aligned to 128 at `:94`). SFB's group stride is therefore **K/128 × SF row pitch**,
  implicit.

Slot bytes at the real shapes (`bench/pool_layout_bench.py bytes` [D]):

| proj | N × K | B row pitch | B group stride | SFB rows × pitch | SFB group stride | SF as extra B rows | pitch if SF rode each B row |
|---|---|---|---|---|---|---|---|
| w1, w3 | 2304 × 5120 | 2560 B | 5,898,240 B | 40 × 9216 B | 368,640 B | 144 rows (6.25 % of N) | 2720 B (16-B ok, but K-major SF) |
| w2 | 5120 × 2304 | 1152 B | 5,898,240 B | 18 × 20,480 B | 368,640 B | 320 rows | 1224 B (**not** a 16-B multiple) |

Per expert slot: 17,694,720 B weights + 1,105,920 B SF (e8m0 = packed int32: the same bytes).

Why "beside" cannot mean "adjacent in one slot" without a repack:
- [weights | SF] as one slot needs W's group stride = SF's group stride = the slot size. W's is N × pitch, so the SF
  would have to live in widened B rows (pitch 2720 / 1224 B): that stores the SF **K-major per row** (the kernel's
  SFB map needs N contiguous: MN-major) and w2's 1224 B pitch is not even TMA-legal (16-B multiples).
- Treating the SF as 144 / 320 extra "rows" of B (`shape_n` = N + 6.25 %) computes 6.25 % more output columns of
  garbage per GEMM (≈ +0.28 ms/step at the 4.45 ms expert-GEMM total [D]) and changes `d`'s shape for the glue.
- Interleaving SF rows inside dead space between B rows needs ≥ 9216 B of dead space per 2560 B row.
Each is either a repack per call, wasted HBM, or wasted MMA. So: not this.

The cheapest alternative — and what was built: **slot-parallel SF**. `sfp[l][p]` is a pool-owned tensor with a
fixed per-expert stride (368,640 B per projection) addressed by the same group index as `hbm_full[l][p]`. The GEMM
needs nothing else (SFB base + group × stride is exactly its addressing); the staging launch already carries six
destination bases (`dtab`) and writes slot g of each; residency writes slot g of each. No second pointer table.

### HBM accounting [D] (40 layers × 295 hot; today = arm C: pool-spare 20, 16 staging + 4 residency, 17/40 pooled)

SF bytes in HBM today: hot e8m0 12.154 + residency e8m0 0.165 + partial pool (17 layers × 315 groups) 5.515 + step SF
0.324 = **18.158 GiB**. Under beside: 40 × (295 + pool-spare) × 1,105,920 B, nothing else.

| beside, pool-spare | residency slots | SF in HBM GiB | extra weight slots vs C | HBM vs C | free after build [E] | e8m0 on Grace |
|---|---|---|---|---|---|---|
| 20 | 4 | 12.978 | 0 | **−5.181** | 6.72 | 12.32 |
| 24 | 8 | 13.142 | +2.637 | **−2.379** | 3.92 | 12.48 |
| 26 | 10 | 13.225 | +3.955 | −0.978 | 2.52 | 12.57 |
| 27 | 11 | 13.266 | +4.614 | −0.278 | 1.82 | 12.61 |
| 28 | 12 | 13.307 | +5.273 | +0.422 | 1.12 | 12.65 |

free after build [E] = arm C's measured 1.541 GiB minus the delta. One residency slot = 40 × (17.69 + 1.11) MB =
0.700 GiB of HBM under beside (its e8m0 copy is on Grace). Host: +12.3–12.6 GiB pinned (494 GiB Grace; the host tier
already pins ~200 GiB of Engram + cold experts).

Alignment: the hot-scale host blocks are 256-B aligned per (layer, projection) and every expert row is 184,320 B (16-B
multiple: the gather's vector loads hold); the buffer is one registration chunk (12.3 GiB < 64 GiB), so no memcpy
crosses a chunk; the residency host buffer is page-aligned, one chunk.

## (3) Implementation (bit-identical by construction; `separate` is the default and today's code path)

- `runtime/expert_pool.py`: `$DSV41_POOL_SF` / `ExpertPool(sf_layout=)`; under beside: `hot_sf_host` (HostBuffer
  "hot_scales"), `hbm[l][p][1]` = its host view until `bind(cuda_views=True)` makes it the UVA alias, `sfp[l][p]` =
  int32 [H, N, K/128] MN-major allocated with the pool, `pack_hot_sf()` (int32 transpose of the host bytes, H2D per
  layer × projection), `packed_sf(l)` (lazy), `add_spare_slots` grows `sfp` with the weights (hot groups carried, spare
  zero, counted in the bytes added), `hbm_bytes()` = weights + packed SF, `describe_sf()`.
- `runtime/loader.py`: hot scales read into the host buffer (parallel positional reads, like cold), registered with
  the host tier, `pack_hot_sf()` after `bind` (`timings["pack_hot_sf"]`), verify hashes the host view.
- `runtime/dg_inplace.py`: `pool_packed_sf` returns the pool's own copy; `DGInplaceBuffers` under beside: store
  "pool" on every layer whatever was requested (`sf_store_why` says so), no `_check_sf_hbm`, no step SF tensor,
  build checks (`check_layout`, `check_rowmatch`) on `pool.sfp`, `describe()["pool_sf_layout"]`.
- `runtime/residency.py`: `sf_store_for(..., pool=None)` → "pool" under beside; `sspare` in a registered host buffer
  ("residency_scales", released by `close()`); budget per slot = weights + one packed group, no rebuilt-copy
  headroom; `slot_view(..., host=True)` for digests; `hbm_bytes()` reports the host e8m0.
- `runtime/static_step.py`: `describe()["pool_sf"]` (hook only).
- `runtime/moe_rows.py`, `runtime/staging_overlap.py`: no change needed (per-layer `step_sf` / `stage_n_work` from
  M3-S2 carry it).

## (4) Gates

CPU, `tests/test_pool_layout_cpu.py` (9 tests, all pass [M]):
- switch parsing; the expert pool's packing helpers == dg_inplace's.
- **pool bytes expert by expert** (two loads of mini4): weights identical; the K-major e8m0 of every hot slot
  identical (now in the host buffer, the Experts' scale params alias it); every hot packed slot == `DI.pack_sf` of
  today's bytes and unpacks to the same codes; cold bytes identical; the loader's verify passes; HBM bytes equal at
  load; growth by 3 slots carries hot groups, zeroes spare, adds exactly 40·3·(weights + SF) bytes.
- **6-row static step beside == separate**, kernel and kernel-overlap, against the separate step store AND pool
  store: torch.equal on every row's logits, main_hidden, caches (engram over the written positions — two separately
  loaded models; the tail is each model's own `torch.empty`), spare weight slots, spare SF groups. Beside's tables ARE
  `pool.sfp`, every layer "pool", n_work = spare budget, the last work lists cold-only, and a recording resolver over
  both emulated stage paths shows **no hot-scale address is ever read under beside** (the same recorder sees them
  under the separate step store and not under the separate pool store: it has teeth).
- negative control: one hot packed group corrupted in `pool.sfp` changes the logits.
- **1-row == 6-row per row** under beside (row-invariant torch mode), the 1-row build sharing `pool.sfp`.
- **residency promotion under beside**: forced + policy swaps on == off (and off beside == off separate); resident SF
  registered for every layer (= `pool.sfp`); residency e8m0 in the host buffer; budget per slot = weights + packed;
  audit vs the checkpoint golden ok; every HBM-resident expert's packed group (promoted ones included) == the packing
  of its checkpoint e8m0; close releases the buffer.
- **staging kernels**: m1_stage_rows and s1_stage_proj ×3 bodies (triton shim) on a beside pool's own storage ==
  the contract; hot groups untouched; the hot e8m0 is not registered with the shim (a read would raise).
- bench CPU subcommands (numbers above).

GPU, `tests/gpu/test_pool_layout_gpu.py` (box; skips on the Mac): pool bytes on cuda (incl. the UVA alias read on the
device, verify); 6-row beside == separate, kernel / kernel-overlap, **eager and captured graph**; 1-row: the
rowmatch build check passes on `pool.sfp` + 1-row beside == separate (6-row vs 1-row full-step agreement reported);
residency = `tests/gpu/test_residency_gpu.py`'s 6-row graph gate under beside (requested step and pool); the device
stage kernels on a beside pool; the bench probe quick (GEMM bit-equality on real shapes).

Mac full suite: `.venv/bin/python -m pytest tests/ -q` = **1285 passed, 202 skipped, 24 failed — the identical 24 fail on
an untouched `git archive HEAD` copy** (1276 passed, 196 skipped, 24 failed; `scratch/pytest-m3s3.log` vs
`scratch/pytest-m3s3-baseline-HEAD.log`, `diff` of the FAILED lines empty): +9 passed / +6 skipped = this round's
two files. The touched modules' suites (staging_overlap, dg_inplace, residency, residency_rows, residency_rebase,
loader_mini, moe_rows_t10, dg_rowmatch): 210 passed, the same 2 stale failures as HEAD (below).

## (5) `bench/pool_layout_bench.py`

- `bytes` / `estimate` / `contract` (CPU): the tables of this memo; `contract` [M] counts the rows the stage contract
  reads on a tiny pool: separate + step lists `{cold.scale 9, cold.weight 9, hot.scale 15}` (n_work 12), beside
  `{cold.scale 9, cold.weight 9}` (n_work 4), hot SF groups untouched.
- `probe` (box): A build — real-shape pools (`--layers 2 --hot 295 --spare 20`) in both layouts: construct, fill,
  register, pack, grow times; `pool.hbm_bytes()`; the process's HBM delta. B gemm — DeepGEMM contiguous on
  `pool.sfp` vs on a copy packed from separate's e8m0, w1 and w2 shapes, bitwise. C stage — per-layer-call µs with
  the hot SF listed (today's step layers) vs cold only (every layer under beside), serial and overlapped, from
  `staging_bytes_bench.py probe` (its `pooled_layer_saves_us` is beside's per-layer saving).

## (6) Box commands (orchestrator; lease-serialized like `bench/box/m3s2.sh`, same `dk` wrapper, image, `$W` = the m3-s1 worktree)

```
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 20 --residency-drain 64 --max-new 256"
FL="bench/first_light.py --model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit"
# t  GPU gates: the new file (all PASS) + the default path's (unchanged code under separate)
dk s3t --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_pool_layout_gpu.py > $S/s3-gpu.log 2>&1
dk s3u --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py > $S/s3-gpu-so.log 2>&1
# p  probe: build bytes/times, GEMM equal, stage us with / without hot SF
dk s3p --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/pool_layout_bench.py probe --out scratch/pool-layout-probe-$(date +%Y%m%d).json > $S/s3-probe.log 2>&1
# a  baseline (= arm C), same window
dk s3a --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s3-A.json > $S/headline-s3-A.log 2>&1
# b  beside, same pool-spare 20 (isolates the layout): expect the log line "sf_store pool: pool SF layout beside (40/40 layers"
dk s3b -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s3-B.json > $S/headline-s3-B.log 2>&1
# c  beside + 8 residency slots
dk s3c -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 ${HL/--pool-spare 20/--pool-spare 24} --json scratch/headline-s3-C.json > $S/headline-s3-C.log 2>&1
# f  1-row stream under beside: byte-identical to the separate run (adoption rule)
dk s3f0 --entrypoint python3 vllm/vllm-openai:v0.31.0 $FL --out scratch/fl-s3-sep.jsonl > $S/fl-s3-sep.log 2>&1
dk s3f1 -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 $FL --out scratch/fl-s3-bes.jsonl > $S/fl-s3-bes.log 2>&1
cmp -s $S/fl-s3-sep.jsonl $S/fl-s3-bes.jsonl && echo "1-row stream byte-identical" || echo "1-row stream DIFFERS"
grep -h "\[dg-inplace\] sf_store\|hbm_after_build" $S/headline-s3-*.log
```

Adopt per arm only with: GPU gate PASS, streams 8/8, self-replay 8/8, overflow 0, the 1-row stream byte-identical.
Read the boot time too (`boot.timings_s`: `pack_hot_sf` is new; registering 12.3 GiB more host memory adds to
`register`) and, if the harness reports it, prefill time (hot scales now read over C2C by the reference prefill).

## (7) Expected ms [E] — the arithmetic

Tonight's arms (two_graphs ms_per_step mean, [M]): A 19.306 (0 layers pooled, 8 residency), B 19.216 (7 layers,
8 residency), C 19.064 (17 layers, 4 residency). Per pooled layer: A→B 0.090/7 = **0.0129 ms**; A→C 0.242/17 =
**0.0142** (charging the 4 lost residency slots nothing); B→C 0.152/10 = 0.0152, or 0.0232 if the 4 slots cost 0.02 ms
each (M3-S2 (2)'s two_graphs average). Beside pools the remaining **23 layers**: 23 × 0.0129..0.0232 = 0.30..0.53 ms
off 19.064 → **b (pool-spare 20): 18.53–18.77, central 18.74** (0.0142/layer). c (pool-spare 24, 8 residency
slots back): a further 0..0.08 ms → **18.45–18.77, central 18.70**. M3-S2's per-layer model (33.1–36.6 µs) × 23 =
0.76–0.84 ms → 18.22–18.30; B and C delivered ~40 % of that model per layer, so treat 18.2 as the ceiling.
Nothing in the decode step reads the moved bytes, so no new per-step cost is modeled [A]; the probe's C section and
arm b measure it.

## Risks and what was not done

- Prefill (and the `ref` / `dg-masked` / `dg-mega` impls) read hot scales over C2C now: bit-identical, slower. M3-S2
  estimated ~+40 ms on a 4 s prefill [E] from the C2C probe's +11.5 µs per cold-expert call (weights + scales; the
  scales alone are 1/16 of those bytes, so likely less). Not on the headline metric; measure it on the box.
- Host memory: +12.3 GiB pinned at pool-spare 20 (+0.35 GiB residency e8m0 at 8 slots).
- `runtime/static_kernels.py` and `bench/box/` are not this round's files: the stage kernels needed no change; the
  box commands above are for the orchestrator's script.
- Two CPU tests fail identically on an untouched `git archive HEAD` copy (`test_staging_overlap_cpu.py::
  test_m3s2_sf_pool_plan_rules` asserts the old `all` default of `$DSV41_SF_POOL_LAYERS`, which M3-S2's adoption
  flipped to `auto`; `test_moe_rows_t10_cpu.py::test_pool_sf_headroom_falls_back_to_step` uses a stub without
  `G_max`): stale tests in files outside this round's ownership.
