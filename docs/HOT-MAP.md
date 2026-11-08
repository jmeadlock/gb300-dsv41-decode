# HOT-MAP: hot-expert maps with N > 295 hot experts per layer

2026-11-08 ~06:20 CST. Tool: `bench/hot_map_from_counts.py` (numpy only, runs in the box container). Tests: `tests/test_hot_map_cpu.py`
(19 pass on the Mac). Maps: `docs/maps/hot-v26ar-seed-{301,304,307}.json`. Purpose: spend the HBM that M3-S3 (`DSV41_POOL_SF=beside`)
freed on more hot experts, i.e. fewer cold experts staged over C2C per step. Labels: [M] measured, [D] derived from a receipt, [A] assumption.

## 1. How the 295 map was built, and what the tool reproduces

`docs/maps/hot-v26ar-seed-295.json` is the hot half of m3's `rowmap-mixed-ar-v1.json` (sha 9b55d83a), written by
`research/dsv41-prefetch-20261023/ar/fit_mixed_rowmap.py`:

- input: `counts-post-20261024-020438.npz`, key `cnt`, int64 `[40, 384]`, cumulative routed decode selections since the v26ar boot (AR1 + AR2);
  128,678,880 selections in total. The file also holds `ewma` (float64 [40,384]), `hot` (int32 [40,295]), `cold` (int32 [40,89]) and a scalar `host_steps`.
- per layer: rank all 384 experts by `(-count, not-in-static-v1-hot, expert id)`, keep the top 295. "Ties -> static membership" means an
  equal count goes to the expert that the older `rowmap-static-v1` already had hot. (That script also keeps row order stable; irrelevant here because `HotMap` sorts ids.)

The tool applies the same ranking for any N:

    hot_N[layer] = base[layer]  U  the best (N - 295) experts outside base by (-count, id)

- Superset of the 295 map by construction, so nothing residency or replay evidence was built on is dropped. N < 295 is refused.
- When the base is itself the top-295 of the ranking (true for the seed on its own counts), `hot_N` is exactly the top-N of the ranking, and the maps nest: 301 within 304 within 307.
- Ties among non-base experts break by lowest id, by default. `--tie-map <rowmap-static-v1.json>` swaps in the original static membership as the tie-set.
  On the real counts the 295 map reproduces with the static map, with the base's own membership, and with no tie-set at all, so the choice does not matter for 295.
  For N > 295 it can only matter on boundary ties, listed below.

Self-test, run first on every invocation (and alone with `--self-test-only`): the pure top-295 fit of the counts must equal the seed map on all 40 layers.
Mismatch -> exit 2 and nothing is written ("not the counts the seed was fitted from"). `--skip-self-test` exists for exploring other counts; outputs are still supersets of the base.

npz inspection: prints every array (`name shape dtype`), then picks the ONE integer array shaped `[40, 384]` (or `[S, 40, 384]` with `--snapshot`).
Zero or several candidates -> exit 2 listing them and asking for `--key`. A taller array (a pinhot dump has 43 rows) needs `--first-layers 40`.
Floats, `[40,295]`/`[40,89]` lists and scalars never qualify.

### What was run on the Mac [M]

A copy of the counts file is at `~/hermes/research/dsv41-prefetch-20261023/ar/results/run1/counts-post-20261024-020438.npz` (sha256 `d9e66a762040465b2a63fa75cc41a875b8b8423aec0241a287790652e25b8c0f`; same selection total, 128,678,880, as the 295 map's own summary).
Against it: self-test PASS on 40/40 layers; N = 295 regenerates `hot-v26ar-seed-295.json` byte for byte (sha256 prefix `18903eaf8377206a`). The three maps below were built from that copy. The box rebuild must give identical sha256s; if the box npz differs, the self-test says so.

| N | extra/layer | file sha256 (prefix) | in-sample cold share | boundary-tie layers |
|---|---|---|---|---|
| 295 | 0 | 18903eaf8377206a | 0.008813 | 2 |
| 301 | +6 | 4b68d8316697a77a | 0.007579 | 3 |
| 304 | +9 | aa36e3fffe5e71cb | 0.007001 | 0 |
| 307 | +12 | 5b432efc1a69f489 | 0.006446 | 1 |

In-sample = scored on the counts that built it, the AR1+AR2 mixed traffic, so a floor, not a forecast. Cold share falls about 14 % at 301 and 27 % at 307 relative to 295.
Boundary-tie layers = layers where the count at rank N equals the count at rank N+1, so the last pick is decided by id. Non-zero values change which of two equally-hot experts is in; they do not change the in-sample cold share.

## 2. How the seed map is consumed (A2-RESIDENCY-DESIGN, runtime/)

- `docs/A2-RESIDENCY-DESIGN.md` §2 chose this seed and recommended a bigger hot set: "boot a bigger hot set ... Keep H uniform" (MegaMoE's `num_experts` is a template constant; non-uniform budgets gain <= 3 %). §3.4 T-R1 already lists "top-327" as a map the bitwise-equality test must hold for.
  Our maps are uniform. Any map is bit-identical in logits to any other by that design claim; the headline still checks `streams exact` 8/8 per map.
- `HotMap.load` (`runtime/placement.py:111`) takes `{"<layer>": [ids]}`, sorts ids, rejects duplicates and out-of-range, and requires all 40 layers. The tool's output is that format.
- Everything downstream takes H from the loaded map, per layer: `ExpertPool` (`expert_pool.py:125,132,174,210,341`), `dg_inplace` (`G_max = max(hot) + spare`, `:671`), `dg_mega.py:621`, `residency.py:285` (`self.H = [len(pool.hot[l]) ...]`), `moe_rows.py:737`, `loader.py:399-404` (host budget).

### Every site that assumes 295 (grep of runtime/ and bench/; runtime code unchanged)

Runtime code, all reached only when no `--map` is given:
- `runtime/loader.py:267`, `_resolve_map`: `HotMap.first_n(295 if hot_per_layer is None else hot_per_layer)`. Docstring at `loader.py:354` ("default 295").
- `runtime/placement.py:320`, `load_hot_map`: same default. Also `placement.py:4` (usage text) and `:329` (`--hot-per-layer` help "default 295").
- No runtime code asserts `len(map) == 295`. Nothing hard-codes 89 cold or 384 - 295. `dg_inplace.py:1011,1115` use layer 0's H as the representative for two self-checks (fine with a uniform map).
- Comments only: `runtime/dg_inplace.py:29,408` ("13.14 GiB at 295 hot + 24 spare"), `runtime/staging_overlap.py:544`.

Bench code:
- Default `--map` = seed-295 in `bench/headline.py:432`, `draft_bench.py:200`, `gap_ledger.py:1158`, `row_divergence_probe.py:843`, `dg_rowmatch_spike.py:164`, `dg_rows_spike.py:81`, `dg_mega_spike.py:147`, `dg_mega_numerics.py:1367,1400`, `dg_contiguous_spike.py:135`. Pass `--map` explicitly for the new maps.
- Explicit seed-295 in the box scripts `bench/box/m3s3.sh`, `m3s1.sh`, `m2-round1.sh` (their `FL=` line). These run first_light; they do not pick up a new map unless edited or copied.
- Hard-coded 295 constants in analytic benches: `bench/staging_bytes_bench.py:70` (`HOT_SEED = 295`, used at `:404,:432`), `bench/pool_layout_bench.py:52` (via `SB.HOT_SEED`), `bench/dg_contiguous_spike.py:24,181` (`[295]*40`), `bench/dg_mega_spike.py:593` (`295 + 16`). They model bytes for the seed; they do not read the map.
- Text label: `bench/gap_ledger.py:376` (`RESIDENCY_OURS` names "295 hot experts/layer ... 89 cold"). It would be wrong for a 301+ run's ledger.
- The s3-B headline was run without `--map` (the default, seed-295); the receipt's `boot.map` holds the path actually loaded, so check it on every new run.

Tests (they pin 295 and stay valid, since the 295 file is unchanged): `tests/test_residency_cpu.py:348,366`, `test_loader_mini.py:235-253` (first_n(295) bytes), `test_dg_inplace_cpu.py:134-140`, `test_dg_mega_cpu.py:252,515-517`, `test_staging_overlap_cpu.py:736`, `test_gap_ledger_cpu.py:202`, `tests/gpu/*` (default map via `$DSV41_MAP`).

Residency interplay: `ResidencyManager` boots from the map and treats it as `boot_hot` (`residency_policy.py:378`). Spare slots (`--pool-spare 20` = 4 residency spares + 16 staging) are added on top of H, independent of N. `spare_budget` with `reserve_gib=8` already yields 0 new slots at 6.9 GiB free (receipt `residency.budget`), so a larger N does not change that.

## 3. HBM arithmetic

One more hot expert per layer, in all 40 layers (all [D] from `docs/headline-s3-B-20261108.json`, the beside-layout headline, residency on):

- weights 3 x 5,898,240 B = 17,694,720; packed SF in the pool, slot-parallel on the same group index = 1,105,920. Per expert 18,800,640 B (matches `docs/A2-RESIDENCY-DESIGN` and `test_residency_cpu`).
- x 40 layers = 752,025,600 B = **0.7004 GiB** per +1/layer. (18.8 MB x 40 is 0.752 GB, i.e. 0.700 GiB, not 0.75 GiB.)
- Receipt: `hbm_after_build_GiB` free **6.951** (total 249.814, allocated 241.686); `hbm_end_GiB` free **6.854** after the 8-prompt run. The 5.18 GiB freed by M3-S3 is already inside these numbers, so it must not be added again.
  Pool: `groups_per_layer [315]` = 295 + 20 spare; `packed_sf_hbm_GiB 12.978`; `hbm_pool_GiB 207.436`.
- The 2 GiB headroom matches `dg_inplace.SF_POOL_HEADROOM_GIB = 2.0`.

| N | +/layer | cost GiB | free left, after-build (6.951) | free left, end-of-run (6.854) | with 2 GiB headroom |
|---|---|---|---|---|---|
| 301 | 6 | 4.202 | 2.749 | 2.652 | fits |
| 302 | 7 | 4.903 | 2.048 | 1.951 | marginal: passes on the after-build figure by 0.05 GiB, fails on the end-of-run figure |
| 304 | 9 | 6.303 | 0.648 | 0.551 | does not fit |
| 307 | 12 | 8.405 | -1.454 | -1.551 | does not fit (over the whole free pool) |

Max extra experts per layer with 2 GiB headroom: (6.951 - 2) / 0.7004 = 7.07 on the after-build free, (6.854 - 2) / 0.7004 = 6.93 on end-of-run free.
**Largest N that fits with margin: 301.** 302 is the arithmetic ceiling and is not safe: first-light capture, a longer fixture or another graph size can eat 0.1 GiB. 304 and 307 do not fit on today's build and are here as candidates if more HBM is freed (they need 6.30 and 8.41 GiB plus 2).

Not counted (small, [A]): host side, each hot expert's e8m0 scales live in Grace (`hot_scales`, 1,105,920 B per expert, +0.04 GiB per +1/layer) while the cold pool loses 18.8 MB per expert (-0.70 GiB per +1/layer), so host memory goes down. Headline-time buffers whose size follows H were not separated in the receipt (the DG buffers are sized by rows x topk, not H).
KV, graph pool and workspaces are inside `allocated` already (the figure is taken after capture).

Expected gain, from the in-sample cold share only: 0.0088 -> 0.0076 at 301 on the fitting traffic. The headline's cold shares on its 8 prompts were 0.054 (two-graph cell) / 0.016 (one-graph cell) with residency on, and per prompt 0.02-0.15 in the two-graph cell; those are shift-phase numbers where the in-sample floor does not apply, so measure it. The 6-row cost model in `docs/headline-s3-B` fits ~0.245 ms per cold-share percentage point (r = 0.97, 8 points, residency-confounded), which bounds the prize at a few tenths of a ms per step for ~1 point of cold share. Do not promise more than what the box run shows.

## 4. Box commands

Run in the `dk` container pattern of `bench/box/pg61-matched.sh` (lease-serialized: take `~/dsv41/box.lock`, no other container running). `$W` = `~/dsv41/decode`, mounted at `/work`; the counts directory is mounted read-only at `/ar`. First `git pull` so `bench/hot_map_from_counts.py`, `tests/test_hot_map_cpu.py` and `docs/maps/hot-v26ar-seed-{301,304,307}.json` are in `$W`.

    # 0. self-test only (does the box npz reproduce the 295 map?) - CPU, a few seconds, no GPU
    docker run --rm --network none -v $W:/work -w /work -v ~/dsv41/ar/run:/ar:ro --entrypoint python3 vllm/vllm-openai:v0.31.0 \
        bench/hot_map_from_counts.py --counts /ar/counts-post-20261024-020438.npz \
        --base-map docs/maps/hot-v26ar-seed-295.json --self-test-only

    # 1. rebuild 301 304 307 on the box; must print PASS and the sha256 prefixes 4b68d831 aa36e3ff 5b432efc.
    #    --force is NOT needed: the files are in git and identical; a mismatch would refuse to overwrite and exit 2.
    docker run --rm --network none -v $W:/work -w /work -v ~/dsv41/ar/run:/ar:ro --entrypoint python3 vllm/vllm-openai:v0.31.0 \
        bench/hot_map_from_counts.py --counts /ar/counts-post-20261024-020438.npz \
        --base-map docs/maps/hot-v26ar-seed-295.json --n 301 304 307 --out-dir docs/maps --report scratch/hot-map-report.json
    git -C $W status --short docs/maps        # expect clean: the box output equals the checked-in files

    # 2. headline per map (same arguments as headline-s3-B plus --map; one run per N, serialized under the lease)
    J=$HOME/dsv41/w0/jit-cache; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
    dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
    HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 20 --residency-drain 64 --max-new 256"
    for n in 301; do   # add 302 only to probe the ceiling; 304 and 307 will not fit today
      dk hm$n -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --map docs/maps/hot-v26ar-seed-$n.json --json scratch/headline-hm-$n.json > $W/scratch/headline-hm-$n.log 2>&1; echo rc=$?
    done

Compare `scratch/headline-hm-301.json` with `docs/headline-s3-B-20261108.json` (295, same arguments otherwise): `boot.map` is the 301 file; `loop.verify.pool_sf.groups_per_layer` is `[321]`; `hbm_after_build_GiB.free` is about 2.75 (6.951 - 4.202); `streams.exact` and `replay_ok` are 8/8; `headline.cold_share` and `ms_per_6row_step_dspark_k5` are the result. If free HBM at the end is under 2 GiB, or the pool falls back to a `step` SF store, the map does not fit: drop to a smaller N.
The residency manager is on in this run, so the cold-share cells are confounded by its state (see `mode_gap.cause` in the s3-B receipt). Run 295 and 301 back to back, or add the residency-off arm, before attributing a ms change to N.
