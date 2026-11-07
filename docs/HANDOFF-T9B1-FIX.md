# HANDOFF — T9b-1 fix round: make `mxfp8_gemv` bandwidth-bound on every shape

Box measurements of your kernel (`bench/gemv_bench.py`, `scratch/gemv-bench.json`, full log `scratch/gemv-bench.log` on the box; copied below) — numerics are perfect (max 1 ulp, ≤ 0.002 % of outputs, every shape) but speed misses both gates:

| shape (N×K) | M=1 gemv vs ref | M=6 gemv vs ref | gemv GB/s M=1 |
|---|---|---|---|
| 32768×1280 (wq_b) | 14.3 vs 19.2 µs, 1.3× | 46.9 vs 19.2, **0.4×** | 2,951 |
| 25600×6144 (engram.wkv) | 55.9 vs 81.7, 1.5× | 217 vs 82, 0.4× | 2,820 |
| 5120×8192 (wo_b) | 69.0 vs 94.6, 1.4× | 221 vs 95, 0.4× | 609 |
| 2304×5120 (shared w1/w3) | 43.5 vs 61.8, 1.4× | 135 vs 62, 0.5× | 272 |
| 1280×5120 (wq_a) | 43.4 vs 60.6 | 133 vs 61 | 151 |
| 512×5120 (wkv) | **43.1** vs 61.4 | 131 vs 62 | **61** |
| per step M=1 | ref 15.77 ms → gemv 11.28 ms (−4.5 ms; gate needs ≈ −13) | | |

Autotune chose `{block_n 64, ku 4, num_warps 8, num_stages 1}` for every shape.

## Two defects, both in `runtime/gemv_mxfp8.py` `t9b_mxfp8_gemv`
1. **Grid = ⌈N/BLOCK_N⌉ programs, no split-K.** For N=512 that is 8 programs on a 148-SM GPU, each walking all K=5120 serially → a ~43 µs floor regardless of bytes (512×5120 is 2.6 MB; at 6 TB/s that is 0.4 µs). Only the N ≥ 25,600 shapes fill the machine, and they are the only ones near bandwidth.
2. **M rows handled serially** (`for m in tl.static_range(M)` with `acc += tl.where(rm == m, …)`) and K order inside the KU window preserved with `for u in static_range(KU): tl.sum(tl.where(ru == u, q, 0), 1)` — each of those is a full-tile reduction per u per m. Work scales as M×KU full-tile ops; at M=6 the kernel is compute-bound on masking, hence 0.4×.

## Required changes
- **Split-K**: grid `(⌈N/BLOCK_N⌉, SPLIT_K)`; program (pn, ks) accumulates its contiguous K-block range **in K order** into fp32 `acc[BLOCK_M, BLOCK_N]` and writes fp32 partials to a static workspace `ws[SPLIT_K, M, N]`; a second small Triton kernel sums `ws` over the split axis **in fixed order 0..SPLIT_K−1** and stores bf16. Choose `SPLIT_K` so programs ≥ ~2 × SM count (148 SMs here; read `torch.cuda.get_device_properties(0).multi_processor_count`), capped so each program still has ≥ 4 K-blocks; it becomes a tuned parameter with the others and is recorded in the config. Workspace is allocated once per (M,N) at tune/first-call time (static for graph capture). Numerics: the split changes the cross-block summation order — document it as the second (and now only other) 1-ulp-class deviation; the bench's drift gate already measures it.
- **All M rows per program, no masking tricks**: load `xt [BLOCK_M, KU, 32]` once per iteration (rows ≥ M masked to 0 with scale code 127), compute the block partials for all rows at once — either `tl.sum(wf[None] * xt[:, None], axis=3)` → `[BLOCK_M, BLOCK_N, KU]` then `acc += tl.sum(q * scale[BLOCK_M, BLOCK_N, KU], axis=2)` (KU small: 2–4), or better, `tl.dot` on fp8 operands with `BLOCK_M = 16` zero-padded rows (tensor cores; fp8×fp8 products are exact in fp32, so the class of deviation is unchanged) one dot per 32-wide K block followed by `acc += part * exp2(sa[:,None] + sb[None,:] − 254)`. Try `tl.dot` first; if Triton 3.7.1 rejects fp8e4nv dot on sm_103 fall back to the broadcast form. **Drop the per-u `tl.where` reconstruction entirely** — order within a KU window is just the order you accumulate the KU partials, loop them.
- Autotune candidates must include `block_n ∈ {32, 64, 128}`, `ku ∈ {2, 4, 8}`, `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3, 4}`, `split_k` as above; confirm in the bench output that different shapes pick different configs (the current all-identical result says the search is not discriminating — check the tune harness times the kernel, not a cached graph of one config).
- Keep: the e8m0 pair-scale arithmetic, the fp32 accumulate per K block, the interface (`mxfp8_gemv(xq, xs, w, s, out, config)`), the flag plumbing, graph-capturability (no host sync, static workspace), the tune-cache format (add `split_k`; old cache entries without it are ignored).
- Update `tests/test_gemv_cpu.py` (CPU algorithm now mirrors split-K + reduce order), `tests/gpu/test_gemv_gpu.py` (add a non-multiple-of-split K and M=1,3,6,8 cases), and the bench gates: **every shape ≥ 2× ref at M=1 AND ≥ 1.5× at M=6; shapes with ≥ 16 MB of weight ≥ 4 TB/s; small shapes ≤ 8 µs**; print the per-step saving at M=1 (target ≥ 12 ms of the 15.8).

Rules unchanged (only `runtime/`, `bench/`, `tests/`; no ssh/docker/network/commit; never import vllm; the Mac has no GPU — say so). Summary ≤ 12 lines ending `T9B1-FIX: <ready-for-box|blocked> — <reason>`.
