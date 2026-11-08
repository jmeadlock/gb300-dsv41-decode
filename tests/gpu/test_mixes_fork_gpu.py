"""GPU tests of M2-L1 (docs/HANDOFF-M2-L1-MIXES-FORK.md): the mHC mixes on a forked graph branch
(runtime/static_step.py MixesFork, $DSV41_MIXES off|fork|fused). Skipped without CUDA/tilelang (Mac).

Gates (fork, bit-identical by construction -- the same kernels on a second stream):
  * static-eager and graph-replay fp32 logits of the fork step == the off step, torch.equal, 1 row and 6 rows, the
    fast path (dg-inplace, gemv, fused glue, flashmla) under $DSV41_FP32_IMPL=fixed and cublas, and the reference-glue
    path; graph == static-eager; replay twice; no host sync in the eager fork step;
  * the captured graph HAS the branch (CUDAGraph debug mode -> cudaGraphDebugDotPrint, bench/mixes_fork_bench.py
    analyze_dag): the same node count as off, +2 forks and +2 joins per layer, 2 kernel nodes per sublayer (fixed:
    t9b5_mix_fp32 + the sinkhorn) off the longest path;
  * `fused` under fp32 fixed == off (same kernel); under cublas it is A3 (reported: rows equal, max |dlogit|, argmax).
Timing, ms saved and the SM-contention check are bench/mixes_fork_bench.py gpu (--profile / --nsys).

Run on the box (orchestrator), same line as tests/gpu/test_static_rows_gpu.py, repo at $W:

  W=$HOME/hermes/gb300-dsv41-decode-wtl1
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_mixes_fork_gpu.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

ROWS = 6
COUNTS = {"logit_rows": 0, "graphs": 0}


def _bench():
    if "mixes_fork_bench" in sys.modules:
        return sys.modules["mixes_fork_bench"]
    spec = importlib.util.spec_from_file_location("mixes_fork_bench", os.path.join(REPO, "bench", "mixes_fork_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mixes_fork_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def _need_gpu():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    from runtime.refmodel import tilelang_available

    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _assert_rows(a, b, what):
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)]
    assert len(a) == len(b) and not bad, (f"{what}: rows {bad[:12]} differ ({len(bad)}/{len(a)}); max |d| "
                                          f"{(a[bad[0]] - b[bad[0]]).abs().max().item() if bad else 0}")
    COUNTS["logit_rows"] += len(a)


def _one(B, model, pool, rows, impl, kw, ids, toks, n):
    """Build under $DSV41_MIXES=impl, eager run, capture (debug graphs), replay x2 -> (eager, g1, g2, dag records)."""
    from runtime.static_step import StaticStep

    with B.env(DSV41_MIXES=impl):
        st = StaticStep(model, pool, max_pos=B.MAX_POS, **(dict(kw, rows=rows) if rows > 1 else kw))
    try:
        assert st.mix_fork.impl == impl
        e = B._run(model, st, ids, toks, n, graph=False)
        with tempfile.TemporaryDirectory() as td, B.debug_graphs() as made:
            st.capture()
            dags = B.dump_and_analyze(made, td)
        g1 = B._run(model, st, ids, toks, n, graph=True)
        g2 = B._run(model, st, ids, toks, n, graph=True)
        _assert_rows(g1, e, f"{impl} rows {rows}: graph vs static-eager")
        _assert_rows(g2, g1, f"{impl} rows {rows}: replay #2 vs #1")
        if impl != "off":
            assert st.mix_fork.pending == 0 and st.mix_fork.forks == st.mix_fork.joins, st.mix_fork.describe()
            _no_sync(B, model, st, ids, toks)
        return e, g1, dags
    finally:
        st.detach()


def _no_sync(B, model, st, ids, toks):
    """The eager fork step launches no host sync (events + record_stream only)."""
    B._prefill(model, ids)
    with torch.inference_mode():
        st.step(toks[:st.rows] if st.rows > 1 else toks[0], len(ids), graph=False)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            st.step(None, len(ids) + st.rows, graph=False)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def _check_branch(dag_off, dag_fork, L, per_sub, what):
    for a, b in zip(dag_off, dag_fork):
        assert "error" not in a and "error" not in b, (what, a, b)
        assert b["nodes"] == a["nodes"], (what, "node count changed", a, b)
        assert b["forks"] - a["forks"] == 2 * L and b["joins"] - a["joins"] == 2 * L, (what, a, b)
        assert b["off_path"] - a["off_path"] == per_sub * 2 * L, (what, a, b)
        assert b["longest_path"] == a["longest_path"] - per_sub * 2 * L, (what, a, b)
        COUNTS["graphs"] += 1
        print(f"    {what}: nodes {a['nodes']}; longest path {a['longest_path']} -> {b['longest_path']}; forks "
              f"{a['forks']} -> {b['forks']}; off-path {b['off_path_names']}")


def test_fork_fast_path_fp32_fixed():
    """The headline config: fork == off, 1 and 6 rows; the branch is 2 kernels per sublayer."""
    _need_gpu()
    B = _bench()
    with B.env(DSV41_FP32_IMPL="fixed"), B.gpu_model("mini4") as (model, pool):
        L = len(model.layers)
        kw = dict(B.FAST)
        for rows in (ROWS, 1):
            ids, toks = B._toks(17, rows * 4, 11 + rows)
            e0, g0, d0 = _one(B, model, pool, rows, "off", kw, ids, toks, 4)
            e1, g1, d1 = _one(B, model, pool, rows, "fork", kw, ids, toks, 4)
            _assert_rows(e1, e0, f"rows {rows}: fork vs off (eager)")
            _assert_rows(g1, g0, f"rows {rows}: fork vs off (graph)")
            _check_branch(d0, d1, L, 2, f"fixed rows {rows}")
            e2, g2, d2 = _one(B, model, pool, rows, "fused", kw, ids, toks, 4)
            _assert_rows(e2, e0, f"rows {rows}: fused vs off under fp32 fixed (eager; the same kernel)")
            _assert_rows(g2, g0, f"rows {rows}: fused vs off under fp32 fixed (graph)")
            _check_branch(d0, d2, L, 2, f"fixed fused rows {rows}")
            print(f"  fp32 fixed rows {rows}: fork == fused == off ({len(e0)} logit rows, eager + graph)")


def test_fork_fast_path_fp32_cublas():
    """cuBLAS mixes on the branch: fork == off; the branch exists (its kernel count is cuBLAS's); fused reported."""
    _need_gpu()
    B = _bench()
    with B.env(DSV41_FP32_IMPL="cublas"), B.gpu_model("mini4") as (model, pool):
        L = len(model.layers)
        kw = dict(B.FAST)
        for rows in (ROWS, 1):
            ids, toks = B._toks(17, rows * 4, 13 + rows)
            e0, g0, d0 = _one(B, model, pool, rows, "off", kw, ids, toks, 4)
            e1, g1, d1 = _one(B, model, pool, rows, "fork", kw, ids, toks, 4)
            _assert_rows(e1, e0, f"cublas rows {rows}: fork vs off (eager)")
            _assert_rows(g1, g0, f"cublas rows {rows}: fork vs off (graph)")
            for a, b in zip(d0, d1):
                assert "error" not in a and "error" not in b, (a, b)
                assert b["nodes"] == a["nodes"] and b["forks"] - a["forks"] == 2 * L, (a, b)
                assert b["off_path"] - a["off_path"] >= 2 * 2 * L, (a, b)
                COUNTS["graphs"] += 1
            print(f"  cublas rows {rows}: fork == off; branch kernel nodes per sublayer "
                  f"{(d1[0]['off_path'] - d0[0]['off_path']) / (2 * L):.1f} {d1[0]['off_path_names']}")
            e2, g2, _ = _one(B, model, pool, rows, "fused", kw, ids, toks, 4)
            eq = sum(torch.equal(x, y) for x, y in zip(g2, g0))
            d = max(float((x - y).abs().max()) for x, y in zip(g2, g0))
            agree = sum(int(x.argmax()) == int(y.argmax()) for x, y in zip(g2, g0))
            print(f"  fused under cublas rows {rows} (A3, reported): bit-equal {eq}/{len(g0)}, max |dlogit| {d:.4e}, "
                  f"argmax {agree}/{len(g0)}")


def test_fork_reference_glue():
    """glue ref (the whole hc_mixes forks: statistic + mm + mul + sinkhorn): fork == off, 1 and 6 rows."""
    _need_gpu()
    B = _bench()
    with B.env(DSV41_FP32_IMPL="fixed"), B.gpu_model("mini4") as (model, pool):
        L = len(model.layers)
        for rows in (ROWS, 1):
            ids, toks = B._toks(17, rows * 3, 17 + rows)
            e0, g0, d0 = _one(B, model, pool, rows, "off", {}, ids, toks, 3)
            e1, g1, d1 = _one(B, model, pool, rows, "fork", {}, ids, toks, 3)
            _assert_rows(e1, e0, f"ref glue rows {rows}: fork vs off (eager)")
            _assert_rows(g1, g0, f"ref glue rows {rows}: fork vs off (graph)")
            for a, b in zip(d0, d1):
                assert b["nodes"] == a["nodes"] and b["forks"] - a["forks"] == 2 * L, (a, b)
                COUNTS["graphs"] += 1
            print(f"  ref glue rows {rows}: fork == off; branch nodes per sublayer "
                  f"{(d1[0]['off_path'] - d0[0]['off_path']) / (2 * L):.1f}")


TESTS = [test_fork_fast_path_fp32_fixed, test_fork_fast_path_fp32_cublas, test_fork_reference_glue]

if __name__ == "__main__":
    import traceback

    results = {}
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            results[t.__name__] = "PASS"
        except unittest.SkipTest as e:
            results[t.__name__] = f"SKIP ({e})"
        except Exception:
            traceback.print_exc()
            results[t.__name__] = "FAIL"
        print(f"{results[t.__name__]:4s} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"bit-equal comparisons executed: {COUNTS['logit_rows']} fp32 logit rows (torch.equal); "
          f"{COUNTS['graphs']} graphs checked for the branch")
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
