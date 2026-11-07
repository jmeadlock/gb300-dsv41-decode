"""GPU tests of M1-T10-MOE (docs/HANDOFF-M1-T10-MOE.md): the 6-row MoE's dedup, staging and routed-GEMM call shape.
Skipped without CUDA/tilelang (Mac); on the box a missing/broken deep_gemm FAILS (it is under test).

  * m1_dedup_warp compiled for the device == the torch mirror (moe_rows.rows_layout) on every output, every layout x
    SF store, heavy cross-row repeats + overflow; == m1_dedup_routes on the M1-T10-PORT outputs;
  * m1_stage_rows on the device == stage_rows_emulated (cold rows from pinned host memory over UVA, real row sizes:
    EVEN path; odd sizes: masked path), both SF stores;
  * mini4 6-row dg-inplace StaticStep in every call shape (contiguous 128 / 64 / 32 / 16, psum, masked) x SF store
    (pool, step) == today's 6-row dg-inplace (contiguous 128, "step" SF, matrix dedup) and == dg-masked: torch.equal
    on logits, main_hidden, caches; graph == static-eager; overflow 0; every non-default call shape passed its build
    check (or the fallback is printed and the test fails: the bar is bit-equal);
  * real layer-0 bytes (checkpoint mounted at /ref): bench/dg_rows_spike.py's variants and the staged bytes; its
    JSON is the item-1 evidence (written to docs/dg-rows-spike-<date>.json).

Run on the box (orchestrator) exactly as tests/gpu/test_static_rows_gpu.py (same image, mounts, DG_JIT_CACHE_DIR):
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_moe_rows_gpu.py
"""
from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_dg_moe_gpu import _need_dg  # noqa: E402
from test_static_rows_gpu import COUNTS, ROWS, _assert_rows, _model, _need_step, _static, _toks  # noqa: E402

CALL_SHAPES = [("contiguous", "pool"), ("contiguous", "step"), ("contiguous64", "pool"), ("contiguous32", "pool"),
               ("contiguous16", "pool"), ("psum", "pool"), ("psum", "step"), ("masked", "pool")]


# ------------------------------------------------------------------------------------------------ dedup
def test_dedup_warp_on_device_equals_mirror():
    _need_dg()
    from runtime import moe_rows as MR

    g = torch.Generator().manual_seed(11)
    dev = "cuda"
    n = 0
    for t in range(48):
        R_, k = [(6, 6), (6, 4), (8, 8), (2, 6)][t % 4]
        E = int(torch.randint(k + 1, 4 * k, (1,), generator=g))
        idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R_)])
        w = torch.rand(R_, k, generator=g)
        hot = torch.rand(E, generator=g) < 0.6
        gtab = torch.full((E,), -1, dtype=torch.int32)
        gtab[hot] = torch.arange(int(hot.sum()), dtype=torch.int32)
        H, S = int(hot.sum()), int(torch.randint(1, 6, (1,), generator=g))
        lay = ("contiguous", "psum", "masked")[t % 3]
        BR = 128 if lay == "masked" else (8, 16, 32, 128)[t % 4]
        G = H + S
        inpl = {"gtab": gtab.to(dev), "H": H, "S": S, "dummy": torch.tensor(0, device=dev), "has_dummy": H > 0,
                "v2": True, "layout": lay, "G": G, "gscr": G, "step_sf": bool(t % 2),
                "mi": torch.full((R_ * k * BR,), -1, dtype=torch.int32, device=dev) if lay == "contiguous" else None,
                "gl": torch.full((G,), -5, dtype=torch.int32, device=dev),
                "ovf": torch.zeros(1, dtype=torch.int32, device=dev)}
        idx_d, w_d = idx.to(dev), w.to(dev)
        m = MR.dedup_routes(idx_d, w_d, BR)
        ext = MR.rows_layout(m, inpl, BR, R_, k)
        buf = MR.DedupBuffers(R_, k, dev)
        got = MR.dedup_fused(buf, idx_d, w_d, BR, inpl=inpl, impl="warp")
        torch.cuda.synchronize()
        for key in ("uniq", "counts", "valid", "slot", "rank", "prow", "pw"):
            assert torch.equal(got[key], m[key]), (t, key)
        for key in ("arow", "posmat", "wl", "sidx", "ids_stage"):
            assert torch.equal(got[key], ext[key]), (t, key)
        if lay == "contiguous":
            assert torch.equal(inpl["mi"].view(R_ * k, BR)[:, :R_], ext["tags"])
        else:
            assert torch.equal(inpl["gl"], ext["gl"])
        assert int(inpl["ovf"]) == int(ext["n_over"])
        # v1 outputs: the warp kernel == the matrix kernel
        v1 = {"gtab": gtab.to(dev), "H": H, "S": S, "dummy": torch.tensor(0, device=dev), "has_dummy": H > 0}
        outs = []
        for impl in ("warp", "matrix"):
            b1 = MR.DedupBuffers(R_, k, dev)
            o = MR.dedup_fused(b1, idx_d, w_d, BR, inpl=v1, impl=impl)
            outs.append({k_: v.clone() for k_, v in o.items() if torch.is_tensor(v)})
        assert all(torch.equal(outs[0][k_], outs[1][k_]) for k_ in outs[1]), t
        n += 1
    print(f"  dedup: {n} device cases equal (warp == mirror, warp == matrix)")


# ------------------------------------------------------------------------------------------------ staging
def test_stage_rows_on_device_equals_contract():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime import moe_rows as MR
    from runtime import static_kernels as SK

    dev = "cuda"
    for N, KH in ((2304, 2560), (40, 64)):          # real w1 row (EVEN path) and an odd size (masked path)
        KS = KH * 2 // 32
        g = torch.Generator().manual_seed(N)
        E, hot = 8, [6, 1, 3, 0, 5]
        cold = [2, 4, 7]
        H, S = len(hot), 3
        src_w = {(p, e): torch.randint(-128, 128, (N, KH), generator=g).to(torch.int8) for p in DI.PROJS
                 for e in range(E)}
        src_s = {(p, e): torch.randint(100, 140, (N, KS), generator=g).to(torch.uint8) for p in DI.PROJS
                 for e in range(E)}
        host_w = {(p, e): src_w[(p, e)].pin_memory() for p in DI.PROJS for e in cold}
        host_s = {(p, e): src_s[(p, e)].pin_memory() for p in DI.PROJS for e in cold}
        for step_sf in (False, True):
            outs = []
            for use_kernel in (False, True):
                full = {p: torch.zeros(H + S, N, KH, dtype=torch.int8, device=dev) for p in DI.PROJS}
                hs = {p: torch.stack([src_s[(p, e)] for e in hot]).to(dev) for p in DI.PROJS}
                for s_, e in enumerate(hot):
                    for p in DI.PROJS:
                        full[p][s_].copy_(src_w[(p, e)])
                C = DI.sf_words(KS)
                sf = {p: DI.packed_sf_zeros(H + S, N, C, dev) for p in DI.PROJS}
                dsts = [full[p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
                dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device=dev)
                ptab = torch.zeros(6, E, dtype=torch.int64)
                reg = {}
                for e in range(E):
                    for i, p in enumerate(DI.PROJS):
                        wt = full[p][hot.index(e)] if e in hot else host_w[(p, e)]
                        st_ = hs[p][hot.index(e)] if e in hot else host_s[(p, e)]
                        ptab[i, e], ptab[3 + i, e] = wt.data_ptr(), st_.data_ptr()
                        reg[wt.data_ptr()] = wt.reshape(-1).view(torch.uint8)
                        reg[st_.data_ptr()] = st_.reshape(-1).view(torch.uint8)
                gtab = torch.full((E,), -1, dtype=torch.int32)
                for s_, e in enumerate(hot):
                    gtab[e] = s_
                idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7]], device=dev)
                m = MR.dedup_routes(idx, torch.rand(3, 3, device=dev), 4)
                inpl = {"gtab": gtab.to(dev), "H": H, "S": S, "dummy": torch.tensor(6, device=dev), "has_dummy": True,
                        "v2": True, "layout": "contiguous", "G": H + S, "gscr": 0, "step_sf": step_sf}
                ext = MR.rows_layout(m, inpl, 4, 3, 3)
                nw = 9 if step_sf else S
                args = (ptab.to(dev), m["uniq"], gtab.to(dev), dtab, dsts, ext["sidx"], ext["wl"], H, N * KH,
                        ((N, C), (N, C)), nw)
                if use_kernel:
                    SK.stage_rows(*args)
                    torch.cuda.synchronize()
                else:
                    DI.stage_rows_emulated(reg.__getitem__, *args)
                outs.append([t.clone() for t in dsts])
            for i, (a_, b_) in enumerate(zip(*outs)):
                assert torch.equal(a_, b_), (N, step_sf, i)
    print("  stage_rows: device kernel == contract (EVEN + masked sizes, both SF stores)")


# ------------------------------------------------------------------------------------------------ the 6-row step
def test_static6_call_shapes_equal_today_and_dg_masked():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime import moe_rows as MR

    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        ids, toks = _toks(18, ROWS * 4, 71)
        os.environ[MR.DEDUP_IMPL_ENV] = "matrix"     # today's run: matrix dedup, contiguous 128, "step" SF
        try:
            base_st = _need_step(model, pool, rows=ROWS, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8,
                                 rows_moe_gemm="contiguous", rows_sf_store="step")
            base = _static(model, base_st, ids, toks, 4)
            base_st.detach()
        finally:
            os.environ.pop(MR.DEDUP_IMPL_ENV, None)
        sm = _need_step(model, pool, rows=ROWS, hidden_layers=hid, moe_impl="dg-masked")
        masked = _static(model, sm, ids, toks, 4)
        sm.detach()
        _assert_rows(base[0], masked[0], "today's 6-row dg-inplace vs dg-masked")
        print(f"  {'call shape / SF store':28s} {'check':>6s} {'bit-equal':>9s}  graph==eager  overflow")
        for gemm, store in CALL_SHAPES:
            st = _need_step(model, pool, rows=ROWS, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8,
                            rows_moe_gemm=gemm, rows_sf_store=store)
            chk = st.dg.layout_check
            assert chk is None or (chk.get("equal") and not chk.get("fallback")), (gemm, chk)
            a = _static(model, st, ids, toks, 4)
            st.capture()
            ag = _static(model, st, ids, toks, 4, graph=True)
            _assert_rows(ag[0], a[0], f"{gemm}/{store}: graph vs static-eager")
            _assert_rows(a[0], base[0], f"{gemm}/{store} vs today's 6-row dg-inplace")
            assert all(torch.equal(x, y) for x, y in zip(a[1], base[1])), gemm
            for k_ in base[2]:
                assert torch.equal(a[2][k_], base[2][k_]), (gemm, store, k_)
            ov = st.overflow_counts()
            assert ov["steps"] == 0 and ov["experts"] == 0, ov
            d = st.describe()
            assert d["dedup_impl"]["impl"] == "warp", d["dedup_impl"]
            print(f"  {gemm + ' / ' + store:28s} {'-' if chk is None else 'equal':>6s} {'yes':>9s}  yes           0")
            st.detach()
            COUNTS["gates"] += 1
        assert DI.pool_sf_key(pool) is not None


# ------------------------------------------------------------------------------------------------ real bytes
def test_real_layer0_bytes_spike():
    _need_dg()
    if not os.path.isdir("/ref"):
        raise unittest.SkipTest("checkpoint not mounted at /ref")
    import json

    import dg_rows_spike

    out = os.path.join(REPO, "docs", f"dg-rows-spike-{time.strftime('%Y%m%d')}.json")
    rc = dg_rows_spike.main(["--out", out, "--iters", "100"])
    rep = json.load(open(out))
    assert rep["checks"]["staging_check"].get("staged_equal_to_checkpoint"), rep["checks"]["staging_check"]
    assert rep["checks"]["dedup"]["warp"]["equal_to_mirror"], rep["checks"]["dedup"]
    v = rep["variants"]
    assert v["contiguous128"].get("bit_equal"), v["contiguous128"]
    print("  real layer-0 bytes:", json.dumps(rep["verdict"]))
    assert rc in (0, 2)


TESTS = [test_dedup_warp_on_device_equals_mirror, test_stage_rows_on_device_equals_contract,
         test_static6_call_shapes_equal_today_and_dg_masked, test_real_layer0_bytes_spike]

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
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
