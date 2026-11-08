"""GPU tests of M3-S3 (docs/HANDOFF-M3-S3-POOL-SF-LAYOUT.md, docs/M3-S3-POOL-SF-LAYOUT.md): the expert pool's SF
"beside" its weights ($DSV41_POOL_SF=beside). Skipped without CUDA (Mac); on the box a missing/broken deep_gemm FAILS.

  * pool bytes expert by expert, mini4 loaded on cuda in both layouts: weights, the K-major e8m0 (beside: in the
    registered host buffer, the Experts' scale params are UVA aliases of it), the packed slot == DeepGEMM's packing
    of the separate pool's bytes, the loader's verify, HBM pool bytes equal at load;
  * the 6-row dg-inplace step (rows_sf_store pool, spare 8), kernel and kernel-overlap, eager AND captured graph:
    beside == separate on every row's logits, main_hidden, the caches (engram over the written positions: two
    separately loaded models) and the spare weight slots; beside's tables ARE pool.sfp, n_work = the spare budget;
  * 1 row: the --moe-rowmatch build check (this 1-row build per row == the R-row call on layer 0's real bytes) passes
    on beside's own SF, and the 1-row step beside == separate; 6-row vs 1-row full-step agreement reported;
  * residency (tests/gpu/test_residency_gpu.py's 6-row graph gate, unchanged) under beside, requested store step and
    pool: on == off, audit vs the checkpoint golden (digest through the host views), restore_boot;
  * the device staging kernels (m1_stage_rows, s1_stage_proj x 3) on a beside pool's own storage, pool-style list:
    cold weights + cold SF land in their slot, hot groups untouched;
  * bench/pool_layout_bench.py probe quick (build, GEMM equality, stage with/without hot SF): no errors, GEMM equal.

Run on the box exactly as tests/gpu/test_staging_overlap_gpu.py (same image, mounts, DG_JIT_CACHE_DIR):
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_pool_layout_gpu.py
"""
from __future__ import annotations

import json
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
from test_static_rows_gpu import COUNTS, ROWS, _assert_rows, _env, _model, _need_step, _seq1, _static, _toks  # noqa: E402

RESULTS = {}
POOL_SF = "DSV41_POOL_SF"


def _cpu_run(r, n_pos):
    """(logits rows, main_hidden, state) -> CPU copies; the engram cache cut to the written positions."""
    lg, hs, stt = r
    st2 = {}
    for k, v in stt.items():
        v = v[:, :n_pos] if k == "engram" else v
        st2[k] = v.detach().cpu().clone()
    return ([x.detach().cpu() for x in lg], [None if h is None else h.detach().cpu() for h in hs], st2)


def _same(a, b, what):
    _assert_rows(a[0], b[0], what)
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a[1], b[1])), (what, "main_hidden")
    bad = sorted(k for k in b[2] if not torch.equal(a[2][k], b[2][k]))
    assert not bad, (what, "caches", bad)
    COUNTS["gates"] += 1


# ------------------------------------------------------------------------------------------------ pool bytes
def test_pool_bytes_mini4():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime.loader import verify_placement

    ref = {}
    with _env(**{POOL_SF: "separate"}), _model("mini4") as (m_s, p_s):
        assert p_s.sf_layout == "separate"
        for l in range(p_s.n_layers):
            for p in DI.PROJS:
                for s in range(len(p_s.hot[l])):
                    ref[(l, p, s)] = (p_s.hbm_full[l][p][s].cpu(), p_s.hbm[l][p][1][s].view(torch.uint8).cpu())
        hbm_s = p_s.hbm_bytes()
    rec = {"slots": len(ref)}
    with _env(**{POOL_SF: "beside"}), _model("mini4") as (m_b, p_b):
        assert p_b.sf_layout == "beside" and p_b.hot_sf_host is not None and p_b.hot_sf_host.chunks
        for (l, p, s), (w, e8) in ref.items():
            e = p_b.hot[l][s]
            assert torch.equal(p_b.hbm_full[l][p][s].cpu(), w), (l, p, s, "weights")
            assert torch.equal(p_b.hot_scale_host(l, s, p).view(torch.uint8), e8), (l, p, s, "e8m0 host")
            assert torch.equal(p_b.sfp[l][p][s].cpu(), DI.pack_sf(e8)), (l, p, s, "packed slot")
            lin = getattr(m_b.layers[l].ffn.experts[e], p)
            assert lin.scale.is_cuda and lin.scale.data_ptr() == p_b.hot_sf_host.ptr + p_b.hot_scale_offset(l, e, p)
            # the UVA alias reads the same bytes on the device
            assert int(lin.scale.view(torch.uint8).long().sum()) == int(e8.long().sum())
        v = verify_placement(m_b, m_b.runtime_load)
        assert v["ok"] == v["n"] and not v["mismatch"], v["mismatch"][:5]
        assert p_b.hbm_bytes() == hbm_s
        rec.update(verify=f"{v['ok']}/{v['n']}", describe=p_b.describe_sf(),
                   pack_s=m_b.runtime_load.timings.get("pack_hot_sf"))
    RESULTS["pool_bytes"] = rec
    print("  pool bytes:", json.dumps(rec, default=str))


# ------------------------------------------------------------------------------------------------ 6-row step
def _runs6(layout, n, ids, toks):
    from test_staging_overlap_gpu import _checkpoint, _restore

    from runtime import dg_inplace as DI
    from runtime import staging_overlap as SO

    out = {}
    with _env(**{POOL_SF: layout, SO.STAGE_ENV: None}), _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        kw = dict(rows=ROWS, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8, rows_sf_store="pool")
        ck = None
        for mode in ("kernel", "kernel-overlap"):
            with _env(**{SO.STAGE_ENV: mode}):
                pool._dg_packed_sf = None
                st = _need_step(model, pool, **kw)
                b = st.dg
                assert b.sf_store == "pool" and set(b.sf_layer.values()) == {"pool"}, (layout, b.sf_store)
                if layout == "beside":
                    assert all(b.layers[l]["sf"][p] is pool.sfp[l][p] for l in b.layers for p in DI.PROJS)
                    assert all(b.stage_n_work(t) == b.spare_limit for t in b.layers.values())
                if ck is None:
                    ck = _checkpoint(model, pool)
                npos = len(ids) + ROWS * n
                _restore(model, pool, None, ck)
                e = _cpu_run(_static(model, st, ids, toks, n), npos)
                sl = [pool.hbm_full[l][p][len(pool.hot[l]):].cpu() for l in range(pool.n_layers) for p in DI.PROJS]
                st.capture()
                _restore(model, pool, None, ck)
                g = _cpu_run(_static(model, st, ids, toks, n, graph=True), npos)
                assert st.overflow_counts()["steps"] == 0
                out[mode] = {"eager": e, "graph": g, "slots": sl}
                st.detach()
    return out


def test_static6_beside_equals_separate_mini4():
    _need_dg()
    n = 4
    ids, toks = _toks(18, ROWS * n, 2031)
    sep = _runs6("separate", n, ids, toks)
    bes = _runs6("beside", n, ids, toks)
    rec = {}
    for mode in sep:
        for path in ("eager", "graph"):
            _same(bes[mode][path], sep[mode][path], f"{mode}/{path}: beside vs separate")
        assert all(torch.equal(x, y) for x, y in zip(bes[mode]["slots"], sep[mode]["slots"])), (mode, "spare slots")
        _same(bes[mode]["graph"], bes[mode]["eager"], f"{mode}: beside graph vs eager")
        rec[mode] = "eager + graph == separate"
    RESULTS["static6"] = rec
    print("  6-row beside == separate:", json.dumps(rec))


# ------------------------------------------------------------------------------------------------ 1 row
def test_one_row_beside_mini4():
    _need_dg()
    n_pos = ROWS * 3
    ids, toks = _toks(18, n_pos, 2032)
    outs, rec = {}, {}
    for layout in ("separate", "beside"):
        with _env(**{POOL_SF: layout}), _model("mini4") as (model, pool):
            s1 = _need_step(model, pool, moe_impl="dg-inplace")
            rm_ = s1.dg.rowmatch or {}
            chk = rm_.get("check") or {}
            rec[layout] = {"rowmatch_on": rm_.get("on"), "check_equal": chk.get("equal"), "sf_store": s1.dg.sf_store}
            if layout == "beside":
                assert s1.dg.sf_store == "pool" and all(s1.dg.layers[l]["sf"][p] is pool.sfp[l][p]
                                                        for l in s1.dg.layers for p in ("w1", "w3", "w2"))
                # the 1-row build's per-row check vs the R-row call (layer 0's real bytes) on pool.sfp: the same
                # outcome as on separate's copy (True wherever the rowmatch call shape holds today)
                assert rm_.get("on") == rec["separate"]["rowmatch_on"], rm_
                assert chk.get("equal") == rec["separate"]["check_equal"], (rm_, rec["separate"])
            outs[layout] = [x.cpu() for x in _seq1(model, s1, ids, toks, n_pos)]
            s1.detach()
            if layout == "beside":   # 6-row vs 1-row full step: reported (the routed GEMM is gated by the check)
                st = _need_step(model, pool, rows=ROWS, moe_impl="dg-inplace")
                a6 = [x.cpu() for x in _static(model, st, ids, toks, 3)[0]]
                rec["six_vs_one_bit_equal_rows"] = f"{sum(torch.equal(x, y) for x, y in zip(a6, outs[layout]))}/{n_pos}"
                st.detach()
    _assert_rows(outs["beside"], outs["separate"], "1-row: beside vs separate")
    RESULTS["one_row"] = rec
    print("  1-row:", json.dumps(rec))


# ------------------------------------------------------------------------------------------------ residency
def test_residency_rows6_beside():
    _need_dg()
    from test_residency_gpu import _rows_combo

    for store in ("step", "pool"):     # beside builds "pool" for either request; the gate is the residency suite's
        with _env(**{POOL_SF: "beside"}):
            _rows_combo("dg-inplace", sf_store=store)
    RESULTS["residency"] = "rows6 on == off, audit ok, restore ok (requested step and pool)"


# ------------------------------------------------------------------------------------------------ stage kernels
def test_stage_kernels_on_beside_pool():
    _need_dg()
    import pool_layout_bench as PB

    from runtime import dg_inplace as DI
    from runtime import staging_overlap as SO
    from runtime import static_kernels as SK
    from runtime.expert_pool import TIER_HBM

    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7], [3, 5, 6]])
    rec = {}
    for how in ("m1", "s1"):
        pool, src = PB.tiny_pool("beside", device="cuda", dim=256, inter=256)
        pool.host.register()
        pool.hot_sf_host.register()
        t = PB.layer_tables(pool)              # CPU-side tables of host views; the device needs UVA addresses
        rows = []
        for p, kind in DI.ORDER6:
            r = []
            for e in range(pool.n_experts):
                tier, s = pool.slot[(0, e)]
                if tier == TIER_HBM:
                    v = pool.hbm_full[0][p][s] if kind == "weight" else pool._hot_sf_view(0, p, cuda=True)[s]
                else:
                    v = pool.cold_view(0, e, p, kind, cuda=True)
                r.append(v.data_ptr())
            rows.append(r)
        ptab6 = torch.tensor(rows, dtype=torch.int64, device="cpu").cuda()
        m, ext = PB.route_lists(pool, t, idx, False, 4)
        dev = lambda x: x.cuda()  # noqa: E731
        hot_before = [x[: t["H"]].clone() for x in t["dsts"]]
        args = (ptab6, dev(m["uniq"]), dev(t["gtab"]), dev(t["dtab"]), t["dsts"], dev(ext["sidx"]), dev(ext["wl"]),
                t["H"], t["row_w"], t["sf_dims"], 4)
        if how == "m1":
            SK.stage_rows(*args)
        else:
            for pj in range(3):
                SO.stage_proj(*args, pj)
        torch.cuda.synchronize()
        assert all(torch.equal(x[: t["H"]], y) for x, y in zip(t["dsts"], hot_before)), (how, "hot groups")
        H = t["H"]
        for c, e in enumerate(pool.cold[0]):
            for i, p in enumerate(DI.PROJS):
                assert torch.equal(t["dsts"][i][H + c].cpu(), src[(p, "weight", e)]), (how, e, p)
                assert torch.equal(DI.unpack_sf(t["dsts"][3 + i][H + c]).cpu(), src[(p, "scale", e)]), (how, e, p)
        rec[how] = "cold weights + SF in their slot, hot groups untouched"
        pool.host.release()
        pool.hot_sf_host.release()
    RESULTS["stage_kernels"] = rec
    print("  stage kernels on a beside pool:", json.dumps(rec))


# ------------------------------------------------------------------------------------------------ bench
def test_bench_probe_quick():
    _need_dg()
    import pool_layout_bench as PB

    out = os.path.join(REPO, "scratch", f"pool-layout-probe-quick-{time.strftime('%Y%m%d')}.json")
    rc = PB.main(["probe", "--layers", "1", "--hot", "48", "--rounds", "4", "--reps", "10", "--out", out])
    rep = json.load(open(out))["probe"]
    assert not rep["errors"], rep["errors"]
    assert rep["gemm"]["equal"], rep["gemm"]
    RESULTS["probe_quick"] = {k: rep.get(k) for k in ("build", "gemm", "stage")}
    print("  probe (quick):", json.dumps(RESULTS["probe_quick"], default=str)[:2000])
    assert rc == 0


TESTS = [test_pool_bytes_mini4, test_static6_beside_equals_separate_mini4, test_one_row_beside_mini4,
         test_residency_rows6_beside, test_stage_kernels_on_beside_pool, test_bench_probe_quick]

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
    os.makedirs(os.path.join(REPO, "scratch"), exist_ok=True)
    with open(os.path.join(REPO, "scratch", f"pool-layout-gpu-test-{time.strftime('%Y%m%d')}.json"), "w") as f:
        json.dump({"results": results, "data": RESULTS, "counts": COUNTS}, f, indent=1, default=str)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
