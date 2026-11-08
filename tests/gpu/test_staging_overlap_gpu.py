"""GPU tests of M3-S1 (docs/HANDOFF-M3-S1-STAGING-OVERLAP.md): cold-expert staging overlap ($DSV41_STAGE).
Skipped without CUDA/tilelang (Mac); on the box a missing/broken deep_gemm FAILS (it is under test).

  * s1_stage_proj on the device == stage_proj_emulated per projection (weights / scales / both; real w1 row = EVEN
    path, an odd size = masked path; nprog 48 / 16 / 8), and the three projections == m1_stage_rows byte for byte;
  * the copy-engine path (ce_plan + one cudaMemcpyAsync per expert x projection from pinned UVA host memory + the
    scale-only launch) == m1_stage_rows;
  * mini4 6-row dg-inplace StaticStep (map: 4 of 8 experts cold per layer, so most layer-steps stage) in every mode
    == the default `kernel` mode, eager: torch.equal on logits, main_hidden, caches, spare slots, packed SF;
    kernel-overlap captured: graph == static-eager (fork/join inside the capture, no host sync); a ce mode under
    capture raises StageCaptureError; overflow 0; graph replay ms/step of kernel vs kernel-overlap on the mini
    (tiny shapes: a smoke number, labeled);
  * the probe (bench/staging_overlap_bench.py probe, quick: 4 rounds, 10 reps) runs end to end and its byte / output
    checks pass; receipt scratch/staging-overlap-probe-quick-<date>.json.

Run on the box exactly as tests/gpu/test_moe_rows_gpu.py (same image, mounts, DG_JIT_CACHE_DIR):
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py
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
from test_static_rows_gpu import COUNTS, ROWS, _assert_rows, _model, _need_step, _static, _toks  # noqa: E402

RESULTS = {}


def _layer(N, KH, dev="cuda", seed=0, pin=True):
    """The synthetic layer of test_moe_rows_gpu (E 8, hot [6,1,3,0,5], cold [2,4,7] in pinned host memory; pin=False:
    plain CPU tensors, tests/test_staging_overlap_cpu.py's dry run of this file's argument plumbing)."""
    from runtime import dg_inplace as DI

    KS = KH * 2 // 32
    g = torch.Generator().manual_seed(seed + N)
    E, hot, cold = 8, [6, 1, 3, 0, 5], [2, 4, 7]
    H, S = len(hot), 3
    src_w = {(p, e): torch.randint(-128, 128, (N, KH), generator=g, device="cpu").to(torch.int8) for p in DI.PROJS
             for e in range(E)}
    src_s = {(p, e): torch.randint(100, 140, (N, KS), generator=g, device="cpu").to(torch.uint8) for p in DI.PROJS
             for e in range(E)}
    keep = (lambda t: t.pin_memory()) if pin else (lambda t: t.clone())
    host_w = {(p, e): keep(src_w[(p, e)]) for p in DI.PROJS for e in cold}
    host_s = {(p, e): keep(src_s[(p, e)]) for p in DI.PROJS for e in cold}
    return dict(N=N, KH=KH, KS=KS, E=E, hot=hot, cold=cold, H=H, S=S, src_w=src_w, src_s=src_s, host_w=host_w,
                host_s=host_s)


def _dsts(L, step_sf, dev="cuda"):
    from runtime import dg_inplace as DI
    from runtime import moe_rows as MR

    N, KH, KS, E, H, S = L["N"], L["KH"], L["KS"], L["E"], L["H"], L["S"]
    hot = L["hot"]
    full = {p: torch.zeros(H + S, N, KH, dtype=torch.int8, device=dev) for p in DI.PROJS}
    hs = {p: torch.stack([L["src_s"][(p, e)] for e in hot]).to(dev) for p in DI.PROJS}
    for s_, e in enumerate(hot):
        for p in DI.PROJS:
            full[p][s_].copy_(L["src_w"][(p, e)])
    C = DI.sf_words(KS)
    sf = {p: DI.packed_sf_zeros(H + S, N, C, dev) for p in DI.PROJS}
    dsts = [full[p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cpu").to(dev)
    ptab = torch.zeros(6, E, dtype=torch.int64, device="cpu")
    reg = {}
    for e in range(E):
        for i, p in enumerate(DI.PROJS):
            wt = full[p][hot.index(e)] if e in hot else L["host_w"][(p, e)]
            st_ = hs[p][hot.index(e)] if e in hot else L["host_s"][(p, e)]
            ptab[i, e], ptab[3 + i, e] = wt.data_ptr(), st_.data_ptr()
            reg[wt.data_ptr()] = wt.reshape(-1).view(torch.uint8)
            reg[st_.data_ptr()] = st_.reshape(-1).view(torch.uint8)
    gtab = torch.full((E,), -1, dtype=torch.int32, device="cpu")
    for s_, e in enumerate(hot):
        gtab[e] = s_
    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7]], device=dev)
    m = MR.dedup_routes(idx, torch.rand(3, 3, device=dev), 4)
    inpl = {"gtab": gtab.to(dev), "H": H, "S": S, "dummy": torch.tensor(6, device=dev), "has_dummy": True,
            "v2": True, "layout": "contiguous", "G": H + S, "gscr": 0, "step_sf": step_sf}
    ext = MR.rows_layout(m, inpl, 4, 3, 3)
    nw = 9 if step_sf else S
    args = (ptab.to(dev), m["uniq"], gtab.to(dev), dtab, dsts, ext["sidx"], ext["wl"], H, N * KH, ((N, C), (N, C)),
            nw)
    return dict(full=full, sf=sf, dsts=dsts, reg=reg, args=args, m=m, ext=ext, gtab=gtab, ptab=ptab, hs=hs)


# ------------------------------------------------------------------------------------------------ kernels
def test_s1_stage_proj_on_device_equals_contract():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime import staging_overlap as SO
    from runtime import static_kernels as SK

    n = 0
    for N, KH in ((2304, 2560), (40, 64)):           # real w1 row (EVEN path) and an odd size (masked path)
        L = _layer(N, KH)
        for step_sf in (False, True):
            for nprog in (48, 16, 8):
                for pj in range(3):
                    for weights, scales in ((True, True), (False, True), (True, False)):
                        outs = []
                        for use_kernel in (False, True):
                            D = _dsts(L, step_sf)
                            if use_kernel:
                                SO.stage_proj(*D["args"], pj, weights=weights, scales=scales, nprog=nprog)
                                torch.cuda.synchronize()
                            else:
                                SO.stage_proj_emulated(D["reg"].__getitem__, *D["args"], pj, weights, scales)
                            outs.append([t.clone() for t in D["dsts"]])
                        for i, (a_, b_) in enumerate(zip(*outs)):
                            assert torch.equal(a_, b_), (N, step_sf, nprog, pj, weights, scales, i)
                        n += 1
            # the three projections == m1_stage_rows
            D1, D2 = _dsts(L, step_sf), _dsts(L, step_sf)
            SK.stage_rows(*D1["args"])
            for pj in range(3):
                SO.stage_proj(*D2["args"], pj)
            torch.cuda.synchronize()
            for i, (a_, b_) in enumerate(zip(D1["dsts"], D2["dsts"])):
                assert torch.equal(a_, b_), (N, step_sf, "s1 x 3 vs m1", i)
            n += 1
    print(f"  s1_stage_proj: {n} device cases == contract / == m1_stage_rows")


def test_ce_path_on_device_equals_kernel():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime import staging_overlap as SO
    from runtime import static_kernels as SK

    for N, KH in ((2304, 2560), (40, 64)):
        L = _layer(N, KH, seed=3)
        for step_sf in (False, True):
            D1, D2 = _dsts(L, step_sf), _dsts(L, step_sf)
            SK.stage_rows(*D1["args"])
            a = D2["args"]
            t = {"gtab": a[2], "H": L["H"], "ptab6": a[0], "w": {p: D2["full"][p] for p in DI.PROJS}}
            plan = SO.ce_plan(t, {"wl": a[6], "uniq": a[1], "sidx": a[5]}, a[-1])
            assert [it["e"] for it in plan] == [2, 4, 7], plan
            for pj in range(3):
                SO.ce_copy_proj(t, plan, pj, a[8])
                SO.stage_proj(*a, pj, weights=False)
            torch.cuda.synchronize()
            for i, (x, y) in enumerate(zip(D1["dsts"], D2["dsts"])):
                assert torch.equal(x, y), (N, step_sf, "ce vs m1", i)
    print("  ce: cudaMemcpyAsync from pinned UVA + scale launch == m1_stage_rows (EVEN + masked sizes, both stores)")


# ------------------------------------------------------------------------------------------------ the 6-row step
def _snap(st, pool):
    from runtime import dg_inplace as DI

    slots = [pool.hbm_full[l][p][len(pool.hot[l]):].clone() for l in range(pool.n_layers) for p in DI.PROJS]
    sf = [st.dg.layers[l]["sf"][p][st.dg.layers[l]["H"]:].clone() for l in sorted(st.dg.layers) for p in DI.PROJS]
    return slots, sf


def _replay_ms(st, model, ids, toks, n):
    from test_static_rows_gpu import _prefill

    _prefill(model, ids)
    with torch.inference_mode():
        st.step(toks[:st.rows], len(ids), graph=True)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for s in range(1, n):
            st.step(toks[s * st.rows:(s + 1) * st.rows], len(ids) + s * st.rows, graph=True)
        e1.record()
        torch.cuda.synchronize()
    return e0.elapsed_time(e1) / max(1, n - 1)


def test_static6_modes_equal_kernel_mini4():
    _need_dg()
    from runtime import staging_overlap as SO

    prev = os.environ.get(SO.STAGE_ENV)
    rec = {}
    try:
        with _model("mini4") as (model, pool):
            hid = (len(model.layers) - 1,)
            n = 4
            ids, toks = _toks(18, ROWS * n, 2026)
            kw = dict(rows=ROWS, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8)
            os.environ[SO.STAGE_ENV] = "kernel"
            st = _need_step(model, pool, **kw)
            assert st.stage_ov.mode == "kernel"
            base = _static(model, st, ids, toks, n)
            base_snap = _snap(st, pool)
            st.capture()
            bg = _static(model, st, ids, toks, n, graph=True)
            _assert_rows(bg[0], base[0], "kernel: graph vs eager")
            rec["kernel_graph_ms_per_step_mini"] = round(_replay_ms(st, model, ids, toks + toks, 2 * n), 4)
            st.detach()
            for mode in ("kernel-overlap", "ce", "ce-overlap"):
                os.environ[SO.STAGE_ENV] = mode
                st = _need_step(model, pool, **kw)
                assert st.stage_ov.mode == mode and st.describe()["stage"]["mode"] == mode
                a = _static(model, st, ids, toks, n)
                _assert_rows(a[0], base[0], f"{mode} vs kernel (eager)")
                assert all(torch.equal(x, y) for x, y in zip(a[1], base[1])), mode
                for k_ in base[2]:
                    assert torch.equal(a[2][k_], base[2][k_]), (mode, k_)
                sl, sf = _snap(st, pool)
                assert all(torch.equal(x, y) for x, y in zip(sl, base_snap[0])), (mode, "spare slots")
                assert all(torch.equal(x, y) for x, y in zip(sf, base_snap[1])), (mode, "packed SF")
                stats = st.stage_ov.stats
                assert stats["launches"] == len(model.layers) * n, stats
                if mode.startswith("ce"):
                    assert stats["ce_copies"] > 0, "no cold expert was staged: the ce path was not exercised"
                    try:
                        st.capture()
                        raise AssertionError(f"{mode} captured without StageCaptureError")
                    except SO.StageCaptureError:
                        pass
                    except RuntimeError as e:          # torch may wrap the error raised inside capture
                        assert "eager-only" in str(e), e
                else:
                    assert st.stage_ov.side is not None
                    st.capture()
                    ag = _static(model, st, ids, toks, n, graph=True)
                    _assert_rows(ag[0], base[0], f"{mode}: graph vs kernel eager")
                    _assert_rows(ag[0], bg[0], f"{mode}: graph vs kernel graph")
                    rec[f"{mode}_graph_ms_per_step_mini"] = round(_replay_ms(st, model, ids, toks + toks, 2 * n), 4)
                ov = st.overflow_counts()
                assert ov["steps"] == 0 and ov["experts"] == 0, ov
                rec[mode] = {"stats": dict(stats), "equal": True}
                st.detach()
                COUNTS["gates"] += 1
    finally:
        if prev is None:
            os.environ.pop(SO.STAGE_ENV, None)
        else:
            os.environ[SO.STAGE_ENV] = prev
    rec["label"] = "[M] mini4 (tiny shapes): graph replay ms/step, a smoke number, not a performance claim"
    RESULTS["mini4"] = rec
    print("  mini4:", json.dumps(rec))


# ------------------------------------------------------------------------------------------------ the probe
def test_probe_quick():
    _need_dg()
    import staging_overlap_bench as B

    out = os.path.join(REPO, "scratch", f"staging-overlap-probe-quick-{time.strftime('%Y%m%d')}.json")
    rc = B.main(["probe", "--rounds", "4", "--reps", "10", "--hot", "48", "--out", out])
    rep = json.load(open(out))["probe"]
    assert not rep["errors"], rep["errors"]
    ch = rep["checks"]
    assert ch["serial_staged_equal_host"], ch
    for k_, v in ch.items():
        if isinstance(v, dict):
            assert all(v.values()), (k_, v)
    RESULTS["probe_quick"] = {"bandwidth_c4": rep["bandwidth"].get("4") or rep["bandwidth"].get(4),
                              "overlap_c2": rep["overlap"].get("2") or rep["overlap"].get(2),
                              "model_measured": rep.get("model_measured")}
    print("  probe (quick):", json.dumps(RESULTS["probe_quick"], default=str)[:1500])
    assert rc == 0


TESTS = [test_s1_stage_proj_on_device_equals_contract, test_ce_path_on_device_equals_kernel,
         test_static6_modes_equal_kernel_mini4, test_probe_quick]

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
    with open(os.path.join(REPO, "scratch", f"staging-overlap-gpu-test-{time.strftime('%Y%m%d')}.json"), "w") as f:
        json.dump({"results": results, "data": RESULTS}, f, indent=1, default=str)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
