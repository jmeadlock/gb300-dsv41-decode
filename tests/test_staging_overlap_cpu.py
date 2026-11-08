"""Mac CPU tests of M3-S1 (docs/HANDOFF-M3-S1-STAGING-OVERLAP.md): the cold-expert staging overlap ($DSV41_STAGE).
`.venv/bin/python -m pytest -q tests/test_staging_overlap_cpu.py`

  * s1_stage_proj (tests/triton_shim.py, program by program) == stage_proj_emulated for each projection, weights /
    scales / both, EVEN and masked row sizes, both SF stores; the three projections together == m1_stage_rows (shim)
    == dg_inplace.stage_rows_emulated: the same bytes, nothing else written;
  * the copy-engine path (ce_plan + ce_copy_proj + the scale-only launch) writes the same bytes;
  * the 6-row dg-inplace StaticStep in every mode (kernel-overlap, ce, ce-overlap) == the default kernel mode:
    torch.equal on every row's logits, main_hidden, the caches, the spare slots and the packed SF; the default
    launches m1_stage_rows once per layer and nothing of staging_overlap; the overlap modes stage w1 -> w3 -> w2,
    hoist the shared expert, and wait on projection p right before GEMM p (call order recorded); fused glue too;
  * a ce mode under graph capture raises StageCaptureError; mode validation; the [E] model's invariants; the bench's
    `model` subcommand runs on the Mac; vllm is never imported.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

from runtime import dg_inplace as DI
from runtime import moe_rows as MR
from runtime import staging_overlap as SO
from test_dg_inplace_cpu import _fill_registry, _static_kernels_under_shim, _synthetic_layer
from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens, emu,  # noqa: F401
                                  rows_dg, rows_glue)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def KS():
    import triton_shim

    mod = triton_shim.load_module_under_shim("runtime/staging_overlap.py", "staging_overlap_shim")
    mod.KERNELS_ON_CPU = True
    return mod, triton_shim


def _case(step_sf, S=2, N=8):
    """The synthetic layer (E 8, H 5 hot [6,1,3,0,5], cold [2,4,7], 3 spare slots), routing 3 rows x 3 experts over
    0, 1, 2, 4, 7 (cold 2, 4, 7 -> compact 0, 1, 2); S = the spare budget (overflow beyond)."""
    L = _synthetic_layer(N=N, seed=5)
    H, KS_ = L["H"], L["KS"]
    sf = {p: DI.packed_sf_zeros(H + L["S"], N, KS_ // 4, "cpu") for p in DI.PROJS}
    dsts = [L["full"][p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64)
    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7]])
    m = MR.dedup_routes(idx, torch.rand(3, 3), 4)
    inpl = {"gtab": L["gtab"], "H": H, "S": S, "dummy": torch.tensor(6), "has_dummy": True, "v2": True,
            "layout": "contiguous", "G": H + L["S"], "gscr": 0, "step_sf": step_sf}
    ext = MR.rows_layout(m, inpl, 4, 3, 3)
    n_work = 9 if step_sf else S
    args = (L["ptab6"], m["uniq"], L["gtab"], dtab, dsts, ext["sidx"], ext["wl"], H, N * L["KH"],
            ((N, KS_ // 4), (N, KS_ // 4)), n_work)
    return L, sf, dsts, m, ext, args


def _snap(dsts):
    return [t.clone() for t in dsts]


# ------------------------------------------------------------------------------------------------ the kernel
@pytest.mark.parametrize("step_sf", [False, True])
@pytest.mark.parametrize("N,block_bytes,nprog", [(8, 64, 2), (132, 64, 3)])
@pytest.mark.parametrize("weights,scales", [(True, True), (False, True), (True, False)])
def test_s1_stage_proj_body_equals_contract(KS, step_sf, N, block_bytes, nprog, weights, scales):
    mod, shim = KS
    for pj in range(3):
        outs = []
        for use_kernel in (False, True):
            L, sf, dsts, m, ext, args = _case(step_sf, N=N)
            before = _snap(dsts)
            if use_kernel:
                shim.ADDRESSES.clear()
                shim.ADDRESSES.register(*dsts, *L["src_w"].values(), *L["src_s"].values())
                try:
                    mod.stage_proj(*args, pj, weights=weights, scales=scales, block_bytes=block_bytes, nprog=nprog)
                finally:
                    shim.ADDRESSES.clear()
                assert mod.s1_stage_proj.launches[-1]["grid"] == (args[-1], nprog)
            else:
                SO.stage_proj_emulated(L["reg"].__getitem__, *args, pj, weights, scales)
            outs.append(_snap(dsts))
            # only projection pj's weight pool / packed SF changed, and only what was asked for
            for i, (a_, b_) in enumerate(zip(before, dsts)):
                touched = (i == pj and weights) or (i == 3 + pj and scales)
                if not touched:
                    assert torch.equal(a_, b_), (pj, i)
        for i, (a_, b_) in enumerate(zip(*outs)):
            assert torch.equal(a_, b_), (pj, i)


@pytest.mark.parametrize("step_sf", [False, True])
@pytest.mark.parametrize("S", [3, 2, 1])
def test_three_projections_equal_m1_stage_rows(KS, step_sf, S):
    """s1 x {w1, w3, w2} (shim) == m1_stage_rows (shim) == stage_rows_emulated, byte for byte."""
    mod, shim = KS
    SK, shim2 = _static_kernels_under_shim()
    res = {}
    for how in ("emu", "m1", "s1"):
        L, sf, dsts, m, ext, args = _case(step_sf, S=S, N=132)
        if how == "emu":
            DI.stage_rows_emulated(L["reg"].__getitem__, *args)
        else:
            sh = shim if how == "s1" else shim2
            sh.ADDRESSES.clear()
            sh.ADDRESSES.register(*dsts, *L["src_w"].values(), *L["src_s"].values())
            try:
                if how == "m1":
                    SK.stage_rows(*args, block_bytes=64, nprog=3)
                else:
                    for pj in range(3):
                        mod.stage_proj(*args, pj, block_bytes=64, nprog=3)
            finally:
                sh.ADDRESSES.clear()
        res[how] = _snap(dsts)
    for how in ("m1", "s1"):
        for i, (a_, b_) in enumerate(zip(res["emu"], res[how])):
            assert torch.equal(a_, b_), (how, i)
    L, sf, dsts, m, ext, args = _case(step_sf, S=S, N=132)
    H = L["H"]
    w = res["s1"][0]
    for c, e in enumerate([2, 4, 7][:S]):
        assert torch.equal(w[H + c], L["src_w"][("w1", e)])


@pytest.mark.parametrize("step_sf", [False, True])
def test_ce_path_writes_the_same_bytes(monkeypatch, step_sf):
    """ce_plan (host) + one copy per staged cold expert x projection + the scale-only launch == stage_rows."""
    L, sf, dsts, m, ext, args = _case(step_sf, S=3, N=132)
    DI.stage_rows_emulated(L["reg"].__getitem__, *args)
    want = _snap(dsts)
    L, sf, dsts, m, ext, args = _case(step_sf, S=3, N=132)
    monkeypatch.setattr(SO, "CPU_RESOLVE", L["reg"].__getitem__)
    t = {"gtab": L["gtab"], "H": L["H"], "ptab6": L["ptab6"], "w": {p: L["full"][p] for p in DI.PROJS}}
    mm = {"wl": ext["wl"], "uniq": m["uniq"], "sidx": ext["sidx"]}
    plan = SO.ce_plan(t, mm, args[-1])
    assert [it["e"] for it in plan] == [2, 4, 7] and [it["grp"] for it in plan] == [L["H"] + c for c in range(3)]
    n = 0
    for pj in range(3):
        n += SO.ce_copy_proj(t, plan, pj, args[8])
        SO.stage_proj(*args, pj, weights=False)
    assert n == 9
    for i, (a_, b_) in enumerate(zip(want, dsts)):
        assert torch.equal(a_, b_), i


def test_stage_proj_validation(monkeypatch):
    L, sf, dsts, m, ext, args = _case(False)
    with pytest.raises(ValueError, match="projection"):
        SO.stage_proj(*args, 3)
    with pytest.raises(ValueError, match="projection"):
        SO.stage_proj(*args, 0, weights=False, scales=False)
    bad = list(args)
    bad[-1] = 0
    with pytest.raises(ValueError, match="n_work"):
        SO.stage_proj(*bad, 0)
    monkeypatch.setattr(SO, "CPU_RESOLVE", None)
    with pytest.raises(RuntimeError, match="CPU_RESOLVE"):
        SO.stage_proj(*args, 0)


# ------------------------------------------------------------------------------------------------ modes
def test_stage_mode_parsing(monkeypatch):
    monkeypatch.delenv(SO.STAGE_ENV, raising=False)
    assert SO.stage_mode() == "kernel"
    for v in SO.STAGE_MODES:
        monkeypatch.setenv(SO.STAGE_ENV, v)
        assert SO.stage_mode() == v
    monkeypatch.setenv(SO.STAGE_ENV, "dma")
    with pytest.raises(ValueError, match="DSV41_STAGE"):
        SO.stage_mode()
    assert SO.CAPTURABLE == {"kernel": True, "kernel-overlap": True, "ce": False, "ce-overlap": False}


@pytest.mark.parametrize("mode", ["ce", "ce-overlap"])
def test_ce_refuses_graph_capture(monkeypatch, mode):
    class Owner:
        dev = torch.device("cpu")
        dg = None

    ov = SO.StageOverlap(Owner(), mode, 2)
    monkeypatch.setattr(ov, "_capturing", lambda: True)

    class SM:
        layer = 0
        _stage_pending = [False] * 3

    with pytest.raises(SO.StageCaptureError, match="eager-only"):
        ov.launch(SM(), {}, {})
    assert ov.describe()["capturable"] is False and ov.describe()["mode"] == mode


def _registry(pool):
    class Reg(dict):
        def __missing__(self, key):
            _fill_registry(self, pool)
            if key not in self.keys():
                raise KeyError(key)
            return dict.get(self, key)

    reg = Reg()
    _fill_registry(reg, pool)
    return reg


def _run_mode(model, pool, fk, monkeypatch, mode, ids, toks, n, hid, **kw):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(SO.STAGE_ENV, mode)
    s0 = fk.calls.get("stage_rows", 0)
    st = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-inplace", _allow_cpu=True, **kw)
    assert st.stage_ov is not None and st.stage_ov.mode == mode and st.describe()["stage"]["mode"] == mode
    st.stage_ov.trace = []
    calls = []
    for sm in st.moes:                                   # record the order of the routed GEMMs and the shared expert
        g0 = sm._gemm

        def gemm(p, which, sm=sm, g0=g0):
            calls.append(("gemm", sm.layer, p))
            return g0(p, which)

        sm._gemm = gemm
    monkeypatch.setattr(MR._RowsMoEBase, "_shared", lambda self, x: (calls.append(("shared", self.layer)),
                                                                     _orig_shared(self, x))[1])
    out = _static_run(model, st, ids, toks, n)
    monkeypatch.setattr(MR._RowsMoEBase, "_shared", _orig_shared)
    n_m1 = fk.calls.get("stage_rows", 0) - s0
    slots = {(l, p): pool.hbm_full[l][p][len(pool.hot[l]):].clone() for l in range(pool.n_layers) for p in DI.PROJS}
    sfs = {(l, p): st.dg.layers[l]["sf"][p][st.dg.layers[l]["H"]:].clone() for l in st.dg.layers for p in DI.PROJS}
    rec = {"out": out, "n_m1": n_m1, "trace": list(st.stage_ov.trace), "calls": calls, "slots": slots, "sf": sfs,
           "stats": dict(st.stage_ov.stats), "ovf": st.overflow_counts()}
    st.detach()
    return rec


_orig_shared = MR._RowsMoEBase._shared


@pytest.mark.parametrize("kind", ["mini4", "mini"])
def test_static6_modes_equal_kernel(rows_dg, monkeypatch, kind):
    model, pool, fk, fake = rows_dg(kind)
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    hid = (len(model.layers) - 1,)
    n = 3
    ids, toks = _tf_tokens(18, 6 * n, 404)
    n_layers = len(model.layers)
    base = _run_mode(model, pool, fk, monkeypatch, "kernel", ids, toks, n, hid)
    assert base["n_m1"] == n_layers * n and base["trace"] == [] and base["stats"]["launches"] == 0
    assert base["ovf"]["steps"] == 0
    # kernel mode: the shared expert after the three routed GEMMs (today's order)
    first = [c for c in base["calls"] if c[1] == 0][:4]
    assert [c[0] for c in first] == ["gemm", "gemm", "gemm", "shared"], first
    staged_any = False
    for mode in ("kernel-overlap", "ce", "ce-overlap"):
        got = _run_mode(model, pool, fk, monkeypatch, mode, ids, toks, n, hid)
        a_lg, a_h, a_s = got["out"]
        b_lg, b_h, b_s = base["out"]
        _assert_rows_equal(a_lg, b_lg, f"{kind}: 6-row {mode} vs kernel")
        assert all(torch.equal(x, y) for x, y in zip(a_h, b_h)), mode
        for k_ in b_s:
            assert torch.equal(a_s[k_], b_s[k_]), (mode, k_)
        for k_ in base["slots"]:
            assert torch.equal(got["slots"][k_], base["slots"][k_]), (mode, "spare slot bytes", k_)
        for k_ in base["sf"]:
            assert torch.equal(got["sf"][k_], base["sf"][k_]), (mode, "packed SF", k_)
        assert got["n_m1"] == 0, mode                     # no m1_stage_rows launch in the other modes
        st_ = got["stats"]
        assert st_["launches"] == n_layers * n and st_["proj_launches"] == 3 * n_layers * n, (mode, st_)
        if mode.startswith("ce"):
            assert st_["host_syncs"] == n_layers * n
            staged_any |= st_["ce_copies"] > 0
        tr = got["trace"]
        per = [e for e in tr if e[1] == 0][:7]
        if SO.OVERLAP[mode]:
            assert per == [("stage", 0, 0), ("stage", 0, 1), ("stage", 0, 2), ("hoist", 0), ("wait", 0, 0),
                           ("wait", 0, 1), ("wait", 0, 2)], per
            assert st_["shared_hoisted"] == n_layers * n
            first = [c for c in got["calls"] if c[1] == 0][:4]
            assert [c[0] for c in first] == ["shared", "gemm", "gemm", "gemm"], first
            assert [c[2] for c in first[1:]] == ["w1", "w3", "w2"]
        else:
            assert st_["shared_hoisted"] == 0
        assert got["ovf"]["steps"] == 0
    assert staged_any, "the mini map routed no cold expert: the ce path was not exercised"


def test_static6_kernel_overlap_fused_glue(rows_glue, monkeypatch):
    """glue_impl fused (the shared expert through FusedGlue.shared_expert, the fused combine): kernel-overlap ==
    kernel, torch.equal."""
    from runtime.static_step import StaticStep

    model, pool, fk, fake = rows_glue("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 9)
    outs = {}
    for mode in ("kernel", "kernel-overlap"):
        monkeypatch.setenv(SO.STAGE_ENV, mode)
        st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace",
                        glue_impl="fused", _allow_cpu=True)
        assert st.glue is not None and st.glue.on("swiglu_shared") and st.glue.on("moe_combine")
        outs[mode] = _static_run(model, st, ids, toks, 2)
        if mode != "kernel":
            assert st.stage_ov.stats["shared_hoisted"] == 2 * len(model.layers)
        st.detach()
    a, b = outs["kernel-overlap"], outs["kernel"]
    _assert_rows_equal(a[0], b[0], "fused glue: kernel-overlap vs kernel")
    assert all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
    for k_ in b[2]:
        assert torch.equal(a[2][k_], b[2][k_]), k_


def test_one_row_and_other_impls_have_no_hook(rows_dg, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setenv(SO.STAGE_ENV, "kernel-overlap")
    s1 = StaticStep(model, pool, max_pos=255, moe_impl="dg-inplace", _allow_cpu=True)
    assert s1.stage_ov is None and s1.describe()["stage"] is None
    s1.detach()
    s2 = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-masked", _allow_cpu=True)
    assert s2.stage_ov is None
    s2.detach()


# ------------------------------------------------------------------------------------------------ model / bench
def test_gpu_file_plumbing_dry_run(monkeypatch):
    """tests/gpu/test_staging_overlap_gpu.py's synthetic layer + argument tuples on the CPU: the contract, the CPU
    stage_proj path and the ce path agree with stage_rows_emulated (catches arg-order slips before a box window)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("so_gpu", os.path.join(REPO, "tests", "gpu",
                                                                         "test_staging_overlap_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    for N, KH in ((40, 64), (16, 128)):
        L = G._layer(N, KH, pin=False)
        for step_sf in (False, True):
            D1 = G._dsts(L, step_sf, dev="cpu")
            DI.stage_rows_emulated(D1["reg"].__getitem__, *D1["args"])
            D2 = G._dsts(L, step_sf, dev="cpu")
            monkeypatch.setattr(SO, "CPU_RESOLVE", D2["reg"].__getitem__)
            for pj in range(3):
                SO.stage_proj(*D2["args"], pj)
            D3 = G._dsts(L, step_sf, dev="cpu")
            monkeypatch.setattr(SO, "CPU_RESOLVE", D3["reg"].__getitem__)
            a = D3["args"]
            t = {"gtab": a[2], "H": L["H"], "ptab6": a[0], "w": {p: D3["full"][p] for p in DI.PROJS}}
            plan = SO.ce_plan(t, {"wl": a[6], "uniq": a[1], "sidx": a[5]}, a[-1])
            assert [it["e"] for it in plan] == [2, 4, 7]
            for pj in range(3):
                SO.ce_copy_proj(t, plan, pj, a[8])
                SO.stage_proj(*a, pj, weights=False)
            for i, (x, y, z) in enumerate(zip(D1["dsts"], D2["dsts"], D3["dsts"])):
                assert torch.equal(x, y) and torch.equal(x, z), (N, step_sf, i)
    assert G.TESTS and all(callable(t) for t in G.TESTS)


def test_model_invariants():
    p = SO.MODEL_DEFAULTS
    for G in (p["kernel_GBps"], p["ce_GBps"]):
        prev_s = prev_o = -1.0
        for c in range(0, 9):
            s = SO.layer_exposed_us(c, G, overlap=False)
            o = SO.layer_exposed_us(c, G)
            assert o <= s and s > prev_s and o >= prev_o
            prev_s, prev_o = s, o
    assert SO.layer_exposed_us(0, 261.0) == 0.0
    assert SO.layer_exposed_us(1, 261.0) == 0.0          # one cold expert hides under shared + GEMM w1 / w3
    r = SO.model_step()
    e = r["staging_exposed_ms_per_step"]
    assert e["kernel-overlap"] < e["kernel (serial, today)"] and e["ce-overlap [needs capturable CE]"] < e[
        "kernel-overlap"]
    assert abs(r["step_ms_E"]["kernel (serial, today)"] - p["step_ms"]) < 1e-6
    assert abs(sum(r["dist"]) - 1.0) < 1e-3


def test_bench_model_subcommand_runs_on_cpu(tmp_path):
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import staging_overlap_bench as B

    out = tmp_path / "m.json"
    assert B.main(["model", "--out", str(out)]) == 0
    d = json.load(open(out))
    assert "kernel-overlap" in d["model"]["staging_exposed_ms_per_step"]
    ov = {c: {"gemm-only": {"exposed_us": 0.0}, "serial": {"exposed_us": 75.0 * c + 3},
              "kernel-overlap/nprog48": {"exposed_us": max(0.0, 70.0 * c - 100)}} for c in range(B.CMAX + 1)}
    mm = B.model_from_measured(ov, 1.58)
    assert mm["ms_per_step"]["kernel-overlap/nprog48"] < mm["ms_per_step"]["serial"]
    assert abs(mm["step_ms_E"]["serial"] - 20.304) < 1e-6
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
