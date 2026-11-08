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
  * M3-S1-FIX: round 1's box failure ('kernel-overlap', 'L1.ckv') reproduced in the GPU file's own sequence with
    kernel vs kernel (replay-loop cache residue, not a race) and fixed by the GPU file's _checkpoint / _restore; the
    stream-ordering model (torch dispatch over the kernel-overlap step: no main-stream byte access overlaps the
    forked staging's between fork and join, ref + fused glue, two negative controls caught); the per-projection
    launch knobs; the guarded none-diag timing diagnostic; the bench's per-projection --cfgs.
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
    assert SO.stage_mode() == "kernel-overlap"   # default since 11/07 (box 19.34 vs 19.93, bit-identical)
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


def _gpu_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("so_gpu", os.path.join(REPO, "tests", "gpu",
                                                                         "test_staging_overlap_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


# ------------------------------------------------------------------------------------------------ M3-S1-FIX
def test_round1_l1ckv_is_replay_residue_and_restore_fixes_it(rows_dg, monkeypatch):
    """Round 1's box failure, AssertionError ('kernel-overlap', 'L1.ckv'), in the GPU file's own sequence on the
    CPU: base = kernel n = 4 steps; then 2n steps (the GPU file's _replay_ms(toks + toks, 2n)); then an n-step run
    is compared to the base. The SAME mode (kernel vs kernel, no fork, no stream) fails the same way: logits and
    main_hidden equal, the caches differ exactly at the compressed entries only the 2n-step run wrote (L1 ratio 2:
    entries 21..32 = positions 42..65). With the GPU file's _checkpoint / _restore before every run, kernel and
    kernel-overlap are equal to the base on everything, run after run."""
    from runtime.static_step import StaticStep

    G = _gpu_module()
    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    hid = (len(model.layers) - 1,)
    n = 4
    ids, toks = _tf_tokens(18, 6 * n, 2026)

    def build(mode):
        monkeypatch.setenv(SO.STAGE_ENV, mode)
        return StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8,
                          _allow_cpu=True)

    st = build("kernel")
    ck = G._checkpoint(model, pool, st)
    base = _static_run(model, st, ids, toks, n)
    base_snap = G._snap(st, pool)
    _static_run(model, st, ids, toks + toks, 2 * n)                 # the replay timing loop's extra positions
    dirty = _static_run(model, st, ids, toks, n)                    # round 1's next run, here in the SAME mode
    _assert_rows_equal(dirty[0], base[0], "residue: logits stay equal")
    assert all(torch.equal(x, y) for x, y in zip(dirty[1], base[1]))
    diff = G._state_diff(dirty[2], base[2])
    assert "L1.ckv" in diff, diff
    last = (18 + 6 * n - 1) // 2                                    # L1 (ratio 2): last entry an n-step run writes
    d = (dirty[2]["L1.ckv"].float() - base[2]["L1.ckv"].float()).abs()
    rows_bad = d.reshape(d.size(0), d.size(1), -1).amax(-1)[0].nonzero().flatten().tolist()
    assert rows_bad and min(rows_bad) > last and max(rows_bad) <= (18 + 12 * n - 1) // 2, rows_bad
    st.detach()
    for mode in ("kernel", "kernel-overlap"):
        st = build(mode)
        for it in range(3):
            G._restore(model, pool, st, ck)
            a = _static_run(model, st, ids, toks, n)
            G._eq_all(a, base, G._snap(st, pool), base_snap, f"{mode} restored #{it}")
        _static_run(model, st, ids, toks + toks, 2 * n)            # dirty it again: the next restore must cope
        G._restore(model, pool, st, ck)
        G._eq_all(_static_run(model, st, ids, toks, n), base, G._snap(st, pool), base_snap, f"{mode} after dirt")
        st.detach()


# ---- the stream-ordering model: every access of the forked staging vs every concurrent main-stream access
def _span(t):
    """[lo, hi) bytes a CPU tensor can touch (strided interior included: conservative)."""
    if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.numel() == 0:
        return None
    es = t.element_size()
    lo = t.data_ptr()
    return lo, lo + sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if s > 0) * es + es


def _flat(v):
    if isinstance(v, torch.Tensor):
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flat(x)


class _StreamModel:
    """Records the step as an event list on the CPU: ("fork", l), side ops (the s1_stage_proj contract run through
    CPU_RESOLVE, plus its pointer / plan-table reads that .tolist() hides from dispatch), ("wait", l, pj), and every
    main-stream aten op with its read / write byte spans (torch dispatch; view ops touch no bytes). On the GPU the
    side op pj runs concurrently with every main op after the fork until the first wait(l, q >= pj) (the side
    stream is serial); `races` lists every overlapping access pair in such a window with a write on either side."""

    def __init__(self, monkeypatch):
        from torch.utils._python_dispatch import TorchDispatchMode

        self.ev = []
        self.stream = "main"
        model = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                sch = func._schema
                ins = [t for a_ in list(args) + list(kwargs.values()) for t in _flat(a_)]
                if any(r.alias_info is not None and not r.alias_info.is_write for r in sch.returns):
                    # a view touches no bytes -- but `to` & co. may alias OR copy: a fresh output storage is a copy
                    st_in = {t.untyped_storage().data_ptr() for t in ins}
                    if all(o.untyped_storage().data_ptr() in st_in for o in _flat(out)):
                        return out
                rd, wr = [], []
                for i, a in enumerate(sch.arguments):
                    v = args[i] if i < len(args) else kwargs.get(a.name)
                    w = a.alias_info is not None and a.alias_info.is_write
                    for t in _flat(v):
                        s = _span(t)
                        if s:
                            (wr if w else rd).append(s)
                model.ev.append(("op", model.stream, str(func), rd, wr))
                return out

        self.mode = Mode
        launch0, proj0, wait0 = SO.StageOverlap.launch, SO.StageOverlap._proj, SO.StageOverlap.wait

        def launch(ov, sm, t, m):
            if ov.mode != "kernel":
                self.ev.append(("fork", sm.layer))
            return launch0(ov, sm, t, m)

        def proj(ov, t, m, pj, n_work, plan, layer):
            prev, self.stream = self.stream, ("side", layer, pj)
            tabs = [m["uniq"], m["sidx"], m["wl"], t["gtab"], t["ptab6"], t["dtab"]]
            self.ev.append(("op", self.stream, "contract-reads", [s for s in map(_span, tabs) if s], []))
            try:
                return proj0(ov, t, m, pj, n_work, plan, layer)
            finally:
                self.stream = prev

        def wait(ov, sm, pj):
            self.ev.append(("wait", sm.layer, pj))
            return wait0(ov, sm, pj)

        monkeypatch.setattr(SO.StageOverlap, "launch", launch)
        monkeypatch.setattr(SO.StageOverlap, "_proj", proj)
        monkeypatch.setattr(SO.StageOverlap, "wait", wait)

    @staticmethod
    def races(ev):
        def hit(xs, ys):
            return any(a < d and c < b for a, b in xs for c, d in ys)

        out, side, open_, layer, forks = [], {}, set(), None, 0
        for e in ev:
            if e[0] == "fork":
                assert not open_, f"L{layer}: forked staging {sorted(open_)} never joined before the next fork"
                layer, side, open_ = e[1], {pj: ([], []) for pj in range(3)}, {0, 1, 2}
                forks += 1
            elif e[0] == "wait":
                assert e[1] == layer, (e, layer)
                open_ -= {pj for pj in open_ if pj <= e[2]}
            elif e[1] != "main":
                _, l_, pj = e[1]
                assert l_ == layer and pj in open_, (e[1], layer, open_)
                side[pj][0].extend(e[3])
                side[pj][1].extend(e[4])
            else:
                for pj in open_:
                    s_rd, s_wr = side[pj]
                    if hit(s_wr, e[3] + e[4]) or hit(s_rd, e[4]):
                        out.append((layer, pj, e[2]))
        assert not open_, f"L{layer}: forked staging {sorted(open_)} never joined at the end of the run"
        return out, forks


@pytest.mark.parametrize("glue", [False, True])
def test_stream_order_model_no_race(request, monkeypatch, glue):
    """M3-S1-FIX: the kernel-overlap step's op sequence (mini4, 6 rows, 2 steps; reference glue and fused glue)
    under the stream model -- no main-stream op between the fork and the join of projection p touches a byte the
    forked staging of p reads or writes with a write on either side (the dedup / plan tables, the spare slots, the
    packed SF; the hoisted shared expert, the act quant and GEMMs w1 / w3 included). Negative control: the same log
    with the w1 wait dropped (GEMM w1 then reads the w1 pool while its staging may still write it) IS a race."""
    from runtime.static_step import StaticStep

    model, pool, fk, fake = request.getfixturevalue("rows_glue" if glue else "rows_dg")("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    monkeypatch.setenv(SO.STAGE_ENV, "kernel-overlap")
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 77)
    kw = {"glue_impl": "fused"} if glue else {}
    st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", _allow_cpu=True,
                    **kw)
    sm_ = _StreamModel(monkeypatch)
    with sm_.mode():
        _static_run(model, st, ids, toks, 2)
    st.detach()
    races, forks = _StreamModel.races(sm_.ev)
    n_side = sum(1 for e in sm_.ev if e[0] == "op" and e[1] != "main")
    n_main = sum(1 for e in sm_.ev if e[0] == "op" and e[1] == "main")
    assert forks == 2 * len(model.layers) and n_side >= 3 * forks and n_main > 100, (forks, n_side, n_main)
    assert not races, races[:10]
    dropped = [e for e in sm_.ev if not (e[0] == "wait" and e[2] == 0)]
    bad, _ = _StreamModel.races(dropped)
    assert bad and all(pj == 0 for _, pj, _ in bad), bad[:5]
    # negative control 2: a main-stream op under the fork that WRITES the plan table the side reads (the work list)
    sm2 = _StreamModel(monkeypatch)
    sh0 = MR._RowsMoEBase._shared

    def shared_dirty(self, x):
        self.last["wl"].add_(0)                                         # same bytes; still a write under the fork
        return sh0(self, x)

    monkeypatch.setattr(MR._RowsMoEBase, "_shared", shared_dirty)
    st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", _allow_cpu=True,
                    **kw)
    with sm2.mode():
        _static_run(model, st, ids, toks, 1)
    st.detach()
    bad2, _ = _StreamModel.races(sm2.ev)
    assert {pj for _, pj, _ in bad2} == {0, 1, 2} and all("add_" in f for _, _, f in bad2), bad2[:5]


def test_per_projection_launch_knobs(monkeypatch):
    """$DSV41_STAGE_OVERLAP_NPROG / _BLOCK_BYTES: one value (all three projections) or three (w1,w3,w2); bad specs
    refuse at build. Bytes are the same for every setting (test_s1_stage_proj_body_equals_contract's grid sweep)."""
    class Owner:
        dev = torch.device("cpu")
        dg = None

    monkeypatch.delenv(SO.NPROG_ENV, raising=False)
    monkeypatch.delenv(SO.BLOCK_ENV, raising=False)
    d = SO.StageOverlap(Owner(), "kernel-overlap", 2).describe()
    assert d["nprog"] == [48, 16, 16] and d["block_bytes"] == [8192, 32768, 32768]   # the 11/07 defaults
    monkeypatch.setenv(SO.NPROG_ENV, "48,16,16")
    monkeypatch.setenv(SO.BLOCK_ENV, "8192,32768,32768")
    ov = SO.StageOverlap(Owner(), "kernel-overlap", 2)
    assert ov.nprog == [48, 16, 16] and ov.block_bytes == [8192, 32768, 32768]
    monkeypatch.setenv(SO.NPROG_ENV, "16")
    assert SO.StageOverlap(Owner(), "kernel-overlap", 2).nprog == [16] * 3
    for bad_n, bad_b in (("48,16", None), ("0", None), (None, "8192,3000,8192")):
        for k, v in ((SO.NPROG_ENV, bad_n), (SO.BLOCK_ENV, bad_b)):
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        with pytest.raises(ValueError):
            SO.StageOverlap(Owner(), "kernel-overlap", 2)


@pytest.mark.parametrize("glue", [False, True])
def test_per_projection_cfg_step_equal(request, monkeypatch, glue):
    """kernel-overlap with NPROG=48,16,16 / BLOCK_BYTES=8192,32768,32768 == kernel on the 6-row step (mini4)."""
    from runtime.static_step import StaticStep

    model, pool, fk, fake = request.getfixturevalue("rows_glue" if glue else "rows_dg")("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 5)
    kw = {"glue_impl": "fused"} if glue else {}
    outs = {}
    for mode, np_, bb in (("kernel", None, None), ("kernel-overlap", "48,16,16", "8192,32768,32768")):
        monkeypatch.setenv(SO.STAGE_ENV, mode)
        for k, v in ((SO.NPROG_ENV, np_), (SO.BLOCK_ENV, bb)):
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", _allow_cpu=True,
                        **kw)
        outs[mode] = _static_run(model, st, ids, toks, 2)
        if mode != "kernel":
            assert st.describe()["stage"]["nprog"] == [48, 16, 16]
        st.detach()
    a, b = outs["kernel-overlap"], outs["kernel"]
    _assert_rows_equal(a[0], b[0], "per-projection cfg vs kernel")
    assert all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
    for k_ in b[2]:
        assert torch.equal(a[2][k_], b[2][k_]), k_


def test_none_diag_is_guarded_and_really_skips_staging(rows_dg, monkeypatch):
    """The timing diagnostic $DSV41_STAGE=none-diag (the ceiling measurement): refused without the ack; with it the
    step never launches staging (the spare slots are not written) and -- mini4 routes cold experts -- the outputs
    are NOT kernel's (so it can never pass for a bit-identical variant); describe() says outputs_valid False."""
    from runtime.static_step import StaticStep

    monkeypatch.delenv(SO.DIAG_ENV, raising=False)
    monkeypatch.setenv(SO.STAGE_ENV, SO.DIAG_MODE)
    with pytest.raises(ValueError, match="wrong outputs"):
        SO.stage_mode()
    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 31)
    outs = {}
    for mode in ("kernel", SO.DIAG_MODE):
        monkeypatch.setenv(SO.STAGE_ENV, mode)
        monkeypatch.setenv(SO.DIAG_ENV, SO.DIAG_ACK)
        st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", _allow_cpu=True)
        before = [pool.hbm_full[l][p][len(pool.hot[l]):].clone() for l in range(pool.n_layers) for p in DI.PROJS]
        s0 = fk.calls.get("stage_rows", 0)
        outs[mode] = _static_run(model, st, ids, toks, 2)
        if mode == SO.DIAG_MODE:
            d = st.describe()["stage"]
            assert d["outputs_valid"] is False and d["stats"]["launches"] == 0
            assert d["stats"]["diag_skipped"] == 2 * len(model.layers)
            assert fk.calls.get("stage_rows", 0) == s0
            after = [pool.hbm_full[l][p][len(pool.hot[l]):] for l in range(pool.n_layers) for p in DI.PROJS]
            assert all(torch.equal(x, y) for x, y in zip(before, after)), "none-diag wrote a spare slot"
        st.detach()
    assert not all(torch.equal(x, y) for x, y in zip(outs[SO.DIAG_MODE][0], outs["kernel"][0])), \
        "none-diag matched kernel: the mini map routed no cold expert, the diagnostic proved nothing"


def test_bench_cfg_parsing():
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import staging_overlap_bench as B

    cf = B.parse_cfgs("48:8192,48:8192/16:32768/16:32768")
    assert cf == [(48, 8192), ((48, 8192), (16, 32768), (16, 32768))]
    assert B._key("kernel-overlap", cf[0]) == "kernel-overlap/np48b8k"
    assert B._key("kernel-overlap", cf[1]) == "kernel-overlap/np48b8k-16b32k-16b32k"
    assert [B._per_proj_cfg(cf[1], p) for p in range(3)] == [(48, 8192), (16, 32768), (16, 32768)]
    assert B._per_proj_cfg(cf[0], 2) == (48, 8192) and B._per_proj_cfg(None, 0) == (None, None)
    with pytest.raises(ValueError):
        B.parse_cfgs("48:8192/16:32768")


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


# ------------------------------------------------------------------------------------------------ M3-S2
# docs/HANDOFF-M3-S2-STAGING-BYTES.md: the sf_store "pool" size derived, the partial pool ($DSV41_SF_POOL_LAYERS) is
# bit-identical to both whole stores (alone and under residency, which must write promoted SF into pooled layers'
# copies only), the cold-count histogram instrument ($DSV41_STAGE_HIST) counts what the step counters count and
# changes nothing, the bench's CPU subcommands run.
REAL_GROUP_B = 2304 * 40 * 4 * 2 + 5120 * 18 * 4          # one pool group's packed SF (w1 + w3 + w2) = 1,105,920 B
GIB = 2 ** 30


def test_m3s2_pool_sf_bytes_derivation():
    class P:
        n_layers = 40
        hbm_full = {l: {"w1": torch.empty(295 + 24, 0)} for l in range(40)}

    N, C = {"w1": 2304, "w3": 2304, "w2": 5120}, {"w1": 40, "w3": 40, "w2": 18}
    assert REAL_GROUP_B == 1_105_920
    assert DI.pool_sf_layer_bytes(P, 7, N, C) == 319 * REAL_GROUP_B
    assert round(DI.pool_sf_bytes(P, N, C) / GIB, 2) == 13.14          # the headline log's "needs 13.14 GiB"


def test_m3s2_sf_pool_plan_rules(monkeypatch):
    LB = [319 * REAL_GROUP_B] * 40
    sB = 319 * REAL_GROUP_B
    free = int(4.93 * GIB)
    plan = lambda spec, head, free=free, cached=(): DI.sf_pool_plan(spec, 40, LB, sB, free, int(head * GIB),  # noqa
                                                                   cached=cached)
    p = plan("all", 6.0)
    assert p["mode"] == "step" and p["layers"] == [] and "using step" in p["why"]
    assert plan("auto", 6.0)["mode"] == "step"                         # 4.93 GiB free < 6.0: not one layer
    p = plan("auto", 2.0)                                               # today's box at headroom 2.0: 7 layers
    assert p["mode"] == "partial" and p["layers"] == list(range(7))
    assert p["need_B"] == 7 * LB[0] + sB and free - p["need_B"] >= 2 * GIB
    assert free - (p["need_B"] + LB[0]) < 2 * GIB                       # an 8th would break the headroom
    assert plan("3", 2.0)["layers"] == [0, 1, 2]
    assert plan("5,39", 2.0)["layers"] == [5, 39]
    assert plan("0", 2.0)["mode"] == "step"
    p = plan("auto", 2.0, cached=[10, 11])                              # already-packed layers cost nothing, first
    assert p["layers"][:0] == [] and {10, 11} <= set(p["layers"]) and len(p["layers"]) == 9
    assert plan("auto", 2.0, free=None)["mode"] == "pool"               # no budget (the Mac): whole
    assert plan("3", 2.0, free=None)["layers"] == [0, 1, 2]
    assert plan("auto", 2.0, free=20 * GIB)["mode"] == "pool" and plan("all", 2.0, free=20 * GIB)["mode"] == "pool"
    for bad in ("abc", "-1", "0,99"):
        with pytest.raises(ValueError):
            DI.parse_sf_pool_layers(bad, 40)
    monkeypatch.delenv(DI.SF_POOL_LAYERS_ENV, raising=False)
    assert DI.parse_sf_pool_layers(None, 40) == ("all", None)           # default = today's all-or-nothing rule


def _m3s2_build(model, pool, monkeypatch, store, spec, **kw):
    from runtime.static_step import StaticStep

    pool._dg_packed_sf = None                                           # a fresh pool copy per build
    if spec is None:
        monkeypatch.delenv(DI.SF_POOL_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(DI.SF_POOL_LAYERS_ENV, spec)
    return StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", rows_sf_store=store, _allow_cpu=True,
                      **kw)


@pytest.mark.parametrize("mode", ["kernel", "kernel-overlap"])
def test_m3s2_partial_pool_bit_identical(rows_dg, monkeypatch, mode):
    """sf_store pool with layers 0 and 2 pooled, 1 and 3 "step" == whole pool == whole step: torch.equal on every
    row's logits, main_hidden, caches and the spare-slot bytes; per-layer work lists (hot slots listed only in step
    layers); a step layer's SF is a StepSFView (not a dict), a pool layer's the pool copy."""
    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    monkeypatch.setenv(SO.STAGE_ENV, mode)
    hid = (len(model.layers) - 1,)
    n_layers = len(model.layers)
    ids, toks = _tf_tokens(18, 18, 77)
    outs, slots = {}, {}
    for name, store, spec in (("step", "step", None), ("pool", "pool", None), ("partial", "pool", "0,2")):
        st = _m3s2_build(model, pool, monkeypatch, store, spec, hidden_layers=hid)
        b = st.dg
        if name == "partial":
            assert b.sf_store == "pool" and b.sf_pool_layers == (0, 2) and "partial pool" in b.sf_store_why
            assert b.sf_layer == {0: "pool", 1: "step", 2: "pool", 3: "step"}
            assert b.describe()["sf_layers"] == {"pool": 2, "step": 2} and b.sf_shared
            for l in range(n_layers):
                t = b.layers[l]
                pooled = l in (0, 2)
                assert isinstance(t["sf"], dict) == pooled and isinstance(t["sf"], DI.StepSFView) != pooled
                assert b.stage_n_work(t) == (b.spare_limit if pooled else b.pairs) and b.step_sf(t) != pooled
            assert b.stage_n_work() == b.pairs                         # no layer given: the conservative superset
        elif name == "pool":
            assert b.sf_store == "pool" and b.sf_pool_layers is None and not b.sf_shared
            assert set(b.sf_layer.values()) == {"pool"}
        else:
            assert set(b.sf_layer.values()) == {"step"}
        outs[name] = _static_run(model, st, ids, toks, 3)
        for sm in st.moes:                                              # the last call's work list
            m = sm.last
            nw = b.stage_n_work(sm.t)
            us = [int(u) for u in m["wl"][:nw].tolist() if u >= 0]
            cold_only = all(int(m["sidx"][u]) >= 0 for u in us)
            assert cold_only == (b.sf_layer[sm.layer] == "pool"), (name, sm.layer, us)
        slots[name] = [pool.hbm_full[l][p][len(pool.hot[l]):].clone() for l in range(n_layers) for p in DI.PROJS]
        st.detach()
    for name in ("pool", "partial"):
        a_, b_ = outs[name], outs["step"]
        _assert_rows_equal(a_[0], b_[0], f"{mode}: {name} vs step")
        assert all(torch.equal(x, y) for x, y in zip(a_[1], b_[1])), name
        for k_ in b_[2]:
            assert torch.equal(a_[2][k_], b_[2][k_]), (name, k_)
        assert all(torch.equal(x, y) for x, y in zip(slots[name], slots["step"])), (name, "spare slots")


def test_m3s2_partial_pool_with_residency_equals_off(emu, monkeypatch):
    """Residency on a partial pool (layers 0, 2 pooled): attach registers the promoted-expert SF target for the
    pooled layers ONLY (a write into the shared step SF would race other layers), and the 6-row step with forced
    + policy swaps == the residency-off whole-step-store run, torch.equal on every row's logits; audit ok."""
    import numpy as np

    from runtime.residency import ResidencyConfig, ResidencyManager, golden_from_index, n_stage_for, sf_store_for
    from test_dg_inplace_cpu import _fake_dg_contig
    from test_dg_moe_cpu import _faithful_act_quant
    from test_residency_cpu import _hot_cold_bias, _install
    from test_residency_rows_cpu import _run6
    from test_static_rows_cpu import _masked_rowwise

    model, pool, _ = emu("mini4")
    fake = _fake_dg_contig()
    fake.m_grouped_fp8_fp4_gemm_nt_masked = _masked_rowwise(fake)
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    monkeypatch.setattr(sys.modules["model"], "act_quant", _faithful_act_quant)
    _install(monkeypatch, pool)
    n = 8
    ids, toks = _tf_tokens(17, 6 * n, 29)
    st = _m3s2_build(model, pool, monkeypatch, "step", None)
    off = _run6(model, st, ids, toks, n)
    st.detach()
    k = model.layers[0].ffn.n_activated_experts
    cfg = ResidencyConfig(drain_n=2, min_abs=1.0, hyst=1.0, halflife=20.0, n_spare=2, n_demote=4, reserve_gib=0.0,
                          force_random=2)
    monkeypatch.setenv(DI.SF_POOL_LAYERS_ENV, "0,2")
    rm = ResidencyManager(pool, model, cfg, n_stage=n_stage_for("dg-inplace", 6, k),
                          sf_store=sf_store_for("dg-inplace", 6))
    st = _m3s2_build(model, pool, monkeypatch, None, "0,2")
    assert st.dg.sf_store == "pool" and st.dg.sf_pool_layers == (0, 2)
    rm.attach(st)
    assert sorted(rm.resident_sf) == [0, 2] and rm.sf_store_got == "pool"
    rng = np.random.default_rng(2)
    on = _run6(model, st, ids, toks, n, rm=rm, after=lambda s: rm.inject_counts(_hot_cold_bias(pool, rng, 30)))
    _assert_rows_equal(on, off, "6-row partial pool + residency vs step store, residency off")
    assert rm.stats["published"] > 0 and rm.state.n_swaps > 0
    moved = {l for l in range(pool.n_layers) for e, s in rm.slot_of[l].items() if s >= len(pool.hot[l])}
    assert moved & {0, 2} and moved & {1, 3}, moved                     # both kinds of layer had promotions
    rm.flush()
    a = rm.audit(golden=golden_from_index(model.runtime_load.index))
    assert a["ok"], a["errors"][:5]
    rm.close()
    st.detach()


def test_m3s2_hist_instrument(rows_dg, monkeypatch, tmp_path):
    """$DSV41_STAGE_HIST: per layer-call cold / hot unique counts == the step's own routed counters (per layer), the
    outputs unchanged, the JSON written by dump(); off by default (no tensors, nothing counted)."""
    from runtime.static_step import StaticStep

    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    hid = (len(model.layers) - 1,)
    n_layers, n = len(model.layers), 3
    ids, toks = _tf_tokens(18, 6 * n, 404)
    outs = {}
    for mode in ("kernel", "kernel-overlap"):
        monkeypatch.setenv(SO.STAGE_ENV, mode)
        for on in (False, True):
            path = tmp_path / f"hist-{mode}.json"
            if on:
                monkeypatch.setenv(SO.HIST_ENV, str(path))
            else:
                monkeypatch.delenv(SO.HIST_ENV, raising=False)
            st = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-inplace",
                            _allow_cpu=True)
            ov = st.stage_ov
            try:
                outs[(mode, on)] = _static_run(model, st, ids, toks, n)
                if not on:
                    assert ov.hist is None and ov.hist_summary() is None and ov.describe()["hist"] is None
                    continue
                s = ov.hist_summary()
                hot_c, cold_c = st.routed_counts()
                assert s["layer_calls"] == n_layers * n and s["P"] == 6 * st.topk
                for l in range(n_layers):
                    assert sum(c * v for c, v in enumerate(s["hist"][l])) == cold_c[l], (l, s["hist"][l], cold_c)
                assert abs(s["hot_mean"] * s["layer_calls"] - sum(hot_c)) < 1e-6 * max(1, sum(hot_c)) + 0.5
                assert sum(cold_c) > 0 and s["cold_max"] >= 1
                assert ov.dump() == str(path) and json.load(open(path))["layer_calls"] == n_layers * n
            finally:
                if ov is not None and ov.hist is not None:
                    SO.hist_forget(ov)
                st.detach()
        a_, b_ = outs[(mode, True)], outs[(mode, False)]
        _assert_rows_equal(a_[0], b_[0], f"{mode}: hist on vs off")


def test_m3s2_bench_cpu_subcommands(tmp_path):
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import staging_bytes_bench as B

    out = tmp_path / "b.json"
    assert B.main(["all", "--out", str(out)]) == 0
    d = json.load(open(out))
    cen = d["census"]["counted"]
    for key, v in cen.items():                                         # rows the contracts read per layer-call
        c, h = v["cold"], v["hot"]
        assert v["step"]["m1_stage_rows"].get("hot.scale", 0) == 3 * h, key
        assert v["pool"]["m1_stage_rows"].get("hot.scale", 0) == 0, key
        for st_ in ("step", "pool"):
            for which in ("m1_stage_rows", "s1_stage_proj_x3"):
                r = v[st_][which]
                assert r.get("cold.weight", 0) == 3 * c and r.get("cold.scale", 0) == 3 * c and "hot.weight" not in r
    cc = d["census"]["contract_check"]
    assert cc["step_equal_pool"] and cc["step-h3_equal_pool"] and cc["listed_h3"] == cc["n_cold"] + 3, cc
    assert d["pool_fit"]["pool_sf"]["total_GiB"] == 13.142
    arm = [r for r in d["pool_fit"]["arms"] if r["pool_spare"] == 24 and r["headroom_GiB"] == 2.0
           and r["spec"] == "auto"][0]
    assert arm["mode"] == "partial" and arm["layers_pooled"] == 7
    e = d["exposure"]
    assert 15 < e["linear_fit"]["us_per_cold_expert_per_layer"] < 30 and e["linear_fit"]["zero_cold_ms"] > 1.0
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
