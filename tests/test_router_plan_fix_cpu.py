"""Mac CPU tests of the M2-T15 fix (docs/HANDOFF-M2-T15-FIX.md): the row-parallel router kernel `m2_router_rows`
(+ the ticketed plan in its last row program), plan-gemv's gate GEMV `m2_gate_fp32` / gate_dot, and the softplus
forms (SOFTPLUS modes 2 / 3). `.venv/bin/python -m pytest -q tests/test_router_plan_fix_cpu.py`

  1. m2_router_rows (grid R) == round 1's single-CTA m2_router_plan bit for bit (ids, w, sc, bsc, the topk order) and
     == the host mirror, for R in {1, 2, 6, 8}, every tie / edge case, every softplus mode;
  2. the ticketed plan: == moe_rows.dedup_fused (the separate m1_dedup_warp launch) on every buffer for every call
     shape; the plan body runs exactly once, in the last row program; the ticket is back to 0 (twice in a row);
  3. gate_dot == fused_glue_r2.fixed_dot_torch (the T9b-5 order at KB lanes) for every launch config (RB < R, RB >
     R, ragged N, U not dividing NK), bf16 and fp32 weights; the default config table / env override;
  4. the softplus forms: the kernel bodies == their mirrors; modes 2 / 3 differ from mode 1 only where exp(x) <
     2^-126, and there they are the fp64 exp rounded once / via 24 bits; the self-check picks the first exact form
     and falls back to F.softplus (plan-gemv included -- round 1 turned plan-gemv off instead);
  5. the switch: $DSV41_ROUTER_PLAN_LAUNCH, launches_per_layer.
Not checked here: compilation, the fences / atomics' device semantics, libdevice vs torch on CUDA, speed (the box:
tests/gpu/test_router_plan_gpu.py, bench/router_plan_bench.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import triton_shim  # noqa: E402
from runtime import fused_glue_r2 as R2  # noqa: E402
from runtime import moe_rows as MR  # noqa: E402
from runtime import router_plan as RP  # noqa: E402
from test_router_plan_cpu import ORDERS, _eq, _od, _rows, _ws  # noqa: E402


@pytest.fixture(scope="module")
def SK():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"),
                                             "static_kernels_m2t15fix_under_shim")
    km = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "moe_rows.py"),
                                            "moe_rows_m2t15fix_under_shim")
    km.KERNELS_ON_CPU = True
    mod._m2_bind_dedup(km.m1_dedup_warp)
    mod._KM = km
    return mod


def _launch_rows(SK, raw, bias, R_, E, k, ordd, mode, plan=None, ws=None):
    ws = ws or _ws(R_, E, k)
    ids = torch.full((R_, k), -9, dtype=torch.long)
    w = torch.full((R_, k), -9.0)
    SK.router_rows_launch(raw.clone(), bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                          eps=1e-20, softplus=mode, plan=plan)
    return ids, w, ws


def _launch_1cta(SK, raw, bias, R_, E, k, ordd, mode):
    ws = _ws(R_, E, k)
    ids = torch.full((R_, k), -9, dtype=torch.long)
    w = torch.full((R_, k), -9.0)
    SK.router_plan_launch(raw.clone(), bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                          eps=1e-20, softplus=mode)
    return ids, w, ws


def _nn(t):
    return torch.nan_to_num(t, nan=7.0)


# ================================================================================ 1. rows kernel == 1-CTA == mirror
@pytest.mark.parametrize("R_,E,k", [(1, 384, 6), (6, 384, 6), (2, 64, 6), (8, 40, 8), (6, 16, 2), (3, 7, 6)])
@pytest.mark.parametrize("case", ["normal", "ties", "mixed", "equal", "repeat", "edges", "nan"])
def test_rows_kernel_equals_single_cta_and_mirror(SK, R_, E, k, case):
    g = torch.Generator().manual_seed(hash(("fix", R_, E, k, case)) % 2 ** 31)
    raw, bias = _rows(g, R_, E, case)
    for t, mode in enumerate((1, 0, 2, 3)):
        src = F.softplus(raw) if mode == 0 else raw
        ordd = _od(ORDERS[(t + R_) % len(ORDERS)])
        n0 = len(SK.m2_router_rows.launches)
        ids, w, ws = _launch_rows(SK, src, bias, R_, E, k, ordd, mode)
        la = SK.m2_router_rows.launches[n0:]
        assert len(la) == 1 and la[0]["grid"] == (R_,) and la[0]["constexprs"]["PLAN"] is False
        i1, w1, ws1 = _launch_1cta(SK, src, bias, R_, E, k, ordd, mode)
        assert _eq(ids, i1) and _eq(_nn(w), _nn(w1)), (case, mode)
        for key in ("sc", "bsc", "tk"):
            assert _eq(_nn(ws[key]), _nn(ws1[key])), (key, case, mode)
        sp = (lambda x: RP.softplus_np(x, mode)) if mode else F.softplus
        ri, rw = RP.route_tail_torch(raw, bias, k, ordd, 1.5, softplus=sp, topk=RP.topk_cuda_torch)
        assert _eq(ids, ri) and torch.equal(torch.isnan(w), torch.isnan(rw)) and _eq(_nn(w), _nn(rw)), (case, mode)
        assert torch.equal(ws["tk"], RP.topk_cuda_torch(sp(raw).sqrt() + bias, k))


def test_rows_kernel_softplus_subnormal_rows(SK):
    """Scores whose exp underflows into the subnormal range (the round-1 probe's mismatch region): with every row's
    scores there, the top-k is decided by sqrt(subnormal softplus) + a zero bias -- the modes differ exactly as their
    softplus does, and each equals its mirror."""
    R_, E, k = 3, 48, 6
    g = torch.Generator().manual_seed(5)
    raw = -88.0 - torch.rand(R_, E, generator=g) * 15.0       # exp in [2^-150, 2^-127): subnormal / zero
    bias = torch.zeros(E)
    ordd = _od(ORDERS[1])
    res = {}
    for mode in (1, 2, 3):
        ids, w, ws = _launch_rows(SK, raw, bias, R_, E, k, ordd, mode)
        ri, rw = RP.route_tail_torch(raw, bias, k, ordd, 1.5, softplus=lambda x, m=mode: RP.softplus_np(x, m),
                                     topk=RP.topk_cuda_torch)
        assert _eq(ids, ri) and _eq(w, rw), mode
        res[mode] = ws["sc"].clone()
    assert (res[2] > 0).all() and (res[3] > 0).sum() > 0      # the fp64 forms keep the subnormal softplus


# ================================================================================ 2. the ticketed plan
def _plan_cases():
    return [("none", None), ("mm", None), ("v1", None),
            ("v2", ("contiguous", False)), ("v2", ("contiguous", True)), ("v2", ("psum", False)),
            ("v2", ("psum", True)), ("v2", ("masked", False)), ("v2", ("masked", True))]


@pytest.mark.parametrize("kind,lay", _plan_cases())
@pytest.mark.parametrize("case", ["normal", "ties", "mixed", "repeat"])
def test_ticket_plan_equals_dedup_fused(SK, kind, lay, case):
    from test_moe_rows_t10_cpu import _fresh, _gtab, _inpl

    KM = SK._KM
    g = torch.Generator().manual_seed(hash(("fixplan", kind, lay, case)) % 2 ** 31)
    for t, (R_, E, k) in enumerate(((6, 24, 6), (6, 40, 6), (4, 16, 3), (8, 64, 8), (2, 384, 6))):
        raw, bias = _rows(g, R_, E, case)
        P = R_ * k
        BR = 128 if lay is not None and lay[0] == "masked" else (R_ + t % 3)
        inpl0 = None
        if kind == "v1":
            gtab = _gtab(g, E, "random")
            inpl0 = {"gtab": gtab, "H": int((gtab >= 0).sum()), "S": 2, "dummy": torch.tensor(1), "has_dummy": True}
        elif kind == "v2":
            gtab = _gtab(g, E, ["random", "all_cold", "zero_cold", "random", "random"][t])
            inpl0 = _inpl(gtab, 1 + t % 3, lay[0], lay[1], BR, R_, k, G_extra=t % 2)
        ordd = _od(ORDERS[t % len(ORDERS)])
        outs = []
        for mine in (True, False):
            inpl = _fresh(inpl0) if kind == "v2" else inpl0
            mm = torch.full((P,), -7, dtype=torch.int32) if kind in ("mm", "v2", "v1") else None
            ids_s = torch.full((P,), -7, dtype=torch.long)
            uv = torch.full((P,), -7, dtype=torch.long)
            buf = MR.DedupBuffers(R_, k, "cpu")
            ws = _ws(R_, E, k)
            if mine:
                ids = torch.empty(R_, k, dtype=torch.long)
                w = torch.empty(R_, k)
                a = RP.dedup_args(buf, R_, k, BR, mm, ids_s, uv, inpl, ids)
                for rep in range(2):                     # the ticket is back to 0: the second launch works the same
                    if inpl is not None and inpl.get("ovf") is not None:
                        inpl["ovf"].zero_()              # (the overflow count accumulates by design)
                    SK.router_rows_launch(raw, bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                                          eps=1e-20, softplus=1, plan=a)
                    assert int(ws["ticket"]) == 0
                    assert SK.m2_router_rows.launches[-1]["grid"] == (R_,)
                m = RP.plan_out_dict(buf, mm, inpl, a)
            else:
                ids, w, _ = _launch_rows(SK, raw, bias, R_, E, k, ordd, 1)
                m = KM.dedup_fused(buf, ids, w, BR, mm=mm, ids_step=ids_s, uval=uv, inpl=inpl, impl="warp")
            outs.append((m, mm, ids_s, uv, inpl, ids, w))
        (m1, mm1, i1, u1, p1, id1, w1), (m0, mm0, i0, u0, p0, id0, w0) = outs
        assert sorted(m1) == sorted(m0)
        for key, v in m0.items():
            if isinstance(v, torch.Tensor):
                assert _eq(m1[key], v), (key, kind, lay, case, t)
            else:
                assert m1[key] == v, key
        assert _eq(id1, id0) and _eq(w1, w0)
        for a_, b_ in ((mm1, mm0), (i1, i0), (u1, u0)):
            assert a_ is None and b_ is None or _eq(a_, b_)
        if kind == "v2":
            for n in ("mi", "gl", "ovf"):
                assert p1[n] is None and p0[n] is None or _eq(p1[n], p0[n]), n


def test_ticket_plan_runs_once_in_the_last_row_program(SK, monkeypatch):
    seen = []
    real = SK.m2_dedup_body

    def spy(*a, **kw):
        seen.append(tuple(triton_shim._PID))
        return real(*a, **kw)
    monkeypatch.setattr(SK, "m2_dedup_body", spy)
    R_, E, k = 6, 32, 6
    raw, bias = _rows(torch.Generator().manual_seed(2), R_, E, "normal")
    buf = MR.DedupBuffers(R_, k, "cpu")
    ids = torch.empty(R_, k, dtype=torch.long)
    a = RP.dedup_args(buf, R_, k, R_, None, None, None, None, ids)
    ws = _ws(R_, E, k)
    SK.router_rows_launch(raw, bias, ws, ids, torch.empty(R_, k), R=R_, E=E, K=k, order=R2.order_args(_od(ORDERS[0])),
                          rs=1.0, eps=1e-20, softplus=1, plan=a)
    assert seen == [(R_ - 1, 0, 0)] and int(ws["ticket"]) == 0
    seen.clear()
    _launch_rows(SK, raw, bias, R_, E, k, _od(ORDERS[0]), 1)       # no plan: no ticket, no dedup
    assert seen == []


def test_rows_launch_rejects_out_of_range(SK):
    ws = _ws(1, 8, 2)
    with pytest.raises(ValueError):
        SK.router_rows_launch(torch.zeros(9, 8), torch.zeros(8), ws, torch.empty(9, 2, dtype=torch.long),
                              torch.empty(9, 2), R=9, E=8, K=2, order=R2.order_args(_od(ORDERS[0])), rs=1.0,
                              eps=1e-20, softplus=1)
    with pytest.raises(ValueError):
        SK.router_rows_launch(torch.zeros(1, 8), torch.zeros(8), ws, torch.empty(1, 2, dtype=torch.long),
                              torch.empty(1, 2), R=1, E=8, K=2, order=R2.order_args(_od(ORDERS[0])), rs=1.0,
                              eps=1e-20, softplus=4)


# ================================================================================ 3. gate_dot
@pytest.mark.parametrize("R_,N,K_,kb,cfg", [
    (1, 40, 1024, 512, (1, 2, 10, 4, 1)), (6, 40, 1024, 512, (1, 2, 10, 4, 1)), (6, 40, 1024, 512, (2, 4, 1, 4, 1)),
    (6, 37, 1024, 512, (8, 4, 2, 4, 1)), (5, 24, 256, 64, (4, 3, 3, 2, 1)), (6, 16, 512, 256, (1, 1, 1, 1, 1)),
    (8, 24, 512, 128, (8, 8, 4, 8, 1)), (3, 9, 128, 128, None)])
@pytest.mark.parametrize("wdt", [torch.bfloat16, torch.float32])
def test_gate_dot_equals_fixed_order_oracle(SK, R_, N, K_, kb, cfg, wdt):
    g = torch.Generator().manual_seed(R_ * 1000 + N + K_)
    x = (torch.randn(R_, K_, generator=g) * 0.5).to(torch.bfloat16).float()
    w = (torch.randn(N, K_, generator=g) * 0.1).to(wdt)
    out = torch.full((R_, N), -3.0)
    n0 = len(SK.m2_gate_fp32.launches)
    SK.gate_dot(x, w, out, kb=kb, cfg=cfg)
    rb, bn = (cfg or SK.gate_config(R_))[:2]
    assert SK.m2_gate_fp32.launches[n0]["grid"] == (-(-N // bn), -(-R_ // rb))
    assert SK.m2_gate_fp32.launches[n0]["enable_fp_fusion"] is False
    assert _eq(out, R2.fixed_dot_torch(x, w, kb))


def test_gate_config_table_and_env(SK, monkeypatch):
    monkeypatch.delenv(SK.M2_GATE_CFG_ENV, raising=False)
    for R_ in (1, 2, 3, 6, 8):
        assert SK.gate_config(R_) == SK.M2_GATE_CFG[min(8, 1 << (R_ - 1).bit_length())]
    monkeypatch.setenv(SK.M2_GATE_CFG_ENV, "8=2,4,5,8,1; 1=1,1,10,2,1")
    assert SK.gate_config(6) == (2, 4, 5, 8, 1) and SK.gate_config(1) == (1, 1, 10, 2, 1)
    with pytest.raises(ValueError):
        SK.gate_dot(torch.zeros(2, 96), torch.zeros(4, 96), torch.zeros(2, 4), kb=64)       # 64 does not divide 96
    assert RP.gate_kb(5120) == 512 and RP.gate_kb(256) == 256 and RP.gate_kb(96) == 32


# ================================================================================ 4. softplus forms
def test_softplus_probe_forms_equal_mirrors(SK):
    x = RP.softplus_probe_inputs(torch.Generator().manual_seed(0), 1 << 12)
    x = torch.cat([x, torch.linspace(-110.0, -80.0, 4001)])
    outs = {}
    for mode in (1, 2, 3):
        got = SK.softplus_probe(x, mode)
        assert _eq(_nn(got), _nn(RP.softplus_np(x, mode))), mode
        assert got[x > 20].equal(x[x > 20])
        outs[mode] = got
    with np.errstate(over="ignore", under="ignore"):
        sub = torch.from_numpy(np.exp(x.numpy()).astype(np.float32) < RP.FLT_MIN) & ~torch.isnan(x)
    for mode in (2, 3):
        same = _eq(_nn(outs[mode][~sub]), _nn(outs[1][~sub]))
        assert same, mode                                   # modes 2 / 3 change only the subnormal-exp region
    ed = np.exp(x[sub].numpy().astype(np.float64))
    assert _eq(outs[2][sub], torch.from_numpy(ed.astype(np.float32)))
    e24 = (ed * 2.0 ** 64).astype(np.float32).astype(np.float64) * 2.0 ** -64
    assert _eq(outs[3][sub], torch.from_numpy(e24.astype(np.float32)))
    with pytest.raises(ValueError):
        SK.softplus_probe(x, 0)


def test_softplus_fp64_form_runs_only_when_a_lane_needs_it(SK, monkeypatch):
    """Modes 2 / 3: the fp64 form (GB300's slow fp64 path) runs in a row program only if some score of the row has
    exp < 2^-126; the result is the same either way (the branch only skips work)."""
    calls = []
    real = SK._m2_softplus_sub.fn

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(SK._m2_softplus_sub, "fn", spy)
    R_, E, k = 3, 40, 6
    raw, bias = _rows(torch.Generator().manual_seed(8), R_, E, "normal")
    for mode in (2, 3):
        n0 = len(calls)
        ids, w, _ = _launch_rows(SK, raw, bias, R_, E, k, _od(ORDERS[0]), mode)
        assert len(calls) == n0                               # |raw| ~ 2: no row needs it
        ri, rw = RP.route_tail_torch(raw, bias, k, _od(ORDERS[0]), 1.5,
                                     softplus=lambda x, m=mode: RP.softplus_np(x, m), topk=RP.topk_cuda_torch)
        assert _eq(ids, ri) and _eq(w, rw)
        raw2 = raw.clone()
        raw2[1, 5] = -95.0                                    # one lane of row 1
        _launch_rows(SK, raw2, bias, R_, E, k, _od(ORDERS[0]), mode)
        assert len(calls) == n0 + 1                           # row 1's program only


class _Glue:
    def __init__(self, rows=6, D=64):
        class _G:
            weight = torch.zeros(8, D)
            topk = 2
            gate_temp = 1.0
        self.o = type("O", (), {"moes": [type("SM", (), {"moe": type("M", (), {"gate": _G()})(), "layer": 0})()]})()
        self.dev = torch.device("cpu")
        self.rows = rows
        self.logs = []

    def on(self, c):
        return True

    def log(self, s):
        self.logs.append(s)


@pytest.mark.parametrize("impl", ["plan", "plan-gemv"])
def test_self_check_softplus_form_choice(SK, monkeypatch, unhook_rows, impl):
    """The probe's choice: the first exact form; mode 1 inexact but 2 exact -> "fused-sub"; none exact -> "torch"
    (plan-gemv stays on, unlike round 1)."""
    monkeypatch.setattr(RP, "SK", SK)
    monkeypatch.setattr(RP, "KERNELS_ON_CPU", True)
    real = SK.softplus_probe
    for broken, want in (((), "fused"), ((1,), "fused-sub"), ((1, 2), "fused-sub24"), ((1, 2, 3), "torch")):
        monkeypatch.setattr(SK, "softplus_probe", lambda x, mode=1, b=broken: real(x, mode) * (
            1.5 if mode in b else 1.0))
        rp = RP.RouterPlan(_Glue(rows=1), impl)
        rp._check_softplus(torch.Generator().manual_seed(0))
        assert rp.softplus == want and rp.on(), (broken, rp.checks)
        assert rp.checks["softplus"]["form"] == want and len(rp.checks["softplus"]["forms"]) == min(len(broken) + 1, 3)
        lp = rp.launches_per_layer()
        if impl == "plan":
            assert lp == ("mm + 2" if want == "torch" else "mm + 1"), lp
        else:
            assert lp == ("3" if want == "torch" else "2"), lp


@pytest.fixture
def unhook_rows():
    yield
    RP.uninstall_rows_hook()


# ================================================================================ 5. the switch
def test_plan_launch_env(monkeypatch):
    monkeypatch.delenv(RP.PLAN_LAUNCH_ENV, raising=False)
    assert RP.plan_launch() == "ticket"
    monkeypatch.setenv(RP.PLAN_LAUNCH_ENV, "split")
    assert RP.plan_launch() == "split"
    monkeypatch.setenv(RP.PLAN_LAUNCH_ENV, "two")
    with pytest.raises(ValueError):
        RP.plan_launch()


def test_launches_per_layer(monkeypatch, unhook_rows):
    monkeypatch.delenv(RP.PLAN_LAUNCH_ENV, raising=False)
    rp = RP.RouterPlan(_Glue(rows=6), "plan")
    assert rp.launches_per_layer() == "mm + 1" and rp.launches_per_layer(1) == "mm + 1"
    rp.launch_form = "split"
    assert rp.launches_per_layer() == "mm + 2" and rp.launches_per_layer(1) == "mm + 1"
    rp.softplus = "torch"
    assert rp.launches_per_layer() == "mm + 3"
    rg = RP.RouterPlan(_Glue(rows=6), "plan-gemv")
    assert rg.launches_per_layer() == "2" and rg.describe()["gate"].endswith("(A3 vs cuBLAS)")
    assert "KB = 64" in rg.describe()["gate"]
