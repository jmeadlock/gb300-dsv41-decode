"""Mac CPU tests of M2-T15 at 6 rows (the glue + the moe_rows._route hook; the kernel itself:
tests/test_router_plan_cpu.py). `.venv/bin/python -m pytest -q tests/test_router_plan_rows_cpu.py`

  * $DSV41_ROUTER_IMPL=plan on CPU: the hook is installed, the plan (unverified on CPU) answers None and the step is
    the fused path's, bit for bit;
  * with the shim kernels: the build-time self-check passes (softplus probe, routing on random + tie-heavy scores,
    plan tensors == dedup_fused on the layer's own call shape), then every layer's MoE takes the plan kernel's mapping
    through the hook -- one plan launch per layer per step, no dedup_fused / router_post launch -- and the step ==
    the fused step (logits, hidden, caches) for dg-inplace, dg-masked and ref MoE;
  * a broken plan kernel fails the self-check and the impl goes off with the reason.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import triton_shim  # noqa: E402
from runtime import fused_glue_r2 as R2  # noqa: E402
from runtime import moe_rows as MR  # noqa: E402
from runtime import router_plan as RP  # noqa: E402
from test_router_plan_cpu import ORDERS, _od  # noqa: E402


@pytest.fixture(scope="module")
def SK():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"),
                                             "static_kernels_m2t15_rows_under_shim")
    km = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "moe_rows.py"),
                                            "moe_rows_m2t15_rows_under_shim")
    km.KERNELS_ON_CPU = True
    mod._m2_bind_dedup(km.m1_dedup_warp)
    return mod


@pytest.fixture
def unhook():
    yield
    RP.uninstall_rows_hook()


from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens, emu,  # noqa: E402,F401
                                  rows_dg, rows_glue)


def _six_row_step(model, pool, impl, monkeypatch, moe_impl="dg-inplace"):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(RP.ROUTER_IMPL_ENV, impl)
    hid = (len(model.layers) - 1,)
    return StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl=moe_impl, fp8_impl="ref",
                      glue_impl="fused", _allow_cpu=True)


def test_six_row_plan_on_cpu_is_the_fused_path(rows_glue, monkeypatch, unhook):
    model, pool, _, _ = rows_glue("mini4")
    ids, toks = _tf_tokens(18, 12, 31)
    st = _six_row_step(model, pool, "plan", monkeypatch)
    assert RP._HOOK["orig"] is not None and st.glue.rplan.on()
    got = _static_run(model, st, ids, toks, 2)
    st.detach()
    ref = _six_row_step(model, pool, "fused", monkeypatch)
    want = _static_run(model, ref, ids, toks, 2)
    ref.detach()
    _assert_rows_equal(got[0], want[0], "6-row plan (CPU = fused path) vs fused")
    assert all(torch.equal(x, y) for x, y in zip(got[1], want[1]))


@pytest.mark.parametrize("moe_impl", ["dg-inplace", "dg-masked", "ref"])
def test_six_row_plan_kernel_through_the_hook_equals_fused(rows_glue, monkeypatch, unhook, SK, moe_impl):
    """The shim kernels inside the 6-row step: the build-time self-check passes (softplus probe, routing, plan
    tensors vs dedup_fused), then every layer's MoE gets the plan kernel's mapping through the _route hook -- one plan
    launch per layer per step, no dedup_fused / router_post / topk launch -- and the step == the fused step."""
    model, pool, _, _ = rows_glue("mini4")
    ids, toks = _tf_tokens(18, 12, 31)
    order = _od(ORDERS[1])
    monkeypatch.setattr(RP, "SK", SK)
    st = _six_row_step(model, pool, "plan", monkeypatch, moe_impl)
    rp = st.glue.rplan
    st.glue.orders["router"] = order                     # CPU's NATIVE sum -> an explicit order for both paths
    monkeypatch.setattr(RP, "KERNELS_ON_CPU", True)
    rp.calibrate(torch.Generator().manual_seed(0))
    assert rp.verified and rp.on() and rp.rows_plan == "on", rp.describe()
    assert rp.checks["softplus"]["mismatch"] == 0 and rp.checks["route"].startswith("equal")
    assert rp.checks["plan_rows"].startswith("equal")
    rp.softplus = "torch"                                # the probe-mismatch form: the step's F.softplus bits
    calls = {"plan": 0, "dedup": 0, "post": 0}
    real_l = SK.router_plan_launch
    monkeypatch.setattr(SK, "router_plan_launch", lambda *a, **k: (calls.__setitem__(
        "plan", calls["plan"] + 1), real_l(*a, **k))[1])
    real_d = MR.dedup_fused
    monkeypatch.setattr(MR, "dedup_fused", lambda *a, **k: (calls.__setitem__("dedup", calls["dedup"] + 1),
                                                            real_d(*a, **k))[1])
    real_p = R2.launch_router_post
    monkeypatch.setattr(R2, "launch_router_post", lambda *a, **k: (calls.__setitem__("post", calls["post"] + 1),
                                                                   real_p(*a, **k))[1])
    got = _static_run(model, st, ids, toks, 2)
    L = len(model.layers)
    assert calls == {"plan": 2 * L, "dedup": 0, "post": 0}, calls
    st.detach()
    monkeypatch.setattr(RP, "KERNELS_ON_CPU", False)
    ref = _six_row_step(model, pool, "fused", monkeypatch, moe_impl)
    ref.glue.orders["router"] = order
    want = _static_run(model, ref, ids, toks, 2)
    ref.detach()
    _assert_rows_equal(got[0], want[0], f"6-row plan kernel (shim) vs fused ({moe_impl})")
    assert all(torch.equal(x, y) for x, y in zip(got[1], want[1]))
    for k_ in want[2]:
        assert torch.equal(got[2][k_], want[2][k_]), k_


def test_self_check_turns_a_broken_plan_off(rows_glue, monkeypatch, unhook, SK):
    model, pool, _, _ = rows_glue("mini4")
    monkeypatch.setattr(RP, "SK", SK)
    st = _six_row_step(model, pool, "plan", monkeypatch)
    st.glue.orders["router"] = _od(ORDERS[0])
    monkeypatch.setattr(RP, "KERNELS_ON_CPU", True)
    real = SK.router_plan_launch

    def broken(raw, bias, ws, ids, w, **k):
        real(raw, bias, ws, ids, w, **k)
        w.mul_(1.0000001)
    monkeypatch.setattr(SK, "router_plan_launch", broken)
    rp = st.glue.rplan
    rp.calibrate(torch.Generator().manual_seed(0))
    assert not rp.on() and rp.status.startswith("off: route differs") and not rp.verified
    st.detach()


