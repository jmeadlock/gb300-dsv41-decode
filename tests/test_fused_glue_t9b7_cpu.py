"""Mac CPU tests of T9b-7 (docs/HANDOFF-T9B7-GLUE-ORDER.md): one reduction order per SHAPE for every fused-glue
cluster, whatever the step's row count R. `.venv/bin/python -m pytest -q tests/test_fused_glue_t9b7_cpu.py`

  1. the policy plumbing: $DSV41_GLUE_ORDER (shape default / rows), cal_rows, `per_row_reductions` (the R-row
     oracle: torch's sum / mean of each [1, 1, ...] token row, restored on exit, nests), the source lint that every
     host draw on a CPU generator names device="cpu" (the 6-row dg_sf_pack self-check crashed on the box without it);
  2. a FusedGlue build at 6 rows picks its orders at the 1-row reference shapes (rows 1; Engram hc_mult) under
     `shape` -- the same picks as the 1-row build -- and at the R-row shapes under `rows` (the pre-T9b-7 A/B), with a
     pick_order that behaves like torch's CUDA reduce (its configuration depends on the number of outputs);
  3. per cluster, on the shim (tests/triton_shim.py: the real @triton.jit bodies, program by program): the R-row
     launch with an explicit order == the 1-row launch on each row alone, bit for bit -- hc_norm (one-program and
     T9b-4 split kernels, with the mix statistic), norm_row (q path; kv path with RoPE, QDQ and the ring), router
     pre/post, Engram gate/out, topk post, and round 1's hc_post, o_rope_qdq, SwiGLU (rows and pair-row forms),
     act_quant into the DeepGEMM rows (pair scatter vs broadcast), the MoE combine (posmat vs plain); the torch
     mirrors likewise;
  4. runtime/fused_glue_r2.glue_row_census on the CPU emulation (dg-inplace, fused glue, 6 rows vs 1 row): every
     cluster on in both builds reads row-invariant; and tests/gpu/test_static_rows_gpu.py's T9b-7 test dry-run.
Not checked here: Triton compilation, torch's CUDA reduction order (the box: tests/gpu/test_static_rows_gpu.py
test_t9b7_glue_clusters_row_invariant, tests/gpu/test_fused_glue_gpu.py test_t9b7_real_width_orders).
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn.functional as F

import triton_shim
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from test_static_emulated_cpu import act_quant as _emu_act_quant
from test_static_rows_cpu import (_cpu_default_device, _prefill, _tf_tokens, emu, rows_dg,  # noqa: F401 (fixtures)
                                  rows_glue)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
R6 = 6


@pytest.fixture(autouse=True)
def _restore_dtype_and_cache():
    prev = torch.get_default_dtype()
    R2._ORDER_CACHE.clear()
    yield
    torch.set_default_dtype(prev)
    R2._ORDER_CACHE.clear()


def _ref_aq(x, block_size=32, scale_fmt="ue8m0", scale_dtype=E8, inplace=False):
    r = _emu_act_quant(x, block_size, scale_fmt, scale_dtype, inplace)
    if inplace or scale_dtype != E8:
        return r
    q, s = r
    return q, s.to(E8)


class _R:
    fp8_block_size, scale_fmt, scale_dtype = 32, "ue8m0", E8
    act_quant = staticmethod(_ref_aq)


@pytest.fixture(scope="module")
def tab():
    return FG.scale_code_table(_R, "cpu")


@pytest.fixture(scope="module")
def silu():
    return FG.silu_table("cpu", 10.0)


@pytest.fixture(scope="module")
def KR():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue_r2.py"), "fused_glue_r2_t9b7")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KF():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue.py"), "fused_glue_t9b7")
    mod.KERNELS_ON_CPU = True
    return mod


def _bits(t):
    t = t.contiguous()
    n = t.element_size()
    return t.view(torch.uint8) if n == 1 else t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[n])


def _eq(a, b):
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _od(order, fin="mul"):
    return {"order": order, "fin": fin, "exact": True, "name": R2.order_name(order, fin)}


NATIVE = {"order": R2.NATIVE, "fin": "mul", "exact": True, "name": R2.NATIVE}


def _act(g, rows, K, lo=-6, hi=7):
    return FG._rand_act(g, rows, K, lo, hi)


def _one_row_order(w, rows=1):
    """The order a 1-row build picks for width w (torch's predicted configuration at the 1-row shape)."""
    return _od(R2.predicted_order(w, rows))


# ================================================================================ 1. policy plumbing
def test_glue_order_env(monkeypatch):
    monkeypatch.delenv(R2.GLUE_ORDER_ENV, raising=False)
    assert R2.glue_order() == R2.GLUE_ORDER_DEFAULT == "shape"
    monkeypatch.setenv(R2.GLUE_ORDER_ENV, "rows")
    assert R2.glue_order() == "rows"
    monkeypatch.setenv(R2.GLUE_ORDER_ENV, "bogus")
    with pytest.raises(ValueError):
        R2.glue_order()
    assert R2.cal_rows("shape", 6) == R2.cal_rows("shape", 1) == 1 and R2.cal_rows("rows", 6) == 6


def test_per_row_reductions_splits_token_rows_and_restores(monkeypatch):
    """Inside the context a float sum / mean of [1, R > 1, ...] over dims >= 2 runs on each fresh [1, 1, ...] row slice
    (the 1-row call's shape); 1-row tensors, reductions over dim 0/1 and int tensors pass through; two_d also splits
    [n > 1, K] over dim 1; the methods are restored afterwards (and nesting restores the outer wrapper)."""
    seen = []
    real_mean, real_sum = torch.Tensor.mean, torch.sum

    def spy_mean(x, *a, **k):
        seen.append(("mean", tuple(x.shape)))
        return real_mean(x, *a, **k)

    def spy_sum(x, *a, **k):
        seen.append(("sum", tuple(x.shape)))
        return real_sum(x, *a, **k)

    monkeypatch.setattr(torch.Tensor, "mean", spy_mean)
    monkeypatch.setattr(torch, "sum", spy_sum)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, R6, 4, 64, generator=g)
    got = {}

    def run(name, fn, two_d=False):
        # the shapes the underlying op saw (a set: the default-device TorchFunctionMode re-enters the patched method)
        seen.clear()
        with R2.per_row_reductions(two_d=two_d):
            got[name] = fn()
        return set(seen), len(seen)

    shapes, n = run("y", lambda: x.square().mean(-1))
    assert shapes == {("mean", (1, 1, 4, 64))} and n >= R6
    assert run("z", lambda: torch.sum(x, dim=2))[0] == {("sum", (1, 1, 4, 64))}
    assert run("one", lambda: x[:, :1].mean(-1))[0] == {("mean", (1, 1, 4, 64))}
    assert run("d1", lambda: x.mean(1))[0] == {("mean", (1, R6, 4, 64))}                # reduces over the token dim
    assert run("two", lambda: torch.randn(R6, 6, generator=g).mean(-1))[0] == {("mean", (R6, 6))}
    assert run("ints", lambda: torch.sum(torch.ones(1, R6, 3, dtype=torch.long), dim=2))[0] == {("sum", (1, R6, 3))}
    assert run("two_split", lambda: torch.randn(R6, 6, generator=g).mean(-1), two_d=True)[0] == {("mean", (1, 6))}
    with R2.per_row_reductions():
        with R2.per_row_reductions(two_d=True):
            pass
        assert torch.Tensor.mean is not spy_mean          # still wrapped after the inner context
    assert torch.Tensor.mean is spy_mean and torch.sum is spy_sum
    assert torch.equal(got["y"], x.square().mean(-1)) and torch.equal(got["z"], real_sum(x, dim=2))   # CPU torch
    assert got["one"].shape == (1, 1, 4) and got["d1"].shape == (1, 4, 64) and got["two"].shape == (R6,)
    assert got["two_split"].shape == (R6,) and got["ints"].dtype == torch.long
    with R2.per_row_reductions(enabled=False):
        assert torch.Tensor.mean is spy_mean


def test_per_row_reductions_leave_no_attribute_behind():
    assert "sum" not in vars(torch.Tensor) and "mean" not in vars(torch.Tensor)
    with R2.per_row_reductions():
        assert "sum" in vars(torch.Tensor)
    assert "sum" not in vars(torch.Tensor) and "mean" not in vars(torch.Tensor)


_DRAWS = {"randperm", "randint", "rand", "randn", "rand_like", "randn_like", "multinomial", "normal"}


@pytest.mark.parametrize("path", ["runtime/fused_glue_r2.py", "runtime/fused_glue.py", "runtime/static_kernels.py",
                                  "bench/glue_census.py"])
def test_every_generator_draw_names_its_device(path):
    """The runtime's default device is cuda: a torch draw with a CPU `generator=` and no `device=` raises on the box
    ("Expected a 'cuda' device type for generator but found 'cpu'" -- dg_sf_pack's 6-row self-check, T9b-7). Every
    such call in the glue's files names its device."""
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _DRAWS:
            kws = {k.arg for k in node.keywords}
            if "generator" in kws and "device" not in kws:
                bad.append(f"{path}:{node.lineno} {node.func.attr}")
    assert not bad, bad


# ================================================================================ 2. the build's picks
def _fake_torch_pick(calls):
    """A pick_order that behaves like torch's CUDA reduce: the configuration it reports depends on the number of
    outputs (rows), as predicted_order; every call recorded. CPU mirrors with these explicit orders differ from CPU
    torch, so the self-checks may turn clusters off -- the picks are what is under test."""
    def pick(dev, w, rows=1, kind="sqmean", **k):
        calls.append((int(w), int(rows), kind))
        o = R2.predicted_order(w, rows)
        return {"order": o, "fin": "mul", "exact": True, "match": 1.0, "max_ulp": 0, "rows": rows, "w": w,
                "kind": kind, "n": 1024, "tried": 1, "name": R2.order_name(o, "mul")}
    return pick


def _build(model, pool, rows, **kw):
    from runtime.static_step import StaticStep

    return StaticStep(model, pool, max_pos=255, rows=rows, moe_impl="dg-inplace", glue_impl="fused",
                      _allow_cpu=True, **kw)


@pytest.mark.parametrize("policy", ["shape", "rows"])
def test_six_row_build_picks_the_one_row_shapes_orders(rows_glue, monkeypatch, policy):
    model, pool, _, _ = rows_glue("mini4")
    monkeypatch.setenv(R2.GLUE_ORDER_ENV, policy)
    calls6, calls1 = [], []
    monkeypatch.setattr(R2, "pick_order", _fake_torch_pick(calls6))
    s6 = _build(model, pool, R6, spare_slots=8)
    monkeypatch.setattr(R2, "pick_order", _fake_torch_pick(calls1))
    s1 = _build(model, pool, 1)
    hc = model.layers[0].hc_mult
    eng = next(l.engram for l in model.layers if l.engram is not None)
    assert calls1 and all(r in (1, hc) for _, r, _ in calls1)
    assert (eng.dim, hc, "sqmean") in calls1 and (eng.dim, hc, "sum") in calls1     # Engram: the [1, 1, hc, D] call
    o1 = {k: v["name"] for k, v in s1.glue.orders.items()}
    o6 = {k: v["name"] for k, v in s6.glue.orders.items()}
    d6 = s6.glue.describe()
    assert d6["r2"]["order_policy"]["policy"] == policy and d6["r2"]["order_policy"]["step_rows"] == R6
    assert d6["calibration"]["r2"]["order_policy"]["cal_rows"] == (1 if policy == "shape" else R6)
    if policy == "shape":
        assert sorted(calls6) == sorted(calls1) and o6 == o1
        assert "orders of the 1-row shapes" in s6.glue.glue_line()
    else:   # the pre-T9b-7 protocol: the R-row reference's shapes; mini4's hc*D = 1024 and Engram differ
        assert all(r in (R6, R6 * hc) for _, r, _ in calls6)
        diff = sorted(k for k in o1 if o1[k] != o6.get(k))
        assert "mix" in diff and "engram" in diff, (o1, o6)
    s6.detach()
    s1.detach()


def test_hc_post_mode_is_picked_on_one_row_draws(rows_glue, monkeypatch):
    """Shape policy: the hc_post mode is picked on 1-row draws (what the 1-row build does) and the 6-row launch is
    then checked against the reference row by row (cal["hc_post"]["rows_check"])."""
    model, pool, _, _ = rows_glue("mini4")
    monkeypatch.delenv(R2.GLUE_ORDER_ENV, raising=False)
    s6 = _build(model, pool, R6, spare_slots=8)
    cal = s6.glue.describe()["calibration"]["hc_post"]
    assert cal["picked"] is not None and cal["rows_check"] == {"rows": R6, "mismatches": 0}
    assert s6.glue.status["hc_post"] == "on"
    s6.detach()


# ================================================================================ 3. per cluster: R rows == 1 row
def _rows_equal(full, parts):
    return all(_eq(full[r], p) for r, p in enumerate(parts))


@pytest.mark.parametrize("mod_name", ["mirror", "kernel"])
@pytest.mark.parametrize("split", [False, True])
def test_hc_norm_rows_equal_one_row(KR, tab, mod_name, split):
    """hc_norm (+ the mix statistic, + y.float(), + act_quant): R rows in one launch == each row alone, in the 1-row
    build's orders (norm over D, mix over hc*D), both kernels (t9b2r2 one program / t9b4 split)."""
    mod = KR if mod_name == "kernel" else R2
    HC, D = 4, 256
    g = torch.Generator().manual_seed(11 + split)
    x = _act(g, R6 * HC, D).view(R6, HC, D)
    pre = (torch.rand(R6, HC, generator=g) + 1e-3).float()
    nw = (torch.randn(D, generator=g) * 0.5 + 1.0).to(BF)
    nord, mord = _one_row_order(D), _one_row_order(HC * D)

    def bufs(n):
        return dict(y=torch.empty(n, D, dtype=BF), yf=torch.empty(n, D), xf=torch.empty(n, HC * D),
                    mrs=torch.empty(n), q=torch.empty(n, D, dtype=FP8), s=torch.empty(n, D // 32, dtype=E8))
    full = bufs(R6)
    mod.launch_hc_norm(x, pre, nw, full["y"], eps=1e-6, norm_ord=nord, hc_mode=1, code_tab=tab, yf=full["yf"],
                       xf=full["xf"], mrs=full["mrs"], mix_ord=mord, mix_eps=1e-6, q=full["q"], s=full["s"],
                       split=split)
    for r in range(R6):
        o = bufs(1)
        mod.launch_hc_norm(x[r], pre[r].contiguous(), nw, o["y"].view(-1), eps=1e-6, norm_ord=nord, hc_mode=1,
                           code_tab=tab, yf=o["yf"].view(-1), xf=o["xf"].view(-1), mrs=o["mrs"], mix_ord=mord,
                           mix_eps=1e-6, q=o["q"].view(-1), s=o["s"].view(-1), split=split)
        for k in full:
            assert _eq(full[k][r:r + 1], o[k].view(1, -1) if k != "mrs" else o[k]), (k, r)


@pytest.mark.parametrize("mod_name", ["mirror", "kernel"])
def test_norm_row_q_and_kv_rows_equal_one_row(KR, tab, mod_name):
    """q path (q_norm + act_quant) and kv path (kv_norm + RoPE + QDQ + the ring write at R slots, each row at its own
    position): R rows == each row alone, 1-row orders."""
    mod = KR if mod_name == "kernel" else R2
    g = torch.Generator().manual_seed(12)
    N, RD, win = 128, 32, 8
    x = _act(g, R6, N, -4, 6)
    nw = (torch.randn(N, generator=g) * 0.3 + 1.0).to(BF)
    ordd = _one_row_order(N)
    qr, q, s = torch.empty(R6, N, dtype=BF), torch.empty(R6, N, dtype=FP8), torch.empty(R6, N // 32, dtype=E8)
    mod.launch_norm_row(x, nw, qr, eps=1e-6, ordd=ordd, code_tab=tab, q=q, s=s)
    for r in range(R6):
        qr1, q1, s1 = torch.empty(N, dtype=BF), torch.empty(N, dtype=FP8), torch.empty(N // 32, dtype=E8)
        mod.launch_norm_row(x[r], nw, qr1, eps=1e-6, ordd=ordd, code_tab=tab, q=q1, s=s1)
        assert _eq(qr[r], qr1) and _eq(q[r], q1) and _eq(s[r], s1), r
    pos = torch.arange(40, 40 + R6)
    inv = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, dtype=torch.float32) / RD))
    fr = torch.view_as_real(torch.polar(torch.ones(R6, RD // 2), pos.view(-1, 1).float() * inv)).contiguous()
    slot = torch.remainder(pos, win)
    ring = torch.zeros(win, N, dtype=BF)
    mod.launch_norm_row(x, nw, ring, eps=1e-6, ordd=ordd, code_tab=tab, slot=slot, qdq=True, rope=(fr, RD, (1, 2)))
    for r in range(R6):
        ring1 = torch.zeros(win, N, dtype=BF)
        mod.launch_norm_row(x[r], nw, ring1, eps=1e-6, ordd=ordd, code_tab=tab, slot=slot[r:r + 1], qdq=True,
                            rope=(fr[r].contiguous(), RD, (1, 2)))
        assert _eq(ring[slot[r]], ring1[slot[r]]), r


@pytest.mark.parametrize("mod_name", ["mirror", "kernel"])
def test_router_rows_equal_one_row(KR, mod_name):
    mod = KR if mod_name == "kernel" else R2
    g = torch.Generator().manual_seed(13)
    E, K = 64, 6
    sp = F.softplus(torch.randn(R6, E, generator=g) * 3)
    bias = torch.randn(E, generator=g) * 0.1
    sc, bsc = torch.empty(R6, E), torch.empty(R6, E)
    mod.launch_router_pre(sp.view(-1), bias, sc.view(-1), bsc.view(-1))
    idx = bsc.topk(K, dim=-1)[1].contiguous()
    ids, w = torch.empty(R6, K, dtype=torch.long), torch.empty(R6, K)
    ordd = _one_row_order(K)
    mod.launch_router_post(idx, sc, ids, w, ordd=ordd, eps=1e-20, rs=1.5)
    for r in range(R6):
        a, b = torch.empty(E), torch.empty(E)
        mod.launch_router_pre(sp[r].contiguous(), bias, a, b)
        assert _eq(a, sc[r]) and _eq(b, bsc[r])
        i1, w1 = torch.empty(K, dtype=torch.long), torch.empty(K)
        mod.launch_router_post(idx[r].contiguous(), a, i1, w1, ordd=ordd, eps=1e-20, rs=1.5)
        assert _eq(i1, ids[r]) and _eq(w1, w[r]), r


@pytest.mark.parametrize("mod_name", ["mirror", "kernel"])
def test_engram_rows_equal_one_row(KR, mod_name):
    """Engram's two statistics and the dot over D per (row, hc copy), in the 1-row build's order (picked for the
    [1, 1, hc, D] call: hc rows), R rows == each row alone."""
    mod = KR if mod_name == "kernel" else R2
    HC, D = 4, 128
    g = torch.Generator().manual_seed(14)
    x = _act(g, R6 * HC, D).view(R6, HC, D)
    kv = _act(g, R6, (HC + 1) * D, -4, 4)
    wp = (torch.randn(HC, D, generator=g).to(BF).float() * torch.randn(HC, D, generator=g).to(BF).float()).contiguous()
    o, so = _one_row_order(D, HC), _od(R2.predicted_order(D, HC))
    c, cmin = R2.f32(D ** -0.5), R2.f32(1e-6)
    z = torch.empty(R6 * HC)
    mod.launch_engram_gate(x, kv.reshape(-1).contiguous(), wp, z, eps=1e-6, ordd=o, sordd=so, c=c, cmin=cmin)
    out = torch.empty(R6, HC, D, dtype=BF)
    mod.launch_engram_out(x, kv.reshape(-1).contiguous(), torch.sigmoid(z), out)
    for r in range(R6):
        z1 = torch.empty(HC)
        mod.launch_engram_gate(x[r], kv[r].contiguous(), wp, z1, eps=1e-6, ordd=o, sordd=so, c=c, cmin=cmin)
        o1 = torch.empty(HC, D, dtype=BF)
        mod.launch_engram_out(x[r], kv[r].contiguous(), torch.sigmoid(z1), o1)
        assert _eq(z1, z[r * HC:(r + 1) * HC]) and _eq(o1, out[r]), r


@pytest.mark.parametrize("mod_name", ["mirror", "kernel"])
def test_topk_post_rows_equal_one_row(KR, mod_name):
    mod = KR if mod_name == "kernel" else R2
    g = torch.Generator().manual_seed(15)
    W, K = 96, 32
    sc = (torch.randint(0, 8, (R6, W), generator=g).float() * 0.5).to(BF)
    idx = sc.topk(K, dim=-1, sorted=False).indices.contiguous()
    clen = torch.tensor([40 - 3 * r for r in range(R6)], dtype=torch.long)
    out = torch.empty(R6, K, dtype=torch.int32)
    mod.launch_topk_post(idx, clen, out, 128)
    for r in range(R6):
        o1 = torch.empty(K, dtype=torch.int32)
        mod.launch_topk_post(idx[r].contiguous(), clen[r:r + 1], o1, 128)
        assert _eq(o1, out[r]), r


@pytest.mark.parametrize("mod_name", ["mirror", "kernel"])
def test_round1_kernels_rows_equal_one_row(KF, tab, silu, mod_name):
    """hc_post, o_rope_qdq, SwiGLU (shared rows; routed pair rows), act_quant into the DeepGEMM rows (pair scatter
    vs one-row broadcast), MoE combine (posmat vs plain): R rows == each row alone."""
    mod = KF if mod_name == "kernel" else FG
    g = torch.Generator().manual_seed(16)
    HC, D, I, k = 4, 128, 96, 4
    # hc_post
    x = _act(g, R6, D)
    res = _act(g, R6 * HC, D).view(R6, HC, D)
    post = (torch.rand(R6, HC, generator=g) * 2).float()
    comb = torch.randn(R6, HC, HC, generator=g).float()
    out = torch.empty(R6, HC, D, dtype=BF)
    mod.launch_hc_post(x, res, post, comb, out, 1)
    for r in range(R6):
        o1 = torch.empty(HC, D, dtype=BF)
        mod.launch_hc_post(x[r].contiguous(), res[r].contiguous(), post[r].contiguous(), comb[r].contiguous(), o1, 1)
        assert _eq(o1, out[r]), ("hc_post", r)
    # o path: inverse RoPE + QDQ, row t at its own position
    H, hd, rd = 2, 64, 16
    o = _act(g, R6, H * hd)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    fr = torch.view_as_real(torch.polar(torch.ones(R6, rd // 2),
                                        torch.arange(7, 7 + R6).view(-1, 1).float() * inv)).contiguous()
    full = o.clone()
    mod.launch_o_rope_qdq(full, fr, tab, hd, rd, (1, 1))
    for r in range(R6):
        o1 = o[r].clone()
        mod.launch_o_rope_qdq(o1, fr[r].contiguous(), tab, hd, rd, (1, 1))
        assert _eq(o1, full[r]), ("o_rope_qdq", r)
    # SwiGLU, shared form (rows) and routed pair-row form (IDX) vs one row
    gate, up = _act(g, 2 * R6, I, -3, 7).view(2, R6, I)
    q, s = torch.empty(R6, I, dtype=FP8), torch.empty(R6, I // 32, dtype=E8)
    mod.launch_swiglu_quant(gate, up, None, q, s, silu, tab, 10.0)
    A = 3 * R6
    ga, ua = _act(g, 2 * A, I, -3, 7).view(2, A, I)
    ri = torch.randperm(A, generator=g)[:R6]
    w = (torch.rand(R6, generator=g) * 2).float()
    qa, sa = torch.zeros(A, I, dtype=FP8), torch.zeros(A, I // 32)
    mod.launch_swiglu_quant(ga, ua, w, qa, sa, silu, tab, 10.0, rows_idx=ri)
    for r in range(R6):
        q1, s1 = torch.empty(1, I, dtype=FP8), torch.empty(1, I // 32, dtype=E8)
        mod.launch_swiglu_quant(gate[r:r + 1], up[r:r + 1], None, q1, s1, silu, tab, 10.0)
        assert _eq(q1[0], q[r]) and _eq(s1[0], s[r]), ("swiglu_shared", r)
        a = int(ri[r])
        q2, s2 = torch.empty(1, I, dtype=FP8), torch.empty(1, I // 32)
        mod.launch_swiglu_quant(ga[a:a + 1], ua[a:a + 1], w[r:r + 1], q2, s2, silu, tab, 10.0)
        assert _eq(q2[0], qa[a]) and _eq(s2[0], sa[a]), ("swiglu_routed", r)
    # act_quant into the DeepGEMM rows: R-row pair scatter vs the 1-row broadcast to k groups
    xd = _act(g, R6, D)
    P = R6 * k
    prow = torch.randint(0, R6, (P,), generator=g)
    arow = torch.randperm(2 * P, generator=g)[:P]
    qd, sd = torch.zeros(2 * P, D, dtype=FP8), torch.zeros(2 * P, D // 32)
    mod.launch_act_quant_bcast(xd, qd, sd, tab, 1, src_rows=prow, dst_rows=arow)
    for p in range(P):
        r = int(prow[p])
        q1, s1 = torch.empty(k, D, dtype=FP8), torch.empty(k, D // 32)
        mod.launch_act_quant_bcast(xd[r:r + 1], q1, s1, tab, k)
        assert _eq(q1[0], qd[arow[p]]) and _eq(s1[k - 1], sd[arow[p]]), ("dg_act", p)
    # MoE combine: posmat (R rows) vs plain (one row's k rows in that order)
    rows = _act(g, 2 * P, D, -8, 8)
    sh = _act(g, R6, D, -8, 8)
    posmat = torch.stack([torch.randperm(2 * P, generator=g)[:k] for _ in range(R6)])
    y = torch.empty(R6, D, dtype=BF)
    mod.launch_moe_combine(rows, sh, y, posmat=posmat)
    for r in range(R6):
        y1 = torch.empty(D, dtype=BF)
        mod.launch_moe_combine(rows.index_select(0, posmat[r]).contiguous(), sh[r].contiguous(), y1)
        assert _eq(y1, y[r]), ("moe_combine", r)


def test_native_mirror_reduces_each_row_alone_under_shape(monkeypatch):
    """The CPU mirrors' NATIVE order is torch's op on the reference call's shape -- under `shape`, on each token row
    alone (the 1-row call); under `rows`, on the R-row shape (the pre-T9b-7 mirror)."""
    seen = []
    real = torch.Tensor.mean

    def spy(x, *a, **k):
        seen.append(tuple(x.shape))
        return real(x, *a, **k)

    monkeypatch.setattr(torch.Tensor, "mean", spy)
    v = torch.randn(R6, 64, generator=torch.Generator().manual_seed(3))
    monkeypatch.setenv(R2.GLUE_ORDER_ENV, "shape")
    a = R2.sqmean_torch(v, 64, NATIVE, (1, R6, 64))
    assert set(seen) == {(1, 1, 64)} and len(seen) >= R6   # (a set: the default-device mode re-enters the patch)
    seen.clear()
    monkeypatch.setenv(R2.GLUE_ORDER_ENV, "rows")
    b = R2.sqmean_torch(v, 64, NATIVE, (1, R6, 64))
    assert set(seen) == {(1, R6, 64)} and _eq(a, b)       # (CPU torch is row-invariant: same values)


# ================================================================================ 4. the census, on the emulation
@contextlib.contextmanager
def _env(**kv):
    prev = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _rowwise_linear(monkeypatch):
    """F.linear one activation row at a time: CPU BLAS (like cuBLAS) is not row-invariant for the router's fp32 gate
    mm -- the step's GEMM site, not the glue's (the census note says so); on the Mac it is isolated this way."""
    lin0 = F.linear

    def linear(x, w, b=None):
        K = x.size(-1)
        M = x.numel() // K
        if M <= 1:
            return lin0(x, w, b)
        xv = x.reshape(M, K)
        return torch.cat([lin0(xv[i:i + 1], w, b) for i in range(M)], 0).view(*x.shape[:-1], -1)

    monkeypatch.setattr(F, "linear", linear)


def test_glue_row_census_on_the_emulation(rows_glue, monkeypatch):
    """glue_row_census of a 1-row and a 6-row dg-inplace fused-glue build (fp8 ref on the Mac: the GEMV parts are n/a):
    the orders record is equal and every cluster on in both builds reads row-invariant; the records are
    census_table-shaped (one family per cluster)."""
    from runtime.row_census import census_table

    model, pool, _, _ = rows_glue("mini4")
    monkeypatch.delenv(R2.GLUE_ORDER_ENV, raising=False)
    _rowwise_linear(monkeypatch)
    with _env(DSV41_FP32_IMPL="fixed"):
        s6 = _build(model, pool, R6, spare_slots=8)
        s1 = _build(model, pool, 1)
        recs = R2.glue_row_census(s1.glue, s6.glue)
        s6.detach()
        s1.detach()
    lines = census_table(recs)
    print("\n".join(lines))
    fam = {r["family"] for r in recs}
    assert "fused glue orders" in fam and "fused glue hc_norm" in fam and "fused glue engram" in fam
    for r in recs:
        if r.get("equal") is None:
            assert r["note"].startswith("n/a"), r
        else:
            assert r["equal"] is True, r
    on6 = {c for c in FG.CLUSTERS if s6.glue.on(c) and s1.glue.on(c)} - set(R2.GLUE_CENSUS_NA)
    ran = {r["family"].split(" ", 2)[-1] for r in recs if r.get("equal") is True}
    assert on6 <= ran, on6 - ran
    assert {"hc_norm", "mix_stat", "kv_path", "q_path", "router", "engram", "indexer_post", "hc_post", "o_rope_qdq",
            "swiglu_shared", "swiglu_routed", "dg_act", "moe_combine"} <= ran, ran


def test_gpu_rows_t9b7_test_dry_run_on_cpu(rows_glue, monkeypatch, capsys):
    """tests/gpu/test_static_rows_gpu.py test_t9b7_glue_clusters_row_invariant on the CPU emulation (DEV cpu; fp8 ref,
    attn ref): the harness (the two builds, both policies, the census, the step-level rows) runs, the gate passes."""
    spec = importlib.util.spec_from_file_location("gpu_rows_t9b7",
                                                  os.path.join(REPO, "tests", "gpu", "test_static_rows_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    @contextlib.contextmanager
    def model_cpu(kind="mini"):
        model, pool, _, _ = rows_glue(kind)
        yield model, pool

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STATIC_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    monkeypatch.setattr(G, "FAST", dict(moe_impl="dg-masked", fp8_impl=None, glue_impl="fused", attn_impl=None))
    monkeypatch.delenv(R2.GLUE_ORDER_ENV, raising=False)
    _rowwise_linear(monkeypatch)
    G.test_t9b7_glue_clusters_row_invariant()
    out = capsys.readouterr().out
    assert "fused clusters row-invariant at 6 rows" in out and "T9b-7 A/B" in out, out
    assert out.count("T9b-7 STEP-LEVEL") == 2, out
    assert os.environ.get(R2.GLUE_ORDER_ENV) is None      # the test's env changes are undone
    print(out)
