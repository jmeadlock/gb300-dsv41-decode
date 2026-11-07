"""Mac CPU tests of T9b-2 round 2 (runtime/fused_glue_r2.py: hc_norm, mix_stat, kv_path, q_path, router, engram,
indexer_post). `.venv/bin/python -m pytest -q tests/test_fused_glue_r2_cpu.py`

  1. the reduction-order family: `order_sum_torch` == an independent scalar float32 simulation of the definition
     (element (k*T + t)*V + v / (k*V + v)*T + t, V accumulators folded left, the T tree halving above 32 and dec/inc
     below) for every family shape on ragged widths; `pick_order` identifies a hidden member (outcome (i)) and
     reports a non-member as outcome (ii) with its match fraction and max ulp; torch's predicted configuration is the
     one the box probe measured at every normed width;
  2. the real `@triton.jit` kernel bodies executed through tests/triton_shim.py == the torch mirrors bit for bit, for
     explicit orders (each family / finish / hc-sum mode / RoPE form), ragged block tails, strided DeepGEMM rows,
     the ring slot, tie-heavy router scores and topk indices;
  3. the mirrors (torch's own reduction: what the CPU step runs) == the reference op sequences of model.py;
  4. StaticStep(glue_impl="fused") with each R2 cluster on/off == glue "ref" (R1's test covers all clusters on);
     launches per step; outcome-(ii) gating (off by default, `approx` with glue_approx); the self-check turning a
     broken kernel off; Generator / first_light --glue-approx plumbing.
Not checked here: Triton compilation, torch's CUDA reduction order, libdevice (tests/gpu/test_fused_glue_r2_gpu.py and
the build-time calibration do that on the box).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import triton_shim
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from test_gemv_cpu import _static_run, emu, gemv_emu  # noqa: F401 (fixtures)
from test_static_emulated_cpu import act_quant as _emu_act_quant

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def _restore_dtype():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


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


# ================================================================================ 0. module contract
def test_constants_agree_with_round1():
    assert (R2.BLOCK, R2.FP8, R2.E8, R2.HC_SUM_MODES) == (FG.BLOCK, FG.FP8, FG.E8, FG.HC_SUM_MODES)
    assert R2.R2_CLUSTERS == FG.R2_CLUSTER_NAMES and set(R2.R2_CLUSTERS) <= set(FG.CLUSTERS)
    assert set(R2.R2_ACT_CLUSTERS) <= set(FG.ACT_CLUSTERS)
    assert FG.FusedGlue.__mro__[1] is R2.R2Glue
    assert R2.LAUNCH == FG.LAUNCH == {"enable_fp_fusion": False}


@pytest.fixture(scope="module")
def KR():
    """runtime/fused_glue_r2.py imported with the shim as `triton` (a private copy), wrappers launching kernels."""
    mod = triton_shim.load_module_under_shim(os.environ.get("FG2_SHIM_MODULE")
                                             or os.path.join(REPO, "runtime", "fused_glue_r2.py"),
                                             "fused_glue_r2_under_shim")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KF():
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue.py"), "fused_glue_under_shim_r2")


def test_triton_helpers_are_verbatim_copies(KR, KF):
    """A jit function resolves its callees in its own module: R2 carries copies of R1's helpers, byte-identical."""
    for name in ("_t9b2_mul_rn", "_t9b2_add_rn", "_t9b2_sub_rn", "_t9b2_rope_re", "_t9b2_rope_im", "_t9b2_code",
                 "_t9b2_pow2", "_t9b2_ipow2", "_t9b2_q8", "_t9b2_bf16"):
        assert inspect.getsource(getattr(KR, name).fn) == inspect.getsource(getattr(KF, name).fn), name


def test_host_constants_never_touch_the_default_device():
    """fac_of / f32 build CPU tensors explicitly (the runtime's default device is cuda; a host scalar must not sync
    or allocate there): under a meta default device they still work."""
    R2._F32_CACHE.clear()
    with torch.device("meta"):
        assert R2.fac_of(5120) == float(np.float32(1.0) / np.float32(5120.0))
        assert R2.f32(5120 ** -0.5) == float(np.float32(5120 ** -0.5))
        x = R2._stat_rows(torch.Generator().manual_seed(0), 4, 96, "sqmean")
    assert x.device.type == "cpu"


# ================================================================================ 1. the reduction-order family
def _np_order_sum(row: np.ndarray, order) -> np.float32:
    """The family's definition, one scalar float32 add at a time (independent of order_sum_torch)."""
    fam, T, V, dec = order
    w = row.size
    KN = -(-w // (T * V))
    acc = np.zeros((T, V), np.float32)
    for k in range(KN):
        for t in range(T):
            for v in range(V):
                i = (k * T + t) * V + v if fam == "vec" else (k * V + v) * T + t
                acc[t, v] = np.float32(acc[t, v] + (row[i] if i < w else np.float32(0)))
    s = [acc[t, 0] for t in range(T)]
    for t in range(T):
        for j in range(1, V):
            s[t] = np.float32(s[t] + acc[t, j])
    n = T
    while n > 1:
        if dec or n > 32:
            s = [np.float32(s[i] + s[i + n // 2]) for i in range(n // 2)]
        else:
            s = [np.float32(s[2 * i] + s[2 * i + 1]) for i in range(n // 2)]
        n //= 2
    return s[0]


@pytest.mark.parametrize("w", [6, 96, 160, 512])
def test_order_sum_torch_is_the_family_definition(w):
    g = torch.Generator().manual_seed(w)
    x = R2._stat_rows(g, 3, w, "sum")
    cands = R2.order_candidates(w)
    assert cands[0] == R2.predicted_order(w) and len(set(cands)) == len(cands)
    for o in cands[:: max(1, len(cands) // 12)] + [("unroll", 64, 2, False), ("vec", 64, 1, False)]:
        got = R2.order_sum_torch(x, o)
        for r in range(x.size(0)):
            want = _np_order_sum(x[r].numpy().astype(np.float32), o)
            assert got[r].view(torch.int32).item() == np.float32(want).view(np.int32), (w, o, r)
    assert _eq(R2.order_sum_torch(x, R2.NATIVE), x.sum(-1))


def test_family_members_differ_on_calibration_rows():
    """The calibration rows separate the family: neighbouring members disagree on some rows (else a pick would not
    mean anything)."""
    w = 512
    x = R2._stat_rows(torch.Generator().manual_seed(1), 256, w, "sqmean").float()
    sq = x * x
    base = R2.order_sum_torch(sq, R2.predicted_order(w))
    for o in [("vec", 128, 4, False), ("vec", 64, 4, True), ("unroll", 128, 4, True), ("vec", 128, 2, True)]:
        assert not _eq(R2.order_sum_torch(sq, o), base), o
    m = R2.mean_finish(base, w, "mul")
    d = R2.mean_finish(base, w, "div")
    assert _eq(m, base * R2.fac_of(w)) and _eq(d, base / w)


@pytest.mark.parametrize("w,rows,want", [(5120, 1, ("vec", 512, 4, True)), (20480, 1, ("vec", 512, 4, True)),
                                         (1280, 1, ("vec", 256, 4, True)), (512, 1, ("vec", 128, 4, True)),
                                         (128, 1, ("vec", 32, 4, True)), (5120, 4, ("vec", 128, 4, True)),
                                         (6, 1, ("vec", 1, 4, True))])
def test_predicted_order_is_torch_launch_configuration(w, rows, want):
    """Reduce.cuh set_block_dimension with 4-wide vector loads; at M=1 this is (vec, min(512, last_pow2(w/4)), 4) --
    the configuration docs/flashmla-probe-20261106.json matched on 100 % of 100k rows at every normed width."""
    assert R2.predicted_order(w, rows) == want
    if rows == 1 and w >= 4:
        assert want[1] == min(512, R2.last_pow2(w // 4))


@pytest.mark.parametrize("hidden,fin,w", [(("vec", 128, 4, True), "mul", 512), (("vec", 128, 4, False), "mul", 512),
                                          (("unroll", 64, 4, True), "div", 480), (("vec", 16, 1, False), "mul", 512)])
def test_pick_order_identifies_a_hidden_member(hidden, fin, w):
    """(w = 480 for the division: at a power of two RN(1/w) is exact and the two finishes coincide.)"""
    ref = lambda x: R2.mean_finish(R2.order_sum_torch(x * x, hidden), w, fin)  # noqa: E731
    rec = R2.pick_order("cpu", w, 1, "sqmean", n=96, force=True, ref_fn=ref)
    assert rec["exact"] and rec["match"] == 1.0 and rec["max_ulp"] == 0 and rec["fin"] == fin
    if hidden == R2.predicted_order(w):
        assert rec["order"] == hidden and rec["tried"] == 1
    fresh = R2._stat_rows(torch.Generator().manual_seed(99), 128, w, "sqmean").float()
    assert _eq(R2.mean_finish(R2.order_sum_torch(fresh * fresh, rec["order"]), w, rec["fin"]), ref(fresh))


def test_pick_order_reports_outcome_ii_for_a_non_member():
    """A correctly rounded (fp64) sum is not a member: exact False, a match fraction < 1 and the max ulp."""
    w = 512
    ref = lambda x: (x.double() * x.double()).sum(-1).float() * R2.fac_of(w)  # noqa: E731
    rec = R2.pick_order("cpu", w, 1, "sqmean", n=128, force=True, ref_fn=ref)
    assert not rec["exact"] and 0.0 < rec["match"] < 1.0 and 1 <= rec["max_ulp"] <= 16
    assert rec["tried"] == rec["candidates"] == len(R2.order_candidates(w))


def test_pick_order_on_cpu_is_torch_native_and_cached():
    rec = R2.pick_order("cpu", 5120, 1)
    assert rec["order"] == R2.NATIVE and rec["exact"]
    R2._ORDER_CACHE.clear()
    a = R2.pick_order("cpu", 96, 1, "sqmean", n=32, force=True)       # CPU torch on these rows: some member or not
    assert ("cpu", "sqmean", 96, 1, 32, 0) in R2._ORDER_CACHE and R2.pick_order("cpu", 96, 1, "sqmean", n=32,
                                                                                force=False) == a
    R2._ORDER_CACHE.clear()


# ================================================================================ 2. kernel bodies under the shim
def _launches(fn):
    return len(fn.launches)


@pytest.mark.parametrize("w", [96, 512, 1280])
def test_kernel_reduction_helpers_are_the_family(KR, w):
    """The kernels' reduction (the `_r2_offs` tile loop, `_r2_finish`'s V fold + T tree, `_r2_fin`), executed through
    the shim on the calibration rows (outliers, 2^+-8 block scales: the members differ there), == order_sum_torch /
    mean_finish for every family member -- every T level, both trees, both layouts, both finishes."""
    tl = KR.tl
    x = R2._stat_rows(torch.Generator().manual_seed(w), 24, w, "sqmean").float()
    for o in R2.order_candidates(w):
        fam, T, V, dec = o
        want = R2.order_sum_torch(x * x, o)
        for r in range(0, x.size(0), 6):
            buf = x[r].numpy().astype(np.float32).copy()
            acc = tl.zeros([T, V], dtype=tl.float32)
            for k in range(-(-w // (T * V))):
                e = KR._r2_offs(k, T, V, fam == "unroll")
                v = tl.load(triton_shim.P(buf, 0) + e, mask=e < w, other=0.0)
                acc = acc + v * v
            s = KR._r2_finish(acc, T, V, dec)
            assert np.float32(triton_shim._v(s)).view(np.int32) == want[r].view(torch.int32).item(), (o, r)
            for fin, div in (("mul", False), ("div", True)):
                m = KR._r2_fin(s, np.float32(R2.fac_of(w)), np.float32(w), div)
                assert np.float32(triton_shim._v(m)).view(np.int32) == \
                    R2.mean_finish(want[r:r + 1], w, fin).view(torch.int32).item(), (o, fin)


ORDERS_D = [(("vec", 64, 4, True), "mul"), (("unroll", 32, 2, False), "div"), (("vec", 8, 1, False), "mul")]


@pytest.mark.parametrize("D", [256, 96])
@pytest.mark.parametrize("mode", R2.HC_SUM_MODES)
@pytest.mark.parametrize("oi", range(len(ORDERS_D)))
def test_kernel_hc_norm(KR, tab, D, mode, oi):
    HC = 4
    g = torch.Generator().manual_seed(D + 10 * mode + oi)
    x = _act(g, HC, D)
    pre = torch.rand(HC, generator=g) + 1e-3
    nw = (torch.randn(D, generator=g) * 0.5 + 1.0).to(BF)
    ordd = _od(*ORDERS_D[oi])
    mix = _od(("vec", 128, 4, oi != 1), "mul")
    outs = {}
    for mod, name in ((R2, "mirror"), (KR, "kernel")):
        o = dict(y=torch.empty(D, dtype=BF), yf=torch.empty(D), xf=torch.empty(HC * D), mrs=torch.empty(1),
                 q=torch.empty(D, dtype=FP8), s=torch.empty(D // 32, dtype=E8),
                 dq3=torch.full((3, 5, D), 7.0).to(FP8), ds3=torch.full((3, 5, D // 32), -1.0))
        n0 = _launches(KR.t9b2r2_hc_norm)
        mod.launch_hc_norm(x, pre, nw, o["y"], eps=1e-6, norm_ord=ordd, hc_mode=mode, code_tab=tab, yf=o["yf"],
                           xf=o["xf"], mrs=o["mrs"], mix_ord=mix, mix_eps=1e-6, q=o["q"], s=o["s"],
                           dq=o["dq3"][:, 0], ds=o["ds3"][:, 0])
        assert _launches(KR.t9b2r2_hc_norm) - n0 == (1 if name == "kernel" else 0)
        outs[name] = o
    for k in outs["mirror"]:
        assert _eq(outs["kernel"][k], outs["mirror"][k]), (k, D, mode, oi)
    assert (outs["kernel"]["dq3"][:, 1:].float() == 7.0).all()                  # only row 0 of each group written


def test_kernel_hc_norm_head_form(KR):
    """The head's call: no mixes, no quantization, fp32 copy of y (no code table)."""
    D, HC = 160, 4
    g = torch.Generator().manual_seed(3)
    x, pre, nw = _act(g, HC, D), torch.rand(HC, generator=g), torch.randn(D, generator=g).to(BF)
    res = []
    for mod in (R2, KR):
        y, yf = torch.empty(D, dtype=BF), torch.empty(D)
        mod.launch_hc_norm(x, pre, nw, y, eps=1e-20, norm_ord=_od(("vec", 32, 4, True)), hc_mode=0, yf=yf)
        res.append((y, yf))
    assert _eq(res[0][0], res[1][0]) and _eq(res[0][1], res[1][1]) and _eq(res[0][1], res[0][0].float())


def _freqs(rd, pos, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    ang = pos * inv
    return torch.polar(torch.ones_like(ang), ang)


@pytest.mark.parametrize("N,RD,slot,qdq,want_q", [(128, 0, None, False, True), (96, 32, 2, True, False),
                                                  (64, 0, None, True, False), (512, 64, 3, True, False),
                                                  (1280 // 4, 0, None, False, True)])
@pytest.mark.parametrize("modes", [(0, 0), (1, 1), (2, 2)])
def test_kernel_norm_row(KR, tab, N, RD, slot, qdq, want_q, modes):
    g = torch.Generator().manual_seed(N + RD + modes[0])
    x = _act(g, 1, N, -4, 6).view(-1)
    nw = (torch.randn(N, generator=g) * 0.3 + 1.0).to(BF)
    fr = torch.view_as_real(_freqs(RD, 77)).reshape(-1, 2).contiguous() if RD else None
    ordd = _od(R2.predicted_order(N), "mul")
    outs = []
    for mod in (R2, KR):
        win = 4
        out = torch.full((win, N) if slot is not None else (N,), 3.0, dtype=BF)
        q = torch.empty(N, dtype=FP8) if want_q else None
        s = torch.empty(N // 32, dtype=E8) if want_q else None
        sl = torch.tensor([slot], dtype=torch.long) if slot is not None else None
        mod.launch_norm_row(x, nw, out, eps=1e-6, ordd=ordd, code_tab=tab, slot=sl, q=q, s=s, qdq=qdq,
                            rope=(fr, RD, modes) if RD else None)
        outs.append((out, q, s))
    for a, b in zip(*outs):
        assert (a is None and b is None) or _eq(a, b)
    if slot is not None:
        other = [r for r in range(4) if r != slot]
        assert (outs[1][0][other].float() == 3.0).all()


def test_norm_row_mirror_is_the_reference_kv_and_q_sequences(tab):
    """NATIVE order: launch_norm_row's mirror == kv_norm -> apply_rotary_emb -> act_quant QDQ -> ring row, and ==
    q_norm + act_quant(qr), written with torch ops as model.py does (CPU torch's own reduction)."""
    g = torch.Generator().manual_seed(5)
    N, RD = 96, 32
    x = _act(g, 1, N, -4, 6).view(1, 1, N)
    nw = (torch.randn(N, generator=g) * 0.3 + 1.0).to(BF)
    f = _freqs(RD, 41).view(1, -1)
    v = x.float()
    kv = (nw * (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6))).to(BF)
    xr = torch.view_as_complex(kv[..., -RD:].float().unflatten(-1, (-1, 2)))
    kv[..., -RD:].copy_(torch.view_as_real(xr * f.view(1, 1, -1)).flatten(-2))
    want = FG.qdq_torch(kv.float().view(-1), tab)
    ring = torch.zeros(4, N, dtype=BF)
    R2.launch_norm_row(x.view(-1), nw, ring, eps=1e-6, ordd=NATIVE, code_tab=tab,
                       slot=torch.tensor([1]), qdq=True, rope=(torch.view_as_real(f).reshape(-1, 2), RD, (0, 0)))
    assert _eq(ring[1], want) and (ring[[0, 2, 3]].float() == 0).all()
    qr = torch.empty(N, dtype=BF)
    q, s = torch.empty(N, dtype=FP8), torch.empty(N // 32, dtype=E8)
    R2.launch_norm_row(x.view(-1), nw, qr, eps=1e-6, ordd=NATIVE, code_tab=tab, q=q, s=s)
    qr_ref = (nw * (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6))).to(BF).view(-1)
    q_ref, s_ref = _ref_aq(qr_ref.view(1, -1))
    assert _eq(qr, qr_ref) and _eq(q, q_ref.view(-1)) and _eq(s, s_ref.view(-1))


def _router_case(g, E, K, ties):
    if ties:   # few distinct values: topk sees exact ties in score + bias, and equal weights
        sc = torch.randint(1, 5, (E,), generator=g).float() * 0.25
        bias = torch.where(torch.rand(E, generator=g) < 0.5, 0.0, 0.25)
    else:
        sc = F.softplus(torch.randn(E, generator=g) * 2).sqrt()
        bias = torch.randn(E, generator=g) * 0.1
    idx = (sc + bias).view(1, -1).topk(K, dim=-1)[1].view(-1)
    return sc, bias, idx


@pytest.mark.parametrize("E,K,ties", [(64, 6, True), (64, 6, False), (384, 6, True), (16, 8, True), (8, 2, False)])
def test_kernel_router(KR, E, K, ties):
    g = torch.Generator().manual_seed(E + K + ties)
    sp = F.softplus(torch.randn(E, generator=g) * 3)
    sc, bias, idx = _router_case(g, E, K, ties)
    res = []
    for mod in (R2, KR):
        a, b = torch.empty(E), torch.empty(E)
        mod.launch_router_pre(sp, bias, a, b)
        ids, w = torch.empty(K, dtype=torch.long), torch.empty(K)
        for o in (_od(R2.predicted_order(K)), _od(("vec", 8, 1, False))):
            mod.launch_router_post(idx, sc, ids, w, ordd=o, eps=1e-20, rs=1.5)
            res.append((a.clone(), b.clone(), ids.clone(), w.clone()))
    for x, y in zip(res[:2], res[2:]):
        assert all(_eq(p, q) for p, q in zip(x, y))
    assert _eq(res[0][0], sp.sqrt()) and _eq(res[0][1], sp.sqrt() + bias)
    # NATIVE mirror == Gate.forward's ops after the topk (model.py:825-828) + the step's ascending sort
    wt = sc.view(1, -1).gather(1, idx.view(1, -1))
    wt /= wt.sum(dim=-1, keepdim=True) + 1e-20
    wt *= 1.5
    ids_r, order = torch.sort(idx)
    ids, w = R2.router_post_torch(idx, sc, NATIVE, 1e-20, 1.5)
    assert _eq(ids, ids_r) and _eq(w, wt[0, order])


def _engram_ref(x, kv, qw, kw, eps, clamp, HC, D):
    """Engram.forward (model.py:353-365) after wkv, token_mask None."""
    key, value = kv.split([HC * D, D], dim=-1)
    key = key.float().unflatten(-1, (HC, D))
    weight = qw.float() * kw.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * D ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


@pytest.mark.parametrize("D,zero_key", [(96, False), (128, True), (160, False)])
def test_kernel_engram(KR, D, zero_key):
    HC = 4
    g = torch.Generator().manual_seed(D + zero_key)
    x = _act(g, HC, D).view(1, 1, HC, D)
    kv = _act(g, 1, (HC + 1) * D, -4, 4).view(1, 1, -1)
    if zero_key:
        kv[..., :D] = 0                                     # dot exactly 0 for copy 0: clamp + copysign(+0)
        kv[..., D:2 * D] = -kv[..., D:2 * D]
    qw, kw = torch.randn(HC, D, generator=g).to(BF), torch.randn(HC, D, generator=g).to(BF)
    wp = (qw.float() * kw.float()).contiguous()
    c, cmin = R2.f32(D ** -0.5), R2.f32(1e-6)
    res = []
    for mod in (R2, KR):
        for o, so in ((_od(("vec", 16, 4, True)), _od(("vec", 32, 2, True))),
                      (_od(("unroll", 8, 4, False), "div"), _od(("vec", 4, 1, False)))):
            z = torch.empty(HC)
            mod.launch_engram_gate(x.view(HC, D), kv.view(-1), wp, z, eps=1e-6, ordd=o, sordd=so, c=c, cmin=cmin)
            out = torch.empty(HC, D, dtype=BF)
            mod.launch_engram_out(x.view(HC, D), kv.view(-1), torch.sigmoid(z), out)
            res.append((z, out))
    for (z1, o1), (z2, o2) in zip(res[:2], res[2:]):
        assert _eq(z1, z2) and _eq(o1, o2)
    z = torch.empty(HC)
    R2.launch_engram_gate(x.view(HC, D), kv.view(-1), wp, z, eps=1e-6, ordd=NATIVE, sordd=NATIVE, c=c, cmin=cmin)
    out = torch.empty(HC, D, dtype=BF)
    R2.launch_engram_out(x.view(HC, D), kv.view(-1), torch.sigmoid(z), out)
    assert _eq(out.view(1, 1, HC, D), _engram_ref(x, kv, qw, kw, 1e-6, 1e-6, HC, D))


@pytest.mark.parametrize("W,K,clen", [(100, 37, 50), (300, 128, 300), (64, 64, 1), (640, 512, 200)])
def test_kernel_topk_post(KR, W, K, clen):
    from runtime.static_step import indexer_topk_static

    g = torch.Generator().manual_seed(W + K)
    sc = (torch.randint(0, 8, (1, 1, W), generator=g).float() * 0.5).to(BF)   # ties everywhere
    sc[..., W // 2:] = -float("inf")
    cl = torch.tensor([clen], dtype=torch.long)
    idx = sc.topk(K, dim=-1, sorted=False).indices
    res = []
    for mod in (R2, KR):
        out = torch.empty(1, 1, K, dtype=torch.int32)
        mod.launch_topk_post(idx.reshape(-1), cl, out.view(-1), 128)
        res.append(out)
    assert _eq(res[0], res[1]) and _eq(res[0], indexer_topk_static(sc, cl, K, 128))


# ================================================================================ 3. mirrors == reference ops
def _hc_cancel(g, HC, D):
    """Copies 0 and 2 carry the same large value with pre_2 = -pre_0 in half the columns: a0 + a2 cancels exactly,
    so the 4-term sum's order shows in the result."""
    x = _act(g, HC, D).float()
    pre = torch.rand(HC, generator=g) + 0.1
    pre[2] = -pre[0]
    half = torch.arange(D) % 2 == 0
    big = torch.randn(D, generator=g) * 4096
    x[0] = torch.where(half, big, x[0])
    x[2] = torch.where(half, big, x[2])
    return x.to(BF), pre


def test_hc_pre_modes_and_cpu_torch_is_mode_0():
    g = torch.Generator().manual_seed(8)
    HC, D = 4, 256
    x, pre = _hc_cancel(g, HC, D)
    ref = torch.sum(pre.view(1, 1, HC, 1) * x.view(1, 1, HC, D).float(), dim=2).to(BF).view(-1)   # model.py:961
    got = {m: R2.hc_pre_torch(x, pre, m) for m in R2.HC_SUM_MODES}
    assert _eq(got[0], ref)
    assert not _eq(got[1], got[0]) and not _eq(got[2], got[0])


def test_hc_norm_mirror_is_hc_pre_rmsnorm_act_quant(tab):
    g = torch.Generator().manual_seed(9)
    HC, D = 4, 256
    x = _act(g, HC, D)
    pre = torch.rand(HC, generator=g)
    nw = torch.randn(D, generator=g).to(BF)
    y, yf, xf, mrs = torch.empty(D, dtype=BF), torch.empty(D), torch.empty(HC * D), torch.empty(1)
    q, s = torch.empty(D, dtype=FP8), torch.empty(D // 32, dtype=E8)
    R2.launch_hc_norm(x, pre, nw, y, eps=1e-20, norm_ord=NATIVE, hc_mode=0, code_tab=tab, yf=yf, xf=xf, mrs=mrs,
                      mix_ord=NATIVE, mix_eps=1e-20, q=q, s=s)
    x4 = x.view(1, 1, HC, D)
    y0 = torch.sum(pre.view(1, 1, HC, 1) * x4.float(), dim=2).to(BF)
    v = y0.float()
    y_ref = (nw * (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-20))).to(BF)       # RMSNorm
    xf_ref = x4.flatten(2).float()
    mrs_ref = torch.rsqrt(xf_ref.square().mean(-1, keepdim=True) + 1e-20)                    # hc_mixes statistic
    q_ref, s_ref = _ref_aq(y_ref.view(1, -1))
    assert _eq(y, y_ref.view(-1)) and _eq(yf, y_ref.float().view(-1)) and _eq(xf, xf_ref.view(-1))
    assert _eq(mrs, mrs_ref.view(-1)) and _eq(q, q_ref.view(-1)) and _eq(s, s_ref.view(-1))


# ================================================================================ 4. StaticStep / plumbing
@pytest.fixture
def glue_emu(gemv_emu, monkeypatch):
    from test_dg_moe_cpu import _fake_deep_gemm

    monkeypatch.setitem(sys.modules, "deep_gemm", _fake_deep_gemm())
    monkeypatch.delenv(FG.DISABLE_ENV, raising=False)
    R2._ORDER_CACHE.clear()
    yield gemv_emu
    R2._ORDER_CACHE.clear()


def _equal_to_ref(model, pool, st, moe, fp8, ids, steps=4):
    from runtime.static_step import StaticStep

    got = _static_run(model, st, ids, steps)
    st.detach()
    ref = StaticStep(model, pool, max_pos=127, moe_impl=moe, fp8_impl=fp8, _allow_cpu=True)
    want = _static_run(model, ref, ids, steps)
    ref.detach()
    return all(torch.equal(a, b) for a, b in zip(got[0], want[0])) and got[1] == want[1]


@pytest.mark.parametrize("off", R2.R2_CLUSTERS)
def test_each_r2_cluster_alone_off_equals_ref(glue_emu, off):
    from runtime.static_step import StaticStep

    model, pool, _ = glue_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    glue_opts={"disable": [off]}, _allow_cpu=True)
    d = st.glue.describe()
    assert d["clusters"][off] == f"off: {FG.DISABLE_ENV}"
    on = set(R2.R2_CLUSTERS) - {off} - ({"mix_stat"} if off == "hc_norm" else set())
    assert on <= set(d["on"]), d["clusters"]
    assert all(d["r2"]["outcome"][c].startswith("(i)") for c in on)
    assert _equal_to_ref(model, pool, st, "dg-masked", "gemv", list(range(30, 47)))


@pytest.mark.parametrize("moe,fp8", [("ref", "ref"), ("dg-masked", "gemv")])
def test_r2_launches_per_step(glue_emu, monkeypatch, moe, fp8):
    """One hc_norm per (block side) + the head, one norm_row per q and kv path, router 2 per layer, Engram 2 per
    Engram layer, one topk post per index-source layer."""
    from runtime.static_step import StaticStep

    model, pool, _ = glue_emu("mini4")
    names = ("launch_hc_norm", "launch_norm_row", "launch_router_pre", "launch_router_post", "launch_engram_gate",
             "launch_engram_out", "launch_topk_post")
    st = StaticStep(model, pool, max_pos=127, moe_impl=moe, fp8_impl=fp8, glue_impl="fused", _allow_cpu=True)
    calls = {n: 0 for n in names}
    for n in names:
        real = getattr(R2, n)

        def wrap(*a, _real=real, _n=n, **k):
            calls[_n] += 1
            return _real(*a, **k)
        monkeypatch.setattr(R2, n, wrap)
    _static_run(model, st, list(range(10, 15)), 1)    # prefill (eager reference) + 1 static step
    L = len(model.layers)
    n_eng = sum(l.engram is not None for l in model.layers)
    n_idx = sum(l.attn.is_index_source for l in model.layers)
    assert calls == {"launch_hc_norm": 2 * L + 1, "launch_norm_row": 2 * L, "launch_router_pre": L,
                     "launch_router_post": L, "launch_engram_gate": n_eng, "launch_engram_out": n_eng,
                     "launch_topk_post": n_idx}, calls
    st.detach()


def _fake_inexact(target_w, kind="sqmean"):
    real = R2.pick_order

    def pick(dev, w, rows=1, kind_=kind, **k):
        rec = real(dev, w, rows, kind_, **k)
        if w == target_w and kind_ == kind:
            rec = dict(rec, exact=False, match=0.9921875, max_ulp=1, name=rec["name"] + "(simulated-inexact)",
                       n=1024)
        return rec
    return pick


def test_outcome_ii_is_off_by_default_and_approx_on_request(glue_emu, monkeypatch):
    """A statistic with no exact order (simulated for q_norm's width) turns q_path off unless glue_approx; with it
    the cluster runs as `approx`, reported (ii) in describe() and the [glue] line."""
    from runtime.static_step import StaticStep

    model, pool, _ = glue_emu("mini4")
    a0 = model.layers[0].attn
    QL = a0.q_lora_rank
    same = {"q_path"} | ({"kv_path"} if a0.head_dim == QL else set())       # (mini4: both 128 wide)
    monkeypatch.setattr(R2, "pick_order", _fake_inexact(QL))
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    d = st.glue.describe()
    for c in same:
        assert d["clusters"][c].startswith("off: (ii) approx only") and c not in d["on"]
        assert d["r2"]["numerics"][c]["outcome"].startswith("(ii) fused-approx")
    assert d["clusters"]["hc_norm"] == "on" and d["clusters"]["router"] == "on" and not d["approx_on"]
    st.detach()
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    glue_opts={"approx": True}, _allow_cpu=True)
    d = st.glue.describe()
    assert set(d["approx_on"]) == same and same <= set(d["on"]) and d["clusters"]["q_path"] == "approx"
    assert d["r2"]["outcome"]["q_path"].startswith("(ii)") and d["r2"]["approx"] is True
    line = st.glue.glue_line()
    ap_field = line.split("; approx ")[1].split(";")[0].split(",")
    assert line.startswith("[glue] fused: on ") and set(ap_field) == same and "q_path (ii)" in line
    json.dumps(d)
    # (the simulated record still holds CPU torch's own order, so the approx step equals ref here)
    assert _equal_to_ref(model, pool, st, "dg-masked", "gemv", list(range(50, 60)))


def test_self_check_turns_broken_r2_kernels_off(glue_emu, monkeypatch):
    """A norm row with a perturbed weight (x 1.0625: visible after the fp8 QDQ) and a router post that skips the
    ascending sort fail their end-to-end self-checks: q_path, kv_path and router go off with the reason; the step
    stays equal to ref."""
    from runtime.static_step import StaticStep

    real_nr = R2.launch_norm_row
    monkeypatch.setattr(R2, "launch_norm_row", lambda x, nw, out, **k: real_nr(
        x, (nw.float() * 1.0625).to(nw.dtype), out, **k))
    real_rp = R2.launch_router_post

    def unsorted(idx, sc, ids_out, w_out, **k):
        real_rp(idx, sc, ids_out, w_out, **k)
        ids_out.copy_(idx)                           # topk order, not ascending
    monkeypatch.setattr(R2, "launch_router_post", unsorted)
    model, pool, _ = glue_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    d = st.glue.describe()
    for c in ("q_path", "kv_path", "router"):
        assert d["clusters"][c].startswith("off: self-check"), (c, d["clusters"][c])
        assert d["calibration"][f"verify_{c}"] != "equal"
    assert d["clusters"]["hc_norm"] == "on" and d["clusters"]["engram"] == "on"
    assert _equal_to_ref(model, pool, st, "dg-masked", "gemv", list(range(70, 85)))


def test_self_check_hc_norm_off_takes_mix_stat_with_it(glue_emu, monkeypatch):
    from runtime.static_step import StaticStep

    real = R2.launch_hc_norm
    monkeypatch.setattr(R2, "launch_hc_norm", lambda *a, hc_mode, **k: real(*a, hc_mode=(hc_mode + 1) % 3, **k))
    model, pool, _ = glue_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    cl = st.glue.describe()["clusters"]
    assert cl["hc_norm"].startswith("off: self-check") and cl["mix_stat"].startswith("off")
    assert cl["q_path"] == "on"
    assert _equal_to_ref(model, pool, st, "dg-masked", "gemv", list(range(90, 100)))


# ---------------------------------------------------------------------------------------- Generator / first_light
class FakeStatic:
    built = []

    def __init__(self, model, moe_impl="ref", fp8_impl="ref", glue_impl="ref", glue_opts=None):
        self.model, self.moe_impl, self.fp8_impl, self.glue_impl = model, moe_impl, fp8_impl, glue_impl
        self.glue_opts = glue_opts
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures, self.glue = {}, [], None
        FakeStatic.built.append((moe_impl, fp8_impl, glue_impl, bool((glue_opts or {}).get("approx"))))

    @property
    def captured(self):
        return bool(self.graphs)

    def capture(self, fills=False):
        self.captures.append(bool(fills))
        self.graphs = {0: "g_even", 1: "g_odd"}
        return {"fills": fills}

    def step(self, token, pos, graph):
        t = int(token.reshape(-1)[0]) if isinstance(token, torch.Tensor) else int(token)
        ids, logits, _ = self.model(torch.tensor([[t]], dtype=torch.long), pos)
        if self.glue_opts and self.glue_opts.get("approx"):     # an approx cluster: a tiny logit offset
            logits = logits.clone()
            logits[0, 5] += 1e-4
            ids = logits.argmax(dim=-1)
        self.tok.copy_(ids.view(1, 1))
        return logits

    def detach(self):
        self.graphs = {}

    def cold_counts(self):
        return None


def _gen(**kw):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStatic.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], static_factory=FakeStatic,
                     **kw)


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light_r2", os.path.join(REPO, "bench",
                                                                                      "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_glue_approx_plumbing():
    gen = _gen(mode="static", glue_impl="fused")
    gen.greedy([5, 9, 12, 3], 3)
    assert gen.glue_approx is False and gen.last_meta["glue_approx"] is False
    old = gen.static
    gen.set_glue_approx(False)
    assert gen.static is old
    gen.set_glue_approx(True)
    assert gen.static is None
    gen.greedy([5, 9, 12, 3], 3)
    assert gen.last_meta["glue_approx"] is True and gen.static.glue_opts == {"approx": True}
    assert FakeStatic.built == [("ref", "ref", "fused", False), ("ref", "ref", "fused", True)]


def test_parity_levels_with_glue_approx_reports_a_drift_row():
    fl = _first_light()
    with open(FIXTURE) as f:
        rows = [json.loads(l) for l in f if l.strip()][:3]
    gen = _gen(mode="graph", glue_impl="fused", glue_approx=True)
    out = []
    res = fl.parity_levels(gen, rows, steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    row = by[("off", "fused glue +approx vs ref glue (same impls)")]
    assert row["gate"] is False and row["level"] == "G" and res["glue_approx"] is True
    assert res["glue_drift"] is not None and res["glue_drift"]["positions_compared"] > 0
    assert res["verdict"] == "PASS"                     # the approx row is reported, not gated
    assert any("drift fused glue +approx vs ref glue" in l for l in out)
    assert any("glue impl fused+approx" in l for l in out)
    assert gen.glue_approx is True and gen.glue_impl == "fused"


def test_first_light_glue_approx_flag():
    fl = _first_light()
    assert fl.build_parser().parse_args(["--model-dir", "x"]).glue_approx is False
    a = fl.build_parser().parse_args(["--model-dir", "x", "--glue-impl", "fused", "--glue-approx"])
    assert a.glue_approx is True
    lines = fl.summary_lines({"date": "d", "env": {}, "errors": [], "generator": {"mode": "graph", "glue_impl": "fused",
                                                                                   "glue_approx": True}}, "r.json")
    assert lines[0].startswith("FIRST LIGHT d | mode graph glue fused+approx")
