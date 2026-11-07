"""Mac CPU tests of T9b-4 (docs/HANDOFF-T9B4-RESIDUAL.md): `.venv/bin/python -m pytest -q tests/test_t9b4_cpu.py`

  1. kernel bodies through tests/triton_shim.py (the real `@triton.jit` functions, program by program):
     * lever 1: t9b4_hc_norm (many programs, redundant statistic) == t9b2r2_hc_norm (one program) == the torch
       mirror on every output, every form (attn / ffn with the DeepGEMM rows / head), ragged block tails, explicit
       reduction orders of the family, fp32 and packed DeepGEMM scales;
     * lever 2(a): t9b4_mxfp8_gemv2 (one launch, pointer pair) == two t9b_mxfp8_gemv launches, partials and bf16,
       each half with its own split partition (EVEN and ragged), M = 1 and 3, both block-partial forms;
     * lever 2(b): the folded reduce -- norm_row / shared SwiGLU reading split partials (_t9b4_ld) == the same kernel
       on t9b_splitk_reduce's bf16 output; gemv_torch_partials summed in split order == gemv_torch_f32;
     * lever 4: packed activation-SF stores (hc_norm, act_quant_bcast, SwiGLU) == the codes of the fp32 scales, laid
       out as DeepGEMM's pack_ue8m0_to_int + MN-major (dg_inplace.ref_pack_from_f32);
  2. StaticStep on the emulated mini checkpoints with real split partials on CPU (gemv_mxfp8.CPU_SPLIT_K = 3):
     fused + T9b-4 == fused without T9b-4 == ref glue (dg-masked; dg-inplace with the packed SF and the router), the
     launches the parts remove, and the self-check turning a broken part off.
Not here (box: tests/gpu/test_t9b4_gpu.py): Triton compilation, tl.dot numerics, DeepGEMM's handling of int32 SF.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest
import torch

import triton_shim
from runtime import dg_inplace as DI
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from runtime import gemv_mxfp8 as GM
from test_fused_glue_r2_cpu import _R, _act, _eq, _od, tab  # noqa: F401 (fixture)
from test_gemv_cpu import _static_run, emu, gemv_emu  # noqa: F401 (fixtures)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
R4 = FG.R4_PARTS


@pytest.fixture(autouse=True)
def _restore_dtype():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(scope="module")
def KR():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue_r2.py"), "fg2_t9b4_shim")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KF():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue.py"), "fg1_t9b4_shim")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KG():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "gemv_t9b4_shim")
    mod.KERNELS_ON_CPU = True
    return mod


def _launches(fn):
    return len(fn.launches)


# ================================================================================ 0. module contract
def test_helpers_are_verbatim_copies(KR, KF):
    for name in ("_t9b4_ld", "_t9b4_store_code", "_t9b2_bf16", "_t9b2_pow2"):
        assert inspect.getsource(getattr(KR, name).fn) == inspect.getsource(getattr(KF, name).fn), name


def test_parts_are_clusters_with_audit_names():
    assert set(R4) <= set(FG.CLUSTERS) and not set(R4) & set(FG.R2_CLUSTER_NAMES)
    assert all(c in FG.CLUSTER_AUDIT for c in R4)


# ================================================================================ 1a. lever 1 (hc_norm split)
ORDERS = [((("vec", 64, 4, True), "mul"), (("vec", 128, 4, True), "mul")),
          ((("unroll", 32, 2, False), "div"), (("vec", 64, 2, False), "div")),
          ((("vec", 8, 1, False), "mul"), (("unroll", 16, 4, True), "mul"))]


def _packed(G, align, KB):
    sf = DI.packed_act_sf(G * align, KB, "cpu")
    return DI.PackedRows(sf, align, G)


@pytest.mark.parametrize("D", [256, 96, 544, 640])
@pytest.mark.parametrize("mode", R2.HC_SUM_MODES)
@pytest.mark.parametrize("oi", range(len(ORDERS)))
def test_hc_split_equals_one_program_and_mirror(KR, tab, monkeypatch, D, mode, oi):
    HC, G = 4, 3
    monkeypatch.setattr(KR, "HC_SPLIT_BB", 2 if D in (544, 640) else 8)  # 544: 9 apply programs + 1, ragged last
    packed = (D // 32) % 4 == 0                                          # the packed SF needs K % 128 == 0
    g = torch.Generator().manual_seed(D + 10 * mode + oi)
    x = _act(g, HC, D)
    pre = torch.rand(HC, generator=g) + 1e-3
    nw = (torch.randn(D, generator=g) * 0.5 + 1.0).to(BF)
    nord, mord = (_od(*o) for o in ORDERS[oi])
    outs = {}
    for name, mod, split in (("mirror", R2, False), ("one", KR, False), ("split", KR, True)):
        o = dict(y=torch.empty(D, dtype=BF), yf=torch.empty(D), xf=torch.empty(HC * D), mrs=torch.empty(1),
                 q=torch.empty(D, dtype=FP8), s=torch.empty(D // 32, dtype=E8),
                 dq=torch.full((G, 5, D), 7.0).to(FP8), ds=torch.full((G, 5, D // 32), -1.0))
        n1, n2 = _launches(KR.t9b2r2_hc_norm), _launches(KR.t9b4_hc_norm)
        mod.launch_hc_norm(x, pre, nw, o["y"], eps=1e-6, norm_ord=nord, hc_mode=mode, code_tab=tab, yf=o["yf"],
                           xf=o["xf"], mrs=o["mrs"], mix_ord=mord, mix_eps=1e-6, q=o["q"], s=o["s"],
                           dq=o["dq"][:, 0], ds=o["ds"][:, 0], split=split)
        if packed:
            pk = _packed(G, 4, D // 32)
            mod.launch_hc_norm(x, pre, nw, torch.empty(D, dtype=BF), eps=1e-6, norm_ord=nord, hc_mode=mode,
                               code_tab=tab, dq=torch.empty(G, D, dtype=FP8), ds=pk, split=split)
            o["packed"] = pk.codes()
        if mod is KR:
            n = 2 if packed else 1
            assert _launches(KR.t9b2r2_hc_norm) - n1 == (0 if split else n)
            assert _launches(KR.t9b4_hc_norm) - n2 == (n if split else 0)
        outs[name] = o
    for k in outs["mirror"]:
        assert _eq(outs["one"][k], outs["mirror"][k]), ("one", k)
        assert _eq(outs["split"][k], outs["one"][k]), ("split", k)
    if packed:
        want = ((outs["one"]["ds"][:, 0].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        assert torch.equal(outs["split"]["packed"], want)
    grid = KR.t9b4_hc_norm.launches[-2 if packed else -1]["grid"]
    assert grid == (-(-(D // 32) // KR.HC_SPLIT_BB) + 1,)


def test_hc_split_head_form(KR):
    D, HC = 160, 4
    g = torch.Generator().manual_seed(3)
    x, pre, nw = _act(g, HC, D), torch.rand(HC, generator=g), torch.randn(D, generator=g).to(BF)
    res = []
    for split in (False, True):
        y, yf = torch.empty(D, dtype=BF), torch.empty(D)
        KR.launch_hc_norm(x, pre, nw, y, eps=1e-20, norm_ord=_od(("vec", 32, 4, True)), hc_mode=0, yf=yf,
                          split=split)
        res.append((y, yf))
    assert KR.t9b4_hc_norm.launches[-1]["grid"] == (-(-(D // 32) // KR.HC_SPLIT_BB),)   # no mix program
    assert _eq(res[0][0], res[1][0]) and _eq(res[0][1], res[1][1])


# ================================================================================ 1b. lever 2 (pair + fold)
def _ops(N, K, M, seed):
    """Operands whose 32-product block sums are exact in fp32 (test_gemv_kernel_shim_cpu._exact_fp8: exponent field
    6..8), so the shim's block-sum order cannot matter and kernel == torch partials holds bit for bit."""
    from test_gemv_kernel_shim_cpu import _exact_fp8

    g = torch.Generator().manual_seed(seed)

    def fp8(*shape):
        return _exact_fp8(*shape, g=g)

    def e8(*shape):
        return torch.randint(118, 136, shape, generator=g, dtype=torch.int32).to(torch.uint8).view(E8)

    return fp8(M, K), e8(M, K // 32), fp8(N, K), e8((N + 31) // 32, K // 32)


def test_partials_sum_to_gemv_torch():
    xq, xs, w, s = _ops(96, 320, 2, 1)
    for sk in (1, 3, 4, 10):
        p = GM.gemv_torch_partials(xq, xs, w, s, sk)
        r = GM.SplitK(p.contiguous(), p.size(0)).reduce_torch() if p.size(0) > 1 else p[0].to(BF)
        assert _eq(r, GM.gemv_torch(xq, xs, w, s, split_k=sk)), sk
        tot = torch.zeros(2, 96)
        for j in range(p.size(0)):
            tot = tot + p[j]
        assert _eq(tot, GM.gemv_torch_f32(xq, xs, w, s, sk))


PAIRS = [  # (M, NA, NB, K, cfg_a, cfg_b)
    (1, 96, 64, 320, GM.GemvConfig(32, 2, 4, 1, 3, 0), GM.GemvConfig(64, 4, 4, 1, 1, 0)),    # ragged a, b unsplit
    (3, 64, 128, 256, GM.GemvConfig(32, 4, 4, 1, 2, 0), GM.GemvConfig(32, 2, 4, 1, 4, 0)),   # both split, EVEN
    (1, 64, 32, 512, GM.GemvConfig(32, 4, 4, 1, 4, 1), GM.GemvConfig(64, 2, 4, 1, 3, 1)),    # MMA, b ragged
]


@pytest.mark.parametrize("case", range(len(PAIRS)))
def test_pair_kernel_equals_two_launches(KG, case):
    M, NA, NB, K, ca, cb = PAIRS[case]
    xq, xs, wa, sa = _ops(NA, K, M, 10 + case)
    _, _, wb, sb = _ops(NB, K, M, 20 + case)
    singles = [KG.gemv_partials(xq, xs, w, s, tag=f"single{case}{i}", config=c)
               for i, (w, s, c) in enumerate(((wa, sa, ca), (wb, sb, cb)))]
    singles = [GM.SplitK(r.ws.clone(), r.nsplit) if hasattr(r, "nsplit") else r.clone() for r in singles]
    n0 = _launches(KG.t9b4_mxfp8_gemv2)
    pa, pb = KG.gemv_pair(xq, xs, wa, sa, wb, sb, tags=(f"pa{case}", f"pb{case}"), configs=(ca, cb))
    assert _launches(KG.t9b4_mxfp8_gemv2) - n0 == 1
    pp = KG.pair_plan(KG.resolve_config(ca, NA, K), KG.resolve_config(cb, NB, K), M, NA, NB, K)
    assert KG.t9b4_mxfp8_gemv2.launches[-1]["grid"] == pp["grid"]
    for got, want, cfg, w, s in ((pa, singles[0], ca, wa, sa), (pb, singles[1], cb, wb, sb)):
        assert hasattr(got, "nsplit") == hasattr(want, "nsplit")
        if hasattr(got, "nsplit"):
            assert got.nsplit == want.nsplit and _eq(got.ws, want.ws)
            assert _eq(got.ws, GM.gemv_torch_partials(xq, xs, w, s, cfg.split_k))
            assert _eq(KG.materialize(got), GM.gemv_torch(xq, xs, w, s, split_k=cfg.split_k))
        else:
            assert _eq(got, want) and _eq(got, GM.gemv_torch(xq, xs, w, s, split_k=1))


def test_pair_plan_refuses_mixed_partial_forms():
    a, b = GM.GemvConfig(32, 4, 4, 1, 2, 0), GM.GemvConfig(32, 4, 4, 1, 2, 1)
    assert GM.pair_plan(a, b, 1, 64, 64, 256) is None
    p = GM.pair_plan(a, GM.GemvConfig(64, 2, 8, 3, 5, 0), 1, 64, 96, 320)
    assert p["TA"] == 2 and p["TB"] == 3 and p["NSA"] == 2 and p["NSB"] == 5 and p["grid"] == (5, 5)
    assert p["PERA"] == 5 and p["PERB"] == 2 and not p["EVENA"] and not p["EVENB"] and p["BLOCK_N"] == 32  # KU 4 (a's)
    p = GM.pair_plan(GM.GemvConfig(32, 2, 4, 1, 5, 0), GM.GemvConfig(64, 4, 8, 3, 3, 0), 1, 64, 96, 320)
    assert p["PERA"] == 2 and p["EVENA"] and p["PERB"] == 4 and not p["EVENB"] and p["KU"] == 2


def _splitk(g, ns, M, N):
    ws = (torch.randn(ns, M, N, generator=g) * torch.exp2(torch.randint(-6, 6, (ns, M, 1), generator=g).float()))
    ws[0, :, ::7] = -0.0                          # -0 partials: 0 + -0 = +0 (the reduce's start), kept here too
    return GM.SplitK(ws.contiguous(), ns)


@pytest.mark.parametrize("N,RD,slot,qdq,ns", [(128, 0, None, False, 3), (96, 32, 2, True, 2), (256, 64, 5, True, 4)])
def test_norm_row_folds_the_reduce(KR, KG, tab, N, RD, slot, qdq, ns):
    from test_fused_glue_r2_cpu import _freqs

    g = torch.Generator().manual_seed(N + ns)
    sk = _splitk(g, ns, 1, N)
    nw = (torch.randn(N, generator=g) * 0.5 + 1.0).to(BF)
    rope = None
    if RD:
        rope = (torch.view_as_real(_freqs(RD, 17)).reshape(-1, 2).float().contiguous(), RD, (1, 2))
    red = KG.materialize(sk).view(-1)                       # t9b_splitk_reduce under the shim
    assert _eq(red, sk.reduce_torch().view(-1))
    outs = []
    for mod, x in ((KR, sk), (KR, red), (R2, sk)):
        out = torch.zeros(8, N, dtype=BF) if slot is not None else torch.empty(N, dtype=BF)
        q, s = torch.empty(N, dtype=FP8), torch.empty(N // 32, dtype=E8)
        mod.launch_norm_row(x, nw, out, eps=1e-6, ordd=_od(("vec", 16, 4, True)), code_tab=tab,
                            slot=torch.tensor([slot]) if slot is not None else None, q=None if qdq else q,
                            s=None if qdq else s, qdq=qdq, rope=rope)
        outs.append((out, q, s) if not qdq else (out,))
    for o in outs[1:]:
        assert all(_eq(a, b) for a, b in zip(outs[0], o))
    assert KR.t9b2r2_norm_row.launches[-2]["constexprs"]["NS"] == ns


@pytest.mark.parametrize("nsg,nsu,packed", [(2, 3, False), (0, 4, True), (3, 0, False)])
def test_swiglu_folds_the_reduce_and_packs(KF, tab, nsg, nsu, packed):
    I, lim = 256, 7.0
    g = torch.Generator().manual_seed(nsg * 10 + nsu)
    silu = FG.silu_table("cpu", lim)
    gk = _splitk(g, nsg, 1, I) if nsg else None
    uk = _splitk(g, nsu, 1, I) if nsu else None
    gb = gk.reduce_torch() if gk else (torch.randn(1, I, generator=g) * 4).to(BF)
    ub = uk.reduce_torch() if uk else (torch.randn(1, I, generator=g) * 4).to(BF)
    res = []
    for gg, uu in ((gk or gb, uk or ub), (gb, ub)):
        q, s = torch.empty(1, I, dtype=FP8), torch.empty(1, I // 32, dtype=torch.float32)
        KF.launch_swiglu_quant(gg, uu, None, q, s, silu, tab, lim)
        res.append((q, s))
    assert _eq(res[0][0], res[1][0]) and _eq(res[0][1], res[1][1])
    q_m, s_m = torch.empty(1, I, dtype=FP8), torch.empty(1, I // 32, dtype=torch.float32)
    FG.launch_swiglu_quant(gk or gb, uk or ub, None, q_m, s_m, silu, tab, lim)        # the torch mirror
    assert _eq(q_m, res[1][0]) and _eq(s_m, res[1][1])
    if packed:
        pk = _packed(1, 4, I // 32)
        KF.launch_swiglu_quant(gk or gb, uk or ub, None, torch.empty(1, I, dtype=FP8), pk, silu, tab, lim)
        want = ((res[1][1].view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        assert torch.equal(pk.codes(), want)


# ================================================================================ 1c. lever 4 (packed SF)
def test_packed_act_sf_is_deepgemm_layout():
    G, align, KB = 3, 4, 16
    pk = _packed(G, align, KB)
    sf = pk.sf
    assert DI.is_mn_major_packed(sf.unsqueeze(0), 1, G * align, KB // 4) and sf.stride(1) == 12
    assert bool((DI.unpack_sf(sf).view(-1) == 127).all())                 # padding rows: 2^0 (= fp32 1.0)
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(100, 140, (G, KB), generator=g)
    for r in range(G):
        pk.store_codes_torch(r, codes[r])
    f32 = torch.ones(G * align, KB)
    f32[::align] = FG.pow2_code(codes)
    assert torch.equal(sf, DI.ref_pack_from_f32(f32)) and torch.equal(pk.codes(), codes.to(torch.uint8))


def test_act_quant_bcast_packed(KF, tab):
    K, G = 384, 3        # 12 blocks: K % 128 == 0, three words per row, AQ_BB = 8 -> a ragged second program
    x = _act(torch.Generator().manual_seed(4), 1, K)
    q32, s32 = torch.empty(G, K, dtype=FP8), torch.empty(G, K // 32)
    KF.launch_act_quant_bcast(x, q32, s32, tab, G)
    for mod in (KF, FG):
        pk = _packed(G, 8, K // 32)
        q = torch.empty(G, K, dtype=FP8)
        mod.launch_act_quant_bcast(x, q, pk, tab, G)
        assert _eq(q, q32) and torch.equal(pk.codes(), ((s32.view(torch.int32) >> 23) & 0xFF).to(torch.uint8))


# ================================================================================ 2. StaticStep (emulated)
@pytest.fixture
def glue_emu(gemv_emu, monkeypatch):
    from test_dg_moe_cpu import _fake_deep_gemm

    monkeypatch.setitem(sys.modules, "deep_gemm", _fake_deep_gemm())
    monkeypatch.delenv(FG.DISABLE_ENV, raising=False)
    monkeypatch.setattr(GM, "CPU_SPLIT_K", 3)        # real split partials on CPU: the fold path runs
    R2._ORDER_CACHE.clear()
    yield gemv_emu
    R2._ORDER_CACHE.clear()


def _count(monkeypatch, mod, names):
    calls = {n: 0 for n in names}
    for n in names:
        real = getattr(mod, n)

        def wrap(*a, _real=real, _n=n, **k):
            calls[_n] += 1
            return _real(*a, **k)
        monkeypatch.setattr(mod, n, wrap)
    return calls


def test_static_step_t9b4_equals_without_and_ref(glue_emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _ = glue_emu("mini4")
    ids = list(range(30, 47))
    kw = dict(max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    st = StaticStep(model, pool, glue_impl="fused", **kw)
    d = st.glue.describe()
    assert {"hc_split", "gemv_pair", "reduce_fold"} <= set(d["on"]) and d["clusters"]["dg_sf_pack"].startswith("n/a")
    assert all(v == "equal" for v in d["calibration"]["r4"].values()), d["calibration"]["r4"]
    seen = {"splitk_norm": 0, "splitk_swiglu": 0}
    real_nr, real_sw = R2.launch_norm_row, FG.launch_swiglu_quant

    def nr(x, *a, **k):
        seen["splitk_norm"] += hasattr(x, "nsplit")
        return real_nr(x, *a, **k)

    def sw(g_, u_, *a, **k):
        seen["splitk_swiglu"] += hasattr(g_, "nsplit") and hasattr(u_, "nsplit")
        return real_sw(g_, u_, *a, **k)
    monkeypatch.setattr(R2, "launch_norm_row", nr)
    monkeypatch.setattr(FG, "launch_swiglu_quant", sw)
    calls = _count(monkeypatch, GM, ("gemv_pair", "gemv_partials", "linear_fp8", "linear_fp8_rope"))
    got = _static_run(model, st, ids, 10)
    st.detach()
    L = len(model.layers)
    n_steps = 10
    assert seen["splitk_norm"] == 2 * L * n_steps and seen["splitk_swiglu"] == L * n_steps, seen
    assert calls["gemv_pair"] == 2 * L * n_steps and calls["gemv_partials"] == 0, calls
    without = StaticStep(model, pool, glue_impl="fused", glue_opts={"disable": list(R4)}, **kw)
    assert not set(R4) & set(without.glue.describe()["on"])
    want = _static_run(model, without, ids, 10)
    without.detach()
    ref = StaticStep(model, pool, **kw)
    ref_out = _static_run(model, ref, ids, 10)
    ref.detach()
    for a, b in ((got, want), (got, ref_out)):
        assert all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and a[1] == b[1]


def test_fold_without_pair(glue_emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _ = glue_emu("mini4")
    ids = list(range(50, 61))
    kw = dict(max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    st = StaticStep(model, pool, glue_impl="fused", glue_opts={"disable": ["gemv_pair"]}, **kw)
    calls = _count(monkeypatch, GM, ("gemv_pair", "gemv_partials"))
    got = _static_run(model, st, ids, 4)
    st.detach()
    L = len(model.layers)
    assert calls == {"gemv_pair": 0, "gemv_partials": 4 * L * 4}, calls          # wq_a, wkv, w1, w3 folded
    ref = StaticStep(model, pool, **kw)
    want = _static_run(model, ref, ids, 4)
    ref.detach()
    assert all(torch.equal(x, y) for x, y in zip(got[0], want[0])) and got[1] == want[1]


def test_self_check_turns_broken_parts_off(glue_emu, monkeypatch):
    from runtime.static_step import StaticStep

    real_p, real_h = GM.gemv_pair, R2.launch_hc_norm

    def bad_pair(*a, **k):
        ra, rb = real_p(*a, **k)
        if hasattr(ra, "nsplit"):
            ra.ws[0].mul_(1.0 + 2 ** -7)
        return ra, rb

    def bad_hc(x, pre, nw, y, **k):
        real_h(x, pre, nw, y, **k)
        if k.get("split"):
            y.mul_(1.0078125)
    monkeypatch.setattr(GM, "gemv_pair", bad_pair)
    monkeypatch.setattr(R2, "launch_hc_norm", bad_hc)
    model, pool, _ = glue_emu("mini4")
    kw = dict(max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    st = StaticStep(model, pool, glue_impl="fused", **kw)
    d = st.glue.describe()
    # the split-vs-one check runs before hc_norm's end-to-end check: a broken split costs hc_split only
    assert d["clusters"]["gemv_pair"].startswith("off: self-check"), d["clusters"]
    assert d["clusters"]["hc_split"].startswith("off: self-check"), d["clusters"]
    assert d["clusters"]["reduce_fold"] == "on" and d["clusters"]["hc_norm"] == "on"
    ids = list(range(5, 15))
    got = _static_run(model, st, ids, 4)
    st.detach()
    monkeypatch.setattr(GM, "gemv_pair", real_p)
    monkeypatch.setattr(R2, "launch_hc_norm", real_h)
    ref = StaticStep(model, pool, **kw)
    want = _static_run(model, ref, ids, 4)
    ref.detach()
    assert all(torch.equal(x, y) for x, y in zip(got[0], want[0])) and got[1] == want[1]


def test_dg_inplace_sf_pack_and_router(gemv_emu, monkeypatch):
    from runtime.static_step import StaticStep
    from test_dg_inplace_cpu import _check, _fake_dg_contig, _install_kernels

    monkeypatch.delenv(FG.DISABLE_ENV, raising=False)
    monkeypatch.setattr(GM, "CPU_SPLIT_K", 3)
    fake = _fake_dg_contig()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    model, pool, _ = gemv_emu("mini4")
    fk = _install_kernels(monkeypatch, pool)
    kw = dict(max_pos=127, moe_impl="dg-inplace", fp8_impl="gemv", _allow_cpu=True)
    st = StaticStep(model, pool, glue_impl="fused", **kw)
    d = st.glue.describe()
    assert d["clusters"]["dg_sf_pack"] == "on" and st.dg.sf_pack and d["r4"]["dg_act_sf_packed"], d["clusters"]
    assert d["calibration"]["r4"]["verify_dg_sf_pack"] == "equal"
    assert st.dg.describe()["act_sf_packed"] is True
    ids = list(range(7, 23))
    calls = _count(monkeypatch, R2, ("launch_router_post",))
    n_pk = len(fake.packed_a)
    got = _static_run(model, st, ids, 6)
    L = len(model.layers)
    assert calls["launch_router_post"] == 6 * L                    # the router now runs on dg-inplace too
    assert len(fake.packed_a) - n_pk == 3 * L * 6                  # every contiguous GEMM took the packed SF
    assert fk.calls["stage"] >= 6 * L
    b = st.dg
    for w in (1, 2):                                               # padding rows of the packed SF untouched
        assert bool((DI.unpack_sf(b.xsp[w]).view(st.topk, b.align, -1)[:, 1:] == 127).all())
    st.detach()
    without = StaticStep(model, pool, glue_impl="fused", glue_opts={"disable": list(R4)}, **kw)
    assert not without.dg.sf_pack
    _check(_static_run(model, without, ids, 6), got, "dg-inplace fused: T9b-4 on vs off")
    without.detach()
    ref = StaticStep(model, pool, **kw)
    _check(_static_run(model, ref, ids, 6), got, "dg-inplace fused + T9b-4 vs ref glue")
    ref.detach()
