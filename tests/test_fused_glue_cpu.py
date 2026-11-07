"""Mac CPU tests of T9b-2 round 1 (`--glue-impl fused`, runtime/fused_glue.py). `.venv/bin/python -m pytest -q
tests/test_fused_glue_cpu.py`

Three layers, none of which needs a GPU:
  1. the torch mirrors (what the wrappers run on CPU) == the unfused reference op sequences they replace, on random
     data with clamp edges, e8m0 scale boundaries (amax at 448 * 2^k and its neighbours), exact zeros, and inputs
     crafted so a rounding-order difference would show (RoPE cancellation, the hc 4-term sum);
  2. the real `@triton.jit` kernel bodies (fused_glue.py's six, the GEMV's RoPE epilogue in both store paths)
     executed program by program through tests/triton_shim.py == the mirrors, bit for bit -- index math, masks,
     constexpr branches, rounding points. Not checked here: Triton compilation, FMA contraction on the device,
     libdevice (tests/gpu/test_fused_glue_gpu.py and the build-time self-check do that on the box). The one-launch
     gather (static_kernels.t9b2_gather_tables) dereferences raw device addresses, which the shim cannot model; its
     semantics are tested through the address-registry emulation (test_static_emulated_cpu._fake_static_kernels)
     and on the GPU;
  3. StaticStep(glue_impl="fused") == glue_impl "ref" logits at every position on the emulated mini checkpoints for
     every moe/fp8 impl pair, both prefill parities; $DSV41_GLUE_DISABLE; the self-check turning a broken cluster
     off; the Generator / first_light plumbing.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

import triton_shim
from runtime import fused_glue as FG
from test_gemv_cpu import _ref_run, _static_run, emu, fp8_gemm_blocks, gemv_emu  # noqa: F401 (fixtures)
from test_static_emulated_cpu import act_quant as _emu_act_quant

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def _cpu_default_device():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


def _ref_aq(x, block_size=32, scale_fmt="ue8m0", scale_dtype=E8, inplace=False):
    """The emulated reference act_quant (tests/test_static_emulated_cpu.act_quant: amax.clamp_min(1e-4) / 448, then
    2^ceil(log2)), with e8m0 scales as kernel.py returns them."""
    r = _emu_act_quant(x, block_size, scale_fmt, scale_dtype, inplace)
    if inplace or scale_dtype != E8:
        return r
    q, s = r
    return q, s.to(E8)


class _R:
    """The reference globals FusedGlue reads (model.py: fp8_block_size 32, scale_fmt ue8m0, scale_dtype e8m0)."""
    fp8_block_size, scale_fmt, scale_dtype = 32, "ue8m0", E8
    act_quant = staticmethod(_ref_aq)


@pytest.fixture(scope="module")
def tabs():
    return FG.scale_code_table(_R, "cpu"), FG.silu_table("cpu", 10.0), FG.silu_table("cpu", 0.0)


def _u8(t):
    return t.view(torch.uint8) if t.dtype in (FP8, E8, torch.uint8) else t.view(torch.int16 if t.element_size() == 2
                                                                                 else torch.int32)


def _eq(a, b):
    return a.shape == b.shape and torch.equal(_u8(a.contiguous()), _u8(b.contiguous()))


def _act(g, rows, K, lo=-6, hi=7):
    return FG._rand_act(g, rows, K, lo, hi)


# ================================================================================ 1. exact helpers + mirrors
def test_fma_f32_is_one_rounding():
    """fma_f32 == the correctly rounded a*b + c (Fraction arithmetic), incl. halfway ties and cancellation."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(4000, generator=g) * torch.exp2(torch.randint(-20, 20, (4000,), generator=g).float())
    b = torch.randn(4000, generator=g)
    c = -(a * b) + torch.randn(4000, generator=g) * 1e-6                       # near-total cancellation
    c[:1000] = torch.randn(1000, generator=g)
    got = FG.fma_f32(a, b, c)
    bad = 0
    for i in range(0, 4000, 7):
        exact = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        r = torch.tensor(float(exact), dtype=torch.float64).float()            # float(Fraction) is RN to double
        lo, hi = torch.nextafter(r, torch.tensor(-float("inf"))), torch.nextafter(r, torch.tensor(float("inf")))
        best = min((r, lo, hi), key=lambda v: (abs(Fraction(float(v)) - exact),
                                                int(v.view(torch.int32)) & 1))  # nearest, ties to even
        bad += int(not torch.equal(got[i], best))
    assert bad == 0
    # exact halfway in fp32: 1 + 2^-24 rounds to even (1.0); 1 + (2^-24 + 2^-47) is just above halfway: rounds up
    one = torch.tensor([1.0])
    h = torch.tensor([2.0 ** -24])
    assert float(FG.fma_f32(h, one, one)) == 1.0
    above = torch.tensor([2.0 ** -24 + 2.0 ** -47])
    assert float(above) == 2.0 ** -24 + 2.0 ** -47                             # representable in fp32
    assert float(FG.fma_f32(above, one, one)) == float(torch.nextafter(one, torch.tensor([2.0])))
    assert float(above * one + one) == float(torch.nextafter(one, torch.tensor([2.0])))
    # where one rounding and two differ: a*b rounds, then the add rounds again
    a2, b2 = torch.tensor([1.0 + 2.0 ** -12]), torch.tensor([1.0 + 2.0 ** -12])   # a*b = 1 + 2^-11 + 2^-24
    c2 = torch.tensor([-1.0])
    assert float(FG.fma_f32(a2, b2, c2)) == 2.0 ** -11 + 2.0 ** -24 and float(a2 * b2 + c2) == 2.0 ** -11


def test_e8m0_decode_and_pow2():
    codes = torch.arange(256, dtype=torch.int32)
    ref = codes.to(torch.uint8).view(E8).float()
    got = FG.e8m0_decode(codes)
    assert torch.equal(got[:255].view(torch.int32), ref[:255].view(torch.int32)) and torch.isnan(got[255])
    c = torch.arange(1, 254)
    assert torch.equal(FG.pow2_code(c), torch.exp2(c.float() - 127)) and torch.equal(FG.inv_pow2_code(c) *
                                                                                     FG.pow2_code(c), torch.ones(253))


def test_scale_table_matches_reference_on_every_bf16_amax(tabs):
    code_tab = tabs[0]
    v = FG.bf16_bits_all()
    fin = torch.isfinite(v.float())
    x = torch.zeros(65536, 32, dtype=BF)
    x[:, 0] = torch.where(fin, v, torch.zeros_like(v))
    _, s = _ref_aq(x)
    assert torch.equal(code_tab, s.view(torch.uint8).reshape(-1))
    assert 1 <= int(code_tab[:0x7F80].min()) and int(code_tab[:0x7F80].max()) <= 253


def test_act_quant_and_qdq_mirror_equal_reference_all_bf16_values(tabs):
    code_tab = tabs[0]
    g = torch.Generator().manual_seed(2)
    allb = FG.bf16_bits_all()
    fin = allb[torch.isfinite(allb.float())]
    n = fin.numel() // 32 * 32
    for x in (fin[:n].view(-1, 32), fin[:n][torch.argsort(fin[:n].float().abs())].view(-1, 32), _act(g, 64, 1024)):
        q_ref, s_ref = _ref_aq(x.contiguous())
        q, code = FG.act_quant_torch(x.float(), code_tab)
        assert _eq(q, q_ref) and torch.equal(code.to(torch.uint8), s_ref.view(torch.uint8))
        y_ref = _ref_aq(x.clone(), inplace=True)
        assert _eq(FG.qdq_torch(x.float(), code_tab), y_ref)


def _ref_swiglu(gate, up, w, lim):
    """model.py:845-853 glue on bf16 GEMM outputs."""
    gate, up = gate.float(), up.float()
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    if w is not None:
        h = w.view(-1, 1) * h
    return h.to(BF)


def _gate_up(g, G, I, lim):
    v = _act(g, 2 * G, I, lo=-3, hi=7).view(2, G, I)
    if lim > 0:
        lb = torch.tensor([lim], dtype=BF)
        up_ulp = float(torch.nextafter(lb.float(), torch.tensor([2.0 * lim])).to(BF))     # may round back to lim
        nxt = float((lb.view(torch.int16) + 1).view(BF))                                   # next bf16 above lim
        prv = float((lb.view(torch.int16) - 1).view(BF))                                   # next bf16 below lim
        for j, val in enumerate((lim, -lim, lim * 1.5, -lim * 1.5, nxt, -nxt, prv, -prv)):
            v[:, :, j] = val
        assert nxt > lim > prv and up_ulp >= lim
    v[:, :, 8] = 0.0
    v[:, :, 9] = -0.0
    return v[0], v[1]


@pytest.mark.parametrize("lim,with_w", [(10.0, True), (10.0, False), (0.0, True), (7.0, False)])
def test_swiglu_mirror_equals_reference_glue(lim, with_w):
    g = torch.Generator().manual_seed(3)
    silu_tab = FG.silu_table("cpu", lim)
    code_tab = FG.scale_code_table(_R, "cpu")
    for _ in range(4):
        gate, up = _gate_up(g, 4, 256, lim)
        w = (torch.rand(4, generator=g) * 2).float() if with_w else None
        h_ref = _ref_swiglu(gate, up, w, lim)
        h = FG.swiglu_torch(gate, up, w, silu_tab, lim)
        assert _eq(h.to(BF), h_ref)
        q_ref, s_ref = _ref_aq(h_ref)
        q, code = FG.act_quant_torch(h, code_tab)
        assert _eq(q, q_ref) and torch.equal(code.to(torch.uint8), s_ref.view(torch.uint8))


def test_silu_table_is_torch_silu_bitwise():
    t = FG.silu_table("cpu", 10.0)
    v = FG.bf16_bits_all().float()
    assert _eq(t[:65536], F.silu(v)) and _eq(t[65536:], F.silu(torch.tensor([10.0])))


def test_combine_mirror_is_slot_order_fp32():
    g = torch.Generator().manual_seed(4)
    for k in (1, 3, 6):
        rows = _act(g, k, 512, -8, 8)
        sh = _act(g, 1, 512, -8, 8)
        y = torch.zeros(1, 512)
        for j in range(k):
            y += rows[j]
        y += sh
        assert _eq(FG.combine_torch(rows, sh), y.type_as(sh).view(-1))


def _hc_ref(x, res, post, comb):
    """Block.hc_post (model.py:964-968)."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)
    return y.type_as(x)


def _hc_data(g, D, HC=4):
    x = _act(g, 1, D).view(1, 1, D)
    res = _act(g, HC, D).view(1, 1, HC, D)
    post = (torch.rand(1, 1, HC, generator=g) * 2).float()
    comb = (torch.randn(1, 1, HC, HC, generator=g) * torch.exp2(torch.randint(-4, 5, (1, 1, HC, HC), generator=g)
                                                                   .float())).float()
    big = (torch.randn(D, generator=g) * torch.exp2(torch.randint(10, 17, (D,), generator=g).float())).to(BF)
    cols = torch.arange(D) % 2 == 0
    res[0, 0, 0, cols] = big[cols]
    res[0, 0, 2, cols] = big[cols]
    comb[0, 0, 2, :] = -comb[0, 0, 0, :]
    return x, res, post, comb


def test_hc_post_modes_are_distinct_and_cpu_torch_is_mode_0():
    g = torch.Generator().manual_seed(5)
    D = 512
    mism = {m: 0 for m in FG.HC_SUM_MODES}
    for _ in range(4):
        x, res, post, comb = _hc_data(g, D)
        ref = _hc_ref(x, res, post, comb).view(4, D)
        outs = {m: FG.hc_post_torch(x.view(D), res.view(4, D), post.view(4), comb.view(4, 4), m)
                for m in FG.HC_SUM_MODES}
        for m in FG.HC_SUM_MODES:
            mism[m] += int((outs[m].view(torch.int16) != ref.view(torch.int16)).sum())
        assert not _eq(outs[0], outs[1]) and not _eq(outs[0], outs[2]) and not _eq(outs[1], outs[2])
    assert mism[0] == 0 and mism[1] > 0 and mism[2] > 0, mism


def _apply_rotary(x, freqs_cis, inverse=False):
    """model.py apply_rotary_emb (in place on x [..., rd] bf16): complex64 multiply, copy back to bf16."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(xc)
    return y


def _freqs(rd, pos, base=10000.0):
    f = 1.0 / (base ** (torch.arange(0, rd, 2).float() / rd))
    return torch.polar(torch.ones(rd // 2), torch.outer(torch.tensor([float(pos)]), f))  # [1, rd/2] complex64


@pytest.mark.parametrize("inverse", [False, True])
def test_rope_mirror_modes_and_cpu_complex_mul(inverse):
    """Each of the three contraction forms is distinguishable on cancellation-crafted pairs, and the mirror with the
    form torch's CPU complex multiply uses equals it bit for bit (the device's form is picked by the self-check)."""
    g = torch.Generator().manual_seed(6)
    H, hd, rd = 4, 64, 16
    mism = {(p, m): 0 for p in ("re", "im") for m in FG.ROPE_MODES}
    for pos in (1, 7, 300, 4000):
        f = _freqs(rd, pos)
        fr = torch.view_as_real(f).reshape(-1, 2).float()
        x = _act(g, 1, H * hd).view(1, 1, H, hd)
        ref = x.clone()
        _apply_rotary(ref[..., -rd:], f, inverse)
        v = x[..., -rd:].float().reshape(-1, 2)
        c = fr[:, 0].repeat(H)
        d = fr[:, 1].repeat(H) * (-1 if inverse else 1)
        rr = ref[..., -rd:].reshape(-1, 2)
        for m in FG.ROPE_MODES:
            mism[("re", m)] += int((FG.rope_re(v[:, 0], v[:, 1], c, d, m).to(BF).view(torch.int16)
                                    != rr[:, 0].view(torch.int16)).sum())
            mism[("im", m)] += int((FG.rope_im(v[:, 0], v[:, 1], c, d, m).to(BF).view(torch.int16)
                                    != rr[:, 1].view(torch.int16)).sum())
    ok_re = [m for m in FG.ROPE_MODES if mism[("re", m)] == 0]
    ok_im = [m for m in FG.ROPE_MODES if mism[("im", m)] == 0]
    assert ok_re and ok_im, mism
    # the whole-row mirror with that form == apply_rotary_emb (non-rope columns untouched)
    f = _freqs(rd, 123)
    x = _act(g, 1, H * hd).view(1, 1, H, hd)
    ref = x.clone()
    _apply_rotary(ref[..., -rd:], f, inverse)
    got = FG.rope_rows_torch(x.view(-1), torch.view_as_real(f).reshape(-1, 2).float(), hd, rd, (ok_re[0], ok_im[0]),
                             inverse)
    assert _eq(got.view(1, 1, H, hd), ref)


# ================================================================================ 2. kernel bodies under the shim
@pytest.fixture(scope="module")
def K():
    """runtime/fused_glue.py imported with the shim as `triton` (a private copy), wrappers set to launch kernels.
    $FG_SHIM_MODULE points it at another file (mutation checks of this test's sensitivity)."""
    mod = triton_shim.load_module_under_shim(os.environ.get("FG_SHIM_MODULE")
                                             or os.path.join(REPO, "runtime", "fused_glue.py"), "fused_glue_under_shim")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KG():
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "gemv_under_shim2")


def _launches(fn):
    return len(fn.launches)


@pytest.mark.parametrize("K_,groups,f32", [(256, 1, False), (288, 3, True), (1024, 2, False), (96, 6, True)])
def test_kernel_act_quant_bcast(K, tabs, K_, groups, f32):
    """KB not a multiple of the 8-block program (288 = 9 blocks, 96 = 3), broadcast to G rows, both scale forms."""
    g = torch.Generator().manual_seed(10 + K_)
    x = _act(g, 1, K_)
    q = torch.full((groups, K_), float("nan"), dtype=torch.float32).to(FP8)
    s = torch.full((groups, K_ // 32), -1.0) if f32 else torch.full((groups, K_ // 32), 77, dtype=torch.uint8).view(E8)
    n0 = _launches(K.t9b2_act_quant_bcast)
    K.launch_act_quant_bcast(x, q, s, tabs[0], groups)
    assert _launches(K.t9b2_act_quant_bcast) == n0 + 1
    assert K.t9b2_act_quant_bcast.launches[-1]["enable_fp_fusion"] is False
    q_ref, s_ref = _ref_aq(x)
    assert _eq(q, q_ref.expand(groups, -1))
    if f32:
        assert torch.equal(s, s_ref.float().expand(groups, -1))
    else:
        assert _eq(s, s_ref.expand(groups, -1))


@pytest.mark.parametrize("lim,with_w,f32", [(10.0, True, True), (10.0, False, False), (0.0, True, True),
                                            (7.0, False, True)])
def test_kernel_swiglu_quant(K, lim, with_w, f32):
    g = torch.Generator().manual_seed(20)
    G_, I = 3, 288
    silu_tab, code_tab = FG.silu_table("cpu", lim), FG.scale_code_table(_R, "cpu")
    gate, up = _gate_up(g, G_, I, lim)
    gbuf = torch.zeros(G_, 2, I, dtype=BF)               # row views with a non-unit row stride (DeepGEMM's d[:, 0])
    ubuf = torch.zeros(G_, 2, I, dtype=BF)
    gbuf[:, 0], ubuf[:, 0] = gate, up
    w = (torch.rand(G_, generator=g) * 2).float() if with_w else None
    qb = torch.zeros(G_, 2, I, dtype=FP8)
    sb = torch.zeros(G_, 2, I // 32) if f32 else torch.zeros(G_, 2, I // 32, dtype=torch.uint8).view(E8)
    K.launch_swiglu_quant(gbuf[:, 0], ubuf[:, 0], w, qb[:, 0], sb[:, 0], silu_tab, code_tab, lim)
    h_ref = _ref_swiglu(gate, up, w, lim)
    q_ref, s_ref = _ref_aq(h_ref)
    assert _eq(qb[:, 0], q_ref) and torch.equal(qb[:, 1].view(torch.uint8), torch.zeros(G_, I, dtype=torch.uint8))
    if f32:
        assert torch.equal(sb[:, 0], s_ref.float())
    else:
        assert _eq(sb[:, 0], s_ref)


def test_kernel_moe_combine(K):
    g = torch.Generator().manual_seed(30)
    for k, N in ((3, 1100), (6, 512), (1, 64)):
        rows = torch.zeros(k, 2, N, dtype=BF)
        rows[:, 0] = _act(g, k, N, -8, 8) if N % 32 == 0 else _act(g, k, N // 32 * 32 + 32, -8, 8)[:, :N]
        sh = _act(g, 1, N // 32 * 32 + 32, -8, 8)[0, :N].contiguous()
        out = torch.full((N,), float("nan"), dtype=BF)
        K.launch_moe_combine(rows[:, 0], sh, out)
        assert _eq(out, FG.combine_torch(rows[:, 0], sh))


@pytest.mark.parametrize("mode", FG.HC_SUM_MODES)
def test_kernel_hc_post(K, mode):
    g = torch.Generator().manual_seed(40 + mode)
    D = 1536                                            # 1.5 programs of 1024: the tail mask
    x, res, post, comb = _hc_data(g, D)
    out = torch.full((4, D), float("nan"), dtype=BF)
    K.launch_hc_post(x.view(D), res.view(4, D), post.view(4), comb.view(4, 4).contiguous(), out, mode)
    assert _eq(out, FG.hc_post_torch(x.view(D), res.view(4, D), post.view(4), comb.view(4, 4), mode))
    if mode == 0:
        assert _eq(out.view(1, 1, 4, D), _hc_ref(x, res, post, comb))


@pytest.mark.parametrize("hd,rd,H", [(64, 16, 4), (512, 64, 2), (32, 0, 3), (96, 32, 3)])
def test_kernel_o_rope_qdq(K, tabs, hd, rd, H):
    g = torch.Generator().manual_seed(50 + hd)
    f = _freqs(max(rd, 2), 77)
    fr = torch.view_as_real(f).reshape(-1, 2).float().contiguous() if rd else torch.zeros(1, 2)
    for modes in ((0, 0), (1, 1), (2, 1), (1, 2)):
        x = _act(g, 1, H * hd).view(-1)
        y = x.clone()
        K.launch_o_rope_qdq(y, fr, tabs[0], hd, rd, modes)
        assert _eq(y, FG.qdq_torch(FG.rope_rows_torch(x, fr, hd, rd, modes, True).float(), tabs[0])
                   if rd else FG.qdq_torch(x.float(), tabs[0]))


def test_kernel_rope_probe(K):
    g = torch.Generator().manual_seed(60)
    n = 1500
    a, b = _act(g, 1, 1504)[0, :n].float(), _act(g, 1, 1504)[0, :n].float()
    c, d = torch.rand(n, generator=g) * 2 - 1, torch.rand(n, generator=g) * 2 - 1
    got = K.rope_probe(a, b, c, d)
    want = FG.rope_probe(a, b, c, d)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    assert not torch.equal(want[0], want[1]) or not torch.equal(want[3], want[4])


def test_shim_fma_and_asm_forms():
    import numpy as np

    g = torch.Generator().manual_seed(61)
    a, b = torch.randn(3000, generator=g), torch.randn(3000, generator=g)
    c = -(a * b) + torch.randn(3000, generator=g) * 1e-5
    got = triton_shim.fma(triton_shim.T(a.numpy()), triton_shim.T(b.numpy()), triton_shim.T(c.numpy())).a
    assert torch.equal(torch.from_numpy(np.ascontiguousarray(got)), FG.fma_f32(a, b, c))


def test_gemv_rope_helpers_are_a_verbatim_copy(K, KG):
    """gemv_mxfp8.py carries its own copy of the RoPE helpers (module docstring): the source must be identical."""
    for name in ("_t9b2_mul_rn", "_t9b2_add_rn", "_t9b2_sub_rn", "_t9b2_rope_re", "_t9b2_rope_im"):
        assert inspect.getsource(getattr(K, name).fn) == inspect.getsource(getattr(KG, name).fn), name


def _gemv_operands(M, N, K_, seed):
    from test_gemv_kernel_shim_cpu import _operands
    return _operands(M, N, K_, seed)


@pytest.mark.parametrize("M,N,K_,cfg_args,hd,rd", [
    (1, 128, 256, (32, 4, 4, 2, 1, 0), 64, 16),        # split 1: the epilogue in the GEMV's own store
    (1, 192, 640, (64, 2, 4, 2, 4, 0), 96, 32),        # split 4: the epilogue in the reduce (pair partner sums)
    (2, 256, 384, (32, 8, 8, 3, 5, 0), 128, 64),       # M 2, split 5 (3, 4 K blocks)
])
@pytest.mark.parametrize("modes", [(0, 0), (1, 1), (2, 2)])
def test_kernel_gemv_rope_epilogue(KG, M, N, K_, cfg_args, hd, rd, modes):
    """t9b_mxfp8_gemv / t9b_splitk_reduce with ROPE == the same kernel without it, then the RoPE mirror (the
    rotation of the bf16 output with the same contraction form)."""
    xq, xs, w, s = _gemv_operands(M, N, K_, 70 + N)
    cfg = KG.GemvConfig(*cfg_args)
    p = KG.launch_plan(cfg, M, N, K_)
    ws = torch.zeros(p["NSPLIT"], M, N) if p["SPLIT"] else None
    plain = torch.full((M, N), float("nan"), dtype=BF)
    KG._launch(xq, xs, w, s, plain, cfg, ws if ws is not None else plain)
    fr = torch.view_as_real(_freqs(rd, 999)).reshape(-1, 2).float().contiguous()
    rot = torch.full((M, N), float("nan"), dtype=BF)
    ws2 = torch.zeros_like(ws) if ws is not None else None
    KG._launch(xq, xs, w, s, rot, cfg, ws2 if ws2 is not None else rot, rope=(fr, hd, rd, modes))
    want = FG.rope_rows_torch(plain, fr, hd, rd, modes, inverse=False)
    assert _eq(rot, want)
    assert not torch.isnan(rot.float()).any()


def test_gemv_rope_cpu_path_and_contract():
    from runtime import gemv_mxfp8 as G

    xq, xs, w, s = _gemv_operands(1, 128, 256, 5)
    fr = torch.view_as_real(_freqs(16, 9)).reshape(-1, 2).float().contiguous()
    plain = G.linear_fp8(xq, xs, w, s)
    got = G.linear_fp8_rope(xq, xs, w, s, fr, 64, 16, (1, 1))
    assert _eq(got, FG.rope_rows_torch(plain, fr, 64, 16, (1, 1), False))
    out = torch.empty(1, 128, dtype=BF)
    for bad in ((fr, 64, 17, (0, 0)), (fr, 60, 16, (0, 0)), (fr[:4], 64, 16, (0, 0)), (fr, 64, 16, (3, 0)),
                (fr.double(), 64, 16, (0, 0))):
        with pytest.raises(ValueError):
            G.mxfp8_gemv(xq, xs, w, s, out, rope=bad)


# ================================================================================ 3. StaticStep / plumbing
@pytest.fixture
def glue_emu(gemv_emu, monkeypatch):
    """gemv_emu (block-faithful fp8_gemm + e8m0 act_quant both sides) + the fake DeepGEMM."""
    from test_dg_moe_cpu import _fake_deep_gemm

    monkeypatch.setitem(sys.modules, "deep_gemm", _fake_deep_gemm())
    monkeypatch.delenv(FG.DISABLE_ENV, raising=False)
    return gemv_emu


def _expected_status(moe, fp8):
    on = {"gather", "swiglu_shared", "hc_post", "o_rope_qdq"}
    if moe == "dg-masked":
        on |= {"dg_act", "swiglu_routed", "moe_combine"}
    if fp8 == "gemv":
        on.add("q_rope_epilogue")
    # round 2 (runtime/fused_glue_r2.py): every cluster applies to the mini checkpoints on CPU (torch's own
    # reduction order: outcome (i)); tests/test_fused_glue_r2_cpu.py covers them
    on |= {"hc_norm", "mix_stat", "kv_path", "q_path", "router", "engram", "indexer_post"}
    # T9b-4 parts (fused_glue.R4_PARTS; tests/test_t9b4_cpu.py covers them): dg_sf_pack is dg-inplace only
    on.add("hc_split")
    if fp8 == "gemv":
        on |= {"gemv_pair", "reduce_fold"}
    return on


@pytest.mark.parametrize("kind", ["mini", "mini4"])
@pytest.mark.parametrize("moe,fp8", [("ref", "ref"), ("dg-masked", "ref"), ("ref", "gemv"), ("dg-masked", "gemv")])
def test_fused_static_step_equals_ref_glue(glue_emu, kind, moe, fp8):
    from runtime import patches
    from runtime.static_step import StaticStep

    model, pool, fk = glue_emu(kind)
    st = StaticStep(model, pool, max_pos=127, moe_impl=moe, fp8_impl=fp8, glue_impl="fused", _allow_cpu=True)
    d = st.glue.describe()
    assert set(d["on"]) == _expected_status(moe, fp8), d["clusters"]
    assert d["clusters"]["concat_free"].startswith("n/a")
    cal = d["calibration"]
    assert cal["act_quant"] == "equal" and all(v == "equal" for k, v in cal.items() if k.startswith("verify_"))
    assert d["modes"]["hc_sum"] == 0 and None not in d["modes"]["rope_fwd"] + d["modes"]["rope_inv"]
    json.dumps(d)
    g = torch.Generator().manual_seed(11)
    runs = []
    for L in (3, 16):                                    # both prefill parities; 6 decode steps
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        n_tab0 = fk.calls["gather_tables"]
        patches.fp8_gemv_stats(reset=True)
        got = _static_run(model, st, ids, 6)
        routed = sum(patches.fp8_gemv_stats()["routed"].values())
        assert fk.calls["gather_tables"] - n_tab0 == 6 * len(model.layers)
        runs.append((ids, got, routed))
    st.detach()
    ref_st = StaticStep(model, pool, max_pos=127, moe_impl=moe, fp8_impl=fp8, _allow_cpu=True)
    assert ref_st.glue is None
    for ids, got, routed in runs:
        patches.fp8_gemv_stats(reset=True)
        want = _static_run(model, ref_st, ids, 6)
        bad = [i for i, (a, b) in enumerate(zip(got[0], want[0])) if not torch.equal(a, b)]
        assert not bad and got[1] == want[1], (kind, moe, fp8, bad)
        assert routed == sum(patches.fp8_gemv_stats()["routed"].values())   # same GEMV routing, same counts
    ref_st.detach()


def test_glue_disable_env_and_unknown_name(glue_emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, fk = glue_emu("mini4")
    monkeypatch.setenv(FG.DISABLE_ENV, "gather, hc_post,q_rope_epilogue")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    cl = st.glue.describe()["clusters"]
    assert cl["gather"] == cl["hc_post"] == cl["q_rope_epilogue"] == f"off: {FG.DISABLE_ENV}"
    assert cl["moe_combine"] == "on"
    ids = list(range(5, 21))
    n0, t0 = fk.calls["gather"], fk.calls["gather_tables"]
    got = _static_run(model, st, ids, 4)
    assert fk.calls["gather_tables"] == t0 and fk.calls["gather"] - n0 == 4 * 6 * len(model.layers)
    st.detach()
    monkeypatch.delenv(FG.DISABLE_ENV)
    ref = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    want = _static_run(model, ref, ids, 4)
    assert all(torch.equal(a, b) for a, b in zip(got[0], want[0])) and got[1] == want[1]
    ref.detach()
    monkeypatch.setenv(FG.DISABLE_ENV, "gather,swiglu")
    with pytest.raises(ValueError, match="unknown clusters"):
        StaticStep(model, pool, max_pos=127, glue_impl="fused", _allow_cpu=True)
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=127, glue_impl="fast", _allow_cpu=True)


def test_self_check_turns_a_broken_cluster_off(glue_emu, monkeypatch):
    """A combine that rounds differently (here: bf16 adds) fails its self-check: the cluster goes off with the
    reason and the step stays on today's ops (still equal to ref)."""
    from runtime.static_step import StaticStep

    def bad_combine(rows, shared, out):
        acc = torch.zeros_like(out)
        for j in range(rows.size(0)):
            acc = acc + rows[j]
        return out.copy_(acc + shared)

    monkeypatch.setattr(FG, "launch_moe_combine", bad_combine)
    model, pool, _ = glue_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    d = st.glue.describe()
    assert d["clusters"]["moe_combine"].startswith("off: self-check") and "moe_combine" in d["unfused"]
    assert d["calibration"]["verify_moe_combine"] != "equal" and d["clusters"]["hc_post"] == "on"
    ids = list(range(40, 57))
    got = _static_run(model, st, ids, 4)
    st.detach()
    ref = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    want = _static_run(model, ref, ids, 4)
    assert all(torch.equal(a, b) for a, b in zip(got[0], want[0]))
    ref.detach()


def test_self_check_rejects_wrong_rope_and_hc_modes(glue_emu, monkeypatch):
    """If no candidate form matched (simulated: every probe form scaled by 1 + 2^-6, visible after the bf16
    rounding), the RoPE clusters go off; same for the hc sum order."""
    from runtime.static_step import StaticStep

    real_probe = FG.rope_probe
    monkeypatch.setattr(FG, "rope_probe", lambda a, b, c, d: real_probe(a, b, c, d) * (1 + 2 ** -6))
    real_hc = FG.launch_hc_post
    monkeypatch.setattr(FG, "launch_hc_post", lambda x, res, post, comb, out, mode: real_hc(
        x, res, post, comb * (1 + 2 ** -20), out, mode))
    model, pool, _ = glue_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    cl = st.glue.describe()["clusters"]
    assert cl["o_rope_qdq"].startswith("off: no RoPE") and cl["q_rope_epilogue"].startswith("off: no RoPE")
    assert cl["hc_post"].startswith("off: no 4-term") and cl["gather"] == "on"
    st.detach()


# ---------------------------------------------------------------------------------------- Generator / first_light
class FakeStaticGlue:
    """Any non-ref impl adds a fixed tiny offset to logit 5 (deterministic, eager == graph); the fused glue adds
    nothing (it is bit-identical), so the G row is all-equal."""

    built = []

    def __init__(self, model, moe_impl="ref", fp8_impl="ref", glue_impl="ref"):
        self.model, self.moe_impl, self.fp8_impl, self.glue_impl = model, moe_impl, fp8_impl, glue_impl
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures = {}, []
        self.glue = None
        FakeStaticGlue.built.append((moe_impl, fp8_impl, glue_impl))

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
        off = (1e-3 if self.moe_impl != "ref" else 0.0) + (2e-3 if self.fp8_impl != "ref" else 0.0)
        if off:
            logits = logits.clone()
            logits[0, 5] += off
            ids = logits.argmax(dim=-1)
        self.tok.copy_(ids.view(1, 1))
        return logits

    def detach(self):
        self.graphs = {}

    def cold_counts(self):
        return None


def _gen(mode="eager", moe_impl="ref", fp8_impl="ref", glue_impl="ref"):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStaticGlue.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=FakeStaticGlue, moe_impl=moe_impl, fp8_impl=fp8_impl, glue_impl=glue_impl)


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()][:n]


def test_generator_glue_impl_plumbing():
    gen = _gen(mode="static", glue_impl="fused")
    gen.greedy([5, 9, 12, 3], 4)
    assert gen.static.glue_impl == "fused" and gen.last_meta["glue_impl"] == "fused"
    old = gen.static
    gen.set_glue_impl("fused")
    assert gen.static is old
    gen.set_glue_impl("ref")
    assert gen.static is None
    gen.greedy([5, 9, 12, 3], 4)
    assert FakeStaticGlue.built == [("ref", "ref", "fused"), ("ref", "ref", "ref")]
    with pytest.raises(ValueError):
        gen.set_glue_impl("fast")


@pytest.mark.parametrize("moe,fp8", [("ref", "ref"), ("dg-masked", "gemv")])
def test_parity_levels_glue_row(moe, fp8):
    fl = _first_light()
    gen = _gen(mode="graph", moe_impl=moe, fp8_impl=fp8, glue_impl="fused")
    out = []
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    row = by[("off", "fused glue vs ref glue (same moe/fp8 impl)")]
    assert row["gate"] and row["all_equal"] and row["level"] == "G" and res["glue_impl"] == "fused"
    assert res["verdict"] == "PASS"
    assert (moe, fp8, "ref") in FakeStaticGlue.built                       # the G reference: same impls, glue ref
    if moe != "ref":
        assert ("ref", "ref", "ref") in FakeStaticGlue.built               # the drift reference is pure T8
    else:
        assert by[("off", "static-eager vs reference-eager")]["gate"]      # fused glue alone stays L0-gated
    for f in ("off", "on"):
        assert by[(f, "graph vs static-eager")]["all_equal"] and by[(f, "graph relaunch vs replay#1")]["all_equal"]
    assert (gen.moe_impl, gen.fp8_impl, gen.glue_impl, gen.mode) == (moe, fp8, "fused", "graph")
    assert gen.static.glue_impl == "fused" and gen.static.captures[-1] is False
    assert any("glue impl fused" in l for l in out)


def test_first_light_glue_flag_and_report(tmp_path, capsys):
    fl = _first_light()
    assert fl.build_parser().parse_args(["--model-dir", "x"]).glue_impl == "ref"
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--moe-impl", "dg-masked", "--fp8-impl",
                                      "gemv", "--glue-impl", "fused", "--parity-levels", "--level-steps", "2",
                                      "--fixture", FIXTURE, "--only", "prose", "--reps", "1", "--max-new", "4",
                                      "--step-cell", "6", "--stats-from", "2", "--out", str(tmp_path / "c.jsonl"),
                                      "--json", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        fl.build_parser().parse_args(["--model-dir", "x", "--glue-impl", "fast"])
    gen = _gen(mode="graph", moe_impl=a.moe_impl, fp8_impl=a.fp8_impl, glue_impl=a.glue_impl)
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["generator"]["glue_impl"] == "fused" and report["step_cell"]["glue_impl"] == "fused"
    assert report["parity_levels"]["glue_impl"] == "fused" and report["parity_levels"]["verdict"] == "PASS"
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | mode graph moe dg-masked fp8 gemv glue fused levels PASS")


def test_census_fused_path_receipt(tmp_path):
    """bench/glue_census.py on the real-structure config: the fused path removes each implemented cluster's audit
    count exactly and >= 1,800 launches in total (the orchestrator's gate)."""
    spec = importlib.util.spec_from_file_location("bench_glue_census", os.path.join(REPO, "bench", "glue_census.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "c.json"
    assert mod.main(["--json", str(out), "--paths", "static-dg-gemv,static-dg-gemv-fused-r1,static-dg-gemv-fused"]) == 0
    rep = json.loads(out.read_text())
    t = rep["t9b2"]
    assert t["after_path"] == "static-dg-gemv-fused-r1"     # R1's table measures R1 alone (R2: test_fused_glue_r2)
    for cl, r in t["per_cluster"].items():
        want = 0 if cl.startswith("#16") else r["audit_predicted"]
        assert r["removed"] == want, (cl, r)
    assert t["removed_total"] == t["removed_by_clusters"] >= 1800
    # round 2 (the same run): fused R1 -> fused R1+R2, every R2 cluster removes launches, the per-cluster regions
    # account for the whole difference, >= 2,000 more removed (the R2 gate)
    t2 = rep["t9b2_r2"]
    assert all(r["removed"] > 0 for r in t2["per_cluster"].values()), t2["per_cluster"]
    assert t2["removed_total"] == t2["removed_by_clusters"] >= 2000, t2
    assert rep["t9b2_total"]["removed"] == t["removed_total"] + t2["removed_total"]
    g = rep["paths"]["static-dg-gemv-fused"]["fused_glue"]
    assert all(g["clusters"][c] == "on" for c in FG.R2_CLUSTER_NAMES)
    assert all(v.startswith("(i)") for v in g["r2"]["outcome"].values())
