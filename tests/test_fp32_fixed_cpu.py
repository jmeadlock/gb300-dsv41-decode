"""Mac CPU tests of T9b-5 (docs/HANDOFF-T9B5-FP32-FIXED.md): the fixed-order fp32 dot for the mHC mixes and the head.

  .venv/bin/python -m pytest -q tests/test_fp32_fixed_cpu.py

What is proven here (the GPU kernel's speed and its compiled arithmetic are tests/gpu/test_fp32_fixed_gpu.py's job):
  * fma_f32_torch (the torch definition's FMA) == the exactly rounded a*b + c (Python Fractions) on random,
    cancelling and fp32-midpoint cases, and == tests/triton_shim.fma (the shim's FMA) bit for bit;
  * fixed_dot_torch == the order written out longhand (per output: KB sequential chains, then the pair tree);
  * runtime/static_kernels.py t9b5_head_fp32 / t9b5_mix_fp32 executed program by program under the shim ==
    fixed_dot_torch bit for bit -- head form (bf16 x, bf16-valued fp32 W) and mixes form (fp32 x, fp32 W, * scale),
    R in {1, 2, 5, 6, 8, 9}, N not a multiple of BN, U not dividing K/KB (the tail loop), every RB / BN / U config;
  * row invariance by construction: row r of an R-row call == the same row alone, for every R and config; replay
    equal; the KB choice is a function of (site, K) only;
  * the switch: $DSV41_FP32_IMPL parsing, the glue / static-step call sites use the fixed order under `fixed` and
    cuBLAS's F.linear under `cublas` (the default).
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import triton_shim  # noqa: E402
from runtime import fused_glue_r2 as R2  # noqa: E402


@pytest.fixture(scope="module")
def SK():
    """runtime/static_kernels.py imported with the shim as triton (a private copy)."""
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"),
                                              "static_kernels_t9b5_under_shim")


def _exact_fma(a: float, b: float, c: float) -> float:
    """RN_fp32(a*b + c) from exact rationals (round half to even), as a Python float."""
    v = Fraction(a) * Fraction(b) + Fraction(c)
    lo = np.float32(float(v))                     # within an ulp; settle the neighbours exactly
    cands = {float(lo), float(np.nextafter(lo, np.float32(np.inf))), float(np.nextafter(lo, np.float32(-np.inf)))}
    best = min(cands, key=lambda f: (abs(Fraction(f) - v), int(np.float32(f).view(np.int32)) & 1))
    return best


def _f32(x) -> np.float32:
    return np.float32(x)


def test_fma_definition_is_exactly_rounded_and_equals_the_shim():
    g = torch.Generator(device="cpu").manual_seed(5)
    a = torch.randn(4000, generator=g, device="cpu")
    b = torch.randn(4000, generator=g, device="cpu")
    c = torch.randn(4000, generator=g, device="cpu")
    # cancelling cases (c ~ -a*b) and fp32-midpoint cases (a*b + c lands on a half-ulp of the fp32 result)
    c[:1000] = -(a[:1000] * b[:1000]) * (1 + torch.randn(1000, generator=g, device="cpu") * 1e-6)
    mids = []
    for i in range(200):
        base = float(np.float32(1.0 + i / 256.0))
        ulp = float(np.spacing(np.float32(base)))
        # a*b = ulp/2 + tiny (exact in fp64, not in fp32): c + a*b sits just above a midpoint
        av, bv = _f32(ulp / 2 * 3), _f32(1.0 / 3.0)
        mids.append((float(av), float(bv), base))
    a2 = torch.tensor([m[0] for m in mids], dtype=torch.float32, device="cpu")
    b2 = torch.tensor([m[1] for m in mids], dtype=torch.float32, device="cpu")
    c2 = torch.tensor([m[2] for m in mids], dtype=torch.float32, device="cpu")
    a, b, c = torch.cat([a, a2]), torch.cat([b, b2]), torch.cat([c, c2])
    got = R2.fma_f32_torch(a, b, c)
    shim = torch.from_numpy(triton_shim.fma(a.numpy(), b.numpy(), c.numpy()).a)
    assert torch.equal(got.view(torch.int32), shim.view(torch.int32))
    idx = list(range(0, 4000, 7)) + list(range(4000, 4200))
    want = torch.tensor([_exact_fma(float(a[i]), float(b[i]), float(c[i])) for i in idx], dtype=torch.float32,
                        device="cpu")
    assert torch.equal(got[idx].view(torch.int32), want.view(torch.int32))
    # and it is not just a*b + c in fp32 (the case set does show the single rounding)
    assert not torch.equal(got, a * b + c)


def _longhand(x: torch.Tensor, w: torch.Tensor, kb: int, scale=None) -> torch.Tensor:
    """The order written out per output with Python fp32 scalars (np.float32 + the exact fma)."""
    R, K = x.shape
    N = w.size(0)
    xn, wn = x.float().numpy(), w.float().numpy()
    out = np.zeros((R, N), dtype=np.float32)
    for r in range(R):
        for n in range(N):
            lanes = []
            for j in range(kb):
                acc = np.float32(0.0)
                for t in range(K // kb):
                    k = t * kb + j
                    acc = np.float32(_exact_fma(float(xn[r, k]), float(wn[n, k]), float(acc)))
                lanes.append(acc)
            while len(lanes) > 1:
                lanes = [np.float32(lanes[2 * i] + lanes[2 * i + 1]) for i in range(len(lanes) // 2)]
            v = lanes[0]
            if scale is not None:
                v = np.float32(v * np.float32(scale[r]))
            out[r, n] = v
    return torch.from_numpy(out)


def test_fixed_dot_torch_is_the_documented_order():
    g = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn(3, 32, generator=g, device="cpu")
    w = torch.randn(5, 32, generator=g, device="cpu")
    sc = torch.rand(3, generator=g, device="cpu") + 0.5
    for kb in (1, 4, 8, 32):
        got = R2.fixed_dot_torch(x, w, kb, sc)
        assert torch.equal(got.view(torch.int32), _longhand(x, w, kb, sc).view(torch.int32)), kb
    # the order is real: different KB give different bits on this data
    outs = {kb: R2.fixed_dot_torch(x, w, kb) for kb in (1, 8, 32)}
    assert not (torch.equal(outs[1], outs[8]) and torch.equal(outs[8], outs[32]))


def _case(site: str, R: int, N: int, K: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if site == "head":   # bf16 activations, bf16 checkpoint values widened to fp32 (products exact)
        x = (torch.randn(R, K, generator=g, device="cpu") * 2).to(torch.bfloat16)
        w = (torch.randn(N, K, generator=g, device="cpu") * 0.05).to(torch.bfloat16).float()
        sc = None
    else:                # the mixes: bf16-valued fp32 stream, a genuine fp32 weight, * rsqrt
        x = (torch.randn(R, K, generator=g, device="cpu") * 3).to(torch.bfloat16).float()
        w = torch.randn(N, K, generator=g, device="cpu") * 0.01
        sc = torch.rand(R, generator=g, device="cpu") + 0.25
    return x, w, sc


def _shim_dot(SK, x, w, site, kb, sc=None, cfg=None):
    out = torch.full((x.size(0), w.size(0)), float("nan"), dtype=torch.float32, device="cpu")
    SK.fixed_dot(x, w, out, kb=kb, site=site, scale=sc, cfg=cfg)
    return out


CFGS = {"head": [None, (1, 4, 2, 4, 1), (2, 8, 3, 4, 1), (8, 16, 1, 8, 1), (4, 2, 5, 4, 1)],
        "mix": [None, (1, 1, 8, 4, 1), (2, 4, 3, 4, 1), (1, 2, 1, 4, 1)]}


@pytest.mark.parametrize("site", ["head", "mix"])
def test_kernel_under_shim_equals_definition_and_is_row_invariant(SK, site):
    N, K = (40, 256) if site == "head" else (24, 512)
    kb = R2.fixed_kb(site, K)
    assert kb == (64 if site == "head" else 512)
    x9, w, sc9 = _case(site, 9, N, K, 3 if site == "head" else 4)
    singles = [_shim_dot(SK, x9[i:i + 1], w, site, kb, sc9[i:i + 1] if sc9 is not None else None) for i in range(9)]
    want = R2.fixed_dot_torch(x9, w, kb, sc9)
    for i in range(9):
        assert torch.equal(singles[i].view(torch.int32), want[i:i + 1].view(torch.int32)), i
    for R in (1, 2, 5, 6, 8, 9):
        x, s_ = x9[:R].contiguous(), (sc9[:R].contiguous() if sc9 is not None else None)
        for cfg in CFGS[site]:
            got = _shim_dot(SK, x, w, site, kb, s_, cfg)
            assert not torch.isnan(got).any(), (R, cfg)
            ref = torch.cat(singles[:R], 0)
            assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), (site, R, cfg)
    # replay
    a, b = _shim_dot(SK, x9[:6].contiguous(), w, site, kb, sc9[:6] if sc9 is not None else None), \
        _shim_dot(SK, x9[:6].contiguous(), w, site, kb, sc9[:6] if sc9 is not None else None)
    assert torch.equal(a, b)


def test_kernel_tree_over_every_kb(SK):
    """KB from 1 to 4096 (every level of _t9b5_tree), K = 2 * KB (two tiles), and K = KB * 3 with U = 2 (the tail)."""
    g = torch.Generator(device="cpu").manual_seed(21)
    for kb in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        for mult, u in ((2, 1), (3, 2)):
            K = kb * mult
            x = torch.randn(2, K, generator=g, device="cpu")
            w = torch.randn(3, K, generator=g, device="cpu")
            got = _shim_dot(SK, x, w, "mix", kb, None, (2, 4, u, 4, 1))
            assert torch.equal(got.view(torch.int32), R2.fixed_dot_torch(x, w, kb).view(torch.int32)), (kb, mult)


def test_kernel_launch_contract(SK):
    x = torch.randn(2, 64, device="cpu")
    w = torch.randn(3, 64, device="cpu")
    out = torch.empty(2, 3, device="cpu")
    with pytest.raises(ValueError):
        SK.fixed_dot(x, w, out, kb=48, site="head")           # not a power of two
    with pytest.raises(ValueError):
        SK.fixed_dot(x, w, out, kb=128, site="head")          # does not divide K
    with pytest.raises(ValueError):
        SK.fixed_dot(x, w.t().contiguous().t(), out, kb=64, site="head")   # w rows not contiguous
    with pytest.raises(ValueError):
        SK.fixed_dot(x, w, torch.empty(2, 3, dtype=torch.bfloat16, device="cpu"), kb=64, site="head")
    k = SK.t9b5_head_fp32
    n0 = len(k.launches)
    SK.fixed_dot(x, w, out, kb=64, site="head")
    assert len(k.launches) == n0 + 1 and k.launches[-1]["enable_fp_fusion"] is False
    assert SK.t9b5_rows_per_program("head", 6) == 8 and SK.t9b5_rows_per_program("head", 1) == 1
    assert SK.t9b5_rows_per_program("mix", 6) == 1


def test_env_config_override(SK, monkeypatch):
    monkeypatch.setenv(SK.T9B5_CFG_ENV, "head:8=4,3,4,2;mix:1=2,1,4,1")
    assert SK.t9b5_config("head", 6) == (8, 4, 3, 4, 2)
    assert SK.t9b5_config("mix", 3) == (1, 2, 1, 4, 1)
    assert SK.t9b5_config("head", 1) == (1,) + SK.T9B5_CFG["head"][1]


def test_switch_and_kb():
    assert R2.fixed_kb("head", 5120) == 64 and R2.fixed_kb("mix", 20480) == 512
    assert R2.fixed_kb("head", 256) == 64 and R2.fixed_kb("mix", 1024) == 512 and R2.fixed_kb("mix", 96) == 32
    old = os.environ.pop(R2.FP32_IMPL_ENV, None)
    try:
        assert R2.fp32_impl() == "cublas" and not R2.fp32_fixed()
        os.environ[R2.FP32_IMPL_ENV] = "fixed"
        assert R2.fp32_fixed()
        os.environ[R2.FP32_IMPL_ENV] = "tf32"
        with pytest.raises(ValueError):
            R2.fp32_impl()
    finally:
        os.environ.pop(R2.FP32_IMPL_ENV, None)
        if old is not None:
            os.environ[R2.FP32_IMPL_ENV] = old


def test_head_and_mixes_fixed_forms_are_row_invariant_on_cpu():
    """head_fixed / hc_mixes_fixed (the oracle / reference-op forms): row r of R rows == the row alone, bitwise."""
    from types import SimpleNamespace

    g = torch.Generator(device="cpu").manual_seed(8)
    V, D, HC = 72, 128, 4
    model = SimpleNamespace(head=SimpleNamespace(weight=(torch.randn(V, D, generator=g, device="cpu") * 0.1).to(torch.bfloat16)
                                                 .float()))
    x = torch.randn(1, 6, D, generator=g, device="cpu").to(torch.bfloat16)
    full = R2.head_fixed(model, x, full_logits=True)
    assert full.shape == (1, 6, V)
    for i in range(6):
        one = R2.head_fixed(model, x[:, i:i + 1])
        assert one.shape == (1, V) and torch.equal(one[0], full[0, i])

    class _R:
        @staticmethod
        def hc_split_sinkhorn(m, sc, bs, hc, it, eps):
            return m * 1.0, sc, bs           # pass the mixes through (the sinkhorn is row-wise elsewhere)

    layer = SimpleNamespace(norm_eps=1e-6, hc_mult=HC, hc_sinkhorn_iters=1, hc_eps=1e-6)
    fn = torch.randn((2 + HC) * HC, HC * D, generator=g, device="cpu") * 0.02
    xs = torch.randn(1, 6, HC, D, generator=g, device="cpu").to(torch.bfloat16)
    m6 = R2.hc_mixes_fixed(_R, layer, xs, fn, None, None)[0]
    for i in range(6):
        m1 = R2.hc_mixes_fixed(_R, layer, xs[:, i:i + 1], fn, None, None)[0]
        assert torch.equal(m1[0, 0], m6[0, i])
    # the reference statistic and * rsqrt: equal to the definition applied by hand
    xf = xs.flatten(2).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    want = R2.fixed_dot_torch(xf.view(6, -1), fn, R2.fixed_kb("mix", HC * D), rs.view(6))
    assert torch.equal(m6.view(6, -1), want)


# ------------------------------------------------------------------------------------------------ the call sites
from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens, emu,  # noqa: E402,F401
                                  rows_dg, rows_glue)


def _count_sites(monkeypatch, model):
    """Counters: fixed-order launches per site, and F.linear calls on the two sites' weights (cuBLAS's path)."""
    import torch.nn.functional as F

    seen = {"mix": 0, "head": 0, "linear_mix": 0, "linear_head": 0}
    real_fixed = R2.launch_fixed_dot

    def fixed(x, w, out, *, site, scale=None):
        seen[site] += 1
        return real_fixed(x, w, out, site=site, scale=scale)

    hc_ptrs = {l.hc_attn_fn.data_ptr() for l in model.layers} | {l.hc_ffn_fn.data_ptr() for l in model.layers}
    head_ptr = model.head.weight.data_ptr()
    real_lin = F.linear

    def lin(x, w, b=None):
        if w.data_ptr() in hc_ptrs:
            seen["linear_mix"] += 1
        elif w.data_ptr() == head_ptr:
            seen["linear_head"] += 1
        return real_lin(x, w, b)

    monkeypatch.setattr(R2, "launch_fixed_dot", fixed)
    monkeypatch.setattr(F, "linear", lin)
    return seen


@pytest.mark.parametrize("fp32,moe_impl", [("fixed", "ref"), ("fixed", "dg-inplace"), ("cublas", "ref")])
def test_static6_call_sites_follow_the_switch(rows_glue, monkeypatch, fp32, moe_impl):
    """6-row StaticStep, glue fused and glue ref, under $DSV41_FP32_IMPL: `fixed` -> every mixes / head evaluation is
    the fixed-order launch (2 per layer + 1 per step) and none is F.linear; the glue's self-check (which adopts the
    same order on the oracle side) keeps hc_norm / mix_stat on; fused == ref glue, torch.equal. `cublas` (the
    default): no fixed launch at all (today's path, unchanged)."""
    from runtime.static_step import StaticStep

    monkeypatch.setenv(R2.FP32_IMPL_ENV, fp32)
    model, pool, _, _ = rows_glue("mini4")
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 41)
    seen = _count_sites(monkeypatch, model)
    st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl=moe_impl, glue_impl="fused",
                    _allow_cpu=True)
    d = st.glue.describe()
    assert st.glue.on("hc_norm") and st.glue.on("mix_stat"), d["clusters"]
    assert d["calibration"]["verify_hc_norm"] == "equal", d["calibration"]
    assert d["r2"]["fp32"]["impl"] == fp32 and st.describe()["fp32_impl"] == fp32
    for k in seen:
        seen[k] = 0
    f_lg, f_h, f_s = _static_run(model, st, ids, toks, 2)
    L = len(model.layers)
    pre = {"linear_mix": 2 * L, "linear_head": 1}   # _static_run's prefill: the reference forward (unchanged)
    if fp32 == "fixed":
        assert seen == {"mix": 2 * 2 * L, "head": 2, **pre}, seen
    else:
        assert seen == {"mix": 0, "head": 0, "linear_mix": 2 * L + 2 * 2 * L, "linear_head": 1 + 2}, seen
    st.detach()
    for k in seen:
        seen[k] = 0
    ref = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl=moe_impl, _allow_cpu=True)
    r_lg, r_h, r_s = _static_run(model, ref, ids, toks, 2)
    if fp32 == "fixed":   # the reference-op path takes the same order (hc_mixes_fixed / head_fixed)
        assert seen == {"mix": 2 * 2 * L, "head": 2, **pre}, seen
    _assert_rows_equal(f_lg, r_lg, f"6-row fused glue vs ref glue, fp32 {fp32}")
    assert all(torch.equal(x, y) for x, y in zip(f_h, r_h))
    ref.detach()


def test_static1_fixed_head_and_mixes(emu, monkeypatch):
    """The 1-row step under `fixed` (glue ref): the same launches at R = 1, and its logits differ from cuBLAS's path
    only through the two sites (the rest of the step is untouched: same argmax on these tokens, tiny |d|)."""
    from runtime.static_step import StaticStep

    from test_static_rows_cpu import _prefill

    model, pool, _ = emu("mini")
    ids, toks = _tf_tokens(17, 4, 7)
    outs = {}
    seen = _count_sites(monkeypatch, model)
    for fp32 in ("cublas", "fixed"):
        monkeypatch.setenv(R2.FP32_IMPL_ENV, fp32)
        st = StaticStep(model, pool, max_pos=255, rows=1, _allow_cpu=True)
        _prefill(model, ids)
        for k in seen:
            seen[k] = 0
        with torch.inference_mode():
            outs[fp32] = torch.stack([st.step(toks[i], len(ids) + i, graph=False)[0].float().clone()
                                      for i in range(4)])
        L = len(model.layers)
        want = ({"mix": 4 * 2 * L, "head": 4, "linear_mix": 0, "linear_head": 0} if fp32 == "fixed" else
                {"mix": 0, "head": 0, "linear_mix": 4 * 2 * L, "linear_head": 4})
        assert seen == want, (fp32, seen)
        st.detach()
    d = (outs["fixed"] - outs["cublas"]).abs().max().item()
    assert d < 1e-2 and torch.equal(outs["fixed"].argmax(-1), outs["cublas"].argmax(-1)), d


def test_gpu_rows_census_dry_run_lists_both_sites_row_invariant(emu, monkeypatch, capsys):
    """tests/gpu/test_static_rows_gpu.py's V0 test on the CPU emulations: the census table has the T9b-5 family for
    both sites (row-invariant) next to cuBLAS on the same inputs, and the step-level table the two fixed cases."""
    import contextlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_rows_t9b5", os.path.join(REPO, "tests", "gpu",
                                                                                 "test_static_rows_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    @contextlib.contextmanager
    def model_cpu(kind="mini"):
        model, pool, _ = emu(kind)
        yield model, pool

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STATIC_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    G.test_v0_census_and_step_level_table()
    out = capsys.readouterr().out
    lines = out.splitlines()
    fam = [l for l in lines if l.strip().startswith("T9b-5 fixed fp32")]
    assert len(fam) == 2 and all(" yes " in l for l in fam), fam
    inv = next(l for l in lines if l.strip().startswith("row-invariant:"))
    assert "T9b-5 fixed fp32" in inv, inv
    row = next(l for l in lines if "ref, fp32 fixed (T9b-5)" in l)
    assert row.split("ref, fp32 fixed (T9b-5)")[1].split()[0] == "24/24", row   # == row_invariant_mode's 24/24
    ref_row = next(l for l in lines if l.strip().startswith("ref ") and "0/24" in l)            # cuBLAS: 0/24
    assert ref_row
    assert any("fast, GEMV M=1 cfg pinned, fp32 fixed" in l for l in lines)
    assert os.environ.get(R2.FP32_IMPL_ENV) in (None, "cublas")     # the test restores the switch
    print(out)


def test_bench_flags():
    """--fp32-impl {cublas,fixed} on first_light.py and headline.py, default cublas (until the box gate passes)."""
    import importlib.util

    for name in ("first_light", "headline"):
        spec = importlib.util.spec_from_file_location(f"bench_{name}_t9b5", os.path.join(REPO, "bench", f"{name}.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        ap = m.build_parser()
        assert ap.parse_args(["--model-dir", "x"]).fp32_impl == "cublas"
        assert ap.parse_args(["--model-dir", "x", "--fp32-impl", "fixed"]).fp32_impl == "fixed"
        with pytest.raises(SystemExit):
            ap.parse_args(["--model-dir", "x", "--fp32-impl", "tf32"])
