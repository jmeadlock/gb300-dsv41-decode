"""Mac CPU tests of T9b-1 (runtime/gemv_mxfp8.py, `--fp8-impl gemv`): `.venv/bin/python -m pytest -q tests/`.

What a CPU can check, and does:
  * the e8m0 scale arithmetic: `pair_scale` = exp2(sa + sb - 254) built from its bits, against the fp32 product of
    torch's own e8m0 decodes on ALL 65,536 code pairs (normal, subnormal, underflow-to-0, inf, NaN), and against the
    reference kernel's two-step `(C * sa) * sb` (kernel.py:269) on random partials;
  * block indexing: output row n reads weight scale s[n // 32, k] (no expansion), K block k reads activation scale
    xs[m, k]; N not a multiple of 32 (the last, partial scale row);
  * the GEMV algorithm (`gemv_torch`, the kernel's arithmetic in torch) == a block-faithful emulation of the reference
    fp8_gemm (per-32 block partial, `acc += C * sa * sb` in K order) bit for bit at split 1, and within bf16 rounding
    of the exact float64 product; rows independent for M = 1..8; the operand contract; configs/cache plumbing (old
    round-1 cache entries ignored);
  * split-K (fix round): the kernel's scale expression pow2(sa) * pow2(sb) == pair_scale on all code pairs; the split
    partition (no empty split, K not a multiple of the split), auto split for >= 2 x SM programs capped at >= 4 K
    blocks; the torch mirror of the split order == an independent spelling, bit for bit, and within 1 ulp / <= 0.5 %
    of the single chain; the candidate grid covers the handoff's search space; the launch plan (grid, EVEN, BLOCK_M);
  * the model.linear patch: routed only inside the scope and only for fp8/e8m0 weights at <= 8 rows (x.numel() // K,
    not x.size(0)); fallthrough reasons; the whole `fp8_impl="gemv"` StaticStep == the reference forward on the mini
    checkpoints when both sides use the same block-faithful GEMM math (pins the act_quant call, shapes, dtypes and
    exactly which Linears are routed: 7 per layer + indexer wq_b + Engram wkv), the prefill and the eager reference
    untouched;
  * Generator `fp8_impl` and first_light's `--fp8-impl` / parity-levels variant rows on the fake model.
Skipped here (box only, tests/gpu/test_gemv_gpu.py): Triton, the real kernel's numerics vs the reference TileLang
kernel, graph capture, speed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import pytest
import torch

from runtime import gemv_mxfp8 as G
from test_static_emulated_cpu import _check, _ref_run, _scale_f, _static_run, emu  # noqa: F401 (emu = fixture)
from test_static_emulated_cpu import act_quant as _emu_act_quant

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


def _codes():
    return torch.arange(256, dtype=torch.int32)


def _e8(codes):
    return codes.to(torch.uint8).view(E8)


def _rand_fp8(*shape, g, lo=0, hi=0x78):
    """Finite e4m3 bytes over [lo, hi) magnitudes, random sign (every finite code incl. subnormals by default)."""
    b = torch.randint(lo, hi, shape, generator=g, dtype=torch.int32) | (torch.randint(0, 2, shape, generator=g,
                                                                                      dtype=torch.int32) << 7)
    return b.to(torch.uint8).view(FP8)


def _rand_e8(*shape, g, lo=118, hi=131):
    return _e8(torch.randint(lo, hi, shape, generator=g, dtype=torch.int32))


# ------------------------------------------------------------------------------------------------ scale arithmetic
def test_pair_scale_all_code_pairs_vs_fp32_product():
    c = _codes()
    f = _e8(c).float()                                   # torch's own e8m0 -> fp32 (exact; 255 -> NaN)
    assert torch.equal(f[:255], torch.tensor([math.ldexp(1.0, e - 127) for e in range(255)], dtype=torch.float32))
    prod = f[:, None] * f[None, :]                       # what (sa * sb) is in fp32
    got = G.pair_scale(c[:, None], c[None, :])
    nan = (c[:, None] == 255) | (c[None, :] == 255)
    assert torch.isnan(got[nan]).all() and torch.isnan(prod[nan]).all()
    assert torch.equal(got[~nan].view(torch.int32), prod[~nan].view(torch.int32))
    # every regime is actually exercised
    e = (c[:, None] + c[None, :] - 254)[~nan]
    g = got[~nan]
    assert bool((g[e > 127] == math.inf).all()) and bool((g[e < -149] == 0).all())
    sub = (e >= -149) & (e < -126)
    assert sub.any() and bool((g[sub] > 0).all()) and bool((g[sub] < 2.0 ** -126).all())
    assert float(G.pair_scale(torch.tensor(0), torch.tensor(105))) == 2.0 ** -149  # smallest subnormal, exact
    assert float(G.pair_scale(torch.tensor(127), torch.tensor(127))) == 1.0


def test_pair_scale_equals_reference_two_step_scaling():
    """kernel.py:269 computes `C * sa * sb` = (C * sa) * sb; for the scales a real model produces (no subnormal or
    overflow on the way) that is C * 2^(sa+sb-254) bit for bit."""
    g = torch.Generator().manual_seed(3)
    C = torch.randn(4096, generator=g) * torch.exp2(torch.randint(-20, 20, (4096,), generator=g).float())
    sa = torch.randint(90, 140, (4096,), generator=g)
    sb = torch.randint(100, 135, (4096,), generator=g)
    two = C * _e8(sa).float() * _e8(sb).float()
    one = C * G.pair_scale(sa, sb)
    assert torch.equal(one.view(torch.int32), two.view(torch.int32))


def test_weight_scale_rows_and_block_m():
    n = torch.arange(100)
    assert torch.equal(G.weight_scale_rows(n), n // 32) and int(G.weight_scale_rows(torch.tensor(99))) == 3
    assert [G.block_m(m) for m in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]
    for bad in (0, 9):
        with pytest.raises(ValueError):
            G.block_m(bad)


# ------------------------------------------------------------------------------------------------ the algorithm
def fp8_gemm_blocks(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """The reference fp8_gemm (kernel.py:226-272) at block 32 as a block-faithful emulation: per 32-wide K block a
    fp32 partial (the MMA), then `acc += C * sa * sb` in K order from 0 with sa = a_s[m, k], sb = b_s[n // 32, k] as
    fp32 (kernel.py:259-269), bf16 out. The partial is the same expression `gemv_torch` uses (the stand-in for the
    tensor core's internal order, which only the box test can compare against)."""
    assert block_size == 32
    K = a.size(-1)
    M, N, kb = a.numel() // K, b.size(0), K // 32
    C = (a.reshape(M, K).float().view(M, 1, kb, 32) * b.float().view(1, N, kb, 32)).sum(-1)
    sa = _scale_f(a_s.reshape(M, kb))
    sb = _scale_f(b_s)[torch.arange(N) // 32]
    acc = torch.zeros(M, N)
    for k in range(kb):
        acc = acc + C[:, :, k] * sa[:, k, None] * sb[None, :, k]
    return acc.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def _operands(M, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    return (_rand_fp8(M, K, g=g), _rand_e8(M, K // 32, g=g, lo=110, hi=128), _rand_fp8(N, K, g=g),
            _rand_e8((N + 31) // 32, K // 32, g=g))


@pytest.mark.parametrize("M,N,K", [(1, 64, 256), (6, 96, 512), (8, 100, 160), (3, 31, 64), (1, 1280, 256)])
def test_gemv_torch_equals_block_faithful_reference(M, N, K):
    xq, xs, w, s = _operands(M, N, K, seed=M * 1000 + N)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        want = fp8_gemm_blocks(xq, xs, w, s, E8, 32)
    finally:
        torch.set_default_dtype(prev)
    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16)
    got = G.mxfp8_gemv(xq, xs, w, s, out)                # CPU -> gemv_torch, into `out`
    assert got is out and torch.equal(got, want)
    # within bf16 rounding of the exact product (float64, the dequantized operands)
    A = xq.double() * _e8(xs.view(torch.uint8).int()).double().repeat_interleave(32, 1)
    B = w.double() * _scale_f(s).double()[torch.arange(N) // 32].repeat_interleave(32, 1)
    exact = A @ B.t()
    err = (got.double() - exact).abs()
    tol = exact.abs() * 2.0 ** -8 + 1e-30 + exact.abs().max() * 2.0 ** -20  # 1/2 bf16 ulp + fp32 accumulation slack
    assert bool((err <= tol).all()), float((err - tol).max())


def test_block_indexing_scale_rows_and_k_blocks():
    """w = x = 1.0: out[m, n] = 32 * sum_k 2^(xs[m,k]-127) * 2^(s[n//32,k]-127), exact. A wrong row index (n % 32, n
    // 16, expansion off by one) or a wrong K-block pairing changes the value."""
    M, N, K = 2, 70, 128
    one = torch.tensor(1.0).to(FP8)
    xq = one.expand(M, K).contiguous()
    w = one.expand(N, K).contiguous()
    xs = _e8(torch.tensor([[127, 128, 126, 129], [120, 127, 127, 130]], dtype=torch.int32))
    s = _e8(torch.tensor([[127, 127, 127, 127], [129, 127, 127, 127], [125, 127, 129, 127]], dtype=torch.int32))
    got = G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16)).double()
    sx, sw = xs.view(torch.uint8).double() - 127, s.view(torch.uint8).double() - 127
    want = torch.stack([torch.stack([32 * torch.exp2(sx[m] + sw[n // 32]).sum() for n in range(N)])
                        for m in range(M)])
    assert torch.equal(got, want.to(torch.bfloat16).double())
    assert len({float(got[0, n]) for n in (0, 32, 64)}) == 3  # the three scale rows differ


@pytest.mark.parametrize("cfg", [None, G.GemvConfig(32, 4, 4, 2, 5, 1)])
def test_rows_independent_m1_to_8(cfg):
    xq, xs, w, s = _operands(8, 48, 640, seed=9)
    full = G.mxfp8_gemv(xq, xs, w, s, torch.empty(8, 48, dtype=torch.bfloat16), config=cfg)
    for M in range(1, 9):
        part = G.mxfp8_gemv(xq[:M].contiguous(), xs[:M].contiguous(), w, s, torch.empty(M, 48, dtype=torch.bfloat16),
                            config=cfg)
        assert torch.equal(part, full[:M])
    for m in range(8):
        one = G.mxfp8_gemv(xq[m:m + 1].contiguous(), xs[m:m + 1].contiguous(), w, s,
                           torch.empty(1, 48, dtype=torch.bfloat16), config=cfg)
        assert torch.equal(one, full[m:m + 1])


def test_operand_contract():
    xq, xs, w, s = _operands(2, 64, 128, seed=1)
    out = torch.empty(2, 64, dtype=torch.bfloat16)
    G.check_operands(xq, xs, w, s, out)
    G.check_operands(xq, xs.view(torch.uint8), w, s.view(torch.uint8), out)  # uint8 scale bytes accepted
    bad = [
        (xq.float(), xs, w, s, out), (xq, xs.float(), w, s, out), (xq, xs, w[:, :96], s, out),
        (xq, xs[:, :3], w, s, out), (xq, xs, w, s[:1], out), (xq, xs, w, s, out.float()),
        (xq, xs, w, s, torch.empty(2, 63, dtype=torch.bfloat16)), (xq[:, :100], xs, w, s, out),
        (torch.cat([xq] * 5), torch.cat([xs] * 5), w, s, None),            # 10 rows > 8
        (xq.t().contiguous().t(), xs, w, s, out),                          # not contiguous
    ]
    for args in bad:
        with pytest.raises((ValueError, TypeError)):
            G.check_operands(*args)
    y = G.linear_fp8(xq.view(1, 2, 128), xs.view(1, 2, 4), w, s)           # [..., K] -> [..., N]
    assert y.shape == (1, 2, 64) and y.dtype == torch.bfloat16
    assert torch.equal(y.view(2, 64), G.mxfp8_gemv(xq, xs, w, s, out))


def test_configs_cache_roundtrip_and_no_tune(tmp_path, monkeypatch):
    G.reset_configs()
    path = str(tmp_path / "gemv-tune.json")
    monkeypatch.setenv("DSV41_GEMV_CACHE", path)
    monkeypatch.setenv("DSV41_GEMV_TUNE", "0")
    try:
        d0 = G.get_config(512, 5120, 1)                                  # tuning off -> default, split resolved
        assert d0 == G.default_config(512, 5120, 1) and d0.split_k == 19  # 16 tiles of 32 -> 19 splits >= 296 progs
        assert (d0.block_n, d0.ku, d0.num_warps, d0.num_stages, d0.mma) == (32, 4, 4, 1, 0)
        c = G.GemvConfig(64, 4, 8, 2, 3, 1)
        G.set_config(32768, 1280, 6, c)                                 # key (N, K, block_m(6) = 8)
        assert G.get_config(32768, 1280, 7) == c and G.config_key(32768, 1280, 6) == (32768, 1280, 8)
        G.set_config(5120, 8192, 1, G.GemvConfig(64, 4, 8, 2))          # split_k 0 (auto) is resolved when set
        assert G.get_config(5120, 8192, 1).split_k == 4
        assert G.save_cache() == path
        data = json.load(open(path))
        assert data["configs"]["32768x1280/m8"] == c.as_dict() == {"block_n": 64, "ku": 4, "num_warps": 8,
                                                                    "num_stages": 2, "split_k": 3, "mma": 1}
        # a round-1 entry (no split_k / mma) in the same file is ignored, not half-loaded
        data["configs"]["1280x5120/m1"] = {"block_n": 64, "ku": 4, "num_warps": 8, "num_stages": 1}
        json.dump(data, open(path, "w"))
        G.reset_configs()
        assert G.get_config(32768, 1280, 8) == c                        # loaded from the file on first miss
        d = G.describe()
        assert d["cache_loaded"] == path and d["configs"]["32768x1280/m8"] == c.as_dict() and d["tune"] is False
        assert d["cache_ignored"] == ["1280x5120/m1"] and "1280x5120/m1" not in d["configs"]
        assert G.get_config(1280, 5120, 1) == G.default_config(1280, 5120, 1)  # ignored entry -> default (tune off)
    finally:
        G.reset_configs()


# ------------------------------------------------------------------------------------------------ split-K
def test_pow2_product_equals_pair_scale_all_code_pairs():
    """The kernel's scale expression pow2(sa) * pow2(sb) (fp32 multiply of two exact powers of two) is the exact pair
    scale on all 65,536 code pairs: subnormal factors (code 0) and results, underflow to 0, inf, NaN."""
    c = _codes()
    p = G.pow2_code(c)
    assert torch.equal(p[:255].view(torch.int32), _e8(c).float()[:255].view(torch.int32)) and torch.isnan(p[255])
    got = G.pow2_code(c)[:, None] * G.pow2_code(c)[None, :]
    want = G.pair_scale(c[:, None], c[None, :])
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got[~nan].view(torch.int32), want[~nan].view(torch.int32))


def test_split_plan_partitions_and_auto_split():
    for KB in (1, 2, 9, 10, 40, 72, 160, 192, 256):
        for sk in range(1, 70):
            per, ns = G.split_plan(KB, sk)
            spans = [(j * per, min((j + 1) * per, KB)) for j in range(ns)]
            assert spans[0][0] == 0 and spans[-1][1] == KB and all(a < b for a, b in spans)       # no empty split
            assert all(spans[j][1] == spans[j + 1][0] for j in range(ns - 1)) and ns <= min(sk, KB)
    assert G.split_plan(40, 7) == (6, 7)      # last split 4 blocks: K not a multiple of the split
    assert G.split_plan(9, 6) == (2, 5)       # 6 requested -> 5 non-empty splits
    assert G.split_plan(160, 1) == (160, 1)
    sms = 148
    # the handoff's shapes: programs >= 2 x 148 unless capped at >= 4 K blocks per program
    assert G.auto_split_k(512, 5120, 32, sms) == 19 and G.auto_split_k(512, 5120, 64, sms) == 37
    assert G.auto_split_k(512, 5120, 128, sms) == 40                  # capped: 160 / 4
    assert G.auto_split_k(32768, 1280, 64, sms) == 1 and G.auto_split_k(25600, 6144, 64, sms) == 1
    assert G.auto_split_k(5120, 8192, 64, sms) == 4 and G.auto_split_k(7, 64, 32, sms) == 1
    for (N, K) in [(1280, 5120), (32768, 1280), (512, 5120), (5120, 8192), (4096, 1280), (2304, 5120), (5120, 2304),
                   (25600, 6144)]:
        for bn in G.TUNE_BLOCK_N:
            sk = G.auto_split_k(N, K, bn, sms)
            per, ns = G.split_plan(K // 32, sk)
            tiles = -(-N // bn)
            assert per >= 4 and (tiles * sk >= 2 * sms or sk == (K // 32) // 4), (N, K, bn, sk)


def _chunked_reference(xq, xs, w, s, split_k):
    """Independent spelling of the split-K order: per split a left fold over its blocks from +0.0 (python reduce
    over [M, N] slices), then a left fold over the split partials from +0.0."""
    import functools

    M, K = xq.shape
    N, kb = w.size(0), K // 32
    q = (xq.float().view(M, 1, kb, 32) * w.float().view(1, N, kb, 32)).sum(-1) \
        * G.pair_scale(xs.view(torch.uint8).int().view(M, 1, kb), s.view(torch.uint8).int()[torch.arange(N) // 32]
                       .view(1, N, kb))
    per = -(-kb // split_k)
    chunks = [range(j, min(j + per, kb)) for j in range(0, kb, per)]
    parts = [functools.reduce(lambda a, k: a + q[:, :, k], ks, torch.zeros(M, N)) for ks in chunks]
    return functools.reduce(lambda a, p: a + p, parts, torch.zeros(M, N))


@pytest.mark.parametrize("M,N,K,split", [(1, 64, 1280, 7), (3, 96, 1280, 4), (6, 100, 320, 3), (8, 40, 5120, 19),
                                         (1, 33, 288, 6), (2, 64, 1280, 1)])
def test_gemv_torch_split_k_order(M, N, K, split):
    """The CPU algorithm (and through it the CPU path of mxfp8_gemv with a config) mirrors the kernel's split-K:
    K-order chains per split, then the partials in split order. KB not a multiple of the split included."""
    xq, xs, w, s = _operands(M, N, K, seed=7 * M + N + split)
    f32 = G.gemv_torch_f32(xq, xs, w, s, split_k=split)
    assert torch.equal(f32.view(torch.int32), _chunked_reference(xq, xs, w, s, split).view(torch.int32))
    cfg = G.GemvConfig(32, 4, 4, 2, split, 1)
    got = G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16), config=cfg)
    assert torch.equal(got, f32.to(torch.bfloat16))
    one = G.gemv_torch(xq, xs, w, s, split_k=1)
    assert torch.equal(one, G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16)))  # no config: 1
    # split-K vs the single chain: equal, or one bf16 rounding of fp32 values that differ by accumulation error only
    d = (got.double() - one.double()).abs()
    tol = one.double().abs() * 2.0 ** -7 + _abs_sum(xq, xs, w, s).double() * 2.0 ** -17
    assert bool((d <= tol).all()), float((d - tol).max())


def _abs_sum(xq, xs, w, s):
    """fp32 sum of |terms| per output (the scale of fp32 accumulation error): the algorithm on |x|, |w|."""
    absb = lambda t: t.view(torch.uint8).bitwise_and(0x7F).view(FP8)  # noqa: E731 - e4m3 sign bit cleared
    return G.gemv_torch_f32(absb(xq), xs, absb(w), s, 1)


def test_split_k_changes_fp32_order_somewhere():
    """The split is really applied (else the mirror tests above would be vacuous): on wide-range data the fp32 sums
    of split 1 and split 8 differ in some outputs, and every difference is within fp32 accumulation error."""
    xq, xs, w, s = _operands(4, 256, 2048, seed=5)
    a = G.gemv_torch_f32(xq, xs, w, s, 1)
    b = G.gemv_torch_f32(xq, xs, w, s, 8)
    assert bool((a != b).any())
    assert bool(((a - b).abs() <= _abs_sum(xq, xs, w, s) * 2.0 ** -17).all())
    bf_a, bf_b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    assert float((bf_a != bf_b).float().mean()) <= 5e-3   # the 1-ulp class: rare bf16 flips (1024 outputs here)


def test_candidates_cover_the_search_space():
    sms = 148
    for (N, K, M) in [(512, 5120, 1), (32768, 1280, 6), (5120, 8192, 8), (7, 64, 3)]:
        cands = G.candidate_configs(N, K, M, sms)
        assert len(set(cands)) == len(cands) and all(c.split_k >= 1 for c in cands)
        assert {c.block_n for c in cands} == {32, 64, 128} and {c.ku for c in cands} == {2, 4, 8}
        assert {c.num_warps for c in cands} == {4, 8} and {c.num_stages for c in cands} == {2, 3, 4}
        assert {c.mma for c in cands} == {0, 1}
        for bn in G.TUNE_BLOCK_N:
            s0 = G.auto_split_k(N, K, bn, sms)
            assert s0 in {c.split_k for c in cands if c.block_n == bn}
        if G.block_m(M) >= 4:
            assert not [c for c in cands if not c.mma and c.block_n == 128]
    small = G.candidate_configs(512, 5120, 1, sms)
    assert {c.split_k for c in small if c.block_n == 32} == {19, 38}  # auto and 2x auto, both >= 4 K blocks each
    big = G.candidate_configs(32768, 1280, 1, sms)
    assert {c.split_k for c in big if c.block_n == 64} == {1, 2}
    c1, c2 = G.GemvConfig(64, 2, 4, 2, 19, 0), G.GemvConfig(32, 8, 8, 4, 19, 0)
    assert G.numerics_key(c1, 512, 5120) == G.numerics_key(c2, 512, 5120) == (9, 18, 0)
    assert G.numerics_key(replace_mma(c1), 512, 5120) == (9, 18, 1)
    assert G.numerics_key(G.GemvConfig(64, 2, 4, 2, 20, 0), 512, 5120) == (8, 20, 0)


def replace_mma(cfg):
    import dataclasses

    return dataclasses.replace(cfg, mma=1)


def test_launch_plan():
    p = G.launch_plan(G.GemvConfig(32, 4, 4, 2, 19, 0), 1, 512, 5120)
    assert p["grid"] == (16, 18) and (p["PER"], p["NSPLIT"]) == (9, 18) and p["SPLIT"] and not p["EVEN"]
    assert p["BLOCK_M"] == 1 and p["reduce_grid"] == (1,)
    p = G.launch_plan(G.GemvConfig(64, 4, 8, 3, 1, 1), 6, 32768, 1280)
    assert p["grid"] == (512, 1) and p["BLOCK_M"] == 16 and p["EVEN"] and not p["SPLIT"] and p["reduce_grid"] is None
    p = G.launch_plan(G.GemvConfig(128, 8, 4, 2, 4, 0), 8, 5120, 8192)
    assert p["grid"] == (40, 4) and p["PER"] == 64 and p["EVEN"] and p["BLOCK_M"] == 8 and p["reduce_grid"] == (40,)
    p = G.launch_plan(G.GemvConfig(32, 8, 4, 2, 7, 1), 3, 40, 1280)        # KB 40, split 7: last split short
    assert (p["PER"], p["NSPLIT"]) == (6, 7) and not p["EVEN"] and p["grid"] == (2, 7)


def test_impl_constants():
    from runtime import generate, static_step

    assert G.FP8_IMPLS == static_step.FP8_IMPLS == generate.FP8_IMPLS == ("ref", "gemv")
    assert G.GEMV_MAX_M == 8


def test_bench_enumerates_the_steps_fp8_linears(tmp_path):
    """bench/gemv_bench.py's shape enumeration on the mini checkpoint: exactly the Linears a decode step routes
    (7 per layer + indexer wq_b + Engram wkv), wo_a and the Engram table excluded with a reason."""
    from mini_ckpt import make_mini_checkpoint
    from runtime.ckpt import ShardIndex

    spec = importlib.util.spec_from_file_location("bench_gemv", os.path.join(REPO, "bench", "gemv_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = make_mini_checkpoint(str(tmp_path / "mini"))
    roles, skipped = mod.enumerate_fp8_linears(ShardIndex.scan(ck["model_dir"]))
    assert set(roles) == mod.EXPECTED_ROLES, (sorted(roles), skipped)
    assert sum(len(v["layers"]) for v in roles.values()) == 17  # 2 x 7 + 1 indexer + 2 Engram (mini)
    assert roles["engram.wkv"]["shape"] == (1280, 1536) and roles["attn.indexer.wq_b"]["layers"] == [1]
    assert any(s.startswith("attn.wo_a:") for s in skipped) and any(s.startswith("engram.embed:") for s in skipped)
    assert mod.pick_layers(list(range(40)), 8) == [0, 5, 10, 15, 20, 25, 30, 35]
    assert mod.bytes_moved(1, 64, 128) == 64 * 128 + 2 * 4 + 128 + 4 + 128


def _bench_module():
    spec = importlib.util.spec_from_file_location("bench_gemv", os.path.join(REPO, "bench", "gemv_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_gates_fix_round():
    """The fix-round gates: >= 2x at M=1 and >= 1.5x at M > 1 per shape; >= 4 TB/s for >= 16 MiB weights; <= 8 µs
    for smaller ones; drift <= 0.5 %; per-step saving >= 12 ms; all_pass only if none of them is False."""
    mod = _bench_module()

    def rec(N, K, m1, m6, mism=1e-5):
        per_m = {1: {"speedup": m1[0], "us_gemv": m1[1], "GBps_gemv": m1[2], "mismatch": mism,
                     "config_invariant": True},
                 6: {"speedup": m6, "us_gemv": 1.0, "GBps_gemv": 1.0, "mismatch": mism, "config_invariant": True}}
        return {"shape": [N, K], "per_m": per_m, "gates": mod.shape_gates(N, K, per_m)}

    big_ok = rec(25600, 6144, (2.5, 30.0, 5300.0), 1.6)
    small_ok = rec(512, 5120, (8.0, 7.5, 360.0), 2.0)
    g = mod.shape_gates(512, 5120, small_ok["per_m"])
    assert g["bw_ge_4TBps_m1"] is None and g["small_le_8us_m1"] is True and g["speedup_ge_1p5x_multirow"] is True
    g = mod.shape_gates(25600, 6144, big_ok["per_m"])
    assert g["bw_ge_4TBps_m1"] is True and g["small_le_8us_m1"] is None
    ok = mod.overall_gates([big_ok, small_ok], {"ms_ref": 15.8, "ms_gemv": 3.0, "saved_ms": 12.8})
    assert ok["all_pass"] and not ok["failing"]
    for bad, name in [(rec(512, 5120, (8.0, 8.5, 300.0), 2.0), "small_le_8us_m1"),
                      (rec(512, 5120, (1.9, 7.0, 300.0), 2.0), "speedup_ge_2x_m1"),
                      (rec(512, 5120, (8.0, 7.0, 300.0), 1.4), "speedup_ge_1p5x_multirow"),
                      (rec(5120, 8192, (3.0, 12.0, 3500.0), 2.0), "bw_ge_4TBps_m1"),
                      (rec(512, 5120, (8.0, 7.0, 300.0), 2.0, mism=6e-3), "mismatch_le_0p5pct")]:
        r = mod.overall_gates([big_ok, bad], {"ms_ref": 15.8, "ms_gemv": 3.0, "saved_ms": 12.8})
        assert not r["all_pass"] and r["failing"] == {name: [bad["shape"]]}, (name, r)
    r = mod.overall_gates([big_ok, small_ok], {"ms_ref": 15.8, "ms_gemv": 4.0, "saved_ms": 11.8})
    assert r["step_saving_ge_12ms_m1"] is False and not r["all_pass"]
    ts = mod.tune_summary({"us": {"a": 3.0, "b": 2.0, "c": 9.0}, "errors": {"d": "E"}, "n_timed": 3, "n_failed": 1})
    assert ts["best"] == ("b", 2.0) and ts["runner_up"] == ("a", 3.0) and ts["spread"] == 4.5
    assert mod.tune_summary(None) == {"source": "cache"}


# ------------------------------------------------------------------------------------------------ the model.linear patch
def _faithful_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """The emulated act_quant returning scales in `scale_dtype` as kernel.py does (e8m0 here; exact for powers of 2)."""
    r = _emu_act_quant(x, block_size, scale_fmt, scale_dtype, inplace)
    if inplace or scale_dtype != E8:
        return r
    q, s = r
    s8 = s.to(E8)
    assert torch.equal(s8.float(), s)
    return q, s8


@pytest.fixture
def gemv_emu(emu, monkeypatch):
    """The emulated mini model with the block-faithful fp8_gemm and an e8m0-returning act_quant on BOTH sides."""
    from runtime import patches

    def build(kind):
        model, pool, fk = emu(kind)
        R = sys.modules["model"]
        monkeypatch.setattr(R, "act_quant", _faithful_act_quant)
        monkeypatch.setattr(R, "fp8_gemm", fp8_gemm_blocks)
        patches.fp8_gemv_stats(reset=True)
        return model, pool, fk

    yield build
    R = sys.modules.get("model")
    if R is not None and hasattr(R, "_orig_linear"):
        patches.fp8_linear_gemv(False)  # leave the reference module as found


def _expected_routed_per_step(model):
    """fp8 Linears a 1-row decode step calls: wq_a, wq_b, wkv, wo_b + shared w1/w3/w2 per layer, indexer wq_b on
    index-source layers, Engram wkv on Engram layers (the real model: 40 x 7 + 8 + 2 = 290)."""
    n = 0
    for layer in model.layers:
        n += 7
        n += layer.attn.indexer is not None
        n += layer.engram is not None
    return n


def test_patch_routing_conditions(gemv_emu):
    from runtime import patches

    model, _, _ = gemv_emu("mini")
    R = sys.modules["model"]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        lin = model.layers[0].attn.wq_a                                        # fp8 [128, 256]
        x1 = torch.randn(1, 1, 256).to(torch.bfloat16)
        patches.fp8_linear_gemv(True)
        assert patches.fp8_linear_gemv_enabled()
        want = R._orig_linear(x1, lin.weight)
        assert torch.equal(lin(x1), want) and not patches.fp8_gemv_stats()["routed"]  # outside the scope: inert
        with patches.fp8_gemv_scope():
            y = lin(x1)
            assert torch.equal(y, want)                       # same math on CPU (block-faithful both sides)
            assert patches.fp8_gemv_stats()["routed"] == {(128, 256, 1): 1}
            x8, x9 = torch.randn(1, 8, 256).to(torch.bfloat16), torch.randn(1, 9, 256).to(torch.bfloat16)
            lin(x8)                                           # 8 rows (x.size(0) == 1): routed
            lin(x9)                                           # 9 rows (x.size(0) == 1 too): falls through
            model.layers[0].ffn.gate(x1.view(1, 256))         # fp32 weight: never considered
            model.layers[0].attn.wo_a.weight                  # bf16, used by einsum, not linear()
            st = patches.fp8_gemv_stats()
            assert st["routed"] == {(128, 256, 1): 1, (128, 256, 8): 1}
            assert st["fallthrough"] == {("rows", 128, 256): 1}
            with pytest.raises(AssertionError):
                lin.bias = torch.nn.Parameter(torch.zeros(128))
                R.linear(x1, lin.weight, lin.bias)            # bias -> reference path -> model.py:184 assert
            lin.bias = None
        torch.set_default_dtype(torch.float32)
        assert patches.fp8_gemv_reason(R, x1, lin.weight) == "default_dtype"
        torch.set_default_dtype(torch.bfloat16)
        assert patches.fp8_gemv_reason(R, x1, lin.weight) is None
        assert patches.fp8_gemv_reason(R, x1, lin.weight.float()) == "dtype"
        assert patches.fp8_gemv_reason(R, torch.randn(1, 2, 255), lin.weight) == "k"
        assert patches.fp8_gemv_reason(R, x1, lin.weight, torch.zeros(1)) == "bias"
    finally:
        torch.set_default_dtype(prev)
    assert not patches.fp8_gemv_active()


@pytest.mark.parametrize("kind", ["mini", "mini4"])
def test_gemv_static_step_equals_reference_under_same_gemm_math(gemv_emu, kind):
    from runtime import patches
    from runtime.static_step import StaticStep

    model, pool, _ = gemv_emu(kind)
    st = StaticStep(model, pool, max_pos=127, fp8_impl="gemv", _allow_cpu=True)
    assert st.fp8_impl == "gemv" and patches.fp8_linear_gemv_enabled()
    per_step = _expected_routed_per_step(model)
    g = torch.Generator().manual_seed(11)
    for L in (2, 3, 16, 17):  # both prefill parities; 10 decode steps cover both graph parities
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        patches.fp8_gemv_stats(reset=True)
        ref = _ref_run(model, ids, 10)
        assert not patches.fp8_gemv_stats()["routed"]  # the eager reference (prefill + decode) never routes
        got = _static_run(model, st, ids, 10)
        _check(got, ref, (kind, "fp8 gemv emulated", L))
        stats = patches.fp8_gemv_stats()
        assert sum(stats["routed"].values()) == 10 * per_step, (stats, per_step)
        assert all(k[2] == 1 for k in stats["routed"]) and not stats["fallthrough"]
    assert per_step == (17 if kind == "mini" else 33)
    st.detach()


def test_gemv_and_dg_masked_together_emulated(gemv_emu, monkeypatch):
    """Both T9 impls in one StaticStep: still exact under identical GEMM math (the shared expert of the dg MoE is
    routed through the GEMV too)."""
    from runtime import patches
    from runtime.static_step import StaticStep
    from test_dg_moe_cpu import _fake_deep_gemm

    fake = _fake_deep_gemm()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    model, pool, _ = gemv_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    ids = list(range(5, 21))
    patches.fp8_gemv_stats(reset=True)
    got = _static_run(model, st, ids, 8)
    _check(got, _ref_run(model, ids, 8), "dg-masked + gemv emulated")
    assert sum(patches.fp8_gemv_stats()["routed"].values()) == 8 * _expected_routed_per_step(model)
    assert len(fake.calls) == 3 * len(model.layers) * 8
    st.detach()
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=127, fp8_impl="dg", _allow_cpu=True)


def test_ref_static_step_unaffected_after_gemv(gemv_emu):
    """A ref StaticStep built after a gemv one (the patch stays installed) routes nothing and stays exact."""
    from runtime import patches
    from runtime.static_step import StaticStep

    model, pool, _ = gemv_emu("mini")
    st = StaticStep(model, pool, max_pos=127, fp8_impl="gemv", _allow_cpu=True)
    st.detach()
    st2 = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    ids = list(range(30, 47))
    patches.fp8_gemv_stats(reset=True)
    _check(_static_run(model, st2, ids, 6), _ref_run(model, ids, 6), "ref after gemv")
    assert not patches.fp8_gemv_stats()["routed"] and patches.fp8_linear_gemv_enabled()
    st2.detach()


# ------------------------------------------------------------------------------------------------ Generator + first_light
class FakeStaticFp8:
    """tests/test_dg_moe_cpu.FakeStaticImpl with an fp8_impl: any non-ref impl adds a fixed tiny offset to logit 5 at
    every decode step (deterministic, eager == graph) -> L0 differs, L1/L2 equal; gemv adds 2e-3, dg-masked 1e-3."""

    built = []

    def __init__(self, model, moe_impl="ref", fp8_impl="ref"):
        self.model, self.moe_impl, self.fp8_impl = model, moe_impl, fp8_impl
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures = {}, []
        FakeStaticFp8.built.append((moe_impl, fp8_impl))

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


def _gen(mode="eager", moe_impl="ref", fp8_impl="ref"):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStaticFp8.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=FakeStaticFp8, moe_impl=moe_impl, fp8_impl=fp8_impl)


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()][:n]


def test_generator_fp8_impl_plumbing():
    gen = _gen(mode="static", fp8_impl="gemv")
    gen.greedy([5, 9, 12, 3], 4)
    assert gen.static.fp8_impl == "gemv" and gen.static.moe_impl == "ref" and gen.last_meta["fp8_impl"] == "gemv"
    old = gen.static
    gen.set_fp8_impl("gemv")
    assert gen.static is old  # no change, no rebuild
    gen.set_fp8_impl("ref")
    assert gen.static is None and old.graphs == {}
    gen.greedy([5, 9, 12, 3], 4)
    assert (gen.static.moe_impl, gen.static.fp8_impl) == ("ref", "ref")
    gen.set_moe_impl("dg-masked")
    gen.set_fp8_impl("gemv")
    gen.greedy([5, 9, 12, 3], 4)
    assert FakeStaticFp8.built == [("ref", "gemv"), ("ref", "ref"), ("dg-masked", "gemv")]
    with pytest.raises(ValueError):
        gen.set_fp8_impl("deepgemm")


@pytest.mark.parametrize("moe,fp8,label", [("ref", "gemv", "gemv"), ("dg-masked", "gemv", "dg-masked+gemv")])
def test_parity_levels_fp8_variant_rows(moe, fp8, label):
    fl = _first_light()
    gen = _gen(mode="graph", moe_impl=moe, fp8_impl=fp8)
    out = []
    det = torch.utils.deterministic
    fill0 = det.fill_uninitialized_memory
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    assert res["verdict"] == "PASS" and (res["moe_impl"], res["fp8_impl"], res["variant"]) == (moe, fp8, label)
    for f in ("off", "on"):
        l0 = by[(f, "static-eager vs reference-eager")]
        assert not l0["gate"] and l0["equal"] == 0
        assert by[(f, "graph vs static-eager")]["all_equal"] and by[(f, "graph relaunch vs replay#1")]["all_equal"]
    t8 = by[("off", "ref static-eager vs reference-eager")]
    assert t8["gate"] and t8["all_equal"]
    dr = by[("off", f"{label} vs ref static-eager")]
    off = (1e-3 if moe != "ref" else 0.0) + 2e-3
    assert not dr["gate"] and dr["max_abs_dlogit"] == pytest.approx(off, rel=1e-3)
    assert ("ref", "ref") in FakeStaticFp8.built  # the drift reference ran with BOTH impls ref
    assert any(l.startswith(f"drift {label} vs ref static-eager") for l in out)
    assert (gen.moe_impl, gen.fp8_impl, gen.mode) == (moe, fp8, "graph")
    assert (gen.static.moe_impl, gen.static.fp8_impl) == (moe, fp8) and gen.static.captures[-1] is False
    assert det.fill_uninitialized_memory == fill0


def test_first_light_fp8_impl_flag_and_report(tmp_path, capsys):
    fl = _first_light()
    assert fl.build_parser().parse_args(["--model-dir", "x"]).fp8_impl == "ref"
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--moe-impl", "dg-masked", "--fp8-impl",
                                      "gemv", "--parity-levels", "--level-steps", "2", "--fixture", FIXTURE, "--only",
                                      "prose", "--reps", "1", "--max-new", "4", "--step-cell", "6", "--stats-from",
                                      "2", "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    assert a.fp8_impl == "gemv"
    with pytest.raises(SystemExit):
        fl.build_parser().parse_args(["--model-dir", "x", "--fp8-impl", "deepgemm"])
    gen = _gen(mode="graph", moe_impl=a.moe_impl, fp8_impl=a.fp8_impl)
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["generator"]["fp8_impl"] == "gemv" and report["step_cell"]["fp8_impl"] == "gemv"
    assert report["fp8_gemv"]["impl"] == "gemv" and "configs" in report["fp8_gemv"]
    assert report["parity_levels"]["variant"] == "dg-masked+gemv" and report["parity_levels"]["verdict"] == "PASS"
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | mode graph moe dg-masked fp8 gemv levels PASS")


# ================================================================================================ T9b-6: gemv3
# runtime/gemv_mxfp8.py section "T9b-6" (docs/HANDOFF-T9B6-GEMV-BYTES.md). Through tests/triton_shim.py the real
# `@triton.jit` bodies run program by program on data whose 32-product block sums are exact in fp32
# (test_gemv_kernel_shim_cpu._exact_fp8), so kernel == the torch mirror of the same split holds bit for bit and only
# the index math, slicing, K order, split fold and counter protocol are under test (not Triton compilation, layouts,
# memory ordering or speed: tests/gpu/test_gemv_gpu.py + bench/gemv_bench.py on the box).
import itertools as _itertools  # noqa: E402
import random as _random  # noqa: E402

import numpy as _np  # noqa: E402

import triton_shim  # noqa: E402

G3 = G.Gemv3Config


@pytest.fixture(scope="module")
def K3():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "gemv_t9b6_shim")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture
def _impl_reset():
    G.set_impl(None)
    yield
    G.set_impl(None)


def _ops3(M, N, K_, seed):
    from test_gemv_kernel_shim_cpu import _operands as _exact_operands

    return _exact_operands(M, N, K_, seed)


def _run3(K3, xq, xs, w, s, cfg, out=None, rope=None):
    """One gemv3 launch through the shim with fresh NaN workspace and zero counters -> (out, ws, cnt, plan)."""
    M, Kd = xq.shape
    N = w.size(0)
    p = K3.launch_plan3(cfg, M, N, Kd)
    ws = torch.full((p["NSPLIT"], M, N), float("nan"), dtype=torch.float32)
    cnt = torch.zeros(p["tiles"], dtype=torch.int32)
    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16) if out is None else out
    K3._launch3(xq, xs, w, s, out, cfg, ws if p["SPLIT"] else out, cnt, rope=rope)
    return out, ws, cnt, p


CASES3 = [  # (M, N, K, Gemv3Config(block_n, kt, warps, stages, split_k, wide))
    (1, 64, 256, G3(32, 4, 4, 2, 1, 0)),       # split 1: direct bf16 store, EVEN, block loads
    (1, 64, 256, G3(32, 8, 4, 2, 1, 1)),       # split 1, one wide [32, 256] load = the whole K
    (1, 100, 1280, G3(32, 8, 4, 3, 7, 1)),     # KB 40 / split 7 -> PER 6 < KT 8: wide tail + short last split
    (3, 96, 640, G3(64, 2, 4, 2, 4, 1)),       # N % BLOCK_N != 0; PER 5 with KT 2 (tail); wide KT 2
    (6, 40, 384, G3(16, 4, 4, 2, 5, 1)),       # N % 32 != 0, BLOCK_N 16; KB 12 / 5 -> (3, 4)
    (8, 128, 1024, G3(32, 4, 8, 2, 4, 1)),     # EVEN split: KB 32 / 4 -> PER 8 = 2 wide tiles of KT 4
    (8, 128, 1024, G3(32, 4, 8, 2, 4, 0)),     # the same plan through block loads
    (2, 72, 2048, G3(32, 16, 4, 4, 4, 0)),     # KT 16 (block loads only), EVEN: KB 64 / 4 -> PER 16
    (6, 48, 512, G3(16, 1, 4, 2, 16, 1)),      # KT 1 wide, PER 1: every program one block
    (5, 33, 288, G3(32, 2, 4, 2, 9, 0)),       # KB 9, split 9 -> PER 1 with KT 2 (tail every program)
    (4, 256, 1536, G3(128, 8, 4, 2, 6, 1)),    # BLOCK_N > N/2, wide KT 8, EVEN 48 / 6 -> PER 8
]


@pytest.mark.parametrize("M,N,Kd,cfg", CASES3, ids=[f"M{c[0]}-N{c[1]}-K{c[2]}-bn{c[3].block_n}-kt{c[3].kt}-"
                                                    f"s{c[3].split_k}-w{c[3].wide}" for c in CASES3])
def test_gemv3_kernel_equals_split_mirror(K3, M, N, Kd, cfg):
    """The one gemv3 launch == gemv_torch with the same split (bit for bit), every ws slot is its split's K-order
    chain alone, the counters are back at 0, exactly one launch (no reduce kernel), grid / constexprs as planned."""
    xq, xs, w, s = _ops3(M, N, Kd, seed=M * 131 + N + Kd + cfg.split_k)
    n_red = len(K3.t9b_splitk_reduce.launches)
    K3.t9b6_gemv3.launches.clear()
    out, ws, cnt, p = _run3(K3, xq, xs, w, s, cfg)
    assert len(K3.t9b6_gemv3.launches) == 1 and len(K3.t9b_splitk_reduce.launches) == n_red
    la = K3.t9b6_gemv3.launches[0]
    assert la["grid"] == p["grid"] == (-(-N // cfg.block_n), p["NSPLIT"])
    ce = la["constexprs"]
    assert (ce["BLOCK_M"], ce["BLOCK_N"], ce["KT"], ce["WIDE"], ce["EVEN"], ce["SPLIT"], ce["CG"]) == \
        (16, cfg.block_n, cfg.kt, bool(cfg.wide), p["EVEN"], p["SPLIT"], False)
    assert not torch.isnan(out.float()).any()
    assert torch.equal(out, G.gemv_torch(xq, xs, w, s, split_k=cfg.split_k))
    assert not cnt.any(), cnt                      # every tile's counter reset by its last program
    if p["SPLIT"]:
        assert not torch.isnan(ws).any()
        for ks in range(p["NSPLIT"]):
            lo, hi = ks * p["PER"], min((ks + 1) * p["PER"], Kd // 32)
            chain = G.gemv_torch_f32(xq[:, lo * 32:hi * 32].contiguous(), xs[:, lo:hi].contiguous(),
                                     w[:, lo * 32:hi * 32].contiguous(), s[:, lo:hi].contiguous(), 1)
            assert torch.equal(ws[ks].view(torch.int32), chain.view(torch.int32)), ks
    # gemv (t9b_mxfp8_gemv + t9b_splitk_reduce, MMA form) with the same partition: the same bits
    v1 = G.GemvConfig(32, 4, 4, 2, cfg.split_k, 1)
    assert G.numerics_key(v1, N, Kd) == tuple(G.numerics_key3(cfg, N, Kd)) + (1,)
    p1 = K3.launch_plan(v1, M, N, Kd)
    o1 = torch.full((M, N), float("nan"), dtype=torch.bfloat16)
    K3._launch(xq, xs, w, s, o1, v1, torch.zeros(p1["NSPLIT"], M, N) if p1["SPLIT"] else o1)
    assert torch.equal(o1, out)


def test_gemv3_wide_slicing_is_k_order():
    """The WIDE slice expression (reshape [BN*KT, 32] -> trans -> reshape [32, BN, KT], then even/odd halving of the
    last axis) yields block u == weight bytes [n, 32u : 32u + 32] for every KT the kernel uses, in the order the
    kernel consumes them (u0..u7): checked with numpy on index-valued tiles (independent of the shim's dot)."""
    def halves(t, h):
        r = t.reshape(t.shape[0], t.shape[1], h, 2)
        return r[..., 0], r[..., 1]

    BN = 16
    for KT in (1, 2, 4, 8):
        wt2 = _np.arange(BN * KT * 32).reshape(BN, KT * 32)
        t = wt2.reshape(BN * KT, 32).T.reshape(32, BN, KT)
        if KT == 1:
            us = [t]
        elif KT == 2:
            us = list(halves(t, 1))
        elif KT == 4:
            e, o = halves(t, 2)
            u0, u2 = halves(e, 1)
            u1, u3 = halves(o, 1)
            us = [u0, u1, u2, u3]
        else:
            e, o = halves(t, 4)
            ee, eo = halves(e, 2)
            oe, oo = halves(o, 2)
            u0, u4 = halves(ee, 1)
            u2, u6 = halves(eo, 1)
            u1, u5 = halves(oe, 1)
            u3, u7 = halves(oo, 1)
            us = [u0, u1, u2, u3, u4, u5, u6, u7]
        for u, sl in enumerate(us):
            assert _np.array_equal(sl.reshape(32, BN), wt2[:, 32 * u:32 * u + 32].T), (KT, u)


class _ShuffledItertools:
    """Stand-in for triton_shim's `itertools`: the launch's programs run in a seeded random order (the shim normally
    runs them in grid order, so split ks = NSPLIT-1 would always arrive last)."""

    def __init__(self, seed):
        self.rng = _random.Random(seed)
        self.orders = []

    def product(self, *its):
        pids = list(_itertools.product(*its))
        self.rng.shuffle(pids)
        self.orders.append(pids)
        return pids


def test_gemv3_any_arrival_order_same_bits(K3, monkeypatch):
    """Whichever program of a tile arrives last, it folds the partials in split order: 12 random program orders give
    the identical output (== the mirror), and the counters end at 0 every time."""
    M, N, Kd, cfg = 3, 96, 1280, G3(32, 4, 4, 2, 7, 1)
    xq, xs, w, s = _ops3(M, N, Kd, seed=77)
    want = G.gemv_torch(xq, xs, w, s, split_k=7)
    lasts = set()
    for seed in range(12):
        sh = _ShuffledItertools(seed)
        monkeypatch.setattr(triton_shim, "itertools", sh)
        out, _, cnt, p = _run3(K3, xq, xs, w, s, cfg)
        monkeypatch.setattr(triton_shim, "itertools", _itertools)
        assert torch.equal(out, want) and not cnt.any(), seed
        order = sh.orders[-1]
        lasts.add(tuple(max((i, ks) for i, (pn, ks, _) in enumerate(order) if pn == t)[1] for t in range(3)))
    assert len(lasts) > 1     # the last arrival really varied


def test_gemv3_fold_is_split_order_not_arrival_order(K3, monkeypatch):
    """Crafted operands make the fold order visible in the bf16 output: KB 3, split 3 (PER 1), block partials
    2^24, 1, -2^24 for output 0 -> ((0 + 2^24) + 1) - 2^24 = 0 in split order, 1 in reverse. Every program order
    (all 6) gives 0."""
    M, N, Kd = 1, 16, 96
    wb = torch.full((N, Kd), 0x38, dtype=torch.uint8)               # e4m3 1.0
    wb[:, 64:] = 0xB8                                              # e4m3 -1.0: block 2 products -1
    w = wb.view(FP8)
    xq = torch.full((M, Kd), 0x38, dtype=torch.uint8).view(FP8)
    xs = _e8(torch.tensor([[127, 127, 127]], dtype=torch.int32))
    s = _e8(torch.tensor([[146, 122, 146]], dtype=torch.int32))   # 32 * 2^19 = 2^24, 32 * 2^-5 = 1, -2^24
    cfg = G3(16, 1, 4, 2, 3, 0)
    for perm in _itertools.permutations(range(3)):
        class _Fixed:
            @staticmethod
            def product(*its, _perm=perm):
                return [(0, ks, 0) for ks in _perm]

        monkeypatch.setattr(triton_shim, "itertools", _Fixed)
        out, ws, cnt, _ = _run3(K3, xq, xs, w, s, cfg)
        monkeypatch.setattr(triton_shim, "itertools", _itertools)
        assert [float(v) for v in ws[:, 0, 0]] == [2.0 ** 24, 1.0, -(2.0 ** 24)]
        assert float(out[0, 0]) == 0.0 and not cnt.any(), perm   # reverse order would give 1.0


def test_gemv3_only_atomic_is_the_int32_counter(K3, monkeypatch):
    """No fp32 atomics: every atomic the kernel issues is on the int32 arrival counters (one per program)."""
    seen = []
    real = K3.tl.atomic_add

    def spy(ptr, val, mask=None):
        seen.append(ptr.buf.dtype)
        return real(ptr, val, mask)

    monkeypatch.setattr(K3.tl, "atomic_add", spy)
    M, N, Kd, cfg = 2, 64, 640, G3(32, 4, 4, 2, 5, 1)
    xq, xs, w, s = _ops3(M, N, Kd, seed=5)
    _, _, _, p = _run3(K3, xq, xs, w, s, cfg)
    assert len(seen) == p["grid"][0] * p["grid"][1] and set(seen) == {_np.dtype(_np.int32)}


def test_gemv3_rows_past_m_and_columns_past_n_untouched(K3):
    M, N, Kd = 3, 40, 512
    xq, xs, w, s = _ops3(M, N, Kd, seed=3)
    for cfg in (G3(64, 4, 4, 2, 3, 1), G3(64, 4, 4, 2, 1, 0)):
        big = torch.full((M + 2, 2 * N), -7.0, dtype=torch.bfloat16)
        out = big[:M, :N]
        _run3(K3, xq, xs, w, s, cfg, out=out)
        assert torch.equal(big[:M, :N], G.gemv_torch(xq, xs, w, s, split_k=cfg.split_k)), cfg
        assert bool((big[:M, N:] == -7).all()) and bool((big[M:] == -7).all()), cfg


@pytest.mark.parametrize("cfg", [G3(32, 4, 4, 2, 1, 1), G3(32, 4, 4, 2, 4, 1), G3(64, 2, 4, 2, 5, 0)])
def test_gemv3_rows_bit_identical_across_m(K3, cfg):
    """Row-invariance through the kernel body: row m of an M-row call == the same row alone (M = 1) == the same row
    inside every other M <= 8, with the one (N, K) config."""
    N, Kd = 96, 640
    xq, xs, w, s = _ops3(8, N, Kd, seed=21)
    full, _, _, _ = _run3(K3, xq, xs, w, s, cfg)
    for M in (1, 2, 5, 6, 7):
        part, _, _, _ = _run3(K3, xq[:M].contiguous(), xs[:M].contiguous(), w, s, cfg)
        assert torch.equal(part, full[:M]), M
    for m in range(8):
        one, _, _, _ = _run3(K3, xq[m:m + 1].contiguous(), xs[m:m + 1].contiguous(), w, s, cfg)
        assert torch.equal(one, full[m:m + 1]), m


@pytest.mark.parametrize("M,N,Kd,cfg,hd,rd,rows", [
    (1, 128, 256, G3(32, 4, 4, 2, 1, 1), 64, 16, False),     # split 1: epilogue in the program's own store
    (1, 192, 640, G3(64, 2, 4, 2, 4, 0), 96, 32, False),     # split 4: epilogue in the last arrival's store
    (3, 256, 384, G3(32, 4, 8, 3, 3, 1), 128, 64, True),     # 3 rows, each its own position
])
@pytest.mark.parametrize("modes", [(0, 0), (1, 1), (2, 2)])
def test_gemv3_rope_epilogue(K3, M, N, Kd, cfg, hd, rd, rows, modes):
    """gemv3 with ROPE == gemv3 without it, then the RoPE mirror; == gemv's (t9b_mxfp8_gemv / reduce) RoPE output of
    the same split; == the CPU path (gemv_torch + rope_rows_torch)."""
    from runtime.fused_glue import rope_rows_torch
    from test_fused_glue_r2_cpu import _freqs

    xq, xs, w, s = _ops3(M, N, Kd, seed=70 + N)
    plain, _, _, _ = _run3(K3, xq, xs, w, s, cfg)
    if rows:
        fr = torch.stack([torch.view_as_real(_freqs(rd, 100 + 7 * m)).reshape(-1, 2).float() for m in range(M)])
        fr = fr.contiguous()
        want = torch.stack([rope_rows_torch(plain[m], fr[m], hd, rd, modes, inverse=False) for m in range(M)])
    else:
        fr = torch.view_as_real(_freqs(rd, 999)).reshape(-1, 2).float().contiguous()
        want = rope_rows_torch(plain, fr, hd, rd, modes, inverse=False)
    rope = (fr, hd, rd, modes)
    rot, _, cnt, _ = _run3(K3, xq, xs, w, s, cfg, rope=rope)
    assert torch.equal(rot, want) and not cnt.any()
    v1 = G.GemvConfig(32, 4, 4, 2, cfg.split_k, 1)
    p1 = K3.launch_plan(v1, M, N, Kd)
    o1 = torch.full((M, N), float("nan"), dtype=torch.bfloat16)
    K3._launch(xq, xs, w, s, o1, v1, torch.zeros(p1["NSPLIT"], M, N) if p1["SPLIT"] else o1, rope=rope)
    assert torch.equal(o1, rot)
    cpu = G.mxfp8_gemv3(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16), config=cfg, rope=rope)
    assert torch.equal(cpu, rot)


PAIRS3 = [  # (M, NA, NB, K, cfg_a, cfg_b)
    (1, 96, 64, 320, G3(32, 2, 4, 2, 3, 1), G3(64, 4, 4, 2, 1, 0)),     # ragged a split 3, b unsplit
    (3, 64, 64, 512, G3(32, 4, 4, 2, 2, 0), G3(32, 2, 4, 2, 4, 1)),     # NA == NB (shared w1 / w3), both split
    (6, 128, 32, 1024, G3(16, 8, 4, 2, 4, 1), G3(64, 2, 4, 2, 8, 0)),   # b ragged vs a's BLOCK_N, many splits
]


@pytest.mark.parametrize("case", range(len(PAIRS3)))
def test_gemv3_pair_one_launch_equals_singles(K3, case):
    M, NA, NB, Kd, ca, cb = PAIRS3[case]
    xq, xs, wa, sa = _ops3(M, NA, Kd, 10 + case)
    _, _, wb, sb = _ops3(M, NB, Kd, 20 + case)
    singles = [K3.gemv_partials(xq, xs, w_, s_, config=c).clone() for w_, s_, c in ((wa, sa, ca), (wb, sb, cb))]
    n0, n3 = len(K3.t9b6_gemv3_pair.launches), K3.PAIR_STATS["pair3"]
    pa, pb = K3.gemv_pair(xq, xs, wa, sa, wb, sb, configs=(ca, cb))
    assert len(K3.t9b6_gemv3_pair.launches) - n0 == 1 and K3.PAIR_STATS["pair3"] - n3 == 1
    pp = K3.pair_plan3(ca, cb, M, NA, NB, Kd)
    assert K3.t9b6_gemv3_pair.launches[-1]["grid"] == pp["grid"]
    for got, single, cfg, w_, s_ in ((pa, singles[0], ca, wa, sa), (pb, singles[1], cb, wb, sb)):
        assert got.dtype == torch.bfloat16 and not hasattr(got, "nsplit")     # reduced inside: never a SplitK
        assert torch.equal(got, single) and torch.equal(got, G.gemv_torch(xq, xs, w_, s_, split_k=cfg.split_k))
    pc = [t for k, t in K3._COUNTERS.items() if k[2].startswith("v3pair/")]
    assert pc and all(not t.any() for t in pc)


def test_gemv3_dispatch_impl_selection(K3, monkeypatch, _impl_reset):
    """$DSV41_GEMV_IMPL / set_impl route mxfp8_gemv, gemv_partials and gemv_pair to gemv3 (one t9b6 launch, bf16
    results, no reduce launch); an explicit GemvConfig / Gemv3Config wins; a bad name raises; the CPU path (no shim)
    computes the same numbers under either impl."""
    M, N, Kd = 2, 64, 512
    xq, xs, w, s = _ops3(M, N, Kd, seed=9)
    monkeypatch.delenv("DSV41_GEMV_IMPL", raising=False)
    assert G.gemv_impl() == "gemv" and G.describe()["impl"] == "gemv"
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv3")
    assert G.gemv_impl() == "gemv3" and G.describe()["impl"] == "gemv3"
    G.set_impl("gemv")
    assert G.gemv_impl() == "gemv"                    # set_impl wins over the environment
    G.set_impl(None)
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv4")
    with pytest.raises(ValueError):
        G.gemv_impl()
    with pytest.raises(ValueError):
        G.set_impl("gemv4")
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv3")
    # the CPU torch path: identical numbers under both impls (the Mac emulated StaticStep tests rely on it)
    a = G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16))
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv")
    assert torch.equal(a, G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16)))
    # the shim (kernels on CPU): gemv3 launches, not gemv's
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv3")
    monkeypatch.setattr(K3, "CPU_SPLIT_K", 4)         # config None on CPU -> DEFAULT_CONFIG3 with this split
    c3, c1, cr = (len(K3.t9b6_gemv3.launches), len(K3.t9b_mxfp8_gemv.launches),
                  len(K3.t9b_splitk_reduce.launches))
    y = K3.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16))
    r = K3.gemv_partials(xq, xs, w, s)
    assert not hasattr(r, "nsplit") and r.dtype == torch.bfloat16 and K3.materialize(r) is r
    assert (len(K3.t9b6_gemv3.launches) - c3, len(K3.t9b_mxfp8_gemv.launches) - c1,
            len(K3.t9b_splitk_reduce.launches) - cr) == (2, 0, 0)
    want = G.gemv_torch(xq, xs, w, s, split_k=4)
    assert torch.equal(y, want) and torch.equal(r, want)
    assert K3.t9b6_gemv3.launches[-1]["grid"][1] == 4
    # an explicit config decides regardless of the impl (a GemvConfig: gemv's CPU path, no gemv3 launch)
    z = K3.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16), config=G.GemvConfig(32, 4, 4, 2, 4, 1))
    assert torch.equal(z, want) and len(K3.t9b6_gemv3.launches) - c3 == 2
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv")
    K3.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16), config=G3(32, 4, 4, 2, 4, 1))
    assert len(K3.t9b6_gemv3.launches) - c3 == 3
    assert not hasattr(K3.gemv_partials(xq, xs, w, s, config=G3(32, 4, 4, 2, 4, 1)), "nsplit")
    assert hasattr(K3.gemv_partials(xq, xs, w, s, tag="t9b6-v1"), "nsplit")   # gemv: partials left for the consumer


def test_gemv3_configs_per_shape_cache_versioned(tmp_path, monkeypatch):
    """One config per (N, K) for every M (no M in the key); the cache file is gemv3's own (derived from
    $DSV41_GEMV_CACHE, never that file), keys carry the version, a file of another version is ignored, gemv's
    loader ignores gemv3 keys should both share a file; tuning off -> DEFAULT_CONFIG3 with its split resolved."""
    G.reset_configs()
    base = str(tmp_path / "gemv-tune.json")
    monkeypatch.setenv("DSV41_GEMV_CACHE", base)
    monkeypatch.delenv("DSV41_GEMV3_CACHE", raising=False)
    monkeypatch.setenv("DSV41_GEMV_TUNE", "0")
    try:
        assert G.cache3_path() == str(tmp_path / "gemv-tune.gemv3.json") != G.cache_path()
        d = G.get_config3(1280, 5120)
        assert d == G.resolve_config3(G.DEFAULT_CONFIG3, 1280, 5120, G.num_sms()) and d.split_k >= 1
        c = G3(32, 8, 4, 3, 10, 1)
        G.set_config3(2304, 5120, c)
        assert G.get_config3(2304, 5120) == c
        assert G.save_cache3() == G.cache3_path()
        data = json.load(open(G.cache3_path()))
        assert data["impl"] == "gemv3" and data["version"] == G.GEMV3_VERSION == 3
        assert data["configs"]["v3:2304x5120"] == c.as_dict()
        assert not os.path.exists(base)                                   # gemv's file untouched
        G.reset_configs()
        assert G.load_cache3() == 2 and G.get_config3(2304, 5120) == c     # back from the file
        d3 = G.describe()["gemv3"]
        assert d3["configs"]["v3:2304x5120"]["numerics_key"] == [16, 10]
        assert d3["configs"]["v3:2304x5120"]["bytes_per_thread"] == 32 * 8 * 32 // (32 * 4)
        # another version's file: ignored as a whole
        data["version"] = 2
        json.dump(data, open(G.cache3_path(), "w"))
        G.reset_configs()
        assert G.load_cache3() == 0 and G.describe()["gemv3"]["cache_ignored"]
        # gemv3 keys in gemv's own file (someone pointed both at one path): gemv's loader skips them
        json.dump({"configs": {"v3:2304x5120": c.as_dict(),
                               "512x5120/m1": G.GemvConfig(32, 2, 4, 4, 19, 1).as_dict()}}, open(base, "w"))
        G.reset_configs()
        assert G.load_cache() == 1 and "v3:2304x5120" in G.describe()["cache_ignored"]
        monkeypatch.setenv("DSV41_GEMV3_CACHE", str(tmp_path / "own.json"))
        assert G.cache3_path() == str(tmp_path / "own.json")
    finally:
        G.reset_configs()


def test_gemv3_splits_candidates_and_plan():
    sms = 152
    assert G.even_splits(160, 8) == [1, 2, 4, 5, 10, 20] and G.even_splits(40, 8) == [1, 5]
    assert G.even_splits(9, 8) == [] and G.even_splits(72, 8) == [1, 3, 9]
    for KB in (40, 72, 160, 192, 256):
        for kt in (4, 8, 16):
            for ns in G.even_splits(KB, kt):
                per, n2 = G.split_plan(KB, ns)
                assert n2 == ns and per * ns == KB and per % kt == 0
    # the auto split: the smallest EVEN split giving >= 2 x SMs programs (else the largest EVEN one)
    assert G.auto_split_k3(1280, 5120, 64, 8, sms) == 20            # 20 tiles: 20 x 20 = 400 >= 304
    assert G.auto_split_k3(5120, 8192, 64, 8, sms) == 4             # 80 tiles x 4 = 320
    assert G.auto_split_k3(32768, 1280, 64, 8, sms) == 1 and G.auto_split_k3(25600, 6144, 64, 8, sms) == 1
    assert G.auto_split_k3(512, 5120, 64, 8, sms) == 20             # 8 tiles: capped at the largest EVEN split
    assert G.auto_split_k3(7, 288, 16, 8, sms) == 1                 # KB 9: no EVEN split; >= 8 blocks per program
    for N, K in [(1280, 5120), (2304, 5120), (5120, 8192), (25600, 6144), (32768, 1280), (512, 5120), (4096, 1280),
                 (5120, 2304)]:
        cands = G.candidate_configs3(N, K, sms)
        assert cands and len(set(cands)) == len(cands)
        for c in cands:
            G.check_config3(c)
            p = G.launch_plan3(c, 1, N, K)
            assert c.split_k >= 1 and p["tile_bytes"] * c.num_stages <= G.TUNE3_SMEM_CAP
            assert p["bytes_per_thread"] >= G.TUNE3_MIN_BYTES_PER_THREAD and not (c.wide and c.kt > 8)
        assert {c.wide for c in cands} == {0, 1} and {4, 8, 16} <= {c.kt for c in cands}
        assert any(G.launch_plan3(c, 1, N, K)["EVEN"] for c in cands)
    with pytest.raises(ValueError):
        G.check_config3(G3(24, 8, 4, 2, 1, 0))
    with pytest.raises(ValueError):
        G.check_config3(G3(32, 16, 4, 2, 1, 1))                    # wide needs kt <= 8
    with pytest.raises(ValueError):
        G.launch_plan3(G3(32, 8, 4, 2, 0, 0), 1, 64, 256)           # unresolved split
    p = G.launch_plan3(G.resolve_config3(G.DEFAULT_CONFIG3, 1280, 5120, sms), 6, 1280, 5120)
    assert (p["grid"], p["PER"], p["EVEN"], p["BLOCK_M"], p["launches"]) == ((20, 20), 8, True, 16, 1)
    # PER 8 == KT 8: one loop iteration per program, so exactly one 16 KB tile in flight per program (stages cannot
    # prefetch a second); 400 programs -> 6.4 MB requested GPU-wide at once (the whole 6.6 MB weight)
    assert p["tile_bytes"] == 64 * 8 * 32 and p["bytes_per_thread"] == 128 and p["iters"] == 1
    assert p["inflight_bytes"] == 16384 and p["programs"] == 400 and p["grid_inflight_bytes"] == 400 * 16384
    p2 = G.launch_plan3(G.resolve_config3(G.DEFAULT_CONFIG3, 5120, 8192, sms), 1, 5120, 8192)
    assert (p2["PER"], p2["iters"], p2["inflight_bytes"]) == (64, 8, 2 * 16384)      # 3 stages: 2 tiles ahead
    assert G.TUNE3_MIN_BYTES_PER_THREAD >= 128                                         # the handoff's wide-load floor
    # the plan never depends on M: one config, one binary, one split for M = 1..8
    c = G.resolve_config3(G.DEFAULT_CONFIG3, 2304, 5120, sms)
    assert len({tuple(sorted((k, str(v)) for k, v in G.launch_plan3(c, m, 2304, 5120).items()))
                for m in range(1, 9)}) == 1
    monkey_quick = os.environ.get("DSV41_GEMV_TUNE_SPACE")
    os.environ["DSV41_GEMV_TUNE_SPACE"] = "quick"
    try:
        q = G.candidate_configs3(1280, 5120, sms)
        assert 1 <= len(q) <= 8 and {c.wide for c in q} == {0, 1}
        assert G.candidate_configs3(16, 64, sms)                     # tiny shapes still get a candidate
    finally:
        if monkey_quick is None:
            os.environ.pop("DSV41_GEMV_TUNE_SPACE", None)
        else:
            os.environ["DSV41_GEMV_TUNE_SPACE"] = monkey_quick


def test_gemv3_counters_static_and_zero():
    a = G.counters("cpu", 7, "t9b6-test")
    assert a.dtype == torch.int32 and a.shape == (7,) and not a.any()
    assert G.counters("cpu", 7, "t9b6-test") is a and G.counters("cpu", 7, "other") is not a


def test_gemv3_kernel_source_contract(K3):
    """The jit entry points are the shim's / Triton's jit objects, M is excluded from specialization on real Triton
    (the wrapper), and the block body is gemv's MMA=1 arithmetic (same expressions as _t9b4_gemv_tile)."""
    import inspect

    blk = inspect.getsource(K3._t9b6_block.fn)
    v1 = inspect.getsource(K3._t9b4_gemv_tile.fn)
    for expr in ("sbf = _t9b_pow2(tl.load(s_rows + kbc).to(tl.int32))", "saf = _t9b_pow2(tl.load(xs_rows + kbc)"
                 ".to(tl.int32))", "q = part * (saf[:, None] * sbf[None, :])", "tl.where(kb < kb_hi, acc + q, acc)"):
        assert expr in blk and expr in v1, expr
    assert "tl.dot(xt, wtT)" in blk and "tl.dot(xt, tl.trans(wt))" in v1
    assert 'do_not_specialize=["M"]' in inspect.getsource(K3._jit_m_free)
    tile = inspect.getsource(K3._t9b6_tile.fn)
    assert tile.count("tl.atomic_add(") == 1 and "atomic_add(cnt + cnt_idx, 1)" in tile


def test_bench_gates3():
    """bench/gemv_bench.py's T9b-6 gates: >= 3 TB/s or <= 4 µs at every measured M; rows / replay / same-plan
    bit-identity; oracle <= gemv's within ORACLE3_TOL; per-step M=1 total <= 2.0 ms; all_pass only if none is False
    and the speed / step gates were measured; per-step totals for both impls and the saving."""
    mod = _bench_module()

    def r3(us, gbps, rows=True, rep=True, ratio=1.0, same=True, mism=1e-4):
        return {"us": us, "GBps": gbps, "row_invariant": rows, "replay_identical": rep, "err_ratio_vs_gemv": ratio,
                "bitident_same_plan": same, "mismatch": mism}

    def rec(N, K, calls, m1, m6, us1=None, us6=None):
        r = {"shape": [N, K], "calls_per_step": calls, "per_m3": {1: m1, 6: m6}, "per_m": {}}
        if us1 is not None:
            r["per_m"] = {1: {"us_gemv": us1}, 6: {"us_gemv": us6}}
        r["gates3"] = mod.shape_gates3(N, K, r["per_m3"])
        return r

    big = rec(5120, 8192, 40, r3(12.0, 3500.0, rows=None), r3(12.5, 3360.0), 19.4, 20.06)
    small = rec(1280, 5120, 40, r3(3.9, 1700.0, rows=None, ratio=None), r3(3.95, 1690.0, ratio=1.004), 9.13, 9.04)
    g = big["gates3"]
    assert g["fast_ge_3TBps_or_le_4us"] and g["row_invariant"] and g["oracle_le_gemv"] and g["replay_identical"]
    assert small["gates3"]["fast_ge_3TBps_or_le_4us"] is True and small["gates3"]["oracle_le_gemv"] is True
    st = mod.step_totals3([big, small], [1, 6])
    assert st["ms_gemv3_m1"] == round((40 * 12.0 + 40 * 3.9) / 1e3, 3) and st["ms_gemv_m1"] == round(
        (40 * 19.4 + 40 * 9.13) / 1e3, 3)
    assert st["saved_ms_m6"] == round(st["ms_gemv_m6"] - st["ms_gemv3_m6"], 3) and st["target_ms_gemv3_m1"] == 2.0
    ok = mod.overall_gates3([big, small], st)
    assert ok["all_pass"] and not ok["failing"] and ok["step_le_2ms_m1"] is True
    for bad, name in [(rec(1280, 5120, 40, r3(4.2, 1600.0), r3(3.9, 1700.0)), "fast_ge_3TBps_or_le_4us"),
                      (rec(1280, 5120, 40, r3(3.9, 1700.0), r3(3.9, 1700.0, rows=False)), "row_invariant"),
                      (rec(1280, 5120, 40, r3(3.9, 1700.0), r3(3.9, 1700.0, ratio=1.006)), "oracle_le_gemv"),
                      (rec(1280, 5120, 40, r3(3.9, 1700.0, rep=False), r3(3.9, 1700.0)), "replay_identical"),
                      (rec(1280, 5120, 40, r3(3.9, 1700.0), r3(3.9, 1700.0, same=False)), "bitident_gemv_same_plan"),
                      (rec(1280, 5120, 40, r3(3.9, 1700.0, mism=6e-3), r3(3.9, 1700.0)), "mismatch_le_0p5pct")]:
        out = mod.overall_gates3([big, bad], mod.step_totals3([big, bad], [1, 6]))
        assert not out["all_pass"] and out["failing"] == {name: [[1280, 5120]]}, (name, out)
    slow = mod.step_totals3([rec(5120, 8192, 290, r3(7.0, 6000.0), r3(7.0, 6000.0))], [1, 6])
    assert slow["ms_gemv3_m1"] == 2.03 and mod.overall_gates3([], slow)["step_le_2ms_m1"] is False
    assert mod.overall_gates3([], {})["all_pass"] is False                 # nothing measured: not a pass
    assert "ms_gemv_m1" not in mod.step_totals3([rec(1280, 5120, 40, r3(3.9, 1700.0), r3(3.9, 1700.0))], [1])


def test_first_light_report_names_gemv3(tmp_path, monkeypatch, _impl_reset):
    """The box selects gemv3 as `DSV41_GEMV_IMPL=gemv3` + `--fp8-impl gemv` (bench/first_light.py's --fp8-impl choices
    and every `fp8_impl == "gemv"` route stay as they are): the first-light report's fp8_gemv section then names the
    kernel path that ran ("gemv3") and carries gemv3's versioned config table."""
    fl = _first_light()
    monkeypatch.setenv("DSV41_GEMV_IMPL", "gemv3")
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--moe-impl", "dg-masked", "--fp8-impl",
                                      "gemv", "--fixture", FIXTURE, "--only", "prose", "--reps", "1", "--max-new",
                                      "4", "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    gen = _gen(mode="graph", moe_impl=a.moe_impl, fp8_impl=a.fp8_impl)
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["generator"]["fp8_impl"] == "gemv"
    assert report["fp8_gemv"]["impl"] == "gemv3" and report["fp8_gemv"]["gemv3"]["version"] == G.GEMV3_VERSION
