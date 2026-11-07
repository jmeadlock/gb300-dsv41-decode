"""Mac CPU tests of T9b-1 (runtime/gemv_mxfp8.py, `--fp8-impl gemv`): `.venv/bin/python -m pytest -q tests/`.

What a CPU can check, and does:
  * the e8m0 scale arithmetic: `pair_scale` = exp2(sa + sb - 254) built from its bits, against the fp32 product of
    torch's own e8m0 decodes on ALL 65,536 code pairs (normal, subnormal, underflow-to-0, inf, NaN), and against the
    reference kernel's two-step `(C * sa) * sb` (kernel.py:269) on random partials;
  * block indexing: output row n reads weight scale s[n // 32, k] (no expansion), K block k reads activation scale
    xs[m, k]; N not a multiple of 32 (the last, partial scale row);
  * the GEMV algorithm (`gemv_torch`, the kernel's arithmetic in torch) == a block-faithful emulation of the reference
    fp8_gemm (per-32 block partial, `acc += C * sa * sb` in K order) bit for bit, and within bf16 rounding of the exact
    float64 product; rows independent for M = 1..8; the operand contract; configs/cache plumbing;
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


def test_rows_independent_m1_to_8():
    xq, xs, w, s = _operands(8, 48, 192, seed=9)
    full = G.mxfp8_gemv(xq, xs, w, s, torch.empty(8, 48, dtype=torch.bfloat16))
    for M in range(1, 9):
        part = G.mxfp8_gemv(xq[:M].contiguous(), xs[:M].contiguous(), w, s, torch.empty(M, 48, dtype=torch.bfloat16))
        assert torch.equal(part, full[:M])
    for m in range(8):
        one = G.mxfp8_gemv(xq[m:m + 1].contiguous(), xs[m:m + 1].contiguous(), w, s,
                           torch.empty(1, 48, dtype=torch.bfloat16))
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
        assert G.get_config(512, 5120, 1) == G.DEFAULT_CONFIG          # tuning off -> default, remembered
        c = G.GemvConfig(64, 4, 8, 1)
        G.set_config(32768, 1280, 6, c)                                 # key (N, K, block_m(6) = 8)
        assert G.get_config(32768, 1280, 7) == c and G.config_key(32768, 1280, 6) == (32768, 1280, 8)
        assert G.save_cache() == path
        data = json.load(open(path))
        assert data["configs"]["32768x1280/m8"] == c.as_dict()
        G.reset_configs()
        assert G.get_config(32768, 1280, 8) == c                        # loaded from the file on first miss
        d = G.describe()
        assert d["cache_loaded"] == path and d["configs"]["32768x1280/m8"] == c.as_dict() and d["tune"] is False
        assert all(cfg.block_n & (cfg.block_n - 1) == 0 and cfg.ku & (cfg.ku - 1) == 0 for cfg in G.CANDIDATES)
        assert G.DEFAULT_CONFIG in G.CANDIDATES
    finally:
        G.reset_configs()


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
