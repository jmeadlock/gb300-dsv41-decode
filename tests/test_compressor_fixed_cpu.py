"""Mac CPU tests of T9b-8: the ratio-2 attention compressor's wkv / wgate in the T9b-5 fixed order under
`--fp32-impl fixed` ($DSV41_FP32_IMPL; no new flag).

  .venv/bin/python -m pytest -q tests/test_compressor_fixed_cpu.py

What is proven here (the compiled GPU arithmetic and the speed are tests/gpu/test_compressor_fixed_gpu.py's job):
  * the order: runtime/static_kernels.py t9b8_comp_fp32 (site "comp", the T9b-5 body _t9b5_dot) executed program by
    program under tests/triton_shim.py == fused_glue_r2.fixed_dot_torch bit for bit, for R in {1, 2, 5, 6, 8, 9}, every
    launch config, N not a multiple of BN; a bf16-stored weight gives the bits of its fp32 widening (exact);
  * row invariance by construction: row r of an R-row call == the row alone (the order is a function of KB only);
  * the reference's math: model.py:444-448 declares wkv / wgate fp32 at ratio > 1 and 463-465 computes
    F.linear(x.float(), W) -- the products are exact, only the sum order is ours; compressor_decode_fixed == the
    reference Compressor.forward decode branch with its two Linears swapped for the fixed dot (pooled AND non-pooled
    positions: latent or None, kv_state / score_state), bit for bit;
  * R = 6 vs R = 1 at the compressor: the 6-row projection + pool_ratio2_rows + state commit (StaticStep's
    _compressor_rows) == six 1-row compressor_decode_fixed calls -- the pooled latents of the completed groups and the
    state at both P0 parities;
  * the call sites: under `cublas` (the default) the 1-row step's compressor is the reference call and its logits are
    byte-identical to the pre-T9b-8 code path; under `fixed` every decode call site (StaticStep 1-row, 6-row, and
    dspark_loop.VerifyStep's deferred commit) launches the fixed kernel twice per ratio-2 layer per step, the eager
    reference / prefill never; the 6-row probe (bench/row_divergence_probe.py, emulated) finds every tap and the
    logits of every row equal to the 1-row step under `fixed`.
"""
from __future__ import annotations

import copy
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
from test_static_rows_cpu import _cpu_default_device, _prefill, _tf_tokens, emu  # noqa: E402,F401 (fixtures)


@pytest.fixture(scope="module")
def SK():
    """runtime/static_kernels.py imported with the shim as triton (a private copy)."""
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"),
                                              "static_kernels_t9b8_under_shim")


@pytest.fixture
def fixed_env(monkeypatch):
    def set_(v):
        monkeypatch.setenv(R2.FP32_IMPL_ENV, v)
    return set_


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _case(R: int, N: int, K: int, seed: int):
    """The compressor's operands: x = x.float() of a bf16 activation, W = bf16 checkpoint values widened to fp32."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(R, K, generator=g, device="cpu") * 2).to(torch.bfloat16).float()
    w = (torch.randn(N, K, generator=g, device="cpu") * 0.05).to(torch.bfloat16).float()
    return x, w


def _shim_dot(SK, x, w, kb, cfg=None):
    out = torch.full((x.size(0), w.size(0)), float("nan"), dtype=torch.float32, device="cpu")
    SK.fixed_dot(x, w, out, kb=kb, site="comp", cfg=cfg)
    return out


CFGS = [None, (1, 2, 8, 4, 1), (8, 2, 8, 4, 1), (2, 4, 3, 4, 1), (4, 1, 5, 4, 2), (8, 8, 1, 8, 1)]


# ------------------------------------------------------------------------------------------------ the kernel
def test_comp_site_registration_and_kb(SK):
    assert R2.fixed_kb("comp", 5120) == 64 and R2.fixed_kb("comp", 256) == 64 and R2.fixed_kb("comp", 96) == 32
    assert SK.T9B5_KERNELS["comp"] is SK.t9b8_comp_fp32
    assert SK.t9b5_rows_per_program("comp", 6) == 8 and SK.t9b5_rows_per_program("comp", 1) == 1
    assert set(SK.T9B5_CFG["comp"]) == {1, 2, 4, 8}
    x, w = _case(2, 6, 128, 1)
    k = SK.t9b8_comp_fp32
    n0 = len(k.launches)
    _shim_dot(SK, x, w, 64)
    assert len(k.launches) == n0 + 1 and k.launches[-1]["enable_fp_fusion"] is False
    assert R2.describe_fp32()["kb"]["comp"] == 64


def test_comp_kernel_under_shim_equals_definition_and_is_row_invariant(SK):
    N, K = 37, 320                      # N not a multiple of BN; K = 5 tiles of KB = 64 (U = 8, 3 do not divide)
    kb = R2.fixed_kb("comp", K)
    assert kb == 64
    x9, w = _case(9, N, K, 7)
    want = R2.fixed_dot_torch(x9, w, kb)
    singles = [_shim_dot(SK, x9[i:i + 1], w, kb) for i in range(9)]
    for i in range(9):
        assert torch.equal(_bits(singles[i]), _bits(want[i:i + 1])), i
    ref = torch.cat(singles, 0)
    for R in (1, 2, 5, 6, 8, 9):
        for cfg in CFGS:
            got = _shim_dot(SK, x9[:R].contiguous(), w, kb, cfg)
            assert not torch.isnan(got).any(), (R, cfg)
            assert torch.equal(_bits(got), _bits(ref[:R])), (R, cfg)
    # replay
    assert torch.equal(_shim_dot(SK, x9[:6].contiguous(), w, kb), _shim_dot(SK, x9[:6].contiguous(), w, kb))
    # a bf16-stored weight (widened in registers) == its fp32 widening, bit for bit
    got_bf = _shim_dot(SK, x9[:6].contiguous(), w.to(torch.bfloat16), kb)
    assert torch.equal(_bits(got_bf), _bits(ref[:6]))
    # the products are the reference's: the fixed sum is within fp32 rounding of cuBLAS-class F.linear
    lin = torch.nn.functional.linear(x9, w)
    assert (ref - lin).abs().max().item() < 1e-4 * lin.abs().max().item()


# ------------------------------------------------------------------------------------------------ the reference
def _compressor(seed=0):
    from runtime.refmodel import import_reference

    R = import_reference(cpu_stubs=True)
    args = R.ModelArgs(dim=128, head_dim=32, rope_head_dim=8, compress_ratios=(0, 2, 0), max_batch_size=1,
                       max_seq_len=256, n_layers=2)
    torch.manual_seed(seed)
    c = R.Compressor(args, 1)
    for p in c.parameters():
        p.data.normal_()
    assert c.compress_ratio == 2 and c.wkv.weight.dtype == torch.float32 and c.wgate.weight.dtype == torch.float32
    with torch.no_grad():           # the loader's widening: bf16 checkpoint values in an fp32 parameter
        c.wkv.weight.copy_(c.wkv.weight.to(torch.bfloat16).float() * 0.1)
        c.wgate.weight.copy_(c.wgate.weight.to(torch.bfloat16).float() * 0.1)
    return c


class _FixedLinear(torch.nn.Module):
    """A Compressor Linear evaluated in the T9b-5 order (the definition): what `--fp32-impl fixed` means for it."""

    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight

    def forward(self, x):
        K = x.size(-1)
        return R2.fixed_dot_torch(x.reshape(-1, K), self.weight, R2.fixed_kb("comp", K)).view(*x.shape[:-1], -1)


@pytest.mark.parametrize("L", [5, 6])
def test_decode_fixed_equals_reference_branch_with_fixed_linears(fixed_env, L):
    """compressor_decode_fixed == model.py:458-485 (decode) with wkv / wgate swapped for the fixed dot, token by
    token, pooled and non-pooled positions: latent (or None) and both state tensors bit-equal."""
    fixed_env("fixed")
    ref = _compressor()
    ours = copy.deepcopy(ref)
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(1, L + 12, 128, generator=g, device="cpu").to(torch.bfloat16)
    with torch.inference_mode():
        ref(x[:, :L], 0)                    # prefill: the reference (cuBLAS-class) path in both
        ours(x[:, :L], 0)
        ref.wkv, ref.wgate = _FixedLinear(ref.wkv), _FixedLinear(ref.wgate)
        assert R2.compressor_fixed(ours)
        n_pool = 0
        for p in range(L, L + 12):
            a = ref(x[:, p:p + 1], p)
            b = R2.compressor_decode_fixed(ours, x[:, p:p + 1], p)
            assert (a is None) == (b is None) == ((p + 1) % 2 != 0), p
            if a is not None:
                n_pool += 1
                assert a.dtype == b.dtype == torch.bfloat16 and torch.equal(a, b), p
            assert torch.equal(_bits(ref.kv_state), _bits(ours.kv_state)), p
            assert torch.equal(ref.score_state, ours.score_state), p
        assert n_pool == 6
    with pytest.raises(ValueError):
        R2.compressor_decode_fixed(ours, x[:, :1], 0)


@pytest.mark.parametrize("L", [5, 6, 9, 10])   # prefill length -> first P0 parity
def test_rows6_projection_and_pooling_equal_six_one_row_calls(fixed_env, L):
    """THE row invariance at the compressor, both parities: the 6-row step's form (one compressor_proj_fixed over the
    6 rows, pool_ratio2_rows, the state commit -- StaticStep._compressor_rows) == the 1-row step's form
    (compressor_decode_fixed per token): the raw projections per row, every pooled latent, and the state."""
    from runtime.static_step import pool_ratio2_rows, ratio2_offset, ratio2_state_rows

    fixed_env("fixed")
    one = _compressor(1)
    six = copy.deepcopy(one)
    g = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn(1, L + 6 * 4, 128, generator=g, device="cpu").to(torch.bfloat16)
    with torch.inference_mode():
        one(x[:, :L], 0)
        six(x[:, :L], 0)
        p0 = L
        for _ in range(4):
            xs = x[:, p0:p0 + 6]
            kv6, sc6 = R2.compressor_proj_fixed(six, xs.float())
            for i in range(6):              # the projections: row i of 6 == row i alone
                kv1, sc1 = R2.compressor_proj_fixed(six, xs[:, i:i + 1].float())
                assert torch.equal(_bits(kv6[:, i:i + 1]), _bits(kv1)) and torch.equal(sc6[:, i:i + 1], sc1), i
            P = torch.tensor([p0], device="cpu")
            pooled = pool_ratio2_rows(six.kv_state[:1, :1], six.score_state[:1, :1], kv6, sc6,
                                      ratio2_offset(P) + torch.arange(6, device="cpu"))
            sidx = ratio2_state_rows(P, 6)
            six.kv_state[:1].copy_(kv6.index_select(1, sidx))
            six.score_state[:1].copy_(sc6.index_select(1, sidx))
            got = six.norm(pooled.to(xs.dtype))
            want = [y for y in (R2.compressor_decode_fixed(one, x[:, p0 + i:p0 + i + 1], p0 + i) for i in range(6))
                    if y is not None]
            assert len(want) == 3 and torch.equal(got, torch.cat(want, 1)), (L, p0)
            assert torch.equal(six.kv_state, one.kv_state) and torch.equal(six.score_state, one.score_state), p0
            p0 += 6


# ------------------------------------------------------------------------------------------------ the call sites
def _n_ratio2(model) -> int:
    return sum(1 for l in model.layers if l.attn.is_kv_source and l.attn.compressor is not None
               and l.attn.compressor.compress_ratio > 1)


def _count(monkeypatch, model):
    """Counters: fixed-order "comp" launches, and F.linear calls on the ratio-2 compressors' weights (cuBLAS)."""
    import torch.nn.functional as F

    seen = {"comp": 0, "linear_comp": 0}
    ptrs = set()
    for l in model.layers:
        c = l.attn.compressor
        if c is not None and c.compress_ratio > 1:
            ptrs |= {c.wkv.weight.data_ptr(), c.wgate.weight.data_ptr()}
    real_fixed, real_lin = R2.launch_fixed_dot, F.linear

    def fixed(x, w, out, *, site, scale=None):
        if site == "comp":
            seen["comp"] += 1
            assert w.data_ptr() in ptrs and x.dtype == torch.float32
        return real_fixed(x, w, out, site=site, scale=scale)

    def lin(x, w, b=None):
        if w.data_ptr() in ptrs:
            seen["linear_comp"] += 1
        return real_lin(x, w, b)

    monkeypatch.setattr(R2, "launch_fixed_dot", fixed)
    monkeypatch.setattr(F, "linear", lin)
    return seen


def _run1(model, pool, ids, toks, n):
    from runtime.static_step import StaticStep

    st = StaticStep(model, pool, max_pos=255, rows=1, _allow_cpu=True)
    _prefill(model, ids)
    with torch.inference_mode():
        out = torch.stack([st.step(toks[i], len(ids) + i, graph=False)[0].float().clone() for i in range(n)])
    st.detach()
    return out


def test_static1_cublas_is_the_old_path_and_fixed_launches(emu, monkeypatch, fixed_env):
    """1-row step: under `cublas` the compressor is the reference call (no fixed launch, F.linear on its weights)
    and the logits are byte-identical to the pre-T9b-8 code path (`_compressor_one` replaced by the old line,
    `comp(x, start_pos)`); under `fixed` two fixed launches per ratio-2 layer per step, no F.linear on them, and the
    logits move only through the sum order (same argmax here, tiny |d|) -- the T9b-5 precedent for the head / mixes."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini")
    n2 = _n_ratio2(model)
    assert n2 >= 1
    ids, toks = _tf_tokens(17, 6, 7)    # positions 17..22: pooled at 17, 19, 21 and not at 18, 20, 22
    seen = _count(monkeypatch, model)
    outs = {}
    for fp32 in ("cublas", "fixed"):
        fixed_env(fp32)
        for k in seen:
            seen[k] = 0
        outs[fp32] = _run1(model, pool, ids, toks, 6)
        if fp32 == "fixed":
            assert seen == {"comp": 6 * 2 * n2, "linear_comp": 1 * 2 * n2}, seen   # the prefill: reference, once
        else:
            assert seen == {"comp": 0, "linear_comp": 7 * 2 * n2}, seen
    fixed_env("cublas")
    monkeypatch.setattr(StaticStep, "_compressor_one", lambda self, comp, x, start_pos: comp(x, start_pos))
    old = _run1(model, pool, ids, toks, 6)
    assert torch.equal(_bits(old), _bits(outs["cublas"])), "cublas: the 1-row logits moved"
    d = (outs["fixed"] - outs["cublas"]).abs().max().item()
    print(f"  1-row fixed vs cublas logits: max |d| {d:.3g}, argmax equal "
          f"{torch.equal(outs['fixed'].argmax(-1), outs['cublas'].argmax(-1))}")
    assert d < 1e-2 and torch.equal(outs["fixed"].argmax(-1), outs["cublas"].argmax(-1)), d


@pytest.mark.parametrize("fp32", ["fixed", "cublas"])
def test_static6_and_verify_step_call_sites(emu, monkeypatch, fixed_env, fp32):
    """6-row StaticStep and VerifyStep with the deferred ratio-2 commit (the loop's form): `fixed` -> 2 fixed
    launches per ratio-2 layer per step, no F.linear on those weights; `cublas` -> none, F.linear as before. The two
    forms leave the same pooled latents (the step's logits) and VerifyStep's commit buffers hold the projections."""
    from runtime.dspark_loop import VerifyStep
    from runtime.static_step import StaticStep

    from test_static_rows_cpu import _static_run

    fixed_env(fp32)
    model, pool, _ = emu("mini4")
    n2 = _n_ratio2(model)
    ids, toks = _tf_tokens(18, 12, 41)
    hid = (len(model.layers) - 1,)
    kw = dict(max_pos=255, rows=6, hidden_layers=hid, _allow_cpu=True)
    seen = _count(monkeypatch, model)
    st = StaticStep(model, pool, **kw)
    for k in seen:
        seen[k] = 0
    a_lg, _, a_s = _static_run(model, st, ids, toks, 2)
    st.detach()
    want = {"comp": 2 * 2 * n2, "linear_comp": 2 * n2} if fp32 == "fixed" else \
        {"comp": 0, "linear_comp": 2 * n2 + 2 * 2 * n2}
    assert seen == want, (fp32, seen)
    vs = VerifyStep(model, pool, **kw)
    vs.defer_ratio2_commit = True
    for k in seen:
        seen[k] = 0
    _prefill(model, ids)
    with torch.inference_mode():
        lg = vs.step(toks[:6], len(ids), graph=False)
    assert seen == ({"comp": 2 * n2, "linear_comp": 2 * n2} if fp32 == "fixed" else
                    {"comp": 0, "linear_comp": 2 * n2 + 2 * n2}), (fp32, seen)   # prefill + the step
    assert all(torch.equal(lg[j], a_lg[j]) for j in range(6))      # the same first step, deferred commit or not
    vs.detach()


def test_rowdiv_probe_fixed_all_equal_emulated(emu, monkeypatch):
    """bench/row_divergence_probe.py on the emulation under $DSV41_FP32_IMPL=fixed, teacher-forced and with
    rejections: every tap (cache.ckv at the pooled rows included) and the logits of every row equal to the 1-row step;
    the compressor projections ran the fixed order (launches counted), not F.linear. (The emulation's own batched
    F.linear / einsum are made row-wise as in tests/test_row_divergence_cpu.py; the fixed dot needs no help.)"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("row_divergence_probe_t9b8",
                                                  os.path.join(REPO, "bench", "row_divergence_probe.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    model, pool, _ = emu("mini4")
    r2 = [i for i, l in enumerate(model.layers) if l.attn.compressor is not None and l.attn.compressor.compress_ratio > 1]
    assert r2
    seen = _count(monkeypatch, model)
    with P.env(DSV41_FP32_IMPL="fixed"):
        for plan in ((5, 5, 5), (5, 0, 2, 5, 1)):
            for k in seen:
                seen[k] = 0
            rep = P.run_probe(model, pool, impls={}, plan=plan, L=18, static_kw={"_allow_cpu": True}, log=print,
                              rowwise="all", row_invariant=True)
            for line in P.report_lines(rep):
                print(line)
            assert rep["summary"]["origin"] is None, rep["summary"]["origin"]
            assert rep["summary"]["rows_compared"] == sum(n + 1 for n in plan)
            assert "cache.ckv" in set(rep["rows"][0]["sites"])
            # both passes' decode steps ran the fixed order on every ratio-2 layer (the prefills are the reference's)
            n_steps = len(plan) + sum(n + 1 for n in plan)       # 6-row steps + the 1-row steps over the same rows
            assert seen["comp"] >= 2 * len(r2) * n_steps, (seen, n_steps)


def test_gpu_rowdiv_census_gate_dry_run_on_cpu(monkeypatch, capsys):
    """tests/gpu/test_row_divergence_gpu.py's real-width census test on the CPU (fixed_dot_torch stands in for the
    kernel): the T9b-8 row 'compressor wkv / wgate, ratio 2' reads row-invariant yes, replay equal, and the census
    table lists it under row-invariant."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_rowdiv_t9b8", os.path.join(REPO, "tests", "gpu",
                                                                                  "test_row_divergence_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "_need_gpu", lambda: None)
    real = G.P.real_width_census
    monkeypatch.setattr(G.P, "real_width_census", lambda dev, **k: real(dev, W=(256,), flashmla=False))
    G.REPORT.clear()
    G.test_real_width_census_reported()
    out = capsys.readouterr().out
    inv = next(l for l in out.splitlines() if l.strip().startswith("row-invariant:"))
    assert "T9b-8 fixed fp32" in inv, inv
    assert any("compressor wkv / wgate, ratio 2" in r and "row-invariant yes" in r for r in G.REPORT), G.REPORT
    print(out)
