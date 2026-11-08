"""Mac CPU tests of M2-T17 (docs/HANDOFF-M2-T17-BF16-HEAD.md): the head weight kept bf16 in HBM, read by the fixed-order
kernel and widened in registers.

  .venv/bin/python -m pytest -q tests/test_head_bf16_cpu.py tests/test_head_bf16_loader_cpu.py

What is proven here (the compiled kernel's bits and its speed are tests/gpu/test_head_bf16_gpu.py's job):
  * runtime/static_kernels.py t9b5_head_fp32 executed program by program under the shim with a bf16 weight ==
    the same call with the weight widened to fp32 == fused_glue_r2.fixed_dot_torch, bit for bit (R in {1, 2, 5, 6, 8, 9},
    N not a multiple of BN, every launch config, the bf16 default table included), and row-invariant;
  * the launch tables: a bf16 weight at site "head" takes the "head_bf16" table / env key, the fp32 weight's table is
    untouched; the kernel contract refuses a dtype it does not widen exactly;
  * the call sites: a 6-row StaticStep whose head weight was narrowed to bf16 (exactly: the values are bf16 checkpoint
    values) gives logits bit-equal to the fp32-stored head's, under both $DSV41_FP32_IMPL settings (a bf16 head is a
    fixed-order head by definition: no F.linear on it, one fixed launch per step), glue fused and ref;
  * install_bf16_head: the module's forward == head_fixed_w on the fp32 copy, last-position and full-logits forms,
    R = 1 rows equal the R = 6 rows.
The loader / draft side (build_and_load(head_dtype=...), the Markov head, DraftStep vs forward_spec with bf16 heads):
tests/test_head_bf16_loader_cpu.py.
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


@pytest.fixture(scope="module")
def SK():
    """runtime/static_kernels.py imported with the shim as triton (a private copy)."""
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"),
                                              "static_kernels_m2t17_under_shim")


def _head_case(R: int, N: int, K: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(R, K, generator=g, device="cpu") * 2).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g, device="cpu") * 0.05).to(torch.bfloat16)
    return x, w


def _shim(SK, x, w, kb, cfg=None):
    out = torch.full((x.size(0), w.size(0)), float("nan"), dtype=torch.float32, device="cpu")
    SK.fixed_dot(x, w, out, kb=kb, site="head", cfg=cfg)
    return out


def _eq(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ------------------------------------------------------------------------------------------------ the kernel
def test_bf16_weight_equals_widened_fp32_weight_under_shim(SK):
    N, K = 41, 256                                  # N not a multiple of BN; KB 64 -> 4 K tiles
    kb = R2.fixed_kb("head", K)
    x9, wb = _head_case(9, N, K, 3)
    wf = wb.float()
    want = R2.fixed_dot_torch(x9, wb, kb)
    assert _eq(want, R2.fixed_dot_torch(x9, wf, kb))          # the definition itself: bf16 and widened agree
    cfgs = [None, SK.t9b5_config("head", 8, torch.bfloat16), (1, 4, 2, 4, 1), (2, 8, 3, 4, 1), (8, 16, 8, 8, 1),
            (4, 2, 5, 4, 1)]
    singles = [_shim(SK, x9[i:i + 1], wb, kb) for i in range(9)]
    for R in (1, 2, 5, 6, 8, 9):
        x = x9[:R].contiguous()
        for cfg in cfgs:
            got_b = _shim(SK, x, wb, kb, cfg)
            got_f = _shim(SK, x, wf, kb, cfg)
            assert not torch.isnan(got_b).any(), (R, cfg)
            assert _eq(got_b, got_f), (R, cfg, "bf16-read != fp32-stored read")
            assert _eq(got_b, want[:R]), (R, cfg, "!= fixed_dot_torch")
            assert _eq(got_b, torch.cat(singles[:R], 0)), (R, cfg, "not row-invariant")
    # fp32 activations (the oracle's x.float()) take the same path
    assert _eq(_shim(SK, x9[:6].float().contiguous(), wb, kb), want[:6])


def test_launch_tables_and_contract(SK):
    t = SK.T9B5_CFG
    assert set(t) == {"head", "mix", "head_bf16", "comp"} and set(t["head_bf16"]) == set(t["head"])  # comp: T9b-8
    for rb in t["head"]:       # a bf16 tile is half the bytes of an fp32 one: U doubles, BN / warps / stages are kept
        bn, u, nw, ns = t["head"][rb]
        assert t["head_bf16"][rb] == (bn, 2 * u, nw, ns), rb
    assert SK.t9b5_config("head", 6) == (8,) + t["head"][8]
    assert SK.t9b5_config("head", 6, torch.float32) == (8,) + t["head"][8]
    assert SK.t9b5_config("head", 6, torch.bfloat16) == (8,) + t["head_bf16"][8]
    assert SK.t9b5_config("head", 1, torch.bfloat16) == (1,) + t["head_bf16"][1]
    assert SK.t9b5_config("head_bf16", 3) == (4,) + t["head_bf16"][4]
    assert SK.t9b5_rows_per_program("head_bf16", 6) == 8 and SK.t9b5_rows_per_program("head_bf16", 1) == 1
    assert SK.t9b5_config("mix", 4, torch.bfloat16) == (1,) + t["mix"][1]    # the mixes never take the head table
    x, wb = _head_case(2, 3, 64, 1)
    out = torch.empty(2, 3, device="cpu")
    for bad_w in (wb.to(torch.float16), wb.view(torch.int16)):
        with pytest.raises(ValueError, match="fp32 or bf16"):
            SK.fixed_dot(x, bad_w, out, kb=64, site="head")
    with pytest.raises(ValueError, match="site"):
        SK.fixed_dot(x, wb, out, kb=64, site="head_bf16")     # a table key, not a kernel
    k = SK.t9b5_head_fp32
    n0 = len(k.launches)
    SK.fixed_dot(x, wb, out, kb=64, site="head")
    assert len(k.launches) == n0 + 1 and k.launches[-1]["enable_fp_fusion"] is False


def test_env_override_key_for_the_bf16_table(SK, monkeypatch):
    monkeypatch.setenv(SK.T9B5_CFG_ENV, "head_bf16:8=4,3,4,2;head:8=2,1,4,1")
    assert SK.t9b5_config("head", 6, torch.bfloat16) == (8, 4, 3, 4, 2)
    assert SK.t9b5_config("head", 6, torch.float32) == (8, 2, 1, 4, 1)
    assert SK.t9b5_config("head", 1, torch.bfloat16) == (1,) + SK.T9B5_CFG["head_bf16"][1]   # other RB untouched


# ------------------------------------------------------------------------------------------------ install_bf16_head
def _toy_head(V=72, D=128, seed=8):
    from torch import nn

    class H(nn.Module):
        pass

    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(V, D, generator=g, device="cpu") * 0.1).to(torch.bfloat16)
    h = H()
    h.weight = nn.Parameter(w, requires_grad=False)
    return h, w.float(), g


def test_install_bf16_head_forward_equals_fixed_on_the_widened_copy():
    h, w32, g = _toy_head()
    x = torch.randn(1, 6, 128, generator=g, device="cpu").to(torch.bfloat16)
    with pytest.raises(ValueError, match="not bf16"):
        R2.install_bf16_head(_fp32_head(w32))
    R2.install_bf16_head(h)
    assert h._m2t17_bf16_head and R2.head_weight_is_bf16(h)
    full = h(x, full_logits=True)
    assert full.shape == (1, 6, 72) and full.dtype == torch.float32
    assert _eq(full, R2.head_fixed_w(w32, x, True))
    for i in range(6):                                  # last-position form; R = 1 row == the R = 6 row
        one = h(x[:, i:i + 1])
        assert one.shape == (1, 72) and _eq(one[0], full[0, i])
    assert _eq(h(x), full[:, -1])
    # the arithmetic is the reference's F.linear(x.float(), W) up to the documented sum order (tiny, not bitwise)
    ref = torch.nn.functional.linear(x.float(), w32)
    assert (ref - full).abs().max().item() < 1e-4
    # a [b, rank] input (the Markov head: head(embed, full_logits=True)) -> [b, V]
    e = torch.randn(1, 128, generator=g, device="cpu").to(torch.bfloat16)
    assert h(e, full_logits=True).shape == (1, 72)


def _fp32_head(w32):
    from torch import nn

    class H(nn.Module):
        pass

    h = H()
    h.weight = nn.Parameter(w32.clone(), requires_grad=False)
    return h


# ------------------------------------------------------------------------------------------------ the call sites
from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens, emu,  # noqa: E402,F401
                                  rows_dg, rows_glue)


def _narrow_head(model):
    """What build_and_load(head_dtype="bf16") leaves: the head weight bf16 (exact: it holds bf16 checkpoint values)
    and the fixed-order forward installed."""
    w = model.head.weight
    nb = w.detach().to(torch.bfloat16)
    assert torch.equal(nb.float(), w.detach())
    model.head.weight = torch.nn.Parameter(nb, requires_grad=False)
    R2.install_bf16_head(model.head)


def _sites(monkeypatch, model):
    """Counters: fixed head launches, and F.linear calls on the head weight (cuBLAS's path)."""
    import torch.nn.functional as F

    seen = {"head": 0, "linear_head": 0}
    real_fixed = R2.launch_fixed_dot
    head_ptr = model.head.weight.data_ptr()
    real_lin = F.linear

    def fixed(x, w, out, *, site, scale=None):
        if site == "head":
            seen["head"] += 1
            assert w.dtype == torch.bfloat16 and w.data_ptr() == head_ptr
        return real_fixed(x, w, out, site=site, scale=scale)

    def lin(x, w, b=None):
        if w.data_ptr() == head_ptr:
            seen["linear_head"] += 1
        return real_lin(x, w, b)

    monkeypatch.setattr(R2, "launch_fixed_dot", fixed)
    monkeypatch.setattr(F, "linear", lin)
    return seen


@pytest.mark.parametrize("fp32,moe_impl,glue", [("fixed", "ref", "fused"), ("fixed", "dg-inplace", "fused"),
                                                ("cublas", "ref", "fused"), ("cublas", "ref", "ref")])
def test_static6_bf16_head_logits_equal_the_fp32_stored_head(rows_glue, monkeypatch, fp32, moe_impl, glue):
    """6-row StaticStep: head fp32-stored (fixed order under `fixed`; cuBLAS under `cublas`, which is a different
    arithmetic and so is only compared under `fixed`) vs the same model with the head narrowed to bf16. Under `fixed`
    the 6-row logits are bit-equal; under `cublas` the bf16 head still runs the fixed kernel (cuBLAS cannot read it:
    one fixed launch per step, no F.linear on the head weight) and the rest of the step is the cublas-mixes step."""
    from runtime.static_step import StaticStep

    monkeypatch.setenv(R2.FP32_IMPL_ENV, fp32)
    model, pool, _, _ = rows_glue("mini4")
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 41)
    kw = dict(max_pos=255, rows=6, hidden_layers=hid, moe_impl=moe_impl, _allow_cpu=True)
    if glue == "fused":
        kw["glue_impl"] = "fused"
    base = None
    if fp32 == "fixed":
        st = StaticStep(model, pool, **kw)
        base = _static_run(model, st, ids, toks, 2)
        st.detach()
    _narrow_head(model)
    seen = _sites(monkeypatch, model)
    st = StaticStep(model, pool, **kw)
    assert not R2.fp32_fixed() or fp32 == "fixed"
    assert R2.head_uses_fixed(model)
    for k in seen:      # the build's own self-check launches are not the step's
        seen[k] = 0
    lg, h, s = _static_run(model, st, ids, toks, 2)
    # the 2 steps launched the head twice through the fixed kernel (prefill goes through the module's own forward,
    # which is the fixed kernel too: 1 more) and never through F.linear
    assert seen["linear_head"] == 0 and seen["head"] == 3, seen
    st.detach()
    if base is not None:
        _assert_rows_equal(lg, base[0], f"6-row logits, bf16 head vs fp32-stored head ({moe_impl}, glue {glue})")
        assert all(torch.equal(a, b) for a, b in zip(h, base[1]))


def test_static1_bf16_head_equals_fp32_stored_fixed_head(emu, monkeypatch):
    from runtime.static_step import StaticStep

    from test_static_rows_cpu import _prefill

    monkeypatch.setenv(R2.FP32_IMPL_ENV, "fixed")
    model, pool, _ = emu("mini")
    ids, toks = _tf_tokens(17, 4, 7)
    outs = {}
    for tag in ("fp32", "bf16"):
        if tag == "bf16":
            _narrow_head(model)
        st = StaticStep(model, pool, max_pos=255, rows=1, _allow_cpu=True)
        _prefill(model, ids)
        with torch.inference_mode():
            outs[tag] = torch.stack([st.step(toks[i], len(ids) + i, graph=False)[0].float().clone()
                                     for i in range(4)])
        st.detach()
    assert _eq(outs["fp32"], outs["bf16"])
