"""Mac CPU tests of M2-T17, loader and draft side (docs/HANDOFF-M2-T17-BF16-HEAD.md; kernel / call sites:
tests/test_head_bf16_cpu.py).

  .venv/bin/python -m pytest -q tests/test_head_bf16_loader_cpu.py

  * build_and_load(head_dtype="bf16") on the mini checkpoints (verify=True: every placed tensor byte-identical to
    its shard, the head's bytes ARE the checkpoint's): head.weight is bf16 (half the bytes), no longer in `widened`,
    the module's forward is the fixed-order bf16 read and its logits are bit-equal to the fp32-stored head's fixed-order
    logits (the default build); everything else in the model is the default build's, byte for byte;
  * the draft's Markov head (mtp.<last>.markov_head.head.weight, bf16 in the checkpoint): kept bf16 with load_mtp, bytes
    accounted (hbm_params / hbm_total / mtp_HBM drop by the freed bytes), forward == fixed order on the widened copy;
  * the switch: head_dtype argument > $DSV41_HEAD_DTYPE > "fp32" (default: today's path, unchanged); an unknown value
    and a checkpoint whose head is not bf16 are refused;
  * the V2 gate of tests/gpu/test_draft_gpu.py on the CPU emulations with $DSV41_HEAD_DTYPE=bf16: DraftStep ==
    the reference forward_spec (whose heads are the same fixed-order modules), drafts / logits / confidence / rings
    torch.equal, and the shared head / Markov head are the bf16 ones.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from runtime import fused_glue_r2 as R2  # noqa: E402
from runtime.loader import (HEAD_DTYPE_ENV, LoadRefused, build_and_load, resolve_head_dtype,  # noqa: E402
                            keep_markov_bf16)
from test_draft_cpu import _cpu_default_device, _gpu_module, emu  # noqa: E402,F401


def _eq(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(HEAD_DTYPE_ENV, raising=False)


def test_resolve_head_dtype(monkeypatch):
    assert resolve_head_dtype() == "fp32" and resolve_head_dtype("bf16") == "bf16"
    monkeypatch.setenv(HEAD_DTYPE_ENV, "bf16")
    assert resolve_head_dtype() == "bf16" and resolve_head_dtype("fp32") == "fp32"   # the argument wins
    monkeypatch.setenv(HEAD_DTYPE_ENV, "fp16")
    with pytest.raises(ValueError, match="not in"):
        resolve_head_dtype()
    with pytest.raises(ValueError, match="not in"):
        resolve_head_dtype("fp8")


def _build(emu, monkeypatch, kind, head_dtype):
    if head_dtype is None:
        monkeypatch.delenv(HEAD_DTYPE_ENV, raising=False)
    else:
        monkeypatch.setenv(HEAD_DTYPE_ENV, head_dtype)
    return emu(kind)


@pytest.mark.parametrize("kind", ["mini", "mini3"])
def test_loader_keeps_head_and_markov_bf16_and_logits_equal_the_fp32_stored_head(emu, monkeypatch, kind):
    m32 = _build(emu, monkeypatch, kind, None)
    w32, mk32 = m32.head.weight.detach().clone(), m32.mtp[-1].markov_head.head.weight.detach().clone()
    r32 = m32.runtime_load
    assert w32.dtype == torch.float32 and "head.weight" in r32.widened and r32.head_dtype == "fp32"
    assert r32.head_bf16 == [] and not hasattr(m32.head, "_m2t17_bf16_head")
    mtp32 = m32.runtime_mtp.bytes["hbm_total"]
    mtp_hbm32 = r32.bytes["mtp_HBM"]
    hp32 = r32.bytes["hbm_params"]

    m16 = _build(emu, monkeypatch, kind, "bf16")                   # verify=True inside emu: shard bytes == placed bytes
    r16, w16 = m16.runtime_load, m16.head.weight
    assert r16.head_dtype == "bf16" and w16.dtype == torch.bfloat16 and tuple(w16.shape) == tuple(w32.shape)
    assert "head.weight" not in r16.widened and r16.verify["mismatch"] == [] and r16.verify["ok"] == r16.verify["n"]
    assert torch.equal(w16.float(), w32)                           # the same values (bf16 checkpoint values)
    assert r16.head_bf16 == ["head.weight", f"mtp.{len(m16.mtp) - 1}.markov_head.head.weight"]
    assert r16.summary()["head_dtype"] == "bf16" and r16.summary()["head_bf16"] == r16.head_bf16
    # bytes: the backbone's hbm_params fell by exactly half the head, the mtp side by half the Markov head
    assert hp32 - r16.bytes["hbm_params"] == _nbytes(w32) // 2
    assert r16.bytes["head_HBM"] == _nbytes(w16) == _nbytes(w32) // 2
    mk16 = m16.mtp[-1].markov_head.head.weight
    assert mk16.dtype == torch.bfloat16 and torch.equal(mk16.float(), mk32)
    assert mtp32 - m16.runtime_mtp.bytes["hbm_total"] == _nbytes(mk32) // 2
    assert mtp_hbm32 - r16.bytes["mtp_HBM"] == _nbytes(mk32) // 2
    assert m16.runtime_mtp.bytes["hbm_total"] == m16.runtime_mtp.bytes["hbm_params"] + \
        m16.runtime_mtp.bytes["hbm_expert_stacks"]
    assert m16.mtp[0].head is m16.head and m16.mtp[0].embed is m16.embed          # still shared with the backbone
    # everything else is the fp32-stored build's, byte for byte
    p32 = dict(m32.named_parameters())
    for n, p in m16.named_parameters():
        if n in ("head.weight", f"mtp.{len(m16.mtp) - 1}.markov_head.head.weight"):
            continue
        q = p32[n]
        assert p.dtype == q.dtype and p.shape == q.shape, n
        assert torch.equal(p.reshape(-1).view(torch.uint8), q.reshape(-1).view(torch.uint8)), n

    # the forward: bit-equal to the fp32-stored head's fixed-order logits, last-position and full forms, R = 1 vs 6
    g = torch.Generator(device="cpu").manual_seed(4)
    D = w32.size(1)
    x = (torch.randn(1, 6, D, generator=g, device="cpu") * 2).to(torch.bfloat16)
    want = R2.head_fixed_w(w32, x, full_logits=True)
    got = m16.head(x, full_logits=True)
    assert got.dtype == torch.float32 and _eq(got, want)
    assert _eq(m16.head(x), want[:, -1])
    for i in range(6):
        assert _eq(m16.head(x[:, i:i + 1])[0], want[0, i])
    # Markov head: head(embed, full_logits=True) on [b, rank]
    tok = torch.tensor([3, 11, 20])
    emb = m16.mtp[-1].markov_head.embed(tok)
    lg, e2 = m16.mtp[-1].markov_head(tok)
    assert _eq(e2, emb) and _eq(lg, R2.head_fixed_w(mk32, emb, True))
    assert lg.shape == (3, mk32.size(0))


def test_default_build_is_unchanged(emu, monkeypatch):
    m = _build(emu, monkeypatch, "mini", None)
    assert m.head.weight.dtype == torch.float32 and m.mtp[-1].markov_head.head.weight.dtype == torch.float32
    assert "forward" not in m.head.__dict__ and "forward" not in m.mtp[-1].markov_head.head.__dict__
    assert m.runtime_load.summary()["head_dtype"] == "fp32" and m.runtime_load.summary()["head_bf16"] == []


def test_keep_markov_bf16_refuses_a_value_that_is_not_bf16_representable(emu, monkeypatch):
    m = _build(emu, monkeypatch, "mini", None)
    w = m.mtp[-1].markov_head.head.weight
    with torch.no_grad():
        w[0, 0] = 1.0 + 2.0 ** -20                      # an fp32 value no bf16 holds
    with pytest.raises(LoadRefused, match="not bf16-representable"):
        keep_markov_bf16(m, m.runtime_load, log=lambda *a: None)
    assert m.mtp[-1].markov_head.head.weight.dtype == torch.float32      # nothing was changed


def test_loader_refuses_a_head_that_is_not_bf16_in_the_checkpoint(tmp_path, monkeypatch):
    from runtime.refmodel import import_reference

    import_reference(cpu_stubs=True)
    """head_dtype=bf16 is only defined for a bf16-stored head: an fp32 shard is refused, not silently narrowed."""
    from mini_ckpt import make_mini_checkpoint
    from runtime import ckpt as ckpt_mod

    import engram
    from mini_ckpt import MINI_CONFIG

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: ([i % 97 for i in range(MINI_CONFIG["vocab_size"])],
                                     MINI_CONFIG["engram_compressed_vocab_size"]))
    ck = make_mini_checkpoint(str(tmp_path / "m"))
    real = ckpt_mod.ShardIndex.scan

    def scan(md):
        idx = real(md)
        idx.entries["head.weight"] = dataclasses.replace(idx.entries["head.weight"], st_dtype="F32")  # "stores fp32"
        return idx

    import runtime.loader as L

    monkeypatch.setattr(L.ShardIndex, "scan", staticmethod(scan))
    prev = torch.get_default_dtype()      # build_meta_model sets the global default dtype to bf16 (generate.py:118)
    try:
        with pytest.raises(LoadRefused, match="needs head.weight stored as bf16"):
            build_and_load(ck["model_dir"], ck["map"], "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                           head_dtype="bf16", log=lambda *a: None)
    finally:
        torch.set_default_dtype(prev)


@pytest.mark.parametrize("kind", ["mini3"])
def test_draft_gate_dry_run_with_bf16_heads(emu, monkeypatch, capsys, kind):
    """tests/gpu/test_draft_gpu.py's V2 gate on the CPU emulations, heads bf16: DraftStep static-eager == the
    reference forward_spec (shared head, 5 sequential Markov-head calls, confidence head)."""
    monkeypatch.setenv(HEAD_DTYPE_ENV, "bf16")
    G = _gpu_module(monkeypatch, emu)
    G._gate_draft(kind, 17)
    out = capsys.readouterr().out
    assert out.count("DraftStep static-eager == reference forward_spec") == 1, out
    model = emu(kind)
    assert model.head.weight.dtype == torch.bfloat16 and model.mtp[-1].markov_head.head.weight.dtype == torch.bfloat16
    assert getattr(model.head, "_m2t17_bf16_head", False) and \
        getattr(model.mtp[-1].markov_head.head, "_m2t17_bf16_head", False)
