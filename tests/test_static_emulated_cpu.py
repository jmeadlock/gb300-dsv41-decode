"""CPU end-to-end check of runtime/static_step.py's plumbing (Mac: `.venv/bin/python -m pytest -q tests/`).

The six TileLang kernels do not run on the Mac, so both sides of the comparison run on CPU *emulations* of them,
installed as the reference module's globals (where model.py and StaticStep both look them up): the reference forward
`model(tokens, pos)` vs `StaticStep.step(tok, pos, graph=False)` on the mini checkpoints, `torch.equal` on the fp32
logits of every position. Kernel fidelity is not what is tested (the box GPU test does that with the real kernels);
what is tested is everything T8 changed around them: device positions, fixed widths, index order, the -1 padding,
parity selection, cache writes, Engram hash, MoE id order / staging / slot-order sum / pointer tables, cold counters.

`sparse_attn` is emulated with the kernel's own algorithm (kernel.py:350-387: 64-wide blocks, -1 = masked, running
max from -1e30, online-softmax rescale, bf16 P), so the design's claim that whole -1 blocks are exact no-ops
(#8/#9) is exercised for real. The pointer-table gather is emulated by looking the addresses up in a registry of the
pool's tensors (on CPU, "cold" experts are plain host views).
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch

from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
from runtime.loader import build_and_load
from runtime.refmodel import import_reference

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ------------------------------------------------------------------------------------------------ kernel emulations
def _scale_f(s):
    return torch.exp2(s.view(torch.uint8).float() - 127) if s.dtype == torch.float8_e8m0fnu else s.float()


def _fp4_deq(b):
    u = b.view(torch.uint8).to(torch.long)
    return torch.stack([E2M1[u & 0xF], E2M1[u >> 4]], -1).flatten(-2)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    z = x.float().unflatten(-1, (-1, block_size))
    s = z.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    if scale_fmt is not None:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    q = (z / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace  # every decode call site is in place (model.py:546, 553, 761)
    z = x.float().unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2 ** -9) / 6).to(torch.float8_e4m3fn).float()
    else:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2 ** -126) / 6)))
    v = (z / s).clamp(-6, 6)
    q = GRID[(v.abs().unsqueeze(-1) - GRID).abs().argmin(-1)] * v.sign()
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    K = a.size(-1)
    M, N = a.numel() // K, b.size(0)
    A = (a.float().view(M, K // block_size, block_size) * a_s.float().view(M, -1, 1)).view(M, K)
    sb = _scale_f(b_s).repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)
    return (A @ (b.float() * sb).t()).to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    K = a.size(-1)
    M, N = a.numel() // K, b.size(0)
    assert b.is_contiguous() and tuple(b_s.shape) == (N, K // 32)  # kernel.py:573-582
    A = (a.float().view(M, K // act_block_size, act_block_size) * a_s.float().view(M, -1, 1)).view(M, K)
    B = _fp4_deq(b) * _scale_f(b_s).repeat_interleave(32, -1)
    return (A @ B.t()).to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc, m = hc_mult, mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py:327-403's algorithm in torch: per (b, m), 64-wide index blocks in order, online softmax."""
    b, m, h, d = q.shape
    topk, blk = topk_idxs.size(-1), 64
    o = torch.empty_like(q)
    for bi in range(b):
        for mi in range(m):
            qv = q[bi, mi].float()
            mx = torch.full((h,), -1e30)
            se = torch.zeros(h)
            acc = torch.zeros(h, d)
            for t in range((topk + blk - 1) // blk):
                ids = torch.full((blk,), -1, dtype=torch.long)
                part = topk_idxs[bi, mi, t * blk:(t + 1) * blk].long()
                ids[:part.numel()] = part
                valid = ids != -1
                kvb = torch.where(valid.unsqueeze(-1), kv[bi, ids.clamp_min(0)], torch.zeros((), dtype=kv.dtype))
                s = torch.where(valid, qv @ kvb.float().t(), torch.tensor(-torch.inf)) * softmax_scale
                prev = mx
                mx = torch.maximum(prev, s.amax(-1))
                sc = torch.exp(prev - mx)
                p = torch.exp(s - mx.unsqueeze(-1))
                se = se * sc + p.sum(-1)
                acc = acc * sc.unsqueeze(-1) + p.to(torch.bfloat16).float() @ kvb.float()
            se = se + torch.exp(attn_sink.float() - mx)
            o[bi, mi] = (acc / se.unsqueeze(-1)).to(q.dtype)
    return o


KERNELS = dict(act_quant=act_quant, fp4_act_quant=fp4_act_quant, fp8_gemm=fp8_gemm, fp4_gemm=fp4_gemm,
               hc_split_sinkhorn=hc_split_sinkhorn, sparse_attn=sparse_attn)


def _fake_static_kernels(registry):
    mod = types.ModuleType("runtime.static_kernels")
    mod.calls = {"gather": 0, "marker": 0, "gather_tables": 0}

    def gather_rows(ptab, ids, dst, row_bytes):
        flat = dst.view(torch.uint8).view(ids.numel(), row_bytes)
        for j, e in enumerate(ids.tolist()):
            flat[j].copy_(registry[int(ptab[e])])
        mod.calls["gather"] += 1

    def gather_tables(ptab6, ids, dtab, dsts, row_w, row_s, scale_f32, ids_out=None):
        """T9b-2's one-launch gather (runtime/static_kernels.py t9b2_gather_tables), emulated through the registry:
        weight rows as bytes; scale rows as bytes or decoded e8m0 -> fp32; ids into ids_out."""
        assert [int(a) for a in dtab.tolist()] == [t.data_ptr() for t in dsts] and ptab6.shape[0] == 6
        k = ids.numel()
        for p in range(3):
            flat = dsts[p].view(torch.uint8).view(k, row_w)
            for j, e in enumerate(ids.tolist()):
                flat[j].copy_(registry[int(ptab6[p, e])])
            for j, e in enumerate(ids.tolist()):
                src = registry[int(ptab6[3 + p, e])]
                if scale_f32:
                    dsts[3 + p].view(k, row_s)[j].copy_(src.view(torch.float8_e8m0fnu).float())
                else:
                    dsts[3 + p].view(torch.uint8).view(k, row_s)[j].copy_(src)
        if ids_out is not None:
            ids_out.copy_(ids)
        mod.calls["gather_tables"] += 1

    def step_marker(ctr):
        ctr += 1
        mod.calls["marker"] += 1

    mod.gather_rows, mod.gather_tables, mod.step_marker = gather_rows, gather_tables, step_marker
    return mod


def _mini4():
    cfg = dict(MINI_CONFIG)
    cfg.update({"n_layers": 4, "compress_ratios": [0, 2, 1, 1, 0], "kv_source_layers": [1, 2],
                "index_source_layers": [1, 2, 3], "candidate_source_layer": 2, "candidate_topk_blocks": 64,
                "candidate_block_size": 8, "index_topk": 64, "dspark_target_layer_ids": [3],
                "n_activated_experts": 4})  # topk >= 3: fp32 accumulation order is observable
    return cfg, {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6], "2": [0, 1, 2, 3], "3": [4, 5, 6, 7]}


@pytest.fixture
def emu(monkeypatch, tmp_path):
    """-> build(kind) -> (model, pool, fake static_kernels module)."""
    R = import_reference(cpu_stubs=True)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: ([i % 97 for i in range(MINI_CONFIG["vocab_size"])],
                                     MINI_CONFIG["engram_compressed_vocab_size"]))
    for k, f in KERNELS.items():
        monkeypatch.setattr(R, k, f)
    prev = torch.get_default_dtype()

    def build(kind):
        if kind == "mini4":
            cfg, mp = _mini4()
            ck = make_mini_checkpoint(str(tmp_path / kind), cfg=cfg)
        else:
            ck = make_mini_checkpoint(str(tmp_path / kind))
            mp = ck["map"]
        model = build_and_load(ck["model_dir"], mp, "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                               log=lambda *a: None)
        pool = model.runtime_load.pool
        reg = {}
        for l in range(pool.n_layers):
            for p in ("w1", "w2", "w3"):
                w, s = pool.hbm[l][p]
                for t in list(w) + list(s):
                    reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
                for e in pool.cold[l]:
                    for kind_ in ("weight", "scale"):
                        t = pool.cold_view(l, e, p, kind_, cuda=False)
                        reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
        fk = _fake_static_kernels(reg)
        monkeypatch.setitem(sys.modules, "runtime.static_kernels", fk)
        model.temperature = 0.0
        return model, pool, fk

    yield build
    torch.set_default_dtype(prev)
    R.MoE.forward = getattr(R.MoE, "_orig_forward", R.MoE.forward)  # leave the reference class as found


def _ref_run(model, ids, n):
    with torch.inference_mode():
        out_ids, logits, _ = model(torch.tensor([ids]), 0)
        res, toks = [logits[0].clone()], [int(out_ids[0])]
        for k in range(n):
            out_ids, logits, _ = model(torch.tensor([[toks[-1]]]), len(ids) + k)
            res.append(logits[0].clone())
            toks.append(int(out_ids[0]))
    return res, toks


def _static_run(model, st, ids, n):
    with torch.inference_mode():
        out_ids, logits, _ = model(torch.tensor([ids]), 0)
        res, toks = [logits[0].clone()], [int(out_ids[0])]
        for k in range(n):
            lg = st.step(toks[-1], len(ids) + k, graph=False)
            res.append(lg[0].clone())
            toks.append(int(st.tok.reshape(-1)[0]))
            assert toks[-1] == int(lg[0].argmax())
    return res, toks


def _check(a, b, what):
    bad = [i for i, (x, y) in enumerate(zip(a[0], b[0])) if not torch.equal(x, y)]
    assert not bad and a[1] == b[1], (what, bad, a[1], b[1])


@pytest.mark.parametrize("kind", ["mini", "mini4"])
def test_static_step_equals_reference_forward_emulated(emu, kind):
    from runtime.static_step import StaticStep

    model, pool, fk = emu(kind)
    st = StaticStep(model, pool, max_pos=127, marker=True, _allow_cpu=True)
    assert st.W == ({2: 64} if kind == "mini" else {1: 128, 2: 64})
    g = torch.Generator().manual_seed(11)
    for L in (2, 3, 16, 17):  # both prefill parities; 10 decode steps cover both graph parities
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        ref = _ref_run(model, ids, 10)
        got = _static_run(model, st, ids, 10)
        _check(got, ref, (kind, L))
    assert fk.calls["marker"] == st.n_counted == 40 and fk.calls["gather"] == 40 * 6 * len(model.layers)
    st.detach()


def test_static_cold_counters_match_reference_hooks(emu):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")
    st = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    ids = list(range(5, 21))
    with torch.inference_mode():
        model(torch.tensor([ids]), 0)
    pool.reset_counters()
    ref = _ref_run(model, ids, 0)  # prefill only, to re-seed the caches
    pool.reset_counters()
    toks = ref[1]
    with torch.inference_mode():
        for k in range(12):
            _, lg, _ = model(torch.tensor([[toks[-1]]]), len(ids) + k)
            toks.append(int(lg.argmax()))
    ref_cold = list(pool.counters["cold_calls"])
    got = _static_run(model, st, ids, 12)
    assert got[1] == toks
    cold, n, k = st.cold_counts()
    assert n == 12 and k == 4 and cold == ref_cold and sum(cold) > 0, (cold, ref_cold)
    st.detach()


def test_generator_static_mode_on_emulated_model(emu):
    from runtime.generate import Generator, dump_record
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    model, pool, _ = emu("mini4")
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu",
                    static_factory=lambda m: StaticStep(m, pool, max_pos=127, _allow_cpu=True))
    ids = list(range(40, 57))
    a = gen.greedy(ids, 9, stop_on_eos=False)
    ma = dict(gen.last_meta)
    gen.set_mode("static")
    b = gen.greedy(ids, 9, stop_on_eos=False)
    mb = dict(gen.last_meta)
    assert dump_record(a) == dump_record(b) and ma["step_sha256"] == mb["step_sha256"]
    assert ma["cold"]["cold_calls_per_layer"] == mb["cold"]["cold_calls_per_layer"]  # hooks vs device counters
    gen.set_mode("eager")
    assert dump_record(gen.greedy(ids, 9, stop_on_eos=False)) == dump_record(a)  # patch inert for eager
    gen.reset_static()


def test_static_step_refuses_bad_positions(emu):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini")
    st = StaticStep(model, pool, max_pos=63, _allow_cpu=True)
    with pytest.raises(ValueError):
        st.set_inputs(5, 0)  # design #7: pos >= 1
    with pytest.raises(ValueError):
        st.set_inputs(5, 64)
    with pytest.raises(RuntimeError):
        st.replay(5, 3)  # no graphs
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=model.max_seq_len, _allow_cpu=True)
    st.detach()
