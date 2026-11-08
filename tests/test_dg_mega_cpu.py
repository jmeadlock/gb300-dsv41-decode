"""Mac CPU tests of M2-L3 (runtime/dg_mega.py, `moe_impl="dg-mega"`): `.venv/bin/python -m pytest -q tests/`.

What a CPU can check, and does:
  * the layout transforms == DeepGEMM's own (`_interleave_weights`, `_transpose_sf_for_utccp`, executed from the
    vendored source text reference/vllm-0.31.0/deep_gemm/mega/__init__.py -- no import of deep_gemm), incl. the
    strides they keep; the index math (l1_row / utccp_pos) and every inverse;
  * the routing map (pool groups, compact cold index over the step's UNIQUE cold experts, overflow -> -1, the
    staging list) == its plain-Python mirror on random routings;
  * the staging contract (stage_mega_emulated) on a synthetic layer, and the Triton kernel body l3_stage_mega
    executed program by program (tests/triton_shim.py) == the contract: both SF storages, several program counts;
  * the whole dg-mega StaticStep on the mini checkpoints (reference kernels emulated, tests/test_static_emulated_cpu)
    with a CONTRACT-CHECKING fake `fp8_fp4_mega_moe` that reads only the mega layouts (it de-interleaves / un-UTCCPs
    them itself) against an oracle computed straight from the model's own Expert params with the kernel's numerics
    (bf16 gate/up, bf16 clamp, SwiGLU in fp32, fp8 requant FROM FP32, combine in column = ascending-id order, shared
    "gemv" added after the bf16 output): bitwise on every layer, at 1 row and 6 rows (rows invariant), both shared
    modes, cold experts staged; self-replay equal; decode steps finite.
Box only (tests/gpu/test_dg_mega_gpu.py, bench/dg_mega_spike.py): the real kernel, its JIT, the ABI, capture, speed.
"""
from __future__ import annotations

import ast
import os
import random
import sys
import types

import pytest
import torch

from runtime import dg_mega as DM
from test_static_emulated_cpu import emu  # noqa: F401 (fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED = os.path.join(REPO, "reference", "vllm-0.31.0", "deep_gemm", "mega", "__init__.py")
FP8 = torch.float8_e4m3fn


def _cpu_gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _vendored(*names):
    """Functions from the vendored deep_gemm/mega/__init__.py source, executed with torch only."""
    tree = ast.parse(open(VENDORED).read())
    ns = {"torch": torch}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module([n], []), VENDORED, "exec"), ns)
    return [ns[n] for n in names]


# ------------------------------------------------------------------------------------------------ layouts
def test_interleave_and_utccp_equal_the_vendored_functions():
    inter, utccp = _vendored("_interleave_weights", "_transpose_sf_for_utccp")
    g = _cpu_gen(0)
    for shape in ((3, 64, 40), (64, 40), (2, 512, 16)):
        t = torch.randint(-128, 127, shape, generator=g, device="cpu").to(torch.int8)
        assert torch.equal(inter(t), DM.interleave_rows(t))
        w = torch.randint(-2 ** 31, 2 ** 31 - 1, shape, generator=g, device="cpu", dtype=torch.int64).to(torch.int32)
        assert torch.equal(inter(w), DM.interleave_rows(w))
    for shape in ((2, 256, 5), (384, 3), (1, 128, 40)):
        s = torch.randint(-2 ** 31, 2 ** 31 - 1, shape, generator=g, device="cpu", dtype=torch.int64).to(torch.int32)
        assert torch.equal(utccp(s), DM.utccp_rows(s))
        mn = s.transpose(-1, -2).contiguous().transpose(-1, -2)          # MN-major storage: strides are kept
        a, b = utccp(mn), DM.utccp_rows(mn)
        assert torch.equal(a, b) and a.stride() == b.stride() == mn.stride()
        assert torch.equal(DM.utccp_rows_inverse(b), s)


def test_index_math_and_inverses():
    for n in range(0, 96):
        p, r = DM.l1_source(n)
        assert DM.l1_row(r, p) == n
    idx = torch.arange(512, dtype=torch.int32, device="cpu").view(512, 1)
    u = DM.utccp_rows(idx).view(-1).tolist()
    assert all(u[DM.utccp_pos(n)] == n for n in range(512)) and all(DM.utccp_src(m) == u[m] for m in range(512))
    rows = torch.arange(64, dtype=torch.int32, device="cpu").view(64, 1)
    il = DM.interleave_rows(rows).view(-1).tolist()
    assert all(il[DM.l1_row(r, p)] == p * 32 + r for p in (0, 1) for r in range(32))


def test_mega_weight_and_sf_builders_round_trip():
    g = _cpu_gen(1)
    G, I, KH, KS = 3, 128, 16, 8
    w1, w3 = (torch.randint(-128, 127, (G, I, KH), generator=g, device="cpu").to(torch.int8) for _ in range(2))
    assert torch.equal(DM.mega_l1_weight(w1, w3), DM.interleave_rows(torch.cat([w1, w3], 1)))
    assert torch.equal(DM.mega_l1_weight(w1[0], w3[0]), DM.interleave_rows(torch.cat([w1[0], w3[0]], 0)))
    gt, ut = DM.deinterleave_rows(DM.mega_l1_weight(w1, w3))
    assert torch.equal(gt, w1) and torch.equal(ut, w3)
    c1, c3, c2 = (torch.randint(0, 255, (G, I, KS), generator=g, device="cpu").to(torch.uint8) for _ in range(3))
    for major in DM.SF_MAJORS:
        sf1 = DM.mega_l1_sf(c1, c3, major)
        assert sf1.dtype == torch.int32 and tuple(sf1.shape) == (G, 2 * I, KS // 4)
        if major == "mn":
            assert sf1.stride() == (2 * I * KS // 4, 1, 2 * I)
        else:
            assert sf1.is_contiguous()
        a, b = DM.l1_sf_codes_of(sf1)
        assert torch.equal(a, c1) and torch.equal(b, c3)
        assert torch.equal(DM.l2_sf_codes_of(DM.mega_l2_sf(c2, major)), c2)
        # == the library's own transform pipeline on the packed words (transform_weights_for_mega_moe's SF half)
        inter, utccp = _vendored("_interleave_weights", "_transpose_sf_for_utccp")
        want = utccp(inter(torch.cat([DM.sf_words(c1), DM.sf_words(c3)], 1)))
        assert torch.equal(sf1.contiguous(), want)
    s32 = torch.randint(100, 140, (I // 32, 2 * KS), generator=g, device="cpu").to(torch.uint8)
    rows = DM.block_codes_per_row(s32, I)
    assert rows.shape == (I, 2 * KS) and all(torch.equal(rows[r], s32[r // 32]) for r in range(I))


def test_act_codes_and_write_act_sf_packings():
    g = _cpu_gen(2)
    R, T, KS = 3, 6, 16
    codes = torch.randint(100, 150, (R, KS), generator=g, device="cpu").to(torch.uint8)
    f32 = torch.exp2(codes.float() - 127)
    assert torch.equal(DM.act_codes(f32), codes) and torch.equal(DM.act_codes(codes.view(torch.float8_e8m0fnu)), codes)
    v32 = torch.zeros(T, KS // 4, dtype=torch.int32, device="cpu")
    DM.write_act_sf(v32, codes, R)
    assert torch.equal(v32[:R].contiguous().view(torch.uint8), codes) and not v32[R:].any()
    v8 = torch.zeros(T, KS, dtype=torch.uint8, device="cpu")
    DM.write_act_sf(v8, codes, R)
    assert torch.equal(v8[:R], codes)
    vf = torch.zeros(T, KS, dtype=torch.float32, device="cpu")
    DM.write_act_sf(vf, codes, R)
    assert torch.equal(vf[:R], f32)
    vmn = torch.zeros(KS // 4, T, dtype=torch.int32, device="cpu").t()      # an MN-major view is written by value
    DM.write_act_sf(vmn, codes, R)
    assert torch.equal(vmn[:R].contiguous().view(torch.uint8), codes)


# ------------------------------------------------------------------------------------------------ routing
@pytest.mark.parametrize("seed", range(6))
def test_mega_route_equals_host_mirror(seed):
    rng = random.Random(seed)
    E, H, k = 64, 40, 6
    hot = rng.sample(range(E), H)
    gtab = [-1] * E
    for s, e in enumerate(hot):
        gtab[e] = s
    for R in (1, 2, 6):
        for S in (1, 3, 6, 16):
            ids = [sorted(rng.sample(range(E), k)) for _ in range(R)]
            got = DM.mega_route(torch.tensor(ids, device="cpu"), torch.tensor(gtab, dtype=torch.int32, device="cpu"),
                                H, S, E)
            want = DM.mega_route_host(ids, gtab, H, S)
            assert got["topk"].tolist() == want["topk"], (R, S)
            assert got["stage"].tolist() == want["stage"], (R, S)
            assert int(got["n_cold"]) == want["n_cold"] and int(got["n_over"]) == want["n_over"]
            assert got["topk"].dtype == torch.int64 and got["stage"].dtype == torch.int64


def test_unique_ids_record():
    ids = torch.tensor([[1, 5, 9], [1, 2, 9]], device="cpu")
    s, f = DM.unique_ids_record(ids, 6)
    assert s.tolist() == [1, 1, 2, 5, 9, 9] and f.tolist() == [1, 0, 1, 1, 1, 0]
    s, f = DM.unique_ids_record(ids[:1], 4)
    assert s.tolist() == [1, 5, 9, 9] and f.tolist() == [1, 1, 1, 0]


# ------------------------------------------------------------------------------------------------ staging
def _syn_layer(I=64, D=128, KS1=4, KS2=4, E=12, H=7, S=3, seed=3):
    """L1 copy / pool w2 / SF copies of a synthetic layer + cold sources; K/2 = D/2 bytes per w1 row."""
    g = _cpu_gen(seed)
    KH1, KH2 = D // 2, I // 2
    src_w = {(p, e): torch.randint(-128, 127, ((I, KH1) if p < 2 else (D, KH2)), generator=g, device="cpu")
             .to(torch.int8) for p in range(3) for e in range(E)}
    src_s = {(p, e): torch.randint(1, 255, ((I, KS1) if p < 2 else (D, KS2)), generator=g, device="cpu")
             .to(torch.uint8) for p in range(3) for e in range(E)}
    hot = list(range(H))
    gtab = torch.tensor([e if e < H else -1 for e in range(E)], dtype=torch.int32, device="cpu")
    ptab6 = torch.tensor([[src_w[(p, e)].data_ptr() for e in range(E)] for p in range(3)]
                         + [[src_s[(p, e)].data_ptr() for e in range(E)] for p in range(3)], dtype=torch.int64,
                         device="cpu")
    reg = {t.data_ptr(): t.reshape(-1).view(torch.uint8) for t in list(src_w.values()) + list(src_s.values())}
    return dict(I=I, D=D, KS1=KS1, KS2=KS2, E=E, H=H, S=S, hot=hot, gtab=gtab, ptab6=ptab6, reg=reg, src_w=src_w,
                src_s=src_s)


def _dsts(L, major, seed=9):
    g = _cpu_gen(seed)
    G = L["H"] + L["S"]
    l1 = torch.randint(-128, 127, (G, 2 * L["I"], L["D"] // 2), generator=g, device="cpu").to(torch.int8)
    l2 = torch.randint(-128, 127, (G, L["D"], L["I"] // 2), generator=g, device="cpu").to(torch.int8)
    c1 = torch.randint(1, 255, (G, L["I"], L["KS1"]), generator=g, device="cpu").to(torch.uint8)
    c3 = torch.randint(1, 255, (G, L["I"], L["KS1"]), generator=g, device="cpu").to(torch.uint8)
    c2 = torch.randint(1, 255, (G, L["D"], L["KS2"]), generator=g, device="cpu").to(torch.uint8)
    return [l1, l2, DM.mega_l1_sf(c1, c3, major), DM.mega_l2_sf(c2, major)]


@pytest.mark.parametrize("major", DM.SF_MAJORS)
def test_stage_emulated_contract(major):
    L = _syn_layer()
    dsts = _dsts(L, major)
    before = [t.clone() for t in dsts]
    H, S = L["H"], L["S"]
    stage = torch.tensor([9, -1, 11], device="cpu")
    DM.stage_mega_emulated(L["reg"].__getitem__, stage, L["ptab6"], dsts, H, major)
    l1, l2, sf1, sf2 = dsts
    for c, e in ((0, 9), (2, 11)):
        g_ = H + c
        assert torch.equal(l1[g_], DM.mega_l1_weight(L["src_w"][(0, e)], L["src_w"][(1, e)]))
        assert torch.equal(l2[g_], L["src_w"][(2, e)])
        a, b = DM.l1_sf_codes_of(sf1[g_:g_ + 1])
        assert torch.equal(a[0], L["src_s"][(0, e)]) and torch.equal(b[0], L["src_s"][(1, e)])
        assert torch.equal(DM.l2_sf_codes_of(sf2[g_:g_ + 1])[0], L["src_s"][(2, e)])
    untouched = [g_ for g_ in range(H + S) if g_ not in (H, H + 2)]
    for t, t0 in zip(dsts, before):
        assert torch.equal(t[untouched], t0[untouched])
    assert dsts[2].stride() == before[2].stride()


def _dg_mega_under_shim():
    import triton_shim

    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "dg_mega.py"), "dg_mega_under_shim")
    return mod, triton_shim


@pytest.mark.parametrize("major", DM.SF_MAJORS)
@pytest.mark.parametrize("nprog,block_words,I,D", [(1, 512, 128, 128), (3, 8, 128, 128), (4, 4, 128, 256),
                                                   (2, 512, 256, 128)])
def test_stage_kernel_body_equals_contract(major, nprog, block_words, I, D):
    """l3_stage_mega executed program by program == stage_mega_emulated (EVEN and partial program loops, scale tiles
    of 64 rows with a masked tail, the w2 copy's masked tail, both SF storages, an empty and two staged slots)."""
    mod, shim = _dg_mega_under_shim()
    outs = []
    for use_kernel in (False, True):
        L = _syn_layer(I=I, D=D, KS1=D // 32, KS2=I // 32, seed=4)
        dsts = _dsts(L, major)
        dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cpu")
        stage = torch.tensor([10, -1, 7], device="cpu")
        if use_kernel:
            shim.ADDRESSES.clear()
            shim.ADDRESSES.register(*dsts, *L["src_w"].values(), *L["src_s"].values())
            try:
                mod.stage_mega(stage, L["ptab6"], dtab, dsts, L["H"], major, nprog=nprog, block_words=block_words)
            finally:
                shim.ADDRESSES.clear()
            assert mod.l3_stage_mega.launches[-1]["grid"] == (3, nprog, 3)
        else:
            DM.stage_mega_emulated(L["reg"].__getitem__, stage, L["ptab6"], dsts, L["H"], major)
        outs.append([t.clone() for t in dsts])
    for i, (a, b) in enumerate(zip(*outs)):
        assert torch.equal(a, b), i


@pytest.mark.parametrize("seed", range(4))
def test_route_kernel_body_equals_mirror(seed):
    """l3_route_mega executed under the shim == mega_route + the copies (topk rows, weights, staging list, overflow
    counter), R = 1 / 2 / 6, spare budgets with and without overflow; rows >= R of topk_idx untouched."""
    mod, _ = _dg_mega_under_shim()
    rng = random.Random(seed)
    E, H, k, T = 384, 295, 6, 6
    hot = rng.sample(range(E), H)
    gt = [-1] * E
    for s_, e in enumerate(hot):
        gt[e] = s_
    gtab = torch.tensor(gt, dtype=torch.int32, device="cpu")
    for R in (1, 2, 6):
        for S in (1, 3, 6, 16):
            ids = torch.tensor([sorted(rng.sample(range(E), k)) for _ in range(R)], device="cpu")
            w = torch.rand(R, k, generator=_cpu_gen(seed), device="cpu")
            topk = torch.full((T, k), -7, dtype=torch.int64, device="cpu")
            tw = torch.zeros(T, k, dtype=torch.float32, device="cpu")
            stage = torch.full((S,), -9, dtype=torch.int64, device="cpu")
            ovf = torch.tensor([5], dtype=torch.int32, device="cpu")
            mod.route_mega(ids, w, gtab, topk, tw, stage, H, S, E, ovf)
            m = DM.mega_route(ids, gtab, H, S, E)
            assert torch.equal(topk[:R], m["topk"]) and (topk[R:] == -7).all(), (R, S)
            assert torch.equal(tw[:R], w) and torch.equal(stage, m["stage"]), (R, S)
            assert int(ovf) == 5 + int(m["n_over"])
            assert mod.l3_route_mega.launches[-1]["grid"] == (1,)


# ------------------------------------------------------------------------------------------------ fake MegaMoE
def _fp4_deq(b):
    from test_static_emulated_cpu import _fp4_deq as f

    return f(b)


def _deq_w(w_bytes, codes):
    """fp4 [N, K/2] int8 + e8m0 codes [N, K/32] -> fp32 [N, K]."""
    return _fp4_deq(w_bytes) * torch.exp2(codes.float() - 127).repeat_interleave(32, -1)


def _deq_fp8(w, codes):
    return w.float() * torch.exp2(codes.float() - 127).repeat_interleave(32, -1)


def _quant_fp32(h):
    """MegaMoE's L2-input quant: per-32 amax of the fp32 values, UE8M0 scale, e4m3 cast (no bf16 step)."""
    blk = h.unflatten(-1, (-1, 32))
    s = torch.exp2(torch.ceil(torch.log2(blk.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)))
    q = (blk / s).clamp(-448.0, 448.0).to(FP8)
    return (q.float() * s).flatten(-2)


def _mega_expert(a, wg, wu, wd, w, lim):
    """One (token, expert) through the kernel's numerics: a fp32 [D] dequantized activation; wg/wu [I, D], wd [D, I]
    fp32 dequantized weights; w the top-k weight. -> bf16 [D]."""
    gate = (a @ wg.t()).to(torch.bfloat16).float()
    up = (a @ wu.t()).to(torch.bfloat16).float()
    if lim > 0:
        gate = torch.clamp(gate, max=lim)
        up = torch.clamp(up, min=-lim, max=lim)
    h = gate / (1.0 + torch.exp(-gate)) * up * w
    return (_quant_fp32(h) @ wd.t()).to(torch.bfloat16)


def _fake_mega():
    """deep_gemm surface dg_mega uses: SymmBuffer (x fp8 [T, D], x_sf int32 [T, D/128], topk_idx int64 [T, k],
    topk_weights fp32 [T, k]), fp8_fp4_mega_moe with its contract asserted. It reads ONLY the mega layouts and
    inverts them itself (deinterleave / un-UTCCP), so any layout, routing or staging error changes the output."""
    mod = types.ModuleType("deep_gemm")
    mod.__file__ = "<fake deep_gemm mega>"
    mod.calls = []

    class SymmBuffer:
        def __init__(self, group, num_experts, T, k, hidden, inter, num_shared_experts=0, mma_type="fp8xfp4",
                     activation="swiglu"):
            assert group.size() == 1 and group.rank() == 0 and mma_type == "fp8xfp4" and activation == "swiglu"
            self.group, self.num_experts, self.num_max_tokens_per_rank, self.num_topk = group, num_experts, T, k
            self.hidden, self.intermediate_hidden, self.num_shared_experts = hidden, inter, num_shared_experts
            self.buffer = torch.zeros(16, dtype=torch.int8, device="cpu")
            self.x = torch.zeros(T, hidden, dtype=FP8, device="cpu")
            self.x_sf = torch.zeros(T, hidden // 128, dtype=torch.int32, device="cpu")
            self.topk_idx = torch.zeros(T, k, dtype=torch.int64, device="cpu")
            self.topk_weights = torch.zeros(T, k, dtype=torch.float32, device="cpu")

    def fp8_fp4_mega_moe(y, l1, l2, sym, shared_l1_weights=None, shared_l2_weights=None,
                         cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu",
                         activation_clamp=None, fast_math=True, activation_alpha=1.0, activation_beta=0.0):
        (L1, SF1), (L2, SF2) = l1, l2
        G, T, k, D, I = sym.num_experts, sym.num_max_tokens_per_rank, sym.num_topk, sym.hidden, \
            sym.intermediate_hidden
        assert recipe == (1, 1, 32) and activation == "swiglu" and fast_math is False
        assert (activation_alpha, activation_beta) == (1.0, 0.0)
        assert L1.dtype == torch.int8 and tuple(L1.shape) == (G, 2 * I, D // 2) and L1.is_contiguous()
        assert L2.dtype == torch.int8 and tuple(L2.shape) == (G, D, I // 2) and L2.is_contiguous()
        assert SF1.dtype == SF2.dtype == torch.int32 and tuple(SF1.shape) == (G, 2 * I, D // 128)
        assert tuple(SF2.shape) == (G, D, I // 128) and tuple(y.shape) == (T, D) and y.dtype == torch.bfloat16
        assert (shared_l1_weights is None) == (sym.num_shared_experts == 0)
        lim = 0.0 if activation_clamp is None else float(activation_clamp)
        cg, cu = DM.l1_sf_codes_of(SF1)
        c2 = DM.l2_sf_codes_of(SF2)
        wg_b, wu_b = DM.deinterleave_rows(L1)
        a_all = (sym.x.float().view(T, -1, 32)
                 * torch.exp2(sym.x_sf.contiguous().view(torch.uint8).float() - 127).view(T, -1, 1)).view(T, D)
        for t in range(T):
            acc = torch.zeros(D, dtype=torch.float32, device="cpu")
            for j in range(k):
                gi = int(sym.topk_idx[t, j])
                if gi < 0:
                    continue
                acc += _mega_expert(a_all[t], _deq_w(wg_b[gi], cg[gi]), _deq_w(wu_b[gi], cu[gi]),
                                    _deq_w(L2[gi], c2[gi]), float(sym.topk_weights[t, j]), lim).float()
            if shared_l1_weights is not None and bool((sym.topk_idx[t] >= 0).any() or sym.x[t].float().any()):
                (S1, SS1), (S2, SS2) = shared_l1_weights, shared_l2_weights
                sg, su = DM.deinterleave_rows(S1.view(torch.int8))
                scg, scu = DM.l1_sf_codes_of(SS1.unsqueeze(0))
                acc += _mega_expert(a_all[t], _deq_fp8(sg.view(FP8), scg[0]), _deq_fp8(su.view(FP8), scu[0]),
                                    _deq_fp8(S2, DM.l2_sf_codes_of(SS2.unsqueeze(0))[0]), 1.0, lim).float()
            y[t] = acc.to(torch.bfloat16)
        mod.calls.append({"G": G, "T": T, "rows_routed": int((sym.topk_idx >= 0).any(1).sum())})

    mod.SymmBuffer = SymmBuffer
    mod.fp8_fp4_mega_moe = fp8_fp4_mega_moe
    mod.use_deterministic_algorithms = lambda flag=True: None
    return mod


def _oracle(moe, x, R_mod, lim, shared):
    """The MoE output with MegaMoE's numerics, straight from the model's Expert params (no mega layouts)."""
    q, s = R_mod.act_quant(x, R_mod.fp8_block_size, R_mod.scale_fmt, R_mod.scale_dtype)
    a = (q.float().view(x.size(0), -1, 32) * torch.exp2(DM.act_codes(s).float() - 127).view(x.size(0), -1, 1)) \
        .view(x.size(0), -1)
    weights, indices = moe.gate(x, None)
    ids, order = torch.sort(indices, dim=-1)
    w = torch.gather(weights, -1, order)
    out = []
    sh = moe.shared_experts
    for t in range(x.size(0)):
        acc = torch.zeros(x.size(1), dtype=torch.float32, device="cpu")
        for j in range(ids.size(1)):
            ex = moe.experts[int(ids[t, j])]
            dq = [_deq_w(getattr(ex, p).weight.view(torch.int8), DM._u8(getattr(ex, p).scale)) for p in
                  ("w1", "w3", "w2")]
            acc += _mega_expert(a[t], dq[0], dq[1], dq[2], float(w[t, j]), lim).float()
        if shared == "fused":
            I = sh.w1.weight.size(0)
            D = x.size(1)
            dq = [_deq_fp8(sh.w1.weight, DM.block_codes_per_row(sh.w1.scale, I)),
                  _deq_fp8(sh.w3.weight, DM.block_codes_per_row(sh.w3.scale, I)),
                  _deq_fp8(sh.w2.weight, DM.block_codes_per_row(sh.w2.scale, D))]
            acc += _mega_expert(a[t], dq[0], dq[1], dq[2], 1.0, lim).float()
        out.append(acc.to(torch.bfloat16))
    y = torch.stack(out)
    if shared == "gemv":
        y = (y.float() + sh(x).float()).to(torch.bfloat16)
    return y


@pytest.fixture
def mega_emu(emu, monkeypatch):
    from runtime import static_step
    from test_dg_moe_cpu import _faithful_act_quant

    fake = _fake_mega()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    monkeypatch.setattr(static_step, "MOE_IMPLS", tuple(static_step.MOE_IMPLS) + ("dg-mega",))
    monkeypatch.setattr(static_step, "MOE_IMPLS_ROWS", tuple(static_step.MOE_IMPLS_ROWS) + ("dg-mega",))

    def build(kind):
        model, pool, fk = emu(kind)
        R_mod = sys.modules["model"]
        monkeypatch.setattr(R_mod, "act_quant", _faithful_act_quant)
        return model, pool, fk, fake, R_mod

    return build


@pytest.mark.parametrize("shared", DM.SHARED_MODES)
@pytest.mark.parametrize("kind", ["mini", "mini4"])
def test_dg_mega_moe_equals_oracle_rows_invariant(mega_emu, monkeypatch, kind, shared):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(DM.SHARED_ENV, shared)
    monkeypatch.setenv(DM.SPARE_ENV, "12")             # the 1-row build serves 6-row calls here: 6 x top-2 <= 12 cold
    model, pool, _, fake, R_mod = mega_emu(kind)
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-mega", _allow_cpu=True)
    b = st.dg
    assert isinstance(b, DM.DGMegaBuffers) and all(isinstance(m, DM.DGMegaMoE) for m in st.moes)
    assert b.T == DM.MAX_TOKENS_DEFAULT and b.S == 12 and pool.spare >= 12 and b.shared == shared
    lim = float(model.layers[0].ffn.shared_experts.swiglu_limit)
    D = model.layers[0].ffn.dim
    g = _cpu_gen(17)
    n_cold = 0
    with torch.inference_mode(), st._active():
        for l, sm in enumerate(st.moes):
            moe = sm.moe
            x6 = (torch.randn(6, D, generator=g, device="cpu") * 2).to(torch.get_default_dtype())
            y6 = sm(x6.clone())
            n_cold += int(sm.last["n_cold"])
            want = _oracle(moe, x6, R_mod, lim, shared)
            assert torch.equal(y6.view(torch.int16), want.view(torch.int16)), (kind, shared, l)
            for r in range(6):                                                    # rows invariant (R = 1 alone)
                y1 = sm(x6[r:r + 1].clone())
                assert torch.equal(y1.view(torch.int16), y6[r:r + 1].view(torch.int16)), (kind, l, r)
            again = sm(x6.clone())                                                # self-replay
            assert torch.equal(again.view(torch.int16), y6.view(torch.int16))
    assert n_cold > 0, "no cold expert was routed: staging untested"
    assert all(c["T"] == DM.MAX_TOKENS_DEFAULT for c in fake.calls)
    d = b.describe()
    assert d["impl"] == "dg-mega" and d["l1_store"] == "copy" and d["sf_major"] == "mn"
    st.detach()


def test_dg_mega_static_step_decodes_and_replays(mega_emu):
    from test_static_emulated_cpu import _static_run

    from runtime.static_step import StaticStep

    model, pool, _, fake, _ = mega_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-mega", _allow_cpu=True)
    ids = list(range(5, 21))
    a = _static_run(model, st, ids, 6)
    b = _static_run(model, st, ids, 6)
    assert all(bool(torch.isfinite(l).all()) for l in a[0])
    assert all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and a[1] == b[1]
    cold, n, k = st.cold_counts()
    assert n == 12 and k == st.topk and sum(cold) > 0
    assert len(fake.calls) == 12 * len(model.layers)
    st.detach()


def test_dg_mega_rows_build_counts_overflow(mega_emu, monkeypatch):
    """rows = 6: spare budget from $DSV41_MEGA_SPARE, the overflow counter wired; with S = 1 any step with two cold
    unique experts in a layer overflows (counted, never silent)."""
    from runtime.static_step import StaticStep

    monkeypatch.setenv(DM.SPARE_ENV, "1")
    model, pool, _, _, _ = mega_emu("mini4")
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-mega", _allow_cpu=True)
    assert st.spare_slots == 1 and st.dg.S == 1 and st._ovf_on and st.dg.ovf is st.ovf_step
    D = model.layers[0].ffn.dim
    g = _cpu_gen(5)
    over = 0
    with torch.inference_mode(), st._active():
        st.ovf_step.zero_()
        for sm in st.moes:
            x = torch.randn(6, D, generator=g, device="cpu").to(torch.get_default_dtype())
            sm(x)
            over += int(sm.last["n_over"])
            assert (sm.last["topk"] == -1).sum() >= int(sm.last["n_over"])
    assert int(st.ovf_step) == over
    st.detach()


def test_dg_mega_refusals(mega_emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _, _, _ = mega_emu("mini")
    monkeypatch.setenv(DM.L1_STORE_ENV, "inplace")
    with pytest.raises(NotImplementedError, match="M2-L3-ABI-FINDINGS"):
        StaticStep(model, pool, max_pos=127, moe_impl="dg-mega", _allow_cpu=True)
    monkeypatch.setenv(DM.L1_STORE_ENV, "bogus")
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=127, moe_impl="dg-mega", _allow_cpu=True)
    monkeypatch.delenv(DM.L1_STORE_ENV)
    with pytest.raises(ValueError):
        StaticStep(model, None, max_pos=127, moe_impl="dg-mega", _allow_cpu=True)


def test_hbm_plan_real_model_l1_copy_does_not_fit():
    """The real model's L1 copy (40 layers x (295 hot + 16 spare) x 2 x 2304 x 2560 B) is 136.7 GiB: dg-mega's 'copy'
    store can never run there; the SF copy alone is the dg-inplace pool store's size."""
    p = DM.hbm_bytes(40, [295 + 16] * 40, 5120, 2304, "copy")
    assert p["l1_copy_GiB"] > 100 and 10 < p["sf_copy_GiB"] < 14
    assert DM.hbm_bytes(40, [311] * 40, 5120, 2304, "inplace")["l1_copy_B"] == 0


# ------------------------------------------------------------------------------------------------ M3-N1b: downstream --dg-path
def _numerics():
    import importlib
    bench = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    return importlib.import_module("dg_mega_numerics")


def test_downstream_dg_path_args_and_arm_names():
    N = _numerics()
    ap = N.build_parser()
    d0 = ap.parse_args(["downstream"])
    assert d0.dg_path is None and d0.jit_cache is None and N.mega_arm_name(d0.dg_path) == "mega"
    d1 = ap.parse_args(["downstream", "--dg-path", "scratch/dg-bf16q", "--jit-cache", "scratch/jit"])
    assert (d1.dg_path, d1.jit_cache) == ("scratch/dg-bf16q", "scratch/jit")
    assert N.mega_arm_name(d1.dg_path) == "mega_bf16q"
    y = ap.parse_args(["layers", "--dg-path", "a", "--jit-cache", "b"])           # same options as layers
    assert (y.dg_path, y.jit_cache) == ("a", "b")
    r = ap.parse_args(["report", "--downstream", "s.json", "--downstream-patched", "p.json"])
    assert (r.downstream, r.downstream_patched) == ("s.json", "p.json")
    # stock run touches nothing
    assert N.activate_dg(d0) is None


def test_activate_dg_uses_layers_mechanism_before_any_import(tmp_path, monkeypatch):
    N = _numerics()
    for m in [m for m in sys.modules if m == "deep_gemm" or m.startswith("deep_gemm.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    root = tmp_path / "dg"
    (root / "deep_gemm" / os.path.dirname(N.CUH)).mkdir(parents=True)
    (root / "deep_gemm" / "__init__.py").write_text("__version__ = 'fake-patched'\n")
    cuh = root / "deep_gemm" / N.CUH
    cuh.write_text("// unpatched\n")
    a = N.build_parser().parse_args(["downstream", "--dg-path", str(root), "--jit-cache", str(tmp_path / "jit")])
    with pytest.raises(SystemExit, match="lacks the patch"):                       # a copy without the patch is refused
        N.activate_dg(a)
    monkeypatch.delitem(sys.modules, "deep_gemm")
    cuh.write_text(f"// {N.PATCH_TAG}\n")
    info = N.activate_dg(a)
    try:
        assert info["method"] == "copy" and info["patched_source"] is True
        assert sys.modules["deep_gemm"].__version__ == "fake-patched"
        assert os.environ["DG_JIT_CACHE_DIR"] == str(tmp_path / "jit") and (tmp_path / "jit").is_dir()
        with pytest.raises(SystemExit, match="already imported"):                  # too late: refuses, never silent
            N.activate_dg(a)
    finally:
        sys.modules.pop("deep_gemm", None)
    no_jit = N.build_parser().parse_args(["downstream", "--dg-path", str(root)])
    with pytest.raises(SystemExit, match="--jit-cache"):
        N.activate_dg(no_jit)


def test_downstream_activates_dg_before_other_imports():
    """cmd_downstream's first statement is activate_dg(a): nothing imports deep_gemm ahead of it."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench",
                            "dg_mega_numerics.py")).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "cmd_downstream")
    first = fn.body[0] if not (isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)) \
        else fn.body[1]
    assert isinstance(first, ast.Assign) and isinstance(first.value, ast.Call) \
        and first.value.func.id == "activate_dg"


def test_report_shows_stock_and_patched_downstream_arms():
    N = _numerics()
    tf = {"positions": 1476, "flips_pg61": 3, "flips_w_pg61": 0.5, "flips_base": 2, "flips_w_base": 0.25,
          "max_abs_dlp_base": 0.1}
    base = {"label": "base", "arm": "base", "layers": None, "fit": None, "tf": None}
    stock = {"summary": [base, {"label": "mega-L0-10", "arm": "mega", "layers": [0, 10], "fit": True, "tf": tf}]}
    patched = {"summary": [base, {"label": "mega_bf16q-L0-10", "arm": "mega_bf16q", "layers": [0, 10], "fit": True,
                                  "tf": dict(tf, flips_w_pg61=0.125)}]}
    text = "\n".join(N.report_lines([], [], stock, None, patched))
    assert text.count("| base ") == 1
    i, j = text.index("| mega (mega-L0-10)"), text.index("| mega_bf16q (mega_bf16q-L0-10)")
    assert i < j and "3 (0.5)" in text and "3 (0.125)" in text
    # patched only (stock receipt missing) still lists its base arm and the bf16q arm
    only = "\n".join(N.report_lines([], [], None, None, patched))
    assert "| base " in only and "mega_bf16q" in only
    # no downstream receipt at all: unchanged behaviour
    assert "flips vs Pg61" not in "\n".join(N.report_lines([], [], None, None))
