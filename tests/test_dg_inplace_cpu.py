"""Mac CPU tests of T9a-2 (runtime/dg_inplace.py, `--moe-impl dg-inplace`): `.venv/bin/python -m pytest -q tests/`.

What a CPU can check, and does:
  * the packed SF layout: our pack (the pool's e8m0 bytes read as little-endian int32, MN-major) == DeepGEMM's own
    packing semantics (deep_gemm/utils/math.py pack_ue8m0_to_int on the fp32 values, then MN-major) for every code;
    strides; round trip; zero init finite;
  * index math: hot -> pool slot, cold -> H + j; m_indices = -1 except block heads; alignment validation; the
    multi-row layout the probe uses; bf16 ulp distance;
  * the HBM plan on the real model's shapes (fp32 resident scales do not fit; packed resident / step costs);
  * the staging contract (stage_inplace_emulated) on synthetic tensors: only cold weights move, scales per sf_store,
    nothing else written;
  * ExpertPool.add_spare_slots: hot bytes preserved, Experts rebound, reference forward unchanged, idempotent;
  * the whole dg-inplace StaticStep vs the reference forward on the mini checkpoints, with the reference kernels and
    BOTH DeepGEMM calls emulated by the same math (a contract-checking fake `m_grouped_fp8_fp4_gemm_nt_contiguous`
    that reads each block's group from m_indices at its BLOCK_M head, unpacks the int32 SF, and NaN-poisons every row
    tagged -1): `torch.equal` on every position's fp32 logits, both sf_stores, alignments 128 and 64; then dg-masked
    and ref StaticSteps on the same (grown) pool are still exact; fused glue with dg-inplace (gather n/a, the dg
    clusters on, equal to the unfused step);
  * Generator / first_light plumbing: --moe-impl dg-inplace, the "dg-inplace vs dg-masked" drift row.
Skipped here (box only, tests/gpu/test_dg_inplace_gpu.py, bench/dg_contiguous_spike.py): DeepGEMM itself, the Triton
staging kernel, CUDA graphs, real numerics and timing.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import types

import pytest
import torch

from runtime import dg_inplace as DI
from runtime.dg_moe import E8M0_F32
from test_dg_moe_cpu import FakeStaticImpl, _fake_deep_gemm, _faithful_act_quant, _rows
from test_static_emulated_cpu import _check, _fake_static_kernels, _fp4_deq, _ref_run, _static_run, emu  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E8 = torch.float8_e8m0fnu
REAL_SHAPES = {"w1": (2304, 2560), "w3": (2304, 2560), "w2": (5120, 1152)}
REAL_SSHAPES = {"w1": (2304, 160), "w3": (2304, 160), "w2": (5120, 72)}


@pytest.fixture(autouse=True)
def _cpu_default_device(monkeypatch):
    torch.set_default_device("cpu")
    monkeypatch.delenv(DI.ALIGN_ENV, raising=False)
    monkeypatch.delenv(DI.SF_STORE_ENV, raising=False)
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ packed SF
def test_pack_sf_equals_deepgemm_packing_every_code():
    g = torch.Generator().manual_seed(3)
    N, KS = 12, 40
    codes = torch.randint(0, 256, (3, N, KS), generator=g).to(torch.uint8)
    codes[0, 0, :4] = torch.tensor([0, 1, 254, 255], dtype=torch.uint8)  # edges: 2^-127 subnormal, NaN
    codes.view(-1)[: 256] = torch.arange(256, dtype=torch.int32).to(torch.uint8)  # every code somewhere
    ours = DI.pack_sf(codes.view(E8))
    f32 = E8M0_F32[codes.long()]
    theirs = DI.ref_pack_from_f32(f32)
    assert ours.dtype == torch.int32 and tuple(ours.shape) == (3, N, KS // 4)
    assert torch.equal(ours, theirs)
    assert DI.is_mn_major_packed(ours, 3, N, KS // 4) and ours.stride() == theirs.stride() == (N * KS // 4, 1, N)
    assert torch.equal(DI.unpack_sf(ours), codes)
    # word (n, c) = bytes 4c..4c+3 of row n, little-endian
    w = int(ours[1, 5, 7]) & 0xFFFFFFFF
    assert [(w >> (8 * i)) & 0xFF for i in range(4)] == [int(v) for v in codes[1, 5, 28:32]]


def test_packed_sf_zeros_layout_and_errors():
    t = DI.packed_sf_zeros(5, 8, 3, "cpu")
    assert DI.is_mn_major_packed(t, 5, 8, 3) and not t.any()
    assert float(E8M0_F32[0]) == 2.0 ** -127  # code 0 is finite: an unstaged group is tiny, never NaN
    t[2].copy_(DI.pack_words(torch.arange(96, dtype=torch.int32).to(torch.uint8).view(8, 12)))
    assert torch.equal(DI.unpack_sf(t[2:3])[0], torch.arange(96, dtype=torch.int32).to(torch.uint8).view(8, 12))
    assert torch.equal(t[[0, 1, 3, 4]], torch.zeros(4, 8, 3, dtype=torch.int32))
    with pytest.raises(ValueError):
        DI.sf_words(18 * 4 + 2)
    with pytest.raises(ValueError):
        DI.packed_sf_zeros(2, 6, 3, "cpu")
    assert DI.sf_words(160) == 40 and DI.sf_words(72) == 18  # the real w1/w3 and w2 scale rows


# ------------------------------------------------------------------------------------------------ index math
def test_group_ids_m_indices_and_alignment():
    E, H, k = 16, 10, 6
    gtab = torch.full((E,), -1, dtype=torch.int32)
    hot = [3, 0, 7, 9, 1, 12, 15, 4, 8, 2]          # slot s holds expert hot[s] (map order, not sorted)
    for s, e in enumerate(hot):
        gtab[e] = s
    ids = torch.tensor([0, 2, 5, 9, 11, 15])          # sorted routed ids; 5 and 11 are cold
    g = DI.group_ids(ids, gtab, H)
    assert g.tolist() == [1, 9, H + 2, 3, H + 4, 6]
    for a in (1, 32, 64, 128):
        mi = DI.m_indices_for(g, a)
        assert mi.dtype == torch.int32 and mi.numel() == k * a
        assert mi.view(k, a)[:, 0].tolist() == g.tolist() and bool((mi.view(k, a)[:, 1:] == -1).all())
    for bad in (0, 3, 96, 512):
        with pytest.raises(ValueError):
            DI.check_align(bad)
    assert DI.check_align(128) == 128


def test_contiguous_rows_multi_row_layout():
    rows = [[1, 4, 9], [4, 9, 2], [9, 1, 7]]
    lay = DI.contiguous_rows(rows, 8)
    assert lay["experts"] == [1, 2, 4, 7, 9] and lay["m_rows"] == 40
    assert lay["counts"] == {1: 2, 2: 1, 4: 2, 7: 1, 9: 3}
    assert lay["rows"][(0, 9)] == 32 and lay["rows"][(1, 9)] == 33 and lay["rows"][(2, 9)] == 34
    assert lay["rows"][(0, 1)] == 0 and lay["rows"][(2, 1)] == 1 and lay["rows"][(1, 2)] == 8
    assert sum(t >= 0 for t in lay["tag"]) == 9 and lay["tag"][8] == 2 and lay["tag"][9] == -1
    with pytest.raises(ValueError):
        DI.contiguous_rows([[1]] * 3, 2)


def test_bf16_ulp_and_compare():
    a = torch.tensor([1.0, -1.0, 0.0, 3.0, 1e-30], dtype=torch.bfloat16)
    up = (a.view(torch.int16) + 1).view(torch.bfloat16)
    assert DI.bf16_ulp(a, up).tolist() == [1, 1, 1, 1, 1]
    assert int(DI.bf16_ulp(torch.tensor([0.0], dtype=torch.bfloat16), torch.tensor([-0.0], dtype=torch.bfloat16))) == 0
    tiny = torch.tensor([0.0], dtype=torch.bfloat16).view(torch.int16) + 1
    assert int(DI.bf16_ulp(tiny.view(torch.bfloat16), -tiny.view(torch.bfloat16))) == 2  # across zero
    c = DI.compare_bf16(up, a)
    assert not c["equal"] and c["max_ulp"] == 1 and c["mismatch_frac"] == 1.0 and c["n_ulp_gt1"] == 0
    assert DI.compare_bf16(a, a.clone())["equal"]


def test_hbm_plan_real_model():
    p = DI.hbm_plan(40, [295] * 40, 6, REAL_SHAPES, REAL_SSHAPES)
    assert p["G_max"] == 301
    assert p["spare_weight_slots_B"] == 40 * 6 * 3 * 5_898_240 == 4_246_732_800
    assert p["sf_step_shared_B"] == 301 * 3 * 368_640 == 332_881_920
    assert p["sf_packed_resident_B"] == 40 * 301 * 3 * 368_640
    assert p["sf_f32_resident_B"] == 4 * p["sf_packed_resident_B"]
    assert p["e8m0_pool_scales_B"] == 40 * 295 * 3 * 368_640
    assert p["sf_f32_resident_GiB"] > 31 > p["sf_packed_resident_GiB"] + p["spare_weight_slots_GiB"] + 8
    assert p["total_step_GiB"] == pytest.approx(4.265, abs=2e-3) and p["sf_packed_resident_GiB"] == pytest.approx(12.401, abs=2e-3)
    assert p["step_scale_traffic_per_layer_B"] == 2 * 6 * 3 * 368_640


# ------------------------------------------------------------------------------------------------ staging contract
def _synthetic_layer(E=8, H=5, S=3, N=8, KS=8, KH=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    hot = [6, 1, 3, 0, 5]
    cold = [e for e in range(E) if e not in hot]
    full = {p: torch.zeros(H + S, N, KH, dtype=torch.int8) for p in DI.PROJS}
    src_w = {(p, e): torch.randint(-128, 128, (N, KH), generator=g).to(torch.int8) for p in DI.PROJS for e in range(E)}
    src_s = {(p, e): torch.randint(100, 140, (N, KS), generator=g).to(torch.uint8) for p in DI.PROJS for e in range(E)}
    for s, e in enumerate(hot):
        for p in DI.PROJS:
            full[p][s].copy_(src_w[(p, e)])
    reg = {}
    for t in list(src_w.values()) + list(src_s.values()) + [full[p][s] for p in DI.PROJS for s in range(H)]:
        reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
    ptab6 = torch.tensor([[(full[p][hot.index(e)] if e in hot else src_w[(p, e)]).data_ptr() for e in range(E)]
                          for p in DI.PROJS] + [[src_s[(p, e)].data_ptr() for e in range(E)] for p in DI.PROJS],
                         dtype=torch.int64)
    gtab = torch.full((E,), -1, dtype=torch.int32)
    for s, e in enumerate(hot):
        gtab[e] = s
    return dict(E=E, H=H, S=S, N=N, KS=KS, KH=KH, hot=hot, cold=cold, full=full, src_w=src_w, src_s=src_s,
                reg=reg, ptab6=ptab6, gtab=gtab)


@pytest.mark.parametrize("all_scales", [True, False])
def test_stage_emulated_contract(all_scales):
    L = _synthetic_layer()
    H, S, N, KS = L["H"], L["S"], L["N"], L["KS"]
    sf = {p: DI.packed_sf_zeros(H + S, N, KS // 4, "cpu") for p in DI.PROJS}
    dsts = [L["full"][p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64)
    before = [t.clone() for t in dsts]
    ids = torch.tensor([1, 2, 7])        # hot slot 1, cold, cold
    align = 4
    mi = torch.full((S * align,), -1, dtype=torch.int32)
    ids_out = torch.zeros(3, dtype=torch.long)
    DI.stage_inplace_emulated(L["reg"].__getitem__, L["ptab6"], ids, L["gtab"], dtab, dsts, mi, H, align,
                              N * L["KH"], ((N, KS // 4), (N, KS // 4)), all_scales, ids_out)
    assert mi.tolist() == [1, -1, -1, -1, H + 1, -1, -1, -1, H + 2, -1, -1, -1] and ids_out.tolist() == [1, 2, 7]
    for p_i, p in enumerate(DI.PROJS):
        w = L["full"][p]
        assert torch.equal(w[:H], before[p_i][:H])                          # hot slots untouched
        assert torch.equal(w[H], before[p_i][H])                            # spare slot 0 unused (slot 0 is hot)
        assert torch.equal(w[H + 1], L["src_w"][(p, 2)]) and torch.equal(w[H + 2], L["src_w"][(p, 7)])
        got = DI.unpack_sf(sf[p])
        assert torch.equal(got[H + 1], L["src_s"][(p, 2)]) and torch.equal(got[H + 2], L["src_s"][(p, 7)])
        if all_scales:
            assert torch.equal(got[1], L["src_s"][(p, 1)])                  # the hot expert's group, step mode
            others = [g_ for g_ in range(H + S) if g_ not in (1, H + 1, H + 2)]
        else:
            others = [g_ for g_ in range(H + S) if g_ not in (H + 1, H + 2)]
        assert not got[others].any()


# ------------------------------------------------------------------------------------------------ the Triton kernel body
def _static_kernels_under_shim():
    import triton_shim

    with triton_shim.installed():
        spec = importlib.util.spec_from_file_location("static_kernels_under_shim",
                                                      os.path.join(REPO, "runtime", "static_kernels.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["static_kernels_under_shim"] = mod
        spec.loader.exec_module(mod)
    return mod, triton_shim


@pytest.mark.parametrize("all_scales", [True, False])
@pytest.mark.parametrize("N,block_bytes,nprog", [(8, 64, 2), (8, 64, 4), (132, 64, 3), (132, 512, 1)])
def test_stage_kernel_body_equals_contract(all_scales, N, block_bytes, nprog):
    """runtime/static_kernels.py t9a2_stage_inplace executed program by program (tests/triton_shim.py, addresses
    resolved through the registered tensors) == stage_inplace_emulated, on identical synthetic layers: EVEN and masked
    weight loops, several scale tiles, both scale modes, a hot / two cold / hot routing."""
    SK, shim = _static_kernels_under_shim()
    KS, KH = 8, 16
    outs = []
    for use_kernel in (False, True):
        L = _synthetic_layer(N=N, KS=KS, KH=KH, seed=5)
        H, S = L["H"], L["S"]
        sf = {p: DI.packed_sf_zeros(H + S, N, KS // 4, "cpu") for p in DI.PROJS}
        dsts = [L["full"][p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
        dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64)
        ids = torch.tensor([1, 2, 7])
        align = 4
        mi = torch.full((S * align,), -1, dtype=torch.int32)
        ids_out = torch.zeros(3, dtype=torch.long)
        args = (L["ptab6"], ids, L["gtab"], dtab, dsts, mi, H, align, N * KH, ((N, KS // 4), (N, KS // 4)),
                all_scales, ids_out)
        if use_kernel:
            shim.ADDRESSES.clear()
            shim.ADDRESSES.register(*dsts, *L["src_w"].values(), *L["src_s"].values())
            try:
                SK.stage_inplace(*args, block_bytes=block_bytes, nprog=nprog)
            finally:
                shim.ADDRESSES.clear()
            launch = SK.t9a2_stage_inplace.launches[-1]
            assert launch["grid"] == (3, nprog, 3)
        else:
            DI.stage_inplace_emulated(L["reg"].__getitem__, *args)
        outs.append([t.clone() for t in dsts] + [mi.clone(), ids_out.clone()])
    for i, (a_, b_) in enumerate(zip(*outs)):
        assert torch.equal(a_, b_), i
    assert outs[1][-2].tolist()[::4] == [1, 5 + 1, 5 + 2]


# ------------------------------------------------------------------------------------------------ emulated end to end
def _fake_dg_contig():
    """The masked fake of tests/test_dg_moe_cpu.py + `m_grouped_fp8_fp4_gemm_nt_contiguous` and the alignment
    setters. The contiguous fake is the kernel's contract: A [M, K] fp8 + fp32 [M, K/32]; B int8 [G, N, K/2]
    contiguous with int32 packed MN-major SF [G, N, K/128]; d [M, N] bf16; m_indices int32 [M]; block b's group is
    m_indices[b * BLOCK_M] (BLOCK_M = the set alignment); -1 blocks are not computed. Within a valid block, rows tagged
    with the group are computed (the reference emulation's math, one row at a time: the same matmul shape as the
    reference's M = 1) and rows tagged -1 are NaN-poisoned, so any consumer of a padding row fails the comparison."""
    mod = _fake_deep_gemm()
    mod.align = [128]
    mod.contig_calls = []
    mod.packed_a = []

    def set_mk_alignment_for_contiguous_layout(v):
        mod.align[0] = int(v)

    def get_mk_alignment_for_contiguous_layout():
        return mod.align[0]

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, m_indices, recipe_a=None, recipe_b=None, **kw):
        (xq, xs), (W, S) = a, b
        M, K = xq.shape
        G, N, KH = W.shape
        assert not kw and recipe_a == (1, 32) and recipe_b == (1, 32)
        if xs.dtype == torch.int32:   # T9b-4: pre-packed activation SF, int32 [M, K/128] MN-major (TMA-aligned M)
            assert tuple(xs.shape) == (M, K // 128) and xs.stride(0) == 1 and xs.stride(1) >= M, (xs.shape, xs.stride())
            mod.packed_a.append(M)
            xs = DI.unpack_sf(xs).view(E8).float()
        assert xq.dtype == torch.float8_e4m3fn and xs.dtype == torch.float32 and tuple(xs.shape) == (M, K // 32)
        assert W.dtype == torch.int8 and W.is_contiguous() and KH == K // 2
        assert DI.is_mn_major_packed(S, G, N, K // 128), (tuple(S.shape), S.stride())
        assert d.dtype == torch.bfloat16 and tuple(d.shape) == (M, N) and d.is_contiguous()
        assert m_indices.dtype == torch.int32 and tuple(m_indices.shape) == (M,)
        BM = mod.align[0]
        assert M % BM == 0
        mi = [int(v) for v in m_indices.tolist()]
        heads = []
        for b0 in range(0, M, BM):
            grp = mi[b0]
            heads.append(grp)
            if grp < 0:
                d[b0:b0 + BM] = float("nan")
                continue
            assert 0 <= grp < G, (grp, G)
            sc = DI.unpack_sf(S[grp]).view(E8).float()
            B = _fp4_deq(W[grp]) * sc.repeat_interleave(32, -1)
            for r in range(b0, b0 + BM):
                if mi[r] == grp:
                    A = (xq[r:r + 1].float().view(1, K // 32, 32) * xs[r:r + 1].view(1, -1, 1)).view(1, K)
                    d[r:r + 1] = (A @ B.t()).to(torch.bfloat16)
                else:
                    assert mi[r] == -1, (r, mi[r], grp)
                    d[r] = float("nan")
        mod.contig_calls.append((M, G, N, K, heads))

    mod.m_grouped_fp8_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
    mod.set_mk_alignment_for_contiguous_layout = set_mk_alignment_for_contiguous_layout
    mod.get_mk_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
    return mod


def _fill_registry(reg, pool):
    reg.clear()
    for l in range(pool.n_layers):
        for p in ("w1", "w2", "w3"):
            full, s = pool.hbm_full[l][p], pool.hbm[l][p][1]
            for t in list(full) + list(s):
                reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
            for e in pool.cold[l]:
                for kind_ in ("weight", "scale"):
                    t = pool.cold_view(l, e, p, kind_, cuda=False)
                    reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)


def _install_kernels(monkeypatch, pool):
    """A fresh emulated runtime.static_kernels over a registry that re-reads the pool on a miss (the pool's weight
    storage moves when dg-inplace adds its spare slots), with `stage_inplace` = the contract emulation."""
    class Reg(dict):
        def __missing__(self, key):
            _fill_registry(self, pool)
            if key not in self.keys():
                raise KeyError(key)
            return dict.get(self, key)

    reg = Reg()
    _fill_registry(reg, pool)
    fk = _fake_static_kernels(reg)
    fk.calls["stage"] = 0

    def stage_inplace(ptab6, ids, gtab, dtab, dsts, mi, H, align, row_w, sf_dims, all_scales, ids_out=None):
        DI.stage_inplace_emulated(reg.__getitem__, ptab6, ids, gtab, dtab, dsts, mi, H, align, row_w, sf_dims,
                                  all_scales, ids_out)
        fk.calls["stage"] += 1

    fk.stage_inplace = stage_inplace
    monkeypatch.setitem(sys.modules, "runtime.static_kernels", fk)
    return fk


@pytest.fixture
def di_emu(emu, monkeypatch):
    fake = _fake_dg_contig()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)

    def build(kind):
        model, pool, _ = emu(kind)
        R = sys.modules["model"]
        monkeypatch.setattr(R, "act_quant", _faithful_act_quant)
        fk = _install_kernels(monkeypatch, pool)
        return model, pool, fk, fake

    return build


def test_add_spare_slots_preserves_rebinds_and_is_idempotent(di_emu):
    model, pool, _, _ = di_emu("mini4")
    ids = list(range(5, 17))
    want = _ref_run(model, ids, 4)
    old = {(l, p): pool.hbm[l][p][0].clone() for l in range(pool.n_layers) for p in pool.proj_nk}
    gen0 = pool.generation
    added = pool.add_spare_slots(4, model)
    per = sum(n * (k_ // 2) for n, k_ in pool.proj_nk.values())
    assert added == pool.n_layers * 4 * per and pool.spare == 4 and pool.generation == gen0 + 1
    for l in range(pool.n_layers):
        for p in pool.proj_nk:
            full = pool.hbm_full[l][p]
            H = len(pool.hot[l])
            assert full.shape[0] == H + 4 and full.is_contiguous() and not full[H:].any()
            assert torch.equal(pool.hbm[l][p][0], old[(l, p)]) and pool.hbm[l][p][0].data_ptr() == full.data_ptr()
            for s, e in enumerate(pool.hot[l]):
                lin = getattr(model.layers[l].ffn.experts[e], p)
                assert lin.weight.data_ptr() == full[s].data_ptr() and lin.weight.dtype == torch.float4_e2m1fn_x2
                assert lin.weight.scale is lin.scale
    _check(_ref_run(model, ids, 4), want, "reference forward after add_spare_slots")
    assert pool.add_spare_slots(4, model) == 0 and pool.add_spare_slots(2, model) == 0 and pool.generation == gen0 + 1


def _dg_inplace_vs_reference(di_emu, kind, monkeypatch, sf_store, align):
    from runtime.dg_inplace import DGInplaceMoE
    from runtime.static_step import StaticStep

    if sf_store is not None:
        monkeypatch.setenv(DI.SF_STORE_ENV, sf_store)
    if align is not None:
        monkeypatch.setenv(DI.ALIGN_ENV, str(align))
    model, pool, fk, fake = di_emu(kind)
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)
    b, k = st.dg, st.topk
    a = align or 128
    assert st.moe_impl == "dg-inplace" and all(isinstance(m, DGInplaceMoE) for m in st.moes)
    assert b.align == a and fake.align[0] == a and b.alignment["now"] == a and b.sf_store == (sf_store or "step")
    assert pool.spare == k and b.xq[1].shape == (k, a, b.dim) and b.d["w2"].shape == (k, a, b.dim)
    assert not b.sf32 and b.mi.shape == (k * a,) and bool((b.mi == -1).all())
    d = b.describe()
    assert d["impl"] == "dg-inplace" and d["hbm_added_B"]["spare_weight_slots"] > 0 and d["spare_slots"] == k
    g = torch.Generator().manual_seed(11)
    n_steps = 0
    for L in (2, 3, 16, 17):
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        ref = _ref_run(model, ids, 10)
        got = _static_run(model, st, ids, 10)
        _check(got, ref, (kind, "dg-inplace emulated", sf_store, align, L))
        n_steps += 10
    n_layers = len(model.layers)
    assert len(fake.contig_calls) == 3 * n_layers * n_steps and fake.calls == []      # masked never called
    assert fk.calls["stage"] == n_layers * n_steps and fk.calls["gather"] == 0 and fk.calls["gather_tables"] == 0
    G = {len(pool.hot[l]) + k for l in range(n_layers)}
    assert all(c[0] == k * a and c[1] in G and all(h >= 0 for h in c[4]) for c in fake.contig_calls)
    cold_heads = sum(1 for c in fake.contig_calls for h in c[4] if h >= c[1] - k)
    assert cold_heads > 0  # the mini maps route to cold experts: the spare-slot path ran
    for w in (1, 2):  # padding rows never written by a step
        assert not b.xq[w][:, 1:].view(torch.uint8).any() and torch.equal(b.xs[w][:, 1:], torch.ones_like(b.xs[w][:, 1:]))
    assert bool((b.mi.view(k, a)[:, 1:] == -1).all())
    return model, pool, st, fake


@pytest.mark.parametrize("kind", ["mini", "mini4"])
@pytest.mark.parametrize("sf_store,align", [(None, None), ("resident", None), ("step", 64)])
def test_dg_inplace_static_step_equals_reference_under_emulated_gemm(di_emu, kind, monkeypatch, sf_store, align):
    _, _, st, _ = _dg_inplace_vs_reference(di_emu, kind, monkeypatch, sf_store, align)
    st.detach()


def test_dg_inplace_then_masked_then_ref_on_the_grown_pool(di_emu, monkeypatch):
    from runtime.static_step import StaticMoE, StaticStep

    model, pool, fk, fake = di_emu("mini4")
    ids = list(range(5, 21))
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)
    got = _static_run(model, st, ids, 12)
    cold_in, n, k = st.cold_counts()
    st.detach()
    st2 = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", _allow_cpu=True)
    n_c = len(fake.contig_calls)
    got2 = _static_run(model, st2, ids, 12)
    assert len(fake.contig_calls) == n_c and len(fake.calls) == 3 * len(model.layers) * 12
    _check(got, got2, "dg-inplace vs dg-masked (emulated)")
    assert st2.cold_counts()[0] == cold_in and sum(cold_in) > 0
    st2.detach()
    st3 = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    assert st3.dg is None and all(type(m) is StaticMoE for m in st3.moes)
    _check(_static_run(model, st3, ids, 12), _ref_run(model, ids, 12), "ref after dg-inplace")
    st3.detach()
    st4 = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)  # rebuild: no regrowth
    assert st4.dg.describe()["hbm_added_B"]["spare_grown_by_this_build"] == 0
    _check(_static_run(model, st4, ids, 12), got, "dg-inplace rebuilt")
    st4.detach()
    with pytest.raises(ValueError, match="ExpertPool"):
        StaticStep(model, None, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)


# ------------------------------------------------------------------------------------------------ fused glue
def test_fused_glue_with_dg_inplace(gemv_emu, monkeypatch):
    from runtime.fused_glue import DISABLE_ENV
    from runtime.static_step import StaticStep

    monkeypatch.delenv(DISABLE_ENV, raising=False)
    fake = _fake_dg_contig()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    model, pool, _ = gemv_emu("mini4")
    fk = _install_kernels(monkeypatch, pool)
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    d = st.glue.describe()
    assert d["clusters"]["gather"].startswith("n/a") and "dg-inplace" in d["clusters"]["gather"]
    assert {"dg_act", "swiglu_routed", "moe_combine", "swiglu_shared", "hc_post"} <= set(d["on"]), d["clusters"]
    assert all(v == "equal" for k_, v in d["calibration"].items() if k_.startswith("verify_"))
    ids = list(range(7, 23))
    got = _static_run(model, st, ids, 6)
    assert fk.calls["stage"] == 6 * len(model.layers)
    st.detach()
    ref = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", fp8_impl="gemv", _allow_cpu=True)
    _check(_static_run(model, ref, ids, 6), got, "fused glue vs ref glue under dg-inplace")
    ref.detach()
    masked = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                        _allow_cpu=True)
    _check(_static_run(model, masked, ids, 6), got, "dg-inplace vs dg-masked, both fused")
    masked.detach()


from test_gemv_cpu import gemv_emu  # noqa: E402,F401 (fixture)


# ------------------------------------------------------------------------------------------------ plumbing
def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen(mode="eager", moe_impl="ref"):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStaticImpl.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=FakeStaticImpl, moe_impl=moe_impl)


def test_impl_tuples_agree():
    from runtime import dg_moe, generate, static_step

    assert dg_moe.MOE_IMPLS == static_step.MOE_IMPLS == generate.MOE_IMPLS == ("ref", "dg-masked", "dg-inplace")


def test_parity_levels_dg_inplace_has_masked_drift_row():
    fl = _first_light()
    gen = _gen(mode="graph", moe_impl="dg-inplace")
    out = []
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    assert res["verdict"] == "PASS" and res["moe_impl"] == "dg-inplace"
    row = by[("off", "dg-inplace vs dg-masked (same other impls)")]
    # the fake adds the same offset for every non-ref impl: dg-inplace == dg-masked exactly
    assert not row["gate"] and row["all_equal"] and row["level"] == "D" and row["mean_abs_dlogit"] == 0.0
    assert res["inplace_drift"]["first_token_divergence"] is None
    assert by[("off", "dg-inplace vs ref static-eager")]["equal"] == 0          # the usual drift row, vs ref
    assert "static-dgmasked/fills-off" in res["runs"] and "dg-masked" in FakeStaticImpl.built
    assert any(l.startswith("drift dg-inplace vs dg-masked (same other impls): first token divergence none")
               for l in out)
    assert gen.moe_impl == "dg-inplace" and gen.static.moe_impl == "dg-inplace"
    res2 = fl.parity_levels(_gen(mode="graph", moe_impl="dg-masked"), _rows(2), steps=3, log=lambda *a, **k: None)
    assert res2["inplace_drift"] is None and not any(r["level"] == "D" for r in res2["rows"])


def test_first_light_accepts_dg_inplace():
    fl = _first_light()
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--moe-impl", "dg-inplace"])
    assert a.moe_impl == "dg-inplace"
    gen = _gen(mode="static", moe_impl="dg-inplace")
    gen.greedy([5, 9, 12, 3], 3)
    assert gen.static.moe_impl == "dg-inplace" and gen.last_meta["moe_impl"] == "dg-inplace"
