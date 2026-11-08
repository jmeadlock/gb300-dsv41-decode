"""Mac CPU tests of M1-T10-MOE (docs/HANDOFF-M1-T10-MOE.md): the 6-row MoE's dedup, staging and GEMM call shape.
`.venv/bin/python -m pytest -q tests/test_moe_rows_t10_cpu.py`

  * m1_dedup_warp (tests/triton_shim.py, program by program) == the torch mirror rows_layout() on every output, for
    every layout (contiguous / psum / masked) x sf_store step / pool, with heavy cross-row repeats, zero-cold and
    all-cold tables and overflow; and the layouts' semantics checked independently: contiguous tags, psum ends
    (group g = rows [align_up(end[g-1]), end[g]), the vendored DeepGEMM test's convention), masked_m and blocks,
    the work list (staged cold slots in compact order, then -- step SF -- the used hot slots), the overflow count;
  * the warp kernel's failure on the default path falls back to m1_dedup_routes once (reported); an explicit
    $DSV41_DEDUP_IMPL re-raises;
  * m1_stage_rows (shim) == stage_rows_emulated; the emulation moves exactly what the 1-row-era stage_inplace moved
    for the same slots (cold bytes + scales; hot scales only under "step") and nothing else;
  * the 6-row dg-inplace StaticStep in every call shape (contiguous 128 / 16, psum, masked) and SF store (pool,
    step) == dg-masked, torch.equal on logits, main_hidden and caches; the GEMM saw the layout's inputs; one staging
    launch per layer over the work list; a call shape that fails its build check falls back to contiguous 128;
  * sf_store "pool": one packed copy shared by the 1-row and the 6-row builds, rebuilt when the pool's storage moves.
"""
from __future__ import annotations

import sys

import pytest
import torch

from runtime import dg_inplace as DI
from runtime import moe_rows as MR
from test_dg_inplace_cpu import _synthetic_layer
from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens, emu,  # noqa: F401
                                  rows_dg, rows_glue)


@pytest.fixture(scope="module")
def KM():
    import triton_shim

    mod = triton_shim.load_module_under_shim("runtime/moe_rows.py", "moe_rows_t10_shim")
    mod.KERNELS_ON_CPU = True
    return mod


def _routing(g, R_, k, E, case):
    if case == "same":            # every row the same k experts: U = k, counts R
        base = torch.randperm(E, generator=g)[:k]
        idx = torch.stack([base[torch.randperm(k, generator=g)] for _ in range(R_)])
    else:
        idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R_)])
    return idx, torch.rand(R_, k, generator=g)


def _gtab(g, E, case):
    if case == "zero_cold":
        return torch.arange(E, dtype=torch.int32)
    if case == "all_cold":
        return torch.full((E,), -1, dtype=torch.int32)
    hot = torch.rand(E, generator=g) < 0.5
    gt = torch.full((E,), -1, dtype=torch.int32)
    gt[hot] = torch.arange(int(hot.sum()), dtype=torch.int32)
    return gt


def _inpl(gtab, S, layout, step_sf, BR, R_, k, G_extra=0):
    H = int((gtab >= 0).sum())
    G = H + S + G_extra
    P = R_ * k
    return {"gtab": gtab, "H": H, "S": S, "dummy": torch.tensor(0), "has_dummy": H > 0, "v2": True,
            "layout": layout, "G": G, "gscr": G, "step_sf": step_sf,
            "mi": torch.full((P * BR,), -1, dtype=torch.int32) if layout == "contiguous" else None,
            "gl": torch.full((G,), -5, dtype=torch.int32), "ovf": torch.zeros(1, dtype=torch.int32)}


def _fresh(inpl):
    return dict(inpl, mi=None if inpl["mi"] is None else inpl["mi"].clone(), gl=inpl["gl"].clone(),
                ovf=inpl["ovf"].clone())


def _check_semantics(m, ext, inpl, BR, R_):
    """The layouts' contracts from the base mapping, independently of rows_layout's formulas."""
    uniq, counts, slot, rank = (m[k_].tolist() for k_ in ("uniq", "counts", "slot", "rank"))
    gt, H, S = inpl["gtab"].tolist(), inpl["H"], inpl["S"]
    P = len(uniq)
    cold_order = [u for u in range(P) if counts[u] > 0 and gt[uniq[u]] < 0]       # ascending id = slot order
    grp, ok = {}, {}
    for u in range(P):
        if counts[u] == 0:
            continue
        if gt[uniq[u]] >= 0:
            grp[u], ok[u] = gt[uniq[u]], True
        else:
            c = cold_order.index(u)
            grp[u], ok[u] = H + c, c < S
    assert int(ext["n_over"]) == sum(1 for u in ok if not ok[u])
    staged = [u for u in cold_order if ok[u]]
    want_wl = staged + ([u for u in range(P) if counts[u] > 0 and gt[uniq[u]] >= 0] if inpl["step_sf"] else [])
    assert ext["wl"].tolist() == want_wl + [-1] * (P - len(want_wl))
    arow = ext["arow"].tolist()
    lay = inpl["layout"]
    G = inpl["G"]
    okpairs = [i for i in range(P) if ok[slot[i]]]
    assert len(set(arow[i] for i in okpairs)) == len(okpairs)                    # distinct A rows
    if lay == "contiguous":
        tags = ext["tags"].tolist()
        for u in range(P):
            want = [grp[u] if (u in ok and ok[u] and r < counts[u]) else -1 for r in range(R_)]
            assert tags[u] == want, (u, tags[u], want)
        for i in range(P):
            assert arow[i] == slot[i] * BR + rank[i]
    elif lay == "psum":
        end = ext["gl"].tolist()
        assert len(end) == G
        for gidx in range(G):
            start = 0 if gidx == 0 else -(-end[gidx - 1] // BR) * BR
            rows_g = sorted(arow[i] for i in okpairs if grp[slot[i]] == gidx)
            assert rows_g == list(range(start, end[gidx])), (gidx, rows_g, start, end[gidx])
    else:
        mm = ext["gl"].tolist()
        for gidx in range(G):
            assert mm[gidx] == sum(counts[u] for u in ok if ok[u] and grp[u] == gidx)
        for i in range(P):
            if ok[slot[i]]:
                assert arow[i] == grp[slot[i]] * BR + rank[i]
            else:
                assert arow[i] // BR == inpl["gscr"]
    pm = ext["posmat"].view(-1).tolist()                       # posmat: row r's k A rows (ascending id order)
    prow = m["prow"].tolist()
    for r in range(R_):
        assert sorted(pm[r * len(pm) // R_:(r + 1) * len(pm) // R_]) == sorted(arow[i] for i in range(P) if prow[i] == r)


# ------------------------------------------------------------------------------------------------ the dedup
@pytest.mark.parametrize("layout", ["contiguous", "psum", "masked"])
@pytest.mark.parametrize("step_sf", [False, True])
@pytest.mark.parametrize("case", ["random", "same", "zero_cold", "all_cold"])
def test_dedup_warp_v2_equals_mirror_and_layout_contract(KM, layout, step_sf, case):
    g = torch.Generator().manual_seed(hash((layout, step_sf, case)) % 2 ** 31)
    for t in range(6):
        R_, k = [(6, 6), (6, 4), (8, 8), (2, 3)][t % 4]
        E = int(torch.randint(k + 1, 3 * k + 2, (1,), generator=g))
        idx, w = _routing(g, R_, k, E, case)
        gtab = _gtab(g, E, case)
        S = int(torch.randint(1, 5, (1,), generator=g))
        BR = 128 if layout == "masked" else [R_, 8, 16, 32][t % 4] if R_ <= 8 else 16
        inpl0 = _inpl(gtab, S, layout, step_sf, BR, R_, k, G_extra=t % 3)
        m = MR.dedup_routes(idx, w, BR)
        ext = MR.rows_layout(m, inpl0, BR, R_, k)
        _check_semantics(m, ext, inpl0, BR, R_)
        outs = []
        for mod in (KM, MR):
            inpl = _fresh(inpl0)
            buf = mod.DedupBuffers(R_, k, "cpu")
            n0 = len(KM.m1_dedup_warp.launches)
            got = mod.dedup_fused(buf, idx, w, BR, inpl=inpl, impl="warp")
            assert len(KM.m1_dedup_warp.launches) - n0 == (1 if mod is KM else 0)
            assert got["layout_done"]
            outs.append((got, inpl))
        for got, inpl in outs:
            for key in ("uniq", "counts", "valid", "slot", "rank", "prow", "pw"):
                assert torch.equal(got[key], m[key]), (key, mod)
            for key in ("arow", "posmat", "wl", "sidx", "ids_stage"):
                assert torch.equal(got[key], ext[key]), key
            if layout == "contiguous":
                mi = inpl["mi"].view(R_ * k, BR)
                assert torch.equal(mi[:, :R_], ext["tags"]) and bool((mi[:, R_:] == -1).all())
            else:
                assert torch.equal(inpl["gl"], ext["gl"])
            assert int(inpl["ovf"]) == int(ext["n_over"])


def test_dedup_warp_v1_matches_matrix_kernel(KM):
    """inpl without v2 (the M1-T10-PORT contract): the warp kernel writes what m1_dedup_routes writes."""
    g = torch.Generator().manual_seed(7)
    for t in range(20):
        R_, k = (6, 6) if t % 2 else (7, 4)
        E = int(torch.randint(k, 3 * k, (1,), generator=g))
        idx, w = _routing(g, R_, k, E, "random")
        gtab = _gtab(g, E, "random")
        inpl = {"gtab": gtab, "H": E, "S": 3, "dummy": torch.tensor(1), "has_dummy": bool(t % 3)}
        res = []
        for impl in ("warp", "matrix"):
            buf = KM.DedupBuffers(R_, k, "cpu")
            ctr = [torch.full((R_ * k,), -7, dtype=dt) for dt in (torch.int32, torch.long, torch.long)]
            got = KM.dedup_fused(buf, idx, w, R_ + t % 3, mm=ctr[0], ids_step=ctr[1], uval=ctr[2], inpl=inpl,
                                 impl=impl)
            res.append(({k_: v.clone() for k_, v in got.items() if torch.is_tensor(v)}, [c.clone() for c in ctr]))
        (a, ca), (b, cb) = res
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), key
        assert all(torch.equal(x, y) for x, y in zip(ca, cb))


def test_dedup_warp_failure_falls_back_once(KM, monkeypatch, capsys):
    calls = []

    class Boom:
        launches = []

        def __getitem__(self, grid):
            def run(*a, **k):
                calls.append(1)
                raise RuntimeError("no tl.sort on this Triton")
            return run

    monkeypatch.setattr(KM, "m1_dedup_warp", Boom())
    monkeypatch.setitem(KM._WARP, "error", None)
    monkeypatch.delenv(MR.DEDUP_IMPL_ENV, raising=False)
    idx = torch.tensor([[0, 1, 2], [1, 2, 3]])
    w = torch.rand(2, 3)
    for _ in range(2):
        buf = KM.DedupBuffers(2, 3, "cpu")
        got = KM.dedup_fused(buf, idx, w, 4)
        ref = MR.dedup_routes(idx, w, 4)
        assert all(torch.equal(got[k_], ref[k_]) for k_ in ref)
    assert len(calls) == 1 and "no tl.sort" in KM.dedup_describe()["warp_error"]
    assert KM.dedup_describe()["impl"] == "matrix" and "m1_dedup_warp failed" in capsys.readouterr().err
    monkeypatch.setitem(KM._WARP, "error", None)
    monkeypatch.setenv(MR.DEDUP_IMPL_ENV, "warp")
    with pytest.raises(RuntimeError, match="no tl.sort"):
        KM.dedup_fused(KM.DedupBuffers(2, 3, "cpu"), idx, w, 4)
    monkeypatch.setenv(MR.DEDUP_IMPL_ENV, "bogus")
    with pytest.raises(ValueError, match="DSV41_DEDUP_IMPL"):
        MR.dedup_impl()


def test_arow_key_range_guard():
    with pytest.raises(ValueError, match="16-bit"):
        MR._check_arow_range({"layout": "masked", "gscr": 600}, 128, 36)
    MR._check_arow_range({"layout": "masked", "gscr": 400}, 128, 36)
    MR._check_arow_range({"layout": "contiguous"}, 256, 64)


# ------------------------------------------------------------------------------------------------ the staging
def _stage_case(step_sf, S, use_kernel=False, N=8, block_bytes=64, nprog=2):
    """The synthetic layer (E 8, H 5 hot [6,1,3,0,5], cold [2,4,7], 3 spare slots); routing of 3 rows x 3 experts
    using experts 0, 1, 2, 4, 7 (cold 2, 4, 7 -> compact 0, 1, 2)."""
    from test_dg_inplace_cpu import _static_kernels_under_shim

    L = _synthetic_layer(N=N, seed=3)
    H, KS = L["H"], L["KS"]
    sf = {p: DI.packed_sf_zeros(H + L["S"], N, KS // 4, "cpu") for p in DI.PROJS}
    dsts = [L["full"][p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64)
    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7]])
    m = MR.dedup_routes(idx, torch.rand(3, 3), 4)
    inpl = {"gtab": L["gtab"], "H": H, "S": S, "dummy": torch.tensor(6), "has_dummy": True, "v2": True,
            "layout": "contiguous", "G": H + L["S"], "gscr": 0, "step_sf": step_sf}
    ext = MR.rows_layout(m, inpl, 4, 3, 3)
    n_work = 9 if step_sf else S
    args = (L["ptab6"], m["uniq"], L["gtab"], dtab, dsts, ext["sidx"], ext["wl"], H, N * L["KH"],
            ((N, KS // 4), (N, KS // 4)), n_work)
    if use_kernel:
        SK, shim = _static_kernels_under_shim()
        shim.ADDRESSES.clear()
        shim.ADDRESSES.register(*dsts, *L["src_w"].values(), *L["src_s"].values())
        try:
            SK.stage_rows(*args, block_bytes=block_bytes, nprog=nprog)
        finally:
            shim.ADDRESSES.clear()
        assert SK.m1_stage_rows.launches[-1]["grid"] == (n_work, nprog, 3)
    else:
        DI.stage_rows_emulated(L["reg"].__getitem__, *args)
    return L, sf, dsts, m, ext


@pytest.mark.parametrize("step_sf", [False, True])
@pytest.mark.parametrize("S", [3, 1])
def test_stage_rows_emulated_moves_cold_bytes_and_scales_once(step_sf, S):
    L, sf, dsts, m, ext = _stage_case(step_sf, S)
    H = L["H"]
    cold = [2, 4, 7]
    for p in DI.PROJS:
        w = L["full"][p]
        for c, e in enumerate(cold):
            if c < S:
                assert torch.equal(w[H + c], L["src_w"][(p, e)])
        assert not w[H + S:].any()
        for s_, e in enumerate(L["hot"]):                                        # hot weights untouched
            assert torch.equal(w[s_], L["src_w"][(p, e)])
        got = DI.unpack_sf(sf[p])
        written = {H + c for c in range(S)}
        for c, e in enumerate(cold[:S]):
            assert torch.equal(got[H + c], L["src_s"][(p, e)])
        if step_sf:                                                              # used hot experts 0 and 1 only
            for e in (0, 1):
                assert torch.equal(got[L["gtab"][e]], L["src_s"][(p, e)])
            written |= {int(L["gtab"][0]), int(L["gtab"][1])}
        assert not got[[g_ for g_ in range(got.size(0)) if g_ not in written]].any()
    assert int(ext["n_over"]) == 3 - S


@pytest.mark.parametrize("step_sf", [False, True])
@pytest.mark.parametrize("N,block_bytes,nprog", [(8, 64, 2), (132, 64, 3)])
def test_stage_rows_kernel_body_equals_contract(step_sf, N, block_bytes, nprog):
    outs = []
    for use_kernel in (False, True):
        L, sf, dsts, m, ext = _stage_case(step_sf, 2, use_kernel=use_kernel, N=N, block_bytes=block_bytes,
                                          nprog=nprog)
        outs.append([t.clone() for t in dsts])
    for i, (a_, b_) in enumerate(zip(*outs)):
        assert torch.equal(a_, b_), i


def test_stage_rows_validation():
    from test_dg_inplace_cpu import _static_kernels_under_shim

    SK, _ = _static_kernels_under_shim()
    L = _synthetic_layer()
    with pytest.raises(ValueError, match="n_work"):
        SK.stage_rows(L["ptab6"], torch.zeros(4, dtype=torch.long), L["gtab"], torch.zeros(6, dtype=torch.long),
                      [L["full"][p] for p in DI.PROJS], torch.zeros(4, dtype=torch.long),
                      torch.zeros(4, dtype=torch.long), 5, 128, ((8, 2), (8, 2)), 5)


# ------------------------------------------------------------------------------------------------ the 6-row step
def _psum_masked_fakes(fake):
    """The rows_dg fake deep_gemm + the psum contiguous call and the masked call with packed int32 SF (the layouts
    M1-T10-MOE adds), with the contiguous fake's per-row math; the original calls are kept for everything else."""
    from test_static_emulated_cpu import _fp4_deq

    E8 = torch.float8_e8m0fnu
    contig, masked = fake.m_grouped_fp8_fp4_gemm_nt_contiguous, fake.m_grouped_fp8_fp4_gemm_nt_masked
    fake.psum_calls, fake.masked_packed_calls = [], []

    def row(xq, xs, B, r):
        K = xq.size(-1)
        A = (xq[r:r + 1].float().view(1, K // 32, 32) * xs[r:r + 1].view(1, -1, 1)).view(1, K)
        return (A @ B.t()).to(torch.bfloat16)

    def contiguous(a, b, d, layout, recipe_a=None, recipe_b=None, use_psum_layout=False, **kw):
        if not use_psum_layout:
            return contig(a, b, d, layout, recipe_a=recipe_a, recipe_b=recipe_b, **kw)
        (xq, xs), (W, S) = a, b
        if xs.dtype == torch.int32:   # dg_sf_pack: packed activation SF, int32 [M, K/128] MN-major
            assert tuple(xs.shape) == (xq.size(0), xq.size(1) // 128) and xs.stride(0) == 1
            fake.packed_a.append(xq.size(0))
            xs = DI.unpack_sf(xs).view(E8).float()
        assert not kw and recipe_a == (1, 32) and recipe_b == (1, 32) and xs.dtype == torch.float32
        assert DI.is_mn_major_packed(S, W.size(0), W.size(1), xq.size(1) // 128)
        G, BM = W.size(0), fake.align[0]
        assert layout.dtype == torch.int32 and tuple(layout.shape) == (G,)
        end = [int(v) for v in layout.tolist()]
        d.fill_(float("nan"))
        start = 0
        for g_ in range(G):
            if end[g_] > start:
                B = _fp4_deq(W[g_]) * DI.unpack_sf(S[g_]).view(E8).float().repeat_interleave(32, -1)
                for r in range(start, end[g_]):
                    d[r:r + 1] = row(xq, xs, B, r)
            start = -(-max(end[g_], start) // BM) * BM
        fake.psum_calls.append((G, BM, end))

    def masked_any(a, b, d, masked_m, expected_m, recipe_a=None, recipe_b=None, **kw):
        (xq, xs), (W, S) = a, b
        if S.dtype != torch.int32:
            return masked(a, b, d, masked_m, expected_m, recipe_a=recipe_a, recipe_b=recipe_b, **kw)
        G, max_m, K = xq.shape
        assert not kw and max_m == 128 and W.size(0) == G and DI.is_mn_major_packed(S, G, W.size(1), K // 128)
        mm = [int(v) for v in masked_m.tolist()]
        for g_ in range(G):
            if mm[g_]:
                B = _fp4_deq(W[g_]) * DI.unpack_sf(S[g_]).view(E8).float().repeat_interleave(32, -1)
                for r in range(mm[g_]):
                    d[g_, r:r + 1] = row(xq[g_], xs[g_], B, r)
            d[g_, mm[g_]:] = float("nan")
        fake.masked_packed_calls.append((G, int(expected_m), mm))

    fake.m_grouped_fp8_fp4_gemm_nt_contiguous = contiguous
    fake.m_grouped_fp8_fp4_gemm_nt_masked = masked_any
    return fake


@pytest.fixture
def rows_t10(rows_dg, monkeypatch):
    def build(kind):
        model, pool, fk, fake = rows_dg(kind)
        return model, pool, fk, _psum_masked_fakes(fake)

    for env in (DI.ROWS_GEMM_ENV, DI.ROWS_SF_STORE_ENV, DI.PSUM_ALIGN_ENV, DI.SF_STORE_ENV):
        monkeypatch.delenv(env, raising=False)
    return build


@pytest.mark.parametrize("gemm,store", [("contiguous", "pool"), ("contiguous", "step"), ("contiguous16", "pool"),
                                        ("psum", "pool"), ("psum", "step"), ("masked", "pool")])
def test_static6_dg_inplace_call_shapes_equal_dg_masked(rows_t10, gemm, store, monkeypatch):
    from runtime import expert_pool as EP
    from runtime.static_step import StaticStep

    if store == "step":                               # the per-step re-pack store is the SEPARATE-layout path
        monkeypatch.setenv(EP.POOL_SF_ENV, "separate")
    model, pool, fk, fake = rows_t10("mini4")
    hid = (len(model.layers) - 1,)
    n = 3
    ids, toks = _tf_tokens(18, 6 * n, 77)
    st = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-inplace", rows_moe_gemm=gemm,
                    rows_sf_store=store, _allow_cpu=True)
    b = st.dg
    lay, al = DI.parse_rows_gemm(gemm)
    assert b.layout == lay and b.sf_store == store and st.describe()["rows_moe_gemm"] == gemm
    if al:
        assert b.align == al
    if gemm != "contiguous":
        assert b.layout_check["equal"] and "fallback" not in b.layout_check, b.layout_check
    else:
        assert b.layout_check is None
    n_layers = len(model.layers)
    p0, m0, s0 = len(fake.psum_calls), len(fake.masked_packed_calls), fk.calls.get("stage_rows", 0)
    c0 = len(fake.contig_calls)
    a_lg, a_h, a_s = _static_run(model, st, ids, toks, n)
    assert fk.calls["stage_rows"] - s0 == n_layers * n                          # one staging launch per layer
    nw = set(fk.calls["stage_rows_n_work"][-n_layers * n:])
    assert nw == ({6 * st.topk} if store == "step" else {st.spare_slots})
    if lay == "psum":
        assert len(fake.psum_calls) - p0 == 3 * n_layers * n and len(fake.contig_calls) == c0
    elif lay == "masked":
        calls = fake.masked_packed_calls[m0:]
        assert len(calls) == 3 * n_layers * n and all(c[1] == 6 for c in calls)
        assert all(max(c[2]) <= 6 for c in calls) and sum(c[2][0] for c in calls) >= 0
    else:
        assert len(fake.contig_calls) - c0 == 3 * n_layers * n
    ov = st.overflow_counts()
    assert ov["steps"] == 0 and ov["experts"] == 0
    st.detach()
    m2 = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-masked", _allow_cpu=True)
    b_lg, b_h, b_s = _static_run(model, m2, ids, toks, n)
    m2.detach()
    _assert_rows_equal(a_lg, b_lg, f"6-row dg-inplace {gemm}/{store} vs dg-masked")
    assert all(torch.equal(x, y) for x, y in zip(a_h, b_h))
    for k_ in b_s:
        assert torch.equal(a_s[k_], b_s[k_]), k_


@pytest.mark.parametrize("gemm", ["contiguous", "psum", "masked"])
def test_fused_glue_rows_sf_pack_per_call_shape(rows_glue, monkeypatch, gemm):
    """glue_impl fused at 6 rows on dg-inplace in each call shape == glue_impl ref (same call shape): dg_sf_pack is
    ported to R rows -- on (self-checked) for contiguous / psum, the GEMMs then take the packed activation SF; n/a
    for masked (fp32 SF)."""
    from runtime.static_step import StaticStep

    for env in (DI.ROWS_GEMM_ENV, DI.ROWS_SF_STORE_ENV, DI.PSUM_ALIGN_ENV):
        monkeypatch.delenv(env, raising=False)
    model, pool, _, fake = rows_glue("mini4")
    _psum_masked_fakes(fake)
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 5)
    st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", rows_moe_gemm=gemm,
                    glue_impl="fused", _allow_cpu=True)
    d = st.glue.describe()
    assert st.dg.layout == gemm
    if gemm == "masked":
        assert d["clusters"]["dg_sf_pack"].startswith("n/a: the masked call shape") and not st.dg.sf_pack
    else:
        assert d["clusters"]["dg_sf_pack"] == "on" and st.dg.sf_pack, d["clusters"]
        assert d["calibration"]["r4"]["verify_dg_sf_pack"] == "equal"
    assert d["clusters"]["dedup"] == "on"
    n0 = len(fake.packed_a)
    f_lg, f_h, f_s = _static_run(model, st, ids, toks, 2)
    assert (len(fake.packed_a) > n0) == (gemm != "masked")
    st.detach()
    ref = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", rows_moe_gemm=gemm,
                     _allow_cpu=True)
    r_lg, r_h, r_s = _static_run(model, ref, ids, toks, 2)
    ref.detach()
    _assert_rows_equal(f_lg, r_lg, f"6-row fused glue vs ref glue ({gemm})")
    assert all(torch.equal(x, y) for x, y in zip(f_h, r_h))
    for k_ in r_s:
        assert torch.equal(f_s[k_], r_s[k_]), k_


def test_packed_rows_codes_at_matches_codes():
    sf = DI.packed_act_sf(10, 8, "cpu", fill_code=0)
    pr = DI.PackedRows(sf, 1, 10)
    g = torch.Generator().manual_seed(1)
    for r in range(10):
        pr.store_codes_torch(r, torch.randint(0, 255, (8,), generator=g))
    rows = torch.tensor([7, 0, 3, 3])
    assert torch.equal(pr.codes_at(rows), pr.codes()[rows])


def test_call_shape_failing_its_check_falls_back(rows_t10, capsys):
    from runtime.static_step import StaticStep

    model, pool, fk, fake = rows_t10("mini")
    psum = fake.m_grouped_fp8_fp4_gemm_nt_contiguous

    def wrong(a, b, d, layout, **kw):          # a psum kernel that is off by one row everywhere
        psum(a, b, d, layout, **kw)
        if kw.get("use_psum_layout"):
            d.copy_(torch.roll(d, 1, 0))

    fake.m_grouped_fp8_fp4_gemm_nt_contiguous = wrong
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", rows_moe_gemm="psum", _allow_cpu=True)
    assert st.dg.layout == "contiguous" and st.dg.align == 128 and st.describe()["rows_moe_gemm"] == "contiguous"
    assert st.dg.layout_check["equal"] is False and "psum -> contiguous" in st.dg.layout_check["fallback"]
    assert "failed its build check" in capsys.readouterr().err
    st.detach()


def test_rows_gemm_validation(rows_t10, monkeypatch):
    from runtime.static_step import StaticStep

    assert DI.parse_rows_gemm(None) == ("contiguous", None) and DI.parse_rows_gemm("contiguous32") == ("contiguous", 32)
    with pytest.raises(ValueError, match="rows moe gemm"):
        DI.parse_rows_gemm("mega")
    model, pool, _, _ = rows_t10("mini")
    with pytest.raises(ValueError, match="rows_moe_gemm"):
        StaticStep(model, pool, max_pos=255, moe_impl="dg-inplace", rows_moe_gemm="psum", _allow_cpu=True)
    with pytest.raises(ValueError, match="rows_moe_gemm"):
        StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-masked", rows_sf_store="pool", _allow_cpu=True)
    monkeypatch.setenv(DI.PSUM_ALIGN_ENV, "4")           # a block must hold the R rows of a group
    with pytest.raises(ValueError, match="alignment 4 < 6"):
        StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", rows_moe_gemm="psum", _allow_cpu=True)


def test_pool_sf_shared_by_one_and_six_row_builds(rows_t10, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _, _ = rows_t10("mini4")
    monkeypatch.setenv(DI.SF_STORE_ENV, "pool")
    s6 = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
    assert s6.dg.sf_store == "pool"
    s1 = StaticStep(model, pool, max_pos=255, moe_impl="dg-inplace", _allow_cpu=True)
    assert s1.dg.sf_store == "pool"
    for l in s6.dg.layers:
        for p in DI.PROJS:
            assert s6.dg.layers[l]["sf"][p].data_ptr() == s1.dg.layers[l]["sf"][p].data_ptr()
            H = s6.dg.layers[l]["H"]
            want = DI.pack_words(pool.hbm[l][p][1])
            assert torch.equal(s6.dg.layers[l]["sf"][p][:H], want)
    d = s6.dg.describe()
    assert d["sf_store"] == "pool" and d["hbm_added_B"]["packed_sf"] > 0 and d["stage_n_work"] == s6.spare_slots
    key = DI.pool_sf_key(pool)
    s1.detach()
    s6.detach()
    pool.add_spare_slots(pool.spare + 1, model)            # storage moves -> a new key, a rebuilt copy
    assert DI.pool_sf_key(pool) != key


def test_glue_census_names_the_warp_dedup():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "glue_census.py")
    src = open(path).read()
    assert '"m1_dedup_warp" if moe_rows.dedup_impl() == "warp"' in src
    assert importlib.util.spec_from_file_location("gc", path) is not None
    assert MR.dedup_impl() in MR.DEDUP_IMPLS and sys.modules.get("vllm") is None


def test_pool_sf_headroom_falls_back_to_step(monkeypatch, capsys):
    """pool SF on the box, the all-or-nothing rule ($DSV41_SF_POOL_LAYERS=all, pinned: the default is now auto):
    allocated only if >= $DSV41_SF_POOL_HEADROOM_GIB (default 2.0 since 11/07, was 6) of HBM stays free, else the
    build uses "step" and says why (the real model's copy is ~12.8 GiB; the 6-row receipts had 17-23 GiB free)."""
    assert DI.SF_POOL_HEADROOM_GIB == 2.0 and DI.SF_POOL_LAYERS_DEFAULT == "auto"     # NEW defaults
    monkeypatch.setenv(DI.SF_POOL_LAYERS_ENV, "all")
    class Pool:
        n_layers, spare = 2, 2
        hot = {0: (1, 2), 1: (3,)}
        hbm_full = {l: {"w1": torch.zeros(4, 1, 1)} for l in range(2)}

    class Stub:
        pool, sf_store, sf_store_why = Pool(), "pool", None
        N = {p: 1024 for p in DI.PROJS}
        C = {p: 256 for p in DI.PROJS}
        dev = torch.device("cuda")

    need = DI.pool_sf_bytes(Stub.pool, Stub.N, Stub.C)
    assert need == 2 * 4 * 3 * 1024 * 256 * 4
    monkeypatch.delenv(DI.SF_POOL_HEADROOM_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (need + 3 * DI.GiB, 250 * DI.GiB))
    s = Stub()
    DI.DGInplaceBuffers._check_sf_hbm(s)
    assert s.sf_store == "pool" and s.sf_store_why is None
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (need + 1 * DI.GiB, 250 * DI.GiB))
    s = Stub()
    DI.DGInplaceBuffers._check_sf_hbm(s)
    assert s.sf_store == "step" and "2.0 GiB headroom" in s.sf_store_why and "[dg-inplace]" in capsys.readouterr().err


def test_alignment_global_left_as_found_for_other_builds(rows_t10):
    """DeepGEMM's mk alignment is process-global; the draft graph (runtime/draft_step.DraftDGBuffers) sets it ONCE at
    init and relies on it. An R-row build in another call shape must not leave its value behind: set around each of
    its own calls only."""
    from runtime.static_step import StaticStep

    model, pool, _, fake = rows_t10("mini4")
    seen = []
    inner = fake.m_grouped_fp8_fp4_gemm_nt_contiguous

    def spy(*a, **k):
        seen.append(fake.align[0])
        return inner(*a, **k)

    fake.m_grouped_fp8_fp4_gemm_nt_contiguous = spy
    fake.align[0] = 128                       # what another build (the draft) set
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", rows_moe_gemm="contiguous32",
                    _allow_cpu=True)
    assert fake.align[0] == 128 and st.dg.align == 32 and st.dg.alignment["now"] == 32
    n0 = len(seen)
    ids, toks = _tf_tokens(17, 12, 3)
    _static_run(model, st, ids, toks, 2)
    assert len(seen) > n0 and set(seen[n0:]) == {32} and fake.align[0] == 128
    st.detach()
