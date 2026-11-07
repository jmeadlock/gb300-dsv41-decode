"""M1-T10-PORT on the Mac: the row axis of every fused-glue kernel, the IDX (pair-row) forms of the R-row MoE glue,
the GEMV's per-row RoPE epilogue and the fused dedup kernel -- each kernel body run through tests/triton_shim.py.

Gate per kernel: ONE launch over R rows == the one-row kernel launched on each row (same per-row arithmetic and
reduction order: the row axis adds nothing but an offset) == the torch mirror (the CPU path), bit for bit; plus the
dedup kernel == runtime/moe_rows.dedup_routes + compact_cold on random routings with cross-row repeats.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

import triton_shim
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from runtime import gemv_mxfp8 as GM
from runtime import moe_rows as MR
from test_fused_glue_r2_cpu import _act, _eq, _freqs, _od, tab  # noqa: F401 (fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def _restore_dtype():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


def _shim(name, alias):
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", name), alias)
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KF():
    return _shim("fused_glue.py", "fg1_m1port_shim")


@pytest.fixture(scope="module")
def KR():
    return _shim("fused_glue_r2.py", "fg2_m1port_shim")


@pytest.fixture(scope="module")
def KG():
    return _shim("gemv_mxfp8.py", "gemv_m1port_shim")


@pytest.fixture(scope="module")
def KM():
    return _shim("moe_rows.py", "moe_rows_m1port_shim")


def _n(fn):
    return len(fn.launches)


# ================================================================================ 0. status
def test_rows_clusters_are_every_cluster_but_the_two_left_off():
    assert set(FG.ROWS_WHY) == {"concat_free", "dg_sf_pack"}
    assert set(FG.ROWS_CLUSTERS) == set(FG.CLUSTERS) - set(FG.ROWS_WHY)


# ================================================================================ 1. dedup (one launch)
def _routing(g, R_, k, E, hot_frac):
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R_)])
    w = torch.rand(R_, k, generator=g)
    gtab = torch.where(torch.rand(E, generator=g) < hot_frac, torch.arange(E, dtype=torch.int32),
                       torch.full((E,), -1, dtype=torch.int32))
    return idx, w, gtab


@pytest.mark.parametrize("R_,k", [(6, 6), (1, 6), (8, 8), (2, 3), (7, 4)])
@pytest.mark.parametrize("inpl", [False, True])
def test_dedup_kernel_equals_torch_mapping(KM, R_, k, inpl):
    g = torch.Generator().manual_seed(100 * R_ + k + inpl)
    for t in range(25):
        E = int(torch.randint(k, 3 * k + 2, (1,), generator=g))       # small E: many cross-row repeats
        idx, w, gtab = _routing(g, R_, k, E, 0.5)
        br = int(torch.randint(R_, R_ + 3, (1,), generator=g))
        ex = None
        if inpl:
            ex = {"gtab": gtab, "H": E, "S": int(torch.randint(1, 5, (1,), generator=g)),
                  "dummy": torch.tensor(int(torch.randint(0, E, (1,), generator=g))), "has_dummy": t % 3 != 0}
        res = []
        for mod in (KM, MR):
            buf = mod.DedupBuffers(R_, k, "cpu")
            ctr = (torch.full((R_ * k,), -7, dtype=torch.int32), torch.full((R_ * k,), -7, dtype=torch.long),
                   torch.full((R_ * k,), -7, dtype=torch.long))
            n0 = _n(KM.m1_dedup_routes)
            m = mod.dedup_fused(buf, idx, w, br, mm=ctr[0], ids_step=ctr[1], uval=ctr[2], inpl=ex)
            assert _n(KM.m1_dedup_routes) - n0 == (1 if mod is KM else 0)
            res.append((m, ctr))
        (mk, ck), (mt, ct) = res
        ref = MR.dedup_routes(idx, w, br)
        for key in ref:
            assert torch.equal(mk[key], ref[key]) and torch.equal(mt[key], ref[key]), (t, key)
        assert all(torch.equal(a, b) for a, b in zip(ck, ct))
        assert torch.equal(ck[0], ref["counts"].int()) and torch.equal(ck[1], ref["uniq"])
        if inpl:
            for key in ("ids_stage", "sidx", "tags"):
                assert torch.equal(mk[key], mt[key]), (t, key)
            host = MR.compact_cold_host(gtab, ref["uniq"].tolist(), ref["counts"].tolist(), ex["H"], ex["S"])
            assert mk["sidx"].tolist() == host["sidx"]


# ================================================================================ 2. R1 kernels (IDX forms, rows)
def test_act_quant_scatter_to_pair_rows(KF, tab):
    g = torch.Generator().manual_seed(1)
    R_, D, A, P = 6, 256, 40, 24
    x = _act(g, R_, D)
    prow = torch.randint(0, R_, (P,), generator=g)
    arow = torch.randperm(A, generator=g)[:P]
    outs = []
    for mod in (KF, FG):
        q = torch.full((A, D), 7.0).to(FP8)
        s = torch.full((A, D // 32), -1.0)
        n0 = _n(KF.t9b2_act_quant_bcast)
        mod.launch_act_quant_bcast(x, q, s, tab, 1, src_rows=prow, dst_rows=arow)
        assert _n(KF.t9b2_act_quant_bcast) - n0 == (1 if mod is KF else 0)
        outs.append((q, s))
    assert _eq(outs[0][0], outs[1][0]) and _eq(outs[0][1], outs[1][1])
    qv, code = FG.act_quant_torch(x.float(), tab)
    assert _eq(outs[0][0].view(torch.uint8)[arow], qv.view(torch.uint8)[prow]) and _eq(outs[0][1][arow], FG.pow2_code(code)[prow])
    rest = torch.ones(A, dtype=torch.bool)
    rest[arow] = False
    assert (outs[0][0].view(torch.uint8)[rest] == torch.full((1,), 7.0).to(FP8).view(torch.uint8)).all() and (outs[0][1][rest] == -1.0).all()


@pytest.mark.parametrize("lim", [0.0, 7.0])
def test_swiglu_on_pair_rows(KF, tab, lim):
    g = torch.Generator().manual_seed(2 + int(lim))
    A, P, I = 48, 24, 192
    gate, up = _act(g, A, I, -3, 7), _act(g, A, I, -3, 7)
    if lim:
        gate[:, 0], up[:, 1], gate[:, 2] = lim, -lim * 1.5, lim * 1.5
    ri = torch.randperm(A, generator=g)[:P]
    w = torch.rand(P, generator=g) * 2
    silu = FG.silu_table("cpu", lim)
    outs = []
    for mod in (KF, FG):
        q = torch.full((A, I), 7.0).to(FP8)
        s = torch.full((A, I // 32), -1.0)
        mod.launch_swiglu_quant(gate, up, w, q, s, silu, tab, lim, rows_idx=ri)
        outs.append((q, s))
    assert _eq(outs[0][0], outs[1][0]) and _eq(outs[0][1], outs[1][1])
    # == the one-row kernel's G form on the gathered rows (its row j = pair j), scattered
    q1, s1 = torch.empty(P, I, dtype=FP8), torch.empty(P, I // 32)
    KF.launch_swiglu_quant(gate.index_select(0, ri), up.index_select(0, ri), w, q1, s1, silu, tab, lim)
    assert _eq(outs[0][0].view(torch.uint8)[ri], q1.view(torch.uint8)) and _eq(outs[0][1][ri], s1)


def test_combine_through_posmat(KF):
    g = torch.Generator().manual_seed(3)
    R_, k, A, N = 6, 6, 60, 320
    rows = _act(g, A, N, -8, 8)
    sh = _act(g, R_, N, -8, 8)
    posmat = torch.stack([torch.randperm(A, generator=g)[:k] for _ in range(R_)])
    outs = []
    for mod in (KF, FG):
        y = torch.empty(R_, N, dtype=BF)
        mod.launch_moe_combine(rows, sh, y, posmat=posmat)
        outs.append(y)
    ref = MR.combine_rows(rows, posmat, sh)
    ref += sh
    assert _eq(outs[0], outs[1]) and _eq(outs[0], ref.to(BF))
    for t in range(R_):   # == the one-row kernel on row t's k rows
        y1 = torch.empty(N, dtype=BF)
        KF.launch_moe_combine(rows[posmat[t]].contiguous(), sh[t].contiguous(), y1)
        assert _eq(outs[0][t], y1)


@pytest.mark.parametrize("modes", [(0, 0), (1, 2)])
def test_o_rope_qdq_rows(KF, tab, modes):
    g = torch.Generator().manual_seed(4 + modes[0])
    R_, H, hd, rd = 6, 4, 64, 32
    o = _act(g, R_, H * hd)
    fr = torch.stack([torch.view_as_real(_freqs(rd, 11 + 5 * t)).reshape(-1, 2) for t in range(R_)]).contiguous()
    got = o.clone()
    n0 = _n(KF.t9b2_o_rope_qdq)
    KF.launch_o_rope_qdq(got, fr, tab, hd, rd, modes)
    assert _n(KF.t9b2_o_rope_qdq) - n0 == 1 and KF.t9b2_o_rope_qdq.launches[-1]["grid"][1] == R_
    mir = o.clone()
    FG.launch_o_rope_qdq(mir, fr, tab, hd, rd, modes)
    assert _eq(got, mir)
    for t in range(R_):
        one = o[t].clone()
        KF.launch_o_rope_qdq(one, fr[t].contiguous(), tab, hd, rd, modes)
        assert _eq(got[t], one), t


# ================================================================================ 3. R2 kernels (row axis)
ORD = [((("vec", 64, 4, True), "mul"), (("vec", 128, 4, True), "mul")),
       ((("unroll", 32, 2, False), "div"), (("vec", 64, 2, False), "div"))]


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("oi", range(len(ORD)))
def test_hc_norm_rows(KR, tab, split, oi):
    R_, HC, D = 6, 4, 256
    g = torch.Generator().manual_seed(5 + oi + 2 * split)
    x = _act(g, R_ * HC, D).view(R_, HC, D)
    pre = torch.rand(R_, HC, generator=g) + 1e-3
    nw = (torch.randn(D, generator=g) * 0.5 + 1.0).to(BF)
    ordd, mix = _od(*ORD[oi][0]), _od(*ORD[oi][1])

    def bufs(T_):
        return dict(y=torch.empty(T_, D, dtype=BF), yf=torch.empty(T_, D), xf=torch.empty(T_, HC * D),
                    mrs=torch.empty(T_), q=torch.empty(T_, D, dtype=FP8), s=torch.empty(T_, D // 32, dtype=E8))

    def run(mod, xx, pp, o):
        mod.launch_hc_norm(xx, pp, nw, o["y"], eps=1e-6, norm_ord=ordd, hc_mode=oi, code_tab=tab, yf=o["yf"],
                           xf=o["xf"], mrs=o["mrs"], mix_ord=mix, mix_eps=1e-6, q=o["q"], s=o["s"], split=split)

    k_ = KR.t9b4_hc_norm if split else KR.t9b2r2_hc_norm
    got, mir = bufs(R_), bufs(R_)
    n0 = _n(k_)
    run(KR, x, pre, got)
    assert _n(k_) - n0 == 1 and k_.launches[-1]["grid"][1] == R_
    run(R2, x, pre, mir)
    for key in got:
        assert _eq(got[key], mir[key]), key
    for t in range(R_):
        one = {k: v.view(-1) for k, v in bufs(1).items()}
        run(KR, x[t].contiguous(), pre[t].contiguous(), one)
        for key in one:
            assert _eq(got[key][t].reshape(-1), one[key]), (t, key)


@pytest.mark.parametrize("N,RD,ring,qdq,ns", [(128, 0, False, False, 0), (96, 32, True, True, 0),
                                              (256, 64, True, True, 3), (128, 0, False, False, 2)])
def test_norm_row_rows(KR, tab, N, RD, ring, qdq, ns):
    R_ = 6
    g = torch.Generator().manual_seed(N + RD + ns)
    if ns:
        ws = torch.randn(ns, R_, N, generator=g) * torch.exp2(torch.randint(-6, 6, (ns, R_, 1), generator=g).float())
        x = GM.SplitK(ws.contiguous(), ns)
        rows_in = [GM.SplitK(ws[:, t:t + 1].contiguous(), ns) for t in range(R_)]
    else:
        x = _act(g, R_, N, -4, 6)
        rows_in = [x[t].contiguous() for t in range(R_)]
    nw = (torch.randn(N, generator=g) * 0.3 + 1.0).to(BF)
    fr = (torch.stack([torch.view_as_real(_freqs(RD, 30 + 3 * t)).reshape(-1, 2) for t in range(R_)]).contiguous()
          if RD else None)
    slot = torch.tensor([9, 2, 14, 3, 0, 7]) if ring else None
    ordd = _od(R2.predicted_order(N, R_), "mul")

    def run(mod, xx, sl, f, T_):
        out = torch.full((16, N), 3.0, dtype=BF) if ring else torch.full((T_, N), 3.0, dtype=BF)
        q = None if qdq else torch.empty(T_, N, dtype=FP8)
        s = None if qdq else torch.empty(T_, N // 32, dtype=E8)
        mod.launch_norm_row(xx, nw, out, eps=1e-6, ordd=ordd, code_tab=tab, slot=sl, q=q, s=s, qdq=qdq,
                            rope=(f, RD, (1, 2)) if RD else None)
        return out, q, s

    n0 = _n(KR.t9b2r2_norm_row)
    got = run(KR, x, slot, fr, R_)
    assert _n(KR.t9b2r2_norm_row) - n0 == 1 and KR.t9b2r2_norm_row.launches[-1]["grid"] == (R_,)
    mir = run(R2, x, slot, fr, R_)
    for a, b in zip(got, mir):
        assert (a is None and b is None) or _eq(a, b)
    for t in range(R_):
        one = run(KR, rows_in[t], slot[t:t + 1] if ring else None, fr[t].contiguous() if RD else None, 1)
        if ring:
            assert _eq(got[0][slot[t]], one[0][slot[t]])
        else:
            assert _eq(got[0][t], one[0][0])
        if not qdq:
            assert _eq(got[1][t], one[1][0]) and _eq(got[2][t], one[2][0])
    if ring:
        rest = [r for r in range(16) if r not in slot.tolist()]
        assert (got[0][rest].float() == 3.0).all()


def test_router_rows(KR):
    R_, E, K = 6, 64, 6
    g = torch.Generator().manual_seed(6)
    sp = F.softplus(torch.randn(R_, E, generator=g) * 3)
    bias = torch.where(torch.rand(E, generator=g) < 0.5, 0.0, 0.25)
    sc = torch.randint(1, 5, (R_, E), generator=g).float() * 0.25        # ties
    idx = (sc + bias).topk(K, dim=-1)[1]
    o = _od(R2.predicted_order(K, R_))
    res = []
    for mod in (KR, R2):
        a, b = torch.empty(R_, E), torch.empty(R_, E)
        mod.launch_router_pre(sp.view(-1), bias, a.view(-1), b.view(-1))
        ids, w = torch.empty(R_, K, dtype=torch.long), torch.empty(R_, K)
        mod.launch_router_post(idx, sc, ids, w, ordd=o, eps=1e-20, rs=1.5)
        res.append((a, b, ids, w))
    assert all(_eq(p, q) for p, q in zip(*res))
    assert _eq(res[0][0], sp.sqrt()) and _eq(res[0][1], sp.sqrt() + bias)
    for t in range(R_):
        ids1, w1 = torch.empty(K, dtype=torch.long), torch.empty(K)
        KR.launch_router_post(idx[t].contiguous(), sc[t].contiguous(), ids1, w1, ordd=o, eps=1e-20, rs=1.5)
        assert _eq(res[0][2][t], ids1) and _eq(res[0][3][t], w1)


def test_engram_rows(KR):
    R_, HC, D = 6, 4, 96
    g = torch.Generator().manual_seed(7)
    x = _act(g, R_ * HC, D).view(R_, HC, D)
    kv = _act(g, R_, (HC + 1) * D, -4, 4)
    kv[0, :D] = 0
    wp = (torch.randn(HC, D, generator=g).to(BF).float() * torch.randn(HC, D, generator=g).to(BF).float())
    o, so = _od(("vec", 16, 4, True)), _od(("vec", 32, 2, True))
    c, cmin = R2.f32(D ** -0.5), R2.f32(1e-6)
    res = []
    for mod in (KR, R2):
        z = torch.empty(R_ * HC)
        mod.launch_engram_gate(x, kv.view(-1), wp.contiguous(), z, eps=1e-6, ordd=o, sordd=so, c=c, cmin=cmin)
        out = torch.empty(R_, HC, D, dtype=BF)
        mod.launch_engram_out(x, kv.view(-1), torch.sigmoid(z), out)
        res.append((z, out))
    assert _eq(res[0][0], res[1][0]) and _eq(res[0][1], res[1][1])
    for t in range(R_):
        z1 = torch.empty(HC)
        KR.launch_engram_gate(x[t].contiguous(), kv[t].contiguous(), wp.contiguous(), z1, eps=1e-6, ordd=o, sordd=so,
                              c=c, cmin=cmin)
        out1 = torch.empty(HC, D, dtype=BF)
        KR.launch_engram_out(x[t].contiguous(), kv[t].contiguous(), torch.sigmoid(z1), out1)
        assert _eq(res[0][0].view(R_, HC)[t], z1) and _eq(res[0][1][t], out1)


def test_topk_post_rows(KR):
    from runtime.static_step import indexer_topk_static

    R_, W, K = 6, 300, 128
    g = torch.Generator().manual_seed(8)
    sc = (torch.randint(0, 8, (1, R_, W), generator=g).float() * 0.5).to(BF)
    clen = torch.tensor([300, 1, 77, 128, 200, 5]).view(R_, 1)
    idx = sc.topk(K, dim=-1, sorted=False).indices
    out = torch.empty(R_, K, dtype=torch.int32)
    KR.launch_topk_post(idx.reshape(R_, K), clen.reshape(R_), out, 128)
    assert _eq(out.view(1, R_, K), indexer_topk_static(sc, clen, K, 128))
    mir = torch.empty(R_, K, dtype=torch.int32)
    R2.launch_topk_post(idx.reshape(R_, K), clen.reshape(R_), mir, 128)
    assert _eq(out, mir)


# ================================================================================ 4. GEMV RoPE epilogue per row
@pytest.mark.parametrize("cfg", [GM.GemvConfig(32, 4, 4, 2, 1, 0), GM.GemvConfig(32, 4, 4, 2, 4, 0),
                                 GM.GemvConfig(64, 2, 4, 2, 3, 1)])
def test_gemv_rope_per_row_positions(KG, cfg):
    from test_gemv_kernel_shim_cpu import _operands

    M, H, hd, rd, Kd = 6, 2, 64, 32, 512
    N = H * hd
    xq, xs, w, s = _operands(M, N, Kd, seed=9 + cfg.split_k)
    fr = torch.stack([torch.view_as_real(_freqs(rd, 3 + 4 * m)).reshape(-1, 2) for m in range(M)]).contiguous()
    plan = KG.launch_plan(cfg, M, N, Kd)
    ws = torch.zeros(plan["NSPLIT"], M, N)
    out = torch.full((M, N), float("nan"), dtype=BF)
    KG._launch(xq, xs, w, s, out, cfg, ws if plan["SPLIT"] else out, rope=(fr, hd, rd, (1, 2)))
    want = torch.empty(M, N, dtype=BF)
    GM.mxfp8_gemv(xq, xs, w, s, want, config=cfg, rope=(fr, hd, rd, (1, 2)))       # CPU: torch + per-row RoPE
    assert _eq(out, want)
    for m in range(M):   # == the one-position epilogue on that row's plain output
        plain = GM.gemv_torch(xq, xs, w, s, split_k=cfg.split_k)[m:m + 1]
        assert _eq(out[m:m + 1], FG.rope_rows_torch(plain, fr[m], hd, rd, (1, 2), inverse=False))
