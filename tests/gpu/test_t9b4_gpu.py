"""GPU tests of T9b-4 (docs/HANDOFF-T9B4-RESIDUAL.md). Skipped without CUDA/tilelang (Mac); on the box a missing triton
FAILS (it is what is under test). Same docker line as tests/gpu/test_fused_glue_r2_gpu.py, with this file:

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_t9b4_gpu.py

Every change `torch.equal` vs the kernel it replaces, at the real model's shapes:
  * lever 1: t9b4_hc_norm (21 programs) vs t9b2r2_hc_norm (1 program) on 256 random streams x both sides' forms +
    the head form + the DeepGEMM rows (fp32 and packed), with the orders pick_order calibrates on THIS device;
    graph-timed us per launch of both (informative: the handoff's target is <= 8 us);
  * lever 2(a): gemv_pair (wq_a || wkv: 1280 + 512 x 5120; shared w1 || w3: 2304 + 2304 x 5120) vs two mxfp8_gemv
    calls with the tuned configs (pair taken or not is printed); lever 2(b): gemv_partials -> norm_row (q / kv forms)
    and -> the shared SwiGLU vs the same kernels on mxfp8_gemv's bf16 output; graph-timed us (informative);
  * lever 4: DeepGEMM's contiguous grouped GEMM with the packed int32 activation SF vs the fp32 SF, bitwise, at the
    w1 (5120 -> 2304) and w2 (2304 -> 5120) shapes on synthetic fp4 bytes (raises -> the part stays off: reported);
  * mini checkpoints: every part's status printed; fused (T9b-4 on) == fused (T9b-4 off, $DSV41_GLUE_DISABLE) over
    prefill + 10 decode steps for dg-masked and dg-inplace, graph == static with the parts on.
"""
from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_fused_glue_gpu import _R, _act, _eq, _gen, _need_glue, _tables  # noqa: E402
from test_static_mini_gpu import _assert_equal_runs, _ids, _model, _run  # noqa: E402

FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
DIM, HC, HD, RD, Q_LORA, INTER, K_ACT, WIN, ALIGN = 5120, 4, 512, 64, 1280, 2304, 6, 128, 128
EPS = 1e-20
R4 = ("hc_split", "gemv_pair", "reduce_fold", "dg_sf_pack")


def _mods():
    from runtime import dg_inplace as DI
    from runtime import fused_glue as FG
    from runtime import fused_glue_r2 as R2
    from runtime import gemv_mxfp8 as GM

    assert R2.HAVE_TRITON and GM.HAVE_TRITON
    return FG, R2, GM, DI


def _graph_us(fn, reps=200):
    fn()
    torch.cuda.synchronize()
    from runtime.gemv_mxfp8 import time_graph

    return time_graph([fn] * reps)


# ------------------------------------------------------------------------------------------------ lever 1
def test_hc_split_real_shapes():
    _need_glue()
    FG, R2, GM, DI = _mods()
    R = _R()
    code_tab, _ = _tables(R)
    nord = R2.pick_order("cuda", DIM, 1, "sqmean")
    mord = R2.pick_order("cuda", HC * DIM, 1, "sqmean")
    print(f"  orders: norm {nord['name']} ({nord['exact']}), mix {mord['name']} ({mord['exact']})")
    g = torch.Generator().manual_seed(7)
    nw = (torch.randn(DIM, generator=g) * 0.3 + 1.0).to(BF).cuda()
    pk = {s: DI.PackedRows(DI.packed_act_sf(K_ACT * ALIGN, DIM // 32, "cuda"), ALIGN, K_ACT) for s in (0, 1)}
    bad = []
    for i in range(256):
        x = _act(g, HC, DIM)
        pre = (torch.rand(HC, generator=g) + 1e-3).cuda()
        outs = []
        for split in (False, True):
            o = dict(y=torch.empty(DIM, dtype=BF, device="cuda"), yf=torch.empty(DIM, device="cuda"),
                     xf=torch.empty(HC * DIM, device="cuda"), mrs=torch.empty(1, device="cuda"),
                     q=torch.empty(DIM, dtype=FP8, device="cuda"), s=torch.empty(DIM // 32, dtype=E8, device="cuda"),
                     dq=torch.zeros(K_ACT, ALIGN, DIM, dtype=FP8, device="cuda"),
                     ds=torch.zeros(K_ACT, ALIGN, DIM // 32, device="cuda"))
            R2.launch_hc_norm(x, pre, nw, o["y"], eps=EPS, norm_ord=nord, hc_mode=0, code_tab=code_tab, yf=o["yf"],
                              xf=o["xf"], mrs=o["mrs"], mix_ord=mord, mix_eps=EPS, q=o["q"], s=o["s"],
                              dq=o["dq"][:, 0], ds=o["ds"][:, 0], split=split)
            R2.launch_hc_norm(x, pre, nw, torch.empty(DIM, dtype=BF, device="cuda"), eps=EPS, norm_ord=nord,
                              hc_mode=0, code_tab=code_tab, dq=torch.empty(K_ACT, DIM, dtype=FP8, device="cuda"),
                              ds=pk[int(split)], split=split)
            hy, hf = torch.empty(DIM, dtype=BF, device="cuda"), torch.empty(DIM, device="cuda")
            R2.launch_hc_norm(x, pre, nw, hy, eps=EPS, norm_ord=nord, hc_mode=0, yf=hf, split=split)
            o.update(hy=hy, hf=hf, packed=pk[int(split)].codes())
            outs.append(o)
        for k in outs[0]:
            if not _eq(outs[0][k], outs[1][k]):
                bad.append((i, k))
        want = ((outs[0]["ds"][:, 0].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        if not torch.equal(outs[1]["packed"], want):
            bad.append((i, "packed vs fp32 codes"))
    assert not bad, bad[:10]
    x = _act(g, HC, DIM)
    pre = (torch.rand(HC, generator=g) + 1e-3).cuda()
    bufs = [torch.empty(DIM, dtype=BF, device="cuda"), torch.empty(HC * DIM, device="cuda"),
            torch.empty(1, device="cuda"), torch.empty(DIM, dtype=FP8, device="cuda"),
            torch.empty(DIM // 32, dtype=E8, device="cuda")]
    us = {}
    for split in (False, True):
        us[split] = _graph_us(lambda s_=split: R2.launch_hc_norm(
            x, pre, nw, bufs[0], eps=EPS, norm_ord=nord, hc_mode=0, code_tab=code_tab, xf=bufs[1], mrs=bufs[2],
            mix_ord=mord, mix_eps=EPS, q=bufs[3], s=bufs[4], split=s_))
    print(f"  hc_norm attn form: one program {us[False]:.2f} us, t9b4 split {us[True]:.2f} us per launch (graph)")


# ------------------------------------------------------------------------------------------------ lever 2
def _ops(N, K, M, seed):
    from runtime.gemv_mxfp8 import _synthetic

    xq, xs, ws, ss = _synthetic(N, K, M, 1, "cuda")
    return xq, xs, ws[0], ss[0]


def test_gemv_pair_and_fold_real_shapes():
    _need_glue()
    FG, R2, GM, DI = _mods()
    R = _R()
    code_tab, silu = _tables(R)
    for (na, nb), tag in (((Q_LORA, HD), "wq_a || wkv"), ((INTER, INTER), "w1 || w3")):
        xq, xs, wa, sa = _ops(na, DIM, 1, 0)
        _, _, wb, sb = _ops(nb, DIM, 1, 1)
        wb = wb.clone()
        wb.view(torch.uint8)[:, ::3] ^= 0x01                         # distinct halves
        ref_a = torch.empty(1, na, dtype=BF, device="cuda")
        ref_b = torch.empty(1, nb, dtype=BF, device="cuda")
        GM.mxfp8_gemv(xq, xs, wa, sa, ref_a)
        GM.mxfp8_gemv(xq, xs, wb, sb, ref_b)
        n0 = dict(GM.PAIR_STATS)
        ra, rb = GM.gemv_pair(xq, xs, wa, sa, wb, sb, tags=(f"t_{na}a", f"t_{nb}b"))
        took = GM.PAIR_STATS["pair"] - n0["pair"]
        assert _eq(GM.materialize(ra), ref_a) and _eq(GM.materialize(rb), ref_b), tag
        pa = GM.gemv_partials(xq, xs, wa, sa, tag=f"p_{na}")
        assert _eq(GM.materialize(pa), ref_a)
        ca, cb = GM.get_config(na, DIM, 1), GM.get_config(nb, DIM, 1)
        print(f"  {tag}: pair {'ONE launch' if took else 'two launches (MMA forms differ)'}; configs a {ca} b {cb}; "
              f"a split {getattr(ra, 'nsplit', 1)}, b split {getattr(rb, 'nsplit', 1)}")
        two = _graph_us(lambda: (GM.mxfp8_gemv(xq, xs, wa, sa, ref_a), GM.mxfp8_gemv(xq, xs, wb, sb, ref_b)))
        one = _graph_us(lambda: GM.gemv_pair(xq, xs, wa, sa, wb, sb, tags=(f"t_{na}a", f"t_{nb}b")))
        print(f"    graph us: two GEMVs + reduces {two:.2f}, pair without reduces {one:.2f}")
        if tag.startswith("wq_a"):                                  # the q / kv consumers on the folded partials
            nwq = (torch.randn(na, generator=torch.Generator().manual_seed(2)) * 0.3 + 1).to(BF).cuda()
            outs = []
            for x_in in (ra, ref_a.view(-1)):
                qr = torch.empty(na, dtype=BF, device="cuda")
                q, s = torch.empty(na, dtype=FP8, device="cuda"), torch.empty(na // 32, dtype=E8, device="cuda")
                R2.launch_norm_row(x_in, nwq, qr, eps=EPS, ordd=R2.pick_order("cuda", na, 1, "sqmean"),
                                   code_tab=code_tab, q=q, s=s)
                outs.append((qr, q, s))
            assert all(_eq(a, b) for a, b in zip(*outs)), "q_norm on partials vs bf16"
            fr = torch.view_as_real(torch.polar(torch.ones(RD // 2), torch.rand(RD // 2) * 3)).float().cuda()
            nwk = (torch.randn(nb) * 0.3 + 1).to(BF).cuda()
            rings = []
            for x_in in (rb, ref_b.view(-1)):
                ring = torch.zeros(WIN, nb, dtype=BF, device="cuda")
                R2.launch_norm_row(x_in, nwk, ring, eps=EPS, ordd=R2.pick_order("cuda", nb, 1, "sqmean"),
                                   code_tab=code_tab, slot=torch.tensor([5], device="cuda"), qdq=True,
                                   rope=(fr.contiguous(), RD, (1, 1)))
                rings.append(ring)
            assert _eq(rings[0], rings[1]), "kv_norm ring row on partials vs bf16"
        else:
            outs = []
            for g_, u_ in ((ra, rb), (ref_a, ref_b)):
                q = torch.empty(1, na, dtype=FP8, device="cuda")
                s = torch.empty(1, na // 32, dtype=E8, device="cuda")
                FG.launch_swiglu_quant(g_, u_, None, q, s, silu, code_tab, 10.0)
                outs.append((q, s))
            assert all(_eq(a, b) for a, b in zip(*outs)), "shared SwiGLU on partials vs bf16"


# ------------------------------------------------------------------------------------------------ lever 4
def test_dg_contiguous_packed_activation_sf():
    _need_glue()
    from test_dg_inplace_gpu import _dg, _need_dg

    _need_dg()
    FG, R2, GM, DI = _mods()
    dg = _dg()
    if hasattr(dg, "use_deterministic_algorithms"):
        dg.use_deterministic_algorithms(True)
    dg.set_mk_alignment_for_contiguous_layout(ALIGN)
    g = torch.Generator().manual_seed(5)
    G, M = 10, K_ACT * ALIGN
    for N, K in ((INTER, DIM), (DIM, INTER)):
        W = torch.randint(-128, 128, (G, N, K // 2), generator=g).to(torch.int8).cuda()
        S = DI.pack_sf(torch.randint(118, 132, (G, N, K // 32), generator=g).to(torch.uint8).cuda())
        xq = torch.zeros(M, K, dtype=FP8, device="cuda")
        xs = torch.ones(M, K // 32, device="cuda")
        heads = (_act(g, K_ACT, K).float() * 0.5).to(torch.bfloat16)
        R = _R()
        qh, sh = R.act_quant(heads, 32, R.scale_fmt, R.scale_dtype)
        xq[::ALIGN] = qh
        xs[::ALIGN] = sh.float()
        pk = DI.PackedRows(DI.packed_act_sf(M, K // 32, "cuda"), ALIGN, K_ACT)
        for r in range(K_ACT):
            pk.store_codes_torch(r, sh.view(torch.uint8)[r])
        mi = torch.full((M,), -1, dtype=torch.int32, device="cuda")
        mi[::ALIGN] = torch.tensor([0, 3, 4, 7, 8, 9], dtype=torch.int32, device="cuda")
        d32 = torch.full((M, N), float("nan"), dtype=BF, device="cuda")
        dpk = torch.full((M, N), float("nan"), dtype=BF, device="cuda")
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous((xq, xs), (W, S), d32, mi, recipe_a=(1, 32), recipe_b=(1, 32))
        try:
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous((xq, pk.sf), (W, S), dpk, mi, recipe_a=(1, 32),
                                                    recipe_b=(1, 32))
        except Exception as e:
            raise AssertionError(f"DeepGEMM refuses the packed int32 activation SF at {N}x{K}: {e!r} -- dg_sf_pack "
                                 f"stays off on this build (the self-check records the same)")
        torch.cuda.synchronize()
        heads_eq = _eq(dpk[::ALIGN], d32[::ALIGN])
        print(f"  contiguous {N}x{K}: packed activation SF == fp32 SF on the block heads: {heads_eq}; whole d "
              f"equal: {_eq(dpk, d32)}")
        assert heads_eq
        f32 = _graph_us(lambda: dg.m_grouped_fp8_fp4_gemm_nt_contiguous((xq, xs), (W, S), d32, mi, recipe_a=(1, 32),
                                                                        recipe_b=(1, 32)))
        p32 = _graph_us(lambda: dg.m_grouped_fp8_fp4_gemm_nt_contiguous((xq, pk.sf), (W, S), dpk, mi,
                                                                        recipe_a=(1, 32), recipe_b=(1, 32)))
        print(f"    graph us per call: fp32 SF (DG converts) {f32:.2f}, packed {p32:.2f}")


# ------------------------------------------------------------------------------------------------ mini checkpoints
def _mini(kind, moe):
    from runtime.fused_glue import DISABLE_ENV

    with _model(kind) as (model, tokdir):
        os.environ.pop(DISABLE_ENV, None)
        gen = _gen(model, tokdir, moe_impl=moe, fp8_impl="gemv", glue_impl="fused")
        st = gen.static_step()
        d = st.glue.describe()
        print(f"  {kind} {moe}: T9b-4 parts {d['r4']['parts']}; checks {d['calibration'].get('r4')}")
        for c in R4:
            s = d["clusters"][c]
            assert s == "on" or s.startswith("n/a") or s.startswith("off: self-check"), (c, s)
        ids = _ids(16, 9)
        on_static = _run(gen, "static", ids)
        on_graph = _run(gen, "graph", ids)
        _assert_equal_runs(on_graph, on_static, f"{kind} {moe}: T9b-4 graph vs static")
        gen.reset_static()
        os.environ[DISABLE_ENV] = ",".join(R4)
        try:
            gen2 = _gen(model, tokdir, moe_impl=moe, fp8_impl="gemv", glue_impl="fused")
            d2 = gen2.static_step().glue.describe()
            assert not set(R4) & set(d2["on"])
            off_static = _run(gen2, "static", ids)
            gen2.reset_static()
        finally:
            os.environ.pop(DISABLE_ENV, None)
        _assert_equal_runs(on_static, off_static, f"{kind} {moe}: fused + T9b-4 vs fused without (10 steps)")
        print(f"  {kind} {moe}: T9b-4 on == off over prefill + {len(on_static[0]) - 1} steps; graph == static")


def test_mini4_dg_masked():
    _need_glue()
    from runtime.dg_moe import import_deep_gemm

    import_deep_gemm()
    _mini("mini4", "dg-masked")


def test_mini4_dg_inplace():
    _need_glue()
    from runtime.dg_moe import import_deep_gemm

    import_deep_gemm()
    _mini("mini4", "dg-inplace")


TESTS = [test_hc_split_real_shapes, test_gemv_pair_and_fold_real_shapes, test_dg_contiguous_packed_activation_sf,
         test_mini4_dg_masked, test_mini4_dg_inplace]

if __name__ == "__main__":
    import traceback

    results = {}
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            results[t.__name__] = "PASS"
        except unittest.SkipTest as e:
            results[t.__name__] = f"SKIP ({e})"
        except Exception:
            traceback.print_exc()
            results[t.__name__] = "FAIL"
        print(f"{results[t.__name__]:4s} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
