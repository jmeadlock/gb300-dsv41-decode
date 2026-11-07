"""GPU tests of T9b-2 round 2 (runtime/fused_glue_r2.py). Skipped without CUDA/tilelang (Mac); on the box a missing
triton FAILS (it is what is under test). Same docker line as tests/gpu/test_fused_glue_gpu.py, with this file:

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_fused_glue_r2_gpu.py

  * the reduction-order question on THIS device: `pick_order` for every statistic of the decode path at the real
    model's shapes (RMSNorm 5120 / q_norm 1280 / kv_norm 512, hc_mixes 20480, Engram 4 x 5120 mean and dot, the
    router's 6-weight sum) over 1,024 random rows each -> the picked family member (outcome (i)) or the best one with
    its match fraction and max ulp (outcome (ii)); printed;
  * each kernel `torch.equal` vs the unfused reference ops at the real shapes, 1,024 random rows per kernel (512 for
    the 4 x 5120 ones), incl. tie-heavy routing scores and topk indices, whenever its order is outcome (i); for an
    outcome-(ii) statistic the mismatch fraction is printed and the test asserts the cluster is gated (off without
    glue_approx);
  * mini checkpoints: every R2 cluster's status and numerics outcome printed; fused == ref static-eager over prefill +
    10 decode steps (both parities), graph == static, replay x2, fills-on recapture, relaunch, no host sync (the
    round-1 helper, which now runs R1 + R2); with glue_approx: fused+approx vs ref drift printed (equal when nothing
    is approx).
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
import torch.nn.functional as F  # noqa: E402

from test_fused_glue_gpu import _R, _act, _eq, _fused_vs_ref, _gen, _need_glue, _tables  # noqa: E402
from test_static_mini_gpu import _assert_equal_runs, _ids, _model, _run  # noqa: E402

FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
DIM, HC, HD, RD, Q_LORA, E, K_ACT, WIN = 5120, 4, 512, 64, 1280, 384, 6, 128
EPS = 1e-20                     # config.json norm_eps
ROWS = 1024


def _R2():
    from runtime import fused_glue_r2 as R2

    assert R2.HAVE_TRITON
    return R2


def _show(rec):
    return (f"{rec['name']} match {rec['match'] * 100:.3f} % max {rec['max_ulp']} ulp "
            f"({rec['tried']}/{rec['candidates']} tried)")


# ------------------------------------------------------------------------------------------------ orders
STATS = [("RMSNorm[5120]", DIM, 1, "sqmean"), ("hc_mixes[20480]", HC * DIM, 1, "sqmean"),
         ("q_norm[1280]", Q_LORA, 1, "sqmean"), ("kv_norm[512]", HD, 1, "sqmean"),
         ("Engram mean[4x5120]", DIM, HC, "sqmean"), ("Engram dot[4x5120]", DIM, HC, "sum"),
         ("router sum[6]", K_ACT, 1, "possum")]


def test_reduction_orders_on_device():
    _need_glue()
    R2 = _R2()
    out = {}
    for name, w, rows, kind in STATS:
        t = time.time()
        rec = R2.pick_order("cuda", w, rows, kind, n=ROWS)
        out[name] = rec
        print(f"  {name:22s} -> {'(i) ' if rec['exact'] else '(ii)'} {_show(rec)} [{time.time() - t:.1f} s]")
        if not rec["exact"]:
            assert rec["max_ulp"] <= 4, f"{name}: outcome (ii) with {rec['max_ulp']} ulp (expected <= a few)"
    pred = {n: R2.predicted_order(w, r) for n, w, r, _ in STATS}
    print("  predicted (Reduce.cuh) == picked: " + ", ".join(
        f"{n} {'yes' if out[n]['exact'] and out[n]['order'] == pred[n] else 'NO'}" for n in pred))


# ------------------------------------------------------------------------------------------------ kernels
def _rows_eq(pairs):
    """pairs: (name, got, want) -> {name: mismatch fraction of elements}."""
    def bits(t):
        t = t.contiguous()
        return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16 if t.element_size() == 2
                                                                        else torch.int32)
    return {n: float((bits(a) != bits(b)).float().mean()) for n, a, b in pairs}


def _check(what, rec, mism):
    bad = {k: v for k, v in mism.items() if v}
    if rec["exact"]:
        assert not bad, f"{what}: order {rec['name']} is exact but the kernel differs: {bad}"
        print(f"  {what}: equal (order {rec['name']})")
    else:
        print(f"  {what}: outcome (ii) {_show(rec)}; output mismatch {bad or 'none'}")


def test_hc_norm_kernel_real_shapes():
    _need_glue()
    R, R2 = _R(), _R2()
    from runtime.dg_moe import e8m0_to_f32

    code_tab, _ = _tables(R)
    nrec = R2.pick_order("cuda", DIM, 1, "sqmean")
    mrec = R2.pick_order("cuda", HC * DIM, 1, "sqmean")
    g = torch.Generator().manual_seed(21)
    norm = R.RMSNorm(DIM, EPS).cuda()
    with torch.no_grad():
        norm.weight.copy_((torch.randn(DIM, generator=g) * 0.2 + 1.0).cuda())
    mism = {}
    for mode in R2.HC_SUM_MODES:
        ok_mode = True
        for r in range(ROWS // 4):
            x = _act(g, HC, DIM).view(1, 1, HC, DIM)
            pre = (torch.rand(1, 1, HC, generator=g) + 1e-3).cuda()
            y0_ref = R.Block.hc_pre(None, x, pre)                                    # model.py:961-962
            y_ref = norm(y0_ref)
            q_ref, s_ref = R.act_quant(y_ref, 32, R.scale_fmt, R.scale_dtype)
            xf_ref = x.flatten(2).float()
            mrs_ref = torch.rsqrt(xf_ref.square().mean(-1, keepdim=True) + EPS)    # model.py:954-955
            y, yf, xf, mrs = (torch.empty(DIM, dtype=BF, device="cuda"), torch.empty(DIM, device="cuda"),
                              torch.empty(HC * DIM, device="cuda"), torch.empty(1, device="cuda"))
            q, s = torch.empty(DIM, dtype=FP8, device="cuda"), torch.empty(DIM // 32, dtype=E8, device="cuda")
            dq = torch.zeros(K_ACT, 8, DIM, dtype=FP8, device="cuda")
            ds = torch.zeros(K_ACT, 8, DIM // 32, device="cuda")
            R2.launch_hc_norm(x.view(HC, DIM), pre.view(HC), norm.weight, y, eps=EPS, norm_ord=nrec, hc_mode=mode,
                              code_tab=code_tab, yf=yf, xf=xf, mrs=mrs, mix_ord=mrec, mix_eps=EPS, q=q, s=s,
                              dq=dq[:, 0], ds=ds[:, 0])
            s32 = torch.empty(K_ACT, DIM // 32, device="cuda")
            e8m0_to_f32(s_ref.view(1, -1), s32, "cast")
            m = _rows_eq([("y", y, y_ref.view(-1)), ("yf", yf, y_ref.float().view(-1)), ("xf", xf, xf_ref.view(-1)),
                          ("mrs", mrs, mrs_ref.view(-1)), ("q", q, q_ref.view(-1)), ("s", s, s_ref.view(-1)),
                          ("dq", dq[:, 0], q_ref.view(1, -1).expand(K_ACT, -1)), ("ds", ds[:, 0], s32)])
            if any(m[k] for k in ("y", "q", "s")) and r == 0:
                ok_mode = False
                break
            for k, v in m.items():
                mism[(mode, k)] = max(mism.get((mode, k), 0.0), v)
        print(f"  hc_pre sum mode {mode}: {'consistent with torch' if ok_mode else 'differs from torch (not picked)'}")
    picked = [m for m in R2.HC_SUM_MODES if not any(mism.get((m, k), 1) for k in ("y", "q", "s", "yf", "dq", "ds"))]
    if nrec["exact"]:
        assert picked, f"no hc_pre sum mode gives equal hc_norm outputs: {mism}"
    if mrec["exact"]:
        assert all(mism.get((m, "mrs"), 0) == 0 for m in R2.HC_SUM_MODES), "mixes statistic differs"
    print(f"  hc_norm [4 x 5120] x {ROWS // 4} rows: modes equal {picked}; norm {_show(nrec)}; mixes {_show(mrec)}")


def test_q_and_kv_path_kernels_real_shapes():
    _need_glue()
    R, R2 = _R(), _R2()
    code_tab, _ = _tables(R)
    from test_fused_glue_gpu import _freqs

    g = torch.Generator().manual_seed(22)
    qrec, krec = R2.pick_order("cuda", Q_LORA, 1, "sqmean"), R2.pick_order("cuda", HD, 1, "sqmean")
    q_norm, kv_norm = R.RMSNorm(Q_LORA, EPS).cuda(), R.RMSNorm(HD, EPS).cuda()
    with torch.no_grad():
        q_norm.weight.copy_((torch.randn(Q_LORA, generator=g) * 0.2 + 1.0).cuda())
        kv_norm.weight.copy_((torch.randn(HD, generator=g) * 0.2 + 1.0).cuda())
    picked = None
    for modes in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
        ok = True
        for r in range(32):
            kvr = _act(g, 1, HD, -4, 6).view(1, 1, HD)
            pos = int(torch.randint(1, 4000, (1,), generator=g))
            f = _freqs(pos).view(1, -1)
            kv = kv_norm(kvr)
            R.apply_rotary_emb(kv[..., -RD:], f)
            R.act_quant(kv, 32, R.scale_fmt, R.scale_dtype, True)
            ring = torch.zeros(1, WIN, HD, dtype=BF, device="cuda")
            want = ring.clone()
            want[0, pos % WIN] = kv.view(-1)
            R2.launch_norm_row(kvr.view(-1), kv_norm.weight, ring, eps=EPS, ordd=krec, code_tab=code_tab,
                               slot=torch.tensor([pos % WIN], device="cuda"), qdq=True,
                               rope=(torch.view_as_real(f).reshape(-1, 2), RD, modes))
            if not _eq(ring, want):
                ok = False
                break
        if ok:
            picked = modes
            break
    if krec["exact"]:
        assert picked is not None, "kv path: no RoPE form gives an equal ring row"
    mism = {}
    for r in range(ROWS):
        qa = _act(g, 1, Q_LORA, -4, 6).view(1, 1, Q_LORA)
        qr_ref = q_norm(qa)
        q_ref, s_ref = R.act_quant(qr_ref, 32, R.scale_fmt, R.scale_dtype)
        qr = torch.empty(Q_LORA, dtype=BF, device="cuda")
        q, s = torch.empty(Q_LORA, dtype=FP8, device="cuda"), torch.empty(Q_LORA // 32, dtype=E8, device="cuda")
        R2.launch_norm_row(qa.view(-1), q_norm.weight, qr, eps=EPS, ordd=qrec, code_tab=code_tab, q=q, s=s)
        for k, v in _rows_eq([("qr", qr, qr_ref.view(-1)), ("q", q, q_ref.view(-1)), ("s", s, s_ref.view(-1))]).items():
            mism[k] = max(mism.get(k, 0.0), v)
    _check(f"q path [1280] x {ROWS} rows", qrec, mism)
    print(f"  kv path [512, rope 64, ring {WIN}]: RoPE forms {picked}; norm {_show(krec)}")


def test_router_kernels_real_shapes_with_ties():
    _need_glue()
    R2 = _R2()
    rec = R2.pick_order("cuda", K_ACT, 1, "possum")
    g = torch.Generator().manual_seed(23)
    w_gate = (torch.randn(E, DIM, generator=g) * 0.02).cuda()
    bias = (torch.randn(E, generator=g) * 0.05).cuda()
    mism = {"ids": 0.0, "w": 0.0}
    for r in range(ROWS):
        tie = r % 2 == 1
        x = (torch.randn(1, DIM, generator=g) * 0.5).to(BF).cuda()
        if tie:   # tie-heavy: quantized scores and a two-valued bias -> exact ties in scores + bias
            sp = (torch.randint(1, 6, (E,), generator=g).float() * 0.25).cuda()
            b = torch.where(torch.rand(E, generator=g) < 0.5, 0.0, 0.25).cuda()
        else:
            sp = F.softplus(F.linear(x.float(), w_gate)).view(-1)
            b = bias
        scores = sp.sqrt().view(1, -1)                                   # model.py:819 (sqrtsoftplus)
        idx_r = (scores + b).topk(K_ACT, dim=-1)[1]                      # model.py:824
        wt = scores.gather(1, idx_r)
        wt /= wt.sum(dim=-1, keepdim=True) + 1e-20                       # model.py:826-827
        wt *= 1.5
        ids_r, order = torch.sort(idx_r[0])                              # static_step's ascending order
        sc, bsc = torch.empty(E, device="cuda"), torch.empty(1, E, device="cuda")
        R2.launch_router_pre(sp, b, sc, bsc.view(-1))
        idx = bsc.topk(K_ACT, dim=-1)[1]
        ids, w = torch.empty(K_ACT, dtype=torch.long, device="cuda"), torch.empty(K_ACT, device="cuda")
        R2.launch_router_post(idx.view(-1), sc, ids, w, ordd=rec, eps=1e-20, rs=1.5)
        assert _eq(sc, scores.view(-1)) and _eq(bsc, scores + b), "router pre (sqrt, +bias)"
        assert torch.equal(idx, idx_r), "same topk input -> same topk"
        m = _rows_eq([("ids", ids, ids_r), ("w", w, wt[0, order])])
        for k, v in m.items():
            mism[k] = max(mism[k], v)
    _check(f"router post [384 -> 6] x {ROWS} rows (half tie-heavy)", rec, mism)
    assert mism["ids"] == 0, "ascending ids differ (the sort of distinct ids is exact)"


def test_engram_kernels_real_shapes():
    _need_glue()
    R, R2 = _R(), _R2()
    rec, srec = R2.pick_order("cuda", DIM, HC, "sqmean"), R2.pick_order("cuda", DIM, HC, "sum")
    g = torch.Generator().manual_seed(24)
    qw = (torch.randn(HC, DIM, generator=g) * 0.3 + 1).to(BF).cuda()
    kw = (torch.randn(HC, DIM, generator=g) * 0.3 + 1).to(BF).cuda()
    wp = (qw.float() * kw.float()).contiguous()
    from types import SimpleNamespace

    mism = {"out": 0.0, "z": 0.0}
    for r in range(ROWS // 2):
        x = _act(g, HC, DIM).view(1, 1, HC, DIM)
        kv = _act(g, 1, (HC + 1) * DIM, -4, 4).view(1, 1, -1)
        if r == 0:
            kv[..., :DIM] = 0                                            # a dot of exactly 0 (clamp, copysign)
        stub = SimpleNamespace(wkv=lambda _x, kv=kv: kv, embed=lambda _h: torch.zeros(1, 1, 1, 1, device="cuda"),
                               hc_mult=HC, dim=DIM, q_weight=qw, k_weight=kw, eps=EPS, clamp_value=1e-6)
        ref = R.Engram.forward(stub, x, None, None)
        z = torch.empty(HC, device="cuda")
        R2.launch_engram_gate(x.view(HC, DIM), kv.view(-1), wp, z, eps=EPS, ordd=rec, sordd=srec,
                              c=R2.f32(DIM ** -0.5), cmin=R2.f32(1e-6))
        out = torch.empty(HC, DIM, dtype=BF, device="cuda")
        R2.launch_engram_out(x.view(HC, DIM), kv.view(-1), torch.sigmoid(z), out)
        mism["out"] = max(mism["out"], _rows_eq([("o", out.view(1, 1, HC, DIM), ref)])["o"])
    _check(f"engram [4 x 5120] x {ROWS // 2} rows", {"exact": rec["exact"] and srec["exact"],
                                                     "name": f"{rec['name']} / {srec['name']}",
                                                     "match": min(rec["match"], srec["match"]),
                                                     "max_ulp": max(rec["max_ulp"], srec["max_ulp"]),
                                                     "tried": rec["tried"], "candidates": rec["candidates"]},
           {"out": mism["out"]})


def test_topk_post_kernel_with_ties():
    _need_glue()
    R2 = _R2()
    from runtime.static_step import indexer_topk_static

    g = torch.Generator().manual_seed(25)
    for W, K in ((2048, 512), (1024, 512), (512, 512), (640, 128)):
        for r in range(ROWS // 8):
            sc = (torch.randint(0, 16, (1, 1, W), generator=g).float() * 0.5).to(BF).cuda()
            sc[..., int(torch.randint(1, W, (1,), generator=g)):] = -float("inf")
            cl = torch.randint(1, W + 1, (1,), generator=g).cuda()
            idx = sc.topk(K, dim=-1, sorted=False).indices
            out = torch.empty(1, 1, K, dtype=torch.int32, device="cuda")
            R2.launch_topk_post(idx.reshape(-1), cl, out.view(-1), 128)
            assert torch.equal(out, indexer_topk_static(sc, cl, K, 128)), (W, K, r)
    print(f"  indexer post-topk (sort + where + offset) == indexer_topk_static on tie-heavy scores, 4 shapes")


# ------------------------------------------------------------------------------------------------ mini checkpoints
def _r2_status(kind, moe, fp8, approx=False):
    from runtime.fused_glue import R2_CLUSTER_NAMES

    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=moe, fp8_impl=fp8, glue_impl="fused", glue_approx=approx)
        d = gen.static_step().glue.describe()
        print(f"  {kind} {moe}/{fp8}{' +approx' if approx else ''}: " + "; ".join(
            f"{c} {d['clusters'][c]}" for c in R2_CLUSTER_NAMES))
        print(f"    orders {d['r2']['orders']}; outcome {d['r2']['outcome']}")
        print("    " + gen.static.glue.glue_line())
        for c in R2_CLUSTER_NAMES:
            st = d["clusters"][c]
            assert st in ("on", "approx") or st.startswith("off: (ii)") or st.startswith("n/a"), (c, st)
            if st == "approx":
                assert approx and d["r2"]["numerics"][c]["outcome"].startswith("(ii)")
        if approx:
            ids = _ids(16, 5)
            fused = _run(gen, "static", ids)
            gen.set_glue_impl("ref")
            ref = _run(gen, "static", ids)
            if not d["approx_on"]:
                _assert_equal_runs(fused, ref, f"{kind}: +approx with nothing approx == ref")
            else:
                print(f"    +approx with {d['approx_on']}: drift vs ref reported by first light (not asserted)")
        gen.reset_static()
        del gen


def test_r2_status_and_fused_vs_ref_mini4_dg_gemv():
    _need_glue()
    from runtime.dg_moe import import_deep_gemm

    import_deep_gemm()
    _r2_status("mini4", "dg-masked", "gemv")
    _fused_vs_ref("mini4", "dg-masked", "gemv")          # R1 + R2: fused == ref, graph == static, no host sync


def test_r2_status_and_fused_vs_ref_mini_refref():
    _need_glue()
    _r2_status("mini", "ref", "ref")
    _fused_vs_ref("mini", "ref", "ref")


def test_r2_glue_approx_mini4():
    _need_glue()
    from runtime.dg_moe import import_deep_gemm

    import_deep_gemm()
    _r2_status("mini4", "dg-masked", "gemv", approx=True)


TESTS = [test_reduction_orders_on_device, test_hc_norm_kernel_real_shapes, test_q_and_kv_path_kernels_real_shapes,
         test_router_kernels_real_shapes_with_ties, test_engram_kernels_real_shapes, test_topk_post_kernel_with_ties,
         test_r2_status_and_fused_vs_ref_mini4_dg_gemv, test_r2_status_and_fused_vs_ref_mini_refref,
         test_r2_glue_approx_mini4]

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
