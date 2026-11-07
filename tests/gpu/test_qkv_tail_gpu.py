"""GPU tests of M2-L10, the q/kv tail (docs/HANDOFF-M2-L10-QKV-TAIL.md). Skipped without CUDA (Mac); on the box a
missing triton FAILS (it is what is under test). Same docker line as tests/gpu/test_t9b4_gpu.py, with this file:

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 -m pytest -q -s tests/gpu/test_qkv_tail_gpu.py

  * real shapes (wq_a || wkv: 1280 + 512 x 5120, rope 64, the tuned GEMV configs, the orders pick_order calibrates on
    THIS device, every RoPE contraction form, 1 and 6 rows, rows per pass 1 / 2 (default) / 4 / 8): l10_qkv_tail (one
    launch) == gemv_pair + norm_row (q) + norm_row (kv), torch.equal on qr / q / s / the whole ring and the GEMV
    results; 64 eager launches in a row and 100 CUDA-graph replays all equal and the tickets at 0 after each (the
    per-output ticket under real CTA concurrency); graph-timed us per call of both (informative; the receipt is
    bench/qkv_tail_bench.py's);
  * the mini4 checkpoint, dg-inplace + gemv + fused glue, 1 and 6 rows, both P0 parities: $DSV41_QKV_TAIL=fused ==
    split, torch.equal on every row's fp32 logits, main_hidden and the caches (the byte-identical gate); graph ==
    static-eager under fused (replay-identical); the build-time self-check says equal and the step took the fused
    launch on every layer.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
DIM, QL, HD, RD, WIN = 5120, 1280, 512, 64, 256
EPS = 1e-20


def _need():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA")
    import triton  # noqa: F401  (FAIL on the box if missing)
    from runtime import fused_glue_r2 as R2
    from runtime import gemv_mxfp8 as GM

    assert R2.HAVE_TRITON and GM.HAVE_TRITON
    return R2, GM


def _bits(t):
    t = t.contiguous()
    n = t.element_size()
    return t.view(torch.uint8) if n == 1 else t.view({2: torch.int16, 4: torch.int32}[n])


def _eq(a, b):
    return tuple(a.shape) == tuple(b.shape) and torch.equal(_bits(a), _bits(b))


def _ops(M, seed):
    from runtime.gemv_mxfp8 import _synthetic

    xq, xs, wa, sa = _synthetic(QL, DIM, M, 1, "cuda")          # (seeded by shape)
    _, _, wb, sb = _synthetic(HD, DIM, M, 1, "cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)          # a different activation per draw: flip bit 0 of
    flip = torch.randint(0, 2, tuple(xq.shape), generator=g, device="cpu").to(torch.uint8).cuda()   # random bytes
    xq = (xq.view(torch.uint8) ^ flip).view(FP8)                 # (|code| <= 0x77 stays finite)
    return xq, xs, wa[0], sa[0], wb[0], sb[0]


def _inputs(M, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qnw = (torch.randn(QL, generator=g, device="cpu") * 0.3 + 1).to(BF).cuda()
    kvnw = (torch.randn(HD, generator=g, device="cpu") * 0.3 + 1).to(BF).cuda()
    pos = torch.arange(300 + seed, 300 + seed + M, device="cpu")
    inv = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, dtype=torch.float32, device="cpu") / RD))
    fr = torch.view_as_real(torch.polar(torch.ones(M, RD // 2, device="cpu"), pos[:, None].float() * inv)).float()
    fr = (fr[0] if M == 1 else fr).contiguous().cuda()
    slot = torch.remainder(pos, WIN).cuda()
    return qnw, kvnw, fr, slot


def _split(R2, GM, ops, tail, M, modes, qo, ko, tab, out=None):
    xq, xs, wa, sa, wb, sb = ops
    qnw, kvnw, fr, slot = tail
    ra, rb = GM.gemv_pair(xq, xs, wa, sa, wb, sb, tags=("l10_ta", "l10_tb"))
    if out is None:
        out = dict(qr=torch.empty(M, QL, dtype=BF, device="cuda"), q=torch.empty(M, QL, dtype=FP8, device="cuda"),
                   s=torch.empty(M, QL // 32, dtype=E8, device="cuda"),
                   ring=torch.full((1, WIN, HD), 3.0, dtype=BF, device="cuda"))
    R2.launch_norm_row(ra if hasattr(ra, "nsplit") else ra.view(M, QL), qnw, out["qr"], eps=EPS, ordd=qo,
                       code_tab=tab, q=out["q"], s=out["s"])
    R2.launch_norm_row(rb if hasattr(rb, "nsplit") else rb.view(M, HD), kvnw, out["ring"], eps=EPS, ordd=ko,
                       code_tab=tab, slot=slot, qdq=True, rope=(fr, RD, modes))
    return out, (ra, rb)


def _fused_fn(R2, GM, ops, tail, M, modes, qo, ko, tab, ticket, rp=None):
    xq, xs, wa, sa, wb, sb = ops
    qnw, kvnw, fr, slot = tail
    out = dict(qr=torch.empty(M, QL, dtype=BF, device="cuda"), q=torch.empty(M, QL, dtype=FP8, device="cuda"),
               s=torch.empty(M, QL // 32, dtype=E8, device="cuda"),
               ring=torch.full((1, WIN, HD), 3.0, dtype=BF, device="cuda"))

    def run():
        return GM.qkv_pair_tail(
            xq, xs, wa, sa, wb, sb, ticket=ticket, tab=tab, tags=("l10_fa", "l10_fb"), rp=rp,
            q_tail=dict(nw=qnw, qr=out["qr"], q=out["q"], s=out["s"], eps=EPS, fac=R2.fac_of(QL), wf=float(QL),
                        order=R2.order_args(qo)),
            kv_tail=dict(nw=kvnw, ring=out["ring"], slot=slot, freqs=fr, FS=int(fr.stride(0)) if fr.dim() == 3 else 0,
                         rd=RD, modes=modes, eps=EPS, fac=R2.fac_of(HD), wf=float(HD), order=R2.order_args(ko)))
    return out, run


def test_real_shapes_fused_equals_split_and_replays():
    R2, GM = _need()
    from runtime import fused_glue as FG
    from runtime.refmodel import import_reference

    tab = FG.scale_code_table(import_reference(), "cuda")
    qo, ko = R2.pick_order("cuda", QL, 1, "sqmean"), R2.pick_order("cuda", HD, 1, "sqmean")
    assert qo["exact"] and ko["exact"], (qo, ko)
    for M in (1, 6):
        for modes in ((0, 0), (1, 1), (2, 2), (1, 2)):
            ops = _ops(M, 10 * M + modes[0])
            tail = _inputs(M, 3 * M + modes[1])
            want, (wa_r, wb_r) = _split(R2, GM, ops, tail, M, modes, qo, ko, tab)
            ticket = torch.zeros(GM.L10_TICKETS, dtype=torch.int32, device="cuda")
            out, run = _fused_fn(R2, GM, ops, tail, M, modes, qo, ko, tab, ticket)
            res = run()
            assert res is not None, "pair_plan refused the real shapes (MMA forms differ): fused stays off"
            torch.cuda.synchronize()
            for k in want:
                assert _eq(out[k], want[k]), (M, modes, k)
            for r, w in ((res[0], wa_r), (res[1], wb_r)):
                assert _eq(r.ws, w.ws) if hasattr(r, "nsplit") else _eq(r, w)
            for rp in (1, 4, 8):                                 # rows per pass: speed only, same bits
                o2, run2 = _fused_fn(R2, GM, ops, tail, M, modes, qo, ko, tab, ticket, rp)
                assert run2() is not None
                torch.cuda.synchronize()
                assert all(_eq(o2[k], want[k]) for k in want), (M, modes, "rp", rp)
            first = {k: v.clone() for k, v in out.items()}
            for _ in range(64):                               # eager, back to back
                run()
            torch.cuda.synchronize()
            assert all(_eq(out[k], first[k]) for k in first) and int(ticket.abs().sum()) == 0
            gr = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(gr):
                run()
            for _ in range(100):
                for k in ("qr", "q", "s"):
                    out[k].view(torch.uint8).fill_(0x55)
                out["ring"][0, tail[3]] = 0.0
                gr.replay()
                torch.cuda.synchronize()
                assert all(_eq(out[k], first[k]) for k in first), "graph replay differs"
                assert int(ticket.abs().sum()) == 0, ticket
            pa = GM.pair_plan(GM.get_config(QL, DIM, M), GM.get_config(HD, DIM, M), M, QL, HD, DIM)
            print(f"  M {M} modes {modes}: fused == split (qr, q, s, ring, GEMV results); 64 eager + 100 graph "
                  f"replays equal, tickets 0; pair plan grid {pa['grid']} NSA {pa['NSA']} NSB {pa['NSB']}")
        from runtime.gemv_mxfp8 import time_graph

        ops = _ops(M, 99)
        tail = _inputs(M, 98)
        ticket = torch.zeros(GM.L10_TICKETS, dtype=torch.int32, device="cuda")
        _, run = _fused_fn(R2, GM, ops, tail, M, (1, 1), qo, ko, tab, ticket)
        run()
        so, _ = _split(R2, GM, ops, tail, M, (1, 1), qo, ko, tab)
        torch.cuda.synchronize()
        us_f = time_graph([run] * 100)
        us_s = time_graph([lambda: _split(R2, GM, ops, tail, M, (1, 1), qo, ko, tab, out=so)] * 100)
        print(f"  M {M}: graph us per layer-call: split (pair + 2 norm_row) {us_s:.2f}, fused {us_f:.2f} "
              f"(informative; receipt: bench/qkv_tail_bench.py)")


def test_mini_static_fused_equals_split_1_and_6_rows():
    R2, GM = _need()
    from test_static_rows_gpu import _eq_runs, _model, _need_step, _need_tilelang, _static, _toks

    _need_tilelang()
    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        for rows in (1, 6):
            for L in (17, 18):
                ids, toks = _toks(L, rows * 4, 90 + L)
                runs = {}
                for mode in ("split", "fused"):
                    os.environ[R2.QKV_TAIL_ENV] = mode
                    try:
                        kw = dict(rows=rows, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
                        if rows > 1:
                            kw.update(hidden_layers=hid, spare_slots=8)
                        st = _need_step(model, pool, **kw)
                        d = st.glue.describe()
                        qt = d["r2"]["qkv_tail"]
                        print(f"  rows {rows} P0 {L} {mode}: qkv_tail {qt['mode']} ({qt['why']}); check "
                              f"{d['calibration'].get('qkv_tail')}")
                        assert qt["mode"] == mode, qt
                        n0 = dict(GM.L10_STATS)
                        a = _static(model, st, ids, toks, 4)
                        if mode == "fused":
                            took = GM.L10_STATS["fused"] - n0["fused"]
                            assert took == 4 * len(model.layers), (took, GM.L10_STATS)
                        st.capture()
                        ag = _static(model, st, ids, toks, 4, graph=True)
                        _eq_runs(ag, a, f"qkv_tail {mode} rows {rows}: graph vs static-eager")
                        runs[mode] = a
                        st.detach()
                    finally:
                        os.environ.pop(R2.QKV_TAIL_ENV, None)
                _eq_runs(runs["fused"], runs["split"], f"rows {rows} P0 {L}: qkv_tail fused vs split")
                print(f"  rows {rows} P0 {L}: fused == split (logits, main_hidden, caches); graph == eager for both")
