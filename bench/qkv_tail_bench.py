#!/usr/bin/env python3
"""M2-L10 bench (docs/HANDOFF-M2-L10-QKV-TAIL.md): the q/kv tail, split (today: wq_a || wkv pair GEMV + q norm_row +
kv norm_row, 3 launches per layer) vs fused (runtime/gemv_mxfp8.l10_qkv_tail, 1 launch). Receipts go to scratch/
(never docs/).

  # launches/step, S2 + S3 (CPU census on the real layer structure, bench/glue_census.py's machinery), 1 and 6 rows:
  .venv/bin/python bench/qkv_tail_bench.py --census --json scratch/qkv-tail-census.json
  # us per layer-call on the box (real shapes: wq_a 1280 x 5120, wkv 512 x 5120, rope 64; tuned GEMV configs; the
  # orders pick_order calibrates on the device), 1 and 6 rows, + bit-equality fused == split on every output:
  python3 bench/qkv_tail_bench.py --timing --json scratch/qkv-tail-<tag>.json
  # --qkv-tail fused|split sets $DSV41_QKV_TAIL for anything this process builds (the census builds both anyway);
  # the step itself (bench/first_light.py, bench/headline.py) reads the same variable: DSV41_QKV_TAIL=fused.

Timing method (--timing): per variant, `--layers` (40) layer-calls captured in one CUDA graph, each layer with its
own weight copies (so wq_a / wkv stream from HBM as in the step) and its own outputs; replayed `--reps` times; us/layer
= graph us / layers; ms/step saved = 40 x (split - fused) / 1000 (the verify step's 40 layers; the draft does not run
these clusters). Bit-equality: every layer's qr / q / s / ring of the fused graph == the split graph's, after the
replays; the tickets read 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "tests"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

DIM, QL, HD, RD, WIN, EPS = 5120, 1280, 512, 64, 256, 1e-20
S23_REGIONS = ("attn.q_path", "attn.kv_path")      # static_step._q_path / _kv_path (glue_census tags)
MODELED = {"ms_per_step_6rows": 0.21, "launches_per_step": 90,
           "source": "docs/M2-FUSED-CHAIN-DESIGN.md section 2.2 L10 economics row [E]; norm_row 3.4 us [M]"}


# ------------------------------------------------------------------------------------------------ census (CPU)
def _region_launches(rep: dict, region: str) -> float:
    c = rep["regions"].get(region, {}).get("classes", {})
    return c.get("launch", 0) + c.get("kernel", 0)


def _kernels_in(rep: dict, regions) -> dict:
    out = {}
    for r in regions:
        for op, n in rep["regions"].get(r, {}).get("ops", {}).items():
            out[op] = round(out.get(op, 0) + n, 2)
    return out


def census(rows_list, steps: int) -> dict:
    import glue_census as GC
    from runtime import fused_glue_r2

    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        R, model, pool, cfg = GC.build(tmp)
        restore = GC.install(R)
        saved = fused_glue_r2.launch_qkv_tail
        fused_glue_r2.launch_qkv_tail = GC.kernel(saved, "l10_qkv_tail", lambda *a, **k: [("alloc", "empty")] * 3)
        prev_env = os.environ.get(fused_glue_r2.QKV_TAIL_ENV)
        prev_dt = torch.get_default_dtype()
        try:
            from test_dg_moe_cpu import _fake_deep_gemm

            for rows in rows_list:
                fake = _fake_deep_gemm()
                fn = fake.m_grouped_fp8_fp4_gemm_nt_masked
                if rows > 1:   # bench/glue_census.main's R-row setup
                    from test_dg_inplace_cpu import _fake_dg_contig
                    from test_static_rows_cpu import _masked_rowwise

                    fake = _fake_dg_contig()
                    fn = _masked_rowwise(fake)
                fake.m_grouped_fp8_fp4_gemm_nt_masked = GC.kernel(fn, "dg_masked_gemm", lambda *x, **y: [])
                sys.modules["deep_gemm"] = fake
                ids = list(range(40, 56))
                per = {}
                for mode in ("split", "fused"):
                    os.environ[fused_glue_r2.QKV_TAIL_ENV] = mode
                    rep = GC.run_path("static-dg-gemv-fused", R, model, pool, ids, steps, rows=rows)
                    per[mode] = {"launches_per_step": rep["launches_per_step"],
                                 "s2_s3_launches_per_step": round(sum(_region_launches(rep, r) for r in S23_REGIONS),
                                                                  1),
                                 "s2_s3_kernels_per_step": _kernels_in(rep, S23_REGIONS)}
                    print(f"rows {rows} {mode:5s}: launches/step {rep['launches_per_step']:8.1f}; S2+S3 "
                          f"{per[mode]['s2_s3_launches_per_step']:6.1f} {per[mode]['s2_s3_kernels_per_step']}",
                          flush=True)
                per["removed_per_step"] = round(per["split"]["launches_per_step"] - per["fused"]["launches_per_step"],
                                                1)
                per["removed_s2_s3"] = round(per["split"]["s2_s3_launches_per_step"]
                                             - per["fused"]["s2_s3_launches_per_step"], 1)
                res[f"rows{rows}"] = per
        finally:
            fused_glue_r2.launch_qkv_tail = saved
            restore()
            torch.set_default_dtype(prev_dt)
            if prev_env is None:
                os.environ.pop(fused_glue_r2.QKV_TAIL_ENV, None)
            else:
                os.environ[fused_glue_r2.QKV_TAIL_ENV] = prev_env
    res["config"] = {k: cfg[k] for k in ("n_layers", "kv_source_layers", "index_source_layers")}
    res["note"] = ("CPU census of the real layer structure (tiny dims): each fused / Triton kernel counts as one "
                   "launch (the GEMV's split-K reduce is not counted on either side: reduce_fold folds it into "
                   "norm_row today)")
    return res


# ------------------------------------------------------------------------------------------------ timing (box)
def _graph_us(calls, reps: int) -> float:
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for c in calls:
            c()
    g.replay()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / reps


def timing(rows_list, layers: int, reps: int, modes, rps=(None,)) -> dict:
    from runtime import fused_glue as FG
    from runtime import fused_glue_r2 as R2
    from runtime import gemv_mxfp8 as GM
    from runtime.refmodel import import_reference

    assert torch.cuda.is_available() and GM.HAVE_TRITON, "--timing runs on the box (CUDA + Triton)"
    dev = "cuda"
    tab = FG.scale_code_table(import_reference(), dev)
    qo, ko = R2.pick_order(dev, QL, 1, "sqmean"), R2.pick_order(dev, HD, 1, "sqmean")
    out = {"orders": {"q": qo["name"], "kv": ko["name"], "exact": bool(qo["exact"] and ko["exact"])},
           "rope_modes": list(modes), "layers": layers, "reps": reps}
    for M in rows_list:
        xq, xs, wa_l, sa_l = GM._synthetic(QL, DIM, M, layers, dev)
        _, _, wb_l, sb_l = GM._synthetic(HD, DIM, M, layers, dev)
        g = torch.Generator(device="cpu").manual_seed(M)
        qnw = [(torch.randn(QL, generator=g, device="cpu") * 0.3 + 1).to(torch.bfloat16).to(dev)
               for _ in range(layers)]
        kvnw = [(torch.randn(HD, generator=g, device="cpu") * 0.3 + 1).to(torch.bfloat16).to(dev)
                for _ in range(layers)]
        pos = torch.arange(300, 300 + M, device="cpu")
        inv = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, dtype=torch.float32, device="cpu") / RD))
        fr = torch.view_as_real(torch.polar(torch.ones(M, RD // 2, device="cpu"), pos[:, None].float() * inv)).float()
        fr = (fr[0] if M == 1 else fr).contiguous().to(dev)
        slot = torch.remainder(pos, WIN).to(dev)

        def bufs():
            return [dict(qr=torch.empty(M, QL, dtype=torch.bfloat16, device=dev),
                         q=torch.empty(M, QL, dtype=torch.float8_e4m3fn, device=dev),
                         s=torch.empty(M, QL // 32, dtype=torch.float8_e8m0fnu, device=dev),
                         ring=torch.zeros(1, WIN, HD, dtype=torch.bfloat16, device=dev)) for _ in range(layers)]
        so, fo = bufs(), bufs()
        ticket = torch.zeros(GM.L10_TICKETS, dtype=torch.int32, device=dev)

        def split_call(i):
            def f():
                ra, rb = GM.gemv_pair(xq, xs, wa_l[i], sa_l[i], wb_l[i], sb_l[i], tags=(f"qkvb_sa{i}", f"qkvb_sb{i}"))
                R2.launch_norm_row(ra if hasattr(ra, "nsplit") else ra.view(M, QL), qnw[i], so[i]["qr"], eps=EPS,
                                   ordd=qo, code_tab=tab, q=so[i]["q"], s=so[i]["s"])
                R2.launch_norm_row(rb if hasattr(rb, "nsplit") else rb.view(M, HD), kvnw[i], so[i]["ring"], eps=EPS,
                                   ordd=ko, code_tab=tab, slot=slot, qdq=True, rope=(fr, RD, tuple(modes)))
            return f

        def fused_call(i, rp):
            def f():
                r = GM.qkv_pair_tail(
                    xq, xs, wa_l[i], sa_l[i], wb_l[i], sb_l[i], ticket=ticket, tab=tab, rp=rp,
                    tags=(f"qkvb_fa{i}", f"qkvb_fb{i}"),
                    q_tail=dict(nw=qnw[i], qr=fo[i]["qr"], q=fo[i]["q"], s=fo[i]["s"], eps=EPS, fac=R2.fac_of(QL),
                                wf=float(QL), order=R2.order_args(qo)),
                    kv_tail=dict(nw=kvnw[i], ring=fo[i]["ring"], slot=slot, freqs=fr,
                                 FS=int(fr.stride(0)) if fr.dim() == 3 else 0, rd=RD, modes=tuple(modes), eps=EPS,
                                 fac=R2.fac_of(HD), wf=float(HD), order=R2.order_args(ko)))
                assert r is not None, "pair_plan refused the real shapes (the halves' MMA forms differ)"
            return f
        us_s = _graph_us([split_call(i) for i in range(layers)], reps) / layers
        pp = GM.pair_plan(GM.get_config(QL, DIM, M), GM.get_config(HD, DIM, M), M, QL, HD, DIM)
        rec = {"us_per_layer": {"split": round(us_s, 3)}, "launches_per_layer": {"split": 3, "fused": 1},
               "pair_plan": {k: pp[k] for k in ("grid", "NSA", "NSB", "BLOCK_N", "KU", "num_warps")} if pp else None,
               "rp": {}}

        def bits(t):
            return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)
        for rp in rps:
            for b in fo:
                for v in b.values():
                    bits(v).zero_()                           # (byte views: e8m0 has no zero value)
            us_f = _graph_us([fused_call(i, rp) for i in range(layers)], reps) / layers
            torch.cuda.synchronize()
            bad = [(i, k) for i in range(layers) for k in so[i] if not torch.equal(bits(so[i][k]), bits(fo[i][k]))]
            rp_eff = GM.qkv_tail_plan(pp, M, QL, HD, rp)["RP"] if pp else None
            rec["rp"][str(rp_eff)] = {"us_per_layer": round(us_f, 3), "us_saved_per_layer": round(us_s - us_f, 3),
                                     "ms_saved_per_step_40_layers": round((us_s - us_f) * 40 / 1000.0, 4),
                                     "bit_identical": not bad, "mismatches": bad[:8],
                                     "tickets_zero": int(ticket.abs().sum()) == 0}
            print(f"rows {M} RP {rp_eff}: us/layer split {us_s:.2f} fused {us_f:.2f} (saved {us_s - us_f:.2f}; "
                  f"{(us_s - us_f) * 40 / 1000.0:.3f} ms/step at 40 layers); bit-identical {not bad}; tickets 0 "
                  f"{int(ticket.abs().sum()) == 0}", flush=True)
        best = min(rec["rp"].items(), key=lambda kv: kv[1]["us_per_layer"])
        rec["best_rp"] = {"RP": best[0], **best[1]}
        out[f"rows{M}"] = rec
        print(f"rows {M}: us/layer split {us_s:.2f} fused {us_f:.2f} (saved {us_s - us_f:.2f}; "
              f"{rec['ms_saved_per_step_40_layers']:.3f} ms/step at 40 layers); bit-identical {not bad}; "
              f"tickets 0 {rec['tickets_zero']}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--census", action="store_true", help="CPU census: launches/step S2+S3, split vs fused")
    ap.add_argument("--timing", action="store_true", help="box: graph us per layer-call, split vs fused")
    ap.add_argument("--rows", default="1,6", help="comma list of row counts")
    ap.add_argument("--steps", type=int, default=2, help="census decode steps (2 = both parities)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--rope-modes", default="1,1", help="RoPE contraction forms for --timing (speed only)")
    ap.add_argument("--rp", default="1,2,4,8", help="--timing: the tails' rows per pass to sweep (speed only; "
                    "runtime default gemv_mxfp8.L10_TAIL_RP, $DSV41_QKV_TAIL_RP)")
    ap.add_argument("--qkv-tail", choices=["split", "fused"], default=None,
                    help="set $DSV41_QKV_TAIL for anything this process builds")
    ap.add_argument("--json", default=None, help="receipt path (default scratch/qkv-tail-<date>.json)")
    a = ap.parse_args(argv)
    if a.qkv_tail:
        os.environ["DSV41_QKV_TAIL"] = a.qkv_tail
    if not (a.census or a.timing):
        a.census = not torch.cuda.is_available()
        a.timing = torch.cuda.is_available()
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "modeled": MODELED, "host": os.uname().nodename}
    if a.census:
        rep["census"] = census(rows, a.steps)
    if a.timing:
        rep["timing"] = timing(rows, a.layers, a.reps, tuple(int(m) for m in a.rope_modes.split(",")),
                               tuple(int(r) for r in a.rp.split(",") if r.strip()))
    path = a.json or os.path.join(REPO, "scratch", f"qkv-tail-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"receipt {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
