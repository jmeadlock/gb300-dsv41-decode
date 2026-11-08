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
M2-L10-FIX (docs/HANDOFF-M2-L10-FIX.md) phases, each its own graph, all measured: `pair` (the GEMV alone), `q_norm_row`
/ `kv_norm_row` (each consumer alone, on the pair's partials), `split` (all three); per fused form (fold = round 1,
stage = the fix) and rows-per-pass / per-tail-CTA RP: the full launch, and `<form>_notail` (TAILS=False: the GEMV +
tickets (+ the stage fold) without the tail work), so tail cost = best - notail and GEMV/ticket overhead = notail -
pair. `compiled`: registers / spills / warps / shared of every kernel (cause (b)); `config` / `pair_plan`: the tuned
GEMV configs both paths resolve (cause (c)). Final line `QKV-TAIL-TIMING: R=1 ... ; R=6 ...`.
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


class _MetaRecorder:
    """Cause (b): the compiled kernels' registers / spills / warps / shared memory. Wraps each Triton JITFunction's
    `run` (what `kernel[grid](...)` calls) and keeps the distinct kernel_meta of what it returned. Best effort: any
    Triton without these attributes records nothing (never fails the bench)."""

    def __init__(self, GM, fns):
        self.GM, self.fns, self.saved, self.seen = GM, fns, {}, {}

    def __enter__(self):
        for name, fn in self.fns.items():
            run = getattr(fn, "run", None)
            if run is None:
                continue
            self.saved[name] = run

            def wrapped(*a, _run=run, _name=name, **k):
                kern = _run(*a, **k)
                try:
                    if not torch.cuda.is_current_stream_capturing():
                        m = self.GM.kernel_meta(kern)
                        if m is not None and m not in self.seen.setdefault(_name, []):
                            self.seen[_name].append(m)
                except Exception:  # noqa: BLE001 - informative only
                    pass
                return kern
            try:
                fn.run = wrapped
            except Exception:  # noqa: BLE001
                self.saved.pop(name, None)
        return self

    def __exit__(self, *exc):
        for name, run in self.saved.items():
            try:
                del self.fns[name].run          # back to the class attribute
            except Exception:  # noqa: BLE001
                self.fns[name].run = run
        return False


def timing(rows_list, layers: int, reps: int, modes, rps=(None,), forms=("fold", "stage"), _dry=None) -> dict:
    """us per layer-call [measured, CUDA events over graph replays of `layers` layer-calls] of every phase, 1 and R
    rows (section docstring), + bit-equality of every fused variant vs split on qr / q / s / ring.
    _dry (tests/test_qkv_tail_cpu.py only): {"GM", "R2": the shim-loaded modules, "dims": (DIM, QL, HD), "tab": the
    act_quant code table} -- the same
    code on CPU through tests/triton_shim.py, each graph replaced by one eager pass and a placeholder time (NOT a
    measurement), so the bookkeeping that crashed round 1's box run is exercised on the Mac."""
    from runtime import fused_glue as FG
    from runtime.refmodel import import_reference

    if _dry is None:
        from runtime import fused_glue_r2 as R2
        from runtime import gemv_mxfp8 as GM

        assert torch.cuda.is_available() and GM.HAVE_TRITON, "--timing runs on the box (CUDA + Triton)"
        dev, graph_us, sync = "cuda", _graph_us, torch.cuda.synchronize
        DIM, QL, HD = globals()["DIM"], globals()["QL"], globals()["HD"]
    else:
        GM, R2, (DIM, QL, HD) = _dry["GM"], _dry["R2"], _dry["dims"]
        dev, sync = "cpu", (lambda: None)

        def graph_us(calls, reps_):
            for c in calls:
                c()
            return float(len(calls))          # placeholder, not a time
    tab = _dry["tab"] if _dry is not None else FG.scale_code_table(import_reference(), dev)
    qo, ko = R2.pick_order(dev, QL, 1, "sqmean"), R2.pick_order(dev, HD, 1, "sqmean")
    out = {"orders": {"q": qo["name"], "kv": ko["name"], "exact": bool(qo["exact"] and ko["exact"])},
           "rope_modes": list(modes), "layers": layers, "reps": reps, "label": "measured (CUDA events, graph replay)",
           "gemv_pin_m1": GM.pin_m1(), "forms": list(forms), "dry_run": _dry is not None}
    fns = {"t9b4_mxfp8_gemv2": GM.t9b4_mxfp8_gemv2, "t9b2r2_norm_row": R2.t9b2r2_norm_row,
           "l10_qkv_tail (fold)": GM.l10_qkv_tail, "l10_qkv_tail2 (stage)": GM.l10_qkv_tail2}
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
        ticket = torch.zeros(GM.L10_TICKET_WORDS, dtype=torch.int32, device=dev)
        pr = {}

        def pair(i):
            pr[i] = GM.gemv_pair(xq, xs, wa_l[i], sa_l[i], wb_l[i], sb_l[i], tags=(f"qkvb_sa{i}", f"qkvb_sb{i}"))
            return pr[i]

        def qnorm(i):
            ra = pr[i][0]
            R2.launch_norm_row(ra if hasattr(ra, "nsplit") else ra.view(M, QL), qnw[i], so[i]["qr"], eps=EPS,
                               ordd=qo, code_tab=tab, q=so[i]["q"], s=so[i]["s"])

        def kvnorm(i):
            rb = pr[i][1]
            R2.launch_norm_row(rb if hasattr(rb, "nsplit") else rb.view(M, HD), kvnw[i], so[i]["ring"], eps=EPS,
                               ordd=ko, code_tab=tab, slot=slot, qdq=True, rope=(fr, RD, tuple(modes)))

        def split_call(i):
            def f():
                pair(i)
                qnorm(i)
                kvnorm(i)
            return f

        def fused_call(i, form, rp, tails):
            def f():
                r = GM.qkv_pair_tail(
                    xq, xs, wa_l[i], sa_l[i], wb_l[i], sb_l[i], ticket=ticket, tab=tab, rp=rp, form=form,
                    tails=tails, tags=(f"qkvb_fa{i}", f"qkvb_fb{i}"),
                    q_tail=dict(nw=qnw[i], qr=fo[i]["qr"], q=fo[i]["q"], s=fo[i]["s"], eps=EPS, fac=R2.fac_of(QL),
                                wf=float(QL), order=R2.order_args(qo)),
                    kv_tail=dict(nw=kvnw[i], ring=fo[i]["ring"], slot=slot, freqs=fr,
                                 FS=int(fr.stride(0)) if fr.dim() == 3 else 0, rd=RD, modes=tuple(modes), eps=EPS,
                                 fac=R2.fac_of(HD), wf=float(HD), order=R2.order_args(ko)))
                assert r is not None, "pair_plan refused the real shapes (the halves' MMA forms differ)"
            return f

        def bits(t):
            return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)

        cfg_a, cfg_b = GM._resolved(QL, DIM, M, dev, None), GM._resolved(HD, DIM, M, dev, None)   # gemv_pair's
        pp = GM.pair_plan(cfg_a, cfg_b, M, QL, HD, DIM)
        rec = {"config": {"wq_a": cfg_a.as_dict(), "wkv": cfg_b.as_dict()},
               "pair_plan": {k: pp[k] for k in ("grid", "NSA", "NSB", "TA", "TB", "BLOCK_M", "BLOCK_N", "KU",
                                                "num_warps", "num_stages")} if pp else None,
               "us_per_layer": {}, "variants": {}, "launches_per_layer": {"split": 3, "fused": 1}}
        us = rec["us_per_layer"]
        with _MetaRecorder(GM, fns) as mr:
            us["pair"] = graph_us([(lambda i=i: pair(i)) for i in range(layers)], reps) / layers
            us["q_norm_row"] = graph_us([(lambda i=i: qnorm(i)) for i in range(layers)], reps) / layers
            us["kv_norm_row"] = graph_us([(lambda i=i: kvnorm(i)) for i in range(layers)], reps) / layers
            us["split"] = graph_us([split_call(i) for i in range(layers)], reps) / layers
            print(f"rows {M} split : {us['split']:7.2f} us/layer-call = pair {us['pair']:.2f} + q norm_row "
                  f"{us['q_norm_row']:.2f} + kv norm_row {us['kv_norm_row']:.2f} (each phase alone; measured)",
                  flush=True)
            for form in forms:
                eff = []
                for rp in rps:
                    tp = GM.qkv_tail_plan(pp, M, QL, HD, rp, form)
                    if tp["RP"] not in eff:
                        eff.append(tp["RP"])
                for rp_eff in eff:
                    for b in fo:
                        for v in b.values():
                            bits(v).zero_()                   # (byte views: e8m0 has no zero value)
                    name = f"{form}_rp{rp_eff}"
                    u = graph_us([fused_call(i, form, rp_eff, True) for i in range(layers)], reps) / layers
                    sync()
                    bad = [(i, k) for i in range(layers) for k in so[i] if not torch.equal(bits(so[i][k]),
                                                                                           bits(fo[i][k]))]
                    t0 = int(ticket.abs().sum()) == 0
                    us[name] = u
                    rec["variants"][name] = {"form": form, "RP": rp_eff, "us_per_layer": round(u, 3),
                                             "us_saved_vs_split": round(us["split"] - u, 3),
                                             "ms_saved_per_step_40_layers": round((us["split"] - u) * 40 / 1000.0, 4),
                                             "bit_identical": not bad, "mismatches": bad[:8], "tickets_zero": t0}
                    print(f"rows {M} {name:9s}: {u:7.2f} us/layer-call (split - this {us['split'] - u:+.2f}); "
                          f"bit-identical {not bad}; tickets 0 {t0}", flush=True)
                nt = f"{form}_notail"
                us[nt] = graph_us([fused_call(i, form, None, False) for i in range(layers)], reps) / layers
                sync()
                rec["variants"][nt] = {"form": form, "probe": "TAILS=False: GEMV + tickets (+ stage fold), no tail",
                                       "us_per_layer": round(us[nt], 3), "tickets_zero": int(ticket.abs().sum()) == 0}
                print(f"rows {M} {nt:9s}: {us[nt]:7.2f} us/layer-call (phase probe: no tail work)", flush=True)
        rec["compiled"] = mr.seen
        rec["us_per_layer"] = {k: round(v, 3) for k, v in us.items()}
        ph = {"split.norm_rows (split - pair)": us["split"] - us["pair"]}
        best = {}
        for form in forms:
            cand = {k: v for k, v in rec["variants"].items() if v.get("form") == form and "RP" in v}
            if not cand:
                continue
            bk = min(cand, key=lambda k: cand[k]["us_per_layer"])
            best[form] = bk
            ph[f"{form}.gemv+tickets (notail - pair)"] = us[f"{form}_notail"] - us["pair"]
            ph[f"{form}.tail (best - notail)"] = us[bk] - us[f"{form}_notail"]
        rec["phases_us"] = {k: round(v, 3) for k, v in ph.items()}
        rec["best"] = {f: {"variant": k, **rec["variants"][k]} for f, k in best.items()}
        bk_all = min(best.values(), key=lambda k: us[k]) if best else None
        rec["best_fused"] = bk_all
        rec["us_saved_per_layer"] = round(us["split"] - us[bk_all], 3) if bk_all else None
        rec["ms_saved_per_step_40_layers"] = round((us["split"] - us[bk_all]) * 40 / 1000.0, 4) if bk_all else None
        rec["fused_beats_split"] = bool(bk_all and us[bk_all] < us["split"])
        rec["bit_identical_all"] = all(v.get("bit_identical", True) for v in rec["variants"].values())
        rec["tickets_zero_all"] = all(v.get("tickets_zero", True) for v in rec["variants"].values())
        out[f"rows{M}"] = rec
        print(f"rows {M} phases (us/layer-call, measured): " + "; ".join(f"{k} {v:+.2f}" for k, v in ph.items()),
              flush=True)
        for name, metas in mr.seen.items():
            print(f"rows {M} compiled {name}: {metas}", flush=True)
        if bk_all:
            print(f"rows {M}: us/layer-call split {us['split']:.2f} vs fused {bk_all} {us[bk_all]:.2f} (saved "
                  f"{rec['us_saved_per_layer']:+.2f}; {rec['ms_saved_per_step_40_layers']:+.3f} ms/step at 40 "
                  f"layers); bit-identical all {rec['bit_identical_all']}; tickets 0 all {rec['tickets_zero_all']}",
                  flush=True)
    line = []
    for M in rows_list:
        r = out[f"rows{M}"]
        if r.get("best_fused"):
            line.append(f"R={M} split {r['us_per_layer']['split']:.2f} vs {r['best_fused']} "
                        f"{r['us_per_layer'][r['best_fused']]:.2f} us/layer-call")
    out["verdict"] = "; ".join(line)
    print("QKV-TAIL-TIMING: " + out["verdict"], flush=True)
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
    ap.add_argument("--rp", default="1,2,4,8", help="--timing: rows per pass of the fold form's one tail CTA / rows "
                    "per tail CTA of the stage form, to sweep (speed only; runtime defaults gemv_mxfp8.L10_TAIL_RP / "
                    "L10_STAGE_RP, $DSV41_QKV_TAIL_RP / $DSV41_QKV_TAIL_STAGE_RP)")
    ap.add_argument("--forms", default="fold,stage", help="--timing: fused forms to time (gemv_mxfp8.L10_FORMS)")
    ap.add_argument("--pin-m1", choices=["on", "off", "env"], default="on",
                    help="$DSV41_GEMV_PIN_M1 for --timing (default on: the headline's gemv_pin_m1, so the R-row GEMV "
                    "configs are the ones the step runs)")
    ap.add_argument("--qkv-tail", choices=["split", "fused"], default=None,
                    help="set $DSV41_QKV_TAIL for anything this process builds")
    ap.add_argument("--json", default=None, help="receipt path (default scratch/qkv-tail-<date>.json)")
    a = ap.parse_args(argv)
    if a.qkv_tail:
        os.environ["DSV41_QKV_TAIL"] = a.qkv_tail
    if a.pin_m1 != "env":
        os.environ["DSV41_GEMV_PIN_M1"] = "1" if a.pin_m1 == "on" else "0"
    if not (a.census or a.timing):
        a.census = not torch.cuda.is_available()
        a.timing = torch.cuda.is_available()
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "modeled": MODELED, "host": os.uname().nodename}
    if a.census:
        rep["census"] = census(rows, a.steps)
    if a.timing:
        rep["timing"] = timing(rows, a.layers, a.reps, tuple(int(m) for m in a.rope_modes.split(",")),
                               tuple(int(r) for r in a.rp.split(",") if r.strip()),
                               tuple(f.strip() for f in a.forms.split(",") if f.strip()))
    path = a.json or os.path.join(REPO, "scratch", f"qkv-tail-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"receipt {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
