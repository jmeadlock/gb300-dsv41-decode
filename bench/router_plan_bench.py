"""M2-T15 bench + gates (docs/HANDOFF-M2-T15-ROUTER-PLAN.md): us per layer of the router (+ the 6-row plan) for the
fused path vs the plan impls, the softplus probe, the constructed-tie gate and the replay gate on recorded router
inputs. GPU only. Receipts go to scratch/ (never docs/).

  # timing + softplus + constructed ties (synthetic, real shapes E=384, D=5120, k=6):
  python3 bench/router_plan_bench.py --json scratch/router-plan-<tag>.json
  # record real router inputs first (eager static step; ~10^5 rows from 8 prompts x 256 steps x 40 layers):
  DSV41_ROUTER_DUMP=scratch/router-dump bench/first_light.py ... --mode static --glue-impl fused ...
  # then the replay gate on them (+ the timing):
  python3 bench/router_plan_bench.py --replay scratch/router-dump --json scratch/router-plan-<tag>.json

Timing method: per impl, `--layers` (40) layer-calls captured in one CUDA graph (distinct per-layer weights, so the
gate weight is read from HBM as in the step), replayed `--reps` times; us/layer = graph ms / layers. The fused path
is exactly route_rows' launches (F.linear on the cached fp32 weight, F.softplus, t9b2r2_router_pre, torch.topk,
t9b2r2_router_post) + at R > 1 the dedup launch (moe_rows.dedup_fused, m1_dedup_warp, dg-inplace contiguous call shape
at ALIGN 16 with a 77 %-hot table); plan = F.linear + one m2_router_plan launch (with the same plan); plan-gemv = one
m2_router_plan launch reading the bf16 weight.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

E, K, DIM, RS, EPS = 384, 6, 5120, 1.5, 1e-20


def _f32(*shape, g=None, scale=1.0):
    return (torch.randn(*shape, generator=g, device="cpu", dtype=torch.float32) * scale)


def _ws(R_, gemv=False, e=E, k=K):
    from runtime.router_plan import NET, _np2

    f = dict(dtype=torch.float32, device="cuda")
    ws = {"sc": torch.zeros(R_, e, **f), "bsc": torch.zeros(R_, e, **f),
          "tk": torch.zeros(R_, k, dtype=torch.long, device="cuda"),
          "net": torch.zeros(_np2(R_), NET, dtype=torch.long, device="cuda"),
          "ticket": torch.zeros(1, dtype=torch.int32, device="cuda")}
    if gemv:
        ws["raw"] = torch.zeros(R_, e, **f)
    return ws


def _inpl(g, R_, BR=16, hot_frac=0.77, S=16, e=E, k=K):
    P = R_ * k
    hot = torch.rand(e, generator=g, device="cpu", dtype=torch.float32) < hot_frac
    gtab = torch.full((e,), -1, dtype=torch.int32, device="cpu")
    gtab[hot] = torch.arange(int(hot.sum()), dtype=torch.int32, device="cpu")
    H = int(hot.sum())
    G = H + S
    return {"gtab": gtab.cuda(), "H": H, "S": S, "dummy": torch.tensor(0, dtype=torch.long, device="cuda"),
            "has_dummy": H > 0, "v2": True, "layout": "contiguous", "G": G, "gscr": G, "step_sf": False,
            "mi": torch.full((P * BR,), -1, dtype=torch.int32, device="cuda"),
            "gl": torch.zeros(G, dtype=torch.int32, device="cuda"), "ovf": torch.zeros(1, dtype=torch.int32,
                                                                                   device="cuda")}, BR


def fused_tail(raw, bias, ordd, k=K, rs=RS):
    """route()/route_rows()'s ops after the mm (F.softplus, t9b2r2_router_pre, torch.topk, t9b2r2_router_post)."""
    from runtime.fused_glue_r2 import launch_router_post, launch_router_pre

    T_, e = raw.shape
    sp = F.softplus(raw).contiguous()
    sc = torch.empty(T_, e, dtype=torch.float32, device="cuda")
    bsc = torch.empty(T_, e, dtype=torch.float32, device="cuda")
    launch_router_pre(sp.view(-1), bias, sc.view(-1), bsc.view(-1))
    idx = bsc.topk(k, dim=-1)[1]
    ids = torch.empty(T_, k, dtype=torch.long, device="cuda")
    w = torch.empty(T_, k, dtype=torch.float32, device="cuda")
    if T_ == 1:
        launch_router_post(idx.view(-1), sc.view(-1), ids.view(-1), w.view(-1), ordd=ordd, eps=EPS, rs=rs)
    else:
        launch_router_post(idx.contiguous(), sc, ids, w, ordd=ordd, eps=EPS, rs=rs)
    return ids, w, idx


def _graph_us(fn, layers: int, reps: int) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / reps / layers


def time_impls(R_: int, layers: int = 40, reps: int = 50, warps=None) -> Dict[str, float]:
    """us per layer: fused / plan / plan (F.softplus kept) / plan-gemv at R_ rows (module docstring)."""
    from runtime import moe_rows as MR
    from runtime import router_plan as RP
    from runtime import static_kernels as SK
    from runtime.fused_glue_r2 import order_args, pick_order
    from runtime.router_plan import GATE_KB

    g = torch.Generator(device="cpu").manual_seed(11)
    ordd = pick_order("cuda", K, R_, "possum")
    oa = order_args(ordd)
    gw = [(_f32(E, DIM, g=g, scale=0.02)).to(torch.bfloat16).cuda() for _ in range(layers)]
    gw32 = [w.float() for w in gw]
    bias = [_f32(E, g=g, scale=0.05).cuda() for _ in range(layers)]
    x = (_f32(R_, DIM, g=g, scale=0.5)).to(torch.bfloat16).float().cuda()
    plans = []
    for i in range(layers):
        if R_ > 1:
            inpl, BR = _inpl(g, R_)
            plans.append((inpl, BR, MR.DedupBuffers(R_, K, "cuda"),
                          torch.zeros(R_ * K, dtype=torch.long, device="cuda"),
                          torch.zeros(R_ * K, dtype=torch.long, device="cuda")))
        else:
            plans.append(None)
    wss = [_ws(R_, gemv=True) for _ in range(layers)]
    outs = [(torch.empty(R_, K, dtype=torch.long, device="cuda"), torch.empty(R_, K, dtype=torch.float32,
                                                                              device="cuda")) for _ in range(layers)]

    def fused():
        for i in range(layers):
            raw = F.linear(x, gw32[i])
            ids, w, _ = fused_tail(raw, bias[i], ordd)
            if plans[i] is not None:
                inpl, BR, buf, ids_s, uv = plans[i]
                MR.dedup_fused(buf, ids, w, BR, ids_step=ids_s, uval=uv, inpl=inpl)

    def plan(softplus=True):
        for i in range(layers):
            raw = F.linear(x, gw32[i])
            if not softplus:
                raw = F.softplus(raw)
            a = None
            if plans[i] is not None:
                inpl, BR, buf, ids_s, uv = plans[i]
                a = RP.dedup_args(buf, R_, K, BR, None, ids_s, uv, inpl, outs[i][0])
            SK.router_plan_launch(raw, bias[i], wss[i], outs[i][0], outs[i][1], R=R_, E=E, K=K, order=oa, rs=RS,
                                  eps=EPS, softplus=softplus, plan=a, num_warps=warps)

    def plan_gemv():
        for i in range(layers):
            a = None
            if plans[i] is not None:
                inpl, BR, buf, ids_s, uv = plans[i]
                a = RP.dedup_args(buf, R_, K, BR, None, ids_s, uv, inpl, outs[i][0])
            SK.router_plan_launch(wss[i]["raw"], bias[i], wss[i], outs[i][0], outs[i][1], R=R_, E=E, K=K, order=oa,
                                  rs=RS, eps=EPS, softplus=True, x=x, gw=gw[i], kb=GATE_KB, plan=a)

    def mm_only():
        for i in range(layers):
            F.linear(x, gw32[i])

    return {"fused": _graph_us(fused, layers, reps), "plan": _graph_us(plan, layers, reps),
            "plan_torch_softplus": _graph_us(lambda: plan(False), layers, reps),
            "plan-gemv": _graph_us(plan_gemv, layers, reps), "gate_mm_alone": _graph_us(mm_only, layers, reps)}


def softplus_probe(n: int = 1 << 22) -> dict:
    from runtime import static_kernels as SK
    from runtime.router_plan import softplus_probe_inputs

    g = torch.Generator(device="cpu").manual_seed(0)
    bits = torch.arange(0, 1 << 32, 256, dtype=torch.int64, device="cpu").to(torch.int32).view(torch.float32)
    x = torch.cat([softplus_probe_inputs(g, n), bits]).cuda()
    got, ref = SK.softplus_probe(x), F.softplus(x)
    bad = (got.view(torch.int32) != ref.view(torch.int32)) & ~(torch.isnan(got) & torch.isnan(ref))
    return {"n": int(x.numel()), "mismatch": int(bad.sum()),
            "examples": [[float(a), float(b), float(c)] for a, b, c in zip(x[bad][:8], got[bad][:8], ref[bad][:8])]}


def _plan_from_raw(raw, bias, oa, softplus_exact: bool, plan=None, ws=None, k=K, rs=RS):
    from runtime import static_kernels as SK

    T_, e = raw.shape
    ws = ws or _ws(T_, e=e, k=k)
    ids = torch.empty(T_, k, dtype=torch.long, device="cuda")
    w = torch.empty(T_, k, dtype=torch.float32, device="cuda")
    src = raw if softplus_exact else F.softplus(raw)
    SK.router_plan_launch(src.contiguous(), bias, ws, ids, w, R=T_, E=e, K=k, order=oa, rs=rs, eps=EPS,
                          softplus=softplus_exact, plan=plan)
    return ids, w, ws


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def tie_gate(n_rows: int, softplus_exact: bool) -> dict:
    from runtime.fused_glue_r2 import order_args, pick_order
    from runtime.router_plan import RouterPlan

    out = {}
    g = torch.Generator(device="cpu").manual_seed(1)
    for R_ in (1, 6):
        ordd = pick_order("cuda", K, R_, "possum")
        oa = order_args(ordd)
        ws = _ws(R_)
        bad = torch.zeros(3, dtype=torch.long, device="cuda")
        n = 0
        while n < n_rows // 2:
            for raw, bias in RouterPlan.raw_draws(g, R_, E, 6):
                raw, bias = raw.cuda(), bias.cuda()
                wi, ww, wt = fused_tail(raw, bias, ordd)
                ids, w, _ = _plan_from_raw(raw, bias, oa, softplus_exact, ws=ws)
                bad[0] += (ws["tk"] != wt).any(-1).sum()
                bad[1] += (ids != wi).any(-1).sum()
                bad[2] += (_bits(w) != _bits(ww)).any(-1).sum()
                n += R_
        out[f"R{R_}"] = {"rows": n, "rows_differing": dict(zip(("topk_order", "ids", "w"), bad.tolist()))}
    return out


def replay_gate(root: str, softplus_exact: bool, max_rows: int = 0) -> dict:
    """The recorded inputs (router_plan's dump hook, the fused path): per record, plan (from the recorded gate scores)
    vs the fused tail recomputed here vs the recorded live outputs (the live weight-sum order when the record carries
    it, else pick_order's); torch.topk's order vs the kernel's; at R > 1 the plan tensors vs dedup_fused (synthetic
    77 %-hot table). Only the tail is gated here (the scores are the recorded cuBLAS ones); the records' x rows are
    kept for a GEMV drift study with the real gate weights, not used here."""
    from runtime import moe_rows as MR
    from runtime import router_plan as RP
    from runtime.fused_glue_r2 import NATIVE, order_args, pick_order

    biases = {}
    for p in glob.glob(os.path.join(root, "router-bias-*.pt")):
        d = torch.load(p, map_location="cpu")
        biases[d["layer"]] = d
    files = sorted(glob.glob(os.path.join(root, "router-dump-*.pt")))
    res = {"files": len(files), "layers": sorted(biases), "rows": 0, "records": 0,
           "rows_differing": {"topk_order": 0, "ids_vs_fused": 0, "w_vs_fused": 0, "ids_vs_live": 0, "w_vs_live": 0},
           "plan_records": 0, "plan_records_differing": 0, "by_rows": {}}
    g = torch.Generator(device="cpu").manual_seed(5)
    orders = {}
    for p in files:
        for rec in torch.load(p, map_location="cpu"):
            meta = biases.get(rec["layer"])
            if meta is None or float(meta.get("gate_temp", 1.0)) != 1.0:
                res.setdefault("skipped", 0)
                res["skipped"] = res["skipped"] + 1
                continue
            k, rs = int(meta["topk"]), float(meta["route_scale"])
            raw = rec["raw"].float().cuda()
            T_, e = raw.shape
            key = (T_, k)
            ro = rec.get("order")
            if ro is not None and ro.get("order") is not None and ro["order"] != NATIVE:
                ordd = {"order": tuple(ro["order"]), "fin": ro.get("fin", "mul"), "exact": True,
                        "name": ro.get("name")}
                res["live_order"] = res.get("live_order", 0) + 1
            else:
                if key not in orders:
                    orders[key] = pick_order("cuda", k, T_, "possum")
                ordd = orders[key]
            oa = order_args(ordd)
            bias = meta["bias"].float().cuda()
            wi, ww, wt = fused_tail(raw, bias, ordd, k, rs)
            ids, w, ws = _plan_from_raw(raw, bias, oa, softplus_exact, k=k, rs=rs)
            d = res["rows_differing"]
            d["topk_order"] += int((ws["tk"] != wt).any(-1).sum())
            d["ids_vs_fused"] += int((ids != wi).any(-1).sum())
            d["w_vs_fused"] += int((_bits(w) != _bits(ww)).any(-1).sum())
            li, lw = rec["ids"].cuda().view(T_, k), rec["w"].float().cuda().view(T_, k)
            d["ids_vs_live"] += int((ids != li).any(-1).sum())
            d["w_vs_live"] += int((_bits(w) != _bits(lw)).any(-1).sum())
            res["rows"] += T_
            res["records"] += 1
            res["by_rows"][str(T_)] = res["by_rows"].get(str(T_), 0) + T_
            if T_ > 1 and T_ * k <= MR.DEDUP_MAX_PAIRS:
                inpl0, BR = _inpl(g, T_, e=e, k=k)
                outs = []
                for mine in (True, False):
                    inpl = dict(inpl0, mi=inpl0["mi"].clone(), gl=inpl0["gl"].clone(), ovf=inpl0["ovf"].clone())
                    buf = MR.DedupBuffers(T_, k, "cuda")
                    ids_s = torch.zeros(T_ * k, dtype=torch.long, device="cuda")
                    uv = torch.zeros(T_ * k, dtype=torch.long, device="cuda")
                    if mine:
                        i2 = torch.empty(T_, k, dtype=torch.long, device="cuda")
                        a = RP.dedup_args(buf, T_, k, BR, None, ids_s, uv, inpl, i2)
                        _plan_from_raw(raw, bias, oa, softplus_exact, plan=a, k=k, rs=rs)
                        m = RP.plan_out_dict(buf, None, inpl, a)
                    else:
                        m = MR.dedup_fused(buf, wi, ww, BR, ids_step=ids_s, uval=uv, inpl=inpl, impl="warp")
                    outs.append((m, inpl, ids_s, uv))
                (m1, p1, i1, u1), (m0, p0, i0, u0) = outs
                ok = all(not isinstance(v, torch.Tensor) or torch.equal(_bits(m1[k_]), _bits(v)) for k_, v in m0.items())
                ok = ok and all(torch.equal(p1[n], p0[n]) for n in ("mi", "gl", "ovf")) and torch.equal(i1, i0) \
                    and torch.equal(u1, u0)
                res["plan_records"] += 1
                res["plan_records_differing"] += int(not ok)
            if max_rows and res["rows"] >= max_rows:
                return res
    return res


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warps", type=int, default=None, help="plan kernel num_warps (default runtime M2_WARPS)")
    ap.add_argument("--tie-rows", type=int, default=100_000, help="constructed-tie gate rows (half at R=1, half R=6)")
    ap.add_argument("--replay", default=None, help="dump dir of router_plan's recording hook")
    ap.add_argument("--max-replay-rows", type=int, default=0)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--json", default=None, help="receipt path (default scratch/router-plan-<date>.json)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    assert torch.cuda.is_available(), "GPU only"
    path = a.json or os.path.join(REPO, "scratch", f"router-plan-{time.strftime('%Y%m%d-%H%M')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "torch": torch.__version__, "device": torch.cuda.get_device_name(), "errors": []}
    t0 = time.time()
    rep["softplus"] = softplus_probe()
    exact = rep["softplus"]["mismatch"] == 0
    print(f"[softplus] {rep['softplus']['mismatch']} of {rep['softplus']['n']:,} differ from F.softplus", flush=True)
    rep["tie_gate"] = {"softplus_in_kernel": exact, **tie_gate(a.tie_rows, exact)}
    print(f"[ties] {rep['tie_gate']}", flush=True)
    if a.replay:
        rep["replay_gate"] = replay_gate(a.replay, exact, a.max_replay_rows)
        print(f"[replay] {rep['replay_gate']}", flush=True)
    if not a.no_timing:
        rep["us_per_layer"] = {}
        for R_ in (1, 6):
            try:
                rep["us_per_layer"][f"R{R_}"] = time_impls(R_, a.layers, a.reps, a.warps)
            except Exception as e:  # report, keep the receipt
                rep["errors"].append(f"timing R={R_}: {type(e).__name__}: {str(e)[:300]}")
            print(f"[timing] R={R_}: {rep['us_per_layer'].get(f'R{R_}')}", flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[receipt] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
