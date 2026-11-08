"""M2-T15 bench + gates (docs/HANDOFF-M2-T15-ROUTER-PLAN.md, fix round docs/HANDOFF-M2-T15-FIX.md): us per layer of
the router (+ the 6-row plan) for the fused path vs the plan impls with a per-phase breakdown, the softplus probe (every
in-kernel form), the constructed-tie gate and the replay gate on recorded router inputs, self-replay. GPU only.
Receipts go to scratch/ (never docs/).

  # timing + softplus + constructed ties (synthetic; real shapes E=384, D=5120, k=6 and the mini E=8, D=256, k=2):
  python3 bench/router_plan_bench.py --json scratch/router-plan-<tag>.json
  # + the gate GEMV config sweep and the router kernel's warps sweep (speed-only knobs; bit-identity asserted):
  python3 bench/router_plan_bench.py --gate-sweep --warps-sweep --json scratch/router-plan-<tag>.json
  # record real router inputs first (eager static step; ~10^5 rows from 8 prompts x 256 steps x 40 layers):
  DSV41_ROUTER_DUMP=scratch/router-dump bench/first_light.py ... --mode static --glue-impl fused ...
  # then the replay gate on them (+ the timing):
  python3 bench/router_plan_bench.py --replay scratch/router-dump --json scratch/router-plan-<tag>.json

Timing method: per variant, `--layers` (40) layer-calls captured in one CUDA graph (distinct per-layer weights, so the
gate weight is read from HBM as in the step), replayed `--reps` times; us/layer = graph ms / layers. Variants:
  fused             route_rows' launches (F.linear on the cached fp32 weight, F.softplus, t9b2r2_router_pre,
                    torch.topk, t9b2r2_router_post) + at R > 1 the dedup launch (moe_rows.dedup_fused, m1_dedup_warp,
                    dg-inplace contiguous call shape at ALIGN 16 with a 77 %-hot table)
  plan_1cta         round 1: F.linear + m2_router_plan (one CTA: all rows + the plan)            [the "before"]
  plan              F.linear + m2_router_rows (grid R; at R > 1 the plan in the last row program, ticket)
  plan_split        F.linear + m2_router_rows + dedup_fused (two launches after the mm)
  plan_tsp          plan with F.softplus kept as a torch op (the form the box uses if no in-kernel form is exact)
  plan-gemv         m2_gate_fp32 (gate_dot: T9b-5 fixed-order fp32 GEMV, bf16 weight) + m2_router_rows (ticket)
  plan-gemv_split / plan-gemv_tsp   as above
Phases (each alone, same graph method): gate_mm (cuBLAS), gate_gemv (gate_dot), softplus (F.softplus), rows_route
(m2_router_rows without plan), rows_ticket_plan (with the plan), dedup (dedup_fused alone), fused_tail (the fused path
minus the mm and the dedup). In-kernel softplus timings use the first exact form from the probe (mode 1 if none: the
cost is the same order; the `_tsp` variants are what runs then).
Self-replay: every plan variant's outputs (ids, w and, at R > 1, every plan tensor) after `--reps` replays == after
the warm-up replays; plan == plan_split == plan_tsp (if a form is exact) bit for bit (same cuBLAS scores).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

E, K, DIM, RS, EPS = 384, 6, 5120, 1.5, 1e-20
SHAPES = {"real": (384, 6, 5120), "mini": (8, 2, 256)}      # (E, k, D): the real model; tests/mini_ckpt.MINI_CONFIG
GATES = {"R1": 10.0, "R6": 25.0}                             # us / layer (HANDOFF-M2-T15-FIX)


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


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t


def _graph_us(fn, layers: int, reps: int, snap=None):
    """us per layer of fn (layers layer-calls) as one CUDA graph; snap() -> tensors: (us, self-replay equal)."""
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
    first = [t.clone() for t in snap()] if snap is not None else None
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1000.0 / reps / layers
    if snap is None:
        return us, None
    return us, all(torch.equal(_bits(x), _bits(y)) for x, y in zip(first, snap()))


class _Setup:
    """Per-(R, shape) synthetic layers: bf16 gate weights (+ the fp32 copies cuBLAS reads), biases, one x, the
    dedup call shapes, per-layer workspaces and outputs; precomputed cuBLAS scores for the tail-only phases."""

    def __init__(self, R_: int, shape: str, layers: int):
        from runtime import moe_rows as MR
        from runtime.fused_glue_r2 import order_args, pick_order

        self.R, self.layers = R_, layers
        self.E, self.k, self.D = SHAPES[shape]
        g = torch.Generator(device="cpu").manual_seed(11)
        self.ordd = pick_order("cuda", self.k, R_, "possum")
        self.oa = order_args(self.ordd)
        self.gw = [(_f32(self.E, self.D, g=g, scale=0.02)).to(torch.bfloat16).cuda() for _ in range(layers)]
        self.gw32 = [w.float() for w in self.gw]
        self.bias = [_f32(self.E, g=g, scale=0.05).cuda() for _ in range(layers)]
        self.x = (_f32(R_, self.D, g=g, scale=0.5)).to(torch.bfloat16).float().cuda()
        self.plans = []
        for _ in range(layers):
            if R_ > 1:
                inpl, BR = _inpl(g, R_, e=self.E, k=self.k)
                self.plans.append((inpl, BR, MR.DedupBuffers(R_, self.k, "cuda"),
                                   torch.zeros(R_ * self.k, dtype=torch.long, device="cuda"),
                                   torch.zeros(R_ * self.k, dtype=torch.long, device="cuda")))
            else:
                self.plans.append(None)
        self.wss = [_ws(R_, gemv=True, e=self.E, k=self.k) for _ in range(layers)]
        self.outs = [(torch.zeros(R_, self.k, dtype=torch.long, device="cuda"),
                      torch.zeros(R_, self.k, dtype=torch.float32, device="cuda")) for _ in range(layers)]
        self.raw = [F.linear(self.x, w) for w in self.gw32]           # static scores for the tail-only phases
        self.sp_raw = [F.softplus(r) for r in self.raw]

    def plan_args(self, i: int):
        from runtime import router_plan as RP

        if self.plans[i] is None:
            return None
        inpl, BR, buf, ids_s, uv = self.plans[i]
        return RP.dedup_args(buf, self.R, self.k, BR, None, ids_s, uv, inpl, self.outs[i][0])

    def rows(self, i: int, raw, mode: int, plan, warps=None) -> None:
        from runtime import static_kernels as SK

        SK.router_rows_launch(raw, self.bias[i], self.wss[i], self.outs[i][0], self.outs[i][1], R=self.R, E=self.E,
                              K=self.k, order=self.oa, rs=RS, eps=EPS, softplus=mode, plan=plan, num_warps=warps)

    def dedup(self, i: int) -> None:
        from runtime import moe_rows as MR

        inpl, BR, buf, ids_s, uv = self.plans[i]
        MR.dedup_fused(buf, self.outs[i][0], self.outs[i][1], BR, ids_step=ids_s, uval=uv, inpl=inpl, impl="warp")

    def snap(self):
        out = [t for o in self.outs for t in o]
        for p in self.plans:
            if p is not None:
                inpl, BR, buf, ids_s, uv = p
                out += [buf.uniq, buf.counts, buf.valid, buf.slot, buf.rank, buf.prow, buf.pw, buf.arow, buf.posmat,
                        buf.wl, buf.ids_stage, buf.sidx, inpl["mi"], inpl["gl"], ids_s, uv]
        return out


def time_impls(R_: int, layers: int = 40, reps: int = 50, warps=None, shape: str = "real", sp_mode: int = 1,
               gate_cfg: Optional[tuple] = None, sp_exact: bool = False) -> Dict[str, object]:
    """us per layer of every variant and phase at R_ rows (module docstring) + self-replay / cross-variant equality.
    sp_mode: the in-kernel softplus form timed (1 / 2 / 3); sp_exact: it is the probe's exact form (then plan ==
    plan_tsp is checked)."""
    from runtime import moe_rows as MR
    from runtime import static_kernels as SK
    from runtime.router_plan import gate_kb

    st = _Setup(R_, shape, layers)
    L = layers
    kb = gate_kb(st.D)
    multi = R_ > 1

    def fused():
        for i in range(L):
            raw = F.linear(st.x, st.gw32[i])
            ids, w, _ = fused_tail(raw, st.bias[i], st.ordd, st.k)
            if multi:
                inpl, BR, buf, ids_s, uv = st.plans[i]
                MR.dedup_fused(buf, ids, w, BR, ids_step=ids_s, uval=uv, inpl=inpl, impl="warp")

    def plan_1cta():
        for i in range(L):
            raw = F.linear(st.x, st.gw32[i])
            SK.router_plan_launch(raw, st.bias[i], st.wss[i], st.outs[i][0], st.outs[i][1], R=R_, E=st.E, K=st.k,
                                  order=st.oa, rs=RS, eps=EPS, softplus=sp_mode, plan=st.plan_args(i))

    def variant(gemv=False, split=False, tsp=False):
        def fn():
            for i in range(L):
                if gemv:
                    raw = st.wss[i]["raw"]
                    SK.gate_dot(st.x, st.gw[i], raw, kb=kb, cfg=gate_cfg)
                else:
                    raw = F.linear(st.x, st.gw32[i])
                mode = sp_mode
                if tsp:
                    raw = F.softplus(raw)
                    mode = 0
                st.rows(i, raw, mode, None if split else st.plan_args(i), warps)
                if split and multi:
                    st.dedup(i)
        return fn

    us: Dict[str, float] = {}
    replay: Dict[str, Optional[bool]] = {}
    snaps: Dict[str, list] = {}

    def run(name, fn, snap=True):
        us[name], replay[name] = _graph_us(fn, L, reps, st.snap if snap else None)
        if snap:
            snaps[name] = [t.clone() for t in st.snap()]

    run("fused", fused, snap=False)
    if multi:                                    # the fused path's own dedup wrote the plan tensors: keep them
        snaps["fused_plan"] = [t.clone() for t in st.snap()[2 * L:]]
    run("plan_1cta", plan_1cta)
    run("plan", variant())
    run("plan_tsp", variant(tsp=True))
    if multi:
        run("plan_split", variant(split=True))
    run("plan-gemv", variant(gemv=True))
    run("plan-gemv_tsp", variant(gemv=True, tsp=True))
    if multi:
        run("plan-gemv_split", variant(gemv=True, split=True))
    # phases
    ph: Dict[str, float] = {}
    ph["gate_mm"] = _graph_us(lambda: [F.linear(st.x, w) for w in st.gw32], L, reps)[0]
    ph["gate_gemv"] = _graph_us(lambda: [SK.gate_dot(st.x, st.gw[i], st.wss[i]["raw"], kb=kb, cfg=gate_cfg)
                                         for i in range(L)], L, reps)[0]
    ph["softplus"] = _graph_us(lambda: [F.softplus(r) for r in st.raw], L, reps)[0]
    ph["rows_route"] = _graph_us(lambda: [st.rows(i, st.raw[i], sp_mode, None, warps) for i in range(L)], L, reps)[0]
    ph["rows_route_tsp"] = _graph_us(lambda: [st.rows(i, st.sp_raw[i], 0, None, warps) for i in range(L)], L,
                                     reps)[0]
    if multi:
        ph["rows_ticket_plan"] = _graph_us(lambda: [st.rows(i, st.raw[i], sp_mode, st.plan_args(i), warps)
                                                    for i in range(L)], L, reps)[0]
        ph["dedup"] = _graph_us(lambda: [st.dedup(i) for i in range(L)], L, reps)[0]
    ph["fused_tail"] = _graph_us(lambda: [fused_tail(st.raw[i], st.bias[i], st.ordd, st.k) for i in range(L)], L,
                                 reps)[0]

    def same(a, b):
        return a in snaps and b in snaps and all(torch.equal(_bits(x), _bits(y)) for x, y in zip(snaps[a], snaps[b]))

    cross = {"plan==plan_1cta": same("plan", "plan_1cta")}
    if multi:
        cross["plan==plan_split"] = same("plan", "plan_split")
        cross["plan-gemv==plan-gemv_split"] = same("plan-gemv", "plan-gemv_split")
        ref_v = "plan" if sp_exact else "plan_tsp"     # the form the box would run
        cross[f"plan tensors: {ref_v}==fused (today's dedup)"] = all(
            torch.equal(_bits(x), _bits(y)) for x, y in zip(snaps[ref_v][2 * L:], snaps["fused_plan"]))
    if sp_exact:
        cross["plan==plan_tsp (in-kernel softplus exact)"] = same("plan", "plan_tsp")
    return {"us_per_layer": us, "phases_us": ph, "self_replay": replay, "cross_equal": cross,
            "softplus_mode": sp_mode, "gate_kb": kb, "gate_cfg": list(gate_cfg or SK.gate_config(R_)),
            "rows_warps": warps or SK.M2_ROWS_WARPS}


def gate_sweep(R_: int, shape: str, layers: int, reps: int) -> dict:
    """gate_dot configs (speed only): us per layer each, the best, and bit-identity of every config's scores."""
    from runtime import static_kernels as SK
    from runtime.fused_glue_r2 import fixed_dot_torch
    from runtime.router_plan import gate_kb

    st = _Setup(R_, shape, layers)
    kb = gate_kb(st.D)
    nk = st.D // kb
    rp = min(8, 1 << max(0, (R_ - 1).bit_length()))
    cands = []
    for rb in sorted({1, 2, rp}):
        if rb > rp:
            continue
        for bn in (1, 2, 4, 8):
            for u in sorted({max(1, min(nk, u_)) for u_ in (nk // 2, nk)}):
                for nw in (2, 4, 8):
                    cands.append((rb, bn, u, nw, 1))
    want = fixed_dot_torch(st.x, st.gw[0], kb)
    res, bad = [], []
    for cfg in cands:
        try:
            out = torch.empty(R_, st.E, dtype=torch.float32, device="cuda")
            SK.gate_dot(st.x, st.gw[0], out, kb=kb, cfg=cfg)
            torch.cuda.synchronize()
            if not torch.equal(_bits(out), _bits(want)):
                bad.append(list(cfg))
            t = _graph_us(lambda c=cfg: [SK.gate_dot(st.x, st.gw[i], st.wss[i]["raw"], kb=kb, cfg=c)
                                         for i in range(layers)], layers, reps)[0]
            res.append((t, list(cfg)))
        except Exception as e:  # a config that does not compile here: recorded
            res.append((float("inf"), list(cfg) + [f"{type(e).__name__}: {str(e)[:120]}"]))
    res.sort(key=lambda r: r[0])
    return {"kb": kb, "n": len(res), "best": res[0], "top8": res[:8], "configs_not_bit_identical": bad,
            "env_line": f"{SK.M2_GATE_CFG_ENV}=\"{rp}={','.join(str(v) for v in res[0][1][:5])}\""}


def softplus_probe(n: int = 1 << 22) -> dict:
    """Every in-kernel softplus form vs F.softplus on the probe values + every 256th fp32 bit pattern; per form the
    mismatch count (NaN vs NaN equal), examples, and whether every mismatch has exp(x) < 2^-126 (the FTZ region)."""
    from runtime import static_kernels as SK
    from runtime.router_plan import SOFTPLUS_FORMS, softplus_probe_inputs

    g = torch.Generator(device="cpu").manual_seed(0)
    bits = torch.arange(0, 1 << 32, 256, dtype=torch.int64, device="cpu").to(torch.int32).view(torch.float32)
    x = torch.cat([softplus_probe_inputs(g, n), bits]).cuda()
    ref = F.softplus(x)
    sub = torch.exp(x.double()) < 2.0 ** -126
    out = {"n": int(x.numel()), "forms": {}, "exact": None}
    for form in ("fused", "fused-sub", "fused-sub24"):
        got = SK.softplus_probe(x, SOFTPLUS_FORMS[form])
        bad = (got.view(torch.int32) != ref.view(torch.int32)) & ~(torch.isnan(got) & torch.isnan(ref))
        out["forms"][form] = {"mismatch": int(bad.sum()), "mismatch_outside_subnormal_exp": int((bad & ~sub).sum()),
                              "examples": [[float(a), float(b), float(c)] for a, b, c in
                                           zip(x[bad][:8], got[bad][:8], ref[bad][:8])]}
        if out["exact"] is None and int(bad.sum()) == 0:
            out["exact"] = form
    out["mismatch"] = out["forms"]["fused"]["mismatch"]          # round 1's field
    return out


def _plan_from_raw(raw, bias, oa, sp_mode: int, plan=None, ws=None, k=K, rs=RS):
    from runtime import static_kernels as SK

    T_, e = raw.shape
    ws = ws or _ws(T_, e=e, k=k)
    ids = torch.empty(T_, k, dtype=torch.long, device="cuda")
    w = torch.empty(T_, k, dtype=torch.float32, device="cuda")
    src = raw if sp_mode else F.softplus(raw)
    SK.router_rows_launch(src.contiguous(), bias, ws, ids, w, R=T_, E=e, K=k, order=oa, rs=rs, eps=EPS,
                          softplus=sp_mode, plan=plan)
    return ids, w, ws


def tie_gate(n_rows: int, sp_mode: int) -> dict:
    """The constructed-tie gate on m2_router_rows (sp_mode: the exact in-kernel form, or 0 = F.softplus first)."""
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
                ids, w, _ = _plan_from_raw(raw, bias, oa, sp_mode, ws=ws)
                bad[0] += (ws["tk"] != wt).any(-1).sum()
                bad[1] += (ids != wi).any(-1).sum()
                bad[2] += (_bits(w) != _bits(ww)).any(-1).sum()
                n += R_
        out[f"R{R_}"] = {"rows": n, "rows_differing": dict(zip(("topk_order", "ids", "w"), bad.tolist()))}
    return out


def replay_gate(root: str, sp_mode: int, max_rows: int = 0) -> dict:
    """The recorded inputs (router_plan's dump hook, the fused path): per record, the router kernel (from the recorded
    gate scores) vs the fused tail recomputed here vs the recorded live outputs (the live weight-sum order when the
    record carries it, else pick_order's); torch.topk's order vs the kernel's; at R > 1 the ticketed plan tensors vs
    dedup_fused (synthetic 77 %-hot table). Only the tail is gated here (the scores are the recorded cuBLAS ones)."""
    from runtime import moe_rows as MR
    from runtime import router_plan as RP
    from runtime.fused_glue_r2 import NATIVE, order_args, pick_order

    biases = {}
    for p in glob.glob(os.path.join(root, "router-bias-*.pt")):
        d = torch.load(p, map_location="cpu")
        biases[d["layer"]] = d
    files = sorted(glob.glob(os.path.join(root, "router-dump-*.pt")))
    res = {"files": len(files), "layers": sorted(biases), "rows": 0, "records": 0, "softplus_mode": sp_mode,
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
            ids, w, ws = _plan_from_raw(raw, bias, oa, sp_mode, k=k, rs=rs)
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
                        _plan_from_raw(raw, bias, oa, sp_mode, plan=a, k=k, rs=rs)
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


def verdict(us: dict, exact_form: Optional[str]) -> dict:
    """The fix's gates on the real shape (R=1 <= 10, R=6 <= 25 us / layer and below fused) per impl, the launch
    form to keep at R=6, and the launches per layer of the configuration that would run (in-kernel softplus if a
    form is exact, else the `_tsp` variants; split + F.softplus is not timed: split + (tsp - plan) stands for it)."""
    out = {}
    tsp = exact_form is None
    r1 = us.get("R1", {}).get("us_per_layer", {})
    r6 = us.get("R6", {}).get("us_per_layer", {})
    if not r1 or not r6:
        return out
    for impl in ("plan", "plan-gemv"):
        sfx = "_tsp" if tsp else ""
        t1 = r1.get(impl + sfx)
        ticket6 = r6.get(impl + sfx)
        split6 = r6.get(impl + "_split")
        if tsp and split6 is not None:
            split6 = split6 + (r6[impl + "_tsp"] - r6[impl])
        form = "ticket" if split6 is None or ticket6 <= split6 else "split"
        t6 = ticket6 if form == "ticket" else split6
        n = 1 + int(tsp) + int(form == "split")
        out[impl] = {"R1_us": t1, "R6_us": t6, "launch_form": form, "R6_ticket_us": ticket6, "R6_split_us": split6,
                     "fused_R1": r1.get("fused"), "fused_R6": r6.get("fused"),
                     "pass_R1": t1 is not None and t1 <= GATES["R1"],
                     "pass_R6": t6 is not None and t6 <= GATES["R6"] and t6 < r6.get("fused", float("inf")),
                     "launches_R6": (f"mm + {n}" if impl == "plan" else str(n + 1)),
                     "numerics": "bit-equal to fused" if impl == "plan" else "own fixed-order oracle (A3 vs fused)"}
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warps", type=int, default=None, help="router-rows kernel num_warps (default runtime 1)")
    ap.add_argument("--shapes", default="real,mini", help="comma list of " + ",".join(SHAPES))
    ap.add_argument("--rows", default="1,6", help="comma list of row counts")
    ap.add_argument("--gate-cfg", default=None, help="gate_dot RB,BN,U,num_warps,num_stages (default: the table)")
    ap.add_argument("--gate-sweep", action="store_true", help="time gate_dot configs (real shape, every --rows)")
    ap.add_argument("--warps-sweep", action="store_true", help="plan / plan-gemv at router-rows warps 1, 2, 4")
    ap.add_argument("--tie-rows", type=int, default=100_000, help="constructed-tie gate rows (half at R=1, half R=6)")
    ap.add_argument("--replay", default=None, help="dump dir of router_plan's recording hook")
    ap.add_argument("--max-replay-rows", type=int, default=0)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--json", default=None, help="receipt path (default scratch/router-plan-<date>.json)")
    return ap


def main(argv=None) -> int:
    from runtime.router_plan import SOFTPLUS_FORMS

    a = build_parser().parse_args(argv)
    assert torch.cuda.is_available(), "GPU only"
    path = a.json or os.path.join(REPO, "scratch", f"router-plan-{time.strftime('%Y%m%d-%H%M')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "torch": torch.__version__, "device": torch.cuda.get_device_name(), "errors": []}
    t0 = time.time()
    rows = [int(r) for r in a.rows.split(",") if r]
    shapes = [s for s in a.shapes.split(",") if s]
    gate_cfg = tuple(int(v) for v in a.gate_cfg.split(",")) if a.gate_cfg else None
    rep["softplus"] = softplus_probe()
    exact = rep["softplus"]["exact"]
    sp_mode = SOFTPLUS_FORMS[exact] if exact else 0
    counts = {f: v["mismatch"] for f, v in rep["softplus"]["forms"].items()}
    print(f"[softplus] exact in-kernel form: {exact} (mismatches {counts} of {rep['softplus']['n']:,})", flush=True)
    rep["tie_gate"] = {"softplus_mode": sp_mode, **tie_gate(a.tie_rows, sp_mode)}
    print(f"[ties] {rep['tie_gate']}", flush=True)
    if a.replay:
        rep["replay_gate"] = replay_gate(a.replay, sp_mode, a.max_replay_rows)
        print(f"[replay] {rep['replay_gate']}", flush=True)
    if not a.no_timing:
        rep["timing"] = {}
        for shape in shapes:
            rep["timing"][shape] = {}
            for R_ in rows:
                try:
                    r = time_impls(R_, a.layers, a.reps, a.warps, shape, sp_mode or 1, gate_cfg, bool(exact))
                    rep["timing"][shape][f"R{R_}"] = r
                    print(f"[timing] {shape} R={R_}: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                                  r["us_per_layer"].items()), flush=True)
                    print(f"         phases: " + ", ".join(f"{k} {v:.2f}" for k, v in r["phases_us"].items())
                          + f" | self-replay {r['self_replay']} | equal {r['cross_equal']}", flush=True)
                except Exception as e:  # report, keep the receipt
                    rep["errors"].append(f"timing {shape} R={R_}: {type(e).__name__}: {str(e)[:300]}")
                    print(f"[timing] {shape} R={R_}: ERROR {rep['errors'][-1]}", flush=True)
        if "real" in rep["timing"]:
            rep["verdict"] = verdict(rep["timing"]["real"], exact)
            print(f"[verdict] {rep['verdict']}", flush=True)
    if a.gate_sweep:
        rep["gate_sweep"] = {}
        for R_ in rows:
            try:
                rep["gate_sweep"][f"R{R_}"] = gate_sweep(R_, "real", a.layers, a.reps)
                gs = rep["gate_sweep"][f"R{R_}"]
                print(f"[gate-sweep] R={R_}: best {gs['best']} ({gs['env_line']}); not bit-identical: "
                      f"{gs['configs_not_bit_identical']}", flush=True)
            except Exception as e:
                rep["errors"].append(f"gate sweep R={R_}: {type(e).__name__}: {str(e)[:300]}")
    if a.warps_sweep:
        rep["warps_sweep"] = {}
        for R_ in rows:
            for nw in (1, 2, 4):
                try:
                    r = time_impls(R_, a.layers, a.reps, nw, "real", sp_mode or 1, gate_cfg, bool(exact))
                    rep["warps_sweep"][f"R{R_}_w{nw}"] = {k: r["us_per_layer"][k] for k in r["us_per_layer"]
                                                          if k.startswith("plan")}
                    rep["warps_sweep"][f"R{R_}_w{nw}"]["rows_route"] = r["phases_us"]["rows_route"]
                    print(f"[warps] R={R_} w={nw}: {rep['warps_sweep'][f'R{R_}_w{nw}']}", flush=True)
                except Exception as e:
                    rep["errors"].append(f"warps sweep R={R_} w={nw}: {type(e).__name__}: {str(e)[:300]}")
    rep["seconds"] = round(time.time() - t0, 1)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[receipt] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
