"""GPU tests of M2-T15 (docs/HANDOFF-M2-T15-ROUTER-PLAN.md; fix round docs/HANDOFF-M2-T15-FIX.md; runtime/router_plan.py,
runtime/static_kernels.py m2_router_rows / m2_gate_fp32, round 1's m2_router_plan). Skipped without CUDA (Mac); on the
box a missing triton FAILS. Same docker line as tests/gpu/test_static_rows_gpu.py with this file:

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_router_plan_gpu.py

  * softplus: every in-kernel form (1 libdevice; 2 / 3 the fp64 subnormal-exp forms) vs F.softplus on 2^22 random
    values + a dense grid around the threshold + every 2^8-th fp32 bit pattern (16.7M) + edges -> mismatch counts
    printed; the first exact form is what the self-check would pick (none: F.softplus stays a torch op);
  * THE tie gate: at the real shapes (E = 384, k = 6) over 10^5 router rows at R = 1 and R = 6 -- realistic scores,
    tie-heavy (17 raw values x a two-valued bias), "mixed" (distinct values AND ties AND ties straddling the 6th
    place), all-equal rows, repeated rows, +-0, beyond-threshold and underflowing scores: m2_router_rows' topk order
    (`tk`) == torch.topk(bsc, 6).indices ELEMENT FOR ELEMENT (order included), and (ids, w) == the fused path (the
    mm-free tail: F.softplus, t9b2r2_router_pre, torch.topk, t9b2r2_router_post) bit for bit, with F.softplus first
    and with the exact in-kernel form (if any); m2_router_rows == round 1's single-CTA kernel bit for bit;
  * the R-row plan in the last row program (ticket) == moe_rows.dedup_fused (m1_dedup_warp on the device) on every
    buffer for every call shape (no inpl / masked_m / inpl v1 / v2 contiguous / psum / masked x step SF), incl.
    overflow; tickets back to 0;
  * plan-gemv: gate_dot == fused_glue_r2.fixed_dot_torch (KB = gate_kb(5120) = 512) for every gate config of a small
    sweep (bit-identical across configs); the route on its scores == its oracle; vs cuBLAS's scores (the fused path):
    raw ulp histogram, rows whose id SET differs, max |dw| -> printed (A3 class, not gated);
  * graph capture: 40 layers of plan (mm + router-rows + ticket plan) and plan-gemv (gate_dot + router-rows + ticket
    plan), own tickets, captured once, replayed 3x == eager, tickets back to 0;
  * mini checkpoints: StaticStep with $DSV41_ROUTER_IMPL=plan (self-check passes: printed) == the fused step,
    torch.equal on logits / main_hidden / caches, 1 row and 6 rows (dg-inplace, both parities, both launch forms),
    graph == static-eager; plan-gemv: its self-check passes and graph == static-eager;
  * recorded inputs: the dump hook on mini4's fused static-eager steps (1 and 6 rows) -> the bench's replay gate (the
    router kernel == fused tail == live outputs, topk order included; ticketed plan tensors == dedup_fused), 0 rows
    differing asserted;
  * timing (informative): bench/router_plan_bench.time_impls at R = 1 and 6 (variants + phases, real shape).
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

E, K, DIM, RS = 384, 6, 5120, 1.5
N_ROWS = 100_000


def _need():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA")
    import triton  # noqa: F401  (FAIL on the box if missing)
    from runtime import static_kernels as SK

    return SK


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _order(R_):
    from runtime import fused_glue_r2 as R2

    rec = R2.pick_order("cuda", K, R_, "possum")
    assert rec["exact"], rec
    return rec


def _ws(R_, E_=E, k=K, gemv=False):
    from runtime.router_plan import NET, _np2

    f = dict(dtype=torch.float32, device="cuda")
    ws = {"sc": torch.zeros(R_, E_, **f), "bsc": torch.zeros(R_, E_, **f),
          "tk": torch.zeros(R_, k, dtype=torch.long, device="cuda"),
          "net": torch.zeros(_np2(R_), NET, dtype=torch.long, device="cuda"),
          "ticket": torch.zeros(1, dtype=torch.int32, device="cuda")}
    if gemv:
        ws["raw"] = torch.zeros(R_, E_, **f)
    return ws


def _fused_tail(raw, bias, ordd):
    """route()/route_rows()'s ops after the mm on the device -> (ids, w, torch.topk's indices)."""
    from runtime.fused_glue_r2 import launch_router_post, launch_router_pre

    T_, E_ = raw.shape
    sp = F.softplus(raw).contiguous()
    sc = torch.empty(T_, E_, dtype=torch.float32, device="cuda")
    bsc = torch.empty(T_, E_, dtype=torch.float32, device="cuda")
    launch_router_pre(sp.view(-1), bias, sc.view(-1), bsc.view(-1))
    idx = bsc.topk(K, dim=-1)[1]
    ids = torch.empty(T_, K, dtype=torch.long, device="cuda")
    w = torch.empty(T_, K, dtype=torch.float32, device="cuda")
    if T_ == 1:
        launch_router_post(idx.view(-1), sc.view(-1), ids.view(-1), w.view(-1), ordd=ordd, eps=1e-20, rs=RS)
    else:
        launch_router_post(idx.contiguous(), sc, ids, w, ordd=ordd, eps=1e-20, rs=RS)
    return ids, w, idx


def _draw(g, R_, kind):
    from runtime.router_plan import RouterPlan

    i = {"normal": 0, "ties": 1, "equal": 2, "repeat": 3, "mixed": 4, "edges": 5}[kind]
    return RouterPlan.raw_draws(g, R_, E, i + 1)[i]


KINDS = ("normal", "ties", "mixed", "equal", "repeat", "edges")


# ------------------------------------------------------------------------------------------------ softplus
def test_softplus_probe_vs_torch():
    SK = _need()
    from runtime.router_plan import SOFTPLUS_FORMS, softplus_probe_inputs

    g = torch.Generator(device="cpu").manual_seed(0)
    parts = [softplus_probe_inputs(g, 1 << 22)]
    bits = torch.arange(0, 1 << 32, 256, dtype=torch.int64, device="cpu").to(torch.int32)   # every 256th fp32 bit pattern
    parts.append(bits.view(torch.float32))
    x = torch.cat(parts).cuda()
    ref = F.softplus(x)
    sub = torch.exp(x.double()) < 2.0 ** -126
    SOFTPLUS["exact"] = False
    SOFTPLUS["mode"] = 0
    for form in ("fused", "fused-sub", "fused-sub24"):
        got = SK.softplus_probe(x, SOFTPLUS_FORMS[form])
        bad = (got.view(torch.int32) != ref.view(torch.int32)) & ~(torch.isnan(got) & torch.isnan(ref))
        nb = int(bad.sum())
        print(f"  softplus form {form}: vs F.softplus on {x.numel():,} fp32 values: {nb} differ "
              f"({int((bad & ~sub).sum())} outside exp(x) < 2^-126)"
              + (f" (first: x={x[bad][:4].tolist()} got={got[bad][:4].tolist()} want={ref[bad][:4].tolist()})"
                 if nb else ""))
        if nb == 0 and not SOFTPLUS["exact"]:
            SOFTPLUS["exact"], SOFTPLUS["mode"] = True, SOFTPLUS_FORMS[form]
    print(f"  -> in-kernel softplus: {'mode %d' % SOFTPLUS['mode'] if SOFTPLUS['exact'] else 'none exact (torch op)'}")


SOFTPLUS = {"exact": None, "mode": 0}


# ------------------------------------------------------------------------------------------------ the tie gate
def test_route_and_topk_order_equal_torch_on_1e5_rows():
    SK = _need()
    from runtime.fused_glue_r2 import order_args

    if SOFTPLUS["exact"] is None:
        test_softplus_probe_vs_torch()
    g = torch.Generator(device="cpu").manual_seed(1)
    modes = [0] + ([SOFTPLUS["mode"]] if SOFTPLUS["exact"] else [])
    for R_ in (1, 6):
        ordd = _order(R_)
        a = order_args(ordd)
        ws = _ws(R_)
        ws1 = _ws(R_)
        n = 0
        bad_t = torch.zeros(4, dtype=torch.long, device="cuda")
        t0 = time.time()
        while n < N_ROWS // 2:
            kind = KINDS[(n // R_) % len(KINDS)]
            raw, bias = _draw(g, R_, kind)
            raw, bias = raw.cuda(), bias.cuda()
            want_ids, want_w, want_tk = _fused_tail(raw, bias, ordd)
            for mode in modes:
                ids = torch.empty(R_, K, dtype=torch.long, device="cuda")
                w = torch.empty(R_, K, dtype=torch.float32, device="cuda")
                src = raw if mode else F.softplus(raw)
                SK.router_rows_launch(src.contiguous(), bias, ws, ids, w, R=R_, E=E, K=K, order=a, rs=RS, eps=1e-20,
                                      softplus=mode)
                bad_t[0] += (ws["tk"] != want_tk).any(-1).sum()
                bad_t[1] += (ids != want_ids).any(-1).sum()
                bad_t[2] += (_bits(w) != _bits(want_w)).any(-1).sum()
                i1 = torch.empty(R_, K, dtype=torch.long, device="cuda")
                w1 = torch.empty(R_, K, dtype=torch.float32, device="cuda")
                SK.router_plan_launch(src.contiguous(), bias, ws1, i1, w1, R=R_, E=E, K=K, order=a, rs=RS,
                                      eps=1e-20, softplus=mode)
                bad_t[3] += ((i1 != ids).any(-1) | (_bits(w1) != _bits(w)).any(-1)).sum()
            n += R_
        bad = dict(zip(("tk", "ids", "w", "vs_1cta"), bad_t.tolist()))
        print(f"  R={R_}: {n:,} rows x softplus modes {modes} ({', '.join(KINDS)}): rows differing {bad} "
              f"[{time.time() - t0:.1f} s]")
        assert not any(bad.values()), bad


# ------------------------------------------------------------------------------------------------ the plan
def test_plan_tensors_equal_dedup_fused_on_device():
    SK = _need()
    from runtime import moe_rows as MR
    from runtime import router_plan as RP
    from runtime.fused_glue_r2 import order_args

    g = torch.Generator(device="cpu").manual_seed(2)
    R_ = 6
    ordd = _order(R_)
    P = R_ * K
    n_eq = 0
    for kind in (None, "mm", "v1", ("contiguous", False), ("contiguous", True), ("psum", False), ("psum", True),
                 ("masked", False), ("masked", True)):
        for t in range(24):
            raw, bias = _draw(g, R_, KINDS[t % len(KINDS)])
            raw, bias = raw.cuda(), bias.cuda()
            hot = torch.rand(E, generator=g, device="cpu", dtype=torch.float32) < [0.77, 0.0, 1.0, 0.5][t % 4]
            gtab = torch.full((E,), -1, dtype=torch.int32, device="cpu")
            gtab[hot] = torch.arange(int(hot.sum()), dtype=torch.int32, device="cpu")
            gtab = gtab.cuda()
            H = int(hot.sum())
            S = [16, 1, 3, 8][t % 4]
            BR = 128 if isinstance(kind, tuple) and kind[0] == "masked" else [6, 8, 16, 128][t % 4]
            outs = []
            for mine in (True, False):
                inpl = None
                mm = None
                if kind == "mm":
                    mm = torch.full((P,), -7, dtype=torch.int32, device="cuda")
                elif kind == "v1":
                    inpl = {"gtab": gtab, "H": H, "S": S, "dummy": torch.tensor(0, device="cuda"), "has_dummy": H > 0}
                elif kind is not None:
                    G = H + S
                    inpl = {"gtab": gtab, "H": H, "S": S, "dummy": torch.tensor(0, device="cuda"), "has_dummy": H > 0,
                            "v2": True, "layout": kind[0], "G": G, "gscr": G, "step_sf": kind[1],
                            "mi": torch.full((P * BR,), -1, dtype=torch.int32, device="cuda")
                            if kind[0] == "contiguous" else None,
                            "gl": torch.full((G,), -5, dtype=torch.int32, device="cuda"),
                            "ovf": torch.zeros(1, dtype=torch.int32, device="cuda")}
                    mm = torch.full((P,), -7, dtype=torch.int32, device="cuda")
                ids_s = torch.full((P,), -7, dtype=torch.long, device="cuda")
                uv = torch.full((P,), -7, dtype=torch.long, device="cuda")
                buf = MR.DedupBuffers(R_, K, "cuda")
                ws = _ws(R_)
                ids = torch.empty(R_, K, dtype=torch.long, device="cuda")
                w = torch.empty(R_, K, dtype=torch.float32, device="cuda")
                if mine:
                    a = RP.dedup_args(buf, R_, K, BR, mm, ids_s, uv, inpl, ids)
                    SK.router_rows_launch(F.softplus(raw), bias, ws, ids, w, R=R_, E=E, K=K, order=order_args(ordd),
                                          rs=RS, eps=1e-20, softplus=0, plan=a)
                    assert int(ws["ticket"]) == 0
                    m = RP.plan_out_dict(buf, mm, inpl, a)
                else:
                    ids, w, _ = _fused_tail(raw, bias, ordd)
                    m = MR.dedup_fused(buf, ids, w, BR, mm=mm, ids_step=ids_s, uval=uv, inpl=inpl, impl="warp")
                outs.append((m, mm, ids_s, uv, inpl))
            (m1, mm1, i1, u1, p1), (m0, mm0, i0, u0, p0) = outs
            assert sorted(m1) == sorted(m0)
            for key, v in m0.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(_bits(m1[key]), _bits(v)), (key, kind, t)
            for x_, y_ in ((mm1, mm0), (i1, i0), (u1, u0)):
                assert x_ is None or torch.equal(x_, y_), kind
            if p0 is not None and p0.get("v2"):
                for nme in ("mi", "gl", "ovf"):
                    assert p0[nme] is None or torch.equal(p1[nme], p0[nme]), (nme, kind, t)
            n_eq += 1
    print(f"  ticketed plan tensors == dedup_fused (m1_dedup_warp) on {n_eq} 6-row draws x 9 call shapes")


# ------------------------------------------------------------------------------------------------ plan-gemv
GATE_CFGS = [None, (1, 1, 10, 4, 1), (1, 2, 5, 4, 1), (2, 2, 10, 4, 1), (8, 4, 2, 8, 1), (1, 4, 10, 2, 1)]


def test_plan_gemv_equals_oracle_and_drift_vs_cublas():
    SK = _need()
    from runtime.fused_glue_r2 import fixed_dot_torch, order_args
    from runtime.router_plan import gate_kb

    kb = gate_kb(DIM)
    if SOFTPLUS["exact"] is None:
        test_softplus_probe_vs_torch()
    mode = SOFTPLUS["mode"] if SOFTPLUS["exact"] else 0
    g = torch.Generator(device="cpu").manual_seed(3)
    gw = (torch.randn(E, DIM, generator=g, device="cpu", dtype=torch.float32) * 0.02).to(torch.bfloat16).cuda()
    gw32 = gw.float()
    bias = (torch.randn(E, generator=g, device="cpu", dtype=torch.float32) * 0.05).cuda()
    stats = {"rows": 0, "raw_ne": 0, "raw_max_ulp": 0, "set_diff": 0, "max_dw": 0.0}
    for R_ in (1, 6):
        ordd = _order(R_)
        ws = _ws(R_, gemv=True)
        for t in range(200 if R_ == 6 else 400):
            x = (torch.randn(R_, DIM, generator=g, device="cpu", dtype=torch.float32) * 0.5).to(torch.bfloat16).float().cuda()
            oracle = fixed_dot_torch(x, gw, kb)
            for cfg in (GATE_CFGS if t < 4 else GATE_CFGS[:1]):      # every config bit-identical (speed-only knobs)
                ws["raw"].fill_(-1.0)
                SK.gate_dot(x, gw, ws["raw"], kb=kb, cfg=cfg)
                assert torch.equal(_bits(ws["raw"]), _bits(oracle)), f"gate_dot {cfg} != fixed_dot_torch"
            ids = torch.empty(R_, K, dtype=torch.long, device="cuda")
            w = torch.empty(R_, K, dtype=torch.float32, device="cuda")
            src = ws["raw"] if mode else F.softplus(ws["raw"])
            SK.router_rows_launch(src.contiguous(), bias, ws, ids, w, R=R_, E=E, K=K, order=order_args(ordd), rs=RS,
                                  eps=1e-20, softplus=mode)
            oi, ow, _ = _fused_tail(oracle, bias, ordd)
            assert torch.equal(ids, oi) and torch.equal(_bits(w), _bits(ow)), "plan-gemv != its oracle"
            cub = F.linear(x, gw32)
            ci, cw, _ = _fused_tail(cub, bias, ordd)
            d = (_bits(oracle).long() - _bits(cub).long()).abs()
            stats["rows"] += R_
            stats["raw_ne"] += int((d > 0).sum())
            stats["raw_max_ulp"] = max(stats["raw_max_ulp"], int(d.max()))
            stats["set_diff"] += int((torch.sort(ids, -1)[0] != torch.sort(ci, -1)[0]).any(-1).sum())
            if torch.equal(ids, ci):
                stats["max_dw"] = max(stats["max_dw"], float((w - cw).abs().max()))
    print(f"  plan-gemv == its oracle (fixed order KB={kb}, {len(GATE_CFGS)} gate configs) on {stats['rows']} rows; "
          f"vs cuBLAS scores: {stats['raw_ne']} of {stats['rows'] * E} raw values differ (max {stats['raw_max_ulp']} "
          f"ulp), {stats['set_diff']} rows route a different expert set, max |dw| {stats['max_dw']:.3e} (A3, reported)")


# ------------------------------------------------------------------------------------------------ graphs
def test_graph_capture_replay_tickets():
    SK = _need()
    from runtime import moe_rows as MR
    from runtime import router_plan as RP
    from runtime.fused_glue_r2 import order_args

    if SOFTPLUS["exact"] is None:
        test_softplus_probe_vs_torch()
    mode = SOFTPLUS["mode"] if SOFTPLUS["exact"] else 0
    g = torch.Generator(device="cpu").manual_seed(4)
    R_ = 6
    ordd = order_args(_order(R_))
    L = 40
    kb = RP.gate_kb(DIM)
    gws = [(torch.randn(E, DIM, generator=g, device="cpu", dtype=torch.float32) * 0.02).to(torch.bfloat16).cuda() for _ in range(4)]
    gw32 = [w_.float() for w_ in gws]
    bias = (torch.randn(E, generator=g, device="cpu", dtype=torch.float32) * 0.05).cuda()
    x = (torch.randn(R_, DIM, generator=g, device="cpu", dtype=torch.float32) * 0.5).to(torch.bfloat16).float().cuda()
    for impl in ("plan", "plan-gemv"):
        wss = [_ws(R_, gemv=True) for _ in range(L)]
        outs = [(torch.empty(R_, K, dtype=torch.long, device="cuda"), torch.empty(R_, K, dtype=torch.float32, device="cuda")) for _ in range(L)]
        bufs = [MR.DedupBuffers(R_, K, "cuda") for _ in range(L)]
        args = [RP.dedup_args(bufs[i], R_, K, 8, None, None, None, None, outs[i][0]) for i in range(L)]

        def chain():
            for i in range(L):
                if impl == "plan-gemv":
                    raw = wss[i]["raw"]
                    SK.gate_dot(x, gws[i % 4], raw, kb=kb)
                else:
                    raw = F.linear(x, gw32[i % 4])
                if not mode:
                    raw = F.softplus(raw)
                SK.router_rows_launch(raw, bias, wss[i], outs[i][0], outs[i][1], R=R_, E=E, K=K, order=ordd, rs=RS,
                                      eps=1e-20, softplus=mode, plan=args[i])

        def tensors():
            return [t for o in outs for t in o] + [t for b in bufs for t in (b.uniq, b.counts, b.slot, b.rank, b.prow,
                                                                             b.pw, b.arow, b.posmat)]
        chain()
        torch.cuda.synchronize()
        eager = [t.clone() for t in tensors()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            chain()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain()
        for rep in range(3):
            for t in tensors():
                t.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(tensors(), eager)), (impl, rep)
            assert all(int(w_["ticket"]) == 0 for w_ in wss)
        print(f"  graph of {L} {impl} layers (router-rows + ticketed plan): 3 replays == eager, tickets 0")


# ------------------------------------------------------------------------------------------------ mini checkpoints
def test_mini_static_plan_equals_fused_1_and_6_rows():
    _need()
    from test_static_rows_gpu import _eq_runs, _model, _need_step, _need_tilelang, _static, _toks

    _need_tilelang()
    from runtime import router_plan as RP

    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        for rows in (6, 1):
            for L in (17, 18):
                ids, toks = _toks(L, rows * 4, 70 + L)
                runs = {}
                for impl, form in (("plan", "ticket"), ("plan", "split"), ("plan-gemv", "ticket"), ("fused", None)):
                    if rows == 1 and form == "split":
                        continue
                    name = impl if form in (None, "ticket") else f"{impl}_{form}"
                    os.environ[RP.ROUTER_IMPL_ENV] = impl
                    if form:
                        os.environ[RP.PLAN_LAUNCH_ENV] = form
                    try:
                        kw = dict(rows=rows, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
                        if rows > 1:
                            kw.update(hidden_layers=hid, spare_slots=8)
                        st = _need_step(model, pool, **kw)
                        if impl != "fused":
                            d = st.glue.rplan.describe()
                            print(f"  rows {rows} P0 {L}: {name} {d['status']}, softplus {d['softplus']}, rows_plan "
                                  f"{d['rows_plan']}, launches {d['launches_per_layer']}, checks {d['checks']}")
                            assert st.glue.rplan.verified and d["status"] == "on", d
                            assert rows == 1 or d["rows_plan"] == "on", d
                        a = _static(model, st, ids, toks, 4)
                        st.capture()
                        ag = _static(model, st, ids, toks, 4, graph=True)
                        _eq_runs(ag, a, f"{name} rows {rows}: graph vs static-eager")
                        runs[name] = a
                        st.detach()
                    finally:
                        os.environ.pop(RP.ROUTER_IMPL_ENV, None)
                        os.environ.pop(RP.PLAN_LAUNCH_ENV, None)
                for name in runs:
                    if name.startswith("plan") and not name.startswith("plan-gemv"):
                        _eq_runs(runs[name], runs["fused"], f"rows {rows} P0 {L}: {name} vs fused")
                print(f"  rows {rows} P0 {L}: plan (both launch forms) == fused (logits, main_hidden, caches); "
                      f"plan-gemv self-checked; graph == eager for all")
    RP.uninstall_rows_hook()


# ------------------------------------------------------------------------------------------------ recorded inputs
def test_mini_recorded_router_inputs_replay_gate():
    """The recording hook on the mini4 checkpoint (fused path, static-eager, 1 and 6 rows, both parities) -> the
    replay gate of bench/router_plan_bench.py on what was recorded: the plan kernel == the fused tail == the live
    outputs on every recorded row, topk order included; 6-row plan tensors == dedup_fused. (The real checkpoint's
    10^5 rows: the bench's --replay on a first_light --mode static dump, module docstring there.)"""
    _need()
    import tempfile

    from test_static_rows_gpu import _model, _need_step, _need_tilelang, _static, _toks

    _need_tilelang()
    sys.path.insert(0, os.path.join(REPO, "bench"))
    from router_plan_bench import replay_gate
    from runtime import router_plan as RP

    if SOFTPLUS["exact"] is None:
        test_softplus_probe_vs_torch()
    root = tempfile.mkdtemp(prefix="router-dump-", dir=os.path.join(REPO, "scratch") if os.path.isdir(
        os.path.join(REPO, "scratch")) else None)
    os.environ.pop(RP.ROUTER_IMPL_ENV, None)
    os.environ[RP.DUMP_ENV] = root
    RP._REC["r"] = None
    try:
        for kind in ("mini4",):
            with _model(kind) as (model, pool):
                hid = (len(model.layers) - 1,)
                for rows in (1, 6):
                    for L in (17, 18):
                        ids, toks = _toks(L, rows * 6, 90 + L + rows)
                        kw = dict(rows=rows, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
                        if rows > 1:
                            kw.update(hidden_layers=hid, spare_slots=8)
                        st = _need_step(model, pool, **kw)
                        _static(model, st, ids, toks, 6)
                        st.detach()
        RP.recorder().flush()
    finally:
        os.environ.pop(RP.DUMP_ENV, None)
        RP._REC["r"] = None
    res = replay_gate(root, SOFTPLUS["mode"] if SOFTPLUS["exact"] else 0)
    print(f"  recorded mini router inputs ({root}): {res}")
    assert res["rows"] > 0 and res["plan_records"] > 0, res
    assert not any(res["rows_differing"].values()) and res["plan_records_differing"] == 0, res


# ------------------------------------------------------------------------------------------------ timing
def _time_graph(fn, n_rep=50):
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
    for _ in range(n_rep):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / n_rep


def test_timing_us_per_layer():
    _need()
    sys.path.insert(0, os.path.join(REPO, "bench"))
    from router_plan_bench import time_impls

    if SOFTPLUS["exact"] is None:
        test_softplus_probe_vs_torch()
    for R_ in (1, 6):
        res = time_impls(R_, layers=40, sp_mode=SOFTPLUS["mode"] or 1, sp_exact=bool(SOFTPLUS["exact"]))
        print(f"  R={R_}: us per layer " + ", ".join(f"{k} {v:.2f}" for k, v in res["us_per_layer"].items()))
        print(f"        phases " + ", ".join(f"{k} {v:.2f}" for k, v in res["phases_us"].items()))
        print(f"        self-replay {res['self_replay']}, equal {res['cross_equal']}")
        assert all(v is not False for v in res["self_replay"].values()), res["self_replay"]
        assert all(res["cross_equal"].values()), res["cross_equal"]


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q", "-s"]))
