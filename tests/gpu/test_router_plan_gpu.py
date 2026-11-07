"""GPU tests of M2-T15 (docs/HANDOFF-M2-T15-ROUTER-PLAN.md; runtime/router_plan.py, runtime/static_kernels.py
m2_router_plan). Skipped without CUDA (Mac); on the box a missing triton FAILS. Same docker line as
tests/gpu/test_static_rows_gpu.py with this file:

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_router_plan_gpu.py

  * softplus: the kernel's libdevice form vs F.softplus on 2^22 random values + a dense grid around the threshold +
    every 2^8-th fp32 bit pattern (16.7M) + edges -> mismatch count printed (0 = the `fused` softplus is exact;
    otherwise the self-check keeps F.softplus as a torch op: asserted);
  * THE tie gate: at the real shapes (E = 384, k = 6) over 10^5 router rows at R = 1 and R = 6 -- realistic scores,
    tie-heavy (17 raw values x a two-valued bias), "mixed" (distinct values AND ties AND ties straddling the 6th
    place), all-equal rows, repeated rows, +-0, beyond-threshold and underflowing scores: the kernel's topk order
    (`tk`) == torch.topk(bsc, 6).indices ELEMENT FOR ELEMENT (order included), and (ids, w) == the fused path (the
    mm-free tail: F.softplus, t9b2r2_router_pre, torch.topk, t9b2r2_router_post) bit for bit, in both softplus forms
    (the libdevice one only if the probe was exact);
  * the R-row plan == moe_rows.dedup_fused (m1_dedup_warp on the device) on every buffer for every call shape (no
    inpl / masked_m / inpl v1 / v2 contiguous / psum / masked x step SF), incl. overflow;
  * plan-gemv: == its oracle (fused_glue_r2.fixed_dot_torch scores + the fused tail) bit for bit; vs cuBLAS's scores
    (the fused path): raw ulp histogram, rows whose id SET differs, max |dw| -> printed (A3 class, not gated);
  * graph capture: 40 plan / plan-gemv launches (one per "layer", own tickets) captured once, replayed 3x == eager,
    tickets back to 0;
  * mini checkpoints: StaticStep with $DSV41_ROUTER_IMPL=plan (self-check passes: printed) == the fused step, torch.equal
    on logits / main_hidden / caches, 1 row and 6 rows (dg-inplace, both parities), graph == static-eager;
  * recorded inputs: the dump hook on mini4's fused static-eager steps (1 and 6 rows) -> the bench's replay gate (plan ==
    fused tail == live outputs, topk order included; plan tensors == dedup_fused), 0 rows differing asserted;
  * timing (informative): us per layer at R = 1 and 6 for fused / plan / plan-gemv, CUDA graph of 40 layer-calls.
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
    from runtime.router_plan import softplus_probe_inputs

    g = torch.Generator(device="cpu").manual_seed(0)
    parts = [softplus_probe_inputs(g, 1 << 22)]
    bits = torch.arange(0, 1 << 32, 256, dtype=torch.int64, device="cpu").to(torch.int32)   # every 256th fp32 bit pattern
    parts.append(bits.view(torch.float32))
    x = torch.cat(parts).cuda()
    got = SK.softplus_probe(x)
    ref = F.softplus(x)
    bad = (got.view(torch.int32) != ref.view(torch.int32)) & ~(torch.isnan(got) & torch.isnan(ref))
    nb = int(bad.sum())
    print(f"  softplus: libdevice log1p(exp) vs F.softplus on {x.numel():,} fp32 values: {nb} differ"
          + (f" (first: x={x[bad][:4].tolist()} got={got[bad][:4].tolist()} want={ref[bad][:4].tolist()})" if nb else ""))
    SOFTPLUS["exact"] = nb == 0


SOFTPLUS = {"exact": None}


# ------------------------------------------------------------------------------------------------ the tie gate
def test_route_and_topk_order_equal_torch_on_1e5_rows():
    SK = _need()
    from runtime.fused_glue_r2 import order_args

    if SOFTPLUS["exact"] is None:
        test_softplus_probe_vs_torch()
    g = torch.Generator(device="cpu").manual_seed(1)
    forms = [False] + ([True] if SOFTPLUS["exact"] else [])
    for R_ in (1, 6):
        ordd = _order(R_)
        a = order_args(ordd)
        ws = _ws(R_)
        n = 0
        bad_t = torch.zeros(3, dtype=torch.long, device="cuda")
        t0 = time.time()
        while n < N_ROWS // 2:
            kind = KINDS[(n // R_) % len(KINDS)]
            raw, bias = _draw(g, R_, kind)
            raw, bias = raw.cuda(), bias.cuda()
            want_ids, want_w, want_tk = _fused_tail(raw, bias, ordd)
            for sp_in_kernel in forms:
                ids = torch.empty(R_, K, dtype=torch.long, device="cuda")
                w = torch.empty(R_, K, dtype=torch.float32, device="cuda")
                src = raw if sp_in_kernel else F.softplus(raw)
                SK.router_plan_launch(src.contiguous(), bias, ws, ids, w, R=R_, E=E, K=K, order=a, rs=RS, eps=1e-20,
                                      softplus=sp_in_kernel)
                bad_t[0] += (ws["tk"] != want_tk).any(-1).sum()
                bad_t[1] += (ids != want_ids).any(-1).sum()
                bad_t[2] += (_bits(w) != _bits(want_w)).any(-1).sum()
            n += R_
        bad = dict(zip(("tk", "ids", "w"), bad_t.tolist()))
        print(f"  R={R_}: {n:,} rows x softplus forms {forms} ({', '.join(KINDS)}): rows differing {bad} "
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
                    SK.router_plan_launch(F.softplus(raw), bias, ws, ids, w, R=R_, E=E, K=K, order=order_args(ordd),
                                          rs=RS, eps=1e-20, softplus=False, plan=a)
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
    print(f"  plan tensors == dedup_fused (m1_dedup_warp) on {n_eq} 6-row draws x 9 call shapes")


# ------------------------------------------------------------------------------------------------ plan-gemv
def test_plan_gemv_equals_oracle_and_drift_vs_cublas():
    SK = _need()
    from runtime.fused_glue_r2 import fixed_dot_torch, order_args
    from runtime.router_plan import GATE_KB

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
            ids = torch.empty(R_, K, dtype=torch.long, device="cuda")
            w = torch.empty(R_, K, dtype=torch.float32, device="cuda")
            SK.router_plan_launch(ws["raw"], bias, ws, ids, w, R=R_, E=E, K=K, order=order_args(ordd), rs=RS,
                                  eps=1e-20, softplus=bool(SOFTPLUS["exact"]), x=x, gw=gw, kb=GATE_KB)
            oracle = fixed_dot_torch(x, gw, GATE_KB)
            assert torch.equal(_bits(ws["raw"]), _bits(oracle)), "GEMV scores != fixed_dot_torch"
            if SOFTPLUS["exact"]:
                oi, ow, _ = _fused_tail(oracle, bias, ordd)
                assert torch.equal(ids, oi) and torch.equal(_bits(w), _bits(ow)), "plan-gemv != its oracle"
            assert int(ws["ticket"]) == 0
            cub = F.linear(x, gw32)
            ci, cw, _ = _fused_tail(cub, bias, ordd)
            d = (_bits(oracle).long() - _bits(cub).long()).abs()
            stats["rows"] += R_
            stats["raw_ne"] += int((d > 0).sum())
            stats["raw_max_ulp"] = max(stats["raw_max_ulp"], int(d.max()))
            stats["set_diff"] += int((torch.sort(ids, -1)[0] != torch.sort(ci, -1)[0]).any(-1).sum())
            if torch.equal(ids, ci):
                stats["max_dw"] = max(stats["max_dw"], float((w - cw).abs().max()))
    print(f"  plan-gemv == its oracle (fixed order KB={GATE_KB}) on {stats['rows']} rows; vs cuBLAS scores: "
          f"{stats['raw_ne']} of {stats['rows'] * E} raw values differ (max {stats['raw_max_ulp']} ulp), "
          f"{stats['set_diff']} rows route a different expert set, max |dw| {stats['max_dw']:.3e} (A3, reported)")


# ------------------------------------------------------------------------------------------------ graphs
def test_graph_capture_replay_tickets():
    SK = _need()
    from runtime.fused_glue_r2 import order_args
    from runtime.router_plan import GATE_KB

    g = torch.Generator(device="cpu").manual_seed(4)
    R_ = 6
    ordd = order_args(_order(R_))
    L = 40
    gws = [(torch.randn(E, DIM, generator=g, device="cpu", dtype=torch.float32) * 0.02).to(torch.bfloat16).cuda() for _ in range(4)]
    bias = (torch.randn(E, generator=g, device="cpu", dtype=torch.float32) * 0.05).cuda()
    x = (torch.randn(R_, DIM, generator=g, device="cpu", dtype=torch.float32) * 0.5).to(torch.bfloat16).float().cuda()
    wss = [_ws(R_, gemv=True) for _ in range(L)]
    outs = [(torch.empty(R_, K, dtype=torch.long, device="cuda"), torch.empty(R_, K, dtype=torch.float32, device="cuda")) for _ in range(L)]

    def chain():
        for i in range(L):
            SK.router_plan_launch(wss[i]["raw"], bias, wss[i], outs[i][0], outs[i][1], R=R_, E=E, K=K, order=ordd,
                                  rs=RS, eps=1e-20, softplus=True, x=x, gw=gws[i % 4], kb=GATE_KB)
    chain()
    torch.cuda.synchronize()
    eager = [(a.clone(), b.clone()) for a, b in outs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    for rep in range(3):
        for a, b in outs:
            a.fill_(-1)
            b.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, ea) and torch.equal(_bits(b), _bits(eb)) for (a, b), (ea, eb) in zip(outs, eager))
        assert all(int(w_["ticket"]) == 0 for w_ in wss)
    print(f"  graph of {L} plan-gemv launches: 3 replays == eager, tickets 0")


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
                for impl in ("plan", "fused"):
                    os.environ[RP.ROUTER_IMPL_ENV] = impl
                    try:
                        kw = dict(rows=rows, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
                        if rows > 1:
                            kw.update(hidden_layers=hid, spare_slots=8)
                        st = _need_step(model, pool, **kw)
                        if impl == "plan":
                            d = st.glue.rplan.describe()
                            print(f"  rows {rows} P0 {L}: plan {d['status']}, softplus {d['softplus']}, rows_plan "
                                  f"{d['rows_plan']}, checks {d['checks']}")
                            assert st.glue.rplan.verified and d["status"] == "on", d
                            assert rows == 1 or d["rows_plan"] == "on", d
                        a = _static(model, st, ids, toks, 4)
                        st.capture()
                        ag = _static(model, st, ids, toks, 4, graph=True)
                        _eq_runs(ag, a, f"{impl} rows {rows}: graph vs static-eager")
                        runs[impl] = a
                        st.detach()
                    finally:
                        os.environ.pop(RP.ROUTER_IMPL_ENV, None)
                _eq_runs(runs["plan"], runs["fused"], f"rows {rows} P0 {L}: plan vs fused")
                print(f"  rows {rows} P0 {L}: plan == fused (logits, main_hidden, caches), graph == eager")
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
    res = replay_gate(root, bool(SOFTPLUS["exact"]))
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

    for R_ in (1, 6):
        res = time_impls(R_, layers=40)
        print(f"  R={R_}: us per layer " + ", ".join(f"{k} {v:.2f}" for k, v in res.items()))


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q", "-s"]))
