"""GPU tests of M2-L3 (runtime/dg_mega.py, `moe_impl="dg-mega"`). Skipped without CUDA/tilelang (Mac); on the box a
missing/broken deep_gemm FAILS (it is what is under test).

  * l3_route_mega on the device == mega_route (the torch mirror), R = 1 / 6, with and without overflow;
  * l3_stage_mega on the device == stage_mega_emulated, cold sources in pinned host memory (UVA), both SF storages;
  * our L1 / SF builders == deep_gemm's own transform_weights_for_mega_moe on the same bytes (both SF storages);
  * mini checkpoints, MoE level (the hard gates): every layer's dg-mega output finite; R = 6 row r == row r alone
    (row-invariant, bitwise, every r); eager replay equal; CUDA-graph replay == eager; the drift report: dg-mega vs
    dg-inplace (mismatch %, max ulp) and both vs an fp64 oracle of the reference's rounding points (mismatch %,
    max ulp, relative L2) -- GATE: dg-mega's relative L2 to the oracle <= dg-inplace's (A3: "fp64-oracle distance
    <= today's"); kernel launches per MoE call (torch.profiler) and graph-replayed us per MoE call, dg-mega vs
    dg-inplace at 1 and 6 rows (informative on the mini; the box's real-shape numbers are bench/dg_mega_spike.py's);
  * mini end to end (Generator, static + graph): graph == static-eager, replay #2 equal; top-1 stream vs dg-inplace
    printed with the first divergence's reference margin (A3: printed, not asserted).

`dg-mega` is not yet in static_step.MOE_IMPLS / generate.MOE_IMPLS (files outside M2-L3's ownership): the tests add it
for their own duration. Run on the box (orchestrator) exactly as tests/gpu/test_dg_inplace_gpu.py:
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_dg_mega_gpu.py
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_dg_moe_gpu import _finite, _gen, _need_dg  # noqa: E402
from test_static_mini_gpu import MAX_POS, N_STEPS, _assert_equal_runs, _ids, _model, _run  # noqa: E402

MEGA, DGI = "dg-mega", "dg-inplace"


@contextlib.contextmanager
def _mega_enabled(spare=None):
    """dg-mega in the impl lists for the duration (+ $DSV41_MEGA_SPARE)."""
    from runtime import generate, static_step

    saved = (static_step.MOE_IMPLS, static_step.MOE_IMPLS_ROWS, generate.MOE_IMPLS, os.environ.get("DSV41_MEGA_SPARE"))
    static_step.MOE_IMPLS = tuple(saved[0]) + (MEGA,)
    static_step.MOE_IMPLS_ROWS = tuple(saved[1]) + (MEGA,)
    generate.MOE_IMPLS = tuple(saved[2]) + (MEGA,)
    if spare is not None:
        os.environ["DSV41_MEGA_SPARE"] = str(spare)
    try:
        yield
    finally:
        static_step.MOE_IMPLS, static_step.MOE_IMPLS_ROWS, generate.MOE_IMPLS = saved[:3]
        if saved[3] is None:
            os.environ.pop("DSV41_MEGA_SPARE", None)
        else:
            os.environ["DSV41_MEGA_SPARE"] = saved[3]


def _dg():
    from runtime.dg_moe import import_deep_gemm

    return import_deep_gemm()


def _g(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


# ------------------------------------------------------------------------------------------------ kernels
def test_route_kernel_on_device_equals_mirror():
    _need_dg()
    from runtime import dg_mega as DM

    rng = random.Random(3)
    E, H, k, T = 384, 295, 6, 6
    hot = rng.sample(range(E), H)
    gt = torch.full((E,), -1, dtype=torch.int32, device="cpu")
    for s_, e in enumerate(hot):
        gt[e] = s_
    gtab = gt.cuda()
    for R in (1, 6):
        for S in (2, 6, 16):
            ids = torch.tensor([sorted(rng.sample(range(E), k)) for _ in range(R)], device="cpu").cuda()
            w = torch.rand(R, k, generator=_g(R), device="cpu").cuda()
            topk = torch.full((T, k), -7, dtype=torch.int64, device="cuda")
            tw = torch.zeros(T, k, dtype=torch.float32, device="cuda")
            stage = torch.full((S,), -9, dtype=torch.int64, device="cuda")
            ovf = torch.zeros(1, dtype=torch.int32, device="cuda")
            DM.route_mega(ids, w, gtab, topk, tw, stage, H, S, E, ovf)
            m = DM.mega_route(ids, gtab, H, S, E)
            torch.cuda.synchronize()
            assert torch.equal(topk[:R], m["topk"]) and bool((topk[R:] == -7).all()), (R, S)
            assert torch.equal(tw[:R], w) and torch.equal(stage, m["stage"]) and int(ovf) == int(m["n_over"]), (R, S)
    print("  l3_route_mega == mega_route on the device (R 1/6, S 2/6/16)")


def test_stage_kernel_on_device_equals_contract():
    _need_dg()
    from runtime import dg_mega as DM

    for major in DM.SF_MAJORS:
        for I, D in ((256, 256), (2304, 5120)):
            g = _g(7)
            E, H, S = 12, 7, 3
            KS1, KS2 = D // 32, I // 32
            src_w = {(p, e): torch.randint(-128, 127, ((I, D // 2) if p < 2 else (D, I // 2)), generator=g,
                                           device="cpu").to(torch.int8).pin_memory() for p in range(3) for e in range(E)}
            src_s = {(p, e): torch.randint(1, 255, ((I, KS1) if p < 2 else (D, KS2)), generator=g, device="cpu")
                     .to(torch.uint8).pin_memory() for p in range(3) for e in range(E)}
            ptab6 = torch.tensor([[src_w[(p, e)].data_ptr() for e in range(E)] for p in range(3)]
                                 + [[src_s[(p, e)].data_ptr() for e in range(E)] for p in range(3)],
                                 dtype=torch.int64, device="cpu").cuda()
            reg = {t.data_ptr(): t.reshape(-1).view(torch.uint8) for t in list(src_w.values()) + list(src_s.values())}
            G = H + S
            outs = []
            for kernel in (False, True):
                gg = _g(9)
                l1 = torch.randint(-128, 127, (G, 2 * I, D // 2), generator=gg, device="cpu").to(torch.int8).cuda()
                l2 = torch.randint(-128, 127, (G, D, I // 2), generator=gg, device="cpu").to(torch.int8).cuda()
                c = [torch.randint(1, 255, (G, n, ks), generator=gg, device="cpu").to(torch.uint8).cuda()
                     for n, ks in ((I, KS1), (I, KS1), (D, KS2))]
                dsts = [l1, l2, DM.mega_l1_sf(c[0], c[1], major), DM.mega_l2_sf(c[2], major)]
                stage = torch.tensor([9, -1, 11], dtype=torch.int64, device="cpu").cuda()
                if kernel:
                    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cpu").cuda()
                    DM.stage_mega(stage, ptab6, dtab, dsts, H, major)
                else:
                    DM.stage_mega_emulated(lambda a: reg[a].cuda(), stage.cpu(), ptab6.cpu(), dsts, H, major)
                torch.cuda.synchronize()
                outs.append([t.clone() for t in dsts])
            for i, (a, b) in enumerate(zip(*outs)):
                assert torch.equal(a, b), (major, I, D, i)
    print("  l3_stage_mega == stage_mega_emulated (pinned host sources; mini and real shapes; both SF storages)")


def test_layouts_equal_library_transform():
    _need_dg()
    from runtime import dg_mega as DM

    dg = _dg()
    g = _g(1)
    G, I, D = 3, 256, 512
    w1, w3 = (torch.randint(-128, 127, (G, I, D // 2), generator=g, device="cpu").to(torch.int8).cuda() for _ in "ab")
    w2 = torch.randint(-128, 127, (G, D, I // 2), generator=g, device="cpu").to(torch.int8).cuda()
    c1, c3 = (torch.randint(1, 255, (G, I, D // 32), generator=g, device="cpu").to(torch.uint8).cuda() for _ in "ab")
    c2 = torch.randint(1, 255, (G, D, I // 32), generator=g, device="cpu").to(torch.uint8).cuda()
    res = {}
    for major in DM.SF_MAJORS:
        sf13 = DM.to_major(torch.cat([DM.sf_words(c1), DM.sf_words(c3)], 1), major)
        sf2 = DM.to_major(DM.sf_words(c2), major)
        (tw1, tsf1), (tw2, tsf2) = dg.transform_weights_for_mega_moe((torch.cat([w1, w3], 1), sf13), (w2, sf2))
        ok = (torch.equal(tw1, DM.mega_l1_weight(w1, w3)) and torch.equal(tsf1, DM.mega_l1_sf(c1, c3, major))
              and torch.equal(tw2, w2) and torch.equal(tsf2, DM.mega_l2_sf(c2, major)))
        res[major] = (ok, tuple(tsf1.stride()), tuple(DM.mega_l1_sf(c1, c3, major).stride()))
    print(f"  ours vs transform_weights_for_mega_moe: {res}")
    assert all(v[0] for v in res.values()), res


# ------------------------------------------------------------------------------------------------ mini, MoE level
def _oracle_layer(moe, x, lim):
    """fp64 oracle (reference rounding points) of the MoE on x [R, D]: exact fp64 totals."""
    from dg_mega_spike import act_quant, deq_fp4, deq_fp8, oracle_expert
    from runtime import dg_mega as DM

    q, s = act_quant(x)
    a64 = (q.double().view(x.size(0), -1, 32) * s.double().view(x.size(0), -1, 1)).view(x.size(0), -1)
    weights, indices = moe.gate(x, None)
    ids, order = torch.sort(indices, dim=-1)
    w = torch.gather(weights, -1, order)
    sh = moe.shared_experts
    sw = [deq_fp8(getattr(sh, p).weight, DM.block_codes_per_row(getattr(sh, p).scale, getattr(sh, p).weight.size(0)))
          for p in ("w1", "w3", "w2")]
    out = []
    for r in range(x.size(0)):
        acc = torch.zeros(x.size(1), dtype=torch.float64, device=x.device)
        for j in range(ids.size(1)):
            ex = moe.experts[int(ids[r, j])]
            dq = [deq_fp4(getattr(ex, p).weight.view(torch.int8), DM._u8(getattr(ex, p).scale)) for p in
                  ("w1", "w3", "w2")]
            acc += oracle_expert(a64[r], dq[0], dq[1], dq[2], float(w[r, j]), lim).double()
        acc += oracle_expert(a64[r], sw[0], sw[1], sw[2], 1.0, lim).double()
        out.append(acc)
    return torch.stack(out)


def _launches(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def _mini_moe_level(kind):
    _need_dg()
    from dg_contiguous_spike import graph_us
    from dg_mega_spike import oracle_dist
    from runtime import dg_inplace as DI
    from runtime import dg_mega as DM
    from runtime.static_step import StaticStep

    with _model(kind) as (model, tokdir), _mega_enabled(spare=16):
        pool = model.runtime_load.pool
        lim = float(model.layers[0].ffn.shared_experts.swiglu_limit)
        D = model.layers[0].ffn.dim
        stm = StaticStep(model, pool, max_pos=MAX_POS, moe_impl=MEGA)
        print(f"  {kind}: dg-mega build {stm.dg.describe()}")
        sti = StaticStep(model, pool, max_pos=MAX_POS, moe_impl=DGI)
        st6 = StaticStep(model, pool, max_pos=MAX_POS, rows=6, moe_impl=DGI)
        g = _g(17)
        gate_ok = []
        with torch.inference_mode():
            for l in range(len(model.layers)):
                x6 = (torch.randn(6, D, generator=g, device="cpu") * 2).to(torch.bfloat16).cuda()
                with stm._active():
                    sm = stm.moes[l]
                    y6 = sm(x6.clone()).clone()
                    y6b = sm(x6.clone()).clone()
                    y1 = [sm(x6[r:r + 1].clone()).clone() for r in range(6)]
                    xin = x6.clone()
                    gm = torch.cuda.CUDAGraph()
                    sm(xin)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(gm):
                        yg = sm(xin)
                    gm.replay()
                    torch.cuda.synchronize()
                    yg = yg.clone()
                    us_m = {R: graph_us(lambda R=R: sm(x6[:R]), iters=100)["us"] for R in (1, 6)}
                    n_m = {R: _launches(lambda R=R: sm(x6[:R])) for R in (1, 6)}
                assert bool(torch.isfinite(y6.float()).all()), (kind, l)
                inv = [bool(torch.equal(y1[r].view(torch.int16), y6[r:r + 1].view(torch.int16))) for r in range(6)]
                assert all(inv), (kind, l, "dg-mega R=6 row vs R=1 alone", inv)
                assert torch.equal(y6.view(torch.int16), y6b.view(torch.int16)), (kind, l, "eager replay")
                assert torch.equal(yg.view(torch.int16), y6.view(torch.int16)), (kind, l, "graph vs eager")
                with sti._active():
                    si = sti.moes[l]
                    yi = torch.cat([si(x6[r:r + 1].clone()).clone() for r in range(6)])
                    us_i1 = graph_us(lambda: si(x6[:1]), iters=100)["us"]
                    n_i1 = _launches(lambda: si(x6[:1]))
                with st6._active():
                    s6 = st6.moes[l]
                    us_i6 = graph_us(lambda: s6(x6), iters=100)["us"]
                    n_i6 = _launches(lambda: s6(x6))
                ex = _oracle_layer(sm.moe, x6, lim)
                dm, di = oracle_dist(y6, ex), oracle_dist(yi, ex)
                gate_ok.append(dm["rel_l2"] <= di["rel_l2"])
                c = DI.compare_bf16(y6.view(-1), yi.view(-1))
                print(f"  {kind} L{l}: dg-mega vs dg-inplace: equal {c['equal']}, mismatch {c['mismatch_frac']*100:.3f} %"
                      f", max ulp {c['max_ulp']}; vs fp64 oracle: mega mism {dm['mismatch_frac']*100:.3f} % ulp "
                      f"{dm['max_ulp']} relL2 {dm['rel_l2']:.3e} | inplace mism {di['mismatch_frac']*100:.3f} % ulp "
                      f"{di['max_ulp']} relL2 {di['rel_l2']:.3e}; launches/MoE call R1 {n_i1} -> {n_m[1]}, R6 {n_i6} "
                      f"-> {n_m[6]}; us/MoE call (graph) R1 {us_i1} -> {us_m[1]}, R6 {us_i6} -> {us_m[6]}; "
                      f"row-invariant {inv}")
        for st in (stm, sti, st6):
            st.detach()
        assert all(gate_ok), (kind, "GATE fp64-oracle relative L2: dg-mega <= dg-inplace per layer", gate_ok)


def test_mini_moe_level_gates_mini():
    _mini_moe_level("mini")


def test_mini_moe_level_gates_mini4():
    _mini_moe_level("mini4")


# ------------------------------------------------------------------------------------------------ mini end to end
def _mini_end_to_end(kind):
    _need_dg()
    with _model(kind) as (model, tokdir), _mega_enabled():
        gen = _gen(model, tokdir, moe_impl=DGI)
        ids = _ids(16, 5)
        ref = _run(gen, "static", ids)
        gen.set_moe_impl(MEGA)
        st = _run(gen, "static", ids)
        _finite(st, f"{kind} dg-mega static")
        g1 = _run(gen, "graph", ids)
        _assert_equal_runs(g1, st, f"{kind}: dg-mega graph vs static-eager")
        g2 = _run(gen, "graph", ids)
        _assert_equal_runs(g2, g1, f"{kind}: dg-mega graph replay #2 vs #1")
        first = next((i for i, (a, b) in enumerate(zip(ref[1], st[1])) if a != b), None)
        margin = None
        if first is not None:
            top2 = torch.topk(ref[0][first].reshape(-1).float(), 2).values
            margin = float(top2[0] - top2[1])
        dl = [float((a - b).abs().max()) for a, b in zip(st[0], ref[0])]
        print(f"  {kind}: dg-mega graph == static-eager, replay equal; top-1 stream vs dg-inplace "
              f"{'EQUAL' if first is None else f'first differs at position {first} (dg-inplace margin {margin:.4f})'}"
              f"; max |dlogit| per position {[round(v, 4) for v in dl]}")
        gen.reset_static()
        del gen


def test_mini_end_to_end_mini():
    _mini_end_to_end("mini")


def test_mini_end_to_end_mini4():
    _mini_end_to_end("mini4")


TESTS = [test_route_kernel_on_device_equals_mirror, test_stage_kernel_on_device_equals_contract,
         test_layouts_equal_library_transform, test_mini_moe_level_gates_mini, test_mini_moe_level_gates_mini4,
         test_mini_end_to_end_mini, test_mini_end_to_end_mini4]

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
