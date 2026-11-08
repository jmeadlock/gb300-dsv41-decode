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

M3-N1 (bench/dg_mega_numerics.py, runtime/dg_mega.MegaSubstitution; docs/M2-L3-NUMERICS.md):
  * GPU: MegaSubstitution on ONE layer of a dg-inplace step == that layer of a full dg-mega step, bitwise (R = 1 and
    6), and restore() gives the dg-inplace output back bitwise;
  * CPU (`test_cpu_*`, run by `pytest tests/` on the Mac -- this is the one test file M3-N1 owns): the quantizers
    and SwiGLU formulas against the existing emulations, the stage decomposition on a synthetic layer (tm_fix ==
    ours bitwise, tm's L2-input bytes differ, the l2_acts matcher on shuffled rows), the kernel patch on the vendored
    .cuh and patch-dg's copy, and MegaSubstitution on the emulated mini step with the contract-checking fake
    MegaMoE (bitwise vs the oracle; restore() bit-identical to before).
"""
from __future__ import annotations

import contextlib
import math
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


# ------------------------------------------------------------------------------------------------ M3-N1 (GPU)
def _substitution_gate(kind):
    """MegaSubstitution on layer 0 of a dg-inplace step == layer 0 of a full dg-mega step (bitwise, R = 6 and each
    row alone); restore() -> the dg-inplace outputs bitwise as before."""
    _need_dg()
    from runtime import dg_mega as DM
    from runtime.static_step import StaticStep

    with _model(kind) as (model, tokdir), _mega_enabled():
        pool = model.runtime_load.pool
        D = model.layers[0].ffn.dim
        sti = StaticStep(model, pool, max_pos=MAX_POS, moe_impl=DGI)
        x6 = (torch.randn(6, D, generator=_g(23), device="cpu") * 2).to(torch.bfloat16).cuda()
        with torch.inference_mode(), sti._active():
            before = [sti.moes[0](x6[r:r + 1].clone()).clone() for r in range(6)]
        stm = StaticStep(model, pool, max_pos=MAX_POS, moe_impl=MEGA)
        with torch.inference_mode(), stm._active():
            want6 = stm.moes[0](x6.clone()).clone()
            want1 = [stm.moes[0](x6[r:r + 1].clone()).clone() for r in range(6)]
        stm.detach()
        del stm
        sub = DM.MegaSubstitution(sti, [0])
        assert isinstance(sti.moes[0], DM.DGMegaMoE) and not isinstance(sti.moes[1], DM.DGMegaMoE)
        with torch.inference_mode(), sti._active():
            got6 = sti.moes[0](x6.clone()).clone()
            got1 = [sti.moes[0](x6[r:r + 1].clone()).clone() for r in range(6)]
        print(f"  {kind}: substitution {sub.describe()['bufs'].get('plan')}")
        assert torch.equal(got6.view(torch.int16), want6.view(torch.int16)), (kind, "R6 substituted vs dg-mega step")
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(got1, want1)), (kind, "R1")
        sub.restore()
        with torch.inference_mode(), sti._active():
            after = [sti.moes[0](x6[r:r + 1].clone()).clone() for r in range(6)]
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(after, before)), \
            (kind, "restore() vs before")
        sti.detach()
        print(f"  {kind}: MegaSubstitution(layer 0) == the dg-mega step bitwise (R 6 and 1); restore() bit-identical")


def test_m3n1_substitution_mini():
    _substitution_gate("mini")


def test_m3n1_substitution_mini4():
    _substitution_gate("mini4")


# ------------------------------------------------------------------------------------------------ M3-N1 (CPU)
def _num():
    import dg_mega_numerics as N

    return N


def test_cpu_quantizers_match_the_emulations():
    """scale_mega == the definition (smallest power of two with amax / s <= 448, floor 2^-22); quant32(scale_mega)
    == test_dg_mega_cpu._quant_fp32 (the fake MegaMoE's requant) bitwise; quant32(scale_ref) == the emulated
    reference act_quant except at fp32 product boundaries (counted: < 0.1 %)."""
    from test_dg_mega_cpu import _quant_fp32
    from test_static_emulated_cpu import act_quant as emu_act_quant

    N = _num()
    g = _g(5)
    z = torch.randn(64, 2048, generator=g, device="cpu") * torch.exp2(torch.randint(-12, 12, (64, 1), generator=g,
                                                                                  device="cpu").float())
    amax = z.view(64, -1, 32).abs().amax(-1)
    s = N.scale_mega(amax)
    assert bool((amax / s <= 448).all()) and bool(((amax / (s / 2) > 448) | (s <= 2.0 ** -22)).all())
    q, sc = N.quant32(z, N.scale_mega)
    assert torch.allclose(N.dequant(q, sc).float(), _quant_fp32(z), rtol=0, atol=0)
    qr, sr = N.quant32(z.to(torch.bfloat16).float(), N.scale_ref)
    qe, se = emu_act_quant(z.to(torch.bfloat16), 32, "ue8m0")
    assert float((sr != se).float().mean()) < 1e-3 and float((qr.view(torch.uint8) != qe.view(torch.uint8))
                                                               .float().mean()) < 1e-3
    # the exact boundary: amax = 448 * 2^k is a power-of-two scale 2^k for MegaMoE
    b = torch.tensor([448.0, 448.0 * 2 ** -5, 448.0 * 2 ** 7], device="cpu")
    assert torch.equal(N.scale_mega(b), torch.tensor([1.0, 2 ** -5, 2.0 ** 7], device="cpu"))


def test_cpu_swiglu_formulas():
    """swiglu_mega == the fake MegaMoE's formula (test_dg_mega_cpu._mega_expert); the reference formula's bitwise
    agreement with it is reported, not asserted (F.silu's CPU and GPU kernels differ; the box measures it)."""
    N = _num()
    g = _g(9)
    gt = (torch.randn(4, 512, generator=g, device="cpu") * 6).to(torch.bfloat16)
    up = (torch.randn(4, 512, generator=g, device="cpu") * 6).to(torch.bfloat16)
    w = torch.rand(4, 1, generator=g, device="cpu")
    gate = torch.clamp(gt.float(), max=10.0)
    upc = torch.clamp(up.float(), min=-10.0, max=10.0)
    want = gate / (1.0 + torch.exp(-gate)) * upc * w
    assert torch.equal(N.swiglu_mega(gt, up, w, 10.0), want)
    eq = float((N.swiglu_ref(gt, up, w, 10.0) == want).float().mean())
    print(f"  swiglu ref vs mega formula on the CPU: {eq * 100:.2f} % bitwise equal")


def _synthetic_window(N, D=512, I=256, n_exp=10, seed=3):
    g = _g(seed)
    x = N.synth_x(6, D, g)
    layer = N.synth_layer(n_exp, D, I, g, x)
    rng = random.Random(seed)
    ids = torch.tensor([sorted(rng.sample(range(n_exp), 6)) for _ in range(6)], dtype=torch.long, device="cpu")
    w = torch.rand(6, 6, generator=g, device="cpu") + 0.05
    w = w / w.sum(-1, keepdim=True) * 1.5

    def deq(e):
        return tuple(N.deq_fp4(*layer["experts"][e][p]) for p in ("w1", "w3", "w2"))

    sw = tuple(N.deq_fp4(*layer["shared"][p]) for p in ("w1", "w3", "w2"))
    a64 = N.dequant(*N.quant32(x.float(), N.scale_ref))
    shared = torch.stack([N.oracle_stages(a64[r], sw[0], sw[1], sw[2], 1.0, 10.0, "bf16")["o"] for r in range(6)])
    return x, ids, w, deq, shared, layer


def test_cpu_decomposition_on_a_synthetic_layer():
    """tm_fix (bf16 before quant) == ours bitwise at every stage (same GEMMs; the SwiGLU formulas agree on these
    values or the bf16 rounding absorbs the difference -- asserted on the L2-input bytes); tm's L2-input bytes differ
    from ours on a few % and its routed distance to the oracle is several times ours; a fake kernel that returns
    tm's output and tm's L2 bytes (rows shuffled, padding rows of zeros) is matched row by row."""
    N = _num()
    x, ids, w, deq, shared, layer = _synthetic_window(N)
    gemm = N.emu_gemm_factory(layer, torch.device("cpu"))
    holder = {}

    def fake_mega():
        r = holder["res"]
        q = r["raw"]["q2"]["tm"][0].view(torch.uint8)
        perm = torch.randperm(q.size(0), generator=_g(1), device="cpu")
        snap = torch.cat([q[perm], torch.zeros(5, q.size(1), dtype=torch.uint8, device="cpu")])
        return r["raw"]["y"]["tm"][0].clone(), snap

    holder["res"] = N.analyze_window(x, ids, w, deq, shared, gemm, want_raw=True)
    res = N.analyze_window(x, ids, w, deq, shared, gemm, mega=fake_mega, want_raw=True)
    s = N.finalize(res["stats"])
    raw = res["raw"]
    assert torch.equal(raw["q2"]["tm_fix"][0].view(torch.uint8), raw["q2"]["ours"][0].view(torch.uint8))
    assert torch.equal(raw["y"]["tm_fix_1r"][1].view(torch.int16), raw["y"]["ours"][1].view(torch.int16))
    assert 0.005 < s["l2_input_pairwise"]["tm_vs_ours"]["mismatch_frac"] < 0.2
    assert s["routed"]["tm"]["rel_l2"] > 2 * s["routed"]["ours"]["rel_l2"]
    assert s["pairwise_routed"]["mega_vs_tm"]["mismatch_frac"] == 0.0
    m = res["l2_match"]
    assert m["rows_written"] == m["rows_matched"] == m["pairs"] == m["pairs_covered"] == 36, m
    assert m["frac_bytes_eq_tm"] == 1.0 and m["frac_bytes_eq_ours"] < 1.0, m
    summ = N.summarize_stats(s)
    assert summ["final"]["mega"] is not None and len(N.summary_lines(summ, "t")) == 7
    merged = N.finalize(N.merge(N.merge(None, res["stats"]), res["stats"]))
    assert abs(merged["routed"]["tm"]["rel_l2"] - s["routed"]["tm"]["rel_l2"]) < 1e-12


def test_cpu_kernel_patch_on_the_vendored_source(tmp_path):
    """The bf16-before-quant patch applies exactly once to the vendored sm100_fp8_fp4_mega_moe.cuh, refuses a second
    time, and patch-dg copies a package with the patched header + a manifest (here from the vendored tree)."""
    N = _num()
    src_pkg = os.path.join(REPO, "reference", "vllm-0.31.0", "deep_gemm")
    txt = open(os.path.join(src_pkg, N.CUH)).read()
    new = N.patch_bf16_before_quant(txt)
    assert new.count(N.PATCH_TAG) == 1 and len(new) == len(txt) + len(N.PATCH_LINES)
    assert "__float22bfloat162_rn(activation_values[i][k])" in new
    try:
        N.patch_bf16_before_quant(new)
        raise AssertionError("double patch accepted")
    except ValueError:
        pass
    out = tmp_path / "dgp"
    assert N.main(["patch-dg", "--src", src_pkg, "--out", str(out)]) == 0
    man = __import__("json").load(open(out / "patch-manifest.json"))
    assert man["sha256_before"] != man["sha256_after"]
    assert N.PATCH_TAG in open(out / "deep_gemm" / N.CUH).read()
    assert N.PATCH_TAG not in open(os.path.join(src_pkg, N.CUH)).read()       # the vendored tree is untouched


def test_cpu_groups_and_tf_metrics():
    N = _num()
    assert N.parse_groups("auto", [0, 10, 20, 39]) == [[0, 10, 20, 39]]
    assert N.parse_groups("0,10|20|39", [0]) == [[0, 10], [20], [39]]
    assert N.split_group([0, 10, 20, 39]) == [[0, 10], [20, 39]] and N.split_group([7]) == []
    rows = [{"tag": "a", "logprobs": [{"logprob": -0.1, "top_logprobs": [{"logprob": -0.1}, {"logprob": -2.1}]},
                                      {"logprob": -0.5, "top_logprobs": [{"logprob": -0.5}, {"logprob": -0.6}]}]}]
    targets = {"a": [5, 7]}
    base = {"a": {"n": 2, "argmax": [5, 7], "lp_target": [-0.1, -0.5], "top_lp": [[-0.1, -2.1], [-0.5, -0.6]]}}
    cand = {"a": {"n": 2, "argmax": [5, 8], "lp_target": [-0.1, -0.7], "top_lp": [[-0.1, -2.1], [-0.4, -0.7]]}}
    m = N.tf_compare(cand, rows, targets, base)
    t = m["total"]
    assert t["positions"] == 2 and t["flips_pg61"] == 1 and abs(t["flips_w_pg61"] - 0.1) < 1e-9
    assert t["flips_base"] == 1 and abs(t["flips_w_base"] - 0.1) < 1e-9 and abs(t["max_abs_dlp_base"] - 0.2) < 1e-9


def test_cpu_layerbox_run_layer_with_fakes():
    """The box `layers` path (LayerBox + run_layer: checkpoint load, cold placement into spare groups for R = 6 and
    each row alone, today's contiguous GEMM, the MegaMoE call, the timer) at small dims on the CPU: the contiguous
    fake of tests/test_dg_inplace_cpu.py + the contract-checking MegaMoE fake of tests/test_dg_mega_cpu.py and a
    synthetic checkpoint. Shapes, routing and bookkeeping only -- the numbers are the box's."""
    from test_dg_inplace_cpu import _fake_dg_contig
    from test_dg_mega_cpu import _fake_mega

    N = _num()
    D, I, E, k, H, S = 256, 256, 16, 6, 12, 4
    g = _g(31)
    x = N.synth_x(12, D, g)
    lay = N.synth_layer(E, D, I, g, x)

    class Ck:
        def expert(self, L, e, p):
            return lay["experts"][e][p]

        def get(self, name):
            p = name.split(".")[-2]
            n_, k_ = (I, D) if p in ("w1", "w3") else (D, I)
            if name.endswith("weight"):
                return (torch.randn(n_, k_, generator=_g(len(name)), device="cpu") * 0.05).to(torch.float8_e4m3fn)
            return torch.full((n_ // 32, k_ // 32), 120, dtype=torch.uint8, device="cpu")

    dg = _fake_dg_contig()
    fm = _fake_mega()
    dg.SymmBuffer, dg.fp8_fp4_mega_moe = fm.SymmBuffer, fm.fp8_fp4_mega_moe
    rng = random.Random(2)
    ids = torch.tensor([sorted(rng.sample(range(E), k)) for _ in range(12)], dtype=torch.long, device="cpu")
    w = torch.rand(12, k, generator=g, device="cpu") + 0.05
    acts = {"rows": {t: {0: {"pos": list(range(1, 7)), "x": x[o:o + 6], "ids": ids[o:o + 6], "w": w[o:o + 6]}}
                     for t, o in (("a", 0), ("b", 6))}}
    wins = N.load_windows(acts, 0, 8, 1)
    assert [v["tag"] for v in wins] == ["a", "b"] and N.load_windows(acts, 0, 8, 2) == []
    box = N.LayerBox(dg, Ck(), 0, list(range(H)), S, "cpu", dims=(D, I, E, k))
    rec = N.run_layer(box, wins, timer=lambda fn, iters: {"us": 0.0, "how": "fake"}, iters=1, log=lambda *a, **kw: None)
    assert rec["G"] == H + S and len(rec["windows"]) == 2 and rec["row_invariant_all"] is True
    assert set(rec["speed"]) == {"mega_R1_hot_us", "mega_R6_hot_us"}
    assert any(v >= H for v in box.placed) and set(box.placed.values()) <= set(range(H, E))
    s6 = rec["R6"]
    assert s6["l2_input_pairwise"]["tm_fix_vs_ours"]["mismatch_frac"] == 0.0
    assert all(math.isfinite(s6["final"][a_]["rel_l2"]) for a_ in N.ARMS)
    assert math.isfinite(rec["R1"]["final"]["mega"]["rel_l2"]) and rec["summary_R6"]["final"]["mega"] is not None
    assert "l2_match" not in rec                                     # the fake SymmBuffer has no l2_acts view
    lines = N.report_lines([{"layers": {"0": rec}}], [], None, None)
    assert any(l.startswith("| 0 | 6 |") for l in lines) and any(l.startswith("| 0 | 0 | 0 |") for l in lines)


def test_cpu_mega_substitution_on_the_emulated_step(mega_emu, monkeypatch):
    """MegaSubstitution on layer 0 of an emulated `ref` StaticStep (pool grown to k spares first, as the dg impls
    do): the substituted layer == the MegaMoE-numerics oracle bitwise (R 6 and 1, cold experts staged); decode runs
    and calls the fake once per step; restore() -> logits bit-identical to before the substitution."""
    from test_dg_mega_cpu import _oracle
    from test_static_emulated_cpu import _static_run

    from runtime import dg_mega as DM
    from runtime.static_step import StaticStep

    model, pool, fk, fake, R_mod = mega_emu("mini4")
    pool.add_spare_slots(6, model)                       # storage moves: re-register the fake gather's sources
    reg = next(c.cell_contents for c in fk.gather_rows.__closure__ if isinstance(c.cell_contents, dict))
    for l in range(pool.n_layers):
        for p in ("w1", "w2", "w3"):
            w, s = pool.hbm[l][p]
            for t in list(w) + list(s):
                reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
    st = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    ids = list(range(5, 21))
    before = _static_run(model, st, ids, 4)
    sub = DM.MegaSubstitution(st, [0])
    n0 = len(fake.calls)
    assert isinstance(st.moes[0], DM.DGMegaMoE) and not isinstance(st.moes[1], DM.DGMegaMoE) and n0 == 1
    lim = float(model.layers[0].ffn.shared_experts.swiglu_limit)
    D = model.layers[0].ffn.dim
    x6 = (torch.randn(6, D, generator=_g(4), device="cpu") * 2).to(torch.get_default_dtype())
    with torch.inference_mode(), st._active():
        y6 = st.moes[0](x6.clone())
        want = _oracle(st.moes[0].moe, x6, R_mod, lim, "gemv")
        assert torch.equal(y6.view(torch.int16), want.view(torch.int16))
        for r in range(6):
            assert torch.equal(st.moes[0](x6[r:r + 1].clone()).view(torch.int16), y6[r:r + 1].view(torch.int16))
    during = _static_run(model, st, ids, 4)
    assert all(bool(torch.isfinite(l).all()) for l in during[0])
    assert len(fake.calls) - n0 == 7 + 4, len(fake.calls)                   # 7 direct calls + 4 decode steps
    sub.restore()
    assert not any(isinstance(m, DM.DGMegaMoE) for m in st.moes) and sub.bufs is None
    after = _static_run(model, st, ids, 4)
    assert all(torch.equal(a, b) for a, b in zip(after[0], before[0])) and after[1] == before[1]
    st.detach()


if __name__ != "__main__":  # pytest: the CPU tests' fixtures (resolved by name in this module); the box script skips
    try:
        from test_dg_mega_cpu import mega_emu  # noqa: E402,F401
        from test_static_emulated_cpu import emu  # noqa: E402,F401
    except Exception:  # pragma: no cover
        pass

TESTS = [test_route_kernel_on_device_equals_mirror, test_stage_kernel_on_device_equals_contract,
         test_layouts_equal_library_transform, test_mini_moe_level_gates_mini, test_mini_moe_level_gates_mini4,
         test_mini_end_to_end_mini, test_mini_end_to_end_mini4, test_m3n1_substitution_mini,
         test_m3n1_substitution_mini4]

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
