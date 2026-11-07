"""GPU tests of T9b-1 (runtime/gemv_mxfp8.py, `fp8_impl="gemv"`). Skipped without CUDA/tilelang/triton (Mac).

  * the kernel's own scale functions on the device (the integer pair scale AND the pow2(sa) * pow2(sb) product the
    GEMV evaluates) == `pair_scale` (CPU) on all 65,536 code pairs, bit for bit (catches a flush-to-zero multiply);
  * GEMV vs the reference TileLang `fp8_gemm` (kernel.py:207-307, block 32, e8m0) on synthetic fp8 weights at every
    distinct fp8 Linear shape of the real model plus the mini shapes and N not a multiple of the tile (or of 32), M in
    {1, 2, 3, 6, 8}, activations from the reference `act_quant`: finite, bf16 mismatch <= 0.5 % of outputs (the 1-ulp
    class; the max ulp distance and the float64 error of both kernels are printed);
  * split-K / MMA (fix round): explicit split configs incl. K/32 not a multiple of the split and a split collapsing to
    fewer splits, M in {1, 3, 6, 8}, both block-partial forms, vs the reference and vs the CPU mirror of the same split
    order (<= 0.5 %); MMA=1 rejected by this Triton is reported, not failed (the tuner skips it the same way);
  * configs with the same numerics_key (split partition, MMA) give bit-identical output (BLOCK_N / KU / warps / stages
    never change the bits); across keys the mismatch is printed; K-loop remainder (K/32 not a multiple of KU);
  * graph capture (incl. split-K + reduce launches) + replay == eager, twice; no host sync after the first call;
    tuning and first-use workspace allocation both refused under capture;
  * mini checkpoints, `fp8_impl="gemv"`: static-eager self-replay equal and no host sync; graph == static-eager,
    replay #2, fills-on recapture, relaunch equal; the top-1 stream over prefill + 10 decode steps equals `ref`'s (on a
    mismatch both streams are printed and the test fails); the routed-call count per step is 7 per layer + indexer
    wq_b + Engram wkv; capture with tuning on (configs chosen in the warm-up, none under capture);
  * informative: mini graph step-cell p50, ref vs gemv (and dg-masked + gemv when deep_gemm imports).

Run on the box (orchestrator), from the repo checkout, inside the library image with the GB300 by CDI; same line as
tests/gpu/test_dg_moe_gpu.py (the Triton cache mount keeps the GEMV compiles across runs):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_gemv_gpu.py

(`python3 tests/gpu/test_gemv_gpu.py` runs every test without pytest; no checkpoint mount needed.)
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_static_mini_gpu import (N_STEPS, _assert_equal_runs, _first_light, _ids, _model,  # noqa: E402
                                  _need_tilelang, _run)

GEMV = "gemv"
E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn
MISMATCH_GATE = 5e-3
# every distinct fp8 Linear shape (N, K) the real 1-row step calls (docs/checkpoint-1017-inventory.json + config):
# attn wq_a, wq_b, wkv, wo_b; indexer wq_b; shared w1=w3, w2; Engram wkv
REAL_SHAPES = [(1280, 5120), (32768, 1280), (512, 5120), (5120, 8192), (4096, 1280), (2304, 5120), (5120, 2304),
               (25600, 6144)]
MINI_SHAPES = [(128, 256), (512, 128), (256, 256), (256, 128), (1280, 1536)]
ODD_SHAPES = [(1000, 512), (2304 + 16, 384), (40, 96), (7, 64)]   # N % tile != 0, N % 32 != 0, K/32 % KU != 0


def _need_gemv():
    _need_tilelang()
    from runtime import gemv_mxfp8

    if not gemv_mxfp8.HAVE_TRITON:
        raise AssertionError("triton not importable in the GPU image")  # on the box: a FAIL, it is under test


def _kernel():
    from runtime.refmodel import import_reference

    import_reference()
    return sys.modules["kernel"]


def _act(M, K, seed):
    """bf16 activations with per-32-block magnitudes 2^-6..2^6 (the T9a spikes' recipe: the model's per-32 scales
    differ block to block), quantized by the reference act_quant (model.py:197)."""
    K_ = _kernel()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(M, K // 32, 32, generator=g) * torch.exp2(torch.randint(-6, 7, (M, K // 32, 1), generator=g)
                                                                .float())).view(M, K).to(torch.bfloat16).cuda()
    return K_.act_quant(x, 32, "ue8m0", E8)


def _weights(N, K, seed):
    """fp8 from N(0, 0.5) (mini_ckpt's recipe) and e8m0 block scales 118..130."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 0.5).to(FP8).cuda()
    s = torch.randint(118, 131, ((N + 31) // 32, K // 32), generator=g, dtype=torch.int32).to(torch.uint8).cuda()
    return w, s.view(E8)


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ulps between a and b (same-sign: |bit difference|; sign change: sum of both distances to 0)."""
    def key(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)  # monotone integer line through +-0
    return (key(a) - key(b)).abs()


def compare(got: torch.Tensor, ref: torch.Tensor, exact=None) -> dict:
    d = (got.float() - ref.float()).abs()
    u = ulp_distance(got, ref)
    out = {"mismatch": float((got != ref).float().mean()), "max_abs": float(d.max()), "max_ulp": int(u.max()),
           "n_gt1ulp": int((u > 1).sum()), "finite": bool(torch.isfinite(got.float()).all())}
    if exact is not None:
        out["mean_err_gemv"] = float((got.double() - exact).abs().mean())
        out["mean_err_ref"] = float((ref.double() - exact).abs().mean())
    return out


def _exact(xq, xs, w, s):
    N = w.size(0)
    A = xq.double() * xs.float().double().repeat_interleave(32, 1)
    rows = torch.arange(N, device=w.device) // 32
    B = w.double() * s.float().double()[rows].repeat_interleave(32, 1)
    return A @ B.t()


# ------------------------------------------------------------------------------------------------ kernel tests
def test_pair_scale_device_all_pairs():
    _need_gemv()
    from runtime.gemv_mxfp8 import pair_scale, pair_scale_device

    c = torch.arange(256, dtype=torch.int32)
    a, b = c.repeat_interleave(256), c.repeat(256)
    want = pair_scale(a, b)
    nan = torch.isnan(want)
    # mode 0: the integer pair scale; mode 1: pow2(sa) * pow2(sb), the expression the GEMV kernel evaluates (a
    # flush-to-zero fp32 multiply would show up here as zeros where subnormals are expected)
    for mode in (0, 1):
        got = pair_scale_device(a.to(torch.uint8).cuda(), b.to(torch.uint8).cuda(), mode=mode).cpu()
        assert torch.equal(torch.isnan(got), nan), mode
        assert torch.equal(got[~nan].view(torch.int32), want[~nan].view(torch.int32)), mode
    print("  device pair scale (integer form and the kernel's pow2 x pow2 form) == CPU pair_scale on 65,536 pairs")


def _vs_reference(shapes, Ms, tag, seed0):
    from runtime.gemv_mxfp8 import DEFAULT_CONFIG, mxfp8_gemv

    K_ = _kernel()
    torch.set_default_dtype(torch.bfloat16)
    worst = []
    for i, (N, K) in enumerate(shapes):
        w, s = _weights(N, K, seed0 + i)
        for M in Ms:
            xq, xs = _act(M, K, seed0 + 100 * i + M)
            ref = K_.fp8_gemm(xq, xs, w, s, E8, block_size=32)
            got = mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16, device="cuda"),
                             config=DEFAULT_CONFIG)
            r = compare(got, ref, _exact(xq, xs, w, s) if N * K <= (64 << 20) else None)
            print(f"  {tag} N={N} K={K} M={M}: mismatch {r['mismatch'] * 100:.3f} % max ulp {r['max_ulp']} "
                  f"(>1 ulp: {r['n_gt1ulp']}) max|d| {r['max_abs']:.3g}"
                  + (f"; mean |err| vs f64: gemv {r['mean_err_gemv']:.3g} ref {r['mean_err_ref']:.3g}"
                     if "mean_err_gemv" in r else ""))
            assert r["finite"], (N, K, M)
            worst.append((r["mismatch"], N, K, M))
    bad = [x for x in worst if x[0] > MISMATCH_GATE]
    assert not bad, f"{tag}: mismatch above {MISMATCH_GATE * 100} % (the 1-ulp class): {bad}"


def test_gemv_vs_reference_real_shapes():
    _need_gemv()
    _vs_reference(REAL_SHAPES, (1, 6), "real", 10)


def test_gemv_vs_reference_mini_and_odd_shapes():
    _need_gemv()
    _vs_reference(MINI_SHAPES + ODD_SHAPES, (1, 2, 3, 6, 8), "mini/odd", 50)


# (N, K, split_k): KB not a multiple of the split (last split shorter), a requested split that collapses to fewer
# non-empty splits, split 1, and the auto split of a real small shape
SPLIT_CASES = [(512, 5120, 7), (1280, 1280, 6), (1000, 384, 5), (2304, 5120, 1), (512, 5120, 19), (40, 288, 4)]


def test_gemv_split_k_and_mma_vs_reference():
    """Explicit split-K / MMA configs at M = 1, 3, 6, 8 vs the reference fp8_gemm (<= 0.5 %, finite) and vs the CPU
    mirror of the same split order (gemv_torch on the device: same split partition, only the intra-block order
    differs, so also the 1-ulp class). MMA=1 (tl.dot on fp8) may be rejected by this Triton on this GPU -- then it is
    reported and the tuner never picks it; a wrong MMA result fails."""
    _need_gemv()
    from runtime import gemv_mxfp8 as G

    K_ = _kernel()
    torch.set_default_dtype(torch.bfloat16)
    bad, mma_err = [], None
    for i, (N, K, sk) in enumerate(SPLIT_CASES):
        w, s = _weights(N, K, 700 + i)
        for M in (1, 3, 6, 8):
            xq, xs = _act(M, K, 800 + 10 * i + M)
            ref = K_.fp8_gemm(xq, xs, w, s, E8, block_size=32)
            mirror = G.gemv_torch(xq, xs, w, s, split_k=sk)
            for mma in (0, 1):
                if mma and mma_err:
                    continue
                cfg = G.GemvConfig(32 if M > 2 or mma else 64, 4 if K // 32 % 4 else 8, 4, 2, sk, mma)
                plan = G.launch_plan(cfg, M, N, K)
                try:
                    got = G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16, device="cuda"),
                                       config=cfg).clone()
                    torch.cuda.synchronize()
                except Exception as e:
                    if not mma:
                        raise
                    mma_err = repr(e)[:300]
                    print(f"  MMA=1 (tl.dot fp8) unavailable here: {mma_err}")
                    continue
                r, rm = compare(got, ref), compare(got, mirror)
                print(f"  split N={N} K={K} M={M} split_k={sk} -> (PER, NSPLIT) = ({plan['PER']}, {plan['NSPLIT']}) "
                      f"EVEN={plan['EVEN']} mma={mma}: vs ref {r['mismatch'] * 100:.3f} % (max ulp {r['max_ulp']}), "
                      f"vs CPU mirror {rm['mismatch'] * 100:.3f} %")
                if not r["finite"] or r["mismatch"] > MISMATCH_GATE or rm["mismatch"] > MISMATCH_GATE:
                    bad.append((N, K, M, sk, mma, r["mismatch"], rm["mismatch"]))
    assert not bad, f"split-K / MMA results outside the 1-ulp class: {bad}"


def _invariance_sample(N, K, M, per_group=6):
    """Up to `per_group` candidates per numerics_key group, spread over the group (BLOCK_N, KU, warps, stages vary)."""
    from runtime import gemv_mxfp8 as G

    groups = {}
    for c in G.candidate_configs(N, K, M, G.num_sms()):
        groups.setdefault(G.numerics_key(c, N, K), []).append(c)
    out = {}
    for k, cs in groups.items():
        step = max(1, len(cs) // per_group)
        out[k] = cs[::step][:per_group]
    return out


def test_every_candidate_config_bit_identical():
    """Configs with the same numerics_key (split partition, MMA) give bit-identical outputs: BLOCK_N / KU / warps /
    stages never change the bits, so a relaunch or a second process replays the same bits for a pinned split + MMA.
    Different keys may differ by an ulp (split order / intra-block order) -- printed, not asserted."""
    _need_gemv()
    from runtime import gemv_mxfp8 as G

    n_groups = n_cfgs = 0
    skipped = {}
    for i, (N, K) in enumerate([(1280, 5120), (512, 5120), (1000, 384), (2304, 5120)]):
        w, s = _weights(N, K, 200 + i)
        for M in (1, 6):
            xq, xs = _act(M, K, 300 + i + M)
            reps = {}
            for key, cfgs in _invariance_sample(N, K, M).items():
                outs = {}
                for cfg in cfgs:
                    try:
                        outs[cfg] = G.mxfp8_gemv(xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16, device="cuda"),
                                                 config=cfg).clone()
                    except Exception as e:  # MMA=1 rejected by this Triton: the tuner skips it the same way
                        skipped[json_key(cfg)] = repr(e)[:120]
                if not outs:
                    continue
                base_cfg = next(iter(outs))
                diff = [c.as_dict() for c, o in outs.items() if not torch.equal(o, outs[base_cfg])]
                assert not diff, (f"N={N} K={K} M={M} numerics {key}: configs {diff} differ from "
                                  f"{base_cfg.as_dict()} -- BLOCK_N/KU/warps/stages changed the bits; pin configs "
                                  f"across processes (DSV41_GEMV_CACHE) before trusting x-proc L2 rows")
                reps[key] = outs[base_cfg]
                n_groups += 1
                n_cfgs += len(outs)
            keys = sorted(reps)
            cross = {f"{a}~{b}": float((reps[a] != reps[b]).float().mean()) for a, b in zip(keys, keys[1:])}
            print(f"  N={N} K={K} M={M}: {len(keys)} numerics groups bit-identical inside; across groups mismatch "
                  f"{cross}")
    print(f"  {n_cfgs} configs in {n_groups} (shape, M, numerics) groups bit-identical within each group; "
          f"{len(skipped)} configs skipped (did not compile/launch)")


def json_key(cfg):
    return str(sorted(cfg.as_dict().items()))


def test_graph_capture_replay_and_no_sync():
    _need_gemv()
    from runtime import gemv_mxfp8 as G

    shapes = [(1280, 5120, 1), (512, 5120, 1), (512, 5120, 6), (32768, 1280, 3)]
    bufs = []
    for i, (N, K, M) in enumerate(shapes):
        w, s = _weights(N, K, 400 + i)
        xq, xs = _act(M, K, 500 + i)
        bufs.append((xq, xs, w, s, torch.empty(M, N, dtype=torch.bfloat16, device="cuda")))
    G.reset_configs()
    for xq, xs, w, s, out in bufs:  # eager: tunes (or defaults), compiles, allocates the split-K workspaces
        G.mxfp8_gemv(xq, xs, w, s, out)
    plans = {G._key_str(G.config_key(b[2].size(0), b[2].size(1), b[0].size(0))):
             G.launch_plan(G.get_config(b[2].size(0), b[2].size(1), b[0].size(0)), b[0].size(0), b[2].size(0),
                           b[2].size(1))["NSPLIT"] for b in bufs}
    assert any(v > 1 for v in plans.values()), plans   # the capture below includes split-K + reduce launches
    eager = [b[4].clone() for b in bufs]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for xq, xs, w, s, out in bufs:
            G.mxfp8_gemv(xq, xs, w, s, out)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for b in bufs:
            G.mxfp8_gemv(*b)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for b in bufs:
            G.mxfp8_gemv(*b)
    for rep in range(2):
        for b in bufs:
            b[4].fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        for b, e in zip(bufs, eager):
            assert torch.equal(b[4], e), f"graph replay {rep} differs from eager"
    # an unknown shape under capture is refused when tuning is on (no timing, no JIT inside a capture)
    w, s = _weights(96, 256, 600)
    xq, xs = _act(1, 256, 601)
    out = torch.empty(1, 96, dtype=torch.bfloat16, device="cuda")
    prev = os.environ.get("DSV41_GEMV_TUNE")
    os.environ["DSV41_GEMV_TUNE"] = "1"
    raised = False
    try:
        g2 = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g2):
                G.mxfp8_gemv(xq, xs, w, s, out)
        except RuntimeError as e:
            raised = "capture" in str(e)
    finally:
        if prev is None:
            os.environ.pop("DSV41_GEMV_TUNE", None)
        else:
            os.environ["DSV41_GEMV_TUNE"] = prev
    assert raised, "an untuned shape under capture must raise, not tune"
    # a known config whose split-K workspace was never allocated eagerly: refused under capture (no allocation there)
    w, s = _weights(640, 5120, 602)
    xq, xs = _act(1, 5120, 603)
    out = torch.empty(1, 640, dtype=torch.bfloat16, device="cuda")
    G.set_config(640, 5120, 1, G.GemvConfig(32, 4, 4, 1, 11, 0))
    raised = False
    g3 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g3):
            G.mxfp8_gemv(xq, xs, w, s, out)
    except RuntimeError as e:
        raised = "capture" in str(e) and "workspace" in str(e)
    assert raised, "a split-K workspace first needed under capture must raise, not allocate"
    print(f"  graph replay x2 == eager (split-K plans {plans}); no host sync; untuned shape and unallocated "
          f"workspace under capture refused; configs {G.describe()['configs']}")


# ------------------------------------------------------------------------------------------------ mini checkpoint
def _gen(model, tokdir, **kw):
    from runtime.generate import Generator, set_determinism
    from test_static_mini_gpu import MAX_POS

    set_determinism(0)
    return Generator(model, tokdir, max_pos=MAX_POS, **kw)


def _expected_routed(model):
    return sum(7 + (l.attn.indexer is not None) + (l.engram is not None) for l in model.layers)


def _gemv_static(kind):
    _need_gemv()
    from runtime import patches

    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, fp8_impl=GEMV)
        ids = _ids(16, 5)
        a = _run(gen, "static", ids)
        patches.fp8_gemv_stats(reset=True)
        b = _run(gen, "static", ids)
        _assert_equal_runs(b, a, f"{kind}: gemv static-eager self-replay")
        st = patches.fp8_gemv_stats()
        assert sum(st["routed"].values()) == N_STEPS * _expected_routed(model) and not st["fallthrough"], st
        gen.set_mode("static")
        gen.prefill(ids)
        gen.step(int(gen.next_token_tensor().reshape(-1)[0]), len(ids))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for p in (len(ids) + 1, len(ids) + 2):
                gen.step(gen.next_token_tensor(), p)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        print(f"  {kind}: gemv static-eager self-replay equal, {_expected_routed(model)} routed/step, no host sync")
        gen.reset_static()
        del gen


def test_gemv_static_selfreplay_mini():
    _gemv_static("mini")


def test_gemv_static_selfreplay_mini4():
    _gemv_static("mini4")


def _gemv_graph(kind):
    _need_gemv()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, fp8_impl=GEMV)
        ids = _ids(16, 7)
        st = _run(gen, "static", ids)
        g1 = _run(gen, "graph", ids)
        assert gen.static.captured and gen.static.fp8_impl == GEMV
        _assert_equal_runs(g1, st, f"{kind}: gemv graph vs static-eager")
        g2 = _run(gen, "graph", ids)
        _assert_equal_runs(g2, g1, f"{kind}: gemv graph replay#2 vs #1")
        gen.capture(fills=True)
        g3 = _run(gen, "graph", ids)
        _assert_equal_runs(g3, g1, f"{kind}: gemv fills-on capture vs fills-off")
        gen.reset_static()
        g4 = _run(gen, "graph", ids)
        _assert_equal_runs(g4, g1, f"{kind}: gemv relaunch vs replay#1")
        print(f"  {kind}: gemv graph == static-eager; replay x2, fills-on recapture, relaunch all equal")
        gen.reset_static()
        del gen


def test_gemv_graph_equals_static_mini():
    _gemv_graph("mini")


def test_gemv_graph_equals_static_mini4():
    _gemv_graph("mini4")


def _gemv_vs_ref(kind):
    _need_gemv()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir)
        ids = _ids(16, 3)
        ref = _run(gen, "static", ids)
        gen.set_fp8_impl(GEMV)
        got = _run(gen, "static", ids)
        if got[1] != ref[1]:
            print(f"  {kind}: TOKEN STREAMS DIFFER\n    ref  {ref[1]}\n    gemv {got[1]}")
        assert got[1] == ref[1], (kind, "gemv vs ref top-1 stream", ref[1], got[1])
        dl = [float((a - b).abs().max()) for a, b in zip(got[0], ref[0])]
        same = sum(torch.equal(a, b) for a, b in zip(got[0], ref[0]))
        print(f"  {kind}: top-1 stream equal over prefill + {N_STEPS} steps; {same}/{len(dl)} positions bit-equal; "
              f"max |dlogit| per position {[round(x, 5) for x in dl]}")
        gen.reset_static()
        del gen


def test_gemv_vs_ref_tokens_mini():
    _gemv_vs_ref("mini")


def test_gemv_vs_ref_tokens_mini4():
    _gemv_vs_ref("mini4")


def test_gemv_capture_with_tuning_and_cache():
    """Tuning on: configs are chosen during the warm-up steps (never under capture), written to the cache file, and a
    reset + reload gives the same configs (the cross-process path). Quick tune space (12-ish candidates per shape)."""
    _need_gemv()
    from runtime import gemv_mxfp8 as G

    with tempfile.TemporaryDirectory() as tmp:
        prev = {k: os.environ.get(k) for k in ("DSV41_GEMV_TUNE", "DSV41_GEMV_CACHE", "DSV41_GEMV_TUNE_SPACE")}
        os.environ["DSV41_GEMV_TUNE"] = "1"
        os.environ["DSV41_GEMV_TUNE_SPACE"] = "quick"
        os.environ["DSV41_GEMV_CACHE"] = os.path.join(tmp, "tune.json")
        G.reset_configs()
        try:
            with _model("mini") as (model, tokdir):
                gen = _gen(model, tokdir, fp8_impl=GEMV, mode="graph")
                ids = _ids(16, 11)
                a = _run(gen, "graph", ids)
                cfgs = dict(G.describe()["configs"])
                assert cfgs and os.path.exists(os.environ["DSV41_GEMV_CACHE"])
                G.reset_configs()
                G.load_cache()
                assert G.describe()["configs"] == cfgs
                gen.reset_static()
                b = _run(gen, "graph", ids)
                _assert_equal_runs(b, a, "mini: tuned graph vs relaunch from the cache file")
                print(f"  tuned during warm-up: {cfgs}; tuned µs {G.describe()['tuned_us'] or '(from cache)'}")
                gen.reset_static()
                del gen
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            G.reset_configs()


def test_gemv_step_cell_informative():
    _need_gemv()
    fl = _first_light()
    combos = [("ref", "ref"), ("ref", GEMV)]
    try:
        from runtime.dg_moe import import_deep_gemm

        import_deep_gemm()
        combos.append(("dg-masked", GEMV))
    except Exception as e:
        print(f"  (dg-masked + gemv skipped: {repr(e)[:80]})")
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, mode="graph")
        ids = _ids(16, 8)
        p50, toks = {}, {}
        for moe, fp8 in combos:
            gen.set_moe_impl(moe)
            gen.set_fp8_impl(fp8)
            gen.set_mode("graph")
            sc = fl.step_cell(gen, ids, 40, stats_from=8)
            p50[(moe, fp8)], toks[(moe, fp8)] = sc["ms_per_step"]["p50"], sc["token_ids"][:N_STEPS + 1]
        base = toks[("ref", "ref")]
        print("  mini4 graph step cell p50: " + ", ".join(f"{m}/{f} {v:.3f} ms" for (m, f), v in p50.items())
              + f"; first {N_STEPS + 1} tokens equal to ref/ref: "
              + str({f"{m}/{f}": t == base for (m, f), t in toks.items()}))
        gen.reset_static()
        del gen


TESTS = [test_pair_scale_device_all_pairs, test_gemv_vs_reference_real_shapes, test_gemv_vs_reference_mini_and_odd_shapes,
         test_gemv_split_k_and_mma_vs_reference, test_every_candidate_config_bit_identical,
         test_graph_capture_replay_and_no_sync,
         test_gemv_static_selfreplay_mini, test_gemv_static_selfreplay_mini4, test_gemv_graph_equals_static_mini,
         test_gemv_graph_equals_static_mini4, test_gemv_vs_ref_tokens_mini, test_gemv_vs_ref_tokens_mini4,
         test_gemv_capture_with_tuning_and_cache, test_gemv_step_cell_informative]

if __name__ == "__main__":
    import traceback

    os.environ.setdefault("DSV41_GEMV_TUNE", "0")  # the mini/kernel tests use DEFAULT_CONFIG unless a test turns it on
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
    sys.exit(1 if n_fail else 0)
