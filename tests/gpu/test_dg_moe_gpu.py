"""GPU tests of T9a-1 (runtime/dg_moe.py, `moe_impl="dg-masked"`) on the synthetic mini checkpoints. Skipped without
CUDA/tilelang (Mac); on the box a missing/broken deep_gemm FAILS (it is what is under test).

  * e8m0 -> fp32 on the device: the conversion StaticStep picked is exact over all 256 codes (both methods reported);
  * dg-masked static-eager runs (finite logits, prefill + 10 steps, both parities), self-replay equal, no host sync;
  * graph replay == static-eager, replay #2 equal, fills-on recapture equal, relaunch equal (mini and mini4);
  * vs `ref`: the top-1 token stream over prefill + 10 decode steps must equal ref's (mini and mini4; on a mismatch
    both streams are printed and the test fails -- not loosened); the per-layer bf16 mismatch rate and max |delta| of
    the MoE output, dg vs ref on the same captured per-layer inputs, are printed (design expects <= 0.1 %; asserted
    finite only);
  * informative: graph step-cell p50, dg-masked vs ref.

Run on the box (orchestrator), from the repo checkout, inside the library image with the GB300 by CDI; same line as
tests/gpu/test_static_mini_gpu.py plus the DeepGEMM JIT cache:

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
    -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_dg_moe_gpu.py

(`python3 tests/gpu/test_dg_moe_gpu.py` runs every test without pytest; no checkpoint mount needed.)
"""
from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_static_mini_gpu import (N_STEPS, MAX_POS, _assert_equal_runs, _first_light, _ids, _model,  # noqa: E402
                                  _need_tilelang, _run)

DG = "dg-masked"


def _need_dg():
    _need_tilelang()
    from runtime.dg_moe import import_deep_gemm

    import_deep_gemm()  # raises (-> FAIL) on the box if deep_gemm cannot be imported without vllm


def _gen(model, tokdir, **kw):
    from runtime.generate import Generator, set_determinism

    set_determinism(0)
    return Generator(model, tokdir, max_pos=MAX_POS, **kw)


def _finite(run, what):
    bad = [i for i, l in enumerate(run[0]) if not bool(torch.isfinite(l).all())]
    assert not bad, f"{what}: non-finite logits at positions {bad}"


# ------------------------------------------------------------------------------------------------ tests
def test_e8m0_conversion_exact_on_device():
    _need_dg()
    from runtime.dg_moe import _bits_equal_table, e8m0_to_f32, pick_e8m0_method

    codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    res = {}
    for m in ("cast", "exp2"):
        out = torch.empty(256, dtype=torch.float32, device="cuda")
        try:
            e8m0_to_f32(codes, out, m)
            res[m] = _bits_equal_table(out)
        except Exception as e:
            res[m] = repr(e)[:120]
    picked = pick_e8m0_method("cuda")
    print(f"  e8m0 -> fp32 on cuda: {res}; picked {picked}")
    assert res[picked] is True


def _dg_static_selfreplay(kind):
    _need_dg()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=DG)
        ids = _ids(16, 5)
        a = _run(gen, "static", ids)
        _finite(a, f"{kind} dg static")
        b = _run(gen, "static", ids)
        _assert_equal_runs(b, a, f"{kind}: dg-masked static-eager self-replay")
        st = gen.static
        assert st.moe_impl == DG and st.dg is not None
        print(f"  {kind}: dg-masked static-eager self-replay equal; {st.dg.describe()}")
        # no host sync inside the dg step either
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
        gen.reset_static()
        del gen


def test_dg_static_eager_selfreplay_mini():
    _dg_static_selfreplay("mini")


def test_dg_static_eager_selfreplay_mini4():
    _dg_static_selfreplay("mini4")


def _dg_graph_vs_static(kind):
    _need_dg()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=DG)
        ids = _ids(16, 7)
        st = _run(gen, "static", ids)
        g1 = _run(gen, "graph", ids)  # prefill() captures first (fills off; DeepGEMM warmed in capture())
        assert gen.static.captured and gen.static.captured_fills is False and gen.static.moe_impl == DG
        _assert_equal_runs(g1, st, f"{kind}: dg graph vs static-eager")
        g2 = _run(gen, "graph", ids)
        _assert_equal_runs(g2, g1, f"{kind}: dg graph replay#2 vs #1")
        gen.capture(fills=True)
        g3 = _run(gen, "graph", ids)
        _assert_equal_runs(g3, g1, f"{kind}: dg fills-on capture vs fills-off")
        gen.reset_static()  # relaunch
        g4 = _run(gen, "graph", ids)
        _assert_equal_runs(g4, g1, f"{kind}: dg relaunch vs replay#1")
        assert g1[2] == st[2], (g1[2], st[2])
        print(f"  {kind}: dg-masked graph == static-eager; replay x2, fills-on recapture, relaunch all equal")
        gen.reset_static()
        del gen


def test_dg_graph_equals_static_eager_mini():
    _dg_graph_vs_static("mini")


def test_dg_graph_equals_static_eager_mini4():
    _dg_graph_vs_static("mini4")


def _moe_inputs(gen, model, ids):
    """Per-layer MoE inputs of one decode step (forward pre-hooks on every layer.ffn) in the current static mode."""
    store = {}
    hooks = [layer.ffn.register_forward_pre_hook(lambda m, args, l=l: store.__setitem__(l, args[0].detach().clone()))
             for l, layer in enumerate(model.layers)]
    try:
        gen.prefill(ids)
        gen.step(int(gen.next_token_tensor().reshape(-1)[0]), len(ids))
        torch.cuda.synchronize()
    finally:
        for h in hooks:
            h.remove()
    return [store[l] for l in range(len(model.layers))]


def _dg_vs_ref(kind):
    _need_dg()
    from runtime.static_step import StaticMoE

    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl="ref")
        ids = _ids(16, 3)
        ref = _run(gen, "static", ids)
        gen.set_moe_impl(DG)
        dg = _run(gen, "static", ids)
        _finite(dg, f"{kind} dg static")
        if dg[1] != ref[1]:
            print(f"  {kind}: TOKEN STREAMS DIFFER\n    ref {ref[1]}\n    dg  {dg[1]}")
        assert dg[1] == ref[1], (kind, "dg-masked vs ref top-1 stream", ref[1], dg[1])
        dl = [float((a - b).abs().max()) for a, b in zip(dg[0], ref[0])]
        print(f"  {kind}: top-1 stream equal over prefill + {N_STEPS} steps; max |dlogit| per position {dl}")
        # per-layer MoE output, dg vs ref on identical inputs (the inputs the dg step actually saw)
        gen.set_mode("static")
        xs = _moe_inputs(gen, model, ids)
        st = gen.static
        rates = []
        with torch.inference_mode(), st._active():
            for l, x in enumerate(xs):
                sm = st.moes[l]
                y_dg = sm(x.clone()).clone()
                y_ref = StaticMoE(st, sm.moe, sm.layer, sm.ptabs)(x.clone()).clone()
                assert bool(torch.isfinite(y_dg).all()), (kind, l)
                mism = float((y_dg != y_ref).float().mean())
                rates.append((l, mism, float((y_dg.float() - y_ref.float()).abs().max())))
        torch.cuda.synchronize()
        for l, m, mx in rates:
            print(f"  {kind} L{l}: MoE output bf16 mismatch {m * 100:.3f} % (design expects <= 0.1 %), max |d| {mx:.3g}"
                  + ("" if m <= 1e-3 else "  <-- above 0.1 %"))
        gen.reset_static()
        del gen


def test_dg_vs_ref_tokens_and_moe_mismatch_mini():
    _dg_vs_ref("mini")


def test_dg_vs_ref_tokens_and_moe_mismatch_mini4():
    _dg_vs_ref("mini4")


def test_dg_graph_step_cell_informative():
    _need_dg()
    fl = _first_light()
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl="ref", mode="graph")
        ids = _ids(16, 8)
        p50 = {}
        toks = {}
        for impl in ("ref", DG):
            gen.set_moe_impl(impl)
            gen.set_mode("graph")
            sc = fl.step_cell(gen, ids, 40, stats_from=8)
            p50[impl], toks[impl] = sc["ms_per_step"]["p50"], sc["token_ids"][:N_STEPS + 1]
        print(f"  mini4 graph step cell p50: ref {p50['ref']:.3f} ms, dg-masked {p50[DG]:.3f} ms; "
              f"first {N_STEPS + 1} tokens equal: {toks['ref'] == toks[DG]}")
        gen.reset_static()
        del gen


TESTS = [test_e8m0_conversion_exact_on_device, test_dg_static_eager_selfreplay_mini,
         test_dg_static_eager_selfreplay_mini4, test_dg_graph_equals_static_eager_mini,
         test_dg_graph_equals_static_eager_mini4, test_dg_vs_ref_tokens_and_moe_mismatch_mini,
         test_dg_vs_ref_tokens_and_moe_mismatch_mini4, test_dg_graph_step_cell_informative]

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
    sys.exit(1 if n_fail else 0)
