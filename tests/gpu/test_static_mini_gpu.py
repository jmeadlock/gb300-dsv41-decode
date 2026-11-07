"""GPU tests of T8 (runtime/static_step.py) on synthetic mini checkpoints. Skipped without CUDA/tilelang (Mac).

  * static-eager == reference-eager, `torch.equal` on the fp32 logits of the prefill + 10 decode steps (both parities),
    with the StaticStep already built (so the static-MoE patch is proven inert for the eager path), and the same
    per-layer cold-expert calls (routing + tier lookup agree);
  * the static step makes no host sync (`torch.cuda.set_sync_debug_mode("error")`);
  * graph replay == static-eager over the same steps, twice, and after a fills-on recapture (L2's uninitialized-read
    detector: NaN fills recorded into the graph);
  * greedy records + fp32-logit digests identical in eager / static / graph mode, an eager replay after graph use still
    identical (capture's dummy prefill leaks nothing), the step marker counts steps inside the graph, first_light's
    step cell in graph mode gives greedy's tokens;
  * the topk-truncated regime (reachable compressed positions > index_topk, design risk 6) as its own test.

Two checkpoints: tests/mini_ckpt.py's MINI_CONFIG (2 layers: ratio 0, ratio 2 kv+index source; 4 cold experts per
layer in registered host memory) and MINI4 (4 layers: ratios 0/2/1/1, layer 2 = ratio-1 kv + index + candidate source,
layer 3 = candidate consumer reading layer 2's index K, 4 active experts so the fp32 slot-order sum is observable --
at topk 2, (0+a)+b == (0+b)+a exactly), index_topk / candidate_topk_blocks set so every test prompt stays in the
design's exact regime (reachable <= k).

Run on the box (orchestrator), from the repo checkout, inside the library image with the GB300 by CDI (same line as
tests/gpu/test_generate_mini_gpu.py, repo at $W):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_static_mini_gpu.py

(`python3 tests/gpu/test_static_mini_gpu.py` runs every test without pytest; no checkpoint mount needed.)
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime.refmodel import tilelang_available  # noqa: E402

N_STEPS = 10
MAX_POS = 255


def _need_tilelang():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mini4_config():
    from mini_ckpt import MINI_CONFIG

    cfg = dict(MINI_CONFIG)
    cfg.update({"n_layers": 4, "compress_ratios": [0, 2, 1, 1, 0], "kv_source_layers": [1, 2],
                "index_source_layers": [1, 2, 3], "candidate_source_layer": 2, "candidate_topk_blocks": 64,
                "candidate_block_size": 8, "index_topk": 64, "dspark_target_layer_ids": [3],
                "n_activated_experts": 4})  # topk >= 3: fp32 accumulation order is observable
    hot = {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6], "2": [0, 1, 2, 3], "3": [4, 5, 6, 7]}
    return cfg, hot


@contextlib.contextmanager
def _model(kind: str = "mini"):
    """Build + load a mini checkpoint on cuda; yields (model, tokdir); releases the host buffers afterwards."""
    from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference
    from tiny_tok import make_wordlevel_tokenizer

    import_reference()
    import engram

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "mini4":
            cfg, hot = _mini4_config()
            ck = make_mini_checkpoint(os.path.join(tmp, "ckpt"), cfg=cfg)
            mp = hot
        else:
            ck = make_mini_checkpoint(os.path.join(tmp, "ckpt"))
            mp = ck["map"]
        model = build_and_load(ck["model_dir"], mp, "cuda", tokenizer=object(), io_threads=4, stage_bytes=1 << 20,
                               log=lambda *a: None)
        tokdir = os.path.join(tmp, "tok")
        make_wordlevel_tokenizer(2048, out_dir=tokdir)
        res = model.runtime_load
        try:
            yield model, tokdir
        finally:
            torch.set_default_device("cpu")
            del model
            torch.cuda.synchronize()
            res.release()


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 2000, (n,), generator=g).tolist()


def _run(gen, mode, ids, n=N_STEPS):
    """prefill + n decode steps in `mode`; fp32 logits clones per position, tokens, decode-only cold calls/layer."""
    gen.set_mode(mode)
    out = [gen.prefill(ids).float().clone()]
    toks = [int(gen.next_token_tensor().reshape(-1)[0])]
    c0 = gen.cold_totals()
    pos = len(ids)
    for _ in range(n):
        out.append(gen.step(toks[-1], pos).float().clone())
        toks.append(int(gen.next_token_tensor().reshape(-1)[0]))
        assert toks[-1] == int(out[-1].argmax()), (mode, pos)
        pos += 1
    c1 = gen.cold_totals()
    cold = None if c0 is None else [b - a for a, b in zip(c0[3], c1[3])]
    return out, toks, cold


def _assert_equal_runs(a, b, what):
    la, ta, _ = a
    lb, tb, _ = b
    bad = [i for i, (x, y) in enumerate(zip(la, lb)) if not torch.equal(x, y)]
    assert len(la) == len(lb) and not bad, (f"{what}: logits differ at positions {bad} (0 = prefill); max |d| at first "
                                            f"{(la[bad[0]] - lb[bad[0]]).abs().max().item() if bad else 0}")
    assert ta == tb, (what, ta, tb)


def _gen(model, tokdir, **kw):
    from runtime.generate import Generator, set_determinism

    set_determinism(0)
    return Generator(model, tokdir, max_pos=MAX_POS, **kw)


# ------------------------------------------------------------------------------------------------ tests
def _static_vs_reference(kind):
    _need_tilelang()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir)
        ids = _ids(16, 5)
        gen.static_step()  # built (patch installed, MoEs bound) BEFORE the eager run: the patch must be inert there
        ref = _run(gen, "eager", ids)
        st = _run(gen, "static", ids)
        assert {p % 2 for p in range(len(ids), len(ids) + N_STEPS)} == {0, 1}
        _assert_equal_runs(st, ref, f"{kind}: static-eager vs reference-eager")
        assert st[2] == ref[2] and sum(ref[2]) > 0, (st[2], ref[2])  # same cold calls per layer, and some exist
        print(f"  {kind}: static-eager == reference-eager over prefill + {N_STEPS} steps; cold/layer {st[2]}")
        gen.reset_static()
        del gen


def test_static_eager_equals_reference_mini():
    _static_vs_reference("mini")


def test_static_eager_equals_reference_mini4_ratio1_candidates():
    _static_vs_reference("mini4")


def test_static_step_has_no_host_sync():
    _need_tilelang()
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, mode="static")
        ids = _ids(16, 6)
        gen.prefill(ids)
        tok = int(gen.next_token_tensor().reshape(-1)[0])
        gen.step(tok, len(ids))  # first call: TileLang/Triton JIT outside the check
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for p in (len(ids) + 1, len(ids) + 2):  # both parities
                gen.step(gen.next_token_tensor(), p)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        gen.reset_static()
        del gen


def _graph_vs_static(kind):
    _need_tilelang()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir)
        ids = _ids(16, 7)
        st = _run(gen, "static", ids)
        g1 = _run(gen, "graph", ids)  # prefill() captures first (fills off)
        assert gen.static.captured and gen.static.captured_fills is False
        _assert_equal_runs(g1, st, f"{kind}: graph vs static-eager")
        g2 = _run(gen, "graph", ids)
        _assert_equal_runs(g2, g1, f"{kind}: graph replay#2 vs #1")
        gen.capture(fills=True)  # NaN fills recorded into the graph: any uninitialized read would show
        g3 = _run(gen, "graph", ids)
        _assert_equal_runs(g3, g1, f"{kind}: fills-on capture vs fills-off")
        gen.reset_static()  # relaunch
        g4 = _run(gen, "graph", ids)
        _assert_equal_runs(g4, g1, f"{kind}: relaunch vs replay#1")
        assert g1[2] == st[2], (g1[2], st[2])
        print(f"  {kind}: graph == static-eager; replay x2, fills-on recapture, relaunch all equal")
        gen.reset_static()
        del gen


def test_graph_replay_equals_static_eager_mini():
    _graph_vs_static("mini")


def test_graph_replay_equals_static_eager_mini4():
    _graph_vs_static("mini4")


def test_greedy_records_all_modes_marker_and_step_cell():
    _need_tilelang()
    from runtime.generate import dump_record

    fl = _first_light()
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, marker=True)
        ids = _ids(16, 8)
        recs, digs = {}, {}
        for mode in ("eager", "static", "graph", "eager"):
            gen.set_mode(mode)
            if mode == "graph":
                gen._ensure_captured()  # warm-ups also run the marker; count replays only
                m0 = gen.static.marker_count()
            rec = gen.greedy(ids, N_STEPS + 1, tag="m", prompt="m", stop_on_eos=False)
            if mode == "graph":
                assert gen.static.marker_count() - m0 == N_STEPS, (gen.static.marker_count(), m0)
            if mode in recs:
                assert dump_record(rec) == dump_record(recs[mode]), f"{mode} replay after graph use differs"
            recs[mode], digs[mode] = rec, gen.last_meta["logits_sha256"]
        assert dump_record(recs["eager"]) == dump_record(recs["static"]) == dump_record(recs["graph"])
        assert digs["eager"] == digs["static"] == digs["graph"]
        greedy_toks = gen.last_meta["token_ids"]
        gen.set_mode("graph")
        sc = fl.step_cell(gen, ids, 20, stats_from=4)
        assert sc["token_ids"][:N_STEPS + 1] == greedy_toks, (sc["token_ids"], greedy_toks)
        c = sc["cold"]
        assert c["routed_calls_per_step"]["min"] == c["routed_calls_per_step"]["max"] == 4 * 4, c  # 4 layers x topk 4
        print(f"  greedy equal in all modes; graph step cell p50 {sc['ms_per_step']['p50']:.2f} ms")
        gen.reset_static()
        del gen


def test_static_topk_truncated_regime_risk6():
    """Reachable compressed positions > index_topk (mini: ratio 2, topk 16; a 48-token prompt -> 24..29 reachable):
    the selection depends on score bits, which come from a W-wide einsum here and an R-wide one in the reference, and
    on topk's tie resolution (design 2.1 #9, risk 6). Equality here is evidence, not a guarantee, for long context."""
    _need_tilelang()
    with _model("mini") as (model, tokdir):
        gen = _gen(model, tokdir)
        ids = _ids(48, 9)
        ref = _run(gen, "eager", ids)
        st = _run(gen, "static", ids)
        _assert_equal_runs(st, ref, "truncated-topk regime: static-eager vs reference-eager (risk 6)")
        g = _run(gen, "graph", ids)
        _assert_equal_runs(g, st, "truncated-topk regime: graph vs static-eager")
        gen.reset_static()
        del gen


TESTS = [test_static_eager_equals_reference_mini, test_static_eager_equals_reference_mini4_ratio1_candidates,
         test_static_step_has_no_host_sync, test_graph_replay_equals_static_eager_mini,
         test_graph_replay_equals_static_eager_mini4, test_greedy_records_all_modes_marker_and_step_cell,
         test_static_topk_truncated_regime_risk6]

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
