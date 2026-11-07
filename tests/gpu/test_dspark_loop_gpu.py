"""GPU tests of M1-T12 (runtime/dspark_loop.py DSparkLoop). Skipped without CUDA/tilelang (Mac);
tests/test_dspark_loop_cpu.py dry-runs this file's gate functions on the CPU kernel emulations (DEV "cpu", eager).

Gates (docs/HANDOFF-M1-T12-LOOP.md 4; docs/M1-DSPARK-DESIGN.md 4 V1/V3/V5), mini checkpoint with the mtp schema
(tests/mini_mtp.py "mini3": 3 draft stages, 2 target layers, a ratio-2 compressor layer):
  * V3 stream: the loop's emitted stream == the 1-row greedy stream (64 tokens), and every emitted token's fp32
    logits row == the 1-row step's logits at that position (sha256 per position), in ROW-INVARIANT MODE: torch's
    F.linear and the reference's two einsums evaluated one row at a time (`row_invariant_mode`, the design's debug
    mode; the fp8/fp4 GEMMs, act_quant, sparse_attn are row-invariant on the box per the M1-T10 census) and, for the
    GEMV variant, DSV41_GEMV_PIN_M1=1. As-is (no row-invariant mode) the stream is REPORTED, not gated: the fp32
    head / compressor cuBLAS GEMMs pick M-dependent splits (M1-T10 census) and can flip a near-tie.
  * V1 forced rejections: scripted drafts (the truth for the first n, a wrong token at n+1) force every accept count
    n in 0..5 (and 5 -> 5, 5 -> 0, both P0 parities); the accept log == the plan, the stream == greedy, every emitted
    row's logits == the 1-row logits -- the commit path (256-ring overwrite, the ratio-2 compressor commit under n)
    at short context and past the 256-ring wrap (L = 250, positions to ~300).
  * EOS: the loop stops at the 1-row greedy's EOS (finish "stop"); steps issued after `done` are no-ops (status,
    out_ids, P0, the next verify input, the compressor state, the draft rings: unchanged).
  * graph (CUDA only): 2 graphs and 1 graph == static-eager (tokens, accept log, per-position logits sha256); replay
    x2 equal; no host sync inside a step; timings recorded for both graph halves.
Real checkpoint (DSV41_MODEL_DIR set): one fixture-like prompt, 64 tokens, fast mode -- stream vs 1-row reported
(first divergence), replay x2 gated.

Run on the box (orchestrator), same line as tests/gpu/test_draft_gpu.py, repo at $W:
  W=$HOME/hermes/gb300-dsv41-decode-wt12
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv [-v /models:/models -e DSV41_MODEL_DIR=/models/DeepSeek-V4.1-...] \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_dspark_loop_gpu.py
"""
from __future__ import annotations

import contextlib
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

from runtime.refmodel import tilelang_available  # noqa: E402

DEV = "cuda"       # tests/test_dspark_loop_cpu.py dry-runs the gates on the CPU emulations (DEV "cpu")
STEP_KW = {}       # ... with {"_allow_cpu": True} (StaticStep, through Generator.static_kw)
LOOP_KW = {}       # ... with {"_allow_cpu": True} (DraftStep)
MAX_POS = 511
COUNTS = {"gates": 0, "tokens": 0, "logit_rows": 0, "steps": 0, "forced_steps": 0, "graph": 0}
REPORT = []
PLAN = [0, 1, 2, 3, 4, 5, 5, 0, 2, 5, 1, 3, 0, 4, 1, 5, 2, 0, 3, 4]  # every n, 5->5, 5->0, both P0 parities


def _need_tilelang():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def row_invariant_mode(enabled: bool = True):
    """docs/M1-DSPARK-DESIGN.md 4's row-invariant mode: F.linear and the reference's row-indexed einsums ("bs...")
    evaluated one row at a time (M = 1 calls, as the 1-row step makes them). Debug only."""
    if not enabled:
        yield
        return
    import torch.nn.functional as F_

    lin0, ein0 = F_.linear, torch.einsum

    def linear(x, w, b=None):
        K = x.size(-1)
        M = x.numel() // K
        if M <= 1:
            return lin0(x, w, b)
        xv = x.reshape(M, K)
        return torch.cat([lin0(xv[i:i + 1], w, b) for i in range(M)], 0).view(*x.shape[:-1], -1)

    def einsum(eq, *ops):
        a = ops[0]
        if eq.replace(" ", "").startswith("bs") and a.dim() >= 3 and a.size(1) > 1:
            return torch.cat([ein0(eq, a[:, i:i + 1], *ops[1:]) for i in range(a.size(1))], 1)
        return ein0(eq, *ops)

    F_.linear, torch.einsum = linear, einsum
    try:
        yield
    finally:
        F_.linear, torch.einsum = lin0, ein0


@contextlib.contextmanager
def gemv_pin_m1(enabled: bool = True):
    key = "DSV41_GEMV_PIN_M1"
    prev = os.environ.get(key)
    if enabled:
        os.environ[key] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


@contextlib.contextmanager
def _model(kind: str = "mini3"):
    """Build + load a mini checkpoint with the mtp schema on cuda, load_mtp=True -> (model, pool); `real` =
    $DSV41_MODEL_DIR."""
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    if kind == "real":
        md = os.environ.get("DSV41_MODEL_DIR")
        if not md:
            raise unittest.SkipTest("DSV41_MODEL_DIR not set")
        mp = os.environ.get("DSV41_MAP") or os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json")
        model = build_and_load(md, mp if os.path.exists(mp) else None, "cuda", load_mtp=True)
    else:
        import engram
        from mini_ckpt import MINI_CONFIG
        from mini_mtp import make_mini_checkpoint_mtp, mtp_config_kind

        engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                         MINI_CONFIG["engram_compressed_vocab_size"])
        tmp = tempfile.TemporaryDirectory()
        ck = make_mini_checkpoint_mtp(os.path.join(tmp.name, "ckpt"), cfg=mtp_config_kind(kind))
        model = build_and_load(ck["model_dir"], ck["map"], "cuda", tokenizer=object(), io_threads=4,
                               stage_bytes=1 << 20, load_mtp=True, log=lambda *a: None)
    model.temperature = 0.0
    res = model.runtime_load
    try:
        yield model, res.pool
    finally:
        torch.set_default_device("cpu")
        del model
        torch.cuda.synchronize()
        res.release()
        if kind != "real":
            tmp.cleanup()


def _toks(n, seed, hi=2000):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, hi, (n,), generator=g, device="cpu").tolist()


def _gen(model, mode="static", eos=(1,), **impls):
    from runtime.generate import Generator
    from tiny_tok import make_wordlevel_tokenizer

    V = model.head.weight.size(0)
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(V), eos_ids=list(eos), device=DEV, mode=mode,
                    max_pos=MAX_POS, **{k: v for k, v in impls.items() if v is not None})
    gen.static_kw = dict(STEP_KW)
    return gen


def _truth(gen, ids, n, stop_on_eos=False):
    """The 1-row greedy stream (token ids) and its per-position fp32-logits sha256 (Generator.greedy)."""
    gen.set_rows(1)
    gen.hidden_layers = ()
    gen.greedy(ids, n, stop_on_eos=stop_on_eos)
    return list(gen.last_meta["token_ids"]), list(gen.last_meta["step_sha256"])


def _loop(gen, max_new, n_graphs=None):
    from runtime.dspark_loop import DSparkLoop

    loop = DSparkLoop(gen, max_new, **LOOP_KW)
    if gen.mode == "graph":
        loop.capture(n_graphs=n_graphs or 2)
    return loop


def _first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


# ------------------------------------------------------------------------------------------------ V3 stream
def _gate_stream(kind="mini3", L=17, max_new=64, mode="static", n_graphs=None, rowinv=True, pin=False, **impls):
    """Loop stream == 1-row greedy (+ per-position logits) in row-invariant mode; as-is: reported. Returns res."""
    with _model(kind) as (model, pool), row_invariant_mode(rowinv), gemv_pin_m1(pin):
        V = model.head.weight.size(0)
        gen = _gen(model, mode, **impls)
        ids = _toks(L, 300 + L, min(2000, V))
        truth, sha = _truth(gen, ids, max_new)
        loop = _loop(gen, max_new, n_graphs)
        try:
            r = loop.generate(ids, max_new, stop_on_eos=False, parity=True)
            r2 = loop.generate(ids, max_new, stop_on_eos=False, parity=False)
        finally:
            loop.close()
        k = _first_div(r["token_ids"], truth)
        bad = [i for i, (x, y) in enumerate(zip(r["step_sha256"], sha)) if x != y]
        tag = (f"{kind} L={L} mode={mode}{'/' + str(n_graphs) + 'g' if mode == 'graph' else ''} "
               f"{'row-invariant' if rowinv else 'as-is'}{' gemv-pinned' if pin else ''} {impls or ''}")
        REPORT.append(f"{tag}: {len(r['token_ids'])} tokens, {r['steps']} steps, {r['tokens_per_step']:.3f} tok/step, "
                      f"stream {'== greedy' if k is None else f'diverges at {k}'}, logits rows differing "
                      f"{len(bad)}/{len(sha)}")
        assert r2["token_ids"] == r["token_ids"] and r2["accepts"] == r["accepts"], f"{tag}: fast run != parity run"
        if rowinv:
            assert k is None, f"{tag}: stream diverges at {k}: loop {r['token_ids'][k:k + 4]} vs {truth[k:k + 4]}"
            assert not bad, f"{tag}: logits rows {bad[:8]} differ from the 1-row step"
        COUNTS["gates"] += 1
        COUNTS["tokens"] += len(truth)
        COUNTS["logit_rows"] += len(sha)
        COUNTS["steps"] += r["steps"]
        print(f"  {tag}: loop == 1-row greedy ({len(truth)} tokens, {r['steps']} steps)" if k is None
              else f"  {tag}: REPORTED divergence at {k}")
        return r


# ------------------------------------------------------------------------------------------------ V1 forced
def _drafts_for(truth, n_out, n_k, V):
    """d'_j (j = 1..5) for the step that starts with n_out emitted: d_j must equal t_{j-1} = truth[n_out + j - 1] to
    be accepted; the first n_k are the truth, the rest a wrong token."""
    out = []
    for j in range(1, 6):
        x = truth[min(n_out + j - 1, len(truth) - 1)]   # past the truth: only the step after the last one asks
        out.append(x if j <= n_k else (x + 1 + j) % V if (x + 1 + j) % V != x else (x + 1) % V)
    return out


def _gate_forced(kind="mini3", L=17, max_new=48, mode="static", n_graphs=None, plan=PLAN, rowinv=True, pin=False,
                 **impls):
    """Scripted acceptor: every step accepts exactly plan[k] drafts; accept log, stream and per-row logits checked."""
    with _model(kind) as (model, pool), row_invariant_mode(rowinv), gemv_pin_m1(pin):
        V = model.head.weight.size(0)
        gen = _gen(model, mode, **impls)
        ids = _toks(L, 700 + L, min(2000, V))
        truth, sha = _truth(gen, ids, max_new + 12)
        loop = _loop(gen, max_new, n_graphs)
        try:
            b, st = loop.b, loop.st
            n_out, k, want = 1, 0, []
            loop.set_force(_drafts_for(truth, 1, plan[0], V))
            loop.start(ids, max_new, stop_on_eos=False, keep_force=True)
            rows_checked = 0
            while True:
                n_k = plan[k % len(plan)]
                nxt_out = n_out + n_k + 1
                loop.set_force(_drafts_for(truth, nxt_out, plan[(k + 1) % len(plan)], V))
                loop.step_once()
                _sync()
                n = int(b.n_last)
                assert n == n_k, (f"{kind} L={L} step {k} (n_out {n_out}, P0 {L + n_out - 1}): accepted {n}, scripted "
                                  f"{n_k}; t_out {st.t_out.tolist()} truth {truth[n_out:n_out + 6]}")
                lcs = st.logits[: n + 1].detach().to("cpu", torch.float32)
                for i in range(n + 1):
                    pos_k = n_out + i
                    if pos_k < len(sha) and pos_k < max_new:
                        import hashlib

                        h = hashlib.sha256(lcs[i].numpy().tobytes()).hexdigest()
                        assert h == sha[pos_k], f"{kind} L={L} step {k} row {i} (token {pos_k}): logits != 1-row"
                        rows_checked += 1
                want.append(n_k)
                n_out = nxt_out
                k += 1
                if bool(int(b.done)):
                    break
                assert k < max_new, "loop did not finish"
            r = b.read()
            assert r["accepts"] == want, (r["accepts"], want)
            got = r["ids"][:max_new]
            assert got == truth[:max_new], f"forced stream diverges at {_first_div(got, truth[:max_new])}"
        finally:
            loop.close()
        COUNTS["gates"] += 1
        COUNTS["forced_steps"] += k
        COUNTS["logit_rows"] += rows_checked
        hist = [want.count(i) for i in range(6)]
        print(f"  {kind} L={L} mode={mode}{'/' + str(n_graphs or 2) + 'g' if mode == 'graph' else ''}: forced "
              f"rejections, {k} steps, accept hist {hist}, positions {L}..{L + n_out - 2}: stream == greedy, "
              f"{rows_checked} logit rows == 1-row")
        return want


# ------------------------------------------------------------------------------------------------ EOS + no-op steps
def _snapshot(loop):
    st, ds = loop.st, loop.ds
    out = {"status": loop.b.status.clone(), "out_ids": loop.b.out_ids.clone(), "pos": st.pos.clone(),
           "tok": st.tok.clone(), "acc": loop.b.acc_log.clone()}
    for comp, _, _ in st._r2.values():
        out[f"kv{id(comp)}"] = comp.kv_state.clone()
        out[f"sc{id(comp)}"] = comp.score_state.clone()
    for i, s in enumerate(ds.stages):
        out[f"ring{i}"] = s.attn.window_kv_cache.clone()
    return out


def _gate_eos(kind="mini3", L=19, max_new=64, mode="static", n_graphs=None):
    with _model(kind) as (model, pool), row_invariant_mode():
        V = model.head.weight.size(0)
        gen = _gen(model, mode)
        ids = _toks(L, 900 + L, min(2000, V))
        free, _ = _truth(gen, ids, max_new)
        k0 = min(17, len(free) - 2)
        e = free[k0]
        k = free.index(e)
        gen.eos_ids = [e]
        truth, _ = _truth(gen, ids, max_new, stop_on_eos=True)
        assert len(truth) == k + 1 and truth[-1] == e
        loop = _loop(gen, max_new, n_graphs)
        try:
            r = loop.generate(ids, max_new, stop_on_eos=True, parity=False)
            assert r["token_ids"] == truth and r["finish_reason"] == "stop", (r["token_ids"], truth)
            assert r["done"]
            _sync()
            s0 = _snapshot(loop)
            for _ in range(3):
                loop.step_once()
            _sync()
            s1 = _snapshot(loop)
            diff = [name for name in s0 if not torch.equal(s0[name], s1[name])]
            assert not diff, f"steps after done changed {diff}"
        finally:
            loop.close()
        COUNTS["gates"] += 1
        print(f"  {kind} L={L}: EOS {e} at token {k}: loop stops there (finish stop, {r['steps']} live steps of "
              f"{r['steps_issued']} issued); 3 steps after done change nothing")


# ------------------------------------------------------------------------------------------------ tests
def test_loop_stream_equals_greedy_static():
    _need_tilelang()
    _gate_stream("mini3", 17, 64, mode="static")
    _gate_stream("mini3", 18, 64, mode="static")      # the other P0 parity at the first step


def test_loop_stream_as_is_reported():
    _need_tilelang()
    _gate_stream("mini3", 17, 64, mode="static", rowinv=False)


def test_loop_forced_rejections_static():
    _need_tilelang()
    _gate_forced("mini3", 17, 48, mode="static")
    _gate_forced("mini3", 250, 56, mode="static", plan=PLAN[3:] + PLAN[:3])   # past the 256-ring wrap


def test_loop_eos_and_noop_steps_static():
    _need_tilelang()
    _gate_eos("mini3", 19, 64, mode="static")


def test_loop_graph_two_and_one_equal_static_replay_no_sync():
    _need_tilelang()
    for ng in (2, 1):
        _gate_stream("mini3", 17, 64, mode="graph", n_graphs=ng)
        _gate_forced("mini3", 17, 48, mode="graph", n_graphs=ng)
    _gate_eos("mini3", 19, 64, mode="graph", n_graphs=2)
    with _model("mini3") as (model, pool), row_invariant_mode():
        gen = _gen(model, "static")
        ids = _toks(17, 41)
        loop = _loop(gen, 64)
        rs = loop.generate(ids, 64, stop_on_eos=False, parity=True)
        loop.close()
        gen.set_mode("graph")
        loop = _loop(gen, 64, 2)
        try:
            g1 = loop.generate(ids, 64, stop_on_eos=False, parity=True)
            g2 = loop.generate(ids, 64, stop_on_eos=False, parity=True)
            f1 = loop.generate(ids, 64, stop_on_eos=False, parity=False, timing=True)
            for x, what in ((g1, "graph #1"), (g2, "graph #2")):
                assert x["token_ids"] == rs["token_ids"] and x["accepts"] == rs["accepts"], what
                assert x["step_sha256"] == rs["step_sha256"], f"{what}: logits differ from static-eager"
            assert f1["token_ids"] == rs["token_ids"] and f1["accepts"] == rs["accepts"]
            assert f1["step_ms"] and f1["verify_ms"] and f1["draft_ms"] and len(f1["step_ms"]) == f1["steps"]
            loop.capture(n_graphs=1)
            g3 = loop.generate(ids, 64, stop_on_eos=False, parity=True)
            assert g3["token_ids"] == rs["token_ids"] and g3["step_sha256"] == rs["step_sha256"], "1 graph"
            loop.start(ids, 64, stop_on_eos=False)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                for _ in range(3):
                    loop.step_once()
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
        finally:
            loop.close()
        COUNTS["graph"] += 1
        print(f"  graph (2 and 1) == static-eager: tokens, accepts, {len(rs['step_sha256'])} logit rows; replay x2; "
              f"no host sync in 3 replays; step {sum(f1['step_ms']) / len(f1['step_ms']):.3f} ms (verify "
              f"{sum(f1['verify_ms']) / len(f1['verify_ms']):.3f} + draft {sum(f1['draft_ms']) / len(f1['draft_ms']):.3f})")


def test_loop_stream_gemv_pinned():
    _need_tilelang()
    _gate_stream("mini3", 17, 64, mode="graph", n_graphs=2, pin=True, fp8_impl="gemv")
    _gate_forced("mini3", 18, 48, mode="graph", n_graphs=2, pin=True, fp8_impl="gemv")


def test_loop_real():
    """Real checkpoint, fast mode (no row-invariant mode): stream vs 1-row REPORTED; replay x2 gated."""
    _need_tilelang()
    if not os.environ.get("DSV41_MODEL_DIR"):
        raise unittest.SkipTest("DSV41_MODEL_DIR not set")
    with _model("real") as (model, pool), gemv_pin_m1():
        from runtime.generate import Generator

        gen = Generator(model, os.environ["DSV41_MODEL_DIR"], mode="graph", max_pos=4095)
        ids = gen.encode([{"role": "user", "content": "Write a Python function that reverses a linked list."}])
        truth, _ = _truth(gen, ids, 64)
        loop = _loop(gen, 64, 2)
        try:
            a = loop.generate(ids, 64, stop_on_eos=False, parity=False)
            b = loop.generate(ids, 64, stop_on_eos=False, parity=False)
        finally:
            loop.close()
        assert a["token_ids"] == b["token_ids"] and a["accepts"] == b["accepts"], "real: replay differs"
        k = _first_div(a["token_ids"], truth)
        REPORT.append(f"real: stream vs 1-row {'equal' if k is None else f'first divergence at {k}'}; "
                      f"{a['tokens_per_step']:.3f} tok/step; step {sum(a['step_ms']) / len(a['step_ms']):.2f} ms")
        COUNTS["gates"] += 1


TESTS = [test_loop_stream_equals_greedy_static, test_loop_stream_as_is_reported, test_loop_forced_rejections_static,
         test_loop_eos_and_noop_steps_static, test_loop_graph_two_and_one_equal_static_replay_no_sync,
         test_loop_stream_gemv_pinned, test_loop_real]

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
    for line in REPORT:
        print(f"  note: {line}")
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"comparisons executed: {COUNTS}")
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
