"""GPU tests of M1-T10 (runtime/static_step.py rows=6, runtime/ref_multirow.py, runtime/moe_rows.py, runtime/row_census.py)
on synthetic mini checkpoints. Skipped without CUDA/tilelang (Mac). docs/HANDOFF-M1-T10-VERIFY6.md "Gates":

  * THE gate: the 6-row static step run eagerly == the reference's own 6-row decode forward (`forward_rows`, the
    reference forward(tokens[:, a:a+6], a) under ref_multirow's multi-row spellings), torch.equal on the fp32 logits
    [6, V], main_hidden and the caches the step writes, with every impl `ref` (moe/fp8/glue/attn) and with attn
    `bounded`; both P0 parities (prefill 17 / 18), mini (ratio 0/2) and mini4 (ratio 0/2/1/1 + candidates);
  * the 256-ring wrapped (P0 300 / 301), 6-row and 1-row (the 1-row step's ring change proven inert past the wrap),
    in the exact indexer regime (`w` checkpoints: index_topk 512, every reachable position kept);
  * graph == static-eager; replay twice; fills-on recapture; relaunch (fresh StaticStep + capture); no host sync;
  * the fast impls (moe dg-masked, fp8 gemv, attn flashmla, glue fused -> ref at 6 rows) drift-graded vs the oracle
    (reported: bit-equal rows, max |dlogit|, argmax agreement; gated only on graph == static-eager);
  * V0: the row-invariance census table (runtime/row_census.py, the real kernels, layer-0 shapes) and the step-level
    V0 table -- the 6-row step's row j vs the 1-row step at P0 + j, same impls (ref; fast with the GEMV's tuned M=6
    configs; fast with the M=1 config pinned), plus the ref step under row_invariant_mode (which families remain).

Run on the box (orchestrator), same line as tests/gpu/test_static_mini_gpu.py, repo at $W:

  W=$HOME/hermes/gb300-dsv41-decode-wt10
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_static_rows_gpu.py

(`python3 tests/gpu/test_static_rows_gpu.py` runs every test without pytest and prints the tables.)
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

ROWS = 6
MAX_POS = 511
DEV = "cuda"      # tests/test_static_rows_cpu.py dry-runs this file's logic on the CPU emulations (DEV "cpu")
STATIC_KW = {}    # ... with {"_allow_cpu": True}
FAST = dict(moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused", attn_impl="flashmla")


def _need_tilelang():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _config(kind: str):
    """(cfg or None, hot map or None). mini / mini4 as T8's test; the `w` kinds keep every reachable compressed
    position at P >= 300 (index_topk 512 >= (P+1)/r; candidates cover all blocks) -- the exact regime (#9)."""
    from mini_ckpt import MINI_CONFIG

    if kind in ("mini4", "mini4w"):
        cfg = dict(MINI_CONFIG)
        cfg.update({"n_layers": 4, "compress_ratios": [0, 2, 1, 1, 0], "kv_source_layers": [1, 2],
                    "index_source_layers": [1, 2, 3], "candidate_source_layer": 2, "candidate_topk_blocks": 64,
                    "candidate_block_size": 8, "index_topk": 64, "dspark_target_layer_ids": [3],
                    "n_activated_experts": 4})
        if kind == "mini4w":
            cfg.update({"index_topk": 512, "candidate_topk_blocks": 64, "candidate_block_size": 8})
        hot = {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6], "2": [0, 1, 2, 3], "3": [4, 5, 6, 7]}
        return cfg, hot
    if kind == "miniw":
        return dict(MINI_CONFIG, index_topk=512), None
    return None, None


@contextlib.contextmanager
def _model(kind: str = "mini"):
    """Build + load a mini checkpoint on cuda -> (model, pool); releases the host buffers afterwards."""
    from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    import engram

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    cfg, hot = _config(kind)
    with tempfile.TemporaryDirectory() as tmp:
        ck = make_mini_checkpoint(os.path.join(tmp, "ckpt"), cfg=cfg) if cfg else \
            make_mini_checkpoint(os.path.join(tmp, "ckpt"))
        mp = hot if hot is not None else ck["map"]
        model = build_and_load(ck["model_dir"], mp, "cuda", tokenizer=object(), io_threads=4, stage_bytes=1 << 20,
                               log=lambda *a: None)
        model.temperature = 0.0
        res = model.runtime_load
        try:
            yield model, res.pool
        finally:
            torch.set_default_device("cpu")
            del model
            torch.cuda.synchronize()
            res.release()


def _toks(L, n, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(2, 2000, (L + n,), generator=g, device="cpu").tolist()
    return t[:L], t[L:]


def _prefill(model, ids):
    with torch.inference_mode():
        model(torch.tensor([ids], device=DEV), 0)


def _state(model):
    out = {}
    for i, l in enumerate(model.layers):
        a = l.attn
        if a.compressor is not None and a.compressor.compress_ratio > 1:
            out[f"L{i}.kv_state"] = a.compressor.kv_state.clone()
            out[f"L{i}.score_state"] = a.compressor.score_state.clone()
        if a.is_kv_source:
            out[f"L{i}.ckv"] = a.compress_kv_cache.clone()
        if a.indexer is not None and a.indexer.owns_k:
            out[f"L{i}.k_cache"] = a.indexer.k_cache.clone()
    if model.engram_hash is not None:
        out["engram"] = model.engram_hash.cache.clone()
    return out


def _oracle(model, ids, toks, n, hid=()):
    from runtime.ref_multirow import forward_rows

    _prefill(model, ids)
    lg, hs = [], []
    for s in range(n):
        _, l, h = forward_rows(model, torch.tensor([toks[s * ROWS:(s + 1) * ROWS]], device=DEV),
                               len(ids) + s * ROWS, hid)
        lg += [l[j].float().clone() for j in range(ROWS)]
        hs.append(None if h is None else h.clone())
    _sync()
    return lg, hs, _state(model)


def _static(model, st, ids, toks, n, graph=False):
    _prefill(model, ids)
    lg, hs = [], []
    with torch.inference_mode():
        for s in range(n):
            l = st.step(toks[s * st.rows:(s + 1) * st.rows] if st.rows > 1 else toks[s], len(ids) + s * st.rows,
                        graph=graph)
            l = l.reshape(st.rows, -1)
            lg += [l[j].float().clone() for j in range(st.rows)]
            hs.append(None if getattr(st, "main_hidden", None) is None else st.main_hidden.clone())
    _sync()
    return lg, hs, _state(model)


def _seq1(model, st1, ids, toks, n_pos):
    """The 1-row static step over the same teacher-forced tokens: logits at positions L .. L + n_pos - 1."""
    _prefill(model, ids)
    out = []
    with torch.inference_mode():
        for i in range(n_pos):
            out.append(st1.step(toks[i], len(ids) + i, graph=False)[0].float().clone())
    _sync()
    return out


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def _eq_rows(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)]


def _assert_rows(a, b, what):
    bad = _eq_rows(a, b)
    assert len(a) == len(b) and not bad, (f"{what}: rows {bad[:12]} differ ({len(bad)}/{len(a)}); max |d| at first "
                                          f"{(a[bad[0]] - b[bad[0]]).abs().max().item() if bad else 0}")


def _drift(a, b):
    d = max(float((x - y).abs().max()) for x, y in zip(a, b))
    agree = sum(int(x.argmax()) == int(y.argmax()) for x, y in zip(a, b))
    return len(a) - len(_eq_rows(a, b)), d, agree


# ------------------------------------------------------------------------------------------------ the gate
def _gate(kind, L, attn_impl="ref", n=4):
    from runtime.static_step import StaticStep

    with _model(kind) as (model, pool):
        hid = (len(model.layers) - 1,)
        ids, toks = _toks(L, ROWS * n, 100 + L)
        st = StaticStep(model, pool, max_pos=MAX_POS, rows=ROWS, hidden_layers=hid, attn_impl=attn_impl, **STATIC_KW)
        o_lg, o_h, o_s = _oracle(model, ids, toks, n, hid)
        s_lg, s_h, s_s = _static(model, st, ids, toks, n)
        _assert_rows(s_lg, o_lg, f"{kind} L={L} attn {attn_impl}: 6-row static-eager vs reference 6-row forward")
        assert all(torch.equal(a, b) for a, b in zip(s_h, o_h)), "main_hidden"
        for k in o_s:
            assert torch.equal(s_s[k], o_s[k]), (kind, L, k)
        print(f"  {kind} P0 {L}..{L + ROWS * (n - 1)} (parity {L % 2}) attn {attn_impl}: 6-row static-eager == "
              f"reference forward_rows, {len(s_lg)} rows x V, main_hidden, caches")
        st.detach()


def test_static6_equals_reference_mini_both_parities():
    _need_tilelang()
    for L in (17, 18):
        _gate("mini", L)


def test_static6_equals_reference_mini4_both_parities():
    _need_tilelang()
    for L in (17, 18):
        _gate("mini4", L)


def test_static6_equals_reference_attn_bounded():
    _need_tilelang()
    for kind in ("mini", "mini4"):
        _gate(kind, 18, attn_impl="bounded")


def test_static6_ring_wrapped_both_parities():
    _need_tilelang()
    for kind in ("mini4w", "miniw"):
        for L in (300, 301):
            _gate(kind, L, n=3)


def test_static1_ring256_past_wrap_equals_reference():
    """#20: the 1-row step on the 256-ring at positions 300..311 == the reference eager (exact regime)."""
    _need_tilelang()
    from runtime.static_step import StaticStep

    with _model("mini4w") as (model, pool):
        st = StaticStep(model, pool, max_pos=MAX_POS, **STATIC_KW)
        ids, _ = _toks(300, 0, 7)
        ref, toks = [], []
        with torch.inference_mode():
            out, lg, _ = model(torch.tensor([ids], device=DEV), 0)
            toks.append(int(out.reshape(-1)[0]))
            for k in range(12):
                out, lg, _ = model(torch.tensor([[toks[-1]]], device=DEV), 300 + k)
                ref.append(lg[0].float().clone())
                toks.append(int(out.reshape(-1)[0]))
        got = _static(model, st, ids, toks, 12)[0]
        _assert_rows(got, ref, "1-row static vs reference past the ring wrap")
        print("  1-row static == reference at positions 300..311 (both rings wrapped)")
        st.detach()


# ------------------------------------------------------------------------------------------------ graph
def test_static6_graph_replay_relaunch_no_sync():
    _need_tilelang()
    from runtime.static_step import StaticStep

    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        ids, toks = _toks(17, ROWS * 4, 21)
        st = StaticStep(model, pool, max_pos=MAX_POS, rows=ROWS, hidden_layers=hid)
        e = _static(model, st, ids, toks, 4)
        st.capture()
        g1 = _static(model, st, ids, toks, 4, graph=True)
        _assert_rows(g1[0], e[0], "6-row graph vs static-eager")
        assert all(torch.equal(a, b) for a, b in zip(g1[1], e[1]))
        g2 = _static(model, st, ids, toks, 4, graph=True)
        _assert_rows(g2[0], g1[0], "6-row graph replay #2 vs #1")
        st.capture(fills=True)
        g3 = _static(model, st, ids, toks, 4, graph=True)
        _assert_rows(g3[0], g1[0], "6-row fills-on recapture vs fills-off")
        st.detach()
        st = StaticStep(model, pool, max_pos=MAX_POS, rows=ROWS, hidden_layers=hid)  # relaunch
        st.capture()
        g4 = _static(model, st, ids, toks, 4, graph=True)
        _assert_rows(g4[0], g1[0], "6-row relaunch vs replay #1")
        # no host sync inside the eager step (after a first call outside the check: JIT)
        _prefill(model, ids)
        with torch.inference_mode():
            st.step(toks[:ROWS], len(ids), graph=False)
            torch.cuda.synchronize()
            dt = torch.tensor(toks[ROWS:3 * ROWS], device=DEV).view(2, ROWS)
            torch.cuda.set_sync_debug_mode("error")
            try:
                st.step(dt[0], len(ids) + ROWS, graph=False)
                st.step(dt[1], len(ids) + 2 * ROWS, graph=False)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        print("  6-row graph == static-eager; replay x2, fills-on recapture, relaunch equal; no host sync")
        st.detach()


# ------------------------------------------------------------------------------------------------ fast impls
def _try_step(model, pool, **kw):
    from runtime.static_step import StaticStep

    try:
        return StaticStep(model, pool, max_pos=MAX_POS, **kw, **STATIC_KW), None
    except Exception as e:  # an impl that cannot build here (no deep_gemm / FlashMLA .so) is reported, not failed
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"


def test_static6_fast_impls_drift_graded():
    _need_tilelang()
    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        ids, toks = _toks(18, ROWS * 4, 33)
        o_lg, _, _ = _oracle(model, ids, toks, 4, hid)
        print(f"  {'impls (6 rows)':52s} {'bit-equal':>9s} {'max|dlogit|':>12s} {'argmax':>7s}  graph==eager")
        combos = [{"moe_impl": "dg-masked"}, {"fp8_impl": "gemv"}, {"attn_impl": "flashmla"}, {"glue_impl": "fused"},
                  dict(FAST)]
        for kw in combos:
            st, why = _try_step(model, pool, rows=ROWS, hidden_layers=hid, **kw)
            label = "+".join(f"{k.split('_')[0]} {v}" for k, v in kw.items())
            if st is None:
                print(f"  {label:52s} n/a ({why})")
                continue
            e = _static(model, st, ids, toks, 4)
            st.capture()
            g = _static(model, st, ids, toks, 4, graph=True)
            _assert_rows(g[0], e[0], f"{label}: graph vs static-eager")
            eq, d, agree = _drift(e[0], o_lg)
            note = f" ({st.glue_note})" if getattr(st, "glue_note", None) else ""
            print(f"  {label:52s} {eq:>5d}/{len(o_lg):<3d} {d:12.4e} {agree:>3d}/{len(o_lg):<3d}  yes{note}")
            st.detach()


# ------------------------------------------------------------------------------------------------ V0
def test_v0_census_and_step_level_table():
    """Not gated: the tables are the deliverable (they decide bitwise vs drift-graded gating of the 6-row graph)."""
    _need_tilelang()
    from runtime.row_census import PIN_ENV, census, census_table, row_invariant_mode

    with _model("mini4") as (model, pool):
        recs = census(model, rows=ROWS)
        for line in census_table(recs):
            print("  " + line)
        ids, toks = _toks(18, ROWS * 4, 44)
        n_pos = ROWS * 4
        print(f"  STEP-LEVEL V0: 6-row step row j vs the 1-row step at P0+j, same impls, {n_pos} positions")
        print(f"  {'impls':44s} {'bit-equal':>9s} {'max|dlogit|':>12s} {'argmax':>7s}")
        cases = [("ref", {}, None, False), ("ref, row_invariant_mode", {}, None, True),
                 ("fast, GEMV tuned M=6 cfg", dict(FAST), "0", False),
                 ("fast, GEMV M=1 cfg pinned", dict(FAST), "1", False)]
        for label, kw, pin, rim in cases:
            prev = os.environ.get(PIN_ENV)
            if pin is not None:
                os.environ[PIN_ENV] = pin
            try:
                cm = row_invariant_mode() if rim else contextlib.nullcontext()
                with cm:
                    kw1 = dict(kw, moe_impl="dg-masked") if kw.get("moe_impl") else kw
                    s1, why1 = _try_step(model, pool, **kw1)
                    s6, why6 = _try_step(model, pool, rows=ROWS, **kw)
                    if s1 is None or s6 is None:
                        print(f"  {label:44s} n/a ({why1 or why6})")
                        continue
                    a1 = _seq1(model, s1, ids, toks, n_pos)
                    a6 = _static(model, s6, ids, toks, 4)[0]
                    eq, d, agree = _drift(a6, a1)
                    print(f"  {label:44s} {eq:>5d}/{n_pos:<3d} {d:12.4e} {agree:>3d}/{n_pos:<3d}")
                    s1.detach()
                    s6.detach()
            finally:
                if pin is not None:
                    if prev is None:
                        os.environ.pop(PIN_ENV, None)
                    else:
                        os.environ[PIN_ENV] = prev


# ------------------------------------------------------------------------------------------------ first light cells
def test_first_light_rows_cells_graph(mode="graph"):
    """bench/first_light.py's --rows 6 pieces in graph mode on mini4: the 1-row cell (6n steps) feeds the 6-row
    teacher-forced cell (n steps); its argmaxes vs the 1-row greedy tokens; teacher-forced records replay
    byte-identically; R6/R1 printed (mini shapes: launch-bound, not the real model's number)."""
    _need_tilelang()
    import importlib.util

    from runtime.generate import Generator, dump_record, set_determinism
    from tiny_tok import make_wordlevel_tokenizer

    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    n = 20
    with _model("mini4") as (model, pool):
        if DEV == "cuda":
            set_determinism(0)
        factory = None
        if STATIC_KW:  # the CPU dry run: StaticStep needs _allow_cpu (Generator's own build passes no extras)
            from runtime.static_step import StaticStep

            factory = lambda m, **kw: StaticStep(m, pool, max_pos=MAX_POS, **kw, **STATIC_KW)  # noqa: E731
        gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device=DEV, mode=mode,
                        max_pos=MAX_POS, static_factory=factory)
        ids, _ = _toks(16, 0, 8)
        sc1 = fl.step_cell(gen, ids, ROWS * n, stats_from=4)
        gen.set_rows(ROWS)
        scr = fl.step_cell_rows(gen, ids, sc1["token_ids"], n, stats_from=4)
        recs = [gen.teacher_forced(ids, sc1["token_ids"][:25]) for _ in range(2)]
        assert dump_record(recs[0]) == dump_record(recs[1]), "teacher-forced self-replay"
        r1 = sc1["ms_per_step"]["mean"]
        r6 = scr["ms_per_step"]["mean"]
        print(f"  first-light cells ({mode}): 1-row {r1:.3f} ms/step, 6-row {r6:.3f} ms/step -> R6/R1 {r6 / r1:.3f} "
              f"(mini); 6-row argmax vs 1-row greedy {scr['argmax_agree']}/{scr['argmax_compared']}; tf replay equal")
        gen.reset_static()


TESTS = [test_static6_equals_reference_mini_both_parities, test_static6_equals_reference_mini4_both_parities,
         test_static6_equals_reference_attn_bounded, test_static6_ring_wrapped_both_parities,
         test_static1_ring256_past_wrap_equals_reference, test_static6_graph_replay_relaunch_no_sync,
         test_static6_fast_impls_drift_graded, test_v0_census_and_step_level_table,
         test_first_light_rows_cells_graph]

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
