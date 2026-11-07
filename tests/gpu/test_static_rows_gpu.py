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

M1-T10-FIX (docs/HANDOFF-M1-T10-FIX.md):
  * Part A: every reference forward here goes through runtime.refmodel.reference_forward (round 1's 8 failures:
    this file ran `model(...)` with the global default device still CPU, so model.py:418-423's bare torch.arange
    built a CPU `topk_idxs` for the CUDA-targeted TileLang sparse_attn; reference generate.py:134 / Generator set
    the default to cuda, this file never did). `_preflight` (in `_model`, before any test's first forward): JIT
    cache targets, the window indices under a CPU-seeded lru cache, one direct reference sparse_attn call on cuda.
    The final line counts the bit-equal comparisons that actually executed.
  * D row: 6-row dg-inplace == 6-row dg-masked (torch.equal: logits, main_hidden, caches; graph == eager);
  * G row: 6-row fused glue == 6-row ref glue on the fast impls (dg-inplace and dg-masked); M1-T10-PORT: every
    ported cluster on (fused_glue.ROWS_CLUSTERS), its build-time self-check at 6 rows equal, and each alone off.
M1-T10-FIX2 (docs/HANDOFF-M1-T10-FIX2.md): the D row at spare_slots = 8 (compact cold index, the real Triton
  staging kernel), no overflow; forced overflow (spare_slots = 1): the device counter == the host expectation from
  each eager step's routing, and the graph replay trips it identically.

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

from runtime.refmodel import reference_device, reference_forward, tilelang_available  # noqa: E402

ROWS = 6
MAX_POS = 511
DEV = "cuda"      # tests/test_static_rows_cpu.py dry-runs this file's logic on the CPU emulations (DEV "cpu")
STATIC_KW = {}    # ... with {"_allow_cpu": True}
FAST = dict(moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused", attn_impl="flashmla")
FAST_INPLACE = dict(FAST, moe_impl="dg-inplace")  # M1-T10-FIX: the 6-row fast path
# M1-T10-FIX: how many bit-equal comparisons actually executed (the round-1 run reached none of them)
COUNTS = {"logit_rows": 0, "main_hidden": 0, "caches": 0, "gates": 0, "g_clusters": 0}


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
            _preflight(model)
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
        reference_forward(model, torch.tensor([ids], device=DEV), 0)  # never depends on the global default device


def _jit_cache_targets(jit_obj, depth: int = 0) -> list:
    """str(target) of every compiled kernel object reachable from a tilelang.jit object's attribute dicts (best
    effort, version-agnostic: it looks for values carrying a `target`; [] = nothing compiled yet or not exposed)."""
    out, seen = [], set()

    def walk(o, d):
        if id(o) in seen or d > 3:
            return
        seen.add(id(o))
        t = getattr(o, "target", None)
        if t is not None and not callable(t) and d > 0 and hasattr(o, "torch_function"):  # a compiled JITKernel
            out.append(str(t))
            return
        vals = []
        if isinstance(o, dict):
            vals = list(o.values())
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            vals = [v for k, v in vars(o).items() if not k.startswith("__")]
        for v in vals:
            if isinstance(v, (dict,)) or hasattr(v, "target") or (hasattr(v, "__dict__") and "cache" in type(v).__name__.lower()):
                walk(v, d + 1)

    walk(jit_obj, depth)
    return out


def _preflight(model):
    """M1-T10-FIX Part A guard, before any reference forward of a test. Round 1's failure: `topk_idxs` reached the
    CUDA-targeted reference sparse_attn as a CPU tensor, because this file ran `model(...)` with the global default
    device still CPU (model.py:418-423 builds the window indices with bare torch.arange; @lru_cache(1) can replay such
    an entry) -- reference generate.py:134 / runtime.generate.Generator set the default to cuda, this file never did.
    Checks: (1) every introspectable sparse_attn JIT cache entry is CUDA-targeted; (2) the window indices built
    under refmodel.reference_device land on cuda even with the lru cache seeded under a CPU default; (3) one direct
    reference sparse_attn call at layer 0's (heads, head_dim, scale) on CUDA tensors -- the cache entry the forward
    uses -- runs (a CPU-targeted kernel would raise here, with this message, not inside a test)."""
    if DEV != "cuda":
        return
    R = sys.modules["model"]
    K = sys.modules["kernel"]
    try:  # best effort: the direct call below is the check that cannot be fooled
        targets = _jit_cache_targets(K.sparse_attn_kernel)
    except Exception as e:  # an exotic attribute raising during introspection: report, do not fail on it
        print(f"  preflight: JIT cache not introspectable ({type(e).__name__}: {e})")
        targets = []
    bad = [t for t in targets if "cuda" not in t.lower()]
    assert not bad, f"reference sparse_attn JIT cache holds non-CUDA kernels {bad}"
    a = model.layers[0].attn
    prev = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    torch.set_default_device("cpu")
    try:
        R.get_window_topk_idxs(a.window_size, 1, 1, 5)           # seed the lru cache with a CPU entry
        with reference_device(DEV):
            idx = R.get_window_topk_idxs(a.window_size, 1, 1, 5)
    finally:
        if prev is not None:
            torch.set_default_device(prev)
    assert idx.device.type == "cuda", f"window indices on {idx.device} under reference_device(cuda)"
    with torch.inference_mode():
        q = torch.zeros(1, 1, a.n_local_heads, a.head_dim, dtype=torch.bfloat16, device=DEV)
        kv = torch.zeros(1, a.window_size, a.head_dim, dtype=torch.bfloat16, device=DEV)
        try:
            o = K.sparse_attn(q, kv, a.attn_sink, idx, a.softmax_scale)
        except RuntimeError as e:
            raise AssertionError(f"reference sparse_attn rejects CUDA inputs (CPU-targeted JIT entry?): {e}") from e
    torch.cuda.synchronize()
    assert o.is_cuda and torch.isfinite(o.float()).all()
    print(f"  preflight: sparse_attn JIT targets {sorted(set(targets)) or '(none introspectable)'}; window idx on "
          f"{idx.device}; direct call on cuda OK")


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
    COUNTS["logit_rows"] += len(a)


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
        assert len(s_h) == len(o_h) == n and all(torch.equal(a, b) for a, b in zip(s_h, o_h)), "main_hidden"
        COUNTS["main_hidden"] += len(s_h)
        for k in o_s:
            assert torch.equal(s_s[k], o_s[k]), (kind, L, k)
        assert set(s_s) == set(o_s) and o_s, (sorted(s_s), sorted(o_s))
        COUNTS["caches"] += len(o_s)
        COUNTS["gates"] += 1
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
            out, lg, _ = reference_forward(model, torch.tensor([ids], device=DEV), 0)
            toks.append(int(out.reshape(-1)[0]))
            for k in range(12):
                out, lg, _ = reference_forward(model, torch.tensor([[toks[-1]]], device=DEV), 300 + k)
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
        combos = [{"moe_impl": "dg-masked"}, {"moe_impl": "dg-inplace"}, {"fp8_impl": "gemv"}, {"attn_impl": "flashmla"},
                  {"glue_impl": "fused"}, dict(FAST), dict(FAST_INPLACE)]
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


# ------------------------------------------------------------------------------------------------ M1-T10-FIX D / G rows
def _need_step(model, pool, **kw):
    st, why = _try_step(model, pool, **{k: v for k, v in kw.items() if v is not None})
    assert st is not None, f"{kw} did not build: {why}"
    return st


def _eq_runs(a, b, what):
    _assert_rows(a[0], b[0], what)
    assert all(torch.equal(x, y) for x, y in zip(a[1], b[1])), f"{what}: main_hidden"
    COUNTS["main_hidden"] += len(a[1])
    for k in b[2]:
        assert torch.equal(a[2][k], b[2][k]), (what, k)
    COUNTS["caches"] += len(b[2])
    COUNTS["gates"] += 1


def test_static6_dg_inplace_equals_dg_masked():
    """D row at 6 rows (M1-T10-FIX Part B 1; M1-T10-FIX2: S = 8 spare slots, compact cold index): dg-inplace
    (DeepGEMM contiguous grouped GEMM over the pool in place) == dg-masked, torch.equal on every row's fp32 logits,
    main_hidden and the caches, both P0 parities, static-eager; dg-inplace graph == its static-eager; no overflow
    (mini4: at most 4 cold unique experts per layer-step < 8). The dg-inplace step is built first (it grows the
    pool; a step built before the growth refuses to run)."""
    _need_tilelang()
    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        for L in (17, 18):
            ids, toks = _toks(L, ROWS * 4, 60 + L)
            si = _need_step(model, pool, rows=ROWS, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8)
            assert si.spare_slots == 8 and si.dg.spare_limit == 8 and pool.spare >= 8
            a = _static(model, si, ids, toks, 4)
            if DEV == "cuda":  # (the CPU dry run has no graphs)
                si.capture()
                ag = _static(model, si, ids, toks, 4, graph=True)
                _assert_rows(ag[0], a[0], "6-row dg-inplace graph vs static-eager")
            ov = si.overflow_counts()
            assert ov["steps"] == 0 and ov["experts"] == 0, ov
            dd = si.dg.describe()
            si.detach()
            sm = _need_step(model, pool, rows=ROWS, hidden_layers=hid, moe_impl="dg-masked")
            b = _static(model, sm, ids, toks, 4)
            sm.detach()
            _eq_runs(a, b, f"D L={L}: 6-row dg-inplace vs dg-masked")
            print(f"  D P0 {L} (parity {L % 2}): 6-row dg-inplace == dg-masked, {len(a[0])} rows x V + main_hidden + "
                  f"caches; graph == eager; spare slots {dd['spare_slots']} (budget {dd['spare_limit']}), groups/layer "
                  f"{dd['groups_per_layer']}, HBM added {dd['hbm_added_GiB']} GiB; overflow steps {ov['steps']}")


def _expected_overflow(st, S):
    """Cold unique experts beyond S over all layers of the last EAGER step (each MoE's `last` routing)."""
    from runtime.moe_rows import compact_cold_host

    return sum(compact_cold_host(m.t["gtab"].tolist(), m.last["uniq"].tolist(), m.last["counts"].tolist(),
                                 m.t["H"], S)["overflow"] for m in st.moes)


def test_static6_dg_inplace_forced_overflow_trips_counter():
    """M1-T10-FIX2 (2): spare_slots = 1 on mini4 (4 of 8 experts hot per layer): the staging kernel's device counter
    counts exactly the cold unique experts beyond the first per layer (host expectation from each eager step's
    routing), the steps still run, and the graph replay of the same inputs trips the counter identically (the
    counter is graph-safe: zeroed and accumulated inside the captured step)."""
    _need_tilelang()
    with _model("mini4") as (model, pool):
        ids, toks = _toks(17, ROWS * 3, 91)
        st = _need_step(model, pool, rows=ROWS, moe_impl="dg-inplace", spare_slots=1)
        assert st.spare_slots == 1
        _prefill(model, ids)
        want = []
        with torch.inference_mode():
            for s in range(3):
                st.step(toks[s * ROWS:(s + 1) * ROWS], len(ids) + s * ROWS, graph=False)
                want.append(_expected_overflow(st, 1))
                assert int(st.ovf_step.item()) == want[-1], (s, int(st.ovf_step.item()), want[-1])
        ov = st.overflow_counts()
        assert sum(want) > 0 and ov["experts"] == sum(want) and ov["steps"] == sum(1 for w in want if w), (ov, want)
        COUNTS["gates"] += 1
        g_ov = None
        if DEV == "cuda":
            st.capture()
            _static(model, st, ids, toks, 3, graph=True)
            g_ov = st.overflow_counts()
            assert g_ov["experts"] == sum(want) and g_ov["steps"] == ov["steps"], (g_ov, want)
        print(f"  overflow S=1: eager {ov['steps']}/3 steps, {ov['experts']} cold experts not staged (host "
              f"expectation {want}); graph {g_ov if g_ov is None else (g_ov['steps'], g_ov['experts'])}")
        st.detach()


def _rows_expected_on(moe, fp8="gemv"):
    """M1-T10-PORT: the clusters that must be on at 6 rows with `moe` / `fp8`: every cluster but the two left off
    (fused_glue.ROWS_WHY) and those n/a by impl (dg-inplace: gather; fp8 ref -- the CPU dry run --: the GEMV parts)."""
    from runtime import fused_glue as FG

    never = set(FG.ROWS_WHY) | ({"gather"} if moe == "dg-inplace" else {"dg_sf_pack"})   # M1-T10-MOE: dg_sf_pack
    if fp8 != "gemv":                                                                      # ported (dg-inplace's)
        never |= {"q_rope_epilogue", "gemv_pair", "reduce_fold"}
    return set(FG.CLUSTERS) - never


def test_static6_fused_glue_equals_ref_glue():
    """G row at 6 rows (M1-T10-FIX Part B 2; M1-T10-PORT: EVERY ported cluster): glue fused -- all clusters
    applicable to the fast impls on at 6 rows, each self-checked at build against the reference glue at 6 rows
    (`verify_<cluster>` == "equal": the per-cluster G rows), only fused_glue.ROWS_WHY n/a -- == glue ref, torch.equal,
    on the fast impls (gemv, flashmla) with dg-masked and dg-inplace; graph == static-eager. Then, dg-inplace, each
    cluster alone OFF (DSV41_GLUE_DISABLE) == ref glue (bisects a cluster that breaks only in combination)."""
    _need_tilelang()
    from runtime import fused_glue as FG

    with _model("mini4") as (model, pool):
        hid = (len(model.layers) - 1,)
        ids, toks = _toks(18, ROWS * 4, 77)
        refs = {}
        for moe in ("dg-inplace", "dg-masked"):
            kw = dict(FAST, moe_impl=moe)
            f = _need_step(model, pool, rows=ROWS, hidden_layers=hid, **kw)
            d = f.glue.describe()
            print(f"  G {moe}: {f.glue.glue_line()[:400]}")
            want = _rows_expected_on(moe, f.fp8_impl)
            assert set(d["on"]) == want, {c: d["clusters"][c] for c in want ^ set(d["on"])}
            ver = {k_: v for k_, v in d["calibration"].items() if k_.startswith("verify_")}
            ver.update({k_: v for k_, v in d["calibration"].get("r4", {}).items() if k_.startswith("verify_")})
            assert all(v == "equal" for v in ver.values()), ver
            print(f"  G {moe} per-cluster self-checks at {ROWS} rows: {', '.join(sorted(ver))} all equal")
            COUNTS["g_clusters"] += len(ver)
            a = _static(model, f, ids, toks, 4)
            if DEV == "cuda":
                f.capture()
                ag = _static(model, f, ids, toks, 4, graph=True)
                _assert_rows(ag[0], a[0], f"6-row fused glue ({moe}) graph vs static-eager")
            f.detach()
            r = _need_step(model, pool, rows=ROWS, hidden_layers=hid, **dict(kw, glue_impl="ref"))
            b = _static(model, r, ids, toks, 4)
            r.detach()
            refs[moe] = b
            refs.setdefault("fp8", f.fp8_impl)
            _eq_runs(a, b, f"G {moe}: 6-row fused glue vs ref glue")
            print(f"  G {moe}: 6-row fused glue ({','.join(d['on'])}) == ref glue, {len(a[0])} rows; graph == eager")
        moe = "dg-inplace"
        prev = os.environ.get(FG.DISABLE_ENV)
        fp8 = refs["fp8"]
        try:
            for c in sorted(_rows_expected_on(moe, fp8)):
                os.environ[FG.DISABLE_ENV] = c
                f = _need_step(model, pool, rows=ROWS, hidden_layers=hid, **dict(FAST, moe_impl=moe))
                assert f.glue.status[c].startswith("off"), f.glue.status[c]
                a = _static(model, f, ids, toks, 4)
                f.detach()
                _eq_runs(a, refs[moe], f"G {moe}: 6-row fused glue without {c} vs ref glue")
            print(f"  G {moe}: each of {len(_rows_expected_on(moe, fp8))} clusters alone off == ref glue")
        finally:
            if prev is None:
                os.environ.pop(FG.DISABLE_ENV, None)
            else:
                os.environ[FG.DISABLE_ENV] = prev


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
         test_static6_dg_inplace_equals_dg_masked, test_static6_dg_inplace_forced_overflow_trips_counter,
         test_static6_fused_glue_equals_ref_glue,
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
    print(f"bit-equal comparisons executed: {COUNTS['gates']} gate runs, {COUNTS['logit_rows']} fp32 logit rows, "
          f"{COUNTS['main_hidden']} main_hidden, {COUNTS['caches']} cache tensors (torch.equal); "
          f"{COUNTS['g_clusters']} per-cluster 6-row self-checks equal")
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
