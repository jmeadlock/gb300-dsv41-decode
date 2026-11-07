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

T12-FIX (docs/HANDOFF-M1-T11-PORT.md Part B; box log docs/loop-gpu-test-20261107.log: "logits rows [17] differ"):
the index in that message is into the per-POSITION digest list (Generator.greedy's step_sha256: entry 0 = the
prefill's logits at position L-1, entry i = position L-1+i), so row 17 of the L=17 free run is position 33 -- the
17th decode position at 1.000 tok/step, NOT the first verify step (entry 1, which matched). Position 33 is the first
position where mini3's ratio-2 indexer has more reachable compressed groups than index_topk (T/r = 34/2 = 17 > 16):
the first position where the top-k has to DROP a group (static_step.py #9, design risk 6: identical only if the k-th
and (k+1)-th reachable scores differ). On the CPU emulation of this exact prompt (verified): position 33 has clen 17,
k 16, and the k-th and (k+1)-th scores tie EXACTLY at 0.0 (3 groups share the k-th value); no other position in
30..37 ties. Which tied group is dropped is unspecified by torch.topk and may differ between a [1, W] and a [6, W]
call on the GPU (the emulation's CPU topk happens to agree). A selection change at one position stays at that row on
mini3 (CPU control: forcing a different 6-row selection at positions 33/34 changed rows 17/18 only) -- the box log's
signature (1/64 rows, no stream divergence). The first verify step (entry 1, position 17) matched, which rules out
the prefill-KV / ctx_count 0 / first-row-token hypotheses. The forced runs use another prompt (seed 700+L).

`_gate_stream` therefore EXPLAINS rows instead of failing on them when, and only when, all of this holds (measured
by `_explain_rows`, static mode, the same prompt and switches): (a) the static loop differs at exactly the same
rows; (b) at each such position the 1-row step's reachable index scores tie (or are within the largest 1-row vs
6-row score difference) across the k-th / (k+1)-th boundary; (c) the 6-row step's selected index set differs from
the 1-row step's; (d) re-running the loop with ONLY the 1-row step's selection injected at those positions makes
every row bit-equal to the 1-row step (the selection is the whole cause). Anything else still fails. The evidence
(score bits equal or not, gap, the two dropped groups) is REPORTed.

T12-FIX2 (box log docs/loop-gpu-test-r2-20261107.log): the r2 runs refuted the tie story for row 17 -- scores AND
selections bit-equal, the injection changed nothing -- and the cache-by-cache diff (`cache_probe`, CPU emulation of
the same prompt and map: tests/test_dspark_loop_cpu.py test_cache_probe_free_loop_caches_equal_one_row) finds every
cache a row reads bit-equal to the 1-row step's entering and leaving every step through position 41, the ratio-2
compressor's two slots included: the first step does NOT mis-commit the prefill's partial group. The cause left
is inside the step and CUDA-only (read from torch's Reduce.cuh; the census below confirms or refutes it on the box):
the reduce kernel chooses the lanes per output from the number of outputs,
so the 6-row step's per-token statistics (the mHC mixes norm over hc*dim, Engram's per-copy mean / sum, ...) sum in
another order than the 1-row step's (the CPU reduces every output with one loop: the emulation never sees it), an
fp32 ulp that the next bf16 cast nearly always absorbs -- 1 row in 64 on mini3. `row_invariant_mode(True)` now
evaluates those reductions one row at a time too; `test_loop_reduction_row_invariance_census` measures every site
(plain: reported; in the mode: gated equal) and `test_loop_stream_as_is_reported` reports the T12 mode
(rowinv="linear") next to it. The tie explanation and the cache diff stay as the diagnostic of any row that still
differs.

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
def row_invariant_mode(enabled=True):
    """docs/M1-DSPARK-DESIGN.md 4's row-invariant mode: F.linear and the reference's row-indexed einsums ("bs...")
    evaluated one row at a time (M = 1 calls, as the 1-row step makes them). Debug only.

    T12-FIX2: enabled=True also evaluates torch's float sum / mean over dims >= 2 of a [1, s > 1, ...] tensor one
    s-row at a time (`_rowwise_reduce`): torch's CUDA reduce kernel picks its thread layout (block_width, the
    threads cooperating on one output) from the NUMBER OF OUTPUTS (Reduce.cuh setReduceConfig / set_block_dimension),
    so the per-token statistics of a 6-row step -- the mHC mixes norm `x.square().mean(-1)` over hc*dim
    (model.py:955), Engram's per-copy `mean(-1)` / `sum(-1)` (model.py:357-358) -- sum in a different order than
    the 1-row step's (mini3: hc*dim = 1024 -> 128 vs 256 lanes per output; Engram 24 vs 4 outputs -> 32 vs 64
    lanes), an fp32 ulp that a bf16 cast usually absorbs and occasionally does not (box r2: 1 row in 64, position
    33). enabled="linear" is the T12 mode (F.linear + einsum only), kept for the reported before/after."""
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
    red = enabled is True
    saved = {}
    if red:
        for owner, name in REDUCTIONS:
            saved[(owner, name)] = (name in vars(owner), getattr(owner, name))
            setattr(owner, name, _rowwise_reduce(getattr(owner, name)))
    try:
        yield
    finally:
        F_.linear, torch.einsum = lin0, ein0
        for (owner, name), (own, fn) in saved.items():
            if own:
                setattr(owner, name, fn)
            else:
                delattr(owner, name)   # inherited (torch.Tensor.sum lives on torch._C.TensorBase)


REDUCTIONS = ((torch.Tensor, "sum"), (torch.Tensor, "mean"), (torch, "sum"), (torch, "mean"))


def _rowwise_reduce(orig):
    """sum / mean wrapper: x float, [1, s > 1, ...], every reduced dim >= 2 -> the reduction of each [1, 1, ...]
    s-slice (the 1-row step's exact call shape and layout), concatenated on dim 1. Anything else: the original."""
    def f(x, *args, **kw):
        if (isinstance(x, torch.Tensor) and x.dim() >= 3 and x.size(0) == 1 and x.size(1) > 1
                and x.is_floating_point() and "out" not in kw):
            dim = args[0] if args else kw.get("dim")
            ds = tuple(dim) if isinstance(dim, (tuple, list)) else (() if dim is None else (dim,))
            if ds and all(isinstance(d, int) and not isinstance(d, bool) for d in ds) \
                    and min(int(d) % x.dim() for d in ds) >= 2:
                return torch.cat([orig(x[:, i:i + 1], *args, **kw) for i in range(x.size(1))], 1)
        return orig(x, *args, **kw)

    return f


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
@contextlib.contextmanager
def indexer_probe(override=None):
    """T12-FIX: wraps runtime/static_step.py `indexer_topk_static` (design #9; `_indexer` looks it up as a module
    global at every call) and records every call of a static step by absolute position: rec[pos] = {"rows", "row",
    "clen", "k", "score" (the row's fp32 index scores, cpu), "sel" (its selected int32 index row, cpu)}; the LAST
    call per position wins (in the loop: the call whose row was emitted -- an emitted position is never recomputed).
    override {pos: sel row}: those rows' selections are replaced (the injection proof). Static mode only: it reads the
    running step's device P0 (`patches._ACTIVE`, set by StaticStep._active) with a host sync; capture-time calls pass
    through unrecorded."""
    import runtime.static_step as SS
    from runtime import patches

    orig = SS.indexer_topk_static
    rec = {}

    def wrapped(index_score, clen, k, offset):
        out = orig(index_score, clen, k, offset)
        step = patches._ACTIVE["step"]
        if step is None or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            return out
        p0 = int(step.pos)
        W = index_score.size(-1)
        sc = index_score.reshape(-1, W)
        o2 = out.reshape(-1, out.size(-1))
        cl = clen.reshape(-1)
        if override:
            hit = [i for i in range(sc.size(0)) if p0 + i in override]
            if hit:
                o2 = o2.clone()
                for i in hit:
                    o2[i] = override[p0 + i].to(device=o2.device, dtype=o2.dtype)
                out = o2.view(out.shape)
        for i in range(sc.size(0)):
            rec[p0 + i] = {"rows": sc.size(0), "row": i, "clen": int(cl[i] if cl.numel() > 1 else cl[0]), "k": int(k),
                           "score": sc[i].detach().float().to("cpu").clone(), "sel": o2[i].detach().to("cpu").clone()}
        return out

    SS.indexer_topk_static = wrapped
    try:
        yield rec
    finally:
        SS.indexer_topk_static = orig


def _sel_set(sel):
    return sorted(int(v) for v in sel.tolist() if int(v) >= 0)


# ------------------------------------------------------------------------------------------------ T12-FIX2 census
CENSUS_SITES = (  # (name, per-row shape from (D, hc, H, W), the reference expression)
    ("mHC mixes norm x.square().mean(-1) [1,R,hc*D] (model.py:955)", lambda D, hc, H, W: (hc * D,),
     lambda t: t.square().mean(-1, keepdim=True)),
    ("RMSNorm x.square().mean(-1) [1,R,D] (model.py:291)", lambda D, hc, H, W: (D,),
     lambda t: t.square().mean(-1, keepdim=True)),
    ("Engram h.square().mean(-1) [1,R,hc,D] (model.py:359)", lambda D, hc, H, W: (hc, D),
     lambda t: t.square().mean(-1)),
    ("Engram (h*w*k).sum(-1) [1,R,hc,D] (model.py:360)", lambda D, hc, H, W: (hc, D), lambda t: t.sum(-1)),
    ("hc_pre torch.sum(dim=2) [1,R,hc,D] (model.py:961)", lambda D, hc, H, W: (hc, D),
     lambda t: torch.sum(t, dim=2)),
    ("hc_post torch.sum(dim=2) [1,R,hc,hc,D] (model.py:967)", lambda D, hc, H, W: (hc, hc, D),
     lambda t: torch.sum(t, dim=2)),
    ("indexer .sum(dim=2) over heads [1,R,H,W] (model.py:558)", lambda D, hc, H, W: (H, W),
     lambda t: t.sum(dim=2)),
)


def _reduction_census(dev, rows=6, trials=48, shapes=None):
    """T12-FIX2: for each per-token reduction the 6-row step makes (CENSUS_SITES), row r of the R-row call vs the
    same row computed alone as a fresh [1, 1, ...] tensor (the 1-row step's call), bitwise, plain and inside
    row_invariant_mode(True). -> [{"site", "shape", "plain_rows_differing", "mode_rows_differing", "rows"}]."""
    shapes = shapes or {"mini3": (256, 4, 4, 256)}
    g = torch.Generator(device="cpu").manual_seed(12)
    out = []
    for tag, (D, hc, H, W) in shapes.items():
        for name, shp, fn in CENSUS_SITES:
            s = shp(D, hc, H, W)
            xs = [torch.randn(1, rows, *s, generator=g, device="cpu").to(dev) for _ in range(trials)]
            res = {}
            for mode in ("plain", "mode"):
                ctx = row_invariant_mode(True) if mode == "mode" else contextlib.nullcontext()
                bad = 0
                with ctx, torch.inference_mode():
                    for x in xs:
                        yr = fn(x)
                        for r in range(rows):
                            y1 = fn(x[:, r:r + 1].clone())
                            bad += int(not torch.equal(yr[:, r:r + 1].reshape(-1), y1.reshape(-1)))
                res[mode] = bad
            out.append({"site": name, "shape": tag, "plain_rows_differing": res["plain"],
                        "mode_rows_differing": res["mode"], "rows": rows * trials})
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    return out


# ------------------------------------------------------------------------------------------------ T12-FIX2 caches
def cache_snapshot(st, p):
    """Every cache a row at position p+1 reads, as the reference's per-token decode leaves it after position p:
    the step's 256-ring slots 0..p (p < 256), compressed KV / index K groups < (p+1)//r, the ratio-2 compressor's
    slots (p % 2 holds kv(p), the other kv(p-1)) and the Engram hash cache 0..p. Clones (eager, syncs)."""
    m = st.model
    d = {}
    for li, layer in enumerate(m.layers):
        a = layer.attn
        d[f"L{li}.ring"] = st._ring_of[id(a)][0, : min(p + 1, st.ring)].clone()
        if a.compress_ratio:
            r = a.compress_ratio
            g = (p + 1) // r
            d[f"L{li}.compress_kv"] = a.compress_kv_cache[0, :g].clone()
            ix = getattr(a, "indexer", None)
            if ix is not None and getattr(ix, "owns_k", False):
                d[f"L{li}.index_k"] = ix.k_cache[0, :g].clone()
            c = a.compressor
            if c is not None and r == 2:
                s = p % 2
                d[f"L{li}.kv_state[p%2]"] = c.kv_state[0, s].clone()
                d[f"L{li}.score_state[p%2]"] = c.score_state[0, s].clone()
                d[f"L{li}.kv_state[(p-1)%2]"] = c.kv_state[0, 1 - s].clone()
                d[f"L{li}.score_state[(p-1)%2]"] = c.score_state[0, 1 - s].clone()
    eh = getattr(m, "engram_hash", None)
    if eh is not None:
        d["engram.cache"] = eh.cache[0, : p + 1].clone()
    return d


def cache_diff(a, b):
    """Names of the snapshot tensors that differ bitwise (with the first differing leading index)."""
    out = []
    for k in a:
        x, y = a[k], b.get(k)
        if y is None or x.shape != y.shape:
            out.append(f"{k}: shape {tuple(x.shape)} vs {None if y is None else tuple(y.shape)}")
            continue
        xv = x.view(torch.uint8) if x.element_size() == 1 else x
        yv = y.view(torch.uint8) if y.element_size() == 1 else y
        if not torch.equal(xv, yv):
            ne = (xv != yv) if not xv.is_floating_point() else ~((xv == yv) | (torch.isnan(xv) & torch.isnan(yv)))
            idx = ne.reshape(ne.size(0), -1).any(-1).nonzero().reshape(-1).tolist() if ne.dim() > 1 else []
            if ne.dim() <= 1 or idx:
                out.append(f"{k}: rows {idx[:6]}" if idx else k)
    return out


@contextlib.contextmanager
def cache_probe():
    """T12-FIX2 ('dump both caches at that position, diff tensor by tensor'): static mode, syncs. rec["1row"][p] =
    the 1-row step's caches after position p; rec["enter"][p] = the loop's caches entering the step with P0 = p+1
    (after the previous commit AND draft); rec["after"][a] = after a live step's commit with A' = a. tag["t"] selects
    which side records ("1row" / "loop")."""
    from runtime.dspark_loop import DSparkLoop
    from runtime.static_step import StaticStep

    rec = {"1row": {}, "enter": {}, "after": {}}
    tag: dict = {"t": None}
    ob, ova = StaticStep._body, DSparkLoop._verify_accept

    def capturing():
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

    def body(self, parity):
        if tag["t"] != "1row" or self.rows != 1 or capturing():
            return ob(self, parity)
        p = int(self.pos)
        ob(self, parity)
        rec["1row"][p] = cache_snapshot(self, p)

    def va(self):
        if tag["t"] != "loop" or capturing():
            return ova(self)
        p = int(self.st.pos) - 1
        rec["enter"][p] = cache_snapshot(self.st, p)
        ova(self)
        if int(self.b.live):
            a = int(self.b.a_new)
            rec["after"][a] = cache_snapshot(self.st, a)

    StaticStep._body, DSparkLoop._verify_accept = body, va
    try:
        yield rec, tag
    finally:
        StaticStep._body, DSparkLoop._verify_accept = ob, ova


def _explain_rows(kind, L, max_new, bad, seed, pin=False, **impls):
    """T12-FIX (module docstring): rebuild, run the 1-row truth and the loop in STATIC mode (row-invariant mode, the
    same prompt / switches) under `indexer_probe`, classify every differing emitted row, then re-run the loop with the
    1-row step's selections injected at those positions. -> evidence dict; ok = the four conditions hold.
    T12-FIX2: the same runs also go through `cache_probe`; every differing row reports which caches differ from the
    1-row step's entering its step (P0 = its position) and after that step's commit (empty = no cache state
    difference: the row's difference is inside the step)."""
    with _model(kind) as (model, pool), row_invariant_mode(True), gemv_pin_m1(pin), cache_probe() as (crec, ctag):
        V = model.head.weight.size(0)
        gen = _gen(model, "static", **impls)
        ids = _toks(L, seed, min(2000, V))
        ctag["t"] = "1row"
        with indexer_probe() as rec1:
            truth, sha = _truth(gen, ids, max_new)
        ctag["t"] = "loop"
        loop = _loop(gen, max_new)
        try:
            with indexer_probe() as rec6:
                r = loop.generate(ids, max_new, stop_on_eos=False, parity=True)
            ctag["t"] = None
            bad_s = [i for i, (x, y) in enumerate(zip(r["step_sha256"], sha)) if x != y]
            rows = []
            for i in bad_s:
                p = L - 1 + i
                a, b = rec1.get(p), rec6.get(p)
                e = {"row": i, "pos": p}
                c1e, c6e = crec["1row"].get(p - 1), crec["enter"].get(p - 1)
                c1a, c6a = crec["1row"].get(p), crec["after"].get(p)
                e["cache_enter_diff"] = cache_diff(c1e, c6e) if c1e is not None and c6e is not None else "n/a"
                e["cache_after_diff"] = cache_diff(c1a, c6a) if c1a is not None and c6a is not None else "n/a"
                if a is None or b is None:
                    e["probe"] = "no indexer call recorded (fused indexer_post or a layer without an indexer?)"
                    rows.append(e)
                    continue
                c, k = a["clen"], a["k"]
                s1, s6 = a["score"][:c], b["score"][:c]
                srt = torch.sort(s1, descending=True).values
                gap = float(srt[k - 1] - srt[k]) if c > k else None
                diff = float((s1 - s6).abs().max()) if c else 0.0
                sel1, sel6 = _sel_set(a["sel"]), _sel_set(b["sel"])
                e.update(clen=c, k=k, verify_row=b["row"], gap_k_k1=gap, kth=float(srt[k - 1]) if c >= k else None,
                         score_bits_equal=bool(torch.equal(s1.view(torch.int32), s6.view(torch.int32))),
                         max_score_diff=diff, sel_equal=sel1 == sel6,
                         only_1row=sorted(set(sel1) - set(sel6)), only_6row=sorted(set(sel6) - set(sel1)),
                         tie=gap is not None and gap <= diff)
                rows.append(e)
            inj = {L - 1 + i: rec1[L - 1 + i]["sel"] for i in bad_s if (L - 1 + i) in rec1}
            with indexer_probe(override=inj):
                r2 = loop.generate(ids, max_new, stop_on_eos=False, parity=True)
            bad2 = [i for i, (x, y) in enumerate(zip(r2["step_sha256"], sha)) if x != y]
        finally:
            loop.close()
    ok = (bad_s == list(bad) and bool(rows) and all(e.get("tie") and not e.get("sel_equal", True) for e in rows)
          and not bad2 and r2["token_ids"] == truth)
    return {"ok": ok, "bad_requested": list(bad), "bad_static": bad_s, "rows": rows, "bad_after_injection": bad2,
            "stream_equal_after_injection": r2["token_ids"] == truth}


def _gate_stream(kind="mini3", L=17, max_new=64, mode="static", n_graphs=None, rowinv=True, pin=False, **impls):
    """Loop stream == 1-row greedy (+ per-position logits) in row-invariant mode; as-is: reported. Returns res.
    Rows differing only through an indexer top-k tie (T12-FIX, module docstring) are explained, not failed."""
    with _model(kind) as (model, pool), row_invariant_mode(rowinv), gemv_pin_m1(pin):
        V = model.head.weight.size(0)
        gen = _gen(model, mode, **impls)
        seed = 300 + L
        ids = _toks(L, seed, min(2000, V))
        truth, sha = _truth(gen, ids, max_new)
        loop = _loop(gen, max_new, n_graphs)
        try:
            r = loop.generate(ids, max_new, stop_on_eos=False, parity=True)
            r2 = loop.generate(ids, max_new, stop_on_eos=False, parity=False)
        finally:
            loop.close()
        k = _first_div(r["token_ids"], truth)
        bad = [i for i, (x, y) in enumerate(zip(r["step_sha256"], sha)) if x != y]
        mname = {True: "row-invariant", "linear": "row-invariant(T12 linear+einsum only)", False: "as-is"}[rowinv]
        tag = (f"{kind} L={L} mode={mode}{'/' + str(n_graphs) + 'g' if mode == 'graph' else ''} "
               f"{mname}{' gemv-pinned' if pin else ''} {impls or ''}")
        REPORT.append(f"{tag}: {len(r['token_ids'])} tokens, {r['steps']} steps, {r['tokens_per_step']:.3f} tok/step, "
                      f"stream {'== greedy' if k is None else f'diverges at {k}'}, logits rows differing "
                      f"{len(bad)}/{len(sha)}{f' {bad[:8]}' if bad and rowinv else ''}")
        assert r2["token_ids"] == r["token_ids"] and r2["accepts"] == r["accepts"], f"{tag}: fast run != parity run"
    if rowinv is True:
        assert k is None, f"{tag}: stream diverges at {k}: loop {r['token_ids'][k:k + 4]} vs {truth[k:k + 4]}"
        if bad:
            ev = _explain_rows(kind, L, max_new, bad, seed, pin=pin, **impls)
            REPORT.append(f"{tag}: T12-FIX rows {bad} explained={ev['ok']}: " + "; ".join(
                f"row {e['row']} pos {e['pos']}: clen {e.get('clen')} > k {e.get('k')}, k-th score {e.get('kth')}, "
                f"gap k/k+1 {e.get('gap_k_k1')}, 6-row vs 1-row scores bit-equal {e.get('score_bits_equal')} (max "
                f"|d| {e.get('max_score_diff')}), dropped only by 6-row {e.get('only_1row')} / only by 1-row "
                f"{e.get('only_6row')}" + (f" [{e['probe']}]" if 'probe' in e else "")
                + f"; T12-FIX2 caches vs the 1-row step: entering the step {e.get('cache_enter_diff') or 'all equal'}, "
                  f"after its commit {e.get('cache_after_diff') or 'all equal'}" for e in ev["rows"])
                + f"; rows differing after injecting the 1-row selections: {ev['bad_after_injection']} (stream equal "
                  f"{ev['stream_equal_after_injection']}); static-mode rows {ev['bad_static']}")
            COUNTS["explained_rows"] = COUNTS.get("explained_rows", 0) + (len(bad) if ev["ok"] else 0)
            assert ev["ok"], f"{tag}: logits rows {bad[:8]} differ from the 1-row step and are NOT explained by an " \
                             f"indexer top-k tie: {ev}"
    COUNTS["gates"] += 1
    COUNTS["tokens"] += len(truth)
    COUNTS["logit_rows"] += len(sha)
    COUNTS["steps"] += r["steps"]
    if rowinv is not True:
        print(f"  {tag}: REPORTED -- stream {'== greedy' if k is None else f'diverges at {k}'}, logits rows differing "
              f"{len(bad)}/{len(sha)}{f' {bad[:8]}' if bad and rowinv else ''}")
    else:
        print(f"  {tag}: loop == 1-row greedy ({len(truth)} tokens, {r['steps']} steps"
              f"{f'; rows {bad} = indexer top-k tie, explained (T12-FIX)' if bad else ''})")
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
    # T12-FIX2 before/after: the T12 mode (F.linear + einsum per row, reductions batched) on the same prompt -- the
    # box r2 runs showed logits row 17 (position 33) differing in it; row_invariant_mode(True) is the gated mode
    _gate_stream("mini3", 17, 64, mode="static", rowinv="linear")


def test_loop_reduction_row_invariance_census():
    """T12-FIX2: the step's per-token reductions (CENSUS_SITES) at 6 rows vs the same row alone, bitwise: REPORTED
    plain (torch's CUDA reduce picks its lanes per output from the number of outputs), GATED equal inside
    row_invariant_mode(True). mini3's shapes and real-model-shaped ones (dim 5120, the inventory's embed width;
    hc 4; 64 index heads over W = 1024)."""
    _need_tilelang()
    rows = _reduction_census(DEV, shapes={"mini3": (256, 4, 4, 256), "real-shaped": (5120, 4, 64, 1024)},
                             trials=16)
    for r in rows:
        REPORT.append(f"reduction census [{r['shape']}] {r['site']}: rows differing 6-row vs 1-row plain "
                      f"{r['plain_rows_differing']}/{r['rows']}, in row_invariant_mode {r['mode_rows_differing']}"
                      f"/{r['rows']}")
    bad = [r for r in rows if r["mode_rows_differing"]]
    assert not bad, f"row_invariant_mode leaves reductions row-variant: {bad}"
    COUNTS["gates"] += 1
    n_plain = sum(1 for r in rows if r["plain_rows_differing"])
    print(f"  reduction census: {len(rows)} sites; plain: {n_plain} sites row-variant at 6 rows (reported); "
          f"row_invariant_mode: all row-invariant")


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
    """Real checkpoint, fast mode (no row-invariant mode): stream vs 1-row REPORTED; replay x2 gated.

    T12-FIX2 (box r2: the 1-row `_truth` hit a device-side `index out of bounds` in its capture warm-up): the fixture
    is now bench/headline.py's build line for line -- DSV41_GEMV_PIN_M1 before the build, the same map
    (`_model("real")`), set_determinism(0), the Generator on the headline impls (dg-inplace / gemv / fused / flashmla,
    spare slots DEFAULT_SPARE_SLOTS_ROWS) and the ExpertPool pre-grown to that spare budget BEFORE any StaticStep is
    built (the 1-row truth's own dg-inplace build is then a no-op growth, and the 6-row VerifyStep's pointer tables
    are built on the same storage). The r2 fixture ran every impl at "ref" -- the 1-row reference-MoE graph path on
    the real checkpoint, which nothing has run since T8 (first-light-graph-20261106, before load_mtp); the headline
    has never exercised it. DSV41_LOOP_REAL_IMPLS=ref keeps that configuration reachable for the diagnosis
    (run it with CUDA_LAUNCH_BLOCKING=1 to localize the faulting kernel)."""
    _need_tilelang()
    if not os.environ.get("DSV41_MODEL_DIR"):
        raise unittest.SkipTest("DSV41_MODEL_DIR not set")
    ref_impls = os.environ.get("DSV41_LOOP_REAL_IMPLS", "headline") == "ref"
    with gemv_pin_m1(), _model("real") as (model, pool):
        from runtime.generate import Generator, set_determinism
        from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS

        det = set_determinism(0)
        impls: dict = {} if ref_impls else dict(moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused",
                                                 attn_impl="flashmla", spare_slots=DEFAULT_SPARE_SLOTS_ROWS)
        gen = Generator(model, os.environ["DSV41_MODEL_DIR"], mode="graph", max_pos=4095, **impls)
        grown = 0
        if not ref_impls and getattr(gen, "pool", None) is not None:
            grown = gen.pool.add_spare_slots(DEFAULT_SPARE_SLOTS_ROWS, model)   # once, before any StaticStep
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
        REPORT.append(f"real ({'ref impls' if ref_impls else 'headline impls ' + str(impls)}; pool spare "
                      f"{getattr(gen.pool, 'spare', None)}, +{grown / 2**30:.2f} GiB; determinism {det}): stream vs "
                      f"1-row {'equal' if k is None else f'first divergence at {k}'}; "
                      f"{a['tokens_per_step']:.3f} tok/step; step {sum(a['step_ms']) / len(a['step_ms']):.2f} ms")
        COUNTS["gates"] += 1


TESTS = [test_loop_reduction_row_invariance_census, test_loop_stream_equals_greedy_static,
         test_loop_stream_as_is_reported, test_loop_forced_rejections_static,
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
