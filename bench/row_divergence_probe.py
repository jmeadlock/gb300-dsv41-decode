#!/usr/bin/env python3
"""row_divergence_probe.py -- T12-DIAG: where does the 6-row verify step's row j first differ from the 1-row step at
P0 + j, now that every fused-glue cluster is row-invariant? (docs/LEDGER.md 2026-11-07 14:03: streams still 2/8,
prose diverges at step 49 with 1-row margin 0.094; docs/headline-20261107h.json.)

    # mini checkpoint (box; the plumbing + the rejection path on the headline impls)
    python3 bench/row_divergence_probe.py --kind mini4 --json scratch/rowdiv-mini4.json
    # the real checkpoint (box): per layer, per row, which tensor differs first and by how many ulp
    python3 bench/row_divergence_probe.py --model-dir /ref --fixture w0/results/phase1/parity-A.jsonl --tag prose \\
        --steps 40 --json scratch/rowdiv-real.json
    # the loop's own P0 sequence for a prompt (its accept log), e.g. shell (stream diverges at token 171 = step 46),
    # with a second set of rejected-row tokens (a rejected row's write must never matter)
    python3 bench/row_divergence_probe.py --model-dir /ref --fixture w0/results/phase1/parity-A.jsonl \\
        --plan-from docs/headline-20261107i-plan-gemv.json:shell --reject-variants 2 --json scratch/rowdiv-shell.json
    # the candidate fixes, one at a time (each is a diagnostic switch, not a shipping path)
    ... --rowwise-linear fp32        # every fp32 F.linear at M > 1 evaluated one row at a time (= the 1-row cuBLAS call)
    ... --rowwise-linear all         # every F.linear (fp32 and bf16)
    ... --row-invariant-mode         # runtime/row_census.row_invariant_mode: F.linear + the "bs..." einsums per row
    ... --flashmla-row-split         # $DSV41_FLASHMLA_ROW_SPLIT=1: one FlashMLA call per query row
    ... --router-impl plan-gemv      # $DSV41_ROUTER_IMPL: the fixed-order gate GEMV (runtime/router_plan.py)
    # synthetic real-width census of the step's cuBLAS / topk / FlashMLA sites (no checkpoint; seconds)
    python3 bench/row_divergence_probe.py --census-only --json scratch/rowdiv-census.json

What it does (eager static steps; the graph replays the same kernels, L1 parity):
  1. prefill, then the 1-row StaticStep over a token stream (`--stream greedy`: its own argmax at every position, the
     headline's truth; `random`: seeded random ids), recording per position, per layer, every TAP below;
  2. prefill again, then the 6-row VerifyStep over the same stream with an accept PLAN (`--plan 5,0,2,...`, default
     all 5 = teacher-forced): a step at P0 feeds the stream's tokens for rows 0..n and WRONG tokens for rows
     n+1..5 (the rejected drafts), commits the ratio-2 compressor state for A' = P0 + n (VerifyStep.commit_ratio2,
     the loop's commit) and the next step starts at P0 + n + 1 -- so rejected rows' KV / compressed / indexer /
     Engram writes are left in the caches exactly as the DSpark loop leaves them (candidate 3);
  3. for every accepted row j: its taps vs the 1-row taps at P0 + j, in execution order -> the FIRST tap that
     differs (layer, site, ulp, mismatch count) and the final logits (equal / argmax / the 1-row top-2 margin).

TAPS (keys = (layer, site, k-th call of that site in that layer)):
  attn.q, attn.idx, attn.tlen        the attention kernel's inputs (q per row, each row's index list, its length)
  attn.kv                            exact int64 checksum of the kv rows each row's list references (cache content)
  attn.sparse                        the attention kernel's output (FlashMLA / reference sparse_attn)
  cache.ckv, cache.idxk              the compressed-KV / indexer-K cache row a kv-source layer's compressor wrote for
                                     the group that row j's position completes (zeros when it completes none): the
                                     latent the loop leaves in the cache vs the 1-row decode's, for that group
  attn.out                           the attention block output (after the o path, wo_a, wo_b)
  indexer.topk                       the DSA indexer's top-k ids (with the static step's offset)
  router.ids, router.w               the fused router's ids / weights (glue.route / route_rows)
  ffn.out, block.out, engram.out     post-MoE, post-block hidden, Engram output
  head.logits                        the head
  linear:<weight>                    every F.linear output (cuBLAS: the gate's fp32 mm, the compressor's fp32 /
                                     bf16 projections, the indexer's weights_proj / wk, the head under cublas)
  einsum:<eq>                        every torch.einsum output (wo_a, the indexer score; the score is compared
                                     on the row's reachable columns only, < (pos + 1) // ratio)
  Reading it: if attn.q/idx/tlen/kv are equal and attn.sparse differs -> the attention kernel (candidate 1);
  attn.kv differs with equal idx -> cache content (an earlier row's / step's write; candidate 3): cache.ckv /
  cache.idxk on the row that completed the group names the write (its linear:*compressor* taps equal -> the
  pooling arithmetic, not the projection); indexer.topk differs -> look at the einsum / weights_proj taps before it
  (candidate 2); linear:*gate* differs first -> cuBLAS's M-dependent fp32 GEMM (the router gate is outside
  --fp32-impl fixed: the real-checkpoint census reads it row-variant, 1600 ulp, docs/first-light-rows6-ported-
  20261107.json "gate (x.float() @ W.float())"; $DSV41_ROUTER_IMPL=plan-gemv replaced it: docs/T12-DIAG-UPDATE.md).
  A row that differs in the LOOP (headline) but not here, with --plan-from replaying the loop's accept log, points
  at what the probe does not reproduce: the real draft tokens in rejected rows (--reject-variants N checks that the
  rejected tokens cannot matter), graph vs eager (L1 parity), or the draft graph's own writes.

Rules: no vllm import (FlashMLA through runtime/attn_flashmla.py's .so load); host tensors are built with explicit
device="cpu"; receipts go to --json (scratch/ on the box, never docs/).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROWS = 6
# the headline's row-dependent switches (bench/headline.py main(): set before any build)
HEADLINE_ENV = {"DSV41_FP32_IMPL": "fixed", "DSV41_MOE_ROWMATCH": "on", "DSV41_GEMV_PIN_M1": "1"}
HEADLINE_IMPLS = dict(moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused", attn_impl="flashmla")
ROWWISE_MODES = ("off", "fp32", "all")
Key = Tuple[object, str, int]


# ------------------------------------------------------------------------------------------------ comparison
def ord_key(t: torch.Tensor) -> torch.Tensor:
    from runtime.row_census import _ord_key

    return _ord_key(t)


def take_row(t: torch.Tensor, R: int, j: int) -> Optional[torch.Tensor]:
    """Row j of an R-row step tensor: [1, R, ...] (row axis 1) or [R, ...] (row axis 0); None if neither."""
    if t.dim() >= 2 and t.size(0) == 1 and t.size(1) == R:
        return t[:, j]
    if t.dim() >= 1 and t.size(0) == R:
        return t[j]
    return None


def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Bitwise comparison of two same-shaped flat tensors -> {equal, n, mism, max_ulp, max_abs} (ulp for floats:
    sign-magnitude distance; ints: count of differing entries and of ids in one set only)."""
    a, b = a.reshape(-1), b.reshape(-1)
    out = {"n": int(a.numel())}
    if a.dtype != b.dtype or a.numel() != b.numel():
        return dict(out, equal=False, note=f"dtype/numel {a.dtype}/{a.numel()} vs {b.dtype}/{b.numel()}")
    if a.is_floating_point():
        ka, kb = ord_key(a), ord_key(b)
        d = (ka - kb).abs()
        m = int((d > 0).sum())
        out.update(equal=m == 0, mism=m, max_ulp=int(d.max()) if m else 0)
        if m:
            fa, fb = a.double(), b.double()
            ok = torch.isfinite(fa) & torch.isfinite(fb)
            out["max_abs"] = float((fa[ok] - fb[ok]).abs().max()) if bool(ok.any()) else None
        return out
    m = int((a != b).sum())
    out.update(equal=m == 0, mism=m)
    if m:
        sa, sb = set(a.tolist()), set(b.tolist())
        out.update(only_6row=len(sa - sb), only_1row=len(sb - sa))
    return out


def family(site: str) -> str:
    """`linear:layers.12.ffn.gate.weight` -> `linear:ffn.gate.weight` (the site without its layer index)."""
    return re.sub(r"layers\.\d+\.", "", site)


# ------------------------------------------------------------------------------------------------ patches
_ACTIVE: List[Optional["Taps"]] = [None]   # the Taps recording right now (one at a time)


@contextlib.contextmanager
def probe_patches(rowwise: str = "off", tap_linear: bool = True, tap_einsum: bool = True):
    """F.linear / torch.einsum wrappers for the probe's lifetime: `rowwise` ("fp32": fp32 x and fp32 weight; "all":
    any dtype) evaluates an M > 1 F.linear one row at a time -- the 1-row step's exact cuBLAS call shape [1, K] --
    and the active Taps records every F.linear / einsum output. Restored on exit."""
    if rowwise not in ROWWISE_MODES:
        raise ValueError(f"rowwise {rowwise!r} not in {ROWWISE_MODES}")
    lin0, ein0 = F.linear, torch.einsum

    def linear(x, w, b=None):
        K = x.size(-1)
        M = x.numel() // max(K, 1)
        tap = _ACTIVE[0]
        # rowwise only inside a recorded step (never the prefill: both passes share the same prefill either way)
        if (tap is not None and rowwise != "off" and M > 1
                and (rowwise == "all" or (x.dtype == torch.float32 and w.dtype == torch.float32))):
            xv = x.reshape(M, K)
            out = torch.cat([lin0(xv[i:i + 1], w, b) for i in range(M)], 0).view(*x.shape[:-1], -1)
        else:
            out = lin0(x, w, b)
        if tap is not None and tap_linear:
            out = tap.rec("linear:" + tap.wname(w), out)
        return out

    def einsum(eq, *ops):
        out = ein0(eq, *ops)
        tap = _ACTIVE[0]
        if tap is not None and tap_einsum and isinstance(eq, str):
            out = tap.rec("einsum:" + eq.replace(" ", ""), out)
        return out

    F.linear, torch.einsum = linear, einsum
    try:
        yield
    finally:
        F.linear, torch.einsum = lin0, ein0


@contextlib.contextmanager
def env(**kv):
    prev = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bump(t: torch.Tensor, R: int, j: int) -> torch.Tensor:
    """A copy of t with element 0 of row j moved by one ulp (bit pattern + 1) -- the probe's blindness check."""
    tt = t.clone()
    r = take_row(tt, R, j) if R > 1 else tt
    if r is None or not r.is_floating_point() or not r.is_contiguous():
        return t
    flat = r.reshape(-1)
    iv = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}.get(flat.dtype)
    if iv is None:
        return t
    flat.view(iv)[0:1].add_(1)
    return tt


# ------------------------------------------------------------------------------------------------ taps
class Taps:
    """Instance-level wrappers on one StaticStep (+ its fused glue's router) recording per-layer tensors while a
    step runs (`begin()` / `end()`), plus the F.linear / einsum records of probe_patches. Records are copied to the
    host (`device="cpu"`) at the call. `inject=(layer, site, row)` moves one element of that row by one ulp in the
    recorded tensor and hands the moved copy downstream (tests: the probe must name exactly that tap)."""

    METHODS = ("_engram", "_block", "_attn", "_sparse_attn", "_indexer", "_ffn", "_head")

    def __init__(self, st, model, inject: Optional[Tuple[object, str, int]] = None):
        self.st, self.model, self.R = st, model, int(st.rows)
        self.inject = inject
        self.names: Dict[int, str] = {id(p): n for n, p in model.named_parameters()}
        for n, b in model.named_buffers():
            self.names.setdefault(id(b), n)
        glue = getattr(st, "glue", None)
        gw = getattr(glue, "gate_w", None) if glue is not None else None
        if isinstance(gw, dict):
            for n, m in model.named_modules():
                if id(m) in gw:
                    self.names[id(gw[id(m)])] = n + ".weight"
        self.layer_of: Dict[int, int] = {}
        for i, l in enumerate(model.layers):
            self.layer_of[id(l)] = i
            if getattr(l, "engram", None) is not None:
                self.layer_of[id(l.engram)] = i
        self.recs: Optional["collections.OrderedDict[Key, torch.Tensor]"] = None
        self.cur: object = "pre"
        self.counts: collections.Counter = collections.Counter()
        self._installed: List[Tuple[object, str]] = []
        self._depth = 0

    def wname(self, w: torch.Tensor) -> str:
        n = self.names.get(id(w))
        return n if n is not None else f"w{list(w.shape)}"

    # ---------------------------------------------------------------- recording
    def rec(self, site: str, t, rows=None):
        if self.recs is None or not isinstance(t, torch.Tensor):
            return t
        k = self.counts[(self.cur, site)]
        self.counts[(self.cur, site)] += 1
        if self.inject is not None and k == 0 and (self.cur, site) == tuple(self.inject[:2]):
            t = _bump(t, self.R, int(self.inject[2]))
        self.recs[(self.cur, site, k)] = t.detach().to("cpu", copy=True)
        return t

    def begin(self) -> None:
        self.recs = collections.OrderedDict()
        self.counts = collections.Counter()
        self.cur = "pre"
        _ACTIVE[0] = self

    def end(self) -> "collections.OrderedDict[Key, torch.Tensor]":
        out, self.recs = self.recs, None
        if _ACTIVE[0] is self:
            _ACTIVE[0] = None
        return out

    # ---------------------------------------------------------------- install
    def install(self) -> "Taps":
        st = self
        S = self.st

        def wrap(obj, name, fn):
            orig = getattr(obj, name)
            setattr(obj, name, fn(orig))
            self._installed.append((obj, name))

        wrap(S, "_engram", lambda o: lambda eng, h, hid: (st._set(st.layer_of.get(id(eng), st.cur)),
                                                           st.rec("engram.out", o(eng, h, hid)))[1])

        def block(o):
            def f(layer, x, pre_mix, parity):
                st._set(st.layer_of.get(id(layer), st.cur))
                h, pre = o(layer, x, pre_mix, parity)
                return st.rec("block.out", h), pre
            return f

        wrap(S, "_block", block)
        wrap(S, "_attn", lambda o: lambda attn, x, parity, xq=None: st.rec("attn.out", o(attn, x, parity, xq)))

        def sparse(o):
            def f(q, kv, attn, topk_idxs, nblk, tlen=None):
                if st.recs is not None:
                    st.rec("attn.q", q)
                    st.rec("attn.idx", topk_idxs)
                    if tlen is not None:
                        st.rec("attn.tlen", tlen.reshape(-1))
                    st.rec("attn.kv", kv_checksums(kv, topk_idxs, tlen))
                return st.rec("attn.sparse", o(q, kv, attn, topk_idxs, nblk, tlen))
            return f

        wrap(S, "_sparse_attn", sparse)

        def compress(o):
            def f(attn, x, qr, parity, offset, qq=None):
                out = o(attn, x, qr, parity, offset, qq)
                if st.recs is not None and getattr(attn, "is_kv_source", False) and attn.compress_ratio:
                    st.rec("cache.ckv", group_rows(st.st, attn.compress_kv_cache, int(attn.compress_ratio)))
                    ix = getattr(attn, "indexer", None)
                    if ix is not None and getattr(ix, "owns_k", False):
                        st.rec("cache.idxk", group_rows(st.st, ix.k_cache, int(ix.compress_ratio)))
                return out
            return f

        wrap(S, "_compress_kv", compress)
        wrap(S, "_indexer", lambda o: lambda ix, x, qr, latent, offset, qq=None:
             st.rec("indexer.topk", o(ix, x, qr, latent, offset, qq)))
        wrap(S, "_ffn", lambda o: lambda layer, x: st.rec("ffn.out", o(layer, x)))

        def head(o):
            def f(layer, h, pre_mix):
                st._set("head")
                return st.rec("head.logits", o(layer, h, pre_mix))
            return f

        wrap(S, "_head", head)
        glue = getattr(S, "glue", None)
        if glue is not None:
            for name in ("route", "route_rows"):
                if hasattr(glue, name):
                    wrap(glue, name, self._router)
        return self

    def _router(self, o):
        def f(moe, x):
            self._depth += 1
            try:
                ids, w = o(moe, x)
            finally:
                self._depth -= 1
            if self._depth == 0:
                self.rec("router.ids", ids)
                self.rec("router.w", w)
            return ids, w
        return f

    def _set(self, layer) -> None:
        self.cur = layer

    def remove(self) -> None:
        for obj, name in reversed(self._installed):
            try:
                delattr(obj, name)
            except AttributeError:
                pass
        self._installed = []
        if _ACTIVE[0] is self:
            _ACTIVE[0] = None


def group_rows(st, cache: torch.Tensor, r: int) -> torch.Tensor:
    """The compressed-cache rows this step just wrote, laid out per query row: [1, R, d] where row j holds group
    (p + 1) // r - 1 for p = P0 + j when p completes a group ((p + 1) % r == 0), zeros otherwise. The 1-row step's
    [1, 1, d] is the same quantity at its position, so row j of the R-row step compares to the 1-row step at P0 + j
    (the latent the loop leaves in the cache for that group vs the one the 1-row decode leaves). Eager (reads P0)."""
    R = int(st.rows)
    p0 = int(st.pos.reshape(-1)[0])          # the body's `pos += rows` runs at its end: P0 during the body
    d = cache.size(-1)
    out = torch.zeros(1, R, d, dtype=cache.dtype, device=cache.device)
    for j in range(R):
        p = p0 + j
        if (p + 1) % r == 0:
            out[0, j] = cache[0, (p + 1) // r - 1]
    return out


def kv_checksums(kv: torch.Tensor, topk_idxs: torch.Tensor, tlen: Optional[torch.Tensor]) -> torch.Tensor:
    """Per query row: an exact int64 checksum (plain sum, position-weighted sum) of the kv rows its index list
    references (entries < tlen, >= 0) -> [rows, 2] int64. Equal checksums with equal lists = the same bytes read
    (up to a 2^-64-class collision); computed on the kv's device, no float arithmetic."""
    idx = topk_idxs.reshape(-1, topk_idxs.size(-1)).long()
    rows, K = idx.shape
    kvf = kv.reshape(-1, kv.size(-1))
    n = kvf.size(0)
    valid = (idx >= 0) & (idx < n)
    if tlen is not None:
        ar = torch.arange(K, device=idx.device).view(1, K)
        valid &= ar < tlen.reshape(-1, 1).long()
    safe = torch.where(valid, idx, torch.zeros_like(idx))
    g = kvf.index_select(0, safe.reshape(-1)).view(rows, K, -1)
    iv = {2: torch.int16, 4: torch.int32}.get(g.element_size())
    bits = (g.contiguous().view(iv) if iv is not None else g.contiguous().view(torch.uint8)).to(torch.int64)
    bits = bits * valid.view(rows, K, 1).to(torch.int64)
    D = bits.size(-1)
    w = (torch.arange(K * D, device=bits.device, dtype=torch.int64) % 1000003 + 1).view(1, K, D)
    return torch.stack([bits.sum((1, 2)), (bits * w).sum((1, 2))], -1)


# ------------------------------------------------------------------------------------------------ the probe
def _prefill(model, ids, dev) -> torch.Tensor:
    from runtime.refmodel import reference_forward

    with torch.inference_mode():
        out = reference_forward(model, torch.tensor([list(ids)], dtype=torch.long, device="cpu").to(dev), 0)
    return out


def _wrong(tok: int, j: int, V: int) -> int:
    x = (int(tok) + 1 + j) % V
    return x if x != int(tok) else (x + 1) % V


def compare_row(rec6, rec1, R: int, j: int, pos: int, ratio_of: Dict[int, int]) -> dict:
    """Row j of a 6-row step's records vs the 1-row step's records at the same position -> {first, diffs,
    unrowed, only6, only1}. `first`: the first differing key in execution order (None = every rowed tap equal)."""
    diffs, unrowed, n_cmp, sites = [], [], 0, set()
    for key, t6 in rec6.items():
        if key not in rec1:
            continue
        r = take_row(t6, R, j)
        u = rec1[key]
        if r is None or r.numel() != u.numel():
            unrowed.append(key)
            continue
        n_cmp += 1
        sites.add(family(key[1]))
        layer, site, _ = key
        a, b = r.reshape(-1), u.reshape(-1)
        if site.startswith("einsum:bshd,btd->bsht") and isinstance(layer, int) and ratio_of.get(layer):
            clen = min((pos + 1) // ratio_of[layer], r.size(-1))   # the score's last axis: compressed positions;
            a, b = r[..., :clen].reshape(-1), u[..., :clen].reshape(-1)  # only reachable ones are ever read
        d = diff(a, b)
        if not d["equal"]:
            diffs.append((key, d))
    first = diffs[0] if diffs else None
    return {"first": None if first is None else {"key": _jkey(first[0]), **first[1]},
            "n_compared": n_cmp, "sites": sorted(sites), "n_diff": len(diffs),
            "diff_keys": [_jkey(k) for k, _ in diffs[:24]],
            "unrowed": [_jkey(k) for k in unrowed[:12]], "n_unrowed": len(unrowed),
            "only6": [_jkey(k) for k in list(rec6.keys() - rec1.keys())[:8]],
            "only1": [_jkey(k) for k in list(rec1.keys() - rec6.keys())[:8]]}


def _jkey(k: Key) -> list:
    return [k[0], k[1], k[2]]


def _margin(lg: torch.Tensor) -> float:
    v = lg.float().topk(2).values
    return float(v[0] - v[1])


def run_probe(model, pool, *, impls: dict, L: int = 18, plan: Sequence[int] = (5, 5, 5, 5), seed: int = 45,
              max_pos: int = 511, spare_slots: Optional[int] = 8, stream: str = "random",
              ids: Optional[Sequence[int]] = None, inject: Optional[Tuple[object, str, int]] = None,
              rowwise: str = "off", row_invariant: bool = False, static_kw: Optional[dict] = None,
              reject_variant: int = 0, log: Callable = print) -> dict:
    """The probe on a loaded model (module docstring). impls: StaticStep impl kwargs (both steps). reject_variant:
    which wrong tokens the rejected rows get (two variants must leave every accepted row's bits unchanged -- the
    rejected rows' writes are overwritten before they are read, dspark_loop.py's cache table). Returns the report
    dict; never raises on a divergence (the caller gates)."""
    from runtime.dspark_loop import VerifyStep
    from runtime.row_census import row_invariant_mode
    from runtime.static_step import StaticStep

    dev = model.embed.weight.device
    static_kw = dict(static_kw or {})
    plan = [int(n) for n in plan]
    if any(not 0 <= n <= ROWS - 1 for n in plan):
        raise ValueError(f"plan entries must be in 0..{ROWS - 1}: {plan}")
    V = int(model.head.weight.size(0))
    kw6 = dict(impls, **static_kw)
    if impls.get("moe_impl") == "dg-inplace" and spare_slots is not None:
        kw6["spare_slots"] = spare_slots
    s6 = VerifyStep(model, pool, max_pos=max_pos, rows=ROWS, **kw6)    # first: it grows the pool (dg-inplace)
    s6.defer_ratio2_commit = True
    try:
        s1 = StaticStep(model, pool, max_pos=max_pos, **impls, **static_kw)
    except BaseException:
        s6.detach()
        raise
    ratio_of = {i: int(l.attn.indexer.compress_ratio) for i, l in enumerate(model.layers)
                if getattr(l.attn, "indexer", None) is not None}
    n_used = sum(n + 1 for n in plan)
    n_pos = n_used + ROWS
    if ids is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.randint(2, min(2000, V), (L,), generator=g, device="cpu").tolist()
    L = len(ids)
    if L + n_pos > max_pos:
        raise ValueError(f"prompt {L} + {n_pos} positions > max_pos {max_pos}")
    rep: dict = {"L": L, "plan": plan, "stream": stream, "seed": seed, "impls": dict(impls), "rowwise": rowwise,
                 "row_invariant_mode": bool(row_invariant), "inject": list(inject) if inject else None,
                 "max_pos": max_pos, "spare_slots": spare_slots, "env": {k: os.environ.get(k) for k in (
                     "DSV41_FP32_IMPL", "DSV41_MOE_ROWMATCH", "DSV41_GEMV_PIN_M1", "DSV41_GLUE_ORDER",
                     "DSV41_ROUTER_IMPL", "DSV41_FLASHMLA_ROW_SPLIT", "DSV41_FLASHMLA_TOPK_LENGTH")},
                 "glue_on": {"1": _glue_on(s1), "6": _glue_on(s6)}, "router_impl": _router_impl(),
                 "reject_variant": int(reject_variant),
                 "attn_on": {"1": bool(getattr(s1, "_fm_on", False)), "6": bool(getattr(s6, "_fm_on", False))}}
    t1 = Taps(s1, model).install()
    t6 = Taps(s6, model, inject=inject).install()
    rim = row_invariant_mode() if row_invariant else contextlib.nullcontext()
    try:
        # rim inside, the probe's wrappers outside: a tap sees each op's whole R-row output once
        with rim, probe_patches(rowwise), torch.inference_mode():
            # 1. the 1-row pass: truth stream + records per position
            out = _prefill(model, ids, dev)                     # (output_ids, logits, main_hidden), model.py:1275
            lgp = out[1] if isinstance(out, (tuple, list)) else out
            first = int(lgp.reshape(-1, V)[-1].argmax()) if stream == "greedy" else None
            g = torch.Generator(device="cpu").manual_seed(seed + 1)
            rnd = torch.randint(2, min(2000, V), (n_pos,), generator=g, device="cpu").tolist()
            toks: List[int] = [first if first is not None else rnd[0]]
            rec1: Dict[int, dict] = {}
            lg1: Dict[int, torch.Tensor] = {}
            t0 = time.time()
            for i in range(n_pos):
                p = L + i
                t1.begin()
                lg = s1.step(toks[i], p, graph=False)
                rec1[p] = t1.end()
                lg1[p] = lg.reshape(-1).detach().to("cpu", torch.float32, copy=True)
                if i + 1 < n_pos:
                    toks.append(int(lg1[p].argmax()) if stream == "greedy" else rnd[i + 1])
            rep["one_row_s"] = round(time.time() - t0, 2)
            # 2. the 6-row pass under the plan
            _prefill(model, ids, dev)
            rows_out, P0, i = [], L, 0
            t0 = time.time()
            for s, n in enumerate(plan):
                feed = [toks[i + j] if j <= n else _wrong(toks[i + j], j + 7 * int(reject_variant), V)
                        for j in range(ROWS)]
                t6.begin()
                lg = s6.step(feed, P0, graph=False)
                rec6 = t6.end()
                lg6 = lg.reshape(ROWS, -1).detach().to("cpu", torch.float32, copy=True)
                s6.commit_ratio2(torch.tensor([P0 + n], dtype=torch.long, device="cpu").to(dev),
                                 torch.tensor([P0], dtype=torch.long, device="cpu").to(dev))
                for j in range(n + 1):
                    p = P0 + j
                    c = compare_row(rec6, rec1[p], ROWS, j, p, ratio_of)
                    a, b = lg6[j], lg1[p]
                    c.update(step=s, row=j, pos=p, n_acc=n,
                             logits_equal=bool(torch.equal(a.view(torch.int32), b.view(torch.int32))),
                             argmax_equal=int(a.argmax()) == int(b.argmax()), margin_1row=_margin(b),
                             logits=diff(a, b), logits_sha=hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16])
                    rows_out.append(c)
                i += n + 1
                P0 += n + 1
            rep["six_row_s"] = round(time.time() - t0, 2)
    finally:
        t1.remove()
        t6.remove()
        s6.detach()
        s1.detach()
    rep["rows"] = rows_out
    rep["taps_compared"] = sum(r["n_compared"] for r in rows_out)
    rep["taps_compared_first_row"] = rows_out[0]["n_compared"] if rows_out else 0
    rep["summary"] = summarize(rows_out)
    return rep


def _router_impl() -> Optional[str]:
    """The router impl the builds read ($DSV41_ROUTER_IMPL or runtime/router_plan.py's default), None if unknown."""
    try:
        from runtime.router_plan import router_impl

        return router_impl()
    except Exception:  # an old tree without router_plan / a bad env value: recorded as unknown
        return None


def _glue_on(st) -> Optional[List[str]]:
    g = getattr(st, "glue", None)
    if g is None:
        return None
    from runtime.fused_glue import CLUSTERS

    return [c for c in CLUSTERS if g.on(c)]


def summarize(rows: List[dict]) -> dict:
    """Per run: rows compared / equal (taps and logits), the earliest differing row and its first tap, a histogram
    of first-differing tap families."""
    n = len(rows)
    bad = [r for r in rows if r["first"] is not None or not r["logits_equal"]]
    hist = collections.Counter()
    for r in bad:
        f = r["first"]
        hist[family(f["key"][1]) if f else "logits only"] += 1
    origin = None
    if bad:
        r = min(bad, key=lambda r: (r["step"], r["row"]))
        origin = {"step": r["step"], "row": r["row"], "pos": r["pos"], "first": r["first"],
                  "logits": r["logits"], "argmax_equal": r["argmax_equal"], "margin_1row": r["margin_1row"]}
    return {"rows_compared": n, "rows_all_taps_equal": sum(r["first"] is None for r in rows),
            "rows_logits_equal": sum(r["logits_equal"] for r in rows),
            "rows_argmax_equal": sum(r["argmax_equal"] for r in rows),
            "first_tap_histogram": dict(hist.most_common()), "origin": origin}


def report_lines(rep: dict) -> List[str]:
    s = rep["summary"]
    out = [f"ROW-DIVERGENCE PROBE | impls {rep['impls']} | rowwise {rep['rowwise']} | row_invariant_mode "
           f"{rep['row_invariant_mode']} | plan {rep['plan']} | stream {rep['stream']} | L {rep['L']}",
           f"  rows compared {s['rows_compared']}: all taps equal {s['rows_all_taps_equal']}, logits bit-equal "
           f"{s['rows_logits_equal']}, argmax equal {s['rows_argmax_equal']}"]
    o = s["origin"]
    if o is None:
        out.append("  first divergence: none (every accepted row bit-equal to the 1-row step, every tap)")
    else:
        f = o["first"]
        what = (f"layer {f['key'][0]} {f['key'][1]}#{f['key'][2]}: {f.get('mism')}/{f.get('n')} differ, max ulp "
                f"{f.get('max_ulp', '-')}" + (f", only-6row ids {f.get('only_6row')}" if 'only_6row' in f else "")
                if f else "no tap differs, logits do")
        out.append(f"  first divergence: step {o['step']} row {o['row']} (pos {o['pos']}): {what}; logits max ulp "
                   f"{o['logits'].get('max_ulp')}, argmax equal {o['argmax_equal']}, 1-row margin "
                   f"{o['margin_1row']:.4f}")
        out.append(f"  first-tap histogram (rows): {s['first_tap_histogram']}")
    bad = [r for r in rep["rows"] if r["first"] is not None or not r["logits_equal"]]
    shown = bad[:24] if bad else rep["rows"][:6]
    if bad:
        out.append(f"  differing rows ({len(bad)}; first 24): step/row/pos, accepted n, first tap, the next taps")
    for r in shown:
        f = r["first"]
        out.append(f"    step {r['step']:2d} row {r['row']} pos {r['pos']:4d} n {r['n_acc']}: "
                   + ("equal" if f is None else f"L{f['key'][0]} {f['key'][1]} ulp {f.get('max_ulp', '-')} "
                      f"({r['n_diff']} taps differ; then {[tuple(k[:2]) for k in r['diff_keys'][1:4]]})")
                   + f"; logits {'=' if r['logits_equal'] else '!='}, argmax {'=' if r['argmax_equal'] else '!='}, "
                     f"1-row margin {r['margin_1row']:.4f}")
    out.append(f"  router impl {rep.get('router_impl')}; glue on (6 rows) {rep.get('glue_on', {}).get('6')}; "
               f"FlashMLA on {rep.get('attn_on')}; env {rep.get('env')}")
    return out


# ------------------------------------------------------------------------------------------------ real-width census
def real_width_census(dev, rows: int = ROWS, seed: int = 0, W: Sequence[int] = (2048, 4096),
                      head: bool = False, flashmla: bool = True) -> List[dict]:
    """Synthetic activations / weights at DeepSeek-V4.1-Flash's real widths (reference/deepseek-inference/
    config.json: dim 5120, 384 routed experts, head_dim 512, index 32 x 128, o_groups 8 x o_lora 1024, hc 4) for the
    step's sites that the mini census cannot reach (cuBLAS picks its kernel from M, N, K): row r of the op at `rows`
    rows vs the op on row r alone (runtime/row_census.compare records). FlashMLA at the real widths (window 128 +
    512 compressed, -1 tails, per-row lengths) plain and row-split, when the .so loads."""
    from runtime.row_census import compare

    dev = torch.device(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device="cpu") * scale).to(dtype).to(dev)

    D, E, HD, NI, DI, G, OR, HC = 5120, 384, 512, 32, 128, 8, 1024, 4
    out: List[dict] = []

    def rec(family_, op, shape, fn):
        try:
            r = fn()
            out.append(dict(r, family=family_, op=op, shape=shape))
        except Exception as e:  # recorded, never fatal
            out.append({"family": family_, "op": op, "shape": shape, "equal": None,
                        "note": f"n/a: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"})

    def lin_case(x, w):
        return compare(F.linear(x, w), [F.linear(x[i:i + 1], w) for i in range(x.size(0))])

    x32 = rnd(rows, D).float()
    xbf = rnd(rows, D)
    rec("cuBLAS fp32 F.linear", "router gate x.float() @ W.float()  (40 layers)", f"[{rows},{D}]x[{E},{D}]",
        lambda: lin_case(x32, rnd(E, D, scale=0.02).float()))
    rec("cuBLAS fp32 F.linear", "compressor wkv / wgate, ratio 2 (layers 2, 8, 14)", f"[{rows},{D}]x[{HD},{D}]",
        lambda: lin_case(x32, rnd(HD, D, scale=0.02).float()))
    rec("cuBLAS bf16 F.linear", "compressor wkv, ratio 1 (layer 20)", f"[{rows},{D}]x[{HD},{D}]",
        lambda: lin_case(xbf, rnd(HD, D, scale=0.02)))
    rec("cuBLAS bf16 F.linear", "indexer weights_proj", f"[{rows},{D}]x[{NI},{D}]",
        lambda: lin_case(xbf, rnd(NI, D, scale=0.02)))
    for m in sorted({max(1, rows // 2), rows}):
        rec("cuBLAS bf16 F.linear", f"indexer wk on {m} latent rows", f"[{m},{HD}]x[{DI},{HD}]",
            lambda m=m: lin_case(rnd(m, HD), rnd(DI, HD, scale=0.05)))
    rec("cuBLAS fp32 F.linear", "mHC mixes (cuBLAS; --fp32-impl fixed replaces it)", f"[{rows},{HC * D}]x[24,{HC * D}]",
        lambda: lin_case(rnd(rows, HC * D).float(), rnd(24, HC * D, scale=0.01).float()))
    if head:
        rec("cuBLAS fp32 F.linear", "head (cuBLAS; --fp32-impl fixed replaces it)", f"[{rows},{D}]x[129280,{D}]",
            lambda: lin_case(x32, rnd(129280, D, scale=0.02).float()))
    o6 = rnd(1, rows, G, HD * 64 // G)
    wo_a = rnd(G, OR, HD * 64 // G, scale=0.02)
    rec("einsum bf16 (cuBLAS bmm)", "wo_a bsgd,grd->bsgr", f"[1,{rows},{G},{HD * 64 // G}]x[{G},{OR},.]",
        lambda: compare(torch.einsum("bsgd,grd->bsgr", o6, wo_a),
                        [torch.einsum("bsgd,grd->bsgr", o6[:, i:i + 1], wo_a) for i in range(rows)]))
    q6 = rnd(1, rows, NI, DI)
    for w_ in W:
        kk = rnd(1, w_, DI)
        rec("einsum bf16 (cuBLAS bmm)", f"indexer bshd,btd->bsht W={w_}", f"[1,{rows},{NI},{DI}]x[1,{w_},{DI}]",
            lambda kk=kk: compare(torch.einsum("bshd,btd->bsht", q6, kk),
                                  [torch.einsum("bshd,btd->bsht", q6[:, i:i + 1], kk) for i in range(rows)]))

        def topk_case(w_=w_):
            sc = (torch.randint(0, 64, (rows, w_), generator=g, device="cpu").float() / 8).to(dev)  # heavy ties
            full = sc.topk(min(512, w_), dim=-1)
            parts = [sc[i:i + 1].topk(min(512, w_), dim=-1) for i in range(rows)]
            r = compare(full.indices, [p.indices for p in parts])
            r["note"] = f"k={min(512, w_)} of {w_}, ties; values equal " + str(
                all(torch.equal(full.values[i], parts[i].values[0]) for i in range(rows)))
            return r

        rec("torch.topk", f"indexer top-512 of W={w_} (ties)", f"[{rows},{w_}]", topk_case)
    if flashmla and dev.type == "cuda":
        rec("sparse_attn FlashMLA", "m=6 vs m=1, real widths (128 + 512, tlen)", "h=64 d=512",
            lambda: _flashmla_case(dev, rows, g, split=False))
        rec("sparse_attn FlashMLA", "row-split (DSV41_FLASHMLA_ROW_SPLIT) vs one call", "h=64 d=512",
            lambda: _flashmla_case(dev, rows, g, split=True))
    return out


def _flashmla_case(dev, rows, g, split: bool) -> dict:
    from runtime import attn_flashmla as AFM
    from runtime.row_census import compare

    H, Dh, win, k, ring = 64, 512, 128, 512, 256
    n = ring + 2048
    q = (torch.randn(1, rows, H, Dh, generator=g, device="cpu") * 0.5).to(torch.bfloat16).to(dev)
    kv = (torch.randn(1, n, Dh, generator=g, device="cpu") * 0.5).to(torch.bfloat16).to(dev)
    sink = (torch.randn(H, generator=g, device="cpu") * 0.1).float().to(dev)
    idx = torch.full((1, rows, win + k), -1, dtype=torch.int32, device="cpu")
    tl = torch.zeros(rows, dtype=torch.int32, device="cpu")
    for j in range(rows):
        p = 700 + j
        idx[0, j, :win] = torch.arange(p - win + 1, p + 1, dtype=torch.int32, device="cpu") % ring
        c = min(k, (p + 1) // 2 - j % 3)
        idx[0, j, win:win + c] = torch.randperm(2048, generator=g, device="cpu")[:c].sort().values.int() + ring
        tl[j] = win + c
    idx, tl = idx.to(dev), tl.to(dev)
    full = AFM.sparse_attn_flashmla(q, kv, sink, idx, Dh ** -0.5, tl, row_split=False)
    if split:
        return compare(AFM.sparse_attn_flashmla(q, kv, sink, idx, Dh ** -0.5, tl, row_split=True), [
            full[:, j:j + 1] for j in range(rows)])
    return compare(full, [AFM.sparse_attn_flashmla(q[:, j:j + 1].contiguous(), kv, sink, idx[:, j:j + 1].contiguous(),
                                                   Dh ** -0.5, tl[j:j + 1], row_split=False) for j in range(rows)])


# ------------------------------------------------------------------------------------------------ fixtures
MINI4 = {"n_layers": 4, "compress_ratios": [0, 2, 1, 1, 0], "kv_source_layers": [1, 2],
         "index_source_layers": [1, 2, 3], "candidate_source_layer": 2, "candidate_topk_blocks": 64,
         "candidate_block_size": 8, "index_topk": 64, "dspark_target_layer_ids": [3], "n_activated_experts": 4}
MINI4_HOT = {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6], "2": [0, 1, 2, 3], "3": [4, 5, 6, 7]}


@contextlib.contextmanager
def mini_model(kind: str = "mini4", device: str = "cuda"):
    """tests/gpu/test_static_rows_gpu.py's `_model` (mini / mini4 / mini4w on cuda) -> (model, pool)."""
    from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    import engram

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    cfg, hot = None, None
    if kind in ("mini4", "mini4w"):
        cfg = dict(MINI_CONFIG, **MINI4)
        if kind == "mini4w":
            cfg.update({"index_topk": 512})
        hot = MINI4_HOT
    with tempfile.TemporaryDirectory() as tmp:
        ck = make_mini_checkpoint(os.path.join(tmp, "ckpt"), cfg=cfg) if cfg else \
            make_mini_checkpoint(os.path.join(tmp, "ckpt"))
        model = build_and_load(ck["model_dir"], hot if hot is not None else ck["map"], device, tokenizer=object(),
                               io_threads=4, stage_bytes=1 << 20, log=lambda *a: None)
        model.temperature = 0.0
        res = model.runtime_load
        try:
            yield model, res.pool
        finally:
            torch.set_default_device("cpu")
            del model
            if device == "cuda":
                torch.cuda.synchronize()
            res.release()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--kind", default="mini4", choices=["mini", "mini4", "mini4w"], help="mini checkpoint (default)")
    ap.add_argument("--model-dir", help="the real checkpoint (overrides --kind)")
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--fixture", help="prompt jsonl (real model; with --tag)")
    ap.add_argument("--tag", help="prompt tag in --fixture (e.g. prose)")
    ap.add_argument("--prompt", help="prompt text (real model)")
    ap.add_argument("--impls", default="headline", choices=["headline", "ref"])
    for k in ("moe", "fp8", "glue", "attn"):
        ap.add_argument(f"--{k}-impl", help=f"override the {k} impl")
    ap.add_argument("--fp32-impl", default="fixed", choices=["fixed", "cublas"])
    ap.add_argument("--moe-rowmatch", default="on", choices=["on", "off"])
    ap.add_argument("--no-gemv-pin", action="store_true")
    ap.add_argument("--glue-order", default=None, choices=[None, "shape", "rows"])
    ap.add_argument("--router-impl", default=None, choices=[None, "fused", "plan", "plan-gemv"])
    ap.add_argument("--flashmla-row-split", action="store_true", help="DSV41_FLASHMLA_ROW_SPLIT=1")
    ap.add_argument("--rowwise-linear", default="off", choices=list(ROWWISE_MODES))
    ap.add_argument("--row-invariant-mode", action="store_true")
    ap.add_argument("--plan", default=None, help="accepts per 6-row step, e.g. 5,0,2,5,1,3 (default all 5)")
    ap.add_argument("--plan-from", default=None, metavar="HEADLINE_JSON:TAG",
                    help="replay a headline run's accept log for TAG (P0 sequence = the loop's; also the default "
                         "--tag); e.g. docs/headline-20261107i-plan-gemv.json:shell")
    ap.add_argument("--reject-variants", type=int, default=1,
                    help="run the plan with N different wrong-token sets in the rejected rows; every accepted row's "
                         "logits must be identical across them (a difference = a rejected row's write leaks)")
    ap.add_argument("--steps", type=int, default=4, help="6-row steps when --plan is not given")
    ap.add_argument("--stream", default=None, choices=[None, "greedy", "random"],
                    help="token stream (default: greedy on the real model, random on the mini)")
    ap.add_argument("--L", type=int, default=18, help="random prompt length (mini)")
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--max-pos", type=int, default=None)
    ap.add_argument("--spare-slots", type=int, default=None)
    ap.add_argument("--census-only", action="store_true", help="only the synthetic real-width census")
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--census-head", action="store_true", help="include the 2.6 GB fp32 head in the census")
    ap.add_argument("--json", default=None, help="receipt path (on the box: scratch/...)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    setenv = dict(HEADLINE_ENV, DSV41_FP32_IMPL=a.fp32_impl, DSV41_MOE_ROWMATCH=a.moe_rowmatch)
    if a.no_gemv_pin:
        setenv["DSV41_GEMV_PIN_M1"] = None
    if a.glue_order:
        setenv["DSV41_GLUE_ORDER"] = a.glue_order
    if a.router_impl:
        setenv["DSV41_ROUTER_IMPL"] = a.router_impl
    if a.flashmla_row_split:
        setenv["DSV41_FLASHMLA_ROW_SPLIT"] = "1"
    for k, v in setenv.items():                      # before any build (the switches are read at build/capture)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv}
    if not a.no_census and dev == "cuda":
        recs = real_width_census(dev, head=a.census_head)
        out["real_width_census"] = recs
        from runtime.row_census import census_table

        print("REAL-WIDTH CENSUS (synthetic; row r at 6 rows vs the same row alone)")
        for line in census_table(recs):
            print("  " + line)
        torch.cuda.empty_cache()
    if a.census_only:
        _write(a.json, out)
        return 0
    if dev != "cuda":
        print("no CUDA: the probe runs on the box (the Mac dry run is tests/test_row_divergence_cpu.py)")
        return 2
    from runtime.generate import set_determinism

    impls = dict(HEADLINE_IMPLS) if a.impls == "headline" else {}
    for k in ("moe", "fp8", "glue", "attn"):
        v = getattr(a, f"{k}_impl")
        if v:
            impls[f"{k}_impl"] = v
    plan = [int(x) for x in a.plan.split(",")] if a.plan else [5] * a.steps
    if a.plan_from:
        plan, tag = plan_from(a.plan_from)
        a.tag = a.tag or tag

    def probe(model, pool, **kw):
        reps = [run_probe(model, pool, impls=impls, plan=plan, seed=a.seed, rowwise=a.rowwise_linear,
                          row_invariant=a.row_invariant_mode, reject_variant=v, **kw)
                for v in range(max(1, a.reject_variants))]
        rep = reps[0]
        if len(reps) > 1:
            rep["reject_variant_leaks"] = variant_leaks(reps)
        return rep

    if a.model_dir:
        from runtime.loader import build_and_load
        from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS

        model = build_and_load(a.model_dir, a.map if a.map and os.path.exists(a.map) else None, "cuda")
        model.temperature = 0.0
        out["determinism"] = set_determinism(0)
        pool = model.runtime_load.pool
        spare = a.spare_slots or DEFAULT_SPARE_SLOTS_ROWS
        if impls.get("moe_impl") == "dg-inplace" and pool is not None:
            pool.add_spare_slots(spare, model)       # once, before any StaticStep (skill pitfall: never re-grow)
        ids = _encode(a.model_dir, a.fixture, a.tag, a.prompt)
        rep = probe(model, pool, max_pos=a.max_pos or 4095, spare_slots=spare, stream=a.stream or "greedy", ids=ids)
    else:
        with mini_model(a.kind) as (model, pool):
            out["determinism"] = set_determinism(0)
            rep = probe(model, pool, L=a.L, max_pos=a.max_pos or 511, spare_slots=a.spare_slots or 8,
                        stream=a.stream or "random")
    out["probe"] = rep
    for line in report_lines(rep):
        print(line)
    if "reject_variant_leaks" in rep:
        lk = rep["reject_variant_leaks"]
        print(f"  rejected-row variants: {a.reject_variants}; accepted rows whose logits change with the rejected "
              f"tokens: {len(lk)} {lk[:12]}")
    _write(a.json, out)
    return 0


def variant_leaks(reps: List[dict]) -> List[list]:
    """(step, row, pos) of every accepted row whose logits differ between runs that differ ONLY in the rejected
    rows' tokens (same plan, same stream): a non-empty list = some cache a rejected row wrote is read before it is
    overwritten (dspark_loop.py's cache table says none is)."""
    base = {(r["step"], r["row"]): r["logits_sha"] for r in reps[0]["rows"]}
    out = []
    for rep in reps[1:]:
        for r in rep["rows"]:
            if base.get((r["step"], r["row"])) != r["logits_sha"]:
                out.append([r["step"], r["row"], r["pos"]])
    return out


def plan_from(spec: str) -> Tuple[List[int], str]:
    """`HEADLINE_JSON:TAG` -> (that prompt's accept log, TAG): replaying it puts every 6-row step at the loop's own
    P0 (the stream is the 1-row greedy one, which the loop emits up to its first divergence)."""
    path, _, tag = spec.rpartition(":")
    if not path or not tag:
        raise ValueError(f"--plan-from wants HEADLINE_JSON:TAG, got {spec!r}")
    with open(path) as f:
        d = json.load(f)
    pr = (d.get("prompts") or {}).get(tag)
    if not pr or not pr.get("accepts"):
        raise ValueError(f"{path}: no accept log for prompt {tag!r} (have {sorted((d.get('prompts') or {}))})")
    return [int(n) for n in pr["accepts"]], tag


def _encode(model_dir: str, fixture: Optional[str], tag: Optional[str], prompt: Optional[str]) -> List[int]:
    """Prompt ids through the checkpoint's own encoding (runtime.generate.Generator.encode, as bench/headline.py)."""
    text = prompt
    if text is None and fixture:
        with open(fixture) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        row = next((r for r in rows if tag is None or r.get("tag") == tag), None)
        if row is None:
            raise SystemExit(f"tag {tag!r} not in {fixture}: {[r.get('tag') for r in rows]}")
        text = row["prompt"]
    if text is None:
        text = "Write a Python function that reverses a linked list."
    from runtime.generate import encode_messages_to_ids, load_encoding, load_tokenizer

    enc, _ = load_encoding(model_dir)
    return list(encode_messages_to_ids(enc, load_tokenizer(model_dir), [{"role": "user", "content": text}], "chat"))


def _write(path: Optional[str], obj: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"[json] {path}")


if __name__ == "__main__":
    sys.exit(main())
