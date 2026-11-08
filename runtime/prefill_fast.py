"""M4-P2: the fast prefill, bit-identical -- layer-major, the routed experts' GEMMs issued as one batch per projection
per layer (docs/HANDOFF-M4-P2-FAST-PREFILL.md; design docs/M4-PREFILL-DESIGN.md 4-7).

    $DSV41_PREFILL=reference|grouped   (default reference until the box gate; runtime/draft_step.py prefill_hidden)
    $DSV41_PREFILL_FILLS=off|on        (default off: torch's fill_uninitialized_memory scoped OFF around the prefill;
                                        box 2026-11-08: -0.55 s, logits + main_hidden bit-equal 8/8, off==on True)
    $DSV41_PREFILL_STREAMS=N           (grouped: CUDA streams the per-expert GEMM launches are spread over; default 4;
                                        1 = everything on the current stream)

What the reference prefill does in the MoE (model.py:891-906), per layer: gate -> ONE `bincount().tolist()` (host
sync) -> for every routed expert in ascending id: `torch.where(indices == i)` (a second host sync, per expert), gather
x[idx], then the reference `Expert` (model.py:843-853: act_quant + fp4_gemm for w1, again for w3, SwiGLU with the
clamps, `weights * x`, bf16, act_quant + fp4_gemm for w2), then `y[idx] += out` (fp32) -> + shared expert. Box
(docs/prefill-anatomy-20261108.log + -prose-analysis): 3,312-3,672 such Expert calls per prefill, 26 launches and
~1.0 ms wall each (~95k launches, GPU busy 19 %): launches, not syncs, are the cost.

`grouped_moe_forward` keeps every arithmetic op of that path and only removes the per-expert Python glue:
  1. gate: the reference `Gate` module itself (same call).
  2. routing, once per layer: a stable sort of the flattened (row, k) pair ids by expert = `torch.where`'s row order
     per expert (row-major nonzero), ONE `bincount().tolist()` (the sync the reference already makes at 896), no
     per-expert sync. Each routed expert e gets a SEGMENT of a layer-wide row buffer: c_e rows starting at a base
     that is a multiple of 32 (the reference GEMMs' block_M, kernel.py:255/500), padded to the next multiple of 32.
     ONE `index_select` gathers x into that layout (pad rows copy token row 0, weight 0).
  3. act_quant ONCE over the whole buffer (`model.act_quant` with linear()'s exact arguments, model.py:187). act_quant
     is per row (amax over each row's own 32-wide groups, kernel.py:40-89): its bytes for a row do not depend on the
     other rows -- the box census at 42/64 rows says row-invariant, and the reference itself quantizes the same rows
     twice (w1, w3) with identical results.
  4. the GEMMs: per expert, the reference TileLang `fp4_gemm_kernel` (kernel.py:478-559, the SAME compiled kernel
     the reference wrapper fp4_gemm, kernel.py:562-595, launches) on that expert's segment: A = the segment's c rows
     (M = c, exactly the reference's M, so the grid ceil(N/128) x ceil(M/32) and the swizzle are the reference's),
     B / scales = the expert's own `w.weight` / `w.weight.scale` (the pool views: hot HBM, cold UVA Grace -- read in
     kernel as today), C = a VIEW into the layer-wide output buffer. A segment base is a multiple of 32 rows, so
     every A / scale / C view starts on a 64-byte boundary (TVM's assumed buffer alignment; checked per launch) and
     act_quant's 32-row tiles never straddle two experts. Nothing else is launched per expert: no
     where(), no gather, no act_quant, no SwiGLU, no allocation, no combine. With N streams the experts' launches are
     spread over N CUDA streams (each launch writes only its own segment; a 32-aligned, 32-padded segment holds even
     an unguarded tile store, so concurrent launches never touch each other's rows); the current stream waits for
     all of them before anything reads the results. Order of execution cannot change a value: no launch reads
     another's output.
  5. SwiGLU on all segment rows at once, the reference's ops in the reference's order: `.float()` x2, clamp(up,
     -lim, lim), clamp(gate, max=lim), silu(gate) * up, weights * h, .to(x.dtype) (all elementwise: a row's values
     depend on that row only); act_quant once; the w2 GEMMs like 4.
  6. combine: `y = zeros fp32`, then for j = 0..k-1: `y += out[pos[:, j]]` where pos[r] are row r's k segment
     positions sorted ascending = its experts in ascending id (bases grow with the expert id). That is exactly the
     reference's per-row fp32 sum `((0 + o_e1) + o_e2) + ...` in ascending expert id (each row has k distinct experts:
     topk), the same fp32 add of the same bf16 operand (model.py:902 `y[idx] += out`). Then `y += shared(x)` and
     `y.type_as(x)` -- the reference lines 905-906.

Everything outside the routed experts (attention, compressor, indexer, KV caches, mHC, Engram, norms, head) is the
reference forward untouched: `prefill_scope` only installs this function as the INSTANCE-level `forward` of each
`model.layers[i].ffn` for the duration of the prefill (removed on exit; the class-level static-MoE patch of
runtime/patches.py is not touched and wins again afterwards).

Launch path: the reference kernel object is called directly only when `model.fp4_gemm` IS kernel.py's `fp4_gemm`
(unpatched) -- the same `fp4_gemm_kernel(N, K, act_block_size=..., scale_dtype=...)` call the wrapper makes, then
`kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)` with c a view. Otherwise (the Mac's CPU emulations, any
patched fp4_gemm) each expert calls `model.fp4_gemm` itself on its c rows and the result is copied into the segment
(and streams are 1). Experts that are not fp4 (or world_size > 1, or an image mask) fall back to the class forward.

`last_prefill()` returns the counters of the last prefill: mode, fills, streams, layers, expert calls (= distinct
routed experts, = the reference's Expert calls), GEMM launches, pairs, padded rows, glue ops (host count of the torch
ops this module issues per layer, gate and shared expert excluded), cold calls and the C2C bytes they read (the
census formula of bench/prefill_anatomy.py), the per-layer fallbacks, the host ms of the forward.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

PREFILL_ENV = "DSV41_PREFILL"
FILLS_ENV = "DSV41_PREFILL_FILLS"
STREAMS_ENV = "DSV41_PREFILL_STREAMS"
MODES = ("reference", "grouped")
DEFAULT_MODE = "reference"     # until the box gate (handoff): flip to "grouped" only on the box receipts
DEFAULT_FILLS = "off"          # box 2026-11-08: fills off == on, 8/8 prompts, -0.55 s
DEFAULT_STREAMS = 4
BLOCK_M = 32                   # kernel.py fp4_gemm_kernel / act_quant_kernel block_M (rows per tile)
MAX_STREAMS = 16

_LAST: Dict = {}
_STREAMS: Dict = {}            # (device index, n) -> [torch.cuda.Stream]
_KERNELS: Dict = {}            # (N, K, act_block_size, tl scale dtype) -> the reference TileLang kernel object


# ------------------------------------------------------------------------------------------------ switches
def prefill_mode(v: Optional[str] = None) -> str:
    """The prefill in force: the argument, else $DSV41_PREFILL, else "reference"."""
    s = (v or os.environ.get(PREFILL_ENV) or DEFAULT_MODE).strip().lower()
    if s not in MODES:
        raise ValueError(f"prefill {s!r} not in {MODES} (${PREFILL_ENV})")
    return s


def prefill_fills(v: Optional[str] = None) -> bool:
    """True = torch's fill_uninitialized_memory left as the caller has it (on under set_determinism); False = scoped
    off around the prefill (the default)."""
    s = (v or os.environ.get(FILLS_ENV) or DEFAULT_FILLS).strip().lower()
    if s not in ("on", "off"):
        raise ValueError(f"{FILLS_ENV}={s!r}: on|off")
    return s == "on"


def prefill_streams(v: Optional[int] = None) -> int:
    n = int(v if v is not None else (os.environ.get(STREAMS_ENV) or DEFAULT_STREAMS))
    if not 1 <= n <= MAX_STREAMS:
        raise ValueError(f"{STREAMS_ENV}={n}: 1..{MAX_STREAMS}")
    return n


def last_prefill() -> dict:
    return dict(_LAST)


def _ref():
    m = sys.modules.get("model")
    if m is None:
        raise RuntimeError("the reference model module is not imported (runtime.refmodel.import_reference)")
    return m


@contextlib.contextmanager
def fills_scope(on: bool):
    """torch.utils.deterministic.fill_uninitialized_memory set to `on` for the scope (restored after)."""
    det = getattr(torch.utils, "deterministic", None)
    prev = getattr(det, "fill_uninitialized_memory", None)
    try:
        if det is not None and prev is not None:
            det.fill_uninitialized_memory = bool(on)
        yield
    finally:
        if det is not None and prev is not None:
            det.fill_uninitialized_memory = prev


# ------------------------------------------------------------------------------------------------ GEMM launch
def _direct_kernel_module():
    """kernel.py when `model.fp4_gemm` is its unpatched `fp4_gemm` and it has the TileLang kernel factory; else None."""
    ref = _ref()
    kern = sys.modules.get("kernel")
    if kern is None or not hasattr(kern, "fp4_gemm_kernel"):
        return None
    if getattr(ref, "fp4_gemm", None) is not getattr(kern, "fp4_gemm", object()):
        return None
    return kern


def _fp4_kernel(kern, N: int, K: int, scale_dtype, act_block_size: int):
    """The object kernel.py's fp4_gemm builds (kernel.py:584-590, same arguments), cached by its arguments."""
    tl_dtype = kern.FE8M0 if scale_dtype == torch.float8_e8m0fnu else kern.FP32
    key = (id(kern), N, K, act_block_size, tl_dtype)
    k = _KERNELS.get(key)
    if k is None:
        k = kern.fp4_gemm_kernel(N, K, act_block_size=act_block_size, scale_dtype=tl_dtype)
        _KERNELS[key] = k
    return k


def _check_weight(w: torch.Tensor, K: int) -> None:
    """kernel.py fp4_gemm's asserts on B (kernel.py:573-582) -- the A / C side is contiguous by construction."""
    s = w.scale
    if not (w.is_contiguous() and s.is_contiguous()):
        raise AssertionError("fp4 expert weight / scale not contiguous (kernel.py:573-574)")
    if tuple(s.shape) != (w.size(0), K // 32):
        raise AssertionError(f"fp4 expert scale shape {tuple(s.shape)} != {(w.size(0), K // 32)} (kernel.py:582)")


class _Launcher:
    """Issues one projection's per-expert GEMMs over the segments (module docstring 4)."""

    def __init__(self, kern, streams: List, scale_dtype, act_block: int):
        self.kern, self.streams, self.scale_dtype, self.act_block = kern, streams, scale_dtype, act_block
        self.launches = 0

    def run(self, segs, a, a_s, out_by_proj: Dict[str, torch.Tensor], weights_by_proj) -> None:
        """segs: [(expert, base, c)]; weights_by_proj: {proj: [weight per seg]} -> out_by_proj[proj][base:base+c]."""
        ref = _ref()
        K = a.size(-1)
        if self.kern is None:                                   # wrapper path (emulation / patched fp4_gemm)
            for j, (_, b0, c) in enumerate(segs):
                for p, ws in weights_by_proj.items():
                    w = ws[j]
                    o = ref.fp4_gemm(a[b0:b0 + c], a_s[b0:b0 + c], w, w.scale, self.scale_dtype,
                                     act_block_size=self.act_block)
                    out_by_proj[p][b0:b0 + c].copy_(o)
                    self.launches += 1
            return
        kernels = {p: _fp4_kernel(self.kern, ws[0].size(0), K, self.scale_dtype, self.act_block)
                   for p, ws in weights_by_proj.items()}
        for p, ws in weights_by_proj.items():
            for w in ws:
                _check_weight(w, K)

        def issue(js):
            for j in js:
                _, b0, c = segs[j]
                av, sv = a[b0:b0 + c], a_s[b0:b0 + c]
                if (av.data_ptr() | sv.data_ptr()) % 64:
                    raise AssertionError(f"segment at row {b0}: A / scale views not 64-B aligned")
                for p, ws in weights_by_proj.items():
                    w = ws[j]
                    ov = out_by_proj[p][b0:b0 + c]
                    if ov.data_ptr() % 64:
                        raise AssertionError(f"segment at row {b0}: C view not 64-B aligned")
                    kernels[p](av.view(c, K), w, ov.view(c, w.size(0)), sv.view(c, -1), w.scale)
                    self.launches += 1

        if len(self.streams) <= 1:
            issue(range(len(segs)))
            return
        main = torch.cuda.current_stream(a.device)
        n = len(self.streams)
        used = self.streams[:min(n, len(segs))]
        for s in used:
            s.wait_stream(main)                                 # inputs (act_quant, buffers) are ready
        for i, s in enumerate(used):
            with torch.cuda.stream(s):
                issue(range(i, len(segs), n))
        for s in used:
            main.wait_stream(s)                                 # nothing reads a segment before its launch is done


def _streams_for(device, n: int) -> List:
    if n <= 1 or device.type != "cuda":
        return []
    key = (device.index if device.index is not None else torch.cuda.current_device(), n)
    ss = _STREAMS.get(key)
    if ss is None:
        ss = [torch.cuda.Stream(device=device) for _ in range(n)]
        _STREAMS[key] = ss
    return ss


# ------------------------------------------------------------------------------------------------ the MoE
GLUE_OPS_FIXED = 30   # ~torch ops (host count) grouped_moe_forward issues per layer besides the GEMMs, gate and shared
#                       expert; + 2 per topk slot (the combine). The box's kineto count (bench/prefill_bench.py) is exact.


def _fallback_reason(moe, image_mask) -> Optional[str]:
    ref = _ref()
    if getattr(ref, "world_size", 1) != 1:
        return "world_size"
    if image_mask is not None:
        return "image_mask"
    for i in range(moe.experts_start_idx, moe.experts_end_idx):
        ex = moe.experts[i]
        if ex is None:
            return "expert_none"
        for p in ("w1", "w2", "w3"):
            if getattr(ex, p).weight.dtype != torch.float4_e2m1fn_x2:
                return "not_fp4"
        break
    return None


def grouped_moe_forward(moe, x: torch.Tensor, image_mask=None, *, streams: int = 1, stats: Optional[dict] = None,
                        pool=None) -> torch.Tensor:
    """The reference MoE.forward (model.py:891-906) with the routed experts batched (module docstring). Bit-equal by
    construction; falls back to the class forward when it does not apply."""
    why = _fallback_reason(moe, image_mask)
    if why is not None:
        if stats is not None:
            stats.setdefault("fallback", []).append({"layer": getattr(moe, "layer_id", None), "why": why})
        return type(moe).forward(moe, x, image_mask)
    ref = _ref()
    t0 = time.perf_counter()
    shape = x.size()
    x = x.view(-1, moe.dim)                                                   # 892-893
    weights, indices = moe.gate(x, None)                                      # 894 (image_mask is None here)
    y = torch.zeros_like(x, dtype=torch.float32)                              # 895
    dev = x.device
    L, k = indices.size(0), indices.size(1)
    flat = indices.flatten()
    order = torch.sort(flat, stable=True).indices                             # row-major pair order per expert
    counts = torch.bincount(flat, minlength=moe.n_routed_experts).tolist()    # 896: the layer's one host sync
    segs, shift, base, start = [], [0] * moe.n_routed_experts, 0, 0
    for e in range(moe.experts_start_idx, moe.experts_end_idx):
        c = counts[e]
        if c == 0:
            continue
        segs.append((e, base, c))
        shift[e] = base - start
        start += c
        base += (c + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    P, Pp = start, base
    sh = torch.tensor(shift, dtype=torch.long, device="cpu").to(dev)
    rows = torch.div(order, k, rounding_mode="floor")                         # token row of each sorted pair
    pos = torch.arange(P, dtype=torch.long, device=dev) + sh[flat[order]]     # its row in the padded layout
    src = torch.zeros(Pp, dtype=torch.long, device=dev).index_copy_(0, pos, rows)
    wp = torch.zeros(Pp, 1, dtype=weights.dtype, device=dev).index_copy_(0, pos, weights.flatten()[order].unsqueeze(-1))
    xs = x.index_select(0, src)                                               # ONE gather (pad rows = token row 0)
    fbs, sfmt, sdt = ref.fp8_block_size, ref.scale_fmt, ref.scale_dtype
    kern = _direct_kernel_module()
    lau = _Launcher(kern, _streams_for(dev, streams) if kern is not None else [], sdt, fbs)
    exs = [moe.experts[e] for e, _, _ in segs]
    odt = torch.get_default_dtype()                                           # kernel.py:583 output dtype
    inter = exs[0].w1.weight.size(0)
    # w1 / w3 (model.py:845-846 -> linear() 186-195)
    xq, xs_s = ref.act_quant(xs, fbs, sfmt, sdt)
    g = torch.empty(Pp, inter, dtype=odt, device=dev)
    u = torch.empty(Pp, inter, dtype=odt, device=dev)
    lau.run(segs, xq, xs_s, {"w1": g, "w3": u}, {"w1": [ex.w1.weight for ex in exs], "w3": [ex.w3.weight for ex in exs]})
    # SwiGLU (847-852), all rows at once, the reference's ops in its order
    gate = g.float()
    up = u.float()
    lim = exs[0].swiglu_limit
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    h = wp * h
    h = h.to(x.dtype)
    # w2 (853)
    hq, hs = ref.act_quant(h, fbs, sfmt, sdt)
    o = torch.empty(Pp, exs[0].w2.weight.size(0), dtype=odt, device=dev)
    lau.run(segs, hq, hs, {"w2": o}, {"w2": [ex.w2.weight for ex in exs]})
    # combine (902): per row, ascending expert id = ascending segment position
    inv = torch.argsort(order, stable=True)                                   # sorted position of pair (r, t)
    pm = pos.index_select(0, inv).view(L, k).sort(dim=1).values
    for j in range(k):
        y += o.index_select(0, pm[:, j])
    y += moe.shared_experts(x)                                                # 905
    out = y.type_as(x).view(shape)                                            # 906
    if stats is not None:
        cold = 0
        if pool is not None:
            from runtime.expert_pool import TIER_HOST

            lid = getattr(moe, "layer_id", None)
            cold = sum(1 for e, _, _ in segs if lid is not None and (lid, e) in pool.slot and pool.tier(lid, e) == TIER_HOST)
        stats["layers"] = stats.get("layers", 0) + 1
        stats["expert_calls"] = stats.get("expert_calls", 0) + len(segs)
        stats["cold_calls"] = stats.get("cold_calls", 0) + cold
        stats["gemm_launches"] = stats.get("gemm_launches", 0) + lau.launches
        stats["pairs"] = stats.get("pairs", 0) + P
        stats["padded_rows"] = stats.get("padded_rows", 0) + Pp
        stats["glue_ops"] = stats.get("glue_ops", 0) + GLUE_OPS_FIXED + 2 * k
        stats["moe_host_ms"] = stats.get("moe_host_ms", 0.0) + (time.perf_counter() - t0) * 1e3
        stats.setdefault("calls_per_layer", []).append(len(segs))
        stats["direct_kernel"] = kern is not None
        stats["streams_used"] = max(1, len(lau.streams))
    return out


# ------------------------------------------------------------------------------------------------ the scope
def _install(model, streams: int, stats: dict, pool) -> List:
    saved = []
    for layer in model.layers:
        moe = layer.ffn
        had = "forward" in moe.__dict__
        saved.append((moe, had, moe.__dict__.get("forward")))

        def fwd(x, image_mask=None, _moe=moe):
            return grouped_moe_forward(_moe, x, image_mask, streams=streams, stats=stats, pool=pool)

        fwd.__m4p2_grouped__ = True
        moe.forward = fwd
    return saved


def _uninstall(saved) -> None:
    for moe, had, prev in saved:
        if had:
            moe.forward = prev
        else:
            moe.__dict__.pop("forward", None)


def grouped_installed(model) -> bool:
    return any(getattr(l.ffn.__dict__.get("forward"), "__m4p2_grouped__", False) for l in model.layers)


@contextlib.contextmanager
def prefill_scope(model, mode: Optional[str] = None, fills: Optional[str] = None, streams: Optional[int] = None):
    """Around ONE prefill forward: fills (default off) and, for mode "grouped", the batched routed experts on every
    `model.layers[i].ffn`. Records `last_prefill()`."""
    m = prefill_mode(mode)
    f_on = prefill_fills(fills)
    n = prefill_streams(streams) if m == "grouped" else 1
    pool = getattr(getattr(model, "runtime_load", None), "pool", None)
    stats: Dict = {"mode": m, "fills": "on" if f_on else "off", "streams": n}
    saved = _install(model, n, stats, pool) if m == "grouped" else []
    t0 = time.perf_counter()
    try:
        with contextlib.nullcontext() if f_on else fills_scope(False):
            yield stats
    finally:
        _uninstall(saved)
        stats["host_ms"] = (time.perf_counter() - t0) * 1e3
        if pool is not None and m == "grouped":
            eb = int(getattr(pool, "expert_bytes", 0) or 0)
            sfb = int(getattr(pool, "sf_group_bytes", 0) or 0)
            beside = bool(getattr(pool, "sf_beside", False))
            calls, cold = stats.get("expert_calls", 0), stats.get("cold_calls", 0)
            stats["c2c_bytes"] = cold * eb + ((calls - cold) * sfb if beside else 0)
        _LAST.clear()
        _LAST.update(stats)


def describe() -> dict:
    """The switches in force (for receipts)."""
    return {"mode": prefill_mode(), "fills": "on" if prefill_fills() else "off", "streams": prefill_streams()}
