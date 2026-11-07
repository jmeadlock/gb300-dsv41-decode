"""Monkeypatches on the unmodified reference model.py, each gated by an explicit flag. None active by default.

kv_numerics_pg61(enable):
    Replace the compressed-KV latent's FP4 QDQ (model.py:761, `fp4_act_quant(latent, 16, True, scale_dtype=e4m3)`)
    with the MXFP8 QDQ Pg61/vLLM `fp8_ds_mla` applies instead (D1 memo §Compressed KV: e4m3 values, block 32, UE8M0 pow2
    scale, no FP4 step). Window KV (model.py:708) already is FP8/32 UE8M0 in both; index K/Q (546/553) are left as the
    reference — at <= 310 tokens the indexer selects every position in both engines (D1 §2), so they are inert for the
    W0 fixture. Purpose: a discriminating experiment, not a product mode — if first-light drift vs Pg61 collapses under
    this flag, the gap is "model definition vs Pg61's KV format", not a runtime bug.

static_moe(enable):
    T8 design table 2.1 row #14: `MoE.forward` (model.py:891-906; host syncs at 896 `bincount().tolist()` and 901
    `torch.where`) -> the graph-capturable static MoE of runtime/static_step.py (`StaticMoE`: device gate, ids sorted
    ascending = the reference's accumulation order, pointer-table gather into static staging, fixed-k reference
    `Expert` evaluations, fp32 sum in slot order, + shared expert; bench/moe_static_spike.py measured it bit-identical).
    The bound object is whatever `StaticStep(..., moe_impl=...)` built: `StaticMoE` for "ref", runtime/dg_moe.py
    `DGMaskedMoE` for "dg-masked" (T9a-1); this patch only dispatches, so the impl is chosen by the StaticStep.
    Takes effect only for a MoE instance a `StaticStep` has bound (`_static_moe`), only while that StaticStep is
    running a step (`active`), and only for one row with no image mask; every other call -- the prefill, the eager
    reference decode, multi-row input -- runs the original forward. `StaticStep.__init__` enables it.

fp8_linear_gemv(enable):
    T9b-1: `linear()`'s fp8 branch (model.py:196-205: `act_quant(x, 32, "ue8m0", e8m0)` then `fp8_gemm(..., block_size
    =32)`) -> the same `act_quant` call, then runtime/gemv_mxfp8.py `linear_fp8` (Triton MXFP8 GEMV) instead of
    `fp8_gemm`. Routed only when ALL hold: inside `fp8_gemv_scope()` (a `StaticStep(fp8_impl="gemv")` opens it while
    it runs a step -- eagerly, in warm-up, and under capture), the weight is float8_e4m3fn with a float8_e8m0fnu
    `weight.scale` of the reference's [ceil(N/32), K/32] shape, the reference globals are the MXFP8 ones
    (fp8_block_size 32, scale_fmt set, scale_dtype e8m0), K % 32 == 0, the activation has 1..8 rows (rows =
    x.numel() // K: the reference's M, kernel.py:291 -- NOT x.size(0), which is the batch and is 1 for a whole
    prefill too), the default dtype is bf16 (the reference output dtype, kernel.py:299), and no bias (model.py:184).
    Everything else -- prefill, the eager reference decode, fp4/bf16/fp32 weights (routed experts, wo_a, compressor,
    indexer wk/weights_proj, gate, head, mHC) -- calls the original `linear`. Installing it is idempotent and inert
    outside the scope; `fp8_gemv_stats()` counts routed and fallen-through calls (host side: eager calls and
    captures, not graph replays).

sparse_attn_bounded(enable):
    T9b-3a: `sparse_attn` (kernel.py:392-403, the TileLang `sparse_attn_kernel` at 310-389, imported into model.py and
    called at model.py:781) -> runtime/kernels/sparse_attn_bounded.py, the same kernel text with its 64-index block
    loop (kernel.py:358, bound `num_blocks = cdiv(topk, 64)` at 325) bounded by a per-row device block count. Routed
    only inside `attn_bound_scope(nblk)` -- which `StaticStep(attn_impl="bounded")` opens around its compressed
    layers' `sparse_attn` call with the count it computed where the indices are built -- and only when q is
    [b, m, h, d] bf16, the indices int32 [b, m, topk] and nblk int32 [b, m] on q's device. Everything else (prefill,
    the eager reference, ratio-0 layers, `attn_impl="ref"`) calls the original. Installing is idempotent and inert
    outside the scope; `attn_bound_stats()` counts routed / fallen-through calls (host side).
"""
import contextlib
import sys


def kv_numerics_pg61(enable: bool = True) -> None:
    ref = sys.modules.get("model") or __import__("model")
    kernel = sys.modules.get("kernel") or __import__("kernel")
    if not hasattr(ref, "_orig_fp4_act_quant"):
        ref._orig_fp4_act_quant = ref.fp4_act_quant

    def patched(x, block_size, inplace=False, scale_dtype=None):
        import torch
        if inplace and block_size == 16 and scale_dtype is torch.float8_e4m3fn:
            # the compressed-KV latent call site (model.py:761): MXFP8 in place instead of FP4
            kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True)
            return None
        return ref._orig_fp4_act_quant(x, block_size, inplace, scale_dtype) if scale_dtype is not None \
            else ref._orig_fp4_act_quant(x, block_size, inplace)

    ref.fp4_act_quant = patched if enable else ref._orig_fp4_act_quant


def static_moe(enable: bool = True) -> None:
    ref = sys.modules.get("model") or __import__("model")
    if not hasattr(ref.MoE, "_orig_forward"):
        ref.MoE._orig_forward = ref.MoE.forward
    orig = ref.MoE._orig_forward

    def forward(self, x, image_mask=None):
        st = self.__dict__.get("_static_moe")
        if st is not None and st.active and image_mask is None and x.numel() == self.dim:
            return st(x)  # design #14 (model.py:891-906)
        return orig(self, x, image_mask)

    forward.__t8_static_moe__ = True
    ref.MoE.forward = forward if enable else orig


def static_moe_enabled() -> bool:
    ref = sys.modules.get("model")
    return ref is not None and getattr(ref.MoE.forward, "__t8_static_moe__", False)


# ------------------------------------------------------------------------------------------------ T9b-1 fp8 GEMV
GEMV_MAX_ROWS = 8
_GEMV_SCOPE = {"depth": 0}
_GEMV_STATS = {"routed": {}, "fallthrough": {}}


@contextlib.contextmanager
def fp8_gemv_scope(on: bool = True):
    """While open (and `fp8_linear_gemv` is installed), eligible fp8 Linear calls run the GEMV. Nests."""
    if not on:
        yield
        return
    _GEMV_SCOPE["depth"] += 1
    try:
        yield
    finally:
        _GEMV_SCOPE["depth"] -= 1


def fp8_gemv_active() -> bool:
    return _GEMV_SCOPE["depth"] > 0


def fp8_gemv_stats(reset: bool = False) -> dict:
    """{"routed": {(N, K, rows): calls}, "fallthrough": {(reason, N, K): calls}} since the last reset (host side)."""
    out = {k: dict(v) for k, v in _GEMV_STATS.items()}
    if reset:
        for v in _GEMV_STATS.values():
            v.clear()
    return out


def fp8_gemv_reason(ref, x, weight, bias=None):
    """None if `linear(x, weight, bias)` may take the GEMV (patches docstring), else why not. Metadata only."""
    import torch

    if weight.dtype != torch.float8_e4m3fn:
        return "dtype"
    sc = getattr(weight, "scale", None)
    if sc is None or sc.dtype != torch.float8_e8m0fnu:
        return "scale_dtype"
    if bias is not None:
        return "bias"
    if (ref.fp8_block_size, ref.scale_dtype) != (32, torch.float8_e8m0fnu) or ref.scale_fmt is None:
        return "ref_globals"
    N, K = weight.shape
    if x.size(-1) != K or K % 32:
        return "k"
    if tuple(sc.shape) != ((N + 31) // 32, K // 32) or not (weight.is_contiguous() and sc.is_contiguous()):
        return "layout"
    rows = x.numel() // K
    if not 1 <= rows <= GEMV_MAX_ROWS:
        return "rows"
    if torch.get_default_dtype() != torch.bfloat16:
        return "default_dtype"
    return None


def fp8_linear_gemv(enable: bool = True) -> None:
    """Install (or restore) the model.linear wrapper (patches docstring). model.py looks `linear` up as a module
    global at every call (Linear/ColumnParallelLinear/RowParallelLinear.forward, Gate.forward), so the wrapper is
    what they reach."""
    ref = sys.modules.get("model") or __import__("model")
    import torch

    fp8 = torch.float8_e4m3fn
    if not hasattr(ref, "_orig_linear"):
        ref._orig_linear = ref.linear
    orig = ref._orig_linear

    def linear(x, weight, bias=None):
        if _GEMV_SCOPE["depth"] and weight.dtype == fp8:
            why = fp8_gemv_reason(ref, x, weight, bias)
            if why is None:
                from runtime.gemv_mxfp8 import linear_fp8

                N, K = weight.shape
                key = (N, K, x.numel() // K)
                _GEMV_STATS["routed"][key] = _GEMV_STATS["routed"].get(key, 0) + 1
                xq, s = ref.act_quant(x, ref.fp8_block_size, ref.scale_fmt, ref.scale_dtype)  # model.py:197
                return linear_fp8(xq, s, weight, weight.scale)                                  # model.py:198-205
            key = (why, weight.size(0), weight.size(1))
            _GEMV_STATS["fallthrough"][key] = _GEMV_STATS["fallthrough"].get(key, 0) + 1
        return orig(x, weight, bias)

    linear.__t9b_fp8_gemv__ = True
    ref.linear = linear if enable else orig


def fp8_linear_gemv_enabled() -> bool:
    ref = sys.modules.get("model")
    return ref is not None and getattr(ref.linear, "__t9b_fp8_gemv__", False)


# ------------------------------------------------------------------------------------------------ T9b-2 helpers
def count_gemv_routed(weight, x) -> None:
    """Record one routed GEMV call (N, K, rows) the way the `linear` wrapper does (fused-glue call sites that run the
    GEMV themselves: runtime/fused_glue.py q_proj, fp8_linear_prequant)."""
    N, K = weight.shape
    key = (N, K, x.numel() // K)
    _GEMV_STATS["routed"][key] = _GEMV_STATS["routed"].get(key, 0) + 1


def fp8_linear_prequant(xq, s, weight):
    """`linear(x, weight)`'s fp8 branch (model.py:196-205) for an activation that is ALREADY quantized (xq fp8 [..., K],
    s e8m0 [..., K/32] = what `act_quant(x, 32, "ue8m0")` returns): the same routing decision the installed `linear`
    would make for x (T9b-1 GEMV inside fp8_gemv_scope when eligible, else the reference `fp8_gemm`), without the
    act_quant. Used by T9b-2's fused SwiGLU, whose kernel writes the w2 input's act_quant itself."""
    ref = sys.modules.get("model") or __import__("model")
    if _GEMV_SCOPE["depth"] and fp8_linear_gemv_enabled():
        why = fp8_gemv_reason(ref, xq, weight)
        if why is None:
            from runtime.gemv_mxfp8 import linear_fp8

            count_gemv_routed(weight, xq)
            return linear_fp8(xq, s, weight, weight.scale)
        key = (why, weight.size(0), weight.size(1))
        _GEMV_STATS["fallthrough"][key] = _GEMV_STATS["fallthrough"].get(key, 0) + 1
    return ref.fp8_gemm(xq, s, weight, weight.scale, ref.scale_dtype, block_size=ref.fp8_block_size)


# ------------------------------------------------------------------------------------------------ T9b-3a bounded attn
_ATTN_SCOPE = {"nblk": None}
_ATTN_STATS = {"routed": 0, "fallthrough": {}}


@contextlib.contextmanager
def attn_bound_scope(nblk):
    """While open (and `sparse_attn_bounded` is installed), an eligible `sparse_attn` call runs the bounded kernel
    with this per-row block count (int32 [b, m] device tensor). `nblk=None` is a no-op scope. Nests (inner wins)."""
    prev = _ATTN_SCOPE["nblk"]
    _ATTN_SCOPE["nblk"] = nblk if nblk is not None else prev
    try:
        yield
    finally:
        _ATTN_SCOPE["nblk"] = prev


def attn_bound_stats(reset: bool = False) -> dict:
    """{"routed": calls, "fallthrough": {reason: calls}} since the last reset (host side)."""
    out = {"routed": _ATTN_STATS["routed"], "fallthrough": dict(_ATTN_STATS["fallthrough"])}
    if reset:
        _ATTN_STATS["routed"] = 0
        _ATTN_STATS["fallthrough"].clear()
    return out


def attn_bound_reason(q, topk_idxs, nblk):
    """None if `sparse_attn(q, kv, sink, topk_idxs, scale)` may take the bounded kernel with `nblk`, else why not."""
    import torch

    if q.dim() != 4 or q.dtype != torch.bfloat16:
        return "q"
    if topk_idxs.dim() != 3 or topk_idxs.dtype != torch.int32 or tuple(topk_idxs.shape[:2]) != tuple(q.shape[:2]):
        return "idx"
    if nblk.dtype != torch.int32 or tuple(nblk.shape) != tuple(q.shape[:2]) or nblk.device != q.device:
        return "nblk"
    return None


def sparse_attn_bounded(enable: bool = True) -> None:
    """Install (or restore) the model.sparse_attn wrapper (patches docstring). model.py looks `sparse_attn` up as a
    module global at the call (model.py:781) and StaticStep calls `R.sparse_attn`, so the wrapper is what both
    reach. The original is whatever `model.sparse_attn` is when this is first installed over it (the kernel.py
    function, or a test's emulation)."""
    ref = sys.modules.get("model") or __import__("model")
    cur = ref.sparse_attn
    if not getattr(cur, "__t9b3a_bounded__", False):
        ref._orig_sparse_attn = cur
    orig = ref._orig_sparse_attn

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        nb = _ATTN_SCOPE["nblk"]
        if nb is not None:
            why = attn_bound_reason(q, topk_idxs, nb)
            if why is None:
                from runtime.kernels import sparse_attn_bounded as SAB

                _ATTN_STATS["routed"] += 1
                return SAB.sparse_attn_bounded(q, kv, attn_sink, topk_idxs, softmax_scale, nb)
            _ATTN_STATS["fallthrough"][why] = _ATTN_STATS["fallthrough"].get(why, 0) + 1
        return orig(q, kv, attn_sink, topk_idxs, softmax_scale)

    sparse_attn.__t9b3a_bounded__ = True
    ref.sparse_attn = sparse_attn if enable else orig


def sparse_attn_bounded_enabled() -> bool:
    ref = sys.modules.get("model")
    return ref is not None and getattr(ref.sparse_attn, "__t9b3a_bounded__", False)


def original_sparse_attn():
    """The reference sparse_attn underneath the wrapper (or the current one when the wrapper is not installed)."""
    ref = sys.modules.get("model") or __import__("model")
    cur = ref.sparse_attn
    return ref._orig_sparse_attn if getattr(cur, "__t9b3a_bounded__", False) else cur
