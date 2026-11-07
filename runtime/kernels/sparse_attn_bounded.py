"""T9b-3a Task A: the reference `sparse_attn` with its 64-index block loop bounded by a per-row DEVICE block count
(`--attn-impl bounded`). reference/deepseek-inference/kernel.py stays untouched; this is a copy of
`sparse_attn_kernel` (kernel.py:310-389) and its wrapper `sparse_attn` (kernel.py:392-403) compiled by the same
TileLang (0.1.12 in the library image) with the same pass configs (kernel.py:9-12). The ONLY changes:

  * one extra argument `nblk: int32 [b, m]` (per (batch, row): how many 64-wide index blocks hold a valid index);
  * the block loop `for t in T.Pipelined(num_blocks, ...)` (kernel.py:358, num_blocks = cdiv(topk, 64) at 325) runs
    `min(nblk[b, m], num_blocks)` iterations instead -- three spellings of that bound (VARIANTS), because which one
    TileLang 0.1.12 compiles is a box question: "extent" (the Pipelined loop's extent read from the tensor, the
    idiom of TileLang's varlen flash-attention examples), "serial" (the same bound on a T.serial loop: no software
    pipelining, data movement only), "guard" (the fixed Pipelined loop with `if t < nblk` around its body).
    Everything inside an iteration, the init (350-356) and the epilogue (382-387) is kernel.py's text verbatim.

Why that is bit-identical (verified in the kernel text, kernel.py:358-387) -- a block whose 64 indices are all -1:
  360  idxs = -1 everywhere;  362  kv_shared = 0;  364  acc_s initialised to -inf;  365  acc_s += q . 0 -> -inf
       (q is finite; a non-finite q would make the reference's padded blocks NaN and this path not -- not a case
       the model produces);  367  -inf * scale = -inf;  368-369  scores_max = max(prev, -inf, ...) = prev
       EXACTLY (reduce_max with clear=False);  371  scores_scale = exp(prev - prev) = exp(+0) = 1.0;
  373  acc_s = exp(-inf - max) = +0 (max >= -1e30 is finite);  374  scores_sum = +0;  376  sum_exp = sum_exp * 1 + 0
       = sum_exp;  377  P = bf16(+0);  379  acc_o *= 1;  380  acc_o += 0 . kv_shared(=0) -> acc_o (acc_o is never
       -0: it starts +0 and a round-to-nearest sum is -0 only if every addend is).
  So a fully -1 block changes no value IF exp(+0) == 1.0 and exp(-inf) == +0 in the compiled kernel (true for
  expf and for the ex2.approx path TileLang's fast-math T.exp lowers to) -- which is exactly what the reference's
  own static step already relies on (static_step.py docstring #9: the static step passes 640 indices where the
  eager reference passes 128 + min(512, T/r), and T8's L0 gate measured static-eager == reference-eager). The
  self-check below re-measures it on the device at build time: bounded == reference at the full padded width AND
  == reference at the narrow (eager) width, `torch.equal`, on the model's shapes. A block that is only partly -1
  (the last valid block) runs in both, unchanged. Blocks are only skipped at the TAIL: a fully -1 leading window
  block (pos < 63) still runs, as in the reference.

The per-row count is computed on the device where the indices are built (runtime/static_step.py `_compress_kv`):
the compressed part is `sort(topk(score)).where(< compress_lens, +offset, -1)` (model.py:579-581; static #9), so
its valid entries are a PREFIX of length <= min(k, compress_lens), and the window part (`win` = 128 entries) always
precedes it. Hence `nblk = ceil((win + min(k, compress_lens)) / 64)` (`nblk_static`) is >= the true last-valid block
(exact whenever no candidate mask removes reachable positions, which at max_pos 4095 it never does) and equals the
eager reference's own block count ceil((128 + min(512, T/r)) / 64) (`ref_width`). Long context (T/r >= 512) gives
nblk = 10 = the full width: correct, no saving.

Mac: no tilelang; the pure functions run, the kernel is built lazily on first use (`HAVE_TILELANG`).
"""
# No `from __future__ import annotations` here: the TileLang prim_func signatures below must stay real expressions,
# exactly as in kernel.py.
from typing import Callable, Dict, List, Optional, Sequence

import torch

BLOCK = 64                               # kernel.py:324 (`block`): indices per block
VARIANTS = ("extent", "serial", "guard")  # build-time order of preference (first that compiles and self-checks)
ATTN_IMPLS = ("ref", "bounded")          # --attn-impl
KERNEL_NAME = "sparse_attn_bounded_kernel_"  # prim_func name prefix (+ "<variant>_"); the reference's: sparse_attn_kernel_

try:  # GPU image only
    import tilelang
    import tilelang.language as T

    HAVE_TILELANG = True
except Exception:  # pragma: no cover - depends on the environment
    tilelang = T = None
    HAVE_TILELANG = False

_STATE: Dict[str, Optional[str]] = {"variant": None}


# ------------------------------------------------------------------------------------------------ pure index math
def blocks_for_width(width: int, block: int = BLOCK) -> int:
    """kernel.py:325: cdiv(topk, block)."""
    return -(-int(width) // block)


def ref_width(pos: int, ratio: int, index_topk: int = 512, win: int = 128) -> int:
    """The index width the eager reference passes at decode position `pos` (model.py:579 `min(index_topk, end_pos //
    ratio)`, end_pos = pos + 1, concatenated after the `win` window slots at 778-779); ratio 0 -> the window only."""
    if not ratio:
        return int(win)
    return int(win) + min(int(index_topk), (int(pos) + 1) // int(ratio))


def valid_blocks(pos: int, ratio: int, k: int, win: int = 128, block: int = BLOCK) -> int:
    """Host mirror of `nblk_static`: ceil((win + min(k, (pos + 1) // ratio)) / block); ratio 0 -> ceil(win / block)."""
    c = 0 if not ratio else min(int(k), (int(pos) + 1) // int(ratio))
    return blocks_for_width(int(win) + c, block)


def nblk_static(clen: torch.Tensor, k: int, win: int, block: int = BLOCK) -> torch.Tensor:
    """Device per-row block count from the device compress_lens (`clen`, int64 [rows] = (pos+1)//r): int32 [1, rows]
    = ceil((win + min(k, clen)) / block). Four elementwise launches, no host sync."""
    n = torch.clamp(clen, max=int(k)) + (int(win) + block - 1)
    return torch.div(n, block, rounding_mode="floor").int().view(1, -1)


def nblk_from_idx(topk_idxs: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Reference for the tests: per (b, m) the number of blocks up to and including the last valid (!= -1) index:
    int32 [b, m] (0 for a row with no valid index)."""
    n = topk_idxs.size(-1)
    pos1 = torch.arange(1, n + 1, device=topk_idxs.device, dtype=torch.long)
    last = torch.where(topk_idxs != -1, pos1, torch.zeros((), dtype=torch.long, device=topk_idxs.device)).amax(-1)
    return torch.div(last + block - 1, block, rounding_mode="floor").int()


# ------------------------------------------------------------------------------------------------ the kernel
if HAVE_TILELANG:
    tilelang.set_log_level("WARNING")
    _FP32, _BF16, _INT32 = "float32", "bfloat16", "int32"
    _pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,   # kernel.py:9-12
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    @tilelang.jit(pass_configs=_pass_configs)
    def sparse_attn_bounded_kernel(h: int, d: int, scale=None, variant: str = "extent"):
        """kernel.py:310-389 with the block loop bounded by nblk[by, bx] (module docstring)."""
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        topk = T.symbolic("topk")
        if scale is None:
            scale = (1.0 / d) ** 0.5

        num_stages = 2
        threads = 256
        block = BLOCK
        num_blocks = tilelang.cdiv(topk, block)
        BF16, FP32, INT32 = _BF16, _FP32, _INT32

        @T.prim_func
        def sparse_attn_bounded_kernel_extent_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
            nblk: T.Tensor[(b, m), INT32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h, d), BF16)
                acc_s_cast = T.alloc_shared((h, block), BF16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h, block), FP32)
                acc_o = T.alloc_fragment((h, d), FP32)
                scores_max = T.alloc_fragment(h, FP32)
                scores_max_prev = T.alloc_fragment(h, FP32)
                scores_scale = T.alloc_fragment(h, FP32)
                scores_sum = T.alloc_fragment(h, FP32)
                sum_exp = T.alloc_fragment(h, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -1e30)
                T.copy(q[by, bx, :, :], q_shared)

                n_live = T.min(nblk[by, bx], num_blocks)
                for t in T.Pipelined(n_live, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])

        @T.prim_func
        def sparse_attn_bounded_kernel_serial_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
            nblk: T.Tensor[(b, m), INT32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h, d), BF16)
                acc_s_cast = T.alloc_shared((h, block), BF16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h, block), FP32)
                acc_o = T.alloc_fragment((h, d), FP32)
                scores_max = T.alloc_fragment(h, FP32)
                scores_max_prev = T.alloc_fragment(h, FP32)
                scores_scale = T.alloc_fragment(h, FP32)
                scores_sum = T.alloc_fragment(h, FP32)
                sum_exp = T.alloc_fragment(h, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -1e30)
                T.copy(q[by, bx, :, :], q_shared)

                n_live = T.min(nblk[by, bx], num_blocks)
                for t in T.serial(n_live):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])

        @T.prim_func
        def sparse_attn_bounded_kernel_guard_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
            nblk: T.Tensor[(b, m), INT32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h, d), BF16)
                acc_s_cast = T.alloc_shared((h, block), BF16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h, block), FP32)
                acc_o = T.alloc_fragment((h, d), FP32)
                scores_max = T.alloc_fragment(h, FP32)
                scores_max_prev = T.alloc_fragment(h, FP32)
                scores_scale = T.alloc_fragment(h, FP32)
                scores_sum = T.alloc_fragment(h, FP32)
                sum_exp = T.alloc_fragment(h, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -1e30)
                T.copy(q[by, bx, :, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    if t < nblk[by, bx]:
                        for i in T.Parallel(block):
                            idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                        for i, j in T.Parallel(block, d):
                            kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                        for i, j in T.Parallel(h, block):
                            acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                        T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(h, block):
                            acc_s[i, j] *= scale
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(h):
                            scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                        for i, j in T.Parallel(h, block):
                            acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(h):
                            sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(h, d):
                            acc_o[i, j] *= scores_scale[i]
                        T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])

        if variant == "extent":
            return sparse_attn_bounded_kernel_extent_
        if variant == "serial":
            return sparse_attn_bounded_kernel_serial_
        if variant == "guard":
            return sparse_attn_bounded_kernel_guard_
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
else:  # pragma: no cover - Mac
    sparse_attn_bounded_kernel = None


def selected_variant() -> Optional[str]:
    """The variant the last passing self-check picked (None before any self-check)."""
    return _STATE["variant"]


def select_variant(variant: Optional[str]) -> None:
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    _STATE["variant"] = variant


def sparse_attn_bounded(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                        softmax_scale: float, nblk: torch.Tensor, variant: Optional[str] = None) -> torch.Tensor:
    """kernel.py:392-403 (the head padding to 16 and the narrow back included) with the extra per-row block count
    `nblk` int32 [b, m] on q's device. `variant` None -> the self-check's pick, else VARIANTS[0]."""
    if sparse_attn_bounded_kernel is None:
        raise RuntimeError("sparse_attn_bounded needs tilelang (GPU image)")
    b, s, h, d = q.size()
    if tuple(nblk.shape) != (b, s) or nblk.dtype != torch.int32:
        raise ValueError(f"nblk must be int32 [{b}, {s}], got {nblk.dtype} {tuple(nblk.shape)}")
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
    o = torch.empty_like(q)
    kernel = sparse_attn_bounded_kernel(q.size(2), d, softmax_scale, variant or _STATE["variant"] or VARIANTS[0])
    kernel(q, kv, o, attn_sink, topk_idxs, nblk.contiguous())
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o


# ------------------------------------------------------------------------------------------------ self-check
def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16),
                                                                     b.contiguous().view(torch.int16))


def _cpu_default(fn):
    """Build synthetic inputs under a CPU default device (at runtime the Generator made the default device cuda, and
    a CPU torch.Generator then cannot drive a bare constructor); callers move the results."""
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        with torch.device("cpu"):
            return fn(*a, **k)

    return w


@_cpu_default
def check_cases(win: int, k: int, kv_len: int, seed: int = 0) -> List[dict]:
    """Index lists in the static step's layout: `win` window slots (oldest first, leading -1 at small pos, as
    static_step.window_topk_idxs_dev) then k compressed slots = c sorted valid positions + offset win, -1 tail.
    One dict per row: {"pos", "c", "idx" (int32 [win + k] cpu)}."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = kv_len - win
    cs = sorted({0, 1, BLOCK - 1, BLOCK, BLOCK + 1, 150, 300, k // 2, k - 1, k})
    rows = []
    for pos in (5, 63, 64, 200):
        ar = torch.arange(win)
        widx = torch.remainder(pos % win + 1 + ar, win)
        widx = torch.where(widx > pos, -1, widx)
        for c in cs:
            c = min(c, k, W)
            comp = torch.full((k,), -1, dtype=torch.long)
            if c:
                comp[:c] = torch.randperm(W, generator=g)[:c].sort().values + win
            rows.append({"pos": pos, "c": c, "idx": torch.cat([widx, comp]).int()})
    return rows


@_cpu_default
def _check_inputs(heads: int, head_dim: int, kv_len: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    q_all = (torch.randn(1, 4, heads, head_dim, generator=g) * 0.5).to(torch.bfloat16)
    kv = torch.randn(1, kv_len, head_dim, generator=g).to(torch.bfloat16)
    sink = torch.randn(heads, generator=g).float()
    return q_all, kv, sink


def self_check(orig: Callable, device, heads: int, head_dim: int, softmax_scale: float, win: int, k: int,
               kv_len: int, attn_sink: Optional[torch.Tensor] = None, variants: Sequence[str] = VARIANTS,
               impl: Optional[Callable] = None, seed: int = 0, select: bool = True) -> dict:
    """Build-time device check (eager, syncs; before any capture). For each variant in order: bounded(full width,
    nblk) must equal `orig` (the reference sparse_attn) at the full padded width AND at the narrow width the eager
    reference passes (win + c), `torch.equal` on the bf16 bits, for every case of `check_cases` one row at a time,
    and for a multi-row call (rows with different nblk in one launch == the same rows alone). The first variant
    that passes is selected. -> {"variant", "results": {variant: "equal" | "mismatch ..." | "error ..."}, ...}."""
    impl = impl or (lambda *a, **kw: sparse_attn_bounded(*a, **kw))
    dev = torch.device(device)
    q_all, kv, sink = _check_inputs(heads, head_dim, kv_len, seed)
    q_all, kv = q_all.to(dev), kv.to(dev)
    sink = attn_sink.detach().float().to(dev) if attn_sink is not None else sink.to(dev)
    cases = check_cases(win, k, kv_len, seed)
    results: Dict[str, str] = {}
    chosen = None
    for v in variants:
        try:
            bad = None
            for ci, cs in enumerate(cases):
                q = q_all[:, ci % 4: ci % 4 + 1].contiguous()
                idx = cs["idx"].to(dev).view(1, 1, -1)
                nb = torch.tensor([[blocks_for_width(win + cs["c"])]], dtype=torch.int32, device=dev)
                got = impl(q, kv, sink, idx, softmax_scale, nb, variant=v)
                want_full = orig(q, kv, sink, idx, softmax_scale)
                want_narrow = orig(q, kv, sink, idx[..., : win + cs["c"]].contiguous(), softmax_scale)
                if not _bits_equal(got, want_full):
                    bad = f"mismatch vs reference at full width {idx.size(-1)}: pos {cs['pos']} c {cs['c']}"
                    break
                if not _bits_equal(got, want_narrow):
                    bad = f"mismatch vs reference at narrow width {win + cs['c']}: pos {cs['pos']} c {cs['c']}"
                    break
            if bad is None:  # several rows, different counts, one launch
                pick = [cases[i] for i in (1, len(cases) // 3, len(cases) // 2, len(cases) - 1)]
                idx = torch.stack([c["idx"] for c in pick]).to(dev).view(1, len(pick), -1)
                nb = torch.tensor([[blocks_for_width(win + c["c"]) for c in pick]], dtype=torch.int32, device=dev)
                got = impl(q_all, kv, sink, idx, softmax_scale, nb, variant=v)
                for r in range(len(pick)):
                    one = impl(q_all[:, r:r + 1].contiguous(), kv, sink, idx[:, r:r + 1].contiguous(),
                               softmax_scale, nb[:, r:r + 1].contiguous(), variant=v)
                    if not _bits_equal(got[:, r:r + 1], one):
                        bad = f"multi-row: row {r} at {len(pick)} rows != the same row alone"
                        break
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            results[v] = "equal" if bad is None else bad
        except Exception as e:  # compile or launch failure: record, try the next spelling
            results[v] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        if results[v] == "equal":
            chosen = v
            break
    if select:
        select_variant(chosen)
    return {"variant": chosen, "results": results, "cases": len(cases), "heads": heads, "head_dim": head_dim,
            "win": win, "k": k, "kv_len": kv_len, "tilelang": getattr(tilelang, "__version__", None)}
