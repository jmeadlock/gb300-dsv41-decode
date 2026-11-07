"""T9b-1: a bandwidth-bound MXFP8 GEMV for the model's fp8 `Linear` layers at M <= 8 rows (`--fp8-impl gemv`).

    out = mxfp8_gemv(xq, xs, w, s, out)    # xq fp8 [M,K], xs e8m0 [M,K/32], w fp8 [N,K], s e8m0 [ceil(N/32),K/32]
    y = linear_fp8(xq, xs, w, s)           # same, allocating out [*xq.shape[:-1], N] bf16 (the reference's c)

What it replaces: the reference `fp8_gemm` (kernel.py:207-307) as called by `linear()` (model.py:196-205) for every
`Linear` whose weight is fp8 with 32x32 E8M0 block scales (attention wq_a/wq_b/wkv/wo_b, indexer wq_b, shared experts
w1/w3/w2, Engram wkv: 290 calls per 1-row step). That kernel is a 32x128 tile GEMM; at M = 1 it reads weights at
~0.3 TB/s (docs/T9A-DEEPGEMM-ABI-20261106.md addendum). Here one program owns BLOCK_N weight rows and all M <= 8
activation rows and streams its rows once, K-major, KU 32-byte K-blocks per row per iteration (16-B vector loads).

Arithmetic, and how it relates to the reference kernel (kernel.py:252-272):
  reference  per 32-wide K block k (in K order): C = fp8 MMA of the [32x32]x[32x128] tile in fp32 (tensor-core
             internal order), then `acc += C * sa[m,k] * sb[n//32,k]` in fp32, acc starting at 0; bf16 store.
  here       per K block k (in K order): p = fp32 sum of the 32 exact e4m3 x e4m3 products of row m and row n,
             then `acc += p * 2^(sa + sb - 254)` in fp32, acc starting at 0; bf16 store (round to nearest even).
  * the scale: sa, sb are e8m0 codes, so sa*sb is the power of two 2^(sa+sb-254); it is built from its bits
    (`pair_scale`), exactly, including the fp32 subnormal range (2^-127..2^-149), 0 below, inf above 2^127, NaN for a
    255 code. (C*sa)*sb == C*2^(sa+sb-254) whenever neither side under/overflows (scales here are ~2^-22..2^0).
  * the K-block accumulation is in the reference's order, one fp32 add per block; the weight scale is read at
    s[n // 32, k] directly (no expansion).
  * the only order difference is INSIDE one 32-wide block: the 32 exact products are summed by `tl.sum` here and by
    the tensor core there. Whenever that block sum is exact in fp32 the two agree bit for bit; otherwise they may
    differ in the last fp32 bit, which can move a bf16 output one ulp. That is the whole drift budget (bench:
    `bench/gemv_bench.py`, gate <= 0.5 % of outputs).

Tuning: only (BLOCK_N, KU, num_warps, num_stages) -- none of which changes the per-output K order. A choice is made
once per (N, K, BLOCK_M) by timing the CANDIDATES under a CUDA graph on rotating synthetic weights (so they stream from
HBM, as in the decode step), kept for the life of the process (a StaticStep relaunch reuses it), and persisted to the
JSON file named by $DSV41_GEMV_CACHE when set (so another process reuses it too). Tuning refuses to run while a CUDA
graph is being captured; StaticStep's warm-up steps run it before capture. $DSV41_GEMV_TUNE=0 skips timing and uses
DEFAULT_CONFIG. Whether the intra-block `tl.sum` order depends on the config is measured, not assumed:
tests/gpu/test_gemv_gpu.py asserts every candidate gives bit-identical output.

Graph capture: one Triton launch per call, static shapes, no host sync, no allocation inside `mxfp8_gemv`.
CPU tensors (the Mac tests) run `gemv_torch`, the same algorithm in torch.

Triton is imported lazily-by-guard: the module imports on the Mac (no triton) and everything but the kernel works.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:  # GPU image only; the Mac tests use gemv_torch
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = tl = None
    HAVE_TRITON = False

FP8_IMPLS = ("ref", "gemv")   # --fp8-impl: ref = reference fp8_gemm (bit-identical, T8) / gemv = this module
GEMV_MAX_M = 8                # rows routed to the GEMV (model.linear patch); more rows -> the reference GEMM
BLOCK = 32                    # MXFP8 block: act_quant(x, 32, "ue8m0") rows (model.py:197) and 32x32 weight blocks
E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn
NAN_BITS = 0x7FC00000
INF_BITS = 0x7F800000


# ------------------------------------------------------------------------------------------------ exact scale math
def pair_scale_bits(sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """int32 bits of the fp32 value 2^(sa + sb - 254) for e8m0 codes sa, sb (any int dtype, broadcastable): the exact
    product 2^(sa-127) * 2^(sb-127). e >= -126: exponent field; -149 <= e < -126: the subnormal 2^e (one mantissa
    bit); e < -149: 0 (an fp32 product rounds 2^-150 to even = 0, and anything smaller to 0); e > 127: inf; either
    code 255 (the e8m0 NaN): NaN. The Triton kernel's `_t9b_pair_scale` is this function line for line."""
    sa = sa.to(torch.int32)
    sb = sb.to(torch.int32)
    e = sa + sb - 254
    normal = (e.clamp(-126, 128) + 127) << 23
    sub = torch.ones_like(e) << (e + 149).clamp(0, 22)
    bits = torch.where(e >= -126, normal, torch.where(e >= -149, sub, torch.zeros_like(e)))
    return torch.where((sa == 255) | (sb == 255), torch.full_like(e, NAN_BITS), bits)


def pair_scale(sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """fp32 2^(sa + sb - 254), bit-exact (see pair_scale_bits)."""
    return pair_scale_bits(sa, sb).view(torch.float32)


def weight_scale_rows(n: torch.Tensor) -> torch.Tensor:
    """Row of the [ceil(N/32), K/32] weight-scale matrix that output row n uses: its 32x32 block row, n // 32
    (kernel.py:263 with group_size 32). The kernel indexes s[n // 32, k] directly."""
    return torch.div(n, BLOCK, rounding_mode="floor")


def block_m(m: int) -> int:
    """The kernel's accumulator height: the next power of two >= m (1, 2, 4, 8)."""
    if not 1 <= int(m) <= GEMV_MAX_M:
        raise ValueError(f"mxfp8_gemv: M = {m} rows outside [1, {GEMV_MAX_M}]")
    return 1 << (int(m) - 1).bit_length()


def check_operands(xq, xs, w, s, out=None) -> Tuple[int, int, int]:
    """Shape/dtype/layout contract of mxfp8_gemv -> (M, N, K). All metadata, no device access."""
    if xq.dim() != 2 or w.dim() != 2:
        raise ValueError(f"mxfp8_gemv: xq {tuple(xq.shape)} and w {tuple(w.shape)} must be 2-D")
    M, K = xq.shape
    N = w.size(0)
    block_m(M)
    if xq.dtype != FP8 or w.dtype != FP8:
        raise TypeError(f"mxfp8_gemv: xq {xq.dtype} / w {w.dtype} must be float8_e4m3fn")
    if xs.dtype not in (E8, torch.uint8) or s.dtype not in (E8, torch.uint8):
        raise TypeError(f"mxfp8_gemv: scales {xs.dtype} / {s.dtype} must be float8_e8m0fnu (or its uint8 bytes)")
    if K % BLOCK or w.size(1) != K:
        raise ValueError(f"mxfp8_gemv: K = {K} (w {tuple(w.shape)}) must match and be a multiple of {BLOCK}")
    if tuple(xs.shape) != (M, K // BLOCK):
        raise ValueError(f"mxfp8_gemv: xs {tuple(xs.shape)} != (M, K/32) = {(M, K // BLOCK)}")
    if tuple(s.shape) != ((N + BLOCK - 1) // BLOCK, K // BLOCK):
        raise ValueError(f"mxfp8_gemv: s {tuple(s.shape)} != (ceil(N/32), K/32) = {((N + 31) // 32, K // 32)}")
    for name, t in (("xq", xq), ("xs", xs), ("w", w), ("s", s)):
        if t.stride(-1) != 1 or not t.is_contiguous():
            raise ValueError(f"mxfp8_gemv: {name} must be contiguous")
    if out is not None:
        if tuple(out.shape) != (M, N) or out.dtype != torch.bfloat16 or out.stride(-1) != 1:
            raise ValueError(f"mxfp8_gemv: out {tuple(out.shape)} {out.dtype} must be a row-major bf16 [M, N]")
    devs = {t.device for t in (xq, xs, w, s) + (() if out is None else (out,))}
    if len(devs) != 1:
        raise ValueError(f"mxfp8_gemv: operands on several devices {devs}")
    return M, N, K


# ------------------------------------------------------------------------------------------------ torch algorithm
def block_partials(xq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 [M, N, K/32]: for each row m, weight row n and 32-wide K block k, the sum of the 32 exact products."""
    M, K = xq.shape
    N = w.size(0)
    kb = K // BLOCK
    return (xq.float().view(M, 1, kb, BLOCK) * w.float().view(1, N, kb, BLOCK)).sum(-1)


def gemv_torch(xq, xs, w, s, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The kernel's algorithm in torch (CPU tests; any device): block partials, scaled by the exact pair scale with
    the weight scale read at s[n // 32, k], accumulated in fp32 one K block at a time in K order from 0, bf16."""
    M, N, K = check_operands(xq, xs, w, s, out)
    kb = K // BLOCK
    part = block_partials(xq, w)                                                  # [M, N, KB]
    sa = xs.view(torch.uint8).to(torch.int32).view(M, 1, kb)
    rows = weight_scale_rows(torch.arange(N, device=w.device))
    sb = s.view(torch.uint8).to(torch.int32).index_select(0, rows).view(1, N, kb)
    q = part * pair_scale(sa, sb)
    acc = torch.zeros(M, N, dtype=torch.float32, device=xq.device)
    for k in range(kb):
        acc = acc + q[:, :, k]
    res = acc.to(torch.bfloat16)
    if out is None:
        return res
    return out.copy_(res)


# ------------------------------------------------------------------------------------------------ configs / tuning
@dataclass(frozen=True)
class GemvConfig:
    block_n: int     # weight rows per program
    ku: int          # 32-wide K blocks per row per loop iteration (KU * 32 contiguous bytes per row)
    num_warps: int
    num_stages: int

    def as_dict(self) -> dict:
        return asdict(self)


# none of these changes the per-output arithmetic: the K-block order is fixed and the intra-block sum is a 32-wide
# tl.sum in every one of them (config invariance is asserted on the box, tests/gpu/test_gemv_gpu.py)
CANDIDATES: Tuple[GemvConfig, ...] = (
    GemvConfig(8, 8, 2, 1), GemvConfig(16, 4, 4, 1), GemvConfig(16, 8, 4, 1), GemvConfig(16, 8, 4, 2),
    GemvConfig(32, 4, 4, 1), GemvConfig(32, 4, 4, 2), GemvConfig(32, 8, 8, 1), GemvConfig(64, 4, 8, 1),
    GemvConfig(64, 8, 8, 2),
)
DEFAULT_CONFIG = GemvConfig(16, 8, 4, 1)
TUNE_STREAM_BYTES = 512 << 20   # rotating synthetic weight copies per timed shape (beyond the L2, ~126 MB here)
TUNE_MAX_COPIES = 256
TUNE_CALLS = 64

_CONFIGS: Dict[Tuple[int, int, int], GemvConfig] = {}
_TUNE_LOG: Dict[Tuple[int, int, int], dict] = {}
_CACHE_LOADED: Optional[str] = None


def config_key(N: int, K: int, M: int) -> Tuple[int, int, int]:
    return int(N), int(K), block_m(M)


def _key_str(key) -> str:
    return f"{key[0]}x{key[1]}/m{key[2]}"


def cache_path() -> Optional[str]:
    return os.environ.get("DSV41_GEMV_CACHE") or None


def tune_enabled() -> bool:
    return os.environ.get("DSV41_GEMV_TUNE", "1") not in ("0", "false", "off")


def load_cache(path: Optional[str] = None) -> int:
    """Merge {"NxK/mB": {block_n, ku, num_warps, num_stages}} from `path` (default $DSV41_GEMV_CACHE) into the
    in-process table (entries already chosen in this process win). Returns how many were added."""
    global _CACHE_LOADED
    path = path or cache_path()
    if not path or not os.path.exists(path):
        return 0
    with open(path) as f:
        raw = json.load(f)
    n = 0
    for k, v in (raw.get("configs") or raw).items():
        nk, mb = k.split("/m")
        N, K = (int(x) for x in nk.split("x"))
        key = (N, K, int(mb))
        if key not in _CONFIGS:
            _CONFIGS[key] = GemvConfig(**{f: int(v[f]) for f in ("block_n", "ku", "num_warps", "num_stages")})
            n += 1
    _CACHE_LOADED = path
    return n


def save_cache(path: Optional[str] = None) -> Optional[str]:
    path = path or cache_path()
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"configs": {_key_str(k): c.as_dict() for k, c in sorted(_CONFIGS.items())},
            "tune_log": {_key_str(k): v for k, v in sorted(_TUNE_LOG.items())}}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    return path


def set_config(N: int, K: int, M: int, cfg: GemvConfig) -> None:
    _CONFIGS[config_key(N, K, M)] = cfg


def reset_configs() -> None:
    global _CACHE_LOADED
    _CONFIGS.clear()
    _TUNE_LOG.clear()
    _CACHE_LOADED = None


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def get_config(N: int, K: int, M: int, device=None) -> GemvConfig:
    """The config for (N, K, block_m(M)): this process's choice, else the cache file's, else tuned now (eager only),
    else DEFAULT_CONFIG when tuning is off. Inside a CUDA-graph capture an unknown shape is an error, not a tune."""
    key = config_key(N, K, M)
    cfg = _CONFIGS.get(key)
    if cfg is not None:
        return cfg
    if _CACHE_LOADED is None and cache_path():
        load_cache()
        cfg = _CONFIGS.get(key)
        if cfg is not None:
            return cfg
    if not tune_enabled():
        cfg = _CONFIGS[key] = DEFAULT_CONFIG
        return cfg
    if _capturing():
        raise RuntimeError(f"mxfp8_gemv: no config for {_key_str(key)} during CUDA-graph capture; run an eager warm-up "
                           f"step first (StaticStep.capture does) or set DSV41_GEMV_TUNE=0")
    cfg = autotune(N, K, M, device)
    save_cache()
    return cfg


def _synthetic(N: int, K: int, M: int, copies: int, device):
    """Valid finite e4m3 bytes (|code| <= 0x77) and e8m0 codes near 1.0, `copies` weight copies."""
    g = torch.Generator(device="cpu").manual_seed(1234 + N + 7 * K)

    def fp8(*shape):
        b = torch.randint(0, 0x78, shape, generator=g, dtype=torch.int32) | (torch.randint(0, 2, shape, generator=g,
                                                                                            dtype=torch.int32) << 7)
        return b.to(torch.uint8).to(device).view(FP8)

    def e8(*shape):
        return torch.randint(120, 134, shape, generator=g, dtype=torch.int32).to(torch.uint8).to(device).view(E8)

    xq, xs = fp8(M, K), e8(M, K // BLOCK)
    w0, s0 = fp8(N, K), e8((N + 31) // 32, K // BLOCK)
    ws = [w0] + [w0.clone() for _ in range(copies - 1)]
    ss = [s0] + [s0.clone() for _ in range(copies - 1)]
    return xq, xs, ws, ss


def time_graph(fn_calls: List, reps: int = 3) -> float:
    """Mean µs per call of `fn_calls` (zero-arg callables) captured into one CUDA graph and replayed `reps` times.
    Every callable must already have run once eagerly (JIT compiled)."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_calls:  # one more eager pass on the capture stream (allocator / lazy init)
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for f in fn_calls:
            f()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * len(fn_calls))


def autotune(N: int, K: int, M: int, device=None, candidates=CANDIDATES, calls: int = TUNE_CALLS) -> GemvConfig:
    """Time every candidate on rotating synthetic weights (enough copies to stream from HBM) under a CUDA graph and
    keep the fastest for (N, K, block_m(M)). Syncs; never call while capturing."""
    if _capturing():
        raise RuntimeError("mxfp8_gemv.autotune during CUDA-graph capture")
    key = config_key(N, K, M)
    device = torch.device(device or "cuda")
    copies = max(1, min(TUNE_MAX_COPIES, math.ceil(TUNE_STREAM_BYTES / max(1, N * K))))
    xq, xs, ws, ss = _synthetic(N, K, M, copies, device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    res, errs = {}, {}
    for cfg in candidates:
        try:
            mxfp8_gemv(xq, xs, ws[0], ss[0], out, config=cfg)
            torch.cuda.synchronize(device)
            fns = [lambda i=i, c=cfg: mxfp8_gemv(xq, xs, ws[i % copies], ss[i % copies], out, config=c)
                   for i in range(max(calls, copies))]
            res[cfg] = time_graph(fns)
        except Exception as e:  # a config that does not compile/launch on this GPU is skipped, and reported
            errs[str(cfg.as_dict())] = repr(e)[:200]
    if not res:
        raise RuntimeError(f"mxfp8_gemv.autotune {_key_str(key)}: no candidate ran: {errs}")
    best = min(res, key=lambda c: res[c])
    _CONFIGS[key] = best
    _TUNE_LOG[key] = {"us": {json.dumps(c.as_dict()): round(t, 3) for c, t in res.items()}, "errors": errs,
                      "best": best.as_dict(), "best_us": round(res[best], 3), "copies": copies}
    del xq, xs, ws, ss, out
    return best


def describe() -> dict:
    return {"impl": "gemv", "triton": getattr(triton, "__version__", None) if HAVE_TRITON else None,
            "tune": tune_enabled(), "cache_file": cache_path(), "cache_loaded": _CACHE_LOADED,
            "configs": {_key_str(k): c.as_dict() for k, c in sorted(_CONFIGS.items())},
            "tuned_us": {_key_str(k): v.get("best_us") for k, v in sorted(_TUNE_LOG.items())}}


# ------------------------------------------------------------------------------------------------ Triton kernel
if HAVE_TRITON:

    @triton.jit
    def _t9b_pair_scale(sa, sb):
        """pair_scale_bits line for line, int32 e8m0 codes -> fp32 2^(sa+sb-254), bit-exact."""
        e = sa + sb - 254
        normal = (tl.minimum(tl.maximum(e, -126), 128) + 127) << 23
        sh = tl.minimum(tl.maximum(e + 149, 0), 22)
        sub = (sh * 0 + 1) << sh
        bits = tl.where(e >= -126, normal, tl.where(e >= -149, sub, 0))
        bits = tl.where((sa == 255) | (sb == 255), 0x7FC00000, bits)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def t9b_pair_scale_probe(sa_ptr, sb_ptr, out_ptr, n, BLOCK_P: tl.constexpr):
        """Test hook: out[i] = _t9b_pair_scale(sa[i], sb[i]) (uint8 codes in, fp32 out)."""
        i = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        ok = i < n
        a = tl.load(sa_ptr + i, mask=ok, other=127).to(tl.int32)
        b = tl.load(sb_ptr + i, mask=ok, other=127).to(tl.int32)
        tl.store(out_ptr + i, _t9b_pair_scale(a, b), mask=ok)

    @triton.jit
    def t9b_mxfp8_gemv(xq, xs, w, s, out, N, KB,
                       stride_xm, stride_xsm, stride_wn, stride_sn, stride_om,
                       M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr,
                       EVEN_K: tl.constexpr):
        """Program p: output columns n in [p*BLOCK_N, (p+1)*BLOCK_N) for all M rows. Per iteration: KU 32-byte K
        blocks of each of its BLOCK_N weight rows (one [BLOCK_N, KU, 32] load, K-major), their e8m0 block scales
        s[n // 32, k], then per row m: 32-wide fp32 block partials, exact pair scale, and one fp32 add per block into
        acc[m, n] in K order. Rows n >= N read row 0 (in bounds, unmasked loads stay vectorized) and are not stored."""
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = rn < N
        rn_in = tl.where(n_ok, rn, 0)
        ru = tl.arange(0, KU)
        rj = tl.arange(0, 32)
        rm = tl.arange(0, BLOCK_M)
        w_rows = w + rn_in.to(tl.int64)[:, None, None] * stride_wn            # [BN, 1, 1]
        s_rows = s + (rn_in // 32).to(tl.int64)[:, None] * stride_sn          # [BN, 1]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for kb0 in range(0, KB, KU):
            kb = kb0 + ru                                                      # [KU] K-block ids
            kofs = kb[None, :, None] * 32 + rj[None, None, :]                  # [1, KU, 32] byte offsets in a row
            if EVEN_K:
                wt = tl.load(w_rows + kofs)
                sb = tl.load(s_rows + kb[None, :]).to(tl.int32)
            else:
                k_ok = kb < KB
                wt = tl.load(w_rows + kofs, mask=k_ok[None, :, None], other=0.0)
                sb = tl.load(s_rows + kb[None, :], mask=k_ok[None, :], other=127).to(tl.int32)
            wf = wt.to(tl.float32)                                             # [BN, KU, 32], exact
            for m in tl.static_range(M):
                if EVEN_K:
                    xt = tl.load(xq + m * stride_xm + kofs)                    # [1, KU, 32]
                    sa = tl.load(xs + m * stride_xsm + kb).to(tl.int32)        # [KU]
                else:
                    xt = tl.load(xq + m * stride_xm + kofs, mask=k_ok[None, :, None], other=0.0)
                    sa = tl.load(xs + m * stride_xsm + kb, mask=k_ok, other=127).to(tl.int32)
                part = tl.sum(wf * xt.to(tl.float32), axis=2)                  # [BN, KU] block partials
                q = part * _t9b_pair_scale(sa[None, :], sb)                    # exact power-of-two scaling
                for u in tl.static_range(KU):                                  # K order within the iteration
                    qu = tl.sum(tl.where(ru[None, :] == u, q, 0.0), axis=1)    # column u (+0.0s: exact)
                    acc += tl.where(rm[:, None] == m, qu[None, :], 0.0)        # row m only (+0.0 elsewhere)
        tl.store(out + rm[:, None] * stride_om + rn[None, :], acc.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & n_ok[None, :])


def pair_scale_device(sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """The kernel's own scale function on the device (test hook): uint8 codes [n] x [n] -> fp32 [n]."""
    if not HAVE_TRITON:
        raise RuntimeError("pair_scale_device needs triton")
    sa, sb = sa.view(torch.uint8).contiguous(), sb.view(torch.uint8).contiguous()
    out = torch.empty(sa.numel(), dtype=torch.float32, device=sa.device)
    t9b_pair_scale_probe[(triton.cdiv(sa.numel(), 1024),)](sa, sb, out, sa.numel(), BLOCK_P=1024)
    return out


def mxfp8_gemv(xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor,
               config: Optional[GemvConfig] = None) -> torch.Tensor:
    """out[M, N] (bf16) <- MXFP8 GEMV of xq [M, K] fp8 (scales xs [M, K/32] e8m0) with w [N, K] fp8 (32x32 block
    scales s [ceil(N/32), K/32] e8m0), M <= 8. One launch on CUDA (graph-capturable once the shape's config is
    known); the torch algorithm on CPU."""
    M, N, K = check_operands(xq, xs, w, s, out)
    if not xq.is_cuda:
        return gemv_torch(xq, xs, w, s, out)
    if not HAVE_TRITON:  # pragma: no cover
        raise RuntimeError("mxfp8_gemv on CUDA needs triton")
    cfg = config or get_config(N, K, M, xq.device)
    kb = K // BLOCK
    t9b_mxfp8_gemv[(triton.cdiv(N, cfg.block_n),)](
        xq, xs.view(torch.uint8), w, s.view(torch.uint8), out, N, kb,
        xq.stride(0), xs.stride(0), w.stride(0), s.stride(0), out.stride(0),
        M=M, BLOCK_M=block_m(M), BLOCK_N=cfg.block_n, KU=cfg.ku, EVEN_K=kb % cfg.ku == 0,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    return out


def linear_fp8(xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The fp8_gemm call of model.py:198-205 as a GEMV: xq [..., K] / xs [..., K/32] from the reference act_quant ->
    a new bf16 [..., N] (kernel.py:299 allocates the same shape in the default dtype, bf16 in this model)."""
    K = xq.size(-1)
    N = w.size(0)
    out = torch.empty(*xq.shape[:-1], N, dtype=torch.bfloat16, device=xq.device)
    mxfp8_gemv(xq.view(-1, K), xs.view(-1, K // BLOCK), w, s, out.view(-1, N))
    return out
