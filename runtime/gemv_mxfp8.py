"""T9b-1: a bandwidth-bound MXFP8 GEMV for the model's fp8 `Linear` layers at M <= 8 rows (`--fp8-impl gemv`).

    out = mxfp8_gemv(xq, xs, w, s, out)    # xq fp8 [M,K], xs e8m0 [M,K/32], w fp8 [N,K], s e8m0 [ceil(N/32),K/32]
    y = linear_fp8(xq, xs, w, s)           # same, allocating out [*xq.shape[:-1], N] bf16 (the reference's c)

What it replaces: the reference `fp8_gemm` (kernel.py:207-307) as called by `linear()` (model.py:196-205) for every
`Linear` whose weight is fp8 with 32x32 E8M0 block scales (attention wq_a/wq_b/wkv/wo_b, indexer wq_b, shared experts
w1/w3/w2, Engram wkv: 290 calls per 1-row step). That kernel is a 32x128 tile GEMM; at M = 1 it reads weights at
~0.3 TB/s (docs/T9A-DEEPGEMM-ABI-20261106.md addendum).

Layout of the work (T9b-1 fix round, docs/HANDOFF-T9B1-FIX.md):
  * split-K: grid (ceil(N / BLOCK_N), NSPLIT). Program (pn, ks) owns BLOCK_N weight rows and the contiguous K-block
    range [ks*PER, min((ks+1)*PER, KB)) (`split_plan`), so even N = 512 fills the GPU: SPLIT_K is chosen so there are
    >= 2 x SM-count programs, capped at >= 4 K blocks per program (`auto_split_k`), and tuned with the rest.
  * all M rows per program at once: per 32-wide K block one [BLOCK_M, 32] activation tile and one [BLOCK_N, 32]
    weight tile, block partials for every (m, n) together -- either `tl.dot` on the fp8 operands (MMA=1, tensor cores,
    BLOCK_M = 16 with rows >= M clamped to row 0 and never stored) or a broadcast multiply + 32-wide `tl.sum`
    (MMA=0, BLOCK_M = next power of two >= M). KU K blocks per loop iteration (unrolled: KU independent loads in
    flight per row), accumulated one after the other in K order.
  * NSPLIT == 1: bf16 stored directly. NSPLIT > 1: fp32 partials into a static workspace ws[NSPLIT, M, N]
    (allocated once per (device, M, N, NSPLIT) at tune / first eager call, never freed: graph-capturable), then
    `t9b_splitk_reduce` sums ws over the split axis in fixed order 0..NSPLIT-1 from +0.0 and stores bf16.

Arithmetic, and how it relates to the reference kernel (kernel.py:252-272):
  reference  per 32-wide K block k (in K order): C = fp8 MMA of the [32x32]x[32x128] tile in fp32 (tensor-core
             internal order), then `acc += C * sa[m,k] * sb[n//32,k]` in fp32, acc starting at 0; bf16 store.
  here       per K block k: p = fp32 sum of the 32 exact e4m3 x e4m3 products of row m and row n (tensor core for
             MMA=1, tl.sum for MMA=0), then `acc += p * 2^(sa + sb - 254)` in fp32 -- within one split in K order
             from +0.0; then the NSPLIT split partials summed in split order from +0.0; bf16 (round to nearest even).
  * the scale: sa, sb are e8m0 codes; 2^(sa+sb-254) is computed as fp32(2^(sa-127)) * fp32(2^(sb-127)), each factor
    built from its bits (`_t9b_pow2`: code 0 -> the subnormal 2^-127, 255 -> NaN). The fp32 product of two powers of
    two is exactly `pair_scale` (subnormal results exact, 2^-150 and below -> 0, above 2^127 -> inf, NaN in -> NaN):
    asserted on all 65,536 code pairs on the CPU and, through the kernel's own helper, on the device
    (tests/gpu/test_gemv_gpu.py, which would catch a flush-to-zero multiply). (C*sa)*sb == C*2^(sa+sb-254) whenever
    neither side under/overflows (scales here are ~2^-22..2^0).
  * the weight scale is read at s[n // 32, k] directly (no expansion).
  * deviations from the reference, both in the 1-ulp class (bench: `bench/gemv_bench.py`, gate <= 0.5 % of outputs):
      1. INSIDE one 32-wide block: the 32 exact products are summed by the tensor core (MMA=1, the same kind of
         instruction as the reference) or by `tl.sum` (MMA=0). Whenever that block sum is exact in fp32 the two agree.
      2. ACROSS blocks, split-K only (NSPLIT > 1): the reference's single K-order chain becomes NSPLIT K-order chains
         added in split order. NSPLIT == 1 keeps the reference's chain exactly.
    Either can move a bf16 output one ulp. Which configs give identical bits: those with the same `numerics_key`
    (same split partition, same MMA) -- asserted on the box (tests/gpu/test_gemv_gpu.py); configs that differ in it
    may differ by an ulp, so a choice is pinned across processes by the tune cache file.

Tuning: (BLOCK_N, KU, num_warps, num_stages, SPLIT_K, MMA). A choice is made once per (N, K, BLOCK_M) by timing
`candidate_configs` under a CUDA graph on rotating synthetic weights (so they stream from HBM, as in the decode step),
kept for the life of the process (a StaticStep relaunch reuses it), and persisted to the JSON file named by
$DSV41_GEMV_CACHE when set (so another process reuses it too; entries from before split-K -- no `split_k` -- are
ignored). Tuning refuses to run while a CUDA graph is being captured; StaticStep's warm-up steps run it before capture.
$DSV41_GEMV_TUNE=0 skips timing and uses DEFAULT_CONFIG (split-K resolved for the shape and SM count).

Graph capture: one Triton launch per call (two when NSPLIT > 1), static shapes, static workspace, no host sync, no
allocation inside a capture. CPU tensors (the Mac tests) run `gemv_torch`, the same algorithm (same split order) in
torch.

Triton is imported lazily-by-guard: the module imports on the Mac (no triton) and everything but the kernel works.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, fields, replace
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
MMA_BLOCK_M = 16              # tl.dot needs M >= 16: the MMA form pads the M rows to 16 (rows >= M never stored)
E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn
NAN_BITS = 0x7FC00000
INF_BITS = 0x7F800000
DEFAULT_SMS = 148             # GB300 (the box); used only where no CUDA device can be asked (the Mac tests)
SPLIT_MIN_KB = 4              # split-K never gives a program fewer than 4 K blocks
SPLIT_PROGRAMS_PER_SM = 2     # split-K aims at >= 2 x SM-count programs


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


def pow2_code(c: torch.Tensor) -> torch.Tensor:
    """fp32 2^(c - 127) for an e8m0 code c, from its bits: c << 23, except code 0 -> the subnormal 2^-127 (bit 22)
    and 255 -> NaN. The kernel's `_t9b_pow2` line for line; pow2_code(sa) * pow2_code(sb) == pair_scale(sa, sb)."""
    c = c.to(torch.int32)
    bits = torch.where(c == 0, torch.full_like(c, 1 << 22), c << 23)
    return torch.where(c == 255, torch.full_like(c, NAN_BITS), bits).view(torch.float32)


def weight_scale_rows(n: torch.Tensor) -> torch.Tensor:
    """Row of the [ceil(N/32), K/32] weight-scale matrix that output row n uses: its 32x32 block row, n // 32
    (kernel.py:263 with group_size 32). The kernel indexes s[n // 32, k] directly."""
    return torch.div(n, BLOCK, rounding_mode="floor")


def block_m(m: int) -> int:
    """The broadcast form's accumulator height: the next power of two >= m (1, 2, 4, 8). Also the tune-key bucket."""
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


# ------------------------------------------------------------------------------------------------ split-K plan
def split_plan(KB: int, split_k: int) -> Tuple[int, int]:
    """(PER, NSPLIT) for KB K blocks and a requested split: PER = ceil(KB / split) K blocks per split, NSPLIT =
    ceil(KB / PER) splits (so no split is empty; the last may be shorter). Split ks covers [ks*PER, min((ks+1)*PER,
    KB)). This, not the requested number, is what fixes the summation order."""
    KB = int(KB)
    if KB < 1:
        raise ValueError(f"split_plan: KB = {KB}")
    s = max(1, min(int(split_k), KB))
    per = -(-KB // s)
    return per, -(-KB // per)


def auto_split_k(N: int, K: int, block_n: int, sms: int) -> int:
    """Smallest split giving >= SPLIT_PROGRAMS_PER_SM x `sms` programs over ceil(N / block_n) column tiles, capped so
    every program keeps >= SPLIT_MIN_KB K blocks (and >= 1)."""
    KB = int(K) // BLOCK
    tiles = -(-int(N) // int(block_n))
    want = -(-SPLIT_PROGRAMS_PER_SM * int(sms) // tiles)
    cap = max(1, KB // SPLIT_MIN_KB)
    return max(1, min(want, cap))


# ------------------------------------------------------------------------------------------------ torch algorithm
def block_partials(xq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 [M, N, K/32]: for each row m, weight row n and 32-wide K block k, the sum of the 32 exact products."""
    M, K = xq.shape
    N = w.size(0)
    kb = K // BLOCK
    return (xq.float().view(M, 1, kb, BLOCK) * w.float().view(1, N, kb, BLOCK)).sum(-1)


def gemv_torch_f32(xq, xs, w, s, split_k: int = 1) -> torch.Tensor:
    """The kernel's fp32 result before the bf16 store, in torch: block partials scaled by pow2(sa) * pow2(sb) (==
    the exact pair scale) with the weight scale read at s[n // 32, k]; per split (`split_plan`) one fp32 add per block
    in K order from +0.0; then the split partials added in split order 0..NSPLIT-1 from +0.0 (the reduce kernel; with
    NSPLIT == 1 that is the one chain, the reference's order)."""
    M, N, K = check_operands(xq, xs, w, s)
    kb = K // BLOCK
    part = block_partials(xq, w)                                                  # [M, N, KB]
    sa = xs.view(torch.uint8).to(torch.int32).view(M, 1, kb)
    rows = weight_scale_rows(torch.arange(N, device=w.device))
    sb = s.view(torch.uint8).to(torch.int32).index_select(0, rows).view(1, N, kb)
    q = part * (pow2_code(sa) * pow2_code(sb))
    per, nsplit = split_plan(kb, split_k)
    total = torch.zeros(M, N, dtype=torch.float32, device=xq.device)
    for j in range(nsplit):
        acc = torch.zeros(M, N, dtype=torch.float32, device=xq.device)
        for k in range(j * per, min((j + 1) * per, kb)):
            acc = acc + q[:, :, k]
        total = total + acc
    return total


def gemv_torch(xq, xs, w, s, out: Optional[torch.Tensor] = None, split_k: int = 1) -> torch.Tensor:
    """The kernel's algorithm in torch (CPU tests; any device), bf16 (round to nearest even). split_k = 1 (the CPU
    default) is the reference's single K-order chain."""
    M, N, K = check_operands(xq, xs, w, s, out)
    res = gemv_torch_f32(xq, xs, w, s, split_k).to(torch.bfloat16)
    if out is None:
        return res
    return out.copy_(res)


# ------------------------------------------------------------------------------------------------ configs / tuning
@dataclass(frozen=True)
class GemvConfig:
    block_n: int        # weight rows per program
    ku: int             # 32-wide K blocks per loop iteration (unrolled; accumulated one by one in K order)
    num_warps: int
    num_stages: int
    split_k: int = 0    # requested K splits (split_plan); 0 = auto_split_k for the shape and SM count at use
    mma: int = 0        # 1: tl.dot on fp8 operands (BLOCK_M 16); 0: broadcast multiply + 32-wide tl.sum

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG_FIELDS = tuple(f.name for f in fields(GemvConfig))

# the search space (docs/HANDOFF-T9B1-FIX.md): every combination, plus per BLOCK_N the auto split and twice it
TUNE_BLOCK_N = (32, 64, 128)
TUNE_KU = (2, 4, 8)
TUNE_WARPS = (4, 8)
TUNE_STAGES = (2, 3, 4)
TUNE_MMA = (0, 1)
# conservative default for DSV41_GEMV_TUNE=0 (the round-1 kernel's proven load/sum form, no software pipelining);
# split_k 0 -> auto for the shape
DEFAULT_CONFIG = GemvConfig(32, 4, 4, 1, 0, 0)
TUNE_STREAM_BYTES = 512 << 20   # rotating synthetic weight copies per timed shape (beyond the L2, ~126 MB here)
TUNE_MAX_COPIES = 256
TUNE_CALLS = 64

_CONFIGS: Dict[Tuple[int, int, int], GemvConfig] = {}
_TUNE_LOG: Dict[Tuple[int, int, int], dict] = {}
_CACHE_LOADED: Optional[str] = None
_CACHE_IGNORED: List[str] = []
_SMS: Dict[str, int] = {}
_WORKSPACE: Dict[Tuple[str, int, int, int], torch.Tensor] = {}


def config_key(N: int, K: int, M: int) -> Tuple[int, int, int]:
    return int(N), int(K), block_m(M)


def _key_str(key) -> str:
    return f"{key[0]}x{key[1]}/m{key[2]}"


def cache_path() -> Optional[str]:
    return os.environ.get("DSV41_GEMV_CACHE") or None


def tune_enabled() -> bool:
    return os.environ.get("DSV41_GEMV_TUNE", "1") not in ("0", "false", "off")


def num_sms(device=None) -> int:
    """Multiprocessor count of `device` (cached); DEFAULT_SMS without CUDA."""
    if not torch.cuda.is_available():
        return DEFAULT_SMS
    dev = torch.device(device or "cuda")
    if dev.type != "cuda":
        return DEFAULT_SMS
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    k = f"cuda:{idx}"
    if k not in _SMS:
        _SMS[k] = int(torch.cuda.get_device_properties(idx).multi_processor_count)
    return _SMS[k]


def resolve_config(cfg: GemvConfig, N: int, K: int, sms: Optional[int] = None) -> GemvConfig:
    """`cfg` with split_k = 0 (auto) replaced by auto_split_k for (N, K, cfg.block_n, sms)."""
    if int(cfg.split_k) > 0:
        return cfg
    return replace(cfg, split_k=auto_split_k(N, K, cfg.block_n, sms or DEFAULT_SMS))


def kernel_block_m(cfg: GemvConfig, M: int) -> int:
    return MMA_BLOCK_M if cfg.mma else block_m(M)


def numerics_key(cfg: GemvConfig, N: int, K: int, sms: Optional[int] = None) -> Tuple[int, int, int]:
    """What fixes a config's output bits: (PER, NSPLIT) of its split partition and the block-partial form (MMA).
    Configs with equal keys give bit-identical outputs (asserted on the box); the rest only change speed."""
    per, nsplit = split_plan(int(K) // BLOCK, resolve_config(cfg, N, K, sms).split_k)
    return per, nsplit, int(cfg.mma)


def tune_space() -> str:
    """$DSV41_GEMV_TUNE_SPACE: "full" (default, the handoff's grid) or "quick" (KU 4, 4 warps, 2 stages; every
    BLOCK_N, split and MMA -- for tests that tune inside a mini-model warm-up)."""
    v = os.environ.get("DSV41_GEMV_TUNE_SPACE", "full")
    return v if v in ("full", "quick") else "full"


def candidate_configs(N: int, K: int, M: int, sms: Optional[int] = None) -> List[GemvConfig]:
    """The tuning candidates for (N, K, M): BLOCK_N x KU x num_warps x num_stages x MMA x {auto split, 2 x auto
    split} (capped; duplicate split partitions dropped). Pruned: the broadcast form (MMA=0) at BLOCK_M >= 4 with
    BLOCK_N = 128 (a [BLOCK_M, 128, 32] fp32 product per block does not fit in registers). tune_space() "quick"
    narrows KU / warps / stages to one value each."""
    sms = sms or DEFAULT_SMS
    KB = int(K) // BLOCK
    bm = block_m(M)
    quick = tune_space() == "quick"
    kus, warps, stages = ((4,), (4,), (2,)) if quick else (TUNE_KU, TUNE_WARPS, TUNE_STAGES)
    out: List[GemvConfig] = []
    for mma in TUNE_MMA:
        for bn in TUNE_BLOCK_N:
            if not mma and bm >= 4 and bn >= 128:
                continue
            s0 = auto_split_k(N, K, bn, sms)
            splits, seen = [], set()
            for sk in (s0, min(max(1, KB // SPLIT_MIN_KB), 2 * s0)):
                plan = split_plan(KB, sk)
                if plan not in seen:
                    seen.add(plan)
                    splits.append(sk)
            for ku, nw, ns, sk in itertools.product(kus, warps, stages, splits):
                out.append(GemvConfig(bn, ku, nw, ns, sk, mma))
    return out


def default_config(N: int, K: int, M: int, sms: Optional[int] = None) -> GemvConfig:
    return resolve_config(DEFAULT_CONFIG, N, K, sms)


def load_cache(path: Optional[str] = None) -> int:
    """Merge {"NxK/mB": {block_n, ku, num_warps, num_stages, split_k, mma}} from `path` (default $DSV41_GEMV_CACHE)
    into the in-process table (entries already chosen in this process win). Entries without every field (written
    before split-K) are ignored and listed in describe()["cache_ignored"]. Returns how many were added."""
    global _CACHE_LOADED
    path = path or cache_path()
    if not path or not os.path.exists(path):
        return 0
    with open(path) as f:
        raw = json.load(f)
    n = 0
    for k, v in (raw.get("configs") or raw).items():
        if not isinstance(v, dict) or any(f not in v for f in CONFIG_FIELDS) or int(v["split_k"]) < 1:
            if k not in _CACHE_IGNORED:
                _CACHE_IGNORED.append(k)
            continue
        nk, mb = k.split("/m")
        N, K = (int(x) for x in nk.split("x"))
        key = (N, K, int(mb))
        if key not in _CONFIGS:
            _CONFIGS[key] = GemvConfig(**{f: int(v[f]) for f in CONFIG_FIELDS})
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
    _CONFIGS[config_key(N, K, M)] = resolve_config(cfg, N, K, num_sms())


def reset_configs() -> None:
    """Forget configs and tune logs (not the workspaces: a captured graph may still point at them)."""
    global _CACHE_LOADED
    _CONFIGS.clear()
    _TUNE_LOG.clear()
    _CACHE_IGNORED.clear()
    _CACHE_LOADED = None


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def get_config(N: int, K: int, M: int, device=None) -> GemvConfig:
    """The config for (N, K, block_m(M)): this process's choice, else the cache file's, else tuned now (eager only),
    else DEFAULT_CONFIG (split resolved) when tuning is off. Inside a CUDA-graph capture an unknown shape is an error,
    not a tune."""
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
        cfg = _CONFIGS[key] = default_config(N, K, M, num_sms(device))
        return cfg
    if _capturing():
        raise RuntimeError(f"mxfp8_gemv: no config for {_key_str(key)} during CUDA-graph capture; run an eager warm-up "
                           f"step first (StaticStep.capture does) or set DSV41_GEMV_TUNE=0")
    cfg = autotune(N, K, M, device)
    save_cache()
    return cfg


def workspace(device, M: int, N: int, nsplit: int) -> torch.Tensor:
    """The static fp32 split-K workspace [nsplit, M, N] for (device, M, N, nsplit): allocated on first (eager) use and
    kept for the life of the process (a captured graph keeps pointing at it). Calls sharing it run in stream order
    (main kernel, then its reduce) so they never overlap. Never allocates inside a capture."""
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (f"{dev.type}:{idx}", int(M), int(N), int(nsplit))
    t = _WORKSPACE.get(key)
    if t is None:
        if _capturing():
            raise RuntimeError(f"mxfp8_gemv: split-K workspace {key} first needed during CUDA-graph capture; run an "
                               f"eager warm-up call with the same config first (StaticStep.capture does)")
        t = _WORKSPACE[key] = torch.empty(int(nsplit), int(M), int(N), dtype=torch.float32, device=dev)
    return t


def _synthetic(N: int, K: int, M: int, copies: int, device):
    """Valid finite e4m3 bytes (|code| <= 0x77) and e8m0 codes near 1.0, `copies` weight copies."""
    g = torch.Generator(device="cpu").manual_seed(1234 + N + 7 * K)

    def fp8(*shape):
        b = torch.randint(0, 0x78, shape, generator=g, dtype=torch.int32, device="cpu") | (torch.randint(
            0, 2, shape, generator=g, dtype=torch.int32, device="cpu") << 7)   # explicit cpu: default device may be cuda
        return b.to(torch.uint8).to(device).view(FP8)

    def e8(*shape):
        return torch.randint(120, 134, shape, generator=g, dtype=torch.int32, device="cpu").to(torch.uint8).to(device).view(E8)

    xq, xs = fp8(M, K), e8(M, K // BLOCK)
    w0, s0 = fp8(N, K), e8((N + 31) // 32, K // BLOCK)
    ws = [w0] + [w0.clone() for _ in range(copies - 1)]
    ss = [s0] + [s0.clone() for _ in range(copies - 1)]
    return xq, xs, ws, ss


def time_graph(fn_calls: List, reps: int = 3) -> float:
    """Mean µs per call of `fn_calls` (zero-arg callables) captured into one fresh CUDA graph and replayed `reps`
    times. Every callable must already have run once eagerly (JIT compiled, workspace allocated)."""
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
    us = a.elapsed_time(b) * 1e3 / (reps * len(fn_calls))
    del g
    return us


def autotune(N: int, K: int, M: int, device=None, candidates: Optional[List[GemvConfig]] = None,
             calls: int = TUNE_CALLS) -> GemvConfig:
    """Time every candidate (default `candidate_configs`) on rotating synthetic weights (enough copies to stream from
    HBM) -- each candidate in its OWN freshly captured graph, its config bound per call -- and keep the fastest for
    (N, K, block_m(M)). Syncs; never call while capturing. The full per-candidate table goes to the tune log."""
    if _capturing():
        raise RuntimeError("mxfp8_gemv.autotune during CUDA-graph capture")
    key = config_key(N, K, M)
    device = torch.device(device or "cuda")
    sms = num_sms(device)
    if candidates is None:
        candidates = candidate_configs(N, K, M, sms)
    candidates = [resolve_config(c, N, K, sms) for c in candidates]
    copies = max(1, min(TUNE_MAX_COPIES, math.ceil(TUNE_STREAM_BYTES / max(1, N * K))))
    xq, xs, ws, ss = _synthetic(N, K, M, copies, device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    res, errs = {}, {}
    for cfg in candidates:
        if cfg in res:
            continue
        try:
            mxfp8_gemv(xq, xs, ws[0], ss[0], out, config=cfg)
            torch.cuda.synchronize(device)
            fns = [lambda i=i, c=cfg: mxfp8_gemv(xq, xs, ws[i % copies], ss[i % copies], out, config=c)
                   for i in range(max(calls, copies))]
            res[cfg] = time_graph(fns)
        except Exception as e:  # a config that does not compile/launch on this GPU is skipped, and reported
            errs[json.dumps(cfg.as_dict())] = repr(e)[:200]
    if not res:
        raise RuntimeError(f"mxfp8_gemv.autotune {_key_str(key)}: no candidate ran: {errs}")
    best = min(res, key=lambda c: res[c])
    ranked = sorted(res, key=lambda c: res[c])
    _CONFIGS[key] = best
    _TUNE_LOG[key] = {"us": {json.dumps(c.as_dict()): round(res[c], 3) for c in ranked}, "errors": errs,
                      "best": best.as_dict(), "best_us": round(res[best], 3), "worst_us": round(res[ranked[-1]], 3),
                      "n_timed": len(res), "n_failed": len(errs), "copies": copies, "sms": sms}
    del xq, xs, ws, ss, out
    return best


def describe() -> dict:
    return {"impl": "gemv", "triton": getattr(triton, "__version__", None) if HAVE_TRITON else None,
            "tune": tune_enabled(), "tune_space": tune_space(), "cache_file": cache_path(),
            "cache_loaded": _CACHE_LOADED,
            "cache_ignored": list(_CACHE_IGNORED),
            "configs": {_key_str(k): c.as_dict() for k, c in sorted(_CONFIGS.items())},
            "tuned_us": {_key_str(k): v.get("best_us") for k, v in sorted(_TUNE_LOG.items())},
            "workspaces": {f"{k[0]}/m{k[1]}/n{k[2]}/s{k[3]}": int(t.numel()) * 4 for k, t in _WORKSPACE.items()}}


# ------------------------------------------------------------------------------------------------ Triton kernels
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
    def _t9b_pow2(c):
        """pow2_code line for line: int32 e8m0 code -> fp32 2^(c-127) (0 -> subnormal 2^-127, 255 -> NaN), exact."""
        bits = tl.where(c == 0, 0x400000, c << 23)
        bits = tl.where(c == 255, 0x7FC00000, bits)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def t9b_pair_scale_probe(sa_ptr, sb_ptr, out_ptr, n, BLOCK_P: tl.constexpr, MODE: tl.constexpr):
        """Test hook (uint8 codes in, fp32 out): MODE 0 out[i] = _t9b_pair_scale(sa[i], sb[i]); MODE 1 out[i] =
        _t9b_pow2(sa[i]) * _t9b_pow2(sb[i]) -- exactly the scale expression the GEMV kernel evaluates."""
        i = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        ok = i < n
        a = tl.load(sa_ptr + i, mask=ok, other=127).to(tl.int32)
        b = tl.load(sb_ptr + i, mask=ok, other=127).to(tl.int32)
        if MODE == 0:
            r = _t9b_pair_scale(a, b)
        else:
            r = _t9b_pow2(a) * _t9b_pow2(b)
        tl.store(out_ptr + i, r, mask=ok)

    @triton.jit
    def t9b_mxfp8_gemv(xq, xs, w, s, out, ws, M, N, KB, PER,
                       stride_xm, stride_xsm, stride_wn, stride_sn, stride_om, freqs,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr, EVEN: tl.constexpr,
                       MMA: tl.constexpr, SPLIT: tl.constexpr, ROPE: tl.constexpr = False, HEAD_DIM: tl.constexpr = 2,
                       ROPE_DIM: tl.constexpr = 0, RE_MODE: tl.constexpr = 0, IM_MODE: tl.constexpr = 0):
        """Program (pn, ks): output columns [pn*BLOCK_N, (pn+1)*BLOCK_N) for all M rows over K blocks
        [ks*PER, min((ks+1)*PER, KB)). Per K block (KU per loop iteration, unrolled, in K order): the [BLOCK_M, 32]
        activation tile and the [BLOCK_N, 32] weight tile, fp32 block partials for every (m, n) (tl.dot or broadcast
        + tl.sum), times pow2(sa[m]) * pow2(sb[n // 32]), one fp32 add into acc[m, n]. Rows m >= M and n >= N read
        row 0 (in bounds, unmasked vector loads) and are never stored. EVEN: every split is PER long and PER % KU == 0;
        otherwise blocks past the split's end re-read its last block and are not added. SPLIT: fp32 acc -> ws[ks]
        (summed by t9b_splitk_reduce); else bf16 -> out. ROPE (T9b-2 #13, direct store only): the bf16-rounded values
        of every head's last ROPE_DIM columns are rotated in pairs by freqs (fp32 [ROPE_DIM/2, 2]) with the
        contraction form (RE_MODE, IM_MODE) torch's complex multiply uses, then rounded to bf16 -- exactly
        apply_rotary_emb on the stored output; the accumulation above is untouched."""
        pid_n = tl.program_id(0)
        ks = tl.program_id(1)
        kb_lo = ks * PER
        kb_hi = tl.minimum(kb_lo + PER, KB)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = rn < N
        rn_in = tl.where(n_ok, rn, 0)
        rm = tl.arange(0, BLOCK_M)
        m_ok = rm < M
        rm_in = tl.where(m_ok, rm, 0)
        rj = tl.arange(0, 32)
        w_rows = w + rn_in.to(tl.int64)[:, None] * stride_wn + rj[None, :]          # [BN, 32] block-0 bytes
        x_rows = xq + rm_in.to(tl.int64)[:, None] * stride_xm + rj[None, :]          # [BM, 32]
        s_rows = s + (rn_in // 32).to(tl.int64) * stride_sn                           # [BN]
        xs_rows = xs + rm_in.to(tl.int64) * stride_xsm                                # [BM]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for kb0 in range(kb_lo, kb_hi, KU):
            for u in tl.static_range(KU):                                             # K order within the window
                kb = kb0 + u
                if EVEN:
                    kbc = kb
                else:
                    kbc = tl.minimum(kb, kb_hi - 1)
                wt = tl.load(w_rows + kbc * 32)                                       # [BN, 32] fp8
                xt = tl.load(x_rows + kbc * 32)                                       # [BM, 32] fp8
                sbf = _t9b_pow2(tl.load(s_rows + kbc).to(tl.int32))                   # [BN] 2^(sb-127)
                saf = _t9b_pow2(tl.load(xs_rows + kbc).to(tl.int32))                  # [BM] 2^(sa-127)
                if MMA:
                    part = tl.dot(xt, tl.trans(wt))                                   # [BM, BN] fp32, tensor core
                else:
                    part = tl.sum(xt.to(tl.float32)[:, None, :] * wt.to(tl.float32)[None, :, :], axis=2)
                q = part * (saf[:, None] * sbf[None, :])                              # exact power-of-two scaling
                if EVEN:
                    acc += q
                else:
                    acc = tl.where(kb < kb_hi, acc + q, acc)
        mask = m_ok[:, None] & n_ok[None, :]
        if SPLIT:
            tl.store(ws + ks * M * N + rm[:, None] * N + rn[None, :], acc, mask=mask)
        elif ROPE:
            xb = acc.to(tl.bfloat16).to(tl.float32)                                   # the stored q values
            a, b = tl.split(tl.reshape(xb, [BLOCK_M, BLOCK_N // 2, 2]))               # even / odd columns
            ce = pid_n * BLOCK_N + 2 * tl.arange(0, BLOCK_N // 2)
            hcol = ce % HEAD_DIM
            is_r = hcol >= HEAD_DIM - ROPE_DIM
            fi = tl.where(is_r, (hcol - (HEAD_DIM - ROPE_DIM)) // 2, 0)
            c = tl.broadcast_to(tl.load(freqs + 2 * fi)[None, :], [BLOCK_M, BLOCK_N // 2])
            d = tl.broadcast_to(tl.load(freqs + 2 * fi + 1)[None, :], [BLOCK_M, BLOCK_N // 2])
            rr = tl.broadcast_to(is_r[None, :], [BLOCK_M, BLOCK_N // 2])
            a2 = tl.where(rr, _t9b2_rope_re(a, b, c, d, RE_MODE), a)
            b2 = tl.where(rr, _t9b2_rope_im(a, b, c, d, IM_MODE), b)
            val = tl.reshape(tl.join(a2, b2), [BLOCK_M, BLOCK_N])
            tl.store(out + rm[:, None] * stride_om + rn[None, :], val.to(tl.bfloat16), mask=mask)
        else:
            tl.store(out + rm[:, None] * stride_om + rn[None, :], acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def t9b_splitk_reduce(ws, out, M, N, NSPLIT, stride_om, freqs, BLOCK_R: tl.constexpr, ROPE: tl.constexpr = False,
                          HEAD_DIM: tl.constexpr = 2, ROPE_DIM: tl.constexpr = 0, RE_MODE: tl.constexpr = 0,
                          IM_MODE: tl.constexpr = 0):
        """out[m, n] = bf16(((0 + ws[0, m, n]) + ws[1, m, n]) + ... + ws[NSPLIT-1, m, n]): fixed split order. ROPE
        (T9b-2 #13): each lane also sums its pair partner n ^ 1 in the same order (so both see the same two values),
        rounds both to bf16 and stores its own component of the rotated pair (columns outside RoPE: unchanged)."""
        i = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        ok = i < M * N
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        for j in range(0, NSPLIT):
            acc += tl.load(ws + j * M * N + i, mask=ok, other=0.0)
        m = i // N
        n = i - m * N
        if ROPE:
            ip = i ^ 1
            accp = tl.zeros([BLOCK_R], dtype=tl.float32)
            for j in range(0, NSPLIT):
                accp += tl.load(ws + j * M * N + ip, mask=ok, other=0.0)
            x = acc.to(tl.bfloat16).to(tl.float32)
            xp = accp.to(tl.bfloat16).to(tl.float32)
            even = (n & 1) == 0
            a = tl.where(even, x, xp)
            b = tl.where(even, xp, x)
            hcol = n % HEAD_DIM
            is_r = hcol >= HEAD_DIM - ROPE_DIM
            fi = tl.where(is_r, (hcol - hcol % 2 - (HEAD_DIM - ROPE_DIM)) // 2, 0)
            c = tl.load(freqs + 2 * fi)
            d = tl.load(freqs + 2 * fi + 1)
            rot = tl.where(even, _t9b2_rope_re(a, b, c, d, RE_MODE), _t9b2_rope_im(a, b, c, d, IM_MODE))
            tl.store(out + m * stride_om + n, tl.where(is_r, rot, x).to(tl.bfloat16), mask=ok)
        else:
            tl.store(out + m * stride_om + n, acc.to(tl.bfloat16), mask=ok)

    # T9b-2: the RoPE helpers, a verbatim copy of runtime/fused_glue.py's (tests/test_fused_glue_cpu.py asserts the
    # source is identical): explicit mul.rn/add.rn/sub.rn PTX and tl.fma, so the result does not depend on this
    # kernel's fp-fusion setting (the GEMV keeps Triton's default; its accumulation numerics are T9b-1's).
    @triton.jit
    def _t9b2_mul_rn(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _t9b2_add_rn(a, b):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _t9b2_sub_rn(a, b):
        return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _t9b2_rope_re(a, b, c, d, MODE: tl.constexpr):
        """rope_re: MODE 0 a*c - b*d, 1 fma(a, c, -(b*d)), 2 fma(-b, d, a*c); a, b, c, d same shape."""
        if MODE == 0:
            r = _t9b2_sub_rn(_t9b2_mul_rn(a, c), _t9b2_mul_rn(b, d))
        elif MODE == 1:
            r = tl.fma(a, c, -_t9b2_mul_rn(b, d))
        else:
            r = tl.fma(-b, d, _t9b2_mul_rn(a, c))
        return r

    @triton.jit
    def _t9b2_rope_im(a, b, c, d, MODE: tl.constexpr):
        """rope_im: MODE 0 a*d + b*c, 1 fma(a, d, b*c), 2 fma(b, c, a*d)."""
        if MODE == 0:
            r = _t9b2_add_rn(_t9b2_mul_rn(a, d), _t9b2_mul_rn(b, c))
        elif MODE == 1:
            r = tl.fma(a, d, _t9b2_mul_rn(b, c))
        else:
            r = tl.fma(b, c, _t9b2_mul_rn(a, d))
        return r


REDUCE_BLOCK = 1024


def pair_scale_device(sa: torch.Tensor, sb: torch.Tensor, mode: int = 0) -> torch.Tensor:
    """The kernel's own scale functions on the device (test hook): uint8 codes [n] x [n] -> fp32 [n]. mode 0: the
    integer pair scale; mode 1: pow2(sa) * pow2(sb), the expression the GEMV uses."""
    if not HAVE_TRITON:
        raise RuntimeError("pair_scale_device needs triton")
    sa, sb = sa.view(torch.uint8).contiguous(), sb.view(torch.uint8).contiguous()
    out = torch.empty(sa.numel(), dtype=torch.float32, device=sa.device)
    t9b_pair_scale_probe[(triton.cdiv(sa.numel(), 1024),)](sa, sb, out, sa.numel(), BLOCK_P=1024, MODE=int(mode))
    return out


def launch_plan(cfg: GemvConfig, M: int, N: int, K: int) -> dict:
    """Grid and constexprs of one call (metadata only; also what the tests check on the Mac)."""
    kb = K // BLOCK
    per, nsplit = split_plan(kb, cfg.split_k)
    return {"grid": (-(-N // cfg.block_n), nsplit), "PER": per, "NSPLIT": nsplit,
            "BLOCK_M": kernel_block_m(cfg, M), "EVEN": per % cfg.ku == 0 and kb % per == 0, "SPLIT": nsplit > 1,
            "reduce_grid": (-(-(M * N) // REDUCE_BLOCK),) if nsplit > 1 else None}


def check_rope(rope, N: int) -> None:
    """rope = (freqs fp32 [rope_dim/2, 2] contiguous, head_dim, rope_dim, (re_mode, im_mode)) for an output of N."""
    freqs, hd, rd, modes = rope
    if hd < 2 or hd % 2 or rd % 2 or not 0 < rd <= hd or N % hd:
        raise ValueError(f"mxfp8_gemv rope: head_dim {hd} / rope_dim {rd} must be even, 0 < rope_dim <= head_dim, "
                         f"and divide N = {N}")
    if freqs.dtype != torch.float32 or tuple(freqs.shape) != (rd // 2, 2) or not freqs.is_contiguous():
        raise ValueError(f"mxfp8_gemv rope: freqs {tuple(freqs.shape)} {freqs.dtype} must be contiguous fp32 "
                         f"[{rd // 2}, 2]")
    if tuple(modes) not in {(a, b) for a in (0, 1, 2) for b in (0, 1, 2)}:
        raise ValueError(f"mxfp8_gemv rope: modes {modes}")


def mxfp8_gemv(xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor,
               config: Optional[GemvConfig] = None, rope=None) -> torch.Tensor:
    """out[M, N] (bf16) <- MXFP8 GEMV of xq [M, K] fp8 (scales xs [M, K/32] e8m0) with w [N, K] fp8 (32x32 block
    scales s [ceil(N/32), K/32] e8m0), M <= 8. On CUDA one launch (two with split-K), graph-capturable once the
    shape's config is known and its workspace exists; on CPU the torch algorithm with the config's split (1 if
    none). rope (T9b-2 #13, `check_rope`): rotate every head's last rope_dim outputs in the store (the bf16 output
    then equals apply_rotary_emb applied to the plain output); same config, same accumulation."""
    M, N, K = check_operands(xq, xs, w, s, out)
    if rope is not None:
        check_rope(rope, N)
    if not xq.is_cuda:
        sk = resolve_config(config, N, K).split_k if config is not None else 1
        res = gemv_torch(xq, xs, w, s, out, split_k=sk)
        if rope is not None:
            from runtime.fused_glue import rope_rows_torch

            freqs, hd, rd, modes = rope
            res.copy_(rope_rows_torch(res, freqs, hd, rd, tuple(modes), inverse=False))
        return res
    if not HAVE_TRITON:  # pragma: no cover
        raise RuntimeError("mxfp8_gemv on CUDA needs triton")
    cfg = resolve_config(config or get_config(N, K, M, xq.device), N, K, num_sms(xq.device))
    p = launch_plan(cfg, M, N, K)
    ws = workspace(xq.device, M, N, p["NSPLIT"]) if p["SPLIT"] else out
    return _launch(xq, xs, w, s, out, cfg, ws, rope=rope)


def _launch(xq, xs, w, s, out, cfg: GemvConfig, ws, rope=None) -> torch.Tensor:
    """The kernel launch(es) of one call for a resolved config; `ws` is the [NSPLIT, M, N] fp32 workspace when the
    plan splits (else unused). Separate from mxfp8_gemv so tests can drive the kernels through a Triton stand-in.
    rope: as mxfp8_gemv (the epilogue goes to the main kernel's store, or to the reduce when the plan splits)."""
    M, K = xq.shape
    N = w.size(0)
    p = launch_plan(cfg, M, N, K)
    rk = {}
    freqs = out
    if rope is not None:
        freqs, hd, rd, modes = rope
        rk = dict(ROPE=True, HEAD_DIM=int(hd), ROPE_DIM=int(rd), RE_MODE=int(modes[0]), IM_MODE=int(modes[1]))
    t9b_mxfp8_gemv[p["grid"]](
        xq, xs.view(torch.uint8), w, s.view(torch.uint8), out, ws, M, N, K // BLOCK, p["PER"],
        xq.stride(0), xs.stride(0), w.stride(0), s.stride(0), out.stride(0), freqs,
        BLOCK_M=p["BLOCK_M"], BLOCK_N=cfg.block_n, KU=cfg.ku, EVEN=p["EVEN"], MMA=bool(cfg.mma), SPLIT=p["SPLIT"],
        num_warps=cfg.num_warps, num_stages=cfg.num_stages, **({} if p["SPLIT"] else rk))
    if p["SPLIT"]:
        t9b_splitk_reduce[p["reduce_grid"]](ws, out, M, N, p["NSPLIT"], out.stride(0), freqs, BLOCK_R=REDUCE_BLOCK,
                                            num_warps=4, **rk)
    return out


def linear_fp8(xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The fp8_gemm call of model.py:198-205 as a GEMV: xq [..., K] / xs [..., K/32] from the reference act_quant ->
    a new bf16 [..., N] (kernel.py:299 allocates the same shape in the default dtype, bf16 in this model)."""
    K = xq.size(-1)
    N = w.size(0)
    out = torch.empty(*xq.shape[:-1], N, dtype=torch.bfloat16, device=xq.device)
    mxfp8_gemv(xq.view(-1, K), xs.view(-1, K // BLOCK), w, s, out.view(-1, N))
    return out


def linear_fp8_rope(xq: torch.Tensor, xs: torch.Tensor, w: torch.Tensor, s: torch.Tensor, freqs: torch.Tensor,
                    head_dim: int, rope_dim: int, modes) -> torch.Tensor:
    """T9b-2 #13: linear_fp8 followed by apply_rotary_emb on every head's last rope_dim outputs (model.py:771-772),
    as one GEMV launch (the rotation is in the store). freqs: fp32 [rope_dim/2, 2] (view_as_real of freqs_cis[pos])."""
    K = xq.size(-1)
    N = w.size(0)
    out = torch.empty(*xq.shape[:-1], N, dtype=torch.bfloat16, device=xq.device)
    mxfp8_gemv(xq.view(-1, K), xs.view(-1, K // BLOCK), w, s, out.view(-1, N),
               rope=(freqs, int(head_dim), int(rope_dim), tuple(int(m) for m in modes)))
    return out
