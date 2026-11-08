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
_WORKSPACE: Dict[tuple, torch.Tensor] = {}   # (device, M, N, nsplit[, tag])


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


def pin_m1() -> bool:
    """$DSV41_GEMV_PIN_M1=1 (M1-T10 V0 census): every M > 1 call uses the M = 1 call's config for its (N, K), so the
    6-row GEMV has the 1-row step's numerics_key whenever the form allows it (read at every get_config, i.e. at eager
    calls and captures; a captured graph keeps the config it was captured with)."""
    return os.environ.get("DSV41_GEMV_PIN_M1", "0") not in ("0", "", "false", "off")


def get_config(N: int, K: int, M: int, device=None) -> GemvConfig:
    """The config for (N, K, block_m(M)): this process's choice, else the cache file's, else tuned now (eager only),
    else DEFAULT_CONFIG (split resolved) when tuning is off. Inside a CUDA-graph capture an unknown shape is an error,
    not a tune. With pin_m1() an M > 1 call takes the M = 1 config."""
    if M > 1 and pin_m1():
        return get_config(N, K, 1, device)
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


def workspace(device, M: int, N: int, nsplit: int, tag: str = "") -> torch.Tensor:
    """The static fp32 split-K workspace [nsplit, M, N] for (device, M, N, nsplit): allocated on first (eager) use and
    kept for the life of the process (a captured graph keeps pointing at it). Calls sharing it run in stream order
    (main kernel, then its reduce) so they never overlap. Never allocates inside a capture. T9b-4 `tag`: a GEMV whose
    reduce is folded into a later consumer (`gemv_partials`) keeps its partials alive past other GEMVs of the same
    shape (shared w1 / w3), so it names its own workspace."""
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else (torch.cuda.current_device() if dev.type == "cuda" else 0)
    key = (f"{dev.type}:{idx}", int(M), int(N), int(nsplit)) + ((str(tag),) if tag else ())
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
            "workspaces": {f"{k[0]}/m{k[1]}/n{k[2]}/s{k[3]}" + (f"/{k[4]}" if len(k) > 4 else ""): int(t.numel()) * 4
                           for k, t in _WORKSPACE.items()}}


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
                       ROPE_DIM: tl.constexpr = 0, RE_MODE: tl.constexpr = 0, IM_MODE: tl.constexpr = 0,
                       stride_fm=0, FROWS: tl.constexpr = False):
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
            if FROWS:   # M1-T10-PORT: row m's own position (freqs row m, stride stride_fm)
                fp = freqs + rm_in.to(tl.int64)[:, None] * stride_fm + 2 * fi[None, :]
                c = tl.load(fp)
                d = tl.load(fp + 1)
            else:
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
                          IM_MODE: tl.constexpr = 0, stride_fm=0, FROWS: tl.constexpr = False):
        """out[m, n] = bf16(((0 + ws[0, m, n]) + ws[1, m, n]) + ... + ws[NSPLIT-1, m, n]): fixed split order. ROPE
        (T9b-2 #13): each lane also sums its pair partner n ^ 1 in the same order (so both see the same two values),
        rounds both to bf16 and stores its own component of the rotated pair (columns outside RoPE: unchanged).
        M1-T10-PORT FROWS: row m takes freqs row m (stride stride_fm), its own position."""
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
            if FROWS:
                c = tl.load(freqs + m.to(tl.int64) * stride_fm + 2 * fi, mask=ok, other=0.0)
                d = tl.load(freqs + m.to(tl.int64) * stride_fm + 2 * fi + 1, mask=ok, other=0.0)
            else:
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


def check_rope(rope, N: int, M: Optional[int] = None) -> None:
    """rope = (freqs fp32 [rope_dim/2, 2] contiguous, head_dim, rope_dim, (re_mode, im_mode)) for an output of N.
    M1-T10-PORT: freqs may be [M, rope_dim/2, 2] (row m rotated by its own position; M = the GEMV's rows)."""
    freqs, hd, rd, modes = rope
    if hd < 2 or hd % 2 or rd % 2 or not 0 < rd <= hd or N % hd:
        raise ValueError(f"mxfp8_gemv rope: head_dim {hd} / rope_dim {rd} must be even, 0 < rope_dim <= head_dim, "
                         f"and divide N = {N}")
    shapes = {(rd // 2, 2)} | ({(int(M), rd // 2, 2)} if M is not None else set())
    if freqs.dtype != torch.float32 or tuple(freqs.shape) not in shapes or not freqs.is_contiguous():
        raise ValueError(f"mxfp8_gemv rope: freqs {tuple(freqs.shape)} {freqs.dtype} must be contiguous fp32 "
                         f"[{rd // 2}, 2] (or [M, {rd // 2}, 2])")
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
        check_rope(rope, N, M)
    if not xq.is_cuda:
        sk = resolve_config(config, N, K).split_k if config is not None else max(1, int(CPU_SPLIT_K))
        res = gemv_torch(xq, xs, w, s, out, split_k=sk)
        if rope is not None:
            from runtime.fused_glue import rope_rows_torch

            freqs, hd, rd, modes = rope
            if freqs.dim() == 3:   # M1-T10-PORT: one position per row
                for m in range(M):
                    res[m].copy_(rope_rows_torch(res[m], freqs[m], hd, rd, tuple(modes), inverse=False))
            else:
                res.copy_(rope_rows_torch(res, freqs, hd, rd, tuple(modes), inverse=False))
        return res
    if not HAVE_TRITON:  # pragma: no cover
        raise RuntimeError("mxfp8_gemv on CUDA needs triton")
    cfg = resolve_config(config or get_config(N, K, M, xq.device), N, K, num_sms(xq.device))
    p = launch_plan(cfg, M, N, K)
    ws = workspace(xq.device, M, N, p["NSPLIT"]) if p["SPLIT"] else out
    return _launch(xq, xs, w, s, out, cfg, ws, rope=rope)


def _launch(xq, xs, w, s, out, cfg: GemvConfig, ws, rope=None, reduce: bool = True) -> torch.Tensor:
    """The kernel launch(es) of one call for a resolved config; `ws` is the [NSPLIT, M, N] fp32 workspace when the
    plan splits (else unused). Separate from mxfp8_gemv so tests can drive the kernels through a Triton stand-in.
    rope: as mxfp8_gemv (the epilogue goes to the main kernel's store, or to the reduce when the plan splits).
    reduce=False (T9b-4 `gemv_partials`): a split plan stops after the main kernel; ws holds the partials."""
    M, K = xq.shape
    N = w.size(0)
    p = launch_plan(cfg, M, N, K)
    rk = {}
    freqs = out
    if rope is not None:
        freqs, hd, rd, modes = rope
        rk = dict(ROPE=True, HEAD_DIM=int(hd), ROPE_DIM=int(rd), RE_MODE=int(modes[0]), IM_MODE=int(modes[1]))
        if freqs.dim() == 3:   # M1-T10-PORT: per-row positions
            rk.update(stride_fm=int(freqs.stride(0)), FROWS=True)
    t9b_mxfp8_gemv[p["grid"]](
        xq, xs.view(torch.uint8), w, s.view(torch.uint8), out, ws, M, N, K // BLOCK, p["PER"],
        xq.stride(0), xs.stride(0), w.stride(0), s.stride(0), out.stride(0), freqs,
        BLOCK_M=p["BLOCK_M"], BLOCK_N=cfg.block_n, KU=cfg.ku, EVEN=p["EVEN"], MMA=bool(cfg.mma), SPLIT=p["SPLIT"],
        num_warps=cfg.num_warps, num_stages=cfg.num_stages, **({} if p["SPLIT"] else rk))
    if p["SPLIT"] and reduce:
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


# ------------------------------------------------------------------------------------------------ T9b-4: pair + fold
# docs/HANDOFF-T9B4-RESIDUAL.md lever 2. Two changes, both bit-identical by construction (asserted on the box by
# tests/gpu/test_t9b4_gpu.py and at build by runtime/fused_glue.py's self-check):
#   (a) `gemv_pair`: two weights that read the SAME activation (wq_a || wkv, shared w1 || w3) in ONE launch. The
#       weights stay separate tensors (pointer pair, no copy): program pn < TA works on weight a, else on b. Each
#       half keeps its OWN tuned split partition (PER, NSPLIT) -- the only thing besides the block-partial form (MMA)
#       that fixes a GEMV's bits (`numerics_key`) -- so each half's output bits are its single launch's. The pair is
#       formed only when both halves' tuned configs agree on MMA; BLOCK_N / KU / warps / stages come from half a
#       (they change speed, not bits). Otherwise: two launches.
#   (b) `gemv_partials`: a split plan stops after the main kernel; the result is a `SplitK` handle on the fp32
#       partials ws[NSPLIT, M, N] and the CONSUMER kernel (runtime/fused_glue*.py: q_norm / kv_norm rows, the shared
#       SwiGLU) sums them in t9b_splitk_reduce's fixed order (+0.0, then split 0, 1, ..., NSPLIT-1, fp32 adds) and
#       rounds to bf16: the value the reduce would have stored. A plan without split returns the bf16 output itself.
CPU_SPLIT_K = 1   # CPU (Mac tests) split for gemv_partials / gemv_pair / mxfp8_gemv without a config: 1 = unchanged
KERNELS_ON_CPU = False   # tests set it under tests/triton_shim.py: gemv_partials / gemv_pair launch the kernels


class SplitK:
    """A GEMV output left as its fp32 split-K partials (see the section comment): ws [nsplit, M, N] fp32 contiguous
    (a static workspace), nsplit >= 2. `reduce_torch()` is the reduce's value in torch (CPU mirrors, tests)."""
    __slots__ = ("ws", "nsplit")

    def __init__(self, ws: torch.Tensor, nsplit: int):
        if ws.dtype != torch.float32 or ws.dim() != 3 or int(ws.size(0)) != int(nsplit) or not ws.is_contiguous():
            raise ValueError(f"SplitK: ws {tuple(ws.shape)} {ws.dtype} is not a contiguous fp32 [{nsplit}, M, N]")
        self.ws, self.nsplit = ws, int(nsplit)

    @property
    def M(self) -> int:
        return int(self.ws.size(1))

    @property
    def N(self) -> int:
        return int(self.ws.size(2))

    @property
    def split_stride(self) -> int:
        return self.M * self.N

    @property
    def device(self):
        return self.ws.device

    def reduce_torch(self) -> torch.Tensor:
        """bf16 [M, N] = ((0 + ws[0]) + ws[1]) + ... (t9b_splitk_reduce's order), rounded to nearest even."""
        acc = torch.zeros(self.M, self.N, dtype=torch.float32, device=self.ws.device)
        for j in range(self.nsplit):
            acc = acc + self.ws[j]
        return acc.to(torch.bfloat16)


def materialize(r) -> torch.Tensor:
    """bf16 [M, N] of a gemv_partials result (a bf16 tensor is returned as is; a SplitK runs the reduce: one launch
    on CUDA -- for a consumer that cannot fold it)."""
    if not (hasattr(r, "nsplit") and hasattr(r, "ws")):     # duck-typed (a test's shim-loaded copy of this module)
        return r
    if not r.ws.is_cuda and not (KERNELS_ON_CPU and HAVE_TRITON):
        return r.reduce_torch()
    out = torch.empty(r.M, r.N, dtype=torch.bfloat16, device=r.device)
    t9b_splitk_reduce[(-(-(r.M * r.N) // REDUCE_BLOCK),)](r.ws, out, r.M, r.N, r.nsplit, out.stride(0), out,
                                                          BLOCK_R=REDUCE_BLOCK, num_warps=4)
    return out


def gemv_torch_partials(xq, xs, w, s, split_k: int) -> torch.Tensor:
    """fp32 [NSPLIT, M, N]: split j's K-order chain from +0.0 (gemv_torch_f32 before its split sum). Summing these in
    split order from +0.0 is gemv_torch_f32 exactly (tests/test_t9b4_cpu.py)."""
    M, N, K = check_operands(xq, xs, w, s)
    kb = K // BLOCK
    part = block_partials(xq, w)
    sa = xs.view(torch.uint8).to(torch.int32).view(M, 1, kb)
    rows = weight_scale_rows(torch.arange(N, device=w.device))
    sb = s.view(torch.uint8).to(torch.int32).index_select(0, rows).view(1, N, kb)
    q = part * (pow2_code(sa) * pow2_code(sb))
    per, nsplit = split_plan(kb, split_k)
    out = torch.empty(nsplit, M, N, dtype=torch.float32, device=xq.device)
    for j in range(nsplit):
        acc = torch.zeros(M, N, dtype=torch.float32, device=xq.device)
        for k in range(j * per, min((j + 1) * per, kb)):
            acc = acc + q[:, :, k]
        out[j] = acc
    return out


def _kern_gemv(t: torch.Tensor) -> bool:
    return t.is_cuda or (KERNELS_ON_CPU and HAVE_TRITON)


def _resolved(N: int, K: int, M: int, device, config: Optional[GemvConfig]) -> GemvConfig:
    if config is None:
        if not torch.device(device).type == "cuda":
            return resolve_config(replace(DEFAULT_CONFIG, split_k=max(1, int(CPU_SPLIT_K))), N, K)
        config = get_config(N, K, M, device)
    return resolve_config(config, N, K, num_sms(device) if torch.device(device).type == "cuda" else None)


def _partials_impl(xq, xs, w, s, tag: str, config: Optional[GemvConfig] = None):
    M, N, K = check_operands(xq, xs, w, s)
    cfg = _resolved(N, K, M, xq.device, config)
    p = launch_plan(cfg, M, N, K)
    if not p["SPLIT"]:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
        if _kern_gemv(xq):
            return _launch(xq, xs, w, s, out, cfg, out)
        return gemv_torch(xq, xs, w, s, out, split_k=cfg.split_k)
    ws = workspace(xq.device, M, N, p["NSPLIT"], tag or f"t9b4:{w.data_ptr():x}")
    if _kern_gemv(xq):
        _launch(xq, xs, w, s, ws[0], cfg, ws, reduce=False)
    else:
        ws.copy_(gemv_torch_partials(xq, xs, w, s, cfg.split_k))
    return SplitK(ws, p["NSPLIT"])


def gemv_partials(xq, xs, w, s, tag: str = "", config: Optional[GemvConfig] = None):
    """The GEMV of mxfp8_gemv (same tuned config, same main kernel) WITHOUT its split-K reduce: -> bf16 [M, N] when
    the plan does not split, else SplitK over the static workspace named `tag` (default: per weight). One launch."""
    return _partials_impl(xq, xs, w, s, tag, config)


def pair_plan(cfg_a: GemvConfig, cfg_b: GemvConfig, M: int, NA: int, NB: int, K: int) -> Optional[dict]:
    """Grid / constexprs of the one-launch pair (metadata only), or None when the halves cannot share a launch (their
    block-partial forms differ). Each half keeps its own (PER, NSPLIT); tile width etc. come from cfg_a."""
    if int(cfg_a.mma) != int(cfg_b.mma):
        return None
    kb = K // BLOCK
    pera, nsa = split_plan(kb, cfg_a.split_k)
    perb, nsb = split_plan(kb, cfg_b.split_k)
    bn, ku = int(cfg_a.block_n), int(cfg_a.ku)
    ta, tb = -(-NA // bn), -(-NB // bn)
    return {"grid": (ta + tb, max(nsa, nsb)), "TA": ta, "TB": tb, "PERA": pera, "PERB": perb, "NSA": nsa, "NSB": nsb,
            "EVENA": pera % ku == 0 and kb % pera == 0, "EVENB": perb % ku == 0 and kb % perb == 0,
            "BLOCK_M": kernel_block_m(cfg_a, M), "BLOCK_N": bn, "KU": ku, "MMA": bool(cfg_a.mma),
            "num_warps": int(cfg_a.num_warps), "num_stages": int(cfg_a.num_stages)}


def _launch_pair(xq, xs, wa, sa, wb, sb, pp: dict, outa, wsa, outb, wsb) -> None:
    M = xq.size(0)
    t9b4_mxfp8_gemv2[pp["grid"]](
        xq, xs.view(torch.uint8), wa, sa.view(torch.uint8), outa, wsa, wb, sb.view(torch.uint8), outb, wsb,
        M, wa.size(0), wb.size(0), xq.size(1) // BLOCK, pp["PERA"], pp["PERB"], pp["NSA"], pp["NSB"], pp["TA"],
        xq.stride(0), xs.stride(0), wa.stride(0), sa.stride(0), outa.stride(0), wb.stride(0), sb.stride(0),
        outb.stride(0),
        BLOCK_M=pp["BLOCK_M"], BLOCK_N=pp["BLOCK_N"], KU=pp["KU"], EVENA=pp["EVENA"], EVENB=pp["EVENB"],
        MMA=pp["MMA"], SPLITA=pp["NSA"] > 1, SPLITB=pp["NSB"] > 1, num_warps=pp["num_warps"],
        num_stages=pp["num_stages"])


PAIR_STATS = {"pair": 0, "split": 0}   # host-side counts of gemv_pair calls that took one launch / fell back to two


def gemv_pair(xq, xs, wa, sa, wb, sb, tags: Tuple[str, str] = ("", ""),
              configs: Tuple[Optional[GemvConfig], Optional[GemvConfig]] = (None, None)):
    """gemv_partials(xq, xs, wa, sa) and gemv_partials(xq, xs, wb, sb) -- each half's own tuned config, its bits
    unchanged -- in ONE launch when `pair_plan` allows (else two). -> (result_a, result_b), each bf16 [M, N] or
    SplitK (section comment)."""
    M, NA, K = check_operands(xq, xs, wa, sa)
    _, NB, _ = check_operands(xq, xs, wb, sb)
    cfg_a = _resolved(NA, K, M, xq.device, configs[0])
    cfg_b = _resolved(NB, K, M, xq.device, configs[1])
    pp = pair_plan(cfg_a, cfg_b, M, NA, NB, K) if _kern_gemv(xq) else None
    if pp is None:
        PAIR_STATS["split"] += 1
        return (_partials_impl(xq, xs, wa, sa, tags[0], cfg_a), _partials_impl(xq, xs, wb, sb, tags[1], cfg_b))
    PAIR_STATS["pair"] += 1
    res, bufs = [], []
    for w, ns, tag in ((wa, pp["NSA"], tags[0]), (wb, pp["NSB"], tags[1])):
        N = w.size(0)
        if ns > 1:
            ws = workspace(xq.device, M, N, ns, tag or f"t9b4:{w.data_ptr():x}")
            bufs.append((ws[0], ws))
            res.append(SplitK(ws, ns))
        else:
            out = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
            bufs.append((out, out))
            res.append(out)
    _launch_pair(xq, xs, wa, sa, wb, sb, pp, bufs[0][0], bufs[0][1], bufs[1][0], bufs[1][1])
    return res[0], res[1]


if HAVE_TRITON:

    @triton.jit
    def _t9b4_gemv_tile(xq, xs, w, s, out, ws, M, N, KB, PER, pid_n, ks, stride_xm, stride_xsm, stride_wn,
                        stride_sn, stride_om, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr,
                        EVEN: tl.constexpr, MMA: tl.constexpr, SPLIT: tl.constexpr):
        """t9b_mxfp8_gemv's body for one (pn, ks) tile, line for line (no RoPE epilogue): the same loads, the same
        block partial (tl.dot or broadcast + tl.sum), the same scale product and the same K-order fp32 adds."""
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
        else:
            tl.store(out + rm[:, None] * stride_om + rn[None, :], acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def t9b4_mxfp8_gemv2(xq, xs, wa, sa, outa, wsa, wb, sb, outb, wsb, M, NA, NB, KB, PERA, PERB, NSA, NSB, TA,
                         stride_xm, stride_xsm, stride_wan, stride_san, stride_oam, stride_wbn, stride_sbn,
                         stride_obm, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr,
                         EVENA: tl.constexpr, EVENB: tl.constexpr, MMA: tl.constexpr, SPLITA: tl.constexpr,
                         SPLITB: tl.constexpr):
        """T9b-4 (a): grid (TA + TB, max(NSA, NSB)). Program (pn, ks): pn < TA -> tile pn of weight a over its split
        ks (if ks < NSA), else tile pn - TA of weight b (if ks < NSB). Each half: its own PER / NSPLIT / EVEN."""
        pid_n = tl.program_id(0)
        ks = tl.program_id(1)
        if pid_n < TA:
            if ks < NSA:
                _t9b4_gemv_tile(xq, xs, wa, sa, outa, wsa, M, NA, KB, PERA, pid_n, ks, stride_xm, stride_xsm,
                                stride_wan, stride_san, stride_oam, BLOCK_M, BLOCK_N, KU, EVENA, MMA, SPLITA)
        else:
            if ks < NSB:
                _t9b4_gemv_tile(xq, xs, wb, sb, outb, wsb, M, NB, KB, PERB, pid_n - TA, ks, stride_xm, stride_xsm,
                                stride_wbn, stride_sbn, stride_obm, BLOCK_M, BLOCK_N, KU, EVENB, MMA, SPLITB)


# ------------------------------------------------------------------------------------------------ M2-L10: q/kv tail
# docs/HANDOFF-M2-L10-QKV-TAIL.md, docs/M2-FUSED-CHAIN-DESIGN.md section 2.2 row L10. The wq_a || wkv pair GEMV
# (t9b4_mxfp8_gemv2's work, tile for tile: `_t9b4_gemv_tile`, unchanged) followed IN THE SAME LAUNCH by the two
# consumers that today are their own launches (runtime/fused_glue_r2.py t9b2r2_norm_row, once per output):
#   q   (half a, wq_a):  q_norm over q_lora_rank in the calibrated order -> qr bf16 + act_quant(qr) (q fp8, s e8m0);
#   kv  (half b, wkv):   kv_norm over head_dim -> RoPE on the last rope_dim (bf16) -> act_quant QDQ -> ring[slot[t]].
# Per-output ticket (the router plan's idiom, runtime/static_kernels.py m2_router_rows): every program of half h that
# owns a (tile, split) fences (fence.acq_rel.gpu in every thread), syncs the CTA and takes an acq_rel ticket on
# ticket[h]; the LAST of half h's TA*NSA (TB*NSB) programs fences again and runs half h's tail for all M rows, reading
# the partials in t9b_splitk_reduce's order (from +0.0, split 0, 1, ...; bf16) -- `_l10_load`, which is `_t9b4_ld`
# with L2 (.cg) loads -- and resets ticket[h] to 0 (a graph replay starts from 0). No grid barrier, no spin: nothing
# waits on another CTA, so any grid size is safe.
# Numerics: the GEMV half is the pair kernel's code under the pair kernel's launch options (Triton's default fp
# fusion: the GEMV's T9b-1 numerics); the tails are t9b2r2_norm_row's body (launched there with enable_fp_fusion=False)
# with the two products that feed an add written as explicit `mul.rn.f32` (`v * v` of the statistic, `sum * RN(1/w)`
# of the mean), so no mul+add can contract to an FMA under this kernel's options -- every other op of the body has no
# add fed by a multiply (the RoPE forms are already explicit). Same tree, same fp32 intermediates -> the same bits as
# pair + norm_row + norm_row (tests/test_qkv_tail_cpu.py through the shim; tests/gpu/test_qkv_tail_gpu.py on the box).
# The helpers below marked "verbatim" are copies of runtime/fused_glue_r2.py's (a jit function resolves its callees in
# its own module; tests/test_qkv_tail_cpu.py asserts the sources are identical).
L10_TICKETS = 2   # int32 per launch site: [0] = half a (q), [1] = half b (kv); 0 between launches
# Rows per pass of the tails (speed only -- registers vs serial passes; every value bit-identical, tested): RP rows at
# once in [RP, ...] tiles, M rows in ceil(M / RP) passes. $DSV41_QKV_TAIL_RP overrides; bench/qkv_tail_bench.py --rp
# sweeps it on the box. Default 2 [A]: at the real q tile (T 256 x V 4) a pass holds ~24 fp32 registers per row per
# thread at 4 warps, so 8 rows would spill and raise the whole kernel's register count (fewer GEMV CTAs per SM).
L10_TAIL_RP = int(os.environ.get("DSV41_QKV_TAIL_RP", "2"))

if HAVE_TRITON:
    try:
        from triton.language.extra import libdevice
    except ImportError:  # pragma: no cover - older Triton layouts
        from triton.language.extra.cuda import libdevice

    # --- verbatim copies (runtime/fused_glue_r2.py)
    @triton.jit
    def _t9b2_code(amax, tab):
        """e8m0 code of a block from its amax (a bf16 value as fp32): the reference's own code, by table."""
        idx = (amax.to(tl.int32, bitcast=True) >> 16) & 0xFFFF
        return tl.load(tab + idx).to(tl.int32)

    @triton.jit
    def _t9b2_pow2(code):
        return (code << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _t9b2_ipow2(code):
        return ((254 - code) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _t9b2_q8(v, inv):
        """fp8 e4m3 RNE of clamp(v * 2^-e, -448, 448) (v * inv is exact; the clamp never binds)."""
        y = v * inv
        y = tl.where(y > 448.0, 448.0, y)
        y = tl.where(y < -448.0, -448.0, y)
        return y.to(tl.float8e4nv, fp_downcast_rounding="rtne")

    @triton.jit
    def _t9b2_bf16(v):
        return v.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    @triton.jit
    def _r2_offs(k, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr):
        """Element offsets [T, V] of tile k: (k*T + t)*V + v (vectorized) or (k*V + v)*T + t (U, unrolled)."""
        t = tl.arange(0, T)[:, None]
        v = tl.arange(0, V)[None, :]
        if U:
            o = v * T + t
        else:
            o = t * V + v
        return k * (T * V) + o

    @triton.jit
    def _r2_col(acc, j, V: tl.constexpr):
        """Column j of acc [T, V] (a sum of one value and exact zeros: exact; acc is never -0)."""
        vv = tl.arange(0, V)[None, :]
        return tl.sum(tl.where(vv == j, acc, 0.0), axis=1)

    @triton.jit
    def _r2_halve(s, N: tl.constexpr):
        """[N] -> [N/2]: s[i] + s[i + N/2]."""
        s2 = tl.reshape(s, [2, N // 2])
        r = tl.arange(0, 2)[:, None]
        return tl.sum(tl.where(r == 0, s2, 0.0), axis=0) + tl.sum(tl.where(r == 1, s2, 0.0), axis=0)

    @triton.jit
    def _r2_pairs(s, N: tl.constexpr):
        """[N] -> [N/2]: s[2i] + s[2i + 1]."""
        s2 = tl.reshape(s, [N // 2, 2])
        c = tl.arange(0, 2)[None, :]
        return tl.sum(tl.where(c == 0, s2, 0.0), axis=1) + tl.sum(tl.where(c == 1, s2, 0.0), axis=1)

    @triton.jit
    def _r2_level(s, N: tl.constexpr, DEC: tl.constexpr):
        if N > 32:
            out = _r2_halve(s, N)
        else:
            if DEC:
                out = _r2_halve(s, N)
            else:
                out = _r2_pairs(s, N)
        return out

    @triton.jit
    def _r2_finish(acc, T: tl.constexpr, V: tl.constexpr, DEC: tl.constexpr):
        """acc [T, V] -> the scalar sum: fold the V accumulators left, then the tree over T."""
        s = _r2_col(acc, 0, V)
        for j in tl.static_range(V - 1):
            s = s + _r2_col(acc, j + 1, V)
        if T >= 512:
            s = _r2_level(s, 512, DEC)
        if T >= 256:
            s = _r2_level(s, 256, DEC)
        if T >= 128:
            s = _r2_level(s, 128, DEC)
        if T >= 64:
            s = _r2_level(s, 64, DEC)
        if T >= 32:
            s = _r2_level(s, 32, DEC)
        if T >= 16:
            s = _r2_level(s, 16, DEC)
        if T >= 8:
            s = _r2_level(s, 8, DEC)
        if T >= 4:
            s = _r2_level(s, 4, DEC)
        if T >= 2:
            s = _r2_level(s, 2, DEC)
        return tl.sum(s, axis=0)

    # --- M2-L10 helpers
    @triton.jit
    def _l10_fence(v):
        """fence.acq_rel.gpu in every thread that holds an element of v (int32); returns v (static_kernels._m2_fence,
        verbatim body)."""
        return tl.inline_asm_elementwise("fence.acq_rel.gpu;\n\tmov.b32 $0, $1;", "=r,r", [v], dtype=tl.int32,
                                         is_pure=False, pack=1)

    @triton.jit
    def _l10_load(x, e, m, NS: tl.constexpr, XS):
        """_t9b4_ld's value with L2 loads (.cg: the data was written by other CTAs of this launch): a bf16 load
        (NS == 0), or bf16 of the sum of NS fp32 split partials x[j*XS + e] from +0.0 in split order."""
        if NS == 0:
            v = tl.load(x + e, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
        else:
            acc = tl.zeros(e.shape, dtype=tl.float32)
            for j in tl.static_range(NS):
                acc = acc + tl.load(x + j * XS + e, mask=m, other=0.0, cache_modifier=".cg")
            v = _t9b2_bf16(acc)
        return v

    @triton.jit
    def _l10_fin(ssum, fac, wf, DIV: tl.constexpr):
        """fused_glue_r2._r2_fin with the multiply explicit (the `+ eps` that follows cannot contract into it)."""
        if DIV:
            r = libdevice.div_rn(ssum, wf)
        else:
            r = _t9b2_mul_rn(ssum, fac)
        return r

    # --- verbatim copies (runtime/static_kernels.py, M2-T15: _r2_finish for RP rows at once, the same adds per row)
    @triton.jit
    def _m2_halve_rows(s, RP: tl.constexpr, N: tl.constexpr):
        """[RP, N] -> [RP, N/2]: s[:, i] + s[:, i + N/2] (each side a sum of one value and exact zeros: exact)."""
        s2 = tl.reshape(s, [RP, 2, N // 2])
        r = tl.arange(0, 2)[None, :, None]
        return tl.sum(tl.where(r == 0, s2, 0.0), axis=1) + tl.sum(tl.where(r == 1, s2, 0.0), axis=1)

    @triton.jit
    def _m2_pairs_rows(s, RP: tl.constexpr, N: tl.constexpr):
        """[RP, N] -> [RP, N/2]: s[:, 2i] + s[:, 2i + 1]."""
        s2 = tl.reshape(s, [RP, N // 2, 2])
        c = tl.arange(0, 2)[None, None, :]
        return tl.sum(tl.where(c == 0, s2, 0.0), axis=2) + tl.sum(tl.where(c == 1, s2, 0.0), axis=2)

    @triton.jit
    def _m2_level_rows(s, RP: tl.constexpr, N: tl.constexpr, DEC: tl.constexpr):
        if N > 32:
            out = _m2_halve_rows(s, RP, N)
        else:
            if DEC:
                out = _m2_halve_rows(s, RP, N)
            else:
                out = _m2_pairs_rows(s, RP, N)
        return out

    @triton.jit
    def _m2_finish_rows(acc, RP: tl.constexpr, T: tl.constexpr, V: tl.constexpr, DEC: tl.constexpr):
        """acc [RP, T, V] -> [RP]: fold the V accumulators left, then the tree over T (= _r2_finish per row)."""
        vv = tl.arange(0, V)[None, None, :]
        s = tl.sum(tl.where(vv == 0, acc, 0.0), axis=2)
        for j in tl.static_range(V - 1):
            s = s + tl.sum(tl.where(vv == j + 1, acc, 0.0), axis=2)
        if T >= 512:
            s = _m2_level_rows(s, RP, 512, DEC)
        if T >= 256:
            s = _m2_level_rows(s, RP, 256, DEC)
        if T >= 128:
            s = _m2_level_rows(s, RP, 128, DEC)
        if T >= 64:
            s = _m2_level_rows(s, RP, 64, DEC)
        if T >= 32:
            s = _m2_level_rows(s, RP, 32, DEC)
        if T >= 16:
            s = _m2_level_rows(s, RP, 16, DEC)
        if T >= 8:
            s = _m2_level_rows(s, RP, 8, DEC)
        if T >= 4:
            s = _m2_level_rows(s, RP, 4, DEC)
        if T >= 2:
            s = _m2_level_rows(s, RP, 2, DEC)
        return tl.sum(s, axis=1)

    @triton.jit
    def _l10_norm_rows(x, M, XS, nw, out, slot, q, s, freqs, FS, tab, eps, fac, wf,
                       N: tl.constexpr, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr,
                       DIV: tl.constexpr, HAS_SLOT: tl.constexpr, HAS_Q: tl.constexpr, QDQ: tl.constexpr,
                       RD: tl.constexpr, RE_MODE: tl.constexpr, IM_MODE: tl.constexpr, BB: tl.constexpr,
                       NS: tl.constexpr, RP: tl.constexpr):
        """runtime/fused_glue_r2.t9b2r2_norm_row's arithmetic (QQ on: both tails quantize) for the M token rows, RP
        rows per pass (a leading [RP] row axis, rows >= M masked off; RP >= M: all rows in one pass). Per row it is
        the same op sequence: the loads (_l10_load = _t9b4_ld's value), the statistic's tiles (_r2_offs) and tree
        (_m2_finish_rows = _r2_finish per row), rsqrt, nw * (x * r) -> bf16, the RoPE pairs (row t's freqs, stride
        FS), the act_quant code by table, then q / s (HAS_Q) and qr (not QDQ), or the QDQ'd row into ring row slot[t]
        (QDQ + HAS_SLOT). The two products that feed adds are explicit mul.rn (block comment). RP is speed only
        (registers vs passes): no value depends on which pass a row is in."""
        for r0 in range(0, M, RP):
            _l10_norm_pass(x, r0 + tl.arange(0, RP), M, XS, nw, out, slot, q, s, freqs, FS, tab, eps, fac, wf,
                           N, T, V, U, DEC, DIV, HAS_SLOT, HAS_Q, QDQ, RD, RE_MODE, IM_MODE, BB, NS, RP)

    @triton.jit
    def _l10_norm_pass(x, rr, M, XS, nw, out, slot, q, s, freqs, FS, tab, eps, fac, wf,
                       N: tl.constexpr, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr,
                       DIV: tl.constexpr, HAS_SLOT: tl.constexpr, HAS_Q: tl.constexpr, QDQ: tl.constexpr,
                       RD: tl.constexpr, RE_MODE: tl.constexpr, IM_MODE: tl.constexpr, BB: tl.constexpr,
                       NS: tl.constexpr, RP: tl.constexpr):
        """One pass of _l10_norm_rows over the rows rr [RP] (rows >= M masked off)."""
        rok = rr < M
        acc = tl.zeros([RP, T, V], dtype=tl.float32)
        for k in range((N + T * V - 1) // (T * V)):
            e = _r2_offs(k, T, V, U)
            m3 = rok[:, None, None] & (e < N)[None, :, :]
            v = _l10_load(x, rr[:, None, None] * N + e[None, :, :], m3, NS, XS)
            acc = acc + _t9b2_mul_rn(v, v)
        r = libdevice.rsqrt(_l10_fin(_m2_finish_rows(acc, RP, T, V, DEC), fac, wf, DIV) + eps)    # [RP]
        if HAS_SLOT:
            rbase = tl.load(slot + rr, mask=rok, other=0) * N
        else:
            rbase = rr * N
        for b0 in range(0, N // 32, BB):
            bi = b0 + tl.arange(0, BB)
            bok = bi < N // 32
            ce = bi[:, None] * 32 + 2 * tl.arange(0, 16)[None, :]                                # [BB, 16]
            m = rok[:, None, None] & (bok[:, None] & (ce >= 0))[None, :, :]                      # [RP, BB, 16]
            xo = rr[:, None, None] * N + ce[None, :, :]
            rb = r[:, None, None]
            a = _t9b2_bf16(tl.load(nw + ce, mask=bok[:, None], other=0.0).to(tl.float32)[None, :, :]
                           * (_l10_load(x, xo, m, NS, XS) * rb))
            b = _t9b2_bf16(tl.load(nw + ce + 1, mask=bok[:, None], other=0.0).to(tl.float32)[None, :, :]
                           * (_l10_load(x, xo + 1, m, NS, XS) * rb))
            if RD > 0:
                is_r = (ce >= N - RD)[None, :, :]
                fi = tl.where(is_r, (ce[None, :, :] - (N - RD)) // 2, 0)
                mr = m & is_r
                fo = rr[:, None, None] * FS + 2 * fi
                c = tl.load(freqs + fo, mask=mr, other=0.0)
                d = tl.load(freqs + fo + 1, mask=mr, other=0.0)
                a2 = tl.where(is_r, _t9b2_bf16(_t9b2_rope_re(a, b, c, d, RE_MODE)), a)
                b2 = tl.where(is_r, _t9b2_bf16(_t9b2_rope_im(a, b, c, d, IM_MODE)), b)
            else:
                a2 = a
                b2 = b
            code = _t9b2_code(tl.maximum(tl.max(tl.abs(a2), axis=2), tl.max(tl.abs(b2), axis=2)), tab)   # [RP, BB]
            inv = _t9b2_ipow2(code)[:, :, None]
            qa = _t9b2_q8(a2, inv)
            qb = _t9b2_q8(b2, inv)
            if HAS_Q:
                tl.store(q + xo, qa, mask=m)
                tl.store(q + xo + 1, qb, mask=m)
                tl.store(s + rr[:, None] * (N // 32) + bi[None, :], code.to(tl.uint8),
                         mask=rok[:, None] & bok[None, :])
            oo = rbase[:, None, None] + ce[None, :, :]
            if QDQ:
                sc = _t9b2_pow2(code)[:, :, None]
                tl.store(out + oo, (qa.to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)
                tl.store(out + oo + 1, (qb.to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne"),
                         mask=m)
            else:
                tl.store(out + oo, a2.to(tl.bfloat16), mask=m)
                tl.store(out + oo + 1, b2.to(tl.bfloat16), mask=m)

    @triton.jit
    def l10_qkv_tail(xq, xs, wa, sa, outa, wsa, wb, sb, outb, wsb, M, NA, NB, KB, PERA, PERB, NSA, NSB, TA, TB,
                     stride_xm, stride_xsm, stride_wan, stride_san, stride_oam, stride_wbn, stride_sbn, stride_obm,
                     ticket, tab,
                     qnw, qr, qq, qs, q_eps, q_fac, q_wf,
                     kvnw, ring, slot, freqs, FS, kv_eps, kv_fac, kv_wf,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr, EVENA: tl.constexpr,
                     EVENB: tl.constexpr, MMA: tl.constexpr, SPLITA: tl.constexpr, SPLITB: tl.constexpr,
                     FT: tl.constexpr,
                     QN: tl.constexpr, QT: tl.constexpr, QV: tl.constexpr, QU: tl.constexpr, QDEC: tl.constexpr,
                     QDIV: tl.constexpr, QBB: tl.constexpr, QNS: tl.constexpr,
                     KN: tl.constexpr, KT: tl.constexpr, KV: tl.constexpr, KUN: tl.constexpr, KDEC: tl.constexpr,
                     KDIV: tl.constexpr, KBB: tl.constexpr, KNS: tl.constexpr, RD: tl.constexpr,
                     RE_MODE: tl.constexpr, IM_MODE: tl.constexpr, RP: tl.constexpr, TAILS: tl.constexpr = True):
        """M2-L10 (section comment): grid (TA + TB, max(NSA, NSB)) = t9b4_mxfp8_gemv2's. Program (pn, ks): the pair
        kernel's tile; then the per-output ticket; the last program of half a runs the q tail (q_norm + act_quant ->
        qr / qq / qs), the last of half b the kv tail (kv_norm + RoPE + QDQ -> ring[slot[t]]), each for all M rows
        at once (_l10_norm_rows, RP = next_pow2(M)). QNS / KNS = the half's split count when it splits, else 0 (its
        bf16 output is read). Every arm of a runtime `if` uses its own variable names. TAILS=False (bench phase probe
        only, M2-L10-FIX): the GEMV + tickets without the tails' work -- no tail output is written."""
        pid_n = tl.program_id(0)
        ks = tl.program_id(1)
        if pid_n < TA:
            if ks < NSA:
                _t9b4_gemv_tile(xq, xs, wa, sa, outa, wsa, M, NA, KB, PERA, pid_n, ks, stride_xm, stride_xsm,
                                stride_wan, stride_san, stride_oam, BLOCK_M, BLOCK_N, KU, EVENA, MMA, SPLITA)
                fza = _l10_fence(tl.zeros([FT], dtype=tl.int32))     # this tile's stores before the ticket
                tl.debug_barrier()
                olda = tl.atomic_add(ticket, 1 + tl.sum(fza, axis=0), sem="acq_rel", scope="gpu")
                if olda == TA * NSA - 1:                             # the last of half a: every partial is in
                    fza2 = _l10_fence(tl.zeros([FT], dtype=tl.int32))
                    tl.debug_barrier()
                    za = tl.sum(fza2, axis=0)                        # (0: the loads below depend on the fence)
                    if TAILS:
                        if QNS == 0:
                            _l10_norm_rows(outa + za, M, 0, qnw, qr, qr, qq, qs, qnw, 0, tab, q_eps, q_fac, q_wf,
                                           QN, QT, QV, QU, QDEC, QDIV, False, True, False, 0, 0, 0, QBB, 0, RP)
                        else:
                            _l10_norm_rows(wsa + za, M, M * QN, qnw, qr, qr, qq, qs, qnw, 0, tab, q_eps, q_fac, q_wf,
                                           QN, QT, QV, QU, QDEC, QDIV, False, True, False, 0, 0, 0, QBB, QNS, RP)
                    tl.store(ticket, 0)
        else:
            if ks < NSB:
                _t9b4_gemv_tile(xq, xs, wb, sb, outb, wsb, M, NB, KB, PERB, pid_n - TA, ks, stride_xm, stride_xsm,
                                stride_wbn, stride_sbn, stride_obm, BLOCK_M, BLOCK_N, KU, EVENB, MMA, SPLITB)
                fzb = _l10_fence(tl.zeros([FT], dtype=tl.int32))
                tl.debug_barrier()
                oldb = tl.atomic_add(ticket + 1, 1 + tl.sum(fzb, axis=0), sem="acq_rel", scope="gpu")
                if oldb == TB * NSB - 1:                             # the last of half b
                    fzb2 = _l10_fence(tl.zeros([FT], dtype=tl.int32))
                    tl.debug_barrier()
                    zb = tl.sum(fzb2, axis=0)
                    if TAILS:
                        if KNS == 0:
                            _l10_norm_rows(outb + zb, M, 0, kvnw, ring, slot, kvnw, kvnw, freqs, FS, tab, kv_eps,
                                           kv_fac, kv_wf, KN, KT, KV, KUN, KDEC, KDIV, True, False, True, RD, RE_MODE,
                                           IM_MODE, KBB, 0, RP)
                        else:
                            _l10_norm_rows(wsb + zb, M, M * KN, kvnw, ring, slot, kvnw, kvnw, freqs, FS, tab, kv_eps,
                                           kv_fac, kv_wf, KN, KT, KV, KUN, KDEC, KDIV, True, False, True, RD, RE_MODE,
                                           IM_MODE, KBB, KNS, RP)
                    tl.store(ticket + 1, 0)


# ------------------------------------------------------------------------------------------------ M2-L10-FIX: stage form
# docs/HANDOFF-M2-L10-FIX.md. The round-1 kernel above (form "fold") was 1.5 ms/step SLOWER at 6 rows on the box
# (headline 20.30 -> 21.80, docs/box-m2-round1-20261107.log): its whole tail -- for every row, the split-K fold of NS
# fp32 partials (read twice: statistic, then apply), the norm, RoPE, QDQ -- runs on the ONE last CTA of each half,
# ceil(M / RP) serial passes, where the split path ran it on M CTAs (t9b2r2_norm_row's grid is (R,)) [cause (a) of the
# handoff; (d) "overlapped before" is not the mechanism: static_step._attn issues q_path / q_proj / kv_path in one
# stream, nothing overlapped -- the parallelism the split path had was across rows]. Form "stage" keeps the ONE launch
# and the bits, and moves the serial work off the single CTA:
#   1. dynamic CTA ids: every CTA first takes v = atomic_add(ticket[0], 1). v < NWORK (= the pair grid's G0 x G1
#      programs, v -> (pn, ks) = (v % G0, v // G0), the CUDA linear order of that grid): the pair kernel's tile,
#      `_t9b4_gemv_tile`, unchanged. v >= NWORK: a TAIL CTA (2 x ceil(M / RP) of them: half a's row groups, then
#      half b's).
#   2. per-tile ticket (split halves): the last of a tile's NS split programs folds that tile's [M, BLOCK_N] partials
#      in t9b_splitk_reduce's order (from +0.0, split 0, 1, ...) and stores the bf16 -- `_t9b4_ld`'s value exactly --
#      into a bf16 [M, N] stage buffer (unsplit halves: the tile's bf16 store already is that buffer), then counts
#      the tile done on ticket[1 + h]. The fold is spread over TA (TB) CTAs instead of one.
#   3. tail CTA (half h, rows [j*RP, (j+1)*RP)): waits until ticket[1 + h] == TA (TB) -- every tile of its half
#      folded -- then runs `_l10_norm_pass` (the round-1 tail body, unchanged) on the bf16 stage rows (NS = 0: one
#      bf16 load per element, not NS fp32 ones). RP = 1 (default): one row per CTA, the split path's parallelism.
#   Why the wait cannot deadlock (CUB's decoupled look-back argument): a tail CTA only waits on work items whose ids
#   are smaller than its own, and an id is handed out only to a CTA that is already running, so everything it waits
#   on is resident or done; no CTA ever waits on one that has not started. No grid barrier, no co-residency
#   assumption, any grid size.
#   Reset (a graph replay starts from 0): the last split CTA of a tile zeroes that tile's ticket; the last tail CTA
#   (ticket[3] == number of tails - 1: by then every CTA has taken its id and every counter access of this launch is
#   done) zeroes ticket[0:4]. One ticket buffer serves every layer's launch because launches on it never overlap
#   (stream order); with PDL (M2-L5) a dependent launch must gdc-wait before its first ticket atomic.
# Numerics: the GEMV is the pair kernel's code under its launch options; the fold is fp32 adds only (no product to
# contract); the tail is `_l10_norm_pass` (round-1's, block comment above); bf16(fold) stored and reloaded is the fp32
# value `_t9b4_ld` / `_l10_load` hand the norm. Same bits as pair + norm_row + norm_row, any program order
# (tests/test_qkv_tail_cpu.py, shim; tests/gpu/test_qkv_tail_gpu.py, box).
L10_FORMS = ("stage", "fold")
L10_FORM_DEFAULT = "stage"
L10_FORM_ENV = "DSV41_QKV_TAIL_FORM"
L10_TICKET_WORDS = 256   # int32 words a caller allocates per launch site (either form): fold uses [0:2]; stage uses
#                          [0:4] + one per column tile (4 + TA + TB <= 256: N up to 8064 at BLOCK_N 32)
L10_STAGE_RP = int(os.environ.get("DSV41_QKV_TAIL_STAGE_RP", "1"))   # rows per tail CTA (stage form; speed only)


def qkv_tail_form(form: Optional[str] = None) -> str:
    v = (form or os.environ.get(L10_FORM_ENV) or L10_FORM_DEFAULT).strip()
    if v not in L10_FORMS:
        raise ValueError(f"${L10_FORM_ENV}={v!r}: expected one of {L10_FORMS}")
    return v


if HAVE_TRITON:

    @triton.jit
    def _l10_reduce_tile(ws, st, M, N, pid_n, NS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """Tile pid_n's fold: st[m, n] = bf16 of the sum of the NS fp32 partials ws[j, m, n] from +0.0 in split order
        (t9b_splitk_reduce / _t9b4_ld's value; .cg: written by other CTAs of this launch), m < M, n in the tile."""
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, BLOCK_M)
        msk = (rm < M)[:, None] & (rn < N)[None, :]
        e = rm[:, None] * N + rn[None, :]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in tl.static_range(NS):
            acc = acc + tl.load(ws + j * (M * N) + e, mask=msk, other=0.0, cache_modifier=".cg")
        tl.store(st + e, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=msk)

    @triton.jit
    def _l10_publish(ticket, HSLOT: tl.constexpr, tslot, ws, st, M, N, pid_n, NS: tl.constexpr, SPLIT: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, FT: tl.constexpr):
        """After a (tile, split) program's stores: split halves take the tile's ticket; its last split folds the tile
        into the stage buffer, zeroes the tile ticket and counts the tile on ticket[HSLOT]. Unsplit halves: count the
        tile directly (its bf16 store is the stage buffer)."""
        f1 = _l10_fence(tl.zeros([FT], dtype=tl.int32))                     # this program's stores before the ticket
        tl.debug_barrier()
        if SPLIT:
            told = tl.atomic_add(ticket + tslot, 1 + tl.sum(f1, axis=0), sem="acq_rel", scope="gpu")
            if told == NS - 1:                                              # the tile's last split: all partials in
                f2 = _l10_fence(tl.zeros([FT], dtype=tl.int32))
                tl.debug_barrier()
                z2 = tl.sum(f2, axis=0)
                tl.store(ticket + tslot, 0)
                _l10_reduce_tile(ws + z2, st, M, N, pid_n, NS, BLOCK_M, BLOCK_N)
                f3 = _l10_fence(tl.zeros([FT], dtype=tl.int32))             # the stage stores before the count
                tl.debug_barrier()
                tl.atomic_add(ticket + HSLOT, 1 + tl.sum(f3, axis=0), sem="acq_rel", scope="gpu")
        else:
            tl.atomic_add(ticket + HSLOT, 1 + tl.sum(f1, axis=0), sem="acq_rel", scope="gpu")

    @triton.jit
    def _l10_wait(ticket, HSLOT: tl.constexpr, target, FT: tl.constexpr):
        """Spin until ticket[HSLOT] >= target (acquire), then a fence in every thread; -> 0 for the loads to depend
        on. Only ever waits on smaller dynamic ids (section comment: no deadlock)."""
        seen = tl.atomic_add(ticket + HSLOT, 0, sem="acquire", scope="gpu")
        while seen < target:
            seen = tl.atomic_add(ticket + HSLOT, 0, sem="acquire", scope="gpu")
        tl.debug_barrier()
        fw = _l10_fence(tl.zeros([FT], dtype=tl.int32))
        tl.debug_barrier()
        return tl.sum(fw, axis=0)

    @triton.jit
    def l10_qkv_tail2(xq, xs, wa, sa, outa, wsa, sta, wb, sb, outb, wsb, stb, M, NA, NB, KB, PERA, PERB, NSA, NSB,
                      TA, TB, G0, NWORK, NTA, NTB,
                      stride_xm, stride_xsm, stride_wan, stride_san, stride_oam, stride_wbn, stride_sbn, stride_obm,
                      ticket, tab,
                      qnw, qr, qq, qs, q_eps, q_fac, q_wf,
                      kvnw, ring, slot, freqs, FS, kv_eps, kv_fac, kv_wf,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr, EVENA: tl.constexpr,
                      EVENB: tl.constexpr, MMA: tl.constexpr, SPLITA: tl.constexpr, SPLITB: tl.constexpr,
                      FT: tl.constexpr,
                      QN: tl.constexpr, QT: tl.constexpr, QV: tl.constexpr, QU: tl.constexpr, QDEC: tl.constexpr,
                      QDIV: tl.constexpr, QBB: tl.constexpr, QNS: tl.constexpr,
                      KN: tl.constexpr, KT: tl.constexpr, KV: tl.constexpr, KUN: tl.constexpr, KDEC: tl.constexpr,
                      KDIV: tl.constexpr, KBB: tl.constexpr, KNS: tl.constexpr, RD: tl.constexpr,
                      RE_MODE: tl.constexpr, IM_MODE: tl.constexpr, RP: tl.constexpr, TAILS: tl.constexpr = True):
        """M2-L10-FIX stage form (section comment): grid (NWORK + NTA + NTB,), dynamic ids. sta / stb: the halves'
        bf16 [M, N] stage rows (the bf16 output itself when the half does not split). ticket words: [0] ids, [1] / [2]
        tiles folded of half a / b, [3] tails done, [4 + pn] the split ticket of column tile pn. QNS / KNS: the half's
        split count when it splits (the fold's constexpr), else 1. TAILS=False: bench phase probe (no tail output)."""
        v = tl.atomic_add(ticket, 1, sem="relaxed", scope="gpu")
        if v < NWORK:
            pid_n = v % G0
            ks = v // G0
            if pid_n < TA:
                if ks < NSA:
                    _t9b4_gemv_tile(xq, xs, wa, sa, outa, wsa, M, NA, KB, PERA, pid_n, ks, stride_xm, stride_xsm,
                                    stride_wan, stride_san, stride_oam, BLOCK_M, BLOCK_N, KU, EVENA, MMA, SPLITA)
                    _l10_publish(ticket, 1, 4 + pid_n, wsa, sta, M, NA, pid_n, QNS, SPLITA, BLOCK_M, BLOCK_N, FT)
            else:
                if ks < NSB:
                    _t9b4_gemv_tile(xq, xs, wb, sb, outb, wsb, M, NB, KB, PERB, pid_n - TA, ks, stride_xm,
                                    stride_xsm, stride_wbn, stride_sbn, stride_obm, BLOCK_M, BLOCK_N, KU, EVENB, MMA,
                                    SPLITB)
                    _l10_publish(ticket, 2, 4 + pid_n, wsb, stb, M, NB, pid_n - TA, KNS, SPLITB, BLOCK_M, BLOCK_N,
                                 FT)
        else:
            j = v - NWORK
            if j < NTA:
                zq = _l10_wait(ticket, 1, TA, FT)
                if TAILS:
                    _l10_norm_pass(sta + zq, j * RP + tl.arange(0, RP), M, 0, qnw, qr, qr, qq, qs, qnw, 0, tab,
                                   q_eps, q_fac, q_wf, QN, QT, QV, QU, QDEC, QDIV, False, True, False, 0, 0, 0, QBB, 0,
                                   RP)
            else:
                zk = _l10_wait(ticket, 2, TB, FT)
                if TAILS:
                    _l10_norm_pass(stb + zk, (j - NTA) * RP + tl.arange(0, RP), M, 0, kvnw, ring, slot, kvnw, kvnw,
                                   freqs, FS, tab, kv_eps, kv_fac, kv_wf, KN, KT, KV, KUN, KDEC, KDIV, True, False,
                                   True, RD, RE_MODE, IM_MODE, KBB, 0, RP)
            fd = _l10_fence(tl.zeros([FT], dtype=tl.int32))                 # this tail's stores before the count
            tl.debug_barrier()
            dn = tl.atomic_add(ticket + 3, 1 + tl.sum(fd, axis=0), sem="acq_rel", scope="gpu")
            if dn == NTA + NTB - 1:                                         # the last CTA of the launch to count
                tl.store(ticket, 0)
                tl.store(ticket + 1, 0)
                tl.store(ticket + 2, 0)
                tl.store(ticket + 3, 0)


def qkv_tail_plan(pp: dict, M: int, NA: int, NB: int, rp: Optional[int] = None, form: str = "fold") -> dict:
    """The tail-side constexprs that depend only on the pair plan and M: which halves split (QNS / KNS: the split
    count read in the tail, 0 = the bf16 output), the per-half ticket counts, the fence width, the tails' rows per
    pass RP (min(next_pow2(M), rp or L10_TAIL_RP), a power of two; speed only) and their 32-block tiles (metadata;
    the Mac tests check it). form "stage" (M2-L10-FIX): RP = rows per tail CTA (default L10_STAGE_RP), NTA = NTB =
    ceil(M / RP) tail CTAs per half, NWORK = the pair grid's programs, ticket words needed 4 + TA + TB."""
    out = {"QNS": int(pp["NSA"]) if pp["NSA"] > 1 else 0, "KNS": int(pp["NSB"]) if pp["NSB"] > 1 else 0,
           "tickets": (int(pp["TA"]) * int(pp["NSA"]), int(pp["TB"]) * int(pp["NSB"])),
           "FT": 32 * int(pp["num_warps"]), "RP": min(_np2(M), _np2(max(1, int(rp or L10_TAIL_RP)))),
           "QBB": min(16, _np2(NA // BLOCK)), "KBB": min(16, _np2(NB // BLOCK)), "form": form}
    if form == "stage":
        rps = min(_np2(M), _np2(max(1, int(rp or L10_STAGE_RP))))
        nt = -(-int(M) // rps)
        g0, g1 = pp["grid"]
        out.update(RP=rps, NTA=nt, NTB=nt, NWORK=int(g0) * int(g1), G0=int(g0),
                   grid=(int(g0) * int(g1) + 2 * nt,), ticket_words=4 + int(pp["TA"]) + int(pp["TB"]))
    return out


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


L10_STATS = {"fused": 0, "refused": 0, "stage": 0, "fold": 0}   # host-side counts of qkv_pair_tail calls: one launch
#                                                                  (by form) / refused
L10_COMPILED: Dict[tuple, dict] = {}   # (form, M, RP, TAILS) -> the compiled kernel's registers / spills (eager only)


def kernel_meta(k) -> Optional[dict]:
    """n_regs / n_spills / num_warps / shared of a Triton launch's returned CompiledKernel (None when unavailable:
    the shim, other Triton versions)."""
    if k is None:
        return None
    md = getattr(k, "metadata", None)
    out = {"n_regs": getattr(k, "n_regs", None), "n_spills": getattr(k, "n_spills", None),
           "num_warps": getattr(md, "num_warps", None), "shared": getattr(md, "shared", None)}
    return out if any(v is not None for v in out.values()) else None


def qkv_pair_tail(xq, xs, wa, sa, wb, sb, *, ticket: torch.Tensor, tab: torch.Tensor, q_tail: dict, kv_tail: dict,
                  tags: Tuple[str, str] = ("", ""),
                  configs: Tuple[Optional[GemvConfig], Optional[GemvConfig]] = (None, None),
                  rp: Optional[int] = None, form: Optional[str] = None, tails: bool = True):
    """M2-L10: gemv_pair(xq, xs, wa, sa, wb, sb) and both consumers in ONE launch (section comments). Kernel path only
    (CUDA, or the shim with KERNELS_ON_CPU); the CPU mirror is runtime/fused_glue_r2.launch_qkv_tail's.
      q_tail  = {nw [NA] bf16, qr [M, NA] bf16, q [M, NA] fp8, s [M, NA/32] e8m0, eps, fac, wf, order (T, V, U, DEC,
                 DIV)}
      kv_tail = {nw [NB] bf16, ring (bf16, rows of NB, contiguous), slot [M] int64, freqs fp32 [RD/2, 2] or
                 [M, RD/2, 2] contiguous (None when RD == 0), FS (freqs row stride), rd, modes (re, im), eps, fac, wf,
                 order}
      ticket  = int32 [>= L10_TICKET_WORDS for form stage, >= L10_TICKETS for fold], zero (left zero by every launch).
      form    = "stage" (M2-L10-FIX, default $DSV41_QKV_TAIL_FORM or L10_FORM_DEFAULT) | "fold" (round 1).
      rp      = fold: the single tail CTA's rows per pass (default L10_TAIL_RP); stage: rows per tail CTA (default
                L10_STAGE_RP). Speed only.
      tails   = False: bench phase probe only (the GEMV + tickets, no tail output written).
    The GEMV configs are gemv_pair's exactly (the same `_resolved` + `pair_plan`: tuned config, split-K, warps,
    stages) -> (result_a, result_b) as gemv_pair's (bf16 [M, N] or SplitK, still valid after the launch), or None
    when the halves cannot share a launch (pair_plan refuses: the caller runs the split path)."""
    M, NA, K = check_operands(xq, xs, wa, sa)
    _, NB, _ = check_operands(xq, xs, wb, sb)
    if not _kern_gemv(xq):
        raise RuntimeError("qkv_pair_tail: kernel path only (CUDA or the shim); the CPU mirror is fused_glue_r2's")
    fm = qkv_tail_form(form)
    if NA % BLOCK or NB % BLOCK or ticket.dtype != torch.int32 or ticket.numel() < L10_TICKETS:
        raise ValueError(f"qkv_pair_tail: NA {NA} / NB {NB} must be 32-multiples, ticket int32 [{L10_TICKETS}]")
    cfg_a = _resolved(NA, K, M, xq.device, configs[0])
    cfg_b = _resolved(NB, K, M, xq.device, configs[1])
    pp = pair_plan(cfg_a, cfg_b, M, NA, NB, K)
    if pp is None:
        L10_STATS["refused"] += 1
        return None
    tp = qkv_tail_plan(pp, M, NA, NB, rp, fm)
    if fm == "stage" and ticket.numel() < tp["ticket_words"]:
        raise ValueError(f"qkv_pair_tail: form stage needs {tp['ticket_words']} int32 ticket words, got "
                         f"{ticket.numel()} (allocate L10_TICKET_WORDS)")
    res, bufs = [], []
    for w, ns, tag in ((wa, pp["NSA"], tags[0]), (wb, pp["NSB"], tags[1])):
        N = w.size(0)
        if ns > 1:
            ws = workspace(xq.device, M, N, ns, tag or f"t9b4:{w.data_ptr():x}")
            st = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device) if fm == "stage" else ws
            bufs.append((ws[0], ws, st))
            res.append(SplitK(ws, ns))
        else:
            out = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
            bufs.append((out, out, out))
            res.append(out)
    qt, kt = q_tail, kv_tail
    qo, ko = qt["order"], kt["order"]
    rd = int(kt.get("rd", 0) or 0)
    modes = tuple(kt.get("modes", (0, 0)))
    fr = kt.get("freqs") if rd else None
    L10_STATS["fused"] += 1
    L10_STATS[fm] += 1
    tail_args = (ticket, tab, qt["nw"], qt["qr"], qt["q"], qt["s"].view(torch.uint8), float(qt["eps"]),
                 float(qt["fac"]), float(qt["wf"]), kt["nw"], kt["ring"], kt["slot"],
                 fr if fr is not None else kt["nw"], int(kt.get("FS", 0) or 0), float(kt["eps"]), float(kt["fac"]),
                 float(kt["wf"]))
    strides = (xq.stride(0), xs.stride(0), wa.stride(0), sa.stride(0), bufs[0][0].stride(0), wb.stride(0),
               sb.stride(0), bufs[1][0].stride(0))
    ce = dict(BLOCK_M=pp["BLOCK_M"], BLOCK_N=pp["BLOCK_N"], KU=pp["KU"], EVENA=pp["EVENA"], EVENB=pp["EVENB"],
              MMA=pp["MMA"], SPLITA=pp["NSA"] > 1, SPLITB=pp["NSB"] > 1, FT=tp["FT"],
              QN=NA, QT=int(qo["T"]), QV=int(qo["V"]), QU=bool(qo["U"]), QDEC=bool(qo["DEC"]), QDIV=bool(qo["DIV"]),
              QBB=tp["QBB"], KN=NB, KT=int(ko["T"]), KV=int(ko["V"]), KUN=bool(ko["U"]), KDEC=bool(ko["DEC"]),
              KDIV=bool(ko["DIV"]), KBB=tp["KBB"], RD=rd, RE_MODE=int(modes[0]), IM_MODE=int(modes[1]), RP=tp["RP"],
              TAILS=bool(tails), num_warps=pp["num_warps"], num_stages=pp["num_stages"])
    if fm == "stage":
        k = l10_qkv_tail2[tp["grid"]](
            xq, xs.view(torch.uint8), wa, sa.view(torch.uint8), bufs[0][0], bufs[0][1], bufs[0][2], wb,
            sb.view(torch.uint8), bufs[1][0], bufs[1][1], bufs[1][2], M, NA, NB, K // BLOCK, pp["PERA"], pp["PERB"],
            pp["NSA"], pp["NSB"], pp["TA"], pp["TB"], tp["G0"], tp["NWORK"], tp["NTA"], tp["NTB"], *strides,
            *tail_args, QNS=max(1, tp["QNS"]), KNS=max(1, tp["KNS"]), **ce)
    else:
        k = l10_qkv_tail[pp["grid"]](
            xq, xs.view(torch.uint8), wa, sa.view(torch.uint8), bufs[0][0], bufs[0][1], wb, sb.view(torch.uint8),
            bufs[1][0], bufs[1][1], M, NA, NB, K // BLOCK, pp["PERA"], pp["PERB"], pp["NSA"], pp["NSB"], pp["TA"],
            pp["TB"], *strides, *tail_args, QNS=tp["QNS"], KNS=tp["KNS"], **ce)
    key = (fm, int(M), int(tp["RP"]), bool(tails))
    if key not in L10_COMPILED and not _capturing():
        meta = kernel_meta(k)
        if meta is not None:
            L10_COMPILED[key] = meta
    return res[0], res[1]
