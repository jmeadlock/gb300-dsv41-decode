"""T9b-2 round 2: the medium-risk glue clusters of the 1-row static step (docs/HANDOFF-T9B2-GLUE-R2.md), imported by
runtime/fused_glue.py (`FusedGlue` inherits `R2Glue`). Same contract as round 1: each cluster is a switch, the
build-time self-check picks every rounding order ON THE DEVICE against torch and turns a cluster off when it cannot
match; nothing ships silently non-identical.

  cluster       audit  replaces (per instance)                                                        -> launches
  hc_norm       #1     hc_pre (.float, *, sum, .to) + RMSNorm (.float, square, mean, +eps, rsqrt, *, w*, .to) and
                       the act_quant of the normed row for wq_a + wkv (attn) / shared w1 + w3 + the DeepGEMM rows +
                       the gate's x.float() (ffn) / the head's x.float() (head)                           -> 1
  mix_stat      #5     hc_mixes' .float, square, mean, +eps, rsqrt (in the hc_norm launch; mm, *rsqrt and the
                       sinkhorn kernel stay)                                                               -> 0
  kv_path       #3     kv_norm (8) + RoPE (3) + act_quant QDQ (2) + ring index_copy_ (1)                  -> 1
  q_path        #6     q_norm (8) + act_quant(qr) for wq_b (and for the indexer's wq_b)                   -> 1
  router        #4     /temp (when temp == 1: exact no-op), sqrt, +bias, gather, sum, +eps, /=, *=, sort, index,
                       x.float() and weight.float() (cached once: an exact cast); F.softplus, the mm and torch.topk
                       stay (topk ties: torch's own order)                                                -> 2 (+3 kept)
  engram        #15    the gate math (~26 ops: two rsqrt statistics, the dot, |.|, clamp, sqrt, copysign, the
                       residual add); torch.sigmoid stays                                                 -> 2 (+1 kept)
  indexer_post  #14    sort(topk idx) + where/+offset/.int (torch.topk stays)                             -> 1

The reduction-order question (docs/T9B3-ATTN-DESIGN.md section 3, probes (a)-(c) in docs/flashmla-probe-20261106.json):
  * torch reduces `x.square().mean(-1)` with a fixed tree: T threads x V per-thread accumulators (vectorized loads,
    element (k*T + t)*V + v, or unrolled, (k*V + v)*T + t), each accumulator sequential from +0 over k, the V
    accumulators folded left, then a halving tree over the T threads (shared memory to 32, then the warp; "dec") or
    shared-memory halving then an adjacent-pair warp tree ("inc"); the mean multiplies by RN(1/w) ("mul") or divides
    ("div"). `order_sum_torch` is that family in torch (exact fp32 elementwise adds); the Triton helpers
    `_r2_offs/_r2_finish` compute exactly the same adds. The box probe matched torch on 100 % of 100k rows at every
    normed width with (vec, min(512, last_pow2(w/4)), 4, dec).
  * `pick_order` (build time, eager, on the device; cached per process) evaluates the family on 1,024 random rows at
    the call's width AND row count (torch's launch configuration depends on both: Engram reduces 4 rows at once) and
    takes the first candidate (torch's predicted configuration first) that matches every row: outcome (i)
    bit-identical. None matches -> outcome (ii): the best candidate is recorded (fraction of matching rows, max ulp of
    the fp32 statistic) and the clusters using that width are `approx`: off unless `glue_approx` (--glue-approx).
  * rsqrt: libdevice.rsqrt == torch.rsqrt bit for bit (probe (c), 1.5M values); sqrt / division: sqrt_rn / div_rn
    (torch's sqrtf and `/` are IEEE). transcendental ops whose libdevice form is not proven equal to torch's
    (softplus' exp/log1p, Engram's sigmoid) stay torch ops.
  * the 4-term hc_pre sum: fused_glue.HC_SUM_MODES, picked on the device against Block.hc_pre.
  * topk: never fused. The router post kernel consumes torch.topk's indices as they come (the weight sum runs in
    torch's topk output order, as the reference's); the sorts that follow are of distinct ids (rank by comparison
    count), hence exact.
T9b-4 (docs/HANDOFF-T9B4-RESIDUAL.md; runtime/README.md section T9b-4): `t9b4_hc_norm` (hc_split: the same work over
1 + ceil(D/32 / HC_SPLIT_BB) programs, the statistic recomputed per program in the calibrated order -- the sums are
not partitioned, so no order is re-derived), norm_row reading GEMV split partials (reduce_fold, `_t9b4_ld`), the
wq_a || wkv pair (gemv_pair) feeding the q and kv paths, and packed DeepGEMM activation-SF stores (dg_sf_pack,
`_t9b4_store_code`); `calibrate_r4` checks each against the path it replaces, bitwise, on the device.
M2-T15 (docs/HANDOFF-M2-T15-ROUTER-PLAN.md, runtime/router_plan.py): $DSV41_ROUTER_IMPL=plan | plan-gemv routes `route`
/ `route_rows` (and, at R rows, moe_rows' dedup through a _route hook) to ONE launch per layer (static_kernels
m2_router_plan: softplus, sqrt, +bias, torch.topk's CUDA result incl. its tie order, the weights, the plan), self-checked
at build against this cluster's own path; default `fused` = the router cluster below, unchanged.
T9b-5 (docs/HANDOFF-T9B5-FP32-FIXED.md; section "T9b-5: fixed fp32" below): $DSV41_FP32_IMPL=fixed puts the mHC mixes'
and the head's fp32 F.linear on runtime/static_kernels.py's fixed-order kernels (row-invariant); default cublas.
T9b-7 (docs/HANDOFF-T9B7-GLUE-ORDER.md; section "T9b-7: one order per shape" below): every cluster's reduction order
is a function of the reduction's SHAPE only, never of the step's row count R ($DSV41_GLUE_ORDER, default `shape`).
The orders are picked once at the 1-row reference shapes (what the 1-row build picks, so that build is unchanged) and
the R-row kernels reduce each row alone in that order; at R > 1 the self-checks compare against the reference glue
evaluated one row at a time (`per_row_reductions`: torch's sum / mean of each [1, 1, ...] row slice, the 1-row step's
call). Invariant: 1-row output byte-identical to the pre-T9b-7 1-row step; R-row output equal per row to the 1-row
output. `rows` = the pre-T9b-7 behaviour (orders calibrated at the R-row reference shapes), kept for A/B.
CPU (the Mac tests, the census): the wrappers run torch mirrors; with no device to calibrate on, the statistics run
torch's own op (order "native"), so the CPU step is bit-identical by construction; the explicit-order mirrors and the
Triton bodies (under tests/triton_shim.py) are tested against each other and against the family's definition.
"""
from __future__ import annotations

import contextlib
import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# runtime/fused_glue.py imports this module to define FusedGlue (it inherits R2Glue), so nothing here imports it at
# module level: its constants are restated (tests check they agree) and its torch helpers are reached lazily.
BLOCK = 32                         # act_quant block (fused_glue.BLOCK)
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
HC_SUM_MODES = (0, 1, 2)           # fused_glue.HC_SUM_MODES


def _cpu_default(fn):
    """fused_glue._cpu_default: synthetic-input builders run with a CPU default device (callers move)."""
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)   # the runtime runs under a bf16 default; builders were written for fp32
        try:
            with torch.device("cpu"):
                return fn(*a, **k)
        finally:
            torch.set_default_dtype(prev)
    return w


def act_quant_torch(v, code_tab):
    from runtime.fused_glue import act_quant_torch as f

    return f(v, code_tab)


def pow2_code(code):
    from runtime.fused_glue import pow2_code as f

    return f(code)

try:  # GPU image only; the Mac uses the torch mirrors (and tests/triton_shim.py for the kernel bodies)
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except ImportError:  # pragma: no cover - older layouts
        from triton.language.extra.cuda import libdevice
    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = tl = libdevice = None
    HAVE_TRITON = False

R2_CLUSTERS = ("hc_norm", "mix_stat", "kv_path", "q_path", "router", "engram", "indexer_post")
R2_AUDIT = {"hc_norm": "#1", "mix_stat": "#5", "kv_path": "#3", "q_path": "#6", "router": "#4 (non-topk)",
            "engram": "#15", "indexer_post": "#14 (non-topk)"}
R2_ACT_CLUSTERS = ("hc_norm", "kv_path", "q_path")     # read the act_quant scale table (round 1's self-check gates it)
INDEXER_SCORE_WHY = ("#14's score glue (relu, x weights as a bf16 product, the 32-head sum over a strided dim, "
                     "masks) is not fused: the head sum's reduction order over a non-contiguous dim is not pinned "
                     "(the box probe found it M-dependent) and it only feeds the topk; 8 x ~4 launches left")
NATIVE = "native"                      # host-only order: torch's own reduction op (the CPU mirror)
MAX_T = 512
CAL_ROWS = 1024
LAUNCH = dict(enable_fp_fusion=False)  # no mul+add contraction anywhere (fused_glue.LAUNCH)
NORM_WARPS = 8
KERNELS_ON_CPU = False                 # tests set it under tests/triton_shim.py: wrappers launch the kernels
_ORDER_CACHE: Dict[tuple, dict] = {}   # pick_order results per (device, kind, width, rows): torch's config is fixed


def _kern(t: torch.Tensor) -> bool:
    return t.is_cuda or (KERNELS_ON_CPU and HAVE_TRITON)


# ------------------------------------------------------------------------------------------------ T9b-7: one order per shape
# docs/HANDOFF-T9B7-GLUE-ORDER.md. torch's CUDA reduce picks its thread layout from the number of OUTPUTS as well as
# the reduced width (Reduce.cuh set_block_dimension; predicted_order's `rows`), so "torch's order" for a per-token
# statistic differs between the 1-row step ([1, 1, w]: one output) and the R-row step ([1, R, w]: R outputs) -- the box
# found it for norm_D, mix, q, engram and engram_sum (docs/headline-20261107e.json rowmatch.glue_orders_differ), and
# mix's R-row order is not even in the family (mix_stat off at 6 rows). The fused kernels reduce each row in ONE
# program, so a kernel's per-row result depends only on its order constexprs: give the R-row build the 1-row build's
# orders and its rows equal the 1-row step's, bit for bit (the T12-FIX2 pattern, docs/loop-gpu-test-r3-20261107.log).
#   shape (default): pick_order at the 1-row reference shape (rows 1; Engram: hc_mult rows, the 1-row call's
#                    [1, 1, hc, D]) for every R; at R > 1 the self-checks' oracle is the reference glue with each
#                    torch sum / mean of a [1, R, ...] tensor evaluated on each [1, 1, ...] row slice
#                    (per_row_reductions) -- i.e. "the 1-row reference, row by row".
#   rows:            the pre-T9b-7 protocol (orders of the R-row reference shapes, oracle = the reference at R rows).
GLUE_ORDER_ENV = "DSV41_GLUE_ORDER"
GLUE_ORDERS = ("shape", "rows")
GLUE_ORDER_DEFAULT = "shape"


def glue_order() -> str:
    v = os.environ.get(GLUE_ORDER_ENV, GLUE_ORDER_DEFAULT) or GLUE_ORDER_DEFAULT
    if v not in GLUE_ORDERS:
        raise ValueError(f"{GLUE_ORDER_ENV}={v!r} not in {GLUE_ORDERS}")
    return v


def cal_rows(policy: str, rows: int) -> int:
    """The row count a cluster's order is calibrated at: 1 (shape) or the step's R (rows). Multiply by the rows one
    reference call reduces together per token (Engram: hc_mult)."""
    return 1 if policy == "shape" else max(1, int(rows))


_RED_OWNERS = ((torch.Tensor, "sum"), (torch.Tensor, "mean"), (torch, "sum"), (torch, "mean"))


def _reduced_dims(x: torch.Tensor, args, kw):
    dim = args[0] if args else kw.get("dim")
    ds = tuple(dim) if isinstance(dim, (tuple, list)) else (() if dim is None else (dim,))
    if not ds or not all(isinstance(d, int) and not isinstance(d, bool) for d in ds):
        return None
    return tuple(int(d) % x.dim() for d in ds)


def _per_row_reduce(orig, two_d: bool):
    """sum / mean wrapper. x float [1, s > 1, ...] reduced over dims >= 2 only -> orig on each fresh [1, 1, ...] slice
    (the 1-row call's shape, a fresh aligned allocation), concatenated on dim 1. two_d: also x float [n > 1, K]
    reduced over dim 1 -> orig on each [1, K] row (Gate.forward's [R, k] weight sum). Anything else: orig."""
    def f(x, *args, **kw):
        if isinstance(x, torch.Tensor) and x.is_floating_point() and "out" not in kw:
            ds = _reduced_dims(x, args, kw) if x.dim() >= 2 else None
            if ds is not None:
                if x.dim() >= 3 and x.size(0) == 1 and x.size(1) > 1 and min(ds) >= 2:
                    return torch.cat([orig(x[:, i:i + 1].clone(), *args, **kw) for i in range(x.size(1))], 1)
                if two_d and x.dim() == 2 and x.size(0) > 1 and ds == (1,):
                    return torch.cat([orig(x[i:i + 1].clone(), *args, **kw) for i in range(x.size(0))], 0)
        return orig(x, *args, **kw)

    return f


@contextlib.contextmanager
def per_row_reductions(enabled: bool = True, two_d: bool = False):
    """T9b-7's R-row oracle: torch.sum / torch.mean / Tensor.sum / Tensor.mean evaluated per token row while open
    (_per_row_reduce; process-global, restored on exit; nests). F.linear, einsum and every kernel are untouched: the
    fp32 GEMM sites follow $DSV41_FP32_IMPL at the R-row call shape, as the fused path runs them."""
    if not enabled:
        yield
        return
    saved = []
    for owner, name in _RED_OWNERS:
        saved.append((owner, name, name in vars(owner), getattr(owner, name)))
        setattr(owner, name, _per_row_reduce(getattr(owner, name), two_d))
    try:
        yield
    finally:
        for owner, name, own, fn in reversed(saved):
            if own:
                setattr(owner, name, fn)
            else:
                delattr(owner, name)   # inherited (torch.Tensor.sum lives on torch._C.TensorBase)

# ------------------------------------------------------------------------------------------------ reduction orders
def last_pow2(n: int) -> int:
    return 1 << (max(1, int(n)).bit_length() - 1)


def order_name(order, fin: Optional[str] = None) -> str:
    if order == NATIVE:
        return NATIVE
    fam, T, V, dec = order
    return f"{fam}_T{T}_V{V}_{'dec' if dec else 'inc'}" + (f"|{fin}" if fin else "")


def predicted_order(w: int, rows: int = 1) -> tuple:
    """torch's reduce launch configuration (Reduce.cuh set_block_dimension, as the box probe confirmed at M=1): V = 4
    vectorized accumulators, block width over the reduced dim, block height over the outputs, 512 threads."""
    V = 4 if w >= 4 else 1
    d0 = min(last_pow2(max(1, w // V)), MAX_T)
    d1 = min(last_pow2(rows), MAX_T)
    bw = min(d0, 32)
    bh = min(d1, MAX_T // bw)
    return ("vec", min(d0, MAX_T // bh), V, True)


def order_candidates(w: int, rows: int = 1) -> List[tuple]:
    """The family (module docstring), torch's predicted configuration first, then the rest by distance from it."""
    p = predicted_order(w, rows)
    cands = []
    for fam in ("vec", "unroll"):
        for T in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            for V in (4, 1, 2, 8):
                if T * V > 4 * max(w, 4) or (T * V) * 64 < w:      # no thread idle twice over / no 64-deep chains
                    continue
                for dec in (True, False):
                    if T <= 1 and not dec:
                        continue
                    cands.append((fam, T, V, dec))
    key = lambda o: (o != p, o[0] != "vec", abs(math.log2(o[1]) - math.log2(p[1])), o[2] != 4, not o[3])  # noqa
    return sorted(set(cands), key=key)


def order_sum_torch(vals: torch.Tensor, order) -> torch.Tensor:
    """fp32 [N, w] -> [N]: the sum in `order` (module docstring), every add an exact fp32 elementwise op."""
    N, w = vals.shape
    if order == NATIVE:
        return vals.sum(-1)
    fam, T, V, dec = order
    tile = T * V
    KN = -(-w // tile)
    v = F.pad(vals.float(), (0, KN * tile - w)) if KN * tile != w else vals.float()
    a = v.reshape(N, KN, T, V) if fam == "vec" else v.reshape(N, KN, V, T).transpose(2, 3)
    acc = torch.zeros(N, T, V, dtype=torch.float32, device=vals.device)
    for k in range(KN):
        acc = acc + a[:, k]
    s = acc[..., 0]
    for j in range(1, V):
        s = s + acc[..., j]
    n = T
    while n > 1:
        if dec or n > 32:
            s = s[:, : n // 2] + s[:, n // 2: n]
        else:
            s = s[:, 0:n:2] + s[:, 1:n:2]
        n //= 2
    return s[:, 0]


_F32_CACHE: Dict[tuple, float] = {}


def f32(v: float) -> float:
    """RN_fp32(v) as a Python float (host only: explicit CPU tensor, never the default device; cached)."""
    k = ("f32", float(v))
    if k not in _F32_CACHE:
        _F32_CACHE[k] = float(torch.tensor(float(v), dtype=torch.float32, device="cpu"))
    return _F32_CACHE[k]


def fac_of(w: int) -> float:
    """RN_fp32(1/w) as a Python float (exact fp32 value): torch's mean factor (fp32 1 / fp32 w)."""
    k = ("fac", int(w))
    if k not in _F32_CACHE:
        _F32_CACHE[k] = float(torch.tensor(1.0, dtype=torch.float32, device="cpu")
                              / torch.tensor(float(w), dtype=torch.float32, device="cpu"))
    return _F32_CACHE[k]


def mean_finish(ssum: torch.Tensor, w: int, fin: str) -> torch.Tensor:
    if fin == "div":
        return ssum / torch.full_like(ssum, float(w))   # a tensor divisor: a true division (CUDA scalar div multiplies)
    return ssum * fac_of(w)


def _native(x: torch.Tensor, fn) -> torch.Tensor:
    """torch's own reduction `fn` on the reference call's shape x (the CPU mirror); T9b-7 `shape` policy: on each
    token row of an R-row shape alone (per_row_reductions -- the 1-row call), so the mirror's order is the shape's."""
    with per_row_reductions(glue_order() == "shape", two_d=True):
        return fn(x)


def sqmean_torch(v: torch.Tensor, w: int, ordd: dict, shape=None) -> torch.Tensor:
    """mean(v^2) over the last dim of fp32 v [N, w] in the picked order; NATIVE = torch's op on `shape` (the
    reference call's shape, so its launch configuration is the reference's)."""
    if ordd["order"] == NATIVE:
        x = v.reshape(shape) if shape is not None else v
        return _native(x, lambda t: t.square().mean(-1)).reshape(-1)
    return mean_finish(order_sum_torch(v * v, ordd["order"]), w, ordd["fin"])


def sum_torch(v: torch.Tensor, ordd: dict, shape=None) -> torch.Tensor:
    if ordd["order"] == NATIVE:
        x = v.reshape(shape) if shape is not None else v
        return _native(x, lambda t: t.sum(-1)).reshape(-1)
    return order_sum_torch(v, ordd["order"])


@_cpu_default
def _stat_rows(g, n: int, w: int, kind: str) -> torch.Tensor:
    """Calibration rows [n, w]: N(0,1) per 32-block scaled by 2^U[-8, 8], four outlier channels per row at 2^10 (the
    residual stream's shape), exact zeros. sqmean: bf16 values (every statistic input is a bf16 tensor); sum: signed
    fp32; possum: positive fp32 (the router's sqrt(softplus) scores)."""
    kb = -(-w // 32)
    x = torch.randn(n, kb, 32, generator=g, device="cpu", dtype=torch.float32) * torch.exp2(
        torch.randint(-8, 9, (n, kb, 1), generator=g, device="cpu").float())   # explicit fp32: runtime default dtype is bf16
    x = x.reshape(n, kb * 32)[:, :w].contiguous()
    k = min(4, w)
    ch = torch.randint(0, w, (n, k), generator=g, device="cpu")
    x.scatter_(1, ch, torch.randn(n, k, generator=g, device="cpu", dtype=torch.float32) * 1024.0)
    z = torch.rand(n, w, generator=g, device="cpu") < 0.02
    x = torch.where(z, torch.zeros((), device="cpu"), x)
    if kind == "sqmean":
        return x.to(torch.bfloat16)
    if kind == "possum":
        return x.abs() + 1e-3
    return x


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    ia, ib = a.contiguous().view(torch.int32).long(), b.contiguous().view(torch.int32).long()
    return int((ia - ib).abs().max()) if ia.numel() else 0


def pick_order(dev, w: int, rows: int = 1, kind: str = "sqmean", n: int = CAL_ROWS, seed: int = 0,
               force: bool = False, ref_fn=None) -> dict:
    """Which order of the family equals torch's reduction (`x.square().mean(-1)` for kind sqmean, `x.sum(-1)` for
    sum / possum) of `rows` rows of width `w` reduced together, on `dev`: {"order", "fin", "exact", "match",
    "max_ulp", "candidates", "tried", "rows", "w", "kind"}. On a CPU device without kernels: order NATIVE (the CPU
    mirror calls torch's op), unless `force`. `ref_fn(x_fp32 [rows, w]) -> [rows]` overrides torch's op (tests)."""
    dev = torch.device(dev)
    key = (str(dev), kind, int(w), int(rows), int(n), int(seed))
    if ref_fn is None and not force and key in _ORDER_CACHE:
        return dict(_ORDER_CACHE[key])
    base = {"w": int(w), "rows": int(rows), "kind": kind, "n": int(n)}
    if dev.type != "cuda" and not force and not KERNELS_ON_CPU:
        return dict(base, order=NATIVE, fin="mul", exact=True, match=1.0, max_ulp=0, candidates=0, tried=0,
                    name=NATIVE)
    g = torch.Generator(device="cpu").manual_seed(20261107 + seed + 7919 * w + rows)
    n = n - n % rows
    x = _stat_rows(g, n, w, kind).to(dev)
    v = x.float()
    ref = torch.empty(n, dtype=torch.float32, device=dev)
    shape = (1, 1, w) if rows == 1 else (1, 1, rows, w)
    for i in range(0, n, rows):
        xi = v[i:i + rows].clone()          # a fresh (aligned) allocation, as the model's tensors: torch picks its
        if ref_fn is not None:              # vector width from the pointer's alignment
            ref[i:i + rows] = ref_fn(xi)
        elif kind == "sqmean":
            ref[i:i + rows] = xi.reshape(shape).square().mean(-1).reshape(-1)
        else:
            ref[i:i + rows] = xi.reshape(shape).sum(-1).reshape(-1)
    vals = v * v if kind == "sqmean" else v
    cands = order_candidates(w, rows)
    best = None
    tried = 0
    for o in cands:
        tried += 1
        ssum = order_sum_torch(vals, o)
        for fin in (("mul", "div") if kind == "sqmean" else ("mul",)):
            st = mean_finish(ssum, w, fin) if kind == "sqmean" else ssum
            eq = st.view(torch.int32) == ref.view(torch.int32)
            frac = float(eq.float().mean())
            rec = dict(base, order=o, fin=fin, exact=frac == 1.0, match=frac, max_ulp=_ulp_diff(st, ref),
                       candidates=len(cands), tried=tried, name=order_name(o, fin))
            if best is None or (frac, -rec["max_ulp"]) > (best["match"], -best["max_ulp"]):
                best = dict(rec, best_at=tried)
            if frac == 1.0:
                if ref_fn is None:
                    _ORDER_CACHE[key] = dict(best)
                return best
    best["tried"] = tried                  # outcome (ii): every member was searched; best_at = the best one's rank
    if ref_fn is None:
        _ORDER_CACHE[key] = dict(best)
    return best


def order_args(ordd: dict) -> dict:
    """Kernel constexprs of an explicit order."""
    fam, T, V, dec = ordd["order"]
    return dict(T=int(T), V=int(V), U=fam == "unroll", DEC=bool(dec), DIV=ordd.get("fin") == "div")


# ------------------------------------------------------------------------------------------------ torch mirrors
def hc_pre_torch(x: torch.Tensor, pre: torch.Tensor, mode: int) -> torch.Tensor:
    """Block.hc_pre (model.py:961-962): x [HC, D] (or M1-T10-PORT [R, HC, D]) bf16, pre [HC] ([R, HC]) fp32 ->
    bf16 [D] ([R, D]), the 4-term sum in fused_glue.HC_SUM_MODES order `mode` (0 = ((0 + a0) + a1) + ...)."""
    HC, D = x.shape[-2], x.shape[-1]
    a = [pre[..., i:i + 1].float() * x[..., i, :].float() for i in range(HC)]
    if mode == 0:
        acc = torch.zeros_like(a[0])
        for t in a:
            acc = acc + t
    elif mode == 1:
        acc = (a[0] + a[1]) + (a[2] + a[3])
    else:
        acc = (a[0] + a[2]) + (a[1] + a[3])
    return acc.to(torch.bfloat16)


def rms_rsqrt_torch(v: torch.Tensor, eps: float, ordd: dict, shape=None) -> torch.Tensor:
    """rsqrt(mean(v^2) + eps) of fp32 v [N, w] (model.py:291-292) -> [N, 1]."""
    var = sqmean_torch(v, v.size(-1), ordd, shape)
    return torch.rsqrt(var.view(-1, 1) + eps)


def rms_apply_torch(v: torch.Tensor, r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(weight * (x * r)).to(bf16) (model.py:292-293); v fp32 [N, w], r [N, 1]."""
    return (w * (v * r)).to(torch.bfloat16)


def router_post_torch(idx: torch.Tensor, sc: torch.Tensor, ordd: dict, eps: float, rs: float):
    """Gate.forward after the topk (model.py:825-828) + StaticMoE's ascending sort (static_step.py): idx [K] int64
    (torch.topk's order), sc [E] fp32 -> (ids [K] int64 ascending, w [K] fp32 in that order). M1-T10-PORT: idx
    [R, K], sc [R, E] -> ([R, K], [R, K]) per row (the weight sum over [R, K] as the reference's)."""
    if idx.dim() == 2:
        R_, K = idx.shape
        wv = sc.gather(1, idx)
        den = sum_torch(wv, ordd, (R_, K)).view(R_, 1) + eps
        wn = (wv / den) * rs
        ids, order = torch.sort(idx, dim=1)
        return ids, wn.gather(1, order)
    wv = sc[idx].view(1, -1)
    den = sum_torch(wv, ordd, (1, wv.size(-1))).view(1, 1) + eps
    wn = (wv / den) * rs
    ids, order = torch.sort(idx)
    return ids, wn.view(-1)[order]


def engram_gate_torch(h: torch.Tensor, key: torch.Tensor, wp: torch.Tensor, eps: float, ordd: dict, sordd: dict,
                      c: float, cmin: float) -> torch.Tensor:
    """Engram.forward's gate input (model.py:356-362, before the sigmoid): h, key fp32 [HC, D] (x.float(),
    key.float()), wp fp32 [HC, D] = q_weight.float() * k_weight.float() -> [HC] fp32 copysign(sqrt(clamp(|dot|))).
    M1-T10-PORT: h, key [R, HC, D] -> [R, HC] (the statistics over [1, R, HC, D] as the reference's)."""
    HC, D = wp.shape
    R_ = h.numel() // (HC * D)
    shape = (1, R_, HC, D)
    hh, kk = h.reshape(R_ * HC, D), key.reshape(R_ * HC, D)
    w = wp if R_ == 1 else wp.unsqueeze(0).expand(R_, HC, D).reshape(R_ * HC, D)
    rh = torch.rsqrt(sqmean_torch(hh, D, ordd, shape) + eps)
    rk = torch.rsqrt(sqmean_torch(kk, D, ordd, shape) + eps)
    dot = sum_torch((hh * w) * kk, sordd, shape) * (rh * rk) * c
    out = torch.copysign(dot.abs().clamp_min(cmin).sqrt(), dot)
    return out if R_ == 1 else out.view(R_, HC)


def indexer_post_torch(idx: torch.Tensor, clen: torch.Tensor, offset: int) -> torch.Tensor:
    s = idx.sort(dim=-1).values
    return torch.where(s < clen, s + offset, -1).int()


# ------------------------------------------------------------------------------------------------ T9b-5: fixed fp32
# docs/HANDOFF-T9B5-FP32-FIXED.md. The step's two fp32 F.linear sites -- the mHC mixes (model.py:956, [R, hc*D] x
# [mix_hc, hc*D]^T, 2 per layer) and the head (model.py:1014, [R, D] x [V, D]^T) -- go to cuBLAS, which picks its
# algorithm / split-K by M: they are the only kernels of the 6-row step that are not row-invariant (W4 4a). With
# $DSV41_FP32_IMPL=fixed (`--fp32-impl fixed` on bench/first_light.py and bench/headline.py) both run our own kernel
# (runtime/static_kernels.py t9b5_head_fp32 / t9b5_mix_fp32) in ONE documented order, the same at every R:
#   KB = fixed_kb(site, K) lanes; lane j sums k = j, j + KB, ... sequentially from +0.0 with one fp32 FMA per term;
#   then the adjacent-pair tree s[i] = s[2i] + s[2i+1] over the lanes; the mixes then * rsqrt[r] (one fp32 multiply,
#   the reference's own `* rsqrt`). `fixed_dot_torch` is that definition in torch (exact FMA emulation in fp64).
# Numerics: a defined, reproducible fp32 sum; it differs from cuBLAS's order (1-ulp class vs the cuBLAS reference),
# and is row-invariant and replay-identical by construction. The oracle side adopts the same kernel under the switch
# (the glue's self-check, static_step's reference-op path, the census): policy W4 4d option A.
# The head's operands: x is bf16 (hc_norm's output) and the head weight holds bf16 checkpoint values widened to fp32
# by runtime/loader.py (do_special: boot record widened_bf16_to_fp32 lists head.weight), so every product is exact in
# fp32 and only the sum order is a choice. The kernel reads whatever dtype the weight has in HBM: fp32 by default, or
# bf16 with M2-T17's `head_dtype="bf16"` (widened in registers: the same products, the same sums; section below).
FP32_IMPL_ENV = "DSV41_FP32_IMPL"
FP32_IMPLS = ("cublas", "fixed")
FP32_IMPL_DEFAULT = "cublas"      # until the box gate passes (HANDOFF-T9B5 step 3)
FIXED_KB = {"head": 64, "mix": 512, "comp": 64}   # lanes per output (the order; K tiles of the real model: 80 / 40 / 80)
FIXED_SITES = tuple(FIXED_KB)


def fp32_impl() -> str:
    v = os.environ.get(FP32_IMPL_ENV, FP32_IMPL_DEFAULT) or FP32_IMPL_DEFAULT
    if v not in FP32_IMPLS:
        raise ValueError(f"{FP32_IMPL_ENV}={v!r} not in {FP32_IMPLS}")
    return v


def fp32_fixed() -> bool:
    return fp32_impl() == "fixed"


def fixed_kb(site: str, K: int) -> int:
    """KB lanes of `site` at reduction width K: FIXED_KB[site], or the largest power of two dividing K if smaller
    (mini checkpoints). A function of (site, K) only: never of R."""
    K = int(K)
    return max(1, min(FIXED_KB[site], K & -K))


def fma_f32_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """RN_fp32(a * b + c) with ONE rounding (fp32 fma), exactly: a * b is exact in fp64 (24 + 24 bits); TwoSum gives
    the fp64 sum's error e; the fp64 sum rounds to fp32 like the exact value except at an fp32 midpoint, where e
    decides (tests/triton_shim.fma is the same construction in numpy). Elementwise, broadcasting."""
    a, b, c = torch.broadcast_tensors(a.float(), b.float(), c.float())
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bb = s - p
    e = (p - (s - bb)) + (cd - bb)
    r = s.float()
    rd = r.double()
    up = s > rd
    inf = torch.tensor(float("inf"), dtype=torch.float32, device=r.device)
    n = torch.nextafter(r, torch.where(up, inf, -inf))
    mid = ((rd + n.double()) * 0.5 == s) & (rd != s)
    take = mid & (e != 0) & ((e > 0) == up)
    return torch.where(take, n, r)


def fixed_dot_torch(x: torch.Tensor, w: torch.Tensor, kb: int, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The T9b-5 order (section comment) in torch: x [R, K], w [N, K] (fp32 or bf16 values) -> [R, N] fp32 on x's
    device. Every op is an exact elementwise fp32 / fp64 op, so CPU and CUDA give the same bits."""
    R, K = x.shape
    N = w.size(0)
    assert w.size(1) == K and K % kb == 0 and kb & (kb - 1) == 0, (tuple(x.shape), tuple(w.shape), kb)
    acc = torch.zeros(R, N, kb, dtype=torch.float32, device=x.device)
    for t in range(K // kb):
        xs = x[:, t * kb:(t + 1) * kb].float()
        ws = w[:, t * kb:(t + 1) * kb].float()
        acc = fma_f32_torch(xs[:, None, :], ws[None, :, :], acc)
    s = acc
    while s.size(-1) > 1:
        s = s[..., 0::2] + s[..., 1::2]
    out = s[..., 0]
    if scale is not None:
        out = out * scale.reshape(R, 1).float()
    return out


def launch_fixed_dot(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, *, site: str,
                     scale: Optional[torch.Tensor] = None) -> None:
    """out [R, N] fp32 <- x [R, K] . w^T (* scale [R]) in the T9b-5 order: the Triton kernel on CUDA (one launch,
    t9b5_<site>_fp32), the torch definition on CPU."""
    kb = fixed_kb(site, x.size(-1))
    if x.is_cuda:
        from runtime.static_kernels import fixed_dot

        fixed_dot(x, w, out, kb=kb, site=site, scale=scale)
    else:
        out.copy_(fixed_dot_torch(x, w, kb, scale))


def hc_mixes_fixed(R, layer, x: torch.Tensor, hc_fn, hc_scale, hc_base):
    """Block.hc_mixes (model.py:950-957) with its F.linear in the T9b-5 order (everything else the reference's own
    ops): the oracle / reference-op form of the fixed mixes. R = the reference module (hc_split_sinkhorn)."""
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.norm_eps)
    T_ = xf.numel() // xf.size(-1)
    mixes = torch.empty(*xf.shape[:-1], hc_fn.size(0), dtype=torch.float32, device=xf.device)
    launch_fixed_dot(xf.reshape(T_, xf.size(-1)), hc_fn, mixes.view(T_, -1), site="mix",
                     scale=rsqrt.reshape(T_).contiguous())
    return R.hc_split_sinkhorn(mixes, hc_scale, hc_base, layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps)


def head_fixed_w(W: torch.Tensor, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
    """ParallelHead.forward (model.py:1010-1014, world_size 1) with F.linear(x.float(), W) in the T9b-5 order:
    x [b, s, D] -> [b, V] (last position) or [b, s, V] (full_logits). W is fp32 or bf16 (M2-T17: a bf16 weight is
    widened in registers by the kernel -- exact, so the same fp32 products and the same sums as the widened copy)."""
    if not full_logits:
        x = x[:, -1]
    D = x.size(-1)
    lead = tuple(x.shape[:-1])
    xr = x.reshape(-1, D)
    out = torch.empty(xr.size(0), W.size(0), dtype=torch.float32, device=x.device)
    launch_fixed_dot(xr, W, out, site="head")
    return out.view(*lead, W.size(0))


def head_fixed(model, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
    """head_fixed_w on the backbone's head (model.head.weight, fp32- or bf16-stored)."""
    return head_fixed_w(model.head.weight, x, full_logits)


# ------------------------------------------------------------------------------------------------ T9b-8: compressor
# The ratio-2 attention compressor's projections, model.py:463-465: `x = x.float(); kv, score = self.wkv(x),
# self.wgate(x)` with wkv / wgate declared fp32 at ratio > 1 (model.py:444-448; runtime/loader.py widens the
# checkpoint's bf16 values, boot record `widened`), i.e. cuBLAS F.linear(fp32, fp32) -- M-dependent, the last site of
# the 6-row step that is not row-invariant (docs/rowdiv-shell-20261107.log). Under $DSV41_FP32_IMPL=fixed every DECODE
# call site of the static step (StaticStep 1-row and R-row, dspark_loop.VerifyStep's deferred-commit form) evaluates
# them in the T9b-5 order, site "comp" (KB = FIXED_KB["comp"]; static_kernels.t9b8_comp_fp32 on CUDA). x is the
# reference's own x.float() (bf16 values) and the weight is read in its HBM dtype (fp32; a bf16 weight would be widened
# in registers, exact), so the products are the reference's and only the sum order changes -- vs cuBLAS that is the
# 1-ulp class of T9b-5, and the 1-row step's latents therefore move with the switch exactly as the head / mixes did.
# The ratio-1 compressor (bf16 F.linear, row-invariant), the prefill and the eager reference are untouched; under
# `cublas` (the default) the call sites make the reference's own calls.
def compressor_fixed(comp) -> bool:
    """True when `comp`'s projections take the fixed order: $DSV41_FP32_IMPL=fixed and compress ratio > 1."""
    return fp32_fixed() and int(getattr(comp, "compress_ratio", 0) or 0) > 1


def compressor_proj_fixed(comp, xf: torch.Tensor):
    """(wkv(xf), wgate(xf)) of a ratio > 1 Compressor in the T9b-5 order: xf [b, s, D] fp32 (the reference's x.float())
    -> (kv, score), each [b, s, head_dim] fp32. Two launches (one per weight), every row in the same order."""
    D = xf.size(-1)
    lead = tuple(xf.shape[:-1])
    xr = xf.reshape(-1, D)
    if not xr.is_contiguous():
        xr = xr.contiguous()
    outs = []
    for lin in (comp.wkv, comp.wgate):
        w = lin.weight
        if w.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"compressor_proj_fixed: weight dtype {w.dtype} (model.py:444-448 declares fp32)")
        o = torch.empty(xr.size(0), w.size(0), dtype=torch.float32, device=xf.device)
        launch_fixed_dot(xr, w, o, site="comp")
        outs.append(o.view(*lead, w.size(0)))
    return outs[0], outs[1]


def compressor_decode_fixed(comp, x: torch.Tensor, start_pos: int):
    """Compressor.forward (model.py:458-485) at decode (start_pos > 0, one token per call) with the projections of
    compressor_proj_fixed; every other op is the reference's own, in its order (slot write 478-480, the pooling
    expression 482 on the [bsz, ratio, d] state, `.to(dtype)` and norm 485). Returns the latent or None."""
    if int(start_pos) <= 0:
        raise ValueError("compressor_decode_fixed: decode only (start_pos > 0; the prefill branch stays reference)")
    bsz = x.size(0)
    ratio, dtype = comp.compress_ratio, x.dtype
    kv, score = compressor_proj_fixed(comp, x.float())                                  # 463-465
    should_compress = (start_pos + 1) % ratio == 0                                      # 477
    slot = start_pos % ratio
    comp.kv_state[:bsz, slot] = kv.squeeze(1)
    comp.score_state[:bsz, slot] = score.squeeze(1)
    if should_compress:
        kv = (comp.kv_state[:bsz] * comp.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
    if not should_compress:
        return None
    return comp.norm(kv.to(dtype))


# ------------------------------------------------------------------------------------------------ M2-T17: bf16 head
# docs/HANDOFF-M2-T17-BF16-HEAD.md. The checkpoint stores the head (and the draft's Markov head) in bf16; model.py:1008
# declares the parameter fp32 only "so the logits come out in fp32 directly", and the loader widened it (2.65 GB read
# per step). With `head_dtype="bf16"` (runtime/loader.py; $DSV41_HEAD_DTYPE) the weight stays bf16 in HBM (1.32 GB) and
# every use goes through the fixed-order kernel, which widens each bf16 tile in registers: bf16 -> fp32 is exact, the
# product of two bf16-valued fp32 numbers is exact, and the order is the same, so the logits are bit-equal to the
# fp32-stored T9b-5 kernel's. cuBLAS cannot take a bf16 weight with an fp32 x (F.linear raises), so a bf16 head is a
# fixed-order head by definition -- independent of $DSV41_FP32_IMPL, which keeps governing the mHC mixes.
def head_weight_is_bf16(head_module) -> bool:
    w = getattr(head_module, "weight", None)
    return w is not None and w.dtype == torch.bfloat16


def head_uses_fixed(model) -> bool:
    """True when the backbone's head runs the fixed-order kernel: $DSV41_FP32_IMPL=fixed, or a bf16-stored weight."""
    return fp32_fixed() or head_weight_is_bf16(model.head)


def install_bf16_head(head_module) -> None:
    """Replace ParallelHead.forward (an instance attribute; world_size 1) on a module whose weight is bf16 by the
    fixed-order bf16 read: forward(x, full_logits=False) == F.linear(x.float(), weight.float()) up to the documented
    sum order. Every user of the module -- the static step, the reference oracle (`model.head(...)`), the draft's
    forward_head, the Markov head's calls -- therefore evaluates the same arithmetic."""
    if not head_weight_is_bf16(head_module):
        raise ValueError("install_bf16_head: the head weight is not bf16 "
                         f"({getattr(getattr(head_module, 'weight', None), 'dtype', None)})")

    orig_cls_forward = type(head_module).forward   # captured at install (the load, outside any oracle patch)

    def forward(x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if type(head_module).forward is not orig_cls_forward and x.dim() == 3 and x.size(1) > 1:
            # runtime/ref_multirow.py patches ParallelHead.forward at call time (model.py:1010-1014: the logits of every
            # row when the step has more than one); an instance attribute shadows that patch, so its rule is applied here
            full_logits = True
        return head_fixed_w(head_module.weight, x, full_logits)

    forward.__qualname__ = "ParallelHead.forward[bf16 fixed-order, M2-T17]"
    head_module.forward = forward
    head_module._m2t17_bf16_head = True


def describe_fp32() -> dict:
    return {"impl": fp32_impl(), "env": FP32_IMPL_ENV, "kb": dict(FIXED_KB),
            "order": "KB lanes, sequential fp32 FMA over K tiles from +0.0, adjacent-pair tree over the lanes "
                     "(+ one fp32 multiply by rsqrt for the mixes)"}


def fp32_fixed_census(model, rows: int = 6, seed: int = 0) -> List[dict]:
    """runtime/row_census.py-format records for the two T9b-5 sites (layer 0's mixes weight, the real head): the
    fixed kernel at `rows` rows vs each row alone, next to cuBLAS F.linear on the SAME inputs (the evidence that
    cuBLAS is what breaks the census). Eager; syncs."""
    from runtime.row_census import compare

    dev = model.embed.weight.device
    g = torch.Generator(device="cpu").manual_seed(20261108 + seed)
    L0 = model.layers[0]
    hc = L0.hc_mult
    D = model.embed.weight.size(1)
    out: List[dict] = []

    def rec(family, op, shape, fn):
        r = {"family": family, "op": op, "shape": shape, "equal": None, "note": ""}
        try:
            with torch.inference_mode():
                cmp_, note = fn()
            r.update(cmp_)
            r["note"] = note
        except Exception as e:  # recorded, never fatal (the census convention)
            r["note"] = f"n/a: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        out.append(r)
        return r

    def fixed(x, w, site, scale=None):
        o = torch.empty(x.size(0), w.size(0), dtype=torch.float32, device=x.device)
        launch_fixed_dot(x, w, o, site=site, scale=scale)
        return o

    xm = (torch.randn(rows, hc * D, generator=g, device="cpu").to(torch.bfloat16).float()).to(dev)
    sm = (torch.rand(rows, generator=g, device="cpu") + 0.5).float().to(dev)
    xh = (torch.randn(rows, D, generator=g, device="cpu")).to(torch.bfloat16).to(dev)
    W = model.head.weight
    for fam, site, x, w, sc, nm in (("T9b-5 fixed fp32", "mix", xm, L0.hc_attn_fn, sm, "mHC mixes (hc_attn_fn)"),
                                    ("T9b-5 fixed fp32", "head", xh, W, None, "head")):
        shape = f"[{rows},{x.size(1)}]x{tuple(w.shape)} KB {fixed_kb(site, x.size(1))}"

        def f_fixed(x=x, w=w, sc=sc, site=site):
            full = fixed(x, w, site, sc)
            parts = [fixed(x[i:i + 1], w, site, sc[i:i + 1] if sc is not None else None) for i in range(rows)]
            again = fixed(x, w, site, sc)
            return compare(full, parts), f"replay equal {bool(torch.equal(full, again))}"

        def f_cublas(x=x, w=w, sc=sc):
            w = w.float() if w.dtype == torch.bfloat16 else w   # M2-T17: a bf16-stored head, widened (transient copy)
            mul = (lambda t, s_: t * s_.view(-1, 1)) if sc is not None else (lambda t, s_: t)
            full = mul(F.linear(x.float(), w), sc)
            parts = [mul(F.linear(x[i:i + 1].float(), w), sc[i:i + 1] if sc is not None else None)
                     for i in range(rows)]
            return compare(full, parts), "same inputs as the fixed row"
        rec(fam, nm, shape, f_fixed)
        rec("cuBLAS fp32 F.linear (same inputs)", nm, shape, f_cublas)
    return out


# ------------------------------------------------------------------------------------------------ Triton kernels
if HAVE_TRITON:

    # --- verbatim copies of runtime/fused_glue.py's helpers (a jit function resolves its callees in its own module;
    #     tests/test_fused_glue_r2_cpu.py checks the source is identical)
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
    def _t9b4_ld(x, e, m, NS: tl.constexpr, XS):
        """fp32 value of x[e]: a bf16 load (NS == 0), or (T9b-4 reduce fold) the bf16 rounding of the fixed-order sum
        of NS fp32 split partials x[j*XS + e], j = 0..NS-1, from +0.0 -- t9b_splitk_reduce's value."""
        if NS == 0:
            v = tl.load(x + e, mask=m, other=0.0).to(tl.float32)
        else:
            acc = tl.zeros(e.shape, dtype=tl.float32)
            for j in tl.static_range(NS):
                acc = acc + tl.load(x + j * XS + e, mask=m, other=0.0)
            v = _t9b2_bf16(acc)
        return v

    @triton.jit
    def _t9b4_store_code(s, row, bi, code, bok, s_gs, s_cs, PACK: tl.constexpr):
        """T9b-4 lever 4: the e8m0 code of 32-block bi of output row `row` into DeepGEMM's packed MN-major activation
        SF (uint8 view of int32 storage [K/128][M]): byte row*s_gs + (bi // 4)*s_cs + bi % 4 (PACK), else the fp32
        scale 2^(code-127) at s[row*s_gs + bi]."""
        if PACK:
            tl.store(s + row * s_gs + (bi >> 2) * s_cs + (bi & 3), code.to(tl.uint8), mask=bok)
        else:
            tl.store(s + row * s_gs + bi, _t9b2_pow2(code), mask=bok)

    # --- the reduction family (module docstring): tile offsets, the V fold, the T tree
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

    @triton.jit
    def _r2_fin(ssum, fac, wf, DIV: tl.constexpr):
        """mean = sum * RN(1/w) or sum / w (IEEE)."""
        if DIV:
            r = libdevice.div_rn(ssum, wf)
        else:
            r = ssum * fac
        return r

    @triton.jit
    def _r2_hcpre(x, pre, e, m, D: tl.constexpr, HC: tl.constexpr, MODE: tl.constexpr):
        """bf16(sum_i pre[i] * x[i, e]) as fp32 (model.py:961-962), the sum in HC_SUM_MODES order MODE."""
        if MODE == 0:
            acc = tl.zeros(e.shape, dtype=tl.float32)
            for i in tl.static_range(HC):
                acc = acc + tl.load(pre + i) * tl.load(x + i * D + e, mask=m, other=0.0).to(tl.float32)
        else:
            a0 = tl.load(pre + 0) * tl.load(x + 0 * D + e, mask=m, other=0.0).to(tl.float32)
            a1 = tl.load(pre + 1) * tl.load(x + 1 * D + e, mask=m, other=0.0).to(tl.float32)
            a2 = tl.load(pre + 2) * tl.load(x + 2 * D + e, mask=m, other=0.0).to(tl.float32)
            a3 = tl.load(pre + 3) * tl.load(x + 3 * D + e, mask=m, other=0.0).to(tl.float32)
            if MODE == 1:
                acc = (a0 + a1) + (a2 + a3)
            else:
                acc = (a0 + a2) + (a1 + a3)
        return _t9b2_bf16(acc)

    # --- #1 + #5
    @triton.jit
    def t9b2r2_hc_norm(x, pre, nw, y, yf, xf, mrs, q, s, dq, ds, dq_gs, ds_gs, ds_cs, G, tab,
                       eps, fac, wf, meps, mfac, mwf,
                       D: tl.constexpr, HC: tl.constexpr, HCMODE: tl.constexpr,
                       NT: tl.constexpr, NV: tl.constexpr, NU: tl.constexpr, NDEC: tl.constexpr, NDIV: tl.constexpr,
                       MT: tl.constexpr, MV: tl.constexpr, MU: tl.constexpr, MDEC: tl.constexpr, MDIV: tl.constexpr,
                       PASS_A: tl.constexpr, HAS_MIX: tl.constexpr, HAS_XF: tl.constexpr, HAS_YF: tl.constexpr,
                       QQ: tl.constexpr, HAS_Q: tl.constexpr, HAS_DG: tl.constexpr, BB: tl.constexpr,
                       DG_PACK: tl.constexpr = False):
        """One program. x [HC, D] bf16 (the residual stream copies), pre [HC] fp32 (pre_mix), nw [D] (norm weight).
        A (PASS_A): xf = x.flatten().float() (HAS_XF) and mrs = rsqrt(mean(xf^2) + meps) in order M* (HAS_MIX:
        hc_mixes' statistic, model.py:954-955). B: y0 = hc_pre(x, pre) (bf16), var = mean(y0^2) in order N*,
        r = rsqrt(var + eps). C, per 32-block: y = bf16(nw * (y0 * r)) -> y (bf16), yf (fp32, HAS_YF), and (QQ) its
        act_quant: q fp8 + s e8m0 code (HAS_Q), the same q into rows g*dq_gs of dq and fp32 scales into ds (HAS_DG:
        DeepGEMM's w1/w3 input, G groups; T9b-4 DG_PACK: e8m0 codes into DeepGEMM's packed SF, _t9b4_store_code).
        T9b-4: t9b4_hc_norm is the same work over several programs (kept here for the fallback / A-B).
        M1-T10-PORT: grid axis 1 = token row t (1 at the one-row step): x / pre / y / yf / xf / mrs / q / s are row t
        of [R, ...] buffers (the statistic is per row, in the order calibrated at R rows)."""
        t = tl.program_id(1)
        x = x + t * (HC * D)
        pre = pre + t * HC
        y = y + t * D
        yf = yf + t * D
        xf = xf + t * (HC * D)
        mrs = mrs + t
        q = q + t * D
        s = s + t * (D // 32)
        if PASS_A:
            acc = tl.zeros([MT, MV], dtype=tl.float32)
            for k in range((HC * D + MT * MV - 1) // (MT * MV)):
                e = _r2_offs(k, MT, MV, MU)
                m = e < HC * D
                v = tl.load(x + e, mask=m, other=0.0).to(tl.float32)
                if HAS_XF:
                    tl.store(xf + e, v, mask=m)
                acc = acc + v * v
            if HAS_MIX:
                ms = _r2_fin(_r2_finish(acc, MT, MV, MDEC), mfac, mwf, MDIV)
                tl.store(mrs, libdevice.rsqrt(ms + meps))
        acc2 = tl.zeros([NT, NV], dtype=tl.float32)
        for k in range((D + NT * NV - 1) // (NT * NV)):
            e = _r2_offs(k, NT, NV, NU)
            m = e < D
            yv = _r2_hcpre(x, pre, e, m, D, HC, HCMODE)
            acc2 = acc2 + yv * yv
        r = libdevice.rsqrt(_r2_fin(_r2_finish(acc2, NT, NV, NDEC), fac, wf, NDIV) + eps)
        for b0 in range(0, D // 32, BB):
            bi = b0 + tl.arange(0, BB)
            bok = bi < D // 32
            cols = bi[:, None] * 32 + tl.arange(0, 32)[None, :]
            m = bok[:, None] & (cols >= 0)
            yv = _r2_hcpre(x, pre, cols, m, D, HC, HCMODE)
            o = _t9b2_bf16(tl.load(nw + cols, mask=m, other=0.0).to(tl.float32) * (yv * r))
            tl.store(y + cols, o.to(tl.bfloat16), mask=m)
            if HAS_YF:
                tl.store(yf + cols, o, mask=m)
            if QQ:
                code = _t9b2_code(tl.max(tl.abs(o), axis=1), tab)
                qv = _t9b2_q8(o, _t9b2_ipow2(code)[:, None])
                if HAS_Q:
                    tl.store(q + cols, qv, mask=m)
                    tl.store(s + bi, code.to(tl.uint8), mask=bok)
                if HAS_DG:
                    for gi in range(G):
                        tl.store(dq + gi * dq_gs + cols, qv, mask=m)
                        _t9b4_store_code(ds, gi, bi, code, bok, ds_gs, ds_cs, DG_PACK)

    # --- T9b-4 lever 1: the hc_norm work over 1 + ceil(D/32 / BB) programs (docs/HANDOFF-T9B4-RESIDUAL.md)
    @triton.jit
    def _t9b4_hc_mix(x, xf, mrs, meps, mfac, mwf, D: tl.constexpr, HC: tl.constexpr, MT: tl.constexpr,
                     MV: tl.constexpr, MU: tl.constexpr, MDEC: tl.constexpr, MDIV: tl.constexpr, HAS_MIX: tl.constexpr,
                     HAS_XF: tl.constexpr):
        """t9b2r2_hc_norm's pass A with its tile loop unrolled (static_range: the loads are independent, the adds
        keep their order -- no reassociation without fast-math), same tiles, same tree."""
        acc = tl.zeros([MT, MV], dtype=tl.float32)
        for k in tl.static_range((HC * D + MT * MV - 1) // (MT * MV)):
            e = _r2_offs(k, MT, MV, MU)
            m = e < HC * D
            v = tl.load(x + e, mask=m, other=0.0).to(tl.float32)
            if HAS_XF:
                tl.store(xf + e, v, mask=m)
            acc = acc + v * v
        if HAS_MIX:
            ms = _r2_fin(_r2_finish(acc, MT, MV, MDEC), mfac, mwf, MDIV)
            tl.store(mrs, libdevice.rsqrt(ms + meps))

    @triton.jit
    def _t9b4_hc_apply(x, pre, nw, y, yf, q, s, dq, ds, dq_gs, ds_gs, ds_cs, G, tab, eps, fac, wf, b0,
                       D: tl.constexpr, HC: tl.constexpr, HCMODE: tl.constexpr, NT: tl.constexpr, NV: tl.constexpr,
                       NU: tl.constexpr, NDEC: tl.constexpr, NDIV: tl.constexpr, HAS_YF: tl.constexpr,
                       QQ: tl.constexpr, HAS_Q: tl.constexpr, HAS_DG: tl.constexpr, DG_PACK: tl.constexpr,
                       BB: tl.constexpr):
        """Pass B -- the WHOLE statistic, exactly t9b2r2_hc_norm's (same tiles, same order; every program computes
        the same r bit for bit, no cross-program combine) -- then pass C for the 32-blocks [b0, b0 + BB) only."""
        acc2 = tl.zeros([NT, NV], dtype=tl.float32)
        for k in tl.static_range((D + NT * NV - 1) // (NT * NV)):
            e = _r2_offs(k, NT, NV, NU)
            m = e < D
            yv = _r2_hcpre(x, pre, e, m, D, HC, HCMODE)
            acc2 = acc2 + yv * yv
        r = libdevice.rsqrt(_r2_fin(_r2_finish(acc2, NT, NV, NDEC), fac, wf, NDIV) + eps)
        bi = b0 + tl.arange(0, BB)
        bok = bi < D // 32
        cols = bi[:, None] * 32 + tl.arange(0, 32)[None, :]
        m = bok[:, None] & (cols >= 0)
        yv = _r2_hcpre(x, pre, cols, m, D, HC, HCMODE)
        o = _t9b2_bf16(tl.load(nw + cols, mask=m, other=0.0).to(tl.float32) * (yv * r))
        tl.store(y + cols, o.to(tl.bfloat16), mask=m)
        if HAS_YF:
            tl.store(yf + cols, o, mask=m)
        if QQ:
            code = _t9b2_code(tl.max(tl.abs(o), axis=1), tab)
            qv = _t9b2_q8(o, _t9b2_ipow2(code)[:, None])
            if HAS_Q:
                tl.store(q + cols, qv, mask=m)
                tl.store(s + bi, code.to(tl.uint8), mask=bok)
            if HAS_DG:
                for gi in range(G):
                    tl.store(dq + gi * dq_gs + cols, qv, mask=m)
                    _t9b4_store_code(ds, gi, bi, code, bok, ds_gs, ds_cs, DG_PACK)

    @triton.jit
    def t9b4_hc_norm(x, pre, nw, y, yf, xf, mrs, q, s, dq, ds, dq_gs, ds_gs, ds_cs, G, tab,
                     eps, fac, wf, meps, mfac, mwf, NAP,
                     D: tl.constexpr, HC: tl.constexpr, HCMODE: tl.constexpr,
                     NT: tl.constexpr, NV: tl.constexpr, NU: tl.constexpr, NDEC: tl.constexpr, NDIV: tl.constexpr,
                     MT: tl.constexpr, MV: tl.constexpr, MU: tl.constexpr, MDEC: tl.constexpr, MDIV: tl.constexpr,
                     PASS_A: tl.constexpr, HAS_MIX: tl.constexpr, HAS_XF: tl.constexpr, HAS_YF: tl.constexpr,
                     QQ: tl.constexpr, HAS_Q: tl.constexpr, HAS_DG: tl.constexpr, DG_PACK: tl.constexpr,
                     BB: tl.constexpr):
        """t9b2r2_hc_norm's outputs, bit for bit, from NAP (+1 with PASS_A) programs: program p < NAP recomputes the
        norm statistic (identical code -> identical r) and writes the 32-blocks [p*BB, (p+1)*BB); program NAP (PASS_A)
        writes xf and the mix statistic. The reduction ORDER is the calibrated one, untouched: the work is split, the
        sums are not (redundant statistic instead of a partitioned reduction + combine). M1-T10-PORT: grid axis 1 =
        token row t, as t9b2r2_hc_norm's."""
        pid = tl.program_id(0)
        t = tl.program_id(1)
        x = x + t * (HC * D)
        pre = pre + t * HC
        y = y + t * D
        yf = yf + t * D
        xf = xf + t * (HC * D)
        mrs = mrs + t
        q = q + t * D
        s = s + t * (D // 32)
        if PASS_A:
            if pid >= NAP:
                _t9b4_hc_mix(x, xf, mrs, meps, mfac, mwf, D, HC, MT, MV, MU, MDEC, MDIV, HAS_MIX, HAS_XF)
            else:
                _t9b4_hc_apply(x, pre, nw, y, yf, q, s, dq, ds, dq_gs, ds_gs, ds_cs, G, tab, eps, fac, wf, pid * BB,
                               D, HC, HCMODE, NT, NV, NU, NDEC, NDIV, HAS_YF, QQ, HAS_Q, HAS_DG, DG_PACK, BB)
        else:
            _t9b4_hc_apply(x, pre, nw, y, yf, q, s, dq, ds, dq_gs, ds_gs, ds_cs, G, tab, eps, fac, wf, pid * BB,
                           D, HC, HCMODE, NT, NV, NU, NDEC, NDIV, HAS_YF, QQ, HAS_Q, HAS_DG, DG_PACK, BB)

    # --- #3 / #6
    @triton.jit
    def t9b2r2_norm_row(x, nw, out, slot, q, s, freqs, tab, eps, fac, wf,
                        N: tl.constexpr, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr,
                        DIV: tl.constexpr, HAS_SLOT: tl.constexpr, QQ: tl.constexpr, HAS_Q: tl.constexpr,
                        QDQ: tl.constexpr, RD: tl.constexpr, RE_MODE: tl.constexpr, IM_MODE: tl.constexpr,
                        BB: tl.constexpr, XS=0, NS: tl.constexpr = 0, FS=0):
        """One program: RMSNorm of one bf16 row x [N] (model.py:288-293, statistic in order T/V/U/DEC/DIV), then per
        32-block as even/odd pairs: RoPE on the last RD columns (RD > 0; the forward rotation by freqs [RD/2, 2],
        contraction forms RE/IM_MODE, bf16), act_quant QDQ in place (QDQ) -> out (+ slot[0] * N: the window ring
        row, HAS_SLOT); and/or the act_quant of the row: q fp8 + s e8m0 (HAS_Q). T9b-4 (NS > 0): x is the GEMV's
        fp32 split partials [NS, N] (split stride XS), reduced on load in the split-K reduce's order (_t9b4_ld).
        M1-T10-PORT: program t = token row t of R (one program at the one-row step): x / q / s / out row t (out:
        ring row slot[t] with HAS_SLOT), freqs row t (stride FS); split partials [NS, R, N]: row t inside each."""
        t = tl.program_id(0)
        x = x + t * N
        q = q + t * N
        s = s + t * (N // 32)
        freqs = freqs + t * FS
        acc = tl.zeros([T, V], dtype=tl.float32)
        for k in range((N + T * V - 1) // (T * V)):
            e = _r2_offs(k, T, V, U)
            m = e < N
            v = _t9b4_ld(x, e, m, NS, XS)
            acc = acc + v * v
        r = libdevice.rsqrt(_r2_fin(_r2_finish(acc, T, V, DEC), fac, wf, DIV) + eps)
        if HAS_SLOT:
            base = out + tl.load(slot + t) * N
        else:
            base = out + t * N
        for b0 in range(0, N // 32, BB):
            bi = b0 + tl.arange(0, BB)
            bok = bi < N // 32
            ce = bi[:, None] * 32 + 2 * tl.arange(0, 16)[None, :]
            m = bok[:, None] & (ce >= 0)
            a = _t9b2_bf16(tl.load(nw + ce, mask=m, other=0.0).to(tl.float32)
                           * (_t9b4_ld(x, ce, m, NS, XS) * r))
            b = _t9b2_bf16(tl.load(nw + ce + 1, mask=m, other=0.0).to(tl.float32)
                           * (_t9b4_ld(x, ce + 1, m, NS, XS) * r))
            if RD > 0:
                is_r = ce >= N - RD
                fi = tl.where(is_r, (ce - (N - RD)) // 2, 0)
                mr = m & is_r
                c = tl.load(freqs + 2 * fi, mask=mr, other=0.0)
                d = tl.load(freqs + 2 * fi + 1, mask=mr, other=0.0)
                a2 = tl.where(is_r, _t9b2_bf16(_t9b2_rope_re(a, b, c, d, RE_MODE)), a)
                b2 = tl.where(is_r, _t9b2_bf16(_t9b2_rope_im(a, b, c, d, IM_MODE)), b)
            else:
                a2 = a
                b2 = b
            if QQ:
                code = _t9b2_code(tl.maximum(tl.max(tl.abs(a2), axis=1), tl.max(tl.abs(b2), axis=1)), tab)
                inv = _t9b2_ipow2(code)[:, None]
                qa = _t9b2_q8(a2, inv)
                qb = _t9b2_q8(b2, inv)
                if HAS_Q:
                    tl.store(q + ce, qa, mask=m)
                    tl.store(q + ce + 1, qb, mask=m)
                    tl.store(s + bi, code.to(tl.uint8), mask=bok)
                if QDQ:
                    sc = _t9b2_pow2(code)[:, None]
                    tl.store(base + ce, (qa.to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne"),
                             mask=m)
                    tl.store(base + ce + 1, (qb.to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne"),
                             mask=m)
                else:
                    tl.store(base + ce, a2.to(tl.bfloat16), mask=m)
                    tl.store(base + ce + 1, b2.to(tl.bfloat16), mask=m)
            else:
                tl.store(base + ce, a2.to(tl.bfloat16), mask=m)
                tl.store(base + ce + 1, b2.to(tl.bfloat16), mask=m)

    # --- #4
    @triton.jit
    def t9b2r2_router_pre(sp, bias, sc, bsc, E, BLOCK_E: tl.constexpr, NT=0, ROWS: tl.constexpr = False):
        """scores = sqrt(softplus) (IEEE sqrt), biased = scores + bias (model.py:819, 824). M1-T10-PORT ROWS: R rows
        of E scores flattened (NT = R*E elements, bias index i % E)."""
        i = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        if ROWS:
            m = i < NT
            bi = i % E
        else:
            m = i < E
            bi = i
        v = libdevice.sqrt_rn(tl.load(sp + i, mask=m, other=0.0))
        tl.store(sc + i, v, mask=m)
        tl.store(bsc + i, v + tl.load(bias + bi, mask=m, other=0.0), mask=m)

    @triton.jit
    def t9b2r2_router_post(idx, sc, ids_out, w_out, K, rs, eps,
                           T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr, KP: tl.constexpr,
                           SE=0):
        """One program: weights = scores[idx] (torch.topk's order), w = weights / (sum + eps) * route_scale
        (model.py:825-828, the sum in order T/V/U/DEC), then ascending by id (distinct ids: rank = how many are
        smaller) -> ids_out int64, w_out fp32. M1-T10-PORT: program t = token row t (idx / ids_out / w_out row t of
        [R, K], sc row t of [R, E], row stride SE); one program at the one-row step."""
        t = tl.program_id(0)
        idx = idx + t * K
        sc = sc + t * SE
        ids_out = ids_out + t * K
        w_out = w_out + t * K
        acc = tl.zeros([T, V], dtype=tl.float32)
        for kk in range((K + T * V - 1) // (T * V)):
            e = _r2_offs(kk, T, V, U)
            me = e < K
            ie = tl.load(idx + e, mask=me, other=0)
            acc = acc + tl.load(sc + ie, mask=me, other=0.0)
        den = _r2_finish(acc, T, V, DEC) + eps
        j = tl.arange(0, KP)
        mj = j < K
        ij = tl.load(idx + j, mask=mj, other=0)
        wj = libdevice.div_rn(tl.load(sc + ij, mask=mj, other=0.0), den) * rs
        rank = tl.sum(tl.where(mj[None, :] & (ij[None, :] < ij[:, None]), 1, 0), axis=1)
        tl.store(ids_out + rank, ij, mask=mj)
        tl.store(w_out + rank, wj, mask=mj)

    # --- #15
    @triton.jit
    def t9b2r2_engram_gate(x, kv, wp, z, eps, fac, wf, c, cmin,
                           D: tl.constexpr, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr,
                           DIV: tl.constexpr, ST: tl.constexpr, SV: tl.constexpr, SU: tl.constexpr,
                           SDEC: tl.constexpr, HC: tl.constexpr = 0):
        """Program i (hc copy): h = x[i].float(), key = kv[i*D:(i+1)*D].float(); rstd = rsqrt(mean(h^2) + eps) *
        rsqrt(mean(key^2) + eps) (order T/V/U/DEC/DIV); dot = sum((h * wp[i]) * key) (order S*) * rstd * c;
        z[i] = copysign(sqrt(max(|dot|, cmin)), dot) (model.py:357-362 up to the sigmoid). M1-T10-PORT: grid axis 1
        = token row t (x row t of [R, HC, D], kv row t of [R, (HC+1)*D], z row t of [R, HC]; HC given)."""
        i = tl.program_id(0)
        t = tl.program_id(1)
        x = x + t * (HC * D)
        kv = kv + t * ((HC + 1) * D)
        z = z + t * HC
        ah = tl.zeros([T, V], dtype=tl.float32)
        ak = tl.zeros([T, V], dtype=tl.float32)
        for k in range((D + T * V - 1) // (T * V)):
            e = _r2_offs(k, T, V, U)
            m = e < D
            h = tl.load(x + i * D + e, mask=m, other=0.0).to(tl.float32)
            kk = tl.load(kv + i * D + e, mask=m, other=0.0).to(tl.float32)
            ah = ah + h * h
            ak = ak + kk * kk
        ad = tl.zeros([ST, SV], dtype=tl.float32)
        for k in range((D + ST * SV - 1) // (ST * SV)):
            e = _r2_offs(k, ST, SV, SU)
            m = e < D
            h = tl.load(x + i * D + e, mask=m, other=0.0).to(tl.float32)
            kk = tl.load(kv + i * D + e, mask=m, other=0.0).to(tl.float32)
            w = tl.load(wp + i * D + e, mask=m, other=0.0)
            ad = ad + (h * w) * kk
        rh = libdevice.rsqrt(_r2_fin(_r2_finish(ah, T, V, DEC), fac, wf, DIV) + eps)
        rk = libdevice.rsqrt(_r2_fin(_r2_finish(ak, T, V, DEC), fac, wf, DIV) + eps)
        dot = (_r2_finish(ad, ST, SV, SDEC) * (rh * rk)) * c
        a = tl.abs(dot)
        a = tl.where(a < cmin, cmin, a)
        sq = libdevice.sqrt_rn(a)
        sgn = (dot.to(tl.int32, bitcast=True) >> 31) << 31          # the sign bit, int32 (no int64 literal)
        zz = (sq.to(tl.int32, bitcast=True) & 0x7FFFFFFF) | sgn
        tl.store(z + i, zz.to(tl.float32, bitcast=True))

    @triton.jit
    def t9b2r2_engram_out(x, kv, gate, out, D: tl.constexpr, HC: tl.constexpr, BLOCK_C: tl.constexpr):
        """Program (pb, i): out[i] = bf16(h[i] + gate[i] * value) (model.py:365), value = kv[HC*D:]. M1-T10-PORT:
        grid axis 2 = token row t (x / out row t of [R, HC, D], kv row t of [R, (HC+1)*D], gate row t of [R, HC])."""
        pb = tl.program_id(0)
        i = tl.program_id(1)
        t = tl.program_id(2)
        x = x + t * (HC * D)
        out = out + t * (HC * D)
        kv = kv + t * ((HC + 1) * D)
        gate = gate + t * HC
        e = pb * BLOCK_C + tl.arange(0, BLOCK_C)
        m = e < D
        h = tl.load(x + i * D + e, mask=m, other=0.0).to(tl.float32)
        v = tl.load(kv + HC * D + e, mask=m, other=0.0).to(tl.float32)
        o = h + tl.load(gate + i) * v
        tl.store(out + i * D + e, o.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)

    # --- #14
    @triton.jit
    def t9b2r2_topk_post(idx, clen, out, K, offset, KP: tl.constexpr, RB: tl.constexpr):
        """Program p: the ascending position (rank among the K distinct topk indices) of indices [p*RB, (p+1)*RB),
        and out[rank] = idx < clen ? idx + offset : -1 as int32 (model.py:580-581). M1-T10-PORT: grid axis 1 =
        token row t (idx / out row t of [R, K], clen[t] -- the [R, 1] compress_lens of the R-row step)."""
        p = tl.program_id(0)
        t = tl.program_id(1)
        idx = idx + t * K
        out = out + t * K
        clen = clen + t
        i = p * RB + tl.arange(0, RB)
        mi = i < K
        ii = tl.load(idx + i, mask=mi, other=0)
        j = tl.arange(0, KP)
        mj = j < K
        ij = tl.load(idx + j, mask=mj, other=0)
        rank = tl.sum(tl.where(mj[None, :] & (ij[None, :] < ii[:, None]), 1, 0), axis=1)
        cl = tl.load(clen)
        v = tl.where(ii < cl, ii + offset, -1)
        tl.store(out + rank, v.to(tl.int32), mask=mi)


# ------------------------------------------------------------------------------------------------ launch wrappers
# One kernel launch each on CUDA (the census counts each as one) and the torch mirror on CPU. FusedGlue reaches them
# through this module's globals at call time (bench/glue_census.py wraps them).
def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


HC_SPLIT_BB = 8       # T9b-4: 32-blocks per apply program of t9b4_hc_norm (D = 5120: 20 apply + 1 mix programs)
HC_SPLIT_WARPS = 8


def _ds_args(ds, dummy):
    """(pointer tensor, row stride, column stride, packed) of the DeepGEMM scale destination: fp32 rows [G, D/32]
    (stride(0)), or a runtime.dg_inplace.PackedRows (T9b-4: bytes of the packed MN-major SF)."""
    if ds is None:
        return dummy, 0, 0, False
    if getattr(ds, "packed", False):
        return ds.u8, ds.row_stride, ds.col_stride, True
    return ds, ds.stride(0), 0, False


def launch_hc_norm(x, pre, nw, y, *, eps: float, norm_ord: dict, hc_mode: int, code_tab=None, yf=None, xf=None,
                   mrs=None, mix_ord: Optional[dict] = None, mix_eps: float = 0.0, q=None, s=None, dq=None, ds=None,
                   split: bool = False):
    """x [HC, D] bf16 contiguous, pre [HC] fp32, nw [D]; y [D] bf16; yf [D] fp32; xf [HC*D] fp32; mrs [1] fp32;
    q [D] fp8, s [D/32] e8m0; dq [G, D] fp8 rows (unit inner stride), ds [G, D/32] fp32 or (T9b-4) a
    runtime.dg_inplace.PackedRows (the codes go straight into DeepGEMM's packed activation SF). split (T9b-4 lever
    1): t9b4_hc_norm over 1 + ceil(D/32 / HC_SPLIT_BB) programs instead of t9b2r2_hc_norm's one (same outputs).
    M1-T10-PORT: x [R, HC, D], pre [R, HC] and every output R rows of the above (row-major, contiguous), one launch
    (grid axis 1 = row); the statistics' orders are the ones calibrated at R rows; no DeepGEMM rows at R > 1."""
    HC, D = x.shape[-2], x.shape[-1]
    R_ = x.numel() // (HC * D)
    if R_ > 1 and dq is not None:
        raise ValueError("launch_hc_norm: the DeepGEMM rows (dq/ds) are one-row only")
    if _kern(x):
        na = order_args(norm_ord)
        ma = order_args(mix_ord) if mrs is not None else dict(T=128, V=4, U=False, DEC=True, DIV=False)
        dummy = y
        G = dq.size(0) if dq is not None else 0
        su = s.view(torch.uint8) if s is not None else dummy
        dsk, ds_gs, ds_cs, packed = _ds_args(ds, dummy)
        args = (x, pre, nw, y, yf if yf is not None else dummy, xf if xf is not None else dummy,
                mrs if mrs is not None else dummy, q if q is not None else dummy, su,
                dq if dq is not None else dummy, dsk, dq.stride(0) if dq is not None else 0, ds_gs, ds_cs, G,
                code_tab if code_tab is not None else dummy,
                float(eps), fac_of(D), float(D), float(mix_eps), fac_of(HC * D), float(HC * D))
        pass_a = xf is not None or mrs is not None
        ce = dict(D=D, HC=HC, HCMODE=int(hc_mode),
                  NT=na["T"], NV=na["V"], NU=na["U"], NDEC=na["DEC"], NDIV=na["DIV"],
                  MT=ma["T"], MV=ma["V"], MU=ma["U"], MDEC=ma["DEC"], MDIV=ma["DIV"],
                  PASS_A=pass_a, HAS_MIX=mrs is not None, HAS_XF=xf is not None,
                  HAS_YF=yf is not None, QQ=q is not None or dq is not None, HAS_Q=q is not None, HAS_DG=dq is not None,
                  DG_PACK=packed)
        rows = (R_,) if R_ > 1 else ()
        if split:
            nap = -(-(D // BLOCK) // HC_SPLIT_BB)
            t9b4_hc_norm[(nap + (1 if pass_a else 0),) + rows](*args, nap, **ce, BB=HC_SPLIT_BB,
                                                               num_warps=HC_SPLIT_WARPS, **LAUNCH)
        else:
            t9b2r2_hc_norm[(1,) + rows](*args, **ce, BB=8, num_warps=NORM_WARPS, **LAUNCH)
        return
    xv = x.float().reshape(R_, HC * D)
    if xf is not None:
        xf.copy_(xv.reshape(xf.shape))
    if mrs is not None:
        mrs.copy_(rms_rsqrt_torch(xv, mix_eps, mix_ord, (1, R_, HC * D)).view(mrs.shape))
    y0 = hc_pre_torch(x.reshape(R_, HC, D), pre.reshape(R_, HC), hc_mode).float().view(R_, D)
    o = rms_apply_torch(y0, rms_rsqrt_torch(y0, eps, norm_ord, (1, R_, D)), nw)
    y.copy_(o.view(y.shape))
    if yf is not None:
        yf.copy_(o.float().view(yf.shape))
    if q is not None or dq is not None:
        qv, code = act_quant_torch(o.float(), code_tab)
        if q is not None:
            q.copy_(qv.view(q.shape))
            s.view(torch.uint8).copy_(code.to(torch.uint8).view(s.shape))
        if dq is not None:
            for gi in range(dq.size(0)):
                dq[gi].copy_(qv.view(-1))
                if getattr(ds, "packed", False):
                    ds.store_codes_torch(gi, code.view(-1))
                else:
                    ds[gi].copy_(pow2_code(code.view(-1)))


def _row_src(x, N: int):
    """(tensor, R, NS, XS): R bf16 rows of N ([N] or [R, N] contiguous), or (T9b-4 reduce fold) a gemv_mxfp8.SplitK
    of R rows -> its fp32 partials [nsplit, R, N], nsplit and split stride (M1-T10-PORT: any R <= 8)."""
    if hasattr(x, "nsplit") and hasattr(x, "ws"):
        if x.N != N:
            raise ValueError(f"norm row: SplitK of width {x.N}, norm width {N}")
        return x.ws, x.M, x.nsplit, x.split_stride
    if x.numel() % N:
        raise ValueError(f"norm row: {x.numel()} elements is not a whole number of {N}-rows")
    return x, x.numel() // N, 0, 0


def launch_norm_row(x, nw, out, *, eps: float, ordd: dict, code_tab=None, slot=None, q=None, s=None,
                    qdq: bool = False, rope=None):
    """x [N] bf16 (or, T9b-4, a one-row gemv_mxfp8.SplitK: the GEMV's partials, reduced in the kernel's load), nw [N];
    out bf16 [N] (or the ring [win, N] with slot [1] int64); q [N] fp8 + s [N/32] e8m0;
    rope = (freqs fp32 [RD/2, 2], RD, (re, im) modes). M1-T10-PORT: R rows ([R, N] contiguous or an R-row SplitK):
    out / q / s [R, ...] (or the ring with slot [R]), freqs [R, RD/2, 2] (row t rotated by its own position); one
    launch, one program per row."""
    N = nw.numel()
    xt, R_, NS, XS = _row_src(x, N)
    RD = int(rope[1]) if rope is not None else 0
    modes = tuple(rope[2]) if rope is not None else (0, 0)
    fr = rope[0] if rope is not None else None
    if _kern(xt):
        a = order_args(ordd)
        dummy = out
        fs = {"FS": int(fr.stride(0))} if fr is not None and fr.dim() == 3 else {}
        t9b2r2_norm_row[(R_,)](
            xt, nw, out, slot if slot is not None else xt, q if q is not None else dummy,
            s.view(torch.uint8) if s is not None else dummy, fr if fr is not None else nw,
            code_tab if code_tab is not None else dummy, float(eps), fac_of(N), float(N),
            N=N, T=a["T"], V=a["V"], U=a["U"], DEC=a["DEC"], DIV=a["DIV"], HAS_SLOT=slot is not None,
            QQ=q is not None or qdq, HAS_Q=q is not None, QDQ=bool(qdq), RD=RD, RE_MODE=int(modes[0]),
            IM_MODE=int(modes[1]), BB=min(16, _np2(N // 32)), XS=int(XS), NS=int(NS), **fs, num_warps=NORM_WARPS,
            **LAUNCH)
        return
    if NS:
        x = x.reduce_torch()
    v = x.float().reshape(R_, N)
    o = rms_apply_torch(v, rms_rsqrt_torch(v, eps, ordd, (1, R_, N)), nw)
    if RD:
        from runtime.fused_glue import rope_rows_torch

        o = torch.stack([rope_rows_torch(o[t], fr[t] if fr.dim() == 3 else fr, N, RD, modes, inverse=False)
                         for t in range(R_)])
    if q is not None or qdq:
        qv, code = act_quant_torch(o.float(), code_tab)
        if q is not None:
            q.copy_(qv.view(q.shape))
            s.view(torch.uint8).copy_(code.to(torch.uint8).view(s.shape))
        if qdq:
            o = (qv.float().unflatten(-1, (-1, BLOCK)) * pow2_code(code).unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    if slot is not None:
        out.view(-1, N).index_copy_(0, slot.reshape(-1), o.view(R_, N))
    else:
        out.view(R_, N).copy_(o.view(R_, N))


def launch_router_pre(sp, bias, sc, bsc):
    """sp, bias, sc, bsc: [E] fp32 contiguous. M1-T10-PORT: sp / sc / bsc R rows of E ([R*E]), bias [E]."""
    E = bias.numel()
    NT = sp.numel()
    if _kern(sp):
        BLOCK_E = min(1024, _np2(E))
        rk = dict(NT=int(NT), ROWS=True) if NT != E else {}
        t9b2r2_router_pre[(triton.cdiv(NT, BLOCK_E),)](sp, bias, sc, bsc, E, BLOCK_E=BLOCK_E, **rk, **LAUNCH)
        return
    v = sp.sqrt()
    sc.copy_(v)
    bsc.copy_((v.view(-1, E) + bias).view(bsc.shape))


def launch_router_post(idx, sc, ids_out, w_out, *, ordd: dict, eps: float, rs: float):
    """idx [K] int64 (topk's order), sc [E] fp32 -> ids_out [K] int64 ascending, w_out [K] fp32. M1-T10-PORT: idx
    [R, K], sc [R, E] -> ids_out / w_out [R, K] (ascending per row), one launch (one program per row)."""
    rows = idx.dim() == 2
    R_ = idx.size(0) if rows else 1
    K = idx.size(-1)
    if _kern(idx):
        a = order_args(ordd)
        se = {"SE": int(sc.stride(0))} if rows else {}
        t9b2r2_router_post[(R_,)](idx, sc, ids_out, w_out, K, float(rs), float(eps), T=a["T"], V=a["V"], U=a["U"],
                                  DEC=a["DEC"], KP=_np2(K), **se, **LAUNCH)
        return
    ids, w = router_post_torch(idx, sc, ordd, eps, rs)
    ids_out.copy_(ids.view(ids_out.shape))
    w_out.copy_(w.view(w_out.shape))


def launch_engram_gate(x, kv, wp, z, *, eps: float, ordd: dict, sordd: dict, c: float, cmin: float):
    """x [HC, D] bf16, kv [(HC+1)*D] bf16, wp [HC, D] fp32 -> z [HC] fp32. M1-T10-PORT: x [R, HC, D], kv R rows of
    (HC+1)*D, z [R*HC]; grid (HC, R)."""
    HC, D = x.shape[-2], x.shape[-1]
    R_ = x.numel() // (HC * D)
    if _kern(x):
        a, sa = order_args(ordd), order_args(sordd)
        t9b2r2_engram_gate[(HC,) + ((R_,) if R_ > 1 else ())](
            x, kv, wp, z, float(eps), fac_of(D), float(D), float(c), float(cmin), D=D, T=a["T"], V=a["V"], U=a["U"],
            DEC=a["DEC"], DIV=a["DIV"], ST=sa["T"], SV=sa["V"], SU=sa["U"], SDEC=sa["DEC"],
            **({"HC": HC} if R_ > 1 else {}), num_warps=NORM_WARPS, **LAUNCH)
        return
    key = kv.reshape(R_, (HC + 1) * D)[:, : HC * D].float().reshape(R_, HC, D)
    zz = engram_gate_torch(x.float().reshape(R_, HC, D) if R_ > 1 else x.float(), key if R_ > 1 else key[0], wp,
                           eps, ordd, sordd, c, cmin)
    z.copy_(zz.reshape(z.shape))


ENGRAM_BLOCK = 1024


def launch_engram_out(x, kv, gate, out):
    """x [HC, D] bf16, kv [(HC+1)*D] bf16, gate [HC] fp32 -> out [HC, D] bf16. M1-T10-PORT: R rows of each (x / out
    [R, HC, D], kv [R*(HC+1)*D], gate [R*HC]); grid axis 2 = row."""
    HC, D = x.shape[-2], x.shape[-1]
    R_ = x.numel() // (HC * D)
    if _kern(x):
        t9b2r2_engram_out[(triton.cdiv(D, ENGRAM_BLOCK), HC) + ((R_,) if R_ > 1 else ())](
            x, kv, gate, out, D=D, HC=HC, BLOCK_C=ENGRAM_BLOCK, **LAUNCH)
        return
    v = kv.reshape(R_, (HC + 1) * D)[:, HC * D:].float().view(R_, 1, D)
    res = (x.float().reshape(R_, HC, D) + gate.reshape(R_, HC, 1) * v).to(torch.bfloat16)
    out.copy_(res.view(out.shape))


TOPK_RB = 16      # [RB, next_pow2(K)] compare tile per program: 16 x 512 = 64 values per thread at 4 warps


def launch_topk_post(idx, clen, out, offset: int):
    """idx [K] int64 distinct, clen [1] int64 -> out [K] int32 ascending, -1 past clen, + offset. M1-T10-PORT: idx /
    out [R, K] (distinct per row), clen [R] (the R-row step's [R, 1] compress_lens); grid axis 1 = row."""
    rows = idx.dim() == 2
    R_ = idx.size(0) if rows else 1
    K = idx.size(-1)
    if _kern(idx):
        RB = min(TOPK_RB, _np2(K))
        t9b2r2_topk_post[(triton.cdiv(K, RB),) + ((R_,) if rows else ())](idx, clen, out, K, int(offset), KP=_np2(K),
                                                                         RB=RB, **LAUNCH)
        return
    out.copy_(indexer_post_torch(idx, clen.reshape(R_, 1) if rows else clen, offset).view(out.shape))


# ------------------------------------------------------------------------------------------------ the R2 glue
def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    n = t.element_size()
    return t.view(torch.uint8) if n == 1 else t.view(torch.int16 if n == 2 else (torch.int32 if n == 4 else
                                                                                   torch.int64))


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return 1.0
    return float((_bits(a) != _bits(b)).float().mean())


class R2Glue:
    """Round-2 clusters, mixed into runtime/fused_glue.FusedGlue (which owns status/on/_off/log/R/o/dev). Call sites:
    runtime/static_step.py (`_mix_pre_norm`, `_q_path`, `_kv_path`, `_head`, `_engram`, `_indexer`), StaticMoE /
    DGMaskedMoE (`route`, `ffn_pre`) and FusedGlue.shared_expert."""

    # ---------------------------------------------------------------- setup
    def _r2_init(self, layers, approx: bool) -> None:
        self.approx = bool(approx)
        self.orders: Dict[str, dict] = {}
        self.hc_pre_mode = 0
        self.numerics: Dict[str, dict] = {}
        self._ffn_ctx: Optional[dict] = None
        self.gate_w: Dict[int, torch.Tensor] = {}
        self.engram_wp: Dict[int, torch.Tensor] = {}
        # T9b-7: the order policy of THIS build (fixed at build: an env change later does not mix policies) and the
        # row count every order is calibrated at (1 under `shape`, whatever R this step runs)
        self.order_policy = glue_order()
        self.cal_rows = cal_rows(self.order_policy, getattr(self, "rows", 1))
        o = self.o
        if self.on("router"):
            for sm in o.moes:
                gate = sm.moe.gate
                w = gate.weight
                self.gate_w[id(gate)] = w if w.dtype == torch.float32 else w.detach().float().contiguous()
        # M2-T15 (runtime/router_plan.py): $DSV41_ROUTER_IMPL plan / plan-gemv -> the router (+ the R-row plan) in one
        # launch, self-checked at build; default "fused" = the clusters below, unchanged
        from runtime import router_plan as RP

        self.router_impl = RP.router_impl() if self.on("router") else "fused"
        self.rplan = RP.RouterPlan(self, self.router_impl) if self.router_impl != "fused" else None
        if self.on("engram"):
            for l in layers:
                if l.engram is not None:
                    e = l.engram
                    self.engram_wp[id(e)] = (e.q_weight.detach().float() * e.k_weight.detach().float()).contiguous()
        self.r2_bytes = {"gate_fp32": sum(t.numel() * 4 for t in self.gate_w.values()
                                          if t.data_ptr() not in {sm.moe.gate.weight.data_ptr() for sm in o.moes}),
                         "engram_wprod": sum(t.numel() * 4 for t in self.engram_wp.values())}

    def _r2_default_orders(self) -> None:
        """calibrate=False (the census): torch's own op on CPU, the predicted configuration on a device."""
        model = self.o.model
        l0 = model.layers[0]

        def d(w, rows=1):
            if self.dev.type != "cuda":
                return {"order": NATIVE, "fin": "mul", "exact": True, "name": NATIVE}
            o_ = predicted_order(w, rows)
            return {"order": o_, "fin": "mul", "exact": True, "name": order_name(o_, "mul")}

        D = l0.attn_norm.dim
        Rr = self.cal_rows   # T9b-7: 1 (shape policy, every R) or (rows policy) the R-row reference's rows
        self.orders.update(norm_D=d(D, Rr), mix=d(self.hc * D, Rr), kv=d(l0.attn.head_dim, Rr),
                           q=d(l0.attn.q_lora_rank, Rr), router=d(l0.ffn.gate.topk, Rr))
        eng = next((l.engram for l in model.layers if l.engram is not None), None)
        if eng is not None:
            self.orders.update(engram=d(eng.dim, eng.hc_mult * Rr), engram_sum=d(eng.dim, eng.hc_mult * Rr))

    def _applicable_r2(self, c: str, layers, mx: bool) -> Optional[str]:
        o = self.o
        R = self.R
        m = o.model
        if c in R2_ACT_CLUSTERS and not mx:
            return "needs the MXFP8 act_quant globals (block 32, ue8m0, e8m0 scales)"
        if getattr(R, "world_size", 1) != 1:
            return "world_size > 1"
        if c in ("hc_norm", "mix_stat"):
            D = layers[0].attn_norm.dim
            if any(l.hc_mult != self.hc for l in layers):
                return "hc_mult differs across layers"
            if D % BLOCK or any(l.attn_norm.dim != D or l.ffn_norm.dim != D for l in layers) or m.norm.dim != D:
                return "norm widths differ or are not 32-multiples"
            return None
        attns = [l.attn for l in layers]
        if c == "kv_path":
            if any(a.head_dim % BLOCK or a.kv_norm.dim != a.head_dim or a.rope_head_dim % 2 for a in attns):
                return "head_dim not a 32-multiple or odd rope dim"
            if any(a.window_kv_cache.dtype != torch.bfloat16 or not a.window_kv_cache.is_contiguous() for a in attns):
                return "window cache not a contiguous bf16 ring"
            return None
        if c == "q_path":
            if any(a.q_lora_rank % BLOCK or a.q_norm.dim != a.q_lora_rank for a in attns):
                return "q_lora_rank not a 32-multiple"
            return None
        if c == "router":
            g0 = layers[0].ffn.gate
            if any(l.ffn.gate.score_func != "sqrtsoftplus" for l in layers):
                return "score_func is not sqrtsoftplus"
            if not (g0.norm_topk_prob and g0.topk > 1):
                return "norm_topk_prob off or topk 1 (the normalization branch of model.py:826)"
            return None
        if c == "engram":
            if not any(l.engram is not None for l in layers):
                return "no Engram layers"
            if any(l.engram is not None and l.engram.dim % BLOCK for l in layers):
                return "Engram dim not a 32-multiple"
            return None
        if c == "indexer_post":
            if not any(a.indexer is not None for a in attns):
                return "no index-source layers"
            return None
        return "unknown"

    def _applicable_r4(self, c: str, layers, mx: bool) -> Optional[str]:
        """T9b-4 parts (fused_glue.R4_PARTS)."""
        o = self.o
        if c == "hc_split":
            return self._applicable_r2("hc_norm", layers, mx)
        if c in ("gemv_pair", "reduce_fold"):
            return None if o.fp8_impl == "gemv" else "fp8_impl ref (the pair / the fold live in the T9b-1 GEMV)"
        if c == "dg_sf_pack":
            if o.moe_impl != "dg-inplace":
                return "moe_impl is not dg-inplace (the contiguous GEMM is the one that converts the activation SF)"
            if getattr(getattr(o, "dg", None), "layout", "contiguous") == "masked":   # M1-T10-MOE
                return "the masked call shape takes fp32 activation SF (no packed form)"
            return None if mx else "needs the MXFP8 act_quant globals"
        return "unknown"

    # ---------------------------------------------------------------- status helpers
    def _r2_norm(self, key: str) -> dict:
        return self.orders[key]

    def _oracle(self, two_d: bool = False):
        """T9b-7: the context the self-checks evaluate the REFERENCE ops in -- at R > 1 under the shape policy, torch's
        per-token sums / means one row at a time (per_row_reductions: the 1-row reference, row by row); else as is."""
        return per_row_reductions(self.order_policy == "shape" and self.rows > 1, two_d=two_d)

    def _set_outcome(self, clusters, rec: dict, what: str) -> None:
        """Record the numerics outcome of the order `rec` for `clusters`: (i) exact keeps them on; (ii) approx turns
        them `approx` (glue_approx) or off."""
        for c in clusters:
            if not self.on(c):
                continue
            n = self.numerics.setdefault(c, {"outcome": "(i) bit-identical", "orders": {}})
            n["orders"][what] = {k: rec[k] for k in ("name", "match", "max_ulp", "rows", "w", "kind", "tried")
                                 if k in rec}
            if not rec["exact"]:
                n["outcome"] = (f"(ii) fused-approx: no order of the family equals torch's {what} statistic "
                                f"(best {rec['name']}: {rec['match'] * 100:.2f} % of {rec['n']} rows, max "
                                f"{rec['max_ulp']} ulp)")
                if self.approx:
                    self.status[c] = "approx"
                else:
                    self._off(c, f"(ii) approx only (enable with glue_approx / --glue-approx): {what} order "
                                 f"{rec['name']} matches {rec['match'] * 100:.2f} % of rows")

    # ---------------------------------------------------------------- calibration (build time, eager, on device)
    def calibrate_r2(self, cal: dict, g) -> None:
        o, R = self.o, self.R
        model = o.model
        layers = list(model.layers)
        l0 = layers[0]
        D = l0.attn_norm.dim
        dev = self.dev
        # T9b-7: the rows each order is picked at -- shape policy: the 1-row reference's (1; Engram hc_mult), the
        # same picks (and the same cache entries) as the 1-row build, whatever R this step runs; rows policy: the
        # R-row reference's (the pre-T9b-7 protocol)
        Rr = self.cal_rows
        r2 = cal.setdefault("r2", {})
        r2["order_policy"] = {"policy": self.order_policy, "cal_rows": Rr, "step_rows": self.rows}
        if self.on("hc_norm"):
            rec = self.orders["norm_D"] = pick_order(dev, D, Rr, "sqmean")
            self._set_outcome(("hc_norm", "mix_stat"), rec, f"RMSNorm[{D}]")
            self._pick_hc_pre(r2, g)
        if self.on("mix_stat"):
            rec = self.orders["mix"] = pick_order(dev, self.hc * D, Rr, "sqmean")
            self._set_outcome(("mix_stat",), rec, f"hc_mixes[{self.hc * D}]")
        if self.on("kv_path"):
            HD = l0.attn.head_dim
            rec = self.orders["kv"] = pick_order(dev, HD, Rr, "sqmean")
            self._set_outcome(("kv_path",), rec, f"kv_norm[{HD}]")
        if self.on("q_path"):
            QL = l0.attn.q_lora_rank
            rec = self.orders["q"] = pick_order(dev, QL, Rr, "sqmean")
            self._set_outcome(("q_path",), rec, f"q_norm[{QL}]")
        if self.on("router"):
            K = l0.ffn.gate.topk
            rec = self.orders["router"] = pick_order(dev, K, Rr, "possum")
            self._set_outcome(("router",), rec, f"topk-weight sum[{K}]")
        if self.on("engram"):
            eng = next(l.engram for l in layers if l.engram is not None)
            rec = self.orders["engram"] = pick_order(dev, eng.dim, eng.hc_mult * Rr, "sqmean")
            self._set_outcome(("engram",), rec, f"Engram mean[{eng.hc_mult}x{eng.dim}]")
            rec2 = self.orders["engram_sum"] = pick_order(dev, eng.dim, eng.hc_mult * Rr, "sum")
            self._set_outcome(("engram",), rec2, f"Engram dot[{eng.hc_mult}x{eng.dim}]")
        if not self.on("hc_norm"):
            self._off("mix_stat", "needs hc_norm (its statistic rides in the hc_norm launch)")
        r2["orders"] = {k: {kk: (order_name(vv) if kk == "order" else vv) for kk, vv in v.items()}
                        for k, v in self.orders.items()}
        # T9b-4: the split hc_norm kernel against the one-program kernel FIRST, so that a broken split costs hc_split
        # only (the end-to-end hc_norm check below then runs on whichever kernel is in force)
        self._r4_check(cal, "hc_split", self._verify_hc_split, g)
        for c, fn in (("hc_norm", self._verify_hc_norm), ("kv_path", self._verify_kv_path),
                      ("q_path", self._verify_q_path), ("router", self._verify_router),
                      ("engram", self._verify_engram), ("indexer_post", self._verify_indexer_post)):
            if not self.on(c):
                continue
            try:
                res = fn(g)
            except Exception as e:  # a kernel that does not compile/run here stays off, with the reason
                res = {"why": f"raised {type(e).__name__}: {str(e)[:300]}", "mismatch": None}
            cal[f"verify_{c}"] = res.get("why") or ("equal" if not res.get("mismatch") else
                                                     f"approx: {res['mismatch']:.3e} of outputs differ")
            if res.get("why"):
                self._off(c, f"self-check: {res['why']}")
                if c == "hc_norm":
                    self._off("mix_stat", "hc_norm is off")
            elif res.get("mismatch") and self.status.get(c) == "on":
                self._off(c, f"self-check: {res['mismatch']:.3e} of outputs differ with an exact order")
        rp = getattr(self, "rplan", None)
        if rp is not None:                     # M2-T15: after the router's own check (the plan's reference)
            if self.on("router"):
                rp.calibrate(g)
            else:
                rp.off("the router cluster is off")
            cal["router_plan"] = rp.describe()
        for c in R2_CLUSTERS:
            if self.on(c) and c not in self.numerics:
                self.numerics[c] = {"outcome": "(i) bit-identical", "orders": {}}

    def _cat_rows(self, make):
        """M1-T10-PORT: self.rows independent draws of a one-row builder (tensors or tuples of [1, 1, ...]),
        concatenated along the token dim 1. One row: the builder's own draw (the one-row self-check unchanged)."""
        outs = [make() for _ in range(self.rows)]
        if len(outs) == 1:
            return outs[0]
        if isinstance(outs[0], (tuple, list)):
            return tuple(torch.cat([o_[i] for o_ in outs], 1) for i in range(len(outs[0])))
        return torch.cat(outs, 1)

    @_cpu_default
    def _stream(self, g, HC: int, D: int, scale_lo: int = -6, scale_hi: int = 7):
        from runtime.fused_glue import _rand_act

        return _rand_act(g, HC, D, scale_lo, scale_hi).view(1, 1, HC, D)

    @_cpu_default
    def _pre_mix(self, g, HC: int):
        return (torch.rand(1, 1, HC, generator=g, device="cpu") + 1e-6).float()

    @_cpu_default
    def _hc_cancel(self, g, HC: int, D: int):
        """hc_pre inputs on which the 4-term sum's ORDER shows (fused_glue._hc_data's idea): in half the columns
        copies 0 and 2 carry the same large value and pre_mix[2] = -pre_mix[0], so a0 + a2 cancels exactly and the
        HC_SUM_MODES round differently. (Random inputs alone agree in every mode almost always: a pick on them would
        be ambiguous and could take a wrong order.)"""
        from runtime.fused_glue import _rand_act

        x = _rand_act(g, HC, D).float()
        pre = torch.rand(HC, generator=g, device="cpu") + 0.1
        if HC >= 3:
            pre[2] = -pre[0]
            half = torch.arange(D, device="cpu") % 2 == 0
            big = torch.randn(D, generator=g, device="cpu") * 4096
            x[0] = torch.where(half, big, x[0])
            x[2] = torch.where(half, big, x[2])
        return x.to(torch.bfloat16).view(1, 1, HC, D), pre.view(1, 1, HC)

    def _hc_inputs(self, g, HC: int, D: int, i: int):
        """Alternating random streams and order-revealing (cancellation) ones."""
        if i % 2:
            return [t.to(self.dev) for t in self._hc_cancel(g, HC, D)]
        return self._stream(g, HC, D).to(self.dev), self._pre_mix(g, HC).to(self.dev)

    def _pick_hc_pre(self, r2: dict, g) -> None:
        layer = self.o.model.layers[0]
        D = layer.attn_norm.dim
        modes = HC_SUM_MODES if self.hc == 4 else (0,)
        mism = {m: 0 for m in modes}
        for i in range(8):
            x, pre = self._hc_inputs(g, self.hc, D, i)
            ref = layer.hc_pre(x, pre)
            for m in modes:
                got = hc_pre_torch(x.view(self.hc, D), pre.view(-1), m)
                mism[m] += int((_bits(got) != _bits(ref.view(-1))).sum())
        ok = [m for m in modes if mism[m] == 0]
        r2["hc_pre"] = {"mismatches": mism, "picked": ok[0] if ok else None, "ambiguous": len(ok) > 1}
        if ok:
            self.hc_pre_mode = ok[0]
        else:
            self._off("hc_norm", f"no 4-term sum order matches Block.hc_pre ({mism})")
            self._off("mix_stat", "hc_norm is off")

    # end-to-end verifications ({"why": str} = broken, {"mismatch": frac} = approx measurement, {} = equal) --------
    def _cmp(self, pairs, approx: bool) -> dict:
        """pairs: (name, got, want[, approx_ok]); approx_ok (default `approx`) = this output may differ (outcome
        (ii)): its mismatch fraction is measured; any other difference is a broken kernel."""
        bad, mism = [], 0.0
        for p in pairs:
            name, a, b = p[:3]
            ok_approx = p[3] if len(p) > 3 else approx
            if _same(a, b):
                continue
            if ok_approx:
                mism = max(mism, _mismatch(a, b))
            else:
                bad.append(name)
        if bad:
            return {"why": f"{', '.join(bad)} differ"}
        return {"mismatch": mism} if mism else {}

    def _verify_hc_norm(self, g) -> dict:
        o, R = self.o, self.R
        model = o.model
        layer = model.layers[0]
        D, HC, Rr = layer.attn_norm.dim, self.hc, self.rows
        apn = self.status.get("hc_norm") == "approx"
        apm = apn or self.status.get("mix_stat") == "approx"
        out = {}
        for side in ("attn", "ffn"):
            for i in range(4):
                x, pre = self._cat_rows(lambda: self._hc_inputs(g, HC, D, i))
                fn, sc, bs = self._mix_params(layer, side)
                norm = layer.attn_norm if side == "attn" else layer.ffn_norm
                with self._oracle():   # T9b-7: at R > 1 (shape policy) the 1-row reference, row by row
                    if fp32_fixed():   # T9b-5: the oracle adopts the fixed-order mixes (policy W4 4d A)
                        pr, po, cb = hc_mixes_fixed(R, layer, x, fn, sc, bs)
                    else:
                        pr, po, cb = layer.hc_mixes(x, fn, sc, bs)
                    y_r = norm(layer.hc_pre(x, pre))
                    q_r, s_r = R.act_quant(y_r, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
                p2, po2, cb2, y, xq = self.mix_pre_norm(layer, x, pre, side)
                ctx, self._ffn_ctx = self._ffn_ctx, None
                pairs = [("pre", p2, pr, apm), ("post", po2, po, apm), ("comb", cb2, cb, apm), ("y", y, y_r)]
                if xq is not None:
                    pairs += [("q", xq[0], q_r), ("s", xq[1], s_r)]
                if side == "ffn" and ctx is not None:
                    if ctx.get("xf") is not None:
                        pairs.append(("yf", ctx["xf"], y_r.float().view(Rr, -1)))
                    if ctx.get("q") is not None:
                        pairs += [("ffn q", ctx["q"], q_r), ("ffn s", ctx["s"], s_r)]
                    if ctx.get("dg"):
                        from runtime.dg_moe import e8m0_to_f32

                        b = o.dg
                        s32 = torch.empty(b.k, D // BLOCK, dtype=torch.float32, device=self.dev)
                        e8m0_to_f32(s_r.view(1, -1), s32, b.e8m0_method)
                        pairs += [("dg q", b.xq[1][:, 0], q_r.view(1, -1).expand(b.k, -1)),
                                  ("dg s", b.xs[1][:, 0], s32)]
                r = self._cmp(pairs, apn)
                if r.get("why"):
                    return {"why": f"{side}: {r['why']}"}
                if r.get("mismatch"):
                    out["mismatch"] = max(out.get("mismatch", 0.0), r["mismatch"])
        # the head: hc_pre + model.norm + head (model.py:1273-1274)
        x, pre = self._cat_rows(lambda: self._hc_inputs(g, HC, D, 1))
        with self._oracle():
            if head_uses_fixed(model):   # T9b-5 / M2-T17: the oracle's head in the fixed order
                ref = head_fixed(model, model.norm(layer.hc_pre(x, pre)), full_logits=Rr > 1)
            elif Rr > 1:   # the R-row step's head: logits of every row (static_step._head, model.py:1010-1014)
                ref = model.head(model.norm(layer.hc_pre(x, pre)), full_logits=True)
            else:
                ref = model.head(model.norm(layer.hc_pre(x, pre)))
        got = self.head(layer, x, pre)
        r = self._cmp([("head logits", got, ref)], apn)
        if r.get("why"):
            return {"why": f"head: {r['why']}"}
        if r.get("mismatch"):
            out["mismatch"] = max(out.get("mismatch", 0.0), r["mismatch"])
        return out

    def _verify_q_path(self, g) -> dict:
        R = self.R
        attn = self.o.model.layers[0].attn
        QL = attn.q_lora_rank
        ap = self.status.get("q_path") == "approx"
        for _ in range(4):
            qa = self._cat_rows(lambda: self._stream(g, 1, QL, -4, 6).view(1, 1, QL)).to(self.dev)
            with self._oracle():
                qr_r = attn.q_norm(qa)
                q_r, s_r = R.act_quant(qr_r, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            qr, (q, s) = self._q_norm_quant(attn, qa)
            r = self._cmp([("qr", qr, qr_r), ("q", q, q_r), ("s", s, s_r)], ap)
            if r:
                return r
        return {}

    def _verify_kv_path(self, g) -> dict:
        R = self.R
        attn = self.o.model.layers[0].attn
        HD, rd, win = attn.head_dim, attn.rope_head_dim, attn.window_size
        ap = self.status.get("kv_path") == "approx"
        S = attn.freqs_cis.size(0)
        Rr = self.rows   # M1-T10-PORT: R rows at positions pos .. pos+R-1 -> R ring slots
        for t in range(4):
            kvr = self._cat_rows(lambda: self._stream(g, 1, HD, -4, 6).view(1, 1, HD)).to(self.dev)
            pos = int(torch.randint(1, max(2, S - Rr + 1), (1,), generator=g, device="cpu"))
            f = attn.freqs_cis[pos:pos + Rr]
            with self._oracle():
                kv = attn.kv_norm(kvr)
                R.apply_rotary_emb(kv[..., -rd:], f)
                R.act_quant(kv, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)
            ring = torch.zeros(1, win, HD, dtype=torch.bfloat16, device=self.dev)
            slot = torch.remainder(torch.arange(pos, pos + Rr, dtype=torch.long), win).to(self.dev)
            self._kv_norm_write(attn, kvr, f, ring, slot)
            want = torch.zeros_like(ring)
            want[0, slot] = kv.view(Rr, HD)
            r = self._cmp([("ring row", ring, want)], ap)
            if r:
                return r
        return {}

    @_cpu_default
    def _router_inputs(self, g, D: int, E: int):
        x = (torch.randn(1, D, generator=g, device="cpu") * 0.5).to(torch.bfloat16)
        # tie-heavy scores for the post kernel: few distinct values, so topk sees exact ties
        sc = torch.randint(1, 5, (E,), generator=g, device="cpu").float() * 0.25
        bias = torch.where(torch.rand(E, generator=g, device="cpu") < 0.5, torch.zeros(E, device="cpu"),
                           torch.full((E,), 0.25, device="cpu"))
        return x, sc, bias

    def _verify_router(self, g) -> dict:
        o = self.o
        ap = self.status.get("router") == "approx"
        for sm in o.moes[:2]:
            moe = sm.moe
            gate = moe.gate
            E, D = gate.weight.shape
            Rr = self.rows   # M1-T10-PORT: the R-row gate ([R, D] -> [R, k]) and the R-row post kernel
            for _ in range(3):
                draws = [self._router_inputs(g, D, E) for _ in range(Rr)]
                x = torch.cat([d_[0] for d_ in draws], 0).to(self.dev)
                sc = torch.stack([d_[1] for d_ in draws]) if Rr > 1 else draws[0][1]
                bias = draws[0][2]
                with self._oracle(two_d=True):   # T9b-7: Gate.forward's [R, k] weight sum one row at a time
                    w_r, i_r = gate(x, None)
                ids_r, order = torch.sort(i_r, dim=-1)
                if Rr > 1:
                    ids, w = self.route_rows(moe, x)
                    r = self._cmp([("ids", ids, ids_r), ("w", w, w_r.gather(1, order))], ap)
                else:
                    ids, w = self.route(moe, x)
                    r = self._cmp([("ids", ids, ids_r[0]), ("w", w, w_r[0, order[0]])], ap)
                if r:
                    return r
                # the post kernel on tie-heavy scores, against the reference ops on the same topk indices
                sc, bias = sc.to(self.dev), bias.to(self.dev)
                idx = (sc + bias).view(Rr, -1).topk(gate.topk, dim=-1)[1]
                wt = sc.view(Rr, -1).gather(1, idx)
                with self._oracle(two_d=True):
                    wt /= wt.sum(dim=-1, keepdim=True) + 1e-20
                wt *= gate.route_scale
                ids_r2, order2 = torch.sort(idx, dim=-1)
                ids2 = torch.empty_like(ids_r2)
                w2 = torch.empty(Rr, gate.topk, dtype=torch.float32, device=self.dev)
                if Rr > 1:
                    launch_router_post(idx, sc.contiguous(), ids2, w2, ordd=self.orders["router"], eps=1e-20,
                                       rs=float(gate.route_scale))
                else:
                    launch_router_post(idx.view(-1), sc, ids2.view(-1), w2.view(-1), ordd=self.orders["router"],
                                       eps=1e-20, rs=float(gate.route_scale))
                r = self._cmp([("tie ids", ids2, ids_r2), ("tie w", w2, wt.gather(1, order2))], ap)
                if r:
                    return r
        return {}

    def _verify_engram(self, g) -> dict:
        R = self.R
        eng = next(l.engram for l in self.o.model.layers if l.engram is not None)
        HC, D = eng.hc_mult, eng.dim
        ap = self.status.get("engram") == "approx"
        for t in range(4):   # M1-T10-PORT: [1, R, HC, D] / [1, R, (HC+1)*D] at R rows
            x = self._cat_rows(lambda: self._stream(g, HC, D)).to(self.dev)
            kv = self._cat_rows(lambda: self._stream(g, 1, (HC + 1) * D, -4, 4).view(1, 1, -1)).to(self.dev)
            if t == 0:   # a dot of exactly 0 and one below the clamp: the copysign / clamp_min edges
                kv[..., :D] = 0
            stub = SimpleNamespace(wkv=lambda _x, kv=kv: kv, embed=lambda _h: torch.zeros(1, 1, 1, 1),
                                   hc_mult=HC, dim=D, q_weight=eng.q_weight, k_weight=eng.k_weight, eps=eng.eps,
                                   clamp_value=eng.clamp_value)
            with self._oracle():   # [1, R, HC, D]: each row's mean / sum over [1, 1, HC, D], the 1-row call
                ref = R.Engram.forward(stub, x, None, None)
            got = self._engram_from_kv(eng, x, kv)
            r = self._cmp([("engram out", got, ref)], ap)
            if r:
                return r
        return {}

    @_cpu_default
    def _index_scores(self, g, W: int):
        sc = (torch.randint(0, 8, (1, 1, W), generator=g, device="cpu").float() * 0.5).to(torch.bfloat16)
        sc[..., W // 2:] = float("-inf")                       # unreachable half (ties at -inf)
        return sc

    def _verify_indexer_post(self, g) -> dict:
        from runtime.static_step import indexer_topk_static

        o = self.o
        for r_ in o.ratios:
            W = o.W[r_]
            ix = next((l.attn.indexer for l in o.model.layers
                       if l.attn.indexer is not None and l.attn.compress_ratio == r_), None)
            if ix is None:
                continue
            k = min(ix.index_topk, W)
            Rr = self.rows   # M1-T10-PORT: [1, R, W] scores, compress_lens [R, 1] (a different length per row)
            for clen_v in (1, max(1, W // 3), W):
                sc = self._cat_rows(lambda: self._index_scores(g, W)).to(self.dev)
                if Rr > 1:
                    clen = torch.tensor([max(1, clen_v - r_) for r_ in range(Rr)], dtype=torch.long,
                                        device=self.dev).view(Rr, 1)
                else:
                    clen = torch.tensor([clen_v], dtype=torch.long, device=self.dev)
                ref = indexer_topk_static(sc, clen, k, 128)
                got = self.indexer_topk(sc, clen, k, 128)
                if not _same(got, ref):
                    return {"why": f"ratio {r_} clen {clen_v}: sorted indices differ"}
        return {}

    # ---------------------------------------------------------------- call-site operations
    def _mix_params(self, layer, side: str):
        if side == "attn":
            return layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
        return layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base

    def mix_tail(self, layer, xf, mrs, side: str, x=None):
        """hc_mixes after its statistic (model.py:956-957): F.linear(x, hc_fn) * rsqrt, then the sinkhorn kernel.
        T9b-5 ($DSV41_FP32_IMPL=fixed): the mm and the * rsqrt in ONE launch (t9b5_mix_fp32, fixed order); it reads
        the bf16 stream x [1, T, HC, D] when given (the same values as xf = x.float(): half the bytes)."""
        fn, sc, bs = self._mix_params(layer, side)
        if fp32_fixed():
            T_ = xf.numel() // xf.size(-1)
            mixes = torch.empty(1, T_, fn.size(0), dtype=torch.float32, device=xf.device)
            src = x.reshape(T_, xf.size(-1)) if x is not None else xf.reshape(T_, xf.size(-1))
            launch_fixed_dot(src, fn, mixes.view(T_, -1), site="mix", scale=mrs.reshape(T_).contiguous())
        else:
            mixes = F.linear(xf, fn) * mrs
        return self.R.hc_split_sinkhorn(mixes, sc, bs, layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps)

    def mix_pre_norm(self, layer, x, pre_mix, side: str):
        """Block.forward's hc_mixes + hc_pre + {attn,ffn}_norm (model.py:984-986 / 991-993) for one row in ONE launch
        (+ the mixes' mm, *rsqrt and sinkhorn) -> (pre, post, comb, y [1,1,D] bf16, xq): xq = (q, s) of y for the
        attention's fp8 Linears; on the ffn side the quantization (shared expert, DeepGEMM rows) and y.float() (the
        gate) are left for the MoE in `_ffn_ctx`."""
        o, R = self.o, self.R
        HC, D = x.size(-2), x.size(-1)
        T_ = x.numel() // (HC * D)          # M1-T10-PORT: T token rows (1, or the R-row step's R), one launch
        dev = x.device
        norm = layer.attn_norm if side == "attn" else layer.ffn_norm
        y = torch.empty(1, T_, D, dtype=x.dtype, device=dev)
        xf = torch.empty(1, T_, HC * D, dtype=torch.float32, device=dev)
        mix_on = self.on("mix_stat")
        mrs = torch.empty(1, T_, 1, dtype=torch.float32, device=dev) if mix_on else None
        q = torch.empty(1, T_, D, dtype=FP8, device=dev)
        s = torch.empty(1, T_, D // BLOCK, dtype=E8, device=dev)
        yf = dq = ds = None
        dg = False
        if side == "ffn":
            if self.on("router"):
                yf = torch.empty(T_, D, dtype=torch.float32, device=dev)
            if T_ == 1 and o.dg is not None and self.on("dg_act"):   # R rows: dg_act_rows scatters by the dedup
                dq, ds, dg = o.dg.xq[1][:, 0], self.dg_scale_dst(1), True
        one = T_ == 1
        launch_hc_norm(x.reshape(HC, D) if one else x.reshape(T_, HC, D),
                       pre_mix.reshape(HC).contiguous() if one else pre_mix.reshape(T_, HC).contiguous(),
                       norm.weight, y.view(-1) if one else y.view(T_, D), eps=norm.eps,
                       norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode, code_tab=self.code_tab,
                       yf=(yf.view(-1) if one else yf) if yf is not None else None,
                       xf=xf.view(-1) if one else xf.view(T_, HC * D),
                       mrs=mrs.view(-1) if mrs is not None else None, mix_ord=self.orders.get("mix"),
                       mix_eps=layer.norm_eps, q=q.view(-1) if one else q.view(T_, D),
                       s=s.view(-1) if one else s.view(T_, D // BLOCK), dq=dq, ds=ds, split=self.on("hc_split"))
        if mrs is None:   # mix_stat off: the reference statistic, on the same fp32 values (one launch fewer: .float())
            with self._oracle():   # T9b-7: at R > 1 (shape policy) each row's statistic as the 1-row step's
                mrs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.norm_eps)
        pre, post, comb = self.mix_tail(layer, xf, mrs, side, x=x if x.dtype == torch.bfloat16 else None)
        if side == "ffn":
            self._ffn_ctx = {"ptr": y.data_ptr(), "xf": yf, "q": q, "s": s, "dg": dg}
            return pre, post, comb, y, None
        return pre, post, comb, y, (q, s)

    def ffn_pre(self, x: torch.Tensor) -> Optional[dict]:
        """The ffn-side precomputation for this MoE input (mix_pre_norm), if x is the row it was made for."""
        c = self._ffn_ctx
        return c if c is not None and c["ptr"] == x.data_ptr() else None

    def ffn_done(self) -> None:
        self._ffn_ctx = None

    def linear_pre(self, xq, lin, x):
        """lin(x) for an x whose act_quant is xq = (q, s): fp8 weights take patches.fp8_linear_prequant (the same
        routing decision linear() makes, without the act_quant); any other weight dtype (fp4, bf16) runs lin(x)."""
        from runtime.patches import fp8_linear_prequant

        if xq is None or lin.weight.dtype != torch.float8_e4m3fn:
            return lin(x)
        return fp8_linear_prequant(xq[0], xq[1], lin.weight)

    def head(self, layer, h, pre_mix):
        """hc_pre + model.norm + ParallelHead (model.py:1273-1274; world_size 1) -> logits [1, V] fp32. M1-T10-PORT:
        T rows -> [1, T, V] (the R-row step's full_logits, model.py:1010-1014), one hc_norm launch."""
        model = self.o.model
        HC, D = h.size(-2), h.size(-1)
        T_ = h.numel() // (HC * D)
        y = torch.empty(1, T_, D, dtype=h.dtype, device=h.device)
        yf = torch.empty(T_, D, dtype=torch.float32, device=h.device)
        fixed = head_uses_fixed(model)   # T9b-5 / M2-T17 (a bf16-stored weight): the head in the fixed order
        #                                  (t9b5_head_fp32) on hc_norm's bf16 y (= yf's values)
        if T_ > 1:
            launch_hc_norm(h.reshape(T_, HC, D), pre_mix.reshape(T_, HC).contiguous(), model.norm.weight,
                           y.view(T_, D), eps=model.norm.eps, norm_ord=self.orders["norm_D"],
                           hc_mode=self.hc_pre_mode, yf=yf, split=self.on("hc_split"))
            if fixed:
                return head_fixed(model, y, full_logits=True)
            return F.linear(yf.view(1, T_, D), model.head.weight)   # = F.linear(x.float(), weight) at [1, T, D]
        launch_hc_norm(h.reshape(HC, D), pre_mix.reshape(HC).contiguous(), model.norm.weight, y.view(-1),
                       eps=model.norm.eps, norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode, yf=yf.view(-1),
                       split=self.on("hc_split"))
        if fixed:
            return head_fixed(model, y)
        return F.linear(yf, model.head.weight)          # = F.linear(x[:, -1].float(), weight) (model.py:1013-1014)

    def _q_norm_quant(self, attn, qa):
        """qa: wq_a's output [1, T, QL] bf16, or (T9b-4) its T-row gemv_mxfp8.SplitK (T = 1, or M1-T10-PORT R)."""
        sk = hasattr(qa, "nsplit")
        QL = qa.N if sk else qa.size(-1)
        T_ = qa.M if sk else qa.numel() // QL
        qr = torch.empty(1, T_, QL, dtype=torch.bfloat16, device=qa.device)
        q = torch.empty(1, T_, QL, dtype=FP8, device=qa.device)
        s = torch.empty(1, T_, QL // BLOCK, dtype=E8, device=qa.device)
        one = T_ == 1
        launch_norm_row(qa if sk else (qa.reshape(-1) if one else qa.reshape(T_, QL)), attn.q_norm.weight,
                        qr.view(-1) if one else qr.view(T_, QL), eps=attn.q_norm.eps, ordd=self.orders["q"],
                        code_tab=self.code_tab, q=q.view(-1) if one else q.view(T_, QL),
                        s=s.view(-1) if one else s.view(T_, QL // BLOCK))
        return qr, (q, s)

    # T9b-4 lever 2: the GEMVs that feed these fused consumers --------------------------------------------------
    def _gemv_ok(self, xq, *lins) -> bool:
        """The linear_pre / fp8_linear_prequant routing decision for x's act_quant xq = (q, s): would every one of
        `lins` take the T9b-1 GEMV here?"""
        from runtime import patches

        if xq is None or not (patches.fp8_gemv_active() and patches.fp8_linear_gemv_enabled()):
            return False
        return all(patches.fp8_gemv_reason(self.R, xq[0], l.weight) is None for l in lins)

    def _rows(self, xq) -> int:
        return xq[0].numel() // xq[0].size(-1)

    def _gemv_fold(self, xq, lin, tag: str):
        """lin(x) on the GEMV WITHOUT its split-K reduce (the consumer folds it): bf16 [M, N] or SplitK."""
        from runtime import patches
        from runtime.gemv_mxfp8 import gemv_partials

        K = xq[0].size(-1)
        patches.count_gemv_routed(lin.weight, xq[0])
        return gemv_partials(xq[0].reshape(-1, K), xq[1].reshape(-1, K // BLOCK), lin.weight, lin.weight.scale,
                             tag=tag)

    def _gemv_pair(self, xq, la, lb, tags):
        """la(x), lb(x) (same input) in one GEMV launch (gemv_mxfp8.gemv_pair); partials stay partials only with
        reduce_fold on (else each is reduced to bf16 here: one reduce launch per split half)."""
        from runtime import patches
        from runtime.gemv_mxfp8 import gemv_pair, materialize

        K = xq[0].size(-1)
        patches.count_gemv_routed(la.weight, xq[0])
        patches.count_gemv_routed(lb.weight, xq[0])
        ra, rb = gemv_pair(xq[0].reshape(-1, K), xq[1].reshape(-1, K // BLOCK), la.weight, la.weight.scale,
                           lb.weight, lb.weight.scale, tags=tags)
        if not self.on("reduce_fold"):
            ra, rb = materialize(ra), materialize(rb)
        return ra, rb

    def _q_input(self, attn, x, xq):
        """wq_a(x) for the fused q path: with gemv_pair (and the fused kv path) wq_a || wkv in one launch -- wkv's
        result is held for this layer's kv_path --, else wq_a alone (reduce folded with reduce_fold)."""
        self._kv_pend = None
        one = xq is not None   # M1-T10-PORT: any row count the GEMV takes (M <= 8); the consumers are row-generic
        if one and self.on("gemv_pair") and self.on("kv_path") and self._gemv_ok(xq, attn.wq_a, attn.wkv):
            qa, kvr = self._gemv_pair(xq, attn.wq_a, attn.wkv, ("wq_a", "wkv"))
            self._kv_pend = (id(attn), kvr)
            return qa
        if one and self.on("reduce_fold") and self._gemv_ok(xq, attn.wq_a):
            return self._gemv_fold(xq, attn.wq_a, "wq_a")
        return self.linear_pre(xq, attn.wq_a, x)

    def q_path(self, attn, x, xq):
        """qr = q_norm(wq_a(x)) (model.py:771) and act_quant(qr) for wq_b (model.py:772, 197) in one launch."""
        return self._q_norm_quant(attn, self._q_input(attn, x, xq))

    def _kv_norm_write(self, attn, kvr, freqs_cis, ring, slot):
        """kv_norm + RoPE + QDQ + the ring write of T rows (slot [T]; M1-T10-PORT: freqs_cis [T, rd/2], row t at its
        own position) in one launch."""
        HD, rd = attn.head_dim, attn.rope_head_dim
        T_ = slot.numel()
        sk = hasattr(kvr, "nsplit")
        fr = torch.view_as_real(freqs_cis)
        fr = fr.reshape(-1, 2) if T_ == 1 else fr.reshape(T_, -1, 2).contiguous()
        launch_norm_row(kvr if sk else (kvr.reshape(-1) if T_ == 1 else kvr.reshape(T_, HD)), attn.kv_norm.weight,
                        ring, eps=attn.kv_norm.eps, ordd=self.orders["kv"], code_tab=self.code_tab,
                        slot=slot.reshape(-1), qdq=True, rope=(fr, rd, self.rope_fwd) if rd else None)

    def kv_path(self, attn, x, xq, freqs_cis, slot, ring=None):
        """_window_kv's kv (model.py:706-708) into the ring slot (model.py:719): wkv, then kv_norm + RoPE + QDQ +
        the ring write in one launch. ring: the static step's own 256-slot ring (M1-T10 #20; default the reference
        window cache)."""
        pend, self._kv_pend = getattr(self, "_kv_pend", None), None
        if pend is not None and pend[0] == id(attn):                 # T9b-4: made by this layer's q path (pair)
            kvr = pend[1]
        elif xq is not None and self.on("reduce_fold") and self._gemv_ok(xq, attn.wkv):
            kvr = self._gemv_fold(xq, attn.wkv, "wkv")
        else:
            kvr = self.linear_pre(xq, attn.wkv, x)
        self._kv_norm_write(attn, kvr, freqs_cis, attn.window_kv_cache if ring is None else ring, slot)

    def route(self, moe, x):
        """Gate.forward (model.py:811-829, text: no image mask) + the static step's ascending sort -> (ids [k] int64,
        w [k] fp32): the mm on the cached fp32 weight, F.softplus, one kernel (sqrt, +bias), torch.topk, one kernel
        (gather, sum, +eps, /, *route_scale, sort)."""
        gate = moe.gate
        rp = getattr(self, "rplan", None)
        if rp is not None:                     # M2-T15 plan impl (None: not applicable here -> this path)
            r = rp.route(moe, x, rows=False)
            if r is not None:
                return r
        ctx = self.ffn_pre(x)
        xf = ctx["xf"] if ctx is not None and ctx.get("xf") is not None else x.float()
        raw = F.linear(xf, self.gate_w[id(gate)])
        if gate.gate_temp != 1.0:   # x / 1.0 is x exactly: skipped when the temperature is 1
            raw = raw / gate.gate_temp
        sp = F.softplus(raw)
        E = sp.size(-1)
        sc = torch.empty(E, dtype=torch.float32, device=x.device)
        bsc = torch.empty(1, E, dtype=torch.float32, device=x.device)
        launch_router_pre(sp.view(-1), gate.bias, sc, bsc.view(-1))
        idx = bsc.topk(gate.topk, dim=-1)[1]
        ids = torch.empty(gate.topk, dtype=torch.long, device=x.device)
        w = torch.empty(gate.topk, dtype=torch.float32, device=x.device)
        launch_router_post(idx.view(-1), sc, ids, w, ordd=self.orders["router"], eps=1e-20, rs=float(gate.route_scale))
        if os.environ.get("DSV41_ROUTER_DUMP"):   # M2-T15 recording (router_plan.DUMP_ENV; eager calls only)
            from runtime.router_plan import record

            record(self, moe, raw, ids, w, xf)
        return ids, w

    def route_rows(self, moe, x):
        """M1-T10-PORT: Gate.forward at R rows (x [R, D]) -> (ids [R, k] int64 ascending per row, w [R, k] fp32):
        the mm on the cached fp32 weight, F.softplus, one kernel (sqrt, +bias) over R*E, torch.topk, one kernel (one
        program per row: gather, sum, +eps, /, *route_scale, sort). The R-row dedup is indifferent to the order of
        a row's k choices (runtime/moe_rows.dedup_routes), so ascending ids feed it as is."""
        gate = moe.gate
        D = x.size(-1)
        T_ = x.numel() // D
        if T_ == 1:
            ids, w = self.route(moe, x)
            return ids.view(1, -1), w.view(1, -1)
        rp = getattr(self, "rplan", None)
        if rp is not None:                     # M2-T15 plan impl (the R-row plan itself: router_plan's _route hook)
            r = rp.route(moe, x, rows=True)
            if r is not None:
                return r
        ctx = self.ffn_pre(x)
        xf = ctx["xf"] if ctx is not None and ctx.get("xf") is not None else x.float()
        raw = F.linear(xf.view(T_, D), self.gate_w[id(gate)])
        if gate.gate_temp != 1.0:
            raw = raw / gate.gate_temp
        sp = F.softplus(raw).contiguous()
        E = sp.size(-1)
        sc = torch.empty(T_, E, dtype=torch.float32, device=x.device)
        bsc = torch.empty(T_, E, dtype=torch.float32, device=x.device)
        launch_router_pre(sp.view(-1), gate.bias, sc.view(-1), bsc.view(-1))
        idx = bsc.topk(gate.topk, dim=-1)[1]
        ids = torch.empty(T_, gate.topk, dtype=torch.long, device=x.device)
        w = torch.empty(T_, gate.topk, dtype=torch.float32, device=x.device)
        launch_router_post(idx.contiguous(), sc, ids, w, ordd=self.orders["router"], eps=1e-20,
                           rs=float(gate.route_scale))
        if os.environ.get("DSV41_ROUTER_DUMP"):   # M2-T15 recording (router_plan.DUMP_ENV; eager calls only)
            from runtime.router_plan import record

            record(self, moe, raw, ids, w, xf.view(T_, D))
        return ids, w

    def _engram_from_kv(self, eng, h, kv):
        """h [1, T, HC, D], kv [1, T, (HC+1)*D] (T = 1, or M1-T10-PORT R): one gate launch, sigmoid, one out launch."""
        HC, D = eng.hc_mult, eng.dim
        T_ = h.numel() // (HC * D)
        hv = h.reshape(HC, D) if T_ == 1 else h.reshape(T_, HC, D)
        z = torch.empty(1, T_, HC, dtype=torch.float32, device=h.device)
        launch_engram_gate(hv, kv.reshape(-1).contiguous(), self.engram_wp[id(eng)], z.view(-1), eps=eng.eps,
                           ordd=self.orders["engram"], sordd=self.orders["engram_sum"],
                           c=f32(eng.dim ** -0.5), cmin=f32(eng.clamp_value))
        gate = torch.sigmoid(z)                                       # model.py:362 (torch's sigmoid)
        out = torch.empty(h.shape, dtype=h.dtype, device=h.device)
        launch_engram_out(hv, kv.reshape(-1).contiguous(), gate.view(-1), out.view(hv.shape))
        return out

    def engram(self, eng, h, hash_ids):
        """Engram.forward (model.py:350-365, token_mask None): the lookup + wkv as the reference, then the gate math
        in one launch, torch.sigmoid, and the residual add in one launch."""
        kv = eng.wkv(eng.embed(hash_ids).flatten(-2))                 # model.py:353
        return self._engram_from_kv(eng, h, kv)

    def indexer_topk(self, index_score, compress_lens, k: int, offset: int):
        """static_step.indexer_topk_static with the sort + where in one launch (torch.topk kept)."""
        idx = index_score.topk(k, dim=-1, sorted=False).indices
        out = torch.empty(idx.shape, dtype=torch.int32, device=idx.device)
        T_ = idx.numel() // idx.size(-1)
        if T_ > 1:   # M1-T10-PORT: [1, R, k] indices, compress_lens [R, 1]: one launch, grid axis 1 = row
            launch_topk_post(idx.reshape(T_, -1), compress_lens.reshape(T_).contiguous(), out.view(T_, -1), offset)
            return out
        launch_topk_post(idx.reshape(-1), compress_lens.reshape(-1), out.view(-1), offset)
        return out

    # ---------------------------------------------------------------- T9b-4 self-checks (build time, eager, device)
    @contextlib.contextmanager
    def _with_parts(self, **on):
        """Temporarily force T9b-4 parts on/off (the A/B of the self-check); restores the statuses."""
        saved = {k: self.status[k] for k in on}
        try:
            for k, v in on.items():
                self.status[k] = "on" if v else "off: (self-check A/B)"
            yield
        finally:
            self.status.update(saved)

    def calibrate_r4(self, cal: dict, g) -> None:
        """Each T9b-4 part against the path it replaces, bitwise, on the device; a mismatch (or a raise) turns the
        part off with the reason (cal["r4"]). dg_sf_pack is switched on in the DeepGEMM buffers only here."""
        if not self.on("hc_norm"):
            self._off("hc_split", "hc_norm is off")
        self._r4_check(cal, "gemv", self._verify_gemv_r4, g)
        self._r4_check(cal, "dg_sf_pack", self._verify_sf_pack, g)
        b = self.o.dg
        if b is not None and hasattr(b, "sf_pack"):
            b.sf_pack = self.on("dg_sf_pack")
            b.sf_pack_why = None if b.sf_pack else self.status.get("dg_sf_pack")
        if not self.on("dg_sf_pack") and b is not None and hasattr(b, "sf_pack"):
            b.sf_pack = False

    def _r4_check(self, cal: dict, name: str, fn, g) -> None:
        r4 = cal.setdefault("r4", {})
        try:
            res = fn(g)
        except Exception as e:  # a part that does not compile/run here stays off, with the reason
            parts = ("gemv_pair", "reduce_fold") if name == "gemv" else (name,)
            res = {c: f"raised {type(e).__name__}: {str(e)[:300]}" for c in parts if self.on(c)}
        for c, why in res.items():
            r4[f"verify_{c}"] = why or "equal"
            if why:
                self._off(c, f"self-check: {why}")

    def _verify_hc_split(self, g) -> dict:
        """t9b4_hc_norm == t9b2r2_hc_norm on every output (both sides' forms, the head's form, the DeepGEMM rows in
        fp32 and -- dg-inplace -- packed form)."""
        if not (self.on("hc_split") and self.on("hc_norm")):
            return {}
        layer = self.o.model.layers[0]
        D, HC, dev = layer.attn_norm.dim, self.hc, self.dev
        mix = self.orders.get("mix") if self.on("mix_stat") else None
        b = self.o.dg
        packed = getattr(b, "xsp_rows", None) is not None and self.on("dg_sf_pack")
        for i in range(4):
            x, pre = self._hc_inputs(g, HC, D, i)
            outs = []
            for split in (False, True):
                o = dict(y=torch.empty(D, dtype=torch.bfloat16, device=dev), yf=torch.empty(D, device=dev),
                         xf=torch.empty(HC * D, device=dev), q=torch.empty(D, dtype=FP8, device=dev),
                         s=torch.empty(D // BLOCK, dtype=E8, device=dev),
                         dq=torch.zeros(3, D, dtype=FP8, device=dev),
                         ds=torch.zeros(3, D // BLOCK, dtype=torch.float32, device=dev))
                if mix is not None:
                    o["mrs"] = torch.empty(1, device=dev)
                launch_hc_norm(x.reshape(HC, D), pre.reshape(HC).contiguous(), layer.ffn_norm.weight, o["y"],
                               eps=layer.ffn_norm.eps, norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode,
                               code_tab=self.code_tab, yf=o["yf"], xf=o["xf"], mrs=o.get("mrs"), mix_ord=mix,
                               mix_eps=layer.norm_eps, q=o["q"], s=o["s"], dq=o["dq"], ds=o["ds"], split=split)
                h = dict(y=torch.empty(D, dtype=torch.bfloat16, device=dev), yf=torch.empty(D, device=dev))
                launch_hc_norm(x.reshape(HC, D), pre.reshape(HC).contiguous(), self.o.model.norm.weight, h["y"],
                               eps=self.o.model.norm.eps, norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode,
                               yf=h["yf"], split=split)
                o.update({f"head {k}": v for k, v in h.items()})
                if packed:
                    pr = b.xsp_rows[1]
                    launch_hc_norm(x.reshape(HC, D), pre.reshape(HC).contiguous(), layer.ffn_norm.weight,
                                   torch.empty(D, dtype=torch.bfloat16, device=dev), eps=layer.ffn_norm.eps,
                                   norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode, code_tab=self.code_tab,
                                   dq=b.xq[1][:, 0], ds=pr, split=split)
                    o["packed codes"] = pr.codes().clone()
                outs.append(o)
            bad = [k for k in outs[0] if not _same(outs[0][k], outs[1][k])]
            if bad:
                return {"hc_split": f"t9b4_hc_norm differs from t9b2r2_hc_norm on {bad}"}
            if packed:
                want = ((outs[0]["ds"].view(torch.int32) >> 23) & 0xFF).to(torch.uint8)[:1].expand(b.k, -1)
                if not torch.equal(outs[0]["packed codes"], want):
                    return {"dg_sf_pack": "hc_norm's packed DeepGEMM codes differ from its fp32 scales"}
        return {"hc_split": None}

    def _verify_gemv_r4(self, g) -> dict:
        """The fused q / kv / shared-expert consumers fed by the folded GEMV (reduce_fold), the pair (gemv_pair) and
        both, against the same consumers fed by linear_pre's bf16 outputs: qr, q, s, the ring row, the shared out."""
        o, R = self.o, self.R
        parts = [c for c in ("reduce_fold", "gemv_pair") if self.on(c)]
        if not parts:
            return {}
        layer = o.model.layers[0]
        attn, ex = layer.attn, layer.ffn.shared_experts
        D, HD, win = layer.attn_norm.dim, attn.head_dim, attn.window_size
        variants = []
        if self.on("reduce_fold"):
            variants.append(("reduce_fold", dict(gemv_pair=False, reduce_fold=True)))
        if self.on("gemv_pair"):
            variants.append(("gemv_pair", dict(gemv_pair=True, reduce_fold=False)))
            if self.on("reduce_fold"):
                variants.append(("gemv_pair", dict(gemv_pair=True, reduce_fold=True)))
        res = {c: None for c in parts}
        prev = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with o._active():
                for t in range(2):
                    x = self._stream(g, 1, D, -4, 6).view(1, 1, D).to(self.dev)
                    xq = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
                    if not self._gemv_ok(xq, attn.wq_a, attn.wkv, ex.w1, ex.w3):
                        return {c: "wq_a / wkv / w1 / w3 do not take the GEMV in this step's scope" for c in parts}
                    pos = 1 + t * 7
                    f = attn.freqs_cis[pos:pos + 1]
                    slot = torch.tensor([pos % win], dtype=torch.long, device=self.dev)

                    def run():
                        qr, (q, s) = self.q_path(attn, x, xq)
                        pend, self._kv_pend = self._kv_pend, None
                        if pend is not None:
                            kvr = pend[1]
                        elif self.on("reduce_fold"):
                            kvr = self._gemv_fold(xq, attn.wkv, "wkv")
                        else:
                            kvr = self.linear_pre(xq, attn.wkv, x)
                        ring = torch.zeros(1, win, HD, dtype=torch.bfloat16, device=self.dev)
                        self._kv_norm_write(attn, kvr, f, ring, slot)
                        sh = self._shared_from_q(ex, x, xq)
                        return {"qr": qr, "q": q, "s": s, "ring": ring, "shared": sh}

                    with self._with_parts(gemv_pair=False, reduce_fold=False):
                        base = run()
                    for c, st in variants:
                        if res[c]:
                            continue
                        with self._with_parts(**st):
                            got = run()
                        bad = [k for k in base if not _same(got[k], base[k])]
                        if bad:
                            res[c] = f"{bad} differ ({st})"
        finally:
            torch.set_default_dtype(prev)
            from runtime import patches
            patches.fp8_gemv_stats(reset=True)
        return res

    def _verify_sf_pack(self, g) -> dict:
        """dg_sf_pack: the fused writers' packed codes == their fp32 scales' codes, then DeepGEMM's contiguous GEMM
        on layer 0's real pool with the packed activation SF == with the fp32 one, bitwise (w1 and w2 shapes)."""
        if not self.on("dg_sf_pack"):
            return {}
        b = self.o.dg
        if b is None or not hasattr(b, "xsp_rows") or not b.layers:
            return {"dg_sf_pack": "no dg-inplace buffers / layer tables"}
        if not (self.on("dg_act") and self.on("swiglu_routed")):
            return {"dg_sf_pack": "needs dg_act and swiglu_routed on (every writer of row 0's scales)"}
        from runtime.fused_glue import _rand_act, launch_act_quant_bcast, launch_swiglu_quant

        if self.rows > 1:
            return self._verify_sf_pack_rows(g)
        k = b.k
        x = _rand_act(g, 1, b.dim).to(self.dev)
        launch_act_quant_bcast(x, b.xq[1][:, 0], b.xs[1][:, 0], self.code_tab, k)
        launch_act_quant_bcast(x, b.xq[1][:, 0], b.xsp_rows[1], self.code_tab, k)
        gate, up = self._rand_gate_up(g, k, b.inter)
        w = (torch.rand(k, generator=g, device="cpu") * 2).float().to(self.dev)
        launch_swiglu_quant(gate, up, w, b.xq[2][:, 0], b.xs[2][:, 0], self.silu_tab, self.code_tab, b.swiglu_limit)
        launch_swiglu_quant(gate, up, w, b.xq[2][:, 0], b.xsp_rows[2], self.silu_tab, self.code_tab, b.swiglu_limit)
        for wi in (1, 2):
            want = ((b.xs[wi][:, 0].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
            if not torch.equal(b.xsp_rows[wi].codes(), want):
                return {"dg_sf_pack": f"packed codes of DeepGEMM input {wi} differ from the fp32 scales"}
        t = b.layers[min(b.layers)]
        prev = b.sf_pack
        try:
            b.mi.view(k, b.align)[:, 0] = torch.arange(k, dtype=torch.int32, device=self.dev) % t["G"]
            for p_, wi in (("w1", 1), ("w2", 2)):
                b.sf_pack = False
                d32 = b.gemm_inplace(p_, t, wi).clone()
                b.sf_pack = True
                dpk = b.gemm_inplace(p_, t, wi).clone()
                if not _same(dpk, d32):
                    return {"dg_sf_pack": f"contiguous GEMM ({p_}) with the packed activation SF differs from fp32 "
                                          f"({_mismatch(dpk, d32):.3e} of outputs)"}
        finally:
            b.sf_pack = prev
            b.mi.fill_(-1)
        return {"dg_sf_pack": None}

    def _verify_sf_pack_rows(self, g) -> dict:
        """M1-T10-MOE: dg_sf_pack at R rows. The R-row writers (act_quant scatter to the pairs' A rows, the routed
        SwiGLU + act_quant at the same rows) into the packed SF via xsp_all == their fp32 scales' codes on exactly
        those rows; then the layout's routed GEMM (gemm_rows on layer 0's real pool, the layout's probe groups) with
        the packed activation SF == with the fp32 one, bitwise, w1 and w2 shapes."""
        from runtime.fused_glue import _rand_act, launch_act_quant_bcast, launch_swiglu_quant

        b = self.o.dg
        if not getattr(b, "xsp_all", None):
            return {"dg_sf_pack": "no packed activation SF in this call shape"}
        Rr, k = self.rows, self.o.topk
        P = Rr * k
        M = b.k * b.align
        dev = self.dev
        D, I = b.dim, b.inter
        heads = torch.arange(P, device="cpu") * b.align           # the GEMM probe's rows (one per block / group)
        # T9b-7: every host draw on the CPU generator AND device (the runtime's default device is cuda: a bare
        # randperm(generator=cpu) raised "Expected a 'cuda' device type for generator", dg_sf_pack off at 6 rows)
        for arow_c in (torch.randperm(M, generator=g, device="cpu")[:P],
                       heads[torch.randperm(P, generator=g, device="cpu")]):
            arow = arow_c.to(dev)
            prow = torch.randint(0, Rr, (P,), generator=g, device="cpu").to(dev)
            x = _rand_act(g, Rr, D).to(dev)
            launch_act_quant_bcast(x, b.xq[1].view(-1, D), b.xs[1].view(-1, D // BLOCK), self.code_tab, 1,
                                   src_rows=prow, dst_rows=arow)
            launch_act_quant_bcast(x, b.xq[1].view(-1, D), b.xsp_all[1], self.code_tab, 1, src_rows=prow,
                                   dst_rows=arow)
            gate, up = self._rand_gate_up(g, M, I)
            w = (torch.rand(P, generator=g, device="cpu") * 2).float().to(dev)
            launch_swiglu_quant(gate, up, w, b.xq[2].view(-1, I), b.xs[2].view(-1, I // BLOCK), self.silu_tab,
                                self.code_tab, b.swiglu_limit, rows_idx=arow)
            launch_swiglu_quant(gate, up, w, b.xq[2].view(-1, I), b.xsp_all[2], self.silu_tab, self.code_tab,
                                b.swiglu_limit, rows_idx=arow)
            for wi, K in ((1, D), (2, I)):
                xs = b.xs[wi].view(-1, K // BLOCK).index_select(0, arow).contiguous()
                want = ((xs.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
                if not torch.equal(b.xsp_all[wi].codes_at(arow), want):
                    return {"dg_sf_pack": f"R-row packed codes of DeepGEMM input {wi} differ from the fp32 scales"}
        t = b.layers[min(b.layers)]
        prev = b.sf_pack
        try:
            b._fill_layout_probe(t["G"], min(b.pairs, t["G"]))
            for p_, wi in (("w1", 1), ("w2", 2)):
                b.sf_pack = False
                d32 = b.gemm_rows(p_, t, wi).clone()
                b.sf_pack = True
                dpk = b.gemm_rows(p_, t, wi).clone()
                if not _same(dpk, d32):
                    return {"dg_sf_pack": f"R-row {b.layout} GEMM ({p_}) with the packed activation SF differs from "
                                          f"fp32 ({_mismatch(dpk, d32):.3e} of outputs)"}
        finally:
            b.sf_pack = prev
            b._reset_layout()
        return {"dg_sf_pack": None}

    def describe_r4(self) -> dict:
        from runtime.gemv_mxfp8 import PAIR_STATS

        b = self.o.dg
        return {"parts": {c: self.status.get(c) for c in ("hc_split", "gemv_pair", "reduce_fold", "dg_sf_pack")},
                "calibration": dict(self.calib.get("r4", {})) if isinstance(self.calib, dict) else {},
                "pair_launches": dict(PAIR_STATS), "hc_split_bb": HC_SPLIT_BB,
                "dg_act_sf_packed": bool(getattr(b, "sf_pack", False)) if b is not None else None}

    def describe_r2(self) -> dict:
        return {"approx": self.approx, "numerics": _jsonable(self.numerics),
                "orders": {k: v.get("name") for k, v in self.orders.items()}, "hc_pre_mode": self.hc_pre_mode,
                "order_policy": {"policy": getattr(self, "order_policy", None), "env": GLUE_ORDER_ENV,
                                 "cal_rows": getattr(self, "cal_rows", None), "step_rows": getattr(self, "rows", 1)},
                "bytes": dict(getattr(self, "r2_bytes", {})), "indexer_score": INDEXER_SCORE_WHY,
                "outcome": {c: self.numerics.get(c, {}).get("outcome") for c in R2_CLUSTERS if self.on(c)},
                "fp32": describe_fp32(), "router_impl": self._describe_router_impl()}

    def _describe_router_impl(self) -> dict:
        rp = getattr(self, "rplan", None)
        if rp is not None:
            return rp.describe()
        from runtime.router_plan import describe_default

        return describe_default()


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# ------------------------------------------------------------------------------------------------ T9b-7 census
GLUE_CENSUS_FAMILY = "fused glue"
# clusters with no per-row arithmetic of their own, or no 1-row counterpart (recorded n/a with the reason)
GLUE_CENSUS_NA = {
    "dedup": "R-row only (no 1-row counterpart): a permutation of the routes, == dedup_routes by its self-check; the "
             "step-level row covers the R-row MoE end to end",
    "gather": "data movement only (pointer-table copies, no arithmetic)",
    "concat_free": "not implemented (fused_glue.CONCAT_FREE_WHY)",
}


def _cmp_named(full: Dict[str, torch.Tensor], parts: Dict[str, List[torch.Tensor]]) -> Tuple[dict, str]:
    """row_census.compare per named output -> (the worst record with equal = all equal, 'name = / !=' note)."""
    from runtime.row_census import compare

    res = {k: compare(full[k], parts[k]) for k in full}
    worst = dict(max(res.values(), key=lambda r: (not r["equal"], r.get("max_ulp", 0) or 0)))
    worst["equal"] = all(r["equal"] for r in res.values())
    return worst, ", ".join(f"{k} {'=' if r['equal'] else '!='}" for k, r in res.items())


def glue_row_census(g1, gR, seed: int = 0) -> List[dict]:
    """T9b-7 (docs/HANDOFF-T9B7-GLUE-ORDER.md item 3): per fused-glue cluster, the R-row build's call-site operation on
    R random token rows vs the 1-row build's operation on each row alone, bitwise -> runtime/row_census.py records
    (family "fused glue <cluster>", so census_table lists each cluster as row-invariant or not). g1 / gR: FusedGlue of
    a 1-row and an R-row StaticStep of the same model (same impls). First record: the two builds' orders and modes.
    A cluster off in either build is recorded n/a with its status. The cuBLAS fp32 sites inside a cluster (the mixes'
    and the head's mm under $DSV41_FP32_IMPL=cublas, the gate's mm) and the GEMV's per-M config ($DSV41_GEMV_PIN_M1)
    are the step's, not the glue's: run with --fp32-impl fixed and the M=1 config pinned to isolate the glue. Eager;
    syncs; mutates the builds' scratch / DeepGEMM input buffers (rewritten by the next step); never fatal."""
    from runtime import fused_glue as FG
    from runtime.row_census import PIN_ENV

    R_ = int(gR.rows)
    o1, oR = g1.o, gR.o
    model = oR.model
    dev = gR.dev
    L0 = model.layers[0]
    attn, ex = L0.attn, L0.ffn.shared_experts
    HC, D = gR.hc, L0.attn_norm.dim
    g = torch.Generator(device="cpu").manual_seed(20261109 + seed)
    out: List[dict] = []

    def rec(c, op, shape, fn):
        r = {"family": f"{GLUE_CENSUS_FAMILY} {c}", "op": op, "shape": shape, "equal": None, "note": ""}
        why = None
        if c in GLUE_CENSUS_NA:
            why = GLUE_CENSUS_NA[c]
        elif c != "orders" and not (g1.on(c) and gR.on(c)):
            why = f"1-row {g1.status.get(c)!s} / {R_}-row {gR.status.get(c)!s}"
        if why is not None:
            r["note"] = f"n/a: {why}"
            out.append(r)
            return r
        try:
            with torch.inference_mode():
                cmp_, note = fn()
            r.update(cmp_)
            r["note"] = note
        except Exception as e:  # recorded, never fatal (the census convention)
            r["note"] = f"n/a: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        out.append(r)
        return r

    def act(rows, K, lo=-6, hi=7):
        return FG._rand_act(g, rows, K, lo, hi).to(dev)

    # the builds' orders and modes ---------------------------------------------------------------------------------
    def f_orders():
        a = {k: v.get("name") for k, v in g1.orders.items()}
        b = {k: v.get("name") for k, v in gR.orders.items()}
        m1 = {"hc_pre": g1.hc_pre_mode, "hc_post": g1.hc_mode, "rope_fwd": tuple(g1.rope_fwd),
              "rope_inv": tuple(g1.rope_inv)}
        mR = {"hc_pre": gR.hc_pre_mode, "hc_post": gR.hc_mode, "rope_fwd": tuple(gR.rope_fwd),
              "rope_inv": tuple(gR.rope_inv)}
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) + \
            sorted(k for k in m1 if m1[k] != mR[k])
        return ({"equal": not diff, "mismatch_pct": None, "max_ulp": None},
                f"policy {getattr(gR, 'order_policy', None)}; " + (f"differ: {diff}" if diff else f"same: {b}"))
    rec("orders", "orders + modes, 1-row vs R-row build", f"R={R_}", f_orders)

    # hc_norm (+ mix_stat, hc_split ride in it): both sides --------------------------------------------------------
    def stream(rows):
        x = FG._rand_act(g, rows * HC, D).view(1, rows, HC, D).to(dev)
        pre = (torch.rand(1, rows, HC, generator=g, device="cpu") + 0.1).float().to(dev)
        return x, pre

    def f_hc_norm():
        full, parts = {}, {}
        for side in ("attn", "ffn"):
            x, pre = stream(R_)
            pR, poR, cbR, yR, xqR = gR.mix_pre_norm(L0, x, pre, side)
            ctx = gR._ffn_ctx
            gR.ffn_done()
            outs = {"pre": pR, "post": poR, "comb": cbR, "y": yR}
            if xqR is not None:
                outs.update(q=xqR[0], s=xqR[1])
            if ctx is not None:
                outs.update({f"ffn {k}": ctx[k] for k in ("q", "s", "xf") if ctx.get(k) is not None})
            for k, v in outs.items():
                full[f"{side} {k}"] = v.reshape(R_, -1)
                parts[f"{side} {k}"] = []
            for r in range(R_):
                p1, po1, cb1, y1, xq1 = g1.mix_pre_norm(L0, x[:, r:r + 1].contiguous(), pre[:, r:r + 1].contiguous(),
                                                        side)
                c1 = g1._ffn_ctx
                g1.ffn_done()
                o1_ = {"pre": p1, "post": po1, "comb": cb1, "y": y1}
                if xq1 is not None:
                    o1_.update(q=xq1[0], s=xq1[1])
                if c1 is not None:
                    o1_.update({f"ffn {k}": c1[k] for k in ("q", "s", "xf") if c1.get(k) is not None})
                for k in outs:
                    parts[f"{side} {k}"].append(o1_[k].reshape(1, -1))
        c, note = _cmp_named(full, parts)
        _hc.update(full=full, parts=parts)
        return c, f"mix_stat {gR.status.get('mix_stat')}, hc_split {gR.status.get('hc_split')}, fp32 {fp32_impl()}; " \
                  + note
    _hc: dict = {}
    rec("hc_norm", "mix_pre_norm attn+ffn", f"[1,{R_},{HC},{D}]", f_hc_norm)

    def f_rides(keys):
        def f():
            if not _hc:
                return {"equal": None}, "n/a: the hc_norm probe did not run"
            ks = [k for k in _hc["full"] if k.split(" ", 1)[1] in keys]
            c, note = _cmp_named({k: _hc["full"][k] for k in ks}, {k: _hc["parts"][k] for k in ks})
            return c, "rides in the hc_norm launch (same probe); " + note
        return f
    rec("mix_stat", "hc_mixes statistic -> pre/post/comb", f"[1,{R_},{HC * D}]", f_rides(("pre", "post", "comb")))
    rec("hc_split", "t9b4_hc_norm (split programs)", f"[1,{R_},{HC},{D}]",
        f_rides(("pre", "post", "comb", "y", "q", "s", "ffn q", "ffn s", "ffn xf")))

    def f_head():
        x, pre = stream(R_)
        full = gR.head(L0, x, pre)
        parts = [g1.head(L0, x[:, r:r + 1].contiguous(), pre[:, r:r + 1].contiguous()) for r in range(R_)]
        return _cmp_named({"logits": full.reshape(R_, -1)}, {"logits": [p.reshape(1, -1) for p in parts]})
    rec("hc_norm", "head (hc_pre + norm + head)", f"[1,{R_},{HC},{D}] x V", f_head)

    # q path / kv path --------------------------------------------------------------------------------------------
    def f_q():
        QL = attn.q_lora_rank
        qa = act(R_, QL, -4, 6).view(1, R_, QL)
        qr, (q, s) = gR._q_norm_quant(attn, qa)
        full = {"qr": qr.reshape(R_, -1), "q": q.reshape(R_, -1), "s": s.reshape(R_, -1)}
        parts = {k: [] for k in full}
        for r in range(R_):
            qr1, (q1, s1) = g1._q_norm_quant(attn, qa[:, r:r + 1].contiguous())
            for k, v in (("qr", qr1), ("q", q1), ("s", s1)):
                parts[k].append(v.reshape(1, -1))
        return _cmp_named(full, parts)
    rec("q_path", "q_norm + act_quant", f"[1,{R_},{attn.q_lora_rank}]", f_q)

    def f_kv():
        HD, win = attn.head_dim, attn.window_size
        S = attn.freqs_cis.size(0)
        kvr = act(R_, HD, -4, 6).view(1, R_, HD)
        pos = int(torch.randint(1, max(2, S - R_ + 1), (1,), generator=g, device="cpu"))
        f = attn.freqs_cis[pos:pos + R_]
        slot = torch.remainder(torch.arange(pos, pos + R_, dtype=torch.long, device="cpu"), win).to(dev)
        ring = torch.zeros(1, win, HD, dtype=torch.bfloat16, device=dev)
        gR._kv_norm_write(attn, kvr, f, ring, slot)
        parts = []
        for r in range(R_):
            ring1 = torch.zeros(1, win, HD, dtype=torch.bfloat16, device=dev)
            g1._kv_norm_write(attn, kvr[:, r:r + 1].contiguous(), f[r:r + 1], ring1, slot[r:r + 1])
            parts.append(ring1[0, slot[r:r + 1]].reshape(1, -1))
        return _cmp_named({"ring rows": ring[0, slot].reshape(R_, -1)}, {"ring rows": parts})
    rec("kv_path", "kv_norm + RoPE + QDQ + ring write", f"[1,{R_},{attn.head_dim}]", f_kv)

    # router ------------------------------------------------------------------------------------------------------
    def f_router():
        moe = oR.moes[0].moe
        x = (torch.randn(R_, D, generator=g, device="cpu") * 0.5).to(torch.bfloat16).to(dev)
        ids, w = gR.route_rows(moe, x)
        parts = {"ids": [], "w": []}
        for r in range(R_):
            i1, w1 = g1.route(moe, x[r:r + 1].contiguous())
            parts["ids"].append(i1.reshape(1, -1))
            parts["w"].append(w1.reshape(1, -1))
        c, note = _cmp_named({"ids": ids.reshape(R_, -1), "w": w.reshape(R_, -1)}, parts)
        return c, f"router_impl {getattr(gR, 'router_impl', 'fused')} (gate mm cuBLAS fp32 at the call's M); {note}"
    rec("router", "gate mm + softplus + pre + topk + post", f"[{R_},{D}]", f_router)

    # Engram ------------------------------------------------------------------------------------------------------
    eng = next((l.engram for l in model.layers if l.engram is not None), None)

    def f_engram():
        h = FG._rand_act(g, R_ * HC, D).view(1, R_, HC, D).to(dev)
        kv = FG._rand_act(g, R_, (HC + 1) * D, -4, 4).view(1, R_, -1).to(dev)
        full = gR._engram_from_kv(eng, h, kv)
        parts = [g1._engram_from_kv(eng, h[:, r:r + 1].contiguous(), kv[:, r:r + 1].contiguous()).reshape(1, -1)
                 for r in range(R_)]
        return _cmp_named({"out": full.reshape(R_, -1)}, {"out": parts})
    if eng is not None:
        rec("engram", "gate math + sigmoid + residual add", f"[1,{R_},{HC},{D}]", f_engram)

    # indexer post ------------------------------------------------------------------------------------------------
    def f_index():
        full, parts = {}, {}
        for r_ in oR.ratios:
            ix = next((l.attn.indexer for l in model.layers
                       if l.attn.indexer is not None and l.attn.compress_ratio == r_), None)
            if ix is None:
                continue
            W = oR.W[r_]
            k = min(ix.index_topk, W)
            sc = (torch.randint(0, 8, (1, R_, W), generator=g, device="cpu").float() * 0.5).to(torch.bfloat16)
            sc[..., W // 2:] = float("-inf")
            sc = sc.to(dev)
            clen = torch.tensor([max(1, W // 3 - j) for j in range(R_)], dtype=torch.long, device="cpu").to(dev)
            full[f"r{r_}"] = gR.indexer_topk(sc, clen.view(R_, 1), k, 128).reshape(R_, -1)
            parts[f"r{r_}"] = [g1.indexer_topk(sc[:, j:j + 1].contiguous(), clen[j:j + 1], k, 128).reshape(1, -1)
                               for j in range(R_)]
        if not full:
            return {"equal": None}, "n/a: no index-source layer"
        return _cmp_named(full, parts)
    rec("indexer_post", "topk + sort + where", f"[1,{R_},W]", f_index)

    # R1: hc_post, o path, q RoPE epilogue -----------------------------------------------------------------------
    def f_hc_post():
        x, res, post, comb = gR._hc_data(g, D, R_)
        full = gR.hc_post(x, res, post, comb)
        parts = [g1.hc_post(x[:, r:r + 1].contiguous(), res[:, r:r + 1].contiguous(), post[:, r:r + 1].contiguous(),
                            comb[:, r:r + 1].contiguous()).reshape(1, -1) for r in range(R_)]
        return _cmp_named({"out": full.reshape(R_, -1)}, {"out": parts})
    rec("hc_post", "post*x + comb*residual sum", f"[1,{R_},{HC},{D}]", f_hc_post)

    def f_o():
        H, hd = attn.n_local_heads, attn.head_dim
        S = attn.freqs_cis.size(0)
        o = act(R_, H * hd).view(1, R_, H, hd)
        pos = int(torch.randint(1, max(2, S - R_ + 1), (1,), generator=g, device="cpu"))
        f = attn.freqs_cis[pos:pos + R_]
        full = o.clone()
        gR.o_path(attn, full, f)
        parts = []
        for r in range(R_):
            o1_ = o[:, r:r + 1].clone()
            g1.o_path(attn, o1_, f[r:r + 1])
            parts.append(o1_.reshape(1, -1))
        return _cmp_named({"o": full.reshape(R_, -1)}, {"o": parts})
    rec("o_rope_qdq", "inverse RoPE + QDQ", f"[1,{R_},{attn.n_local_heads},{attn.head_dim}]", f_o)

    def f_qproj():
        from runtime import patches

        K = attn.wq_b.in_features
        S = attn.freqs_cis.size(0)
        prev = torch.get_default_dtype()
        prev_pin = os.environ.get(PIN_ENV)
        os.environ[PIN_ENV] = "1"          # the GEMV's M=1 config at every M (the headline's pin)
        try:
            torch.set_default_dtype(torch.bfloat16)
            qr = act(R_, K, -4, 4).view(1, R_, K)
            pos = int(torch.randint(1, max(2, S - R_ + 1), (1,), generator=g, device="cpu"))
            f = attn.freqs_cis[pos:pos + R_]
            with oR._active():
                full = gR.q_proj(attn, qr, f)
            parts = []
            with o1._active():
                for r in range(R_):
                    p = g1.q_proj(attn, qr[:, r:r + 1].contiguous(), f[r:r + 1])
                    parts.append(None if p is None else p.reshape(1, -1))
            if full is None or any(p is None for p in parts):
                return {"equal": None}, "n/a: wq_b does not take the GEMV in this scope"
            c, note = _cmp_named({"q": full.reshape(R_, -1)}, {"q": parts})
            return c, f"GEMV M=1 config pinned; {note}"
        finally:
            torch.set_default_dtype(prev)
            if prev_pin is None:
                os.environ.pop(PIN_ENV, None)
            else:
                os.environ[PIN_ENV] = prev_pin
            patches.fp8_gemv_stats(reset=True)
    rec("q_rope_epilogue", "wq_b GEMV + RoPE epilogue", f"[1,{R_},{attn.wq_b.in_features}]", f_qproj)

    def f_gemv_parts():
        """gemv_pair / reduce_fold: the q path and the shared expert fed by the paired / folded GEMV (wq_a || wkv,
        w1 || w3), M=1 config pinned, R rows vs each row alone."""
        from runtime import patches

        Rm = gR.R
        prev = torch.get_default_dtype()
        prev_pin = os.environ.get(PIN_ENV)
        os.environ[PIN_ENV] = "1"
        try:
            torch.set_default_dtype(torch.bfloat16)
            x = act(R_, D, -4, 6).view(1, R_, D)
            xq = Rm.act_quant(x, Rm.fp8_block_size, Rm.scale_fmt, Rm.scale_dtype)

            def run(glue, o_, xx, qq):
                with o_._active():
                    qr, (q, s) = glue.q_path(attn, xx, qq)
                    pend, glue._kv_pend = getattr(glue, "_kv_pend", None), None
                    sh = glue._shared_from_q(ex, xx, qq)
                d = {"qr": qr, "q": q, "s": s, "shared": sh}
                if pend is not None:
                    from runtime.gemv_mxfp8 import materialize

                    d["kv (pair)"] = materialize(pend[1])
                return d

            full = run(gR, oR, x, xq)
            parts = {k: [] for k in full}
            for r in range(R_):
                p = run(g1, o1, x[:, r:r + 1].contiguous(), (xq[0][:, r:r + 1].contiguous(),
                                                              xq[1][:, r:r + 1].contiguous()))
                for k in full:
                    parts[k].append(p[k].reshape(1, -1))
            c, note = _cmp_named({k: v.reshape(R_, -1) for k, v in full.items()}, parts)
            return c, f"pair {gR.status.get('gemv_pair')}, fold {gR.status.get('reduce_fold')}; M=1 pinned; {note}"
        finally:
            torch.set_default_dtype(prev)
            if prev_pin is None:
                os.environ.pop(PIN_ENV, None)
            else:
                os.environ[PIN_ENV] = prev_pin
            patches.fp8_gemv_stats(reset=True)
    for c_ in ("gemv_pair", "reduce_fold"):
        rec(c_, "q path + shared expert on the paired / folded GEMV", f"[1,{R_},{D}]", f_gemv_parts)

    # MoE glue kernels: shared SwiGLU, routed SwiGLU (pair rows), act_quant into the DeepGEMM rows, combine, packed SF

    def f_sw_shared():
        I = ex.w1.out_features
        gate, up = gR._rand_gate_up(g, R_, I)
        q = torch.empty(R_, I, dtype=FP8, device=dev)
        s = torch.empty(R_, I // BLOCK, dtype=E8, device=dev)
        FG.launch_swiglu_quant(gate, up, None, q, s, gR.silu_tab, gR.code_tab, ex.swiglu_limit)
        parts = {"q": [], "s": []}
        for r in range(R_):
            q1 = torch.empty(1, I, dtype=FP8, device=dev)
            s1 = torch.empty(1, I // BLOCK, dtype=E8, device=dev)
            FG.launch_swiglu_quant(gate[r:r + 1], up[r:r + 1], None, q1, s1, g1.silu_tab, g1.code_tab, ex.swiglu_limit)
            parts["q"].append(q1)
            parts["s"].append(s1)
        return _cmp_named({"q": q, "s": s}, parts)
    rec("swiglu_shared", "SwiGLU + act_quant (shared)", f"[{R_},{ex.w1.out_features}]", f_sw_shared)

    bR, b1 = getattr(oR, "dg", None), getattr(o1, "dg", None)

    def pairs():
        P = R_ * oR.topk
        A = bR.xq[1].numel() // bR.dim
        prow = torch.randint(0, R_, (P,), generator=g, device="cpu").to(dev)
        arow = torch.randperm(A, generator=g, device="cpu")[:P].to(dev)
        return P, A, prow, arow

    def f_sw_routed():
        P, A, _, arow = pairs()
        I = bR.inter
        gate, up = gR._rand_gate_up(g, A, I)
        w = (torch.rand(P, generator=g, device="cpu") * 2).float().to(dev)
        q = torch.zeros(A, I, dtype=FP8, device=dev)
        s = torch.zeros(A, I // BLOCK, dtype=torch.float32, device=dev)
        FG.launch_swiglu_quant(gate, up, w, q, s, gR.silu_tab, gR.code_tab, bR.swiglu_limit, rows_idx=arow)
        parts = {"q": [], "s": []}
        for p in range(P):
            a = int(arow[p])
            q1 = torch.empty(1, I, dtype=FP8, device=dev)
            s1 = torch.empty(1, I // BLOCK, dtype=torch.float32, device=dev)
            FG.launch_swiglu_quant(gate[a:a + 1], up[a:a + 1], w[p:p + 1], q1, s1, g1.silu_tab, g1.code_tab,
                                   b1.swiglu_limit)
            parts["q"].append(q1)
            parts["s"].append(s1)
        return _cmp_named({"q": q.index_select(0, arow), "s": s.index_select(0, arow)}, parts)
    rec("swiglu_routed", "SwiGLU + act_quant (pair rows vs one row)", "R*k pairs", f_sw_routed)

    def f_dg_act():
        P, A, prow, arow = pairs()
        Dd = bR.dim
        x = act(R_, Dd)
        q = torch.zeros(A, Dd, dtype=FP8, device=dev)
        s = torch.zeros(A, Dd // BLOCK, dtype=torch.float32, device=dev)
        FG.launch_act_quant_bcast(x, q, s, gR.code_tab, 1, src_rows=prow, dst_rows=arow)
        parts = {"q": [], "s": []}
        k1 = b1.k
        for p in range(P):
            r = int(prow[p])
            q1 = torch.empty(k1, Dd, dtype=FP8, device=dev)
            s1 = torch.empty(k1, Dd // BLOCK, dtype=torch.float32, device=dev)
            FG.launch_act_quant_bcast(x[r:r + 1], q1, s1, g1.code_tab, k1)
            parts["q"].append(q1[:1])
            parts["s"].append(s1[:1])
        return _cmp_named({"q": q.index_select(0, arow), "s": s.index_select(0, arow)}, parts)
    rec("dg_act", "act_quant into the DeepGEMM rows", f"[{R_},{getattr(bR, 'dim', '?')}]", f_dg_act)

    def f_combine():
        k = oR.topk
        A = 2 * R_ * k
        N = bR.dim
        rows = FG._rand_act(g, A, N, lo=-8, hi=8).to(dev)
        sh = FG._rand_act(g, R_, N, lo=-8, hi=8).to(dev)
        posmat = torch.stack([torch.randperm(A, generator=g, device="cpu")[:k] for _ in range(R_)]).to(dev)
        y = torch.empty(R_, N, dtype=torch.bfloat16, device=dev)
        FG.launch_moe_combine(rows, sh, y, posmat=posmat)
        parts = []
        for r in range(R_):
            y1 = torch.empty(N, dtype=torch.bfloat16, device=dev)
            FG.launch_moe_combine(rows.index_select(0, posmat[r]).contiguous(), sh[r].contiguous(), y1)
            parts.append(y1.view(1, -1))
        return _cmp_named({"y": y}, {"y": parts})
    rec("moe_combine", "ascending-id combine + shared", f"[{R_},{getattr(bR, 'dim', '?')}]", f_combine)

    def f_sf_pack():
        if not (getattr(bR, "sf_pack", False) and getattr(b1, "sf_pack", False)):
            return {"equal": None}, (f"n/a: packed SF not in use (1-row {getattr(b1, 'sf_pack', None)}, "
                                     f"{R_}-row {getattr(bR, 'sf_pack', None)})")
        P, A, prow, arow = pairs()
        Dd = bR.dim
        KB = Dd // BLOCK
        x = act(R_, Dd)
        FG.launch_act_quant_bcast(x, bR.xq[1].view(-1, Dd), bR.xsp_all[1], gR.code_tab, 1, src_rows=prow,
                                  dst_rows=arow)
        full = bR.xsp_all[1].codes_at(arow)[:, :KB].contiguous()
        parts = []
        for p in range(P):
            r = int(prow[p])
            FG.launch_act_quant_bcast(x[r:r + 1], b1.xq[1][:, 0], b1.xsp_rows[1], g1.code_tab, b1.k)
            parts.append(b1.xsp_rows[1].codes()[:1, :KB].contiguous())
        return _cmp_named({"packed codes": full}, {"packed codes": parts})
    rec("dg_sf_pack", "packed DeepGEMM activation SF codes", f"[{R_},{getattr(bR, 'dim', '?')}]", f_sf_pack)

    for c in ("dedup", "gather", "concat_free"):
        rec(c, "-", "-", lambda: ({"equal": None}, ""))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return out
