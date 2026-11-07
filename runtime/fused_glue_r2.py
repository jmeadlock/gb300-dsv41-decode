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
CPU (the Mac tests, the census): the wrappers run torch mirrors; with no device to calibrate on, the statistics run
torch's own op (order "native"), so the CPU step is bit-identical by construction; the explicit-order mirrors and the
Triton bodies (under tests/triton_shim.py) are tested against each other and against the family's definition.
"""
from __future__ import annotations

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


def sqmean_torch(v: torch.Tensor, w: int, ordd: dict, shape=None) -> torch.Tensor:
    """mean(v^2) over the last dim of fp32 v [N, w] in the picked order; NATIVE = torch's op on `shape` (the
    reference call's shape, so its launch configuration is the reference's)."""
    if ordd["order"] == NATIVE:
        x = v.reshape(shape) if shape is not None else v
        return x.square().mean(-1).reshape(-1)
    return mean_finish(order_sum_torch(v * v, ordd["order"]), w, ordd["fin"])


def sum_torch(v: torch.Tensor, ordd: dict, shape=None) -> torch.Tensor:
    if ordd["order"] == NATIVE:
        x = v.reshape(shape) if shape is not None else v
        return x.sum(-1).reshape(-1)
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
    """Block.hc_pre (model.py:961-962) for one row: x [HC, D] bf16, pre [HC] fp32 -> bf16 [D], the 4-term sum in
    fused_glue.HC_SUM_MODES order `mode` (0 = ((0 + a0) + a1) + ...)."""
    HC, D = x.shape
    a = [pre[i].float() * x[i].float() for i in range(HC)]
    if mode == 0:
        acc = torch.zeros(D, dtype=torch.float32, device=x.device)
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
    (torch.topk's order), sc [E] fp32 -> (ids [K] int64 ascending, w [K] fp32 in that order)."""
    wv = sc[idx].view(1, -1)
    den = sum_torch(wv, ordd, (1, wv.size(-1))).view(1, 1) + eps
    wn = (wv / den) * rs
    ids, order = torch.sort(idx)
    return ids, wn.view(-1)[order]


def engram_gate_torch(h: torch.Tensor, key: torch.Tensor, wp: torch.Tensor, eps: float, ordd: dict, sordd: dict,
                      c: float, cmin: float) -> torch.Tensor:
    """Engram.forward's gate input (model.py:356-362, before the sigmoid): h, key fp32 [HC, D] (x.float(),
    key.float()), wp fp32 [HC, D] = q_weight.float() * k_weight.float() -> [HC] fp32 copysign(sqrt(clamp(|dot|)))."""
    HC, D = h.shape
    shape = (1, 1, HC, D)
    rh = torch.rsqrt(sqmean_torch(h, D, ordd, shape) + eps)
    rk = torch.rsqrt(sqmean_torch(key, D, ordd, shape) + eps)
    dot = sum_torch((h * wp) * key, sordd, shape) * (rh * rk) * c
    return torch.copysign(dot.abs().clamp_min(cmin).sqrt(), dot)


def indexer_post_torch(idx: torch.Tensor, clen: torch.Tensor, offset: int) -> torch.Tensor:
    s = idx.sort(dim=-1).values
    return torch.where(s < clen, s + offset, -1).int()


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
    def t9b2r2_hc_norm(x, pre, nw, y, yf, xf, mrs, q, s, dq, ds, dq_gs, ds_gs, G, tab,
                       eps, fac, wf, meps, mfac, mwf,
                       D: tl.constexpr, HC: tl.constexpr, HCMODE: tl.constexpr,
                       NT: tl.constexpr, NV: tl.constexpr, NU: tl.constexpr, NDEC: tl.constexpr, NDIV: tl.constexpr,
                       MT: tl.constexpr, MV: tl.constexpr, MU: tl.constexpr, MDEC: tl.constexpr, MDIV: tl.constexpr,
                       PASS_A: tl.constexpr, HAS_MIX: tl.constexpr, HAS_XF: tl.constexpr, HAS_YF: tl.constexpr,
                       QQ: tl.constexpr, HAS_Q: tl.constexpr, HAS_DG: tl.constexpr, BB: tl.constexpr):
        """One program. x [HC, D] bf16 (the residual stream copies), pre [HC] fp32 (pre_mix), nw [D] (norm weight).
        A (PASS_A): xf = x.flatten().float() (HAS_XF) and mrs = rsqrt(mean(xf^2) + meps) in order M* (HAS_MIX:
        hc_mixes' statistic, model.py:954-955). B: y0 = hc_pre(x, pre) (bf16), var = mean(y0^2) in order N*,
        r = rsqrt(var + eps). C, per 32-block: y = bf16(nw * (y0 * r)) -> y (bf16), yf (fp32, HAS_YF), and (QQ) its
        act_quant: q fp8 + s e8m0 code (HAS_Q), the same q into rows g*dq_gs of dq and fp32 scales into ds (HAS_DG:
        DeepGEMM's w1/w3 input, G groups)."""
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
                        tl.store(ds + gi * ds_gs + bi, _t9b2_pow2(code), mask=bok)

    # --- #3 / #6
    @triton.jit
    def t9b2r2_norm_row(x, nw, out, slot, q, s, freqs, tab, eps, fac, wf,
                        N: tl.constexpr, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr,
                        DIV: tl.constexpr, HAS_SLOT: tl.constexpr, QQ: tl.constexpr, HAS_Q: tl.constexpr,
                        QDQ: tl.constexpr, RD: tl.constexpr, RE_MODE: tl.constexpr, IM_MODE: tl.constexpr,
                        BB: tl.constexpr):
        """One program: RMSNorm of one bf16 row x [N] (model.py:288-293, statistic in order T/V/U/DEC/DIV), then per
        32-block as even/odd pairs: RoPE on the last RD columns (RD > 0; the forward rotation by freqs [RD/2, 2],
        contraction forms RE/IM_MODE, bf16), act_quant QDQ in place (QDQ) -> out (+ slot[0] * N: the window ring
        row, HAS_SLOT); and/or the act_quant of the row: q fp8 + s e8m0 (HAS_Q)."""
        acc = tl.zeros([T, V], dtype=tl.float32)
        for k in range((N + T * V - 1) // (T * V)):
            e = _r2_offs(k, T, V, U)
            m = e < N
            v = tl.load(x + e, mask=m, other=0.0).to(tl.float32)
            acc = acc + v * v
        r = libdevice.rsqrt(_r2_fin(_r2_finish(acc, T, V, DEC), fac, wf, DIV) + eps)
        if HAS_SLOT:
            base = out + tl.load(slot) * N
        else:
            base = out
        for b0 in range(0, N // 32, BB):
            bi = b0 + tl.arange(0, BB)
            bok = bi < N // 32
            ce = bi[:, None] * 32 + 2 * tl.arange(0, 16)[None, :]
            m = bok[:, None] & (ce >= 0)
            a = _t9b2_bf16(tl.load(nw + ce, mask=m, other=0.0).to(tl.float32)
                           * (tl.load(x + ce, mask=m, other=0.0).to(tl.float32) * r))
            b = _t9b2_bf16(tl.load(nw + ce + 1, mask=m, other=0.0).to(tl.float32)
                           * (tl.load(x + ce + 1, mask=m, other=0.0).to(tl.float32) * r))
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
    def t9b2r2_router_pre(sp, bias, sc, bsc, E, BLOCK_E: tl.constexpr):
        """scores = sqrt(softplus) (IEEE sqrt), biased = scores + bias (model.py:819, 824)."""
        i = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        m = i < E
        v = libdevice.sqrt_rn(tl.load(sp + i, mask=m, other=0.0))
        tl.store(sc + i, v, mask=m)
        tl.store(bsc + i, v + tl.load(bias + i, mask=m, other=0.0), mask=m)

    @triton.jit
    def t9b2r2_router_post(idx, sc, ids_out, w_out, K, rs, eps,
                           T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr, KP: tl.constexpr):
        """One program: weights = scores[idx] (torch.topk's order), w = weights / (sum + eps) * route_scale
        (model.py:825-828, the sum in order T/V/U/DEC), then ascending by id (distinct ids: rank = how many are
        smaller) -> ids_out int64, w_out fp32."""
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
                           SDEC: tl.constexpr):
        """Program i (hc copy): h = x[i].float(), key = kv[i*D:(i+1)*D].float(); rstd = rsqrt(mean(h^2) + eps) *
        rsqrt(mean(key^2) + eps) (order T/V/U/DEC/DIV); dot = sum((h * wp[i]) * key) (order S*) * rstd * c;
        z[i] = copysign(sqrt(max(|dot|, cmin)), dot) (model.py:357-362 up to the sigmoid)."""
        i = tl.program_id(0)
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
        """Program (pb, i): out[i] = bf16(h[i] + gate[i] * value) (model.py:365), value = kv[HC*D:]."""
        pb = tl.program_id(0)
        i = tl.program_id(1)
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
        and out[rank] = idx < clen ? idx + offset : -1 as int32 (model.py:580-581)."""
        p = tl.program_id(0)
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


def launch_hc_norm(x, pre, nw, y, *, eps: float, norm_ord: dict, hc_mode: int, code_tab=None, yf=None, xf=None,
                   mrs=None, mix_ord: Optional[dict] = None, mix_eps: float = 0.0, q=None, s=None, dq=None, ds=None):
    """x [HC, D] bf16 contiguous, pre [HC] fp32, nw [D]; y [D] bf16; yf [D] fp32; xf [HC*D] fp32; mrs [1] fp32;
    q [D] fp8, s [D/32] e8m0; dq [G, D] fp8 rows (unit inner stride), ds [G, D/32] fp32."""
    HC, D = x.shape
    if _kern(x):
        na = order_args(norm_ord)
        ma = order_args(mix_ord) if mrs is not None else dict(T=128, V=4, U=False, DEC=True, DIV=False)
        dummy = y
        G = dq.size(0) if dq is not None else 0
        su = s.view(torch.uint8) if s is not None else dummy
        t9b2r2_hc_norm[(1,)](
            x, pre, nw, y, yf if yf is not None else dummy, xf if xf is not None else dummy,
            mrs if mrs is not None else dummy, q if q is not None else dummy, su,
            dq if dq is not None else dummy, ds if ds is not None else dummy,
            dq.stride(0) if dq is not None else 0, ds.stride(0) if ds is not None else 0, G,
            code_tab if code_tab is not None else dummy,
            float(eps), fac_of(D), float(D), float(mix_eps), fac_of(HC * D), float(HC * D),
            D=D, HC=HC, HCMODE=int(hc_mode),
            NT=na["T"], NV=na["V"], NU=na["U"], NDEC=na["DEC"], NDIV=na["DIV"],
            MT=ma["T"], MV=ma["V"], MU=ma["U"], MDEC=ma["DEC"], MDIV=ma["DIV"],
            PASS_A=xf is not None or mrs is not None, HAS_MIX=mrs is not None, HAS_XF=xf is not None,
            HAS_YF=yf is not None, QQ=q is not None or dq is not None, HAS_Q=q is not None, HAS_DG=dq is not None,
            BB=8, num_warps=NORM_WARPS, **LAUNCH)
        return
    xv = x.float()
    if xf is not None:
        xf.copy_(xv.reshape(-1))
    if mrs is not None:
        src = xf.view(1, -1) if xf is not None else xv.reshape(1, -1)
        mrs.copy_(rms_rsqrt_torch(src, mix_eps, mix_ord, (1, 1, HC * D)).view(-1))
    y0 = hc_pre_torch(x, pre, hc_mode).float().view(1, D)
    o = rms_apply_torch(y0, rms_rsqrt_torch(y0, eps, norm_ord, (1, 1, D)), nw)
    y.copy_(o.view(-1))
    if yf is not None:
        yf.copy_(o.float().view(-1))
    if q is not None or dq is not None:
        qv, code = act_quant_torch(o.float(), code_tab)
        if q is not None:
            q.copy_(qv.view(-1))
            s.view(torch.uint8).copy_(code.view(-1).to(torch.uint8))
        if dq is not None:
            for gi in range(dq.size(0)):
                dq[gi].copy_(qv.view(-1))
                ds[gi].copy_(pow2_code(code.view(-1)))


def launch_norm_row(x, nw, out, *, eps: float, ordd: dict, code_tab=None, slot=None, q=None, s=None,
                    qdq: bool = False, rope=None):
    """x [N] bf16, nw [N]; out bf16 [N] (or the ring [win, N] with slot [1] int64); q [N] fp8 + s [N/32] e8m0;
    rope = (freqs fp32 [RD/2, 2], RD, (re, im) modes)."""
    N = x.numel()
    RD = int(rope[1]) if rope is not None else 0
    modes = tuple(rope[2]) if rope is not None else (0, 0)
    if _kern(x):
        a = order_args(ordd)
        dummy = out
        t9b2r2_norm_row[(1,)](
            x, nw, out, slot if slot is not None else x, q if q is not None else dummy,
            s.view(torch.uint8) if s is not None else dummy, rope[0] if rope is not None else nw,
            code_tab if code_tab is not None else dummy, float(eps), fac_of(N), float(N),
            N=N, T=a["T"], V=a["V"], U=a["U"], DEC=a["DEC"], DIV=a["DIV"], HAS_SLOT=slot is not None,
            QQ=q is not None or qdq, HAS_Q=q is not None, QDQ=bool(qdq), RD=RD, RE_MODE=int(modes[0]),
            IM_MODE=int(modes[1]), BB=min(16, _np2(N // 32)), num_warps=NORM_WARPS, **LAUNCH)
        return
    v = x.float().view(1, N)
    o = rms_apply_torch(v, rms_rsqrt_torch(v, eps, ordd, (1, 1, N)), nw).view(-1)
    if RD:
        from runtime.fused_glue import rope_rows_torch

        o = rope_rows_torch(o, rope[0], N, RD, modes, inverse=False)
    if q is not None or qdq:
        qv, code = act_quant_torch(o.float(), code_tab)
        if q is not None:
            q.copy_(qv)
            s.view(torch.uint8).copy_(code.to(torch.uint8))
        if qdq:
            o = (qv.float().unflatten(-1, (-1, BLOCK)) * pow2_code(code).unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    dst = out.view(-1, N)[int(slot.reshape(-1)[0])] if slot is not None else out.view(-1)
    dst.copy_(o)


def launch_router_pre(sp, bias, sc, bsc):
    """sp, bias, sc, bsc: [E] fp32 contiguous."""
    E = sp.numel()
    if _kern(sp):
        BLOCK_E = min(1024, _np2(E))
        t9b2r2_router_pre[(triton.cdiv(E, BLOCK_E),)](sp, bias, sc, bsc, E, BLOCK_E=BLOCK_E, **LAUNCH)
        return
    v = sp.sqrt()
    sc.copy_(v)
    bsc.copy_(v + bias)


def launch_router_post(idx, sc, ids_out, w_out, *, ordd: dict, eps: float, rs: float):
    """idx [K] int64 (topk's order), sc [E] fp32 -> ids_out [K] int64 ascending, w_out [K] fp32."""
    K = idx.numel()
    if _kern(idx):
        a = order_args(ordd)
        t9b2r2_router_post[(1,)](idx, sc, ids_out, w_out, K, float(rs), float(eps), T=a["T"], V=a["V"], U=a["U"],
                                 DEC=a["DEC"], KP=_np2(K), **LAUNCH)
        return
    ids, w = router_post_torch(idx, sc, ordd, eps, rs)
    ids_out.copy_(ids)
    w_out.copy_(w)


def launch_engram_gate(x, kv, wp, z, *, eps: float, ordd: dict, sordd: dict, c: float, cmin: float):
    """x [HC, D] bf16, kv [(HC+1)*D] bf16, wp [HC, D] fp32 -> z [HC] fp32."""
    HC, D = x.shape
    if _kern(x):
        a, sa = order_args(ordd), order_args(sordd)
        t9b2r2_engram_gate[(HC,)](x, kv, wp, z, float(eps), fac_of(D), float(D), float(c), float(cmin), D=D,
                                  T=a["T"], V=a["V"], U=a["U"], DEC=a["DEC"], DIV=a["DIV"], ST=sa["T"], SV=sa["V"],
                                  SU=sa["U"], SDEC=sa["DEC"], num_warps=NORM_WARPS, **LAUNCH)
        return
    key = kv[: HC * D].float().view(HC, D)
    z.copy_(engram_gate_torch(x.float(), key, wp, eps, ordd, sordd, c, cmin))


ENGRAM_BLOCK = 1024


def launch_engram_out(x, kv, gate, out):
    """x [HC, D] bf16, kv [(HC+1)*D] bf16, gate [HC] fp32 -> out [HC, D] bf16."""
    HC, D = x.shape
    if _kern(x):
        t9b2r2_engram_out[(triton.cdiv(D, ENGRAM_BLOCK), HC)](x, kv, gate, out, D=D, HC=HC, BLOCK_C=ENGRAM_BLOCK,
                                                             **LAUNCH)
        return
    v = kv[HC * D:].float().view(1, D)
    out.copy_((x.float() + gate.view(HC, 1) * v).to(torch.bfloat16))


TOPK_RB = 16      # [RB, next_pow2(K)] compare tile per program: 16 x 512 = 64 values per thread at 4 warps


def launch_topk_post(idx, clen, out, offset: int):
    """idx [K] int64 distinct, clen [1] int64 -> out [K] int32 ascending, -1 past clen, + offset."""
    K = idx.numel()
    if _kern(idx):
        RB = min(TOPK_RB, _np2(K))
        t9b2r2_topk_post[(triton.cdiv(K, RB),)](idx, clen, out, K, int(offset), KP=_np2(K), RB=RB, **LAUNCH)
        return
    out.copy_(indexer_post_torch(idx, clen, offset))


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
        o = self.o
        if self.on("router"):
            for sm in o.moes:
                gate = sm.moe.gate
                w = gate.weight
                self.gate_w[id(gate)] = w if w.dtype == torch.float32 else w.detach().float().contiguous()
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
        self.orders.update(norm_D=d(D), mix=d(self.hc * D), kv=d(l0.attn.head_dim), q=d(l0.attn.q_lora_rank),
                           router=d(l0.ffn.gate.topk))
        eng = next((l.engram for l in model.layers if l.engram is not None), None)
        if eng is not None:
            self.orders.update(engram=d(eng.dim, eng.hc_mult), engram_sum=d(eng.dim, eng.hc_mult))

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

    # ---------------------------------------------------------------- status helpers
    def _r2_norm(self, key: str) -> dict:
        return self.orders[key]

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
        r2 = cal.setdefault("r2", {})
        if self.on("hc_norm"):
            rec = self.orders["norm_D"] = pick_order(dev, D, 1, "sqmean")
            self._set_outcome(("hc_norm", "mix_stat"), rec, f"RMSNorm[{D}]")
            self._pick_hc_pre(r2, g)
        if self.on("mix_stat"):
            rec = self.orders["mix"] = pick_order(dev, self.hc * D, 1, "sqmean")
            self._set_outcome(("mix_stat",), rec, f"hc_mixes[{self.hc * D}]")
        if self.on("kv_path"):
            HD = l0.attn.head_dim
            rec = self.orders["kv"] = pick_order(dev, HD, 1, "sqmean")
            self._set_outcome(("kv_path",), rec, f"kv_norm[{HD}]")
        if self.on("q_path"):
            QL = l0.attn.q_lora_rank
            rec = self.orders["q"] = pick_order(dev, QL, 1, "sqmean")
            self._set_outcome(("q_path",), rec, f"q_norm[{QL}]")
        if self.on("router"):
            K = l0.ffn.gate.topk
            rec = self.orders["router"] = pick_order(dev, K, 1, "possum")
            self._set_outcome(("router",), rec, f"topk-weight sum[{K}]")
        if self.on("engram"):
            eng = next(l.engram for l in layers if l.engram is not None)
            rec = self.orders["engram"] = pick_order(dev, eng.dim, eng.hc_mult, "sqmean")
            self._set_outcome(("engram",), rec, f"Engram mean[{eng.hc_mult}x{eng.dim}]")
            rec2 = self.orders["engram_sum"] = pick_order(dev, eng.dim, eng.hc_mult, "sum")
            self._set_outcome(("engram",), rec2, f"Engram dot[{eng.hc_mult}x{eng.dim}]")
        if not self.on("hc_norm"):
            self._off("mix_stat", "needs hc_norm (its statistic rides in the hc_norm launch)")
        r2["orders"] = {k: {kk: (order_name(vv) if kk == "order" else vv) for kk, vv in v.items()}
                        for k, v in self.orders.items()}
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
        for c in R2_CLUSTERS:
            if self.on(c) and c not in self.numerics:
                self.numerics[c] = {"outcome": "(i) bit-identical", "orders": {}}

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
        D, HC = layer.attn_norm.dim, self.hc
        apn = self.status.get("hc_norm") == "approx"
        apm = apn or self.status.get("mix_stat") == "approx"
        out = {}
        for side in ("attn", "ffn"):
            for i in range(4):
                x, pre = self._hc_inputs(g, HC, D, i)
                fn, sc, bs = self._mix_params(layer, side)
                norm = layer.attn_norm if side == "attn" else layer.ffn_norm
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
                        pairs.append(("yf", ctx["xf"], y_r.float().view(1, -1)))
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
        x, pre = self._hc_inputs(g, HC, D, 1)
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
            qa = self._stream(g, 1, QL, -4, 6).view(1, 1, QL).to(self.dev)
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
        for t in range(4):
            kvr = self._stream(g, 1, HD, -4, 6).view(1, 1, HD).to(self.dev)
            pos = int(torch.randint(1, max(2, S), (1,), generator=g, device="cpu"))
            f = attn.freqs_cis[pos:pos + 1]
            kv = attn.kv_norm(kvr)
            R.apply_rotary_emb(kv[..., -rd:], f)
            R.act_quant(kv, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)
            ring = torch.zeros(1, win, HD, dtype=torch.bfloat16, device=self.dev)
            slot = torch.tensor([pos % win], dtype=torch.long, device=self.dev)
            self._kv_norm_write(attn, kvr, f, ring, slot)
            want = torch.zeros_like(ring)
            want[0, pos % win] = kv.view(-1)
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
            for _ in range(3):
                x, sc, bias = self._router_inputs(g, D, E)
                x = x.to(self.dev)
                w_r, i_r = gate(x, None)
                ids_r, order = torch.sort(i_r[0])
                ids, w = self.route(moe, x)
                r = self._cmp([("ids", ids, ids_r), ("w", w, w_r[0, order])], ap)
                if r:
                    return r
                # the post kernel on tie-heavy scores, against the reference ops on the same topk indices
                sc, bias = sc.to(self.dev), bias.to(self.dev)
                idx = (sc + bias).view(1, -1).topk(gate.topk, dim=-1)[1]
                wt = sc.view(1, -1).gather(1, idx)
                wt /= wt.sum(dim=-1, keepdim=True) + 1e-20
                wt *= gate.route_scale
                ids_r2, order2 = torch.sort(idx[0])
                ids2 = torch.empty_like(ids_r2)
                w2 = torch.empty(gate.topk, dtype=torch.float32, device=self.dev)
                launch_router_post(idx.view(-1), sc, ids2, w2, ordd=self.orders["router"], eps=1e-20,
                                   rs=float(gate.route_scale))
                r = self._cmp([("tie ids", ids2, ids_r2), ("tie w", w2, wt[0, order2])], ap)
                if r:
                    return r
        return {}

    def _verify_engram(self, g) -> dict:
        R = self.R
        eng = next(l.engram for l in self.o.model.layers if l.engram is not None)
        HC, D = eng.hc_mult, eng.dim
        ap = self.status.get("engram") == "approx"
        for t in range(4):
            x = self._stream(g, HC, D).to(self.dev)
            kv = self._stream(g, 1, (HC + 1) * D, -4, 4).view(1, 1, -1).to(self.dev)
            if t == 0:   # a dot of exactly 0 and one below the clamp: the copysign / clamp_min edges
                kv[..., :D] = 0
            stub = SimpleNamespace(wkv=lambda _x, kv=kv: kv, embed=lambda _h: torch.zeros(1, 1, 1, 1),
                                   hc_mult=HC, dim=D, q_weight=eng.q_weight, k_weight=eng.k_weight, eps=eng.eps,
                                   clamp_value=eng.clamp_value)
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
            for clen_v in (1, max(1, W // 3), W):
                sc = self._index_scores(g, W).to(self.dev)
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

    def mix_tail(self, layer, xf, mrs, side: str):
        """hc_mixes after its statistic (model.py:956-957): F.linear(x, hc_fn) * rsqrt, then the sinkhorn kernel."""
        fn, sc, bs = self._mix_params(layer, side)
        mixes = F.linear(xf, fn) * mrs
        return self.R.hc_split_sinkhorn(mixes, sc, bs, layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps)

    def mix_pre_norm(self, layer, x, pre_mix, side: str):
        """Block.forward's hc_mixes + hc_pre + {attn,ffn}_norm (model.py:984-986 / 991-993) for one row in ONE launch
        (+ the mixes' mm, *rsqrt and sinkhorn) -> (pre, post, comb, y [1,1,D] bf16, xq): xq = (q, s) of y for the
        attention's fp8 Linears; on the ffn side the quantization (shared expert, DeepGEMM rows) and y.float() (the
        gate) are left for the MoE in `_ffn_ctx`."""
        o, R = self.o, self.R
        HC, D = x.size(-2), x.size(-1)
        dev = x.device
        norm = layer.attn_norm if side == "attn" else layer.ffn_norm
        y = torch.empty(1, 1, D, dtype=x.dtype, device=dev)
        xf = torch.empty(1, 1, HC * D, dtype=torch.float32, device=dev)
        mix_on = self.on("mix_stat")
        mrs = torch.empty(1, 1, 1, dtype=torch.float32, device=dev) if mix_on else None
        q = torch.empty(1, 1, D, dtype=FP8, device=dev)
        s = torch.empty(1, 1, D // BLOCK, dtype=E8, device=dev)
        yf = dq = ds = None
        dg = False
        if side == "ffn":
            if self.on("router"):
                yf = torch.empty(1, D, dtype=torch.float32, device=dev)
            if o.dg is not None and self.on("dg_act"):
                dq, ds, dg = o.dg.xq[1][:, 0], o.dg.xs[1][:, 0], True
        launch_hc_norm(x.reshape(HC, D), pre_mix.reshape(HC).contiguous(), norm.weight, y.view(-1), eps=norm.eps,
                       norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode, code_tab=self.code_tab,
                       yf=yf.view(-1) if yf is not None else None, xf=xf.view(-1),
                       mrs=mrs.view(-1) if mrs is not None else None, mix_ord=self.orders.get("mix"),
                       mix_eps=layer.norm_eps, q=q.view(-1), s=s.view(-1), dq=dq, ds=ds)
        if mrs is None:   # mix_stat off: the reference statistic, on the same fp32 values (one launch fewer: .float())
            mrs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.norm_eps)
        pre, post, comb = self.mix_tail(layer, xf, mrs, side)
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
        """hc_pre + model.norm + ParallelHead (model.py:1273-1274; world_size 1) -> logits [1, V] fp32."""
        model = self.o.model
        HC, D = h.size(-2), h.size(-1)
        y = torch.empty(1, 1, D, dtype=h.dtype, device=h.device)
        yf = torch.empty(1, D, dtype=torch.float32, device=h.device)
        launch_hc_norm(h.reshape(HC, D), pre_mix.reshape(HC).contiguous(), model.norm.weight, y.view(-1),
                       eps=model.norm.eps, norm_ord=self.orders["norm_D"], hc_mode=self.hc_pre_mode, yf=yf.view(-1))
        return F.linear(yf, model.head.weight)          # = F.linear(x[:, -1].float(), weight) (model.py:1013-1014)

    def _q_norm_quant(self, attn, qa):
        QL = qa.size(-1)
        qr = torch.empty(1, 1, QL, dtype=torch.bfloat16, device=qa.device)
        q = torch.empty(1, 1, QL, dtype=FP8, device=qa.device)
        s = torch.empty(1, 1, QL // BLOCK, dtype=E8, device=qa.device)
        launch_norm_row(qa.reshape(-1), attn.q_norm.weight, qr.view(-1), eps=attn.q_norm.eps, ordd=self.orders["q"],
                        code_tab=self.code_tab, q=q.view(-1), s=s.view(-1))
        return qr, (q, s)

    def q_path(self, attn, x, xq):
        """qr = q_norm(wq_a(x)) (model.py:771) and act_quant(qr) for wq_b (model.py:772, 197) in one launch."""
        qa = self.linear_pre(xq, attn.wq_a, x)
        return self._q_norm_quant(attn, qa)

    def _kv_norm_write(self, attn, kvr, freqs_cis, ring, slot):
        HD, rd = attn.head_dim, attn.rope_head_dim
        launch_norm_row(kvr.reshape(-1), attn.kv_norm.weight, ring, eps=attn.kv_norm.eps, ordd=self.orders["kv"],
                        code_tab=self.code_tab, slot=slot, qdq=True,
                        rope=(torch.view_as_real(freqs_cis).reshape(-1, 2), rd, self.rope_fwd) if rd else None)

    def kv_path(self, attn, x, xq, freqs_cis, slot, ring=None):
        """_window_kv's kv (model.py:706-708) into the ring slot (model.py:719): wkv, then kv_norm + RoPE + QDQ +
        the ring write in one launch. ring: the static step's own 256-slot ring (M1-T10 #20; default the reference
        window cache)."""
        kvr = self.linear_pre(xq, attn.wkv, x)
        self._kv_norm_write(attn, kvr, freqs_cis, attn.window_kv_cache if ring is None else ring, slot)

    def route(self, moe, x):
        """Gate.forward (model.py:811-829, text: no image mask) + the static step's ascending sort -> (ids [k] int64,
        w [k] fp32): the mm on the cached fp32 weight, F.softplus, one kernel (sqrt, +bias), torch.topk, one kernel
        (gather, sum, +eps, /, *route_scale, sort)."""
        gate = moe.gate
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
        return ids, w

    def _engram_from_kv(self, eng, h, kv):
        HC, D = eng.hc_mult, eng.dim
        z = torch.empty(1, 1, HC, dtype=torch.float32, device=h.device)
        launch_engram_gate(h.reshape(HC, D), kv.reshape(-1), self.engram_wp[id(eng)], z.view(-1), eps=eng.eps,
                           ordd=self.orders["engram"], sordd=self.orders["engram_sum"],
                           c=f32(eng.dim ** -0.5), cmin=f32(eng.clamp_value))
        gate = torch.sigmoid(z)                                       # model.py:362 (torch's sigmoid)
        out = torch.empty(h.shape, dtype=h.dtype, device=h.device)
        launch_engram_out(h.reshape(HC, D), kv.reshape(-1), gate.view(-1), out.view(HC, D))
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
        launch_topk_post(idx.reshape(-1), compress_lens.reshape(-1), out.view(-1), offset)
        return out

    def describe_r2(self) -> dict:
        return {"approx": self.approx, "numerics": _jsonable(self.numerics),
                "orders": {k: v.get("name") for k, v in self.orders.items()}, "hc_pre_mode": self.hc_pre_mode,
                "bytes": dict(getattr(self, "r2_bytes", {})), "indexer_score": INDEXER_SCORE_WHY,
                "outcome": {c: self.numerics.get(c, {}).get("outcome") for c in R2_CLUSTERS if self.on(c)}}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
