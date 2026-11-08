"""T9b-2 round 1: the zero/low-risk glue clusters of the 1-row static step as fused kernels (`--glue-impl fused`),
bit-identical to `ref` (today's static path) by construction and by a device self-check at build time.

    st = StaticStep(model, pool, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused")
    st.glue.describe()     # which clusters are on, the numerics modes the self-check picked, why any is off

Spec: docs/HANDOFF-T9B2-GLUE-R1.md; cluster numbers and op counts are docs/GLUE-AUDIT.md's table. Each cluster is a
switch; a cluster that is not applicable to the StaticStep's impls is "n/a", one that fails its self-check or is named
in $DSV41_GLUE_DISABLE (comma list) is "off" and its call site runs today's ops unchanged.

  cluster          audit  replaces (per instance)                                            -> launches
  gather           #11+#9 6 x t8_gather_rows (+ dg: 3 x e8m0->fp32 convert_scales, ids_step copy) -> 1
                          (the converts and the ids copy run inside the same launch; static_kernels.gather_tables)
  dg_act           #9     act_quant(x) + load_act(1) q/s copies                                    -> 1
  swiglu_routed    #2+#9  .float x2, clamp x2, silu, * , weights *, .to(bf16), act_quant(h), load_act(2) -> 1
  swiglu_shared    #2     .float x2, clamp x2, silu, *, .to(bf16), act_quant(h) inside w2's linear -> 1 (+ the GEMV)
  moe_combine      #8     zeros_like, k x `+=`, `+= shared`, type_as                              -> 1
  hc_post          #7     `post*x`, `comb*residual`, sum over hc, `+`, type_as                     -> 1
  o_rope_qdq       #10    inverse RoPE (.float, conj clone, complex mul, copy_) + act_quant QDQ + copy_ -> 1
  q_rope_epilogue  #13    q RoPE (.float, complex mul, copy_) folded into the wq_b GEMV's store    -> 0
  concat_free      #16    not in round 1 (see CLUSTER_DOC)

How bit-identity is obtained (no approximation anywhere; every rounding point is the reference's):
  * act_quant (model.py:187 -> kernel.py act_quant, block 32, ue8m0): the e8m0 scale of a block depends only on the
    block's amax, which is a bf16 value. The scale is therefore READ from a 65,536-entry table that the reference
    act_quant itself produced at build time, one row per bf16 bit pattern (`scale_code_table`): no constant (1/448)
    or log2/ceil formula is re-derived. The values are x * 2^-(code-127) (exact: a power-of-two scaling, the same
    value as the kernel's x / s), the clamp to +-448 (a no-op: amax <= 448 s), and fp8 e4m3 round-to-nearest-even;
    QDQ = bf16(fp32(q) * s) (exact). The self-check compares against the reference on every finite bf16 value.
  * SwiGLU (model.py:845-853): gate and up are bf16 GEMM outputs `.float()`ed, so silu's input is a bf16 value (or
    swiglu_limit after the clamp). silu is READ from a table of torch's own `F.silu` over all 65,536 bf16 values plus
    the limit (`silu_table`), so it is torch's silu bit for bit whatever libm/FMA form torch compiled. The clamps
    are compares (NaN-propagating like torch.clamp); `silu*up`, `w*h` are single fp32 multiplies; `.to(bf16)` RNE.
  * MoE combine (model.py:895-906): fp32 `0 + out[0] + ... + out[k-1] + shared` in slot order, then bf16 RNE.
  * mHC post (model.py:964-968): post[j]*x + sum_i comb[i,j]*res[i] in fp32; the 4-term sum's order is the one
    torch's reduction uses, picked by the self-check from HC_SUM_MODES (0: ((0+a0)+a1)+a2)+a3, 1: (a0+a1)+(a2+a3),
    2: (a0+a2)+(a1+a3)) on random data against the real `Block.hc_post`.
  * RoPE (model.py:392-407): torch multiplies complex64 as (a*c - b*d, a*d + b*c), compiled by nvcc with FMA
    contraction allowed, so each component is one of three forms (ROPE_MODES: 0 plain, 1 fma(a,c,-(b*d)) /
    fma(a,d,b*c), 2 fma(-b,d,a*c) / fma(b,c,a*d)). The self-check evaluates all three with a probe kernel (explicit
    `mul.rn`/`fma.rn`, never contracted) against `apply_rotary_emb` on random and cancellation-crafted inputs with
    the model's own freqs rows, forward (q, #13) and inverse (o, #10) separately, and uses the matching form. The
    rotated value is rounded to bf16 (the reference's `y.copy_(x)`) before anything else reads it.
  * Every Triton kernel here is launched with enable_fp_fusion=False (no mul+add contraction), and the RoPE helpers
    use `mul.rn/add.rn/sub.rn` PTX (never fused by LLVM or ptxas) so the same helper is exact inside the GEMV
    epilogue, which keeps the GEMV's own fusion setting (its accumulation numerics are T9b-1's, unchanged).
  * Self-check (`FusedGlue.calibrate`, at StaticStep build, eager, syncs, before any capture): tables; act_quant vs
    the reference on all finite bf16 values (+ random rows); the RoPE and hc-sum modes; then every enabled cluster's
    fused op vs its reference op sequence on random data at the model's shapes (q_rope_epilogue with the real wq_b
    weights). A mismatch turns that cluster off (reason in describe()), it never approximates.

CPU (the Mac tests, the census): the same entry points run a torch mirror of each kernel (same algorithm, same table
lookups, same modes; `fma_f32` is an exact fp32 fma), so the CPU tests exercise the call sites, buffers and tables;
the kernel bodies run on the Mac through tests/triton_shim.py.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from runtime import pdl as _pdl   # M2-L5: PDL switch ($DSV41_PDL, default off)

def _cpu_default(fn):
    """Synthetic-input builders were written under a CPU default device; at runtime the default device is cuda.
    Run them with a CPU default so every bare constructor (tensor/arange/zeros/rand) lands on the CPU; callers move."""
    import functools
    @functools.wraps(fn)
    def w(*a, **k):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.device("cpu"):
                return fn(*a, **k)
        finally:
            torch.set_default_dtype(prev)
    return w


try:  # GPU image only; the Mac uses the torch mirrors
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = tl = None
    HAVE_TRITON = False

GLUE_IMPLS = ("ref", "fused")   # --glue-impl: ref = today's static path (unchanged) / fused = this module
R1_CLUSTERS = ("gather", "dg_act", "swiglu_routed", "swiglu_shared", "moe_combine", "hc_post", "o_rope_qdq",
               "q_rope_epilogue", "concat_free")
# T9b-2 round 2 (runtime/fused_glue_r2.py): the medium-risk clusters
R2_CLUSTER_NAMES = ("hc_norm", "mix_stat", "kv_path", "q_path", "router", "engram", "indexer_post")
# T9b-4 (docs/HANDOFF-T9B4-RESIDUAL.md): sub-switches of existing clusters / impls, each bit-identical or data movement
# only and self-checked at build like the rest (DSV41_GLUE_DISABLE names them for A/B and bisecting)
R4_PARTS = ("hc_split", "gemv_pair", "reduce_fold", "dg_sf_pack")
# M1-T10-PORT: the R-row MoE's dedup (runtime/moe_rows.dedup_fused: sort-by-id + unique + slot/rank + posmat +
# masked_m / m_indices tags + the counters in one launch); rows > 1 only
M1_PARTS = ("dedup",)
CLUSTERS = R1_CLUSTERS + R2_CLUSTER_NAMES + R4_PARTS + M1_PARTS
CLUSTER_AUDIT = {"gather": "#11+#9", "dg_act": "#9", "swiglu_routed": "#2+#9", "swiglu_shared": "#2",
                 "moe_combine": "#8", "hc_post": "#7", "o_rope_qdq": "#10", "q_rope_epilogue": "#13",
                 "concat_free": "#16", "hc_norm": "#1", "mix_stat": "#5", "kv_path": "#3", "q_path": "#6",
                 "router": "#4 (non-topk)", "engram": "#15", "indexer_post": "#14 (non-topk)",
                 "hc_split": "T9b-4 L1", "gemv_pair": "T9b-4 L2a", "reduce_fold": "T9b-4 L2b",
                 "dg_sf_pack": "T9b-4 L4", "dedup": "M1-T10-PORT #3"}
DEDUP_MAX_PAIRS = 64          # runtime/moe_rows.DEDUP_MAX_PAIRS (one-program dedup kernel)
CONCAT_FREE_WHY = ("not implemented in round 1: the window ring is per layer while the compressed cache and the "
                   "published topk_idxs are shared by every layer of a kv/index group, so one buffer per layer "
                   "needs either per-layer copies of the compressed rows (no launch saved) or re-allocating the "
                   "model's caches into per-group pools with per-layer index offsets; deferred, 76 launches")
ACT_CLUSTERS = ("dg_act", "swiglu_routed", "swiglu_shared", "o_rope_qdq",   # need the act_quant table parity
                "hc_norm", "kv_path", "q_path")
DISABLE_ENV = "DSV41_GLUE_DISABLE"
# M1-T10-FIX / M1-T10-PORT: the clusters that run at rows > 1 (the 6-row verify step). Every kernel has a row axis
# (grid axis or one program per row) and keeps the one-row kernel's per-element arithmetic and, per row, its reduction
# order. T9b-7 (fused_glue_r2 "T9b-7: one order per shape"): that order is the 1-row build's (picked at the 1-row
# reference shapes, $DSV41_GLUE_ORDER=shape, default) and each cluster is self-checked at R rows against the
# reference glue evaluated one row at a time; `rows` restores M1-T10-PORT's protocol (orders picked again for the
# R-row reference shapes, checked against the reference glue at R rows). The R-row MoE's
# routed glue runs through the dedup map (runtime/moe_rows.py: dedup_fused, then the IDX forms of the act_quant /
# SwiGLU / combine kernels). Not ported, with the reason: ROWS_WHY.
ROWS_WHY = {
    "concat_free": CONCAT_FREE_WHY,
}
# M1-T10-MOE: dg_sf_pack is ported to R rows -- the R-row writers (dg_act_rows, routed_swiglu_rows) store the codes
# through a PackedRows over EVERY A row (dg_inplace.DGInplaceBuffers.xsp_all: row r = A row r, the dedup's arow), and
# fused_glue_r2._verify_sf_pack runs the R-row A/B self-check (codes == the fp32 scales' codes on the pairs' rows,
# then the layout's GEMM with packed vs fp32 SF, bitwise). The masked call shape keeps fp32 SF (n/a there).
ROWS_CLUSTERS = tuple(c for c in CLUSTERS if c not in ROWS_WHY)
NBF16 = 65536                   # bf16 bit patterns (table rows)
SILU_LIM = NBF16                # silu table entry holding silu(swiglu_limit)
ROPE_MODES = (0, 1, 2)
HC_SUM_MODES = (0, 1, 2)
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BLOCK = 32                      # act_quant block (model.py fp8_block_size)
LAUNCH = dict(enable_fp_fusion=False)


# ------------------------------------------------------------------------------------------------ exact fp32 helpers
def fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fma(a, b, c) = RN(a*b + c) with ONE rounding, exactly (torch has no fma op): a*b is exact in float64; the
    float64 sum s and its exact error e (TwoSum) represent a*b + c exactly; RN to fp32 of s is the answer unless s
    is exactly halfway between two fp32 neighbours, where e decides (e == 0: ties-to-even, which s.float() did)."""
    a, b, c = (t.to(torch.float32) for t in torch.broadcast_tensors(a, b, c))
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bb = s - p
    e = (p - (s - bb)) + (cd - bb)
    r = s.float()
    rd = r.double()
    inf = torch.full_like(r, float("inf"))
    n = torch.nextafter(r, torch.where(s > rd, inf, -inf))
    mid = ((rd + n.double()) * 0.5 == s) & (rd != s)
    take_n = mid & (e != 0) & ((e > 0) == (s > rd))
    return torch.where(take_n, n, r)


def rope_re(a, b, c, d, mode: int):
    """Real part of (a + ib)(c + id) in the given contraction form (module docstring)."""
    if mode == 0:
        return a * c - b * d
    if mode == 1:
        return fma_f32(a, c, -(b * d))
    return fma_f32(-b, d, a * c)


def rope_im(a, b, c, d, mode: int):
    if mode == 0:
        return a * d + b * c
    if mode == 1:
        return fma_f32(a, d, b * c)
    return fma_f32(b, c, a * d)


def bf16_bits_all(device=None) -> torch.Tensor:
    """bf16 [65536]: every bit pattern, in bit order."""
    return torch.arange(NBF16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device or "cpu")


def pow2_code(code: torch.Tensor) -> torch.Tensor:
    """fp32 2^(code-127) for e8m0 codes 1..254 (int tensor)."""
    return (code.to(torch.int32) << 23).view(torch.float32)


def inv_pow2_code(code: torch.Tensor) -> torch.Tensor:
    """fp32 2^(127-code) for codes 1..253: the exact reciprocal of pow2_code."""
    return ((254 - code.to(torch.int32)) << 23).view(torch.float32)


def e8m0_decode(code: torch.Tensor) -> torch.Tensor:
    """fp32 value of e8m0 codes (0 -> 2^-127 subnormal, 255 -> NaN) == torch's e8m0 cast for 0..254."""
    c = code.to(torch.int32)
    bits = torch.where(c == 0, torch.full_like(c, 0x400000), c << 23)
    return torch.where(c == 255, torch.full_like(c, 0x7FC00000), bits).view(torch.float32)


# ------------------------------------------------------------------------------------------------ tables
def scale_code_table(R, device) -> torch.Tensor:
    """uint8 [65536]: the e8m0 code the REFERENCE act_quant (block 32, ue8m0) gives a block whose amax has bf16 bits
    i (row i = [bf16(i), 0 x 31]). Exhaustive over every finite amax; the kernels index it by amax's bf16 bits. Rows
    of non-finite patterns (an inf/NaN amax: never a real activation block) hold the code of amax 0."""
    v = bf16_bits_all()
    v = torch.where(torch.isfinite(v.float()), v, torch.zeros_like(v))
    x = torch.zeros(NBF16, BLOCK, dtype=torch.bfloat16, device=device)
    x[:, 0] = v.to(device)
    _, s = R.act_quant(x, BLOCK, R.scale_fmt, R.scale_dtype)
    return s.reshape(NBF16).view(torch.uint8).contiguous()


def silu_table(device, lim: float) -> torch.Tensor:
    """fp32 [65537]: torch's F.silu of every bf16 value (as fp32, bit order), then silu(fp32(lim)) at SILU_LIM."""
    v = torch.cat([bf16_bits_all(device).float(), torch.tensor([float(lim)], dtype=torch.float32, device=device)])
    return F.silu(v).contiguous()


def table_index(v: torch.Tensor) -> torch.Tensor:
    """bf16 bit pattern (0..65535) of bf16-valued fp32 v."""
    return ((v.contiguous().view(torch.int32) >> 16) & 0xFFFF).long()


# ------------------------------------------------------------------------------------------------ torch mirrors
def act_quant_torch(v: torch.Tensor, code_tab: torch.Tensor):
    """The kernels' act_quant on bf16-valued fp32 v [..., K] -> (q fp8 [..., K], codes int32 [..., K/32])."""
    blk = v.float().unflatten(-1, (-1, BLOCK))
    amax = blk.abs().amax(-1)
    code = code_tab[table_index(amax)].to(torch.int32)
    q = (blk * inv_pow2_code(code).unsqueeze(-1)).clamp(-448.0, 448.0).to(FP8)
    return q.flatten(-2), code


def qdq_torch(v: torch.Tensor, code_tab: torch.Tensor) -> torch.Tensor:
    """act_quant(..., inplace=True)'s value: bf16(fp32(q) * s) of bf16-valued fp32 v."""
    q, code = act_quant_torch(v, code_tab)
    return (q.float().unflatten(-1, (-1, BLOCK)) * pow2_code(code).unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def swiglu_torch(gate_bf16, up_bf16, w, silu_tab, lim: float):
    """Fused SwiGLU value (bf16-valued fp32) from bf16 gate/up rows; w [rows] fp32 or None."""
    g = gate_bf16.float()
    u = up_bf16.float()
    sv = silu_tab[table_index(g)]
    if lim > 0:
        sv = torch.where(g > lim, silu_tab[SILU_LIM], sv)
        u = torch.where(u < -lim, torch.full_like(u, -lim), u)
        u = torch.where(u > lim, torch.full_like(u, lim), u)
    h = sv * u
    if w is not None:
        h = w.view(-1, 1) * h
    return h.to(torch.bfloat16).float()


def combine_torch(rows_bf16: torch.Tensor, shared_bf16: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(rows_bf16.size(-1), dtype=torch.float32, device=rows_bf16.device)
    for j in range(rows_bf16.size(0)):
        acc = acc + rows_bf16[j].float()
    acc = acc + shared_bf16.reshape(-1).float()
    return acc.to(torch.bfloat16)


def hc_post_torch(x, res, post, comb, mode: int) -> torch.Tensor:
    """x [D] bf16, res [HC, D] bf16, post [HC] fp32, comb [HC, HC] fp32 -> [HC, D] bf16 (the kernel's order)."""
    HC, D = res.shape
    xf, rf = x.float(), res.float()
    out = []
    for j in range(HC):
        p = post[j] * xf
        a = [torch.zeros(D, dtype=torch.float32, device=x.device) + comb[i, j] * rf[i] for i in range(HC)]
        if mode == 0:
            s = torch.zeros(D, dtype=torch.float32, device=x.device)
            for i in range(HC):
                s = s + comb[i, j] * rf[i]
        elif mode == 1:
            s = (a[0] + a[1]) + (a[2] + a[3])
        else:
            s = (a[0] + a[2]) + (a[1] + a[3])
        out.append(p + s)
    return torch.stack(out).to(torch.bfloat16)


def rope_pairs_torch(a, b, c, d, modes: Tuple[int, int]):
    """Rotated (re, im) in fp32, then the bf16 rounding of `y.copy_(x)`, as fp32 values."""
    return (rope_re(a, b, c, d, modes[0]).to(torch.bfloat16).float(),
            rope_im(a, b, c, d, modes[1]).to(torch.bfloat16).float())


def rope_rows_torch(x_bf16: torch.Tensor, freqs_real: torch.Tensor, head_dim: int, rope_dim: int,
                    modes: Tuple[int, int], inverse: bool) -> torch.Tensor:
    """x [..., n_heads * head_dim] bf16 -> same with the last rope_dim of every head rotated (fp32 math, bf16 out).
    freqs_real: [rope_dim/2, 2] fp32 (cos, sin) of this position."""
    shp = x_bf16.shape
    v = x_bf16.float().reshape(-1, head_dim)
    r = v[:, head_dim - rope_dim:].unflatten(-1, (-1, 2))
    c = freqs_real.reshape(-1, 2)[:, 0]
    d = freqs_real.reshape(-1, 2)[:, 1]
    if inverse:
        d = -d
    re, im = rope_pairs_torch(r[..., 0], r[..., 1], c, d, modes)
    v = torch.cat([v[:, :head_dim - rope_dim], torch.stack([re, im], -1).flatten(-2)], -1)
    return v.reshape(shp).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------ Triton kernels
if HAVE_TRITON:

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

    @triton.jit
    def t9b2_act_quant_bcast(x, x_rs, q, q_gs, s, s_gs, G, KB, tab, BB: tl.constexpr, S_F32: tl.constexpr,
                             s_cs=0, S_PACK: tl.constexpr = False, xi=0, oi=0, IDX: tl.constexpr = False,
                             PDL: tl.constexpr = 0):
        """Program (pb, r): 32-blocks [pb*BB, (pb+1)*BB) of input row r (bf16, row stride x_rs) -> fp8 values and the
        block scales (fp32 value or uint8 e8m0 code) written to output rows r*G + g for g < G (strides q_gs, s_gs):
        dg_act writes one activation row into row 0 of every DeepGEMM group this way. T9b-4 S_PACK: the codes go
        into DeepGEMM's packed MN-major SF bytes (_t9b4_store_code; s_gs / s_cs in bytes). M1-T10-PORT IDX: program
        r is the R-row MoE's sorted pair r: input row xi[r] (its token row) -> output row oi[r] (its A row), G = 1 --
        the act_quant of the R rows + the index_select / index_copy scatter of runtime/moe_rows.py in one launch."""
        if PDL > 0:
            tl.extra.cuda.gdc_wait()
        if PDL == 1:
            tl.extra.cuda.gdc_launch_dependents()
        pb = tl.program_id(0)
        r = tl.program_id(1)
        bi = pb * BB + tl.arange(0, BB)
        bok = bi < KB
        cols = bi[:, None] * 32 + tl.arange(0, 32)[None, :]
        m = bok[:, None] & (cols >= 0)
        if IDX:
            v = tl.load(x + tl.load(xi + r) * x_rs + cols, mask=m, other=0.0).to(tl.float32)
        else:
            v = tl.load(x + r * x_rs + cols, mask=m, other=0.0).to(tl.float32)
        code = _t9b2_code(tl.max(tl.abs(v), axis=1), tab)
        qv = _t9b2_q8(v, _t9b2_ipow2(code)[:, None])
        if IDX:
            ro = tl.load(oi + r)
            tl.store(q + ro * q_gs + cols, qv, mask=m)
            if S_PACK:
                _t9b4_store_code(s, ro, bi, code, bok, s_gs, s_cs, True)
            elif S_F32:
                tl.store(s + ro * s_gs + bi, _t9b2_pow2(code), mask=bok)
            else:
                tl.store(s + ro * s_gs + bi, code.to(tl.uint8), mask=bok)
        else:
            for g in range(G):
                o = r * G + g
                tl.store(q + o * q_gs + cols, qv, mask=m)
                if S_PACK:
                    _t9b4_store_code(s, o, bi, code, bok, s_gs, s_cs, True)
                elif S_F32:
                    tl.store(s + o * s_gs + bi, _t9b2_pow2(code), mask=bok)
                else:
                    tl.store(s + o * s_gs + bi, code.to(tl.uint8), mask=bok)
        if PDL == 2:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def t9b2_swiglu_quant(g, g_gs, u, u_gs, w, q, q_gs, s, s_gs, KB, silu_tab, tab, lim,
                          BB: tl.constexpr, HAS_W: tl.constexpr, HAS_LIM: tl.constexpr, S_F32: tl.constexpr,
                          s_cs=0, GS=0, US=0, S_PACK: tl.constexpr = False, NSG: tl.constexpr = 0,
                          NSU: tl.constexpr = 0, ri=0, IDX: tl.constexpr = False, PDL: tl.constexpr = 0):
        """Program (pb, j): group/row j, 32-blocks [pb*BB, (pb+1)*BB). gate/up bf16 -> silu(clamp(gate)) by table,
        clamp(up), silu*up, (w[j] *), bf16 RNE, then act_quant -> fp8 q and the scale (fp32 or e8m0 code). T9b-4:
        NSG / NSU > 0 = gate / up are the GEMV's fp32 split partials (split strides GS / US) reduced on load in the
        split-K reduce's order (_t9b4_ld); S_PACK = codes into DeepGEMM's packed SF bytes (_t9b4_store_code).
        M1-T10-PORT IDX: program j is the R-row MoE's sorted pair j: gate / up / q / s rows ri[j] (its A row), the
        weight w[j] (its routing weight in sorted-pair order) -- runtime/moe_rows.py's index_select + glue +
        act_quant + index_copy in one launch."""
        if PDL > 0:
            tl.extra.cuda.gdc_wait()
        if PDL == 1:
            tl.extra.cuda.gdc_launch_dependents()
        pb = tl.program_id(0)
        j = tl.program_id(1)
        bi = pb * BB + tl.arange(0, BB)
        bok = bi < KB
        cols = bi[:, None] * 32 + tl.arange(0, 32)[None, :]
        m = bok[:, None] & (cols >= 0)
        if IDX:
            rj = tl.load(ri + j)
            gv = _t9b4_ld(g + rj * g_gs, cols, m, NSG, GS)
            uv = _t9b4_ld(u + rj * u_gs, cols, m, NSU, US)
        else:
            gv = _t9b4_ld(g + j * g_gs, cols, m, NSG, GS)
            uv = _t9b4_ld(u + j * u_gs, cols, m, NSU, US)
        sv = tl.load(silu_tab + ((gv.to(tl.int32, bitcast=True) >> 16) & 0xFFFF))
        if HAS_LIM:
            sv = tl.where(gv > lim, tl.load(silu_tab + 65536), sv)
            uv = tl.where(uv < -lim, -lim, uv)
            uv = tl.where(uv > lim, lim, uv)
        h = sv * uv
        if HAS_W:
            h = tl.load(w + j) * h
        h = _t9b2_bf16(h)
        code = _t9b2_code(tl.max(tl.abs(h), axis=1), tab)
        if IDX:
            tl.store(q + rj * q_gs + cols, _t9b2_q8(h, _t9b2_ipow2(code)[:, None]), mask=m)
            if S_PACK:
                _t9b4_store_code(s, rj, bi, code, bok, s_gs, s_cs, True)
            elif S_F32:
                tl.store(s + rj * s_gs + bi, _t9b2_pow2(code), mask=bok)
            else:
                tl.store(s + rj * s_gs + bi, code.to(tl.uint8), mask=bok)
        else:
            tl.store(q + j * q_gs + cols, _t9b2_q8(h, _t9b2_ipow2(code)[:, None]), mask=m)
            if S_PACK:
                _t9b4_store_code(s, j, bi, code, bok, s_gs, s_cs, True)
            elif S_F32:
                tl.store(s + j * s_gs + bi, _t9b2_pow2(code), mask=bok)
            else:
                tl.store(s + j * s_gs + bi, code.to(tl.uint8), mask=bok)
        if PDL == 2:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def t9b2_moe_combine(d, d_gs, KS, sh, out, N, BLOCK_C: tl.constexpr, pidx=0, IDX: tl.constexpr = False,
                         PDL: tl.constexpr = 0):
        """out = bf16(((0 + d[0]) + d[1] + ... + d[KS-1]) + sh), fp32 adds in slot order. M1-T10-PORT IDX: program
        (pb, t) = token row t of the R-row MoE: its KS rows of d are pidx[t*KS + j] (runtime/moe_rows.py `posmat`:
        its pairs' A rows in ascending expert id), sh / out are row t -- combine_rows + `+= shared` + type_as."""
        if PDL > 0:
            tl.extra.cuda.gdc_wait()
        if PDL == 1:
            tl.extra.cuda.gdc_launch_dependents()
        i = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        ok = i < N
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        if IDX:
            t = tl.program_id(1)
            for j in range(KS):
                acc = acc + tl.load(d + tl.load(pidx + t * KS + j) * d_gs + i, mask=ok, other=0.0).to(tl.float32)
            acc = acc + tl.load(sh + t * N + i, mask=ok, other=0.0).to(tl.float32)
            tl.store(out + t * N + i, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=ok)
        else:
            for j in range(KS):
                acc = acc + tl.load(d + j * d_gs + i, mask=ok, other=0.0).to(tl.float32)
            acc = acc + tl.load(sh + i, mask=ok, other=0.0).to(tl.float32)
            tl.store(out + i, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=ok)
        if PDL == 2:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def t9b2_hc_post(x, res, post, comb, out, D, HC: tl.constexpr, BLOCK_C: tl.constexpr, MODE: tl.constexpr,
                     PDL: tl.constexpr = 0):
        """Program (pb, j, t): out[t, j, i] = bf16(post[t, j]*x[t, i] + sum_r comb[t, r, j]*res[t, r, i]) (sum
        order: MODE); token row t (M1-T10-FIX: grid axis 2 = rows, 1 at the one-row step) offsets every pointer."""
        if PDL > 0:
            tl.extra.cuda.gdc_wait()
        if PDL == 1:
            tl.extra.cuda.gdc_launch_dependents()
        pb = tl.program_id(0)
        j = tl.program_id(1)
        t = tl.program_id(2)
        x = x + t * D
        res = res + t * (HC * D)
        post = post + t * HC
        comb = comb + t * (HC * HC)
        out = out + t * (HC * D)
        i = pb * BLOCK_C + tl.arange(0, BLOCK_C)
        ok = i < D
        p = tl.load(post + j) * tl.load(x + i, mask=ok, other=0.0).to(tl.float32)
        z = tl.zeros([BLOCK_C], dtype=tl.float32)
        if MODE == 0:
            sacc = z
            for r in tl.static_range(HC):
                sacc = sacc + tl.load(comb + r * HC + j) * tl.load(res + r * D + i, mask=ok, other=0.0).to(tl.float32)
        else:
            a0 = z + tl.load(comb + 0 * HC + j) * tl.load(res + 0 * D + i, mask=ok, other=0.0).to(tl.float32)
            a1 = z + tl.load(comb + 1 * HC + j) * tl.load(res + 1 * D + i, mask=ok, other=0.0).to(tl.float32)
            a2 = z + tl.load(comb + 2 * HC + j) * tl.load(res + 2 * D + i, mask=ok, other=0.0).to(tl.float32)
            a3 = z + tl.load(comb + 3 * HC + j) * tl.load(res + 3 * D + i, mask=ok, other=0.0).to(tl.float32)
            if MODE == 1:
                sacc = (a0 + a1) + (a2 + a3)
            else:
                sacc = (a0 + a2) + (a1 + a3)
        tl.store(out + j * D + i, (p + sacc).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=ok)
        if PDL == 2:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def t9b2_o_rope_qdq(o, NB, freqs, tab, HD: tl.constexpr, RD: tl.constexpr, BB: tl.constexpr,
                        RE_MODE: tl.constexpr, IM_MODE: tl.constexpr, FS=0, PDL: tl.constexpr = 0):
        """In place on one bf16 row of NB 32-blocks (heads of HD, RoPE on the last RD of each): program pb takes
        blocks [pb*BB, (pb+1)*BB) as even/odd element pairs; rope columns get the INVERSE rotation by (c, -d)
        rounded to bf16; then the block's act_quant QDQ, bf16(fp32(q) * s), is stored back. M1-T10-PORT: grid axis
        1 = token row t (1 at the one-row step): row t of o (stride NB*32) with its own freqs row (stride FS)."""
        if PDL > 0:
            tl.extra.cuda.gdc_wait()
        if PDL == 1:
            tl.extra.cuda.gdc_launch_dependents()
        pb = tl.program_id(0)
        t = tl.program_id(1)
        o = o + t * (NB * 32)
        freqs = freqs + t * FS
        bi = pb * BB + tl.arange(0, BB)
        bok = bi < NB
        ce = bi[:, None] * 32 + 2 * tl.arange(0, 16)[None, :]          # even columns [BB, 16]
        m = bok[:, None] & (ce >= 0)
        a = tl.load(o + ce, mask=m, other=0.0).to(tl.float32)
        b = tl.load(o + ce + 1, mask=m, other=0.0).to(tl.float32)
        hcol = ce % HD
        is_r = hcol >= HD - RD
        fi = tl.where(is_r, (hcol - (HD - RD)) // 2, 0)
        c = tl.load(freqs + 2 * fi)
        dd = -tl.load(freqs + 2 * fi + 1)                                # freqs_cis.conj()
        ar = tl.where(is_r, _t9b2_bf16(_t9b2_rope_re(a, b, c, dd, RE_MODE)), a)
        br = tl.where(is_r, _t9b2_bf16(_t9b2_rope_im(a, b, c, dd, IM_MODE)), b)
        amax = tl.maximum(tl.max(tl.abs(ar), axis=1), tl.max(tl.abs(br), axis=1))
        code = _t9b2_code(amax, tab)
        inv = _t9b2_ipow2(code)[:, None]
        sc = _t9b2_pow2(code)[:, None]
        ya = (_t9b2_q8(ar, inv).to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne")
        yb = (_t9b2_q8(br, inv).to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(o + ce, ya, mask=m)
        tl.store(o + ce + 1, yb, mask=m)
        if PDL == 2:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def t9b2_rope_probe(a, b, c, d, out, n, BLOCK_P: tl.constexpr):
        """Self-check hook: out[k*n + i] = rope_re(MODE k) for k = 0..2, out[(3+k)*n + i] = rope_im(MODE k)."""
        i = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        ok = i < n
        av = tl.load(a + i, mask=ok, other=0.0)
        bv = tl.load(b + i, mask=ok, other=0.0)
        cv = tl.load(c + i, mask=ok, other=0.0)
        dv = tl.load(d + i, mask=ok, other=0.0)
        tl.store(out + 0 * n + i, _t9b2_rope_re(av, bv, cv, dv, 0), mask=ok)
        tl.store(out + 1 * n + i, _t9b2_rope_re(av, bv, cv, dv, 1), mask=ok)
        tl.store(out + 2 * n + i, _t9b2_rope_re(av, bv, cv, dv, 2), mask=ok)
        tl.store(out + 3 * n + i, _t9b2_rope_im(av, bv, cv, dv, 0), mask=ok)
        tl.store(out + 4 * n + i, _t9b2_rope_im(av, bv, cv, dv, 1), mask=ok)
        tl.store(out + 5 * n + i, _t9b2_rope_im(av, bv, cv, dv, 2), mask=ok)


KERNELS_ON_CPU = False  # tests/test_fused_glue_cpu.py sets it under tests/triton_shim.py: wrappers launch the kernels


def _kern(t: torch.Tensor) -> bool:
    """Launch the Triton kernel (CUDA, or a CPU tensor under the test shim) rather than the torch mirror."""
    return t.is_cuda or (KERNELS_ON_CPU and HAVE_TRITON)


AQ_BB = 8         # 32-blocks per program (act_quant / swiglu / o path)
COMBINE_BLOCK = 1024
HC_BLOCK = 1024
PROBE_BLOCK = 1024


# ------------------------------------------------------------------------------------------------ launch wrappers
# Each wrapper is one kernel launch on CUDA (the census counts it as one) and the torch mirror on CPU. Call sites
# reach them through this module's globals at call time (bench/glue_census.py wraps them).
def _is_packed(s) -> bool:
    return getattr(s, "packed", False)


def _split_src(t):
    """(tensor, rows, cols, row stride, NS, split stride) of a bf16 [R, I] row view or (T9b-4) a gemv_mxfp8.SplitK."""
    if hasattr(t, "nsplit") and hasattr(t, "ws"):
        return t.ws, t.M, t.N, t.N, t.nsplit, t.split_stride
    return t, t.size(0), t.size(1), t.stride(0), 0, 0


def _store_scales_torch(s, row: int, code: torch.Tensor) -> None:
    """CPU mirror of a scale-row store: fp32 2^(code-127), e8m0 bytes, or (T9b-4) packed DeepGEMM SF bytes."""
    if _is_packed(s):
        s.store_codes_torch(row, code)
    elif s.dtype == torch.float32:
        s[row].copy_(pow2_code(code))
    else:
        s.view(torch.uint8)[row].copy_(code.to(torch.uint8))


def launch_act_quant_bcast(x, q, s, code_tab, groups: int, src_rows=None, dst_rows=None):
    """x [R, K] bf16 (unit inner stride) -> q rows (r*groups + g) fp8 [.., K] and s rows (fp32 or e8m0/uint8, or
    T9b-4: a runtime.dg_inplace.PackedRows -- the codes straight into DeepGEMM's packed activation SF).
    M1-T10-PORT: src_rows / dst_rows int64 [P] (both or neither; groups 1): output row dst_rows[p] <- act_quant of
    input row src_rows[p], one program row per p (the R-row MoE's pair scatter)."""
    R_, K = x.shape
    KB = K // BLOCK
    pk = _is_packed(s)
    s_f32 = (not pk) and s.dtype == torch.float32
    idx = src_rows is not None
    if idx and (dst_rows is None or int(groups) != 1):
        raise ValueError("launch_act_quant_bcast: src_rows needs dst_rows and groups 1")
    if _kern(x):
        if pk:
            su, s_gs, s_cs = s.u8, s.row_stride, s.col_stride
        else:
            su = s if s_f32 else s.view(torch.uint8)
            s_gs, s_cs = su.stride(0), 0
        P = src_rows.numel() if idx else R_
        t9b2_act_quant_bcast[(triton.cdiv(KB, AQ_BB), P)](
            x, x.stride(0), q, q.stride(0), su, s_gs, int(groups), KB, code_tab, BB=AQ_BB, S_F32=s_f32,
            s_cs=int(s_cs), S_PACK=pk, **(dict(xi=src_rows, oi=dst_rows, IDX=True) if idx else {}), **LAUNCH,
            **_pdl.kw("t9b2_act_quant_bcast"))
        return
    qv, code = act_quant_torch(x.float(), code_tab)
    if idx:
        for p, (r, o) in enumerate(zip(src_rows.tolist(), dst_rows.tolist())):
            q[o].copy_(qv[r])
            _store_scales_torch(s, o, code[r])
        return
    for r in range(R_):
        for g in range(groups):
            o = r * groups + g
            q[o].copy_(qv[r])
            _store_scales_torch(s, o, code[r])


def launch_swiglu_quant(g, u, w, q, s, silu_tab, code_tab, lim: float, rows_idx=None):
    """g, u: [G, I] bf16 row views (unit inner stride) or (T9b-4 reduce fold) gemv_mxfp8.SplitK partials of the
    shared expert's w1 / w3 GEMVs; w: [G] fp32 or None; q [G, I] fp8; s [G, I/32] fp32/e8m0 or (T9b-4) a
    runtime.dg_inplace.PackedRows. M1-T10-PORT rows_idx int64 [P]: program p reads gate / up rows rows_idx[p] of the
    [A, I] row views g / u, w[p], and writes q / s rows rows_idx[p] (the R-row MoE's sorted pairs)."""
    gt, G_, I, g_gs, nsg, gs = _split_src(g)
    ut, _, _, u_gs, nsu, us = _split_src(u)
    KB = I // BLOCK
    pk = _is_packed(s)
    s_f32 = (not pk) and s.dtype == torch.float32
    idx = rows_idx is not None
    if idx and (nsg or nsu):
        raise ValueError("launch_swiglu_quant: rows_idx with split-K partials")
    if _kern(gt):
        if pk:
            su, s_gs, s_cs = s.u8, s.row_stride, s.col_stride
        else:
            su = s if s_f32 else s.view(torch.uint8)
            s_gs, s_cs = su.stride(0), 0
        P = rows_idx.numel() if idx else G_
        t9b2_swiglu_quant[(triton.cdiv(KB, AQ_BB), P)](
            gt, g_gs, ut, u_gs, w if w is not None else gt, q, q.stride(0), su, s_gs, KB,
            silu_tab, code_tab, float(lim), BB=AQ_BB, HAS_W=w is not None, HAS_LIM=lim > 0, S_F32=s_f32,
            s_cs=int(s_cs), GS=int(gs), US=int(us), S_PACK=pk, NSG=int(nsg), NSU=int(nsu),
            **(dict(ri=rows_idx, IDX=True) if idx else {}), **LAUNCH, **_pdl.kw("t9b2_swiglu_quant"))
        return
    if idx:
        ri = rows_idx.reshape(-1)
        h = swiglu_torch(g.index_select(0, ri), u.index_select(0, ri), w, silu_tab, lim)
        qv, code = act_quant_torch(h, code_tab)
        q.view(torch.uint8).index_copy_(0, ri, qv.contiguous().view(torch.uint8))
        for p, r in enumerate(ri.tolist()):
            _store_scales_torch(s, r, code[p])
        return
    gb = g.reduce_torch() if nsg else g
    ub = u.reduce_torch() if nsu else u
    h = swiglu_torch(gb, ub, w, silu_tab, lim)
    qv, code = act_quant_torch(h, code_tab)
    q.copy_(qv)
    if pk or not s_f32:
        for r in range(G_):
            _store_scales_torch(s, r, code[r])
    else:
        s.copy_(pow2_code(code))


def launch_moe_combine(rows, shared, out, posmat=None):
    """rows [KS, N] bf16 (row views, unit inner), shared [N] bf16, out [N] bf16. M1-T10-PORT posmat int64 [R, k]:
    rows = all A rows [A, N], shared / out [R, N]: out[t] = bf16((0 + sum_j rows[posmat[t, j]]) + shared[t])."""
    if posmat is not None:
        R_, KS = posmat.shape
        N = rows.size(1)
        if _kern(rows):
            t9b2_moe_combine[(triton.cdiv(N, COMBINE_BLOCK), R_)](rows, rows.stride(0), KS, shared, out, N,
                                                                  BLOCK_C=COMBINE_BLOCK, pidx=posmat, IDX=True,
                                                                  **LAUNCH, **_pdl.kw("t9b2_moe_combine"))
            return out
        y = torch.zeros(R_, N, dtype=torch.float32, device=rows.device)
        for j in range(KS):
            y = y + rows.index_select(0, posmat[:, j]).float()
        return out.copy_((y + shared.reshape(R_, N).float()).to(torch.bfloat16))
    KS, N = rows.shape
    if _kern(rows):
        t9b2_moe_combine[(triton.cdiv(N, COMBINE_BLOCK),)](rows, rows.stride(0), KS, shared, out, N,
                                                            BLOCK_C=COMBINE_BLOCK, **LAUNCH,
                                                            **_pdl.kw("t9b2_moe_combine"))
        return out
    return out.copy_(combine_torch(rows, shared))


def launch_hc_post(x, res, post, comb, out, mode: int):
    """x [D] or [T, D], res / out [HC, D] or [T, HC, D] bf16 contiguous; post [HC] / [T, HC], comb [HC, HC] /
    [T, HC, HC] fp32 contiguous (T token rows, one launch)."""
    HC, D = res.shape[-2], res.shape[-1]
    T_ = res.numel() // (HC * D)
    if _kern(x):
        t9b2_hc_post[(triton.cdiv(D, HC_BLOCK), HC, T_)](x, res, post, comb, out, D, HC=HC, BLOCK_C=HC_BLOCK,
                                                         MODE=int(mode), **LAUNCH, **_pdl.kw("t9b2_hc_post"))
        return out
    xs, rs, ps, cs = x.reshape(T_, D), res.reshape(T_, HC, D), post.reshape(T_, HC), comb.reshape(T_, HC, HC)
    return out.copy_(torch.stack([hc_post_torch(xs[t], rs[t], ps[t], cs[t], mode) for t in range(T_)])
                     .view(out.shape))


def launch_o_rope_qdq(o_row, freqs_real, code_tab, head_dim: int, rope_dim: int, modes: Tuple[int, int]):
    """In place on o_row [N] bf16 contiguous (N = n_heads * head_dim): inverse RoPE + act_quant QDQ. M1-T10-PORT:
    o_row [R, N] with freqs_real [R, rope_dim/2, 2] (row t rotated by its own position), one launch."""
    rows = o_row.dim() == 2
    R_ = o_row.size(0) if rows else 1
    N = o_row.numel() // R_
    if _kern(o_row):
        NB = N // BLOCK
        grid = (triton.cdiv(NB, AQ_BB), R_) if rows else (triton.cdiv(NB, AQ_BB),)
        fs = {"FS": int(freqs_real.stride(0))} if rows and freqs_real.dim() == 3 else {}
        t9b2_o_rope_qdq[grid](o_row, NB, freqs_real, code_tab, HD=head_dim, RD=rope_dim, BB=AQ_BB,
                              RE_MODE=int(modes[0]), IM_MODE=int(modes[1]), **fs, **LAUNCH,
                              **_pdl.kw("t9b2_o_rope_qdq"))
        return o_row
    if rows:
        for t in range(R_):
            _o_rope_qdq_torch(o_row[t], freqs_real[t] if freqs_real.dim() == 3 else freqs_real, code_tab, head_dim,
                              rope_dim, modes)
        return o_row
    return _o_rope_qdq_torch(o_row, freqs_real, code_tab, head_dim, rope_dim, modes)


def _o_rope_qdq_torch(o_row, freqs_real, code_tab, head_dim: int, rope_dim: int, modes) -> torch.Tensor:
    """launch_o_rope_qdq's CPU mirror for one row."""
    r = rope_rows_torch(o_row, freqs_real, head_dim, rope_dim, modes, inverse=True) if rope_dim else o_row
    return o_row.copy_(qdq_torch(r.float(), code_tab))


def rope_probe(a, b, c, d) -> torch.Tensor:
    """[6, n] fp32: rope_re modes 0..2, rope_im modes 0..2 of fp32 [n] inputs (the kernel's helpers on CUDA)."""
    n = a.numel()
    if _kern(a):
        out = torch.empty(6, n, dtype=torch.float32, device=a.device)
        t9b2_rope_probe[(triton.cdiv(n, PROBE_BLOCK),)](a, b, c, d, out, n, BLOCK_P=PROBE_BLOCK, **LAUNCH)
        return out
    return torch.stack([rope_re(a, b, c, d, m) for m in ROPE_MODES] + [rope_im(a, b, c, d, m) for m in ROPE_MODES])


# ------------------------------------------------------------------------------------------------ the glue object
from runtime.fused_glue_r2 import R2Glue  # noqa: E402  (round 2: kernels, calibration and call-site operations)


class FusedGlue(R2Glue):
    """Per-StaticStep fused glue: tables, modes, per-cluster status, and the call-site operations. Built by
    `StaticStep(glue_impl="fused")` after its MoE buffers; `calibrate=False` skips the self-check (the census, which
    counts ops, not bits) and assumes every applicable cluster on with modes (0, 0) / 0 (round 2: torch's own
    reduction on CPU, the predicted order on CUDA). `approx=True` (--glue-approx) also enables round-2 clusters whose
    statistic has no bit-identical order on this device (outcome (ii), status "approx")."""

    def __init__(self, owner, *, calibrate: bool = True, disable: Optional[List[str]] = None, log=None,
                 approx: bool = False):
        self.o = owner
        self.R = owner.R
        self.dev = owner.dev
        self.log = log or (lambda *a, **k: None)
        self.rope_fwd: Tuple[int, int] = (0, 0)
        self.rope_inv: Tuple[int, int] = (0, 0)
        self.hc_mode = 0
        self.calib: Dict[str, object] = {}
        self.status: Dict[str, str] = {}
        model = owner.model
        layers = list(model.layers)
        a0 = layers[0].attn
        self.head_dim, self.rope_dim, self.n_heads = a0.head_dim, a0.rope_head_dim, a0.n_local_heads
        self.hc = model.hc_mult
        sh = layers[0].ffn.shared_experts
        self.lim = float(sh.swiglu_limit)
        env = [c.strip() for c in os.environ.get(DISABLE_ENV, "").split(",") if c.strip()]
        manual = set(env) | set(disable or [])
        bad = manual - set(CLUSTERS)
        if bad:
            raise ValueError(f"{DISABLE_ENV}/disable names unknown clusters {sorted(bad)}; known {CLUSTERS}")
        R = self.R
        mx = (R.fp8_block_size, R.scale_dtype) == (BLOCK, E8) and R.scale_fmt is not None
        self.rows = int(getattr(owner, "rows", 1))
        for c in CLUSTERS:
            why = self._applicable(c, layers, mx)
            if why is None and self.rows > 1 and c not in ROWS_CLUSTERS:  # M1-T10-PORT: not ported to rows
                why = f"rows={self.rows}: {ROWS_WHY.get(c, 'one-row kernel')}"
            self.status[c] = "on" if why is None else f"n/a: {why}"
        for c in manual:
            if self.status[c] == "on":
                self.status[c] = f"off: {DISABLE_ENV}"
        self.code_tab = scale_code_table(R, self.dev) if mx else None
        self.silu_tab = silu_table(self.dev, self.lim)
        self._ptab6: Dict[int, torch.Tensor] = {}
        self._dtab: Optional[torch.Tensor] = None
        self._dsts: List[torch.Tensor] = []
        if self.on("gather"):
            self._build_gather(layers)
        if not self.on("hc_norm"):
            self.status["mix_stat"] = self.status["mix_stat"] if self.status["mix_stat"] != "on" else \
                "off: needs hc_norm"
        self._r2_init(layers, approx)
        if calibrate:
            self.calibrate()
        else:
            self.calib["skipped"] = True
            self._r2_default_orders()
            self._off("dg_sf_pack", "not self-checked (calibrate=False)")

    # ---------------------------------------------------------------- status
    def on(self, c: str) -> bool:
        return self.status.get(c) in ("on", "approx")

    def _off(self, c: str, why: str) -> None:
        if self.on(c):
            self.status[c] = f"off: {why}"
            self.log(f"[fused-glue] {c} OFF: {why}")

    def _applicable(self, c, layers, mx) -> Optional[str]:
        o, R = self.o, self.R
        dg = o.moe_impl in ("dg-masked", "dg-inplace")  # T9a-2: dg-inplace reuses dg-masked's act/SwiGLU/combine
        if c in R2_CLUSTER_NAMES:
            return self._applicable_r2(c, layers, mx)
        if c in R4_PARTS:
            return self._applicable_r4(c, layers, mx)
        if c in ACT_CLUSTERS and not mx:
            return "needs the MXFP8 act_quant globals (block 32, ue8m0, e8m0 scales)"
        if c == "concat_free":
            return CONCAT_FREE_WHY
        if c == "gather":
            if o.moe_impl == "dg-inplace":
                return "dg-inplace stages cold rows + packed scales in its own launch (runtime/dg_inplace.py)"
            rb = o.row_bytes
            if not (rb[("w1", "weight")] == rb[("w3", "weight")] == rb[("w2", "weight")]
                    and rb[("w1", "scale")] == rb[("w3", "scale")] == rb[("w2", "scale")]):
                return "the three projections' staging rows differ in size"
            return None
        if c in ("dg_act", "swiglu_routed", "moe_combine"):
            return None if dg else "moe_impl ref (StaticMoE runs reference Experts)"
        if c == "swiglu_shared":
            ex = layers[0].ffn.shared_experts
            ok = all(getattr(getattr(l.ffn.shared_experts, p).weight, "dtype", None) == FP8 for l in layers
                     for p in ("w1", "w2", "w3"))
            if not ok or ex.w1.out_features % BLOCK:
                return "shared experts are not fp8 Linears with a 32-multiple inter dim"
            return None
        if c == "hc_post":
            return None
        if c == "dedup":
            if self.rows == 1:
                return "rows=1: the one-row MoE routes k distinct experts (no dedup)"
            if self.rows * o.topk > DEDUP_MAX_PAIRS:
                return f"rows x topk = {self.rows * o.topk} pairs > {DEDUP_MAX_PAIRS} (one-program dedup kernel)"
            return None
        attns = [l.attn for l in layers]
        if c == "o_rope_qdq":
            same = all((a.head_dim, a.rope_head_dim) == (self.head_dim, self.rope_dim) for a in attns)
            if not same or self.head_dim % BLOCK or self.rope_dim % 2 or (self.head_dim * self.n_heads) % BLOCK:
                return "attention head layout differs across layers or is not 32-aligned"
            return None
        if c == "q_rope_epilogue":
            if o.fp8_impl != "gemv":
                return "fp8_impl ref (the epilogue lives in the T9b-1 GEMV)"
            same = all((a.head_dim, a.rope_head_dim) == (self.head_dim, self.rope_dim) for a in attns)
            if not same or any(a.wq_b.weight.dtype != FP8 for a in attns) or self.rope_dim % 2:
                return "wq_b not fp8 or head layout differs across layers"
            return None
        return "unknown"

    # ---------------------------------------------------------------- gather tables (#11 + #9)
    def _build_gather(self, layers) -> None:
        o = self.o
        dg = o.moe_impl == "dg-masked"
        order = [("w1", "weight"), ("w3", "weight"), ("w2", "weight"), ("w1", "scale"), ("w3", "scale"),
                 ("w2", "scale")]
        for sm in o.moes:
            self._ptab6[sm.layer] = torch.stack([sm.ptabs[k] for k in order]).contiguous()
        dsts = [o.staging[k] for k in order[:3]]
        dsts += [o.dg.sf32[p] for p in ("w1", "w3", "w2")] if dg else [o.staging[k] for k in order[3:]]
        self._dsts = dsts
        self._dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device=self.dev)
        self.scale_f32 = dg

    def gather(self, sm, ids: torch.Tensor) -> None:
        """All six staging tables of `sm`'s layer in one launch (+ dg: fp32 scales into sf32; + ids_step)."""
        from runtime.static_kernels import gather_tables

        o = self.o
        gather_tables(self._ptab6[sm.layer], ids, self._dtab, self._dsts, o.row_bytes[("w1", "weight")],
                      o.row_bytes[("w1", "scale")], self.scale_f32, o.ids_step[sm.layer] if o.counters else None)

    # ---------------------------------------------------------------- MoE (dg-masked)
    def dg_act(self, x: torch.Tensor) -> None:
        """act_quant(x) into row 0 of every group of DeepGEMM's w1/w3 input buffer (fp8 + fp32 scales)."""
        b = self.o.dg
        launch_act_quant_bcast(x.view(1, -1), b.xq[1][:, 0], self.dg_scale_dst(1), self.code_tab, b.k)

    def routed_swiglu(self, w: torch.Tensor) -> None:
        """Routed SwiGLU on the w1/w3 GEMM outputs (row 0 of each group) -> act_quant -> row 0 of the w2 input."""
        b = self.o.dg
        launch_swiglu_quant(b.d["w1"][:, 0], b.d["w3"][:, 0], w, b.xq[2][:, 0], self.dg_scale_dst(2), self.silu_tab,
                            self.code_tab, b.swiglu_limit)

    def dg_scale_dst(self, which: int):
        """Where the DeepGEMM input `which`'s row-0 scales go: the fp32 xs rows, or (T9b-4 dg_sf_pack) the packed
        activation SF that the contiguous GEMM then takes as is."""
        b = self.o.dg
        return b.xsp_rows[which] if getattr(b, "sf_pack", False) else b.xs[which][:, 0]

    def combine(self, out_rows: torch.Tensor, shared: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """y = bf16((0 + sum_j out[j]) + shared) as MoE.forward's fp32 accumulation (model.py:895-906)."""
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        launch_moe_combine(out_rows, shared.reshape(-1), y.view(-1))
        return y

    # ---------------------------------------------------------------- MoE at R rows (M1-T10-PORT, runtime/moe_rows.py)
    def dg_act_rows(self, x: torch.Tensor, m: dict) -> None:
        """act_quant(x) of the R rows straight into the pair A rows of DeepGEMM's w1/w3 input: A row arow[t] <- row
        prow[t] (fp8 + fp32 scales), one launch (the reference: act_quant, 2 index_select, e8m0_to_f32, 2 index_copy)."""
        b = self.o.dg
        D = x.size(-1)
        launch_act_quant_bcast(x.reshape(-1, D), b.xq[1].view(-1, D), self.dg_scale_rows(1, D // BLOCK), self.code_tab,
                               1, src_rows=m["prow"], dst_rows=m["arow"])

    def dg_scale_rows(self, which: int, kb: int):
        """M1-T10-MOE: where the R-row writers' scales of DeepGEMM input `which` go -- the fp32 xs as [A rows, K/32],
        or (dg_sf_pack ported to R rows) the packed activation SF addressed by A row."""
        b = self.o.dg
        if getattr(b, "sf_pack", False) and getattr(b, "xsp_all", None):
            return b.xsp_all[which]
        return b.xs[which].view(-1, kb)

    def routed_swiglu_rows(self, gate_rows: torch.Tensor, up_rows: torch.Tensor, m: dict) -> None:
        """The pairs' SwiGLU (model.py:845-853) on the w1/w3 GEMM outputs [A, I] at rows arow, weights pw, then
        act_quant into the same A rows of the w2 input: one launch (reference: 2 index_select, clamp, silu, *, *w,
        bf16, act_quant, e8m0_to_f32, 2 index_copy)."""
        b = self.o.dg
        I = gate_rows.size(-1)
        launch_swiglu_quant(gate_rows, up_rows, m["pw"], b.xq[2].view(-1, I), self.dg_scale_rows(2, I // BLOCK),
                            self.silu_tab, self.code_tab, b.swiglu_limit, rows_idx=m["arow"])

    def combine_rows(self, out_rows: torch.Tensor, posmat: torch.Tensor, shared: torch.Tensor,
                     x: torch.Tensor) -> torch.Tensor:
        """moe_rows.combine_rows + `+= shared` + type_as(x) (model.py:895-906 at R rows) in one launch: row t =
        bf16((0 + sum_j out[posmat[t, j]]) + shared[t]), fp32 adds in ascending expert id."""
        D = x.size(-1)
        R_ = x.numel() // D
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        launch_moe_combine(out_rows, shared.reshape(R_, D).contiguous(), y.view(R_, D), posmat=posmat)
        return y

    def shared_expert(self, ex, x: torch.Tensor) -> torch.Tensor:
        """Expert.forward (model.py:843-853, weights None) with the SwiGLU + bf16 + w2's act_quant in one kernel; w1,
        w3 and w2 keep their GEMM path (patches.fp8_linear_prequant routes w2 exactly as linear() would)."""
        from runtime.patches import fp8_linear_prequant

        ctx = self.ffn_pre(x) if self.on("hc_norm") else None
        xq = None
        if ctx is not None and ctx.get("q") is not None:      # round 2 #1: act_quant(x) already made by hc_norm
            xq = (ctx["q"].view(*x.shape), ctx["s"].view(*x.shape[:-1], -1))
        return self._shared_from_q(ex, x, xq)

    def _shared_from_q(self, ex, x: torch.Tensor, xq) -> torch.Tensor:
        from runtime.patches import fp8_linear_prequant

        I = ex.w1.weight.size(0)
        lead = tuple(x.shape[:-1])
        g2 = u2 = None
        if xq is not None and (self.on("gemv_pair") or self.on("reduce_fold")) and \
                self._gemv_ok(xq, ex.w1, ex.w3):              # T9b-4 L2: one launch for w1 || w3, reduce folded
            if self.on("gemv_pair"):
                g2, u2 = self._gemv_pair(xq, ex.w1, ex.w3, ("w1", "w3"))
            else:
                g2, u2 = self._gemv_fold(xq, ex.w1, "w1"), self._gemv_fold(xq, ex.w3, "w3")
        elif xq is not None:
            g2 = self.linear_pre(xq, ex.w1, x).view(-1, I)
            u2 = self.linear_pre(xq, ex.w3, x).view(-1, I)
        else:
            g2 = ex.w1(x).view(-1, I)
            u2 = ex.w3(x).view(-1, I)
        rows = g2.M if hasattr(g2, "nsplit") else g2.size(0)
        q = torch.empty(rows, I, dtype=FP8, device=x.device)
        s = torch.empty(rows, I // BLOCK, dtype=E8, device=x.device)
        launch_swiglu_quant(g2, u2, None, q, s, self.silu_tab, self.code_tab, ex.swiglu_limit)
        return fp8_linear_prequant(q.view(*lead, I), s.view(*lead, I // BLOCK), ex.w2.weight)

    # ---------------------------------------------------------------- mHC post (#7)
    def hc_post(self, x, residual, post, comb) -> torch.Tensor:
        """Block.hc_post (model.py:964-968) for T token rows (1, or the R-row step's R): x [1,T,D], residual
        [1,T,HC,D], post [1,T,HC], comb [1,T,HC,HC] -> [1,T,HC,D] (dtype of x), one launch."""
        HC, D = residual.size(-2), residual.size(-1)
        T_ = residual.numel() // (HC * D)
        out = torch.empty(residual.shape, dtype=x.dtype, device=x.device)
        launch_hc_post(x.reshape(T_, D).contiguous(), residual.reshape(T_, HC, D).contiguous(),
                       post.reshape(T_, HC).contiguous(), comb.reshape(T_, HC, HC).contiguous(),
                       out.view(T_, HC, D), self.hc_mode)
        return out

    # ---------------------------------------------------------------- attention (#10, #13)
    def o_path(self, attn, o: torch.Tensor, freqs_cis: torch.Tensor) -> None:
        """In place on the sparse_attn output o [1,T,H,head_dim]: inverse RoPE + act_quant QDQ (model.py:783-788);
        T token rows (1, or M1-T10-PORT the R-row step's R with freqs_cis [R, rd/2]: one launch, row t at its own
        position)."""
        T_ = o.numel() // (attn.n_local_heads * attn.head_dim)
        fr = torch.view_as_real(freqs_cis)
        if T_ > 1:
            launch_o_rope_qdq(o.view(T_, -1), fr.reshape(T_, -1, 2).contiguous(), self.code_tab, attn.head_dim,
                              attn.rope_head_dim, self.rope_inv)
            return
        launch_o_rope_qdq(o.view(-1), fr.reshape(-1, 2), self.code_tab, attn.head_dim,
                          attn.rope_head_dim, self.rope_inv)

    def q_proj(self, attn, qr: torch.Tensor, freqs_cis: torch.Tensor, qq=None) -> Optional[torch.Tensor]:
        """wq_b(qr) on the GEMV with q's RoPE in its store (model.py:770-772), or None when this call would not take
        the GEMV (then the caller runs the reference ops). qq: act_quant(qr), already made by the fused q path (R2)."""
        from runtime import patches
        from runtime.gemv_mxfp8 import linear_fp8_rope

        R, w = self.R, attn.wq_b.weight
        if not (patches.fp8_gemv_active() and patches.fp8_linear_gemv_enabled()):
            return None
        if patches.fp8_gemv_reason(R, qr, w) is not None:
            return None
        patches.count_gemv_routed(w, qr)
        if qq is not None:
            xq, s = qq
        else:
            xq, s = R.act_quant(qr, R.fp8_block_size, R.scale_fmt, R.scale_dtype)   # model.py:197 (inside wq_b)
        T_ = xq.numel() // xq.size(-1)
        fr = torch.view_as_real(freqs_cis)       # M1-T10-PORT: [T, rd/2, 2] at T rows (row m at its own position)
        fr = fr.reshape(T_, -1, 2).contiguous() if T_ > 1 else fr.reshape(-1, 2)
        out = linear_fp8_rope(xq, s, w, w.scale, fr, attn.head_dim, attn.rope_head_dim, self.rope_fwd)
        return out.unflatten(-1, (attn.n_local_heads, attn.head_dim))

    # ---------------------------------------------------------------- self-check
    def calibrate(self) -> dict:
        """The build-time self-check (module docstring). Turns clusters off on any mismatch; returns self.calib."""
        g = torch.Generator(device="cpu").manual_seed(20261106)
        cal = self.calib
        try:
            self._check_tables(cal)
            if any(self.on(c) for c in ACT_CLUSTERS):
                self._check_act_quant(cal, g)
            if self.on("o_rope_qdq") or self.on("q_rope_epilogue") or self.on("kv_path"):
                self._pick_rope(cal, g)
            if self.on("hc_post"):
                self._pick_hc(cal, g)
            for c, fn in (("dg_act", self._verify_dg_act), ("swiglu_routed", self._verify_swiglu_routed),
                          ("swiglu_shared", self._verify_swiglu_shared), ("moe_combine", self._verify_combine),
                          ("o_rope_qdq", self._verify_o_path), ("q_rope_epilogue", self._verify_q_proj),
                          ("gather", self._verify_gather), ("dedup", self._verify_dedup)):
                if self.on(c):
                    why = fn(g)
                    cal[f"verify_{c}"] = why or "equal"
                    if why:
                        self._off(c, f"self-check: {why}")
            self.calibrate_r2(cal, g)        # round 2 (runtime/fused_glue_r2.py): orders, then end-to-end checks
            self.calibrate_r4(cal, g)        # T9b-4 parts (runtime/fused_glue_r2.py): each == the path it replaces
        finally:
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
        return cal

    # tables -----------------------------------------------------------------------------------------
    def _check_tables(self, cal) -> None:
        if self.code_tab is None:
            return
        pos = self.code_tab[: 0x7F81].to(torch.int32)          # +0 .. +inf amax patterns
        fin = pos[: 0x7F80]
        lo, hi = int(fin.min()), int(fin.max())
        cal["code_table_range"] = [lo, hi]
        if lo < 1 or hi > 253:
            for c in ACT_CLUSTERS:
                self._off(c, f"reference act_quant codes {lo}..{hi} outside 1..253 for finite amax")

    # act_quant ----------------------------------------------------------------------------------------
    @_cpu_default
    def _act_inputs(self, g, K: int) -> List[torch.Tensor]:
        allb = bf16_bits_all()
        fin = allb[torch.isfinite(allb.float())]
        n = fin.numel() // BLOCK * BLOCK
        ex = fin[:n].view(-1, BLOCK)                                         # every finite bf16 value
        srt = fin[:n][torch.argsort(fin[:n].float().abs())].view(-1, BLOCK)  # similar magnitudes per block
        rnd = _rand_act(g, 64, K)
        return [t.to(self.dev) for t in (ex, srt, rnd)]

    def _check_act_quant(self, cal, g) -> None:
        R = self.R
        bad = []
        for x in self._act_inputs(g, 1024):
            q_ref, s_ref = R.act_quant(x.contiguous(), BLOCK, R.scale_fmt, R.scale_dtype)
            q = torch.empty_like(q_ref)
            s = torch.empty_like(s_ref)
            launch_act_quant_bcast(x, q, s, self.code_tab, 1)
            ok = torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and \
                torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
            y_ref = R.act_quant(x.clone(), BLOCK, R.scale_fmt, R.scale_dtype, True)
            y = x.clone()   # QDQ through the o-path kernel with no RoPE columns (RD 0)
            launch_o_rope_qdq(y.view(-1), torch.zeros(1, 2, dtype=torch.float32, device=self.dev), self.code_tab,
                              BLOCK, 0, (0, 0))
            if not ok or not torch.equal(y.view(torch.int16), y_ref.view(torch.int16)):
                bad.append(tuple(x.shape))
        cal["act_quant"] = "equal" if not bad else f"MISMATCH on {bad}"
        if bad:
            for c in ACT_CLUSTERS:
                self._off(c, "act_quant self-check mismatch vs the reference kernel")

    # RoPE modes ----------------------------------------------------------------------------------------
    @_cpu_default
    def _rope_data(self, g, attn, inverse: bool):
        """[(x_bf16 [1,1,H,hd], freqs complex [1, rd/2])] at several positions; half the heads hold pairs crafted so
        the rotated component nearly cancels (where the contraction form shows)."""
        H, hd, rd = self.n_heads, self.head_dim, self.rope_dim
        S = attn.freqs_cis.size(0)
        out = []
        for t in range(12):
            pos = int(torch.randint(1, max(2, S), (1,), generator=g, device="cpu"))
            f = attn.freqs_cis[pos:pos + 1]                                      # [1, rd/2] complex
            fr = torch.view_as_real(f).reshape(-1, 2).float().cpu()
            c, d = fr[:, 0], fr[:, 1]
            if inverse:
                d = -d
            x = _rand_act(g, 1, H * hd).view(H, hd)
            a = (torch.rand(H, rd // 2, generator=g, device="cpu") + 1.0) * torch.where(torch.rand(H, rd // 2, generator=g, device="cpu") < 0.5,
                                                                           -1.0, 1.0)
            a = a.to(torch.bfloat16).float()
            # re ~ 0: b = a*c/d ; im ~ 0: b = -a*d/c (bf16), on alternating heads
            dd = torch.where(d.abs() < 1e-30, torch.full_like(d, 1e-30), d)
            cc = torch.where(c.abs() < 1e-30, torch.full_like(c, 1e-30), c)
            b_re = (a * c / dd).clamp(-1e4, 1e4).to(torch.bfloat16).float()
            b_im = (-a * d / cc).clamp(-1e4, 1e4).to(torch.bfloat16).float()
            pairs = torch.where((torch.arange(H) % 4 == 1).view(H, 1), b_re, b_im)
            craft = torch.stack([a, pairs], -1).flatten(-2).to(torch.bfloat16)
            rows = torch.arange(H) % 2 == 1
            x[rows, hd - rd:] = craft[rows]
            out.append((x.view(1, 1, H, hd).to(self.dev), f))
        return out

    def _pick_rope(self, cal, g) -> None:
        R = self.R
        attn = self.o.model.layers[0].attn
        rd, hd = self.rope_dim, self.head_dim
        for inverse, key, clusters in ((False, "rope_fwd", ("q_rope_epilogue", "kv_path")),
                                       (True, "rope_inv", ("o_rope_qdq",))):
            if not any(self.on(c) for c in clusters):
                continue
            mism = {("re", m): 0 for m in ROPE_MODES}
            mism.update({("im", m): 0 for m in ROPE_MODES})
            n_tot = 0
            for x, f in self._rope_data(g, attn, inverse):
                ref = x.clone()
                R.apply_rotary_emb(ref[..., -rd:], f, inverse)
                v = x[..., -rd:].float().reshape(-1, 2)
                fr = torch.view_as_real(f).reshape(1, rd // 2, 2).float().expand(self.n_heads, rd // 2, 2).reshape(-1, 2)
                c, d = fr[:, 0].contiguous(), fr[:, 1].contiguous()
                if inverse:
                    d = -d
                cand = rope_probe(v[:, 0].contiguous(), v[:, 1].contiguous(), c, d).to(torch.bfloat16)
                rr = ref[..., -rd:].reshape(-1, 2)
                for m in ROPE_MODES:
                    mism[("re", m)] += int((cand[m].view(torch.int16) != rr[:, 0].view(torch.int16)).sum())
                    mism[("im", m)] += int((cand[3 + m].view(torch.int16) != rr[:, 1].view(torch.int16)).sum())
                n_tot += rr.size(0)
            pick = []
            for part in ("re", "im"):
                ok = [m for m in ROPE_MODES if mism[(part, m)] == 0]
                pick.append(ok[0] if ok else None)
            cal[key] = {"mismatches": {f"{p}{m}": v for (p, m), v in mism.items()}, "pairs": n_tot,
                        "picked": pick, "ambiguous": [p for p in ("re", "im")
                                                      if sum(mism[(p, m)] == 0 for m in ROPE_MODES) > 1]}
            if None in pick:
                for c in clusters:
                    self._off(c, f"no RoPE contraction form matches apply_rotary_emb ({key} {cal[key]['mismatches']})")
            else:
                setattr(self, key, (pick[0], pick[1]))

    # hc_post mode -----------------------------------------------------------------------------------
    @_cpu_default
    def _hc_data(self, g, D, rows: Optional[int] = None):
        """Random hc_post inputs; in half the columns copy 0 and copy 2 carry the same large value and comb's row 2
        is minus row 0, so a0 + a2 cancels exactly and the 4-term sum's ORDER shows in the result (the three
        HC_SUM_MODES round differently there). rows: token rows (default the step's; T9b-7: the pick draws 1)."""
        HC, T_ = self.hc, (self.rows if rows is None else int(rows))  # M1-T10-FIX; 1 = the draws of T9b-2
        x = _rand_act(g, T_, D).view(1, T_, D)
        res = _rand_act(g, T_ * HC, D).view(1, T_, HC, D)
        post = (torch.rand(1, T_, HC, generator=g, device="cpu") * 2).float()
        comb = (torch.randn(1, T_, HC, HC, generator=g, device="cpu") * torch.exp2(torch.randint(-4, 5, (1, T_, HC, HC), generator=g, device="cpu")
                                                                       .float())).float()
        if HC == 4:
            big = (torch.randn(D, generator=g, device="cpu") * torch.exp2(torch.randint(10, 17, (D,), generator=g, device="cpu").float()))
            big = big.to(torch.bfloat16)
            cols = torch.arange(D, device="cpu") % 2 == 0
            res[0, :, 0, cols] = big[cols]
            res[0, :, 2, cols] = big[cols]
            comb[0, :, 2, :] = -comb[0, :, 0, :]
        return [t.to(self.dev) for t in (x, res, post, comb)]

    def _pick_hc(self, cal, g) -> None:
        """hc_post's 4-term sum order (HC_SUM_MODES) against Block.hc_post. T9b-7 shape policy: picked on 1-row draws
        (the 1-row reference's call, whatever R this step runs), then at R > 1 the R-row launch is checked against the
        reference evaluated one row at a time (fused_glue_r2.per_row_reductions); rows policy: picked at R rows."""
        layer = self.o.model.layers[0]
        D = layer.attn_norm.dim
        modes = HC_SUM_MODES if self.hc == 4 else (0,)
        mism = {m: 0 for m in modes}
        shape = getattr(self, "order_policy", "rows") == "shape"
        pick_rows = 1 if shape else self.rows
        for _ in range(6):
            x, res, post, comb = self._hc_data(g, D, pick_rows)
            ref = layer.hc_post(x, res, post, comb)
            for m in modes:
                self.hc_mode = m
                got = self.hc_post(x, res, post, comb)
                mism[m] += int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        ok = [m for m in modes if mism[m] == 0]
        cal["hc_post"] = {"mismatches": mism, "picked": ok[0] if ok else None, "ambiguous": len(ok) > 1}
        if ok:
            self.hc_mode = ok[0]
        else:
            self.hc_mode = 0
            self._off("hc_post", f"no 4-term sum order matches Block.hc_post ({mism})")
            return
        if shape and self.rows > 1:
            bad = 0
            for _ in range(3):
                x, res, post, comb = self._hc_data(g, D)
                with self._oracle():
                    ref = layer.hc_post(x, res, post, comb)
                got = self.hc_post(x, res, post, comb)
                bad += int((got.view(torch.int16) != ref.view(torch.int16)).sum())
            cal["hc_post"]["rows_check"] = {"rows": self.rows, "mismatches": bad}
            if bad:
                self._off("hc_post", f"{self.rows}-row launch differs from the 1-row reference row by row ({bad})")

    # end-to-end verifications (None = equal, else why) ---------------------------------------------
    @_cpu_default
    def _rand_gate_up(self, g, G_, I):
        """bf16 [G, I] gate/up with clamp edges: |v| up to 64 so both clamps bind, exact +-lim, zeros."""
        v = _rand_act(g, 2 * G_, I, lo=-3, hi=7).view(2, G_, I)
        if self.lim > 0:
            v[:, :, 0] = self.lim
            v[:, :, 1] = -self.lim
            v[:, :, 2] = self.lim * 1.5
            v[:, :, 3] = -self.lim * 1.5
        v[:, :, 4] = 0.0
        return v[0].to(self.dev), v[1].to(self.dev)

    def _ref_swiglu(self, gate, up, w, lim, dtype):
        """Expert.forward's glue (model.py:845-853) as DGMaskedMoE / the reference run it."""
        gate, up = gate.float(), up.float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up
        if w is not None:
            h = w.view(-1, 1) * h
        return h.to(dtype)

    def _verify_swiglu_routed(self, g) -> Optional[str]:
        R, b = self.R, self.o.dg
        if self.rows > 1:
            return self._verify_swiglu_routed_rows(g)
        for _ in range(3):
            gate, up = self._rand_gate_up(g, b.k, b.inter)
            w = (torch.rand(b.k, generator=g, device="cpu") * 2).float().to(self.dev)
            h = self._ref_swiglu(gate, up, w, b.swiglu_limit, torch.bfloat16)
            q_ref, s_ref = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            s_ref32 = torch.empty(b.k, b.inter // BLOCK, dtype=torch.float32, device=self.dev)
            from runtime.dg_moe import e8m0_to_f32
            e8m0_to_f32(s_ref, s_ref32, b.e8m0_method)
            q = torch.empty(b.k, b.inter, dtype=FP8, device=self.dev)
            s = torch.empty(b.k, b.inter // BLOCK, dtype=torch.float32, device=self.dev)
            launch_swiglu_quant(gate, up, w, q, s, self.silu_tab, self.code_tab, b.swiglu_limit)
            if not (torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
                    and torch.equal(s.view(torch.int32), s_ref32.view(torch.int32))):
                return "routed SwiGLU+act_quant differs from the reference glue"
        return None

    def _verify_swiglu_routed_rows(self, g) -> Optional[str]:
        """M1-T10-PORT: the IDX form (pairs at arbitrary A rows) == index_select + the reference glue + act_quant +
        e8m0_to_f32 + index_copy (DGMaskedMoERows' path), rows not addressed untouched."""
        from runtime.dg_moe import e8m0_to_f32

        R, b = self.R, self.o.dg
        P = b.k                         # R*k slots = the pair count
        A = 2 * P
        I = b.inter
        for _ in range(3):
            gate, up = self._rand_gate_up(g, A, I)
            ri = torch.randperm(A, generator=g, device="cpu")[:P].to(self.dev)
            w = (torch.rand(P, generator=g, device="cpu") * 2).float().to(self.dev)
            h = self._ref_swiglu(gate.index_select(0, ri), up.index_select(0, ri), w, b.swiglu_limit, torch.bfloat16)
            q_ref, s_ref = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            s_ref32 = torch.empty(P, I // BLOCK, dtype=torch.float32, device=self.dev)
            e8m0_to_f32(s_ref, s_ref32, b.e8m0_method)
            qw = torch.zeros(A, I, dtype=FP8, device=self.dev)
            sw = torch.zeros(A, I // BLOCK, dtype=torch.float32, device=self.dev)
            qw.view(torch.uint8).index_copy_(0, ri, q_ref.contiguous().view(torch.uint8))
            sw.index_copy_(0, ri, s_ref32)
            q, s = torch.zeros_like(qw), torch.zeros_like(sw)
            launch_swiglu_quant(gate, up, w, q, s, self.silu_tab, self.code_tab, b.swiglu_limit, rows_idx=ri)
            if not (torch.equal(q.view(torch.uint8), qw.view(torch.uint8)) and torch.equal(s.view(torch.int32),
                                                                                         sw.view(torch.int32))):
                return f"routed SwiGLU+act_quant at {self.rows} rows (pair rows) differs from the reference glue"
        return None

    def _verify_swiglu_shared(self, g) -> Optional[str]:
        R = self.R
        ex = self.o.model.layers[0].ffn.shared_experts
        I = ex.w1.out_features
        T_ = self.rows  # the step's rows (the kernel's grid axis 1; M1-T10-FIX)
        for _ in range(3):
            gate, up = self._rand_gate_up(g, T_, I)
            h = self._ref_swiglu(gate, up, None, ex.swiglu_limit, torch.bfloat16)
            q_ref, s_ref = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            q = torch.empty(T_, I, dtype=FP8, device=self.dev)
            s = torch.empty(T_, I // BLOCK, dtype=E8, device=self.dev)
            launch_swiglu_quant(gate, up, None, q, s, self.silu_tab, self.code_tab, ex.swiglu_limit)
            if not (torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
                    and torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))):
                return "shared SwiGLU+act_quant differs from the reference glue"
        return None

    def _verify_dg_act(self, g) -> Optional[str]:
        from runtime.dg_moe import e8m0_to_f32

        R, b = self.R, self.o.dg
        if self.rows > 1:
            return self._verify_dg_act_rows(g)
        for _ in range(3):
            x = _rand_act(g, 1, b.dim).to(self.dev)
            q_ref, s_ref = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            s32 = torch.empty(b.k, b.dim // BLOCK, dtype=torch.float32, device=self.dev)
            e8m0_to_f32(s_ref, s32, b.e8m0_method)
            q = torch.empty(b.k, b.dim, dtype=FP8, device=self.dev)
            s = torch.empty(b.k, b.dim // BLOCK, dtype=torch.float32, device=self.dev)
            launch_act_quant_bcast(x, q, s, self.code_tab, b.k)
            if not (torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8).expand(b.k, -1))
                    and torch.equal(s.view(torch.int32), s32.view(torch.int32))):
                return "act_quant into the DeepGEMM rows differs from act_quant + load_act"
        return None

    def _verify_dg_act_rows(self, g) -> Optional[str]:
        """M1-T10-PORT: act_quant of R rows scattered to pair A rows == act_quant + index_select + e8m0_to_f32 +
        index_copy (DGMaskedMoERows._load)."""
        from runtime.dg_moe import e8m0_to_f32

        R, b = self.R, self.o.dg
        T_, P, D = self.rows, b.k, b.dim
        A = b.xq[1].numel() // D
        for _ in range(3):
            x = _rand_act(g, T_, D).to(self.dev)
            prow = torch.randint(0, T_, (P,), generator=g, device="cpu").to(self.dev)
            arow = torch.randperm(A, generator=g, device="cpu")[:P].to(self.dev)
            q_ref, s_ref = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            s32 = torch.empty(T_, D // BLOCK, dtype=torch.float32, device=self.dev)
            e8m0_to_f32(s_ref, s32, b.e8m0_method)
            qw = torch.zeros(A, D, dtype=FP8, device=self.dev)
            sw = torch.zeros(A, D // BLOCK, dtype=torch.float32, device=self.dev)
            qw.view(torch.uint8).index_copy_(0, arow, q_ref.contiguous().view(torch.uint8).index_select(0, prow))
            sw.index_copy_(0, arow, s32.index_select(0, prow))
            q, s = torch.zeros_like(qw), torch.zeros_like(sw)
            launch_act_quant_bcast(x, q, s, self.code_tab, 1, src_rows=prow, dst_rows=arow)
            if not (torch.equal(q.view(torch.uint8), qw.view(torch.uint8))
                    and torch.equal(s.view(torch.int32), sw.view(torch.int32))):
                return f"act_quant into the pair A rows at {self.rows} rows differs from act_quant + _load"
        return None

    def _verify_combine(self, g) -> Optional[str]:
        b = self.o.dg
        if self.rows > 1:
            return self._verify_combine_rows(g)
        for _ in range(3):
            rows = _rand_act(g, b.k, b.dim, lo=-8, hi=8).to(self.dev)
            sh = _rand_act(g, 1, b.dim, lo=-8, hi=8).to(self.dev)
            y = torch.zeros(1, b.dim, dtype=torch.float32, device=self.dev)
            for j in range(b.k):
                y += rows[j]
            y += sh
            ref = y.type_as(sh)
            got = self.combine(rows, sh, sh)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return "combine differs from the fp32 slot-order sum"
        return None

    def _verify_combine_rows(self, g) -> Optional[str]:
        """M1-T10-PORT: the posmat combine + shared + bf16 == moe_rows.combine_rows; y += shared; y.type_as(x)."""
        from runtime.moe_rows import combine_rows

        b, T_, k = self.o.dg, self.rows, self.o.topk
        A = 2 * b.k
        for _ in range(3):
            rows = _rand_act(g, A, b.dim, lo=-8, hi=8).to(self.dev)
            sh = _rand_act(g, T_, b.dim, lo=-8, hi=8).to(self.dev)
            posmat = torch.stack([torch.randperm(A, generator=g, device="cpu")[:k] for _ in range(T_)]).to(self.dev)
            y = combine_rows(rows, posmat, sh)
            y += sh
            ref = y.type_as(sh)
            got = self.combine_rows(rows, posmat, sh, sh)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return f"combine at {self.rows} rows differs from combine_rows + shared"
        return None

    def _verify_o_path(self, g) -> Optional[str]:
        R = self.R
        attn = self.o.model.layers[0].attn
        G_ = attn.n_local_groups
        T_ = self.rows
        data = self._rope_data(g, attn, True)
        if T_ > 1:   # M1-T10-PORT: T rows at T different positions per draw
            data = [(torch.cat([data[(i + j) % len(data)][0] for j in range(T_)], 1),
                     torch.cat([data[(i + j) % len(data)][1] for j in range(T_)], 0)) for i in range(4)]
        for x, f in data[:4]:
            ref = x.clone()
            R.apply_rotary_emb(ref[..., -self.rope_dim:], f, True)
            ref = ref.view(1, T_, G_, -1)
            R.act_quant(ref, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)
            got = x.clone()
            self.o_path(attn, got, f)
            if not torch.equal(got.view(torch.int16).view(-1), ref.view(torch.int16).view(-1)):
                return "inverse RoPE + QDQ differs from apply_rotary_emb + act_quant"
        return None

    def _verify_q_proj(self, g) -> Optional[str]:
        R, o = self.R, self.o
        attn = o.model.layers[0].attn
        K = attn.wq_b.in_features
        prev = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with o._active():
                T_ = self.rows   # M1-T10-PORT: T rows, row m at position pos + m
                for t in range(3):
                    qr = _rand_act(g, T_, K, lo=-4, hi=4).view(1, T_, K).to(self.dev)
                    pos = int(torch.randint(1, attn.freqs_cis.size(0) - T_ + 1, (1,), generator=g, device="cpu"))
                    f = attn.freqs_cis[pos:pos + T_]
                    got = self.q_proj(attn, qr, f)
                    if got is None:
                        return "wq_b does not take the GEMV in this step's scope"
                    ref = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
                    R.apply_rotary_emb(ref[..., -self.rope_dim:], f)
                    if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                        return "wq_b GEMV + RoPE epilogue differs from GEMV then apply_rotary_emb"
        finally:
            torch.set_default_dtype(prev)
            from runtime import patches
            patches.fp8_gemv_stats(reset=True)
        return None

    def _verify_gather(self, g) -> Optional[str]:
        """Layer 0's real pointer tables: the one-launch gather == 6 x gather_rows (+ dg: convert_scales)."""
        from runtime.static_kernels import gather_rows

        o = self.o
        sm = o.moes[0]
        E = sm.moe.n_routed_experts
        if self.rows > 1:   # M1-T10-PORT: the R*k dedup slots (ids ascending, padded with repeats like uniq)
            ids = torch.randint(0, E, (o.n_slots,), generator=g, device="cpu").sort().values.to(self.dev)
        else:
            ids = torch.randperm(E, generator=g, device="cpu")[: o.topk].sort().values.to(self.dev)
        ref = {}
        for key, tab in sm.ptabs.items():
            gather_rows(tab, ids, o.staging[key], o.row_bytes[key])
            ref[key] = o.staging[key].clone()
        if self.scale_f32:
            for p in ("w1", "w3", "w2"):
                o.dg.convert_scales(p, o.staging[(p, "scale")])
                ref[("f32", p)] = o.dg.sf32[p].clone()
        for t in self._dsts:
            t.zero_()
        ids_prev = o.ids_step[sm.layer].clone() if o.counters else None
        self.gather(sm, ids)
        try:
            for i, key in enumerate([("w1", "weight"), ("w3", "weight"), ("w2", "weight")]):
                if not torch.equal(o.staging[key].view(torch.uint8), ref[key].view(torch.uint8)):
                    return f"gathered {key} differs"
            for i, p in enumerate(("w1", "w3", "w2")):
                if self.scale_f32:
                    if not torch.equal(o.dg.sf32[p].view(torch.int32), ref[("f32", p)].view(torch.int32)):
                        return f"fp32 scales of {p} differ"
                elif not torch.equal(o.staging[(p, "scale")].view(torch.uint8), ref[(p, "scale")].view(torch.uint8)):
                    return f"gathered ({p}, scale) differs"
            if o.counters and not torch.equal(o.ids_step[sm.layer], ids):
                return "ids_step row not written"
        finally:
            if ids_prev is not None:
                o.ids_step[sm.layer].copy_(ids_prev)
        return None

    def _verify_dedup(self, g) -> Optional[str]:
        """M1-T10-PORT: the one-launch dedup == moe_rows.dedup_routes (+ masked_m, the counters' rows and -- dg-inplace
        -- compact_cold's stage ids / sidx / m_indices tags) on random routings of the step's R x k pairs with heavy
        cross-row repeats, on layer 0's real hot-slot table."""
        from runtime import moe_rows as MR

        o = self.o
        T_, k = self.rows, o.topk
        sm = o.moes[0]
        E = sm.moe.n_routed_experts
        inpl = sm._route_kw().get("inpl") if hasattr(sm, "_route_kw") else None
        P = T_ * k
        for t in range(8):
            span = min(E, k + 2 + 3 * t)                  # few distinct experts first: many repeats
            idx = torch.stack([torch.randperm(span, generator=g, device="cpu")[:k] for _ in range(T_)])
            idx = (idx + int(torch.randint(0, E - span + 1, (1,), generator=g, device="cpu"))).to(self.dev)
            w = torch.rand(T_, k, generator=g, device="cpu").float().to(self.dev)
            br = T_ + (t % 3)
            if inpl is not None and inpl.get("v2"):   # M1-T10-MOE: private m_indices / gl / counter (live ones stay)
                lay = inpl.get("layout", "contiguous")
                if lay == "masked":
                    br = int(o.dg.block_rows)
                inpl = dict(inpl, mi=torch.full((P * br,), -1, dtype=torch.int32, device=self.dev)
                            if lay == "contiguous" else None,
                            gl=torch.zeros(max(int(inpl.get("G") or 0), 1), dtype=torch.int32, device=self.dev),
                            ovf=torch.zeros(1, dtype=torch.int32, device=self.dev))
            buf = MR.DedupBuffers(T_, k, self.dev)
            mm = torch.full((P,), -7, dtype=torch.int32, device=self.dev)
            ids = torch.full((P,), -7, dtype=torch.long, device=self.dev)
            uv = torch.full((P,), -7, dtype=torch.long, device=self.dev)
            got = MR.dedup_fused(buf, idx, w, br, mm=mm, ids_step=ids, uval=uv, inpl=inpl)
            ref = MR.dedup_routes(idx, w, br)
            ext = MR.rows_layout(ref, inpl, br, T_, k) if inpl is not None and inpl.get("v2") else None
            for key, v in ref.items():
                if ext is not None and key in ("arow", "posmat"):
                    v = ext[key]
                if not torch.equal(got[key], v):
                    return f"dedup {key} differs from dedup_routes"
            if not (torch.equal(mm, ref["counts"].int()) and torch.equal(ids, ref["uniq"])
                    and torch.equal(uv, ref["valid"])):
                return "dedup masked_m / counter rows differ"
            if ext is not None:
                if inpl["mi"] is not None:
                    got_tags = inpl["mi"].view(P, br)[:, :T_]
                    if not (torch.equal(got_tags, ext["tags"]) and bool((inpl["mi"].view(P, br)[:, T_:] == -1).all())):
                        return "dedup's m_indices tags differ"
                if ext["gl"] is not None and not torch.equal(inpl["gl"][: ext["gl"].numel()], ext["gl"]):
                    return "dedup's psum / masked_m vector differs"
                if not (torch.equal(got["wl"], ext["wl"]) and int(inpl["ovf"]) == int(ext["n_over"])
                        and torch.equal(got["sidx"], ext["sidx"]) and torch.equal(got["ids_stage"], ext["ids_stage"])):
                    return "dedup's work list / overflow count / stage ids differ"
                continue
            if inpl is not None:
                gq = inpl["gtab"].index_select(0, ref["uniq"]).long()
                cold, ok, grp, sidx = MR.compact_cold(gq, ref["counts"], int(inpl["H"]), int(inpl["S"]))
                stage = torch.where(ref["counts"] > 0, ref["uniq"], inpl["dummy"]) if inpl["has_dummy"] else ref["uniq"]
                n_ok = torch.where(ok, ref["counts"], 0)
                ar = torch.arange(T_, dtype=torch.long, device=self.dev).view(1, -1)
                tags = torch.where(ar < n_ok.view(-1, 1), grp.view(-1, 1), -1).to(torch.int32)
                if not (torch.equal(got["sidx"], sidx) and torch.equal(got["ids_stage"], stage)
                        and torch.equal(got["tags"], tags)):
                    return "dedup's dg-inplace stage ids / compact cold index / m_indices tags differ"
        return None

    # ---------------------------------------------------------------- report
    def describe(self) -> dict:
        d = {"impl": "fused", "triton": getattr(triton, "__version__", None) if HAVE_TRITON else None,
             "clusters": dict(self.status), "audit": dict(CLUSTER_AUDIT),
             "on": [c for c in CLUSTERS if self.on(c)],
             "approx_on": [c for c in CLUSTERS if self.status.get(c) == "approx"],
             "unfused": {c: s for c, s in self.status.items() if not self.on(c)},
             "modes": {"rope_fwd": list(self.rope_fwd), "rope_inv": list(self.rope_inv), "hc_sum": self.hc_mode,
                       "hc_pre_sum": self.hc_pre_mode},
             "calibration": _jsonable(self.calib), "disable_env": os.environ.get(DISABLE_ENV)}
        d["r2"] = self.describe_r2()
        d["r4"] = self.describe_r4()
        return d

    def glue_line(self) -> str:
        """One line for first light's `[glue]` print: clusters on (exact), approx, off, and the numerics outcome."""
        on = [c for c in CLUSTERS if self.status.get(c) == "on"]
        ap = [c for c in CLUSTERS if self.status.get(c) == "approx"]
        off = [c for c in CLUSTERS if not self.on(c)]
        out = self.describe_r2()["outcome"]
        num = ", ".join(f"{c} {'(i)' if (v or '').startswith('(i)') else '(ii)'}" for c, v in out.items())
        pol = getattr(self, "order_policy", "rows")
        how = (f"orders of the 1-row shapes, self-checked row by row at {self.rows} rows" if pol == "shape" else
               f"self-checked at {self.rows} rows")
        rows = (f"rows {self.rows} ({how}; not ported: {','.join(ROWS_WHY)}); " if self.rows > 1 else "")
        return (f"[glue] fused: {rows}on {','.join(on) or '-'}; approx {','.join(ap) or '-'}; off {','.join(off) or '-'}; "
                f"r2 numerics {num or '-'}; orders {self.describe_r2()['orders']}")


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v



@_cpu_default
def _rand_act(g, rows: int, K: int, lo: int = -6, hi: int = 7) -> torch.Tensor:
    """bf16 [rows, K] activations: N(0,1) per 32-block scaled by 2^U[lo, hi), a few zeros, and amax values exactly at
    the scale boundaries 448 * 2^k * {1, 1 - 2^-8, 1 + 2^-7} (CPU generator; the caller moves it)."""
    kb = K // BLOCK
    x = torch.randn(rows, kb, BLOCK, generator=g, device="cpu") * torch.exp2(torch.randint(lo, hi, (rows, kb, 1), generator=g, device="cpu").float())
    k = torch.randint(-12, 4, (rows, kb), generator=g, device="cpu").float()
    edge = 448.0 * torch.exp2(k) * torch.tensor([1.0, 1 - 2 ** -8, 1 + 2 ** -7], device="cpu")[torch.randint(0, 3, (rows, kb),
                                                                                                  generator=g, device="cpu")]
    sel = torch.rand(rows, kb, generator=g, device="cpu") < 0.3
    sign = torch.where(torch.rand(rows, kb, generator=g, device="cpu") < 0.5, -1.0, 1.0)
    x[..., 0] = torch.where(sel, edge * sign, x[..., 0])
    x[..., 1] = torch.where(torch.rand(rows, kb, generator=g, device="cpu") < 0.1, torch.zeros((), device="cpu"), x[..., 1])
    return x.view(rows, K).to(torch.bfloat16)
