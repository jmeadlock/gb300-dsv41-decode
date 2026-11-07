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

def _cpu_default(fn):
    """Synthetic-input builders were written under a CPU default device; at runtime the default device is cuda.
    Run them with a CPU default so every bare constructor (tensor/arange/zeros/rand) lands on the CPU; callers move."""
    import functools
    @functools.wraps(fn)
    def w(*a, **k):
        with torch.device("cpu"):
            return fn(*a, **k)
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
CLUSTERS = R1_CLUSTERS + R2_CLUSTER_NAMES
CLUSTER_AUDIT = {"gather": "#11+#9", "dg_act": "#9", "swiglu_routed": "#2+#9", "swiglu_shared": "#2",
                 "moe_combine": "#8", "hc_post": "#7", "o_rope_qdq": "#10", "q_rope_epilogue": "#13",
                 "concat_free": "#16", "hc_norm": "#1", "mix_stat": "#5", "kv_path": "#3", "q_path": "#6",
                 "router": "#4 (non-topk)", "engram": "#15", "indexer_post": "#14 (non-topk)"}
CONCAT_FREE_WHY = ("not implemented in round 1: the window ring is per layer while the compressed cache and the "
                   "published topk_idxs are shared by every layer of a kv/index group, so one buffer per layer "
                   "needs either per-layer copies of the compressed rows (no launch saved) or re-allocating the "
                   "model's caches into per-group pools with per-layer index offsets; deferred, 76 launches")
ACT_CLUSTERS = ("dg_act", "swiglu_routed", "swiglu_shared", "o_rope_qdq",   # need the act_quant table parity
                "hc_norm", "kv_path", "q_path")
DISABLE_ENV = "DSV41_GLUE_DISABLE"
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
    def t9b2_act_quant_bcast(x, x_rs, q, q_gs, s, s_gs, G, KB, tab, BB: tl.constexpr, S_F32: tl.constexpr):
        """Program (pb, r): 32-blocks [pb*BB, (pb+1)*BB) of input row r (bf16, row stride x_rs) -> fp8 values and the
        block scales (fp32 value or uint8 e8m0 code) written to output rows r*G + g for g < G (strides q_gs, s_gs):
        dg_act writes one activation row into row 0 of every DeepGEMM group this way."""
        pb = tl.program_id(0)
        r = tl.program_id(1)
        bi = pb * BB + tl.arange(0, BB)
        bok = bi < KB
        cols = bi[:, None] * 32 + tl.arange(0, 32)[None, :]
        m = bok[:, None] & (cols >= 0)
        v = tl.load(x + r * x_rs + cols, mask=m, other=0.0).to(tl.float32)
        code = _t9b2_code(tl.max(tl.abs(v), axis=1), tab)
        qv = _t9b2_q8(v, _t9b2_ipow2(code)[:, None])
        for g in range(G):
            o = r * G + g
            tl.store(q + o * q_gs + cols, qv, mask=m)
            if S_F32:
                tl.store(s + o * s_gs + bi, _t9b2_pow2(code), mask=bok)
            else:
                tl.store(s + o * s_gs + bi, code.to(tl.uint8), mask=bok)

    @triton.jit
    def t9b2_swiglu_quant(g, g_gs, u, u_gs, w, q, q_gs, s, s_gs, KB, silu_tab, tab, lim,
                          BB: tl.constexpr, HAS_W: tl.constexpr, HAS_LIM: tl.constexpr, S_F32: tl.constexpr):
        """Program (pb, j): group/row j, 32-blocks [pb*BB, (pb+1)*BB). gate/up bf16 -> silu(clamp(gate)) by table,
        clamp(up), silu*up, (w[j] *), bf16 RNE, then act_quant -> fp8 q and the scale (fp32 or e8m0 code)."""
        pb = tl.program_id(0)
        j = tl.program_id(1)
        bi = pb * BB + tl.arange(0, BB)
        bok = bi < KB
        cols = bi[:, None] * 32 + tl.arange(0, 32)[None, :]
        m = bok[:, None] & (cols >= 0)
        gv = tl.load(g + j * g_gs + cols, mask=m, other=0.0).to(tl.float32)
        uv = tl.load(u + j * u_gs + cols, mask=m, other=0.0).to(tl.float32)
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
        tl.store(q + j * q_gs + cols, _t9b2_q8(h, _t9b2_ipow2(code)[:, None]), mask=m)
        if S_F32:
            tl.store(s + j * s_gs + bi, _t9b2_pow2(code), mask=bok)
        else:
            tl.store(s + j * s_gs + bi, code.to(tl.uint8), mask=bok)

    @triton.jit
    def t9b2_moe_combine(d, d_gs, KS, sh, out, N, BLOCK_C: tl.constexpr):
        """out = bf16(((0 + d[0]) + d[1] + ... + d[KS-1]) + sh), fp32 adds in slot order."""
        i = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        ok = i < N
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for j in range(KS):
            acc = acc + tl.load(d + j * d_gs + i, mask=ok, other=0.0).to(tl.float32)
        acc = acc + tl.load(sh + i, mask=ok, other=0.0).to(tl.float32)
        tl.store(out + i, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=ok)

    @triton.jit
    def t9b2_hc_post(x, res, post, comb, out, D, HC: tl.constexpr, BLOCK_C: tl.constexpr, MODE: tl.constexpr):
        """Program (pb, j): out[j, i] = bf16(post[j]*x[i] + sum_r comb[r, j]*res[r, i]) (sum order: MODE)."""
        pb = tl.program_id(0)
        j = tl.program_id(1)
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

    @triton.jit
    def t9b2_o_rope_qdq(o, NB, freqs, tab, HD: tl.constexpr, RD: tl.constexpr, BB: tl.constexpr,
                        RE_MODE: tl.constexpr, IM_MODE: tl.constexpr):
        """In place on one bf16 row of NB 32-blocks (heads of HD, RoPE on the last RD of each): program pb takes
        blocks [pb*BB, (pb+1)*BB) as even/odd element pairs; rope columns get the INVERSE rotation by (c, -d)
        rounded to bf16; then the block's act_quant QDQ, bf16(fp32(q) * s), is stored back."""
        pb = tl.program_id(0)
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
def launch_act_quant_bcast(x, q, s, code_tab, groups: int):
    """x [R, K] bf16 (unit inner stride) -> q rows (r*groups + g) fp8 [.., K] and s rows (fp32 or e8m0/uint8)."""
    R_, K = x.shape
    KB = K // BLOCK
    s_f32 = s.dtype == torch.float32
    if _kern(x):
        su = s if s_f32 else s.view(torch.uint8)
        t9b2_act_quant_bcast[(triton.cdiv(KB, AQ_BB), R_)](
            x, x.stride(0), q, q.stride(0), su, su.stride(0), int(groups), KB, code_tab, BB=AQ_BB, S_F32=s_f32,
            **LAUNCH)
        return
    qv, code = act_quant_torch(x.float(), code_tab)
    for r in range(R_):
        for g in range(groups):
            o = r * groups + g
            q[o].copy_(qv[r])
            if s_f32:
                s[o].copy_(pow2_code(code[r]))
            else:
                s.view(torch.uint8)[o].copy_(code[r].to(torch.uint8))


def launch_swiglu_quant(g, u, w, q, s, silu_tab, code_tab, lim: float):
    """g, u: [G, I] bf16 row views (unit inner stride); w: [G] fp32 or None; q [G, I] fp8; s [G, I/32] fp32/e8m0."""
    G_, I = g.shape
    KB = I // BLOCK
    s_f32 = s.dtype == torch.float32
    if _kern(g):
        su = s if s_f32 else s.view(torch.uint8)
        t9b2_swiglu_quant[(triton.cdiv(KB, AQ_BB), G_)](
            g, g.stride(0), u, u.stride(0), w if w is not None else g, q, q.stride(0), su, su.stride(0), KB,
            silu_tab, code_tab, float(lim), BB=AQ_BB, HAS_W=w is not None, HAS_LIM=lim > 0, S_F32=s_f32, **LAUNCH)
        return
    h = swiglu_torch(g, u, w, silu_tab, lim)
    qv, code = act_quant_torch(h, code_tab)
    q.copy_(qv)
    if s_f32:
        s.copy_(pow2_code(code))
    else:
        s.view(torch.uint8).copy_(code.to(torch.uint8))


def launch_moe_combine(rows, shared, out):
    """rows [KS, N] bf16 (row views, unit inner), shared [N] bf16, out [N] bf16."""
    KS, N = rows.shape
    if _kern(rows):
        t9b2_moe_combine[(triton.cdiv(N, COMBINE_BLOCK),)](rows, rows.stride(0), KS, shared, out, N,
                                                            BLOCK_C=COMBINE_BLOCK, **LAUNCH)
        return out
    return out.copy_(combine_torch(rows, shared))


def launch_hc_post(x, res, post, comb, out, mode: int):
    """x [D], res [HC, D], out [HC, D] bf16 contiguous; post [HC], comb [HC, HC] fp32 contiguous."""
    HC, D = res.shape
    if _kern(x):
        t9b2_hc_post[(triton.cdiv(D, HC_BLOCK), HC)](x, res, post, comb, out, D, HC=HC, BLOCK_C=HC_BLOCK,
                                                     MODE=int(mode), **LAUNCH)
        return out
    return out.copy_(hc_post_torch(x, res, post, comb, mode))


def launch_o_rope_qdq(o_row, freqs_real, code_tab, head_dim: int, rope_dim: int, modes: Tuple[int, int]):
    """In place on o_row [N] bf16 contiguous (N = n_heads * head_dim): inverse RoPE + act_quant QDQ."""
    N = o_row.numel()
    if _kern(o_row):
        NB = N // BLOCK
        t9b2_o_rope_qdq[(triton.cdiv(NB, AQ_BB),)](o_row, NB, freqs_real, code_tab, HD=head_dim, RD=rope_dim,
                                                   BB=AQ_BB, RE_MODE=int(modes[0]), IM_MODE=int(modes[1]), **LAUNCH)
        return o_row
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
        for c in CLUSTERS:
            why = self._applicable(c, layers, mx)
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

    # ---------------------------------------------------------------- status
    def on(self, c: str) -> bool:
        return self.status.get(c) in ("on", "approx")

    def _off(self, c: str, why: str) -> None:
        if self.on(c):
            self.status[c] = f"off: {why}"
            self.log(f"[fused-glue] {c} OFF: {why}")

    def _applicable(self, c, layers, mx) -> Optional[str]:
        o, R = self.o, self.R
        dg = o.moe_impl == "dg-masked"
        if c in R2_CLUSTER_NAMES:
            return self._applicable_r2(c, layers, mx)
        if c in ACT_CLUSTERS and not mx:
            return "needs the MXFP8 act_quant globals (block 32, ue8m0, e8m0 scales)"
        if c == "concat_free":
            return CONCAT_FREE_WHY
        if c == "gather":
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
        launch_act_quant_bcast(x.view(1, -1), b.xq[1][:, 0], b.xs[1][:, 0], self.code_tab, b.k)

    def routed_swiglu(self, w: torch.Tensor) -> None:
        """Routed SwiGLU on the w1/w3 GEMM outputs (row 0 of each group) -> act_quant -> row 0 of the w2 input."""
        b = self.o.dg
        launch_swiglu_quant(b.d["w1"][:, 0], b.d["w3"][:, 0], w, b.xq[2][:, 0], b.xs[2][:, 0], self.silu_tab,
                            self.code_tab, b.swiglu_limit)

    def combine(self, out_rows: torch.Tensor, shared: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """y = bf16((0 + sum_j out[j]) + shared) as MoE.forward's fp32 accumulation (model.py:895-906)."""
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        launch_moe_combine(out_rows, shared.reshape(-1), y.view(-1))
        return y

    def shared_expert(self, ex, x: torch.Tensor) -> torch.Tensor:
        """Expert.forward (model.py:843-853, weights None) with the SwiGLU + bf16 + w2's act_quant in one kernel; w1,
        w3 and w2 keep their GEMM path (patches.fp8_linear_prequant routes w2 exactly as linear() would)."""
        from runtime.patches import fp8_linear_prequant

        ctx = self.ffn_pre(x) if self.on("hc_norm") else None
        if ctx is not None and ctx.get("q") is not None:      # round 2 #1: act_quant(x) already made by hc_norm
            xq = (ctx["q"].view(*x.shape), ctx["s"].view(*x.shape[:-1], -1))
            g = self.linear_pre(xq, ex.w1, x)
            u = self.linear_pre(xq, ex.w3, x)
        else:
            g = ex.w1(x)
            u = ex.w3(x)
        I = g.size(-1)
        g2, u2 = g.view(-1, I), u.view(-1, I)
        q = torch.empty(g2.shape, dtype=FP8, device=x.device)
        s = torch.empty(g2.size(0), I // BLOCK, dtype=E8, device=x.device)
        launch_swiglu_quant(g2, u2, None, q, s, self.silu_tab, self.code_tab, ex.swiglu_limit)
        return fp8_linear_prequant(q.view(*g.shape), s.view(*g.shape[:-1], I // BLOCK), ex.w2.weight)

    # ---------------------------------------------------------------- mHC post (#7)
    def hc_post(self, x, residual, post, comb) -> torch.Tensor:
        """Block.hc_post (model.py:964-968) for one row: x [1,1,D], residual [1,1,HC,D], post [1,1,HC], comb
        [1,1,HC,HC] -> [1,1,HC,D] (dtype of x)."""
        HC, D = residual.size(-2), residual.size(-1)
        out = torch.empty(residual.shape, dtype=x.dtype, device=x.device)
        launch_hc_post(x.reshape(D), residual.reshape(HC, D), post.reshape(HC).contiguous(),
                       comb.reshape(HC, HC).contiguous(), out.view(HC, D), self.hc_mode)
        return out

    # ---------------------------------------------------------------- attention (#10, #13)
    def o_path(self, attn, o: torch.Tensor, freqs_cis: torch.Tensor) -> None:
        """In place on the sparse_attn output o [1,1,H,head_dim]: inverse RoPE + act_quant QDQ (model.py:783-788)."""
        launch_o_rope_qdq(o.view(-1), torch.view_as_real(freqs_cis).reshape(-1, 2), self.code_tab, attn.head_dim,
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
        out = linear_fp8_rope(xq, s, w, w.scale, torch.view_as_real(freqs_cis).reshape(-1, 2), attn.head_dim,
                              attn.rope_head_dim, self.rope_fwd)
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
                          ("gather", self._verify_gather)):
                if self.on(c):
                    why = fn(g)
                    cal[f"verify_{c}"] = why or "equal"
                    if why:
                        self._off(c, f"self-check: {why}")
            self.calibrate_r2(cal, g)        # round 2 (runtime/fused_glue_r2.py): orders, then end-to-end checks
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
    def _hc_data(self, g, D):
        """Random hc_post inputs; in half the columns copy 0 and copy 2 carry the same large value and comb's row 2
        is minus row 0, so a0 + a2 cancels exactly and the 4-term sum's ORDER shows in the result (the three
        HC_SUM_MODES round differently there)."""
        HC = self.hc
        x = _rand_act(g, 1, D).view(1, 1, D)
        res = _rand_act(g, HC, D).view(1, 1, HC, D)
        post = (torch.rand(1, 1, HC, generator=g, device="cpu") * 2).float()
        comb = (torch.randn(1, 1, HC, HC, generator=g, device="cpu") * torch.exp2(torch.randint(-4, 5, (1, 1, HC, HC), generator=g, device="cpu")
                                                                       .float())).float()
        if HC == 4:
            big = (torch.randn(D, generator=g, device="cpu") * torch.exp2(torch.randint(10, 17, (D,), generator=g, device="cpu").float()))
            big = big.to(torch.bfloat16)
            cols = torch.arange(D, device="cpu") % 2 == 0
            res[0, 0, 0, cols] = big[cols]
            res[0, 0, 2, cols] = big[cols]
            comb[0, 0, 2, :] = -comb[0, 0, 0, :]
        return [t.to(self.dev) for t in (x, res, post, comb)]

    def _pick_hc(self, cal, g) -> None:
        layer = self.o.model.layers[0]
        D = layer.attn_norm.dim
        modes = HC_SUM_MODES if self.hc == 4 else (0,)
        mism = {m: 0 for m in modes}
        for _ in range(6):
            x, res, post, comb = self._hc_data(g, D)
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

    def _verify_swiglu_shared(self, g) -> Optional[str]:
        R = self.R
        ex = self.o.model.layers[0].ffn.shared_experts
        I = ex.w1.out_features
        for _ in range(3):
            gate, up = self._rand_gate_up(g, 1, I)
            h = self._ref_swiglu(gate, up, None, ex.swiglu_limit, torch.bfloat16)
            q_ref, s_ref = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            q = torch.empty(1, I, dtype=FP8, device=self.dev)
            s = torch.empty(1, I // BLOCK, dtype=E8, device=self.dev)
            launch_swiglu_quant(gate, up, None, q, s, self.silu_tab, self.code_tab, ex.swiglu_limit)
            if not (torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
                    and torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))):
                return "shared SwiGLU+act_quant differs from the reference glue"
        return None

    def _verify_dg_act(self, g) -> Optional[str]:
        from runtime.dg_moe import e8m0_to_f32

        R, b = self.R, self.o.dg
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

    def _verify_combine(self, g) -> Optional[str]:
        b = self.o.dg
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

    def _verify_o_path(self, g) -> Optional[str]:
        R = self.R
        attn = self.o.model.layers[0].attn
        G_ = attn.n_local_groups
        for x, f in self._rope_data(g, attn, True)[:4]:
            ref = x.clone()
            R.apply_rotary_emb(ref[..., -self.rope_dim:], f, True)
            ref = ref.view(1, 1, G_, -1)
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
                for t in range(3):
                    qr = _rand_act(g, 1, K, lo=-4, hi=4).view(1, 1, K).to(self.dev)
                    pos = int(torch.randint(1, attn.freqs_cis.size(0), (1,), generator=g, device="cpu"))
                    f = attn.freqs_cis[pos:pos + 1]
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
        return d

    def glue_line(self) -> str:
        """One line for first light's `[glue]` print: clusters on (exact), approx, off, and the numerics outcome."""
        on = [c for c in CLUSTERS if self.status.get(c) == "on"]
        ap = [c for c in CLUSTERS if self.status.get(c) == "approx"]
        off = [c for c in CLUSTERS if not self.on(c)]
        out = self.describe_r2()["outcome"]
        num = ", ".join(f"{c} {'(i)' if (v or '').startswith('(i)') else '(ii)'}" for c, v in out.items())
        return (f"[glue] fused: on {','.join(on) or '-'}; approx {','.join(ap) or '-'}; off {','.join(off) or '-'}; "
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
