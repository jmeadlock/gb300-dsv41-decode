"""pdl_kernels.py -- M2-L5 probe kernels: COPIES (not edits) of two of our Triton kernels with PDL added.

  * `pdl_gemv`       <- runtime/gemv_mxfp8.py `t9b_mxfp8_gemv` (the T9b-1 MXFP8 GEMV; ROPE epilogue dropped: the probe's
                        pair never uses it). The K loop is verbatim (tests/test_pdl_probe_cpu.py compares the loop text
                        with the runtime's); added: `USE_PDL` (gdc_wait + gdc_launch_dependents before the loop,
                        OUTSIDE it -- Triton 3.7.1 pipeliner bug #11905, W1 sec 1.3) and `PREFETCH` (producer-
                        independent L2 prefetch of this program's weight slice BEFORE the wait).
  * `pdl_norm_quant` <- runtime/fused_glue_r2.py `t9b2r2_norm_row` in the q-path form (RMSNorm of the GEMV's fp32 split
                        partials folded in the load as `_t9b4_ld`, then act_quant -> fp8 q + e8m0 s; no RoPE, no ring
                        slot, no QDQ). Added: `USE_PDL` (gdc_wait at entry before the first read of the producer's
                        output, gdc_launch_dependents right after) and row strides QRS / SRS so the quantized row can
                        be written into the first N columns of the NEXT GEMV's [R, K_next] activation buffer.
Everything else (helpers, reduction order, quantization) is copied line for line from the two runtime files; the
arithmetic is therefore the runtime's, and PDL / prefetch only move timing. `runtime/` is not edited and not imported
here (this file is loaded under tests/triton_shim.py by the Mac test, which must not pollute sys.modules).

The incantation (what a kernel owner copies):

    kernel[grid](args..., USE_PDL=True, num_warps=..., launch_pdl=True)          # host: the launch attribute
    # kernel body, first thing before the first read of any producer output, never inside a loop:
    tl.extra.cuda.gdc_wait()                                                      # griddepcontrol.wait
    tl.extra.cuda.gdc_launch_dependents()                                         # griddepcontrol.launch_dependents
    # (vLLM reference/vllm-0.31.0/deepseek_v41/common/ops/query_quant.py:40-46,108,126 does exactly this)
    # tier B (GEMV): before gdc_wait, prefetch the producer-independent bytes (weights, weight scales) into L2:
    #   PREFETCH=1  prefetch.global.L2 [addr]            one per 128 B line of each weight row's K slice
    #   PREFETCH=2  cp.async.bulk.prefetch.L2.global [addr], bytes   one per weight row (K slice contiguous)

Shapes are [N, K] weights: wq_a-like 1280 x 5120 ("A") and 5120 x 8192 ("B"), 6 rows.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple

import torch

try:  # GPU image only; the Mac test loads this file under tests/triton_shim.py
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

BLOCK = 32                 # MXFP8 block (model.py:197)
MMA_BLOCK_M = 16
GEMV_MAX_M = 8
DEFAULT_SMS = 148          # GB300; the probe passes the real SM count
SPLIT_MIN_KB = 4
SPLIT_PROGRAMS_PER_SM = 2
LINE = 128                 # L2 line / prefetch granule in bytes
NORM_WARPS = 8             # fused_glue_r2.NORM_WARPS
NORM_EPS = 1e-6
SHAPES = {"A": (1280, 5120), "B": (5120, 8192)}          # (N, K)
PF_NONE, PF_LINES, PF_BULK = 0, 1, 2


# ------------------------------------------------------------------------------------------------ Triton kernels
if HAVE_TRITON:

    # --- helpers: verbatim copies (runtime/gemv_mxfp8.py _t9b_pow2; runtime/fused_glue_r2.py the rest)
    @triton.jit
    def _t9b_pow2(c):
        """pow2_code line for line: int32 e8m0 code -> fp32 2^(c-127) (0 -> subnormal 2^-127, 255 -> NaN), exact."""
        bits = tl.where(c == 0, 0x400000, c << 23)
        bits = tl.where(c == 255, 0x7FC00000, bits)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _t9b2_code(amax, tab):
        """e8m0 code of a block from its amax (a bf16 value as fp32): the reference's own code, by table."""
        idx = (amax.to(tl.int32, bitcast=True) >> 16) & 0xFFFF
        return tl.load(tab + idx).to(tl.int32)

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

    # --- PDL tier-B helpers (NEW, not in the runtime): L2 prefetch of producer-independent bytes
    @triton.jit
    def _pf_l2(ptr):
        """prefetch.global.L2 of the 128 B line holding each pointer of the block (one instruction per element)."""
        return tl.inline_asm_elementwise("{ prefetch.global.L2 [$1]; mov.u32 $0, 0; }", "=r,l",
                                         [ptr.to(tl.int64)], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _pf_bulk(ptr, nbytes):
        """cp.async.bulk.prefetch.L2.global [ptr], nbytes per element (16 B aligned address, size % 16 == 0)."""
        return tl.inline_asm_elementwise("{ cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }", "=r,l,r",
                                         [ptr.to(tl.int64), nbytes], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def pdl_gemv(xq, xs, w, s, out, ws, M, N, KB, PER,
                 stride_xm, stride_xsm, stride_wn, stride_sn, stride_om,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KU: tl.constexpr, EVEN: tl.constexpr,
                 MMA: tl.constexpr, SPLIT: tl.constexpr,
                 USE_PDL: tl.constexpr = False, PREFETCH: tl.constexpr = 0, PF_LINES: tl.constexpr = 1,
                 PF_SLINES: tl.constexpr = 1):
        """t9b_mxfp8_gemv (runtime/gemv_mxfp8.py) without the RoPE epilogue. Program (pn, ks): output columns
        [pn*BLOCK_N, (pn+1)*BLOCK_N) for all M rows over K blocks [ks*PER, min((ks+1)*PER, KB)); SPLIT: fp32 acc ->
        ws[ks] (reduced by whoever consumes it), else bf16 -> out.
        PDL: USE_PDL -> gdc_wait() then gdc_launch_dependents() once per program, before the K loop (outside any
        loop). PREFETCH (1 = per-line prefetch.global.L2, 2 = per-row cp.async.bulk.prefetch.L2.global) issues, BEFORE
        the wait, the L2 prefetch of this program's weight slice and weight-scale slice -- bytes the producer does
        not write. The activation (xq / xs) is the producer's output: never prefetched, read only after the wait."""
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
        if PREFETCH > 0:
            # weight scales: BLOCK_N // 32 distinct scale rows, each a contiguous run of kb_hi - kb_lo bytes
            pf_sr = tl.minimum(pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32), (N + 31) // 32 - 1)
            pf_sj = tl.arange(0, PF_SLINES)
            pf_soff = kb_lo + tl.minimum(pf_sj * 128, kb_hi - kb_lo - 1)
            _pf_l2(s + pf_sr.to(tl.int64)[:, None] * stride_sn + pf_soff[None, :])
        if PREFETCH == 1:
            pf1_rn = tl.minimum(rn, N - 1)                  # rows past N would only re-prefetch a valid row
            pf1_j = tl.arange(0, PF_LINES)
            pf1_off = kb_lo * 32 + tl.minimum(pf1_j * 128, (kb_hi - kb_lo) * 32 - 1)
            _pf_l2(w + pf1_rn.to(tl.int64)[:, None] * stride_wn + pf1_off[None, :])
        elif PREFETCH == 2:
            pf2_rn = tl.minimum(rn, N - 1)
            pf2_ptr = w + pf2_rn.to(tl.int64) * stride_wn + kb_lo * 32
            pf2_n = tl.zeros([BLOCK_N], dtype=tl.int32) + (kb_hi - kb_lo) * 32
            _pf_bulk(pf2_ptr, pf2_n)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
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
    def pdl_norm_quant(x, nw, out, q, s, tab, eps, fac, wf, XS, QRS, SRS,
                       N: tl.constexpr, T: tl.constexpr, V: tl.constexpr, U: tl.constexpr, DEC: tl.constexpr,
                       DIV: tl.constexpr, BB: tl.constexpr, NS: tl.constexpr = 0, USE_PDL: tl.constexpr = False):
        """t9b2r2_norm_row (runtime/fused_glue_r2.py), q-path form. Program t = token row t: x row t (bf16 [R, N], or
        NS > 0: the GEMV's fp32 split partials [NS, R, N], split stride XS, reduced on load in the split-K reduce's
        order); RMSNorm statistic in order T/V/U/DEC/DIV; out[t] = bf16(weight * (x * r)); act_quant -> q[t * QRS +
        0:N] fp8 and s[t * SRS + 0:N/32] e8m0 (QRS / SRS = the destination row strides: the next GEMV's [R, K_next]).
        PDL: gdc_wait() before the first read of the producer's output, gdc_launch_dependents() right after (so the
        successor GEMV's CTAs can be admitted and prefetch its weights while this kernel runs)."""
        t = tl.program_id(0)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        x = x + t * N
        q = q + t * QRS
        s = s + t * SRS
        acc = tl.zeros([T, V], dtype=tl.float32)
        for k in range((N + T * V - 1) // (T * V)):
            e = _r2_offs(k, T, V, U)
            m = e < N
            v = _t9b4_ld(x, e, m, NS, XS)
            acc = acc + v * v
        r = libdevice.rsqrt(_r2_fin(_r2_finish(acc, T, V, DEC), fac, wf, DIV) + eps)
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
            code = _t9b2_code(tl.maximum(tl.max(tl.abs(a), axis=1), tl.max(tl.abs(b), axis=1)), tab)
            inv = _t9b2_ipow2(code)[:, None]
            qa = _t9b2_q8(a, inv)
            qb = _t9b2_q8(b, inv)
            tl.store(q + ce, qa, mask=m)
            tl.store(q + ce + 1, qb, mask=m)
            tl.store(s + bi, code.to(tl.uint8), mask=bok)
            tl.store(base + ce, a.to(tl.bfloat16), mask=m)
            tl.store(base + ce + 1, b.to(tl.bfloat16), mask=m)


# ------------------------------------------------------------------------------------------------ host side
@dataclass(frozen=True)
class Cfg:
    """runtime.gemv_mxfp8.GemvConfig's fields (same meaning)."""
    block_n: int
    ku: int
    num_warps: int
    num_stages: int
    split_k: int = 0     # 0 = auto_split_k for the shape and SM count
    mma: int = 0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


DEFAULT_CFG = Cfg(32, 4, 4, 1, 0, 0)     # runtime DEFAULT_CONFIG (the untuned, proven form)


def block_m(m: int) -> int:
    if not 1 <= int(m) <= GEMV_MAX_M:
        raise ValueError(f"pdl gemv: M = {m} rows outside [1, {GEMV_MAX_M}]")
    return 1 << (int(m) - 1).bit_length()


def split_plan(KB: int, split_k: int) -> Tuple[int, int]:
    """(PER, NSPLIT) -- runtime.gemv_mxfp8.split_plan."""
    KB = int(KB)
    s = max(1, min(int(split_k), KB))
    per = -(-KB // s)
    return per, -(-KB // per)


def auto_split_k(N: int, K: int, block_n: int, sms: int) -> int:
    """runtime.gemv_mxfp8.auto_split_k."""
    KB = int(K) // BLOCK
    tiles = -(-int(N) // int(block_n))
    want = -(-SPLIT_PROGRAMS_PER_SM * int(sms) // tiles)
    cap = max(1, KB // SPLIT_MIN_KB)
    return max(1, min(want, cap))


def resolve(cfg: Cfg, N: int, K: int, sms: int = DEFAULT_SMS) -> Cfg:
    if int(cfg.split_k) > 0:
        return cfg
    return Cfg(cfg.block_n, cfg.ku, cfg.num_warps, cfg.num_stages, auto_split_k(N, K, cfg.block_n, sms), cfg.mma)


def gemv_plan(cfg: Cfg, M: int, N: int, K: int) -> dict:
    """runtime.gemv_mxfp8.launch_plan (metadata only)."""
    kb = K // BLOCK
    per, nsplit = split_plan(kb, cfg.split_k)
    return {"grid": (-(-N // cfg.block_n), nsplit), "PER": per, "NSPLIT": nsplit,
            "BLOCK_M": MMA_BLOCK_M if cfg.mma else block_m(M), "EVEN": per % cfg.ku == 0 and kb % per == 0,
            "SPLIT": nsplit > 1}


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def prefetch_lines(per: int) -> int:
    """Points per weight row for PREFETCH=1: ceil(PER*32/128) line starts + 1 clamped to the slice's last byte (an
    unaligned run of L bytes touches up to ceil(L/128) + 1 lines), rounded up to a power of two (tl.arange)."""
    return _np2(-(-int(per) * BLOCK // LINE) + 1)


def prefetch_slines(per: int) -> int:
    """Same for the weight-scale run (PER bytes per scale row)."""
    return _np2(-(-int(per) // LINE) + 1)


def last_pow2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1)


def norm_order(N: int, rows: int = 1) -> dict:
    """runtime.fused_glue_r2.predicted_order(N, rows) as order_args(): torch's reduce configuration (V = 4
    accumulators, T threads, vectorized layout, decreasing tree below 32). A fixed legal order is all the probe
    needs: PDL changes no reduction order."""
    V = 4 if N >= 4 else 1
    d0 = min(last_pow2(max(1, N // V)), 512)
    d1 = min(last_pow2(rows), 512)
    bw = min(d0, 32)
    bh = min(d1, 512 // bw)
    return {"T": min(d0, 512 // bh), "V": V, "U": False, "DEC": True, "DIV": False}


def fac_of(w: int) -> float:
    """RN_fp32(1/w) as a Python float (runtime.fused_glue_r2.fac_of)."""
    return float(torch.tensor(1.0, dtype=torch.float32, device="cpu")
                 / torch.tensor(float(w), dtype=torch.float32, device="cpu"))


def launch_gemv(xq, xs, w, s, out, ws, cfg: Cfg, *, pdl: bool = False, prefetch: int = 0) -> dict:
    """One pdl_gemv launch. xq [M, K] fp8, xs [M, K/32] e8m0, w [N, K] fp8, s [ceil(N/32), K/32] e8m0, out [M, N] bf16
    (unused when the plan splits), ws fp32 [NSPLIT, M, N] (the split partials, when it does). `launch_pdl=` is the
    launch attribute (cudaLaunchAttributeProgrammaticStreamSerialization); USE_PDL the in-kernel wait/trigger;
    prefetch 0 / 1 / 2 as the module docstring. Returns the plan."""
    if not HAVE_TRITON:  # pragma: no cover
        raise RuntimeError("pdl_gemv needs triton")
    M, K = xq.shape
    N = w.size(0)
    p = gemv_plan(cfg, M, N, K)
    pf = {}
    if prefetch:
        pf = dict(PF_LINES=prefetch_lines(p["PER"]), PF_SLINES=prefetch_slines(p["PER"]))
    pdl_gemv[p["grid"]](
        xq, xs.view(torch.uint8), w, s.view(torch.uint8), out, ws, M, N, K // BLOCK, p["PER"],
        xq.stride(0), xs.stride(0), w.stride(0), s.stride(0), out.stride(0),
        BLOCK_M=p["BLOCK_M"], BLOCK_N=cfg.block_n, KU=cfg.ku, EVEN=p["EVEN"], MMA=bool(cfg.mma), SPLIT=p["SPLIT"],
        USE_PDL=bool(pdl), PREFETCH=int(prefetch), **pf,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages, launch_pdl=bool(pdl))
    return p


def launch_norm_quant(x, nw, out, q, s, tab, *, N: int, R: int, ns: int, xs_stride: int, qrs: int, srs: int,
                      pdl: bool = False) -> None:
    """One pdl_norm_quant launch (grid R). x: the GEMV's bf16 output [R, N] (ns = 0) or its fp32 partials
    [ns, R, N] (split stride xs_stride = R * N); out bf16 [R, N]; q fp8 / s e8m0 destination buffers whose row
    strides are qrs / srs (the first N columns of each row are written)."""
    if not HAVE_TRITON:  # pragma: no cover
        raise RuntimeError("pdl_norm_quant needs triton")
    o = norm_order(N)
    pdl_norm_quant[(R,)](
        x, nw, out, q, s.view(torch.uint8), tab, NORM_EPS, fac_of(N), float(N), int(xs_stride), int(qrs), int(srs),
        N=N, T=o["T"], V=o["V"], U=o["U"], DEC=o["DEC"], DIV=o["DIV"], BB=min(16, _np2(N // BLOCK)), NS=int(ns),
        USE_PDL=bool(pdl), num_warps=NORM_WARPS, enable_fp_fusion=False, launch_pdl=bool(pdl))


def describe() -> Dict[str, object]:
    return {"triton": getattr(triton, "__version__", None) if HAVE_TRITON else None, "shapes": SHAPES,
            "default_cfg": DEFAULT_CFG.as_dict()}
