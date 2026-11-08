"""Triton kernels for the T8 static step (GPU only; imported lazily by runtime/static_step.py and runtime/generate.py,
so the Mac tests never need triton).

  t8_gather_rows   the pointer-table gather: program (j, blk) copies block `blk` of the row whose address is
                   ptab[ids[j]] into slot j of a static staging buffer. The address may be an HBM pool slot or a UVA
                   address of registered Grace memory; the kernel does not care which (that is how cold experts cross
                   C2C inside the graph). T9b-3a: 16-byte vectors per lane (notes below).
  t8_step_marker   a one-program no-op with a fixed name (it increments a device counter), launched once per decode
                   step so nsys can find step boundaries after graph replay, where NVTX ranges do not survive
                   (T8-T9A-DESIGN.md section 2.3 step 6).
  t9b2_gather_tables  T9b-2 (`--glue-impl fused`, clusters #11 + #9): a layer's six gathers in one launch, dg-masked's
                   e8m0 -> fp32 scale conversion and the ids_step copy folded in (data movement + an exact decode).
                   T9b-3a: 16-byte vectors per lane.
  t9b5_head_fp32 / t9b5_mix_fp32   T9b-5 (`--fp32-impl fixed`, docs/HANDOFF-T9B5-FP32-FIXED.md): out[r, n] =
                   sum_k x[r, k] * w[n, k] in fp32 in ONE fixed order per output (fixed_dot below): the head
                   (F.linear(x.float(), head.weight), model.py:1014) and the mHC mixes (F.linear(x, hc_fn) * rsqrt,
                   model.py:956). Row-invariant by construction; the CPU mirror is fused_glue_r2.fixed_dot_torch.

T9b-3a Task B (docs/HANDOFF-T9B3A-ATTN-BOUND.md): the T8 / T9b-2 kernels (kept below, bodies unchanged, as
`t8_gather_rows_u8` / `t9b2_gather_tables_u8`: the GPU equality test's reference, gather_bench's baseline and
`DSV41_GATHER_IMPL=u8`) load uint8 through a row address read from memory, whose alignment Triton cannot know, so
every lane moves ONE byte: ~1.1-1.3 TB/s on an 8 TB/s part (STEP-ANATOMY-T9B1). The new kernels view rows as int64
words and declare the loaded row addresses 16-B aligned (`tl.multiple_of`), so each lane moves 16-B vectors; the
masks disappear when a row is a whole number of blocks (`EVEN*`, true for every real table at 8 KiB blocks: w1/w3/w2
rows 5,898,240 B = 720 x 8 KiB, scale rows 368,640 B = 45 x 8 KiB). Same bytes to the same places, so the output is
identical by construction (GPU test: torch.equal vs the _u8 kernels). Preconditions, checked on the host: row bytes
% 16 == 0 and destinations 16-B aligned (per call, metadata only: no sync), source row addresses 16-B aligned
(StaticStep._build_moe at build: the pool aligns host blocks to 256 B, torch allocations are >= 512-B aligned, row
sizes are multiples of 16, so this always holds for the real model). Triton unifies same-named variables across the
arms of a runtime `if` (types must match), so every arm below uses its own names.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import triton
import triton.language as tl

from runtime import pdl as _pdl   # M2-L5: PDL / pre-wait prefetch switch ($DSV41_PDL, default off)

HAVE_TRITON = True         # (tests/triton_shim.load_module_under_shim asserts it)
GATHER_BLOCK = 8192        # bytes per program of the _u8 kernels (bench/moe_static_spike.py:126)
VEC_BLOCK_BYTES = int(os.environ.get("DSV41_GATHER_BLOCK_BYTES", "8192"))  # bytes per program, vec kernels
VEC_WARPS = int(os.environ.get("DSV41_GATHER_WARPS", "4"))
VEC_ALIGN = 16
GATHER_IMPLS = ("vec", "u8")
GATHER_IMPL_ENV = "DSV41_GATHER_IMPL"  # = runtime.static_step.GATHER_IMPL_ENV


_VEC = {"error": None}  # first failure of a default-path vec launch (compile error): the process then uses u8


def gather_impl() -> str:
    v = os.environ.get(GATHER_IMPL_ENV, "vec")
    if v not in GATHER_IMPLS:
        raise ValueError(f"{GATHER_IMPL_ENV}={v!r} not in {GATHER_IMPLS}")
    return v


def describe() -> dict:
    """Which gather the default path uses (for reports): impl, vec block/warps, and the vec error if it fell back."""
    return {"impl": "u8" if _VEC["error"] else gather_impl(), "vec_block_bytes": VEC_BLOCK_BYTES,
            "vec_warps": VEC_WARPS, "vec_error": _VEC["error"]}


def _vec_failed(e: Exception) -> None:
    _VEC["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    print(f"[static_kernels] 16-B vector gather failed ({_VEC['error']}); using the byte gather (u8) from now on",
          file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ T8 / T9b-2 (u8)
# M2-L5 (runtime/pdl.py): every kernel of this file that the step launches takes PDL (0 off, 1 gdc_wait +
# gdc_launch_dependents at entry, 2 wait at entry + trigger at the end) -- the wait is the first device operation, before
# any load -- and the fixed-order fp32 dots also PDL_PF (pre-wait L2 prefetch of the program's weight rows).
@triton.jit
def _pdl_pf_l2(ptr):
    """prefetch.global.L2 of the 128 B line holding each address of the block: an L2 hint (nothing is read into a
    register, no ordering, no fault for an in-bounds address)."""
    return tl.inline_asm_elementwise("{ prefetch.global.L2 [$1]; mov.u32 $0, 0; }", "=r,l",
                                     [ptr.to(tl.int64)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _pdl_prefetch(base, r0, nrows, stride, c0, clen, NR: tl.constexpr, EPL: tl.constexpr, NL: tl.constexpr,
                  CH: tl.constexpr):
    """The 128 B lines of rows [r0, r0 + NR) (clamped to nrows - 1) x elements [c0, c0 + clen) of `base` (row
    stride `stride` elements, EPL elements per line): NL line points per row (runtime.pdl.lines; the last point
    clamped to the run's last element), CH points per row per chunk (runtime.pdl.chunk)."""
    pf_r = tl.minimum(r0 + tl.arange(0, NR), nrows - 1).to(tl.int64)
    for pf_c in tl.static_range(NL // CH):
        pf_j = pf_c * CH + tl.arange(0, CH)
        pf_o = c0 + tl.minimum(pf_j * EPL, clen - 1)
        _pdl_pf_l2(base + pf_r[:, None] * stride + pf_o[None, :])


@triton.jit
def t8_gather_rows_u8(ptab, ids, dst, row_bytes: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = 0):
    """The T8 kernel (bench/moe_static_spike.py:117-130), unchanged: one byte per lane."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    j = tl.program_id(0)
    blk = tl.program_id(1)
    e = tl.load(ids + j)
    src = tl.load(ptab + e).to(tl.pointer_type(tl.uint8))
    off = blk * BLOCK + tl.arange(0, BLOCK)
    m = off < row_bytes
    v = tl.load(src + off, mask=m)
    tl.store(dst + j * row_bytes + off, v, mask=m)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def t9b2_gather_tables_u8(ptab, ids, dtab, ids_out, E, row_w, row_s, NBW, BLOCK: tl.constexpr,
                          SCALE_F32: tl.constexpr, IDS: tl.constexpr, PDL: tl.constexpr = 0):
    """The T9b-2 kernel, unchanged (contract: t9b2_gather_tables)."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    j = tl.program_id(0)
    blk = tl.program_id(1)
    p = tl.program_id(2)
    e = tl.load(ids + j)
    # Triton unifies same-named variables across if/else arms (types must match): each arm uses its own names.
    if blk < NBW:
        src_w = tl.load(ptab + p * E + e).to(tl.pointer_type(tl.uint8))
        dst_w = tl.load(dtab + p).to(tl.pointer_type(tl.uint8))
        off_w = blk * BLOCK + tl.arange(0, BLOCK)
        m_w = off_w < row_w
        tl.store(dst_w + j * row_w + off_w, tl.load(src_w + off_w, mask=m_w), mask=m_w)
    else:
        src_s = tl.load(ptab + (3 + p) * E + e).to(tl.pointer_type(tl.uint8))
        off_s = (blk - NBW) * BLOCK + tl.arange(0, BLOCK)
        m_s = off_s < row_s
        c = tl.load(src_s + off_s, mask=m_s, other=127)
        if SCALE_F32:
            ci = c.to(tl.int32)
            bits = tl.where(ci == 0, 0x400000, ci << 23)
            bits = tl.where(ci == 255, 0x7FC00000, bits)
            dst_f = tl.load(dtab + 3 + p).to(tl.pointer_type(tl.float32))
            tl.store(dst_f + j * row_s + off_s, bits.to(tl.float32, bitcast=True), mask=m_s)
        else:
            dst_u = tl.load(dtab + 3 + p).to(tl.pointer_type(tl.uint8))
            tl.store(dst_u + j * row_s + off_s, c, mask=m_s)
    if IDS:
        if blk == 0:
            if p == 0:
                tl.store(ids_out + j, e)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


# ------------------------------------------------------------------------------------------------ T9b-3a (vec)
@triton.jit
def t8_gather_rows(ptab, ids, dst, ROW_WORDS: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr,
                   PDL: tl.constexpr = 0):
    """Program (j, blk): int64 words [blk*BLOCK, (blk+1)*BLOCK) of row ptab[ids[j]] -> the same words of slot j of
    dst (int64 view). Row address 16-B aligned (host-checked) and declared so -> 16-B vector loads/stores."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    j = tl.program_id(0)
    blk = tl.program_id(1)
    e = tl.load(ids + j)
    src = tl.multiple_of(tl.load(ptab + e).to(tl.pointer_type(tl.int64)), 16)
    off = blk * BLOCK + tl.arange(0, BLOCK)
    d = dst + j * ROW_WORDS
    if EVEN:
        tl.store(d + off, tl.load(src + off))
    else:
        m = off < ROW_WORDS
        tl.store(d + off, tl.load(src + off, mask=m), mask=m)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def t9b2_gather_tables(ptab, ids, dtab, ids_out, E, ROW_W: tl.constexpr, ROW_S: tl.constexpr, NBW: tl.constexpr,
                       BW: tl.constexpr, BS: tl.constexpr, EVEN_W: tl.constexpr, EVEN_S: tl.constexpr,
                       SCALE_F32: tl.constexpr, IDS: tl.constexpr, PDL: tl.constexpr = 0):
    """One launch for a layer's six staging tables. ptab: int64 [6, E] source row addresses (w1, w3, w2 weights, then
    their scales); dtab: int64 [6] destination bases. Program (j, blk, p): blk < NBW copies int64 words
    [blk*BW, ...) of projection p's weight row ids[j] (ROW_W words); blk >= NBW handles its scale row (ROW_S bytes):
    as int64 words, BS per program (SCALE_F32 0), or BS e8m0 codes per program decoded to their exact fp32 values
    (SCALE_F32 1: 0 -> 2^-127, 255 -> NaN, else code << 23 -- runtime/dg_moe.py's convert_scales, which this
    replaces) at destination slot j. IDS: program (j, 0, 0) also writes ids[j] to ids_out[j] (StaticStep.ids_step's
    row). All row addresses 16-B aligned (host-checked) and declared so -> 16-B vectors per lane."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    j = tl.program_id(0)
    blk = tl.program_id(1)
    p = tl.program_id(2)
    e = tl.load(ids + j)
    if blk < NBW:
        w_src = tl.multiple_of(tl.load(ptab + p * E + e).to(tl.pointer_type(tl.int64)), 16)
        w_dst = tl.multiple_of(tl.load(dtab + p).to(tl.pointer_type(tl.int64)), 16)
        w_off = blk * BW + tl.arange(0, BW)
        w_d = w_dst + j * ROW_W
        if EVEN_W:
            tl.store(w_d + w_off, tl.load(w_src + w_off))
        else:
            w_m = w_off < ROW_W
            tl.store(w_d + w_off, tl.load(w_src + w_off, mask=w_m), mask=w_m)
    else:
        if SCALE_F32:
            f_src = tl.multiple_of(tl.load(ptab + (3 + p) * E + e).to(tl.pointer_type(tl.uint8)), 16)
            f_dst = tl.multiple_of(tl.load(dtab + 3 + p).to(tl.pointer_type(tl.float32)), 16)
            f_off = (blk - NBW) * BS + tl.arange(0, BS)
            if EVEN_S:
                f_c = tl.load(f_src + f_off)
            else:
                f_c = tl.load(f_src + f_off, mask=f_off < ROW_S, other=127)
            f_ci = f_c.to(tl.int32)
            f_bits = tl.where(f_ci == 0, 0x400000, f_ci << 23)
            f_bits = tl.where(f_ci == 255, 0x7FC00000, f_bits)
            f_v = f_bits.to(tl.float32, bitcast=True)
            f_d = f_dst + j * ROW_S
            if EVEN_S:
                tl.store(f_d + f_off, f_v)
            else:
                tl.store(f_d + f_off, f_v, mask=f_off < ROW_S)
        else:
            u_src = tl.multiple_of(tl.load(ptab + (3 + p) * E + e).to(tl.pointer_type(tl.int64)), 16)
            u_dst = tl.multiple_of(tl.load(dtab + 3 + p).to(tl.pointer_type(tl.int64)), 16)
            u_off = (blk - NBW) * BS + tl.arange(0, BS)
            u_d = u_dst + j * (ROW_S // 8)
            if EVEN_S:
                tl.store(u_d + u_off, tl.load(u_src + u_off))
            else:
                u_m = u_off < ROW_S // 8
                tl.store(u_d + u_off, tl.load(u_src + u_off, mask=u_m), mask=u_m)
    if IDS:
        if blk == 0:
            if p == 0:
                tl.store(ids_out + j, e)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


# ------------------------------------------------------------------------------------------------ wrappers
def _block_bytes(block_bytes) -> int:
    bb = int(block_bytes or VEC_BLOCK_BYTES)
    if bb < VEC_ALIGN or bb & (bb - 1):
        raise ValueError(f"gather block {bb} B must be a power of two >= {VEC_ALIGN}")
    return bb


def gather_rows(ptab: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor, row_bytes: int, impl: Optional[str] = None,
                block_bytes: Optional[int] = None, num_warps: Optional[int] = None) -> None:
    """dst: contiguous [k, ...] staging, k = ids.numel(), each slot `row_bytes` bytes. ptab: int64 [n_experts] row
    addresses (16-B aligned for "vec": checked at StaticStep build); ids: int64 [k] on the device. impl: "vec"
    (default, $DSV41_GATHER_IMPL) or "u8" (the T8 kernel); "vec" uses "u8" when row_bytes or dst is not 16-B aligned.
    block_bytes / num_warps: vec tuning (bench/gather_bench.py; defaults $DSV41_GATHER_BLOCK_BYTES / _WARPS).
    Default path only (impl None): a vec launch that raises (compile error) is reported once and the process falls
    back to u8; an explicit impl="vec" raises."""
    k = ids.numel()
    assert dst.is_contiguous() and dst.numel() * dst.element_size() == k * row_bytes, (dst.shape, k, row_bytes)
    explicit = impl is not None
    impl = impl or ("u8" if _VEC["error"] else gather_impl())
    if impl == "vec" and row_bytes % VEC_ALIGN == 0 and dst.data_ptr() % VEC_ALIGN == 0:
        bw = _block_bytes(block_bytes) // 8
        words = row_bytes // 8
        try:
            t8_gather_rows[(k, triton.cdiv(words, bw))](ptab, ids, dst.view(torch.int64), words, bw,
                                                        words % bw == 0, num_warps=int(num_warps or VEC_WARPS),
                                                        **_pdl.kw("t8_gather_rows"))
            return
        except Exception as e:
            if explicit:
                raise
            _vec_failed(e)
    t8_gather_rows_u8[(k, triton.cdiv(row_bytes, GATHER_BLOCK))](ptab, ids, dst.view(torch.uint8), row_bytes,
                                                                 GATHER_BLOCK, **_pdl.kw("t8_gather_rows_u8"))


def gather_tables(ptab6: torch.Tensor, ids: torch.Tensor, dtab: torch.Tensor, dsts, row_w: int, row_s: int,
                  scale_f32: bool, ids_out=None, impl: Optional[str] = None, block_bytes: Optional[int] = None,
                  num_warps: Optional[int] = None) -> None:
    """T9b-2: the six gather_rows calls of a layer (+ dg-masked's three scale conversions, + the ids_step copy) as ONE
    launch. ptab6 int64 [6, E] (w1/w3/w2 weight rows, then w1/w3/w2 scale rows); dtab int64 [6] = the data_ptr()s of
    `dsts` (staging weights x3, then staging scales x3 or dg's fp32 sf32 x3), built once at construction (no H2D copy
    under capture); `dsts` is read here only for the host-side alignment check (the CPU emulation in the tests copies
    through it). ids_out: int64 [k] or None. impl / block_bytes / num_warps as gather_rows."""
    k = ids.numel()
    explicit = impl is not None
    impl = impl or ("u8" if _VEC["error"] else gather_impl())
    io = ids_out if ids_out is not None else ids
    if impl == "vec" and row_w % VEC_ALIGN == 0 and row_s % VEC_ALIGN == 0 and all(
            t.data_ptr() % VEC_ALIGN == 0 for t in dsts):
        bb = _block_bytes(block_bytes)
        bw = bb // 8
        ww = row_w // 8
        nbw = triton.cdiv(ww, bw)
        if scale_f32:  # BS = e8m0 codes per program
            bs, nbs, even_s = bb, triton.cdiv(row_s, bb), row_s % bb == 0
        else:          # BS = int64 words per program
            bs, nbs, even_s = bw, triton.cdiv(row_s // 8, bw), (row_s // 8) % bw == 0
        try:
            t9b2_gather_tables[(k, nbw + nbs, 3)](ptab6, ids, dtab, io, ptab6.size(1), ww, row_s, nbw, bw, bs,
                                                 ww % bw == 0, even_s, bool(scale_f32), ids_out is not None,
                                                 num_warps=int(num_warps or VEC_WARPS),
                                                 **_pdl.kw("t9b2_gather_tables"))
            return
        except Exception as e:
            if explicit:
                raise
            _vec_failed(e)
    nbw = triton.cdiv(row_w, GATHER_BLOCK)
    nbs = triton.cdiv(row_s, GATHER_BLOCK)
    t9b2_gather_tables_u8[(k, nbw + nbs, 3)](ptab6, ids, dtab, io, ptab6.size(1), row_w, row_s, nbw, GATHER_BLOCK,
                                            bool(scale_f32), ids_out is not None,
                                            **_pdl.kw("t9b2_gather_tables_u8"))


@triton.jit
def t8_step_marker(ctr, PDL: tl.constexpr = 0):
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(ctr, tl.load(ctr) + 1)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


# ------------------------------------------------------------------------------------------------ T9a-2 (dg-inplace)
STAGE_NPROG = int(os.environ.get("DSV41_STAGE_NPROG", "48"))  # programs per (slot, projection); 720 8-KiB blocks / 48
STAGE_BN = 64                                                   # scale rows per tile of the int32 transpose


@triton.jit
def _t9a2_pack_scales(ptab, dtab, e, E, p, grp, q, N13, C13, N2, C2, BN: tl.constexpr, BC: tl.constexpr,
                      NPROG: tl.constexpr):
    """Scale row of expert e, projection p (e8m0 [N, C*4] bytes = int32 [N, C] words, 4 consecutive K-scales per
    word, little-endian = DeepGEMM's pack_ue8m0_to_int) -> group `grp` of the MN-major packed SF tensor at dtab[3+p]
    (int32 storage [G][C][N]): an int32 transpose, tiles of BN rows, tile t handled by program q = t % NPROG."""
    is2 = p == 2
    N = tl.where(is2, N2, N13)
    C = tl.where(is2, C2, C13)
    s_src = tl.load(ptab + (3 + p) * E + e).to(tl.pointer_type(tl.int32))
    s_dst = tl.load(dtab + 3 + p).to(tl.pointer_type(tl.int32)) + grp * (C * N)
    nb = (N + BN - 1) // BN
    c = tl.arange(0, BC)
    cm = c < C
    for it in range(0, (nb + NPROG - 1) // NPROG):
        n = (q + it * NPROG) * BN + tl.arange(0, BN)
        m = (n < N)[:, None] & cm[None, :]
        v = tl.load(s_src + n[:, None] * C + c[None, :], mask=m)
        tl.store(s_dst + c[None, :] * N + n[:, None], v, mask=m)


@triton.jit
def t9a2_stage_inplace(ptab, ids, gtab, dtab, mi, ids_out, sidx, ovf, E, H, S, ALIGN, N13, C13, N2, C2,
                       ROW_W: tl.constexpr, BW: tl.constexpr, NBW: tl.constexpr, BN: tl.constexpr,
                       BC: tl.constexpr, NPROG: tl.constexpr, EVEN_W: tl.constexpr, ALL_SCALES: tl.constexpr,
                       IDS: tl.constexpr, SIDX: tl.constexpr, PDL: tl.constexpr = 0):
    """T9a-2: one launch per layer before its three contiguous grouped GEMMs (runtime/dg_inplace.py). Program
    (j, q, p): e = ids[j] (sorted routed id), g = gtab[e] (its hot pool slot, or -1 = cold). Spare index c = j
    (SIDX 0, the 1-row step: k slots, S = k spares) or c = sidx[j] (SIDX 1, M1-T10-FIX2's 6-row step: the compact
    cold index, -1 = skip this slot). group = g if hot else H + c. (j, 0, 0) writes mi[j*ALIGN] = group (-1 for a
    skipped or overflowing cold slot; every other mi row stays -1), IDS: ids_out[j] = e, and SIDX: a cold slot with
    c >= S (more cold experts than spare slots) adds 1 to the step's overflow counter ovf[0] (nothing of it is
    staged: its GEMM rows are wrong, never out of bounds). Staged cold slots only (0 <= c < S): program q copies
    8-KiB weight blocks q, q+NPROG, ... of row ptab[p, e] (an HBM or UVA address) into slot `group` of the layer's
    [H+S', N, K/2] weight pool at dtab[p] (int64 words, 16-B vectors as t8_gather_rows). Scales: hot + staged cold
    (ALL_SCALES, the shared packed SF of sf_store "step") or staged cold only ("resident"): _t9a2_pack_scales. Row
    addresses 16-B aligned (host-checked)."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    j = tl.program_id(0)
    q = tl.program_id(1)
    p = tl.program_id(2)
    e = tl.load(ids + j)
    g = tl.load(gtab + e).to(tl.int64)
    cold = g < 0
    hot = g >= 0
    if SIDX:
        c = tl.load(sidx + j).to(tl.int64)
    else:
        c = g * 0 + j                            # int64 (g * 0 + ...: promotes j without a cast)
    fits = (c >= 0) & (c < S)
    stage = cold & fits
    grp = tl.where(cold, c + H, g)
    head = tl.where(hot | stage, grp, g * 0 - 1)
    if q == 0:
        if p == 0:
            tl.store(mi + j * ALIGN, head.to(tl.int32))
            if IDS:
                tl.store(ids_out + j, e)
            if SIDX:
                if cold & (c >= S):
                    tl.atomic_add(ovf, 1)
    if stage:
        w_src = tl.multiple_of(tl.load(ptab + p * E + e).to(tl.pointer_type(tl.int64)), 16)
        w_dst = tl.multiple_of(tl.load(dtab + p).to(tl.pointer_type(tl.int64)), 16)
        w_row = w_dst + grp * ROW_W
        if EVEN_W:
            for it in range(0, NBW // NPROG):
                we_off = (q + it * NPROG) * BW + tl.arange(0, BW)
                tl.store(w_row + we_off, tl.load(w_src + we_off))
        else:
            for it2 in range(0, (NBW + NPROG - 1) // NPROG):
                wm_off = (q + it2 * NPROG) * BW + tl.arange(0, BW)
                wm_m = wm_off < ROW_W
                tl.store(w_row + wm_off, tl.load(w_src + wm_off, mask=wm_m), mask=wm_m)
    if ALL_SCALES:
        if hot | stage:
            _t9a2_pack_scales(ptab, dtab, e, E, p, grp, q, N13, C13, N2, C2, BN, BC, NPROG)
    else:
        if stage:
            _t9a2_pack_scales(ptab, dtab, e, E, p, grp, q, N13, C13, N2, C2, BN, BC, NPROG)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


def stage_inplace(ptab6: torch.Tensor, ids: torch.Tensor, gtab: torch.Tensor, dtab: torch.Tensor, dsts, mi,
                  H: int, align: int, row_w: int, sf_dims, all_scales: bool, ids_out=None,
                  block_bytes: Optional[int] = None, nprog: Optional[int] = None,
                  num_warps: Optional[int] = None, *, sidx: Optional[torch.Tensor] = None,
                  spare: Optional[int] = None, ovf: Optional[torch.Tensor] = None) -> None:
    """T9a-2 (runtime/dg_inplace.py, contract: dg_inplace.stage_inplace_emulated): ONE launch per layer.
    ptab6 int64 [6, E] source rows (w1/w3/w2 weights, then their e8m0 scales; HBM pool slot or UVA host address);
    gtab int32 [E] hot slot or -1; dtab int64 [6] = data_ptr()s of `dsts` = the layer's three [H+S, N, K/2] weight
    pools, then the three packed SF tensors (int32 [G, N, C] MN-major); mi int32 [k*align] (-1 except the block
    heads this writes); sf_dims ((N13, C13), (N2, C2)) = (rows, int32 words per row) of the w1/w3 and w2 scales.
    `dsts` is read only for the host-side alignment check.
    M1-T10-FIX2 (the 6-row step): sidx int64 [k] = each slot's compact cold index (-1: skip the slot), spare = the
    spare slots a cold expert may use (c < spare), ovf int32 [1] = the step's overflow counter (one atomic add per
    cold slot with c >= spare). All three or none (None: c = j, the 1-row contract; spare defaults to k there)."""
    k = ids.numel()
    (n13, c13), (n2, c2) = sf_dims
    if row_w % VEC_ALIGN or any(t.data_ptr() % VEC_ALIGN for t in dsts):
        raise ValueError(f"stage_inplace: weight row {row_w} B or a destination not {VEC_ALIGN}-B aligned")
    use_sidx = sidx is not None
    if use_sidx != (ovf is not None) or (use_sidx and spare is None):
        raise ValueError("stage_inplace: sidx, spare and ovf go together")
    S = int(spare) if spare is not None else k
    bw = _block_bytes(block_bytes) // 8
    ww = row_w // 8
    nbw = triton.cdiv(ww, bw)
    npg = int(nprog or STAGE_NPROG)
    bc = triton.next_power_of_2(max(int(c13), int(c2)))
    even = ww % bw == 0 and nbw % npg == 0
    t9a2_stage_inplace[(k, npg, 3)](ptab6, ids, gtab, dtab, mi, ids_out if ids_out is not None else ids,
                                    sidx if use_sidx else ids, ovf if use_sidx else mi,
                                    ptab6.size(1), int(H), S, int(align), int(n13), int(c13), int(n2), int(c2),
                                    ww, bw, nbw, STAGE_BN, bc, npg, even, bool(all_scales), ids_out is not None,
                                    use_sidx, num_warps=int(num_warps or VEC_WARPS),
                                    **_pdl.kw("t9a2_stage_inplace"))


def step_marker(ctr: torch.Tensor) -> None:
    """One program, one warp: `ctr[0] += 1`. Fixed kernel name `t8_step_marker` for nsys."""
    t8_step_marker[(1,)](ctr, num_warps=1, **_pdl.kw("t8_step_marker"))


# ------------------------------------------------------------------------------------------------ M1-T10-MOE (rows)
@triton.jit
def m1_stage_rows(ptab, uniq, gtab, sidx, wl, dtab, E, H, N13, C13, N2, C2,
                  ROW_W: tl.constexpr, BW: tl.constexpr, NBW: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
                  NPROG: tl.constexpr, EVEN_W: tl.constexpr, PDL: tl.constexpr = 0):
    """M1-T10-MOE: the R-row dg-inplace staging launch, driven by the dedup's work list. Program (j, q, p): slot
    u = wl[j] (-1: nothing to do -- the program exits at once), expert e = uniq[u], g = gtab[e]. A cold slot (g < 0;
    the dedup lists only staged ones, compact index c = sidx[u] < spare budget) copies 8-KiB weight blocks q,
    q+NPROG, ... of row ptab[p, e] (HBM or UVA) into pool slot H + c (int64 words, 16-B vectors), then packs its
    scales into group H + c of the packed SF; a hot slot (listed only under sf_store "step") packs its scales into
    group g. So the scale work is once per unique expert (<= U slots, not R*k blocks) and, with the hot SF
    pre-packed (sf_store "pool" / "resident"), only cold experts are touched at all. m_indices, the overflow count
    and the stage ids are the dedup's (moe_rows.m1_dedup_warp). Contract: dg_inplace.stage_rows_emulated."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    j = tl.program_id(0)
    q = tl.program_id(1)
    p = tl.program_id(2)
    u = tl.load(wl + j)
    if u >= 0:
        e = tl.load(uniq + u)
        g = tl.load(gtab + e).to(tl.int64)
        c = tl.load(sidx + u).to(tl.int64)
        cold = g < 0
        grp = tl.where(cold, c + H, g)
        if cold:
            r_src = tl.multiple_of(tl.load(ptab + p * E + e).to(tl.pointer_type(tl.int64)), 16)
            r_dst = tl.multiple_of(tl.load(dtab + p).to(tl.pointer_type(tl.int64)), 16)
            r_row = r_dst + grp * ROW_W
            if EVEN_W:
                for it in range(0, NBW // NPROG):
                    re_off = (q + it * NPROG) * BW + tl.arange(0, BW)
                    tl.store(r_row + re_off, tl.load(r_src + re_off))
            else:
                for it2 in range(0, (NBW + NPROG - 1) // NPROG):
                    rm_off = (q + it2 * NPROG) * BW + tl.arange(0, BW)
                    rm_m = rm_off < ROW_W
                    tl.store(r_row + rm_off, tl.load(r_src + rm_off, mask=rm_m), mask=rm_m)
        _t9a2_pack_scales(ptab, dtab, e, E, p, grp, q, N13, C13, N2, C2, BN, BC, NPROG)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


def stage_rows(ptab6: torch.Tensor, uniq: torch.Tensor, gtab: torch.Tensor, dtab: torch.Tensor, dsts,
               sidx: torch.Tensor, wl: torch.Tensor, H: int, row_w: int, sf_dims, n_work: int,
               block_bytes: Optional[int] = None, nprog: Optional[int] = None,
               num_warps: Optional[int] = None) -> None:
    """M1-T10-MOE: ONE launch per layer of the R-row dg-inplace step (m1_stage_rows) over the first `n_work` entries
    of the dedup's work list `wl` (the spare budget S when only cold experts are listed, R*k with sf_store "step").
    ptab6 / gtab / dtab / dsts / sf_dims as stage_inplace; uniq / sidx / wl int64 [R*k] from the dedup."""
    (n13, c13), (n2, c2) = sf_dims
    if row_w % VEC_ALIGN or any(t.data_ptr() % VEC_ALIGN for t in dsts):
        raise ValueError(f"stage_rows: weight row {row_w} B or a destination not {VEC_ALIGN}-B aligned")
    nw = int(n_work)
    if not 1 <= nw <= wl.numel():
        raise ValueError(f"stage_rows: n_work {nw} outside [1, {wl.numel()}]")
    bw = _block_bytes(block_bytes) // 8
    ww = row_w // 8
    nbw = triton.cdiv(ww, bw)
    npg = int(nprog or STAGE_NPROG)
    bc = triton.next_power_of_2(max(int(c13), int(c2)))
    even = ww % bw == 0 and nbw % npg == 0
    m1_stage_rows[(nw, npg, 3)](ptab6, uniq, gtab, sidx, wl, dtab, ptab6.size(1), int(H), int(n13), int(c13),
                                int(n2), int(c2), ww, bw, nbw, STAGE_BN, bc, npg, even,
                                num_warps=int(num_warps or VEC_WARPS), **_pdl.kw("m1_stage_rows"))


# ------------------------------------------------------------------------------------------------ T9b-5 (fp32 fixed)
# docs/HANDOFF-T9B5-FP32-FIXED.md. out[r, n] = sum_k x[r, k] * w[n, k] in fp32, in ONE order per output that does not
# depend on R, on the launch configuration or on the device's choice of algorithm (cuBLAS picks split-K / gemv vs
# sgemm by M, so the 6-row step's head and mixes differ from the 1-row step's, docs/HEADLINE-20261107.md). The order
# (fused_glue_r2.fixed_dot_torch is its CPU definition; tests/test_fp32_fixed_cpu.py runs this body under the shim):
#   KB lanes (a power of two dividing K); lane j accumulates k = j, j + KB, j + 2 KB, ... sequentially from +0.0 with
#   one IEEE fp32 FMA per term (acc = fma(x, w, acc)); then an adjacent-pair tree over the lanes,
#   s[i] = s[2i] + s[2i+1], log2(KB) levels; then (HAS_SC) one fp32 multiply by sc[r] (the mixes' * rsqrt).
# Launched with enable_fp_fusion=False (no other contraction exists, but nothing is left to the compiler). Nothing
# else is a free parameter of the arithmetic: BN (outputs per program), RB (rows per program: the rows share every
# weight tile, so the weight is read once per step at any R <= 8), U (K tiles unrolled per loop trip: more loads in
# flight, same adds in the same order), num_warps and num_stages change speed only (the GPU test asserts every
# config is bit-identical). Rows r >= R and outputs n >= N are masked: they never feed a stored output.
T9B5_KB_MAX = 4096
T9B5_RB_MAX = 8
# per site and rows-per-program RB: (BN, U, num_warps, num_stages). Speed-only knobs (bench/t9b5_bench.py sweeps them);
# override: DSV41_T9B5_CFG="head:8=16,2,8,1;mix:1=1,8,4,1".
T9B5_CFG = {"head": {1: (32, 4, 4, 1), 2: (32, 2, 4, 1), 4: (16, 4, 4, 1), 8: (16, 4, 8, 1)},
            "mix": {1: (1, 20, 16, 1)}}   # mixes: 40 K tiles in 2 trips (latency: few programs at 1 row)
# M2-T17 (docs/HANDOFF-M2-T17-BF16-HEAD.md): the head weight kept in its checkpoint dtype, bf16 in HBM -- the SAME kernel
# (t9b5_head_fp32: the tile is loaded as bf16 and widened to fp32 in registers by `.to(tl.float32)`, which is exact, so
# every product and every sum is the fp32-stored head's, bit for bit), half the bytes. A bf16 tile is half the bytes of an
# fp32 tile, so U (K tiles per loop trip) is doubled to keep the bytes in flight per program the same as the fp32 config's.
# UNTUNED starting point: bench/head_bench.py sweeps (BN, U, num_warps, num_stages) on the box and prints the best as a
# DSV41_T9B5_CFG string (site key "head_bf16:<RB>"); every config is bit-identical (speed-only knobs).
T9B5_CFG["head_bf16"] = {1: (32, 8, 4, 1), 2: (32, 4, 4, 1), 4: (16, 8, 4, 1), 8: (16, 8, 8, 1)}
T9B5_CFG_ENV = "DSV41_T9B5_CFG"


@triton.jit
def _t9b5_pairs(s, M: tl.constexpr, H: tl.constexpr):
    """[M, 2H] -> [M, H]: s[:, 2i] + s[:, 2i + 1] (one IEEE fp32 add each)."""
    pa, pb = tl.split(tl.reshape(s, [M, H, 2]))
    return pa + pb


@triton.jit
def _t9b5_tree(acc, RB: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    """acc [RB, BN, KB] -> [RB, BN]: the adjacent-pair tree over the KB lanes (KB a power of two <= 4096)."""
    s = tl.reshape(acc, [RB * BN, KB])
    if KB >= 4096:
        s = _t9b5_pairs(s, RB * BN, 2048)
    if KB >= 2048:
        s = _t9b5_pairs(s, RB * BN, 1024)
    if KB >= 1024:
        s = _t9b5_pairs(s, RB * BN, 512)
    if KB >= 512:
        s = _t9b5_pairs(s, RB * BN, 256)
    if KB >= 256:
        s = _t9b5_pairs(s, RB * BN, 128)
    if KB >= 128:
        s = _t9b5_pairs(s, RB * BN, 64)
    if KB >= 64:
        s = _t9b5_pairs(s, RB * BN, 32)
    if KB >= 32:
        s = _t9b5_pairs(s, RB * BN, 16)
    if KB >= 16:
        s = _t9b5_pairs(s, RB * BN, 8)
    if KB >= 8:
        s = _t9b5_pairs(s, RB * BN, 4)
    if KB >= 4:
        s = _t9b5_pairs(s, RB * BN, 2)
    if KB >= 2:
        s = _t9b5_pairs(s, RB * BN, 1)
    return tl.reshape(s, [RB, BN])


@triton.jit
def _t9b5_dot(x, w, out, sc, R, N, sxr, swn, sor, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr,
              RB: tl.constexpr, U: tl.constexpr, HAS_SC: tl.constexpr):
    """Program (pn, pr): outputs n in [pn*BN, (pn+1)*BN) of rows r in [pr*RB, (pr+1)*RB). x [R, K] (fp32 or bf16
    values, row stride sxr), w [N, K] (fp32 or bf16, row stride swn), out [R, N] fp32 (row stride sor), K = NK*KB."""
    pn = tl.program_id(0)
    pr = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    r = pr * RB + tl.arange(0, RB)
    j = tl.arange(0, KB)
    mn = n < N
    mr = r < R
    wp = w + n.to(tl.int64)[:, None] * swn + j[None, :]
    xp = x + r.to(tl.int64)[:, None] * sxr + j[None, :]
    wm = mn[:, None] & (j[None, :] >= 0)
    xm = mr[:, None] & (j[None, :] >= 0)
    acc = tl.zeros([RB, BN, KB], dtype=tl.float32)
    for t in range(NK // U):
        for u in tl.static_range(U):
            k0 = (t * U + u) * KB
            wv = tl.load(wp + k0, mask=wm, other=0.0).to(tl.float32)
            xv = tl.load(xp + k0, mask=xm, other=0.0).to(tl.float32)
            acc = tl.fma(tl.broadcast_to(xv[:, None, :], [RB, BN, KB]), tl.broadcast_to(wv[None, :, :], [RB, BN, KB]),
                         acc)
    for t2 in range((NK // U) * U, NK):          # the tiles past the last whole unroll (same order: tile t2 next)
        k2 = t2 * KB
        wv2 = tl.load(wp + k2, mask=wm, other=0.0).to(tl.float32)
        xv2 = tl.load(xp + k2, mask=xm, other=0.0).to(tl.float32)
        acc = tl.fma(tl.broadcast_to(xv2[:, None, :], [RB, BN, KB]), tl.broadcast_to(wv2[None, :, :], [RB, BN, KB]),
                     acc)
    s = _t9b5_tree(acc, RB, BN, KB)
    if HAS_SC:
        s = s * tl.load(sc + r, mask=mr, other=1.0)[:, None]
    tl.store(out + r.to(tl.int64)[:, None] * sor + n[None, :], s, mask=mr[:, None] & mn[None, :])


@triton.jit
def t9b5_head_fp32(x, w, out, sc, R, N, sxr, swn, sor, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr,
                   RB: tl.constexpr, U: tl.constexpr, HAS_SC: tl.constexpr, PDL: tl.constexpr = 0,
                   PDL_PF: tl.constexpr = 0, PF_EPL: tl.constexpr = 32, PF_NL: tl.constexpr = 1,
                   PF_CH: tl.constexpr = 1):
    """The head (model.py:1014): logits [R, V] = x [R, D] . W [V, D]^T (named launch for nsys). M2-L5: PDL_PF
    prefetches the program's BN weight rows, then gdc_wait (runtime/pdl.py)."""
    if PDL_PF:
        _pdl_prefetch(w, tl.program_id(0) * BN, N, swn, 0, NK * KB, BN, PF_EPL, PF_NL, PF_CH)
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    _t9b5_dot(x, w, out, sc, R, N, sxr, swn, sor, NK, KB, BN, RB, U, HAS_SC)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def t9b5_mix_fp32(x, w, out, sc, R, N, sxr, swn, sor, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr,
                  RB: tl.constexpr, U: tl.constexpr, HAS_SC: tl.constexpr, PDL: tl.constexpr = 0,
                  PDL_PF: tl.constexpr = 0, PF_EPL: tl.constexpr = 32, PF_NL: tl.constexpr = 1,
                  PF_CH: tl.constexpr = 1):
    """The mHC mixes (model.py:956): [R, mix_hc] = (x [R, hc*D] . hc_fn^T) * rsqrt[R] (named launch for nsys).
    M2-L5: PDL_PF prefetches the program's BN rows of hc_fn, then gdc_wait (runtime/pdl.py)."""
    if PDL_PF:
        _pdl_prefetch(w, tl.program_id(0) * BN, N, swn, 0, NK * KB, BN, PF_EPL, PF_NL, PF_CH)
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    _t9b5_dot(x, w, out, sc, R, N, sxr, swn, sor, NK, KB, BN, RB, U, HAS_SC)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


T9B5_KERNELS = {"head": t9b5_head_fp32, "mix": t9b5_mix_fp32}
T9B5_PDL_NAMES = {"head": "t9b5_head_fp32", "mix": "t9b5_mix_fp32", "comp": "t9b8_comp_fp32"}   # runtime.pdl.KERNELS


def _t9b5_env_cfg() -> dict:
    out = {}
    for item in (os.environ.get(T9B5_CFG_ENV) or "").split(";"):
        item = item.strip()
        if not item:
            continue
        key, val = item.split("=")
        site, rb = key.split(":")
        out[(site.strip(), int(rb))] = tuple(int(v) for v in val.split(","))
    return out


def t9b5_rows_per_program(site: str, R: int) -> int:
    """RB: the mixes run one row per program (their 2 MB weight is L2-resident across the R programs that share
    it); the head (fp32- or bf16-stored: site "head" / "head_bf16") puts every row (<= 8) in one program so its
    weight is read once per step."""
    if site == "mix":
        return 1
    return min(T9B5_RB_MAX, triton.next_power_of_2(max(1, int(R))))


def t9b5_config(site: str, R: int, wdtype: Optional[torch.dtype] = None) -> tuple:
    """(RB, BN, U, num_warps, num_stages) for a call of `site` at R rows. A bf16-stored head weight (wdtype
    torch.bfloat16 at site "head", or site "head_bf16" itself) takes the "head_bf16" table / env key."""
    if site == "head" and wdtype == torch.bfloat16:
        site = "head_bf16"
    rb = t9b5_rows_per_program(site, R)
    env = _t9b5_env_cfg()
    bn, u, nw, ns = env.get((site, rb)) or T9B5_CFG[site].get(rb) or T9B5_CFG[site][max(T9B5_CFG[site])]
    return rb, bn, u, nw, ns


def fixed_dot(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, *, kb: int, site: str,
              scale: Optional[torch.Tensor] = None, cfg: Optional[tuple] = None) -> None:
    """T9b-5: out [R, N] fp32 <- x [R, K] . w [N, K]^T (* scale [R]) in the fixed order of KB = kb lanes (the
    contract above; fused_glue_r2.fixed_dot_torch on CPU). site "head" / "mix" picks the named kernel and the
    default launch config; cfg = (RB, BN, U, num_warps, num_stages) overrides it (speed only). w may be fp32 or
    bf16 (M2-T17: widened in registers, exact; the head's bf16 weight takes the "head_bf16" launch table)."""
    if site not in T9B5_KERNELS:
        raise ValueError(f"fixed_dot: site {site!r} not in {tuple(T9B5_KERNELS)}")
    if w.dtype not in (torch.float32, torch.bfloat16) or x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"fixed_dot: x {x.dtype} / w {w.dtype} must be fp32 or bf16 (the kernel widens bf16 exactly)")
    if x.dim() != 2 or w.dim() != 2 or out.dim() != 2:
        raise ValueError(f"fixed_dot: x {tuple(x.shape)}, w {tuple(w.shape)}, out {tuple(out.shape)} must be 2-D")
    R, K = x.shape
    N = w.size(0)
    if w.size(1) != K or tuple(out.shape) != (R, N) or out.dtype != torch.float32:
        raise ValueError(f"fixed_dot: x [{R}, {K}], w {tuple(w.shape)}, out {tuple(out.shape)} {out.dtype}")
    if x.stride(1) != 1 or w.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("fixed_dot: x / w / out rows must be contiguous")
    if kb < 1 or kb & (kb - 1) or kb > T9B5_KB_MAX or K % kb:
        raise ValueError(f"fixed_dot: kb {kb} must be a power of two <= {T9B5_KB_MAX} dividing K = {K}")
    if scale is not None and (scale.numel() != R or scale.dtype != torch.float32 or not scale.is_contiguous()):
        raise ValueError(f"fixed_dot: scale must be fp32 contiguous [R = {R}]")
    if R == 0:
        return
    rb, bn, u, nw, ns = cfg if cfg is not None else t9b5_config(site, R, w.dtype)
    nk = K // kb
    u = max(1, min(int(u), nk))
    T9B5_KERNELS[site][(triton.cdiv(N, bn), triton.cdiv(R, rb))](
        x, w, out, scale if scale is not None else out, R, N, x.stride(0), w.stride(0), out.stride(0),
        nk, kb, bn, rb, u, scale is not None, num_warps=nw, num_stages=ns, enable_fp_fusion=False,
        **_pdl.kw(T9B5_PDL_NAMES[site], pf=lambda: _pdl.pf_rows(K, w.element_size(), bn)))


# ------------------------------------------------------------------------------------------------ T9b-8 (compressor)
# docs/LEDGER.md 2026-11-07 15:05 (T12-DIAG, docs/rowdiv-shell-20261107.log): the ratio-2 attention compressor's wkv /
# wgate (model.py:463-465: F.linear(x.float(), W), W declared fp32 at ratio > 1 by model.py:444-448 and widened from the
# checkpoint's bf16 by runtime/loader.py, so every product is exact in fp32 and only the sum order is a choice) were the
# last cuBLAS fp32 site of the 6-row step that is not row-invariant (85 % of elements differ 6 rows vs 1). Under
# `--fp32-impl fixed` they run THE T9b-5 order (_t9b5_dot above; KB = fused_glue_r2.FIXED_KB["comp"]) under their own
# launch name, site "comp" of fixed_dot. Real shapes: [R, 5120] x [512, 5120] fp32 (10.5 MB per weight, 3 ratio-2
# layers x 2 weights per step). N is small, so BN is small (128-512 programs on one weight) and every row <= 8 shares a
# program (RB = next_power_of_2(R): the weight is read once per launch at any R). (BN, U, num_warps, num_stages) are
# speed-only (UNTUNED starting point; override "comp:<RB>=BN,U,nw,ns" in $DSV41_T9B5_CFG).
@triton.jit
def t9b8_comp_fp32(x, w, out, sc, R, N, sxr, swn, sor, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr,
                   RB: tl.constexpr, U: tl.constexpr, HAS_SC: tl.constexpr, PDL: tl.constexpr = 0,
                   PDL_PF: tl.constexpr = 0, PF_EPL: tl.constexpr = 32, PF_NL: tl.constexpr = 1,
                   PF_CH: tl.constexpr = 1):
    """The ratio-2 compressor's wkv / wgate (model.py:464-465): [R, head_dim] = x [R, D] . W [head_dim, D]^T (named
    launch for nsys). M2-L5: PDL_PF prefetches the program's BN weight rows, then gdc_wait (runtime/pdl.py)."""
    if PDL_PF:
        _pdl_prefetch(w, tl.program_id(0) * BN, N, swn, 0, NK * KB, BN, PF_EPL, PF_NL, PF_CH)
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    _t9b5_dot(x, w, out, sc, R, N, sxr, swn, sor, NK, KB, BN, RB, U, HAS_SC)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


T9B5_KERNELS["comp"] = t9b8_comp_fp32
# RB 1: 512 one-output programs (x is one 20 KB row); RB >= 2: 4 outputs per program so the R rows of x (L2 reads,
# R x 20 KB per program) are shared by 4 weight rows, 128 programs, num_stages 2 to keep the next trip's loads in flight.
T9B5_CFG["comp"] = {1: (1, 16, 4, 1), 2: (4, 8, 4, 2), 4: (4, 8, 4, 2), 8: (4, 8, 4, 2)}


# ------------------------------------------------------------------------------------------------ M2-T15 (L2)
# docs/HANDOFF-M2-T15-ROUTER-PLAN.md, docs/M2-FUSED-CHAIN-DESIGN.md section 2.2 row L2: the router (Gate.forward,
# model.py:811-829) and, at R rows, the MoE plan (runtime/moe_rows.m1_dedup_warp's tensors) in ONE launch per layer,
# `m2_router_plan`. Host side, CPU mirrors, the tie rule's derivation and the switch: runtime/router_plan.py.
#
#   GEMV (constexpr; `--router-impl plan-gemv`): program p computes the gate scores of experts [p*BN, (p+1)*BN) for
#     all R rows with the T9b-5 order (_t9b5_dot: KB lanes, sequential fp32 FMA from +0.0, adjacent-pair tree; the
#     bf16 weight is widened in registers, exactly) into `raw`; every thread fences (fence.acq_rel.gpu), the CTA
#     syncs, one acq_rel atomic takes a ticket; the LAST program to arrive runs the tail and resets the ticket to 0
#     (graph replays start from 0). Without GEMV (`--router-impl plan`): ONE program; `raw` holds the scores of the
#     gate mm that stays a torch op (cuBLAS, exactly today's).
#   tail (one CTA): SOFTPLUS (constexpr): F.softplus's CUDA formula `x > 20 ? x : log1p(exp(x))` with libdevice
#     log1p / exp (beta = 1: x * 1 and / 1 are exact) -- else `raw` already holds torch's F.softplus output; sqrt_rn;
#     + bias (model.py:819, 824); then torch.topk's CUDA result REPRODUCED (router_plan.py "the tie rule"):
#       selection = gatherTopK: every element whose radix key (TopKTypeConfig<float>::convert) is above the k-th
#         largest key, then the lowest-index elements equal to it (\"the first seen set\") -- here k rounds of a max
#         over (radix key << 16 | 65535 - index), the exact same set;
#       order = SmallBitonicSort (sortKeyValueInplace, sort size 32, GTOp<float, handleNaN=true>, bitonicSwap's
#         `(comp(kA, kB) && validA) || !validB`) applied to gatherTopK's output layout (the above-k-th elements in
#         index order, then the tied ones in index order): with no tie among the k selected values the network's
#         output is the descending order (rank = #greater); with a tie the 32-lane network itself runs (15 stages,
#         through the `net` scratch row: store, barrier, load the partner lane).
#     then Gate.forward's tail in torch.topk's order = t9b2r2_router_post's arithmetic for all rows at once (the
#     weight sum in the calibrated order -- the same adds, see _m2_finish_rows --, div_rn, * route_scale, ascending
#     sort) -> ids_o [R, k] int64 ascending, w_o [R, k] fp32; PLAN (constexpr, R > 1): runtime/moe_rows.m1_dedup_warp called as a
#     device function on (ids_o, w_o) -- the SAME code the separate dedup launch runs, so the plan tensors (uniq,
#     counts, ..., m_indices tags / psum / masked_m, work list, overflow) are its tensors bit for bit.
#
# M2-T15 fix (docs/HANDOFF-M2-T15-FIX.md): the single-CTA tail above serializes R rows' scoring / top-k / weights and
# the plan in one CTA (R=6: 42.6 us vs fused 39.5). The runtime now launches `m2_router_rows` instead (router_plan.py
# RouterPlan; `m2_router_plan` stays for the bench's before/after and the legacy in-kernel GEMV):
#   grid (R,): program r runs _m2_route on row r alone (RP = 1, r0 = r) -- the SAME per-row arithmetic as the 1-CTA
#     tail (every reduction in _m2_route is along a row; the CTA-wide tie flag only picks between the rank formula and
#     the network, which agree on a row without ties), so each row's ids / w / tk are bit-equal to it by construction;
#   PLAN (constexpr): every program fences, syncs, takes an acq_rel ticket; the LAST to arrive fences again and runs
#     m1_dedup_warp (the same device function, on the R rows' ids_o / w_o) and resets the ticket to 0. The other form
#     (router_plan `split`) launches the rows kernel without PLAN and then moe_rows.dedup_fused (today's dedup launch).
#   SOFTPLUS is a mode: 0 = `raw` already holds F.softplus; 1 = libdevice log1p(exp) (Triton links libdevice with
#     FTZ: results below 2^-126 flush to +0 -- the box probe's 28k / 21M mismatches, docs/router-plan-bench-20261107);
#     2 / 3 = mode 1 where exp(x) >= 2^-126, else exp in fp64 (libdevice __nv_exp: fp64 is never flushed) rounded to
#     fp32 once (2) or first to 24 bits then into the subnormal range (3, a scale-then-round expf's double rounding),
#     log1p(d) = d for those d. 2 / 3 are hypotheses about CUDA's non-FTZ expf: the build-time probe picks the first
#     exact mode, else F.softplus stays a torch op.
# `m2_gate_fp32` (plan-gemv's gate scores): T9b-5's _t9b5_dot (the tuned fixed-order fp32 GEMV of the head / mixes)
# under its own launch name and config table (M2_GATE_CFG); order = KB lanes (router_plan.gate_kb), a function of
# (KB) only -- never of R or of the launch config.
# Every arm of a runtime `if` uses its own variable names (Triton unifies same-named variables across arms).
try:
    from triton.language.extra import libdevice as _m2_ld
except ImportError:  # pragma: no cover - older Triton layouts
    from triton.language.extra.cuda import libdevice as _m2_ld

M2_NET = 32                        # torch SmallBitonicSort's sort_size (Sort.cu): k <= 32 -> one 32-lane network
M2_SOFTPLUS_THRESHOLD = 20.0       # F.softplus's default threshold (beta 1)
M2_WARPS = int(os.environ.get("DSV41_ROUTER_PLAN_WARPS", "4"))
M2_GEMV_CFG = {"BN": 4, "U": 4, "num_warps": 4}   # speed only (the order is a function of KB alone)
# M2-T15 fix: m2_router_rows' warps per row program (1: every 1-D op of the row and of m1_dedup_warp -- designed for
# one warp -- is a register / shuffle op). Speed only: no float reduction in the tail depends on it (sums of one
# value and exact zeros, integer max / sum).
M2_ROWS_WARPS = int(os.environ.get("DSV41_ROUTER_ROWS_WARPS", "1"))
M2_FLT_MIN = 1.1754943508222875e-38   # 2^-126: below it Triton's FTZ libdevice exp returns +0
# plan-gemv's gate GEMV (m2_gate_fp32), keyed by next_power_of_2(R): (RB, BN, U, num_warps, num_stages). Speed only
# (bench/router_plan_bench.py --gate-sweep times candidates; all are bit-identical: the order is KB's).
# Override: DSV41_GATE_CFG="1=1,2,10,4,1;8=1,2,10,4,1".
M2_GATE_CFG = {1: (1, 2, 10, 4, 1), 2: (1, 2, 10, 4, 1), 4: (1, 2, 10, 4, 1), 8: (1, 2, 10, 4, 1)}
M2_GATE_CFG_ENV = "DSV41_GATE_CFG"
_M2 = {"dedup": None}              # runtime/moe_rows.m1_dedup_warp, bound on first use (tests bind the shim's copy)


def _m2_bind_dedup(fn=None) -> None:
    """Bind the device function the PLAN tail calls (moe_rows.m1_dedup_warp unless `fn`, the tests' shim copy)."""
    global m2_dedup_body
    if fn is None:
        from runtime.moe_rows import m1_dedup_warp as fn
    m2_dedup_body = fn
    _M2["dedup"] = fn


# Bound at import on the real Triton, so the plan kernel's JIT cache key (computed at its first launch, from the
# sources of everything it calls) always includes m1_dedup_warp's source: an edit to moe_rows.py recompiles it. Under
# tests/triton_shim.py the tests bind the shim copy (_m2_bind_dedup) -- importing moe_rows here would load it with the
# shim as its triton.
m2_dedup_body = None
if getattr(triton, "__version__", "") != "numpy-shim":
    try:
        from runtime.moe_rows import m1_dedup_warp as m2_dedup_body  # noqa: F811
        _M2["dedup"] = m2_dedup_body
    except ImportError:  # pragma: no cover - bound on first use instead
        m2_dedup_body = None


# --- the weight sum's reduction family (fused_glue_r2's module docstring: tile offsets (k*T + t)*V + v / unrolled
# (k*V + v)*T + t, each accumulator sequential from +0, the V accumulators folded left, then the T tree -- halving above
# 32, below it halving (DEC) or adjacent pairs), here for all R rows at once ([RP, ...] tiles, the same adds per row as
# fused_glue_r2._r2_finish on one row). _r2_offs is a verbatim copy (tests/test_router_plan_cpu.py checks it); the
# rows forms are tested against fused_glue_r2.order_sum_torch's definition for every family shape used.
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


# --- M2-T15 helpers
@triton.jit
def _m2_fence(v):
    """fence.acq_rel.gpu in every thread that holds an element of v (int32); returns v (the value keeps the asm)."""
    return tl.inline_asm_elementwise("fence.acq_rel.gpu;\n\tmov.b32 $0, $1;", "=r,r", [v], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _m2_softplus(v):
    """F.softplus(v) (beta 1, threshold 20) as ATen's CUDA kernel computes it: (v * 1) > 20 ? v : log1p(exp(v)) / 1."""
    return tl.where(v > 20.0, v, _m2_ld.log1p(_m2_ld.exp(v)))


@triton.jit
def _m2_softplus_sub(v, MODE: tl.constexpr):
    """SOFTPLUS modes 2 / 3 (section comment): _m2_softplus where libdevice's (FTZ) exp(v) is >= 2^-126; below, the
    subnormal-or-zero exp(v) from fp64 -- rounded to fp32 once (2) or to 24 bits and then to the subnormal grid (3)
    -- and log1p(d) = d."""
    e = _m2_ld.exp(v)
    lp = _m2_ld.log1p(e)
    ed = _m2_ld.exp(tl.minimum(v, -80.0).to(tl.float64))   # (only lanes with v < -87.3 use it; no overflow)
    if MODE == 2:
        es = ed.to(tl.float32)
    else:
        e24 = (ed * 18446744073709551616.0).to(tl.float32)                 # * 2^64 (exact): 24-bit rounding
        es = (e24.to(tl.float64) * 5.421010862427522e-20).to(tl.float32)  # * 2^-64 (exact), then the subnormal one
    return tl.where(v > 20.0, v, tl.where(e < 1.1754943508222875e-38, es, lp))


@triton.jit
def _m2_softplus_mode(v, MODE: tl.constexpr):
    """SOFTPLUS mode 1 / 2 / 3 (section comment; mode 0 never calls this). Modes 2 / 3 run the fp64 form only when
    some lane of the block needs it (exp(v) < 2^-126 below the threshold): GB300's fp64 rate is a small fraction of
    fp32, so a block without such a lane pays one compare + one reduction only. (`spx` is meant to be unified across
    the two arms: same type, the arm's result.)"""
    sp1 = _m2_softplus(v)
    if MODE == 1:
        spx = sp1
    else:
        n_sub = tl.sum(((_m2_ld.exp(v) < 1.1754943508222875e-38) & (v <= 20.0)).to(tl.int32))
        if n_sub > 0:
            spx = _m2_softplus_sub(v, MODE)
        else:
            spx = sp1
    return spx


@triton.jit
def _m2_radix_key(b):
    """TopKTypeConfig<float>::convert (SortingRadixSelect.cuh) as int64 in [0, 2^32): bits ^ 0x80000000 for a clear
    sign bit, ~bits for a set one, 0xffffffff for NaN. Orders like the floats, with -0 < +0 and every NaN on top."""
    bits = b.to(tl.int32, bitcast=True)
    s = bits.to(tl.int64)
    u = tl.where(s < 0, s + 4294967296, s)                 # the uint32 bits (int64 literals only: no uint32 typing)
    k = tl.where(s < 0, 4294967296 - u - 1, (u + 1073741824) + 1073741824)
    return tl.where(b != b, k * 0 + 4294967296 - 1, k)


@triton.jit
def _m2_gt_key(b):
    """An int32 key whose `>` is GTOp<float, handleNaN=true> (SortingCommon.cuh): NaN above everything (NaNs equal),
    -0 == +0, otherwise the floats' order."""
    bits = b.to(tl.int32, bitcast=True)
    k = tl.where(bits < 0, 0 - (bits & 0x7FFFFFFF), bits)
    return tl.where(b != b, 0x7FFFFFFF, k)


@triton.jit
def _m2_bitonic(net, tk, si, bv, gpos, mr, rr, kk, KC: tl.constexpr):
    """torch's SmallBitonicSort network on [RP, 32] lanes per row (SortUtils.cuh bitonicSort / bitonicSwap): lane
    gpos[r, i] starts with (key of bv, valid, si); lanes >= KC are invalid (key 0.0, value 0). Packed int64 lanes:
    gt_key << 32 | valid << 20 | index. Writes the first KC lanes' indices to tk[r, :KC] (torch.topk's order)."""
    L = tl.arange(0, 32)[None, :]
    base = net + rr * 32
    tl.store(base + L, tl.zeros([1, 32], dtype=tl.int64), mask=mr)
    tl.debug_barrier()
    pk0 = _m2_gt_key(bv).to(tl.int64) * 4294967296 + 1048576 + si
    tl.store(base + gpos, pk0, mask=mr & (kk < KC))
    tl.debug_barrier()
    cur = tl.load(base + L, mask=mr, other=0)
    for lg0 in tl.static_range(5):            # size = 2 << lg0: 2, 4, 8, 16 (flag = pos & size), then 32 (dir false)
        for ls in tl.static_range(lg0 + 1):    # stride = size / 2 ... 1
            STR = 1 << (lg0 - ls)
            tl.debug_barrier()
            tl.store(base + L, cur, mask=mr)
            tl.debug_barrier()
            par = tl.load(base + (L ^ STR), mask=mr, other=0)
            lower = (L & STR) == 0
            a = tl.where(lower, cur, par)
            b = tl.where(lower, par, cur)
            ka = a >> 32
            kb = b >> 32
            va = (a >> 20) & 1
            vb = (b >> 20) & 1
            sw = ((ka > kb) & (va == 1)) | (vb == 0)
            dirv = (L & (2 << lg0)) != 0
            do = sw == dirv
            cur = tl.where(lower, tl.where(do, b, a), tl.where(do, a, b))
    tl.store(tk + rr * KC + L, cur & 1048575, mask=mr & (L < KC))


@triton.jit
def _m2_route(raw, bias, sc, bsc, tk, net, ids_o, w_o, R, E, rs, eps, r0,
              SOFTPLUS: tl.constexpr, RP: tl.constexpr, EP: tl.constexpr, KC: tl.constexpr, KP: tl.constexpr,
              OT: tl.constexpr, OV: tl.constexpr, OU: tl.constexpr, ODEC: tl.constexpr):
    """The tail's routing part for rows [r0, r0 + RP) (one CTA): scores, the top-k with torch.topk's CUDA result, the
    weights. Every reduction is along a row, so a row's results do not depend on r0 / RP (the 1-CTA tail: r0 = 0,
    RP = R's power of two; m2_router_rows: r0 = the program's row, RP = 1)."""
    rr = r0 + tl.arange(0, RP)[:, None]
    ee = tl.arange(0, EP)[None, :]
    mr = rr < R
    me = ee < E
    m = mr & me
    v = tl.load(raw + rr * E + ee, mask=m, other=0.0, cache_modifier=".cg")
    if SOFTPLUS == 0:
        sp = v
    else:
        sp = _m2_softplus_mode(v, SOFTPLUS)
    s_ = _m2_ld.sqrt_rn(sp)
    b_ = s_ + tl.load(bias + ee, mask=me, other=0.0)
    tl.store(sc + rr * E + ee, s_, mask=m)
    tl.store(bsc + rr * E + ee, b_, mask=m)
    # selection (gatherTopK's set): k rounds of max over (radix key, lowest index)
    pk = tl.where(m, _m2_radix_key(b_) * 65536 + (65535 - ee), -1)
    kk = tl.arange(0, KP)[None, :]
    si = tl.zeros([RP, KP], dtype=tl.int64)
    sk = tl.zeros([RP, KP], dtype=tl.int64)
    for j in tl.static_range(KC):
        mx = tl.max(pk, axis=1)
        si = tl.where(kk == j, (65535 - (mx & 65535))[:, None], si)
        sk = tl.where(kk == j, (mx >> 16)[:, None], sk)
        pk = tl.where(pk == mx[:, None], -1, pk)
    tl.debug_barrier()                                      # sc / bsc visible to the whole CTA
    vk = kk < KC
    vrk = mr & vk
    bv = tl.load(bsc + rr * E + si, mask=vrk, other=0.0)
    scv = tl.load(sc + rr * E + si, mask=vrk, other=0.0)
    # gatherTopK's output layout: above-k-th in index order, then the tied-at-k-th in index order
    kth = tl.max(tl.where(kk == KC - 1, sk, -1), axis=1)
    g32 = ((sk > kth[:, None]) & vk).to(tl.int32)
    ki = tl.arange(0, KP)[None, :, None]
    kj = tl.arange(0, KP)[None, None, :]
    vj3 = (kj < KC).to(tl.int32)
    bef = (si[:, None, :] < si[:, :, None]).to(tl.int32) * vj3
    n_gt = tl.sum(g32, axis=1)
    c_gt = tl.sum(bef * g32[:, None, :], axis=2)
    c_eq = tl.sum(bef * (1 - g32[:, None, :]), axis=2)
    gpos = tl.where(g32 == 1, c_gt, n_gt[:, None] + c_eq)
    # SmallBitonicSort: descending by GTOp; ties -> the network itself
    gk = _m2_gt_key(bv)
    gji = (gk[:, None, :] > gk[:, :, None]).to(tl.int32)    # [r, i, j]: value j above value i
    pos = tl.sum(gji * vj3, axis=2)
    teq = (gk[:, None, :] == gk[:, :, None]).to(tl.int32) * (ki != kj).to(tl.int32) * vj3 \
        * (ki < KC).to(tl.int32) * mr.to(tl.int32)[:, :, None]
    n_tie = tl.sum(tl.sum(tl.sum(teq, axis=2), axis=1), axis=0)
    if n_tie == 0:
        tl.store(tk + rr * KC + pos, si, mask=vrk)
    else:
        _m2_bitonic(net, tk, si, bv, gpos, mr, rr, kk, KC)
    tl.debug_barrier()
    # Gate.forward's tail in torch.topk's order (t9b2r2_router_post's arithmetic, all rows at once): weights =
    # scores[idx], their sum in the calibrated order, w = weights / (sum + eps) * route_scale, ascending by id
    tkv = tl.load(tk + rr * KC + kk, mask=vrk, other=0)                                   # [RP, KP] topk order
    hit = (si[:, None, :] == tkv[:, :, None]) & (kj < KC) & (ki < KC)                     # [r, p, i]
    wv = tl.sum(tl.where(hit, scv[:, None, :], 0.0), axis=2)                              # sc[tk[r, p]] (exact)
    acc = tl.zeros([RP, OT, OV], dtype=tl.float32)
    pp = tl.arange(0, KP)[None, None, :]
    for t2 in tl.static_range((KC + OT * OV - 1) // (OT * OV)):
        e = tl.reshape(_r2_offs(t2, OT, OV, OU), [OT * OV])                               # tile t2's positions
        val = tl.sum(tl.where(pp == e[None, :, None], wv[:, None, :], 0.0), axis=2)        # [RP, OT*OV]
        acc = acc + tl.reshape(tl.where((e < KC)[None, :], val, 0.0), [RP, OT, OV])
    den = _m2_finish_rows(acc, RP, OT, OV, ODEC) + eps
    wj = _m2_ld.div_rn(wv, den[:, None]) * rs
    rank = tl.sum(((tkv[:, None, :] < tkv[:, :, None]) & (kj < KC)).to(tl.int32), axis=2)  # distinct ids per row
    tl.store(ids_o + rr * KC + rank, tkv, mask=vrk)
    tl.store(w_o + rr * KC + rank, wj, mask=vrk)
    tl.debug_barrier()


@triton.jit
def _m2_tail(raw, bias, sc, bsc, tk, net, ids_o, w_o, R, E, rs, eps,
             uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
             mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
             BR, H, S, TS, G, GSCR, SOFTPLUS: tl.constexpr, RP: tl.constexpr, EP: tl.constexpr, KC: tl.constexpr,
             KP: tl.constexpr, OT: tl.constexpr, OV: tl.constexpr, OU: tl.constexpr, ODEC: tl.constexpr,
             PLAN: tl.constexpr, DKP: tl.constexpr, NRP: tl.constexpr, HAS_MM: tl.constexpr, HAS_CTR: tl.constexpr,
             INPL: tl.constexpr, HAS_DUMMY: tl.constexpr, LAYOUT: tl.constexpr, HAS_TAGS: tl.constexpr,
             HAS_WL: tl.constexpr, STEP_SF: tl.constexpr, HAS_OVF: tl.constexpr):
    """One CTA: the routing (_m2_route), then (PLAN) moe_rows.m1_dedup_warp on its outputs."""
    _m2_route(raw, bias, sc, bsc, tk, net, ids_o, w_o, R, E, rs, eps, 0, SOFTPLUS, RP, EP, KC, KP, OT, OV, OU, ODEC)
    if PLAN:
        m2_dedup_body(ids_o, w_o, uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                      mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                      R * KC, KC, R, BR, H, S, TS, G, GSCR,
                      DKP, NRP, HAS_MM, HAS_CTR, INPL, HAS_DUMMY, LAYOUT, HAS_TAGS, HAS_WL, STEP_SF, HAS_OVF)


@triton.jit
def m2_router_plan(x, gw, raw, bias, sc, bsc, tk, net, ticket, ids_o, w_o,
                   R, E, sxr, swn, rs, eps, NPROG,
                   uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                   mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                   BR, H, S, TS, G, GSCR,
                   GEMV: tl.constexpr, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr, RB: tl.constexpr,
                   U: tl.constexpr, FT: tl.constexpr, SOFTPLUS: tl.constexpr,
                   RP: tl.constexpr, EP: tl.constexpr, KC: tl.constexpr, KP: tl.constexpr,
                   OT: tl.constexpr, OV: tl.constexpr, OU: tl.constexpr, ODEC: tl.constexpr,
                   PLAN: tl.constexpr, DKP: tl.constexpr, NRP: tl.constexpr, HAS_MM: tl.constexpr,
                   HAS_CTR: tl.constexpr, INPL: tl.constexpr, HAS_DUMMY: tl.constexpr, LAYOUT: tl.constexpr,
                   HAS_TAGS: tl.constexpr, HAS_WL: tl.constexpr, STEP_SF: tl.constexpr, HAS_OVF: tl.constexpr,
                   PDL: tl.constexpr = 0):
    """M2-T15: the router (+ GEMV: the gate mm) (+ PLAN: the R-row MoE plan) in one launch (section comment).
    M2-L5: PDL (runtime/pdl.py) -- the wait is above the GEMV and the ticket."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    if GEMV:
        _t9b5_dot(x, gw, raw, raw, R, E, sxr, swn, E, NK, KB, BN, RB, U, False)
        fz = _m2_fence(tl.zeros([FT], dtype=tl.int32))
        tl.debug_barrier()
        old = tl.atomic_add(ticket, 1 + tl.sum(fz, axis=0), sem="acq_rel", scope="gpu")
        if old == NPROG - 1:                                  # the last program to arrive: every raw score is in
            fz2 = _m2_fence(tl.zeros([FT], dtype=tl.int32))
            tl.debug_barrier()
            _m2_tail(raw + tl.sum(fz2, axis=0), bias, sc, bsc, tk, net, ids_o, w_o, R, E, rs, eps,
                     uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                     mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                     BR, H, S, TS, G, GSCR, SOFTPLUS, RP, EP, KC, KP, OT, OV, OU, ODEC,
                     PLAN, DKP, NRP, HAS_MM, HAS_CTR, INPL, HAS_DUMMY, LAYOUT, HAS_TAGS, HAS_WL, STEP_SF, HAS_OVF)
            tl.store(ticket, 0)
    else:
        _m2_tail(raw, bias, sc, bsc, tk, net, ids_o, w_o, R, E, rs, eps,
                 uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                 mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                 BR, H, S, TS, G, GSCR, SOFTPLUS, RP, EP, KC, KP, OT, OV, OU, ODEC,
                 PLAN, DKP, NRP, HAS_MM, HAS_CTR, INPL, HAS_DUMMY, LAYOUT, HAS_TAGS, HAS_WL, STEP_SF, HAS_OVF)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def m2_router_rows(raw, bias, sc, bsc, tk, net, ticket, ids_o, w_o, R, E, rs, eps,
                   uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                   mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                   BR, H, S, TS, G, GSCR,
                   FT: tl.constexpr, SOFTPLUS: tl.constexpr, EP: tl.constexpr, KC: tl.constexpr,
                   KP: tl.constexpr, OT: tl.constexpr, OV: tl.constexpr, OU: tl.constexpr, ODEC: tl.constexpr,
                   PLAN: tl.constexpr, DKP: tl.constexpr, NRP: tl.constexpr, HAS_MM: tl.constexpr,
                   HAS_CTR: tl.constexpr, INPL: tl.constexpr, HAS_DUMMY: tl.constexpr, LAYOUT: tl.constexpr,
                   HAS_TAGS: tl.constexpr, HAS_WL: tl.constexpr, STEP_SF: tl.constexpr, HAS_OVF: tl.constexpr,
                   PDL: tl.constexpr = 0):
    """M2-T15 fix (section comment): grid (R,), program r routes row r; PLAN: the last program to arrive runs
    m1_dedup_warp on all R rows' ids_o / w_o and resets the ticket. M2-L5: PDL (runtime/pdl.py) -- the wait is above
    every load and the ticket; the inlined m1_dedup_warp keeps its PDL default 0 (no gdc_* inside)."""
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    r0 = tl.program_id(0)
    _m2_route(raw, bias, sc, bsc, tk, net, ids_o, w_o, R, E, rs, eps, r0, SOFTPLUS, 1, EP, KC, KP, OT, OV, OU, ODEC)
    if PLAN:
        fz = _m2_fence(tl.zeros([FT], dtype=tl.int32))          # this row's ids_o / w_o stores before the ticket
        tl.debug_barrier()
        old = tl.atomic_add(ticket, 1 + tl.sum(fz, axis=0), sem="acq_rel", scope="gpu")
        if old == R - 1:                                          # the last row in: every row's ids / w are written
            fz2 = _m2_fence(tl.zeros([FT], dtype=tl.int32))
            tl.debug_barrier()
            z2 = tl.sum(fz2, axis=0)                              # (0: the loads below depend on the fence)
            m2_dedup_body(ids_o + z2, w_o + z2, uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o,
                          posmat_o, mm_o, ids_s, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                          R * KC, KC, R, BR, H, S, TS, G, GSCR,
                          DKP, NRP, HAS_MM, HAS_CTR, INPL, HAS_DUMMY, LAYOUT, HAS_TAGS, HAS_WL, STEP_SF, HAS_OVF)
            tl.store(ticket, 0)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def m2_gate_fp32(x, w, out, R, N, sxr, swn, sor, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr,
                 RB: tl.constexpr, U: tl.constexpr, PDL: tl.constexpr = 0, PDL_PF: tl.constexpr = 0,
                 PF_EPL: tl.constexpr = 32, PF_NL: tl.constexpr = 1, PF_CH: tl.constexpr = 1):
    """plan-gemv's gate scores (model.py:819's x.float() @ W^T): T9b-5's fixed-order fp32 dot, its own launch name.
    M2-L5: PDL_PF prefetches the program's BN gate-weight rows, then gdc_wait (runtime/pdl.py)."""
    if PDL_PF:
        _pdl_prefetch(w, tl.program_id(0) * BN, N, swn, 0, NK * KB, BN, PF_EPL, PF_NL, PF_CH)
    if PDL > 0:
        tl.extra.cuda.gdc_wait()
    if PDL == 1:
        tl.extra.cuda.gdc_launch_dependents()
    _t9b5_dot(x, w, out, out, R, N, sxr, swn, sor, NK, KB, BN, RB, U, False)
    if PDL == 2:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def m2_softplus_probe(x, out, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    """The tail's softplus (SOFTPLUS mode MODE) on n values (the build-time / GPU-test probe vs F.softplus)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, _m2_softplus_mode(tl.load(x + i, mask=m, other=0.0), MODE), mask=m)


def softplus_probe(x: torch.Tensor, mode: int = 1) -> torch.Tensor:
    """m2_softplus_probe over x (fp32, contiguous): the tail's softplus values in SOFTPLUS mode `mode` (1, 2, 3)."""
    if int(mode) not in (1, 2, 3):
        raise ValueError(f"softplus_probe: mode {mode} not in (1, 2, 3)")
    x = x.contiguous().view(-1)
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    m2_softplus_probe[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK, int(mode), num_warps=4, enable_fp_fusion=False)
    return out


def _m2_gate_env_cfg() -> dict:
    out = {}
    for item in (os.environ.get(M2_GATE_CFG_ENV) or "").split(";"):
        item = item.strip()
        if item:
            key, val = item.split("=")
            out[int(key)] = tuple(int(v) for v in val.split(","))
    return out


def gate_config(R: int) -> tuple:
    """(RB, BN, U, num_warps, num_stages) of m2_gate_fp32 at R rows (env override, else M2_GATE_CFG)."""
    rp = min(8, triton.next_power_of_2(max(1, int(R))))
    return _m2_gate_env_cfg().get(rp) or M2_GATE_CFG.get(rp) or M2_GATE_CFG[max(M2_GATE_CFG)]


def gate_dot(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, *, kb: int, cfg: Optional[tuple] = None) -> None:
    """plan-gemv: out [R, E] fp32 <- x [R, D] . w [E, D]^T in the T9b-5 order of KB = kb lanes (fixed_dot's contract;
    fused_glue_r2.fixed_dot_torch is the CPU definition). cfg = (RB, BN, U, num_warps, num_stages): speed only."""
    if x.dim() != 2 or w.dim() != 2 or out.dim() != 2:
        raise ValueError(f"gate_dot: x {tuple(x.shape)}, w {tuple(w.shape)}, out {tuple(out.shape)} must be 2-D")
    R, K = x.shape
    N = w.size(0)
    if w.size(1) != K or tuple(out.shape) != (R, N) or out.dtype != torch.float32:
        raise ValueError(f"gate_dot: x [{R}, {K}], w {tuple(w.shape)}, out {tuple(out.shape)} {out.dtype}")
    if x.stride(1) != 1 or w.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("gate_dot: x / w / out rows must be contiguous")
    if kb < 1 or kb & (kb - 1) or kb > T9B5_KB_MAX or K % kb:
        raise ValueError(f"gate_dot: kb {kb} must be a power of two <= {T9B5_KB_MAX} dividing K = {K}")
    if R == 0:
        return
    rb, bn, u, nw, ns = cfg if cfg is not None else gate_config(R)
    nk = K // kb
    u = max(1, min(int(u), nk))
    m2_gate_fp32[(triton.cdiv(N, bn), triton.cdiv(R, rb))](
        x, w, out, R, N, x.stride(0), w.stride(0), out.stride(0), nk, kb, bn, rb, u,
        num_warps=nw, num_stages=ns, enable_fp_fusion=False,
        **_pdl.kw("m2_gate_fp32", pf=lambda: _pdl.pf_rows(K, w.element_size(), bn)))


def _plan_kw(plan: Optional[dict], ids: torch.Tensor):
    if plan is not None:
        return plan["ptrs"], plan["ints"], dict(plan["cx"], PLAN=True)
    return ((ids,) * 21, (0,) * 6,
            dict(PLAN=False, DKP=1, NRP=1, HAS_MM=False, HAS_CTR=False, INPL=False, HAS_DUMMY=False, LAYOUT=0,
                 HAS_TAGS=False, HAS_WL=False, STEP_SF=False, HAS_OVF=False))


def router_rows_launch(raw: torch.Tensor, bias: torch.Tensor, ws: dict, ids: torch.Tensor, w: torch.Tensor, *,
                       R: int, E: int, K: int, order: dict, rs: float, eps: float, softplus: int,
                       plan: Optional[dict] = None, num_warps: Optional[int] = None) -> None:
    """Launch m2_router_rows, grid (R,) (M2-T15 fix; runtime/router_plan.py builds the arguments). raw [R, E] fp32
    row-major: the gate scores (softplus = SOFTPLUS mode 1 / 2 / 3) or F.softplus of them (softplus = 0); ws: sc /
    bsc [R, E] fp32, tk [R, K] int64, net [>= R, 32] int64, ticket int32 [1] (0 between launches); ids [R, K] int64 /
    w [R, K] fp32 out (row-major); order = fused_glue_r2.order_args of the weight-sum order; plan = router_plan
    .dedup_args (the last row program runs m1_dedup_warp on them) or None."""
    EP = triton.next_power_of_2(int(E))
    KP = triton.next_power_of_2(int(K))
    if K > M2_NET or not 1 <= int(R) <= 8 or E >= 65536:
        raise ValueError(f"router_rows: R {R}, E {E}, K {K} outside the kernel's range (1 <= R <= 8, E < 65536, "
                         f"K <= 32)")
    if int(softplus) not in (0, 1, 2, 3):
        raise ValueError(f"router_rows: softplus mode {softplus} not in (0, 1, 2, 3)")
    if raw.stride(-1) != 1 or (raw.dim() == 2 and raw.stride(0) != int(E)):
        raise ValueError("router_rows: raw must be row-major [R, E]")
    if plan is not None and _M2["dedup"] is None:
        _m2_bind_dedup()
    nw = int(num_warps or M2_ROWS_WARPS)
    ptrs, ints, cx = _plan_kw(plan, ids)
    m2_router_rows[(int(R),)](
        raw, bias, ws["sc"], ws["bsc"], ws["tk"], ws["net"], ws["ticket"], ids, w, int(R), int(E), float(rs),
        float(eps), *ptrs, *ints,
        FT=32 * nw, SOFTPLUS=int(softplus), EP=EP, KC=int(K), KP=KP, OT=int(order["T"]), OV=int(order["V"]),
        OU=bool(order["U"]), ODEC=bool(order["DEC"]), num_warps=nw, enable_fp_fusion=False, **cx,
        **_pdl.kw("m2_router_rows"))


def router_plan_launch(raw: torch.Tensor, bias: torch.Tensor, ws: dict, ids: torch.Tensor, w: torch.Tensor, *,
                       R: int, E: int, K: int, order: dict, rs: float, eps: float, softplus: int,
                       x: Optional[torch.Tensor] = None, gw: Optional[torch.Tensor] = None, kb: int = 64,
                       gemv_cfg: Optional[dict] = None, plan: Optional[dict] = None,
                       num_warps: Optional[int] = None) -> None:
    """Launch m2_router_plan -- the M2-T15 round-1 single-CTA form (kept for the bench's before / after and its
    in-kernel GEMV; the runtime uses router_rows_launch). raw [R, E] fp32: the gate scores (plan) or
    the GEMV's output scratch (x [R, D] + gw [E, D] given: plan-gemv); ws: sc / bsc [R, E] fp32, tk [R, K] int64, net
    [RP, 32] int64, ticket int32 [1] (0 between launches); ids [R, K] int64 / w [R, K] fp32 out; order = the
    calibrated weight-sum order's constexprs (fused_glue_r2.order_args: T, V, U, DEC); plan = the dedup's arguments
    (router_plan.dedup_args: `ptrs` in m1_dedup_warp's order, `ints` (BR, H, S, TS, G, GSCR) and `cx` constexprs).
    softplus: a SOFTPLUS mode (bool True = 1)."""
    RP = triton.next_power_of_2(max(1, int(R)))
    EP = triton.next_power_of_2(int(E))
    KP = triton.next_power_of_2(int(K))
    if K > M2_NET or RP > 8 or E >= 65536:
        raise ValueError(f"router_plan: R {R}, E {E}, K {K} outside the kernel's range (R <= 8, E < 65536, K <= 32)")
    gemv = x is not None
    if plan is not None and _M2["dedup"] is None:
        _m2_bind_dedup()
    if gemv:
        cfg = dict(M2_GEMV_CFG, **(gemv_cfg or {}))
        D = x.size(-1)
        if x.stride(-1) != 1 or gw.stride(-1) != 1 or D % kb:
            raise ValueError("router_plan: x / gate weight rows must be contiguous and K a multiple of KB")
        bn, u, nw = int(cfg["BN"]), max(1, min(int(cfg["U"]), D // kb)), int(cfg["num_warps"])
        nprog = triton.cdiv(int(E), bn)
        g = dict(GEMV=True, NK=D // kb, KB=int(kb), BN=bn, RB=RP, U=u, FT=32 * nw)
        xs, ws_ = x.stride(0) if x.dim() == 2 else D, gw.stride(0)
    else:
        nw = int(num_warps or M2_WARPS)
        nprog = 1
        g = dict(GEMV=False, NK=1, KB=1, BN=1, RB=1, U=1, FT=32 * nw)
        x = gw = raw
        xs = ws_ = 0
    if plan is not None:
        ptrs, ints, cx = plan["ptrs"], plan["ints"], dict(plan["cx"], PLAN=True)
    else:
        ptrs, ints, cx = _plan_kw(None, ids)
    m2_router_plan[(nprog,)](
        x, gw, raw, bias, ws["sc"], ws["bsc"], ws["tk"], ws["net"], ws["ticket"], ids, w,
        int(R), int(E), xs, ws_, float(rs), float(eps), nprog, *ptrs, *ints,
        SOFTPLUS=int(softplus), RP=RP, EP=EP, KC=int(K), KP=KP, OT=int(order["T"]), OV=int(order["V"]),
        OU=bool(order["U"]), ODEC=bool(order["DEC"]), num_warps=nw, enable_fp_fusion=False, **g, **cx,
        **_pdl.kw("m2_router_plan"))
