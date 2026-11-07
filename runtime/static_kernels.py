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
@triton.jit
def t8_gather_rows_u8(ptab, ids, dst, row_bytes: tl.constexpr, BLOCK: tl.constexpr):
    """The T8 kernel (bench/moe_static_spike.py:117-130), unchanged: one byte per lane."""
    j = tl.program_id(0)
    blk = tl.program_id(1)
    e = tl.load(ids + j)
    src = tl.load(ptab + e).to(tl.pointer_type(tl.uint8))
    off = blk * BLOCK + tl.arange(0, BLOCK)
    m = off < row_bytes
    v = tl.load(src + off, mask=m)
    tl.store(dst + j * row_bytes + off, v, mask=m)


@triton.jit
def t9b2_gather_tables_u8(ptab, ids, dtab, ids_out, E, row_w, row_s, NBW, BLOCK: tl.constexpr,
                          SCALE_F32: tl.constexpr, IDS: tl.constexpr):
    """The T9b-2 kernel, unchanged (contract: t9b2_gather_tables)."""
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


# ------------------------------------------------------------------------------------------------ T9b-3a (vec)
@triton.jit
def t8_gather_rows(ptab, ids, dst, ROW_WORDS: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    """Program (j, blk): int64 words [blk*BLOCK, (blk+1)*BLOCK) of row ptab[ids[j]] -> the same words of slot j of
    dst (int64 view). Row address 16-B aligned (host-checked) and declared so -> 16-B vector loads/stores."""
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


@triton.jit
def t9b2_gather_tables(ptab, ids, dtab, ids_out, E, ROW_W: tl.constexpr, ROW_S: tl.constexpr, NBW: tl.constexpr,
                       BW: tl.constexpr, BS: tl.constexpr, EVEN_W: tl.constexpr, EVEN_S: tl.constexpr,
                       SCALE_F32: tl.constexpr, IDS: tl.constexpr):
    """One launch for a layer's six staging tables. ptab: int64 [6, E] source row addresses (w1, w3, w2 weights, then
    their scales); dtab: int64 [6] destination bases. Program (j, blk, p): blk < NBW copies int64 words
    [blk*BW, ...) of projection p's weight row ids[j] (ROW_W words); blk >= NBW handles its scale row (ROW_S bytes):
    as int64 words, BS per program (SCALE_F32 0), or BS e8m0 codes per program decoded to their exact fp32 values
    (SCALE_F32 1: 0 -> 2^-127, 255 -> NaN, else code << 23 -- runtime/dg_moe.py's convert_scales, which this
    replaces) at destination slot j. IDS: program (j, 0, 0) also writes ids[j] to ids_out[j] (StaticStep.ids_step's
    row). All row addresses 16-B aligned (host-checked) and declared so -> 16-B vectors per lane."""
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
                                                        words % bw == 0, num_warps=int(num_warps or VEC_WARPS))
            return
        except Exception as e:
            if explicit:
                raise
            _vec_failed(e)
    t8_gather_rows_u8[(k, triton.cdiv(row_bytes, GATHER_BLOCK))](ptab, ids, dst.view(torch.uint8), row_bytes,
                                                                 GATHER_BLOCK)


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
                                                 num_warps=int(num_warps or VEC_WARPS))
            return
        except Exception as e:
            if explicit:
                raise
            _vec_failed(e)
    nbw = triton.cdiv(row_w, GATHER_BLOCK)
    nbs = triton.cdiv(row_s, GATHER_BLOCK)
    t9b2_gather_tables_u8[(k, nbw + nbs, 3)](ptab6, ids, dtab, io, ptab6.size(1), row_w, row_s, nbw, GATHER_BLOCK,
                                            bool(scale_f32), ids_out is not None)


@triton.jit
def t8_step_marker(ctr):
    tl.store(ctr, tl.load(ctr) + 1)


def step_marker(ctr: torch.Tensor) -> None:
    """One program, one warp: `ctr[0] += 1`. Fixed kernel name `t8_step_marker` for nsys."""
    t8_step_marker[(1,)](ctr, num_warps=1)
