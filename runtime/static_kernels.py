"""Triton kernels for the T8 static step (GPU only; imported lazily by runtime/static_step.py and runtime/generate.py,
so the Mac tests never need triton).

  t8_gather_rows   the spike's pointer-table gather (bench/moe_static_spike.py:117-130), unchanged: program (j, blk)
                   copies bytes [blk*BLOCK, (blk+1)*BLOCK) of the row whose address is ptab[ids[j]] into slot j of a
                   static staging buffer. The address may be an HBM pool slot or a UVA address of registered Grace
                   memory; the kernel does not care which (that is how cold experts cross C2C inside the graph).
  t8_step_marker   a one-program no-op with a fixed name (it increments a device counter), launched once per decode
                   step so nsys can find step boundaries after graph replay, where NVTX ranges do not survive
                   (T8-T9A-DESIGN.md section 2.3 step 6).
  t9b2_gather_tables  T9b-2 (`--glue-impl fused`, clusters #11 + #9): a layer's six gathers in one launch, dg-masked's
                   e8m0 -> fp32 scale conversion and the ids_step copy folded in (data movement + an exact decode).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

GATHER_BLOCK = 8192  # bytes per program, as the spike (bench/moe_static_spike.py:126)


@triton.jit
def t8_gather_rows(ptab, ids, dst, row_bytes: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    blk = tl.program_id(1)
    e = tl.load(ids + j)
    src = tl.load(ptab + e).to(tl.pointer_type(tl.uint8))
    off = blk * BLOCK + tl.arange(0, BLOCK)
    m = off < row_bytes
    v = tl.load(src + off, mask=m)
    tl.store(dst + j * row_bytes + off, v, mask=m)


def gather_rows(ptab: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor, row_bytes: int) -> None:
    """dst: contiguous [k, ...] staging, k = ids.numel(), each slot `row_bytes` bytes (viewed as uint8 here so the
    store type equals the load type). ptab: int64 [n_experts] addresses; ids: int64 [k] on the device."""
    k = ids.numel()
    assert dst.is_contiguous() and dst.numel() * dst.element_size() == k * row_bytes, (dst.shape, k, row_bytes)
    t8_gather_rows[(k, triton.cdiv(row_bytes, GATHER_BLOCK))](ptab, ids, dst.view(torch.uint8), row_bytes,
                                                              GATHER_BLOCK)


@triton.jit
def t8_step_marker(ctr):
    tl.store(ctr, tl.load(ctr) + 1)


def step_marker(ctr: torch.Tensor) -> None:
    """One program, one warp: `ctr[0] += 1`. Fixed kernel name `t8_step_marker` for nsys."""
    t8_step_marker[(1,)](ctr, num_warps=1)


# ------------------------------------------------------------------------------------------------ T9b-2 (#11 + #9)
@triton.jit
def t9b2_gather_tables(ptab, ids, dtab, ids_out, E, row_w, row_s, NBW, BLOCK: tl.constexpr,
                       SCALE_F32: tl.constexpr, IDS: tl.constexpr):
    """One launch for a layer's six staging tables. ptab: int64 [6, E] source row addresses (w1, w3, w2 weights, then
    their scales); dtab: int64 [6] destination bases. Program (j, blk, p): blk < NBW copies bytes [blk*BLOCK, ...) of
    projection p's weight row ids[j] (t8_gather_rows, unchanged); blk >= NBW handles its scale row: bytes as they are
    (SCALE_F32 0) or each e8m0 code decoded to its exact fp32 value (SCALE_F32 1: 0 -> 2^-127, 255 -> NaN, else
    code << 23 -- runtime/dg_moe.py's convert_scales, which this replaces) at destination slot j. IDS: program
    (j, 0, 0) also writes ids[j] to ids_out[j] (StaticStep.ids_step's row: the counters' copy)."""
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


def gather_tables(ptab6: torch.Tensor, ids: torch.Tensor, dtab: torch.Tensor, dsts, row_w: int, row_s: int,
                  scale_f32: bool, ids_out=None) -> None:
    """T9b-2: the six gather_rows calls of a layer (+ dg-masked's three scale conversions, + the ids_step copy) as ONE
    launch. ptab6 int64 [6, E] (w1/w3/w2 weight rows, then w1/w3/w2 scale rows); dtab int64 [6] = the data_ptr()s of
    `dsts` (staging weights x3, then staging scales x3 or dg's fp32 sf32 x3), built once at construction (no H2D copy
    under capture); `dsts` is unused here (the CPU emulation in the tests uses it). ids_out: int64 [k] or None."""
    k = ids.numel()
    nbw = triton.cdiv(row_w, GATHER_BLOCK)
    nbs = triton.cdiv(row_s, GATHER_BLOCK)
    t9b2_gather_tables[(k, nbw + nbs, 3)](ptab6, ids, dtab, ids_out if ids_out is not None else ids,
                                         ptab6.size(1), row_w, row_s, nbw, GATHER_BLOCK, bool(scale_f32),
                                         ids_out is not None)
