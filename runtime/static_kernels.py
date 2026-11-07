"""Triton kernels for the T8 static step (GPU only; imported lazily by runtime/static_step.py and runtime/generate.py,
so the Mac tests never need triton).

  t8_gather_rows   the spike's pointer-table gather (bench/moe_static_spike.py:117-130), unchanged: program (j, blk)
                   copies bytes [blk*BLOCK, (blk+1)*BLOCK) of the row whose address is ptab[ids[j]] into slot j of a
                   static staging buffer. The address may be an HBM pool slot or a UVA address of registered Grace
                   memory; the kernel does not care which (that is how cold experts cross C2C inside the graph).
  t8_step_marker   a one-program no-op with a fixed name (it increments a device counter), launched once per decode
                   step so nsys can find step boundaries after graph replay, where NVTX ranges do not survive
                   (T8-T9A-DESIGN.md section 2.3 step 6).
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
