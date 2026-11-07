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
                       IDS: tl.constexpr, SIDX: tl.constexpr):
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
                                    use_sidx, num_warps=int(num_warps or VEC_WARPS))


def step_marker(ctr: torch.Tensor) -> None:
    """One program, one warp: `ctr[0] += 1`. Fixed kernel name `t8_step_marker` for nsys."""
    t8_step_marker[(1,)](ctr, num_warps=1)


# ------------------------------------------------------------------------------------------------ M1-T10-MOE (rows)
@triton.jit
def m1_stage_rows(ptab, uniq, gtab, sidx, wl, dtab, E, H, N13, C13, N2, C2,
                  ROW_W: tl.constexpr, BW: tl.constexpr, NBW: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
                  NPROG: tl.constexpr, EVEN_W: tl.constexpr):
    """M1-T10-MOE: the R-row dg-inplace staging launch, driven by the dedup's work list. Program (j, q, p): slot
    u = wl[j] (-1: nothing to do -- the program exits at once), expert e = uniq[u], g = gtab[e]. A cold slot (g < 0;
    the dedup lists only staged ones, compact index c = sidx[u] < spare budget) copies 8-KiB weight blocks q,
    q+NPROG, ... of row ptab[p, e] (HBM or UVA) into pool slot H + c (int64 words, 16-B vectors), then packs its
    scales into group H + c of the packed SF; a hot slot (listed only under sf_store "step") packs its scales into
    group g. So the scale work is once per unique expert (<= U slots, not R*k blocks) and, with the hot SF
    pre-packed (sf_store "pool" / "resident"), only cold experts are touched at all. m_indices, the overflow count
    and the stage ids are the dedup's (moe_rows.m1_dedup_warp). Contract: dg_inplace.stage_rows_emulated."""
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
                                num_warps=int(num_warps or VEC_WARPS))


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
                   RB: tl.constexpr, U: tl.constexpr, HAS_SC: tl.constexpr):
    """The head (model.py:1014): logits [R, V] = x [R, D] . W [V, D]^T (named launch for nsys)."""
    _t9b5_dot(x, w, out, sc, R, N, sxr, swn, sor, NK, KB, BN, RB, U, HAS_SC)


@triton.jit
def t9b5_mix_fp32(x, w, out, sc, R, N, sxr, swn, sor, NK: tl.constexpr, KB: tl.constexpr, BN: tl.constexpr,
                  RB: tl.constexpr, U: tl.constexpr, HAS_SC: tl.constexpr):
    """The mHC mixes (model.py:956): [R, mix_hc] = (x [R, hc*D] . hc_fn^T) * rsqrt[R] (named launch for nsys)."""
    _t9b5_dot(x, w, out, sc, R, N, sxr, swn, sor, NK, KB, BN, RB, U, HAS_SC)


T9B5_KERNELS = {"head": t9b5_head_fp32, "mix": t9b5_mix_fp32}


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
    it); the head puts every row (<= 8) in one program so its 2.65 GB weight is read once per step."""
    if site == "mix":
        return 1
    return min(T9B5_RB_MAX, triton.next_power_of_2(max(1, int(R))))


def t9b5_config(site: str, R: int) -> tuple:
    """(RB, BN, U, num_warps, num_stages) for a call of `site` at R rows."""
    rb = t9b5_rows_per_program(site, R)
    env = _t9b5_env_cfg()
    bn, u, nw, ns = env.get((site, rb)) or T9B5_CFG[site].get(rb) or T9B5_CFG[site][max(T9B5_CFG[site])]
    return rb, bn, u, nw, ns


def fixed_dot(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, *, kb: int, site: str,
              scale: Optional[torch.Tensor] = None, cfg: Optional[tuple] = None) -> None:
    """T9b-5: out [R, N] fp32 <- x [R, K] . w [N, K]^T (* scale [R]) in the fixed order of KB = kb lanes (the
    contract above; fused_glue_r2.fixed_dot_torch on CPU). site "head" / "mix" picks the named kernel and the
    default launch config; cfg = (RB, BN, U, num_warps, num_stages) overrides it (speed only)."""
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
    rb, bn, u, nw, ns = cfg if cfg is not None else t9b5_config(site, R)
    nk = K // kb
    u = max(1, min(int(u), nk))
    T9B5_KERNELS[site][(triton.cdiv(N, bn), triton.cdiv(R, rb))](
        x, w, out, scale if scale is not None else out, R, N, x.stride(0), w.stride(0), out.stride(0),
        nk, kb, bn, rb, u, scale is not None, num_warps=nw, num_stages=ns, enable_fp_fusion=False)
