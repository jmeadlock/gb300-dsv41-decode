"""T9a-2: in-place hot MoE (`--moe-impl dg-inplace`): DeepGEMM's CONTIGUOUS grouped GEMM reads the hot experts
straight from the HBM pool; only cold experts are staged (into spare pool slots), so the per-layer 113 MB gather of
dg-masked becomes ~0.05 cold experts/layer/step at the seed map plus a small scale re-layout.

    st = StaticStep(model, pool, max_pos=4095, moe_impl="dg-inplace")   # everything else as runtime/static_step.py

Spec: docs/HANDOFF-T9A2-INPLACE-MOE.md; design docs/T8-T9A-DESIGN.md sections 0.6, 3.2, 3.3; the masked call and its
numerics docs/T9A-DEEPGEMM-ABI-20261106.md; probe bench/dg_contiguous_spike.py. `DGInplaceMoE` has StaticMoE's
interface and DGMaskedMoE's glue (act_quant into row 0 of each A block, SwiGLU, w2 requant, slot-order combine, shared
expert -- the same methods, fused or not). What changes, per layer:

  pool    every layer's per-projection WEIGHT pool grows from [H, N, K/2] to [H+S, N, K/2], S = topk spare slots
          (ExpertPool.add_spare_slots, at StaticStep build; hot slots and the e8m0 scale pools are untouched, so the
          eager prefill, `ref` and `dg-masked` read exactly what they read before).
  scales  DeepGEMM's sm100 kernels consume weight scales as int32 words of 4 packed UE8M0 codes, MN-major
          ([G, N, K/128] with strides (N*K/128, 1, N)); an fp32 SF tensor is converted to that by the library ON EVERY
          CALL (transform_sf_into_required_layout), which over a whole pool is 1.3 GB/layer: impossible. The packed
          int32 word for (n, 4c..4c+3) is exactly the checkpoint's 4 e8m0 bytes read as a little-endian int32
          (DeepGEMM's pack_ue8m0_to_int is `(fp32_bits >> 23)` = the e8m0 code), so packing is an int32 transpose
          of the pool's own bytes -- no value conversion at all, hence the same numbers dg-masked's fp32 path feeds
          the kernel. Storage, `sf_store` ($DSV41_DG_SF_STORE):
            "step" (default): ONE shared packed SF tensor per projection, [max(H)+S, N, K/128] (0.31 GiB on the real
                     model); the layer's staging launch writes the k selected experts' scales into their groups each
                     step (6.6 MB read + write per layer). The kernel reads SF only for groups named in m_indices, so
                     stale groups from earlier layers are never read.
            "resident": a packed copy of every layer's hot scales built once at construction (12.4 GiB on the real
                     model; only cold experts' scales are written per step).
          fp32 resident scales would be 49.6 GiB (4 B/scale) and do not fit (hbm_plan()).
  stage   ONE launch per layer (runtime/static_kernels.py t9a2_stage_inplace; contract = stage_inplace_emulated below):
          group[j] = loc slot of ids[j] if hot else H + j; cold experts' weight rows (HBM or UVA host address, the same
          pointer tables as dg-masked) are copied into pool slot H + j; scales are packed into the SF tensor (all k in
          "step", cold only in "resident"); m_indices[j*ALIGN] = group[j] (every other row stays -1); ids_step.
          M1-T10-FIX2 (the 6-row step, runtime/moe_rows.py DGInplaceMoERows): `sidx` = a compact cold index per slot
          (cold slot -> H + c, c < spare_slots, default 8) and a device overflow counter instead of H + j, so the
          pool needs S spare slots, not one per A block.
  gemm    3 x m_grouped_fp8_fp4_gemm_nt_contiguous((xq [k*ALIGN, K], xs fp32), (W [H+S, N, K/2] int8, SF int32 packed),
          d [k*ALIGN, N], m_indices, recipe_a=(1,32), recipe_b=(1,32)). A block j = rows [j*ALIGN, (j+1)*ALIGN): row 0
          is the token (the masked buffers' layout with max_m = ALIGN), rows 1.. are zero and tagged -1; output row
          j*ALIGN = expert ids[j] = slot j of dg-masked, so the glue and combine are unchanged. The scheduler takes a
          block's group from m_indices[m_block * BLOCK_M] (scheduler/gemm.cuh:173) and skips -1 blocks (331), so
          BLOCK_M must not exceed ALIGN: set_mk_alignment_for_contiguous_layout(ALIGN) at build (T8-T9A risk 3).

Numerics: the same block-scaled UMMA kernel family as dg-masked on the same bytes (fp8 activations, int8 e2m1 weights,
scales identical); expected bit-identical to dg-masked, at worst its 1-ulp class -- the probe and the GPU test measure.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from runtime.dg_moe import (ACT_BLOCK, PROJS, RECIPE_A, RECIPE_B, WEIGHT_BLOCK, DGBuffers, DGMaskedMoE,
                            import_deep_gemm)

MK_ALIGN = 128                     # contiguous-layout rows per group block (= BLOCK_M cap); $DSV41_DG_ALIGN
ALIGN_ENV = "DSV41_DG_ALIGN"
SF_STORES = ("step", "resident", "pool")
SF_STORE_ENV = "DSV41_DG_SF_STORE"
SF_PACK = 4                        # e8m0 codes per int32 word
ORDER6 = (("w1", "weight"), ("w3", "weight"), ("w2", "weight"), ("w1", "scale"), ("w3", "scale"), ("w2", "scale"))
GiB = 2 ** 30
# M1-T10-MOE (the R-row dg-inplace step): how the routed GEMM is called, and where the hot experts' packed SF live
ROWS_LAYOUTS = ("contiguous", "psum", "masked")      # = runtime/moe_rows.LAYOUT_IDS
ROWS_GEMMS = ("contiguous", "contiguous64", "contiguous32", "contiguous16", "psum", "masked")
ROWS_GEMM_ENV = "DSV41_ROWS_MOE_GEMM"                # first_light --rows-moe-gemm
ROWS_SF_STORE_ENV = "DSV41_ROWS_SF_STORE"            # first_light --rows-sf-store
ROWS_SF_STORE_DEFAULT = "pool"
PSUM_ALIGN_ENV = "DSV41_ROWS_PSUM_ALIGN"             # psum: group-start alignment (= the kernel's BLOCK_M)
EXPECTED_M_ENV = "DSV41_ROWS_EXPECTED_M"             # psum / masked: the heuristic's expected rows per group
SF_POOL_HEADROOM_ENV = "DSV41_SF_POOL_HEADROOM_GIB"  # pool / resident SF: keep this much HBM free, else "step"
SF_POOL_HEADROOM_GIB = 6.0      # pool SF = hbm_plan's sf_packed_resident_B (12.8 GiB real model, S 16); the 6-row
                                # receipts had 17.1 (S 16) / 22.7 (S 8) GiB free after build: capture needs margin
LAYOUT_CHECK_ENV = "DSV41_ROWS_LAYOUT_CHECK"         # "0": skip the build-time layout self-check
MASKED_MAX_M = 128                                   # masked: max_m >= BLOCK_M (ABI memo)
ROWS_GEMM_DEFAULT = "contiguous16"   # the R-row call shape (probe 2026-11-07, docs/dg-rows-spike-20261107.log)
# T9a-3 (docs/HANDOFF-T9A3-DG-ROWMATCH.md): the 1-row step's routed GEMM in the R-row step's call shape, so that a
# token's expert outputs are computed by the same kernel instance at 1 and at R rows (first_light / headline
# --moe-rowmatch). "on": the R-row call's ALIGN and its A-block count R*k (the same M, N, K, G: the same kernel
# config), the token's k blocks first, the rest tagged -1; "align": the R-row ALIGN with k blocks only (M = k*ALIGN);
# "off": T9a-2's call (ALIGN 128, k blocks). Any mode but off is checked at build per row against the R-row call on
# layer 0's real pool bytes (DGInplaceBuffers.check_rowmatch); a mismatch falls back to off (the record says why).
ROWMATCH_ENV = "DSV41_MOE_ROWMATCH"
ROWMATCH_MODES = ("on", "align", "off")
ROWMATCH_DEFAULT = "on"
ROWMATCH_ROWS_ENV = "DSV41_MOE_ROWMATCH_ROWS"        # R of the step the 1-row path matches
ROWMATCH_ROWS_DEFAULT = 6                            # the DSpark verify step (k = 5 drafts + the bonus row)


def rows_gemm_spec(req: Optional[str] = None) -> str:
    """The R-row dg-inplace call shape in force: the build's request, else $DSV41_ROWS_MOE_GEMM, else the default."""
    return req or os.environ.get(ROWS_GEMM_ENV) or ROWS_GEMM_DEFAULT


def rowmatch_mode(mode: Optional[str] = None) -> str:
    v = mode or os.environ.get(ROWMATCH_ENV) or ROWMATCH_DEFAULT
    if v not in ROWMATCH_MODES:
        raise ValueError(f"moe rowmatch {v!r} not in {ROWMATCH_MODES} (${ROWMATCH_ENV})")
    return v


def rowmatch_plan(topk: int, mode: Optional[str] = None, rows: Optional[int] = None,
                  rows_spec: Optional[str] = None) -> dict:
    """T9a-3: the 1-row dg-inplace build's call shape -> {"mode", "rows", "rows_gemm", "align", "blocks", "on", "why"}
    (align / blocks None = T9a-2's ALIGN 128 x topk). Only a contiguous R-row call shape has a 1-row twin."""
    m = rowmatch_mode(mode)
    R_ = int(rows or os.environ.get(ROWMATCH_ROWS_ENV) or ROWMATCH_ROWS_DEFAULT)
    spec = rows_gemm_spec(rows_spec)
    out = {"mode": m, "rows": R_, "rows_gemm": spec, "align": None, "blocks": None, "on": False, "why": None}
    if m == "off":
        out["why"] = "off (requested)"
        return out
    layout, al = parse_rows_gemm(spec)
    if layout != "contiguous":
        out["why"] = f"the R-row call shape {spec!r} is not contiguous: no 1-row twin, T9a-2's call kept"
        return out
    al = check_align(al if al is not None else int(os.environ.get(ALIGN_ENV, MK_ALIGN)))
    if not 1 <= R_ <= al:
        out["why"] = f"rows {R_} outside [1, ALIGN {al}]"
        return out
    out.update(align=al, blocks=R_ * int(topk) if m == "on" else int(topk), on=True)
    return out


@contextlib.contextmanager
def mk_alignment(dg, align: int):
    """DeepGEMM's process-global contiguous BLOCK_M cap := align for the calls inside; restored after."""
    f = getattr(dg, "set_mk_alignment_for_contiguous_layout", None)
    g = getattr(dg, "get_mk_alignment_for_contiguous_layout", None)
    prev = g() if g is not None else None
    if f is not None:
        f(int(align))
    try:
        yield
    finally:
        if f is not None and prev is not None and int(prev) != int(align):
            f(int(prev))


def contiguous_blocks_layout(blocks: Sequence[Sequence[Tuple[object, int]]], align: int, n_blocks: int
                             ) -> Tuple[Dict[object, int], List[int]]:
    """T9a-3: blocks[b] = [(key, group), ...] -> ({key: A row b*align + t}, m_indices [n_blocks*align]: the group on
    every entry's row, -1 elsewhere). Blocks >= len(blocks) are unused (all -1)."""
    if len(blocks) > n_blocks:
        raise ValueError(f"{len(blocks)} blocks > {n_blocks}")
    mi = [-1] * (int(n_blocks) * int(align))
    rows: Dict[object, int] = {}
    for b, ents in enumerate(blocks):
        if len(ents) > align:
            raise ValueError(f"block {b}: {len(ents)} rows > alignment {align}")
        for t, (key, grp) in enumerate(ents):
            if key in rows:
                raise ValueError(f"key {key!r} twice")
            rows[key] = b * int(align) + t
            mi[b * int(align) + t] = int(grp)
    return rows, mi


def one_row_blocks(r: int, experts: Sequence[int], gmap: Dict[int, int]) -> List[List[Tuple[object, int]]]:
    """The 1-row step's blocks for row r alone: one block per routed expert, ascending id, the token in row 0."""
    return [[((int(r), int(e)), int(gmap[int(e)]))] for e in sorted(int(v) for v in experts)]


def r_row_blocks(row_experts: Sequence[Sequence[int]], gmap: Dict[int, int]) -> List[List[Tuple[object, int]]]:
    """The R-row step's blocks (runtime/moe_rows.py dedup): one block per unique expert, ascending id; in it the rows
    routed to it, ascending (pair (r, e) at row `rank`)."""
    uniq = sorted({int(e) for r in row_experts for e in r})
    return [[((r, e), int(gmap[e])) for r, ex in enumerate(row_experts) if e in {int(v) for v in ex}] for e in uniq]


def pack_act_sf_from_f32(xs: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two activation scales [M, KS] -> the packed MN-major int32 SF DeepGEMM takes (packed_act_sf
    layout; code = fp32 exponent bits, DeepGEMM's pack_ue8m0_to_int)."""
    M, KS = xs.shape
    codes = ((xs.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    out = packed_act_sf(M, KS, xs.device)
    out.copy_(codes.view(torch.int32))
    return out


def run_contiguous(dg, W: torch.Tensor, SF: torch.Tensor, blocks, acts: Dict[object, Tuple[torch.Tensor, torch.Tensor]],
                   align: int, n_blocks: int, *, packed_a: bool = False) -> Dict[object, torch.Tensor]:
    """T9a-3: ONE m_grouped_fp8_fp4_gemm_nt_contiguous call on temporary buffers in the call shape (align, n_blocks)
    with the entries of `blocks` (contiguous_blocks_layout); acts[key] = (fp8 [K], fp32 [K/32]) on W's device.
    -> {key: bf16 [N] output row} (clones)."""
    dev = W.device
    rows, mi = contiguous_blocks_layout(blocks, align, n_blocks)
    keys = list(rows)
    K = int(acts[keys[0]][0].numel())
    N = int(W.size(1))
    M = int(n_blocks) * int(align)
    A = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device=dev)
    S = torch.ones(M, K // ACT_BLOCK, dtype=torch.float32, device=dev)
    idx = torch.tensor([rows[k_] for k_ in keys], dtype=torch.long, device="cpu").to(dev)
    A.view(torch.uint8).index_copy_(0, idx, torch.stack([acts[k_][0].reshape(-1).view(torch.uint8) for k_ in keys]))
    S.index_copy_(0, idx, torch.stack([acts[k_][1].reshape(-1).float() for k_ in keys]))
    mit = torch.tensor(mi, dtype=torch.int32, device="cpu").to(dev)
    D = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    sfa = pack_act_sf_from_f32(S) if packed_a else S
    with mk_alignment(dg, align):
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous((A, sfa), (W, SF), D, mit, recipe_a=RECIPE_A, recipe_b=RECIPE_B)
    out = D.index_select(0, idx)
    return {k_: out[i].clone() for i, k_ in enumerate(keys)}


def parse_rows_gemm(spec: Optional[str]) -> Tuple[str, Optional[int]]:
    """--rows-moe-gemm -> (layout, contiguous alignment or None = $DSV41_DG_ALIGN / 128):
    "contiguous" (today: R*k blocks of ALIGN rows, m_indices tags), "contiguousNN" (the same at ALIGN = NN, the
    kernel's BLOCK_M cap: fewer padded MMA rows per block), "psum" (contiguous with use_psum_layout over the WHOLE
    pool's groups: empty groups cost nothing, the swap-AB kernels' effective M is the real rows aligned to 16),
    "masked" (masked grouped GEMM with the pool itself as B, G = H+S groups, masked_m[g] = rows of group g)."""
    s = spec or "contiguous"
    if s not in ROWS_GEMMS:
        raise ValueError(f"rows moe gemm {s!r} not in {ROWS_GEMMS}")
    if s.startswith("contiguous"):
        return "contiguous", (int(s[len("contiguous"):]) if len(s) > len("contiguous") else None)
    return s, None


# ------------------------------------------------------------------------------------------------ packed SF layout
def sf_words(ks: int) -> int:
    """int32 words per scale row of `ks` per-32 e8m0 codes (K/32 -> K/128)."""
    if ks % SF_PACK:
        raise ValueError(f"{ks} scales per row is not a multiple of {SF_PACK} (K must be a multiple of 128)")
    return ks // SF_PACK


def _u8(s: torch.Tensor) -> torch.Tensor:
    return s if s.dtype == torch.uint8 else s.view(torch.uint8)


def packed_sf_zeros(G: int, N: int, C: int, device) -> torch.Tensor:
    """int32 [G, N, C] MN-major view (strides (C*N, 1, N)) over zeroed storage [G][C][N] -- the layout DeepGEMM's
    get_mn_major_tma_aligned_packed_ue8m0_tensor returns for N % 4 == 0 (its TMA alignment for 4-byte elements).
    Code 0 = 2^-127: finite, so a group never staged reads as tiny, never NaN."""
    if N % 4:
        raise ValueError(f"N={N}: MN-major packed SF needs N % 4 == 0 (TMA alignment); not supported here")
    return torch.zeros(G, C, N, dtype=torch.int32, device=device).transpose(1, 2)


def pack_words(s: torch.Tensor) -> torch.Tensor:
    """e8m0 [..., N, KS] (uint8 or float8_e8m0fnu storage, last dim contiguous) -> int32 [..., N, KS/4]: the same
    bytes, 4 consecutive K-scales per little-endian word (no value change)."""
    u = _u8(s)
    if u.stride(-1) != 1:
        u = u.contiguous()
    sf_words(u.size(-1))
    return u.view(torch.int32)


def pack_sf(s: torch.Tensor) -> torch.Tensor:
    """e8m0 [..., N, KS] -> new int32 [..., N, KS/4] MN-major tensor (pack_words, then MN-major storage)."""
    w = pack_words(s)
    return w.transpose(-1, -2).contiguous().transpose(-1, -2)


def unpack_sf(packed: torch.Tensor) -> torch.Tensor:
    """int32 [..., N, C] (any strides) -> uint8 e8m0 codes [..., N, 4C]."""
    p = packed.contiguous()
    if p.dim() and p.stride(-1) != 1:    # C == 1: contiguous() keeps the MN-major stride of the size-1 last dim
        p = torch.empty(p.shape, dtype=p.dtype, device=p.device).copy_(p)
    return p.view(torch.uint8)


def ref_pack_from_f32(sf: torch.Tensor) -> torch.Tensor:
    """DeepGEMM's own packing semantics in torch, for the tests: deep_gemm/utils/math.py pack_ue8m0_to_int
    (`(fp32_bits >> 23).to(uint8).view(int)`) on fp32 power-of-two scales [..., N, KS], then MN-major."""
    x = sf.float().contiguous().view(torch.int32)
    w = (x >> 23).to(torch.uint8).view(torch.int32)
    return w.transpose(-1, -2).contiguous().transpose(-1, -2)


def is_mn_major_packed(sf: torch.Tensor, G: int, N: int, C: int) -> bool:
    return (sf.dtype == torch.int32 and tuple(sf.shape) == (G, N, C) and sf.stride(1) == 1 and sf.stride(2) == N
            and (G == 1 or sf.stride(0) == C * N))


def packed_act_sf(M: int, KS: int, device, fill_code: int = 127) -> torch.Tensor:
    """T9b-4 lever 4: DeepGEMM's packed activation SF for A [M, K]: int32 [M, KS/4] MN-major (strides (1, Ma), Ma =
    M rounded up to 4 = get_mn_major_tma_aligned_packed_ue8m0_tensor's TMA alignment for 4-byte words) over storage
    [KS/4][Ma], every byte = `fill_code` (127 = 2^0, the fp32 path's padding-row scale 1.0)."""
    C = sf_words(KS)
    Ma = -(-int(M) // 4) * 4
    word = int.from_bytes(bytes([fill_code] * 4), "little", signed=True)
    base = torch.full((C, Ma), word, dtype=torch.int32, device=device)
    return base.t()[: int(M)]


class PackedRows:
    """T9b-4 lever 4: `rows` output rows of an act_quant writer (row r = A row r*`align`, the head of block r) as
    bytes of a packed MN-major activation SF `sf` (packed_act_sf). The code of 32-block b of row r lives at byte
    r*row_stride + (b // 4)*col_stride + b % 4 of the storage (little-endian: byte i of word c = K-block 4c + i,
    DeepGEMM's pack_ue8m0_to_int). The fused writers (runtime/fused_glue*.py) store there instead of fp32 scales."""
    packed = True

    def __init__(self, sf: torch.Tensor, align: int, rows: int):
        M, C = sf.shape
        if sf.dtype != torch.int32 or sf.stride(0) != 1 or sf.stride(1) < M:
            raise ValueError(f"PackedRows: sf {tuple(sf.shape)} {sf.dtype} strides {sf.stride()} is not MN-major int32")
        if (rows - 1) * align >= M:
            raise ValueError(f"PackedRows: {rows} rows x align {align} > M = {M}")
        self.sf, self.align, self.rows, self.kb = sf, int(align), int(rows), 4 * int(C)
        self.u8 = sf.as_strided((C * sf.stride(1),), (1,), sf.storage_offset()).view(torch.uint8)
        self.row_stride = 4 * self.align
        self.col_stride = 4 * int(sf.stride(1))

    def _index(self, row: int) -> torch.Tensor:
        b = torch.arange(self.kb, device=self.u8.device)
        return row * self.row_stride + (b // 4) * self.col_stride + b % 4

    def store_codes_torch(self, row: int, codes: torch.Tensor) -> None:
        self.u8[self._index(int(row))] = codes.reshape(-1).to(torch.uint8).to(self.u8.device)

    def codes(self) -> torch.Tensor:
        """uint8 [rows, KS]: the codes the writers stored (tests, the self-check)."""
        return torch.stack([self.u8[self._index(r)] for r in range(self.rows)])

    def codes_at(self, rows: torch.Tensor) -> torch.Tensor:
        """M1-T10-MOE: uint8 [len(rows), KS], the codes of output rows `rows` (one gather; the R-row self-check)."""
        b = torch.arange(self.kb, device=self.u8.device)
        off = rows.to(self.u8.device).view(-1, 1) * self.row_stride + ((b // 4) * self.col_stride + b % 4).view(1, -1)
        return self.u8[off]


# ------------------------------------------------------------------------------------------------ index math
def check_align(align: int) -> int:
    a = int(align)
    if a < 1 or a & (a - 1) or a > 256:
        raise ValueError(f"contiguous alignment {a} must be a power of two in [1, 256]")
    return a


def group_ids(ids: torch.Tensor, gtab: torch.Tensor, H: int) -> torch.Tensor:
    """group of sorted slot j = hot pool slot gtab[ids[j]] (>= 0), or the spare slot H + j for a cold expert (-1)."""
    g = gtab[ids].long()
    j = torch.arange(ids.numel(), dtype=torch.long, device=ids.device)
    return torch.where(g < 0, H + j, g)


def m_indices_for(groups: torch.Tensor, align: int) -> torch.Tensor:
    """int32 [k*align]: -1 everywhere except row j*align = groups[j] (one token row per A block)."""
    k = groups.numel()
    mi = torch.full((k * align,), -1, dtype=torch.int32, device=groups.device)
    mi.view(k, align)[:, 0] = groups.to(torch.int32)
    return mi


def contiguous_rows(row_experts: Sequence[Sequence[int]], align: int) -> dict:
    """Multi-row (M > 1) contiguous layout, for the probe: row r routed to experts row_experts[r]. Unique experts u
    (ascending) get A blocks [u_i*align, u_i*align + cnt_u); rows within a block in ascending r. Returns
    {"experts": unique ids, "rows": {(r, e): A row}, "m_rows": total A rows, "counts": per expert, "tag": per-row
    expert id or -1 (the caller maps ids to groups)}. Needs every count <= align."""
    uniq = sorted({int(e) for r in row_experts for e in r})
    rows, counts = {}, {}
    for i, e in enumerate(uniq):
        rs = [r for r, ex in enumerate(row_experts) if e in [int(v) for v in ex]]
        if len(rs) > align:
            raise ValueError(f"expert {e}: {len(rs)} rows > alignment {align}")
        counts[e] = len(rs)
        for t, r in enumerate(rs):
            rows[(r, e)] = i * align + t
    tag = [-1] * (len(uniq) * align)
    for (r, e), a in rows.items():
        tag[a] = e
    return {"experts": uniq, "rows": rows, "m_rows": len(uniq) * align, "counts": counts, "tag": tag}


def bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|ulp distance| between bf16 tensors (sign-magnitude bits -> a monotone integer line; +0 == -0)."""
    def key(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (key(a) - key(b)).abs()


def compare_bf16(got: torch.Tensor, want: torch.Tensor) -> dict:
    """bitwise comparison summary: mismatch fraction, max ulp, max |delta|, equal."""
    u = bf16_ulp(got, want)
    fin = torch.isfinite(got.float()) & torch.isfinite(want.float())
    return {"equal": bool(torch.equal(got.view(torch.int16), want.view(torch.int16))),
            "mismatch_frac": float((u != 0).float().mean()), "max_ulp": int(u.max()) if u.numel() else 0,
            "n_ulp_gt1": int((u > 1).sum()), "max_abs": float((got.float() - want.float()).abs()[fin].max())
            if bool(fin.any()) else float("nan"), "n": int(u.numel()), "nonfinite": int((~fin).sum())}


# ------------------------------------------------------------------------------------------------ HBM budget
def hbm_plan(n_layers: int, hot: Sequence[int], S: int, shapes: Dict[str, Tuple[int, int]],
             sshapes: Dict[str, Tuple[int, int]]) -> dict:
    """Bytes each scale-storage option and the spare weight slots cost (real model: 40 layers x 295 hot, S = 6,
    w1/w3 [2304, 2560] + [2304, 160], w2 [5120, 1152] + [5120, 72]). `hot` = H per layer."""
    w_row = {p: shapes[p][0] * shapes[p][1] for p in PROJS}
    s_row = {p: sshapes[p][0] * sshapes[p][1] for p in PROJS}
    G = [h + S for h in hot]
    spare = n_layers * S * sum(w_row.values())
    packed_res = sum(G) * sum(s_row.values())
    f32_res = 4 * packed_res
    step = max(G) * sum(s_row.values())
    per_layer_step_bytes = 2 * S * sum(s_row.values())  # read e8m0 + write packed, k = S selected experts
    out = {"n_layers": n_layers, "S": S, "G_max": max(G), "spare_weight_slots_B": spare,
           "sf_step_shared_B": step, "sf_packed_resident_B": packed_res, "sf_f32_resident_B": f32_res,
           "e8m0_pool_scales_B": sum(hot) * sum(s_row.values()), "step_scale_traffic_per_layer_B": per_layer_step_bytes,
           "total_step_B": spare + step, "total_resident_B": spare + packed_res}
    out.update({k[:-2] + "_GiB": round(v / GiB, 3) for k, v in list(out.items()) if k.endswith("_B")})
    return out


# ------------------------------------------------------------------------------------------------ staging contract
def stage_inplace_emulated(resolve, ptab6, ids, gtab, dtab, dsts, mi, H: int, align: int, row_w: int, sf_dims,
                           all_scales: bool, ids_out=None, *, sidx=None, spare: Optional[int] = None,
                           ovf=None) -> None:
    """The contract of runtime/static_kernels.py `stage_inplace` (t9a2_stage_inplace) in torch, through
    `resolve(address) -> uint8 tensor of the source row` (the Mac tests' registry; the GPU test's oracle):
    spare index c[j] = j (sidx None: the 1-row step) or sidx[j] (M1-T10-FIX2: the compact cold index, -1 = skip);
    group[j] = gtab[ids[j]] if >= 0 else H + c[j]; a cold slot is STAGED iff 0 <= c[j] < spare (spare: default k);
    mi[j*align] = group[j] for hot and staged slots, -1 for the others; staged j: weight rows of w1/w3/w2 -> slot
    group[j] of dsts[0..2]; scales (hot and staged j if all_scales, else staged j) -> group[j] of the packed SF
    dsts[3..5]; ids_out[j] = ids[j]; sidx given: ovf[0] += the number of cold slots with c[j] >= spare (overflow:
    nothing of them is written). Nothing else is written."""
    assert [int(a) for a in dtab.tolist()] == [t.data_ptr() for t in dsts] and ptab6.shape[0] == 6
    k = ids.numel()
    if (sidx is None) != (ovf is None) or (sidx is not None and spare is None):
        raise ValueError("stage_inplace: sidx, spare and ovf go together")
    S = k if spare is None else int(spare)
    idl = [int(v) for v in ids.tolist()]
    cl = list(range(k)) if sidx is None else [int(v) for v in sidx.tolist()]
    n_over = 0
    for j, e in enumerate(idl):
        g = int(gtab[e])
        cold = g < 0
        c = cl[j]
        staged = cold and 0 <= c < S
        if cold and c >= S:
            n_over += 1
        grp = H + c if cold else g
        mi[j * align] = grp if (not cold or staged) else -1
        for p in range(3):
            if staged:
                src = resolve(int(ptab6[p, e]))
                dsts[p][grp].view(torch.uint8).view(-1).copy_(src.view(-1)[:row_w])
            if staged or (all_scales and not cold):
                N, C = sf_dims[1] if p == 2 else sf_dims[0]
                src = resolve(int(ptab6[3 + p, e]))
                dsts[3 + p][grp].copy_(src.view(-1)[:N * C * SF_PACK].view(N, C * SF_PACK).view(torch.int32))
    if ids_out is not None:
        ids_out.copy_(ids)
    if ovf is not None and n_over:
        ovf.add_(n_over)
    assert k * align == mi.numel()


def stage_rows_emulated(resolve, ptab6, uniq, gtab, dtab, dsts, sidx, wl, H: int, row_w: int, sf_dims,
                        n_work: int) -> None:
    """M1-T10-MOE: the contract of runtime/static_kernels.py `stage_rows` (m1_stage_rows) in torch: for j <
    n_work with u = wl[j] >= 0, e = uniq[u], g = gtab[e]: cold (g < 0) -> its weight rows of w1/w3/w2 into slot
    H + sidx[u] of dsts[0..2] and its scales into that group of the packed SF dsts[3..5]; hot -> its scales into
    group g. Nothing else is written (m_indices, overflow and stage ids are the dedup's)."""
    assert [int(a) for a in dtab.tolist()] == [t.data_ptr() for t in dsts] and ptab6.shape[0] == 6
    wll, ul, sl = [int(v) for v in wl.tolist()], [int(v) for v in uniq.tolist()], [int(v) for v in sidx.tolist()]
    for j in range(int(n_work)):
        u = wll[j]
        if u < 0:
            continue
        e = ul[u]
        g = int(gtab[e])
        cold = g < 0
        grp = H + sl[u] if cold else g
        for p in range(3):
            if cold:
                dsts[p][grp].view(torch.uint8).view(-1).copy_(resolve(int(ptab6[p, e])).view(-1)[:row_w])
            N, C = sf_dims[1] if p == 2 else sf_dims[0]
            src = resolve(int(ptab6[3 + p, e]))
            dsts[3 + p][grp].copy_(src.view(-1)[:N * C * SF_PACK].view(N, C * SF_PACK).view(torch.int32))


# ------------------------------------------------------------------------------------------------ pool SF store
def pool_sf_key(pool) -> tuple:
    """What a pool-owned packed SF copy is valid for: the storage generation, the spare count and the hot sets."""
    return (getattr(pool, "generation", None), int(pool.spare),
            tuple(tuple(int(e) for e in pool.hot[l]) for l in range(pool.n_layers)),
            tuple(pool.hbm_full[l]["w1"].data_ptr() for l in range(pool.n_layers)))


def pool_sf_bytes(pool, N: Dict[str, int], C: Dict[str, int]) -> int:
    """Bytes of a whole-pool packed SF copy (every layer's H + S groups, three projections)."""
    per = sum(N[p] * C[p] * 4 for p in PROJS)
    return sum(int(pool.hbm_full[l]["w1"].size(0)) * per for l in range(pool.n_layers))


def pool_packed_sf(pool, layer: int, N: Dict[str, int], C: Dict[str, int], device) -> Dict[str, torch.Tensor]:
    """M1-T10-MOE sf_store "pool": ONE packed MN-major copy of every layer's hot scales, owned by the pool and shared
    by every dg-inplace build on it (the 1-row and the R-row steps alike), built per layer on first use; the spare
    groups start zero (code 0 = 2^-127, finite) and are written by the staging launches. Keyed by pool_sf_key: a
    storage move (add_spare_slots) or a different hot set rebuilds it (the old copy goes with the stale builds)."""
    key = pool_sf_key(pool)
    cache = getattr(pool, "_dg_packed_sf", None)
    if cache is None or cache["key"] != key:
        cache = pool._dg_packed_sf = {"key": key, "layers": {}}
    if layer not in cache["layers"]:
        H = len(pool.hot[layer])
        G = int(pool.hbm_full[layer]["w1"].size(0))
        sf = {}
        with torch.inference_mode(False), torch.no_grad():
            for p in PROJS:
                t = packed_sf_zeros(G, N[p], C[p], device)
                if H:
                    t[:H].copy_(pack_words(pool.hbm[layer][p][1]))
                sf[p] = t
        cache["layers"][layer] = sf
    return cache["layers"][layer]


# ------------------------------------------------------------------------------------------------ static buffers
class DGInplaceBuffers(DGBuffers):
    """dg-masked's buffers with max_m = ALIGN (the activation buffers ARE the contiguous A: [k, ALIGN, K] viewed as
    [k*ALIGN, K]), no fp32 weight-scale staging, plus m_indices, the packed SF storage and per-layer tables.
    M1-T10-MOE (the R-row step, runtime/moe_rows.py DGInplaceMoERows): `layout` -- "contiguous" (the above, any
    power-of-two ALIGN >= rows), "psum" (the same A [k*ALIGN, K], groups by the use_psum_layout prefix sums over
    the pool's groups in `gl`), "masked" (A [G_max + 1, 128, K]: block g = pool group g, masked_m in `gl`, block
    G_max = the scratch block of overflowing pairs) -- and sf_store "pool" (pool_packed_sf)."""

    def __init__(self, owner, shapes, sshapes, *, pool, swiglu_limit: float, deterministic: bool = True, dg=None,
                 align: Optional[int] = None, sf_store: Optional[str] = None, slots: Optional[int] = None,
                 spare_slots: Optional[int] = None, layout: str = "contiguous", expected_m: Optional[int] = None,
                 gemm_blocks: Optional[int] = None):
        """slots: A blocks (default topk; M1-T10-FIX's R-row MoE, runtime/moe_rows.py DGInplaceMoERows, uses R*topk:
        block u holds slot u's count <= R rows). spare_slots (M1-T10-FIX2): spare pool slots per layer the cold
        experts of one step may use (default = slots: the 1-row contract, spare j for slot j; the R-row MoE passes
        its compact-index budget S, default 8, and counts overflows). layout / expected_m: M1-T10-MOE (R rows).
        gemm_blocks (T9a-3, contiguous only): A blocks the GEMM is CALLED with (>= slots; default slots). The
        buffers the glue / staging / combine see (xq, xs, d, mi, xsp: `slots` blocks) are prefix views of
        gemm_blocks-block storage whose extra blocks stay zero / one / -1 forever, so the call is the R-row step's
        (M = gemm_blocks * ALIGN) with only the token's blocks tagged."""
        if pool is None:
            raise ValueError("dg-inplace needs the ExpertPool (it reads the hot pool in place)")
        if layout not in ROWS_LAYOUTS:
            raise ValueError(f"layout {layout!r} not in {ROWS_LAYOUTS}")
        rows = int(getattr(owner, "rows", 1))
        if layout != "contiguous" and rows == 1:
            raise ValueError(f"layout {layout!r} is the R-row step's (rows > 1)")
        dg = dg if dg is not None else import_deep_gemm()
        pairs = int(slots or owner.topk)
        rem = int(expected_m or os.environ.get(EXPECTED_M_ENV, 0) or rows)   # (DGBuffers' expected_m is masked's 8)
        hot = [len(pool.hot[l]) for l in range(pool.n_layers)]
        G_max = max(hot) + pool.spare
        if layout == "masked":
            align = check_align(align if align is not None else MK_ALIGN)
            blocks, brows = G_max + 1, MASKED_MAX_M
        else:
            if align is None and layout == "psum":
                align = self._psum_align(dg, rem)
            align = check_align(align if align is not None else int(os.environ.get(ALIGN_ENV, MK_ALIGN)))
            if align < rows:   # one block of ALIGN rows per used slot / group holds its <= R rows
                raise ValueError(f"{layout} alignment {align} < {rows} rows per block")
            blocks, brows = pairs, align
        gb = blocks if gemm_blocks is None else int(gemm_blocks)
        if gb != blocks and (layout != "contiguous" or gb < blocks):
            raise ValueError(f"gemm_blocks {gb}: contiguous layout only, >= the {blocks} A blocks")
        super().__init__(owner, shapes, sshapes, swiglu_limit=swiglu_limit, deterministic=deterministic, dg=dg,
                         rows=brows, sf32=False, slots=gb)
        # T9a-3: the GEMM-call tensors (gemm_blocks blocks); the step's tensors are their first `blocks` blocks
        self.gemm_blocks = gb
        self._g = {"xq": self.xq, "xs": self.xs, "d": self.d}
        if gb != blocks:
            self.k = blocks
            self.xq = {w: t[:blocks] for w, t in self._g["xq"].items()}
            self.xs = {w: t[:blocks] for w, t in self._g["xs"].items()}
            self.d = {p: t[:blocks] for p, t in self._g["d"].items()}
            self.masked_m = self.masked_m[:blocks]
        self.rowmatch: Optional[dict] = None    # T9a-3: set by build_one_row (the 1-row step's plan + build check)
        self.layout, self.align, self.pairs, self.G_max = layout, align, pairs, G_max
        self.rows_expected_m = rem              # masked layout: the heuristic's rows per group (handoff: R)
        self.block_rows = brows                 # A rows per block (the dedup's block_rows)
        self.gscr = G_max                       # masked: the scratch block of overflowing pairs
        self.sf_store = sf_store or os.environ.get(SF_STORE_ENV, "step")
        if self.sf_store not in SF_STORES:
            raise ValueError(f"sf_store {self.sf_store!r} not in {SF_STORES} (${SF_STORE_ENV})")
        self.spare_limit = pairs if spare_slots is None else int(spare_slots)
        if not 1 <= self.spare_limit <= pairs:
            raise ValueError(f"spare_slots {self.spare_limit} outside [1, {pairs}] (A blocks per layer)")
        if pool.spare < self.spare_limit:
            raise ValueError(f"pool has {pool.spare} spare slots per layer, dg-inplace needs {self.spare_limit} "
                             f"(ExpertPool.add_spare_slots, done by StaticStep before this)")
        self.pool = pool
        self.fn = getattr(self.dg, "m_grouped_fp8_fp4_gemm_nt_contiguous")
        self.fn_masked = getattr(self.dg, "m_grouped_fp8_fp4_gemm_nt_masked", None)
        self.alignment = self._set_alignment(align)
        d, k = self.dev, self.k
        self._g["mi"] = torch.full((gb * align,), -1, dtype=torch.int32, device=d) if layout == "contiguous" else None
        self.mi = self._g["mi"][: k * align] if layout == "contiguous" else None
        self.gl = torch.zeros(G_max, dtype=torch.int32, device=d) if layout != "contiguous" else None
        # T9b-4 lever 4: packed activation SF (A [k*ALIGN, K]) for the w1/w3 input (1) and the w2 input (2); used
        # instead of the fp32 xs only once runtime/fused_glue.py's self-check turned `sf_pack` on (it owns every
        # writer of row 0 of each block then). M1-T10-MOE: `xsp_all` = the same SF addressed by ANY A row (the R-row
        # writers scatter to the pairs' rows, dedup arow); the masked layout has no packed SF (fp32 only).
        self.sf_pack = False
        self.sf_pack_why: Optional[str] = "not enabled (needs glue_impl fused: dg_act + swiglu_routed + self-check)"
        self.xsp, self.xsp_rows, self.xsp_all = {}, {}, {}
        self._g["xsp"] = {}
        if layout != "masked":
            M = k * align
            self._g["xsp"] = {w: packed_act_sf(gb * align, (self.dim if w == 1 else self.inter) // ACT_BLOCK, d)
                              for w in (1, 2)}
            self.xsp = {w: t[:M] for w, t in self._g["xsp"].items()}
            self.xsp_rows = {w: PackedRows(self.xsp[w], align, k) for w in (1, 2)}
            self.xsp_all = {w: PackedRows(self.xsp[w], 1, M) for w in (1, 2)}
        self.C = {p: sf_words(self.K[p] // WEIGHT_BLOCK) for p in PROJS}
        self.sf_dims = ((self.N["w1"], self.C["w1"]), (self.N["w2"], self.C["w2"]))
        rw = {p: self.N[p] * self.K[p] // 2 for p in PROJS}
        if len(set(rw.values())) != 1:
            raise ValueError(f"dg-inplace staging assumes equal weight rows across projections, got {rw}")
        self.row_w = rw["w1"]
        self.sf_store_why: Optional[str] = None
        if self.sf_store in ("pool", "resident"):
            self._check_sf_hbm()
        self.sf_shared: Dict[str, torch.Tensor] = {}
        if self.sf_store == "step":
            self.sf_shared = {p: packed_sf_zeros(self.G_max, self.N[p], self.C[p], d) for p in PROJS}
        self.layers: Dict[int, dict] = {}
        self.plan = hbm_plan(pool.n_layers, hot, pool.spare, {p: shapes[p] for p in PROJS},
                             {p: sshapes[p] for p in PROJS})
        self.spare_added_B = 0  # set by StaticStep (what add_spare_slots returned on this build)
        self.layout_check: Optional[dict] = None

    @staticmethod
    def _psum_align(dg, expected_m: int) -> int:
        """psum: the group-start alignment = the kernel's BLOCK_M. $DSV41_ROWS_PSUM_ALIGN, else the library's
        theoretical value for `expected_m` rows per group, else 32."""
        env = os.environ.get(PSUM_ALIGN_ENV)
        if env:
            return int(env)
        f = getattr(dg, "get_theoretical_mk_alignment_for_contiguous_layout", None)
        if f is not None:
            try:
                return int(f(int(expected_m)))
            except Exception:  # pragma: no cover - library dependent
                pass
        return 32

    def _check_sf_hbm(self) -> None:
        """pool / resident SF: refuse to allocate if it would leave less than $DSV41_SF_POOL_HEADROOM_GIB of HBM
        free (the store falls back to "step", the reason in sf_store_why). A pool copy that already exists (another
        build made it) costs nothing."""
        pool = self.pool
        if self.sf_store == "pool":
            c = getattr(pool, "_dg_packed_sf", None)
            if c is not None and c["key"] == pool_sf_key(pool) and len(c["layers"]) == pool.n_layers:
                return
        need = pool_sf_bytes(pool, self.N, self.C)
        if self.dev.type != "cuda":
            return
        free, _ = torch.cuda.mem_get_info(self.dev)
        head = float(os.environ.get(SF_POOL_HEADROOM_ENV, SF_POOL_HEADROOM_GIB)) * GiB
        if free - need < head:
            self.sf_store_why = (f"{self.sf_store} needs {need / GiB:.2f} GiB, {free / GiB:.2f} GiB free < "
                                 f"{head / GiB:.1f} GiB headroom: using step")
            print(f"[dg-inplace] sf_store {self.sf_store_why}", file=sys.stderr, flush=True)
            self.sf_store = "step"

    def _set_alignment(self, align: int) -> dict:
        """DeepGEMM's global BLOCK_M cap for contiguous layouts := ALIGN (utils_deep_gemm.py:398-413: a mismatch
        reads the wrong group per block -> IMA under replay). Refuses if the library reports another value."""
        f = getattr(self.dg, "set_mk_alignment_for_contiguous_layout", None)
        g = getattr(self.dg, "get_mk_alignment_for_contiguous_layout", None)
        prev = g() if g else None
        if f is not None:
            f(align)
        now = g() if g else None
        if now is not None and int(now) != align:
            raise RuntimeError(f"deep_gemm contiguous alignment is {now} after setting {align}")
        if f is not None and prev is not None and int(prev) != align:   # M1-T10-MOE: leave the global as found
            f(int(prev))
        return {"prev": prev, "set": align, "now": now, "settable": f is not None}

    @contextlib.contextmanager
    def _align_now(self):
        """M1-T10-MOE: the cap is process-global and other builds share it (the 1-row step, the R-row step in another
        call shape, the draft graph's DraftDGBuffers, which sets it once at init): each of OUR GEMM calls runs with
        our ALIGN and puts the previous value back after (host calls only; under capture the launch records the
        kernel picked inside)."""
        f = getattr(self.dg, "set_mk_alignment_for_contiguous_layout", None)
        g = getattr(self.dg, "get_mk_alignment_for_contiguous_layout", None)
        prev = g() if g is not None else None
        if f is not None and prev != self.align:
            f(self.align)
        try:
            yield
        finally:
            if f is not None and prev is not None and prev != self.align:
                f(prev)

    # ---------------------------------------------------------------- per-layer tables (build time)
    def layer_tables(self, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]) -> dict:
        pool, d = self.pool, self.dev
        H = len(pool.hot[layer])
        w = {p: pool.hbm_full[layer][p] for p in PROJS}
        G = w["w1"].size(0)
        for p in PROJS:
            if tuple(w[p].shape) != (G, self.N[p], self.K[p] // 2) or not w[p].is_contiguous():
                raise ValueError(f"L{layer} {p}: pool {tuple(w[p].shape)} is not a contiguous [H+S, N, K/2]")
        if G < H + self.spare_limit:
            raise ValueError(f"L{layer}: {G} pool slots < H {H} + spare slots {self.spare_limit}")
        if self.sf_store == "step":
            sf = {p: self.sf_shared[p][:G] for p in PROJS}
        elif self.sf_store == "pool":
            sf = pool_packed_sf(pool, layer, self.N, self.C, d)
        else:
            sf = {}
            with torch.inference_mode(False), torch.no_grad():
                for p in PROJS:
                    t = packed_sf_zeros(G, self.N[p], self.C[p], d)
                    t[:H].copy_(pack_words(pool.hbm[layer][p][1]))
                    sf[p] = t
        for p in PROJS:
            assert is_mn_major_packed(sf[p], G, self.N[p], self.C[p]), (p, sf[p].shape, sf[p].stride())
        gt = torch.full((pool.n_experts,), -1, dtype=torch.int32)
        for slot, e in enumerate(pool.hot[layer]):
            gt[e] = slot
        dsts = [w["w1"], w["w3"], w["w2"], sf["w1"], sf["w3"], sf["w2"]]
        t = {"layer": layer, "H": H, "G": G, "w": w, "sf": sf, "gtab": gt.to(d),
             "ptab6": torch.stack([ptabs[k_] for k_ in ORDER6]).contiguous(), "dsts": dsts,
             "dtab": torch.tensor([x.data_ptr() for x in dsts], dtype=torch.int64, device=d)}
        self.layers[layer] = t
        return t

    # ---------------------------------------------------------------- per-step ops (graph-capturable)
    def stage(self, t: dict, ids: torch.Tensor, ids_out: Optional[torch.Tensor] = None,
              sidx: Optional[torch.Tensor] = None, ovf: Optional[torch.Tensor] = None) -> None:
        """sidx / ovf (M1-T10-FIX2, the R-row MoE): compact cold index per slot and the step's overflow counter;
        the bound is spare_limit. None: the 1-row contract (cold slot j -> spare j)."""
        from runtime.static_kernels import stage_inplace

        kw = {} if sidx is None else {"sidx": sidx, "spare": self.spare_limit, "ovf": ovf}
        stage_inplace(t["ptab6"], ids, t["gtab"], t["dtab"], t["dsts"], self.mi, t["H"], self.align, self.row_w,
                      self.sf_dims, self.sf_store == "step", ids_out, **kw)

    def stage_n_work(self) -> int:
        """Work-list entries the R-row staging launch scans: staged cold slots only (<= spare budget) when the hot
        scales are pre-packed, the used slots (<= R*k) under sf_store "step"."""
        return self.pairs if self.sf_store == "step" else self.spare_limit

    def stage_rows(self, t: dict, m: Dict[str, torch.Tensor]) -> None:
        """M1-T10-MOE: the R-row staging launch from the dedup's work list (cold weight rows + their scales; hot
        scales too under sf_store "step", once per unique expert)."""
        from runtime.static_kernels import stage_rows

        stage_rows(t["ptab6"], m["uniq"], t["gtab"], t["dtab"], t["dsts"], m["sidx"], m["wl"], t["H"], self.row_w,
                   self.sf_dims, self.stage_n_work())

    def _sfa(self, which: int, M: int):
        return self.xsp[which] if self.sf_pack else self.xs[which].view(M, -1)   # T9b-4: pre-packed (no DG convert)

    def gemm_inplace(self, p: str, t: dict, which: int) -> torch.Tensor:
        """d[p] [k, ALIGN, N] <- contiguous grouped GEMM of A = xq[which] as [k*ALIGN, K] (+ fp32 scales) against the
        layer's whole pool W [G, N, K/2] with packed SF, groups from m_indices. Row j*ALIGN = sorted slot j.
        M1-T10-MOE: psum / masked layouts go through gemm_rows."""
        if self.layout != "contiguous":
            return self.gemm_rows(p, t, which)
        Mg = self.gemm_blocks * self.align          # T9a-3: the call's M (= k * ALIGN unless padded)
        xs = self._g["xsp"][which] if self.sf_pack else self._g["xs"][which].view(Mg, -1)
        with self._align_now():
            self.fn((self._g["xq"][which].view(Mg, -1), xs), (t["w"][p], t["sf"][p]),
                    self._g["d"][p].view(Mg, -1), self._g["mi"], recipe_a=RECIPE_A, recipe_b=RECIPE_B)
        self.n_calls += 1
        return self.d[p]

    def gemm_rows(self, p: str, t: dict, which: int) -> torch.Tensor:
        """M1-T10-MOE: the routed GEMM of projection p in this build's layout over the layer's pool (B = W [G, N, K/2],
        packed SF): contiguous (m_indices), psum (gl[:G] = each group's END row; group g starts at gl[g-1] aligned
        up to the mk alignment = ALIGN, reference/deepgemm-tests generate_m_grouped_contiguous + scheduler/gemm.cuh;
        empty groups have end == start: no blocks) or masked (A, d [G, 128, .] prefixes of the G_max + 1 blocks,
        masked_m = gl[:G])."""
        if self.layout == "contiguous":
            return self.gemm_inplace(p, t, which)
        G = int(t["G"])
        if self.layout == "psum":
            M = self.k * self.align
            with self._align_now():
                self.fn((self.xq[which].view(M, -1), self._sfa(which, M)), (t["w"][p], t["sf"][p]),
                        self.d[p].view(M, -1), self.gl[:G], recipe_a=RECIPE_A, recipe_b=RECIPE_B,
                        use_psum_layout=True)
        else:
            if self.fn_masked is None:
                raise RuntimeError("deep_gemm has no m_grouped_fp8_fp4_gemm_nt_masked")
            self.fn_masked((self.xq[which][:G], self.xs[which][:G]), (t["w"][p], t["sf"][p]), self.d[p][:G],
                           self.gl[:G], int(self.rows_expected_m), recipe_a=RECIPE_A, recipe_b=RECIPE_B)
        self.n_calls += 1
        return self.d[p]

    def _fill_layout_probe(self, G: int, n: int) -> None:
        """warmup: n valid groups 0..n-1 with one row each in the layout's terms (m_indices heads / psum / masked_m)."""
        if self.layout == "contiguous":
            self.mi.view(self.k, self.align)[:, 0] = torch.arange(self.k, dtype=torch.int32, device=self.dev) % G
        elif self.layout == "psum":
            g = torch.arange(G, dtype=torch.int32, device=self.dev)
            self.gl[:G] = torch.where(g < n, g * self.align + 1, torch.full_like(g, n * self.align))
        else:
            g = torch.arange(G, dtype=torch.int32, device=self.dev)
            self.gl[:G] = (g < n).to(torch.int32)

    def _reset_layout(self) -> None:
        if self.mi is not None:
            self.mi.fill_(-1)
        if self.gl is not None:
            self.gl.zero_()

    def warmup(self) -> None:
        """First-call JIT of both GEMM shapes for every distinct group count, on real block heads, before any
        capture; m_indices / gl are reset afterwards (the next dedup writes them)."""
        seen = set()
        for l in sorted(self.layers):
            t = self.layers[l]
            if t["G"] in seen:
                continue
            seen.add(t["G"])
            self._fill_layout_probe(t["G"], min(self.pairs, t["G"]))
            for p in PROJS:
                self.gemm_inplace(p, t, 2 if p == "w2" else 1)
        self._reset_layout()

    # ---------------------------------------------------------------- M1-T10-MOE build-time layout self-check
    def check_layout(self, seed: int = 20261107, n_groups: Optional[int] = None) -> dict:
        """This build's GEMM call shape (layout + alignment) against the T9a-2 contiguous call at ALIGN = 128 on
        layer 0's REAL pool bytes: `n_groups` (default min(R*k, 22, H)) distinct hot groups with 1..R rows each,
        random fp8 activations with power-of-two per-32 scales, both GEMM shapes (w1: K = dim, w2: K = inter);
        every real output row bitwise. Uses temporary reference buffers and a temporary packed SF of layer 0;
        restores this build's A / m_indices / gl. Returns {"equal": bool, ...}; the caller falls back on False."""
        pool, d = self.pool, self.dev
        R_ = int(getattr(self.owner, "rows", 1))
        H0 = len(pool.hot[0])
        W = {p: pool.hbm_full[0][p] for p in ("w1", "w2")}
        G0 = W["w1"].size(0)
        U = int(n_groups or min(self.pairs, 22, H0))
        if U < 1:
            return {"equal": True, "skipped": "layer 0 has no hot experts"}
        g = torch.Generator(device="cpu").manual_seed(seed)
        groups = sorted(torch.randperm(H0, generator=g, device="cpu")[:U].tolist())
        counts = torch.randint(1, R_ + 1, (U,), generator=g, device="cpu").tolist()
        sfp = {}
        with torch.inference_mode(False), torch.no_grad():
            for p in W:
                t_ = packed_sf_zeros(G0, self.N[p], self.C[p], d)
                t_[:H0].copy_(pack_words(pool.hbm[0][p][1]))
                sfp[p] = t_
        tt = {"w": W, "sf": sfp, "G": G0}
        res = {"layout": self.layout, "align": self.align, "groups": U, "rows": sum(counts), "projs": {}}
        try:
            for p, which in (("w1", 1), ("w2", 2)):
                K = self.dim if which == 1 else self.inter
                n = sum(counts)
                xq = (torch.randn(n, K, generator=g, device="cpu") * 2).clamp(-448, 448).to(torch.float8_e4m3fn).to(d)
                xs = torch.exp2(torch.randint(-3, 4, (n, K // ACT_BLOCK), generator=g, device="cpu").float()).to(d)
                # reference: contiguous, ALIGN 128, U blocks
                Ar = torch.zeros(U * MK_ALIGN, K, dtype=torch.float8_e4m3fn, device=d)
                Sr = torch.ones(U * MK_ALIGN, K // ACT_BLOCK, dtype=torch.float32, device=d)
                mir = torch.full((U * MK_ALIGN,), -1, dtype=torch.int32, device=d)
                dr = torch.zeros(U * MK_ALIGN, self.N[p], dtype=torch.bfloat16, device=d)
                rows_ref, rows_got = [], []
                i = 0
                for b, (gr, c) in enumerate(zip(groups, counts)):
                    for r in range(c):
                        Ar[b * MK_ALIGN + r] = xq[i]
                        Sr[b * MK_ALIGN + r] = xs[i]
                        mir[b * MK_ALIGN + r] = gr
                        rows_ref.append(b * MK_ALIGN + r)
                        i += 1
                setf = getattr(self.dg, "set_mk_alignment_for_contiguous_layout", None)
                getf = getattr(self.dg, "get_mk_alignment_for_contiguous_layout", None)
                prev_al = getf() if getf is not None else None
                if setf is not None:
                    setf(MK_ALIGN)
                try:
                    self.fn((Ar, Sr), (W[p], sfp[p]), dr, mir, recipe_a=RECIPE_A, recipe_b=RECIPE_B)
                finally:
                    if setf is not None and prev_al is not None:
                        setf(prev_al)
                # this build's layout, in its own buffers
                self._reset_layout()
                xqb = self.xq[which].view(-1, K)
                xsb = self.xs[which].view(-1, K // ACT_BLOCK)
                prev_pack, self.sf_pack = self.sf_pack, False
                order = sorted(range(U), key=lambda b_: groups[b_])          # psum: blocks by ascending group
                pblk = {b_: j for j, b_ in enumerate(order)}
                i = 0
                for b, (gr, c) in enumerate(zip(groups, counts)):
                    for r in range(c):
                        if self.layout == "contiguous":
                            a = b * self.align + r
                            self.mi[a] = gr
                        elif self.layout == "psum":
                            a = pblk[b] * self.align + r
                        else:
                            a = gr * self.block_rows + r
                        xqb[a] = xq[i]
                        xsb[a] = xs[i]
                        rows_got.append(a)
                        i += 1
                if self.layout == "psum":
                    gg = torch.arange(G0, device=d).view(-1, 1)
                    gv = torch.tensor(groups, device=d).view(1, -1)
                    cv = torch.tensor(counts, device=d).view(1, -1)
                    self.gl[:G0] = ((gv < gg).sum(1) * self.align + torch.where(gv == gg, cv, 0).sum(1)).int()
                elif self.layout == "masked":
                    for gr, c in zip(groups, counts):
                        self.gl[gr] = c
                try:
                    out = self.gemm_inplace(p, tt, which)
                finally:
                    self.sf_pack = prev_pack
                got = out.view(-1, self.N[p])[torch.tensor(rows_got, device=d)]
                want = dr[torch.tensor(rows_ref, device=d)]
                res["projs"][p] = compare_bf16(got, want)
            res["equal"] = all(v["equal"] for v in res["projs"].values())
        except Exception as e:
            res["equal"] = False
            res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            self._reset_layout()
            for w_ in (1, 2):
                self.xq[w_].zero_()
                self.xs[w_].fill_(1.0)
            del sfp
        self.layout_check = res
        return res

    # ---------------------------------------------------------------- T9a-3 build-time row-match check
    def check_rowmatch(self, rows: int, ref_align: int, ref_blocks: Optional[int] = None, seed: int = 20261107
                       ) -> dict:
        """T9a-3: is THIS 1-row build's routed GEMM per-row bit-equal to the R-row step's call? On layer 0's REAL pool
        bytes: R rows each routed to k distinct hot groups (drawn from a small candidate set, so groups are shared
        across rows as in the 6-row step), random fp8 activations with power-of-two per-32 scales (w1: one row per
        token; w2: one row per (token, expert) pair). Reference = the R-row call (contiguous, ALIGN ref_align,
        ref_blocks = R*k A blocks, one block per unique group with its rows ascending: runtime/moe_rows.py's layout)
        on temporary buffers; this build = each row alone through gemm_inplace on the build's OWN buffers (padding
        blocks included); T9a-2's call (ALIGN 128, k blocks) is measured too, for the record. Every output row
        bitwise. Restores the build's buffers. -> {"equal": bool (this build == the R-row call on every row), ...}."""
        pool, d = self.pool, self.dev
        k, R_ = int(self.k), int(rows)
        H0 = len(pool.hot[0])
        W = {p: pool.hbm_full[0][p] for p in ("w1", "w2")}
        G0 = W["w1"].size(0)
        rb = int(ref_blocks or R_ * k)
        res = {"rows": R_, "k": k, "ref": {"align": int(ref_align), "blocks": rb},
               "build": {"align": self.align, "blocks": self.gemm_blocks}, "projs": {}}
        if H0 < k:
            res.update(equal=False, error=f"layer 0 has {H0} hot experts < topk {k}")
            return res
        g = torch.Generator(device="cpu").manual_seed(seed)
        nc = min(H0, max(k + 1, (R_ * k) // 2))
        cand = sorted(torch.randperm(H0, generator=g, device="cpu")[:nc].tolist())
        rexp = [sorted(cand[i] for i in torch.randperm(nc, generator=g, device="cpu")[:k].tolist()) for _ in range(R_)]
        gmap = {e: e for e in cand}                       # expert id := its hot slot (group) on layer 0
        res["routing"] = rexp
        sfp = {}
        prev_pack = self.sf_pack
        try:
            with torch.inference_mode(False), torch.no_grad():
                for p in W:
                    t_ = packed_sf_zeros(G0, self.N[p], self.C[p], d)
                    t_[:H0].copy_(pack_words(pool.hbm[0][p][1]))
                    sfp[p] = t_
            tt = {"w": W, "sf": sfp, "G": G0}
            all_eq = True
            for p, which in (("w1", 1), ("w2", 2)):
                K = self.dim if which == 1 else self.inter
                keys = [(r, e) for r in range(R_) for e in rexp[r]]
                n_src = R_ if which == 1 else len(keys)
                q = (torch.randn(n_src, K, generator=g, device="cpu") * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
                s = torch.exp2(torch.randint(-3, 4, (n_src, K // ACT_BLOCK), generator=g, device="cpu").float())
                q, s = q.to(d), s.to(d)
                acts = {kk: ((q[kk[0]], s[kk[0]]) if which == 1 else (q[i], s[i])) for i, kk in enumerate(keys)}
                ref = run_contiguous(self.dg, W[p], sfp[p], r_row_blocks(rexp, gmap), acts, int(ref_align), rb)
                today = {}
                for r in range(R_):
                    today.update(run_contiguous(self.dg, W[p], sfp[p], one_row_blocks(r, rexp[r], gmap), acts,
                                                MK_ALIGN, k))
                # this build, one row at a time, through its own buffers and call
                self.sf_pack = False
                got = {}
                for r in range(R_):
                    self._reset_layout()
                    for j, e in enumerate(rexp[r]):
                        self.xq[which][j, 0].view(torch.uint8).copy_(acts[(r, e)][0].view(torch.uint8))
                        self.xs[which][j, 0].copy_(acts[(r, e)][1])
                        self.mi[j * self.align] = gmap[e]
                    out = self.gemm_inplace(p, tt, which)
                    for j, e in enumerate(rexp[r]):
                        got[(r, e)] = out[j, 0].clone()

                def rows_eq(a, b):
                    return sum(all(torch.equal(a[(r, e)].view(torch.int16), b[(r, e)].view(torch.int16))
                                   for e in rexp[r]) for r in range(R_))

                cat = lambda m: torch.stack([m[kk] for kk in keys])  # noqa: E731
                rec = {"build_vs_rows": rows_eq(got, ref), "today_vs_rows": rows_eq(today, ref),
                       "build_vs_today": rows_eq(got, today), "of": R_,
                       "build_vs_rows_cmp": compare_bf16(cat(got), cat(ref)),
                       "today_vs_rows_cmp": compare_bf16(cat(today), cat(ref))}
                res["projs"][p] = rec
                all_eq = all_eq and rec["build_vs_rows"] == R_
            res["equal"] = all_eq
            res["today_equal"] = all(v["today_vs_rows"] == R_ for v in res["projs"].values())
        except Exception as e:
            res["equal"] = False
            res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            self.sf_pack = prev_pack
            self._reset_layout()
            for w_ in (1, 2):
                self._g["xq"][w_].zero_()
                self._g["xs"][w_].fill_(1.0)
            del sfp
        return res

    def describe(self) -> dict:
        r = super().describe()
        free = None
        if self.dev.type == "cuda":
            f, tot = torch.cuda.mem_get_info(self.dev)
            free = {"free_GiB": round(f / GiB, 3), "total_GiB": round(tot / GiB, 3),
                    "allocated_GiB": round(torch.cuda.memory_allocated(self.dev) / GiB, 3)}
        if self.sf_store == "step":
            sf_b = sum(t.untyped_storage().nbytes() for t in self.sf_shared.values())
        elif self.sf_store == "pool":
            c = getattr(self.pool, "_dg_packed_sf", None) or {"layers": {}}
            sf_b = sum(t.untyped_storage().nbytes() for sf in c["layers"].values() for t in sf.values())
        else:
            sf_b = sum(t["sf"][p].untyped_storage().nbytes() for t in self.layers.values() for p in PROJS)
        pool = self.pool
        spare_b = sum((pool.hbm_full[l][p].numel() - pool.hbm[l][p][0].numel()) * pool.hbm_full[l][p].element_size()
                      for l in range(pool.n_layers) for p in PROJS)
        extra = (self.mi.numel() * 4 if self.mi is not None else 0) + (self.gl.numel() * 4 if self.gl is not None else 0)
        r.update({"impl": "dg-inplace", "layout": self.layout, "align": self.align, "alignment": self.alignment,
                  "rows_expected_m": self.rows_expected_m if self.layout != "contiguous" else None,
                  "block_rows": self.block_rows, "a_blocks": self.k, "layout_check": self.layout_check,
                  "gemm_blocks": self.gemm_blocks, "gemm_m": self.gemm_blocks * self.block_rows,
                  "rowmatch": self.rowmatch,
                  "sf_store": self.sf_store, "sf_store_why": self.sf_store_why, "stage_n_work": self.stage_n_work(),
                  "act_sf_packed": self.sf_pack, "act_sf_packed_why": None if self.sf_pack else self.sf_pack_why,
                  "spare_slots": self.pool.spare, "spare_limit": self.spare_limit, "G_max": self.G_max,
                  "max_m": self.block_rows,
                  "groups_per_layer": sorted({t["G"] for t in self.layers.values()}),
                  "hbm_added_B": {"spare_weight_slots": spare_b, "packed_sf": sf_b,
                                  "spare_grown_by_this_build": self.spare_added_B},
                  "hbm_added_GiB": round((spare_b + sf_b) / GiB, 3), "hbm_after_build": free,
                  "plan": self.plan, "buffer_bytes": r["buffer_bytes"] + extra})
        return r


# ------------------------------------------------------------------------------------------------ the MoE
def build_one_row(owner, shapes, sshapes, *, pool, swiglu_limit: float, deterministic: bool = True,
                  mode: Optional[str] = None) -> "DGInplaceBuffers":
    """T9a-3: the 1-row dg-inplace step's buffers in the --moe-rowmatch call shape (rowmatch_plan; $DSV41_MOE_ROWMATCH,
    default on). Unless off or $DSV41_ROWS_LAYOUT_CHECK=0, the build is checked per row against the R-row call on
    layer 0's real bytes (check_rowmatch); a mismatch (or a failing check) builds T9a-2's call instead and says why.
    The record is b.rowmatch (describe()["rowmatch"])."""
    plan = rowmatch_plan(owner.topk, mode)
    kw = dict(pool=pool, swiglu_limit=swiglu_limit, deterministic=deterministic)
    if not plan["on"]:
        b = DGInplaceBuffers(owner, shapes, sshapes, **kw)
        b.rowmatch = dict(plan, check=None)
        return b
    b = DGInplaceBuffers(owner, shapes, sshapes, align=plan["align"], gemm_blocks=plan["blocks"], **kw)
    rec = None
    if os.environ.get(LAYOUT_CHECK_ENV, "1") != "0":
        rec = b.check_rowmatch(plan["rows"], plan["align"], plan["rows"] * int(owner.topk))
        if not rec.get("equal"):
            why = rec.get("error") or {p: {k_: v[k_] for k_ in ("build_vs_rows", "of")}
                                       for p, v in rec.get("projs", {}).items()}
            print(f"[dg-inplace] moe rowmatch {plan['mode']!r} (ALIGN {plan['align']}, {plan['blocks']} blocks) failed "
                  f"its build check ({why}); using T9a-2's call (ALIGN {MK_ALIGN}, {owner.topk} blocks)",
                  file=sys.stderr, flush=True)
            del b
            b = DGInplaceBuffers(owner, shapes, sshapes, **kw)
            b.rowmatch = dict(plan, on=False, check=rec, why=f"build check failed: {why}")
            return b
    b.rowmatch = dict(plan, check=rec)
    return b


class DGInplaceMoE(DGMaskedMoE):
    """One layer's MoE with the routed GEMMs reading the hot pool in place (module docstring). Same interface as
    StaticMoE / DGMaskedMoE; the act / SwiGLU / combine / shared sub-steps are DGMaskedMoE's."""

    def __init__(self, owner, moe, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]):
        super().__init__(owner, moe, layer, ptabs)
        self.t = owner.dg.layer_tables(layer, ptabs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        o, moe, b = self.owner, self.moe, self.owner.dg
        glue = o.glue
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893
        if glue is not None and glue.on("router"):
            ids, w = glue.route(moe, x)                                # T9b-2 R2 #4 (T9b-4: was not wired here)
        else:
            weights, indices = moe.gate(x, None)                       # model.py:894
            ids, order = torch.sort(indices[0])                        # ascending id = model.py:897 loop order
            w = weights[0, order]
        self._stage(ids)                                               # 1 launch: cold rows, scales, m_indices
        self._act(x)                                                   # model.py:187, once for w1 and w3
        if glue is not None and glue.on("swiglu_routed"):
            b.gemm_inplace("w1", self.t, 1)                                       # model.py:845
            b.gemm_inplace("w3", self.t, 1)                                       # model.py:846
            self._swiglu_fused(w)
        else:
            gate = b.gemm_inplace("w1", self.t, 1)[:, 0].float()                  # model.py:845
            up = b.gemm_inplace("w3", self.t, 1)[:, 0].float()                    # model.py:846
            self._swiglu(gate, up, w, x.dtype)
        out = b.gemm_inplace("w2", self.t, 2)[:, 0]                               # [k, DIM] bf16
        return self._combine(out, x).view(shape)

    def _stage(self, ids: torch.Tensor) -> None:
        o = self.owner
        o.dg.stage(self.t, ids, o.ids_step[self.layer] if o.counters else None)
