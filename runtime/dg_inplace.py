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

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from runtime.dg_moe import ACT_BLOCK, PROJS, RECIPE_A, RECIPE_B, WEIGHT_BLOCK, DGBuffers, DGMaskedMoE

MK_ALIGN = 128                     # contiguous-layout rows per group block (= BLOCK_M cap); $DSV41_DG_ALIGN
ALIGN_ENV = "DSV41_DG_ALIGN"
SF_STORES = ("step", "resident")
SF_STORE_ENV = "DSV41_DG_SF_STORE"
SF_PACK = 4                        # e8m0 codes per int32 word
ORDER6 = (("w1", "weight"), ("w3", "weight"), ("w2", "weight"), ("w1", "scale"), ("w3", "scale"), ("w2", "scale"))
GiB = 2 ** 30


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
    return packed.contiguous().view(torch.uint8)


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


# ------------------------------------------------------------------------------------------------ static buffers
class DGInplaceBuffers(DGBuffers):
    """dg-masked's buffers with max_m = ALIGN (the activation buffers ARE the contiguous A: [k, ALIGN, K] viewed as
    [k*ALIGN, K]), no fp32 weight-scale staging, plus m_indices, the packed SF storage and per-layer tables."""

    def __init__(self, owner, shapes, sshapes, *, pool, swiglu_limit: float, deterministic: bool = True, dg=None,
                 align: Optional[int] = None, sf_store: Optional[str] = None, slots: Optional[int] = None,
                 spare_slots: Optional[int] = None):
        """slots: A blocks (default topk; M1-T10-FIX's R-row MoE, runtime/moe_rows.py DGInplaceMoERows, uses R*topk:
        block u holds slot u's count <= R rows). spare_slots (M1-T10-FIX2): spare pool slots per layer the cold
        experts of one step may use (default = slots: the 1-row contract, spare j for slot j; the R-row MoE passes
        its compact-index budget S, default 8, and counts overflows)."""
        align = check_align(align if align is not None else int(os.environ.get(ALIGN_ENV, MK_ALIGN)))
        if align < int(getattr(owner, "rows", 1)):
            raise ValueError(f"contiguous alignment {align} < {owner.rows} rows per block")
        super().__init__(owner, shapes, sshapes, swiglu_limit=swiglu_limit, deterministic=deterministic, dg=dg,
                         rows=align, sf32=False, slots=slots)
        self.align = align
        self.sf_store = sf_store or os.environ.get(SF_STORE_ENV, "step")
        if self.sf_store not in SF_STORES:
            raise ValueError(f"sf_store {self.sf_store!r} not in {SF_STORES} (${SF_STORE_ENV})")
        if pool is None:
            raise ValueError("dg-inplace needs the ExpertPool (it reads the hot pool in place)")
        self.spare_limit = self.k if spare_slots is None else int(spare_slots)
        if not 1 <= self.spare_limit <= self.k:
            raise ValueError(f"spare_slots {self.spare_limit} outside [1, {self.k}] (A blocks per layer)")
        if pool.spare < self.spare_limit:
            raise ValueError(f"pool has {pool.spare} spare slots per layer, dg-inplace needs {self.spare_limit} "
                             f"(ExpertPool.add_spare_slots, done by StaticStep before this)")
        self.pool = pool
        self.fn = getattr(self.dg, "m_grouped_fp8_fp4_gemm_nt_contiguous")
        self.alignment = self._set_alignment(align)
        d, k = self.dev, self.k
        self.mi = torch.full((k * align,), -1, dtype=torch.int32, device=d)
        # T9b-4 lever 4: packed activation SF (A [k*ALIGN, K]) for the w1/w3 input (1) and the w2 input (2); used
        # instead of the fp32 xs only once runtime/fused_glue.py's self-check turned `sf_pack` on (it owns every
        # writer of row 0 of each block then)
        self.sf_pack = False
        self.sf_pack_why: Optional[str] = "not enabled (needs glue_impl fused: dg_act + swiglu_routed + self-check)"
        self.xsp = {w: packed_act_sf(k * align, (self.dim if w == 1 else self.inter) // ACT_BLOCK, d) for w in (1, 2)}
        self.xsp_rows = {w: PackedRows(self.xsp[w], align, k) for w in (1, 2)}
        self.C = {p: sf_words(self.K[p] // WEIGHT_BLOCK) for p in PROJS}
        self.sf_dims = ((self.N["w1"], self.C["w1"]), (self.N["w2"], self.C["w2"]))
        rw = {p: self.N[p] * self.K[p] // 2 for p in PROJS}
        if len(set(rw.values())) != 1:
            raise ValueError(f"dg-inplace staging assumes equal weight rows across projections, got {rw}")
        self.row_w = rw["w1"]
        hot = [len(pool.hot[l]) for l in range(pool.n_layers)]
        self.G_max = max(hot) + pool.spare
        self.sf_shared: Dict[str, torch.Tensor] = {}
        if self.sf_store == "step":
            self.sf_shared = {p: packed_sf_zeros(self.G_max, self.N[p], self.C[p], d) for p in PROJS}
        self.layers: Dict[int, dict] = {}
        self.plan = hbm_plan(pool.n_layers, hot, pool.spare, {p: shapes[p] for p in PROJS},
                             {p: sshapes[p] for p in PROJS})
        self.spare_added_B = 0  # set by StaticStep (what add_spare_slots returned on this build)

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
        return {"prev": prev, "set": align, "now": now, "settable": f is not None}

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

    def gemm_inplace(self, p: str, t: dict, which: int) -> torch.Tensor:
        """d[p] [k, ALIGN, N] <- contiguous grouped GEMM of A = xq[which] as [k*ALIGN, K] (+ fp32 scales) against the
        layer's whole pool W [G, N, K/2] with packed SF, groups from m_indices. Row j*ALIGN = sorted slot j."""
        M = self.k * self.align
        sfa = self.xsp[which] if self.sf_pack else self.xs[which].view(M, -1)   # T9b-4: pre-packed (no DG convert)
        self.fn((self.xq[which].view(M, -1), sfa), (t["w"][p], t["sf"][p]),
                self.d[p].view(M, -1), self.mi, recipe_a=RECIPE_A, recipe_b=RECIPE_B)
        self.n_calls += 1
        return self.d[p]

    def warmup(self) -> None:
        """First-call JIT of both GEMM shapes for every distinct group count, on real block heads (slots 0..k-1),
        before any capture; m_indices is reset to all -1 afterwards (the next stage writes the heads)."""
        seen = set()
        for l in sorted(self.layers):
            t = self.layers[l]
            if t["G"] in seen:
                continue
            seen.add(t["G"])
            self.mi.view(self.k, self.align)[:, 0] = torch.arange(self.k, dtype=torch.int32, device=self.dev) % t["G"]
            for p in PROJS:
                self.gemm_inplace(p, t, 2 if p == "w2" else 1)
        self.mi.fill_(-1)

    def describe(self) -> dict:
        r = super().describe()
        free = None
        if self.dev.type == "cuda":
            f, tot = torch.cuda.mem_get_info(self.dev)
            free = {"free_GiB": round(f / GiB, 3), "total_GiB": round(tot / GiB, 3),
                    "allocated_GiB": round(torch.cuda.memory_allocated(self.dev) / GiB, 3)}
        sf_b = sum(t.untyped_storage().nbytes() for t in self.sf_shared.values()) if self.sf_store == "step" else \
            sum(t["sf"][p].untyped_storage().nbytes() for t in self.layers.values() for p in PROJS)
        pool = self.pool
        spare_b = sum((pool.hbm_full[l][p].numel() - pool.hbm[l][p][0].numel()) * pool.hbm_full[l][p].element_size()
                      for l in range(pool.n_layers) for p in PROJS)
        r.update({"impl": "dg-inplace", "align": self.align, "alignment": self.alignment, "sf_store": self.sf_store,
                  "act_sf_packed": self.sf_pack, "act_sf_packed_why": None if self.sf_pack else self.sf_pack_why,
                  "spare_slots": self.pool.spare, "spare_limit": self.spare_limit, "G_max": self.G_max,
                  "max_m": self.align,
                  "groups_per_layer": sorted({t["G"] for t in self.layers.values()}),
                  "hbm_added_B": {"spare_weight_slots": spare_b, "packed_sf": sf_b,
                                  "spare_grown_by_this_build": self.spare_added_B},
                  "hbm_added_GiB": round((spare_b + sf_b) / GiB, 3), "hbm_after_build": free,
                  "plan": self.plan, "buffer_bytes": r["buffer_bytes"] + self.mi.numel() * 4})
        return r


# ------------------------------------------------------------------------------------------------ the MoE
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
