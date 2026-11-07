"""M1-T10: the static MoE at R rows per step (the 6-row verify step), docs/M1-DSPARK-DESIGN.md 2.1 rows "MoE routing",
"DG masked GEMM" and "combine".

    st = StaticStep(model, pool, rows=6, moe_impl="ref" | "dg-masked")   # runtime/static_step.py builds these

What the reference does at R rows (model.py:891-906): the gate on all R rows (894) -> indices [R, k]; then for every
expert id i in ascending order with at least one routed row, `idx, top = where(indices == i)` (rows ascending) and
`y[idx] += expert(x[idx], weights[idx, top, None])` -- each expert ONCE on all its rows (M = count), fp32 y += bf16
expert output, so each row's sum runs over its k experts in ascending id; then `y += shared(x)` at M = R.

The static version (`dedup_routes`, all device ops, no host sync, graph-capturable): the R*k (row, expert) pairs are
stable-sorted by expert id (ties keep the flat order r*k + j, i.e. rows ascending = where()'s order), unique flags ->
slot = cumsum - 1 (U <= R*k slots, ascending id), rank = position inside the slot's run; slot u's expert is gathered
into staging slot u (slots >= U gather the largest routed id: valid memory, results unused, masked_m 0); the
activation of pair (r, e) sits in row `rank` of slot u's block (A row u*M + rank); the combine adds, per row, its k
outputs in ascending expert id (`posmat`, the reference's order). Nothing about this depends on P0's parity.

  StaticMoERows (moe_impl "ref"): per slot one reference `Expert` evaluation on its [R, D] block (rows >= count
    are stale but finite; their outputs are never read). Bit-identical to the reference iff the routed GEMM's rows
    do not depend on M (the reference evaluates M = count rows) -- the V0 census measures that (runtime/row_census.py).
  DGMaskedMoERows (moe_impl "dg-masked"): runtime/dg_moe.py's DGBuffers with R*k groups; act_quant of the R rows ONCE
    (model.py:187: the same bytes the reference computes per expert), scattered to A rows; masked_m[u] = count (0 for
    unused slots), expected_m 8 as at 1 row; w1/w3 -> the reference SwiGLU glue on the P = R*k pair rows -> act_quant
    -> w2 -> per-row ascending-id combine -> + shared.

  DGInplaceMoERows (moe_impl "dg-inplace", M1-T10-FIX / FIX2): the contiguous grouped GEMM over the hot pool in
    place, cold experts staged into S spare pool slots by a compact cold index (S = StaticStep spare_slots, default
    8; more cold unique experts than S in a layer-step = an overflow, counted on the device, never silent). Same
    dedup, glue and combine.

The fused glue at R > 1 (M1-T10-PORT, runtime/fused_glue.py `ROWS_CLUSTERS`): router (`route_rows`: the gate's post-
topk glue in one launch per layer), `dedup` (`dedup_fused`: the whole mapping below -- sort-by-id, unique, slot/rank,
arow / prow / pw / posmat, masked_m, the counters, and for dg-inplace the compact cold index, the stage ids and the
m_indices tags -- in ONE Triton launch, `m1_dedup_routes`, CPU mirror = the torch functions here), dg_act (the R rows'
act_quant scattered to the pair A rows in one launch), swiglu_routed (gather + SwiGLU + weight + act_quant + scatter
in one launch), moe_combine (per-row ascending-id combine + shared + bf16 in one launch), swiglu_shared.

M1-T10-MOE (docs/HANDOFF-M1-T10-MOE.md): the dedup is `m1_dedup_warp` by default -- one program of ONE warp (bitonic
tl.sort of id*KP + i keys, neighbour compare + cumsum for slots, one scratch row for the run starts), the matrix kernel
`m1_dedup_routes` kept as $DSV41_DEDUP_IMPL=matrix and as the automatic fallback if the warp kernel fails to compile;
for the R-row dg-inplace step it also writes the GEMM call shape's inputs (m_indices tags / psum ends / masked_m), the
overflow count and the staging work list, and lays arow / posmat out for the call shape (`rows_layout` = its torch
mirror); the staging is one launch over that work list (static_kernels.stage_rows: cold bytes + scales once per
unique expert; hot scales pre-packed in the pool's SF copy, dg_inplace sf_store "pool").
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:  # GPU image only; the Mac runs the torch mapping (and tests/triton_shim.py for the kernel body)
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = tl = None
    HAVE_TRITON = False

PROJS = ("w1", "w3", "w2")
KERNELS_ON_CPU = False   # tests set it under tests/triton_shim.py: dedup_fused launches the kernel
DEDUP_MAX_PAIRS = 64     # R*k pairs in the one-program dedup kernel (R <= 8 rows, k <= 8)


def dedup_routes(indices: torch.Tensor, weights: torch.Tensor, block_rows: int) -> Dict[str, torch.Tensor]:
    """indices [R, k] int64 (distinct ids per row), weights [R, k] fp32 -> the slot mapping (module docstring),
    S = R*k:
      uniq [S]    expert id of slot u (ascending for u < U; the largest routed id for u >= U)
      counts [S]  rows routed to slot u (0 for u >= U) -> masked_m
      valid [S]   counts > 0 as int64
      slot/rank/prow/pw [S]  per sorted pair t: its slot, rank in the slot, source row, routing weight
      arow [S]    A row of sorted pair t = slot * block_rows + rank
      posmat [R, k]  A rows of row r's pairs in ascending expert id (the combine order)
    Device ops only (sort, cumsum, cummax, compare-sum, gather): no host sync."""
    R, k = indices.shape
    S = R * k
    dev = indices.device
    flat = indices.reshape(-1)
    sid, perm = torch.sort(flat, stable=True)
    ar = torch.arange(S, device=dev)
    first = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), sid[1:] != sid[:-1]])
    slot = torch.cumsum(first.long(), 0) - 1
    start = torch.cummax(torch.where(first, ar, torch.zeros_like(ar)), 0).values
    rank = ar - start
    counts = (slot.view(S, 1) == ar.view(1, S)).long().sum(0)
    firstpos = torch.cumsum(counts, 0) - counts
    uniq = sid.index_select(0, firstpos.clamp(max=S - 1))
    arow = slot * int(block_rows) + rank
    prow = torch.div(perm, k, rounding_mode="floor")
    pw = weights.reshape(-1).index_select(0, perm)
    inv = torch.argsort(perm)
    posflat = arow.index_select(0, inv).view(R, k)
    order = torch.argsort(indices, dim=1, stable=True)
    posmat = posflat.gather(1, order)
    return {"uniq": uniq, "counts": counts, "valid": (counts > 0).long(), "slot": slot, "rank": rank, "prow": prow,
            "pw": pw, "arow": arow, "posmat": posmat}


def dedup_routes_host(indices, weights) -> dict:
    """The same mapping from plain Python (tests): {expert: [(row, j, weight), ...] rows ascending} in ascending id."""
    out: Dict[int, list] = {}
    for r, row in enumerate(indices):
        for j, e in enumerate(row):
            out.setdefault(int(e), []).append((r, j, float(weights[r][j])))
    return {e: sorted(v) for e, v in sorted(out.items())}


# ------------------------------------------------------------------------------------------------ M1-T10-PORT dedup
class DedupBuffers:
    """One layer's static outputs of the fused dedup (graph-stable addresses; allocated at build). P = R*k pairs.
    M1-T10-MOE: `wl` = the staging work list (slot ids, -1 padded), `scr` = the warp kernel's int32 scratch row
    (sorted ids [KP], first positions [KP + 1], per-slot block [KP])."""

    def __init__(self, rows: int, k: int, device):
        P = int(rows) * int(k)
        i64 = dict(dtype=torch.long, device=device)
        self.rows, self.k, self.P = int(rows), int(k), P
        self.uniq, self.counts, self.valid = (torch.zeros(P, **i64) for _ in range(3))
        self.slot, self.rank, self.prow, self.arow = (torch.zeros(P, **i64) for _ in range(4))
        self.pw = torch.zeros(P, dtype=torch.float32, device=device)
        self.posmat = torch.zeros(int(rows), int(k), **i64)
        self.ids_stage = torch.zeros(P, **i64)
        self.sidx = torch.zeros(P, **i64)
        self.tags = torch.zeros(P, int(rows), dtype=torch.int32, device=device)
        self.wl = torch.full((P,), -1, dtype=torch.long, device=device)
        self.KP = _np2(P)
        self.scr = torch.zeros(3 * self.KP + 1, dtype=torch.int32, device=device)


if HAVE_TRITON:
    @triton.jit
    def m1_dedup_routes(idx, w, uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                        mm_o, ids_o, uval_o, gtab, stage_o, sidx_o, tags_o, dummy,
                        P, K, NR, BR, H, S,
                        KP: tl.constexpr, NRP: tl.constexpr, HAS_MM: tl.constexpr, HAS_CTR: tl.constexpr,
                        INPL: tl.constexpr, HAS_DUMMY: tl.constexpr):
        """ONE program: dedup_routes' whole mapping for P = R*k (row r, choice j) pairs, by compare matrices over
        [KP, KP] (no sort / cumsum primitive): pair i's stable sorted position = #{id_j < id_i} + #{j < i, id_j ==
        id_i}; its rank = the second term; its slot = #{distinct ids < id_i}; counts[u] = #{pairs in slot u};
        uniq[u] = slot u's id (u < U) or the largest routed id (u >= U, dedup_routes' clamp); posmat[r, c] = the A
        row of row r's pair with the c-th smallest id. HAS_MM: masked_m (int32) = counts. HAS_CTR: the step
        counters' rows (ids_step = uniq, uvalid_step = counts > 0). INPL (dg-inplace): g = gtab[uniq]; the compact
        cold index c = cumsum(cold) - 1 (moe_rows.compact_cold), sidx / group / ok, the stage ids (unused slots ->
        the hot stand-in `dummy` with HAS_DUMMY) and the m_indices tags [P, NR] (row < counts of an ok slot: group,
        else -1)."""
        ii = tl.arange(0, KP)
        okp = ii < P
        v = tl.load(idx + ii, mask=okp, other=0).to(tl.int64)
        vi = v[:, None]
        vj = v[None, :]
        both = okp[:, None] & okp[None, :]
        before = ii[None, :] < ii[:, None]
        lt_m = (vj < vi) & both
        eq_m = (vj == vi) & both & before
        rank = tl.sum(eq_m.to(tl.int64), 1)
        pos = tl.sum(lt_m.to(tl.int64), 1) + rank
        first = (rank == 0) & okp
        slot = tl.sum((lt_m & first[None, :]).to(tl.int64), 1)
        n_u = tl.sum(first.to(tl.int64), 0)
        vmax = tl.max(tl.where(okp, v, -1), 0)
        in_u = (slot[None, :] == ii[:, None]) & okp[None, :]                  # [u, i]: pair i in slot u
        counts = tl.sum(in_u.to(tl.int64), 1)
        uq_lo = tl.sum(tl.where(in_u & first[None, :], vj, 0), 1)
        uniq = tl.where(ii < n_u, uq_lo, vmax)
        arow = slot * BR + rank
        prow = ii // K
        tl.store(slot_o + pos, slot, mask=okp)
        tl.store(rank_o + pos, rank, mask=okp)
        tl.store(arow_o + pos, arow, mask=okp)
        tl.store(prow_o + pos, prow, mask=okp)
        tl.store(pw_o + pos, tl.load(w + ii, mask=okp, other=0.0), mask=okp)
        same_row = (prow[None, :] == prow[:, None]) & both
        col = tl.sum((lt_m & same_row).to(tl.int64), 1)                      # ids distinct within a row
        tl.store(posmat_o + prow * K + col, arow, mask=okp)
        valid = (counts > 0).to(tl.int64)
        tl.store(uniq_o + ii, uniq, mask=okp)
        tl.store(counts_o + ii, counts, mask=okp)
        tl.store(valid_o + ii, valid, mask=okp)
        if HAS_MM:
            tl.store(mm_o + ii, counts.to(tl.int32), mask=okp)
        if HAS_CTR:
            tl.store(ids_o + ii, uniq, mask=okp)
            tl.store(uval_o + ii, valid, mask=okp)
        if INPL:
            gq = tl.load(gtab + uniq, mask=okp, other=0).to(tl.int64)
            cold = (gq < 0) & (counts > 0) & okp
            upto = (ii[None, :] <= ii[:, None]) & cold[None, :]
            cidx = tl.sum(upto.to(tl.int64), 1) - 1
            sidx = tl.where(cold, cidx, -1)
            grp = tl.where(cold, cidx + H, gq)
            n_ok = tl.where(sidx < S, counts, 0)
            if HAS_DUMMY:
                stage_ids = tl.where(counts > 0, uniq, tl.load(dummy))
            else:
                stage_ids = uniq
            tl.store(stage_o + ii, stage_ids, mask=okp)
            tl.store(sidx_o + ii, sidx, mask=okp)
            rr = tl.arange(0, NRP)
            tag = tl.where(rr[None, :] < n_ok[:, None], grp[:, None], -1).to(tl.int32)
            tl.store(tags_o + ii[:, None] * NR + rr[None, :], tag, mask=okp[:, None] & (rr[None, :] < NR))

    @triton.jit
    def m1_dedup_warp(idx, w, uniq_o, counts_o, valid_o, slot_o, rank_o, prow_o, pw_o, arow_o, posmat_o,
                      mm_o, ids_o, uval_o, gtab, stage_o, sidx_o, tags_o, wl_o, gl_o, ovf, scr, dummy,
                      P, K, NR, BR, H, S, TS, G, GSCR,
                      KP: tl.constexpr, NRP: tl.constexpr, HAS_MM: tl.constexpr, HAS_CTR: tl.constexpr,
                      INPL: tl.constexpr, HAS_DUMMY: tl.constexpr, LAYOUT: tl.constexpr, HAS_TAGS: tl.constexpr,
                      HAS_WL: tl.constexpr, STEP_SF: tl.constexpr, HAS_OVF: tl.constexpr):
        """M1-T10-MOE: m1_dedup_routes' mapping in ONE program of ONE warp (num_warps=1: every 1-D op below is a
        register / warp-shuffle op; the [KP, KP] compare matrices are gone except psum's group ranks). Keys
        id * KP + i (unique, so the bitonic tl.sort is the stable sort-by-id of dedup_routes); run starts by
        comparing each sorted id with its neighbour through the int32 scratch row `scr` (store, bar.sync, load);
        slot = cumsum(first) - 1; each slot's first position -> rank, count (next first - this first) and id, now
        laid out one slot per lane. INPL (dg-inplace): the compact cold index (cumsum), stage ids, sidx, and
        (M1-T10-MOE) the overflow count (one atomic), the staging work list `wl` (staged cold slots c = 0.., then --
        STEP_SF -- the used hot slots; -1 after), the m_indices tags of the contiguous layout straight into m_indices
        (row stride TS; HAS_TAGS), and the per-slot A block of the psum (LAYOUT 1: ok slots by ascending group, the
        G-entry psum written to gl) / masked (LAYOUT 2: block = group, masked_m [G] zeroed then scattered to gl)
        layouts -> arow = block * BR + rank (masked: pairs of an overflowing slot -> scratch block GSCR). posmat by
        a second sort of (row, sorted position) keys carrying arow in the low 16 bits."""
        t = tl.arange(0, KP)
        okt = t < P
        v = tl.load(idx + t, mask=okt, other=0).to(tl.int32)
        s = tl.sort(tl.where(okt, v * KP + t, (1 << 30) + t))
        sid = s // KP                                         # expert id of sorted pair t (huge on padding lanes)
        perm = s % KP                                         # its flat index r*K + j
        tl.store(scr + t, sid)
        tl.debug_barrier()
        prev = tl.load(scr + t - 1, mask=t > 0, other=-1)
        first = okt & ((t == 0) | (sid != prev))
        fi = first.to(tl.int32)
        slot = tl.cumsum(fi, 0) - 1
        n_u = tl.sum(fi, 0)
        vmax = tl.max(tl.where(okt, sid, -1), 0)
        fp = scr + KP                                         # first sorted position of slot u; P from u = U on
        tl.store(fp + slot, t, mask=first)
        tl.store(fp + t, t * 0 + P, mask=t >= n_u)
        tl.store(fp + KP, P)
        tl.debug_barrier()
        rank = t - tl.load(fp + tl.where(okt, slot, 0))
        u = t                                                 # slot lanes from here on
        oku = u < P
        used = u < n_u
        fpu = tl.load(fp + u)
        counts = tl.where(used, tl.load(fp + u + 1) - fpu, 0)
        uniq = tl.where(used, tl.load(scr + tl.where(used, fpu, 0)), vmax)
        valid = (counts > 0).to(tl.int32)
        tl.store(uniq_o + u, uniq, mask=oku)
        tl.store(counts_o + u, counts, mask=oku)
        tl.store(valid_o + u, valid, mask=oku)
        if HAS_MM:
            tl.store(mm_o + u, counts, mask=oku)
        if HAS_CTR:
            tl.store(ids_o + u, uniq, mask=oku)
            tl.store(uval_o + u, valid, mask=oku)
        if INPL:
            gq = tl.load(gtab + uniq, mask=oku, other=0).to(tl.int32)
            cold = (gq < 0) & (counts > 0)
            ci = tl.cumsum(cold.to(tl.int32), 0) - 1
            sidx = tl.where(cold, ci, -1)
            grp = tl.where(cold, ci + H, gq)
            okc = sidx < S
            okused = okc & (counts > 0)
            n_ok = tl.where(okc, counts, 0)
            if HAS_DUMMY:
                stage_ids = tl.where(counts > 0, uniq, tl.load(dummy).to(tl.int32))
            else:
                stage_ids = uniq
            tl.store(stage_o + u, stage_ids, mask=oku)
            tl.store(sidx_o + u, sidx, mask=oku)
            if HAS_OVF:
                n_over = tl.sum((cold & (sidx >= S)).to(tl.int32), 0)
                if n_over > 0:
                    tl.atomic_add(ovf, n_over)
            if HAS_WL:
                staged = cold & okc
                n_st = tl.sum(staged.to(tl.int32), 0)
                if STEP_SF:
                    hu = (gq >= 0) & (counts > 0)
                    wpos = tl.where(staged, ci, n_st + tl.cumsum(hu.to(tl.int32), 0) - 1)
                    inw = staged | hu
                    n_w = n_st + tl.sum(hu.to(tl.int32), 0)
                else:
                    wpos = ci
                    inw = staged
                    n_w = n_st
                tl.store(wl_o + wpos, u, mask=inw)
                tl.store(wl_o + u, u * 0 - 1, mask=oku & (u >= n_w))
            if HAS_TAGS:
                rr = tl.arange(0, NRP)
                tag = tl.where(rr[None, :] < n_ok[:, None], grp[:, None], -1)
                tl.store(tags_o + u[:, None] * TS + rr[None, :], tag, mask=oku[:, None] & (rr[None, :] < NR))
            if LAYOUT == 1:
                ga = tl.where(okused, grp, 1 << 30)
                b_ok = tl.sum((ga[None, :] < ga[:, None]).to(tl.int32), 1)
                nou = oku & (ga >= (1 << 30))
                b_no = tl.sum(okused.to(tl.int32), 0) + tl.cumsum(nou.to(tl.int32), 0) - 1
                blk = tl.where(okused, b_ok, b_no)
                for g0 in range(0, G, KP):
                    gg = g0 + tl.arange(0, KP)
                    nb = tl.sum((ga[None, :] < gg[:, None]).to(tl.int32), 1)
                    cg = tl.sum(tl.where(ga[None, :] == gg[:, None], counts[None, :], 0), 1)
                    tl.store(gl_o + gg, nb * BR + cg, mask=gg < G)
            elif LAYOUT == 2:
                for g1 in range(0, G, KP):
                    gz = g1 + tl.arange(0, KP)
                    tl.store(gl_o + gz, gz * 0, mask=gz < G)
                tl.debug_barrier()
                tl.store(gl_o + tl.where(okused, grp, 0), counts, mask=okused)
                blk = tl.where(okused, grp, -1)
            else:
                blk = u
            tl.store(scr + 2 * KP + 1 + u, blk)
            tl.debug_barrier()
            bt = tl.load(scr + 2 * KP + 1 + tl.where(okt, slot, 0))
            if LAYOUT == 2:
                arow = tl.where(bt >= 0, bt * BR + rank, GSCR * BR + t)
            else:
                arow = bt * BR + rank
        else:
            arow = slot * BR + rank
        prow = perm // K
        tl.store(slot_o + t, slot, mask=okt)
        tl.store(rank_o + t, rank, mask=okt)
        tl.store(arow_o + t, arow, mask=okt)
        tl.store(prow_o + t, prow, mask=okt)
        tl.store(pw_o + t, tl.load(w + perm, mask=okt, other=0.0), mask=okt)
        s2 = tl.sort(tl.where(okt, ((prow * KP + t) << 16) | arow, (1 << 30) + t))
        tl.store(posmat_o + t, s2 & 0xFFFF, mask=okt)


def _np2(n: int) -> int:
    return 1 if int(n) <= 1 else 1 << (int(n) - 1).bit_length()


# ------------------------------------------------------------------------------------------------ M1-T10-MOE layouts
LAYOUT_IDS = {"contiguous": 0, "psum": 1, "masked": 2}   # = runtime/dg_inplace.ROWS_LAYOUTS
DEDUP_IMPLS = ("warp", "matrix")      # warp: m1_dedup_warp (default); matrix: the M1-T10-PORT m1_dedup_routes
DEDUP_IMPL_ENV = "DSV41_DEDUP_IMPL"
ARow_BITS = 16                        # posmat's second sort carries arow in the low 16 bits
_WARP = {"error": None}               # first failure of a default-path warp launch: the process then uses matrix


def dedup_impl() -> str:
    v = os.environ.get(DEDUP_IMPL_ENV, "warp")
    if v not in DEDUP_IMPLS:
        raise ValueError(f"{DEDUP_IMPL_ENV}={v!r} not in {DEDUP_IMPLS}")
    return v


def dedup_describe() -> dict:
    return {"impl": "matrix" if _WARP["error"] else dedup_impl(), "warp_error": _WARP["error"]}


def rows_layout(m: Dict[str, torch.Tensor], inpl: dict, block_rows: int, R_: int, k: int) -> Dict[str, torch.Tensor]:
    """The dg-inplace extras of a base mapping m (dedup_routes' keys) in torch device ops (no host sync, graph-
    capturable): the CPU mirror of m1_dedup_warp's INPL part and the CUDA fallback behind the matrix kernel.
    inpl: gtab, H, S, dummy, has_dummy, and (M1-T10-MOE) layout ("contiguous" | "psum" | "masked"), G, gscr,
    step_sf. Returns sidx, ids_stage, tags [P, R] (contiguous), wl [P], n_over (0-d), gl [G] (psum / masked) or
    None, and arow [P] / posmat [R, k] of the layout."""
    uniq, counts, slot, rank, prow = m["uniq"], m["counts"], m["slot"], m["rank"], m["prow"]
    P = uniq.numel()
    dev = uniq.device
    H, S, BR = int(inpl["H"]), int(inpl["S"]), int(block_rows)
    lay = inpl.get("layout", "contiguous")
    g = inpl["gtab"].index_select(0, uniq).long()
    cold, ok, grp, sidx = compact_cold(g, counts, H, S)
    used = counts > 0
    okused = used & ok
    ar = torch.arange(P, dtype=torch.long, device=dev)
    out = {"sidx": sidx, "n_over": (cold & ~ok).sum(),
           "ids_stage": torch.where(used, uniq, inpl["dummy"]) if inpl["has_dummy"] else uniq.clone()}
    n_ok = torch.where(ok, counts, 0)
    rr = torch.arange(R_, dtype=torch.long, device=dev).view(1, -1)
    out["tags"] = torch.where(rr < n_ok.view(-1, 1), grp.view(-1, 1), -1).to(torch.int32)
    staged = cold & ok
    n_st = staged.sum()
    c = torch.cumsum(cold.long(), 0) - 1
    if inpl.get("step_sf"):
        hu = used & (g >= 0)
        pos = torch.where(staged, c, n_st + torch.cumsum(hu.long(), 0) - 1)
        inw = staged | hu
    else:
        pos = c
        inw = staged
    wl = torch.full((P + 1,), -1, dtype=torch.long, device=dev)
    wl.scatter_(0, torch.where(inw, pos, P), torch.where(inw, ar, -1))
    out["wl"] = wl[:P]
    G = int(inpl.get("G") or 0)
    big = 1 << 30
    if lay == "psum":
        ga = torch.where(okused, grp, big)
        b_ok = (ga.view(1, -1) < ga.view(-1, 1)).long().sum(1)
        b_no = okused.long().sum() + torch.cumsum((~okused).long(), 0) - 1
        blk = torch.where(okused, b_ok, b_no)
        gg = torch.arange(G, dtype=torch.long, device=dev).view(-1, 1)
        nb = (ga.view(1, -1) < gg).long().sum(1)
        cg = torch.where(ga.view(1, -1) == gg, counts.view(1, -1), 0).sum(1)
        out["gl"] = (nb * BR + cg).to(torch.int32)
    elif lay == "masked":
        blk = torch.where(okused, grp, -1)
        gl = torch.zeros(G + 1, dtype=torch.long, device=dev)
        gl.scatter_(0, torch.where(okused, grp, G), torch.where(okused, counts, 0))
        out["gl"] = gl[:G].to(torch.int32)
    else:
        blk = ar
        out["gl"] = None
    bt = blk.index_select(0, slot)
    if lay == "masked":
        arow = torch.where(bt >= 0, bt * BR + rank, int(inpl["gscr"]) * BR + ar)
    else:
        arow = bt * BR + rank
    out["arow"] = arow
    order = torch.argsort(prow * P + ar)          # row-major, then sorted position (= ascending id within a row)
    out["posmat"] = arow.index_select(0, order).view(R_, k)
    return out


def _check_arow_range(inpl: Optional[dict], block_rows: int, P: int) -> None:
    lay = (inpl or {}).get("layout", "contiguous")
    top = (int(inpl["gscr"]) + 1) * int(block_rows) if lay == "masked" else _np2(P) * int(block_rows)
    if top > (1 << ARow_BITS):
        raise ValueError(f"dedup: A rows up to {top} do not fit posmat's {ARow_BITS}-bit sort key")


def dedup_fused(buf: DedupBuffers, indices: torch.Tensor, weights: torch.Tensor, block_rows: int, *, mm=None,
                ids_step=None, uval=None, inpl: Optional[dict] = None, impl: Optional[str] = None
                ) -> Dict[str, torch.Tensor]:
    """dedup_routes (+ masked_m `mm`, the counters' rows, and with `inpl` the dg-inplace staging inputs) into the
    static buffers `buf`: ONE launch on CUDA / under the shim -- m1_dedup_warp (default, $DSV41_DEDUP_IMPL) or the
    M1-T10-PORT m1_dedup_routes ("matrix") -- else the torch functions (CPU mirror). Returns dedup_routes' dict
    (views of `buf`) + mm_written, and with inpl: ids_stage, sidx, tags.
    M1-T10-MOE, inpl["v2"] (the R-row DGInplaceMoERows): also the staging work list `wl`, the overflow count added
    to inpl["ovf"], the contiguous tags written straight into inpl["mi"] (row stride block_rows; else buf.tags),
    the psum / masked_m vector into inpl["gl"][:G], and arow / posmat of inpl["layout"]; returns layout_done.
    The matrix kernel (or a warp launch that fails to compile on the default path: reported once, then matrix for
    the process) does the base mapping and rows_layout() the extras in torch device ops."""
    R_, k = indices.shape
    P = R_ * k
    if P != buf.P or P > DEDUP_MAX_PAIRS:
        raise ValueError(f"dedup_fused: {R_}x{k} pairs vs buffers of {buf.P} (max {DEDUP_MAX_PAIRS})")
    idx = indices.reshape(-1).contiguous()
    wt = weights.reshape(-1).float().contiguous()
    out = {"uniq": buf.uniq, "counts": buf.counts, "valid": buf.valid, "slot": buf.slot, "rank": buf.rank,
           "prow": buf.prow, "pw": buf.pw, "arow": buf.arow, "posmat": buf.posmat, "mm_written": mm is not None}
    v2 = bool(inpl is not None and inpl.get("v2"))
    lay = inpl.get("layout", "contiguous") if v2 else "contiguous"
    mi = inpl.get("mi") if v2 else None
    if inpl is not None:
        out.update(ids_stage=buf.ids_stage, sidx=buf.sidx)
        if mi is None:
            out["tags"] = buf.tags
    if v2:
        out.update(wl=buf.wl, layout_done=True)
        _check_arow_range(inpl, block_rows, P)
    kern = HAVE_TRITON and (idx.is_cuda or KERNELS_ON_CPU)
    explicit = impl is not None or DEDUP_IMPL_ENV in os.environ
    use = impl or ("matrix" if _WARP["error"] else dedup_impl())
    if kern and use == "warp":
        try:
            _launch_warp(buf, idx, wt, R_, k, block_rows, mm, ids_step, uval, inpl, v2, lay, mi)
            return out
        except Exception as e:  # compile failure on this Triton: report once, fall back to the matrix kernel
            if explicit:
                raise
            _WARP["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
            print(f"[moe_rows] m1_dedup_warp failed ({_WARP['error']}); using m1_dedup_routes from now on",
                  file=sys.stderr, flush=True)
    if kern:
        dmy = inpl["dummy"] if inpl is not None else idx
        m1_dedup_routes[(1,)](
            idx, wt, buf.uniq, buf.counts, buf.valid, buf.slot, buf.rank, buf.prow, buf.pw, buf.arow, buf.posmat,
            mm if mm is not None else buf.counts, ids_step if ids_step is not None else buf.uniq,
            uval if uval is not None else buf.valid, inpl["gtab"] if inpl is not None else idx,
            buf.ids_stage, buf.sidx, buf.tags, dmy.reshape(-1),
            P, k, R_, int(block_rows), int(inpl["H"]) if inpl is not None else 0,
            int(inpl["S"]) if inpl is not None else 0,
            KP=_np2(P), NRP=_np2(R_), HAS_MM=mm is not None, HAS_CTR=ids_step is not None,
            INPL=inpl is not None, HAS_DUMMY=bool(inpl is not None and inpl["has_dummy"]), num_warps=4)
        if v2:
            _apply_extras(buf, out, rows_layout(out, inpl, block_rows, R_, k), inpl, mi, block_rows, R_)
        return out
    m = dedup_routes(idx.view(R_, k), wt.view(R_, k), block_rows)
    for key in ("uniq", "counts", "valid", "slot", "rank", "prow", "pw", "arow", "posmat"):
        out[key].copy_(m[key])
    if mm is not None:
        mm.copy_(m["counts"].to(mm.dtype))
    if ids_step is not None:
        ids_step.copy_(m["uniq"])
        uval.copy_(m["valid"])
    if inpl is not None:
        ext = rows_layout(m, dict(inpl, layout=lay) if v2 else dict(inpl, layout="contiguous"), block_rows, R_, k)
        buf.ids_stage.copy_(ext["ids_stage"])
        buf.sidx.copy_(ext["sidx"])
        if v2:
            _apply_extras(buf, out, ext, inpl, mi, block_rows, R_)
        else:
            buf.tags.copy_(ext["tags"])
    return out


def _apply_extras(buf, out, ext, inpl, mi, block_rows: int, R_: int) -> None:
    """rows_layout()'s results into the static buffers and the DeepGEMM inputs (device copies)."""
    buf.wl.copy_(ext["wl"])
    buf.arow.copy_(ext["arow"])
    buf.posmat.copy_(ext["posmat"])
    if mi is None:
        buf.tags.copy_(ext["tags"])
    _write_layout(ext, dict(inpl, mi=mi), block_rows, R_)


def _launch_warp(buf, idx, wt, R_, k, block_rows, mm, ids_step, uval, inpl, v2, lay, mi) -> None:
    P = R_ * k
    has_in = inpl is not None
    dmy = inpl["dummy"] if has_in else idx
    tags_t, ts = (mi, int(block_rows)) if mi is not None else (buf.tags, R_)
    gl = inpl.get("gl") if v2 else None
    ovf = inpl.get("ovf") if v2 else None
    m1_dedup_warp[(1,)](
        idx, wt, buf.uniq, buf.counts, buf.valid, buf.slot, buf.rank, buf.prow, buf.pw, buf.arow, buf.posmat,
        mm if mm is not None else buf.counts, ids_step if ids_step is not None else buf.uniq,
        uval if uval is not None else buf.valid, inpl["gtab"] if has_in else idx,
        buf.ids_stage, buf.sidx, tags_t, buf.wl, gl if gl is not None else buf.scr,
        ovf if ovf is not None else buf.scr, buf.scr, dmy.reshape(-1),
        P, k, R_, int(block_rows), int(inpl["H"]) if has_in else 0, int(inpl["S"]) if has_in else 0, ts,
        int(inpl.get("G") or 0) if v2 else 0, int(inpl.get("gscr") or 0) if v2 else 0,
        KP=buf.KP, NRP=_np2(R_), HAS_MM=mm is not None, HAS_CTR=ids_step is not None, INPL=has_in,
        HAS_DUMMY=bool(has_in and inpl["has_dummy"]), LAYOUT=LAYOUT_IDS[lay] if v2 else 0,
        HAS_TAGS=has_in and (not v2 or lay == "contiguous"), HAS_WL=v2, STEP_SF=bool(v2 and inpl.get("step_sf")),
        HAS_OVF=ovf is not None, num_warps=1)


def combine_rows(out_rows: torch.Tensor, posmat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y [R, D] fp32 = 0 + out_rows[posmat[:, 0]] + ... + out_rows[posmat[:, k-1]] (fp32 += bf16, ascending expert id
    per row = model.py:895-902's order)."""
    y = torch.zeros_like(x, dtype=torch.float32)
    for j in range(posmat.size(1)):
        y += out_rows.index_select(0, posmat[:, j])
    return y


def _rows2d(d: torch.Tensor) -> torch.Tensor:
    """A GEMM output [blocks, rows, N] as its A rows [blocks*rows, N] (the layouts' arow index it)."""
    return d.view(-1, d.size(-1))


class _RowsMoEBase:
    def __init__(self, owner, moe, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]):
        self.owner, self.moe, self.layer, self.ptabs = owner, moe, layer, ptabs
        self.last: Dict[str, torch.Tensor] = {}  # the last call's mapping (tests / census; eager only)
        # M1-T10-PORT: this layer's static dedup outputs (dedup_fused; allocated at build, never under capture)
        self.dbuf = DedupBuffers(owner.rows, owner.topk, owner.dev)

    @property
    def active(self) -> bool:
        return self.owner.active

    def _route(self, x: torch.Tensor, block_rows: int, mm=None, inpl: Optional[dict] = None
               ) -> Dict[str, torch.Tensor]:
        """Gate + dedup. M1-T10-PORT: glue `router` -> FusedGlue.route_rows (ids ascending per row: the dedup is
        indifferent to the order inside a row, see dedup_routes); glue `dedup` -> dedup_fused (one launch: the mapping,
        masked_m `mm`, the counters and -- `inpl`, dg-inplace -- the compact cold index, stage ids and tags)."""
        o = self.owner
        g = o.glue
        if g is not None and g.on("router"):
            indices, weights = g.route_rows(self.moe, x)              # model.py:894, 2 fused launches + 3 kept
        else:
            weights, indices = self.moe.gate(x, None)                 # model.py:894 at R rows (text: no image mask)
        if g is not None and g.on("dedup"):                            # M1-T10-PORT #3: the mapping in 1 launch
            m = dedup_fused(self.dbuf, indices, weights, block_rows, mm=mm,
                            ids_step=o.ids_step[self.layer] if o.counters else None,
                            uval=o.uvalid_step[self.layer] if o.counters else None, inpl=inpl)
        else:
            m = dedup_routes(indices, weights, block_rows)
            if o.counters:
                o.ids_step[self.layer].copy_(m["uniq"])
                o.uvalid_step[self.layer].copy_(m["valid"])
        self.last = m
        return m

    def _gather(self, uniq: torch.Tensor) -> None:
        from runtime.static_kernels import gather_rows

        o = self.owner
        for key, tab in self.ptabs.items():                            # one launch per (proj, weight|scale), S slots
            gather_rows(tab, uniq, o.staging[key], o.row_bytes[key])

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        glue = self.owner.glue
        if glue is not None and glue.on("swiglu_shared"):
            return glue.shared_expert(self.moe.shared_experts, x)      # fused SwiGLU (row-parallel, self-checked)
        return self.moe.shared_experts(x)                              # model.py:905 at M = R


class StaticMoERows(_RowsMoEBase):
    """moe_impl "ref" at R rows: per slot one reference Expert evaluation on its [R, D] block."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        o, moe = self.owner, self.moe
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893, [R, D]
        R_, D = x.shape
        S = o.n_slots
        m = self._route(x, R_)
        if o.glue is not None and o.glue.on("gather"):                 # M1-T10-PORT: the 6 tables in 1 launch
            o.glue.gather(self, m["uniq"])
        else:
            self._gather(m["uniq"])
        bufs = getattr(o, "_rows_moe_bufs", None)
        if bufs is None or bufs[0].dtype != x.dtype:
            bufs = o._rows_moe_bufs = (torch.zeros(S * R_, D, dtype=x.dtype, device=x.device),
                                       torch.zeros(S * R_, 1, dtype=torch.float32, device=x.device))
        xb, wb = bufs
        xb.index_copy_(0, m["arow"], x.index_select(0, m["prow"]))      # pair (r, e) -> row `rank` of slot u
        wb.index_copy_(0, m["arow"], m["pw"].view(-1, 1))
        xv, wv = xb.view(S, R_, D), wb.view(S, R_, 1)
        outs = [ex(xv[u], wv[u]) for u, ex in enumerate(o.staged_experts)]  # model.py:902 (M = R rows per slot)
        y = combine_rows(torch.stack(outs).view(S * R_, -1), m["posmat"], x)
        y += self._shared(x)                                           # model.py:905
        return y.type_as(x).view(shape)                                # model.py:906


class DGMaskedMoERows(_RowsMoEBase):
    """moe_impl "dg-masked" at R rows: 3 DeepGEMM masked grouped GEMMs over R*k groups, masked_m = rows per slot."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        o, moe, b, R = self.owner, self.moe, self.owner.dg, self.owner.R
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893
        glue = o.glue
        M = self._block_rows()
        m = self._route(x, M, **self._route_kw())
        m = self._weights(m) or m                                      # M1-T10-MOE: dg-inplace may re-lay arow
        arow, prow = m["arow"], m["prow"]
        if glue is not None and glue.on("dg_act"):                     # M1-T10-PORT: act_quant + scatter, 1 launch
            glue.dg_act_rows(x, m)
        else:
            q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)   # model.py:187, the R rows once
            self._load(1, q.index_select(0, prow), s.index_select(0, prow), arow)
        if glue is not None and glue.on("swiglu_routed"):              # M1-T10-PORT: gather+glue+act_quant, 1 launch
            g1 = _rows2d(self._gemm("w1", 1))                                      # model.py:845
            g3 = _rows2d(self._gemm("w3", 1))                                      # model.py:846
            glue.routed_swiglu_rows(g1, g3, m)
        else:
            gate = _rows2d(self._gemm("w1", 1)).index_select(0, arow).float()          # model.py:845
            up = _rows2d(self._gemm("w3", 1)).index_select(0, arow).float()            # model.py:846
            lim = b.swiglu_limit
            if lim > 0:                                                    # model.py:847-849
                up = torch.clamp(up, min=-lim, max=lim)
                gate = torch.clamp(gate, max=lim)
            h = F.silu(gate) * up                                          # model.py:850
            h = m["pw"].view(-1, 1) * h                                    # model.py:852
            h = h.to(x.dtype)                                              # model.py:853
            q2, s2 = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)  # model.py:187 inside w2
            self._load(2, q2, s2, arow)
        out = _rows2d(self._gemm("w2", 2))                            # [A rows, D] bf16
        if glue is not None and glue.on("moe_combine"):                # M1-T10-PORT: combine + shared + bf16, 1 launch
            return glue.combine_rows(out, m["posmat"], self._shared(x), x).view(shape)
        y = combine_rows(out, m["posmat"], x)
        y += self._shared(x)
        return y.type_as(x).view(shape)

    def _route_kw(self) -> dict:
        return {"mm": self.owner.dg.masked_m}                          # dedup_fused writes masked_m (dg-masked)

    # the two places dg-masked and dg-inplace differ (DGInplaceMoERows below)
    def _block_rows(self) -> int:
        return self.owner.dg.rows                                      # masked: max_m rows per group

    def _weights(self, m: Dict[str, torch.Tensor]) -> None:
        """Slot u's expert m["uniq"][u] -> staging slot u (weights + e8m0 scales), scales -> fp32 (sf32)."""
        o, b = self.owner, self.owner.dg
        if o.glue is not None and o.glue.on("gather"):                 # M1-T10-PORT: 6 tables + fp32 scales, 1 launch
            o.glue.gather(self, m["uniq"])
        else:
            self._gather(m["uniq"])
            for p in PROJS:                                            # e8m0 -> fp32, one op per projection
                b.convert_scales(p, o.staging[(p, "scale")])
        if not m.get("mm_written"):
            b.masked_m.copy_(m["counts"].to(torch.int32))

    def _gemm(self, p: str, which: int) -> torch.Tensor:
        o = self.owner
        return o.dg.gemm(p, o.staging[(p, "weight")], which)

    def _load(self, which: int, q: torch.Tensor, s: torch.Tensor, arow: torch.Tensor) -> None:
        """A rows `arow` of xq[which] / xs[which] <- the pairs' quantized activation (fp8) and fp32 scales."""
        from runtime.dg_moe import e8m0_to_f32

        b = self.owner.dg
        K = q.size(-1)
        b.xq[which].view(torch.uint8).view(-1, K).index_copy_(0, arow, q.contiguous().view(torch.uint8))  # bytes
        s32 = torch.empty(s.shape, dtype=torch.float32, device=s.device)
        e8m0_to_f32(s, s32, b.e8m0_method)
        b.xs[which].view(-1, s32.size(-1)).index_copy_(0, arow, s32)


class DGInplaceMoERows(DGMaskedMoERows):
    """moe_impl "dg-inplace" at R rows (M1-T10-FIX, M1-T10-FIX2): DeepGEMM's CONTIGUOUS grouped GEMM over the layer's
    whole pool [H + S', N, K/2] (S' >= S spare slots), the same dedup as dg-masked. Slot u (expert uniq[u], ascending
    id) is A block u = rows [u*ALIGN, u*ALIGN + counts[u]) (pair (r, e) at row u*ALIGN + rank, rows ascending),
    group = the expert's hot pool slot, or -- a cold expert -- spare slot H + c with the COMPACT cold index
    c = cumsum(is_cold over the used slots in ascending-id order) - 1 (FIX2; FIX used H + u, which needed R*k = 36
    spare slots per layer = 25.5 GB on the real model and OOMed next to the 1-row step). S = owner.spare_slots
    (default 8; the seed map routes ~0.3 cold unique experts per layer per step). The staging launch copies cold
    expert c into H + c; m_indices tags each block's count rows with its group and everything else -1 (padding rows
    and unused slots u >= U): each row's k pairs reach their k groups. Combine = per-row ascending expert id
    (posmat), the SwiGLU glue and act_quant are dg-masked's on the same pair rows, so the only change vs dg-masked
    is the GEMM's weight source (in place vs a staged copy) and vs FIX only the spare index (same bytes, same
    GEMM rows): bit-equal to both.
    OVERFLOW: a layer with more than S cold unique experts in a step stages only the first S (ascending id); the
    rest are NOT staged, their blocks are tagged -1 (the GEMM skips them: no out-of-bounds read) and the kernel adds
    one per such expert to the step's device counter `owner.ovf_step` -- that layer's results (hence the step's
    logits) are WRONG for that step, never silently: StaticStep.overflow_counts() / first_light's `overflow steps`.
    Cost: S spare weight slots per layer (40 x 8 x 17.7 MB = 5.7 GB on the real model at S = 8).
    M1-T10-MOE: the dedup (m1_dedup_warp, or its torch mirror rows_layout) also writes m_indices / the psum or
    masked_m vector, the overflow count and the staging work list, and lays out arow / posmat for the build's GEMM
    call shape (DGInplaceBuffers.layout: contiguous at any ALIGN >= R, psum, masked); the ONE staging launch
    (static_kernels.stage_rows) then moves only the staged cold experts' bytes + scales (and, sf_store "step", the
    used hot experts' scales once each). Same bytes into the same GEMM rows' groups: bit-equal by construction for
    contiguous at ALIGN 128; the other call shapes are checked at build (DGInplaceBuffers.check_layout)."""

    def __init__(self, owner, moe, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]):
        super().__init__(owner, moe, layer, ptabs)
        b = owner.dg
        self.t = b.layer_tables(layer, ptabs)
        pool = b.pool
        d = owner.dev
        hot = pool.hot[layer]
        # a hot expert stands in for unused slots in the staging launch (no weight copy; its scales re-written with
        # the same bytes); none hot -> the slot's own (duplicate) id with compact index -1 (skipped by the launch)
        self.dummy = torch.tensor(int(hot[0]) if len(hot) else -1, dtype=torch.long, device=d)
        self.has_dummy = len(hot) > 0
        self.ar_rows = torch.arange(owner.rows, dtype=torch.long, device=d).view(1, -1)
        self.S = int(b.spare_limit)

    def _block_rows(self) -> int:
        return self.owner.dg.block_rows

    def _weights(self, m: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """M1-T10-MOE: the layout's m_indices / gl, the overflow count, arow / posmat and the work list are the
        dedup's (dedup_fused, layout_done); the reference dedup (glue `dedup` off) gets them from rows_layout here.
        Then ONE staging launch (cold bytes + scales from the work list)."""
        o, b, t = self.owner, self.owner.dg, self.t
        if not m.get("layout_done"):
            inpl = self._inpl()
            R_ = o.rows
            ext = rows_layout(m, inpl, b.block_rows, R_, m["posmat"].size(1))
            m = _with_extras(m, ext)
            _write_layout(ext, inpl, b.block_rows, R_)
            self.last = m
        b.stage_rows(t, m)                                             # 1 launch: staged cold c -> H + c (+ SF)
        return m

    def _gemm(self, p: str, which: int) -> torch.Tensor:
        return self.owner.dg.gemm_rows(p, self.t, which)

    def _inpl(self) -> dict:
        o, b, t = self.owner, self.owner.dg, self.t
        return {"gtab": t["gtab"], "H": int(t["H"]), "S": self.S, "dummy": self.dummy, "has_dummy": self.has_dummy,
                "v2": True, "layout": b.layout, "mi": b.mi, "gl": b.gl, "G": int(t["G"]), "gscr": b.gscr,
                "ovf": getattr(o, "ovf_step", None), "step_sf": b.sf_store == "step"}

    def _route_kw(self) -> dict:
        """dedup_fused's dg-inplace extras: the layer's hot-slot table, H, the spare budget, the stand-in id, and
        (M1-T10-MOE, v2) the layout, m_indices / gl, the scratch block, the overflow counter, the store."""
        return {"inpl": self._inpl()}


def _with_extras(m: Dict[str, torch.Tensor], ext: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(m)
    out.update(arow=ext["arow"], posmat=ext["posmat"], wl=ext["wl"], sidx=ext["sidx"], ids_stage=ext["ids_stage"],
               layout_done=True)
    return out


def _write_layout(ext: Dict[str, torch.Tensor], inpl: dict, block_rows: int, R_: int) -> None:
    """rows_layout()'s m_indices tags / gl / overflow count into the DeepGEMM inputs (device ops)."""
    if inpl.get("mi") is not None:
        P = ext["tags"].size(0)
        inpl["mi"].view(-1)[: P * int(block_rows)].view(P, int(block_rows))[:, :R_].copy_(ext["tags"])
    if ext["gl"] is not None:
        inpl["gl"][: ext["gl"].numel()].copy_(ext["gl"])
    if inpl.get("ovf") is not None:
        inpl["ovf"].add_(ext["n_over"].to(inpl["ovf"].dtype))


def compact_cold(g: torch.Tensor, counts: torch.Tensor, H: int, S: int):
    """M1-T10-FIX2's compact cold index, device ops only (no host sync). g [U'] = gtab[uniq] (hot pool slot or -1),
    counts [U'] (0 = unused slot) -> (cold [U'] bool = a used slot whose expert is cold; ok [U'] bool = not an
    overflowing cold slot; grp [U'] int64 = hot slot, or H + c for cold; sidx [U'] int64 = c for cold slots, -1
    otherwise), c = cumsum(cold) - 1 over the slots in order (ascending expert id), cold slot staged iff c < S."""
    cold = (g < 0) & (counts > 0)
    c = torch.cumsum(cold, 0) - 1                       # bool cumsum -> int64
    sidx = torch.where(cold, c, -1)
    grp = torch.where(cold, c + int(H), g)
    ok = sidx < int(S)                                  # sidx -1 (not cold) is always ok
    return cold, ok, grp, sidx


def compact_cold_host(gtab, uniq, counts, H: int, S: int) -> dict:
    """compact_cold from plain Python (tests): {"grp": [...], "sidx": [...], "ok": [...], "overflow": n}."""
    out = {"grp": [], "sidx": [], "ok": [], "overflow": 0}
    c = -1
    for e, n in zip(uniq, counts):
        g = int(gtab[int(e)])
        if g < 0 and n > 0:
            c += 1
            out["grp"].append(H + c)
            out["sidx"].append(c)
            out["ok"].append(c < S)
            out["overflow"] += int(c >= S)
        else:
            out["grp"].append(g)  # hot slot, or -1 for an unused slot's cold duplicate (never tagged)
            out["sidx"].append(-1)
            out["ok"].append(True)
    return out
