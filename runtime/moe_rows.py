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
"""
from __future__ import annotations

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
    """One layer's static outputs of the fused dedup (graph-stable addresses; allocated at build). P = R*k pairs."""

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


def _np2(n: int) -> int:
    return 1 if int(n) <= 1 else 1 << (int(n) - 1).bit_length()


def dedup_fused(buf: DedupBuffers, indices: torch.Tensor, weights: torch.Tensor, block_rows: int, *, mm=None,
                ids_step=None, uval=None, inpl: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    """dedup_routes (+ masked_m `mm`, the counters' rows, and with `inpl` the dg-inplace staging inputs) into the
    static buffers `buf`: ONE launch (m1_dedup_routes) on CUDA / under the shim, else the torch functions (CPU
    mirror). Returns dedup_routes' dict (views of `buf`) + mm_written, and with inpl: ids_stage, sidx, tags."""
    R_, k = indices.shape
    P = R_ * k
    if P != buf.P or P > DEDUP_MAX_PAIRS:
        raise ValueError(f"dedup_fused: {R_}x{k} pairs vs buffers of {buf.P} (max {DEDUP_MAX_PAIRS})")
    idx = indices.reshape(-1).contiguous()
    wt = weights.reshape(-1).float().contiguous()
    out = {"uniq": buf.uniq, "counts": buf.counts, "valid": buf.valid, "slot": buf.slot, "rank": buf.rank,
           "prow": buf.prow, "pw": buf.pw, "arow": buf.arow, "posmat": buf.posmat, "mm_written": mm is not None}
    if inpl is not None:
        out.update(ids_stage=buf.ids_stage, sidx=buf.sidx, tags=buf.tags)
    if HAVE_TRITON and (idx.is_cuda or KERNELS_ON_CPU):
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
        uniq, counts = m["uniq"], m["counts"]
        g = inpl["gtab"].index_select(0, uniq).long()
        cold, ok, grp, sidx = compact_cold(g, counts, int(inpl["H"]), int(inpl["S"]))
        buf.ids_stage.copy_(torch.where(counts > 0, uniq, inpl["dummy"]) if inpl["has_dummy"] else uniq)
        buf.sidx.copy_(sidx)
        n_ok = torch.where(ok, counts, 0)
        ar = torch.arange(R_, dtype=torch.long, device=idx.device).view(1, -1)
        buf.tags.copy_(torch.where(ar < n_ok.view(-1, 1), grp.view(-1, 1), -1).to(torch.int32))
    return out


def combine_rows(out_rows: torch.Tensor, posmat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y [R, D] fp32 = 0 + out_rows[posmat[:, 0]] + ... + out_rows[posmat[:, k-1]] (fp32 += bf16, ascending expert id
    per row = model.py:895-902's order)."""
    y = torch.zeros_like(x, dtype=torch.float32)
    for j in range(posmat.size(1)):
        y += out_rows.index_select(0, posmat[:, j])
    return y


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
        S, M = b.k, self._block_rows()
        m = self._route(x, M, **self._route_kw())
        self._weights(m)
        arow, prow = m["arow"], m["prow"]
        if glue is not None and glue.on("dg_act"):                     # M1-T10-PORT: act_quant + scatter, 1 launch
            glue.dg_act_rows(x, m)
        else:
            q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)   # model.py:187, the R rows once
            self._load(1, q.index_select(0, prow), s.index_select(0, prow), arow)
        if glue is not None and glue.on("swiglu_routed"):              # M1-T10-PORT: gather+glue+act_quant, 1 launch
            g1 = self._gemm("w1", 1).view(S * M, -1)                               # model.py:845
            g3 = self._gemm("w3", 1).view(S * M, -1)                               # model.py:846
            glue.routed_swiglu_rows(g1, g3, m)
        else:
            gate = self._gemm("w1", 1).view(S * M, -1).index_select(0, arow).float()   # model.py:845
            up = self._gemm("w3", 1).view(S * M, -1).index_select(0, arow).float()     # model.py:846
            lim = b.swiglu_limit
            if lim > 0:                                                    # model.py:847-849
                up = torch.clamp(up, min=-lim, max=lim)
                gate = torch.clamp(gate, max=lim)
            h = F.silu(gate) * up                                          # model.py:850
            h = m["pw"].view(-1, 1) * h                                    # model.py:852
            h = h.to(x.dtype)                                              # model.py:853
            q2, s2 = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)  # model.py:187 inside w2
            self._load(2, q2, s2, arow)
        out = self._gemm("w2", 2).view(S * M, -1)                     # [S*M, D] bf16
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
    Cost: S spare weight slots per layer (40 x 8 x 17.7 MB = 5.7 GB on the real model at S = 8)."""

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
        return self.owner.dg.align

    def _weights(self, m: Dict[str, torch.Tensor]) -> None:
        o, b, t = self.owner, self.owner.dg, self.t
        if "tags" in m:   # M1-T10-PORT dedup_fused made the stage ids, the compact cold index and the tags
            b.stage(t, m["ids_stage"], None, sidx=m["sidx"], ovf=o.ovf_step)   # 1 launch: cold c -> H + c, packed SF
            b.mi.view(b.k, b.align)[:, : o.rows].copy_(m["tags"])             # after the launch's head writes
            return
        uniq, counts = m["uniq"], m["counts"]
        ids = torch.where(counts > 0, uniq, self.dummy) if self.has_dummy else uniq
        g = t["gtab"].index_select(0, uniq).long()
        cold, ok, grp, sidx = compact_cold(g, counts, t["H"], self.S)
        b.stage(t, ids, None, sidx=sidx, ovf=o.ovf_step)              # 1 launch: cold c -> H + c, packed SF
        n_ok = torch.where(ok, counts, 0)                             # an overflowing block: every row -1
        tags = torch.where(self.ar_rows < n_ok.view(-1, 1), grp.view(-1, 1), -1)
        b.mi.view(b.k, b.align)[:, : o.rows].copy_(tags.to(torch.int32))  # after the launch's head writes

    def _gemm(self, p: str, which: int) -> torch.Tensor:
        return self.owner.dg.gemm_inplace(p, self.t, which)

    def _route_kw(self) -> dict:
        """dedup_fused's dg-inplace extras: the layer's hot-slot table, H, the spare budget, the stand-in id."""
        return {"inpl": {"gtab": self.t["gtab"], "H": int(self.t["H"]), "S": self.S, "dummy": self.dummy,
                         "has_dummy": self.has_dummy}}


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
