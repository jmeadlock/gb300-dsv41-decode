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

The fused glue is not used at R > 1 (StaticStep builds none). dg-inplace at R > 1 needs up to R*k cold spare slots per
layer (27 GB at 36 on the real model) or a hot/cold split; not implemented here (StaticStep refuses it).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PROJS = ("w1", "w3", "w2")


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

    @property
    def active(self) -> bool:
        return self.owner.active

    def _route(self, x: torch.Tensor, block_rows: int) -> Dict[str, torch.Tensor]:
        o = self.owner
        weights, indices = self.moe.gate(x, None)                     # model.py:894 at R rows (text: no image mask)
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
        from runtime.dg_moe import e8m0_to_f32

        o, moe, b, R = self.owner, self.moe, self.owner.dg, self.owner.R
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893
        S, M = b.k, b.rows
        m = self._route(x, M)
        self._gather(m["uniq"])
        for p in PROJS:                                                # e8m0 -> fp32, one op per projection
            b.convert_scales(p, o.staging[(p, "scale")])
        arow, prow = m["arow"], m["prow"]
        q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)   # model.py:187, the R rows once
        self._load(1, q.index_select(0, prow), s.index_select(0, prow), arow)
        b.masked_m.copy_(m["counts"].to(torch.int32))
        gate = b.gemm("w1", o.staging[("w1", "weight")], 1).view(S * M, -1).index_select(0, arow).float()  # 845
        up = b.gemm("w3", o.staging[("w3", "weight")], 1).view(S * M, -1).index_select(0, arow).float()    # 846
        lim = b.swiglu_limit
        if lim > 0:                                                    # model.py:847-849
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up                                          # model.py:850
        h = m["pw"].view(-1, 1) * h                                    # model.py:852
        h = h.to(x.dtype)                                              # model.py:853
        q2, s2 = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)  # model.py:187 inside w2
        self._load(2, q2, s2, arow)
        out = b.gemm("w2", o.staging[("w2", "weight")], 2).view(S * M, -1)     # [S*M, D] bf16
        y = combine_rows(out, m["posmat"], x)
        y += self._shared(x)
        return y.type_as(x).view(shape)

    def _load(self, which: int, q: torch.Tensor, s: torch.Tensor, arow: torch.Tensor) -> None:
        """A rows `arow` of xq[which] / xs[which] <- the pairs' quantized activation (fp8) and fp32 scales."""
        from runtime.dg_moe import e8m0_to_f32

        b = self.owner.dg
        K = q.size(-1)
        b.xq[which].view(-1, K).index_copy_(0, arow, q)
        s32 = torch.empty(s.shape, dtype=torch.float32, device=s.device)
        e8m0_to_f32(s, s32, b.e8m0_method)
        b.xs[which].view(-1, s32.size(-1)).index_copy_(0, arow, s32)
