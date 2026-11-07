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

The fused glue at R > 1 runs only its row-parallel clusters (runtime/fused_glue.py `ROWS_CLUSTERS`); the R-row MoEs
take the fused shared-expert SwiGLU when it is on, their routed glue is the reference ops above.
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
        S, M = b.k, self._block_rows()
        m = self._route(x, M)
        self._weights(m)
        arow, prow = m["arow"], m["prow"]
        q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)   # model.py:187, the R rows once
        self._load(1, q.index_select(0, prow), s.index_select(0, prow), arow)
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
        y = combine_rows(out, m["posmat"], x)
        y += self._shared(x)
        return y.type_as(x).view(shape)

    # the two places dg-masked and dg-inplace differ (DGInplaceMoERows below)
    def _block_rows(self) -> int:
        return self.owner.dg.rows                                      # masked: max_m rows per group

    def _weights(self, m: Dict[str, torch.Tensor]) -> None:
        """Slot u's expert m["uniq"][u] -> staging slot u (weights + e8m0 scales), scales -> fp32 (sf32)."""
        o, b = self.owner, self.owner.dg
        self._gather(m["uniq"])
        for p in PROJS:                                                # e8m0 -> fp32, one op per projection
            b.convert_scales(p, o.staging[(p, "scale")])
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
