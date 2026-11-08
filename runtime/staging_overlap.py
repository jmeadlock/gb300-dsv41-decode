"""M3-S1: the R-row dg-inplace step's cold-expert staging taken off the critical path ($DSV41_STAGE).
Spec: docs/HANDOFF-M3-S1-STAGING-OVERLAP.md. Receipts / probe: bench/staging_overlap_bench.py.

Today (`kernel`, the default) every layer of the 6-row step runs ONE m1_stage_rows launch (runtime/static_kernels.py)
after the dedup and before its three DeepGEMM calls: the staged cold experts' weight rows are read from pinned Grace
over C2C (UVA) into spare pool slots H + c, their e8m0 scales packed into the SF groups. Measured 114 us/layer =
4.55 ms/step, ~263 GB/s (docs/GAP-LEDGER-OURS-20261107.md), serially before 4.45 ms of expert GEMM.

Modes ($DSV41_STAGE, read at StaticStep build; describe()["stage"]):
  kernel          unchanged: b.stage_rows(t, m), one launch, serial.
  kernel-overlap  graph-capturable. After the dedup the step records an event, a forked capture stream (`side`)
                  waits on it and runs s1_stage_proj three times -- w1, then w3, then w2 (bytes + scales of the
                  staged cold experts of that projection; hot scales too under sf_store "step") -- recording one event
                  per projection. The main stream meanwhile computes the shared expert (hoisted from the combine:
                  same kernels, same input, its result held until the combine) and the act quant, then waits on
                  projection p's event immediately before GEMM p. The w2 wait is the join (the side stream is
                  serial, so ev[w2] implies ev[w1], ev[w3]): every forked op is joined inside the layer, before the
                  combine, as graph capture requires. No host sync anywhere; the DeepGEMM call shapes are untouched
                  (no hot/cold split), so --moe-rowmatch and the 6-row == 1-row invariance are unaffected.
                  Knobs: $DSV41_STAGE_OVERLAP_NPROG (programs per staged slot, default 48) and
                  $DSV41_STAGE_OVERLAP_BLOCK_BYTES (bytes per program per trip, default 8 KiB), each one value or
                  three comma-separated (w1,w3,w2; M3-S1-FIX: e.g. NPROG=48,16,16 BLOCK_BYTES=8192,32768,32768 --
                  a fast w1 copy, which GEMM w1 waits on, and low-contention w3 / w2 copies, which hide under the
                  GEMMs; probe B: 48 programs cost the GEMMs 21-35 us/layer at c >= 2, 16 x 32 KiB 3-17).
                  RISK (cf. M2-L1, the
                  forked mixes that lost 0.35 ms at 6 rows): the copy's CTAs share SMs with the shared-expert GEMVs
                  and the persistent DeepGEMM kernel; stream priority is not reserved capacity. Fewer programs
                  with more bytes in flight each is the lever; bench/staging_overlap_bench.py measures both.
  ce              copy-engine staging: per staged cold expert one cudaMemcpyAsync per projection (torch copy_ of the
                  expert's UVA home -> its pool slot), scales by s1_stage_proj(weights=False). EAGER ONLY.
  ce-overlap      ce's copies with kernel-overlap's fork / per-projection join. EAGER ONLY.

Why ce cannot be captured (and the handoff's memcpy-node hypothesis does not hold as stated): a memcpy node's source
and destination are fixed when the graph is built, but WHICH experts are cold in a layer -- and so which Grace rows
to read and which spare slot they go to -- is decided on the device by that layer's router in that step. A captured
memcpy can only ever copy the expert it was captured with (docs/T8-T9A-DESIGN.md 2.2 and A2-RESIDENCY-DESIGN.md 3.2
say the same). The ce modes therefore read the work list on the host (one sync per layer), which is fine for an
eager probe of CE bytes / bandwidth on real shapes and is refused under capture (StageCaptureError). Making CE
capturable needs device-selected copies: CUDA 12.8+ SWITCH conditional nodes (per layer x spare slot x projection a
switch over the layer's cold experts, each case a memcpy node, the value set by a device kernel after the dedup;
~43k memcpy nodes in the step, and residency moves would need exec-node updates) -- bench/staging_overlap_bench.py
reports whether the box's torch / cuda-python expose that API; it is not implemented here.

Bytes: s1_stage_proj(p) writes exactly what m1_stage_rows' projection-p programs write (contract:
stage_proj_emulated == dg_inplace.stage_rows_emulated restricted to p; the union over p is stage_rows_emulated);
ce writes the same weight bytes by memcpy and packs the scales with the same kernel code. Nothing that feeds
arithmetic changes, so every mode is bit-identical to `kernel` by construction; the tests check the bytes and the
logits anyway.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import torch

try:  # GPU image only; the Mac runs the torch contract (and tests/triton_shim.py for the kernel body)
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = tl = None
    HAVE_TRITON = False

STAGE_ENV = "DSV41_STAGE"
STAGE_MODES = ("kernel", "kernel-overlap", "ce", "ce-overlap")
STAGE_DEFAULT = "kernel"
CAPTURABLE = {"kernel": True, "kernel-overlap": True, "ce": False, "ce-overlap": False}
OVERLAP = {"kernel": False, "kernel-overlap": True, "ce": False, "ce-overlap": True}
NPROG_ENV = "DSV41_STAGE_OVERLAP_NPROG"   # s1_stage_proj programs per staged slot (default: static_kernels' 48)
BLOCK_ENV = "DSV41_STAGE_OVERLAP_BLOCK_BYTES"   # bytes per program per loop trip (default 8 KiB): fewer programs
#                                               need more bytes in flight each to keep the C2C link busy
PROJ_INDEX = {"w1": 0, "w3": 1, "w2": 2}  # = dg_inplace.ORDER6 / the ptab6 + dtab rows
PROJ_NAMES = ("w1", "w3", "w2")
# the static_kernels constants (that module imports triton unconditionally; these read the same env vars)
STAGE_NPROG = int(os.environ.get("DSV41_STAGE_NPROG", "48"))
VEC_BLOCK_BYTES = int(os.environ.get("DSV41_GATHER_BLOCK_BYTES", "8192"))
VEC_WARPS = int(os.environ.get("DSV41_GATHER_WARPS", "4"))
VEC_ALIGN = 16
STAGE_BN = 64
SF_PACK = 4
KERNELS_ON_CPU = False                 # tests set it under tests/triton_shim.py: stage_proj launches the kernel body
CPU_RESOLVE: Optional[Callable[[int], torch.Tensor]] = None   # tests: address -> uint8 tensor (the CPU contract path)


class StageCaptureError(RuntimeError):
    """A ce mode met graph capture: copy-engine sources would be fixed at capture, the cold set is not."""


# M3-S1-FIX diagnostic (NOT a mode of the runtime): no cold staging at all -- the spare slots keep whatever they
# held, so any step that routes a cold expert computes WRONG outputs. It exists for one box measurement: the
# headline ms/step with staging removed is the ceiling of what hiding the staging can buy (kernel - none-diag), and
# kernel-overlap - none-diag is the staging still exposed in the REAL graph (the probe's [E] only models it).
# Measure it on the teacher-forced 6-row step cell (bench/first_light.py --rows 6 --teacher-forced): the input tokens
# stay fixed; the stale slots hold real experts' weights, so hidden states stay in range, but deeper layers' routing
# drifts -- compare the cell's cold share (expert_pool counters) with the kernel run's before trusting the delta.
# A generation-loop (headline) number with it is meaningless (the stream diverges, so does the work).
# Refused unless $DSV41_STAGE_DIAG=wrong-outputs-ok; never publish a stream / parity number taken with it.
DIAG_MODE = "none-diag"
DIAG_ENV, DIAG_ACK = "DSV41_STAGE_DIAG", "wrong-outputs-ok"


def stage_mode(req: Optional[str] = None) -> str:
    v = req or os.environ.get(STAGE_ENV) or STAGE_DEFAULT
    if v == DIAG_MODE:
        if os.environ.get(DIAG_ENV) != DIAG_ACK:
            raise ValueError(f"${STAGE_ENV}={DIAG_MODE} skips cold staging (wrong outputs): a timing diagnostic "
                             f"only, needs ${DIAG_ENV}={DIAG_ACK}")
        return v
    if v not in STAGE_MODES:
        raise ValueError(f"${STAGE_ENV}={v!r} not in {STAGE_MODES}")
    return v


def _block_bytes(block_bytes) -> int:
    bb = int(block_bytes or VEC_BLOCK_BYTES)
    if bb < VEC_ALIGN or bb & (bb - 1):
        raise ValueError(f"stage block {bb} B must be a power of two >= {VEC_ALIGN}")
    return bb


def _per_proj(spec: Optional[str], conv, name: str) -> List[Optional[int]]:
    """'' / None -> [None] * 3 (stage_proj's defaults); '48' -> [48] * 3; '48,16,16' -> one per projection (w1, w3,
    w2). Launch shape only: every setting writes the same bytes (s1_stage_proj's contract)."""
    if not spec:
        return [None, None, None]
    vals = [v.strip() for v in str(spec).split(",")]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3 or any(not v for v in vals):
        raise ValueError(f"${name}={spec!r}: one value or three (w1,w3,w2)")
    out = [conv(v) for v in vals]
    if any(int(v) < 1 for v in out):
        raise ValueError(f"${name}={spec!r}: values must be >= 1")
    return out


# ------------------------------------------------------------------------------------------------ the kernel
if HAVE_TRITON:
    @triton.jit
    def _s1_pack_scales(ptab, dtab, e, E, p, grp, q, N13, C13, N2, C2, BN: tl.constexpr, BC: tl.constexpr,
                        NPROG: tl.constexpr):
        """static_kernels._t9a2_pack_scales, verbatim (a copy: that module needs triton at import): expert e's
        projection-p e8m0 scale rows as int32 words -> group `grp` of the MN-major packed SF at dtab[3 + p]."""
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
    def s1_stage_proj(ptab, uniq, gtab, sidx, wl, dtab, E, H, PJ, N13, C13, N2, C2,
                      ROW_W: tl.constexpr, BW: tl.constexpr, NBW: tl.constexpr, BN: tl.constexpr,
                      BC: tl.constexpr, NPROG: tl.constexpr, EVEN_W: tl.constexpr, WEIGHTS: tl.constexpr,
                      SCALES: tl.constexpr):
        """M3-S1: m1_stage_rows for ONE projection PJ (0 w1, 1 w3, 2 w2; a runtime scalar). Program (j, q): slot
        u = wl[j] (-1: exit), e = uniq[u], g = gtab[e]; a cold slot (g < 0, compact index c = sidx[u]) copies
        8-KiB weight blocks q, q+NPROG, ... of row ptab[PJ, e] (HBM or UVA) into pool slot H + c of dtab[PJ]
        (WEIGHTS), and every listed slot packs its PJ scales into its group (SCALES; hot slots are listed only
        under sf_store "step"). WEIGHTS=False is the ce modes' scale-only launch."""
        j = tl.program_id(0)
        q = tl.program_id(1)
        u = tl.load(wl + j)
        if u >= 0:
            e = tl.load(uniq + u)
            g = tl.load(gtab + e).to(tl.int64)
            c = tl.load(sidx + u).to(tl.int64)
            cold = g < 0
            grp = tl.where(cold, c + H, g)
            if WEIGHTS:
                if cold:
                    p_src = tl.multiple_of(tl.load(ptab + PJ * E + e).to(tl.pointer_type(tl.int64)), 16)
                    p_dst = tl.multiple_of(tl.load(dtab + PJ).to(tl.pointer_type(tl.int64)), 16)
                    p_row = p_dst + grp * ROW_W
                    if EVEN_W:
                        for it in range(0, NBW // NPROG):
                            pe_off = (q + it * NPROG) * BW + tl.arange(0, BW)
                            tl.store(p_row + pe_off, tl.load(p_src + pe_off))
                    else:
                        for it2 in range(0, (NBW + NPROG - 1) // NPROG):
                            pm_off = (q + it2 * NPROG) * BW + tl.arange(0, BW)
                            pm_m = pm_off < ROW_W
                            tl.store(p_row + pm_off, tl.load(p_src + pm_off, mask=pm_m), mask=pm_m)
            if SCALES:
                _s1_pack_scales(ptab, dtab, e, E, PJ, grp, q, N13, C13, N2, C2, BN, BC, NPROG)


# ------------------------------------------------------------------------------------------------ contract
def stage_proj_emulated(resolve, ptab6, uniq, gtab, dtab, dsts, sidx, wl, H: int, row_w: int, sf_dims,
                        n_work: int, pj: int, weights: bool = True, scales: bool = True) -> None:
    """The contract of s1_stage_proj in torch (dg_inplace.stage_rows_emulated restricted to projection pj): for
    j < n_work with u = wl[j] >= 0, e = uniq[u], g = gtab[e]: cold -> (weights) its pj weight row into slot
    H + sidx[u] of dsts[pj]; (scales) its pj scales into that group of the packed SF dsts[3 + pj]; hot -> (scales)
    its pj scales into group g. Nothing else is written. `resolve(address)` -> uint8 tensor of the source row."""
    assert [int(a) for a in dtab.tolist()] == [t.data_ptr() for t in dsts] and ptab6.shape[0] == 6
    wll, ul, sl = [int(v) for v in wl.tolist()], [int(v) for v in uniq.tolist()], [int(v) for v in sidx.tolist()]
    N, C = sf_dims[1] if pj == 2 else sf_dims[0]
    for j in range(int(n_work)):
        u = wll[j]
        if u < 0:
            continue
        e = ul[u]
        g = int(gtab[e])
        cold = g < 0
        grp = H + sl[u] if cold else g
        if weights and cold:
            dsts[pj][grp].view(torch.uint8).view(-1).copy_(resolve(int(ptab6[pj, e])).view(-1)[:row_w])
        if scales:
            src = resolve(int(ptab6[3 + pj, e]))
            dsts[3 + pj][grp].copy_(src.view(-1)[:N * C * SF_PACK].view(N, C * SF_PACK).view(torch.int32))


def stage_proj(ptab6: torch.Tensor, uniq: torch.Tensor, gtab: torch.Tensor, dtab: torch.Tensor, dsts,
               sidx: torch.Tensor, wl: torch.Tensor, H: int, row_w: int, sf_dims, n_work: int, pj: int, *,
               weights: bool = True, scales: bool = True, block_bytes: Optional[int] = None,
               nprog: Optional[int] = None, num_warps: Optional[int] = None) -> None:
    """ONE launch of s1_stage_proj (grid (n_work, nprog)) for projection pj -- arguments as static_kernels.stage_rows.
    Without a device kernel (the Mac) the contract runs through CPU_RESOLVE (tests)."""
    (n13, c13), (n2, c2) = sf_dims
    if pj not in (0, 1, 2) or not (weights or scales):
        raise ValueError(f"stage_proj: projection {pj!r}, weights {weights}, scales {scales}")
    if row_w % VEC_ALIGN or any(t.data_ptr() % VEC_ALIGN for t in dsts):
        raise ValueError(f"stage_proj: weight row {row_w} B or a destination not {VEC_ALIGN}-B aligned")
    nw = int(n_work)
    if not 1 <= nw <= wl.numel():
        raise ValueError(f"stage_proj: n_work {nw} outside [1, {wl.numel()}]")
    if HAVE_TRITON and (uniq.is_cuda or KERNELS_ON_CPU):
        bw = _block_bytes(block_bytes) // 8
        ww = row_w // 8
        nbw = triton.cdiv(ww, bw)
        npg = int(nprog or STAGE_NPROG)
        bc = triton.next_power_of_2(max(int(c13), int(c2)))
        even = ww % bw == 0 and nbw % npg == 0
        s1_stage_proj[(nw, npg)](ptab6, uniq, gtab, sidx, wl, dtab, ptab6.size(1), int(H), int(pj), int(n13),
                                 int(c13), int(n2), int(c2), ww, bw, nbw, STAGE_BN, bc, npg, even, bool(weights),
                                 bool(scales), num_warps=int(num_warps or VEC_WARPS))
        return
    if CPU_RESOLVE is None:
        raise RuntimeError("stage_proj: no device kernel here and no staging_overlap.CPU_RESOLVE (tests set it)")
    stage_proj_emulated(CPU_RESOLVE, ptab6, uniq, gtab, dtab, dsts, sidx, wl, H, row_w, sf_dims, nw, pj,
                        weights, scales)


# ------------------------------------------------------------------------------------------------ copy engine
class _CAI:
    def __init__(self, ptr: int, n: int):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def address_bytes(addr: int, n: int, device) -> torch.Tensor:
    """uint8 [n] CUDA view of device-addressable memory at `addr` (an HBM slot or a UVA address in registered host
    memory, as bench/c2c_probe.py); CPU: CPU_RESOLVE (tests)."""
    if torch.device(device).type == "cuda":
        return torch.as_tensor(_CAI(int(addr), int(n)), device=device)
    if CPU_RESOLVE is None:
        raise RuntimeError("address_bytes on CPU needs staging_overlap.CPU_RESOLVE (tests)")
    return CPU_RESOLVE(int(addr)).reshape(-1)[: int(n)]


def ce_plan(t: dict, m: Dict[str, torch.Tensor], n_work: int) -> List[dict]:
    """HOST SYNC (eager only): the staged cold experts of this layer-step from the dedup's work list, in work-list
    order: [{"u", "e", "grp", "src": [w1, w3, w2 row addresses]}]."""
    nw = int(n_work)
    wl = [int(v) for v in m["wl"][:nw].tolist()]
    uniq = [int(v) for v in m["uniq"].tolist()]
    sidx = [int(v) for v in m["sidx"].tolist()]
    gtab = t["gtab"]
    H = int(t["H"])
    out = []
    for u in wl:
        if u < 0:
            continue
        e = uniq[u]
        if int(gtab[e]) >= 0:      # a hot slot listed for its scales only (sf_store "step")
            continue
        out.append({"u": u, "e": e, "grp": H + sidx[u],
                    "src": [int(v) for v in t["ptab6"][:3, e].tolist()]})
    return out


def ce_copy_proj(t: dict, plan: List[dict], pj: int, row_w: int) -> int:
    """One cudaMemcpyAsync (torch copy_, on the current stream) per planned cold expert: its projection-pj weight
    row -> pool slot grp. Returns copies issued."""
    w = t["w"][PROJ_NAMES[pj]]
    for it in plan:
        dst = w[it["grp"]].view(torch.uint8).view(-1)
        dst.copy_(address_bytes(it["src"][pj], row_w, w.device), non_blocking=True)
    return len(plan)


# ------------------------------------------------------------------------------------------------ the step hook
class StageOverlap:
    """One per R-row dg-inplace StaticStep (owner.stage_ov). DGInplaceMoERows calls `launch` where it called
    b.stage_rows, and `wait(p)` before GEMM p. Streams and events are created here, at build (never under capture);
    one route event and three projection events per layer, re-recorded every step."""

    def __init__(self, owner, mode: str, n_layers: int):
        self.owner, self.mode, self.n_layers = owner, stage_mode(mode), int(n_layers)
        self.dev = torch.device(owner.dev)
        self.cuda = self.dev.type == "cuda"
        self.overlap = OVERLAP.get(self.mode, False)
        self.capturable = CAPTURABLE.get(self.mode, True)         # none-diag: nothing to capture but the GEMMs
        # per projection (w1, w3, w2): "48" = all three, "48,16,16" = one each (M3-S1-FIX: w1 is the latency-
        # critical copy -- GEMM w1 waits on it with only the shared expert as cover -- w3 / w2 hide under GEMMs, where
        # fewer programs with more bytes in flight each contend less with the persistent DeepGEMM, probe B)
        self.nprog = _per_proj(os.environ.get(NPROG_ENV), int, NPROG_ENV)
        self.block_bytes = _per_proj(os.environ.get(BLOCK_ENV), _block_bytes, BLOCK_ENV)
        self.side = None
        self.ev_route: List = []
        self.ev: List[List] = []
        if self.cuda and self.overlap:
            self.side = torch.cuda.Stream(device=self.dev)
            self.ev_route = [torch.cuda.Event() for _ in range(self.n_layers)]
            self.ev = [[torch.cuda.Event() for _ in range(3)] for _ in range(self.n_layers)]
        self.stats = {"launches": 0, "proj_launches": 0, "ce_copies": 0, "host_syncs": 0, "waits": 0,
                      "shared_hoisted": 0}
        self.trace: Optional[list] = None   # tests: ("stage", layer, pj) / ("wait", layer, pj) / ("hoist", layer)

    @property
    def active(self) -> bool:
        return self.mode != "kernel"

    def _note(self, *ev) -> None:
        if self.trace is not None:
            self.trace.append(ev)

    def _capturing(self) -> bool:
        return self.cuda and torch.cuda.is_current_stream_capturing()

    def _proj(self, t: dict, m: Dict[str, torch.Tensor], pj: int, n_work: int, plan, layer: int) -> None:
        b = self.owner.dg
        if plan is None:
            stage_proj(t["ptab6"], m["uniq"], t["gtab"], t["dtab"], t["dsts"], m["sidx"], m["wl"], t["H"], b.row_w,
                       b.sf_dims, n_work, pj, nprog=self.nprog[pj], block_bytes=self.block_bytes[pj])
        else:
            self.stats["ce_copies"] += ce_copy_proj(t, plan, pj, b.row_w)
            stage_proj(t["ptab6"], m["uniq"], t["gtab"], t["dtab"], t["dsts"], m["sidx"], m["wl"], t["H"], b.row_w,
                       b.sf_dims, n_work, pj, weights=False, nprog=self.nprog[pj],
                       block_bytes=self.block_bytes[pj])
        self.stats["proj_launches"] += 1
        self._note("stage", layer, pj)

    def launch(self, sm, t: dict, m: Dict[str, torch.Tensor]) -> bool:
        """Stage layer sm.layer's cold experts (the dedup's work list) in this mode. Returns True when the caller
        must hoist the shared expert and wait per projection (the overlap modes)."""
        b = self.owner.dg
        if self.mode == "kernel":
            b.stage_rows(t, m)
            return False
        if self.mode == DIAG_MODE:                       # diagnostic: no staging launch at all (wrong outputs)
            self.stats["diag_skipped"] = self.stats.get("diag_skipped", 0) + 1
            return False
        if self._capturing() and not self.capturable:
            raise StageCaptureError(
                f"${STAGE_ENV}={self.mode} is eager-only: a captured memcpy's source / destination are fixed at "
                f"capture, the cold experts are chosen by the router on the device each step "
                f"(runtime/staging_overlap.py docstring); use kernel or kernel-overlap for graphs")
        if any(sm._stage_pending):
            raise RuntimeError(f"L{sm.layer}: the previous layer-call's forked staging was never joined")
        n_work = b.stage_n_work()
        plan = None
        if self.mode in ("ce", "ce-overlap"):
            plan = ce_plan(t, m, n_work)                 # host sync (eager only, checked above)
            self.stats["host_syncs"] += 1
        self.stats["launches"] += 1
        l = sm.layer
        if self.side is not None:
            cur = torch.cuda.current_stream(self.dev)
            self.ev_route[l].record(cur)
            self.side.wait_event(self.ev_route[l])
            with torch.cuda.stream(self.side):
                for pj in range(3):
                    self._proj(t, m, pj, n_work, plan, l)
                    self.ev[l][pj].record(self.side)
            sm._stage_pending = [True, True, True]
        else:                                            # serial (ce), or the overlap order without streams (CPU)
            for pj in range(3):
                self._proj(t, m, pj, n_work, plan, l)
        return self.overlap

    def wait(self, sm, pj: int) -> None:
        """Main stream waits for projection pj's staging (before GEMM pj). The side stream is serial, so the w2
        wait joins everything this layer forked."""
        self._note("wait", sm.layer, pj)
        if not sm._stage_pending[pj]:
            return
        torch.cuda.current_stream(self.dev).wait_event(self.ev[sm.layer][pj])
        self.stats["waits"] += 1
        if pj == 2:
            sm._stage_pending = [False, False, False]
        else:
            sm._stage_pending[pj] = False

    def hoisted(self, sm) -> None:
        self.stats["shared_hoisted"] += 1
        self._note("hoist", sm.layer)

    def describe(self) -> dict:
        return {"mode": self.mode, "capturable": self.capturable, "overlap": self.overlap,
                "outputs_valid": self.mode != DIAG_MODE,
                "side_stream": self.side is not None, "nprog": [v or STAGE_NPROG for v in self.nprog],
                "block_bytes": [v or VEC_BLOCK_BYTES for v in self.block_bytes], "stats": dict(self.stats)}


def build(owner, n_layers: int, mode: Optional[str] = None) -> StageOverlap:
    return StageOverlap(owner, stage_mode(mode), n_layers)


# ------------------------------------------------------------------------------------------------ the model [E]
# Per-layer timeline of the 6-row MoE after the dedup, in us. Defaults are docs/gap-tf-anatomy-20261107.json kernel
# means [M] (nsys static-eager, 6-row teacher-forced on Pg61 tokens, seed map 295/89, spare 16, 40 layers):
#   GEMM per DeepGEMM call 37.1 (sm100_fp8_fp4_gemm_1d1d_impl, 120/step); act quant 2.3 (t9b2_act_quant_bcast);
#   routed SwiGLU 3.1 (t9b2_swiglu_quant, 1 of its 2 per layer); shared expert ~30 (its fp8 GEMVs + split-K reduce
#   + SwiGLU: the moe.shared family, 0.881 ms / 40 = 22.0 [M] plus the shared SwiGLU and reduce ~8 [E]);
#   staging 113.8 us/layer at 15236 cold unique / 9640 layer-steps = 1.58 cold/layer (docs/gap-tf-static-20261107.json)
#   -> 18.8 MB x 1.58 / 113.8 us = 261 GB/s effective, the zero-cold launch cost not separable [M/D].
EXPERT_BYTES = 18_800_640                      # w1 + w3 + w2 weights + scales of one routed expert (A2 memo 0.1)
MODEL_DEFAULTS = {"gemm_us": 37.1, "act_us": 2.3, "swiglu_us": 3.1, "shared_us": 30.0, "launch_us": 3.0,
                  "kernel_GBps": 261.0, "ce_GBps": 366.0, "cold_mean": 1.58, "layers": 40, "c_max": 16,
                  "serial_ms_measured": 4.552, "step_ms": 20.304, "contention": 0.0}


def layer_exposed_us(c: int, GBps: float, p: Optional[dict] = None, *, overlap: bool = True) -> float:
    """Staging time on the critical path of one layer with c staged cold experts at GBps per expert byte.
    serial: 3 projections back to back (+ launch). overlap: kernel-overlap's schedule -- side: w1, w3, w2 copies
    of a = c*b/3/GBps each (+ launch once); main: act + shared, then GEMM p no earlier than copy p's end, SwiGLU
    between w3 and w2. `contention` stretches each GEMM by that fraction while a copy is running [A]."""
    q = dict(MODEL_DEFAULTS, **(p or {}))
    if c <= 0:
        return 0.0 if overlap else q["launch_us"]
    a = c * EXPERT_BYTES / 3 / (GBps * 1e3)    # us per projection (GB/s = 1e3 B/us)
    if not overlap:
        return 3 * a + q["launch_us"]
    g = q["gemm_us"]
    s = q["launch_us"]
    ends = [s + a, s + 2 * a, s + 3 * a]       # side stream: copy p done
    t = q["act_us"] + q["shared_us"]           # main: hoisted shared expert + act quant
    for i in range(3):
        t = max(t, ends[i])
        gi = g * (1 + q["contention"]) if t < ends[2] else g
        t += gi
        if i == 1:
            t += q["swiglu_us"]
    base = q["act_us"] + q["shared_us"] + 3 * g + q["swiglu_us"]
    return max(0.0, t - base)


def poisson(lam: float, c_max: int) -> List[float]:
    import math

    ps = [math.exp(-lam) * lam ** c / math.factorial(c) for c in range(c_max + 1)]
    ps[-1] += max(0.0, 1.0 - sum(ps))
    return ps


def model_step(p: Optional[dict] = None, dist: Optional[List[float]] = None) -> dict:
    """[E] staging ms/step on the critical path for each design, cold experts per layer ~ `dist` (default Poisson
    at the measured mean 1.58 [A: the per-layer distribution is not in any receipt]). Calibrates nothing: the
    serial-kernel row is the model's own; the measured 4.552 is reported next to it."""
    q = dict(MODEL_DEFAULTS, **(p or {}))
    d = dist or poisson(q["cold_mean"], q["c_max"])
    L = q["layers"]

    def ms(GBps, overlap):
        return L * sum(pc * layer_exposed_us(c, GBps, q, overlap=overlap) for c, pc in enumerate(d)) / 1e3

    rows = {"kernel (serial, today)": ms(q["kernel_GBps"], False),
            "kernel-overlap": ms(q["kernel_GBps"], True),
            "ce serial [needs capturable CE]": ms(q["ce_GBps"], False),
            "ce-overlap [needs capturable CE]": ms(q["ce_GBps"], True)}
    base = rows["kernel (serial, today)"]
    return {"assumptions": q, "dist": [round(x, 4) for x in d],
            "staging_exposed_ms_per_step": {k: round(v, 3) for k, v in rows.items()},
            "step_ms_E": {k: round(q["step_ms"] - (q["serial_ms_measured"] - v * q["serial_ms_measured"] / base), 3)
                          for k, v in rows.items()},
            "per_c_exposed_us": {c: {"kernel": round(layer_exposed_us(c, q["kernel_GBps"], q, overlap=False), 1),
                                     "kernel-overlap": round(layer_exposed_us(c, q["kernel_GBps"], q), 1),
                                     "ce": round(layer_exposed_us(c, q["ce_GBps"], q, overlap=False), 1),
                                     "ce-overlap": round(layer_exposed_us(c, q["ce_GBps"], q), 1)}
                                 for c in range(0, 7)}}
