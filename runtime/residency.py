"""A2-2: adaptive expert residency -- the host-side swap policy on the pointer-table gather.

    rm = ResidencyManager(pool, model, ResidencyConfig(), n_stage=n_stage_for("dg-inplace", rows=1, topk=6))
    st = StaticStep(model, pool, ..., moe_impl="dg-inplace")   # built AFTER the manager grew the pool
    rm.attach(st)                                               # discovers + verifies every pointer table
    ... st.step(...) / st.replay(...); rm.after_step() ...      # runtime/generate.py does this with residency=

Spec: docs/HANDOFF-A2-2-RESIDENCY.md; design docs/A2-RESIDENCY-DESIGN.md 3.1-3.4 and 5 "A2-2". Policy: m3's v26ar.2
decision core, vendored byte-identically as runtime/residency_policy.py (m3 is never imported).

Mechanism (design 3.1: a byte copy plus a pointer-table write, no kernel change, no graph recapture):
  slots    per layer the HBM slots are the H boot-hot slots of runtime/expert_pool.py's pools plus `n_spare` spare
           slots appended AFTER the step's own staging slots: weights live in the pool's grown [H + n_stage +
           n_spare, N, K/2] tensors (ExpertPool.add_spare_slots, done here before the StaticStep is built, so the
           dg-inplace GEMM reads a promoted expert in place at group = its slot), scales in this manager's
           [n_spare, N, K/32] e8m0 buffers (boot slots keep the pool's own scale rows). n_stage = what the step
           stages into (dg-inplace: topk * rows cold slots at H..H+n_stage-1; ref / dg-masked: 0).
  grace    a boot-cold expert's Grace home is its slot in the pool's cold HostBuffer (immutable); a boot-hot expert
           gets a home in this manager's demotion HostBuffer (n_demote slots per layer, cudaHostRegister'd, no slot
           crossing a registration chunk) the first time it is demoted: one D2H of its HBM bytes, then it keeps
           that home for good (a later demotion is a pointer write).
  promote  copy stream: wait(the slot's free event) -> 6 H2D copies (UVA alias -> slot) [+ the packed SF group
           when dg-inplace keeps scales "resident"] -> record event. Published later (below).
  publish  between two steps, on the compute (= replay) stream: wait(copy event) -> write the expert's entry in
           EVERY table some kernel reads (each StaticMoE's six `ptabs`, the fused glue's stacked `_ptab6`, dg-inplace's
           `t["ptab6"]` + `t["gtab"]`, the step's `cold_tab`, the pool's `loc`) -> record an event; the demoted
           expert's slot rejoins the free list carrying that event, so a copy into it waits until every replay
           enqueued before the publish (the last readers) has finished. Replays never stall: publishing waits
           for copy events the host has seen complete (`Event.query`), except a forced flush before the next
           barrier (stream-ordered, still no host sync). Table writes are `fill_` of one entry (a scalar kernel
           arg: no H2D, no sync); a table's ADDRESS never changes (design 3.3).
  single   this manager is the only writer of those tables after `attach`; attach also scans the step's object
  writer   graph for any other int64 [.., E] tensor holding expert addresses and refuses to run if it finds one it
           does not own (a copy some kernel might read stale = design 3.3's second illegal case).
  eager    every routed Expert's w1/w2/w3 parameters are re-pointed at the expert's current home on publish, so the
           reference prefill (eager, outside the graph) reads the same bytes as the static step.

Policy (design 3.2): per-(layer, expert) hit counter `cnt` int32 [L, E] on the device, bumped after every step from
the routed ids the step already exposes (`StaticStep.ids_step`, x `uvalid_step` at rows > 1) with a one-hot sum
(deterministic, no atomics, outside the graph). Every `drain_n` host steps: async D2H of `cnt` into pinned memory;
the next step boundary that sees the copy done runs `policy.drain` (EWMA, queue), and every step boundary runs one
barrier = `policy.take_barrier` / `is_stale` / `commit` in apply_pending's order, actuating each swap (a veto -- no
free slot, no demotion slot -- skips the commit; with none, it IS apply_pending). Start values = shipped v26
(drain_n 64, swaps_per_drain 2, halflife 500, hyst 1.5, min_abs 8) except swaps_per_barrier = 4: design 3.2
"Budget" caps swaps at <= 4 per step [A].

Determinism (design 3.4): P1 bytes -- every address a table can name holds the checkpoint's bytes (copies of
immutable homes, published only after their event, slots reused only after their last reader); P2 -- the step's
arithmetic does not depend on the tier (ref/dg-masked stage in ascending-id slot order; dg-inplace's group index
only selects which in-place weights a block reads); P3 counters feed only the policy; P4 the policy is
deterministic and copy timing is irrelevant by P1. `audit()` checks P1 + the I1/I2 invariants (T-R2, T-R3).

Test-only mutation hooks (`_mutation`, T-R4): "publish_before_copy", "skip:<table kind>", "corrupt_promoted".
"""
from __future__ import annotations

import hashlib
import math
import mmap
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from runtime import residency_policy as P
from runtime.host_tier import GiB, MAX_REGISTER_CHUNK, HostBuffer, check_mem_available

ORDER6 = (("w1", "weight"), ("w3", "weight"), ("w2", "weight"), ("w1", "scale"), ("w3", "scale"), ("w2", "scale"))
PROJS = ("w1", "w3", "w2")
TIER_HBM, TIER_HOST, TIER_DEMOTED = 0, 1, 2  # pool.loc[..., 1]: HBM slot / cold-pool slot / demotion-pool slot
KV_BYTES_PER_ROW_32K = 150e6  # docs/LEDGER.md 2026-11-06 14:17: ~150 MB per row at 32K context
ALIGN = 256
MUTATIONS = ("publish_before_copy", "corrupt_promoted")  # + "skip:<ptab|ptab6|gtab|cold_tab|loc>"


# ------------------------------------------------------------------------------------------------ config
@dataclass
class ResidencyConfig:
    """Knobs. Policy fields = residency_policy.Params (v26 start values; swaps_per_barrier 4, design 3.2 Budget)."""

    drain_n: int = 64
    swaps_per_drain: int = 2
    halflife: float = 500.0
    hyst: float = 1.5
    min_abs: float = 8.0
    swaps_per_barrier: int = 4
    frozen: bool = False          # True = count and drain, never queue (the "frozen seed" arm)
    n_spare: Optional[int] = None  # spare HBM slots per layer; None = min(spare_cap, the headroom calc)
    spare_cap: int = 4            # design 3.1 [A] r = 4 (in-flight swaps per layer <= swaps_per_drain = 2)
    n_demote: int = 64            # Grace demotion slots per layer (design 3.1 [A] D = 64)
    reserve_gib: float = 8.0      # HBM kept back for capture/workspaces/staging (design 2 [A]: 4.2 + ~4 GiB)
    kv_rows: int = 1              # KV reservation: rows x 150 MB x kv_ctx / 32K
    kv_ctx: int = 32768
    hbm_free_bytes: Optional[int] = None  # override the measured free HBM (CPU tests)
    host_reserve_gib: float = 16.0  # MemAvailable kept free when pinning the demotion pool
    force_random: int = 0         # forced random swaps queued per step (the forced-random-swap run, T-R1)
    seed: int = 0                 # for force_random
    verbose: bool = False

    @classmethod
    def from_any(cls, x) -> Optional["ResidencyConfig"]:
        """None / False / "off" -> None; True / "on" -> defaults; dict -> fields; a config -> itself."""
        if x is None or x is False or (isinstance(x, str) and x.lower() in ("off", "none", "0", "")):
            return None
        if x is True or (isinstance(x, str) and x.lower() in ("on", "1")):
            return cls()
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            names = {f.name for f in fields(cls)}
            bad = sorted(set(x) - names)
            if bad:
                raise ValueError(f"unknown residency option(s) {bad}")
            return cls(**x)
        raise TypeError(f"residency: {x!r}")

    def params(self) -> P.Params:
        return P.Params(drain_n=int(self.drain_n), swaps_per_drain=int(self.swaps_per_drain),
                        halflife=float(self.halflife), hyst=float(self.hyst), min_abs=float(self.min_abs),
                        swaps_per_barrier=int(self.swaps_per_barrier), frozen_at_boot=bool(self.frozen))

    def as_dict(self) -> dict:
        return asdict(self)


def n_stage_for(moe_impl: str, rows: int, topk: int, spare_slots: Optional[int] = None) -> int:
    """Staging slots the StaticStep appends to every layer's weight pool (runtime/static_step.py `_build_moe`):
    dg-inplace stages cold experts at H + c; ref and dg-masked stage into their own buffers (0).
      1 row   : c < topk (cold slot j -> H + j).
      R rows  : c < S = the compact cold index budget (M1-T10-FIX2), S = `spare_slots` or static_step's
                DEFAULT_SPARE_SLOTS_ROWS (16), capped at R * topk (StaticStep: `min(S, ns)`). The residency slots
                start at H + S, after the staging slots the dedup kernel addresses (never H + R*topk = 36)."""
    if moe_impl != "dg-inplace":
        return 0
    if int(rows) <= 1:
        return int(topk)
    if spare_slots is None:
        from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS as spare_slots
    return min(int(spare_slots), int(topk) * int(rows))


def sf_store_for(moe_impl: str, rows: int) -> Optional[str]:
    """Where the dg-inplace step keeps the packed scale factors of the HBM-resident experts (runtime/dg_inplace.py):
    "step" (the staging launch re-packs them every step from the pointer tables), or "pool" / "resident" (packed
    once; the manager must write a promoted expert's packed SF itself). None: not dg-inplace. This is the REQUESTED
    store (env / defaults, as StaticStep reads them); `attach` re-reads the one the build ended up with."""
    if moe_impl != "dg-inplace":
        return None

    from runtime import dg_inplace as DI

    if int(rows) > 1:
        return os.environ.get(DI.ROWS_SF_STORE_ENV) or DI.ROWS_SF_STORE_DEFAULT
    return os.environ.get(DI.SF_STORE_ENV, "step")


def add_cli_args(ap) -> None:
    """The --residency* flags (bench/headline.py; bench/first_light.py declares the same names itself)."""
    ap.add_argument("--residency", choices=["off", "on"], default="off",
                    help="A2-2: host-side swap policy (m3 v26 core) on the static/graph step's pointer tables")
    ap.add_argument("--residency-drain", type=int, default=64, help="host steps between counter drains")
    ap.add_argument("--residency-swaps-per-drain", type=int, default=2)
    ap.add_argument("--residency-hyst", type=float, default=1.5)
    ap.add_argument("--residency-min-abs", type=float, default=8.0)
    ap.add_argument("--residency-halflife", type=float, default=500.0)
    ap.add_argument("--residency-cap", type=int, default=4, help="swaps applied per step (swaps_per_barrier)")
    ap.add_argument("--residency-spare", type=int, default=None, help="spare HBM slots/layer (default: headroom calc)")
    ap.add_argument("--residency-demote", type=int, default=64, help="Grace demotion slots/layer")
    ap.add_argument("--residency-reserve-gib", type=float, default=8.0, help="HBM held back in the headroom calc")
    ap.add_argument("--residency-frozen", action="store_true", help="count + drain, never swap (frozen-map arm)")
    ap.add_argument("--residency-force-random", type=int, default=0, help="forced random swaps per step (T-R1)")


def config_from_args(a) -> Optional[dict]:
    """--residency on -> ResidencyConfig fields (None = off); the inverse of add_cli_args."""
    if getattr(a, "residency", "off") != "on":
        return None
    return {"drain_n": a.residency_drain, "swaps_per_drain": a.residency_swaps_per_drain, "hyst": a.residency_hyst,
            "min_abs": a.residency_min_abs, "halflife": a.residency_halflife, "swaps_per_barrier": a.residency_cap,
            "n_spare": a.residency_spare, "n_demote": a.residency_demote, "reserve_gib": a.residency_reserve_gib,
            "frozen": bool(a.residency_frozen), "force_random": a.residency_force_random}


def spare_budget(free_bytes: Optional[int], n_layers: int, expert_bytes: int, weight_bytes: int, *, n_stage_new: int,
                 reserve_gib: float, kv_rows: int, kv_ctx: int, step_extra_bytes: int = 0,
                 sf_group_bytes: int = 0, prepaid_bytes: int = 0) -> dict:
    """HBM headroom -> affordable spare slots per layer. free_bytes = free HBM when the manager is built (before the
    StaticStep build); minus the staging slots this manager is about to allocate for the step, the KV reservation
    (docs/LEDGER.md: ~150 MB per row at 32K) and a reserve for capture / workspaces [A]; one spare slot costs one
    expert (weights + scales) in every layer. A2-2 rebase: `step_extra_bytes` = what the StaticStep build allocates
    AFTER the manager (the R-row dg-inplace step's pool-owned packed-SF copy, 12.8 GiB at 16 spare slots, plus the
    headroom its build insists on, else it falls back to the slower "step" store and the A/B is confounded);
    `sf_group_bytes` = the packed SF one more pool group adds in that copy (one spare slot = +1 group per layer);
    `prepaid_bytes` = weight slots already in HBM beyond the step's staging (a caller that grew the pool once up front,
    bench/headline.py): they are not free, but they are not new cost either."""
    kv = int(kv_rows * KV_BYTES_PER_ROW_32K * kv_ctx / 32768)
    stage = int(n_stage_new) * n_layers * int(weight_bytes)
    reserve = int(reserve_gib * GiB)
    extra = int(step_extra_bytes)
    per_slot = n_layers * (int(expert_bytes) + int(sf_group_bytes))
    out = {"free_GiB": None if free_bytes is None else round(free_bytes / GiB, 3), "kv_GiB": round(kv / GiB, 3),
           "stage_new_GiB": round(stage / GiB, 3), "reserve_GiB": round(reserve / GiB, 3),
           "step_extra_GiB": round(extra / GiB, 3),
           "per_spare_slot_all_layers_GiB": round(per_slot / GiB, 4), "affordable_per_layer": None}
    out["prepaid_GiB"] = round(int(prepaid_bytes) / GiB, 3)
    if free_bytes is not None:
        out["affordable_per_layer"] = max(0, (int(free_bytes) + int(prepaid_bytes) - kv - stage - reserve - extra)
                                          // per_slot)
    return out


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


class _Tab:
    __slots__ = ("kind", "t", "key", "name")

    def __init__(self, kind: str, t: torch.Tensor, key, name: str):
        self.kind, self.t, self.key, self.name = kind, t, key, name


class _Swap:
    __slots__ = ("layer", "e_in", "e_out", "slot", "slot_out", "ev", "demo_new", "item", "deferred")

    def __init__(self, layer, e_in, e_out, slot, slot_out, ev, demo_new, item):
        self.layer, self.e_in, self.e_out, self.slot, self.slot_out = layer, e_in, e_out, slot, slot_out
        self.ev, self.demo_new, self.item, self.deferred = ev, demo_new, item, None


# ------------------------------------------------------------------------------------------------ the manager
class ResidencyManager:
    def __init__(self, pool, model: nn.Module, cfg: Optional[ResidencyConfig] = None, *, n_stage: int = 0,
                 log: Optional[Callable[[str], None]] = None, sf_store: Optional[str] = None):
        """n_stage: the step's own staging slots per layer (`n_stage_for`). sf_store: the packed-SF store the step will
        use (`sf_store_for`; "pool" / "resident" make the budget count the packed-SF copy and its headroom)."""
        self.cfg = cfg or ResidencyConfig()
        self.pool, self.model = pool, model
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.dev = pool.device
        self.cuda = self.dev.type == "cuda"
        self.L, self.E = pool.n_layers, pool.n_experts
        self.H = [len(pool.hot[l]) for l in range(self.L)]
        self.n_stage = int(n_stage)
        if getattr(pool, "residency", None) is not None:
            raise RuntimeError("this ExpertPool already has a ResidencyManager (close() it first)")
        c = self.cfg
        weight_bytes = sum(n * (k // 2) for n, k in pool.proj_nk.values())
        free = c.hbm_free_bytes
        if free is None and self.cuda:
            free = torch.cuda.mem_get_info(self.dev)[0]
        self.sf_store_req = sf_store     # requested (sf_store_for); sf_store_got = what the step build ended up with
        self.sf_store_got: Optional[str] = None
        s_row = sum(n * (k // 32) for n, k in pool.proj_nk.values())  # packed SF bytes of one pool group (K/128 int32)
        sf_group = s_row if sf_store in ("pool", "resident") else 0
        step_extra = 0
        if sf_group:
            from runtime import dg_inplace as DI

            # the step's packed-SF copy before the residency slots (always paid: growing the pool re-keys any older
            # copy, runtime/dg_inplace.py pool_sf_key, so it is rebuilt) + the headroom its build insists on
            groups = sum(self.H) + self.L * max(self.n_stage, int(pool.spare))
            head = float(os.environ.get(DI.SF_POOL_HEADROOM_ENV, DI.SF_POOL_HEADROOM_GIB)) * GiB
            step_extra = int(groups * s_row + head)
        self.budget = spare_budget(free, self.L, pool.expert_bytes, weight_bytes,
                                   n_stage_new=max(0, self.n_stage - pool.spare), reserve_gib=c.reserve_gib,
                                   kv_rows=c.kv_rows, kv_ctx=c.kv_ctx, step_extra_bytes=step_extra,
                                   sf_group_bytes=sf_group,
                                   prepaid_bytes=max(0, int(pool.spare) - self.n_stage) * self.L * weight_bytes)
        aff = self.budget["affordable_per_layer"]
        if c.n_spare is not None:
            self.n_spare = int(c.n_spare)
            if aff is not None and self.n_spare > aff:
                self.log(f"[residency] WARNING n_spare {self.n_spare} > affordable {aff} per layer (headroom calc)")
        else:
            self.n_spare = int(c.spare_cap) if aff is None else int(min(c.spare_cap, aff))
        if self.n_spare < 1:
            raise RuntimeError(f"residency: no spare HBM slot affordable ({self.budget})")
        # weights: the pool grows to H + n_stage + n_spare (before the StaticStep build: storage moves)
        self.spare_added_B = pool.add_spare_slots(self.n_stage + self.n_spare, model)
        self.base = [self.H[l] + self.n_stage for l in range(self.L)]  # first residency slot per layer
        self.pool_gen = pool.generation
        # scales of the spare slots
        self.sspare: Dict[int, Dict[str, torch.Tensor]] = {}
        with torch.inference_mode(False), torch.no_grad():
            for l in range(self.L):
                self.sspare[l] = {}
                for p in PROJS:
                    s = pool.hbm[l][p][1]
                    self.sspare[l][p] = torch.zeros(self.n_spare, *s.shape[1:], dtype=s.dtype, device=self.dev)
        # demotion pool (Grace)
        self.D = int(min(c.n_demote, max(self.H) if self.H else 0))
        self._demo_layout()
        self.demo: Optional[HostBuffer] = None
        self.demo_mem: Optional[dict] = None
        if self.D > 0:
            nbytes = self.L * self.D * self.slot_stride
            if self.cuda:
                self.demo_mem = check_mem_available(nbytes, int(c.host_reserve_gib * GiB), "residency-demotion-pool")
            self.demo = HostBuffer(nbytes, "residency_demotion")
            if self.cuda:
                chunk = max(1, MAX_REGISTER_CHUNK // self.slot_stride) * self.slot_stride
                self.demo.register(chunk=chunk)
        # streams / counters
        self.copy_stream = torch.cuda.Stream(device=self.dev) if self.cuda else None
        with torch.inference_mode(False):
            self.cnt = torch.zeros(self.L, self.E, dtype=torch.int32, device=self.dev)
            self.cnt_host = torch.zeros(self.L, self.E, dtype=torch.int32, device="cpu", pin_memory=self.cuda)
            self._ar = torch.arange(self.E, dtype=torch.long, device=self.dev)
        # residency state (physical)
        self.slot_of: List[Dict[int, int]] = []
        self.occ: List[Dict[int, int]] = []
        self.free: List[deque] = []
        self.grace: List[Dict[int, Tuple[str, Optional[int]]]] = []
        self.demo_free: List[List[int]] = []
        for l in range(self.L):
            self.slot_of.append({e: s for s, e in enumerate(pool.hot[l])})
            self.occ.append({s: e for s, e in enumerate(pool.hot[l])})
            self.free.append(deque([self.base[l] + i, None] for i in range(self.n_spare)))
            self.grace.append({e: ("cold", None) for e in pool.cold[l]})
            self.demo_free.append(list(range(self.D)))
        self.boot_slot = [dict(self.slot_of[l]) for l in range(self.L)]
        # policy state (mirror)
        self.params = c.params()
        self.state = self._boot_state()
        self.host_steps = 0
        self._pending_read = None
        self.inflight: List[_Swap] = []
        self.st = None
        self.tabs: Dict[int, List[_Tab]] = {}
        self.resident_sf: Dict[int, Dict[str, torch.Tensor]] = {}
        self._views: Dict[tuple, torch.Tensor] = {}
        self._mutation: Optional[str] = None
        self._deferred: List[Callable[[], None]] = []
        self._rng = np.random.default_rng(int(c.seed))
        self.stats = {"issued": 0, "published": 0, "vetoed": 0, "vetoed_no_slot": 0, "vetoed_no_demotion_slot": 0,
                      "d2h_demotions": 0, "forced_flush": 0, "forced_swaps": 0, "drains": 0, "table_writes": 0,
                      "restores": 0}
        self.drain_log: List[dict] = []
        self.swap_steps: List[Tuple[int, int]] = []  # (host step, swaps issued) for steps with > 0
        pool.residency = self
        self.log(self.describe_line())

    # ---------------------------------------------------------------- layout helpers
    def _demo_layout(self) -> None:
        nk = self.pool.proj_nk
        self.demo_off: Dict[Tuple[str, str], Tuple[int, torch.dtype, Tuple[int, int]]] = {}
        off = 0
        for p, kind in ORDER6:
            n, k = nk[p]
            if kind == "weight":
                dt, shape, nb = torch.int8, (n, k // 2), n * (k // 2)
            else:
                dt, shape, nb = torch.float8_e8m0fnu, (n, k // 32), n * (k // 32)
            self.demo_off[(p, kind)] = (off, dt, shape)
            off = _align(off + nb)
        self.slot_stride = _align(off, mmap.PAGESIZE)

    def _boot_state(self) -> "P.State":
        rows = {l: {"hot": list(self.pool.hot[l]), "cold": list(self.pool.cold[l])} for l in range(self.L)}
        st = P.State.from_rows(rows, self.E, self.L, self.params)
        st.freeze = bool(self.cfg.frozen)
        return st

    @property
    def rows(self) -> int:
        return int(getattr(self.st, "rows", 1)) if self.st is not None else 1

    # ---------------------------------------------------------------- views (homes)
    def slot_view(self, l: int, s: int, p: str, kind: str) -> torch.Tensor:
        pool = self.pool
        if kind == "weight":
            return pool.hbm_full[l][p][s]
        if s < self.H[l]:
            return pool.hbm[l][p][1][s]
        i = s - self.base[l]
        if not 0 <= i < self.n_spare:
            raise IndexError(f"L{l} slot {s} is not a boot or residency slot")
        return self.sspare[l][p][i]

    def demo_view(self, l: int, d: int, p: str, kind: str, cuda: Optional[bool] = None) -> torch.Tensor:
        cuda = self.cuda if cuda is None else cuda
        key = ("demo", l, d, p, kind, cuda)
        v = self._views.get(key)
        if v is None:
            off, dt, shape = self.demo_off[(p, kind)]
            base = (l * self.D + d) * self.slot_stride + off
            with torch.inference_mode(False):
                v = self.demo.cuda_view(dt, shape, base, self.dev) if cuda else self.demo.host_view(dt, shape, base)
            self._views[key] = v
        return v

    def grace_view(self, l: int, e: int, p: str, kind: str, cuda: Optional[bool] = None) -> torch.Tensor:
        cuda = self.cuda if cuda is None else cuda
        where, d = self.grace[l][e]
        if where == "demo":
            return self.demo_view(l, d, p, kind, cuda)
        key = ("cold", l, e, p, kind, cuda)
        v = self._views.get(key)
        if v is None:
            with torch.inference_mode(False):
                v = self.pool.cold_view(l, e, p, kind, cuda=cuda)
            self._views[key] = v
        return v

    def home_view(self, l: int, e: int, p: str, kind: str) -> torch.Tensor:
        s = self.slot_of[l].get(e)
        return self.slot_view(l, s, p, kind) if s is not None else self.grace_view(l, e, p, kind)

    def home_ptr(self, l: int, e: int, p: str, kind: str) -> int:
        return self.home_view(l, e, p, kind).data_ptr()

    def loc_of(self, l: int, e: int) -> Tuple[int, int]:
        s = self.slot_of[l].get(e)
        if s is not None:
            return s, TIER_HBM
        where, d = self.grace[l][e]
        if where == "demo":
            return d, TIER_DEMOTED
        return self.pool.slot[(l, e)][1], TIER_HOST

    def address_views(self) -> Dict[int, torch.Tensor]:
        """Every address a table can name -> its uint8 bytes (host-side views): the CPU emulations' registry."""
        out: Dict[int, torch.Tensor] = {}
        for l in range(self.L):
            slots = list(range(self.H[l])) + [self.base[l] + i for i in range(self.n_spare)]
            for s in slots:
                for p, kind in ORDER6:
                    t = self.slot_view(l, s, p, kind)
                    out[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
            for e in self.pool.cold[l]:
                for p, kind in ORDER6:
                    t = self.pool.cold_view(l, e, p, kind, cuda=False)
                    out[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
            for d in range(self.D):
                for p, kind in ORDER6:
                    t = self.demo_view(l, d, p, kind, cuda=False)
                    out[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
        return out

    # ---------------------------------------------------------------- attach
    def attach(self, st) -> dict:
        """Take ownership of `st`'s pointer tables: discover them, check they name the boot homes, refuse any
        unknown copy. `st` must be built on this pool after the manager (no storage move since)."""
        if getattr(st, "pool", None) is not self.pool:
            raise ValueError("residency: the StaticStep is not built on this manager's ExpertPool")
        if getattr(st, "_pool_gen", None) != self.pool.generation or self.pool.generation != self.pool_gen:
            raise RuntimeError("residency: the pool's storage moved after the manager grew it (build order: "
                               "ResidencyManager, then StaticStep, then attach)")
        if not st.counters:
            raise ValueError("residency needs StaticStep(counters=True): the hit counter reads st.ids_step")
        if self.moved():
            raise RuntimeError("residency: attach on a moved pool (restore_boot first)")
        self.st = st
        tabs: Dict[int, List[_Tab]] = {l: [] for l in range(self.L)}
        seen = set()

        def add(l, kind, t, key, name):
            k = (t.data_ptr(), kind, key, tuple(t.shape))
            if k not in seen:
                seen.add(k)
                tabs[l].append(_Tab(kind, t, key, name))

        for sm in st.moes:
            l = sm.layer
            for key, t in sm.ptabs.items():
                add(l, "ptab", t, key, f"moes[{l}].ptabs[{key}]")
            tt = getattr(sm, "t", None)
            if isinstance(tt, dict):
                if "ptab6" in tt:
                    add(l, "ptab6", tt["ptab6"], None, f"moes[{l}].t[ptab6]")
                if "gtab" in tt:
                    add(l, "gtab", tt["gtab"], None, f"moes[{l}].t[gtab]")
                sf = tt.get("sf")
                if getattr(getattr(st, "dg", None), "sf_store", None) in ("resident", "pool") and isinstance(sf, dict):
                    self.resident_sf[l] = sf   # packed once at build: a promoted expert's packed SF is ours to write
        glue = getattr(st, "glue", None)
        for l, t in (getattr(glue, "_ptab6", None) or {}).items():
            add(l, "ptab6", t, None, f"glue._ptab6[{l}]")
        self.tabs = tabs
        if self.cuda and self.pool.host is not None:  # a memcpy must not cross a cudaHostRegister chunk (host_tier)
            for l in range(self.L):
                for e in self.pool.cold[l]:
                    for p, kind in ORDER6:
                        t = self.pool.cold_view(l, e, p, kind, cuda=False)
                        lo, hi = t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()
                        if not any(c <= lo and hi <= c + n for c, n in self.pool.host.chunks):
                            raise RuntimeError(f"residency: cold L{l} E{e} {p}.{kind} crosses a registration chunk")
        rep = self.check_tables()
        if rep["bad"]:
            self.st = None
            raise RuntimeError(f"residency: a pointer table does not name the boot homes: {rep['bad'][:4]}")
        unknown = self._scan_unknown(st)
        if unknown:
            self.st = None
            raise RuntimeError(f"residency: pointer-table copies this manager does not own: {unknown[:6]} "
                               "(register them in ResidencyManager.attach or the swap would leave them stale)")
        got = getattr(getattr(st, "dg", None), "sf_store", None)
        self.sf_store_got = got
        if self.sf_store_req in ("pool", "resident") and got == "step":
            # dg_inplace._check_sf_hbm: free HBM - the packed-SF copy < $DSV41_SF_POOL_HEADROOM_GIB -> the slower
            # "step" store (hot scales re-packed every step): the on/off A/B would be confounded -> say so loudly
            self.log(f"[residency] WARNING the step fell back to sf_store 'step' ({getattr(st.dg, 'sf_store_why', None)});"
                     f" fewer spare slots (--residency-spare) or a lower $DSV41_SF_POOL_HEADROOM_GIB keeps 'pool'")
        self.baseline()
        for l in range(self.L):  # every Grace view a publish can need, made now (view creation off the step path)
            for d in range(self.D):
                for p, kind in ORDER6:
                    self.demo_view(l, d, p, kind)
        n_tabs = sum(len(v) for v in tabs.values())
        self.log(f"[residency] attached: {n_tabs} pointer tables over {self.L} layers "
                 f"(+ cold_tab, loc{', resident SF' if self.resident_sf else ''}); freeze={int(self.state.freeze)}")
        return {"tables": n_tabs, "kinds": sorted({t.kind for v in tabs.values() for t in v})}

    def _scan_unknown(self, st) -> List[str]:
        allptr = set()
        for l in range(self.L):
            for e in range(self.E):
                for p, kind in ORDER6:
                    allptr.add(self.home_ptr(l, e, p, kind))
        known = {(t.t.data_ptr(), tuple(t.t.shape), tuple(t.t.stride())) for v in self.tabs.values() for t in v}
        found, seen = [], set()
        q = deque([(st, "st", 0)])
        while q:
            o, path, d = q.popleft()
            if id(o) in seen or o is self or o is self.pool or o is self.model:
                continue
            seen.add(id(o))
            if isinstance(o, torch.Tensor):
                if (o.dtype == torch.int64 and o.dim() in (1, 2) and o.numel() and o.shape[-1] == self.E
                        and o.numel() <= 64 * self.E):
                    if (o.data_ptr(), tuple(o.shape), tuple(o.stride())) not in known:
                        if set(o.detach().reshape(-1).tolist()) & allptr:
                            found.append(path)
                continue
            if d >= 6 or isinstance(o, nn.Module):
                continue
            if isinstance(o, dict):
                items = list(o.items())
            elif isinstance(o, (list, tuple)):
                items = list(enumerate(o))
            elif hasattr(o, "__dict__") and type(o).__module__.startswith("runtime"):
                items = list(vars(o).items())
            else:
                continue
            for k, v in items:
                q.append((v, f"{path}.{k}" if not isinstance(k, int) else f"{path}[{k}]", d + 1))
        return found

    def expected(self, l: int, tab: _Tab) -> List:
        if tab.kind == "ptab":
            p, kind = tab.key
            return [self.home_ptr(l, e, p, kind) for e in range(self.E)]
        if tab.kind == "ptab6":
            return [[self.home_ptr(l, e, p, kind) for e in range(self.E)] for p, kind in ORDER6]
        if tab.kind == "gtab":
            return [self.slot_of[l].get(e, -1) for e in range(self.E)]
        raise ValueError(tab.kind)

    def check_tables(self) -> dict:
        """Read back every owned table (+ cold_tab, loc) and compare with the published homes. Syncs."""
        bad = []
        for l, lst in self.tabs.items():
            for tab in lst:
                if tab.t.detach().cpu().tolist() != self.expected(l, tab):
                    bad.append(tab.name)
        st = self.st
        if st is not None and getattr(st, "cold_tab", None) is not None:
            want = [[0 if e in self.slot_of[l] else 1 for e in range(self.E)] for l in range(self.L)]
            if st.cold_tab.detach().cpu().tolist() != want:
                bad.append("st.cold_tab")
        want = [[list(self.loc_of(l, e)) for e in range(self.E)] for l in range(self.L)]
        if self.pool.loc.detach().cpu().tolist() != want or self.pool.loc_cpu.tolist() != want:
            bad.append("pool.loc")
        return {"bad": bad, "n": sum(len(v) for v in self.tabs.values()) + 2}

    # ---------------------------------------------------------------- per step
    def baseline(self) -> None:
        """policy.baseline on the current counter (the hook's GO: prev = cnt, ewma = 0, seen = host_steps)."""
        P.baseline(self.state, self.cnt.detach().cpu().numpy().astype(np.int64), self.host_steps)

    def after_step(self, live: Optional[torch.Tensor] = None) -> None:
        """Call once after every StaticStep step/replay (on the replay stream's order): count, publish the swaps
        whose copies are done, drain if a counter read landed, one barrier, every drain_n steps a counter read.
        live: the DSpark loop's device-side live flag (runtime/dspark_loop.py LoopBuffers.live, 1 element): steps the
        loop issues after `done` are predicated no-ops whose routed ids are stale, so they are not counted (a device
        multiply, no host sync)."""
        with torch.inference_mode():
            self._count(live)
            self.host_steps += 1
            for fn in self._deferred:
                fn()
            self._deferred = []
            self._publish_ready()
            self._maybe_drain()
            if self.cfg.force_random:
                self._queue_random(int(self.cfg.force_random))
            n = self._barrier()
            if n:
                self.swap_steps.append((self.host_steps, n))
            if self.host_steps % max(1, int(self.params.drain_n)) == 0:
                self._read_counters()

    def _count(self, live: Optional[torch.Tensor] = None) -> None:
        st = self.st
        hit = st.ids_step.unsqueeze(-1) == self._ar                       # [L, ns, E]
        if st.rows > 1:
            hit = hit & (st.uvalid_step.unsqueeze(-1) > 0)
        n = hit.sum(1, dtype=torch.int32)
        if live is not None:
            n = n * (live.reshape(-1)[:1] > 0).to(torch.int32)
        self.cnt.add_(n)

    def inject_counts(self, delta) -> None:
        """Synthetic counters (tests / probes): cnt += delta ([L, E] ints), on the stream's order."""
        with torch.inference_mode():
            self.cnt.add_(torch.as_tensor(np.asarray(delta), dtype=torch.int32, device="cpu").to(self.dev))

    def _read_counters(self) -> None:
        if self._pending_read is not None:
            hs, ev = self._pending_read
            if ev is not None:
                ev.synchronize()
            self._drain_now(hs)
        if self.cuda:
            self.cnt_host.copy_(self.cnt, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev))
        else:
            self.cnt_host.copy_(self.cnt)
            ev = None
        self._pending_read = (self.host_steps, ev)

    def _maybe_drain(self) -> None:
        if self._pending_read is None:
            return
        hs, ev = self._pending_read
        if ev is None or ev.query():
            self._drain_now(hs)

    def _drain_now(self, hs: int) -> None:
        self._pending_read = None
        cur = self.cnt_host.numpy().astype(np.int64)
        res = P.drain(self.state, self.params, cur, hs)
        self.stats["drains"] += 1
        rec = {"host_steps": hs, "steps": res.steps, "sel": res.sel, "cold": res.cold,
               "cold_share": res.cold / res.sel if res.sel else None, "queued": res.queued,
               "swaps": self.state.n_swaps, "published": self.stats["published"], "stale": self.state.stale,
               "vetoed": self.stats["vetoed"], "hot_moved": P.hot_moved(self.state)}
        self.drain_log.append(rec)
        if self.cfg.verbose:
            self.log(f"[residency] drain @{hs}: sel {res.sel} cold {res.cold} "
                     f"({rec['cold_share'] if rec['cold_share'] is None else round(rec['cold_share'], 4)}) "
                     f"queued {res.queued} swaps {self.state.n_swaps}")

    # ---------------------------------------------------------------- swaps
    def force_swaps(self, swaps: Sequence[Tuple[int, int, int]]) -> int:
        """Queue explicit swaps (layer, in_id = a cold expert, out_id = a hot expert) on the policy's pending
        queue, as a ctrl injection would; they apply at the next barrier (stale ones are dropped there)."""
        n = 0
        for l, ein, eout in swaps:
            hot, cold = self.state.hot[l], self.state.cold[l]
            if ein not in cold or eout not in hot:
                raise ValueError(f"L{l}: force swap needs a cold in_id and a hot out_id ({ein}, {eout})")
            self.state.pending.append({"layer": int(l), "in_id": int(ein), "out_id": int(eout),
                                       "hot_row": hot.index(eout), "cold_row": cold.index(ein), "ewma_in": -1.0,
                                       "ewma_out": -1.0, "ewma_coolest_hot": -1.0,
                                       "step": int(self.state.drain_steps), "forced": True})
            n += 1
        self.stats["forced_swaps"] += n
        return n

    def _queue_random(self, k: int) -> None:
        busy = {(p["layer"], p["in_id"]) for p in self.state.pending} | \
            {(p["layer"], p["out_id"]) for p in self.state.pending}
        out = []
        for _ in range(k):
            l = int(self._rng.integers(self.L))
            if not self.state.cold[l] or not self.state.hot[l]:
                continue
            ein = int(self._rng.choice(self.state.cold[l]))
            eout = int(self._rng.choice(self.state.hot[l]))
            if (l, ein) in busy or (l, eout) in busy:
                continue
            busy |= {(l, ein), (l, eout)}
            out.append((l, ein, eout))
        self.force_swaps(out)

    def _barrier(self) -> int:
        """policy.apply_pending's loop (take_barrier, layer check, is_stale, actuate, commit) with a veto."""
        if not self.state.pending:
            return 0
        if self.inflight:
            self.stats["forced_flush"] += 1
            self.flush(force=True)
        n = 0
        for item in P.take_barrier(self.state, self.params):
            if item["layer"] not in self.state.hot:
                continue
            if P.is_stale(self.state, item):
                self.state.stale += 1
                continue
            if not self._actuate(item):
                self.stats["vetoed"] += 1
                continue
            P.commit(self.state, item)
            n += 1
        return n

    def _take_free(self, l: int):
        if not self.free[l]:
            mine = [sw for sw in self.inflight if sw.layer == l]
            for sw in mine:
                self._publish(sw)
            self.inflight = [sw for sw in self.inflight if sw.layer != l]
        if not self.free[l]:
            return None
        return self.free[l].popleft()

    def _actuate(self, item: dict) -> bool:
        l, ein, eout = int(item["layer"]), int(item["in_id"]), int(item["out_id"])
        if ein in self.slot_of[l] or eout not in self.slot_of[l]:
            raise RuntimeError(f"L{l}: mirror and residency disagree on ({ein} in, {eout} out)")
        demo_new = None
        if eout not in self.grace[l]:
            if not self.demo_free[l]:
                self.stats["vetoed_no_demotion_slot"] += 1
                return False
            demo_new = self.demo_free[l][0]
        fr = self._take_free(l)
        if fr is None:
            self.stats["vetoed_no_slot"] += 1
            return False
        slot, ev_free = fr
        if demo_new is not None:
            self.demo_free[l].pop(0)
        slot_out = self.slot_of[l][eout]
        sw = _Swap(l, ein, eout, slot, slot_out, None, demo_new, item)

        def copies():
            self._copies(sw, ev_free)

        if self._mutation == "publish_before_copy":
            self._publish(sw)                     # T-R4: tables name the slot before its bytes are there
            self._deferred.append(copies)
        else:
            copies()
            self.inflight.append(sw)
        self.stats["issued"] += 1
        return True

    def _copies(self, sw: _Swap, ev_free) -> None:
        l, s = sw.layer, sw.slot
        if self.cuda:
            cs = self.copy_stream
            ctx = torch.cuda.stream(cs)
        else:
            cs, ctx = None, _nullctx()
        with ctx:
            if ev_free is not None:
                cs.wait_event(ev_free)
            for p, kind in ORDER6:
                _u8(self.slot_view(l, s, p, kind)).copy_(_u8(self.grace_view(l, sw.e_in, p, kind)),
                                                       non_blocking=True)
            if self._mutation == "corrupt_promoted":
                b = _u8(self.slot_view(l, s, "w1", "weight")).view(-1)[:1]
                b.add_(1)
            if l in self.resident_sf:
                from runtime.dg_inplace import pack_words
                for p in PROJS:
                    self.resident_sf[l][p][s].copy_(pack_words(self.slot_view(l, s, p, "scale")))
            if sw.demo_new is not None:
                so = sw.slot_out
                for p, kind in ORDER6:
                    _u8(self.demo_view(l, sw.demo_new, p, kind)).copy_(_u8(self.slot_view(l, so, p, kind)),
                                                                     non_blocking=True)
                self.stats["d2h_demotions"] += 1
            if self.cuda:
                sw.ev = torch.cuda.Event()
                sw.ev.record(cs)

    def _publish_ready(self) -> None:
        if not self.inflight:
            return
        keep = []
        for sw in self.inflight:
            if sw.ev is None or sw.ev.query():
                self._publish(sw)
            else:
                keep.append(sw)
        self.inflight = keep

    def flush(self, force: bool = True) -> None:
        """Publish every in-flight swap (force: whether or not its copy is done -- the replay stream waits on the
        copy event, the host does not)."""
        with torch.inference_mode():
            for fn in self._deferred:  # only the T-R4 mutation defers copies; never leave them behind
                fn()
            self._deferred = []
            for sw in self.inflight:
                self._publish(sw)
            self.inflight = []

    def _publish(self, sw: _Swap) -> None:
        l, ein, eout = sw.layer, sw.e_in, sw.e_out
        cur = torch.cuda.current_stream(self.dev) if self.cuda else None
        if sw.ev is not None:
            cur.wait_event(sw.ev)
        if sw.demo_new is not None:
            self.grace[l][eout] = ("demo", sw.demo_new)
        del self.slot_of[l][eout]
        del self.occ[l][sw.slot_out]
        self.slot_of[l][ein] = sw.slot
        self.occ[l][sw.slot] = ein
        self._write_tables(l, ein)
        self._write_tables(l, eout)
        ev = None
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record(cur)
        self.free[l].append([sw.slot_out, ev])
        self._rebind(l, ein)
        self._rebind(l, eout)
        self.stats["published"] += 1

    def _write_tables(self, l: int, e: int) -> None:
        skip = self._mutation[5:] if (self._mutation or "").startswith("skip:") else None
        ptr = {pk: self.home_ptr(l, e, *pk) for pk in ORDER6}
        s = self.slot_of[l].get(e)
        n = 0
        with torch.inference_mode():
            for tab in self.tabs.get(l, ()):
                if tab.kind == skip:
                    continue
                if tab.kind == "ptab":
                    tab.t[e].fill_(ptr[tab.key])
                    n += 1
                elif tab.kind == "ptab6":
                    for r, pk in enumerate(ORDER6):
                        tab.t[r, e].fill_(ptr[pk])
                    n += 6
                elif tab.kind == "gtab":
                    tab.t[e].fill_(-1 if s is None else s)
                    n += 1
            st = self.st
            if st is not None and skip != "cold_tab":
                st.cold_tab[l, e].fill_(0 if s is not None else 1)
                n += 1
            if skip != "loc":
                slot, tier = self.loc_of(l, e)
                self.pool.loc[l, e, 0].fill_(slot)
                self.pool.loc[l, e, 1].fill_(tier)
                self.pool.loc_cpu[l, e, 0] = slot
                self.pool.loc_cpu[l, e, 1] = tier
                n += 2
        self.stats["table_writes"] += n

    def _rebind(self, l: int, e: int) -> None:
        ex = self.model.layers[l].ffn.experts[e]
        with torch.inference_mode(False), torch.no_grad():
            for p in PROJS:
                lin = getattr(ex, p)
                w = self.home_view(l, e, p, "weight").view(torch.float4_e2m1fn_x2)
                s = self.home_view(l, e, p, "scale")
                lin.weight = nn.Parameter(w, requires_grad=False)
                lin.scale = nn.Parameter(s, requires_grad=False)
                lin.weight.scale = lin.scale  # model.py:222

    # ---------------------------------------------------------------- restore / close
    def moved(self) -> bool:
        return any(self.slot_of[l] != self.boot_slot[l] for l in range(self.L)) or bool(self.inflight)

    def restore_boot(self) -> int:
        """Put every boot-hot expert back in its boot slot (bytes from its demotion home), every table / param /
        loc back to the boot placement, the policy mirror to the boot rows (EWMA 0). Synchronizes. Returns the
        experts whose placement changed."""
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        self.flush(force=True)
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        changed = 0
        with torch.inference_mode():
            for l in range(self.L):
                if self.slot_of[l] == self.boot_slot[l]:
                    continue
                moved = set(self.slot_of[l]) ^ set(self.boot_slot[l])
                for e, s in self.boot_slot[l].items():
                    if self.slot_of[l].get(e) != s:
                        for p, kind in ORDER6:
                            _u8(self.slot_view(l, s, p, kind)).copy_(_u8(self.grace_view(l, e, p, kind)))
                        if l in self.resident_sf:
                            from runtime.dg_inplace import pack_words
                            for p in PROJS:
                                self.resident_sf[l][p][s].copy_(pack_words(self.slot_view(l, s, p, "scale")))
                        moved.add(e)
                self.slot_of[l] = dict(self.boot_slot[l])
                self.occ[l] = {s: e for e, s in self.boot_slot[l].items()}
                self.free[l] = deque([self.base[l] + i, None] for i in range(self.n_spare))
                for e in sorted(moved):
                    self._write_tables(l, e)
                    self._rebind(l, e)
                changed += len(moved)
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        self.state = self._boot_state()
        self._pending_read = None
        self.baseline()
        self.stats["restores"] += 1
        return changed

    def detach(self) -> None:
        """Restore the boot placement and drop the step (a later StaticStep build may then grow the pool)."""
        if self.moved():
            self.restore_boot()
        self.st = None
        self.tabs = {}
        self.resident_sf = {}

    def close(self) -> None:
        """detach, then release the demotion pool and hand the pool back (pool.residency = None)."""
        self.detach()
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        self._views = {}
        if self.demo is not None:
            self.demo.release()
            self.demo = None
        if getattr(self.pool, "residency", None) is self:
            self.pool.residency = None

    # ---------------------------------------------------------------- audit (T-R2, T-R3)
    def digest(self, l: int, e: int, where: str = "home") -> str:
        """sha256 over the expert's six tensors (ORDER6) at its current home ("home") or its Grace copy ("grace").
        Copies HBM bytes to the host."""
        h = hashlib.sha256()
        s = self.slot_of[l].get(e) if where == "home" else None
        for p, kind in ORDER6:
            if s is not None:
                t = self.slot_view(l, s, p, kind).detach()
                t = t.cpu() if t.device.type != "cpu" else t
            else:
                t = self.grace_view(l, e, p, kind, cuda=False)
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def audit(self, golden: Optional[Callable[[int, int], str]] = None, experts: str = "moved",
              tables: bool = True) -> dict:
        """I1 one published home per expert, no shared slot, spares / free slots disjoint from published, H hot per
        layer, mirror == residency; every table == the published homes (T-R3); with `golden(l, e) -> sha256`
        the bytes at every checked expert's home (T-R2; experts "moved" = not at their boot home, or "all").
        Syncs. Returns {"ok", "errors", ...}."""
        if self.cuda:
            torch.cuda.synchronize(self.dev)
        errs = []
        for l in range(self.L):
            so, oc = self.slot_of[l], self.occ[l]
            if len(set(so.values())) != len(so) or {s: e for e, s in so.items()} != oc:
                errs.append(f"L{l}: slot_of/occ inconsistent")
            if len(so) != self.H[l]:
                errs.append(f"L{l}: {len(so)} HBM-resident experts != H {self.H[l]}")
            fr = {s for s, _ in self.free[l]}
            if fr & set(oc):
                errs.append(f"L{l}: free slots {sorted(fr & set(oc))} are published")
            allowed = set(range(self.H[l])) | {self.base[l] + i for i in range(self.n_spare)}
            if not set(oc) <= allowed or not fr <= allowed or len(fr) + len(oc) != len(allowed) - \
                    sum(1 for sw in self.inflight if sw.layer == l):
                errs.append(f"L{l}: slot accounting (published {len(oc)}, free {len(fr)}, of {len(allowed)})")
            for e in range(self.E):
                if e not in so and e not in self.grace[l]:
                    errs.append(f"L{l} E{e}: no home")
            if not self.inflight and set(self.state.hot[l]) != set(so):
                errs.append(f"L{l}: policy mirror hot set != HBM-resident set")
        if tables and self.st is not None:
            errs += [f"table {b} != published homes" for b in self.check_tables()["bad"]]
        for l in range(self.L):
            for e in range(self.E):
                ex = self.model.layers[l].ffn.experts[e]
                for p in PROJS:
                    lin = getattr(ex, p)
                    if lin.weight.data_ptr() != self.home_ptr(l, e, p, "weight") or \
                            lin.scale.data_ptr() != self.home_ptr(l, e, p, "scale"):
                        errs.append(f"L{l} E{e} {p}: Expert parameter not bound to its home")
        checked = 0
        if golden is not None:
            for l in range(self.L):
                for e in range(self.E):
                    moved = self.slot_of[l].get(e) != self.boot_slot[l].get(e) or \
                        self.grace[l].get(e, ("cold",))[0] == "demo"
                    if experts == "moved" and not moved:
                        continue
                    checked += 1
                    want = golden(l, e)
                    if self.digest(l, e) != want:
                        errs.append(f"L{l} E{e}: bytes at its home differ from the checkpoint")
                    if e in self.grace[l] and e in self.slot_of[l] and self.digest(l, e, "grace") != want:
                        errs.append(f"L{l} E{e}: bytes of its Grace copy differ from the checkpoint")
        return {"ok": not errs, "errors": errs, "experts_checked": checked, "layers": self.L}

    # ---------------------------------------------------------------- report
    def hbm_bytes(self) -> dict:
        w_row = sum(n * (k // 2) for n, k in self.pool.proj_nk.values())
        s_row = sum(n * (k // 32) for n, k in self.pool.proj_nk.values())
        spare_w = self.L * self.n_spare * w_row
        spare_s = self.L * self.n_spare * s_row
        return {"spare_weight_B": spare_w, "spare_scale_B": spare_s, "pool_grown_B": self.spare_added_B,
                "spare_GiB": round((spare_w + spare_s) / GiB, 3),
                "demotion_pool_GiB": round((self.demo.nbytes if self.demo else 0) / GiB, 3)}

    def swaps_per_step(self) -> dict:
        n = self.host_steps
        tot = sum(k for _, k in self.swap_steps)
        return {"steps": n, "swaps": tot, "mean": tot / n if n else 0.0,
                "max": max((k for _, k in self.swap_steps), default=0),
                "steps_with_swaps": len(self.swap_steps), "cap": int(self.params.swaps_per_barrier)}

    def describe_line(self) -> str:
        b = self.budget
        hb = self.hbm_bytes()
        return (f"[residency] HBM free {b['free_GiB']} GiB - KV {b['kv_GiB']} - new staging {b['stage_new_GiB']} - "
                f"reserve {b['reserve_GiB']} -> affordable {b['affordable_per_layer']} spare slots/layer "
                f"({b['per_spare_slot_all_layers_GiB']} GiB each over {self.L} layers); n_spare {self.n_spare} "
                f"(+{hb['spare_GiB']} GiB HBM), n_stage {self.n_stage}, demotion pool {self.D}/layer = "
                f"{hb['demotion_pool_GiB']} GiB Grace; drain_n {self.params.drain_n} swaps/drain "
                f"{self.params.swaps_per_drain} cap/step {self.params.swaps_per_barrier} hyst {self.params.hyst} "
                f"min_abs {self.params.min_abs} halflife {self.params.halflife}")

    def describe(self) -> dict:
        free = None
        if self.cuda:
            f, tot = torch.cuda.mem_get_info(self.dev)
            free = {"free_GiB": round(f / GiB, 3), "total_GiB": round(tot / GiB, 3),
                    "allocated_GiB": round(torch.cuda.memory_allocated(self.dev) / GiB, 3)}
        return {"config": self.cfg.as_dict(), "params": self.params.as_dict(), "n_spare": self.n_spare,
                "n_stage": self.n_stage, "sf_store": {"requested": self.sf_store_req, "got": self.sf_store_got}, "n_demote": self.D, "budget": self.budget, "hbm": self.hbm_bytes(),
                "hbm_now": free, "host_steps": self.host_steps, "stats": dict(self.stats),
                "policy": {"n_swaps": self.state.n_swaps, "stale": self.state.stale, "drains": self.state.drains,
                           "cum_sel": self.state.cum_sel, "cum_cold": self.state.cum_cold,
                           "cum_cold_share": self.state.cum_cold / self.state.cum_sel if self.state.cum_sel else None,
                           "hot_moved": P.hot_moved(self.state), "freeze": self.state.freeze,
                           "pending": len(self.state.pending)},
                "swaps_per_step": self.swaps_per_step(), "drain_log": list(self.drain_log),
                "inflight": len(self.inflight), "line": self.describe_line()}


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def golden_from_index(index) -> Callable[[int, int], str]:
    """`golden(l, e)` = sha256 of the checkpoint bytes of expert e of layer l (ORDER6), read from the shards through
    runtime/ckpt.py's ShardIndex (model.runtime_load.index). For audits of moved experts only."""
    def g(l: int, e: int) -> str:
        h = hashlib.sha256()
        for p, kind in ORDER6:
            t = index.read_tensor(f"layers.{l}.ffn.experts.{e}.{p}.{kind}")
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    return g
