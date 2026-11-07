# ==== VENDORED: m3.control.policy (milo-moe-manager) — DO NOT EDIT BELOW THE END-OF-HEADER LINE ====
# Source  : ~/hermes/milo-moe-manager/m3/control/policy.py, git fa467f5a03404afe5fe8ca1b48690bc52f768f96
#           (2026-10-25); byte-identical to cards/M2/m3-hook/m3/control/policy.py, the copy Card M2 arm C ran.
# sha256  : 1aafea83557d09fd2d85089fd543c377e12f95dcdc082fd715d30bcc29494fec (of everything below the header)
# Why     : A2-2 (docs/HANDOFF-A2-2-RESIDENCY.md, docs/A2-RESIDENCY-DESIGN.md 3.2) runs m3's v26ar.2 decision core
#           host-side in runtime/residency.py. m3 is never imported at runtime; this copy is the policy. Pure numpy.
# Licence : Apache-2.0, "Milo-MoE-Manager (m3), Copyright 2026 James Meadlock" (its NOTICE); this repo is Apache-2.0.
# Checked : tests/test_residency_cpu.py pins the body's sha256, compares it with the m3 checkout when present, and
#           runs the lockstep tests on the vendored Card M2 run-2 snapshots (tests/data/cardM2/).
# ==== END-OF-HEADER ====
"""Pure policy core of the online residency controller (W5 §4.2 (g)). No torch, no I/O, no threads.

This is a port of the decision half of hook v26ar.2 (``PIN_COUNTER=atomic``), source of truth
``~/hermes/research/dsv41-prefetch-20261023/ar/hook-v26ar2/e4_adaptive.py``:

==========================  ======================================  ==========================
m3.control.policy           hook v26ar.2 ``e4_adaptive.py``          lines (v26ar.2 file)
==========================  ======================================  ==========================
``baseline``                ``_atomic_baseline``                      L602-608
``drain`` (+ ``step``)      ``_drain_atomic``                         L611-630
``queue``                   ``_queue_atomic``                         L633-674
``apply_pending``           ``apply_pending_swaps``                   L194-241
``queue_restore``           ``_queue_restore``                        L775-799
``hot_moved``               ``hot_set_diff_from_boot``                L802-803
``apply_ctrl``              ``_ctrl_poll`` (decision part)            L693-740
==========================  ======================================  ==========================

v26ar.2 differs from v26ar.1 only by env-gated telemetry (CUDA event pairs around swap copies, a
per-host-step ring, ``apply_ts`` in swaps.jsonl; ``hook-v26ar2/DIFF.md``). None of it touches a
decision, so this core is lockstep-equal to both; ``tests/test_policy_lockstep.py`` AST-extracts the
functions from each hook version and checks EWMA, pending queue, applied counts, stale counter and the
hot/cold row tables after every drain and every barrier.

Semantics kept bit-for-bit (do not "clean up" any of these, the lockstep test fails on each):

* counts are the cumulative device counter read as int32 and widened to int64; the drain delta is
  ``(cur - prev) % 2**32`` (wrap-safe);
* ``steps = max(1, host_now - seen_cursor)``; ``ewma *= 0.5 ** (steps / halflife)`` then
  ``ewma += delta`` in place on float64 (raw deltas, no normalisation);
* the telemetry cold sum is over the *current* cold membership at drain time;
* ``queue`` skips layers with any pending swap (busy), walks layers in sorted order, sorts hot EWMA
  ascending and cold EWMA descending with a *stable* argsort, pairs the k-th hottest cold row with the
  k-th coolest hot row for ``k < min(swaps_per_drain, n_cold, n_hot)`` and breaks at the first pair with
  ``v < max(hv * hyst, min_abs)``;
* ``freeze`` stops queueing only; drains still update the EWMA and barriers still apply;
* ``apply_pending`` pops at most ``swaps_per_barrier`` (0 = unlimited) FIFO and drops a swap as stale
  if the mirror no longer holds ``out_id`` at ``hot_row`` and ``in_id`` at ``cold_row``;
* ``restore_map`` (hook) queues hot<->cold swaps that restore every layer's hot SET to the boot map
  (row positions may differ; numerics do not depend on rows) and does **not** reset the EWMA. Per the AR
  round finding (§Corrections 4) the cell protocol is freeze -> restore -> drain -> baseline; the m3
  extension ``Params.restore_resets_ewma`` (default False = hook) makes ``apply_ctrl`` do it in one verb.

m3 extension (not in the hook, not lockstep-checked): ``queue_restore(exact=True)`` queues a swap
sequence that restores the identical row table (so the restored ``map_id`` equals the boot ``map_id``);
it costs up to 3 swaps per misplaced row because only hot<->cold transpositions exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

COUNTER_MOD = 1 << 32

#: knob name in ``ctrl.json`` -> (Params attribute, cast); order = hook ``_ctrl_poll`` L715-722.
CTRL_KNOBS = (
    ("swaps_per_drain", "swaps_per_drain", int),
    ("swaps_per_barrier", "swaps_per_barrier", int),
    ("hyst", "hyst", float),
    ("min_abs", "min_abs", float),
    ("halflife", "halflife", float),
    ("drain_n", "drain_n", int),
)


@dataclass
class Params:
    """Controller parameters. Defaults = shipped v26 (AR6-SPEC-v2 Arm 0 / cell B0; W5 §4.5 ``policy:``)."""

    drain_n: int = 64              # host steps between drains (side-thread cadence)
    swaps_per_drain: int = 2       # per-layer queue budget per drain
    halflife: float = 500.0        # EWMA half-life in host steps
    hyst: float = 1.5              # in-candidate must beat hyst x the out-candidate's EWMA
    min_abs: float = 8.0           # ... and at least min_abs
    swaps_per_barrier: int = 8     # swaps applied per execute_model barrier (0 = unlimited)
    frozen_at_boot: bool = True    # hook ADAPTIVE_FREEZE default at boot
    restore_resets_ewma: bool = False  # m3 extension; False = hook v26ar.2 behaviour

    @classmethod
    def from_config(cls, d: dict | None) -> "Params":
        """Accept the W5 §4.5 ``policy:`` block (``halflife_steps`` alias; unknown keys ignored)."""
        d = dict(d or {})
        if "halflife_steps" in d and "halflife" not in d:
            d["halflife"] = d.pop("halflife_steps")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class State:
    """Controller state. ``hot[l]`` / ``cold[l]`` are row-ordered expert ids (== hook ``_ar_hot_cpu``)."""

    hot: dict[int, list[int]]
    cold: dict[int, list[int]]
    boot_hot: dict[int, list[int]]
    boot_cold: dict[int, list[int]]
    ewma: np.ndarray                 # float64 [n_layers_counter, E]
    prev: np.ndarray                 # int64 [n_layers_counter, E] last counter read
    freeze: bool = True
    pending: list[dict] = field(default_factory=list)
    seen_cursor: int = 0
    drain_steps: int = 0
    n_swaps: int = 0
    stale: int = 0
    drains: int = 0
    cum_sel: int = 0
    cum_cold: int = 0

    @classmethod
    def from_rows(cls, rows: dict[int, dict[str, list[int]]], E: int, n_layers: int | None = None,
                  params: Params | None = None) -> "State":
        """Boot state from a rowmap-v0 style ``{layer: {"hot": [...], "cold": [...]}}``."""
        params = params or Params()
        Ls = sorted(int(k) for k in rows)
        n = n_layers if n_layers is not None else (max(Ls) + 1 if Ls else 0)
        hot = {L: [int(e) for e in rows[L]["hot"]] for L in Ls}
        cold = {L: [int(e) for e in rows[L]["cold"]] for L in Ls}
        return cls(hot=hot, cold=cold, boot_hot={L: list(v) for L, v in hot.items()},
                   boot_cold={L: list(v) for L, v in cold.items()},
                   ewma=np.zeros((n, E), dtype=np.float64), prev=np.zeros((n, E), dtype=np.int64),
                   freeze=bool(params.frozen_at_boot))

    # --- views -------------------------------------------------------------------------------------
    def rows(self) -> dict[int, dict[str, list[int]]]:
        return {L: {"hot": list(self.hot[L]), "cold": list(self.cold[L])} for L in sorted(self.hot)}

    def busy(self) -> set[int]:
        return {p["layer"] for p in self.pending}


@dataclass
class DrainResult:
    steps: int
    sel: int
    cold: int
    queued: int


def counter_read(cum: np.ndarray) -> np.ndarray:
    """What the hook's ``_read_counts`` returns for a cumulative counter: int32 device buffer -> int64."""
    return np.asarray(cum).astype(np.int64).astype(np.int32).astype(np.int64)


# --- ctrl verbs ------------------------------------------------------------------------------------
def baseline(state: State, cur: np.ndarray, host_steps: int) -> None:
    """``_atomic_baseline``: prev = cur, ewma = 0, seen_cursor = host_steps."""
    state.prev = np.asarray(cur, dtype=np.int64)
    state.ewma[:] = 0.0
    state.seen_cursor = int(host_steps)


def apply_ctrl(state: State, params: Params, c: dict, cur: np.ndarray | None = None,
               host_steps: int | None = None) -> dict:
    """Decision half of ``_ctrl_poll`` (L693-740) for one parsed ctrl.json document.

    Mutates ``state``/``params`` in hook order (count, freeze, knobs, baseline, restore_map, dump) and
    returns ``{"msg": [...], "count": bool|None, "restore_queued": int|None, "dump": tag|None}``; the
    caller performs the side effects (counter on/off, npz dump). ``baseline`` needs ``cur``/``host_steps``.
    """
    out = {"msg": [], "count": None, "restore_queued": None, "dump": None}
    if "count" in c:
        out["count"] = bool(c["count"])
        out["msg"].append(f"count={int(out['count'])}")
    if "freeze" in c:
        state.freeze = bool(c["freeze"])
        out["msg"].append(f"freeze={int(state.freeze)}")
    for key, attr, cast in CTRL_KNOBS:
        if key in c:
            setattr(params, attr, cast(c[key]))
            out["msg"].append(f"{attr}={getattr(params, attr)}")
    if c.get("baseline"):
        if cur is None or host_steps is None:
            raise ValueError("ctrl baseline needs the current counter and host_steps")
        baseline(state, cur, host_steps)
    if c.get("restore_map"):
        out["restore_queued"] = queue_restore(state)
        if params.restore_resets_ewma:
            state.ewma[:] = 0.0
        out["msg"].append(f"restore_map queued={out['restore_queued']}")
    if c.get("dump"):
        out["dump"] = str(c["dump"])
    return out


# --- side thread -----------------------------------------------------------------------------------
def step(state: State, params: Params, delta: np.ndarray, steps: int) -> list[dict]:
    """W5 ``policy.step``: fold one drain's count delta into the EWMA and return the swaps queued."""
    state.ewma *= 0.5 ** (steps / params.halflife)
    state.ewma += delta
    n0 = len(state.pending)
    queue(state, params)
    return state.pending[n0:]


def drain(state: State, params: Params, cur: np.ndarray, host_steps: int) -> DrainResult:
    """``_drain_atomic``: cumulative counter read ``cur`` (int64 of the int32 buffer) at ``host_steps``."""
    cur = np.asarray(cur, dtype=np.int64)
    delta = (cur - state.prev) % COUNTER_MOD
    state.prev = cur
    host_now = int(host_steps)
    steps = max(1, host_now - int(state.seen_cursor))
    state.seen_cursor = host_now
    state.drain_steps += steps
    state.ewma *= 0.5 ** (steps / params.halflife)
    state.ewma += delta
    sel = int(delta.sum())
    cold = 0
    for L, cl in state.cold.items():
        if L < delta.shape[0]:
            cold += int(delta[L, cl].sum())
    state.cum_sel += sel
    state.cum_cold += cold
    state.drains += 1
    q = queue(state, params)
    return DrainResult(steps=steps, sel=sel, cold=cold, queued=q)


def queue(state: State, params: Params) -> int:
    """``_queue_atomic``: append swap decisions to ``state.pending``; returns how many were queued."""
    if state.freeze:
        return 0
    busy = state.busy()
    hyst, min_abs, budget = float(params.hyst), float(params.min_abs), int(params.swaps_per_drain)
    queued = []
    for L in sorted(state.hot):
        if L in busy or L >= state.ewma.shape[0]:
            continue
        ew = state.ewma[L]
        hot = state.hot[L]
        cold = state.cold[L]
        hv = ew[hot]
        cv = ew[cold]
        h_order = np.argsort(hv, kind="stable")
        c_order = np.argsort(-cv, kind="stable")
        coolest = float(hv[h_order[0]]) if len(h_order) else 0.0
        for k in range(min(budget, len(c_order), len(h_order))):
            r_c = int(c_order[k])
            r_h = int(h_order[k])
            v = float(cv[r_c])
            hvv = float(hv[r_h])
            if v < max(hvv * hyst, min_abs):
                break
            queued.append({
                "layer": L, "in_id": int(cold[r_c]), "out_id": int(hot[r_h]), "hot_row": r_h,
                "cold_row": r_c, "ewma_in": v, "ewma_out": hvv, "ewma_coolest_hot": coolest,
                "step": int(state.drain_steps),
            })
    state.pending.extend(queued)
    return len(queued)


# --- main thread barrier ---------------------------------------------------------------------------
def take_barrier(state: State, params: Params) -> list[dict]:
    """Pop the swaps one barrier may apply (FIFO, at most ``swaps_per_barrier``; 0 = all)."""
    cap = int(params.swaps_per_barrier or 0)
    pending = state.pending
    if cap > 0 and len(pending) > cap:
        state.pending = pending[cap:]
        return pending[:cap]
    state.pending = []
    return pending


def is_stale(state: State, item: dict) -> bool:
    L = item["layer"]
    return (state.hot[L][item["hot_row"]] != item["out_id"]
            or state.cold[L][item["cold_row"]] != item["in_id"])


def commit(state: State, item: dict) -> None:
    """Flip the mirror for one swap (the actuator flips two device row-map entries at the same point)."""
    L = item["layer"]
    state.hot[L][item["hot_row"]] = item["in_id"]
    state.cold[L][item["cold_row"]] = item["out_id"]
    state.n_swaps += 1


def apply_pending(state: State, params: Params, actuate=None) -> list[dict]:
    """``apply_pending_swaps``: one execute_model barrier. Returns the swaps applied, in order.

    ``actuate(item)``, if given, is called before each mirror flip (row copies, then the two row-map
    entries; I3 ordering is the actuator's job). A swap queued against an older map is dropped as stale.
    """
    applied = []
    for item in take_barrier(state, params):
        if item["layer"] not in state.hot:
            continue
        if is_stale(state, item):
            state.stale += 1
            continue
        if actuate is not None:
            actuate(item)
        commit(state, item)
        applied.append(item)
    return applied


# --- restore ---------------------------------------------------------------------------------------
def queue_restore(state: State, exact: bool = False) -> int:
    """``_queue_restore`` (hook, ``exact=False``): restore each non-busy layer's hot SET to the boot map.

    ``exact=True`` (m3 extension) restores the identical row table (hot and cold row order) with a
    simulated swap chain; every queued swap is valid against the state it will meet if applied FIFO.
    """
    busy = state.busy()
    queued = []
    for L in sorted(state.hot):
        if L in busy:
            continue
        hot = state.hot[L]
        cold = state.cold[L]
        if not exact:
            target = set(state.boot_hot[L])
            out_rows = [r for r, e in enumerate(hot) if e not in target]
            in_rows = [r for r, e in enumerate(cold) if e in target]
            for rh, rc in zip(out_rows, in_rows):
                queued.append({
                    "layer": L, "in_id": int(cold[rc]), "out_id": int(hot[rh]), "hot_row": rh,
                    "cold_row": rc, "ewma_in": -1.0, "ewma_out": -1.0, "ewma_coolest_hot": -1.0,
                    "step": int(state.drain_steps), "restore": True,
                })
        else:
            queued.extend(_exact_chain(L, list(hot), list(cold), state.boot_hot[L], state.boot_cold[L],
                                       int(state.drain_steps)))
    state.pending.extend(queued)
    return len(queued)


def _exact_chain(L, hot, cold, t_hot, t_cold, step_):
    out = []

    def sw(rh, rc):
        out.append({"layer": L, "in_id": int(cold[rc]), "out_id": int(hot[rh]), "hot_row": rh,
                    "cold_row": rc, "ewma_in": -1.0, "ewma_out": -1.0, "ewma_coolest_hot": -1.0,
                    "step": step_, "restore": True, "exact": True})
        hot[rh], cold[rc] = cold[rc], hot[rh]

    for r in range(len(hot)):
        e = t_hot[r]
        if hot[r] == e:
            continue
        if e in cold:
            sw(r, cold.index(e))
        else:
            r2 = hot.index(e)
            wrong = [c for c in range(len(cold)) if cold[c] != t_cold[c]]
            c = wrong[0] if wrong else 0
            sw(r2, c)
            sw(r, c)
    for c in range(len(cold)):
        e = t_cold[c]
        if cold[c] == e:
            continue
        c2 = cold.index(e)
        sw(0, c2)
        sw(0, c)
        sw(0, c2)
    assert hot == list(t_hot) and cold == list(t_cold)
    return out


def hot_moved(state: State) -> int:
    """``hot_set_diff_from_boot``: experts in a hot set now that were not hot at boot (summed)."""
    return sum(len(set(state.hot[L]) - set(state.boot_hot[L])) for L in state.hot)


# --- row-map view (I3) -----------------------------------------------------------------------------
def row_map(state: State, L: int, E: int) -> list[int]:
    """Device row-map encoding of layer ``L`` (hook ``swap_rows_known``): hot row r -> r, cold row c -> -(c+1)."""
    rm: list = [None] * E
    for r, e in enumerate(state.hot[L]):
        rm[e] = r
    for c, e in enumerate(state.cold[L]):
        rm[e] = -(c + 1)
    return rm
