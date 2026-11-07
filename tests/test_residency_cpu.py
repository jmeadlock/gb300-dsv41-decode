"""Mac CPU tests of A2-2 (runtime/residency.py + runtime/residency_policy.py): `.venv/bin/python -m pytest -q tests/`.

Policy (docs/A2-RESIDENCY-DESIGN.md 3.2, 3.4 T-R3/T-R4; handoff step 3):
  * the vendored policy is m3's `m3/control/policy.py` byte for byte (pinned sha256; equal to the m3 checkout when it
    is on this machine) -- m3 itself is never imported or executed;
  * lockstep on Card M2 run 2 arm C (m3's own policy live on the GB300; tests/data/cardM2/, provenance.json): (a) the
    2,165 recorded swaps replayed through the vendored barrier from the boot rowmap reproduce all 16 dumped mirrors
    row for row, 0 stale; (b) `queue` on each dump's EWMA, from the rows before that dump's last drain, reproduces
    that drain's recorded decisions exactly (layer, in, out, EWMA values bit-equal) -- i.e. our copy decides what
    m3 decided on the box; (c) G5b-style pair drains (snapshot k -> k+1 as one recorded counter delta) equal an
    independent re-derivation of the documented rules (wrap delta, decay, sel/cold, queue), several knob sets;
    (d) the harness is not vacuous: decision-changing mutations of the vendored source are each caught;
  * with the m3 checkout present, (a)+(b) also run on the raw receipts of arms A and A2 (the hand hook v26ar.2).

Mechanism, on the CPU emulations of tests/test_static_emulated_cpu.py (registry-backed gather / staging):
  * T-R1: residency off vs on (forced random swaps every step + policy swaps from synthetic counters), fp32 logits
    torch.equal at every position, ref and dg-inplace (both scale stores); a promoted expert is routed the very step
    after it is published; Generator plumbing (residency=) gives byte-identical greedy records;
  * T-R2/T-R3: audit -- invariants, every pointer table / cold_tab / loc == the published homes, Expert params
    bound to their homes, sha256 of every moved expert's bytes == the checkpoint shards;
  * the manager's barrier is apply_pending (a shadow policy fed the same counter reads stays in lockstep);
  * T-R4 mutations: publish before the copy -> logits diverge; corrupt a promoted byte -> audit + logits; skip one
    table kind (ptabs / ptab6 / gtab) -> audit; an unknown pointer-table copy -> attach refuses;
  * restore_boot / reset_static: boot placement, params and tables back; a fresh step == the reference;
  * the spare-slot headroom calc on the real model's numbers.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

from runtime import residency_policy as P
from runtime.residency import (ORDER6, ResidencyConfig, ResidencyManager, golden_from_index, n_stage_for,
                               spare_budget)
from test_static_emulated_cpu import _fake_static_kernels, _ref_run, emu  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "tests", "data", "cardM2")
BUNDLE = os.path.join(DATA, "run2-armC.npz")
M3 = os.path.expanduser("~/hermes/milo-moe-manager")
M3_POLICY = os.path.join(M3, "m3", "control", "policy.py")
M3_RUN2 = os.path.join(M3, "cards", "M2", "results-run2")
M3_ROWMAP = os.path.join(M3, "cards", "M2", "m3-hook", "rowmap-mixed-ar-v1.json")
POLICY_SHA = "1aafea83557d09fd2d85089fd543c377e12f95dcdc082fd715d30bcc29494fec"
HEADER_END = "# ==== END-OF-HEADER ====\n"
need_m3 = pytest.mark.skipif(not os.path.exists(M3_RUN2), reason="milo-moe-manager checkout not on this machine")


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ================================================================================================ policy: vendoring
def _vendored_body() -> str:
    with open(os.path.join(REPO, "runtime", "residency_policy.py")) as f:
        txt = f.read()
    assert txt.count(HEADER_END) == 1
    return txt.split(HEADER_END, 1)[1]


def test_vendored_policy_is_m3s_file_byte_for_byte():
    body = _vendored_body()
    assert hashlib.sha256(body.encode()).hexdigest() == POLICY_SHA
    if os.path.exists(M3_POLICY):
        with open(M3_POLICY) as f:
            assert f.read() == body, "m3/control/policy.py changed since it was vendored: re-vendor + re-run lockstep"
    with open(os.path.join(DATA, "provenance.json")) as f:
        prov = json.load(f)
    with open(BUNDLE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == prov["bundle_sha256"]
    assert prov["n_swaps"] == 2165 and prov["n_snapshots"] == 16 and "DRAIN_N=64" in prov["ctrl_go"]
    src = open(os.path.join(REPO, "runtime", "residency.py")).read()
    assert "import m3" not in src and "from m3" not in src and "vllm" not in src.replace("vLLM", "")


# ================================================================================================ policy: lockstep
def _load_bundle():
    z = np.load(BUNDLE)
    sw = [{"layer": int(a), "in_id": int(b), "out_id": int(c), "step": int(d), "apply_step": int(e),
           "ewma_in": float(f), "ewma_out": float(g), "ewma_coolest_hot": float(h)}
          for a, b, c, d, e, f, g, h in zip(z["sw_layer"], z["sw_in"], z["sw_out"], z["sw_step"], z["sw_apply_step"],
                                            z["sw_ewma_in"], z["sw_ewma_out"], z["sw_ewma_coolest"])]
    rows = {l: {"hot": [int(x) for x in z["boot_hot"][l]], "cold": [int(x) for x in z["boot_cold"][l]]}
            for l in range(z["boot_hot"].shape[0])}
    snaps = [{"host_steps": int(z["snap_host_steps"][i]), "hot": z["snap_hot"][i], "cold": z["snap_cold"][i],
              "ewma": z["snap_ewma"][i], "cnt": z["snap_cnt"][i]} for i in range(len(z["snap_host_steps"]))]
    cum = np.cumsum(z["drain_dsteps"])
    drains = [{"host_steps": int(h), "drain_steps": int(c)} for h, c in zip(z["drain_host_steps"], cum)]
    return rows, sw, snaps, drains


def _load_raw_arm(arm: str):
    snap = os.path.join(M3_RUN2, f"arm-{arm}", "snap")
    with open(M3_ROWMAP) as f:
        rm = json.load(f)["layers"]
    rows = {int(l): {"hot": list(v["hot"]), "cold": list(v["cold"])} for l, v in rm.items()}
    sw = [r for r in map(json.loads, open(os.path.join(snap, "swaps.jsonl"))) if "layer" in r]
    snaps = []
    for p in sorted(glob.glob(os.path.join(snap, "counts-post-*.npz"))):
        z = np.load(p)
        snaps.append({"host_steps": int(z["host_steps"]), "hot": z["hot"], "cold": z["cold"], "ewma": z["ewma"],
                      "cnt": z["cnt"]})
    tel = [json.loads(l) for l in open(os.path.join(snap, "telemetry.jsonl"))]
    cum, drains = 0, []
    for t in tel:
        if t.get("dsteps", 0) > 0:
            cum += t["dsteps"]
            drains.append({"host_steps": t["host_steps"], "drain_steps": cum})
    return rows, sw, snaps, drains


def _rows_eq(state, snap) -> bool:
    return all(state.hot[l] == [int(x) for x in snap["hot"][l]] and state.cold[l] == [int(x) for x in snap["cold"][l]]
               for l in range(len(snap["hot"])))


def _replay_check(rows, sw, snaps, pol=P, tolerate_race=False) -> dict:
    """(a): replay the recorded swaps (in apply order, rows resolved on the live mirror) through take_barrier /
    is_stale / commit (apply_pending) from the boot rows; at every dump the mirror must equal the dumped rows."""
    st = pol.State.from_rows(rows, 384, len(rows))
    i, ok, races = 0, 0, 0
    for z in snaps:
        hs = z["host_steps"]
        j = i
        while j < len(sw) and sw[j]["apply_step"] < hs:
            j += 1
        k = j
        while k < len(sw) and sw[k]["apply_step"] <= hs:
            k += 1
        for stop in ((j, k) if tolerate_race else (k,)):
            while i < stop:
                r = sw[i]
                L = r["layer"]
                st.pending.append({"layer": L, "in_id": r["in_id"], "out_id": r["out_id"],
                                   "hot_row": st.hot[L].index(r["out_id"]), "cold_row": st.cold[L].index(r["in_id"])})
                pol.apply_pending(st, pol.Params(swaps_per_barrier=0))
                i += 1
            if _rows_eq(st, z):
                ok += 1
                races += stop != k
                break
        else:
            raise AssertionError(f"mirror differs from the dump at host step {hs}")
        while i < k:  # (race tolerated: the dump came before these barriers) -> apply them now
            r = sw[i]
            L = r["layer"]
            st.pending.append({"layer": L, "in_id": r["in_id"], "out_id": r["out_id"],
                               "hot_row": st.hot[L].index(r["out_id"]), "cold_row": st.cold[L].index(r["in_id"])})
            pol.apply_pending(st, pol.Params(swaps_per_barrier=0))
            i += 1
    return {"dumps_equal": ok, "swaps": st.n_swaps, "stale": st.stale, "races": races}


def _decision_check(sw, snaps, drains, pol=P, params=None, tolerate_race=False) -> dict:
    """(b): per dump, the last drain D before it: undo D's swaps that the dump's mirror already shows, then
    `queue` on the dumped EWMA must reproduce D's recorded decisions (ids + EWMA values bit-equal). tolerate_race:
    a dump at the very host step of a drain may hold the EWMA from before that drain (the hook drains on a side
    thread) -> then the previous drain must match instead."""
    params = params or pol.Params()
    by = {}
    for r in sw:
        by.setdefault(r["step"], []).append(r)
    n_dec = n_eq = races = 0
    for z in snaps:
        hs = z["host_steps"]
        cands = [d for d in drains if d["host_steps"] <= hs]
        tries = [cands[-1]] + ([cands[-2]] if tolerate_race and cands[-1]["host_steps"] == hs else [])
        for t, D in enumerate(tries):
            got, want = _queue_on_dump(z, by.get(D["drain_steps"], []), pol, params)
            if got == want:
                break
        else:
            raise AssertionError(f"dump @{hs}: queue decided {got[:3]}... vs recorded {want[:3]}...")
        races += t
        n_dec += len(want)
        n_eq += 1
    return {"dumps": n_eq, "decisions": n_dec, "races": races}


def _queue_on_dump(z, recs, pol, params):
    hot = {l: [int(x) for x in z["hot"][l]] for l in range(len(z["hot"]))}
    cold = {l: [int(x) for x in z["cold"][l]] for l in range(len(z["cold"]))}
    for r in reversed(recs):
        L = r["layer"]
        if r["in_id"] in hot[L] and r["out_id"] in cold[L]:
            rh, rc = hot[L].index(r["in_id"]), cold[L].index(r["out_id"])
            hot[L][rh], cold[L][rc] = r["out_id"], r["in_id"]
    st = pol.State.from_rows({l: {"hot": hot[l], "cold": cold[l]} for l in hot}, 384, len(hot))
    st.ewma[:] = z["ewma"]
    st.freeze = False
    pol.queue(st, params)
    got = [(q["layer"], q["in_id"], q["out_id"], q["ewma_in"], q["ewma_out"], q["ewma_coolest_hot"])
           for q in st.pending]
    want = [(r["layer"], r["in_id"], r["out_id"], r["ewma_in"], r["ewma_out"], r["ewma_coolest_hot"]) for r in recs]
    return got, want


def _oracle_drain(z0, z1, rows, p):
    """Independent re-derivation of the documented drain + queue rules (policy.py docstring) for one pair drain."""
    delta = (z1["cnt"].astype(np.int64) - z0["cnt"].astype(np.int64)) % (1 << 32)
    steps = max(1, z1["host_steps"] - z0["host_steps"])
    ewma = z0["ewma"].copy()
    ewma *= 0.5 ** (steps / p.halflife)
    ewma += delta
    sel = int(delta.sum())
    cold = sum(int(delta[l, rows[l]["cold"]].sum()) for l in rows)
    dec = []
    for l in sorted(rows):
        hot, cl = rows[l]["hot"], rows[l]["cold"]
        hv, cv = ewma[l][hot], ewma[l][cl]
        ho, co = np.argsort(hv, kind="stable"), np.argsort(-cv, kind="stable")
        for k in range(min(p.swaps_per_drain, len(co), len(ho))):
            v, h = float(cv[co[k]]), float(hv[ho[k]])
            if v < max(h * p.hyst, p.min_abs):
                break
            dec.append((l, cl[int(co[k])], hot[int(ho[k])], v, h))
    return ewma, steps, sel, cold, dec


KNOBS = [P.Params(), P.Params(swaps_per_drain=8), P.Params(hyst=1.0, min_abs=4.0),
         P.Params(hyst=1.1, min_abs=8.0, swaps_per_drain=4), P.Params(min_abs=1.0, swaps_per_drain=16, halflife=50.0)]


def _pair_drain_check(snaps, pol=P) -> int:
    """(c): each consecutive dump pair as ONE drain (a real recorded count delta + the dump's own EWMA)."""
    n = 0
    for p in KNOBS:
        p = pol.Params(**p.as_dict())
        for z0, z1 in zip(snaps[:-1], snaps[1:]):
            rows = {l: {"hot": [int(x) for x in z0["hot"][l]], "cold": [int(x) for x in z0["cold"][l]]}
                    for l in range(len(z0["hot"]))}
            st = pol.State.from_rows(rows, 384, len(rows))
            st.ewma[:] = z0["ewma"]
            st.prev = z0["cnt"].astype(np.int64)
            st.seen_cursor = z0["host_steps"]
            st.freeze = False
            res = pol.drain(st, p, z1["cnt"], z1["host_steps"])
            ewma, steps, sel, cold, dec = _oracle_drain(z0, z1, rows, p)
            assert res.steps == steps and res.sel == sel and res.cold == cold
            assert np.array_equal(st.ewma, ewma), "EWMA not bit-exact"
            got = [(q["layer"], q["in_id"], q["out_id"], q["ewma_in"], q["ewma_out"]) for q in st.pending]
            assert got == dec and res.queued == len(dec)
            pol.apply_pending(st, p)
            assert st.n_swaps == min(len(dec), p.swaps_per_barrier or len(dec)) and st.stale == 0
            n += 1
    return n


def test_lockstep_card_m2_arm_c_replay_reproduces_every_dump():
    rows, sw, snaps, _ = _load_bundle()
    r = _replay_check(rows, sw, snaps)
    assert r == {"dumps_equal": 16, "swaps": 2165, "stale": 0, "races": 0}, r


def test_lockstep_card_m2_arm_c_queue_reproduces_m3_live_decisions():
    _, sw, snaps, drains = _load_bundle()
    r = _decision_check(sw, snaps, drains)
    assert r["dumps"] == 16 and r["decisions"] > 30 and r["races"] == 0, r


def test_lockstep_card_m2_pair_drains_equal_the_documented_rules():
    _, _, snaps, _ = _load_bundle()
    assert _pair_drain_check(snaps) == 15 * len(KNOBS)


@need_m3
@pytest.mark.parametrize("arm", ["A", "C", "A2"])
def test_lockstep_raw_receipts_all_arms(arm):
    rows, sw, snaps, drains = _load_raw_arm(arm)
    r = _replay_check(rows, sw, snaps, tolerate_race=True)
    assert r["dumps_equal"] == len(snaps) and r["stale"] == 0 and r["swaps"] == len(sw), r
    assert r["races"] <= 1, r  # one dump raced its own barrier on A2 (dump between apply and mirror flip)
    d = _decision_check(sw, snaps, drains, tolerate_race=True)
    assert d["dumps"] == len(snaps) and d["races"] <= 1, d
    assert _pair_drain_check(snaps) == (len(snaps) - 1) * len(KNOBS)


MUTANTS = {
    "hysteresis dropped": ("if v < max(hvv * hyst, min_abs):", "if v < min_abs:"),
    "pair every cold with the coolest hot": ("r_h = int(h_order[k])", "r_h = int(h_order[0])"),
    "hot sorted descending": ('h_order = np.argsort(hv, kind="stable")', 'h_order = np.argsort(-hv, kind="stable")'),
    "half the decay in drain": ("    state.drain_steps += steps\n    state.ewma *= 0.5 ** (steps / params.halflife)",
                                "    state.drain_steps += steps\n    state.ewma *= 0.5 ** (steps / (2 * params.halflife))"),
    "steps off by one": ("steps = max(1, host_now - int(state.seen_cursor))",
                         "steps = max(1, host_now - int(state.seen_cursor)) + 1"),
    "no wrap-safe delta": ("delta = (cur - state.prev) % COUNTER_MOD", "delta = np.abs(cur - state.prev)"),
}


def _wrap_check(pol=P) -> None:
    """The int32 device counter wraps: cumulative reads 2^31 - 8 -> -2^31 + 8 are a delta of 16."""
    st = pol.State.from_rows({0: {"hot": [0, 1], "cold": [2, 3]}}, 4, 1)
    prev = pol.counter_read(np.array([[2 ** 31 - 8, 0, 0, 0]]))
    cur = pol.counter_read(np.array([[2 ** 31 + 8, 0, 3, 0]]))
    assert int(prev[0, 0]) == 2 ** 31 - 8 and int(cur[0, 0]) == -2 ** 31 + 8
    pol.baseline(st, prev, 0)
    res = pol.drain(st, pol.Params(), cur, 10)
    assert res.sel == 19 and res.cold == 3 and float(st.ewma[0, 0]) == 16.0


def test_drain_wraps_the_int32_counter():
    _wrap_check()


@pytest.mark.parametrize("name", list(MUTANTS))
def test_lockstep_catches_mutations(name):
    old, new = MUTANTS[name]
    src = _vendored_body()
    assert src.count(old) == 1, name
    mod = types.ModuleType("residency_policy_mutant")
    sys.modules["residency_policy_mutant"] = mod
    try:
        exec(compile(src.replace(old, new), "residency_policy_mutant", "exec"), mod.__dict__)
        rows, sw, snaps, drains = _load_bundle()
        caught = []
        for fn in (lambda: _decision_check(sw, snaps, drains, pol=mod), lambda: _pair_drain_check(snaps, pol=mod),
                   lambda: _wrap_check(pol=mod)):
            try:
                fn()
            except AssertionError:
                caught.append(True)
        assert caught, f"mutation {name!r} not detected"
    finally:
        sys.modules.pop("residency_policy_mutant", None)


def test_hotmap_loads_m3_rowmap_v0(tmp_path):
    from runtime.placement import HotMap, PlacementConfig

    rows, _, _, _ = _load_bundle()
    p = tmp_path / "rowmap.json"
    p.write_text(json.dumps({"layers": {str(l): v for l, v in rows.items()}, "summary": {}}))
    cfg = PlacementConfig(n_layers=40, n_routed_experts=384, engram_layer_ids=(), compress_ratios=(), dim=1,
                          moe_inter_dim=1)
    hm = HotMap.load(str(p), cfg)
    assert all(hm.hot_ids(l) == tuple(sorted(rows[l]["hot"])) for l in range(40))
    seed = HotMap.load(os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"), cfg)
    assert all(seed.hot_ids(l) == hm.hot_ids(l) for l in range(40))  # A2-1's seed map = this rowmap's hot sets


# ================================================================================================ headroom calc
def test_spare_budget_on_the_real_model_numbers():
    """docs/first-light-rows6-20261107.json rows_mode.hbm_1row free 29.91 GiB after the 1-row build; an expert
    18,800,640 B (A2 memo 0.1); KV 1 row at 32K ~150 MB (LEDGER); reserve 8 GiB [A]."""
    free = int(29.91 * 2 ** 30)
    eb, wb = 18_800_640, 3 * 5_898_240
    b = spare_budget(free, 40, eb, wb, n_stage_new=0, reserve_gib=8.0, kv_rows=1, kv_ctx=32768)
    want = (free - int(150e6) - 8 * 2 ** 30) // (40 * eb)
    assert b["affordable_per_layer"] == want and want >= ResidencyConfig().spare_cap
    b6 = spare_budget(free, 40, eb, wb, n_stage_new=6, reserve_gib=8.0, kv_rows=6, kv_ctx=32768)
    assert b6["affordable_per_layer"] == (free - int(6 * 150e6) - 6 * 40 * wb - 8 * 2 ** 30) // (40 * eb)
    # A2-2 rebase: the 6-row dg-inplace step builds a pool-owned packed-SF copy AFTER the manager (sum(H) + 40 x 16
    # groups, 1.05 MiB each, + a 6 GiB build headroom) and each spare slot adds one more SF group
    s_row = 2 * 2304 * 160 + 5120 * 72      # packed SF bytes of one pool group (w1/w3 [2304, 160], w2 [5120, 72])
    extra = 40 * (295 + 16) * s_row + 6 * 2 ** 30
    bp = spare_budget(free, 40, eb, wb, n_stage_new=0, reserve_gib=8.0, kv_rows=6, kv_ctx=32768,
                      step_extra_bytes=extra, sf_group_bytes=s_row)
    assert bp["affordable_per_layer"] == (free - int(6 * 150e6) - 8 * 2 ** 30 - extra) // (40 * (eb + s_row))
    assert bp["affordable_per_layer"] < b6["affordable_per_layer"] and bp["step_extra_GiB"] > 6
    assert spare_budget(None, 40, eb, wb, n_stage_new=0, reserve_gib=8, kv_rows=1, kv_ctx=1)["affordable_per_layer"] \
        is None
    # the 1-row step stages into topk slots; the 6-row step by a compact cold index into S = 16 (M1-T10-PORT; FIX's
    # R*topk = 36 per layer was 25.5 GB and OOMed), S capped at R*topk; the residency slots start after them
    assert n_stage_for("dg-inplace", 1, 6) == 6 and n_stage_for("dg-inplace", 6, 6) == 16
    assert n_stage_for("dg-inplace", 6, 6, 8) == 8 and n_stage_for("dg-inplace", 2, 6, 16) == 12
    assert n_stage_for("ref", 6, 6) == 0 and n_stage_for("dg-masked", 1, 6) == 0 and n_stage_for("dg-masked", 6, 6) == 0
    from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS
    assert DEFAULT_SPARE_SLOTS_ROWS == 16
    assert ResidencyConfig.from_any("off") is None and ResidencyConfig.from_any(None) is None
    assert ResidencyConfig.from_any("on") == ResidencyConfig() and ResidencyConfig.from_any({"drain_n": 8}).drain_n == 8
    with pytest.raises(ValueError):
        ResidencyConfig.from_any({"drain": 8})


def test_residency_stage_rule():
    """A2-2 FIX: the step keeps 8 cold-staging slots at 6 rows (topk at 1 row); the pool's other spare slots are residency's."""
    from runtime.residency import residency_stage
    c = ResidencyConfig()
    assert residency_stage(c, "dg-inplace", 6, 6) == 8 and residency_stage(c, "dg-inplace", 1, 6) == 6
    assert residency_stage(c, "dg-inplace", 6, 6, spare_slots=12) == 12      # an explicit --spare-slots is the staging
    assert residency_stage(ResidencyConfig(stage_keep=10), "dg-inplace", 6, 6, spare_slots=12) == 10
    assert residency_stage(c, "dg-inplace", 2, 6) == 8 and residency_stage(c, "dg-inplace", 1, 6, 16) == 6
    assert residency_stage(ResidencyConfig(stage_keep=40), "dg-inplace", 2, 6) == 12   # capped at R * topk
    assert residency_stage(c, "ref", 6, 6) == 0 and residency_stage(c, "dg-masked", 6, 6) == 0


def test_manager_promotes_into_the_pools_free_spare_slots_without_growing(emu):
    """Box r1: pool pre-grown to 16, free HBM below the reserve -> 'no spare HBM slot affordable'. Now the 16 - 8
    slots the pool already holds are the residency slots: no growth, no new HBM, whatever the free HBM."""
    model, pool, _ = emu("mini4")
    pool.add_spare_slots(16, model)
    gen0, spare0 = pool.generation, pool.spare
    # the box numbers: 8.6 GiB free, reserve 8 GiB -> nothing new is affordable
    rm = ResidencyManager(pool, model, ResidencyConfig(n_demote=2, hbm_free_bytes=int(8.646 * 2 ** 30)), n_stage=8)
    assert rm.n_spare == 8 and rm.existing == 8 and rm.budget["new_per_layer"] == 0
    assert rm.spare_added_B == 0 and pool.generation == gen0 and pool.spare == spare0
    assert rm.base == [len(pool.hot[l]) + 8 for l in range(pool.n_layers)]
    for l in range(rm.L):   # residency slots sit behind the 8 staging slots, inside the pool's existing block
        assert all(rm.base[l] + i < pool.hbm_full[l]["w1"].size(0) for i in range(rm.n_spare))
    rm.close()
    # an explicit smaller n_spare uses the first free slots; a larger one than the pool + budget holds is capped
    rm = ResidencyManager(pool, model, ResidencyConfig(n_spare=3, n_demote=2, hbm_free_bytes=0), n_stage=8)
    assert rm.n_spare == 3 and rm.spare_added_B == 0
    rm.close()
    rm = ResidencyManager(pool, model, ResidencyConfig(n_spare=12, n_demote=2, hbm_free_bytes=0), n_stage=8)
    assert rm.n_spare == 8 and pool.spare == spare0       # wants 4 new, none affordable: capped, warned, not grown
    rm.close()
    # more free HBM: the shortfall is allocated (one growth), never more than asked
    rm = ResidencyManager(pool, model, ResidencyConfig(n_spare=12, n_demote=2, reserve_gib=0.0,
                                                       hbm_free_bytes=1 << 40), n_stage=8)
    assert rm.n_spare == 12 and rm.budget["new_per_layer"] == 4 and pool.spare == 20 and rm.spare_added_B > 0
    rm.close()
    # default (n_spare None): all free slots; with none free in place -> spare_cap new ones, only if affordable
    model, pool, _ = emu("mini4")
    with pytest.raises(RuntimeError, match="no residency slot"):
        ResidencyManager(pool, model, ResidencyConfig(n_demote=2, hbm_free_bytes=0), n_stage=6)
    rm = ResidencyManager(pool, model, ResidencyConfig(n_demote=2, reserve_gib=0.0, hbm_free_bytes=1 << 40), n_stage=6)
    assert rm.n_spare == ResidencyConfig().spare_cap and pool.spare == 6 + rm.n_spare
    rm.close()


def test_host_tensors_carry_device_cpu_explicitly(emu):
    """Box r1: 'read target must be host memory, got cuda:0' -- a host buffer made under the runtime's cuda default
    device. Every torch factory call in residency.py names its device, and the golden-digest read does not depend on
    the default device (a meta default device stands in for cuda here)."""
    import ast
    import inspect
    from runtime import residency as R

    fac = {"zeros", "ones", "empty", "full", "tensor", "arange", "as_tensor", "zeros_like", "empty_like", "from_numpy",
           "linspace", "randn", "rand", "randint", "randperm"}
    bad = []
    for n in ast.walk(ast.parse(inspect.getsource(R))):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "torch" and n.func.attr in fac and n.func.attr != "from_numpy":
            if not any(k.arg == "device" for k in n.keywords):
                bad.append(n.lineno)
    assert not bad, f"torch factory calls without device= in residency.py lines {bad}"
    model, pool, _ = emu("mini4")
    assert pool.loc_cpu.device.type == "cpu"
    g = golden_from_index(model.runtime_load.index)
    want = g(0, pool.hot[0][0])
    try:
        torch.set_default_device("meta")
        assert g(0, pool.hot[0][0]) == want
    finally:
        torch.set_default_device("cpu")


# ================================================================================================ mechanism (emulated)
class _Reg(dict):
    """Address -> uint8 bytes for the emulated gather / staging: the pool's slots and cold pool, plus everything
    the residency manager can publish (its spare scale slots, demotion pool). Refilled on a miss (storage moves)."""

    def __init__(self, pool):
        super().__init__()
        self.pool = pool
        self.refill()

    def refill(self):
        self.clear()
        pool = self.pool
        for l in range(pool.n_layers):
            for p in ("w1", "w2", "w3"):
                for t in list(pool.hbm_full[l][p]) + list(pool.hbm[l][p][1]):
                    self[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
                for e in pool.cold[l]:
                    for kind in ("weight", "scale"):
                        t = pool.cold_view(l, e, p, kind, cuda=False)
                        self[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
        rm = getattr(pool, "residency", None)
        if rm is not None:
            self.update(rm.address_views())

    def __missing__(self, key):
        self.refill()
        if key not in self.keys():
            raise KeyError(key)
        return dict.get(self, key)


def _install(monkeypatch, pool):
    from runtime import dg_inplace as DI

    reg = _Reg(pool)
    fk = _fake_static_kernels(reg)
    fk.calls["stage"] = 0
    fk.reg = reg

    def stage_inplace(ptab6, ids, gtab, dtab, dsts, mi, H, align, row_w, sf_dims, all_scales, ids_out=None, **kw):
        DI.stage_inplace_emulated(reg.__getitem__, ptab6, ids, gtab, dtab, dsts, mi, H, align, row_w, sf_dims,
                                  all_scales, ids_out, **kw)
        fk.calls["stage"] += 1

    fk.stage_inplace = stage_inplace

    def stage_rows(ptab6, uniq, gtab, dtab, dsts, sidx, wl, H, row_w, sf_dims, n_work, **kw):
        """M1-T10-MOE: the R-row staging launch (cold bytes + scales from the dedup's work list) = its contract."""
        DI.stage_rows_emulated(reg.__getitem__, ptab6, uniq, gtab, dtab, dsts, sidx, wl, H, row_w, sf_dims, n_work)
        fk.calls["stage"] += 1

    fk.stage_rows = stage_rows
    monkeypatch.setitem(sys.modules, "runtime.static_kernels", fk)
    return fk


@pytest.fixture
def remu(emu, monkeypatch):
    """-> build(kind, impl) -> (model, pool, fk): the emulated mini checkpoints with the residency-aware registry;
    impl "dg-inplace" also installs the contract-checking fake deep_gemm and the faithful act_quant."""
    def build(kind, impl="ref"):
        model, pool, _ = emu(kind)
        if impl == "dg-inplace":
            from test_dg_inplace_cpu import _fake_dg_contig
            from test_dg_moe_cpu import _faithful_act_quant

            monkeypatch.setitem(sys.modules, "deep_gemm", _fake_dg_contig())
            monkeypatch.setattr(sys.modules["model"], "act_quant", _faithful_act_quant)
        fk = _install(monkeypatch, pool)
        return model, pool, fk
    return build


def _step_ids(L, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 2000, (L,), generator=g).tolist()


def _run(model, st, ids, n, rm=None, before=None, after=None):
    """prefill + n static-eager steps; rm.after_step() after each; per-position fp32 logits + tokens + the routed
    ids each step saw."""
    with torch.inference_mode():
        out_ids, logits, _ = model(torch.tensor([ids]), 0)
        res, toks, routed = [logits[0].clone()], [int(out_ids[0])], []
        for k in range(n):
            if before is not None:
                before(k)
            lg = st.step(toks[-1], len(ids) + k, graph=False)
            res.append(lg[0].clone())
            toks.append(int(st.tok.reshape(-1)[0]))
            routed.append(st.ids_step.clone())
            if after is not None:
                after(k)
            if rm is not None:
                rm.after_step()
    return res, toks, routed


def _equal(a, b):
    bad = [i for i, (x, y) in enumerate(zip(a[0], b[0])) if not torch.equal(x, y)]
    return not bad and a[1] == b[1], bad


def _cfg(**kw):
    base = dict(drain_n=4, min_abs=1.0, hyst=1.0, halflife=50.0, n_spare=2, n_demote=4, reserve_gib=0.0)
    base.update(kw)
    return ResidencyConfig(**base)


def _hot_cold_bias(pool, rng, scale=40):
    """Synthetic per-step counter delta that makes cold experts look hot (drives real policy promotions)."""
    d = np.zeros((pool.n_layers, pool.n_experts), dtype=np.int64)
    for l in range(pool.n_layers):
        d[l, list(pool.cold[l])] = rng.integers(0, scale, len(pool.cold[l]))
    return d


@pytest.mark.parametrize("impl,sf_store", [("ref", None), ("dg-inplace", None), ("dg-inplace", "resident")])
def test_residency_on_equals_off_with_forced_and_policy_swaps(remu, monkeypatch, impl, sf_store):
    """T-R1 + T-R2 + T-R3 (emulated): 40 decode steps with residency off, then on with one forced random swap per
    step AND policy swaps from synthetic counters -> every position's fp32 logits torch.equal, tokens equal; a
    promoted expert is read by a step; the audit (tables, params, sha256 of moved experts vs the shards) passes."""
    from runtime import dg_inplace as DI
    from runtime.static_step import StaticStep

    if sf_store:
        monkeypatch.setenv(DI.SF_STORE_ENV, sf_store)
    model, pool, fk = remu("mini4", impl)
    ids, n = _step_ids(16), 40
    st = StaticStep(model, pool, max_pos=127, moe_impl=impl, _allow_cpu=True)
    off = _run(model, st, ids, n)
    st.detach()
    k = model.layers[0].ffn.n_activated_experts
    rm = ResidencyManager(pool, model, _cfg(force_random=1), n_stage=n_stage_for(impl, 1, k))
    st = StaticStep(model, pool, max_pos=127, moe_impl=impl, _allow_cpu=True)
    info = rm.attach(st)
    assert set(info["kinds"]) == ({"ptab"} if impl == "ref" else {"ptab", "ptab6", "gtab"})
    rng = np.random.default_rng(5)
    read_promoted = []

    def after(k_):
        for l in range(rm.L):  # the step just run read these ids through the tables as published before it
            if any(rm.slot_of[l].get(int(e), -1) >= rm.base[l] for e in st.ids_step[l].tolist()):
                read_promoted.append((k_, l))
        rm.inject_counts(_hot_cold_bias(pool, rng))

    on = _run(model, st, ids, n, rm=rm, after=after)
    eq, bad = _equal(on, off)
    assert eq, (impl, sf_store, bad[:5])
    s = rm.stats
    assert s["published"] >= n - 2 and s["d2h_demotions"] > 0 and s["drains"] >= n // 4 - 1 and s["vetoed"] == 0
    assert rm.state.n_swaps > s["forced_swaps"] - rm.state.stale  # policy swaps too, not just the forced ones
    assert read_promoted, "no step read an expert from a residency slot"
    rm.flush()
    a = rm.audit(golden=golden_from_index(model.runtime_load.index))
    assert a["ok"], a["errors"][:6]
    assert a["experts_checked"] > 0
    assert rm.swaps_per_step()["max"] <= rm.params.swaps_per_barrier
    # restore: boot placement, params, tables; a fresh non-residency step == the reference forward
    rm.close()
    assert pool.residency is None and not rm.moved()
    for l in range(pool.n_layers):
        for s_, e in enumerate(pool.hot[l]):
            assert model.layers[l].ffn.experts[e].w1.weight.data_ptr() == pool.hbm_full[l]["w1"][s_].data_ptr()
    st.detach()
    st2 = StaticStep(model, pool, max_pos=127, moe_impl=impl, _allow_cpu=True)
    eq, bad = _equal(_run(model, st2, ids, 12), (off[0][:13], off[1][:13]))
    assert eq, ("after restore", bad[:5])
    st2.detach()
    if impl == "ref":
        ref = _ref_run(model, ids, 12)
        assert all(torch.equal(x, y) for x, y in zip(ref[0], off[0][:13]))


def test_manager_barrier_is_apply_pending(remu):
    """The manager's barrier (veto-capable) == policy.apply_pending: a shadow State fed the same counter reads at the
    same host steps stays row-for-row equal, swap for swap (no vetoes, no forced swaps)."""
    from runtime.static_step import StaticStep

    model, pool, _ = remu("mini4")
    rm = ResidencyManager(pool, model, _cfg(n_spare=2), n_stage=0)
    st = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    rm.attach(st)
    shadow = P.State.from_rows({l: {"hot": list(pool.hot[l]), "cold": list(pool.cold[l])} for l in range(rm.L)},
                               rm.E, rm.L, rm.params)
    shadow.freeze = False
    P.baseline(shadow, np.zeros((rm.L, rm.E), dtype=np.int64), 0)
    reads = []
    orig = rm._drain_now

    def spy(hs):
        reads.append((hs, rm.cnt_host.numpy().astype(np.int64).copy()))
        orig(hs)

    rm._drain_now = spy
    rng = np.random.default_rng(1)
    checks = 0

    def after(k_):
        nonlocal checks
        rm.inject_counts(_hot_cold_bias(pool, rng, 30))

    ids = _step_ids(17, 3)
    with torch.inference_mode():
        out_ids, _, _ = model(torch.tensor([ids]), 0)
        tok = int(out_ids[0])
        for k in range(48):
            st.step(tok, len(ids) + k, graph=False)
            tok = int(st.tok.reshape(-1)[0])
            after(k)
            n_reads = len(reads)
            rm.after_step()
            for hs, cur in reads[n_reads:]:
                P.drain(shadow, rm.params, cur, hs)
            P.apply_pending(shadow, rm.params)
            assert shadow.hot == rm.state.hot and shadow.cold == rm.state.cold and shadow.n_swaps == rm.state.n_swaps
            assert np.array_equal(shadow.ewma, rm.state.ewma)
            checks += 1
    assert rm.state.n_swaps > 0 and checks == 48 and rm.stats["vetoed"] == 0
    rm.close()
    st.detach()


def _routed_cold_target(pool, routed_next):
    """(layer, a boot-cold expert routed by the next step, a boot-hot expert NOT routed by it)."""
    for l in range(pool.n_layers):
        ids = set(routed_next[l].tolist())
        c = [e for e in pool.cold[l] if e in ids]
        h = [e for e in pool.hot[l] if e not in ids]
        if c and h:
            return l, c[0], h[0]
    return None


@pytest.mark.parametrize("mutation", ["publish_before_copy", "corrupt_promoted", "skip:ptab", "skip:ptab6",
                                      "skip:gtab"])
def test_mutations_are_caught(remu, mutation):
    """T-R4: each broken actuator is caught by T-R1 (logits) or T-R2/T-R3 (audit)."""
    from runtime.static_step import StaticStep

    impl = "ref" if mutation in ("skip:ptab",) else "dg-inplace"
    model, pool, _ = remu("mini4", impl)
    ids, n = _step_ids(16, 11), 10
    st = StaticStep(model, pool, max_pos=127, moe_impl=impl, _allow_cpu=True)
    off = _run(model, st, ids, n)
    st.detach()
    k = model.layers[0].ffn.n_activated_experts
    rm = ResidencyManager(pool, model, _cfg(frozen=True), n_stage=n_stage_for(impl, 1, k))
    st = StaticStep(model, pool, max_pos=127, moe_impl=impl, _allow_cpu=True)
    rm.attach(st)
    rm._mutation = mutation
    at = 3
    tgt = _routed_cold_target(pool, off[2][at + 1])
    assert tgt is not None
    swaps = [tgt]
    for l in range(pool.n_layers):  # plus one swap per other layer: every table kind gets written
        if l != tgt[0]:
            swaps.append((l, pool.cold[l][-1], pool.hot[l][-1]))

    def after(k_):
        if k_ == at:
            rm.force_swaps(swaps)

    on = _run(model, st, ids, n, rm=rm, after=after)
    rm.flush()
    eq, _ = _equal(on, off)
    a = rm.audit(golden=golden_from_index(model.runtime_load.index))
    if mutation == "publish_before_copy":
        assert not eq, "T-R1 missed a publish before the copy"
    elif mutation == "corrupt_promoted":
        assert not eq and not a["ok"] and any("differ from the checkpoint" in x for x in a["errors"])
    else:
        assert not a["ok"] and any("table" in x for x in a["errors"]), a
    rm._mutation = None
    rm.close()
    st.detach()


def test_attach_refuses_unknown_pointer_table_copies(remu):
    from runtime.static_step import StaticStep

    model, pool, _ = remu("mini")
    rm = ResidencyManager(pool, model, _cfg(), n_stage=0)
    st = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    st.moes[1].rogue_copy = st.moes[1].ptabs[("w2", "scale")].clone()  # a table some future kernel might read
    with pytest.raises(RuntimeError, match="does not own"):
        rm.attach(st)
    del st.moes[1].rogue_copy
    rm.attach(st)
    st2 = StaticStep(model, pool, max_pos=127, counters=False, _allow_cpu=True)
    with pytest.raises(ValueError, match="counters"):
        rm.attach(st2)
    with pytest.raises(RuntimeError, match="already has"):
        ResidencyManager(pool, model, _cfg(), n_stage=0)
    rm.force_swaps([(0, pool.cold[0][0], pool.hot[0][0])])
    with torch.inference_mode():
        model(torch.tensor([_step_ids(5)]), 0)
        st.step(3, 5, graph=False)
        rm.after_step()
    assert rm.moved()
    with pytest.raises(RuntimeError, match="restore_boot"):
        pool.add_spare_slots(pool.spare + 2, model)  # a storage move would drop the promoted experts' bytes
    rm.close()
    st.detach()
    st2.detach()


def test_generator_residency_plumbing_byte_identical(remu):
    """Generator(residency=...) in static mode: greedy records byte-identical to residency off (and to eager), swaps
    published during the call, reset_static restores the boot placement and drops the manager."""
    from runtime.generate import Generator, dump_record
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    model, pool, _ = remu("mini4")
    fac = lambda m, **kw: StaticStep(m, pool, max_pos=127, _allow_cpu=True, **kw)  # noqa: E731
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", static_factory=fac)
    ids = list(range(40, 57))
    eager = gen.greedy(ids, 24, stop_on_eos=False)
    gen.set_mode("static")
    off = gen.greedy(ids, 24, stop_on_eos=False)
    m_off = dict(gen.last_meta)
    gen.set_residency({"drain_n": 4, "min_abs": 1.0, "hyst": 1.0, "n_spare": 2, "n_demote": 4, "force_random": 2,
                       "reserve_gib": 0.0})
    assert gen.static is None and gen.residency is not None
    gen.static_step()
    rm = gen.residency_mgr
    assert rm is not None and pool.residency is rm
    on = gen.greedy(ids, 24, stop_on_eos=False)
    m_on = dict(gen.last_meta)
    on2 = gen.greedy(ids, 24, stop_on_eos=False)  # a second call under a different residency
    assert dump_record(on) == dump_record(off) == dump_record(eager) == dump_record(on2)
    assert m_on["step_sha256"] == m_off["step_sha256"]
    assert m_on["residency"]["published"] > 0 and rm.moved()
    gen.set_residency(None)
    assert gen.residency_mgr is None and pool.residency is None and not rm.moved()
    assert dump_record(gen.greedy(ids, 24, stop_on_eos=False)) == dump_record(off)
    gen.reset_static()


def _first_light():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    return fl


def test_first_light_residency_on_vs_off(remu, tmp_path, capsys):
    """bench/first_light.py --residency on (static mode, emulated mini4): reps + step cell under forced + policy
    swaps; `--residency-ref` = the off run's report -> rep 1 byte-identical, step-cell tokens equal; the report's
    residency section (curve, swaps/step, HBM, audit) and the 12-line summary with the residency line."""
    from runtime.generate import Generator
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    fl = _first_light()
    model, pool, _ = remu("mini4")
    with open(tmp_path / "fx.jsonl", "w") as f:
        for tag, text in (("prose", "w5 w9 w12 w3 w7"), ("json", "w40 w41 w42"), ("tool", "w90 w91 w2 w3")):
            f.write(json.dumps({"tag": tag, "prompt": text}) + "\n")
    common = ["--model-dir", "x", "--mode", "static", "--fixture", str(tmp_path / "fx.jsonl"), "--reps", "2",
              "--max-new", "20", "--step-cell", "24", "--stats-from", "1", "--max-pos", "127"]

    def run(tag, extra):
        a = fl.build_parser().parse_args(common + ["--out", str(tmp_path / f"{tag}.jsonl"),
                                                   "--json", str(tmp_path / f"{tag}.json")] + extra)
        gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                        max_pos=127, residency=fl.residency_config(a),
                        static_factory=lambda m, **kw: StaticStep(m, pool, max_pos=127, _allow_cpu=True, **kw))
        gen.encode = lambda msgs: [7, 8] + [int(w[1:]) for w in msgs[0]["content"].split()]
        report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "static"}}
        rc = fl.measure(gen, a, report, n_layers=len(model.layers), out=a.out, json_path=a.json, fixture2=None)
        gen.reset_static()
        return rc, report

    rc0, r0 = run("off", [])
    assert rc0 == 0, r0["errors"] and "residency" not in r0
    rc1, r1 = run("on", ["--residency", "on", "--residency-drain", "4", "--residency-min-abs", "1",
                         "--residency-hyst", "1.0", "--residency-spare", "2", "--residency-demote", "4",
                         "--residency-reserve-gib", "0", "--residency-force-random", "1",
                         "--residency-ref", str(tmp_path / "off.json"), "--residency-audit"])
    assert rc1 == 0, r1["errors"]
    rr = r1["residency"]
    assert rr["on"] and rr["vs_off"]["rep1_files_byte_identical"] and rr["vs_off"]["step_cell_tokens_equal"]
    assert rr["audit"]["ok"] and rr["audit"]["experts_checked"] > 0
    assert rr["swaps_per_step"]["swaps"] > 0 and rr["swaps_per_step"]["max"] <= 4
    assert len(rr["cold_share_curve"]) >= 10 and rr["n_spare"] == 2 and rr["hbm"]["spare_weight_B"] > 0
    assert r1["self_replay"][2]["identical"] == 3  # rep 2 ran under a different residency than rep 1
    assert rr["step_cell_cold_share_first64"] is not None and set(rr["rep_cold_share"]) == {"1", "2"}
    lines = capsys.readouterr().out.strip().splitlines()
    assert any(l.startswith("residency: ") for l in lines[-12:]) and lines[-1].startswith("errors: 0")
    print("\n".join(lines[-12:]))


def test_gpu_test_combo_logic_dry_run_on_cpu(remu, monkeypatch, capsys):
    """tests/gpu/test_residency_gpu.py's main gate (`_combo`: off, on with policy + forced next-step-routed swaps,
    audit, off again after restore) executed on the CPU emulation in static mode, so the box run meets only CUDA
    (graphs, streams, events), not harness bugs."""
    import contextlib
    import importlib.util

    from runtime.generate import Generator
    from runtime.static_step import StaticStep

    spec = importlib.util.spec_from_file_location("gpu_residency", os.path.join(REPO, "tests", "gpu",
                                                                                "test_residency_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    model, pool, _ = remu("mini4")

    @contextlib.contextmanager
    def model_cpu(kind="mini4"):
        yield model, None

    def gen_cpu(model_, tokdir, moe_impl="ref", glue_impl="ref", residency=None):
        return Generator(model_, eos_ids=[], device="cpu", mode="static", moe_impl=moe_impl, glue_impl=glue_impl,
                         residency=residency,
                         static_factory=lambda m, **kw: StaticStep(m, pool, max_pos=127, _allow_cpu=True, **kw))

    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    monkeypatch.setattr(G, "_gen", gen_cpu)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    G._combo("ref", "ref")
    out = capsys.readouterr().out
    assert "ref/ref: on == off over prefill + 64 graph steps" in out
    print(out)
