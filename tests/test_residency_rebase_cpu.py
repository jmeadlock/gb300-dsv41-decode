"""Mac CPU tests of the A2-2 REBASE onto main's 6-row path (docs/HANDOFF-A2-2-REBASE.md): `.venv/bin/python -m pytest -q tests/`.

main moved under the branch: the 6-row dg-inplace step stages cold experts by a COMPACT COLD INDEX into S = 16 spare
slots per layer (not R*topk = 36), keeps the hot experts' packed scale factors in a POOL-OWNED copy ("pool" store,
written once at build), counts overflow on the device, and the DSpark loop replays the step with no host decision
(runtime/dspark_loop.py). What these tests pin:

  * n_stage_for / sf_store_for / the headroom calc follow the new step (S, not R*topk; the packed-SF copy and its build
    headroom are charged to the budget; one spare slot adds one SF group);
  * the residency slots start AFTER the staging slots (never inside H..H+S-1, which the dedup kernel addresses);
  * with the "pool" store a promoted expert's packed SF is the manager's to write (negative control: without it the
    promoted expert's outputs are wrong), with "step" the staging launch re-packs it each step;
  * 6 rows, residency on == off, torch.equal on every row's fp32 logits, with forced + policy swaps (both stores, S = 8
    and the default 16), overflow counter 0;
  * Generator plumbing at 6 rows (spare_slots -> n_stage, kv_rows follows the rows);
  * DSparkLoop calls the manager once per step with the device `live` flag; steps after `done` are not counted; the
    emitted tokens / accept lengths are identical with residency on (eager loop on the emulations, forced swaps);
  * bench/headline.py: the residency flags, the report (curve, swaps/step, manager ms, vs-off), the window stats.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

from runtime.residency import (ResidencyConfig, ResidencyManager, golden_from_index, n_stage_for, sf_store_for,
                               spare_budget)
from test_dspark_loop_cpu import emu as dspark_emu  # noqa: F401
from test_residency_cpu import _hot_cold_bias, _install, remu  # noqa: F401
from test_static_rows_cpu import _assert_rows_equal, _prefill, _tf_tokens, emu  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ================================================================================================ plan arithmetic
def test_sf_store_for_follows_the_steps_defaults_and_env(monkeypatch):
    from runtime import dg_inplace as DI

    monkeypatch.delenv(DI.ROWS_SF_STORE_ENV, raising=False)
    monkeypatch.delenv(DI.SF_STORE_ENV, raising=False)
    assert sf_store_for("ref", 6) is None and sf_store_for("dg-masked", 1) is None
    assert sf_store_for("dg-inplace", 6) == "pool" == DI.ROWS_SF_STORE_DEFAULT      # rows default (M1-T10-MOE)
    assert sf_store_for("dg-inplace", 1) == "step"                                  # 1-row default
    monkeypatch.setenv(DI.ROWS_SF_STORE_ENV, "step")
    assert sf_store_for("dg-inplace", 6) == "step" and sf_store_for("dg-inplace", 1) == "step"
    monkeypatch.setenv(DI.SF_STORE_ENV, "resident")
    assert sf_store_for("dg-inplace", 1) == "resident"


def test_budget_charges_the_pool_sf_copy_and_its_headroom(emu, monkeypatch):   # noqa: F811
    from runtime import dg_inplace as DI

    model, pool, _ = emu("mini4")
    L, H = pool.n_layers, [len(pool.hot[l]) for l in range(pool.n_layers)]
    s_row = sum(n * (k // 32) for n, k in pool.proj_nk.values())
    big = 1 << 40
    cfg = ResidencyConfig(n_spare=2, n_demote=2, reserve_gib=0.0, hbm_free_bytes=big)
    monkeypatch.delenv(DI.SF_POOL_HEADROOM_ENV, raising=False)
    off = ResidencyManager(pool, model, cfg, n_stage=0, sf_store="step")
    assert off.budget["step_extra_GiB"] == 0
    off.close()
    on = ResidencyManager(pool, model, cfg, n_stage=16, sf_store="pool")
    # sum(H) + L x max(n_stage, pool.spare) groups of packed SF + the 6 GiB headroom the pool store's build insists on
    want = (sum(H) + L * 16) * s_row + int(DI.SF_POOL_HEADROOM_GIB * 2 ** 30)
    assert on.budget["step_extra_GiB"] == round(want / 2 ** 30, 3) and on.budget["step_extra_GiB"] >= DI.SF_POOL_HEADROOM_GIB
    assert on.budget["per_spare_slot_all_layers_GiB"] >= round(L * (pool.expert_bytes + s_row) / 2 ** 30, 4) - 1e-4
    assert on.sf_store_req == "pool" and on.n_stage == 16
    on.close()
    # the env knob moves the headroom
    monkeypatch.setenv(DI.SF_POOL_HEADROOM_ENV, "1.0")
    sp = max(16, pool.spare)   # the earlier manager grew the pool to 16 + n_spare: the groups count follows pool.spare
    m2 = ResidencyManager(pool, model, cfg, n_stage=16, sf_store="resident")
    assert m2.budget["step_extra_GiB"] == round(((sum(H) + L * sp) * s_row + 2 ** 30) / 2 ** 30, 3)
    m2.close()
    # a caller that pre-grew the pool (bench/headline.py: staging + residency slots in ONE growth) has those weight
    # slots already allocated: prepaid bytes are not new cost
    eb = pool.expert_bytes
    base_kw = dict(n_stage_new=0, reserve_gib=0.0, kv_rows=1, kv_ctx=1, step_extra_bytes=0, sf_group_bytes=0)
    free = 10 * L * eb
    a0 = spare_budget(free, L, eb, 3 * 1024, **base_kw)
    a1 = spare_budget(free, L, eb, 3 * 1024, prepaid_bytes=3 * L * eb, **base_kw)
    assert a1["affordable_per_layer"] == a0["affordable_per_layer"] + 3 and a1["prepaid_GiB"] > 0 == a0["prepaid_GiB"]
    pre = ResidencyManager(pool, model, ResidencyConfig(n_demote=2, reserve_gib=0.0, hbm_free_bytes=big),
                           n_stage=8, sf_store="step")   # pool.spare is 18 here (the managers above grew it)
    assert pre.budget["prepaid_GiB"] > 0 and pre.n_stage == 8
    pre.close()
    # too little HBM for a single spare slot -> a clear refusal, not a silent fallback
    tight = ResidencyConfig(n_demote=2, reserve_gib=0.0, hbm_free_bytes=(sum(H) + L * pool.spare) * s_row)
    with pytest.raises(RuntimeError, match="no spare HBM slot affordable"):
        ResidencyManager(pool, model, tight, n_stage=16, sf_store="pool")


# ================================================================================================ 6 rows, residency on == off
def _run6(model, st, ids, toks, n, rm=None, after=None):
    _prefill(model, ids)
    out = []
    with torch.inference_mode():
        for s in range(n):
            lg = st.step(toks[s * st.rows:(s + 1) * st.rows], len(ids) + s * st.rows, graph=False)
            out += [lg[j].clone() for j in range(st.rows)]
            if after is not None:
                after(s)
            if rm is not None:
                rm.after_step()
    return out


def _dg_fakes(monkeypatch):
    from test_dg_inplace_cpu import _fake_dg_contig
    from test_dg_moe_cpu import _faithful_act_quant
    from test_static_rows_cpu import _masked_rowwise

    fake = _fake_dg_contig()
    fake.m_grouped_fp8_fp4_gemm_nt_masked = _masked_rowwise(fake)
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    monkeypatch.setattr(sys.modules["model"], "act_quant", _faithful_act_quant)


@pytest.mark.parametrize("store,S", [("pool", None), ("pool", 8), ("step", None)])
def test_rows6_dg_inplace_on_equals_off_by_sf_store(emu, monkeypatch, store, S):   # noqa: F811
    """6-row dg-inplace, residency slots behind S = 16 (default) / 8 staging slots: forced random swaps every step +
    policy swaps -> logits equal off; no promoted expert sits in a staging slot; overflow 0; the manager owns the
    packed SF iff the store is not re-packed per step; the audit passes."""
    from runtime import dg_inplace as DI
    from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS, StaticStep

    model, pool, _ = emu("mini4")
    _dg_fakes(monkeypatch)
    _install(monkeypatch, pool)
    monkeypatch.setenv(DI.ROWS_SF_STORE_ENV, store)
    n = 6
    ids, toks = _tf_tokens(17, 6 * n, 33)
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", spare_slots=S, _allow_cpu=True)
    off = _run6(model, st, ids, toks, n)
    st.detach()
    k = model.layers[0].ffn.n_activated_experts
    ns = min(S or DEFAULT_SPARE_SLOTS_ROWS, 6 * k)
    assert n_stage_for("dg-inplace", 6, k, S) == ns and sf_store_for("dg-inplace", 6) == store
    cfg = ResidencyConfig(drain_n=2, min_abs=1.0, hyst=1.0, halflife=20.0, n_spare=2, n_demote=4, reserve_gib=0.0,
                          force_random=2)
    rm = ResidencyManager(pool, model, cfg, n_stage=ns, sf_store=store)
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", spare_slots=S, _allow_cpu=True)
    assert st.spare_slots == ns and pool.spare == ns + rm.n_spare
    rm.attach(st)
    assert bool(rm.resident_sf) == (store == "pool") and st.dg.sf_store == store
    rng = np.random.default_rng(3)
    on = _run6(model, st, ids, toks, n, rm=rm, after=lambda s: rm.inject_counts(_hot_cold_bias(pool, rng, 30)))
    _assert_rows_equal(on, off, f"6-row dg-inplace {store} S={ns}: residency on vs off")
    assert rm.stats["published"] > 0 and rm.stats["vetoed"] == 0
    moved_to_spare = 0
    for l in range(rm.L):   # an expert's slot is a boot slot (< H) or a residency slot -- never a staging slot H..H+ns-1
        for e, s in rm.slot_of[l].items():
            assert s < rm.H[l] or rm.H[l] + ns <= s < rm.H[l] + ns + rm.n_spare, (l, e, s, ns)
            moved_to_spare += s >= rm.H[l]
    assert moved_to_spare > 0 or rm.stats["published"] > 0
    ov = st.overflow_counts()
    assert ov["steps"] == 0 and ov["experts"] == 0 and ov["spare_slots"] == ns, ov
    rm.flush()
    a = rm.audit(golden=golden_from_index(model.runtime_load.index))
    assert a["ok"], a["errors"][:5]
    rm.close()
    st.detach()


def test_rows6_pool_store_promoted_sf_negative_control(emu, monkeypatch):   # noqa: F811
    """The pool store packs the hot experts' scales ONCE; a promoted expert's packed SF is the manager's job. Without
    it (resident_sf dropped) the promoted experts read zeroed scales and the logits must move -- the control that
    makes the on == off test above able to see this bug."""
    from runtime import dg_inplace as DI
    from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS, StaticStep

    model, pool, _ = emu("mini4")
    _dg_fakes(monkeypatch)
    _install(monkeypatch, pool)
    monkeypatch.setenv(DI.ROWS_SF_STORE_ENV, "pool")
    n = 6
    ids, toks = _tf_tokens(17, 6 * n, 35)
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
    off = _run6(model, st, ids, toks, n)
    st.detach()
    k = model.layers[0].ffn.n_activated_experts
    cfg = ResidencyConfig(drain_n=2, min_abs=1.0, hyst=1.0, halflife=20.0, n_spare=2, n_demote=4, reserve_gib=0.0,
                          force_random=3)
    rm = ResidencyManager(pool, model, cfg, n_stage=min(DEFAULT_SPARE_SLOTS_ROWS, 6 * k), sf_store="pool")
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
    rm.attach(st)
    assert rm.resident_sf
    rm.resident_sf = {}   # the bug: promotions copy the weights but forget the packed SF
    on = _run6(model, st, ids, toks, n, rm=rm)
    assert rm.stats["published"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(on, off)), "negative control: a missing packed SF went unseen"
    rm.close()
    st.detach()


# ================================================================================================ Generator plumbing, 6 rows
def test_generator_rows6_residency_plumbing(remu, monkeypatch):   # noqa: F811
    """Generator(rows=6, moe_impl=dg-inplace, spare_slots=S, residency=...): the manager is sized from the step's S
    (n_stage == S, base = H + S), kv_rows follows the rows, teacher-forced records are byte-identical on vs off, and a
    spare_slots change drops the step and the manager."""
    from runtime.generate import Generator, dump_record
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    model, pool, _ = remu("mini4", "dg-inplace")
    fac = lambda m, **kw: StaticStep(m, pool, max_pos=127, _allow_cpu=True, **kw)   # noqa: E731
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                    static_factory=fac, rows=6, moe_impl="dg-inplace", spare_slots=8)
    ids = list(range(40, 57))
    targets = list(range(70, 70 + 25))
    off = gen.teacher_forced(ids, targets)
    cfg = {"drain_n": 2, "min_abs": 1.0, "hyst": 1.0, "n_spare": 2, "n_demote": 4, "force_random": 2, "reserve_gib": 0.0}
    gen.set_residency(cfg)
    assert gen.static is None
    gen.static_step()
    rm = gen.residency_mgr
    assert rm.n_stage == gen.static.spare_slots == 8 and rm.cfg.kv_rows == 6
    assert rm.base == [len(pool.hot[l]) + 8 for l in range(pool.n_layers)] and pool.spare == 8 + 2
    on = gen.teacher_forced(ids, targets)
    assert dump_record(on) == dump_record(off) and rm.stats["published"] > 0
    gen.set_spare_slots(12)
    assert gen.static is None and gen.residency_mgr is None and pool.residency is None
    gen.static_step()
    assert gen.residency_mgr.n_stage == 12 == gen.static.spare_slots
    gen.reset_static()


# ================================================================================================ the DSpark loop
def test_dspark_loop_calls_the_manager_each_step_with_the_live_flag(monkeypatch):
    """step_once -> gen.residency_mgr.after_step(live=b.live) exactly once per step (graph-free path); no manager ->
    no call. The hook runs after the step's end event, so step/verify/draft ms exclude it (timed: events around it)."""
    from runtime.dspark_loop import DSparkLoop

    calls = []

    class FakeMgr:
        def after_step(self, live=None):
            calls.append(live)

    class FakeGen:
        residency_mgr = FakeMgr()

    loop = DSparkLoop.__new__(DSparkLoop)
    loop.gen, loop.graphs, loop.n_graphs, loop.n_replays = FakeGen(), {}, 0, 0
    loop.b = type("B", (), {"live": torch.ones(1, dtype=torch.long)})()
    loop._hook_evs = []
    loop._step_eager = lambda: None
    loop.step_once()
    loop.step_once()
    assert len(calls) == 2 and all(c is loop.b.live for c in calls) and loop.n_replays == 2
    FakeGen.residency_mgr = None
    loop.step_once()
    assert len(calls) == 2


def test_hit_counter_ignores_not_live_steps(emu, monkeypatch):   # noqa: F811
    """after_step(live): a step the loop issued after `done` (live 0) leaves the counters alone; live 1 counts."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")
    _install(monkeypatch, pool)
    rm = ResidencyManager(pool, model, ResidencyConfig(frozen=True, n_spare=1, n_demote=1, reserve_gib=0.0), n_stage=0)
    st = StaticStep(model, pool, max_pos=255, rows=6, _allow_cpu=True)
    rm.attach(st)
    ids, toks = _tf_tokens(17, 6, 37)
    _prefill(model, ids)
    with torch.inference_mode():
        st.step(toks[:6], len(ids), graph=False)
        rm.after_step(live=torch.zeros(1, dtype=torch.long))
        assert int(rm.cnt.sum()) == 0 and rm.host_steps == 1
        rm.after_step(live=torch.ones(1, dtype=torch.long))
        assert int(rm.cnt.sum()) == int(st.uvalid_step.sum()) > 0
        rm.after_step()
        assert int(rm.cnt.sum()) == 2 * int(st.uvalid_step.sum())
    rm.close()
    st.detach()


def test_loop_with_residency_on_emits_the_same_tokens(emu_loop):
    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator
    from tiny_tok import make_wordlevel_tokenizer

    G, model, pool = emu_loop
    B = _headline()
    with G.row_invariant_mode():
        gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                        max_pos=255)
        gen.static_kw = {"_allow_cpu": True}
        g = torch.Generator().manual_seed(11)
        prompts = {}
        for tag, L in (("a", 17), ("b", 18)):
            ids = torch.randint(2, 2000, (L,), generator=g).tolist()
            rec = gen.greedy(ids, 20, tag=tag)
            m = gen.last_meta
            prompts[tag] = {"ids": ids, "r1_ids": list(m["token_ids"]), "r1_sha": list(m["step_sha256"]),
                            "r1_rec": rec}
        reports = {}
        for name, cfg in (("off", None),
                          ("on", {"drain_n": 2, "min_abs": 1.0, "hyst": 1.0, "halflife": 20.0, "n_spare": 2,
                                  "n_demote": 4, "force_random": 2, "reserve_gib": 0.0})):
            gen.set_residency(cfg)
            loop = DSparkLoop(gen, 20, _allow_cpu=True)
            rep = {"errors": [], "date": "dry", "mode": "static", "impls": {}, "max_new": 20}
            results = B.run_cells(loop, prompts, 20, [0], reps=2, parity_reps=1, stop_on_eos=True, rep=rep)
            assert not rep["errors"], rep["errors"]
            B.aggregate(rep, prompts, results, [0], 4)
            if cfg is not None:
                a = _Args(residency="on", residency_ref=None, residency_audit=True, **{f"residency_{k}": v for k, v in {
                    "drain": 2, "swaps_per_drain": 2, "hyst": 1.0, "min_abs": 1.0, "halflife": 20.0, "cap": 4,
                    "spare": 2, "demote": 4, "reserve_gib": 0.0, "frozen": False, "force_random": 2}.items()})
                rep["residency"] = B.residency_report(gen, a, rep, results, [0])
                assert gen.residency_mgr.stats["published"] > 0 and rep["residency"]["on"]
            reports[name] = rep
            loop.close()
    off, on = reports["off"], reports["on"]
    for tag in prompts:
        assert on["prompts"][tag]["accepts"] == off["prompts"][tag]["accepts"]
        assert on["prompts"][tag]["tokens_sha256"] == off["prompts"][tag]["tokens_sha256"]
        assert off["prompts"][tag]["stream"]["exact"] and on["prompts"][tag]["stream"]["exact"]
        assert on["prompts"][tag]["cold_share_by_rep"] and all(x is not None for x in on["prompts"][tag]["cold_share_by_rep"])
    rr = on["residency"]
    assert rr["swaps_per_step"]["swaps"] > 0 and rr["swaps_per_step"]["cap"] == 4
    assert rr["cold_share_curve"] and rr["cold_share_by_seq"] and rr["audit"]["ok"], rr.get("audit")
    assert rr["n_stage"] == 0 and rr["n_spare"] == 2 and rr["hbm"]["spare_GiB"] >= 0
    json.dumps(rr, default=str)   # serializable
    # --residency-ref: the off report as the reference -> accepts / tokens equal, ms delta computed
    ref_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "a22-rebase-ref.json")
    with open(ref_path, "w") as f:
        json.dump(off, f, default=str)
    a2 = _Args(residency="on", residency_ref=ref_path, residency_audit=False, residency_drain=2,
               residency_swaps_per_drain=2, residency_hyst=1.0, residency_min_abs=1.0, residency_halflife=20.0,
               residency_cap=4, residency_spare=2, residency_demote=4, residency_reserve_gib=0.0,
               residency_frozen=False, residency_force_random=2)
    on["errors"] = []
    gen.set_residency({"n_spare": 2, "n_demote": 4, "reserve_gib": 0.0})
    gen.static_step()
    vs = B.residency_report(gen, a2, on, {0: {}}, [0])["vs_off"]
    assert vs["all_accepts_equal"] is True and vs["all_tokens_equal"] is True and not on["errors"]
    gen.reset_static()
    os.remove(ref_path)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _headline():
    spec = importlib.util.spec_from_file_location("headline_rebase_t", os.path.join(REPO, "bench", "headline.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


@pytest.fixture
def emu_loop(dspark_emu, monkeypatch):
    """tests/test_dspark_loop_cpu.py's mtp-schema emulations (row-invariant GEMMs), with the residency-aware kernel
    registry installed over its fixed one (promoted / demoted addresses resolve) -> (gpu loop module, model, pool)."""
    import test_dspark_loop_cpu as T

    G = T._gpu_module(monkeypatch, dspark_emu)
    model, pool = dspark_emu("mini3")
    _install(monkeypatch, pool)
    return G, model, pool


def test_headline_residency_flags_and_window():
    B = _headline()
    ap = B.build_parser()
    a = ap.parse_args(["--model-dir", "/m", "--residency", "on", "--residency-drain", "64", "--residency-cap", "3",
                       "--residency-spare", "2", "--residency-ref", "x.json", "--residency-audit"])
    from runtime.residency import config_from_args

    cfg = config_from_args(a)
    assert cfg["drain_n"] == 64 and cfg["swaps_per_barrier"] == 3 and cfg["n_spare"] == 2 and a.residency_audit
    assert config_from_args(ap.parse_args(["--model-dir", "/m"])) is None
    runs = [{"step_ms": [10.0] * 3 + [20.0, 30.0], "verify_ms": [6.0] * 3 + [12.0, 18.0],
             "draft_ms": [4.0] * 3 + [8.0, 12.0], "accepts": [0, 1, 2, 3, 4], "mgr_ms": [0.1, 0.1, 0.1, 0.2, 0.4]}]
    w = B.window(runs, 3)
    assert w["ms_per_step"]["mean"] == 25.0 and abs(w["ms_manager_per_step"]["mean"] - 0.3) < 1e-9
    assert abs(w["ms_per_step_incl_manager"] - 25.3) < 1e-9
    assert B.window([{"step_ms": [1.0, 1.0], "accepts": [0, 0]}], 0)["ms_manager_per_step"] is None
    c = {}
    B._cold_delta(c, (100, 10), (180, 30))
    assert c == {"routed_calls": 100, "cold_calls": 20, "cold_share": 0.2}
