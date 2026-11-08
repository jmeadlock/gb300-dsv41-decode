"""STEADY-STATE (docs/HEADLINE-COLUMNS.md): CPU tests for bench/headline.py's `--steady-state` pass and its receipt stamping
(bench/receipt.py). Stdlib + pytest + torch (headline imports runtime.generate lazily); no GPU, no model: the loop and the
residency manager are fakes whose outputs are synthetic, every expected number is recomputed here from the same synthetic
runs. The pass's own contract:

  * default ON with residency on, OFF without, `--no-steady-state` off, `--steady-state` forces it;
  * it runs AFTER aggregate() and the residency report, never touches headline.* / prompts / streams, and re-captures only
    when the loop's graphs are not the headline cell's;
  * the block carries ms/step (+ verify + draft), tok/step, ms/token as total/total, cold share, swaps, streams vs 1-row and
    self-replay for the pass alone, and the delta vs the headline cell;
  * the receipt names the condition at begin() and carries the measured block at finish().
"""
import copy
import importlib.util
import json
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench")
for p in (REPO, BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

import receipt as R  # noqa: E402

HEADLINE_PY = os.path.join(BENCH, "headline.py")
DOC = os.path.join(REPO, "docs", "HEADLINE-COLUMNS.md")


def _headline():
    spec = importlib.util.spec_from_file_location("headline_steady_t", HEADLINE_PY)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


@pytest.fixture(scope="module")
def B():
    return _headline()


# ------------------------------------------------------------------------------------------------ synthetic runs
def _run(ids, accepts, step, *, verify=None, draft=None, mgr=None, cold=(10, 100), logits="L", record=None):
    n = len(accepts)
    verify = verify if verify is not None else [s * 0.6 for s in step]
    draft = draft if draft is not None else [s * 0.4 for s in step]
    return {"token_ids": list(ids), "accepts": list(accepts), "steps": n, "steps_issued": n + 8,
            "tokens_per_step": (sum(accepts) + n) / n, "step_ms": list(step), "verify_ms": list(verify),
            "draft_ms": list(draft), "mgr_ms": list(mgr) if mgr is not None else [0.1] * n,
            "prefill_ms": 3800.0, "wall_ms": 1000.0, "finish_reason": "length",
            "cold_calls": cold[0], "routed_calls": cold[1], "cold_share": cold[0] / cold[1],
            "logits_sha256": logits, "step_sha256": [f"h{i}" for i in range(len(ids))],
            "record": record if record is not None else {"tag": "x", "tokens": list(ids)}}


IDS = {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10]}
ACC = {"a": [2, 1, 2], "b": [3, 0]}
R1 = {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 99]}          # a exact vs the 1-row truth, b diverges at 3


def _prompts():
    return {t: {"ids": [0, 0], "r1_ids": list(R1[t]), "r1_sha": [f"h{i}" for i in range(len(R1[t]))],
                "r1_rec": None, "r1_ms_per_token": 5.0} for t in IDS}


def _results(step_scale=1.0, cold=(10, 100), ids=None, flip=None):
    ids = ids or IDS
    out = {}
    for t in ids:
        steps = [(10.0 + i) * step_scale for i in range(len(ACC[t]))]
        mk = lambda: _run(ids[t], ACC[t], steps, cold=cold)  # noqa: E731
        out[t] = {"fast": [mk(), mk()], "parity": [mk(), mk()]}
    if flip:
        t = flip
        out[t]["parity"][1] = _run(ids[t], ACC[t], [1.0, 1.0], cold=cold, logits="OTHER")
    return out


class FakeMgr:
    def __init__(self):
        self.host_steps = 100
        self.swap_steps = [(5, 2), (50, 1)]
        self.stats = {"issued": 3, "published": 3, "drains": 2}
        self.drain_log = [{"host_steps": 64, "cold_share": 0.12}, {"host_steps": 128, "cold_share": 0.09}]

        class _S:
            n_swaps = 3

        self.state = _S()

    def advance(self, steps, swaps, cold_shares):
        base = self.host_steps
        self.host_steps += steps
        self.swap_steps.append((base + 3, swaps))
        self.stats["issued"] += swaps
        self.stats["published"] += swaps
        self.state.n_swaps += swaps
        for c in cold_shares:
            self.stats["drains"] += 1
            self.drain_log.append({"host_steps": self.host_steps, "cold_share": c})


def _headline_rep(B, graphs=(2, 1)):
    prompts = _prompts()
    results = {2: _results(1.0, cold=(12, 100)), 1: _results(0.99, cold=(10, 100))}
    rep = {"errors": []}
    B.aggregate(rep, prompts, {g: results[g] for g in graphs}, list(graphs), 1)
    return prompts, rep


# ------------------------------------------------------------------------------------------------ flags
def test_flags_default_none_on_off(B):
    ap = B.build_parser()
    base = ["--model-dir", "/m"]
    assert ap.parse_args(base).steady_state is None
    assert ap.parse_args(base + ["--steady-state"]).steady_state is True
    assert ap.parse_args(base + ["--no-steady-state"]).steady_state is False


def test_resolve_truth_table(B):
    res = {"drain_n": 64}
    assert B.resolve_steady_state(None, res, [2, 1])[0] is True                    # default: residency on -> ON
    assert B.resolve_steady_state(None, None, [2, 1])[0] is False                  # default: residency off -> off
    assert B.resolve_steady_state(False, res, [2, 1]) == (False, "--no-steady-state")
    on, why = B.resolve_steady_state(True, None, [2, 1])                           # forced control pass
    assert on is True and "control" in why
    assert B.resolve_steady_state(True, res, [])[0] is False                       # nothing to repeat
    assert B.CELL_KEYS == {2: "two_graphs", 1: "one_graph", 0: "eager"}


# ------------------------------------------------------------------------------------------------ manager telemetry
def test_mgr_snapshot_delta(B):
    rm = FakeMgr()
    s0 = B.mgr_snapshot(rm)
    assert s0["host_steps"] == 100 and s0["swaps"] == 3 and s0["policy_swaps"] == 3 and s0["drain_log_n"] == 2
    rm.advance(40, 2, [0.06, 0.05])
    d = B.mgr_delta(s0, B.mgr_snapshot(rm))
    assert d["manager_steps"] == 40 and d["swaps"] == 2 and abs(d["swaps_per_step"] - 2 / 40) < 1e-12
    assert d["steps_with_swaps"] == 1 and d["issued"] == 2 and d["published"] == 2 and d["policy_swaps"] == 2
    assert d["drains"] == 2 and d["drain_cold_share"] == {"n": 2, "first": 0.06, "last": 0.05, "mean": 0.055}
    assert d["policy_steps_before"] == 100 and d["swaps_before"] == 3 and d["drains_before"] == 2
    assert B.mgr_snapshot(None) is None and B.mgr_delta(None, s0) is None
    assert B.mgr_snapshot(object()) is None                                       # telemetry never raises


# ------------------------------------------------------------------------------------------------ the block
def test_block_values_recompute_and_headline_untouched(B):
    prompts, rep = _headline_rep(B)
    before = json.dumps({k: rep[k] for k in ("headline", "prompts", "streams", "replay_ok", "n_prompts")}, sort_keys=True)
    rm = FakeMgr()
    s0 = B.mgr_snapshot(rm)
    rm.advance(30, 1, [0.04])
    run = {"results": _results(0.9, cold=(5, 100)), "snap0": s0, "snap1": B.mgr_snapshot(rm), "recaptured": True,
           "capture": {"n_graphs": 2}, "overflow": {"steps": 0}, "start_wall": 1000.0, "end_wall": 1180.0, "error": None}
    ss = B.steady_state_block(run, prompts, rep, 2, 1, reps=2, parity_reps=2, why="default: residency on")
    after = json.dumps({k: rep[k] for k in ("headline", "prompts", "streams", "replay_ok", "n_prompts")}, sort_keys=True)
    assert before == after and "steady_state" not in rep            # headline.* and every key under it unmoved
    # recompute from the synthetic runs: window = steps >= 1 of every fast run, pooled
    steps = [(10.0 + i) * 0.9 for t in IDS for _ in range(2) for i in range(len(ACC[t])) if i >= 1]
    toks = [a + 1 for t in IDS for _ in range(2) for i, a in enumerate(ACC[t]) if i >= 1]
    assert ss["enabled"] and ss["cell"] == "two_graphs" and ss["ok"] and ss["stats_from_step"] == 1
    assert abs(ss["total_gpu_ms"] - sum(steps)) < 1e-9 and ss["total_emitted_tokens"] == sum(toks)
    assert abs(ss["ms_per_token"] - sum(steps) / sum(toks)) < 1e-12                     # total / total
    assert abs(ss["ms_per_6row_step_dspark_k5"] - sum(steps) / len(steps)) < 1e-12
    assert abs(ss["ms_verify_graph"] - 0.6 * sum(steps) / len(steps)) < 1e-9
    assert abs(ss["ms_draft_graph"] - 0.4 * sum(steps) / len(steps)) < 1e-9
    assert abs(ss["tokens_per_step"] - sum(toks) / len(toks)) < 1e-12
    assert abs(ss["ms_per_step_incl_manager"] - (sum(steps) / len(steps) + 0.1)) < 1e-9
    assert abs(ss["cold_share"] - 0.05) < 1e-12                                          # 5 cold / 100 routed per fast run
    assert ss["window"]["total_gpu_ms"] == ss["total_gpu_ms"] and ss["ms_per_token_definition"] == B.MS_PER_TOKEN_DEF
    # swaps and the state it was warmed by
    assert ss["swaps"]["swaps"] == 1 and ss["swaps"]["manager_steps"] == 30 and ss["policy_steps_before"] == 100
    assert ss["recaptured"] is True and ss["wall_s"] == 180.0 and ss["overflow"] == {"steps": 0}
    # streams vs the 1-row truth: a exact, b diverges at 3; self-replay ok for both; tokens equal the headline cell
    assert ss["streams"] == {"exact": 1, "prompts": 2} and ss["replay_ok"] == 2 and ss["n_prompts"] == 2
    assert ss["prompts"]["a"]["stream"]["exact"] is True and ss["prompts"]["b"]["stream"]["first_divergence"] == 3
    assert all(q["equal_headline_cell"] for q in ss["prompts"].values()) and ss["vs_headline"]["streams_equal_headline_cell"] == 2
    hw = rep["headline"]["two_graphs"]
    v = ss["vs_headline"]
    assert v["headline_ms_per_step"] == hw["ms_per_step"]["mean"] and v["steady_ms_per_step"] == ss["ms_per_6row_step_dspark_k5"]
    assert abs(v["delta_ms_per_step"] - (ss["ms_per_6row_step_dspark_k5"] - hw["ms_per_step"]["mean"])) < 1e-12
    assert abs(v["delta_pct"] - v["delta_ms_per_step"] / hw["ms_per_step"]["mean"] * 100) < 1e-9
    assert abs(v["headline_cold_share"] - 0.12) < 1e-12 and abs(v["steady_cold_share"] - 0.05) < 1e-12
    json.dumps(ss)                                                                       # serialisable as is


def test_block_catches_replay_break_and_token_change(B):
    prompts, rep = _headline_rep(B)
    run = {"results": _results(1.0, flip="a"), "snap0": None, "snap1": None, "start_wall": 0.0, "end_wall": 1.0}
    ss = B.steady_state_block(run, prompts, rep, 2, 1, reps=2, parity_reps=2)
    assert ss["replay_ok"] == 1 and ss["prompts"]["a"]["replay_ok"] is False and ss["prompts"]["b"]["replay_ok"] is True
    assert ss["swaps"] is None and ss["policy_steps_before"] is None
    moved = dict(IDS, b=[7, 8, 9, 11])        # a different continuation than the headline cell emitted
    run2 = {"results": _results(1.0, ids=moved), "snap0": None, "snap1": None}
    ss2 = B.steady_state_block(run2, prompts, rep, 2, 1, reps=2, parity_reps=2)
    assert ss2["prompts"]["b"]["equal_headline_cell"] is False and ss2["vs_headline"]["streams_equal_headline_cell"] == 1


def test_block_for_eager_and_one_graph_cells(B):
    prompts = _prompts()
    rep = {"errors": []}
    B.aggregate(rep, prompts, {0: _results()}, [0], 1)
    ss = B.steady_state_block({"results": _results()}, prompts, rep, 0, 1, reps=1, parity_reps=2)
    assert ss["cell"] == "eager" and ss["vs_headline"]["headline_ms_per_step"] == rep["headline"]["eager"]["ms_per_step"]["mean"]
    assert B.steady_state_off_block("x") == {"enabled": False, "why": "x", "definition": B.STEADY_STATE_DEF}


# ------------------------------------------------------------------------------------------------ run_steady_state
class FakeSt:
    def __init__(self):
        self.ovf = {"steps": 0, "experts": 0, "steps_counted": 0, "spare_slots": 16}

    def overflow_counts(self):
        return dict(self.ovf)


class FakeLoop:
    def __init__(self, n_graphs, rm, fail_on=None):
        self.n_graphs, self.rm, self.captures, self.calls, self.st, self.fail_on = n_graphs, rm, [], [], FakeSt(), fail_on

    def capture(self, n_graphs=2):
        self.captures.append(n_graphs)
        self.n_graphs = n_graphs
        return {"n_graphs": n_graphs}

    def generate(self, ids, max_new, *, stop_on_eos=True, parity=False, tag=None, prompt=None, profile=None):
        self.calls.append(("par" if parity else "fast", tag))
        if self.fail_on and not parity and len(self.calls) == self.fail_on:
            raise RuntimeError("boom")
        self.rm.advance(3, 0, [])
        self.st.ovf["steps_counted"] += 3
        return _run(IDS["a"], ACC["a"], [10.0, 11.0, 12.0])


def test_run_steady_state_recaptures_only_when_needed(B):
    rm = FakeMgr()
    rep = {"errors": []}
    loop = FakeLoop(1, rm)      # the one-graph cell ran last
    out = B.run_steady_state(loop, rm, _prompts(), 16, 2, reps=2, parity_reps=2, stop_on_eos=True, rep=rep, log=lambda *a, **k: None)
    assert loop.captures == [2] and out["recaptured"] is True and out["error"] is None
    assert [c[0] for c in loop.calls] == ["fast", "fast", "par", "par"] * 2                  # reps then parity, per prompt
    assert out["snap1"]["host_steps"] - out["snap0"]["host_steps"] == 24 and out["overflow"]["steps_counted"] == 24
    loop2 = FakeLoop(2, rm)     # graphs "2": still on the headline cell's graph
    out2 = B.run_steady_state(loop2, rm, _prompts(), 16, 2, reps=1, parity_reps=0, stop_on_eos=True, rep=rep, log=lambda *a, **k: None)
    assert loop2.captures == [] and out2["recaptured"] is False and len(loop2.calls) == 2
    loop0 = FakeLoop(0, rm)     # eager: no capture ever
    B.run_steady_state(loop0, rm, _prompts(), 16, 0, reps=1, parity_reps=0, stop_on_eos=True, rep=rep, log=lambda *a, **k: None)
    assert loop0.captures == [] and rep["errors"] == []


def test_run_steady_state_failure_is_an_error_not_a_crash(B):
    rm = FakeMgr()
    rep = {"errors": []}
    loop = FakeLoop(2, rm, fail_on=1)
    out = B.run_steady_state(loop, rm, _prompts(), 16, 2, reps=1, parity_reps=0, stop_on_eos=True, rep=rep, log=lambda *a, **k: None)
    assert any(e["stage"] == "loop-steady-2g" and e["tag"] == "a" for e in rep["errors"])      # per-prompt failure recorded
    assert "b" in out["results"] and "a" not in out["results"] and out["error"] is None
    class Boom(FakeLoop):
        def capture(self, n_graphs=2):
            raise RuntimeError("capture died")
    out2 = B.run_steady_state(Boom(1, rm), rm, _prompts(), 16, 2, reps=1, parity_reps=0, stop_on_eos=True, rep=rep, log=lambda *a, **k: None)
    assert "capture died" in out2["error"] and rep["errors"][-1]["stage"] == "steady-state"


# ------------------------------------------------------------------------------------------------ summary + report
def test_summary_lines_headline_lines_unchanged_and_steady_lines(B):
    prompts, rep = _headline_rep(B)
    rep.update(date="d", mode="graph", impls={}, max_new=8, json="j", overflow=None)
    base = B.summary_lines(rep)
    assert not [l for l in base if "steady_state" in l]                                  # no block: nothing printed
    rm = FakeMgr()
    s0 = B.mgr_snapshot(rm)
    rm.advance(30, 1, [0.04])
    ss = B.steady_state_block({"results": _results(0.9, cold=(5, 100)), "snap0": s0, "snap1": B.mgr_snapshot(rm),
                               "start_wall": 0.0, "end_wall": 60.0}, prompts, rep, 2, 1, reps=2, parity_reps=2)
    rep["steady_state"] = ss
    lines = B.summary_lines(rep)
    extra = [l for l in lines if l not in base]
    assert [l for l in lines if l in base] == base                                       # every old line still there, verbatim
    text = "\n".join(extra)
    assert text.count("ms/token = total GPU ms / total emitted tokens = ") == 1
    assert f"{ss['total_gpu_ms']:.1f} ms / {ss['total_emitted_tokens']} tokens = {ss['total_gpu_ms'] / ss['total_emitted_tokens']:.3f}" in text
    assert "residency carried over, after 100 manager steps" in text and "cold share 5.00% (headline 12.00%)" in text
    assert "swaps 1 (0.033/step over 30 steps)" in text and "streams exact vs 1-row: 1/2" in text and "self-replay ok: 2/2" in text
    rep["steady_state"] = B.steady_state_off_block("--no-steady-state")
    assert B.summary_lines(rep)[-0] == base[0] and "  steady_state: off (--no-steady-state)" in B.summary_lines(rep)


def test_main_orders_the_pass_after_the_headline_and_before_the_dump():
    full = open(HEADLINE_PY).read()
    src = full[full.index("def main("):]
    i = src.index
    assert i("aggregate(rep, prompts, results, run_graphs, a.stats_from)") < i("residency_report(gen, a, rep, results, run_graphs, audit=False)") \
        < i("ss_run = run_steady_state(loop,") < i("residency_audit(gen, rep, rep[\"residency\"])") < i('rep["mode_gap"] = mode_gap(rep)') \
        < i('_receipt.finish(rep.get("receipt"), steady_state=rep.get("steady_state"))') < i("json.dump(slim, f")
    assert full.count("_receipt.begin(") == 1 and full.count("_receipt.finish(") == 1 and "steady_state=ss" in full
    # the pass never writes under rep["headline"]
    assert not re.search(r'rep\["headline"\]\[[^\]]+\]\s*=', full[full.index("def run_steady_state"):full.index("def summary_lines")])
    assert i('rep["overflow"] = ovf') < i("ss_run = run_steady_state(loop,")   # overflow read before the recapture zeroes it


# ------------------------------------------------------------------------------------------------ receipt
def test_receipt_condition_and_block(B, repo_env=None):
    cfg = {"enabled": True, "why": "default: residency on", "cell": "two_graphs", "reps": 2, "parity_reps": 2, "stats_from_step": 32}
    rc = R.begin(argv=[], env={}, repo="/nonexistent", steady_state=cfg)
    assert "steady_state" in R.RECEIPT_FIELDS and "steady_state_condition" in R.RECEIPT_FIELDS and set(rc) == set(R.RECEIPT_FIELDS)
    assert rc["steady_state_condition"].startswith("on: two_graphs again after the headline cells, residency state carried over (no reset)")
    assert rc["steady_state"] is None                                                    # measured later, at finish()
    assert R.begin(argv=[], env={}, repo="/nonexistent")["steady_state_condition"] == "n/a"
    off = R.begin(argv=[], env={}, repo="/nonexistent", steady_state={"enabled": False, "why": "--no-steady-state"})
    assert off["steady_state_condition"] == "off (--no-steady-state)"
    prompts, rep = _headline_rep(B)
    rm = FakeMgr()
    s0 = B.mgr_snapshot(rm)
    rm.advance(30, 2, [0.04])
    block = B.steady_state_block({"results": _results(0.9), "snap0": s0, "snap1": B.mgr_snapshot(rm), "start_wall": 1.0, "end_wall": 61.0,
                                  "overflow": {"steps": 0}}, prompts, rep, 2, 1, reps=2, parity_reps=2)
    R.finish(rc, steady_state=block)
    st = rc["steady_state"]
    for k in ("ms_per_6row_step_dspark_k5", "ms_verify_graph", "ms_draft_graph", "tokens_per_step", "ms_per_token", "total_gpu_ms",
              "total_emitted_tokens", "cold_share", "policy_steps_before", "replay_ok", "wall_s"):
        assert st[k] == block[k], k
    assert st["swaps"] == 2 and st["streams_exact"] == 1 and st["streams_prompts"] == 2 and st["overflow_steps"] == 0 and st["cell"] == "two_graphs"
    assert abs(st["delta_ms_per_step_vs_headline"] - block["vs_headline"]["delta_ms_per_step"]) < 1e-12
    assert f"steady-state two_graphs {block['ms_per_6row_step_dspark_k5']:.2f} ms/step" in R.summary_line(rc)
    json.dumps(rc, default=str)
    # disabled / absent pass: the field is None and finish never raises
    R.finish(off, steady_state=B.steady_state_off_block("x"))
    assert off["steady_state"] is None
    assert R.finish(dict(R.begin(argv=[], env={}, repo="/nonexistent")), steady_state="garbage")["steady_state"] is None
    assert R.finish(None, steady_state=block) is None


def test_begin_receipt_in_headline_stamps_the_plan(B, monkeypatch):
    seen = {}
    monkeypatch.setattr(B._receipt, "begin", lambda **kw: seen.update(kw) or {"ok": 1})
    ap = B.build_parser()
    a = ap.parse_args(["--model-dir", "/m"])                                             # residency default ON since 11/07
    B._begin_receipt(a, {}, None, [])
    assert seen["steady_state"]["enabled"] is True and seen["steady_state"]["cell"] == "two_graphs" and seen["steady_state"]["reps"] == 2
    B._begin_receipt(ap.parse_args(["--model-dir", "/m", "--residency", "off"]), {}, None, [])
    assert seen["steady_state"]["enabled"] is False
    B._begin_receipt(ap.parse_args(["--model-dir", "/m", "--no-steady-state"]), {}, None, [])
    assert seen["steady_state"]["enabled"] is False and seen["steady_state"]["why"] == "--no-steady-state"
    B._begin_receipt(ap.parse_args(["--model-dir", "/m", "--mode", "static"]), {}, None, [])
    assert seen["steady_state"]["cell"] == "eager"


# ------------------------------------------------------------------------------------------------ docs
def test_columns_doc_names_every_contract_point():
    assert os.path.exists(DOC), "docs/HEADLINE-COLUMNS.md"
    doc = open(DOC).read()
    for needle in ("headline.two_graphs", "steady_state", "--steady-state", "--no-steady-state", "policy_steps_before",
                   "ms_per_6row_step_dspark_k5", "total GPU ms / total emitted tokens", "receipt", "cold share", "swaps",
                   "headline.two_graphs.ms_per_step.mean", "steady_state.ms_per_6row_step_dspark_k5", "steady_state.ms_per_token"):
        assert needle in doc, needle


# ------------------------------------------------------------------------------------------------ emulated loop, end to end
from test_residency_rebase_cpu import _Args, dspark_emu, emu, emu_loop, _install, remu  # noqa: E402,F401  (fixtures + helpers)


def test_steady_pass_on_the_emulated_loop_with_a_live_residency_manager(emu_loop):
    """The real DSparkLoop + ResidencyManager on the CPU emulation (eager, row-invariant): headline cell, residency report,
    then the steady-state pass -> same tokens as the headline cell, streams exact vs the 1-row truth, a live swap count,
    the policy carried over (steps before == the manager's steps after the cell), headline untouched by the pass."""
    import torch

    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator
    from tiny_tok import make_wordlevel_tokenizer

    G, model, pool = emu_loop
    B = _headline()
    cfg = {"drain_n": 2, "min_abs": 1.0, "hyst": 1.0, "halflife": 20.0, "n_spare": 2, "n_demote": 4, "force_random": 2,
           "reserve_gib": 0.0}
    with G.row_invariant_mode():
        gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static", max_pos=255)
        gen.static_kw = {"_allow_cpu": True}
        g = torch.Generator().manual_seed(11)
        prompts = {}
        for tag, L in (("a", 17), ("b", 18)):
            ids = torch.randint(2, 2000, (L,), generator=g).tolist()
            rec = gen.greedy(ids, 20, tag=tag)
            m = gen.last_meta
            prompts[tag] = {"ids": ids, "r1_ids": list(m["token_ids"]), "r1_sha": list(m["step_sha256"]), "r1_rec": rec}
        gen.set_residency(cfg)
        loop = DSparkLoop(gen, 20, _allow_cpu=True)
        rep = {"errors": [], "date": "dry", "mode": "static", "impls": {}, "max_new": 20}
        results = B.run_cells(loop, prompts, 20, [0], reps=2, parity_reps=1, stop_on_eos=True, rep=rep)
        B.aggregate(rep, prompts, results, [0], 4)
        a = _Args(residency="on", residency_ref=None, residency_audit=True, **{f"residency_{k}": v for k, v in {
            "drain": 2, "swaps_per_drain": 2, "hyst": 1.0, "min_abs": 1.0, "halflife": 20.0, "cap": 4, "spare": 2,
            "demote": 4, "reserve_gib": 0.0, "frozen": False, "force_random": 2}.items()})
        rep["residency"] = B.residency_report(gen, a, rep, results, [0], audit=False)
        assert "audit" not in rep["residency"]
        head_before = json.dumps({k: rep[k] for k in ("headline", "prompts", "streams", "residency")}, sort_keys=True, default=str)
        rm = gen.residency_mgr
        steps_before = rm.host_steps
        assert B.resolve_steady_state(None, rep["residency"]["config"], [0])[0] is True
        run = B.run_steady_state(loop, rm, prompts, 20, 0, reps=2, parity_reps=1, stop_on_eos=True, rep=rep,
                                 log=lambda *x, **k: None)
        ss = B.steady_state_block(run, prompts, rep, 0, 4, reps=2, parity_reps=1, why="default: residency on")
        B.residency_audit(gen, rep, rep["residency"])
        assert rep["errors"] == [] and run["error"] is None and ss["ok"]
        assert ss["cell"] == "eager" and ss["policy_steps_before"] == steps_before > 0       # carried over, not reset
        assert ss["swaps"]["manager_steps"] == rm.host_steps - steps_before > 0
        assert ss["swaps"]["swaps"] == sum(k for s, k in rm.swap_steps if s > steps_before)
        assert all(q["equal_headline_cell"] for q in ss["prompts"].values()) and ss["vs_headline"]["streams_equal_headline_cell"] == 2
        assert ss["streams"] == {"exact": 2, "prompts": 2} and ss["replay_ok"] == 2
        # no CUDA events on the CPU emulation: the timing fields are empty, the structure and the numerics checks are real
        assert ss["total_gpu_ms"] is None and ss["ms_per_token"] is None and ss["cold_share"] is not None
        assert rep["residency"]["audit"]["ok"]
        after = json.loads(json.dumps({k: rep[k] for k in ("headline", "prompts", "streams")}, sort_keys=True, default=str))
        assert {k: v for k, v in json.loads(head_before).items() if k != "residency"} == after       # the pass moved nothing it does not own
        rep["steady_state"] = ss
        json.dumps(rep, default=str)
        assert any("steady_state (eager again" in l for l in B.steady_state_lines(ss))
        loop.close()
