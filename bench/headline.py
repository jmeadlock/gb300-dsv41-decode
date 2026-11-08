#!/usr/bin/env python3
"""headline.py -- M1-T12: the project's headline instrument, "ms per 6-row step, DSpark k=5" (PROJECT.md "The number").

  python3 bench/headline.py --model-dir /models/DeepSeek-V4.1-... [--map docs/maps/hot-v26ar-seed-295.json] \\
      [--fixture w0/results/phase1/parity-A.jsonl] [--max-new 256] [--moe-impl dg-inplace --fp8-impl gemv \\
      --glue-impl fused --attn-impl flashmla] [--graphs 2,1] [--r1-ref w0/results/first-light-cand-night-20261107.jsonl] \\
      [--json docs/headline-<date>.json]

What it does (docs/HANDOFF-M1-T12-LOOP.md 3):
  0. DSV41_GEMV_PIN_M1=1 before any build (default; `--no-gemv-pin` drops it): the 6-row fp8 Linears take the 1-row
     GEMV configs (M1-T10 census: the tuned M=6 configs are not row-invariant). Boot with load_mtp=True.
  1. truth: the 1-row greedy per fixture prompt in this process (Generator.greedy, --mode, the given impls): token ids
     + per-position fp32-logits sha256. --r1-ref: the fresh 1-row token strings also compared with a stored record.
  2. the loop (runtime/dspark_loop.py DSparkLoop: verify 6 rows -> accept by exact match -> commit -> draft, on the
     device; async EOS poll every --poll-every steps), built once (the dg-inplace pool pre-grown first), captured as
     TWO graphs (A = verify|accept|commit, B = draft|next input) so the ms is attributed. Per prompt:
       fast x --reps   (no host sync in the decode; CUDA events per step: start | after A | after B)
       parity x --parity-reps (sync per step: the emitted rows' fp32 logits -> per-position sha256 + a parity record)
  3. recaptured as ONE graph per step (--graphs containing 1): fast x 1 per prompt.
Reported per prompt and pooled: (a) emitted stream vs the 1-row stream: exact, or the first divergence (index, both
tokens, the 1-row top-1/top-2 logprob margin there = how near the tie was) and how many emitted rows' logits are bit-equal
to the 1-row's; (b) ms per 6-row step (steps >= --stats-from of each sequence; verify graph + draft graph), tokens per
step (accepted drafts + 1), ms per token = sum(step ms) / sum(tokens) over the window, prefill ms (reference prefill +
draft-ring seeding + first draft, synced); (c) self-replay: fast reps, parity reps (records byte-identical, logits
digests equal), fast == parity, 1 graph == 2 graphs (tokens + accept-length sequences); (d) HBM after the build.
A2-2 (docs/A2-2-BOX-PLAN.md): `--residency on --residency-drain 64 [--residency-spare N ...]` puts the host-side swap
policy (runtime/residency.py) on the loop's 6-row verify step (the 1-row truth above runs on the static map). The pool
is pre-grown once to 16 spare slots/layer; the step stages cold experts into `--residency-stage` of them (default 8,
else --spare-slots) and residency promotes into the REST (8): no new HBM unless --residency-spare asks for more than the
pool holds and free HBM - reserve allows. A cold-unique count above the staging slots overflows (wrong logits for that
step): the `overflow` field and the residency line report it; give the off run the same `--spare-slots`/`--pool-spare`. The loop calls the manager after every step; the
report key `residency` has swaps/step, the cold-share curve (per drain and per sequence), the headroom calc and HBM, the
manager's GPU ms per step (`ms_manager_per_step`; `ms_per_step_incl_manager` is the number to compare with the off run)
and, with `--residency-ref <the off run's JSON>`, per prompt emitted-token / accept-length equality (an error if not).
`mode_gap` (LOOSE-1107): one_graph vs two_graphs. Without residency they agree within 0.03 ms in every run; with
residency on the two_graphs cell is measured first and the one_graph cell last, at a different swap-policy state (cold
share), so the report carries the gap, its split (verify vs draft graph, mean/median/p90), each cell's cold share, and
a weighted fit of step ms vs cold share (modeled, labeled) -- one printed `mode gap:` line. Defaults are unchanged.
LOOSE-ENDS-2: `ms/token` is TOTAL GPU ms / TOTAL emitted tokens over the window (pooled over prompts, not a mean of
ratios); the window stores `total_gpu_ms` and `total_emitted_tokens` and the summary prints "ms/token = total GPU ms /
total emitted tokens = X ms / N tokens = Y". The report carries a `receipt` block (bench/receipt.py: git commit +
dirty-tree sha256, DSV41_* switches, impls, checkpoint config hash, fixture hash, torch/triton/cuda, driver, GPU uuid,
SM/mem clocks at start and end, residency condition, start/end wall time).
The C1 instrument proper (nsys cudaProfilerApi range) is M1-T13; --profile TAG:FIRST:N opens that range around N steps
of one prompt's first fast run, for `nsys profile --capture-range=cudaProfilerApi`.

Box line (orchestrator; the fast path of M1-T10/T11; --spare-slots 16 because M1-T10 measured 123/1018 overflowing
steps at 8 -- an overflowed layer's outputs are wrong for that step, which would show up as stream divergences; the
summary's `overflow` field reports it either way):
  W=$HOME/hermes/gb300-dsv41-decode-wt12
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v /models:/models \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/headline.py \\
    --model-dir /models/DeepSeek-V4.1-... --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused \\
    --attn-impl flashmla --spare-slots 16 --r1-ref w0/results/first-light-cand-night-20261107.jsonl
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import receipt as _receipt  # noqa: E402  (bench/receipt.py: LOOSE-ENDS-2 receipt stamping)

METRIC = "ms per 6-row step, DSpark k=5"           # PROJECT.md "The number" (C1, 6 rows, DSpark k=5)
PG61_MS_PER_STEP = 7.36                              # PROJECT.md baseline (production, 2026-10-29)
CHARTER_BAR_MS = 5.0                                 # PROJECT.md bar
MS_PER_TOKEN_DEF = ("total GPU ms / total emitted tokens: sum of the CUDA-event step times (verify graph + draft graph) "
                    "over the window's live steps, divided by the sum of tokens emitted in those steps "
                    "(accepted drafts + 1 per step), pooled over all prompts and fast reps")


# ------------------------------------------------------------------------------------------------ pure helpers
def _stats(xs: List[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def pct(q):
        k = (len(s) - 1) * q / 100.0
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": pct(50), "p90": pct(90), "min": s[0], "max": s[-1]}


def window(runs: Sequence[dict], stats_from: int) -> dict:
    """Pool the live steps with index >= stats_from of every run -> ms/step (total, verify, draft), tokens/step,
    ms/token over that window; plus the same over all live steps (`all_*`)."""
    step, ver, dra, tok, all_step, all_tok, mgr = [], [], [], [], [], [], []
    for r in runs:
        ms = r.get("step_ms") or []
        acc = r.get("accepts") or []
        v, d = r.get("verify_ms"), r.get("draft_ms")
        mg = r.get("mgr_ms") or []     # A2-2: the residency manager's GPU ms after each step (not in step_ms)
        for i, m in enumerate(ms):
            all_step.append(m)
            all_tok.append(acc[i] + 1)
            if i >= stats_from:
                step.append(m)
                tok.append(acc[i] + 1)
                if v:
                    ver.append(v[i])
                if d:
                    dra.append(d[i])
                if i < len(mg):
                    mgr.append(mg[i])
    out = {"stats_from_step": stats_from, "steps": len(step), "ms_per_step": _stats(step),
           "ms_manager_per_step": _stats(mgr) if mgr else None,
           "ms_per_step_incl_manager": ((sum(step) / len(step) + sum(mgr) / len(mgr)) if step and mgr else None),
           "ms_verify_graph": _stats(ver) if ver else None, "ms_draft_graph": _stats(dra) if dra else None,
           "tokens_per_step": (sum(tok) / len(tok)) if tok else None,
           "ms_per_token": (sum(step) / sum(tok)) if tok else None,
           # LOOSE-ENDS-2 (Astra): ms/token is TOTAL GPU ms / TOTAL emitted tokens over the window (pooled over prompts),
           # not a mean of per-prompt ratios; the two totals are stored so the quotient can be re-derived from the receipt.
           "total_gpu_ms": sum(step) if step else None, "total_manager_ms": sum(mgr) if mgr else None, "total_emitted_tokens": sum(tok) if tok else None,
           "ms_per_token_definition": MS_PER_TOKEN_DEF,
           "all_steps": len(all_step), "all_ms_per_step": _stats(all_step),
           "all_tokens_per_step": (sum(all_tok) / len(all_tok)) if all_tok else None,
           "all_ms_per_token": (sum(all_step) / sum(all_tok)) if all_tok else None,
           "all_total_gpu_ms": sum(all_step) if all_step else None,
           "all_total_emitted_tokens": sum(all_tok) if all_tok else None}
    return out


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def margin_at(rec: Optional[dict], k: Optional[int]) -> Optional[float]:
    """top-1 minus top-2 logprob of a parity record's entry k (how near the tie was), or None."""
    if rec is None or k is None:
        return None
    lp = rec.get("logprobs") or []
    if k >= len(lp):
        return None
    top = lp[k].get("top_logprobs") or []
    return float(top[0]["logprob"]) - float(top[1]["logprob"]) if len(top) >= 2 else None


def compare_stream(loop_ids, r1_ids, loop_sha=None, r1_sha=None, loop_rec=None, r1_rec=None) -> dict:
    k = first_divergence(loop_ids, r1_ids)
    out = {"exact": k is None, "first_divergence": k, "loop_tokens": len(loop_ids), "r1_tokens": len(r1_ids)}
    if k is not None:
        out["loop_token"] = loop_ids[k] if k < len(loop_ids) else None
        out["r1_token"] = r1_ids[k] if k < len(r1_ids) else None
        out["r1_margin_at_divergence"] = margin_at(r1_rec, k)
        out["loop_margin_at_divergence"] = margin_at(loop_rec, k)
    if loop_sha is not None and r1_sha is not None:
        n = min(len(loop_sha), len(r1_sha), k if k is not None else len(loop_sha))
        out["logit_rows_compared"] = n
        out["logit_rows_bit_equal"] = sum(x == y for x, y in zip(loop_sha[:n], r1_sha[:n]))
    return out


def glue_orders(st) -> Optional[dict]:
    """T9a-3: the fused glue's reduction orders of a StaticStep ({key: order name}) or None (glue ref / no orders).
    A key whose order differs between the 1-row truth step and the 6-row verify step reduces that statistic in a
    different fp32 order at 1 and 6 rows: the two steps' rows cannot be bit-equal through it."""
    o = getattr(getattr(st, "glue", None), "orders", None)
    if not o:
        return None
    return {str(k): (v.get("name") or str(v.get("order"))) if isinstance(v, dict) else str(v) for k, v in o.items()}


def glue_off(st) -> Optional[List[str]]:
    g = getattr(st, "glue", None)
    if g is None or not hasattr(g, "status"):
        return None
    return sorted(c for c in g.status if not g.on(c))


def rowmatch_record(st1) -> dict:
    """T9a-3: the 1-row (truth) step's DG call shape and its per-row build check, plus its glue orders / off list."""
    dg = getattr(st1, "dg", None)
    rm = getattr(dg, "rowmatch", None)
    chk = (rm or {}).get("check") or {}
    return {"dg": None if rm is None else {
                "mode": rm.get("mode"), "on": rm.get("on"), "why": rm.get("why"), "align": getattr(dg, "align", None),
                "gemm_blocks": getattr(dg, "gemm_blocks", None), "rows_gemm": rm.get("rows_gemm"),
                "check_equal": chk.get("equal"), "today_equal": chk.get("today_equal"),
                "check": {p: {k_: v.get(k_) for k_ in ("build_vs_rows", "today_vs_rows", "of")}
                          for p, v in (chk.get("projs") or {}).items()}},
            "glue_orders_one_row": glue_orders(st1), "glue_off_one_row": glue_off(st1)}


def orders_compare(rec: dict, st6) -> dict:
    """rec (rowmatch_record) + the verify step's glue orders / off list -> which orders differ."""
    a, b = rec.get("glue_orders_one_row"), glue_orders(st6)
    rec["glue_orders_verify"] = b
    rec["glue_off_verify"] = glue_off(st6)
    if a is None or b is None:
        rec["glue_orders_differ"] = None
    else:
        rec["glue_orders_differ"] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    o1, o6 = rec.get("glue_off_one_row"), rec.get("glue_off_verify")
    rec["glue_off_only_verify"] = None if o1 is None or o6 is None else sorted(set(o6) - set(o1))
    return rec


def rowmatch_line(rm: dict) -> str:
    d = rm.get("dg") or {}
    s = (f"DG 1-row call {'ON' if d.get('on') else 'off'} ({d.get('mode')}, align {d.get('align')} x "
         f"{d.get('gemm_blocks')} blocks; per-row build check {d.get('check_equal')}, today's call "
         f"{d.get('today_equal')})") if d else "DG n/a"
    diff = rm.get("glue_orders_differ")
    if diff is not None:
        s += f" | glue orders 1-row vs verify differ at {diff or 'none'}"
    only = rm.get("glue_off_only_verify")
    if only:
        s += f" | glue off only at 6 rows: {only}"
    return s


MODE_GAP_TOL_MS = 0.05      # every run without residency has one_graph and two_graphs within 0.03 ms (docs/headline-*.json)


def cold_pooled(runs: Sequence[dict]) -> Optional[float]:
    """Cold share of the routed expert calls over `runs` (call-weighted: sum cold / sum routed), or None when the
    runs carry no counters (a static-map run without residency still has them; JSONs before LOOSE-1107 do not for
    the one_graph cell)."""
    cold = sum(x.get("cold_calls") or 0 for x in runs if x.get("routed_calls"))
    tot = sum(x.get("routed_calls") or 0 for x in runs)
    return (cold / tot) if tot else None


def _wls(pts: Sequence[tuple]) -> Optional[dict]:
    """Weighted least squares y = a + b x over (x, y, w) -> {n, intercept, slope, r} (None: < 3 points or no spread)."""
    pts = [(float(x), float(y), float(w)) for x, y, w in pts if x is not None and y is not None and w]
    if len(pts) < 3:
        return None
    sw = sum(w for _, _, w in pts)
    mx = sum(w * x for x, _, w in pts) / sw
    my = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    syy = sum(w * (y - my) ** 2 for _, y, w in pts)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
    if sxx <= 0 or syy <= 0:
        return None
    b = sxy / sxx
    return {"n": len(pts), "intercept": my - b * mx, "slope": b, "r": sxy / (sxx * syy) ** 0.5}


def mode_gap(rep: dict) -> Optional[dict]:
    """LOOSE-1107: why one_graph and two_graphs differ in a run (None when the run has only one of them).

    Both cells replay the same kernels; the loop adds ~0 (every run without residency has them within 0.03 ms). What
    can differ between the cells is the STATE the run is in when each is measured: `run_cells` measures the first
    cell (two_graphs: fast x reps + parity x parity_reps per prompt) first and the one_graph cell LAST, one fast run
    per prompt, and with residency on the manager swaps experts between replays for the whole run, so the cold
    share -- cold experts are staged and cost verify time -- falls from cell to cell (docs/headline-20261107-
    residency.json: code 12.9 % -> 10.0 % -> 7.5 % -> 6.2 % over its first four runs). The report therefore carries
    the gap, where it sits (verify graph vs draft graph, mean vs median, per prompt), the residency state of each
    cell (cold share when recorded), a weighted fit of the two_graphs cell's own per-prompt step ms against its cold
    share, and -- when the one_graph cell's cold share is not recorded -- a MODELED one_graph ms from the fit and the
    policy's drain log over the last drains (labeled modeled). `within_tolerance` False without residency would be
    a real mode effect; with residency on the cells are not comparable and `cause` says so."""
    h = rep.get("headline") or {}
    w2, w1 = h.get("two_graphs"), h.get("one_graph")
    if not w2 or not w1 or not (w2.get("ms_per_step") or {}).get("n") or not (w1.get("ms_per_step") or {}).get("n"):
        return None
    g = lambda w, k: (w.get(k) or {}).get("mean") if w.get(k) else None      # noqa: E731
    q = lambda w, k: (w.get("ms_per_step") or {}).get(k)                        # noqa: E731
    m2, m1 = g(w2, "ms_per_step"), g(w1, "ms_per_step")
    res = rep.get("residency") or {}
    res_on = bool(rep.get("residency_config")) or bool(res.get("on"))
    out: dict = {"two_graphs_ms": m2, "one_graph_ms": m1, "gap_ms": m2 - m1, "gap_pct": (m2 - m1) / m1 * 100,
                 "tolerance_ms": MODE_GAP_TOL_MS, "within_tolerance": abs(m2 - m1) <= MODE_GAP_TOL_MS,
                 "gap_p50_ms": q(w2, "p50") - q(w1, "p50"), "gap_p90_ms": q(w2, "p90") - q(w1, "p90"),
                 "steps": {"two_graphs": w2.get("steps"), "one_graph": w1.get("steps")},
                 "draft_graph_ms_two_graphs": g(w2, "ms_draft_graph"), "verify_graph_ms_two_graphs": g(w2, "ms_verify_graph"),
                 "residency_on": res_on, "cell_order": "two_graphs measured first, one_graph last",
                 "sf_store": res.get("sf_store") if res_on else None}
    if res_on:
        mg2, mg1 = g(w2, "ms_manager_per_step"), g(w1, "ms_manager_per_step")
        out["manager_ms_per_step"] = {"two_graphs": mg2, "one_graph": mg1}
        out["gap_incl_manager_ms"] = (w2.get("ms_per_step_incl_manager") - w1.get("ms_per_step_incl_manager")
                                      if w2.get("ms_per_step_incl_manager") and w1.get("ms_per_step_incl_manager")
                                      else None)
    c2, c1 = w2.get("cold_share"), w1.get("cold_share")
    out["cold_share"] = {"two_graphs": c2, "one_graph": c1}
    per, pts = {}, []
    for tag, p in (rep.get("prompts") or {}).items():
        a, b = p.get("window") or {}, p.get("window_one_graph") or {}
        a_all, b_all = (a.get("all_ms_per_step") or {}).get("mean"), (b.get("all_ms_per_step") or {}).get("mean")
        if a_all is None or b_all is None:
            continue
        reps_ = [x for x in (p.get("cold_share_by_rep") or []) if x is not None]
        cold2 = (sum(reps_) / len(reps_)) if reps_ else None
        per[tag] = {"steps": p.get("steps"), "two_graphs_ms": a_all, "one_graph_ms": b_all, "gap_ms": a_all - b_all,
                    "cold_share_two_graphs_reps": p.get("cold_share_by_rep"),
                    "cold_share_one_graph_reps": p.get("cold_share_by_rep_one_graph")}
        pts.append((cold2, a_all, p.get("steps") or 0))
    out["per_prompt"] = per
    fit = _wls(pts) if res_on else None
    if fit:
        out["fit_two_graphs_ms_vs_cold_share"] = dict(
            fit, ms_per_cold_pct_point=fit["slope"] / 100.0,
            what="weighted LSQ over prompts: mean step ms of the two_graphs cell (all live steps) vs the mean cold "
                 "share of its fast reps; MODELED relation, not a measurement of the one_graph cell")
    if fit and res_on:
        if c2 is not None and c1 is not None:
            out["gap_predicted_from_cold_share_ms"] = fit["slope"] * (c2 - c1)      # modeled
        else:   # one_graph cold share not recorded: take the policy's drain log over the last one_graph steps
            dl = res.get("drain_log") or []
            n1, host = int(w1.get("all_steps") or 0), int(res.get("host_steps") or 0)
            tail = [r["cold_share"] for r in dl if r.get("cold_share") is not None and r.get("host_steps", 0) > host - n1]
            if tail:
                cs = sum(tail) / len(tail)
                out["one_graph_ms_modeled"] = {"cold_share_drain_tail": cs, "drains": len(tail),
                                               "ms": fit["intercept"] + fit["slope"] * cs,
                                               "actual_ms": (w1.get("all_ms_per_step") or {}).get("mean") or m1,
                                               "what": "MODELED: fit intercept + slope x the cold share of the drains "
                                                       "covering the last one_graph steps (drain log granularity: 64 steps)"}
    # the verdict
    if out["within_tolerance"]:
        out["cause"] = (f"none: the cells agree within {MODE_GAP_TOL_MS} ms (same kernels in the same order; the "
                        f"residency hook runs after each step's end event in both)")
        out["headline_mode"] = "either: the cells are the same measurement"
    elif not res_on:
        out["cause"] = (f"UNEXPLAINED mode effect: residency off, gap {out['gap_ms']:+.3f} ms; check draft vs verify "
                        f"graph attribution and run order")
        out["headline_mode"] = "unresolved"
    else:
        bits = [f"residency on: the cells are measured at different policy states (two_graphs first, one_graph last)"]
        if c2 is not None and c1 is not None:
            bits.append(f"cold share {c2 * 100:.2f}% (two_graphs) vs {c1 * 100:.2f}% (one_graph)")
        if fit:
            bits.append(f"two_graphs step ms rises {fit['slope'] / 100:.3f} ms per cold-share point across prompts "
                        f"(r {fit['r']:.2f}, n {fit['n']})")
        if "one_graph_ms_modeled" in out:
            mm = out["one_graph_ms_modeled"]
            bits.append(f"modeled one_graph {mm['ms']:.2f} ms at drain-tail cold share {mm['cold_share_drain_tail'] * 100:.2f}% "
                        f"(actual {mm['actual_ms']:.2f})")
        sf = out.get("sf_store") or {}
        if sf.get("got") and sf.get("got") != sf.get("requested"):
            bits.append(f"sf_store {sf.get('requested')}->{sf.get('got')} is a per-run setting, identical in both cells "
                        f"(it cannot make one mode slower)")
        out["cause"] = "; ".join(bits)
        out["headline_mode"] = ("not decidable from this run: the cells differ in residency state, so the gap is "
                                "not a property of the graph mode. Compare modes at a fixed state (--residency-frozen "
                                "after a warm run, or both cells after the same number of passes). Defaults unchanged")
    return out


def mode_gap_line(mg: Optional[dict]) -> Optional[str]:
    if not mg:
        return None
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"          # noqa: E731
    s = (f"  mode gap: two_graphs {f(mg['two_graphs_ms'])} - one_graph {f(mg['one_graph_ms'])} = {mg['gap_ms']:+.3f} ms "
         f"(median {mg['gap_p50_ms']:+.2f}, p90 {mg['gap_p90_ms']:+.2f}; draft graph {f(mg['draft_graph_ms_two_graphs'])}"
         f" ms in the two_graphs cell) | {mg['cause']}")
    return s


def summary_lines(rep: dict) -> List[str]:
    h = rep.get("headline") or {}
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = [f"HEADLINE {rep.get('date')} | {METRIC} | mode {rep.get('mode')} | impls {rep.get('impls')} | "
             f"gemv pin M1 {rep.get('gemv_pin_m1')} | max_new {rep.get('max_new')} | prompts {rep.get('n_prompts')}"]
    sw = rep.get("small_wins")
    if sw:   # SMALL-WINS: what the row-conditional defaults resolved to (runtime.fused_glue_r2.resolved_line's content)
        lines.append("  small-wins: " + "; ".join(
            f"R={r} mixes={c['mixes']}({c['mixes_src']}) qkv_tail={c['qkv_tail']}({c['qkv_tail_src']})"
            for r, c in (sw.get("rows") or {}).items()) + f"; fp8 {sw.get('fp8_impl_arg')}"
            + (f" (kernel {sw.get('gemv_kernel')})" if sw.get("gemv_kernel") else ""))
    rm = rep.get("rowmatch")
    if rm:
        lines.append(f"  rowmatch: {rowmatch_line(rm)}")
    for key in ("two_graphs", "one_graph", "eager"):
        w = h.get(key)
        if not w:
            continue
        ms = (w.get("ms_per_step") or {}).get("mean")
        v = (w.get("ms_verify_graph") or {}).get("mean") if w.get("ms_verify_graph") else None
        d = (w.get("ms_draft_graph") or {}).get("mean") if w.get("ms_draft_graph") else None
        mg = (w.get("ms_manager_per_step") or {}).get("mean") if w.get("ms_manager_per_step") else None
        mgs = f" [+ residency manager {f(mg, 3)} ms/step = {f(w.get('ms_per_step_incl_manager'))}]" if mg is not None else ""
        lines.append(f"  {key:10s}: {f(ms)} ms/step{mgs} (verify {f(v)} + draft {f(d)}) | {f(w.get('tokens_per_step'), 3)} "
                     f"tok/step | {f(w.get('ms_per_token'))} ms/token | window steps >= {w.get('stats_from_step')}: "
                     f"{w.get('steps')} (all {w.get('all_steps')} steps: {f((w.get('all_ms_per_step') or {}).get('mean'))}"
                     f" ms/step, {f(w.get('all_tokens_per_step'), 3)} tok/step)")
        tg, tt = w.get("total_gpu_ms"), w.get("total_emitted_tokens")
        if tg is not None and tt:
            lines.append(f"  {key:10s}  ms/token = total GPU ms / total emitted tokens = {f(tg, 1)} ms / {tt} tokens = "
                         f"{f(tg / tt, 3)}  [measured: CUDA events; window steps >= {w.get('stats_from_step')}, {w.get('steps')} steps]")
    mgl = mode_gap_line(rep.get("mode_gap"))
    if mgl:
        lines.append(mgl)
    lines.append(f"  vs Pg61 {PG61_MS_PER_STEP} ms/step (bar {CHARTER_BAR_MS}); prefill {f(h.get('prefill_ms_mean'), 1)} ms "
                 f"mean; HBM after build {rep.get('hbm_after_build_GiB')}")
    lines.append(f"  {'prompt':6s} {'tok':>4s} {'steps':>5s} {'tok/st':>6s} {'ms/step':>8s} {'ms/tok':>7s} "
                 f"{'stream vs 1-row':28s} {'replay':6s}")
    for tag, p in (rep.get("prompts") or {}).items():
        s = p.get("stream") or {}
        st = "exact" if s.get("exact") else (f"diverge @{s.get('first_divergence')} (r1 margin "
                                             f"{f(s.get('r1_margin_at_divergence'), 3)})" if s else "-")
        w = p.get("window") or {}
        lines.append(f"  {tag:6s} {p.get('tokens', 0):4d} {p.get('steps', 0):5d} {f(p.get('tokens_per_step'), 3):>6s} "
                     f"{f((w.get('all_ms_per_step') or {}).get('mean')):>8s} {f(w.get('all_ms_per_token')):>7s} "
                     f"{st:28s} {'ok' if p.get('replay_ok') else 'DIFF':6s}")
    sm = rep.get("streams") or {}
    lines.append(f"  streams exact vs 1-row: {sm.get('exact')}/{sm.get('prompts')}; self-replay ok: "
                 f"{rep.get('replay_ok')}/{rep.get('n_prompts')}; overflow {rep.get('overflow')}; errors "
                 f"{len(rep.get('errors') or [])} | {rep.get('json')}")
    return lines


# ------------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--hot-per-layer", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--fixture", default=os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl"))
    ap.add_argument("--only", default=None, help="comma-separated fixture tags")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pos", type=int, default=4095)
    ap.add_argument("--mode", choices=["static", "graph"], default="graph")
    ap.add_argument("--graphs", default="2,1", help="loop graph counts to run, in order (2 first: attributed ms)")
    ap.add_argument("--moe-impl", choices=["ref", "dg-masked", "dg-inplace"], default="ref")
    ap.add_argument("--fp8-impl", choices=["ref", "gemv", "gemv3"], default="ref",
                    help="ref | gemv | gemv3 (= gemv + the T9b-6 v3 kernel path, $DSV41_GEMV_IMPL=gemv3; A3 class)")
    ap.add_argument("--glue-impl", choices=["ref", "fused"], default="ref")
    ap.add_argument("--attn-impl", choices=["ref", "bounded", "flashmla"], default="ref")
    ap.add_argument("--fp32-impl", choices=["cublas", "fixed"], default="cublas",
                    help="T9b-5: the mixes' / head's fp32 F.linear: cublas, or our fixed-order kernels (row-invariant; "
                         "sets $DSV41_FP32_IMPL before any build, so the 1-row truth and the loop both use it)")
    ap.add_argument("--draft-impls", choices=["same", "ref"], default="same",
                    help="M1-T11-PORT: same = the draft on the verify's impls (draft_step.draft_impls_for); ref = T11")
    ap.add_argument("--spare-slots", type=int, default=None, help="6-row dg-inplace spare slots/layer (default 16)")
    ap.add_argument("--pool-spare", type=int, default=None,
                    help="pre-grow the pool to at least this many spare slots/layer (default: --spare-slots or 16; with "
                         "--residency on never below 16). Give the residency-off run the same value as the on run so "
                         "both start from the same HBM")
    ap.add_argument("--moe-rowmatch", choices=["on", "align", "off"], default="off",
                    help="T9a-3: the 1-row (truth) dg-inplace routed GEMM in the 6-row step's call shape, checked per "
                         "row at build (runtime/dg_inplace.build_one_row; sets $DSV41_MOE_ROWMATCH before any build)")
    ap.add_argument("--no-gemv-pin", action="store_true", help="do not set DSV41_GEMV_PIN_M1=1")
    ap.add_argument("--reps", type=int, default=2, help="fast (async) runs per prompt at 2 graphs")
    ap.add_argument("--parity-reps", type=int, default=2, help="parity (per-step sync) runs per prompt")
    ap.add_argument("--poll-every", type=int, default=8)
    ap.add_argument("--stats-from", type=int, default=32)
    ap.add_argument("--no-stop-on-eos", action="store_true")
    ap.add_argument("--r1-ref", default=None, help="stored 1-row JSONL to compare token strings with (optional)")
    ap.add_argument("--profile", default=None, help="TAG:FIRST:N -> cudaProfilerStart/Stop around N steps")
    ap.add_argument("--pg61-step", default=os.path.join(REPO, "w0", "results", "phase1", "step-pg61.json"))
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--json", default=None, help="default docs/headline-<YYYYMMDD>.json")
    # A2-2: adaptive residency (runtime/residency.py) on the 6-row verify step; the loop calls the manager every step
    from runtime.residency import add_cli_args

    add_cli_args(ap)
    ap.add_argument("--residency-ref", default=None,
                    help="the residency-off run's headline JSON: per prompt tokens/accepts equal, ms/step delta")
    ap.add_argument("--residency-audit", action="store_true",
                    help="end of run: sha256 of every moved expert (HBM and Grace copies) vs the checkpoint shards")
    return ap


def _hbm(torch) -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"free": round(free / 2**30, 3), "total": round(total / 2**30, 3),
            "allocated": round(torch.cuda.memory_allocated() / 2**30, 3),
            "max_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 3)}


def _routed(loop) -> Optional[tuple]:
    """(hot calls, cold calls) summed over layers since capture (StaticStep.routed_counts; syncs -- called between
    sequences, never inside one)."""
    try:
        rc = loop.st.routed_counts()
    except Exception:
        return None
    return None if rc is None else (sum(rc[0]), sum(rc[1]))


def _cold_delta(res: dict, c0: Optional[tuple], c1: Optional[tuple]) -> None:
    """A2-2: routed expert calls of one sequence and their cold share (the device counters run inside the graph, so
    they include the loop's few predicated no-op steps; the share is a ratio, the bias is a handful of steps)."""
    if c0 is None or c1 is None:
        return
    hot, cold = c1[0] - c0[0], c1[1] - c0[1]
    res["routed_calls"] = hot + cold
    res["cold_calls"] = cold
    res["cold_share"] = cold / (hot + cold) if hot + cold else None


def run_prompts(loop, prompts: Dict[str, dict], max_new: int, *, reps: int, parity_reps: int, stop_on_eos: bool,
                profile=None, log=print, errors: Optional[list] = None, label: str = "2g") -> Dict[str, dict]:
    """Per prompt: `reps` fast runs then `parity_reps` parity runs -> {tag: {"fast": [...], "parity": [...]}}."""
    out: Dict[str, dict] = {}
    for tag, p in prompts.items():
        try:
            fast, par = [], []
            for i in range(reps):
                prof = None
                if profile and profile[0] == tag and i == 0:
                    prof = (profile[1], profile[2])
                c0 = _routed(loop)
                fast.append(loop.generate(p["ids"], max_new, stop_on_eos=stop_on_eos, parity=False, profile=prof))
                _cold_delta(fast[-1], c0, _routed(loop))
            for _ in range(parity_reps):
                c0 = _routed(loop)
                par.append(loop.generate(p["ids"], max_new, stop_on_eos=stop_on_eos, parity=True, tag=tag,
                                         prompt=p.get("prompt")))
                _cold_delta(par[-1], c0, _routed(loop))
            out[tag] = {"fast": fast, "parity": par}
            r = fast[0] if fast else par[0]
            ms = r.get("step_ms") or []
            log(f"[loop {label}] {tag:6s} {len(r['token_ids'])} tok in {r['steps']} steps "
                f"({r['tokens_per_step'] or 0:.3f} tok/step), {sum(ms) / max(1, len(ms)):.2f} ms/step, prefill "
                f"{r['prefill_ms']:.1f} ms", flush=True)
        except Exception:
            tb = traceback.format_exc()
            if errors is not None:
                errors.append({"stage": f"loop-{label}", "tag": tag, "traceback": tb})
            log(f"[loop {label}] {tag}: FAILED\n{tb}", file=sys.stderr, flush=True)
    return out


def run_cells(loop, prompts: Dict[str, dict], max_new: int, graphs: Sequence[int], *, reps: int, parity_reps: int,
              stop_on_eos: bool, profile=None, rep: Optional[dict] = None, log=print) -> Dict[int, Dict[str, dict]]:
    """For each graph count (0 = eager, the gen's "static" mode): (re)capture, then the prompts. The first cell gets
    `reps` fast + `parity_reps` parity runs per prompt (and the HBM-after-build reading); later cells 1 fast run."""
    import torch

    rep = {} if rep is None else rep
    rep.setdefault("errors", [])
    results: Dict[int, Dict[str, dict]] = {}
    for ng in graphs:
        try:
            if ng:
                rep.setdefault("capture", {})[str(ng)] = loop.capture(n_graphs=ng)
            if ng == graphs[0]:
                rep["hbm_after_build_GiB"] = _hbm(torch)
            first_cell = ng == graphs[0]
            results[ng] = run_prompts(loop, prompts, max_new, reps=max(1, reps) if first_cell else 1,
                                      parity_reps=parity_reps if first_cell else 0, stop_on_eos=stop_on_eos,
                                      profile=profile if first_cell else None, errors=rep["errors"],
                                      label=f"{ng}g" if ng else "eager", log=log)
        except Exception:
            rep["errors"].append({"stage": f"capture-{ng}", "traceback": traceback.format_exc()})
            log(f"[loop {ng}g] FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    return results


def aggregate(rep: dict, prompts: Dict[str, dict], results: Dict[int, Dict[str, dict]], graphs: Sequence[int],
              stats_from: int) -> dict:
    """(a)-(c) per prompt + the pooled headline -> rep["prompts"], rep["streams"], rep["replay_ok"], rep["headline"].
    prompts[tag]: {"ids", "r1_ids", "r1_sha", "r1_rec", ...}; results[n_graphs][tag]: {"fast": [...], "parity": [...]}
    (DSparkLoop.generate results). The first entry of `graphs` is the attributed (2-graph) cell."""
    from runtime.generate import dump_record

    first = results.get(graphs[0], {})
    per: Dict[str, dict] = {}
    n_exact = n_replay = 0
    for tag, p in prompts.items():
        res = first.get(tag)
        if not res:
            continue
        fast, par = res["fast"], res["parity"]
        base = par[0] if par else fast[0]
        ids_ = base["token_ids"]
        s = compare_stream(ids_, p["r1_ids"], base.get("step_sha256"), p.get("r1_sha"), base.get("record"),
                           p.get("r1_rec"))
        replay = all(x["token_ids"] == ids_ and x["accepts"] == base["accepts"] for x in fast + par)
        if len(par) > 1:
            replay = replay and all(dump_record(x["record"]) == dump_record(par[0]["record"]) and
                                    x["logits_sha256"] == par[0]["logits_sha256"] for x in par[1:])
        one = (results.get(1) or {}).get(tag) if graphs[0] != 1 else None
        one_equal = None
        if one:
            one_equal = all(x["token_ids"] == ids_ and x["accepts"] == base["accepts"] for x in one["fast"])
            replay = replay and one_equal
        n_exact += bool(s["exact"])
        n_replay += bool(replay)
        acc = base["accepts"]
        per[tag] = {"prompt_tokens": len(p["ids"]), "tokens": len(ids_), "steps": base["steps"],
                    "tokens_sha256": hashlib.sha256(json.dumps([int(t) for t in ids_]).encode()).hexdigest(),
                    "cold_share_by_rep": [x.get("cold_share") for x in fast],
                    "tokens_per_step": base["tokens_per_step"], "accept_hist": [acc.count(i) for i in range(6)],
                    "accepts": acc, "stream": s, "replay_ok": replay, "one_graph_equal": one_equal,
                    "prefill_ms": [x["prefill_ms"] for x in fast], "r1_ms_per_token": p.get("r1_ms_per_token"),
                    "r1_vs_ref": p.get("r1_vs_ref"), "finish_reason": base["finish_reason"],
                    "window": window(fast, stats_from),
                    "window_one_graph": window(one["fast"], stats_from) if one else None,
                    "cold_share_by_rep_one_graph": [x.get("cold_share") for x in one["fast"]] if one else None,
                    "steps_issued": [x["steps_issued"] for x in fast],
                    "wall_ms_fast": [x["wall_ms"] for x in fast]}
    rep["prompts"] = per
    rep["n_prompts"] = len(per)
    rep["streams"] = {"exact": n_exact, "prompts": len(per)}
    rep["replay_ok"] = n_replay
    head = {"metric": METRIC, "instrument": "CUDA events per step (start | after verify graph | after draft graph), "
                                            f"live steps >= {stats_from} of each sequence, pooled over prompts"}
    for ng, key in ((2, "two_graphs"), (1, "one_graph"), (0, "eager")):
        rr = results.get(ng)
        if rr:
            head[key] = window([x for v in rr.values() for x in v["fast"]], stats_from)
            head[key]["cold_share"] = cold_pooled([x for v in rr.values() for x in v["fast"]])
    w2 = head.get("two_graphs") or head.get("one_graph") or head.get("eager") or {}
    head["ms_per_6row_step_dspark_k5"] = (w2.get("ms_per_step") or {}).get("mean")
    head["ms_verify_graph"] = (w2.get("ms_verify_graph") or {}).get("mean") if w2.get("ms_verify_graph") else None
    head["ms_draft_graph"] = (w2.get("ms_draft_graph") or {}).get("mean") if w2.get("ms_draft_graph") else None
    head["tokens_per_step"] = w2.get("tokens_per_step")
    head["ms_per_token"] = w2.get("ms_per_token")
    head["total_gpu_ms"] = w2.get("total_gpu_ms")                  # LOOSE-ENDS-2: the two totals behind ms_per_token
    head["total_emitted_tokens"] = w2.get("total_emitted_tokens")
    head["ms_per_token_definition"] = MS_PER_TOKEN_DEF
    if head.get("one_graph"):
        head["ms_per_6row_step_one_graph"] = (head["one_graph"].get("ms_per_step") or {}).get("mean")
    pf = [x for v in per.values() for x in v["prefill_ms"]]
    head["prefill_ms_mean"] = sum(pf) / len(pf) if pf else None
    rep["headline"] = head
    return rep


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def residency_report(gen, a, rep: dict, results: Dict[int, Dict[str, dict]], graphs: Sequence[int]) -> dict:
    """A2-2 (docs/A2-2-BOX-PLAN.md): the manager's budget / HBM / stats / policy telemetry, swaps per step (mean, max
    vs the cap), the cold-share curve over the whole run -- per drain (the policy's cold picks / routed picks since
    the previous drain) and per sequence in run order (the step's cold-call counters, `cold_share_by_seq`) -- the
    manager's GPU ms per step, and with --residency-ref <the off run's headline JSON> per prompt: emitted tokens
    sha256 and accept lengths equal, plus the ms/step delta. Call before loop.close() (close restores the boot map)."""
    from runtime.residency import config_from_args

    rm = getattr(gen, "residency_mgr", None)
    out: Dict = {"config": config_from_args(a), "on": rm is not None}
    if rm is None:
        return out
    try:
        rm.flush()
        d = rm.describe()
        out.update({k: v for k, v in d.items() if k != "drain_log"})
        out["drain_log"] = d["drain_log"]
        out["cold_share_curve"] = [[r["host_steps"], r["cold_share"]] for r in d["drain_log"]]
        cs = [r["cold_share"] for r in d["drain_log"] if r["cold_share"] is not None]
        out["cold_share_first_drain"] = cs[0] if cs else None
        out["cold_share_last4_drains"] = _mean(cs[-4:])
        seq = []
        for tag, res in (results.get(graphs[0]) or {}).items():      # chronological: fast reps, then parity reps
            for kind in ("fast", "parity"):
                for i, x in enumerate(res.get(kind) or []):
                    seq.append({"tag": tag, "kind": kind, "i": i, "steps": x.get("steps"),
                                "cold_share": x.get("cold_share"), "routed_calls": x.get("routed_calls")})
        out["cold_share_by_seq"] = seq
        h = rep.get("headline") or {}
        w = h.get("two_graphs") or h.get("one_graph") or h.get("eager") or {}
        out["ms_manager_per_step"] = (w.get("ms_manager_per_step") or {}).get("mean")
        out["ms_per_step_incl_manager"] = w.get("ms_per_step_incl_manager")
        out["overflow"] = rep.get("overflow")
        if a.residency_ref:
            with open(a.residency_ref) as f:
                ref = json.load(f)
            vs: Dict = {"ref": a.residency_ref, "prompts": {}}
            for tag, p in (rep.get("prompts") or {}).items():
                rp = (ref.get("prompts") or {}).get(tag)
                if not rp:
                    continue
                vs["prompts"][tag] = {"accepts_equal": rp.get("accepts") == p.get("accepts"),
                                      "tokens_equal": (rp.get("tokens_sha256") == p.get("tokens_sha256")
                                                       if rp.get("tokens_sha256") else None)}
            vs["all_accepts_equal"] = all(v["accepts_equal"] for v in vs["prompts"].values()) and bool(vs["prompts"])
            vs["all_tokens_equal"] = (all(v["tokens_equal"] for v in vs["prompts"].values())
                                      if vs["prompts"] and all(v["tokens_equal"] is not None
                                                               for v in vs["prompts"].values()) else None)
            rh = ref.get("headline") or {}
            rw = rh.get("two_graphs") or rh.get("one_graph") or rh.get("eager") or {}
            m0, m1 = (rw.get("ms_per_step") or {}).get("mean"), (w.get("ms_per_step") or {}).get("mean")
            if m0 and m1:
                vs["step_ms_off"], vs["step_ms_on"] = m0, m1
                vs["step_ms_delta_pct"] = (m1 - m0) / m0 * 100
                if w.get("ms_per_step_incl_manager"):
                    vs["step_ms_on_incl_manager"] = w["ms_per_step_incl_manager"]
                    vs["step_ms_delta_incl_manager_pct"] = (w["ms_per_step_incl_manager"] - m0) / m0 * 100
            out["vs_off"] = vs
            if vs["all_accepts_equal"] is False or vs["all_tokens_equal"] is False:
                rep["errors"].append({"stage": "residency_vs_off", "traceback": f"residency on != off: {vs['prompts']}"})
        if a.residency_audit:
            from runtime.residency import golden_from_index

            out["audit"] = rm.audit(golden=golden_from_index(gen.model.runtime_load.index))
            if not out["audit"]["ok"]:
                rep["errors"].append({"stage": "residency_audit", "traceback": str(out["audit"]["errors"][:8])})
    except Exception:
        tb = traceback.format_exc()
        print(f"[residency] report FAILED\n{tb}", file=sys.stderr, flush=True)
        rep["errors"].append({"stage": "residency_report", "traceback": tb})
    return out


def residency_line(rr: dict) -> str:
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    sp = rr.get("swaps_per_step") or {}
    vo = rr.get("vs_off") or {}
    hb = rr.get("hbm") or {}
    return (f"  residency: {sp.get('swaps', 0)} swaps ({f(sp.get('mean'), 3)}/step, max {sp.get('max')}, cap "
            f"{sp.get('cap')}); cold share drain1 {f(rr.get('cold_share_first_drain'))} -> last4 "
            f"{f(rr.get('cold_share_last4_drains'))}; manager {f(rr.get('ms_manager_per_step'), 3)} ms/step; "
            f"+{hb.get('spare_GiB')} GiB HBM, n_spare {rr.get('n_spare')} (stage {rr.get('n_stage')}, overflow steps "
            f"{(rr.get('overflow') or {}).get('steps', '-')}); vs off: accepts equal "
            f"{vo.get('all_accepts_equal', '-')}, tokens equal {vo.get('all_tokens_equal', '-')}, ms/step "
            f"{f(vo.get('step_ms_delta_pct'), 2)}%")[:400]


def _begin_receipt(a, impls: dict, map_path: Optional[str], argv) -> dict:
    from runtime.residency import config_from_args

    try:
        res = config_from_args(a)
    except Exception:  # noqa: BLE001
        res = None
    return _receipt.begin(argv=argv, impls=impls, model_dir=a.model_dir, fixture=a.fixture,
                          residency=res, map_path=map_path, config_path=a.config)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not a.no_gemv_pin:
        os.environ["DSV41_GEMV_PIN_M1"] = "1"     # before any GEMV get_config (builds, warm-ups, captures)
    os.environ["DSV41_FP32_IMPL"] = a.fp32_impl   # T9b-5 (= runtime.fused_glue_r2.FP32_IMPL_ENV), before any build
    from runtime.gemv_mxfp8 import apply_fp8_impl_arg

    fp8_arg = a.fp8_impl                          # SMALL-WINS: "gemv3" = gemv + $DSV41_GEMV_IMPL=gemv3, before any build
    a.fp8_impl = apply_fp8_impl_arg(fp8_arg)
    os.environ["DSV41_MOE_ROWMATCH"] = a.moe_rowmatch   # T9a-3 (= runtime.dg_inplace.ROWMATCH_ENV), before any build
    import torch

    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load
    from runtime.residency import config_from_args

    date = time.strftime("%Y%m%d")
    json_path = a.json or os.path.join(REPO, "docs", f"headline-{date}.json")
    graphs = [int(x) for x in a.graphs.split(",") if x.strip()]
    impls = {"moe": a.moe_impl, "fp8": fp8_arg, "glue": a.glue_impl, "attn": a.attn_impl, "fp32": a.fp32_impl,
             "moe_rowmatch": a.moe_rowmatch}
    rep: Dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
                 "metric": METRIC, "mode": a.mode, "impls": impls, "max_new": a.max_new,
                 "gemv_pin_m1": os.environ.get("DSV41_GEMV_PIN_M1") == "1", "errors": [], "json": json_path,
                 "pg61_ms_per_step": PG61_MS_PER_STEP, "bar_ms_per_step": CHARTER_BAR_MS}
    t0 = time.time()
    mp = a.map if a.map and os.path.exists(a.map) else None
    rep["receipt"] = _begin_receipt(a, impls, mp, rep["argv"])      # LOOSE-ENDS-2: what produced this number (never raises)
    model = build_and_load(a.model_dir, mp, "cuda", hot_per_layer=a.hot_per_layer, config_path=a.config,
                           reserve_host_gib=a.reserve_host_gib, load_mtp=True)
    rep["boot"] = model.runtime_load.summary()
    rep["boot"]["wall_s"] = time.time() - t0
    rep["mtp"] = model.runtime_mtp.summary()
    rep["determinism"] = set_determinism(0)
    gen = Generator(model, a.tokenizer_dir or a.model_dir, mode=a.mode, max_pos=a.max_pos, moe_impl=a.moe_impl,
                    fp8_impl=a.fp8_impl, glue_impl=a.glue_impl, attn_impl=a.attn_impl, spare_slots=a.spare_slots)
    if a.moe_impl == "dg-inplace" and getattr(gen, "pool", None) is not None:
        from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS

        n = a.spare_slots or DEFAULT_SPARE_SLOTS_ROWS   # grow ONCE before any build (bench/draft_bench.py note)
        if a.pool_spare is not None:
            n = max(n, a.pool_spare)
        elif config_from_args(a) is not None:
            n = max(n, 20)   # 11/07 arm C: 16 staging + 4 residency leaves HBM for 17 layers of pooled SF (19.06 ms/step)
        if config_from_args(a) is not None:
            # A2-2 FIX: the staging keeps `stage` of the pool's slots (--residency-stage, else --spare-slots, else 8)
            # and residency promotes into the REST: the default pool (16) already holds 8 -- no new allocation. Only
            # an explicit --residency-spare larger than that grows the pool (here, once, before any build: a
            # re-growth moves every layer's weight block; FIX2's box OOM was one).
            from runtime.residency import ResidencyConfig, residency_stage

            topk = int(model.layers[0].ffn.n_activated_experts)
            stage = residency_stage(ResidencyConfig(stage_keep=a.residency_stage), "dg-inplace", 6, topk, a.spare_slots)
            n = max(n, DEFAULT_SPARE_SLOTS_ROWS, stage + (a.residency_spare or 0))
        b = gen.pool.add_spare_slots(n, model)
        print(f"[pool] pre-grown to {gen.pool.spare} spare slots/layer (+{b / 2**30:.2f} GiB)", flush=True)
    with open(a.fixture) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    stop = not a.no_stop_on_eos
    r1_ref = {}
    if a.r1_ref and os.path.exists(a.r1_ref):
        with open(a.r1_ref) as f:
            r1_ref = {r["tag"]: r for r in (json.loads(l) for l in f if l.strip())}

    # 1. truth at 1 row
    prompts: Dict[str, dict] = {}
    for r in rows:
        try:
            ids = gen.encode([{"role": "user", "content": r["prompt"]}])
            rec = gen.greedy(ids, a.max_new, tag=r["tag"], prompt=r["prompt"], stop_on_eos=stop)
            m = gen.last_meta
            prompts[r["tag"]] = {"ids": ids, "prompt": r["prompt"], "r1_ids": list(m["token_ids"]),
                                 "r1_sha": list(m["step_sha256"]), "r1_rec": rec,
                                 "r1_ms_per_token": m.get("decode_ms_per_token")}
            if r["tag"] in r1_ref:
                a_ = [e["token"] for e in rec["logprobs"]]
                b_ = [e["token"] for e in r1_ref[r["tag"]].get("logprobs", [])]
                k_ = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y),
                          None if len(a_) == len(b_) else min(len(a_), len(b_)))
                prompts[r["tag"]]["r1_vs_ref"] = {"equal": k_ is None, "first_divergence": k_, "ref": a.r1_ref}
            print(f"[r1] {r['tag']:6s} prompt {len(ids)} -> {len(m['token_ids'])} tokens, "
                  f"{m.get('decode_ms_per_token') or 0:.2f} ms/token (1-row)", flush=True)
        except Exception:
            rep["errors"].append({"stage": "r1", "tag": r["tag"], "traceback": traceback.format_exc()})
            print(f"[r1] {r['tag']}: FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)

    # 2. the loop, 2 graphs (then 1)
    rep["rowmatch"] = rowmatch_record(getattr(gen, "static", None))   # T9a-3: before the loop resets the 1-row step
    profile = None
    if a.profile:
        t_, f_, n_ = a.profile.split(":")
        profile = (t_, int(f_), int(n_))
    res_cfg = config_from_args(a)
    if res_cfg is not None:
        # A2-2: residency covers the loop's 6-row verify step only; the 1-row truth above ran on the static map. The
        # manager grows the pool by its spare slots on top of the pre-grown staging slots (one more growth, before
        # the verify step's tables exist -- DSparkLoop resets the static step first), then attaches to the verify step.
        gen.set_residency(res_cfg)
    rep["residency_config"] = res_cfg
    loop = DSparkLoop(gen, a.max_new, poll_every=a.poll_every, draft_impls=None if a.draft_impls == "same" else "ref")
    rep["loop"] = loop.describe()
    orders_compare(rep["rowmatch"], loop.st)
    print(f"[rowmatch] {rowmatch_line(rep['rowmatch'])}", flush=True)
    _sw_rows = sorted({1, int(getattr(loop.st, "rows", 6) or 6)})   # SMALL-WINS: the 1-row truth and the loop's verify step
    _gk = None if a.fp8_impl != "gemv" else __import__("runtime.gemv_mxfp8", fromlist=["gemv_impl"]).gemv_impl()
    from runtime.fused_glue_r2 import resolved_choices, resolved_line

    rep["small_wins"] = dict(resolved_choices(_sw_rows), fp8_impl_arg=fp8_arg, gemv_kernel=_gk,
                             mixes_step=(loop.st.mix_fork.describe() if hasattr(loop.st, "mix_fork") else None))
    print(resolved_line(_sw_rows, _gk), flush=True)
    run_graphs = graphs if a.mode == "graph" else [0]
    results = run_cells(loop, prompts, a.max_new, run_graphs, reps=a.reps, parity_reps=a.parity_reps,
                        stop_on_eos=stop, profile=profile, rep=rep)
    ovf = loop.st.overflow_counts()
    rep["overflow"] = ovf
    rep["routed"] = None
    try:
        rc = loop.st.routed_counts()
        if rc is not None:
            rep["routed"] = {"hot_calls": sum(rc[0]), "cold_calls": sum(rc[1])}
    except Exception:
        pass

    # 3. compare + aggregate
    aggregate(rep, prompts, results, run_graphs, a.stats_from)
    if res_cfg is not None:
        rep["residency"] = residency_report(gen, a, rep, results, run_graphs)   # before loop.close() restores the map
    rep["mode_gap"] = mode_gap(rep)             # LOOSE-1107: one_graph vs two_graphs, and why they differ if they do
    try:
        from draft_bench import pg61_tokens_per_chunk

        rep["pg61"] = pg61_tokens_per_chunk(a.pg61_step)
    except Exception:
        rep["pg61"] = None
    rep["hbm_end_GiB"] = _hbm(torch)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    _receipt.finish(rep.get("receipt"))
    slim = json.loads(json.dumps(rep, default=str))
    with open(json_path, "w") as f:
        json.dump(slim, f, indent=1)
    for line in summary_lines(rep):
        print(line)
    print(_receipt.summary_line(rep.get("receipt")))
    if rep.get("residency", {}).get("on"):
        print(residency_line(rep["residency"]))
    loop.close()
    return 0 if not rep["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
