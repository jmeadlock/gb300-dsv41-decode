#!/usr/bin/env python3
"""gap_ledger.py -- the matched-workload gap ledger: our 6-row DSpark step vs production (Pg61), column by column.
docs/HANDOFF-GAP-LEDGER.md (Astra review 2026-11-07, accepted); protocol docs/GAP-LEDGER-PLAN.md.

Every ledger cell is {"value", "kind", "how"}; kind is one of
  measured     -- read from a receipt (CUDA events, /metrics, SSE arrival times, a graph census, a sidecar counter);
  derived      -- arithmetic over measured cells; `how` carries the arithmetic WITH the numbers;
  static       -- from an nsys profile of the eager static path [static, diagnostic]: kernel durations launched
                  eagerly, NOT an additive accounting of the captured graph (the graph's own time is the event cell);
  not measured -- `how` says why and what would measure it.
No other kind exists: a number with no receipt is not written.

Subcommands
  build              (Mac or box; stdlib only) inputs -> scratch/gap-ledger-ours.json + scratch/gap-ledger-ours.md
      [--headline docs/headline-20261107j.json] [--tf scratch/gap-tf-graph.json] [--anatomy tf-anatomy.json]
      [--loop-anatomy loop-anatomy.json] [--pg61-step w0/results/phase1/step-pg61.json]
      [--pg61-fixture w0/results/phase1/parity-A.jsonl] [--pg61-drive scratch/pg61-drive.json]
      [--peer3 docs/data/peer3_stats.json] [--out scratch/gap-ledger-ours.json] [--md scratch/gap-ledger-ours.md]
  ours-tf            (box, in the runtime container) the 6-row step teacher-forced to Pg61's OWN emitted tokens
                     (parity-A.jsonl token bytes -> ids via the checkpoint's tokenizer.json), stride 6 (every row a
                     Pg61 token, every row accepted), CUDA events around each step; --census: the captured 6-row
                     graph's node/edge census and, with --census-loop, the DSpark loop's verify/draft graphs (A, B, AB);
                     --mode static --profile W:N --nvtx-layers: the same cell under nsys (cudaProfilerApi range).
  ours-loop-profile  (box, nsys) the headline loop in static mode with NVTX "decode_step"/"verify"/"draft" ranges, so
                     step_anatomy.py splits the kernels by phase (draft families). Own stream, not matched prefixes.
  pg61-drive         (box HOST, stdlib, needs Pg61 bound on :30006 -- James's word) the fixture prompts streamed:
                     a logprobs pass (per-chunk token counts = per-step emissions, token bytes vs parity-A) and
                     --reps timing passes without logprobs (SSE arrivals + /metrics spec counters), w0_drive's bodies.

Rules: never import vllm; GPU subcommands import torch lazily (the build path and the tests are stdlib).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step_anatomy import FAMILIES, family_of  # noqa: E402  (stdlib module; the family map lives there)

EXPERT_BYTES = 18_800_640          # one routed expert, all three projections + scales (A2-RESIDENCY-DESIGN.md 0.1)
N_LAYERS = 40
KINDS = ("measured", "derived", "static", "not measured")
TAG = {"measured": "M", "derived": "D", "static": "S", "not measured": "-"}
CUPTI_DEADLOCK = ("Pg61 cannot run under nsys/CUPTI: the engine stalls after 1-2 requests in the peer-tier-v3 spin "
                  "(w0/W0-RESULTS.md phases 2 and 2b, both trace modes); reopen only with a non-CUPTI instrument "
                  "(nsys --gpu-metrics-devices sampling) or a peer3-off arm, which is not Pg61")


# ------------------------------------------------------------------------------------------------ cells
def cell(value, kind: str, how: str) -> dict:
    if kind not in KINDS:
        raise ValueError(f"cell kind {kind!r} not in {KINDS}")
    if kind == "not measured":
        value = None
    elif value is None:
        raise ValueError(f"a {kind} cell needs a value ({how})")
    return {"value": value, "kind": kind, "how": how}


def nm(why: str) -> dict:
    return cell(None, "not measured", why)


def _f(x, d=3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def _ratio(num: float, den: float, what: str, unit_mul: float = 1.0, d: int = 3) -> Tuple[Optional[float], str]:
    if not den:
        return None, f"{what}: denominator 0"
    v = unit_mul * num / den
    mul = "" if unit_mul == 1.0 else f"{unit_mul:g} x "
    return v, f"{what} = {mul}{num:.{d}f} / {den:.{d}f} = {v:.{d}f}"


# ------------------------------------------------------------------------------------------------ Pg61 side
def pg61_step_cell(step: dict, src: str) -> Dict[str, dict]:
    """w0/results/phase1/step-pg61.json (w0_drive.py step: 3 x 512-token greedy on STEP_PROMPT, thinking off; steps =
    /metrics spec_decode_num_drafts delta, decode wall = first..last content-chunk arrival). NOT an nsys profile:
    no kernel attribution exists for Pg61 (CUPTI_DEADLOCK)."""
    reps = [r for r in step.get("reps") or [] if r.get("steps") and r.get("decode_wall_s")]
    out: Dict[str, dict] = {"condition": cell(
        f"STEP_PROMPT (66 prompt tokens), 512 greedy tokens x {len(reps)} reps, thinking off; not the fixture",
        "measured", f"{src}: usage.prompt_tokens, max 512, w0/w0_drive.py STEP_PROMPT")}
    if not reps:
        return out
    dw = sum(r["decode_wall_s"] for r in reps)
    st1 = sum(r["steps"] - 1 for r in reps)
    stp = sum(r["steps"] for r in reps)
    gen = sum(r["gen_tokens"] for r in reps)
    wall = sum(r["wall_s"] for r in reps)
    v, how = _ratio(dw, st1, "sum decode_wall_s / sum (steps - 1)", 1000.0)
    out["step_gpu_ms"] = cell(v, "derived", f"wall-clock, not GPU: {how} ms over reps {[r['rep'] for r in reps]} "
                                            f"(SSE arrivals / metrics drafts; per rep {[r.get('step_ms_from_metrics') for r in reps]})")
    v, how = _ratio(gen, stp, "sum gen_tokens / sum steps")
    out["tokens_per_step"] = cell(v, "derived", how)
    v, how = _ratio(dw, gen - len(reps), "sum decode_wall_s / sum (gen_tokens - 1)", 1000.0)
    out["wall_ms_per_token_decode"] = cell(v, "derived", how + " ms (the first chunk = the prefill's token)")
    v, how = _ratio(wall, gen, "sum wall_s / sum gen_tokens", 1000.0)
    out["wall_ms_per_token_e2e"] = cell(v, "derived", how + " ms (incl. TTFT + HTTP)")
    out["verify_gpu_ms"] = nm("Pg61's /metrics have no verify/draft split; " + CUPTI_DEADLOCK)
    out["draft_gpu_ms"] = nm("as verify: " + CUPTI_DEADLOCK)
    out["gpu_ms_per_token"] = nm("no GPU timer on Pg61 (only SSE wall + /metrics)")
    return out


def pg61_fixture(rows: Sequence[dict], src: str) -> Dict[str, dict]:
    """parity-A.jsonl: Pg61's greedy on the 8 fixture prompts (non-streamed, top-20 logprobs, 256 max) -- the matched
    prompts and the very tokens the teacher-forced cell feeds. Only end-to-end wall per prompt was recorded."""
    rows = [r for r in rows if r.get("n_tokens")]
    out: Dict[str, dict] = {}
    if not rows:
        return out
    wall = sum(r["wall_s"] for r in rows)
    tok = sum(r["n_tokens"] for r in rows)
    ptok = [r.get("usage", {}).get("prompt_tokens") for r in rows]
    out["condition"] = cell(f"fixture {len(rows)} prompts ({','.join(r['tag'] for r in rows)}), prompt tokens {ptok}, "
                            f"{tok} emitted, greedy T=0, top-20 logprobs, non-streamed", "measured", src)
    v, how = _ratio(wall, tok, "sum wall_s / sum n_tokens", 1000.0)
    out["wall_ms_per_token_e2e"] = cell(v, "derived", how + " ms (non-streamed: incl. prefill, HTTP and top-20 "
                                                           "logprob work; one request each)")
    for k in ("step_gpu_ms", "tokens_per_step", "wall_ms_per_token_decode"):
        out[k] = nm("parity-A has no step count or arrival times -> the matched re-run (pg61-drive)")
    return out


def pg61_drive(drive: dict, src: str) -> Dict[str, dict]:
    """scratch/pg61-drive.json (this file's pg61-drive): per prompt a logprobs pass and timing passes."""
    out: Dict[str, dict] = {}
    pr = drive.get("prompts") or {}
    reps = [r for p in pr.values() for r in p.get("time") or [] if r.get("steps") and r.get("decode_wall_s")]
    eq = {t: (p.get("count") or {}).get("equal_to_parity_a") for t, p in pr.items()}
    out["condition"] = cell(f"fixture {len(pr)} prompts, streamed, {len(reps)} timing passes; Pg61 stream == "
                            f"parity-A per prompt: {eq}", "measured", src)
    if not reps:
        return out
    dw = sum(r["decode_wall_s"] for r in reps)
    st1 = sum(r["steps"] - 1 for r in reps)
    stp = sum(r["steps"] for r in reps)
    gen = sum(r["gen_tokens"] for r in reps)
    first = sum(r.get("first_chunk_tokens") or 1 for r in reps)
    wall = sum(r["wall_s"] for r in reps)
    v, how = _ratio(dw, st1, "sum decode_wall_s / sum (steps - 1)", 1000.0)
    out["step_gpu_ms"] = cell(v, "derived", "wall-clock, not GPU: " + how + " ms")
    v, how = _ratio(gen, stp, "sum gen_tokens / sum steps")
    out["tokens_per_step"] = cell(v, "derived", how)
    v, how = _ratio(dw, gen - first, "sum decode_wall_s / sum (gen_tokens - first_chunk_tokens)", 1000.0)
    out["wall_ms_per_token_decode"] = cell(v, "derived", how + " ms")
    v, how = _ratio(wall, gen, "sum wall_s / sum gen_tokens", 1000.0)
    out["wall_ms_per_token_e2e"] = cell(v, "derived", how + " ms (streamed, incl. TTFT)")
    counts = [c for p in pr.values() for c in (p.get("count") or {}).get("step_tokens") or []]
    if counts:
        out["step_emissions"] = cell({str(k): counts.count(k) for k in sorted(set(counts))}, "measured",
                                     "histogram of tokens per SSE chunk in the logprobs pass (= accepted + 1 per "
                                     "engine step, the first chunk being the prefill's token)")
    return out


def pg61_peer3(p3: dict, src: str) -> Dict[str, dict]:
    """docs/data/peer3_stats.json (Pg61's sidecar, peer tier v3; A2-RESIDENCY-DESIGN.md 1.3 field table)."""
    t6 = (p3.get("by_T") or {}).get("6")
    if not t6:
        return {}
    out = {"cold_experts_per_layer_call": cell(t6["cold_exp_mean"], "measured",
                                               f"{src} by_T['6'].cold_exp_mean (distinct cold experts per 6-row "
                                               f"layer-call, {t6['calls']} calls)")}
    w, how = t6["gb_wait_us_mean"] * N_LAYERS / 1000.0, f"{t6['gb_wait_us_mean']} us x {N_LAYERS} layers / 1000"
    out["cold_exposed_ms"] = cell(w, "derived", f"GB300's exposed wait for the RTX's cold-expert work per 6-row step = "
                                                f"{how} = {w:.3f} ms (by_T['6'].gb_wait_us_mean; RTX busy "
                                                f"{t6['busy_us_mean']} us per cold call runs concurrently)")
    return out


# ------------------------------------------------------------------------------------------------ our side
def ours_headline(h: dict, src: str) -> Dict[str, dict]:
    """bench/headline.py JSON: the free-running loop (own stream), two graphs -> verify / draft split."""
    hd = h.get("headline") or {}
    w = hd.get("two_graphs") or {}
    out: Dict[str, dict] = {}
    impls = h.get("impls")
    out["condition"] = cell(f"fixture {h.get('n_prompts')} prompts, own greedy stream (DSpark k=5 loop), max_new "
                            f"{h.get('max_new')}, impls {impls}, map {os.path.basename(str((h.get('boot') or {}).get('map')))}"
                            f", spare 16, residency {(h.get('residency') or {}).get('on', 'off')}", "measured",
                            f"{src} argv {h.get('argv')}")
    sf = w.get("stats_from_step")
    win = f"live steps >= {sf}, {w.get('steps')} steps pooled over prompts x reps"
    for key, k2 in (("verify_gpu_ms", "ms_verify_graph"), ("draft_gpu_ms", "ms_draft_graph"), ("step_gpu_ms", "ms_per_step")):
        m = (w.get(k2) or {}).get("mean")
        out[key] = cell(m, "measured", f"{src} headline.two_graphs.{k2}.mean: CUDA events per step ({win})") \
            if m is not None else nm(f"{src}: no two_graphs.{k2}")
    if w.get("tokens_per_step") is not None:
        out["tokens_per_step"] = cell(w["tokens_per_step"], "measured", f"{src} two_graphs.tokens_per_step "
                                                                        f"(sum tokens / sum steps, {win})")
    if w.get("ms_per_token") is not None:
        out["gpu_ms_per_token"] = cell(w["ms_per_token"], "measured", f"{src} two_graphs.ms_per_token = sum step ms / sum "
                                                                      f"tokens ({win}); all steps: "
                                                                      f"{_f(w.get('all_ms_per_token'))}")
    pr = h.get("prompts") or {}
    wall = sum(sum(p.get("wall_ms_fast") or []) for p in pr.values())
    pre = sum(sum(p.get("prefill_ms") or []) for p in pr.values())
    tok = sum(p.get("tokens", 0) * len(p.get("wall_ms_fast") or []) for p in pr.values())
    if tok:
        v, how = _ratio(wall, tok, "sum wall_ms_fast / sum (tokens x fast reps)")
        out["wall_ms_per_token_e2e"] = cell(v, "derived", how + " ms (host wall incl. the reference prefill, "
                                                                f"{pre / max(1, sum(len(p.get('prefill_ms') or []) for p in pr.values())):.0f} ms/prompt)")
        v, how = _ratio(wall - pre, tok, "sum (wall_ms_fast - prefill_ms) / sum (tokens x fast reps)")
        out["wall_ms_per_token_decode"] = cell(v, "derived", how + " ms (host wall of the decode loop, all steps incl. "
                                                                   "the async poll)")
    cs = w.get("cold_share")
    if cs is not None:
        out["cold_share"] = cell(cs, "measured", f"{src} two_graphs.cold_share (cold / routed unique-expert calls)")
    return out


def ours_tf(tf: dict, src: str) -> Dict[str, dict]:
    """ours-tf JSON: the 6-row step teacher-forced on Pg61's tokens (verify-shaped body; no draft, no accept)."""
    out: Dict[str, dict] = {}
    cells = [c for rep in tf.get("reps") or [] for c in rep.get("prompts") or [] if c.get("step_ms")]
    tok_ok = tf.get("targets") or {}
    out["condition"] = cell(f"fixture prompts teacher-forced to Pg61's parity-A tokens (stride 6, every row a Pg61 "
                            f"token), mode {tf.get('mode')}, impls {tf.get('impls')}, map "
                            f"{os.path.basename(str(tf.get('map')))}, spare {tf.get('spare_slots')}, residency off; "
                            f"token ids ok {sum(1 for v in tok_ok.values() if v.get('ok'))}/{len(tok_ok)}",
                            "measured", src)
    if not cells:
        return out
    ms = [x for c in cells for x in c["step_ms"]]
    if tf.get("mode") == "graph":
        v, how = _ratio(sum(ms), len(ms), "sum step ms / steps")
        out["verify_gpu_ms"] = cell(v, "measured", f"{src}: CUDA events around gen.step_rows (graph replay of the "
                                                   f"6-row step = the T10 body the loop's verify graph wraps, without its accept/commit kernels), "
                                                   f"{how}; {len(cells)} prompt cells")
    else:
        out["verify_gpu_ms"] = nm(f"{src} ran mode {tf.get('mode')} (eager: host-bound wall, not the graph); the "
                                  f"graph-mode ours-tf run gives this cell")
    out["draft_gpu_ms"] = nm("the teacher-forced cell has no draft (no acceptance); draft GPU ms is the headline's")
    out["tokens_per_step"] = nm("teacher-forced: 6 positions per step by construction, not a yield")
    hot = sum(c.get("hot_unique", 0) for c in cells if c.get("hot_unique") is not None)
    cold = sum(c.get("cold_unique", 0) for c in cells if c.get("cold_unique") is not None)
    nstep = sum(len(c["step_ms"]) for c in cells if c.get("cold_unique") is not None)
    if nstep:
        b = cold * EXPERT_BYTES / nstep
        out["expert_bytes_c2c"] = cell(b, "derived", f"cold unique expert calls {cold} x {EXPERT_BYTES} B / {nstep} "
                                                     f"steps = {b / 1e6:.1f} MB/step (read by t9a2/m1 staging kernels "
                                                     f"over UVA, stream-ordered before the expert GEMM: no overlap "
                                                     f"with it by construction)")
        b = hot * EXPERT_BYTES / nstep
        out["expert_bytes_hbm"] = cell(b, "derived", f"hot unique expert calls {hot} x {EXPERT_BYTES} B / {nstep} "
                                                     f"steps = {b / 1e9:.2f} GB/step (read in place from the HBM pool)")
        out["cold_share"] = cell(cold / (hot + cold) if hot + cold else 0.0, "derived",
                                 f"cold / (hot + cold) unique calls = {cold} / {hot + cold}")
    g = tf.get("census") or {}
    for key, gc in (g.get("tf_graphs") or {}).items():
        _census_cells(out, f"tf6[{key}]", gc, src)
    for key, gc in (g.get("loop_graphs") or {}).items():
        _census_cells(out, f"loop[{key}]", gc, src)
    return out


def _census_cells(out: dict, label: str, gc: dict, src: str) -> None:
    if gc.get("error"):
        out[f"graph_nodes {label}"] = nm(f"{src} census error: {gc['error']}")
        return
    out[f"graph_nodes {label}"] = cell(gc["nodes"], "measured", f"{src}: cudaGraphGetNodes, {gc.get('kernel_nodes')} "
                                                                f"kernel nodes, types {gc.get('node_types')}")
    out[f"graph_edges {label}"] = cell(gc["edges"], "measured", f"{src}: {gc.get('api')}, kinds {gc.get('edge_kinds')}")
    if gc.get("kernel_families"):
        out[f"graph_kernel_families {label}"] = cell(gc["kernel_families"], "measured",
                                                     f"{src}: kernel node names (cuFuncGetName) -> family_of, no NVTX "
                                                     f"roles in a graph (GEMV/top-k unsplit)")


def ours_anatomy(an: dict, src: str, graph_ms: Optional[dict], phase: Optional[str] = None) -> Dict[str, dict]:
    """step_anatomy.py --out JSON (GAP-LEDGER keys) -> per-family [static] cells, sum, busy, overlap, concurrency, and
    unattributed = the graph's event ms - sum of static kernel ms (derived; the caveat travels with it)."""
    out: Dict[str, dict] = {}
    fams = an.get("families")
    tagp = f" phase {phase}" if phase else ""
    if fams is None:
        for f in FAMILIES:
            out[f"fam.{f}"] = nm(f"{src} predates the step_anatomy family keys (top-40 kernels only, no full sum): rerun "
                                 f"bench/step_anatomy.py on the sqlite")
        return out
    if phase:
        fams = (an.get("families_by_phase") or {}).get(phase) or {}
    ns = an.get("n_steps")
    caveat = (f"[static, diagnostic] {src}{tagp}: eager static-path kernel durations over {ns} steps, mapped by "
              f"step_anatomy.family_of (roles {'known' if an.get('roles_known') else 'UNKNOWN'})")
    total = 0.0
    for f in FAMILIES:
        v = fams.get(f)
        if v is None:
            out[f"fam.{f}"] = cell(0.0, "static", caveat + "; no kernel of this family in the profile")
        else:
            out[f"fam.{f}"] = cell(v["ms_per_step"], "static", caveat + f"; {v['launches_per_step']:.1f} launches/step")
            total += v["ms_per_step"]
    out["sum_kernels"] = cell(total, "static", caveat + f"; sum over families = {total:.3f} ms/step")
    if not phase:
        for k, lab in (("busy_ms_per_step", "busy_ms"), ("overlap", "overlap"), ("concurrency", "concurrency")):
            if an.get(k) is not None:
                out[lab] = cell(an[k], "static", f"[static, diagnostic] {src} {k} (busy = union of kernel intervals; "
                                                 f"overlap = busy / step wall; concurrency = sum / busy); the eager "
                                                 f"wall is host-launch-bound, so overlap here is NOT the graph's")
        if an.get("unmapped"):
            out["unmapped_names"] = cell(an["unmapped"], "static", f"{src}: kernel names no family rule matched")
        mc = an.get("memcpy") or {}
        if mc:
            out["memcpy"] = cell(mc, "static", f"[static, diagnostic] {src}: CUPTI memcpy per step by kind "
                                               f"(HtoD from registered Grace = copy-engine C2C)")
    if graph_ms and graph_ms.get("value") is not None:
        g = graph_ms["value"]
        out["unattributed"] = cell(g - total, "derived",
                                   f"graph event ms - sum static kernel ms = {g:.3f} - {total:.3f} = {g - total:.3f} "
                                   f"ms/step ({graph_ms['how'][:80]}...); static durations are diagnostic, not an "
                                   f"additive accounting of the graph: a negative value means the graph ran its "
                                   f"kernels faster than eager, not negative gaps")
    else:
        out["unattributed"] = nm("needs the graph-mode event cell for the same workload")
    return out


# ------------------------------------------------------------------------------------------------ the ledger
METRICS: List[Tuple[str, str]] = [
    ("condition", "workload condition"),
    ("residency", "expert residency condition"),
    ("verify_gpu_ms", "verify GPU ms/step"),
    ("draft_gpu_ms", "draft GPU ms/step"),
    ("step_gpu_ms", "complete step ms (GPU events; Pg61: wall/steps)"),
    ("tokens_per_step", "tokens per step"),
    ("gpu_ms_per_token", "GPU ms per emitted token (sum step ms / sum tokens)"),
    ("wall_ms_per_token_decode", "wall ms per emitted token, decode (total / total)"),
    ("wall_ms_per_token_e2e", "wall ms per emitted token, incl. prefill (total / total)"),
] + [(f"fam.{f}", lab) for f, lab in FAMILIES.items()] + [
    ("sum_kernels", "sum of kernel ms/step"),
    ("unattributed", "unattributed = step - sum kernels (gaps/scheduling)"),
    ("busy_ms", "GPU busy (union) ms/step"),
    ("overlap", "measured overlap = busy / wall"),
    ("concurrency", "sum kernel / busy"),
    ("expert_bytes_hbm", "routed expert bytes from HBM per verify step"),
    ("expert_bytes_c2c", "cold expert bytes over C2C per verify step"),
    ("cold_share", "cold share of routed unique-expert calls"),
    ("cold_experts_per_layer_call", "cold experts per 6-row layer-call"),
    ("cold_exposed_ms", "cold-expert work exposed on GB300 per step"),
    ("graph_nodes", "captured graph nodes"),
]
COLUMNS = [
    ("ours.headline", "ours: headline loop (own stream)"),
    ("ours.matched", "ours: 6-row teacher-forced on Pg61 tokens"),
    ("ours.loop_static", "ours: loop static profile (phase split)"),
    ("pg61.step_cell", "Pg61: W0 step cell (STEP_PROMPT)"),
    ("pg61.fixture", "Pg61: parity-A (fixture, matched tokens)"),
    ("pg61.matched", "Pg61: matched re-run (pg61-drive)"),
    ("pg61.sidecar", "Pg61: peer-tier-v3 sidecar counters"),
]
PENDING = {
    "ours.matched": ("tf", "box command 1: gap_ledger.py ours-tf --mode graph --census-loop (events + graph census on "
                           "Pg61's tokens); command 2: nsys + ours-tf --mode static --profile, then step_anatomy.py"),
    "ours.loop_static": ("loop_anatomy", "box command 3 (optional): nsys + gap_ledger.py ours-loop-profile, then "
                                         "step_anatomy.py (draft families)"),
    "pg61.matched": ("pg61_drive", "box command 4: gap_ledger.py pg61-drive on the box host -- needs Pg61 bound on "
                                   ":30006, i.e. James's word to boot it (lane is DARK by charter)"),
}
RESIDENCY_OURS = ("seed map docs/maps/hot-v26ar-seed-295.json: 295 hot experts/layer in HBM, 89 cold in pinned Grace, "
                  "cold read over C2C (UVA) into 16 spare slots/layer by the staging kernel; residency manager off")
RESIDENCY_PG61 = ("NOT all-HBM: Garner rowmap-mix-v1-h255 (255 hot/layer in GB300 HBM, 129 cold) with the cold experts "
                  "COMPUTED on the RTX PRO 6000 by the peer-tier-v3 sidecar, not read over C2C "
                  "(A2-RESIDENCY-DESIGN.md 0.2, w0/W0-RESULTS.md phase 2); confirm the map from the ds41 policy at the "
                  "re-run. The two conditions cannot be equalised without changing one runtime; the ledger reports "
                  "each side's cold work instead (bytes / exposure rows)")


def build_ledger(*, headline=None, headline_src="", tf=None, tf_src="", anatomy=None, anatomy_src="",
                 loop_anatomy=None, loop_anatomy_src="", pg61_step=None, pg61_step_src="", fixture=None,
                 fixture_src="", drive=None, drive_src="", peer3=None, peer3_src="") -> dict:
    cols: Dict[str, Dict[str, dict]] = {c: {} for c, _ in COLUMNS}
    if headline:
        cols["ours.headline"].update(ours_headline(headline, headline_src))
        cols["ours.headline"]["residency"] = cell(RESIDENCY_OURS, "measured", f"{headline_src} boot.map, --spare-slots")
    if tf:
        cols["ours.matched"].update(ours_tf(tf, tf_src))
        cols["ours.matched"]["residency"] = cell(RESIDENCY_OURS, "measured", f"{tf_src} map, spare_slots")
    if anatomy:
        cols["ours.matched"].update(ours_anatomy(anatomy, anatomy_src, cols["ours.matched"].get("verify_gpu_ms")))
    if loop_anatomy:
        lc = cols["ours.loop_static"]
        lc["condition"] = cell("headline loop, static mode (eager), own stream, NVTX verify/draft phases", "measured",
                               loop_anatomy_src)
        for ph in ("verify", "draft"):
            sub = ours_anatomy(loop_anatomy, loop_anatomy_src, cols["ours.headline"].get(f"{ph}_gpu_ms"), phase=ph)
            for k, v in sub.items():
                lc[f"{k}@{ph}"] = v
    if pg61_step:
        cols["pg61.step_cell"].update(pg61_step_cell(pg61_step, pg61_step_src))
    if fixture:
        cols["pg61.fixture"].update(pg61_fixture(fixture, fixture_src))
    if drive:
        cols["pg61.matched"].update(pg61_drive(drive, drive_src))
    for c in ("pg61.step_cell", "pg61.fixture", "pg61.matched"):
        if cols[c]:
            cols[c]["residency"] = cell(RESIDENCY_PG61, "measured", "A2-RESIDENCY-DESIGN.md 0.2; w0/W0-RESULTS.md")
            for f in FAMILIES:
                cols[c].setdefault(f"fam.{f}", nm(CUPTI_DEADLOCK))
            for k in ("sum_kernels", "unattributed", "busy_ms", "overlap", "concurrency"):
                cols[c].setdefault(k, nm(CUPTI_DEADLOCK))
            cols[c].setdefault("expert_bytes_hbm", nm("Pg61's hot-expert reads are inside MegaMoE; no counter"))
            cols[c].setdefault("expert_bytes_c2c", nm("Pg61 does not read cold experts over C2C (the RTX computes them)"))
            cols[c].setdefault("graph_nodes", nm("vLLM's captured graphs are not walkable from outside the engine"))
    if peer3:
        cols["pg61.sidecar"].update(pg61_peer3(peer3, peer3_src))
        if cols["pg61.sidecar"]:
            cols["pg61.sidecar"]["condition"] = cell("Pg61 sidecar stats, all requests in its window, by_T['6'] = the "
                                                     "6-row layer-calls; not the fixture", "measured", peer3_src)
            cols["pg61.sidecar"]["residency"] = cell(RESIDENCY_PG61, "measured", "A2-RESIDENCY-DESIGN.md 0.2")
    for c in ("ours.headline",):
        if cols[c]:
            for f in FAMILIES:
                cols[c].setdefault(f"fam.{f}", nm("the loop's graphs cannot be profiled per kernel (nsys stalls on "
                                                  "graph replay); attribution: ours.matched (verify) / "
                                                  "ours.loop_static (verify + draft)"))
            for k in ("expert_bytes_hbm", "expert_bytes_c2c"):
                cols[c].setdefault(k, nm("headline records the cold share, not per-step unique counts; see "
                                         "ours.matched"))
    gaps = []

    def gap(a, b, key, matched):
        x, y = cols[a].get(key), cols[b].get(key)
        if x and y and x["value"] is not None and y["value"] is not None:
            gaps.append({"metric": key, "ours": a, "pg61": b, "matched": matched,
                         "value": x["value"] - y["value"],
                         "how": f"{a} {x['value']:.3f} - {b} {y['value']:.3f} = {x['value'] - y['value']:+.3f}"})
    for b, m in (("pg61.matched", "prompts matched; tokens: ours own stream vs Pg61's"),
                 ("pg61.step_cell", "NOT matched: different prompt, 512 tokens")):
        for key in ("step_gpu_ms", "tokens_per_step", "wall_ms_per_token_decode"):
            gap("ours.headline", b, key, m)
    gap("ours.headline", "pg61.fixture", "wall_ms_per_token_e2e",
        "prompts matched; ours streamed fast reps incl. reference prefill vs Pg61 non-streamed with top-20 logprobs")
    for c, v in cols.items():          # every cell of a present column is one of the three kinds -- never blank
        if v:
            for k, _ in METRICS:
                v.setdefault(k, nm(f"{c}: this source does not record it"))
    return {"schema": "gap-ledger/1", "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "columns": cols,
            "column_labels": dict(COLUMNS), "metrics": METRICS, "gaps": gaps,
            "inputs": {"headline": headline_src, "tf": tf_src, "anatomy": anatomy_src,
                       "loop_anatomy": loop_anatomy_src, "pg61_step": pg61_step_src, "pg61_fixture": fixture_src,
                       "pg61_drive": drive_src, "peer3": peer3_src},
            "resolvable_pg61_from_existing": resolvable_pg61(cols),
            "pending": {c: why for c, (inp, why) in PENDING.items() if not cols[c]}}


def resolvable_pg61(cols: Dict[str, Dict[str, dict]]) -> List[str]:
    out = []
    for c in ("pg61.step_cell", "pg61.fixture", "pg61.sidecar"):
        for k, v in cols[c].items():
            if v["kind"] != "not measured" and k not in ("condition", "residency") and f"{k}" not in out:
                out.append(k)
    return out


def _cell_text(v: Optional[dict]) -> str:
    if not v:
        return ""
    if v["kind"] == "not measured":
        return "not measured"
    x = v["value"]
    if isinstance(x, float):
        s = f"{x:.3e}" if abs(x) >= 1e6 else f"{x:.3f}"
    elif isinstance(x, (dict, list)):
        s = json.dumps(x)[:60]
    else:
        s = str(x)[:60]
    return f"{s} [{TAG[v['kind']]}]"


def markdown(led: dict) -> str:
    cols = [c for c, _ in COLUMNS if led["columns"].get(c)]
    L = [f"# Gap ledger -- ours vs Pg61 ({led['date']})", "",
         "Tags: [M] measured, [D] derived (arithmetic below), [S] nsys static [static, diagnostic] -- eager kernel "
         "durations, not an additive accounting of the captured graph; `not measured` says why in the JSON.", ""]
    L.append("| metric | " + " | ".join(led["column_labels"][c] for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    keys = [k for k, _ in led["metrics"]]
    extra = sorted({k for c in cols for k in led["columns"][c] if k not in keys})
    labels = dict(led["metrics"])
    for k in keys + extra:
        if k in ("condition", "residency"):
            continue
        row = [_cell_text(led["columns"][c].get(k)) for c in cols]
        if not any(row):
            continue
        L.append(f"| {labels.get(k, k)} | " + " | ".join(row) + " |")
    L += ["", "## Conditions"]
    for c in cols:
        for k in ("condition", "residency"):
            v = led["columns"][c].get(k)
            if v:
                L.append(f"- {c} {k}: {v['value']}")
    L += ["", "## Gaps (ours - Pg61)"]
    for g in led["gaps"]:
        L.append(f"- {g['metric']}: {g['how']} ({g['matched']})")
    L += ["", "## Derivations"]
    for c in cols:
        for k, v in led["columns"][c].items():
            if v["kind"] in ("derived", "measured") and k not in ("condition", "residency"):
                L.append(f"- {c} {k}: {v['how']}")
    L += ["", f"Pg61 cells resolvable from existing receipts: {', '.join(led['resolvable_pg61_from_existing'])}"]
    if led.get("pending"):
        L += ["", "## Pending columns"] + [f"- {c}: {why}" for c, why in led["pending"].items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ token bytes -> ids
def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte-level BPE alphabet (the checkpoint's tokenizer.json is byte-level): byte -> printable char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def vocab_by_bytes(tok_json: dict) -> Tuple[Dict[bytes, int], Dict[str, int], List[bytes]]:
    """tokenizer.json -> (raw bytes -> id for the BPE vocab, added-token content -> id, ambiguous byte strings)."""
    dec = {c: b for b, c in bytes_to_unicode().items()}
    table: Dict[bytes, int] = {}
    amb: List[bytes] = []
    for s, i in (tok_json.get("model") or {}).get("vocab", {}).items():
        try:
            raw = bytes(dec[ch] for ch in s)
        except KeyError:
            raw = s.encode("utf-8")
        if raw in table and table[raw] != i:
            amb.append(raw)
            continue
        table[raw] = int(i)
    added = {t["content"]: int(t["id"]) for t in tok_json.get("added_tokens") or []}
    return table, added, amb


def pg61_ids(rec: dict, table: Dict[bytes, int], added: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """A parity-A record's emitted tokens -> ids (by the `bytes` field vLLM returns per token; added/special tokens
    by content). Returns (ids, positions that could not be mapped)."""
    ids, bad = [], []
    for k, e in enumerate(rec.get("logprobs") or []):
        tok = e.get("token") or ""
        if tok in added:
            ids.append(added[tok])
            continue
        b = e.get("bytes")
        raw = bytes(b) if b is not None else tok.encode("utf-8")
        i = table.get(raw)
        if i is None:
            bad.append(k)
            ids.append(-1)
        else:
            ids.append(i)
    return ids, bad


# ------------------------------------------------------------------------------------------------ graph census (box)
# edge_label / tally_edges / the cudart walk: copied from bench/pdl_probe.py (M2-L5 probe, commit 6070e0d, branch
# m2-l5 -- not on main yet; when runtime/pdl.graph_census lands, prefer it). Added: kernel node names via the driver
# API (cuGraphKernelNodeGetParams_v2 + cuFuncGetName / cuKernelGetName) -> step_anatomy.family_of, and edges by
# (from family, to family). Names are best-effort: any failure leaves the counts intact and reports the error.
NODE_TYPES = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "graph", 5: "empty", 6: "wait_event",
              7: "event_record", 8: "ext_sem_signal", 9: "ext_sem_wait", 10: "mem_alloc", 11: "mem_free",
              12: "batch_mem_op", 13: "conditional"}


def edge_label(edge_type: int, from_port: int, to_port: int) -> str:
    """cudaGraphEdgeData -> a name. type 0 default / 1 programmatic; from_port 0 default, 1 programmatic (stream
    capture of a PDL launch), 2 launch completion."""
    if edge_type == 0:
        return "default" if from_port == 0 else f"default(from_port={from_port})"
    if edge_type == 1:
        return {1: "programmatic", 2: "programmatic(launch_completion)"}.get(from_port, f"programmatic(from_port={from_port})")
    return f"type{edge_type}(from_port={from_port},to_port={to_port})"


def tally_edges(edges: Sequence[Tuple[int, int, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t, fp, tp in edges:
        k = edge_label(int(t), int(fp), int(tp))
        out[k] = out.get(k, 0) + 1
    return out


def family_tally(names: Sequence[Optional[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for n in names:
        f = family_of(n, "", None, roles_known=False) if n else "unnamed"
        out[f] = out.get(f, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def _cudart():
    import ctypes

    cands: List[str] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart" in line:
                    p = line.split()[-1]
                    if p not in cands:
                        cands.append(p)
    except OSError:
        pass
    cands += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    err = []
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError as e:
            err.append(str(e))
    raise RuntimeError("no libcudart: " + "; ".join(err[:3]))


def _kernel_names(nodes, idx: Sequence[int]) -> Tuple[Dict[int, str], Optional[str]]:
    import ctypes as C

    try:
        cu = C.CDLL("libcuda.so.1")

        class KP(C.Structure):   # CUDA_KERNEL_NODE_PARAMS_v2
            _fields_ = [("func", C.c_void_p), ("gridDimX", C.c_uint), ("gridDimY", C.c_uint), ("gridDimZ", C.c_uint),
                        ("blockDimX", C.c_uint), ("blockDimY", C.c_uint), ("blockDimZ", C.c_uint),
                        ("sharedMemBytes", C.c_uint), ("kernelParams", C.c_void_p), ("extra", C.c_void_p),
                        ("kern", C.c_void_p), ("ctx", C.c_void_p)]
        getp = cu.cuGraphKernelNodeGetParams_v2
        getp.argtypes = [C.c_void_p, C.POINTER(KP)]
        fname = cu.cuFuncGetName
        fname.argtypes = [C.POINTER(C.c_char_p), C.c_void_p]
        kname = getattr(cu, "cuKernelGetName", None)
        if kname is not None:
            kname.argtypes = [C.POINTER(C.c_char_p), C.c_void_p]
        out: Dict[int, str] = {}
        for i in idx:
            p = KP()
            if getp(nodes[i], C.byref(p)) != 0:
                continue
            s = C.c_char_p()
            if p.func and fname(C.byref(s), p.func) == 0 and s.value:
                out[i] = s.value.decode(errors="replace")
            elif p.kern and kname is not None and kname(C.byref(s), p.kern) == 0 and s.value:
                out[i] = s.value.decode(errors="replace")
        return out, None
    except Exception as e:  # noqa: BLE001
        return {}, f"{type(e).__name__}: {e}"


def graph_census(g, names: bool = True) -> dict:
    """Node count, node types and edge kinds of a CUDAGraph made with keep_graph=True (via cudart, ctypes), plus
    kernel node families and family-pair edge counts when the driver gives kernel names. {'error'} on failure."""
    import ctypes as C

    try:
        raw = int(g.raw_cuda_graph())
        lib = _cudart()
        vp, st = C.c_void_p, C.c_size_t

        class EdgeData(C.Structure):
            _fields_ = [("from_port", C.c_ubyte), ("to_port", C.c_ubyte), ("type", C.c_ubyte),
                        ("reserved", C.c_ubyte * 5)]

        def chk(rc, what):
            if rc != 0:
                lib.cudaGetErrorString.restype = C.c_char_p
                raise RuntimeError(f"{what}: cudart error {rc}: {lib.cudaGetErrorString(rc).decode()}")

        ver = C.c_int(0)
        lib.cudaRuntimeGetVersion(C.byref(ver))
        n = st(0)
        lib.cudaGraphGetNodes.argtypes = [vp, C.c_void_p, C.POINTER(st)]
        chk(lib.cudaGraphGetNodes(vp(raw), None, C.byref(n)), "cudaGraphGetNodes(count)")
        nodes = (vp * max(1, n.value))()
        chk(lib.cudaGraphGetNodes(vp(raw), nodes, C.byref(n)), "cudaGraphGetNodes")
        lib.cudaGraphNodeGetType.argtypes = [vp, C.POINTER(C.c_int)]
        types: Dict[int, int] = {}
        ntype: List[int] = []
        for i in range(n.value):
            t = C.c_int(-1)
            chk(lib.cudaGraphNodeGetType(nodes[i], C.byref(t)), "cudaGraphNodeGetType")
            types[t.value] = types.get(t.value, 0) + 1
            ntype.append(t.value)
        fn = getattr(lib, "cudaGraphGetEdges_v2", None)
        used = "cudaGraphGetEdges_v2"
        if fn is None:
            fn, used = getattr(lib, "cudaGraphGetEdges", None), "cudaGraphGetEdges"
            if fn is None or ver.value < 13000:
                raise RuntimeError(f"no 5-argument cudaGraphGetEdges in this cudart (version {ver.value})")
        fn.argtypes = [vp, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(st)]
        ne = st(0)
        chk(fn(vp(raw), None, None, None, C.byref(ne)), used + "(count)")
        m = max(1, ne.value)
        frm, to, ed = (vp * m)(), (vp * m)(), (EdgeData * m)()
        chk(fn(vp(raw), frm, to, ed, C.byref(ne)), used)
        edges = [(ed[i].type, ed[i].from_port, ed[i].to_port) for i in range(ne.value)]
        rep = {"cudart_version": ver.value, "api": used, "nodes": n.value, "kernel_nodes": types.get(0, 0),
               "node_types": {NODE_TYPES.get(k, str(k)): v for k, v in sorted(types.items())},
               "edges": ne.value, "edge_kinds": tally_edges(edges)}
        if getattr(g, "_census_note", None):
            rep["instantiate_note"] = g._census_note     # keep_graphs(): torch had already instantiated it
        if names:
            kidx = [i for i, t in enumerate(ntype) if t == 0]
            nm_, err = _kernel_names(nodes, kidx)
            if err:
                rep["names_error"] = err
            else:
                rep["kernel_names_resolved"] = len(nm_)
                rep["kernel_families"] = family_tally([nm_.get(i) for i in kidx])
                pos = {int(nodes[i] or 0): i for i in range(n.value)}

                def fam(ptr):
                    i = pos.get(int(ptr or 0))
                    if i is None:
                        return "?"
                    if ntype[i] != 0:
                        return NODE_TYPES.get(ntype[i], str(ntype[i]))
                    nm1 = nm_.get(i)
                    return family_of(nm1, "", None, roles_known=False) if nm1 else "unnamed"
                pairs: Dict[str, int] = {}
                for j in range(ne.value):
                    k = f"{fam(frm[j])} -> {fam(to[j])} [{edge_label(ed[j].type, ed[j].from_port, ed[j].to_port)}]"
                    pairs[k] = pairs.get(k, 0) + 1
                rep["edge_family_pairs"] = dict(sorted(pairs.items(), key=lambda x: -x[1])[:40])
        return rep
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        return {"error": f"{type(e).__name__}: {e}"}


def _is_double_instantiate(e: BaseException) -> bool:
    """True for torch's complaint about instantiating a CUDAGraph that is not (or no longer) in the 'not yet
    instantiated, keep_graph=True' state: 'instantiate() is intended to be called by the user only when
    keep_graph=true' (and its 'already instantiated' sibling). Any other error is real and propagates."""
    msg = str(e)
    return "instantiate" in msg and any(w in msg for w in ("keep_graph", "already", "only once", "twice"))


def keep_graphs():
    """Make every torch.cuda.CUDAGraph() constructed from now on keep its cudaGraph_t (keep_graph=True) so the
    runtime's own captures (StaticStep, DSparkLoop: `torch.cuda.CUDAGraph()` looked up at call time) can be walked
    afterwards by graph_census (which only READS raw_cuda_graph(); it never instantiates).

    The kept graph is instantiated exactly once, at capture end, so replay() never pays the lazy-instantiate cost:
    `instantiate` is idempotent here (a second call -- ours, the runtime's, or torch's own -- is a no-op), and a
    torch complaint about instantiating twice at capture end (the LOOSE-ENDS-2 crash: `RuntimeError: instantiate()
    is intended to be called by the user only when keep_graph=true`, seen with --census-loop, whose DSparkLoop
    captures several graphs back to back) is recorded on the graph (`_census_note`) instead of raised. Any other
    instantiate error still raises. Returns an undo(). Callers should run it in try/finally."""
    import torch

    orig = torch.cuda.CUDAGraph

    class _Kept(orig):
        def __new__(cls, keep_graph=True):          # a caller asking for keep_graph=False is overridden: the census needs the graph
            return super().__new__(cls, True)

        def instantiate(self):
            if self.__dict__.get("_census_inst"):
                return
            try:
                super().instantiate()
            except RuntimeError as e:
                if not _is_double_instantiate(e):
                    raise
                self._census_note = str(e)[:200]    # already instantiated by torch / the runtime: fine, the exec exists
            self._census_inst = True

        def capture_end(self):
            super().capture_end()
            self.instantiate()

    torch.cuda.CUDAGraph = _Kept

    def undo():
        torch.cuda.CUDAGraph = orig
    return undo


# ------------------------------------------------------------------------------------------------ box: our runtime
def _env_like_headline(a) -> None:
    """The headline's environment (bench/headline.py main): GEMV M=1 pin, fp32 impl, rowmatch -- before any build."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not a.no_gemv_pin:
        os.environ["DSV41_GEMV_PIN_M1"] = "1"
    os.environ["DSV41_FP32_IMPL"] = a.fp32_impl
    os.environ["DSV41_MOE_ROWMATCH"] = a.moe_rowmatch


def _build(a, load_mtp: bool):
    """Boot + Generator exactly as bench/headline.py does (same impl flags, the dg-inplace pool pre-grown ONCE to the
    spare budget before any build), plus the 6-row hidden layers (the verify body's main_hidden copies)."""
    import torch  # noqa: F401

    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load

    mp = a.map if a.map and os.path.exists(a.map) else None
    t0 = time.time()
    model = build_and_load(a.model_dir, mp, "cuda", reserve_host_gib=a.reserve_host_gib, load_mtp=load_mtp)
    boot = model.runtime_load.summary()
    boot["wall_s"] = time.time() - t0
    det = set_determinism(0, False)     # fills off (= first_light --no-fill-uninit; graphs are captured fills-off too)
    with open(os.path.join(a.model_dir, "config.json")) as f:
        hidden = [int(x) for x in json.load(f).get("dspark_target_layer_ids") or []]
    gen = Generator(model, a.model_dir, mode=a.mode, max_pos=a.max_pos, moe_impl=a.moe_impl, fp8_impl=a.fp8_impl,
                    glue_impl=a.glue_impl, attn_impl=a.attn_impl, spare_slots=a.spare_slots, hidden_layers=hidden)
    if a.moe_impl == "dg-inplace" and getattr(gen, "pool", None) is not None:
        from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS

        gen.pool.add_spare_slots(a.spare_slots or DEFAULT_SPARE_SLOTS_ROWS, model)
    return model, gen, {"boot": boot, "determinism": det, "hidden_layers": hidden, "map": mp}


def load_targets(gen, rows: Sequence[dict], model_dir: str) -> Tuple[Dict[str, List[int]], Dict[str, dict], Dict[str, List[int]]]:
    """Per fixture prompt: our prompt ids (checkpoint encoding, = Pg61's prompt_tokens on 8/8 per ENVIRONMENT.md T0)
    and Pg61's emitted ids (bytes -> ids); checks recorded, a prompt with an unmapped token is skipped."""
    with open(os.path.join(model_dir, "tokenizer.json")) as f:
        table, added, amb = vocab_by_bytes(json.load(f))
    ids_, tg, chk = {}, {}, {}
    for r in rows:
        ids = gen.encode([{"role": "user", "content": r["prompt"]}])
        t, bad = pg61_ids(r, table, added)
        pt = (r.get("usage") or {}).get("prompt_tokens")
        text_ok = None
        try:
            text_ok = gen.decode([x for x in t if x >= 0]) == r.get("text")
        except Exception:  # noqa: BLE001
            pass
        chk[r["tag"]] = {"prompt_tokens_ours": len(ids), "prompt_tokens_pg61": pt, "n_targets": len(t),
                         "unmapped_positions": bad[:20], "decode_equals_text": text_ok,
                         "ok": not bad and (pt is None or pt == len(ids)), "ambiguous_vocab_bytes": len(amb)}
        if chk[r["tag"]]["ok"]:
            ids_[r["tag"]], tg[r["tag"]] = ids, t
    return ids_, chk, tg


def _routed(gen) -> Optional[Tuple[int, int]]:
    """(hot, cold) routed calls of the R-row static step since its capture/build: one per UNIQUE expert per layer per
    step at rows > 1 (StaticStep.routed_counts; syncs). None without counters. Deltas around a cell = that cell's."""
    st = getattr(gen, "static", None)
    rc = st.routed_counts() if st is not None and hasattr(st, "routed_counts") else None
    return (sum(rc[0]), sum(rc[1])) if rc is not None else None


def run_census(gen, a, rep: dict) -> dict:
    """The graph census of --census / --census-loop: the StaticStep's graphs, and with --census-loop the DSpark
    loop's (2-graph A/B, then 1-graph AB/B). READS the kept graphs only (graph_census never instantiates); the one
    instantiate per graph happened at capture end under keep_graphs(). Stored in rep["census"], printed; a failure
    in the loop stage is recorded in rep["errors"] and the tf_graphs census is kept."""
    cen = {"tf_graphs": {}, "loop_graphs": {}}
    st = getattr(gen, "static", None)
    for k, g in (getattr(st, "graphs", None) or {}).items():
        cen["tf_graphs"][str(k)] = graph_census(g)
    if a.census_loop:
        try:
            from runtime.dspark_loop import DSparkLoop

            loop = DSparkLoop(gen, a.max_new)
            try:
                loop.capture(n_graphs=2)
                for k, g in loop.graphs.items():
                    cen["loop_graphs"][f"2g:{k}"] = graph_census(g)
                loop.capture(n_graphs=1)
                for k, g in loop.graphs.items():
                    cen["loop_graphs"][f"1g:{k}"] = graph_census(g)
            finally:
                loop.close()
        except Exception:
            import traceback

            rep["errors"].append({"stage": "census_loop", "traceback": traceback.format_exc()})
    rep["census"] = cen
    for grp, d in cen.items():
        for k, c in d.items():
            print(f"[census] {grp} {k}: " + (c.get("error") or f"{c['nodes']} nodes ({c['kernel_nodes']} kernel), "
                                             f"{c['edges']} edges {c['edge_kinds']}"), flush=True)
    return cen


def cmd_ours_tf(a) -> int:
    _env_like_headline(a)
    import torch

    undo = keep_graphs() if a.census else None
    model, gen, info = _build(a, load_mtp=a.census_loop)
    from runtime.generate import install_layer_nvtx
    from first_light import step_cell_rows   # the M1-T10 R-row teacher-forced cell (events around step_rows only)

    if a.nvtx_layers:
        install_layer_nvtx(model)
    gen.set_rows(6)
    with open(a.fixture) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    ids_, chk, tg = load_targets(gen, rows, a.model_dir)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "mode": a.mode, "map": info["map"],
           "spare_slots": a.spare_slots or 16, "impls": {"moe": a.moe_impl, "fp8": a.fp8_impl, "glue": a.glue_impl,
                                                        "attn": a.attn_impl, "fp32": a.fp32_impl,
                                                        "moe_rowmatch": a.moe_rowmatch},
           "gemv_pin_m1": os.environ.get("DSV41_GEMV_PIN_M1") == "1", "boot": info["boot"],
           "determinism": info["determinism"], "hidden_layers": info["hidden_layers"], "targets": chk, "reps": [],
           "errors": []}
    print(f"[tf] targets ok {sum(1 for v in chk.values() if v['ok'])}/{len(chk)}: "
          + ", ".join(f"{k}:{'ok' if v['ok'] else 'SKIP'}" for k, v in chk.items()), flush=True)
    if a.profile:
        w, n = (int(x) for x in a.profile.split(":"))
        rep["profile_window"] = gen.arm_profile(w, n)
    with torch.inference_mode():
        gen.static_step()          # build (and, graph mode, capture) BEFORE the first cell: the routed counters reset
        gen._ensure_captured()     # at capture, so a delta around the first prompt would otherwise be lost
    for rr in range(a.reps):
        pr = []
        for tag, ids in ids_.items():
            t = tg[tag]
            n = (len(t) - 1) // 6
            if n < 1:
                continue
            try:
                c0 = _routed(gen)
                r = step_cell_rows(gen, ids, t, n, stats_from=1)
                c1 = _routed(gen)
                rec = {"tag": tag, "rep": rr, "steps": n, "positions": 6 * n, "step_ms": r["step_ms"],
                       "prefill_ms": r["prefill_ms"], "argmax_agree_with_pg61": r["argmax_agree"],
                       "argmax_compared": r["argmax_compared"], "first_disagreement": r["first_disagreement"]}
                if c0 is not None and c1 is not None:
                    rec["hot_unique"], rec["cold_unique"] = c1[0] - c0[0], c1[1] - c0[1]
                pr.append(rec)
                print(f"[tf rep{rr}] {tag:6s} {n} steps, {sum(r['step_ms']) / n:.3f} ms/step, agree "
                      f"{r['argmax_agree']}/{r['argmax_compared']}", flush=True)
            except Exception:
                import traceback

                rep["errors"].append({"tag": tag, "rep": rr, "traceback": traceback.format_exc()})
                print(f"[tf rep{rr}] {tag}: FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        rep["reps"].append({"rep": rr, "prompts": pr})
    ovf = getattr(gen.static, "overflow_counts", None) if getattr(gen, "static", None) is not None else None
    rep["overflow"] = ovf() if ovf else None
    if a.census:
        try:
            run_census(gen, a, rep)
        finally:
            undo()      # always restore torch.cuda.CUDAGraph, also when the census raises
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        rep["hbm_end_GiB"] = {"free": round(free / 2**30, 3), "total": round(total / 2**30, 3)}
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(json.loads(json.dumps(rep, default=str)), f, indent=1)
    cells = [c for r in rep["reps"] for c in r["prompts"]]
    ms = [x for c in cells for x in c["step_ms"]]
    if ms:
        print(f"[tf] {a.mode}: {len(ms)} steps, {sum(ms) / len(ms):.3f} ms/step pooled -> {a.json}", flush=True)
    return 0 if not rep["errors"] else 2


def cmd_ours_loop_profile(a) -> int:
    """The headline loop, static mode, NVTX decode_step/verify/draft around the loop bodies (instance-attribute wraps;
    runtime/ untouched), cudaProfilerApi window via DSparkLoop.generate(profile=(first, n))."""
    _env_like_headline(a)
    import torch

    model, gen, info = _build(a, load_mtp=True)
    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import install_layer_nvtx

    if a.nvtx_layers:
        install_layer_nvtx(model)
    nv = torch.cuda.nvtx
    loop = DSparkLoop(gen, a.max_new)

    def wrap(fn, name):
        def w(*x, **k):
            nv.range_push(name)
            try:
                return fn(*x, **k)
            finally:
                nv.range_pop()
        return w
    loop._verify_accept = wrap(loop._verify_accept, "verify")
    loop._draft = wrap(loop._draft, "draft")
    loop.step_once = wrap(loop.step_once, "decode_step")
    with open(a.fixture) as f:
        rows = {json.loads(l)["tag"]: json.loads(l) for l in f if l.strip()}
    r = rows[a.tag]
    ids = gen.encode([{"role": "user", "content": r["prompt"]}])
    w, n = (int(x) for x in a.profile.split(":"))
    res = loop.generate(ids, a.max_new, timing=True, profile=(w, n), tag=a.tag)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "tag": a.tag, "profile": [w, n],
           "steps": res.get("steps"), "tokens_per_step": res.get("tokens_per_step"),
           "step_ms_eager": res.get("step_ms"), "verify_ms_eager": res.get("verify_ms"),
           "draft_ms_eager": res.get("draft_ms"), "boot": info["boot"]}
    with open(a.json, "w") as f:
        json.dump(json.loads(json.dumps(rep, default=str)), f, indent=1)
    loop.close()
    print(f"[loop-profile] {a.tag}: {res.get('steps')} steps, window {w}:{n} -> {a.json}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ box host: Pg61
def _stream(base: str, body: dict, timeout: int = 900) -> dict:
    """POST a streamed chat completion; per content/usage chunk: arrival time, logprob token count + bytes, the
    running usage.completion_tokens when the server streams continuous usage."""
    import urllib.request

    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks = []
    usage = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            t = time.time()
            u = d.get("usage")
            if u:
                usage = u
            ch = (d.get("choices") or [None])[0]
            if ch is None:
                continue
            lp = ((ch.get("logprobs") or {}).get("content")) or []
            content = (ch.get("delta") or {}).get("content") or ""
            if not content and not lp and not u:
                continue
            chunks.append({"t": t - t0, "chars": len(content), "lp_tokens": len(lp),
                           "lp_bytes": [e.get("bytes") for e in lp], "completion_tokens": (u or {}).get("completion_tokens")})
    return {"chunks": chunks, "usage": usage, "wall_s": time.time() - t0}


def step_tokens_from_chunks(chunks: Sequence[dict]) -> List[int]:
    """Tokens emitted per SSE chunk (= per engine step): the logprob entries when present, else the delta of the
    running completion_tokens (continuous usage)."""
    out, last = [], 0
    for c in chunks:
        if c.get("lp_tokens"):
            out.append(int(c["lp_tokens"]))
        elif c.get("completion_tokens") is not None:
            d = int(c["completion_tokens"]) - last
            last = int(c["completion_tokens"])
            if d > 0:
                out.append(d)
    return out


def cmd_pg61_drive(a) -> int:
    sys.path.insert(0, os.path.join(REPO, "w0"))
    import w0_drive as W   # stdlib; FIXTURE, chat_body (the parity/step cells' exact request bodies), metrics

    W.BASE = a.base
    fix = {r["tag"]: r for r in (json.loads(l) for l in open(a.fixture) if l.strip())}
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "base": a.base, "model": W.MODEL, "max_tokens": a.max_new,
           "argv": sys.argv[1:], "prompts": {}}
    for tag, prompt in W.FIXTURE:
        if a.only and tag not in a.only.split(","):
            continue
        p = {"count": None, "time": []}
        W.one_stream(prompt, 32)                                   # warm (prefix cache, graphs), as w0_drive step
        b = W.chat_body(prompt, a.max_new, stream=True)
        b["logprobs"], b["top_logprobs"] = True, 0
        b["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        s = _stream(a.base, b)
        st = step_tokens_from_chunks(s["chunks"])
        got = [x for c in s["chunks"] for x in c["lp_bytes"]]
        want = [e.get("bytes") for e in (fix.get(tag) or {}).get("logprobs") or []]
        p["count"] = {"step_tokens": st, "n_tokens": len(got), "equal_to_parity_a": got == want if want else None,
                      "first_diff": next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), None),
                      "usage": s["usage"]}
        for rr in range(a.reps):
            b = W.chat_body(prompt, a.max_new, stream=True)
            b["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
            m0 = W.metrics()
            s = _stream(a.base, b)
            m1 = W.metrics()
            ts = [c["t"] for c in s["chunks"] if c["chars"]]
            stp = step_tokens_from_chunks(s["chunks"])
            rec = {"rep": rr, "wall_s": round(s["wall_s"], 4), "ttft_s": round(ts[0], 4) if ts else None,
                   "decode_wall_s": round(ts[-1] - ts[0], 4) if len(ts) > 1 else None, "chunks": len(ts),
                   "step_tokens_usage": stp, "first_chunk_tokens": stp[0] if stp else None, "usage": s["usage"]}
            if "num_drafts_total" in m0 and "num_drafts_total" in m1:
                rec.update({"steps": m1["num_drafts_total"] - m0["num_drafts_total"],
                            "gen_tokens": m1["generation_tokens_total"] - m0["generation_tokens_total"],
                            "accepted": m1["num_accepted_tokens_total"] - m0["num_accepted_tokens_total"],
                            "preemptions": m1.get("num_preemptions_total", 0) - m0.get("num_preemptions_total", 0)})
            p["time"].append(rec)
        rep["prompts"][tag] = p
        t = p["time"]
        print(f"pg61 {tag}: equal_to_parity_a={p['count']['equal_to_parity_a']} steps={[x.get('steps') for x in t]} "
              f"decode_wall={[x.get('decode_wall_s') for x in t]}", flush=True)
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    print(f"pg61-drive -> {a.json}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ build (stdlib)
def _load(path: Optional[str]):
    if not path:
        return None, ""
    if not os.path.exists(path):
        print(f"[build] missing input {path} -> its columns stay not measured", file=sys.stderr)
        return None, ""
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()], os.path.relpath(path, REPO)
        return json.load(f), os.path.relpath(path, REPO)


def cmd_build(a) -> int:
    h, hs = _load(a.headline)
    tf, ts = _load(a.tf)
    an, ans = _load(a.anatomy)
    la, las = _load(a.loop_anatomy)
    ps, pss = _load(a.pg61_step)
    fx, fxs = _load(a.pg61_fixture)
    dr, drs = _load(a.pg61_drive)
    p3, p3s = _load(a.peer3)
    led = build_ledger(headline=h, headline_src=hs, tf=tf, tf_src=ts, anatomy=an, anatomy_src=ans, loop_anatomy=la,
                       loop_anatomy_src=las, pg61_step=ps, pg61_step_src=pss, fixture=fx, fixture_src=fxs, drive=dr,
                       drive_src=drs, peer3=p3, peer3_src=p3s)
    for path in (a.out, a.md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(led, f, indent=1)
    md = markdown(led)
    with open(a.md, "w") as f:
        f.write(md)
    print(md)
    print(f"[build] -> {a.out}, {a.md}")
    return 0


def _runtime_args(ap) -> None:
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--fixture", default=os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--mode", choices=["static", "graph"], default="graph")
    ap.add_argument("--max-pos", type=int, default=4095)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--moe-impl", default="dg-inplace")
    ap.add_argument("--fp8-impl", default="gemv")
    ap.add_argument("--glue-impl", default="fused")
    ap.add_argument("--attn-impl", default="flashmla")
    ap.add_argument("--fp32-impl", choices=["cublas", "fixed"], default="fixed")
    ap.add_argument("--moe-rowmatch", choices=["on", "align", "off"], default="on")
    ap.add_argument("--spare-slots", type=int, default=16)
    ap.add_argument("--no-gemv-pin", action="store_true")
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--nvtx-layers", action="store_true")
    ap.add_argument("--json", required=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--headline", default=os.path.join(REPO, "docs", "headline-20261107j.json"))
    b.add_argument("--tf", default=None)
    b.add_argument("--anatomy", default=None)
    b.add_argument("--loop-anatomy", default=None)
    b.add_argument("--pg61-step", default=os.path.join(REPO, "w0", "results", "phase1", "step-pg61.json"))
    b.add_argument("--pg61-fixture", default=os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl"))
    b.add_argument("--pg61-drive", default=None)
    b.add_argument("--peer3", default=os.path.join(REPO, "docs", "data", "peer3_stats.json"))
    b.add_argument("--out", default=os.path.join(REPO, "scratch", "gap-ledger-ours.json"))
    b.add_argument("--md", default=os.path.join(REPO, "scratch", "gap-ledger-ours.md"))
    t = sub.add_parser("ours-tf")
    _runtime_args(t)
    t.add_argument("--reps", type=int, default=2)
    t.add_argument("--profile", default=None, help="W:N -> cudaProfilerStart before step W+1 (counted over the "
                                                   "whole run's 6-row steps), Stop after W+N")
    t.add_argument("--census", action="store_true")
    t.add_argument("--census-loop", action="store_true", help="also the DSpark loop graphs (boots with load_mtp)")
    lp = sub.add_parser("ours-loop-profile")
    _runtime_args(lp)
    lp.add_argument("--tag", default="prose")
    lp.add_argument("--profile", default="16:32", help="FIRST:N loop steps inside the cudaProfilerApi range")
    d = sub.add_parser("pg61-drive")
    d.add_argument("--base", default=os.environ.get("W0_BASE", "http://127.0.0.1:30006"))
    d.add_argument("--fixture", default=os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl"))
    d.add_argument("--only", default=None)
    d.add_argument("--max-new", type=int, default=256)
    d.add_argument("--reps", type=int, default=3)
    d.add_argument("--json", required=True)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd in (None, "build"):
        if a.cmd is None:
            a = build_parser().parse_args(["build"] + list(argv or []))
        return cmd_build(a)
    if a.cmd == "ours-tf":
        if a.census_loop:
            a.census = True
        if a.profile and a.mode == "graph":
            raise SystemExit("--profile needs --mode static (nsys stalls on the graph-replay path)")
        return cmd_ours_tf(a)
    if a.cmd == "ours-loop-profile":
        a.mode = "static"
        return cmd_ours_loop_profile(a)
    if a.cmd == "pg61-drive":
        return cmd_pg61_drive(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
