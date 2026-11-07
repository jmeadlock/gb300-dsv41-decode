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
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

METRIC = "ms per 6-row step, DSpark k=5"           # PROJECT.md "The number" (C1, 6 rows, DSpark k=5)
PG61_MS_PER_STEP = 7.36                              # PROJECT.md baseline (production, 2026-10-29)
CHARTER_BAR_MS = 5.0                                 # PROJECT.md bar


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
    step, ver, dra, tok, all_step, all_tok = [], [], [], [], [], []
    for r in runs:
        ms = r.get("step_ms") or []
        acc = r.get("accepts") or []
        v, d = r.get("verify_ms"), r.get("draft_ms")
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
    out = {"stats_from_step": stats_from, "steps": len(step), "ms_per_step": _stats(step),
           "ms_verify_graph": _stats(ver) if ver else None, "ms_draft_graph": _stats(dra) if dra else None,
           "tokens_per_step": (sum(tok) / len(tok)) if tok else None,
           "ms_per_token": (sum(step) / sum(tok)) if tok else None,
           "all_steps": len(all_step), "all_ms_per_step": _stats(all_step),
           "all_tokens_per_step": (sum(all_tok) / len(all_tok)) if all_tok else None,
           "all_ms_per_token": (sum(all_step) / sum(all_tok)) if all_tok else None}
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


def summary_lines(rep: dict) -> List[str]:
    h = rep.get("headline") or {}
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = [f"HEADLINE {rep.get('date')} | {METRIC} | mode {rep.get('mode')} | impls {rep.get('impls')} | "
             f"gemv pin M1 {rep.get('gemv_pin_m1')} | max_new {rep.get('max_new')} | prompts {rep.get('n_prompts')}"]
    for key in ("two_graphs", "one_graph", "eager"):
        w = h.get(key)
        if not w:
            continue
        ms = (w.get("ms_per_step") or {}).get("mean")
        v = (w.get("ms_verify_graph") or {}).get("mean") if w.get("ms_verify_graph") else None
        d = (w.get("ms_draft_graph") or {}).get("mean") if w.get("ms_draft_graph") else None
        lines.append(f"  {key:10s}: {f(ms)} ms/step (verify {f(v)} + draft {f(d)}) | {f(w.get('tokens_per_step'), 3)} "
                     f"tok/step | {f(w.get('ms_per_token'))} ms/token | window steps >= {w.get('stats_from_step')}: "
                     f"{w.get('steps')} (all {w.get('all_steps')} steps: {f((w.get('all_ms_per_step') or {}).get('mean'))}"
                     f" ms/step, {f(w.get('all_tokens_per_step'), 3)} tok/step)")
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
    ap.add_argument("--fp8-impl", choices=["ref", "gemv"], default="ref")
    ap.add_argument("--glue-impl", choices=["ref", "fused"], default="ref")
    ap.add_argument("--attn-impl", choices=["ref", "bounded", "flashmla"], default="ref")
    ap.add_argument("--spare-slots", type=int, default=None, help="6-row dg-inplace spare slots/layer (default 8)")
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
    return ap


def _hbm(torch) -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"free": round(free / 2**30, 3), "total": round(total / 2**30, 3),
            "allocated": round(torch.cuda.memory_allocated() / 2**30, 3),
            "max_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 3)}


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
                fast.append(loop.generate(p["ids"], max_new, stop_on_eos=stop_on_eos, parity=False, profile=prof))
            for _ in range(parity_reps):
                par.append(loop.generate(p["ids"], max_new, stop_on_eos=stop_on_eos, parity=True, tag=tag,
                                         prompt=p.get("prompt")))
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
                    "tokens_per_step": base["tokens_per_step"], "accept_hist": [acc.count(i) for i in range(6)],
                    "accepts": acc, "stream": s, "replay_ok": replay, "one_graph_equal": one_equal,
                    "prefill_ms": [x["prefill_ms"] for x in fast], "r1_ms_per_token": p.get("r1_ms_per_token"),
                    "r1_vs_ref": p.get("r1_vs_ref"), "finish_reason": base["finish_reason"],
                    "window": window(fast, stats_from),
                    "window_one_graph": window(one["fast"], stats_from) if one else None,
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
    w2 = head.get("two_graphs") or head.get("one_graph") or head.get("eager") or {}
    head["ms_per_6row_step_dspark_k5"] = (w2.get("ms_per_step") or {}).get("mean")
    head["ms_verify_graph"] = (w2.get("ms_verify_graph") or {}).get("mean") if w2.get("ms_verify_graph") else None
    head["ms_draft_graph"] = (w2.get("ms_draft_graph") or {}).get("mean") if w2.get("ms_draft_graph") else None
    head["tokens_per_step"] = w2.get("tokens_per_step")
    head["ms_per_token"] = w2.get("ms_per_token")
    if head.get("one_graph"):
        head["ms_per_6row_step_one_graph"] = (head["one_graph"].get("ms_per_step") or {}).get("mean")
    pf = [x for v in per.values() for x in v["prefill_ms"]]
    head["prefill_ms_mean"] = sum(pf) / len(pf) if pf else None
    rep["headline"] = head
    return rep


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not a.no_gemv_pin:
        os.environ["DSV41_GEMV_PIN_M1"] = "1"     # before any GEMV get_config (builds, warm-ups, captures)
    import torch

    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load

    date = time.strftime("%Y%m%d")
    json_path = a.json or os.path.join(REPO, "docs", f"headline-{date}.json")
    graphs = [int(x) for x in a.graphs.split(",") if x.strip()]
    impls = {"moe": a.moe_impl, "fp8": a.fp8_impl, "glue": a.glue_impl, "attn": a.attn_impl}
    rep: Dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
                 "metric": METRIC, "mode": a.mode, "impls": impls, "max_new": a.max_new,
                 "gemv_pin_m1": os.environ.get("DSV41_GEMV_PIN_M1") == "1", "errors": [], "json": json_path,
                 "pg61_ms_per_step": PG61_MS_PER_STEP, "bar_ms_per_step": CHARTER_BAR_MS}
    t0 = time.time()
    mp = a.map if a.map and os.path.exists(a.map) else None
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
    profile = None
    if a.profile:
        t_, f_, n_ = a.profile.split(":")
        profile = (t_, int(f_), int(n_))
    loop = DSparkLoop(gen, a.max_new, poll_every=a.poll_every)
    rep["loop"] = loop.describe()
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
    try:
        from draft_bench import pg61_tokens_per_chunk

        rep["pg61"] = pg61_tokens_per_chunk(a.pg61_step)
    except Exception:
        rep["pg61"] = None
    rep["hbm_end_GiB"] = _hbm(torch)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    slim = json.loads(json.dumps(rep, default=str))
    with open(json_path, "w") as f:
        json.dump(slim, f, indent=1)
    for line in summary_lines(rep):
        print(line)
    loop.close()
    return 0 if not rep["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
