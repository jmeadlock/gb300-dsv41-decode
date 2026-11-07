#!/usr/bin/env python3
"""draft_bench.py -- M1-T11 bench hook: ms per draft call (graph) and draft acceptance on the W0 fixture.

  python3 bench/draft_bench.py --model-dir /models/DeepSeek-V4.1-... [--map docs/maps/hot-v26ar-seed-295.json] \
      --fixture w0/results/phase1/parity-A.jsonl --max-new 256 [--moe-impl dg-inplace --fp8-impl gemv \
      --glue-impl fused --attn-impl flashmla] [--json docs/draft-bench-<date>.json]

Boot with `load_mtp=True` (runtime/mtp_loader.py; the report carries its bytes). Then, per fixture prompt:

  (1) truth: the 1-row greedy decode (`Generator.greedy`, --mode, the given impls) -> token_ids, i.e. targets[i] = the
      token at position L + i (the 1-row greedy records are the truth the drafts are graded against);
  (2) teacher-forced drafting: the reference prefill with main_hidden (layers dspark_target_layer_ids, model.py:
      1267-1269) + the reference draft-ring seeding (forward_spec at 0); the first draft at A' = L-1 with ctx_count 0;
      then the 6-row verify step (StaticStep rows=6, hidden_layers = the targets) fed targets[6s..6s+5] at
      P0 = L + 6s (every row is truth, so every row is a committed position), and after each verify SIX drafts,
      A' = P0 + j with ctx_count = j + 1 (j = 0..5; the bonus t_n = targets[A'+1-L]): a draft at EVERY position, the
      rings written exactly once per position (the 6 calls write prefixes of the same 6 rows, idempotently);
  (3) acceptance: n(A') = how many leading drafts d'_k equal targets[A'+1+k-L] (k = 1..5, positions A'+2..A'+6;
      truncated where the truth ends). Reported: mean n over all positions, the histogram, and the simulated loop
      (A' -> A' + 1 + n(A') from A' = L-1, design 1.3): tokens/step = mean(1 + n) along that path.

Then the draft graph is timed: CUDA events around `DraftStep.replay` (ctx_count 6, the max context write), N replays
after warm-up -> mean / p50 / p90 ms per draft call; plus the in-loop replays' event times. No loop yet (M1-T12):
the verify argmax is not used to accept anything here; truth is the 1-row record.

M1-T11-PORT: `--draft-impls same` (default) builds the DraftStep on the verify's impls mapped by
runtime/draft_step.py `draft_impls_for` (--fp8-impl gemv -> GEMV, --moe-impl dg-* -> DeepGEMM over the resident
stacks, --glue-impl fused -> the draft's fused clusters, --attn-impl flashmla -> FlashMLA); `--draft-impls ref` keeps
the T11 reference draft. The report's `draft_step` is DraftStep.describe() (which clusters / arm actually came on).

Production's numbers for context (not ours): Pg61's stream delivered `completion_tokens / chunks` tokens per step on
the W0 step prompt (`--pg61-step w0/results/phase1/step-pg61.json`, read and printed if the file exists; the design memo
1.3 quotes 2.75 / 2.63 / 2.93), with probabilistic drafts -- greedy drafting here is a different policy.
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

BLOCK = 5
ROWS = 6


# ------------------------------------------------------------------------------------------------ pure logic
def accepted(drafts: Sequence[int], targets: Sequence[int], a_pos: int, L: int) -> int:
    """Leading drafts that match the truth: d_k (k = 1..len) is the draft for position a_pos + 1 + k, whose truth is
    targets[a_pos + 1 + k - L]; stops at the first mismatch or where the truth ends."""
    n = 0
    for k, d in enumerate(drafts, start=1):
        i = a_pos + 1 + k - L
        if i >= len(targets) or int(d) != int(targets[i]):
            break
        n += 1
    return n


def simulate_loop(per_pos: Dict[int, int], a0: int, a_last: int) -> dict:
    """The speculative loop's path over per-position accept counts: from A' = a0, each step emits 1 + n(A') tokens and
    moves A' to A' + 1 + n(A'); stops once A' passes a_last (the last committed position with truth) or has no
    draft recorded."""
    a, accepts = a0, []
    while a <= a_last and a in per_pos:
        n = per_pos[a]
        accepts.append(n)
        a += 1 + n
    steps = len(accepts)
    tokens = sum(1 + n for n in accepts)
    return {"steps": steps, "tokens": tokens, "tokens_per_step": tokens / steps if steps else None,
            "accepts": accepts}


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


def pg61_tokens_per_chunk(path: str) -> Optional[dict]:
    """step-pg61.json's reps -> completion_tokens / chunks per rep (what the design memo calls tokens/step)."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d.get("reps", []):
        ct = (r.get("usage") or {}).get("completion_tokens")
        ch = r.get("chunks")
        if ct and ch:
            out.append(round(ct / ch, 4))
    return {"path": path, "tokens_per_chunk": out}


# ------------------------------------------------------------------------------------------------ one prompt
@torch.inference_mode()
def draft_prompt(gen, ds, ids: List[int], targets: List[int], hidden: Sequence[int], graph: bool) -> dict:
    """(2)+(3) for one prompt -> per-position accept counts, the simulated loop, in-loop draft-call event times."""
    from runtime.draft_step import prefill_hidden, seed_draft_rings

    L = len(ids)
    dev = gen.device
    st = gen.static_step()
    if gen.mode == "graph" and not st.captured:
        st.capture()
    out, _, mh = prefill_hidden(gen.model, torch.tensor([ids], device=dev), hidden)
    first = out.reshape(-1)[:1]
    seed_draft_rings(gen.model, first, mh)
    n_steps = (len(targets) - 1) // ROWS
    feed = torch.tensor(targets[: ROWS * n_steps + 1], dtype=torch.long, device=dev)
    per_pos: Dict[int, int] = {}
    drafts_at: Dict[int, torch.Tensor] = {}
    evs = []

    def draft(a, c, tok):
        if graph:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ds.set_inputs(a, c, tok=tok)
            s.record()
            ds.graph.replay()
            e.record()
            evs.append((s, e))
        else:
            ds.step(a, c, tok=tok)
        drafts_at[a] = ds.drafts.clone()

    draft(L - 1, 0, first)                                           # the first draft after the prefill
    for s_ in range(n_steps):
        p0 = L + ROWS * s_
        gen.step_rows(feed[ROWS * s_: ROWS * s_ + ROWS], p0)        # teacher-forced 6-row verify -> main_hidden
        for j in range(ROWS):                                        # a draft at every committed position
            a = p0 + j
            if a + BLOCK > ds.max_pos:
                break
            draft(a, j + 1, feed[ROWS * s_ + 1: ROWS * s_ + ROWS + 1])  # tok[j] = targets[a + 1 - L] = the bonus
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    for a, d in drafts_at.items():
        per_pos[a] = accepted(d.tolist(), targets, a, L)
    a_last = L + len(targets) - 2  # the last position whose next token has a truth
    sim = simulate_loop(per_pos, L - 1, a_last)
    hist = [0] * (BLOCK + 1)
    for n in per_pos.values():
        hist[n] += 1
    ms = [s.elapsed_time(e) for s, e in evs]
    return {"prompt_tokens": L, "targets": len(targets), "positions": len(per_pos),
            "mean_accepted_per_position": sum(per_pos.values()) / max(1, len(per_pos)), "hist": hist,
            "loop": {k: v for k, v in sim.items() if k != "accepts"}, "loop_accepts": sim["accepts"],
            "draft_ms_in_loop": _stats(ms), "draft_ms_list": ms}


def time_graph(ds, a_pos: int, n: int = 200, warm: int = 20) -> dict:
    """CUDA events around N back-to-back replays at ctx_count 6 (rings are re-written with the same rows)."""
    ds.set_inputs(a_pos, 6)
    for _ in range(warm):
        ds.graph.replay()
    torch.cuda.synchronize(ds.dev)
    evs = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        ds.graph.replay()
        e.record()
        evs.append((s, e))
    torch.cuda.synchronize(ds.dev)
    t0 = time.perf_counter()
    for _ in range(n):
        ds.graph.replay()
    torch.cuda.synchronize(ds.dev)
    wall = (time.perf_counter() - t0) / n * 1e3
    st = _stats([s.elapsed_time(e) for s, e in evs])
    st["back_to_back_wall_ms"] = wall
    return st


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
    ap.add_argument("--mode", choices=["static", "graph"], default="graph", help="the 1-row truth + 6-row verify")
    ap.add_argument("--draft-mode", choices=["eager", "graph"], default="graph")
    ap.add_argument("--moe-impl", choices=["ref", "dg-masked", "dg-inplace"], default="ref")
    ap.add_argument("--fp8-impl", choices=["ref", "gemv"], default="ref")
    ap.add_argument("--glue-impl", choices=["ref", "fused"], default="ref")
    ap.add_argument("--attn-impl", choices=["ref", "bounded", "flashmla"], default="ref")
    ap.add_argument("--draft-impls", choices=["same", "ref"], default="same",
                    help="same = the verify's impls mapped by draft_step.draft_impls_for; ref = the T11 reference draft")
    ap.add_argument("--time-replays", type=int, default=200)
    ap.add_argument("--pg61-step", default=os.path.join(REPO, "w0", "results", "phase1", "step-pg61.json"))
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--json", default=None, help="default docs/draft-bench-<YYYYMMDD>.json")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from runtime.draft_step import DraftStep, draft_impls_for
    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load

    date = time.strftime("%Y%m%d")
    json_path = a.json or os.path.join(REPO, "docs", f"draft-bench-{date}.json")
    report: Dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
                    "errors": []}
    t0 = time.time()
    mp = a.map if a.map and os.path.exists(a.map) else None
    model = build_and_load(a.model_dir, mp, "cuda", hot_per_layer=a.hot_per_layer, config_path=a.config,
                           reserve_host_gib=a.reserve_host_gib, load_mtp=True)
    report["boot"] = model.runtime_load.summary()
    report["boot"]["wall_s"] = time.time() - t0
    report["mtp"] = model.runtime_mtp.summary()
    report["determinism"] = set_determinism(0)
    hidden = list(model.runtime_mtp.target_layer_ids)
    gen = Generator(model, a.tokenizer_dir or a.model_dir, mode=a.mode, max_pos=a.max_pos, moe_impl=a.moe_impl,
                    fp8_impl=a.fp8_impl, glue_impl=a.glue_impl, attn_impl=a.attn_impl)
    if a.moe_impl == "dg-inplace" and getattr(gen, "pool", None) is not None:
        # grow the pool to the 6-row spare budget ONCE before any StaticStep build (re-growth after the 1-row build
        # keeps the old blocks referenced -> +24 GiB -> OOM; see first_light rows mode)
        from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS
        b = gen.pool.add_spare_slots(DEFAULT_SPARE_SLOTS_ROWS, model)
        print(f"[pool] pre-grown to {gen.pool.spare} spare slots/layer (+{b / 2**30:.2f} GiB)", flush=True)
    with open(a.fixture) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]

    # (1) truth at 1 row
    truth: Dict[str, dict] = {}
    for r in rows:
        try:
            ids = gen.encode([{"role": "user", "content": r["prompt"]}])
            gen.greedy(ids, a.max_new, tag=r["tag"], prompt=r["prompt"])
            truth[r["tag"]] = {"ids": ids, "targets": list(gen.last_meta["token_ids"])}
            print(f"[truth] {r['tag']:6s} prompt {len(ids)} -> {len(truth[r['tag']]['targets'])} tokens", flush=True)
        except Exception:
            report["errors"].append({"stage": "truth", "tag": r["tag"], "traceback": traceback.format_exc()})
            print(f"[truth] {r['tag']}: FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)

    # (2)+(3) at 6 rows
    gen.reset_static()
    gen.hidden_layers = tuple(hidden)   # the verify step's main_hidden (static_step.py hidden_layers)
    gen.set_rows(ROWS)
    st = gen.static_step()
    if a.mode == "graph":
        st.capture()
    free0 = torch.cuda.mem_get_info()[0]
    if st.main_hidden is None:
        raise RuntimeError("the 6-row StaticStep has no main_hidden (hidden_layers empty?)")
    dimpls = draft_impls_for(a.moe_impl, a.fp8_impl, a.glue_impl, a.attn_impl) if a.draft_impls == "same" else {}
    report["draft_impls"] = dimpls
    ds = DraftStep(model, main_hidden=st.main_hidden, **dimpls)  # the verify graph's output buffer IS the draft's input
    report["draft_step"] = ds.describe()
    if a.draft_mode == "graph":
        report["draft_capture"] = ds.capture()
    report["hbm_after_draft_build_GiB"] = {"free": round(torch.cuda.mem_get_info()[0] / 2**30, 3),
                                           "free_before_draft": round(free0 / 2**30, 3),
                                           "allocated": round(torch.cuda.memory_allocated() / 2**30, 3)}
    per_prompt: Dict[str, dict] = {}
    for tag, t in truth.items():
        try:
            res = draft_prompt(gen, ds, t["ids"], t["targets"], hidden, a.draft_mode == "graph")
            per_prompt[tag] = res
            lp = res["loop"]
            print(f"[accept] {tag:6s} {res['positions']} drafts: mean accepted/position "
                  f"{res['mean_accepted_per_position']:.3f}, hist {res['hist']}; loop {lp['steps']} steps "
                  f"{lp['tokens']} tok -> {lp['tokens_per_step'] or 0:.3f} tok/step", flush=True)
        except Exception:
            report["errors"].append({"stage": "accept", "tag": tag, "traceback": traceback.format_exc()})
            print(f"[accept] {tag}: FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    report["acceptance"] = per_prompt
    pos = sum(r["positions"] for r in per_prompt.values())
    acc = sum(r["mean_accepted_per_position"] * r["positions"] for r in per_prompt.values())
    steps = sum(r["loop"]["steps"] for r in per_prompt.values())
    toks = sum(r["loop"]["tokens"] for r in per_prompt.values())
    hist = [sum(r["hist"][i] for r in per_prompt.values()) for i in range(BLOCK + 1)]
    report["acceptance_total"] = {"prompts": len(per_prompt), "positions": pos,
                                  "mean_accepted_per_position": acc / pos if pos else None, "hist": hist,
                                  "loop_steps": steps, "loop_tokens": toks,
                                  "loop_tokens_per_step": toks / steps if steps else None}
    # timing
    if a.draft_mode == "graph":
        try:
            a_pos = min(1000, ds.max_pos - BLOCK)
            report["draft_ms"] = time_graph(ds, a_pos, a.time_replays)
        except Exception:
            report["errors"].append({"stage": "time", "traceback": traceback.format_exc()})
    in_loop = [x for r in per_prompt.values() for x in r.pop("draft_ms_list", [])]
    report["draft_ms_in_loop"] = _stats(in_loop)
    report["pg61"] = pg61_tokens_per_chunk(a.pg61_step)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    at, dm = report["acceptance_total"], report.get("draft_ms") or {}
    f2 = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    print(f"DRAFT BENCH | draft impls {report.get('draft_impls') or 'ref'} (glue on "
          f"{(report.get('draft_step') or {}).get('glue', {}).get('on')}, attn on "
          f"{((report.get('draft_step') or {}).get('attn') or {}).get('on')}) | mtp HBM "
          f"{report['mtp']['bytes_GiB'].get('hbm_total')} GiB | draft graph "
          f"{f2(dm.get('mean'))} ms/call (p50 {f2(dm.get('p50'))}, p90 {f2(dm.get('p90'))}, back-to-back "
          f"{f2(dm.get('back_to_back_wall_ms'))}) | acceptance {at['prompts']} prompts {at['positions']} positions: "
          f"{f2(at['mean_accepted_per_position'])} accepted drafts/position, loop "
          f"{f2(at['loop_tokens_per_step'])} tokens/step | Pg61 tokens/chunk "
          f"{(report['pg61'] or {}).get('tokens_per_chunk')} | errors {len(report['errors'])} | {json_path}")
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
