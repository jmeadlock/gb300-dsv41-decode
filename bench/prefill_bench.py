#!/usr/bin/env python3
"""prefill_bench.py -- M4-P2: the grouped (bit-identical) prefill vs the reference prefill.

Handoff docs/HANDOFF-M4-P2-FAST-PREFILL.md; runtime runtime/prefill_fast.py ($DSV41_PREFILL=reference|grouped,
$DSV41_PREFILL_FILLS, $DSV41_PREFILL_STREAMS); design docs/M4-PREFILL-DESIGN.md (sections 7-8 carry the results).

Subcommands (labels: [M] measured by this tool, [D] derived from a receipt, [E] modeled, [A] assumption):

  run    BOX ONLY (GPU container). Boots like bench/headline.py (map seed-301, load_mtp, set_determinism(0): the
         headline's state), encodes the fixture prompts, warms both modes once (TileLang JIT, cuBLAS handles), then
         per prompt:
           * reference: --reps prefills exactly as today (mode reference, fills ON): prefill_hidden ms (synced),
             seed_draft_rings ms, first token, sha256 of the last row's fp32 logits and of main_hidden, and (first rep)
             the sha256 of every buffer of every main layer (window KV, compressed KV, indexer K, ratio-2 compressor
             kv_state / score_state) and of the draft stages' rings after seeding;
           * grouped, for every --streams count: --reps prefills (mode grouped, fills off = the new default): the same
             records + equality with the reference (first, logits, main_hidden; caches on the first rep), self-replay
             across reps, the counters of runtime/prefill_fast.py `last_prefill()` (expert calls, GEMM launches, cold
             calls, C2C bytes, padded rows);
           * --kineto: one more grouped prefill per streams count under torch.profiler (CUDA activity): kernels
             launched (total and fp4_gemm), GPU busy (union of kernel intervals) / prefill wall.
         TTFT [M] = prefill ms + seed ms + the first draft (2.9 ms [D], headline graph replay; not re-measured here).
         Receipt: --json (default scratch/prefill-bench-<date>.json).
  model  Mac: the modeled TTFT [E] of the grouped prefill per fixture prompt from the M4-P1 box receipt
         (docs/prefill-anatomy-20261108.json: census, roles, fills on/off ms) -- the summary line's [E].
  box    print the box command lines (GPU test on the mini, this bench on the real checkpoint, the 256-token headline
         streams with DSV41_PREFILL=grouped against today's receipt).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_MAP = os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-301.json")
DEFAULT_FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
ANATOMY_RECEIPT = os.path.join(REPO, "docs", "prefill-anatomy-20261108.json")
HEADLINE_RECEIPT = os.path.join(REPO, "docs", "headline-20261108-main-ss.json")
FIRST_DRAFT_MS = 2.9          # [D] headline: the first draft is one draft-graph replay (docs/M4-PREFILL-DESIGN.md 1)
BOX_W = "$HOME/dsv41/decode"
BOX_M = "/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170"
BOX_GPU = "nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55"

# model parameters ([D] from the M4-P1 receipts, [A] assumed; printed by `model`)
LAUNCHES_PER_EXPERT_REF = 26      # [D] nsys prose: 26.0 launches per reference Expert call
FP4_GEMM_US_HOT = 79.4            # [D] nsys prose: fp4_gemm GPU us per launch, hot expert
FP4_GEMM_US_COLD = 88.4           # [D] ... cold expert (UVA reads of Grace in-kernel)
HOST_US_PER_TORCH_OP = 12.0       # [D] M0 anatomy 14.4 us/launch host x 0.839 (fills off)
HOST_US_PER_GEMM_LAUNCH = 20.0    # [A] a direct TileLang kernel call from Python (adapter checks + launch)
STREAM_CONCURRENCY_CAP = 8        # [A] latency-bound fp4_gemm (16-32 CTAs each) overlaps ~linearly up to 8 streams


# ================================================================================================ helpers
def _sha_t(t) -> str:
    import torch

    return hashlib.sha256(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _sha_bytes(t) -> str:
    import torch

    u = t.detach()
    if u.is_complex():
        u = torch.view_as_real(u)
    u = u.contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(u.to("cpu").numpy().tobytes()).hexdigest()


def cache_digests(model) -> Dict[str, str]:
    """sha256 of the raw bytes of every buffer of every main layer and of the draft stages (after seeding)."""
    out = {f"layers.{n}": _sha_bytes(b) for n, b in model.layers.named_buffers()}
    mtp = getattr(model, "mtp", None)
    for i, st in enumerate(mtp or []):
        for n, b in st.named_buffers():
            out[f"mtp.{i}.{n}"] = _sha_bytes(b)
    return out


@contextlib.contextmanager
def _env(**kv):
    prev = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def prefill_run(model, ids: Sequence[int], hidden: Sequence[int], mode: str, fills: str, streams: int,
                device=None, caches: bool = False) -> dict:
    """One prefill as DSparkLoop.start runs it (prefill_hidden in `mode`, then seed_draft_rings), each synced and
    timed -> {prefill_ms, seed_ms, first, logits_sha, hidden_sha, counters[, caches]}."""
    import torch

    from runtime.draft_step import prefill_hidden, seed_draft_rings
    from runtime.prefill_fast import last_prefill

    dev = torch.device(device) if device is not None else model.embed.weight.device
    cuda = dev.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    tokens = torch.tensor([list(ids)], dtype=torch.long, device=dev)
    with _env(DSV41_PREFILL_FILLS=fills, DSV41_PREFILL_STREAMS=streams):
        sync()
        t0 = time.perf_counter()
        out, logits, mh = prefill_hidden(model, tokens, hidden, mode=mode)
        sync()
        t1 = time.perf_counter()
    cnt = last_prefill()
    seed_ms = None
    if getattr(model, "mtp", None) and mh is not None:
        seed_draft_rings(model, out.reshape(-1)[:1], mh)
        sync()
        seed_ms = (time.perf_counter() - t1) * 1e3
    lg = logits.reshape(-1, logits.size(-1))[-1]
    r = {"prefill_ms": (t1 - t0) * 1e3, "seed_ms": seed_ms, "first": int(out.reshape(-1)[0]),
         "logits_sha": _sha_t(lg), "hidden_sha": _sha_t(mh) if mh is not None else None,
         "counters": {k: v for k, v in cnt.items() if k != "calls_per_layer"},
         "calls_per_layer": cnt.get("calls_per_layer")}
    if caches:
        r["caches"] = cache_digests(model)
    return r


def kineto_prefill(model, ids, hidden, mode: str, fills: str, streams: int) -> dict:
    """One prefill under torch.profiler (CUDA activity): kernels, fp4_gemm kernels, launch calls, GPU busy / wall."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from runtime.draft_step import prefill_hidden

    dev = model.embed.weight.device
    tokens = torch.tensor([list(ids)], dtype=torch.long, device=dev)
    torch.cuda.synchronize(dev)
    with _env(DSV41_PREFILL_FILLS=fills, DSV41_PREFILL_STREAMS=streams):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            prefill_hidden(model, tokens, hidden, mode=mode)
            torch.cuda.synchronize(dev)
            wall = (time.perf_counter() - t0) * 1e3
    kern, launches, fp4 = [], 0, 0
    for e in prof.events():
        dt = str(getattr(e, "device_type", ""))
        if dt.endswith("CUDA") and getattr(e, "time_range", None) is not None:
            kern.append((e.time_range.start, e.time_range.end))
            fp4 += "fp4_gemm" in e.name
        elif e.name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx"):
            launches += 1
    busy = 0.0
    cur = None
    for s, t in sorted(kern):
        if cur is None or s > cur[1]:
            if cur is not None:
                busy += cur[1] - cur[0]
            cur = [s, t]
        else:
            cur[1] = max(cur[1], t)
    if cur is not None:
        busy += cur[1] - cur[0]
    return {"kernels": len(kern), "fp4_gemm_kernels": fp4, "launch_calls": launches, "gpu_busy_ms": busy / 1e3,
            "traced_wall_ms": wall, "gpu_busy_pct": 100.0 * busy / 1e3 / wall if wall else None}


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


# ================================================================================================ run (box)
def build_run_parser(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--only", default="all", help="comma-separated fixture tags ('all' = every prompt)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--streams", default="1,4,8", help="grouped: CUDA stream counts to run, e.g. 1,4,8")
    ap.add_argument("--caches", choices=["first", "none"], default="first",
                    help="sha256 every layer buffer after the first rep of each mode (gate 3)")
    ap.add_argument("--kineto", action="store_true", help="one profiled grouped prefill per streams count")
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--device", default="cuda", help=argparse.SUPPRESS)   # tests: cpu (emulated mini model)
    ap.add_argument("--json", default=None, help="default scratch/prefill-bench-<date>.json")


def run_main(a) -> int:
    import torch

    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load

    rep: Dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "errors": [],
                 "labels": "[M] measured on this box run; ttft adds the first draft 2.9 ms [D]"}
    json_path = a.json or os.path.join(REPO, "scratch", f"prefill-bench-{time.strftime('%Y%m%d-%H%M')}.json")
    t0 = time.time()
    mp = a.map if a.map and os.path.exists(a.map) else None
    model = build_and_load(a.model_dir, mp, a.device, config_path=a.config, reserve_host_gib=a.reserve_host_gib,
                           load_mtp=True)
    rep["boot"] = {"wall_s": time.time() - t0, "map": mp}
    rep["determinism"] = set_determinism(0)
    gen = Generator(model, a.model_dir, mode="eager", nvtx=False)
    mt = getattr(model, "runtime_mtp", None)
    hidden = tuple(int(i) for i in mt.target_layer_ids) if mt is not None else ()
    streams = [int(s) for s in str(a.streams).split(",") if s.strip()]
    caches = a.caches == "first"
    with open(a.fixture) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.only and a.only != "all":
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    rep["prompts"] = {}
    warmed = False
    for r in rows:
        tag = r["tag"]
        P: Dict = {"grouped": {}}
        rep["prompts"][tag] = P
        try:
            ids = gen.encode([{"role": "user", "content": r["prompt"]}])
            P["L"] = len(ids)
            if not warmed:
                prefill_run(model, ids, hidden, "reference", "on", 1, a.device)
                for s in streams:
                    prefill_run(model, ids, hidden, "grouped", "off", s, a.device)
                warmed = True
            ref = [prefill_run(model, ids, hidden, "reference", "on", 1, a.device, caches=caches and i == 0)
                   for i in range(max(1, a.reps))]
            P["reference"] = {"runs": [{k: v for k, v in x.items() if k != "caches"} for x in ref],
                              "prefill_ms_median": _median(x["prefill_ms"] for x in ref),
                              "seed_ms_median": _median(x["seed_ms"] for x in ref),
                              "self_replay_equal": len({(x["first"], x["logits_sha"], x["hidden_sha"])
                                                        for x in ref}) == 1}
            r0 = ref[0]
            for s in streams:
                g = [prefill_run(model, ids, hidden, "grouped", "off", s, a.device, caches=caches and i == 0)
                     for i in range(max(1, a.reps))]
                G = {"runs": [{k: v for k, v in x.items() if k not in ("caches", "calls_per_layer")} for x in g],
                     "prefill_ms_median": _median(x["prefill_ms"] for x in g),
                     "seed_ms_median": _median(x["seed_ms"] for x in g),
                     "equal_reference": all(x["first"] == r0["first"] and x["logits_sha"] == r0["logits_sha"]
                                            and x["hidden_sha"] == r0["hidden_sha"] for x in g),
                     "self_replay_equal": len({(x["first"], x["logits_sha"], x["hidden_sha"]) for x in g}) == 1,
                     "counters": g[0]["counters"], "calls_per_layer": g[0]["calls_per_layer"]}
                if caches:
                    diff = sorted(k for k in r0["caches"] if r0["caches"][k] != g[0]["caches"].get(k))
                    G["caches_equal"] = not diff and set(r0["caches"]) == set(g[0]["caches"])
                    G["caches_differing"] = diff[:16]
                    G["caches_compared"] = len(r0["caches"])
                else:
                    G["caches_equal"] = None
                pm, sm = G["prefill_ms_median"], G["seed_ms_median"] or 0.0
                G["ttft_ms"] = pm + sm + FIRST_DRAFT_MS
                if a.kineto and torch.device(a.device).type == "cuda":
                    try:
                        G["kineto"] = kineto_prefill(model, ids, hidden, "grouped", "off", s)
                    except Exception:  # noqa: BLE001
                        G["kineto"] = {"error": traceback.format_exc().splitlines()[-1]}
                P["grouped"][str(s)] = G
                c = G["counters"]
                print(f"[prefill-bench] {tag} L={P['L']} streams {s}: reference {P['reference']['prefill_ms_median']:.1f}"
                      f" ms -> grouped {pm:.1f} ms (TTFT {G['ttft_ms']:.1f} ms); equal {G['equal_reference']}, "
                      f"caches {G['caches_equal']}, replay {G['self_replay_equal']}; {c.get('expert_calls')} expert "
                      f"calls ({c.get('cold_calls')} cold) -> {c.get('gemm_launches')} GEMM launches "
                      f"(reference ~{LAUNCHES_PER_EXPERT_REF * (c.get('expert_calls') or 0)} expert launches); "
                      f"C2C {(c.get('c2c_bytes') or 0) / 1e9:.2f} GB"
                      + (f"; kineto {G['kineto']}" if "kineto" in G else ""), flush=True)
            if a.kineto and torch.device(a.device).type == "cuda" and not rep.get("_ref_kineto_done"):
                rep["_ref_kineto_done"] = True        # ~95k kernels to parse: the reference is profiled once
                try:
                    P["reference"]["kineto"] = kineto_prefill(model, ids, hidden, "reference", "on", 1)
                except Exception:  # noqa: BLE001
                    P["reference"]["kineto"] = {"error": traceback.format_exc().splitlines()[-1]}
        except Exception:  # noqa: BLE001
            rep["errors"].append({"stage": "prompt", "tag": tag, "traceback": traceback.format_exc()})
            print(f"[prefill-bench] {tag}: FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    rep["summary"] = summarize(rep)
    for line in summary_lines(rep["summary"]):
        print(line, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json.loads(json.dumps(rep, default=str)), f, indent=1)
    print(f"[prefill-bench] receipt {json_path}; errors {len(rep['errors'])}", flush=True)
    return 0 if not rep["errors"] and rep["summary"]["all_equal"] else 2


def summarize(rep: dict) -> dict:
    out: Dict = {"prompts": len(rep.get("prompts") or {}), "by_streams": {}}
    eq = []
    for P in (rep.get("prompts") or {}).values():
        for s, G in (P.get("grouped") or {}).items():
            b = out["by_streams"].setdefault(s, {"prefill_ms": [], "ttft_ms": [], "ref_ms": [], "gemm_launches": [],
                                                 "expert_calls": []})
            b["prefill_ms"].append(G["prefill_ms_median"])
            b["ttft_ms"].append(G["ttft_ms"])
            b["ref_ms"].append((P.get("reference") or {}).get("prefill_ms_median"))
            b["gemm_launches"].append(G["counters"].get("gemm_launches"))
            b["expert_calls"].append(G["counters"].get("expert_calls"))
            eq.append(bool(G["equal_reference"]) and G["caches_equal"] is not False and bool(G["self_replay_equal"]))
    for b in out["by_streams"].values():
        n = len(b["prefill_ms"])
        b["mean_prefill_ms"] = sum(b["prefill_ms"]) / n if n else None
        b["mean_ttft_ms"] = sum(b["ttft_ms"]) / n if n else None
        rr = [x for x in b["ref_ms"] if x is not None]
        b["mean_ref_ms"] = sum(rr) / len(rr) if rr else None
        b["mean_gemm_launches"] = sum(b["gemm_launches"]) / n if n else None
        b["mean_expert_launches_before"] = LAUNCHES_PER_EXPERT_REF * sum(b["expert_calls"]) / n if n else None
    out["all_equal"] = bool(eq) and all(eq)
    return out


def summary_lines(s: dict) -> List[str]:
    lines = [f"[summary] {s['prompts']} prompts; all bit-equal (first, logits, main_hidden, caches, replay): "
             f"{s['all_equal']}"]
    for k, b in sorted(s["by_streams"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"[summary] streams {k}: prefill {b['mean_ref_ms']:.0f} -> {b['mean_prefill_ms']:.0f} ms, TTFT "
                     f"{b['mean_ttft_ms']:.0f} ms [M]; expert launches ~{b['mean_expert_launches_before']:.0f} [D] "
                     f"-> {b['mean_gemm_launches']:.0f} GEMM launches [M]")
    return lines


# ================================================================================================ model (Mac)
def model_report(path: str = ANATOMY_RECEIPT) -> dict:
    """The grouped prefill's TTFT [E] per fixture prompt, from the M4-P1 box receipt (module docstring)."""
    with open(path) as f:
        d = json.load(f)
    params = {
        "launches_per_expert_ref": (LAUNCHES_PER_EXPERT_REF, "[D] nsys prose"),
        "fp4_gemm_us_hot": (FP4_GEMM_US_HOT, "[D] nsys prose"), "fp4_gemm_us_cold": (FP4_GEMM_US_COLD, "[D] nsys prose"),
        "host_us_per_torch_op": (HOST_US_PER_TORCH_OP, "[D] M0 anatomy, fills off"),
        "host_us_per_gemm_launch": (HOST_US_PER_GEMM_LAUNCH, "[A]"),
        "stream_concurrency_cap": (STREAM_CONCURRENCY_CAP, "[A]"), "first_draft_ms": (FIRST_DRAFT_MS, "[D] headline"),
    }
    from runtime.prefill_fast import GLUE_OPS_FIXED

    out: Dict = {"source": path, "params": {k: {"value": v, "label": l} for k, (v, l) in params.items()},
                 "prompts": {}}
    for tag, P in (d.get("prompts") or {}).items():
        h = P.get("hooked") or {}
        cen, roles = h.get("census") or {}, h.get("roles") or {}
        if not cen or not roles:
            continue
        on, off = P["fills_on_prefill_ms_median"], P["fills_off_prefill_ms_median"]
        fr = off / on
        seed = P.get("fills_off_seed_ms_median") or 0.0
        nl = len(cen["calls_per_layer"])
        k = int(cen["topk"])
        calls, cold = int(cen["expert_calls"]), int(cen["cold_calls"])
        g = lambda r: (roles.get(r) or {}).get("host_ms", 0.0)                       # noqa: E731
        nonmoe = (g("L*") - g("moe") + g("head")) * fr          # attention, compressor, indexer, mHC, Engram, head
        fixed = (g("gate") + g("shared")) * fr                  # the MoE's gate + shared expert, unchanged
        glue = nl * (GLUE_OPS_FIXED + 2 * k) * HOST_US_PER_TORCH_OP / 1e3
        gemm_n = 3 * calls
        gemm_host = gemm_n * HOST_US_PER_GEMM_LAUNCH / 1e3
        gemm_gpu1 = 3 * ((calls - cold) * FP4_GEMM_US_HOT + cold * FP4_GEMM_US_COLD) / 1e3

        def ttft(s):
            return nonmoe + fixed + glue + max(gemm_host, gemm_gpu1 / min(s, STREAM_CONCURRENCY_CAP)) + seed \
                + FIRST_DRAFT_MS

        out["prompts"][tag] = {
            "L": P["L"], "expert_calls": calls, "cold_calls": cold, "c2c_GB": cen.get("c2c_bytes", 0) / 1e9,
            "launches": {"expert_before": LAUNCHES_PER_EXPERT_REF * calls, "gemm_after": gemm_n,
                         "glue_after": nl * (GLUE_OPS_FIXED + 2 * k)},
            "terms_ms": {"nonmoe": nonmoe, "gate_shared": fixed, "glue": glue, "gemm_host": gemm_host,
                         "gemm_gpu_1stream": gemm_gpu1, "seed": seed},
            "ttft_ms": {"today_fills_on": on + (P.get("fills_on_seed_ms_median") or 0.0) + FIRST_DRAFT_MS,
                        "fills_off": off + seed + FIRST_DRAFT_MS,
                        "grouped_s1": ttft(1), "grouped_s4": ttft(4), "grouped_s8": ttft(8)}}
    return out


def model_lines(rep: dict) -> List[str]:
    lines = [f"M4-P2 modeled TTFT of the grouped prefill [E], from {os.path.relpath(rep['source'], REPO)} [D]"]
    for k, v in rep["params"].items():
        lines.append(f"  param {k:26s} {v['value']:>8} {v['label']}")
    lines.append(f"  {'prompt':6s} {'L':>3s} {'calls':>5s} {'cold':>4s} {'launch ref':>10s} {'gemm':>6s} "
                 f"{'today':>7s} {'fills-off':>9s} {'grp s1':>7s} {'grp s4':>7s} {'grp s8':>7s}   (ms; [D] [D] [E] [E] [E])")
    for tag, p in sorted(rep["prompts"].items(), key=lambda kv: kv[1]["L"]):
        t = p["ttft_ms"]
        lines.append(f"  {tag:6s} {p['L']:>3d} {p['expert_calls']:>5d} {p['cold_calls']:>4d} "
                     f"{p['launches']['expert_before']:>10d} {p['launches']['gemm_after']:>6d} "
                     f"{t['today_fills_on']:>7.0f} {t['fills_off']:>9.0f} {t['grouped_s1']:>7.0f} "
                     f"{t['grouped_s4']:>7.0f} {t['grouped_s8']:>7.0f}")
    ps = list(rep["prompts"].values())
    if ps:
        m = lambda key: sum(p["ttft_ms"][key] for p in ps) / len(ps)              # noqa: E731
        lines.append(f"  mean   today {m('today_fills_on'):.0f} [D] | fills off {m('fills_off'):.0f} [D] | grouped "
                     f"1 stream {m('grouped_s1'):.0f} [E] | 4 streams {m('grouped_s4'):.0f} [E] | 8 streams "
                     f"{m('grouped_s8'):.0f} [E]")
    return lines


# ================================================================================================ box lines
def box_lines() -> List[str]:
    d = (f'D="docker run --rm --network none --device {BOX_GPU} --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK '
         f'-v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 '
         f'-v $HOME/dsv41/w0/jit-cache/tilelang:/root/.tilelang -v $HOME/dsv41/w0/jit-cache/triton:/root/.triton '
         f'-v $HOME/dsv41/w0/jit-cache/nv:/root/.nv -v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm '
         f'-e DG_JIT_CACHE_DIR=/root/.deep_gemm -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json"')
    hl = ("bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl "
          "flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency-drain 64 --max-new 256")
    return [
        "# box: Station, clone ~/dsv41/decode on branch m4-p2 (git pull first); refuse if a ds41/mk- container is up",
        f"W={BOX_W}; M={BOX_M}",
        "docker ps --format '{{.Names}}' | grep -q 'ds41\\|mk-' && echo refusing && exit 1",
        "mkdir -p $W/scratch",
        d,
        "# 1) gate on the mini (real kernels): MoE per layer, prefill + every cache, loop streams (~3 min)",
        "$D --name mk-m4p2-test --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_prefill_fast_gpu.py "
        "> $W/scratch/m4p2-gpu-test.log 2>&1",
        "# 2) the real checkpoint, 8 fixture prompts: reference vs grouped at 1/4/8 streams (logits, main_hidden,",
        "#    every layer buffer + draft rings, replay, counters), one kineto-profiled prefill each (~8 min incl. boot)",
        "$D --name mk-m4p2-bench --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/prefill_bench.py run "
        "--model-dir /ref --only all --reps 3 --streams 1,4,8 --kineto --json /work/scratch/prefill-bench-m4p2.json "
        "> $W/scratch/prefill-bench-m4p2.log 2>&1",
        "# 3) the 256-token streams after the grouped prefill vs today's receipt (tokens_sha256 + accepts per prompt:",
        "#    the residency_ref block's all_tokens_equal / all_accepts_equal; prefill_ms per prompt) (~12 min)",
        f"$D --name mk-m4p2-headline -e DSV41_PREFILL=grouped -e DSV41_PREFILL_STREAMS=4 --entrypoint python3 "
        f"vllm/vllm-openai:v0.31.0 {hl} --residency-ref docs/headline-20261108-main-ss.json "
        f"--json scratch/headline-m4p2-grouped.json > $W/scratch/headline-m4p2-grouped.log 2>&1",
        "# copy scratch/m4p2-gpu-test.log, prefill-bench-m4p2.{json,log}, headline-m4p2-grouped.{json,log} to the Mac",
        "# docs/ (never write docs/ on the box); flip DEFAULT_MODE in runtime/prefill_fast.py only if all three pass",
    ]


# ================================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    build_run_parser(sub.add_parser("run", help="box: reference vs grouped prefills (GPU)"))
    mo = sub.add_parser("model", help="Mac: modeled TTFT of the grouped prefill")
    mo.add_argument("--receipt", default=ANATOMY_RECEIPT)
    mo.add_argument("--json", default=None)
    sub.add_parser("box", help="print the box command lines")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run_main(a)
    if a.cmd == "model":
        rep = model_report(a.receipt)
        for line in model_lines(rep):
            print(line)
        if a.json:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            with open(a.json, "w") as f:
                json.dump(rep, f, indent=1)
        return 0
    for line in box_lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
