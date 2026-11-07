#!/usr/bin/env python3
"""mixes_fork_bench.py -- M2-L1 instrument (docs/HANDOFF-M2-L1-MIXES-FORK.md): the mHC mixes off the critical path.

Two modes.

  census (CPU, the Mac): the per-step launch census of bench/glue_census.py on its 40-layer real-structure synthetic
  checkpoint, today's fused path (`static-dg-gemv-fused`, $DSV41_FP32_IMPL as given, default fixed = the headline),
  for $DSV41_MIXES off and fork, at the given row counts. Every launch inside MixesFork.run is tagged `mixes.branch`,
  so the report gives launches/step on the step's stream before -> after (the critical path, as launches) and the
  branch's launches per sublayer. The T9b-5 fixed dot (CPU definition: many torch ops) counts as ONE kernel named per
  site, as on the GPU. Caveat (glue_census's): an aten op is one launch; cuBLAS's F.linear at M=1 is 2 kernels on the
  GPU (dot + reduce_1Block, docs/M2-FUSED-CHAIN-DESIGN.md 1, S1) -- only the cublas census is affected.

      .venv/bin/python bench/mixes_fork_bench.py census [--rows 1,6] [--fp32-impl fixed] [--steps 2] \\
          [--json scratch/mixes-fork-census.json]

  gpu (the box, the orchestrator runs it): synthetic checkpoints on cuda (`mini4` = the GPU tests' 4-layer mini,
  `real40` = glue_census's 40-layer real structure at tiny dims: the real launch structure, 80 sublayers), the fast
  path (moe dg-inplace, fp8 gemv, glue fused, attn flashmla, $DSV41_FP32_IMPL default fixed) at each row count, one
  StaticStep per mixes impl (off, fork, fused):
    * bit-identity: static-eager and graph-replay logits of every impl vs `off` (torch.equal, teacher-forced tokens),
      replay twice; `fused` is reported vs off (== off under fp32 fixed; A3 vs cuBLAS under cublas);
    * the graph's branch (CUDAGraph debug mode -> cudaGraphDebugDotPrint): nodes, edges, fork nodes (out-degree >=
      2), join nodes (in-degree >= 2), the longest path in nodes (the critical path as launches) and the names of the
      nodes off it -- with `off` the graph is a chain (longest path == nodes);
    * ms/step: graph replays timed with CUDA events, impls interleaved per rep, median over reps (`--mode graph`), or
      the static-eager step (`--mode eager`: the fork is real there too, two streams);
    * SM contention: torch.profiler kernel durations per family (gemv, branch kernels, the rest) per step, off vs fork
      (`--profile`): the branch must not slow the main path's GEMVs;
    * `--nsys`: cudaProfilerStart/Stop around the timed window and a `decode_step` NVTX range per step, so
          nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -o <out> \\
              python3 bench/mixes_fork_bench.py gpu --nsys --mixes off,fork ...
      gives bench/step_anatomy.py's input (per-kernel mean µs with and without the branch).

      python3 bench/mixes_fork_bench.py gpu [--kinds mini4,real40] [--rows 6,1] [--mixes off,fork,fused] \\
          [--mode graph] [--reps 7] [--iters 32] [--profile] [--nsys] [--json scratch/mixes-fork-bench.json]

  Same docker line as tests/gpu/test_static_rows_gpu.py (repo at $W, `--entrypoint python3 vllm/vllm-openai:v0.31.0
  bench/mixes_fork_bench.py gpu ...`). The real model needs no bench of its own: $DSV41_MIXES reaches every StaticStep
  build, so `DSV41_MIXES=fork python3 bench/headline.py <today's argv>` is the real-checkpoint A/B, and the same env
  under nsys with bench/step_anatomy.py is the box's contention check.

No numbers are invented: everything printed is measured by this run; the modeled box saving lives in the summary
lines labelled [E] and is computed from the receipts cited there.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench")
TESTS = os.path.join(REPO, "tests")
for p in (REPO, BENCH, TESTS, os.path.join(TESTS, "gpu")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

MAX_POS = 511
FAST = dict(moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused", attn_impl="flashmla")
BRANCH_NAMES = ("t9b5_mix_fp32", "sinkhorn")   # kernels the fork moves (fp32 fixed); cuBLAS's mixes mm is ambiguous
# [M] receipts the modeled box saving is built from (docs/fp32-fixed-gpu-test-20261107.log lines 33/41: the T9b-5
# mixes kernel at its default config; docs/STEP-ANATOMY-R2-20261107.md: hc_split_sinkhorn 4.4 µs mean at 1 row)
RECEIPT_US = {"t9b5_mix_fp32 R=1": 5.27, "t9b5_mix_fp32 R=6": 5.41, "hc_split_sinkhorn R=1": 4.4}


@contextlib.contextmanager
def env(**kv):
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


def modeled_box_ms(sublayers: int = 80, hidden=(0.6, 1.0)) -> dict:
    """[E] the box saving per verify step if the branch is hidden 60-100 %: sublayers x (mix kernel + sinkhorn) at
    the receipts' µs (RECEIPT_US). The 6-row sinkhorn is taken at its 1-row mean (one CTA per row: [A])."""
    out = {}
    for r, mix in ((1, RECEIPT_US["t9b5_mix_fp32 R=1"]), (6, RECEIPT_US["t9b5_mix_fp32 R=6"])):
        per = mix + RECEIPT_US["hc_split_sinkhorn R=1"]
        out[f"rows {r}"] = {"branch_us_per_sublayer": round(per, 2),
                            "ms_lo": round(sublayers * per * hidden[0] / 1e3, 3),
                            "ms_hi": round(sublayers * per * hidden[1] / 1e3, 3)}
    out["basis"] = ("[E] 80 sublayers x (t9b5_mix_fp32 5.27/5.41 us [M fp32-fixed-gpu-test-20261107.log] + "
                    "hc_split_sinkhorn 4.4 us [M STEP-ANATOMY-R2-20261107.md]) hidden 60-100 %")
    return out


# ------------------------------------------------------------------------------------------------ census (CPU)
def _fixed_dot_kernel(GC, fn):
    names = {"mix": "t9b5_mix_fp32", "head": "t9b5_head_fp32", "comp": "t9b8_comp_fp32"}

    def w(*a, **k):
        GC.C.add("kernel", "K:" + names.get(k.get("site"), "t9b5_fixed_dot"))
        GC.C.paused += 1
        try:
            return fn(*a, **k)
        finally:
            GC.C.paused -= 1
    return w


def census_events(events, steps: int) -> dict:
    """launches (launch + kernel) per step: total, inside the branch (`mixes.branch` on the stack), by op name."""
    tot = br = 0
    names = collections.Counter()
    for stack, cls, name in events:
        if cls not in ("launch", "kernel"):
            continue
        tot += 1
        if "mixes.branch" in stack:
            br += 1
            names[name] += 1
    return {"launches_per_step": round(tot / steps, 2), "branch_per_step": round(br / steps, 2),
            "main_per_step": round((tot - br) / steps, 2),
            "branch_ops_per_step": {k: round(v / steps, 2) for k, v in sorted(names.items())}}


def run_census(rows_list: List[int], steps: int = 2, fp32: str = "fixed", log=print) -> dict:
    import glue_census as GC
    from runtime import fused_glue_r2 as R2
    from runtime import static_step as SS
    from test_dg_moe_cpu import _fake_deep_gemm

    out = {"fp32_impl": fp32, "steps": steps, "rows": {}}
    for rows in rows_list:
        with tempfile.TemporaryDirectory() as tmp, env(DSV41_FP32_IMPL=fp32):
            R, model, pool, cfg = GC.build(tmp)
            fake = _fake_deep_gemm()
            fn = fake.m_grouped_fp8_fp4_gemm_nt_masked
            if rows > 1:
                from test_dg_inplace_cpu import _fake_dg_contig
                from test_static_rows_cpu import _masked_rowwise

                fake = _fake_dg_contig()
                fn = _masked_rowwise(fake)
            fake.m_grouped_fp8_fp4_gemm_nt_masked = GC.kernel(fn, "dg_masked_gemm", lambda *x, **y: [])
            prev_dg = sys.modules.get("deep_gemm")
            sys.modules["deep_gemm"] = fake
            restore = GC.install(R)
            saved = [(R2, "launch_fixed_dot", R2.launch_fixed_dot), (SS.MixesFork, "run", SS.MixesFork.run)]
            R2.launch_fixed_dot = _fixed_dot_kernel(GC, R2.launch_fixed_dot)
            SS.MixesFork.run = GC.tagged(SS.MixesFork.run, "mixes.branch")
            prev = torch.get_default_dtype()
            try:
                ids = list(range(40, 56))
                rec = {}
                for impl in ("off", "fork"):
                    with env(DSV41_MIXES=impl):
                        rep = GC.run_path("static-dg-gemv-fused", R, model, pool, ids, steps, rows=rows)
                    c = census_events(list(GC.C.events), steps)
                    c["glue_on"] = (rep.get("fused_glue") or {}).get("clusters")
                    rec[impl] = c
                    log(f"  census rows {rows} mixes {impl:4s}: launches/step {c['launches_per_step']:7.1f}, on the "
                        f"step's stream {c['main_per_step']:7.1f}, on the branch {c['branch_per_step']:6.1f} "
                        f"{c['branch_ops_per_step']}")
            finally:
                for obj, attr, val in reversed(saved):
                    setattr(obj, attr, val)
                restore()
                torch.set_default_dtype(prev)
                if prev_dg is None:
                    sys.modules.pop("deep_gemm", None)
                else:
                    sys.modules["deep_gemm"] = prev_dg
            n_sub = 2 * cfg["n_layers"]
            off, fk = rec["off"], rec["fork"]
            rec["summary"] = {
                "sublayers": n_sub,
                "critical_path_launches_before": off["main_per_step"],
                "critical_path_launches_after": fk["main_per_step"],
                "moved_off_path": round(off["main_per_step"] - fk["main_per_step"], 2),
                "branch_launches_per_sublayer": round(fk["branch_per_step"] / n_sub, 3),
                "total_launches_equal": off["launches_per_step"] == fk["launches_per_step"]}
            log(f"  census rows {rows}: critical path {off['main_per_step']:.1f} -> {fk['main_per_step']:.1f} "
                f"launches/step ({rec['summary']['moved_off_path']:.1f} moved, "
                f"{rec['summary']['branch_launches_per_sublayer']:.2f} per sublayer x {n_sub})")
            out["rows"][str(rows)] = rec
    return out


# ------------------------------------------------------------------------------------------------ graph DOT analysis
_NODE = re.compile(r'^\s*"([^"]+)"\s*\[(.*)$')
_EDGE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')


def parse_dot(text: str) -> dict:
    """cudaGraphDebugDotPrint output -> {"nodes": {id: label text}, "edges": [(a, b)]}. A node's declaration may span
    lines (the label holds newlines): its text runs until the next declaration / edge line."""
    nodes: Dict[str, str] = {}
    edges = []
    cur = None
    for line in text.splitlines():
        m = _EDGE.match(line)
        if m:
            edges.append((m.group(1), m.group(2)))
            cur = None
            continue
        m = _NODE.match(line)
        if m and "->" not in line:
            cur = m.group(1)
            nodes[cur] = m.group(2)
            continue
        if cur is not None:
            nodes[cur] += "\n" + line
    for a, b in edges:
        nodes.setdefault(a, "")
        nodes.setdefault(b, "")
    return {"nodes": nodes, "edges": edges}


def analyze_dag(nodes: Dict[str, str], edges, branch_names=BRANCH_NAMES) -> dict:
    """Fork / join counts, the longest path in nodes, and which named nodes sit off it."""
    succ = collections.defaultdict(list)
    indeg = collections.Counter()
    outdeg = collections.Counter()
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
        outdeg[a] += 1
    order, q = [], [n for n in nodes if indeg[n] == 0]
    deg = dict(indeg)
    while q:
        n = q.pop()
        order.append(n)
        for m in succ[n]:
            deg[m] -= 1
            if deg[m] == 0:
                q.append(m)
    if len(order) != len(nodes):
        raise ValueError(f"graph has a cycle or dangling edges ({len(order)} of {len(nodes)} ordered)")
    best = {n: 1 for n in nodes}
    prev: Dict[str, Optional[str]] = {n: None for n in nodes}
    for n in order:
        for m in succ[n]:
            if best[n] + 1 > best[m]:
                best[m], prev[m] = best[n] + 1, n
    end = max(best, key=lambda n: best[n]) if best else None
    path = set()
    while end is not None:
        path.add(end)
        end = prev[end]
    off = [n for n in nodes if n not in path]

    def name_of(n):
        lab = nodes[n]
        for k in branch_names:
            if k in lab:
                return k
        return "other"
    return {"nodes": len(nodes), "edges": len(edges), "longest_path": len(path),
            "forks": sum(1 for n in nodes if outdeg[n] >= 2), "joins": sum(1 for n in nodes if indeg[n] >= 2),
            "off_path": len(off), "off_path_names": dict(collections.Counter(name_of(n) for n in off))}


@contextlib.contextmanager
def debug_graphs():
    """torch.cuda.CUDAGraph() -> a graph in debug mode (keeps the cudaGraph_t for debug_dump); yields the list."""
    real = torch.cuda.CUDAGraph
    made = []

    def factory(*a, **k):
        g = real(*a, **k)
        g.enable_debug_mode()
        made.append(g)
        return g
    torch.cuda.CUDAGraph = factory
    try:
        yield made
    finally:
        torch.cuda.CUDAGraph = real


def dump_and_analyze(graphs, tmpdir: str) -> List[dict]:
    out = []
    for i, g in enumerate(graphs):
        p = os.path.join(tmpdir, f"g{i}.dot")
        g.debug_dump(p)
        with open(p) as f:
            d = parse_dot(f.read())
        out.append(analyze_dag(d["nodes"], d["edges"]))
    return out


# ------------------------------------------------------------------------------------------------ gpu (box)
def _config(kind: str):
    from mini_ckpt import MINI_CONFIG

    if kind == "mini4":
        cfg = dict(MINI_CONFIG)
        cfg.update({"n_layers": 4, "compress_ratios": [0, 2, 1, 1, 0], "kv_source_layers": [1, 2],
                    "index_source_layers": [1, 2, 3], "candidate_source_layer": 2, "candidate_topk_blocks": 64,
                    "candidate_block_size": 8, "index_topk": 64, "dspark_target_layer_ids": [3],
                    "n_activated_experts": 4})
        hot = {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6], "2": [0, 1, 2, 3], "3": [4, 5, 6, 7]}
        return cfg, hot
    if kind == "real40":
        from glue_census import real_structure_config

        return real_structure_config()
    raise ValueError(f"kind {kind!r}: mini4 | real40")


@contextlib.contextmanager
def gpu_model(kind: str):
    from mini_ckpt import make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    import engram

    cfg, hot = _config(kind)
    V, cv = cfg["vocab_size"], cfg["engram_compressed_vocab_size"]
    engram.build_compressed_token_map = lambda tok: ([i % 97 for i in range(V)], cv)
    with tempfile.TemporaryDirectory() as tmp:
        ck = make_mini_checkpoint(os.path.join(tmp, "ckpt"), cfg=cfg)
        model = build_and_load(ck["model_dir"], hot, "cuda", tokenizer=object(), io_threads=4, stage_bytes=1 << 20,
                               log=lambda *a: None)
        model.temperature = 0.0
        res = model.runtime_load
        try:
            yield model, res.pool
        finally:
            torch.set_default_device("cpu")
            del model
            torch.cuda.synchronize()
            res.release()


def _toks(L, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randint(2, 2000, (L + n,), generator=g, device="cpu").tolist()
    return t[:L], t[L:]


def _prefill(model, ids):
    from runtime.refmodel import reference_forward

    with torch.inference_mode():
        reference_forward(model, torch.tensor([ids], dtype=torch.long, device="cuda"), 0)


def _run(model, st, ids, toks, n, graph):
    _prefill(model, ids)
    out = []
    with torch.inference_mode():
        for s in range(n):
            tk = toks[s * st.rows:(s + 1) * st.rows] if st.rows > 1 else toks[s]
            lg = st.step(tk, len(ids) + s * st.rows, graph=graph).reshape(st.rows, -1)
            out += [lg[j].float().clone() for j in range(st.rows)]
    torch.cuda.synchronize()
    return out


def _eq(a, b) -> int:
    """rows differing (torch.equal per row)."""
    return sum(1 for x, y in zip(a, b) if not torch.equal(x, y)) + abs(len(a) - len(b))


def _time_window(model, st, ids, iters: int, mode: str, nsys: bool) -> float:
    """ms per step over `iters` consecutive steps after a prefill (graph replays or static-eager steps)."""
    _prefill(model, ids)
    L = len(ids)
    with torch.inference_mode():
        st.set_inputs(None if st.rows > 1 else 3, L)
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0.record()
        for k in range(iters):
            if nsys:
                torch.cuda.nvtx.range_push("decode_step")
            if mode == "graph":
                st.replay(L + k * st.rows)
            else:
                st.step(None, L + k * st.rows, graph=False)
            if nsys:
                torch.cuda.nvtx.range_pop()
        t1.record()
        torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def _profile(model, st, ids, iters: int, mode: str) -> dict:
    """Kernel durations per family per step (torch.profiler / CUPTI), or {"error": ...}."""
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as e:  # pragma: no cover - box only
        return {"error": f"{type(e).__name__}: {e}"}
    _prefill(model, ids)
    L = len(ids)
    try:
        with torch.inference_mode():
            st.set_inputs(None if st.rows > 1 else 3, L)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for k in range(iters):
                    if mode == "graph":
                        st.replay(L + k * st.rows)
                    else:
                        st.step(None, L + k * st.rows, graph=False)
                torch.cuda.synchronize()
        from torch.autograd import DeviceType

        fam = collections.defaultdict(lambda: [0, 0.0])
        for e in prof.events():
            if getattr(e, "device_type", None) != DeviceType.CUDA:   # kernels / memcpy only (CPU ops would double
                continue                                              # count their kernels' device time)
            t = float(e.time_range.end - e.time_range.start)
            name = e.name
            f = ("branch:" + next(k for k in BRANCH_NAMES if k in name)) if any(k in name for k in BRANCH_NAMES) \
                else ("gemv" if "gemv" in name else "other")
            fam[f][0] += 1
            fam[f][1] += t
        if not fam:
            return {"error": "the profiler saw no device events (graph kernels not traced?): use --mode eager or "
                             "--nsys"}
        return {f: {"per_step": round(c / iters, 2), "us_per_step": round(t / iters, 2),
                    "mean_us": round(t / c, 3) if c else None} for f, (c, t) in sorted(fam.items())}
    except Exception as e:  # pragma: no cover - box only
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def run_gpu(a, log=print) -> dict:
    from runtime.refmodel import tilelang_available
    from runtime.static_step import StaticStep

    if not torch.cuda.is_available() or not tilelang_available():
        raise SystemExit("gpu mode needs CUDA + tilelang (the box, inside vllm/vllm-openai:v0.31.0)")
    impls = a.mixes.split(",")
    if impls[0] != "off":
        raise SystemExit("--mixes must start with off (the baseline every impl is compared with)")
    out = {"impls": impls, "fp32_impl": os.environ.get("DSV41_FP32_IMPL"), "mode": a.mode, "kinds": {}}
    for kind in a.kinds.split(","):
        krec = out["kinds"][kind] = {}
        try:
            with gpu_model(kind) as (model, pool):
                n_sub = 2 * len(model.layers)
                for rows in [int(r) for r in a.rows.split(",")]:   # 6 first: its dg-inplace build grows the pool once
                    rec = krec[str(rows)] = {"sublayers": n_sub}
                    ids, toks = _toks(17, rows * a.check_steps, 7 + rows)
                    steps, logs, graphs = {}, {}, {}
                    try:
                        for impl in impls:
                            with env(DSV41_MIXES=impl):
                                kw = dict(FAST, rows=rows) if rows > 1 else dict(FAST)
                                st = steps[impl] = StaticStep(model, pool, max_pos=MAX_POS, **kw)
                            e = _run(model, st, ids, toks, a.check_steps, graph=False)
                            with tempfile.TemporaryDirectory() as td, debug_graphs() as made:
                                st.capture()
                                try:
                                    graphs[impl] = dump_and_analyze(made, td)
                                except Exception as ex:   # debug dump unavailable: report, keep going
                                    graphs[impl] = [{"error": f"{type(ex).__name__}: {str(ex)[:160]}"}]
                            g1 = _run(model, st, ids, toks, a.check_steps, graph=True)
                            g2 = _run(model, st, ids, toks, a.check_steps, graph=True)
                            logs[impl] = (e, g1, g2)
                            rec.setdefault("identity", {})[impl] = {
                                "graph_vs_eager_rows_differing": _eq(g1, e), "replay2_vs_1_rows_differing": _eq(g2, g1),
                                "rows_compared": len(e), "describe_mixes": st.describe()["mixes"],
                                "glue_line": st.glue.glue_line() if st.glue is not None else None}
                        e0, g0, _ = logs["off"]
                        for impl in impls[1:]:
                            e, g1, _ = logs[impl]
                            rec["identity"][impl].update({"eager_vs_off_rows_differing": _eq(e, e0),
                                                          "graph_vs_off_rows_differing": _eq(g1, g0)})
                        rec["graph"] = graphs
                        # timing: impls interleaved per rep, median
                        times = {impl: [] for impl in impls}
                        if a.nsys:
                            torch.cuda.profiler.start()
                        for _ in range(a.warm):
                            for impl in impls:
                                _time_window(model, steps[impl], ids, a.iters, a.mode, False)
                        for _ in range(a.reps):
                            for impl in impls:
                                if a.nsys:
                                    torch.cuda.nvtx.range_push(f"mixes_{impl}_rows{rows}")
                                times[impl].append(_time_window(model, steps[impl], ids, a.iters, a.mode, a.nsys))
                                if a.nsys:
                                    torch.cuda.nvtx.range_pop()
                        if a.nsys:
                            torch.cuda.profiler.stop()
                        rec["ms_per_step"] = {impl: {"median": round(statistics.median(v), 4),
                                                     "min": round(min(v), 4), "all": [round(x, 4) for x in v]}
                                              for impl, v in times.items()}
                        base = rec["ms_per_step"]["off"]["median"]
                        rec["saved_us_per_step"] = {impl: round((base - rec["ms_per_step"][impl]["median"]) * 1e3, 1)
                                                    for impl in impls[1:]}
                        if a.profile:
                            rec["kernels"] = {impl: _profile(model, steps[impl], ids, a.iters, a.mode)
                                              for impl in impls}
                            ko, kf = rec["kernels"].get("off", {}), rec["kernels"].get("fork", {})
                            if "gemv" in ko and "gemv" in kf and ko["gemv"]["mean_us"]:
                                rec["contention"] = {
                                    "gemv_mean_us_off": ko["gemv"]["mean_us"], "gemv_mean_us_fork": kf["gemv"]["mean_us"],
                                    "gemv_ratio": round(kf["gemv"]["mean_us"] / ko["gemv"]["mean_us"], 4),
                                    "other_ratio": round(kf["other"]["mean_us"] / ko["other"]["mean_us"], 4)
                                    if ko.get("other", {}).get("mean_us") and kf.get("other", {}).get("mean_us")
                                    else None}
                    finally:
                        for st in steps.values():
                            st.detach()
                    _log_rows(kind, rows, rec, impls, log)
        except SystemExit:
            raise
        except Exception as ex:   # one kind failing (e.g. real40 on an untested path) is recorded, not fatal
            import traceback

            traceback.print_exc()
            krec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            log(f"  {kind}: ERROR {krec['error']}")
    out["modeled_box"] = modeled_box_ms()
    return out


def _log_rows(kind, rows, rec, impls, log):
    log(f"  [{kind} rows {rows}] {rec['sublayers']} sublayers")
    for impl in impls:
        i = rec["identity"][impl]
        g = rec["graph"][impl]
        gs = "; ".join(f"nodes {x['nodes']} longest {x['longest_path']} forks {x['forks']} joins {x['joins']} "
                       f"off-path {x['off_path_names']}" if "error" not in x else x["error"] for x in g)
        ms = rec["ms_per_step"][impl]
        vs = "" if impl == "off" else (f" | vs off: eager {i['eager_vs_off_rows_differing']} / graph "
                                       f"{i['graph_vs_off_rows_differing']} rows differ; saved "
                                       f"{rec['saved_us_per_step'][impl]:+.1f} us/step")
        log(f"    {impl:5s} {ms['median']:.4f} ms/step (min {ms['min']:.4f}) | graph==eager "
            f"{i['rows_compared'] - i['graph_vs_eager_rows_differing']}/{i['rows_compared']}, replay2==1 "
            f"{i['rows_compared'] - i['replay2_vs_1_rows_differing']}/{i['rows_compared']}{vs}")
        log(f"          graph: {gs}")
    if rec.get("contention"):
        log(f"    contention: gemv mean {rec['contention']['gemv_mean_us_off']} -> "
            f"{rec['contention']['gemv_mean_us_fork']} us (x{rec['contention']['gemv_ratio']}), other "
            f"x{rec['contention']['other_ratio']}")


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M2-L1 mixes fork: CPU launch census / box A/B")
    ap.add_argument("what", choices=["census", "gpu"])
    ap.add_argument("--rows", default=None, help="census default 1,6; gpu default 6,1 (6 first: pool growth)")
    ap.add_argument("--fp32-impl", choices=["cublas", "fixed"], default="fixed",
                    help="$DSV41_FP32_IMPL for the run (default fixed = today's headline)")
    ap.add_argument("--steps", type=int, default=2, help="census: steps counted")
    ap.add_argument("--kinds", default="mini4,real40", help="gpu: synthetic checkpoints")
    ap.add_argument("--mixes", default="off,fork,fused", help="gpu: impls ($DSV41_MIXES), off first")
    ap.add_argument("--mode", choices=["graph", "eager"], default="graph", help="gpu: timed path")
    ap.add_argument("--check-steps", type=int, default=4, help="gpu: teacher-forced steps compared bitwise")
    ap.add_argument("--iters", type=int, default=32, help="gpu: steps per timed window")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--profile", action="store_true", help="gpu: torch.profiler kernel families (contention)")
    ap.add_argument("--nsys", action="store_true", help="gpu: cudaProfilerStart/Stop + decode_step NVTX ranges")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.environ["DSV41_FP32_IMPL"] = a.fp32_impl
    if a.what == "census":
        rows = [int(r) for r in (a.rows or "1,6").split(",")]
        rep = run_census(rows, a.steps, a.fp32_impl)
        a.json = a.json or os.path.join(REPO, "scratch", "mixes-fork-census.json")
    else:
        a.rows = a.rows or "6,1"
        a.iters = max(1, min(a.iters, (MAX_POS - 17) // 6 - 1))
        rep = run_gpu(a)
        a.json = a.json or os.path.join(REPO, "scratch", "mixes-fork-bench.json")
    rep.update({"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": list(argv or sys.argv[1:]),
                "wall_s": round(time.time() - t0, 1), "modeled_box": modeled_box_ms()})
    mb = rep["modeled_box"]
    print(f"  modeled box [E]: rows 1 {mb['rows 1']['ms_lo']}-{mb['rows 1']['ms_hi']} ms/step, rows 6 "
          f"{mb['rows 6']['ms_lo']}-{mb['rows 6']['ms_hi']} ms/step ({mb['basis']})")
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"report {a.json} ({rep['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
