#!/usr/bin/env python3
"""gemv_bench.py -- T9b-1 microbench: the Triton MXFP8 GEMV (runtime/gemv_mxfp8.py) vs the reference `fp8_gemm`
(kernel.py:207-307) on the checkpoint's own fp8 Linear weights, at every distinct shape the 1-row step calls.

  python3 bench/gemv_bench.py --model-dir /ref [--layers 8] [--ms 1,6] [--json scratch/gemv-bench.json] \
      [--tune-cache scratch/gemv-tune.json]

Shapes are enumerated from the checkpoint (runtime/ckpt.py ShardIndex, the loader's own names): every
`layers.<L>.<role>.weight` stored F8_E4M3 with a 32x32-block F8_E8M0 `.scale` -- routed experts (fp4), `wo_a` (the
loader dequantizes it to bf16, model.py:785) and the Engram table (not a Linear) excluded -- grouped by (N, K); the
per-step call count of a role is its number of layers (cross-checked against 290 = 40 x 7 + 8 + 2).

Per shape, per M in --ms:
  * operands: real weight/scale bytes of up to --layers layers per role (evenly spaced); activations bf16 [M, K] with
    per-32-block magnitudes 2^-6..2^6 (the T9a spikes' recipe), quantized by the reference act_quant (model.py:197);
  * correctness on every loaded real layer: bf16 mismatch rate of GEMV vs reference fp8_gemm, max |delta|, max bf16
    ulp distance, outputs > 1 ulp apart; mean |error| of both vs the float64 product of the dequantized operands (first
    layer, shapes <= 64 M elements); gate: mismatch <= 0.5 % ("1-ulp class") -- any shape above is named;
  * config invariance: candidate configs (runtime/gemv_mxfp8.candidate_configs, up to 12) with the SAME numerics_key
    (split partition + block-partial form) as the tuned one are bit-identical to it (first layer); configs of other
    keys differ by design (split order) and are only counted;
  * timing: each impl captured as one CUDA graph of >= 64 calls cycling over enough weight copies (real layers, then
    clones of them) to exceed --stream-mib (default 512 MiB > L2), so weights stream from HBM as in the decode step;
    µs/call = replay time / calls (no launch gaps: the graph is what the step replays; split-K calls include their
    reduce launch). GB/s = (weight + weight-scale + activation + activation-scale + output bytes) / time;
  * tuning discrimination: per shape and M the tuner's best / runner-up / worst µs and how many candidates ran or failed
    (from the tune log), and at the end how many distinct configs the shapes picked;
  * gates (docs/HANDOFF-T9B1-FIX.md), per shape: GEMV >= 2x the reference at M=1 AND >= 1.5x at every measured M > 1
    (M=6 by default); shapes with >= 16 MiB of weight >= 4 TB/s at M=1; smaller shapes <= 8 µs at M=1; bf16 mismatch
    <= 0.5 %. Per step at M=1: the saving vs the reference (target >= 12 ms of the ~15.8 ms).
Also: the per-step GEMM time at M=1 (sum over roles of calls x µs) for both impls, and the tuned configs (written to
--tune-cache, which runtime/gemv_mxfp8 reads via $DSV41_GEMV_CACHE).

T9b-6 (`--impl gemv3|both`, default both; docs/HANDOFF-T9B6-GEMV-BYTES.md): per shape and M the gemv3 kernel too, on
the same operands -- its (N, K) config (tuned once per shape by autotune3 at M = 1 and 8, written to --tune-cache3),
drift vs the reference on every real layer, fp64-oracle mean |err| vs gemv's, bit-identity with gemv's MMA=1 form at
the same split plan, every row alone == its row in the M-row call, a graph replayed twice == eager, µs / GB/s and the
config's bytes per thread per iteration and in flight per program. GATES3: per-step M=1 total <= 2.0 ms; every shape
>= 3 TB/s or <= 4 µs (every measured M); rows / replay / same-plan bit-identity; oracle <= gemv's within 0.5 %. Per-step
totals at every M for both impls. Exit code: GATES3 (gemv's T9b-1 gates with --impl gemv). Box line: the one below
plus `--impl both --json scratch/gemv-bench-t9b6.json` (receipts go to scratch/, never docs/).

Box command (same image and mounts as the T9a-1 runs; checkpoint read-only at /ref):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    -v "$W":/work -w /work \
    -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/gemv_bench.py --model-dir /ref
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn
EXPECTED_ROLES = {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "attn.indexer.wq_b", "ffn.shared_experts.w1",
                  "ffn.shared_experts.w2", "ffn.shared_experts.w3", "engram.wkv"}
EXPECTED_CALLS_PER_STEP = 290
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)\.weight$")
# gates (docs/HANDOFF-T9B1-FIX.md)
GATE_SPEEDUP_M1 = 2.0
GATE_SPEEDUP_MULTI = 1.5
GATE_BIG_BYTES = 16 << 20
GATE_BIG_GBPS = 4000.0
GATE_SMALL_US = 8.0
GATE_STEP_SAVING_MS = 12.0
GATE_MISMATCH = 5e-3
INVARIANCE_SAMPLE = 12
# T9b-6 gemv3 gates (docs/HANDOFF-T9B6-GEMV-BYTES.md): per-step M=1 total <= 2.0 ms (gemv: 3.43 ms in
# docs/gemv-bench-r2-20261106.json); every shape class >= 3 TB/s or <= 4 us (checked at every measured M); rows
# bit-identical to the row alone; graph replay == eager; fp64-oracle mean |err| <= gemv's within ORACLE3_TOL
GATE3_STEP_MS_M1 = 2.0
GATE3_GBPS = 3000.0
GATE3_US = 4.0
ORACLE3_TOL = 5e-3   # relative slack on the MEAN |err|: two valid fixed fp32 orders flip a bf16 rounding on <= 0.5 %
#                      of outputs (the 1-ulp class), moving the mean by about that fraction either way


# ------------------------------------------------------------------------------------------------ enumeration
def enumerate_fp8_linears(idx) -> Tuple[Dict[str, dict], List[str]]:
    """{role: {"shape": (N, K), "layers": [...]}} for every layer-indexed fp8 Linear with 32x32 e8m0 block scales;
    plus the names skipped and why."""
    roles: Dict[str, dict] = {}
    skipped = set()
    for name in sorted(idx.names()):
        m = LAYER_RE.match(name)
        if not m:
            continue
        L, role = int(m.group(1)), m.group(2)
        e = idx[name]
        if e.st_dtype != "F8_E4M3" or len(e.shape) != 2:
            continue
        if ".experts." in role and "shared_experts" not in role:
            continue  # routed experts are I8/fp4 anyway
        sname = name[:-len(".weight")] + ".scale"
        if sname not in idx:
            skipped.add(f"{role}: no .scale")
            continue
        se = idx[sname]
        N, K = e.shape
        if se.st_dtype != "F8_E8M0" or tuple(se.shape) != ((N + 31) // 32, (K + 31) // 32):
            skipped.add(f"{role}: scale {se.st_dtype} [rows, {se.shape[-1]}] is not 32x32 blocks")  # Engram table
            continue
        if role.endswith("wo_a"):
            skipped.add(f"{role}: dequantized to bf16 by the loader (einsum, model.py:785-789)")
            continue
        r = roles.setdefault(role, {"shape": (N, K), "layers": []})
        if r["shape"] != (N, K):
            raise ValueError(f"{role}: shape {(N, K)} at layer {L} != {r['shape']}")
        r["layers"].append(L)
    return roles, sorted(skipped)


def pick_layers(layers: List[int], n: int) -> List[int]:
    if len(layers) <= n:
        return list(layers)
    step = len(layers) / n
    return [layers[int(i * step)] for i in range(n)]


# ------------------------------------------------------------------------------------------------ helpers
def activations(M: int, K: int, seed: int, K_mod):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(M, K // 32, 32, generator=g, device="cpu")
         * torch.exp2(torch.randint(-6, 7, (M, K // 32, 1), generator=g, device="cpu").float())
         ).view(M, K).to(torch.bfloat16).cuda()
    return K_mod.act_quant(x, 32, "ue8m0", E8)


def ulp_distance(a, b):
    def key(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (key(a) - key(b)).abs()


def exact_f64(xq, xs, w, s):
    N = w.size(0)
    A = xq.double() * xs.float().double().repeat_interleave(32, 1)
    B = w.double() * s.float().double()[torch.arange(N, device=w.device) // 32].repeat_interleave(32, 1)
    return A @ B.t()


def bytes_moved(M, N, K) -> int:
    return N * K + ((N + 31) // 32) * (K // 32) + M * K + M * (K // 32) + 2 * M * N


def shape_gates(N: int, K: int, per_m: Dict[int, dict]) -> dict:
    """The per-shape gates from the per-M records ({M: {"speedup", "us_gemv", "GBps_gemv", "mismatch"}})."""
    big = N * K >= GATE_BIG_BYTES
    m1 = per_m.get(1)
    multi = {M: r for M, r in per_m.items() if M > 1}
    return {
        "speedup_ge_2x_m1": (m1["speedup"] >= GATE_SPEEDUP_M1) if m1 else None,
        "speedup_ge_1p5x_multirow": all(r["speedup"] >= GATE_SPEEDUP_MULTI for r in multi.values()) if multi else None,
        "bw_ge_4TBps_m1": (m1["GBps_gemv"] >= GATE_BIG_GBPS) if (m1 and big) else None,
        "small_le_8us_m1": (m1["us_gemv"] <= GATE_SMALL_US) if (m1 and not big) else None,
        "mismatch_le_0p5pct": all(r["mismatch"] <= GATE_MISMATCH for r in per_m.values()),
    }


def overall_gates(shape_recs: List[dict], step: dict) -> dict:
    gs = [r["gates"] for r in shape_recs if r.get("gates")]

    def every(name):
        vals = [g[name] for g in gs if g.get(name) is not None]
        return all(vals) if vals else None

    def failing(name):
        return [r["shape"] for r in shape_recs if r.get("gates", {}).get(name) is False]

    out = {
        "every_shape_ge_2x_m1": every("speedup_ge_2x_m1"),
        "every_shape_ge_1p5x_multirow": every("speedup_ge_1p5x_multirow"),
        "big_shapes_ge_4TBps_m1": every("bw_ge_4TBps_m1"),
        "small_shapes_le_8us_m1": every("small_le_8us_m1"),
        "drift_1ulp_class": every("mismatch_le_0p5pct"),
        "step_saving_ge_12ms_m1": (step["saved_ms"] >= GATE_STEP_SAVING_MS) if step.get("ms_ref") else None,
        "config_invariant_all": all(v["config_invariant"] for r in shape_recs for v in r["per_m"].values()),
    }
    out["failing"] = {k: failing(k) for k in ("speedup_ge_2x_m1", "speedup_ge_1p5x_multirow", "bw_ge_4TBps_m1",
                                               "small_le_8us_m1", "mismatch_le_0p5pct") if failing(k)}
    out["shapes_over_0p5pct"] = [r["shape"] for r in shape_recs
                                 if any(v["mismatch"] > GATE_MISMATCH for v in r["per_m"].values())]
    out["all_pass"] = all(out[k] is not False for k in ("every_shape_ge_2x_m1", "every_shape_ge_1p5x_multirow",
                                                         "big_shapes_ge_4TBps_m1", "small_shapes_le_8us_m1",
                                                         "drift_1ulp_class", "step_saving_ge_12ms_m1")) \
        and out["every_shape_ge_2x_m1"] is not None
    return out


def tune_summary(log: Optional[dict]) -> dict:
    """best / runner-up / worst µs and counts from one runtime/gemv_mxfp8 tune-log entry (None if from the cache)."""
    if not log:
        return {"source": "cache"}
    us = sorted(log["us"].items(), key=lambda kv: kv[1])
    return {"source": "tuned", "n_timed": log.get("n_timed", len(us)), "n_failed": log.get("n_failed", 0),
            "best": us[0], "runner_up": us[1] if len(us) > 1 else None, "worst": us[-1],
            "spread": round(us[-1][1] / us[0][1], 2) if us[0][1] else None,
            "errors_sample": list(log.get("errors", {}).items())[:2]}


# ------------------------------------------------------------------------------------------------ T9b-6 gates
def shape_gates3(N: int, K: int, per_m3: Dict[int, dict]) -> dict:
    """Per-shape gemv3 gates from the per-M records ({M: {"us", "GBps", "mismatch", "row_invariant",
    "replay_identical", "err_ratio_vs_gemv", "bitident_same_plan"}}); a check a record lacks is None (not run)."""
    def every(pred, key):
        vals = [pred(r) for r in per_m3.values() if r.get(key) is not None]
        return all(vals) if vals else None

    return {
        "fast_ge_3TBps_or_le_4us": every(lambda r: r["GBps"] >= GATE3_GBPS or r["us"] <= GATE3_US, "us"),
        "row_invariant": every(lambda r: r["row_invariant"], "row_invariant"),
        "replay_identical": every(lambda r: r["replay_identical"], "replay_identical"),
        "oracle_le_gemv": every(lambda r: r["err_ratio_vs_gemv"] <= 1.0 + ORACLE3_TOL, "err_ratio_vs_gemv"),
        "bitident_gemv_same_plan": every(lambda r: r["bitident_same_plan"], "bitident_same_plan"),
        "mismatch_le_0p5pct": every(lambda r: r["mismatch"] <= GATE_MISMATCH, "mismatch"),
    }


GATES3 = ("fast_ge_3TBps_or_le_4us", "row_invariant", "replay_identical", "oracle_le_gemv", "bitident_gemv_same_plan",
          "mismatch_le_0p5pct")


def overall_gates3(shape_recs: List[dict], step3: dict) -> dict:
    """Every shape's gemv3 gates + the per-step M=1 total <= GATE3_STEP_MS_M1; all_pass if none is False and the
    step total and the speed gate were measured."""
    gs = [r["gates3"] for r in shape_recs if r.get("gates3")]
    out = {}
    for name in GATES3:
        vals = [g[name] for g in gs if g.get(name) is not None]
        out[name] = all(vals) if vals else None
    out["step_le_2ms_m1"] = (step3["ms_gemv3_m1"] <= GATE3_STEP_MS_M1) if step3.get("ms_gemv3_m1") is not None \
        else None
    out["failing"] = {name: [r["shape"] for r in shape_recs if r.get("gates3", {}).get(name) is False]
                      for name in GATES3 if any(r.get("gates3", {}).get(name) is False for r in shape_recs)}
    out["all_pass"] = all(out[k] is not False for k in GATES3 + ("step_le_2ms_m1",)) \
        and out["step_le_2ms_m1"] is not None and out["fast_ge_3TBps_or_le_4us"] is not None
    return out


def step_totals3(shape_recs: List[dict], Ms: List[int]) -> dict:
    """Per-step GEMV time (sum over shapes of calls x µs) for gemv3 and, where measured, gemv, at every M."""
    st = {}
    for M in Ms:
        for impl, key, src in (("gemv3", "us", "per_m3"), ("gemv", "us_gemv", "per_m")):
            recs = [r for r in shape_recs if M in r.get(src, {})]
            if recs and len(recs) == len(shape_recs):
                st[f"ms_{impl}_m{M}"] = round(sum(r["calls_per_step"] * r[src][M][key] for r in recs) / 1e3, 3)
        if f"ms_gemv_m{M}" in st and f"ms_gemv3_m{M}" in st:
            st[f"saved_ms_m{M}"] = round(st[f"ms_gemv_m{M}"] - st[f"ms_gemv3_m{M}"], 3)
    st["target_ms_gemv3_m1"] = GATE3_STEP_MS_M1
    return st


def measure_gemv3(G, K_mod, reals, copies, xq, xs, M: int, N: int, K: int, a, err_gemv: Optional[float],
                  dev) -> dict:
    """One shape at one M on gemv3 (the (N, K) config, tuned on first use): drift vs the reference on every real
    layer; on the first layer: fp64-oracle mean |err| (vs gemv's), bit-identity with gemv's MMA=1 form at the same
    split plan, each row alone == its row in this call, a graph of the calls replayed twice == eager; timing under a
    CUDA graph over the weight copies (one launch per call: no reduce)."""
    cfg = G.get_config3(N, K, dev)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    mism = tot = gt1 = 0
    max_ulp = 0
    for name, w, s in reals:
        ref = K_mod.fp8_gemm(xq, xs, w, s, E8, block_size=32)
        got = G.mxfp8_gemv3(xq, xs, w, s, out, config=cfg).clone()
        u = ulp_distance(got, ref)
        mism += int((got != ref).sum())
        tot += got.numel()
        gt1 += int((u > 1).sum())
        max_ulp = max(max_ulp, int(u.max()))
        if not bool(torch.isfinite(got.float()).all()):
            raise FloatingPointError(f"{name} M={M}: non-finite gemv3 output")
    _, w0, s0 = reals[0]
    base = G.mxfp8_gemv3(xq, xs, w0, s0, out, config=cfg).clone()
    err3 = ratio = None
    if N * K <= (64 << 20):
        ex = exact_f64(xq, xs, w0, s0)
        err3 = float((base.double() - ex).abs().mean())
        ratio = err3 / err_gemv if err_gemv else None
    same = G.mxfp8_gemv(xq, xs, w0, s0, torch.empty(M, N, dtype=torch.bfloat16, device=dev),
                        config=G.GemvConfig(32, 2, 4, 3, cfg.split_k, 1))
    rows_ok = all(torch.equal(G.mxfp8_gemv3(xq[m:m + 1].contiguous(), xs[m:m + 1].contiguous(), w0, s0,
                                            torch.empty(1, N, dtype=torch.bfloat16, device=dev), config=cfg),
                              base[m:m + 1]) for m in range(M)) if M > 1 else None
    # replay identity: a small graph of the same call (warmed on the capture stream, as time_graph does), replayed
    # twice into a NaN-filled output
    rep_out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        G.mxfp8_gemv3(xq, xs, w0, s0, rep_out, config=cfg)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        G.mxfp8_gemv3(xq, xs, w0, s0, rep_out, config=cfg)
    replay_ok = True
    for _ in range(2):
        rep_out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        replay_ok = replay_ok and torch.equal(rep_out, base)
    del g
    n_copies = len(copies)
    ncalls = max(a.calls, n_copies)
    us = G.time_graph([lambda i=i: G.mxfp8_gemv3(xq, xs, copies[i % n_copies][0], copies[i % n_copies][1], out,
                                                 config=cfg) for i in range(ncalls)], a.reps)
    plan = G.launch_plan3(cfg, M, N, K)
    log = G._TUNE_LOG3.get((N, K))
    return {"config": cfg.as_dict(), "numerics_key": list(G.numerics_key3(cfg, N, K)), "grid": list(plan["grid"]),
            "EVEN": plan["EVEN"], "tile_bytes": plan["tile_bytes"], "bytes_per_thread": plan["bytes_per_thread"],
            "inflight_bytes": plan["inflight_bytes"], "us": round(us, 3),
            "GBps": round(bytes_moved(M, N, K) / us / 1e3, 1), "mismatch": mism / tot, "max_ulp": max_ulp,
            "n_gt1ulp": gt1, "mean_abs_err_vs_f64": err3, "err_gemv_vs_f64": err_gemv, "err_ratio_vs_gemv": ratio,
            "bitident_same_plan": bool(torch.equal(base, same)), "row_invariant": rows_ok,
            "replay_identical": replay_ok,
            "tune": {"source": "tuned", "best_us": log.get("best_us"), "n_timed": log.get("n_timed"),
                     "n_failed": log.get("n_failed"), "worst_us": log.get("worst_us")} if log else {"source": "cache"}}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, default=8, help="real layers loaded per role (evenly spaced)")
    ap.add_argument("--ms", default="1,6")
    ap.add_argument("--stream-mib", type=int, default=512, help="weight bytes cycled per timed graph (> L2)")
    ap.add_argument("--calls", type=int, default=64, help="min calls per timed graph")
    ap.add_argument("--reps", type=int, default=5, help="graph replays timed")
    ap.add_argument("--json", default=os.path.join(REPO, "scratch", "gemv-bench.json"))
    ap.add_argument("--tune-cache", default=os.path.join(REPO, "scratch", "gemv-tune.json"))
    ap.add_argument("--only", default=None, help="comma-separated roles")
    ap.add_argument("--tune-space", choices=("full", "quick"), default="full",
                    help="full = the handoff's grid (~100-200 candidates per shape and M bucket; first run compiles "
                         "them all, the Triton cache mount keeps them); quick = KU 4 / 4 warps / 2 stages only")
    ap.add_argument("--impl", choices=("gemv", "gemv3", "both"), default="both",
                    help="T9b-6: gemv (T9b-1 kernel + its gates), gemv3 (the bytes-in-flight kernel + GATES3; "
                         "gemv's fp64 error from its cached / default config), both (default; exit code = GATES3)")
    ap.add_argument("--tune-cache3", default=os.path.join(REPO, "scratch", "gemv-tune.gemv3.json"),
                    help="gemv3's own versioned config cache ($DSV41_GEMV3_CACHE); the default is the path gemv3 "
                         "derives from $DSV41_GEMV_CACHE=scratch/gemv-tune.json, so a later run finds it unprompted")
    a = ap.parse_args(argv)
    os.environ["DSV41_GEMV_CACHE"] = a.tune_cache
    os.environ["DSV41_GEMV3_CACHE"] = a.tune_cache3
    os.environ.setdefault("DSV41_GEMV_TUNE", "1")
    os.environ["DSV41_GEMV_TUNE_SPACE"] = a.tune_space

    from runtime import gemv_mxfp8 as G
    from runtime.ckpt import ShardIndex
    from runtime.refmodel import import_reference

    import_reference()
    K_mod = sys.modules["kernel"]
    torch.set_default_dtype(torch.bfloat16)  # the reference fp8_gemm allocates its output in the default dtype
    dev = torch.device("cuda")
    t_start = time.time()
    idx = ShardIndex.scan(a.model_dir)
    roles, skipped = enumerate_fp8_linears(idx)
    if a.only:
        roles = {r: v for r, v in roles.items() if r in set(a.only.split(","))}
    calls_per_step = sum(len(v["layers"]) for v in roles.values())
    unknown = sorted(set(roles) - EXPECTED_ROLES)
    missing = sorted(EXPECTED_ROLES - set(roles)) if not a.only else []
    print(f"fp8 Linear roles: " + ", ".join(f"{r} {v['shape']} x{len(v['layers'])}" for r, v in sorted(roles.items())))
    print(f"calls/step {calls_per_step} (expected {EXPECTED_CALLS_PER_STEP}); unknown roles {unknown}; missing "
          f"{missing}; skipped kinds {skipped}")
    shapes: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for r, v in sorted(roles.items()):
        shapes[v["shape"]].append(r)
    Ms = [int(x) for x in a.ms.split(",")]
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
              "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                      "capability": torch.cuda.get_device_capability(), "sms": torch.cuda.get_device_properties(0)
                      .multi_processor_count, "triton": G.describe()["triton"]},
              "roles": {r: {"shape": list(v["shape"]), "calls_per_step": len(v["layers"])} for r, v in roles.items()},
              "calls_per_step": calls_per_step, "unknown_roles": unknown, "missing_roles": missing,
              "skipped": skipped, "shapes": [], "gates": {}}
    for (N, K), rs in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
        t0 = time.time()
        reals = []
        for r in rs:
            for L in pick_layers(roles[r]["layers"], a.layers):
                w = idx.read_tensor(f"layers.{L}.{r}.weight").to(dev)
                s = idx.read_tensor(f"layers.{L}.{r}.scale").to(dev)
                reals.append((f"L{L}.{r}", w, s))
        wbytes = N * K
        n_copies = max(len(reals), min(256, math.ceil((a.stream_mib << 20) / wbytes)))
        copies = [(reals[i % len(reals)][1] if i < len(reals) else reals[i % len(reals)][1].clone(),
                   reals[i % len(reals)][2] if i < len(reals) else reals[i % len(reals)][2].clone())
                  for i in range(n_copies)]
        rec = {"shape": [N, K], "roles": rs, "calls_per_step": sum(len(roles[r]["layers"]) for r in rs),
               "weight_MiB": round(wbytes / 2**20, 2), "real_layers": [n for n, _, _ in reals],
               "timing_copies": n_copies, "timing_stream_MiB": round(n_copies * wbytes / 2**20, 1), "per_m": {},
               "per_m3": {}}
        for M in Ms:
            xq, xs = activations(M, K, 1000 * M + N % 997 + K, K_mod)
            out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
            if a.impl == "gemv3":   # T9b-6 only: gemv's fp64 error from its cached / default config, no gemv timing
                rec["per_m3"][M] = r3 = measure_gemv3(G, K_mod, reals, copies, xq, xs, M, N, K, a,
                                                      gemv_err_vs_f64(G, xq, xs, reals[0][1], reals[0][2], dev), dev)
                print_gemv3(N, K, rs, M, r3, len(reals))
                continue
            cfg = G.get_config(N, K, M, dev)   # tunes (synthetic rotating weights) unless cached
            # correctness on every real layer
            mism = tot = gt1 = 0
            max_abs, max_ulp = 0.0, 0
            per_layer = {}
            for name, w, s in reals:
                ref = K_mod.fp8_gemm(xq, xs, w, s, E8, block_size=32)
                got = G.mxfp8_gemv(xq, xs, w, s, out).clone()
                u = ulp_distance(got, ref)
                m_ = int((got != ref).sum())
                per_layer[name] = round(m_ / got.numel(), 6)
                mism += m_
                tot += got.numel()
                gt1 += int((u > 1).sum())
                max_ulp = max(max_ulp, int(u.max()))
                max_abs = max(max_abs, float((got.float() - ref.float()).abs().max()))
                if not bool(torch.isfinite(got.float()).all()):
                    raise FloatingPointError(f"{name} M={M}: non-finite GEMV output")
            name0, w0, s0 = reals[0]
            err = None
            if N * K <= (64 << 20):
                ex = exact_f64(xq, xs, w0, s0)
                ref0 = K_mod.fp8_gemm(xq, xs, w0, s0, E8, block_size=32)
                got0 = G.mxfp8_gemv(xq, xs, w0, s0, out).clone()
                err = {"gemv": float((got0.double() - ex).abs().mean()), "ref": float((ref0.double() - ex).abs().mean())}
            # config invariance: same numerics_key as the tuned config -> bit-identical (sampled; others counted)
            base = G.mxfp8_gemv(xq, xs, w0, s0, out, config=cfg).clone()
            nkey = G.numerics_key(cfg, N, K, G.num_sms(dev))
            cands = G.candidate_configs(N, K, M, G.num_sms(dev))
            same = [c for c in cands if G.numerics_key(c, N, K, G.num_sms(dev)) == nkey and c != cfg]
            same = same[::max(1, len(same) // INVARIANCE_SAMPLE)][:INVARIANCE_SAMPLE]
            variant, inv_errs = [], 0
            for c in same:
                try:
                    if not torch.equal(G.mxfp8_gemv(xq, xs, w0, s0, out, config=c), base):
                        variant.append(c.as_dict())
                except Exception:  # did not compile/launch here (e.g. MMA=1 on this Triton): the tuner skips it too
                    inv_errs += 1
            # timing under a CUDA graph, weights cycling over the copies
            ncalls = max(a.calls, n_copies)
            us_gemv = G.time_graph([lambda i=i: G.mxfp8_gemv(xq, xs, copies[i % n_copies][0], copies[i % n_copies][1],
                                                             out, config=cfg) for i in range(ncalls)], a.reps)
            us_ref = G.time_graph([lambda i=i: K_mod.fp8_gemm(xq, xs, copies[i % n_copies][0], copies[i % n_copies][1],
                                                              E8, block_size=32) for i in range(ncalls)], a.reps)
            bm = bytes_moved(M, N, K)
            plan = G.launch_plan(cfg, M, N, K)
            ts = tune_summary(G._TUNE_LOG.get(G.config_key(N, K, M)))
            r = {"config": cfg.as_dict(), "numerics_key": list(nkey), "grid": list(plan["grid"]),
                 "us_gemv": round(us_gemv, 3), "us_ref": round(us_ref, 3),
                 "speedup": round(us_ref / us_gemv, 2), "GBps_gemv": round(bm / us_gemv / 1e3, 1),
                 "GBps_ref": round(bm / us_ref / 1e3, 1), "mismatch": mism / tot, "outputs_compared": tot,
                 "max_abs": max_abs, "max_ulp": max_ulp, "n_gt1ulp": gt1, "mismatch_per_layer": per_layer,
                 "mean_abs_err_vs_f64": err, "config_invariant": not variant, "configs_differing": variant,
                 "invariance_checked": len(same) - inv_errs, "invariance_errors": inv_errs, "tune": ts}
            rec["per_m"][M] = r
            print(f"[{N}x{K}] {'/'.join(rs)} M={M}: gemv {us_gemv:.2f} us ({r['GBps_gemv']:.0f} GB/s) vs ref "
                  f"{us_ref:.2f} us ({r['GBps_ref']:.0f} GB/s) = {r['speedup']:.1f}x | mismatch "
                  f"{r['mismatch'] * 100:.3f} % over {len(reals)} layers, max ulp {max_ulp} (>1: {gt1}), "
                  f"config {cfg.as_dict()} grid {tuple(plan['grid'])}, invariant {not variant} "
                  f"({r['invariance_checked']} same-numerics configs)", flush=True)
            if ts.get("source") == "tuned":
                print(f"    tune: {ts['n_timed']} timed, {ts['n_failed']} failed; best {ts['best'][1]} us, runner-up "
                      f"{ts['runner_up'][1] if ts['runner_up'] else '-'} us, worst {ts['worst'][1]} us "
                      f"(spread {ts['spread']}x)" + (f"; e.g. error {ts['errors_sample'][0]}"
                                                    if ts["errors_sample"] else ""), flush=True)
            if a.impl == "both":
                rec["per_m3"][M] = r3 = measure_gemv3(G, K_mod, reals, copies, xq, xs, M, N, K, a,
                                                      err["gemv"] if err else None, dev)
                print_gemv3(N, K, rs, M, r3, len(reals))
        rec["gates"] = shape_gates(N, K, rec["per_m"]) if rec["per_m"] else {}
        rec["gates3"] = shape_gates3(N, K, rec["per_m3"]) if rec["per_m3"] else {}
        rec["wall_s"] = round(time.time() - t0, 1)
        report["shapes"].append(rec)
        del reals, copies
        torch.cuda.empty_cache()
    # per-step totals at M = 1 (GEMM only; act_quant is unchanged by T9b-1)
    st = {"ms_gemv": 0.0, "ms_ref": 0.0}
    for rec in report["shapes"]:
        m1 = rec["per_m"].get(1)
        if m1:
            st["ms_gemv"] += rec["calls_per_step"] * m1["us_gemv"] / 1e3
            st["ms_ref"] += rec["calls_per_step"] * m1["us_ref"] / 1e3
    st = {k: round(v, 3) for k, v in st.items()}
    st["saved_ms"] = round(st["ms_ref"] - st["ms_gemv"], 3)
    st["target_saved_ms"] = GATE_STEP_SAVING_MS
    report["per_step_m1"] = st
    report["gates"] = overall_gates(report["shapes"], st)
    picked = {f"{r['shape'][0]}x{r['shape'][1]}/M{M}": json.dumps(v["config"], sort_keys=True)
              for r in report["shapes"] for M, v in r["per_m"].items()}
    report["distinct_configs"] = {"n_distinct": len(set(picked.values())), "n_keys": len(picked), "picked": picked}
    report["tuned"] = G.describe()
    report["tune_cache"] = G.save_cache(a.tune_cache) if a.impl != "gemv3" else None
    st3: dict = {}
    if a.impl != "gemv":
        st3 = step_totals3(report["shapes"], Ms)
        report["per_step3"] = st3
        report["gates3"] = overall_gates3(report["shapes"], st3)
        report["tune_cache3"] = G.save_cache3(a.tune_cache3)
    report["wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1, default=str)
    if a.impl != "gemv3":
        print(f"per step at M=1 (GEMMs only): ref {st['ms_ref']:.2f} ms -> gemv {st['ms_gemv']:.2f} ms "
              f"(saves {st['saved_ms']:.2f} ms over {calls_per_step} calls; target >= {GATE_STEP_SAVING_MS:.0f} ms)")
        print(f"tuned configs: {report['distinct_configs']['n_distinct']} distinct over "
              f"{report['distinct_configs']['n_keys']} (shape, M) keys")
        print(f"GATES: {json.dumps(report['gates'])}")
    if a.impl != "gemv":
        print("per step, GEMV only (sum of calls x us): " + ", ".join(f"{k} {v}" for k, v in st3.items()))
        print(f"GATES3: {json.dumps(report['gates3'])}")
    print(f"report {a.json}; tuned configs {report['tune_cache']} / {report.get('tune_cache3')}")
    gates = report["gates3"] if a.impl != "gemv" else report["gates"]
    return 0 if gates["all_pass"] else 1


def gemv_err_vs_f64(G, xq, xs, w0, s0, dev) -> Optional[float]:
    """gemv's mean |err| vs the float64 product (the T9b-6 oracle gate's baseline) with gemv's cached config for this
    (N, K, M) -- else its default, never a fresh tune -- for `--impl gemv3`."""
    M, K = xq.shape
    N = w0.size(0)
    if N * K > (64 << 20):
        return None
    if G.cache_path():
        G.load_cache()
    cfg = G._CONFIGS.get(G.config_key(N, K, M)) or G.default_config(N, K, M, G.num_sms(dev))
    got = G.mxfp8_gemv(xq, xs, w0, s0, torch.empty(M, N, dtype=torch.bfloat16, device=dev), config=cfg)
    return float((got.double() - exact_f64(xq, xs, w0, s0)).abs().mean())


def print_gemv3(N, K, rs, M, r3: dict, n_layers: int) -> None:
    ratio = r3["err_ratio_vs_gemv"]
    ratio = None if ratio is None else round(ratio, 5)
    print(f"[{N}x{K}] {'/'.join(rs)} M={M}: gemv3 {r3['us']:.2f} us ({r3['GBps']:.0f} GB/s) | mismatch vs ref "
          f"{r3['mismatch'] * 100:.3f} % over {n_layers} layers, max ulp {r3['max_ulp']} | rows "
          f"{r3['row_invariant']} replay {r3['replay_identical']} same-plan==gemv {r3['bitident_same_plan']} "
          f"f64 ratio {ratio} | config {r3['config']} grid {tuple(r3['grid'])} EVEN {r3['EVEN']}, "
          f"{r3['bytes_per_thread']} B/thread/iter, {r3['inflight_bytes']} B in flight/program | tune {r3['tune']}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
