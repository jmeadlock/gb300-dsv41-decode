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
  * config invariance: every runtime/gemv_mxfp8.CANDIDATES config bit-identical to the tuned one (first layer);
  * timing: each impl captured as one CUDA graph of >= 64 calls cycling over enough weight copies (real layers, then
    clones of them) to exceed --stream-mib (default 512 MiB > L2), so weights stream from HBM as in the decode step;
    µs/call = replay time / calls (no launch gaps: the graph is what the step replays). GB/s = (weight + weight-scale
    + activation + activation-scale + output bytes) / time;
  * gates per shape: GEMV >= 2x faster than the reference at M=1; shapes >= 32 MiB: >= 3 TB/s at M=1.
Also: the per-step GEMM time at M=1 (sum over roles of calls x µs) for both impls, and the tuned configs (written to
--tune-cache, which runtime/gemv_mxfp8 reads via $DSV41_GEMV_CACHE).

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
from typing import Dict, List, Tuple

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
    x = (torch.randn(M, K // 32, 32, generator=g) * torch.exp2(torch.randint(-6, 7, (M, K // 32, 1), generator=g)
                                                                .float())).view(M, K).to(torch.bfloat16).cuda()
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
    a = ap.parse_args(argv)
    os.environ["DSV41_GEMV_CACHE"] = a.tune_cache
    os.environ.setdefault("DSV41_GEMV_TUNE", "1")

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
               "timing_copies": n_copies, "timing_stream_MiB": round(n_copies * wbytes / 2**20, 1), "per_m": {}}
        for M in Ms:
            xq, xs = activations(M, K, 1000 * M + N % 997 + K, K_mod)
            out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
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
            # config invariance
            base = G.mxfp8_gemv(xq, xs, w0, s0, out, config=cfg).clone()
            variant = [c.as_dict() for c in G.CANDIDATES
                       if not torch.equal(G.mxfp8_gemv(xq, xs, w0, s0, out, config=c), base)]
            # timing under a CUDA graph, weights cycling over the copies
            ncalls = max(a.calls, n_copies)
            us_gemv = G.time_graph([lambda i=i: G.mxfp8_gemv(xq, xs, copies[i % n_copies][0], copies[i % n_copies][1],
                                                             out, config=cfg) for i in range(ncalls)], a.reps)
            us_ref = G.time_graph([lambda i=i: K_mod.fp8_gemm(xq, xs, copies[i % n_copies][0], copies[i % n_copies][1],
                                                              E8, block_size=32) for i in range(ncalls)], a.reps)
            bm = bytes_moved(M, N, K)
            r = {"config": cfg.as_dict(), "us_gemv": round(us_gemv, 3), "us_ref": round(us_ref, 3),
                 "speedup": round(us_ref / us_gemv, 2), "GBps_gemv": round(bm / us_gemv / 1e3, 1),
                 "GBps_ref": round(bm / us_ref / 1e3, 1), "mismatch": mism / tot, "outputs_compared": tot,
                 "max_abs": max_abs, "max_ulp": max_ulp, "n_gt1ulp": gt1, "mismatch_per_layer": per_layer,
                 "mean_abs_err_vs_f64": err, "config_invariant": not variant, "configs_differing": variant}
            rec["per_m"][M] = r
            print(f"[{N}x{K}] {'/'.join(rs)} M={M}: gemv {us_gemv:.2f} us ({r['GBps_gemv']:.0f} GB/s) vs ref "
                  f"{us_ref:.2f} us ({r['GBps_ref']:.0f} GB/s) = {r['speedup']:.1f}x | mismatch "
                  f"{r['mismatch'] * 100:.3f} % over {len(reals)} layers, max ulp {max_ulp} (>1: {gt1}), "
                  f"config {cfg.as_dict()}, invariant {not variant}", flush=True)
        m1 = rec["per_m"].get(1)
        if m1 is not None:
            big = wbytes >= (32 << 20)
            rec["gates"] = {"speedup_ge_2x_m1": m1["speedup"] >= 2.0,
                            "bw_ge_3TBps_m1": (m1["GBps_gemv"] >= 3000.0) if big else None,
                            "mismatch_le_0p5pct": all(v["mismatch"] <= 5e-3 for v in rec["per_m"].values())}
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
    report["per_step_m1"] = st
    gs = [r["gates"] for r in report["shapes"] if r.get("gates")]
    report["gates"] = {
        "every_shape_ge_2x_m1": all(g["speedup_ge_2x_m1"] for g in gs) if gs else None,
        "big_shapes_ge_3TBps_m1": all(g["bw_ge_3TBps_m1"] for g in gs if g["bw_ge_3TBps_m1"] is not None) if gs else None,
        "drift_1ulp_class": all(g["mismatch_le_0p5pct"] for g in gs) if gs else None,
        "shapes_over_0p5pct": [r["shape"] for r in report["shapes"]
                               if any(v["mismatch"] > 5e-3 for v in r["per_m"].values())],
        "config_invariant_all": all(v["config_invariant"] for r in report["shapes"] for v in r["per_m"].values()),
    }
    report["tuned"] = G.describe()
    report["tune_cache"] = G.save_cache(a.tune_cache)
    report["wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"per step at M=1 (GEMMs only): ref {st['ms_ref']:.2f} ms -> gemv {st['ms_gemv']:.2f} ms "
          f"(saves {st['saved_ms']:.2f} ms over {calls_per_step} calls)")
    print(f"GATES: {json.dumps(report['gates'])}")
    print(f"report {a.json}; tuned configs {report['tune_cache']}")
    ok = report["gates"]["every_shape_ge_2x_m1"] and report["gates"]["big_shapes_ge_3TBps_m1"] is not False \
        and report["gates"]["drift_1ulp_class"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
