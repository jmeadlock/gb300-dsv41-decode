#!/usr/bin/env python3
"""t9b5_bench.py -- T9b-5 (docs/HANDOFF-T9B5-FP32-FIXED.md): the fixed-order fp32 kernels (runtime/static_kernels.py
t9b5_head_fp32 / t9b5_mix_fp32) vs cuBLAS F.linear at the real shapes, graph-timed, over a launch-config sweep.

  python3 bench/t9b5_bench.py [--rows 1,5,6,8] [--model-dir /models/DeepSeek-V4.1-...] [--kb-sweep] \\
      [--json docs/t9b5-bench-<date>.json]

  * per site (head [R, 5120] x [129280, 5120]^T fp32 weight = 2.65 GB; mixes [R, 20480] x [24, 20480]^T * rsqrt) and
    R: every (RB, BN, U, num_warps, num_stages) of the sweep -> us per call, TB/s over the weight bytes; each config's
    output asserted bit-equal to the default config's (the knobs are speed-only by construction) and row r at R rows
    bit-equal to row r alone; cuBLAS F.linear at the same R (and whether IT is row-invariant on these inputs);
  * the best config per (site, RB) printed as a DSV41_T9B5_CFG string (paste it into static_kernels.T9B5_CFG or the
    environment);
  * --kb-sweep: the ORDER alternatives (KB lanes; a numerics change, so informative only: us and the fp64-oracle
    distance on 64 rows per KB) -- FIXED_KB in runtime/fused_glue_r2.py is the shipped choice;
  * --model-dir: the real head weight (bf16 in the checkpoint, widened to fp32 as the loader does), else synthetic.

Same docker line as tests/gpu/test_fp32_fixed_gpu.py with `bench/t9b5_bench.py` as the script.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

DIM, HC, V, MIX = 5120, 4, 129280, 24


def sweep_configs(site: str, R: int):
    """Speed-only knobs: (RB, BN, U, num_warps, num_stages)."""
    out = [None]
    if site == "head":
        rbs = (1, 2, 4, 8) if R <= 1 else (4, 8)
        for rb, bn, u, nw, ns in itertools.product(rbs, (8, 16, 32, 64), (2, 4, 8), (4, 8), (1, 2)):
            if rb * bn * 64 > 32768:      # accumulators per program (fp32): keep under ~128 regs/thread at 8 warps
                continue
            out.append((rb, bn, u, nw, ns))
    else:
        for rb, bn, u, nw, ns in itertools.product((1, 2), (1, 2, 4), (4, 8, 20, 40), (4, 8, 16), (1, 2)):
            out.append((rb, bn, u, nw, ns))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,5,6,8")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--kb-sweep", action="store_true")
    ap.add_argument("--quick", action="store_true", help="a few configs per site (smoke)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F

    from runtime import fused_glue_r2 as R2
    from runtime import static_kernels as SK
    from runtime.gemv_mxfp8 import time_graph

    dev = "cuda"
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "device": torch.cuda.get_device_name(),
           "fixed_kb": dict(R2.FIXED_KB), "default_cfg": {k: {str(rb): v for rb, v in d.items()}
                                                          for k, d in SK.T9B5_CFG.items()}, "sites": {}}
    g = torch.Generator(device="cpu").manual_seed(0)
    if a.model_dir:
        from runtime.ckpt import ShardIndex

        src = ShardIndex.scan(a.model_dir).read_tensor("head.weight")
        rep["head_weight"] = f"real ({src.dtype} in the checkpoint) widened to fp32"
        Wh = src.to(dev).float()
        del src
    else:
        Wh = torch.empty(V, DIM, dtype=torch.float32, device=dev)
        for i in range(0, V, 8192):
            n = min(8192, V - i)
            Wh[i:i + n] = (torch.randn(n, DIM, generator=g, device="cpu") * 0.02).to(torch.bfloat16).float().to(dev)
        rep["head_weight"] = "synthetic bf16-valued fp32"
    Wm = (torch.randn(MIX, HC * DIM, generator=g, device="cpu") * 0.01).to(dev)
    best_env = []
    for site, W in (("head", Wh), ("mix", Wm)):
        K = W.size(1)
        kb = R2.fixed_kb(site, K)
        nb = W.numel() * W.element_size()
        srep = rep["sites"][site] = {"shape": list(W.shape), "kb": kb, "weight_bytes": nb, "rows": {}}
        reps = 10 if site == "head" else 100
        for R in rows:
            if site == "head":
                x = (torch.randn(R, K, generator=g, device="cpu") * 2).to(torch.bfloat16).to(dev)
                sc = None
            else:
                x = (torch.randn(R, K, generator=g, device="cpu") * 3).to(torch.bfloat16).float().to(dev)
                sc = (torch.rand(R, generator=g, device="cpu") + 0.25).to(dev)
            out = torch.empty(R, W.size(0), dtype=torch.float32, device=dev)
            SK.fixed_dot(x, W, out, kb=kb, site=site, scale=sc)
            base = out.clone()
            singles = []
            for i in range(R):
                o1 = torch.empty(1, W.size(0), dtype=torch.float32, device=dev)
                SK.fixed_dot(x[i:i + 1], W, o1, kb=kb, site=site, scale=sc[i:i + 1] if sc is not None else None)
                singles.append(o1)
            row_inv = torch.equal(torch.cat(singles).view(torch.int32), base.view(torch.int32))
            cfgs = sweep_configs(site, R)
            if a.quick:
                cfgs = cfgs[:6]
            res = []
            for cfg in cfgs:
                try:
                    f = lambda c=cfg: SK.fixed_dot(x, W, out, kb=kb, site=site, scale=sc, cfg=c)  # noqa: E731
                    f()
                    torch.cuda.synchronize()
                    same = torch.equal(out.view(torch.int32), base.view(torch.int32))
                    us = time_graph([f] * reps, reps=3)
                    res.append({"cfg": list(SK.t9b5_config(site, R)) if cfg is None else list(cfg),
                                "default": cfg is None, "us": us, "TBps": nb / us / 1e6, "bit_equal_default": same})
                    if not same:
                        print(f"[t9b5] {site} R={R} cfg {cfg}: OUTPUT DIFFERS from the default config", flush=True)
                except Exception as e:  # a config that does not compile here is reported, not fatal
                    res.append({"cfg": list(cfg) if cfg else None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            fc = lambda: F.linear(x.float(), W) * sc.view(-1, 1) if sc is not None else F.linear(x.float(), W)  # noqa
            fc()
            usc = time_graph([fc] * reps, reps=3)
            cb6 = fc()
            cb1 = torch.cat([(F.linear(x[i:i + 1].float(), W) * sc[i:i + 1].view(-1, 1)) if sc is not None
                             else F.linear(x[i:i + 1].float(), W) for i in range(R)])
            ok = [r_ for r_ in res if "us" in r_]
            best = min(ok, key=lambda r_: r_["us"]) if ok else None
            dflt = next((r_ for r_ in ok if r_["default"]), None)
            srep["rows"][str(R)] = {"row_invariant": row_inv, "all_configs_bit_equal": all(
                r_.get("bit_equal_default") for r_ in ok), "configs": res, "best": best, "default": dflt,
                "cublas_us": usc, "cublas_TBps": nb / usc / 1e6,
                "cublas_row_invariant": bool(torch.equal(cb6.view(torch.int32), cb1.view(torch.int32)))}
            print(f"[t9b5] {site:4s} R={R}: default {dflt['us'] if dflt else float('nan'):8.2f} us | best "
                  f"{best['us'] if best else float('nan'):8.2f} us {best['cfg'] if best else None} "
                  f"({best['TBps'] if best else 0:.2f} TB/s) | cuBLAS {usc:8.2f} us | ours row-invariant {row_inv}, "
                  f"cuBLAS row-invariant {srep['rows'][str(R)]['cublas_row_invariant']}, "
                  f"{len(ok)}/{len(res)} configs ran", flush=True)
            if best:
                rb = best["cfg"][0]
                best_env.append(f"{site}:{rb}={','.join(str(v) for v in best['cfg'][1:])}")
        if a.kb_sweep:
            srep["kb_sweep"] = {}
            x = (torch.randn(64, K, generator=g, device="cpu") * 2).to(torch.bfloat16).to(dev)
            orc = x.double() @ W.double().t()
            mag = x.double().abs() @ W.double().abs().t()
            for kb2 in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
                if K % kb2:
                    continue
                o = torch.empty(64, W.size(0), dtype=torch.float32, device=dev)
                try:
                    f = lambda: SK.fixed_dot(x[:6], W, o[:6], kb=kb2, site=site)  # noqa: E731
                    f()
                    us = time_graph([f] * reps, reps=3)
                    SK.fixed_dot(x, W, o, kb=kb2, site=site)
                    be = ((o.double() - orc).abs() / (2.0 ** -24 * mag.clamp_min(1e-300)))
                    srep["kb_sweep"][str(kb2)] = {"us_R6": us, "be_max": float(be.max()), "be_mean": float(be.mean())}
                    print(f"[t9b5] {site:4s} KB={kb2:5d}: {us:8.2f} us at R=6 (default cfg), u*sum|xw| max "
                          f"{float(be.max()):.3f} mean {float(be.mean()):.4f}", flush=True)
                except Exception as e:
                    srep["kb_sweep"][str(kb2)] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            del orc, mag
    rep["best_env"] = ";".join(dict.fromkeys(best_env))
    print(f"[t9b5] best configs: {SK.T9B5_CFG_ENV}=\"{rep['best_env']}\"")
    path = a.json or os.path.join(REPO, "docs", f"t9b5-bench-{time.strftime('%Y%m%d')}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"[t9b5] -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
