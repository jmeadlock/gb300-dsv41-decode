#!/usr/bin/env python3
"""head_bench.py -- M2-T17 (docs/HANDOFF-M2-T17-BF16-HEAD.md): the head read as bf16 (its checkpoint dtype) by the
T9b-5 fixed-order kernel vs the fp32-stored head (the widened copy the loader makes today), graph-timed over a launch
sweep, with the bit-equality gate on every config.

  python3 bench/head_bench.py [--rows 1,2,5,6,8] [--model-dir /models/DeepSeek-V4.1-...] [--quick] \\
      [--json scratch/head-bench-<date>.json]

  per R: every (RB, BN, U, num_warps, num_stages) of the sweep x {bf16-stored, fp32-stored} -> us per call and TB/s over
  the bytes that storage reads (bf16 1.32 GB, fp32 2.65 GB); each output asserted BIT-EQUAL to the fp32-stored default
  config's (same products, same order: the knobs and the storage are speed-only) and row r of the R-row call bit-equal to
  row r alone; cuBLAS F.linear on the widened weight at the same R for reference. The best bf16 config per RB is printed as
  a DSV41_T9B5_CFG string ("head_bf16:<RB>=BN,U,warps,stages"): paste it into static_kernels.T9B5_CFG["head_bf16"]
  (the shipped table is an untuned starting point) or export it. The handoff's target: <= 250 us at R = 1 and 6.
  --model-dir: the real head.weight (bf16 in the checkpoint), else a synthetic bf16 head of the same shape.
  The receipt goes to scratch/ (never docs/).

The 1-row step delta and the 6-row step / headline are the existing benches, A/B'd by the environment switch:
  DSV41_HEAD_DTYPE=fp32 python3 bench/first_light.py --fp32-impl fixed ... --json scratch/fl-head-fp32.json
  DSV41_HEAD_DTYPE=bf16 python3 bench/first_light.py --fp32-impl fixed ... --json scratch/fl-head-bf16.json
  DSV41_HEAD_DTYPE=bf16 python3 bench/headline.py --fp32-impl fixed ...     (head + draft's shared / Markov heads bf16)
(runtime/loader.py: build_and_load(head_dtype=...) reads $DSV41_HEAD_DTYPE; the boot log line says `head bf16`.)
Same docker line as tests/gpu/test_head_bf16_gpu.py with `bench/head_bench.py` as the script.
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

DIM, V = 5120, 129280
TARGET_US = 250.0


def sweep_configs(R: int, quick: bool):
    """Speed-only knobs: None (the shipped default for the storage) + (RB, BN, U, num_warps, num_stages)."""
    out = [None]
    rbs = (1, 2) if R <= 1 else (4, 8)
    for rb, bn, u, nw, ns in itertools.product(rbs, (8, 16, 32, 64), (4, 8, 16), (4, 8), (1, 2, 3)):
        if rb * bn * 64 > 32768:      # fp32 accumulators per program: keep under ~128 regs/thread at 8 warps
            continue
        out.append((rb, bn, u, nw, ns))
    return out[:7] if quick else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,2,5,6,8")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="a few configs per R (smoke)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F

    from runtime import fused_glue_r2 as R2
    from runtime import static_kernels as SK
    from runtime.gemv_mxfp8 import time_graph

    dev = "cuda"
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    g = torch.Generator(device="cpu").manual_seed(0)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "device": torch.cuda.get_device_name(),
           "target_us": TARGET_US, "kb": R2.fixed_kb("head", DIM),
           "tables": {k: {str(rb): list(v) for rb, v in d.items()} for k, d in SK.T9B5_CFG.items() if k != "mix"},
           "rows": {}}
    if a.model_dir:
        from runtime.ckpt import ShardIndex

        src = ShardIndex.scan(a.model_dir).read_tensor("head.weight")
        assert src.dtype == torch.bfloat16, f"the checkpoint stores the head as {src.dtype}"
        Wb = src.to(dev)
        rep["head_weight"] = "real (bf16 in the checkpoint)"
        del src
    else:
        Wb = torch.empty(V, DIM, dtype=torch.bfloat16, device=dev)
        for i in range(0, V, 8192):
            n = min(8192, V - i)
            Wb[i:i + n] = (torch.randn(n, DIM, generator=g, device="cpu") * 0.02).to(torch.bfloat16).to(dev)
        rep["head_weight"] = "synthetic bf16"
    Wf = Wb.float()
    nb = {"bf16": Wb.numel() * Wb.element_size(), "fp32": Wf.numel() * Wf.element_size()}
    rep["weight_bytes"] = dict(nb)
    rep["hbm_saved_bytes"] = nb["fp32"] - nb["bf16"]
    print(f"[head] weight bytes: bf16 {nb['bf16'] / 1e9:.3f} GB, fp32 {nb['fp32'] / 1e9:.3f} GB "
          f"(saves {rep['hbm_saved_bytes'] / 1e9:.3f} GB); {rep['head_weight']}", flush=True)
    best_env = []
    all_equal = True
    for R in rows:
        x = (torch.randn(R, DIM, generator=g, device="cpu") * 2).to(torch.bfloat16).to(dev)
        out = torch.empty(R, V, dtype=torch.float32, device=dev)
        ref = torch.empty_like(out)
        SK.fixed_dot(x, Wf, ref, kb=rep["kb"], site="head")            # the fp32-stored default: the reference bits
        ref = ref.clone()
        singles = []
        for i in range(R):
            o1 = torch.empty(1, V, dtype=torch.float32, device=dev)
            SK.fixed_dot(x[i:i + 1], Wb, o1, kb=rep["kb"], site="head")
            singles.append(o1)
        row_inv = torch.equal(torch.cat(singles).view(torch.int32), ref.view(torch.int32))
        res = {"bf16": [], "fp32": []}
        reps = 10
        for tag, W_ in (("bf16", Wb), ("fp32", Wf)):
            for cfg in sweep_configs(R, a.quick):
                try:
                    f = lambda c=cfg, w_=W_: SK.fixed_dot(x, w_, out, kb=rep["kb"], site="head", cfg=c)  # noqa: E731
                    f()
                    torch.cuda.synchronize()
                    same = torch.equal(out.view(torch.int32), ref.view(torch.int32))
                    all_equal &= same
                    us = time_graph([f] * reps, reps=3)
                    res[tag].append({"cfg": list(SK.t9b5_config("head", R, W_.dtype)) if cfg is None else list(cfg),
                                     "default": cfg is None, "us": us, "TBps": nb[tag] / us / 1e6,
                                     "bit_equal_fp32_default": same})
                    if not same:
                        print(f"[head] {tag}-stored R={R} cfg {cfg}: OUTPUT DIFFERS from the fp32-stored default",
                              flush=True)
                except Exception as e:  # a config that does not compile here is reported, not fatal
                    res[tag].append({"cfg": list(cfg) if cfg else None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        fc = lambda: F.linear(x.float(), Wf)  # noqa: E731
        fc()
        usc = time_graph([fc] * reps, reps=3)
        summ = {"row_invariant_bf16": row_inv, "cublas_us": usc, "cublas_TBps": nb["fp32"] / usc / 1e6}
        for tag in ("bf16", "fp32"):
            ok = [r_ for r_ in res[tag] if "us" in r_]
            summ[tag] = {"best": min(ok, key=lambda r_: r_["us"]) if ok else None,
                         "default": next((r_ for r_ in ok if r_["default"]), None), "n_ok": len(ok), "n": len(res[tag])}
        rep["rows"][str(R)] = {"configs": res, **summ}
        b, d32 = summ["bf16"], summ["fp32"]
        met = "MET" if b["best"] and b["best"]["us"] <= TARGET_US else "MISSED"
        print(f"[head] R={R}: bf16 default {b['default']['us'] if b['default'] else float('nan'):8.2f} us | bf16 best "
              f"{b['best']['us'] if b['best'] else float('nan'):8.2f} us {b['best']['cfg'] if b['best'] else None} "
              f"({b['best']['TBps'] if b['best'] else 0:.2f} TB/s) | fp32 default "
              f"{d32['default']['us'] if d32['default'] else float('nan'):8.2f} us | cuBLAS fp32 {usc:8.2f} us | "
              f"R-row == rows alone: {row_inv} | target <= {TARGET_US:.0f} us: {met}", flush=True)
        if b["best"]:
            rb = b["best"]["cfg"][0]
            best_env.append(f"head_bf16:{rb}={','.join(str(v) for v in b['best']['cfg'][1:])}")
    rep["all_configs_bit_equal_fp32_default"] = bool(all_equal)
    rep["best_env"] = ";".join(dict.fromkeys(best_env))
    print(f"[head] every config of both storages bit-equal to the fp32-stored default: {all_equal}")
    print(f"[head] best bf16 configs: {SK.T9B5_CFG_ENV}=\"{rep['best_env']}\"")
    path = a.json or os.path.join(REPO, "scratch", f"head-bench-{time.strftime('%Y%m%d')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"[head] -> {path}")
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
