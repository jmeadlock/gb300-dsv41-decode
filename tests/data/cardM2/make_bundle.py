#!/usr/bin/env python3
"""Rebuild tests/data/cardM2/run2-armC.npz from the Card M2 receipts in the milo-moe-manager checkout (data only: no
m3 code is imported or executed). Provenance (sha256 of every source file) goes to provenance.json next to it.

    .venv/bin/python tests/data/cardM2/make_bundle.py [--m3 ~/hermes/milo-moe-manager] [--arm C]

What Card M2 run 2 arm C is (cards/M2/BRIEF-M2.md): DSV4.1-Flash-1017 on the GB300 with m3's own HOOK dir
(m3.actuators.flashinfer_split driving m3.control.policy, the file runtime/residency_policy.py vendors), boot map
rowmap-mixed-ar-v1.json (sha256 9b55d83a...), knobs = shipped v26 (drain_n 64, swaps_per_drain 2, halflife 500, hyst
1.5, min_abs 8, swaps_per_barrier 8), unfrozen at host step 12215, a dump every ~15 s. Its swaps.jsonl is therefore
every decision m3's policy made live; each counts-post dump holds the mirror rows and the EWMA after the last drain.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", default=os.path.expanduser("~/hermes/milo-moe-manager"))
    ap.add_argument("--arm", default="C")
    a = ap.parse_args()
    card = os.path.join(a.m3, "cards", "M2")
    snap = os.path.join(card, "results-run2", f"arm-{a.arm}", "snap")
    rowmap = os.path.join(card, "m3-hook", "rowmap-mixed-ar-v1.json")
    files = sorted(glob.glob(os.path.join(snap, "counts-post-*.npz")))
    with open(rowmap) as f:
        rm = json.load(f)["layers"]
    L = len(rm)
    boot_hot = np.array([rm[str(l)]["hot"] for l in range(L)], dtype=np.int16)
    boot_cold = np.array([rm[str(l)]["cold"] for l in range(L)], dtype=np.int16)
    zs = [np.load(p) for p in files]
    sw = []
    with open(os.path.join(snap, "swaps.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "layer" in r:
                sw.append(r)
    tel = []
    with open(os.path.join(snap, "telemetry.jsonl")) as f:
        for line in f:
            tel.append(json.loads(line))
    drains = [t for t in tel if t.get("dsteps", 0) > 0]
    go = [t for t in tel if str(t.get("event", "")).startswith("ctrl freeze=0")]
    out = os.path.join(HERE, f"run2-arm{a.arm}.npz")
    np.savez_compressed(
        out,
        boot_hot=boot_hot, boot_cold=boot_cold,
        snap_host_steps=np.array([int(z["host_steps"]) for z in zs], dtype=np.int64),
        snap_hot=np.stack([z["hot"] for z in zs]).astype(np.int16),
        snap_cold=np.stack([z["cold"] for z in zs]).astype(np.int16),
        snap_ewma=np.stack([z["ewma"] for z in zs]).astype(np.float64),
        snap_cnt=np.stack([z["cnt"] for z in zs]).astype(np.int64),
        sw_layer=np.array([r["layer"] for r in sw], dtype=np.int16),
        sw_in=np.array([r["in_id"] for r in sw], dtype=np.int16),
        sw_out=np.array([r["out_id"] for r in sw], dtype=np.int16),
        sw_step=np.array([r["step"] for r in sw], dtype=np.int64),
        sw_apply_step=np.array([r["apply_step"] for r in sw], dtype=np.int64),
        sw_ewma_in=np.array([r["ewma_in"] for r in sw], dtype=np.float64),
        sw_ewma_out=np.array([r["ewma_out"] for r in sw], dtype=np.float64),
        sw_ewma_coolest=np.array([r["ewma_coolest_hot"] for r in sw], dtype=np.float64),
        drain_host_steps=np.array([d["host_steps"] for d in drains], dtype=np.int64),
        drain_dsteps=np.array([d["dsteps"] for d in drains], dtype=np.int64),
        drain_sel=np.array([d["sel"] for d in drains], dtype=np.int64),
        drain_cold=np.array([d["cold"] for d in drains], dtype=np.int64),
        drain_queued=np.array([d["queued"] for d in drains], dtype=np.int64),
        go_host_steps=np.array([go[0]["host_steps"] if go else -1], dtype=np.int64),
    )
    prov = {"source_checkout": a.m3, "arm": a.arm, "bundle": os.path.basename(out), "bundle_sha256": sha(out),
            "files": {os.path.relpath(p, a.m3): sha(p) for p in
                      [rowmap, os.path.join(snap, "swaps.jsonl"), os.path.join(snap, "telemetry.jsonl")] + files},
            "n_snapshots": len(zs), "n_swaps": len(sw), "n_drains": len(drains),
            "ctrl_go": go[0]["event"] if go else None}
    with open(os.path.join(HERE, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps({k: v for k, v in prov.items() if k != "files"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
