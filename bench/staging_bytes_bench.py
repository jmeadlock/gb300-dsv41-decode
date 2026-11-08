#!/usr/bin/env python3
"""staging_bytes_bench.py -- M3-S2 (docs/HANDOFF-M3-S2-STAGING-BYTES.md, results docs/M3-S2-STAGING-BYTES.md): what
does the 6-row step's staging launch move per layer-call, why does sf_store "pool" not fit, and what does the
staging still expose after M3-S1's kernel-overlap (19.34 ms/step vs 16.64 with staging skipped)?

Subcommands (CPU unless marked; every number is labeled [M] measured, [D] derived from receipts, [E] modeled,
[A] assumed):
  census    bytes per layer-call by term -- (a) cold expert weights, (b) cold expert scales, (c) HOT expert scales
            (only under sf_store "step"), (d) the rest -- COUNTED [M] on a synthetic layer through the runtime's own
            dedup (moe_rows.dedup_routes + rows_layout) and staging contracts (dg_inplace.stage_rows_emulated,
            staging_overlap.stage_proj_emulated) for both stores, then scaled by the real rows' bytes at the receipts'
            hot / cold unique counts; and the time attribution of the serial 113.8 us m1_stage_rows [D] from the M3-S1
            probe's cold-only (pool-style) launch times.
  pool-fit  the pool-owned packed SF derived (13.14 GiB), and what fits per $DSV41_SF_POOL_HEADROOM_GIB x
            --pool-spare / --spare-slots, decided by runtime/dg_inplace.sf_pool_plan (the code the build runs) on
            the box's free HBM at the check (4.93 GiB, every headline log), with ms/step [E] per arm.
  exposure  the two headline receipts (kernel-overlap 19.34 / 18.65 ms vs none-diag 16.64 / 16.62, two cells at
            two cold shares) against the M3-S1 probe's pool-style per-c exposure: how much of the exposed 2.7 ms is
            per-cold-expert and how much is c-independent (the hot-SF repack + fork/join) [D].
  hist      summarize a $DSV41_STAGE_HIST json (runtime/staging_overlap.StageOverlap.count, box): cold-count
            distribution, overflow risk per staging budget, cold removed by +N hot experts per layer.
  probe     (GPU, box) real shapes (bench/staging_overlap_bench.py's synthetic layer: hot pool + packed SF in HBM,
            cold experts in a registered hugepage HostBuffer, 22 unique experts per round): step-style work lists
            (cold slots + the used HOT slots, sf_store "step") vs pool-style (cold slots only) for
              A serial   m1_stage_rows and s1_stage_proj x 3 (runtime cfg 48,16,16 / 8K,32K,32K), c = 0..2,
                         plus the hot-scale pack alone (weights=False)
              B overlap  kernel-overlap against the real DeepGEMM contiguous16 chain + the shared-expert stand-in,
                         c = 0..2: exposed_us step-style vs pool-style (= what a pooled layer saves)
              C hot sweep c = 0, step-style with h of the used hot slots listed: exposed_us vs h
              D checks   step-style and pool-style write the same slot / SF bytes and the same GEMM outputs
  all       census + pool-fit + exposure (+ probe with CUDA).

Box (library image, repo at /work, DG JIT cache mounted; no checkpoint):
  python3 bench/staging_bytes_bench.py probe --out scratch/staging-bytes-probe-<date>.json
deep_gemm comes from vllm's third_party WITHOUT importing vllm (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime import dg_inplace as DI  # noqa: E402
from runtime import moe_rows as MR  # noqa: E402
from runtime import staging_overlap as SO  # noqa: E402

GiB = 2 ** 30
PROJS = ("w1", "w3", "w2")
# ---- the real model (docs/checkpoint-1017-inventory.json; runtime/dg_inplace.hbm_plan docstring)
DIM, INTER, TOPK, ROWS, LAYERS, EXPERTS = 5120, 2304, 6, 6, 40, 384
NK = {"w1": (INTER, DIM), "w3": (INTER, DIM), "w2": (DIM, INTER)}       # (N, K)
W_ROW = {p: n * k // 2 for p, (n, k) in NK.items()}                      # fp4 bytes per projection row
S_ROW = {p: n * k // 32 for p, (n, k) in NK.items()}                     # e8m0 bytes = packed int32 bytes
EXPERT_W = sum(W_ROW.values())                                           # 17,694,720
EXPERT_S = sum(S_ROW.values())                                           # 1,105,920
HOT_SEED = 295                                                           # docs/maps/hot-v26ar-seed-295.json
# ---- receipts (paths read when present; these constants are their values, for a checkout without docs/)
R_ANATOMY_STAGE_US = 113.8          # docs/gap-tf-anatomy-20261107.txt: m1_stage_rows 4.55 ms / 40 launches [M]
R_TF = {"hot_unique": 195298, "cold_unique": 15236, "layer_calls": 9640}   # docs/gap-tf-static-20261107.json [M]
R_PROBE_M1_US = {0: 2.83, 1: 52.46, 2: 100.49, 3: 148.7, 4: 199.13}       # docs/s1-bench-20261107.json A [M]
R_PROBE_OV_US = {0: 3.24, 1: 17.37, 2: 48.57, 3: 95.28, 4: 141.59}        # ... B kernel-overlap/np48b8k [M]
R_PROBE_SERIAL_MS, R_SERIAL_MS = 3.404, 4.552   # probe serial at Poisson(1.58) vs the anatomy's 4.552 [M]
R_HEADLINE = {"kernel-overlap": {"two_graphs": (19.342, 0.05453), "one_graph": (18.652, 0.01635)},
              "none-diag": {"two_graphs": (16.637, 0.04583), "one_graph": (16.624, 0.00541)}}   # [M]
R_FREE_AT_CHECK_GIB = 4.93          # `[dg-inplace] sf_store pool needs 13.14 GiB, 4.93 GiB free` (every headline) [M]
R_FREE_AFTER_BUILD_GIB = 3.631      # docs/headline-s1-shapes-20261107.json hbm_after_build_GiB.free [M]
R_STEP = {"two_graphs": 19.342, "nonediag": 16.637}
R_RESIDENCY_GAIN_MS = 0.31          # LEDGER 2026-11-07 20:40: residency on vs off, serial staging era [M]
CUR_POOL_SPARE, CUR_STAGE, CUR_RES = 24, 16, 8


def _date() -> str:
    return time.strftime("%Y%m%d")


def _load(path):
    p = os.path.join(REPO, path)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def receipts() -> dict:
    """The receipts' numbers, re-read from docs/ when the files are there (else the constants above)."""
    out = {"tf": dict(R_TF), "probe_m1_us": dict(R_PROBE_M1_US), "probe_ov_us": dict(R_PROBE_OV_US),
           "headline": json.loads(json.dumps(R_HEADLINE)), "sources": {}}
    tf = _load("docs/gap-tf-static-20261107.json")
    if tf:
        ps = [p for r in tf.get("reps", []) for p in r.get("prompts", [])]
        if ps:
            out["tf"] = {"hot_unique": sum(p["hot_unique"] for p in ps),
                         "cold_unique": sum(p["cold_unique"] for p in ps),
                         "layer_calls": sum(p["steps"] for p in ps) * LAYERS}
            out["sources"]["tf"] = "docs/gap-tf-static-20261107.json"
    pb = _load("docs/s1-bench-20261107.json")
    if pb and "probe" in pb:
        try:
            out["probe_m1_us"] = {int(c): v["kernel_m1_stage_rows"]["us"] for c, v in pb["probe"]["bandwidth"].items()}
            out["probe_ov_us"] = {int(c): v["kernel-overlap/np48b8k"]["exposed_us"]
                                  for c, v in pb["probe"]["overlap"].items()}
            out["sources"]["probe"] = "docs/s1-bench-20261107.json"
        except Exception:
            pass
    for mode, f in (("kernel-overlap", "docs/headline-s1-shapes-20261107.json"),
                    ("none-diag", "docs/headline-s1-nonediag-20261107.json")):
        h = _load(f)
        if h and "headline" in h:
            out["headline"][mode] = {c: (h["headline"][c]["ms_per_step"]["mean"], h["headline"][c]["cold_share"])
                                     for c in ("two_graphs", "one_graph")}
            out["sources"][mode] = f
    t = {k: float(v) for k, v in out["tf"].items()}
    t["hot_per_call"] = t["hot_unique"] / t["layer_calls"]
    t["cold_per_call"] = t["cold_unique"] / t["layer_calls"]
    t["unique_per_call"] = (t["hot_unique"] + t["cold_unique"]) / t["layer_calls"]
    out["tf"] = t
    return out


# ------------------------------------------------------------------------------------------------ census
class _CountingRegistry(dict):
    """address -> uint8 row; every resolve() is one source row read; `tag[address]` = (tier, kind, proj)."""

    def __init__(self):
        super().__init__()
        self.tag = {}
        self.reads = {}

    def resolve(self, addr):
        k = self.tag[int(addr)]
        self.reads[k] = self.reads.get(k, 0) + 1
        return self[int(addr)]


def _mini_layer(H: int, E: int, seed: int, prepack: bool = False):
    """A synthetic layer at mini shapes (N 8 / K 128): hot pool + packed SF (prepack: the hot groups pre-packed, as
    the pool store holds them), cold rows in a registry, the tables."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nk = {"w1": (8, 128), "w3": (8, 128), "w2": (8, 128)}
    S = 16
    G = H + S
    reg = _CountingRegistry()
    full = {p: torch.zeros(G, n, k // 2, dtype=torch.int8, device="cpu") for p, (n, k) in nk.items()}
    C = {p: DI.sf_words(k // 32) for p, (n, k) in nk.items()}
    sf = {p: DI.packed_sf_zeros(G, nk[p][0], C[p], "cpu") for p in PROJS}
    ptab = torch.zeros(6, E, dtype=torch.int64, device="cpu")
    gtab = torch.full((E,), -1, dtype=torch.int32, device="cpu")
    keep = []
    for e in range(E):
        hot = e < H
        for i, p in enumerate(PROJS):
            n, k = nk[p]
            w = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8, generator=g, device="cpu")
            s = torch.randint(118, 125, (n, k // 32), dtype=torch.uint8, generator=g, device="cpu")
            if hot:
                full[p][e].copy_(w)
                if prepack:
                    sf[p][e].copy_(DI.pack_words(s))
            keep += [w, s]
            for kind, t in (("weight", w), ("scale", s)):
                a = t.data_ptr()
                reg[a] = t.view(torch.uint8).view(-1)
                reg.tag[a] = ("hot" if hot else "cold", kind, p)
            ptab[i, e], ptab[3 + i, e] = w.data_ptr(), s.data_ptr()
        if hot:
            gtab[e] = e
    dsts = [full[p] for p in PROJS] + [sf[p] for p in PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cpu")
    sf_dims = ((nk["w1"][0], C["w1"]), (nk["w2"][0], C["w2"]))
    return {"H": H, "S": S, "G": G, "E": E, "reg": reg, "ptab": ptab, "gtab": gtab, "dsts": dsts, "dtab": dtab,
            "sf_dims": sf_dims, "row_w": nk["w1"][0] * nk["w1"][1] // 2, "keep": keep}


def _route(rng, H, E, n_unique, n_cold):
    """6 rows x top-6 over exactly n_unique distinct experts, n_cold of them cold (ids >= H)."""
    cold = rng.sample(range(H, E), n_cold)
    hot = rng.sample(range(H), n_unique - n_cold)
    ex = sorted(hot + cold)
    for _ in range(100000):
        rr = [sorted(rng.sample(ex, TOPK)) for _ in range(ROWS)]
        if len({e for r in rr for e in r}) == len(ex):
            return rr
    raise RuntimeError("no routing cover")


def census_counted(cases=((22, 0), (22, 1), (22, 2), (22, 4)), seed: int = 20261108) -> dict:
    """[M] rows the runtime's dedup + staging contracts read per layer-call, sf_store step vs pool, on a synthetic
    layer (H 40 hot of 64): {"<unique>/<cold>": {"step": {(tier, kind): rows}, "pool": ...}}."""
    rng = random.Random(seed)
    out = {}
    for U, c in cases:
        L = _mini_layer(40, 64, seed + 7 * U + c)
        idx = torch.tensor(_route(rng, L["H"], L["E"], U, c), dtype=torch.long, device="cpu")
        w = torch.rand(ROWS, TOPK, device="cpu")
        m = MR.dedup_routes(idx, w, 16)
        rec = {}
        for store in ("step", "pool"):
            inpl = {"gtab": L["gtab"], "H": L["H"], "S": L["S"], "dummy": torch.tensor(0), "has_dummy": True,
                    "v2": True, "layout": "contiguous", "G": L["G"], "gscr": L["G"], "step_sf": store == "step"}
            ext = MR.rows_layout(m, inpl, 16, ROWS, TOPK)
            n_work = ROWS * TOPK if store == "step" else L["S"]
            for which in ("m1_stage_rows", "s1_stage_proj_x3"):
                L["reg"].reads = {}
                if which == "m1_stage_rows":
                    DI.stage_rows_emulated(L["reg"].resolve, L["ptab"], m["uniq"], L["gtab"], L["dtab"], L["dsts"],
                                           ext["sidx"], ext["wl"], L["H"], L["row_w"], L["sf_dims"], n_work)
                else:
                    for pj in range(3):
                        SO.stage_proj_emulated(L["reg"].resolve, L["ptab"], m["uniq"], L["gtab"], L["dtab"], L["dsts"],
                                               ext["sidx"], ext["wl"], L["H"], L["row_w"], L["sf_dims"], n_work, pj)
                by = {}
                for (tier, kind, p), n in L["reg"].reads.items():
                    by[f"{tier}.{kind}"] = by.get(f"{tier}.{kind}", 0) + n
                rec.setdefault(store, {})[which] = by
            rec[store]["work_list_entries"] = int((ext["wl"][:n_work] >= 0).sum())
        rec["unique"], rec["cold"], rec["hot"] = U, c, U - c
        out[f"{U}u/{c}c"] = rec
    return out


def contract_check(seed: int = 5, U: int = 22, c: int = 3) -> dict:
    """[M] the probe's D check on the CPU contracts: on a layer whose hot SF groups are pre-packed (the pool store),
    a step-style work list (+ the used hot slots; also truncated to h hot) writes exactly the bytes a pool-style
    list writes -- the hot groups get the same values again (staging_overlap.stage_proj_emulated per projection)."""
    rng = random.Random(seed)
    res = {}
    base = None
    idx = None
    for style in ("pool", "step", "step-h3"):
        L = _mini_layer(40, 64, seed, prepack=True)
        if idx is None:
            idx = torch.tensor(_route(rng, L["H"], L["E"], U, c), dtype=torch.long, device="cpu")
        m = MR.dedup_routes(idx, torch.ones(ROWS, TOPK, device="cpu"), 16)
        inpl = {"gtab": L["gtab"], "H": L["H"], "S": L["S"], "dummy": torch.tensor(0), "has_dummy": True,
                "v2": True, "layout": "contiguous", "G": L["G"], "gscr": L["G"], "step_sf": False}
        ext = MR.rows_layout(m, inpl, 16, ROWS, TOPK)
        if style == "pool":
            wl, nw = ext["wl"], L["S"]
        else:
            ss = step_style(m, inpl, 16, ROWS, TOPK)
            res["n_cold"], res["n_hot"] = ss["n_cold"], ss["n_hot"]
            wl, nw = ss["wl_step"], ROWS * TOPK
            if style == "step-h3":
                wl = truncate_hot(wl, ss["n_cold"], 3)
                res["listed_h3"] = int((wl >= 0).sum())
        for pj in range(3):
            SO.stage_proj_emulated(L["reg"].resolve, L["ptab"], m["uniq"], L["gtab"], L["dtab"], L["dsts"],
                                   ext["sidx"], wl, L["H"], L["row_w"], L["sf_dims"], nw, pj)
        snap = [t.clone() for t in L["dsts"]]
        if base is None:
            base = snap
        else:
            res[f"{style}_equal_pool"] = all(torch.equal(x, y) for x, y in zip(snap, base))
    return res


def bytes_per_call(hot: float, cold: float, store: str) -> dict:
    """[D] bytes one 6-row layer-call's staging moves at real shapes (reads = writes for each term)."""
    a = cold * EXPERT_W
    b = cold * EXPERT_S
    c = hot * EXPERT_S if store == "step" else 0.0
    return {"a_cold_weights_B": a, "b_cold_scales_B": b, "c_hot_scales_B": c,
            "link": {"a": "C2C read (Grace -> HBM write)", "b": "C2C read", "c": "HBM read + HBM write (transpose)"},
            "c2c_read_B": a + b, "hbm_read_B": c, "hbm_write_B": a + b + c,
            "d_other": "work list / tables / m_indices: < 2 KB; the zero-work launch: 2.8 us [M probe c=0]"}


def time_attribution(rc: dict) -> dict:
    """[D] the serial 113.8 us/layer of m1_stage_rows (sf_store step, seed map) split into the cold-only launch the
    M3-S1 probe measured (pool-style work list, same kernel, real shapes) and the rest."""
    pm = rc["probe_m1_us"]
    cs = sorted(c for c in pm if c >= 1)
    slope = (pm[cs[-1]] - pm[cs[0]]) / (cs[-1] - cs[0])
    icpt = pm[cs[0]] - slope * cs[0]
    lam = rc["tf"]["cold_per_call"]
    p0 = math.exp(-lam)                                    # [A] Poisson only fixes the c = 0 weight; t is linear
    cold_only = p0 * pm[0] + (1 - p0) * icpt + slope * lam
    rest = R_ANATOMY_STAGE_US - cold_only
    hot = rc["tf"]["hot_per_call"]
    return {"anatomy_m1_stage_rows_us": R_ANATOMY_STAGE_US, "cold_per_call": round(lam, 3),
            "hot_per_call": round(hot, 2), "probe_slope_us_per_cold_expert": round(slope, 2),
            "probe_GBps_per_cold_expert": round(EXPERT_W / (slope * 1e3), 1) if slope > 0 else None,
            "probe_intercept_us": round(icpt, 2), "probe_zero_cold_us": pm[0],
            "cold_only_at_mean_us": round(cold_only, 1), "rest_us": round(rest, 1),
            "rest_per_hot_expert_us": round(rest / hot, 2) if hot else None,
            "rest_hot_sf_HBM_GBps_rw": round(2 * hot * EXPERT_S / (rest * 1e3), 0) if rest > 0 else None,
            "probe_serial_vs_real_ms": [R_PROBE_SERIAL_MS, R_SERIAL_MS],
            "probe_serial_excess_us_per_layer": round((R_SERIAL_MS - R_PROBE_SERIAL_MS) / LAYERS * 1e3, 1),
            "label": "[D] cold_only = the probe's pool-style launch at the receipts' mean cold count (linear in c "
                     "for c >= 1, measured c = 0); rest = anatomy - cold_only = the step store's hot-scale repack "
                     "(+ any box-vs-probe difference); the probe's own serial (3.404 ms) vs the anatomy's 4.552 is a "
                     "second, independent reading of the same excess"}


def cmd_census(a) -> dict:
    rc = receipts()
    counted = census_counted()
    cc = contract_check()
    tf = rc["tf"]
    hl = rc["headline"]["kernel-overlap"]
    U = tf["unique_per_call"]
    cond = {"tf seed map (anatomy cell, residency off)": (tf["hot_per_call"], tf["cold_per_call"])}
    for cell, (_, share) in hl.items():
        cond[f"headline {cell} (residency on, cold share {share * 100:.2f} %)"] = ((1 - share) * U, share * U)
    rows = {}
    for name, (h, c) in cond.items():
        st, po = bytes_per_call(h, c, "step"), bytes_per_call(h, c, "pool")
        rows[name] = {"hot": round(h, 2), "cold": round(c, 3),
                      "MB": {"a_cold_weights": round(st["a_cold_weights_B"] / 1e6, 2),
                             "b_cold_scales": round(st["b_cold_scales_B"] / 1e6, 2),
                             "c_hot_scales_step": round(st["c_hot_scales_B"] / 1e6, 2),
                             "c_hot_scales_pool": round(po["c_hot_scales_B"] / 1e6, 2)},
                      "dominant_bytes_step": max(("a_cold_weights", st["a_cold_weights_B"]),
                                                 ("c_hot_scales", st["c_hot_scales_B"]), key=lambda x: x[1])[0]}
    ta = time_attribution(rc)
    out = {"label": "[M] rows counted through the runtime's contracts on a synthetic layer; [D] bytes = counted rows "
                    "x real row bytes at the receipts' hot / cold unique counts per 6-row layer-call",
           "real_row_bytes": {"expert_weights": EXPERT_W, "expert_scales": EXPERT_S, "per_proj_weight": W_ROW,
                              "per_proj_scale": S_ROW},
           "counted": counted, "contract_check": cc, "per_condition": rows, "time": ta, "sources": rc["sources"]}
    print("[M] rows read per layer-call (synthetic layer, runtime contracts): step vs pool")
    for k, v in counted.items():
        print(f"  {k}: step {v['step']['m1_stage_rows']}  pool {v['pool']['m1_stage_rows']}  "
              f"(work list {v['step']['work_list_entries']} vs {v['pool']['work_list_entries']})")
    print("[D] MB per layer-call at real shapes (reads; writes equal):")
    for k, v in rows.items():
        print(f"  {k}: hot {v['hot']} cold {v['cold']} -> {v['MB']}  dominant(step bytes) {v['dominant_bytes_step']}")
    print(f"[D] serial m1_stage_rows {ta['anatomy_m1_stage_rows_us']} us = cold-only {ta['cold_only_at_mean_us']} us "
          f"(probe slope {ta['probe_slope_us_per_cold_expert']} us/cold expert = {ta['probe_GBps_per_cold_expert']} "
          f"GB/s) + rest {ta['rest_us']} us ({ta['rest_per_hot_expert_us']} us per hot expert's scales); probe "
          f"serial excess {ta['probe_serial_excess_us_per_layer']} us/layer")
    return out


# ------------------------------------------------------------------------------------------------ exposure
def _poisson(lam, n=16):
    ps = [math.exp(-lam) * lam ** c / math.factorial(c) for c in range(n + 1)]
    ps[-1] += max(0.0, 1 - sum(ps))
    return ps


def cmd_exposure(a) -> dict:
    """[D] exposed staging = kernel-overlap - none-diag per cell; vs the probe's pool-style exposure at the cell's
    cold count (Poisson [A]); the residual is the c-independent part."""
    rc = receipts()
    U = rc["tf"]["unique_per_call"]
    ov = rc["probe_ov_us"]
    cs = sorted(ov)
    sl = ov[cs[-1]] - ov[cs[-2]]

    def ov_us(c):
        return ov[c] if c in ov else ov[cs[-1]] + sl * (c - cs[-1])

    cells = {}
    for cell in ("two_graphs", "one_graph"):
        ms, share = rc["headline"]["kernel-overlap"][cell]
        nd, _ = rc["headline"]["none-diag"][cell]
        c = share * U
        exp_ms = ms - nd
        pool_like = LAYERS * sum(p * ov_us(k) for k, p in enumerate(_poisson(c))) / 1e3
        cells[cell] = {"step_ms": ms, "none_diag_ms": nd, "exposed_ms": round(exp_ms, 3), "cold_share": share,
                       "cold_per_call": round(c, 3), "probe_pool_style_exposed_ms_E": round(pool_like, 3),
                       "residual_ms": round(exp_ms - pool_like, 3),
                       "residual_us_per_layer": round((exp_ms - pool_like) / LAYERS * 1e3, 1)}
    c2, c1 = cells["two_graphs"], cells["one_graph"]
    slope = (c2["exposed_ms"] - c1["exposed_ms"]) / (c2["cold_per_call"] - c1["cold_per_call"])
    icpt = c1["exposed_ms"] - slope * c1["cold_per_call"]
    out = {"cells": cells, "unique_per_call_A": round(U, 2),
           "linear_fit": {"ms_per_cold_per_call": round(slope, 3),
                          "us_per_cold_expert_per_layer": round(slope / LAYERS * 1e3, 1),
                          "zero_cold_ms": round(icpt, 3), "zero_cold_us_per_layer": round(icpt / LAYERS * 1e3, 1)},
           "probe_zero_cold_us_per_layer_pool_style": ov[0],
           "label": "[D] exposed = kernel-overlap - none-diag (same cell, same run config); probe_pool_style = the "
                    "M3-S1 probe's kernel-overlap/np48b8k exposure (pool-style work list: NO hot scales) at Poisson("
                    "cold per call) [A]; residual = the part the probe has no counterpart for: the step store's hot-"
                    "scale repack on the side stream + its contention (the GEMM chain), + the real graph's extra "
                    "joins. The 2-point linear fit reads the same split without the probe."}
    print("[D] exposed staging per headline cell (kernel-overlap - none-diag):")
    for k, v in cells.items():
        print(f"  {k}: {v['exposed_ms']} ms at {v['cold_per_call']} cold/call; probe pool-style {v['probe_pool_style_exposed_ms_E']}"
              f" -> residual {v['residual_ms']} ms = {v['residual_us_per_layer']} us/layer")
    lf = out["linear_fit"]
    print(f"[D] 2-point fit: {lf['us_per_cold_expert_per_layer']} us per cold expert per layer-call, "
          f"{lf['zero_cold_us_per_layer']} us/layer at zero cold ({lf['zero_cold_ms']} ms/step)")
    return out


# ------------------------------------------------------------------------------------------------ pool-fit
def pool_bytes(H=HOT_SEED, spare=CUR_POOL_SPARE, layers=LAYERS) -> dict:
    per_layer = (H + spare) * EXPERT_S
    return {"groups_per_layer": H + spare, "per_group_B": EXPERT_S, "per_layer_B": per_layer,
            "total_B": layers * per_layer, "total_GiB": round(layers * per_layer / GiB, 3),
            "hot_part_GiB": round(layers * H * EXPERT_S / GiB, 3),
            "spare_part_GiB": round(layers * spare * EXPERT_S / GiB, 3),
            "step_shared_B": per_layer, "step_shared_GiB": round(per_layer / GiB, 3),
            "derivation": f"{layers} layers x ({H} hot + {spare} spare) groups x (2304x40 + 2304x40 + 5120x18) "
                          f"int32 words x 4 B = {layers} x {H + spare} x {EXPERT_S:,} B"}


def free_at_check(pool_spare: int, stage: int, free0: float = R_FREE_AT_CHECK_GIB) -> float:
    """[D] free HBM at the SF check for another --pool-spare (residency slots = pool_spare - stage): every slot
    dropped frees its weights in 40 layers; a residency slot also its e8m0 scale buffer (ResidencyManager.sspare)."""
    d_slots = CUR_POOL_SPARE - pool_spare
    d_res = CUR_RES - (pool_spare - stage)
    return free0 + (d_slots * LAYERS * EXPERT_W + d_res * LAYERS * EXPERT_S) / GiB


def cmd_pool_fit(a) -> dict:
    pb = pool_bytes()
    exp_ = cmd_exposure(a) if not getattr(a, "_exposure", None) else a._exposure
    saved_lo = time_attribution(receipts())["rest_us"]            # serial-derived hot-SF time per layer [D]
    saved_hi = exp_["cells"]["two_graphs"]["residual_us_per_layer"]  # overlap residual per layer [D]
    arms = []
    for pool_spare, stage in ((24, 16), (20, 16), (17, 16), (16, 8)):
        res = pool_spare - stage
        free = free_at_check(pool_spare, stage)
        G = HOT_SEED + pool_spare
        layer_B = [G * EXPERT_S] * LAYERS
        for head in (6.0, 2.5, 2.0, 1.5):
            for spec in ("all", "auto"):
                p = DI.sf_pool_plan(spec, LAYERS, layer_B, G * EXPERT_S, int(free * GiB), int(head * GiB))
                k = len(p["layers"])
                post = R_FREE_AT_CHECK_GIB - R_FREE_AFTER_BUILD_GIB - pb["step_shared_GiB"]   # besides the step SF
                after = free - p["need_B"] / GiB - post - (G * EXPERT_S / GiB if p["mode"] == "step" else 0.0)
                arms.append({"pool_spare": pool_spare, "stage_slots": stage, "residency_slots": res,
                             "headroom_GiB": head, "spec": spec, "free_at_check_GiB": round(free, 2),
                             "mode": p["mode"], "layers_pooled": k, "need_GiB": round(p["need_B"] / GiB, 2),
                             "free_after_build_GiB_E": round(after, 2),
                             "saved_ms_E": [round(k * saved_lo / 1e3, 2), round(k * saved_hi / 1e3, 2)]})
    out = {"pool_sf": pb, "free_at_check_GiB_M": R_FREE_AT_CHECK_GIB, "free_after_build_GiB_M": R_FREE_AFTER_BUILD_GIB,
           "post_check_consumption_GiB_D": round(R_FREE_AT_CHECK_GIB - R_FREE_AFTER_BUILD_GIB, 2),
           "saved_us_per_pooled_layer_D": [saved_lo, saved_hi], "arms": arms,
           "label": "[D] pool SF from the real shapes; free at check from the headline logs; arms decided by "
                    "runtime/dg_inplace.sf_pool_plan (the build's own rule); free_after_build = free at check - "
                    "the plan's need - the 0.97 GiB the receipt's build consumed after the check besides the step "
                    "SF (1.30 - 0.33); saved = layers pooled x [serial-derived hot-SF us, overlap residual us] [E]"}
    print(f"[D] pool SF = {pb['derivation']} = {pb['total_GiB']} GiB (hot part {pb['hot_part_GiB']}, spare part "
          f"{pb['spare_part_GiB']}); the step store's shared copy {pb['step_shared_GiB']} GiB")
    print(f"[D] free at check {R_FREE_AT_CHECK_GIB} GiB, after build {R_FREE_AFTER_BUILD_GIB} -> the build "
          f"consumed {out['post_check_consumption_GiB_D']} GiB after the check (incl. the 0.33 GiB step SF)")
    print("  pool-spare stage res headroom spec  free  mode     layers need  after  saved_ms[E]")
    for r in arms:
        if r["spec"] == "all" and r["headroom_GiB"] != 6.0:
            continue
        print(f"  {r['pool_spare']:10d} {r['stage_slots']:5d} {r['residency_slots']:3d} {r['headroom_GiB']:8.1f} "
              f"{r['spec']:5s} {r['free_at_check_GiB']:5.2f} {r['mode']:8s} {r['layers_pooled']:6d} "
              f"{r['need_GiB']:5.2f} {r['free_after_build_GiB_E']:6.2f}  {r['saved_ms_E']}")
    return out


# ------------------------------------------------------------------------------------------------ hist
def cmd_hist(a) -> dict:
    with open(a.hist) as f:
        h = json.load(f)
    allh = h["hist_all"]
    tot = max(1, sum(allh))
    out = {"layer_calls": h["layer_calls"], "cold_mean": h["cold_mean"], "cold_max": h["cold_max"],
           "hot_mean": h["hot_mean"], "p_cold_gt": h["p_cold_gt"],
           "overflow_budget": {S: sum(allh[S + 1:]) for S in (4, 8, 12, 16) if S + 1 <= len(allh)},
           "cold_removed_by_top_n_hot": h["cold_removed_per_layer_call_by_top_n_hot"],
           "dist": [round(v / tot, 5) for v in allh]}
    out["ms_saved_by_top_n_hot_E"] = {n: round(v * LAYERS * a.us_per_cold / 1e3, 3)
                                      for n, v in out["cold_removed_by_top_n_hot"].items()}
    out["hbm_per_n_hot_GiB"] = {n: round(int(n) * LAYERS * (EXPERT_W + EXPERT_S) / GiB, 2)
                                for n in out["cold_removed_by_top_n_hot"]}
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ probe (GPU)
RUNTIME_CFG = ((48, 8192), (16, 32768), (16, 32768))     # $DSV41_STAGE_OVERLAP_NPROG / _BLOCK_BYTES defaults


def step_style(m, inpl, block_rows, R_, k) -> dict:
    """The sf_store "step" work list of a dedup `m` whose pool-style extras were built with `inpl` (step_sf False):
    {"wl_step": staged cold slots, then the used HOT slots, -1 after; "n_cold", "n_hot"} (moe_rows.rows_layout)."""
    ext = MR.rows_layout(m, dict(inpl, step_sf=True), block_rows, R_, k)
    return {"wl_step": ext["wl"], "n_hot": int(((m["counts"] > 0) & (ext["sidx"] < 0)).sum()),
            "n_cold": int((ext["sidx"] >= 0).sum())}


def truncate_hot(wl_step: torch.Tensor, n_cold: int, h: int) -> torch.Tensor:
    """wl_step keeping the staged cold entries and the first h hot ones (the hot sweep's lists)."""
    w = wl_step.clone()
    w[int(n_cold) + int(h):] = -1
    return w


def cmd_probe(a) -> dict:
    import staging_overlap_bench as B

    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": a.argv, "errors": [], "label": {
        "A": "[M] serial launches, synthetic real shapes, graph replay, distinct cold experts per round",
        "B": "[M] kernel-overlap vs the real DeepGEMM contiguous16 chain + a bf16 shared-expert STAND-IN",
        "C": "[M] c = 0, step-style work list truncated to h hot slots",
        "D": "[M] step-style vs pool-style: slot / SF / GEMM-output bytes"}}

    def step(name, fn):
        try:
            r = fn()
            out[name] = r
            print("OK  ", name, json.dumps(r, default=str)[:600], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            out[name] = {"error": tb[-2000:]}
            out["errors"].append(name)
            print("FAIL", name, tb[-2000:], flush=True)
            return None

    if not torch.cuda.is_available():
        out["errors"].append("no CUDA")
        return out
    dev = torch.device("cuda")
    rec = []
    orig = MR.rows_layout

    def spy(m, inpl, block_rows, R_, k):              # record each round's dedup to build its step-style list
        rec.append((m, dict(inpl), block_rows, R_, k))
        return orig(m, inpl, block_rows, R_, k)

    MR.rows_layout = spy
    try:
        sa = argparse.Namespace(seed=a.seed, hot=a.hot, rounds=a.rounds, host=a.host)
        st = B._Setup(sa, dev)
    finally:
        MR.rows_layout = orig
    it = iter(rec)
    P = ROWS * TOPK
    for c in range(B.CMAX + 1):
        for rd in st.rounds[c]:
            m, inpl, br, R_, k = next(it)
            rd.update(step_style(m, inpl, br, R_, k))
    out["setup"] = {"H": st.H, "S": st.S, "rounds": st.R, "unique": B.UNIQUE, "host": st.host_kind,
                    "hot_listed_c0": st.rounds[0][0]["n_hot"], "runtime_cfg": RUNTIME_CFG}
    reps, R = a.reps, st.R
    side = torch.cuda.Stream()
    ev_r = [torch.cuda.Event() for _ in range(R)]
    ev_p = [[torch.cuda.Event() for _ in range(3)] for _ in range(R)]

    def wl_of(rd, style, h=None):
        if style == "pool":
            return rd["wl"], st.S
        if h is not None:                              # the staged cold + the first h hot entries (built
            return rd["wl_h"][int(h)], P               # before any capture: hot_sweep)
        return rd["wl_step"], P

    def m1(rd, style):
        from runtime import static_kernels as SK

        wl, nw = wl_of(rd, style)
        SK.stage_rows(st.ptab, rd["uniq"], st.gtab, st.dtab, st.dsts, rd["sidx"], wl, st.H, st.row_w, st.sf_dims, nw)

    def proj(rd, pj, style, h=None, weights=True):
        wl, nw = wl_of(rd, style, h)
        nprog, bb = RUNTIME_CFG[pj]
        SO.stage_proj(st.ptab, rd["uniq"], st.gtab, st.dtab, st.dsts, rd["sidx"], wl, st.H, st.row_w, st.sf_dims,
                      nw, pj, weights=weights, nprog=nprog, block_bytes=bb)

    def per_round_us(fn):
        def body():
            for r in range(R):
                fn(r)
        return B._graph_ms(body, reps) * 1e3 / R

    def serial():
        res = {}
        for c in range(3):
            row = {}
            for style in ("pool", "step"):
                row[f"m1_stage_rows/{style}"] = per_round_us(lambda r, c=c, s=style: m1(st.rounds[c][r], s))
                row[f"s1_proj_x3/{style}"] = per_round_us(
                    lambda r, c=c, s=style: [proj(st.rounds[c][r], pj, s) for pj in range(3)])
            row["hot_scale_pack_x3/step"] = per_round_us(
                lambda r, c=c: [proj(st.rounds[c][r], pj, "step", weights=False) for pj in range(3)])
            row["hot_sf_term_m1_us"] = row["m1_stage_rows/step"] - row["m1_stage_rows/pool"]
            row["hot_sf_term_s1_us"] = row["s1_proj_x3/step"] - row["s1_proj_x3/pool"]
            res[c] = {k_: round(v, 2) for k_, v in row.items()}
            print(f"  [A] c={c}: " + ", ".join(f"{k_} {v:.1f}" for k_, v in res[c].items()), flush=True)
        return res

    step("serial", serial)

    from runtime.dg_moe import import_deep_gemm

    dg = import_deep_gemm()
    assert not any(m_ == "vllm" or m_.startswith("vllm.") for m_ in sys.modules)
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()
    prev_align = dg.get_mk_alignment_for_contiguous_layout()
    dg.set_mk_alignment_for_contiguous_layout(B.ALIGN)

    def layer(c, r, style, h=None):
        rd = st.rounds[c][r]
        cur = torch.cuda.current_stream()
        if style is not None:
            ev_r[r].record(cur)
            side.wait_event(ev_r[r])
            with torch.cuda.stream(side):
                for pj in range(3):
                    proj(rd, pj, style, h)
                    ev_p[r][pj].record(side)
        st.shared()
        for pj, p in enumerate(PROJS):
            if style is not None:
                cur.wait_event(ev_p[r][pj])
            st.gemm(dg, rd, p)

    def overlap():
        res = {}
        for c in range(3):
            row = {"gemm-only": per_round_us(lambda r, c=c: layer(c, r, None))}
            for style in ("pool", "step"):
                row[f"kernel-overlap/{style}"] = per_round_us(lambda r, c=c, s=style: layer(c, r, s))
            base = row["gemm-only"]
            res[c] = {k_: {"us_per_layer": round(x, 2), "exposed_us": round(x - base, 2)} for k_, x in row.items()}
            res[c]["pooled_layer_saves_us"] = round(row["kernel-overlap/step"] - row["kernel-overlap/pool"], 2)
            print(f"  [B] c={c}: " + ", ".join(f"{k_} {x:.1f} (+{x - base:.1f})" for k_, x in row.items()), flush=True)
        return res

    step("overlap", overlap)

    def hot_sweep():
        n_hot = min(rd["n_hot"] for rd in st.rounds[0])
        hs = sorted({0, n_hot // 4, n_hot // 2, (3 * n_hot) // 4, n_hot})
        for rd in st.rounds[0]:
            rd["wl_h"] = {}
            for h in hs:
                rd["wl_h"][h] = truncate_hot(rd["wl_step"], rd["n_cold"], h)
        torch.cuda.synchronize()
        base = per_round_us(lambda r: layer(0, r, None))
        res = {"gemm_only_us": round(base, 2), "n_hot": n_hot, "exposed_us_by_h": {}}
        for h in hs:
            x = per_round_us(lambda r, h=h: layer(0, r, "step", h))
            res["exposed_us_by_h"][h] = round(x - base, 2)
        print(f"  [C] c=0 exposed by hot slots listed: {res['exposed_us_by_h']}", flush=True)
        return res

    step("hot_sweep", hot_sweep)

    def checks():
        res = {}
        c = 2
        snaps = {}
        for style in ("pool", "step"):
            for p in PROJS:
                st.full[p][st.H:].zero_()
                st.sf[p][st.H:].zero_()
            for x in st.d.values():
                x.zero_()
            layer(c, 0, style)
            torch.cuda.synchronize()
            snaps[style] = {"slots": [st.full[p][st.H:].clone() for p in PROJS],
                            "sf_all": [st.sf[p].clone() for p in PROJS],
                            "d": [st.d[p].index_select(0, st.rounds[c][0]["arow"]).clone() for p in PROJS]}
        for k_ in ("slots", "sf_all", "d"):
            res[k_] = all(torch.equal(x, y) for x, y in zip(snaps["pool"][k_], snaps["step"][k_]))
        return res

    step("checks", checks)
    dg.set_mk_alignment_for_contiguous_layout(prev_align)
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("cmd", choices=("census", "pool-fit", "exposure", "hist", "probe", "all"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--hist", default=None, help="hist: a $DSV41_STAGE_HIST json")
    ap.add_argument("--us-per-cold", type=float, default=20.3,
                    help="hist: exposed us per cold expert per layer-call under kernel-overlap [D exposure fit]")
    ap.add_argument("--hot", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--host", choices=("hostbuffer", "pinned"), default="hostbuffer")
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    out = {}
    if a.cmd in ("census", "all"):
        out["census"] = cmd_census(a)
    if a.cmd in ("exposure", "all", "pool-fit"):
        out["exposure"] = cmd_exposure(a)
        a._exposure = out["exposure"]
    if a.cmd in ("pool-fit", "all"):
        out["pool_fit"] = cmd_pool_fit(a)
    if a.cmd == "hist":
        if not a.hist:
            ap.error("hist needs --hist <json>")
        out["hist"] = cmd_hist(a)
    if a.cmd == "probe" or (a.cmd == "all" and torch.cuda.is_available()):
        out["probe"] = cmd_probe(a)
    path = a.out or os.path.join(REPO, "scratch", f"staging-bytes-{a.cmd}-{_date()}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)
    errs = out.get("probe", {}).get("errors", []) if isinstance(out.get("probe"), dict) else []
    return 2 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
