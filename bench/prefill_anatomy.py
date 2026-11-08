#!/usr/bin/env python3
"""prefill_anatomy.py -- M4-P1: where the ~4 s of reference prefill go, and what a fast prefill would cost.

Handoff docs/HANDOFF-M4-P1-PREFILL-ANATOMY.md; design + numbers docs/M4-PREFILL-DESIGN.md. Our TTFT is the reference
eager forward: runtime/dspark_loop.py `start()` -> runtime/draft_step.py `prefill_hidden()` -> `model(tokens, 0)`
(reference model.py:1245-1275) on the tiered loader's experts (hot in HBM, cold in pinned Grace read over UVA,
runtime/expert_pool.py `bind`), then `seed_draft_rings` (forward_spec at start_pos 0) and the first draft (a graph).

Subcommands (labels: [M] measured by this tool, [D] derived from a receipt, [E] modeled, [A] assumption):

  run      BOX ONLY (GPU container). Boots the model like bench/headline.py (map, load_mtp, set_determinism(0): the
           headline's fills-on state), encodes fixture prompts, warms once (TileLang JIT), then per prompt:
             * --reps untraced prefills: prefill_hidden ms, seed_draft_rings ms, first token, logits/main_hidden sha;
             * --fills both|off: the same with torch's fill_uninitialized_memory off, + bit-equality vs fills on;
             * one HOOKED prefill (PrefillProbe): forward hooks on every layer / engram / attn / compressor / indexer /
               moe / gate / shared expert / routed Expert / head -> host ms + GPU-timeline ms (CUDA events) per role,
               and the expert CENSUS (calls, rows, hot vs cold per layer = distinct routed experts; cold bytes over
               C2C; under DSV41_POOL_SF=beside also the hot experts' K-major scales read from Grace);
               --profile wraps exactly this prefill in cudaProfilerStart/Stop with NVTX ranges (prefill, seed_draft,
               L{i}, and the roles above, innermost wins) for `nsys --capture-range=cudaProfilerApi`;
             * --probe-moe L,L,..: the MoE inputs of those layers are captured during the hooked prefill and the
               reference MoE.forward is compared (torch.equal, synced wall ms) with `sliced_moe_forward`: the same
               reference Expert calls with ONE host sync per layer instead of one per expert (no where()/nonzero),
               combine either in the reference spelling (`y[idx] += ...`) or `index_add_`. Bench-side only: the
               runtime is not changed. This is the "stage B" measurement of the design;
             * --rows-probe 42,64: runtime/row_census.py `census(model, rows=R)` -- is row r of every kernel family at
               R rows bit-equal to the same row alone? (the decode path's 1-vs-6 row invariance, extended to R).
           Receipt: --json (put it under scratch/ on the box).
  analyze  <profile.sqlite> [--out x.json]: stdlib sqlite over an nsys export of `run --profile`: per phase (prefill,
           seed_draft) wall / GPU busy (union) / sum of kernels / idle, kernels per family (bench/step_anatomy.py
           `prefill_family_of`, attributed to the innermost NVTX role range of the kernel's LAUNCH), per role host
           wall, launches and GPU ms, per routed-expert call (hot vs cold: wall, kernels, GPU ms, fp4_gemm us),
           expert ranges per layer (the census from nsys alone), runtime-API census (launch / sync / memcpy calls and
           the host time blocked in syncs), memcpy bytes by kind. Kernel-side UVA reads of Grace are not memcpys: they
           show only as the cold fp4_gemm's longer duration (cold - hot per launch).
  model    Mac / CPU: the modeled anatomy and the alternatives' TTFT for given prompt lengths, from receipts
           (docs/step-anatomy-m0-20261106.json and its .md, docs/c2c-probe-gb300-20261106.json, the headline) and the
           decode routing counts (expert calls per layer under an independence model). Prints every parameter with
           its label. CPU timing is never used for anything.
  box      print the box command lines (run, nsys run + export + analyze).
"""
from __future__ import annotations

import argparse
import bisect
import collections
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step_anatomy import COPY_KIND, PREFILL_FAMILIES, PREFILL_ROLES, prefill_family_of  # noqa: E402

DEFAULT_MAP = os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-301.json")
DEFAULT_FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
DEFAULT_COUNTS = os.path.expanduser(
    "~/hermes/research/dsv41-prefetch-20261023/ar/results/run1/counts-post-20261024-020438.npz")
REF_CONFIG = os.path.join(REPO, "reference", "deepseek-inference", "config.json")
HEADLINE_RECEIPT = os.path.join(REPO, "docs", "headline-20261108-main-ss.json")
M0_ANATOMY = os.path.join(REPO, "docs", "step-anatomy-m0-20261106.json")
C2C_RECEIPT = os.path.join(REPO, "docs", "c2c-probe-gb300-20261106.json")
FIXTURE_LENGTHS = (31, 34, 38, 42, 46, 47, 49, 54)   # the 8 fixture prompts' token counts (headline receipt)


# ================================================================================================ the model (CPU)
def _jload(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def model_params(m0_path: str = M0_ANATOMY, c2c_path: str = C2C_RECEIPT, headline_path: str = HEADLINE_RECEIPT,
                 config_path: str = REF_CONFIG) -> Dict[str, dict]:
    """Every input of the TTFT model: {name: {"v": value, "label": D|A|E, "src": where it comes from}}. Receipt
    values are read from the JSON receipts when present (the hard-coded fallbacks are the same receipts' numbers)."""
    P: Dict[str, dict] = {}

    def put(name, v, label, src):
        P[name] = {"v": v, "label": label, "src": src}

    cfg = _jload(config_path) or {}
    dim, inter = int(cfg.get("dim", 5120)), int(cfg.get("moe_inter_dim", 2304))
    put("n_layers", int(cfg.get("n_layers", 40)), "D", "reference config.json")
    put("n_experts", int(cfg.get("n_routed_experts", 384)), "D", "reference config.json")
    put("topk", int(cfg.get("n_activated_experts", 6)), "D", "reference config.json")
    put("index_n_heads", int(cfg.get("index_n_heads", 32)), "D", "reference config.json")
    eb = 3 * inter * dim // 2 + 3 * inter * dim // 32          # w1, w3, w2: fp4 [N, K/2] + e8m0 [N, K/32]
    put("expert_bytes", eb, "D", "dims: 3 x (fp4 N*K/2 + e8m0 N*K/32), runtime/expert_pool.py expert_bytes")
    put("sf_bytes", 3 * inter * dim // 32, "D", "one expert's e8m0 scales (beside: hot ones live on Grace)")
    put("moe_flop_per_pair", 3 * 2 * dim * inter, "D", "w1 + w3 + w2 GEMM flops per (token, expert)")
    nh, hd = int(cfg.get("n_heads", 64)), int(cfg.get("head_dim", 512))
    ql, og, ol = int(cfg.get("q_lora_rank", 1280)), int(cfg.get("o_groups", 8)), int(cfg.get("o_lora_rank", 1024))
    # wq_a, wq_b, wkv, wo_a (block-diagonal: groups x o_lora x heads*head_dim/groups), wo_b, shared expert, gate
    dense = (dim * ql + ql * nh * hd + dim * hd + nh * hd * ol + og * ol * dim + 3 * dim * inter
             + int(cfg.get("n_routed_experts", 384)) * dim)
    put("dense_flop_per_token", 2 * dense * int(cfg.get("n_layers", 40)), "D",
        "2 x (wq_a + wq_b + wkv + wo_a (block-diag) + wo_b + shared expert + gate) x layers; head on 1 row only")

    m0 = _jload(m0_path) or {}
    rl, rm = m0.get("role_launches_per_step") or {}, m0.get("role_ms_per_step") or {}
    fp4 = next((t for t in (m0.get("top_kernels") or []) if "fp4_gemm" in t[0]), None)
    put("m0_launches_per_step", 20550, "D", "STEP-ANATOMY-M0-20261106.md: 1-row eager step, 20,550 launches")
    put("m0_gpu_busy_ms", 108.6, "D", "STEP-ANATOMY-M0-20261106.md: GPU busy 108.6 ms/step")
    put("m0_eager_ms_fills", 404.9, "D", "STEP-ANATOMY-M0-20261106.md: untraced eager step, NaN-fill on")
    put("m0_eager_ms_nofill", 357.1, "D", "STEP-ANATOMY-M0-20261106.md: untraced eager step, --no-fill-uninit")
    put("m0_moe_launches", float(rl.get("moe", 14480.0)), "D", "step-anatomy-m0 json role_launches_per_step.moe")
    put("m0_moe_gpu_ms", float(rm.get("moe", 84.40)), "D", "step-anatomy-m0 json role_ms_per_step.moe")
    put("m0_expert_calls", 240, "D", "1 row x 6 experts x 40 layers (fp4_gemm 720 launches = 3 per call)")
    put("fp4_us", float(1000 * fp4[1] / fp4[2]) if fp4 else 78.1, "D", "step-anatomy-m0 json: fp4_gemm us/launch, M<=8")
    put("moe_fixed_launches_per_layer", 40, "A",
        "gate (~13) + shared expert (~13) + zeros/bincount/tolist/y+=shared/type_as (~7) + their fills, "
        "counted from model.py:811-829, 843-853, 891-906")
    put("moe_fixed_gpu_us_per_layer", 185.0, "A", "shared expert 3 x fp8_gemm ~55 us + gate ~20 us")
    put("tail_us", float((1000 * fp4[1] / fp4[2]) if fp4 else 78.1) + 10.0, "A",
        "GPU work left after an expert's last launch (its w2 fp4_gemm + glue), waited for at the next where() sync")
    put("where_share", 4.0 / 26.0, "A", "python-dispatched ops per expert call removed by one sync per layer "
                                        "(eq + nonzero + their compaction) / ops per call (~26 incl. y[idx] +=)")
    nonmoe = 20550 - float(rl.get("moe", 14480.0))
    put("nonmoe_launches_per_layer", nonmoe / 40.0, "D", "(20,550 - moe launches) / 40, M0 1-row eager")
    put("nonmoe_gpu_ms", 108.6 - float(rm.get("moe", 84.40)), "D", "M0: GPU busy - moe GPU")
    put("seed_ms", 0.5, "E", "forward_spec at start_pos 0: main_proj + 3 x (wkv, norm, RoPE, QDQ, ring write)")
    hl = _jload(headline_path) or {}
    loop = (((hl.get("prompts") or {}).get("prose") or {}).get("window") or {})
    put("draft_ms", float((loop.get("ms_draft_graph") or {}).get("mean", 2.91)), "D", "headline: draft graph ms")
    put("verify6_ms", float((loop.get("ms_verify_graph") or {}).get("mean", 15.09)), "D", "headline: verify graph ms")
    put("r1_step_ms", 9.76, "D", "LEDGER 2026-11-08 02:37: 1-row graph step 9.76 ms")
    put("hbm_free_gib", float((hl.get("hbm_after_build_GiB") or {}).get("free", 2.72)), "D",
        "headline: free HBM after the decode build")
    c2c = _jload(c2c_path) or {}
    w1 = ((c2c.get("fp4_gemm_w1_real") or {}).get("M1") or {})
    put("c2c_extra_us", float(w1.get("uva_us", 102.8)) - float(w1.get("hbm_us", 91.3)), "D",
        "c2c probe: reference fp4_gemm w1 at M=1, UVA - HBM")
    put("h2d_GBps", float(c2c.get("h2d_copy_1GiB_GBps", 390.3)), "D", "c2c probe: copy engine H2D")
    put("uva_GBps", float(c2c.get("kernel_read_host_1GiB_GBps", 230.2)), "D", "c2c probe: kernel UVA read")
    put("n_sm", 148, "A", "concurrent CTAs for the reference sparse_attn (1 CTA per query row)")
    put("sparse_attn_us", 74.8, "D", "step-anatomy-m0: reference sparse_attn us per 1-row call")
    put("flashmla_speedup", 125.0 / 14.0, "D", "flashmla probe: reference 125-154 us vs FlashMLA 13.8-14.4 us")
    put("tl_pflops", 0.4, "A", "TileLang reference fp8_gemm / fp4_gemm sustained PFLOP/s at large M (no TMA, no WS)")
    put("tl_grouped_TBps", 2.0, "A", "a grouped reference-kernel-text fp4_gemm at full occupancy, HBM weight stream")
    put("dg_pflops", 2.0, "A", "DeepGEMM fp8 x fp4 / fp8 GEMMs at large M")
    put("hbm_TBps", 6.0, "A", "HBM weight stream of a TMA kernel")
    put("launch_floor_ms", 5.0, "A", "40 layers x ~25 kernels x ~5 us when GPU-bound")
    return P


def _v(P, k):
    return P[k]["v"]


def derived(P: Dict[str, dict]) -> Dict[str, float]:
    """The per-call / per-layer costs the model uses, from model_params."""
    host_launch_us = 1000.0 * (_v(P, "m0_eager_ms_fills") - _v(P, "m0_gpu_busy_ms")) / _v(P, "m0_launches_per_step")
    nofill = (_v(P, "m0_eager_ms_nofill") - _v(P, "m0_gpu_busy_ms")) / (_v(P, "m0_eager_ms_fills")
                                                                         - _v(P, "m0_gpu_busy_ms"))
    nl = _v(P, "n_layers")
    lpc = (_v(P, "m0_moe_launches") - nl * _v(P, "moe_fixed_launches_per_layer")) / _v(P, "m0_expert_calls")
    gpu_call = (1000.0 * _v(P, "m0_moe_gpu_ms") - nl * _v(P, "moe_fixed_gpu_us_per_layer")) / _v(P, "m0_expert_calls")
    host_call = lpc * host_launch_us
    return {"host_us_per_launch": host_launch_us, "nofill_host_factor": nofill, "launches_per_expert_call": lpc,
            "host_us_per_expert_call": host_call, "gpu_us_per_expert_call": gpu_call,
            "wall_us_per_expert_call": max(host_call + _v(P, "tail_us"), gpu_call),
            "nonmoe_host_ms_per_layer": _v(P, "nonmoe_launches_per_layer") * host_launch_us / 1000.0,
            "nonmoe_gpu_ms_per_layer": _v(P, "nonmoe_gpu_ms") / nl}


def load_counts(path: Optional[str], n_layers: int = 40, n_experts: int = 384):
    """Decode routing counts [n_layers, n_experts] from the hot-map npz (key `cnt`, else the one int array of that
    shape) or None."""
    if not path or not os.path.exists(path):
        return None
    import numpy as np

    z = np.load(path)
    if "cnt" in z.files:
        return np.asarray(z["cnt"], dtype=np.float64)
    c = [k for k in z.files if z[k].shape == (n_layers, n_experts) and z[k].dtype.kind in "iu"]
    return np.asarray(z[c[0]], dtype=np.float64) if len(c) == 1 else None


def load_hot_map(path: Optional[str]) -> Optional[Dict[int, set]]:
    m = _jload(path) if path else None
    return {int(k): set(int(e) for e in v) for k, v in m.items()} if m else None


def expected_touched(L: int, k: int, probs: Sequence[float]) -> List[float]:
    """Independence model [E]: expert e is among a token's k picks with q_e = 1 - (1 - p_e)^k, and is touched by an
    L-token prefill with 1 - (1 - q_e)^L. Upper bound in practice (a prompt's tokens route alike). -> per expert."""
    out = []
    for p in probs:
        q = 1.0 - (1.0 - min(max(p, 0.0), 1.0)) ** k
        out.append(1.0 - (1.0 - q) ** L)
    return out


def expert_census_model(L: int, P: Dict[str, dict], counts=None, hot: Optional[Dict[int, set]] = None) -> dict:
    """Modeled expert calls of an L-token reference prefill: per layer the expected number of DISTINCT routed experts
    (each is called once per MoE.forward, model.py:897-902), split hot/cold by the hot map. counts None -> uniform."""
    nl, E, k = _v(P, "n_layers"), _v(P, "n_experts"), _v(P, "topk")
    per_layer, cold_l = [], []
    for l in range(nl):
        if counts is not None:
            row = [float(x) for x in counts[l]]
            s = sum(row) or 1.0
            probs = [x / s for x in row]
        else:
            probs = [1.0 / E] * E
        t = expected_touched(L, k, probs)
        per_layer.append(sum(t))
        cold_l.append(sum(t[e] for e in range(E) if hot is not None and e not in hot.get(l, set())))
    calls, cold = sum(per_layer), sum(cold_l)
    return {"L": L, "model": "independence, " + ("decode counts" if counts is not None else "uniform"),
            "calls": calls, "cold_calls": cold, "calls_per_layer_mean": calls / nl,
            "calls_per_layer_max": max(per_layer), "cold_per_layer_mean": cold / nl, "cold_per_layer_max": max(cold_l),
            "bound_pairs": L * k * nl, "cold_bytes": cold * _v(P, "expert_bytes")}


def ttft_model(L: int, cen: dict, P: Dict[str, dict]) -> dict:
    """Modeled TTFT [E] (ms) of today's prefill and of the design alternatives (docs/M4-PREFILL-DESIGN.md section 3).
    Each entry: total and its terms. `cen` = expert_census_model(L) (or a measured census with the same keys)."""
    d = derived(P)
    nl, k = _v(P, "n_layers"), _v(P, "topk")
    calls, cold = cen["calls"], cen["cold_calls"]
    eb = _v(P, "expert_bytes")
    tail = _v(P, "seed_ms") + _v(P, "draft_ms")
    moe_fixed_host = nl * _v(P, "moe_fixed_launches_per_layer") * d["host_us_per_launch"] / 1000.0
    nonmoe = nl * max(d["nonmoe_host_ms_per_layer"], d["nonmoe_gpu_ms_per_layer"])
    # large-L GPU terms of the reference kernels (compute-bound)
    moe_flop = L * k * nl * _v(P, "moe_flop_per_pair")
    dense_flop = L * _v(P, "dense_flop_per_token")
    attn_ref_ms = nl * math.ceil(L / _v(P, "n_sm")) * _v(P, "sparse_attn_us") / 1000.0
    tl_ms = lambda fl: fl / (_v(P, "tl_pflops") * 1e15) * 1e3  # noqa: E731
    dg_ms = lambda fl: fl / (_v(P, "dg_pflops") * 1e15) * 1e3  # noqa: E731
    out = {}
    # today: per expert call wall = python launches + the GPU tail waited for at the next where() sync
    moe_today = calls * d["wall_us_per_expert_call"] / 1000.0
    gpu_large = tl_ms(moe_flop) + tl_ms(dense_flop) + attn_ref_ms
    today = moe_today + moe_fixed_host + nonmoe + tail
    today = max(today, gpu_large + tail)
    oom_L = math.sqrt(_v(P, "hbm_free_gib") * 2 ** 30 / (2 * _v(P, "index_n_heads") * 2))
    out["today"] = {"ms": today, "moe_loop_ms": moe_today, "moe_fixed_ms": moe_fixed_host, "nonmoe_ms": nonmoe,
                    "gpu_compute_ms": gpu_large, "seed_draft_ms": tail, "fits_hbm": L <= oom_L,
                    "max_single_shot_L": oom_L, "numerics": "today"}
    nf = d["nofill_host_factor"]
    out["fills_off"] = {"ms": max((moe_today - calls * _v(P, "tail_us") / 1000.0) * nf + calls * _v(P, "tail_us")
                                  / 1000.0 + (moe_fixed_host + nonmoe) * nf + tail, gpu_large + tail),
                        "numerics": "bit-identical (fills only touch never-read memory; T8 L2 fills on==off 8/8)",
                        "days": 0.1, "fits_hbm": L <= oom_L}
    out["A_prestage_cold"] = {"ms": today - cold * 3 * _v(P, "c2c_extra_us") / 1000.0,
                              "saved_ms": cold * 3 * _v(P, "c2c_extra_us") / 1000.0, "c2c_bytes": cold * eb,
                              "numerics": "bit-identical", "days": 1.0, "fits_hbm": L <= oom_L}
    hb = d["host_us_per_expert_call"] * nf * (1.0 - _v(P, "where_share"))
    moe_b = calls * max(hb, d["gpu_us_per_expert_call"]) / 1000.0
    out["B_sliced_moe"] = {"ms": max(moe_b + (moe_fixed_host + nonmoe) * nf + tail, gpu_large + tail),
                           "host_us_per_call": hb, "numerics": "bit-identical (same kernels, same per-row ops/order)",
                           "days": 0.5, "fits_hbm": L <= oom_L}
    # C: grouped reference-kernel-text fp4_gemm, one launch per projection per layer, fills off
    host_c = nl * (30 + _v(P, "nonmoe_launches_per_layer")) * d["host_us_per_launch"] * nf / 1000.0
    moe_c = max(calls * eb / (_v(P, "tl_grouped_TBps") * 1e12) * 1e3, tl_ms(moe_flop))
    cold_c = cold * eb / (_v(P, "h2d_GBps") * 1e9) * 1e3
    gpu_c = moe_c + max(_v(P, "nonmoe_gpu_ms"), tl_ms(dense_flop) + attn_ref_ms) + 0.5 * cold_c
    out["C_grouped_ref_kernel"] = {"ms": max(host_c, gpu_c) + nl * 0.02 + tail, "host_ms": host_c, "gpu_ms": gpu_c,
                                   "moe_gpu_ms": moe_c, "cold_stage_ms": cold_c,
                                   "numerics": "bit-identical by construction (kernel.py text + row indirection); "
                                               "box torch.equal gate", "days": 2.5,
                                   "needs": "query-chunked indexer beyond max_single_shot_L"}
    chunks = max(0, math.ceil((L - 1) / 6))
    out["D_chunked_rows6"] = {"ms": _v(P, "r1_step_ms") + chunks * _v(P, "verify6_ms") + tail,
                              "chunks": chunks, "numerics": "A3 (decode kernels: GEMV, DeepGEMM, FlashMLA, fixed fp32, "
                                                            "decode-spelling ratio-2 pooling)", "days": 2.0}
    hot_b = (calls - cold) * eb
    moe_e = max(hot_b / (_v(P, "hbm_TBps") * 1e12) * 1e3, dg_ms(moe_flop))
    att_e = attn_ref_ms / _v(P, "flashmla_speedup")
    gpu_e = moe_e + dg_ms(dense_flop) + att_e + max(0.0, cold_c - moe_e) + _v(P, "launch_floor_ms")
    out["E_production_class"] = {"ms": gpu_e + tail, "moe_ms": moe_e, "cold_exposed_ms": max(0.0, cold_c - moe_e),
                                 "numerics": "A3 (DeepGEMM grouped, FlashMLA sparse_prefill_fwd, DG fp8 dense)",
                                 "days": 6.5}
    return out


def model_report(Ls: Sequence[int], counts_path: Optional[str] = DEFAULT_COUNTS, map_path: str = DEFAULT_MAP,
                 measured: Optional[Dict[int, float]] = None) -> dict:
    P = model_params()
    counts = load_counts(counts_path, _v(P, "n_layers"), _v(P, "n_experts"))
    hot = load_hot_map(map_path)
    rows = []
    for L in Ls:
        cen = expert_census_model(L, P, counts, hot)
        rows.append({"L": L, "census": cen, "ttft": ttft_model(L, cen, P),
                     "measured_prefill_ms": (measured or {}).get(L)})
    return {"params": P, "derived": derived(P), "counts": counts_path if counts is not None else None,
            "map": map_path if hot else None, "rows": rows}


def measured_prefill_by_L(path: str = HEADLINE_RECEIPT) -> Dict[int, float]:
    hl = _jload(path) or {}
    out = {}
    for p in (hl.get("prompts") or {}).values():
        xs = p.get("prefill_ms") or []
        if xs and p.get("prompt_tokens"):
            out[int(p["prompt_tokens"])] = sum(xs) / len(xs)
    return out


def model_lines(rep: dict) -> List[str]:
    d = rep["derived"]
    out = ["MODEL [E] -- inputs (label, source):"]
    for k, p in rep["params"].items():
        v = p["v"]
        out.append(f"  {k:30s} {v:>14.4g} [{p['label']}] {p['src']}" if isinstance(v, (int, float))
                   else f"  {k:30s} {v} [{p['label']}] {p['src']}")
    out.append("derived: " + ", ".join(f"{k} {v:.1f}" for k, v in d.items()))
    out.append(f"expert calls: {rep['rows'][0]['census']['model']}; counts {rep['counts']}; map {rep['map']}")
    hdr = f"{'L':>6} {'calls':>7} {'cold':>6} {'cold GB':>7} {'meas':>7} {'today':>8} {'fills0':>8} {'A':>8} " \
          f"{'B':>8} {'C':>8} {'D':>8} {'E':>8}  (ms)"
    out.append(hdr)
    for r in rep["rows"]:
        c, t = r["census"], r["ttft"]
        m = r["measured_prefill_ms"]
        flag = "" if t["today"]["fits_hbm"] else "  today/A/B: single-shot indexer OOM"
        out.append(f"{r['L']:>6} {c['calls']:>7.0f} {c['cold_calls']:>6.0f} {c['cold_bytes'] / 1e9:>7.2f} "
                   f"{(f'{m:.0f}' if m else '-'):>7} {t['today']['ms']:>8.0f} {t['fills_off']['ms']:>8.0f} "
                   f"{t['A_prestage_cold']['ms']:>8.0f} {t['B_sliced_moe']['ms']:>8.0f} "
                   f"{t['C_grouped_ref_kernel']['ms']:>8.0f} {t['D_chunked_rows6']['ms']:>8.0f} "
                   f"{t['E_production_class']['ms']:>8.0f}{flag}")
    return out


# ================================================================================================ the probe (box)
def _nvtx(on: bool):
    if not on:
        return None
    try:
        import torch

        return torch.cuda.nvtx if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


class PrefillProbe:
    """Forward pre/post hooks on the reference modules of one prefill (module docstring of this file): host
    perf_counter per call, optional CUDA events (GPU timeline), optional NVTX ranges, and the routed-expert census.
    Hooks return None (a pre-hook's return value would replace the module inputs) and are removed on exit."""

    def __init__(self, model, pool=None, *, nvtx: bool = False, events: bool = False, capture_moe=()):
        self.model, self.pool = model, pool
        self.nv = _nvtx(nvtx)
        self.events = bool(events)
        self.capture_moe = set(int(x) for x in capture_moe)
        self.moe_inputs: Dict[int, object] = {}
        self.records: List[dict] = []
        self._stack: List[tuple] = []
        self._handles: List = []

    # ------------------------------------------------------------ hooks
    def _tier_cold(self, layer: int, e: int) -> bool:
        if self.pool is None:
            return False
        from runtime.expert_pool import TIER_HOST

        return self.pool.tier(layer, e) == TIER_HOST

    def _wrap(self, mod, role: str, layer: Optional[int], expert: Optional[int] = None):
        import torch

        def pre(m, inputs):
            ev = None
            if self.events:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
            rows = None
            if inputs and hasattr(inputs[0], "shape"):
                rows = int(inputs[0].numel() // inputs[0].shape[-1]) if inputs[0].dim() else None
            if role == "moe" and layer in self.capture_moe:
                self.moe_inputs[layer] = inputs[0].detach().clone()
            self._stack.append((role, layer, expert, time.perf_counter(), ev, rows))
            if self.nv is not None:
                self.nv.range_push(role)
            return None

        def post(m, inputs, output):
            if self.nv is not None:
                self.nv.range_pop()
            role_, layer_, expert_, t0, ev0, rows = self._stack.pop()
            ev1 = None
            if self.events:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
            self.records.append({"role": role_, "layer": layer_, "expert": expert_, "rows": rows,
                                 "host_ms": (time.perf_counter() - t0) * 1e3, "_ev": (ev0, ev1)})
            return None

        self._handles.append(mod.register_forward_pre_hook(pre))
        self._handles.append(mod.register_forward_hook(post))

    def __enter__(self):
        m = self.model
        for i, layer in enumerate(m.layers):
            self._wrap(layer, f"L{i}", i)
            if getattr(layer, "engram", None) is not None:
                self._wrap(layer.engram, "engram", i)
            self._wrap(layer.attn, "attn", i)
            if getattr(layer.attn, "compressor", None) is not None:
                self._wrap(layer.attn.compressor, "compressor", i)
            if getattr(layer.attn, "indexer", None) is not None:
                self._wrap(layer.attn.indexer, "indexer", i)
            ffn = layer.ffn
            self._wrap(ffn, "moe", i)
            self._wrap(ffn.gate, "gate", i)
            self._wrap(ffn.shared_experts, "shared", i)
            for e, ex in enumerate(ffn.experts):
                if ex is not None:
                    self._wrap(ex, "expert.cold" if self._tier_cold(i, e) else "expert.hot", i, e)
        self._wrap(m.head, "head", None)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # ------------------------------------------------------------ results
    def finalize(self) -> None:
        """GPU-timeline ms per record (call after a device sync)."""
        for r in self.records:
            e0, e1 = r.pop("_ev", (None, None))
            r["gpu_ms"] = float(e0.elapsed_time(e1)) if (e0 is not None and e1 is not None) else None

    def census(self, L: int, topk: int) -> dict:
        """Routed experts of THIS prefill, per layer: calls (= distinct experts: the reference calls each routed
        expert once with all its rows, model.py:897-902), rows (sum = L * topk), hot / cold calls, cold ids; the C2C
        bytes those calls read (cold weights + scales; under the 'beside' SF layout the hot experts' K-major scales
        are host-resident too, runtime/expert_pool.py bind)."""
        nl = len(self.model.layers)
        per = [{"layer": l, "calls": 0, "hot": 0, "cold": 0, "rows": 0, "cold_ids": []} for l in range(nl)]
        for r in self.records:
            if not r["role"].startswith("expert."):
                continue
            p = per[r["layer"]]
            p["calls"] += 1
            p["rows"] += int(r["rows"] or 0)
            if r["role"] == "expert.cold":
                p["cold"] += 1
                p["cold_ids"].append(r["expert"])
            else:
                p["hot"] += 1
        calls = sum(p["calls"] for p in per)
        cold = sum(p["cold"] for p in per)
        eb = int(getattr(self.pool, "expert_bytes", 0) or 0) if self.pool is not None else 0
        sfb = int(getattr(self.pool, "sf_group_bytes", 0) or 0) if self.pool is not None else 0
        beside = bool(getattr(self.pool, "sf_beside", False)) if self.pool is not None else False
        c2c = cold * eb + ((calls - cold) * sfb if beside else 0)
        return {"L": L, "topk": topk, "expert_calls": calls, "hot_calls": calls - cold, "cold_calls": cold,
                "rows": sum(p["rows"] for p in per), "routed_pairs": L * topk * nl,
                "calls_per_layer": [p["calls"] for p in per], "cold_per_layer": [p["cold"] for p in per],
                "cold_ids": {p["layer"]: p["cold_ids"] for p in per if p["cold_ids"]},
                "expert_bytes": eb, "cold_bytes": cold * eb, "sf_beside": beside,
                "hot_scale_bytes_from_host": (calls - cold) * sfb if beside else 0, "c2c_bytes": c2c,
                "c2c_ms_at_copy_engine": c2c / 390.3e9 * 1e3 if c2c else 0.0}

    def roles(self) -> dict:
        """Per role: calls, host ms (inclusive), GPU-timeline ms; expert calls split hot/cold with per-call means;
        moe exclusive host ms = moe - (gate + shared + experts) = the loop's where()/index/y[idx]+= and syncs."""
        agg = collections.defaultdict(lambda: {"n": 0, "host_ms": 0.0, "gpu_ms": 0.0})
        for r in self.records:
            a = agg["L*" if r["role"].startswith("L") else r["role"]]
            a["n"] += 1
            a["host_ms"] += r["host_ms"]
            a["gpu_ms"] += r["gpu_ms"] or 0.0
        out = {k: dict(v) for k, v in agg.items()}
        for k in ("expert.hot", "expert.cold"):
            if k in out and out[k]["n"]:
                out[k]["host_us_per_call"] = 1000 * out[k]["host_ms"] / out[k]["n"]
                out[k]["gpu_us_per_call"] = 1000 * out[k]["gpu_ms"] / out[k]["n"]
        inner = sum(out.get(k, {}).get("host_ms", 0.0) for k in ("gate", "shared", "expert.hot", "expert.cold"))
        if "moe" in out:
            out["moe"]["exclusive_host_ms"] = out["moe"]["host_ms"] - inner
        return out


# ------------------------------------------------------------------------------------------------ stage-B probe
def sliced_moe_forward(moe, x, combine: str = "iadd"):
    """The reference MoE.forward (model.py:891-906) with the per-expert `torch.where(indices == i)` (a host sync per
    routed expert, nonzero) replaced by ONE stable sort + ONE bincount().tolist() per layer (the reference already
    syncs there, 896). Per expert, in ascending id: the same rows in the same order (stable sort of the row-major
    pair index = where()'s row order), the same `weights[idx, top]`, the same reference `Expert` call; combine
    "iadd" = the reference's `y[idx] += out` spelling, "index_add" = `y.index_add_(0, idx, out.float())` (idx unique
    within an expert, so each element gets exactly one fp32 add of the same operands). Bench-side probe only."""
    import torch

    if combine not in ("iadd", "index_add"):
        raise ValueError(f"combine {combine!r}")
    shape = x.size()
    x = x.view(-1, moe.dim)
    weights, indices = moe.gate(x, None)
    y = torch.zeros_like(x, dtype=torch.float32)
    k = indices.size(1)
    flat = indices.flatten()
    order = torch.sort(flat, stable=True).indices
    counts = torch.bincount(flat, minlength=moe.n_routed_experts).tolist()
    rows = torch.div(order, k, rounding_mode="floor")
    w = weights.flatten()[order].unsqueeze(-1)
    off = 0
    for i in range(moe.experts_start_idx, moe.experts_end_idx):
        c = counts[i]
        if c == 0:
            continue
        idx = rows[off:off + c]
        out = moe.experts[i](x[idx], w[off:off + c])
        if combine == "iadd":
            y[idx] += out
        else:
            y.index_add_(0, idx, out.float())
        off += c
    y += moe.shared_experts(x)
    return y.type_as(x).view(shape)


def moe_probe(model, inputs: Dict[int, object], reps: int = 3, log=print) -> List[dict]:
    """For each captured layer: reference MoE.forward vs sliced_moe_forward (both combines): torch.equal and synced
    wall ms (median of reps)."""
    import torch

    cuda = torch.cuda.is_available()
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    out = []
    for l in sorted(inputs):
        moe, x = model.layers[l].ffn, inputs[l]
        rec = {"layer": l, "rows": int(x.numel() // x.shape[-1])}
        fns = {"reference": lambda: moe(x), "sliced_iadd": lambda: sliced_moe_forward(moe, x, "iadd"),
               "sliced_index_add": lambda: sliced_moe_forward(moe, x, "index_add")}
        res = {}
        with torch.inference_mode():
            for name, fn in fns.items():
                ts = []
                y = None
                for _ in range(max(1, reps)):
                    sync()
                    t0 = time.perf_counter()
                    y = fn()
                    sync()
                    ts.append((time.perf_counter() - t0) * 1e3)
                res[name] = y
                rec[f"{name}_ms"] = sorted(ts)[len(ts) // 2]
        rec["equal_iadd"] = bool(torch.equal(res["reference"], res["sliced_iadd"]))
        rec["equal_index_add"] = bool(torch.equal(res["reference"], res["sliced_index_add"]))
        out.append(rec)
        log(f"[moe-probe] L{l} rows {rec['rows']}: reference {rec['reference_ms']:.2f} ms | sliced iadd "
            f"{rec['sliced_iadd_ms']:.2f} (equal {rec['equal_iadd']}) | sliced index_add "
            f"{rec['sliced_index_add_ms']:.2f} (equal {rec['equal_index_add']})")
    return out


# ------------------------------------------------------------------------------------------------ one prefill
def _sha(t) -> str:
    import torch

    return hashlib.sha256(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


def prefill_once(model, ids: Sequence[int], hidden: Sequence[int], *, seed: bool = True, nvtx: bool = False,
                 device=None) -> dict:
    """One prefill as DSparkLoop.start runs it (prefill_hidden, then seed_draft_rings when the model has mtp stages),
    each synced and timed; NVTX "prefill" / "seed_draft" when nvtx. -> {prefill_ms, seed_ms, first, logits_sha,
    hidden_sha}. The first draft (a graph replay, ~2.9 ms, headline) is not run here."""
    import torch

    from runtime.draft_step import prefill_hidden, seed_draft_rings

    dev = torch.device(device) if device is not None else model.embed.weight.device
    cuda = dev.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    nv = _nvtx(nvtx and cuda)
    tokens = torch.tensor([list(ids)], dtype=torch.long, device=dev)
    sync()
    t0 = time.perf_counter()
    if nv:
        nv.range_push("prefill")
    out, logits, mh = prefill_hidden(model, tokens, hidden)
    if nv:
        nv.range_pop()
    sync()
    t1 = time.perf_counter()
    seed_ms = None
    if seed and getattr(model, "mtp", None) and mh is not None:
        if nv:
            nv.range_push("seed_draft")
        seed_draft_rings(model, out.reshape(-1)[:1], mh)
        if nv:
            nv.range_pop()
        sync()
        seed_ms = (time.perf_counter() - t1) * 1e3
    lg = logits.reshape(-1, logits.size(-1))[-1]
    return {"prefill_ms": (t1 - t0) * 1e3, "seed_ms": seed_ms, "first": int(out.reshape(-1)[0]),
            "logits_sha": _sha(lg), "hidden_sha": _sha(mh) if mh is not None else None}


@contextlib.contextmanager
def fills(on: bool):
    """torch.utils.deterministic.fill_uninitialized_memory scoped (the headline runs with it on: set_determinism)."""
    import torch

    det = getattr(torch.utils, "deterministic", None)
    prev = getattr(det, "fill_uninitialized_memory", None)
    try:
        if det is not None and prev is not None:
            det.fill_uninitialized_memory = bool(on)
        yield
    finally:
        if det is not None and prev is not None:
            det.fill_uninitialized_memory = prev


# ------------------------------------------------------------------------------------------------ run (box)
def build_run_parser(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--only", default="prose", help="comma-separated fixture tags ('all' = every prompt)")
    ap.add_argument("--reps", type=int, default=3, help="untraced prefills per prompt (fills as the headline: on)")
    ap.add_argument("--fills", choices=["on", "off", "both"], default="both",
                    help="both = also --reps prefills with fill_uninitialized_memory off + bit-equality")
    ap.add_argument("--no-hooked", action="store_true", help="skip the hooked (census) prefill")
    ap.add_argument("--no-events", action="store_true", help="hooked prefill without CUDA events")
    ap.add_argument("--profile", action="store_true",
                    help="cudaProfilerStart/Stop + NVTX around the hooked prefill (run under nsys)")
    ap.add_argument("--probe-moe", default="", help="layers whose MoE is re-run reference vs sliced, e.g. 0,1,14,39")
    ap.add_argument("--probe-reps", type=int, default=3)
    ap.add_argument("--rows-probe", default="", help="row_census.census rows, e.g. 42,64")
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--device", default="cuda", help=argparse.SUPPRESS)   # tests: cpu (emulated mini model)
    ap.add_argument("--json", default=None, help="default scratch/prefill-anatomy-<date>.json")


def run_main(a) -> int:
    import torch

    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load

    rep: Dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "errors": [],
                 "labels": "[M] measured on this box run"}
    json_path = a.json or os.path.join(REPO, "scratch", f"prefill-anatomy-{time.strftime('%Y%m%d-%H%M')}.json")
    t0 = time.time()
    mp = a.map if a.map and os.path.exists(a.map) else None
    model = build_and_load(a.model_dir, mp, a.device, config_path=a.config, reserve_host_gib=a.reserve_host_gib,
                           load_mtp=True)
    cuda = torch.device(a.device).type == "cuda"
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    rep["boot"] = {"wall_s": time.time() - t0, "map": mp}
    try:
        rep["boot"]["summary"] = model.runtime_load.summary()
    except Exception:  # noqa: BLE001
        pass
    rep["determinism"] = set_determinism(0)          # the headline's state: deterministic algorithms, fills ON
    gen = Generator(model, a.model_dir, mode="eager", nvtx=False)
    pool = getattr(getattr(model, "runtime_load", None), "pool", None)
    mt = getattr(model, "runtime_mtp", None)
    hidden = tuple(int(i) for i in mt.target_layer_ids) if mt is not None else ()
    topk = int(model.layers[0].ffn.n_activated_experts)
    rep["pool"] = pool.describe_sf() if pool is not None and hasattr(pool, "describe_sf") else None
    with open(a.fixture) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if a.only and a.only != "all":
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    rep["prompts"] = {}
    warmed = False
    for r in rows:
        tag = r["tag"]
        P: Dict = {}
        rep["prompts"][tag] = P
        try:
            ids = gen.encode([{"role": "user", "content": r["prompt"]}])
            P["L"] = len(ids)
            if not warmed:                          # TileLang JIT / cuBLAS handles: never timed
                prefill_once(model, ids, hidden)
                warmed = True
            P["fills_on"] = [prefill_once(model, ids, hidden) for _ in range(max(0, a.reps))]
            if a.fills in ("off", "both"):
                with fills(False):
                    P["fills_off"] = [prefill_once(model, ids, hidden) for _ in range(max(0, a.reps))]
                if P["fills_on"] and P["fills_off"]:
                    ref = P["fills_on"][0]
                    P["fills_off_bit_equal"] = all(x["first"] == ref["first"] and x["logits_sha"] == ref["logits_sha"]
                                                   and x["hidden_sha"] == ref["hidden_sha"] for x in P["fills_off"])
            for k in ("fills_on", "fills_off"):
                xs = P.get(k) or []
                if xs:
                    P[f"{k}_prefill_ms_median"] = sorted(x["prefill_ms"] for x in xs)[len(xs) // 2]
                    ss = [x["seed_ms"] for x in xs if x["seed_ms"] is not None]
                    P[f"{k}_seed_ms_median"] = sorted(ss)[len(ss) // 2] if ss else None
            P["self_replay_equal"] = len({(x["first"], x["logits_sha"], x["hidden_sha"])
                                          for x in P["fills_on"]}) <= 1
            print(f"[prefill] {tag} L={P['L']}: fills on {P.get('fills_on_prefill_ms_median')} ms "
                  f"(seed {P.get('fills_on_seed_ms_median')}), fills off {P.get('fills_off_prefill_ms_median')} ms, "
                  f"off==on {P.get('fills_off_bit_equal')}, replay {P['self_replay_equal']}", flush=True)
            if not a.no_hooked:
                cap = [int(x) for x in a.probe_moe.split(",") if x.strip()]
                probe = PrefillProbe(model, pool, nvtx=a.profile and cuda, events=cuda and not a.no_events,
                                     capture_moe=cap)
                sync()
                if a.profile and cuda:
                    torch.cuda.profiler.start()
                with probe:
                    hk = prefill_once(model, ids, hidden, nvtx=a.profile)
                sync()
                if a.profile and cuda:
                    torch.cuda.profiler.stop()
                probe.finalize()
                P["hooked"] = {"run": hk, "roles": probe.roles(), "census": probe.census(P["L"], topk)}
                c = P["hooked"]["census"]
                same = hk["logits_sha"] == P["fills_on"][0]["logits_sha"] if P["fills_on"] else None
                P["hooked"]["equal_untraced"] = same
                print(f"[census] {tag}: {c['expert_calls']} expert calls ({c['hot_calls']} hot, {c['cold_calls']} "
                      f"cold), per layer mean {c['expert_calls'] / max(1, len(c['calls_per_layer'])):.1f} max "
                      f"{max(c['calls_per_layer'] or [0])}; C2C {c['c2c_bytes'] / 1e9:.2f} GB; hooked prefill "
                      f"{hk['prefill_ms']:.0f} ms (logits equal to untraced: {same})", flush=True)
                for k_, v in sorted(P["hooked"]["roles"].items(), key=lambda kv: -kv[1]["host_ms"]):
                    print(f"  role {k_:12s} n {v['n']:6d} host {v['host_ms']:9.1f} ms gpu {v['gpu_ms']:9.1f} ms"
                          + (f"  per call host {v['host_us_per_call']:.0f} us gpu {v['gpu_us_per_call']:.0f} us"
                             if "host_us_per_call" in v else "")
                          + (f"  exclusive host {v['exclusive_host_ms']:.1f} ms" if "exclusive_host_ms" in v else ""),
                          flush=True)
                if cap:
                    P["moe_probe"] = moe_probe(model, probe.moe_inputs, reps=a.probe_reps)
                    with fills(False):
                        P["moe_probe_fills_off"] = moe_probe(model, probe.moe_inputs, reps=a.probe_reps)
                a.no_hooked = a.no_hooked or a.profile    # one profiled prefill per run (one nsys capture range)
        except Exception:  # noqa: BLE001
            rep["errors"].append({"stage": "prompt", "tag": tag, "traceback": traceback.format_exc()})
            print(f"[prefill] {tag}: FAILED\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    if a.rows_probe:
        from runtime.row_census import census, census_table

        rep["rows_probe"] = {}
        for R in [int(x) for x in a.rows_probe.split(",") if x.strip()]:
            try:
                recs = census(model, rows=R, gemv_timing=False)
                rep["rows_probe"][str(R)] = recs
                for line in census_table(recs):
                    print(f"[rows {R}] {line}", flush=True)
            except Exception:  # noqa: BLE001
                rep["errors"].append({"stage": f"rows_probe {R}", "traceback": traceback.format_exc()})
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json.loads(json.dumps(rep, default=str)), f, indent=1)
    print(f"[prefill-anatomy] receipt {json_path}; errors {len(rep['errors'])}", flush=True)
    return 0 if not rep["errors"] else 2


# ================================================================================================ analyze (sqlite)
SYNC_RX = ("Synchronize",)
LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaGraphLaunch")


def innermost(ranges: List[tuple], times: List[float]) -> List[Optional[tuple]]:
    """For each time t, the innermost range (start, end, name, ...) containing it, for properly nested ranges (one
    thread's NVTX push/pop). Sweep: O((R + T) log). Ranges are tuples whose [0], [1] are start, end."""
    rs = sorted(ranges, key=lambda r: (r[0], -r[1]))
    order = sorted(range(len(times)), key=lambda i: times[i])
    out: List[Optional[tuple]] = [None] * len(times)
    stack: List[tuple] = []
    j = 0
    for i in order:
        t = times[i]
        while j < len(rs) and rs[j][0] <= t:
            r = rs[j]
            while stack and stack[-1][1] < r[0]:
                stack.pop()
            stack.append(r)
            j += 1
        while stack and stack[-1][1] < t:
            stack.pop()
        out[i] = stack[-1] if stack and stack[-1][0] <= t <= stack[-1][1] else None
    return out


def _union_ms(iv: List[Tuple[float, float]]) -> float:
    busy, last = 0.0, None
    for s, e in sorted(iv):
        if last is None or s > last:
            busy += e - s
            last = e
        elif e > last:
            busy += e - last
            last = e
    return busy / 1e6


def analyze(db_path: str, log=print) -> dict:
    db = sqlite3.connect(db_path)
    q = lambda s, *a: db.execute(s, a).fetchall()  # noqa: E731
    tables = {r[0] for r in q("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return {"error": f"no CUPTI_ACTIVITY_KIND_KERNEL in {sorted(tables)}"}
    kern = q("""select k.start, k.end, k.correlationId, coalesce(s.value, ''), coalesce(d.value, '')
                from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id = k.shortName
                left join StringIds d on d.id = k.demangledName order by k.start""")
    rt, api = {}, []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        cols = {r[1] for r in q("pragma table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")}
        name_sql = "coalesce((select value from StringIds where id = nameId), '')" if "nameId" in cols else "''"
        for s, e, cid, nm in q(f"select start, end, correlationId, {name_sql} from CUPTI_ACTIVITY_KIND_RUNTIME"):
            rt[cid] = s
            api.append((s, e, nm or ""))
    mcpy = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mcpy = q("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start")
    nvtx = []
    if "NVTX_EVENTS" in tables:
        nvtx = [(s, e, t or "", tid) for s, e, t, tid in
                q("select start, end, coalesce(text, (select value from StringIds where id=textId)), globalTid "
                  "from NVTX_EVENTS where end is not null")]
    phases = sorted((s, e, t) for s, e, t, _ in nvtx if t in ("prefill", "seed_draft"))
    if not phases:
        return {"error": "no prefill / seed_draft NVTX range (was the profile taken with `run --profile`?)"}
    main_tid = collections.Counter(tid for _, _, t, tid in nvtx if t == "prefill").most_common(1)
    main_tid = main_tid[0][0] if main_tid else None
    own = [(s, e, t) for s, e, t, tid in nvtx if main_tid is None or tid == main_tid]
    roles = [(s, e, t) for s, e, t in own if t in PREFILL_ROLES]
    layers = [(s, e, t) for s, e, t in own if t.startswith("L") and t[1:].isdigit()]
    roles_known = any(t == "moe" for _, _, t in roles)
    log(f"kernels {len(kern)}  runtime-api {len(api)}  nvtx {len(nvtx)}  phases {len(phases)}  role ranges "
        f"{len(roles)}  layer ranges {len(layers)}  memcpy {len(mcpy)}")
    launch_t = [rt.get(k[2], k[0]) for k in kern]
    k_role = innermost(roles, launch_t)
    k_layer = innermost(layers, launch_t)
    k_phase = innermost(phases, launch_t)
    rep: Dict = {"roles_known": roles_known, "phases": {}, "family_labels": PREFILL_FAMILIES}
    for ph in ("prefill", "seed_draft"):
        pr = [r for r in phases if r[2] == ph]
        if not pr:
            continue
        ps, pe = pr[0][0], pr[0][1]                 # one profiled prefill per capture (run --profile)
        idx = [i for i, p in enumerate(k_phase) if p is not None and p[2] == ph and p[0] == ps]
        fam = collections.defaultdict(lambda: [0.0, 0])
        role_k = collections.defaultdict(lambda: [0.0, 0])
        layer_k = collections.defaultdict(float)
        names = collections.defaultdict(lambda: [0.0, 0, ""])
        for i in idx:
            s, e, _, nm, full = kern[i]
            r = k_role[i][2] if k_role[i] is not None else None
            f = prefill_family_of(nm, full, r, roles_known)
            ms = (e - s) / 1e6
            fam[f][0] += ms
            fam[f][1] += 1
            role_k[r or "other"][0] += ms
            role_k[r or "other"][1] += 1
            if k_layer[i] is not None:
                layer_k[k_layer[i][2]] += ms
            n_ = names[nm]
            n_[0] += ms
            n_[1] += 1
            n_[2] = n_[2] or f
        wall = (pe - ps) / 1e6
        busy = _union_ms([(kern[i][0], kern[i][1]) for i in idx])
        ksum = sum(fam[f][0] for f in fam)
        apis = collections.Counter()
        sync_ms = 0.0
        for s, e, nm in api:
            if ps <= s <= pe:
                apis[nm] += 1
                if any(x in nm for x in SYNC_RX):
                    sync_ms += (e - s) / 1e6
        mc = collections.defaultdict(lambda: [0, 0.0, 0])
        for s, e, nb, kind in mcpy:
            if ps <= s <= pe:
                m = mc[COPY_KIND.get(kind, str(kind))]
                m[0] += nb or 0
                m[1] += (e - s) / 1e6
                m[2] += 1
        R: Dict = {"wall_ms": wall, "busy_ms": busy, "sum_kernel_ms": ksum, "idle_ms": wall - busy,
                   "gpu_busy_share": busy / wall if wall else None, "kernels": len(idx),
                   "families": {f: {"ms": v[0], "launches": v[1]} for f, v in fam.items()},
                   "role_kernels": {r: {"ms": v[0], "launches": v[1]} for r, v in role_k.items()},
                   "layer_kernel_ms": dict(layer_k),
                   "top_kernels": sorted(([n, v[0], v[1], v[2]] for n, v in names.items()), key=lambda x: -x[1])[:30],
                   "api": dict(apis.most_common()),
                   "launch_calls": sum(c for n, c in apis.items() if any(n.startswith(x) for x in LAUNCH_NAMES)),
                   "sync_calls": sum(c for n, c in apis.items() if any(x in n for x in SYNC_RX)),
                   "sync_blocked_ms": sync_ms,
                   "memcpy": {k: {"bytes": v[0], "ms": v[1], "n": v[2]} for k, v in mc.items()}}
        if ph == "prefill":
            R.update(_per_range(kern, k_role, roles, layers, ps, pe, roles_known))
        rep["phases"][ph] = R
    return rep


def _per_range(kern, k_role, roles, layers, ps, pe, roles_known) -> dict:
    """Role ranges inside the prefill: host wall per role, the kernels each launched, and per expert call (hot / cold)
    wall, kernel count, GPU ms and the fp4_gemm us -> the census from nsys alone (expert ranges per layer)."""
    by_range = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])   # id(range) -> launches, gpu ms, fp4 ms, fp4 n
    for i, r in enumerate(k_role):
        if r is None or not (ps <= r[0] <= pe):
            continue
        s, e, _, nm, full = kern[i]
        b = by_range[(r[0], r[1], r[2])]
        b[0] += 1
        b[1] += (e - s) / 1e6
        if "fp4_gemm" in nm or "fp4_gemm" in full:
            b[2] += (e - s) / 1e6
            b[3] += 1
    inside = [r for r in roles if ps <= r[0] <= pe]
    lay = innermost(layers, [r[0] for r in inside])
    role_wall = collections.defaultdict(lambda: [0, 0.0, 0, 0.0])
    per_layer = collections.defaultdict(lambda: {"hot": 0, "cold": 0})
    ex = {"expert.hot": [], "expert.cold": []}
    for r, L_ in zip(inside, lay):
        w = role_wall[r[2]]
        w[0] += 1
        w[1] += (r[1] - r[0]) / 1e6
        b = by_range.get((r[0], r[1], r[2]), [0, 0.0, 0.0, 0])
        w[2] += b[0]
        w[3] += b[1]
        if r[2] in ex:
            ex[r[2]].append(((r[1] - r[0]) / 1e6, b[0], b[1], b[2], b[3]))
            if L_ is not None:
                per_layer[L_[2]]["hot" if r[2] == "expert.hot" else "cold"] += 1
    out = {"role_ranges": {k: {"n": v[0], "wall_ms": v[1], "launches": v[2], "gpu_ms": v[3]}
                           for k, v in role_wall.items()}}
    for k, xs in ex.items():
        if xs:
            n = len(xs)
            fp4n = sum(x[4] for x in xs)
            out[f"{k}.per_call"] = {"n": n, "wall_us": 1000 * sum(x[0] for x in xs) / n,
                                    "launches": sum(x[1] for x in xs) / n, "gpu_us": 1000 * sum(x[2] for x in xs) / n,
                                    "fp4_us_per_launch": 1000 * sum(x[3] for x in xs) / fp4n if fp4n else None}
    if "moe" in out["role_ranges"]:
        inner = sum(out["role_ranges"].get(k, {}).get("wall_ms", 0.0)
                    for k in ("gate", "shared", "expert.hot", "expert.cold"))
        out["role_ranges"]["moe"]["exclusive_wall_ms"] = out["role_ranges"]["moe"]["wall_ms"] - inner
    out["experts_per_layer"] = {k: v for k, v in sorted(per_layer.items(), key=lambda kv: int(kv[0][1:]))}
    out["expert_calls"] = sum(v["hot"] + v["cold"] for v in per_layer.values())
    return out


def analyze_lines(rep: dict) -> List[str]:
    if rep.get("error"):
        return [rep["error"]]
    out = [f"roles {'known' if rep['roles_known'] else 'UNKNOWN (no moe NVTX range)'}"]
    for ph, R in rep["phases"].items():
        out.append(f"\n[{ph}] wall {R['wall_ms']:.1f} ms, GPU busy {R['busy_ms']:.1f} ms "
                   f"({100 * (R['gpu_busy_share'] or 0):.0f} %), idle {R['idle_ms']:.1f}, kernels {R['kernels']}, "
                   f"launch calls {R['launch_calls']}, syncs {R['sync_calls']} ({R['sync_blocked_ms']:.1f} ms blocked)")
        for f, v in sorted(R["families"].items(), key=lambda x: -x[1]["ms"]):
            out.append(f"  {f:20s} {v['ms']:9.2f} ms {v['launches']:8d} launches")
        for k, v in R.get("memcpy", {}).items():
            out.append(f"  memcpy {k}: {v['bytes'] / 1e6:.2f} MB in {v['n']} copies, {v['ms']:.2f} ms")
        if "role_ranges" in R:
            out.append("  role ranges (host wall incl. nested | kernels launched inside | their GPU ms):")
            for k, v in sorted(R["role_ranges"].items(), key=lambda x: -x[1]["wall_ms"]):
                out.append(f"    {k:12s} n {v['n']:6d} wall {v['wall_ms']:9.1f} ms  launches {v['launches']:7d}  "
                           f"gpu {v['gpu_ms']:8.1f} ms"
                           + (f"  exclusive {v['exclusive_wall_ms']:.1f} ms" if "exclusive_wall_ms" in v else ""))
            for k in ("expert.hot", "expert.cold"):
                pc = R.get(f"{k}.per_call")
                if pc:
                    out.append(f"  {k} per call: n {pc['n']}, wall {pc['wall_us']:.0f} us, {pc['launches']:.1f} "
                               f"launches, GPU {pc['gpu_us']:.0f} us, fp4_gemm "
                               f"{pc['fp4_us_per_launch'] or 0:.1f} us/launch")
            out.append(f"  expert calls (ranges) {R.get('expert_calls')}")
    return out


# ================================================================================================ box commands
BOX_W = "$HOME/dsv41/decode"
BOX_M = "/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170"
BOX_GPU = "nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55"


def box_lines(tag: str = "prose") -> List[str]:
    d = (f'D="docker run --rm --network none --device {BOX_GPU} --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK '
         f'-v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 '
         f'-v $HOME/dsv41/w0/jit-cache/tilelang:/root/.tilelang -v $HOME/dsv41/w0/jit-cache/triton:/root/.triton '
         f'-v $HOME/dsv41/w0/jit-cache/nv:/root/.nv -v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm '
         f'-e DG_JIT_CACHE_DIR=/root/.deep_gemm -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json"')
    nsys = ("--cap-add SYS_ADMIN -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro --entrypoint "
            "/nsys/target-linux-sbsa-armv8/nsys vllm/vllm-openai:v0.31.0 profile --trace=cuda,nvtx --sample=none "
            "--cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true "
            f"-o /work/scratch/prof/prefill-{tag} python3")
    return [
        "# box: Station milo@192.168.1.9, clone ~/dsv41/decode (git pull first); refuse if a ds41/mk- container is up",
        f"W={BOX_W}; M={BOX_M}",
        "docker ps --format '{{.Names}}' | grep -q 'ds41\\|mk-' && echo refusing && exit 1",
        "mkdir -p $W/scratch/prof",
        d,
        "# 1) untraced timing (fills on = headline state, fills off + bit-equality), hooked census, stage-B MoE probe,",
        "#    row-invariance census at 42 and 64 rows (~5 min incl. 63 s boot)",
        f"$D --name mk-prefill --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/prefill_anatomy.py run "
        f"--model-dir /ref --only all --reps 3 --fills both --probe-moe 0,1,2,14,20,39 --rows-probe 42,64 "
        f"--json /work/scratch/prefill-anatomy-all.json > $W/scratch/prefill-anatomy-all.log 2>&1",
        "# 2) nsys: ONE hooked prefill of the 42-token prompt with NVTX ranges (traced run is slower; use 1) for ms)",
        f"$D --name mk-prefill-nsys {nsys} bench/prefill_anatomy.py run --model-dir /ref --only {tag} --reps 0 "
        f"--fills on --profile --json /work/scratch/prefill-anatomy-{tag}-nsys.json "
        f"> $W/scratch/prefill-anatomy-{tag}-nsys.log 2>&1",
        f"/usr/local/bin/nsys export --type sqlite -o $W/scratch/prof/prefill-{tag}.sqlite "
        f"$W/scratch/prof/prefill-{tag}.nsys-rep",
        f"python3 $W/bench/prefill_anatomy.py analyze $W/scratch/prof/prefill-{tag}.sqlite "
        f"--out $W/scratch/prefill-anatomy-{tag}-nsys-analysis.json > $W/scratch/prefill-anatomy-{tag}-analysis.log",
        "# copy scratch/prefill-anatomy-*.json + logs to the Mac docs/ (never write docs/ on the box)",
    ]


# ================================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    build_run_parser(sub.add_parser("run", help="box: timed + hooked prefills (GPU)"))
    an = sub.add_parser("analyze", help="nsys sqlite of `run --profile`")
    an.add_argument("sqlite")
    an.add_argument("--out", default=None)
    mo = sub.add_parser("model", help="Mac: modeled anatomy + alternatives")
    mo.add_argument("--L", default=",".join(str(x) for x in FIXTURE_LENGTHS) + ",128,512,4096,32768")
    mo.add_argument("--counts", default=DEFAULT_COUNTS)
    mo.add_argument("--map", default=DEFAULT_MAP)
    mo.add_argument("--uniform", action="store_true", help="ignore the routing counts (uniform routing)")
    mo.add_argument("--json", default=None)
    bx = sub.add_parser("box", help="print the box command lines")
    bx.add_argument("--tag", default="prose")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run_main(a)
    if a.cmd == "analyze":
        rep = analyze(a.sqlite)
        for line in analyze_lines(rep):
            print(line)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rep, f, indent=1)
        return 0 if not rep.get("error") else 1
    if a.cmd == "model":
        Ls = [int(x) for x in a.L.split(",") if x.strip()]
        rep = model_report(Ls, None if a.uniform else a.counts, a.map, measured_prefill_by_L())
        for line in model_lines(rep):
            print(line)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rep, f, indent=1, default=str)
        return 0
    for line in box_lines(a.tag):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
