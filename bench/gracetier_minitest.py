#!/usr/bin/env python3
"""gracetier_minitest.py -- P1 mini-test: the box probe for gracetier (b), the fixed-slot stage-into-slots design
(docs/P1-GRACETIER-DESIGN.md section 2.2; this file implements section 5's steps M0..M7; runner
bench/box/gracetier_minitest.sh). One JSON receipt (--json, default scratch/gracetier-minitest-<date>.json).

Subcommands (each phase prints its [M] numbers and writes them into the receipt as soon as it ends):
  vmm        M0 preflight (date, nvidia-smi, MemAvailable, HBM free) + M1, THE GATE: one physical scratch mapped behind
             two "layer" VAs (write via one, read via the other), a private straddle granule per layer, the torch
             view, then DeepGEMM MegaMoE (G = 384, real shapes) on VMM-backed L1 / L2 vs the same bytes in a plain
             allocation. torch.equal (on the bf16 bit patterns) or exit 3: stop, fall back to section 2.2's
             compaction design. Nothing after M1 runs on a failed gate.
  numerics   M2 load layer --layer (default 10): its cold experts from the checkpoint into a registered Grace pool
             (MegaMoE layout), hot experts + all SF into HBM (the VMM tensors), the all-HBM 384-group reference
             (group order) and a stock-order reference (group = expert id); describe_symm at G = 384 vs 255; Garner's
             router() called twice with an all->=0 row_map, one entry changed in between (guarded: a missing
             /garner/fastpath reports `not run`). M3 activations: the layer's MoE inputs captured by our runtime's
             reference forward on the 8-tag fixture (--acts, bench/dg_mega_numerics.py downstream's capture) plus
             synthetic routings forcing c in {0,1,2,4,8,16} distinct cold experts at T in {1,6,16,64}; saved. M4 (b)
             numerics: MegaMoE(G = 384, VMM scratch filled by gt_stage, after a second "layer" VA dirtied the
             scratch) vs the all-HBM reference, every T and c: torch.equal at all T (gate) -- on failure rows exact
             and fraction of bytes equal, as m3n1 reports.
  timing     M5 (b) timing by CUDA-graph replay: gt_stage us vs c; MegaMoE G = 384 us vs c; stage + MegaMoE; G = 255
             at the same (hot-only) routing; slopes = us per cold expert at T = 1/6/16/64; bar: stage slope <= 50 us.
             M6 (a): a cold MegaMoE (G_c = C) whose weights are UVA views of the Grace pool: does it run, is it
             torch.equal to the HBM-weight copy, flip class of bf16(y_hot) + bf16(y_cold) vs the reference, serial us
             vs c.
  r8         M7 R8: a plain read kernel and gt_stage on {8,16,48,152} programs reading Grace, alone and concurrent
             with the hot MegaMoE at set_num_sms(144), T = 6; the hot MegaMoE's slowdown; the side-stream fork/join
             cost; the P1.1 bar (concurrent (a) model with the measured rate <= 0.7 x (b)'s measured serial cost) when
             a timing receipt is given (--timing-json; `all` passes it).
  all        vmm -> numerics -> timing -> r8, each a child process (a CUDA fault in one phase cannot take the others'
             numbers with it); the children's receipts are merged into --json. vmm exit 3 stops everything.

Every number in the receipt is measured in this run or computed from numbers measured in this run (named); what was
not measured says `not measured` / `not run: <why>`. deep_gemm and torch only (runtime.dg_moe.import_deep_gemm: never
vllm); Garner's router is imported from --garner (the runner mounts the box's Pg61 fastpath read-only there).

Exit codes: 0 ran, gates passed | 2 a phase raised (see receipt errors) | 3 M1 gate failed | 4 M4 gate failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime import gracetier_pool as GP  # noqa: E402
from runtime import gracetier_stage as GS  # noqa: E402
from runtime import gracetier_vmm as GV  # noqa: E402

SCHEMA = "gracetier-minitest/1"
TS = (1, 6, 16, 64)
CS = (0, 1, 2, 4, 8, 16)
R8_PROGRAMS = (8, 16, 48, 152)
MEGA_SMS = 144                 # Garner's shipped kNumSMs for MegaMoE (B/:111)
MAX_TOKENS = 64                # one SymmBuffer per G at T_max = 64; a T-row call fills rows [:T], the rest -1
STAGE_BAR_US = 50.0            # section 5 M5 bar
P11_FACTOR = 0.7               # section 5 M7 bar
P11_CS = (1, 2, 4)
ROUTE_SCALE = 1.5
TOPK = 6
LIM_FALLBACK = 10.0            # used only if config.json has no swiglu_limit (recorded as the source)
EXIT_OK, EXIT_ERRORS, EXIT_M1, EXIT_M4 = 0, 2, 3, 4
BUSY_USED_MIB = 8192           # M0 refuses above this nvidia-smi memory.used (this process has not allocated yet)
NOT_MEASURED = "not measured"
SUBS = ("vmm", "numerics", "timing", "r8")
PHASES = {"vmm": ("M0", "M1"), "numerics": ("M2", "M3", "M4"), "timing": ("M5", "M6"), "r8": ("M7",)}


# ================================================================================================ pure helpers
def synth_routing(T: int, c: int, hot: Sequence[int], cold: Sequence[int], k: int, rng: random.Random):
    """ids [T][k] (expert ids, distinct per row, ascending) with EXACTLY c distinct cold experts; every other pick
    hot. None when infeasible (c > T*k, or c > len(cold), or too few hot experts to fill a row)."""
    if c > T * k or c > len(cold) or len(hot) < k:
        return None
    coldset = rng.sample(list(cold), c)
    rows: List[List[int]] = [[] for _ in range(T)]
    for i, e in enumerate(coldset):
        rows[i % T].append(int(e))
    for r in rows:
        pool = [h for h in hot if h not in r]
        r.extend(int(h) for h in rng.sample(pool, k - len(r)))
    return [sorted(r) for r in rows]


def synth_weights(T: int, k: int, rng: random.Random) -> List[List[float]]:
    out = []
    for _ in range(T):
        w = [rng.random() + 0.1 for _ in range(k)]
        s = sum(w)
        out.append([ROUTE_SCALE * v / s for v in w])
    return out


def distinct_cold(ids: Sequence[Sequence[int]], row: Sequence[int], H: int) -> int:
    return len({int(e) for r in ids for e in r if int(row[int(e)]) >= H})


def compare_bits(a: torch.Tensor, b: torch.Tensor) -> dict:
    """bf16 [R, D] vs bf16 [R, D] on the bit patterns: torch.equal, rows exact, fraction of elements (2-byte words)
    equal, max |a - b| over finite pairs."""
    ai, bi = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    eq = ai == bi
    fin = torch.isfinite(a.float()) & torch.isfinite(b.float())
    d = (a.float() - b.float()).abs()
    mx = float(d[fin].max()) if bool(fin.any()) else None
    return {"equal": bool(torch.equal(ai, bi)), "rows": int(a.shape[0]), "rows_exact": int(eq.all(-1).sum()),
            "frac_bytes_equal": round(float(eq.float().mean()), 6), "max_abs": mx,
            "nonfinite": int((~torch.isfinite(a.float())).sum() + (~torch.isfinite(b.float())).sum())}


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> Optional[dict]:
    """Least squares y = a + b x over the measured points -> {"slope", "intercept", "n"} (None if < 2 x values)."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None]
    if len({x for x, _ in pts}) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    b = sxy / sxx
    return {"slope": round(b, 3), "intercept": round(my - b * mx, 3), "n": n}


def gbps(nbytes: float, us: Optional[float]) -> Optional[float]:
    return None if not us else round(nbytes / (us * 1e3), 2)


def p11_bar(fork_join_us, cold_intercept_us, rate_gbps, w_hot_us, b_serial_us: Dict[int, float],
            weights_b: int, cs: Sequence[int] = P11_CS, factor: float = P11_FACTOR) -> dict:
    """Section 2.3 / 5 M7 bar: (a)-concurrent model per layer-call at c cold experts, every input measured in this
    run: fork_join_us (M7) + max(0, cold_intercept_us (M6 fit, T = 6) + c x weights_b / rate (M7, GB/s at 8
    programs under MegaMoE) - w_hot_us (M5, hot MegaMoE at T = 6)), vs (b)'s measured serial cost b_serial_us[c]
    (M5: stage + MegaMoE(c) - MegaMoE(0) at T = 6). Pass at c iff model <= factor x measured."""
    ins = {"fork_join_us": fork_join_us, "cold_intercept_us": cold_intercept_us, "rate_GBps": rate_gbps,
           "w_hot_us": w_hot_us, "weights_B": weights_b, "factor": factor}
    missing = [k for k, v in ins.items() if v is None]
    if missing or not b_serial_us:
        return {"status": NOT_MEASURED, "missing": missing + ([] if b_serial_us else ["b_serial_us"]), "inputs": ins}
    rows = {}
    for c in cs:
        if c not in b_serial_us or b_serial_us[c] is None:
            continue
        a = fork_join_us + max(0.0, cold_intercept_us + c * weights_b / (rate_gbps * 1e3) - w_hot_us)
        b = float(b_serial_us[c])
        rows[c] = {"a_model_us": round(a, 2), "b_measured_us": round(b, 2), "pass": a <= factor * b}
    if not rows:
        return {"status": NOT_MEASURED, "missing": ["b_serial_us at c in %s" % (list(cs),)], "inputs": ins}
    return {"status": "pass" if all(r["pass"] for r in rows.values()) else "fail", "per_c": rows, "inputs": ins}


def m5_summary(table: List[dict], ts: Sequence[int] = TS, bar_us: float = STAGE_BAR_US) -> dict:
    """M5 rows {"T", "c", "stage_us", "mega384_us", "both_us"} -> per T: fits (us per cold expert) and the bar."""
    out = {}
    for T in ts:
        rs = [r for r in table if r["T"] == T]
        f_stage = fit_line([r["c"] for r in rs], [r.get("stage_us") for r in rs])
        f_mega = fit_line([r["c"] for r in rs], [r.get("mega384_us") for r in rs])
        f_both = fit_line([r["c"] for r in rs], [r.get("both_us") for r in rs])
        out[str(T)] = {"stage": f_stage, "mega384": f_mega, "stage_plus_mega": f_both,
                       "stage_slope_us": None if f_stage is None else f_stage["slope"],
                       "mega_marginal_us": None if f_mega is None else f_mega["slope"],
                       "bar_stage_slope_le_us": bar_us,
                       "bar_pass": None if f_stage is None else f_stage["slope"] <= bar_us}
    return out


def b_serial_from_m5(table: List[dict], T: int = 6) -> Dict[int, float]:
    """(b)'s measured serial cost at T: stage + MegaMoE at c, minus MegaMoE at c = 0 (the hot-only call)."""
    rs = {r["c"]: r for r in table if r["T"] == T}
    base = rs.get(0, {}).get("mega384_us")
    if base is None:
        return {}
    return {c: round(r["both_us"] - base, 2) for c, r in rs.items() if c > 0 and r.get("both_us") is not None}


def default_json() -> str:
    return os.path.join("scratch", f"gracetier-minitest-{time.strftime('%Y%m%d-%H%M%S')}.json")


def child_argv(a, sub: str, json_path: str, extra: Sequence[str] = ()) -> List[str]:
    """argv for an `all` child: the same options, another subcommand and receipt path."""
    v = [sub, "--json", json_path, "--layer", str(a.layer), "--model-dir", a.model_dir, "--rowmap", a.rowmap,
         "--acts", a.acts, "--garner", a.garner, "--max-tokens", str(a.max_tokens), "--iters", str(a.iters),
         "--seed", str(a.seed), "--ts", a.ts, "--cs", a.cs, "--programs", a.programs, "--fast-math", a.fast_math]
    if a.allow_busy:
        v.append("--allow-busy")
    return v + list(extra)


def import_garner_router(path: Optional[str]):
    """(router function, module) or (None, "not run: <why>"); never raises. The import is the shipped Python ABI
    (B/:1044-1088) + libw4boundary.so's own ABI check (B/:207-231)."""
    if not path or not os.path.isdir(path):
        return None, f"not run: {path!r} is not mounted (the runner mounts the box's Pg61 fastpath there)"
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        import importlib

        B = importlib.import_module("fastpath_boundary")
    except Exception as e:  # noqa: BLE001
        return None, f"not run: import fastpath_boundary failed: {type(e).__name__}: {str(e)[:200]}"
    finally:
        if added and path in sys.path:
            sys.path.remove(path)
    if any(m == "vllm" or m.startswith("vllm.") for m in sys.modules):
        return None, "not run: importing fastpath_boundary pulled in vllm"
    try:
        B.lib()
    except Exception as e:  # noqa: BLE001
        return None, f"not run: libw4boundary.so: {type(e).__name__}: {str(e)[:200]}"
    return B.router, B


def verdict_line(phases: dict) -> str:
    def g(*path):
        d = phases
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return None
            d = d[p]
        return d

    def f(v, nd=1):
        return NOT_MEASURED if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    m5 = g("M5", "summary") or {}
    st = "/".join(f((m5.get(str(T)) or {}).get("stage_slope_us")) for T in TS)
    mm = "/".join(f((m5.get(str(T)) or {}).get("mega_marginal_us")) for T in TS)
    r8 = g("M7", "headline", "read_GBps_8_under_mega")
    return (f"GRACETIER-MINITEST: M1 torch.equal={f(g('M1', 'gate'))}; M4 torch.equal all T={f(g('M4', 'gate'))}; "
            f"stage us/expert T1/6/16/64={st}; MegaMoE marginal us/expert={mm}; "
            f"M6 uva={f(g('M6', 'uva', 'status'))}; R8 GB/s @8 under MegaMoE={f(r8)}; "
            f"P1.1 bar={f(g('M7', 'p11', 'status'))}")


def merge_children(parts: Dict[str, Optional[dict]]) -> dict:
    phases, errors = {}, []
    for sub in SUBS:
        d = parts.get(sub)
        if not d:
            continue
        phases.update(d.get("phases") or {})
        errors += [f"{sub}: {e}" for e in (d.get("errors") or [])]
    return {"phases": phases, "errors": errors}


# ================================================================================================ receipt
class Receipt:
    def __init__(self, path: str, sub: str, a=None):
        self.path = path
        self.d = {"schema": SCHEMA, "sub": sub, "argv": sys.argv[1:], "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                  "design": "docs/P1-GRACETIER-DESIGN.md section 5", "phases": {}, "errors": [], "exit": None}
        try:
            from receipt import begin

            self.d["receipt"] = begin(argv=sys.argv[1:], impls={"probe": "gracetier_minitest", "sub": sub},
                                      model_dir=getattr(a, "model_dir", None), map_path=getattr(a, "rowmap", None))
        except Exception as e:  # noqa: BLE001
            self.d["receipt"] = {"error": f"{type(e).__name__}: {e}"}
        self.save()

    def phase(self, name: str, data: dict) -> dict:
        self.d["phases"][name] = data
        self.save()
        return data

    def error(self, name: str, tb: str) -> None:
        self.d["errors"].append(f"{name}: {tb[-2500:]}")
        print(f"[{name}] FAILED\n{tb}", flush=True)
        self.save()

    def done(self, rc: int) -> int:
        self.d["exit"] = rc
        try:
            from receipt import finish

            finish(self.d.get("receipt"))
        except Exception:  # noqa: BLE001
            pass
        self.d["verdict"] = verdict_line(self.d["phases"])
        self.save()
        print(self.d["verdict"], flush=True)
        print("receipt", self.path, flush=True)
        return rc

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.d, fh, indent=1, default=str)
        os.replace(tmp, self.path)


def log(msg: str) -> None:
    print(msg, flush=True)


# ================================================================================================ box: shared parts
def _dg():  # pragma: no cover - box only
    from runtime.dg_moe import import_deep_gemm

    dg = import_deep_gemm()
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules), "vllm imported"
    return dg


def _sync() -> None:  # pragma: no cover - box only
    torch.cuda.synchronize()


def _device() -> torch.device:  # pragma: no cover - box only
    return torch.device("cuda", torch.cuda.current_device())


def _swiglu_limit(model_dir: str) -> Tuple[float, str]:
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        if cfg.get("swiglu_limit") is not None:
            return float(cfg["swiglu_limit"]), "config.json swiglu_limit"
    except Exception:  # noqa: BLE001
        pass
    return LIM_FALLBACK, "fallback constant (config.json has no swiglu_limit)"


def _graph_us(fn, iters: int) -> dict:  # pragma: no cover - box only
    from dg_contiguous_spike import graph_us

    return graph_us(fn, iters=iters)


def _us(fn, iters: int, notes: Optional[list] = None, key: str = "") -> Optional[float]:  # pragma: no cover - box
    """us per call by CUDA-graph replay; a fallback to eager timing (capture failed) is recorded in `notes`."""
    r = _graph_us(fn, iters)
    if r.get("how") != "graph" and notes is not None:
        notes.append({key: r})
    return r.get("us")


def smi_used_mib(line: Optional[str]) -> Optional[float]:
    """memory.used (MiB) from `nvidia-smi --query-gpu=name,uuid,memory.used,... --format=csv,noheader`."""
    try:
        f = [v.strip() for v in str(line).splitlines()[0].split(",")]
        return float(f[2].split()[0])
    except Exception:  # noqa: BLE001
        return None


def mega_call(dg, y, l1, sf1, l2, sf2, symm, lim: float, fast_math: bool, nsms: Optional[int] = None):
    """dg.fp8_fp4_mega_moe in runtime/dg_mega.py's form (recipe (1,1,32), swiglu, clamp); at kNumSMs = nsms when
    given (Garner's _mega_moe_call pattern, B/:508-519)."""  # pragma: no cover - box only
    from runtime.dg_mega import MEGA_RECIPE

    prev = None
    if nsms:
        prev = dg.get_num_sms()
        dg.set_num_sms(min(int(nsms), prev))
    try:
        dg.fp8_fp4_mega_moe(y, (l1, sf1), (l2, sf2), symm, recipe=MEGA_RECIPE, activation="swiglu",
                            activation_clamp=lim, fast_math=bool(fast_math))
    finally:
        if prev is not None:
            dg.set_num_sms(prev)
    return y


def quant_rows(x: torch.Tensor):  # pragma: no cover - box only
    """bf16 [R, D] -> (fp8 [R, D], e8m0 codes uint8 [R, D/32]) with the reference act_quant's scale (LayerBox)."""
    from dg_mega_numerics import quant32, scale_ref
    from runtime.dg_mega import act_codes

    q, s = quant32(x.float(), scale_ref)
    return q, act_codes(s)


def set_inputs(symm, topk: torch.Tensor, w: torch.Tensor, q: torch.Tensor, codes: torch.Tensor) -> None:
    """Rows [:R] of the SymmBuffer's input views; rows >= R carry topk -1, weight 0, x 0."""  # pragma: no cover
    from runtime.dg_mega import write_act_sf

    R = int(topk.shape[0])
    symm.topk_idx.fill_(-1)
    symm.topk_weights.zero_()
    symm.x.view(torch.uint8).zero_()
    symm.topk_idx[:R].copy_(topk)
    symm.topk_weights[:R].copy_(w.float())
    symm.x[:R].copy_(q)
    write_act_sf(symm.x_sf, codes, R)


def synth_x(T: int, D: int, g: torch.Generator) -> torch.Tensor:
    """N(0,1) with per-32 block magnitudes 2^-4..2^4 (bf16) -- used only when no captured activations exist."""
    m = torch.exp2(torch.randint(-4, 5, (T, D // 32, 1), generator=g, device="cpu").float())
    x = torch.randn(T, D // 32, 32, generator=g, dtype=torch.float32, device="cpu") * m
    return x.view(T, D).to(torch.bfloat16)


# ================================================================================================ M0 / M1
def phase_m0(a, rec: Receipt) -> dict:  # pragma: no cover - box only
    from runtime.host_tier import mem_available_bytes

    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "MemAvailable_GiB": None, "nvidia_smi": None,
           "compute_apps": None}
    av = mem_available_bytes()
    out["MemAvailable_GiB"] = None if av is None else round(av / GV.GiB, 2)
    for key, q in (("nvidia_smi", ["--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu,clocks.sm",
                                   "--format=csv,noheader"]),
                   ("compute_apps", ["--query-compute-apps=pid,used_memory", "--format=csv,noheader"])):
        try:
            out[key] = subprocess.run(["nvidia-smi"] + q, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            out[key] = f"error: {e!r}"
    f, t = torch.cuda.mem_get_info()
    out["hbm_free_GiB"], out["hbm_total_GiB"] = round(f / GV.GiB, 2), round(t / GV.GiB, 2)
    out["memory_used_MiB"] = smi_used_mib(out["nvidia_smi"])
    apps = bool(out["compute_apps"]) and not str(out["compute_apps"]).startswith("error")
    used = out["memory_used_MiB"] is not None and out["memory_used_MiB"] > BUSY_USED_MIB
    out["busy"] = apps or used
    log(f"[M0] {out['date']} MemAvailable {out['MemAvailable_GiB']} GiB, HBM free {out['hbm_free_GiB']} / "
        f"{out['hbm_total_GiB']} GiB, nvidia-smi memory.used {out['memory_used_MiB']} MiB, compute apps: "
        f"{out['compute_apps']!r}")
    rec.phase("M0", out)
    if out["busy"] and not a.allow_busy:
        raise SystemExit(f"[M0] the GPU is not idle (compute apps {out['compute_apps']!r}, memory.used "
                         f"{out['memory_used_MiB']} MiB > {BUSY_USED_MIB}): the window owner stops :30006 first "
                         f"(--allow-busy overrides)")
    return out


def phase_m1(a, rec: Receipt, dg) -> dict:  # pragma: no cover - box only
    from runtime import dg_mega as DM

    dev = _device()
    GV.ensure_context(dev.index)
    rm = GP.load_rowmap(a.rowmap)
    H = len(rm[a.layer]["hot"])
    G, D, I, k = GP.E_DEFAULT, GP.DIM, GP.INTER, TOPK
    lim, lim_src = _swiglu_limit(a.model_dir)
    fm = _fast_maths(a)[0]
    out: dict = {"H": H, "G": G, "D": D, "I": I, "fast_math": fm, "swiglu_limit": lim, "swiglu_limit_source": lim_src}
    free0 = torch.cuda.mem_get_info()[0]
    arena = GV.VmmArena(dev.index)
    out["granularity"] = arena.gran
    out["granularity_minimum"] = arena.drv.granularity(dev.index, recommended=False)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    try:
        # (1) aliasing on a small synthetic plan with a straddle granule
        gran = arena.gran
        B = gran * 5 // 8
        plan = GV.TensorPlan("m1", 8, 3, B, gran)
        shared = arena.alloc(plan.shared_bytes, "m1.shared")
        A = GV.GroupedVA(arena, plan, shared, map_hot=True, name="m1.A")
        Bv = GV.GroupedVA(arena, plan, shared, map_hot=False, name="m1.B")
        tA = A.tensor(torch.uint8, (plan.va_bytes,))
        p1 = torch.randint(0, 256, (plan.shared_bytes,), generator=g, dtype=torch.int32, device="cpu")
        p1 = p1.to(torch.uint8).to(dev)
        tA[plan.shared_va_off:].copy_(p1)
        _sync()
        via_b = Bv.bytes_view(plan.shared_va_off, plan.shared_bytes).clone()
        so = GV.align_down(plan.hot_bytes, gran)
        p2 = torch.full((gran,), 0x5A, dtype=torch.uint8, device=dev)
        p3 = torch.full((gran,), 0xA5, dtype=torch.uint8, device=dev)
        Bv.bytes_view(so, gran).copy_(p2)
        tA[so:so + gran].copy_(p3)
        _sync()
        out["alias_write_A_read_B"] = bool(torch.equal(via_b, p1))
        out["straddle_private"] = bool(torch.equal(Bv.bytes_view(so, gran), p2) and torch.equal(tA[so:so + gran], p3))
        out["shared_untouched_by_straddle_writes"] = bool(torch.equal(Bv.bytes_view(plan.shared_va_off,
                                                                                    plan.shared_bytes), p1))
        out["torch_view"] = {"data_ptr_is_va": tA.data_ptr() == A.va, "device": str(tA.device),
                             "contiguous": tA.is_contiguous(), "plan": plan.describe()}
        del tA, via_b, p1, p2, p3
        # (2) MegaMoE G = 384, real shapes: VMM-backed L1 / L2 vs the same bytes in plain allocations
        lv = GV.LayerVMM(arena, {0: H}, G, D, I)
        l1, l2 = lv.l1(0), lv.l2(0)
        gd = torch.Generator(device=dev).manual_seed(a.seed)
        for t in (l1, l2):
            t.view(torch.uint8).random_(0, 256, generator=gd)          # every byte is a valid e2m1 pair
        _sync()
        p_l1, p_l2 = l1.clone(), l2.clone()
        codes = {}
        for p, (n, kk) in (("w1", (I, D)), ("w3", (I, D)), ("w2", (D, I))):
            codes[p] = torch.empty(G, n, kk // 32, dtype=torch.uint8, device=dev).random_(118, 125, generator=gd)
        sf1, sf2 = GP.sf_banks(codes)
        rng = random.Random(a.seed)
        T = 6
        ids = [sorted(rng.sample(range(G), k)) for _ in range(T)]
        r0 = {H - 1, H, G - 1}                     # the straddle groups (hot tail / cold head) and the last group
        while len(r0) < k:
            r0.add(rng.randrange(G))
        ids[0] = sorted(r0)
        topk = torch.tensor(ids, dtype=torch.long, device=dev)
        w = torch.tensor(synth_weights(T, k, rng), dtype=torch.float32, device=dev)
        q, sc = quant_rows(synth_x(T, D, g).to(dev))
        symm = DM.make_symm_buffer(dg, G, MAX_TOKENS, k, D, I, 0)
        y = torch.zeros(MAX_TOKENS, D, dtype=torch.bfloat16, device=dev)
        set_inputs(symm, topk, w, q, sc)
        mega_call(dg, y, l1, sf1, l2, sf2, symm, lim, fm)
        _sync()
        y_vmm = y[:T].clone()
        set_inputs(symm, topk, w, q, sc)
        mega_call(dg, y, p_l1, sf1, p_l2, sf2, symm, lim, fm)
        _sync()
        y_plain = y[:T].clone()
        out["megamoe_groups_routed"] = ids
        out["weights_equal_plain"] = bool(torch.equal(l1, p_l1) and torch.equal(l2, p_l2))
        out["megamoe_vmm_vs_plain"] = compare_bits(y_vmm, y_plain)
        out["vmm"] = lv.describe()
        del l1, l2, p_l1, p_l2, y, y_vmm, y_plain, symm, codes, sf1, sf2
    except Exception as e:  # noqa: BLE001 - VMM / MegaMoE-on-VMM failing IS the gate's answer: record, gate = False
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out["traceback"] = traceback.format_exc()[-2500:]
    finally:
        _sync()
        arena.release()
        torch.cuda.empty_cache()
    out["release_order"] = [e[0] for e in arena.log[-6:]]
    out["hbm_free_delta_after_release_MiB"] = round((torch.cuda.mem_get_info()[0] - free0) / GV.MiB, 1)
    out["gate"] = bool(out.get("alias_write_A_read_B") and out.get("straddle_private")
                       and (out.get("megamoe_vmm_vs_plain") or {}).get("equal"))
    log(f"[M1] gran {out['granularity']} B; alias A->B {out.get('alias_write_A_read_B')}; straddle private "
        f"{out.get('straddle_private')}; MegaMoE G=384 VMM vs plain {out.get('megamoe_vmm_vs_plain')}; "
        f"GATE torch.equal = {out['gate']}" + (f"; error {out['error']}" if out.get("error") else ""))
    rec.phase("M1", out)
    return out


def _fast_maths(a) -> List[bool]:
    return {"0": [False], "1": [True], "both": [False, True]}[a.fast_math]


def _ints(s: str) -> List[int]:
    return [int(v) for v in str(s).split(",") if v.strip()]


# ================================================================================================ M2: the layer
class LayerSetup:  # pragma: no cover - box only
    """Layer L on the box: Grace pool (cold, registered), VMM L1 / L2 (hot groups filled, cold groups = the shared
    scratch), SF banks for all G groups (group order), the all-HBM reference (group order) and the stock-order
    reference (group = expert id), a second 'layer' VA (B) over the same scratch, SymmBuffers at G = 384 / 255."""

    def __init__(self, a, dg, want_ref: bool = True, want_stock: bool = True):
        from runtime import dg_mega as DM
        from runtime.ckpt import ShardIndex

        self.a, self.dg, self.DM = a, dg, DM
        self.dev = _device()
        GV.ensure_context(self.dev.index)
        self.L = L = a.layer
        rm = GP.load_rowmap(a.rowmap)
        self.hot, self.cold = rm[L]["hot"], rm[L]["cold"]
        self.H, self.C = len(self.hot), len(self.cold)
        self.G, self.k = GP.E_DEFAULT, TOPK
        self.row = GP.row_map(self.hot, self.cold, self.G)
        self.row_dev = torch.tensor(self.row, dtype=torch.int32, device="cpu").to(self.dev)
        self.row_long = self.row_dev.long()
        self.lim, self.lim_src = _swiglu_limit(a.model_dir)
        info: dict = {"layer": L, "H": self.H, "C": self.C, "G": self.G, "rowmap": a.rowmap,
                      "rowmap_meta": GP.rowmap_meta(a.rowmap), "swiglu_limit": self.lim,
                      "swiglu_limit_source": self.lim_src,
                      "row_map": GP.row_map_properties(self.row, self.H, self.G)}
        t0 = time.time()
        self.idx = ShardIndex.scan(a.model_dir)
        info["scan_s"] = round(time.time() - t0, 2)
        e0 = self.idx[GP.expert_names(L, self.hot[0], "w1")[0]]
        self.I, self.D = int(e0.shape[0]), int(e0.shape[1]) * 2
        D, I = self.D, self.I
        info.update(D=D, I=I, expert_bytes=GP.cold_expert_bytes(D, I))
        self.b1, self.b2 = GP.l1_bytes(D, I), GP.l2_bytes(D, I)
        free0 = torch.cuda.mem_get_info()[0]
        # Grace pool: read + register
        self.pool = GP.GracePool(GP.PoolLayout({L: self.C}, D, I), name=f"gracetier_L{L}")
        r = self.pool.read_layer(self.idx, L, self.cold)
        info["pool_read"] = dict(r, GBps=gbps(r["bytes"], r["s"] * 1e6))
        info["pool_register_s"] = round(self.pool.register(), 3)
        info["pool"] = self.pool.describe()
        # codes (all groups, group order) -> SF banks
        t0 = time.time()
        hc = GP.read_codes(self.idx, L, self.hot, D, I)
        cc = GP.read_codes(self.idx, L, self.cold, D, I)
        info["cold_tensors_read"] = self.pool.tensors_read + 3 * self.C
        self.codes = {p: torch.cat([hc[p], cc[p]], 0).to(self.dev) for p in GP.PROJ}
        self.codes_cold = {p: cc[p].to(self.dev) for p in GP.PROJ}
        self.sf1, self.sf2 = GP.sf_banks(self.codes)
        self.sf1_hot, self.sf2_hot = GP.sf_banks({p: v[:self.H] for p, v in self.codes.items()})
        self.sf1_cold, self.sf2_cold = GP.sf_banks(self.codes_cold)
        info["codes_s"] = round(time.time() - t0, 2)
        # VMM: layer L (hot mapped) + layer "B" (straddle + shared only)
        self.arena = GV.VmmArena(self.dev.index)
        self.vmm = GV.LayerVMM(self.arena, {L: self.H, -1: self.H}, self.G, D, I, map_hot={L: True, -1: False})
        self.l1, self.l2 = self.vmm.l1(L), self.vmm.l2(L)
        t0 = time.time()
        tmp = {p: torch.empty(I, D // 2, dtype=torch.int8, device="cpu") for p in ("w1", "w3")}
        tmp2 = torch.empty(D, I // 2, dtype=torch.int8, device="cpu")
        for s, e in enumerate(self.hot):
            for p in ("w1", "w3"):
                self.idx.read_tensor_into(GP.expert_names(L, e, p)[0], tmp[p])
            DM.mega_l1_weight(tmp["w1"].to(self.dev), tmp["w3"].to(self.dev), out=self.l1[s])
            self.idx.read_tensor_into(GP.expert_names(L, e, "w2")[0], tmp2)
            self.l2[s].copy_(tmp2.to(self.dev))
        _sync()
        info["hot_load_s"] = round(time.time() - t0, 2)
        self.src_tab = self.pool.src_table(L, self.dev)
        self.dtab = GS.make_dtab(self.l1, self.l2)
        vb = self.vmm.va[-1]
        self.dtab_b = torch.tensor([vb["l1"].va, vb["l2"].va], dtype=torch.int64, device="cpu").to(self.dev)
        self.src_tab_b = self.pool.src_table(L, self.dev, slots=[(c + 1) % self.C for c in range(self.C)])
        self.cnt = torch.zeros(self.C, dtype=torch.int32, device=self.dev)
        self.all_cold = torch.arange(self.H, self.G, dtype=torch.long, device=self.dev)
        # references
        self.ref_l1 = self.ref_l2 = self.st_l1 = self.st_l2 = self.st_sf1 = self.st_sf2 = None
        if want_ref:
            t0 = time.time()
            self.ref_l1 = torch.empty(self.G, 2 * I, D // 2, dtype=torch.int8, device=self.dev)
            self.ref_l2 = torch.empty(self.G, D, I // 2, dtype=torch.int8, device=self.dev)
            self.ref_l1[:self.H].copy_(self.l1[:self.H])
            self.ref_l2[:self.H].copy_(self.l2[:self.H])
            self.ref_l1[self.H:].copy_(self.pool.host_l1(L))
            self.ref_l2[self.H:].copy_(self.pool.host_l2(L))
            _sync()
            info["ref_build_s"] = round(time.time() - t0, 2)
            if want_stock:
                perm = self.row_long
                self.st_l1 = self.ref_l1.index_select(0, perm)
                self.st_l2 = self.ref_l2.index_select(0, perm)
                self.st_sf1, self.st_sf2 = GP.sf_banks({p: v.index_select(0, perm) for p, v in self.codes.items()})
                _sync()
        # SymmBuffers
        self.symm = DM.make_symm_buffer(dg, self.G, a.max_tokens, self.k, D, I, 0)
        self.symm_h = DM.make_symm_buffer(dg, self.H, a.max_tokens, self.k, D, I, 0)
        self.y = torch.zeros(a.max_tokens, D, dtype=torch.bfloat16, device=self.dev)
        self.y_h = torch.zeros(a.max_tokens, D, dtype=torch.bfloat16, device=self.dev)
        info["describe_symm"] = {str(self.G): DM.describe_symm(self.symm), str(self.H): DM.describe_symm(self.symm_h)}
        info["vmm"] = self.vmm.describe()
        info["hbm_used_by_setup_GiB"] = round((free0 - torch.cuda.mem_get_info()[0]) / GV.GiB, 2)
        self.info = info

    # ---------------------------------------------------------------- per case
    def groups_of(self, ids: torch.Tensor) -> torch.Tensor:
        return self.row_long[ids.to(self.dev).long()]

    def stage(self, symm, R: int, *, nprog=None, src_tab=None, dtab=None, groups=None) -> None:
        g = symm.topk_idx[:R].reshape(-1) if groups is None else groups
        GS.gt_stage(g, self.H, self.src_tab if src_tab is None else src_tab, self.dtab if dtab is None else dtab,
                    self.cnt, self.b1, self.b2, nprog=nprog)

    def dirty(self) -> None:
        """Stage the WRONG expert into every cold slot through the second layer's VA (shared scratch + B's straddle)."""
        GS.gt_stage(self.all_cold, self.H, self.src_tab_b, self.dtab_b, None, self.b1, self.b2)

    def mega_vmm(self, fm: bool, nsms=None):
        return mega_call(self.dg, self.y, self.l1, self.sf1, self.l2, self.sf2, self.symm, self.lim, fm, nsms)

    def mega_ref(self, fm: bool, nsms=None):
        return mega_call(self.dg, self.y, self.ref_l1, self.sf1, self.ref_l2, self.sf2, self.symm, self.lim, fm, nsms)

    def mega_stock(self, fm: bool, nsms=None):
        return mega_call(self.dg, self.y, self.st_l1, self.st_sf1, self.st_l2, self.st_sf2, self.symm, self.lim, fm,
                         nsms)

    def mega_hot(self, fm: bool, nsms=None):
        return mega_call(self.dg, self.y_h, self.l1[:self.H], self.sf1_hot, self.l2[:self.H], self.sf2_hot,
                         self.symm_h, self.lim, fm, nsms)

    def close(self) -> None:
        _sync()
        for n in ("l1", "l2", "ref_l1", "ref_l2", "st_l1", "st_l2"):
            setattr(self, n, None)
        _sync()
        try:
            self.arena.release()
        finally:
            self.pool.release()
            self.idx.close()
            torch.cuda.empty_cache()


def phase_m2(a, rec: Receipt, setup: LayerSetup, acts_x: Optional[torch.Tensor]) -> dict:  # pragma: no cover
    out = dict(setup.info)
    out["router"] = router_check(a, setup, acts_x)
    log(f"[M2] L{setup.L}: {setup.C} cold experts {out['pool_read']['bytes'] / 1e9:.2f} GB read in "
        f"{out['pool_read']['s']} s, registered in {out['pool_register_s']} s; hot load {out['hot_load_s']} s; "
        f"symm bytes G=384 {out['describe_symm'][str(setup.G)]['num_bytes']} vs G={setup.H} "
        f"{out['describe_symm'][str(setup.H)]['num_bytes']}; router: {out['router'].get('status')}")
    rec.phase("M2", out)
    return out


def router_check(a, setup: LayerSetup, acts_x: Optional[torch.Tensor]) -> dict:  # pragma: no cover - box only
    """Garner's router() twice on the same T = 6 hidden states with gracetier's all->=0 row_map; between the calls
    one entry is swapped with an expert the first call did not pick. Never raises."""
    router, B = import_garner_router(a.garner)
    if router is None:
        return {"status": B}
    try:
        L, dev, dg, idx = setup.L, setup.dev, setup.dg, setup.idx
        names = {"gate_w": f"layers.{L}.ffn.gate.weight", "bias": f"layers.{L}.ffn.gate.bias"}
        for n in names.values():
            if n not in idx:
                return {"status": f"not run: {n} not in the checkpoint index"}
        gate_w = idx.read_tensor(names["gate_w"]).to(torch.bfloat16).contiguous().to(dev)
        bias = idx.read_tensor(names["bias"]).float().contiguous().to(dev)
        T = 6
        g0 = torch.Generator(device="cpu").manual_seed(a.seed)
        li = acts_x[:T] if acts_x is not None and acts_x.shape[0] >= T else synth_x(T, setup.D, g0)
        li = li.to(torch.bfloat16).contiguous().to(dev)
        symm = setup.DM.make_symm_buffer(dg, setup.G, 16, setup.k, setup.D, setup.I, 1)
        sbm = int(dg.get_block_m_for_mega_moe(1, setup.G, symm.num_max_tokens_per_rank, T, setup.k, "fp8xfp4"))

        def call(rm):
            symm.topk_idx.fill_(-7)
            ids, w, hot, has, pos, rows = router(li, gate_w, bias, rm, symm.topk_idx[:T], symm.topk_weights[:T],
                                                 layer=L, prep=(symm.x, symm.x_sf, symm.shared_l1_acts_sf, sbm))
            _sync()
            return {"ids": ids.clone(), "w": w.clone(), "has": has.clone(), "mega_idx": symm.topk_idx[:T].clone(),
                    "mega_w": symm.topk_weights[:T].clone()}

        rm1 = setup.row_dev.clone()
        c1 = call(rm1)
        e_star = int(c1["ids"][0, 0])
        picked = set(int(v) for v in c1["ids"].reshape(-1).tolist())
        e2 = next(e for e in range(setup.G) if e not in picked)
        rm2 = rm1.clone()
        rm2[e_star], rm2[e2] = rm1[e2].clone(), rm1[e_star].clone()
        c2 = call(rm2)
        want1 = rm1.long()[c1["ids"]]
        want2 = rm2.long()[c2["ids"]]
        at = c2["ids"] == e_star
        res = {"status": "ran", "T": T, "sbm": sbm, "rows_hidden_states": "captured acts" if acts_x is not None
               else "synthetic", "ids_call1": c1["ids"].tolist(), "mega_idx_call1": c1["mega_idx"].tolist(),
               "has_all_zero": bool((c1["has"] == 0).all() and (c2["has"] == 0).all()),
               "topk_idx_equals_row_map_of_ids": bool(torch.equal(c1["mega_idx"], want1)
                                                       and torch.equal(c2["mega_idx"], want2)),
               "same_routing_both_calls": bool(torch.equal(c1["ids"], c2["ids"])),
               "changed_entry": {"expert": e_star, "group_before": int(rm1[e_star]), "group_after": int(rm2[e_star]),
                                 "swapped_with": e2},
               "reads_row_map_per_call": bool(bool(at.any()) and torch.equal(c2["mega_idx"][at],
                                                                          rm2.long()[e_star].expand(int(at.sum())))
                                              and not torch.equal(c1["mega_idx"], c2["mega_idx"])),
               "mega_w_equals_topk_w": bool(torch.equal(c1["mega_w"], c1["w"]))}
        return res
    except Exception as e:  # noqa: BLE001
        return {"status": f"error: {type(e).__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()[-2000:]}


# ================================================================================================ M3: activations
def load_acts(path: str, L: int):  # pragma: no cover - box only (the file is a box capture)
    """-> (rows dict {"x","ids","w"} of every tag concatenated, the 8 six-row windows, meta) or (None, [], why)."""
    if not path or not os.path.exists(path):
        return None, [], f"not run: no activations file {path!r} (bench/dg_mega_numerics.py downstream writes it)"
    try:
        acts = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        acts = torch.load(path, map_location="cpu")
    from dg_mega_numerics import load_windows

    xs, ids, ws, tags = [], [], [], []
    for tag in sorted(acts.get("rows", {})):
        per = acts["rows"][tag] or {}
        d = per.get(L) if L in per else per.get(str(L))
        if not d:
            continue
        x = torch.as_tensor(d["x"]).to(torch.bfloat16)
        xs.append(x.reshape(-1, x.shape[-1]))
        ids.append(torch.as_tensor(d["ids"]).long().reshape(-1, TOPK))
        ws.append(torch.as_tensor(d["w"]).float().reshape(-1, TOPK))
        tags.append(tag)
    if not xs:
        return None, [], f"not run: {path} has no rows for layer {L}"
    rows = {"x": torch.cat(xs), "ids": torch.cat(ids), "w": torch.cat(ws), "tags": tags}
    wins = load_windows(acts, L, 8, 1)
    return rows, wins, {"tags": tags, "rows": int(rows["x"].shape[0]), "windows": len(wins), "meta": acts.get("meta")}


def build_cases(a, setup: LayerSetup, rows, wins) -> List[dict]:  # pragma: no cover - box only
    """Captured cases (T rows of the concatenated fixture rows; the 8 six-row windows) + synthetic c-forcing cases."""
    cases = []
    ts, cs = _ints(a.ts), _ints(a.cs)
    if rows is not None:
        for T in ts:
            if rows["x"].shape[0] >= T:
                cases.append({"name": f"acts-T{T}", "source": "acts", "T": T, "x": rows["x"][:T],
                              "ids": rows["ids"][:T], "w": rows["w"][:T]})
        for wdx, win in enumerate(wins):
            cases.append({"name": f"win{wdx}-{win['tag']}", "source": "acts-window", "T": 6,
                          "x": torch.as_tensor(win["x"]).to(torch.bfloat16).reshape(6, -1),
                          "ids": torch.as_tensor(win["ids"]).long().reshape(6, TOPK),
                          "w": torch.as_tensor(win["w"]).float().reshape(6, TOPK)})
    rng = random.Random(a.seed)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    for T in ts:
        for c in cs:
            ids = synth_routing(T, c, setup.hot, setup.cold, TOPK, rng)
            if ids is None:
                cases.append({"name": f"syn-T{T}-c{c}", "source": "synthetic", "T": T, "c_req": c,
                              "skipped": f"infeasible: c {c} > T x k {T * TOPK}"})
                continue
            x = rows["x"][:T] if rows is not None and rows["x"].shape[0] >= T else synth_x(T, setup.D, g)
            cases.append({"name": f"syn-T{T}-c{c}", "source": "synthetic", "T": T, "c_req": c,
                          "x_source": "acts" if rows is not None and rows["x"].shape[0] >= T else "synthetic",
                          "x": x, "ids": torch.tensor(ids, dtype=torch.long, device="cpu"),
                          "w": torch.tensor(synth_weights(T, TOPK, rng), dtype=torch.float32, device="cpu")})
    for cs_ in cases:
        if "ids" in cs_:
            cs_["c"] = distinct_cold(cs_["ids"].tolist(), setup.row, setup.H)
    return cases


def phase_m3(a, rec: Receipt, setup: LayerSetup, acts):  # pragma: no cover - box only
    rows, wins, meta = acts
    cases = build_cases(a, setup, rows, wins)
    path = os.path.join(os.path.dirname(os.path.abspath(rec.path)), f"gracetier-m3-cases-L{setup.L}.pt")
    torch.save([{k: v for k, v in c.items()} for c in cases], path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    out = {"acts": meta if isinstance(meta, dict) else {"status": meta}, "cases_path": path, "cases_sha256": h,
           "cases": [{k: v for k, v in c.items() if k in ("name", "source", "T", "c_req", "c", "skipped", "x_source")}
                     for c in cases]}
    log(f"[M3] {len([c for c in cases if 'ids' in c])} cases ({sum(c['source'] != 'synthetic' for c in cases)} "
        f"from captured activations; acts: {out['acts'].get('status', 'loaded')}) -> {path}")
    rec.phase("M3", out)
    return cases


# ================================================================================================ M4: (b) numerics
def phase_m4(a, rec: Receipt, setup: LayerSetup, cases: List[dict]) -> dict:  # pragma: no cover - box only
    res, gate_all = [], True
    for fm in _fast_maths(a):
        for case in cases:
            if "ids" not in case:
                continue
            T = case["T"]
            ids = case["ids"].to(setup.dev)
            groups = setup.groups_of(ids)
            w = case["w"].to(setup.dev)
            q, sc = quant_rows(case["x"].to(setup.dev))
            plan = GS.stage_plan(groups.reshape(-1).tolist(), setup.H, setup.C)
            setup.dirty()
            set_inputs(setup.symm, groups, w, q, sc)
            setup.cnt.zero_()
            setup.stage(setup.symm, T)
            _sync()
            cnt_ok = all(int(setup.cnt[c]) == n for c, n in plan.items()) and int(setup.cnt.sum()) == sum(plan.values())
            staged_ok = all(torch.equal(setup.l1[setup.H + c], setup.ref_l1[setup.H + c])
                            and torch.equal(setup.l2[setup.H + c], setup.ref_l2[setup.H + c]) for c in plan)
            y_vmm = setup.mega_vmm(fm)[:T].clone()
            set_inputs(setup.symm, groups, w, q, sc)
            y_ref = setup.mega_ref(fm)[:T].clone()
            r = {"case": case["name"], "T": T, "c": case.get("c"), "fast_math": fm, "cnt_ok": cnt_ok,
                 "staged_bytes_equal_ref": staged_ok, "vs_ref": compare_bits(y_vmm, y_ref)}
            if setup.st_l1 is not None:
                set_inputs(setup.symm, ids, w, q, sc)
                y_st = setup.mega_stock(fm)[:T].clone()
                r["vs_stock_order"] = compare_bits(y_vmm, y_st)
            _sync()
            gate_all &= bool(r["vs_ref"]["equal"])
            res.append(r)
            log(f"[M4] fm={int(fm)} {case['name']:>18s} T={T:<2d} c={case.get('c')!s:>2s}: vs ref "
                f"{r['vs_ref']['equal']} (rows exact {r['vs_ref']['rows_exact']}/{T}, frac eq "
                f"{r['vs_ref']['frac_bytes_equal']}); vs stock order {(r.get('vs_stock_order') or {}).get('equal')}; "
                f"cnt {cnt_ok}; staged bytes {staged_ok}")
    by_t = {str(T): all(r["vs_ref"]["equal"] for r in res if r["T"] == T) if any(r["T"] == T for r in res)
            else NOT_MEASURED for T in _ints(a.ts)}
    out = {"cases": res, "n": len(res), "equal_by_T": by_t, "gate": bool(res) and gate_all,
           "vs_stock_order_all_equal": all((r.get("vs_stock_order") or {}).get("equal", False) for r in res)
           if res and "vs_stock_order" in res[0] else NOT_MEASURED,
           "condition": f"SymmBuffer T_max {a.max_tokens}; a T-row call fills rows [:T]; kNumSMs = deep_gemm default"}
    log(f"[M4] GATE torch.equal at all T = {out['gate']} ({out['n']} calls; by T {by_t}); stock-order arm "
        f"all equal = {out['vs_stock_order_all_equal']}")
    rec.phase("M4", out)
    return out


# ================================================================================================ M5 / M6: timing
def phase_m5(a, rec: Receipt, setup: LayerSetup, cases: List[dict]) -> dict:  # pragma: no cover - box only
    fm = True                 # Garner's shipped MegaMoE call (B/:1371-1376: fast_math=True), kNumSMs 144
    table: List[dict] = []
    notes: List[dict] = []
    for case in cases:
        if case.get("source") != "synthetic" or "ids" not in case:
            continue
        T, c = case["T"], case["c"]
        groups = setup.groups_of(case["ids"])
        w = case["w"].to(setup.dev)
        q, sc = quant_rows(case["x"].to(setup.dev))
        set_inputs(setup.symm, groups, w, q, sc)
        setup.stage(setup.symm, T)
        _sync()
        row = {"T": T, "c": c}
        row["stage_us"] = _us(lambda: setup.stage(setup.symm, T), a.iters, notes, f"stage T{T} c{c}")
        row["mega384_us"] = _us(lambda: setup.mega_vmm(fm, MEGA_SMS), a.iters, notes, f"mega384 T{T} c{c}")
        row["both_us"] = _us(lambda: (setup.stage(setup.symm, T), setup.mega_vmm(fm, MEGA_SMS)),
                             a.iters, notes, f"both T{T} c{c}")
        if c == 0:
            set_inputs(setup.symm_h, groups, w, q, sc)
            row["mega255_us"] = _us(lambda: setup.mega_hot(fm, MEGA_SMS), a.iters, notes, f"mega255 T{T}")
        table.append(row)
        log(f"[M5] T={T:<2d} c={c:<2d}: stage {row['stage_us']} us, MegaMoE G=384 {row['mega384_us']} us, both "
            f"{row['both_us']} us" + (f", G={setup.H} {row['mega255_us']} us" if c == 0 else ""))
    summ = m5_summary(table, _ints(a.ts))
    out = {"table": table, "summary": summ, "b_serial_us_T6": b_serial_from_m5(table, 6), "fast_math": fm,
           "num_sms": MEGA_SMS, "iters": a.iters, "how": "CUDA graph replay (bench/dg_contiguous_spike.graph_us)",
           "timing_fallbacks": notes}
    for T, s in summ.items():
        log(f"[M5] T={T}: stage slope {s['stage_slope_us']} us/expert (bar <= {STAGE_BAR_US}: {s['bar_pass']}), "
            f"MegaMoE marginal {s['mega_marginal_us']} us/expert")
    rec.phase("M5", out)
    return out


def phase_m6(a, rec: Receipt, setup: LayerSetup, cases: List[dict]) -> dict:  # pragma: no cover - box only
    from runtime import dg_mega as DM

    fm = True
    notes: List[dict] = []
    out: dict = {"fast_math": fm, "num_sms": MEGA_SMS, "G_cold": setup.C, "timing_fallbacks": notes}
    rec.phase("M6", dict(out, uva={"status": "started (a crash here means TMA over UVA faulted)"}))
    try:
        symm_c = DM.make_symm_buffer(setup.dg, setup.C, a.max_tokens, setup.k, setup.D, setup.I, 0)
        y_c = torch.zeros(a.max_tokens, setup.D, dtype=torch.bfloat16, device=setup.dev)
        uva_l1, uva_l2 = setup.pool.cuda_l1(setup.L), setup.pool.cuda_l2(setup.L)
        hbm_l1, hbm_l2 = setup.pool.host_l1(setup.L).to(setup.dev), setup.pool.host_l2(setup.L).to(setup.dev)
    except Exception as e:  # noqa: BLE001
        out["uva"] = {"status": f"error (set-up of the G = {setup.C} module): {type(e).__name__}: {str(e)[:300]}"}
        log(f"[M6] {out['uva']['status']}")
        return rec.phase("M6", out)

    def cold(l1, l2, nsms=MEGA_SMS):
        return mega_call(setup.dg, y_c, l1, setup.sf1_cold, l2, setup.sf2_cold, symm_c, setup.lim, fm, nsms)

    def split(groups):
        tc = torch.where(groups >= setup.H, groups - setup.H, torch.full_like(groups, -1))
        th = torch.where(groups < setup.H, groups, torch.full_like(groups, -1))
        return tc, th

    syn = [c for c in cases if c.get("source") == "synthetic" and "ids" in c]
    probe = next((c for c in syn if c["T"] == 6 and c["c"] >= 1), syn[0] if syn else None)
    uva: dict = {"status": "not run: no case"}
    if probe is not None:
        T = probe["T"]
        tc, _ = split(setup.groups_of(probe["ids"]))
        w = probe["w"].to(setup.dev)
        q, sc = quant_rows(probe["x"].to(setup.dev))
        try:
            set_inputs(symm_c, tc, w, q, sc)
            y_uva = cold(uva_l1, uva_l2)[:T].clone()
            _sync()
            set_inputs(symm_c, tc, w, q, sc)
            y_hbm = cold(hbm_l1, hbm_l2)[:T].clone()
            _sync()
            cmp = compare_bits(y_uva, y_hbm)
            uva = {"status": "runs, equal" if cmp["equal"] else "runs, DIFFERS", "case": probe["name"], "cmp": cmp}
        except Exception as e:  # noqa: BLE001
            uva = {"status": f"error: {type(e).__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()[-1500:]}
    out["uva"] = uva
    rec.phase("M6", out)
    log(f"[M6] TMA over UVA: {uva['status']}")
    if not str(uva["status"]).startswith("runs"):
        return out
    flips, table = [], []
    for case in [c for c in cases if "ids" in c]:
        T = case["T"]
        groups = setup.groups_of(case["ids"])
        tc, th = split(groups)
        w = case["w"].to(setup.dev)
        q, sc = quant_rows(case["x"].to(setup.dev))
        set_inputs(setup.symm_h, th, w, q, sc)
        y_hot = setup.mega_hot(fm, MEGA_SMS)[:T].clone()
        set_inputs(symm_c, tc, w, q, sc)
        y_cold = cold(uva_l1, uva_l2)[:T].clone()
        y_a = (y_hot.float() + y_cold.float()).to(torch.bfloat16)
        set_inputs(setup.symm, groups, w, q, sc)
        y_ref = setup.mega_ref(fm, MEGA_SMS)[:T].clone()
        _sync()
        flips.append({"case": case["name"], "T": T, "c": case.get("c"), "vs_ref": compare_bits(y_a, y_ref)})
        if case.get("source") == "synthetic":
            set_inputs(symm_c, tc, w, q, sc)
            r = {"T": T, "c": case["c"], "cold_uva_us": _us(lambda: cold(uva_l1, uva_l2), a.iters, notes, f"cold T{T}")}
            set_inputs(setup.symm_h, th, w, q, sc)
            r["hot255_us"] = _us(lambda: setup.mega_hot(fm, MEGA_SMS), a.iters, notes, f"hot T{T}")
            table.append(r)
            log(f"[M6] T={T:<2d} c={case['c']:<2d}: cold MegaMoE over UVA {r['cold_uva_us']} us, hot G={setup.H} "
                f"{r['hot255_us']} us; flip class vs ref: {flips[-1]['vs_ref']}")
    fits = {}
    for T in _ints(a.ts):
        rs = [r for r in table if r["T"] == T]
        f = fit_line([r["c"] for r in rs], [r["cold_uva_us"] for r in rs])
        fits[str(T)] = {"cold_uva": f, "us_per_cold_expert": None if f is None else f["slope"],
                        "intercept_us": None if f is None else f["intercept"]}
    mx = [x["vs_ref"]["max_abs"] for x in flips if x["vs_ref"]["max_abs"] is not None]
    out.update(flips=flips, table=table, fits=fits,
               flip_class={"calls": len(flips), "all_equal": all(x["vs_ref"]["equal"] for x in flips),
                           "rows_exact": sum(x["vs_ref"]["rows_exact"] for x in flips),
                           "rows": sum(x["vs_ref"]["rows"] for x in flips),
                           "min_frac_bytes_equal": min((x["vs_ref"]["frac_bytes_equal"] for x in flips), default=None),
                           "max_abs": max(mx) if mx else None})
    log(f"[M6] flip class bf16(y_hot)+bf16(y_cold) vs reference: {out['flip_class']}; us per cold expert by T "
        f"{ {T: v['us_per_cold_expert'] for T, v in fits.items()} }")
    rec.phase("M6", out)
    return out


# ================================================================================================ M7: R8
def phase_m7(a, rec: Receipt, setup: LayerSetup, cases: List[dict], timing: Optional[dict]) -> dict:  # pragma: no cover
    dev = setup.dev
    notes: List[dict] = []
    out: dict = {"programs": _ints(a.programs), "mega_sms": MEGA_SMS, "T": 6, "timing_fallbacks": notes,
                 "placement": "[A] one program = one CTA; the hardware scheduler places CTAs (no SM pinning)"}
    # fork/join + empty launch (the design's 3.34 us input, measured here)
    side = torch.cuda.Stream(device=dev)
    touch = torch.zeros(4, dtype=torch.int32, device=dev)

    def fj():
        cur = torch.cuda.current_stream(dev)
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            GS.gt_touch(touch)
        cur.wait_stream(side)

    out["fork_join_empty_us"] = _us(fj, a.iters, notes, "fork_join")
    out["empty_launch_us"] = _us(lambda: GS.gt_touch(touch), a.iters, notes, "empty")
    case = next((c for c in cases if c.get("source") == "synthetic" and c["T"] == 6 and c.get("c") == 8), None)
    if case is None:
        out["status"] = "not run: no T = 6, c = 8 synthetic case"
        return rec.phase("M7", out)
    groups = setup.groups_of(case["ids"])
    w = case["w"].to(dev)
    q, sc = quant_rows(case["x"].to(dev))
    th = torch.where(groups < setup.H, groups, torch.full_like(groups, -1))
    set_inputs(setup.symm_h, th, w, q, sc)
    set_inputs(setup.symm, groups, w, q, sc)
    stage_groups = setup.symm.topk_idx[:6].reshape(-1).clone()
    n_staged = len(GS.stage_plan(stage_groups.tolist(), setup.H, setup.C))
    bw = GS.BLOCK_WORDS_DEFAULT
    N = 40

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def mega_alone_us() -> float:
        a0, a1 = ev(), ev()
        _sync()
        a0.record()
        for _ in range(N):
            setup.mega_hot(True, MEGA_SMS)
        a1.record()
        _sync()
        return a0.elapsed_time(a1) * 1e3 / N

    def concurrent(side_fn) -> dict:
        cur = torch.cuda.current_stream(dev)
        e_f, s0, s1, m0, m1 = ev(), ev(), ev(), ev(), ev()
        _sync()
        e_f.record(cur)
        side.wait_event(e_f)
        with torch.cuda.stream(side):
            s0.record(side)
            side_fn()
            s1.record(side)
        m0.record(cur)
        for _ in range(N):
            setup.mega_hot(True, MEGA_SMS)
        m1.record(cur)
        _sync()
        ts = {k: e_f.elapsed_time(e) * 1e3 for k, e in (("s0", s0), ("s1", s1), ("m0", m0), ("m1", m1))}
        side_us, main_us = ts["s1"] - ts["s0"], ts["m1"] - ts["m0"]
        ov = max(0.0, min(ts["s1"], ts["m1"]) - max(ts["s0"], ts["m0"]))
        return {"side_us": round(side_us, 2), "mega_us_per_call": round(main_us / N, 2),
                "overlap_frac_of_side": round(ov / side_us, 3) if side_us > 0 else None,
                "offsets_us": {k: round(v, 2) for k, v in ts.items()}}

    def median_of(fn, n=3):
        rs = [fn() for _ in range(n)]
        rs.sort(key=lambda r: r["side_us"])
        return rs[len(rs) // 2]

    W = mega_alone_us()
    out["mega_hot_alone_us_per_call_eager"] = round(W, 2)
    rows = []
    for P in _ints(a.programs):
        r: dict = {"programs": P}
        n_src = min(setup.C, 16)
        rd_out = torch.zeros(P * bw, dtype=torch.int64, device=dev)
        rd_bytes = n_src * setup.b1
        r["read_alone_us"] = _us(lambda: GS.gt_read(setup.src_tab, n_src, setup.b1, rd_out, nprog=P),
                                 max(5, a.iters // 10), notes, f"read {P}")
        r["read_alone_GBps"] = gbps(rd_bytes, r["read_alone_us"])
        npg = max(1, P // max(n_staged, 1))
        r["stage_nprog"] = npg
        r["stage_programs_touched"] = npg * n_staged
        st_bytes = n_staged * (setup.b1 + setup.b2)
        r["stage_alone_us"] = _us(lambda: setup.stage(setup.symm, 6, nprog=npg, groups=stage_groups),
                                  max(5, a.iters // 10), notes, f"stage {P}")
        r["stage_alone_GBps"] = gbps(st_bytes, r["stage_alone_us"])
        # concurrent: size the side work to ~0.8 x the N-call MegaMoE window at the alone rate
        win_us = N * W * 0.8
        n_rd = n_src
        if r["read_alone_GBps"]:
            n_rd = int(max(1, min(setup.C, round(r["read_alone_GBps"] * 1e3 * win_us / setup.b1))))
        cr = median_of(lambda: concurrent(lambda: GS.gt_read(setup.src_tab, n_rd, setup.b1, rd_out, nprog=P)))
        cr["bytes"] = n_rd * setup.b1
        cr["GBps"] = gbps(cr["bytes"], cr["side_us"])
        cr["mega_slowdown"] = round(cr["mega_us_per_call"] / W, 3) if W else None
        r["read_concurrent"] = cr
        K = 1
        if r["stage_alone_us"]:
            K = int(max(1, round(win_us / r["stage_alone_us"])))

        def stage_k():
            for _ in range(K):
                setup.stage(setup.symm, 6, nprog=npg, groups=stage_groups)

        cs_ = median_of(lambda: concurrent(stage_k))
        cs_["bytes"] = K * st_bytes
        cs_["launches"] = K
        cs_["GBps"] = gbps(cs_["bytes"], cs_["side_us"])
        cs_["mega_slowdown"] = round(cs_["mega_us_per_call"] / W, 3) if W else None
        r["stage_concurrent"] = cs_
        rows.append(r)
        log(f"[M7] {P:>3d} programs: read alone {r['read_alone_GBps']} GB/s, under MegaMoE {cr['GBps']} GB/s "
            f"(overlap {cr['overlap_frac_of_side']}, MegaMoE x{cr['mega_slowdown']}); gt_stage alone "
            f"{r['stage_alone_GBps']} GB/s, under MegaMoE {cs_['GBps']} GB/s (x{cs_['mega_slowdown']})")
        rec.phase("M7", dict(out, rows=rows))
    out["rows"] = rows
    r8 = next((r for r in rows if r["programs"] == 8), None)
    out["headline"] = {"read_GBps_8_under_mega": None if r8 is None else r8["read_concurrent"]["GBps"],
                       "stage_GBps_8_under_mega": None if r8 is None else r8["stage_concurrent"]["GBps"],
                       "mega_slowdown_read_8": None if r8 is None else r8["read_concurrent"]["mega_slowdown"]}
    out["p11"] = p11_from(timing, out, setup.b1 + setup.b2)
    log(f"[M7] R8 headline {out['headline']}; P1.1 bar: {out['p11'].get('status')}")
    return rec.phase("M7", out)


def p11_from(timing: Optional[dict], m7: dict, weights_b: int) -> dict:
    """The P1.1 bar from a timing receipt's M5 / M6 phases and this M7 (pure; `not measured` when an input is
    missing)."""
    ph = (timing or {}).get("phases", timing or {})
    m5, m6 = ph.get("M5") or {}, ph.get("M6") or {}
    t6 = [r for r in (m5.get("table") or []) if r.get("T") == 6 and r.get("c") == 0]
    w_hot = t6[0].get("mega255_us") if t6 else None
    ci = ((m6.get("fits") or {}).get("6") or {}).get("intercept_us")
    b = {int(k): v for k, v in (m5.get("b_serial_us_T6") or {}).items()}
    rate = (m7.get("headline") or {}).get("read_GBps_8_under_mega")
    res = p11_bar(m7.get("fork_join_empty_us"), ci, rate, w_hot, b, weights_b)
    srate = (m7.get("headline") or {}).get("stage_GBps_8_under_mega")
    res["with_gt_stage_rate"] = p11_bar(m7.get("fork_join_empty_us"), ci, srate, w_hot, b, weights_b)
    return res


# ================================================================================================ subcommands
def _gate_from(path: Optional[str]) -> Optional[bool]:
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    m1 = (d.get("phases") or {}).get("M1")
    return None if not m1 else bool(m1.get("gate"))


def _run_gate(a, rec: Receipt, dg) -> int:  # pragma: no cover - box only
    """M1 from --gate-json, else M0 + M1 here. EXIT_M1 on failure (nothing after M1 may run)."""
    ok = _gate_from(a.gate_json)
    if ok is None:
        phase_m0(a, rec)
        ok = phase_m1(a, rec, dg)["gate"]
    else:
        rec.d["gate_from"] = a.gate_json
    if not ok:
        log("[M1] GATE FAILED: VMM-backed MegaMoE is not torch.equal to the plain allocation. STOP -- fall back to "
            "section 2.2's compaction design (per-layer spare groups). M2..M7 not run.")
        return EXIT_M1
    return EXIT_OK


def cmd_vmm(a, rec: Receipt) -> int:  # pragma: no cover - box only
    dg = _dg()
    phase_m0(a, rec)
    ok = phase_m1(a, rec, dg)["gate"]
    if not ok:
        log("[M1] GATE FAILED: stop (exit 3); fall back to section 2.2's compaction design.")
        return EXIT_M1
    return EXIT_OK


def _setup(a, dg, want_ref: bool, want_stock: bool):  # pragma: no cover - box only
    acts = load_acts(a.acts, a.layer)
    setup = LayerSetup(a, dg, want_ref=want_ref, want_stock=want_stock)
    return setup, acts


def cmd_numerics(a, rec: Receipt) -> int:  # pragma: no cover - box only
    dg = _dg()
    rc = _run_gate(a, rec, dg)
    if rc:
        return rc
    setup, acts = _setup(a, dg, want_ref=True, want_stock=True)
    try:
        phase_m2(a, rec, setup, None if acts[0] is None else acts[0]["x"])
        cases = phase_m3(a, rec, setup, acts)
        m4 = phase_m4(a, rec, setup, cases)
    finally:
        setup.close()
    return EXIT_OK if m4["gate"] else EXIT_M4


def cmd_timing(a, rec: Receipt) -> int:  # pragma: no cover - box only
    dg = _dg()
    rc = _run_gate(a, rec, dg)
    if rc:
        return rc
    setup, acts = _setup(a, dg, want_ref=True, want_stock=False)
    try:
        cases = phase_m3(a, rec, setup, acts)
        phase_m5(a, rec, setup, cases)
        phase_m6(a, rec, setup, cases)
    finally:
        setup.close()
    return EXIT_OK


def cmd_r8(a, rec: Receipt) -> int:  # pragma: no cover - box only
    dg = _dg()
    rc = _run_gate(a, rec, dg)
    if rc:
        return rc
    setup, acts = _setup(a, dg, want_ref=False, want_stock=False)
    timing = json.load(open(a.timing_json)) if a.timing_json and os.path.exists(a.timing_json) else None
    try:
        cases = phase_m3(a, rec, setup, acts)
        phase_m7(a, rec, setup, cases, timing)
    finally:
        setup.close()
    return EXIT_OK


def cmd_all(a, rec: Receipt) -> int:  # pragma: no cover - box only
    base = os.path.splitext(rec.path)[0]
    parts: Dict[str, Optional[dict]] = {}
    rcs: Dict[str, int] = {}
    paths = {s: f"{base}.{s}.json" for s in SUBS}
    for sub in SUBS:
        extra: List[str] = []
        if sub != "vmm":
            extra += ["--gate-json", paths["vmm"]]
        if sub == "r8":
            extra += ["--timing-json", paths["timing"]]
        cmd = [sys.executable, os.path.abspath(__file__)] + child_argv(a, sub, paths[sub], extra)
        log(f"########## {sub} ({time.strftime('%H:%M:%S')}): {' '.join(cmd[2:])}")
        rcs[sub] = subprocess.run(cmd).returncode
        parts[sub] = json.load(open(paths[sub])) if os.path.exists(paths[sub]) else None
        m = merge_children(parts)
        rec.d["phases"], rec.d["errors"] = m["phases"], m["errors"]
        rec.d["children"] = {s: {"rc": rcs[s], "receipt": paths[s]} for s in rcs}
        rec.save()
        if sub == "vmm" and rcs[sub] != EXIT_OK:
            log(f"[all] vmm (M0 / M1) exited {rcs[sub]}: stopping before M2"
                + (" -- the M1 gate failed (exit 3)" if rcs[sub] == EXIT_M1 else ""))
            return rcs[sub]
    if rcs.get("numerics") == EXIT_M4:
        return EXIT_M4
    return EXIT_OK if all(v == EXIT_OK for v in rcs.values()) else EXIT_ERRORS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="P1 gracetier mini-test probe (docs/P1-GRACETIER-DESIGN.md section 5)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in SUBS + ("all",):
        p = sp.add_parser(name)
        p.add_argument("--json", default=None, help="receipt path (default scratch/gracetier-minitest-<date>.json)")
        p.add_argument("--layer", type=int, default=10)
        p.add_argument("--model-dir", default="/ref")
        p.add_argument("--rowmap", default=GP.default_rowmap_path(REPO))
        p.add_argument("--acts", default=os.path.join("scratch", "m3n1-acts.pt"),
                       help="MoE inputs captured by bench/dg_mega_numerics.py downstream (our reference forward)")
        p.add_argument("--garner", default="/garner/fastpath", help="Pg61 fastpath dir (router()); read-only mount")
        p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
        p.add_argument("--iters", type=int, default=200)
        p.add_argument("--seed", type=int, default=20261108)
        p.add_argument("--ts", default=",".join(str(t) for t in TS))
        p.add_argument("--cs", default=",".join(str(c) for c in CS))
        p.add_argument("--programs", default=",".join(str(n) for n in R8_PROGRAMS))
        p.add_argument("--fast-math", default="both", choices=["0", "1", "both"],
                       help="M4's MegaMoE fast_math arms (M1 uses the first; M5-M7 use 1, Garner's shipped call)")
        p.add_argument("--allow-busy", action="store_true", help="run although other compute processes hold the GPU")
        p.add_argument("--gate-json", default=None, help="a vmm receipt whose M1 gate passed (else M0 + M1 run here)")
        p.add_argument("--timing-json", default=None, help="r8: a timing receipt (M5 / M6) for the P1.1 bar")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.json = a.json or default_json()
    rec = Receipt(a.json, a.cmd, a)
    fn = {"vmm": cmd_vmm, "numerics": cmd_numerics, "timing": cmd_timing, "r8": cmd_r8, "all": cmd_all}[a.cmd]
    try:
        rc = fn(a, rec)
    except SystemExit as e:
        rec.error(a.cmd, str(e))
        rc = EXIT_ERRORS
    except Exception:  # noqa: BLE001
        rec.error(a.cmd, traceback.format_exc())
        rc = EXIT_ERRORS
    return rec.done(rc)


if __name__ == "__main__":
    sys.exit(main())
