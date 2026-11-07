#!/usr/bin/env python3
"""first_light.py -- T6 first light: boot the real checkpoint once, then the M0 parity measurements 1-3 (plan section 7)
and one timed step cell.

  python3 bench/first_light.py --model-dir /ref --hot-per-layer 295 \
      --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-cand.jsonl \
      --max-new 256 --reps 2 --step-cell 512 [--map hot.json] [--profile --profile-warm 32 --profile-steps 64]

(a) every fixture prompt, greedy to --max-new, written as parity JSONL: rep 1 -> --out (`cand`), rep 2 -> <out stem>-replay
    .jsonl (`cand-replay`), rep k>2 -> <stem>-replay<k-1>.jsonl;
(b) bench/parity.py on every rep against parity-A (ref) / parity-A2 (ref2); its table is printed;
(c) self-replay: rep 1 vs every later rep must be byte-identical JSONL and have identical fp32-logits digests
    (all logits, every position); per prompt the first divergence is printed;
(d) step cell: one greedy decode of --step-cell steps on the --step-tag prompt (default `prose`): prefill, then N decode
    steps, each timed with CUDA events around `Generator.step` only (the next id stays on the device, no logprob work
    inside the window). Step k = the k-th decode forward (k = 1..N); stats over k >= --stats-from (32) ->
    mean / p50 / p90 ms per step and tokens/s; EOS does not stop the cell (it is recorded);
(e) expert_pool's cold-hit counters: per-layer cold calls for rep 1 and for the step cell, and the cold share of every
    decode step (cold calls / routed-expert calls; 6 routed experts x 40 layers = 240 calls per row-step).
(f) T8 (docs/T8-T9A-DESIGN.md 2.4): `--mode eager|static|graph` selects how every decode step of (a) and (d) runs
    (reference forward / the static step eagerly / its CUDA-graph replay); `--parity-levels` first runs the bit-identity
    levels on --level-steps (32) decode steps x every fixture prompt, fp32-logit sha256 per position, with
    fill_uninitialized_memory off and then on: L0 static-eager == reference-eager, L1 graph == static-eager, L2 graph
    replay twice + a relaunch (fresh StaticStep, new capture) equal, plus fills-on == fills-off; prints the table
    (`parity_levels` in the report). `--levels-ref <earlier report.json>` adds "this process vs that one" rows (the
    cross-process relaunch). `--marker` launches the fixed-name `t8_step_marker` kernel once per decode step.
(g) T9a-1: `--moe-impl dg-masked` (default `ref`) runs every static/graph step's routed experts through DeepGEMM's
    masked grouped GEMM (runtime/dg_moe.py). With --parity-levels: L1/L2 are gated as before; L0 (static-eager vs
    reference-eager) is reported, not gated (not bit-identical by design); an extra ref static-eager run adds the gated
    row "ref static-eager vs reference-eager" and the drift row "dg-masked vs ref static-eager" (first token
    divergence + mean |delta logit| over same-context positions). The report records `generator.moe_impl`,
    `step_cell.moe_impl` and `dg_moe` (deep_gemm path, e8m0 conversion, deterministic flag). `--levels-ref` x-proc rows
    compare runs of the same label: point it at a report made with the same --moe-impl.
(g2) T9a-2: `--moe-impl dg-inplace` reads the hot experts in place (runtime/dg_inplace.py). Parity levels as dg-masked,
    plus one static-eager dg-masked run (same fp8/glue/attn impls) and the drift row "dg-inplace vs dg-masked (same
    other impls)" (reported, not gated; expected 0 or the 1-ulp class). `dg_moe` in the report carries the scale-store
    choice, the HBM it added and `hbm_after_build` (free/total/allocated after the StaticStep build).
(h) T9b-1: `--fp8-impl gemv` (default `ref`) runs every static/graph step's fp8 Linear layers through the Triton MXFP8
    GEMV (runtime/gemv_mxfp8.py). It combines with --moe-impl: the parity levels treat any non-ref impl the same way
    (L0 reported not gated; one extra static-eager run with BOTH impls `ref` gives the gated T8 row and the drift row,
    labelled e.g. "dg-masked+gemv vs ref static-eager"). The report records `generator.fp8_impl`, `step_cell.fp8_impl`
    and `fp8_gemv` (triton version, the per-shape tuned configs, their µs).
(i) T9b-2: `--glue-impl fused` (default `ref`) runs the static/graph step's zero/low-risk glue clusters as fused kernels
    (runtime/fused_glue.py), bit-identical to `ref` by construction. With --parity-levels one more static-eager run
    with glue `ref` and the SAME moe/fp8 impls adds the gated row "fused glue vs ref glue (same moe/fp8 impl)"
    (level G, fp32-logit sha256 per position); L0 stays gated when moe/fp8 are both ref (the fused glue does not make
    the step a variant), and the drift reference ("ref static-eager") runs with glue `ref` too (pure T8). The report
    records `generator.glue_impl`, `step_cell.glue_impl`, `parity_levels.glue_impl` and `fused_glue` (clusters on /
    off and why, the self-check's picked RoPE / hc-sum modes).
(j) T9b-3a: `--attn-impl bounded` (default `ref`) runs the static/graph step's compressed-layer `sparse_attn` through
    runtime/kernels/sparse_attn_bounded.py (the reference TileLang kernel with its 64-index block loop bounded by a
    device per-row count; bit-identical by construction and by its build-time self-check). With --parity-levels one
    more static-eager run with attn `ref` and the SAME moe/fp8/glue impls adds the gated row "bounded attn vs ref
    attn (same moe/fp8/glue impl)" (level A, fp32-logit sha256 per position); like the fused glue it does not make
    the step a variant (L0 stays gated when moe/fp8 are ref), and the drift reference runs with attn `ref`. The
    report records `generator.attn_impl`, `step_cell.attn_impl`, `parity_levels.attn_impl`, `attn_bound` (on/off,
    the TileLang loop spelling the self-check picked, its per-variant results) and `gather` (the staging gather in
    use: 16-B vector or the byte kernel, and why).
(k) T9b-3c: `--attn-impl flashmla` runs EVERY layer's `sparse_attn` of the static/graph step on FlashMLA's
    sparse_prefill_fwd (runtime/attn_flashmla.py; loaded from vllm's .so via torch.ops, no vllm import). A3-class: the
    step is a variant (label gains "flashmla": L0 static rows reported, not gated; the gated "ref static-eager vs
    reference-eager" row and the "<label> vs ref static-eager" drift row as for dg-masked/gemv), and the attn-ref
    static-eager run (same moe/fp8/glue) gives the drift row "flashmla vs ref attn (same other impls)" (level A,
    first token divergence + mean |delta logit|, not gated; `parity_levels.attn_drift`). `attn_bound` in the report
    holds the flashmla describe (on/off and why, the .so, the self-check's error ratios vs the reference).
(l) M1-T10: `--rows 6 --teacher-forced` (docs/HANDOFF-M1-T10-VERIFY6.md 2-4): phase 1 at 1 row -- the fixture greedy
    records (the teacher-forcing targets; parity vs Pg61 as (b)) and a 1-row step cell of 6 x --step-cell steps (its
    tokens feed phase 2's cell; R1); phase 2 at 6 rows -- `Generator.teacher_forced` over every prompt, --reps times
    (each step feeds the 1-row record's next 6 tokens; every row accepted), self-replay as (c), the records compared
    position by position to the 1-row records (V0 at the step level: bit-equal positions, argmax agreement, first
    divergence), and the 6-row step cell (--step-cell steps, stats from --stats-from, CUDA events around `step_rows`
    only, the 6 tokens staged on the device) -> ms per 6-row step, ms per position, R6/R1 (over the same positions),
    HBM after the 6-row build; then the V0 row-invariance census table (runtime/row_census.py). --moe-impl applies to
    both steps (M1-T10-FIX: dg-inplace runs at 6 rows too; --rows-moe-impl overrides the 6-row step's); --glue-impl
    fused at 6 rows runs the row-parallel clusters (shared SwiGLU, mHC post; `[glue]` line of the 6-row step, report
    `rows_mode.fused_glue_rows`) and the reference ops for the one-row clusters. Report key `rows_mode`.
Writes docs/first-light-<YYYYMMDD>.json (or --json) and prints a 12-line summary. One prompt failing logs its
traceback and the run continues.

CUBLAS_WORKSPACE_CONFIG is set here before torch is imported (cuBLAS reads it when it creates its handle).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from runtime.generate import (Generator, dump_record, install_layer_nvtx, record_divergence,  # noqa: E402
                              set_determinism)

PARITY = os.path.join(REPO, "bench", "parity.py")
ATTN_VARIANTS = ("flashmla",)  # attention impls that are not bit-identical (A3-class): drift row, L0 not gated


# ------------------------------------------------------------------------------------------------ helpers
def load_fixture(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def percentile(xs: List[float], q: float) -> Optional[float]:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def rep_path(out: str, rep: int) -> str:
    if rep == 1:
        return out
    stem, ext = os.path.splitext(out)
    return f"{stem}-replay{'' if rep == 2 else rep - 1}{ext or '.jsonl'}"


def _stats(xs: List[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": percentile(xs, 50), "p90": percentile(xs, 90),
            "min": min(xs), "max": max(xs)}


# ------------------------------------------------------------------------------------------------ (a) fixture rep
def run_rep(gen: Generator, rows: List[dict], max_new: int, out_path: str, rep: int, log=print) -> List[dict]:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results = []
    with open(out_path, "w") as f:
        for row in rows:
            tag = row["tag"]
            res: Dict = {"tag": tag, "rep": rep}
            try:
                ids = gen.encode([{"role": "user", "content": row["prompt"]}])
                want = (row.get("usage") or {}).get("prompt_tokens")
                res.update(prompt_tokens=len(ids), pg61_prompt_tokens=want,
                           prompt_tokens_match=None if want is None else len(ids) == want)
                rec = gen.greedy(ids, max_new, tag=tag, prompt=row["prompt"])
                f.write(dump_record(rec) + "\n")
                f.flush()
                res.update({k: v for k, v in gen.last_meta.items()})
                res.update(n_tokens=rec["n_tokens"], finish_reason=rec["finish_reason"])
                dm = res.get("decode_ms_per_token")
                log(f"[rep{rep}] {tag:6s} prompt {len(ids)} (pg61 {want}) -> {rec['n_tokens']} tok "
                    f"{rec['finish_reason']}, prefill {res['prefill_s'] * 1e3:.0f} ms, "
                    f"decode {'-' if dm is None else f'{dm:.1f}'} ms/tok, "
                    f"cold {res.get('cold', {}).get('cold_share', float('nan')):.3f}", flush=True)
            except Exception:
                tb = traceback.format_exc()
                print(f"[rep{rep}] {tag}: FAILED\n{tb}", file=sys.stderr, flush=True)
                res["error"] = tb
            results.append(res)
    return results


# ------------------------------------------------------------------------------------------------ (b) parity
def run_parity(cand: str, ref: str, ref2: Optional[str], json_out: str, log=print) -> dict:
    cmd = [sys.executable, PARITY, "--ref", ref, "--cand", cand, "--json", json_out]
    if ref2:
        cmd += ["--ref2", ref2]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return {"error": tb}
    log(p.stdout.rstrip())
    if p.returncode not in (0, 1):
        print(p.stderr, file=sys.stderr)
    out = {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:]}
    try:
        with open(json_out) as f:
            out["per_prompt"] = json.load(f)
    except Exception as e:
        out["per_prompt_error"] = repr(e)
    last = [l for l in p.stdout.splitlines() if l.startswith("identical")]
    out["verdict_line"] = last[-1] if last else None
    return out


# ------------------------------------------------------------------------------------------------ (c) self-replay
def self_replay(path_a: str, path_b: str, metas_a: List[dict], metas_b: List[dict]) -> dict:
    def load(p):
        try:
            with open(p) as f:
                return {json.loads(l)["tag"]: json.loads(l) for l in f if l.strip()}
        except FileNotFoundError:
            return {}

    a, b = load(path_a), load(path_b)
    da = {m["tag"]: m.get("logits_sha256") for m in metas_a}
    db = {m["tag"]: m.get("logits_sha256") for m in metas_b}
    per = {}
    for tag in sorted(set(a) | set(b) | set(da) | set(db)):
        if tag not in a or tag not in b:
            per[tag] = {"identical": False, "missing": [x for x, d in (("a", a), ("b", b)) if tag not in d]}
            continue
        div = record_divergence(a[tag], b[tag])
        per[tag] = {"identical": div is None, "first_divergence": div,
                    "logits_digest_equal": da.get(tag) is not None and da.get(tag) == db.get(tag)}
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        files_equal = fa.read() == fb.read()
    n_id = sum(1 for v in per.values() if v["identical"])
    n_dig = sum(1 for v in per.values() if v.get("logits_digest_equal"))
    return {"a": path_a, "b": path_b, "files_byte_identical": files_equal, "identical": n_id,
            "logits_digest_equal": n_dig, "n": len(per), "per_prompt": per}


# ------------------------------------------------------------------------------------------------ (d)+(e) step cell
@torch.inference_mode()
def step_cell(gen: Generator, ids: List[int], n_steps: int, stats_from: int = 32) -> dict:
    """Prefill, then `n_steps` decode steps timed one by one (CUDA events around `gen.step`; wall clock off-GPU).
    The fed token stays on the device. Cold-hit deltas per step from expert_pool's host counters (MoE.forward syncs
    to the host every layer, model.py:896, so they are current when `step` returns)."""
    cuda = gen.is_cuda
    L = len(ids)
    if L + n_steps + 1 > gen.max_seq_len:
        raise ValueError(f"step cell needs {L + n_steps + 1} positions > max_seq_len {gen.max_seq_len}")
    if L + n_steps - 1 > gen.max_pos:
        raise ValueError(f"step cell's last position {L + n_steps - 1} > max_pos {gen.max_pos} ({gen.mode} mode)")
    t0 = time.perf_counter()
    if hasattr(gen, "_ensure_captured"):
        gen._ensure_captured()  # graph mode: capture outside the prefill timing
        t0 = time.perf_counter()
    gen.prefill(ids)
    if cuda:
        torch.cuda.synchronize(gen.device)
    prefill_s = time.perf_counter() - t0
    toks = [gen.next_token_tensor().reshape(-1)[:1].clone()]
    c_start = gen.cold_totals()
    evs, wall_ms, cold_steps = [], [], []
    pos = L
    t_cell = time.perf_counter()
    for _ in range(n_steps):
        c0 = gen.cold_totals()
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
        tw = time.perf_counter()
        gen.step(toks[-1], pos)
        if cuda:
            e.record()
            evs.append((s, e))
        else:
            wall_ms.append((time.perf_counter() - tw) * 1e3)
        pos += 1
        toks.append(gen.next_token_tensor().reshape(-1)[:1].clone())
        c1 = gen.cold_totals()
        if c0 is not None and c1 is not None:
            hot, cold = c1[0] - c0[0], c1[1] - c0[1]
            cold_steps.append((hot, cold))
    if cuda:
        torch.cuda.synchronize(gen.device)
    cell_s = time.perf_counter() - t_cell
    ms = [s.elapsed_time(e) for s, e in evs] if cuda else wall_ms
    token_ids = torch.cat(toks).tolist()
    window = ms[stats_from - 1:] if stats_from >= 1 else ms  # step k is ms[k-1]
    st = _stats(window)
    out = {"prompt_tokens": L, "n_steps": n_steps, "stats_from_step": stats_from, "timer": "cuda_events" if cuda else "wall",
           "prefill_ms": prefill_s * 1e3, "cell_wall_s": cell_s, "ms_per_step": st,
           "tokens_per_s_from_mean": (1e3 / st["mean"]) if st.get("mean") else None,
           "tokens_per_s_wall_all_steps": n_steps / cell_s if cell_s > 0 else None,
           "step_ms": [round(x, 4) for x in ms], "token_ids": token_ids,
           "eos_positions": [i for i, t in enumerate(token_ids) if t in set(gen.eos_ids)]}
    if cold_steps:
        shares = [c / (h + c) if h + c else 0.0 for h, c in cold_steps]
        colds = [float(c) for _, c in cold_steps]
        c_end = gen.cold_totals()
        out["cold"] = {"cold_calls_per_step": _stats(colds), "cold_share_per_step": _stats(shares),
                       "routed_calls_per_step": _stats([float(h + c) for h, c in cold_steps]),
                       "cold_calls_per_layer": [b - a for a, b in zip(c_start[3], c_end[3])],
                       "cold_calls_by_step": [c for _, c in cold_steps]}
    return out


# ------------------------------------------------------------------------------------------------ (l) M1-T10 rows
ROWS_MOE: Dict[str, str] = {}  # 1-row MoE impls not ported to rows > 1 -> what the R-row step runs (M1-T10-FIX: none)


def rows_moe_impl(moe: str, rows_moe: str = "auto") -> str:
    return ROWS_MOE.get(moe, moe) if rows_moe == "auto" else rows_moe


def _hbm(gen) -> Optional[dict]:
    if not gen.is_cuda:
        return None
    torch.cuda.synchronize(gen.device)
    free, total = torch.cuda.mem_get_info(gen.device)
    return {"allocated_GiB": round(torch.cuda.memory_allocated(gen.device) / 2**30, 3),
            "max_allocated_GiB": round(torch.cuda.max_memory_allocated(gen.device) / 2**30, 3),
            "free_GiB": round(free / 2**30, 3), "total_GiB": round(total / 2**30, 3)}


@torch.inference_mode()
def step_cell_rows(gen: Generator, ids: List[int], targets: List[int], n_steps: int, stats_from: int = 32) -> dict:
    """The R-row teacher-forced step cell (R = gen.rows): prefill, then `n_steps` R-row steps, step s feeding
    targets[R*s .. R*s+R-1] at positions L+R*s.. (staged on the device before the cell: no H2D copy inside the timed
    window), each timed with CUDA events around `gen.step_rows` only. Row j's argmax is compared (after the cell) with
    targets[R*s+j+1] -- with the 1-row cell's tokens as targets, that is the 6-row graph's agreement with greedy."""
    R_ = gen.rows
    cuda = gen.is_cuda
    L = len(ids)
    need = R_ * n_steps
    if len(targets) < need + 1:
        raise ValueError(f"{len(targets)} targets < {need + 1} for {n_steps} {R_}-row steps")
    if L + need - 1 > gen.max_pos:
        raise ValueError(f"step cell's last position {L + need - 1} > max_pos {gen.max_pos}")
    gen._ensure_captured()
    hbm = _hbm(gen)
    t0 = time.perf_counter()
    gen.prefill(ids)
    if cuda:
        torch.cuda.synchronize(gen.device)
    prefill_s = time.perf_counter() - t0
    feed = torch.tensor(targets[:need], dtype=torch.long, device=gen.device).view(n_steps, R_)
    outs, evs, wall_ms = [], [], []
    pos = L
    t_cell = time.perf_counter()
    for s_ in range(n_steps):
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
        tw = time.perf_counter()
        gen.step_rows(feed[s_], pos)
        if cuda:
            e.record()
            evs.append((s, e))
        else:
            wall_ms.append((time.perf_counter() - tw) * 1e3)
        outs.append(gen.next_token_tensor().reshape(-1).clone())
        pos += R_
    if cuda:
        torch.cuda.synchronize(gen.device)
    cell_s = time.perf_counter() - t_cell
    ms = [s.elapsed_time(e) for s, e in evs] if cuda else wall_ms
    own = torch.stack(outs).reshape(-1).tolist()
    agree = [o == t for o, t in zip(own, targets[1:need + 1])]
    window = ms[stats_from - 1:] if stats_from >= 1 else ms
    st = _stats(window)
    return {"rows": R_, "prompt_tokens": L, "n_steps": n_steps, "positions": need, "stats_from_step": stats_from,
            "timer": "cuda_events" if cuda else "wall", "prefill_ms": prefill_s * 1e3, "cell_wall_s": cell_s,
            "ms_per_step": st, "ms_per_position_mean": st["mean"] / R_ if st.get("mean") else None,
            "positions_per_s_from_mean": (1e3 * R_ / st["mean"]) if st.get("mean") else None,
            "step_ms": [round(x, 4) for x in ms], "argmax_agree": sum(agree), "argmax_compared": len(agree),
            "first_disagreement": next((i for i, a in enumerate(agree) if not a), None), "hbm_after_build": hbm}


def compare_to_r1(tf: dict, r1: dict) -> dict:
    """A teacher-forced record vs the 1-row record that gave its targets: positions whose logprob entry is identical
    (the chosen token's logprob and the top-k list, repr-exact floats), max |dlogprob| of the chosen token, the first
    divergence (record_divergence)."""
    la, lb = tf.get("logprobs") or [], r1.get("logprobs") or []
    n = min(len(la), len(lb))
    same = sum(1 for i in range(n) if la[i] == lb[i])
    d = [abs(float(la[i]["logprob"]) - float(lb[i]["logprob"])) for i in range(n)]
    return {"positions": n, "identical_positions": same, "max_abs_dlogprob_chosen": max(d) if d else 0.0,
            "mean_abs_dlogprob_chosen": sum(d) / n if n else 0.0, "first_divergence": record_divergence(tf, r1)}


def run_rep_tf(gen: Generator, rows: List[dict], targets: Dict[str, list], out_path: str, rep: int,
               log=print) -> List[dict]:
    """(a) at R rows: `gen.teacher_forced` per fixture prompt on the 1-row record's tokens -> JSONL at out_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results = []
    with open(out_path, "w") as f:
        for row in rows:
            tag = row["tag"]
            res: Dict = {"tag": tag, "rep": rep}
            try:
                if tag not in targets:
                    raise RuntimeError(f"{tag}: no 1-row record (phase 1 failed for it)")
                ids = gen.encode([{"role": "user", "content": row["prompt"]}])
                rec = gen.teacher_forced(ids, targets[tag], tag=tag, prompt=row["prompt"])
                f.write(dump_record(rec) + "\n")
                f.flush()
                res.update({k: v for k, v in gen.last_meta.items() if k not in ("token_ids",)})
                res.update(n_tokens=rec["n_tokens"], prompt_tokens=len(ids))
                log(f"[tf{gen.rows} rep{rep}] {tag:6s} prompt {len(ids)} -> {rec['n_tokens']} positions in "
                    f"{res['n_steps']} steps, {res['decode_ms_per_step']:.1f} ms/step (incl. logprob work), argmax "
                    f"agree {res['argmax_agree']}/{rec['n_tokens']}", flush=True)
            except Exception:
                tb = traceback.format_exc()
                print(f"[tf rep{rep}] {tag}: FAILED\n{tb}", file=sys.stderr, flush=True)
                res["error"] = tb
            results.append(res)
    return results


def measure_rows(gen: Generator, a, report: Dict, *, out: str, json_path: str, fixture2: Optional[str]) -> int:
    """(l): phase 1 at 1 row, phase 2 at --rows rows, the census; report `rows_mode`, the 12-line rows summary."""
    R_ = int(a.rows)
    rm: Dict = {"rows": R_, "teacher_forced": True, "moe_impl_r1": gen.moe_impl, "hidden_layers": gen.hidden_layers,
                "moe_impl_rows": rows_moe_impl(gen.moe_impl, a.rows_moe_impl)}
    report["rows_mode"] = rm
    rows = load_fixture(a.fixture)
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    try:
        gen.enc
        report.setdefault("generator", {})["encoding"] = gen.encoding_path
    except Exception:
        report["errors"].append({"stage": "encoding", "traceback": traceback.format_exc()})
    stem, ext = os.path.splitext(out)
    step_row = next((r for r in load_fixture(a.fixture) if r["tag"] == a.step_tag), None)

    # phase 1: 1 row
    gen.set_rows(1)
    p1 = f"{stem}-r1{ext or '.jsonl'}"
    t = time.time()
    res1 = run_rep(gen, rows, a.max_new, p1, 1)
    rm["r1"] = {"path": p1, "wall_s": time.time() - t, "prompts": res1}
    report["errors"] += [{"stage": "r1", "tag": r["tag"], "traceback": r["error"]} for r in res1 if "error" in r]
    if not a.skip_parity:
        rm["parity_r1"] = run_parity(p1, a.fixture, fixture2, p1 + ".parity.json")
    targets = {r["tag"]: r["token_ids"] for r in res1 if "token_ids" in r}
    sc1 = None
    if a.step_cell > 0 and step_row is not None:
        try:
            ids = gen.encode([{"role": "user", "content": step_row["prompt"]}])
            sc1 = step_cell(gen, ids, R_ * a.step_cell, a.stats_from)
            ms = sc1.pop("step_ms")
            same = ms[R_ * a.stats_from - 1:]  # the 1-row steps at the 6-row window's positions
            sc1["ms_per_step_same_positions"] = _stats(same)
            sc1["ms_per_step_first_n"] = _stats(ms[a.stats_from - 1: a.step_cell])
            rm["step_cell_r1"] = {k: v for k, v in sc1.items() if k != "token_ids"}
        except Exception:
            tb = traceback.format_exc()
            print(f"[step-cell r1] FAILED\n{tb}", file=sys.stderr)
            report["errors"].append({"stage": "step_cell_r1", "traceback": tb})
    rm["hbm_1row"] = _hbm(gen)
    # the 1-row cell again with the R-row step's MoE impl when it differs (dg-inplace -> dg-masked): R6/R1 at the same
    # MoE impl separates "6 rows" from "staging copies of every routed expert" (dg-masked gathers, dg-inplace does not)
    if sc1 is not None and rm["moe_impl_rows"] != gen.moe_impl:
        try:
            gen.set_moe_impl(rm["moe_impl_rows"])
            ids = gen.encode([{"role": "user", "content": step_row["prompt"]}])
            scm = step_cell(gen, ids, a.step_cell, a.stats_from)
            scm.pop("token_ids", None)
            scm.pop("step_ms", None)
            rm["step_cell_r1_rows_moe"] = scm
        except Exception:
            tb = traceback.format_exc()
            print(f"[step-cell r1 {rm['moe_impl_rows']}] FAILED\n{tb}", file=sys.stderr)
            report["errors"].append({"stage": "step_cell_r1_rows_moe", "traceback": tb})

    # phase 2: R rows, teacher-forced
    gen.reset_static()
    gen.set_moe_impl(rm["moe_impl_rows"])
    gen.set_rows(R_)
    reps: Dict[int, dict] = {}
    for rep in range(1, a.reps + 1):
        p = f"{stem}-tf{R_}{'' if rep == 1 else f'-replay{rep - 1}'}{ext or '.jsonl'}"
        t = time.time()
        res = run_rep_tf(gen, rows, targets, p, rep)
        reps[rep] = {"path": p, "wall_s": time.time() - t, "prompts": res}
        report["errors"] += [{"stage": f"tf rep{rep}", "tag": r["tag"], "traceback": r["error"]}
                             for r in res if "error" in r]
    rm["reps"] = reps
    st = getattr(gen, "static", None)
    if st is not None:
        rm["static"] = st.describe() if hasattr(st, "describe") else None
        rm["glue_note"] = getattr(st, "glue_note", None)
        glue = getattr(st, "glue", None)
        if glue is not None and hasattr(glue, "describe"):  # M1-T10-FIX: the 6-row step's own clusters
            rm["fused_glue_rows"] = glue.describe()
            rm["glue_rows_on"] = rm["fused_glue_rows"].get("on")
            if hasattr(glue, "glue_line"):
                print(glue.glue_line(), flush=True)
        dg = getattr(st, "dg", None)
        if dg is not None and hasattr(dg, "describe"):
            try:
                rm["dg_rows"] = dg.describe()
            except Exception:
                report["errors"].append({"stage": "dg_rows_describe", "traceback": traceback.format_exc()})
    if 1 in reps and not a.skip_parity:
        rm["parity_tf"] = run_parity(reps[1]["path"], a.fixture, fixture2, reps[1]["path"] + ".parity.json")
    rm["self_replay"] = {}
    for rep in range(2, a.reps + 1):
        sr = self_replay(reps[1]["path"], reps[rep]["path"], reps[1]["prompts"], reps[rep]["prompts"])
        rm["self_replay"][rep] = sr
        print(f"[self-replay tf rep1 vs rep{rep}] {sr['identical']}/{sr['n']} byte-identical, "
              f"{sr['logits_digest_equal']}/{sr['n']} logits digests equal, files equal {sr['files_byte_identical']}")
    try:
        with open(p1) as f1, open(reps[1]["path"]) as f2:
            r1 = {json.loads(l)["tag"]: json.loads(l) for l in f1 if l.strip()}
            tf = {json.loads(l)["tag"]: json.loads(l) for l in f2 if l.strip()}
        per = {tag: compare_to_r1(tf[tag], r1[tag]) for tag in sorted(set(r1) & set(tf))}
        agree = {r["tag"]: r.get("argmax_agree") for r in reps[1]["prompts"] if "argmax_agree" in r}
        rm["vs_r1"] = {"per_prompt": per, "argmax_agree": agree,
                       "positions": sum(v["positions"] for v in per.values()),
                       "identical_positions": sum(v["identical_positions"] for v in per.values()),
                       "prompts_identical": sum(1 for v in per.values() if v["first_divergence"] is None),
                       "n": len(per)}
        print(f"[tf{R_} vs 1-row records] {rm['vs_r1']['identical_positions']}/{rm['vs_r1']['positions']} positions "
              f"identical, {rm['vs_r1']['prompts_identical']}/{len(per)} prompts byte-identical", flush=True)
    except Exception:
        report["errors"].append({"stage": "vs_r1", "traceback": traceback.format_exc()})
    if sc1 is not None:
        try:
            ids = gen.encode([{"role": "user", "content": step_row["prompt"]}])
            scr = step_cell_rows(gen, ids, sc1["token_ids"], a.step_cell, a.stats_from)
            r6 = scr["ms_per_step"].get("mean")
            r1m = sc1["ms_per_step_same_positions"].get("mean")
            scr["r6_over_r1"] = (r6 / r1m) if r6 and r1m else None
            scr["r6_over_r1_first_n"] = (r6 / sc1["ms_per_step_first_n"]["mean"]
                                         if r6 and sc1["ms_per_step_first_n"].get("mean") else None)
            scm = rm.get("step_cell_r1_rows_moe")
            scr["r6_over_r1_same_moe"] = (r6 / scm["ms_per_step"]["mean"]
                                          if scm and r6 and scm["ms_per_step"].get("mean") else None)
            rm["step_cell_rows"] = scr
            print(f"[step-cell {R_}-row] mean {r6:.3f} ms/step ({scr['ms_per_position_mean']:.3f} ms/position); "
                  f"1-row same positions {r1m:.3f} ms -> R{R_}/R1 {scr['r6_over_r1']:.3f}", flush=True)
        except Exception:
            tb = traceback.format_exc()
            print(f"[step-cell rows] FAILED\n{tb}", file=sys.stderr)
            report["errors"].append({"stage": "step_cell_rows", "traceback": tb})
    rm["hbm_rows"] = _hbm(gen)

    # the V0 census (eager; layer-0 shapes; the GEMV with tuned vs pinned-M=1 configs)
    if not a.skip_census:
        try:
            from runtime.row_census import census, census_table

            recs = census(gen.model, rows=R_)
            rm["v0_census"] = recs
            for line in census_table(recs):
                print(line, flush=True)
        except Exception:
            report["errors"].append({"stage": "v0_census", "traceback": traceback.format_exc()})

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    for line in summary_rows_lines(report, json_path):
        print(line)
    return 0 if not report["errors"] else 2


def summary_rows_lines(r: dict, json_path: str) -> List[str]:
    """Exactly 12 lines (the --rows summary)."""
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    rm = r.get("rows_mode", {})
    g = r.get("generator", {})
    b = r.get("boot", {})
    R_ = rm.get("rows")
    lines = [f"FIRST LIGHT {r['date']} | rows {R_} teacher-forced | mode {g.get('mode')} | moe {rm.get('moe_impl_r1')}"
             f" (rows: {rm.get('moe_impl_rows')}) fp8 {g.get('fp8_impl')} glue {g.get('glue_impl')}"
             f"{(' (rows: ' + (','.join(rm['glue_rows_on']) or 'ref') + ')') if rm.get('glue_rows_on') is not None else ''}"
             f" attn {g.get('attn_impl')} | HBM boot "
             f"{f(b.get('cuda_allocated_GiB'))} GiB | {(r.get('env') or {}).get('device')}"]
    p1 = (rm.get("r1") or {}).get("prompts", [])
    ok1 = [x for x in p1 if "error" not in x]
    lines.append(f"1-row records: {len(ok1)}/{len(p1)} prompts, {sum(x.get('n_tokens', 0) for x in ok1)} tokens; parity "
                 f"vs Pg61: {(rm.get('parity_r1') or {}).get('verdict_line') or 'not run'}"[:200])
    for rep in (1, 2):
        ps = (rm.get("reps") or {}).get(rep, {}).get("prompts", [])
        ok = [x for x in ps if "error" not in x]
        dm = [x["decode_ms_per_step"] for x in ok if x.get("decode_ms_per_step")]
        lines.append(f"tf{R_} rep{rep}: {len(ok)}/{len(ps)} prompts, {sum(x.get('n_tokens', 0) for x in ok)} positions, "
                     f"mean {f(sum(dm) / len(dm) if dm else None, 1)} ms per {R_}-row step (incl. logprob work)")
    lines.append(f"tf{R_} parity vs Pg61: {(rm.get('parity_tf') or {}).get('verdict_line') or 'not run'}"[:200])
    sr = (rm.get("self_replay") or {}).get(2)
    lines.append(f"self-replay tf rep1 vs rep2: {sr['identical']}/{sr['n']} byte-identical, {sr['logits_digest_equal']}/"
                 f"{sr['n']} fp32-logit digests equal" if sr else "self-replay: not run")
    v = rm.get("vs_r1")
    lines.append(f"V0 (tf{R_} vs 1-row records): {v['identical_positions']}/{v['positions']} positions identical, "
                 f"{v['prompts_identical']}/{v['n']} prompts byte-identical; argmax agree "
                 f"{sum(x or 0 for x in v['argmax_agree'].values())}/{v['positions']}" if v else "V0 vs 1-row: not run")
    cen = rm.get("v0_census") or []
    yes = sorted({c["family"] for c in cen if c.get("equal") is True})
    no = sorted({c["family"] for c in cen if c.get("equal") is False})
    lines.append(f"census row-invariant: {', '.join(yes) or '-'}"[:240])
    lines.append(f"census NOT row-invariant: {', '.join(no) or '-'}"[:240])
    s1 = rm.get("step_cell_r1") or {}
    sr_ = rm.get("step_cell_rows") or {}
    m1 = (s1.get("ms_per_step_same_positions") or {})
    lines.append(f"1-row cell: {s1.get('n_steps', '-')} steps, same positions mean {f(m1.get('mean'), 3)} / p50 "
                 f"{f(m1.get('p50'), 3)} ms per step; first-{sr_.get('n_steps', '-')} mean "
                 f"{f((s1.get('ms_per_step_first_n') or {}).get('mean'), 3)} ms")
    ms = sr_.get("ms_per_step") or {}
    lines.append(f"{R_}-row cell: steps {sr_.get('stats_from_step', '-')}..{sr_.get('n_steps', '-')} mean "
                 f"{f(ms.get('mean'), 3)} / p50 {f(ms.get('p50'), 3)} / p90 {f(ms.get('p90'), 3)} ms per step, "
                 f"{f(sr_.get('ms_per_position_mean'), 3)} ms/position; argmax agree {sr_.get('argmax_agree', '-')}/"
                 f"{sr_.get('argmax_compared', '-')}")
    hb = sr_.get("hbm_after_build") or {}
    lines.append(f"R{R_}/R1 = {f(sr_.get('r6_over_r1'), 3)} (same positions; first-n {f(sr_.get('r6_over_r1_first_n'), 3)}; "
                 f"vs 1-row {rm.get('moe_impl_rows')} {f(sr_.get('r6_over_r1_same_moe'), 3)})"
                 f" | HBM after {R_}-row build {f(hb.get('allocated_GiB'))} GiB alloc, {f(hb.get('free_GiB'))} GiB free "
                 f"| errors: {len(r['errors'])} | report {json_path}")
    assert len(lines) == 12, len(lines)
    return lines


# ------------------------------------------------------------------------------------------------ (f) parity levels
def _det_module():
    return getattr(torch.utils, "deterministic", None)


def _level_run(gen: Generator, prompts, steps: int, keep: Optional[Dict[str, list]] = None) -> Dict[str, dict]:
    """Greedy `steps`+1 forwards per prompt -> per-tag digests and tokens. `keep` (a dict) also receives each tag's
    fp32 CPU logits per position (for the dg-vs-ref drift row; not stored in the report)."""
    out = {}
    for tag, ids in prompts:
        sink_prev = getattr(gen, "logits_sink", None)
        if keep is not None:
            store = keep[tag] = []
            gen.logits_sink = lambda i, lc, store=store: store.append(lc.detach().to("cpu", torch.float32).clone())
        try:
            gen.greedy(ids, steps + 1, stop_on_eos=False)
        finally:
            if keep is not None:
                gen.logits_sink = sink_prev
        m = gen.last_meta
        out[tag] = {"sha": m["logits_sha256"], "steps": list(m["step_sha256"]), "tokens": list(m["token_ids"])}
    return out


def logit_drift(a_logits: Dict[str, list], b_logits: Dict[str, list], a_run: Dict[str, dict],
                b_run: Dict[str, dict]) -> dict:
    """Drift of run a vs run b (same prompts): per prompt, the first token divergence (index into the generated ids;
    0 = the prefill's argmax) and |delta logit| over the positions whose context is still identical in both runs
    (every position up to and including the first token divergence; position 0 = prefill, identical by construction,
    is excluded). mean_abs_dlogit = mean over compared decode positions of the vocab-mean |a - b|."""
    per, all_means, mx, first = {}, [], 0.0, None
    for tag in sorted(set(a_logits) & set(b_logits)):
        ta, tb = a_run[tag]["tokens"], b_run[tag]["tokens"]
        la, lb = a_logits[tag], b_logits[tag]
        n = min(len(ta), len(tb), len(la), len(lb))
        div = next((i for i in range(n) if ta[i] != tb[i]), None)
        last = n - 1 if div is None else div
        means = []
        for i in range(1, last + 1):
            d = (la[i] - lb[i]).abs()
            means.append(float(d.mean()))
            mx = max(mx, float(d.max()))
        all_means += means
        per[tag] = {"first_token_divergence": div, "positions_compared": len(means),
                    "mean_abs_dlogit": sum(means) / len(means) if means else 0.0,
                    "max_pos_mean_abs_dlogit": max(means) if means else 0.0}
        if div is not None and first is None:
            first = f"{tag}@{div}"
    return {"first_token_divergence": first, "positions_compared": len(all_means),
            "mean_abs_dlogit": sum(all_means) / len(all_means) if all_means else 0.0, "max_abs_dlogit": mx,
            "prompts_token_identical": sum(1 for v in per.values() if v["first_token_divergence"] is None),
            "per_prompt_drift": per}


def compare_levels(a: Dict[str, dict], b: Dict[str, dict]) -> dict:
    """Per prompt: whole-run digest equal; first differing position (0 = the prefill logits) when not."""
    tags = sorted(set(a) | set(b))
    per, first = {}, None
    for t in tags:
        if t not in a or t not in b:
            per[t] = {"equal": False, "first_step": None, "missing": True}
        else:
            eq = a[t]["sha"] == b[t]["sha"]
            fs = None
            if not eq:
                sa, sb = a[t]["steps"], b[t]["steps"]
                fs = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), min(len(sa), len(sb)))
            per[t] = {"equal": eq, "first_step": fs}
        if not per[t]["equal"] and first is None:
            first = f"{t}@{per[t]['first_step']}"
    n_eq = sum(1 for v in per.values() if v["equal"])
    return {"equal": n_eq, "n": len(tags), "all_equal": n_eq == len(tags) and len(tags) > 0, "first_diff": first,
            "per_prompt": per}


def parity_levels(gen: Generator, rows: List[dict], steps: int = 32, log=print,
                  ref_runs: Optional[Dict[str, Dict[str, dict]]] = None) -> dict:
    """T8 design 2.4 on `steps` decode steps x every row: L0 (static-eager vs reference-eager), L1 (graph vs
    static-eager), L2 (graph replay #2 and a relaunch vs replay #1), each with fill_uninitialized_memory off and on,
    plus fills-on vs fills-off. `ref_runs` (an earlier report's `parity_levels.runs`) adds cross-process rows.
    T9a-1: with `gen.moe_impl != "ref"` every static/graph run uses that impl; L0 is then reported but not gated (the
    DeepGEMM MoE is not bit-identical by design), and one more static-eager run with moe_impl "ref" (fills off) adds
    two rows: "ref static-eager vs reference-eager" (the T8 invariant, gated) and the drift row "<impl> vs ref
    static-eager" (first token divergence + mean |delta logit| over same-context positions, `logit_drift`; reported,
    not gated). T9b-1: `gen.fp8_impl != "ref"` is handled the same way; the variant label joins the non-ref impls
    ("dg-masked", "gemv", "dg-masked+gemv") and the drift reference runs with both impls "ref". Leaves the Generator
    in its original mode and impls with fills restored and, in graph mode, graphs captured fills-off."""
    prompts = [(r["tag"], gen.encode([{"role": "user", "content": r["prompt"]}])) for r in rows]
    det = _det_module()
    fill0 = getattr(det, "fill_uninitialized_memory", None) if det is not None else None
    mode0 = gen.mode
    impl = getattr(gen, "moe_impl", "ref")
    fimpl = getattr(gen, "fp8_impl", "ref")
    gimpl = getattr(gen, "glue_impl", "ref")
    gapprox = gimpl != "ref" and bool(getattr(gen, "glue_approx", False))  # T9b-2 R2: outcome-(ii) clusters may run
    aimpl = getattr(gen, "attn_impl", "ref")
    attn_variant = aimpl in ATTN_VARIANTS  # T9b-3c: an A3-class attention makes the step a variant (bounded does not)
    label = "+".join(x for x in (impl, fimpl, aimpl if attn_variant else "ref") if x != "ref") or "ref"
    variant = label != "ref"
    runs: Dict[str, Dict[str, dict]] = {}
    timings: Dict[str, float] = {}
    fills_list = [False, True] if fill0 is not None else [None]
    sfx0 = "" if fills_list[0] is None else "/fills-off"
    kept: Dict[str, Dict[str, list]] = {}

    def timed(label, fn):
        t = time.time()
        runs[label] = fn()
        timings[label] = round(time.time() - t, 2)
        log(f"[levels] {label}: {timings[label]:.1f} s", flush=True)

    def set_fill(f):
        if det is not None and f is not None:
            det.fill_uninitialized_memory = f

    try:
        for f in fills_list:
            sfx = "" if f is None else ("/fills-on" if f else "/fills-off")
            set_fill(f)
            gen.set_mode("eager")
            timed("eager" + sfx, lambda: _level_run(gen, prompts, steps))
            gen.reset_static()
            gen.set_mode("static")
            keep = kept.setdefault(label, {}) if (variant or gapprox) and sfx == sfx0 else None
            timed("static" + sfx, lambda: _level_run(gen, prompts, steps, keep))
            gen.set_mode("graph")
            gen.capture(fills=bool(f))
            timed("graph1" + sfx, lambda: _level_run(gen, prompts, steps))
            timed("graph2" + sfx, lambda: _level_run(gen, prompts, steps))
            gen.reset_static()  # relaunch: new StaticStep (pointer tables, staging, buffers) + a new capture
            gen.set_mode("graph")
            gen.capture(fills=bool(f))
            timed("relaunch" + sfx, lambda: _level_run(gen, prompts, steps))
        if aimpl != "ref":  # T9b-3a/3c: the same static step with the reference sparse_attn (same moe/fp8/glue impls)
            set_fill(fills_list[0])
            gen.set_attn_impl("ref")
            gen.set_mode("static")
            keep = kept.setdefault("attnref", {}) if attn_variant else None
            timed("static-attnref" + sfx0, lambda: _level_run(gen, prompts, steps, keep))
            gen.set_attn_impl(aimpl)
        if gimpl != "ref":  # T9b-2: the same static step with the reference glue (same moe/fp8 impls)
            set_fill(fills_list[0])
            gen.set_glue_impl("ref")
            gen.set_mode("static")
            keep = kept.setdefault("glueref", {}) if gapprox else None
            timed("static-glueref" + sfx0, lambda: _level_run(gen, prompts, steps, keep))
        if impl == "dg-inplace":  # T9a-2: the same static step with dg-masked (same fp8/glue/attn impls)
            set_fill(fills_list[0])
            if hasattr(gen, "set_glue_impl"):
                gen.set_glue_impl(gimpl)
            gen.set_moe_impl("dg-masked")
            gen.set_mode("static")
            timed("static-dgmasked" + sfx0, lambda: _level_run(gen, prompts, steps, kept.setdefault("dgmasked", {})))
            gen.set_moe_impl(impl)
        if variant:  # the drift reference: the same static step with the bit-identical MoE and fp8 GEMMs
            set_fill(fills_list[0])
            gen.set_moe_impl("ref")
            if hasattr(gen, "set_fp8_impl"):
                gen.set_fp8_impl("ref")
            if hasattr(gen, "set_glue_impl"):
                gen.set_glue_impl("ref")
            if hasattr(gen, "set_attn_impl"):
                gen.set_attn_impl("ref")
            gen.set_mode("static")
            timed("static-ref" + sfx0, lambda: _level_run(gen, prompts, steps, kept.setdefault("ref", {})))
    finally:
        set_fill(fill0)
        if variant:
            gen.set_moe_impl(impl)
            if hasattr(gen, "set_fp8_impl"):
                gen.set_fp8_impl(fimpl)
        if hasattr(gen, "set_glue_impl"):
            gen.set_glue_impl(gimpl)
        if hasattr(gen, "set_attn_impl"):
            gen.set_attn_impl(aimpl)
        gen.set_mode(mode0)
        if mode0 == "graph":
            gen.capture(fills=False)

    gated_l0 = not variant and not gapprox     # an outcome-(ii) glue is a variant of the reference too
    table = []
    for f in fills_list:
        sfx = "" if f is None else ("/fills-on" if f else "/fills-off")
        fl = "-" if f is None else ("on" if f else "off")
        table += [(fl, "L0", "static-eager vs reference-eager", gated_l0,
                   compare_levels(runs["static" + sfx], runs["eager" + sfx])),
                  (fl, "L1", "graph vs static-eager", True, compare_levels(runs["graph1" + sfx], runs["static" + sfx])),
                  (fl, "L2", "graph replay#2 vs replay#1", True,
                   compare_levels(runs["graph2" + sfx], runs["graph1" + sfx])),
                  (fl, "L2", "graph relaunch vs replay#1", True,
                   compare_levels(runs["relaunch" + sfx], runs["graph1" + sfx]))]
    if len(fills_list) == 2:
        for k in ("eager", "static", "graph1"):
            # "eager: fills-on vs fills-off" is the reference's own determinism: reported, not gated
            table.append(("on/off", "L2", f"{k}: fills-on vs fills-off", k != "eager",
                          compare_levels(runs[k + "/fills-on"], runs[k + "/fills-off"])))
    drift = None
    if variant:
        fl0 = "-" if fills_list[0] is None else "off"
        table.append((fl0, "L0", "ref static-eager vs reference-eager", True,
                      compare_levels(runs["static-ref" + sfx0], runs["eager" + sfx0])))
        drift = logit_drift(kept[label], kept["ref"], runs["static" + sfx0], runs["static-ref" + sfx0])
        c = compare_levels(runs["static" + sfx0], runs["static-ref" + sfx0])
        c.update({k: v for k, v in drift.items() if k != "per_prompt_drift"})
        c["per_prompt"] = {t: dict(v, **drift["per_prompt_drift"].get(t, {})) for t, v in c["per_prompt"].items()}
        table.append((fl0, "drift", f"{label} vs ref static-eager", False, c))
    glue_drift = None
    if gimpl != "ref" and not gapprox:
        fl0 = "-" if fills_list[0] is None else "off"
        table.append((fl0, "G", "fused glue vs ref glue (same moe/fp8 impl)", True,
                      compare_levels(runs["static" + sfx0], runs["static-glueref" + sfx0])))
    elif gimpl != "ref":  # T9b-2 R2 --glue-approx: outcome-(ii) clusters may differ -> a drift row, reported, not gated
        fl0 = "-" if fills_list[0] is None else "off"
        glue_drift = logit_drift(kept[label], kept["glueref"], runs["static" + sfx0], runs["static-glueref" + sfx0])
        c = compare_levels(runs["static" + sfx0], runs["static-glueref" + sfx0])
        c.update({k: v for k, v in glue_drift.items() if k != "per_prompt_drift"})
        c["per_prompt"] = {t: dict(v, **glue_drift["per_prompt_drift"].get(t, {})) for t, v in c["per_prompt"].items()}
        table.append((fl0, "G", "fused glue +approx vs ref glue (same impls)", False, c))
    if aimpl != "ref" and not attn_variant:
        fl0 = "-" if fills_list[0] is None else "off"
        table.append((fl0, "A", "bounded attn vs ref attn (same moe/fp8/glue)", True,
                      compare_levels(runs["static" + sfx0], runs["static-attnref" + sfx0])))
    attn_drift = None
    inplace_drift = None
    if impl == "dg-inplace":  # T9a-2: drift vs dg-masked (expected 0 or the 1-ulp class), reported, not gated
        fl0 = "-" if fills_list[0] is None else "off"
        inplace_drift = logit_drift(kept[label], kept["dgmasked"], runs["static" + sfx0],
                                    runs["static-dgmasked" + sfx0])
        c = compare_levels(runs["static" + sfx0], runs["static-dgmasked" + sfx0])
        c.update({k: v for k, v in inplace_drift.items() if k != "per_prompt_drift"})
        c["per_prompt"] = {t: dict(v, **inplace_drift["per_prompt_drift"].get(t, {})) for t, v in c["per_prompt"].items()}
        table.append((fl0, "D", "dg-inplace vs dg-masked (same other impls)", False, c))
    if attn_variant:  # T9b-3c: A3-class -> a drift row (first token divergence + mean |dlogit|), reported, not gated
        fl0 = "-" if fills_list[0] is None else "off"
        attn_drift = logit_drift(kept[label], kept["attnref"], runs["static" + sfx0], runs["static-attnref" + sfx0])
        c = compare_levels(runs["static" + sfx0], runs["static-attnref" + sfx0])
        c.update({k: v for k, v in attn_drift.items() if k != "per_prompt_drift"})
        c["per_prompt"] = {t: dict(v, **attn_drift["per_prompt_drift"].get(t, {})) for t, v in c["per_prompt"].items()}
        table.append((fl0, "A", f"{aimpl} vs ref attn (same other impls)", False, c))
    if ref_runs:
        for k in sorted(set(runs) & set(ref_runs)):
            table.append(("x-proc", "L2", f"{k}: this process vs --levels-ref", True,
                          compare_levels(runs[k], ref_runs[k])))
    rows_out = [{"fills": fl, "level": lv, "what": what, "gate": gate,
                 **{k: v for k, v in c.items() if k != "per_prompt"}, "per_prompt": c["per_prompt"]}
                for fl, lv, what, gate, c in table]
    gate_rows = [r for r in rows_out if r["gate"]]
    verdict = "PASS" if all(r["all_equal"] for r in gate_rows) else "FAIL"
    log(f"moe impl {impl}" + ("" if fimpl == "ref" else f", fp8 impl {fimpl}")
        + ("" if gimpl == "ref" else f", glue impl {gimpl}{'+approx' if gapprox else ''}")
        + ("" if aimpl == "ref" else f", attn impl {aimpl}")
        + ("" if gated_l0 else " (L0 static-eager rows and the drift row are reported, not gated)"))
    log(f"{'fills':7s}{'level':6s}{'comparison':42s}{'equal':>8s}  first diff (tag@position)")
    for r in rows_out:
        log(f"{r['fills']:7s}{r['level']:6s}{r['what']:42s}{r['equal']:>4d}/{r['n']:<3d}  {r['first_diff'] or '-'}"
            + ("" if r["gate"] else "  (not gated)"))
    if drift is not None:
        log(f"drift {label} vs ref static-eager: first token divergence {drift['first_token_divergence'] or 'none'}, "
            f"{drift['prompts_token_identical']}/{len(prompts)} prompts token-identical, mean |dlogit| "
            f"{drift['mean_abs_dlogit']:.3e} (max {drift['max_abs_dlogit']:.3e}) over {drift['positions_compared']} "
            f"same-context positions")
    if glue_drift is not None:
        log(f"drift fused glue +approx vs ref glue: first token divergence "
            f"{glue_drift['first_token_divergence'] or 'none'}, {glue_drift['prompts_token_identical']}/{len(prompts)} "
            f"prompts token-identical, mean |dlogit| {glue_drift['mean_abs_dlogit']:.3e} (max "
            f"{glue_drift['max_abs_dlogit']:.3e}) over {glue_drift['positions_compared']} same-context positions")
    if attn_drift is not None:
        log(f"drift {aimpl} vs ref attn (same other impls): first token divergence "
            f"{attn_drift['first_token_divergence'] or 'none'}, {attn_drift['prompts_token_identical']}/{len(prompts)} "
            f"prompts token-identical, mean |dlogit| {attn_drift['mean_abs_dlogit']:.3e} (max "
            f"{attn_drift['max_abs_dlogit']:.3e}) over {attn_drift['positions_compared']} same-context positions")
    if inplace_drift is not None:
        log(f"drift dg-inplace vs dg-masked (same other impls): first token divergence "
            f"{inplace_drift['first_token_divergence'] or 'none'}, {inplace_drift['prompts_token_identical']}/"
            f"{len(prompts)} prompts token-identical, mean |dlogit| {inplace_drift['mean_abs_dlogit']:.3e} (max "
            f"{inplace_drift['max_abs_dlogit']:.3e}) over {inplace_drift['positions_compared']} same-context positions")
    log(f"PARITY LEVELS ({steps} steps x {len(prompts)} prompts, fp32-logit sha256 per position): {verdict}")
    return {"steps": steps, "n_prompts": len(prompts), "moe_impl": impl, "fp8_impl": fimpl, "glue_impl": gimpl,
            "glue_approx": gapprox, "attn_impl": aimpl, "variant": label, "verdict": verdict, "rows": rows_out,
            "drift": drift, "glue_drift": glue_drift, "attn_drift": attn_drift, "inplace_drift": inplace_drift,
            "timings_s": timings, "runs": runs}


# ------------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="T6 first light: parity reps, self-replay, step cell, cold hits.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hot-per-layer", type=int, default=None)
    ap.add_argument("--map", default=None, help="{layer: [hot ids]} JSON (overrides --hot-per-layer)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer-dir", default=None, help="default: --model-dir")
    ap.add_argument("--fixture", default="w0/results/phase1/parity-A.jsonl")
    ap.add_argument("--fixture2", default=None, help="default: parity-A2.jsonl next to --fixture")
    ap.add_argument("--out", default=None, help="rep-1 JSONL (default scratch/first-light-cand-<date>.jsonl)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--step-cell", type=int, default=512, help="decode steps in the step cell (0 = skip)")
    ap.add_argument("--step-tag", default="prose")
    ap.add_argument("--stats-from", type=int, default=32)
    ap.add_argument("--only", default=None, help="comma-separated fixture tags (default: all)")
    ap.add_argument("--kv-numerics", choices=["reference", "pg61"], default="reference",
                    help="pg61 = MXFP8 QDQ on the compressed-KV latent instead of the reference FP4 (runtime/patches.py)")
    ap.add_argument("--skip-parity", action="store_true", help="step cell only (e.g. under nsys)")
    ap.add_argument("--profile", action="store_true", help="cudaProfilerStart/Stop around step-cell steps")
    ap.add_argument("--profile-warm", type=int, default=32)
    ap.add_argument("--profile-steps", type=int, default=64)
    ap.add_argument("--nvtx-layers", action="store_true", help="NVTX L{i}/engram/attn/moe/head ranges (hooks)")
    ap.add_argument("--no-fill-uninit", action="store_true",
                    help="torch.utils.deterministic.fill_uninitialized_memory = False")
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--json", default=None, help="default docs/first-light-<YYYYMMDD>.json")
    ap.add_argument("--mode", choices=["eager", "static", "graph"], default="eager",
                    help="decode step: reference forward / T8 static step eagerly / its CUDA-graph replay")
    ap.add_argument("--moe-impl", choices=["ref", "dg-masked", "dg-inplace"], default="ref",
                    help="static/graph routed MoE: ref = reference fp4_gemm per expert (bit-identical, T8) / "
                         "dg-masked = 3 DeepGEMM masked grouped GEMMs per layer (T9a-1, runtime/dg_moe.py) / "
                         "dg-inplace = 3 DeepGEMM contiguous grouped GEMMs over the hot pool, cold experts staged "
                         "into spare slots (T9a-2, runtime/dg_inplace.py; adds a drift row vs dg-masked)")
    ap.add_argument("--fp8-impl", choices=["ref", "gemv"], default="ref",
                    help="static/graph fp8 Linear layers: ref = reference fp8_gemm (bit-identical, T8) / gemv = "
                         "Triton MXFP8 GEMV at <= 8 rows (T9b-1, runtime/gemv_mxfp8.py)")
    ap.add_argument("--glue-impl", choices=["ref", "fused"], default="ref",
                    help="static/graph glue: ref = today's torch glue ops / fused = the zero/low-risk glue clusters "
                         "as fused kernels, bit-identical to ref (T9b-2, runtime/fused_glue.py)")
    ap.add_argument("--glue-approx", action="store_true",
                    help="with --glue-impl fused: also run the round-2 clusters whose reduction order has no "
                         "bit-identical match on this device (outcome (ii), runtime/fused_glue_r2.py); the G row "
                         "becomes a drift row (reported, not gated)")
    ap.add_argument("--attn-impl", choices=["ref", "bounded", "flashmla"], default="ref",
                    help="static/graph sparse_attn: ref = reference TileLang kernel over the fixed 640-index width / "
                         "bounded = the same kernel with its block loop bounded by a device per-row valid-block "
                         "count, bit-identical (T9b-3a, runtime/kernels/sparse_attn_bounded.py) / flashmla = "
                         "FlashMLA sparse_prefill_fwd on every layer, A3-class (T9b-3c, runtime/attn_flashmla.py)")
    ap.add_argument("--max-pos", type=int, default=4095,
                    help="static/graph: largest decode position (fixes the compressed-KV cat and index-score widths)")
    ap.add_argument("--marker", action="store_true", help="launch t8_step_marker once per decode step (nsys)")
    ap.add_argument("--parity-levels", action="store_true", help="T8 L0/L1/L2 bit-identity levels before (a)-(e)")
    ap.add_argument("--level-steps", type=int, default=32, help="decode steps per prompt in --parity-levels")
    ap.add_argument("--levels-ref", default=None, help="earlier report JSON: compare its parity_levels runs")
    ap.add_argument("--rows", type=int, default=1, help="M1-T10: rows per decode step (6: the verify step; needs "
                                                         "--teacher-forced)")
    ap.add_argument("--teacher-forced", action="store_true", help="M1-T10: feed the 1-row records' next tokens")
    ap.add_argument("--rows-moe-impl", choices=["auto", "ref", "dg-masked", "dg-inplace"], default="auto",
                    help="the R-row step's MoE (auto: --moe-impl)")
    ap.add_argument("--hidden-layers", default="auto",
                    help="main_hidden layers: comma list, 'none', or auto (--rows > 1: the checkpoint config's "
                         "dspark_target_layer_ids, 37,38,39 on the real model -- the verify graph's T11 output)")
    ap.add_argument("--skip-census", action="store_true", help="--rows: skip the V0 census")
    return ap


def hidden_layers_arg(a) -> List[int]:
    """--hidden-layers -> layer ids (auto: [] at 1 row; at --rows > 1 the checkpoint config's dspark_target_layer_ids,
    which the loader's overlay blanks on the model itself)."""
    v = (a.hidden_layers or "").strip()
    if v in ("", "none"):
        return []
    if v != "auto":
        return [int(x) for x in v.split(",") if x.strip()]
    if a.rows == 1:
        return []
    path = a.config or os.path.join(a.model_dir, "config.json")
    try:
        with open(path) as f:
            return [int(x) for x in json.load(f).get("dspark_target_layer_ids") or []]
    except (OSError, ValueError):
        return []


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.rows != 1 and not a.teacher_forced:
        build_parser().error("--rows > 1 is the teacher-forced verify step: add --teacher-forced")

    date = time.strftime("%Y%m%d")
    out = a.out or os.path.join(REPO, "scratch", f"first-light-cand-{date}.jsonl")
    json_path = a.json or os.path.join(REPO, "docs", f"first-light-{date}.json")
    fixture2 = a.fixture2 or os.path.join(os.path.dirname(a.fixture), "parity-A2.jsonl")
    if not os.path.exists(fixture2):
        fixture2 = None
    report: Dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
                    "errors": []}
    from runtime.loader import build_and_load

    t0 = time.time()
    if a.kv_numerics == "pg61":
        from runtime.refmodel import import_reference
        import_reference(); from runtime import patches; patches.kv_numerics_pg61(True); print("[patch] kv_numerics=pg61 active", flush=True)
    model = build_and_load(a.model_dir, a.map, "cuda", hot_per_layer=a.hot_per_layer, config_path=a.config,
                           reserve_host_gib=a.reserve_host_gib)
    report["boot"] = model.runtime_load.summary()
    report["boot"]["wall_s"] = time.time() - t0
    report["boot"]["cuda_allocated_GiB"] = round(torch.cuda.memory_allocated() / 2**30, 3)
    # knobs after the boot, so the loader runs exactly as T2 verified it (deterministic mode would NaN-fill every
    # torch.empty it makes); CUBLAS_WORKSPACE_CONFIG is already in the environment (top of file), before any GEMM
    report["determinism"] = set_determinism(0, False if a.no_fill_uninit else None)
    report["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                     "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None}
    gen = Generator(model, a.tokenizer_dir or a.model_dir, mode=a.mode, max_pos=a.max_pos, marker=a.marker,
                    moe_impl=a.moe_impl, fp8_impl=a.fp8_impl, glue_impl=a.glue_impl, attn_impl=a.attn_impl,
                    glue_approx=a.glue_approx,
                    hidden_layers=hidden_layers_arg(a))
    report["generator"] = {"eos_ids": gen.eos_ids, "top_k": gen.top_k, "max_seq_len": gen.max_seq_len,
                           "tokenizer_dir": gen.tokenizer_dir, "mode": gen.mode, "max_pos": gen.max_pos,
                           "marker": gen.marker, "moe_impl": gen.moe_impl, "fp8_impl": gen.fp8_impl,
                           "glue_impl": gen.glue_impl, "glue_approx": gen.glue_approx, "attn_impl": gen.attn_impl}
    if a.nvtx_layers:
        install_layer_nvtx(model)
    if a.rows != 1:
        return measure_rows(gen, a, report, out=out, json_path=json_path, fixture2=fixture2)
    return measure(gen, a, report, n_layers=len(model.layers), out=out, json_path=json_path, fixture2=fixture2)


def measure(gen: Generator, a, report: Dict, *, n_layers: int, out: str, json_path: str,
            fixture2: Optional[str]) -> int:
    """Everything after boot: (a)-(e), the report JSON and the 12-line summary. Exit code 0, or 2 on any error."""
    rows = load_fixture(a.fixture)
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    try:
        gen.enc  # import encoding.py now, so a failure is reported once, not per prompt
        report.setdefault("generator", {})["encoding"] = gen.encoding_path
    except Exception:
        report["errors"].append({"stage": "encoding", "traceback": traceback.format_exc()})

    # (f) T8 parity levels, before anything else touches the static step
    if getattr(a, "parity_levels", False):
        try:
            ref_runs = None
            if getattr(a, "levels_ref", None):
                with open(a.levels_ref) as f:
                    ref_runs = (json.load(f).get("parity_levels") or {}).get("runs")
            report["parity_levels"] = parity_levels(gen, rows, a.level_steps, ref_runs=ref_runs)
            if report["parity_levels"]["verdict"] != "PASS":
                report["errors"].append({"stage": "parity_levels", "traceback": "verdict FAIL (see table)"})
        except Exception:
            tb = traceback.format_exc()
            print(f"[parity-levels] FAILED\n{tb}", file=sys.stderr, flush=True)
            report["errors"].append({"stage": "parity_levels", "traceback": tb})

    # (a) reps, (b) parity, (c) self-replay
    reps: Dict[int, dict] = {}
    if not a.skip_parity:
        for rep in range(1, a.reps + 1):
            p = rep_path(out, rep)
            t = time.time()
            res = run_rep(gen, rows, a.max_new, p, rep)
            reps[rep] = {"path": p, "wall_s": time.time() - t, "prompts": res}
            report["errors"] += [{"stage": f"rep{rep}", "tag": r["tag"], "traceback": r["error"]}
                                 for r in res if "error" in r]
        report["reps"] = reps
        report["parity"] = {}
        for rep, r in reps.items():
            print(f"--- parity rep {rep}: {r['path']} vs {a.fixture}" + (f" / {fixture2}" if fixture2 else ""))
            report["parity"][rep] = run_parity(r["path"], a.fixture, fixture2, r["path"] + ".parity.json")
        report["self_replay"] = {}
        for rep in range(2, a.reps + 1):
            sr = self_replay(reps[1]["path"], reps[rep]["path"], reps[1]["prompts"], reps[rep]["prompts"])
            report["self_replay"][rep] = sr
            for tag, v in sr["per_prompt"].items():
                if not v["identical"] or not v.get("logits_digest_equal"):
                    print(f"[self-replay rep1 vs rep{rep}] {tag}: first divergence {v.get('first_divergence')}, "
                          f"logits digest equal {v.get('logits_digest_equal')}")
            print(f"[self-replay rep1 vs rep{rep}] {sr['identical']}/{sr['n']} byte-identical, "
                  f"{sr['logits_digest_equal']}/{sr['n']} logits digests equal, files equal {sr['files_byte_identical']}")
        cold_l = [0] * n_layers
        hot_t = cold_t = 0
        for r in reps.get(1, {}).get("prompts", []):
            c = r.get("cold")
            if c:
                hot_t += c["hot_calls"]
                cold_t += c["cold_calls"]
                cold_l = [x + y for x, y in zip(cold_l, c["cold_calls_per_layer"])]
        report["cold_rep1"] = {"hot_calls": hot_t, "cold_calls": cold_t,
                               "cold_share": cold_t / (hot_t + cold_t) if hot_t + cold_t else None,
                               "cold_calls_per_layer": cold_l}

    # (d)+(e) step cell
    if a.step_cell > 0:
        try:
            row = next(r for r in load_fixture(a.fixture) if r["tag"] == a.step_tag)
            ids = gen.encode([{"role": "user", "content": row["prompt"]}])
            if a.profile:
                report["profile_window"] = gen.arm_profile(a.profile_warm, a.profile_steps)
            sc = step_cell(gen, ids, a.step_cell, a.stats_from)
            sc["mode"], sc["moe_impl"] = gen.mode, getattr(gen, "moe_impl", "ref")
            sc["fp8_impl"] = getattr(gen, "fp8_impl", "ref")
            sc["glue_impl"] = getattr(gen, "glue_impl", "ref")
            sc["attn_impl"] = getattr(gen, "attn_impl", "ref")
            r1 = next((r for r in reps.get(1, {}).get("prompts", []) if r["tag"] == a.step_tag and "token_ids" in r), None)
            if r1 is not None:
                n = min(len(r1["token_ids"]), len(sc["token_ids"]))
                same = next((i for i in range(n) if r1["token_ids"][i] != sc["token_ids"][i]), None)
                sc["vs_rep1_tokens"] = {"compared": n, "first_divergence": same}
            report["step_cell"] = sc
        except Exception:
            tb = traceback.format_exc()
            print(f"[step-cell] FAILED\n{tb}", file=sys.stderr)
            report["errors"].append({"stage": "step_cell", "traceback": tb})

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    report.setdefault("generator", {})["moe_impl"] = getattr(gen, "moe_impl", "ref")
    report["generator"]["fp8_impl"] = getattr(gen, "fp8_impl", "ref")
    report["generator"]["glue_impl"] = getattr(gen, "glue_impl", "ref")
    report["generator"]["glue_approx"] = bool(getattr(gen, "glue_approx", False))
    report["generator"]["attn_impl"] = getattr(gen, "attn_impl", "ref")
    static = getattr(gen, "static", None)
    if static is not None and hasattr(static, "attn_describe"):
        report["attn_bound"] = static.attn_describe()  # T9b-3a: on/off (why), the loop spelling, self-check results
    if static is not None and hasattr(static, "moes"):
        try:
            from runtime.static_kernels import describe as gather_describe

            report["gather"] = gather_describe()  # T9b-3a: 16-B vector gather or the byte kernel (and why)
        except Exception as e:  # e.g. the Mac's emulated static_kernels: informative only, never an error
            report["gather"] = {"error": f"{type(e).__name__}: {e}"}
    glue = getattr(getattr(gen, "static", None), "glue", None)
    if glue is not None and hasattr(glue, "describe"):
        report["fused_glue"] = glue.describe()  # clusters on / off (and why), self-check modes and results
        if hasattr(glue, "glue_line"):
            print(glue.glue_line(), flush=True)  # [glue] clusters on/approx/off, R2 numerics (i)/(ii), orders
    if report["generator"]["fp8_impl"] == "gemv":
        try:
            from runtime.gemv_mxfp8 import describe as gemv_describe

            report["fp8_gemv"] = gemv_describe()  # triton version, per-shape configs and tuned µs
        except Exception:
            report["errors"].append({"stage": "fp8_gemv_describe", "traceback": traceback.format_exc()})
    dg = getattr(getattr(gen, "static", None), "dg", None)
    if dg is not None and hasattr(dg, "describe"):
        report["dg_moe"] = dg.describe()  # deep_gemm path/version, e8m0 conversion, deterministic flag, buffers
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    for line in summary_lines(report, json_path):
        print(line)
    return 0 if not report["errors"] else 2


def summary_lines(r: dict, json_path: str) -> List[str]:
    """Exactly 12 lines."""
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    b = r.get("boot", {})
    g = r.get("generator", {})
    pl = (r.get("parity_levels") or {}).get("verdict")
    moe = g.get("moe_impl", "ref")
    fp8 = g.get("fp8_impl", "ref")
    glue = g.get("glue_impl", "ref")
    if glue != "ref" and g.get("glue_approx"):
        glue += "+approx"
    attn = g.get("attn_impl", "ref")
    if attn != "ref" and r.get("attn_bound") is not None and not r["attn_bound"].get("on"):
        attn += "(OFF)"
    lines = [f"FIRST LIGHT {r['date']} | mode {g.get('mode', 'eager')}{'' if moe == 'ref' else f' moe {moe}'}"
             f"{'' if fp8 == 'ref' else f' fp8 {fp8}'}{'' if glue == 'ref' else f' glue {glue}'}"
             f"{'' if attn == 'ref' else f' attn {attn}'}"
             f"{'' if pl is None else f' levels {pl}'} | "
             f"map {b.get('map')} | boot {f(b.get('wall_s'), 0)} s | "
             f"HBM {f(b.get('cuda_allocated_GiB'))} GiB | {r['env'].get('device')}"]
    reps = r.get("reps", {})
    p1 = reps.get(1, {}).get("prompts", [])
    m = [x.get("prompt_tokens_match") for x in p1]
    lines.append(f"fixture encode: {sum(1 for x in m if x)}/{len(m)} prompt-token counts == Pg61 usage.prompt_tokens")
    for rep in (1, 2):
        ps = reps.get(rep, {}).get("prompts", [])
        ok = [x for x in ps if "error" not in x]
        dms = [x["decode_ms_per_token"] for x in ok if x.get("decode_ms_per_token")]
        lines.append(f"rep{rep}: {len(ok)}/{len(ps)} prompts ok, {sum(x.get('n_tokens', 0) for x in ok)} tokens, "
                     f"{f(reps.get(rep, {}).get('wall_s'), 0)} s, decode mean {f(sum(dms) / len(dms) if dms else None, 1)} ms/tok (incl. logprob work)")
    for rep in (1, 2):
        pr = r.get("parity", {}).get(rep, {})
        lines.append(f"parity rep{rep} vs Pg61-A: {pr.get('verdict_line') or pr.get('error', 'not run')!s}"[:160])
    sr = r.get("self_replay", {}).get(2)
    if sr:
        bad = [f"{t}@{'missing' if v.get('missing') else (v.get('first_divergence') or {}).get('index')}"
               for t, v in sr["per_prompt"].items() if not v["identical"] or not v.get("logits_digest_equal")]
        lines.append(f"self-replay rep1 vs rep2: {sr['identical']}/{sr['n']} byte-identical, {sr['logits_digest_equal']}/"
                     f"{sr['n']} fp32-logit digests equal; diverged: {', '.join(bad) or 'none'}")
    else:
        lines.append("self-replay: not run")
    sc = r.get("step_cell")
    if sc:
        st = sc["ms_per_step"]
        lines.append(f"step cell ({sc['timer']}): steps {sc['stats_from_step']}..{sc['n_steps']} mean {f(st.get('mean'))} / "
                     f"p50 {f(st.get('p50'))} / p90 {f(st.get('p90'))} ms per step")
        lines.append(f"step cell: {f(sc.get('tokens_per_s_from_mean'))} tok/s (1/mean), {f(sc.get('tokens_per_s_wall_all_steps'))} "
                     f"tok/s wall; prefill {f(sc.get('prefill_ms'), 0)} ms; vs rep1 tokens {sc.get('vs_rep1_tokens')}")
        c = sc.get("cold")
        if c:
            cs, cc = c["cold_share_per_step"], c["cold_calls_per_step"]
            lines.append(f"cold share/step mean {f(cs.get('mean'), 4)} p50 {f(cs.get('p50'), 4)} p90 {f(cs.get('p90'), 4)}; "
                         f"cold calls/step mean {f(cc.get('mean'))} (routed {f(c['routed_calls_per_step'].get('mean'), 0)}/step)")
            per = c["cold_calls_per_layer"]
            top = sorted(range(len(per)), key=lambda i: -per[i])[:5]
            lines.append("cold calls per layer (step cell), top 5: " + ", ".join(f"L{i}={per[i]}" for i in top))
        else:
            lines += ["cold counters: not installed", "-"]
    else:
        lines += ["step cell: not run", "-", "-", "-"]
    lines.append(f"errors: {len(r['errors'])} | report {json_path}")
    assert len(lines) == 12, len(lines)
    return lines


if __name__ == "__main__":
    sys.exit(main())
