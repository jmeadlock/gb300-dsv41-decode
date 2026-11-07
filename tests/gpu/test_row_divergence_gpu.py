"""T12-DIAG GPU test (box, inside vllm/vllm-openai:v0.31.0): the 6-row verify step vs the 1-row step, per layer, per
row, through bench/row_divergence_probe.py. Run: `python3 tests/gpu/test_row_divergence_gpu.py` (exit 1 on failure).

Conditions: the headline impls (moe dg-inplace, fp8 gemv, glue fused, attn flashmla) with $DSV41_FP32_IMPL=fixed,
$DSV41_MOE_ROWMATCH=on, $DSV41_GEMV_PIN_M1=1 and $DSV41_ROUTER_IMPL=plan-gemv (docs/T12-DIAG-UPDATE.md: the router's
cuBLAS fp32 gate mm was the residual; the fixed-order gate GEMV made 7/8 headline streams exact). PLAIN = exactly
that; FIX = that + every fp32 F.linear at M > 1 evaluated one row at a time (the 1-row cuBLAS call shape: the
compressor's fp32 wkv / wgate) + FlashMLA one call per row.

Gated (mini4):
  * teacher-forced (plan 5,5,5,5), PLAIN and FIX: every tap of every row (attention inputs / kv checksum / output,
    the compressed-cache rows a step writes (cache.ckv / cache.idxk), indexer top-k, router ids / weights, F.linear
    and einsum outputs, ffn / block / Engram outputs, head) and the fp32 logits bit-equal to the 1-row step;
  * with rejections (plan 5,0,2,5,1,3,0,4: rejected rows' writes left in the caches, ratio-2 state committed at A'
    as the DSpark loop does), PLAIN: the same on every accepted row, AND the accepted rows' logits are identical
    for two different wrong-token sets in the rejected rows (no rejected write is read: candidate 3);
  * the probe is not blind: a one-ulp move of layer 1's ffn output on row 2 of the first 6-row step is reported as
    exactly that tap (step 0, row 2, layer 1, ffn.out), rows 0-1 stay equal;
  * FlashMLA row split (one s_q = 1 call per row) == the plain multi-row call at the real widths (h 64, d 512,
    128 window + 512 compressed with per-row lengths): the prefill kernel has no split-KV / num_splits.
Reported (not gated): the synthetic real-width census of the step's cuBLAS / topk / FlashMLA sites; and, with
$DSV41_MODEL_DIR set, the real checkpoint: $DSV41_ROWDIV_PLAN_FROM=<headline json>:<tag> replays that prompt's accept
log (the loop's own P0 sequence; e.g. docs/headline-20261107i-plan-gemv.json:shell, whose stream diverges at token
171 = step 46) on the 1-row greedy stream of the prompt from $DSV41_ROWDIV_FIXTURE, PLAIN then FIX, plus a second
rejected-token variant -- the first (layer, row, tensor) that differs, in ulp, and whether a rejected row leaks.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import time
import traceback
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("row_divergence_probe", os.path.join(REPO, "bench",
                                                                                    "row_divergence_probe.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

DEV = "cuda"      # tests/test_row_divergence_cpu.py dry-runs this file on the CPU emulations (DEV "cpu")
STATIC_KW: dict = {}
IMPLS = dict(P.HEADLINE_IMPLS)
ENV = dict(P.HEADLINE_ENV, DSV41_ROUTER_IMPL="plan-gemv")
FIX = {"rowwise": "fp32", "row_split": True}
PLAN_TF = (5, 5, 5, 5)
PLAN_REJ = (5, 0, 2, 5, 1, 3, 0, 4)
REPORT: list = []
COUNTS = {"rows": 0, "taps": 0, "gates": 0}


def _need_gpu():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    from runtime.refmodel import tilelang_available

    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


@contextlib.contextmanager
def _model(kind: str = "mini4"):
    with P.mini_model(kind, DEV) as mp:
        yield mp


def _run(model, pool, plan, fix: bool, **kw):
    e = dict(ENV, DSV41_FLASHMLA_ROW_SPLIT="1" if fix and FIX["row_split"] else None)
    with P.env(**e):
        return P.run_probe(model, pool, impls=IMPLS, plan=plan, rowwise=FIX["rowwise"] if fix else "off",
                           static_kw=STATIC_KW, log=print, **kw)


def _bad_rows(rep):
    return [r for r in rep["rows"] if r["first"] is not None or not r["logits_equal"]]


def _assert_all_equal(rep, what):
    s = rep["summary"]
    for line in P.report_lines(rep):
        print("  " + line)
    bad = _bad_rows(rep)
    assert not bad, (f"{what}: {len(bad)}/{len(rep['rows'])} rows differ from the 1-row step; first: step "
                     f"{bad[0]['step']} row {bad[0]['row']} pos {bad[0]['pos']}: {bad[0]['first']} (logits "
                     f"{bad[0]['logits']})")
    assert s["rows_compared"] == sum(n + 1 for n in rep["plan"]), s
    assert rep["taps_compared_first_row"] > 0, "the probe compared no tap"
    COUNTS["rows"] += s["rows_compared"]
    COUNTS["taps"] += rep["taps_compared"]
    COUNTS["gates"] += 1


# ------------------------------------------------------------------------------------------------ tests
def test_rowdiv_mini4_teacher_forced():
    _need_gpu()
    with _model("mini4") as (model, pool):
        out = {}
        for fix in (False, True):
            rep = _run(model, pool, PLAN_TF, fix=fix)
            out[fix] = rep
            assert rep["taps_compared_first_row"] >= 8 * len(model.layers), rep["taps_compared_first_row"]
        # gate after both ran, so a PLAIN failure still says whether the FIX rescues it
        tags = {f: ("all equal" if not _bad_rows(r) else P.report_lines(r)[2].strip()) for f, r in out.items()}
        print(f"  mini4 teacher-forced (router {out[False]['router_impl']}): plain: {tags[False]}; fix: {tags[True]}")
        _assert_all_equal(out[True], "mini4 teacher-forced, FIX")
        _assert_all_equal(out[False], "mini4 teacher-forced, PLAIN")


def test_rowdiv_mini4_rejections():
    _need_gpu()
    with _model("mini4") as (model, pool):
        rep = _run(model, pool, PLAN_REJ, fix=False)
        alt = _run(model, pool, PLAN_REJ, fix=False, reject_variant=1)
        leaks = P.variant_leaks([rep, alt])
        print(f"  mini4 rejections {list(PLAN_REJ)}: rows differing {len(_bad_rows(rep))}; accepted rows changed by "
              f"the rejected tokens: {leaks or 'none'}")
        _assert_all_equal(rep, "mini4 rejections, PLAIN")
        assert not leaks, f"accepted rows depend on the rejected rows' tokens: {leaks}"
        COUNTS["gates"] += 1
        print(f"  mini4 rejections: accepted rows all equal to the 1-row step; no rejected-row leak")


def test_rowdiv_injection_is_named():
    _need_gpu()
    with _model("mini4") as (model, pool):
        rep = _run(model, pool, (5,), fix=False, inject=(1, "ffn.out", 2))
        o = rep["summary"]["origin"]
        assert o is not None and (o["step"], o["row"]) == (0, 2) and o["first"]["key"] == [1, "ffn.out", 0], o
        assert o["first"]["max_ulp"] == 1 and o["first"]["mism"] == 1, o
        assert all(r["first"] is None for r in rep["rows"] if r["row"] < 2), [r["first"] for r in rep["rows"]]
        COUNTS["gates"] += 1
        print(f"  injection (layer 1 ffn.out row 2, +1 ulp): reported as {o['first']['key']} at step 0 row 2; "
              f"rows 0-1 equal")


def test_flashmla_row_split_real_widths():
    _need_gpu()
    from runtime import attn_flashmla as AFM

    try:
        AFM.load_ops()
    except RuntimeError as e:
        raise unittest.SkipTest(f"FlashMLA .so not loadable here: {e}")
    g = torch.Generator(device="cpu").manual_seed(5)
    split = P._flashmla_case(torch.device(DEV), P.ROWS, g, split=True)
    plain = P._flashmla_case(torch.device(DEV), P.ROWS, g, split=False)
    print(f"  FlashMLA real widths: row-split == one call {split['equal']}; m=6 vs m=1 {plain['equal']} "
          f"(max ulp {plain.get('max_ulp')})")
    assert split["equal"], split
    REPORT.append(f"FlashMLA m=6 vs m=1 at real widths: equal {plain['equal']} (max ulp {plain.get('max_ulp')})")
    COUNTS["gates"] += 1


def test_real_width_census_reported():
    _need_gpu()
    from runtime.row_census import census_table

    recs = P.real_width_census(DEV)
    for line in census_table(recs):
        print("  " + line)
    REPORT.append("real-width census NOT row-invariant: " + (", ".join(
        f"{r['op']} ({r.get('max_ulp')} ulp)" for r in recs if r.get("equal") is False) or "none"))
    # T9b-8: the compressor's wkv / wgate at ratio 2 under --fp32-impl fixed (the kernel the step runs) reads yes
    comp = [r for r in recs if r["family"] == "T9b-8 fixed fp32" and "compressor wkv / wgate, ratio 2" in r["op"]]
    assert len(comp) == 1 and comp[0]["equal"] is True, comp
    assert "replay equal True" in comp[0].get("note", ""), comp
    REPORT.append(f"real-width census: {comp[0]['op']}: row-invariant yes ({comp[0]['note']})")


def test_rowdiv_real_reported():
    """$DSV41_MODEL_DIR: the real checkpoint (headline build line: GEMV pin, map, pool pre-grown to 16 spare slots).
    $DSV41_ROWDIV_PLAN_FROM (headline json:tag) replays that prompt's accept log; else $DSV41_ROWDIV_STEPS
    teacher-forced steps. $DSV41_ROWDIV_FIXTURE (+ the tag) picks the prompt. PLAIN, FIX, and a second
    rejected-token variant (PLAIN)."""
    _need_gpu()
    md = os.environ.get("DSV41_MODEL_DIR")
    if not md:
        raise unittest.SkipTest("DSV41_MODEL_DIR not set")
    from runtime.generate import set_determinism
    from runtime.loader import build_and_load
    from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS

    spec = os.environ.get("DSV41_ROWDIV_PLAN_FROM")
    if spec:
        plan, tag = P.plan_from(spec)
    else:
        plan, tag = (5,) * int(os.environ.get("DSV41_ROWDIV_STEPS", "12")), os.environ.get("DSV41_ROWDIV_TAG")
    with P.env(**ENV):
        mp = os.environ.get("DSV41_MAP") or os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json")
        model = build_and_load(md, mp if os.path.exists(mp) else None, "cuda")
        model.temperature = 0.0
        set_determinism(0)
        pool = model.runtime_load.pool
        pool.add_spare_slots(DEFAULT_SPARE_SLOTS_ROWS, model)
        fx = os.environ.get("DSV41_ROWDIV_FIXTURE")
        ids = P._encode(md, fx if fx and os.path.exists(fx) else None, tag, None)
        kw = dict(ids=ids, max_pos=4095, spare_slots=DEFAULT_SPARE_SLOTS_ROWS, stream="greedy")
        reps = {}
        for name, fix, var in (("PLAIN", False, 0), ("FIX", True, 0), ("PLAIN rejected-variant 1", False, 1)):
            reps[name] = rep = _run(model, pool, plan, fix=fix, reject_variant=var, **kw)
            print(f"  real {tag or ''} ({len(ids)} prompt tokens, {len(plan)} steps) {name}:")
            for line in P.report_lines(rep):
                print("    " + line)
            REPORT.append(f"real {tag or ''} {name}: " + " | ".join(l.strip() for l in P.report_lines(rep)[1:3]))
        leaks = P.variant_leaks([reps["PLAIN"], reps["PLAIN rejected-variant 1"]])
        REPORT.append(f"real {tag or ''}: accepted rows changed by the rejected tokens: {leaks[:12] or 'none'}")


TESTS = [test_rowdiv_mini4_teacher_forced, test_rowdiv_mini4_rejections, test_rowdiv_injection_is_named,
         test_flashmla_row_split_real_widths, test_real_width_census_reported, test_rowdiv_real_reported]

if __name__ == "__main__":
    results = {}
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            results[t.__name__] = "PASS"
        except unittest.SkipTest as e:
            results[t.__name__] = f"SKIP ({e})"
        except Exception:
            traceback.print_exc()
            results[t.__name__] = "FAIL"
        print(f"{results[t.__name__]:4s} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    for line in REPORT:
        print(f"  note: {line}")
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"comparisons executed: {COUNTS}")
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
