"""GPU tests of T9a-3 (docs/HANDOFF-T9A3-DG-ROWMATCH.md): one DeepGEMM expert path for 1 and 6 rows. Skipped without
CUDA/tilelang (Mac); on the box a missing/broken deep_gemm FAILS (it is under test).

  * real bytes (checkpoint at /ref or $DSV41_CKPT): bench/dg_rowmatch_spike.py on layer 0's real pool for 8 routed
    sets of 6 rows x 6 experts, every set routing 1-3 cold experts (staged in spare slots): GATED -- the
    --moe-rowmatch on call (ALIGN 16, 36 blocks, one token) per-row bitwise == the 6-row step's call, fp32 and
    packed activation SF; REPORTED -- every other pair (today's 1-row call, ALIGN-only, c128, masked 1 vs 6, the
    on call over G vs G + 16 groups), the
    graph-timed us/layer of the 1-row shapes and the projected 1-row step (receipt scratch/dg-rowmatch-spike-*.json);
  * build check on mini4: the 1-row dg-inplace StaticStep builds in the on shape and its build-time per-row check
    (real DeepGEMM, layer 0's mini bytes) is equal;
  * mini4 6-row static logits == 1-row static logits per row, --fp32-impl fixed, dg-inplace on both (1-row: rowmatch
    on; 6-row: contiguous16, S = 8), both P0 parities, 24 positions each:
      GATED  "MoE isolated": every non-MoE op row-invariant by construction (fp8 ref, glue ref, attn ref, fp32 fixed,
             and tests/gpu/test_dspark_loop_gpu.row_invariant_mode(True): F.linear / einsum / sum / mean one row at a
             time) -> the only path left that sees 6 rows together is the MoE (dedup, DeepGEMM, combine);
      REPORTED the same with rowmatch off (did the DG call shape ever matter?), and the headline's impls (gemv pinned
             M=1, fused glue, flashmla, fp32 fixed, no debug mode) with the fused glue's reduction orders of both steps
             side by side: an order that differs between the 1-row and the 6-row build reduces that statistic in a
             different fp32 order at 1 and 6 rows (fused_glue_r2.calibrate_r2 picks each step's orders to match
             torch at ITS row count), which no MoE change can make bit-equal.

Run on the box (orchestrator), same image / mounts as tests/gpu/test_moe_rows_gpu.py (+ -v /models... or /ref):
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_dg_rowmatch_gpu.py
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_dg_moe_gpu import _need_dg  # noqa: E402
from test_static_rows_gpu import ROWS, _model, _need_step, _seq1, _static, _toks  # noqa: E402

FAST = dict(fp8_impl="gemv", glue_impl="fused", attn_impl="flashmla")
REPORT = []


@contextlib.contextmanager
def _env(**kv):
    prev = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _orders(st):
    o = getattr(getattr(st, "glue", None), "orders", None) or {}
    return {k: (v.get("name") if isinstance(v, dict) else str(v)) for k, v in o.items()}


def _rows_equal(a6, a1):
    assert len(a6) == len(a1)
    return [i for i, (x, y) in enumerate(zip(a6, a1)) if not torch.equal(x, y)]


# ------------------------------------------------------------------------------------------------ real bytes
def test_real_bytes_rowmatch_spike():
    _need_dg()
    ck = os.environ.get("DSV41_CKPT", "/ref")
    if not os.path.isdir(ck):
        raise unittest.SkipTest(f"checkpoint not mounted at {ck}")
    import dg_rowmatch_spike as sp

    out = os.path.join(REPO, "scratch", f"dg-rowmatch-spike-{time.strftime('%Y%m%d')}-gputest.json")
    rc = sp.main(["--ckpt", ck, "--sets", "8", "--iters", "100", "--out", out])
    rep = json.load(open(out))
    v = rep["verdict"]
    assert rep["pool"]["cold_bytes_equal_checkpoint"], rep["pool"]
    assert min(rep["pool"]["cold_per_set"]) >= 1 and len(rep["pool"]["sets"]) == 8
    for key, p in sorted(rep["pairs"].items()):
        print(f"  {key:30s} rows bit-equal {p['rows_equal']:3d}/{p['of']}  max ulp {p['max_ulp']}")
    t = rep["timing"].get("us_per_layer") or {}
    print(f"  us/layer (3 GEMMs, one token): today {t.get('r1_c128_pk')} | on {t.get('r1_c16_pad_pk')} | align "
          f"{t.get('r1_c16')} | 6-row {t.get('r6_c16')}; projected 1-row step (on) {t.get('projected_1row_ms_on')} ms "
          f"(gate {v.get('gate_1row_ms')})")
    print("  VERDICT", json.dumps(v))
    assert not v["failed"], v["failed"]
    print(f"  num_groups invariant (G vs G + 16): {v.get('num_groups_invariant')}")
    assert v["rowmatch_on_equal_r6"] and v["rowmatch_on_packed_equal_r6"], v
    assert rc in (0, 2)


# ------------------------------------------------------------------------------------------------ build check
def test_build_check_mini4_real_deepgemm():
    _need_dg()
    from runtime import dg_inplace as DI

    with _model("mini4") as (model, pool), _env(**{DI.ROWMATCH_ENV: "on"}):
        st = _need_step(model, pool, moe_impl="dg-inplace")
        rm = st.dg.rowmatch
        print(f"  mini4 1-row build: rowmatch {rm['mode']} -> on={rm['on']} align {st.dg.align} gemm_blocks "
              f"{st.dg.gemm_blocks}; check {json.dumps({p: v['build_vs_rows'] for p, v in rm['check']['projs'].items()})}"
              f"/{rm['check']['rows']}, today's call equal {rm['check']['today_equal']}")
        assert rm["on"] and rm["check"]["equal"], rm
        assert st.dg.align == 16 and st.dg.gemm_blocks == ROWS * st.topk
        st.detach()


# ------------------------------------------------------------------------------------------------ 6-row vs 1-row steps
def _one_vs_six(model, pool, L, n_pos, rowmatch, kw, rowinv):
    from runtime import dg_inplace as DI
    from runtime.fused_glue_r2 import FP32_IMPL_ENV
    from test_dspark_loop_gpu import row_invariant_mode

    ids, toks = _toks(L, n_pos, 90 + L)
    with _env(**{DI.ROWMATCH_ENV: rowmatch, FP32_IMPL_ENV: "fixed", "DSV41_GEMV_PIN_M1": "1"}), \
            row_invariant_mode(rowinv):
        s6 = _need_step(model, pool, rows=ROWS, moe_impl="dg-inplace", spare_slots=8, rows_moe_gemm="contiguous16",
                        **kw)
        s1 = _need_step(model, pool, moe_impl="dg-inplace", **kw)
        try:
            a1 = _seq1(model, s1, ids, toks, n_pos)
            a6 = _static(model, s6, ids, toks, n_pos // ROWS)[0]
            bad = _rows_equal(a6, a1)
            ov = s6.overflow_counts()
            info = {"rowmatch_on": s1.dg.rowmatch["on"], "align1": s1.dg.align, "blocks1": s1.dg.gemm_blocks,
                    "check": (s1.dg.rowmatch.get("check") or {}).get("equal"), "overflow": ov,
                    "orders1": _orders(s1), "orders6": _orders(s6)}
        finally:
            s1.detach()
            s6.detach()
    return bad, info


def test_mini4_static6_equals_static1_moe_isolated():
    """GATED: with every non-MoE op row-invariant by construction, the 6-row step's rows == the 1-row step's."""
    _need_dg()
    with _model("mini4") as (model, pool):
        pool.add_spare_slots(8, model)                  # grow once, before any build (no re-growth between them)
        n_pos = ROWS * 4
        res = {}
        for rowmatch in ("on", "off"):
            for L in (17, 18):
                bad, info = _one_vs_six(model, pool, L, n_pos, rowmatch, {}, True)
                res[(rowmatch, L)] = bad
                assert info["overflow"]["steps"] == 0, info["overflow"]
                print(f"  MoE isolated, rowmatch {rowmatch:3s} (1-row align {info['align1']} x {info['blocks1']}, "
                      f"check {info['check']}), L={L}: 6-row vs 1-row rows bit-equal {n_pos - len(bad)}/{n_pos}"
                      f"{'' if not bad else ' differing ' + str(bad)}")
        assert not res[("on", 17)] and not res[("on", 18)], res


def test_mini4_headline_impls_reported():
    """REPORTED (not gated): the headline's impls, no debug mode; the fused glue orders of both steps."""
    _need_dg()
    with _model("mini4") as (model, pool):
        pool.add_spare_slots(8, model)
        n_pos = ROWS * 4
        for rowmatch in ("on", "off"):
            for L in (17, 18):
                try:
                    bad, info = _one_vs_six(model, pool, L, n_pos, rowmatch, FAST, False)
                except Exception as e:        # REPORTED only: an impl that cannot build / run here is reported
                    print(f"  headline impls n/a: {type(e).__name__}: {str(e)[:300]}")
                    REPORT.append({"rowmatch": rowmatch, "L": L, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                    return
                diff = sorted(k for k in set(info["orders1"]) | set(info["orders6"])
                              if info["orders1"].get(k) != info["orders6"].get(k))
                print(f"  headline impls, rowmatch {rowmatch:3s}, L={L}: 6-row vs 1-row rows bit-equal "
                      f"{n_pos - len(bad)}/{n_pos}; glue orders differing 1-row vs 6-row: "
                      f"{ {k: (info['orders1'].get(k), info['orders6'].get(k)) for k in diff} or 'none'}")
                REPORT.append({"rowmatch": rowmatch, "L": L, "equal": n_pos - len(bad), "of": n_pos,
                               "orders_differ": diff})


TESTS = [test_real_bytes_rowmatch_spike, test_build_check_mini4_real_deepgemm,
         test_mini4_static6_equals_static1_moe_isolated, test_mini4_headline_impls_reported]

if __name__ == "__main__":
    import traceback

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
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
