"""bench/glue_census.py (the instrument behind docs/GLUE-AUDIT.md) on the Mac: the eager reference path's kernel counts
must equal what nsys measured on the box (docs/STEP-ANATOMY-M0-20261106.md / step-anatomy-m0-20261106.json
`top_kernels`: fp4_gemm 720, fp8_gemm 290, act_quant 1090, sparse_attn 40, hc_split_sinkhorn 80, fp4_quant 13), and
the T9b-1 path must route exactly the 290 fp8 Linears to the GEMV. Run in a subprocess: the census swaps
`runtime.static_kernels` / `deep_gemm` in sys.modules for CPU fakes."""
import collections
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kernel_counts(paths):
    out = {}
    for name, d in paths.items():
        c = collections.Counter()
        for reg in d["regions"].values():
            for op, n in reg["ops"].items():
                if op.startswith("K:"):
                    c[op[2:]] += n
        out[name] = c
    return out


def test_census_matches_anatomy_kernel_counts(tmp_path):
    js = tmp_path / "census.json"
    r = subprocess.run([sys.executable, os.path.join(REPO, "bench", "glue_census.py"), "--json", str(js)],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    rep = json.loads(js.read_text())
    k = _kernel_counts(rep["paths"])
    anatomy = {"fp4_gemm": 720, "fp8_gemm": 290, "act_quant": 1090, "sparse_attn": 40, "hc_split_sinkhorn": 80,
               "fp4_act_quant": 13}
    assert {n: k["eager-ref"][n] for n in anatomy} == anatomy
    # T8: the same reference GEMMs plus the pointer-table gather (6 tables x 40 layers)
    assert k["static-ref"]["fp8_gemm"] == 290 and k["static-ref"]["fp4_gemm"] == 720
    assert k["static-ref"]["t8_gather_rows"] == 240
    # T9a-1 + T9b-1: no reference GEMM left; 290 GEMVs; 3 masked grouped GEMMs per layer
    g = k["static-dg-gemv"]
    assert g["fp8_gemm"] == 0 and g["fp4_gemm"] == 0
    assert g["mxfp8_gemv"] == 290
    assert g["dg_masked_gemm"] == 120
    # the per-layer structure is what the audit's tables assume
    lt = rep["paths"]["static-dg-gemv"]["layer_types"]
    assert sum(v["layers"] for t, v in lt.items() if t != "step") == 40
    assert lt["r1"]["layers"] == 15 and lt["r2"]["layers"] == 15
