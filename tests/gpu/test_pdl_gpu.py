"""GPU gate of M2-L5 (runtime/pdl.py; docs/HANDOFF-M2-L5-PDL-IMPL.md "Do (1)-(3)") on the synthetic mini checkpoint:
PDL / pre-wait prefetch move timing only, so every mode must give the SAME BITS as $DSV41_PDL=off.

  * per mode in on / on+late / prefetch / prefetch+late, fast impls (dg-inplace, gemv, flashmla, fused glue), fp32
    fixed (today's headline form), the router's default plan-gemv:
      - 6-row static-eager == the off 6-row static-eager (fp32 logits of every row, main_hidden, every cache the step
        writes), 6-row graph == its eager, replay #2 == #1;
      - 1-row static-eager over the same teacher-forced positions == the off 1-row run, 1-row graph == its eager;
      - so "1 row == 6 rows per row" keeps exactly off's relation (printed: rows equal 6-row vs 1-row, per mode);
  * graph census (keep_graph capture of the 6-row step, runtime.pdl.graph_census): programmatic edges into owned
    kernels == the census's expectation, none into borrowed kernels, owned grids that exceed the SM count;
  * MEASURED mini step: graph replay ms per mode (median of --replays, events), the delta vs off -- the mini's shapes
    are tiny, so this checks direction and overhead, not the headline number (bench/headline.py is the gate).

Run on the box (orchestrator), same line as tests/gpu/test_static_rows_gpu.py, repo at $W:

  W=$HOME/hermes/gb300-dsv41-decode-wtl5
  docker run --rm --name mk-pdl-$(date +%s) --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \\
    --ipc=host --ulimit memlock=-1:-1 -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_pdl_gpu.py

(`python3 tests/gpu/test_pdl_gpu.py` runs every test without pytest and prints the tables; receipt
scratch/pdl-gpu-<utc>.json.)
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_static_rows_gpu import (FAST_INPLACE, MAX_POS, ROWS, _assert_rows, _model, _need_tilelang,  # noqa: E402
                                  _seq1, _static, _toks)

MODES = [("on", "entry"), ("on", "late"), ("prefetch", "entry"), ("prefetch", "late")]
N_STEPS = 3
REPLAYS = int(os.environ.get("PDL_GPU_REPLAYS", "200"))
RECEIPT: dict = {"modes": {}, "census": {}, "timing_label": "MEASURED (mini checkpoint graph replay)"}


@contextlib.contextmanager
def _env(mode: str, trigger: str = "entry"):
    keys = {"DSV41_PDL": mode, "DSV41_PDL_TRIGGER": trigger, "DSV41_FP32_IMPL": "fixed"}
    prev = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _kept_graphs():
    """StaticStep.capture builds `torch.cuda.CUDAGraph()`: make those keep_graph=True (census needs the topology;
    instantiated right after capture). Yields False when this torch has no keep_graph (census skipped)."""
    orig = torch.cuda.CUDAGraph
    try:
        orig(keep_graph=True)
    except TypeError:
        yield False
        return
    torch.cuda.CUDAGraph = lambda *a, **k: orig(keep_graph=True)
    try:
        yield True
    finally:
        torch.cuda.CUDAGraph = orig


def _replay_ms(st, ids, toks) -> float:
    """Median ms of one graph replay of the 6-row step (inputs as the first teacher-forced step)."""
    from test_static_rows_gpu import _prefill

    _prefill(st.model, ids)
    with torch.inference_mode():
        st.step(toks[:ROWS], len(ids), graph=True)
        torch.cuda.synchronize()
        g = st.graphs[0]
        ts = []
        for _ in range(REPLAYS):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def _run_mode(model, pool, ids, toks, mode, trigger, census: bool):
    from runtime import pdl
    from runtime.static_step import StaticStep

    hid = (len(model.layers) - 1,)
    out = {}
    with _env(mode, trigger):
        pdl.TALLY.clear()
        st = StaticStep(model, pool, max_pos=MAX_POS, rows=ROWS, hidden_layers=hid, **FAST_INPLACE)
        out["e6"] = _static(model, st, ids, toks, N_STEPS)
        with _kept_graphs() if census else contextlib.nullcontext(False) as kept:
            st.capture()
            if kept:
                for g in st.graphs.values():
                    g.instantiate()
        if kept:
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            out["census"] = pdl.graph_census(st.graphs[0], sms=sms)
        out["g6"] = _static(model, st, ids, toks, N_STEPS, graph=True)
        out["g6b"] = _static(model, st, ids, toks, N_STEPS, graph=True)
        out["ms"] = _replay_ms(st, ids, toks)
        out["switch"] = pdl.describe()
        st.detach()
        st1 = StaticStep(model, pool, max_pos=MAX_POS, hidden_layers=hid, **FAST_INPLACE)
        out["e1"] = _seq1(model, st1, ids, toks, ROWS * N_STEPS)
        st1.capture()
        out["g1"] = _static(model, st1, ids, toks, ROWS * N_STEPS, graph=True)[0]
        st1.detach()
    return out


def _same_run(a, b, what):
    _assert_rows(a[0], b[0], f"{what}: logits")
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a[1], b[1])), f"{what}: main_hidden"
    assert set(a[2]) == set(b[2]) and all(torch.equal(a[2][k], b[2][k]) for k in b[2]), f"{what}: caches"


def test_pdl_modes_bit_identical_to_off_graph_replay_rows_census_timing():
    _need_tilelang()
    with _model("mini4") as (model, pool):
        ids, toks = _toks(17, ROWS * N_STEPS, 55)
        base = _run_mode(model, pool, ids, toks, "off", "entry", census=True)
        _same_run(base["g6"], base["e6"], "off: 6-row graph vs eager")
        rows_eq_off = sum(torch.equal(a, b) for a, b in zip(base["e6"][0], base["e1"]))
        RECEIPT["modes"]["off"] = {"ms": base["ms"], "rows6_eq_rows1": rows_eq_off}
        print(f"  {'mode':18s} {'6-row eager':>11s} {'graph':>6s} {'replay':>6s} {'1-row':>6s} {'1r graph':>8s} "
              f"{'6r==1r rows':>11s} {'ms/replay':>10s} {'delta':>8s}")
        print(f"  {'off':18s} {'ref':>11s} {'ok':>6s} {'-':>6s} {'ref':>6s} {'-':>8s} "
              f"{rows_eq_off:>5d}/{len(base['e1']):<5d} {base['ms']:10.4f} {'':>8s}")
        for mode, trig in MODES:
            r = _run_mode(model, pool, ids, toks, mode, trig, census=(trig == "entry"))
            label = f"{mode}+{trig}"
            _same_run(r["e6"], base["e6"], f"{label}: 6-row eager vs off")
            _same_run(r["g6"], r["e6"], f"{label}: 6-row graph vs eager")
            _same_run(r["g6b"], r["g6"], f"{label}: 6-row replay #2 vs #1")
            _assert_rows(r["e1"], base["e1"], f"{label}: 1-row eager vs off")
            _assert_rows(r["g1"], r["e1"], f"{label}: 1-row graph vs eager")
            rows_eq = sum(torch.equal(a, b) for a, b in zip(r["e6"][0], r["e1"]))
            assert rows_eq == rows_eq_off
            sw = r["switch"]
            assert sw["enabled"] and (mode != "prefetch" or sw["prefetch"]), sw
            launched = {k for k, v in sw["host_launches"].items() if v["pdl"]}
            RECEIPT["modes"][label] = {"ms": r["ms"], "delta_ms": r["ms"] - base["ms"], "rows6_eq_rows1": rows_eq,
                                       "pdl_kernels_launched": sorted(launched),
                                       "prefetch_kernels_launched": sorted(k for k, v in sw["host_launches"].items()
                                                                           if v["prefetch"])}
            print(f"  {label:18s} {'== off':>11s} {'ok':>6s} {'ok':>6s} {'== off':>6s} {'ok':>8s} "
                  f"{rows_eq:>5d}/{len(r['e1']):<5d} {r['ms']:10.4f} {r['ms'] - base['ms']:+8.4f}")
            if "census" in r:
                c = r["census"]
                RECEIPT["census"][label] = c
                from runtime import pdl

                pdl.print_census(c, tag=f" mini {label}")   # reported (bits are the gate): a borrowed library may
                # launch itself with PDL, and memset / memcpy producers need not form programmatic edges
        if "error" not in base.get("census", {"error": 1}):
            pdl_off = base["census"]
            print(f"  off graph: {pdl_off['nodes']} nodes, edges {pdl_off['edge_kinds']} (programmatic here = a borrowed "
                  f"library's own PDL)")
        print("  every PDL mode == off (6-row eager / graph / replay, 1-row eager / graph); MEASURED mini ms above")


if __name__ == "__main__":
    t0 = time.time()
    test_pdl_modes_bit_identical_to_off_graph_replay_rows_census_timing()
    os.makedirs(os.path.join(REPO, "scratch"), exist_ok=True)
    path = os.path.join(REPO, "scratch", time.strftime("pdl-gpu-%Y%m%d-%H%M%S.json", time.gmtime()))
    RECEIPT["wall_s"] = round(time.time() - t0, 1)
    with open(path, "w") as f:
        json.dump(RECEIPT, f, indent=1, default=str)
    print(f"receipt {path}")
