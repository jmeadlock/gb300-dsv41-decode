"""Mac CPU tests of M2-L1 (docs/HANDOFF-M2-L1-MIXES-FORK.md): the mHC mixes forked off the step's stream
(runtime/static_step.py MixesFork, runtime/fused_glue_r2.py "M2-L1: mixes fork", $DSV41_MIXES off|fork|fused).

What the CPU can prove (the streams, the graph's branch and the timing are the box's: tests/gpu/test_mixes_fork_gpu.py,
bench/mixes_fork_bench.py gpu):
  * the switch: env parsing, read once per build, `mix_fixed` (fused = the T9b-5 order for the mixes only);
  * MixesFork's contract on the deferred emulation (`check`): outputs poisoned until join, inputs unchanged at join
    (a write-after-read raises), counts, engaged only inside the step's _active() scope;
  * plumbing: with the emulation on, a fork/fused step's logits are bit-identical to the off step (1 and 6 rows,
    fused glue and reference glue, fp32 cublas and fixed) -- so no consumer reads a branch output before its join and
    no main-path op writes a branch input before it; 2 forks and 2 joins per layer per step, nothing pending;
  * the emulation has teeth: a step whose hc_post skips the join gets NaN logits;
  * the launch census (bench/mixes_fork_bench.py census): 2 launches per sublayer (t9b5_mix_fp32 + the sinkhorn) move
    off the step's stream under fp32 fixed, 3 under cublas (mm, mul, sinkhorn); totals unchanged;
  * the graph-DOT analysis the box bench/test uses (forks, joins, longest path, off-path kernel names).
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from runtime import fused_glue_r2 as R2
from test_static_rows_cpu import _cpu_default_device, _prefill, _tf_tokens, emu, rows_dg, rows_glue  # noqa: F401


@pytest.fixture(autouse=True)
def _mixes_env(monkeypatch):
    # bench/mixes_fork_bench.py writes os.environ directly: `import` does setdefault(CUBLAS_WORKSPACE_CONFIG) and
    # main() assigns DSV41_FP32_IMPL=fixed (a bench-side leak; not monkeypatch-tracked). Restore the whole
    # environment after every test here, else later modules (test_residency_cpu) run with the fixed fp32 impl.
    import os

    saved = dict(os.environ)
    monkeypatch.delenv(R2.MIXES_ENV, raising=False)
    monkeypatch.delenv("DSV41_MIXES_PRIORITY", raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _run(model, st, ids, toks, n):
    """n teacher-forced static-eager steps after a prefill -> fp32 logit rows."""
    _prefill(model, ids)
    out = []
    with torch.inference_mode():
        for s in range(n):
            tk = toks[s * st.rows:(s + 1) * st.rows] if st.rows > 1 else toks[s]
            lg = st.step(tk, len(ids) + s * st.rows, graph=False).reshape(st.rows, -1)
            out += [lg[j].clone() for j in range(st.rows)]
    return out


def _bits_equal(a, b):
    return len(a) == len(b) and all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a, b))


# ------------------------------------------------------------------------------------------------ the switch
def test_mixes_switch_and_mix_fixed(monkeypatch):
    # SMALL-WINS: the default is the row-conditional `auto` (fork at 1 row, off otherwise); see test_small_wins_cpu.py
    assert R2.MIXES_DEFAULT == "auto" and R2.MIXES_IMPLS == ("off", "fork", "fused")
    assert R2.mixes_impl(1) == "fork" and R2.mixes_impl(6) == "off" and R2.mixes_impl() == "off"
    for v in R2.MIXES_IMPLS:                   # an explicit value wins at every row count
        monkeypatch.setenv(R2.MIXES_ENV, v)
        assert R2.mixes_impl() == v == R2.mixes_impl(1) == R2.mixes_impl(6)
    monkeypatch.setenv(R2.MIXES_ENV, "sideways")
    with pytest.raises(ValueError, match="DSV41_MIXES"):
        R2.mixes_impl()
    monkeypatch.setenv(R2.FP32_IMPL_ENV, "cublas")
    assert [R2.mix_fixed(i) for i in ("off", "fork", "fused")] == [False, False, True]
    monkeypatch.setenv(R2.FP32_IMPL_ENV, "fixed")
    assert all(R2.mix_fixed(i) for i in ("off", "fork", "fused"))


def test_mixes_fork_contract_on_the_deferred_emulation():
    from runtime.static_step import MixesFork

    with pytest.raises(ValueError):
        MixesFork("cpu", "sideways")
    with pytest.raises(ValueError):
        MixesFork("cpu", "fork", priority="urgent")
    off = MixesFork("cpu", "off")
    assert not off.engaged(types.SimpleNamespace(active=True))
    fk = MixesFork("cpu", "fork", check=True)
    assert not fk.engaged(types.SimpleNamespace(active=False)) and fk.engaged(types.SimpleNamespace(active=True))
    x = torch.arange(6, dtype=torch.float32, device="cpu")
    out = fk.run(lambda: (x * 2, x + 1), (x, None))
    assert fk.pending == 1 and all(torch.isnan(o).all() for o in out)       # poisoned until the join
    fk.join()
    assert fk.pending == 0 and torch.equal(out[0], x * 2) and torch.equal(out[1], x + 1)
    out = fk.run(lambda: (x * 3,), (x,))
    x.add_(1)                                                                # main writes a branch input: WAR
    with pytest.raises(RuntimeError, match="changed between the fork and the join"):
        fk.join()
    assert fk.forks == 2 and fk.joins == 2 and fk.pending == 0
    inline = MixesFork("cpu", "fork")                                        # census mode: no extra torch ops
    y = inline.run(lambda: (x * 2,), (x,))
    assert torch.equal(y[0], x * 2)
    inline.finish()
    assert inline.pending == 0
    fk.run(lambda: (x * 2,), (x,))
    fk.discard()
    assert fk.pending == 0
    d = fk.describe()
    assert d["impl"] == "fork" and d["stream"] == "inline (CPU)" and d["priority"] == "default"


# ------------------------------------------------------------------------------------------------ the step
def _counting_fixed_dot(monkeypatch):
    seen = {"mix": 0, "head": 0, "comp": 0}
    real = R2.launch_fixed_dot

    def w(*a, **k):
        seen[k["site"]] += 1
        return real(*a, **k)
    monkeypatch.setattr(R2, "launch_fixed_dot", w)
    return seen


def _build(model, pool, rows, impl, monkeypatch, **kw):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(R2.MIXES_ENV, impl)
    st = StaticStep(model, pool, max_pos=255, rows=rows, _allow_cpu=True, **kw)
    monkeypatch.delenv(R2.MIXES_ENV)        # read once at build: the env no longer matters
    st.mix_fork.check = True
    return st


@pytest.mark.parametrize("rows", [1, 6])
@pytest.mark.parametrize("fp32", ["cublas", "fixed"])
def test_fork_bit_identical_fused_glue(rows_glue, monkeypatch, rows, fp32):
    """THE CPU gate: the fork (and, under fp32 fixed, fused) step == the off step bit for bit, fused glue, the
    deferred emulation on (any early read / late write would show); fork/join counts; the same clusters on."""
    monkeypatch.setenv(R2.FP32_IMPL_ENV, fp32)
    model, pool, _, _ = rows_glue("mini4")
    L = len(model.layers)
    n = 2
    ids, toks = _tf_tokens(18, rows * n, 41 + rows)
    kw = dict(moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
    off = _build(model, pool, rows, "off", monkeypatch, **kw)
    base = _run(model, off, ids, toks, n)
    on_off = off.glue.describe()["on"]
    assert off.mix_fork.forks == 0 and off.describe()["mixes"]["stream"] == "main"
    off.detach()
    impls = ["fork", "fused"] if fp32 == "fixed" else ["fork"]
    for impl in impls:
        st = _build(model, pool, rows, impl, monkeypatch, **kw)
        assert st.glue.on("hc_norm") and st.glue.describe()["on"] == on_off, st.glue.describe()["clusters"]
        f0 = st.mix_fork.forks                           # the build's self-checks never fork
        assert f0 == 0
        got = _run(model, st, ids, toks, n)
        assert _bits_equal(got, base), f"{impl} rows {rows} fp32 {fp32}: logits differ from off"
        fk = st.mix_fork
        assert fk.forks == fk.joins == 2 * L * n and fk.pending == 0, fk.describe()
        assert st.describe()["mixes"]["impl"] == impl
        st.detach()


@pytest.mark.parametrize("rows", [1, 6])
def test_fused_under_cublas_takes_the_fixed_mixes_only(rows_glue, monkeypatch, rows):
    """`fused` with $DSV41_FP32_IMPL=cublas: the mixes go to the T9b-5 kernel (2 per layer per step), the head and the
    compressor stay cuBLAS; the glue's hc_norm self-check (oracle: fixed mixes) passes; logits finite."""
    monkeypatch.setenv(R2.FP32_IMPL_ENV, "cublas")
    model, pool, _, _ = rows_glue("mini4")
    L = len(model.layers)
    ids, toks = _tf_tokens(18, rows * 2, 5)
    st = _build(model, pool, rows, "fused", monkeypatch, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
    assert st.glue.on("hc_norm"), st.glue.describe()["clusters"]
    seen = _counting_fixed_dot(monkeypatch)
    got = _run(model, st, ids, toks, 2)
    assert seen == {"mix": 2 * L * 2, "head": 0, "comp": 0}, seen
    assert all(torch.isfinite(g).all() for g in got)
    st.detach()


@pytest.mark.parametrize("rows", [1, 6])
def test_fork_bit_identical_reference_glue(emu, monkeypatch, rows):
    """The reference-op call site (glue ref): the whole hc_mixes forks; == off bit for bit; counts."""
    model, pool, _ = emu("mini4")
    L = len(model.layers)
    ids, toks = _tf_tokens(17, rows * 2, 9 + rows)
    off = _build(model, pool, rows, "off", monkeypatch)
    base = _run(model, off, ids, toks, 2)
    off.detach()
    st = _build(model, pool, rows, "fork", monkeypatch)
    got = _run(model, st, ids, toks, 2)
    assert _bits_equal(got, base)
    assert st.mix_fork.forks == st.mix_fork.joins == 2 * L * 2 and st.mix_fork.pending == 0
    st.detach()


def test_emulation_catches_a_missing_join(rows_glue, monkeypatch):
    """Teeth: hc_post without the join reads the poisoned post/comb -> NaN reaches the main path (the logits, or an
    emulated kernel's check first), never the off step's values."""
    from runtime.static_step import StaticStep

    model, pool, _, _ = rows_glue("mini4")
    ids, toks = _tf_tokens(18, 2, 3)
    kw = dict(moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused")
    st = _build(model, pool, 1, "fork", monkeypatch, **kw)
    orig = StaticStep._hc_post

    def no_join(self, layer, x, residual, post, comb):
        j = self.mix_fork.join
        self.mix_fork.join = lambda: None
        try:
            return orig(self, layer, x, residual, post, comb)
        finally:
            del self.mix_fork.join
    monkeypatch.setattr(StaticStep, "_hc_post", no_join)
    try:
        got = _run(model, st, ids, toks, 1)
    except AssertionError as e:   # the CPU emulations' own checks (e8m0 scale round trip) trip on the NaN first
        assert "nan" in str(e)
    else:
        assert all(torch.isnan(g).any() for g in got)
    st.mix_fork.discard()
    st.detach()


# ------------------------------------------------------------------------------------------------ census, DOT
def _bench():
    import importlib.util
    import os

    if "mixes_fork_bench" in sys.modules:
        return sys.modules["mixes_fork_bench"]
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "mixes_fork_bench.py")
    spec = importlib.util.spec_from_file_location("mixes_fork_bench", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mixes_fork_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fp32,per_sub,names", [
    ("fixed", 2, {"K:hc_split_sinkhorn", "K:t9b5_mix_fp32"}),
    ("cublas", 3, {"K:hc_split_sinkhorn", "mm", "mul"})])
def test_census_moves_the_mixes_off_the_path(fp32, per_sub, names, tmp_path):
    """In a subprocess (as tests/test_glue_census_cpu.py: the census swaps runtime.static_kernels / deep_gemm in
    sys.modules for CPU fakes)."""
    import json
    import os
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    js = tmp_path / "census.json"
    r = subprocess.run([sys.executable, os.path.join(repo, "bench", "mixes_fork_bench.py"), "census", "--rows", "1",
                        "--steps", "1", "--fp32-impl", fp32, "--json", str(js)],
                       cwd=repo, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    rep = json.loads(js.read_text())
    assert rep["fp32_impl"] == fp32
    r = rep["rows"]["1"]
    s = r["summary"]
    assert s["sublayers"] == 80 and s["total_launches_equal"]
    assert s["moved_off_path"] == 80 * per_sub and s["branch_launches_per_sublayer"] == per_sub
    assert set(r["fork"]["branch_ops_per_step"]) == names and r["off"]["branch_per_step"] == 0


_DOT_CHAIN = '''digraph dot {
subgraph cluster_1 {
label="graph_1" graph[style="dashed"];
"graph_1_node_0"[style="solid" shape="rectangle" label="0
KERNEL
t9b2r2_hc_norm
"];
"graph_1_node_1"[style="solid" shape="rectangle" label="1
KERNEL
t9b5_mix_fp32
"];
"graph_1_node_2"[style="solid" shape="rectangle" label="2
KERNEL
hc_split_sinkhorn_kernel__kernel
"];
"graph_1_node_3"[style="solid" shape="rectangle" label="3
KERNEL
t9b_mxfp8_gemv
"];
"graph_1_node_4"[style="solid" shape="rectangle" label="4
KERNEL
t9b2_hc_post
"];
"graph_1_node_0" -> "graph_1_node_1";
"graph_1_node_1" -> "graph_1_node_2";
"graph_1_node_2" -> "graph_1_node_3";
"graph_1_node_3" -> "graph_1_node_4";
}
}
'''


def test_dot_analysis_chain_and_branch():
    B = _bench()
    d = B.parse_dot(_DOT_CHAIN)
    assert len(d["nodes"]) == 5 and len(d["edges"]) == 4 and "t9b5_mix_fp32" in d["nodes"]["graph_1_node_1"]
    a = B.analyze_dag(d["nodes"], d["edges"])
    assert (a["nodes"], a["longest_path"], a["forks"], a["joins"], a["off_path"]) == (5, 5, 0, 0, 0)
    # the forked capture: hc_norm -> {gemv (main), mix -> sinkhorn (branch)} -> hc_post; plus a second gemv on main
    forked = _DOT_CHAIN.replace('"graph_1_node_0" -> "graph_1_node_1";\n"graph_1_node_1" -> "graph_1_node_2";\n'
                                '"graph_1_node_2" -> "graph_1_node_3";\n"graph_1_node_3" -> "graph_1_node_4";\n',
                                '"graph_1_node_0" -> "graph_1_node_1";\n"graph_1_node_1" -> "graph_1_node_2";\n'
                                '"graph_1_node_0" -> "graph_1_node_3";\n"graph_1_node_3" -> "graph_1_node_5";\n'
                                '"graph_1_node_5" -> "graph_1_node_4";\n"graph_1_node_2" -> "graph_1_node_4";\n')
    forked = forked.replace('"graph_1_node_0" -> ', '"graph_1_node_5"[label="5\nKERNEL\nt9b_mxfp8_gemv\n"];\n'
                            '"graph_1_node_0" -> ', 1)
    d = B.parse_dot(forked)
    a = B.analyze_dag(d["nodes"], d["edges"])
    assert (a["nodes"], a["forks"], a["joins"]) == (6, 1, 1)
    assert a["longest_path"] == 4 and a["off_path"] == 2                    # hc_norm, gemv, gemv, hc_post
    assert a["off_path_names"] == {"t9b5_mix_fp32": 1, "sinkhorn": 1}
    with pytest.raises(ValueError, match="cycle"):
        B.analyze_dag({"a": "", "b": ""}, [("a", "b"), ("b", "a")])


def test_modeled_box_numbers_come_from_the_receipts():
    B = _bench()
    m = B.modeled_box_ms()
    per1 = B.RECEIPT_US["t9b5_mix_fp32 R=1"] + B.RECEIPT_US["hc_split_sinkhorn R=1"]
    assert m["rows 1"]["ms_hi"] == round(80 * per1 / 1e3, 3) and m["rows 1"]["ms_lo"] == round(48 * per1 / 1e3, 3)
    assert "[E]" in m["basis"]


def test_bench_gpu_report_lines_and_refusal_on_the_mac(capsys):
    """The box mode's report formatting on a synthetic record (catches a KeyError before the box does); without CUDA
    the gpu mode refuses with a reason."""
    B = _bench()
    dag = {"nodes": 10, "edges": 9, "longest_path": 10, "forks": 0, "joins": 0, "off_path": 0, "off_path_names": {}}
    ident = {"graph_vs_eager_rows_differing": 0, "replay2_vs_1_rows_differing": 0, "rows_compared": 24,
             "eager_vs_off_rows_differing": 0, "graph_vs_off_rows_differing": 0}
    rec = {"sublayers": 8, "identity": {"off": dict(ident), "fork": dict(ident)},
           "graph": {"off": [dag], "fork": [{"error": "debug dump unavailable"}]},
           "ms_per_step": {"off": {"median": 1.0, "min": 0.9}, "fork": {"median": 0.95, "min": 0.9}},
           "saved_us_per_step": {"fork": 50.0},
           "contention": {"gemv_mean_us_off": 9.0, "gemv_mean_us_fork": 9.1, "gemv_ratio": 1.011, "other_ratio": 1.0}}
    lines = []
    B._log_rows("mini4", 6, rec, ["off", "fork"], lines.append)
    text = "\n".join(lines)
    assert "saved +50.0 us/step" in text and "debug dump unavailable" in text and "x1.011" in text
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="CUDA"):
            B.main(["gpu", "--kinds", "mini4"])


def test_gpu_test_module_imports_and_skips_without_cuda():
    import importlib.util
    import os

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "test_mixes_fork_gpu.py")
    spec = importlib.util.spec_from_file_location("test_mixes_fork_gpu_dry", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TESTS
    if not torch.cuda.is_available():
        import unittest

        with pytest.raises(unittest.SkipTest):
            mod.TESTS[0]()
