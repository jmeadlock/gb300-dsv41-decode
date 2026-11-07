"""Mac CPU tests of T9a-1 (runtime/dg_moe.py, `--moe-impl dg-masked`): `.venv/bin/python -m pytest -q tests/`.

What a CPU can check, and does:
  * e8m0 -> fp32: the exact table (2^(e-127), 2^-127 subnormal at 0, NaN at 255) == torch's own e8m0 decode ==
    the spike's exp2(e8m0 - 127); both device methods exact over all 256 codes, also into strided/broadcast rows;
  * the call contract constants (max_m 128, expected_m 8, recipe_a = recipe_b = (1, 32), masked_m = ones int32) and
    the static buffer shapes/dtypes, rows >= 1 of the activation buffers untouched by steps;
  * `import_deep_gemm`: imports from the given third_party dir with the dir on sys.path only for the import, refuses
    when any vllm module is loaded, reuses an already-imported deep_gemm;
  * the whole dg-masked StaticStep vs the reference forward on the mini checkpoints, with the reference kernels AND
    DeepGEMM emulated on CPU by the same math (tests/test_static_emulated_cpu.py's emulations + a contract-checking
    fake `m_grouped_fp8_fp4_gemm_nt_masked` that also NaN-poisons every output row >= masked_m): `torch.equal` on
    every position's fp32 logits. Under identical GEMM math, dg-masked must be bit-identical, so this pins everything
    T9a-1 wrote around the GEMM: staging reuse, scale conversion, row-0 placement, the one shared act_quant, the glue's
    op order and dtypes, the w2 requant, the slot-order fp32 sum, the shared expert;
  * Generator `moe_impl` plumbing and first_light's `--moe-impl` / parity-levels drift row on the fake model.
Skipped here (box only, tests/gpu/test_dg_moe_gpu.py): DeepGEMM itself, its JIT, CUDA-graph capture, real numerics.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types

import pytest
import torch

from runtime import dg_moe
from test_static_emulated_cpu import _check, _fp4_deq, _ref_run, _static_run, emu  # noqa: F401 (emu = fixture)
from test_static_emulated_cpu import act_quant as _emu_act_quant

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
E8 = torch.float8_e8m0fnu


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ e8m0 -> fp32
def test_e8m0_table_exact():
    t = dg_moe.e8m0_table()
    assert t.dtype == torch.float32 and t.shape == (256,)
    for e in (0, 1, 2, 100, 126, 127, 128, 200, 253, 254):
        assert float(t[e]) == math.ldexp(1.0, e - 127), e
    assert float(t[0]) == 2.0 ** -127 and float(t[0]) != 0.0  # fp32 subnormal, not flushed
    assert math.isnan(float(t[255]))
    u = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    torch_decode = u.view(E8).float()
    spike = torch.exp2(u[:255].to(torch.int32).float() - 127.0)  # bench/dg_masked_spike.py:55
    assert torch.equal(torch_decode[:255].view(torch.int32), t[:255].view(torch.int32))
    assert torch.equal(spike.view(torch.int32), t[:255].view(torch.int32))
    assert torch.isnan(torch_decode[255])


@pytest.mark.parametrize("method", ["cast", "exp2"])
def test_e8m0_to_f32_methods_exact_and_broadcast(method):
    u = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    out = torch.empty(256)
    dg_moe.e8m0_to_f32(u, out, method)
    assert dg_moe._bits_equal_table(out)
    dg_moe.e8m0_to_f32(u.view(E8), out.zero_(), method)  # e8m0-typed source too
    assert dg_moe._bits_equal_table(out)
    # one activation row broadcast into row 0 of every group of a [k, MAX_M, K/32] buffer (load_act's use)
    k, kb = 3, 8
    s = torch.randint(100, 140, (1, kb), dtype=torch.int32).to(torch.uint8).view(E8)
    buf = torch.ones(k, dg_moe.MAX_M, kb)
    dg_moe.e8m0_to_f32(s, buf[:, 0], method)
    assert torch.equal(buf[:, 0], s.float().expand(k, kb)) and torch.equal(buf[:, 1:], torch.ones(k, dg_moe.MAX_M - 1, kb))


def test_pick_e8m0_method_cpu():
    assert dg_moe.pick_e8m0_method("cpu") == "cast"


def test_contract_constants():
    from runtime import generate, static_step

    assert dg_moe.MAX_M == 128 and dg_moe.EXPECTED_M == 8
    assert dg_moe.RECIPE_A == (1, 32) and dg_moe.RECIPE_B == (1, 32)
    assert dg_moe.MOE_IMPLS == static_step.MOE_IMPLS == generate.MOE_IMPLS == ("ref", "dg-masked", "dg-inplace")


# ------------------------------------------------------------------------------------------------ deep_gemm import
def test_import_deep_gemm_path_vllm_guard_and_reuse(tmp_path, monkeypatch):
    pkg = tmp_path / "third_party" / "deep_gemm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("FAKE = 'tp'\nimport torch\n")
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    path = str(tmp_path / "third_party")
    before = list(sys.path)
    mod = dg_moe.import_deep_gemm(path)
    assert mod.FAKE == "tp" and mod.__file__.startswith(path) and sys.path == before  # path only for the import
    assert dg_moe.import_deep_gemm("/nonexistent") is mod  # already imported -> reused
    monkeypatch.delitem(sys.modules, "deep_gemm")
    with pytest.raises(ImportError):
        dg_moe.import_deep_gemm(str(tmp_path / "nowhere"))
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    with pytest.raises(RuntimeError, match="vllm"):
        dg_moe.import_deep_gemm(path)


# ------------------------------------------------------------------------------------------------ emulated end to end
def _fake_deep_gemm():
    """m_grouped_fp8_fp4_gemm_nt_masked emulated with the reference fp4_gemm emulation's math
    (test_static_emulated_cpu.fp4_gemm), per group on the masked_m valid rows; asserts the spike's call contract and
    NaN-poisons rows >= masked_m so any consumer of a padding row fails the logits comparison."""
    mod = types.ModuleType("deep_gemm")
    mod.__file__ = "<fake deep_gemm>"
    mod.calls = []
    mod.deterministic = []

    def use_deterministic_algorithms(flag=True):
        mod.deterministic.append(flag)

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe_a=None, recipe_b=None, **kw):
        (xq, xs), (W, S) = a, b
        G, max_m, K = xq.shape
        N = W.size(1)
        assert not kw and recipe_a == (1, 32) and recipe_b == (1, 32) and expected_m == 8
        assert max_m >= 128 and xq.dtype == torch.float8_e4m3fn and xs.dtype == torch.float32
        assert tuple(xs.shape) == (G, max_m, K // 32)
        assert W.dtype == torch.int8 and tuple(W.shape) == (G, N, K // 2) and W.is_contiguous()
        assert S.dtype == torch.float32 and tuple(S.shape) == (G, N, K // 32) and S.is_contiguous()
        assert d.dtype == torch.bfloat16 and tuple(d.shape) == (G, max_m, N)
        assert masked_m.dtype == torch.int32 and tuple(masked_m.shape) == (G,)
        for g in range(G):
            m = int(masked_m[g])
            A = (xq[g, :m].float().view(m, K // 32, 32) * xs[g, :m].view(m, -1, 1)).view(m, K)
            B = _fp4_deq(W[g]) * S[g].repeat_interleave(32, -1)
            d[g, :m] = (A @ B.t()).to(torch.bfloat16)
            d[g, m:] = float("nan")
        mod.calls.append((G, max_m, N, K, [int(v) for v in masked_m]))

    mod.m_grouped_fp8_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
    mod.use_deterministic_algorithms = use_deterministic_algorithms
    return mod


def _faithful_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """The emulated act_quant, returning scales in `scale_dtype` as kernel.py does (e8m0 here; exact for powers of 2).
    Used by both sides of every comparison."""
    r = _emu_act_quant(x, block_size, scale_fmt, scale_dtype, inplace)
    if inplace or scale_dtype != E8:
        return r
    q, s = r
    s8 = s.to(E8)
    assert torch.equal(s8.float(), s)
    return q, s8


@pytest.fixture
def dg_emu(emu, monkeypatch):
    fake = _fake_deep_gemm()
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)

    def build(kind):
        model, pool, fk = emu(kind)
        R = sys.modules["model"]
        monkeypatch.setattr(R, "act_quant", _faithful_act_quant)
        return model, pool, fk, fake

    return build


@pytest.mark.parametrize("kind", ["mini", "mini4"])
def test_dg_masked_static_step_equals_reference_under_emulated_gemm(dg_emu, kind):
    from runtime.static_step import StaticStep

    model, pool, fk, fake = dg_emu(kind)
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", _allow_cpu=True)
    b, k = st.dg, st.topk
    from runtime.dg_moe import DGMaskedMoE

    assert st.moe_impl == "dg-masked" and all(isinstance(m, DGMaskedMoE) for m in st.moes)
    assert fake.deterministic == [True] and b.deterministic["applied"] and b.e8m0_method == "cast"
    assert (b.dim, b.inter) == (256, 256)  # MINI_CONFIG dim / moe_inter_dim
    assert b.masked_m.dtype == torch.int32 and torch.equal(b.masked_m, torch.ones(k, dtype=torch.int32))
    assert b.xq[1].shape == (k, 128, b.dim) and b.xq[2].shape == (k, 128, b.inter)
    assert b.xq[1].dtype == torch.float8_e4m3fn and b.xs[1].shape == (k, 128, b.dim // 32)
    assert b.xs[2].shape == (k, 128, b.inter // 32) and b.xs[1].dtype == torch.float32
    for p in ("w1", "w3", "w2"):
        n, kh = st.staging[(p, "weight")].shape[1:]
        assert b.sf32[p].shape == (k, n, 2 * kh // 32) and b.d[p].shape == (k, 128, n)
    g = torch.Generator().manual_seed(11)
    n_steps = 0
    for L in (2, 3, 16, 17):  # both prefill parities; 10 decode steps cover both graph parities
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        ref = _ref_run(model, ids, 10)
        got = _static_run(model, st, ids, 10)
        _check(got, ref, (kind, "dg-masked emulated", L))
        n_steps += 10
    n_layers = len(model.layers)
    assert len(fake.calls) == 3 * n_layers * n_steps and fk.calls["gather"] == 6 * n_layers * n_steps
    assert all(c[1] == 128 and c[4] == [1] * k for c in fake.calls)
    for w in (1, 2):  # padding rows never written by a step
        assert not b.xq[w][:, 1:].view(torch.uint8).any() and torch.equal(b.xs[w][:, 1:], torch.ones_like(b.xs[w][:, 1:]))
    st.detach()


def test_dg_masked_cold_counters_and_ref_unchanged(dg_emu):
    """Counters work the same under dg-masked; a ref StaticStep built afterwards on the same model is still exact."""
    from runtime.static_step import StaticMoE, StaticStep

    model, pool, _, fake = dg_emu("mini4")
    ids = list(range(5, 21))
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", _allow_cpu=True)
    got = _static_run(model, st, ids, 12)
    cold_dg, n, k = st.cold_counts()
    st.detach()
    st2 = StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    assert st2.dg is None and all(type(m) is StaticMoE for m in st2.moes)
    n_calls = len(fake.calls)
    got2 = _static_run(model, st2, ids, 12)
    assert len(fake.calls) == n_calls  # ref never touches DeepGEMM
    _check(got2, _ref_run(model, ids, 12), "ref after dg")
    _check(got, got2, "dg vs ref (emulated)")
    assert st2.cold_counts()[0] == cold_dg and n == 12 and k == 4 and sum(cold_dg) > 0
    st2.detach()
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=127, moe_impl="dg-mega", _allow_cpu=True)


# ------------------------------------------------------------------------------------------------ Generator + first_light
class FakeStaticImpl:
    """tests/test_static_cpu.FakeStatic with a moe_impl: "dg-masked" adds a fixed tiny offset to logit 5 at every
    decode step (deterministic, the same eagerly and in the graph) -> L0 differs, L1/L2 equal."""

    built = []

    def __init__(self, model, moe_impl="ref"):
        self.model, self.moe_impl = model, moe_impl
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures = {}, []
        FakeStaticImpl.built.append(moe_impl)

    @property
    def captured(self):
        return bool(self.graphs)

    def capture(self, fills=False):
        self.captures.append(bool(fills))
        self.graphs = {0: "g_even", 1: "g_odd"}
        return {"fills": fills}

    def step(self, token, pos, graph):
        t = int(token.reshape(-1)[0]) if isinstance(token, torch.Tensor) else int(token)
        ids, logits, _ = self.model(torch.tensor([[t]], dtype=torch.long), pos)
        if self.moe_impl != "ref":
            logits = logits.clone()
            logits[0, 5] += 1e-3
            ids = logits.argmax(dim=-1)
        self.tok.copy_(ids.view(1, 1))
        return logits

    def detach(self):
        self.graphs = {}

    def cold_counts(self):
        return None


def _gen(mode="eager", moe_impl="ref"):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStaticImpl.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=FakeStaticImpl, moe_impl=moe_impl)


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()][:n]


def test_generator_moe_impl_plumbing():
    gen = _gen(mode="static", moe_impl="dg-masked")
    gen.greedy([5, 9, 12, 3], 4)
    assert gen.static.moe_impl == "dg-masked" and gen.last_meta["moe_impl"] == "dg-masked"
    old = gen.static
    gen.set_moe_impl("dg-masked")
    assert gen.static is old  # no change, no rebuild
    gen.set_moe_impl("ref")
    assert gen.static is None and old.graphs == {}
    gen.greedy([5, 9, 12, 3], 4)
    assert gen.static.moe_impl == "ref" and FakeStaticImpl.built == ["dg-masked", "ref"]
    with pytest.raises(ValueError):
        gen.set_moe_impl("dg-mega")
    seen = []
    gen.logits_sink = lambda i, lc: seen.append((i, lc.shape))
    gen.greedy([5, 9, 12, 3], 4, stop_on_eos=False)
    assert [i for i, _ in seen] == [0, 1, 2, 3]


def test_parity_levels_dg_masked_drift_row():
    fl = _first_light()
    gen = _gen(mode="graph", moe_impl="dg-masked")
    out = []
    det = torch.utils.deterministic
    fill0 = det.fill_uninitialized_memory
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    assert res["verdict"] == "PASS" and res["moe_impl"] == "dg-masked"
    for f in ("off", "on"):
        l0 = by[(f, "static-eager vs reference-eager")]
        assert not l0["gate"] and l0["equal"] == 0  # reported, not gated
        assert by[(f, "graph vs static-eager")]["all_equal"] and by[(f, "graph vs static-eager")]["gate"]
        assert by[(f, "graph relaunch vs replay#1")]["all_equal"]
    t8 = by[("off", "ref static-eager vs reference-eager")]
    assert t8["gate"] and t8["all_equal"] and t8["n"] == 3
    dr = by[("off", "dg-masked vs ref static-eager")]
    assert not dr["gate"] and dr["equal"] == 0 and dr["positions_compared"] > 0
    # the fake adds exactly 1e-3 to one of 64 logits at every decode position (fp32: up to 1 ulp off)
    assert dr["mean_abs_dlogit"] == pytest.approx(1e-3 / 64, rel=1e-3) and dr["max_abs_dlogit"] == pytest.approx(1e-3, rel=1e-3)
    assert res["drift"]["mean_abs_dlogit"] == dr["mean_abs_dlogit"]
    assert all("first_token_divergence" in v for v in dr["per_prompt"].values())
    assert "static-ref/fills-off" in res["runs"]
    assert any(l.startswith("drift dg-masked vs ref static-eager: first token divergence") for l in out)
    assert any(l.startswith("PARITY LEVELS (4 steps x 3 prompts") and l.endswith("PASS") for l in out)
    assert gen.moe_impl == "dg-masked" and gen.mode == "graph" and gen.static.moe_impl == "dg-masked"
    assert gen.static.captures[-1] is False and det.fill_uninitialized_memory == fill0
    assert gen.logits_sink is None


def test_parity_levels_ref_unchanged_shape():
    fl = _first_light()
    gen = _gen(mode="graph")
    res = fl.parity_levels(gen, _rows(2), steps=3, log=lambda *a, **k: None)
    assert res["verdict"] == "PASS" and res["drift"] is None and len(res["rows"]) == 4 + 4 + 3
    assert all(r["gate"] == (r["what"] != "eager: fills-on vs fills-off") for r in res["rows"])
    assert FakeStaticImpl.built and set(FakeStaticImpl.built) == {"ref"}


def test_logit_drift_first_divergence_and_same_context_window():
    fl = _first_light()
    z = [torch.zeros(4) for _ in range(5)]
    b = [torch.zeros(4), torch.full((4,), 0.5), torch.full((4,), 1.0), torch.full((4,), 9.0), torch.full((4,), 9.0)]
    a_run = {"t": {"tokens": [1, 2, 3, 4, 5]}}
    b_run = {"t": {"tokens": [1, 2, 7, 4, 5]}}  # diverges at generated index 2: positions 1..2 compared
    d = fl.logit_drift({"t": z}, {"t": b}, a_run, b_run)
    assert d["first_token_divergence"] == "t@2" and d["positions_compared"] == 2
    assert d["mean_abs_dlogit"] == pytest.approx(0.75) and d["max_abs_dlogit"] == 1.0
    assert d["prompts_token_identical"] == 0


def test_first_light_moe_impl_flag_and_report(tmp_path, capsys):
    fl = _first_light()
    assert fl.build_parser().parse_args(["--model-dir", "x"]).moe_impl == "ref"
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--moe-impl", "dg-masked",
                                      "--parity-levels", "--level-steps", "2", "--fixture", FIXTURE, "--only",
                                      "prose", "--reps", "1", "--max-new", "4", "--step-cell", "6", "--stats-from",
                                      "2", "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    assert a.moe_impl == "dg-masked"
    with pytest.raises(SystemExit):
        fl.build_parser().parse_args(["--model-dir", "x", "--moe-impl", "dg-mega"])
    gen = _gen(mode="graph", moe_impl=a.moe_impl)
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["generator"]["moe_impl"] == "dg-masked" and report["step_cell"]["moe_impl"] == "dg-masked"
    assert report["parity_levels"]["verdict"] == "PASS" and report["parity_levels"]["drift"] is not None
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | mode graph moe dg-masked levels PASS")
