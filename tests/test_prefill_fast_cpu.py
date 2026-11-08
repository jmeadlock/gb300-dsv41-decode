"""Mac tests of M4-P2 (`.venv/bin/python -m pytest -q tests/test_prefill_fast_cpu.py`): runtime/prefill_fast.py, the
prefill_hidden / DSparkLoop.start call sites, and bench/prefill_bench.py's pure helpers.

The TileLang kernels do not run on the Mac: both sides of every comparison run on the CPU kernel emulations
(tests/test_static_emulated_cpu.py KERNELS; fp4/fp8 GEMMs evaluated row by row = row-invariant like the box kernels,
M1-T10 census; hc_split_sinkhorn per token). Kernel fidelity is not what is tested -- the box GPU test does that with
the real kernels; what is tested is everything M4-P2 changed around them: the stable-sort routing, the 32-aligned
padded segments, the single gather / act_quant, the per-expert GEMM launches over views (both the wrapper path and
the direct-kernel path, the latter with a fake `kernel` module whose kernel also scribbles NaN over its tile's
unguarded tail rows), the SwiGLU on all rows, the ascending-id fp32 combine, the scope's install / uninstall, the
counters -- and tests/gpu/test_prefill_fast_gpu.py's gates dry-run on these emulations (DEV "cpu").
"""
from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import sys
import types

import pytest
import torch

from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
from mini_mtp import make_mini_checkpoint_mtp, mtp_config_kind
from runtime.loader import build_and_load
from runtime.refmodel import import_reference
from test_dspark_loop_cpu import _per_token
from test_static_emulated_cpu import KERNELS, _fake_static_kernels, _mini4
from test_static_rows_cpu import _rowwise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


@pytest.fixture
def emu(monkeypatch, tmp_path):
    """-> build(kind) -> (model, pool) on the CPU emulations: "mini4" (4 layers, top-4 of 8, no mtp) or "mini3"
    (mtp schema, 3 draft stages, load_mtp=True, fake static gather kernels for the loop)."""
    R = import_reference(cpu_stubs=True)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: ([i % 97 for i in range(MINI_CONFIG["vocab_size"])],
                                     MINI_CONFIG["engram_compressed_vocab_size"]))
    for k, f in KERNELS.items():
        monkeypatch.setattr(R, k, f)
    monkeypatch.setattr(R, "fp4_gemm", _rowwise(KERNELS["fp4_gemm"]))
    monkeypatch.setattr(R, "fp8_gemm", _rowwise(KERNELS["fp8_gemm"]))
    monkeypatch.setattr(R, "hc_split_sinkhorn", _per_token(KERNELS["hc_split_sinkhorn"]))
    prev = torch.get_default_dtype()
    count = itertools.count()

    def build(kind="mini4"):
        d = str(tmp_path / f"{kind}-{next(count)}")
        if kind == "mini4":
            cfg, mp = _mini4()
            ck = make_mini_checkpoint(d, cfg=cfg)
            mtp = False
        else:
            ck = make_mini_checkpoint_mtp(d, cfg=mtp_config_kind(kind))
            mp, mtp = ck["map"], True
        model = build_and_load(ck["model_dir"], mp, "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                               load_mtp=mtp, log=lambda *a: None)
        model.temperature = 0.0
        pool = model.runtime_load.pool
        if mtp:
            reg = {}
            for l in range(pool.n_layers):
                for p in ("w1", "w2", "w3"):
                    w, s = pool.hbm[l][p]
                    for t in list(w) + list(s):
                        reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
                    for e in pool.cold[l]:
                        for kind_ in ("weight", "scale"):
                            t = pool.cold_view(l, e, p, kind_, cuda=False)
                            reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
            monkeypatch.setitem(sys.modules, "runtime.static_kernels", _fake_static_kernels(reg))
        return model, pool

    yield build
    torch.set_default_dtype(prev)
    R.MoE.forward = getattr(R.MoE, "_orig_forward", R.MoE.forward)


def _gpu_module(monkeypatch, emu, name="gpu_prefill_fast"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "tests", "gpu", "test_prefill_fast_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    @contextlib.contextmanager
    def model_cpu(kind="mini3"):
        yield emu(kind)

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STEP_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "LOOP_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    return G


def _ids(n, seed=3, hi=2000):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [int(x) for x in torch.randint(3, hi, (n,), generator=g, device="cpu")]


def _capture_moe_inputs(model, pool, ids):
    import prefill_anatomy as PA

    probe = PA.PrefillProbe(model, pool, capture_moe=range(len(model.layers)))
    with probe:
        PA.prefill_once(model, ids, (), seed=False, device="cpu")
    return probe.moe_inputs


# ------------------------------------------------------------------------------------------------ switches
def test_switches_defaults_env_and_refusals(monkeypatch):
    from runtime import prefill_fast as PF

    for k in (PF.PREFILL_ENV, PF.FILLS_ENV, PF.STREAMS_ENV):
        monkeypatch.delenv(k, raising=False)
    assert PF.prefill_mode() == "grouped"              # default since the 11/08 box gate (bit-equal 8/8, TTFT 3.9 s -> <1 s)
    assert PF.prefill_fills() is False                 # fills off by default (box: bit-equal, -0.55 s)
    assert PF.prefill_streams() == PF.DEFAULT_STREAMS
    monkeypatch.setenv(PF.PREFILL_ENV, "Grouped")
    monkeypatch.setenv(PF.FILLS_ENV, "on")
    monkeypatch.setenv(PF.STREAMS_ENV, "2")
    assert (PF.prefill_mode(), PF.prefill_fills(), PF.prefill_streams()) == ("grouped", True, 2)
    assert PF.prefill_mode("reference") == "reference"
    assert PF.describe() == {"mode": "grouped", "fills": "on", "streams": 2}
    for bad in (lambda: PF.prefill_mode("fast"), lambda: PF.prefill_fills("maybe"), lambda: PF.prefill_streams(0),
                lambda: PF.prefill_streams(PF.MAX_STREAMS + 1)):
        with pytest.raises(ValueError):
            bad()


def test_fills_scope_sets_and_restores():
    from runtime.prefill_fast import fills_scope

    det = torch.utils.deterministic
    prev = det.fill_uninitialized_memory
    try:
        det.fill_uninitialized_memory = True
        with fills_scope(False):
            assert det.fill_uninitialized_memory is False
        assert det.fill_uninitialized_memory is True
        with pytest.raises(RuntimeError):
            with fills_scope(False):
                raise RuntimeError("x")
        assert det.fill_uninitialized_memory is True
    finally:
        det.fill_uninitialized_memory = prev


# ------------------------------------------------------------------------------------------------ the MoE
@pytest.mark.parametrize("kind,L", [("mini4", 23), ("mini4", 1), ("mini4", 77), ("mini3", 41)])
def test_grouped_moe_equal_reference_wrapper_path(emu, kind, L):
    """Per layer, the captured prefill input: reference MoE.forward == grouped_moe_forward (torch.equal); counters."""
    from runtime.prefill_fast import grouped_moe_forward

    model, pool = emu(kind)
    xs = _capture_moe_inputs(model, pool, _ids(L, seed=L))
    with torch.inference_mode():
        for l, x in xs.items():
            moe = model.layers[l].ffn
            st = {}
            got = grouped_moe_forward(moe, x, streams=4, stats=st, pool=pool)
            assert torch.equal(moe(x), got), f"{kind} L={L} layer {l}"
            _, idx = moe.gate(x.view(-1, moe.dim), None)
            used = set(int(e) for e in idx.flatten())
            assert st["expert_calls"] == len(used) and st["gemm_launches"] == 3 * len(used)
            assert st["cold_calls"] == len(used & set(pool.cold[l]))
            assert st["pairs"] == L * moe.n_activated_experts
            assert st["padded_rows"] % 32 == 0 and st["direct_kernel"] is False and st["streams_used"] == 1


def _fake_kernel_module(rowwise_fp4, calls):
    """A stand-in for kernel.py exposing `fp4_gemm` (the wrapper model.py calls) and `fp4_gemm_kernel` (the TileLang
    factory). The kernel object writes C = A @ B^T like the wrapper, records (M, row offset of A in its buffer), and
    then scribbles NaN over the rows [M, ceil(M/32)*32) of C's storage -- an unguarded tail-tile store, which the
    32-aligned, 32-padded segments must absorb."""
    mod = types.ModuleType("kernel")
    mod.FE8M0, mod.FP32 = "float8_e8m0fnu", "float32"
    mod.fp4_gemm = rowwise_fp4
    built = []

    def fp4_gemm_kernel(N, K, act_block_size=128, out_dtype="bfloat16", accum_dtype="float32", scale_dtype="float32"):
        built.append((N, K, act_block_size, scale_dtype))
        sdt = torch.float8_e8m0fnu if scale_dtype == mod.FE8M0 else torch.float32

        def kern(A, B, C, SA, SB):
            M = A.size(0)
            assert A.shape == (M, K) and C.shape == (M, N) and SA.size(0) == M and B.size(0) == N
            assert A.is_contiguous() and C.is_contiguous() and SA.is_contiguous()
            calls.append({"M": M, "a_off": A.storage_offset() // K, "c_off": C.storage_offset() // N, "N": N})
            C.copy_(rowwise_fp4(A, SA, B, SB, sdt, act_block_size=act_block_size))
            pad = (M + 31) // 32 * 32
            if pad > M:
                C.as_strided((pad, N), C.stride())[M:].fill_(float("nan"))
        return kern

    mod.fp4_gemm_kernel = fp4_gemm_kernel
    mod.built = built
    return mod


@pytest.mark.parametrize("kind,L", [("mini4", 37), ("mini3", 19)])
def test_grouped_moe_direct_kernel_path(emu, monkeypatch, kind, L):
    """The box path: `model.fp4_gemm is kernel.fp4_gemm` -> the kernel object is called directly on segment views
    (M = the expert's own row count, A / C views at 32-aligned row offsets); NaN scribbled over every tile's tail
    rows changes nothing; the kernel objects are built once per (N, K)."""
    from runtime import prefill_fast as PF

    R = import_reference(cpu_stubs=True)
    model, pool = emu(kind)
    xs = _capture_moe_inputs(model, pool, _ids(L, seed=100 + L))
    calls = []
    fake = _fake_kernel_module(R.fp4_gemm, calls)
    monkeypatch.setitem(sys.modules, "kernel", fake)
    monkeypatch.setattr(R, "fp4_gemm", fake.fp4_gemm)
    monkeypatch.setattr(PF, "_KERNELS", {})
    assert PF._direct_kernel_module() is fake
    with torch.inference_mode():
        for l, x in xs.items():
            moe = model.layers[l].ffn
            calls.clear()
            st = {}
            got = PF.grouped_moe_forward(moe, x, streams=1, stats=st, pool=pool)
            assert st["direct_kernel"] is True
            assert torch.equal(moe(x), got), f"{kind} L={L} layer {l}"
            assert not torch.isnan(got).any()
            _, idx = moe.gate(x.view(-1, moe.dim), None)
            cnt = torch.bincount(idx.flatten(), minlength=moe.n_routed_experts).tolist()
            want = [c for c in cnt if c]
            assert len(calls) == 3 * len(want) == st["gemm_launches"]
            # w1, w3 per expert (same A segment), then w2 per expert: M = c, offsets 32-aligned and increasing
            w13, w2 = calls[:2 * len(want)], calls[2 * len(want):]
            assert [c["M"] for c in w13[::2]] == want == [c["M"] for c in w2]
            offs = [c["a_off"] for c in w2]
            assert all(o % 32 == 0 for o in offs) and offs == sorted(offs) and len(set(offs)) == len(offs)
            assert all(c["a_off"] == c["c_off"] for c in calls)
    D, I = model.layers[0].ffn.dim, model.layers[0].ffn.experts[0].w1.weight.size(0)
    assert len(fake.built) == len({(I, D), (D, I)})   # w1/w3 (N=inter, K=dim), w2 (N=dim, K=inter): cached


def test_grouped_moe_falls_back(emu, monkeypatch):
    """image_mask / world_size > 1 -> the class forward (recorded); the result is the reference's."""
    from runtime import prefill_fast as PF

    R = import_reference(cpu_stubs=True)
    model, pool = emu("mini4")
    xs = _capture_moe_inputs(model, pool, _ids(9, seed=4))
    moe, x = model.layers[0].ffn, xs[0]
    with torch.inference_mode():
        st = {}
        got = PF.grouped_moe_forward(moe, x, torch.zeros(x.shape[:-1], dtype=torch.bool), stats=st)
        assert torch.equal(got, moe(x)) and st["fallback"][0]["why"] == "image_mask" and "layers" not in st
        monkeypatch.setattr(R, "world_size", 2)
        assert PF._fallback_reason(moe, None) == "world_size"


# ------------------------------------------------------------------------------------------------ the scope
def test_prefill_scope_installs_and_removes(emu, monkeypatch):
    from runtime import patches
    from runtime import prefill_fast as PF

    R = import_reference(cpu_stubs=True)
    model, pool = emu("mini4")
    patches.static_moe(True)                         # the class-level patch a StaticStep installs
    cls_fwd = R.MoE.forward
    with PF.prefill_scope(model, "grouped") as st:
        assert PF.grouped_installed(model) and st["mode"] == "grouped"
        assert R.MoE.forward is cls_fwd
    assert not PF.grouped_installed(model) and R.MoE.forward is cls_fwd
    assert all("forward" not in l.ffn.__dict__ for l in model.layers)
    with pytest.raises(RuntimeError):
        with PF.prefill_scope(model, "grouped"):
            raise RuntimeError("boom")
    assert not PF.grouped_installed(model)
    with PF.prefill_scope(model, "reference"):
        assert not PF.grouped_installed(model)
    assert PF.last_prefill()["mode"] == "reference"
    # a pre-existing instance forward survives the scope
    sentinel = lambda x, image_mask=None: x                                    # noqa: E731
    model.layers[0].ffn.forward = sentinel
    with PF.prefill_scope(model, "grouped"):
        assert model.layers[0].ffn.forward is not sentinel
    assert model.layers[0].ffn.forward is sentinel
    del model.layers[0].ffn.forward


def test_prefill_hidden_modes_equal_and_counters(emu, monkeypatch):
    """prefill_hidden: reference vs grouped (arg and env): ids, logits, main_hidden, every layer buffer equal; the
    counters; DSV41_PREFILL picks the mode when no arg is given."""
    from runtime.draft_step import prefill_hidden
    from runtime.prefill_fast import PREFILL_ENV, last_prefill

    model, pool = emu("mini3")
    hidden = tuple(int(i) for i in model.runtime_mtp.target_layer_ids)
    tok = torch.tensor([_ids(33, seed=8)], dtype=torch.long, device="cpu")
    out = []
    for mode, env in (("reference", None), (None, "grouped"), ("grouped", "reference")):
        if env is None:
            monkeypatch.delenv(PREFILL_ENV, raising=False)
        else:
            monkeypatch.setenv(PREFILL_ENV, env)
        o, lg, mh = prefill_hidden(model, tok, hidden, mode=mode)
        out.append((o.clone(), lg.clone(), mh.clone(), {n: b.clone() for n, b in model.layers.named_buffers()},
                    last_prefill()))
    a = out[0]
    assert a[4]["mode"] == "reference" and out[1][4]["mode"] == "grouped" and out[2][4]["mode"] == "grouped"
    for b in out[1:]:
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        assert all(torch.equal(a[3][n], b[3][n]) for n in a[3])
    st = out[1][4]
    assert st["layers"] == len(model.layers) and st["gemm_launches"] == 3 * st["expert_calls"]
    assert st["fills"] == "off" and "c2c_bytes" in st and st["c2c_bytes"] >= st["cold_calls"] * pool.expert_bytes
    assert not any("forward" in l.ffn.__dict__ for l in model.layers)


# ------------------------------------------------------------------------------------------------ GPU gates, dry run
def test_gpu_moe_layers_gate_dry_run_on_cpu(emu, monkeypatch, capsys):
    G = _gpu_module(monkeypatch, emu)
    n = G._gate_moe_layers("mini4", 29) + G._gate_moe_layers("mini3", 61)
    assert n == 2 * (4 + 2)
    assert "grouped MoE == reference" in capsys.readouterr().out


def test_gpu_prefill_gate_dry_run_on_cpu(emu, monkeypatch, capsys):
    G = _gpu_module(monkeypatch, emu)
    G._gate_prefill("mini4", 37)
    G._gate_prefill("mini3", 130, streams=1)      # past the 128 window
    out = capsys.readouterr().out
    assert out.count("== reference (fills on)") == 2 and G.COUNTS["buffers_equal"] > 0, out


def test_gpu_stream_gate_dry_run_on_cpu(emu, monkeypatch, capsys):
    """The loop's stream, accept lengths and every emitted row's logits after a grouped prefill == after the
    reference prefill; DSparkLoop.start records the prefill mode."""
    G = _gpu_module(monkeypatch, emu)
    res = G._gate_stream("mini3", 17, 32, mode="static")
    assert res["grouped"][0]["lp"]["gemm_launches"] == 3 * res["grouped"][0]["lp"]["expert_calls"] > 0
    assert "after grouped prefill == after reference prefill" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the bench
def test_bench_model_and_box_lines():
    import prefill_bench as PB

    rep = PB.model_report()
    assert rep["prompts"] and all(p["L"] for p in rep["prompts"].values())
    for p in rep["prompts"].values():
        t = p["ttft_ms"]
        assert t["today_fills_on"] > t["fills_off"] > t["grouped_s1"] >= t["grouped_s4"] > 0
        assert p["launches"]["expert_before"] == 26 * p["expert_calls"]
        assert p["launches"]["gemm_after"] == 3 * p["expert_calls"]
    lines = PB.model_lines(rep)
    assert any("[E]" in l for l in lines) and any("[D]" in l for l in lines)
    box = "\n".join(PB.box_lines())
    assert "DSV41_PREFILL=grouped" in box and "--residency-ref" in box and "prefill_bench.py run" in box
    assert "test_prefill_fast_gpu.py" in box
    assert PB.main(["box"]) == 0 and PB.main(["model"]) == 0


def test_bench_run_smoke_on_emulated_mini(emu, monkeypatch, tmp_path):
    """bench/prefill_bench.py run end to end on the CPU (boot / tokenizer / determinism stubbed): reference vs grouped
    per prompt, equality of first / logits / main_hidden / caches, replay, counters, the receipt."""
    import json

    import prefill_bench as PB
    import runtime.generate
    import runtime.loader

    model, pool = emu("mini3")
    monkeypatch.setattr(runtime.loader, "build_and_load", lambda *a, **k: model)
    monkeypatch.setattr(runtime.generate, "set_determinism", lambda *a, **k: {"stub": True})

    class FakeGen:
        def __init__(self, *a, **k):
            pass

        def encode(self, msgs):
            return _ids(11 + len(msgs[0]["content"]) % 7, seed=len(msgs[0]["content"]))

    monkeypatch.setattr(runtime.generate, "Generator", FakeGen)
    fx = tmp_path / "fx.jsonl"
    fx.write_text("\n".join(json.dumps({"tag": t, "prompt": p}) for t, p in
                            [("prose", "write a poem"), ("code", "a function"), ("skip", "x")]) + "\n")
    out = tmp_path / "r.json"
    rc = PB.main(["run", "--model-dir", str(tmp_path), "--map", "", "--fixture", str(fx), "--only", "prose,code",
                  "--reps", "2", "--streams", "1,4", "--device", "cpu", "--json", str(out)])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["errors"] == [], rep["errors"]
    assert sorted(rep["prompts"]) == ["code", "prose"]
    for P in rep["prompts"].values():
        for s in ("1", "4"):
            g = P["grouped"][s]
            assert g["equal_reference"] is True and g["caches_equal"] is True and g["self_replay_equal"] is True
            assert g["counters"]["gemm_launches"] == 3 * g["counters"]["expert_calls"]
    assert rep["summary"]["all_equal"] is True
