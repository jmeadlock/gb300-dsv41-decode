"""Mac CPU tests of T9b-3c (`.venv/bin/python -m pytest -q tests/test_attn_flashmla_cpu.py`).

FlashMLA's .so and an SM100 GPU exist only on the box, so this file checks what is built AROUND the op call:

  1. The per-row length: `topk_length_static` / `host_topk_length` == the width the eager reference passes,
     128 + min(512, T/r), for every position 1..4,096 at both ratios, and ceil(len / 64) == T9b-3a's block count.
  2. On the static step's real index lists (window + fixed-width top-k with -inf masking, ties): the length is exact
     -- every slot before it in the compressed part is valid, every slot from it on is -1 -- so the kernel's
     "only indices[:topk_length]" drops nothing.
  3. The argument construction (`prepare`): real shapes go through with no copy and no padding; mini shapes pad heads
     to 64, head dim to 512, topk to a multiple of 128 with -1, the sink with 0; b > 1 offsets each batch's indices.
  4. `sparse_attn_flashmla` through a torch emulation of sparse_prefill_fwd's contract == the fp64 oracle (padding is
     value-neutral), the length-limited call == the -1-masked call, rows independent.
  5. `load_ops` guards: vllm already imported / no .so / a load that pulls vllm in / op missing -> refused; the module
     never imports vllm.
  6. `self_check` accepts a faithful op and rejects a wrong scale and a raising op.
  7. StaticStep(attn_impl="flashmla") on the emulated mini checkpoints with the emulated op: every layer routes, the
     compressed calls carry exactly the host length, ratio-0 calls none, each call within bf16 class of the reference
     algorithm on the same inputs (the random mini then amplifies that through its FP8 QDQs, so logits are only
     checked finite here); a failing load leaves the reference in force (logits == the reference forward).
  8. Generator / first_light plumbing: --attn-impl flashmla, the level-A drift row (not gated), the variant label.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import types

import pytest
import torch

from runtime import attn_flashmla as AFM
from runtime.kernels import sparse_attn_bounded as SAB
from test_attn_bound_cpu import _static_list
from test_static_emulated_cpu import KERNELS, _ref_run, _static_run, emu  # noqa: F401  (emu is a fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
WIN, K = 128, 512


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    saved = dict(AFM._STATE)
    yield
    AFM._STATE.clear()
    AFM._STATE.update(saved)
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ the op, emulated
class FakeOps:
    """torch emulation of `_flashmla_C.sparse_prefill_fwd`'s contract (flash_mla_interface.py:197-234): q [s_q, h_q,
    d_qk] bf16, kv [s_kv, 1, d_qk] bf16, indices [s_q, 1, topk] int32 (-1 or >= s_kv = skip), only the leading
    topk_length[i] indices of row i, softmax in `dt` with the sink in the denominator -> [out bf16, max_logits, lse].
    Records every call's argument shapes."""

    def __init__(self, dt=torch.float32, scale_mult=1.0):
        self.dt, self.scale_mult, self.calls = dt, scale_mult, []

    def sparse_prefill_fwd(self, q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out):
        s_q, h, d = q.shape
        assert kv.dim() == 3 and kv.size(1) == 1 and kv.size(-1) == d == 512 and d_v == 512
        assert indices.dtype == torch.int32 and indices.shape[:2] == (s_q, 1) and indices.size(-1) % 128 == 0
        assert h in (64, 128) and attn_sink.dtype == torch.float32 and attn_sink.shape == (h,)
        assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
        self.calls.append({"q": tuple(q.shape), "kv": tuple(kv.shape), "idx": tuple(indices.shape),
                           "tlen": None if topk_length is None else topk_length.tolist()})
        o = torch.zeros(s_q, h, d_v, dtype=self.dt)
        mxs = torch.zeros(s_q, h)
        lses = torch.zeros(s_q, h)
        kvf = kv[:, 0].to(self.dt)
        for i in range(s_q):
            ids = indices[i, 0]
            if topk_length is not None:
                ids = ids[: int(topk_length[i])]
            ids = ids[(ids >= 0) & (ids < kv.size(0))].long()
            k = kvf[ids]
            s = (q[i].to(self.dt) @ k.T) * (sm_scale * self.scale_mult)
            mx = s.amax(-1) if ids.numel() else torch.full((h,), -math.inf, dtype=self.dt)
            m2 = torch.maximum(mx, attn_sink.to(self.dt))
            e = torch.exp(s - m2.unsqueeze(-1))
            den = e.sum(-1) + torch.exp(attn_sink.to(self.dt) - m2)
            o[i] = (e @ k[:, :d_v]) / den.unsqueeze(-1)
            mxs[i], lses[i] = mx.float(), (m2 + torch.log(e.sum(-1))).float()
        return [o.to(torch.bfloat16), mxs, lses]


def _rand(shape, g, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------ 1. the length
@pytest.mark.parametrize("ratio", [1, 2])
def test_topk_length_equals_reference_width_pos_1_to_4096(ratio):
    pos = torch.arange(1, 4097)
    clen = torch.div(pos + 1, ratio, rounding_mode="floor")
    dev = AFM.topk_length_static(clen, K, WIN)
    assert dev.dtype == torch.int32 and tuple(dev.shape) == (4096,)
    nblk = SAB.nblk_static(clen, K, WIN)
    for p in range(1, 4097):
        w = SAB.ref_width(p, ratio, K, WIN)
        assert AFM.host_topk_length(p, ratio, K, WIN) == w == int(dev[p - 1]), (ratio, p)
        assert SAB.blocks_for_width(int(dev[p - 1])) == int(nblk[0, p - 1])
    assert AFM.host_topk_length(4095, ratio, K, WIN) == WIN + K == 640
    assert AFM.host_topk_length(100, 0, K, WIN) is None


# ------------------------------------------------------------------------------------------------ 2. real lists
@pytest.mark.parametrize("ratio,W", [(2, 2048), (1, 4096)])
def test_topk_length_is_exact_on_static_index_lists(ratio, W):
    g = torch.Generator().manual_seed(10 + ratio)
    k = min(K, W)
    for pos in list(range(1, 700, 3)) + list(range(700, 4096, 41)):
        idx, clen = _static_list(pos, ratio, W, k, g, ties=pos % 3 == 0)
        tl = int(AFM.topk_length_static(clen, k, WIN))
        comp = idx[0, 0, WIN:]
        assert bool((comp[: tl - WIN] >= WIN).all()), (ratio, pos)          # every slot before it valid
        assert bool((idx[0, 0, tl:] == -1).all()), (ratio, pos)             # every slot from it on -1
        assert torch.equal(comp[: tl - WIN], comp[: tl - WIN].sort().values)


# ------------------------------------------------------------------------------------------------ 3. construction
def test_padding_helpers():
    assert [AFM.padded_heads(h) for h in (1, 4, 64, 65, 128)] == [64, 64, 64, 128, 128]
    with pytest.raises(ValueError):
        AFM.padded_heads(129)
    assert [AFM.padded_topk(n) for n in (128, 144, 192, 384, 640, 641)] == [128, 256, 256, 384, 640, 768]


def test_prepare_real_shapes_no_copy():
    g = torch.Generator().manual_seed(0)
    q = _rand((1, 1, 64, 512), g)
    kv = _rand((1, WIN + 4096, 512), g)
    sink = torch.randn(64, generator=g)
    idx = torch.full((1, 1, 640), -1, dtype=torch.int32)
    idx[0, 0, :200] = torch.arange(200, dtype=torch.int32)
    tl = torch.tensor([200], dtype=torch.int32)
    qf, kvf, ix, sk, t, shp, padded = AFM.prepare(q, kv, sink, idx, tl)
    assert not padded and shp == (1, 1, 64, 512)
    assert qf.shape == (1, 64, 512) and kvf.shape == (WIN + 4096, 1, 512) and ix.shape == (1, 1, 640)
    assert qf.data_ptr() == q.data_ptr() and kvf.data_ptr() == kv.data_ptr() and ix.data_ptr() == idx.data_ptr()
    assert sk.data_ptr() == sink.data_ptr() and t.data_ptr() == tl.data_ptr()
    # ratio-0: the 128 window slots, a contiguous slice of the window cache
    cache = _rand((4, WIN, 512), g)
    _, kvf, ix, _, t, _, padded = AFM.prepare(q, cache[:1], sink, idx[..., :WIN].contiguous(), None)
    assert not padded and t is None and kvf.data_ptr() == cache.data_ptr() and ix.shape == (1, 1, 128)


def test_prepare_mini_shapes_pad():
    g = torch.Generator().manual_seed(1)
    q = _rand((1, 1, 4, 128), g)
    kv = _rand((1, WIN + 64, 128), g)
    sink = torch.randn(4, generator=g)
    idx = torch.arange(WIN + 16, dtype=torch.int32).view(1, 1, -1)
    qf, kvf, ix, sk, _, shp, padded = AFM.prepare(q, kv, sink, idx, None)
    assert padded and shp == (1, 1, 4, 128)
    assert qf.shape == (1, 64, 512) and torch.equal(qf[0, :4, :128], q[0, 0]) and not qf[0, 4:].any()
    assert not qf[0, :, 128:].any()
    assert kvf.shape == (WIN + 64, 1, 512) and torch.equal(kvf[:, 0, :128], kv[0]) and not kvf[:, 0, 128:].any()
    assert ix.shape == (1, 1, 256) and torch.equal(ix[0, 0, :144], idx[0, 0]) and bool((ix[0, 0, 144:] == -1).all())
    assert sk.shape == (64,) and torch.equal(sk[:4], sink) and not sk[4:].any()


def test_batch_offset_equals_separate_calls():
    g = torch.Generator().manual_seed(2)
    ops = FakeOps()
    q = _rand((2, 3, 4, 64), g)
    kv = _rand((2, WIN + 64, 64), g)
    sink = torch.randn(4, generator=g)
    idx = torch.randint(-1, WIN + 64, (2, 3, 192), generator=g).int()
    both = AFM.sparse_attn_flashmla(q, kv, sink, idx, 0.1, None, ops=ops)
    for bi in range(2):
        one = AFM.sparse_attn_flashmla(q[bi:bi + 1], kv[bi:bi + 1], sink, idx[bi:bi + 1], 0.1, None, ops=ops)
        assert torch.equal(both[bi:bi + 1], one), bi
    assert ops.calls[0]["kv"] == (2 * (WIN + 64), 1, 512) and ops.calls[0]["q"] == (6, 64, 512)


# ------------------------------------------------------------------------------------------------ 4. the wrapper
@pytest.mark.parametrize("h,d", [(64, 512), (4, 128)])
def test_wrapper_matches_oracle_and_length_is_masking(h, d):
    g = torch.Generator().manual_seed(3)
    ops = FakeOps(torch.float64)
    kv = _rand((1, WIN + 300, d), g)
    sink = torch.randn(h, generator=g)
    rows = SAB.check_cases(WIN, 256, WIN + 300, seed=4)
    for r in rows[::3]:
        q = _rand((1, 1, h, d), g, 0.5)
        idx = r["idx"].view(1, 1, -1)
        tl = torch.tensor([WIN + r["c"]], dtype=torch.int32)
        got = AFM.sparse_attn_flashmla(q, kv, sink, idx, d ** -0.5, tl, ops=ops)
        assert got.dtype == torch.bfloat16 and got.shape == q.shape and got.is_contiguous()
        o64 = AFM.oracle64(q, kv, sink, idx, d ** -0.5)
        assert torch.equal(got, o64.to(torch.bfloat16)), (r["pos"], r["c"])   # fp64 op -> exactly the oracle's bf16
        assert torch.equal(got, AFM.sparse_attn_flashmla(q, kv, sink, idx, d ** -0.5, None, ops=ops))
        assert torch.equal(AFM.oracle64(q, kv, sink, idx, d ** -0.5, tl), o64)
    assert ops.calls[0]["tlen"] is not None and ops.calls[1]["tlen"] is None
    # rows independent: four rows (different lengths) in one call == each alone
    pick = [rows[i] for i in (1, 7, 15, 30)]
    idx = torch.stack([p["idx"] for p in pick]).view(1, 4, -1)
    tl = torch.tensor([WIN + p["c"] for p in pick], dtype=torch.int32)
    q = _rand((1, 4, h, d), g, 0.5)
    got = AFM.sparse_attn_flashmla(q, kv, sink, idx, 0.05, tl, ops=ops)
    for i in range(4):
        assert torch.equal(got[:, i:i + 1], AFM.sparse_attn_flashmla(q[:, i:i + 1].contiguous(), kv, sink,
                                                                     idx[:, i:i + 1].contiguous(), 0.05, tl[i:i + 1],
                                                                     ops=ops))


def test_ineligible_inputs_raise():
    q = torch.zeros(1, 1, 4, 64, dtype=torch.bfloat16)
    kv = torch.zeros(1, 200, 64, dtype=torch.bfloat16)
    idx = torch.zeros(1, 1, 128, dtype=torch.int32)
    assert AFM.reason(q, kv, idx, None) is None
    assert AFM.reason(q.float(), kv, idx, None) == "q"
    assert AFM.reason(q, kv, idx.long(), None) == "idx"
    assert AFM.reason(q, kv[..., :32], idx, None) == "kv"
    assert AFM.reason(torch.zeros(1, 1, 4, 576, dtype=torch.bfloat16), kv, idx, None) == "shape"
    assert AFM.reason(q, kv, idx, torch.zeros(2, dtype=torch.int32)) == "topk_length"
    with pytest.raises(ValueError, match="ineligible"):
        AFM.sparse_attn_flashmla(q, kv, None, idx.long(), 0.1, None, ops=FakeOps())


# ------------------------------------------------------------------------------------------------ 5. loading
def _fresh():
    AFM._STATE.update(ops=None, so=None, error=None)


def test_load_ops_guards(monkeypatch, tmp_path):
    _fresh()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    with pytest.raises(RuntimeError, match="already imported"):
        AFM.load_ops(roots=[str(tmp_path)])
    monkeypatch.delitem(sys.modules, "vllm")
    with pytest.raises(RuntimeError, match="no vllm"):
        AFM.load_ops(roots=[str(tmp_path)])
    so = tmp_path / "vllm" / "_flashmla_C.abi3.so"
    so.parent.mkdir()
    so.write_bytes(b"\x7fELF")
    assert AFM.find_so([str(tmp_path)]) == [str(so)]
    loads = []

    def taint(p):
        loads.append(p)
        sys.modules["vllm"] = types.ModuleType("vllm")

    try:
        with pytest.raises(RuntimeError, match="imported vllm Python"):
            AFM.load_ops(roots=[str(tmp_path)], _load=taint)
    finally:
        sys.modules.pop("vllm", None)
    assert loads == [str(so)]
    with pytest.raises(RuntimeError, match="missing"):           # nothing registers torch.ops._flashmla_C here
        AFM.load_ops(roots=[str(tmp_path)], _load=lambda p: None)
    assert AFM._STATE["ops"] is None and not AFM.loaded()["loaded"]


def test_module_never_imports_vllm():
    src = open(AFM.__file__).read()
    assert not re.search(r"^\s*(import|from)\s+vllm\b", src, re.M)
    assert "vllm" not in sys.modules or not any(m.startswith("vllm") for m in AFM.vllm_modules())


def test_kernel_on_device_box_only():
    pytest.skip("FlashMLA's .so (vllm/_flashmla_C.abi3.so) and an SM100 GPU exist only in the box image: the op, "
                "its numerics vs the reference kernel, graph capture and the mini checkpoints run in "
                "tests/gpu/test_attn_flashmla_gpu.py")


# ------------------------------------------------------------------------------------------------ 6. self-check
def test_self_check_accepts_and_rejects():
    ref = KERNELS["sparse_attn"]
    ops = FakeOps()
    ok = AFM.self_check(ref, "cpu", 4, 32, 0.2, WIN, 64, WIN + 256,
                        impl=lambda *a: AFM.sparse_attn_flashmla(*a, ops=ops))
    assert ok["ok"] and ok["reason"] is None and ok["max_err_ratio"] <= AFM.CHECK_MAX_RATIO, ok
    assert ok["topk_length_eq_masked"] and ok["rows_m_invariant"] and ok["cases"] > 30
    assert ok["padded_heads"] == 64 and ok["padded_topk"] == {"window": 128, "compressed": 256}
    assert any(c["tlen"] is None for c in ops.calls) and any(c["tlen"] is not None for c in ops.calls)
    json.dumps(ok)
    bad_ops = FakeOps(scale_mult=2.0)
    bad = AFM.self_check(ref, "cpu", 4, 32, 0.2, WIN, 64, WIN + 256,
                         impl=lambda *a: AFM.sparse_attn_flashmla(*a, ops=bad_ops))
    assert not bad["ok"] and "fp64-oracle error" in bad["reason"], bad

    def boom(*a):
        raise RuntimeError("CUDA error: no kernel image is available\nmore")

    err = AFM.self_check(ref, "cpu", 4, 32, 0.2, WIN, 64, WIN + 256, impl=boom)
    assert not err["ok"] and err["reason"].startswith("RuntimeError: CUDA error"), err


# ------------------------------------------------------------------------------------------------ 7. StaticStep
@pytest.mark.parametrize("kind", ["mini", "mini4"])
def test_flashmla_static_step_routes_every_layer(emu, monkeypatch, kind):
    from runtime.static_step import StaticStep

    model, pool, _ = emu(kind)
    ops = FakeOps()
    monkeypatch.setattr(AFM, "load_ops", lambda *a, **k: ops)
    monkeypatch.setattr(AFM, "loaded", lambda: {"so": "<emulated>", "loaded": True, "vllm_modules": []})
    st = StaticStep(model, pool, max_pos=127, attn_impl="flashmla", _allow_cpu=True)
    d = st.attn_describe()
    assert d["impl"] == "flashmla" and d["on"] and d["check"]["ok"] and d["reason"] is None, d
    json.dumps(d)
    diffs = []
    wrapped = AFM.sparse_attn_flashmla

    def spy(q, kv, sink, idx, scale, tl=None, ops=None):
        o = wrapped(q, kv, sink, idx, scale, tl)
        r = KERNELS["sparse_attn"](q, kv, sink, idx, scale)
        diffs.append(float((o.float() - r.float()).norm() / r.float().norm()))
        return o

    monkeypatch.setattr(AFM, "sparse_attn_flashmla", spy)
    attns = [l.attn for l in model.layers]
    srcs, src = [], None
    for a in attns:
        src = a if a.is_index_source else src
        srcs.append(src if a.compress_ratio else None)
    g = torch.Generator().manual_seed(5)
    for L in (2, 17):
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        ops.calls.clear()
        got = _static_run(model, st, ids, 6)
        assert len(ops.calls) == 6 * len(attns)
        for n, c in enumerate(ops.calls):
            pos, a, s_ = L + n // len(attns), attns[n % len(attns)], srcs[n % len(attns)]
            if a.compress_ratio:
                ix = s_.indexer
                k = min(ix.index_topk, st.W[ix.compress_ratio])
                assert c["tlen"] == [AFM.host_topk_length(pos, ix.compress_ratio, k, WIN)], (n, c)
                assert c["idx"] == (1, 1, AFM.padded_topk(WIN + k))
            else:
                assert c["tlen"] is None and c["idx"] == (1, 1, WIN)
            assert c["q"] == (1, 64, 512)
        ref = _ref_run(model, ids, 6)
        for a_, b_ in zip(got[0], ref[0]):   # A3-class: close per call (below), not equal; the random mini amplifies
            assert bool(torch.isfinite(a_).all())                    # bf16-level differences through its FP8 QDQs
    assert diffs and max(diffs) < 1e-2, max(diffs)               # every routed call vs the reference algorithm
    st.detach()


def test_flashmla_topk_length_knob(emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini")
    ops = FakeOps()
    monkeypatch.setattr(AFM, "load_ops", lambda *a, **k: ops)
    monkeypatch.setenv(AFM.TOPK_LENGTH_ENV, "0")
    st = StaticStep(model, pool, max_pos=127, attn_impl="flashmla", _allow_cpu=True)
    d = st.attn_describe()
    assert d["on"] and d["check"]["topk_length"] is False, d
    ops.calls.clear()
    _static_run(model, st, list(range(30, 47)), 3)
    assert len(ops.calls) == 3 * len(model.layers) and all(c["tlen"] is None for c in ops.calls)
    st.detach()


def test_flashmla_load_failure_keeps_reference(emu, monkeypatch, capsys):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")

    def refuse(*a, **k):
        raise RuntimeError("no vllm/_flashmla_C*.so under ['/usr/local/lib/python3.12/dist-packages']")

    monkeypatch.setattr(AFM, "load_ops", refuse)
    st = StaticStep(model, pool, max_pos=127, attn_impl="flashmla", _allow_cpu=True)
    d = st.attn_describe()
    assert not d["on"] and "no vllm/_flashmla_C" in d["reason"], d
    assert "attn_impl flashmla is OFF" in capsys.readouterr().err
    ids = list(range(30, 47))
    got = _static_run(model, st, ids, 5)
    ref = _ref_run(model, ids, 5)
    assert all(torch.equal(a, b) for a, b in zip(got[0], ref[0])) and got[1] == ref[1]
    st.detach()


# ------------------------------------------------------------------------------------------------ 8. plumbing
class FakeStaticFM:
    """attn_impl "flashmla" shifts every logit by 1e-3 (an A3-class impl: digests differ, argmax does not)."""

    built = []

    def __init__(self, model, moe_impl="ref", fp8_impl="ref", glue_impl="ref", attn_impl="ref"):
        self.model, self.attn_impl = model, attn_impl
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures = {}, []
        self.glue = None
        FakeStaticFM.built.append((moe_impl, fp8_impl, glue_impl, attn_impl))

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
        self.tok.copy_(ids.view(1, 1))
        return logits + 1e-3 if self.attn_impl == "flashmla" else logits

    def attn_describe(self):
        return {"impl": self.attn_impl, "on": self.attn_impl == "flashmla", "reason": None}

    def detach(self):
        self.graphs = {}

    def cold_counts(self):
        return None


def _gen(mode="eager", **kw):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStaticFM.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=FakeStaticFM, **kw)


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()][:n]


def test_generator_flashmla_plumbing():
    gen = _gen(mode="static", attn_impl="flashmla")
    gen.greedy([5, 9, 12, 3], 4)
    assert gen.static.attn_impl == "flashmla" and gen.last_meta["attn_impl"] == "flashmla"
    gen.set_attn_impl("bounded")
    assert gen.static is None
    from runtime import static_step

    assert static_step.ATTN_IMPLS == ("ref", "bounded", "flashmla")


def test_parity_levels_flashmla_drift_row():
    fl = _first_light()
    gen = _gen(mode="graph", glue_impl="fused", attn_impl="flashmla")
    out = []
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    row = by[("off", "flashmla vs ref attn (same other impls)")]
    assert row["level"] == "A" and not row["gate"] and row["equal"] == 0
    assert row["first_token_divergence"] is None and abs(row["mean_abs_dlogit"] - 1e-3) < 1e-6
    assert res["attn_drift"]["prompts_token_identical"] == 3
    assert ("off", "bounded attn vs ref attn (same moe/fp8/glue)") not in by
    assert not by[("off", "static-eager vs reference-eager")]["gate"]          # a variant: L0 reported
    assert by[("off", "ref static-eager vs reference-eager")]["gate"]
    assert by[("off", "ref static-eager vs reference-eager")]["all_equal"]
    assert by[("off", "graph vs static-eager")]["all_equal"] and by[("off", "graph relaunch vs replay#1")]["all_equal"]
    assert res["variant"] == "flashmla" and res["verdict"] == "PASS" and res["attn_impl"] == "flashmla"
    assert ("ref", "ref", "fused", "ref") in FakeStaticFM.built                # the A reference: same impls, attn ref
    assert (gen.attn_impl, gen.glue_impl, gen.mode) == ("flashmla", "fused", "graph")
    assert any("drift flashmla vs ref attn" in l for l in out)


def test_first_light_flashmla_flag_report_and_summary(tmp_path, capsys):
    fl = _first_light()
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--attn-impl", "flashmla",
                                      "--parity-levels", "--level-steps", "2", "--fixture", FIXTURE, "--only",
                                      "prose", "--reps", "1", "--max-new", "4", "--step-cell", "6", "--stats-from",
                                      "2", "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    assert a.attn_impl == "flashmla"
    gen = _gen(mode="graph", attn_impl=a.attn_impl)
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["generator"]["attn_impl"] == "flashmla" and report["step_cell"]["attn_impl"] == "flashmla"
    assert report["parity_levels"]["verdict"] == "PASS" and report["attn_bound"]["impl"] == "flashmla"
    assert not report["errors"], report["errors"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | mode graph attn flashmla levels PASS")
    report["attn_bound"]["on"] = False
    assert fl.summary_lines(report, "x")[0].startswith("FIRST LIGHT d | mode graph attn flashmla(OFF) levels PASS")
