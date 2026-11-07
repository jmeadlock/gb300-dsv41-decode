"""Mac tests of M1-T12 (`.venv/bin/python -m pytest -q tests/test_dspark_loop_cpu.py`): runtime/dspark_loop.py.

  * acceptance math on synthetic argmax / draft tensors: n = cumprod(match).sum() for every n in 0..5 and a fuzz
    against the host rule (bench/draft_bench.py `accepted`'s rule, step-local);
  * emission + bookkeeping (LoopBuffers.accept) against a host simulation of the loop: predicated stores at n_out,
    n_out += n+1, the accept log, A' = P0+n, ctx_count = n+1, n = 0 and n = 5 steps;
  * EOS / max_new predication: EOS among the emitted rows sets done, EOS in a rejected row does not; every step after
    done is a no-op (no emission, A' = P0-1, ctx 0, log unchanged);
  * the ratio-2 compressor commit: `ratio2_commit_src` reproduces the reference's per-token state writes
    (model.py:477-480) for every P0 parity x n in 0..5 and the no-op step; n = 5 is StaticStep's own commit;
  * VerifyStep (deferred commit + commit at n = 5) == StaticStep(rows=6) teacher-forced, logits and state;
  * tests/gpu/test_dspark_loop_gpu.py's gates on the CPU emulations (DEV "cpu", eager, row-invariant mode): the
    loop's stream == the 1-row greedy and every emitted row's fp32 logits == the 1-row step's; forced rejections at
    every n (short context and past the 256-ring wrap); EOS stop and no-op steps;
  * bench/headline.py's pure helpers (window stats, divergence table, summary lines).
"""
from __future__ import annotations

import contextlib
import functools
import importlib.util
import itertools
import os
import sys

import pytest
import torch

from mini_ckpt import MINI_CONFIG
from mini_mtp import make_mini_checkpoint_mtp, mtp_config_kind
from runtime.loader import build_and_load
from runtime.refmodel import import_reference
from test_static_emulated_cpu import KERNELS, _fake_static_kernels
from test_static_rows_cpu import _rowwise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cpu_default(fn):
    """Synthetic builders run with the CPU as the default device (a Generator elsewhere may have set another)."""
    @functools.wraps(fn)
    def w(*a, **k):
        with torch.device("cpu"):
            return fn(*a, **k)
    return w


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ acceptance math
@_cpu_default
def _case(n, seed, V=50):
    """A verify input tok [1, 6] and row argmaxes t [6] with exactly n accepted drafts."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, V, (6,), generator=g)
    d = torch.randint(0, V, (5,), generator=g)
    d[:n] = t[:n]
    if n < 5:
        d[n] = (t[n] + 1) % V
    tok = torch.cat([torch.randint(0, V, (1,), generator=g), d]).view(1, 6)
    return tok, t


@pytest.mark.parametrize("n", range(6))
def test_accept_count_every_n(n):
    from runtime.dspark_loop import accept_count, host_accept

    for seed in range(20):
        tok, t = _case(n, 31 * n + seed)
        got = accept_count(tok, t)
        assert got.shape == (1,) and got.dtype == torch.long
        assert int(got) == n == host_accept(tok[0].tolist(), t.tolist())


def test_accept_count_fuzz_small_vocab():
    from runtime.dspark_loop import accept_count, host_accept

    g = torch.Generator().manual_seed(5)
    seen = set()
    for _ in range(600):
        tok = torch.randint(0, 2, (1, 6), generator=g)
        t = torch.randint(0, 2, (6,), generator=g)
        n = int(accept_count(tok, t))
        assert n == host_accept(tok[0].tolist(), t.tolist())
        seen.add(n)
    assert seen == set(range(6))


class _Sim:
    """Host simulation of the loop's bookkeeping (module docstring of runtime/dspark_loop.py)."""

    def __init__(self, first, L, max_new, eos):
        self.out, self.L, self.max_new, self.eos = [first], L, max_new, set(eos)
        self.p0, self.done, self.acc = L, (first in self.eos) or max_new <= 1, []

    def step(self, tok, t):
        from runtime.dspark_loop import host_accept

        if self.done:
            return {"live": 0, "a_new": self.p0 - 1, "ctx": 0}
        n = host_accept(tok, t)
        em = list(t[: n + 1])
        self.out += em
        self.acc.append(n)
        if any(x in self.eos for x in em) or len(self.out) >= self.max_new:
            self.done = True
        a = self.p0 + n
        self.p0 = a + 1
        return {"live": 1, "a_new": a, "ctx": n + 1}


def _run_bookkeeping(steps, first=7, L=20, max_new=30, eos=(99,)):
    from runtime.dspark_loop import LoopBuffers

    b = LoopBuffers(max_new, eos, "cpu")
    b.reset(torch.tensor([first]), L, max_new)
    sim = _Sim(first, L, max_new, eos)
    for tok, t in steps:
        want = sim.step(tok[0].tolist(), t.tolist())
        b.accept(tok, t)
        assert int(b.live) == want["live"] and int(b.a_new) == want["a_new"] and int(b.ctx) == want["ctx"]
        assert int(b.done) == int(sim.done)
        b.p0.copy_(b.a_new + 1)                  # the loop: P0' = A' + 1 (st.pos), read back at the next step
        assert int(b.p0) == sim.p0 if want["live"] else int(b.p0) == sim.p0
        r = b.read()
        assert r["ids"] == sim.out and r["accepts"] == sim.acc and r["n_out"] == len(sim.out)
        assert int(b.p0) == L + r["n_out"] - 1   # the invariant P0 = L + n_out - 1
    return b, sim


def test_emission_and_bookkeeping_every_n():
    ns = [0, 5, 3, 0, 1, 2, 4, 5]
    steps = [_case(n, 100 + i) for i, n in enumerate(ns)]
    b, sim = _run_bookkeeping(steps, max_new=200)
    assert sim.acc == ns and not sim.done
    # slots past n_out keep their bytes (predicated stores): zero after reset
    assert b.out_ids[len(sim.out):].eq(0).all()


def test_eos_in_accepted_row_sets_done_then_noop():
    tok, t = _case(3, 7)
    t = t.clone()
    t[2] = 99                                    # EOS emitted (row 2 <= n = 3)
    tok[0, 1:4] = t[:3]
    rest = [_case(n, 50 + n) for n in (5, 0, 2)]
    b, sim = _run_bookkeeping([(tok, t)] + rest)
    assert sim.done and sim.acc == [3]
    r = b.read()
    assert r["n_out"] == 5 and r["done"] and r["accepts"] == [3]
    from runtime.dspark_loop import finalize_ids

    ids, fin = finalize_ids(r["ids"], 30, [99], True)
    assert fin == "stop" and ids == r["ids"][:4]       # cut at the EOS (kept): tokens after it in the step dropped


def test_eos_in_rejected_row_does_not_stop():
    tok, t = _case(1, 9)
    t = t.clone()
    t[3] = 99                                    # row 3 > n = 1: not emitted
    _, sim = _run_bookkeeping([(tok, t), _case(2, 4)])
    assert not sim.done and sim.acc == [1, 2]


def test_max_new_sets_done_and_freezes():
    steps = [_case(5, i) for i in range(6)]
    b, sim = _run_bookkeeping(steps, max_new=14)
    assert sim.done and sim.acc == [5, 5, 5]     # 1 + 6 + 6 = 13 < 14, then 19 >= 14
    r = b.read()
    assert r["n_out"] == 19 and r["accepts"] == [5, 5, 5]
    from runtime.dspark_loop import finalize_ids

    ids, fin = finalize_ids(r["ids"], 14, [99], True)
    assert len(ids) == 14 and fin == "length"


def test_first_token_eos_or_max_new_one_is_done():
    from runtime.dspark_loop import LoopBuffers

    b = LoopBuffers(8, [3], "cpu")
    b.reset(torch.tensor([3]), 10, 8)
    assert int(b.done) == 1
    b.reset(torch.tensor([4]), 10, 1)
    assert int(b.done) == 1
    b.reset(torch.tensor([4]), 10, 8)
    assert int(b.done) == 0
    b.set_eos(False)
    b.reset(torch.tensor([3]), 10, 8)
    assert int(b.done) == 0
    with pytest.raises(ValueError):
        b.reset(torch.tensor([4]), 10, 9)


# ------------------------------------------------------------------------------------------------ compressor commit
@pytest.mark.parametrize("p0,n", list(itertools.product([4, 5, 130, 131, 300], range(-1, 6))))
def test_ratio2_commit_equals_per_token_reference(p0, n):
    """n = -1 is the no-op step (A' = P0-1). The reference decode writes state[p % 2] = kv(p) per token
    (model.py:477-480); after committing positions <= A' slot s holds kv of the last position <= A' with parity s."""
    from runtime.dspark_loop import ratio2_commit_src
    from runtime.static_step import ratio2_state_rows

    g = torch.Generator().manual_seed(p0 * 7 + n)
    d = 4
    kv = torch.randn(p0 + 6, d, generator=g)            # kv of positions 0 .. P0+5
    state = torch.full((2, d), float("nan"))
    for p in range(p0):                                  # committed before this step
        state[p % 2] = kv[p]
    pre = state.clone()
    a_new = p0 + n
    ref = pre.clone()
    for p in range(p0, a_new + 1):
        ref[p % 2] = kv[p]
    seq = torch.cat([pre, kv[p0:p0 + 6]], 0)            # [state0, state1, r0..r5]
    src = ratio2_commit_src(torch.tensor([a_new]), torch.tensor([p0]), torch.arange(2))
    got = seq.index_select(0, src)
    assert torch.equal(got, ref), (p0, n, src.tolist())
    if n == 5:  # StaticStep's own commit (all 6 rows): slot 0 <- last even row, slot 1 <- last odd row
        assert (src - 2).tolist() == ratio2_state_rows(torch.tensor([p0]), 6).tolist()


def test_finalize_ids():
    from runtime.dspark_loop import finalize_ids

    assert finalize_ids([5, 6, 1, 7], 10, [1], True) == ([5, 6, 1], "stop")
    assert finalize_ids([5, 6, 1, 7], 10, [1], False) == ([5, 6, 1, 7], "length")
    assert finalize_ids([5, 6, 7, 8, 9], 3, [1], True) == ([5, 6, 7], "length")
    assert finalize_ids([5, 6, 7, 1], 3, [1], True) == ([5, 6, 7], "length")   # EOS past max_new


# ------------------------------------------------------------------------------------------------ model-level (emulated)
def _per_token(fn):
    """hc_split_sinkhorn(mixes [b, s, mix], ...) evaluated one token row at a time (outputs concatenated on dim 1)."""
    def g(mixes, *args, **kw):
        if mixes.size(1) <= 1:
            return fn(mixes, *args, **kw)
        outs = [fn(mixes[:, i:i + 1].contiguous(), *args, **kw) for i in range(mixes.size(1))]
        return tuple(torch.cat([o[j] for o in outs], 1) for j in range(len(outs[0])))
    return g


def test_emulated_sinkhorn_per_token_matches_one_row_calls():
    """Why the fixture wraps it: the per-token wrapper == 1-row calls bit for bit (the plain emulation may not be)."""
    import_reference(cpu_stubs=True)
    f = _per_token(KERNELS["hc_split_sinkhorn"])
    g = torch.Generator().manual_seed(0)
    for s in (2, 6):
        mix = torch.randn(1, s, 24, generator=g) * 3
        sc, base = torch.randn(3, generator=g), torch.randn(24, generator=g)
        a = f(mix, sc, base, 4, 20, 1e-6)
        for i in range(s):
            b = KERNELS["hc_split_sinkhorn"](mix[:, i:i + 1].contiguous(), sc, base, 4, 20, 1e-6)
            assert all(torch.equal(x[:, i:i + 1], y) for x, y in zip(a, b))


@pytest.fixture
def emu(monkeypatch, tmp_path):
    """-> build(kind) -> (model, pool): a mini checkpoint with the mtp schema, load_mtp=True, CPU emulations, the
    rowwise fp8/fp4 GEMMs (row-invariant like the box kernels, M1-T10 census) and the fake static gather kernels."""
    R = import_reference(cpu_stubs=True)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: ([i % 97 for i in range(MINI_CONFIG["vocab_size"])],
                                     MINI_CONFIG["engram_compressed_vocab_size"]))
    for k, f in KERNELS.items():
        monkeypatch.setattr(R, k, f)
    monkeypatch.setattr(R, "fp4_gemm", _rowwise(KERNELS["fp4_gemm"]))
    monkeypatch.setattr(R, "fp8_gemm", _rowwise(KERNELS["fp8_gemm"]))
    # the TileLang hc_split_sinkhorn is one CTA per token row (reference kernel.py:421, `T.Kernel(n)`): row-invariant
    # by construction. Its torch emulation is not (CPU elementwise sigmoid takes the vector path on a 6-row tensor
    # and the scalar tail on a 1-row one: 1 ulp in `pre`, measured on a random [1,6,24] input) -> run it per row.
    monkeypatch.setattr(R, "hc_split_sinkhorn", _per_token(KERNELS["hc_split_sinkhorn"]))
    prev = torch.get_default_dtype()
    count = itertools.count()

    def build(kind="mini3"):
        ck = make_mini_checkpoint_mtp(str(tmp_path / f"{kind}-{next(count)}"), cfg=mtp_config_kind(kind))
        model = build_and_load(ck["model_dir"], ck["map"], "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                               load_mtp=True, log=lambda *a: None)
        model.temperature = 0.0
        pool = model.runtime_load.pool
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


def test_verify_step_deferred_commit_equals_static_step(emu):
    """VerifyStep with the deferred commit, committed at n = 5, == StaticStep(rows=6) teacher-forced: logits per row
    and the ratio-2 compressor state after every step (both P0 parities)."""
    from runtime.dspark_loop import VerifyStep
    from runtime.static_step import StaticStep

    model, pool = emu("mini3")
    comps = [l.attn.compressor for l in model.layers if l.attn.compressor is not None
             and l.attn.compressor.compress_ratio == 2]
    assert comps, "mini3 has a ratio-2 compressor"
    g = torch.Generator().manual_seed(3)
    for L in (17, 18):
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        toks = torch.randint(2, 2000, (18,), generator=g).tolist()
        runs = []
        for cls in (StaticStep, VerifyStep):
            st = cls(model, pool, max_pos=255, rows=6, _allow_cpu=True)
            if cls is VerifyStep:
                st.defer_ratio2_commit = True
            with torch.inference_mode():
                model(torch.tensor([ids]), 0)
                out = []
                for s in range(3):
                    p0 = L + 6 * s
                    lg = st.step(toks[6 * s: 6 * s + 6], p0, graph=False)
                    if cls is VerifyStep:
                        st.commit_ratio2(torch.tensor([p0 + 5]), torch.tensor([p0]))
                    out.append((lg.clone(), [c.kv_state.clone() for c in comps], [c.score_state.clone() for c in comps]))
            runs.append(out)
            st.detach()
        for (la, ka, sa), (lb, kb, sb) in zip(*runs):
            assert torch.equal(la, lb)
            assert all(torch.equal(x, y) for x, y in zip(ka, kb)) and all(torch.equal(x, y) for x, y in zip(sa, sb))


def _gpu_module(monkeypatch, emu, name="gpu_loop"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "tests", "gpu", "test_dspark_loop_gpu.py"))
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


def test_gpu_stream_gate_dry_run_on_cpu(emu, monkeypatch, capsys):
    """V3 in row-invariant mode on the emulations: stream == 1-row greedy, every emitted row's logits == 1-row; both
    first-step P0 parities; the as-is run (CPU torch matmuls are not row-invariant) reported."""
    G = _gpu_module(monkeypatch, emu)
    G._gate_stream("mini3", 17, 40)
    G._gate_stream("mini3", 18, 40)
    G._gate_stream("mini3", 17, 24, rowinv=False)
    out = capsys.readouterr().out
    assert out.count("loop == 1-row greedy") >= 2, out
    assert G.COUNTS["gates"] == 3 and G.COUNTS["logit_rows"] >= 80, G.COUNTS
    print(out, "\n".join(G.REPORT))


def test_gpu_forced_rejections_dry_run_on_cpu(emu, monkeypatch, capsys):
    """V1: every n in 0..5 forced, the commit path (256-ring overwrite, ratio-2 commit) still matches greedy."""
    G = _gpu_module(monkeypatch, emu)
    want = G._gate_forced("mini3", 17, 48)
    assert set(want) == set(range(6))
    out = capsys.readouterr().out
    assert "forced rejections" in out and G.COUNTS["forced_steps"] >= 10, (out, G.COUNTS)
    print(out)


def test_gpu_forced_rejections_ring_wrap_dry_run_on_cpu(emu, monkeypatch, capsys):
    """V1 past the 256-ring wrap (L = 250: positions 250..~305), design risk 1."""
    G = _gpu_module(monkeypatch, emu)
    want = G._gate_forced("mini3", 250, 56, plan=G.PLAN[3:] + G.PLAN[:3])
    assert set(want) == set(range(6))
    print(capsys.readouterr().out)


def test_gpu_eos_gate_dry_run_on_cpu(emu, monkeypatch, capsys):
    G = _gpu_module(monkeypatch, emu)
    G._gate_eos("mini3", 19, 48)
    out = capsys.readouterr().out
    assert "change nothing" in out, out
    print(out)


def test_loop_refuses_bad_setups(emu):
    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator
    from tiny_tok import make_wordlevel_tokenizer

    model, pool = emu("mini3")
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                    max_pos=255)
    gen.static_kw = {"_allow_cpu": True}
    loop = DSparkLoop(gen, 32, _allow_cpu=True)
    assert loop.describe()["ratio2_compressors"] == 1 and gen.rows == 6 and gen.hidden_layers == loop.hidden
    with pytest.raises(ValueError):
        loop.check_bounds(240, 32)              # positions past max_pos 255
    with pytest.raises(RuntimeError):
        loop.capture()                          # CPU
    with pytest.raises(ValueError):
        loop.set_force([1, 2, 3])
    loop.close()
    assert gen.rows == 1 and gen.hidden_layers == () and gen.static_cls is None
    gen.static_factory = lambda m, **kw: object()
    with pytest.raises(TypeError):
        DSparkLoop(gen, 32, _allow_cpu=True)


# ------------------------------------------------------------------------------------------------ bench/headline.py
def _headline():
    spec = importlib.util.spec_from_file_location("headline_t", os.path.join(REPO, "bench", "headline.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


def test_headline_window_and_stream_helpers():
    B = _headline()
    runs = [{"step_ms": [10.0] * 3 + [20.0, 30.0], "verify_ms": [6.0] * 3 + [12.0, 18.0],
             "draft_ms": [4.0] * 3 + [8.0, 12.0], "accepts": [0, 1, 2, 3, 4]},
            {"step_ms": [5.0, 5.0], "verify_ms": [3.0, 3.0], "draft_ms": [2.0, 2.0], "accepts": [5, 5]}]
    w = B.window(runs, 3)
    assert w["steps"] == 2 and w["ms_per_step"]["mean"] == 25.0 and w["ms_verify_graph"]["mean"] == 15.0
    assert w["ms_draft_graph"]["mean"] == 10.0 and w["tokens_per_step"] == 4.5 and w["ms_per_token"] == 50.0 / 9
    assert w["all_steps"] == 7 and w["all_tokens_per_step"] == (1 + 2 + 3 + 4 + 5 + 6 + 6) / 7
    assert B.window([], 32)["ms_per_step"] == {"n": 0} and B.window([], 32)["tokens_per_step"] is None
    assert B.first_divergence([1, 2, 3], [1, 2, 3]) is None and B.first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert B.first_divergence([1, 2], [1, 2, 3]) == 2
    rec = {"logprobs": [{"top_logprobs": [{"logprob": -0.1}, {"logprob": -0.3}]}]}
    assert abs(B.margin_at(rec, 0) - 0.2) < 1e-12 and B.margin_at(rec, 1) is None and B.margin_at(None, 0) is None
    s = B.compare_stream([1, 2, 9], [1, 2, 3], ["a", "b", "c"], ["a", "x", "c"], rec, rec)
    assert not s["exact"] and s["first_divergence"] == 2 and s["logit_rows_compared"] == 2
    assert s["logit_rows_bit_equal"] == 1 and s["loop_token"] == 9 and s["r1_token"] == 3


def test_headline_cells_dry_run_on_cpu(emu, monkeypatch):
    """bench/headline.py's run_cells -> aggregate -> summary_lines on the emulations (eager cell, row-invariant mode):
    2 prompts, fast x 2 + parity x 2 each; streams exact vs the 1-row greedy, self-replay ok, the table renders."""
    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator
    from tiny_tok import make_wordlevel_tokenizer

    B = _headline()
    G = _gpu_module(monkeypatch, emu)
    model, pool = emu("mini3")
    with G.row_invariant_mode():
        gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                        max_pos=255)
        gen.static_kw = {"_allow_cpu": True}
        prompts = {}
        g = torch.Generator().manual_seed(11)
        for tag, L in (("a", 17), ("b", 18)):
            ids = torch.randint(2, 2000, (L,), generator=g).tolist()
            rec = gen.greedy(ids, 20, tag=tag)
            m = gen.last_meta
            prompts[tag] = {"ids": ids, "r1_ids": list(m["token_ids"]), "r1_sha": list(m["step_sha256"]),
                            "r1_rec": rec}
        loop = DSparkLoop(gen, 20, _allow_cpu=True)
        rep = {"errors": [], "date": "dry", "mode": "static", "impls": {}, "max_new": 20}
        results = B.run_cells(loop, prompts, 20, [0], reps=2, parity_reps=2, stop_on_eos=True, rep=rep)
        loop.close()
    assert not rep["errors"], rep["errors"]
    B.aggregate(rep, prompts, results, [0], 4)
    assert rep["streams"] == {"exact": 2, "prompts": 2} and rep["replay_ok"] == 2, rep["prompts"]
    for p in rep["prompts"].values():
        assert p["stream"]["logit_rows_bit_equal"] == p["stream"]["logit_rows_compared"] == p["tokens"]
        assert p["replay_ok"] and sum(p["accept_hist"]) == p["steps"]
    lines = B.summary_lines(rep)
    assert lines[0].startswith("HEADLINE dry | ms per 6-row step, DSpark k=5")
    assert "streams exact vs 1-row: 2/2" in lines[-1] and any(l.strip().startswith("a ") for l in lines), lines
    print("\n".join(lines))
