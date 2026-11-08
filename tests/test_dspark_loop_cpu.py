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


# ------------------------------------------------------------------------------------------------ T12-FIX2
def _gpu_loop_module(name="gpu_loop_t12fix2"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "tests", "gpu", "test_dspark_loop_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G


def test_rowwise_reduce_mode_semantics_and_restore():
    """row_invariant_mode(True)'s sum / mean wrappers: same values as torch for every call form, per-row only on
    [1, s > 1, ...] float tensors reduced over dims >= 2; torch.Tensor.sum / mean / torch.sum / mean restored exactly
    (the methods are inherited from TensorBase: no attribute left on torch.Tensor); "linear" leaves them alone."""
    G = _gpu_loop_module()
    before = {(o, n): (n in vars(o), getattr(o, n)) for o, n in G.REDUCTIONS}
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(1, 6, 4, 32, generator=g, device="cpu")
    xi = torch.randint(0, 9, (1, 6, 4, 32), generator=g, device="cpu")
    calls = [lambda t: t.sum(-1), lambda t: t.mean(-1, keepdim=True), lambda t: t.sum(dim=(2, 3)),
             lambda t: torch.sum(t, 2), lambda t: torch.mean(t, dim=-1), lambda t: t.sum(), lambda t: t.sum(dim=1),
             lambda t: t.square().mean(-1, keepdim=True), lambda t: t.sum(-1, dtype=torch.float64)]
    want = [f(x) for f in calls] + [xi.sum(-1)]
    with G.row_invariant_mode("linear"):
        assert all(getattr(o, n) is before[(o, n)][1] for o, n in G.REDUCTIONS)
    with G.row_invariant_mode(True):
        assert all(getattr(o, n) is not before[(o, n)][1] for o, n in G.REDUCTIONS)
        got = [f(x) for f in calls] + [xi.sum(-1)]
    for w, gv in zip(want, got):
        assert w.shape == gv.shape and w.dtype == gv.dtype and torch.equal(w, gv)
    after = {(o, n): (n in vars(o), getattr(o, n)) for o, n in G.REDUCTIONS}
    assert after == before


def test_reduction_census_dry_run_on_cpu():
    """tests/gpu/test_dspark_loop_gpu.py's census on the CPU: every site row-invariant inside the mode (CPU torch
    reduces each output with the same loop, so the plain column is expected 0 here too; on CUDA it is reported)."""
    G = _gpu_loop_module()
    rows = G._reduction_census("cpu", trials=4)
    assert len(rows) == len(G.CENSUS_SITES) and all(r["mode_rows_differing"] == 0 for r in rows), rows
    print("\n".join(f"{r['site']}: plain {r['plain_rows_differing']}/{r['rows']}" for r in rows))


def test_cache_probe_free_loop_caches_equal_one_row(emu, monkeypatch):
    """T12-FIX2 on the CPU emulation, the box's failing free run (mini3, L = 17, seed 317; 1.000 tok/step): every
    cache a row reads (256-ring, compressed KV, index K, the ratio-2 compressor's two slots, the Engram hash cache)
    is bit-equal to the 1-row step's at the same position, entering every step (after the previous commit AND draft)
    and after every commit, through position 40 (past position 33, the first clen 17 > k 16)."""
    G = _gpu_module(monkeypatch, emu)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",            # the GPU fixture's map (tests/gpu _model)
                        lambda tok: (list(range(MINI_CONFIG["vocab_size"])), MINI_CONFIG["engram_compressed_vocab_size"]))
    model, _ = emu("mini3")
    L, n = 17, 26
    with G.row_invariant_mode(True), G.cache_probe() as (rec, tag):
        gen = G._gen(model, "static")
        ids = G._toks(L, 300 + L, 2000)
        tag["t"] = "1row"
        truth, sha = G._truth(gen, ids, n)
        tag["t"] = "loop"
        loop = G._loop(gen, n)
        r = loop.generate(ids, n, stop_on_eos=False, parity=True)
        tag["t"] = None
        loop.close()
    assert r["token_ids"] == truth and r["step_sha256"] == sha
    assert set(rec["after"]) >= set(range(L, L + n - 1)), sorted(rec["after"])
    for p in range(L, L + n - 1):
        if p > L:                                                         # no 1-row step ran at the prefill's L-1
            assert G.cache_diff(rec["1row"][p - 1], rec["enter"][p - 1]) == [], (p, "entering")
        assert G.cache_diff(rec["1row"][p], rec["after"][p]) == [], (p, "after")


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


# ------------------------------------------------------------------------------------------------ bf16 helpers
def _draft_gpu_module(name="gpu_draft_t12fix2"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "tests", "gpu", "test_draft_gpu.py"))
    D_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D_)
    return D_


def test_dg_evidence_fp64_helpers():
    """T12-FIX2: bf16 spacing at binade edges; fp4 e2m1 x e8m0 rows and fp8 x block-scale dequantization are exact."""
    from test_static_emulated_cpu import _fp4_deq, _scale_f

    D_ = _draft_gpu_module()
    v = torch.tensor([1.0, 1.5, 1.9921875, 2.0, 0.5, -3.0, 4.0, 7.9], dtype=torch.float64, device="cpu")
    assert D_.bf16_spacing(v).tolist() == [2 ** -7, 2 ** -7, 2 ** -7, 2 ** -6, 2 ** -8, 2 ** -6, 2 ** -5, 2 ** -5]
    g = torch.Generator(device="cpu").manual_seed(2)
    w = torch.randint(0, 256, (4, 32), generator=g, dtype=torch.uint8, device="cpu")
    s = torch.randint(120, 134, (4, 2), generator=g, dtype=torch.uint8, device="cpu").view(torch.float8_e8m0fnu)
    want = _fp4_deq(w).double() * _scale_f(s).double().repeat_interleave(32, -1)
    assert torch.equal(D_.fp4_rows_f64(w, s), want)
    q = (torch.randn(3, 256, generator=g, device="cpu") * 50).to(torch.float8_e4m3fn)
    sc = torch.rand(3, 2, generator=g, device="cpu") + 0.25
    assert torch.equal(D_.act_f64(q, sc), q.double() * sc.double().repeat_interleave(128, -1))
    d = D_.oracle_dist(torch.tensor([1.0078125], dtype=torch.bfloat16, device="cpu"),
                       torch.tensor([1.0], dtype=torch.float64, device="cpu"),
                       torch.tensor([[1.0]], dtype=torch.float64, device="cpu"))
    assert d["ulp"] == 1.0 and d["excess"] == 0.0


def test_dg_moe_evidence_a3_vs_slips(emu):
    """T12-FIX2 (tests/gpu/test_draft_gpu.py `dg_moe_evidence`, the P2 gate's A3-or-slip classifier) on the CPU
    emulation, the reference MoE's own per-pair intermediates standing in for the DG buffers: (a) as-is -> A3,
    0 ulp; (b) one pair's w2 output 1 ulp off at one element (an accumulation-order flip) -> still A3; (c) a wrong
    expert's w2 output in one pair (an indexing slip) -> NOT A3; (d) a scatter slip in one pair's w1/w3 input -> NOT
    A3; (e) a corrupted combine (a pair added twice) -> NOT A3."""
    from runtime.draft_step import DraftStep
    from runtime.refmodel import reference_device

    D_ = _draft_gpu_module()
    model, _ = emu("mini3")
    ds = DraftStep(model, _allow_cpu=True)
    st = ds.stages[0]
    dim = st.ffn.dim
    g = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn(1, ds.block, dim, generator=g, device="cpu").to(torch.bfloat16)
    with torch.inference_mode(), reference_device(x.device):
        want = st.ffn(x).clone()
        sh = st.ffn.shared_experts(x.view(-1, dim))

    def combine(v):
        with torch.inference_mode():
            y = torch.zeros(x.view(-1, dim).shape, dtype=torch.float32, device="cpu")
            for j in range(v["posmat"].size(1)):
                y += v["out"].index_select(0, v["posmat"][:, j]).float()
            y += sh
            return y.type_as(x).view(x.shape)

    def clone(v):
        return {k: t.clone() for k, t in v.items()}

    view, y0 = D_.reference_pairs_view(ds.R, st, x)
    assert torch.equal(y0, want)
    ev = D_.dg_moe_evidence(ds.R, st, x, y0, want, view)
    assert ev["a3"] and ev["replicate_ok"] and ev["combine_dg_ok"], ev["why"]
    assert ev["pair_max_ulp"] == 0 and ev["w2_inputs_equal"] == ev["n_pairs"] == ds.block * st.ffn.gate.topk
    assert not ev["elems"]
    pm = view["posmat"]
    a, b = int(pm[0, 0]), int(pm[1, 1])
    # (b) one element of one pair output 1 bf16 ulp off (magnitude +1 in the sign-magnitude code)
    v = clone(view)
    c = int(v["out"][a].float().abs().argmax())
    with torch.inference_mode():
        v["out"][a, c] = (v["out"][a, c:c + 1].view(torch.int16) + 1).view(torch.bfloat16)[0]
    ev = D_.dg_moe_evidence(ds.R, st, x, combine(v), want, v)
    assert ev["a3"] and ev["pair_max_ulp"] == 1, ev["why"]
    # (c) the wrong expert's output in pair a
    v = clone(view)
    with torch.inference_mode():
        v["out"][a] = view["out"][b]
    ev = D_.dg_moe_evidence(ds.R, st, x, combine(v), want, v)
    assert not ev["a3"] and any("w2" in w for w in ev["why"]), ev["why"]
    # (d) pair a's w1/w3 input from another row
    v = clone(view)
    with torch.inference_mode():
        v["xq1"][a] = view["xq1"][b]
    ev = D_.dg_moe_evidence(ds.R, st, x, combine(view), want, v)
    assert not ev["a3"] and any("scatter" in w for w in ev["why"]), ev["why"]
    # (e) a pair added twice in the combine
    v = clone(view)
    bad = combine(view).clone()
    with torch.inference_mode():
        bad.view(-1, dim)[0] = (bad.view(-1, dim)[0].float() + view["out"][a].float()).to(torch.bfloat16)
    ev = D_.dg_moe_evidence(ds.R, st, x, bad, want, v)
    assert not ev["a3"] and not ev["combine_dg_ok"], ev["why"]


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


# ------------------------------------------------------------------------------------------------ LOOSE-1107
class _ShimGraph:
    """A CUDAGraph stand-in: `replay()` runs the captured body eagerly and counts itself."""

    def __init__(self, fn):
        self.fn, self.replays = fn, 0

    def replay(self):
        self.replays += 1
        self.fn()


def _install_graph_shim(loop, n_graphs):
    """Give a CPU loop the graph structure of DSparkLoop.capture(): n_graphs 2 = A (verify|accept|commit, inside
    st._active()) + B (draft|next input); 1 = AB (A then the draft outside the verify scope) + B for the first draft.
    The bodies are the loop's own (`_verify_accept`, `_draft`), so only the launch structure differs."""
    st = loop.st

    def A():
        with torch.inference_mode(), st._active():
            loop._verify_accept()

    def B():
        with torch.inference_mode():
            loop._draft()

    if n_graphs == 2:
        loop.graphs = {"A": _ShimGraph(A), "B": _ShimGraph(B)}
    else:
        loop.graphs = {"AB": _ShimGraph(lambda: (A(), B())), "B": _ShimGraph(B)}
    loop.n_graphs = n_graphs


class _HookStub:
    """Stands in for the ResidencyManager: records the order of its per-step calls."""

    def __init__(self):
        self.calls = 0

    def after_step(self, live=None):
        self.calls += 1


def test_one_graph_and_two_graphs_modes_make_the_same_accept_plan(emu, monkeypatch):
    """LOOSE-1107 (3): the one_graph and two_graphs launch structures (graph shim: AB + B vs A + B) run the same
    bodies, so on the same prompt they emit the same tokens, the same accept plan, and bit-equal fp32 logits per
    emitted position as the eager loop -- and the residency hook (A2-2) fires once per issued step after the
    replay(s), in both modes. Stream and logits are also the 1-row greedy's (row-invariant mode)."""
    G = _gpu_module(monkeypatch, emu)
    model, _ = emu("mini3")
    n, L = 24, 17
    with G.row_invariant_mode(True):
        gen = G._gen(model, "static")
        ids = G._toks(L, 300 + L, 2000)
        truth, sha = G._truth(gen, ids, n)
        loop = G._loop(gen, n)
        hook = _HookStub()
        monkeypatch.setattr(gen, "residency_mgr", hook, raising=False)
        runs = {}
        for mode in (0, 2, 1):
            if mode:
                _install_graph_shim(loop, mode)
            hook.calls = 0
            for parity in (True, False):
                r = loop.generate(ids, n, stop_on_eos=False, parity=parity)
                assert hook.calls == r["steps_issued"], (mode, parity, hook.calls, r["steps_issued"])
                assert r["overrun_steps"] == r["steps_issued"] - r["steps"] == 0          # LOOP-OVERRUN: per-step sync path
                assert r["poll"] == {"polls": 0, "drains": 0, "enqueue_us": 0.0, "wait_ms": 0.0, "poll_every": 1,
                                     "lookahead": 1, "tail_sync": True}
                hook.calls = 0
                runs[(mode, parity)] = r
            if mode == 2:
                assert loop.graphs["A"].replays == loop.graphs["B"].replays - 2 == sum(
                    runs[(2, p)]["steps_issued"] for p in (True, False)), {k: g.replays for k, g in loop.graphs.items()}
            if mode == 1:
                assert loop.graphs["AB"].replays == sum(runs[(1, p)]["steps_issued"] for p in (True, False))
                assert loop.graphs["B"].replays == 2                   # only the first draft after each prefill
        gen.residency_mgr = None                                       # close() would ask the stub to restore the map
        loop.close()
    base = runs[(0, True)]
    assert base["token_ids"] == truth and base["step_sha256"] == sha
    assert len(base["accepts"]) == base["steps"] > 1 and set(base["accepts"]) <= set(range(6))
    for key, r in runs.items():
        assert r["token_ids"] == base["token_ids"], key
        assert r["accepts"] == base["accepts"], key
        assert r["steps"] == base["steps"] and r["steps_issued"] == base["steps_issued"], key
        if key[1]:
            assert r["step_sha256"] == base["step_sha256"] and r["logits_sha256"] == base["logits_sha256"], key


def _cell(mean, p50, p90, n, verify=None, draft=None, mgr=None, cold=None):
    w = {"stats_from_step": 4, "steps": n, "ms_per_step": {"n": n, "mean": mean, "p50": p50, "p90": p90},
         "ms_verify_graph": {"n": n, "mean": verify} if verify else None,
         "ms_draft_graph": {"n": n, "mean": draft} if draft else None,
         "ms_manager_per_step": {"n": n, "mean": mgr} if mgr is not None else None,
         "ms_per_step_incl_manager": (mean + mgr) if mgr is not None else None, "cold_share": cold}
    return w


def test_mode_gap_report_cases():
    B = _headline()
    assert B.mode_gap({"headline": {"two_graphs": _cell(20.0, 20.0, 20.1, 10)}}) is None       # one cell only
    # no residency, cells equal: nothing to explain
    h = {"two_graphs": _cell(20.30, 20.3, 20.5, 100, 17.4, 2.9), "one_graph": _cell(20.29, 20.3, 20.5, 50)}
    mg = B.mode_gap({"headline": h, "residency_config": None})
    assert mg["within_tolerance"] and mg["cause"].startswith("none") and not mg["residency_on"]
    assert "mode gap" in B.mode_gap_line(mg) and "+0.010" in B.mode_gap_line(mg)
    # no residency, cells differ: flagged as unexplained, never blamed on residency
    h["one_graph"] = _cell(19.0, 19.0, 19.2, 50)
    mg = B.mode_gap({"headline": h, "residency_config": None})
    assert not mg["within_tolerance"] and mg["cause"].startswith("UNEXPLAINED") and mg["headline_mode"] == "unresolved"
    # residency on: two_graphs ms = 20 + 0.4 ms per cold-share point (4 prompts), one_graph at ~zero cold share
    cold = {"a": 0.01, "b": 0.05, "c": 0.09, "d": 0.13}
    prompts = {t: {"steps": 50, "cold_share_by_rep": [c, c], "window": {"all_ms_per_step": {"mean": 20 + 40 * c}},
                   "window_one_graph": {"all_ms_per_step": {"mean": 20.1}}} for t, c in cold.items()}
    m2 = sum(20 + 40 * c for c in cold.values()) / 4
    h = {"two_graphs": _cell(m2, m2, m2 + 1, 200, 17.0, 2.9, 0.05), "one_graph": _cell(20.1, 20.1, 20.2, 100, mgr=0.04)}
    rep = {"headline": h, "prompts": prompts, "residency_config": {"drain_n": 64},
           "residency": {"on": True, "sf_store": {"requested": "pool", "got": "step"}, "host_steps": 448,
                         "drain_log": [{"host_steps": 64 * i, "cold_share": 0.05 if i < 6 else 0.0025}
                                       for i in range(1, 8)]}}
    rep["headline"]["one_graph"]["all_steps"] = 100
    mg = B.mode_gap(rep)
    fit = mg["fit_two_graphs_ms_vs_cold_share"]
    assert abs(fit["slope"] - 40.0) < 1e-9 and abs(fit["intercept"] - 20.0) < 1e-9 and fit["r"] > 0.999999
    assert mg["residency_on"] and not mg["within_tolerance"] and mg["gap_ms"] > 0.5
    mm = mg["one_graph_ms_modeled"]                  # the drains over the last 100 steps (host steps 384, 448): both at 0.25 %
    assert abs(mm["cold_share_drain_tail"] - 0.0025) < 1e-12 and abs(mm["ms"] - 20.1) < 1e-9, mm
    assert "residency state" in mg["cause"] or "policy states" in mg["cause"]
    assert "sf_store pool->step" in mg["cause"] and mg["headline_mode"].startswith("not decidable")
    # both cells instrumented: the fit predicts the gap from the cold shares (modeled)
    rep["headline"]["two_graphs"]["cold_share"], rep["headline"]["one_graph"]["cold_share"] = 0.07, 0.0025
    mg = B.mode_gap(rep)
    assert abs(mg["gap_predicted_from_cold_share_ms"] - 40.0 * (0.07 - 0.0025)) < 1e-9
    assert "7.00% (two_graphs) vs 0.25% (one_graph)" in mg["cause"]
    # the cold-share helper: call-weighted, None without counters
    assert B.cold_pooled([{"routed_calls": 100, "cold_calls": 10}, {"routed_calls": 300, "cold_calls": 10}]) == 0.05
    assert B.cold_pooled([{}]) is None


def test_mode_gap_on_the_residency_receipt():
    """docs/headline-20261107-residency.json (box): 21.82 vs 20.41 ms. The gap is the residency state, not the graph
    mode: the two_graphs cell's per-prompt step ms track its cold share (r > 0.95) and the fit at the cold share of the
    last drains predicts the one_graph cell to 0.1 ms. (Modeled numbers; the receipt is read, nothing is measured.)"""
    import json

    path = os.path.join(REPO, "docs", "headline-20261107-residency.json")
    if not os.path.exists(path):
        pytest.skip("receipt not in this checkout")
    with open(path) as f:
        rep = json.load(f)
    mg = _headline().mode_gap(rep)
    assert abs(mg["gap_ms"] - 1.4137) < 1e-3 and mg["residency_on"] and not mg["within_tolerance"]
    assert mg["draft_graph_ms_two_graphs"] < 2.91 and mg["verify_graph_ms_two_graphs"] > 18.9      # all of it is verify
    assert mg["fit_two_graphs_ms_vs_cold_share"]["r"] > 0.95
    mm = mg["one_graph_ms_modeled"]
    assert abs(mm["ms"] - mm["actual_ms"]) < 0.1, mm
    assert mg["sf_store"] == {"requested": "pool", "got": "step"}


def _probe_module():
    spec = importlib.util.spec_from_file_location("row_divergence_probe_loose", os.path.join(
        REPO, "bench", "row_divergence_probe.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    return P


def test_probe_mini_model_restores_the_engram_token_map(monkeypatch, tmp_path):
    """LOOSE-1107 (2): tests/gpu/test_row_divergence_gpu.py builds mini models and then the REAL checkpoint in one
    process. mini_model patched engram.build_compressed_token_map to a 2048-entry identity map and never undid it, so
    the real build's Engram token_map had 2048 rows and `token_map[input_ids]` (real ids up to ~129k) was a
    device-side index assert -> CUDA_ERROR_ASSERT at the next cuModuleLoadData (docs/rowdiv-gpu-test-r2-20261107.log).
    The patch must be scoped to the context, also when the body raises."""
    import types

    import engram
    import mini_ckpt
    import runtime.loader
    import runtime.refmodel

    P = _probe_module()
    real_import = runtime.refmodel.import_reference
    monkeypatch.setattr(runtime.refmodel, "import_reference", lambda *a, **k: real_import(cpu_stubs=True))
    monkeypatch.setattr(mini_ckpt, "make_mini_checkpoint", lambda d, cfg=None: {"model_dir": d, "map": None})
    seen = []

    def fake_build(model_dir, hot, device, **kw):
        tm, n_classes = engram.build_compressed_token_map(kw.get("tokenizer"))
        seen.append((len(tm), n_classes))
        res = types.SimpleNamespace(pool=object(), release=lambda: seen.append("released"))
        return types.SimpleNamespace(temperature=1.0, runtime_load=res)

    monkeypatch.setattr(runtime.loader, "build_and_load", fake_build)
    marker = lambda tok: ([0] * 129280, 99092)                                  # noqa: E731 (stands in for the real map)
    monkeypatch.setattr(engram, "build_compressed_token_map", marker)
    with P.mini_model("mini4", "cpu") as (model, pool):
        assert engram.build_compressed_token_map is not marker                  # patched inside
        assert model.temperature == 0.0
    assert seen[0] == (MINI_CONFIG["vocab_size"], MINI_CONFIG["engram_compressed_vocab_size"]) and seen[-1] == "released"
    assert engram.build_compressed_token_map is marker                          # restored: the real build sees its own map
    with pytest.raises(RuntimeError, match="boom"):
        with P.mini_model("mini", "cpu"):
            raise RuntimeError("boom")
    assert engram.build_compressed_token_map is marker                          # restored on an exception too


def test_probe_check_token_ids_names_a_mini_sized_map():
    """The host-side guard that replaces the late device assert: an id past the embedding or the Engram token_map is a
    ValueError naming the table, before any kernel runs."""
    import types

    P = _probe_module()
    model = types.SimpleNamespace(
        embed=types.SimpleNamespace(weight=torch.zeros(129280, 1, device="cpu")),
        engram_hash=types.SimpleNamespace(token_map=torch.zeros(2048, dtype=torch.long, device="cpu")))
    P.check_token_ids(model, [0, 5, 2047])
    with pytest.raises(ValueError, match="engram token_map has 2048 rows"):
        P.check_token_ids(model, [3, 128803, 7])
    with pytest.raises(ValueError, match="embed has 129280 rows"):
        P.check_token_ids(types.SimpleNamespace(embed=model.embed, engram_hash=None), [129280])
    P.check_token_ids(types.SimpleNamespace(embed=model.embed, engram_hash=None), [129279])
    P.check_token_ids(model, [])


# ------------------------------------------------------------------------------------------------ LOOP-OVERRUN
# The async poll schedule (runtime/dspark_loop.py PollPlan + DSparkLoop._decode_async) on CPU: the real
# LoopBuffers.accept under a simulated device (snapshot-at-record polls, pinned-ring reuse checked), every schedule vs the
# host truth (_Sim): same stream, same final state, overrun bounds, and the 11/08 fixture's measured overrun reproduced.
def _scripted_steps(ns, eos_step=None, n_steps=400, seed=0):
    """[(tok, t)] with the accept lengths `ns` (cycled); the step `eos_step` (1-based live step) emits EOS 99 at row n."""
    out = []
    for i in range(n_steps):
        n = ns[i % len(ns)]
        tok, t = _case(n, 1000 * seed + i)
        if eos_step is not None and i + 1 == eos_step:
            t = t.clone()
            t[n] = 99
        out.append((tok, t))
    return out


def _async_sim_cls():
    from runtime.dspark_loop import DSparkLoop, LoopBuffers

    class SimLoop(DSparkLoop):
        """DSparkLoop minus the model: step_once = LoopBuffers.accept on scripted (tok, t); polls snapshot the status at
        record time (a real poll's D2H lands after the steps issued before it and before the ones after it)."""

        def __init__(self, steps, max_new, poll_every, lookahead, tail_sync, first=7, L=20, eos=(99,)):
            self.b = LoopBuffers(max_new, eos, "cpu")
            self.b.reset(torch.tensor([first]), L, max_new)
            self.steps, self.k = steps, 0
            self.poll_every, self.lookahead, self.tail_sync = poll_every, lookahead, tail_sync
            self.records = self.waits = 0

        def step_once(self, events=None):
            tok, t = self.steps[self.k % len(self.steps)]
            self.k += 1
            self.b.accept(tok, t)
            self.b.p0.copy_(self.b.a_new + 1)

        def _poll_buffers(self, n):
            return [torch.zeros(3, dtype=torch.long) for _ in range(n)]

        def _poll_record(self, buf):
            buf.copy_(self.b.status)
            self.records += 1
            return (buf, buf.tolist())

        def _poll_wait(self, token):
            buf, snap = token
            assert buf.tolist() == snap, "a pinned poll buffer was overwritten before its poll was read"
            self.waits += 1

    return SimLoop


def _truth(steps, max_new, first=7, L=20):
    sim = _Sim(first, L, max_new, (99,))
    for tok, t in steps:
        if sim.done:
            break
        sim.step(tok[0].tolist(), t.tolist())
    assert sim.done
    return sim


def _run_async(steps, max_new, poll_every, lookahead, tail_sync):
    SimLoop = _async_sim_cls()
    loop = SimLoop(steps, max_new, poll_every, lookahead, tail_sync)
    cap = max(0, max_new - 1) + (lookahead + 1) * poll_every
    issued, stats = loop._decode_async(max_new, cap, None, False, None, [])
    return loop, issued, stats, cap


def _state(loop):
    """Everything that reaches the emitted stream or the next step's inputs. `ctx` / `n_last` are excluded: a no-op step
    rewrites them to 0 / -1 by design (the draft then writes no ring row); the next sequence's reset() clears them."""
    b = loop.b
    return (b.out_ids.tolist(), b.acc_log.tolist(), b.status.tolist(), int(b.p0), int(b.a_new))


def test_poll_plan_bounds_and_schedule():
    from runtime.dspark_loop import PollPlan

    p = PollPlan(max_new=20, poll_every=4, lookahead=1, tail_sync=True)
    # with no observation bound(s) = 1 + 6 s: 19 < 20 at s = 3, 25 >= 20 at s = 4
    assert [s for s in range(1, 9) if p.poll_now(s)] == [4, 5, 6, 7, 8]
    assert not p.length_end_possible(3) and p.length_end_possible(4)
    assert p.poll_now(4) and not p.poll_now(1) and not p.poll_now(2) and not p.poll_now(3)
    p.observe(2, 9, False)                                    # 9 tokens after 2 steps: bound(s) = 9 + 6 (s - 2)
    assert p.n_out_bound(2) == 9 and p.n_out_bound(4) == 21 and p.length_end_possible(4) and not p.length_end_possible(3)
    p.observe(1, 100, True)                                   # an older observation never rewinds the bound ...
    assert p.obs_step == 2 and p.obs_n_out == 9 and p.done   # ... but done is sticky
    off = PollPlan(max_new=20, poll_every=8, lookahead=1, tail_sync=False)
    assert [s for s in range(1, 25) if off.poll_now(s)] == [8, 16, 24] and not off.must_drain(10)
    assert not PollPlan(20).must_drain(0)                      # nothing issued: nothing to wait for
    assert PollPlan(20, 1, 1, True).must_drain(4) and not PollPlan(20, 1, 1, True).must_drain(3)


def test_poll_plan_python_cost_is_microseconds():
    """The host-side policy (no CUDA) costs well under a microsecond-scale budget per step: the poll's real cost is the
    event + pinned copy, measured on the box by `bench/headline.py --poll-probe` / res['poll']['enqueue_us']."""
    import time

    from runtime.dspark_loop import PollPlan

    p = PollPlan(256, 1, 1, True)
    t0 = time.perf_counter()
    n = 20000
    for s in range(1, n + 1):
        p.poll_now(s % 300)
        p.must_drain(s % 300)
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"PollPlan: {us:.2f} us per step (poll_now + must_drain, CPU)")
    assert us < 50


@pytest.mark.parametrize("policy", [(1, 1, True), (1, 1, False), (1, 0, True), (1, 0, False), (1, 2, True),
                                    (4, 1, True), (8, 1, False), (8, 1, True), (3, 2, False)])
def test_async_poll_streams_identical_and_overrun_bounds(policy):
    P, LA, TAIL = policy
    n_cases = n_eos = n_len = 0
    for max_new in (2, 5, 16, 33, 64, 200):
        for pattern_seed, ns in enumerate(([3, 0, 5, 2], [0], [5], [1, 4, 2, 0, 3], [2, 2, 5, 0, 1, 1])):
            for eos_step in (None, 1, 2, 3, 4, 5, 8, 9, 17, 40):
                steps = _scripted_steps(ns, eos_step, n_steps=max_new + 40, seed=pattern_seed)
                truth = _truth(steps, max_new)
                j = len(truth.acc)
                by_eos = any(x == 99 for x in truth.out[1:])
                loop, issued, stats, cap = _run_async(steps, max_new, P, LA, TAIL)
                n_cases += 1
                n_eos += by_eos
                n_len += not by_eos
                r = loop.b.read()
                # (1) the stream and the accounting are the host truth's, whatever the schedule
                assert r["done"] and r["ids"] == truth.out and r["accepts"] == truth.acc, (policy, max_new, ns, eos_step)
                # (2) no-op steps change nothing: the final device state equals the state after exactly j live steps
                ref = _async_sim_cls()(steps, max_new, 1, 0, False)
                for _ in range(j):
                    ref.step_once()
                assert _state(loop) == _state(ref), (policy, max_new, ns, eos_step)
                # (3) overrun bounds
                over = issued - j
                assert over >= 0 and issued <= cap
                if not TAIL:
                    assert over == min(P * (-(-j // P) + LA) - j, cap - j), (policy, max_new, j, over)
                else:
                    assert over <= P * (-(-j // P) + LA) - j, (policy, max_new, j, over)
                    if P == 1:
                        assert over <= LA, (policy, max_new, ns, eos_step, j, over)
                        if not by_eos:
                            assert over == 0, (policy, max_new, ns, eos_step, j, over)
                # (4) the poll bookkeeping
                assert stats["polls"] == loop.records and loop.waits <= loop.records
    assert n_eos > 100 and n_len > 100, (n_eos, n_len)


def test_default_schedule_overrun_is_one_step_and_zero_for_length_endings():
    steps = _scripted_steps([3, 1, 4], None, n_steps=300)
    for max_new in (16, 64, 128):
        loop, issued, stats, _ = _run_async(steps, max_new, 1, 1, True)
        j = len(_truth(steps, max_new).acc)
        assert issued == j and stats["drains"] >= 1                      # length ending: no step after done
    steps = _scripted_steps([3, 1, 4], eos_step=11, n_steps=300)
    loop, issued, stats, _ = _run_async(steps, 256, 1, 1, True)
    assert issued == 11 + 1                                              # EOS ending: exactly `lookahead` steps late
    assert stats["polls"] == issued                                      # one poll per step ...
    assert stats["drains"] == 0                                          # ... and never a drain far from max_new
    old, issued_old, _, _ = _run_async(steps, 256, 8, 1, False)
    assert issued_old == 8 * (-(-11 // 8) + 1) == 24                     # the pre-fix schedule: 13 no-op steps


def test_async_drain_edge_cases():
    """done already at the first poll; max_new = 2 (one step reaches it); a sequence that ends at the cap's edge."""
    steps = _scripted_steps([5], None, n_steps=50)
    for max_new in (2, 3, 7):
        for pol in ((1, 1, True), (1, 0, True), (8, 1, True), (8, 1, False)):
            loop, issued, _, cap = _run_async(steps, max_new, *pol)
            truth = _truth(steps, max_new)
            assert loop.b.read()["ids"] == truth.out and issued <= cap
    steps = _scripted_steps([0], None, n_steps=400)                      # 1 token/step: the longest sequences
    for pol in ((1, 1, True), (8, 1, False)):
        loop, issued, _, cap = _run_async(steps, 64, *pol)
        assert loop.b.read()["done"] and issued <= cap and loop.b.read()["n_out"] == 64


FIXTURE_20261108 = os.path.join(REPO, "docs", "headline-20261108-main-ss.json")


@pytest.mark.skipif(not os.path.exists(FIXTURE_20261108), reason="needs docs/headline-20261108-main-ss.json")
def test_fixture_overrun_reproduced_and_fixed():
    """The 11/08 headline's per-prompt accept plans replayed through the real LoopBuffers + the old schedule (poll 8,
    lookahead 1) reproduce its measured steps_issued exactly (97 overrun steps over the 8 prompts) -- the simulation
    is the measured behaviour; the new default then overruns <= 1 step per prompt (0 for max_new endings)."""
    import json

    d = json.load(open(FIXTURE_20261108))
    max_new = d["max_new"]
    old_total = new_total = 0
    rows = []
    for tag, p in d["prompts"].items():
        acc = p["accepts"]
        steps = []
        for i, n in enumerate(acc):
            tok, t = _case(n, 77 + i, V=60)
            if i == len(acc) - 1 and p["finish_reason"] == "stop":
                t = t.clone()
                t[n] = 99
            steps.append((tok, t))
        steps += [_case(0, 9000 + i, V=60) for i in range(40)]            # never reached
        assert len(_truth(steps, max_new).acc) == len(acc), tag           # the plan ends where the real run ended
        _, issued_old, _, _ = _run_async(steps, max_new, 8, 1, False)
        assert issued_old == p["steps_issued"][0], (tag, issued_old, p["steps_issued"])
        loop, issued_new, stats, _ = _run_async(steps, max_new, 1, 1, True)
        assert issued_new - len(acc) <= 1
        if p["finish_reason"] == "length":
            assert issued_new == len(acc), tag
        old_total += issued_old - len(acc)
        new_total += issued_new - len(acc)
        rows.append((tag, len(acc), p["finish_reason"], issued_old - len(acc), issued_new - len(acc), stats["drains"]))
    assert old_total == 97
    assert new_total == sum(1 for _, _, f, *_ in rows if f == "stop")
    print("LOOP-OVERRUN fixture replay (prompt, live steps, ended by, overrun old, overrun new, drains):")
    for row in rows:
        print("  ", row)
    print(f"  total overrun: old {old_total} steps, new {new_total} steps")


# ---- bench/headline.py: the decode-wall block
def _wall_run(ids_n, live, issued, step_ms, wall, overrun_ms, prefill=3800.0, poll=None):
    r = {"token_ids": list(range(ids_n)), "accepts": [2] * live, "steps": live, "steps_issued": issued,
         "overrun_steps": issued - live, "step_ms": [step_ms] * live, "overrun_ms": [step_ms] * (issued - live),
         "decode_wall_ms": wall, "decode_wall_ms_excl_overrun": wall - overrun_ms, "prefill_ms": prefill}
    if poll is not None:
        r["poll"] = poll
    return r


def test_decode_wall_block():
    B = _headline()
    pl = {"polls": 10, "drains": 1, "enqueue_us": 80.0, "wait_ms": 5.0, "poll_every": 1, "lookahead": 1, "tail_sync": True}
    a = _wall_run(21, 10, 11, 18.0, 10 * 18.0 + 18.0 + 1.0, 18.0, poll=pl)       # 20 decoded tokens, 1 overrun step
    b = _wall_run(11, 5, 6, 20.0, 5 * 20.0 + 20.0 + 0.5, 20.0, poll=pl)         # 10 decoded tokens
    w = B.decode_wall([a, b])
    assert w["decode_tokens"] == 30 and w["overrun_steps"] == [1, 1] and w["overrun_steps_total"] == 2
    assert w["overrun_steps_max"] == 1 and w["overrun_ms_total"] == 38.0
    wall_excl = (199.0 - 18.0) + (120.5 - 20.0)
    assert abs(w["ms_per_token_wall_excl_overrun"] - wall_excl / 30) < 1e-12
    assert abs(w["ms_per_token_wall_incl_overrun"] - (199.0 + 120.5) / 30) < 1e-12
    assert abs(w["ms_per_token_gpu"] - (180.0 + 100.0) / 30) < 1e-12
    assert abs(w["host_gap_ms_per_token"] - (wall_excl - 280.0) / 30) < 1e-12 and w["host_gap_ms_per_token"] < 0.1
    assert abs(w["overrun_ms_per_token"] - 38.0 / 30) < 1e-12
    assert w["ttft_ms"]["mean"] == 3800.0 and w["poll"]["polls"] == 20 and w["poll"]["lookahead"] == 1
    assert w["poll_enqueue_us_per_poll"] == 8.0
    assert w["gate"]["overrun_max_le_2"] is True and w["gate"]["host_gap_le_0.1_ms_per_token"] is True
    bad = B.decode_wall([_wall_run(21, 10, 13, 18.0, 10 * 18.0 + 3 * 18.0 + 9.0, 54.0)])
    assert bad["gate"]["overrun_max_le_2"] is False and bad["gate"]["host_gap_le_0.1_ms_per_token"] is False
    # runs from before LOOP-OVERRUN (no overrun_steps / wall keys): overrun falls back to issued - steps, the rest is None
    old = B.decode_wall([{"token_ids": [1, 2, 3], "accepts": [1], "steps": 1, "steps_issued": 9, "prefill_ms": 1.0}])
    assert old["overrun_steps"] == [8] and old["ms_per_token_wall_excl_overrun"] is None and old["host_gap_ms_per_token"] is None
    assert B.decode_wall([])["overrun_steps_max"] is None


def test_decode_wall_lines_and_summary():
    B = _headline()
    pl = {"polls": 10, "drains": 1, "enqueue_us": 80.0, "wait_ms": 5.0, "poll_every": 1, "lookahead": 1, "tail_sync": True}
    w = B.decode_wall([_wall_run(21, 10, 11, 18.0, 199.0, 18.0, poll=pl)])
    line = B.decode_wall_line("two_graphs", w)
    assert "excl overrun" in line and "incl overrun" in line and "overrun steps 1 total, max 1" in line
    assert "TTFT 3800.0 ms" in line and "poll every 1 lookahead 1 tail_sync True" in line
    assert "gate overrun<=2 ok, host gap<=0.1 ok" in line
    assert B.decode_wall_line("x", None) is None


def test_headline_parser_poll_flags_and_probe_report():
    B = _headline()
    ap = B.build_parser()
    a = ap.parse_args(["--model-dir", "/m"])
    assert (a.poll_every, a.lookahead, a.tail_sync, a.poll_probe) == (1, 1, True, False)
    a = ap.parse_args(["--model-dir", "/m", "--poll-every", "8", "--no-tail-sync", "--lookahead", "2"])
    assert (a.poll_every, a.lookahead, a.tail_sync) == (8, 2, False)
    assert ap.parse_args(["--poll-probe"]).model_dir is None
    rep = B.poll_probe_report([5.0, 7.0, 6.0], [20.0, 30.0, 25.0], 18.0,
                              {"none": [18.0, 18.0, 18.0], "la0": [18.06, 18.07, 18.05], "la1": [18.0, 18.0, 18.001],
                               "batch8": [18.0, 18.0, 18.0]})
    assert abs(rep["lookahead0_bubble_us_per_step"] - 60.0) < 1e-6
    assert abs(rep["breakeven_steps_lookahead0_vs_one_overrun_step"] - 300.0) < 1e-3
    assert rep["patterns"]["la1"]["bubble_us_per_step_vs_none"] < 1.0 and rep["idle_enqueue_us"]["p50"] == 6.0
    if not torch.cuda.is_available():
        assert B.poll_probe(ap.parse_args(["--poll-probe"])) == 3            # clean refusal without CUDA
