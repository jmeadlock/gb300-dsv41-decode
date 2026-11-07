"""Mac CPU tests of M1-T10 (`.venv/bin/python -m pytest -q tests/test_static_rows_cpu.py`): the 6-row teacher-forced
verify step (runtime/static_step.py rows=6), its oracle (runtime/ref_multirow.py) and the R-row MoE mapping
(runtime/moe_rows.py), on the CPU kernel emulations of tests/test_static_emulated_cpu.py.

What is proven here (plumbing; kernel fidelity is the box GPU test's job, tests/gpu/test_static_rows_gpu.py):
  * #20 ring: window_idxs_ring == the reference's window list at ring == win, and over the 256-ring it lists the same
    positions in the same order; the 1-row static step stays equal to the reference past the wrap (pos >= 300);
  * ratio-2 single-graph pooling == the reference's per-token decode path, both P0 parities (pooled values and the
    compressor state), and the oracle's spelling == both;
  * dedup_routes round trip against a host mapping (collisions, all-distinct, all-equal-across-rows);
  * oracle sanity: forward_rows' row j vs the 1-row reference step at the same position (V0 at the reference level,
    on the emulations) -- reported, and token-exact;
  * THE gate on CPU: 6-row static-eager == forward_rows bit for bit (fp32 logits, main_hidden, the compressor state,
    the caches), mini (ratio 0/2) and mini4 (ratio 0/2/1/1 + candidates), both P0 parities, ring wrapped;
  * a 1-row and a 6-row StaticStep on one model dispatch to their own MoEs.

The emulated routed-expert GEMM (fp4_gemm) is made row-by-row here (`_rowwise`) so that its rows do not depend on M:
that is the property the reference relies on when it evaluates an expert at M = count rows and the static step at M = R
-- a property of the real kernel that the GPU test's census measures, not something the CPU can show.
"""
from __future__ import annotations

import copy
import json
import sys

import pytest
import torch

from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
from runtime.loader import build_and_load
from runtime.refmodel import import_reference
from test_static_emulated_cpu import KERNELS, _fake_static_kernels, _mini4


def _rowwise(fn):
    """A GEMM emulation evaluated one activation row at a time (row values independent of M)."""
    def g(a, a_s, b, b_s, *args, **kw):
        K = a.size(-1)
        M = a.numel() // K
        av, sv = a.reshape(M, K), a_s.reshape(M, -1)
        rows = [fn(av[i:i + 1], sv[i:i + 1], b, b_s, *args, **kw) for i in range(M)]
        return torch.cat(rows, 0).view(*a.shape[:-1], -1)
    return g


@pytest.fixture
def emu(monkeypatch, tmp_path):
    R = import_reference(cpu_stubs=True)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: ([i % 97 for i in range(MINI_CONFIG["vocab_size"])],
                                     MINI_CONFIG["engram_compressed_vocab_size"]))
    for k, f in KERNELS.items():
        monkeypatch.setattr(R, k, f)
    monkeypatch.setattr(R, "fp4_gemm", _rowwise(KERNELS["fp4_gemm"]))
    monkeypatch.setattr(R, "fp8_gemm", _rowwise(KERNELS["fp8_gemm"]))
    prev = torch.get_default_dtype()

    def build(kind):
        if kind in ("mini4", "mini4w"):
            cfg, mp = _mini4()
            if kind == "mini4w":  # the exact regime at positions >= 300: every reachable position kept (#9, risk 6)
                cfg = dict(cfg, index_topk=512, candidate_topk_blocks=64, candidate_block_size=8)
            ck = make_mini_checkpoint(str(tmp_path / kind), cfg=cfg)
        elif kind == "miniw":  # mini (ratio 0/2) in the exact regime at >= 300
            ck = make_mini_checkpoint(str(tmp_path / kind), cfg=dict(MINI_CONFIG, index_topk=512))
            mp = ck["map"]
        else:
            ck = make_mini_checkpoint(str(tmp_path / kind))
            mp = ck["map"]
        model = build_and_load(ck["model_dir"], mp, "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                               log=lambda *a: None)
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
        fk = _fake_static_kernels(reg)
        monkeypatch.setitem(sys.modules, "runtime.static_kernels", fk)
        model.temperature = 0.0
        return model, pool, fk

    yield build
    torch.set_default_dtype(prev)
    R.MoE.forward = getattr(R.MoE, "_orig_forward", R.MoE.forward)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ #20 ring
def test_window_idxs_ring_equals_reference_at_ring_win_and_lists_same_positions_at_256():
    from runtime.static_step import RING, window_idxs_ring, window_topk_idxs_dev

    R = import_reference(cpu_stubs=True)
    win = 128
    ar = torch.arange(win)
    for pos in list(range(1, 700)) + [1023, 4095, 32767]:
        ref = R.get_window_topk_idxs(win, 1, 1, pos)
        assert torch.equal(window_idxs_ring(torch.tensor([pos]), win, win, ar), ref), pos
        assert torch.equal(window_topk_idxs_dev(torch.tensor([pos]), win, ar), ref)
        got = window_idxs_ring(torch.tensor([pos]), win, RING, ar)[0, 0]
        # same positions, same order: ring slot s of the 128-ring = position p with p % 128 == s, p in (pos-128, pos]
        want_pos = torch.where(ref[0, 0] >= 0, pos - torch.remainder(pos - ref[0, 0].long(), win), -1)
        got_pos = torch.where(got >= 0, pos - torch.remainder(pos - got.long(), RING), -1)
        assert torch.equal(got_pos, want_pos), pos
    # 6 rows: row j == the 1-row list at P0 + j
    for p0 in (1, 2, 125, 126, 127, 128, 255, 256, 300, 301):
        rows = window_idxs_ring(torch.arange(p0, p0 + 6), win, RING, ar)
        for j in range(6):
            assert torch.equal(rows[0, j], window_idxs_ring(torch.tensor([p0 + j]), win, RING, ar)[0, 0])


def test_multirow_window_idxs_point_at_the_same_positions():
    from runtime.ref_multirow import multirow_window_idxs

    R = import_reference(cpu_stubs=True)
    win = 128
    for p0 in (1, 5, 127, 128, 129, 300):
        got = multirow_window_idxs(win, 6, p0)[0]
        for j in range(6):
            ref = R.get_window_topk_idxs(win, 1, 1, p0 + j)[0, 0]  # 1-row slots at P0+j (ring holds <= P0+j)
            pos_ref = torch.where(ref >= 0, p0 + j - torch.remainder(p0 + j - ref.long(), win), -1)
            g = got[j].long()
            pos_got = torch.where(g < 0, -1, torch.where(g < win, p0 - 1 - torch.remainder(p0 - 1 - g, win),
                                                         p0 + g - win))
            assert torch.equal(pos_got, pos_ref), (p0, j)


# ------------------------------------------------------------------------------------------------ ratio-2 pooling
def _compressor(R, seed=0):
    args = R.ModelArgs(dim=64, head_dim=32, rope_head_dim=8, compress_ratios=(0, 2, 0), max_batch_size=1,
                       max_seq_len=256, n_layers=2)
    torch.manual_seed(seed)
    c = R.Compressor(args, 1)
    for p in c.parameters():
        p.data.normal_()
    return c


class _Feeder(torch.nn.Module):
    """Stands in for a Compressor Linear: returns the next precomputed row (so the per-token reference sees exactly the
    6-row projection's values -- the projection's own M-dependence is V0's business, not the pooling logic's)."""

    def __init__(self):
        super().__init__()
        self.rows, self.i = None, 0

    def forward(self, x):
        r = self.rows[:, self.i:self.i + x.size(1)]
        self.i += x.size(1)
        return r


@pytest.mark.parametrize("L", [5, 6, 9, 10])  # prefill length -> first P0 parity
def test_ratio2_single_graph_pooling_equals_per_token_reference(L):
    from runtime.ref_multirow import multirow_decode
    from runtime.static_step import pool_ratio2_rows, ratio2_offset, ratio2_state_rows

    R = import_reference(cpu_stubs=True)
    seq = _compressor(R)
    st = copy.deepcopy(seq)
    orc = copy.deepcopy(seq)
    x = torch.randn(1, L + 6 * 5, 64).to(torch.bfloat16)
    with torch.inference_mode():
        for c in (seq, st, orc):
            c(x[:, :L], 0)
        fk, fg = _Feeder(), _Feeder()
        seq.wkv, seq.wgate = fk, fg
        p0 = L
        for _ in range(5):
            xs = x[:, p0:p0 + 6]
            xf = xs.float()
            kv, sc = st.wkv(xf), st.wgate(xf)                     # the 6-row projection (model.py:464-465)
            fk.rows, fk.i, fg.rows, fg.i = kv, 0, sc, 0
            seq_prev_kv0, seq_prev_sc0 = st.kv_state[:1, :1].clone(), st.score_state[:1, :1].clone()
            pooled_seq = []                                       # the reference decode branch, token by token
            for i in range(6):
                y = seq(x[:, p0 + i:p0 + i + 1], p0 + i)
                if y is not None:
                    pooled_seq.append(y)
            want = torch.cat(pooled_seq, dim=1)                   # [1, 3, d] after norm
            # the static spelling (device offset, one graph)
            P = torch.tensor([p0])
            pooled = pool_ratio2_rows(st.kv_state[:1, :1], st.score_state[:1, :1], kv, sc, ratio2_offset(P) +
                                      torch.arange(6))
            sidx = ratio2_state_rows(P, 6)
            st.kv_state[:1].copy_(kv.index_select(1, sidx))
            st.score_state[:1].copy_(sc.index_select(1, sidx))
            got = st.norm(pooled.to(xs.dtype))
            # fp32, pre-norm: the decode expression (model.py:482) on each group's (state0, state1) pair == the
            # prefill expression (473-475) the single graph uses -- the bf16 norm output above would hide an ulp
            seq_kv = torch.cat([seq_prev_kv0, kv], dim=1)[:, (ratio2_offset(P) + torch.arange(6))]
            seq_sc = torch.cat([seq_prev_sc0, sc], dim=1)[:, (ratio2_offset(P) + torch.arange(6))]
            for gi in range(3):
                dec = (seq_kv[:, 2 * gi:2 * gi + 2] * seq_sc[:, 2 * gi:2 * gi + 2].softmax(dim=1)).sum(dim=1)
                assert torch.equal(dec, pooled[:, gi]), (L, p0, gi)
            # the oracle's spelling (its own 6-row projection: same weights, same rows)
            with multirow_decode():
                got_o = orc(xs, p0)
            assert want.shape == got.shape == got_o.shape == (1, 3, 32)
            assert torch.equal(got, want) and torch.equal(got_o, want), (L, p0)
            for c in (st, orc):
                assert torch.equal(c.kv_state, seq.kv_state) and torch.equal(c.score_state, seq.score_state), p0
            p0 += 6


# ------------------------------------------------------------------------------------------------ dedup mapping
@pytest.mark.parametrize("case", ["random", "distinct", "same"])
def test_dedup_routes_round_trip(case):
    from runtime.moe_rows import dedup_routes, dedup_routes_host

    g = torch.Generator().manual_seed(3)
    R_, k, E = 6, 6, 16
    for trial in range(40):
        if case == "random":
            idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R_)])
        elif case == "distinct":
            idx = torch.randperm(R_ * k + 4, generator=g)[:R_ * k].view(R_, k)
        else:
            idx = torch.randperm(E, generator=g)[:k].expand(R_, k).clone()
            idx = idx[:, torch.randperm(k, generator=g)]
        w = torch.rand(R_, k, generator=g)
        for M in (R_, 128):
            m = dedup_routes(idx, w, M)
            host = dedup_routes_host(idx.tolist(), w.tolist())
            U = len(host)
            uniq = m["uniq"].tolist()
            assert uniq[:U] == list(host) and all(u == max(host) for u in uniq[U:])
            assert m["counts"].tolist()[:U] == [len(v) for v in host.values()] and sum(m["counts"].tolist()[U:]) == 0
            assert m["valid"].sum() == U
            # every pair lands in its expert's slot, ranks 0..cnt-1 in ascending row order, with its weight
            seen = {}
            for t in range(R_ * k):
                u, rk, r = int(m["slot"][t]), int(m["rank"][t]), int(m["prow"][t])
                assert uniq[u] == list(host)[u] and int(m["arow"][t]) == u * M + rk
                seen.setdefault(u, []).append((rk, r, float(m["pw"][t])))
            for u, (e, lst) in enumerate(host.items()):
                assert sorted(seen[u]) == [(i, r, wt) for i, (r, j, wt) in enumerate(lst)]
            # combine order: row r's A rows in ascending expert id
            arow_of = {(int(m["prow"][t]), uniq[int(m["slot"][t])]): int(m["arow"][t]) for t in range(R_ * k)}
            for r in range(R_):
                want = [arow_of[(r, e)] for e in sorted(idx[r].tolist())]
                assert m["posmat"][r].tolist() == want, (case, trial, r)


# ------------------------------------------------------------------------------------------------ the 6-row step
def _prefill(model, ids):
    with torch.inference_mode():
        out, logits, _ = model(torch.tensor([ids]), 0)
    return int(out.reshape(-1)[0])


def _ref_seq_run(model, ids, toks, n_steps, rows):
    """The 1-row reference over the teacher-forced tokens: logits at positions L .. L + rows*n_steps - 1."""
    _prefill(model, ids)
    out = []
    with torch.inference_mode():
        for i in range(rows * n_steps):
            _, lg, _ = model(torch.tensor([[toks[i]]]), len(ids) + i)
            out.append(lg[0].clone())
    return out


def _oracle_run(model, ids, toks, n_steps, rows, hidden=()):
    from runtime.ref_multirow import forward_rows

    _prefill(model, ids)
    out, hid = [], []
    for s in range(n_steps):
        _, lg, h = forward_rows(model, torch.tensor([toks[s * rows:(s + 1) * rows]]), len(ids) + s * rows, hidden)
        out += [lg[j].clone() for j in range(rows)]
        hid.append(None if h is None else h.clone())
    return out, hid, _state(model)


def _static_run(model, st, ids, toks, n_steps, graph=False):
    _prefill(model, ids)
    out, hid = [], []
    with torch.inference_mode():
        for s in range(n_steps):
            lg = st.step(toks[s * st.rows:(s + 1) * st.rows], len(ids) + s * st.rows, graph=graph)
            out += [lg[j].clone() for j in range(st.rows)]
            hid.append(None if st.main_hidden is None else st.main_hidden.clone())
            assert torch.equal(st.t_out, lg.argmax(-1))
    return out, hid, _state(model)


def _state(model):
    """The caches a step writes besides the window ring: compressor state, compressed KV, index K, engram hash."""
    out = {}
    for i, l in enumerate(model.layers):
        a = l.attn
        if a.compressor is not None and a.compressor.compress_ratio > 1:
            out[f"L{i}.kv_state"] = a.compressor.kv_state.clone()
            out[f"L{i}.score_state"] = a.compressor.score_state.clone()
        if a.is_kv_source:
            out[f"L{i}.ckv"] = a.compress_kv_cache.clone()
        if a.indexer is not None and a.indexer.owns_k:
            out[f"L{i}.k_cache"] = a.indexer.k_cache.clone()
    if model.engram_hash is not None:
        out["engram"] = model.engram_hash.cache.clone()
    return out


def _tf_tokens(L, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 2000, (L,), generator=g).tolist(), torch.randint(2, 2000, (n,), generator=g).tolist()


def _assert_rows_equal(a, b, what):
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)]
    assert len(a) == len(b) and not bad, (what, bad[:8], len(bad),
                                          float((a[bad[0]] - b[bad[0]]).abs().max()) if bad else 0)


@pytest.mark.parametrize("kind,L", [("mini", 17), ("mini", 18), ("mini4", 17), ("mini4", 18), ("mini4w", 300),
                                    ("mini4w", 301), ("miniw", 300)])
def test_static6_equals_reference_multirow_oracle(emu, kind, L):
    """THE gate (emulated): 6-row static-eager == forward_rows, fp32 logits per row, main_hidden, and the caches;
    P0 parity both ways (L odd/even, then +6 keeps it), ring wrapped at L >= 300."""
    from runtime.static_step import StaticStep

    model, pool, fk = emu(kind)
    hid = (len(model.layers) - 1,)
    n = 4
    ids, toks = _tf_tokens(L, 6 * n, 11 + L)
    st = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, _allow_cpu=True)
    o_lg, o_h, o_state = _oracle_run(model, ids, toks, n, 6, hid)
    s_lg, s_h, s_state = _static_run(model, st, ids, toks, n)
    _assert_rows_equal(s_lg, o_lg, f"{kind} L={L}: static6 vs oracle logits")
    assert all(torch.equal(a, b) for a, b in zip(s_h, o_h)), "main_hidden"
    for k_ in o_state:
        assert torch.equal(s_state[k_], o_state[k_]), (kind, L, k_)
    assert st.ring_positions_equal(L + 6 * n - 1) is not None  # helper callable (rings re-synced by the last prefill)
    rc = st.routed_counts()
    assert rc is not None and sum(rc[0]) + sum(rc[1]) > 0
    st.detach()


def _rowwise_torch(monkeypatch):
    """docs/M1-DSPARK-DESIGN.md 4's row-invariant mode on the CPU: torch's F.linear and the reference's two einsums
    (indexer score bshd,btd->bsht; wo_a bsgd,grd->bsgr) evaluated one query row at a time."""
    import torch.nn.functional as F_

    lin0, ein0 = F_.linear, torch.einsum

    def linear(x, w, b=None):
        K = x.size(-1)
        M = x.numel() // K
        if M <= 1:
            return lin0(x, w, b)
        xv = x.reshape(M, K)
        return torch.cat([lin0(xv[i:i + 1], w, b) for i in range(M)], 0).view(*x.shape[:-1], -1)

    def einsum(eq, *ops):
        a = ops[0]
        if eq.replace(" ", "").startswith("bs") and a.dim() >= 3 and a.size(1) > 1:
            return torch.cat([ein0(eq, a[:, i:i + 1], *ops[1:]) for i in range(a.size(1))], 1)
        return ein0(eq, *ops)

    monkeypatch.setattr(F_, "linear", linear)
    monkeypatch.setattr(torch, "einsum", einsum)


@pytest.mark.parametrize("kind", ["mini", "mini4"])
def test_oracle_rows_vs_sequential_reference_v0_emulated(emu, kind, monkeypatch):
    """V0 at the reference level: forward_rows' row j vs the 1-row reference at P0 + j.
    (a) as is: CPU torch matmuls are NOT row-invariant (M=6 vs M=1 differ by fp32 ulps; on random mini weights an ulp
        can flip a MoE routing near-tie) -> reported only;
    (b) row-invariant mode (F.linear/einsum row by row): the oracle's ring, compressor, indexer and head logic must
        then reproduce the sequential 1-row reference BIT FOR BIT -- this is what makes forward_rows a faithful
        6-row reference -- and the 6-row static step must equal the 1-row static step (V0 strict on the CPU)."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu(kind)
    L, n = 21, 3
    ids, toks = _tf_tokens(L, 6 * n, 5)
    seq = _ref_seq_run(model, ids, toks, n, 6)
    orc, _, _ = _oracle_run(model, ids, toks, n, 6)
    eq = sum(torch.equal(a, b) for a, b in zip(seq, orc))
    md = max(float((a - b).abs().max()) for a, b in zip(seq, orc))
    print(f"  V0 as-is (CPU emulations, {kind}): {eq}/{len(seq)} rows bit-equal to the 1-row reference, "
          f"max |dlogit| {md:.3e}")
    _rowwise_torch(monkeypatch)
    seq = _ref_seq_run(model, ids, toks, n, 6)
    orc, _, o_state = _oracle_run(model, ids, toks, n, 6)
    _assert_rows_equal(orc, seq, f"{kind}: row-invariant mode, oracle vs sequential 1-row reference")
    s1 = StaticStep(model, pool, max_pos=255, _allow_cpu=True)
    s6 = StaticStep(model, pool, max_pos=255, rows=6, _allow_cpu=True)
    a6, _, s_state = _static_run(model, s6, ids, toks, n)
    _prefill(model, ids)
    a1 = []
    with torch.inference_mode():
        for i in range(6 * n):
            a1.append(s1.step(toks[i], L + i, graph=False)[0].clone())
    _assert_rows_equal(a6, a1, f"{kind}: row-invariant mode, static6 vs static1")
    _assert_rows_equal(a6, seq, f"{kind}: row-invariant mode, static6 vs reference")
    print(f"  V0 row-invariant mode ({kind}): oracle == static6 == static1 == 1-row reference, {len(seq)} rows")
    s1.detach()
    s6.detach()


def test_static1_past_the_ring_wrap_equals_reference(emu):
    """#20 on the 1-row path: the 256-ring at positions 300..309 (both wrapped rings) == the reference eager, in the
    exact indexer regime (mini4w: every reachable position kept; the truncated regime is risk 6, reported below)."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4w")
    st = StaticStep(model, pool, max_pos=511, _allow_cpu=True)
    ids, _ = _tf_tokens(300, 0, 7)
    ref, toks = [], [_prefill(model, ids)]
    with torch.inference_mode():
        for k in range(10):
            out, lg, _ = model(torch.tensor([[toks[-1]]]), 300 + k)
            ref.append(lg[0].clone())
            toks.append(int(out[0]))
    got, gt = [], [_prefill(model, ids)]
    with torch.inference_mode():
        for k in range(10):
            lg = st.step(gt[-1], 300 + k, graph=False)
            got.append(lg[0].clone())
            gt.append(int(st.tok.reshape(-1)[0]))
    _assert_rows_equal(got, ref, "1-row static vs reference past the wrap")
    assert gt == toks
    st.detach()


def test_one_and_six_row_steps_share_a_model(emu):
    """Two StaticSteps on one model: each step's MoE dispatch reaches its own objects (patches.active_static_step)."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini")
    s1 = StaticStep(model, pool, max_pos=255, _allow_cpu=True)
    s6 = StaticStep(model, pool, max_pos=255, rows=6, _allow_cpu=True)
    ids, toks = _tf_tokens(20, 12, 9)
    o_lg, _, _ = _oracle_run(model, ids, toks, 2, 6)
    a_lg, _, _ = _static_run(model, s6, ids, toks, 2)
    _assert_rows_equal(a_lg, o_lg, "6-row with a 1-row step built on the same model")
    _prefill(model, ids)
    with torch.inference_mode():
        lg1 = s1.step(toks[0], 20, graph=False)[0].clone()
        _, ref1, _ = model(torch.tensor([[toks[0]]]), 20)
    assert torch.equal(lg1, ref1[0])
    s1.detach()
    s6.detach()
    assert not model.layers[0].ffn.__dict__.get("_static_moes")


def test_rows_validation(emu):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini")
    for bad in (3, 5, 10):
        with pytest.raises(ValueError):
            StaticStep(model, pool, max_pos=255, rows=bad, _allow_cpu=True)
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
    st = StaticStep(model, pool, max_pos=255, rows=6, _allow_cpu=True)
    with pytest.raises(ValueError):
        st.set_inputs([1, 2, 3, 4, 5, 6], 251)  # 251 + 5 > 255
    with pytest.raises(ValueError):
        st.set_inputs([1, 2, 3], 10)
    st.detach()


def test_truncated_topk_regime_reported(emu):
    """Risk 6 / design risk 3 (reachable compressed positions > index_topk: mini4 at P >= 300, k = 64): the selected
    set depends on score bits from a W-wide einsum (static) vs an end_pos-wide one (reference/oracle), and on topk's
    tie order. On the CPU emulations this regime is NOT bit-equal even at 1 row (pre-existing, T8 #9); reported here,
    gated nowhere -- the box GPU test measures it with the real kernels."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")
    st = StaticStep(model, pool, max_pos=511, rows=6, _allow_cpu=True)
    ids, toks = _tf_tokens(300, 18, 13)
    o_lg, _, _ = _oracle_run(model, ids, toks, 3, 6)
    s_lg, _, _ = _static_run(model, st, ids, toks, 3)
    eq = sum(torch.equal(a, b) for a, b in zip(s_lg, o_lg))
    agree = sum(int(a.argmax()) == int(b.argmax()) for a, b in zip(s_lg, o_lg))
    print(f"  truncated regime (mini4, P0 300..317): {eq}/18 rows bit-equal static6 vs oracle, argmax agree {agree}/18")
    assert eq >= 1  # the first row of the first step selects before any width effect can compound
    st.detach()


def test_generator_teacher_forced_records(emu):
    """The driver (runtime/generate.py rows=6): teacher-forced records in eager (= the oracle, ref_multirow) and static
    mode are byte-identical; under row_invariant_mode both are byte-identical to the 1-row greedy record that supplied
    the targets (V0 strict on the CPU; 23 targets = 1 + 3*6 + 4: the last step padded)."""
    from runtime.generate import Generator, dump_record, record_divergence
    from runtime.row_census import row_invariant_mode
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    model, pool, _ = emu("mini4")
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", max_pos=255,
                    static_factory=lambda m, **kw: StaticStep(m, pool, max_pos=255, _allow_cpu=True, **kw))
    ids = list(range(40, 57))
    with row_invariant_mode():
        g1 = gen.greedy(ids, 23, stop_on_eos=False)
        targets = gen.last_meta["token_ids"]
        gen.set_rows(6)
        with pytest.raises(RuntimeError):
            gen.step(5, 20)
        tf_e = gen.teacher_forced(ids, targets)
        me = dict(gen.last_meta)
        gen.set_mode("static")
        tf_s = gen.teacher_forced(ids, targets)
        ms = dict(gen.last_meta)
    assert gen.static.rows == 6 and me["n_steps"] == ms["n_steps"] == 4
    assert dump_record(tf_e) == dump_record(tf_s) and me["step_sha256"] == ms["step_sha256"]
    assert record_divergence(tf_s, g1) is None, record_divergence(tf_s, g1)
    assert ms["argmax_agree"] == len(targets) and ms["first_disagreement"] is None
    # outside the mode: eager (oracle) == static still holds bit for bit; vs the 1-row record it is drift
    gen.set_mode("eager")
    a = gen.teacher_forced(ids, targets)
    gen.set_mode("static")
    b = gen.teacher_forced(ids, targets)
    assert dump_record(a) == dump_record(b)
    print(f"  driver (as-is vs the 1-row record made inside the mode, so the prefill differs too): first divergence {record_divergence(b, g1)}; argmax agree "
          f"{gen.last_meta['argmax_agree']}/{len(targets)}")
    gen.set_rows(1)
    assert gen.static is None


def test_census_runs_on_cpu_emulations(emu):
    """runtime/row_census.py on the Mac: every family either runs or says why not (CUDA-only kernels -> n/a); the
    row-wise emulations come out row-invariant; the table prints."""
    from runtime.row_census import census, census_table

    model, pool, _ = emu("mini4")
    recs = census(model, rows=6, gemv_timing=False)
    for line in census_table(recs):
        print("  " + line)
    fam = {r["family"]: r for r in recs}
    assert fam["act_quant"]["equal"] is True
    assert fam["fp4 expert GEMM (ref fp4_gemm)"]["equal"] is True  # _rowwise emulation
    assert fam["fp8 GEMV tuned (M=6 cfg)"]["note"].startswith("n/a")
    assert all(r["equal"] is not None or r["note"].startswith("n/a") for r in recs), [r for r in recs
                                                                                       if r["equal"] is None]


def test_first_light_rows_teacher_forced_measure(emu, tmp_path, capsys):
    """bench/first_light.py --rows 6 --teacher-forced on the emulated mini4 (static mode): phase 1 1-row records + a
    1-row cell of 6 x n steps, phase 2 teacher-forced reps + self-replay + V0 vs the 1-row records + the 6-row cell
    (R6/R1) + the census; the 12-line rows summary."""
    import importlib.util
    import os

    from runtime.generate import Generator
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(repo, "bench", "first_light.py"))
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    model, pool, _ = emu("mini4")
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "static", "--rows", "6", "--teacher-forced",
                                      "--fixture", str(tmp_path / "fx.jsonl"), "--reps", "2", "--max-new", "9",
                                      "--step-cell", "3", "--stats-from", "1", "--skip-parity", "--max-pos", "255",
                                      "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        fl.main(["--model-dir", "x", "--rows", "6"])  # --rows > 1 without --teacher-forced: refused before boot
    with open(tmp_path / "fx.jsonl", "w") as f:
        for tag, text in (("prose", "w5 w9 w12 w3 w7"), ("json", "w40 w41 w42")):
            f.write(json.dumps({"tag": tag, "prompt": text}) + "\n")
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                    max_pos=255, static_factory=lambda m, **kw: StaticStep(m, pool, max_pos=255, _allow_cpu=True, **kw))
    gen.encode = lambda msgs: [7, 8] + [int(w[1:]) for w in msgs[0]["content"].split()]  # no chat template here
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "static", "fp8_impl": "ref",
                                                                                 "glue_impl": "ref", "attn_impl": "ref"}}
    rc = fl.measure_rows(gen, a, report, out=a.out, json_path=a.json, fixture2=None)
    rm = report["rows_mode"]
    assert rc == 0, report["errors"]
    assert rm["self_replay"][2]["identical"] == rm["self_replay"][2]["n"] == 2
    assert rm["vs_r1"]["positions"] == 18 and rm["step_cell_r1"]["n_steps"] == 18
    sc = rm["step_cell_rows"]
    assert sc["n_steps"] == 3 and sc["argmax_compared"] == 18 and sc["r6_over_r1"] > 0
    assert any(c["family"] == "act_quant" for c in rm["v0_census"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | rows 6 teacher-forced | mode static")
    print("\n".join(lines[-12:]))


def test_gpu_test_logic_dry_run_on_cpu(emu, monkeypatch, capsys):
    """tests/gpu/test_static_rows_gpu.py's gate / wrap / V0 functions executed on the CPU emulations (DEV "cpu"): the
    box run then only meets the real kernels, not harness bugs. (Graph capture and the fast impls are CUDA-only.)"""
    import contextlib
    import importlib.util
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("gpu_rows", os.path.join(repo, "tests", "gpu", "test_static_rows_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    @contextlib.contextmanager
    def model_cpu(kind="mini"):
        model, pool, _ = emu(kind)
        yield model, pool

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STATIC_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    G._gate("mini", 17, n=2)
    G._gate("mini4", 18, n=2)
    G._gate("mini4w", 301, n=2)
    G.test_static1_ring256_past_wrap_equals_reference()
    G.test_v0_census_and_step_level_table()
    G.test_first_light_rows_cells_graph(mode="static")
    out = capsys.readouterr().out
    assert out.count("== reference forward_rows") == 3 and "positions 300..311" in out
    assert "STEP-LEVEL V0" in out and "ref, row_invariant_mode" in out and "tf replay equal" in out
    print(out)
