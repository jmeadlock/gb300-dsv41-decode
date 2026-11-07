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
    with pytest.raises(ValueError, match="ExpertPool"):  # M1-T10-FIX: dg-inplace runs at 6 rows, on the pool
        StaticStep(model, None, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
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




# ------------------------------------------------------------------------------------------------ M1-T10-FIX
def test_reference_forward_independent_of_global_default_device(emu):
    """Part A on the Mac: the box failure's mechanism and its fix. The reference builds its window indices with bare
    factories on the DEFAULT device (model.py:418-423, memoized by @lru_cache(1)); round 1's GPU test ran the
    reference forward with the default device still CPU while the model and the TileLang kernel were CUDA. Here the
    global default is a foreign device ("meta") and the lru cache is seeded under it: a plain model(...) call breaks
    (the negative control), refmodel.reference_forward / forward_rows / StaticStep.capture's path do not, and give the
    same bits as under the matching default."""
    from runtime.ref_multirow import forward_rows
    from runtime.refmodel import reference_forward

    model, pool, _ = emu("mini")
    R = sys.modules["model"]
    ids, toks = _tf_tokens(17, 6, 3)
    with torch.inference_mode():
        _, want, _ = reference_forward(model, torch.tensor([ids]), 0)
        want = want.clone()
        _, want6, _ = forward_rows(model, torch.tensor([toks]), 17)
        want6 = want6.clone()
    torch.set_default_device("meta")
    try:
        R.get_window_topk_idxs(model.layers[0].attn.window_size, 1, len(ids), 0)  # a foreign cached entry
        with pytest.raises(Exception):
            with torch.inference_mode():
                model(torch.tensor([ids], device="cpu"), 0)                         # round 1's call
        with torch.inference_mode():
            _, got, _ = reference_forward(model, torch.tensor([ids], device="cpu"), 0)
            _, got6, _ = forward_rows(model, torch.tensor([toks], device="cpu"), 17)
    finally:
        torch.set_default_device("cpu")
    assert got.device.type == "cpu" and torch.equal(got, want) and torch.equal(got6, want6)


def _masked_rowwise(fake):
    """deep_gemm's masked GEMM emulated one row at a time with the contiguous fake's math (tests/test_dg_inplace_cpu
    _fake_dg_contig): CPU matmuls are not row-invariant, the real kernels' per-row results are the GPU test's D row."""
    from test_static_emulated_cpu import _fp4_deq

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe_a=None, recipe_b=None, **kw):
        (xq, xs), (W, S) = a, b
        G, max_m, K = xq.shape
        assert not kw and recipe_a == (1, 32) and recipe_b == (1, 32) and expected_m == 8 and max_m >= 128
        assert S.dtype == torch.float32 and tuple(S.shape) == (G, W.size(1), K // 32)
        for g in range(G):
            m = int(masked_m[g])
            if m:
                B = _fp4_deq(W[g]) * S[g].repeat_interleave(32, -1)
                for r in range(m):
                    A = (xq[g, r:r + 1].float().view(1, K // 32, 32) * xs[g, r:r + 1].view(1, -1, 1)).view(1, K)
                    d[g, r:r + 1] = (A @ B.t()).to(torch.bfloat16)
            d[g, m:] = float("nan")
        fake.calls.append((G, max_m, W.size(1), K, [int(v) for v in masked_m]))

    return m_grouped_fp8_fp4_gemm_nt_masked


@pytest.fixture
def rows_dg(emu, monkeypatch):
    """The rows emulation + fake deep_gemm (contiguous contract fake, row-wise masked) + the registry-backed
    static_kernels with the stage_inplace contract (re-reads the pool: dg-inplace moves its storage)."""
    from test_dg_inplace_cpu import _fake_dg_contig, _install_kernels
    from test_dg_moe_cpu import _faithful_act_quant

    fake = _fake_dg_contig()
    fake.m_grouped_fp8_fp4_gemm_nt_masked = _masked_rowwise(fake)
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)

    def build(kind):
        model, pool, _ = emu(kind)
        monkeypatch.setattr(sys.modules["model"], "act_quant", _faithful_act_quant)
        fk = _install_kernels(monkeypatch, pool)
        return model, pool, fk, fake

    return build


@pytest.mark.parametrize("kind,L", [("mini", 17), ("mini4", 18), ("mini4w", 301)])
def test_static6_dg_inplace_equals_dg_masked_emulated(rows_dg, kind, L):
    """Part B (1): the 6-row dg-inplace step == the 6-row dg-masked step, torch.equal on every row's fp32 logits,
    main_hidden and the caches (the GPU test's D row on the emulation), both P0 parities, ring wrapped; the
    contiguous calls' contract: 6k blocks, each valid block's count rows tagged with its group (hot pool slot or,
    M1-T10-FIX2, the compact spare slot H + c, S = 8 spare slots), padding and unused slots -1; the cold
    (spare-slot) path ran; no overflow."""
    from runtime.moe_rows import DGInplaceMoERows, DGMaskedMoERows
    from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS, StaticStep

    model, pool, fk, fake = rows_dg(kind)
    hid = (len(model.layers) - 1,)
    n = 3
    ids, toks = _tf_tokens(L, 6 * n, 23 + L)
    st = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-inplace", _allow_cpu=True)
    k = st.topk
    B = 6 * k
    S = min(DEFAULT_SPARE_SLOTS_ROWS, B)
    b = st.dg
    assert pool.spare == S == b.spare_limit == st.spare_slots and b.k == B
    assert all(isinstance(m, DGInplaceMoERows) for m in st.moes)
    assert b.mi.shape == (B * b.align,) and st.describe()["moe_impl"] == "dg-inplace"
    a_lg, a_h, a_s = _static_run(model, st, ids, toks, n)
    calls = list(fake.contig_calls)
    rc = st.routed_counts()
    ov = st.overflow_counts()
    st.detach()
    n_layers = len(model.layers)
    assert len(calls) == 3 * n_layers * n and fk.calls["stage"] == n_layers * n and not fake.calls
    assert ov["steps"] == 0 and ov["experts"] == 0 and ov["steps_counted"] == n
    Gs = {len(pool.hot[l]) + S for l in range(n_layers)}
    assert all(c[0] == B * b.align and c[1] in Gs for c in calls)
    assert sum(1 for c in calls for h in c[4] if h >= c[1] - S) > 0      # cold experts went through spare slots
    assert any(h < 0 for c in calls for h in c[4])                        # unused slots (U < S) tagged -1
    m2 = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-masked", _allow_cpu=True)
    assert all(type(m) is DGMaskedMoERows for m in m2.moes)
    b_lg, b_h, b_s = _static_run(model, m2, ids, toks, n)
    assert len(fake.calls) == 3 * n_layers * n
    _assert_rows_equal(a_lg, b_lg, f"{kind} L={L}: 6-row dg-inplace vs dg-masked")
    assert all(torch.equal(x, y) for x, y in zip(a_h, b_h))
    for k_ in b_s:
        assert torch.equal(a_s[k_], b_s[k_]), (kind, L, k_)
    assert m2.routed_counts() == rc and sum(rc[1]) > 0                    # same unique/cold calls per layer
    m2.detach()


def test_dg_inplace_rows_growth_invalidates_earlier_steps(rows_dg):
    """A step built before a dg-inplace build grew the pool holds stale pointer tables: it refuses to run."""
    from runtime.static_step import StaticStep

    model, pool, _, _ = rows_dg("mini")
    early = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-masked", _allow_cpu=True)
    late = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
    ids, toks = _tf_tokens(20, 6, 4)
    _prefill(model, ids)
    with pytest.raises(RuntimeError, match="storage moved"):
        early.step(toks, 20, graph=False)
    with torch.inference_mode():
        late.step(toks, 20, graph=False)
    early.detach()
    late.detach()


@pytest.fixture
def rows_glue(rows_dg, monkeypatch):
    """rows_dg + the e8m0 act_quant and the block-faithful fp8_gemm (row-wise) the fused glue's tables need."""
    from test_gemv_cpu import fp8_gemm_blocks

    monkeypatch.delenv("DSV41_GLUE_DISABLE", raising=False)

    def build(kind):
        out = rows_dg(kind)
        monkeypatch.setattr(sys.modules["model"], "fp8_gemm", _rowwise(fp8_gemm_blocks))
        return out

    return build


@pytest.mark.parametrize("moe_impl,fp8_impl", [("ref", "ref"), ("dg-inplace", "ref"), ("dg-inplace", "gemv"),
                                               ("dg-masked", "ref")])
def test_fused_glue_rows_equals_ref_glue_emulated(rows_glue, monkeypatch, moe_impl, fp8_impl):
    """M1-T10-PORT (the GPU test's G row on the Mac's torch mirrors): glue_impl fused at 6 rows -- EVERY cluster
    applicable to the impls on (self-checked at 6 rows) except the two left off (fused_glue.ROWS_WHY, n/a with a
    `rows=6:` reason) -- == glue_impl ref at 6 rows, torch.equal; one launch per cluster instance for all 6 rows."""
    from runtime import fused_glue as FG
    from runtime import fused_glue_r2 as R2
    from runtime import moe_rows as MR
    from runtime.static_step import StaticStep

    model, pool, _, _ = rows_glue("mini4")
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 31)
    st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl=moe_impl, fp8_impl=fp8_impl,
                    glue_impl="fused", _allow_cpu=True)
    d = st.glue.describe()
    never = set(FG.ROWS_WHY)
    if moe_impl == "ref":
        never |= {"dg_act", "swiglu_routed", "moe_combine"}
    if moe_impl == "dg-inplace":
        never |= {"gather"}
    else:                                              # M1-T10-MOE: dg_sf_pack is dg-inplace's (ported to R rows)
        never |= {"dg_sf_pack"}
    if fp8_impl == "ref":
        never |= {"q_rope_epilogue", "gemv_pair", "reduce_fold"}
    assert set(d["on"]) == set(FG.CLUSTERS) - never, d["clusters"]
    assert all(s.startswith("n/a") for c, s in d["clusters"].items() if c in never), d["clusters"]
    if moe_impl == "dg-inplace":   # M1-T10-MOE: on at 6 rows after its R-row self-check; the GEMMs took packed SF
        assert d["clusters"]["dg_sf_pack"] == "on" and st.dg.sf_pack, d["clusters"]
        assert d["calibration"]["r4"]["verify_dg_sf_pack"] == "equal", d["calibration"]
    assert all(v == "equal" for k_, v in d["calibration"].items() if k_.startswith("verify_")), d["calibration"]
    assert all(v == "equal" for v in d["calibration"].get("r4", {}).values()), d["calibration"]
    assert d["calibration"]["hc_post"]["picked"] is not None
    assert st.glue_note and "hc_norm" in st.glue_note and "rows 6" in st.glue.glue_line()
    names = {FG: ["launch_hc_post", "launch_swiglu_quant", "launch_o_rope_qdq", "launch_act_quant_bcast",
                  "launch_moe_combine"],
             R2: ["launch_hc_norm", "launch_norm_row", "launch_router_post", "launch_engram_gate",
                  "launch_topk_post"],
             MR: ["dedup_fused"]}
    seen = {n: 0 for ns in names.values() for n in ns}
    for mod, ns in names.items():
        for n in ns:
            real = getattr(mod, n)

            def wrap(*a, _real=real, _n=n, **k):
                seen[_n] += 1
                return _real(*a, **k)
            monkeypatch.setattr(mod, n, wrap)
    f_lg, f_h, f_s = _static_run(model, st, ids, toks, 2)
    L = len(model.layers)
    n_eng = sum(l.engram is not None for l in model.layers)
    n_idx = sum(l.attn.is_index_source for l in model.layers)
    dg = moe_impl != "ref"
    want = {"launch_hc_post": 2 * 2 * L, "launch_swiglu_quant": 2 * L * (2 if dg else 1), "launch_o_rope_qdq": 2 * L,
            "launch_act_quant_bcast": 2 * L if dg else 0, "launch_moe_combine": 2 * L if dg else 0,
            "launch_hc_norm": 2 * (2 * L + 1), "launch_norm_row": 2 * 2 * L, "launch_router_post": 2 * L,
            "launch_engram_gate": 2 * n_eng, "launch_topk_post": 2 * n_idx, "dedup_fused": 2 * L}
    assert seen == want, seen                          # per step: one launch per instance for all 6 rows
    st.detach()
    ref = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=hid, moe_impl=moe_impl, fp8_impl=fp8_impl,
                     _allow_cpu=True)
    r_lg, r_h, r_s = _static_run(model, ref, ids, toks, 2)
    _assert_rows_equal(f_lg, r_lg, f"6-row fused glue vs ref glue ({moe_impl}, {fp8_impl})")
    assert all(torch.equal(x, y) for x, y in zip(f_h, r_h))
    for k_ in r_s:
        assert torch.equal(f_s[k_], r_s[k_]), k_
    ref.detach()



def test_first_light_rows_accepts_dg_inplace_and_fused_glue(rows_glue, tmp_path, capsys):
    """`first_light.py --rows 6 --teacher-forced --moe-impl dg-inplace --glue-impl fused` (M1-T10-FIX): the 6-row step
    runs dg-inplace (no dg-masked substitution) with the row-parallel fused clusters; self-replay, the cells, the
    [glue] line of the 6-row step and the 12-line summary."""
    import importlib.util
    import os

    from runtime import fused_glue as FG
    from runtime.generate import Generator
    from runtime.static_step import StaticStep
    from tiny_tok import make_wordlevel_tokenizer

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(repo, "bench", "first_light.py"))
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    model, pool, _, _ = rows_glue("mini4")
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "static", "--rows", "6", "--teacher-forced",
                                      "--moe-impl", "dg-inplace", "--glue-impl", "fused",
                                      "--fixture", str(tmp_path / "fx.jsonl"), "--reps", "2", "--max-new", "9",
                                      "--step-cell", "2", "--stats-from", "1", "--skip-parity", "--skip-census",
                                      "--max-pos", "255", "--out", str(tmp_path / "c.jsonl"),
                                      "--json", str(tmp_path / "r.json")])
    assert fl.rows_moe_impl(a.moe_impl, a.rows_moe_impl) == "dg-inplace"
    with open(tmp_path / "fx.jsonl", "w") as f:
        for tag, text in (("prose", "w5 w9 w12 w3 w7"), ("json", "w40 w41 w42")):
            f.write(json.dumps({"tag": tag, "prompt": text}) + "\n")
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                    max_pos=255, moe_impl="dg-inplace", glue_impl="fused",
                    static_factory=lambda m, **kw: StaticStep(m, pool, max_pos=255, _allow_cpu=True, **kw))
    gen.encode = lambda msgs: [7, 8] + [int(w[1:]) for w in msgs[0]["content"].split()]
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "static", "fp8_impl": "ref",
                                                                                 "glue_impl": "fused", "attn_impl": "ref"}}
    rc = fl.measure_rows(gen, a, report, out=a.out, json_path=a.json, fixture2=None)
    rm = report["rows_mode"]
    assert rc == 0 and not report["errors"], report["errors"]
    assert rm["moe_impl_rows"] == "dg-inplace" and "step_cell_r1_rows_moe" not in rm
    assert rm["static"]["moe_impl"] == "dg-inplace" and rm["static"]["glue_effective"] == "fused"
    assert set(rm["glue_rows_on"]) == set(FG.ROWS_CLUSTERS) - {"gather", "q_rope_epilogue", "gemv_pair", "reduce_fold"}
    assert rm["dg_rows"]["spare_slots"] == 16                 # M1-T10-PORT default (min(16, R*k = 24))
    assert rm["dg_rows"]["spare_limit"] == 16 and rm["static"]["spare_slots"] == 16
    # M1-T10-FIX2 (3): the builds' records -- the 1-row dg-inplace build grew the pool to topk = 4 spare slots, the
    # 6-row build to 16 (12 more per layer, M1-T10-PORT); (2): no overflow in any 6-row step
    b1, br = rm["build_1row"], rm["build_rows"]
    per = sum(n_ * (k_ // 2) for n_, k_ in pool.proj_nk.values())
    assert b1["spare_slots"] == 4 and b1["spare_added_B"] == pool.n_layers * 4 * per and b1["spare_limit"] is None
    assert br["spare_slots"] == 16 and br["spare_added_B"] == pool.n_layers * 12 * per and br["spare_limit"] == 16
    assert rm["overflow"]["steps"] == 0 and rm["overflow"]["spare_slots"] == 16 and rm["overflow"]["steps_counted"] > 0
    assert rm["self_replay"][2]["identical"] == rm["self_replay"][2]["n"] == 2
    assert rm["step_cell_rows"]["n_steps"] == 2 and rm["step_cell_rows"]["r6_over_r1"] > 0
    out = capsys.readouterr().out
    assert "[glue] fused: rows 6" in out
    assert "[hbm] 1-row build (moe dg-inplace)" in out and "[hbm] 6-row build (moe dg-inplace, spare budget 16)" in out
    assert "[overflow] 6-row dg-inplace: 0/" in out
    lines = out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | rows 6 teacher-forced | mode static | moe dg-inplace (rows: dg-inplace)")
    assert "glue fused (rows: " in lines[-12]
    assert "| overflow steps: 0 (0 experts, spare 16/layer) |" in lines[-1] and "HBM after build: 1-row" in lines[-1]
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
    assert G.COUNTS["gates"] == 3 and G.COUNTS["logit_rows"] > 0 and G.COUNTS["caches"] > 0, G.COUNTS
    print(out)


def test_gpu_test_d_g_rows_dry_run_on_cpu(rows_glue, monkeypatch, capsys):
    """M1-T10-FIX: the GPU test's D row (6-row dg-inplace == dg-masked) and G row (fused glue == ref glue at 6 rows)
    executed on the CPU emulations (DEV "cpu"; no graphs; G on the CPU-available impls: fp8 ref, attn ref)."""
    import contextlib
    import importlib.util
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("gpu_rows2", os.path.join(repo, "tests", "gpu", "test_static_rows_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    @contextlib.contextmanager
    def model_cpu(kind="mini"):
        model, pool, _, _ = rows_glue(kind)
        yield model, pool

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STATIC_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    monkeypatch.setattr(G, "FAST", dict(moe_impl="dg-masked", fp8_impl=None, glue_impl="fused", attn_impl=None))
    real = sys.modules["deep_gemm"].m_grouped_fp8_fp4_gemm_nt_contiguous

    def finite_skips(a, b, d, m_indices, **kw):  # the forced-overflow case: skipped blocks finite, as on the GPU
        real(a, b, d, m_indices, **kw)
        d.nan_to_num_(0.0)

    G.test_static6_dg_inplace_equals_dg_masked()
    G.test_static6_fused_glue_equals_ref_glue()
    monkeypatch.setattr(sys.modules["deep_gemm"], "m_grouped_fp8_fp4_gemm_nt_contiguous", finite_skips)
    G.test_static6_dg_inplace_forced_overflow_trips_counter()
    out = capsys.readouterr().out
    n_off = len(G._rows_expected_on("dg-inplace", "ref"))    # M1-T10-PORT: the G row's each-cluster-alone-off runs
    assert out.count("6-row dg-inplace == dg-masked") == 2 and out.count("== ref glue") == 3, out
    assert f"each of {n_off} clusters alone off == ref glue" in out and "all equal" in out, out
    assert "(budget 8)" in out and "overflow steps 0" in out and "overflow S=1: eager " in out, out
    assert G.COUNTS["gates"] == 5 + n_off and G.COUNTS["logit_rows"] == (4 + n_off) * 24, G.COUNTS
    assert G.COUNTS["g_clusters"] > 0, G.COUNTS
    print(out)
