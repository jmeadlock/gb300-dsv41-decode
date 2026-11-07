"""Mac CPU tests of T8's pure parts (`.venv/bin/python -m pytest -q tests/`), against the reference's own host
functions via `runtime.refmodel.import_reference(cpu_stubs=True)`:

  * design #6: device window indices == get_window_topk_idxs for pos 1..4,096 (values AND order);
  * design #3/#13: rope rows gathered by a device index == the reference's slices; the derived write index, latent
    rope index and compress_lens == the reference's host ints; parity selection and the representative position;
  * design #12: the reference Compressor at ratio 2 driven with representative_pos(parity) == driven with the true pos
    (outputs, None-ness, kv/score state) over a decode run;
  * design #9-#11: the padded fixed-width indexer tail (topk, candidate blocks, the candidate consumer) == the
    reference's dynamic-width tail on synthetic scores, heavy ties included, wherever the design argues identity
    (reachable <= k, no tie at the k-th/(k+1)-th boundary); at a boundary tie only "both are valid top-k sets" holds
    and is what is asserted (design risk 6);
  * design #2: engram_hash_static == NgramHashState.forward over prefill + decode;
  * the Generator's eager/static/graph modes and first_light's --parity-levels table on a fake model + fake static
    step (the harness the box runs).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os

import pytest
import torch

from runtime.refmodel import import_reference
from runtime.static_step import (compressor_pools, engram_hash_static, freqs_rows, indexer_topk_static, parity_of,
                                 representative_pos, select_candidate_blocks_static, window_topk_idxs_dev)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")


@pytest.fixture(scope="module")
def R():
    return import_reference(cpu_stubs=True)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ #6 window
def test_window_idxs_match_reference_pos_1_to_4096(R):
    win = 128
    ar = torch.arange(win)
    for pos in range(1, 4097):
        ref = R.get_window_topk_idxs(win, 1, 1, pos)
        got = window_topk_idxs_dev(torch.tensor([pos]), win, ar)
        assert got.dtype == ref.dtype == torch.int32 and tuple(got.shape) == tuple(ref.shape) == (1, 1, win)
        assert torch.equal(got, ref), pos


def test_window_idxs_small_window_wraps(R):
    for win in (4, 8, 16):
        ar = torch.arange(win)
        for pos in range(1, 6 * win):
            assert torch.equal(window_topk_idxs_dev(torch.tensor([pos]), win, ar), R.get_window_topk_idxs(win, 1, 1, pos))


# ------------------------------------------------------------------------------------------------ #3, #13, #12 ints
def test_derived_positions_and_parity(R):
    for pos in range(1, 4097):
        p = torch.tensor([pos])
        assert int(torch.remainder(p, 128)) == pos % 128  # model.py:719
        for r in (1, 2):
            assert int(torch.div(p, r, rounding_mode="floor")) == pos // r       # model.py:547, 762
            assert int(torch.div(p + 1, r, rounding_mode="floor")) == (pos + 1) // r  # model.py:568, 745
            assert int(p + (1 - r)) == pos + 1 - r                               # model.py:542, 757
            assert (pos + 1) // r >= 1                                           # #7: compress_len never 0
        # #12: the graph is chosen by parity; the ratio-2 compressor pools exactly on odd pos
        assert parity_of(pos) == pos % 2
        assert compressor_pools(pos, 2) == (pos % 2 == 1)
        assert compressor_pools(pos, 1)
        rp = representative_pos(parity_of(pos))
        assert rp >= 1 and rp % 2 == pos % 2 and compressor_pools(rp, 2) == compressor_pools(pos, 2)


def test_freqs_rows_equal_reference_slices(R):
    tables = [R.precompute_freqs_cis(64, 4096, 65536, 160000, 16, 32, 1),  # compressed layers: YaRN
              R.precompute_freqs_cis(64, 4096, 0, 10000, 16, 32, 1)]       # ratio-0 layers
    for t in tables:
        tr = torch.view_as_real(t)
        for pos in list(range(1, 300)) + [1023, 2047, 4095]:
            got = freqs_rows(tr, torch.tensor([pos]))
            assert got.dtype == t.dtype and torch.equal(got, t[pos:pos + 1])                     # model.py:768, 552
            for r in (1, 2):
                assert torch.equal(freqs_rows(tr, torch.tensor([pos + 1 - r])), t[pos + 1 - r].unsqueeze(0))  # 757


# ------------------------------------------------------------------------------------------------ #12 compressor
def test_compressor_ratio2_representative_pos_equals_true_pos(R):
    args = R.ModelArgs(dim=64, head_dim=32, rope_head_dim=8, compress_ratios=(0, 2, 0), max_batch_size=1,
                       max_seq_len=128, n_layers=2)
    torch.manual_seed(0)
    c_ref = R.Compressor(args, 1)
    for p in c_ref.parameters():
        p.data.normal_()
    c_st = copy.deepcopy(c_ref)
    for L in (5, 6):
        x = torch.randn(1, L + 12, 64).to(torch.bfloat16)
        with torch.inference_mode():
            a, b = c_ref(x[:, :L], 0), c_st(x[:, :L], 0)
            assert (a is None) == (b is None) and (a is None or torch.equal(a, b))
            for pos in range(L, L + 12):
                ya = c_ref(x[:, pos:pos + 1], pos)
                yb = c_st(x[:, pos:pos + 1], representative_pos(parity_of(pos)))
                assert (ya is None) == (yb is None) == (not compressor_pools(pos, 2)), pos
                if ya is not None:
                    assert torch.equal(ya, yb), pos
                assert torch.equal(c_ref.kv_state, c_st.kv_state) and torch.equal(c_ref.score_state, c_st.score_state)


# ------------------------------------------------------------------------------------------------ #9-#11 indexer tail
def _ref_topk(score_R, R_, k_cfg, offset):
    """model.py:579-581 at decode (width = R_, compress_lens = the int R_)."""
    k = min(k_cfg, R_)
    idxs = score_R.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < R_, idxs + offset, -1).int()


def _pad(score_R, W, g):
    """Full-width score: reachable prefix, then garbage (incl. NaN/inf) that masking must neutralize."""
    R_ = score_R.size(-1)
    junk = torch.randn(1, 1, W - R_, generator=g).to(score_R.dtype) * 100
    if W - R_ >= 3:
        junk[..., 0], junk[..., 1], junk[..., 2] = float("nan"), float("inf"), -float("inf")
    return torch.cat([score_R, junk], dim=-1)


def _static_tail(score_W, R_, k_cfg, offset):
    W = score_W.size(-1)
    s = score_W.clone()
    clen = torch.tensor([R_])
    s.masked_fill_(torch.arange(W) >= clen, -torch.inf)  # design #10
    return s, clen, indexer_topk_static(s, clen, min(k_cfg, W), offset)


def _boundary_tie(score_R, k):
    v = score_R.reshape(-1).float().sort(descending=True).values
    return v.numel() > k and bool(v[k - 1] == v[k])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_indexer_topk_padded_equals_dynamic(R, dtype):
    g = torch.Generator().manual_seed(1)
    k_cfg, offset, n_tie_cases = 16, 128, 0
    for W in (16, 40, 64, 200):
        for R_ in range(1, W + 1):
            for kind in ("cont", "ties", "allequal"):
                if kind == "cont":
                    s = torch.randn(1, 1, R_, generator=g)
                elif kind == "ties":
                    s = torch.randint(0, 4, (1, 1, R_), generator=g).float()
                else:
                    s = torch.zeros(1, 1, R_)
                s = s.to(dtype)
                ref = _ref_topk(s, R_, k_cfg, offset)
                _, _, got = _static_tail(_pad(s, W, g), R_, k_cfg, offset)
                n = ref.size(-1)
                assert got.size(-1) == min(k_cfg, W) >= n
                assert torch.equal(got[..., n:], torch.full_like(got[..., n:], -1)), (W, R_, kind)
                if R_ <= k_cfg or not _boundary_tie(s, k_cfg):
                    # the design's identity argument: every reachable position kept, or a unique top-k set
                    assert torch.equal(got[..., :n], ref), (W, R_, kind)
                else:
                    # boundary tie (risk 6): only "both are valid top-k sets" can be claimed
                    n_tie_cases += 1
                    sel_ref = s.reshape(-1)[(ref.reshape(-1) - offset).long()].float().sort().values
                    sel_got = s.reshape(-1)[(got.reshape(-1)[:n] - offset).long()].float().sort().values
                    assert torch.equal(sel_ref, sel_got), (W, R_, kind)
    assert n_tie_cases > 0  # the tie branch really was exercised


def test_candidate_blocks_padded_equals_dynamic(R):
    g = torch.Generator().manual_seed(2)
    for bs, tb in ((8, 4), (8, 64), (4, 3)):
        for W in (32, 64, 100):
            for R_ in range(1, W + 1):
                s = torch.randn(1, 1, R_, generator=g)  # continuous: no block-score ties
                ref = R.select_candidate_blocks(s, R_, tb, bs)
                s_w, clen, _ = _static_tail(_pad(s, W, g), R_, 1, 0)
                got = select_candidate_blocks_static(s_w, clen, tb, bs)
                assert got.shape == s_w.shape and got.dtype == torch.bool
                assert torch.equal(got[..., :R_], ref), (bs, tb, W, R_)
                # beyond reach only the newest (pinned) block's tail may be True; those scores are -inf already
                tail_true = got[..., R_:].nonzero()[:, -1] + R_
                assert all(int(i) // bs == (R_ - 1) // bs for i in tail_true), (bs, tb, W, R_)
    # the candidate-block helper with an int compress_lens is the reference function itself
    s = torch.randn(1, 1, 37, generator=g)
    assert torch.equal(select_candidate_blocks_static(s, 37, 3, 8), R.select_candidate_blocks(s, 37, 3, 8))


def test_indexer_tail_with_candidate_source_and_consumer(R):
    """Layer-20/24-style: the source publishes candidates, the consumer masks its own scores with them, both topk."""
    g = torch.Generator().manual_seed(3)
    bs, tb, k_cfg, off = 8, 64, 512, 128
    for W in (64, 300, 1024):
        for R_ in (1, 2, 7, 8, 9, 63, 64, min(W, 65), min(W, 299), W):
            src, con = (torch.randn(1, 1, R_, generator=g).to(torch.bfloat16) for _ in range(2))
            cand = R.select_candidate_blocks(src, R_, tb, bs)
            ref_src = _ref_topk(src, R_, k_cfg, off)
            ref_con = _ref_topk(con.masked_fill(~cand, -torch.inf), R_, k_cfg, off)
            s_w, clen, got_src = _static_tail(_pad(src, W, g), R_, k_cfg, off)
            cand_s = select_candidate_blocks_static(s_w, clen, tb, bs)
            c_w, _, _ = _static_tail(_pad(con, W, g), R_, k_cfg, off)
            got_con = indexer_topk_static(c_w.masked_fill(~cand_s, -torch.inf), clen, min(k_cfg, W), off)
            for ref, got in ((ref_src, got_src), (ref_con, got_con)):
                n = ref.size(-1)
                assert torch.equal(got[..., :n], ref) and bool((got[..., n:] == -1).all()), (W, R_)


# ------------------------------------------------------------------------------------------------ #2 engram hash
def test_engram_hash_static_matches_reference(R, monkeypatch):
    import engram

    V, CV = 300, 37
    monkeypatch.setattr(engram, "build_compressed_token_map", lambda tok: ([i % CV for i in range(V)], CV))
    args = R.ModelArgs(vocab_size=V, engram_layer_ids=(0, 1), engram_num_embeddings=(1000, 1000),
                       engram_max_ngram_size=4, engram_vocab_size=100, engram_n_heads=2, engram_head_dim=32,
                       engram_pad_id=2, engram_compressed_vocab_size=CV, max_seq_len=64, max_batch_size=1)
    layout = engram.EngramLayout.from_args(args)
    st_ref = engram.NgramHashState(args, layout, tokenizer=object())
    st_ref.cache.zero_()
    st_st = copy.deepcopy(st_ref)
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, V, (1, 40), generator=g)
    with torch.inference_mode():
        for L in (1, 3, 9):
            assert torch.equal(st_ref(ids[:, :L], 0), st_st(ids[:, :L], 0))
            for pos in range(L, L + 20):
                a = st_ref(ids[:, pos:pos + 1], pos)
                b = engram_hash_static(st_st, ids[:, pos:pos + 1], torch.tensor([pos]))
                assert a.dtype == b.dtype and torch.equal(a, b), (L, pos)
                assert torch.equal(st_ref.cache[:, :pos + 1], st_st.cache[:, :pos + 1])


# ------------------------------------------------------------------------------------------------ Generator modes
class FakeStatic:
    """Stand-in for StaticStep over tiny_tok.FakeModel: same protocol (step/capture/captured/tok/detach)."""

    def __init__(self, model, perturb_graph_at=None):
        self.model, self.perturb_graph_at = model, perturb_graph_at
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures, self.calls = {}, [], []

    @property
    def captured(self):
        return bool(self.graphs)

    def capture(self, fills=False):
        self.captures.append(bool(fills))
        self.graphs = {0: "g_even", 1: "g_odd"}
        return {"fills": fills}

    def step(self, token, pos, graph):
        t = int(token.reshape(-1)[0]) if isinstance(token, torch.Tensor) else int(token)
        self.calls.append((t, int(pos), bool(graph)))
        ids, logits, _ = self.model(torch.tensor([[t]], dtype=torch.long), pos)
        if graph and self.perturb_graph_at is not None and pos == self.perturb_graph_at:
            logits = logits.clone()
            logits[0, 5] += 1e-3
            ids = logits.argmax(dim=-1)
        self.tok.copy_(ids.view(1, 1))
        return logits

    def detach(self):
        self.graphs = {}

    def cold_counts(self):
        return None


def _gen(static_factory=None, model=None, mode="eager"):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    m = model or FakeModel()
    return Generator(m, tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=static_factory or FakeStatic), m


def test_generator_modes_same_records_and_capture_before_prefill():
    from runtime.generate import dump_record

    gen, m = _gen()
    recs, metas = {}, {}
    for mode in ("eager", "static", "graph"):
        gen.set_mode(mode)
        m.calls.clear()
        recs[mode] = gen.greedy([5, 9, 12, 3], 7, tag="t")
        metas[mode] = dict(gen.last_meta)
        assert metas[mode]["mode"] == mode and len(metas[mode]["step_sha256"]) == 7
        assert m.calls[0] == ((1, 4), 0)  # prefill is the reference call in every mode
    assert dump_record(recs["eager"]) == dump_record(recs["static"]) == dump_record(recs["graph"])
    assert metas["eager"]["logits_sha256"] == metas["static"]["logits_sha256"] == metas["graph"]["logits_sha256"]
    assert metas["eager"]["step_sha256"] == metas["graph"]["step_sha256"]
    st = gen.static
    assert st.captures == [False]  # captured once, on graph mode's first prefill, fills off
    assert [c[2] for c in st.calls] == [False] * 6 + [True] * 6 and [c[1] for c in st.calls[:6]] == [4, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError):
        gen.set_mode("cuda-graph")
    gen.reset_static()
    assert gen.static is None and st.graphs == {}


def test_generator_static_max_pos_guard():
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    gen = Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode="static", max_pos=10,
                    static_factory=FakeStatic)
    assert gen.max_pos == 10
    gen.greedy([5, 9, 12, 3], 8)  # last decode position 4 + 8 - 2 = 10
    with pytest.raises(ValueError):
        gen.greedy([5, 9, 12, 3], 9)
    gen.set_mode("eager")
    assert gen.max_pos == gen.max_seq_len - 1


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()][:n]


def test_parity_levels_pass_on_fake():
    fl = _first_light()
    gen, _ = _gen(mode="graph")
    out = []
    det = torch.utils.deterministic
    fill0 = det.fill_uninitialized_memory
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    assert res["verdict"] == "PASS" and res["n_prompts"] == 3 and res["steps"] == 4
    assert [r["what"] for r in res["rows"][:4]] == ["static-eager vs reference-eager", "graph vs static-eager",
                                                    "graph replay#2 vs replay#1", "graph relaunch vs replay#1"]
    assert len(res["rows"]) == 4 + 4 + 3 and all(r["equal"] == 3 for r in res["rows"])
    assert any(l.startswith("PARITY LEVELS (4 steps x 3 prompts") and l.endswith("PASS") for l in out)
    assert det.fill_uninitialized_memory == fill0 and gen.mode == "graph" and gen.static.captures[-1] is False
    assert all(len(v["steps"]) == 5 for run in res["runs"].values() for v in run.values())


def test_parity_levels_reports_first_divergence():
    fl = _first_light()
    rows = _rows()
    gen, _ = _gen(static_factory=lambda m: FakeStatic(m, perturb_graph_at=None))
    L = len(gen.encode([{"role": "user", "content": rows[0]["prompt"]}]))
    gen.static_factory = lambda m: FakeStatic(m, perturb_graph_at=L + 2)  # 3rd decode step, graph mode only
    res = fl.parity_levels(gen, rows[:1], steps=4, log=lambda *a, **k: None)
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    assert res["verdict"] == "FAIL"
    assert by[("off", "static-eager vs reference-eager")]["all_equal"]
    l1 = by[("off", "graph vs static-eager")]
    assert not l1["all_equal"] and l1["first_diff"] == f"{rows[0]['tag']}@3"  # position 0 = prefill logits
    assert by[("off", "graph replay#2 vs replay#1")]["all_equal"]  # deterministic, just wrong


def test_first_light_parser_t8_flags_and_measure_runs_levels(tmp_path, capsys):
    fl = _first_light()
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--parity-levels", "--level-steps", "3",
                                      "--max-pos", "511", "--marker", "--fixture", FIXTURE, "--only", "prose,json",
                                      "--reps", "1", "--max-new", "6", "--step-cell", "8", "--stats-from", "2",
                                      "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    assert (a.mode, a.parity_levels, a.level_steps, a.max_pos, a.marker) == ("graph", True, 3, 511, True)
    assert fl.build_parser().parse_args(["--model-dir", "x"]).mode == "eager"
    gen, _ = _gen(mode="graph")
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["parity_levels"]["verdict"] == "PASS" and report["parity_levels"]["n_prompts"] == 2
    assert report["step_cell"]["n_steps"] == 8
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | mode graph levels PASS")
