"""Mac CPU tests of T9b-3a (`.venv/bin/python -m pytest -q tests/test_attn_bound_cpu.py`).

  1. The valid-block count: `valid_blocks` / the device `nblk_static` == ceil(width / 64) of the width the eager
     reference passes, 128 + min(512, T/r) (model.py:579, 778-779), for every position 1..4,096 at both compressor
     ratios; block-count rounding at the edges (128 / 278 / 428 / 640 -> 2 / 5 / 7 / 10).
  2. On the static step's real index lists (window_topk_idxs_dev + indexer_topk_static over fixed-width scores with the
     -inf masking, heavy ties): every slot past nblk * 64 is -1 and nblk == the last valid block exactly; with a
     candidate mask that removes reachable positions the count is an upper bound (still exact-safe).
  3. The kernel's algorithm (kernel.py:350-387 in torch, as tests/test_static_emulated_cpu.py): bounded == full padded
     width == narrow width, bitwise, single and multi-row -- the "-1 blocks are exact no-ops" claim, in fp32 torch.
  4. self_check picks the first variant; a broken bounded impl (drops the last valid block) is rejected.
  5. runtime/patches.py `sparse_attn_bounded`: routing only inside `attn_bound_scope`, reasons, restore, re-install.
  6. StaticStep(attn_impl="bounded") on the emulated mini checkpoints (mini, mini4, and mini4 with index_topk 256 so
     blocks are really skipped): logits == attn_impl "ref" == the reference forward at every position; the bounded
     call sees exactly the compressed layers with nblk == the host formula; a failing self-check leaves the step on
     the reference; the 16-B gather alignment check at build.
  7. Generator / first_light plumbing: --attn-impl, the level-A row, the report and the summary line.

The TileLang kernel and the vector gather themselves only run on the box (tests/gpu/test_attn_bound_gpu.py).
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch

from runtime.kernels import sparse_attn_bounded as SAB
from runtime.static_step import (indexer_topk_static, select_candidate_blocks_static, window_topk_idxs_dev)
from test_static_emulated_cpu import KERNELS, _ref_run, _static_run, emu  # noqa: F401  (emu is a fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
WIN, K = 128, 512


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ 1. count math
@pytest.mark.parametrize("ratio", [1, 2])
def test_valid_count_equals_reference_width_blocks_pos_1_to_4096(ratio):
    pos = torch.arange(1, 4097)
    clen = torch.div(pos + 1, ratio, rounding_mode="floor")
    dev = SAB.nblk_static(clen, K, WIN)
    assert dev.dtype == torch.int32 and tuple(dev.shape) == (1, 4096)
    for p in range(1, 4097):
        w = SAB.ref_width(p, ratio, K, WIN)
        assert w == WIN + min(K, (p + 1) // ratio)
        want = -(-w // 64)
        assert SAB.valid_blocks(p, ratio, K, WIN) == want == int(dev[0, p - 1]), (ratio, p)
    # the saving vanishes at long context: T/r >= 512 -> the full 10 blocks of the static width (640)
    assert SAB.valid_blocks(512 * ratio - 1, ratio, K, WIN) == 10 == SAB.blocks_for_width(WIN + K)
    assert SAB.valid_blocks(4095, ratio, K, WIN) == 10
    assert SAB.valid_blocks(100, 0, K, WIN) == 2 and SAB.ref_width(100, 0) == 128


def test_block_rounding():
    cases = {0: 0, 1: 1, 63: 1, 64: 1, 65: 2, 127: 2, 128: 2, 129: 3, 192: 3, 193: 4, 278: 5, 428: 7, 640: 10}
    for w, nb in cases.items():
        assert SAB.blocks_for_width(w) == nb, w
    # the fixture regime the memo names (2-5 blocks at short context) vs the static step's 10
    assert [SAB.valid_blocks(p, 2, K) for p in (1, 127, 128, 300, 1022, 1023)] == [3, 3, 3, 5, 10, 10]
    assert [SAB.valid_blocks(p, 1, K) for p in (1, 63, 64, 300, 511)] == [3, 3, 4, 7, 10]


# ------------------------------------------------------------------------------------------------ 2. real lists
def _static_list(pos, ratio, W, k, g, ties=False, cand=None):
    p = torch.tensor([pos])
    win_idx = window_topk_idxs_dev(p, WIN, torch.arange(WIN))
    score = torch.randn(1, 1, W, generator=g).to(torch.bfloat16)
    if ties:
        score = (score * 2).round() / 2
    clen = torch.div(p + 1, ratio, rounding_mode="floor")
    score.masked_fill_(torch.arange(W) >= clen, -torch.inf)
    if cand is not None:
        score = score.masked_fill(~cand(score, clen), -torch.inf)
    comp = indexer_topk_static(score, clen, k, WIN)
    return torch.cat([win_idx, comp], dim=-1), clen


@pytest.mark.parametrize("ratio,W", [(2, 2048), (1, 4096)])
def test_nblk_is_last_valid_block_of_static_index_list(ratio, W):
    g = torch.Generator().manual_seed(ratio)
    k = min(K, W)
    for pos in list(range(1, 1100)) + list(range(1100, 4096, 37)):
        idx, clen = _static_list(pos, ratio, W, k, g, ties=pos % 3 == 0)
        nb = SAB.nblk_static(clen, k, WIN)
        assert torch.equal(SAB.nblk_from_idx(idx), nb.view(1, 1)), (ratio, pos)
        assert bool((idx[..., int(nb) * 64:] == -1).all())
        assert int(nb) == SAB.valid_blocks(pos, ratio, k, WIN)


def test_nblk_upper_bound_under_candidate_mask():
    """A candidate source that masks reachable blocks to -inf: the selected set then holds -inf picks whose identity
    is topk's tie order, but whatever it picks, the valid entries (idx < clen) are a sorted prefix of length
    <= min(k, clen), so the arithmetic count is >= the true last valid block and every skipped slot is -1."""
    g = torch.Generator().manual_seed(7)
    W, k, ratio = 1024, 256, 1
    cand = lambda s, c: select_candidate_blocks_static(s, c, 4, 16)  # noqa: E731  (4 x 16 = 64 positions kept)
    for pos in range(70, 1000, 13):
        idx, clen = _static_list(pos, ratio, W, k, g, cand=cand)
        nb = SAB.nblk_static(clen, k, WIN)
        true = SAB.nblk_from_idx(idx)
        assert int(true) <= int(nb) and bool((idx[..., int(nb) * 64:] == -1).all()), pos


# ------------------------------------------------------------------------------------------------ 3. algorithm
def emu_bounded(q, kv, attn_sink, topk_idxs, softmax_scale, nblk=None, variant=None, strict=False, log=None):
    """kernel.py:350-387 (as test_static_emulated_cpu.sparse_attn) with the block loop bounded by nblk[b, m]."""
    b, m, h, d = q.shape
    topk, blk = topk_idxs.size(-1), 64
    full = (topk + blk - 1) // blk
    if nblk is not None:
        assert nblk.dtype == torch.int32 and tuple(nblk.shape) == (b, m)
        if strict:  # the count never cuts a valid index
            for bi in range(b):
                for mi in range(m):
                    assert bool((topk_idxs[bi, mi, int(nblk[bi, mi]) * blk:] == -1).all())
        if log is not None:
            log.append((nblk.tolist(), topk))
    o = torch.empty_like(q)
    for bi in range(b):
        for mi in range(m):
            nb = full if nblk is None else min(int(nblk[bi, mi]), full)
            qv = q[bi, mi].float()
            mx = torch.full((h,), -1e30)
            se = torch.zeros(h)
            acc = torch.zeros(h, d)
            for t in range(nb):
                ids = torch.full((blk,), -1, dtype=torch.long)
                part = topk_idxs[bi, mi, t * blk:(t + 1) * blk].long()
                ids[:part.numel()] = part
                valid = ids != -1
                kvb = torch.where(valid.unsqueeze(-1), kv[bi, ids.clamp_min(0)], torch.zeros((), dtype=kv.dtype))
                s = torch.where(valid, qv @ kvb.float().t(), torch.tensor(-torch.inf)) * softmax_scale
                prev = mx
                mx = torch.maximum(prev, s.amax(-1))
                sc = torch.exp(prev - mx)
                p = torch.exp(s - mx.unsqueeze(-1))
                se = se * sc + p.sum(-1)
                acc = acc * sc.unsqueeze(-1) + p.to(torch.bfloat16).float() @ kvb.float()
            se = se + torch.exp(attn_sink.float() - mx)
            o[bi, mi] = (acc / se.unsqueeze(-1)).to(q.dtype)
    return o


def test_emulated_kernel_padded_blocks_are_exact_noops():
    ref = KERNELS["sparse_attn"]
    g = torch.Generator().manual_seed(3)
    h, d, W = 16, 64, 1024
    kv = torch.randn(1, WIN + W, d, generator=g).to(torch.bfloat16)
    sink = torch.randn(h, generator=g)
    rows = SAB.check_cases(WIN, K, WIN + W, seed=5)
    assert {r["c"] for r in rows} >= {0, 1, 64, 150, 300, 512}
    for r in rows:
        q = torch.randn(1, 1, h, d, generator=g).to(torch.bfloat16)
        idx = r["idx"].view(1, 1, -1)
        nb = torch.tensor([[SAB.blocks_for_width(WIN + r["c"])]], dtype=torch.int32)
        assert torch.equal(SAB.nblk_from_idx(idx), nb) or r["pos"] < 63 and r["c"] == 0
        got = emu_bounded(q, kv, sink, idx, 0.125, nb, strict=True)
        assert torch.equal(got, ref(q, kv, sink, idx, 0.125)), (r["pos"], r["c"])
        assert torch.equal(got, ref(q, kv, sink, idx[..., :WIN + r["c"]].contiguous(), 0.125)), (r["pos"], r["c"])
    pick = [rows[i] for i in (2, 9, 17, 30)]
    idx = torch.stack([p["idx"] for p in pick]).view(1, 4, -1)
    nb = torch.tensor([[SAB.blocks_for_width(WIN + p["c"]) for p in pick]], dtype=torch.int32)
    q = torch.randn(1, 4, h, d, generator=g).to(torch.bfloat16)
    assert torch.equal(emu_bounded(q, kv, sink, idx, 0.125, nb), ref(q, kv, sink, idx, 0.125))


# ------------------------------------------------------------------------------------------------ 4. self-check
def test_self_check_selects_and_rejects():
    ref = KERNELS["sparse_attn"]
    chk = SAB.self_check(ref, "cpu", 4, 32, 0.2, WIN, 64, WIN + 256, impl=emu_bounded, select=False)
    assert chk["variant"] == SAB.VARIANTS[0] and chk["results"] == {SAB.VARIANTS[0]: "equal"}, chk

    def short(q, kv, sink, idx, scale, nblk, variant=None):  # drops the last valid block
        return emu_bounded(q, kv, sink, idx, scale, (nblk - 1).clamp_min(0))

    bad = SAB.self_check(ref, "cpu", 4, 32, 0.2, WIN, 64, WIN + 256, impl=short, select=False)
    assert bad["variant"] is None and all(v.startswith("mismatch") for v in bad["results"].values()), bad
    assert set(bad["results"]) == set(SAB.VARIANTS)

    def boom(*a, variant=None, **k):
        if variant != "guard":
            raise RuntimeError("tilelang: loop extent must be a constant\nmore")
        return emu_bounded(*a, **k)

    pick = SAB.self_check(ref, "cpu", 4, 32, 0.2, WIN, 64, WIN + 256, impl=boom, select=False)
    assert pick["variant"] == "guard" and pick["results"]["extent"].startswith("error: RuntimeError: tilelang")


# ------------------------------------------------------------------------------------------------ 5. the patch
def test_patch_routes_only_inside_scope(monkeypatch):
    from runtime import patches
    from runtime.refmodel import import_reference

    R = import_reference(cpu_stubs=True)
    calls = []
    orig = lambda *a: calls.append("orig") or "o"  # noqa: E731
    monkeypatch.setattr(R, "sparse_attn", orig)
    monkeypatch.setattr(SAB, "sparse_attn_bounded", lambda *a: calls.append(("bounded", a[-1])) or "b")
    patches.attn_bound_stats(reset=True)
    patches.sparse_attn_bounded(True)
    try:
        assert patches.sparse_attn_bounded_enabled() and patches.original_sparse_attn() is orig
        q = torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)
        idx = torch.full((1, 1, 192), -1, dtype=torch.int32)
        nb = torch.tensor([[3]], dtype=torch.int32)
        assert R.sparse_attn(q, None, None, idx, 0.1) == "o"                     # no scope
        with patches.attn_bound_scope(nb):
            assert R.sparse_attn(q, None, None, idx, 0.1) == "b"
            assert R.sparse_attn(q, None, None, idx.long(), 0.1) == "o"          # int64 indices: fall through
            with patches.attn_bound_scope(nb.view(1, 1, 1)):
                assert R.sparse_attn(q, None, None, idx, 0.1) == "o"             # wrong nblk shape
        with patches.attn_bound_scope(None):
            assert R.sparse_attn(q, None, None, idx, 0.1) == "o"
        assert calls[1][0] == "bounded" and calls[1][1] is nb
        st = patches.attn_bound_stats()
        assert st["routed"] == 1 and st["fallthrough"] == {"idx": 1, "nblk": 1}
        patches.sparse_attn_bounded(True)                                       # idempotent
        assert patches.original_sparse_attn() is orig
    finally:
        patches.sparse_attn_bounded(False)
    assert R.sparse_attn is orig and not patches.sparse_attn_bounded_enabled()
    other = lambda *a: "x"  # noqa: E731
    monkeypatch.setattr(R, "sparse_attn", other)                                 # a later emulation
    patches.sparse_attn_bounded(True)
    assert patches.original_sparse_attn() is other
    patches.sparse_attn_bounded(False)


# ------------------------------------------------------------------------------------------------ 6. StaticStep
def _variant(emu_build, monkeypatch, **over):
    """mini4 with config overrides. mini4w: index_topk 256 -- at max_pos 255 the static widths are 128 + 128 (r=2) /
    128 + 256 (r=1) = 4 / 6 blocks while the fixture positions need 3, so the bound really skips blocks. mini4r:
    index sources [1, 2] only, so layer 3 (ratio 1) reuses layer 2's published indices and block count."""
    import test_static_emulated_cpu as E

    cfg, mp = E._mini4()
    cfg.update(over)
    monkeypatch.setattr(E, "_mini4", lambda: (dict(cfg), mp))
    return emu_build("mini4")


VARIANT_CFG = {"mini4w": {"index_topk": 256}, "mini4r": {"index_topk": 256, "index_source_layers": [1, 2]}}


@pytest.mark.parametrize("kind", ["mini", "mini4", "mini4w", "mini4r"])
def test_bounded_static_step_equals_ref_and_reference(emu, monkeypatch, tmp_path, kind):
    from runtime import patches
    from runtime.static_step import StaticStep

    model, pool, fk = _variant(emu, monkeypatch, **VARIANT_CFG[kind]) if kind in VARIANT_CFG else emu(kind)
    log = []
    monkeypatch.setattr(SAB, "sparse_attn_bounded",
                        lambda *a, variant=None: emu_bounded(*a, variant=variant, strict=True, log=log))
    max_pos = 255 if kind in VARIANT_CFG else 127
    st = StaticStep(model, pool, max_pos=max_pos, attn_impl="bounded", _allow_cpu=True)
    d = st.attn_describe()
    assert d["on"] and d["variant"] == SAB.VARIANTS[0] and d["check"]["results"] == {SAB.VARIANTS[0]: "equal"}, d
    json.dumps(d)
    assert patches.sparse_attn_bounded_enabled()
    attns = [l.attn for l in model.layers]
    comp = [a for a in attns if a.compress_ratio]
    src, srcs = None, []                                                # the index source each compressed layer reads
    for a in comp:
        src = a if a.is_index_source else src
        srcs.append(src)
    if kind == "mini4r":
        assert not comp[-1].is_index_source and srcs[-1] is comp[-2]
    g = torch.Generator().manual_seed(11)
    runs = []
    skipped = 0
    for L in (2, 3, 16, 17):
        ids = torch.randint(2, 2000, (L,), generator=g).tolist()
        log.clear()
        got = _static_run(model, st, ids, 8)
        assert len(log) == 8 * len(comp), (len(log), len(comp))
        for n, (nb, width) in enumerate(log):                          # layer order, one call per compressed layer
            pos = L + n // len(comp)
            s_ = srcs[n % len(comp)]
            ix_ratio = s_.indexer.compress_ratio
            k = min(s_.indexer.index_topk, st.W[ix_ratio])
            assert width == WIN + k and nb == [[SAB.valid_blocks(pos, ix_ratio, k, WIN)]], (n, nb)
            skipped += SAB.blocks_for_width(width) - nb[0][0]
        runs.append((ids, got))
    st.detach()
    if kind in VARIANT_CFG:
        assert skipped > 0
    ref_st = StaticStep(model, pool, max_pos=max_pos, _allow_cpu=True)
    assert ref_st.attn_describe() == {"impl": "ref", "on": False}
    for ids, got in runs:
        log.clear()
        want = _static_run(model, ref_st, ids, 8)
        assert not log                                                  # ref step never routes
        bad = [i for i, (a, b) in enumerate(zip(got[0], want[0])) if not torch.equal(a, b)]
        assert not bad and got[1] == want[1], (kind, bad)
        ref = _ref_run(model, ids, 8)
        bad = [i for i, (a, b) in enumerate(zip(got[0], ref[0])) if not torch.equal(a, b)]
        assert not bad and got[1] == ref[1], (kind, "vs reference forward", bad)
    ref_st.detach()


def test_failing_self_check_keeps_reference(emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")
    log = []
    monkeypatch.setattr(SAB, "sparse_attn_bounded", lambda q, kv, s, i, sc, nb, variant=None: emu_bounded(
        q, kv, s, i, sc, (nb - 1).clamp_min(0), log=log))
    st = StaticStep(model, pool, max_pos=127, attn_impl="bounded", _allow_cpu=True)
    d = st.attn_describe()
    assert not d["on"] and d["variant"] is None and "self-check" in d["reason"], d
    log.clear()
    ids = list(range(30, 47))
    got = _static_run(model, st, ids, 5)
    assert not log
    ref = _ref_run(model, ids, 5)
    assert all(torch.equal(a, b) for a, b in zip(got[0], ref[0])) and got[1] == ref[1]
    st.detach()
    with pytest.raises(ValueError):
        StaticStep(model, pool, max_pos=127, attn_impl="fast", _allow_cpu=True)


def test_gather_alignment_checked_at_build(emu, monkeypatch):
    from runtime import static_step as SS

    model, pool, _ = emu("mini")
    monkeypatch.setattr(SS, "GATHER_ALIGN", 1 << 62)
    with pytest.raises(ValueError, match="16-B vector gather"):
        SS.StaticStep(model, pool, max_pos=127, _allow_cpu=True)
    monkeypatch.setenv(SS.GATHER_IMPL_ENV, "u8")
    SS.StaticStep(model, pool, max_pos=127, _allow_cpu=True).detach()


# ------------------------------------------------------------------------------------------------ 7. plumbing
class FakeStaticAttn:
    """Bit-identical attn impls change nothing (so the A row is all-equal); records how it was built."""

    built = []

    def __init__(self, model, moe_impl="ref", fp8_impl="ref", glue_impl="ref", attn_impl="ref"):
        self.model, self.attn_impl = model, attn_impl
        self.tok = torch.zeros(1, 1, dtype=torch.long)
        self.graphs, self.captures = {}, []
        self.glue = None
        FakeStaticAttn.built.append((moe_impl, fp8_impl, glue_impl, attn_impl))

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
        return logits

    def attn_describe(self):
        return {"impl": self.attn_impl, "on": self.attn_impl == "bounded", "variant": "extent"}

    def detach(self):
        self.graphs = {}

    def cold_counts(self):
        return None


def _gen(mode="eager", **kw):
    from runtime.generate import Generator
    from tiny_tok import FakeModel, make_wordlevel_tokenizer

    FakeStaticAttn.built = []
    return Generator(FakeModel(), tokenizer=make_wordlevel_tokenizer(64), eos_ids=[1], mode=mode,
                     static_factory=FakeStaticAttn, **kw)


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=3):
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()][:n]


def test_generator_attn_impl_plumbing():
    gen = _gen(mode="static", attn_impl="bounded")
    gen.greedy([5, 9, 12, 3], 4)
    assert gen.static.attn_impl == "bounded" and gen.last_meta["attn_impl"] == "bounded"
    old = gen.static
    gen.set_attn_impl("bounded")
    assert gen.static is old
    gen.set_attn_impl("ref")
    assert gen.static is None
    gen.greedy([5, 9, 12, 3], 4)
    assert FakeStaticAttn.built == [("ref", "ref", "ref", "bounded"), ("ref", "ref", "ref", "ref")]
    with pytest.raises(ValueError):
        gen.set_attn_impl("flashmla")


def test_parity_levels_attn_row():
    fl = _first_light()
    gen = _gen(mode="graph", glue_impl="fused", attn_impl="bounded")
    out = []
    res = fl.parity_levels(gen, _rows(), steps=4, log=lambda *a, **k: out.append(" ".join(map(str, a))))
    by = {(r["fills"], r["what"]): r for r in res["rows"]}
    row = by[("off", "bounded attn vs ref attn (same moe/fp8/glue)")]
    assert row["gate"] and row["all_equal"] and row["level"] == "A" and res["attn_impl"] == "bounded"
    assert by[("off", "static-eager vs reference-eager")]["gate"]       # not a variant: L0 stays gated
    assert res["verdict"] == "PASS"
    assert ("ref", "ref", "fused", "ref") in FakeStaticAttn.built      # the A reference: same impls, attn ref
    assert ("ref", "ref", "ref", "bounded") in FakeStaticAttn.built    # the G reference keeps attn bounded
    assert (gen.attn_impl, gen.glue_impl, gen.mode) == ("bounded", "fused", "graph")
    assert gen.static.attn_impl == "bounded" and gen.static.captures[-1] is False
    assert any("attn impl bounded" in l for l in out)


def test_first_light_attn_flag_report_and_summary(tmp_path, capsys):
    fl = _first_light()
    assert fl.build_parser().parse_args(["--model-dir", "x"]).attn_impl == "ref"
    with pytest.raises(SystemExit):
        fl.build_parser().parse_args(["--model-dir", "x", "--attn-impl", "flashmla"])
    a = fl.build_parser().parse_args(["--model-dir", "x", "--mode", "graph", "--attn-impl", "bounded",
                                      "--parity-levels", "--level-steps", "2", "--fixture", FIXTURE, "--only",
                                      "prose", "--reps", "1", "--max-new", "4", "--step-cell", "6", "--stats-from",
                                      "2", "--out", str(tmp_path / "c.jsonl"), "--json", str(tmp_path / "r.json")])
    gen = _gen(mode="graph", attn_impl=a.attn_impl)
    report = {"date": "d", "env": {"device": None}, "errors": [], "generator": {"mode": "graph"}}
    fl.measure(gen, a, report, n_layers=2, out=a.out, json_path=a.json, fixture2=None)
    assert report["generator"]["attn_impl"] == "bounded" and report["step_cell"]["attn_impl"] == "bounded"
    assert report["parity_levels"]["attn_impl"] == "bounded" and report["parity_levels"]["verdict"] == "PASS"
    assert report["attn_bound"]["on"] and not report["errors"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-12].startswith("FIRST LIGHT d | mode graph attn bounded levels PASS")
    report["attn_bound"]["on"] = False
    assert fl.summary_lines(report, "x")[0].startswith("FIRST LIGHT d | mode graph attn bounded(OFF) levels PASS")
