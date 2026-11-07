"""Mac tests of M1-T11 (`.venv/bin/python -m pytest -q tests/test_draft_cpu.py`): runtime/draft_step.py,
runtime/mtp_loader.py, bench/draft_bench.py's acceptance logic.

  * positional / predication math: context positions, validity, the bonus index, ring slots; the device window list
    == the reference's get_dspark_topk_idxs (model.py:1022-1031) followed by -1 padding, and the kernel's own
    algorithm (the emulated sparse_attn: 64-wide blocks, entries past topk = -1) gives the SAME BITS on both lists;
  * the Markov chain on a toy head: sequential argmax (draft i+1 depends on draft i), temperature 0 = argmax;
  * the mtp loader on mini checkpoints: every mtp.* tensor placed (verify sha256), expert stacks bound as views,
    temperature 0, shared embed/head, refusal on a missing / unexpected key;
  * the V2 gate of tests/gpu/test_draft_gpu.py executed on the CPU kernel emulations (DEV "cpu"; fp8/fp4 GEMMs
    evaluated row by row as the box census measured the real kernels): DraftStep static-eager == reference
    forward_spec for the first draft after a prefill (ctx 0 and 1), ctx 1/3/6 after a 6-row verify, chained drafts,
    a prompt longer than the 128-ring; drafts, logits, confidence and rings torch.equal;
  * predication: a draft with ctx_count c changes exactly the c ring slots of positions A'-c+1..A' in every stage.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys

import pytest
import torch

from mini_ckpt import MINI_CONFIG
from mini_mtp import make_mini_checkpoint_mtp, mtp_config_kind
from runtime.loader import LoadRefused, build_and_load
from runtime.refmodel import import_reference
from test_static_emulated_cpu import KERNELS
from test_static_rows_cpu import _rowwise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield


@pytest.fixture
def emu(monkeypatch, tmp_path):
    """-> build(kind, **ckpt_kw) -> model (mini checkpoint with the mtp schema, load_mtp=True, CPU emulations)."""
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

    def build(kind="mini3", **kw):
        ck = make_mini_checkpoint_mtp(str(tmp_path / kind), cfg=mtp_config_kind(kind), **kw)
        model = build_and_load(ck["model_dir"], ck["map"], "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                               load_mtp=True, verify=True, log=lambda *a: None)
        model.temperature = 0.0
        return model

    yield build
    torch.set_default_dtype(prev)
    R.MoE.forward = getattr(R.MoE, "_orig_forward", R.MoE.forward)


def _gpu_module(monkeypatch, emu, name="gpu_draft"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "tests", "gpu", "test_draft_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    @contextlib.contextmanager
    def model_cpu(kind="mini3"):
        yield emu(kind)

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STEP_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_tilelang", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    return G


# ------------------------------------------------------------------------------------------------ index math
@pytest.mark.parametrize("a", [0, 1, 2, 5, 62, 63, 64, 122, 126, 127, 128, 129, 200, 1000])
def test_window_idxs_equal_reference_list_then_minus_one(a):
    from runtime.draft_step import dspark_window_idxs_dev, reference_dspark_idxs

    win, blk = 128, 5
    got = dspark_window_idxs_dev(torch.tensor([a]), win, blk, torch.arange(win + blk))
    assert got.shape == (1, blk, win + blk) and got.dtype == torch.int32 and got.is_contiguous()
    ref = reference_dspark_idxs(win, blk, a)
    n = ref.size(-1)
    assert torch.equal(got[..., :n], ref)
    assert (got[..., n:] == -1).all()


@pytest.mark.parametrize("a", [3, 40, 58, 59, 60, 123, 127, 300])
def test_padded_list_gives_the_kernels_bits(a):
    """The kernel algorithm (kernel.py:350-387, emulated: 64-wide blocks, `t*64+i < topk` else -1, running max from
    -1e30, online rescale, bf16 P) on the reference's variable-length list and on our fixed-width padded list."""
    from runtime.draft_step import dspark_window_idxs_dev, reference_dspark_idxs

    g = torch.Generator().manual_seed(a)
    win, blk, h, d = 128, 5, 4, 64
    q = (torch.randn(1, blk, h, d, generator=g) * 3).to(torch.bfloat16)
    kv = (torch.randn(1, win + blk, d, generator=g) * 3).to(torch.bfloat16)
    sink = torch.randn(h, generator=g)
    sa = KERNELS["sparse_attn"]
    o_ref = sa(q, kv, sink, reference_dspark_idxs(win, blk, a), d ** -0.5)
    o_pad = sa(q, kv, sink, dspark_window_idxs_dev(torch.tensor([a]), win, blk, torch.arange(win + blk)), d ** -0.5)
    assert torch.equal(o_ref, o_pad)


@pytest.mark.parametrize("c", range(7))
def test_ctx_positions_validity_bonus(c):
    from runtime.draft_step import bonus_index, ctx_positions

    a = 300
    pos, valid = ctx_positions(torch.tensor([a]), torch.tensor([c]), torch.arange(6))
    assert pos.tolist() == [a - c + 1 + i for i in range(6)]
    assert valid.tolist() == [i < c for i in range(6)]
    assert [p for p, v in zip(pos.tolist(), valid.tolist()) if v] == list(range(a - c + 1, a + 1))
    assert int(bonus_index(torch.tensor([c]))) == max(c - 1, 0)
    slots = torch.remainder(pos, 128).tolist()
    assert len(set(slots)) == 6  # 6 consecutive positions: distinct slots -> the predicated write is deterministic


# ------------------------------------------------------------------------------------------------ Markov chain
def test_markov_chain_toy_head_is_sequential():
    """forward_head (model.py:1140-1159, what DraftStep calls) on a toy head: zero base logits, a Markov bias that
    sends token t to (t + 1) % V -> drafts b+1, b+2, ... (each from the PREVIOUS draft, 1153-1156). A parallel
    reading (every row biased by the bonus) would give b+1 five times."""
    R = import_reference(cpu_stubs=True)
    from torch import nn

    V, D, r, hc, blk = 16, 8, 16, 4, 5
    torch.manual_seed(0)
    mh = R.DSparkMarkovHead.__new__(R.DSparkMarkovHead)
    nn.Module.__init__(mh)
    emb = R.ParallelEmbedding.__new__(R.ParallelEmbedding)
    nn.Module.__init__(emb)
    emb.vocab_size, emb.dim, emb.part_vocab_size = V, r, V
    emb.vocab_start_idx, emb.vocab_end_idx = 0, V
    emb.weight = nn.Parameter(torch.eye(V, r, dtype=torch.bfloat16), requires_grad=False)
    head = R.ParallelHead.__new__(R.ParallelHead)
    nn.Module.__init__(head)
    head.vocab_size, head.dim, head.part_vocab_size = V, r, V
    w = torch.zeros(V, r)
    for t in range(V):
        w[(t + 1) % V, t] = 10.0
    head.weight = nn.Parameter(w, requires_grad=False)
    mh.embed, mh.head = emb, head

    class Toy(nn.Module):
        pass

    blk_mod = Toy()
    blk_mod.block_size = blk
    blk_mod.markov_head = mh
    blk_mod.norm = nn.Identity()
    base = R.ParallelHead.__new__(R.ParallelHead)
    nn.Module.__init__(base)
    base.vocab_size, base.dim, base.part_vocab_size = V, D, V
    base.weight = nn.Parameter(torch.zeros(V, D), requires_grad=False)
    blk_mod.head = base
    conf = R.DSparkConfidenceHead.__new__(R.DSparkConfidenceHead)
    nn.Module.__init__(conf)
    lin = R.Linear.__new__(R.Linear)
    nn.Module.__init__(lin)
    lin.weight = nn.Parameter(torch.randn(1, D + r), requires_grad=False)
    lin.register_parameter("scale", None)
    lin.register_parameter("bias", None)
    conf.proj = lin
    blk_mod.confidence_head = conf
    blk_mod.hc_pre = lambda x, pm: R.Block.hc_pre(None, x, pm)
    blk_mod.temperature = 0.0
    x = torch.randn(1, blk, hc, D).to(torch.bfloat16)
    pm = torch.zeros(1, blk, hc)
    pm[..., 0] = 1
    b = 3
    out, logits, c = R.DSparkBlock.forward_head(blk_mod, x, pm, torch.tensor([b]))
    assert out.tolist() == [[b, b + 1, b + 2, b + 3, b + 4, b + 5]]
    assert logits.shape == (1, blk, V) and c.shape == (1, blk)
    for i in range(blk):  # row i's bias came from output_ids[i]
        assert int(logits[0, i].argmax()) == (b + i + 1) % V and float(logits[0, i].max()) == 10.0


# ------------------------------------------------------------------------------------------------ mtp loader
@pytest.mark.parametrize("kind", ["mini", "mini3"])
def test_mtp_loader_places_everything(emu, kind):
    R = import_reference(cpu_stubs=True)
    model = emu(kind)
    cfg = mtp_config_kind(kind)
    mt = model.runtime_mtp
    assert len(model.mtp) == cfg["n_mtp_layers"] == mt.n_stages
    assert mt.verify["mismatch"] == [] and mt.verify["n"] == mt.n_tensors
    assert mt.target_layer_ids == tuple(cfg["dspark_target_layer_ids"]) and mt.block_size == 5
    assert model.target_layer_ids == () and len(model.layers) == cfg["n_layers"]  # backbone keeps the overlay
    E = cfg["dspark_n_routed_experts"]
    for i, st in enumerate(model.mtp):
        assert isinstance(st, R.DSparkBlock) and st.temperature == 0.0
        assert st.embed is model.embed and st.head is model.head
        assert len(st.ffn.experts) == E and st.ffn.n_activated_experts == cfg["dspark_n_activated_experts"]
        for e in range(E):
            for p in ("w1", "w2", "w3"):
                lin = getattr(st.ffn.experts[e], p)
                w, s = mt.stacks[i][p]
                assert lin.weight.data_ptr() == w[e].data_ptr() and lin.scale.data_ptr() == s[e].data_ptr()
                assert lin.weight.dtype == torch.float4_e2m1fn_x2 and lin.weight.scale is lin.scale
        assert st.attn.wo_a.weight.dtype == torch.bfloat16 and st.attn.compress_ratio == 0
    last = model.mtp[-1]
    assert last.markov_head.head.weight.dtype == torch.float32
    assert last.confidence_head.proj.weight.dtype == torch.float32
    assert sorted(mt.widened) == sorted([f"mtp.{len(model.mtp) - 1}.confidence_head.proj.weight",
                                         f"mtp.{len(model.mtp) - 1}.markov_head.head.weight"])
    assert len(mt.dequantized) == len(model.mtp)
    b = mt.bytes
    assert b["hbm_total"] == b["hbm_params"] + b["hbm_expert_stacks"] > 0
    assert model.runtime_load.bytes["mtp_HBM"] == b["hbm_total"]
    own = {id(p): p for n, p in model.mtp.named_parameters() if not n.split(".", 1)[1].startswith(("embed.", "head."))}
    assert b["hbm_params"] + b["hbm_expert_stacks"] == sum(p.numel() * p.element_size() for p in own.values())


def test_mtp_loader_refuses_missing_and_unexpected(emu):
    with pytest.raises(LoadRefused, match="1 missing"):
        emu("mini", drop=("mtp.0.attn.kv_norm.weight",))
    with pytest.raises(LoadRefused, match="1 unexpected"):
        emu("mini", extra=(("mtp.0.attn.extra_thing.weight", "BF16", (4,)),))


# ------------------------------------------------------------------------------------------------ the V2 gate (dry run)
def test_gpu_draft_gate_dry_run_on_cpu(emu, monkeypatch, capsys):
    """tests/gpu/test_draft_gpu.py's V2 gate on the CPU emulations: mini (1 stage), mini3 (3 stages, 2 target
    layers), mini3w (4 experts top-3: duplicate experts across the block rows), and a 150-token prompt (the
    prefill's ring-wrap branch, drafts at A' >= 128)."""
    G = _gpu_module(monkeypatch, emu)
    G._gate_draft("mini", 17)
    G._gate_draft("mini3", 17)
    G._gate_draft("mini3w", 18, chain=False, rowwise=False)
    G._gate_draft("mini3", 150, cs=(6,), chain=True, rowwise=False)
    out = capsys.readouterr().out
    assert out.count("DraftStep static-eager == reference forward_spec") == 4, out
    assert G.COUNTS["gates"] == 4 and G.COUNTS["drafts"] >= 4 * 4, G.COUNTS
    assert all("a byte no-op" in r for r in G.REPORT if "re-write" in r), G.REPORT
    assert any("ctx_rowwise" in r and r.endswith("True") for r in G.REPORT), G.REPORT
    print(out, "\n".join(G.REPORT), G.COUNTS)


def test_predication_changes_exactly_the_context_slots(emu):
    from runtime.draft_step import DraftStep, ring_restore, ring_snapshot

    G_ids = list(range(10, 27))  # L = 17
    model = emu("mini3")
    from runtime.draft_step import prefill_hidden, seed_draft_rings

    out, _, mh = prefill_hidden(model, torch.tensor([G_ids]), model.runtime_mtp.target_layer_ids)
    seed_draft_rings(model, out.reshape(-1)[:1], mh)
    snap = ring_snapshot(model)
    ds = DraftStep(model, _allow_cpu=True)
    g = torch.Generator().manual_seed(3)
    mh6 = torch.randn(6, ds.hidden_dim, generator=g).to(torch.bfloat16)
    for c in range(7):
        ring_restore(model, snap)
        a = 20 + c
        ds.step(a, c, main_hidden=mh6, tok=torch.arange(6) + 5)
        for st, s0 in zip(model.mtp, snap):
            changed = (st.attn.window_kv_cache != s0).flatten(2).any(-1)[0].nonzero().flatten().tolist()
            want = sorted(p % 128 for p in range(a - c + 1, a + 1))
            assert set(changed) <= set(want) and (len(changed) == c), (c, changed, want)
        assert int(ds.out_ids[0, 0]) == 5 + max(c - 1, 0)  # the bonus = tok[ctx_count - 1]


def test_set_inputs_validation_and_refusals(emu):
    from runtime.draft_step import DraftStep

    model = emu("mini")
    ds = DraftStep(model, _allow_cpu=True)
    with pytest.raises(ValueError):
        ds.set_inputs(10, 7)            # ctx_count > 6
    with pytest.raises(ValueError):
        ds.set_inputs(model.max_seq_len - 5, 1)   # block positions past max_pos
    with pytest.raises(ValueError):
        ds.set_inputs(3, 4)             # context row at position 0
    with pytest.raises(ValueError):
        ds.set_inputs(10, None)
    with pytest.raises(RuntimeError):
        ds.replay(10, 1)                # no graph
    ds.set_inputs(model.max_seq_len - 6, 6)
    model.mtp[0].temperature = 1.0
    with pytest.raises(ValueError, match="greedy"):
        DraftStep(model, _allow_cpu=True)
    model.mtp[0].temperature = 0.0
    with pytest.raises(ValueError, match="CUDA"):
        DraftStep(model)


# ------------------------------------------------------------------------------------------------ bench logic
def test_bench_acceptance_logic():
    spec = importlib.util.spec_from_file_location("draft_bench", os.path.join(REPO, "bench", "draft_bench.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    truth = list(range(100, 130))   # targets[i] = token at position L + i
    L = 7
    # a draft at A' predicts positions A'+2..A'+6 = targets[A'+2-L ..]
    exact = [truth[a + 2 - L: a + 7 - L] for a in range(L - 1, L + 20)]
    assert B.accepted(exact[0], truth, L - 1, L) == 5
    assert B.accepted([truth[1], 0, truth[3], 0, 0], truth, L - 1, L) == 1
    assert B.accepted([0] * 5, truth, L - 1, L) == 0
    assert B.accepted(truth[24:29], truth, L + 22, L) == 5
    assert B.accepted(truth[27:30] + [130, 131], truth, L + 25, L) == 3   # the truth ends after 3 positions
    per = {a: (5 if a % 2 else 0) for a in range(L - 1, L + 29)}
    sim = B.simulate_loop(per, L - 1, L + len(truth) - 1)
    assert sim["steps"] >= 1 and sim["tokens"] == sum(1 + n for n in sim["accepts"])
    assert abs(sim["tokens_per_step"] - sim["tokens"] / sim["steps"]) < 1e-12


def test_bench_draft_prompt_dry_run_on_cpu(emu, monkeypatch):
    """bench/draft_bench.py `draft_prompt` on the CPU emulations with the real plumbing: the 1-row StaticStep's greedy
    tokens as truth, the 6-row StaticStep (hidden_layers = the mtp targets) as the verify, DraftStep aliasing its
    main_hidden. Checks the harness (a draft at every position, the loop path), and that the drafts it records equal
    the reference oracle's at two positions (ring state built by the bench's own sequence of draft calls)."""
    from runtime.draft_step import DraftStep, reference_draft
    from runtime.generate import Generator
    from runtime.static_step import StaticStep
    from test_static_emulated_cpu import _fake_static_kernels
    from tiny_tok import make_wordlevel_tokenizer

    spec = importlib.util.spec_from_file_location("draft_bench_dry", os.path.join(REPO, "bench", "draft_bench.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    model = emu("mini3")
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
    hidden = tuple(model.runtime_mtp.target_layer_ids)
    factory = lambda m, **kw: StaticStep(m, pool, max_pos=511, _allow_cpu=True,  # noqa: E731
                                         hidden_layers=hidden if kw.get("rows", 1) > 1 else (), **kw)
    gen = Generator(model, tokenizer=make_wordlevel_tokenizer(2048), eos_ids=[1], device="cpu", mode="static",
                    static_factory=factory)
    ids = list(range(40, 57))
    gen.greedy(ids, 20, stop_on_eos=False)
    targets = list(gen.last_meta["token_ids"])
    gen.reset_static()
    gen.set_rows(6)
    st = gen.static_step()
    ds = DraftStep(model, main_hidden=st.main_hidden, _allow_cpu=True)
    seen = {}
    orig_step = ds.step

    def spy(a, c, graph=False, main_hidden=None, tok=None):
        out = orig_step(a, c, graph=graph, main_hidden=main_hidden, tok=tok)
        if a in (len(ids) + 2, len(ids) + 9):
            seen[a] = (c, ds.drafts.clone(), st.main_hidden[:c].clone(), ds.tok[:c].clone())
        return out

    ds.step = spy
    res = B.draft_prompt(gen, ds, ids, targets, hidden, graph=False)
    n_steps = (len(targets) - 1) // 6
    assert res["positions"] == 1 + 6 * n_steps and sum(res["hist"]) == res["positions"]
    assert res["loop"]["tokens"] == sum(1 + n for n in res["loop_accepts"]) and res["loop"]["steps"] >= 1
    assert set(seen) == {len(ids) + 2, len(ids) + 9}
    # oracle at A' = L+9 (ctx 4 after the second verify), its rings rebuilt independently: prefill seed, then one
    # reference forward_spec call per position L..A' with the reference 6-row verify's main_hidden rows
    a = len(ids) + 9
    c, d_bench, mh_c, tok_c = seen[a]
    assert c == 4 and tok_c[-1].item() == targets[a + 1 - len(ids)]  # the bonus = the truth at A'+1
    from runtime.draft_step import prefill_hidden, seed_draft_rings
    from runtime.ref_multirow import forward_rows

    out, _, mh = prefill_hidden(model, torch.tensor([ids]), hidden)
    seed_draft_rings(model, out.reshape(-1)[:1], mh)
    L = len(ids)
    rows, toks = [], []
    for s_ in range(2):
        _, _, h6 = forward_rows(model, torch.tensor([targets[6 * s_: 6 * s_ + 6]]), L + 6 * s_, hidden)
        rows.append(h6)
        toks += targets[6 * s_ + 1: 6 * s_ + 7]
    allh = torch.cat(rows, 0)
    ref = reference_draft(model, a, allh[: a - L + 1], toks[: a - L + 1])
    assert torch.equal(ref[0].reshape(-1)[1:], d_bench), (ref[0].tolist(), d_bench.tolist())
    assert res["mean_accepted_per_position"] >= 0
