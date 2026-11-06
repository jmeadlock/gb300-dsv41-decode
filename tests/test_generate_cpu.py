"""CPU tests of the T5 driver's pure parts (Mac: `.venv/bin/python -m pytest -q tests/`).

  * fixture encode: the checkpoint's encoding.py (vendored copy) in chat mode + a DSv4.1 `tokenizer.json` reproduce
    the 8 prompt-token counts and first ids of docs/fixture-render-20261106.json (the box render with the
    checkpoint's own tokenizer). The real checkpoint is not on the Mac: the tokenizer comes from $DSV41_TOKENIZER_DIR
    or a known local DSv4.1 copy, and the test skips with the reason when neither exists or `tokenizers` is missing;
  * token strings: single-id decode re-creates every fixture token string of all 8 Pg61 records, and EOS decodes to
    the fixture's `<｜end▁of▁sentence｜>` bytes;
  * logprob record shaping from fake logits (fp32 log-softmax over the full vocab, stable top-k, strings/bytes);
  * greedy over a fake model: the generate.py call pattern, EOS/length, n_tokens/text, temperature 0;
  * JSONL round trip through bench/parity.py's load()/compare()/grade(); self-replay divergence report;
  * profiler window (start after N warm steps, stop after M) with a recording profiler; the first_light step cell
    and summary on the fake model.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch

from runtime.generate import (Generator, dump_record, encode_messages_to_ids, eos_ids_from, load_encoding,
                              logprob_entry, make_record, record_divergence)
from tiny_tok import FakeModel, log_softmax_ref, make_wordlevel_tokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(REPO, "reference", "deepseek-inference")
FIXTURE = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
FIXTURE2 = os.path.join(REPO, "w0", "results", "phase1", "parity-A2.jsonl")
RENDER = os.path.join(REPO, "docs", "fixture-render-20261106.json")
# DSv4.1 tokenizer.json candidates on the Mac (the checkpoint's own is only on the box). The fallback is a modified
# copy of DeepSeek-V4-Flash-Vision-Exp's (only id 129264 renamed, per its README); the tests below are what qualify it.
TOKENIZER_CANDIDATES = [os.environ.get("DSV41_TOKENIZER_DIR"),
                        os.path.expanduser("~/code/deepseek-recipe/static/tokenizers/v41")]

tokenizers = pytest.importorskip("tokenizers", reason="`tokenizers` not installed in this venv")


def _import_parity():
    spec = importlib.util.spec_from_file_location("bench_parity", os.path.join(REPO, "bench", "parity.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _import_first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fixture():
    with open(FIXTURE) as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def dsv41_tok():
    for d in TOKENIZER_CANDIDATES:
        if d and os.path.exists(os.path.join(d, "tokenizer.json")):
            return tokenizers.Tokenizer.from_file(os.path.join(d, "tokenizer.json")), d
    pytest.skip("no DSv4.1 tokenizer.json on this machine (set DSV41_TOKENIZER_DIR); the real one is on the box")


@pytest.fixture(autouse=True)
def _restore_default_device():
    yield
    torch.set_default_device("cpu")


# ------------------------------------------------------------------------------------------------ encode
def test_fixture_encode_reproduces_box_render(dsv41_tok):
    tok, src = dsv41_tok
    enc, path = load_encoding(None)
    assert path.endswith(os.path.join("encoding", "encoding.py"))
    with open(RENDER) as f:
        render = {r["tag"]: r for r in json.load(f)}
    rows = _fixture()
    assert len(rows) == 8
    got = {}
    for r in rows:
        ids = encode_messages_to_ids(enc, tok, [{"role": "user", "content": r["prompt"]}], "chat")
        got[r["tag"]] = len(ids)
        box = render[r["tag"]]["rendered"]["chat"]
        assert len(ids) == box["n"] == r["usage"]["prompt_tokens"], (r["tag"], len(ids), box["n"])
        assert ids[:6] == box["ids_head"], (r["tag"], ids[:6], box["ids_head"])
    assert list(got.values()) == [42, 47, 54, 38, 34, 49, 46, 31]
    print(f"tokenizer {src}: counts {got}")


def test_generator_encode_matches_and_eos(dsv41_tok):
    tok, _ = dsv41_tok
    eos = eos_ids_from(REF_DIR, tok)  # the checkpoint's tokenizer_config.json (vendored) names the EOS token
    assert eos == [1]
    gen = Generator(FakeModel(), tokenizer=tok, eos_ids=eos)
    ids = gen.encode([{"role": "user", "content": _fixture()[0]["prompt"]}])
    assert len(ids) == 42 and ids[:2] == [0, 128803]
    rows = {r["tag"]: r for r in _fixture()}
    eos_fx = rows["json"]["logprobs"][-1]
    assert gen.token_str(1) == eos_fx["token"] and list(gen.token_str(1).encode("utf-8")) == eos_fx["bytes"]


def test_single_id_decode_recreates_fixture_token_strings(dsv41_tok):
    """Re-tokenize every Pg61 completion text, single-id decode each id: the strings (and bytes) equal the fixture's
    logprob tokens, position by position, for all 8 records -- the token-string convention the parity grader keys on."""
    tok, _ = dsv41_tok
    gen = Generator(FakeModel(), tokenizer=tok, eos_ids=[1])
    for r in _fixture():
        fx = r["logprobs"][:-1] if r["finish_reason"] == "stop" else r["logprobs"]
        ids = tok.encode(r["text"], add_special_tokens=False).ids
        assert [gen.token_str(i) for i in ids] == [e["token"] for e in fx], r["tag"]
        assert all(list(gen.token_str(i).encode("utf-8")) == e["bytes"] for i, e in zip(ids, fx)), r["tag"]
        assert gen.decode(ids) == r["text"], r["tag"]


# ------------------------------------------------------------------------------------------------ shaping
def test_logprob_entry_shapes_fp32_full_vocab_stable_topk():
    V = 50
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(V, generator=g, dtype=torch.float32) * 3
    logits[7] = logits[11] = logits[3] = float(logits.max()) + 1.0  # three-way tie at the top
    logits_bf16 = logits.to(torch.bfloat16)  # any dtype in, fp32 math
    names = {i: f"tok{i}" for i in range(V)}
    e = logprob_entry(logits, 3, 20, names.__getitem__)
    ref = log_softmax_ref(logits)
    assert e["token"] == "tok3" and e["bytes"] == list(b"tok3")
    assert e["logprob"] == float(torch.log_softmax(logits, -1)[3])
    assert abs(e["logprob"] - ref[3]) < 1e-5
    top = e["top_logprobs"]
    assert len(top) == 20 and [t["token"] for t in top[:3]] == ["tok3", "tok7", "tok11"]  # ties -> lower id first
    assert all(top[i]["logprob"] >= top[i + 1]["logprob"] for i in range(19))
    want = sorted(range(V), key=lambda i: (-ref[i], i))[:20]
    assert [t["token"] for t in top] == [f"tok{i}" for i in want]
    assert set(e) == {"token", "logprob", "bytes", "top_logprobs"}
    assert set(top[0]) == {"token", "logprob", "bytes"}
    lse = torch.logsumexp(torch.tensor([float(x) for x in torch.log_softmax(logits, -1)]), 0)
    assert abs(float(lse)) < 1e-5
    e2 = logprob_entry(logits_bf16, 3, 5, names.__getitem__)
    assert len(e2["top_logprobs"]) == 5 and isinstance(e2["logprob"], float)


def test_record_schema_matches_fixture():
    fx = _fixture()[3]  # json: stop
    rec = make_record(tag="json", prompt=fx["prompt"], model_name="m", prompt_tokens=38, gen_ids=[5, 6, 1],
                      entries=fx["logprobs"][:3], finish_reason="stop", text="x")
    assert set(rec) <= set(fx) and set(fx) - set(rec) == {"wall_s"}
    assert set(rec["usage"]) <= set(fx["usage"])
    assert rec["usage"] == {"prompt_tokens": 38, "total_tokens": 41, "completion_tokens": 3}
    assert set(rec["logprobs"][0]) == set(fx["logprobs"][0])


# ------------------------------------------------------------------------------------------------ greedy (fake)
def _gen(model=None, **kw):
    tok = make_wordlevel_tokenizer(64)
    return Generator(model or FakeModel(), tokenizer=tok, eos_ids=[1], **kw)


def test_greedy_call_pattern_length():
    m = FakeModel()
    gen = _gen(m)
    assert m.temperature == 0.0  # Generator makes sample() an argmax
    ids = [5, 9, 12, 3]
    rec = gen.greedy(ids, 6, tag="t", prompt="p")
    # generate.py: prefill tokens[:, 0:L] at 0, then one token at its own position; max_new forwards in total
    assert m.calls == [((1, 4), 0), ((1, 1), 4), ((1, 1), 5), ((1, 1), 6), ((1, 1), 7), ((1, 1), 8)]
    assert rec["finish_reason"] == "length" and rec["n_tokens"] == 6 and len(rec["logprobs"]) == 6
    meta = gen.last_meta
    want, last = [], ids[-1]
    for k in range(6):
        last = int(m.logits_for(4 + k, last).argmax())
        want.append(last)
    assert meta["token_ids"] == want
    assert [e["token"] for e in rec["logprobs"]] == [f"t{i}" for i in want]
    assert rec["text"] == " ".join(f"t{i}" for i in want)
    assert rec["usage"] == {"prompt_tokens": 4, "total_tokens": 10, "completion_tokens": 6}
    assert all(e["token"] == e["top_logprobs"][0]["token"] for e in rec["logprobs"])
    assert len(meta["logits_sha256"]) == 64 and meta["n_forward"] == 6


def test_greedy_stops_on_eos_keeps_it_in_logprobs():
    m = FakeModel(eos_at_pos=6)  # the 3rd generated token (positions 4, 5, 6)
    gen = _gen(m)
    rec = gen.greedy([5, 9, 12, 3], 10)
    assert rec["finish_reason"] == "stop" and rec["n_tokens"] == 3
    assert rec["logprobs"][-1]["token"] == "<eos>" and "<eos>" not in rec["text"]
    assert len(m.calls) == 3  # no forward after EOS
    assert gen.greedy([5, 9, 12, 3], 10, stop_on_eos=False)["n_tokens"] == 10


def test_greedy_refuses_past_max_seq_len():
    gen = _gen(FakeModel(max_seq_len=8))
    with pytest.raises(ValueError):
        gen.greedy([1, 2, 3, 4, 5], 4)


def test_self_replay_identical_and_divergence_report():
    gen = _gen()
    a = gen.greedy([5, 9, 12, 3], 8, tag="t")
    da = gen.last_meta["logits_sha256"]
    b = gen.greedy([5, 9, 12, 3], 8, tag="t")
    assert dump_record(a) == dump_record(b) and gen.last_meta["logits_sha256"] == da
    assert record_divergence(a, b) is None
    c = json.loads(dump_record(b))
    c["logprobs"][5]["top_logprobs"][3]["logprob"] += 1e-7
    assert record_divergence(a, c) == {"kind": "logprob", "index": 5, "abs_dlogprob_chosen": 0.0}
    c["logprobs"][2]["token"] = "zzz"
    assert record_divergence(a, c)["kind"] == "token" and record_divergence(a, c)["index"] == 2


# ------------------------------------------------------------------------------------------------ parity round trip
def test_jsonl_roundtrip_through_parity(tmp_path):
    parity = _import_parity()
    gen = _gen()
    recs = [gen.greedy([5, 9, 12, k], 12, tag=f"p{k}", prompt=f"prompt {k}") for k in range(3)]
    p = tmp_path / "cand.jsonl"
    p.write_text("".join(dump_record(r) + "\n" for r in recs))
    loaded = parity.load(str(p))
    assert list(loaded) == ["p0", "p1", "p2"] and loaded["p1"] == recs[1]
    for tag, r in loaded.items():
        m = parity.compare(r, r)
        assert m["first_divergence"] is None and m["top1_flips"] == 0 and m["top1_flips_weighted"] == 0.0
        assert m["max_abs_dlogprob"] == 0.0 and m["shared_prefix"] == r["n_tokens"]
    rows, ok = parity.grade(loaded, loaded, loaded)
    assert ok and all(v == "ok" for *_, v in rows)


def test_parity_on_the_real_fixture_records(tmp_path):
    """A Pg61 record re-serialized by dump_record still loads and compares with itself as identical."""
    parity = _import_parity()
    rows = _fixture()
    p = tmp_path / "a.jsonl"
    p.write_text("".join(dump_record({k: r[k] for k in r if k != "wall_s"}) + "\n" for r in rows))
    a, ref = parity.load(str(p)), parity.load(FIXTURE)
    for tag in ref:
        m = parity.compare(ref[tag], a[tag])
        assert m["first_divergence"] is None and m["top1_flips"] == 0 and m["max_abs_dlogprob"] == 0.0


# ------------------------------------------------------------------------------------------------ profiler + cell
class _RecProfiler:
    def __init__(self, gen_ref):
        self.events, self.gen_ref = [], gen_ref

    def start(self):
        self.events.append(("start", self.gen_ref[0].n_steps))

    def stop(self):
        self.events.append(("stop", self.gen_ref[0].n_steps))


def test_profile_window_start_after_warm_stop_after_m():
    ref = [None]
    prof = _RecProfiler(ref)
    gen = _gen(profiler=prof)
    ref[0] = gen
    gen.greedy([5, 9], 4)  # 3 decode steps before arming: the window counts from arm time
    w = gen.arm_profile(warm=2, steps=3)
    gen.greedy([5, 9], 10)  # 9 decode steps
    assert prof.events == [("start", 3 + 2), ("stop", 3 + 2 + 3)]
    assert w["started"] == 5 and w["stopped"] == 8


def test_first_light_step_cell_and_summary_on_fake_model(tmp_path):
    fl = _import_first_light()
    m = FakeModel()
    gen = _gen(m)
    sc = fl.step_cell(gen, [5, 9, 12, 3], 40, stats_from=8)
    assert sc["timer"] == "wall" and sc["n_steps"] == 40 and len(sc["step_ms"]) == 40
    assert sc["ms_per_step"]["n"] == 33 and sc["ms_per_step"]["p50"] <= sc["ms_per_step"]["p90"]
    assert len(sc["token_ids"]) == 41
    gen.greedy([5, 9, 12, 3], 12, stop_on_eos=False)
    assert sc["token_ids"][:12] == gen.last_meta["token_ids"]  # step cell and greedy agree token for token
    assert fl.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5 and fl.percentile([5.0], 90) == 5.0
    assert fl.rep_path("x/c.jsonl", 1) == "x/c.jsonl" and fl.rep_path("x/c.jsonl", 2) == "x/c-replay.jsonl"
    assert fl.rep_path("x/c.jsonl", 3) == "x/c-replay2.jsonl"
    report = {"date": "d", "env": {"device": None}, "errors": [], "step_cell": sc}
    assert len(fl.summary_lines(report, "j.json")) == 12
    assert len(fl.summary_lines({"date": "d", "env": {}, "errors": []}, "j.json")) == 12


def test_first_light_self_replay_helper(tmp_path):
    fl = _import_first_light()
    gen = _gen()
    paths, metas = [], []
    for rep in (1, 2):
        p = tmp_path / f"r{rep}.jsonl"
        ms = []
        with open(p, "w") as f:
            for k in range(2):
                rec = gen.greedy([5, 9, k], 6, tag=f"p{k}")
                f.write(dump_record(rec) + "\n")
                ms.append(dict(gen.last_meta, tag=f"p{k}"))
        paths.append(str(p))
        metas.append(ms)
    sr = fl.self_replay(paths[0], paths[1], metas[0], metas[1])
    assert sr["files_byte_identical"] and sr["identical"] == 2 and sr["logits_digest_equal"] == 2


class _FlakyModel(FakeModel):
    """Raises on its 4th prefill (the fixture's 4th prompt, `json`, in rep 1 only)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prefills = 0

    def __call__(self, tokens, start_pos):
        if start_pos == 0:
            self.prefills += 1
            if self.prefills == 4:
                raise RuntimeError("injected failure")
        return super().__call__(tokens, start_pos)


def test_first_light_measure_end_to_end_on_fake_model(tmp_path, capsys):
    """Everything first_light does after boot, on the real fixture with a fake model: 2 reps written as JSONL, one
    prompt failing (traceback logged, run continues), bench/parity.py run as a subprocess on both reps, self-replay,
    the step cell, the report JSON and the 12-line summary."""
    fl = _import_first_light()
    gen = _gen(_FlakyModel())
    out, js = str(tmp_path / "cand.jsonl"), str(tmp_path / "fl.json")
    a = fl.build_parser().parse_args(["--model-dir", "unused", "--fixture", FIXTURE, "--out", out, "--json", js,
                                      "--max-new", "12", "--reps", "2", "--step-cell", "40", "--stats-from", "8"])
    report = {"date": "d", "env": {"device": None}, "errors": []}
    rc = fl.measure(gen, a, report, n_layers=3, out=out, json_path=js, fixture2=FIXTURE2)
    assert rc == 2 and [(e["stage"], e.get("tag")) for e in report["errors"]] == [("rep1", "json")]
    assert "injected failure" in report["errors"][0]["traceback"]
    assert os.path.exists(fl.rep_path(out, 2))
    with open(out) as f:
        assert [json.loads(l)["tag"] for l in f] == ["prose", "code", "math", "shell", "tool", "long", "story"]
    p1 = report["parity"][1]
    assert p1["returncode"] == 1 and p1["verdict_line"].startswith("identical 0/8")  # fake tokens != Pg61's
    assert p1["per_prompt"]["json"]["verdict"] == "MISSING"
    assert report["parity"][2]["per_prompt"]["json"]["verdict"] == "FAIL"  # present in rep 2
    sr = report["self_replay"][2]
    assert sr["n"] == 8 and sr["identical"] == 7 and not sr["per_prompt"]["json"]["identical"]
    assert all(v["identical"] and v["logits_digest_equal"] for t, v in sr["per_prompt"].items() if t != "json")
    sc = report["step_cell"]
    assert sc["n_steps"] == 40 and sc["vs_rep1_tokens"] == {"compared": 12, "first_divergence": None}
    assert all(r["prompt_tokens_match"] is False for r in report["reps"][1]["prompts"] if "error" not in r)
    with open(js) as f:
        assert json.load(f)["self_replay"]["2"]["identical"] == 7
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("errors: 1 | report ") and lines[-12].startswith("FIRST LIGHT")


def test_greedy_refuses_non_finite_logits():
    class NaNModel(FakeModel):
        def logits_for(self, pos, last_tok):
            x = super().logits_for(pos, last_tok)
            if pos == 6:
                x[3] = float("nan")
            return x

    with pytest.raises(FloatingPointError):
        _gen(NaNModel()).greedy([5, 9, 12, 3], 8)
