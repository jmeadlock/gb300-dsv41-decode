"""GAP-LEDGER (docs/HANDOFF-GAP-LEDGER.md): CPU tests for bench/gap_ledger.py and the bench/step_anatomy.py family
extension. Stdlib + pytest only; the real receipts in the repo (headline j, W0 step cell, parity-A, peer3 stats) are
read where they exist, and every expected number is recomputed here from the receipt itself, never typed in."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import gap_ledger as G  # noqa: E402
import step_anatomy as S  # noqa: E402

STEP = os.path.join(REPO, "w0", "results", "phase1", "step-pg61.json")
PARITY = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
HEADLINE = os.path.join(REPO, "docs", "headline-20261107j.json")
PEER3 = os.path.join(REPO, "docs", "data", "peer3_stats.json")
need = lambda p: pytest.mark.skipif(not os.path.exists(p), reason=f"{p} not in this checkout")


def _all_cells(led):
    for c, col in led["columns"].items():
        for k, v in col.items():
            yield c, k, v


# ------------------------------------------------------------------------------------------------ family map
def test_family_rules_role_split_and_unsplit():
    f = S.family_of
    assert f("sm100_fp8_fp4_gemm_1d1d_impl") == "moe.routed.gemm"
    assert f("t9b_mxfp8_gemv", "", "moe") == "moe.shared"
    assert f("t9b_mxfp8_gemv", "", "attn") == "dense.proj"
    assert f("t9b_mxfp8_gemv", "", None) == "dense.proj"          # roles known, launched outside any module range
    assert f("t9b_mxfp8_gemv", "", "moe", roles_known=False) == "fp8_gemv.unsplit"
    assert f("gatherTopK", "", "moe") == "moe.router"
    assert f("gatherTopK", "", None) == "glue.indexer"
    assert f("gatherTopK", "", None, roles_known=False) == "topk.unsplit"
    assert f("Kernel2", "cutlass::Kernel2<...>", "moe") == "moe.router"
    assert f("Kernel2", "cutlass::Kernel2<...>", None) == "glue.norm_mhc"
    for name, fam in (("m2_router_plan", "moe.router"), ("t9b8_comp_fp32", "glue.compressor"),
                      ("t9b5_head_fp32", "dense.head"), ("t9a2_stage_inplace", "moe.routed.staging"),
                      ("m1_dedup_warp", "moe.routed.staging"), ("t9b2_moe_combine", "moe.routed.combine"),
                      ("t9b2_swiglu_quant", "moe.swiglu"), ("t8_step_marker", "instrument"),
                      ("vectorized_elementwise_kernel", "glue.torch"), ("t9b2r2_engram_gate", "glue.engram")):
        assert f(name) == fam, name
    assert f("kernel") == "unmapped"                                 # deliberately unidentified
    assert set(fam for fam, _, _ in S.FAMILY_RULES) <= set(S.FAMILIES)


def test_family_mangled_names_match():
    assert S.family_of("_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_implILj128EEEvv") == "moe.routed.gemm"


# ------------------------------------------------------------------------------------------------ step_anatomy on a synthetic sqlite
def _mk_profile(path):
    db = sqlite3.connect(path)
    db.executescript("""
    create table StringIds (id integer primary key, value text);
    create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, correlationId int, shortName int, demangledName int,
        streamId int, gridX int, gridY int, gridZ int);
    create table CUPTI_ACTIVITY_KIND_RUNTIME (start int, end int, correlationId int);
    create table NVTX_EVENTS (start int, end int, text text, textId int, globalTid int);
    create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, bytes int, copyKind int);
    """)
    names = ["t9b_mxfp8_gemv", "sm100_fp8_fp4_gemm_1d1d_impl", "t9b2_moe_combine", "kernel"]
    for i, n in enumerate(names):
        db.execute("insert into StringIds values (?, ?)", (i + 1, n))
    cid = [0]

    def k(s, e, name_id, launch):
        cid[0] += 1
        db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?)",
                   (s, e, cid[0], name_id, name_id, 7, 4, 1, 1))
        db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?)", (launch, launch + 1, cid[0]))
    for base in (0, 100_000):            # two 100 us steps (ns)
        db.execute("insert into NVTX_EVENTS values (?,?,?,?,?)", (base, base + 100_000, "decode_step", None, 1))
        db.execute("insert into NVTX_EVENTS values (?,?,?,?,?)", (base + 1, base + 60_000, "verify", None, 1))
        db.execute("insert into NVTX_EVENTS values (?,?,?,?,?)", (base + 60_001, base + 99_999, "draft", None, 1))
        db.execute("insert into NVTX_EVENTS values (?,?,?,?,?)", (base + 20, base + 50_000, "moe", None, 1))
        k(base + 10_000, base + 20_000, 1, base + 5)          # gemv outside moe -> dense.proj, 10 us
        k(base + 20_000, base + 40_000, 2, base + 30)         # DG in moe, 20 us
        k(base + 30_000, base + 45_000, 1, base + 40)         # gemv in moe -> shared, 15 us, overlaps DG by 10 us
        k(base + 50_000, base + 55_000, 3, base + 49_000)     # combine in moe, 5 us
        k(base + 70_000, base + 80_000, 4, base + 65_000)     # "kernel" in draft, outside moe -> unmapped, 10 us
        db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?)", (base + 1000, base + 3000, 1 << 20, 1))
    db.commit()
    db.close()


def test_step_anatomy_families_and_overlap(tmp_path):
    p = str(tmp_path / "prof.sqlite")
    _mk_profile(p)
    rep = S.analyze(p, log=lambda *a, **k: None)
    assert rep["n_steps"] == 2 and rep["roles_known"]
    fam = {f: round(v["ms_per_step"] * 1e3, 6) for f, v in rep["families"].items()}   # us/step
    assert fam == {"dense.proj": 10.0, "moe.routed.gemm": 20.0, "moe.shared": 15.0, "moe.routed.combine": 5.0,
                   "unmapped": 10.0}
    assert abs(rep["sum_kernel_ms_per_step"] * 1e3 - 60.0) < 1e-9
    # union: [10,45] + [50,55] + [70,80] = 35 + 5 + 10 = 50 us
    assert abs(rep["busy_ms_per_step"] * 1e3 - 50.0) < 1e-9
    assert abs(rep["concurrency"] - 60.0 / 50.0) < 1e-12
    assert abs(rep["overlap"] - 50.0 / 100.0) < 1e-12
    ph = rep["families_by_phase"]
    assert set(ph) == {"verify", "draft"}
    assert abs(ph["draft"]["unmapped"]["ms_per_step"] * 1e3 - 10.0) < 1e-9
    assert "moe.routed.gemm" in ph["verify"] and "unmapped" not in ph["verify"]
    assert rep["unmapped"] == ["kernel"]
    assert rep["memcpy"]["HtoD"]["bytes_per_step"] == 1 << 20
    assert rep["kernels_all"]["t9b_mxfp8_gemv"]["families"] == {"dense.proj": 1.0, "moe.shared": 1.0}
    assert rep["kernels_all"]["t9b_mxfp8_gemv"]["grids"] == {"4x1x1": 4}


def test_step_anatomy_cli_writes_json(tmp_path):
    p = str(tmp_path / "prof.sqlite")
    _mk_profile(p)
    out = str(tmp_path / "a.json")
    r = subprocess.run([sys.executable, os.path.join(BENCH, "step_anatomy.py"), p, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "families [static, diagnostic]" in r.stdout and "role" in r.stdout
    d = json.load(open(out))
    assert "families" in d and "top_kernels" in d and not any(k.startswith("_") for k in d)


# ------------------------------------------------------------------------------------------------ Pg61 side
@need(STEP)
def test_pg61_step_cell_arithmetic():
    d = json.load(open(STEP))
    reps = d["reps"]
    c = G.pg61_step_cell(d, "step")
    dw = sum(r["decode_wall_s"] for r in reps)
    assert c["step_gpu_ms"]["kind"] == "derived"
    assert abs(c["step_gpu_ms"]["value"] - 1000 * dw / sum(r["steps"] - 1 for r in reps)) < 1e-12
    assert abs(c["tokens_per_step"]["value"] - sum(r["gen_tokens"] for r in reps) / sum(r["steps"] for r in reps)) < 1e-12
    assert abs(c["wall_ms_per_token_e2e"]["value"] - 1000 * sum(r["wall_s"] for r in reps)
               / sum(r["gen_tokens"] for r in reps)) < 1e-12
    assert "not GPU" in c["step_gpu_ms"]["how"]
    for k in ("verify_gpu_ms", "draft_gpu_ms", "gpu_ms_per_token"):
        assert c[k]["kind"] == "not measured" and c[k]["value"] is None


@need(PARITY)
def test_pg61_fixture_total_over_total():
    rows = [json.loads(l) for l in open(PARITY) if l.strip()]
    c = G.pg61_fixture(rows, "parity")
    want = 1000 * sum(r["wall_s"] for r in rows) / sum(r["n_tokens"] for r in rows)
    mean_of_ratios = sum(1000 * r["wall_s"] / r["n_tokens"] for r in rows) / len(rows)
    assert abs(c["wall_ms_per_token_e2e"]["value"] - want) < 1e-12
    assert abs(want - mean_of_ratios) > 1e-6          # the two differ on this fixture: the ledger uses total/total
    assert c["step_gpu_ms"]["kind"] == "not measured"


@need(PEER3)
def test_pg61_peer3_cells():
    d = json.load(open(PEER3))
    c = G.pg61_peer3(d, "peer3")
    t6 = d["by_T"]["6"]
    assert c["cold_experts_per_layer_call"]["value"] == t6["cold_exp_mean"]
    assert abs(c["cold_exposed_ms"]["value"] - t6["gb_wait_us_mean"] * 40 / 1000) < 1e-12


def test_pg61_drive_columns_and_chunks():
    assert G.step_tokens_from_chunks([{"lp_tokens": 1}, {"lp_tokens": 3}, {"lp_tokens": 0, "completion_tokens": None}]) == [1, 3]
    assert G.step_tokens_from_chunks([{"lp_tokens": 0, "completion_tokens": 1}, {"lp_tokens": 0, "completion_tokens": 4},
                                      {"lp_tokens": 0, "completion_tokens": 4}]) == [1, 3]
    drive = {"prompts": {"a": {"count": {"step_tokens": [1, 3, 3, 2], "equal_to_parity_a": True},
                               "time": [{"steps": 4, "gen_tokens": 9, "decode_wall_s": 0.03, "wall_s": 0.05,
                                         "first_chunk_tokens": 1}]},
                         "b": {"count": {"step_tokens": [1, 6], "equal_to_parity_a": False},
                               "time": [{"steps": 2, "gen_tokens": 7, "decode_wall_s": 0.01, "wall_s": 0.02,
                                         "first_chunk_tokens": 1}]}}}
    c = G.pg61_drive(drive, "drive")
    assert abs(c["step_gpu_ms"]["value"] - 1000 * 0.04 / 4) < 1e-12        # (4-1) + (2-1) = 4 intervals
    assert abs(c["tokens_per_step"]["value"] - 16 / 6) < 1e-12
    assert abs(c["wall_ms_per_token_decode"]["value"] - 1000 * 0.04 / 14) < 1e-12
    assert c["step_emissions"]["value"] == {"1": 2, "2": 1, "3": 2, "6": 1}


# ------------------------------------------------------------------------------------------------ our side
@need(HEADLINE)
def test_ours_headline_cells():
    h = json.load(open(HEADLINE))
    c = G.ours_headline(h, "hl")
    w = h["headline"]["two_graphs"]
    assert c["verify_gpu_ms"]["value"] == w["ms_verify_graph"]["mean"] and c["verify_gpu_ms"]["kind"] == "measured"
    assert c["draft_gpu_ms"]["value"] == w["ms_draft_graph"]["mean"]
    assert c["step_gpu_ms"]["value"] == w["ms_per_step"]["mean"]
    assert c["gpu_ms_per_token"]["value"] == w["ms_per_token"]
    pr = h["prompts"].values()
    wall = sum(sum(p["wall_ms_fast"]) for p in pr)
    tok = sum(p["tokens"] * len(p["wall_ms_fast"]) for p in pr)
    assert abs(c["wall_ms_per_token_e2e"]["value"] - wall / tok) < 1e-9


def test_ours_tf_cells_and_bytes():
    tf = {"mode": "graph", "map": "/work/docs/maps/hot-v26ar-seed-295.json", "spare_slots": 16,
          "targets": {"a": {"ok": True}, "b": {"ok": False}},
          "reps": [{"prompts": [{"tag": "a", "step_ms": [10.0, 12.0], "hot_unique": 2000, "cold_unique": 10},
                                {"tag": "c", "step_ms": [11.0], "hot_unique": 900, "cold_unique": 0}]}],
          "census": {"tf_graphs": {"0": {"nodes": 2000, "kernel_nodes": 1990, "edges": 1999, "api": "x",
                                         "edge_kinds": {"default": 1999}, "node_types": {"kernel": 1990}}},
                     "loop_graphs": {"2g:B": {"error": "no keep_graph"}}}}
    c = G.ours_tf(tf, "tf")
    assert c["verify_gpu_ms"]["value"] == 11.0 and c["verify_gpu_ms"]["kind"] == "measured"
    assert c["expert_bytes_c2c"]["value"] == 10 * G.EXPERT_BYTES / 3
    assert c["expert_bytes_hbm"]["value"] == 2900 * G.EXPERT_BYTES / 3
    assert abs(c["cold_share"]["value"] - 10 / 2910) < 1e-15
    assert c["graph_nodes tf6[0]"]["value"] == 2000 and c["graph_edges tf6[0]"]["value"] == 1999
    assert c["graph_nodes loop[2g:B]"]["kind"] == "not measured"
    assert c["draft_gpu_ms"]["kind"] == "not measured"
    tf["mode"] = "static"
    assert G.ours_tf(tf, "tf")["verify_gpu_ms"]["kind"] == "not measured"   # eager wall is not the graph


def test_ours_anatomy_unattributed_and_old_format():
    an = {"n_steps": 4, "roles_known": True, "families": {"moe.routed.gemm": {"ms_per_step": 3.0, "launches_per_step": 40},
                                                         "dense.proj": {"ms_per_step": 2.5, "launches_per_step": 200}},
          "busy_ms_per_step": 5.4, "overlap": 0.1, "concurrency": 1.02, "unmapped": []}
    g = G.cell(7.0, "measured", "events")
    c = G.ours_anatomy(an, "an", g)
    assert c["fam.moe.routed.gemm"]["kind"] == "static" and c["fam.moe.routed.gemm"]["value"] == 3.0
    assert c["fam.attn"]["value"] == 0.0 and c["fam.attn"]["kind"] == "static"
    assert c["sum_kernels"]["value"] == 5.5
    assert c["unattributed"]["kind"] == "derived" and c["unattributed"]["value"] == 1.5
    assert "7.000 - 5.500 = 1.500" in c["unattributed"]["how"]
    assert "[static, diagnostic]" in c["busy_ms"]["how"]
    old = G.ours_anatomy({"top_kernels": []}, "old", g)
    assert all(v["kind"] == "not measured" for v in old.values())
    assert G.ours_anatomy(an, "an", None)["unattributed"]["kind"] == "not measured"


# ------------------------------------------------------------------------------------------------ the ledger
def test_cell_kinds_enforced():
    with pytest.raises(ValueError):
        G.cell(1.0, "estimated", "x")
    with pytest.raises(ValueError):
        G.cell(None, "measured", "x")
    assert G.nm("why")["value"] is None


@need(HEADLINE)
@need(STEP)
@need(PARITY)
def test_build_ledger_every_cell_classified(tmp_path):
    out, md = str(tmp_path / "l.json"), str(tmp_path / "l.md")
    rc = G.main(["build", "--out", out, "--md", md])
    assert rc == 0
    led = json.load(open(out))
    n = 0
    for c, k, v in _all_cells(led):
        n += 1
        assert v["kind"] in G.KINDS, (c, k)
        assert (v["value"] is None) == (v["kind"] == "not measured"), (c, k)
        assert v["how"], (c, k)
    assert n > 100
    for c in ("ours.headline", "pg61.step_cell", "pg61.fixture"):
        assert all(k in led["columns"][c] for k, _ in G.METRICS), c
    for f in G.FAMILIES:                                      # no Pg61 family cell can be anything but not measured
        assert led["columns"]["pg61.step_cell"][f"fam.{f}"]["kind"] == "not measured"
    assert "step_gpu_ms" in led["resolvable_pg61_from_existing"]
    assert set(led["pending"]) == {"ours.matched", "ours.loop_static", "pg61.matched"}
    text = open(md).read()
    assert "| " in text and "|  |" not in text                # no blank cell in the table
    g = {x["metric"]: x for x in led["gaps"] if x["pg61"] == "pg61.step_cell"}
    assert "NOT matched" in g["step_gpu_ms"]["matched"]


# ------------------------------------------------------------------------------------------------ token bytes -> ids
def test_bytes_to_unicode_bijection():
    m = G.bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256


def _tok_json(pieces):
    enc = G.bytes_to_unicode()
    vocab = {"".join(enc[b] for b in p): i for i, p in enumerate(pieces)}
    return {"model": {"type": "BPE", "vocab": vocab},
            "added_tokens": [{"id": len(pieces), "content": "<｜end▁of▁sentence｜>"}]}


def test_pg61_ids_synthetic_vocab():
    pieces = [b"A", b" mixture", "é".encode(), b"\xe2\x80", b"\n\n"]
    table, added, amb = G.vocab_by_bytes(_tok_json(pieces))
    assert not amb and len(table) == len(pieces)
    rec = {"logprobs": [{"token": "A", "bytes": [65]}, {"token": " mixture", "bytes": list(b" mixture")},
                        {"token": "é", "bytes": list("é".encode())}, {"token": "bytes:\\xe2\\x80", "bytes": [0xE2, 0x80]},
                        {"token": "\n\n", "bytes": None}, {"token": "<｜end▁of▁sentence｜>", "bytes": []},
                        {"token": "zz", "bytes": list(b"zz")}]}
    ids, bad = G.pg61_ids(rec, table, added)
    assert ids == [0, 1, 2, 3, 4, 5, -1] and bad == [6]


@need(PARITY)
def test_pg61_ids_cover_parity_a_bytes():
    rows = [json.loads(l) for l in open(PARITY) if l.strip()]
    pieces = sorted({bytes(e["bytes"]) for r in rows for e in r["logprobs"] if e.get("bytes") is not None})
    table, added, _ = G.vocab_by_bytes(_tok_json(pieces))
    for r in rows:
        ids, bad = G.pg61_ids(r, table, added)
        assert not bad and len(ids) == r["n_tokens"], r["tag"]
        assert sum(1 for i in ids if i >= len(pieces)) == (r["finish_reason"] == "stop")   # the EOS: added token
        text = b"".join(pieces[i] for i in ids if i < len(pieces)).decode("utf-8", errors="replace")
        assert text == r["text"], r["tag"]


# ------------------------------------------------------------------------------------------------ census helpers
def test_edge_tally_and_census_error_path():
    assert G.tally_edges([(0, 0, 0), (1, 1, 0), (1, 1, 0), (1, 2, 0)]) == {
        "default": 1, "programmatic": 2, "programmatic(launch_completion)": 1}
    assert "error" in G.graph_census(object())                # never raises
    assert G.family_tally(["t9b_mxfp8_gemv", None, "sm100_fp8_fp4_gemm_1d1d_impl"]) == {
        "fp8_gemv.unsplit": 1, "unnamed": 1, "moe.routed.gemm": 1}


def test_keep_graphs_swaps_and_restores():
    torch = pytest.importorskip("torch")
    orig = torch.cuda.CUDAGraph
    undo = G.keep_graphs()
    try:
        assert torch.cuda.CUDAGraph is not orig and issubclass(torch.cuda.CUDAGraph, orig)
    finally:
        undo()
    assert torch.cuda.CUDAGraph is orig


def test_import_is_stdlib_only():
    code = ("import sys; sys.path.insert(0, %r); import gap_ledger; "
            "bad = [m for m in ('torch', 'vllm') if m in sys.modules]; assert not bad, bad" % BENCH)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_box_parsers():
    ap = G.build_parser()
    a = ap.parse_args(["ours-tf", "--model-dir", "/ref", "--json", "x.json", "--census-loop"])
    assert a.mode == "graph" and a.fp32_impl == "fixed" and a.moe_rowmatch == "on" and a.spare_slots == 16
    with pytest.raises(SystemExit):
        G.main(["ours-tf", "--model-dir", "/ref", "--json", "x.json", "--profile", "4:8"])   # graph + nsys refused
    a = ap.parse_args(["pg61-drive", "--json", "y.json"])
    assert a.reps == 3 and a.max_new == 256


# ------------------------------------------------------------------------------------------------ LOOSE-ENDS-2: census without a 2nd instantiate
class _FakeGraph:
    """torch.cuda.CUDAGraph's keep_graph contract (torch >= 2.13), CPU-only. keep_graph=False instantiates at capture
    end and refuses a user instantiate(); keep_graph=True waits for ONE explicit instantiate() and refuses a second --
    'instantiate() is intended to be called by the user only when keep_graph=true' is the message of both refusals.
    `auto` = the variant whose capture_end also instantiates (what the --census-loop crash looked like)."""
    auto = False

    def __new__(cls, keep_graph=False):
        self = object.__new__(cls)
        self.keep, self.ended, self.exec, self.inst_calls = keep_graph, False, False, 0
        return self

    def capture_end(self):
        self.ended = True
        if not self.keep or self.auto:
            self.exec = True

    def instantiate(self):
        self.inst_calls += 1
        if not self.keep or self.exec:
            raise RuntimeError("instantiate() is intended to be called by the user only when keep_graph=true")
        self.exec = True

    def raw_cuda_graph(self):
        return 1


class _AutoGraph(_FakeGraph):
    auto = True


def _patched_torch(monkeypatch, cls):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", cls)
    return torch


def test_keep_graphs_instantiates_once_and_replays_without_a_second(monkeypatch):
    torch = _patched_torch(monkeypatch, _FakeGraph)
    undo = G.keep_graphs()
    try:
        g = torch.cuda.CUDAGraph()
        assert g.keep is True                       # kept, so the census can read the cudaGraph_t
        g.capture_end()
        assert g.exec and g.inst_calls == 1         # exactly one instantiate, at capture end
        g.instantiate()                             # a caller's (or the runtime's) own call: a no-op, not a crash
        assert g.inst_calls == 1
        assert torch.cuda.CUDAGraph(keep_graph=False).keep is True   # an explicit keep_graph=False is overridden
    finally:
        undo()
    assert torch.cuda.CUDAGraph is _FakeGraph


def test_keep_graphs_survives_torch_instantiating_at_capture_end(monkeypatch):
    """The --census-loop crash: capture_end already instantiated (torch's own, or a graph the runtime built with
    keep_graph=False) and our extra instantiate() raised 'intended to be called by the user only when keep_graph=true'."""
    torch = _patched_torch(monkeypatch, _AutoGraph)
    undo = G.keep_graphs()
    try:
        g = torch.cuda.CUDAGraph()
        g.capture_end()                             # must not raise
        assert g.exec and "intended to be called by the user" in g._census_note
        g.instantiate()                             # still a no-op afterwards
    finally:
        undo()
    # without the guard the same sequence raises (the regression this fixes)
    bare = _AutoGraph(keep_graph=True)
    bare.capture_end()
    with pytest.raises(RuntimeError, match="intended to be called by the user"):
        bare.instantiate()


def test_keep_graphs_only_swallows_the_double_instantiate_error(monkeypatch):
    class _Broken(_FakeGraph):
        def instantiate(self):
            raise RuntimeError("CUDA error: out of memory")

    torch = _patched_torch(monkeypatch, _Broken)
    undo = G.keep_graphs()
    try:
        g = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="out of memory"):
            g.capture_end()
    finally:
        undo()
    assert G._is_double_instantiate(RuntimeError("instantiate() is intended to be called by the user only when keep_graph=true"))
    assert not G._is_double_instantiate(RuntimeError("CUDA error: out of memory")) and not G._is_double_instantiate(ValueError("x"))


@pytest.mark.parametrize("cls", [_FakeGraph, _AutoGraph])
def test_run_census_loop_reads_graphs_without_instantiating_twice(monkeypatch, cls, capsys):
    import types

    torch = _patched_torch(monkeypatch, cls)
    pytest.importorskip("runtime.dspark_loop")
    import runtime.dspark_loop as DL

    class _Loop:
        made = []

        def __init__(self, gen, max_new):
            self.graphs, self.closed = {}, False
            _Loop.made.append(self)

        def capture(self, n_graphs):
            self.graphs = {}
            for name in (("A", "B") if n_graphs == 2 else ("AB", "B")):
                g = torch.cuda.CUDAGraph()
                g.capture_end()
                self.graphs[name] = g

        def close(self):
            self.closed = True

    monkeypatch.setattr(DL, "DSparkLoop", _Loop)
    undo = G.keep_graphs()
    try:
        tf = torch.cuda.CUDAGraph()
        tf.capture_end()
        gen = types.SimpleNamespace(static=types.SimpleNamespace(graphs={6: tf}))
        rep = {"errors": []}
        cen = G.run_census(gen, types.SimpleNamespace(census_loop=True, max_new=8), rep)
    finally:
        undo()
    assert rep["errors"] == [] and rep["census"] is cen
    assert list(cen["tf_graphs"]) == ["6"] and list(cen["loop_graphs"]) == ["2g:A", "2g:B", "1g:AB", "1g:B"]
    assert all(isinstance(c, dict) for d in cen.values() for c in d.values())    # (no libcudart here: {'error': ...})
    assert _Loop.made[0].closed
    assert tf.inst_calls == 1 and all(g.inst_calls == 1 for g in _Loop.made[0].graphs.values())   # census never instantiated
    assert "[census] loop_graphs 2g:A" in capsys.readouterr().out


def test_run_census_records_a_failing_loop_and_keeps_the_tf_census(monkeypatch):
    import types

    _patched_torch(monkeypatch, _FakeGraph)
    pytest.importorskip("runtime.dspark_loop")
    import runtime.dspark_loop as DL

    class _Bad:
        def __init__(self, gen, max_new):
            self.graphs, self.closed = {}, False
            _Bad.me = self

        def capture(self, n_graphs):
            raise RuntimeError("boom")

        def close(self):
            self.closed = True

    monkeypatch.setattr(DL, "DSparkLoop", _Bad)
    gen = types.SimpleNamespace(static=types.SimpleNamespace(graphs={6: object()}))
    rep = {"errors": []}
    cen = G.run_census(gen, types.SimpleNamespace(census_loop=True, max_new=8), rep)
    assert list(cen["tf_graphs"]) == ["6"] and cen["loop_graphs"] == {} and _Bad.me.closed
    assert rep["errors"][0]["stage"] == "census_loop" and "boom" in rep["errors"][0]["traceback"]


def test_graph_census_never_calls_instantiate():
    g = _FakeGraph(keep_graph=True)
    out = G.graph_census(g)                       # no libcudart on the CPU box -> an error dict, never an instantiate
    assert isinstance(out, dict) and g.inst_calls == 0
