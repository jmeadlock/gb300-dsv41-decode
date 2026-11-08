"""M4-P1 (docs/HANDOFF-M4-P1-PREFILL-ANATOMY.md): bench/prefill_anatomy.py and step_anatomy's prefill family map, on
the Mac. CPU timing means nothing here: what is tested is the instrument's plumbing -- the family rules, the NVTX
innermost-range attribution and the sqlite analyzer on a synthetic nsys export, the hook probe's census on the
emulated mini model (calls per layer = distinct routed experts, rows = L * topk, hot/cold by pool tier, hooks removed,
outputs unperturbed), the stage-B sliced MoE's bit-equality with the reference MoE.forward, and the TTFT model's
arithmetic and labels.
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BENCH = os.path.join(REPO, "bench")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import prefill_anatomy as PA  # noqa: E402
import step_anatomy as SA  # noqa: E402


# ------------------------------------------------------------------------------------------------ family map
@pytest.mark.parametrize("name,role,fam", [
    ("fp4_gemm_kernel__kernel", "expert.cold", "moe.routed.gemm"),
    ("fp4_gemm_kernel__kernel", "expert.hot", "moe.routed.gemm"),
    ("act_quant_kernel__kernel", "expert.hot", "moe.routed.quant"),
    ("vectorized_elementwise_kernel", "expert.cold", "moe.routed.glue"),
    ("DeviceSelectSweepKernel", "moe", "moe.routed.index"),
    ("write_indices", "moe", "moe.routed.index"),
    ("index_elementwise_kernel", "expert.hot", "moe.routed.index"),
    ("indexing_backward_kernel", "moe", "moe.routed.combine"),
    ("DeviceRadixSortSingleTileKernel", "moe", "moe.routed.combine"),
    ("nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT", "gate", "moe.router"),
    ("gatherTopK", "gate", "moe.router"),
    ("fp8_gemm_kernel__kernel", "shared", "moe.shared"),
    ("vectorized_elementwise_kernel", "moe", "moe.other"),
    ("fp8_gemm_kernel__kernel", "attn", "dense.proj"),
    ("fp8_gemm_kernel__kernel", "engram", "engram"),
    ("fp8_gemm_kernel__kernel", None, "dense.proj"),
    ("sparse_attn_kernel__kernel", "attn", "attn.sparse"),
    ("sparse_attn_fwd_kernel", "attn", "attn.sparse"),
    ("nvjet_sm103_tst_88x64_64x11_4x1_v_bz_TNN", "attn", "attn.wo_a"),
    ("act_quant_kernel__kernel", "attn", "glue.quant_rope"),
    ("fp4_quant_kernel__kernel", "attn", "glue.quant_rope"),
    ("nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT", None, "glue.mhc"),
    ("hc_split_sinkhorn_kernel__kernel", None, "glue.mhc"),
    ("gatherTopK", "indexer", "attn.indexer"),
    ("vectorized_elementwise_kernel", "compressor", "attn.compressor"),
    ("gemv2T_kernel_val", "head", "dense.head"),
    ("CatArrayBatchedCopy_vectorized", None, "glue.torch"),
    ("vectorized_elementwise_kernel", None, "glue.torch"),
    ("kernel", None, "unmapped"),
])
def test_prefill_family_map(name, role, fam):
    assert SA.prefill_family_of(name, "", role) == fam
    assert fam in SA.PREFILL_FAMILIES


def test_prefill_map_roles_unknown_skips_role_rules():
    assert SA.prefill_family_of("fp8_gemm_kernel__kernel", "", "shared", roles_known=False) == "dense.proj"
    assert SA.prefill_family_of("act_quant_kernel__kernel", "", "expert.hot", roles_known=False) == "glue.quant_rope"
    assert SA.prefill_family_of("fp4_gemm_kernel__kernel", "", None, roles_known=False) == "moe.routed.gemm"


def test_decode_family_map_unchanged():
    """The prefill map is additive: the decode map still answers as before."""
    assert SA.family_of("fp4_gemm_kernel__kernel", "", None) == "moe.routed.gemm"
    assert SA.family_of("t9b_mxfp8_gemv", "", "moe") == "moe.shared"
    assert SA.family_of("t9b_mxfp8_gemv", "", "attn") == "dense.proj"
    assert SA.family_of("kernel", "", None) == "unmapped"


# ------------------------------------------------------------------------------------------------ innermost sweep
def test_innermost_nested_ranges():
    rs = [(0, 100, "moe"), (10, 20, "gate"), (30, 60, "expert.hot"), (40, 50, "inner"), (200, 300, "attn")]
    ts = [5, 15, 35, 45, 55, 70, 150, 250, 300, 301]
    got = [r[2] if r else None for r in PA.innermost(rs, ts)]
    assert got == ["moe", "gate", "expert.hot", "inner", "expert.hot", "moe", None, "attn", "attn", None]


def test_innermost_matches_brute_force():
    import random

    rnd = random.Random(7)
    rs, t = [], 0
    for _ in range(40):                       # nested trees: a parent with 0-3 children, each with 0-2 children
        s = t
        kids = []
        c = s + 1
        for _ in range(rnd.randint(0, 3)):
            ks, ke = c, c + rnd.randint(2, 9)
            kids.append((ks, ke, f"k{ks}"))
            c2 = ks + 1
            for _ in range(rnd.randint(0, 2)):
                if c2 + 1 < ke:
                    g = (c2, min(ke - 1, c2 + rnd.randint(0, 2)), f"g{c2}")
                    kids.append(g)
                    c2 = g[1] + 1
            c = ke + 1
        rs.append((s, c, f"p{s}"))
        rs.extend(kids)
        t = c + rnd.randint(0, 3)
    ts = [rnd.uniform(0, t) for _ in range(500)]

    def brute(x):
        best = None
        for r in rs:
            if r[0] <= x <= r[1] and (best is None or r[0] > best[0] or (r[0] == best[0] and r[1] < best[1])):
                best = r
        return best

    assert PA.innermost(rs, ts) == [brute(x) for x in ts]


# ------------------------------------------------------------------------------------------------ sqlite analyzer
def _fake_nsys(path):
    db = sqlite3.connect(path)
    db.executescript("""
        create table StringIds (id integer primary key, value text);
        create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, correlationId int, shortName int,
                                                 demangledName int, streamId int);
        create table CUPTI_ACTIVITY_KIND_RUNTIME (start int, end int, correlationId int, nameId int);
        create table NVTX_EVENTS (start int, end int, text text, textId int, globalTid int);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, bytes int, copyKind int);
    """)
    sid = {}

    def s(v):
        if v not in sid:
            sid[v] = len(sid) + 1
            db.execute("insert into StringIds values (?, ?)", (sid[v], v))
        return sid[v]

    nv = [(0, 100000, "prefill"), (1000, 50000, "L0"), (2000, 10000, "attn"), (11000, 49000, "moe"),
          (11500, 13000, "gate"), (14000, 20000, "expert.hot"), (21000, 30000, "expert.cold"),
          (31000, 35000, "shared"), (60000, 90000, "L1"), (61000, 89000, "moe"), (62000, 70000, "expert.hot"),
          (100100, 105000, "seed_draft")]
    for a, b, t in nv:
        db.execute("insert into NVTX_EVENTS values (?, ?, ?, null, 1)", (a, b, t))
    db.execute("insert into NVTX_EVENTS values (0, 10, 'other-thread', null, 2)")
    # (launch time, kernel start, duration, name)
    ks = [(2100, 2500, 55000 // 100, "fp8_gemm_kernel__kernel"),
          (3000, 3600, 750, "sparse_attn_kernel__kernel"),
          (11600, 11700, 200, "nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"),
          (13500, 13600, 30, "DeviceSelectSweepKernel"),
          (14100, 14200, 26, "act_quant_kernel__kernel"),
          (14300, 14400, 800, "fp4_gemm_kernel__kernel"),
          (14500, 15300, 10, "vectorized_elementwise_kernel"),
          (21100, 21200, 26, "act_quant_kernel__kernel"),
          (21300, 21400, 920, "fp4_gemm_kernel__kernel"),
          (31100, 31200, 550, "fp8_gemm_kernel__kernel"),
          (36000, 36100, 40, "indexing_backward_kernel"),
          (55000, 55100, 20, "hc_split_sinkhorn_kernel__kernel"),
          (62100, 62200, 780, "fp4_gemm_kernel__kernel"),
          (100200, 100300, 50, "fp8_gemm_kernel__kernel")]
    for cid, (lt, st, du, nm) in enumerate(ks, 1):
        db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, 7)",
                   (st, st + du, cid, s(nm), s(nm)))
        db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, ?)",
                   (lt, lt + 5, cid, s("cuLaunchKernel" if "kernel__kernel" in nm else "cudaLaunchKernel_v7000")))
    db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (13700, 13950, 900, ?)",
               (s("cudaStreamSynchronize_v3020"),))
    db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (40000, 40400, 901, ?)",
               (s("cudaStreamSynchronize_v3020"),))
    db.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (13650, 13660, 8, 2)")
    db.commit()
    db.close()
    return ks


def test_analyze_synthetic_profile(tmp_path):
    p = str(tmp_path / "x.sqlite")
    _fake_nsys(p)
    rep = PA.analyze(p, log=lambda *a: None)
    assert rep["roles_known"]
    pf = rep["phases"]["prefill"]
    assert pf["kernels"] == 13 and rep["phases"]["seed_draft"]["kernels"] == 1
    fam = pf["families"]
    assert fam["moe.routed.gemm"]["launches"] == 3
    assert fam["moe.routed.quant"]["launches"] == 2
    assert fam["moe.routed.glue"]["launches"] == 1
    assert fam["moe.routed.index"]["launches"] == 1          # the where()/nonzero compaction in the moe range
    assert fam["moe.routed.combine"]["launches"] == 1
    assert fam["moe.router"]["launches"] == 1 and fam["moe.shared"]["launches"] == 1
    assert fam["dense.proj"]["launches"] == 1 and fam["attn.sparse"]["launches"] == 1
    assert fam["glue.mhc"]["launches"] == 1
    assert pf["sync_calls"] == 2 and abs(pf["sync_blocked_ms"] - (250 + 400) / 1e6) < 1e-12
    assert pf["launch_calls"] == 13
    assert pf["memcpy"]["DtoH"]["bytes"] == 8
    assert pf["experts_per_layer"] == {"L0": {"hot": 1, "cold": 1}, "L1": {"hot": 1, "cold": 0}}
    assert pf["expert_calls"] == 3
    hot, cold = pf["expert.hot.per_call"], pf["expert.cold.per_call"]
    assert hot["n"] == 2 and cold["n"] == 1
    assert abs(cold["fp4_us_per_launch"] - 0.92) < 1e-9 and abs(hot["fp4_us_per_launch"] - 0.79) < 1e-9
    assert abs(cold["wall_us"] - 9.0) < 1e-9
    assert pf["role_ranges"]["moe"]["n"] == 2
    excl = pf["role_ranges"]["moe"]["exclusive_wall_ms"]
    assert abs(excl - ((38000 + 28000) - (1500 + 6000 + 9000 + 4000 + 8000)) / 1e6) < 1e-12
    assert 0 < pf["gpu_busy_share"] < 1 and pf["wall_ms"] == 0.1
    lines = PA.analyze_lines(rep)
    assert any("expert.cold per call" in l for l in lines)


def test_analyze_refuses_without_prefill_range(tmp_path):
    p = str(tmp_path / "y.sqlite")
    db = sqlite3.connect(p)
    db.executescript("create table StringIds (id integer primary key, value text);"
                     "create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, correlationId int, shortName int,"
                     " demangledName int);")
    db.close()
    assert "error" in PA.analyze(p, log=lambda *a: None)


# ------------------------------------------------------------------------------------------------ the model
def test_expected_touched_uniform_closed_form():
    E, k = 384, 6
    for L in (1, 42, 4096):
        t = sum(PA.expected_touched(L, k, [1.0 / E] * E))
        q = 1 - (1 - 1 / E) ** k
        assert abs(t - E * (1 - (1 - q) ** L)) < 1e-9
        assert t <= min(E, L * k) + 1e-9
    a, b = (sum(PA.expected_touched(L, k, [1.0 / E] * E)) for L in (31, 54))
    assert a < b


def test_census_model_bounds_and_split():
    P = PA.model_params()
    hot = {l: set(range(301)) for l in range(40)}
    c = PA.expert_census_model(42, P, None, hot)
    assert c["calls"] <= c["bound_pairs"] and c["calls_per_layer_max"] <= 384
    assert 0 < c["cold_calls"] < c["calls"]
    assert abs(c["cold_bytes"] - c["cold_calls"] * P["expert_bytes"]["v"]) < 1e-3
    assert P["expert_bytes"]["v"] == 3 * 2304 * 5120 // 2 + 3 * 2304 * 5120 // 32


def test_model_params_read_receipts():
    P = PA.model_params()
    assert abs(P["fp4_us"]["v"] - 78.1) < 0.1          # step-anatomy-m0: 56.256 ms / 720 launches
    assert abs(P["c2c_extra_us"]["v"] - 11.5) < 0.05     # c2c probe: 102.8 - 91.3
    assert P["m0_moe_launches"]["v"] == pytest.approx(14480.0)
    assert all(p["label"] in ("D", "A", "E") for p in P.values())
    d = PA.derived(P)
    assert 0.5 < d["nofill_host_factor"] < 1.0
    assert d["wall_us_per_expert_call"] >= d["gpu_us_per_expert_call"]


def test_ttft_model_ordering_and_labels():
    P = PA.model_params()
    hot = {l: set(range(301)) for l in range(40)}
    for L in (42, 4096):
        cen = PA.expert_census_model(L, P, None, hot)
        t = PA.ttft_model(L, cen, P)
        assert t["today"]["ms"] > t["fills_off"]["ms"] > t["B_sliced_moe"]["ms"] > t["C_grouped_ref_kernel"]["ms"]
        assert t["E_production_class"]["ms"] < t["C_grouped_ref_kernel"]["ms"]
        assert t["A_prestage_cold"]["ms"] <= t["today"]["ms"]
        assert t["D_chunked_rows6"]["chunks"] == math.ceil((L - 1) / 6)
        for k in ("fills_off", "A_prestage_cold", "B_sliced_moe", "C_grouped_ref_kernel"):
            assert "bit-identical" in t[k]["numerics"]
        for k in ("D_chunked_rows6", "E_production_class"):
            assert t[k]["numerics"].startswith("A3")
    big = PA.ttft_model(32768, PA.expert_census_model(32768, P, None, hot), P)
    assert not big["today"]["fits_hbm"] and big["today"]["max_single_shot_L"] < 32768


@pytest.mark.skipif(not os.path.exists(PA.DEFAULT_COUNTS), reason="decode routing counts not on this machine")
def test_model_calibration_against_the_headline_receipt():
    """Regression of the calibration, not a measurement: with the decode counts and the hot-301 map, the modeled
    reference prefill at the fixture lengths stays within 15 % of the headline receipt's measured prefill_ms."""
    meas = PA.measured_prefill_by_L()
    rep = PA.model_report(sorted(meas), PA.DEFAULT_COUNTS, PA.DEFAULT_MAP, meas)
    for r in rep["rows"]:
        assert abs(r["ttft"]["today"]["ms"] - r["measured_prefill_ms"]) / r["measured_prefill_ms"] < 0.15
    lines = PA.model_lines(rep)
    assert any("independence" in l for l in lines)


def test_box_lines():
    lines = PA.box_lines("prose")
    txt = "\n".join(lines)
    assert "--network none" in txt and "--capture-range=cudaProfilerApi" in txt
    assert "bench/prefill_anatomy.py run" in txt and "bench/prefill_anatomy.py analyze" in txt
    assert "--json /work/scratch/" in txt and "/work/docs" not in txt
    assert "refusing" in txt
    assert PA.main(["box"]) == 0


# ------------------------------------------------------------------------------------------------ emulated model
@pytest.fixture
def mini(monkeypatch, tmp_path):
    """-> build(kind) -> (model, pool) on the CPU kernel emulations (tests/test_static_emulated_cpu.py)."""
    from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference
    from test_static_emulated_cpu import KERNELS, _mini4

    R = import_reference(cpu_stubs=True)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: ([i % 97 for i in range(MINI_CONFIG["vocab_size"])],
                                     MINI_CONFIG["engram_compressed_vocab_size"]))
    for k, f in KERNELS.items():
        monkeypatch.setattr(R, k, f)
    prev = torch.get_default_dtype()

    def build(kind="mini4"):
        if kind == "mini4":
            cfg, mp = _mini4()
            ck = make_mini_checkpoint(str(tmp_path / kind), cfg=cfg)
        else:
            ck = make_mini_checkpoint(str(tmp_path / kind))
            mp = ck["map"]
        model = build_and_load(ck["model_dir"], mp, "cpu", tokenizer=object(), cpu_stubs=True, io_threads=2,
                               log=lambda *a: None)
        model.temperature = 0.0
        return model, model.runtime_load.pool

    yield build
    torch.set_default_dtype(prev)


def _ids(n, vocab=2048, seed=3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [int(x) for x in torch.randint(3, vocab - 64, (n,), generator=g, device="cpu")]


def _hooks_left(model) -> int:
    n = 0
    for m in model.modules():
        n += len(m._forward_hooks) + len(m._forward_pre_hooks)
    return n


def test_probe_census_on_emulated_mini(mini):
    model, pool = mini("mini4")
    ids = _ids(19)
    base_hooks = _hooks_left(model)
    plain = PA.prefill_once(model, ids, (), seed=False)
    nl = len(model.layers)
    probe = PA.PrefillProbe(model, pool, capture_moe=range(nl))
    with probe:
        hk = PA.prefill_once(model, ids, (), seed=False)
    probe.finalize()
    assert _hooks_left(model) == base_hooks                    # every hook removed
    assert hk["logits_sha"] == plain["logits_sha"] and hk["first"] == plain["first"]   # hooks do not perturb
    k = int(model.layers[0].ffn.n_activated_experts)
    c = probe.census(len(ids), k)
    assert c["rows"] == len(ids) * k * nl == c["routed_pairs"]
    for l in range(nl):                                        # calls = distinct routed experts, split by pool tier
        moe = model.layers[l].ffn
        x = probe.moe_inputs[l]
        with torch.inference_mode():
            _, idx = moe.gate(x.view(-1, moe.dim), None)
        used = set(int(e) for e in idx.flatten())
        assert c["calls_per_layer"][l] == len(used)
        assert c["cold_per_layer"][l] == len(used & set(pool.cold[l]))
        assert sorted(c["cold_ids"].get(l, [])) == sorted(used & set(pool.cold[l]))
    assert c["cold_bytes"] == c["cold_calls"] * pool.expert_bytes
    roles = probe.roles()
    assert roles["expert.hot"]["n"] + roles.get("expert.cold", {"n": 0})["n"] == c["expert_calls"]
    assert roles["moe"]["n"] == nl and roles["L*"]["n"] == nl and roles["head"]["n"] == 1
    assert "exclusive_host_ms" in roles["moe"]
    assert all(r["gpu_ms"] is None for r in probe.records)     # no CUDA events on the CPU


@pytest.mark.parametrize("combine", ["iadd", "index_add"])
def test_sliced_moe_bit_equal_reference_emulated(mini, combine):
    """Stage B of the design: one host sync per layer instead of one per expert, same Expert calls on the same rows
    in the same order, same fp32 ascending-id accumulation -> torch.equal with the reference MoE.forward."""
    model, pool = mini("mini4")
    ids = _ids(23, seed=11)
    nl = len(model.layers)
    probe = PA.PrefillProbe(model, pool, capture_moe=range(nl))
    with probe:
        PA.prefill_once(model, ids, (), seed=False)
    with torch.inference_mode():
        for l in range(nl):
            moe, x = model.layers[l].ffn, probe.moe_inputs[l]
            assert torch.equal(moe(x), PA.sliced_moe_forward(moe, x, combine))


def test_moe_probe_report_on_emulated_mini(mini):
    model, pool = mini("mini4")
    probe = PA.PrefillProbe(model, pool, capture_moe=[0, 1])
    with probe:
        PA.prefill_once(model, _ids(17, seed=5), (), seed=False)
    recs = PA.moe_probe(model, probe.moe_inputs, reps=1, log=lambda *a: None)
    assert [r["layer"] for r in recs] == [0, 1]
    assert all(r["equal_iadd"] and r["equal_index_add"] for r in recs)
    assert all(r["reference_ms"] > 0 for r in recs)


def test_sliced_moe_refuses_unknown_combine(mini):
    model, _ = mini("mini4")
    moe = model.layers[0].ffn
    with pytest.raises(ValueError):
        PA.sliced_moe_forward(moe, torch.zeros(1, 2, moe.dim, device="cpu"), "atomic")


def test_run_main_smoke_on_emulated_mini(mini, monkeypatch, tmp_path):
    """The box path end to end on the CPU (boot / tokenizer / determinism stubbed): timed reps, fills off + equality,
    hooked census, the MoE probe, the receipt. Catches plumbing errors before a box window does."""
    import json

    import runtime.generate
    import runtime.loader

    model, pool = mini("mini4")
    monkeypatch.setattr(runtime.loader, "build_and_load", lambda *a, **k: model)
    monkeypatch.setattr(runtime.generate, "set_determinism", lambda *a, **k: {"stub": True})

    class FakeGen:
        def __init__(self, *a, **k):
            pass

        def encode(self, msgs):
            return _ids(11 + len(msgs[0]["content"]) % 7, seed=len(msgs[0]["content"]))

    monkeypatch.setattr(runtime.generate, "Generator", FakeGen)
    fx = tmp_path / "fx.jsonl"
    fx.write_text("\n".join(json.dumps({"tag": t, "prompt": p}) for t, p in
                            [("prose", "write a poem"), ("code", "a function"), ("skip", "x")]) + "\n")
    out = tmp_path / "r.json"
    rc = PA.main(["run", "--model-dir", str(tmp_path), "--map", "", "--fixture", str(fx), "--only", "prose,code",
                  "--reps", "2", "--fills", "both", "--probe-moe", "0,1", "--probe-reps", "1", "--device", "cpu",
                  "--json", str(out)])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["errors"] == []
    assert sorted(rep["prompts"]) == ["code", "prose"]
    for t, P in rep["prompts"].items():
        assert len(P["fills_on"]) == 2 and len(P["fills_off"]) == 2
        assert P["fills_off_bit_equal"] is True and P["self_replay_equal"] is True
        assert P["hooked"]["equal_untraced"] is True
        assert P["hooked"]["census"]["rows"] == P["L"] * 4 * len(model.layers)
        assert all(r["equal_iadd"] and r["equal_index_add"] for r in P["moe_probe"])
