"""Mac CPU tests of T12-DIAG (`.venv/bin/python -m pytest -q tests/test_row_divergence_cpu.py`): the probe's plumbing
(bench/row_divergence_probe.py), the FlashMLA row split (runtime/attn_flashmla.py), and a dry run of
tests/gpu/test_row_divergence_gpu.py on the CPU kernel emulations (DEV "cpu"), so a box window meets the kernels,
not harness bugs. Kernel fidelity (cuBLAS, FlashMLA, DeepGEMM) is the box's job.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from test_attn_flashmla_cpu import FakeOps
from test_static_rows_cpu import _cpu_default_device, emu, rows_dg, rows_glue  # noqa: F401 (fixtures)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("row_divergence_probe_t", os.path.join("bench", "row_divergence_probe.py"))


# ------------------------------------------------------------------------------------------------ pure helpers
def test_take_row_and_diff():
    t = torch.arange(2 * 6 * 3, dtype=torch.float32, device="cpu").view(1, 6, 2, 3)
    assert torch.equal(P.take_row(t, 6, 4), t[:, 4])
    u = torch.arange(6 * 5, device="cpu").view(6, 5)
    assert torch.equal(P.take_row(u, 6, 2), u[2])
    assert P.take_row(torch.zeros(1, 3, 4, device="cpu"), 6, 0) is None
    a = torch.tensor([1.0, -2.0, 0.0], device="cpu")
    b = a.clone()
    b.view(torch.int32)[1] += 3                      # 3 ulp in magnitude
    d = P.diff(a, b)
    assert not d["equal"] and d["mism"] == 1 and d["max_ulp"] == 3
    assert P.diff(a, a.clone())["equal"]
    z = P.diff(torch.tensor([0.0], device="cpu"), torch.tensor([-0.0], device="cpu"))
    assert z["equal"], z                             # +0 == -0 on the ordinal line (bits differ, value equal)
    i = P.diff(torch.tensor([1, 2, 3], device="cpu"), torch.tensor([1, 2, 4], device="cpu"))
    assert i["mism"] == 1 and i["only_6row"] == 1 and i["only_1row"] == 1
    hb = torch.tensor([1.5], dtype=torch.bfloat16, device="cpu")
    hb2 = hb.clone()
    hb2.view(torch.int16)[0] += 1
    assert P.diff(hb, hb2)["max_ulp"] == 1
    assert P.family("linear:layers.12.ffn.gate.weight") == "linear:ffn.gate.weight"


def test_kv_checksums_follow_referenced_bytes_only():
    g = torch.Generator(device="cpu").manual_seed(0)
    kv = torch.randn(1, 40, 8, generator=g, device="cpu").to(torch.bfloat16)
    idx = torch.tensor([[[3, 7, 9, -1], [1, 2, 30, 31]]], dtype=torch.int32, device="cpu")
    tl = torch.tensor([3, 3], dtype=torch.int32, device="cpu")
    c0 = P.kv_checksums(kv, idx, tl)
    assert c0.shape == (2, 2) and c0.dtype == torch.int64
    kv2 = kv.clone()
    kv2[0, 31, 0] = kv2[0, 31, 0] + 1                # past row 1's length: not read
    kv2[0, 20, 3] = 7.0                              # referenced by nobody
    assert torch.equal(P.kv_checksums(kv2, idx, tl), c0)
    kv2[0, 9].view(torch.int16)[5] += 1              # one bit in a row row 0 reads
    c2 = P.kv_checksums(kv2, idx, tl)
    assert not torch.equal(c2[0], c0[0]) and torch.equal(c2[1], c0[1])
    # one row alone == its row of the batched checksum (what the probe compares)
    assert torch.equal(P.kv_checksums(kv, idx[:, 1:2], tl[1:2])[0], c0[1])
    assert torch.equal(P.kv_checksums(kv, idx, None)[1], P.kv_checksums(kv, idx[:, 1:2], None)[0])


def test_probe_patches_rowwise_and_restore():
    lin0, ein0 = F.linear, torch.einsum
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(1, 6, 64, generator=g, device="cpu")
    w = torch.randn(32, 64, generator=g, device="cpu")
    per_row = torch.cat([lin0(x[0, i:i + 1], w) for i in range(6)], 0).view(1, 6, 32)

    class Fake:                                       # an active Taps stand-in (rowwise only inside a step)
        recs = None

        def wname(self, w):
            return "w"

        def rec(self, site, t):
            Fake.seen.append(site)
            return t

    Fake.seen = []
    with P.probe_patches("fp32"):
        assert torch.equal(F.linear(x, w), lin0(x, w))   # no step recording: untouched
        P._ACTIVE[0] = Fake()
        try:
            assert torch.equal(F.linear(x, w), per_row)
            xb, wb = x.to(torch.bfloat16), w.to(torch.bfloat16)
            assert torch.equal(F.linear(xb, wb), lin0(xb, wb))   # fp32 mode leaves bf16 batched
            torch.einsum("bsd,nd->bsn", x, w)
        finally:
            P._ACTIVE[0] = None
    assert F.linear is lin0 and torch.einsum is ein0
    assert Fake.seen == ["linear:w", "linear:w", "einsum:bsd,nd->bsn"]
    with pytest.raises(ValueError):
        with P.probe_patches("bogus"):
            pass


def test_flashmla_row_split_calls_and_values(monkeypatch):
    from runtime import attn_flashmla as AFM

    g = torch.Generator(device="cpu").manual_seed(2)
    rows, h, d, n, w = 6, 16, 64, 300, 160
    q = (torch.randn(1, rows, h, d, generator=g, device="cpu")).to(torch.bfloat16)
    kv = (torch.randn(1, n, d, generator=g, device="cpu")).to(torch.bfloat16)
    sink = torch.randn(h, generator=g, device="cpu")
    idx = torch.randint(0, n, (1, rows, w), generator=g, device="cpu").int()
    idx[..., -30:] = -1
    tl = torch.full((rows,), w - 30, dtype=torch.int32, device="cpu")
    ops = FakeOps()
    one = AFM.sparse_attn_flashmla(q, kv, sink, idx, 0.1, tl, ops=ops, row_split=False)
    assert [c["q"][0] for c in ops.calls] == [rows]
    ops.calls.clear()
    monkeypatch.setenv(AFM.ROW_SPLIT_ENV, "1")
    assert AFM.use_row_split()
    split = AFM.sparse_attn_flashmla(q, kv, sink, idx, 0.1, tl, ops=ops)
    assert [c["q"][0] for c in ops.calls] == [1] * rows and [c["tlen"] for c in ops.calls] == [[w - 30]] * rows
    assert split.shape == one.shape and split.is_contiguous()
    assert torch.equal(split, one)                    # the emulation is per row: same bits either way
    ops.calls.clear()
    AFM.sparse_attn_flashmla(q[:, :1].contiguous(), kv, sink, idx[:, :1].contiguous(), 0.1, tl[:1], ops=ops)
    assert [c["q"][0] for c in ops.calls] == [1]      # a 1-row call is one call either way
    monkeypatch.setenv(AFM.ROW_SPLIT_ENV, "0")
    assert not AFM.use_row_split()


def test_real_width_census_runs_on_cpu():
    recs = P.real_width_census("cpu", W=(256,), flashmla=False)
    ops = [r["op"] for r in recs]
    assert any("router gate" in o for o in ops) and any("compressor wkv / wgate" in o for o in ops)
    assert any("indexer bshd,btd->bsht" in o for o in ops) and any("top-512" in o for o in ops)
    assert all("equal" in r for r in recs), [r for r in recs if "equal" not in r]
    assert not [r for r in recs if r.get("equal") is None], [r for r in recs if r.get("equal") is None]


def test_parser_defaults():
    a = P.build_parser().parse_args([])
    assert a.kind == "mini4" and a.impls == "headline" and a.rowwise_linear == "off" and a.fp32_impl == "fixed"
    a = P.build_parser().parse_args(["--plan", "5,0,2", "--rowwise-linear", "fp32", "--flashmla-row-split",
                                     "--router-impl", "plan-gemv", "--reject-variants", "2"])
    assert a.plan == "5,0,2" and a.flashmla_row_split and a.router_impl == "plan-gemv" and a.reject_variants == 2


def test_plan_from_headline_receipt():
    path = os.path.join(REPO, "docs", "headline-20261107i-plan-gemv.json")
    if not os.path.exists(path):
        pytest.skip("receipt not in this tree")
    plan, tag = P.plan_from(path + ":shell")
    assert tag == "shell" and len(plan) == 69 and all(0 <= n <= 5 for n in plan)
    # the loop's step 46 starts at P0 = L + 166 and emits tokens 167..172 (the divergence at 171 is its row 4)
    n_out = 1
    for n in plan[:46]:
        n_out += n + 1
    assert n_out == 167 and plan[46] == 5
    with pytest.raises(ValueError):
        P.plan_from(path + ":nope")


def test_variant_leaks():
    r = lambda s, j, h: {"step": s, "row": j, "pos": 10 + j, "logits_sha": h}   # noqa: E731
    a = {"rows": [r(0, 0, "x"), r(0, 1, "y")]}
    assert P.variant_leaks([a, {"rows": [r(0, 0, "x"), r(0, 1, "y")]}]) == []
    assert P.variant_leaks([a, {"rows": [r(0, 0, "x"), r(0, 1, "z")]}]) == [[0, 1, 11]]


# ------------------------------------------------------------------------------------------------ the probe, emulated
RIM = dict(rowwise="all", row_invariant=True)   # the CPU's own F.linear / einsum are not row-invariant either


def test_probe_teacher_forced_and_rejections_equal_emulated(emu):
    model, pool, _ = emu("mini4")
    for plan in ((5, 5, 5), (5, 0, 2, 5, 1, 3, 0, 4)):
        rep = P.run_probe(model, pool, impls={}, plan=plan, L=18, static_kw={"_allow_cpu": True}, log=print, **RIM)
        for line in P.report_lines(rep):
            print(line)
        s = rep["summary"]
        assert s["rows_compared"] == sum(n + 1 for n in plan)
        assert s["origin"] is None, s["origin"]
        assert rep["taps_compared_first_row"] >= 8 * len(model.layers), rep["taps_compared_first_row"]
        sites = set(rep["rows"][0]["sites"])
        want = {"block.out", "attn.out", "attn.sparse", "attn.q", "attn.idx", "attn.kv", "ffn.out", "cache.ckv",
                "cache.idxk", "indexer.topk", "head.logits", "engram.out", "einsum:bsgd,grd->bsgr",
                "einsum:bshd,btd->bsht"}
        assert want <= sites, want - sites
        assert any(s.startswith("linear:") and "gate" in s for s in sites), sites      # the router gate's mm
        assert any(s.startswith("linear:") and "compressor.w" in s for s in sites), sites


def test_probe_reports_the_injected_tap_emulated(emu):
    model, pool, _ = emu("mini4")
    rep = P.run_probe(model, pool, impls={}, plan=(5,), L=18, static_kw={"_allow_cpu": True},
                      inject=(1, "ffn.out", 2), log=print, **RIM)
    o = rep["summary"]["origin"]
    assert o is not None and (o["step"], o["row"]) == (0, 2), o
    assert o["first"]["key"] == [1, "ffn.out", 0] and o["first"]["max_ulp"] == 1 and o["first"]["mism"] == 1, o
    assert all(r["first"] is None and r["logits_equal"] for r in rep["rows"] if r["row"] < 2)
    later = [r for r in rep["rows"] if r["row"] > 2 and r["first"] is not None]
    # later rows read row 2's KV from layer 2 on (when the one-ulp move survives the bf16 roundings at all)
    assert all(r["first"]["key"][0] >= 2 for r in later), [r["first"] for r in later]


def test_probe_greedy_stream_emulated(emu):
    model, pool, _ = emu("mini")
    rep = P.run_probe(model, pool, impls={}, plan=(2, 5), L=17, stream="greedy", static_kw={"_allow_cpu": True},
                      log=print, **RIM)
    assert rep["summary"]["origin"] is None and rep["summary"]["rows_compared"] == 9


# ------------------------------------------------------------------------------------------------ GPU test dry run
def _gpu_module(monkeypatch, builder):
    G = _load("gpu_rowdiv", os.path.join("tests", "gpu", "test_row_divergence_gpu.py"))

    @contextlib.contextmanager
    def model_cpu(kind="mini4"):
        yield builder(kind)

    monkeypatch.setattr(G, "DEV", "cpu")
    monkeypatch.setattr(G, "STATIC_KW", {"_allow_cpu": True})
    monkeypatch.setattr(G, "_need_gpu", lambda: None)
    monkeypatch.setattr(G, "_model", model_cpu)
    return G


def _rim(G, monkeypatch, seen=None):
    """The CPU's own F.linear / einsum are not row-invariant: every dry-run probe runs row-wise (rowwise all +
    row_invariant_mode), PLAIN and FIX alike -- the dry run proves the harness, the box proves the kernels."""
    orig = G.P.run_probe

    def run(*a, **k):
        rep = orig(*a, **dict(k, rowwise="all", row_invariant=True))
        if seen is not None:
            seen.setdefault("glue", rep["glue_on"])
            seen.setdefault("sites", set()).update(rep["rows"][0]["sites"])
        return rep

    monkeypatch.setattr(G.P, "run_probe", run)


def test_gpu_test_dry_run_on_cpu(emu, monkeypatch, capsys):
    """The GPU test's gated functions on the emulation with the reference impls (no FlashMLA / DeepGEMM on the
    Mac)."""
    G = _gpu_module(monkeypatch, lambda kind: emu(kind)[:2])
    monkeypatch.setattr(G, "IMPLS", {})
    _rim(G, monkeypatch)
    G.test_rowdiv_mini4_teacher_forced()
    G.test_rowdiv_mini4_rejections()
    G.test_rowdiv_injection_is_named()
    out = capsys.readouterr().out
    assert "plain: all equal; fix: all equal" in out, out
    assert "accepted rows all equal to the 1-row step; no rejected-row leak" in out, out
    assert "reported as [1, 'ffn.out', 0] at step 0 row 2" in out, out
    assert G.COUNTS["gates"] == 5 and G.COUNTS["rows"] == 2 * 24 + 28, G.COUNTS
    print(out)


def test_gpu_test_dry_run_dg_inplace_fused_on_cpu(rows_glue, monkeypatch, capsys):
    """The same gated runs on the headline-like impls the emulation has (dg-inplace + fused glue, fp8 ref, attn ref;
    the router `fused` here so its route / route_rows taps run): router.ids / router.w and the R-row dg-inplace MoE
    go through the probe."""
    G = _gpu_module(monkeypatch, lambda kind: rows_glue(kind)[:2])
    monkeypatch.setattr(G, "IMPLS", dict(moe_impl="dg-inplace", glue_impl="fused"))
    monkeypatch.setattr(G, "ENV", dict(G.ENV, DSV41_ROUTER_IMPL="fused"))
    monkeypatch.setattr(G, "PLAN_REJ", (5, 0, 2, 5))
    seen: dict = {}
    _rim(G, monkeypatch, seen)
    G.test_rowdiv_mini4_teacher_forced()
    G.test_rowdiv_mini4_rejections()
    out = capsys.readouterr().out
    assert "router" in (seen["glue"]["6"] or []) and "router" in (seen["glue"]["1"] or []), seen["glue"]
    assert {"router.ids", "router.w", "cache.ckv"} <= seen["sites"], seen["sites"]
    assert G.COUNTS["gates"] == 4, G.COUNTS
    print(out)


def test_probe_without_fix_is_reported_emulated(emu, capsys):
    """No fix on the emulation: whatever the CPU's own batched F.linear / einsum / reductions do at 6 rows is
    reported as a first (layer, site) with ulp -- the report is well formed either way (no gate on the values)."""
    model, pool, _ = emu("mini4")
    rep = P.run_probe(model, pool, impls={}, plan=(5, 1), L=18, static_kw={"_allow_cpu": True}, log=print)
    lines = P.report_lines(rep)
    print("\n".join(lines))
    o = rep["summary"]["origin"]
    assert o is None or (o["first"] is None or isinstance(o["first"]["key"][1], str))
    assert lines[2].startswith("  first divergence: ")
