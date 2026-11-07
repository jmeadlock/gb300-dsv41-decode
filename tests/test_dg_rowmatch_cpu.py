"""Mac CPU tests of T9a-3 (docs/HANDOFF-T9A3-DG-ROWMATCH.md): the 1-row dg-inplace routed GEMM in the 6-row step's call
shape (`--moe-rowmatch`, runtime/dg_inplace.py rowmatch_plan / build_one_row / check_rowmatch, the padded A blocks of
DGInplaceBuffers(gemm_blocks=...)), the spike's pure parts (bench/dg_rowmatch_spike.py) and the bench flags.

What a CPU can check, and does: the plan per mode / env / call shape; the block layouts of the 1-row and R-row calls
(runtime/moe_rows.py's ordering); the packed activation SF from fp32 scales == DeepGEMM's packing; run_contiguous /
the spike's Contig on the contract-checking fake contiguous GEMM (tests/test_dg_inplace_cpu.py: per-row math, -1
rows NaN-poisoned) -> padded and unpadded shapes per-row equal; the build check passes on the faithful fake and FALLS
BACK (T9a-2's call, the reason recorded and printed) on a fake whose rows depend on their block's occupancy; the
emulated 1-row step in each mode == the reference forward lives in tests/test_dg_inplace_cpu.py (parametrized
rowmatch off / on / align). Box only (tests/gpu/test_dg_rowmatch_gpu.py): DeepGEMM's real per-row behaviour, timing.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

from runtime import dg_inplace as DI
from test_dg_inplace_cpu import _fake_dg_contig, _install_kernels, di_emu  # noqa: F401 (fixture)
from test_static_emulated_cpu import emu  # noqa: F401 (fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    torch.set_default_device("cpu")
    for v in (DI.ROWMATCH_ENV, DI.ROWMATCH_ROWS_ENV, DI.ROWS_GEMM_ENV, DI.ALIGN_ENV, DI.SF_STORE_ENV,
              DI.LAYOUT_CHECK_ENV):
        monkeypatch.delenv(v, raising=False)
    yield
    torch.set_default_device("cpu")


def _bench(name):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", os.path.join(REPO, "bench", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ the plan
def test_rowmatch_plan_modes_env_and_shapes(monkeypatch):
    p = DI.rowmatch_plan(6)
    assert p["mode"] == "on" == DI.ROWMATCH_DEFAULT and p["on"] and p["rows"] == 6 and p["rows_gemm"] == "contiguous16"
    assert p["align"] == 16 and p["blocks"] == 36                     # the 6-row call: ALIGN 16, R*k = 36 blocks
    a = DI.rowmatch_plan(6, "align")
    assert a["on"] and a["align"] == 16 and a["blocks"] == 6
    o = DI.rowmatch_plan(6, "off")
    assert not o["on"] and o["align"] is None and o["blocks"] is None and "off" in o["why"]
    monkeypatch.setenv(DI.ROWMATCH_ENV, "align")
    monkeypatch.setenv(DI.ROWMATCH_ROWS_ENV, "4")
    monkeypatch.setenv(DI.ROWS_GEMM_ENV, "contiguous32")
    e = DI.rowmatch_plan(2)
    assert e["mode"] == "align" and e["rows"] == 4 and e["align"] == 32 and e["blocks"] == 2
    assert DI.rowmatch_plan(2, "on")["blocks"] == 8
    monkeypatch.setenv(DI.ROWS_GEMM_ENV, "contiguous")           # ALIGN from $DSV41_DG_ALIGN or 128, as the R-row build
    monkeypatch.setenv(DI.ALIGN_ENV, "64")
    assert DI.rowmatch_plan(6, "on")["align"] == 64
    for spec in ("psum", "masked"):                               # no 1-row twin
        r = DI.rowmatch_plan(6, "on", rows_spec=spec)
        assert not r["on"] and "not contiguous" in r["why"]
    r = DI.rowmatch_plan(6, "on", rows=8, rows_spec="contiguous16")
    assert r["on"] and r["blocks"] == 48
    r = DI.rowmatch_plan(6, "on", rows=32, rows_spec="contiguous16")
    assert not r["on"] and "outside" in r["why"]
    with pytest.raises(ValueError, match="moe rowmatch"):
        DI.rowmatch_plan(6, "maybe")
    monkeypatch.delenv(DI.ROWS_GEMM_ENV)
    assert DI.rows_gemm_spec() == DI.ROWS_GEMM_DEFAULT == "contiguous16" and DI.rows_gemm_spec("psum") == "psum"


# ------------------------------------------------------------------------------------------------ block layouts
def test_block_layouts_one_row_and_r_rows():
    rexp = [[4, 9, 1], [9, 2, 4], [7, 9, 1]]
    gmap = {1: 10, 2: 11, 4: 3, 7: 20, 9: 0}
    one = DI.one_row_blocks(1, rexp[1], gmap)
    assert one == [[((1, 2), 11)], [((1, 4), 3)], [((1, 9), 0)]]          # ascending id, one block each
    rr = DI.r_row_blocks(rexp, gmap)
    assert [[k for k, _ in b] for b in rr] == [[(0, 1), (2, 1)], [(1, 2)], [(0, 4), (1, 4)], [(2, 7)],
                                               [(0, 9), (1, 9), (2, 9)]]
    assert [b[0][1] for b in rr] == [10, 11, 3, 20, 0]
    rows, mi = DI.contiguous_blocks_layout(rr, 4, 9)
    assert len(mi) == 36 and rows[(2, 9)] == 4 * 4 + 2 and rows[(1, 4)] == 2 * 4 + 1
    assert mi[16:20] == [0, 0, 0, -1] and all(v == -1 for v in mi[20:])
    rows1, mi1 = DI.contiguous_blocks_layout(one, 16, 9)
    assert [rows1[k] for k in ((1, 2), (1, 4), (1, 9))] == [0, 16, 32] and sum(v >= 0 for v in mi1) == 3
    with pytest.raises(ValueError):
        DI.contiguous_blocks_layout(rr, 2, 9)        # a block holds 3 rows > ALIGN 2
    with pytest.raises(ValueError):
        DI.contiguous_blocks_layout(rr, 4, 4)        # 5 blocks > 4


def test_pack_act_sf_from_f32_is_deepgemm_packing():
    g = torch.Generator().manual_seed(2)
    xs = torch.exp2(torch.randint(-20, 20, (10, 16), generator=g).float())
    sf = DI.pack_act_sf_from_f32(xs)
    want = DI.packed_act_sf(10, 16, "cpu")
    pr = DI.PackedRows(want, 1, 10)
    for r in range(10):
        pr.store_codes_torch(r, ((xs[r].view(torch.int32) >> 23) & 0xFF))
    assert sf.shape == (10, 4) and sf.stride() == want.stride() and torch.equal(sf, want)
    assert torch.equal(DI.PackedRows(sf, 1, 10).codes(), pr.codes())


# ------------------------------------------------------------------------------------------------ calls on the fake GEMM
def _synthetic_pool(G=7, N=8, K=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randint(-128, 128, (G, N, K // 2), generator=g).to(torch.int8)
    codes = torch.randint(120, 134, (G, N, K // 32), generator=g).to(torch.uint8)
    SF = DI.packed_sf_zeros(G, N, K // 128, "cpu")
    SF.copy_(DI.pack_words(codes))
    return W, SF


def _acts(keys, K, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for kk in keys:
        q = (torch.randn(K, generator=g) * 2).clamp(-448, 448).to(FP8)
        s = torch.exp2(torch.randint(-3, 4, (K // 32,), generator=g).float())
        out[kk] = (q, s)
    return out


def test_run_contiguous_padded_and_r_row_shapes_equal_per_row():
    fake = _fake_dg_contig()
    W, SF = _synthetic_pool()
    rexp = [[0, 2, 5], [2, 3, 5], [1, 2, 6]]
    gmap = {e: e for e in range(7)}
    keys = [(r, e) for r, ex in enumerate(rexp) for e in ex]
    acts = _acts(keys, 256, 3)
    ref = DI.run_contiguous(fake, W, SF, DI.r_row_blocks(rexp, gmap), acts, 16, 9)
    assert fake.align[0] == 128                                     # the global left as found
    for r in range(3):
        for al, nb, pk in ((128, 3, False), (16, 3, False), (16, 9, False), (16, 9, True)):
            got = DI.run_contiguous(fake, W, SF, DI.one_row_blocks(r, rexp[r], gmap), acts, al, nb, packed_a=pk)
            for e in rexp[r]:
                assert torch.equal(got[(r, e)].view(torch.int16), ref[(r, e)].view(torch.int16)), (r, e, al, nb, pk)
    M, G, N, K, heads = fake.contig_calls[-1]
    assert M == 9 * 16 and heads[:3] == [1, 2, 6] and heads[3:] == [-1] * 6
    assert fake.packed_a[-1] == 9 * 16


def test_spike_pure_helpers():
    sp = _bench("dg_rowmatch_spike")
    hot, cold = list(range(0, 40)), list(range(40, 60))
    sets = sp.make_sets(hot, cold, 8, 6, 6, seed=1)
    assert len(sets) == 8
    for s, rexp in enumerate(sets):
        assert len(rexp) == 6 and all(len(r) == 6 == len(set(r)) and r == sorted(r) for r in rexp)
        n_cold = len({e for r in rexp for e in r if e >= 40})
        assert n_cold == 1 + s % 3                                  # every set routes >= 1 cold expert
    fake = _fake_dg_contig()
    W, SF = _synthetic_pool(G=12)
    rexp = [[0, 2, 5], [2, 3, 11]]
    gmap = {e: e for e in range(12)}
    keys = sp.keys_of(rexp)
    acts = _acts(keys, 256, 5)
    c = sp.Contig(fake, W, SF, DI.r_row_blocks(rexp, gmap), acts, 16, 6, False, torch.device("cpu"))
    c()
    a1 = c.out()
    b1 = DI.run_contiguous(fake, W, SF, DI.r_row_blocks(rexp, gmap), acts, 16, 6)
    assert all(torch.equal(a1[k], b1[k]) for k in keys)
    outs = {p: a1 for p in sp.PROJS}
    assert sp.rows_equal(outs, outs, rexp) == 2 and sp.max_ulp(outs, outs, rexp) == 0
    other = {p: dict(a1) for p in sp.PROJS}
    other["w2"][(1, 11)] = (a1[(1, 11)].view(torch.int16) + 1).view(torch.bfloat16)
    assert sp.rows_equal(outs, other, rexp) == 1 and sp.max_ulp(outs, other, rexp) == 1
    ap = sp.build_parser().parse_args([])
    assert ap.sets == 8 and ap.rows == 6 and ap.rows_align == 16 and ap.out is None   # receipt -> scratch/


def test_spike_main_dry_run_on_cpu(monkeypatch, tmp_path):
    """bench/dg_rowmatch_spike.py main() end to end on a fake checkpoint (mini shapes) and the fake deep_gemm
    (contiguous + packed-SF masked, per-row math): every arm runs, every pair is compared, cold experts staged and
    checked, the verdict written; only the timing (CUDA graphs) is an error here."""
    import json
    import types

    from test_moe_rows_t10_cpu import _psum_masked_fakes

    sp = _bench("dg_rowmatch_spike")
    monkeypatch.setattr(sp, "DIM", 256)
    monkeypatch.setattr(sp, "INTER", 128)
    monkeypatch.setattr(sp, "E", 40)
    monkeypatch.setattr(sp, "NK", {"w1": (128, 256), "w3": (128, 256), "w2": (256, 128)})

    class Ckpt:
        def __init__(self, root):
            self.root = root

        def expert(self, l, e, p):
            N, K = sp.NK[p]
            g = torch.Generator().manual_seed(1000 * e + sp.PROJS.index(p))
            return (torch.randint(-128, 128, (N, K // 2), generator=g).to(torch.int8),
                    torch.randint(120, 134, (N, K // 32), generator=g).to(torch.uint8))

    monkeypatch.setitem(sys.modules, "dg_contiguous_spike", types.SimpleNamespace(Ckpt=Ckpt, graph_us=None))
    fake = _psum_masked_fakes(_fake_dg_contig())
    fake.use_deterministic_algorithms = lambda flag=True: None
    monkeypatch.setitem(sys.modules, "deep_gemm", fake)
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"0": list(range(0, 40, 2))}))       # 20 hot, 20 cold
    out = tmp_path / "spike.json"
    rc = sp.main(["--device", "cpu", "--sets", "3", "--map", str(mp), "--out", str(out), "--ckpt", "x"])
    rep = json.loads(out.read_text())
    assert rc == 2 and rep["errors"] == ["us_per_layer"], rep["errors"]
    assert rep["pool"]["cold_bytes_equal_checkpoint"] and rep["pool"]["cold_per_set"] == [1, 2, 3]
    assert len(rep["pairs"]) == 55 and all(v["of"] == 18 for v in rep["pairs"].values())
    v = rep["verdict"]
    assert v["all_equal"] and v["rowmatch_on_equal_r6"] and v["rowmatch_on_packed_equal_r6"] and v["today_equal_r6"]
    assert v["masked_1_equal_6"] and v["num_groups_invariant"] and not v["failed"] and v["not_equal_pairs"] == []
    assert fake.masked_packed_calls and fake.packed_a and fake.align[0] == 128


# ------------------------------------------------------------------------------------------------ the build check
def _occupancy_sensitive(fake):
    """A contiguous GEMM whose rows are NOT independent of their block: a row in a block with > 1 tagged rows gets
    +1 bf16 ulp (the failure mode the build check exists for)."""
    inner = fake.m_grouped_fp8_fp4_gemm_nt_contiguous

    def gemm(a, b, d, m_indices, **kw):
        inner(a, b, d, m_indices, **kw)
        BM = fake.align[0]
        mi = m_indices.view(-1, BM)
        busy = ((mi >= 0).sum(1) > 1).repeat_interleave(BM)
        dv = d.view(torch.int16)
        dv[busy] += 1

    fake.m_grouped_fp8_fp4_gemm_nt_contiguous = gemm
    return fake


def test_build_check_passes_then_falls_back_on_a_row_dependent_gemm(di_emu, monkeypatch, capsys):  # noqa: F811
    from runtime.static_step import StaticStep

    model, pool, fk, fake = di_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)
    rm = st.dg.rowmatch
    assert rm["on"] and rm["check"]["equal"] and rm["check"]["today_equal"]
    assert rm["check"]["build"] == {"align": 16, "blocks": 6 * st.topk} and rm["check"]["ref"]["blocks"] == 6 * st.topk
    assert all(v["build_vs_rows"] == v["of"] == 6 for v in rm["check"]["projs"].values())
    st.detach()
    _occupancy_sensitive(fake)
    st2 = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)
    rm2 = st2.dg.rowmatch
    assert not rm2["on"] and rm2["mode"] == "on" and "build check failed" in rm2["why"]
    assert rm2["check"]["equal"] is False and rm2["check"]["today_equal"] is False
    assert st2.dg.align == 128 and st2.dg.gemm_blocks == st2.topk
    assert "moe rowmatch 'on'" in capsys.readouterr().err
    st2.detach()
    monkeypatch.setenv(DI.LAYOUT_CHECK_ENV, "0")                  # unchecked: the requested shape is built as is
    st3 = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)
    assert st3.dg.rowmatch["on"] and st3.dg.rowmatch["check"] is None and st3.dg.align == 16
    st3.detach()


def test_check_rowmatch_restores_the_build_buffers(di_emu):  # noqa: F811
    from runtime.static_step import StaticStep

    model, pool, fk, fake = di_emu("mini")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-inplace", _allow_cpu=True)
    b = st.dg
    rec = b.check_rowmatch(6, 16, 6 * st.topk, seed=5)
    assert rec["equal"] and rec["routing"] and len(rec["routing"]) == 6
    for w in (1, 2):
        assert not b._g["xq"][w].view(torch.uint8).any() and bool((b._g["xs"][w] == 1).all())
    assert bool((b._g["mi"] == -1).all()) and not b.sf_pack
    with pytest.raises(ValueError, match="gemm_blocks"):
        DI.DGInplaceBuffers(st, {p: (8, 8) for p in DI.PROJS}, {}, pool=pool, swiglu_limit=0.0, gemm_blocks=1)
    st.detach()


# ------------------------------------------------------------------------------------------------ bench flags
def test_first_light_and_headline_flags(monkeypatch):
    fl = _bench("first_light")
    for v in (DI.ROWMATCH_ENV, DI.ROWS_GEMM_ENV, DI.ROWS_SF_STORE_ENV):   # recorded: teardown removes what the
        monkeypatch.setenv(v, "x")                                        # bench writes into os.environ below
        monkeypatch.delenv(v)
    a = fl.build_parser().parse_args(["--model-dir", "/x"])
    assert a.moe_rowmatch == "off"   # default off: box 2026-11-07 showed +1.7 ms at 1 row and no stream gain (LEDGER)
    a = fl.build_parser().parse_args(["--model-dir", "/x", "--moe-rowmatch", "off"])
    env = fl.rows_moe_env(a)
    assert env[DI.ROWMATCH_ENV] == "off" and os.environ[DI.ROWMATCH_ENV] == "off"
    hl = _bench("headline")
    h = hl.build_parser().parse_args(["--model-dir", "/x"])
    assert h.moe_rowmatch == "off"
    assert hl.build_parser().parse_args(["--model-dir", "/x", "--moe-rowmatch", "align"]).moe_rowmatch == "align"


def test_headline_rowmatch_record_and_glue_orders():
    hl = _bench("headline")

    class Glue:
        def __init__(self, orders, off):
            self.orders = orders
            self.status = {c: ("off" if c in off else "on") for c in ("hc_norm", "mix_stat", "engram", "gather")}

        def on(self, c):
            return self.status[c] == "on"

    class DG:
        align, gemm_blocks = 16, 36
        rowmatch = {"mode": "on", "on": True, "why": None, "rows_gemm": "contiguous16",
                    "check": {"equal": True, "today_equal": True,
                              "projs": {"w1": {"build_vs_rows": 6, "today_vs_rows": 6, "of": 6}}}}

    class St:
        def __init__(self, glue, dg=None):
            self.glue, self.dg = glue, dg

    one = St(Glue({"norm_D": {"name": "vec_T512_V4_dec|mul"}, "mix": {"name": "vec_T512_V4_dec|mul"},
                   "kv": {"name": "vec_T128_V4_dec|mul"}}, off={"gather"}), DG())
    six = St(Glue({"norm_D": {"name": "vec_T128_V4_dec|mul"}, "mix": {"name": "vec_T128_V8_dec|mul"},
                   "kv": {"name": "vec_T128_V4_dec|mul"}}, off={"gather", "mix_stat"}))
    rec = hl.rowmatch_record(one)
    assert rec["dg"]["on"] and rec["dg"]["check_equal"] and rec["dg"]["gemm_blocks"] == 36
    hl.orders_compare(rec, six)
    assert rec["glue_orders_differ"] == ["mix", "norm_D"] and rec["glue_off_only_verify"] == ["mix_stat"]
    line = hl.rowmatch_line(rec)
    assert "DG 1-row call ON" in line and "differ at ['mix', 'norm_D']" in line and "mix_stat" in line
    rep = {"rowmatch": rec, "headline": {}, "prompts": {}, "streams": {}}
    assert any(l.startswith("  rowmatch: DG 1-row call ON") for l in hl.summary_lines(rep))
    empty = hl.orders_compare(hl.rowmatch_record(St(None)), St(None))
    assert empty["dg"] is None and empty["glue_orders_differ"] is None and hl.rowmatch_line(empty) == "DG n/a"
    assert sys.modules.get("vllm") is None
