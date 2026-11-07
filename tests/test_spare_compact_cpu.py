"""Mac CPU tests of M1-T10-FIX2 (docs/HANDOFF-M1-T10-FIX2.md): the 6-row dg-inplace step's compact cold index.
`.venv/bin/python -m pytest -q tests/test_spare_compact_cpu.py`

  * compact_cold (device ops) == compact_cold_host on dedup_routes mappings: routed ties (one expert on many rows ->
    one slot), zero-cold layers, all-cold layers (H = 0, the duplicate-id unused slots), overflow counts;
  * the staging contract with sidx (stage_inplace_emulated): cold expert c -> spare H + c, skipped slots -1, an
    overflowing cold slot writes nothing and bumps ovf; the Triton kernel body (tests/triton_shim.py, program by
    program) == the contract, counter included;
  * the 6-row dg-inplace StaticStep at S = 8 (default), S = the max cold count (boundary) and S = R*k == dg-masked,
    torch.equal (logits, main_hidden, caches); cold groups are H, H+1, ... in block order; overflow 0;
  * forced overflow (S = 1): the device counter trips by exactly the host expectation from each step's routing, and
    the step still runs;
  * validation (spare_slots only at rows > 1 with dg-inplace; >= 1; clamped to R*k) and Generator plumbing.
"""
from __future__ import annotations

import sys

import pytest
import torch

from runtime import dg_inplace as DI
from runtime.moe_rows import compact_cold, compact_cold_host, dedup_routes
from test_dg_inplace_cpu import _static_kernels_under_shim, _synthetic_layer
from test_static_rows_cpu import _assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens, emu, rows_dg  # noqa: F401


# ------------------------------------------------------------------------------------------------ the mapping
def _gtab(E, hot):
    g = torch.full((E,), -1, dtype=torch.int32)
    for s, e in enumerate(hot):
        g[e] = s
    return g


@pytest.mark.parametrize("case", ["random", "ties", "zero_cold", "all_cold"])
def test_compact_cold_mapping_equals_host(case):
    g = torch.Generator().manual_seed(11)
    R_, k, E = 6, 6, 32
    for trial in range(30):
        if case == "ties":  # every row routes to the same k experts (one slot, R rows), half swap one for another id
            perm = torch.randperm(E, generator=g)
            base, extra = perm[:k], perm[k:k + R_]
            idx = base.expand(R_, k).clone()
            idx[: R_ // 2, -1] = extra[: R_ // 2]
            idx = torch.stack([r[torch.randperm(k, generator=g)] for r in idx])
        else:
            idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R_)])
        w = torch.rand(R_, k, generator=g)
        if case == "zero_cold":
            hot = sorted(set(idx.reshape(-1).tolist()))           # every routed expert hot
        elif case == "all_cold":
            hot = []
        else:
            hot = sorted(torch.randperm(E, generator=g)[: E // 2].tolist())
        gt = _gtab(E, hot)
        m = dedup_routes(idx, w, 128)
        uniq, counts = m["uniq"], m["counts"]
        H = len(hot)
        for S in (1, 3, 8, R_ * k):
            cold, ok, grp, sidx = compact_cold(gt.index_select(0, uniq).long(), counts, H, S)
            host = compact_cold_host(gt.tolist(), uniq.tolist(), counts.tolist(), H, S)
            assert grp.tolist() == host["grp"] and sidx.tolist() == host["sidx"] and ok.tolist() == host["ok"]
            used = [(int(e), int(n)) for e, n in zip(uniq, counts) if n > 0]
            cold_ids = [e for e, n in used if e not in hot]
            assert int(cold.sum()) == len(cold_ids) and host["overflow"] == max(0, len(cold_ids) - S)
            # cold slots get 0, 1, 2, ... in ascending expert id; everything else -1
            assert [c for c in sidx.tolist() if c >= 0] == list(range(len(cold_ids)))
            assert ok.tolist() == [c < S for c in sidx.tolist()]           # -1 (not cold) is always ok
            if case == "zero_cold":
                assert not cold.any() and all(c == -1 for c in sidx.tolist())
            if case == "all_cold":
                assert int(cold.sum()) == len(used)
            if case == "ties":
                assert int(counts.max()) >= R_ // 2                         # one slot carries many rows


# ------------------------------------------------------------------------------------------------ staging contract
def _contract_case(all_scales, spare, N=8, KS=8, KH=16, seed=0, use_kernel=False, block_bytes=64, nprog=2):
    """The synthetic layer of tests/test_dg_inplace_cpu.py (E 8, H 5 hot [6,1,3,0,5], cold [2,4,7], 3 spare slots),
    six A blocks: experts [0, 1, 2, 4, 7] used + an unused slot carrying the hot dummy 6; compact cold index
    2 -> 0, 4 -> 1, 7 -> 2."""
    L = _synthetic_layer(N=N, KS=KS, KH=KH, seed=seed)
    H, S3 = L["H"], L["S"]
    sf = {p: DI.packed_sf_zeros(H + S3, N, KS // 4, "cpu") for p in DI.PROJS}
    dsts = [L["full"][p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64)
    ids = torch.tensor([0, 1, 2, 4, 7, 6])
    sidx = torch.tensor([-1, -1, 0, 1, 2, -1])
    align = 4
    mi = torch.full((6 * align,), -1, dtype=torch.int32)
    ovf = torch.zeros(1, dtype=torch.int32)
    ids_out = torch.zeros(6, dtype=torch.long)
    args = (L["ptab6"], ids, L["gtab"], dtab, dsts, mi, H, align, N * KH, ((N, KS // 4), (N, KS // 4)), all_scales,
            ids_out)
    if use_kernel:
        SK, shim = _static_kernels_under_shim()
        shim.ADDRESSES.clear()
        shim.ADDRESSES.register(*dsts, *L["src_w"].values(), *L["src_s"].values())
        try:
            SK.stage_inplace(*args, block_bytes=block_bytes, nprog=nprog, sidx=sidx, spare=spare, ovf=ovf)
        finally:
            shim.ADDRESSES.clear()
        assert SK.t9a2_stage_inplace.launches[-1]["grid"] == (6, nprog, 3)
    else:
        DI.stage_inplace_emulated(L["reg"].__getitem__, *args, sidx=sidx, spare=spare, ovf=ovf)
    return L, sf, dsts, mi, ovf, ids_out


@pytest.mark.parametrize("all_scales", [True, False])
@pytest.mark.parametrize("spare", [3, 2, 1])
def test_stage_emulated_compact_index_and_overflow(all_scales, spare):
    L, sf, dsts, mi, ovf, ids_out = _contract_case(all_scales, spare)
    H = L["H"]
    gt = L["gtab"].tolist()
    cold = [2, 4, 7]
    staged = cold[:spare]
    heads = mi.tolist()[::4]
    want = [gt[0], gt[1]] + [H + c if c < spare else -1 for c in range(3)] + [gt[6]]
    assert heads == want and all(v == -1 for i, v in enumerate(mi.tolist()) if i % 4)
    assert int(ovf) == 3 - spare and ids_out.tolist() == [0, 1, 2, 4, 7, 6]
    for p_i, p in enumerate(DI.PROJS):
        w = L["full"][p]
        for c, e in enumerate(cold):
            if c < spare:
                assert torch.equal(w[H + c], L["src_w"][(p, e)])
        assert not w[H + spare:].any()                                      # overflow: nothing written
        got = DI.unpack_sf(sf[p])
        for c, e in enumerate(staged):
            assert torch.equal(got[H + c], L["src_s"][(p, e)])
        written = {H + c for c in range(spare)} | ({gt[0], gt[1], gt[6]} if all_scales else set())
        if all_scales:
            for e in (0, 1, 6):
                assert torch.equal(got[gt[e]], L["src_s"][(p, e)])
        assert not got[[g_ for g_ in range(got.size(0)) if g_ not in written]].any()


@pytest.mark.parametrize("all_scales", [True, False])
@pytest.mark.parametrize("spare", [3, 1])
@pytest.mark.parametrize("N,block_bytes,nprog", [(8, 64, 2), (132, 64, 3)])
def test_stage_kernel_body_with_sidx_equals_contract(all_scales, spare, N, block_bytes, nprog):
    outs = []
    for use_kernel in (False, True):
        L, sf, dsts, mi, ovf, ids_out = _contract_case(all_scales, spare, N=N, seed=5, use_kernel=use_kernel,
                                                       block_bytes=block_bytes, nprog=nprog)
        outs.append([t.clone() for t in dsts] + [mi.clone(), ovf.clone(), ids_out.clone()])
    for i, (a_, b_) in enumerate(zip(*outs)):
        assert torch.equal(a_, b_), i
    assert int(outs[1][-2]) == 3 - spare


def test_stage_sidx_args_go_together():
    L, sf, dsts, mi, ovf, ids_out = _contract_case(True, 3)
    with pytest.raises(ValueError, match="together"):
        DI.stage_inplace_emulated(L["reg"].__getitem__, L["ptab6"], torch.tensor([0, 1]), L["gtab"],
                                  torch.tensor([t.data_ptr() for t in dsts]), dsts, mi[:8], L["H"], 4, 128,
                                  ((8, 2), (8, 2)), True, None, sidx=torch.tensor([-1, -1]), spare=None, ovf=ovf)


# ------------------------------------------------------------------------------------------------ the 6-row step
def _max_cold(st):
    """The largest cold unique-expert count any layer saw in the last eager step (from each MoE's `last`)."""
    out = []
    for m in st.moes:
        gt = m.t["gtab"].tolist()
        n = [int(gt[int(e)] < 0 and int(c) > 0) for e, c in zip(m.last["uniq"], m.last["counts"])]
        out.append(sum(n))
    return out


@pytest.mark.parametrize("spare", [None, "max_cold", "rk"])
def test_static6_dg_inplace_compact_equals_dg_masked(rows_dg, spare):
    """S = 8 (default), S = exactly the largest per-layer cold count (the boundary: c = S-1 is the last staged) and
    S = R*k (FIX's slot count, compact index) -> torch.equal to dg-masked; groups of cold blocks are H, H+1, ...
    in block (ascending id) order; overflow 0; the stage launch got the compact index every time."""
    from runtime.static_step import DEFAULT_SPARE_SLOTS_ROWS, StaticStep

    model, pool, fk, fake = rows_dg("mini4")
    hid = (len(model.layers) - 1,)
    n = 3
    ids, toks = _tf_tokens(18, 6 * n, 41)
    m2 = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-masked", _allow_cpu=True)
    b_lg, b_h, b_s = _static_run(model, m2, ids, toks, n)
    m2.detach()
    k = 4
    if spare == "max_cold":  # mini4: 4 hot of 8 per layer -> at most 4 cold unique experts per layer-step
        S = 4
    elif spare == "rk":
        S = 6 * k
    else:
        S = None
    st = StaticStep(model, pool, max_pos=511, rows=6, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=S,
                    _allow_cpu=True)
    S_eff = S or DEFAULT_SPARE_SLOTS_ROWS
    assert st.spare_slots == S_eff == st.dg.spare_limit and pool.spare == S_eff and st.dg.k == 6 * k
    assert st.describe()["spare_slots"] == S_eff and st.dg.describe()["spare_limit"] == S_eff
    n0 = len(fake.contig_calls)
    s0 = fk.calls.get("stage_sidx", 0)
    a_lg, a_h, a_s = _static_run(model, st, ids, toks, n)
    calls = fake.contig_calls[n0:]
    n_layers = len(model.layers)
    assert fk.calls["stage_sidx"] - s0 == n_layers * n
    n_cold = 0
    for c in calls:
        G = c[1]
        H = G - pool.spare
        cold_h = [h - H for h in c[4] if h >= H]
        assert cold_h == list(range(len(cold_h))), c[4]                     # compact: H, H+1, ... in block order
        n_cold += len(cold_h)
    assert n_cold > 0
    if spare == "max_cold":
        assert max(len([h for h in c[4] if h >= c[1] - pool.spare]) for c in calls) == 4   # the boundary is hit
    ov = st.overflow_counts()
    assert ov == {"steps": 0, "experts": 0, "steps_counted": n, "spare_slots": S_eff}
    _assert_rows_equal(a_lg, b_lg, f"6-row dg-inplace S={S_eff} vs dg-masked")
    assert all(torch.equal(x, y) for x, y in zip(a_h, b_h))
    for k_ in b_s:
        assert torch.equal(a_s[k_], b_s[k_]), k_
    st.detach()


def test_static6_dg_inplace_forced_overflow_trips_counter(rows_dg):
    """S = 1 on mini4 (up to 4 cold unique experts per layer-step): the device counter counts exactly the cold
    experts beyond the first per layer and the steps where that happened (host expectation from every MoE's routing
    of each eager step); the step runs; heads never exceed H + S - 1."""
    from runtime.moe_rows import compact_cold_host
    from runtime.static_step import StaticStep

    model, pool, fk, fake = rows_dg("mini4")
    real = fake.m_grouped_fp8_fp4_gemm_nt_contiguous

    def finite_skips(a, b, d, m_indices, **kw):  # skipped (-1) blocks keep finite values, as on the GPU (stale rows)
        real(a, b, d, m_indices, **kw)           # (the fake NaN-poisons them; NaN would break the e8m0 act_quant)
        d.nan_to_num_(0.0)

    fake.m_grouped_fp8_fp4_gemm_nt_contiguous = finite_skips
    st = StaticStep(model, pool, max_pos=511, rows=6, moe_impl="dg-inplace", spare_slots=1, _allow_cpu=True)
    assert pool.spare == 1 and st.spare_slots == 1
    ids, toks = _tf_tokens(17, 6 * 3, 9)
    from test_static_rows_cpu import _prefill

    _prefill(model, ids)
    want_steps = want_exp = 0
    n0 = len(fake.contig_calls)
    with torch.inference_mode():
        for s in range(3):
            st.step(toks[6 * s:6 * s + 6], len(ids) + 6 * s, graph=False)
            over = 0
            for m in st.moes:
                h = compact_cold_host(m.t["gtab"].tolist(), m.last["uniq"].tolist(), m.last["counts"].tolist(),
                                      m.t["H"], 1)
                over += h["overflow"]
            want_exp += over
            want_steps += int(over > 0)
            assert int(st.ovf_step) == over                                 # this step's counter
    ov = st.overflow_counts()
    assert want_steps > 0 and ov["steps"] == want_steps and ov["experts"] == want_exp and ov["steps_counted"] == 3
    for c in fake.contig_calls[n0:]:
        assert max(c[4]) <= c[1] - 1 and all(h < c[1] - pool.spare + 1 for h in c[4])
    st.detach()


def test_spare_slots_validation_and_clamp(rows_dg):
    from runtime.static_step import StaticStep

    model, pool, _, _ = rows_dg("mini")
    with pytest.raises(ValueError, match="spare_slots"):
        StaticStep(model, pool, max_pos=255, moe_impl="dg-inplace", spare_slots=4, _allow_cpu=True)   # 1 row
    with pytest.raises(ValueError, match="spare_slots"):
        StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-masked", spare_slots=4, _allow_cpu=True)
    with pytest.raises(ValueError, match="spare_slots"):
        StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", spare_slots=0, _allow_cpu=True)
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", spare_slots=99, _allow_cpu=True)
    assert st.spare_slots == st.n_slots == 6 * st.topk and pool.spare == st.n_slots     # clamped to R*k
    st.detach()
    s1 = StaticStep(model, pool, max_pos=255, moe_impl="dg-inplace", _allow_cpu=True)    # 1 row: topk, no ovf
    assert s1.spare_slots is None and s1.overflow_counts() is None and s1.dg.spare_limit == s1.topk
    s1.detach()


def test_generator_passes_spare_slots_only_to_the_rows_dg_inplace_build():
    from runtime.generate import Generator

    seen = []

    class Fake:
        def __init__(self, m, **kw):
            seen.append(kw)

        def detach(self):
            pass

    gen = Generator(torch.nn.Linear(2, 2), tokenizer=object(), eos_ids=[1], device="cpu", mode="static",
                    set_default_device=False, static_factory=lambda m, **kw: Fake(m, **kw), moe_impl="dg-inplace",
                    spare_slots=5)
    gen.static_step()
    gen.set_rows(6)
    gen.static_step()
    gen.set_spare_slots(3)                    # rows dg-inplace build -> dropped and rebuilt with the new budget
    assert gen.static is None
    gen.static_step()
    gen.set_moe_impl("dg-masked")
    gen.static_step()
    assert [k.get("spare_slots") for k in seen] == [None, 5, 3, None]
    with pytest.raises(ValueError):
        gen.set_spare_slots(0)


def test_first_light_spare_slots_flag():
    import importlib.util
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("bench_first_light_fx2", os.path.join(repo, "bench", "first_light.py"))
    fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fl)
    base = ["--model-dir", "x", "--rows", "6", "--teacher-forced", "--moe-impl", "dg-inplace"]
    assert fl.build_parser().parse_args(base).spare_slots is None
    assert fl.build_parser().parse_args(base + ["--spare-slots", "12"]).spare_slots == 12
    r = {"date": "d", "errors": [], "generator": {}, "boot": {}, "env": {},
         "rows_mode": {"rows": 6, "overflow": {"steps": 2, "experts": 3, "spare_slots": 8, "steps_counted": 9},
                       "build_1row": {"hbm": {"allocated_GiB": 222.9, "free_GiB": 25.5}},
                       "build_rows": {"hbm": {"allocated_GiB": 224.2, "free_GiB": 24.1}, "spare_added_GiB": 1.318}}}
    lines = fl.summary_rows_lines(r, "x.json")
    assert len(lines) == 12 and "| overflow steps: 2 (3 experts, spare 8/layer) |" in lines[-1]
    assert "1-row 222.90 alloc / 25.50 free, 6-row 224.20 / 24.10 GiB (+1.318 GiB spare)" in lines[-1]
    r["rows_mode"].pop("overflow")
    assert "| overflow steps: n/a |" in fl.summary_rows_lines(r, "x.json")[-1]
