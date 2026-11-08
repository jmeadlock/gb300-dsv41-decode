"""M3-S3 (docs/HANDOFF-M3-S3-POOL-SF-LAYOUT.md, docs/M3-S3-POOL-SF-LAYOUT.md): the expert pool's SF "beside" its
weights ($DSV41_POOL_SF=beside, runtime/expert_pool.py) -- the CPU gates. Values never change, bytes move:

  * the switch: default "separate" (today), "beside", anything else refused;
  * pool bytes expert by expert (two loads of the same mini checkpoint): weights identical; the K-major e8m0 of every
    hot slot identical (now in the host buffer `hot_sf_host`, the Experts' scale params alias it); the pool-owned
    packed SF of every hot slot == DeepGEMM's packing of today's bytes (dg_inplace.pack_sf; unpacks to the same
    codes); cold bytes identical; the loader's verify passes; HBM bytes at load equal (packed group = e8m0 row bytes);
    spare growth carries the hot groups, zeroes the spare ones and counts the packed bytes;
  * the 6-row dg-inplace static step, kernel and kernel-overlap: beside == separate under the "step" store AND the
    "pool" store, torch.equal on every row's logits, main_hidden, caches, the spare weight slots, and the spare SF
    groups (vs the separate pool store); under beside every layer is "pool" whatever was requested, its tables
    ARE the pool's sfp, the work lists are cold-only (n_work = spare budget) and a recording resolver shows the
    stage never reads a hot scale address; a negative control (one hot packed group corrupted) changes the logits;
  * 1-row == 6-row per row under beside (row-invariant torch mode; the 1-row build shares the pool SF);
  * residency on a beside pool: forced + policy swaps == residency off (and == separate), resident SF registered for
    EVERY layer, promoted slots' packed group == the packing of the expert's checkpoint e8m0, residency e8m0 in a
    host buffer, the budget counts weights + one packed group per slot, audit with the checkpoint golden ok;
  * the staging kernel bodies (triton shim) on a beside pool's own storage == the contract, hot groups untouched,
    the hot e8m0 never resolvable (not registered with the shim);
  * bench/pool_layout_bench.py CPU subcommands.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

from runtime import dg_inplace as DI
from runtime import expert_pool as EP
from runtime import staging_overlap as SO
from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _prefill, _rowwise_torch,  # noqa: F401
                                  _static_run, _tf_tokens, emu, rows_dg)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "bench") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "bench"))
E8 = torch.float8_e8m0fnu


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (EP.POOL_SF_ENV, DI.SF_STORE_ENV, DI.ROWS_SF_STORE_ENV, DI.SF_POOL_LAYERS_ENV):
        monkeypatch.delenv(k, raising=False)
    yield


def _registry(pool):
    class Reg(dict):
        def __missing__(self, key):
            from test_dg_inplace_cpu import _fill_registry

            _fill_registry(self, pool)
            if key not in self.keys():
                raise KeyError(key)
            return dict.get(self, key)

    return Reg()


def _hot_scale_addrs(pool) -> set:
    return {pool.hbm[l][p][1][s].data_ptr() for l in range(pool.n_layers) for p in DI.PROJS
            for s in range(len(pool.hot[l]))}


# ------------------------------------------------------------------------------------------------ the switch
def test_layout_switch(monkeypatch):
    monkeypatch.delenv(EP.POOL_SF_ENV, raising=False)
    assert EP.pool_sf_layout() == "beside"   # default since 11/08 (box 18.74 vs 19.06, bit-identical)
    monkeypatch.setenv(EP.POOL_SF_ENV, "separate")
    assert EP.pool_sf_layout() == "separate"
    monkeypatch.setenv(EP.POOL_SF_ENV, "beside")
    assert EP.pool_sf_layout() == "beside" and EP.pool_sf_layout("separate") == "separate"
    monkeypatch.setenv(EP.POOL_SF_ENV, "adjacent")
    with pytest.raises(ValueError, match="not in"):
        EP.pool_sf_layout()
    # the helpers are dg_inplace's, bit for bit
    s = torch.randint(0, 256, (3, 8, 16), dtype=torch.uint8)
    assert torch.equal(EP.pack_words_e8m0(s.view(E8)), DI.pack_words(s.view(E8)))
    a, b = EP.packed_sf_empty(5, 8, 4, "cpu"), DI.packed_sf_zeros(5, 8, 4, "cpu")
    assert a.shape == b.shape and a.stride() == b.stride() and DI.is_mn_major_packed(a, 5, 8, 4)


# ------------------------------------------------------------------------------------------------ pool bytes
def _load(emu, monkeypatch, layout, kind="mini4"):
    monkeypatch.setenv(EP.POOL_SF_ENV, layout)
    model, pool, _ = emu(kind)
    assert pool.sf_layout == layout
    return model, pool


def test_pool_bytes_equal_expert_by_expert(emu, monkeypatch):
    from runtime.loader import verify_placement

    m_s, p_s = _load(emu, monkeypatch, "separate")
    m_b, p_b = _load(emu, monkeypatch, "beside")
    rl = m_b.runtime_load
    assert p_s.hot_sf_host is None and not p_s.sfp
    assert rl.host_buffers.get("hot_scales") is p_b.hot_sf_host and rl.bytes["pool_HOST_hot_scales"] == p_b.hot_sf_bytes
    assert "pack_hot_sf" in rl.timings and p_b.sf_packed == set(range(p_b.n_layers))
    n_hot = 0
    for l in range(p_s.n_layers):
        assert p_b.hot[l] == p_s.hot[l] and p_b.cold[l] == p_s.cold[l]
        H = len(p_s.hot[l])
        n_hot += H
        for p in DI.PROJS:
            n, k = p_s.proj_nk[p]
            sfp = p_b.sfp[l][p]
            assert DI.is_mn_major_packed(sfp, H, n, k // 128), (l, p, sfp.shape, sfp.stride())
            assert p_b.hbm[l][p][1].device.type == "cpu"                       # the e8m0 lives in the host buffer
            for s, e in enumerate(p_s.hot[l]):
                assert torch.equal(p_b.hbm_full[l][p][s], p_s.hbm_full[l][p][s]), (l, p, e, "weights")
                e8 = p_s.hbm[l][p][1][s].view(torch.uint8)
                assert torch.equal(p_b.hbm[l][p][1][s].view(torch.uint8), e8), (l, p, e, "e8m0 (host)")
                assert torch.equal(sfp[s], DI.pack_sf(p_s.hbm[l][p][1][s])), (l, p, e, "packed slot")
                assert torch.equal(DI.unpack_sf(sfp[s]), e8), (l, p, e, "packed slot unpacks to the same codes")
                lin = getattr(m_b.layers[l].ffn.experts[e], p)
                assert lin.scale.data_ptr() == p_b.hot_sf_host.ptr + p_b.hot_scale_offset(l, e, p)
                assert lin.weight.data_ptr() == p_b.hbm_full[l][p][s].data_ptr()
            for e in p_s.cold[l]:
                for kind_ in ("weight", "scale"):
                    a = p_s.cold_view(l, e, p, kind_, cuda=False).view(torch.uint8)
                    b = p_b.cold_view(l, e, p, kind_, cuda=False).view(torch.uint8)
                    assert torch.equal(a, b), (l, p, e, kind_)
    # bytes: at load the HBM pool holds the same number of bytes (a packed group is the e8m0 rows' bytes); the e8m0
    # copy (>= every hot slot's scales) is on the host
    assert p_b.hbm_bytes() == p_s.hbm_bytes()
    assert p_b.hot_sf_bytes >= n_hot * p_s.sf_group_bytes
    v = verify_placement(m_b, rl)
    assert v["ok"] == v["n"] and not v["mismatch"], v["mismatch"][:5]
    # spare growth: the packed SF grows with the weights, hot groups carried, spare groups zero
    before = {(l, p): p_b.sfp[l][p][: len(p_b.hot[l])].clone() for l in range(p_b.n_layers) for p in DI.PROJS}
    gen = p_b.generation
    add_b, add_s = p_b.add_spare_slots(3, m_b), p_s.add_spare_slots(3, m_s)
    L = p_b.n_layers
    assert add_s == L * 3 * p_s.weight_bytes and add_b == L * 3 * (p_b.weight_bytes + p_b.sf_group_bytes)
    assert p_b.generation == gen + 1
    for l in range(L):
        H = len(p_b.hot[l])
        for p in DI.PROJS:
            n, k = p_b.proj_nk[p]
            assert DI.is_mn_major_packed(p_b.sfp[l][p], H + 3, n, k // 128)
            assert torch.equal(p_b.sfp[l][p][:H], before[(l, p)]) and not p_b.sfp[l][p][H:].any()
    d = p_b.describe_sf()
    assert d["sf_layout"] == "beside" and d["groups_per_layer"] == sorted({len(p_b.hot[l]) + 3 for l in range(L)})
    assert p_b.hbm_bytes() - p_s.hbm_bytes() == L * 3 * p_b.sf_group_bytes   # the spare slots' packed groups


# ------------------------------------------------------------------------------------------------ the 6-row step
def _run6(model, pool, monkeypatch, store, ids, toks, n, record=None):
    from runtime.static_step import StaticStep

    reg = _registry(pool)
    orig = DI.stage_rows_emulated

    def resolve_rec(get):
        def r(a):
            if record is not None:
                record.add(int(a))
            return get(a)
        return r

    monkeypatch.setattr(SO, "CPU_RESOLVE", resolve_rec(reg.__getitem__))
    monkeypatch.setattr(DI, "stage_rows_emulated", lambda resolve, *a, **k: orig(resolve_rec(resolve), *a, **k))
    pool._dg_packed_sf = None
    st = StaticStep(model, pool, max_pos=255, rows=6, hidden_layers=(len(model.layers) - 1,), moe_impl="dg-inplace",
                    rows_sf_store=store, _allow_cpu=True)
    out = _static_run(model, st, ids, toks, n)
    b = st.dg
    slots = [pool.hbm_full[l][p][len(pool.hot[l]):].clone() for l in range(pool.n_layers) for p in DI.PROJS]
    sfs = [b.layers[l]["sf"][p][b.layers[l]["H"]:].clone() for l in sorted(b.layers) for p in DI.PROJS]
    lists = {sm.layer: (sm.last, b.stage_n_work(sm.t)) for sm in st.moes}
    monkeypatch.setattr(DI, "stage_rows_emulated", orig)
    return st, {"out": out, "slots": slots, "sf": sfs, "lists": lists}


def _same(a, b, what, sf=True, n_pos=None):
    """n_pos (two separately LOADED models): the engram hash cache is compared over the positions written
    (reference engram.py:167 writes cache[:, start:start + seqlen]; the tail is each model's own torch.empty)."""
    _assert_rows_equal(a["out"][0], b["out"][0], what)
    assert all(torch.equal(x, y) for x, y in zip(a["out"][1], b["out"][1])), (what, "main_hidden")
    for k_ in b["out"][2]:
        x, y = a["out"][2][k_], b["out"][2][k_]
        if k_ == "engram" and n_pos is not None:
            x, y = x[:, :n_pos], y[:, :n_pos]
        assert torch.equal(x, y), (what, k_)
    assert all(torch.equal(x, y) for x, y in zip(a["slots"], b["slots"])), (what, "spare weight slots")
    if sf:
        assert all(torch.equal(x, y) for x, y in zip(a["sf"], b["sf"])), (what, "spare SF groups")


@pytest.mark.parametrize("mode", ["kernel", "kernel-overlap"])
def test_static6_beside_equals_separate(rows_dg, monkeypatch, mode):
    monkeypatch.setenv(SO.STAGE_ENV, mode)
    n = 3
    ids, toks = _tf_tokens(18, 6 * n, 77)
    monkeypatch.setenv(EP.POOL_SF_ENV, "separate")
    model, pool, fk, fake = rows_dg("mini4")
    ref = {}
    hot_sep = _hot_scale_addrs(pool)
    for store in ("step", "pool"):
        seen = set()
        st, ref[store] = _run6(model, pool, monkeypatch, store, ids, toks, n, record=seen)
        assert st.dg.sf_store == store and set(st.dg.sf_layer.values()) == {store}
        # the recorder has teeth: the step store re-packs the used hot experts' scales, the pool store does not
        assert bool(seen & hot_sep) == (store == "step"), (store, len(seen & hot_sep))
        st.detach()
    _same(ref["pool"], ref["step"], "separate: pool vs step store", sf=False)
    monkeypatch.setenv(EP.POOL_SF_ENV, "beside")
    model_b, pool_b, fk_b, _ = rows_dg("mini4")
    hot_e8 = _hot_scale_addrs(pool_b)
    staged_any = False
    for store in ("step", "pool"):                    # whatever is requested, beside's store is the pool's own SF
        seen = set()
        st, got = _run6(model_b, pool_b, monkeypatch, store, ids, toks, n, record=seen)
        b = st.dg
        assert b.sf_store == "pool" and b.sf_beside and set(b.sf_layer.values()) == {"pool"} and not b.sf_shared
        assert b.sf_pool_layers is None and b.sf_pool_plan is None
        assert (store == "pool") == (b.sf_store_why is None) and (store == "pool" or "beside" in b.sf_store_why)
        d = b.describe()
        assert d["pool_sf_layout"] == "beside" and d["stage_n_work"] == b.spare_limit
        assert d["hbm_added_B"]["packed_sf"] == sum(t.untyped_storage().nbytes() for sf in pool_b.sfp.values()
                                                    for t in sf.values())
        for l, t in b.layers.items():
            for p in DI.PROJS:
                assert t["sf"][p] is pool_b.sfp[l][p]                   # the tables ARE the pool's packed SF
            assert not b.step_sf(t) and b.stage_n_work(t) == b.spare_limit
        for l, (m, nw) in got["lists"].items():
            us = [int(u) for u in m["wl"][:nw].tolist() if u >= 0]
            assert all(int(m["sidx"][u]) >= 0 for u in us), (l, us)     # cold slots only
            staged_any |= bool(us)
        assert seen and not (seen & hot_e8), "the stage read a hot expert's scale bytes under beside"
        npos = len(ids) + 6 * n
        _same(got, ref["step"], f"{mode}: beside ({store} requested) vs separate step store", sf=False, n_pos=npos)
        _same(got, ref["pool"], f"{mode}: beside ({store} requested) vs separate pool store", n_pos=npos)
        st.detach()
    assert staged_any, "no cold expert was staged: the comparison did not exercise the stage"


def test_beside_gemm_reads_the_pool_sf_negative_control(rows_dg, monkeypatch):
    """Corrupt one hot packed group (of an expert the first steps route to) in pool.sfp: the logits change -- the
    equality above is not vacuous, the GEMM reads beside's SF."""
    from runtime.static_step import StaticStep

    monkeypatch.setenv(EP.POOL_SF_ENV, "beside")
    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    ids, toks = _tf_tokens(18, 12, 77)
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)
    good = _static_run(model, st, ids, toks, 2)
    with torch.inference_mode():
        for l in range(pool.n_layers):
            for p in DI.PROJS:
                pool.sfp[l][p][: len(pool.hot[l])].add_(1)                # every hot code x2: a different weight
    bad = _static_run(model, st, ids, toks, 2)
    assert not all(torch.equal(x, y) for x, y in zip(good[0], bad[0]))
    st.detach()


# ------------------------------------------------------------------------------------------------ 1 row == 6 rows
def test_one_row_equals_six_rows_per_row_beside(rows_dg, monkeypatch):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(EP.POOL_SF_ENV, "beside")
    model, pool, fk, fake = rows_dg("mini4")
    monkeypatch.setattr(SO, "CPU_RESOLVE", _registry(pool).__getitem__)
    _rowwise_torch(monkeypatch)
    L, n = 21, 3
    ids, toks = _tf_tokens(L, 6 * n, 5)
    s6 = StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", _allow_cpu=True)   # grows the pool
    s1 = StaticStep(model, pool, max_pos=255, moe_impl="dg-inplace", _allow_cpu=True)           # fits in it
    assert s1.dg.sf_store == "pool" and s6.dg.sf_store == "pool"
    for l in s6.dg.layers:
        for p in DI.PROJS:
            assert s1.dg.layers[l]["sf"][p] is s6.dg.layers[l]["sf"][p] is pool.sfp[l][p]
    a6, _, _ = _static_run(model, s6, ids, toks, n)
    _prefill(model, ids)
    a1 = []
    with torch.inference_mode():
        for i in range(6 * n):
            a1.append(s1.step(toks[i], L + i, graph=False)[0].clone())
    _assert_rows_equal(a6, a1, "beside: static6 vs static1 per row (row-invariant mode)")
    s1.detach()
    s6.detach()


# ------------------------------------------------------------------------------------------------ residency
def test_residency_promotion_under_beside(emu, monkeypatch):
    import numpy as np

    from runtime.residency import ResidencyConfig, ResidencyManager, golden_from_index, n_stage_for, sf_store_for
    from test_dg_inplace_cpu import _fake_dg_contig
    from test_dg_moe_cpu import _faithful_act_quant
    from test_residency_cpu import _hot_cold_bias, _install
    from test_residency_rows_cpu import _run6 as run6
    from test_static_rows_cpu import _masked_rowwise
    from runtime.static_step import StaticStep

    def setup(layout):
        monkeypatch.setenv(EP.POOL_SF_ENV, layout)
        model, pool, _ = emu("mini4")
        fake = _fake_dg_contig()
        fake.m_grouped_fp8_fp4_gemm_nt_masked = _masked_rowwise(fake)
        monkeypatch.setitem(sys.modules, "deep_gemm", fake)
        monkeypatch.setattr(sys.modules["model"], "act_quant", _faithful_act_quant)
        _install(monkeypatch, pool)
        return model, pool

    def build(model, pool, store):
        pool._dg_packed_sf = None
        return StaticStep(model, pool, max_pos=255, rows=6, moe_impl="dg-inplace", rows_sf_store=store,
                          _allow_cpu=True)

    n = 8
    ids, toks = _tf_tokens(17, 6 * n, 29)
    model, pool = setup("separate")
    st = build(model, pool, "step")
    off_sep = run6(model, st, ids, toks, n)
    st.detach()
    model, pool = setup("beside")
    st = build(model, pool, "step")
    off = run6(model, st, ids, toks, n)
    st.detach()
    _assert_rows_equal(off, off_sep, "residency off: beside vs separate")
    k = model.layers[0].ffn.n_activated_experts
    assert sf_store_for("dg-inplace", 1, pool) == "pool" and sf_store_for("dg-inplace", 6, pool) == "pool"
    cfg = ResidencyConfig(drain_n=2, min_abs=1.0, hyst=1.0, halflife=20.0, n_spare=2, n_demote=4, reserve_gib=0.0,
                          force_random=2, hbm_free_bytes=64 * 2 ** 30)
    rm = ResidencyManager(pool, model, cfg, n_stage=n_stage_for("dg-inplace", 6, k),
                          sf_store=sf_store_for("dg-inplace", 6, pool))
    assert rm.beside and rm.sspare_buf is not None
    assert rm.budget["per_spare_slot_all_layers_GiB"] == round(pool.n_layers * (pool.weight_bytes
                                                                                 + pool.sf_group_bytes) / 2 ** 30, 4)
    assert rm.budget["step_extra_GiB"] == 0.0
    for l in range(pool.n_layers):                                   # residency e8m0 in its host buffer
        for p in DI.PROJS:
            lo, hi = rm.sspare_buf.ptr, rm.sspare_buf.ptr + rm.sspare_buf.nbytes
            assert lo <= rm.sspare[l][p].data_ptr() < hi and rm.sspare[l][p].data_ptr() == rm.sspare_host[l][p].data_ptr()
    st = build(model, pool, None)
    rm.attach(st)
    assert sorted(rm.resident_sf) == list(range(pool.n_layers)) and rm.sf_store_got == "pool"
    for l in rm.resident_sf:
        assert all(rm.resident_sf[l][p] is pool.sfp[l][p] for p in DI.PROJS)
    rng = np.random.default_rng(2)
    on = run6(model, st, ids, toks, n, rm=rm, after=lambda s: rm.inject_counts(_hot_cold_bias(pool, rng, 30)))
    _assert_rows_equal(on, off, "beside: residency on (forced + policy swaps) vs off")
    assert rm.stats["published"] > 0 and rm.state.n_swaps > 0
    rm.flush()
    golden = golden_from_index(model.runtime_load.index)
    a = rm.audit(golden=golden)
    assert a["ok"], a["errors"][:5]
    # every HBM-resident expert's packed group == DeepGEMM's packing of its checkpoint e8m0 (promoted ones included)
    idx = model.runtime_load.index
    n_prom = 0
    for l in range(pool.n_layers):
        for e, s in rm.slot_of[l].items():
            for p in DI.PROJS:
                ent = idx.entries[f"layers.{l}.ffn.experts.{e}.{p}.scale"]
                ck = torch.empty(ent.shape, dtype=ent.dtype, device="cpu")
                idx.read_into(f"layers.{l}.ffn.experts.{e}.{p}.scale", ck)
                assert torch.equal(pool.sfp[l][p][s], DI.pack_sf(ck)), (l, e, s, p)
            n_prom += int(s >= len(pool.hot[l]))
    assert n_prom > 0
    d = rm.hbm_bytes()
    assert d["sf_layout"] == "beside" and d["spare_e8m0_host_B"] >= pool.n_layers * 2 * pool.sf_group_bytes
    rm.close()
    assert rm.sspare_buf is None
    st.detach()


# ------------------------------------------------------------------------------------------------ kernel bodies
def test_stage_kernel_bodies_on_a_beside_pool():
    """m1_stage_rows and s1_stage_proj x 3 (triton shim) on a beside pool's OWN storage (weights hbm_full, SF sfp),
    pool-style work list == stage_rows_emulated; hot groups untouched; the hot e8m0 is not registered with the shim
    (a read of it would raise), so no hot SF byte is read."""
    import pool_layout_bench as PB  # noqa: E402  (bench/ on sys.path below)
    import triton_shim
    from test_dg_inplace_cpu import _static_kernels_under_shim

    SK, shim2 = _static_kernels_under_shim()
    mod = triton_shim.load_module_under_shim("runtime/staging_overlap.py", "staging_overlap_shim_m3s3")
    mod.KERNELS_ON_CPU = True
    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7], [3, 5, 6]])
    res = {}
    for how in ("emu", "m1", "s1"):
        pool, src = PB.tiny_pool("beside")
        t = PB.layer_tables(pool)
        m, ext = PB.route_lists(pool, t, idx, False, 4)
        hot_before = [x[: t["H"]].clone() for x in t["dsts"]]
        args = (t["ptab6"], m["uniq"], t["gtab"], t["dtab"], t["dsts"], ext["sidx"], ext["wl"], t["H"], t["row_w"],
                t["sf_dims"], 4)
        if how == "emu":
            kinds = PB.address_kinds(pool)
            DI.stage_rows_emulated(lambda a: kinds[int(a)][1], *args)
        else:
            sh = triton_shim if how == "s1" else shim2
            cold_src = [pool.cold_view(0, e, p, kind_, cuda=False) for e in pool.cold[0] for p in DI.PROJS
                        for kind_ in ("weight", "scale")]
            sh.ADDRESSES.clear()
            sh.ADDRESSES.register(*t["dsts"], *cold_src)
            try:
                if how == "m1":
                    SK.stage_rows(*args, block_bytes=64, nprog=3)
                else:
                    for pj in range(3):
                        mod.stage_proj(*args, pj, block_bytes=64, nprog=3)
            finally:
                sh.ADDRESSES.clear()
        assert all(torch.equal(x[: t["H"]], y) for x, y in zip(t["dsts"], hot_before)), (how, "hot groups")
        res[how] = [x.clone() for x in t["dsts"]]
        H = t["H"]
        for c, e in enumerate(pool.cold[0]):                           # cold 2, 4, 7 -> spare 0, 1, 2
            for i, p in enumerate(DI.PROJS):
                assert torch.equal(t["dsts"][i][H + c], src[(p, "weight", e)]), (how, e, p)
                assert torch.equal(DI.unpack_sf(t["dsts"][3 + i][H + c]), src[(p, "scale", e)]), (how, e, p)
    for how in ("m1", "s1"):
        assert all(torch.equal(x, y) for x, y in zip(res["emu"], res[how])), how


# ------------------------------------------------------------------------------------------------ bench
def test_bench_cpu_subcommands(tmp_path):
    import pool_layout_bench as PB

    out = tmp_path / "pl.json"
    assert PB.main(["all", "--out", str(out)]) == 0
    d = json.load(open(out))
    b = d["bytes"]
    assert b["expert_weight_B"] == 17_694_720 and b["expert_sf_B"] == 1_105_920
    assert b["slot"]["w1"]["SFB_group_stride_B"] == 368_640 and b["slot"]["w2"]["pitch_16B_ok"] is False
    t = b["today_C"]
    assert abs(t["sf_hbm_GiB"] - 18.158) < 0.002, t
    r20 = [r for r in b["beside"] if r["pool_spare"] == 20][0]
    r24 = [r for r in b["beside"] if r["pool_spare"] == 24][0]
    assert abs(r20["sf_hbm_GiB"] - 12.978) < 0.002 and abs(r20["hbm_vs_C_GiB"] + 5.181) < 0.002
    assert abs(r24["hbm_vs_C_GiB"] + 2.379) < 0.002 and r24["free_after_build_GiB_E"] > 3.5
    e = d["estimate"]
    assert e["layers_left"] == 23 and 18.4 < e["beside_pool_spare_20_E"]["central"] < 18.9
    c = d["contract"]
    assert c["separate"]["rows_read"].get("hot.scale", 0) > 0 and "hot.scale" not in c["beside"]["rows_read"]
    assert c["beside"]["hot_sf_groups_untouched"] and c["beside"]["rows_read"]["cold.weight"] == 9
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)

