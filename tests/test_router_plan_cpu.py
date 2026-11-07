"""Mac CPU tests of M2-T15 (docs/HANDOFF-M2-T15-ROUTER-PLAN.md): the router + R-row plan in one launch
(runtime/router_plan.py, runtime/static_kernels.py m2_router_plan). `.venv/bin/python -m pytest -q tests/test_router_plan_cpu.py`

  1. the tie rule's host form: the 32-lane SmallBitonicSort as written in SortUtils.cuh (an independent
     transliteration here) == router_plan.bitonic_small; the census of all 6^6 value patterns quoted in the module
     docstring (always descending; 6,988 stable / 15,392 reverse-stable / 24,276 neither -> no closed form, the network
     must run on ties); gatherTopK's set (radix keys: -0 < +0, NaN on top; lowest-index ties); topk_cuda_host ==
     torch.topk whenever the answer is unique;
  2. the real @triton.jit body under tests/triton_shim.py == the mirrors bit for bit: ids + weights for R in {1, 2, 6, 8},
     E up to 384, k in {2, 6, 8}, both softplus forms, explicit weight-sum orders, tie-heavy / all-equal / repeated /
     NaN / +-0 / beyond-threshold rows (the tie path runs the network: checked to have run); the GEMV form (fixed
     T9b-5 order, several BN / U, ragged E) == fixed_dot_torch + the same tail, ticket back to 0, multi-program
     ticket order; the PLAN form == moe_rows.dedup_fused (the separate m1_dedup_warp launch, shim) on every buffer for
     no inpl / masked_m / inpl v1 / v2 contiguous / psum / masked, step SF on/off, overflow;
  3. the verbatim copies of the reduction helpers; the switch ($DSV41_ROUTER_IMPL) and the 1-row glue wiring (the
     6-row step: tests/test_router_plan_rows_cpu.py);
  4. the recording hook writes what the replay bench reads.
Not checked here: Triton compilation, torch.topk / F.softplus on CUDA, speed (tests/gpu/test_router_plan_gpu.py and
bench/router_plan_bench.py on the box).
"""
from __future__ import annotations

import inspect
import itertools
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import triton_shim  # noqa: E402
from runtime import fused_glue_r2 as R2  # noqa: E402
from runtime import moe_rows as MR  # noqa: E402
from runtime import router_plan as RP  # noqa: E402


@pytest.fixture(scope="module")
def SK():
    """runtime/static_kernels.py under the shim, its PLAN tail bound to the shim copy of moe_rows.m1_dedup_warp."""
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"),
                                             "static_kernels_m2t15_under_shim")
    km = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "moe_rows.py"), "moe_rows_m2t15_under_shim")
    km.KERNELS_ON_CPU = True
    mod._m2_bind_dedup(km.m1_dedup_warp)
    mod._KM = km
    return mod


@pytest.fixture(scope="module")
def KR():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue_r2.py"),
                                             "fused_glue_r2_m2t15_under_shim")
    mod.KERNELS_ON_CPU = True
    return mod


def _od(order):
    return {"order": order, "fin": "mul", "exact": True, "name": R2.order_name(order, "mul")}


def _bits(t):
    t = t.contiguous()
    if t.dtype.is_floating_point:
        return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
    return t


def _eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


# ================================================================================ 1. the tie rule
def _bitonic_transliteration(keys, vals, n, size=32):
    """SortUtils.cuh bitonicSort, thread by thread (independent of router_plan.bitonic_small): keys compared by `>`
    (GTOp keys), lanes >= n invalid with key 0 / value 0."""
    K = [int(x) for x in keys[:n]] + [0] * (size - n)
    V = [int(x) for x in vals[:n]] + [0] * (size - n)
    OK = [True] * n + [False] * (size - n)

    def swap(i, j, d):
        s = (K[i] > K[j] and OK[i]) or not OK[j]
        if s == d:
            K[i], K[j] = K[j], K[i]
            V[i], V[j] = V[j], V[i]
            OK[i], OK[j] = OK[j], OK[i]

    sz = 2
    while sz < size:
        st = sz // 2
        while st > 0:
            for tid in range(size // 2):
                flag = (tid & (sz // 2)) != 0
                pos = 2 * tid - (tid & (st - 1))
                swap(pos, pos + st, flag)
            st //= 2
        sz *= 2
    st = size // 2
    while st > 0:
        for tid in range(size // 2):
            pos = 2 * tid - (tid & (st - 1))
            swap(pos, pos + st, False)
        st //= 2
    return V


def test_bitonic_matches_transliteration_and_tie_census():
    g = np.random.default_rng(0)
    for _ in range(300):
        n = int(g.integers(1, 9))
        keys = g.integers(0, 4, n)
        vals = np.arange(n)
        assert list(RP.bitonic_small(keys, vals, n)) == _bitonic_transliteration(keys, vals, n)
    stable = rev = other = 0
    for pat in itertools.product(range(6), repeat=6):
        keys = np.array(pat, dtype=np.int64)
        out = list(RP.bitonic_small(keys, np.arange(6), 6)[:6])
        assert [pat[i] for i in out] == sorted(pat, reverse=True)
        if out == sorted(range(6), key=lambda i: (-pat[i], i)):
            stable += 1
        elif out == sorted(range(6), key=lambda i: (-pat[i], -i)):
            rev += 1
        else:
            other += 1
    assert (stable, rev, other) == (6988, 15392, 24276)


def test_radix_and_gt_keys():
    v = np.array([-np.inf, -2.0, -1e-45, -0.0, 0.0, 1e-45, 3.0, np.inf, np.nan, -np.nan], np.float32)
    rk = RP.radix_key(v)
    assert list(np.argsort(rk[:8], kind="stable")) == list(range(8))       # strictly increasing, -0 < +0
    assert rk[3] < rk[4] and rk[8] == rk[9] == 0xFFFFFFFF
    gk = RP.gt_key(v)
    assert gk[3] == gk[4] and gk[8] == gk[9] == 0x7FFFFFFF and all(np.diff(gk[:8])[[0, 1, 4, 5, 6]] > 0)


def test_selection_is_the_first_seen_set():
    row = np.array([1.0, 2.0, 2.0, 0.5, 2.0, 2.0, 3.0, 2.0, 2.0], np.float32)
    got = RP.topk_cuda_host(row, 4)                      # 3.0 above; ties at 2.0 -> indices 1, 2, 4
    assert sorted(got.tolist()) == [1, 2, 4, 6] and got[0] == 6
    row = np.array([0.0, -0.0, 0.0, -0.0, 1.0], np.float32)
    assert sorted(RP.topk_cuda_host(row, 3).tolist()) == [0, 2, 4]   # +0 above -0 in the radix order
    row = np.array([1.0, np.nan, 5.0, np.nan], np.float32)
    assert sorted(RP.topk_cuda_host(row, 2).tolist()) == [1, 3]      # NaNs are the largest


def test_topk_cuda_host_equals_torch_when_unique():
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        E = int(torch.randint(8, 400, (1,), generator=g))
        k = int(torch.randint(1, 9, (1,), generator=g))
        row = torch.randn(E, generator=g)
        assert torch.equal(RP.topk_cuda_torch(row, k), row.topk(k).indices)


# ================================================================================ 2. the kernel under the shim
def _rows(g, R_, E, case):
    if case == "normal":
        raw = torch.randn(R_, E, generator=g) * 2.0
        bias = torch.randn(E, generator=g) * 0.05
    elif case == "ties":
        raw = torch.randint(-8, 9, (R_, E), generator=g).float() * 0.25
        bias = torch.where(torch.rand(E, generator=g) < 0.5, 0.0, 0.25)
    elif case == "equal":
        raw = torch.full((R_, E), 0.5)
        bias = torch.zeros(E)
    elif case == "repeat":
        raw = (torch.randn(1, E, generator=g) * 3.0).repeat(R_, 1)
        bias = torch.randn(E, generator=g) * 0.05
    elif case == "edges":
        raw = torch.randn(R_, E, generator=g) * 30.0
        raw[:, ::5] = -200.0
        raw[:, 1::7] = 20.0
        bias = torch.where(torch.rand(E, generator=g) < 0.5, -0.0, 0.0)
    elif case == "mixed":   # top-k = distinct values AND ties, and ties straddling the k-th place
        raw = torch.full((R_, E), -5.0)
        for r in range(R_):
            n = min(E, 10)
            pat = torch.tensor([3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5][:n])[torch.randperm(n, generator=g)]
            raw[r, torch.randperm(E, generator=g)[:n]] = pat
        bias = torch.zeros(E)
    else:   # nan
        raw = torch.randn(R_, E, generator=g)
        raw[:, 3] = float("nan")
        bias = torch.randn(E, generator=g) * 0.05
    return raw.contiguous(), bias.contiguous()


def _ws(R_, E, k, gemv=False):
    ws = {"sc": torch.zeros(R_, E), "bsc": torch.zeros(R_, E), "tk": torch.zeros(R_, k, dtype=torch.long),
          "net": torch.zeros(RP._np2(R_), 32, dtype=torch.long), "ticket": torch.zeros(1, dtype=torch.int32)}
    if gemv:
        ws["raw"] = torch.zeros(R_, E)
    return ws


ORDERS = [("vec", 1, 4, True), ("vec", 8, 1, False), ("unroll", 2, 2, True), ("vec", 4, 2, False)]


@pytest.mark.parametrize("R_,E,k", [(1, 384, 6), (6, 384, 6), (2, 64, 6), (8, 40, 8), (6, 16, 2), (3, 7, 6)])
@pytest.mark.parametrize("case", ["normal", "ties", "mixed", "equal", "repeat", "edges", "nan"])
def test_kernel_route_equals_mirror(SK, R_, E, k, case):
    g = torch.Generator().manual_seed(hash((R_, E, k, case)) % 2 ** 31)
    raw, bias = _rows(g, R_, E, case)
    for t, softplus in enumerate((True, False)):
        src = raw if softplus else F.softplus(raw)
        ordd = _od(ORDERS[(t + R_) % len(ORDERS)])
        ws = _ws(R_, E, k)
        ids = torch.full((R_, k), -9, dtype=torch.long)
        w = torch.full((R_, k), -9.0)
        n0 = len(SK.m2_router_plan.launches)
        SK.router_plan_launch(src.clone(), bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                              eps=1e-20, softplus=softplus)
        assert len(SK.m2_router_plan.launches) == n0 + 1 and SK.m2_router_plan.launches[-1]["grid"] == (1,)
        ri, rw = RP.route_tail_torch(raw, bias, k, ordd, 1.5, softplus=RP.softplus_np if softplus else F.softplus,
                                     topk=RP.topk_cuda_torch)
        if case == "nan":   # NaN weights: payload bits are not a contract; the ids and NaN-ness are
            assert _eq(ids, ri) and torch.equal(torch.isnan(w), torch.isnan(rw))
            assert torch.equal(torch.nan_to_num(w, nan=7.0), torch.nan_to_num(rw, nan=7.0))
        else:
            assert _eq(ids, ri) and _eq(w, rw), (case, softplus)
        # the scratch holds sc / bsc / torch.topk's order (what the network or the rank formula produced)
        sp = RP.softplus_np(raw) if softplus else F.softplus(raw)
        nn = lambda t: torch.nan_to_num(t, nan=7.0)  # noqa: E731
        assert _eq(nn(ws["sc"]), nn(sp.sqrt())) and _eq(nn(ws["bsc"]), nn(sp.sqrt() + bias))
        assert torch.equal(ws["tk"], RP.topk_cuda_torch(sp.sqrt() + bias, k))


def test_kernel_tie_path_runs_the_network(SK, monkeypatch):
    """A row whose k selected values tie takes the network branch (and only then); its order is the network's."""
    calls = []
    real = SK._m2_bitonic.fn

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(SK._m2_bitonic, "fn", spy)
    E, k = 32, 6
    for case, want in (("normal", 0), ("equal", 1)):
        raw, bias = _rows(torch.Generator().manual_seed(3), 2, E, case)
        ws = _ws(2, E, k)
        ids, w = torch.empty(2, k, dtype=torch.long), torch.empty(2, k)
        n0 = len(calls)
        SK.router_plan_launch(raw, bias, ws, ids, w, R=2, E=E, K=k, order=R2.order_args(_od(ORDERS[0])), rs=1.0,
                              eps=1e-20, softplus=True)
        assert len(calls) - n0 == want
    # all-equal row: the network's (non-stable) order of the 6 lowest indices
    assert ws["tk"][0].tolist() == RP.topk_cuda_host(np.full(E, float(np.sqrt(RP.softplus_np(
        torch.tensor([0.5])).item())), np.float32), k).tolist()
    assert ws["tk"][0].tolist() != list(range(6))


@pytest.mark.parametrize("R_,E,D,cfg", [(1, 40, 256, {"BN": 4, "U": 3}), (6, 40, 256, {"BN": 8, "U": 4}),
                                        (6, 16, 128, {"BN": 1, "U": 1}), (5, 24, 192, {"BN": 2, "U": 2})])
def test_kernel_gemv_equals_fixed_order_oracle(SK, R_, E, D, cfg):
    g = torch.Generator().manual_seed(R_ * 1000 + E)
    x = (torch.randn(R_, D, generator=g) * 0.5).to(torch.bfloat16).float()
    gw = (torch.randn(E, D, generator=g) * 0.1).to(torch.bfloat16)
    bias = torch.randn(E, generator=g) * 0.05
    k = 6
    kb = 64
    ordd = _od(ORDERS[1])
    ws = _ws(R_, E, k, gemv=True)
    ids, w = torch.empty(R_, k, dtype=torch.long), torch.empty(R_, k)
    for rep in range(2):                                  # the ticket is back to 0: a second launch works the same
        SK.router_plan_launch(ws["raw"], bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                              eps=1e-20, softplus=True, x=x, gw=gw, kb=kb, gemv_cfg=cfg)
        assert int(ws["ticket"]) == 0
        assert SK.m2_router_plan.launches[-1]["grid"] == (-(-E // cfg["BN"]),)
        raw = R2.fixed_dot_torch(x, gw, kb)
        assert _eq(ws["raw"], raw)
        ri, rw = RP.route_tail_torch(raw, bias, k, ordd, 1.5, softplus=RP.softplus_np, topk=RP.topk_cuda_torch)
        assert _eq(ids, ri) and _eq(w, rw)


def test_kernel_gemv_tail_runs_once_in_the_last_program(SK, monkeypatch):
    seen = []
    real = SK._m2_tail.fn

    def spy(*a, **kw):
        seen.append(tuple(triton_shim._PID))
        return real(*a, **kw)
    monkeypatch.setattr(SK._m2_tail, "fn", spy)
    R_, E, D, k = 2, 24, 128, 6
    g = torch.Generator().manual_seed(9)
    x = torch.randn(R_, D, generator=g).to(torch.bfloat16).float()
    gw = torch.randn(E, D, generator=g).to(torch.bfloat16)
    ws = _ws(R_, E, k, gemv=True)
    SK.router_plan_launch(ws["raw"], torch.zeros(E), ws, torch.empty(R_, k, dtype=torch.long), torch.empty(R_, k),
                          R=R_, E=E, K=k, order=R2.order_args(_od(ORDERS[0])), rs=1.0, eps=1e-20, softplus=True,
                          x=x, gw=gw, kb=64, gemv_cfg={"BN": 4, "U": 1})
    assert seen == [(E // 4 - 1, 0, 0)]


def _plan_cases():
    return [("none", None), ("mm", None), ("v1", None),
            ("v2", ("contiguous", False)), ("v2", ("contiguous", True)), ("v2", ("psum", False)),
            ("v2", ("psum", True)), ("v2", ("masked", False)), ("v2", ("masked", True))]


@pytest.mark.parametrize("kind,lay", _plan_cases())
@pytest.mark.parametrize("case", ["normal", "ties", "mixed", "repeat"])
def test_kernel_plan_equals_dedup_fused(SK, kind, lay, case):
    """PLAN: the plan kernel's buffers == the separate dedup launch (moe_rows.dedup_fused -> m1_dedup_warp, shim) on
    the same route, every buffer and every DeepGEMM input; the returned dict == dedup_fused's."""
    from test_moe_rows_t10_cpu import _fresh, _gtab, _inpl

    KM = SK._KM
    g = torch.Generator().manual_seed(hash((kind, lay, case)) % 2 ** 31)
    for t, (R_, E, k) in enumerate(((6, 24, 6), (6, 40, 6), (4, 16, 3), (8, 64, 8))):
        raw, bias = _rows(g, R_, E, case)
        P = R_ * k
        BR = 128 if lay is not None and lay[0] == "masked" else (R_ + t % 3)
        inpl0 = None
        if kind == "v1":
            gtab = _gtab(g, E, "random")
            inpl0 = {"gtab": gtab, "H": int((gtab >= 0).sum()), "S": 2, "dummy": torch.tensor(1), "has_dummy": True}
        elif kind == "v2":
            gtab = _gtab(g, E, ["random", "all_cold", "zero_cold", "random"][t])
            inpl0 = _inpl(gtab, 1 + t % 3, lay[0], lay[1], BR, R_, k, G_extra=t % 2)
        ordd = _od(ORDERS[t % len(ORDERS)])
        outs = []
        for mine in (True, False):
            inpl = _fresh(inpl0) if kind == "v2" else inpl0
            mm = torch.full((P,), -7, dtype=torch.int32) if kind in ("mm", "v2", "v1") else None
            ids_s = torch.full((P,), -7, dtype=torch.long)
            uv = torch.full((P,), -7, dtype=torch.long)
            buf = MR.DedupBuffers(R_, k, "cpu")
            ws = _ws(R_, E, k)
            ids, w = torch.empty(R_, k, dtype=torch.long), torch.empty(R_, k)
            if mine:
                a = RP.dedup_args(buf, R_, k, BR, mm, ids_s, uv, inpl, ids)
                SK.router_plan_launch(raw, bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                                      eps=1e-20, softplus=True, plan=a)
                m = RP.plan_out_dict(buf, mm, inpl, a)
            else:
                SK.router_plan_launch(raw, bias, ws, ids, w, R=R_, E=E, K=k, order=R2.order_args(ordd), rs=1.5,
                                      eps=1e-20, softplus=True)
                m = KM.dedup_fused(buf, ids, w, BR, mm=mm, ids_step=ids_s, uval=uv, inpl=inpl, impl="warp")
            outs.append((m, mm, ids_s, uv, inpl, ids, w))
        (m1, mm1, i1, u1, p1, id1, w1), (m0, mm0, i0, u0, p0, id0, w0) = outs
        assert sorted(m1) == sorted(m0)
        for key, v in m0.items():
            if isinstance(v, torch.Tensor):
                assert _eq(m1[key], v), (key, kind, lay, case, t)
            else:
                assert m1[key] == v, key
        assert _eq(id1, id0) and _eq(w1, w0)
        for a_, b_ in ((mm1, mm0), (i1, i0), (u1, u0)):
            assert a_ is None and b_ is None or _eq(a_, b_)
        if kind == "v2":
            for n in ("mi", "gl", "ovf"):
                assert p1[n] is None and p0[n] is None or _eq(p1[n], p0[n]), n
        # and the base mapping is the definition's (the torch mirror of the dedup)
        ref = MR.dedup_routes(id0, w0, BR)
        for key in ("uniq", "counts", "valid", "slot", "rank", "prow", "pw"):
            assert _eq(m1[key], ref[key]), key


def test_dedup_args_follow_launch_warp():
    """dedup_args mirrors moe_rows._launch_warp's argument list (same order as m1_dedup_warp's signature)."""
    src = inspect.getsource(MR._launch_warp)
    for frag in ("buf.uniq, buf.counts, buf.valid, buf.slot, buf.rank, buf.prow, buf.pw, buf.arow, buf.posmat",
                 "mm if mm is not None else buf.counts", "ids_step if ids_step is not None else buf.uniq",
                 "uval if uval is not None else buf.valid", "buf.ids_stage, buf.sidx, tags_t, buf.wl",
                 "gl if gl is not None else buf.scr", "ovf if ovf is not None else buf.scr, buf.scr, dmy.reshape(-1)",
                 "HAS_TAGS=has_in and (not v2 or lay == \"contiguous\")", "STEP_SF=bool(v2 and inpl.get(\"step_sf\"))"):
        assert frag in src, frag


def test_reduction_helpers(SK, KR):
    """_r2_offs is fused_glue_r2's verbatim; the row-parallel finish equals order_sum_torch (the family's
    definition, which t9b2r2_router_post's _r2_finish implements) on every shape the router can be calibrated to."""
    import textwrap

    a_ = textwrap.dedent(inspect.getsource(SK._r2_offs.fn))
    b_ = textwrap.dedent(inspect.getsource(KR._r2_offs.fn))
    assert a_ == b_
    g = torch.Generator().manual_seed(4)
    for K_ in (2, 3, 6, 8):
        for order in R2.order_candidates(K_, 6)[:24] + ORDERS:
            fam, T_, V_, dec = order
            if T_ * V_ > 64:
                continue
            ordd = _od(order)
            sc = torch.rand(6, 40, generator=g) * torch.exp2(torch.randint(-6, 7, (6, 40), generator=g).float())
            idx = torch.stack([torch.randperm(40, generator=g)[:K_] for _ in range(6)])
            ids, w = torch.empty(6, K_, dtype=torch.long), torch.empty(6, K_)
            # route_tail with the identity softplus on sc^2 (sqrt(sc^2) may not be sc: feed the scores via raw so the
            # kernel's sc is sqrt(raw)); compare against the definition on the kernel's own sc
            raw = sc
            ws = _ws(6, 40, K_)
            SK.router_plan_launch(raw, torch.zeros(40), ws, ids, w, R=6, E=40, K=K_, order=R2.order_args(ordd),
                                  rs=1.5, eps=1e-20, softplus=False)
            ri, rw = R2.router_post_torch(ws["tk"], ws["sc"], ordd, 1e-20, 1.5)
            assert _eq(ids, ri) and _eq(w, rw), order


def test_softplus_probe_kernel(SK):
    x = RP.softplus_probe_inputs(torch.Generator().manual_seed(0), 1 << 12)
    got = SK.softplus_probe(x)
    assert _eq(got, RP.softplus_np(x))
    assert got[x > 20].equal(x[x > 20])


# ================================================================================ 3. the switch and the glue
def test_router_impl_env(monkeypatch):
    monkeypatch.delenv(RP.ROUTER_IMPL_ENV, raising=False)
    assert RP.router_impl() == "fused"
    for v in RP.ROUTER_IMPLS:
        monkeypatch.setenv(RP.ROUTER_IMPL_ENV, v)
        assert RP.router_impl() == v
    monkeypatch.setenv(RP.ROUTER_IMPL_ENV, "topk")
    with pytest.raises(ValueError):
        RP.router_impl()


@pytest.fixture
def unhook():
    yield
    RP.uninstall_rows_hook()


from test_fused_glue_r2_cpu import _equal_to_ref, glue_emu  # noqa: E402,F401
from test_gemv_cpu import _static_run as _static_run1, emu, gemv_emu  # noqa: E402,F401


@pytest.mark.parametrize("impl", ["fused", "plan", "plan-gemv"])
def test_one_row_glue_impls_on_cpu_equal_ref(glue_emu, monkeypatch, unhook, impl):
    """1 row, CPU: the default builds no plan; plan / plan-gemv build one, report it (CPU: the fused path is its
    mirror, the kernels are unchecked here), and the step stays equal to ref."""
    from runtime.static_step import StaticStep

    monkeypatch.setenv(RP.ROUTER_IMPL_ENV, impl)
    model, pool, _ = glue_emu("mini4")
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    d = st.glue.describe()["r2"]["router_impl"]
    if impl == "fused":
        assert st.glue.rplan is None and d["impl"] == "fused"
    else:
        assert d["impl"] == impl and d["status"] == "on" and d["checks"]["device"].startswith("cpu")
        assert not st.glue.rplan.verified and RP._HOOK["orig"] is None      # 1 row: no hook
    assert _equal_to_ref(model, pool, st, "dg-masked", "gemv", list(range(30, 40)))


# ================================================================================ 4. recording
def test_dump_hook_writes_replayable_records(glue_emu, monkeypatch, tmp_path, unhook):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(RP.DUMP_ENV, str(tmp_path))
    monkeypatch.delenv(RP.ROUTER_IMPL_ENV, raising=False)
    model, pool, _ = glue_emu("mini4")
    RP._REC["r"] = None
    st = StaticStep(model, pool, max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", glue_impl="fused",
                    _allow_cpu=True)
    _static_run1(model, st, list(range(10, 14)), 2)
    RP.recorder().flush()
    st.detach()
    dumps = sorted(p for p in os.listdir(tmp_path) if p.startswith("router-dump"))
    biases = sorted(p for p in os.listdir(tmp_path) if p.startswith("router-bias"))
    assert dumps and len(biases) == len(model.layers)
    recs = [r for p in dumps for r in torch.load(os.path.join(tmp_path, p))]
    assert recs and any("x" in r for r in recs)
    b = {torch.load(os.path.join(tmp_path, p))["layer"]: torch.load(os.path.join(tmp_path, p)) for p in biases}
    for r in recs:
        meta = b[r["layer"]]
        ri, rw = RP.route_tail_torch(r["raw"], meta["bias"], meta["topk"], NATIVE_ORD, meta["route_scale"])
        assert torch.equal(ri, r["ids"]) and torch.equal(rw, r["w"])
    RP._REC["r"] = None


NATIVE_ORD = {"order": R2.NATIVE, "fin": "mul", "exact": True, "name": R2.NATIVE}
