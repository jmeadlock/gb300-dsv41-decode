"""GPU gate of the P1 gracetier first build (docs/P1-GRACETIER-DESIGN.md section 5 file 7), run in the mini-test
window by bench/box/gracetier_minitest.sh before the probe. Skipped without CUDA (Mac); on the box a missing
deep_gemm FAILS the MegaMoE tests (it is what is under test).

  * VMM (runtime/gracetier_vmm.py): one physical scratch behind two VAs -- a write through one is read through the
    other; the straddle granule is private per VA; torch ops on the views; release in reverse order.
  * gt_stage on the device == stage_emulated (sources in a registered Grace HostBuffer, read over UVA), plain and
    VMM-backed destinations, mini and real shapes; graph-captured: new groups written in place, the replay stages the
    new experts (device-side selection).
  * gt_read on the device == read_emulated.
  * MegaMoE G = 384 at real shapes: VMM-backed L1 / L2 vs the same bytes in plain allocations (the M1 gate), and
    after gt_stage of cold groups vs an all-HBM 384-group copy (the M4 gate on synthetic bytes): torch.equal.
  * A3 residency (docs/HANDOFF-A3-RESIDENCY.md, runtime/gracetier_res.py), the checkpoint's layer 10 (needs the
    checkpoint at $DSV41_MODEL_DIR, default /ref -- its absence FAILS on the box): an all-HBM G = 384 reference at the
    boot map vs the pool after 32 random swaps (the scratch dirtied through a second layer's VA before every barrier
    and every stage), MegaMoE output for random routings at T = 1 / 6 / 16 / 64: torch.equal, eager and through ONE
    CUDA graph captured before the swaps; the device counter == the picks per expert; the audit; restore_map brings
    the boot bytes back.

Run on the box inside the library image (the runner does):
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_gracetier_gpu.py
"""
from __future__ import annotations

import os
import random
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime import gracetier_pool as GP  # noqa: E402
from runtime import gracetier_stage as GS  # noqa: E402
from runtime import gracetier_vmm as GV  # noqa: E402

MiB = 1024 ** 2


def _need_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")


def _need_triton():
    _need_cuda()
    if not GS.HAVE_TRITON:
        raise unittest.SkipTest("triton missing: run inside vllm/vllm-openai:v0.31.0")


def _need_dg():
    _need_triton()
    from runtime.dg_moe import import_deep_gemm

    return import_deep_gemm()    # raises (-> FAIL) on the box if deep_gemm cannot be imported without vllm


def _dev():
    GV.ensure_context(torch.cuda.current_device())
    return torch.device("cuda", torch.cuda.current_device())


def _pool(D, I, C, seed):
    """A registered pool of C random experts (layer 0)."""
    pool = GP.GracePool(GP.PoolLayout({0: C}, D, I), name=f"gt-test-{seed}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool.buf.cpu.random_(0, 256, generator=g)
    pool.register()
    return pool


def _resolver(pool, dev):
    base = pool.buf.ptr

    def resolve(addr):
        return pool.buf.cpu[addr - base:].to(dev)

    return resolve


# ------------------------------------------------------------------------------------------------ VMM
def test_vmm_alias_straddle_and_release():
    _need_cuda()
    dev = _dev()
    ar = GV.VmmArena(dev.index)
    gran = ar.gran
    plan = GV.TensorPlan("t", G=8, H=3, B=gran * 5 // 8, gran=gran)
    assert plan.straddle
    try:
        sh = ar.alloc(plan.shared_bytes, "shared")
        A = GV.GroupedVA(ar, plan, sh, map_hot=True, name="A")
        B = GV.GroupedVA(ar, plan, sh, map_hot=False, name="B")
        tA = A.tensor(torch.uint8, (plan.va_bytes,))
        assert tA.data_ptr() == A.va and tA.is_cuda
        pat = torch.randint(0, 256, (plan.shared_bytes,), dtype=torch.uint8, device=dev)
        tA[plan.shared_va_off:].copy_(pat)
        torch.cuda.synchronize()
        assert torch.equal(B.bytes_view(plan.shared_va_off, plan.shared_bytes), pat)
        so = GV.align_down(plan.hot_bytes, gran)
        B.bytes_view(so, gran).fill_(0x11)
        tA[so:so + gran].fill_(0x22)
        torch.cuda.synchronize()
        assert bool((B.bytes_view(so, gran) == 0x11).all()) and bool((tA[so:so + gran] == 0x22).all())
        assert torch.equal(B.bytes_view(plan.shared_va_off, plan.shared_bytes), pat)
        # torch ops through the view: arithmetic on an int32 view of the shared range
        v = A.tensor(torch.int32, (plan.va_bytes // 4,))[plan.shared_va_off // 4:]
        v.fill_(3)
        assert int(v.long().sum()) == 3 * v.numel()
        del tA, v
    finally:
        torch.cuda.synchronize()
        ar.release()
    ops = [e[0] for e in ar.log]
    assert ops.index("unmap") < ops.index("free") < ops.index("release")
    print(f"  VMM gran {gran} B: write via A == read via B, straddle private, release in reverse order")


# ------------------------------------------------------------------------------------------------ gt_stage / gt_read
def _stage_case(D, I, C, H, vmm: bool, seed=3):
    dev = _dev()
    pool = _pool(D, I, C, seed)
    G = H + C
    src = pool.src_table(0, dev)
    groups = torch.tensor([[H, 0, H + C - 1, -1, 1, H + 2], [H + 2, H, 2, -1, -1, H + 1]], dtype=torch.int64,
                          device=dev).reshape(-1).contiguous()
    outs, ar = [], None
    try:
        for kernel in (False, True):
            if vmm and kernel:
                ar = GV.VmmArena(dev.index)
                lv = GV.LayerVMM(ar, {0: H}, G, D, I)
                l1, l2 = lv.l1(0), lv.l2(0)
            else:
                l1 = torch.empty(G, 2 * I, D // 2, dtype=torch.int8, device=dev)
                l2 = torch.empty(G, D, I // 2, dtype=torch.int8, device=dev)
            gg = torch.Generator(device=dev).manual_seed(seed + 1)
            l1.view(torch.uint8).random_(0, 256, generator=gg)
            l2.view(torch.uint8).random_(0, 256, generator=gg)
            cnt = torch.zeros(C, dtype=torch.int32, device=dev)
            if kernel:
                GS.gt_stage(groups, H, src, GS.make_dtab(l1, l2), cnt, pool.layout.B1, pool.layout.B2)
            else:
                GS.stage_emulated(groups, H, src, [l1, l2], cnt, _resolver(pool, dev))
            torch.cuda.synchronize()
            outs.append([l1.clone(), l2.clone(), cnt.clone()])
            del l1, l2
        for i, (a, b) in enumerate(zip(*outs)):
            assert torch.equal(a, b), (D, I, vmm, i)
    finally:
        torch.cuda.synchronize()
        if ar is not None:
            ar.release()
        pool.release()
    return outs[1][2].tolist()


def test_gt_stage_equals_contract_plain_and_vmm():
    _need_triton()
    for D, I in ((256, 128), (5120, 2304)):
        for vmm in (False, True):
            cnt = _stage_case(D, I, 5, 7, vmm)
            assert cnt == [2, 1, 2, 0, 1], cnt
    print("  gt_stage == stage_emulated (UVA sources; plain and VMM destinations; mini and real shapes)")


def test_gt_stage_is_graph_capturable_with_device_selection():
    _need_triton()
    dev = _dev()
    D, I, C, H = 5120, 2304, 6, 4
    pool = _pool(D, I, C, 11)
    try:
        G = H + C
        src = pool.src_table(0, dev)
        l1 = torch.zeros(G, 2 * I, D // 2, dtype=torch.int8, device=dev)
        l2 = torch.zeros(G, D, I // 2, dtype=torch.int8, device=dev)
        dtab = GS.make_dtab(l1, l2)
        groups = torch.tensor([H, 0, 1, -1], dtype=torch.int64, device=dev)
        cnt = torch.zeros(C, dtype=torch.int32, device=dev)
        fn = lambda: GS.gt_stage(groups, H, src, dtab, cnt, pool.layout.B1, pool.layout.B2)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            fn()
        groups.copy_(torch.tensor([H + 3, H + 5, 2, -1], dtype=torch.int64, device=dev))
        l1.zero_()
        l2.zero_()
        g.replay()
        torch.cuda.synchronize()
        host1, host2 = pool.host_l1(0), pool.host_l2(0)
        for c in (3, 5):
            assert torch.equal(l1[H + c].cpu(), host1[c]) and torch.equal(l2[H + c].cpu(), host2[c])
        assert not bool(l1[H].any()) and not bool(l1[H + 1].any())
        del g
    finally:
        torch.cuda.synchronize()
        pool.release()
    print("  gt_stage captured in a CUDA graph: replay stages the experts named by the new in-place groups")


def test_gt_read_equals_contract():
    _need_triton()
    dev = _dev()
    D, I, C = 5120, 2304, 3
    pool = _pool(D, I, C, 5)
    try:
        src = pool.src_table(0, dev)
        for nprog, bw in ((8, 1024), (152, 1024), (3, 4096)):
            out = torch.zeros(nprog * bw, dtype=torch.int64, device=dev)
            GS.gt_read(src, C, pool.layout.B1, out, nprog=nprog, block_words=bw)
            torch.cuda.synchronize()
            want = GS.read_emulated(src, C, pool.layout.B1 // 8, nprog, bw, _resolver(pool, "cpu"))
            assert torch.equal(out.view(nprog, bw).cpu(), want), (nprog, bw)
    finally:
        pool.release()
    print("  gt_read == read_emulated (8 / 152 / 3 programs)")


# ------------------------------------------------------------------------------------------------ MegaMoE gates
def _mega_world(dg, dev, seed=7):
    import gracetier_minitest as GT
    from runtime import dg_mega as DM

    G, H, D, I, k = 384, 255, 5120, 2304, 6
    gd = torch.Generator(device=dev).manual_seed(seed)
    codes = {p: torch.empty(G, n, kk // 32, dtype=torch.uint8, device=dev).random_(118, 125, generator=gd)
             for p, (n, kk) in (("w1", (I, D)), ("w3", (I, D)), ("w2", (D, I)))}
    sf1, sf2 = GP.sf_banks(codes)
    symm = DM.make_symm_buffer(dg, G, 64, k, D, I, 0)
    y = torch.zeros(64, D, dtype=torch.bfloat16, device=dev)
    rng = random.Random(seed)
    T = 6
    q, sc = GT.quant_rows(GT.synth_x(T, D, torch.Generator(device="cpu").manual_seed(seed)).to(dev))
    w = torch.tensor(GT.synth_weights(T, k, rng), dtype=torch.float32, device=dev)
    return dict(G=G, H=H, D=D, I=I, k=k, T=T, sf1=sf1, sf2=sf2, symm=symm, y=y, q=q, sc=sc, w=w, rng=rng, GT=GT,
                gd=gd)


def test_megamoe_g384_vmm_equals_plain():
    dg = _need_dg()
    dev = _dev()
    W = _mega_world(dg, dev)
    GT = W["GT"]
    ar = GV.VmmArena(dev.index)
    try:
        lv = GV.LayerVMM(ar, {0: W["H"]}, W["G"], W["D"], W["I"])
        l1, l2 = lv.l1(0), lv.l2(0)
        l1.view(torch.uint8).random_(0, 256, generator=W["gd"])
        l2.view(torch.uint8).random_(0, 256, generator=W["gd"])
        p1, p2 = l1.clone(), l2.clone()
        ids = [sorted(W["rng"].sample(range(W["G"]), W["k"])) for _ in range(W["T"])]
        ids[0] = sorted({W["H"] - 1, W["H"], W["G"] - 1, 0, 100, 300})
        topk = torch.tensor(ids, dtype=torch.long, device=dev)
        res = []
        for a1, a2 in ((l1, l2), (p1, p2)):
            GT.set_inputs(W["symm"], topk, W["w"], W["q"], W["sc"])
            GT.mega_call(dg, W["y"], a1, W["sf1"], a2, W["sf2"], W["symm"], 10.0, False)
            torch.cuda.synchronize()
            res.append(W["y"][:W["T"]].clone())
        cmp = GT.compare_bits(res[0], res[1])
        assert cmp["equal"], cmp
        del l1, l2
    finally:
        torch.cuda.synchronize()
        ar.release()
    print(f"  MegaMoE G=384 on VMM L1/L2 == plain allocation (torch.equal): {cmp}")


def test_megamoe_staged_cold_equals_all_hbm():
    dg = _need_dg()
    dev = _dev()
    W = _mega_world(dg, dev, seed=9)
    GT = W["GT"]
    G, H, D, I = W["G"], W["H"], W["D"], W["I"]
    C = G - H
    pool = _pool(D, I, C, 13)
    ar = GV.VmmArena(dev.index)
    try:
        lv = GV.LayerVMM(ar, {0: H, 1: H}, G, D, I, map_hot={0: True, 1: False})
        l1, l2 = lv.l1(0), lv.l2(0)
        l1[:H].view(torch.uint8).random_(0, 256, generator=W["gd"])
        l2[:H].view(torch.uint8).random_(0, 256, generator=W["gd"])
        ref1 = torch.empty(G, 2 * I, D // 2, dtype=torch.int8, device=dev)
        ref2 = torch.empty(G, D, I // 2, dtype=torch.int8, device=dev)
        ref1[:H].copy_(l1[:H])
        ref2[:H].copy_(l2[:H])
        ref1[H:].copy_(pool.host_l1(0))
        ref2[H:].copy_(pool.host_l2(0))
        src = pool.src_table(0, dev)
        wrong = pool.src_table(0, dev, slots=[(c + 1) % C for c in range(C)])
        dtab_b = torch.tensor([lv.va[1]["l1"].va, lv.va[1]["l2"].va], dtype=torch.int64, device="cpu").to(dev)
        all_cold = torch.arange(H, G, dtype=torch.long, device=dev)
        for c_req in (1, 2, 4, 8):
            ids = GT.synth_routing(W["T"], c_req, list(range(H)), list(range(H, G)), W["k"], W["rng"])
            topk = torch.tensor(ids, dtype=torch.long, device=dev)           # identity row_map: ids are groups
            GS.gt_stage(all_cold, H, wrong, dtab_b, None, pool.layout.B1, pool.layout.B2)   # dirty via layer 1
            GT.set_inputs(W["symm"], topk, W["w"], W["q"], W["sc"])
            GS.gt_stage(W["symm"].topk_idx[:W["T"]].reshape(-1), H, src, GS.make_dtab(l1, l2), None,
                        pool.layout.B1, pool.layout.B2)
            GT.mega_call(dg, W["y"], l1, W["sf1"], l2, W["sf2"], W["symm"], 10.0, True)
            torch.cuda.synchronize()
            y_b = W["y"][:W["T"]].clone()
            GT.set_inputs(W["symm"], topk, W["w"], W["q"], W["sc"])
            GT.mega_call(dg, W["y"], ref1, W["sf1"], ref2, W["sf2"], W["symm"], 10.0, True)
            torch.cuda.synchronize()
            cmp = GT.compare_bits(y_b, W["y"][:W["T"]])
            assert cmp["equal"], (c_req, cmp)
        del l1, l2
    finally:
        torch.cuda.synchronize()
        ar.release()
        pool.release()
    print("  MegaMoE G=384 with gt_stage-filled shared scratch (dirtied by a 2nd layer VA) == all-HBM copy, "
          "c = 1/2/4/8 (torch.equal)")


# ------------------------------------------------------------------------------------------------ A3 residency
RES_LAYER = 10
RES_SWAPS = 32
RES_TS = (1, 6, 16, 64)
RES_ROUTINGS = 4               # random routings per T (the first names moved experts on purpose)


def _res_mismatch(tag, cmp):
    """A difference is reported as a class (rows exact / max |d|, the way mini-test M6 did), never hidden."""
    return (f"A3 residency {tag}: MegaMoE output differs from the all-HBM reference: rows exact "
            f"{cmp['rows_exact']}/{cmp['rows']}, elements equal {cmp['frac_bytes_equal']}, max|d| {cmp['max_abs']}, "
            f"nonfinite {cmp['nonfinite']}")


def test_residency_swaps_equal_all_hbm_layer10():
    dg = _need_dg()
    import numpy as np

    import gracetier_minitest as GT
    from runtime import gracetier_res as GR

    model_dir = os.environ.get("DSV41_MODEL_DIR", "/ref")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"A3 residency gate needs the checkpoint (layer {RES_LAYER}) at {model_dir} "
                           f"(DSV41_MODEL_DIR); the runner mounts it at /ref")
    a = GT.build_parser().parse_args(["numerics", "--layer", str(RES_LAYER), "--model-dir", model_dir])
    S = GT.LayerSetup(a, dg, want_ref=True, want_stock=False)
    L, H, G, C, k, dev = S.L, S.H, S.G, S.C, S.k, S.dev
    out = {}
    try:
        # the reference: all-HBM weights + SF banks in the BOOT group order; it never changes
        ref_sf1, ref_sf2 = S.sf1.clone(), S.sf2.clone()
        row_boot = S.row_long.clone()
        # the pool side: the tables the server keeps at fixed addresses, and the manager over them
        row_map = S.row_dev.clone()
        g2e = torch.tensor(GR.g2e_list(S.row), dtype=torch.int32, device="cpu").to(dev)
        cnt = torch.zeros(L + 1, G, dtype=torch.int32, device=dev)
        va = S.vmm.va[L]
        view = GR.LayerView(L, H, row_map, g2e, S.l1, S.l2, S.sf1, S.sf2, S.pool.cuda_l1(L, dev),
                            S.pool.cuda_l2(L, dev), cuts1=(va["l1"].plan.shared_va_off,),
                            cuts2=(va["l2"].plan.shared_va_off,))
        cfg = GR.ResConfig(mode="adaptive", freeze=True, dir=None)
        t0 = time.time()
        mgr = GR.GtResManager({L: view}, {L: S.hot}, {L: S.cold}, cnt, cfg, log=lambda m: print("   ", m, flush=True))
        out["manager_s"] = round(time.time() - t0, 2)
        rng = random.Random(20261109)
        nrng = np.random.default_rng(20261109)
        ids_buf = torch.full((a.max_tokens, k), -1, dtype=torch.long, device=dev)
        want_cnt = [0] * G

        def step():
            """One layer-call as the server makes it (fastpath_gracetier._forward): groups = row_map[ids] on the
            device, gt_stage with the counter, MegaMoE. Reads only fixed-address tensors: capturable once."""
            S.symm.topk_idx.copy_(torch.where(ids_buf >= 0, row_map[ids_buf.clamp(min=0)].to(ids_buf.dtype), ids_buf))
            GS.gt_stage(S.symm.topk_idx.reshape(-1), H, S.src_tab, S.dtab, S.cnt, S.b1, S.b2, g2e=g2e, ecnt=cnt[L])
            S.mega_vmm(True)

        def load(ids, T):
            """Inputs for T rows (weights, activations; the SymmBuffer's id rows are written by step())."""
            g = torch.Generator(device="cpu").manual_seed(rng.randrange(1 << 30))
            q, sc = GT.quant_rows(GT.synth_x(T, S.D, g).to(dev))
            w = torch.tensor(GT.synth_weights(T, k, rng), dtype=torch.float32, device=dev)
            topk = torch.tensor(ids, dtype=torch.long, device=dev)
            GT.set_inputs(S.symm, topk, w, q, sc)                     # topk_idx is overwritten by the caller
            return topk, w, q, sc

        def reference(topk, w, q, sc, T):
            GT.set_inputs(S.symm, row_boot[topk], w, q, sc)
            GT.mega_call(dg, S.y, S.ref_l1, ref_sf1, S.ref_l2, ref_sf2, S.symm, S.lim, True)
            torch.cuda.synchronize()
            return S.y[:T].clone()

        def run(ids, T, how, graph=None):
            topk, w, q, sc = load(ids, T)
            y_ref = reference(topk, w, q, sc, T)
            GT.set_inputs(S.symm, topk, w, q, sc)
            ids_buf.fill_(-1)
            ids_buf[:T].copy_(topk)
            S.dirty()                                                 # the scratch holds the WRONG experts
            if graph is None:
                step()
            else:
                graph.replay()
            torch.cuda.synchronize()
            for r in ids:
                for e in r:
                    want_cnt[e] += 1
            cmp = GT.compare_bits(S.y[:T], y_ref)
            assert cmp["equal"], _res_mismatch(f"T={T} {how}", cmp)
            return cmp

        def routings(T, moved):
            outs = []
            for i in range(RES_ROUTINGS):
                ids = [rng.sample(range(G), k) for _ in range(T)]
                if i == 0 and moved:                                  # name experts that changed tier
                    for r, e in zip(ids, rng.sample(moved, min(len(moved), T))):
                        if e not in r:
                            r[0] = e
                outs.append(ids)
            return outs

        # 0. the boot map: equal before any swap, eager; then capture the step ONCE (T_max rows, as the server's
        #    graphs are captured before any swap happens)
        for T in RES_TS:
            for ids in routings(T, []):
                run(ids, T, "boot map, eager")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                step()
            out["graph"] = "captured before the swaps"
        except Exception as e:  # noqa: BLE001 - reported in the result line; the eager arms still gate
            graph = None
            out["graph"] = f"NOT captured ({type(e).__name__}: {str(e)[:160]}); eager only"
        torch.cuda.synchronize()
        cnt.zero_()
        want_cnt[:] = [0] * G
        for T in RES_TS:
            run(routings(T, [])[0], T, "boot map, graph" if graph is not None else "boot map, eager", graph)

        # 1. 32 random swaps through the queue and the barrier (the serving path), <= 4 per barrier
        barriers = 0
        while mgr.stats["swaps"] < RES_SWAPS:
            mgr.queue_random(min(cfg.swaps_per_barrier, RES_SWAPS - mgr.stats["swaps"] - len(mgr.state.pending)), nrng)
            S.dirty()
            mgr.barrier()
            barriers += 1
            assert barriers < 64 * RES_SWAPS, "barriers apply no swaps"
        torch.cuda.synchronize()
        moved = sorted(e for e in range(G) if mgr.row[L][e] != S.row[e])
        out.update(swaps=mgr.stats["swaps"], barriers=barriers, experts_displaced=len(moved),
                   copies=mgr.stats["copies"], stale=mgr.state.stale)
        assert mgr.stats["swaps"] == RES_SWAPS and moved, out
        assert row_map.cpu().tolist() == mgr.row[L] and g2e.cpu().tolist() == GR.g2e_list(mgr.row[L])
        assert sorted(mgr.row[L]) == list(range(G))
        assert torch.equal(S.src_tab, S.pool.src_table(L, dev))        # the source table never changes

        # 2. after the swaps: the same captured graph and eager calls, every T, vs the untouched reference
        for T in RES_TS:
            for i, ids in enumerate(routings(T, moved)):
                use = graph if i % 2 == 0 else None
                run(ids, T, "after 32 swaps, " + ("eager" if use is None else "graph"), use)
        got = cnt[L].cpu().tolist()
        assert got == want_cnt, ("the device counter is not the picks per expert",
                                 [(e, got[e], want_cnt[e]) for e in range(G) if got[e] != want_cnt[e]][:8])
        out["counted_picks"] = sum(got)
        assert not bool(cnt[:L].any())

        # 3. the audit (every moved expert + 64 random), then restore_map, the audit again, the boot bytes
        a1 = mgr.audit(random_per_layer=64)
        assert a1["mismatches"] == 0 and a1["moved"] >= len(moved), a1
        rec = mgr.apply_ctrl({"freeze": True, "restore_map": True})
        n = 0
        while mgr.state.pending:
            S.dirty()
            mgr.barrier()
            n += 1
            assert n < 4096
        torch.cuda.synchronize()
        assert mgr.at_boot_map() and row_map.cpu().tolist() == S.row
        assert torch.equal(S.l1[:H], S.ref_l1[:H]) and torch.equal(S.l2[:H], S.ref_l2[:H])
        assert torch.equal(S.pool.host_l1(L), S.ref_l1[H:].cpu()) and torch.equal(S.pool.host_l2(L), S.ref_l2[H:].cpu())
        assert torch.equal(S.sf1, ref_sf1) and torch.equal(S.sf2, ref_sf2)
        a2 = mgr.audit(random_per_layer=64)
        assert a2["mismatches"] == 0, a2
        for T in RES_TS:
            run(routings(T, moved)[0], T, "after restore_map", graph)
        out.update(audit_checked=(a1["checked"], a2["checked"]), restore_queued=rec["restore_queued"],
                   restore_barriers=n, audit_ms=(a1.get("ms"), a2.get("ms")), stream_changes=mgr.stats["stream_changes"])
        del graph, view, mgr
    finally:
        torch.cuda.synchronize()
        S.close()
    print(f"  A3 residency, layer {RES_LAYER} (H = {H}): {RES_SWAPS} swaps -> MegaMoE G=384 == all-HBM reference at "
          f"T = {RES_TS} (torch.equal; eager, and one CUDA graph: {out.get('graph')}); counter == picks; audit 0; "
          f"restore_map -> boot bytes. {out}")


TESTS = [test_vmm_alias_straddle_and_release, test_gt_stage_equals_contract_plain_and_vmm,
         test_gt_stage_is_graph_capturable_with_device_selection, test_gt_read_equals_contract,
         test_megamoe_g384_vmm_equals_plain, test_megamoe_staged_cold_equals_all_hbm,
         test_residency_swaps_equal_all_hbm_layer10]

if __name__ == "__main__":
    import traceback

    results = {}
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            results[t.__name__] = "PASS"
        except unittest.SkipTest as e:
            results[t.__name__] = f"SKIP ({e})"
        except Exception:
            traceback.print_exc()
            results[t.__name__] = "FAIL"
        print(f"{results[t.__name__]:4s} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
