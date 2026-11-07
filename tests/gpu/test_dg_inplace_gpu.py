"""GPU tests of T9a-2 (runtime/dg_inplace.py, `moe_impl="dg-inplace"`). Skipped without CUDA/tilelang (Mac); on the
box a missing/broken deep_gemm FAILS (it is what is under test).

  * packed SF: runtime/dg_inplace.pack_sf == deep_gemm's own get_mn_major_tma_aligned_packed_ue8m0_tensor of the fp32
    values (equal values; strides printed);
  * the Triton staging kernel (t9a2_stage_inplace) == its contract (dg_inplace.stage_inplace_emulated) on the device,
    real row sizes (EVEN path) and odd ones, cold sources in pinned host memory (UVA), both scale modes;
  * contiguous over a [H+S] pool vs masked over the gathered staging, same bytes and activations: bitwise, or the
    mismatch rate and max ulp are printed and max ulp must be <= 1 (synthetic bytes always; real layer-0 bytes when
    the checkpoint is mounted at /ref);
  * mini checkpoints: dg-inplace static-eager runs (finite), self-replay equal, no host sync; graph == static-eager,
    replay #2, fills-on recapture, relaunch equal; top-1 stream over prefill + 10 steps == dg-masked's (printed on a
    mismatch, not loosened); per-layer MoE output dg-inplace vs dg-masked on identical inputs (mismatch %, max ulp);
  * informative: mini4 graph step-cell p50, dg-inplace vs dg-masked.

Run on the box (orchestrator) exactly as tests/gpu/test_dg_moe_gpu.py (same image, mounts, DG_JIT_CACHE_DIR), with
`-v /path/to/checkpoint:/ref:ro` for the real-bytes check:
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_dg_inplace_gpu.py
"""
from __future__ import annotations

import os
import random
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_dg_moe_gpu import _finite, _gen, _moe_inputs, _need_dg  # noqa: E402
from test_static_mini_gpu import N_STEPS, _assert_equal_runs, _first_light, _ids, _model, _run  # noqa: E402

DGI = "dg-inplace"
DGM = "dg-masked"
E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn


def _dg():
    from runtime.dg_moe import import_deep_gemm

    return import_deep_gemm()


# ------------------------------------------------------------------------------------------------ layout + kernel
def test_pack_sf_equals_deepgemm_packing():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime.dg_moe import E8M0_F32

    dg = _dg()
    codes = torch.randint(100, 150, (4, 256, 40), dtype=torch.int32, device="cuda").to(torch.uint8)
    ours = DI.pack_sf(codes)
    f32 = E8M0_F32.cuda()[codes.long()]
    res = {}
    for name, arg, want in (("3d", f32, ours), ("2d", f32[0], ours[0])):
        try:
            theirs = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(arg)
            res[name] = (torch.equal(theirs, want), tuple(theirs.stride()), tuple(want.stride()))
        except Exception as e:
            res[name] = repr(e)[:160]
    print(f"  pack_sf vs deep_gemm get_mn_major_tma_aligned_packed_ue8m0_tensor: {res}")
    assert any(isinstance(v, tuple) and v[0] for v in res.values()), res


def _layer(H, S, N, KH, KS, E, cold_host=True, seed=0):
    from runtime import dg_inplace as DI

    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    hot = rng.sample(range(E), H)
    cold = [e for e in range(E) if e not in hot]
    full = {p: torch.zeros(H + S, N[p], KH[p], dtype=torch.int8, device="cuda") for p in DI.PROJS}
    pool_s = {p: torch.randint(110, 140, (H, N[p], KS[p]), generator=g).to(torch.uint8).cuda() for p in DI.PROJS}
    for p in DI.PROJS:
        full[p][:H] = torch.randint(-128, 128, (H, N[p], KH[p]), generator=g).to(torch.int8).cuda()
    host = {}
    for e in cold:
        for p in DI.PROJS:
            w = torch.randint(-128, 128, (N[p], KH[p]), generator=g).to(torch.int8)
            s = torch.randint(110, 140, (N[p], KS[p]), generator=g).to(torch.uint8)
            host[(p, e)] = (w.pin_memory(), s.pin_memory()) if cold_host else (w.cuda(), s.cuda())
    ptab = torch.zeros(6, E, dtype=torch.int64)
    gtab = torch.full((E,), -1, dtype=torch.int32)
    for i, p in enumerate(DI.PROJS):
        for e in range(E):
            if e in hot:
                s_ = hot.index(e)
                ptab[i, e], ptab[3 + i, e] = full[p][s_].data_ptr(), pool_s[p][s_].data_ptr()
            else:
                ptab[i, e], ptab[3 + i, e] = host[(p, e)][0].data_ptr(), host[(p, e)][1].data_ptr()
    for s_, e in enumerate(hot):
        gtab[e] = s_
    reg = {}
    for p in DI.PROJS:
        for s_ in range(H):
            reg[full[p][s_].data_ptr()] = full[p][s_].reshape(-1).view(torch.uint8)
            reg[pool_s[p][s_].data_ptr()] = pool_s[p][s_].reshape(-1)
    for (p, e), (w, s) in host.items():
        reg[w.data_ptr()] = w.reshape(-1).view(torch.uint8)
        reg[s.data_ptr()] = s.reshape(-1)
    return dict(H=H, S=S, hot=hot, cold=cold, full=full, pool_s=pool_s, host=host, ptab=ptab.cuda(),
                gtab=gtab.cuda(), reg=reg)


def _stage_case(N, KH, KS, all_scales, cold_host):
    from runtime import dg_inplace as DI
    from runtime.static_kernels import stage_inplace

    H, S, E, A = 20, 6, 32, 128
    outs = []
    for kernel in (True, False):
        L = _layer(H, S, N, KH, KS, E, cold_host, seed=7)
        C = {p: KS[p] // 4 for p in DI.PROJS}
        sf = {p: DI.packed_sf_zeros(H + S, N[p], C[p], "cuda") for p in DI.PROJS}
        dsts = [L["full"][p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
        dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cuda")
        ids = torch.tensor(sorted(L["hot"][:3] + L["cold"][:3]), dtype=torch.int64, device="cuda")
        mi = torch.full((S * A,), -1, dtype=torch.int32, device="cuda")
        ids_out = torch.zeros(S, dtype=torch.int64, device="cuda")
        args = (L["ptab"], ids, L["gtab"], dtab, dsts, mi, H, A, N["w1"] * KH["w1"],
                ((N["w1"], C["w1"]), (N["w2"], C["w2"])), all_scales, ids_out)
        if kernel:
            stage_inplace(*args)
        else:
            reg = L["reg"]
            DI.stage_inplace_emulated(lambda a: reg[a].cuda(), *args)
        torch.cuda.synchronize()
        outs.append([t.clone() for t in dsts] + [mi, ids_out])
    bad = [i for i, (a, b) in enumerate(zip(*outs)) if not torch.equal(a, b)]
    assert not bad, ("stage kernel vs contract", N, KH, all_scales, cold_host, bad)


def test_stage_kernel_equals_contract_real_rows():
    _need_dg()
    N = {"w1": 2304, "w3": 2304, "w2": 5120}
    KH = {"w1": 2560, "w3": 2560, "w2": 1152}
    KS = {"w1": 160, "w3": 160, "w2": 72}
    for all_scales in (True, False):
        _stage_case(N, KH, KS, all_scales, cold_host=True)
    print("  t9a2_stage_inplace == contract: real rows (EVEN path), pinned-host cold sources, both scale modes")


def test_stage_kernel_equals_contract_odd_rows():
    _need_dg()
    N = {"w1": 260, "w3": 260, "w2": 520}
    KH = {"w1": 128, "w3": 128, "w2": 64}
    KS = {"w1": 8, "w3": 8, "w2": 4}
    for all_scales in (True, False):
        _stage_case(N, KH, KS, all_scales, cold_host=False)
    print("  t9a2_stage_inplace == contract: odd rows (masked path), several scale tiles")


# ------------------------------------------------------------------------------------------------ contiguous vs masked
def _contig_vs_masked(get_expert, hot, cold, tag):
    """Pool [H+S] from get_expert(e, p) -> (int8 [N, K/2], uint8 [N, K/32]); 6 routed ids with 2 cold; the masked
    path (gather -> fp32 SF -> masked) vs stage -> contiguous over the pool; returns per-projection comparisons."""
    from runtime import dg_inplace as DI
    from runtime.dg_moe import E8M0_F32
    from runtime.static_kernels import gather_tables, stage_inplace

    dg = _dg()
    if hasattr(dg, "use_deterministic_algorithms"):
        dg.use_deterministic_algorithms(True)
    H, S, A = len(hot), 6, 128
    E = max(hot + cold) + 1
    w0, s0 = get_expert(hot[0], "w1")
    w2_, s2_ = get_expert(hot[0], "w2")
    NK = {"w1": (w0.shape[0], w0.shape[1] * 2), "w3": (w0.shape[0], w0.shape[1] * 2),
          "w2": (w2_.shape[0], w2_.shape[1] * 2)}
    full = {p: torch.zeros(H + S, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device="cuda") for p in DI.PROJS}
    pool_s = {p: torch.zeros(H, NK[p][0], NK[p][1] // 32, dtype=torch.uint8, device="cuda") for p in DI.PROJS}
    for s_, e in enumerate(hot):
        for p in DI.PROJS:
            w, s = get_expert(e, p)
            full[p][s_].copy_(w)
            pool_s[p][s_].copy_(s)
    host = {(p, e): tuple(t.contiguous().pin_memory() for t in get_expert(e, p)) for e in cold for p in DI.PROJS}
    dummy = torch.zeros(full["w1"][0].numel(), dtype=torch.int8, device="cuda")
    ptab = torch.zeros(6, E, dtype=torch.int64)
    gtab = torch.full((E,), -1, dtype=torch.int32)
    for i, p in enumerate(DI.PROJS):
        for e in range(E):
            if e in hot:
                ptab[i, e], ptab[3 + i, e] = full[p][hot.index(e)].data_ptr(), pool_s[p][hot.index(e)].data_ptr()
            elif e in cold:
                ptab[i, e], ptab[3 + i, e] = host[(p, e)][0].data_ptr(), host[(p, e)][1].data_ptr()
            else:
                ptab[i, e] = ptab[3 + i, e] = dummy.data_ptr()
    for s_, e in enumerate(hot):
        gtab[e] = s_
    ptab, gtab = ptab.cuda(), gtab.cuda()
    ids = torch.tensor(sorted(hot[1:5] + cold[:2]), dtype=torch.int64, device="cuda")
    C = {p: NK[p][1] // 128 for p in DI.PROJS}
    sf = {p: DI.packed_sf_zeros(H + S, NK[p][0], C[p], "cuda") for p in DI.PROJS}
    g = torch.Generator().manual_seed(1)
    DIM, INTER = NK["w1"][1], NK["w2"][1]

    def act(R, K):
        x = torch.randn(R, K, generator=g)
        x = (x.view(R, K // 32, 32) * torch.exp2(torch.randint(-6, 7, (R, K // 32, 1), generator=g).float()))
        x = x.view(R, K).to(torch.bfloat16).float()
        amax = x.view(R, K // 32, 32).abs().amax(-1).clamp_min(1e-4)
        code = (torch.ceil(torch.log2(amax / 448.0)) + 127).clamp(1, 254).to(torch.uint8)  # ue8m0 per 32
        sc = E8M0_F32[code.long()]
        q = (x.view(R, K // 32, 32) / sc.unsqueeze(-1)).clamp(-448, 448).view(R, K).to(FP8)
        return q.cuda(), sc.cuda()

    qa, sa = act(1, DIM)
    q2, s2 = act(S, INTER)
    xb = {1: (torch.zeros(S, A, DIM, dtype=FP8, device="cuda"), torch.ones(S, A, DIM // 32, device="cuda")),
          2: (torch.zeros(S, A, INTER, dtype=FP8, device="cuda"), torch.ones(S, A, INTER // 32, device="cuda"))}
    xb[1][0][:, 0], xb[1][1][:, 0] = qa, sa
    xb[2][0][:, 0], xb[2][1][:, 0] = q2, s2
    stg = {p: torch.zeros(S, *full[p].shape[1:], dtype=torch.int8, device="cuda") for p in DI.PROJS}
    sf32 = {p: torch.zeros(S, NK[p][0], NK[p][1] // 32, device="cuda") for p in DI.PROJS}
    mdsts = [stg[p] for p in DI.PROJS] + [sf32[p] for p in DI.PROJS]
    mdtab = torch.tensor([t.data_ptr() for t in mdsts], dtype=torch.int64, device="cuda")
    row_w, row_s = full["w1"][0].numel(), pool_s["w1"][0].numel()
    gather_tables(ptab, ids, mdtab, mdsts, row_w, row_s, True)
    idsts = [full[p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    idtab = torch.tensor([t.data_ptr() for t in idsts], dtype=torch.int64, device="cuda")
    mi = torch.full((S * A,), -1, dtype=torch.int32, device="cuda")
    dg.set_mk_alignment_for_contiguous_layout(A)
    stage_inplace(ptab, ids, gtab, idtab, idsts, mi, H, A, row_w,
                  ((NK["w1"][0], C["w1"]), (NK["w2"][0], C["w2"])), True)
    res = {}
    for p in DI.PROJS:
        w = 2 if p == "w2" else 1
        dm = torch.zeros(S, A, NK[p][0], dtype=torch.bfloat16, device="cuda")
        dc = torch.full((S * A, NK[p][0]), float("nan"), dtype=torch.bfloat16, device="cuda")
        dg.m_grouped_fp8_fp4_gemm_nt_masked(xb[w], (stg[p], sf32[p]), dm, torch.ones(S, dtype=torch.int32,
                                                                                        device="cuda"), 8,
                                            recipe_a=(1, 32), recipe_b=(1, 32))
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous((xb[w][0].view(S * A, -1), xb[w][1].view(S * A, -1)),
                                                (full[p], sf[p]), dc, mi, recipe_a=(1, 32), recipe_b=(1, 32))
        torch.cuda.synchronize()
        res[p] = DI.compare_bf16(dc.view(S, A, -1)[:, 0], dm[:, 0])
    print(f"  {tag}: contiguous-in-place vs masked-over-staging (2 of 6 cold): {res}")
    for p, r in res.items():
        assert r["nonfinite"] == 0 and r["max_ulp"] <= 1, (tag, p, r)
    return res


def test_contiguous_vs_masked_synthetic_bytes():
    _need_dg()
    g = torch.Generator().manual_seed(3)
    cache = {}

    def get(e, p):
        if (e, p) not in cache:
            N, K = (2304, 5120) if p != "w2" else (5120, 2304)
            cache[(e, p)] = (torch.randint(-128, 128, (N, K // 2), generator=g).to(torch.int8),
                             torch.randint(118, 132, (N, K // 32), generator=g).to(torch.uint8))
        return cache[(e, p)]

    _contig_vs_masked(get, list(range(0, 20)), [20, 21, 22], "synthetic layer (20 hot + 6 spare)")


def test_contiguous_vs_masked_real_layer0_bytes():
    _need_dg()
    root = os.environ.get("DSV41_REF_CKPT", "/ref")
    if not os.path.isdir(root) or not any(f.endswith(".safetensors") for f in os.listdir(root)):
        raise unittest.SkipTest(f"no checkpoint at {root}")
    sys.path.insert(0, os.path.join(REPO, "bench"))
    from dg_contiguous_spike import Ckpt

    ck = Ckpt(root)
    _contig_vs_masked(lambda e, p: ck.expert(0, e, p), list(range(0, 24)), [100, 211, 300],
                      "real layer-0 bytes (24 hot + 6 spare)")


# ------------------------------------------------------------------------------------------------ mini end to end
def _inplace_selfreplay(kind):
    _need_dg()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=DGI)
        ids = _ids(16, 5)
        a = _run(gen, "static", ids)
        _finite(a, f"{kind} dg-inplace static")
        b = _run(gen, "static", ids)
        _assert_equal_runs(b, a, f"{kind}: dg-inplace static-eager self-replay")
        st = gen.static
        assert st.moe_impl == DGI and st.dg is not None
        print(f"  {kind}: dg-inplace static-eager self-replay equal; {st.dg.describe()}")
        gen.set_mode("static")
        gen.prefill(ids)
        gen.step(int(gen.next_token_tensor().reshape(-1)[0]), len(ids))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for p in (len(ids) + 1, len(ids) + 2):
                gen.step(gen.next_token_tensor(), p)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        gen.reset_static()
        del gen


def test_inplace_static_selfreplay_mini():
    _inplace_selfreplay("mini")


def test_inplace_static_selfreplay_mini4():
    _inplace_selfreplay("mini4")


def _inplace_graph_vs_static(kind):
    _need_dg()
    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=DGI)
        ids = _ids(16, 7)
        st = _run(gen, "static", ids)
        g1 = _run(gen, "graph", ids)
        assert gen.static.captured and gen.static.captured_fills is False and gen.static.moe_impl == DGI
        _assert_equal_runs(g1, st, f"{kind}: dg-inplace graph vs static-eager")
        g2 = _run(gen, "graph", ids)
        _assert_equal_runs(g2, g1, f"{kind}: dg-inplace graph replay#2 vs #1")
        gen.capture(fills=True)
        g3 = _run(gen, "graph", ids)
        _assert_equal_runs(g3, g1, f"{kind}: dg-inplace fills-on capture vs fills-off")
        gen.reset_static()
        g4 = _run(gen, "graph", ids)
        _assert_equal_runs(g4, g1, f"{kind}: dg-inplace relaunch vs replay#1")
        print(f"  {kind}: dg-inplace graph == static-eager; replay x2, fills-on recapture, relaunch all equal")
        gen.reset_static()
        del gen


def test_inplace_graph_equals_static_mini():
    _inplace_graph_vs_static("mini")


def test_inplace_graph_equals_static_mini4():
    _inplace_graph_vs_static("mini4")


def _inplace_vs_masked(kind):
    _need_dg()
    from runtime import dg_inplace as DI

    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=DGM)
        ids = _ids(16, 3)
        m = _run(gen, "static", ids)
        gen.set_moe_impl(DGI)
        i = _run(gen, "static", ids)
        _finite(i, f"{kind} dg-inplace static")
        if i[1] != m[1]:
            print(f"  {kind}: TOKEN STREAMS DIFFER\n    dg-masked  {m[1]}\n    dg-inplace {i[1]}")
        assert i[1] == m[1], (kind, "dg-inplace vs dg-masked top-1 stream", m[1], i[1])
        dl = [float((a - b).abs().max()) for a, b in zip(i[0], m[0])]
        print(f"  {kind}: top-1 stream == dg-masked over prefill + {N_STEPS} steps; max |dlogit| per position {dl}")
        gen.set_mode("static")
        xs = _moe_inputs(gen, model, ids)              # the inputs the dg-inplace step saw
        st = gen.static
        with torch.inference_mode(), st._active():
            y_in = [st.moes[l](x.clone()).clone() for l, x in enumerate(xs)]
        gen.set_moe_impl(DGM)
        gen.set_mode("static")
        gen.prefill(ids)
        st2 = gen.static_step()
        with torch.inference_mode(), st2._active():
            y_m = [st2.moes[l](x.clone()).clone() for l, x in enumerate(xs)]
        torch.cuda.synchronize()
        for l, (a, b) in enumerate(zip(y_in, y_m)):
            r = DI.compare_bf16(a.reshape(-1), b.reshape(-1))
            print(f"  {kind} L{l}: MoE output dg-inplace vs dg-masked: equal {r['equal']}, mismatch "
                  f"{r['mismatch_frac'] * 100:.4f} %, max ulp {r['max_ulp']}")
            assert r["nonfinite"] == 0, (kind, l, r)
        gen.reset_static()
        del gen


def test_inplace_vs_masked_tokens_and_moe_mini():
    _inplace_vs_masked("mini")


def test_inplace_vs_masked_tokens_and_moe_mini4():
    _inplace_vs_masked("mini4")


def test_inplace_graph_step_cell_informative():
    _need_dg()
    fl = _first_light()
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=DGM, mode="graph")
        ids = _ids(16, 8)
        p50, toks = {}, {}
        for impl in (DGM, DGI):
            gen.set_moe_impl(impl)
            gen.set_mode("graph")
            sc = fl.step_cell(gen, ids, 40, stats_from=8)
            p50[impl], toks[impl] = sc["ms_per_step"]["p50"], sc["token_ids"][:N_STEPS + 1]
        print(f"  mini4 graph step cell p50: dg-masked {p50[DGM]:.3f} ms, dg-inplace {p50[DGI]:.3f} ms; "
              f"first {N_STEPS + 1} tokens equal: {toks[DGM] == toks[DGI]}")
        gen.reset_static()
        del gen


TESTS = [test_pack_sf_equals_deepgemm_packing, test_stage_kernel_equals_contract_real_rows,
         test_stage_kernel_equals_contract_odd_rows, test_contiguous_vs_masked_synthetic_bytes,
         test_contiguous_vs_masked_real_layer0_bytes, test_inplace_static_selfreplay_mini,
         test_inplace_static_selfreplay_mini4, test_inplace_graph_equals_static_mini,
         test_inplace_graph_equals_static_mini4, test_inplace_vs_masked_tokens_and_moe_mini,
         test_inplace_vs_masked_tokens_and_moe_mini4, test_inplace_graph_step_cell_informative]

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
