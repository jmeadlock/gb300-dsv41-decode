#!/usr/bin/env python3
"""dg_rows_spike.py -- M1-T10-MOE item 1 probe (box): the 6-row MoE's routed GEMM call shape on layer 0's REAL pool
bytes. 6 rows x top-6 = 36 pairs over 22 unique experts (19 hot + 3 cold staged into the spare slots by the R-row
staging launch), the exact layer-0 pool the runtime builds ([H+S, N, K/2] int8 + the pool-owned packed SF).

Call shapes (every one through the runtime's own layout code: moe_rows.rows_layout / dg_inplace.DGInplaceBuffers'
gemm_rows conventions), each timed as 3 GEMMs per layer (w1, w3, w2; CUDA-graph replay) and compared bitwise on every
real output row against today's call (contiguous, ALIGN 128, one 128-row block per pair slot = 36 x 128 = 4,608 A
rows of which 36 are real):
  * contiguousNN   the same blocks at ALIGN = NN (the BLOCK_M cap: 64, 32, 16 and the library's theoretical value);
  * psum           contiguous with use_psum_layout over the WHOLE pool's G = H+S groups (grouped_layout[g] = group
                   g's end row; start = previous end aligned up to the mk alignment; empty groups have no blocks;
                   the swap-AB kernels size the last block of a group to align(rows, UMMA_STEP_N) -- gemm.cuh:202);
  * masked         m_grouped_fp8_fp4_gemm_nt_masked with the POOL as B (G = H+S groups, masked_m[g] = rows of
                   group g, 0 for unused groups, max_m = 128 -- the ABI memo's trap), expected_m = 6, packed SF.
  * mega           MegaMoE (fp8_fp4_mega_moe / get_block_m_for_mega_moe): availability and signature only -- it fuses
                   the SwiGLU + requant epilogue, a numerics change outside this item's bit-equal bar.
Also timed: the dedup kernels (m1_dedup_warp vs m1_dedup_routes, 6x6 pairs, dg-inplace extras) and the staging
(t9a2_stage_inplace over 36 slots with all scales = today's "step" store, vs m1_stage_rows over the work list: S
cold slots with the pool SF store, 36 with "step").

Run on the box like bench/dg_contiguous_spike.py (library image, checkpoint at /ref, repo at /work, DG JIT cache):
  python3 bench/dg_rows_spike.py [--aligns 64,32,16] [--out docs/dg-rows-spike-<date>.json]
deep_gemm is imported from vllm/third_party with vllm never imported (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime import dg_inplace as DI  # noqa: E402
from runtime import moe_rows as MR  # noqa: E402
from runtime.dg_moe import import_deep_gemm  # noqa: E402

FP8 = torch.float8_e4m3fn
PROJS = ("w1", "w3", "w2")
DIM, INTER, TOPK, E = 5120, 2304, 6, 384
NK = {"w1": (INTER, DIM), "w3": (INTER, DIM), "w2": (DIM, INTER)}   # (N, K)
ROWS, SPARE, UNIQUE, COLD = 6, 8, 22, 3
RECIPE = dict(recipe_a=(1, 32), recipe_b=(1, 32))


def routing(hot, cold, seed):
    """6 rows x 6 distinct experts covering exactly UNIQUE experts (UNIQUE - COLD hot + COLD cold): a random cover."""
    rng = random.Random(seed)
    experts = sorted(rng.sample(hot, UNIQUE - COLD) + rng.sample(cold, COLD))
    for _ in range(100000):
        rows = [sorted(rng.sample(experts, TOPK)) for _ in range(ROWS)]
        if len({e for r in rows for e in r}) == UNIQUE:
            return experts, torch.tensor(rows)
    raise RuntimeError("no cover found")


def acts(n, K, g):
    """fp8 values with power-of-two per-32 scales (the act_quant contract; numerics are compared across call shapes
    on identical inputs, so synthetic activations suffice -- the weights are the real bytes)."""
    q = (torch.randn(n, K, generator=g) * 3).clamp(-448, 448).to(FP8)
    s = torch.exp2(torch.randint(-6, 7, (n, K // 32), generator=g).float())
    return q, s


def main(argv=None):
    from dg_contiguous_spike import Ckpt, graph_us

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ref")
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--aligns", default="64,32,16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", f"dg-rows-spike-{time.strftime('%Y%m%d')}.json"))
    a = ap.parse_args(argv)
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "errors": [], "checks": {}, "variants": {}}

    def step(name, fn, into=None):
        try:
            r = fn()
            (into if into is not None else out["checks"])[name] = r
            print("OK  ", name, json.dumps(r, default=str)[:400], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            (into if into is not None else out["checks"])[name] = {"error": tb[-1500:]}
            out["errors"].append(name)
            print("FAIL", name, tb[-1500:], flush=True)
            return None

    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(a.seed)
    dg = import_deep_gemm()
    out["deep_gemm"] = {"file": getattr(dg, "__file__", None), "version": getattr(dg, "__version__", None)}
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    from runtime import static_kernels as SK

    # ---------------------------------------------------------------- the API surface (item 1 a / c questions)
    def api():
        r = {}
        for name in ("m_grouped_fp8_fp4_gemm_nt_contiguous", "m_grouped_fp8_fp4_gemm_nt_masked",
                     "get_mk_alignment_for_contiguous_layout", "set_mk_alignment_for_contiguous_layout",
                     "get_theoretical_mk_alignment_for_contiguous_layout", "fp8_fp4_mega_moe",
                     "get_block_m_for_mega_moe", "transform_weights_for_mega_moe"):
            f = getattr(dg, name, None)
            if f is None:
                r[name] = None
                continue
            try:
                sig = str(inspect.signature(f))
            except (TypeError, ValueError):
                sig = None
            r[name] = {"signature": sig, "doc": (getattr(f, "__doc__", None) or "")[:600]}
        r["alignment_default"] = dg.get_mk_alignment_for_contiguous_layout()
        for em in (1, 6, 36):
            try:
                r[f"theoretical_alignment_em{em}"] = int(dg.get_theoretical_mk_alignment_for_contiguous_layout(em))
            except Exception as e:
                r[f"theoretical_alignment_em{em}"] = repr(e)[:200]
        f = getattr(dg, "get_block_m_for_mega_moe", None)
        if f is not None:
            for args in ((ROWS * TOPK,), (ROWS, TOPK), (ROWS * TOPK, E)):
                try:
                    r[f"mega_block_m{args}"] = f(*args)
                except Exception as e:
                    r[f"mega_block_m{args}"] = repr(e)[:200]
        return r

    apir = step("api", api) or {}

    # ---------------------------------------------------------------- layer-0 pool: H hot + S spare, packed SF
    L = a.layer
    hot = [int(e) for e in json.load(open(a.map))[str(L)]]
    H, S = len(hot), SPARE
    cold_all = [e for e in range(E) if e not in set(hot)]
    experts, idx = routing(hot, cold_all, a.seed)
    cold_used = [e for e in experts if e not in set(hot)]
    ck = Ckpt(a.ckpt)
    t0 = time.time()
    full = {p: torch.zeros(H + S, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in PROJS}
    pool_s = {p: torch.empty(H, NK[p][0], NK[p][1] // 32, dtype=torch.uint8, device=dev) for p in PROJS}
    for slot, e in enumerate(hot):
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            full[p][slot].copy_(w)
            pool_s[p][slot].copy_(s)
    host = {}
    for e in cold_used:
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            host[(p, e, "weight")] = w.contiguous().pin_memory()
            host[(p, e, "scale")] = s.contiguous().pin_memory()
    C = {p: DI.sf_words(NK[p][1] // 32) for p in PROJS}
    sf = {}
    for p in PROJS:
        sf[p] = DI.packed_sf_zeros(H + S, NK[p][0], C[p], dev)
        sf[p][:H].copy_(DI.pack_words(pool_s[p]))
    dummy_w = torch.zeros(NK["w1"][0] * NK["w1"][1] // 2, dtype=torch.int8, device=dev)
    dummy_s = torch.zeros(NK["w1"][0] * NK["w1"][1] // 32, dtype=torch.uint8, device=dev)
    ptab = torch.zeros(6, E, dtype=torch.int64)
    gtab = torch.full((E,), -1, dtype=torch.int32)
    for e in range(E):
        for i, p in enumerate(PROJS):
            if e in hot:
                slot = hot.index(e)
                ptab[i, e], ptab[3 + i, e] = full[p][slot].data_ptr(), pool_s[p][slot].data_ptr()
            elif e in cold_used:
                ptab[i, e], ptab[3 + i, e] = host[(p, e, "weight")].data_ptr(), host[(p, e, "scale")].data_ptr()
            else:
                ptab[i, e], ptab[3 + i, e] = dummy_w.data_ptr(), dummy_s.data_ptr()
    for slot, e in enumerate(hot):
        gtab[e] = slot
    ptab, gtab = ptab.to(dev), gtab.to(dev)
    dsts = [full[p] for p in PROJS] + [sf[p] for p in PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device=dev)
    sf_dims = ((NK["w1"][0], C["w1"]), (NK["w2"][0], C["w2"]))
    row_w = NK["w1"][0] * NK["w1"][1] // 2
    G = H + S
    torch.cuda.synchronize()
    out["pool"] = {"layer": L, "H": H, "S": S, "G": G, "experts": experts, "cold_used": cold_used,
                   "routing": idx.tolist(), "load_s": round(time.time() - t0, 1)}
    print(f"[pool] layer {L}: {H} hot + {S} spare; routing 6x6 over {UNIQUE} experts ({COLD} cold)", flush=True)

    # ---------------------------------------------------------------- the dedup mapping + cold staging (real kernels)
    P = ROWS * TOPK
    w_r = torch.rand(ROWS, TOPK, generator=g)
    idx_d, w_d = idx.to(dev), w_r.to(dev)
    base = MR.dedup_routes(idx_d, w_d, 128)
    inpl_c = {"gtab": gtab, "H": H, "S": S, "dummy": torch.tensor(hot[0], device=dev), "has_dummy": True, "v2": True,
              "layout": "contiguous", "G": G, "gscr": G, "step_sf": False}
    ext0 = MR.rows_layout(base, inpl_c, 128, ROWS, TOPK)

    def staging_check():
        SK.stage_rows(ptab, base["uniq"], gtab, dtab, dsts, ext0["sidx"], ext0["wl"], H, row_w, sf_dims, S)
        torch.cuda.synchronize()
        ok = True
        for c, e in enumerate(cold_used):
            for i, p in enumerate(PROJS):
                ok &= bool(torch.equal(full[p][H + c].cpu(), host[(p, e, "weight")]))
                ok &= bool(torch.equal(DI.unpack_sf(sf[p][H + c:H + c + 1]).cpu()[0], host[(p, e, "scale")]
                                       .view(NK[p][0], -1)))
        return {"staged_equal_to_checkpoint": ok, "n_over": int(ext0["n_over"]),
                "wl": ext0["wl"].tolist()[:8]}

    step("staging_check", staging_check)

    xin = {1: acts(ROWS, DIM, g), 2: acts(P, INTER, g)}   # w1/w3 input per row; w2 input per pair

    # ---------------------------------------------------------------- one call shape
    def make(layout, align):
        dg.set_mk_alignment_for_contiguous_layout(align)
        br = 128 if layout == "masked" else align
        inpl = dict(inpl_c, layout=layout)
        ext = MR.rows_layout(base, inpl, br, ROWS, TOPK)
        nblk = G + 1 if layout == "masked" else P
        M = nblk * br
        st = {"layout": layout, "align": align, "br": br, "ext": ext, "M": M}
        st["xq"] = {w_: torch.zeros(M, K, dtype=FP8, device=dev) for w_, K in ((1, DIM), (2, INTER))}
        st["xs"] = {w_: torch.ones(M, K // 32, dtype=torch.float32, device=dev) for w_, K in ((1, DIM), (2, INTER))}
        st["d"] = {p: torch.zeros(M, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}
        arow = ext["arow"]
        q1, s1 = xin[1]
        st["xq"][1].view(torch.uint8).index_copy_(0, arow, q1.to(dev).index_select(0, base["prow"]).view(torch.uint8))
        st["xs"][1].index_copy_(0, arow, s1.to(dev).index_select(0, base["prow"]))
        q2, s2 = xin[2]
        st["xq"][2].view(torch.uint8).index_copy_(0, arow, q2.to(dev).view(torch.uint8))
        st["xs"][2].index_copy_(0, arow, s2.to(dev))
        if layout == "contiguous":
            mi = torch.full((M,), -1, dtype=torch.int32, device=dev)
            mi.view(P, br)[:, :ROWS].copy_(ext["tags"])
            st["mi"] = mi
        else:
            st["gl"] = ext["gl"].to(dev)
        return st

    def gemm(st, p):
        w_ = 2 if p == "w2" else 1
        dg.set_mk_alignment_for_contiguous_layout(st["align"])
        if st["layout"] == "contiguous":
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous((st["xq"][w_], st["xs"][w_]), (full[p], sf[p]), st["d"][p],
                                                    st["mi"], **RECIPE)
        elif st["layout"] == "psum":
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous((st["xq"][w_], st["xs"][w_]), (full[p], sf[p]), st["d"][p],
                                                    st["gl"], use_psum_layout=True, **RECIPE)
        else:
            br = st["br"]
            dg.m_grouped_fp8_fp4_gemm_nt_masked(
                (st["xq"][w_].view(-1, br, st["xq"][w_].size(-1))[:G], st["xs"][w_].view(-1, br, st["xs"][w_].size(-1))[:G]),
                (full[p], sf[p]), st["d"][p].view(-1, br, NK[p][0])[:G], st["gl"], ROWS, **RECIPE)

    def rows_of(st):
        return {p: st["d"][p].index_select(0, st["ext"]["arow"]).clone() for p in PROJS}

    ref = None
    variants = [("contiguous128", "contiguous", 128)]
    theo = apir.get("theoretical_alignment_em6")
    aligns = [int(v) for v in a.aligns.split(",") if v]
    if isinstance(theo, int) and theo not in aligns and theo != 128:
        aligns.append(theo)
    variants += [(f"contiguous{al}", "contiguous", al) for al in aligns if al >= ROWS]
    psum_aligns = sorted({al for al in [theo if isinstance(theo, int) else 32, 32, 16] if al >= ROWS})
    variants += [(f"psum{al}", "psum", al) for al in psum_aligns]
    variants += [("masked", "masked", 128)]
    for name, layout, al in variants:
        def run(layout=layout, al=al):
            st = make(layout, al)
            for p in PROJS:
                gemm(st, p)
            torch.cuda.synchronize()
            got = rows_of(st)
            r = {"layout": layout, "align": al, "A_rows": st["M"], "real_rows": P}
            nonlocal_ref = ref if ref is not None else got
            r["equal"] = {p: DI.compare_bf16(got[p], nonlocal_ref[p]) for p in PROJS}
            r["bit_equal"] = all(v["equal"] for v in r["equal"].values())
            r["us_per_layer"] = graph_us(lambda: [gemm(st, p) for p in PROJS], iters=a.iters)
            r["us_per_gemm"] = {p: graph_us(lambda p=p: gemm(st, p), iters=a.iters)["us"] for p in PROJS}
            r["_rows"] = got
            return r

        r = step(name, run, out["variants"])
        if r is not None and "_rows" in r:
            if ref is None:
                ref = r["_rows"]
            del r["_rows"]
        dg.set_mk_alignment_for_contiguous_layout(128)

    # ---------------------------------------------------------------- dedup + staging timings
    def dedup_times():
        r = {}
        for impl in ("matrix", "warp"):
            buf = MR.DedupBuffers(ROWS, TOPK, dev)
            mi = torch.full((P * 128,), -1, dtype=torch.int32, device=dev)
            inpl = dict(inpl_c, mi=mi, gl=torch.zeros(G, dtype=torch.int32, device=dev),
                        ovf=torch.zeros(1, dtype=torch.int32, device=dev))
            got = MR.dedup_fused(buf, idx_d, w_d, 128, inpl=inpl, impl=impl)
            eq = all(torch.equal(got[k_], ext0[k_]) for k_ in ("arow", "posmat", "wl", "sidx")) and \
                torch.equal(mi.view(P, 128)[:, :ROWS], ext0["tags"])
            r[impl] = {"us": graph_us(lambda: MR.dedup_fused(buf, idx_d, w_d, 128, inpl=inpl, impl=impl),
                                      iters=a.iters)["us"], "equal_to_mirror": bool(eq)}
        return r

    step("dedup", dedup_times)

    def staging_times():
        sidx_old = ext0["sidx"]
        ids_stage = torch.where(base["counts"] > 0, base["uniq"], inpl_c["dummy"])
        mi = torch.full((P * 128,), -1, dtype=torch.int32, device=dev)
        r = {"stage_inplace_36_all_scales": graph_us(lambda: SK.stage_inplace(
            ptab, ids_stage, gtab, dtab, dsts, mi, H, 128, row_w, sf_dims, True, None, sidx=sidx_old, spare=S,
            ovf=torch.zeros(1, dtype=torch.int32, device=dev)), iters=a.iters)["us"]}
        ext_s = MR.rows_layout(base, dict(inpl_c, step_sf=True), 128, ROWS, TOPK)
        r["stage_rows_pool_nwork_S"] = graph_us(lambda: SK.stage_rows(
            ptab, base["uniq"], gtab, dtab, dsts, ext0["sidx"], ext0["wl"], H, row_w, sf_dims, S), iters=a.iters)["us"]
        r["stage_rows_step_nwork_36"] = graph_us(lambda: SK.stage_rows(
            ptab, base["uniq"], gtab, dtab, dsts, ext_s["sidx"], ext_s["wl"], H, row_w, sf_dims, P), iters=a.iters)["us"]
        return r

    step("staging", staging_times)

    # ---------------------------------------------------------------- verdict
    ok = {k_: v for k_, v in out["variants"].items() if isinstance(v, dict) and v.get("bit_equal")}
    best = min(ok, key=lambda k_: ok[k_]["us_per_layer"]["us"]) if ok else None
    def flag(name):
        """variant -> first_light --rows-moe-gemm (psumNN: `psum` with $DSV41_ROWS_PSUM_ALIGN=NN if NN is not the
        library's theoretical value the runtime picks by default)."""
        if name is None:
            return None
        if name == "contiguous128":
            return "contiguous"
        if name.startswith("psum"):
            al = int(name[4:])
            return "psum" if al == theo else f"psum (DSV41_ROWS_PSUM_ALIGN={al})"
        return name if name in DI.ROWS_GEMMS else f"{name} (DSV41_DG_ALIGN)"

    out["verdict"] = {"fastest_bit_equal": best,
                      "us_per_layer": {k_: v["us_per_layer"]["us"] for k_, v in ok.items()},
                      "not_bit_equal": [k_ for k_, v in out["variants"].items()
                                        if isinstance(v, dict) and "bit_equal" in v and not v["bit_equal"]],
                      "failed": [k_ for k_, v in out["variants"].items() if isinstance(v, dict) and "error" in v],
                      "rows_moe_gemm_flag": flag(best)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", a.out)
    print("VERDICT", json.dumps(out["verdict"]))
    return 0 if not out["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
