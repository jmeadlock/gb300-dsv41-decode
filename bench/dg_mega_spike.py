#!/usr/bin/env python3
"""dg_mega_spike.py -- M2-L3 ABI spike (box): DeepGEMM MegaMoE (fp8_fp4_mega_moe, world size 1) for the routed MoE
on layer 0's REAL expert bytes, against today's dg-inplace chain. docs/HANDOFF-M2-L3-MEGAMOE.md step (1); the layout
it builds is runtime/dg_mega.py's; findings docs/M2-L3-ABI-FINDINGS.md.

Answers, each recorded in the JSON (receipt under scratch/ by default):
  api       every mega entry point the INSTALLED deep_gemm exposes, with its pybind signature / doc (E2,
            T8-T9A-DESIGN.md:199), the installed mega/__init__.py source vs the vendored copy (sha256), the token
            alignment, get_block_m_for_mega_moe probes, and the SymmBuffer's views (shape / dtype / strides) at
            (G, T = 6 tokens, top-6, 5120, 2304, shared 0 and 1).
  layout    our L1 / SF builders (dg_mega.mega_l1_weight / mega_l1_sf / mega_l2_sf) == the library's own
            transform_weights_for_mega_moe on the same real bytes (8 pool slots: bitwise), and the L2 weight passed
            to the kernel IS the pool's w2 tensor (data_ptr identity: in place, no copy). The L1 tensor is an
            interleaved COPY (the pool cannot provide it in place: FINDINGS).
  call      the first call that works: SF storage "mn" (MN-major, the TMA coordinates' reading) then "k"; with
            the shared expert unfused (0) and fused (1). Exceptions recorded verbatim.
  numerics  routed-only MegaMoE output (bf16 [R, 5120]) vs today's dg-inplace chain (contiguous GEMMs on the same
            pool + the reference glue + fp32 combine in ascending-id order, rounded to bf16) on identical inputs:
            equal / mismatch % / max ulp, at R = 1 (row 0 alone) and R = 6 (6 rows x top-6 over 22 unique experts,
            3 of them cold -- staged by dg_mega.stage_mega from pinned host memory, the runtime's staging launch).
            Row invariance: R = 6 row r vs row r alone (bitwise, every r). Replay: two calls equal.
            fp64 oracle (the reference model's rounding points -- bf16 gate/up, bf16 SwiGLU output re-quantized per
            32, bf16 expert outputs -- with fp64 dots and sums; shared expert likewise): mismatch % / max ulp /
            relative L2 of (a) dg-inplace routed + oracle shared, (b) mega routed + oracle shared added after the
            bf16 output ("gemv" mode's extra rounding), (c) mega with the fused shared expert. Gate: (b)/(c) <= (a).
            Activations are SYNTHETIC (N(0,1) with per-32 block magnitudes 2^-6..2^6) and the router weights
            synthetic (positive, normalized, x route_scale 1.5); the weights are the real bytes.
  speed     CUDA-graph-replayed us per layer: mega routed-only and fused-shared at R = 1 and 6 (hot only, and with
            the 3-cold staging launch), vs today's three contiguous GEMMs in the runtime's call shape (contiguous16,
            36 blocks: the 1-row rowmatch call and the 6-row call alike). Graph capture of the mega launch is itself
            a probe (the record says "graph" or the capture error and an eager fallback).
  hbm       dg_mega.hbm_bytes for the real model (the L1 copy that does not fit) + mem_get_info now.

Run on the box like bench/dg_rows_spike.py (library image, checkpoint at /ref, repo at /work, DG JIT cache):
  python3 bench/dg_mega_spike.py [--spare 16] [--out scratch/dg-mega-spike-<date>.json]
deep_gemm is imported from vllm/third_party with vllm never imported (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import hashlib
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
from runtime import dg_mega as DM  # noqa: E402
from runtime.dg_moe import import_deep_gemm  # noqa: E402

FP8 = torch.float8_e4m3fn
PROJS = ("w1", "w3", "w2")
DIM, INTER, TOPK, E = 5120, 2304, 6, 384
NK = {"w1": (INTER, DIM), "w3": (INTER, DIM), "w2": (DIM, INTER)}   # (N, K)
ROWS, UNIQUE, COLD = 6, 22, 3
LIM, ROUTE_SCALE = 10.0, 1.5        # config.json swiglu_limit / route_scale
MEGA_NAMES = ("fp8_fp4_mega_moe", "bf16_mega_moe", "SymmBuffer", "get_symm_buffer_for_mega_moe",
              "transform_weights_for_mega_moe", "get_block_m_for_mega_moe", "bf16_mega_gate", "mega_mhc")
C_NAMES = ("fp8_fp4_mega_moe", "get_symm_buffer_size_for_mega_moe", "get_token_alignment_for_mega_moe",
           "get_block_m_for_mega_moe")
VENDORED = os.path.join(REPO, "reference", "vllm-0.31.0", "deep_gemm", "mega", "__init__.py")


# ------------------------------------------------------------------------------------------------ helpers
def cpu_gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def act_quant(x):
    """The reference act_quant(x, 32, "ue8m0") in torch: per-32 amax (floor 1e-4) / 448 -> 2^ceil(log2), e4m3."""
    z = x.float().unflatten(-1, (-1, 32))
    s = torch.exp2(torch.ceil(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)))
    return (z / s).clamp(-448.0, 448.0).to(FP8).flatten(-2), s.squeeze(-1)


def synth_x(R, g):
    x = torch.randn(R, DIM, generator=g, device="cpu")
    m = torch.exp2(torch.randint(-6, 7, (R, DIM // 32, 1), generator=g, device="cpu").float())
    return (x.view(R, DIM // 32, 32) * m).view(R, DIM).to(torch.bfloat16)


def routing(hot, cold, seed):
    rng = random.Random(seed)
    experts = sorted(rng.sample(hot, UNIQUE - COLD) + rng.sample(cold, COLD))
    for _ in range(100000):
        rows = [sorted(rng.sample(experts, TOPK)) for _ in range(ROWS)]
        if len({e for r in rows for e in r}) == UNIQUE:
            return experts, rows
    raise RuntimeError("no cover found")


def deq_fp4(w_i8, codes):
    """int8 packed e2m1 [N, K/2] + e8m0 codes [N, K/32] -> fp64 [N, K] (exact)."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64,
                       device=w_i8.device)
    u = w_i8.view(torch.uint8).long()
    v = torch.stack([lut[u & 0xF], lut[u >> 4]], -1).flatten(-2)
    return v * torch.exp2(codes.double() - 127).repeat_interleave(32, -1)


def deq_fp8(w, codes):
    return w.double() * torch.exp2(codes.double() - 127).repeat_interleave(32, -1)


def oracle_expert(a, wg, wu, wd, w, lim=LIM):
    """The reference model's expert (model.py:845-853) with fp64 dots: bf16 gate/up, clamp, SwiGLU, * w, bf16,
    act_quant, w2, bf16. a fp64 [D] (the dequantized act_quant(x)); weights fp64."""
    gate = (wg @ a).to(torch.bfloat16).double()
    up = (wu @ a).to(torch.bfloat16).double()
    gate = torch.clamp(gate, max=lim)
    up = torch.clamp(up, min=-lim, max=lim)
    h = (gate * torch.sigmoid(gate) * up * w).to(torch.bfloat16)
    q2, s2 = act_quant(h.view(1, -1))
    h2 = (q2.double().view(-1, 32) * s2.double().view(-1, 1)).view(-1)
    return (wd @ h2).to(torch.bfloat16)


def bf16_cmp(got, want):
    return DI.compare_bf16(got.contiguous().view(-1), want.contiguous().view(-1))


def oracle_dist(got_bf16, exact64):
    want = exact64.to(torch.bfloat16)
    r = bf16_cmp(got_bf16, want)
    d = got_bf16.double() - exact64
    r["rel_l2"] = float(d.norm() / exact64.norm())
    return r


# ------------------------------------------------------------------------------------------------ the probe
def main(argv=None):
    from dg_contiguous_spike import Ckpt, graph_us

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ref")
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--spare", type=int, default=16, help="spare slots: G = H + S (the 6-row build's default 16)")
    ap.add_argument("--tokens", type=int, default=DM.MAX_TOKENS_DEFAULT, help="T, the SymmBuffer's token rows")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", f"dg-mega-spike-{time.strftime('%Y%m%d')}.json"))
    a = ap.parse_args(argv)
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "errors": [], "checks": {}, "labels": {"activations": "synthetic", "router_weights": "synthetic",
                                                   "expert_weights": "real layer bytes"}}

    def step(name, fn):
        try:
            r = fn()
            out["checks"][name] = r
            print("OK  ", name, json.dumps(r, default=str)[:1500], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            out["checks"][name] = {"error": tb[-2500:]}
            out["errors"].append(name)
            print("FAIL", name, tb[-2500:], flush=True)
            return None

    dev = torch.device("cuda")
    dg = import_deep_gemm()
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    out["deep_gemm"] = {"file": getattr(dg, "__file__", None), "version": getattr(dg, "__version__", None)}
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()

    # ---------------------------------------------------------------- (1) the API surface, introspected
    def api():
        r = {"dg_mega_names": sorted(n for n in dir(dg) if "mega" in n.lower())}
        for n in MEGA_NAMES:
            f = getattr(dg, n, None)
            if f is None:
                r[n] = None
                continue
            try:
                sig = str(inspect.signature(f))
            except (TypeError, ValueError):
                sig = None
            r[n] = {"signature": sig, "doc": (getattr(f, "__doc__", None) or "")[:1500]}
        C = getattr(dg, "_C", None)
        r["_C_mega_names"] = sorted(n for n in dir(C) if "mega" in n.lower()) if C is not None else None
        for n in C_NAMES:
            f = getattr(C, n, None) if C is not None else None
            r[f"_C.{n}"] = None if f is None else (getattr(f, "__doc__", None) or "")[:1500]
        if C is not None and hasattr(C, "get_token_alignment_for_mega_moe"):
            r["token_alignment"] = int(C.get_token_alignment_for_mega_moe())
        f = getattr(dg, "get_block_m_for_mega_moe", None)
        if f is not None:
            for args in ((ROWS * TOPK,), (ROWS, TOPK), (ROWS * TOPK, E), (a.tokens, 311, TOPK), (1,), (6,)):
                try:
                    r[f"get_block_m{args}"] = f(*args)
                except Exception as e:
                    r[f"get_block_m{args}"] = repr(e)[:300]
        try:
            import deep_gemm.mega as dgm
            src = inspect.getsource(dgm)
            r["mega_py_sha256_installed"] = hashlib.sha256(src.encode()).hexdigest()
            r["mega_py_sha256_vendored"] = hashlib.sha256(open(VENDORED).read().encode()).hexdigest()
            r["mega_py_same_as_vendored"] = r["mega_py_sha256_installed"] == r["mega_py_sha256_vendored"]
        except Exception as e:
            r["mega_py_source"] = repr(e)[:300]
        for nm in ("_interleave_weights", "_transpose_sf_for_utccp", "fp8_fp4_mega_moe", "SymmBuffer"):
            try:
                import deep_gemm.mega as dgm
                r[f"source:{nm}"] = inspect.getsource(getattr(dgm, nm))[:2500]
            except Exception as e:
                r[f"source:{nm}"] = repr(e)[:200]
        return r

    step("api", api)

    # ---------------------------------------------------------------- layer pool: H hot + S spare (as the runtime)
    L = a.layer
    hot = [int(e) for e in json.load(open(a.map))[str(L)]]
    H, S = len(hot), int(a.spare)
    G = H + S
    cold_all = [e for e in range(E) if e not in set(hot)]
    experts, rows = routing(hot, cold_all, a.seed)
    cold_used = [e for e in experts if e not in set(hot)]
    ck = Ckpt(a.ckpt)
    t0 = time.time()
    full = {p: torch.zeros(G, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in PROJS}
    codes = {p: torch.zeros(G, NK[p][0], NK[p][1] // 32, dtype=torch.uint8, device=dev) for p in PROJS}
    for slot, e in enumerate(hot):
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            full[p][slot].copy_(w)
            codes[p][slot].copy_(s)
    host = {}
    for e in cold_used:
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            host[(p, e, "weight")] = w.contiguous().pin_memory()
            host[(p, e, "scale")] = s.contiguous().pin_memory()
    shared = {}
    for p in PROJS:
        shared[p] = (ck.get(f"layers.{L}.ffn.shared_experts.{p}.weight").to(dev),
                     DM.act_codes(ck.get(f"layers.{L}.ffn.shared_experts.{p}.scale")).to(dev))
    torch.cuda.synchronize()
    out["pool"] = {"layer": L, "H": H, "S": S, "G": G, "experts": experts, "cold_used": cold_used, "routing": rows,
                   "load_s": round(time.time() - t0, 1),
                   "shared_dtypes": {p: [str(shared[p][0].dtype), list(shared[p][1].shape)] for p in PROJS}}
    print(f"[pool] layer {L}: {H} hot + {S} spare; 6x6 over {UNIQUE} experts ({COLD} cold)", flush=True)
    gtab = torch.full((E,), -1, dtype=torch.int32, device="cpu")
    for slot, e in enumerate(hot):
        gtab[e] = slot
    gtab = gtab.to(dev)
    dummy_w = torch.zeros(NK["w1"][0] * NK["w1"][1] // 2, dtype=torch.int8, device=dev)
    dummy_s = torch.zeros(NK["w1"][0] * NK["w1"][1] // 32, dtype=torch.uint8, device=dev)
    ptab = torch.zeros(6, E, dtype=torch.int64, device="cpu")
    for e in range(E):
        for i, p in enumerate(PROJS):
            if e in hot:
                sl = hot.index(e)
                ptab[i, e], ptab[3 + i, e] = full[p][sl].data_ptr(), codes[p][sl].data_ptr()
            elif e in cold_used:
                ptab[i, e], ptab[3 + i, e] = host[(p, e, "weight")].data_ptr(), host[(p, e, "scale")].data_ptr()
            else:
                ptab[i, e], ptab[3 + i, e] = dummy_w.data_ptr(), dummy_s.data_ptr()
    ptab = ptab.to(dev)

    # ---------------------------------------------------------------- (2) the mega layouts from OUR pool
    mega = {}

    def layouts():
        r = {}
        mega["l1"] = DM.mega_l1_weight(full["w1"], full["w3"])                       # the interleaved COPY
        mega["l2"] = full["w2"]                                                      # in place
        for major in DM.SF_MAJORS:
            mega[("sf1", major)] = DM.mega_l1_sf(codes["w1"], codes["w3"], major)
            mega[("sf2", major)] = DM.mega_l2_sf(codes["w2"], major)
        r["l2_is_pool_w2"] = mega["l2"].data_ptr() == full["w2"].data_ptr() and mega["l2"].is_contiguous()
        r["l1_copy_bytes"] = mega["l1"].numel()
        f = getattr(dg, "transform_weights_for_mega_moe", None)
        if f is not None:
            n = 8
            w13 = torch.cat([full["w1"][:n], full["w3"][:n]], 1)
            for major in DM.SF_MAJORS:
                sf13 = DM.to_major(torch.cat([DM.sf_words(codes["w1"][:n]), DM.sf_words(codes["w3"][:n])], 1), major)
                sf2 = DM.to_major(DM.sf_words(codes["w2"][:n]), major)
                try:
                    (tw1, tsf1), (tw2, tsf2) = f((w13, sf13), (full["w2"][:n], sf2))
                    r[f"library_transform_{major}"] = {
                        "l1_equal": bool(torch.equal(tw1, mega["l1"][:n])),
                        "sf1_equal_values": bool(torch.equal(tsf1, mega[("sf1", major)][:n])),
                        "sf1_strides_lib_ours": [list(tsf1.stride()), list(mega[("sf1", major)][:n].stride())],
                        "l2_equal": bool(torch.equal(tw2, full["w2"][:n])),
                        "sf2_equal_values": bool(torch.equal(tsf2, mega[("sf2", major)][:n]))}
                except Exception as e:
                    r[f"library_transform_{major}"] = repr(e)[:400]
            del w13
        r["mem_get_info_GiB"] = [round(v / 2 ** 30, 2) for v in torch.cuda.mem_get_info()]
        return r

    step("layouts", layouts)

    # ---------------------------------------------------------------- SymmBuffers + their views (E2)
    syms = {}

    def symm():
        r = {}
        for n_sh in (0, 1):
            try:
                sym = DM.make_symm_buffer(dg, G, a.tokens, TOPK, DIM, INTER, n_sh)
                sym.topk_idx.fill_(-1)
                sym.topk_weights.zero_()
                syms[n_sh] = sym
                r[f"shared{n_sh}"] = DM.describe_symm(sym)
            except Exception as e:
                r[f"shared{n_sh}"] = {"error": repr(e)[:600]}
        return r

    step("symm_buffer", symm)

    # ---------------------------------------------------------------- inputs
    g = cpu_gen(a.seed)
    x6 = synth_x(ROWS, g).to(dev)
    wr = torch.rand(ROWS, TOPK, generator=g, device="cpu") + 0.05
    wr = (wr / wr.sum(-1, keepdim=True) * ROUTE_SCALE).to(dev)
    ids6 = torch.tensor(rows, dtype=torch.long, device="cpu").to(dev)
    q6, s6 = act_quant(x6)
    sh_l1 = DM.mega_l1_weight(shared["w1"][0], shared["w3"][0])
    sh_w = {}
    for major in DM.SF_MAJORS:
        sh_w[major] = ((sh_l1, DM.mega_l1_sf(DM.block_codes_per_row(shared["w1"][1], INTER),
                                             DM.block_codes_per_row(shared["w3"][1], INTER), major)),
                       (shared["w2"][0].contiguous(), DM.mega_l2_sf(DM.block_codes_per_row(shared["w2"][1], DIM),
                                                                    major)))
    stage_buf = torch.full((S,), -1, dtype=torch.long, device=dev)
    y = torch.zeros(a.tokens, DIM, dtype=torch.bfloat16, device=dev)

    def prep(sym, R, major, do_stage=True):
        """Route rows [:R] (ids6[:R]) into sym (l3_route_mega: 1 launch), stage their cold experts, write x / x_sf.
        Graph-capturable."""
        DM.route_mega(ids6[:R].contiguous(), wr[:R].contiguous(), gtab, sym.topk_idx, sym.topk_weights, stage_buf,
                      H, S, E)
        if do_stage:
            DM.stage_mega(stage_buf, ptab, dtab[major], dsts[major], H, major)
        sym.x[:R].copy_(q6[:R])
        DM.write_act_sf(sym.x_sf, DM.act_codes(s6[:R]), R)

    dsts = {mj: [mega["l1"], mega["l2"], mega[("sf1", mj)], mega[("sf2", mj)]] for mj in DM.SF_MAJORS} \
        if "l1" in mega else {}
    dtab = {mj: torch.tensor([t.data_ptr() for t in v], dtype=torch.int64, device="cpu").to(dev)
            for mj, v in dsts.items()}

    def call(sym, major, fused):
        kw = {"shared_l1_weights": sh_w[major][0], "shared_l2_weights": sh_w[major][1]} if fused else {}
        dg.fp8_fp4_mega_moe(y, (mega["l1"], mega[("sf1", major)]), (mega["l2"], mega[("sf2", major)]), sym,
                            recipe=DM.MEGA_RECIPE, activation="swiglu", activation_clamp=LIM, fast_math=False, **kw)
        return y

    def reset(sym):
        sym.topk_idx.fill_(-1)
        sym.topk_weights.zero_()
        sym.x.view(torch.uint8).zero_()

    # ---------------------------------------------------------------- (3) which call works (run after (4) is defined)
    works, works_all = {}, {}

    def first_call():
        r = {}
        for fused in (0, 1):
            if fused not in syms:
                r[f"shared{fused}"] = "no SymmBuffer"
                continue
            for major in DM.SF_MAJORS:
                key = f"shared{fused}_{major}"
                try:
                    reset(syms[fused])
                    prep(syms[fused], ROWS, major)
                    call(syms[fused], major, fused)
                    torch.cuda.synchronize()
                    yy = y[:ROWS].clone()
                    r[key] = {"ok": True, "finite": bool(torch.isfinite(yy.float()).all()),
                              "max_abs": float(yy.float().abs().max()), "padding_rows_zero":
                              bool((y[ROWS:] == 0).all()) if a.tokens > ROWS else None}
                    works_all.setdefault(fused, []).append(major)
                except Exception as e:
                    r[key] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:800]}"}
        # a call that does not raise can still read a wrong layout: pick the routed-only call's SF storage by its
        # distance to today's chain (a wrong layout gives ~100 % mismatch); the fused call uses the same storage
        if len(works_all.get(0, [])) > 1:
            ip = inplace_rows(ROWS).to(torch.bfloat16)
            score = {}
            for major in works_all[0]:
                reset(syms[0])
                prep(syms[0], ROWS, major)
                yy = call(syms[0], major, 0)[:ROWS].clone()
                score[major] = DI.compare_bf16(yy.view(-1), ip.view(-1))["mismatch_frac"]
            works[0] = min(score, key=lambda k_: score[k_])
            r["layout_pick_mismatch_vs_inplace"] = score
        elif works_all.get(0):
            works[0] = works_all[0][0]
        if works_all.get(1):
            works[1] = works[0] if works.get(0) in works_all[1] else works_all[1][0]
        r["working"] = {f"shared{k}": v for k, v in works.items()}
        return r


    # ---------------------------------------------------------------- (4) today's chain (dg-inplace), same inputs
    sfp = {p: DI.packed_sf_zeros(G, NK[p][0], NK[p][1] // 128, dev) for p in PROJS}
    for p in PROJS:
        sfp[p][:H].copy_(DI.pack_words(codes[p][:H]))

    def inplace_rows(R):
        """routed fp32 sums (ascending id) of rows [:R] through today's contiguous GEMMs + reference glue."""
        m = DM.mega_route(ids6[:R], gtab, H, S, E)
        gmap = {}
        for r_ in range(R):
            for j in range(TOPK):
                gmap[int(ids6[r_, j])] = int(m["topk"][r_, j])
        # cold experts' bytes into their groups for the dg-inplace arm (same slots the mega staging used)
        for e, grp in gmap.items():
            if e in cold_used:
                for p in PROJS:
                    full[p][grp].copy_(host[(p, e, "weight")].to(dev))
                    sfp[p][grp:grp + 1].copy_(DI.pack_sf(host[(p, e, "scale")].to(dev).unsqueeze(0)))
        s32 = s6.float()
        acts1 = {(r_, int(e)): (q6[r_], s32[r_]) for r_ in range(R) for e in ids6[r_].tolist()}
        blocks = [blk for r_ in range(R) for blk in DI.one_row_blocks(r_, ids6[r_].tolist(), gmap)]
        n_blk = len(blocks)
        g1 = DI.run_contiguous(dg, full["w1"], sfp["w1"], blocks, acts1, DI.MK_ALIGN, n_blk)
        g3 = DI.run_contiguous(dg, full["w3"], sfp["w3"], blocks, acts1, DI.MK_ALIGN, n_blk)
        acts2 = {}
        for r_ in range(R):
            for j, e in enumerate(ids6[r_].tolist()):
                gate, up = g1[(r_, e)].float(), g3[(r_, e)].float()
                up = torch.clamp(up, min=-LIM, max=LIM)
                gate = torch.clamp(gate, max=LIM)
                h = (torch.nn.functional.silu(gate) * up * wr[r_, j]).to(torch.bfloat16)
                qq, ss = act_quant(h.view(1, -1))
                acts2[(r_, e)] = (qq[0], ss[0].float())
        o2 = DI.run_contiguous(dg, full["w2"], sfp["w2"], blocks, acts2, DI.MK_ALIGN, n_blk)
        ys = []
        for r_ in range(R):
            acc = torch.zeros(DIM, dtype=torch.float32, device=dev)
            for e in ids6[r_].tolist():
                acc += o2[(r_, e)].float()
            ys.append(acc)
        return torch.stack(ys)

    def oracle_rows(R):
        """fp64 oracle of routed (per row) and shared outputs (bf16 each), and the exact fp64 totals."""
        a64 = (q6[:R].double().view(R, -1, 32) * s6[:R].double().view(R, -1, 1)).view(R, DIM)
        m = DM.mega_route(ids6[:R], gtab, H, S, E)
        routed = torch.zeros(R, DIM, dtype=torch.float64, device=dev)
        for e in sorted({int(v) for v in ids6[:R].reshape(-1).tolist()}):
            grp = None
            for r_ in range(R):
                if e in ids6[r_].tolist():
                    grp = int(m["topk"][r_, ids6[r_].tolist().index(e)])
            if e in cold_used:
                wd = {p: deq_fp4(host[(p, e, "weight")].to(dev), host[(p, e, "scale")].to(dev)) for p in PROJS}
            else:
                wd = {p: deq_fp4(full[p][grp], codes[p][grp]) for p in PROJS}
            for r_ in range(R):
                row = ids6[r_].tolist()
                if e in row:
                    routed[r_] += oracle_expert(a64[r_], wd["w1"], wd["w3"], wd["w2"], float(wr[r_, row.index(e)])
                                                ).double()
            del wd
        sw = {p: deq_fp8(shared[p][0], DM.block_codes_per_row(shared[p][1], shared[p][0].size(0))) for p in PROJS}
        sh = torch.stack([oracle_expert(a64[r_], sw["w1"], sw["w3"], sw["w2"], 1.0) for r_ in range(R)])
        return routed, sh

    def numerics():
        r = {}
        if 0 not in works:
            return {"skipped": "no working routed-only call"}
        mj0 = works[0]
        res = {}
        for R in (ROWS, 1):
            reset(syms[0])
            prep(syms[0], R, mj0)
            yy = call(syms[0], mj0, 0)[:R].clone()
            reset(syms[0])
            prep(syms[0], R, mj0)
            yy2 = call(syms[0], mj0, 0)[:R].clone()
            torch.cuda.synchronize()
            res[R] = yy
            ip = inplace_rows(R)
            r[f"R{R}"] = {"mega_vs_inplace_routed": bf16_cmp(yy, ip.to(torch.bfloat16)),
                          "replay_equal": bool(torch.equal(yy.view(torch.int16), yy2.view(torch.int16)))}
            if R == ROWS:
                routed64, sh = oracle_rows(R)
                exact = routed64 + sh.double()
                r["oracle"] = {
                    "inplace_routed_plus_oracle_shared": oracle_dist((ip + sh.float()).to(torch.bfloat16), exact),
                    "mega_gemv_mode": oracle_dist((yy.float() + sh.float()).to(torch.bfloat16), exact),
                    "mega_routed_vs_oracle_routed": oracle_dist(yy, routed64),
                    "inplace_routed_vs_oracle_routed": oracle_dist(ip.to(torch.bfloat16), routed64)}
                if 1 in works:
                    mj1 = works[1]
                    reset(syms[1])
                    prep(syms[1], R, mj1)
                    yf = call(syms[1], mj1, 1)[:R].clone()
                    torch.cuda.synchronize()
                    r["oracle"]["mega_fused_shared"] = oracle_dist(yf, exact)
                    r["fused_vs_gemv_mode"] = bf16_cmp(yf, (yy.float() + sh.float()).to(torch.bfloat16))
                d = r["oracle"]
                r["gate_oracle_distance_le_today"] = {
                    k_: d[k_]["mismatch_frac"] <= d["inplace_routed_plus_oracle_shared"]["mismatch_frac"]
                    for k_ in ("mega_gemv_mode", "mega_fused_shared") if k_ in d}
        if 1 in res and ROWS in res:
            # row invariance: row r of R = 6 vs row r alone (R = 1 prep uses row 0; loop the others)
            inv = [bool(torch.equal(res[1][0].view(torch.int16), res[ROWS][0].view(torch.int16)))]
            for r_ in range(1, ROWS):
                reset(syms[0])
                m = DM.mega_route(ids6[r_:r_ + 1], gtab, H, S, E)
                syms[0].topk_idx[:1].copy_(m["topk"])
                syms[0].topk_weights[:1].copy_(wr[r_:r_ + 1])
                stage_buf.copy_(m["stage"])
                DM.stage_mega(stage_buf, ptab, dtab[mj0], dsts[mj0], H, mj0)
                syms[0].x[:1].copy_(q6[r_:r_ + 1])
                DM.write_act_sf(syms[0].x_sf, DM.act_codes(s6[r_:r_ + 1]), 1)
                y1 = call(syms[0], mj0, 0)[:1].clone()
                inv.append(bool(torch.equal(y1[0].view(torch.int16), res[ROWS][r_].view(torch.int16))))
            r["row_invariant_R1_vs_R6"] = inv
        return r

    step("first_call", first_call)
    step("numerics", numerics)

    # ---------------------------------------------------------------- (5) speed
    def speed():
        r = {}
        for fused in (0, 1):
            if fused not in works:
                continue
            mj = works[fused]
            sym = syms[fused]
            for R in (1, ROWS):
                reset(sym)
                prep(sym, R, mj)
                r[f"mega_shared{fused}_R{R}_hot"] = graph_us(lambda: call(sym, mj, fused), iters=a.iters)
                r[f"mega_shared{fused}_R{R}_route_stage_act_call"] = graph_us(
                    lambda: (prep(sym, R, mj), call(sym, mj, fused)), iters=a.iters)
                reset(sym)
        # today's 3 contiguous GEMMs in the runtime's call shape (contiguous16, R*k = 36 blocks)
        al, nb = 16, ROWS * TOPK
        M = al * nb
        A1 = torch.zeros(M, DIM, dtype=FP8, device=dev)
        S1 = torch.ones(M, DIM // 32, dtype=torch.float32, device=dev)
        A2 = torch.zeros(M, INTER, dtype=FP8, device=dev)
        S2 = torch.ones(M, INTER // 32, dtype=torch.float32, device=dev)
        mi = torch.full((M,), -1, dtype=torch.int32, device=dev)
        D1 = torch.zeros(M, INTER, dtype=torch.bfloat16, device=dev)
        D3 = torch.zeros(M, INTER, dtype=torch.bfloat16, device=dev)
        D2 = torch.zeros(M, DIM, dtype=torch.bfloat16, device=dev)
        for R in (1, ROWS):
            mi.fill_(-1)
            uniq = sorted({int(v) for v in ids6[:R].reshape(-1).tolist()})
            for b_, e in enumerate(uniq):
                cnt = sum(e in ids6[r_].tolist() for r_ in range(R))
                if e in hot:
                    mi[b_ * al: b_ * al + cnt] = hot.index(e)

            def three():
                with DI.mk_alignment(dg, al):
                    dg.m_grouped_fp8_fp4_gemm_nt_contiguous((A1, S1), (full["w1"], sfp["w1"]), D1, mi,
                                                            recipe_a=(1, 32), recipe_b=(1, 32))
                    dg.m_grouped_fp8_fp4_gemm_nt_contiguous((A1, S1), (full["w3"], sfp["w3"]), D3, mi,
                                                            recipe_a=(1, 32), recipe_b=(1, 32))
                    dg.m_grouped_fp8_fp4_gemm_nt_contiguous((A2, S2), (full["w2"], sfp["w2"]), D2, mi,
                                                            recipe_a=(1, 32), recipe_b=(1, 32))
            r[f"inplace_3gemm_contiguous16_R{R}"] = graph_us(three, iters=a.iters)
        r["note"] = ("mega_*_hot = the launch alone; *_route_stage_act_call = routing ops + the staging launch (3 cold "
                     "at R = 6) + x/x_sf writes + the launch. Today's S9+S10+S11 also has SwiGLU, combine, the shared "
                     "GEMVs and the dedup/stage launches around the 3 GEMMs: compare against the step anatomy, not "
                     "against inplace_3gemm alone.")
        return r

    step("speed", speed)

    # ---------------------------------------------------------------- (6) hbm
    out["hbm_real_model"] = DM.hbm_bytes(40, [295 + 16] * 40, DIM, INTER, "copy")
    out["mem_get_info_GiB"] = [round(v / 2 ** 30, 2) for v in torch.cuda.mem_get_info()]

    # ---------------------------------------------------------------- verdict
    num = out["checks"].get("numerics") or {}
    spd = out["checks"].get("speed") or {}
    out["verdict"] = {
        "abi_call_works": {f"shared{k}": v for k, v in works.items()},
        "l2_in_place": (out["checks"].get("layouts") or {}).get("l2_is_pool_w2"),
        "l1_in_place": False,
        "mega_vs_inplace_routed_R6": (num.get(f"R{ROWS}") or {}).get("mega_vs_inplace_routed"),
        "row_invariant": num.get("row_invariant_R1_vs_R6"),
        "replay_equal": [(num.get(f"R{R}") or {}).get("replay_equal") for R in (1, ROWS)],
        "oracle_gate": num.get("gate_oracle_distance_le_today"),
        "us": {k_: v.get("us") for k_, v in spd.items() if isinstance(v, dict)},
        "errors": out["errors"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", a.out)
    print("VERDICT", json.dumps(out["verdict"], default=str))
    return 0 if not out["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
