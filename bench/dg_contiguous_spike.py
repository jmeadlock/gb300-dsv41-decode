#!/usr/bin/env python3
"""dg_contiguous_spike.py -- T9a-2 probe (box): DeepGEMM m_grouped_fp8_fp4_gemm_nt_contiguous reading layer 0's
WHOLE hot pool in place ([H+S, N, K/2] int8, S = 6 spare slots for cold experts), vs today's dg-masked path
(gather of the 6 selected experts into [6, N, K/2] + 3 masked calls), on real layer-0 expert bytes.

Questions (docs/HANDOFF-T9A2-INPLACE-MOE.md "Design" step 1), every answer recorded in the JSON:
  1. SF form. The library converts an fp32 SF tensor to packed int32 UE8M0 MN-major on every call -- over a whole pool
     that is the per-call cost that rules fp32 out (timed here: `f32_full_pool`). Does it accept our packed int32 SF
     (runtime/dg_inplace.pack_sf: the pool's e8m0 bytes as little-endian int32, MN-major) as-is? Is our packing
     bit-equal to deep_gemm's own get_mn_major_tma_aligned_packed_ue8m0_tensor of the fp32 values?
  2. Numerics. contiguous vs masked on the same expert set and activations: bf16 mismatch %, max ulp (and both vs the
     reference TileLang fp4_gemm for context). Sets: 6 hot; 5 hot + 1 cold; 6 cold (cold rows staged by the
     t9a2_stage_inplace kernel from pinned host memory over C2C, as in the runtime).
  3. Alignment: rows per group block (= BLOCK_M cap), swept (default 128, 64, + the library's theoretical value).
  4. Speed per layer at M = 1 (graph-replayed sequences): masked = 1 gather_tables launch (+fp32 scales) + 3 masked
     calls; in-place = 1 stage launch (+ packed scales, m_indices) + 3 contiguous calls; components separately;
     1-cold staging cost. M = 6 rows with 6 and 22 distinct experts (masked: gather U experts + masked_m = rows per
     expert; contiguous: one block per expert, rows tagged).
  5. Masked over pool VIEWS (a strided B with group stride = several slots): accepted? equal? (T8-T9A section 0.6
     says it cannot read the pool in place; one call to confirm.)
  6. HBM: dg_inplace.hbm_plan for the real model + cuda mem_get_info now.

Run on the box like bench/dg_masked_spike.py (library image, checkpoint at /ref, repo at /work, DeepGEMM JIT cache):
  python3 bench/dg_contiguous_spike.py [--hot 295] [--aligns 128,64] [--out docs/dg-contiguous-spike-<date>.json]
deep_gemm is imported from vllm/third_party with vllm never imported (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime import dg_inplace as DI  # noqa: E402
from runtime.dg_moe import E8M0_F32, import_deep_gemm  # noqa: E402

E8 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn
PROJS = ("w1", "w3", "w2")
DIM, INTER, TOPK = 5120, 2304, 6
NK = {"w1": (INTER, DIM), "w3": (INTER, DIM), "w2": (DIM, INTER)}   # (N, K)


# ------------------------------------------------------------------------------------------------ checkpoint
class Ckpt:
    def __init__(self, root):
        from safetensors import safe_open

        self.root, self._open, self.where = root, safe_open, {}
        for f in sorted(x for x in os.listdir(root) if x.endswith(".safetensors")):
            with open(os.path.join(root, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                for k in json.loads(fh.read(n)):
                    self.where[k] = f
        self.h = {}

    def get(self, name):
        f = self.where[name]
        if f not in self.h:
            self.h[f] = self._open(os.path.join(self.root, f), "pt", device="cpu")
        return self.h[f].get_tensor(name)

    def expert(self, l, e, p):
        w = self.get(f"layers.{l}.ffn.experts.{e}.{p}.weight").view(torch.int8)
        s = self.get(f"layers.{l}.ffn.experts.{e}.{p}.scale").view(torch.uint8)
        return w, s


# ------------------------------------------------------------------------------------------------ timing
def graph_us(fn, iters=200, warm=3):
    """µs per call of fn() replayed from a CUDA graph (eager events fallback, error recorded)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    rec = {}
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(warm):
            g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        rec["us"] = round(a.elapsed_time(b) * 1e3 / iters, 2)
        rec["how"] = "graph"
        del g
    except Exception as e:  # pragma: no cover - box only
        rec["graph_error"] = repr(e)[:300]
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        rec["us"] = round(a.elapsed_time(b) * 1e3 / iters, 2)
        rec["how"] = "eager"
    return rec


def vary_blocks(x, g):
    """N(0,1) values with per-32 block magnitudes 2^-6..2^6 (the masked spike's VARY_BLOCKS: per-32 scales matter)."""
    R, K = x.shape
    m = torch.exp2(torch.randint(-6, 7, (R, K // 32, 1), generator=g).float())
    return (x.float().view(R, K // 32, 32) * m).view(R, K).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------ the probe
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ref")
    ap.add_argument("--inference", default="/ref/inference", help="reference kernel.py dir (act_quant, fp4_gemm)")
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--hot", type=int, default=None, help="use only the first N hot experts (default: all)")
    ap.add_argument("--aligns", default="128,64")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-f32-full", action="store_true", help="skip the fp32 whole-pool SF variant")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", f"dg-contiguous-spike-{time.strftime('%Y%m%d')}.json"))
    a = ap.parse_args(argv)

    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "errors": [], "checks": {}}

    def step(name, fn):
        try:
            r = fn()
            out["checks"][name] = r
            print("OK  ", name, json.dumps(r, default=str)[:400], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            out["checks"][name] = {"error": tb[-1500:]}
            out["errors"].append(name)
            print("FAIL", name, tb[-1500:], flush=True)
            return None

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    dev = torch.device("cuda")
    dg = import_deep_gemm()
    out["deep_gemm"] = {"file": getattr(dg, "__file__", None), "version": getattr(dg, "__version__", None)}
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()
    out["deterministic"] = det is not None
    sys.path.insert(0, a.inference)
    import kernel as K  # noqa: E402  (the reference TileLang kernels: act_quant, fp4_gemm)
    from runtime import static_kernels as SK

    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    f0, tot = torch.cuda.mem_get_info()
    out["hbm_start"] = {"free_GiB": round(f0 / 2**30, 3), "total_GiB": round(tot / 2**30, 3)}
    out["hbm_plan_real_model"] = DI.hbm_plan(40, [295] * 40, TOPK, {p: (NK[p][0], NK[p][1] // 2) for p in PROJS},
                                             {p: (NK[p][0], NK[p][1] // 32) for p in PROJS})

    # ---------------------------------------------------------------- layer-0 pool (HBM hot + pinned-host cold)
    L = a.layer
    hot_all = [int(e) for e in json.load(open(a.map))[str(L)]]
    hot = hot_all[: a.hot] if a.hot else hot_all
    H, S = len(hot), TOPK
    E = 384
    cold_all = [e for e in range(E) if e not in set(hot)]
    ck = Ckpt(a.ckpt)
    t0 = time.time()
    full = {p: torch.zeros(H + S, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in PROJS}
    pool_s = {p: torch.empty(H, NK[p][0], NK[p][1] // 32, dtype=torch.uint8, device=dev) for p in PROJS}
    for slot, e in enumerate(hot):
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            full[p][slot].copy_(w)
            pool_s[p][slot].copy_(s)
    rng = random.Random(a.seed)
    cold_used = sorted(rng.sample(cold_all, min(6, len(cold_all))))
    host = {}
    for e in cold_used:
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            host[(p, e, "weight")] = w.contiguous().pin_memory()
            host[(p, e, "scale")] = s.contiguous().pin_memory()
    torch.cuda.synchronize()
    out["pool"] = {"layer": L, "H": H, "S": S, "cold_used": cold_used, "load_s": round(time.time() - t0, 1),
                   "pool_weight_GiB": round(sum(t.numel() for t in full.values()) / 2**30, 3)}
    print(f"[pool] layer {L}: {H} hot + {S} spare slots loaded in {out['pool']['load_s']} s", flush=True)

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
    C = {p: DI.sf_words(NK[p][1] // 32) for p in PROJS}
    sf = {p: DI.packed_sf_zeros(H + S, NK[p][0], C[p], dev) for p in PROJS}
    sf_dims = ((NK["w1"][0], C["w1"]), (NK["w2"][0], C["w2"]))
    row_w = NK["w1"][0] * NK["w1"][1] // 2
    row_s = NK["w1"][0] * NK["w1"][1] // 32

    # ---------------------------------------------------------------- 1. SF packing vs deep_gemm's own
    def check_pack():
        p = "w1"
        f32 = E8M0_F32.to(dev)[pool_s[p][:8].long()]
        fn = getattr(dg, "get_mn_major_tma_aligned_packed_ue8m0_tensor", None)
        ours = DI.pack_sf(pool_s[p][:8])
        r = {"have_dg_fn": fn is not None, "ours_shape": list(ours.shape), "ours_stride": list(ours.stride()),
             "equal_to_torch_ref_semantics": bool(torch.equal(DI.ref_pack_from_f32(f32), ours)), "attempts": {}}
        if fn is None:
            return r
        for name, arg, want in (("3d_f32", f32, ours), ("2d_f32", f32[0], ours[0]),
                                ("2d_u8", pool_s[p][0], ours[0])):
            try:
                theirs = fn(arg)
                r["attempts"][name] = {
                    "shape": list(theirs.shape), "stride": list(theirs.stride()), "dtype": str(theirs.dtype),
                    "equal": bool(theirs.dtype == want.dtype and tuple(theirs.shape) == tuple(want.shape)
                                  and torch.equal(theirs, want)),
                    "same_strides": list(theirs.stride()) == list(want.stride())}
            except Exception as e:
                r["attempts"][name] = {"error": repr(e)[:300]}
        r["equal"] = any(v.get("equal") for v in r["attempts"].values())
        return r

    step("sf_packing", check_pack)

    # ---------------------------------------------------------------- activations (the model's act_quant)
    def quant_rows(x):
        q, s = K.act_quant(x.contiguous(), 32, "ue8m0", E8)
        return q, E8M0_F32.to(dev)[s.view(torch.uint8).long()]

    xa = vary_blocks(torch.randn(1, DIM, generator=g), g).to(dev)                       # the token (w1/w3 input)
    x2 = vary_blocks(torch.randn(S, INTER, generator=g), g).to(dev)                     # per-slot w2 inputs
    qa, sa = quant_rows(xa)
    q2, s2 = quant_rows(x2)

    def masked_bufs(G, maxm=128):
        return {1: (torch.zeros(G, maxm, DIM, dtype=FP8, device=dev), torch.ones(G, maxm, DIM // 32, device=dev)),
                2: (torch.zeros(G, maxm, INTER, dtype=FP8, device=dev), torch.ones(G, maxm, INTER // 32, device=dev))}

    def fill_one_row(bufs, k):
        bufs[1][0][:k, 0] = qa
        bufs[1][1][:k, 0] = sa
        bufs[2][0][:k, 0] = q2[:k]
        bufs[2][1][:k, 0] = s2[:k]

    # masked path state (k = 6)
    stg = {p: torch.zeros(S, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in PROJS}
    sf32 = {p: torch.zeros(S, NK[p][0], NK[p][1] // 32, dtype=torch.float32, device=dev) for p in PROJS}
    mdsts = [stg[p] for p in PROJS] + [sf32[p] for p in PROJS]
    mdtab = torch.tensor([t.data_ptr() for t in mdsts], dtype=torch.int64, device=dev)
    mb = masked_bufs(S)
    fill_one_row(mb, S)
    md = {p: torch.zeros(S, 128, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}
    masked_m = torch.ones(S, dtype=torch.int32, device=dev)

    def masked_gemm(p):
        w = 2 if p == "w2" else 1
        dg.m_grouped_fp8_fp4_gemm_nt_masked(mb[w], (stg[p], sf32[p]), md[p], masked_m, 8,
                                            recipe_a=(1, 32), recipe_b=(1, 32))

    def masked_layer(ids):
        SK.gather_tables(ptab, ids, mdtab, mdsts, row_w, row_s, True)
        for p in PROJS:
            masked_gemm(p)

    # in-place path state
    idsts = [full[p] for p in PROJS] + [sf[p] for p in PROJS]
    idtab = torch.tensor([t.data_ptr() for t in idsts], dtype=torch.int64, device=dev)
    ib = {}

    def inplace_state(A):
        if A not in ib:
            bufs = mb if A == 128 else masked_bufs(S, A)
            if A != 128:
                fill_one_row(bufs, S)
            ib[A] = {"bufs": bufs, "mi": torch.full((S * A,), -1, dtype=torch.int32, device=dev),
                     "d": {p: torch.zeros(S, A, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}}
        return ib[A]

    def contig_gemm(p, A, sf_use=None):
        st = ib[A]
        w = 2 if p == "w2" else 1
        xq, xs = st["bufs"][w]
        M = S * A
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous((xq.view(M, -1), xs.view(M, -1)), (full[p], (sf_use or sf)[p]),
                                                st["d"][p].view(M, -1), st["mi"], recipe_a=(1, 32), recipe_b=(1, 32))

    def stage(ids, A, all_scales=True):
        SK.stage_inplace(ptab, ids, gtab, idtab, idsts, ib[A]["mi"], H, A, row_w, sf_dims, all_scales)

    def inplace_layer(ids, A):
        stage(ids, A)
        for p in PROJS:
            contig_gemm(p, A)

    set_align = getattr(dg, "set_mk_alignment_for_contiguous_layout", None)
    get_align = getattr(dg, "get_mk_alignment_for_contiguous_layout", None)
    out["alignment_default"] = get_align() if get_align else None
    theo = None
    try:
        theo = int(dg.get_theoretical_mk_alignment_for_contiguous_layout(1))
    except Exception as e:
        out["alignment_theoretical_error"] = repr(e)[:200]
    out["alignment_theoretical_m1"] = theo
    aligns = [int(v) for v in a.aligns.split(",") if v]
    if theo and theo not in aligns:
        aligns.append(theo)

    sets = {"6hot": sorted(rng.sample(hot, 6)),
            "5hot+1cold": sorted(rng.sample(hot, 5) + [cold_used[0]]),
            "6cold": cold_used[:6]}
    out["expert_sets"] = sets

    def ref_rows(ids_l):
        """reference TileLang fp4_gemm per expert for context: {p: [6, N] bf16}."""
        r = {}
        for p in PROJS:
            rows = []
            for j, e in enumerate(ids_l):
                w, s = ck.expert(L, e, p)
                wf = w.to(dev).view(torch.float4_e2m1fn_x2)
                se = s.to(dev).view(E8)
                if p == "w2":
                    qq, ss = K.act_quant(x2[j:j + 1].contiguous(), 32, "ue8m0", E8)
                else:
                    qq, ss = K.act_quant(xa.contiguous(), 32, "ue8m0", E8)
                rows.append(K.fp4_gemm(qq, ss, wf, se, E8, act_block_size=32)[0])
            r[p] = torch.stack(rows)
        return r

    # ---------------------------------------------------------------- 2+3. numerics: contiguous vs masked
    numerics = {}
    for A in (128,):  # production alignment now; the other alignments run last (a bad one must not poison the rest)
        def run_align(A=A):
            if set_align is None:
                raise RuntimeError("deep_gemm has no set_mk_alignment_for_contiguous_layout")
            set_align(A)
            now = get_align() if get_align else None
            inplace_state(A)
            res = {"set": A, "now": now}
            for name, ids_l in sets.items():
                ids = torch.tensor(ids_l, dtype=torch.int64, device=dev)
                masked_layer(ids)
                for t in ib[A]["d"].values():
                    t.fill_(float("nan"))
                inplace_layer(ids, A)
                torch.cuda.synchronize()
                mi_heads = ib[A]["mi"].view(S, A)[:, 0].tolist()
                want_heads = [hot.index(e) if e in hot else H + j for j, e in enumerate(ids_l)]
                rr = {"mi_heads_ok": mi_heads == want_heads}
                for p in PROJS:
                    got = ib[A]["d"][p][:, 0]
                    rr[p] = DI.compare_bf16(got, md[p][:, 0])
                cold_j = [j for j, e in enumerate(ids_l) if e not in hot]
                if cold_j:  # the staged spare slots hold the cold bytes
                    rr["staged_bytes_ok"] = all(
                        torch.equal(full[p][H + j].cpu(), host[(p, ids_l[j], "weight")]) for p in PROJS for j in cold_j)
                    rr["staged_scales_ok"] = all(
                        torch.equal(DI.unpack_sf(sf[p][H + j:H + j + 1])[0].cpu(), host[(p, ids_l[j], "scale")])
                        for p in PROJS for j in cold_j)
                res[name] = rr
            return res

        numerics[A] = step(f"numerics_align{A}", run_align)
    if set_align is not None:
        set_align(128)
    run_align_any = run_align

    def vs_reference():
        ids_l = sets["5hot+1cold"]
        ids = torch.tensor(ids_l, dtype=torch.int64, device=dev)
        masked_layer(ids)
        inplace_state(128)
        inplace_layer(ids, 128)
        ref = ref_rows(ids_l)
        torch.cuda.synchronize()
        return {p: {"masked_vs_ref": DI.compare_bf16(md[p][:, 0], ref[p]),
                    "inplace_vs_ref": DI.compare_bf16(ib[128]["d"][p][:, 0], ref[p])} for p in PROJS}

    step("vs_reference_fp4_gemm", vs_reference)

    # fp32 whole-pool SF (the library transforms it per call): correctness + its cost
    if not a.no_f32_full:
        def f32_full():
            sf32_full = {p: E8M0_F32.to(dev)[torch.cat([pool_s[p], torch.zeros(S, *pool_s[p].shape[1:],
                                                                                  dtype=torch.uint8, device=dev)]).long()]
                         for p in PROJS}
            ids_l = sets["6hot"]
            ids = torch.tensor(ids_l, dtype=torch.int64, device=dev)
            inplace_state(128)
            stage(ids, 128)
            masked_layer(ids)
            r = {}
            for p in PROJS:
                contig_gemm(p, 128, sf_use=sf32_full)
                torch.cuda.synchronize()
                r[p] = DI.compare_bf16(ib[128]["d"][p][:, 0], md[p][:, 0])
            r["us_w1_call"] = graph_us(lambda: contig_gemm("w1", 128, sf_use=sf32_full), iters=20)
            r["sf_bytes_per_layer"] = sum(t.numel() * 4 for t in sf32_full.values())
            del sf32_full
            torch.cuda.empty_cache()
            return r

        step("f32_full_pool_sf", f32_full)

    # ---------------------------------------------------------------- 4. speed, M = 1
    def speed_m1():
        r = {}
        inplace_state(128)
        ids_h = torch.tensor(sets["6hot"], dtype=torch.int64, device=dev)
        ids_c = torch.tensor(sets["5hot+1cold"], dtype=torch.int64, device=dev)
        r["masked_layer_6hot"] = graph_us(lambda: masked_layer(ids_h), a.iters)
        r["inplace_layer_6hot"] = graph_us(lambda: inplace_layer(ids_h, 128), a.iters)
        r["masked_layer_1cold"] = graph_us(lambda: masked_layer(ids_c), a.iters)
        r["inplace_layer_1cold"] = graph_us(lambda: inplace_layer(ids_c, 128), a.iters)
        r["gather_tables_6"] = graph_us(lambda: SK.gather_tables(ptab, ids_h, mdtab, mdsts, row_w, row_s, True), a.iters)
        r["stage_6hot_step_scales"] = graph_us(lambda: stage(ids_h, 128, True), a.iters)
        r["stage_6hot_resident"] = graph_us(lambda: stage(ids_h, 128, False), a.iters)
        r["stage_1cold_step_scales"] = graph_us(lambda: stage(ids_c, 128, True), a.iters)
        stage(ids_h, 128)
        SK.gather_tables(ptab, ids_h, mdtab, mdsts, row_w, row_s, True)
        for p in PROJS:
            r[f"masked_{p}"] = graph_us(lambda p=p: masked_gemm(p), a.iters)
            r[f"contig_{p}"] = graph_us(lambda p=p: contig_gemm(p, 128), a.iters)
        m, i = r["masked_layer_6hot"]["us"], r["inplace_layer_6hot"]["us"]
        r["per_step_40_layers_ms"] = {"masked": round(40 * m / 1e3, 3), "inplace": round(40 * i / 1e3, 3),
                                      "saving": round(40 * (m - i) / 1e3, 3)}
        r["gate_inplace_le_masked"] = bool(i <= m)
        return r

    step("speed_m1", speed_m1)

    # ---------------------------------------------------------------- 4b. M = 6 rows, 6 / 22 distinct experts
    def speed_m6(n_unique):
        R = 6
        for tries in range(1000):
            pool_ids = rng.sample(hot, n_unique)
            rows = [sorted(pool_ids) if n_unique == 6 else sorted(rng.sample(pool_ids, 6)) for _ in range(R)]
            if len({e for r_ in rows for e in r_}) == n_unique:
                break
        A = 128
        lay = DI.contiguous_rows(rows, A)
        U = lay["experts"]
        nU = len(U)
        xr = vary_blocks(torch.randn(R, DIM, generator=g), g).to(dev)
        qr, sr = quant_rows(xr)
        h2 = vary_blocks(torch.randn(R * 6, INTER, generator=g), g).to(dev)
        qh, sh = quant_rows(h2)
        # masked: groups = unique experts, rows of expert u at [u, t]
        mbm = masked_bufs(nU)
        mm = torch.zeros(nU, dtype=torch.int32)
        cbuf = {1: (torch.zeros(nU * A, DIM, dtype=FP8, device=dev), torch.ones(nU * A, DIM // 32, device=dev)),
                2: (torch.zeros(nU * A, INTER, dtype=FP8, device=dev), torch.ones(nU * A, INTER // 32, device=dev))}
        mi = torch.full((nU * A,), -1, dtype=torch.int32)
        k2 = 0
        for (r_, e), arow in sorted(lay["rows"].items(), key=lambda kv: kv[1]):
            u, t = divmod(arow, A)
            mbm[1][0][u, t], mbm[1][1][u, t] = qr[r_], sr[r_]
            cbuf[1][0][arow], cbuf[1][1][arow] = qr[r_], sr[r_]
            mbm[2][0][u, t], mbm[2][1][u, t] = qh[k2], sh[k2]
            cbuf[2][0][arow], cbuf[2][1][arow] = qh[k2], sh[k2]
            k2 += 1
            mm[u] += 1
            mi[arow] = hot.index(e)
        mm, mi = mm.to(dev), mi.to(dev)
        ids = torch.tensor(U, dtype=torch.int64, device=dev)
        st_u = {p: torch.zeros(nU, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in PROJS}
        sf_u = {p: torch.zeros(nU, NK[p][0], NK[p][1] // 32, dtype=torch.float32, device=dev) for p in PROJS}
        ud = [st_u[p] for p in PROJS] + [sf_u[p] for p in PROJS]
        udt = torch.tensor([t.data_ptr() for t in ud], dtype=torch.int64, device=dev)
        mdm = {p: torch.zeros(nU, 128, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}
        cdm = {p: torch.zeros(nU * A, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}
        mi_dummy = torch.full((nU * A,), -1, dtype=torch.int32, device=dev)
        expm = max(8, int(max(lay["counts"].values())))

        def m_layer():
            SK.gather_tables(ptab, ids, udt, ud, row_w, row_s, True)
            for p in PROJS:
                w = 2 if p == "w2" else 1
                dg.m_grouped_fp8_fp4_gemm_nt_masked(mbm[w], (st_u[p], sf_u[p]), mdm[p], mm, expm,
                                                    recipe_a=(1, 32), recipe_b=(1, 32))

        def c_layer():
            SK.stage_inplace(ptab, ids, gtab, idtab, idsts, mi_dummy, H, A, row_w, sf_dims, True)
            for p in PROJS:
                w = 2 if p == "w2" else 1
                dg.m_grouped_fp8_fp4_gemm_nt_contiguous(cbuf[w], (full[p], sf[p]), cdm[p], mi,
                                                        recipe_a=(1, 32), recipe_b=(1, 32))

        set_align(A)
        m_layer()
        c_layer()
        torch.cuda.synchronize()
        res = {"rows": rows, "unique": nU, "counts": lay["counts"]}
        for p in PROJS:
            gm = torch.stack([mdm[p][a_ // A, a_ % A] for a_ in sorted(lay["rows"].values())])
            gc = torch.stack([cdm[p][a_] for a_ in sorted(lay["rows"].values())])
            res[p] = DI.compare_bf16(gc, gm)
        res["masked_layer"] = graph_us(m_layer, a.iters)
        res["inplace_layer"] = graph_us(c_layer, a.iters)
        res["per_step_40_layers_ms"] = {"masked": round(40 * res["masked_layer"]["us"] / 1e3, 3),
                                        "inplace": round(40 * res["inplace_layer"]["us"] / 1e3, 3)}
        return res

    step("m6_6distinct", lambda: speed_m6(6))
    step("m6_22distinct", lambda: speed_m6(22))

    # ---------------------------------------------------------------- 5. masked over strided pool views
    def masked_views():
        stride_slots = 5
        slots = [i * stride_slots for i in range(S)]
        p = "w1"
        Wv = full[p].as_strided((S, *full[p].shape[1:]), (stride_slots * full[p].stride(0), *full[p].stride()[1:]))
        Sv32 = torch.zeros(S, NK[p][0], NK[p][1] // 32, dtype=torch.float32, device=dev)
        Sv32.copy_(E8M0_F32.to(dev)[pool_s[p][slots].long()])
        d = torch.zeros(S, 128, NK[p][0], dtype=torch.bfloat16, device=dev)
        r = {"slots": slots, "B_is_contiguous": Wv.is_contiguous()}
        try:
            dg.m_grouped_fp8_fp4_gemm_nt_masked(mb[1], (Wv, Sv32), d, masked_m, 8, recipe_a=(1, 32), recipe_b=(1, 32))
            torch.cuda.synchronize()
            Wc = full[p][slots].contiguous()
            d2 = torch.zeros_like(d)
            dg.m_grouped_fp8_fp4_gemm_nt_masked(mb[1], (Wc, Sv32), d2, masked_m, 8, recipe_a=(1, 32), recipe_b=(1, 32))
            torch.cuda.synchronize()
            r["accepted"] = True
            r["vs_gathered_copy"] = DI.compare_bf16(d[:, 0], d2[:, 0])
        except Exception as e:
            r["accepted"] = False
            r["error"] = repr(e)[:400]
        return r

    step("masked_over_pool_views", masked_views)

    # ---------------------------------------------------------------- 3b. the other alignments (informative)
    for A in [x for x in aligns if x != 128]:
        numerics[A] = step(f"numerics_align{A}", lambda A=A: run_align_any(A))
    if set_align is not None:
        set_align(128)

    # ---------------------------------------------------------------- verdict
    f1, _ = torch.cuda.mem_get_info()
    out["hbm_end"] = {"free_GiB": round(f1 / 2**30, 3)}
    # the gates use the production alignment (128); other alignments are informative (failures recorded, not fatal)
    opt = [f"numerics_align{A}" for A in aligns if A != 128]
    out["optional_failures"] = [e for e in out["errors"] if e in opt]
    out["errors"] = [e for e in out["errors"] if e not in opt]
    num = [numerics[128]] if numerics.get(128) else []
    out["numerics_by_align"] = {A: (None if not v else {
        "max_ulp": max(v[s_][p]["max_ulp"] for s_ in sets for p in PROJS),
        "all_equal": all(v[s_][p]["equal"] for s_ in sets for p in PROJS)}) for A, v in numerics.items()}
    max_ulp = max((v[s_][p]["max_ulp"] for v in num for s_ in sets for p in PROJS), default=None)
    sp = out["checks"].get("speed_m1") or {}
    out["verdict"] = {
        "max_ulp_inplace_vs_masked": max_ulp,
        "bit_identical_all": all(v[s_][p]["equal"] for v in num for s_ in sets for p in PROJS) if num else None,
        "mi_heads_ok": all(v[s_]["mi_heads_ok"] for v in num for s_ in sets) if num else None,
        "gate_ulp_le_1": (max_ulp is not None and max_ulp <= 1),
        "gate_speed_m1": sp.get("gate_inplace_le_masked"),
        "errors": out["errors"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[probe] wrote {a.out}")
    print("VERDICT", json.dumps(out["verdict"]))
    return 0 if not out["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
