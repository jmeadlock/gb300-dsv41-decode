#!/usr/bin/env python3
"""dg_masked_spike.py — T9a-1 ABI/numerics probe: DeepGEMM m_grouped_fp8_fp4_gemm_nt_masked over a gathered
[G,N,K/2] expert staging buffer (real layer-0 expert bytes), vs the reference TileLang fp4_gemm per expert.
Questions: does the call accept our bytes (fp4 e2m1x2 [G,N,K/2] + e8m0 [G,N,K/32] via transform_sf_into_required_layout,
activations fp8 [G,M,K] + e8m0 [G,M,K/32]); what layout/recipe works; output drift vs reference (bf16 mismatch rate,
max |d|); µs for 3 calls (w1,w3,w2 shapes) at M=1 and M=6 vs 18 reference launches. deep_gemm imported from
vllm/third_party WITHOUT importing vllm."""
import json, os, struct, sys, time, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/third_party"); sys.path.insert(0, "/ref/inference")
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
import deep_gemm as dg  # noqa
import kernel as K  # noqa
from safetensors import safe_open  # noqa
assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
E8 = torch.float8_e8m0fnu; G = 6; DIM = 5120; INTER = 2304
def shard_of(name):
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
_c = {}
def load(name):
    f = shard_of(name); _c.setdefault(f, safe_open("/ref/" + f, "pt", device="cpu")); return _c[f].get_tensor(name)
out = {"calls": {}}
def rec(k, fn):
    try: r = fn(); out["calls"][k] = r; print("OK  ", k, r, flush=True)
    except Exception as e: out["calls"][k] = {"err": repr(e)[:600]}; print("FAIL", k, repr(e)[:600], flush=True)

ids = [3, 17, 100, 211, 300, 383]
for proj, N, Kd in (("w1", INTER, DIM), ("w2", DIM, INTER)):
    W = torch.stack([load(f"layers.0.ffn.experts.{e}.{proj}.weight") for e in ids]).cuda()            # int8 [G,N,K/2]
    S = torch.stack([load(f"layers.0.ffn.experts.{e}.{proj}.scale").view(torch.uint8) for e in ids]).cuda()  # [G,N,K/32]
    Wfp4 = W.view(torch.float4_e2m1fn_x2); Se8 = S.view(E8)
    for M in (1, 6):
        x = torch.randn(G, M, Kd, dtype=torch.bfloat16)
        # reference per expert
        ref = torch.empty(G, M, N, dtype=torch.bfloat16)
        xq = torch.empty(G, M, Kd, dtype=torch.float8_e4m3fn); xs = torch.empty(G, M, Kd // 32, dtype=E8)
        for g in range(G):
            q, s = K.act_quant(x[g], 32, "ue8m0", E8); xq[g] = q; xs[g] = s
            ref[g] = K.fp4_gemm(q, s, Wfp4[g], Se8[g], E8, act_block_size=32)
        torch.cuda.synchronize()
        def t_ref():
            for g in range(G): K.fp4_gemm(xq[g], xs[g], Wfp4[g], Se8[g], E8, act_block_size=32)
        # DeepGEMM: layouts. Try the documented transform on both operands' SF.
        e2f = lambda t: torch.exp2(t.view(torch.uint8).to(torch.int32).float() - 127.0)   # e8m0 -> fp32 (exact)
        xs_f = e2f(xs).contiguous(); S_f = e2f(S).contiguous()                              # [G,M,K/32], [G,N,K/32]
        def run(variant):
            d = torch.empty(G, M, N, dtype=torch.bfloat16)
            masked_m = torch.full((G,), M, dtype=torch.int32)
            if variant == "fp32_sf":
                sfa, sfb = xs_f, S_f
            elif variant == "transform_sf":
                sfa = dg.transform_sf_into_required_layout(xs_f, mn=M, k=Kd, recipe=(1, 1, 32), num_groups=G, is_sfa=True)
                sfb = dg.transform_sf_into_required_layout(S_f, mn=N, k=Kd, recipe=(1, 1, 32), num_groups=G, is_sfa=False)
            elif variant == "packed_mn_major":
                sfa = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(xs_f); sfb = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(S_f)
            dg.m_grouped_fp8_fp4_gemm_nt_masked((xq, sfa), (W, sfb), d, masked_m, M, recipe_a=(1, 32), recipe_b=(1, 32))   # W: int8 packed, low nibble first; DG test convention
            torch.cuda.synchronize(); return d, (sfa.dtype, tuple(sfa.shape), sfb.dtype, tuple(sfb.shape))
        for variant in ("fp32_sf",):
            def probe(variant=variant):
                d, layouts = run(variant)
                diff = (d.float() - ref.float()); mism = (d != ref).float().mean().item()
                rel = (diff.norm() / ref.float().norm()).item()
                for _ in range(3): run(variant)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(20): run(variant)
                torch.cuda.synchronize(); us = (time.perf_counter() - t0) / 20 * 1e6
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(20): t_ref()
                torch.cuda.synchronize(); us_ref = (time.perf_counter() - t0) / 20 * 1e6
                return {"bf16_mismatch_rate": round(mism, 5), "rel_err": round(rel, 6), "max_abs": round(diff.abs().max().item(), 5), "equal": bool(torch.equal(d, ref)),
                        "us_dg_incl_sf_transform": round(us, 1), "us_ref_6x": round(us_ref, 1), "layouts": str(layouts)}
            rec(f"{proj}_M{M}_{variant}", probe)
json.dump(out, open("/work/scratch/dg-masked-spike.json", "w"), indent=1)
