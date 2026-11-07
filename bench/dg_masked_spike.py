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
from deep_gemm.utils.math import per_token_cast_to_fp8  # noqa
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
        MAXM = 128   # DeepGEMM masked kernels need max_m >= 128 (measured: 64/8 corrupt later groups)
        x = torch.randn(G, M, Kd, dtype=torch.bfloat16)
        if os.environ.get("VARY_BLOCKS", "1") == "1":   # per-32 block magnitudes spanning 2^-6..2^6 so per-32 and per-128 scales differ
            x = (x.view(G, M, Kd // 32, 32) * torch.exp2(torch.randint(-6, 7, (G, M, Kd // 32, 1)).float())).view(G, M, Kd).to(torch.bfloat16)
        # reference per expert (per-32 e8m0 activation scales, the model's act_quant)
        ref = torch.empty(G, M, N, dtype=torch.bfloat16)
        for g in range(G):
            q, s = K.act_quant(x[g], 32, "ue8m0", E8); ref[g] = K.fp4_gemm(q, s, Wfp4[g], Se8[g], E8, act_block_size=32)
        def t_ref():
            for g in range(G):
                q, s = K.act_quant(x[g], 32, "ue8m0", E8); K.fp4_gemm(q, s, Wfp4[g], Se8[g], E8, act_block_size=32)
        # DeepGEMM path: activations fp8 with per-128 ue8m0 scales (the only sm100 fp8xfp4 recipe DG tests), padded to MAXM rows
        xq = torch.zeros(G, MAXM, Kd, dtype=torch.float8_e4m3fn); xs = torch.ones(G, MAXM, Kd // 128, dtype=torch.float32)
        for g in range(G):
            q, s = per_token_cast_to_fp8(x[g], use_ue8m0=True, gran_k=128); xq[g, :M] = q; xs[g, :M] = s
        S_f = torch.exp2(S.view(torch.uint8).to(torch.int32).float() - 127.0).contiguous()   # checkpoint e8m0 -> fp32 (exact)
        masked_m = torch.full((G,), M, dtype=torch.int32)
        def run(variant):
            d = torch.empty(G, MAXM, N, dtype=torch.bfloat16)
            dg.m_grouped_fp8_fp4_gemm_nt_masked((xq, xs), (W, S_f), d, masked_m, 8, recipe_a=(1, 128), recipe_b=(1, 32))
            torch.cuda.synchronize(); return d[:, :M], ("fp8+fp32/128", "int8+fp32/32")
        # also: fp32 reference from the DG-quantized activations and exact-unpacked weights (isolates DG's own error)
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
