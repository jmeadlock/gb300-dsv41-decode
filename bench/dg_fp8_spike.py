#!/usr/bin/env python3
"""dg_fp8_spike.py — can DeepGEMM fp8_gemm_nt replace the reference fp8_gemm (kernel.py:207-307) at M=1/6?
Reference layout: x fp8 [M,K] + e8m0 [M,K/32] (act_quant 32 ue8m0); w fp8 [N,K] + e8m0 [N/32, K/32] (32x32 blocks).
Try DeepGEMM: a=(fp8, fp32 [M,K/32]) recipe_a=(1,32); b=(fp8, fp32 [N/32,K/32]) recipe_b=(32,32); also recipe_b=(1,32) after
expanding the 32x32 scales to per-row (exact: each row's scale is its block's). Real weights: layers.0.attn.wq_a? use a few real
fp8 Linear weights of different shapes. Measures mismatch vs reference and µs."""
import json, os, struct, sys, time, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/third_party"); sys.path.insert(0, "/ref/inference")
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
import deep_gemm as dg  # noqa
import kernel as K  # noqa
from safetensors import safe_open  # noqa
E8 = torch.float8_e8m0fnu; out = {}
def shard_of(name):
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
_c = {}
def load(name):
    f = shard_of(name); _c.setdefault(f, safe_open("/ref/" + f, "pt", device="cpu")); return _c[f].get_tensor(name)
names = ["layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2", "layers.3.attn.wq_a", "layers.3.attn.wkv", "layers.3.attn.wo_b", "layers.1.engram.wkv"]
e2f = lambda t: torch.exp2(t.view(torch.uint8).to(torch.int32).float() - 127.0)
def bench(fn, it=50):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for nm in names:
    try:
        w = load(nm + ".weight").cuda(); s = load(nm + ".scale").cuda()
    except Exception as e:
        out[nm] = {"skip": repr(e)[:100]}; print("skip", nm, repr(e)[:100]); continue
    if w.dtype != torch.float8_e4m3fn: out[nm] = {"skip": f"dtype {w.dtype}"}; print("skip", nm, w.dtype); continue
    N, Kd = w.shape; s_f = e2f(s).contiguous()                      # [N/32, K/32]
    s_row = s_f.repeat_interleave(32, dim=0).contiguous()           # [N, K/32] per-row expansion (exact)
    for M in (1, 6):
        x = (torch.randn(M, Kd, dtype=torch.bfloat16).view(M, Kd // 32, 32) * torch.exp2(torch.randint(-6, 7, (M, Kd // 32, 1)).float())).view(M, Kd).to(torch.bfloat16)
        xq, xs = K.act_quant(x, 32, "ue8m0", E8); ref = K.fp8_gemm(xq, xs, w, s, E8, block_size=32)
        xs_f = e2f(xs).contiguous()
        MAXM = 128; xq_p = torch.zeros(MAXM, Kd, dtype=torch.float8_e4m3fn); xq_p[:M] = xq; xs_p = torch.ones(MAXM, Kd // 32); xs_p[:M] = xs_f
        res = {}
        for variant, (sb, rb) in {"b32x32": (s_f, (32, 32)), "b_row32": (s_row, (1, 32))}.items():
            try:
                d = torch.empty(MAXM, N, dtype=torch.bfloat16)
                dg.fp8_gemm_nt((xq_p, xs_p), (w, sb), d, recipe_a=(1, 32), recipe_b=rb); torch.cuda.synchronize()
                dd = d[:M]; mism = (dd != ref).float().mean().item(); rel = ((dd.float() - ref.float()).norm() / ref.float().norm()).item()
                us = bench(lambda: dg.fp8_gemm_nt((xq_p, xs_p), (w, sb), d, recipe_a=(1, 32), recipe_b=rb))
                res[variant] = {"mismatch": round(mism, 5), "rel": round(rel, 6), "max": round((dd.float() - ref.float()).abs().max().item(), 4), "us": round(us, 1)}
            except Exception as e:
                res[variant] = {"err": repr(e)[:200]}
        # M-exact (no padding) attempt for M=1/6
        try:
            d2 = torch.empty(M, N, dtype=torch.bfloat16); dg.fp8_gemm_nt((xq, xs_f), (w, s_row), d2, recipe_a=(1, 32), recipe_b=(1, 32)); torch.cuda.synchronize()
            res["b_row32_noPad"] = {"mismatch": round((d2 != ref).float().mean().item(), 5), "us": round(bench(lambda: dg.fp8_gemm_nt((xq, xs_f), (w, s_row), d2, recipe_a=(1, 32), recipe_b=(1, 32))), 1)}
        except Exception as e:
            res["b_row32_noPad"] = {"err": repr(e)[:160]}
        res["ref_us"] = round(bench(lambda: K.fp8_gemm(xq, xs, w, s, E8, block_size=32)), 1)
        out[f"{nm}[{N}x{Kd}]_M{M}"] = res; print(f"{nm} [{N}x{Kd}] M={M}", res, flush=True)
json.dump(out, open("/work/scratch/dg-fp8-spike.json", "w"), indent=1)
