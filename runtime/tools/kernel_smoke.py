#!/usr/bin/env python3
"""kernel_smoke.py — compile and run every TileLang kernel in DeepSeek's reference kernel.py on this GPU at small shapes.
Checks: JIT compiles on this arch, runs, finite output, and agreement with a torch reference where cheap.
Run inside vllm/vllm-openai:v0.31.0 with the reference dir mounted at /ref and an output dir at /out."""
import sys, time, json, torch
sys.path.insert(0, "/ref")
import kernel as K  # noqa: E402

dev = torch.device("cuda")
torch.set_default_dtype(torch.bfloat16)  # model.py set_dtype(): GEMM outputs allocate in the default dtype
cap = torch.cuda.get_device_capability(); name = torch.cuda.get_device_name()
print("gpu", name, "cap", cap, "torch", torch.__version__, flush=True)
out = {"gpu": name, "cap": cap, "torch": torch.__version__, "results": {}}

def rec(k, fn):
    t0 = time.time()
    try:
        r = fn(); torch.cuda.synchronize(); dt = time.time() - t0
        out["results"][k] = {"ok": True, "s": round(dt, 2), **(r or {})}
        print(f"OK   {k:22s} {dt:6.1f}s {r}", flush=True)
    except Exception as e:
        out["results"][k] = {"ok": False, "err": repr(e)[:400]}
        print(f"FAIL {k:22s} {repr(e)[:400]}", flush=True)

def t_act_quant():
    x = torch.randn(64, 5120, device=dev, dtype=torch.bfloat16)
    y, s = K.act_quant(x, block_size=128)
    deq = (y.float().view(64, -1, 128) * s.view(64, -1, 1)).view(64, 5120)
    err = ((deq - x.float()).norm() / x.float().norm()).item()
    return {"y": str(y.dtype), "s": str(s.dtype), "rel_err": round(err, 4)}

def t_act_quant_ue8m0():
    x = torch.randn(64, 5120, device=dev, dtype=torch.bfloat16)
    y, s = K.act_quant(x, block_size=128, scale_fmt="ue8m0")
    return {"y": str(y.dtype), "s": str(s.dtype), "s_shape": list(s.shape)}

def t_fp4_act_quant():
    x = torch.randn(64, 5120, device=dev, dtype=torch.bfloat16)
    y, s = K.fp4_act_quant(x)
    return {"y": str(y.dtype), "shape": list(y.shape), "s": str(s.dtype), "s_shape": list(s.shape)}

def t_fp8_gemm():
    M, N, Kd = 64, 2304, 5120
    x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, Kd, device=dev, dtype=torch.bfloat16) * 0.05)
    xq, xs = K.act_quant(x, block_size=128)
    # weight: fp8 e4m3 with per-128x128 block scales (fp32), as fp8_gemm expects scales_b[N/128, K/128]
    wb = w.float().view(N // 128, 128, Kd // 128, 128)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    ws = (amax / 448.0).view(N // 128, Kd // 128).contiguous()
    wq = (wb / (amax / 448.0)).view(N, Kd).to(torch.float8_e4m3fn).contiguous()
    y = K.fp8_gemm(xq, xs, wq, ws)
    ref = x.float() @ w.float().t()
    err = ((y.float() - ref).norm() / ref.norm()).item()
    return {"rel_err": round(err, 4), "y": str(y.dtype)}

def t_fp4_gemm():
    M, N, Kd = 64, 2304, 5120
    x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
    xq, xs = K.act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    # random packed fp4 weights + e8m0 scales (structure test only; no numeric reference for e2m1 here)
    b = torch.randint(0, 256, (N, Kd // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    bs = torch.full((N, Kd // 32), 127, device=dev, dtype=torch.uint8).view(torch.float8_e8m0fnu)  # scale 2^0
    y = K.fp4_gemm(xq, xs, b, bs, scale_dtype=torch.float8_e8m0fnu, act_block_size=32)
    return {"shape": list(y.shape), "finite": bool(torch.isfinite(y.float()).all()), "y": str(y.dtype)}

def t_hc_sinkhorn():
    hc = 4; n = 64
    mixes = torch.randn(1, n, (2 + hc) * hc, device=dev, dtype=torch.float32)
    scale = torch.ones(3, device=dev, dtype=torch.float32); base = torch.zeros((2 + hc) * hc, device=dev, dtype=torch.float32)
    pre, post, comb = K.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
    rs = comb.sum(-1); cs = comb.sum(-2)
    return {"pre": list(pre.shape), "comb": list(comb.shape), "row_sum_dev": round((rs - 1).abs().max().item(), 5), "col_sum_dev": round((cs - 1).abs().max().item(), 5)}

def t_sparse_attn():
    h, d, topk, n = 64, 512, 512, 4096
    q = torch.randn(1, 1, h, d, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(1, n, d, device=dev, dtype=torch.bfloat16)
    sink = torch.full((h,), -1e4, device=dev, dtype=torch.float32)
    idx = torch.arange(topk, device=dev, dtype=torch.int32).view(1, 1, topk)
    y = K.sparse_attn(q, kv, sink, idx, d ** -0.5)
    # torch reference over the selected rows (sink negligible)
    kvs = kv[0, :topk].float(); qs = q[0, 0].float()
    att = torch.softmax(qs @ kvs.t() * d ** -0.5, dim=-1); ref = att @ kvs
    err = ((y[0, 0].float() - ref).norm() / ref.norm()).item()
    return {"shape": list(y.shape), "rel_err": round(err, 4)}

for k, fn in [("act_quant", t_act_quant), ("act_quant_ue8m0", t_act_quant_ue8m0), ("fp4_act_quant", t_fp4_act_quant),
              ("fp8_gemm", t_fp8_gemm), ("fp4_gemm", t_fp4_gemm), ("hc_split_sinkhorn", t_hc_sinkhorn), ("sparse_attn", t_sparse_attn)]:
    rec(k, fn)
json.dump(out, open("/out/kernel-smoke.json", "w"), indent=1)
print(json.dumps({k: v.get("ok") for k, v in out["results"].items()}))
