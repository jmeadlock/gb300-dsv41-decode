#!/usr/bin/env python3
"""kernel_smoke2.py — T1 remainder: the model's exact kernel configs on real checkpoint bytes, at M=1 and M=6.
- fp8_gemm block-32 with E8M0 scales (model.py linear(): act_quant(x, 32, "ue8m0", e8m0) + fp8_gemm(..., block_size=32)) on a real
  shared-expert w1 [2304,5120] and attention wkv; compared with a torch dequant reference.
- fp4_gemm on a REAL routed expert (layers.0.ffn.experts.0.w1: I8 [2304,2560] viewed as float4_e2m1fn_x2 + E8M0 [2304,160]);
  compared with a torch reference that unpacks the e2m1 nibbles exactly.
- inplace QDQ variants: act_quant(..., inplace=True), fp4_act_quant(k, 32, True) and fp4_act_quant(latent, 16, True, scale_dtype=e4m3).
- F.embedding on e4m3 / e8m0 tensors (Engram path, model.py:317-320).
Run inside the library image with /models (checkpoint root) at /ref, /out writable."""
import json, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/ref/inference")
torch.set_default_dtype(torch.bfloat16)
import kernel as K  # noqa: E402
from safetensors import safe_open  # noqa: E402

dev = torch.device("cuda"); out = {"results": {}}
inv = json.load(open("/inv/checkpoint-1017-inventory.json"))
def shard_of(name):
    # inventory stores only examples; scan headers for the file holding `name`
    import os, struct
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
    raise KeyError(name)
def load(name):
    with safe_open("/ref/" + shard_of(name), "pt", device="cpu") as f:
        return f.get_tensor(name)

def rec(k, fn):
    t0 = time.time()
    try:
        r = fn(); torch.cuda.synchronize(); out["results"][k] = {"ok": True, "s": round(time.time() - t0, 2), **(r or {})}
        print(f"OK   {k:28s} {out['results'][k]}", flush=True)
    except Exception as e:
        out["results"][k] = {"ok": False, "err": repr(e)[:400]}; print(f"FAIL {k:28s} {repr(e)[:400]}", flush=True)

E8M0 = torch.float8_e8m0fnu
def e8m0_to_f32(s): return torch.exp2(s.view(torch.uint8).float() - 127.0)

def fp8_linear_ref_and_kernel(name_w, name_s, M):
    w = load(name_w).to(dev); s = load(name_s).to(dev)   # e4m3 [N,K], e8m0 [N/32,K/32]
    N, Kd = w.shape
    x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
    xq, xs = K.act_quant(x, 32, "ue8m0", E8M0)
    y = K.fp8_gemm(xq, xs, w, s, E8M0, block_size=32)
    # reference: dequant both to fp32, fp32 matmul
    wd = (w.float().view(N // 32, 32, Kd // 32, 32) * e8m0_to_f32(s).view(N // 32, 1, Kd // 32, 1)).view(N, Kd)
    xd = (xq.float().view(M, Kd // 32, 32) * e8m0_to_f32(xs).view(M, Kd // 32, 1)).view(M, Kd)
    ref = xd @ wd.t()
    err = ((y.float() - ref).norm() / ref.norm()).item()
    return {"N": N, "K": Kd, "M": M, "rel_err_vs_dequant_ref": round(err, 5), "y_dtype": str(y.dtype)}

def t_fp8_shared_w1_m1(): return fp8_linear_ref_and_kernel("layers.0.ffn.shared_experts.w1.weight", "layers.0.ffn.shared_experts.w1.scale", 1)
def t_fp8_shared_w1_m6(): return fp8_linear_ref_and_kernel("layers.0.ffn.shared_experts.w1.weight", "layers.0.ffn.shared_experts.w1.scale", 6)
def t_fp8_engram_wkv_m1(): return fp8_linear_ref_and_kernel("layers.1.engram.wkv.weight", "layers.1.engram.wkv.scale", 1)

def unpack_e2m1(b_u8):
    """exact e2m1 decode of packed nibbles [N,K/2] uint8 -> [N,K] fp32 (low nibble first, as float4_e2m1fn_x2)."""
    lo = b_u8 & 0xF; hi = b_u8 >> 4
    nib = torch.stack([lo, hi], dim=-1).flatten(-2)
    sign = torch.where(nib & 0x8 != 0, -1.0, 1.0)
    e = (nib >> 1) & 0x3; m = nib & 0x1
    mag = torch.where(e == 0, m.float() * 0.5, torch.exp2(e.float() - 1.0) * (1.0 + m.float() * 0.5))
    return sign * mag

def fp4_expert(M):
    w = load("layers.0.ffn.experts.0.w1.weight"); s = load("layers.0.ffn.experts.0.w1.scale")  # I8 [2304,2560], e8m0 [2304,160]
    assert w.dtype == torch.int8 and s.dtype == E8M0, (w.dtype, s.dtype)
    N, K2 = w.shape; Kd = K2 * 2
    wq = w.view(torch.float4_e2m1fn_x2).to(dev).contiguous(); sq = s.to(dev).contiguous()
    x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
    xq, xs = K.act_quant(x, 32, "ue8m0", E8M0)
    y = K.fp4_gemm(xq, xs, wq, sq, E8M0, act_block_size=32)
    wd = unpack_e2m1(w.view(torch.uint8).to(dev)).view(N, Kd // 32, 32) * e8m0_to_f32(sq).view(N, Kd // 32, 1)
    xd = (xq.float().view(M, Kd // 32, 32) * e8m0_to_f32(xs).view(M, Kd // 32, 1)).view(M, Kd)
    ref = xd @ wd.view(N, Kd).t()
    err = ((y.float() - ref).norm() / ref.norm()).item()
    return {"N": N, "K": Kd, "M": M, "rel_err_vs_exact_unpack_ref": round(err, 5), "w_bytes": w.numel(), "s_min": int(s.view(torch.uint8).min()), "s_max": int(s.view(torch.uint8).max())}
def t_fp4_expert_m1(): return fp4_expert(1)
def t_fp4_expert_m6(): return fp4_expert(6)

def t_inplace_qdq():
    kv = torch.randn(1, 1, 512, device=dev, dtype=torch.bfloat16); kv0 = kv.clone()
    K.act_quant(kv, 32, "ue8m0", E8M0, True)
    d1 = (kv.float() - kv0.float()).abs().max().item()
    k = torch.randn(1, 1, 32, 128, device=dev, dtype=torch.bfloat16); k0 = k.clone(); K.fp4_act_quant(k, 32, True)
    d2 = (k.float() - k0.float()).abs().max().item()
    lat = torch.randn(1, 1, 512, device=dev, dtype=torch.bfloat16); l0 = lat.clone(); K.fp4_act_quant(lat, 16, True, scale_dtype=torch.float8_e4m3fn)
    d3 = (lat.float() - l0.float()).abs().max().item()
    return {"window_fp8_qdq_maxdiff": round(d1, 4), "indexK_fp4_qdq_maxdiff": round(d2, 4), "latent_fp4_e4m3x16_qdq_maxdiff": round(d3, 4), "changed": [d1 > 0, d2 > 0, d3 > 0]}

def t_embedding_fp8():
    w = torch.randn(1000, 256, device=dev).to(torch.float8_e4m3fn); s = torch.full((1000, 8), 127, device=dev, dtype=torch.uint8).view(E8M0)
    idx = torch.randint(0, 1000, (24,), device=dev)
    r = {}
    for nm, t in (("e4m3", w), ("e8m0", s)):
        try:
            v = F.embedding(idx, t); r[nm] = "embedding ok " + str(v.dtype)
        except Exception as e:
            try:
                v = t.view(torch.uint8)[idx]; r[nm] = "embedding FAIL, uint8 gather ok: " + repr(e)[:60]
            except Exception as e2:
                r[nm] = "both fail " + repr(e2)[:60]
        try:
            v2 = torch.index_select(t, 0, idx); r[nm + "_index_select"] = "ok"
        except Exception as e:
            r[nm + "_index_select"] = "FAIL " + repr(e)[:60]
    return r

for k, fn in [("fp8_shared_w1_M1", t_fp8_shared_w1_m1), ("fp8_shared_w1_M6", t_fp8_shared_w1_m6), ("fp8_engram_wkv_M1", t_fp8_engram_wkv_m1),
              ("fp4_real_expert_M1", t_fp4_expert_m1), ("fp4_real_expert_M6", t_fp4_expert_m6), ("inplace_qdq", t_inplace_qdq), ("embedding_fp8", t_embedding_fp8)]:
    rec(k, fn)
json.dump(out, open("/out/kernel-smoke2.json", "w"), indent=1)
print(json.dumps({k: v.get("ok") for k, v in out["results"].items()}))
