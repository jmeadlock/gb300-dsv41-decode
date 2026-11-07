#!/usr/bin/env python3
"""dg_control_spike.py — establish a CORRECT DeepGEMM fp8xfp4 masked grouped GEMM call with DeepGEMM's own
quantizers on random data, then diff those quantizers' bytes/scales against our checkpoint convention.
Steps: (1) random bf16 A[G,M,K], B[G,N,K] -> DG per_token_cast_to_fp8(gran 32) / per_token_cast_to_fp4(gran 32);
try (a) fp32 sf, (b) packed int32 sf, (c) transform_sf_into_required_layout on fp32; compare each against the fp32
dequant matmul; (2) for a real expert w1: quantize its *dequantized* bf16 values with DG's fp4 cast and compare the
packed int8 codes and scales to the checkpoint's bytes (nibble order / scale encoding diff)."""
import json, os, struct, sys, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/third_party"); sys.path.insert(0, "/ref/inference")
torch.set_default_device("cuda")
import deep_gemm as dg  # noqa
from deep_gemm.utils.math import per_token_cast_to_fp8, per_token_cast_to_fp4, cast_back_from_fp4, unpack_ue8m0_from_int  # noqa
from safetensors import safe_open  # noqa
out = {}
G, M, N, Kd = 6, 6, 2304, 5120
torch.manual_seed(0)
A = torch.randn(G, M, Kd, dtype=torch.bfloat16); B = torch.randn(G, N, Kd, dtype=torch.bfloat16) * 0.05
aq, asf, bq, bsf = [], [], [], []
for g in range(G):
    q, s = per_token_cast_to_fp8(A[g], use_ue8m0=True, gran_k=32); aq.append(q); asf.append(s)
    q4, s4 = per_token_cast_to_fp4(B[g], use_ue8m0=True, gran_k=32); bq.append(q4); bsf.append(s4)
aq = torch.stack(aq); asf = torch.stack(asf); bq = torch.stack(bq); bsf = torch.stack(bsf)
print("A", aq.dtype, aq.shape, asf.dtype, asf.shape, "| B", bq.dtype, bq.shape, bsf.dtype, bsf.shape)
# fp32 reference from the SAME quantized values
A_d = torch.stack([(aq[g].float().view(M, -1, 32) * asf[g].view(M, -1, 1)).view(M, Kd) for g in range(G)])
B_d = torch.stack([cast_back_from_fp4(bq[g], bsf[g], gran_k=32) for g in range(G)])
ref = torch.einsum("gmk,gnk->gmn", A_d, B_d)
masked_m = torch.full((G,), M, dtype=torch.int32)
def run(variant):
    d = torch.empty(G, M, N, dtype=torch.bfloat16)
    if variant in ("fp32", "fp32_ab"):
        sa, sb = asf, bsf
    elif variant == "packed":
        sa = dg.pack_ue8m0_to_int(asf); sb = dg.pack_ue8m0_to_int(bsf)
    elif variant == "transform":
        sa = dg.transform_sf_into_required_layout(asf, mn=M, k=Kd, recipe=(1, 1, 32), num_groups=G, is_sfa=True)
        sb = dg.transform_sf_into_required_layout(bsf, mn=N, k=Kd, recipe=(1, 1, 32), num_groups=G, is_sfa=False)
    elif variant == "mnmajor":
        sa = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(asf); sb = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(bsf)
    if variant.endswith("_ab"):
        dg.m_grouped_fp8_fp4_gemm_nt_masked((aq, sa), (bq, sb), d, masked_m, M, recipe_a=(1, 32), recipe_b=(1, 32))
    else:
        dg.m_grouped_fp8_fp4_gemm_nt_masked((aq, sa), (bq, sb), d, masked_m, M, recipe=(1, 1, 32))
    torch.cuda.synchronize(); return d, (str(sa.dtype), tuple(sa.shape), str(sb.dtype), tuple(sb.shape))
for v in ("fp32_ab", "fp32", "transform"):
    try:
        d, lay = run(v); rel = ((d.float() - ref).norm() / ref.norm()).item()
        out[v] = {"rel_err": round(rel, 6), "max_abs": round((d.float() - ref).abs().max().item(), 4), "layouts": lay}
    except Exception as e:
        out[v] = {"err": repr(e)[:300]}
    print(v, out[v], flush=True)
# (2) checkpoint convention vs DeepGEMM's packer, real expert
def shard_of(name):
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
f = shard_of("layers.0.ffn.experts.3.w1.weight")
with safe_open("/ref/" + f, "pt", device="cpu") as fh:
    w = fh.get_tensor("layers.0.ffn.experts.3.w1.weight").cuda(); s = fh.get_tensor("layers.0.ffn.experts.3.w1.scale").cuda()
s_f = torch.exp2(s.view(torch.uint8).to(torch.int32).float() - 127.0)
w_deq = cast_back_from_fp4(w, s_f, gran_k=32)            # assumes DG nibble order; if wrong, the re-pack below will show
rq, rs = per_token_cast_to_fp4(w_deq.to(torch.bfloat16), use_ue8m0=True, gran_k=32)
out["ckpt_vs_dg_pack"] = {"codes_equal_frac": (rq == w).float().mean().item(), "scales_equal_frac": (rs == s_f).float().mean().item(),
                          "lo_nibble_equal_frac": ((rq & 0xF) == (w & 0xF)).float().mean().item(), "hi_nibble_equal_frac": (((rq >> 4) & 0xF) == ((w >> 4) & 0xF)).float().mean().item(),
                          "swapped_nibble_equal_frac": ((((rq & 0xF) << 4) | ((rq >> 4) & 0xF)) == w).float().mean().item()}
print(out["ckpt_vs_dg_pack"]); json.dump(out, open("/work/scratch/dg-control-spike.json", "w"), indent=1)
