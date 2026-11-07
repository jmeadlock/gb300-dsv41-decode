#!/usr/bin/env python3
"""moe_static_spike.py — can the reference MoE be made graph-capturable without changing its arithmetic?

Reference MoE.forward (model.py:891-905): bincount().tolist() host sync + a Python loop over all 384 experts,
y[idx] += expert(x[idx], w) in ascending expert-id order, fp32 accumulate, then + shared expert, cast to bf16.

Static version (this spike): routing on device (same Gate), sort the top-k ids ascending (same accumulation order),
gather the k selected experts' weights (w1/w3/w2 + scales) through a POINTER TABLE into a static staging buffer
[k, N, K/2] via one Triton kernel (reads HBM or UVA host memory alike), then k fixed-shape Expert evaluations on the
staging slots, accumulated in slot order. No host sync, no data-dependent shapes -> capturable.

Checks on a real layer (layers.0.ffn, real weights, hot rows loaded to HBM, a few experts deliberately left in
registered host memory so the gather crosses C2C): bit-identity (torch.equal) of static vs reference output over
32 random inputs; µs per layer eager; µs per layer under torch.cuda.CUDAGraph replay.
Run in the library image with /ref = checkpoint root, repo at /work."""
import ctypes, json, mmap, os, struct, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/ref/inference"); torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
import kernel as K  # noqa
import triton, triton.language as tl  # noqa
from safetensors import safe_open  # noqa

dev = torch.device("cuda"); cudart = torch.cuda.cudart(); E8 = torch.float8_e8m0fnu
L = 0; NE = 384; TOPK = 6; DIM = 5120; INTER = 2304; K2 = DIM // 2
def shard_of(name):
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
_cache = {}
def load(name):
    f = shard_of(name)
    if f not in _cache: _cache[f] = safe_open("/ref/" + f, "pt", device="cpu")
    return _cache[f].get_tensor(name)

# ---- reference pieces (copied call pattern, not re-implemented)
def fp4_linear(x, w, s):  # model.py linear() for fp4 weights, block 32 ue8m0 activations
    xq, xs = K.act_quant(x, 32, "ue8m0", E8)
    return K.fp4_gemm(xq, xs, w, s, E8, act_block_size=32)
def expert_fwd(x, w1, s1, w3, s3, w2, s2, weight, limit):
    gate = fp4_linear(x, w1, s1).float(); up = fp4_linear(x, w3, s3).float()
    if limit > 0: up = torch.clamp(up, min=-limit, max=limit); gate = torch.clamp(gate, max=limit)
    h = F.silu(gate) * up
    if weight is not None: h = weight * h
    return fp4_linear(h.to(torch.bfloat16), w2, s2)

cfg = json.load(open("/ref/inference/config.json")); limit = cfg.get("swiglu_limit", 0.0); route_scale = cfg.get("route_scale", 1.0)
import model as RM; _args = RM.ModelArgs(**cfg); GATE_TEMP = _args.gate_temp; NORM_TOPK = _args.norm_topk_prob
print("gate_temp", GATE_TEMP, "norm_topk_prob", NORM_TOPK)
print("swiglu_limit", limit, "route_scale", route_scale, "score_func", cfg.get("score_func"), "norm_topk", cfg.get("norm_topk_prob"))

# ---- weights: all 384 experts of layer 0 (6.7 GiB); 16 of them placed in registered host memory
t0 = time.time()
names = ["w1", "w3", "w2"]
W = {n: torch.empty(NE, INTER if n != "w2" else DIM, (DIM if n != "w2" else INTER) // 2, dtype=torch.int8, device="cpu", pin_memory=False) for n in names}
S = {n: torch.empty(NE, INTER if n != "w2" else DIM, (DIM if n != "w2" else INTER) // 32, dtype=torch.uint8, device="cpu") for n in names}
for e in range(NE):
    for n in names:
        W[n][e] = load(f"layers.{L}.ffn.experts.{e}.{n}.weight"); S[n][e] = load(f"layers.{L}.ffn.experts.{e}.{n}.scale").view(torch.uint8)
print(f"loaded layer {L} experts in {time.time() - t0:.0f} s")
COLD = set(range(0, NE, 24))  # 16 cold experts spread across ids
class _CAI:
    def __init__(self, ptr, n): self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}
def host_reg(t_cpu):
    n = t_cpu.numel() * t_cpu.element_size(); mm = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    buf = (ctypes.c_char * n).from_buffer(mm); ptr = ctypes.addressof(buf)
    torch.frombuffer(mm, dtype=torch.uint8)[:] = t_cpu.contiguous().view(torch.uint8).flatten()
    assert cudart.cudaHostRegister(ptr, n, 3) == 0
    return (mm, buf), torch.as_tensor(_CAI(ptr, n), device="cuda")
keep = []
dev_w, dev_s = {}, {}   # per expert: tensor views (HBM or UVA)
for n in names:
    dev_w[n] = []; dev_s[n] = []
    for e in range(NE):
        if e in COLD:
            k1, cw = host_reg(W[n][e]); k2, cs = host_reg(S[n][e]); keep += [k1, k2]
            dev_w[n].append(cw.view(torch.int8).view(W[n][e].shape)); dev_s[n].append(cs.view(S[n][e].shape))
        else:
            dev_w[n].append(W[n][e].to(dev)); dev_s[n].append(S[n][e].to(dev))
torch.cuda.synchronize(); print("placed: hot", NE - len(COLD), "cold(UVA)", len(COLD), f"HBM {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
# pointer tables [NE] int64 for each of w1,w3,w2 weights and scales
ptab = {n: torch.tensor([t.data_ptr() for t in dev_w[n]], dtype=torch.int64, device=dev) for n in names}
stab = {n: torch.tensor([t.data_ptr() for t in dev_s[n]], dtype=torch.int64, device=dev) for n in names}
gate_w = load(f"layers.{L}.ffn.gate.weight").to(dev); gate_b = load(f"layers.{L}.ffn.gate.bias").to(dev).float()
print("gate", gate_w.dtype, gate_w.shape, "bias", gate_b.shape)
sh = {n: (load(f"layers.{L}.ffn.shared_experts.{n}.weight").to(dev), load(f"layers.{L}.ffn.shared_experts.{n}.scale").to(dev)) for n in names}
def fp8_linear(x, w, s):
    xq, xs = K.act_quant(x, 32, "ue8m0", E8); return K.fp8_gemm(xq, xs, w, s, E8, block_size=32)
def shared_fwd(x):
    gate = fp8_linear(x, *sh["w1"]).float(); up = fp8_linear(x, *sh["w3"]).float()
    if limit > 0: up = torch.clamp(up, min=-limit, max=limit); gate = torch.clamp(gate, max=limit)
    return fp8_linear((F.silu(gate) * up).to(torch.bfloat16), *sh["w2"])

def gate_fwd(x):  # model.py:811-829 (sqrtsoftplus / noaux_tc as configured)
    scores = F.linear(x.float(), gate_w.float()) / GATE_TEMP
    sf = cfg.get("score_func")
    scores = scores.softmax(-1) if sf == "softmax" else scores.sigmoid() if sf == "sigmoid" else F.softplus(scores).sqrt()
    indices = (scores + gate_b).topk(TOPK, dim=-1)[1]
    weights = scores.gather(1, indices)
    if NORM_TOPK and TOPK > 1: weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    weights *= route_scale
    return weights, indices

# ---- reference MoE (model.py:891-905) for 1 row
def moe_ref(x):
    weights, indices = gate_fwd(x)
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=NE).tolist()
    for i in range(NE):
        if counts[i] == 0: continue
        idx, top = torch.where(indices == i)
        y[idx] += expert_fwd(x[idx], dev_w["w1"][i].view(torch.float4_e2m1fn_x2), dev_s["w1"][i].view(E8), dev_w["w3"][i].view(torch.float4_e2m1fn_x2), dev_s["w3"][i].view(E8),
                             dev_w["w2"][i].view(torch.float4_e2m1fn_x2), dev_s["w2"][i].view(E8), weights[idx, top, None], limit)
    y += shared_fwd(x)
    return y.type_as(x)

# ---- static MoE: pointer-table gather into staging, fixed k evaluations
@triton.jit
def gather_rows(ptab, ids, dst, row_bytes: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0); blk = tl.program_id(1)
    e = tl.load(ids + j)
    src = tl.load(ptab + e).to(tl.pointer_type(tl.uint8))
    off = blk * BLOCK + tl.arange(0, BLOCK); m = off < row_bytes
    v = tl.load(src + off, mask=m); tl.store(dst + j * row_bytes + off, v, mask=m)
STG_W = {n: torch.empty(TOPK, *W[n].shape[1:], dtype=torch.int8, device=dev) for n in names}
STG_S = {n: torch.empty(TOPK, *S[n].shape[1:], dtype=torch.uint8, device=dev) for n in names}
BLOCK = 8192
def gather(ids_sorted):
    for n in names:
        rb = W[n][0].numel(); gather_rows[(TOPK, triton.cdiv(rb, BLOCK))](ptab[n], ids_sorted, STG_W[n], rb, BLOCK)
        rb = S[n][0].numel(); gather_rows[(TOPK, triton.cdiv(rb, BLOCK))](stab[n], ids_sorted, STG_S[n], rb, BLOCK)
def moe_static(x):
    weights, indices = gate_fwd(x)                       # [1,k]
    ids_sorted, order = torch.sort(indices[0]); w_sorted = weights[0, order]
    gather(ids_sorted)
    y = torch.zeros_like(x, dtype=torch.float32)
    for j in range(TOPK):                                # fixed trip count, slot-indexed: capturable
        y += expert_fwd(x, STG_W["w1"][j].view(torch.float4_e2m1fn_x2), STG_S["w1"][j].view(E8), STG_W["w3"][j].view(torch.float4_e2m1fn_x2), STG_S["w3"][j].view(E8),
                        STG_W["w2"][j].view(torch.float4_e2m1fn_x2), STG_S["w2"][j].view(E8), w_sorted[j].view(1, 1), limit)
    y += shared_fwd(x)
    return y.type_as(x)

torch.manual_seed(0); res = {"equal": 0, "n": 0, "cold_hits": 0}
xs = [torch.randn(1, DIM, dtype=torch.bfloat16) for _ in range(32)]
for x in xs:
    a = moe_ref(x); b = moe_static(x); res["n"] += 1; res["equal"] += int(torch.equal(a, b))
    _, idx = gate_fwd(x); res["cold_hits"] += sum(int(i) in COLD for i in idx[0].tolist())
    if not torch.equal(a, b): print("  mismatch max|d|", (a.float() - b.float()).abs().max().item())
print("bit-identical static vs reference:", res)
# timing
def bench(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
x0 = xs[0]; res["ref_us"] = bench(lambda: moe_ref(x0)); res["static_eager_us"] = bench(lambda: moe_static(x0))
# CUDA graph of the static version
sx = x0.clone(); sy = torch.empty_like(x0)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): sy.copy_(moe_static(sx))
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): sy.copy_(moe_static(sx))
ok = 0
for x in xs[:16]:
    sx.copy_(x); g.replay(); torch.cuda.synchronize(); ok += int(torch.equal(sy, moe_ref(x)))
res["graph_equal_ref"] = f"{ok}/16"; res["static_graph_us"] = bench(lambda: g.replay())
print(json.dumps(res, indent=1)); json.dump(res, open("/work/scratch/moe-static-spike.json", "w"), indent=1)
