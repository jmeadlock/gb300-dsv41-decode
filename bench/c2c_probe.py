#!/usr/bin/env python3
"""c2c_probe.py — recipe-A thesis bound: how fast can the GB300 read pinned Grace memory, and what does the reference
fp4_gemm cost when its weights live in registered host memory (UVA view) instead of HBM?
Measures (all on real expert bytes from the checkpoint, layers.0.ffn.experts.0..k):
  1. cudaHostRegister cost for an exact-size anonymous mmap buffer (GB/s of registration)
  2. copy-engine H2D from registered host memory (GB/s), 18.8 MB expert granularity and 1 GiB bulk
  3. kernel-side read of host-mapped memory: torch.sum over a UVA view (GB/s), and
  4. fp4_gemm (reference kernel) M=1 and M=6 with w1 in HBM vs w1 as a UVA view of registered host memory (µs/call)
  5. Engram-style gather: index_select of 24 random 264-B rows from a 1 GiB host-registered e4m3 table (µs/call) vs HBM copy
Run inside the library image with /ref (checkpoint root), /out writable, GB300 only."""
import ctypes, json, mmap, os, struct, sys, time, torch
sys.path.insert(0, "/ref/inference")
torch.set_default_dtype(torch.bfloat16)
import kernel as K  # noqa: E402
from safetensors import safe_open  # noqa: E402

dev = torch.device("cuda"); out = {}
cudart = torch.cuda.cudart()

def shard_of(name):
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
    raise KeyError(name)
def load(name):
    with safe_open("/ref/" + shard_of(name), "pt", device="cpu") as f:
        return f.get_tensor(name)

class _CAI:
    def __init__(self, ptr, nbytes): self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}

def host_registered(nbytes):
    """exact-size anonymous mmap, cudaHostRegister(Mapped=2|Portable=1), returns (mmap, host_ptr, cuda_u8_view)"""
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    buf = (ctypes.c_char * nbytes).from_buffer(mm); ptr = ctypes.addressof(buf)
    t0 = time.time(); err = cudart.cudaHostRegister(ptr, nbytes, 3); dt = time.time() - t0
    assert err == 0, err
    cu = torch.as_tensor(_CAI(ptr, nbytes), device="cuda")
    assert cu.device.type == "cuda" and cu.data_ptr() == ptr
    return mm, buf, ptr, cu, dt

def bench(fn, iters, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters

# 1+2+3: 1 GiB buffer
N1 = 1 << 30
mm, buf, ptr, cu, treg = host_registered(N1)
host_t = torch.frombuffer(mm, dtype=torch.uint8); host_t[:] = torch.arange(N1, dtype=torch.uint8) if False else 1
out["register_1GiB_s"] = round(treg, 3); out["register_GBps"] = round(N1 / treg / 1e9, 1)
dst = torch.empty(N1, dtype=torch.uint8, device=dev)
t = bench(lambda: dst.copy_(cu, non_blocking=True), 10); out["h2d_copy_1GiB_GBps"] = round(N1 / t / 1e9, 1)
t = bench(lambda: cu.sum(), 10); out["kernel_read_host_1GiB_GBps"] = round(N1 / t / 1e9, 1)
t = bench(lambda: dst.sum(), 10); out["kernel_read_hbm_1GiB_GBps"] = round(N1 / t / 1e9, 1)
# expert-granularity copies: 18.8 MB each, 36 of them (the M1 cold-set upper bound)
E = 18800640; n_e = 36
slots = torch.empty(n_e * E, dtype=torch.uint8, device=dev)
def copy_experts():
    for i in range(n_e): slots[i * E:(i + 1) * E].copy_(cu[i * E:(i + 1) * E], non_blocking=True)
t = bench(copy_experts, 10); out["h2d_36_experts_ms"] = round(t * 1e3, 3); out["h2d_expert_GBps"] = round(n_e * E / t / 1e9, 1)
t = bench(lambda: slots[:E].copy_(cu[:E], non_blocking=True), 20); out["h2d_1_expert_us"] = round(t * 1e6, 1)

# 4: fp4_gemm HBM vs UVA weights, real bytes
w = load("layers.0.ffn.experts.0.w1.weight"); s = load("layers.0.ffn.experts.0.w1.scale")
Nn, K2 = w.shape; Kd = K2 * 2
w_hbm = w.view(torch.float4_e2m1fn_x2).to(dev).contiguous(); s_hbm = s.to(dev).contiguous()
mw, bw, pw, cw, _ = host_registered(w.numel()); torch.frombuffer(mw, dtype=torch.int8)[:] = w.flatten()
ms_, bs_, ps_, cs_, _ = host_registered(s.numel()); torch.frombuffer(ms_, dtype=torch.uint8)[:] = s.view(torch.uint8).flatten()
w_uva = cw.view(torch.float4_e2m1fn_x2).view(Nn, K2); s_uva = cs_.view(torch.float8_e8m0fnu).view(Nn, Kd // 32)
res4 = {}
for M in (1, 6):
    x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16); xq, xs = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    y_h = K.fp4_gemm(xq, xs, w_hbm, s_hbm, torch.float8_e8m0fnu, act_block_size=32)
    y_u = K.fp4_gemm(xq, xs, w_uva, s_uva, torch.float8_e8m0fnu, act_block_size=32)
    eq = bool(torch.equal(y_h, y_u))
    th = bench(lambda: K.fp4_gemm(xq, xs, w_hbm, s_hbm, torch.float8_e8m0fnu, act_block_size=32), 50)
    tu = bench(lambda: K.fp4_gemm(xq, xs, w_uva, s_uva, torch.float8_e8m0fnu, act_block_size=32), 50)
    res4[f"M{M}"] = {"hbm_us": round(th * 1e6, 1), "uva_us": round(tu * 1e6, 1), "equal": eq, "uva_eff_GBps": round((w.numel() + s.numel()) / tu / 1e9, 1)}
    print("fp4_gemm", M, res4[f"M{M}"], flush=True)
out["fp4_gemm_w1_real"] = res4

# 5: Engram-style gather (24 rows of 256 B e4m3 + 8 B e8m0 from a 1 GiB table)
rows = N1 // 256
tab_uva = cu.view(torch.float8_e4m3fn).view(rows, 256)
tab_hbm = dst.view(torch.float8_e4m3fn).view(rows, 256)
idx = torch.randint(0, rows, (24,), device=dev)
tg_u = bench(lambda: torch.index_select(tab_uva, 0, idx), 200); tg_h = bench(lambda: torch.index_select(tab_hbm, 0, idx), 200)
out["engram_gather_24rows_uva_us"] = round(tg_u * 1e6, 1); out["engram_gather_24rows_hbm_us"] = round(tg_h * 1e6, 1)
# with a fresh random idx each call (defeats any caching)
idxs = [torch.randint(0, rows, (24,), device=dev) for _ in range(64)]; i = [0]
def g(): i[0] = (i[0] + 1) % 64; return torch.index_select(tab_uva, 0, idxs[i[0]])
out["engram_gather_24rows_uva_randidx_us"] = round(bench(g, 200) * 1e6, 1)

print(json.dumps(out, indent=1)); json.dump(out, open("/out/c2c-probe.json", "w"), indent=1)
