#!/usr/bin/env python3
"""overlap_probe.py — does a copy-engine H2D stream from registered Grace memory run concurrently with compute on the GB300,
and at what cost to each? HBM-resident fp4_gemm loop (reference kernel, real expert bytes) on stream A; 1 GiB H2D copies on stream B."""
import ctypes, json, mmap, os, struct, sys, time, torch
sys.path.insert(0, "/ref/inference"); torch.set_default_dtype(torch.bfloat16)
import kernel as K  # noqa: E402
from safetensors import safe_open  # noqa: E402
dev = torch.device("cuda"); cudart = torch.cuda.cudart()
def shard_of(name):
    for f in sorted(x for x in os.listdir("/ref") if x.endswith(".safetensors")):
        with open("/ref/" + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        if name in hdr: return f
def load(name):
    with safe_open("/ref/" + shard_of(name), "pt", device="cpu") as f: return f.get_tensor(name)
class _CAI:
    def __init__(self, ptr, n): self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}
N1 = 1 << 30
mm = mmap.mmap(-1, N1, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS); buf = (ctypes.c_char * N1).from_buffer(mm); ptr = ctypes.addressof(buf)
assert cudart.cudaHostRegister(ptr, N1, 3) == 0
cu = torch.as_tensor(_CAI(ptr, N1), device="cuda"); dst = torch.empty(N1, dtype=torch.uint8, device=dev)
w = load("layers.0.ffn.experts.0.w1.weight"); s = load("layers.0.ffn.experts.0.w1.scale")
Nn, K2 = w.shape; Kd = K2 * 2
w_hbm = w.view(torch.float4_e2m1fn_x2).to(dev).contiguous(); s_hbm = s.to(dev).contiguous()
x = torch.randn(6, Kd, device=dev, dtype=torch.bfloat16); xq, xs = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
gemm = lambda: K.fp4_gemm(xq, xs, w_hbm, s_hbm, torch.float8_e8m0fnu, act_block_size=32)
# also a bandwidth-heavy HBM kernel: 1 GiB HBM->HBM copy
src2 = torch.empty(N1, dtype=torch.uint8, device=dev); dst2 = torch.empty(N1, dtype=torch.uint8, device=dev)
hbmcopy = lambda: dst2.copy_(src2, non_blocking=True)
sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
def timed(fn, n, stream):
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
        stream.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(stream)
        for _ in range(n): fn()
        e1.record(stream); stream.synchronize(); return e0.elapsed_time(e1) / n
out = {}
out["gemm_alone_us"] = round(timed(gemm, 200, sA) * 1e3, 1)
out["h2d_alone_ms"] = round(timed(lambda: dst.copy_(cu, non_blocking=True), 10, sB), 3)
out["hbmcopy_alone_ms"] = round(timed(hbmcopy, 10, sA), 3)
# concurrent: gemm loop on A while H2D on B
def concurrent(kfn, nk, cfn, nc):
    torch.cuda.synchronize()
    eA0, eA1, eB0, eB1 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    with torch.cuda.stream(sB):
        eB0.record(sB)
        for _ in range(nc): cfn()
        eB1.record(sB)
    with torch.cuda.stream(sA):
        eA0.record(sA)
        for _ in range(nk): kfn()
        eA1.record(sA)
    torch.cuda.synchronize(); return eA0.elapsed_time(eA1) / nk, eB0.elapsed_time(eB1) / nc
g, c = concurrent(gemm, 2000, lambda: dst.copy_(cu, non_blocking=True), 40)
out["gemm_with_h2d_us"] = round(g * 1e3, 1); out["h2d_with_gemm_ms"] = round(c, 3)
g, c = concurrent(hbmcopy, 40, lambda: dst.copy_(cu, non_blocking=True), 40)
out["hbmcopy_with_h2d_ms"] = round(g, 3); out["h2d_with_hbmcopy_ms"] = round(c, 3)
out["h2d_alone_GBps"] = round(N1 / out["h2d_alone_ms"] / 1e6, 1); out["h2d_with_gemm_GBps"] = round(N1 / out["h2d_with_gemm_ms"] / 1e6, 1)
out["hbmcopy_alone_GBps_rw"] = round(2 * N1 / out["hbmcopy_alone_ms"] / 1e6, 1)
print(json.dumps(out, indent=1)); json.dump(out, open("/out/overlap-probe.json", "w"), indent=1)
