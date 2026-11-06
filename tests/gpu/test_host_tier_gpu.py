"""GPU half of the T2+T3 tests: cudaHostRegister + UVA views, the Engram gather through the host tier, expert-pool
views feeding the reference fp4_gemm, and the loader end to end on the mini checkpoint. Skipped without CUDA (Mac).

Run on the box (orchestrator), from the repo checkout, inside the library image with the GB300 by CDI:

  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    -v "$HOME/hermes/gb300-dsv41-decode":/work -w /work \
    -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro -e DSV41_MODEL_DIR=/ref \
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_host_tier_gpu.py

(`python3 tests/gpu/test_host_tier_gpu.py` runs every test without pytest; `python3 -m pytest -q tests/gpu` works
too if pytest is in the image. The jit-cache mount targets assume the tools' default cache dirs; match whatever the
T1 runs used. `DSV41_MODEL_DIR` is optional: with it, the gather and fp4_gemm tests use real checkpoint bytes
(`layers.1.engram.embed.*` first 2M rows, `layers.0.ffn.experts.0-5`); without it, random valid bytes.)
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from runtime.host_tier import HostBuffer, mem_available_bytes, register_all  # noqa: E402
from runtime.refmodel import tilelang_available  # noqa: E402

MODEL_DIR = os.environ.get("DSV41_MODEL_DIR")
E8M0 = torch.float8_e8m0fnu
FP4 = torch.float4_e2m1fn_x2


def _need_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")


def _need_tilelang():
    _need_cuda()
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _real_index():
    if MODEL_DIR and os.path.isdir(MODEL_DIR):
        from runtime.ckpt import ShardIndex

        return ShardIndex.scan(MODEL_DIR)
    return None


# --------------------------------------------------------------------------------------------- 1. register + view
def test_register_chunks_and_cuda_view():
    _need_cuda()
    n = 64 * 1024 * 1024 + 4096 * 3 + 123  # not a chunk multiple
    hb = HostBuffer(n, "t1")
    g = torch.Generator().manual_seed(1)
    hb.cpu.copy_(torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8))
    secs = hb.register(chunk=16 * 1024 * 1024)  # 5 chunks: exercises the chunk loop
    assert len(hb.chunks) == 5 and hb.is_registered()
    u8 = hb.cuda_view(torch.uint8, (n,))
    assert u8.device.type == "cuda" and u8.data_ptr() == hb.ptr
    # a memcpy may not cross a registration chunk (measured: cudaErrorInvalidValue); kernels may. Read chunk-wise.
    assert torch.equal(hb.copy_chunks(torch.empty(n, dtype=torch.uint8, device="cuda")).cpu(), hb.cpu)
    assert int(u8.sum()) == int(hb.cpu.sum())  # kernel over the whole alias across all 5 chunks
    # writes through the host side are visible to the GPU (same memory, no copy)
    hb.cpu[:16] = 7
    torch.cuda.synchronize()
    assert int(u8[:16].sum()) == 7 * 16
    v = hb.cuda_view(torch.float8_e4m3fn, (1000, 256), offset=4096)
    assert v.data_ptr() == hb.ptr + 4096 and v.dtype == torch.float8_e4m3fn
    print(f"  register {n / 2**20:.1f} MiB in 5 chunks: {secs * 1e3:.1f} ms")
    del u8, v
    hb.release()
    assert not hb.chunks
    hb.close()


def test_register_rollback_on_failure():
    _need_cuda()
    real = torch.cuda.cudart()

    class Flaky:  # third chunk fails; the first two must be unregistered again
        def __init__(self):
            self.calls, self.unreg = 0, []

        def cudaHostRegister(self, ptr, n, flags):
            self.calls += 1
            return 1 if self.calls == 3 else real.cudaHostRegister(ptr, n, flags)

        def cudaHostUnregister(self, ptr):
            self.unreg.append(ptr)
            return real.cudaHostUnregister(ptr)

    flaky = Flaky()
    orig = torch.cuda.cudart
    torch.cuda.cudart = lambda: flaky
    hb = HostBuffer(8 * 1024 * 1024, "t2")
    try:
        try:
            hb.register(chunk=2 * 1024 * 1024)
            raise AssertionError("register should have failed")
        except RuntimeError as e:
            assert "cudaHostRegister" in str(e)
        assert hb.chunks == [] and len(flaky.unreg) == 2
    finally:
        torch.cuda.cudart = orig
    hb.register()  # the memory is registrable again: the rollback really unregistered
    hb.release()
    hb.close()


def test_register_all_mem_available():
    _need_cuda()
    bufs = [HostBuffer(4 * 1024 * 1024, f"b{i}") for i in range(3)]
    log = register_all(bufs, reserve_bytes=1 << 30)
    assert all(b.is_registered() for b in bufs)
    if mem_available_bytes() is not None:
        assert log["before"]["mem_available_GiB"] is not None and log["after"]["mem_available_GiB"] is not None
    for b in bufs:
        b.release()
        b.close()


# --------------------------------------------------------------------------------------------- 2. Engram gather
def test_engram_gather_uva_vs_hbm_1e6_ids():
    """torch.equal of F.embedding through the host tier vs an HBM copy, 10^6 random ids, plus the reference's
    dequant (model.py:317-320) on both."""
    _need_cuda()
    rows, dim = 2_000_000, 256
    hw = HostBuffer(rows * dim, "engram_w")
    hs = HostBuffer(rows * dim // 32, "engram_s")
    idx = _real_index()
    src = "random"
    if idx is not None and "layers.1.engram.embed.weight" in idx:
        idx.read_into("layers.1.engram.embed.weight", hw.memoryview(), 0, rows * dim)
        idx.read_into("layers.1.engram.embed.scale", hs.memoryview(), 0, rows * dim // 32)
        src = "real layers.1.engram.embed first 2M rows"
    else:
        g = torch.Generator().manual_seed(2)
        hw.cpu.copy_((torch.randn(rows * dim, generator=g, dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8))
        hs.cpu.copy_(torch.randint(118, 131, (rows * dim // 32,), generator=g, dtype=torch.uint8))
    hw.register()
    hs.register()
    w_uva = hw.cuda_view(torch.float8_e4m3fn, (rows, dim))
    s_uva = hs.cuda_view(E8M0, (rows, dim // 32))
    w_hbm = w_uva.view(torch.uint8).clone().view(torch.float8_e4m3fn)  # byte copy into HBM (no fp8 kernels)
    s_hbm = s_uva.view(torch.uint8).clone().view(E8M0)
    assert w_hbm.data_ptr() != w_uva.data_ptr()
    ids = torch.randint(0, rows, (1_000_000,), device="cuda")
    a, b = F.embedding(ids, w_uva), F.embedding(ids, w_hbm)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    sa, sb = F.embedding(ids, s_uva), F.embedding(ids, s_hbm)
    assert torch.equal(sa.view(torch.uint8), sb.view(torch.uint8))
    deq = lambda v, s: (v.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)  # noqa: E731
    assert torch.equal(deq(a, sa), deq(b, sb))

    def timed(w, s, ids24, it=200):
        for _ in range(5):
            F.embedding(ids24, w), F.embedding(ids24, s)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(it):
            F.embedding(ids24, w), F.embedding(ids24, s)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it * 1e6

    ids24 = torch.randint(0, rows, (24,), device="cuda")
    print(f"  gather 24 rows (weight+scale) [{src}]: UVA {timed(w_uva, s_uva, ids24):.1f} us, "
          f"HBM {timed(w_hbm, s_hbm, ids24):.1f} us")
    del w_uva, s_uva, a, sa
    hw.release(); hs.release(); hw.close(); hs.close()  # noqa: E702


# --------------------------------------------------------------------------------------------- 3. pool -> fp4_gemm
def test_expert_pool_views_feed_reference_fp4_gemm():
    """Hot (HBM slot) and cold (UVA host slot) expert views through the unchanged reference Expert/linear/fp4_gemm
    give outputs torch.equal to the same expert with contiguous HBM copies of its bytes, M=1 and M=6."""
    _need_tilelang()
    from runtime.expert_pool import ExpertPool, TIER_HOST
    from runtime.loader import _ParamFactoriesToMeta
    from runtime.placement import HotMap
    from runtime.refmodel import import_reference

    ref = import_reference()
    torch.set_default_dtype(torch.bfloat16)
    dim, inter, n_exp = 5120, 2304, 6
    hot = HotMap({0: [0, 3]}, 1, n_exp, "t")
    pool = ExpertPool(1, n_exp, dim, inter, hot, "cuda")
    idx = _real_index()
    g = torch.Generator().manual_seed(3)
    for e in range(n_exp):
        for p in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                name = f"layers.0.ffn.experts.{e}.{p}.{kind}"
                if pool.tier(0, e) == TIER_HOST:
                    mv = pool.host_memoryview(0, e, p, kind)
                    if idx is not None:
                        idx.read_into(name, mv)
                    else:
                        t = torch.frombuffer(mv, dtype=torch.uint8)
                        t.copy_(torch.randint(0, 256, (len(mv),), generator=g, dtype=torch.uint8) if kind == "weight"
                                else torch.randint(118, 123, (len(mv),), generator=g, dtype=torch.uint8))
                else:
                    dst = pool.hbm_dst(0, e, p, kind)
                    if idx is not None:
                        buf = torch.empty(dst.numel(), dtype=torch.uint8)
                        idx.read_into(name, buf)
                    else:
                        buf = (torch.randint(0, 256, (dst.numel(),), generator=g, dtype=torch.uint8) if kind == "weight"
                               else torch.randint(118, 123, (dst.numel(),), generator=g, dtype=torch.uint8))
                    dst.reshape(-1).view(torch.uint8).copy_(buf)
    pool.host.register()
    with torch.device("cuda"), _ParamFactoriesToMeta():
        experts = nn.ModuleList([ref.Expert(dim, inter, dtype=FP4, swiglu_limit=10.0) for _ in range(n_exp)])
        twins = nn.ModuleList([ref.Expert(dim, inter, dtype=FP4, swiglu_limit=10.0) for _ in range(n_exp)])
    fake = types.SimpleNamespace(layers=[types.SimpleNamespace(ffn=types.SimpleNamespace(experts=experts))])
    pool.bind(fake, cuda_views=True)
    pool.install_counters(fake)

    def hbm_copy(t):  # byte copy into fresh contiguous HBM, same dtype
        return t.detach().view(torch.uint8).clone().view(t.dtype)

    for e in range(n_exp):
        ex, twin = experts[e], twins[e]
        for p in ("w1", "w2", "w3"):
            src, dst = getattr(ex, p), getattr(twin, p)
            dst.weight = nn.Parameter(hbm_copy(src.weight), requires_grad=False)
            dst.scale = nn.Parameter(hbm_copy(src.scale), requires_grad=False)
            dst.weight.scale = dst.scale
            assert dst.weight.data_ptr() != src.weight.data_ptr() and dst.weight.dtype == FP4
        on_host = pool.tier(0, e) == TIER_HOST
        if on_host:
            assert pool.host.ptr <= ex.w1.weight.data_ptr() < pool.host.ptr + pool.host.nbytes
        for M in (1, 6):
            x = torch.randn(M, dim, device="cuda", dtype=torch.bfloat16)
            w = torch.rand(M, 1, device="cuda", dtype=torch.float32)
            y_pool, y_twin = ex(x, w), twin(x, w)
            assert torch.equal(y_pool, y_twin), (e, M, on_host)
            assert torch.isfinite(y_pool.float()).all()
    s = pool.counter_summary()
    assert s["cold_calls"] == 2 * 4 and s["hot_calls"] == 2 * 2, s
    print(f"  fp4_gemm via pool views == HBM copies for {n_exp} experts x M=1,6; counters {s}")
    del experts, fake
    pool.remove_counters()


# --------------------------------------------------------------------------------------------- 4. loader end to end
def _load_mini(tmp, map_arg, verify=True):
    from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    ref = import_reference()
    import engram

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    ck = make_mini_checkpoint(tmp) if not os.path.exists(os.path.join(tmp, "hot_map.json")) else {
        "model_dir": tmp, "map": os.path.join(tmp, "hot_map.json")}
    m = map_arg if map_arg is not None else ck["map"]
    model = build_and_load(ck["model_dir"], m, "cuda", tokenizer=object(), verify=verify, io_threads=4,
                           stage_bytes=1 << 20, log=lambda *a: None)
    return ref, model


def test_build_and_load_mini_gpu_verify():
    _need_tilelang()
    with tempfile.TemporaryDirectory() as tmp:
        ref, model = _load_mini(tmp, None)
        res = model.runtime_load
        assert res.verify["mismatch"] == [] and res.verify["ok"] == res.verify["n"]
        for n, hb in res.host_buffers.items():
            assert hb.is_registered(), n
        e = model.layers[0].engram.embed
        assert e.weight.device.type == "cuda" and e.weight.data_ptr() == res.host_buffers["layers.0.engram.embed.weight"].ptr
        cold = model.layers[0].ffn.experts[1].w1.weight  # expert 1 is cold in layer 0 (MINI_HOT_MAP)
        assert cold.device.type == "cuda" and res.pool.host.ptr <= cold.data_ptr() < res.pool.host.ptr + res.pool.host.nbytes
        assert res.pool.loc.device.type == "cuda" and res.pool.loc.dtype == torch.int32
        print(f"  mini load on GPU: {res.verify['n']} tensors verified; {res.summary()['bytes_GiB']}")
        del model, e, cold
        torch.cuda.synchronize()
        res.release()


def test_mini_forward_tiered_equals_all_hbm():
    """Reference forward (prefill 16 + 2 teacher-forced decode steps) on the mini checkpoint: logits with the
    tiered placement (cold experts + Engram over UVA) torch.equal those with every expert in HBM, and a replay."""
    _need_tilelang()
    with tempfile.TemporaryDirectory() as tmp:
        _, tiered = _load_mini(tmp, None, verify=False)
        _, all_hbm = _load_mini(tmp, {"0": list(range(8)), "1": list(range(8))}, verify=False)
        ids = torch.randint(0, 2000, (1, 19), generator=torch.Generator().manual_seed(4)).cuda()  # before the default device flips
        torch.set_default_device("cuda")  # generate.py:134: the forward builds index tensors on the default device
        try:
            def run(model):
                outs = [model.forward(ids[:, :16], 0)[1]]
                for pos in (16, 17):
                    outs.append(model.forward(ids[:, pos:pos + 1], pos)[1])
                return outs

            a = run(tiered)
            b = run(all_hbm)
            a2 = run(tiered)  # self-replay (decision 14)
        finally:
            torch.set_default_device("cpu")
        for i, (x, y, x2) in enumerate(zip(a, b, a2)):
            assert torch.isfinite(x).all(), i
            assert torch.equal(x, y), f"step {i}: tiered != all-HBM, max diff {(x - y).abs().max().item()}"
            assert torch.equal(x, x2), f"step {i}: replay differs"
        s = tiered.runtime_load.pool.counter_summary()
        assert s["cold_calls"] > 0, s
        print(f"  forward: 3 steps torch.equal tiered vs all-HBM and on replay; tiered cold share {s['cold_share']:.2f}")


TESTS = [test_register_chunks_and_cuda_view, test_register_rollback_on_failure, test_register_all_mem_available,
         test_engram_gather_uva_vs_hbm_1e6_ids, test_expert_pool_views_feed_reference_fp4_gemm,
         test_build_and_load_mini_gpu_verify, test_mini_forward_tiered_equals_all_hbm]

if __name__ == "__main__":
    import traceback

    results = {}
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            results[t.__name__] = "PASS"
        except unittest.SkipTest as e:
            results[t.__name__] = f"SKIP ({e})"
        except Exception:
            traceback.print_exc()
            results[t.__name__] = "FAIL"
        print(f"{results[t.__name__]:4s} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
