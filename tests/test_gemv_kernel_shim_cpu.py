"""Mac CPU test of the T9b-1 Triton kernels' bodies (runtime/gemv_mxfp8.py: t9b_mxfp8_gemv, t9b_splitk_reduce,
t9b_pair_scale_probe), executed program by program through tests/triton_shim.py (numpy stand-in for the Triton
subset they use). `.venv/bin/python -m pytest -q tests/test_gemv_kernel_shim_cpu.py`

Checks the kernels' index math and control flow -- not Triton compilation, layouts, tl.dot legality on sm_103 or
speed (box only, tests/gpu/test_gemv_gpu.py + bench/gemv_bench.py):
  * every (M, N, K, config) case: the kernel output == gemv_torch with the same split (bit for bit), on data whose
    32-product block sums are exact in fp32 (so only the cross-block order -- the thing split-K changes -- can show);
  * every workspace slot ws[ks, m, n] is written and equals the fp32 K-order chain over split ks's K range alone;
  * no out-of-bounds load/store, no two lanes storing to one address, every output written, nothing past M / N;
  * both block-partial forms (MMA 0/1), EVEN and tail (KU remainder, short last split), split 1 (direct bf16 store)
    and > 1 (workspace + reduce), N not a multiple of BLOCK_N or of 32, M = 1, 3, 6, 8;
  * the device scale probe in both modes == pair_scale on all 65,536 code pairs.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import triton_shim
from runtime import gemv_mxfp8 as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu


@pytest.fixture(scope="module")
def K():
    """The kernel module imported with the shim as `triton` (a private copy; runtime.gemv_mxfp8 is untouched).
    $GEMV_SHIM_MODULE points it at another file (mutation checks of this test's sensitivity)."""
    return triton_shim.load_module_under_shim(os.environ.get("GEMV_SHIM_MODULE")
                                              or os.path.join(REPO, "runtime", "gemv_mxfp8.py"))


def _exact_fp8(*shape, g):
    """e4m3 with exponent field 6..8 (|v| in [0.5, 4)): any 32 products of two of them sum exactly in fp32."""
    e = torch.randint(6, 9, shape, generator=g, dtype=torch.int32)
    m = torch.randint(0, 8, shape, generator=g, dtype=torch.int32)
    sg = torch.randint(0, 2, shape, generator=g, dtype=torch.int32)
    return ((sg << 7) | (e << 3) | m).to(torch.uint8).view(FP8)


def _operands(M, N, K_, seed):
    g = torch.Generator().manual_seed(seed)
    xq, w = _exact_fp8(M, K_, g=g), _exact_fp8(N, K_, g=g)
    xs = torch.randint(110, 131, (M, K_ // 32), generator=g, dtype=torch.int32).to(torch.uint8).view(E8)
    s = torch.randint(118, 132, ((N + 31) // 32, K_ // 32), generator=g, dtype=torch.int32).to(torch.uint8).view(E8)
    return xq, xs, w, s


CASES = [  # (M, N, K, GemvConfig(block_n, ku, warps, stages, split_k, mma))
    (1, 64, 256, G.GemvConfig(32, 4, 4, 2, 1, 0)),      # split 1: direct bf16 store, EVEN
    (1, 100, 1280, G.GemvConfig(32, 4, 4, 2, 7, 0)),    # KB 40 / split 7 -> PER 6 (KU tail) + short last split
    (3, 96, 640, G.GemvConfig(64, 2, 4, 2, 4, 0)),      # N % BLOCK_N != 0; PER 5 with KU 2 (tail)
    (6, 40, 384, G.GemvConfig(32, 8, 8, 3, 5, 0)),      # N % 32 != 0; KB 12 / 5 -> (3, 4); KU 8 > PER
    (8, 128, 1024, G.GemvConfig(32, 4, 4, 2, 4, 0)),    # EVEN split: KB 32 / 4 -> PER 8
    (1, 64, 256, G.GemvConfig(32, 4, 4, 2, 1, 1)),      # MMA, split 1
    (3, 100, 1280, G.GemvConfig(32, 4, 4, 2, 7, 1)),    # MMA, tail + short split, N % 32 != 0
    (6, 96, 640, G.GemvConfig(64, 8, 4, 2, 3, 1)),      # MMA, PER 7 with KU 8
    (8, 40, 1024, G.GemvConfig(128, 2, 4, 2, 8, 1)),    # MMA, BLOCK_N > N, EVEN split
    (8, 33, 288, G.GemvConfig(32, 2, 4, 2, 9, 1)),      # KB 9, split 9 -> PER 1: every program one block
]


@pytest.mark.parametrize("M,N,Kd,cfg", CASES, ids=[f"M{c[0]}-N{c[1]}-K{c[2]}-s{c[3].split_k}-mma{c[3].mma}"
                                                    for c in CASES])
def test_kernel_body_equals_split_mirror(K, M, N, Kd, cfg):
    xq, xs, w, s = _operands(M, N, Kd, seed=M * 131 + N + Kd + cfg.split_k)
    plan = K.launch_plan(cfg, M, N, Kd)
    ws = torch.full((plan["NSPLIT"], M, N), float("nan"), dtype=torch.float32)
    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16)
    triton_shim.STORES.clear()
    K.t9b_mxfp8_gemv.launches.clear()
    K.t9b_splitk_reduce.launches.clear()
    K._launch(xq, xs, w, s, out, cfg, ws if plan["SPLIT"] else out)
    # the launches the plan describes
    assert K.t9b_mxfp8_gemv.launches[0]["grid"] == plan["grid"]
    ce = K.t9b_mxfp8_gemv.launches[0]["constexprs"]
    assert (ce["BLOCK_M"], ce["BLOCK_N"], ce["KU"], ce["EVEN"], ce["MMA"], ce["SPLIT"]) == \
        (plan["BLOCK_M"], cfg.block_n, cfg.ku, plan["EVEN"], bool(cfg.mma), plan["SPLIT"])
    assert len(K.t9b_splitk_reduce.launches) == (1 if plan["SPLIT"] else 0)
    # every output written, == the torch mirror of the same split order, bit for bit
    assert not torch.isnan(out.float()).any()
    want = G.gemv_torch(xq, xs, w, s, split_k=cfg.split_k)
    assert torch.equal(out, want), (out.float() - want.float()).abs().max()
    if plan["SPLIT"]:
        assert not torch.isnan(ws).any()                     # every [ks, m, n] slot written
        per = plan["PER"]
        for ks in range(plan["NSPLIT"]):
            lo, hi = ks * per, min((ks + 1) * per, Kd // 32)
            chain = G.gemv_torch_f32(xq[:, lo * 32:hi * 32].contiguous(), xs[:, lo:hi].contiguous(),
                                     w[:, lo * 32:hi * 32].contiguous(), s[:, lo:hi].contiguous(), 1)
            assert torch.equal(ws[ks].view(torch.int32), chain.view(torch.int32)), ks
        # the reduce is the split-order fold of exactly these partials
        fold = torch.zeros(M, N)
        for ks in range(plan["NSPLIT"]):
            fold = fold + ws[ks]
        assert torch.equal(out, fold.to(torch.bfloat16))
    # the split is not a no-op on this data: some fp32 result differs from the single chain when the split really
    # regroups (PER > 1; with PER == 1 the split-order fold IS the K-order chain -- asserted too)
    same = torch.equal(G.gemv_torch_f32(xq, xs, w, s, cfg.split_k), G.gemv_torch_f32(xq, xs, w, s, 1))
    if plan["NSPLIT"] > 1 and plan["PER"] > 1:
        assert not same
    elif plan["PER"] == 1:
        assert same


def test_rows_past_m_and_columns_past_n_untouched(K):
    """The kernel writes only out[:M, :N] even though BLOCK_M (16 with MMA) and BLOCK_N exceed them: `out` is a
    [M, N] view (row stride 2N) of a bigger sentinel buffer, whose other elements must keep their sentinel."""
    M, N, Kd = 3, 40, 512
    xq, xs, w, s = _operands(M, N, Kd, seed=3)
    for cfg in (G.GemvConfig(64, 4, 4, 2, 3, 1), G.GemvConfig(64, 4, 4, 2, 1, 0)):
        big = torch.full((M + 2, 2 * N), -7.0, dtype=torch.bfloat16)
        out = big[:M, :N]
        assert out.stride(0) == 2 * N
        plan = K.launch_plan(cfg, M, N, Kd)
        ws = torch.full((plan["NSPLIT"], M, N), float("nan"), dtype=torch.float32)
        K._launch(xq, xs, w, s, out, cfg, ws if plan["SPLIT"] else out)
        assert torch.equal(big[:M, :N], G.gemv_torch(xq, xs, w, s, split_k=cfg.split_k)), cfg
        assert bool((big[:M, N:] == -7).all()) and bool((big[M:] == -7).all()), cfg


def test_reduce_order_is_split_order(K):
    """t9b_splitk_reduce folds ws over splits in order 0..NSPLIT-1 from +0.0. Crafted partials make the order visible
    even after the bf16 store: (2^24 + 1) - 2^24 = 0 in that order, but 1 in reverse ((-2^24 + 1) + 2^24)."""
    M, N, NS = 2, 3, 3
    ws = torch.zeros(NS, M, N)
    ws[:, 0, 0] = torch.tensor([2.0 ** 24, 1.0, -(2.0 ** 24)])
    ws[:, 0, 1] = torch.tensor([1.0, 2.0 ** 24, -(2.0 ** 24)])
    ws[:, 1, 2] = torch.tensor([-(2.0 ** 24), 1.0, 2.0 ** 24])
    ws[:, 1, 0] = torch.tensor([3.0, -1.5, 0.25])
    fwd = torch.zeros(M, N)
    for j in range(NS):
        fwd = fwd + ws[j]
    rev = torch.zeros(M, N)
    for j in reversed(range(NS)):
        rev = rev + ws[j]
    assert not torch.equal(fwd.to(torch.bfloat16), rev.to(torch.bfloat16))   # the data can tell the orders apart
    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16)
    K.t9b_splitk_reduce[(1,)](ws, out, M, N, NS, out.stride(0), BLOCK_R=K.REDUCE_BLOCK, num_warps=4)
    assert torch.equal(out, fwd.to(torch.bfloat16)), (out, fwd)
    assert float(out[0, 0]) == 0.0 and float(out[1, 2]) == 1.0


def test_probe_both_modes_all_pairs(K):
    c = torch.arange(256, dtype=torch.int32)
    a, b = c.repeat_interleave(256).to(torch.uint8), c.repeat(256).to(torch.uint8)
    want = G.pair_scale(a, b)
    nan = torch.isnan(want)
    for mode in (0, 1):
        out = torch.empty(a.numel(), dtype=torch.float32)
        with np.errstate(over="ignore"):   # mode 1 multiplies 2^127 x 2^127 -> inf, as intended
            K.t9b_pair_scale_probe[(K.triton.cdiv(a.numel(), 1024),)](a, b, out, a.numel(), BLOCK_P=1024, MODE=mode)
        assert torch.equal(torch.isnan(out), nan), mode
        assert torch.equal(out[~nan].view(torch.int32), want[~nan].view(torch.int32)), mode


def test_shim_rejects_out_of_bounds():
    """The shim's guards are live (else the 'no OOB' claims above would be vacuous)."""
    buf = np.zeros(4, np.float32)
    with pytest.raises(AssertionError):
        triton_shim.load(triton_shim.P(buf, np.array([0, 4])))
    with pytest.raises(AssertionError):
        triton_shim.store(triton_shim.P(buf, np.array([1, 1])), triton_shim.T(np.ones(2, np.float32)))
    with pytest.raises(AssertionError):
        triton_shim.dot(triton_shim.T(np.ones((8, 32))), triton_shim.T(np.ones((32, 32))))
