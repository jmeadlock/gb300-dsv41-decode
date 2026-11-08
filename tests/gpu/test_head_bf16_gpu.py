"""GPU tests of M2-T17 (docs/HANDOFF-M2-T17-BF16-HEAD.md): the head weight kept bf16 in HBM (its checkpoint dtype),
read by the T9b-5 fixed-order kernel (runtime/static_kernels.py t9b5_head_fp32) and widened in registers.
Skipped without CUDA (Mac); on the box a missing triton FAILS (it is what is under test). Same docker line as
tests/gpu/test_fp32_fixed_gpu.py, with this file (+ -e DSV41_MODEL_DIR=/models/<the checkpoint> for the real head, the
real-checkpoint draft gate, and the boot gate):

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_head_bf16_gpu.py

  (a) THE gate -- bit-equal: the bf16-stored head's logits == the fp32-stored head's (the same kernel on the widened
      copy: same products, same order -> must be exact; a mismatch is a finding, not a tolerance), over ALL 129,280
      logits, at R in {1, 2, 5, 6, 8}, on every launch config of the fp32 and bf16 sweeps, on the real head when
      $DSV41_MODEL_DIR is set (else a synthetic bf16 head); row-invariance R = 6 vs six R = 1 calls (6/6 rows)
      for the bf16 weight; the kernel == fused_glue_r2.fixed_dot_torch on the first 4,096 vocab rows;
  (b) replay: eager x2, a fresh launch, a CUDA graph replayed twice -> bit-equal;
  (c) HBM: the weight bytes of the two storages (target: -1.33 GB), printed;
  (d) head ms at R = 1 and R = 6, fp32-stored default vs bf16-stored (default table and the bf16 sweep), graph-timed,
      TB/s over the bytes actually read; the handoff's target line (<= 0.25 ms) is printed MET / MISSED (informative:
      the box gates are first_light / headline);
  (e) the loader + draft on the mini checkpoints on cuda: build_and_load(head_dtype="bf16") logits == the default
      build's, verify=True (bytes == the shard's), the Markov head bf16; the V2 draft gate (DraftStep == the reference
      forward_spec, whose heads are the same fixed-order modules) with bf16 heads, plus the exact P1 gate (GEMV + fused
      glue); on the real checkpoint (when set) the V2 draft gate too.
"""
from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

DIM, V = 5120, 129280
HEAD_TARGET_MS = 0.25
SWEEP_BF16 = [None, (1, 64, 4, 4, 1), (1, 32, 8, 4, 1), (1, 32, 16, 4, 1), (2, 32, 8, 4, 1), (8, 32, 4, 8, 1),
              (8, 16, 8, 8, 1), (8, 16, 16, 8, 1), (8, 8, 8, 4, 1), (8, 16, 8, 8, 2), (8, 16, 4, 4, 3)]
SWEEP_FP32 = [None, (1, 32, 4, 4, 1), (2, 32, 2, 4, 1), (8, 16, 4, 8, 1), (8, 8, 4, 4, 1)]
_W = {}


def _need():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")


def _mods():
    from runtime import fused_glue_r2 as R2
    from runtime import static_kernels as SK

    assert SK.HAVE_TRITON
    return R2, SK


def _head_weights():
    """(W bf16 [V, D] on cuda, label): the real head.weight as the checkpoint stores it when $DSV41_MODEL_DIR is
    set, else a synthetic bf16 head of the same shape. The fp32 copy is made per test (W.float(): the loader's widening)."""
    if "head" in _W:
        return _W["head"]
    md = os.environ.get("DSV41_MODEL_DIR")
    if md:
        from runtime.ckpt import ShardIndex

        src = ShardIndex.scan(md).read_tensor("head.weight")
        assert src.dtype == torch.bfloat16, f"the checkpoint stores the head as {src.dtype}"
        W = src.cuda()
        label = "real head.weight (bf16 in the checkpoint)"
        del src
    else:
        g = torch.Generator(device="cpu").manual_seed(1)
        W = torch.empty(V, DIM, dtype=torch.bfloat16, device="cuda")
        for i in range(0, V, 8192):
            n = min(8192, V - i)
            W[i:i + n] = (torch.randn(n, DIM, generator=g, device="cpu") * 0.02).to(torch.bfloat16).cuda()
        label = "synthetic bf16 head (set DSV41_MODEL_DIR for the real one)"
    _W["head"] = (W, label)
    return _W["head"]


def _x(R, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(R, DIM, generator=g, device="cpu") * 2).to(torch.bfloat16).cuda()


def _run(SK, R2, x, w, cfg=None, out=None):
    out = out if out is not None else torch.empty(x.size(0), w.size(0), dtype=torch.float32, device="cuda")
    SK.fixed_dot(x, w, out, kb=R2.fixed_kb("head", x.size(1)), site="head", cfg=cfg)
    return out


def _eq(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ------------------------------------------------------------------------------------------------ (a) + (b)
def test_bf16_head_bit_equal_to_fp32_stored_and_row_invariant():
    _need()
    R2, SK = _mods()
    Wb, label = _head_weights()
    Wf = Wb.float()                                         # the loader's widening: the fp32-stored head
    lines = []
    for R in (1, 2, 5, 6, 8):
        x = _x(R, 100 + R)
        ref = _run(SK, R2, x, Wf)                           # fp32-stored, default config: the T9b-5 kernel as shipped
        got = _run(SK, R2, x, Wb)                           # bf16-stored, the "head_bf16" default table
        assert _eq(got, ref), f"R={R}: bf16-stored logits != fp32-stored logits (default configs)"
        nbad = 0
        for cfg in SWEEP_BF16[1:] + SWEEP_FP32[1:]:
            for W_, tag in ((Wb, "bf16"), (Wf, "fp32")):
                g2 = _run(SK, R2, x, W_, cfg)
                if not _eq(g2, ref):
                    nbad += 1
                    print(f"  R={R} {tag}-stored cfg {cfg}: DIFFERS from the fp32-stored default", flush=True)
        assert nbad == 0, f"R={R}: {nbad} launch configs change the bits"
        lines.append(f"  R={R}: bit-equal over all {ref.numel()} logits, {len(SWEEP_BF16) - 1 + len(SWEEP_FP32) - 1} "
                     f"configs x 2 storages equal")
    x6 = _x(6, 7)
    base = _run(SK, R2, x6, Wb)
    one = torch.cat([_run(SK, R2, x6[i:i + 1], Wb) for i in range(6)], 0)
    assert _eq(base, one), "bf16-stored: R=6 differs from six R=1 calls"
    ref_t = R2.fixed_dot_torch(x6, Wb[:4096], R2.fixed_kb("head", DIM))
    assert _eq(base[:, :4096], ref_t), "bf16 kernel != fixed_dot_torch (first 4096 vocab rows)"
    lines.append("  bf16-stored R=6 == six R=1 calls (6/6 rows bit-equal); == fixed_dot_torch on 4096 vocab rows")
    # (b) replay
    again = _run(SK, R2, x6, Wb)
    gr = torch.cuda.CUDAGraph()
    gout = torch.empty_like(base)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(SK, R2, x6, Wb, out=gout)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(gr):
        _run(SK, R2, x6, Wb, out=gout)
    reps = []
    for _ in range(2):
        gout.zero_()
        gr.replay()
        torch.cuda.synchronize()
        reps.append(gout.clone())
    assert _eq(again, base) and all(_eq(r, base) for r in reps), "replay / relaunch / graph differ"
    lines.append("  replay: eager x2, graph replay x2 -> bit-equal")
    print(f"M2-T17 (a)+(b) [{label}]:\n" + "\n".join(lines))
    del Wf


# ------------------------------------------------------------------------------------------------ (c) + (d)
def test_hbm_bytes_and_head_ms_rows_1_and_6():
    _need()
    R2, SK = _mods()
    from runtime.gemv_mxfp8 import time_graph

    Wb, label = _head_weights()
    nb16 = Wb.numel() * Wb.element_size()
    Wf = Wb.float()
    nb32 = Wf.numel() * Wf.element_size()
    print(f"M2-T17 (c) HBM: head weight fp32-stored {nb32 / 2**30:.3f} GiB ({nb32 / 1e9:.3f} GB), bf16-stored "
          f"{nb16 / 2**30:.3f} GiB ({nb16 / 1e9:.3f} GB): saves {(nb32 - nb16) / 1e9:.3f} GB ({label})")
    assert nb32 - nb16 == nb16
    for R in (1, 6):
        x = _x(R, 40 + R)
        out = torch.empty(R, Wb.size(0), dtype=torch.float32, device="cuda")
        reps = 20
        rows = {}
        for tag, W_, sweep in (("fp32-stored", Wf, SWEEP_FP32), ("bf16-stored", Wb, SWEEP_BF16)):
            nb = W_.numel() * W_.element_size()
            best = None
            for cfg in sweep:
                f = lambda c=cfg, w_=W_: _run(SK, R2, x, w_, c, out=out)  # noqa: E731
                f()
                us = time_graph([f] * reps, reps=3)
                cfg_s = ("default " + str(SK.t9b5_config("head", R, W_.dtype))) if cfg is None else str(cfg)
                print(f"  head R={R} {tag:11s} {cfg_s:30s} {us:9.2f} us  {nb / us / 1e6:6.2f} TB/s", flush=True)
                if cfg is None:
                    rows[tag + " default"] = us
                if best is None or us < best[0]:
                    best = (us, cfg_s)
            rows[tag + " best"] = best
        b = rows["bf16-stored best"]
        met = "MET" if b[0] <= HEAD_TARGET_MS * 1e3 else "MISSED"
        print(f"M2-T17 (d) R={R}: fp32-stored default {rows['fp32-stored default']:.1f} us | bf16-stored default "
              f"{rows['bf16-stored default']:.1f} us | bf16-stored best {b[0]:.1f} us {b[1]} | target <= "
              f"{HEAD_TARGET_MS * 1e3:.0f} us: {met}")
    del Wf


# ------------------------------------------------------------------------------------------------ (e)
class _Env:
    """$DSV41_HEAD_DTYPE = v (None: unset) for the duration of a with-block."""

    def __init__(self, v):
        self.v = v

    def __enter__(self):
        self.old = os.environ.get("DSV41_HEAD_DTYPE")
        if self.v is None:
            os.environ.pop("DSV41_HEAD_DTYPE", None)
        else:
            os.environ["DSV41_HEAD_DTYPE"] = self.v

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("DSV41_HEAD_DTYPE", None)
        else:
            os.environ["DSV41_HEAD_DTYPE"] = self.old


def _need_tilelang():
    import test_draft_gpu as DG

    DG._need_tilelang()
    return DG


def test_loader_and_markov_head_on_mini_cuda():
    DG = _need_tilelang()
    R2, SK = _mods()
    gx = torch.Generator(device="cpu").manual_seed(3)
    ge = torch.Generator(device="cpu").manual_seed(5)
    with _Env(None), DG._model("mini3") as m32:
        w32 = m32.head.weight.detach().clone()
        mk32 = m32.mtp[-1].markov_head.head.weight.detach().clone()
        assert w32.dtype == torch.float32 and mk32.dtype == torch.float32
        x = (torch.randn(1, 6, w32.size(1), generator=gx, device="cpu") * 2).to(torch.bfloat16).cuda()
        xe = torch.randn(3, mk32.size(1), generator=ge, device="cpu").to(torch.bfloat16).cuda()
        want = R2.head_fixed_w(w32, x, True)
        wantk = R2.head_fixed_w(mk32, xe, True)
    with _Env("bf16"), DG._model("mini3") as m16:
        assert m16.head.weight.dtype == torch.bfloat16 and torch.equal(m16.head.weight.float(), w32)
        mk = m16.mtp[-1].markov_head.head.weight
        assert mk.dtype == torch.bfloat16 and torch.equal(mk.float(), mk32)
        assert m16.runtime_load.head_bf16 == ["head.weight", f"mtp.{len(m16.mtp) - 1}.markov_head.head.weight"]
        assert _eq(m16.head(x, full_logits=True), want), "bf16-stored head module != fp32-stored fixed-order logits"
        assert _eq(m16.mtp[-1].markov_head.head(xe, full_logits=True), wantk)
        print(f"M2-T17 (e) loader mini3 on cuda: head {tuple(m16.head.weight.shape)} bf16 == fp32-stored fixed "
              f"logits; Markov head bf16 == fp32-stored; head_HBM {m16.runtime_load.bytes['head_HBM']} B")


def test_draft_gates_with_bf16_heads_mini():
    DG = _need_tilelang()
    with _Env("bf16"):
        DG._gate_draft("mini3", 17)
        DG._gate_draft("mini3", 17, impls={"fp8_impl": "gemv", "glue_impl": "fused"})
    print("M2-T17 (e) draft V2 + P1 gates with bf16 heads: DraftStep == reference forward_spec (torch.equal)")


def test_draft_gate_real_with_bf16_heads():
    DG = _need_tilelang()
    if not os.environ.get("DSV41_MODEL_DIR"):
        raise unittest.SkipTest("DSV41_MODEL_DIR not set")
    with _Env("bf16"):
        DG.test_draft_equals_reference_real()
    print("M2-T17 (e) real checkpoint: draft V2 gate with bf16 heads passed")


TESTS = [test_bf16_head_bit_equal_to_fp32_stored_and_row_invariant, test_hbm_bytes_and_head_ms_rows_1_and_6,
         test_loader_and_markov_head_on_mini_cuda, test_draft_gates_with_bf16_heads_mini,
         test_draft_gate_real_with_bf16_heads]

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
