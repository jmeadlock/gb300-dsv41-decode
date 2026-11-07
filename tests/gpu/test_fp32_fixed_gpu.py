"""GPU tests of T9b-5 (docs/HANDOFF-T9B5-FP32-FIXED.md): the fixed-order fp32 kernels for the mHC mixes and the head
(runtime/static_kernels.py t9b5_mix_fp32 / t9b5_head_fp32; the order: runtime/fused_glue_r2.py T9b-5 section).
Skipped without CUDA (Mac); on the box a missing triton FAILS (it is what is under test). Same docker line as
tests/gpu/test_t9b4_gpu.py, with this file (and -e DSV41_MODEL_DIR=/models/<the checkpoint> for the real head):

  ... -e DSV41_MODEL_DIR=/models/DeepSeek-V4.1-... --entrypoint python3 vllm/vllm-openai:v0.31.0 \\
      tests/gpu/test_fp32_fixed_gpu.py

  (a) THE gate: ours at R = 6 vs ours at R = 1 on the same rows, bit-equal per row, at the real shapes (mixes
      [R, 20480] x [24, 20480]^T * rsqrt; head [R, 5120] x [129280, 5120]^T) and every launch config of the sweep
      (RB / BN / U / warps / stages change speed only); cuBLAS F.linear on the SAME inputs printed next to it (it
      fails today: the evidence); the kernel == fused_glue_r2.fixed_dot_torch (the order's torch definition, run on
      the device) on the mixes and on 4,096 head rows;
  (b) replay: eager twice, a fresh launch, and a CUDA graph replayed twice -> bit-equal;
  (c) fp64-oracle distances on 1,024 random rows: ours vs cuBLAS (max / mean |err| in fp32 ulps of the oracle,
      argmax agreement, fraction bit-equal to cuBLAS), on the real head when $DSV41_MODEL_DIR is set (else a
      synthetic bf16-valued head, said so);
  (d) ms at R = 1 and R = 6, ours vs cuBLAS, graph-timed (head: 2.65 GB fp32 read per call, TB/s printed; the target
      line is the handoff's <= 0.55 ms at both R, informative here -- the box gate is first_light / headline).
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
import torch.nn.functional as F  # noqa: E402

DIM, HC, V, MIX = 5120, 4, 129280, 24
HEAD_TARGET_MS = 0.55
SWEEP = {"head": [None, (1, 64, 2, 4, 1), (1, 32, 8, 4, 1), (2, 32, 4, 4, 1), (8, 32, 2, 8, 1), (8, 8, 4, 4, 1),
                  (8, 16, 4, 8, 2), (8, 16, 2, 4, 3)],
         "mix": [None, (1, 1, 4, 4, 1), (1, 2, 8, 4, 1), (1, 1, 20, 8, 1), (1, 1, 40, 16, 1), (2, 1, 8, 4, 2),
                 (8, 4, 2, 4, 1)]}
_W = {}


def _need():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")


def _mods():
    from runtime import fused_glue_r2 as R2
    from runtime import static_kernels as SK

    assert SK.HAVE_TRITON
    return R2, SK


def _head_weight():
    """(W fp32 [V, D] on cuda, label): the real head (bf16 in the checkpoint, widened exactly, as the loader does)
    when $DSV41_MODEL_DIR is set, else a synthetic bf16-valued head of the same shape."""
    if "head" in _W:
        return _W["head"]
    md = os.environ.get("DSV41_MODEL_DIR")
    if md:
        from runtime.ckpt import ShardIndex

        idx = ShardIndex.scan(md)
        src = idx.read_tensor("head.weight")
        label = f"real head.weight ({src.dtype} in the checkpoint, widened to fp32 as runtime/loader.py)"
        W = src.cuda().float()
        del src
    else:
        g = torch.Generator(device="cpu").manual_seed(1)
        W = torch.empty(V, DIM, dtype=torch.float32, device="cuda")
        for i in range(0, V, 8192):   # CPU-seeded, bf16-valued, in chunks (the host draw of 0.66 G values is slow)
            n = min(8192, V - i)
            W[i:i + n] = (torch.randn(n, DIM, generator=g, device="cpu") * 0.02).to(torch.bfloat16).float().cuda()
        label = "synthetic bf16-valued head (set DSV41_MODEL_DIR for the real one)"
    _W["head"] = (W, label)
    return _W["head"]


def _mix_weight():
    if "mix" not in _W:
        g = torch.Generator(device="cpu").manual_seed(2)
        _W["mix"] = ((torch.randn(MIX, HC * DIM, generator=g, device="cpu") * 0.01).cuda(), "synthetic fp32 hc_fn")
    return _W["mix"]


def _rows(site, R, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if site == "head":
        x = (torch.randn(R, DIM, generator=g, device="cpu") * 2).to(torch.bfloat16).cuda()
        return x, None
    x = (torch.randn(R, HC * DIM, generator=g, device="cpu") * 3).to(torch.bfloat16).float().cuda()
    sc = (torch.rand(R, generator=g, device="cpu") + 0.25).cuda()
    return x, sc


def _ours(SK, R2, x, w, site, sc=None, cfg=None, out=None):
    out = out if out is not None else torch.empty(x.size(0), w.size(0), dtype=torch.float32, device="cuda")
    SK.fixed_dot(x, w, out, kb=R2.fixed_kb(site, x.size(1)), site=site, scale=sc, cfg=cfg)
    return out


def _cublas(x, w, sc=None):
    o = F.linear(x.float(), w)
    return o * sc.view(-1, 1) if sc is not None else o


def _bits_eq(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def _per_row(fn, x, sc):
    return torch.cat([fn(x[i:i + 1], sc[i:i + 1] if sc is not None else None) for i in range(x.size(0))], 0)


# ------------------------------------------------------------------------------------------------ (a) + (b)
def test_row_invariance_gate_and_config_sweep():
    _need()
    R2, SK = _mods()
    lines = []
    for site in ("mix", "head"):
        w, label = _mix_weight() if site == "mix" else _head_weight()
        x6, sc6 = _rows(site, 6, 10 if site == "mix" else 11)
        one = _per_row(lambda xi, si: _ours(SK, R2, xi, w, site, si), x6, sc6)
        base = _ours(SK, R2, x6, w, site, sc6)
        assert _bits_eq(base, one), f"{site}: ours at R=6 differs from ours at R=1"
        for cfg in SWEEP[site][1:]:
            got = _ours(SK, R2, x6, w, site, sc6, cfg)
            assert _bits_eq(got, base), f"{site}: config {cfg} changes the bits"
            got1 = _ours(SK, R2, x6[:1], w, site, sc6[:1] if sc6 is not None else None, cfg)
            assert _bits_eq(got1, base[:1]), f"{site}: config {cfg} at R=1 changes the bits"
        cb6 = _cublas(x6, w, sc6)
        cb1 = _per_row(lambda xi, si: _cublas(xi, w, si), x6, sc6)
        from runtime.row_census import compare

        c = compare(cb6, [cb1[i:i + 1] for i in range(6)])
        lines.append(f"  {site:4s} ours R=6 vs R=1: bit-equal 6/6 rows, {len(SWEEP[site]) - 1} configs equal | "
                     f"cuBLAS same inputs: equal {c['equal']} ({c['mismatch_pct']:.2f} % of elements, max "
                     f"{c['max_ulp']} ulp) | {label}")
        # the kernel == the order's torch definition, on the device
        if site == "mix":
            ref = R2.fixed_dot_torch(x6, w, R2.fixed_kb(site, x6.size(1)), sc6)
            assert _bits_eq(base, ref), "mix kernel != fixed_dot_torch"
        else:
            ws = w[:4096]
            ref = R2.fixed_dot_torch(x6, ws, R2.fixed_kb(site, DIM))
            assert _bits_eq(base[:, :4096], ref), "head kernel != fixed_dot_torch (first 4096 vocab rows)"
        # (b) replay: eager again, a fresh output, a graph replayed twice
        again = _ours(SK, R2, x6, w, site, sc6)
        out = torch.empty_like(base)
        _ours(SK, R2, x6, w, site, sc6, out=out)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        gout = torch.empty_like(base)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ours(SK, R2, x6, w, site, sc6, out=gout)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(gr):
            _ours(SK, R2, x6, w, site, sc6, out=gout)
        reps = []
        for _ in range(2):
            gout.zero_()
            gr.replay()
            torch.cuda.synchronize()
            reps.append(gout.clone())
        assert _bits_eq(again, base) and _bits_eq(out, base) and all(_bits_eq(r, base) for r in reps), \
            f"{site}: replay / relaunch / graph differ"
        lines.append(f"  {site:4s} replay: eager x2, fresh launch, graph replay x2 -> bit-equal")
    print("\n".join(lines))


# ------------------------------------------------------------------------------------------------ (c)
def _ulp32(t: torch.Tensor) -> torch.Tensor:
    """fp32 ulp at each value of the fp64 tensor t (spacing of its fp32 rounding)."""
    f = t.float().abs()
    return (torch.nextafter(f, torch.full_like(f, float("inf"))) - f).double()


def test_fp64_oracle_distance_1024_rows():
    """Distances to the fp64 oracle in two units: fp32 ulps of the result (on well-conditioned outputs, |oracle| >=
    2^-10 * sum|x w|) and the backward-error unit u * sum_k |x_k w_k| (u = 2^-24; the worst-case bound of our
    order is about (K/KB + log2 KB) of those units, so the gate is < 256)."""
    _need()
    R2, SK = _mods()
    u = 2.0 ** -24
    for site in ("head", "mix"):
        w, label = _head_weight() if site == "head" else _mix_weight()
        wd = w.double()
        wa = wd.abs()
        x, sc = _rows(site, 1024, 30 if site == "head" else 31)
        st = {k: {"ulp_max": 0.0, "ulp_sum": 0.0, "ulp_n": 0, "be_max": 0.0, "be_sum": 0.0, "n": 0}
              for k in ("ours", "cublas")}
        eq_cb = arg_o = arg_c = arg_oc = 0
        for i in range(0, 1024, 128):
            xs = x[i:i + 128]
            ss = sc[i:i + 128] if sc is not None else None
            o = _ours(SK, R2, xs, w, site, ss)
            cb = _cublas(xs, w, ss)
            orc = xs.double() @ wd.t()
            mag = xs.double().abs() @ wa.t()
            if ss is not None:
                orc = orc * ss.double().view(-1, 1)
                mag = mag * ss.double().view(-1, 1)
            well = orc.abs() >= mag * 2.0 ** -10
            ulp = _ulp32(orc)
            for k, v in (("ours", o), ("cublas", cb)):
                err = (v.double() - orc).abs()
                be = err / (u * mag.clamp_min(1e-300))
                eu = (err / ulp)[well]
                s_ = st[k]
                s_["be_max"] = max(s_["be_max"], float(be.max()))
                s_["be_sum"] += float(be.sum())
                s_["n"] += be.numel()
                if eu.numel():
                    s_["ulp_max"] = max(s_["ulp_max"], float(eu.max()))
                    s_["ulp_sum"] += float(eu.sum())
                    s_["ulp_n"] += eu.numel()
            eq_cb += int((o.view(torch.int32) == cb.view(torch.int32)).sum())
            if site == "head":
                ao, ac, ar = o.argmax(-1), cb.argmax(-1), orc.argmax(-1)
                arg_o += int((ao == ar).sum())
                arg_c += int((ac == ar).sum())
                arg_oc += int((ao == ac).sum())
            del orc, mag, ulp
        n = st["ours"]["n"]

        def f(k):
            s_ = st[k]
            return (f"{k} ulp max {s_['ulp_max']:.2f} mean {s_['ulp_sum'] / max(1, s_['ulp_n']):.4f} "
                    f"(well-conditioned {s_['ulp_n'] / n * 100:.1f} %), u*sum|xw| max {s_['be_max']:.3f} mean "
                    f"{s_['be_sum'] / n:.4f}")
        print(f"  {site:4s} fp64 oracle, 1024 rows ({label}):\n    {f('ours')}\n    {f('cublas')}\n"
              f"    ours == cuBLAS bitwise on {eq_cb / n * 100:.2f} % of elements"
              + (f"; argmax == oracle: ours {arg_o}/1024, cuBLAS {arg_c}/1024; ours == cuBLAS {arg_oc}/1024"
                 if site == "head" else ""))
        assert st["ours"]["be_max"] < 256, f"{site}: ours {st['ours']['be_max']} u*sum|xw| from the fp64 oracle"
        del wd, wa


# ------------------------------------------------------------------------------------------------ (d)
def test_timing_rows_1_and_6():
    _need()
    R2, SK = _mods()
    from runtime.gemv_mxfp8 import time_graph

    for site in ("head", "mix"):
        w, label = _head_weight() if site == "head" else _mix_weight()
        nb = w.numel() * w.element_size()
        for R in (1, 6):
            x, sc = _rows(site, R, 40 + R)
            out = torch.empty(R, w.size(0), dtype=torch.float32, device="cuda")
            reps = 20 if site == "head" else 200
            best = None
            for cfg in SWEEP[site]:
                f = lambda c=cfg: _ours(SK, R2, x, w, site, sc, c, out=out)  # noqa: E731
                f()
                us = time_graph([f] * reps, reps=3)
                cfg_s = "default " + str(SK.t9b5_config(site, R)) if cfg is None else str(cfg)
                print(f"  {site:4s} R={R} ours {cfg_s:28s} {us:9.2f} us  {nb / us / 1e6:6.2f} TB/s")
                if best is None or us < best[0]:
                    best = (us, cfg_s)
            fc = lambda: _cublas(x, w, sc)  # noqa: E731
            fc()
            usc = time_graph([fc] * reps, reps=3)
            print(f"  {site:4s} R={R} cuBLAS {'':27s} {usc:9.2f} us  {nb / usc / 1e6:6.2f} TB/s | best ours "
                  f"{best[0]:.2f} us ({best[1]})" + (f" | target <= {HEAD_TARGET_MS * 1e3:.0f} us: "
                                                     f"{'MET' if best[0] <= HEAD_TARGET_MS * 1e3 else 'MISSED'}"
                                                     if site == "head" else ""))


TESTS = [test_row_invariance_gate_and_config_sweep, test_fp64_oracle_distance_1024_rows, test_timing_rows_1_and_6]

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
