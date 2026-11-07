"""GPU tests of T9b-8: the ratio-2 attention compressor's wkv / wgate in the T9b-5 fixed order under `--fp32-impl fixed`
(runtime/static_kernels.py t9b8_comp_fp32, site "comp"; the call sites: runtime/static_step.py _compressor_one /
_compressor_rows / _compressor_proj, runtime/dspark_loop.py VerifyStep; the order: runtime/fused_glue_r2.py T9b-5 and
T9b-8 sections). Skipped without CUDA (Mac); on the box a missing triton FAILS. Same docker line as
tests/gpu/test_fp32_fixed_gpu.py with this file (+ -e DSV41_MODEL_DIR=/models/<the checkpoint> to read the real layer-2
/ 8 / 14 compressor weights):

  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_compressor_fixed_gpu.py

  (a) THE gate at the real shapes ([R, 5120] x [512, 5120]^T, fp32 weight = bf16 checkpoint values widened, x =
      x.float() of bf16): ours at R = 2..8 vs ours at R = 1 on the same rows, bit-equal per row, every launch config
      of the sweep; the kernel == fused_glue_r2.fixed_dot_torch run on the device; replay (eager x2, fresh output,
      CUDA graph replayed x2) bit-equal; cuBLAS F.linear on the SAME inputs printed next to it (today's site: it is
      the one that fails);
  (b) fp64-oracle distances on 1,024 rows, ours vs cuBLAS (gate: ours < 256 u * sum|x w|);
  (c) us per launch at R = 1 and R = 6, ours (default + sweep) vs cuBLAS, graph-timed over 16 distinct weights (168 MB,
      more than L2: every launch reads its weight from HBM, as in the step) -> the per-step cost of the 6 launches
      (3 ratio-2 layers x wkv + wgate) and the delta vs cuBLAS;
  (d) the mini checkpoint (needs tilelang, like tests/gpu/test_row_divergence_gpu.py): under `cublas` the 1-row
      step's logits are byte-identical to the pre-T9b-8 code path (the old `comp(x, start_pos)` line); under `fixed`
      the 1-row and the 6-row step launch the fixed kernel (counted), the 6-row step's rows equal the 1-row step's
      logits at pooled and non-pooled positions, and the 1-row fixed-vs-cublas logit distance is printed (the
      switch moves the 1-row step by the sum order only, the T9b-5 precedent).
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

DIM, HD = 5120, 512
N_SITES = 6                     # 3 ratio-2 kv-source layers (2, 8, 14) x (wkv, wgate) per step
SWEEP = [None, (1, 1, 8, 4, 1), (1, 2, 16, 4, 1), (1, 1, 40, 4, 1), (8, 1, 8, 4, 1), (8, 2, 4, 4, 1), (8, 4, 8, 4, 1),
         (8, 2, 8, 8, 1), (8, 4, 4, 8, 2), (8, 8, 2, 8, 1)]
_W = {}


def _need():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")


def _mods():
    from runtime import fused_glue_r2 as R2
    from runtime import static_kernels as SK

    assert SK.HAVE_TRITON
    return R2, SK


def _weights():
    """[(name, W fp32 [512, 5120] on cuda)]: the real ratio-2 compressor weights (bf16 / fp32 in the checkpoint,
    widened to fp32 as runtime/loader.py does) when $DSV41_MODEL_DIR is set, else two synthetic bf16-valued ones."""
    if "w" in _W:
        return _W["w"]
    md = os.environ.get("DSV41_MODEL_DIR")
    out = []
    if md:
        from runtime.ckpt import ShardIndex

        idx = ShardIndex.scan(md)
        for L in (2, 8, 14):
            for nm in ("wkv", "wgate"):
                name = f"layers.{L}.attn.compressor.{nm}.weight"
                src = idx.read_tensor(name)
                out.append((f"{name} ({src.dtype} in the checkpoint)", src.cuda().float()))
    else:
        g = torch.Generator(device="cpu").manual_seed(1)
        for nm in ("wkv", "wgate"):
            w = (torch.randn(HD, DIM, generator=g, device="cpu") * 0.02).to(torch.bfloat16).float().cuda()
            out.append((f"synthetic bf16-valued {nm} (set DSV41_MODEL_DIR for the real ones)", w))
    _W["w"] = out
    return out


def _rows(R, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(R, DIM, generator=g, device="cpu") * 2).to(torch.bfloat16).float().cuda()


def _ours(SK, R2, x, w, cfg=None, out=None):
    out = out if out is not None else torch.empty(x.size(0), w.size(0), dtype=torch.float32, device="cuda")
    SK.fixed_dot(x, w, out, kb=R2.fixed_kb("comp", x.size(1)), site="comp", cfg=cfg)
    return out


def _bits_eq(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ------------------------------------------------------------------------------------------------ (a)
def test_comp_row_invariance_gate_and_config_sweep():
    _need()
    R2, SK = _mods()
    from runtime.row_census import compare

    x8 = _rows(8, 10)
    for name, w in _weights():
        one = torch.cat([_ours(SK, R2, x8[i:i + 1], w) for i in range(8)], 0)
        for R in range(2, 9):
            assert _bits_eq(_ours(SK, R2, x8[:R].contiguous(), w), one[:R]), f"{name}: ours at R={R} != R=1 rows"
        for cfg in SWEEP[1:]:
            assert _bits_eq(_ours(SK, R2, x8[:6].contiguous(), w, cfg), one[:6]), f"{name}: config {cfg} moved bits"
            assert _bits_eq(_ours(SK, R2, x8[:1].contiguous(), w, cfg), one[:1]), f"{name}: config {cfg} at R=1"
        ref = R2.fixed_dot_torch(x8[:6], w, R2.fixed_kb("comp", DIM))
        assert _bits_eq(one[:6], ref), f"{name}: kernel != fixed_dot_torch"
        # replay: eager again, a fresh output, a CUDA graph replayed twice
        x6 = x8[:6].contiguous()
        base = _ours(SK, R2, x6, w)
        again = _ours(SK, R2, x6, w)
        gout = torch.empty_like(base)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ours(SK, R2, x6, w, out=gout)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            _ours(SK, R2, x6, w, out=gout)
        reps = []
        for _ in range(2):
            gout.zero_()
            gr.replay()
            torch.cuda.synchronize()
            reps.append(gout.clone())
        assert _bits_eq(again, base) and all(_bits_eq(r, base) for r in reps), f"{name}: replay differs"
        c = compare(F.linear(x6, w), [F.linear(x6[i:i + 1], w) for i in range(6)])
        print(f"  comp ours R=2..8 vs R=1: bit-equal, {len(SWEEP) - 1} configs equal, == fixed_dot_torch, replay "
              f"equal | cuBLAS same inputs: equal {c['equal']} ({c['mismatch_pct']:.2f} % of elements, max "
              f"{c['max_ulp']} ulp) | {name}")


# ------------------------------------------------------------------------------------------------ (b)
def test_comp_fp64_oracle_distance_1024_rows():
    _need()
    R2, SK = _mods()
    u = 2.0 ** -24
    name, w = _weights()[0]
    wd = w.double()
    wa = wd.abs()
    x = _rows(1024, 30)
    st = {k: {"be_max": 0.0, "be_sum": 0.0, "n": 0} for k in ("ours", "cublas")}
    eq = 0
    for i in range(0, 1024, 128):
        xs = x[i:i + 128]
        o, cb = _ours(SK, R2, xs, w), F.linear(xs, w)
        orc = xs.double() @ wd.t()
        mag = xs.double().abs() @ wa.t()
        for k, v in (("ours", o), ("cublas", cb)):
            be = (v.double() - orc).abs() / (u * mag.clamp_min(1e-300))
            st[k]["be_max"] = max(st[k]["be_max"], float(be.max()))
            st[k]["be_sum"] += float(be.sum())
            st[k]["n"] += be.numel()
        eq += int((o.view(torch.int32) == cb.view(torch.int32)).sum())
    n = st["ours"]["n"]
    print(f"  comp fp64 oracle, 1024 rows ({name}): u*sum|xw| max / mean: ours {st['ours']['be_max']:.3f} / "
          f"{st['ours']['be_sum'] / n:.4f}, cuBLAS {st['cublas']['be_max']:.3f} / {st['cublas']['be_sum'] / n:.4f}; "
          f"ours == cuBLAS bitwise on {eq / n * 100:.2f} % of elements")
    assert st["ours"]["be_max"] < 256, st["ours"]


# ------------------------------------------------------------------------------------------------ (c)
def test_comp_timing_rows_1_and_6():
    _need()
    R2, SK = _mods()
    from runtime.gemv_mxfp8 import time_graph

    w0 = _weights()[0][1]
    ws = [w0.clone() for _ in range(16)]          # 168 MB: more than L2, every launch reads HBM (as in the step)
    nb = w0.numel() * 4
    res = {}
    for R in (1, 6):
        x = _rows(R, 40 + R)
        outs = [torch.empty(R, HD, dtype=torch.float32, device="cuda") for _ in ws]
        best = None
        for cfg in SWEEP:
            fns = [lambda w=w, o=o, c=cfg: _ours(SK, R2, x, w, c, out=o) for w, o in zip(ws, outs)]
            for f in fns:
                f()
            us = time_graph(fns * 4, reps=5)
            cfg_s = "default " + str(SK.t9b5_config("comp", R)) if cfg is None else str(cfg)
            print(f"  comp R={R} ours {cfg_s:30s} {us:8.2f} us/launch  {nb / us / 1e6:6.2f} TB/s")
            if cfg is None:
                res[(R, "default")] = us
            if best is None or us < best[0]:
                best = (us, cfg_s)
        fcs = [lambda w=w: F.linear(x, w) for w in ws]
        for f in fcs:
            f()
        usc = time_graph(fcs * 4, reps=5)
        res[(R, "cublas")] = usc
        print(f"  comp R={R} cuBLAS {'':29s} {usc:8.2f} us/launch  {nb / usc / 1e6:6.2f} TB/s | best ours "
              f"{best[0]:.2f} us ({best[1]})")
        print(f"  comp R={R} per step ({N_SITES} launches): ours default {N_SITES * res[(R, 'default')]:.1f} us, "
              f"best {N_SITES * best[0]:.1f} us, cuBLAS {N_SITES * usc:.1f} us -> delta (default - cuBLAS) "
              f"{N_SITES * (res[(R, 'default')] - usc):+.1f} us/step")


# ------------------------------------------------------------------------------------------------ (d)
def _need_mini():
    _need()
    from runtime.refmodel import tilelang_available

    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _env(**kv):
    import importlib.util

    spec = importlib.util.spec_from_file_location("row_divergence_probe_t9b8g",
                                                  os.path.join(REPO, "bench", "row_divergence_probe.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    return P


def test_mini_1row_cublas_unchanged_and_rows_fixed():
    """mini4 (layer 1: ratio 2): (1) `cublas`: the 1-row step's logits == the pre-T9b-8 code path (the old
    `comp(x, start_pos)` line), bit for bit, and no fixed launch; (2) `fixed`, at layer 1's real (mini) weights: the
    6-row form (one projection of 6 rows, pool_ratio2_rows, the state commit = StaticStep._compressor_rows) == six 1-row
    decode calls (fused_glue_r2.compressor_decode_fixed = the 1-row step's form): every pooled latent and the state,
    both P0 parities; (3) `fixed` 1-row step: launches counted, the logit distance to `cublas` printed (the sum order
    is the only change). The whole-step 6-row vs 1-row gate under `fixed` (every tap, cache.ckv at the pooled rows,
    the logits) is tests/gpu/test_row_divergence_gpu.py's mini4 PLAIN run."""
    _need_mini()
    import copy

    from runtime import fused_glue_r2 as R2
    from runtime.refmodel import reference_forward
    from runtime.static_step import StaticStep, pool_ratio2_rows, ratio2_offset, ratio2_state_rows

    P = _env()
    with P.mini_model("mini4", "cuda") as (model, pool):
        g = torch.Generator(device="cpu").manual_seed(41)
        ids = torch.randint(2, 2000, (18,), generator=g, device="cpu").tolist()
        toks = torch.randint(2, 2000, (6,), generator=g, device="cpu").tolist()

        def run1():
            st = StaticStep(model, pool, max_pos=255, rows=1)
            with torch.inference_mode():
                reference_forward(model, torch.tensor([ids], dtype=torch.long, device="cpu").cuda(), 0)
                out = torch.stack([st.step(toks[i], len(ids) + i, graph=False)[0].float().clone() for i in range(6)])
            torch.cuda.synchronize()
            st.detach()
            return out

        n = {"comp": 0}
        real = R2.launch_fixed_dot

        def counted(x, w, out, *, site, scale=None):
            n["comp"] += site == "comp"
            return real(x, w, out, site=site, scale=scale)

        R2.launch_fixed_dot = counted
        old_one = StaticStep._compressor_one
        try:
            with P.env(DSV41_FP32_IMPL="cublas"):
                new_cb = run1()
                StaticStep._compressor_one = lambda self, comp, x, start_pos: comp(x, start_pos)   # the old line
                old_cb = run1()
                StaticStep._compressor_one = old_one
            assert n["comp"] == 0 and _bits_eq(new_cb, old_cb), "cublas: the 1-row logits moved"
            comp0 = next(l.attn.compressor for l in model.layers
                         if l.attn.compressor is not None and l.attn.compressor.compress_ratio == 2)
            with P.env(DSV41_FP32_IMPL="fixed"), torch.inference_mode():
                D = comp0.wkv.weight.size(1)
                for L in (5, 6):                                  # first P0 parity
                    one, six = copy.deepcopy(comp0), copy.deepcopy(comp0)
                    one.kv_state.zero_(), six.kv_state.zero_()
                    one.score_state.fill_(-float("inf")), six.score_state.fill_(-float("inf"))
                    gx = torch.Generator(device="cpu").manual_seed(L)
                    x = torch.randn(1, L + 24, D, generator=gx, device="cpu").to(torch.bfloat16).cuda()
                    one(x[:, :L], 0)
                    six(x[:, :L], 0)
                    for p0 in range(L, L + 24, 6):
                        xs = x[:, p0:p0 + 6]
                        kv6, sc6 = R2.compressor_proj_fixed(six, xs.float())
                        Pd = torch.tensor([p0], device="cpu").cuda()
                        pooled = pool_ratio2_rows(six.kv_state[:1, :1], six.score_state[:1, :1], kv6, sc6,
                                                  ratio2_offset(Pd) + torch.arange(6, device="cpu").cuda())
                        sidx = ratio2_state_rows(Pd, 6)
                        six.kv_state[:1].copy_(kv6.index_select(1, sidx))
                        six.score_state[:1].copy_(sc6.index_select(1, sidx))
                        got = six.norm(pooled.to(xs.dtype))
                        want = [y for y in (R2.compressor_decode_fixed(one, x[:, p0 + i:p0 + i + 1], p0 + i)
                                            for i in range(6)) if y is not None]
                        assert len(want) == 3 and torch.equal(got, torch.cat(want, 1)), (L, p0, "pooled latents")
                        assert torch.equal(six.kv_state, one.kv_state) and \
                            torch.equal(six.score_state, one.score_state), (L, p0, "state")
            with P.env(DSV41_FP32_IMPL="fixed"):
                n["comp"] = 0
                fx = run1()
            n2 = sum(1 for l in model.layers if l.attn.is_kv_source and l.attn.compressor is not None
                     and l.attn.compressor.compress_ratio > 1)
            assert n2 >= 1 and n["comp"] == 6 * 2 * n2, (n, n2)   # 6 steps x (wkv, wgate) per ratio-2 layer
            d = (fx - new_cb).abs().max().item()
            print(f"  mini4: cublas 1-row == pre-T9b-8 path (bit-equal, 0 fixed launches); fixed: 6-row compressor "
                  f"form == six 1-row calls (pooled latents + state, both parities, 4 steps each); 1-row fixed vs "
                  f"cublas logits max |d| {d:.3g}, bit-equal {_bits_eq(fx, new_cb)}, argmax equal "
                  f"{torch.equal(fx.argmax(-1), new_cb.argmax(-1))}")
        finally:
            R2.launch_fixed_dot = real
            StaticStep._compressor_one = old_one


TESTS = [test_comp_row_invariance_gate_and_config_sweep, test_comp_fp64_oracle_distance_1024_rows,
         test_comp_timing_rows_1_and_6, test_mini_1row_cublas_unchanged_and_rows_fixed]

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
