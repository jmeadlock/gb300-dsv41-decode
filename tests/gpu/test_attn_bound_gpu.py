"""GPU tests of T9b-3a (`--attn-impl bounded`, runtime/kernels/sparse_attn_bounded.py; the 16-B vector staging gather,
runtime/static_kernels.py). Skipped without CUDA/tilelang (Mac); on the box a missing triton FAILS.

  * bounded `sparse_attn` vs the reference kernel.py `sparse_attn`, `torch.equal`, at the real shapes (64 heads x 512,
    softmax_scale 512^-0.5, kv = 128 window + 2048 / 4096 compressed rows) at index widths 128 / 278 / 428 / 640 with
    -1 tails (and leading -1 window slots), against the reference at the full padded 640 width AND at the narrow
    width the eager reference passes; one row and six rows with different counts in one launch. Every TileLang loop
    spelling (extent / serial / guard) is tried and reported; at least one must compile and match. Informative: µs
    per call, reference at 640 vs bounded at 2 / 5 / 7 / 10 blocks;
  * the vector gather (gather_rows / gather_tables, scales as bytes and as dg's fp32 decode, ids_out) == the T8/T9b-2
    byte kernels, `torch.equal`, on real-size rows (5,898,240 / 368,640 B) in HBM and in registered host memory (UVA),
    at the default and every swept block/warps config (incl. non-EVEN scale blocks); and on mini4's real pointer
    tables through the StaticStep default path;
  * mini checkpoints (mini, mini4, mini4w = mini4 with index_topk 256 so 3 of 6 blocks are skipped, mini4r = mini4w
    with a compressed layer that reuses another layer's indices and count): bounded self-check on;
    bounded static-eager == ref static-eager fp32 logits over prefill + 10 decode steps (both parities); no host
    sync; graph == static-eager, replay #2, fills-on recapture, relaunch; mini4w also with dg-masked / gemv / fused
    glue. Informative: graph step-cell p50 ref vs bounded.

Run on the box (orchestrator), from the repo checkout, inside the library image (same line as
tests/gpu/test_fused_glue_gpu.py, plus IPC_LOCK for the registered host rows):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    --cap-add IPC_LOCK -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_attn_bound_gpu.py

(`python3 tests/gpu/test_attn_bound_gpu.py` runs every test without pytest; no checkpoint mount needed.)
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_static_mini_gpu import (MAX_POS, N_STEPS, _assert_equal_runs, _first_light, _ids, _mini4_config,  # noqa: E402
                                  _model, _need_tilelang, _run)

HEADS, HD, WIN, K = 64, 512, 128, 512
WIDTHS = (128, 278, 428, 640)


def _need_triton():
    _need_tilelang()
    import triton  # noqa: F401  (FAIL on the box if missing)


def _kernel_mod():
    from runtime.refmodel import import_reference

    import_reference()
    import kernel  # the reference kernel.py (reference/deepseek-inference), as model.py imports it

    return kernel


def _gen(model, tokdir, **kw):
    from runtime.generate import Generator, set_determinism

    set_determinism(0)
    return Generator(model, tokdir, max_pos=MAX_POS, **kw)


VARIANT_CFG = {"mini4w": {"index_topk": 256}, "mini4r": {"index_topk": 256, "index_source_layers": [1, 2]}}


@contextlib.contextmanager
def _model_w(kind):
    """mini / mini4 as test_static_mini_gpu._model; "mini4w" = mini4 with index_topk 256 (at MAX_POS 255 the static
    widths are 128 + 128 / 128 + 256 = 4 / 6 blocks, the fixture positions need 3); "mini4r" = mini4w with index
    sources [1, 2] only (layer 3 reuses layer 2's indices and its published block count)."""
    if kind not in VARIANT_CFG:
        with _model(kind) as mt:
            yield mt
        return
    import test_static_mini_gpu as S

    cfg, hot = _mini4_config()
    cfg.update(VARIANT_CFG[kind])
    orig = S._mini4_config
    S._mini4_config = lambda: (dict(cfg), hot)
    try:
        with _model("mini4") as mt:
            yield mt
    finally:
        S._mini4_config = orig


def _bits(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


# ------------------------------------------------------------------------------------------------ the kernel
def test_bounded_kernel_vs_reference_real_shapes():
    _need_tilelang()
    from runtime.kernels import sparse_attn_bounded as SAB

    kmod = _kernel_mod()
    torch.set_default_device("cpu")
    scale = HD ** -0.5
    passing = []
    for W in (2048, 4096):
        for v in SAB.VARIANTS:
            chk = SAB.self_check(kmod.sparse_attn, "cuda", HEADS, HD, scale, WIN, K, WIN + W, variants=(v,),
                                 select=False)
            print(f"  kv {WIN}+{W}: variant {v}: {chk['results'][v]} ({chk['cases']} cases, widths incl. {WIDTHS})")
            if chk["results"][v] == "equal" and v not in passing and W == 4096:
                passing.append(v)
    assert passing, "no TileLang spelling of the bounded loop compiled and matched the reference"
    # six rows, the four widths + two leading -1 windows, one launch, vs each row alone vs the reference
    g = torch.Generator(device="cpu").manual_seed(9)
    kv = torch.randn(1, WIN + 4096, HD, generator=g).to(torch.bfloat16).cuda()
    sink = torch.randn(HEADS, generator=g).float().cuda()
    cases = SAB.check_cases(WIN, K, WIN + 4096, seed=3)
    pick = [next(c for c in cases if c["pos"] == 200 and WIN + c["c"] == w) for w in WIDTHS]
    pick += [next(c for c in cases if c["pos"] == 5 and c["c"] == 0), next(c for c in cases if c["pos"] == 63)]
    idx = torch.stack([c["idx"] for c in pick]).view(1, 6, -1).cuda()
    nb = torch.tensor([[SAB.blocks_for_width(WIN + c["c"]) for c in pick]], dtype=torch.int32, device="cuda")
    assert torch.equal(nb[0, :4].cpu(), torch.tensor([2, 5, 7, 10], dtype=torch.int32))
    q = (torch.randn(1, 6, HEADS, HD, generator=g) * 0.5).to(torch.bfloat16).cuda()
    want = kmod.sparse_attn(q, kv, sink, idx, scale)
    for v in passing:
        got = SAB.sparse_attn_bounded(q, kv, sink, idx, scale, nb, variant=v)
        assert _bits(got, want), f"{v}: 6-row bounded != reference at 640"
        for r, c in enumerate(pick):
            narrow = kmod.sparse_attn(q[:, r:r + 1].contiguous(), kv, sink,
                                      idx[:, r:r + 1, :WIN + c["c"]].contiguous(), scale)
            assert _bits(got[:, r:r + 1], narrow), f"{v}: row {r} != reference at narrow width {WIN + c['c']}"
    print(f"  six rows (widths {WIDTHS} + two leading -1 windows) in one launch == reference at 640 and at each "
          f"row's narrow width; spellings that pass: {passing}")
    # informative timing: one row, µs per call
    q1 = q[:, :1].contiguous()

    def us(fn, n=200):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1e3 / n

    full = idx[:, :1].contiguous()
    t_ref = us(lambda: kmod.sparse_attn(q1, kv, sink, full, scale))
    rows = [f"reference@640 {t_ref:.1f}"]
    for v in passing:
        for nbk in (2, 5, 7, 10):
            n1 = torch.tensor([[nbk]], dtype=torch.int32, device="cuda")
            rows.append(f"{v}@{nbk}blk {us(lambda: SAB.sparse_attn_bounded(q1, kv, sink, full, scale, n1, v)):.1f}")
    print("  µs/call (1 row, eager loop, incl. wrapper): " + ", ".join(rows))


# ------------------------------------------------------------------------------------------------ the gather
def _bench():
    spec = importlib.util.spec_from_file_location("bench_gather", os.path.join(REPO, "bench", "gather_bench.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vector_gather_equals_byte_gather_real_rows():
    _need_triton()
    gb = _bench()
    tb = gb.Tables(gb.real_shapes(), 8, 6, "cuda")
    try:
        ids = tb.id_sets("hot", 2) + tb.id_sets("cold", 2, seed=2)
        mixed = torch.tensor([0, 3, 7, 8, 11, 13], device="cuda")
        cfgs = [(None, None), (4096, 4), (8192, 8), (16384, 4), (32768, 8)]  # 32 KiB: scale rows not EVEN
        res = tb.verify(ids + [mixed], configs=cfgs)
        bad = {k: v for k, v in res.items() if v != "equal"}
        assert not bad, bad
        print(f"  vector gather == byte gather on real-size rows (hot HBM, cold UVA, mixed): {len(res)} arm/configs")
    finally:
        tb.release()


def test_vector_gather_on_static_step_tables():
    _need_triton()
    from runtime import static_kernels as SK

    with _model("mini4") as (model, tokdir):
        for impl in ("ref", "dg-masked"):
            if impl != "ref":
                try:
                    from runtime.dg_moe import import_deep_gemm

                    import_deep_gemm()
                except Exception as e:  # pragma: no cover
                    print(f"  (dg-masked skipped: {e})")
                    continue
            gen = _gen(model, tokdir, moe_impl=impl, glue_impl="fused")
            st = gen.static_step()
            for li, sm in enumerate(st.moes):
                E = sm.moe.n_routed_experts
                ids = torch.randperm(E, generator=torch.Generator(device="cpu").manual_seed(li),
                                     device="cpu")[: st.topk].sort().values.cuda()
                want = {}
                for key, tab in sm.ptabs.items():
                    SK.gather_rows(tab, ids, st.staging[key], st.row_bytes[key], impl="u8")
                    want[key] = st.staging[key].clone()
                for key, tab in sm.ptabs.items():
                    st.staging[key].zero_()
                    SK.gather_rows(tab, ids, st.staging[key], st.row_bytes[key])        # default path: vec
                    assert torch.equal(st.staging[key].view(torch.uint8), want[key].view(torch.uint8)), (impl, li, key)
                if st.glue is not None and st.glue.on("gather"):
                    for t in st.glue._dsts:
                        t.zero_()
                    st.glue.gather(sm, ids)                                              # default path: vec tables
                    torch.cuda.synchronize()
                    for p in ("w1", "w3", "w2"):
                        assert torch.equal(st.staging[(p, "weight")].view(torch.uint8),
                                           want[(p, "weight")].view(torch.uint8)), (impl, li, p)
            assert SK.describe()["impl"] == "vec" and SK.describe()["vec_error"] is None, SK.describe()
            gen.reset_static()
            del gen
    print("  default-path (vec) gathers == byte gathers on mini4's pointer tables (ref and dg-masked)")


# ------------------------------------------------------------------------------------------------ mini checkpoints
def _bounded_vs_ref(kind, moe="ref", fp8="ref", glue="ref"):
    _need_triton()
    if moe != "ref":
        from runtime.dg_moe import import_deep_gemm

        import_deep_gemm()
    with _model_w(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=moe, fp8_impl=fp8, glue_impl=glue, attn_impl="bounded")
        st = gen.static_step()
        d = st.attn_describe()
        print(f"  {kind} {moe}/{fp8}/{glue}: attn {d.get('variant')} on={d['on']} {(d.get('check') or {}).get('results')}")
        assert d["on"], d
        widths = sorted({WIN + min(a.indexer.index_topk, st.W[a.compress_ratio]) for a in (l.attn for l in model.layers)
                         if a.compress_ratio and a.indexer is not None})
        runs = {}
        for L, seed in ((16, 5), (17, 6)):
            ids = _ids(L, seed)
            runs[L] = (ids, _run(gen, "static", ids))
        ids = runs[16][0]
        gen.set_mode("static")                                      # no host sync in the bounded step
        gen.prefill(ids)
        gen.step(int(gen.next_token_tensor().reshape(-1)[0]), len(ids))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for p in (len(ids) + 1, len(ids) + 2):
                gen.step(gen.next_token_tensor(), p)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        g1 = _run(gen, "graph", ids)
        _assert_equal_runs(g1, runs[16][1], f"{kind}: bounded graph vs bounded static-eager")
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: bounded graph replay#2")
        gen.capture(fills=True)
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: bounded fills-on recapture")
        gen.reset_static()
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: bounded relaunch")
        gen.set_attn_impl("ref")
        for L, (ids_, bounded) in runs.items():
            _assert_equal_runs(bounded, _run(gen, "static", ids_), f"{kind} L{L}: bounded vs ref attn static-eager")
        assert gen.static.attn_describe() == {"impl": "ref", "on": False}
        print(f"  {kind} {moe}/{fp8}/{glue}: bounded == ref attn static-eager (prefill + {N_STEPS} steps, both "
              f"parities; static widths {widths}); graph == static, replay x2, fills-on, relaunch; no host sync")
        gen.reset_static()
        del gen


def test_bounded_vs_ref_mini():
    _bounded_vs_ref("mini")


def test_bounded_vs_ref_mini4():
    _bounded_vs_ref("mini4")


def test_bounded_vs_ref_mini4w():
    _bounded_vs_ref("mini4w")


def test_bounded_vs_ref_mini4r():
    _bounded_vs_ref("mini4r")


def test_bounded_vs_ref_mini4w_dg_gemv_fused():
    _bounded_vs_ref("mini4w", "dg-masked", "gemv", "fused")


def test_bounded_step_cell_informative():
    _need_triton()
    fl = _first_light()
    with _model_w("mini4w") as (model, tokdir):
        p50 = {}
        for attn in ("ref", "bounded"):
            gen = _gen(model, tokdir, attn_impl=attn, mode="graph")
            sc = fl.step_cell(gen, _ids(16, 9), 40, 8)
            p50[attn] = sc["ms_per_step"]["p50"]
            gen.reset_static()
            del gen
        print(f"  mini4w graph step cell p50: attn ref {p50['ref']:.3f} ms, bounded {p50['bounded']:.3f} ms")


TESTS = [test_bounded_kernel_vs_reference_real_shapes, test_vector_gather_equals_byte_gather_real_rows,
         test_vector_gather_on_static_step_tables, test_bounded_vs_ref_mini, test_bounded_vs_ref_mini4,
         test_bounded_vs_ref_mini4w, test_bounded_vs_ref_mini4r, test_bounded_vs_ref_mini4w_dg_gemv_fused, test_bounded_step_cell_informative]

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
