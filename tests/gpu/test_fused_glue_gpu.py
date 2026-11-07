"""GPU tests of T9b-2 round 1 (`glue_impl="fused"`, runtime/fused_glue.py). Skipped without CUDA/tilelang (Mac); on
the box a missing triton FAILS (it is what is under test).

  * every fused kernel `torch.equal` vs the unfused ops it replaces, at the real model's shapes (dim 5120, moe_inter
    2304, k 6, hc 4, 64 heads x 512 with rope 64, wq_b 32768 x 1280), with clamp edges and e8m0 scale boundaries;
    act_quant vs the reference tilelang kernel on every finite bf16 value; the RoPE contraction forms and the hc sum
    orders that match torch on this device are printed (at least one must match each, else that cluster stays
    unfused and the test says which);
  * the one-launch gather == 6 x t8_gather_rows (+ convert_scales) on a mini checkpoint's real pointer tables
    (HBM slots and registered-host cold experts);
  * mini checkpoints (mini, mini4), both impl pairs (ref/ref and dg-masked/gemv): FusedGlue's self-check turns every
    applicable cluster on; fused static-eager == ref static-eager fp32 logits over prefill + 10 decode steps (both
    parities), same tokens; no host sync in the fused step; graph replay == static-eager, replay #2, fills-on
    recapture and relaunch equal; informative: graph step-cell p50 fused vs ref.

Run on the box (orchestrator), from the repo checkout, inside the library image with the GB300 by CDI; same line as
tests/gpu/test_dg_moe_gpu.py:

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_fused_glue_gpu.py

(`python3 tests/gpu/test_fused_glue_gpu.py` runs every test without pytest; no checkpoint mount needed.)
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

from test_static_mini_gpu import (N_STEPS, MAX_POS, _assert_equal_runs, _first_light, _ids, _model,  # noqa: E402
                                  _need_tilelang, _run)

FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
DIM, INTER, K_ACT, HC, HEADS, HD, RD, Q_LORA, LIM = 5120, 2304, 6, 4, 64, 512, 64, 1280, 10.0


def _need_glue():
    _need_tilelang()
    import triton  # noqa: F401  (FAIL on the box if missing)
    from runtime import fused_glue as FG

    assert FG.HAVE_TRITON


def _R():
    from runtime.refmodel import import_reference

    return import_reference()


def _eq(a, b):
    v = (lambda t: t.view(torch.uint8)) if a.element_size() == 1 else (
        lambda t: t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32))
    return a.shape == b.shape and torch.equal(v(a.contiguous()), v(b.contiguous()))


def _gen(model, tokdir, **kw):
    from runtime.generate import Generator, set_determinism

    set_determinism(0)
    return Generator(model, tokdir, max_pos=MAX_POS, **kw)


def _act(g, rows, K, lo=-6, hi=7):
    from runtime.fused_glue import _rand_act

    return _rand_act(g, rows, K, lo, hi).cuda()


def _tables(R, lim=LIM):
    from runtime import fused_glue as FG

    return FG.scale_code_table(R, "cuda"), FG.silu_table("cuda", lim)


# ------------------------------------------------------------------------------------------------ kernels
def test_act_quant_kernels_vs_reference_all_bf16():
    _need_glue()
    from runtime import fused_glue as FG

    R = _R()
    code_tab, _ = _tables(R)
    allb = FG.bf16_bits_all()
    fin = allb[torch.isfinite(allb.float())]
    n = fin.numel() // 32 * 32
    g = torch.Generator().manual_seed(1)
    for x in (fin[:n].view(-1, 32).cuda(), fin[:n][torch.argsort(fin[:n].float().abs())].view(-1, 32).cuda(),
              _act(g, 64, DIM)):
        q_ref, s_ref = R.act_quant(x.contiguous(), 32, R.scale_fmt, R.scale_dtype)
        q, s = torch.empty_like(q_ref), torch.empty_like(s_ref)
        FG.launch_act_quant_bcast(x, q, s, code_tab, 1)
        assert _eq(q, q_ref) and _eq(s, s_ref), "act_quant kernel vs reference"
        y_ref = R.act_quant(x.clone(), 32, R.scale_fmt, R.scale_dtype, True)
        y = x.clone()
        FG.launch_o_rope_qdq(y.view(-1), torch.zeros(1, 2, device="cuda"), code_tab, 32, 0, (0, 0))
        assert _eq(y, y_ref), "QDQ kernel vs reference act_quant(inplace)"
    # dg_act: broadcast into row 0 of k groups with fp32 scales (= act_quant + load_act)
    x = _act(g, 1, DIM)
    q_ref, s_ref = R.act_quant(x, 32, R.scale_fmt, R.scale_dtype)
    xq = torch.zeros(K_ACT, 8, DIM, dtype=FP8, device="cuda")
    xs = torch.zeros(K_ACT, 8, DIM // 32, device="cuda")
    FG.launch_act_quant_bcast(x, xq[:, 0], xs[:, 0], code_tab, K_ACT)
    assert _eq(xq[:, 0], q_ref.expand(K_ACT, -1)) and torch.equal(xs[:, 0], s_ref.float().expand(K_ACT, -1))
    assert not xq[:, 1:].view(torch.uint8).any() and not xs[:, 1:].any()
    print("  act_quant / QDQ / dg_act kernels == reference on all finite bf16 + real shapes")


def _gate_up(g, G, I, lim):
    from runtime.fused_glue import _rand_act

    v = _rand_act(g, 2 * G, I, -3, 7).view(2, G, I)
    lb = torch.tensor([lim], dtype=BF)
    nxt, prv = float((lb.view(torch.int16) + 1).view(BF)), float((lb.view(torch.int16) - 1).view(BF))
    for j, val in enumerate((lim, -lim, 1.5 * lim, -1.5 * lim, nxt, -nxt, prv, -prv, 0.0, -0.0)):
        v[:, :, j] = val
    return v[0].cuda(), v[1].cuda()


def _ref_glue(gate, up, w, lim):
    gate, up = gate.float(), up.float()
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    if w is not None:
        h = w.view(-1, 1) * h
    return h.to(BF)


def test_swiglu_kernels_vs_reference_real_shapes():
    _need_glue()
    from runtime import fused_glue as FG

    R = _R()
    code_tab, silu_tab = _tables(R)
    g = torch.Generator().manual_seed(2)
    for G_, with_w, f32 in ((K_ACT, True, True), (1, False, False)):          # routed (dg rows), shared
        for _ in range(4):
            gate, up = _gate_up(g, G_, INTER, LIM)
            gb = torch.zeros(G_, 8, INTER, dtype=BF, device="cuda")
            ub = torch.zeros_like(gb)
            gb[:, 0], ub[:, 0] = gate, up
            w = (torch.rand(G_, generator=g) * 2).float().cuda() if with_w else None
            h = _ref_glue(gate, up, w, LIM)
            q_ref, s_ref = R.act_quant(h, 32, R.scale_fmt, R.scale_dtype)
            q = torch.zeros(G_, 8, INTER, dtype=FP8, device="cuda")
            s = torch.zeros(G_, 8, INTER // 32, device="cuda") if f32 else \
                torch.zeros(G_, 8, INTER // 32, dtype=torch.uint8, device="cuda").view(E8)
            FG.launch_swiglu_quant(gb[:, 0], ub[:, 0], w, q[:, 0], s[:, 0], silu_tab, code_tab, LIM)
            assert _eq(q[:, 0], q_ref), ("swiglu q", G_)
            assert torch.equal(s[:, 0], s_ref.float()) if f32 else _eq(s[:, 0], s_ref), ("swiglu s", G_)
    print("  routed + shared SwiGLU+act_quant kernels == reference glue at [6|1, 2304]")


def test_combine_and_hc_post_kernels():
    _need_glue()
    from runtime import fused_glue as FG

    g = torch.Generator().manual_seed(3)
    for _ in range(4):
        rows = torch.zeros(K_ACT, 8, DIM, dtype=BF, device="cuda")
        rows[:, 0] = _act(g, K_ACT, DIM, -8, 8)
        sh = _act(g, 1, DIM, -8, 8)
        y = torch.zeros(1, DIM, device="cuda")
        for j in range(K_ACT):
            y += rows[j, 0]
        y += sh
        out = torch.empty(DIM, dtype=BF, device="cuda")
        FG.launch_moe_combine(rows[:, 0], sh.view(-1), out)
        assert _eq(out, y.type_as(sh).view(-1)), "combine"
    match = {m: 0 for m in FG.HC_SUM_MODES}
    for _ in range(6):
        x = _act(g, 1, DIM).view(1, 1, DIM)
        res = _act(g, HC, DIM).view(1, 1, HC, DIM)
        post = (torch.rand(1, 1, HC, generator=g) * 2).cuda()
        comb = (torch.randn(1, 1, HC, HC, generator=g) * torch.exp2(torch.randint(-4, 5, (1, 1, HC, HC),
                                                                                   generator=g).float())).cuda()
        big = (torch.randn(DIM, generator=g) * torch.exp2(torch.randint(10, 17, (DIM,), generator=g).float())).to(BF)
        cols = (torch.arange(DIM) % 2 == 0).cuda()
        res[0, 0, 0, cols] = big.cuda()[cols]
        res[0, 0, 2, cols] = big.cuda()[cols]
        comb[0, 0, 2, :] = -comb[0, 0, 0, :]
        ref = (post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)
               ).type_as(x).view(HC, DIM)
        for m in FG.HC_SUM_MODES:
            out = torch.empty(HC, DIM, dtype=BF, device="cuda")
            FG.launch_hc_post(x.view(DIM), res.view(HC, DIM), post.view(HC), comb.view(HC, HC).contiguous(), out, m)
            match[m] += int((out.view(torch.int16) != ref.view(torch.int16)).sum())
    ok = [m for m in FG.HC_SUM_MODES if match[m] == 0]
    print(f"  combine == fp32 slot-order sum at [6, 5120]; hc_post mismatches per sum order {match} -> {ok}")
    assert ok, f"no hc sum order matches torch on this device ({match}): cluster #7 must stay unfused"


def _freqs(pos, rd=RD, theta=10000.0):
    f = 1.0 / (theta ** (torch.arange(0, rd, 2).float() / rd))
    return torch.polar(torch.ones(rd // 2), torch.outer(torch.tensor([float(pos)]), f)).cuda()


def test_rope_kernels_modes_on_device():
    """o path (#10) and the wq_b GEMV epilogue (#13): which contraction forms equal apply_rotary_emb here."""
    _need_glue()
    from runtime import fused_glue as FG
    from runtime import gemv_mxfp8 as G

    R = _R()
    code_tab, _ = _tables(R)
    g = torch.Generator().manual_seed(4)
    modes = [(a, b) for a in FG.ROPE_MODES for b in FG.ROPE_MODES]
    o_ok, q_ok = set(modes), set(modes)
    w = (torch.randint(0, 0x78, (HEADS * HD, Q_LORA), generator=g, dtype=torch.int32) |
         (torch.randint(0, 2, (HEADS * HD, Q_LORA), generator=g, dtype=torch.int32) << 7)).to(torch.uint8).view(FP8)
    ws = torch.randint(118, 131, (HEADS * HD // 32, Q_LORA // 32), generator=g, dtype=torch.int32).to(
        torch.uint8).view(E8)
    w, ws = w.cuda(), ws.cuda()
    for pos in (1, 37, 1000, 4095):
        f = _freqs(pos)
        fr = torch.view_as_real(f).reshape(-1, 2).float().contiguous()
        o = _act(g, 1, HEADS * HD).view(1, 1, HEADS, HD)
        ref = o.clone()
        R.apply_rotary_emb(ref[..., -RD:], f, True)
        ref = ref.view(1, 1, 8, -1)
        R.act_quant(ref, 32, R.scale_fmt, R.scale_dtype, True)
        qr = _act(g, 1, Q_LORA, -4, 4)
        xq, xs = R.act_quant(qr, 32, R.scale_fmt, R.scale_dtype)
        plain = G.linear_fp8(xq, xs, w, ws).view(1, 1, HEADS, HD)
        qref = plain.clone()
        R.apply_rotary_emb(qref[..., -RD:], f)
        for m in modes:
            y = o.clone()
            FG.launch_o_rope_qdq(y.view(-1), fr, code_tab, HD, RD, m)
            if not _eq(y.view(-1), ref.view(-1)):
                o_ok.discard(m)
            got = G.linear_fp8_rope(xq, xs, w, ws, fr, HD, RD, m).view(1, 1, HEADS, HD)
            if not _eq(got, qref):
                q_ok.discard(m)
    print(f"  RoPE forms equal to apply_rotary_emb on this device: o path (inverse) {sorted(o_ok)}, "
          f"wq_b GEMV epilogue (forward) {sorted(q_ok)}")
    assert o_ok and q_ok, "a RoPE cluster has no matching form: it must stay unfused (reported by describe())"


def test_gather_tables_vs_six_gathers():
    _need_glue()
    from runtime.static_kernels import gather_rows

    with _model("mini4") as (model, tokdir):
        for impl in ("ref", "dg-masked"):
            gen = _gen(model, tokdir, moe_impl=impl, glue_impl="fused")
            st = gen.static_step()
            assert st.glue.on("gather"), st.glue.describe()["clusters"]
            for li, sm in enumerate(st.moes):
                E = sm.moe.n_routed_experts
                ids = torch.randperm(E, generator=torch.Generator().manual_seed(li))[: st.topk].sort().values.cuda()
                ref = {}
                for key, tab in sm.ptabs.items():
                    gather_rows(tab, ids, st.staging[key], st.row_bytes[key])
                    ref[key] = st.staging[key].clone()
                if impl == "dg-masked":
                    for p in ("w1", "w3", "w2"):
                        st.dg.convert_scales(p, st.staging[(p, "scale")])
                        ref[p] = st.dg.sf32[p].clone()
                for t in st.glue._dsts:
                    t.zero_()
                st.glue.gather(sm, ids)
                torch.cuda.synchronize()
                for p in ("w1", "w3", "w2"):
                    assert _eq(st.staging[(p, "weight")], ref[(p, "weight")]), (impl, li, p)
                    if impl == "dg-masked":
                        assert torch.equal(st.dg.sf32[p].view(torch.int32), ref[p].view(torch.int32)), (impl, li, p)
                    else:
                        assert _eq(st.staging[(p, "scale")], ref[(p, "scale")]), (impl, li, p)
                assert torch.equal(st.ids_step[sm.layer], ids)
            gen.reset_static()
            del gen
    print("  one-launch gather == 6 x gather_rows (+ convert_scales) on mini4's tables, ref and dg-masked")


# ------------------------------------------------------------------------------------------------ mini checkpoints
def _fused_vs_ref(kind, moe, fp8):
    _need_glue()
    if moe != "ref":
        from runtime.dg_moe import import_deep_gemm

        import_deep_gemm()
    from runtime.fused_glue import CLUSTERS

    with _model(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=moe, fp8_impl=fp8, glue_impl="fused")
        st = gen.static_step()
        d = st.glue.describe()
        print(f"  {kind} {moe}/{fp8}: clusters {d['clusters']}; modes {d['modes']}")
        want_on = {"gather", "swiglu_shared", "hc_post", "o_rope_qdq"} | (
            {"dg_act", "swiglu_routed", "moe_combine"} if moe != "ref" else set()) | (
            {"q_rope_epilogue"} if fp8 == "gemv" else set())
        assert set(d["on"]) == want_on, {c: d["clusters"][c] for c in CLUSTERS if c not in d["on"]}
        runs = {}
        for L, seed in ((16, 5), (17, 6)):                               # both prefill parities
            ids = _ids(L, seed)
            runs[L] = (ids, _run(gen, "static", ids))
        ids = runs[16][0]                                                # no host sync in the fused step
        gen.set_mode("static")
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
        _assert_equal_runs(g1, runs[16][1], f"{kind} {moe}/{fp8}: fused graph vs fused static-eager")
        g2 = _run(gen, "graph", ids)
        _assert_equal_runs(g2, g1, f"{kind}: fused graph replay#2")
        gen.capture(fills=True)
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: fused fills-on recapture")
        gen.reset_static()
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: fused relaunch")
        gen.set_glue_impl("ref")
        for L, (ids_, fused) in runs.items():
            ref = _run(gen, "static", ids_)
            _assert_equal_runs(fused, ref, f"{kind} {moe}/{fp8} L{L}: fused glue vs ref glue static-eager")
        assert gen.static.glue is None
        print(f"  {kind} {moe}/{fp8}: fused == ref static-eager (prefill + {N_STEPS} steps, both parities); graph "
              f"== static, replay x2, fills-on, relaunch; no host sync")
        gen.reset_static()
        del gen


def test_fused_vs_ref_mini_refref():
    _fused_vs_ref("mini", "ref", "ref")


def test_fused_vs_ref_mini4_refref():
    _fused_vs_ref("mini4", "ref", "ref")


def test_fused_vs_ref_mini_dg_gemv():
    _fused_vs_ref("mini", "dg-masked", "gemv")


def test_fused_vs_ref_mini4_dg_gemv():
    _fused_vs_ref("mini4", "dg-masked", "gemv")


def test_fused_step_cell_informative():
    _need_glue()
    fl = _first_light()
    with _model("mini4") as (model, tokdir):
        p50 = {}
        for glue in ("ref", "fused"):
            gen = _gen(model, tokdir, moe_impl="dg-masked", fp8_impl="gemv", glue_impl=glue, mode="graph")
            sc = fl.step_cell(gen, _ids(16, 9), 40, 8)
            p50[glue] = sc["ms_per_step"]["p50"]
            gen.reset_static()
            del gen
        print(f"  mini4 dg-masked/gemv graph step cell p50: glue ref {p50['ref']:.3f} ms, fused {p50['fused']:.3f} ms")


TESTS = [test_act_quant_kernels_vs_reference_all_bf16, test_swiglu_kernels_vs_reference_real_shapes,
         test_combine_and_hc_post_kernels, test_rope_kernels_modes_on_device, test_gather_tables_vs_six_gathers,
         test_fused_vs_ref_mini_refref, test_fused_vs_ref_mini4_refref, test_fused_vs_ref_mini_dg_gemv,
         test_fused_vs_ref_mini4_dg_gemv, test_fused_step_cell_informative]

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
