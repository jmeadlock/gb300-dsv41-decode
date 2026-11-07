"""GPU tests of T9b-3c (`--attn-impl flashmla`, runtime/attn_flashmla.py). Skipped without CUDA/tilelang (Mac); on
the box a FlashMLA load failure FAILS (it must load from vllm's .so with no vllm Python).

  * load: torch.ops.load_library on <dist-packages>/vllm/_flashmla_C*.so, 'vllm' not in sys.modules before and after;
  * real shapes (64 heads x 512, softmax_scale 512^-0.5, sink; kv = 128-slot window ring + 2048 (r=2) / 4096 (r=1)
    compressed rows at max_pos 4095, values through the reference QDQs: window act_quant FP8/ue8m0-32, latent
    fp4_act_quant e4m3-16), indices as the static step builds them (window oldest-first, leading -1 at p < 127;
    compressed sorted, -1 tail, 640 wide; > 512 reachable -> 512 sorted picks as a top-k stand-in), at positions
    300 / 1100 / 4000, 1 and 6 rows, ratio 0 / 2 / 1: flashmla vs the reference kernel.py `sparse_attn` -- mismatch %,
    max |delta|, and the fp64-oracle error ratio (gate <= 1.1x the reference kernel's own error, every case; outputs
    finite). Reported: topk_length call == -1-masked call, 6 rows == each row alone;
  * graph: 40 calls (2 ratio-0 + 18 ratio-2 + 20 ratio-1, per-layer window rings, one compressed cache per kv-source
    group, as the probe) in one CUDA graph, b = 1 and 6, p = 300 / 1100 / 4000: replay x2 equal and graph == eager
    (gated); us/call for flashmla (C) vs the reference at 640 (informative) and, at 300 and 4000, FlashMLA's decode
    kernel over 528 / 288 B records (B, the probe's arm, informative: whether C's cost grows with context where B's
    does not -- the handoff's B-vs-C question);
  * mini checkpoints (mini, mini4, mini4w = mini4 at index_topk 256; mini4w also with dg-masked / gemv / fused glue):
    flashmla self-check on; static-eager runs prefill + 10 decode steps (both parities); no host sync; graph ==
    static-eager, replay #2, fills-on recapture, relaunch (gated, bitwise); top-1 stream == attn `ref` with the same
    other impls over the 10 steps (gated; both streams printed when not), max relative logit difference reported.
    Informative: graph step-cell p50 ref vs flashmla on mini4w.

Run on the box (orchestrator), from the repo checkout, inside the library image (same mounts as the other GPU tests;
no DeepGEMM cache needed unless the dg-masked mini case runs, which then JITs into the container):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    --cap-add IPC_LOCK -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_attn_flashmla_gpu.py

(`python3 tests/gpu/test_attn_flashmla_gpu.py` runs every test without pytest; no checkpoint mount needed.)
"""
from __future__ import annotations

import os
import random
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

from test_attn_bound_gpu import _gen, _kernel_mod, _model_w, _need_triton  # noqa: E402
from test_static_mini_gpu import N_STEPS, _assert_equal_runs, _first_light, _ids, _need_tilelang, _run  # noqa: E402

H, D, WIN, TOPK = 64, 512, 128, 512
SCALE = D ** -0.5
MAX_POS = 4095
WR = {2: (MAX_POS + 1) // 2, 1: (MAX_POS + 1) // 1}
POSITIONS = (300, 1100, 4000)
ROWS = (1, 6)
TYPES = [0] * 2 + [2] * 18 + [1] * 20           # compress_ratios[0:40] (config.json)
RATIO_GATE = 1.1                                 # handoff: fp64-oracle error <= 1.1x the reference kernel's
BF, E8, E4, U8 = torch.bfloat16, torch.float8_e8m0fnu, torch.float8_e4m3fn, torch.uint8
PAGE = 64
DEV = "cuda"                                     # helpers build on this device (a CPU dry run may override it)


def _need_flashmla():
    _need_tilelang()
    from runtime import attn_flashmla as AFM

    assert not AFM.vllm_modules(), f"vllm already imported in the test process: {AFM.vllm_modules()}"
    AFM.load_ops()                                       # FAILS on the box if it cannot load without vllm
    assert not AFM.vllm_modules(), AFM.vllm_modules()
    return AFM


def _bits(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _rel(o, ref64):
    return ((o.double() - ref64).norm() / ref64.norm()).item()


# ------------------------------------------------------------------------------------------------ inputs (probe)
def _win_idx(pos):
    """rt/static_step.py window_topk_idxs_dev on the host: slots oldest-first, -1 where slot > pos."""
    oldest = pos % WIN + 1
    return [s if s <= pos else -1 for s in ((oldest + j) % WIN for j in range(WIN))]


def _comp_valid(pos, r):
    c = (pos + 1) // r
    if c <= TOPK:
        return list(range(c))
    rnd = random.Random(1000003 * pos + r)
    return sorted(rnd.sample(range(c), TOPK))


class Data:
    """Per layer type: window rows for every position (pre-QDQ -> reference QDQ in place = the bf16 cache values,
    and codes/scales for the 528 B records) and, for r > 0, the compressed latent (fp4 QDQ e4m3-16 in place, codes for
    the 288 B records). Synthetic N(0,1) values (no checkpoint here); q is N(0,1) bf16."""

    def __init__(self, Kmod, n_pos, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.K, self.t = Kmod, {}
        for t in (0, 2, 1):
            d = {}
            win = torch.randn(n_pos, D, generator=g, device=DEV).to(BF)
            d["win"] = Kmod.act_quant(win.clone(), 32, "ue8m0", E8, True)
            c, s = Kmod.act_quant(win.clone(), 32, "ue8m0", E8)
            d["win_codes"], d["win_sc"] = c.view(U8), s.view(U8)
            if t:
                lat = torch.randn(WR[t], D, generator=g, device=DEV).to(BF)
                d["lat"] = Kmod.fp4_act_quant(lat.clone(), 16, True, scale_dtype=E4)
                c, s = Kmod.fp4_act_quant(lat.clone(), 16, False, scale_dtype=E4)
                d["lat_pages"] = _pack_pages(c.view(U8), s.view(U8), PAGE)
            d["sink"] = torch.randn(H, generator=g, device=DEV).float()
            self.t[t] = d
        self.g = g


def _pack_pages(data_u8, scale_u8, P):
    """vLLM's V4.1 page: P data rows then P scale rows (cache_utils.py:481-484; bench/flashmla_probe.py)."""
    T_, bd = data_u8.shape
    bs = scale_u8.size(1)
    nb = -(-T_ // P)
    if nb * P != T_:
        data_u8 = torch.cat([data_u8, data_u8.new_zeros(nb * P - T_, bd)])
        scale_u8 = torch.cat([scale_u8, scale_u8.new_full((nb * P - T_, bs), 127)])
    page = torch.empty(nb, P * (bd + bs), dtype=U8, device=data_u8.device)
    page[:, :P * bd] = data_u8.reshape(nb, P * bd)
    page[:, P * bd:] = scale_u8.reshape(nb, P * bs)
    return page.view(nb, P, 1, bd + bs)


def _case(data, t, p, b, comp=None):
    """One call's inputs at rows p..p+b-1 (one ring written up to p+b-1, as the probe): q [1, b, H, D], kv [1, n, D],
    idx [1, b, w] (w = 128 at r = 0, 640 else), tlen [b] (None at r = 0), plus the B-arm records/indices."""
    d = data.t[t]
    pos = [p + i for i in range(b)]
    p_last = pos[-1]
    ring_pos = [p_last - ((p_last - s) % WIN) for s in range(WIN)]
    rp = torch.tensor([max(x, 0) for x in ring_pos], device=DEV)
    empty = torch.tensor([x < 0 for x in ring_pos], device=DEV)
    ring = d["win"][rp].clone()
    ring[empty] = 0
    rc, rs = d["win_codes"][rp].clone(), d["win_sc"][rp].clone()
    rc[empty], rs[empty] = 0, 127
    wl = [_win_idx(x) for x in pos]
    q = torch.randn(1, b, H, D, generator=data.g, device=DEV).to(BF)
    ci = {"t": t, "q": q, "sink": d["sink"], "wpages": _pack_pages(rc, rs, PAGE)}
    wvalid = [[s for s in w if s >= 0] for w in wl]
    ci["B_widx"] = torch.tensor([[v + [-1] * (WIN - len(v))] for v in wvalid], dtype=torch.int32, device=DEV)
    ci["B_wlen"] = torch.tensor([len(v) for v in wvalid], dtype=torch.int32, device=DEV)
    if t == 0:
        ci["kv"] = ring.unsqueeze(0).contiguous()
        ci["idx"] = torch.tensor([wl], dtype=torch.int32, device=DEV)
        ci["tlen"] = None
    else:
        lat = comp if comp is not None else d["lat"]
        cv = [_comp_valid(x, t) for x in pos]
        ci["kv"] = torch.cat([ring, lat], 0).unsqueeze(0).contiguous()
        ci["idx"] = torch.tensor([[w + [v + WIN for v in c] + [-1] * (TOPK - len(c)) for w, c in zip(wl, cv)]],
                                 dtype=torch.int32, device=DEV)
        ci["tlen"] = torch.tensor([WIN + len(c) for c in cv], dtype=torch.int32, device=DEV)
        ci["cpages"] = d["lat_pages"]
        ci["B_cidx"] = torch.tensor([[c + [-1] * (TOPK - len(c))] for c in cv], dtype=torch.int32, device=DEV)
        ci["B_clen"] = torch.tensor([len(c) for c in cv], dtype=torch.int32, device=DEV)
    return ci


# ------------------------------------------------------------------------------------------------ load
def test_load_without_vllm():
    AFM = _need_flashmla()
    info = AFM.loaded()
    import torch as _t

    schema = [str(getattr(_t.ops._flashmla_C.sparse_prefill_fwd, ov)._schema)
              for ov in _t.ops._flashmla_C.sparse_prefill_fwd.overloads()]
    assert info["loaded"] and not info["vllm_modules"], info
    print(f"  loaded {info['so']} with no vllm module; {schema}")


# ------------------------------------------------------------------------------------------------ real shapes
def test_flashmla_vs_reference_real_shapes():
    AFM = _need_flashmla()
    Kmod = _kernel_mod()
    torch.set_default_device("cpu")
    data = Data(Kmod, max(POSITIONS) + max(ROWS) + 1, seed=1)
    worst, rows_out, bad = 0.0, [], []
    tl_eq_all, minv_all = True, True
    for t in (0, 2, 1):
        for p in POSITIONS:
            for b in ROWS:
                ci = _case(data, t, p, b)
                got = AFM.sparse_attn_flashmla(ci["q"], ci["kv"], ci["sink"], ci["idx"], SCALE, ci["tlen"])
                want = Kmod.sparse_attn(ci["q"], ci["kv"], ci["sink"], ci["idx"], SCALE)
                o64 = AFM.oracle64(ci["q"], ci["kv"], ci["sink"], ci["idx"], SCALE)
                torch.cuda.synchronize()
                assert got.shape == want.shape and got.dtype == want.dtype and got.is_contiguous()
                assert bool(torch.isfinite(got.float()).all()), (t, p, b)
                e_fm, e_ref = _rel(got, o64), _rel(want, o64)
                ratio = e_fm / e_ref
                worst = max(worst, ratio)
                mism = (got.view(torch.int16) != want.view(torch.int16)).float().mean().item() * 100
                dmax = (got.float() - want.float()).abs().max().item()
                tl_eq = True
                if ci["tlen"] is not None:
                    tl_eq = _bits(got, AFM.sparse_attn_flashmla(ci["q"], ci["kv"], ci["sink"], ci["idx"], SCALE, None))
                minv = all(_bits(got[:, r:r + 1], AFM.sparse_attn_flashmla(
                    ci["q"][:, r:r + 1].contiguous(), ci["kv"], ci["sink"], ci["idx"][:, r:r + 1].contiguous(), SCALE,
                    None if ci["tlen"] is None else ci["tlen"][r:r + 1])) for r in range(b))
                tl_eq_all &= tl_eq
                minv_all &= minv
                rows_out.append(f"    r={t} p={p:4d} b={b}: width {ci['idx'].size(-1)} mismatch {mism:6.2f}% max|d| "
                                f"{dmax:.2e} err fm {e_fm:.3e} ref {e_ref:.3e} ratio {ratio:.4f} tlen==mask {tl_eq} "
                                f"rows-indep {minv}")
                if ratio > RATIO_GATE:
                    bad.append((t, p, b, round(ratio, 4)))
    print("\n".join(rows_out))
    print(f"  worst fp64-oracle ratio {worst:.4f} (gate {RATIO_GATE}); topk_length == -1-masked everywhere: "
          f"{tl_eq_all}; rows independent everywhere: {minv_all}")
    assert not bad, f"fp64-oracle error > {RATIO_GATE}x the reference kernel's: {bad}"


# ------------------------------------------------------------------------------------------------ graph + speed
def _group_of(layer):
    return 2 if layer < 8 else 8 if layer < 14 else 14 if layer < 20 else 20


def _step_layers(data, b, p):
    per_type = {t: _case(data, t, p, b) for t in (0, 2, 1)}
    comp, layers = {}, []
    for l, t in enumerate(TYPES):
        ci = dict(per_type[t])
        ring = ci["kv"][0, :WIN].clone()
        ci["wpages"] = ci["wpages"].clone()
        if t:
            g = _group_of(l)
            if g not in comp:
                comp[g] = (data.t[t]["lat"].clone(), ci["cpages"].clone())
            ci["kv"] = torch.cat([ring, comp[g][0]], 0).unsqueeze(0).contiguous()
            ci["cpages"] = comp[g][1]
        else:
            ci["kv"] = ring.unsqueeze(0).contiguous()
        layers.append(ci)
    return layers


class _Sched:
    def __init__(self):
        self.tsm = self.ns = None


def _arm_B(ops, ci, sched):
    """bench/flashmla_probe.py arm_B: sparse_decode_fwd over the 528 B window pages (+ 288 B compressed pages)."""
    t = ci["t"]
    qB = ci["q"][0].unsqueeze(1)                                           # [b, 1, H, D] rows as batch
    out, lse, tsm, ns = ops.sparse_decode_fwd(qB, ci["wpages"], ci["B_widx"], ci["B_wlen"], ci["sink"], sched.tsm,
                                              sched.ns, ci["cpages"] if t else None, ci["B_cidx"] if t else None,
                                              ci["B_clen"] if t else None, D, SCALE, None)
    sched.tsm, sched.ns = tsm, ns
    return out[:, 0]


def _time_graph(body, reps=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
        eager = [o.clone() for o in body()]
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = body()
    g.replay()
    torch.cuda.synchronize()
    r1 = [o.clone() for o in outs]
    g.replay()
    torch.cuda.synchronize()
    replay_eq = all(_bits(a, b) for a, b in zip(r1, outs))
    graph_eq = all(_bits(a, b) for a, b in zip(r1, eager))
    for _ in range(3):
        g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / reps
    del g
    return {"replay_equal": replay_eq, "graph_eq_eager": graph_eq, "us_per_graph": us, "us_per_call": us / len(r1)}


def test_flashmla_graph_capture_replay_and_speed():
    AFM = _need_flashmla()
    Kmod = _kernel_mod()
    torch.set_default_device("cpu")
    data = Data(Kmod, max(POSITIONS) + max(ROWS) + 1, seed=2)
    ops = AFM.load_ops()
    bad, lines = [], []
    for b in ROWS:
        for p in POSITIONS:
            layers = _step_layers(data, b, p)
            arms = {
                "C flashmla": lambda: [AFM.sparse_attn_flashmla(L["q"], L["kv"], L["sink"], L["idx"], SCALE, L["tlen"])
                                       for L in layers],
                "ref@640": lambda: [Kmod.sparse_attn(L["q"], L["kv"], L["sink"], L["idx"], SCALE) for L in layers],
            }
            if p in (300, 4000):
                def bB():
                    sch = {0: _Sched(), 1: _Sched(), 2: _Sched()}   # one per layer type, inside the captured body
                    return [_arm_B(ops, L, sch[L["t"]]) for L in layers]
                arms["B decode 528/288 (info)"] = bB
            res = {}
            for name, body in arms.items():
                try:
                    res[name] = _time_graph(body)
                except Exception as e:  # B is informative only
                    if name.startswith("C"):
                        raise
                    res[name] = {"err": f"{type(e).__name__}: {str(e)[:200]}"}
            r = res["C flashmla"]
            if not (r["replay_equal"] and r["graph_eq_eager"]):
                bad.append((b, p, r))
            lines.append(f"    b={b} p={p:4d}: " + "; ".join(
                f"{k} {v['us_per_call']:.2f} us/call ({v['us_per_graph']:.0f} us/40) replay_eq {v['replay_equal']} "
                f"graph==eager {v['graph_eq_eager']}" if "err" not in v else f"{k} ERR {v['err']}"
                for k, v in res.items()))
            del layers
            torch.cuda.empty_cache()
    print("\n".join(lines))
    assert not bad, f"flashmla graph replay unequal or graph != eager: {bad}"


# ------------------------------------------------------------------------------------------------ mini checkpoints
def _flashmla_vs_ref(kind, moe="ref", fp8="ref", glue="ref"):
    _need_triton()
    _need_flashmla()
    if moe != "ref":
        from runtime.dg_moe import import_deep_gemm

        import_deep_gemm()
    with _model_w(kind) as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl=moe, fp8_impl=fp8, glue_impl=glue, attn_impl="flashmla")
        st = gen.static_step()
        d = st.attn_describe()
        chk = d.get("check") or {}
        print(f"  {kind} {moe}/{fp8}/{glue}: flashmla on={d['on']} reason={d.get('reason')} max_err_ratio="
              f"{chk.get('max_err_ratio')} tlen==mask {chk.get('topk_length_eq_masked')} rows-indep "
              f"{chk.get('rows_m_invariant')} padded heads {chk.get('padded_heads')}")
        assert d["on"], d
        runs = {}
        for L, seed in ((16, 5), (17, 6)):
            ids = _ids(L, seed)
            runs[L] = (ids, _run(gen, "static", ids))
        ids = runs[16][0]
        gen.set_mode("static")                                      # no host sync in the flashmla step
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
        _assert_equal_runs(g1, runs[16][1], f"{kind}: flashmla graph vs flashmla static-eager")
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: flashmla graph replay#2")
        gen.capture(fills=True)
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: flashmla fills-on recapture")
        gen.reset_static()
        _assert_equal_runs(_run(gen, "graph", ids), g1, f"{kind}: flashmla relaunch")
        gen.set_attn_impl("ref")
        top1_bad = []
        for L, (ids_, fm) in runs.items():
            ref = _run(gen, "static", ids_)
            rel = max(((a - b).norm() / b.norm()).item() for a, b in zip(fm[0], ref[0]))
            print(f"  {kind} L{L}: max relative logit difference flashmla vs ref attn {rel:.3e}; top-1 equal "
                  f"{fm[1] == ref[1]}")
            if fm[1] != ref[1]:
                print(f"    flashmla top-1: {fm[1]}\n    ref      top-1: {ref[1]}")
                top1_bad.append(L)
        assert gen.static.attn_describe() == {"impl": "ref", "on": False}
        gen.reset_static()
        del gen
        assert not top1_bad, f"{kind}: top-1 stream differs from attn ref at prompt lengths {top1_bad} (printed above)"
        print(f"  {kind} {moe}/{fp8}/{glue}: flashmla static-eager ran prefill + {N_STEPS} steps (both parities); "
              f"graph == static, replay x2, fills-on, relaunch; no host sync; top-1 == ref attn")


def test_flashmla_mini():
    _flashmla_vs_ref("mini")


def test_flashmla_mini4():
    _flashmla_vs_ref("mini4")


def test_flashmla_mini4w():
    _flashmla_vs_ref("mini4w")


def test_flashmla_mini4w_dg_gemv_fused():
    _flashmla_vs_ref("mini4w", "dg-masked", "gemv", "fused")


def test_flashmla_step_cell_informative():
    _need_triton()
    _need_flashmla()
    fl = _first_light()
    with _model_w("mini4w") as (model, tokdir):
        p50 = {}
        for attn in ("ref", "flashmla"):
            gen = _gen(model, tokdir, attn_impl=attn, mode="graph")
            sc = fl.step_cell(gen, _ids(16, 9), 40, 8)
            p50[attn] = sc["ms_per_step"]["p50"]
            gen.reset_static()
            del gen
        print(f"  mini4w graph step cell p50: attn ref {p50['ref']:.3f} ms, flashmla {p50['flashmla']:.3f} ms")


TESTS = [test_load_without_vllm, test_flashmla_vs_reference_real_shapes, test_flashmla_graph_capture_replay_and_speed,
         test_flashmla_mini, test_flashmla_mini4, test_flashmla_mini4w, test_flashmla_mini4w_dg_gemv_fused,
         test_flashmla_step_cell_informative]

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
