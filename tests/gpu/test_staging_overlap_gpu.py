"""GPU tests of M3-S1 (docs/HANDOFF-M3-S1-STAGING-OVERLAP.md): cold-expert staging overlap ($DSV41_STAGE).
Skipped without CUDA/tilelang (Mac); on the box a missing/broken deep_gemm FAILS (it is under test).

  * s1_stage_proj on the device == stage_proj_emulated per projection (weights / scales / both; real w1 row = EVEN
    path, an odd size = masked path; nprog 48 / 16 / 8), and the three projections == m1_stage_rows byte for byte;
  * the copy-engine path (ce_plan + one cudaMemcpyAsync per expert x projection from pinned UVA host memory + the
    scale-only launch) == m1_stage_rows;
  * mini4 6-row dg-inplace StaticStep (map: 4 of 8 experts cold per layer, so most layer-steps stage) in every mode
    == the default `kernel` mode, eager: torch.equal on logits, main_hidden, caches, spare slots, packed SF;
    kernel-overlap captured: graph == static-eager (fork/join inside the capture, no host sync); a ce mode under
    capture raises StageCaptureError; overflow 0; graph replay ms/step of kernel vs kernel-overlap on the mini
    (tiny shapes: a smoke number, labeled). M3-S1-FIX: every compared run starts from the pre-base checkpoint
    (round 1's ('kernel-overlap', 'L1.ckv') was the replay timing loop's cache residue, see _checkpoint), kernel-
    overlap is compared $DSV41_S1_GATE_ITERS (20) times eager + 20 times as a graph, the per-projection launch shape
    (NPROG 48,16,16 / BLOCK_BYTES 8192,32768,32768) 5 + 5 times, and a residue control shows the round-1 mechanism
    with kernel vs kernel (logits equal, L1.ckv differs);
  * the probe (bench/staging_overlap_bench.py probe, quick: 4 rounds, 10 reps) runs end to end and its byte / output
    checks pass; receipt scratch/staging-overlap-probe-quick-<date>.json.

Run on the box exactly as tests/gpu/test_moe_rows_gpu.py (same image, mounts, DG_JIT_CACHE_DIR):
  ... --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from test_dg_moe_gpu import _need_dg  # noqa: E402
from test_static_rows_gpu import COUNTS, ROWS, _assert_rows, _model, _need_step, _static, _toks  # noqa: E402

RESULTS = {}


def _layer(N, KH, dev="cuda", seed=0, pin=True):
    """The synthetic layer of test_moe_rows_gpu (E 8, hot [6,1,3,0,5], cold [2,4,7] in pinned host memory; pin=False:
    plain CPU tensors, tests/test_staging_overlap_cpu.py's dry run of this file's argument plumbing)."""
    from runtime import dg_inplace as DI

    KS = KH * 2 // 32
    g = torch.Generator().manual_seed(seed + N)
    E, hot, cold = 8, [6, 1, 3, 0, 5], [2, 4, 7]
    H, S = len(hot), 3
    src_w = {(p, e): torch.randint(-128, 128, (N, KH), generator=g, device="cpu").to(torch.int8) for p in DI.PROJS
             for e in range(E)}
    src_s = {(p, e): torch.randint(100, 140, (N, KS), generator=g, device="cpu").to(torch.uint8) for p in DI.PROJS
             for e in range(E)}
    keep = (lambda t: t.pin_memory()) if pin else (lambda t: t.clone())
    host_w = {(p, e): keep(src_w[(p, e)]) for p in DI.PROJS for e in cold}
    host_s = {(p, e): keep(src_s[(p, e)]) for p in DI.PROJS for e in cold}
    return dict(N=N, KH=KH, KS=KS, E=E, hot=hot, cold=cold, H=H, S=S, src_w=src_w, src_s=src_s, host_w=host_w,
                host_s=host_s)


def _dsts(L, step_sf, dev="cuda"):
    from runtime import dg_inplace as DI
    from runtime import moe_rows as MR

    N, KH, KS, E, H, S = L["N"], L["KH"], L["KS"], L["E"], L["H"], L["S"]
    hot = L["hot"]
    full = {p: torch.zeros(H + S, N, KH, dtype=torch.int8, device=dev) for p in DI.PROJS}
    hs = {p: torch.stack([L["src_s"][(p, e)] for e in hot]).to(dev) for p in DI.PROJS}
    for s_, e in enumerate(hot):
        for p in DI.PROJS:
            full[p][s_].copy_(L["src_w"][(p, e)])
    C = DI.sf_words(KS)
    sf = {p: DI.packed_sf_zeros(H + S, N, C, dev) for p in DI.PROJS}
    dsts = [full[p] for p in DI.PROJS] + [sf[p] for p in DI.PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cpu").to(dev)
    ptab = torch.zeros(6, E, dtype=torch.int64, device="cpu")
    reg = {}
    for e in range(E):
        for i, p in enumerate(DI.PROJS):
            wt = full[p][hot.index(e)] if e in hot else L["host_w"][(p, e)]
            st_ = hs[p][hot.index(e)] if e in hot else L["host_s"][(p, e)]
            ptab[i, e], ptab[3 + i, e] = wt.data_ptr(), st_.data_ptr()
            reg[wt.data_ptr()] = wt.reshape(-1).view(torch.uint8)
            reg[st_.data_ptr()] = st_.reshape(-1).view(torch.uint8)
    gtab = torch.full((E,), -1, dtype=torch.int32, device="cpu")
    for s_, e in enumerate(hot):
        gtab[e] = s_
    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7]], device=dev)
    m = MR.dedup_routes(idx, torch.rand(3, 3, device=dev), 4)
    inpl = {"gtab": gtab.to(dev), "H": H, "S": S, "dummy": torch.tensor(6, device=dev), "has_dummy": True,
            "v2": True, "layout": "contiguous", "G": H + S, "gscr": 0, "step_sf": step_sf}
    ext = MR.rows_layout(m, inpl, 4, 3, 3)
    nw = 9 if step_sf else S
    args = (ptab.to(dev), m["uniq"], gtab.to(dev), dtab, dsts, ext["sidx"], ext["wl"], H, N * KH, ((N, C), (N, C)),
            nw)
    return dict(full=full, sf=sf, dsts=dsts, reg=reg, args=args, m=m, ext=ext, gtab=gtab, ptab=ptab, hs=hs)


# ------------------------------------------------------------------------------------------------ kernels
def test_s1_stage_proj_on_device_equals_contract():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime import staging_overlap as SO
    from runtime import static_kernels as SK

    n = 0
    for N, KH in ((2304, 2560), (40, 64)):           # real w1 row (EVEN path) and an odd size (masked path)
        L = _layer(N, KH)
        for step_sf in (False, True):
            for nprog in (48, 16, 8):
                for pj in range(3):
                    for weights, scales in ((True, True), (False, True), (True, False)):
                        outs = []
                        for use_kernel in (False, True):
                            D = _dsts(L, step_sf)
                            if use_kernel:
                                SO.stage_proj(*D["args"], pj, weights=weights, scales=scales, nprog=nprog)
                                torch.cuda.synchronize()
                            else:
                                SO.stage_proj_emulated(D["reg"].__getitem__, *D["args"], pj, weights, scales)
                            outs.append([t.clone() for t in D["dsts"]])
                        for i, (a_, b_) in enumerate(zip(*outs)):
                            assert torch.equal(a_, b_), (N, step_sf, nprog, pj, weights, scales, i)
                        n += 1
            # the three projections == m1_stage_rows
            D1, D2 = _dsts(L, step_sf), _dsts(L, step_sf)
            SK.stage_rows(*D1["args"])
            for pj in range(3):
                SO.stage_proj(*D2["args"], pj)
            torch.cuda.synchronize()
            for i, (a_, b_) in enumerate(zip(D1["dsts"], D2["dsts"])):
                assert torch.equal(a_, b_), (N, step_sf, "s1 x 3 vs m1", i)
            n += 1
    print(f"  s1_stage_proj: {n} device cases == contract / == m1_stage_rows")


def test_ce_path_on_device_equals_kernel():
    _need_dg()
    from runtime import dg_inplace as DI
    from runtime import staging_overlap as SO
    from runtime import static_kernels as SK

    for N, KH in ((2304, 2560), (40, 64)):
        L = _layer(N, KH, seed=3)
        for step_sf in (False, True):
            D1, D2 = _dsts(L, step_sf), _dsts(L, step_sf)
            SK.stage_rows(*D1["args"])
            a = D2["args"]
            t = {"gtab": a[2], "H": L["H"], "ptab6": a[0], "w": {p: D2["full"][p] for p in DI.PROJS}}
            plan = SO.ce_plan(t, {"wl": a[6], "uniq": a[1], "sidx": a[5]}, a[-1])
            assert [it["e"] for it in plan] == [2, 4, 7], plan
            for pj in range(3):
                SO.ce_copy_proj(t, plan, pj, a[8])
                SO.stage_proj(*a, pj, weights=False)
            torch.cuda.synchronize()
            for i, (x, y) in enumerate(zip(D1["dsts"], D2["dsts"])):
                assert torch.equal(x, y), (N, step_sf, "ce vs m1", i)
    print("  ce: cudaMemcpyAsync from pinned UVA + scale launch == m1_stage_rows (EVEN + masked sizes, both stores)")


# ------------------------------------------------------------------------------------------------ the 6-row step
def _snap(st, pool):
    from runtime import dg_inplace as DI

    slots = [pool.hbm_full[l][p][len(pool.hot[l]):].clone() for l in range(pool.n_layers) for p in DI.PROJS]
    sf = [st.dg.layers[l]["sf"][p][st.dg.layers[l]["H"]:].clone() for l in sorted(st.dg.layers) for p in DI.PROJS]
    return slots, sf


def _live_state(model, pool, st=None):
    """Every tensor a step writes that outlives the step: the model's buffers (window ring, compressor state,
    compressed KV, index K, engram cache, ...) + the tensors _state() reads, the pool's spare slots and -- given a
    step -- its packed-SF spare groups. Live tensors (not clones), deduplicated by address."""
    from runtime import dg_inplace as DI

    ts = [b for _, b in model.named_buffers()]
    for i, l in enumerate(model.layers):
        a = l.attn
        if a.compressor is not None and a.compressor.compress_ratio > 1:
            ts += [a.compressor.kv_state, a.compressor.score_state]
        if a.is_kv_source:
            ts.append(a.compress_kv_cache)
        if a.indexer is not None and a.indexer.owns_k:
            ts.append(a.indexer.k_cache)
    if getattr(model, "engram_hash", None) is not None:
        ts.append(model.engram_hash.cache)
    ts += [pool.hbm_full[l][p][len(pool.hot[l]):] for l in range(pool.n_layers) for p in DI.PROJS]
    if st is not None:
        ts += [st.dg.layers[l]["sf"][p][st.dg.layers[l]["H"]:] for l in sorted(st.dg.layers) for p in DI.PROJS]
    seen, out = set(), []
    for t in ts:
        k = (t.data_ptr(), t.numel(), t.dtype)
        if t.numel() and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _checkpoint(model, pool, st=None):
    """M3-S1-FIX: a copy of _live_state's values. Round 1's mini4 failure ('kernel-overlap', 'L1.ckv') was this
    file's own residue: _replay_ms ran the kernel graph 2n = 8 steps (positions 18..65) AFTER the n = 4-step base
    snapshot (18..41), so compressed-KV entries 21..32 of L1 (ratio 2) held replay values in every later run and
    the whole-buffer torch.equal failed -- for ANY mode, `kernel` included (tests/test_staging_overlap_cpu.py
    reproduces it with kernel vs kernel). Restoring the pre-base checkpoint before every compared run makes each
    run start from the same bytes; nothing is masked (the comparison is still the whole buffer)."""
    return [t.clone() for t in _live_state(model, pool, st)]


def _restore(model, pool, st, ck):
    """Copy a checkpoint's values back into this step's live tensors (same order: model buffers, the pool's spare
    slots, then the step's packed-SF spare groups -- a step built later gets the base step's build-time SF)."""
    live = _live_state(model, pool, st)
    assert len(live) == len(ck) and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(live, ck)), \
        "checkpoint layout differs from this step's live state"
    with torch.inference_mode():
        for t, c in zip(live, ck):
            t.copy_(c)
    if any(t.is_cuda for t in live):
        torch.cuda.synchronize()


def _state_diff(a, b):
    return sorted(k for k in b if not torch.equal(a[k], b[k]))


def _replay_ms(st, model, ids, toks, n):
    from test_static_rows_gpu import _prefill

    _prefill(model, ids)
    with torch.inference_mode():
        st.step(toks[:st.rows], len(ids), graph=True)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for s in range(1, n):
            st.step(toks[s * st.rows:(s + 1) * st.rows], len(ids) + s * st.rows, graph=True)
        e1.record()
        torch.cuda.synchronize()
    return e0.elapsed_time(e1) / max(1, n - 1)


GATE_ITERS = int(os.environ.get("DSV41_S1_GATE_ITERS", "20"))   # M3-S1-FIX: kernel-overlap runs per path (>= 20)


def _eq_all(a, b, sn, bsn, what):
    """torch.equal on every row's logits, main_hidden, the whole caches, the spare slots and the packed SF."""
    _assert_rows(a[0], b[0], what)
    assert all(torch.equal(x, y) for x, y in zip(a[1], b[1])), (what, "main_hidden")
    d = _state_diff(a[2], b[2])
    assert not d, (what, "caches", d)
    assert all(torch.equal(x, y) for x, y in zip(sn[0], bsn[0])), (what, "spare slots")
    assert all(torch.equal(x, y) for x, y in zip(sn[1], bsn[1])), (what, "packed SF")
    COUNTS["caches"] += len(b[2])


def test_static6_modes_equal_kernel_mini4():
    """Every mode == kernel on mini4 (6 rows, 4 of 8 experts cold per layer). M3-S1-FIX: each compared run starts
    from the pre-base checkpoint (_checkpoint docstring: round 1's L1.ckv failure was the replay timing loop's
    residue), kernel-overlap is compared GATE_ITERS times eager and GATE_ITERS times as a graph (a timing-dependent
    race would show as a flaky iteration), and a residue control re-runs kernel WITHOUT the restore after the
    replay to show the round-1 mechanism on this box (logits equal, L1.ckv differs)."""
    _need_dg()
    from runtime import staging_overlap as SO

    prev = os.environ.get(SO.STAGE_ENV)
    iters = max(1, GATE_ITERS)
    rec = {"iters": iters, "label": "[M] mini4 (tiny shapes): graph replay ms/step, a smoke number, not a "
                                    "performance claim"}
    RESULTS["mini4"] = rec                     # incremental: the receipt keeps what ran even if an assert fires
    try:
        with _model("mini4") as (model, pool):
            hid = (len(model.layers) - 1,)
            n_layers = len(model.layers)
            n = 4
            ids, toks = _toks(18, ROWS * n, 2026)
            kw = dict(rows=ROWS, hidden_layers=hid, moe_impl="dg-inplace", spare_slots=8)
            os.environ[SO.STAGE_ENV] = "kernel"
            st = _need_step(model, pool, **kw)
            assert st.stage_ov.mode == "kernel"
            ck = _checkpoint(model, pool, st)                     # before any step: the state every run starts from
            rec["checkpoint_tensors"] = len(ck)
            base = _static(model, st, ids, toks, n)
            base_snap = _snap(st, pool)
            st.capture()
            _restore(model, pool, st, ck)
            bg = _static(model, st, ids, toks, n, graph=True)
            _eq_all(bg, base, _snap(st, pool), base_snap, "kernel: graph vs eager")
            rec["kernel_graph_ms_per_step_mini"] = round(_replay_ms(st, model, ids, toks + toks, 2 * n), 4)
            # residue control: the round-1 sequence (no restore after the 2n-step replay), same mode as the base
            dirty = _static(model, st, ids, toks, n)
            _assert_rows(dirty[0], base[0], "residue control: kernel after the replay, logits")
            rec["residue_control_cache_diff"] = _state_diff(dirty[2], base[2])
            assert "L1.ckv" in rec["residue_control_cache_diff"], (
                "the round-1 residue did not reproduce with kernel vs kernel: the L1.ckv diagnosis is wrong here",
                rec["residue_control_cache_diff"])
            st.detach()
            # M3-S1-FIX: + the per-projection launch shape (fast w1 copy, low-contention w3 / w2), same bytes
            runs = [("kernel-overlap", None, None, iters), ("kernel-overlap", "48,16,16", "8192,32768,32768",
                                                            max(2, iters // 4)), ("ce", None, None, 2),
                    ("ce-overlap", None, None, 2)]
            for mode, np_, bb, n_it in runs:
                os.environ[SO.STAGE_ENV] = mode
                for k_, v_ in ((SO.NPROG_ENV, np_), (SO.BLOCK_ENV, bb)):
                    if v_ is None:
                        os.environ.pop(k_, None)
                    else:
                        os.environ[k_] = v_
                tag = mode if np_ is None else f"{mode}/np{np_.replace(',', '-')}"
                st = _need_step(model, pool, **kw)
                assert st.stage_ov.mode == mode and st.describe()["stage"]["mode"] == mode
                n_eager = n_it
                stats0 = None
                for it in range(n_eager):
                    _restore(model, pool, st, ck)
                    a = _static(model, st, ids, toks, n)
                    _eq_all(a, base, _snap(st, pool), base_snap, f"{tag} vs kernel (eager #{it})")
                    if stats0 is None:
                        stats0 = dict(st.stage_ov.stats)
                assert stats0["launches"] == n_layers * n, stats0
                if mode.startswith("ce"):
                    assert stats0["ce_copies"] > 0, "no cold expert was staged: the ce path was not exercised"
                    try:
                        st.capture()
                        raise AssertionError(f"{mode} captured without StageCaptureError")
                    except SO.StageCaptureError:
                        pass
                    except RuntimeError as e:          # torch may wrap the error raised inside capture
                        assert "eager-only" in str(e), e
                else:
                    assert st.stage_ov.side is not None
                    st.capture()
                    for it in range(n_it):
                        _restore(model, pool, st, ck)
                        ag = _static(model, st, ids, toks, n, graph=True)
                        _eq_all(ag, base, _snap(st, pool), base_snap, f"{tag}: graph vs kernel eager (#{it})")
                        _assert_rows(ag[0], bg[0], f"{tag}: graph vs kernel graph (#{it})")
                    rec[f"{tag}_graph_ms_per_step_mini"] = round(_replay_ms(st, model, ids, toks + toks, 2 * n), 4)
                ov = st.overflow_counts()
                assert ov["steps"] == 0 and ov["experts"] == 0, ov
                rec[tag] = {"stats_first_eager_run": stats0, "eager_runs_equal": n_eager,
                            "graph_runs_equal": 0 if mode.startswith("ce") else n_it,
                            "nprog": st.describe()["stage"]["nprog"],
                            "block_bytes": st.describe()["stage"]["block_bytes"]}
                st.detach()
                COUNTS["gates"] += 1
    finally:
        if prev is None:
            os.environ.pop(SO.STAGE_ENV, None)
        else:
            os.environ[SO.STAGE_ENV] = prev
        os.environ.pop(SO.NPROG_ENV, None)
        os.environ.pop(SO.BLOCK_ENV, None)
    print("  mini4:", json.dumps(rec, default=str))


# ------------------------------------------------------------------------------------------------ the probe
def test_probe_quick():
    _need_dg()
    import staging_overlap_bench as B

    out = os.path.join(REPO, "scratch", f"staging-overlap-probe-quick-{time.strftime('%Y%m%d')}.json")
    rc = B.main(["probe", "--rounds", "4", "--reps", "10", "--hot", "48", "--out", out])
    rep = json.load(open(out))["probe"]
    assert not rep["errors"], rep["errors"]
    ch = rep["checks"]
    assert ch["serial_staged_equal_host"], ch
    for k_, v in ch.items():
        if isinstance(v, dict):
            assert all(v.values()), (k_, v)
    RESULTS["probe_quick"] = {"bandwidth_c4": rep["bandwidth"].get("4") or rep["bandwidth"].get(4),
                              "overlap_c2": rep["overlap"].get("2") or rep["overlap"].get(2),
                              "model_measured": rep.get("model_measured")}
    print("  probe (quick):", json.dumps(RESULTS["probe_quick"], default=str)[:1500])
    assert rc == 0


TESTS = [test_s1_stage_proj_on_device_equals_contract, test_ce_path_on_device_equals_kernel,
         test_static6_modes_equal_kernel_mini4, test_probe_quick]

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
    os.makedirs(os.path.join(REPO, "scratch"), exist_ok=True)
    with open(os.path.join(REPO, "scratch", f"staging-overlap-gpu-test-{time.strftime('%Y%m%d')}.json"), "w") as f:
        json.dump({"results": results, "data": RESULTS}, f, indent=1, default=str)
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
