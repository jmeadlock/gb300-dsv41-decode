"""GPU tests of A2-2 (runtime/residency.py) on the synthetic mini checkpoints, CUDA graphs on. Skipped without
CUDA/tilelang (Mac); the Mac half is tests/test_residency_cpu.py + tests/test_residency_rows_cpu.py.

  1. residency on vs off, 64 graph-replayed decode steps (mini4: 4 layers, 8 experts, 4 hot, topk 4): policy swaps
     driven by synthetic counters + forced swaps of experts the NEXT step routes (from the off run's ids) mid-run ->
     fp32 logits torch.equal at every step and the prefill; tokens equal; audit (tables, params, sha256 of every
     moved expert vs the shards) ok; residency off again afterwards (restore_boot) still equal. moe/glue impls:
     ref/ref, ref/fused (the glue's stacked _ptab6), dg-inplace/ref (gtab + in-place GEMM over residency slots).
  2. event ordering: forced swaps of next-step-routed experts EVERY step, n_spare 1 (every swap reuses a slot freed by
     the previous one -> the free event is what protects it), every swap's copies queued behind ~1 GiB of dummy H2D on
     the copy stream, the host never syncs (device-fed tokens, device-side logit clones), so publishes race queued
     replays -> still bit-equal. Negative control (gated): the "publish_before_copy" mutation on the same schedule
     MUST diverge -- otherwise this test could not see an ordering bug.
  3. after_step makes no host sync (torch.cuda.set_sync_debug_mode("error")) while swapping.
  4. (report only) graph step time with residency on vs off on the mini model.

Run on the box (orchestrator), from the repo checkout, inside the library image (same line as the other GPU tests):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_residency_gpu.py
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from test_static_mini_gpu import MAX_POS, _ids, _model, _need_tilelang  # noqa: E402

N_STEPS = 64
COMBOS = (("ref", "ref"), ("ref", "fused"), ("dg-inplace", "ref"))


def _cfg(**kw):
    from runtime.residency import ResidencyConfig

    base = dict(drain_n=8, min_abs=1.0, hyst=1.0, halflife=100.0, n_spare=2, n_demote=4, reserve_gib=0.0)
    base.update(kw)
    return ResidencyConfig(**base)


def _gen(model, tokdir, moe_impl="ref", glue_impl="ref", residency=None):
    from runtime.generate import Generator, set_determinism

    set_determinism(0)
    return Generator(model, tokdir, max_pos=MAX_POS, mode="graph", moe_impl=moe_impl, glue_impl=glue_impl,
                     residency=residency)


def _need_impl(moe_impl):
    if moe_impl == "dg-inplace":
        try:
            import deep_gemm  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"deep_gemm unavailable: {e}")


def _steps(gen, ids, n, before=None):
    """prefill + n graph steps fed from the device (no host sync in the loop); device clones of every step's fp32
    logits and routed ids; one sync at the end."""
    gen.static_step()  # built (and the residency manager attached) before the first `before` hook
    out = [gen.prefill(ids).float().clone()]
    routed = []
    pos = len(ids)
    for k in range(n):
        if before is not None:
            before(k)
        lg = gen.step(gen.next_token_tensor(), pos)
        out.append(lg.float().clone())
        routed.append(gen.static.ids_step.clone())
        pos += 1
    torch.cuda.synchronize()
    return out, [r.cpu() for r in routed]


def _assert_equal(a, b, what):
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)]
    assert len(a) == len(b) and not bad, (f"{what}: logits differ at positions {bad[:8]} (0 = prefill); max |d| "
                                          f"{(a[bad[0]] - b[bad[0]]).abs().max().item() if bad else 0}")


def _routed_swaps(rm, routed_next, demote_routed=False):
    """One swap per layer: a mirror-cold expert the next step routes in, a mirror-hot one out (not routed next
    step, or routed when demote_routed)."""
    out = []
    for l in range(rm.L):
        ids = set(routed_next[l].tolist())
        cin = [e for e in rm.state.cold[l] if e in ids]
        hout = [e for e in rm.state.hot[l] if (e in ids) == demote_routed] or list(rm.state.hot[l])
        if cin and hout:
            out.append((l, cin[0], hout[0]))
    return out


def _hot_cold_bias(rm, rng, scale=12):
    d = np.zeros((rm.L, rm.E), dtype=np.int64)
    for l in range(rm.L):
        d[l, rm.state.cold[l]] = rng.integers(0, scale, len(rm.state.cold[l]))
    return d


def _combo(moe_impl, glue_impl):
    _need_tilelang()
    _need_impl(moe_impl)
    from runtime.residency import golden_from_index

    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, moe_impl, glue_impl)
        ids = _ids(16, 21)
        off, routed = _steps(gen, ids, N_STEPS)
        gen.set_residency(_cfg())
        rng = np.random.default_rng(4)

        def before(k):
            rm = gen.residency_mgr
            rm.inject_counts(_hot_cold_bias(rm, rng))
            if 20 <= k < 40 and k + 1 < N_STEPS:  # forced, mid-run: experts the next step routes come in / go out
                rm.force_swaps(_routed_swaps(rm, routed[k + 1], demote_routed=k % 2 == 1))

        on, _ = _steps(gen, ids, N_STEPS, before)
        rm = gen.residency_mgr
        _assert_equal(on, off, f"{moe_impl}/{glue_impl}: residency on vs off")
        s = rm.stats
        assert s["published"] > 20 and s["forced_swaps"] > 10 and rm.state.n_swaps > s["forced_swaps"], s
        kinds = sorted({t.kind for v in rm.tabs.values() for t in v})
        rm.flush()
        a = rm.audit(golden=golden_from_index(model.runtime_load.index))
        assert a["ok"], a["errors"][:6]
        d = rm.describe()
        gen.set_residency(None)  # restore_boot + close; the next run rebuilds and recaptures on the boot map
        assert model.runtime_load.pool.residency is None
        again, _ = _steps(gen, ids, N_STEPS)
        _assert_equal(again, off, f"{moe_impl}/{glue_impl}: off after restore_boot vs off")
        print(f"  {moe_impl}/{glue_impl}: on == off over prefill + {N_STEPS} graph steps; tables {kinds}; "
              f"{s['published']} swaps published ({s['forced_swaps']} forced), {s['d2h_demotions']} D2H demotions, "
              f"audit {a['experts_checked']} experts ok; swaps/step {d['swaps_per_step']}")
        gen.reset_static()
        del gen


def test_residency_on_equals_off_ref():
    _combo("ref", "ref")


def test_residency_on_equals_off_fused_glue():
    _combo("ref", "fused")


def test_residency_on_equals_off_dg_inplace():
    _combo("dg-inplace", "ref")


def _event_ordering_run(gen, ids, routed, mutation=None):
    gen.set_residency(_cfg(n_spare=1, frozen=True, drain_n=1 << 20))
    gen.static_step()
    rm = gen.residency_mgr
    rm._mutation = mutation
    big_src = torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True)
    big_dst = torch.empty(256 << 20, dtype=torch.uint8, device=rm.dev)
    orig = rm._copies

    def slow(sw, ev):  # ~1 GiB of H2D queued on the copy stream ahead of every swap's own copies
        with torch.cuda.stream(rm.copy_stream):
            for _ in range(4):
                big_dst.copy_(big_src, non_blocking=True)
        orig(sw, ev)

    rm._copies = slow

    def before(k):
        if k + 1 < len(routed):
            rm.force_swaps(_routed_swaps(rm, routed[k + 1], demote_routed=k % 3 == 0))

    out, _ = _steps(gen, ids, len(routed), before)
    stats = dict(rm.stats)
    rm._mutation = None
    rm._copies = orig
    gen.set_residency(None)
    return out, stats


def test_residency_swaps_while_replays_queued_event_ordering():
    _need_tilelang()
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir)
        ids = _ids(16, 23)
        off, routed = _steps(gen, ids, 32)
        on, stats = _event_ordering_run(gen, ids, routed)
        _assert_equal(on, off, "swaps racing queued replays (n_spare 1, slow copies)")
        assert stats["published"] > len(routed) and stats["forced_flush"] > 0, stats
        bad, bstats = _event_ordering_run(gen, ids, routed, mutation="publish_before_copy")
        n_bad = sum(not torch.equal(x, y) for x, y in zip(bad, off))
        assert n_bad > 0, "negative control: publish-before-copy went unseen -> this test cannot see ordering bugs"
        print(f"  event ordering: {stats['published']} swaps (n_spare 1, ~1 GiB dummy H2D ahead of each) racing "
              f"queued replays -> bit-equal; {stats['forced_flush']} forced flushes; negative control diverged at "
              f"{n_bad}/{len(off)} positions")
        gen.reset_static()
        del gen


def test_residency_after_step_has_no_host_sync():
    _need_tilelang()
    with _model("mini4") as (model, tokdir):
        gen = _gen(model, tokdir, residency=_cfg(drain_n=64, force_random=2))
        ids = _ids(16, 24)
        gen.prefill(ids)
        pos = len(ids)
        for _ in range(3):  # warm: JIT, first copies, views
            gen.step(gen.next_token_tensor(), pos)
            pos += 1
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(16):
                gen.step(gen.next_token_tensor(), pos)
                pos += 1
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        assert gen.residency_mgr.stats["published"] > 0
        gen.reset_static()
        del gen


def test_residency_step_time_report():
    """Report only: CUDA-event ms per graph step, residency off vs on (policy + 1 forced swap per step)."""
    _need_tilelang()
    with _model("mini4") as (model, tokdir):
        res = {}
        for name, cfg in (("off", None), ("on", _cfg(force_random=1))):
            gen = _gen(model, tokdir, residency=cfg)
            ids = _ids(16, 25)
            gen.prefill(ids)
            pos = len(ids)
            evs = []
            for _ in range(N_STEPS):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                gen.step(gen.next_token_tensor(), pos)
                e.record()
                evs.append((s, e))
                pos += 1
            torch.cuda.synchronize()
            ms = sorted(s.elapsed_time(e) for s, e in evs[8:])
            res[name] = ms[len(ms) // 2]
            gen.reset_static()
            del gen
        print(f"  mini4 graph step p50: off {res['off']:.3f} ms, on {res['on']:.3f} ms "
              f"({(res['on'] - res['off']) / res['off'] * 100:+.1f} %; mini-model numbers, not the real step)")


TESTS = [test_residency_on_equals_off_ref, test_residency_on_equals_off_fused_glue,
         test_residency_on_equals_off_dg_inplace, test_residency_swaps_while_replays_queued_event_ordering,
         test_residency_after_step_has_no_host_sync, test_residency_step_time_report]

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
