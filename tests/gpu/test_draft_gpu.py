"""GPU tests of M1-T11 (runtime/draft_step.py DraftStep, runtime/mtp_loader.py). Skipped without CUDA/tilelang (Mac);
tests/test_draft_cpu.py dry-runs this file's gate functions on the CPU kernel emulations (DEV "cpu").

V2 (docs/M1-DSPARK-DESIGN.md section 4): the static draft step run eagerly == the reference `forward_spec` run eagerly
(runtime/draft_step.py `reference_draft`: forward_spec decode calls, one context row each, under
refmodel.reference_device), torch.equal on the drafts out_ids [1, 6], the fp32 logits [5, V] (Markov bias included),
the confidences [5] and all three stages' 128-slot rings, for:
  * the first draft after a prefill: DraftStep ctx_count = 0 at A' = L-1 (design 2.3; no rewrite of the last prompt
    slot) and ctx_count = 1 with the prefill's last main_hidden row (= the reference decode call exactly). The
    reference call re-writes slot (L-1) % 128 at decode numerics: the test reports whether that rewrite is a byte
    no-op (design risk 7); ctx_count = 0 is gated equal to the reference only when it is;
  * a mid-sequence draft after a 6-row verify (main_hidden from the reference 6-row forward, runtime/ref_multirow.py
    `forward_rows`, the M1-T10 oracle), ctx_count in {1, 3, 6};
  * two chained drafts (ctx 3, then ctx 6 after a second verify): the rings accumulate as the reference's do;
  * a prompt longer than the 128-ring (the prefill's wrap branch, model.py:1049-1053; drafts at A' >= 128);
  * `ctx_rowwise` (M = 1 context GEMMs) == the default (M = 6): reported, gated only on CPU.
The temperature fix on both sides: runtime/mtp_loader.py sets every stage's `temperature = 0` (DSparkBlock captures
args.temperature at model.py:1113 and samples with it at 1156 via `sample`, 1288-1295); DraftStep refuses any other
value and calls the same `forward_head` (1140-1159), whose `sample(., 0)` is `argmax(-1)` (1291-1292).

Graph: graph == static-eager (ctx 0/1/3/6), replay twice, fills-on recapture, relaunch (fresh DraftStep + capture),
no host sync in the eager step. Real checkpoint: set DSV41_MODEL_DIR (and DSV41_MAP, default the seed map) and the
V2 gate also runs on the real model (`real`).

M1-T11-PORT (runtime/draft_step.py impl switches), per switch the verify's own standard:
  * P1 exact: fp8_impl "gemv" + glue_impl "fused" (moe / attn ref) == the reference forward_spec run INSIDE the same
    fp8_gemv_scope (the GEMV patch routes the reference Linears too: same act_quant, same GEMV), torch.equal on
    drafts, logits, confidences, rings -- the V2 cases above; the fused clusters must all self-check "on".
  * P2 DG MoE: each stage's DraftDGMoE (DeepGEMM contiguous over the resident stack) vs the reference eager MoE
    (model.py:891-906, the stage's own module) on the same x: bf16 ulp statistics REPORTED (the documented 1-ulp
    class of dg_moe.py); gated: finite, the same routed experts, max ulp <= DG_MAX_ULP (a gross-error guard, not a
    measured number). The full-impl draft vs the reference: drafts agreement and max |logit delta| REPORTED.
  * P3 graph: all four switches on (gemv, dg-inplace, fused, flashmla): graph == static-eager, replay x2, relaunch,
    no host sync, `describe()` printed (which clusters / attention arm are actually on).

Run on the box (orchestrator), same line as tests/gpu/test_static_rows_gpu.py, repo at $W:

  W=$HOME/hermes/gb300-dsv41-decode-wt11
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv [-v /models:/models -e DSV41_MODEL_DIR=/models/DeepSeek-V4.1-...] \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_draft_gpu.py
"""
from __future__ import annotations

import contextlib
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

from runtime.refmodel import tilelang_available  # noqa: E402

DEV = "cuda"       # tests/test_draft_cpu.py dry-runs the gates on the CPU emulations (DEV "cpu")
STEP_KW = {}       # ... with {"_allow_cpu": True}
COUNTS = {"gates": 0, "drafts": 0, "logit_rows": 0, "rings": 0, "graph": 0}
REPORT = []        # lines printed at the end (rewrite no-op check, rowwise check)
DG_MAX_ULP = 8     # P2 gross-error guard on the DG MoE vs the reference eager MoE (reported, see module docstring)
FULL = {"fp8_impl": "gemv", "moe_impl": "dg-inplace", "glue_impl": "fused", "attn_impl": "flashmla"}


def _need_tilelang():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def _model(kind: str = "mini3"):
    """Build + load a mini checkpoint with the mtp schema (tests/mini_mtp.py) on cuda, load_mtp=True -> model; or
    `real` = $DSV41_MODEL_DIR."""
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    if kind == "real":
        md = os.environ.get("DSV41_MODEL_DIR")
        if not md:
            raise unittest.SkipTest("DSV41_MODEL_DIR not set")
        mp = os.environ.get("DSV41_MAP") or os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json")
        model = build_and_load(md, mp if os.path.exists(mp) else None, "cuda", load_mtp=True)
        model.temperature = 0.0
        res = model.runtime_load
        try:
            print(f"  real: mtp {model.runtime_mtp.summary()['bytes_GiB']}; HBM free "
                  f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
            yield model
        finally:
            torch.set_default_device("cpu")
            del model
            torch.cuda.synchronize()
            res.release()
        return
    import engram
    from mini_ckpt import MINI_CONFIG
    from mini_mtp import make_mini_checkpoint_mtp, mtp_config_kind

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    with tempfile.TemporaryDirectory() as tmp:
        ck = make_mini_checkpoint_mtp(os.path.join(tmp, "ckpt"), cfg=mtp_config_kind(kind))
        model = build_and_load(ck["model_dir"], ck["map"], "cuda", tokenizer=object(), io_threads=4,
                               stage_bytes=1 << 20, load_mtp=True, log=lambda *a: None)
        model.temperature = 0.0
        res = model.runtime_load
        try:
            yield model
        finally:
            torch.set_default_device("cpu")
            del model
            torch.cuda.synchronize()
            res.release()


def _toks(n, seed, hi=2000):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, hi, (n,), generator=g, device="cpu").tolist()


def _hidden_layers(model):
    return tuple(model.runtime_mtp.target_layer_ids)


def _prefill(model, ids):
    """Reference prefill with main_hidden + the reference draft-ring seeding -> (first token [1], main_hidden)."""
    from runtime.draft_step import prefill_hidden, seed_draft_rings

    out, _, mh = prefill_hidden(model, torch.tensor([ids], device=DEV), _hidden_layers(model))
    first = out.reshape(-1)[:1].clone()
    seed_draft_rings(model, first, mh)
    _sync()
    return first, mh


def _verify6(model, toks, p0):
    """The reference 6-row verify forward (M1-T10 oracle) -> (t [6], main_hidden [6, H])."""
    from runtime.ref_multirow import forward_rows

    ids, _, mh = forward_rows(model, torch.tensor([toks], device=DEV), p0, _hidden_layers(model))
    return ids.reshape(-1).clone(), mh.clone()


def _ref_run(model, a, mh_rows, toks, gemv=False):
    from runtime.draft_step import reference_draft, ring_snapshot
    from runtime.patches import fp8_gemv_scope

    with fp8_gemv_scope(gemv):     # P1: the reference Linears on the GEMV too (DraftStep(fp8_impl="gemv") installed it)
        out, lg, cf = reference_draft(model, a, mh_rows, toks)
    _sync()
    return out.reshape(-1).clone(), lg.reshape(lg.size(1), -1).float().clone(), cf.reshape(-1).clone(), \
        ring_snapshot(model)


def _ds_run(model, ds, a, c, mh=None, tok=None, graph=False):
    from runtime.draft_step import ring_snapshot

    ds.step(a, c, graph=graph, main_hidden=mh, tok=tok)
    _sync()
    return ds.out_ids.reshape(-1).clone(), ds.logits.clone(), ds.confidence.clone(), ring_snapshot(model)


def _eq(a, b, what, rings=True):
    """a, b = (out_ids, logits, confidence, rings) -> torch.equal on each (rings optional)."""
    assert torch.equal(a[0], b[0]), f"{what}: drafts {a[0].tolist()} vs {b[0].tolist()}"
    bad = [i for i in range(a[1].size(0)) if not torch.equal(a[1][i], b[1][i])]
    assert not bad, f"{what}: logit rows {bad} differ, max |d| {(a[1] - b[1]).abs().max().item()}"
    assert torch.equal(a[2], b[2]), f"{what}: confidence {a[2].tolist()} vs {b[2].tolist()}"
    COUNTS["drafts"] += 1
    COUNTS["logit_rows"] += a[1].size(0)
    if rings:
        for s, (x, y) in enumerate(zip(a[3], b[3])):
            assert torch.equal(x, y), f"{what}: stage {s} ring differs at slots " \
                                      f"{(x != y).flatten(2).any(-1).nonzero()[:, 1].tolist()[:16]}"
        COUNTS["rings"] += len(a[3])


def _rings_equal(x, y):
    return all(torch.equal(p, q) for p, q in zip(x, y))


# ------------------------------------------------------------------------------------------------ the V2 gate
def _gate_draft(kind="mini3", L=17, cs=(1, 3, 6), chain=True, rowwise=True, impls=None):
    """V2 (impls None) or P1 (impls: GEMV / fused glue, both bit-identical given the GEMV scope on the oracle)."""
    from runtime.draft_step import DraftStep, ring_restore, ring_snapshot

    impls = dict(impls or {})
    if impls.get("moe_impl", "ref") != "ref" or impls.get("attn_impl", "ref") != "ref":
        raise ValueError("_gate_draft is the exact gate: DG MoE / FlashMLA are not bit-identical (P2 / P3)")
    gv = impls.get("fp8_impl") == "gemv"
    with _model(kind) as model:
        V = model.head.weight.size(0)
        hi = min(2000, V)
        ids = _toks(L, 100 + L, hi)
        first, mh_pre = _prefill(model, ids)
        snap0 = ring_snapshot(model)
        ds = DraftStep(model, **STEP_KW, **impls)
        tag = f"{kind} L={L}{' ' + str(impls) if impls else ''}"
        if impls.get("glue_impl") == "fused":
            st = ds.glue.status
            off = {c: s for c, s in st.items() if not (s == "on" or s.startswith("n/a"))}
            assert not off, f"{tag}: fused draft clusters failed their self-check: {off}"
            REPORT.append(f"{tag}: draft glue {ds.glue.describe()['on']} on (hc_mode {ds.glue.hc_mode}, rope_inv "
                          f"{ds.glue.rope_inv})")
        # -- first draft after the prefill
        ref1 = _ref_run(model, L - 1, mh_pre[0, L - 1:L], [first], gv)
        noop = _rings_equal(ref1[3], snap0)
        ring_restore(model, snap0)
        got1 = _ds_run(model, ds, L - 1, 1, mh=mh_pre[0, L - 1:L], tok=first)
        _eq(got1, ref1, f"{tag} first draft, ctx 1 (the reference decode call)")
        ring_restore(model, snap0)
        got0 = _ds_run(model, ds, L - 1, 0, tok=first)
        assert _rings_equal(got0[3], snap0), "ctx_count 0 wrote a ring"
        REPORT.append(f"{tag}: reference re-write of slot (L-1)%128 at decode numerics is "
                      f"{'a byte no-op' if noop else 'NOT a byte no-op (design risk 7)'}")
        if noop:
            _eq(got0, ref1, f"{tag} first draft, ctx 0 (no rewrite)")
        else:  # not gated: ctx 0 is the design's path (prefill-numerics slot), the reference re-writes it
            same = torch.equal(got0[0], ref1[0]) and torch.equal(got0[1], ref1[1])
            REPORT.append(f"{tag}: ctx 0 drafts {got0[0].tolist()} vs reference {ref1[0].tolist()}; "
                          f"logits equal {same}")
        # -- mid-sequence: a 6-row verify at P0 = L, then ctx_count rows
        ring_restore(model, snap0)
        t6, mh6 = _verify6(model, [int(first)] + _toks(5, 7 + L, hi), L)
        for c in cs:
            a = L + c - 1
            ring_restore(model, snap0)
            ref = _ref_run(model, a, mh6[:c], t6[:c], gv)
            ring_restore(model, snap0)
            got = _ds_run(model, ds, a, c, mh=mh6, tok=t6)
            _eq(got, ref, f"{tag} ctx {c} at A'={a}")
            if rowwise and c == max(cs):
                ring_restore(model, snap0)
                dr = DraftStep(model, ctx_rowwise=True, **STEP_KW, **impls)
                gr = _ds_run(model, dr, a, c, mh=mh6, tok=t6)
                same = (torch.equal(gr[0], got[0]) and torch.equal(gr[1], got[1]) and torch.equal(gr[2], got[2])
                        and _rings_equal(gr[3], got[3]))
                REPORT.append(f"{tag} ctx {c}: ctx_rowwise (M=1 context GEMMs) == default (M=6): {same}")
                if DEV == "cpu":
                    assert same, "rowwise context write differs on the row-invariant emulation"
        # -- chained: ctx 3 at A'=L+2, a second verify at P0'=L+3, ctx 6 at A'=L+8
        if chain:
            ring_restore(model, snap0)
            t6b, mh6b = _verify6(model, [int(t6[2])] + _toks(5, 9 + L, hi), L + 3)
            ring_restore(model, snap0)
            _ref_run(model, L + 2, mh6[:3], t6[:3], gv)
            refc = _ref_run(model, L + 8, mh6b, t6b, gv)
            ring_restore(model, snap0)
            _ds_run(model, ds, L + 2, 3, mh=mh6, tok=t6)
            gotc = _ds_run(model, ds, L + 8, 6, mh=mh6b, tok=t6b)
            _eq(gotc, refc, f"{tag} chained ctx 3 -> ctx 6")
        COUNTS["gates"] += 1
        print(f"  {tag}: DraftStep static-eager == reference forward_spec{' (GEMV scope)' if gv else ''} (first "
              f"draft ctx 1{'/0' if noop else ''}, mid ctx {list(cs)}{', chained' if chain else ''}): drafts, logits, "
              f"confidence, rings")
        return ds


# ------------------------------------------------------------------------------------------------ M1-T11-PORT
def _gate_dg_moe(kind="mini3", n_x=3):
    """P2: DraftDGMoE (DeepGEMM contiguous over the resident stack) vs the stage's reference eager MoE on the same x."""
    from runtime.dg_inplace import compare_bf16
    from runtime.draft_step import DraftStep
    from runtime.refmodel import reference_device

    with _model(kind) as model:
        ds = DraftStep(model, moe_impl="dg-inplace", **STEP_KW)
        D = ds.stages[0].attn_norm.dim
        g = torch.Generator(device="cpu").manual_seed(11)
        worst, lines = 0, []
        for s, (st, moe) in enumerate(zip(ds.stages, ds.moes)):
            for j in range(n_x):
                x = (torch.randn(1, ds.block, D, generator=g, device="cpu") * (0.5 + j)).to(torch.bfloat16).to(DEV)
                with torch.inference_mode():
                    got = moe(x).clone()
                    ids_dg = sorted(int(e) for e in moe.last["uniq"][moe.last["counts"] > 0].tolist())
                    with reference_device(x.device):
                        want = st.ffn(x)                                  # model.py:891-906, eager reference
                    _, ind = st.ffn.gate(x.view(-1, D), None)
                    ids_ref = sorted({int(e) for e in ind.reshape(-1).tolist()})
                _sync()
                assert ids_dg == ids_ref, f"{kind} stage {s}: DG routed {ids_dg} vs reference {ids_ref}"
                assert bool(torch.isfinite(got.float()).all()), f"{kind} stage {s}: non-finite DG MoE output"
                c = compare_bf16(got.reshape(-1), want.reshape(-1))
                worst = max(worst, c["max_ulp"])
                lines.append(f"s{s}x{j}: equal {c['equal']}, mismatch {100 * c['mismatch_frac']:.3f}%, max ulp "
                             f"{c['max_ulp']}, max |d| {c['max_abs']:.3g}")
                assert c["max_ulp"] <= DG_MAX_ULP, f"{kind} stage {s} x{j}: DG MoE vs reference {c}"
        REPORT.append(f"{kind}: draft DG MoE vs reference eager MoE (bf16): " + "; ".join(lines))
        COUNTS["gates"] += 1
        print(f"  {kind}: draft DG MoE == reference eager MoE within the 1-ulp class (max ulp {worst}; "
              f"{ds.dgm.describe()['bytes']})")


def _gate_full_graph(kind="mini3", L=17, impls=None):
    """P3: all switches: graph == static-eager, replay x2, relaunch, no host sync; drafts vs the reference reported."""
    from runtime.draft_step import DraftStep, ring_restore, ring_snapshot

    impls = dict(FULL if impls is None else impls)
    with _model(kind) as model:
        ids = _toks(L, 5)
        first, mh_pre = _prefill(model, ids)
        t6, mh6 = _verify6(model, [int(first)] + _toks(5, 6), L)
        snap0 = ring_snapshot(model)
        ds = DraftStep(model, **STEP_KW, **impls)
        d = ds.describe()
        REPORT.append(f"{kind} full impls {impls}: glue on {d['glue'].get('on')}, attn {d['attn']}, moe "
                      f"{d['moe'].get('impl')} {d['moe'].get('bytes')}")
        cases = [(L - 1, 0), (L, 1), (L + 2, 3), (L + 5, 6)]

        def run(step, graph):
            out = []
            for a, c in cases:
                ring_restore(model, snap0)
                out.append(_ds_run(model, step, a, c, mh=mh6, tok=t6 if c else first, graph=graph))
            return out

        e = run(ds, False)
        for x in e:
            assert bool(torch.isfinite(x[1]).all()), f"{kind}: non-finite draft logits ({impls})"
        agree, dmax = [], 0.0
        for (a, c), x in zip(cases, e):
            if c:
                ring_restore(model, snap0)
                r = _ref_run(model, a, mh6[:c], t6[:c], impls.get("fp8_impl") == "gemv")
                agree.append(int(torch.equal(x[0], r[0])))
                dmax = max(dmax, (x[1] - r[1]).abs().max().item())
        REPORT.append(f"{kind} full impls: drafts == reference forward_spec (GEMV scope) in {sum(agree)}/{len(agree)} "
                      f"cases; max |logit delta| {dmax:.4g} (DG / FlashMLA classes, reported)")
        if DEV != "cuda":
            COUNTS["gates"] += 1
            return
        ds.capture()
        g1 = run(ds, True)
        for (a, c), x, y in zip(cases, g1, e):
            _eq(x, y, f"{kind} full: graph vs static-eager A'={a} ctx {c}")
        g2 = run(ds, True)
        for x, y in zip(g2, g1):
            _eq(x, y, f"{kind} full: graph replay #2 vs #1")
        ds2 = DraftStep(model, **STEP_KW, **impls)
        ds2.capture()
        g3 = run(ds2, True)
        for x, y in zip(g3, g1):
            _eq(x, y, f"{kind} full: relaunch vs replay #1")
        ring_restore(model, snap0)
        with torch.inference_mode():
            ds.step(L + 5, 6, main_hidden=mh6, tok=t6)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                ds.forward_static()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        COUNTS["graph"] += 1
        COUNTS["gates"] += 1
        print(f"  {kind} full impls {impls}: graph == static-eager (ctx 0/1/3/6); replay x2, relaunch equal; no host "
              f"sync; draft buffers {ds.staging_bytes() / 2**20:.1f} MiB")


def test_draft_port_exact_mini3():
    """P1: GEMV + fused glue == forward_spec under the GEMV scope, bit for bit (V2's cases)."""
    _need_tilelang()
    _gate_draft("mini3", 17, impls={"fp8_impl": "gemv", "glue_impl": "fused"})
    _gate_draft("mini3w", 18, chain=False, rowwise=False, impls={"glue_impl": "fused"})
    _gate_draft("mini3", 150, cs=(6,), chain=True, rowwise=False, impls={"fp8_impl": "gemv", "glue_impl": "fused"})


def test_draft_port_dg_moe_mini3():
    _need_tilelang()
    _gate_dg_moe("mini3")
    _gate_dg_moe("mini3w")


def test_draft_port_full_graph_mini3():
    _need_tilelang()
    _gate_full_graph("mini3", 17)


def test_draft_port_real():
    """P1 / P2 / P3 on the real checkpoint (DSV41_MODEL_DIR)."""
    _need_tilelang()
    if not os.environ.get("DSV41_MODEL_DIR"):
        raise unittest.SkipTest("DSV41_MODEL_DIR not set")
    _gate_draft("real", 40, cs=(1, 6), chain=False, rowwise=False, impls={"fp8_impl": "gemv", "glue_impl": "fused"})
    _gate_dg_moe("real", n_x=2)
    _gate_full_graph("real", 40)


def test_draft_equals_reference_mini():
    _need_tilelang()
    _gate_draft("mini", 17)


def test_draft_equals_reference_mini3():
    _need_tilelang()
    _gate_draft("mini3", 17)
    _gate_draft("mini3w", 18, chain=False, rowwise=False)


def test_draft_equals_reference_ring_wrapped():
    _need_tilelang()
    _gate_draft("mini3", 150, cs=(6,), chain=True, rowwise=False)


def test_draft_equals_reference_real():
    """The V2 gate on the real checkpoint (DSV41_MODEL_DIR): 3 stages, 128 experts top-3, targets 37-39."""
    _need_tilelang()
    if not os.environ.get("DSV41_MODEL_DIR"):
        raise unittest.SkipTest("DSV41_MODEL_DIR not set")
    _gate_draft("real", 40, cs=(1, 3, 6), chain=True, rowwise=True)


# ------------------------------------------------------------------------------------------------ graph
def test_draft_graph_replay_relaunch_no_sync():
    _need_tilelang()
    from runtime.draft_step import DraftStep, ring_restore, ring_snapshot

    with _model("mini3") as model:
        L = 17
        ids = _toks(L, 5)
        first, mh_pre = _prefill(model, ids)
        t6, mh6 = _verify6(model, [int(first)] + _toks(5, 6), L)
        snap0 = ring_snapshot(model)
        ds = DraftStep(model)
        cases = [(L - 1, 0), (L, 1), (L + 2, 3), (L + 5, 6)]

        def run(step, graph):
            out = []
            for a, c in cases:
                ring_restore(model, snap0)
                out.append(_ds_run(model, step, a, c, mh=mh6, tok=t6 if c else first, graph=graph))
            return out

        e = run(ds, False)
        ds.capture()
        g1 = run(ds, True)
        for (a, c), x, y in zip(cases, g1, e):
            _eq(x, y, f"graph vs static-eager A'={a} ctx {c}")
        g2 = run(ds, True)
        for x, y in zip(g2, g1):
            _eq(x, y, "graph replay #2 vs #1")
        ds.capture(fills=True)
        g3 = run(ds, True)
        for x, y in zip(g3, g1):
            _eq(x, y, "fills-on recapture vs fills-off")
        ds2 = DraftStep(model)
        ds2.capture()
        g4 = run(ds2, True)
        for x, y in zip(g4, g1):
            _eq(x, y, "relaunch vs replay #1")
        ring_restore(model, snap0)
        with torch.inference_mode():
            ds.step(L + 5, 6, main_hidden=mh6, tok=t6)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                ds.forward_static()
                ds.forward_static()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        COUNTS["graph"] += 1
        print(f"  draft graph == static-eager (ctx 0/1/3/6); replay x2, fills-on recapture, relaunch equal; no host "
              f"sync; staging {ds.staging_bytes() / 2**20:.1f} MiB")


TESTS = [test_draft_equals_reference_mini, test_draft_equals_reference_mini3, test_draft_equals_reference_ring_wrapped,
         test_draft_graph_replay_relaunch_no_sync, test_draft_equals_reference_real,
         test_draft_port_exact_mini3, test_draft_port_dg_moe_mini3, test_draft_port_full_graph_mini3,
         test_draft_port_real]

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
    for line in REPORT:
        print(f"  note: {line}")
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"bit-equal comparisons executed: {COUNTS['gates']} gate runs, {COUNTS['drafts']} draft sets, "
          f"{COUNTS['logit_rows']} fp32 logit rows, {COUNTS['rings']} ring tensors, {COUNTS['graph']} graph runs")
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
