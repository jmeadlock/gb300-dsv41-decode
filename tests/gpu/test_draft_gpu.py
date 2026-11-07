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
    class of dg_moe.py); gated: finite, the same routed experts, and (T12-FIX2) the A3 class by `dg_moe_evidence`:
    every routed pair's w1 / w3 / w2 GEMM within 1 bf16 ulp (+ an accumulation-error allowance) of an fp64 oracle
    on THAT path's own quantized inputs, no scatter / SwiGLU / combine slip. An element past DG_MAX_ULP is no longer
    a failure by itself (box r2: real stage 1 x1, 1 / 25,600 elements at 16 ulp, |d| 0.5): it is REPORTED with its
    row / column, experts, per-expert terms, both paths' distance from the fp64 oracle of the final value and the
    cancellation factor max|term| / |y|; a wrong expert / row / SF would put a GEMM far from its oracle and fail.
    The full-impl draft vs the reference: drafts agreement and max |logit delta| REPORTED.
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
DG_MAX_ULP = 8     # P2: combined-output elements past this get the A3 evidence line (reported; T12-FIX2: not a gate)
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


# ------------------------------------------------------------------------------------------------ T12-FIX2: P2 evidence
# The P2 gate's question for any combined-output element past the 1-ulp class: accumulation order (A3, the class of
# the main model's DG vs TileLang MoE) or an indexing slip (a wrong expert / row / SF somewhere in the DG path)?
# `dg_moe_evidence` answers it per routed pair with an fp64 oracle of each GEMM ON THAT PATH'S OWN quantized inputs
# (fp8 codes x block scales, fp4 e2m1 x e8m0 weights dequantized exactly): a GEMM that read the wrong expert, row or
# scale is far from its oracle; a GEMM that only summed in another order is within 1 bf16 ulp of it. The chain:
#   x act_quant (A of w1/w3)  DG bytes == act_quant(x[row])                     a scatter slip shows here
#   w1, w3                    each path within 1 ulp of its own fp64 oracle      expert / SF slip shows here
#   w2 input                  DG bytes == act_quant(bf16(w * silu(g1) * g3)) of DG's OWN g1/g3 (SwiGLU reproduces);
#                             equality with the reference's input REPORTED (a 1-ulp g1/g3 can flip an fp8 code)
#   w2                        each path within 1 ulp of its own fp64 oracle
#   combine                   each path's output == bf16(fp32 0 + its pair outputs in ascending id + shared)
# All of it holding = the final difference is A3 (GEMM accumulation order, propagated / amplified by cancellation in
# the fp32 combine: reported as max |term| / |y| at each element past the 1-ulp class); anything else fails the gate.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def bf16_spacing(v: torch.Tensor) -> torch.Tensor:
    """bf16 ulp at |v| (fp64 in, fp64 out): 2^(e - 8) for |v| = m * 2^e, m in [0.5, 1); the subnormal floor 2^-133."""
    _, e = torch.frexp(v.abs().to(torch.float64))
    return torch.exp2(torch.clamp(e.to(torch.float64) - 8.0, min=-133.0))


def fp4_rows_f64(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp4 e2m1 pairs [N, K/2] (low nibble first, tests/test_static_emulated_cpu.py `_fp4_deq`) x e8m0 scales
    [N, K/32] -> exact fp64 [N, K]."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=w.device)
    u = w.contiguous().view(torch.uint8).to(torch.long)
    q = torch.stack([lut[u & 0xF], lut[u >> 4]], -1).flatten(-2)
    sc = torch.exp2(s.contiguous().view(torch.uint8).to(torch.float64) - 127.0)
    return q * sc.repeat_interleave(q.size(-1) // sc.size(-1), -1)


def act_f64(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 codes [.., K] x per-block scales [.., K/B] (fp32 values or e8m0 codes) -> exact fp64 [.., K]."""
    if s.dtype == torch.float8_e8m0fnu:
        sf = torch.exp2(s.contiguous().view(torch.uint8).to(torch.float64) - 127.0)
    else:
        sf = s.to(torch.float64)
    return q.float().to(torch.float64) * sf.repeat_interleave(q.size(-1) // sf.size(-1), -1)


def _scale_f32(s: torch.Tensor) -> torch.Tensor:
    if s.dtype == torch.float8_e8m0fnu:
        return torch.exp2(s.contiguous().view(torch.uint8).to(torch.float32) - 127.0)
    return s.float()


ORACLE_TOL = 2.0 ** -12  # GEMM vs its fp64 oracle: |d| <= 1 bf16 ulp + ORACLE_TOL * sum|a*w| (accumulation error
#                         bound, generous for fp8/fp4 MMA accumulation); a wrong expert / row / SF block is O(K^-1/2)


def oracle_dist(got: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> dict:
    """got [N] (bf16) vs the exact W @ A (fp64 [N, K] x [K]): max distance in bf16 ulps of the exact value, and the
    max excess over 1 ulp normalized by sum_k |A_k W_nk| (the scale of any accumulation-order error)."""
    exact = W @ A
    absum = W.abs() @ A.abs()
    d = (got.to(torch.float64) - exact).abs()
    sp = bf16_spacing(exact)
    ex = torch.clamp(d - sp, min=0.0) / torch.clamp(absum, min=1e-300)
    return {"ulp": float((d / sp).max()) if d.numel() else 0.0, "excess": float(ex.max()) if d.numel() else 0.0}


def dg_moe_evidence(R, st, x: torch.Tensor, got: torch.Tensor, want: torch.Tensor, dg: dict, swiglu_ref: bool = True,
                    elem_ulp: int = 1, max_elems: int = 8) -> dict:
    """The chain above for one DraftDGMoE call. st: the stage (its reference MoE modules); x [1, R, D] bf16; got /
    want: the DG / reference outputs; dg: the DG path's buffers {"xq1", "xs1", "g1", "g3", "xq2", "xs2", "out"} (A rows)
    and {"posmat"} [R, k] (A rows of row r's pairs, ascending id); swiglu_ref: the DG SwiGLU ran the reference ops
    (glue off) so its w2 input must reproduce exactly. -> {"a3": bool, "why": [...], "pairs": [...], "elems": [...]}."""
    from runtime.dg_inplace import bf16_ulp
    from runtime.refmodel import reference_device

    import torch.nn.functional as F_

    moe = st.ffn
    D = moe.dim
    x2 = x.reshape(-1, D)
    why, pairs, elems = [], [], []
    with torch.inference_mode(), reference_device(x.device):
        weights, indices = moe.gate(x2, None)                         # model.py:894
        ref = {}
        y = torch.zeros(x2.shape, dtype=torch.float32, device=x2.device)
        for e in sorted({int(v) for v in indices.reshape(-1).tolist()}):   # model.py:897-902, ascending id
            ex = moe.experts[e]
            cap = {}
            hooks = [ex.w1.register_forward_hook(lambda m_, a_, o_: cap.__setitem__("g1", o_.detach().clone())),
                     ex.w3.register_forward_hook(lambda m_, a_, o_: cap.__setitem__("g3", o_.detach().clone())),
                     ex.w2.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach().clone()))]
            try:
                idx, top = torch.where(indices == e)
                out = ex(x2[idx], weights[idx, top, None])
            finally:
                for h_ in hooks:
                    h_.remove()
            y[idx] += out
            q2, s2 = R.act_quant(cap["h"].contiguous(), R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            for t, r in enumerate(idx.tolist()):
                ref[(r, e)] = {"w": float(weights[r, top[t]]), "g1": cap["g1"][t], "g3": cap["g3"][t], "h": cap["h"][t],
                               "q2": q2[t], "s2": s2[t], "out": out[t]}
        sh = moe.shared_experts(x2)                                   # model.py:905
        y += sh
        y_ref = y.type_as(x2)
        replicate_ok = bool(torch.equal(y_ref.view(torch.int16), want.reshape(-1, D).view(torch.int16)))
        if not replicate_ok:
            why.append("the test's replication of the reference MoE != st.ffn(x)")
        qx, sx = R.act_quant(x2.contiguous(), R.fp8_block_size, R.scale_fmt, R.scale_dtype)
        lim = float(moe.experts[next(iter(ref))[1]].swiglu_limit) if ref else 0.0
        posmat = dg["posmat"]
        Rr, k = posmat.shape
        for r in range(Rr):
            es = sorted(int(v) for v in indices[r].tolist())
            for j, e in enumerate(es):
                a = int(posmat[r, j])
                rp, ex = ref[(r, e)], moe.experts[e]
                ev = {"row": r, "expert": e, "arow": a, "w": rp["w"]}
                ev["x_in_equal"] = bool(torch.equal(dg["xq1"][a].view(torch.uint8), qx[r].view(torch.uint8))
                                        and torch.equal(dg["xs1"][a].float(), _scale_f32(sx[r])))
                ax_ref, ax_dg = act_f64(qx[r], sx[r]), act_f64(dg["xq1"][a], dg["xs1"][a])
                for p, key in (("w1", "g1"), ("w3", "g3")):
                    lin = getattr(ex, p)
                    W = fp4_rows_f64(lin.weight, lin.scale)
                    ev[f"{key}_ref_oracle"] = oracle_dist(rp[key], ax_ref, W)
                    ev[f"{key}_dg_oracle"] = oracle_dist(dg[key][a], ax_dg, W)
                    ev[f"{key}_dg_vs_ref_ulp"] = int(bf16_ulp(dg[key][a].contiguous(), rp[key].contiguous()).max())
                    del W
                ev["w2_in_equal_ref"] = bool(torch.equal(dg["xq2"][a].view(torch.uint8), rp["q2"].view(torch.uint8))
                                             and torch.equal(dg["xs2"][a].float(), _scale_f32(rp["s2"])))
                if not ev["w2_in_equal_ref"]:
                    ev["w2_in_codes_differing"] = int((dg["xq2"][a].view(torch.uint8) != rp["q2"].view(torch.uint8)).sum())
                if swiglu_ref:                                        # model.py:845-853 on the DG's own g1 / g3
                    gt, up = dg["g1"][a].float(), dg["g3"][a].float()
                    if lim > 0:
                        up = torch.clamp(up, min=-lim, max=lim)
                        gt = torch.clamp(gt, max=lim)
                    hh = (torch.tensor(rp["w"], dtype=torch.float32, device=gt.device) * (F_.silu(gt) * up)).to(x2.dtype)
                    q3, s3 = R.act_quant(hh.view(1, -1).contiguous(), R.fp8_block_size, R.scale_fmt, R.scale_dtype)
                    ev["w2_in_reproduces_from_dg_g13"] = bool(
                        torch.equal(dg["xq2"][a].view(torch.uint8), q3[0].view(torch.uint8))
                        and torch.equal(dg["xs2"][a].float(), _scale_f32(s3[0])))
                W2 = fp4_rows_f64(ex.w2.weight, ex.w2.scale)
                ev["w2_ref_oracle"] = oracle_dist(rp["out"], act_f64(rp["q2"], rp["s2"]), W2)
                ev["w2_dg_oracle"] = oracle_dist(dg["out"][a], act_f64(dg["xq2"][a], dg["xs2"][a]), W2)
                ev["w2_dg_vs_ref_ulp"] = int(bf16_ulp(dg["out"][a].contiguous(), rp["out"].contiguous()).max())
                del W2
                pairs.append(ev)
        # combine: each path's output from its own pair outputs (fp32 0 + ascending id + shared -> bf16)
        yd = torch.zeros(x2.shape, dtype=torch.float32, device=x2.device)
        for j in range(k):
            yd += dg["out"].index_select(0, posmat[:, j].long()).float()
        yd += sh
        combine_dg_ok = bool(torch.equal(yd.type_as(x2).view(torch.int16), got.reshape(-1, D).view(torch.int16)))
        if not combine_dg_ok:
            why.append("DG output != its own pair outputs combined (posmat / combine slip)")
        # elements past the 1-ulp class: fp64 oracle of the final value and the cancellation factor
        u = bf16_ulp(got.reshape(-1).contiguous(), want.reshape(-1).contiguous())
        bad = torch.nonzero(u > elem_ulp).reshape(-1)[:max_elems].tolist()
        for f in bad:
            r, c = divmod(int(f), D)
            es = sorted(int(v) for v in indices[r].tolist())
            terms, y64 = [], float(sh[r, c])
            for j, e in enumerate(es):
                a = int(posmat[r, j])
                rp, ex = ref[(r, e)], moe.experts[e]
                W2c = fp4_rows_f64(ex.w2.weight[c:c + 1], ex.w2.scale[c:c + 1])[0]
                o_ref = float((act_f64(rp["q2"], rp["s2"]) * W2c).sum())
                o_dg = float((act_f64(dg["xq2"][a], dg["xs2"][a]) * W2c).sum())
                y64 += o_ref
                terms.append({"expert": e, "ref": float(rp["out"][c]), "dg": float(dg["out"][a][c]), "oracle_ref": o_ref,
                              "oracle_dg": o_dg})
            g_, w_ = float(got.reshape(-1)[f]), float(want.reshape(-1)[f])
            sp = float(bf16_spacing(torch.tensor(y64, dtype=torch.float64, device="cpu")))
            big = max([abs(t["ref"]) for t in terms] + [abs(float(sh[r, c]))])
            elems.append({"row": r, "col": c, "ulp": int(u[f]), "dg": g_, "ref": w_, "oracle": y64,
                          "dg_oracle_ulp": abs(g_ - y64) / sp, "ref_oracle_ulp": abs(w_ - y64) / sp,
                          "shared": float(sh[r, c]), "terms": terms, "cancellation": big / max(abs(y64), 1e-30)})
    for ev in pairs:
        tag = f"row {ev['row']} expert {ev['expert']}"
        if not ev["x_in_equal"]:
            why.append(f"{tag}: DG w1/w3 input != act_quant(x[row]) (scatter slip)")
        for key in ("g1", "g3", "w2"):
            for side in ("ref", "dg"):
                o_ = ev[f"{key}_{side}_oracle"]
                if o_["excess"] > ORACLE_TOL:
                    why.append(f"{tag}: {side} {key} is {o_['ulp']:.2f} ulp from its fp64 oracle (excess "
                               f"{o_['excess']:.3g} of sum|a*w| > {ORACLE_TOL:.3g})")
        if swiglu_ref and not ev.get("w2_in_reproduces_from_dg_g13", True):
            why.append(f"{tag}: DG w2 input does not reproduce from its own w1/w3 outputs (SwiGLU / quant slip)")
    return {"a3": replicate_ok and not why, "why": why, "replicate_ok": replicate_ok, "combine_dg_ok": combine_dg_ok,
            "pairs": pairs, "elems": elems,
            "pair_max_ulp": max([p["w2_dg_vs_ref_ulp"] for p in pairs] or [0]),
            "w2_inputs_equal": sum(p["w2_in_equal_ref"] for p in pairs), "n_pairs": len(pairs)}


def dg_buffers_view(ds, moe) -> dict:
    """The DraftDGMoE call's A-row buffers (runtime/draft_step.py DraftDGBuffers) for `dg_moe_evidence`."""
    b = ds.dgm
    return {"xq1": b.xq[1], "xs1": b.xs[1], "g1": b.d["w1"], "g3": b.d["w3"], "xq2": b.xq[2], "xs2": b.xs[2],
            "out": b.d["w2"], "posmat": moe.last["posmat"]}


def reference_pairs_view(R, st, x: torch.Tensor):
    """The reference MoE's own per-pair intermediates in the DG A-row layout (moe_rows.dedup_routes, block rows = R)
    -> (view for `dg_moe_evidence`, the combined output from them). A self-test: evidence(reference as DG) is A3 with
    0 ulp everywhere (tests/test_dspark_loop_cpu.py, CPU emulation)."""
    from runtime.moe_rows import dedup_routes
    from runtime.refmodel import reference_device

    moe = st.ffn
    D = moe.dim
    x2 = x.reshape(-1, D)
    with torch.inference_mode(), reference_device(x.device):
        weights, indices = moe.gate(x2, None)
        Rr, k = indices.shape
        m = dedup_routes(indices, weights.float(), Rr)
        M = Rr * k * Rr
        e0 = moe.experts[int(indices[0, 0])]
        I_ = e0.w1.out_features
        qx, sx = R.act_quant(x2.contiguous(), R.fp8_block_size, R.scale_fmt, R.scale_dtype)
        dev = x.device
        v = {"xq1": torch.zeros(M, D, dtype=qx.dtype, device=dev), "xs1": torch.ones(M, sx.size(-1), device=dev),
             "g1": torch.zeros(M, I_, dtype=x.dtype, device=dev), "g3": torch.zeros(M, I_, dtype=x.dtype, device=dev),
             "out": torch.zeros(M, D, dtype=x.dtype, device=dev), "posmat": m["posmat"]}
        xq2 = xs2 = None
        for e in sorted({int(t) for t in indices.reshape(-1).tolist()}):
            ex = moe.experts[e]
            cap = {}
            hooks = [ex.w1.register_forward_hook(lambda m_, a_, o_: cap.__setitem__("g1", o_.detach().clone())),
                     ex.w3.register_forward_hook(lambda m_, a_, o_: cap.__setitem__("g3", o_.detach().clone())),
                     ex.w2.register_forward_pre_hook(lambda m_, a_: cap.__setitem__("h", a_[0].detach().clone()))]
            try:
                idx, top = torch.where(indices == e)
                out = ex(x2[idx], weights[idx, top, None])
            finally:
                for h_ in hooks:
                    h_.remove()
            q2, s2 = R.act_quant(cap["h"].contiguous(), R.fp8_block_size, R.scale_fmt, R.scale_dtype)
            if xq2 is None:
                xq2 = torch.zeros(M, q2.size(-1), dtype=q2.dtype, device=dev)
                xs2 = torch.ones(M, s2.size(-1), device=dev)
            for t, r in enumerate(idx.tolist()):
                j = sorted(int(u) for u in indices[r].tolist()).index(e)
                a = int(m["posmat"][r, j])
                v["xq1"][a], v["xs1"][a] = qx[r], _scale_f32(sx[r])
                v["g1"][a], v["g3"][a], v["out"][a] = cap["g1"][t], cap["g3"][t], out[t]
                xq2[a], xs2[a] = q2[t], _scale_f32(s2[t])
        v["xq2"], v["xs2"] = xq2, xs2
        y = torch.zeros(x2.shape, dtype=torch.float32, device=dev)
        for j in range(k):
            y += v["out"].index_select(0, m["posmat"][:, j]).float()
        y += moe.shared_experts(x2)
    return v, y.type_as(x2).view(x.shape)


def _evidence_line(tag: str, ev: dict) -> str:
    def mx(key, side, what):
        return max(p[f"{key}_{side}_oracle"][what] for p in ev["pairs"]) if ev["pairs"] else 0.0

    s = (f"{tag}: {'A3' if ev['a3'] else 'NOT A3'} -- {ev['n_pairs']} pairs, per-pair w2 output DG vs ref max ulp "
         f"{ev['pair_max_ulp']}, w2 inputs bit-equal {ev['w2_inputs_equal']}/{ev['n_pairs']}; vs fp64 oracle (ref/dg) "
         + ", ".join(f"{k_} max ulp {mx(k_, 'ref', 'ulp'):.2f}/{mx(k_, 'dg', 'ulp'):.2f} excess "
                     f"{mx(k_, 'ref', 'excess'):.2g}/{mx(k_, 'dg', 'excess'):.2g}" for k_ in ("g1", "g3", "w2"))
         + f"; combine reproduces {ev['combine_dg_ok']}")
    for e in ev["elems"]:
        s += (f"; element row {e['row']} col {e['col']}: {e['ulp']} ulp (DG {e['dg']:.6g}, ref {e['ref']:.6g}, fp64 "
              f"oracle {e['oracle']:.9g}: DG {e['dg_oracle_ulp']:.2f} / ref {e['ref_oracle_ulp']:.2f} ulp from it; "
              f"cancellation max|term|/|y| {e['cancellation']:.1f}; terms " + ", ".join(
                  f"e{t['expert']} ref {t['ref']:.6g} dg {t['dg']:.6g} oracle {t['oracle_ref']:.9g}"
                  for t in e["terms"]) + f", shared {e['shared']:.6g})")
    if ev["why"]:
        s += "; why not: " + " | ".join(ev["why"][:6])
    return s


# ------------------------------------------------------------------------------------------------ M1-T11-PORT
def _gate_dg_moe(kind="mini3", n_x=3):
    """P2: DraftDGMoE (DeepGEMM contiguous over the resident stack) vs the stage's reference eager MoE on the same x.
    T12-FIX2: an output past DG_MAX_ULP no longer fails by itself; it must be A3 by `dg_moe_evidence` (per-pair
    GEMMs within 1 ulp of their fp64 oracles, no scatter / SwiGLU / combine slip), reported with the element, its
    experts and both paths' oracle distances; and every pair's evidence is gated on every call."""
    from runtime.dg_inplace import compare_bf16
    from runtime.draft_step import DraftStep
    from runtime.refmodel import reference_device

    with _model(kind) as model:
        ds = DraftStep(model, moe_impl="dg-inplace", **STEP_KW)
        D = ds.stages[0].attn_norm.dim
        g = torch.Generator(device="cpu").manual_seed(11)
        glue_sw = ds.glue is not None and ds.glue.on("swiglu_routed")
        worst, lines, a3_lines = 0, [], []
        for s, (st, moe) in enumerate(zip(ds.stages, ds.moes)):
            for j in range(n_x):
                x = (torch.randn(1, ds.block, D, generator=g, device="cpu") * (0.5 + j)).to(torch.bfloat16).to(DEV)
                with torch.inference_mode():
                    got = moe(x).clone()
                    ids_dg = sorted(int(e) for e in moe.last["uniq"][moe.last["counts"] > 0].tolist())
                    view = {k_: v.clone() for k_, v in dg_buffers_view(ds, moe).items()}
                    with reference_device(x.device):
                        want = st.ffn(x)                                  # model.py:891-906, eager reference
                    _, ind = st.ffn.gate(x.view(-1, D), None)
                    ids_ref = sorted({int(e) for e in ind.reshape(-1).tolist()})
                _sync()
                assert ids_dg == ids_ref, f"{kind} stage {s}: DG routed {ids_dg} vs reference {ids_ref}"
                assert bool(torch.isfinite(got.float()).all()), f"{kind} stage {s}: non-finite DG MoE output"
                c = compare_bf16(got.reshape(-1), want.reshape(-1))
                worst = max(worst, c["max_ulp"])
                ev = dg_moe_evidence(ds.R, st, x, got, want, view, swiglu_ref=not glue_sw)
                _sync()
                lines.append(f"s{s}x{j}: equal {c['equal']}, mismatch {100 * c['mismatch_frac']:.3f}%, max ulp "
                             f"{c['max_ulp']}, max |d| {c['max_abs']:.3g}, per-pair max ulp {ev['pair_max_ulp']}")
                if c["max_ulp"] > 1 or not ev["a3"]:
                    a3_lines.append(_evidence_line(f"{kind} s{s}x{j}", ev))
                assert ev["a3"], f"{kind} stage {s} x{j}: DG MoE vs reference is NOT the A3 class: " + \
                    _evidence_line(f"{kind} s{s}x{j}", ev)
                if c["max_ulp"] > DG_MAX_ULP:
                    REPORT.append(f"{kind} stage {s} x{j}: {c['n_ulp_gt1']} element(s) past the {DG_MAX_ULP}-ulp guard,"
                                  f" explained A3 (accumulation order, see the evidence line)")
        REPORT.append(f"{kind}: draft DG MoE vs reference eager MoE (bf16): " + "; ".join(lines))
        REPORT.extend(f"A3 evidence {ln}" for ln in a3_lines)
        COUNTS["gates"] += 1
        print(f"  {kind}: draft DG MoE == reference eager MoE within the A3 class (max ulp {worst}, every pair's GEMMs "
              f"within 1 ulp of their fp64 oracles; {ds.dgm.describe()['bytes']})")


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
