"""M1-T11: the DSpark draft step -- static, graph-capturable, sampling-free (docs/M1-DSPARK-DESIGN.md 1.2, 2.3).

    model = build_and_load(model_dir, map, "cuda", load_mtp=True)      # model.mtp = the 3 reference DSparkBlocks
    ds = DraftStep(model)                                              # or DraftStep(model, main_hidden=st.main_hidden)
    ds.set_inputs(a_pos=A, ctx_count=n + 1, main_hidden=mh[6, 3*dim], tok=t[6])
    ds.forward_static()  /  ds.capture(); ds.replay(A, n + 1)
    ds.drafts [5] int64 (d'1..d'5 for positions A+2..A+6), ds.out_ids [1, 6] = [t_n, d'1..d'5],
    ds.logits [5, V] fp32 (Markov bias included), ds.confidence [5] fp32 (pre-sigmoid, model.py:1100)

Inputs (static buffers, device):
  main_hidden [6, n_targets*dim]  the verify step's hc-means of the inputs of layers 37-39, one row per verify row
                                  (StaticStep(rows=6, hidden_layers=...).main_hidden can be passed in and aliased)
  tok [6] int64                    the verify step's argmaxes t_0..t_5; the block's first id is the bonus
                                  t_n = tok[ctx_count - 1] (tok[0] when ctx_count == 0: the prefill's token)
  a_pos [1] int64                  A' = the last committed position (= P0 + n)
  ctx_count [1] int64              rows of main_hidden to write into the draft rings (= n + 1; 0 after a prefill)

The reference this mirrors, op for op (model.py; `forward_spec` 1278-1285 at start_pos = A', one main row):
  forward_embed 1131-1138   main_x = main_norm(main_proj(main_hidden)) (called on all 6 rows at once; fp8 GEMM rows
                            are independent of M, M1-T10 census) and the block ids [t_n, noise x 4] -> embed -> hc copies
  DSparkAttention 1041-1044 per stage, per context row i < ctx_count at position p_i = A' - ctx_count + 1 + i:
     + 1067                 kv_norm(wkv(main_x)) -> RoPE at p_i -> act_quant QDQ -> ring slot p_i % 128. Static: all 6
                            rows are computed; rows i >= ctx_count write back the slot's current bytes (the 6 slots of
                            6 consecutive positions are distinct, so the write is a value no-op for them). The reference
                            writes one row per forward_spec call; ctx_count > 1 rows = that many consecutive calls
                            (vLLM's precompute_and_store_context_kv, dspark.py:166-197; design 1.3 step 4).
  DSparkAttention 1056-1077 block rows at A'+1..A'+5: q / kv / RoPE / QDQ as the reference; attention over
                            cat([ring(128) | block(5)]) with the reference's index list (get_dspark_topk_idxs,
                            1022-1031: ring slots 0..min(128, A'+1)-1 in SLOT order, then 128..132) built on the
                            device at a fixed width 133 by moving the -1 padding to the END: the kernel already treats
                            entries past `topk` as -1 inside the last 64-block (kernel.py:358-359), so the blocks it
                            sees are the reference's plus whole -1 blocks, which are exact no-ops (static_step.py #9).
  Block 983-996             hc mixes / pre / norms / post: the reference module methods
  MoE 891-906               draft MoE (top-3 of 128, all HBM): runtime/moe_rows.py's dedup (`dedup_routes`, imported,
                            S = 5 * 3 slots), the slot's expert rows gathered from the stage's expert STACK
                            (runtime/mtp_loader.py) with index_select into one shared staging set, one reference
                            `Expert` per slot on its [5, D] block, per-row ascending-id combine (`combine_rows`) +
                            shared expert (StaticMoERows' conventions; experts are row-independent, M1-T10 census)
  forward_head 1140-1159    the reference method itself (device ops only): hc_pre, norm, the shared fp32 head on 5
                            rows, the 5-step Markov chain (sequential argmax: sample(., 0) = argmax, 1291-1292; the
                            stages' temperature is 0, runtime/mtp_loader.py) and the confidence head.

No host sync anywhere; one CUDA graph (`capture`). `capture` warms up and records with ctx_count = 0 (a value no-op on
the rings) and leaves ctx_count at 0.

After the prefill (design 2.3 "After prefill", risk 7): `seed_draft_rings` runs the reference prefill path
(forward_spec at start_pos 0, model.py:1046-1054, 1126-1128, 1283-1284), then the first draft is this step with
ctx_count = 0 and A' = L - 1 (no rewrite of slot (L-1) % 128 at decode numerics).

M1-T11-PORT (docs/HANDOFF-M1-T11-PORT.md): the same op chain on the verify step's fast kernels, one switch each, the
defaults ("ref") being exactly the T11 step above. `DraftStep(model, fp8_impl=, moe_impl=, glue_impl=, attn_impl=)`:
  fp8_impl "gemv"       every fp8 Linear of the draft (main_proj at M = 6, the context wkv at M = 6, wq_a / wq_b / wkv /
                        wo_b and the shared experts at M = 5) runs runtime/gemv_mxfp8.py through runtime/patches.py's
                        `fp8_linear_gemv` wrapper, routed by the step's own `fp8_gemv_scope` around `_body` (the verify's
                        mechanism, static_step.py #16; DSV41_GEMV_PIN_M1 applies). Same act_quant, the GEMV's block-sum
                        order: bit-equal to the reference forward_spec run inside the same scope (that is the oracle the
                        GPU test uses). The fp32 heads (shared head, the 5 sequential Markov-head calls, confidence) stay
                        cuBLAS fp32 (M2's kernel, W3/W4), 5 launches of the same GEMM in the graph.
  moe_impl "dg-inplace" (verify names "dg-masked" map here too) the stage's routed experts through DeepGEMM's CONTIGUOUS
                        grouped GEMM straight over the stage's resident expert stack ([E, N, K/2] int8, runtime/
                        mtp_loader.py; all E experts are HBM-resident, so there is no staging and no cold path): the
                        R-row dedup (runtime/moe_rows.py `dedup_fused`, one launch: slots / ranks / posmat and, through
                        its dg-inplace form with gtab = identity, the m_indices tags), pair (r, e) at A row slot*ALIGN +
                        rank, the e8m0 scales packed ONCE at build into DeepGEMM's MN-major int32 words (dg_inplace
                        `pack_sf`: the same bytes, no value change; ~141 MB per real stage), then w1 / w3 -> SwiGLU glue
                        -> act_quant -> w2 -> per-row ascending-id combine + shared expert (DGInplaceMoERows' ops). The
                        block-scaled UMMA accumulation is DeepGEMM's: the documented 1-ulp class vs the reference fp4_gemm
                        (dg_moe.py), measured and reported by the GPU test against the eager reference MoE.
  glue_impl "fused"     runtime/fused_glue.py's row-generic launch wrappers at R = 5 (one launch per instance): mHC post
                        (`launch_hc_post`), the o-path inverse RoPE + QDQ (`launch_o_rope_qdq`), the shared-expert SwiGLU
                        + w2 act_quant (`launch_swiglu_quant` + patches.fp8_linear_prequant) and, with dg-inplace, the
                        pair act_quant scatter / routed SwiGLU / combine (the IDX forms). Each cluster is self-checked at
                        build on the draft's shapes against the reference ops it replaces (sum / RoPE contraction orders
                        picked by bit-equality, as FusedGlue.calibrate does) and stays on the reference ops if it fails
                        (`describe()["glue"]`). StaticStep's round-2 clusters (hc_norm, kv_path, q_path, router) are bound
                        to the backbone layers and stay reference ops here.
  attn_impl "flashmla"  the [ring | block] window attention through runtime/attn_flashmla.py (A3-class, as the verify's
                        arm C) with topk_length = min(128, A'+1) + 5 per row (the reference list, -1 tail masked);
                        build-time attn_flashmla.self_check on the draft's heads / head_dim / sink, else reference.
                        "bounded" maps to the reference here (3 blocks of 64 at every A' >= 123: nothing to bound).
The Markov chain and confidence head stay the reference's torch ops (tiny). No host sync in any impl.
"""
from __future__ import annotations

import contextlib
import functools
import itertools
import os
import sys
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

PROJS = ("w1", "w3", "w2")
CTX_ROWS = 6  # verify rows: 1 bonus + 5 drafts (design 1.3)
DRAFT_FP8_IMPLS = ("ref", "gemv")
DRAFT_MOE_IMPLS = ("ref", "dg-inplace")
DRAFT_GLUE_IMPLS = ("ref", "fused")
DRAFT_ATTN_IMPLS = ("ref", "flashmla")
DRAFT_GLUE_CLUSTERS = ("hc_post", "o_rope_qdq", "swiglu_shared", "dg_act", "swiglu_routed", "moe_combine")


def draft_impls_for(moe_impl: str = "ref", fp8_impl: str = "ref", glue_impl: str = "ref", attn_impl: str = "ref") -> dict:
    """The verify step's impl names -> the draft's (module docstring): dg-masked / dg-inplace -> dg-inplace (the draft's
    experts are all resident), bounded -> ref; the rest by name."""
    return {"moe_impl": "dg-inplace" if moe_impl in ("dg-masked", "dg-inplace") else "ref",
            "fp8_impl": fp8_impl if fp8_impl in DRAFT_FP8_IMPLS else "ref",
            "glue_impl": glue_impl if glue_impl in DRAFT_GLUE_IMPLS else "ref",
            "attn_impl": attn_impl if attn_impl in DRAFT_ATTN_IMPLS else "ref"}


def _cpu_default(fn):
    """Host-built synthetic data runs with the CPU as the default device (a Generator may have set cuda)."""
    @functools.wraps(fn)
    def w(*a, **k):
        with torch.device("cpu"):
            return fn(*a, **k)
    return w


def _ref():
    m = sys.modules.get("model")
    if m is None:
        from runtime.refmodel import import_reference
        m = import_reference()
    return m


# ------------------------------------------------------------------------------------------------ pure index math
def ctx_positions(a_pos: torch.Tensor, ctx_count: torch.Tensor, ar_ctx: torch.Tensor):
    """-> (positions [C] int64, valid [C] bool): context row i sits at A' - ctx_count + 1 + i, valid iff i < ctx_count
    (the accepted verify rows 0..n are positions P0..P0+n = A'-n..A')."""
    pos = a_pos.view(1) - ctx_count.view(1) + 1 + ar_ctx
    return pos, ar_ctx < ctx_count.view(1)


def bonus_index(ctx_count: torch.Tensor) -> torch.Tensor:
    """Index into tok[6] of the block's first id: t_n = tok[ctx_count - 1]; tok[0] when ctx_count == 0."""
    return torch.clamp(ctx_count.view(1) - 1, min=0)


def dspark_window_idxs_dev(a_pos: torch.Tensor, win: int, block: int, ar_idx: torch.Tensor) -> torch.Tensor:
    """get_dspark_topk_idxs (model.py:1022-1031) from a device A' at the fixed width win + block -> int32
    [1, block, win + block]: entries j < n = min(win, A'+1) are ring slots j (slot order), the next `block` are
    win + 0..block-1, the rest -1 (the reference's list, then -1 padding at the end)."""
    n = torch.clamp(a_pos.view(1) + 1, max=win)
    j = ar_idx
    idx = torch.where(j < n, j, torch.where(j < n + block, win + (j - n), -1))
    return idx.int().view(1, 1, -1).expand(1, block, -1).contiguous()


def reference_dspark_idxs(win: int, block: int, start_pos: int) -> torch.Tensor:
    """model.py:1025-1031's list for one start_pos, on the CPU, without the lru cache (tests)."""
    m = torch.cat([torch.arange(min(win, start_pos + 1)), win + torch.arange(block)])
    return m.int().view(1, 1, -1).expand(1, block, -1).contiguous()


def freqs_rows(table_real: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Rows `idx` of a complex freqs_cis through its fp32 view -> complex64 [len(idx), rd/2] (= freqs_cis[p:p+k])."""
    return torch.view_as_complex(table_real.index_select(0, idx))


# ------------------------------------------------------------------------------------------------ reference helpers
@torch.inference_mode()
def prefill_hidden(model, tokens: torch.Tensor, hidden_layers: Sequence[int]):
    """The reference prefill `model(tokens, 0)` with `target_layer_ids` set for this call only (model.py:1211,
    1267-1269) -> (ids, logits, main_hidden [1, L, n*dim]). Runs under refmodel.reference_device(tokens.device)."""
    from runtime.refmodel import reference_forward

    prev = getattr(model, "target_layer_ids", ())
    model.target_layer_ids = tuple(int(i) for i in hidden_layers)
    try:
        return reference_forward(model, tokens, 0)
    finally:
        model.target_layer_ids = prev


@torch.inference_mode()
def seed_draft_rings(model, first_tok: torch.Tensor, main_hidden: torch.Tensor) -> None:
    """The reference's prefill path of the draft (forward_spec at start_pos 0: every stage seeds its 128-ring with the
    prompt's main_kv, model.py:1046-1054, 1126-1128; returns None, 1283-1284)."""
    from runtime.refmodel import reference_device

    with reference_device(main_hidden.device):
        out = model.forward_spec(first_tok.reshape(-1)[:1], main_hidden, 0)
    assert out is None


@torch.inference_mode()
def reference_draft(model, a_pos: int, ctx_hidden: torch.Tensor, ctx_toks: Sequence):
    """The oracle: `forward_spec` decode calls at start_pos a_pos - c + 1 + i, i = 0..c-1, each with ONE main_hidden
    row (ctx_hidden [c, H]) and input id ctx_toks[i] (the last one = the bonus t_n); returns the last call's
    (output_ids [1, 6], logits [1, 5, V], confidence [1, 5]). c = 1 with the prefill's last row at a_pos = L-1 is the
    reference first draft after a prefill (it re-writes slot (L-1) % 128 at decode numerics, design risk 7)."""
    from runtime.refmodel import reference_device

    c = ctx_hidden.size(0)
    if c < 1:
        raise ValueError("reference_draft needs >= 1 context row (the reference decode path writes one per call)")
    dev = ctx_hidden.device
    res = None
    with reference_device(dev):
        for i in range(c):
            sp = int(a_pos) - c + 1 + i
            if sp < 1:
                raise ValueError(f"start_pos {sp} < 1 is the reference's prefill path")
            t = ctx_toks[i]
            t = t.reshape(1).to(dev) if isinstance(t, torch.Tensor) else torch.tensor([int(t)], device=dev)
            res = model.forward_spec(t, ctx_hidden[i].reshape(1, 1, -1), sp)
    return res


def ring_snapshot(model) -> List[torch.Tensor]:
    return [st.attn.window_kv_cache.clone() for st in model.mtp]


def ring_restore(model, snap: List[torch.Tensor]) -> None:
    with torch.inference_mode():
        for st, s in zip(model.mtp, snap):
            st.attn.window_kv_cache.copy_(s)


# ------------------------------------------------------------------------------------------------ static draft MoE
class DraftMoE:
    """One stage's static MoE (module docstring): gate -> dedup (moe_rows.dedup_routes) -> index_select of the
    slots' experts from the stage's stacks into the shared staging -> per-slot reference Expert -> combine -> shared."""

    def __init__(self, owner: "DraftStep", moe: nn.Module, stacks: Dict[str, tuple]):
        self.owner, self.moe, self.stacks = owner, moe, stacks
        self.last: Dict[str, torch.Tensor] = {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from runtime.moe_rows import combine_rows, dedup_routes

        o, moe = self.owner, self.moe
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893, [5, D]
        R_, D = x.shape
        S = o.n_slots
        weights, indices = moe.gate(x, None)                           # model.py:894 (drafts are text)
        m = dedup_routes(indices, weights, R_)
        self.last = m
        for p in PROJS:                                                # slot u <- expert uniq[u] (whole rows)
            torch.index_select(self.stacks[p][0], 0, m["uniq"], out=o.staging_w[p])
            torch.index_select(self.stacks[p][1], 0, m["uniq"], out=o.staging_s[p])
        xb, wb = o.moe_xb, o.moe_wb
        xb.index_copy_(0, m["arow"], x.index_select(0, m["prow"]))     # pair (r, e) -> row `rank` of slot u
        wb.index_copy_(0, m["arow"], m["pw"].view(-1, 1))
        xv, wv = xb.view(S, R_, D), wb.view(S, R_, 1)
        outs = [ex(xv[u], wv[u]) for u, ex in enumerate(o.staged_experts)]  # model.py:902
        y = combine_rows(torch.stack(outs).view(S * R_, -1), m["posmat"], x)  # model.py:895-902's order per row
        y += o._shared(moe.shared_experts, x)                          # model.py:905
        return y.type_as(x).view(shape)                                # model.py:906


# ------------------------------------------------------------------------------------------------ M1-T11-PORT: DG MoE
class DraftDGBuffers:
    """The draft's DeepGEMM contiguous grouped-GEMM buffers, shared by the stages (they run one after another): A
    [P*ALIGN, K] fp8 + fp32 scales for the w1/w3 input (1) and the w2 input (2), outputs d[p] [P*ALIGN, N] bf16,
    m_indices [P*ALIGN] int32 (-1 except the pair rows of used slots), P = block * topk A blocks; per stage the
    resident weights (the stack itself, int8 [E, N, K/2]) and the packed MN-major int32 scales (built once here).
    Group of slot u = its expert id (gtab = identity: every expert resident)."""

    def __init__(self, owner, mt):
        from runtime.dg_inplace import ALIGN_ENV, MK_ALIGN, check_align, is_mn_major_packed, pack_sf
        from runtime.dg_moe import ACT_BLOCK, WEIGHT_BLOCK, import_deep_gemm, pick_e8m0_method

        R, d = owner.R, owner.dev
        if (R.fp8_block_size, R.fp4_block_size, R.scale_fmt) != (ACT_BLOCK, WEIGHT_BLOCK, "ue8m0"):
            raise ValueError(f"draft dg-inplace assumes per-32 ue8m0 activation and weight scales; the reference has "
                             f"fp8_block_size={R.fp8_block_size} fp4_block_size={R.fp4_block_size} {R.scale_fmt}")
        self.owner, self.dev = owner, d
        self.rows, self.k = int(owner.block), int(owner.topk)
        self.P = self.rows * self.k
        self.align = check_align(int(os.environ.get(ALIGN_ENV, MK_ALIGN)))
        if self.align < self.rows:
            raise ValueError(f"contiguous alignment {self.align} < {self.rows} rows per block")
        self.dg = import_deep_gemm()
        self.fn = getattr(self.dg, "m_grouped_fp8_fp4_gemm_nt_contiguous")
        self.alignment = self._set_alignment(self.align)
        self.deterministic = self._set_deterministic()
        self.e8m0_method = pick_e8m0_method(d)
        n_st = len(owner.stages)
        w1, s1 = mt.stacks[0]["w1"]
        E, I, Dh = (int(v) for v in w1.shape)
        D = 2 * Dh
        self.E, self.D, self.I = E, D, I
        self.N = {"w1": I, "w3": I, "w2": D}
        self.K = {"w1": D, "w3": D, "w2": I}
        for s in range(n_st):
            for p in PROJS:
                w, sc = mt.stacks[s][p]
                want_w, want_s = (E, self.N[p], self.K[p] // 2), (E, self.N[p], self.K[p] // WEIGHT_BLOCK)
                if tuple(w.shape) != want_w or tuple(sc.shape) != want_s or not (w.is_contiguous() and
                                                                                  sc.is_contiguous()):
                    raise ValueError(f"mtp.{s} {p}: stack {tuple(w.shape)}/{tuple(sc.shape)} is not a contiguous "
                                     f"{want_w}/{want_s}")
                if w.dtype != torch.int8:
                    raise ValueError(f"mtp.{s} {p}: stack weights {w.dtype}, DeepGEMM takes int8 e2m1 pairs")
        lims = {float(st.ffn.experts[0].swiglu_limit) for st in owner.stages} | \
            {float(st.ffn.shared_experts.swiglu_limit) for st in owner.stages}
        if len(lims) != 1:
            raise ValueError(f"draft stages' swiglu limits differ: {sorted(lims)}")
        self.swiglu_limit = lims.pop()
        M = self.P * self.align
        f32, bf16, fp8 = torch.float32, torch.bfloat16, torch.float8_e4m3fn
        self.xq = {1: torch.zeros(M, D, dtype=fp8, device=d), 2: torch.zeros(M, I, dtype=fp8, device=d)}
        self.xs = {1: torch.ones(M, D // ACT_BLOCK, dtype=f32, device=d),
                   2: torch.ones(M, I // ACT_BLOCK, dtype=f32, device=d)}
        self.d = {p: torch.zeros(M, self.N[p], dtype=bf16, device=d) for p in PROJS}
        self.mi = torch.full((M,), -1, dtype=torch.int32, device=d)
        self.w = [{p: mt.stacks[s][p][0] for p in PROJS} for s in range(n_st)]
        with torch.inference_mode(False), torch.no_grad():
            self.sf = [{p: pack_sf(mt.stacks[s][p][1]) for p in PROJS} for s in range(n_st)]
        for s in range(n_st):
            for p in PROJS:
                assert is_mn_major_packed(self.sf[s][p], E, self.N[p], self.K[p] // 128), (s, p)
        self.gtab = torch.arange(E, dtype=torch.int32, device=d)
        self.dummy = torch.zeros((), dtype=torch.long, device=d)
        self.n_calls = 0

    def _set_deterministic(self) -> dict:
        """deep_gemm.use_deterministic_algorithms (the verify's DGBuffers sets the same process-wide flag)."""
        f = getattr(self.dg, "use_deterministic_algorithms", None)
        if f is None:
            return {"requested": True, "applied": False, "note": "not available"}
        try:
            try:
                f(True)
            except TypeError:
                f()
            return {"requested": True, "applied": True, "note": None}
        except Exception as e:  # pragma: no cover - depends on the library build
            return {"requested": True, "applied": False, "note": repr(e)[:200]}

    def _set_alignment(self, align: int) -> dict:
        """DeepGEMM's global BLOCK_M cap for contiguous layouts := ALIGN (dg_inplace.DGInplaceBuffers._set_alignment:
        the verify step sets the same process-wide value from the same $DSV41_DG_ALIGN)."""
        f = getattr(self.dg, "set_mk_alignment_for_contiguous_layout", None)
        g = getattr(self.dg, "get_mk_alignment_for_contiguous_layout", None)
        prev = g() if g else None
        if f is not None:
            f(align)
        now = g() if g else None
        if now is not None and int(now) != align:
            raise RuntimeError(f"deep_gemm contiguous alignment is {now} after setting {align}")
        return {"prev": prev, "set": align, "now": now, "settable": f is not None}

    def load(self, which: int, q: torch.Tensor, s: torch.Tensor, arow: torch.Tensor) -> None:
        """A rows `arow` of xq[which] / xs[which] <- the pairs' act_quant bytes (fp8) and fp32 scales (moe_rows'
        DGMaskedMoERows._load)."""
        from runtime.dg_moe import e8m0_to_f32

        self.xq[which].view(torch.uint8).index_copy_(0, arow, q.contiguous().view(torch.uint8))
        s32 = torch.empty(s.shape, dtype=torch.float32, device=s.device)
        e8m0_to_f32(s, s32, self.e8m0_method)
        self.xs[which].index_copy_(0, arow, s32)

    def gemm(self, stage: int, p: str, which: int) -> torch.Tensor:
        """d[p] <- contiguous grouped GEMM of A = (xq[which], xs[which]) against stage `stage`'s whole stack
        (int8 [E, N, K/2] + packed SF), groups from m_indices (dg_inplace.gemm_inplace's call)."""
        from runtime.dg_moe import RECIPE_A, RECIPE_B

        self.fn((self.xq[which], self.xs[which]), (self.w[stage][p], self.sf[stage][p]), self.d[p], self.mi,
                recipe_a=RECIPE_A, recipe_b=RECIPE_B)
        self.n_calls += 1
        return self.d[p]

    def warmup(self) -> None:
        """First-call JIT of both GEMM shapes on real block heads, before any capture; m_indices reset to -1."""
        self.mi.view(self.P, self.align)[:, 0] = torch.arange(self.P, dtype=torch.int32, device=self.dev) % self.E
        for p in PROJS:
            self.gemm(0, p, 2 if p == "w2" else 1)
        self.mi.fill_(-1)

    def nbytes(self) -> dict:
        buf = sum(t.numel() * t.element_size() for t in [*self.xq.values(), *self.xs.values(), *self.d.values(),
                                                         self.mi])
        sf = sum(t.untyped_storage().nbytes() for st in self.sf for t in st.values())
        return {"buffers": buf, "packed_sf": sf}

    def describe(self) -> dict:
        return {"impl": "dg-inplace (resident stacks)", "deep_gemm": getattr(self.dg, "__file__", None),
                "align": self.align, "alignment": self.alignment, "deterministic": self.deterministic,
                "e8m0_to_f32": self.e8m0_method, "a_blocks": self.P, "experts": self.E, "dim": self.D,
                "inter": self.I, "bytes": self.nbytes(), "calls": self.n_calls}


class DraftDGMoE:
    """One stage's MoE on DraftDGBuffers (module docstring, moe_impl "dg-inplace"): gate -> dedup_fused (identity
    group table: the tags are each used slot's expert id on its count rows) -> act_quant scatter -> w1, w3 -> SwiGLU
    -> act_quant -> w2 -> per-row ascending-id combine + shared expert. Device ops only."""

    def __init__(self, owner: "DraftStep", moe: nn.Module, stage: int):
        from runtime.moe_rows import DedupBuffers

        self.owner, self.moe, self.stage = owner, moe, int(stage)
        b = owner.dgm
        self.dbuf = DedupBuffers(b.rows, b.k, owner.dev)
        self.inpl = {"gtab": b.gtab, "H": b.E, "S": 1, "dummy": b.dummy, "has_dummy": True}
        self.last: Dict[str, torch.Tensor] = {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        from runtime import fused_glue as FG
        from runtime.moe_rows import combine_rows, dedup_fused

        o, moe, b, R = self.owner, self.moe, self.owner.dgm, self.owner.R
        g = o.glue
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893, [5, D]
        R_, D = x.shape
        weights, indices = moe.gate(x, None)                           # model.py:894 (drafts are text)
        m = dedup_fused(self.dbuf, indices, weights, b.align, inpl=self.inpl)   # 1 launch on CUDA
        b.mi.view(b.P, b.align)[:, :R_].copy_(m["tags"])              # used slots' count rows -> their expert
        self.last = m
        arow, prow = m["arow"], m["prow"]
        if g is not None and g.on("dg_act"):                           # act_quant + scatter, 1 launch
            FG.launch_act_quant_bcast(x, b.xq[1], b.xs[1], g.code_tab, 1, src_rows=prow, dst_rows=arow)
        else:
            q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)   # model.py:187, the rows once
            b.load(1, q.index_select(0, prow), s.index_select(0, prow), arow)
        g1 = b.gemm(self.stage, "w1", 1)                               # model.py:845
        g3 = b.gemm(self.stage, "w3", 1)                               # model.py:846
        if g is not None and g.on("swiglu_routed"):                    # gather + SwiGLU + w + act_quant, 1 launch
            FG.launch_swiglu_quant(g1, g3, m["pw"], b.xq[2], b.xs[2], g.silu_tab, g.code_tab, b.swiglu_limit,
                                   rows_idx=arow)
        else:
            gate = g1.index_select(0, arow).float()
            up = g3.index_select(0, arow).float()
            lim = b.swiglu_limit
            if lim > 0:                                                # model.py:847-849
                up = torch.clamp(up, min=-lim, max=lim)
                gate = torch.clamp(gate, max=lim)
            h = F.silu(gate) * up                                      # model.py:850
            h = m["pw"].view(-1, 1) * h                                # model.py:852
            h = h.to(x.dtype)                                          # model.py:853
            q2, s2 = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)  # model.py:187 inside w2
            b.load(2, q2, s2, arow)
        out = b.gemm(self.stage, "w2", 2)                              # [P*ALIGN, D] bf16
        sh = o._shared(moe.shared_experts, x)                          # model.py:905
        if g is not None and g.on("moe_combine"):                      # combine + shared + bf16, 1 launch
            y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            FG.launch_moe_combine(out, sh.reshape(R_, D).contiguous(), y, posmat=m["posmat"])
            return y.view(shape)
        y = combine_rows(out, m["posmat"], x)                          # model.py:895-902's order per row
        y += sh
        return y.type_as(x).view(shape)                                # model.py:906


# ------------------------------------------------------------------------------------------------ M1-T11-PORT: glue
class DraftGlue:
    """The fused-glue clusters the draft uses at R = 5 (module docstring, glue_impl "fused"): tables, the picked sum /
    RoPE contraction orders, per-cluster status ("on", "off: <why>", "n/a: <why>") and the call-site operations.
    Built after the MoE; `calibrate()` self-checks every applicable cluster on the draft's shapes against the
    reference ops (eager, syncs; before any capture)."""

    def __init__(self, owner: "DraftStep", log=None):
        from runtime import fused_glue as FG

        self.o, self.R, self.dev = owner, owner.R, owner.dev
        self.log = log or (lambda *a, **k: None)
        R = self.R
        st0 = owner.stages[0]
        a0 = st0.attn
        self.head_dim, self.rope_dim, self.n_heads = a0.head_dim, a0.rope_head_dim, a0.n_local_heads
        self.hc = int(st0.hc_mult)
        self.lim = float(st0.ffn.shared_experts.swiglu_limit)
        self.hc_mode = 0
        self.rope_inv = (0, 0)
        self.status: Dict[str, str] = {}
        self.calib: Dict[str, object] = {}
        mx = (R.fp8_block_size, R.scale_dtype) == (FG.BLOCK, FG.E8) and R.scale_fmt is not None
        dg = owner.moe_impl == "dg-inplace"
        attns = [st.attn for st in owner.stages]
        for c in DRAFT_GLUE_CLUSTERS:
            why = None
            if c != "hc_post" and not mx:
                why = "needs the MXFP8 act_quant globals (block 32, ue8m0, e8m0 scales)"
            elif c in ("dg_act", "swiglu_routed", "moe_combine") and not dg:
                why = "moe_impl ref (reference Experts)"
            elif c == "swiglu_shared":
                ok = all(getattr(getattr(st.ffn.shared_experts, p).weight, "dtype", None) == FG.FP8
                         for st in owner.stages for p in PROJS)
                lims = {float(st.ffn.shared_experts.swiglu_limit) for st in owner.stages}
                if not ok or st0.ffn.shared_experts.w1.out_features % FG.BLOCK or len(lims) != 1:
                    why = "shared experts are not fp8 Linears with a 32-multiple inter dim and one swiglu limit"
            elif c == "o_rope_qdq":
                same = all((a.head_dim, a.rope_head_dim, a.n_local_heads) ==
                           (self.head_dim, self.rope_dim, self.n_heads) for a in attns)
                if not same or self.head_dim % FG.BLOCK or self.rope_dim % 2:
                    why = "attention head layout differs across stages or is not 32-aligned"
            self.status[c] = "on" if why is None else f"n/a: {why}"
        env = {c.strip() for c in os.environ.get(FG.DISABLE_ENV, "").split(",") if c.strip()}
        for c in env & set(DRAFT_GLUE_CLUSTERS):
            if self.status[c] == "on":
                self.status[c] = f"off: {FG.DISABLE_ENV}"
        self.code_tab = FG.scale_code_table(R, self.dev) if mx else None
        self.silu_tab = FG.silu_table(self.dev, self.lim)

    # ---------------------------------------------------------------- status
    def on(self, c: str) -> bool:
        return self.status.get(c) == "on"

    def _off(self, c: str, why: str) -> None:
        if self.on(c):
            self.status[c] = f"off: {why}"
            self.log(f"[draft-glue] {c} OFF: {why}")

    def describe(self) -> dict:
        return {"clusters": dict(self.status), "on": [c for c in self.status if self.on(c)],
                "hc_mode": self.hc_mode, "rope_inv": list(self.rope_inv), "calibration": dict(self.calib)}

    # ---------------------------------------------------------------- call-site operations
    def hc_post(self, x, residual, post, comb) -> torch.Tensor:
        """Block.hc_post (model.py:964-968) for the T rows in one launch (FusedGlue.hc_post)."""
        from runtime.fused_glue import launch_hc_post

        HC, D = residual.size(-2), residual.size(-1)
        T_ = residual.numel() // (HC * D)
        out = torch.empty(residual.shape, dtype=x.dtype, device=x.device)
        launch_hc_post(x.reshape(T_, D).contiguous(), residual.reshape(T_, HC, D).contiguous(),
                       post.reshape(T_, HC).contiguous(), comb.reshape(T_, HC, HC).contiguous(),
                       out.view(T_, HC, D), self.hc_mode)
        return out

    def o_path(self, attn, o: torch.Tensor, freqs_cis: torch.Tensor) -> None:
        """In place on o [1, T, H, head_dim]: inverse RoPE (row t at its own position) + act_quant QDQ, one launch
        (model.py:1070-1073; FusedGlue.o_path's R-row form)."""
        from runtime.fused_glue import launch_o_rope_qdq

        T_ = o.numel() // (attn.n_local_heads * attn.head_dim)
        fr = torch.view_as_real(freqs_cis).reshape(T_, -1, 2).contiguous()
        launch_o_rope_qdq(o.view(T_, -1), fr, self.code_tab, attn.head_dim, attn.rope_head_dim, self.rope_inv)

    def shared_expert(self, ex, x: torch.Tensor) -> torch.Tensor:
        """Expert.forward (model.py:843-853, weights None): w1, w3 on their GEMM path, SwiGLU + bf16 + w2's act_quant
        in one launch, w2 routed as linear() would (patches.fp8_linear_prequant) -- FusedGlue._shared_from_q."""
        from runtime import fused_glue as FG
        from runtime.patches import fp8_linear_prequant

        I = ex.w1.weight.size(0)
        lead = tuple(x.shape[:-1])
        g2 = ex.w1(x).reshape(-1, I)
        u2 = ex.w3(x).reshape(-1, I)
        q = torch.empty(g2.size(0), I, dtype=FG.FP8, device=x.device)
        s = torch.empty(g2.size(0), I // FG.BLOCK, dtype=FG.E8, device=x.device)
        FG.launch_swiglu_quant(g2, u2, None, q, s, self.silu_tab, self.code_tab, float(ex.swiglu_limit))
        return fp8_linear_prequant(q.view(*lead, I), s.view(*lead, I // FG.BLOCK), ex.w2.weight)

    # ---------------------------------------------------------------- self-check
    @_cpu_default
    def _rand(self, g, *shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device="cpu") * scale).to(dtype)

    def calibrate(self) -> dict:
        """Self-check every "on" cluster on the draft's shapes (module docstring); a mismatch turns it off."""
        g = torch.Generator(device="cpu").manual_seed(20261107)
        cal = self.calib
        try:
            if self.on("hc_post"):
                self._pick_hc(cal, g)
            if self.on("o_rope_qdq"):
                self._pick_rope(cal, g)
            if self.on("swiglu_shared"):
                why = self._verify_shared(g)
                cal["verify_swiglu_shared"] = why or "equal"
                if why:
                    self._off("swiglu_shared", f"self-check: {why}")
            dgc = [c for c in ("dg_act", "swiglu_routed", "moe_combine") if self.on(c)]
            if dgc:
                for c, why in self._verify_dg(g, dgc).items():
                    cal[f"verify_{c}"] = why or "equal"
                    if why:
                        self._off(c, f"self-check: {why}")
        except Exception as e:  # a launch / build failure: everything left on goes off, the reference ops run
            for c in list(self.status):
                self._off(c, f"self-check raised {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
            cal["error"] = repr(e)[:300]
        return cal

    def _pick_hc(self, cal, g) -> None:
        from runtime.fused_glue import HC_SUM_MODES

        st0, B, D = self.o.stages[0], self.o.block, self.o.stages[0].attn_norm.dim
        modes = HC_SUM_MODES if self.hc == 4 else (0,)
        mism = {m: 0 for m in modes}
        for _ in range(4):
            x = self._rand(g, 1, B, D, scale=2.0).to(self.dev)
            res = self._rand(g, 1, B, self.hc, D, scale=2.0).to(self.dev)
            post = self._rand(g, 1, B, self.hc, dtype=torch.float32).abs().to(self.dev)
            comb = self._rand(g, 1, B, self.hc, self.hc, dtype=torch.float32).abs().to(self.dev)
            ref = st0.hc_post(x, res, post, comb)
            for m in modes:
                self.hc_mode = m
                got = self.hc_post(x, res, post, comb)
                mism[m] += int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        ok = [m for m in modes if mism[m] == 0]
        cal["hc_post"] = {"mismatches": mism, "picked": ok[0] if ok else None}
        self.hc_mode = ok[0] if ok else 0
        if not ok:
            self._off("hc_post", f"no 4-term sum order matches Block.hc_post at {B} rows ({mism})")

    def _pick_rope(self, cal, g) -> None:
        from runtime.fused_glue import ROPE_MODES

        o_, R = self.o, self.R
        attn = o_.stages[0].attn
        B, H, hd, rd = o_.block, attn.n_local_heads, attn.head_dim, attn.rope_head_dim
        cases = []
        for _ in range(3):
            o = self._rand(g, 1, B, H, hd, scale=3.0).to(self.dev)
            pos = torch.randint(0, o_.max_pos + 1, (B,), generator=g, device="cpu").to(self.dev)
            f = freqs_rows(o_.tables[0], pos)
            ref = o.clone()
            R.apply_rotary_emb(ref[..., -rd:], f, True)                              # model.py:1070
            R.act_quant(ref.view(1, B, attn.n_local_groups, -1), R.fp8_block_size, R.scale_fmt, R.scale_dtype,
                        True)                                                        # model.py:1072-1073
            cases.append((o, f, ref))
        mism = {}
        for modes in itertools.product(ROPE_MODES, ROPE_MODES):
            self.rope_inv = modes
            n = 0
            for o, f, ref in cases:
                got = o.clone()
                self.o_path(attn, got, f)
                n += int((got.view(torch.int16) != ref.view(torch.int16)).sum())
            mism[f"{modes[0]}{modes[1]}"] = n
        ok = [m for m in itertools.product(ROPE_MODES, ROPE_MODES) if mism[f"{m[0]}{m[1]}"] == 0]
        cal["o_rope_qdq"] = {"mismatches": mism, "picked": list(ok[0]) if ok else None}
        self.rope_inv = ok[0] if ok else (0, 0)
        if not ok:
            self._off("o_rope_qdq", f"no RoPE contraction order matches the reference o path ({mism})")

    def _verify_shared(self, g) -> Optional[str]:
        o_ = self.o
        D = o_.stages[0].attn_norm.dim
        with o_._gemv_scope():
            for i, st in enumerate(o_.stages):
                ex = st.ffn.shared_experts
                x = self._rand(g, 1, o_.block, D, scale=1.5).to(self.dev)
                ref = ex(x)
                got = self.shared_expert(ex, x)
                if got.shape != ref.shape or not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                    return f"stage {i}: {int((got.view(torch.int16) != ref.view(torch.int16)).sum())} bf16 mismatches"
        return None

    def _verify_dg(self, g, clusters) -> Dict[str, Optional[str]]:
        """Each dg cluster alone vs all dg clusters off (the reference glue), the stage MoE end to end on a random x,
        every stage: bit-equal outputs."""
        o_ = self.o
        D = o_.stages[0].attn_norm.dim
        saved = dict(self.status)
        out: Dict[str, Optional[str]] = {c: None for c in clusters}
        try:
            with o_._gemv_scope():
                for i, moe in enumerate(o_.moes):
                    x = self._rand(g, 1, o_.block, D, scale=1.5).to(self.dev)
                    for c in clusters:
                        self.status[c] = "off: calibrating"
                    base = moe(x).clone()
                    for c in clusters:
                        self.status[c] = "on"
                        got = moe(x)
                        self.status[c] = "off: calibrating"
                        if out[c] is None and not torch.equal(got.view(torch.int16), base.view(torch.int16)):
                            out[c] = f"stage {i}: {int((got.view(torch.int16) != base.view(torch.int16)).sum())} " \
                                     f"bf16 mismatches vs the reference glue"
        finally:
            self.status.update(saved)
        return out


# ------------------------------------------------------------------------------------------------ the step
class DraftStep:
    """The static DSpark draft step over `model.mtp` (module docstring). `main_hidden` / `tok`: existing device
    tensors to alias (e.g. the 6-row StaticStep's `main_hidden` and `t_out`), else own buffers. `ctx_rowwise`: compute
    the context rows' main_proj/main_norm/wkv one row at a time (M = 1, the reference decode's shapes) instead of
    once on all rows -- a debug knob for a GEMM that is not row-invariant. fp8_impl / moe_impl / glue_impl /
    attn_impl: the M1-T11-PORT switches (module docstring; "ref" = the T11 step)."""

    def __init__(self, model: nn.Module, *, ctx_rows: int = CTX_ROWS, main_hidden: Optional[torch.Tensor] = None,
                 tok: Optional[torch.Tensor] = None, ctx_rowwise: bool = False, fp8_impl: str = "ref",
                 moe_impl: str = "ref", glue_impl: str = "ref", attn_impl: str = "ref", log=None,
                 _allow_cpu: bool = False):
        R = _ref()
        self.R, self.model = R, model
        self.log = log or (lambda *a, **k: print(*a, file=sys.stderr, flush=True))
        for name, v, ok in (("fp8_impl", fp8_impl, DRAFT_FP8_IMPLS), ("moe_impl", moe_impl, DRAFT_MOE_IMPLS),
                            ("glue_impl", glue_impl, DRAFT_GLUE_IMPLS), ("attn_impl", attn_impl, DRAFT_ATTN_IMPLS)):
            if v not in ok:
                raise ValueError(f"DraftStep {name} {v!r} not in {ok} (draft_impls_for maps the verify's names)")
        self.fp8_impl, self.moe_impl, self.glue_impl, self.attn_impl = fp8_impl, moe_impl, glue_impl, attn_impl
        stages = list(getattr(model, "mtp", ()))
        if not stages:
            raise ValueError("DraftStep: model.mtp is empty (build_and_load(..., load_mtp=True))")
        mt = getattr(model, "runtime_mtp", None)
        if mt is None or not mt.stacks:
            raise ValueError("DraftStep: model.runtime_mtp missing (the draft experts' stacks come from "
                             "runtime/mtp_loader.attach_mtp)")
        self.stages = stages
        self.dev = model.embed.weight.device
        if self.dev.type != "cuda" and not _allow_cpu:
            raise ValueError(f"DraftStep needs the model on CUDA (got {self.dev})")
        first, last = stages[0], stages[-1]
        self.block = int(first.block_size)
        for i, st in enumerate(stages):
            if int(st.block_size) != self.block:
                raise ValueError(f"mtp.{i}: block_size {st.block_size} != {self.block}")
            if float(st.temperature) != 0.0:
                raise ValueError(f"mtp.{i}.temperature = {st.temperature}: the draft step is greedy (model.py:1113, "
                                 f"1156); runtime/mtp_loader sets 0")
            if st.attn.compress_ratio != 0:
                raise ValueError(f"mtp.{i}: compress ratio {st.attn.compress_ratio}")
            if st.attn.window_kv_cache.size(0) < 1:
                raise ValueError(f"mtp.{i}: no window cache")
        self.win = first.attn.window_size
        if any(st.attn.window_size != self.win for st in stages):
            raise ValueError("draft stages' window sizes differ")
        self.max_pos = int(model.max_seq_len) - 1
        self.C = int(ctx_rows)
        if not 1 <= self.C <= self.win - self.block:
            raise ValueError(f"ctx_rows {ctx_rows} outside [1, {self.win - self.block}]")
        self.ctx_rowwise = bool(ctx_rowwise)
        d = self.dev
        i64 = dict(dtype=torch.long, device=d)
        self.hidden_dim = int(first.main_proj.in_features)
        if first.main_proj.weight.size(1) != self.hidden_dim:
            raise ValueError(f"main_proj in_features {self.hidden_dim} != weight {tuple(first.main_proj.weight.shape)}")
        self.dtype = model.embed.weight.dtype
        if main_hidden is None:
            main_hidden = torch.zeros(self.C, self.hidden_dim, dtype=self.dtype, device=d)
        if tuple(main_hidden.shape) != (self.C, self.hidden_dim) or main_hidden.device != d:
            raise ValueError(f"main_hidden {tuple(main_hidden.shape)} on {main_hidden.device}: want "
                             f"({self.C}, {self.hidden_dim}) on {d}")
        if tok is None:
            tok = torch.zeros(self.C, **i64)
        if tok.numel() != self.C or tok.dtype != torch.long or tok.device != d:
            raise ValueError(f"tok must be {self.C} int64 on {d}")
        self.main_hidden, self.tok = main_hidden, tok
        self.a_pos = torch.zeros(1, **i64)
        self.ctx_count = torch.zeros(1, **i64)
        self.out_ids = torch.zeros(1, self.block + 1, **i64)
        self.drafts = self.out_ids[0, 1:]
        V = model.head.weight.size(0)
        self.logits = torch.zeros(self.block, V, dtype=torch.float32, device=d)
        self.confidence = torch.zeros(self.block, dtype=torch.float32, device=d)
        self.ar_ctx = torch.arange(self.C, **i64)
        self.ar_blk = torch.arange(self.block, **i64)
        self.ar_idx = torch.arange(self.win + self.block, **i64)
        self.tables = [torch.view_as_real(st.attn.freqs_cis) for st in stages]
        self.glue: Optional[DraftGlue] = None
        self.dgm: Optional[DraftDGBuffers] = None
        self._fm_on = False
        self.attn_rec: dict = {"impl": attn_impl, "on": attn_impl == "ref"}
        self._tlen: Optional[torch.Tensor] = None
        if fp8_impl == "gemv":
            from runtime import patches

            patches.fp8_linear_gemv(True)  # inert outside a fp8_gemv_scope (this step opens its own in _body)
        self._build_moe(R, mt)
        if glue_impl == "fused":           # tables + build-time self-check (eager; syncs; before any capture)
            self.glue = DraftGlue(self, log=self.log)
            self.glue.calibrate()
        if attn_impl == "flashmla":
            self._flashmla_selfcheck()
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured_fills: Optional[bool] = None
        self.n_replays = 0

    # ---------------------------------------------------------------- build
    def _build_moe(self, R, mt) -> None:
        st0 = self.stages[0]
        moe0 = st0.ffn
        k = int(moe0.n_activated_experts)
        for i, st in enumerate(self.stages):
            if int(st.ffn.n_activated_experts) != k or any(e is None for e in st.ffn.experts):
                raise ValueError(f"mtp.{i}: MoE shape differs or experts missing")
        self.topk = k
        S = self.block * k
        self.n_slots = S
        for i, st in enumerate(self.stages):
            stk = mt.stacks[i]
            for e in (0, len(st.ffn.experts) - 1):  # the experts' params must be views into the stacks
                for p in PROJS:
                    lin = getattr(st.ffn.experts[e], p)
                    if lin.weight.data_ptr() != stk[p][0][e].data_ptr() or \
                            lin.scale.data_ptr() != stk[p][1][e].data_ptr():
                        raise ValueError(f"mtp.{i} expert {e} {p}: not bound to its stack row")
        self.staging_w: Dict[str, torch.Tensor] = {}
        self.staging_s: Dict[str, torch.Tensor] = {}
        self.staged_experts: List[nn.Module] = []
        self.moe_xb = self.moe_wb = None
        if self.moe_impl == "dg-inplace":   # M1-T11-PORT: DeepGEMM over the resident stacks, no staging
            self.dgm = DraftDGBuffers(self, mt)
            self.moes = [DraftDGMoE(self, st.ffn, i) for i, st in enumerate(self.stages)]
            return
        from runtime.static_step import _expert_view, _linear_view

        ex0 = moe0.experts[0]
        for p in PROJS:
            w, s = mt.stacks[0][p]
            self.staging_w[p] = torch.zeros((S, *w.shape[1:]), dtype=w.dtype, device=self.dev)
            self.staging_s[p] = torch.zeros((S, *s.shape[1:]), dtype=s.dtype, device=self.dev)
        lim = moe0.shared_experts.swiglu_limit
        for j in range(S):
            lins = {}
            for p in PROJS:
                lin0 = getattr(ex0, p)
                lins[p] = _linear_view(R, self.staging_w[p][j].view(torch.float4_e2m1fn_x2),
                                       self.staging_s[p][j].view(torch.float8_e8m0fnu),
                                       lin0.in_features, lin0.out_features)
            self.staged_experts.append(_expert_view(R, lins["w1"], lins["w2"], lins["w3"], lim))
        D = moe0.dim
        self.moe_xb = torch.zeros(S * self.block, D, dtype=self.dtype, device=self.dev)
        self.moe_wb = torch.zeros(S * self.block, 1, dtype=torch.float32, device=self.dev)
        self.moes = [DraftMoE(self, st.ffn, mt.stacks[i]) for i, st in enumerate(self.stages)]

    def _flashmla_selfcheck(self) -> None:
        """attn_impl "flashmla": load the ops (no vllm), then runtime/attn_flashmla.self_check on the draft's
        attention shapes (heads, head_dim, softmax_scale, sink of stage 0; win + block index slots, win + block kv
        rows) against the reference sparse_attn and an fp64 oracle. Fails -> the reference sparse_attn runs."""
        from runtime import attn_flashmla as AFM
        from runtime import patches

        a = self.stages[0].attn
        rec = {"impl": "flashmla", "on": False, "reason": None, "check": None, "load": None}
        if any((s.attn.n_local_heads, s.attn.head_dim) != (a.n_local_heads, a.head_dim) for s in self.stages):
            rec["reason"] = "draft stages' attention shapes differ"
        else:
            try:
                AFM.load_ops()
                rec["load"] = AFM.loaded()
                chk = AFM.self_check(patches.original_sparse_attn(), self.dev, a.n_local_heads, a.head_dim,
                                     a.softmax_scale, self.win, self.block, self.win + self.block,
                                     attn_sink=a.attn_sink)
                rec["check"] = chk
                if chk["ok"]:
                    rec["on"] = True
                else:
                    rec["reason"] = "self-check: " + str(chk["reason"])
            except Exception as e:
                rec["reason"] = f"{type(e).__name__}: {e}"
        if not rec["on"]:
            self.log(f"[draft_step] attn_impl flashmla is OFF: {rec['reason']} (the reference sparse_attn runs)")
        self._fm_on = rec["on"]
        self.attn_rec = rec

    def staging_bytes(self) -> int:
        if self.dgm is not None:
            b = self.dgm.nbytes()
            return b["buffers"] + b["packed_sf"]
        return sum(t.numel() * t.element_size() for t in list(self.staging_w.values()) +
                   list(self.staging_s.values()) + [self.moe_xb, self.moe_wb])

    def _gemv_scope(self):
        """fp8_impl "gemv": the scope in which runtime/patches.py's linear wrapper routes the draft's fp8 Linears to
        the GEMV (static_step.py #16's mechanism); a no-op scope otherwise."""
        from runtime.patches import fp8_gemv_scope

        return fp8_gemv_scope(self.fp8_impl == "gemv")

    def _shared(self, ex, x: torch.Tensor) -> torch.Tensor:
        """The shared expert (model.py:905): the reference module, or the fused SwiGLU path (glue swiglu_shared)."""
        if self.glue is not None and self.glue.on("swiglu_shared"):
            return self.glue.shared_expert(ex, x)
        return ex(x)

    # ---------------------------------------------------------------- body
    def _main_x(self, first, mh):
        """main_norm(main_proj(main_hidden)) (model.py:1133) for the C context rows."""
        if not self.ctx_rowwise:
            return first.main_norm(first.main_proj(mh))
        return torch.cat([first.main_norm(first.main_proj(mh[:, i:i + 1])) for i in range(self.C)], dim=1)

    def _ctx_write(self, s: int, attn, main_x, pos_ctx, valid) -> None:
        """model.py:1041-1044 + 1067 for the C context rows of stage s, predicated on `valid`."""
        R = self.R
        if self.ctx_rowwise:
            kv = torch.cat([attn.kv_norm(attn.wkv(main_x[:, i:i + 1])) for i in range(self.C)], dim=1)
        else:
            kv = attn.kv_norm(attn.wkv(main_x))                                  # model.py:1042
        fpos = torch.clamp(pos_ctx, 0, self.max_pos)                             # invalid rows: any in-range row
        R.apply_rotary_emb(kv[..., -attn.rope_head_dim:], freqs_rows(self.tables[s], fpos))  # 1041, 1043
        R.act_quant(kv, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)     # 1044
        ring = attn.window_kv_cache
        slots = torch.remainder(pos_ctx, self.win)                               # 1067: start_pos % win
        old = ring[:1].index_select(1, slots)
        ring[:1].index_copy_(1, slots, torch.where(valid.view(1, -1, 1), kv, old))

    def _attn(self, s: int, attn, x, freqs, idx):
        """DSparkAttention.forward's decode branch (model.py:1056-1077) without the ring write (done in _ctx_write)."""
        R = self.R
        rd = attn.rope_head_dim
        qr = attn.q_norm(attn.wq_a(x))                                           # 1059
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))    # 1060
        R.apply_rotary_emb(q[..., -rd:], freqs)                                  # 1061
        kv = attn.kv_norm(attn.wkv(x))                                           # 1062
        R.apply_rotary_emb(kv[..., -rd:], freqs)                                 # 1063
        R.act_quant(kv, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)     # 1064
        kv = torch.cat([attn.window_kv_cache[:1], kv], dim=1)                    # 1068
        o = self._sparse_attn(attn, q, kv, idx)                                  # 1069
        if self.glue is not None and self.glue.on("o_rope_qdq"):                 # 1070-1073 in one launch
            self.glue.o_path(attn, o, freqs)
            o = o.view(1, self.block, attn.n_local_groups, -1)
        else:
            R.apply_rotary_emb(o[..., -rd:], freqs, True)                        # 1070
            o = o.view(1, self.block, attn.n_local_groups, -1)                   # 1072
            R.act_quant(o, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)  # 1073
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)  # 1074
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)                              # 1075
        return attn.wo_b(o.flatten(2))                                           # 1076

    def _sparse_attn(self, attn, q, kv, idx):
        """model.py:1069: the reference kernel, or (attn_impl flashmla, self-check passed) FlashMLA with the per-row
        length min(win, A'+1) + block (the entries after it are the -1 tail)."""
        if self._fm_on:
            from runtime.attn_flashmla import sparse_attn_flashmla, use_topk_length

            return sparse_attn_flashmla(q, kv, attn.attn_sink, idx, attn.softmax_scale,
                                        self._tlen if use_topk_length() else None)
        return self.R.sparse_attn(q, kv, attn.attn_sink, idx, attn.softmax_scale)

    def _hc_post(self, st, x, residual, post, comb):
        if self.glue is not None and self.glue.on("hc_post"):
            return self.glue.hc_post(x, residual, post, comb)
        return st.hc_post(x, residual, post, comb)

    def _block(self, s: int, st, x, pre_mix, freqs, idx):
        """Block.forward (model.py:983-996) with the static attention and the static draft MoE."""
        residual = x
        attn_pre, attn_post, attn_comb = st.hc_mixes(x, st.hc_attn_fn, st.hc_attn_scale, st.hc_attn_base)
        x = st.attn_norm(st.hc_pre(x, pre_mix))
        x = self._attn(s, st.attn, x, freqs, idx)
        x = self._hc_post(st, x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = st.hc_mixes(x, st.hc_ffn_fn, st.hc_ffn_scale, st.hc_ffn_base)
        x = st.ffn_norm(st.hc_pre(x, attn_pre))
        x = self.moes[s](x)
        x = self._hc_post(st, x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def _body(self) -> None:
        with self._gemv_scope():
            self._body_ops()

    def _body_ops(self) -> None:
        R = self.R
        first, last = self.stages[0], self.stages[-1]
        bonus = self.tok.index_select(0, bonus_index(self.ctx_count))           # t_n [1]
        mh = self.main_hidden.view(1, self.C, -1)
        h, main_x = first.forward_embed(mh, bonus)                               # model.py:1131-1138
        if self.ctx_rowwise:
            main_x = self._main_x(first, mh)
        pos_ctx, valid = ctx_positions(self.a_pos, self.ctx_count, self.ar_ctx)
        for s, st in enumerate(self.stages):
            self._ctx_write(s, st.attn, main_x, pos_ctx, valid)
        pos_blk = self.a_pos + 1 + self.ar_blk                                   # model.py:1057
        idx = dspark_window_idxs_dev(self.a_pos, self.win, self.block, self.ar_idx)
        if self._fm_on:                                                          # FlashMLA per-row index count
            n = torch.clamp(self.a_pos.view(1) + 1, max=self.win) + self.block
            self._tlen = n.int().expand(self.block).contiguous()
        pre_mix = R.make_identity_pre_mix(h, first.hc_mult)                      # model.py:1280
        for s, st in enumerate(self.stages):                                     # model.py:1281-1282
            freqs = freqs_rows(self.tables[s], pos_blk)
            h, pre_mix = self._block(s, st, h, pre_mix, freqs, idx)
        out_ids, logits, conf = last.forward_head(h, pre_mix, bonus)             # model.py:1285, 1140-1159
        self.out_ids.copy_(out_ids)
        self.logits.copy_(logits.view(self.block, -1))
        self.confidence.copy_(conf.view(self.block))

    # ---------------------------------------------------------------- public
    def set_inputs(self, a_pos: Optional[int] = None, ctx_count: Optional[int] = None,
                   main_hidden: Optional[torch.Tensor] = None, tok=None) -> None:
        """Host-side setter (fills/copies, no sync). A' + block must stay <= max_pos; context rows at positions >= 1."""
        with torch.inference_mode():
            if a_pos is not None or ctx_count is not None:
                if a_pos is None or ctx_count is None:
                    raise ValueError("set a_pos and ctx_count together")
                a, c = int(a_pos), int(ctx_count)
                if not 0 <= c <= self.C:
                    raise ValueError(f"ctx_count {c} outside [0, {self.C}]")
                if not 0 <= a <= self.max_pos - self.block:
                    raise ValueError(f"A' {a}: block positions A'+1..A'+{self.block} must be <= max_pos {self.max_pos}")
                if c and a - c + 1 < 1:
                    raise ValueError(f"context rows from position {a - c + 1}: position 0 belongs to the prefill")
                self.a_pos.fill_(a)
                self.ctx_count.fill_(c)
            if main_hidden is not None:
                mh = main_hidden.reshape(-1, self.hidden_dim)
                self.main_hidden[: mh.size(0)].copy_(mh)
            if tok is not None:
                if isinstance(tok, torch.Tensor):
                    t = tok.reshape(-1).to(device=self.dev, dtype=torch.long)
                    self.tok[: t.numel()].copy_(t)
                else:
                    ids = [int(x) for x in tok]
                    self.tok[: len(ids)].copy_(torch.tensor(ids, dtype=torch.long))

    def forward_static(self) -> torch.Tensor:
        """Run the step eagerly from the static buffers. Returns `drafts` [5] (a view of out_ids)."""
        with torch.inference_mode():
            self._body()
        return self.drafts

    def step(self, a_pos: int, ctx_count: int, graph: bool = False, main_hidden=None, tok=None) -> torch.Tensor:
        self.set_inputs(a_pos, ctx_count, main_hidden, tok)
        if graph:
            if self.graph is None:
                raise RuntimeError("replay before capture()")
            self.graph.replay()
            self.n_replays += 1
            return self.drafts
        return self.forward_static()

    def replay(self, a_pos: Optional[int] = None, ctx_count: Optional[int] = None) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("replay before capture()")
        if a_pos is not None or ctx_count is not None:
            self.set_inputs(a_pos, ctx_count)
        self.graph.replay()
        self.n_replays += 1
        return self.drafts

    def capture(self, fills: bool = False, warmup: int = 2, pool=None) -> dict:
        """Warm-up steps on a side stream, then ONE graph of `_body`. Both run with ctx_count = 0 (the context write
        is then a value no-op on the rings) at the current A'; ctx_count is left at 0."""
        import time

        det = getattr(torch.utils, "deterministic", None)
        prev_fill = getattr(det, "fill_uninitialized_memory", None)
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        try:
            if det is not None and prev_fill is not None:
                det.fill_uninitialized_memory = bool(fills)
            with torch.inference_mode():
                self.ctx_count.zero_()
                s = torch.cuda.Stream(device=self.dev)
                s.wait_stream(torch.cuda.current_stream(self.dev))
                with torch.cuda.stream(s):
                    for _ in range(warmup):
                        self._body()
                torch.cuda.current_stream(self.dev).wait_stream(s)
                torch.cuda.synchronize(self.dev)
                t_warm = time.perf_counter() - t0
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    self._body()
                torch.cuda.synchronize(self.dev)
        finally:
            if det is not None and prev_fill is not None:
                det.fill_uninitialized_memory = prev_fill
        self.graph = g
        self.captured_fills = bool(fills)
        self.n_replays = 0
        return {"warmup_s": t_warm, "capture_s": time.perf_counter() - t0 - t_warm, "fills": bool(fills)}

    def release_graph(self) -> None:
        self.graph = None
        self.captured_fills = None

    def describe(self) -> dict:
        return {"stages": len(self.stages), "block": self.block, "ctx_rows": self.C, "win": self.win,
                "topk": self.topk, "n_slots": self.n_slots, "staging_bytes": self.staging_bytes(),
                "hidden_dim": self.hidden_dim, "ctx_rowwise": self.ctx_rowwise, "max_pos": self.max_pos,
                "captured": self.graph is not None,
                "impls": {"fp8": self.fp8_impl, "moe": self.moe_impl, "glue": self.glue_impl, "attn": self.attn_impl},
                "moe": self.dgm.describe() if self.dgm is not None else {"impl": "ref (staged reference Experts)"},
                "glue": self.glue.describe() if self.glue is not None else {"impl": "ref"},
                "attn": {k: v for k, v in self.attn_rec.items() if k != "check"} if self.attn_rec else None}


@contextlib.contextmanager
def no_sync(enabled: bool = True):
    """torch.cuda.set_sync_debug_mode('error') for the block (CUDA only)."""
    if not (enabled and torch.cuda.is_available()):
        yield
        return
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")
