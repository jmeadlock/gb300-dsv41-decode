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
"""
from __future__ import annotations

import contextlib
import sys
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

PROJS = ("w1", "w3", "w2")
CTX_ROWS = 6  # verify rows: 1 bonus + 5 drafts (design 1.3)


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
        y += moe.shared_experts(x)                                     # model.py:905
        return y.type_as(x).view(shape)                                # model.py:906


# ------------------------------------------------------------------------------------------------ the step
class DraftStep:
    """The static DSpark draft step over `model.mtp` (module docstring). `main_hidden` / `tok`: existing device
    tensors to alias (e.g. the 6-row StaticStep's `main_hidden` and `t_out`), else own buffers. `ctx_rowwise`: compute
    the context rows' main_proj/main_norm/wkv one row at a time (M = 1, the reference decode's shapes) instead of
    once on all rows -- a debug knob for a GEMM that is not row-invariant."""

    def __init__(self, model: nn.Module, *, ctx_rows: int = CTX_ROWS, main_hidden: Optional[torch.Tensor] = None,
                 tok: Optional[torch.Tensor] = None, ctx_rowwise: bool = False, _allow_cpu: bool = False):
        R = _ref()
        self.R, self.model = R, model
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
        self._build_moe(R, mt)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured_fills: Optional[bool] = None
        self.n_replays = 0

    # ---------------------------------------------------------------- build
    def _build_moe(self, R, mt) -> None:
        from runtime.static_step import _expert_view, _linear_view

        st0 = self.stages[0]
        moe0 = st0.ffn
        k = int(moe0.n_activated_experts)
        for i, st in enumerate(self.stages):
            if int(st.ffn.n_activated_experts) != k or any(e is None for e in st.ffn.experts):
                raise ValueError(f"mtp.{i}: MoE shape differs or experts missing")
        self.topk = k
        S = self.block * k
        self.n_slots = S
        ex0 = moe0.experts[0]
        self.staging_w: Dict[str, torch.Tensor] = {}
        self.staging_s: Dict[str, torch.Tensor] = {}
        for p in PROJS:
            w, s = mt.stacks[0][p]
            self.staging_w[p] = torch.zeros((S, *w.shape[1:]), dtype=w.dtype, device=self.dev)
            self.staging_s[p] = torch.zeros((S, *s.shape[1:]), dtype=s.dtype, device=self.dev)
        lim = moe0.shared_experts.swiglu_limit
        self.staged_experts: List[nn.Module] = []
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
        self.moes: List[DraftMoE] = []
        for i, st in enumerate(self.stages):
            stk = mt.stacks[i]
            for e in (0, len(st.ffn.experts) - 1):  # the experts' params must be views into the stacks
                for p in PROJS:
                    lin = getattr(st.ffn.experts[e], p)
                    if lin.weight.data_ptr() != stk[p][0][e].data_ptr() or \
                            lin.scale.data_ptr() != stk[p][1][e].data_ptr():
                        raise ValueError(f"mtp.{i} expert {e} {p}: not bound to its stack row")
            self.moes.append(DraftMoE(self, st.ffn, stk))

    def staging_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in list(self.staging_w.values()) +
                   list(self.staging_s.values()) + [self.moe_xb, self.moe_wb])

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
        o = R.sparse_attn(q, kv, attn.attn_sink, idx, attn.softmax_scale)        # 1069
        R.apply_rotary_emb(o[..., -rd:], freqs, True)                            # 1070
        o = o.view(1, self.block, attn.n_local_groups, -1)                       # 1072
        R.act_quant(o, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)      # 1073
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)  # 1074
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)                              # 1075
        return attn.wo_b(o.flatten(2))                                           # 1076

    def _block(self, s: int, st, x, pre_mix, freqs, idx):
        """Block.forward (model.py:983-996) with the static attention and the static draft MoE."""
        residual = x
        attn_pre, attn_post, attn_comb = st.hc_mixes(x, st.hc_attn_fn, st.hc_attn_scale, st.hc_attn_base)
        x = st.attn_norm(st.hc_pre(x, pre_mix))
        x = self._attn(s, st.attn, x, freqs, idx)
        x = st.hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = st.hc_mixes(x, st.hc_ffn_fn, st.hc_ffn_scale, st.hc_ffn_base)
        x = st.ffn_norm(st.hc_pre(x, attn_pre))
        x = self.moes[s](x)
        x = st.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def _body(self) -> None:
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
                "captured": self.graph is not None}


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
