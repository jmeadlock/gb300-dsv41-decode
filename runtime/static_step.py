"""T8: the fixed-shape 1-row decode step, eager (`forward_static`) or as one CUDA graph per pos parity (`replay`).

    st = StaticStep(model, pool, max_pos=4095)       # model = runtime.loader.build_and_load(...), pool = its ExpertPool
    st.capture()                                     # dummy prefill, 2 warm-ups per parity on a side stream, g_even/g_odd
    model(prompt, 0)                                 # the real prefill stays the reference's, eager
    logits = st.step(tok, L, graph=True)             # [1, vocab] fp32 static buffer; st.tok holds the greedy id

Spec: docs/T8-T9A-DESIGN.md section 2. Every row of table 2.1 is implemented here as a replacement that keeps the
reference arithmetic: the same reference functions/modules are called in the same order on the same values, only
host-side ints become device tensors and dynamic lengths become fixed lengths (cites are reference model.py lines):

  #1  generate.py token/logits per step -> static `tok[1,1]` int64, `pos[1]` int64, `logits[1,V]` fp32; the step
      ends with `tok.copy_(argmax)` (model.py:1292) and `pos += 1`. D2H only when the caller copies `logits`.
  #2  engram.py:167,169 -> `engram_hash_static`: `index_copy_` at device pos, `positions = pos`.
  #3  model.py:768, 552 -> `freqs_rows(table, pos)`: index_select of the fp32 view of `freqs_cis` (a value copy).
  #4  model.py:709 prefill branch -> not here: prefill stays the reference's own forward, eager.
  #5  model.py:719 ring write -> `window_kv_cache[:1].index_copy_(1, pos % win, kv)`.
  #6  model.py:721, 409-426 -> `window_topk_idxs_dev`: (pos%win + 1 + arange(win)) % win, then -1 where > pos. Same
      ORDER as the host function (sparse_attn's online softmax accumulates in index order).
  #7  model.py:745, 730 compress_len == 0 -> unreachable for pos >= 1 at ratio <= 2; `step` refuses pos < 1.
  #8  model.py:764, 776-779 dynamic cat -> stopgap: cat with the compressed cache's first W_r = (max_pos+1)//r rows
      (fixed), indices unchanged; unreferenced rows are never gathered by sparse_attn (kernel.py:360-362).
  #9  model.py:579-581 topk width 128+min(512, T/r) -> k = min(index_topk, W_r) over the W_r-wide score, positions
      >= T/r at -inf, then the reference's own sort + where(< compress_lens, +offset, -1). For T/r <= index_topk the
      reference keeps every reachable position and so do we; the extra slots are -1 and whole -1 blocks of 64 are
      exact no-ops in sparse_attn (kernel.py:363-380: exp(m-m)=1, sums of exact 0). T/r > index_topk: identical only
      if the k-th and (k+1)-th reachable scores differ (design risk 6) -- see tests/test_static_cpu.py.
  #10 model.py:555, 564-568 -> index_k[:1, :W_r] + masked_fill(arange(W_r) >= compress_lens, -inf); compress_lens on
      the device. Score bits of unreachable rows never matter (replaced by -inf, NaN included).
  #11 model.py:597-611 candidate blocks -> `select_candidate_blocks_static` = the reference function at fixed width
      W_r, compress_lens a device tensor, repeat_interleave(int) spelled as expand+flatten (same values, cannot sync).
  #12 model.py:477-484, 537, 747-748 ratio-2 pooling only when (pos+1) % 2 == 0 -> two graphs chosen by pos parity;
      inside each, the reference `Compressor.forward` is called with `representative_pos(parity)`: at ratio 2 the
      decode branch reads start_pos only through `start_pos % 2` and `(start_pos + 1) % 2` (model.py:477-478), so the
      ops and values are the reference's. Ratio 1 ignores start_pos (model.py:461-462). Ratios > 2 are refused.
  #13 model.py:542, 547, 757, 762 -> device `pos + 1 - r` rope row, `index_copy_` at `pos // r`.
  #14 model.py:891-906 -> `StaticMoE` (installed through runtime/patches.py `static_moe`): device gate -> sort ids
      ascending (the reference's accumulation order) -> Triton pointer-table gather into static [k,N,K/2] staging
      (pointer tables built once per layer from runtime/expert_pool.py: hot -> HBM slot, cold -> UVA host address) ->
      k fixed reference `Expert` evaluations on the staging slots, fp32 sum in slot order -> + shared expert.
      T9a-1: `moe_impl="dg-masked"` swaps StaticMoE for runtime/dg_moe.py `DGMaskedMoE` (same gate/sort/gather, then
      3 DeepGEMM masked grouped GEMMs over the same staging instead of 3k fp4_gemm; not bit-identical by design).
      `moe_impl="ref"` (default) is exactly the T8 step. T9a-2: `moe_impl="dg-inplace"` (runtime/dg_inplace.py)
      grows the pool's weight tensors by topk spare slots per layer, then runs DeepGEMM's contiguous grouped GEMM
      over [H+S, N, K/2] in place: one staging launch per layer moves only cold rows (+ the packed scales).
  #15 kernel.py wrapper allocations -> the graph's private pool; capture runs with fill_uninitialized_memory off
      unless `capture(fills=True)` (the L2 uninitialized-read detector records the NaN fills as kernels).
  #16 T9b-1, model.py:196-205 (`linear()`'s fp8 branch: act_quant + `fp8_gemm` at block 32) -> with
      `fp8_impl="gemv"` the same act_quant, then runtime/gemv_mxfp8.py's Triton MXFP8 GEMV (installed through
      runtime/patches.py `fp8_linear_gemv`, routed only inside this step's `_active()` scope: eager static steps,
      warm-ups and capture; the prefill and the eager reference never see it). 290 calls/step on the real model (40 x
      wq_a, wq_b, wkv, wo_b, shared w1/w3/w2 + 8 indexer wq_b + 2 Engram wkv). Not bit-identical by design (the
      32-wide block sum's order; K-block order and scales are the reference's). `fp8_impl="ref"` (default) is T8.
  #17 T9b-2, docs/HANDOFF-T9B2-GLUE-R1.md -> with `glue_impl="fused"` the zero/low-risk glue clusters of
      docs/GLUE-AUDIT.md run as fused kernels (runtime/fused_glue.py: MoE gather/act/SwiGLU/combine, shared-expert
      SwiGLU, mHC post, o-path inverse RoPE + QDQ, q RoPE in the wq_b GEMV store). Bit-identical to `glue_impl="ref"`
      (default: the ops above, unchanged) with the same moe/fp8 impls; each cluster is self-checked at construction
      and stays on the reference ops if it fails (`st.glue.describe()`). The sub-steps are methods (`_q_proj`,
      `_o_path`, `_hc_post`; StaticMoE `_gather`, `_shared`) so the census can attribute launches.
  #18 T9b-3a, docs/HANDOFF-T9B3A-ATTN-BOUND.md -> with `attn_impl="bounded"` the compressed layers' `sparse_attn`
      (kernel.py:392-403) runs runtime/kernels/sparse_attn_bounded.py: the reference kernel text with its 64-index
      block loop bounded by a per-row device count `nblk = ceil((win + min(k, compress_lens)) / 64)` computed where
      the indices are built (`_compress_kv`, once per (ratio, k) per step, published with `sa.topk_idxs` for the
      layers that reuse them). Rows #8/#9 leave every slot past that count -1, and whole -1 blocks are exact no-ops
      in the kernel (kernel.py:358-380; the module docstring walks the ops), so the bound skips work, not values.
      Ratio-0 layers (128 window indices = 2 blocks) keep the reference call. Self-checked at build against the
      reference at the full and the narrow width (`attn_describe()`); a failing check leaves `ref` in force.
      `attn_impl="ref"` (default) is exactly the T9b-2 step. MoE staging gather (#14/#17): 16-byte vector copies
      (runtime/static_kernels.py), so every source row address must be 16-B aligned: checked at build.
  #19 T9b-3c, docs/HANDOFF-T9B3C-FLASHMLA.md -> with `attn_impl="flashmla"` EVERY layer's `sparse_attn` (ratio 0 and
      compressed) runs runtime/attn_flashmla.py: FlashMLA's `sparse_prefill_fwd` (loaded from vllm's .so through
      torch.ops, never `import vllm`) over the same bf16 kv / int32 indices / sink / scale. Compressed layers pass the
      per-row count `topk_length = win + min(k, compress_lens)`, computed where the indices are built (as #18's block
      count, published with `sa.topk_idxs`); ratio-0 layers pass none. A3-class drift (not bit-identical; q stays
      bf16). Build-time self-check vs the reference and an fp64 oracle (`attn_describe()`); a failing check or load
      leaves the reference in force.

Not computed: `main_hiddens` (model.py:1267-1269) -- MTP-only, never feeds the logits; the loader's overlay sets
dspark_target_layer_ids=() and `StaticStep` refuses a model that has target layers.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

PARITIES = (0, 1)
SUPPORTED_RATIOS = (0, 1, 2)
MOE_IMPLS = ("ref", "dg-masked", "dg-inplace")  # T9a-1: "ref" = StaticMoE (bit-identical), "dg-masked" =
#                                               runtime/dg_moe.py; T9a-2: "dg-inplace" = runtime/dg_inplace.py
FP8_IMPLS = ("ref", "gemv")       # T9b-1: "ref" = reference fp8_gemm (bit-identical), "gemv" = runtime/gemv_mxfp8.py
GLUE_IMPLS = ("ref", "fused")     # T9b-2: "ref" = the glue ops as below, "fused" = runtime/fused_glue.py (bit-identical)
ATTN_IMPLS = ("ref", "bounded", "flashmla")  # T9b-3a: "ref" = reference sparse_attn, "bounded" = runtime/kernels
#                                              (bit-identical); T9b-3c: "flashmla" = runtime/attn_flashmla.py (A3-class)
GATHER_ALIGN = 16                 # T9b-3a: the vectorized staging gather moves 16 B per lane (static_kernels.py)
GATHER_IMPL_ENV = "DSV41_GATHER_IMPL"  # "u8" = the byte-per-lane gather of T8/T9b-2 (bisecting), default "vec"
DUMMY_PROMPT = (0, 1)


# ------------------------------------------------------------------------------------------------ pure index math
def parity_of(pos: int) -> int:
    """Which graph runs position `pos` (g_even = 0, g_odd = 1)."""
    return int(pos) % 2


def compressor_pools(pos: int, ratio: int) -> bool:
    """model.py:477: a ratio-r compressor pools at decode position pos iff (pos + 1) % r == 0."""
    return (int(pos) + 1) % ratio == 0


def representative_pos(parity: int) -> int:
    """A start_pos >= 1 with the given parity. At ratio 2 the reference decode branch (model.py:476-482) depends on
    start_pos only through `start_pos % 2` and `(start_pos + 1) % 2`; at ratio 1 not at all (461-462)."""
    return 2 + (int(parity) & 1)


def window_topk_idxs_dev(pos: torch.Tensor, win: int, ar_win: torch.Tensor) -> torch.Tensor:
    """Design #6: the decode branch of get_window_topk_idxs (model.py:422-426) from a device `pos` [1] int64:
    oldest = pos % win + 1, slots oldest..win-1 then 0..oldest-1 (same order), -1 where slot > pos. -> int32 [1,1,win]."""
    idx = torch.remainder(torch.remainder(pos, win) + 1 + ar_win, win)
    idx = torch.where(idx > pos, -1, idx)
    return idx.int().view(1, 1, win)


def freqs_rows(table_real: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Design #3/#13: rows `idx` of a complex `freqs_cis` table through its fp32 view (`table_real` =
    view_as_real(freqs_cis), [S, rd/2, 2]) -> complex64 [len(idx), rd/2]. Same values as freqs_cis[p:p+1] /
    freqs_cis[p].unsqueeze(0) (a copy)."""
    return torch.view_as_complex(table_real.index_select(0, idx))


def select_candidate_blocks_static(logits: torch.Tensor, compress_lens, topk_blocks: int,
                                   block_size: int) -> torch.Tensor:
    """model.py:584-611 line for line at a fixed width, `compress_lens` an int or a device tensor; the final
    `repeat_interleave(block_size, -1)` is written as unsqueeze/expand/flatten (identical values, never a host sync)."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.unsqueeze(-1).expand(*keep.shape, block_size).flatten(-2)[..., :width]


def indexer_topk_static(index_score: torch.Tensor, compress_lens, k: int, offset: int) -> torch.Tensor:
    """model.py:579-581 with the width fixed: k = min(index_topk, W) is the caller's; unreachable scores must already
    be -inf (design #9/#10)."""
    idxs = index_score.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def engram_hash_static(st, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Design #2: NgramHashState.forward (engram.py:159-184) for one text token, batch 1, `pos` a device [1] int64.
    The cache write `cache[:, start_pos:start_pos+1] = compressed` becomes index_copy_; `positions` = pos. The rest is
    the reference's code (token_mask None: text only)."""
    compressed = st.token_map[tok]                                      # engram.py:164, [1, 1]
    st.cache[:1].index_copy_(1, pos, compressed)                        # engram.py:167
    positions = pos.view(1, 1)                                          # engram.py:169
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(st.layout.max_ngram_size):                       # engram.py:171-175
        source = st.cache[:1].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == st.DEAD)
        tokens.append(torch.where(blocked, st.pad_id, source))
    tokens = torch.stack(tokens, dim=-1)
    products = tokens.unsqueeze(2) * st.multipliers                     # engram.py:179-184
    rolling, hashes = products[..., 0], []
    for i in range(1, st.layout.max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % st.primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + st.offsets


def _ref():
    m = sys.modules.get("model")
    if m is None:
        from runtime.refmodel import import_reference
        m = import_reference()
    return m


# ------------------------------------------------------------------------------------------------ per-step scalars
class _StepScalars:
    """Device scalars derived from `pos` once per step (lazily, so a graph only records the ones it uses)."""

    def __init__(self, owner: "StaticStep"):
        self.o = owner
        self.c: Dict[tuple, torch.Tensor] = {}

    def _get(self, key, fn):
        v = self.c.get(key)
        if v is None:
            v = self.c[key] = fn()
        return v

    @property
    def pos(self):
        return self.o.pos

    def p_win(self):
        return self._get(("p_win",), lambda: torch.remainder(self.o.pos, self.o.win))

    def win_idx(self):
        return self._get(("win_idx",), lambda: window_topk_idxs_dev(self.o.pos, self.o.win, self.o.ar_win))

    def div(self, r):  # write index pos // r (model.py:547, 762)
        return self._get(("div", r), lambda: torch.div(self.o.pos, r, rounding_mode="floor"))

    def clen(self, r):  # compress_lens = end_pos // r (model.py:568)
        return self._get(("clen", r), lambda: torch.div(self.o.pos + 1, r, rounding_mode="floor"))

    def nblk(self, r, k, win):  # #18: ceil((win + min(k, compress_lens)) / 64), int32 [1, 1]
        from runtime.kernels.sparse_attn_bounded import nblk_static

        return self._get(("nblk", r, k, win), lambda: nblk_static(self.clen(r), k, win))

    def tlen(self, r, k, win):  # #19: win + min(k, compress_lens), int32 [rows]
        from runtime.attn_flashmla import topk_length_static

        return self._get(("tlen", r, k, win), lambda: topk_length_static(self.clen(r), k, win))

    def unreach(self, r):  # arange(W_r) >= compress_lens
        return self._get(("unreach", r), lambda: self.o.ar[r] >= self.clen(r))

    def freqs(self, table):  # freqs_cis[pos : pos+1] (model.py:552, 768)
        k = table.data_ptr()
        return self._get(("f", k), lambda: freqs_rows(self.o.tables[k], self.o.pos))

    def freqs_lat(self, table, r):  # freqs_cis[pos + 1 - r].unsqueeze(0) (model.py:542, 757)
        k = table.data_ptr()
        return self._get(("fl", k, r), lambda: freqs_rows(self.o.tables[k], self.o.pos + (1 - r)))


# ------------------------------------------------------------------------------------------------ static MoE (#14)
def _linear_view(R, w: torch.Tensor, s: torch.Tensor, in_features: int, out_features: int) -> nn.Module:
    """A reference `Linear` over existing storage (no allocation): fp4 weight view + E8M0 scale view, `weight.scale`
    attached as model.py:222 does, no bias."""
    lin = R.Linear.__new__(R.Linear)
    nn.Module.__init__(lin)
    lin.in_features, lin.out_features = in_features, out_features
    lin.weight = nn.Parameter(w, requires_grad=False)
    lin.scale = nn.Parameter(s, requires_grad=False)
    lin.weight.scale = lin.scale
    lin.register_parameter("bias", None)
    return lin


def _expert_view(R, w1, w2, w3, swiglu_limit: float) -> nn.Module:
    """A reference `Expert` (model.py:832-853) whose three Linears are views; its forward is the reference's."""
    ex = R.Expert.__new__(R.Expert)
    nn.Module.__init__(ex)
    ex.w1, ex.w2, ex.w3 = w1, w2, w3
    ex.swiglu_limit = swiglu_limit
    return ex


class StaticMoE:
    """One layer's static MoE (design #14). Called by the patched `MoE.forward` (runtime/patches.py `static_moe`)
    only while its owner is running a step."""

    def __init__(self, owner: "StaticStep", moe: nn.Module, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]):
        self.owner, self.moe, self.layer, self.ptabs = owner, moe, layer, ptabs

    @property
    def active(self) -> bool:
        return self.owner.active

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        o, moe = self.owner, self.moe
        glue = o.glue
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893
        if glue is not None and glue.on("router"):
            ids, w = glue.route(moe, x)                                # T9b-2 R2 #4: gate + sort, 2 launches + 3
        else:
            weights, indices = moe.gate(x, None)                       # model.py:894 (text: no image mask)
            ids, order = torch.sort(indices[0])                        # ascending id = model.py:897 loop order
            w = weights[0, order]
        if glue is not None and glue.on("gather"):
            self._gather_fused(ids)                                    # T9b-2 #11: one launch (+ ids_step)
        else:
            if o.counters:
                self._record_ids(ids)
            self._gather(ids)
        y = torch.zeros_like(x, dtype=torch.float32)                   # model.py:895
        for j, ex in enumerate(o.staged_experts):                      # fixed k, slot order = ascending id
            y += ex(x, w[j].view(1, 1))                                # model.py:902 (1 row: idx = [0])
        y += self._shared(x)                                           # model.py:905
        return y.type_as(x).view(shape)                                # model.py:906

    def _gather_fused(self, ids: torch.Tensor) -> None:
        self.owner.glue.gather(self, ids)

    def _record_ids(self, ids: torch.Tensor) -> None:
        self.owner.ids_step[self.layer].copy_(ids)

    def _gather(self, ids: torch.Tensor) -> None:
        from runtime.static_kernels import gather_rows

        o = self.owner
        for key, tab in self.ptabs.items():                            # one launch per (proj, weight|scale)
            gather_rows(tab, ids, o.staging[key], o.row_bytes[key])

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        glue = self.owner.glue
        if glue is not None and glue.on("swiglu_shared"):
            return glue.shared_expert(self.moe.shared_experts, x)      # T9b-2 #2 (shared)
        return self.moe.shared_experts(x)


# ------------------------------------------------------------------------------------------------ the step
class StaticStep:
    """Fixed-shape 1-row decode step over the reference model (module docstring for the row-by-row mapping).

    model: the reference Transformer on CUDA (batch-1 caches). pool: runtime.expert_pool.ExpertPool the model's
    routed experts are bound to, or None (pointer tables then come from the experts' own parameters). max_pos: the
    largest decode position this step serves; it fixes W_r = (max_pos+1)//r (compressed-KV cat length, index-score
    width). marker: launch `t8_step_marker` first in every step. counters: record each layer's routed ids per step on
    the device and accumulate cold calls (for first_light's cold-share report; does not touch the logits). moe_impl:
    "ref" (StaticMoE, bit-identical) or "dg-masked" (runtime/dg_moe.py; dg_deterministic -> deep_gemm's
    use_deterministic_algorithms(True)). fp8_impl: "ref" (reference fp8_gemm) or "gemv" (runtime/gemv_mxfp8.py,
    design #16)."""

    def __init__(self, model: nn.Module, pool=None, *, max_pos: Optional[int] = None, marker: bool = False,
                 counters: bool = True, moe_impl: str = "ref", dg_deterministic: bool = True,
                 fp8_impl: str = "ref", glue_impl: str = "ref", glue_opts: Optional[dict] = None,
                 attn_impl: str = "ref", _allow_cpu: bool = False):
        R = _ref()
        from runtime import patches

        if moe_impl not in MOE_IMPLS:
            raise ValueError(f"moe_impl {moe_impl!r} not in {MOE_IMPLS}")
        if fp8_impl not in FP8_IMPLS:
            raise ValueError(f"fp8_impl {fp8_impl!r} not in {FP8_IMPLS}")
        if glue_impl not in GLUE_IMPLS:
            raise ValueError(f"glue_impl {glue_impl!r} not in {GLUE_IMPLS}")
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl {attn_impl!r} not in {ATTN_IMPLS}")
        if fp8_impl == "gemv" and ((R.fp8_block_size, R.scale_dtype) != (32, torch.float8_e8m0fnu)
                                    or R.scale_fmt is None):
            raise ValueError(f"fp8_impl gemv assumes MXFP8 (block 32, ue8m0) activations; the reference has "
                             f"fp8_block_size={R.fp8_block_size} scale_fmt={R.scale_fmt} scale_dtype={R.scale_dtype}")
        self.moe_impl, self.dg_deterministic = moe_impl, bool(dg_deterministic)
        self.fp8_impl = fp8_impl
        self.glue_impl = glue_impl
        self.attn_impl = attn_impl
        self.attn_bound: Optional[dict] = None  # #18: the self-check record when attn_impl == "bounded"
        self._attn_on = False                   # #18: bounded kernel in force (self-check passed)
        self._idx_nblk: Optional[torch.Tensor] = None  # #18: the block count published with sa.topk_idxs
        self.attn_flashmla: Optional[dict] = None  # #19: the self-check record when attn_impl == "flashmla"
        self._fm_on = False                        # #19: FlashMLA in force (load + self-check passed)
        self._idx_tlen: Optional[torch.Tensor] = None  # #19: the topk_length published with sa.topk_idxs
        self.glue = None  # runtime.fused_glue.FusedGlue when glue_impl == "fused" (built after the MoE buffers)
        self.dg = None  # runtime.dg_moe.DGBuffers when moe_impl == "dg-masked"
        self.R, self.model, self.pool = R, model, pool
        self.dev = model.embed.weight.device
        if self.dev.type != "cuda" and not _allow_cpu:
            raise ValueError(f"StaticStep needs the model on CUDA (got {self.dev})")
        if getattr(model, "target_layer_ids", ()):
            raise ValueError("StaticStep: model has dspark target layers (main_hiddens); build with the loader overlay")
        msl = int(model.max_seq_len)
        self.max_pos = msl - 1 if max_pos is None else int(max_pos)
        if not (len(DUMMY_PROMPT) + 4 <= self.max_pos < msl):
            raise ValueError(f"max_pos {self.max_pos} must be in [{len(DUMMY_PROMPT) + 4}, {msl - 1}]")
        self.marker, self.counters = bool(marker), bool(counters)
        layers = list(model.layers)
        attns = [l.attn for l in layers]
        self.win = attns[0].window_size
        ratios = sorted({a.compress_ratio for a in attns})
        bad = [r for r in ratios if r not in SUPPORTED_RATIOS]
        if bad:
            raise ValueError(f"compress ratios {bad}: two parity graphs cover ratio <= 2 only (design #12)")
        if any(a.window_kv_cache.size(0) < 1 or a.window_size != self.win for a in attns):
            raise ValueError("window caches/sizes differ across layers")
        d = self.dev
        i64 = dict(dtype=torch.long, device=d)
        self.ratios = [r for r in ratios if r]
        self.W = {r: (self.max_pos + 1) // r for r in self.ratios}
        self.ar = {r: torch.arange(self.W[r], **i64) for r in self.ratios}
        self.ar_win = torch.arange(self.win, **i64)
        self.tables: Dict[int, torch.Tensor] = {}
        for a in attns:
            self.tables.setdefault(a.freqs_cis.data_ptr(), torch.view_as_real(a.freqs_cis))
            if a.indexer is not None:
                if a.indexer.freqs_cis is None:
                    a.indexer.freqs_cis = a.freqs_cis  # what model.py:734-735 sets at the first indexer call
                self.tables.setdefault(a.indexer.freqs_cis.data_ptr(), torch.view_as_real(a.indexer.freqs_cis))
        # static inputs/outputs (#1)
        self.tok = torch.zeros(1, 1, **i64)
        self.pos = torch.zeros(1, **i64)
        self.logits = torch.zeros(1, model.head.weight.size(0), dtype=torch.float32, device=d)
        self.marker_ctr = torch.zeros(1, dtype=torch.int32, device=d)
        # static MoE (#14)
        self.topk = layers[0].ffn.n_activated_experts
        self._build_moe(R, layers)
        patches.static_moe(True)
        if self.fp8_impl == "gemv":
            patches.fp8_linear_gemv(True)  # design #16: inert outside this step's _active() scope
        self.active = False
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.captured_fills: Optional[bool] = None
        self._dev_pos: Optional[int] = None
        self.n_counted = 0
        self._c: Optional[_StepScalars] = None
        if self.glue_impl == "fused":  # #17: tables + build-time self-check (eager; syncs; before any capture)
            from runtime.fused_glue import FusedGlue

            self.glue = FusedGlue(self, **(glue_opts or {}))
        if self.attn_impl == "bounded":  # #18: install the routed wrapper, then the build-time self-check (eager)
            patches.sparse_attn_bounded(True)
            self._attn_selfcheck(attns)
        if self.attn_impl == "flashmla":  # #19: load the ops (no vllm), then the build-time self-check (eager)
            self._flashmla_selfcheck(attns)

    def _flashmla_selfcheck(self, attns) -> None:
        """#19: runtime/attn_flashmla.self_check on this model's attention shapes (heads, head_dim, softmax_scale and
        attn_sink of the first compressed layer, else layer 0; window + k index slots; window + W_max kv rows) against
        the reference sparse_attn and an fp64 oracle. Fails -> the step stays on the reference (reason recorded)."""
        from runtime import attn_flashmla as AFM
        from runtime import patches

        comp = [a for a in attns if a.compress_ratio]
        a = comp[0] if comp else attns[0]
        ks = [min(x.indexer.index_topk, self.W[x.compress_ratio]) for x in comp if x.indexer is not None]
        k = max(ks) if ks else (max(self.W.values()) if self.W else 0)
        kv_len = self.win + (max(self.W.values()) if self.W else 0)
        rec = {"impl": "flashmla", "on": False, "reason": None, "check": None, "load": None}
        try:
            AFM.load_ops()
            rec["load"] = AFM.loaded()
            chk = AFM.self_check(patches.original_sparse_attn(), self.dev, a.n_local_heads, a.head_dim,
                                 a.softmax_scale, self.win, k, kv_len, attn_sink=a.attn_sink)
            rec["check"] = chk
            if chk["ok"]:
                rec["on"] = True
            else:
                rec["reason"] = "self-check: " + str(chk["reason"])
        except Exception as e:
            rec["reason"] = f"{type(e).__name__}: {e}"
        if not rec["on"]:
            print(f"[static_step] attn_impl flashmla is OFF: {rec['reason']} (the reference sparse_attn runs)",
                  file=sys.stderr, flush=True)
        self._fm_on = rec["on"]
        self.attn_flashmla = rec

    def _attn_selfcheck(self, attns) -> None:
        """#18: runtime/kernels/sparse_attn_bounded.self_check on this model's attention shapes (heads, head_dim,
        softmax_scale and attn_sink of the first compressed layer; window + k index slots; window + W_max kv rows)
        against the reference sparse_attn under the wrapper. Picks the first TileLang loop spelling that compiles and
        is bit-equal; none -> the bounded path stays off (reason recorded) and the step is attn_impl "ref"'s."""
        from runtime import patches
        from runtime.kernels import sparse_attn_bounded as SAB

        comp = [a for a in attns if a.compress_ratio]
        rec = {"impl": "bounded", "on": False, "variant": None, "reason": None, "check": None}
        if not comp:
            rec["reason"] = "no compressed layers"
        else:
            a = comp[0]
            ks = [min(x.indexer.index_topk, self.W[x.compress_ratio]) for x in comp if x.indexer is not None]
            k = max(ks) if ks else max(self.W.values())
            try:
                chk = SAB.self_check(patches.original_sparse_attn(), self.dev, a.n_local_heads, a.head_dim,
                                     a.softmax_scale, self.win, k, self.win + max(self.W.values()),
                                     attn_sink=a.attn_sink)
                rec["check"] = chk
                if chk["variant"] is not None:
                    rec.update(on=True, variant=chk["variant"])
                else:
                    rec["reason"] = "self-check: no variant compiled and matched (" + "; ".join(
                        f"{v}: {r}" for v, r in chk["results"].items()) + ")"
            except Exception as e:
                rec["reason"] = f"self-check raised {type(e).__name__}: {e}"
        if not rec["on"]:
            print(f"[static_step] attn_impl bounded is OFF: {rec['reason']} (the reference sparse_attn runs)",
                  file=sys.stderr, flush=True)
        self._attn_on = rec["on"]
        self.attn_bound = rec

    def attn_describe(self) -> dict:
        """#18: {"impl", "on", "variant", "reason", "check"} (`impl` "ref" -> nothing else); #19 flashmla: {"impl",
        "on", "reason", "check", "load"}."""
        if self.attn_impl == "flashmla":
            return dict(self.attn_flashmla or {}, impl="flashmla")
        if self.attn_impl != "bounded":
            return {"impl": "ref", "on": False}
        return dict(self.attn_bound or {}, impl="bounded")

    # ---------------------------------------------------------------- build
    def _build_moe(self, R, layers) -> None:
        from runtime.expert_pool import TIER_HBM

        d, pool, k = self.dev, self.pool, self.topk
        moe0 = layers[0].ffn
        ex0 = next(e for e in moe0.experts if e is not None)
        shapes = {p: tuple(getattr(ex0, p).weight.shape) for p in ("w1", "w3", "w2")}
        sshapes = {p: tuple(getattr(ex0, p).scale.shape) for p in ("w1", "w3", "w2")}
        self.staging: Dict[Tuple[str, str], torch.Tensor] = {}
        self.row_bytes: Dict[Tuple[str, str], int] = {}
        for p in ("w1", "w3", "w2"):
            n, kh = shapes[p]
            self.staging[(p, "weight")] = torch.zeros(k, n, kh, dtype=torch.int8, device=d)
            self.staging[(p, "scale")] = torch.zeros(k, *sshapes[p], dtype=torch.uint8, device=d)
            self.row_bytes[(p, "weight")] = n * kh
            self.row_bytes[(p, "scale")] = sshapes[p][0] * sshapes[p][1]
        lim = moe0.shared_experts.swiglu_limit
        self.staged_experts: List[nn.Module] = []
        for j in range(k):
            lins = {}
            for p in ("w1", "w3", "w2"):
                lin0 = getattr(ex0, p)
                lins[p] = _linear_view(R, self.staging[(p, "weight")][j].view(torch.float4_e2m1fn_x2),
                                       self.staging[(p, "scale")][j].view(torch.float8_e8m0fnu),
                                       lin0.in_features, lin0.out_features)
            self.staged_experts.append(_expert_view(R, lins["w1"], lins["w2"], lins["w3"], lim))
        moe_cls = StaticMoE
        if self.moe_impl == "dg-masked":  # T9a-1: same staging + pointer tables, routed GEMMs on DeepGEMM
            from runtime.dg_moe import DGBuffers, DGMaskedMoE

            self.dg = DGBuffers(self, shapes, sshapes, swiglu_limit=lim, deterministic=self.dg_deterministic)
            moe_cls = DGMaskedMoE
        elif self.moe_impl == "dg-inplace":  # T9a-2: contiguous grouped GEMM over the hot pool + spare slots
            from runtime.dg_inplace import DGInplaceBuffers, DGInplaceMoE

            if pool is None:
                raise ValueError("moe_impl dg-inplace needs the model's ExpertPool")
            added = pool.add_spare_slots(k, self.model)  # before the pointer tables below (storage moves)
            self.dg = DGInplaceBuffers(self, shapes, sshapes, pool=pool, swiglu_limit=lim,
                                       deterministic=self.dg_deterministic)
            self.dg.spare_added_B = added
            moe_cls = DGInplaceMoE
        n_layers, n_exp = len(layers), moe0.n_routed_experts
        cold = torch.zeros(n_layers, n_exp, dtype=torch.long)
        gather_vec = os.environ.get(GATHER_IMPL_ENV, "vec") != "u8"
        if gather_vec:
            bad = {k_: v for k_, v in self.row_bytes.items() if v % GATHER_ALIGN}
            if bad:
                raise ValueError(f"staging row bytes {bad} not multiples of {GATHER_ALIGN} (16-B vector gather); "
                                 f"{GATHER_IMPL_ENV}=u8 uses the byte gather")
        self.moes: list = []  # StaticMoE (ref) or runtime.dg_moe.DGMaskedMoE (dg-masked), one per layer
        for l, layer in enumerate(layers):
            moe = layer.ffn
            if moe.n_activated_experts != k or moe.n_routed_experts != n_exp or any(e is None for e in moe.experts):
                raise ValueError(f"layer {l}: MoE shape differs or experts missing (world_size > 1?)")
            ptabs = {}
            for p in ("w1", "w3", "w2"):
                for kind in ("weight", "scale"):
                    ptrs = []
                    for e in range(n_exp):
                        lin = getattr(moe.experts[e], p)
                        t = lin.weight if kind == "weight" else lin.scale
                        if not t.is_contiguous() or t.numel() * t.element_size() != self.row_bytes[(p, kind)]:
                            raise ValueError(f"L{l} E{e} {p}.{kind}: not a contiguous {self.row_bytes[(p, kind)]} B row")
                        ptr = t.data_ptr()
                        if pool is not None:
                            tier, slot = pool.slot[(l, e)]
                            if tier == TIER_HBM:
                                src = pool.hbm[l][p][0 if kind == "weight" else 1][slot]
                            else:
                                src = pool.cold_view(l, e, p, kind, cuda=d.type == "cuda")
                                cold[l, e] = 1
                            if src.data_ptr() != ptr:
                                raise ValueError(f"L{l} E{e} {p}.{kind}: expert param is not bound to its pool slot")
                            ptr = src.data_ptr()
                        if ptr % GATHER_ALIGN and gather_vec:
                            raise ValueError(f"L{l} E{e} {p}.{kind}: source row at {ptr:#x} is not {GATHER_ALIGN}-B "
                                             f"aligned (16-B vector gather); {GATHER_IMPL_ENV}=u8 uses the byte gather")
                        ptrs.append(ptr)
                    ptabs[(p, kind)] = torch.tensor(ptrs, dtype=torch.int64, device=d)
            sm = moe_cls(self, moe, l, ptabs)
            moe._static_moe = sm
            self.moes.append(sm)
        self.cold_tab = cold.to(d)
        self.ids_step = torch.zeros(n_layers, k, dtype=torch.long, device=d)
        self.cold_acc = torch.zeros(n_layers, dtype=torch.long, device=d)

    # ---------------------------------------------------------------- body
    @contextlib.contextmanager
    def _active(self):
        from runtime.patches import fp8_gemv_scope

        prev, self.active = self.active, True
        try:
            with fp8_gemv_scope(self.fp8_impl == "gemv"):  # design #16
                yield
        finally:
            self.active = prev

    def _indexer(self, ix, x, qr, latent, offset, qq=None):
        """Indexer.forward (model.py:527-581), decode, fixed width (design #9-#11, #13). qq: act_quant(qr) when the
        fused q path made it (#17 R2: the indexer's wq_b reuses it)."""
        R, sa, c, g = self.R, self.R.shared_attn, self._c, self.glue
        ratio, rd = ix.compress_ratio, ix.rope_head_dim
        if ix.owns_k and latent is not None:                                   # model.py:537-547
            k = ix.k_norm(ix.wk(latent))
            R.apply_rotary_emb(k[..., -rd:], c.freqs_lat(ix.freqs_cis, ratio))
            R.fp4_act_quant(k, R.fp4_block_size, True)
            ix.k_cache[:1].index_copy_(1, c.div(ratio), k)
        if ix.owns_k:
            sa.index_k = ix.k_cache
        qb = g.linear_pre(qq, ix.wq_b, qr) if g is not None else ix.wq_b(qr)
        q = qb.unflatten(-1, (ix.n_local_heads, ix.index_head_dim))            # model.py:551-553
        R.apply_rotary_emb(q[..., -rd:], c.freqs(ix.freqs_cis))
        R.fp4_act_quant(q, R.fp4_block_size, True)
        W = self.W[ratio]
        index_k = sa.index_k[:1, :W]                                           # model.py:555 at fixed width
        weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        clen = c.clen(ratio)
        index_score.masked_fill_(c.unreach(ratio), -torch.inf)                 # #10
        if ix.is_candidate_source:                                             # model.py:570-576
            sa.candidates = select_candidate_blocks_static(index_score, clen, ix.candidate_topk_blocks,
                                                           ix.candidate_block_size)
        elif ix.uses_candidates:
            index_score = index_score.masked_fill(~sa.candidates, -torch.inf)
        if g is not None and g.on("indexer_post"):                            # #17 R2 #14: sort + where, 1 launch
            return g.indexer_topk(index_score, clen, min(ix.index_topk, W), offset)
        return indexer_topk_static(index_score, clen, min(ix.index_topk, W), offset)  # #9

    def _compress_kv(self, attn, x, qr, parity, offset, qq=None):
        """Attention._compress_kv + _compress_topk_idxs (model.py:723-764), decode."""
        R, sa, c = self.R, self.R.shared_attn, self._c
        ratio, rd = attn.compress_ratio, attn.rope_head_dim
        latent = None
        if attn.is_kv_source:
            latent = attn.compressor(x, representative_pos(parity))           # #12 (model.py:748)
            sa.compress_kv = attn.compress_kv_cache
        if attn.is_index_source:                                              # model.py:726-738 (len > 0, #7)
            idxs = self._indexer(attn.indexer, x, qr, latent, offset, qq)
            sa.topk_idxs = idxs
            if self._attn_on:                                                 # #18: count where idxs are built
                ix = attn.indexer
                self._idx_nblk = c.nblk(ix.compress_ratio, min(ix.index_topk, self.W[ix.compress_ratio]), offset)
            if self._fm_on:                                                   # #19: the same count, unrounded
                ix = attn.indexer
                self._idx_tlen = c.tlen(ix.compress_ratio, min(ix.index_topk, self.W[ix.compress_ratio]), offset)
        else:
            idxs = sa.topk_idxs
        if latent is not None:                                                # model.py:752-762
            R.apply_rotary_emb(latent[..., -rd:], c.freqs_lat(attn.freqs_cis, ratio))
            R.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            attn.compress_kv_cache[:1].index_copy_(1, c.div(ratio), latent)  # #13
        return sa.compress_kv[:1, : self.W[ratio]], idxs                      # #8 stopgap: fixed length

    def _attn(self, attn, x, parity, xq=None):
        """Attention.forward (model.py:766-791), decode. xq: act_quant(x) when the fused hc_norm made it (#17 R2:
        wq_a and wkv take it instead of quantizing x twice)."""
        c = self._c
        freqs_cis = c.freqs(attn.freqs_cis)                                   # #3
        qr, qq = self._q_path(attn, x, xq)                                    # model.py:771
        q = self._q_proj(attn, qr, freqs_cis, qq)
        self._kv_path(attn, x, xq, freqs_cis)                                 # _window_kv, model.py:706-708, 719
        kv = attn.window_kv_cache[:1]                                         # model.py:720
        topk_idxs = c.win_idx()                                               # #6
        nblk = tlen = None
        if attn.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(attn, x, qr, parity, kv.size(1), qq)
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
            nblk = self._idx_nblk if self._attn_on else None                  # #18 (None: reference call)
            tlen = self._idx_tlen if self._fm_on else None                    # #19
        o = self._sparse_attn(q, kv, attn, topk_idxs, nblk, tlen)
        o = self._o_path(attn, o, freqs_cis)
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return attn.wo_b(o.flatten(2))

    def _q_path(self, attn, x, xq):
        """qr = q_norm(wq_a(x)) (model.py:771) -> (qr, act_quant(qr) or None) (#17 R2 #6: one launch for both)."""
        g = self.glue
        if g is not None and g.on("q_path"):
            return g.q_path(attn, x, xq)
        qa = g.linear_pre(xq, attn.wq_a, x) if g is not None else attn.wq_a(x)
        return attn.q_norm(qa), None

    def _kv_path(self, attn, x, xq, freqs_cis):
        """kv_norm(wkv(x)), RoPE, act_quant QDQ, the ring write at pos % win (model.py:706-708, 719; #5) (#17 R2 #3:
        everything after wkv in one launch)."""
        R, c, g = self.R, self._c, self.glue
        if g is not None and g.on("kv_path"):
            g.kv_path(attn, x, xq, freqs_cis, c.p_win())
            return
        kv = attn.kv_norm(g.linear_pre(xq, attn.wkv, x) if g is not None else attn.wkv(x))
        R.apply_rotary_emb(kv[..., -attn.rope_head_dim:], freqs_cis)
        R.act_quant(kv, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)
        attn.window_kv_cache[:1].index_copy_(1, c.p_win(), kv)

    def _sparse_attn(self, q, kv, attn, topk_idxs, nblk, tlen=None):
        """model.py:781 (#18: inside the bounded scope when a block count is published; #19: FlashMLA on every layer
        when its self-check passed, `tlen` on compressed layers)."""
        if self._fm_on:
            from runtime.attn_flashmla import sparse_attn_flashmla, use_topk_length

            if not use_topk_length():
                tlen = None
            elif attn.compress_ratio and tlen is None:
                raise RuntimeError("flashmla: compressed layer without a published topk_length")
            return sparse_attn_flashmla(q, kv, attn.attn_sink, topk_idxs, attn.softmax_scale, tlen)
        if nblk is None:
            return self.R.sparse_attn(q, kv, attn.attn_sink, topk_idxs, attn.softmax_scale)
        from runtime.patches import attn_bound_scope

        with attn_bound_scope(nblk):
            return self.R.sparse_attn(q, kv, attn.attn_sink, topk_idxs, attn.softmax_scale)

    def _q_proj(self, attn, qr, freqs_cis, qq=None):
        """model.py:770-772: q = wq_b(qr) per head, RoPE on the last rope_head_dim (#17: in the GEMV's store; R2: qq
        = act_quant(qr) from the fused q path)."""
        g = self.glue
        if g is not None and g.on("q_rope_epilogue"):
            q = g.q_proj(attn, qr, freqs_cis, qq)
            if q is not None:
                return q
        qb = g.linear_pre(qq, attn.wq_b, qr) if g is not None else attn.wq_b(qr)
        q = qb.unflatten(-1, (attn.n_local_heads, attn.head_dim))
        self.R.apply_rotary_emb(q[..., -attn.rope_head_dim:], freqs_cis)
        return q

    def _o_path(self, attn, o, freqs_cis):
        """model.py:781-786: inverse RoPE on the attention output, per-group view, act_quant QDQ in place (#17: one
        kernel)."""
        R = self.R
        g = self.glue
        if g is not None and g.on("o_rope_qdq"):
            g.o_path(attn, o, freqs_cis)
            return o.view(1, 1, attn.n_local_groups, -1)
        R.apply_rotary_emb(o[..., -attn.rope_head_dim:], freqs_cis, True)
        o = o.view(1, 1, attn.n_local_groups, -1)
        R.act_quant(o, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)
        return o

    def _hc_post(self, layer, x, residual, post, comb):
        """Block.hc_post (model.py:964-968) (#17: one kernel)."""
        g = self.glue
        if g is not None and g.on("hc_post"):
            return g.hc_post(x, residual, post, comb)
        return layer.hc_post(x, residual, post, comb)

    def _block(self, layer, x, pre_mix, parity):
        """Block.forward (model.py:970-996) with the static attention; the FFN is the module call (static MoE via
        the patch)."""
        residual = x
        attn_pre, attn_post, attn_comb, x, xq = self._mix_pre_norm(layer, x, pre_mix, "attn")
        x = self._attn(layer.attn, x, parity, xq)
        x = self._hc_post(layer, x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb, x, _ = self._mix_pre_norm(layer, x, attn_pre, "ffn")
        x = self._ffn(layer, x)
        x = self._hc_post(layer, x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def _mix_pre_norm(self, layer, x, pre_mix, side):
        """hc_mixes + hc_pre + {attn,ffn}_norm (model.py:984-986 / 991-993) -> (pre, post, comb, normed x, xq) (#17
        R2 #1 + #5: one launch for the statistic of the mixes, hc_pre, the norm and its act_quant)."""
        g = self.glue
        if g is not None and g.on("hc_norm") and x.is_contiguous() and pre_mix.is_contiguous():
            return g.mix_pre_norm(layer, x, pre_mix, side)
        if side == "attn":
            pre, post, comb = layer.hc_mixes(x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
            return pre, post, comb, layer.attn_norm(layer.hc_pre(x, pre_mix)), None
        pre, post, comb = layer.hc_mixes(x, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
        return pre, post, comb, layer.ffn_norm(layer.hc_pre(x, pre_mix)), None

    def _ffn(self, layer, x):
        """layer.ffn(x, None) (model.py:994; the static MoE through the patch). The fused glue's ffn-side
        precomputation (act_quant, x.float()) lives only for this call."""
        try:
            return layer.ffn(x, None)
        finally:
            if self.glue is not None:
                self.glue.ffn_done()

    def _engram(self, eng, h, hash_ids):
        """Engram.forward (model.py:350-365, text) (#17 R2 #15: the gate math in two launches + torch's sigmoid)."""
        g = self.glue
        if g is not None and g.on("engram") and h.is_contiguous():
            return g.engram(eng, h, hash_ids)
        return eng(h, hash_ids, None)

    def _head(self, layer, h, pre_mix):
        """hc_pre + norm + head (model.py:1273-1274) (#17 R2 #1: hc_pre + norm + x.float() in one launch)."""
        g, m = self.glue, self.model
        if g is not None and g.on("hc_norm") and h.is_contiguous() and pre_mix.is_contiguous():
            return g.head(layer, h, pre_mix)
        return m.head(m.norm(layer.hc_pre(h, pre_mix)))

    def _body(self, parity: int) -> None:
        """Transformer.forward (model.py:1244-1275) + sample (1288-1292) for one token at device `pos`."""
        R, m = self.R, self.model
        if m.temperature != 0:
            raise ValueError("StaticStep is greedy only (model.temperature must be 0)")
        self._c = _StepScalars(self)
        self._idx_nblk = self._idx_tlen = None
        if self.marker:
            from runtime.static_kernels import step_marker
            step_marker(self.marker_ctr)
        tok = self.tok
        hashes = engram_hash_static(m.engram_hash, tok, self.pos) if m.engram_hash is not None else None  # #2
        h = m.embed(tok)
        h = h.unsqueeze(2).repeat(1, 1, m.hc_mult, 1)
        pre_mix = R.make_identity_pre_mix(h, m.hc_mult)
        for layer in m.layers:
            if layer.engram is not None:
                h = self._engram(layer.engram, h, hashes[:, :, layer.engram.layer_hash_index, :])
            h, pre_mix = self._block(layer, h, pre_mix, parity)
        logits = self._head(layer, h, pre_mix)
        self.logits.copy_(logits)
        self.tok.copy_(logits.argmax(dim=-1).view(1, 1))                      # model.py:1292, #1
        self.pos.add_(1)
        if self.counters:
            self.cold_acc.add_(self.cold_tab.gather(1, self.ids_step).sum(-1))
        self._c = None
        self._idx_nblk = self._idx_tlen = None

    # ---------------------------------------------------------------- public
    def forward_static(self, parity: int) -> torch.Tensor:
        """Run the step eagerly from the static buffers (graph off). The device `pos` must have this parity."""
        with torch.inference_mode(), self._active():
            self._body(int(parity) & 1)
        return self.logits

    def set_inputs(self, token, pos: int) -> None:
        """tok <- token (int, or a device tensor holding one id: no sync); pos <- pos (skipped when the device already
        holds it, i.e. right after the previous step)."""
        pos = int(pos)
        if not 1 <= pos <= self.max_pos:
            raise ValueError(f"decode position {pos} outside [1, max_pos={self.max_pos}] (design #7: pos >= 1)")
        if token is not None:
            if isinstance(token, torch.Tensor):
                self.tok.copy_(token.reshape(1, 1))
            else:
                self.tok.fill_(int(token))
        if self._dev_pos != pos:
            self.pos.fill_(pos)
        self._dev_pos = pos

    def replay(self, pos: int, token=None) -> torch.Tensor:
        """Replay g_even/g_odd by pos parity. Returns the static logits buffer (overwritten by the next step)."""
        if not self.graphs:
            raise RuntimeError("replay before capture()")
        self.set_inputs(token, pos)
        self.graphs[parity_of(pos)].replay()
        self._dev_pos = int(pos) + 1
        self.n_counted += 1
        return self.logits

    def step(self, token, pos: int, graph: bool) -> torch.Tensor:
        """One decode step at `pos` feeding `token`: graph replay or the same code eagerly."""
        if graph:
            return self.replay(pos, token)
        self.set_inputs(token, pos)
        out = self.forward_static(parity_of(pos))
        self._dev_pos = int(pos) + 1
        self.n_counted += 1
        return out

    @property
    def captured(self) -> bool:
        return bool(self.graphs)

    def capture(self, fills: bool = False, warmup: int = 2) -> dict:
        """Design 2.3: dummy prefill (reference, eager), `warmup` steps per parity on a side stream (JIT compiles,
        cuBLAS handle, allocator), then capture g_even and g_odd with one shared mempool. fill_uninitialized_memory
        is `fills` throughout (default off: fills recorded into a graph would replay as kernels). Leaves the model's
        caches holding the dummy sequence: the next real prefill overwrites from position 0 (runtime/generate.py
        docstring). Returns timings."""
        import time

        det = getattr(torch.utils, "deterministic", None)
        prev_fill = getattr(det, "fill_uninitialized_memory", None)
        self.graphs = {}
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        try:
            if det is not None and prev_fill is not None:
                det.fill_uninitialized_memory = bool(fills)
            with torch.inference_mode():
                toks = torch.tensor([list(DUMMY_PROMPT)], dtype=torch.long, device=self.dev)
                out_ids, _, _ = self.model(toks, 0)
                p = len(DUMMY_PROMPT)
                self.tok.copy_(out_ids.reshape(1, 1))
                self.pos.fill_(p)
                s = torch.cuda.Stream(device=self.dev)
                s.wait_stream(torch.cuda.current_stream(self.dev))
                with torch.cuda.stream(s):
                    if self.dg is not None:
                        self.dg.warmup()  # DeepGEMM first-call JIT (both GEMM shapes) before anything is captured
                    for _ in range(warmup):
                        for _ in PARITIES:
                            self.forward_static(parity_of(p))
                            p += 1
                torch.cuda.current_stream(self.dev).wait_stream(s)
                torch.cuda.synchronize(self.dev)
                t_warm = time.perf_counter() - t0
                pool = torch.cuda.graph_pool_handle()
                for parity in PARITIES:
                    g = torch.cuda.CUDAGraph()
                    with self._active(), torch.cuda.graph(g, pool=pool):
                        self._body(parity)
                    self.graphs[parity] = g
                torch.cuda.synchronize(self.dev)
        finally:
            if det is not None and prev_fill is not None:
                det.fill_uninitialized_memory = prev_fill
            self._c = None
        sa = self.R.shared_attn
        sa.topk_idxs = sa.candidates = None  # in-graph tensors; the next forward re-publishes them
        with torch.inference_mode():  # cold_acc is an inference tensor (built under the caller's inference_mode); in-place ops need the mode
            self.cold_acc.zero_()
        self.n_counted = 0
        self._dev_pos = None
        self.captured_fills = bool(fills)
        return {"warmup_s": t_warm, "capture_s": time.perf_counter() - t0 - t_warm, "fills": bool(fills),
                "warmup_steps": warmup * len(PARITIES)}

    def release_graphs(self) -> None:
        self.graphs = {}
        self.captured_fills = None

    def detach(self) -> None:
        """Unbind this StaticStep from the model (MoE `_static_moe`) and drop its graphs, so it can be freed."""
        self.release_graphs()
        for sm in self.moes:
            if sm.moe.__dict__.get("_static_moe") is sm:
                del sm.moe.__dict__["_static_moe"]

    def cold_counts(self) -> Optional[Tuple[List[int], int, int]]:
        """(cold calls per layer, steps counted, routed experts per row-step) since capture, or None. Syncs."""
        if not self.counters:
            return None
        return [int(v) for v in self.cold_acc.tolist()], self.n_counted, self.topk

    def marker_count(self) -> int:
        return int(self.marker_ctr.item())
