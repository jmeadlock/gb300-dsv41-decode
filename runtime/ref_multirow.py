"""M1-T10's oracle: the reference forward over R > 1 decode rows, `forward_rows(model, tokens[1, R], start_pos)`.

The reference cannot run a multi-token DECODE step as written: its decode branch is one-token code -- the ring write
`window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)` (model.py:719) and the compressor's slot writes (479-480)
put an [1, R, d] value into one slot, the window list has one row (422-426), compress_lens is a scalar (568), the
latent RoPE row is one position (542, 757) and the head keeps x[:, -1] (1012-1013). (So the M1-T10 handoff's
"forward(tokens[:, a:a+6], a) is the oracle" holds only after the patches below.)

`multirow_decode()` replaces exactly those lines, for start_pos > 0 and seqlen > 1 only (anything else calls the
original), each with the reference's OWN multi-row spelling:

  Attention._window_kv   kv as 706-708 for all rows; the DSpark idiom `cat([window_kv_cache, kv])` (1068) as the
                         attended kv, row j listing positions P0+j-127..P0+j oldest first (= the 1-row order,
                         422-426): ring slot p % 128 for p < P0, 128 + (p - P0) for this step's rows, -1 for p < 0;
                         then 719's write for every row (the ring ends as R sequential steps leave it).
  Compressor.forward     ratio 1 unchanged (461-462 takes any row count). Ratio 2: wkv/wgate on all rows (464-465);
                         then token by token exactly the decode branch (476-482): state slot pos % ratio <- the row,
                         a group completing at pos pools the state with 482's expression; the step's groups stacked,
                         norm (485). (The prefill spelling 473-475 on [1, G, 2, d] can differ by an fp32 ulp.)
  Attention._compress_kv the latents' RoPE rows g * ratio per completed group g (753, 757), write at
                         [start_pos // ratio, + G) (762).
  Indexer.forward        index K as 544-547 with per-group RoPE rows; per-row compress_lens and the unreachable mask
                         in the prefill spelling (564-566) shifted to start_pos; the rest unchanged (570-581).
  ParallelHead.forward   full_logits when more than one row (1010-1014).

Everything else -- embedding, Engram (NgramHashState handles any start_pos/seqlen), mHC, the sparse_attn kernel, MoE
(multi-row native, model.py:891-906), norms, the head GEMM -- is the unmodified reference at R rows. The patches are
installed only inside the context and restored after.
"""
from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import torch


def _R():
    from runtime.static_step import _ref

    return _ref()


def multirow_window_idxs(win: int, seqlen: int, start_pos: int, device=None) -> torch.Tensor:
    """int32 [1, seqlen, win]: row j's positions P0+j-win+1 .. P0+j oldest first over kv = [ring(win) | rows(seqlen)]."""
    j = torch.arange(seqlen, device=device).view(-1, 1)
    p = start_pos + j - (win - 1) + torch.arange(win, device=device).view(1, -1)
    idx = torch.where(p < start_pos, torch.remainder(p, win), win + (p - start_pos))
    return torch.where(p < 0, -1, idx).int().unsqueeze(0)


@contextlib.contextmanager
def multirow_decode():
    """Install the multi-row decode spellings on the reference classes (module docstring); restore on exit."""
    R = _R()
    saved = []

    def put(cls, name, fn):
        saved.append((cls, name, cls.__dict__[name]))
        setattr(cls, name, fn)

    o_window_kv = R.Attention._window_kv
    o_compress_kv = R.Attention._compress_kv
    o_comp_fwd = R.Compressor.forward
    o_ix_fwd = R.Indexer.forward
    o_head_fwd = R.ParallelHead.forward

    def _window_kv(self, x, freqs_cis, start_pos):
        bsz, seqlen, _ = x.size()
        if start_pos == 0 or seqlen == 1:
            return o_window_kv(self, x, freqs_cis, start_pos)
        win = self.window_size
        if seqlen > win:
            raise ValueError(f"multirow decode: {seqlen} rows > window {win}")
        kv = self.kv_norm(self.wkv(x))                                            # model.py:706
        R.apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs_cis)              # 707
        R.act_quant(kv, R.fp8_block_size, R.scale_fmt, R.scale_dtype, True)       # 708
        window_kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)            # 1068's [ring | rows]
        idxs = multirow_window_idxs(win, seqlen, start_pos, x.device).expand(bsz, -1, -1).contiguous()
        pos = torch.arange(start_pos, start_pos + seqlen, device=x.device)
        self.window_kv_cache[:bsz, pos % win] = kv                               # 719, every row
        return window_kv, idxs

    def _compress_kv(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        if start_pos == 0 or seqlen == 1:
            return o_compress_kv(self, x, qr, start_pos, offset)
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio                              # 745
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)                                # 748 (patched below)
            R.shared_attn.compress_kv = self.compress_kv_cache
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)  # 751
        if latent is not None:
            g0 = start_pos // ratio
            rows = (g0 + torch.arange(latent.size(1), device=x.device)) * ratio   # group g -> position g*ratio
            R.apply_rotary_emb(latent[..., -self.rope_head_dim:], self.freqs_cis[rows])     # 753-759
            R.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)              # 761
            self.compress_kv_cache[:bsz, g0: g0 + latent.size(1)] = latent                  # 762
        return R.shared_attn.compress_kv[:bsz, :compress_len], idxs              # 764

    def comp_forward(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        if ratio == 1 or start_pos == 0 or seqlen == 1:
            return o_comp_fwd(self, x, start_pos)
        dtype = x.dtype
        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)                                   # 464-465, all rows at once
        pooled = []
        for i in range(seqlen):                                                  # 476-482, token by token
            pos = start_pos + i
            slot = pos % ratio
            self.kv_state[:bsz, slot] = kv[:, i]
            self.score_state[:bsz, slot] = score[:, i]
            if (pos + 1) % ratio == 0:
                pooled.append((self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True))
        if not pooled:
            return None                                                          # 483-484
        return self.norm(torch.cat(pooled, dim=1).to(dtype))                     # 485 on the step's groups

    def ix_forward(self, x, qr, latent, start_pos, offset):
        bsz, seqlen, _ = x.size()
        if start_pos == 0 or seqlen == 1:
            return o_ix_fwd(self, x, qr, latent, start_pos, offset)
        assert self.freqs_cis is not None
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        sa = R.shared_attn
        if self.owns_k and latent is not None:                                   # 537-547
            g0 = start_pos // ratio
            rows = (g0 + torch.arange(latent.size(1), device=x.device)) * ratio
            k = self.k_norm(self.wk(latent))
            R.apply_rotary_emb(k[..., -rd:], self.freqs_cis[rows])
            R.fp4_act_quant(k, R.fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio: start_pos // ratio + k.size(1)] = k
        if self.owns_k:
            sa.index_k = self.k_cache
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))  # 551-553
        R.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        R.fp4_act_quant(q, R.fp4_block_size, True)
        index_k = sa.index_k[:bsz, : end_pos // ratio]                            # 555
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        # 564-566's per-row compress_lens, rows at positions start_pos .. end_pos - 1
        compress_lens = (torch.arange(start_pos + 1, end_pos + 1, device=x.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(end_pos // ratio, device=x.device) >= compress_lens, -torch.inf)
        if self.is_candidate_source:                                             # 570-576
            sa.candidates = R.select_candidate_blocks(index_score, compress_lens, self.candidate_topk_blocks,
                                                      self.candidate_block_size)
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~sa.candidates, -torch.inf)
        topk = min(self.index_topk, end_pos // ratio)                            # 579-581
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    def head_forward(self, x, full_logits=False):
        return o_head_fwd(self, x, full_logits=full_logits or x.size(1) > 1)    # 1010-1014

    put(R.Attention, "_window_kv", _window_kv)
    put(R.Attention, "_compress_kv", _compress_kv)
    put(R.Compressor, "forward", comp_forward)
    put(R.Indexer, "forward", ix_forward)
    put(R.ParallelHead, "forward", head_forward)
    try:
        yield
    finally:
        for cls, name, fn in reversed(saved):
            setattr(cls, name, fn)


@torch.inference_mode()
def forward_rows(model, tokens: torch.Tensor, start_pos: int, hidden_layers: Sequence[int] = ()):
    """The reference forward of R decode rows `tokens` [1, R] at positions start_pos.. (start_pos >= 1) under
    `multirow_decode` -> (argmax ids [R], fp32 logits [R, V], main_hidden [R, n*dim] or None). `hidden_layers` sets
    model.target_layer_ids for this call only (model.py:1211, 1267-1269)."""
    if start_pos < 1:
        raise ValueError("forward_rows is the decode path: start_pos >= 1 (the prefill is model(tokens, 0))")
    from runtime.refmodel import reference_forward

    prev = getattr(model, "target_layer_ids", ())
    model.target_layer_ids = tuple(hidden_layers)
    try:
        with multirow_decode():  # the reference's bare factories land on tokens' device (refmodel.reference_device)
            ids, logits, hidden = reference_forward(model, tokens, int(start_pos))
    finally:
        model.target_layer_ids = prev
    R_ = tokens.size(1)
    return ids.reshape(R_), logits.reshape(R_, -1), (None if hidden is None else hidden.reshape(R_, -1))
