"""M1-T12: the DSpark loop -- verify (6 rows) -> accept by exact match -> commit -> draft, all on the device.

    gen = Generator(model, tok_dir, mode="graph", ...)          # model = build_and_load(..., load_mtp=True)
    loop = DSparkLoop(gen, max_new=256)                          # 6-row VerifyStep + DraftStep + device buffers
    loop.capture(n_graphs=2)                                     # verify|accept|commit graph + draft graph (or 1)
    res = loop.generate(ids, 256)                                # res["token_ids"], res["accepts"], step timings

Design: docs/M1-DSPARK-DESIGN.md 1.3-1.5, 2.2-2.4, 5 (risks 1, 6, 7); handoff docs/HANDOFF-M1-T12-LOOP.md.

## One step (no host decision anywhere; graph-capturable)

State entering a step: device P0 (`st.pos`), verify input `tok[1,6] = [b, d1..d5]` (b = the last emitted token, the
bonus, at position P0; d_j = the draft for position P0+j), `n_out` tokens emitted so far (`out_ids[:n_out]`; out[0] is
the prefill's token), `done`. Invariant: P0 = L + n_out - 1.

  verify   StaticStep(rows=6) body (`VerifyStep`, below) at positions P0..P0+5 -> logits[6,V], t_out[6] (row argmaxes),
           main_hidden[6, 3*dim]. Row i's argmax t_i predicts position P0+i+1.
  accept   draft d_j sits at row j and predicts position P0+j, i.e. it must equal t_{j-1}:
               match = (tok[1:6] == t[0:5]);  n = cumprod(match).sum()  in 0..5;  bonus = t_n
           emitted: d_1..d_n, t_n == t_0..t_n (d_j == t_{j-1} for j <= n) -> predicated stores
           out_ids[n_out + i] = t_i for i <= n (other slots keep their bytes); n_out += n + 1. EOS among the emitted
           tokens or n_out >= max_new sets `done`. accept log[n_steps] = n.
  commit   A' = P0 + n (committed prefix end). Per cache, below: only the ratio-2 compressor state needs an explicit
           commit (`VerifyStep.commit_ratio2`); everything else is overwritten by the next step before it is read.
  draft    DraftStep(main_hidden, t_out, A' = P0+n, ctx_count = n+1) -> out_ids [t_n, d'1..d'5] -> next `tok`;
           P0' = A' + 1.
  done     steps issued after `done` are predicated no-ops: no emission, A' := P0 - 1 (the commit then restores the
           compressor state, P0 stays, the draft writes no ring row with ctx_count 0, `tok` is not overwritten).
The host replays steps back to back and, after a step, issues one non-blocking D2H of (n_out, done, n_steps) into pinned
memory + an event (a "poll"); it waits on the event of the poll `lookahead` polls back, so the queue never drains.
LOOP-OVERRUN (docs/HANDOFF-LOOP-OVERRUN.md): the loop used to poll every 8 steps with lookahead 1, i.e. it learned of
`done` (EOS / max_new) up to (lookahead+1)*8 - 1 = 15 steps late and issued 9-15 no-op graph replays (~18 ms each) after
every sequence (97 steps = 1.8 s over the 8 fixture prompts). Now (`PollPlan`, defaults poll_every=1, lookahead=1,
tail_sync=True):
  * a poll after EVERY step (`poll_every` 1) with `lookahead` polls in flight: the host learns of `done` after issuing
    `lookahead` more steps -> overrun == lookahead (1) for an EOS ending; the pipeline never drains because the host
    waits for the poll of step k only after step k+1 is queued behind it.
  * tail sync (max_new endings): every step emits <= 6 tokens, so from the last observed n_out the host knows the
    earliest step that could set `done` by length. Before issuing a step it cannot rule that out for, it drains (waits
    for the poll of the newest issued step): one short GPU bubble per tail step instead of one no-op step (~18 ms).
    Length endings -> overrun 0; the cost is a few drains only in the last ~2-4 steps.
  * poll_every > 1 / lookahead > 1 / tail_sync False reproduce the old behaviour (A/B on the box).
Arithmetic is untouched: the host only chooses WHEN to stop issuing; every step after `done` is a predicated no-op (see
`done` above), so the emitted stream is the same whatever the poll schedule (tests/test_dspark_loop_cpu.py runs the real
`LoopBuffers.accept` under every schedule and compares the streams).

Device-side early stop (feasibility only, NOT implemented): the same overrun could be made ~free by skipping the graph body
once `done` is set -- CUDA conditional graph nodes (IF: CUDA 12.4; SWITCH: 12.8; the box has CUDA 13 / torch 2.13) with
the condition handle written by a one-thread kernel (`cudaGraphSetConditional(h, 1 - done)`) at the top of the graph;
the body (verify|accept|commit|draft) would become the IF node's body graph, a skipped replay costing ~5-10 us instead of
18 ms. Since a post-`done` step leaves every cache and counter unchanged by design (the T12 no-op), skipping is
byte-identical for the stream, and the host could keep polling in 8-step batches again. Obstacles, in order of size:
(1) torch.cuda.CUDAGraph exposes no conditional-node API that we know of (unchecked on torch 2.13): the capture would need the driver API through
cuda-python (cuGraphConditionalHandleCreate, cuGraphAddNode with CU_GRAPH_NODE_TYPE_CONDITIONAL, cuStreamBeginCaptureToGraph
into the body graph) and our own graph-pool / external-memory handling -- the current captures share torch's private pool;
(2) everything captured today (DeepGEMM / FlashMLA / TileLang / Triton launches, ~thousands of nodes, PDL-attributed
launches if M2-L5 lands) must be legal inside a conditional body graph; the driver documents restrictions on what a
conditional body may contain, and we have not tried any of our kernels there -- UNVERIFIED, needs a box probe;
(3) the residency hook's per-step host work (between replays) is unaffected, but nsys / step-anatomy tooling would see a
nested graph; (4) the benefit is bounded by what the host poll already recovers (the
remaining overrun is `lookahead` steps = 18 ms per request), so this is worth doing only if lookahead must grow to
absorb host jitter. Recommendation: not now; revisit if the box shows host stalls forcing lookahead >= 2.

Prefill -> first draft (design 2.3, risk 7): the reference prefill with main_hidden (`draft_step.prefill_hidden`) and
the reference draft-ring seeding (forward_spec at start_pos 0, `seed_draft_rings`), then the draft step with
ctx_count = 0 and A' = L-1 (no decode-numerics rewrite of slot (L-1) % 128).

## Acceptance rule (production's), and where it is read

Exact greedy match, confidence head off, k = 5 drafts -> 6 verify rows:
  * our checkpoint's reference: the loop is "out of scope for this repo" (reference model.py:129-130); greedy sampling
    is `sample(logits, 0) = logits.argmax(-1)` (model.py:1288-1292), used by the target and, with the stages'
    temperature 0 (runtime/mtp_loader.py), by the draft's Markov chain.
  * vendored vLLM 0.31.0 (`reference/vllm-0.31.0/deepseek_v41/`, re-checked for this task): it holds the draft MODEL
    only. `compute_confidence` (nvidia/dspark.py:358-363, "per-position acceptance probability") is defined and never
    called inside the model's forward (grep: the def is the only reference); the context-KV insert for accepted
    target positions is `precompute_and_store_context_kv` (nvidia/dspark.py:166-197). No rejection / acceptance code
    exists in the vendored tree (grep accept|reject: none outside unrelated attention helpers).
  * the rule itself is in vLLM's model runner v2, NOT vendored: research/round2-20261107/W2-dspark-yield.md F2 read
    v0.31.0 upstream: `_rejection_kernel` greedy branch `accepted = target_argmax == draft_sampled`, first mismatch
    stops and stores the target argmax (`spec_decode/rejection_sampler_utils.py:571-595`; block / probabilistic
    branches only at temp != 0); confidence head only with `enable_adaptive_verification` (default False,
    `config/speculative.py:552`; `spec_decode/dspark/speculator.py:106-113`), which Pg61 does not set
    (w0/results/ds41-args.json). bench/draft_bench.py `accepted` is the same rule host-side (2.97 tok/step, M1-T11).
  Under this rule the emitted stream is the target's greedy stream computed with the verify step's numerics; drafts
  only change how many tokens a step emits (design 1.4).

## Commit, cache by cache (which need an explicit rollback, and why) -- design 1.5

  cache                                  indexed   after a step with A' = P0+n            rollback
  ------------------------------------------------------------------------------------------------------------------
  main window rings, StaticStep #20      pos%256   rows P0+n+1..P0+5 hold rejected KV      none (write-before-read)
  (256 slots per layer)                            The next step writes P0'..P0'+5 = P0+n+1..P0+n+6 first, which
                                                   covers every rejected slot; a write of P0..P0+5 evicts only
                                                   positions <= P0-251, older than any row's window (row j reads
                                                   P0+j-127..P0+j); every row reads positions <= its own, all either
                                                   committed or rewritten earlier in the same step (write, then
                                                   attend, model.py:719-720). The 128-ring would NOT be safe (rows
                                                   evict their own window, design 1.5) -- hence the 256-ring.
  reference 128 window ring              pos%128   untouched by the static steps (prefill/eager only)   n/a
  compressed KV, ratio 2 and 1            pos//r   groups ending past A' are garbage      none (write-before-read)
                                                   A garbage group ends at 2g+1 (r=2) or p (r=1) in [A'+1, P0+5] c
                                                   [P0', P0'+5]: the next step's same-layer compressor rewrites it
                                                   (index_copy_ at div(r), static_step.py `_compress_kv`) before that
                                                   layer's attention reads; every read is masked to groups with end
                                                   <= the row's own position (per-row compress_lens).
  indexer K cache                         pos//r   same                                    none (owner layer writes
                                                   before the later layers that read sa.index_k)
  ratio-2 compressor kv/score_state       pos%2    StaticStep(rows=6) commits slot 0 <- last even row, slot 1 <-
  (the tail of an incomplete group)                last odd row of ALL 6 rows (static_step.py:867-869), i.e. as
                                                   if n = 5. The next step's pooling reads slot 0 when P0' is odd
                                                   (group (A', A'+1) with A' even): it must hold kv(A'), row n.
                                                   EXPLICIT COMMIT (`commit_ratio2`): VerifyStep keeps the 8-row
                                                   sequence [state0, state1, r0..r5] of the step and, under n, sets
                                                   slot s := the row of q_s = the last position <= A' with parity s
                                                   (row q_s - P0 + 2 if q_s >= P0, else the pre-step slot s): the
                                                   reference's per-token writes, model.py:477-480. n = 5 gives the
                                                   StaticStep's own commit bit for bit; a no-op step (A' = P0-1)
                                                   restores the pre-step state.
  ratio-1 compressor                      --       no state (model.py:461-462)                          n/a
  Engram hash cache                       pos      rows past A' hold rejected ids          none (written at
                                                   P0'..P0'+5 first, then gathered at <= pos, engram.py:167-173)
  draft rings x3 (128 slots)              pos%128  only context rows i < ctx_count = n+1 (positions P0..A', all
                                                   accepted) are written; block rows never are     nothing to undo
  shared_attn, MoE staging / DG buffers   --       per forward / per step scratch                  n/a
So exactly ONE cache needs an explicit commit: the ratio-2 compressor's two-slot state, because it is the only state
that is not position-indexed (its slots are reused every 2 positions, so the next step's write never lands where the
rejected row's value sits before it is read).
T12-FIX2 checked this table on the CPU emulation, cache by cache (tests/gpu/test_dspark_loop_gpu.py `cache_probe`,
tests/test_dspark_loop_cpu.py test_cache_probe_free_loop_caches_equal_one_row): in the free L = 17 run every cache
above is bit-equal to the 1-row step's entering and after every step (first step's partial group included).

## Static-shape details

* the verify graph is the T10 6-row body, unchanged except the deferred ratio-2 commit; `st.pos` (P0) is overwritten
  with P0' after the body (the body's own `pos += 6` is the teacher-forced advance).
* DraftStep aliases the verify step's `main_hidden` and `t_out` (its `tok`): context row i = verify row i = position
  P0+i = A'-n+i; the bonus is tok[ctx_count-1] = t_n. Its `a_pos`/`ctx_count` are written on the device.
* the draft runs OUTSIDE the verify step's `_active()` scope. M1-T11-PORT: it runs on the verify's fast kernels by
  default (`draft_impls` None = runtime/draft_step.py `draft_impls_for(gen's impls)`: GEMV through its OWN
  fp8_gemv_scope, DeepGEMM over the resident draft stacks, the fused glue at 5 rows, FlashMLA); `draft_impls="ref"`
  keeps the T11 reference draft (bit-equal to forward_spec). The draft only proposes tokens: the emitted stream is
  the verify's whatever the draft impls (design 1.4); they change accept counts, never emitted values.
* GEMV row-invariance: set DSV41_GEMV_PIN_M1=1 before the build (runtime/gemv_mxfp8.py `pin_m1`) so the 6-row fp8
  Linears use the 1-row configs (M1-T10 census); the fp32 head (cuBLAS) stays M-dependent.
* forced drafts (V1 scripted acceptor): `force_on`/`force_tok` replace d'1..d'5 after the draft (one `where` in the
  draft graph; off by default).
"""
from __future__ import annotations

import collections
import hashlib
import time
from typing import Dict, List, Optional, Sequence

import torch

from runtime.static_step import StaticStep, pool_ratio2_rows

ROWS = 6    # verify rows: 1 bonus + 5 drafts (design 1.1)
BLOCK = 5   # drafts per step (config dspark_block_size)
SLACK = 12  # out_ids slots past max_new (a live step writes <= n_out + 5 <= max_new + 4)


# ------------------------------------------------------------------------------------------------ device math
def accept_count(tok: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """tok [1, 6] = [b, d1..d5] (the verify input), t [6] = row argmaxes -> n [1] int64 = the number of leading drafts
    with d_j == t_{j-1} (exact greedy match; the first mismatch stops)."""
    d = tok.reshape(-1)[1:ROWS]
    match = (d == t.reshape(-1)[: ROWS - 1]).long()
    return torch.cumprod(match, 0).sum().view(1)


def emit_rows(out_ids: torch.Tensor, n_out: torch.Tensor, t: torch.Tensor, n: torch.Tensor, live: torch.Tensor,
              ar: torch.Tensor) -> torch.Tensor:
    """Predicated stores out_ids[n_out + i] = t[i] for i <= n when live; every other addressed slot is rewritten with
    its own bytes (6 distinct slots while live). -> valid [6] bool."""
    idx = torch.clamp(n_out + ar, max=out_ids.numel() - 1)
    valid = (ar <= n) & (live > 0)
    old = out_ids.index_select(0, idx)
    out_ids.index_copy_(0, idx, torch.where(valid, t.reshape(-1), old))
    return valid


def eos_hit(t: torch.Tensor, valid: torch.Tensor, eos: torch.Tensor) -> torch.Tensor:
    """Any emitted token (valid rows) in the EOS set -> [1] bool. eos may hold -1 sentinels (never a token id)."""
    return ((t.reshape(-1, 1) == eos.reshape(1, -1)).any(-1) & valid).any().view(1)


def ratio2_commit_src(a_new: torch.Tensor, p0: torch.Tensor, ar2: torch.Tensor) -> torch.Tensor:
    """For state slots s = 0, 1: q_s = the last position <= A' with q_s % 2 == s (the reference's per-token decode
    writes leave kv(q_s) in slot s, model.py:477-480) -> its index into the step's sequence [state0, state1, r0..r5]:
    q_s - P0 + 2 when q_s is a row of this step (q_s >= P0), else s (the pre-step slot already holds it)."""
    q = a_new.view(1) - torch.remainder(a_new.view(1) - ar2, 2)
    return torch.where(q >= p0.view(1), q - p0.view(1) + 2, ar2)


class LoopBuffers:
    """The loop's device state (static buffers; every update is a device op, no host sync).

    out_ids [max_new + SLACK] emitted tokens; status [3] = (n_out, done, n_steps) -- one D2H of 24 bytes polls it;
    acc_log [max_new] accepted drafts per live step (-1 = unused); p0 (this step's P0), a_new (A' after the commit),
    ctx (the draft's ctx_count), live (this step was live), n_last (this step's n, -1 when it was a no-op);
    max_new_t [1] the per-call bound; eos [E] (padded / disabled with -1)."""

    def __init__(self, max_new: int, eos_ids: Sequence[int], device, n_eos: Optional[int] = None):
        i64 = dict(dtype=torch.long, device=device)
        self.cap = int(max_new)
        self.out_ids = torch.zeros(self.cap + SLACK, **i64)
        self.status = torch.zeros(3, **i64)
        self.n_out, self.done, self.n_steps = self.status[0:1], self.status[1:2], self.status[2:3]
        self.acc_log = torch.full((max(1, self.cap),), -1, **i64)
        self.p0 = torch.zeros(1, **i64)
        self.a_new = torch.zeros(1, **i64)
        self.ctx = torch.zeros(1, **i64)
        self.live = torch.zeros(1, **i64)
        self.n_last = torch.full((1,), -1, **i64)
        self.max_new_t = torch.full((1,), self.cap, **i64)
        self.eos_ids = [int(e) for e in eos_ids]
        self.eos = torch.full((max(1, n_eos or len(self.eos_ids)),), -1, **i64)
        self.ar = torch.arange(ROWS, **i64)
        self.ar2 = torch.arange(2, **i64)
        self.set_eos(True)

    def set_eos(self, enabled: bool) -> None:
        with torch.inference_mode():
            self.eos.fill_(-1)
            if enabled and self.eos_ids:
                self.eos[: len(self.eos_ids)].copy_(torch.tensor(self.eos_ids, dtype=torch.long))

    def reset(self, first: torch.Tensor, p0: int, max_new: int) -> None:
        """After a prefill: out_ids[0] = the prefill's token (device [1]), n_out = 1, P0 = p0 = L, done iff the first
        token is EOS or max_new <= 1. No sync."""
        if not 1 <= int(max_new) <= self.cap:
            raise ValueError(f"max_new {max_new} outside [1, {self.cap}] (the loop's buffers)")
        with torch.inference_mode():
            f = first.reshape(-1)[:1].to(self.out_ids.dtype)
            self.out_ids.zero_()
            self.out_ids[:1].copy_(f)
            self.status.zero_()
            self.n_out.fill_(1)
            self.max_new_t.fill_(int(max_new))
            hit = (f.view(1, 1) == self.eos.view(1, -1)).any().view(1)
            self.done.copy_((hit | (self.n_out >= self.max_new_t)).long())
            self.acc_log.fill_(-1)
            self.p0.fill_(int(p0))
            self.a_new.fill_(int(p0) - 1)
            self.ctx.zero_()
            self.live.fill_(1)
            self.n_last.fill_(-1)

    def accept(self, tok: torch.Tensor, t: torch.Tensor) -> None:
        """The device phase after the verify (design 2.2): n, predicated emission, EOS / max_new -> done, the accept
        log, A' and the draft's ctx_count. `p0` must hold this step's P0."""
        self.live.copy_(1 - self.done)
        live = self.live
        n = accept_count(tok, t)
        valid = emit_rows(self.out_ids, self.n_out, t, n, live, self.ar)
        hit = eos_hit(t, valid, self.eos)
        slot = torch.clamp(self.n_steps, max=self.acc_log.numel() - 1)
        old = self.acc_log.index_select(0, slot)
        self.acc_log.index_copy_(0, slot, torch.where(live > 0, n, old))
        self.n_steps.add_(live)
        self.n_out.add_((n + 1) * live)
        self.done.copy_(((self.done > 0) | hit | (self.n_out >= self.max_new_t)).long())
        self.n_last.copy_(torch.where(live > 0, n, -1))
        self.a_new.copy_(torch.where(live > 0, self.p0 + n, self.p0 - 1))
        self.ctx.copy_((n + 1) * live)

    def read(self) -> dict:
        """Host view (syncs): n_out, done, n_steps, the emitted ids, the accept log."""
        s = self.status.tolist()
        return {"n_out": s[0], "done": bool(s[1]), "n_steps": s[2], "ids": self.out_ids[: s[0]].tolist(),
                "accepts": self.acc_log[: s[2]].tolist()}


def finalize_ids(ids: Sequence[int], max_new: int, eos: Sequence[int], stop_on_eos: bool):
    """The emitted stream cut like `Generator.greedy`: at the first EOS (kept) and at max_new -> (ids, finish)."""
    out = [int(i) for i in ids][: int(max_new)]
    e = set(int(x) for x in eos)
    if stop_on_eos:
        for k, tkn in enumerate(out):
            if tkn in e:
                return out[: k + 1], "stop"
    return out, "length"


def host_accept(tok: Sequence[int], t: Sequence[int]) -> int:
    """Host reference of `accept_count` (tests)."""
    n = 0
    for j in range(1, ROWS):
        if int(tok[j]) != int(t[j - 1]):
            break
        n += 1
    return n


class PollPlan:
    """The host's poll schedule (pure Python, no torch: tests/test_dspark_loop_cpu.py drives it against the real
    LoopBuffers). It decides (a) after which issued steps to record a poll (`poll_now`) and (b) when the host must stop
    and wait for the newest poll before issuing another step (`must_drain`, the tail sync); `observe` feeds it what a
    completed poll read. A step emits at most ROWS = 6 tokens, so n_out after `s` issued steps is at most
    n_obs + ROWS * (s - s_obs) for the last observation (n_obs after s_obs steps; initially 1 after 0): when that
    bound is below max_new the sequence cannot have ended by LENGTH yet. EOS endings are not predictable and are
    caught by the polls themselves (overrun = lookahead steps)."""

    def __init__(self, max_new: int, poll_every: int = 1, lookahead: int = 1, tail_sync: bool = True):
        self.max_new = int(max_new)
        self.poll_every = max(1, int(poll_every))
        self.lookahead = max(0, int(lookahead))
        self.tail_sync = bool(tail_sync)
        self.obs_step, self.obs_n_out, self.done = 0, 1, False

    def observe(self, step: int, n_out: int, done: bool) -> None:
        """A completed poll: the device status right after `step` issued steps. Older observations are ignored."""
        if step >= self.obs_step:
            self.obs_step, self.obs_n_out = int(step), int(n_out)
        self.done = self.done or bool(done)

    def n_out_bound(self, issued: int) -> int:
        return self.obs_n_out + ROWS * (int(issued) - self.obs_step)

    def length_end_possible(self, issued: int) -> bool:
        """Could the sequence already be done by max_new after `issued` steps?"""
        return self.n_out_bound(issued) >= self.max_new

    def poll_now(self, issued: int) -> bool:
        """Record a poll right after step number `issued` (1-based)?"""
        return issued % self.poll_every == 0 or (self.tail_sync and self.length_end_possible(issued))

    def must_drain(self, issued: int) -> bool:
        """Wait for the newest poll before issuing step `issued + 1`? (the tail sync)"""
        return self.tail_sync and issued > 0 and self.length_end_possible(issued)


# ------------------------------------------------------------------------------------------------ the verify step
class VerifyStep(StaticStep):
    """StaticStep (rows = 6) whose ratio-2 compressor-state commit is deferred to the loop (`defer_ratio2_commit`):
    the step keeps [state0, state1, r0..r5] (kv and score, fp32) per ratio-2 compressor and `commit_ratio2(A', P0)`
    writes the state the reference's per-token decode would leave after position A' (module docstring). With the
    flag off this is StaticStep exactly (the override calls the parent). Pooling is the parent's own op
    (`pool_ratio2_rows` on the pre-step slot 0), so the latents and logits are bit-identical either way."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.defer_ratio2_commit = False
        self._r2: Dict[int, tuple] = {}
        self._r2_fresh = False
        d = self.dev
        self._ar2 = torch.arange(2, dtype=torch.long, device=d)
        if self.rows > 1:
            for layer in self.model.layers:
                comp = getattr(layer.attn, "compressor", None)
                if comp is not None and comp.compress_ratio == 2:
                    if int(comp.kv_state.size(1)) != 2:
                        raise ValueError("ratio-2 compressor state is not 2 slots")
                    hd = comp.kv_state.size(-1)
                    self._r2[id(comp)] = (comp, torch.zeros(1, self.rows + 2, hd, dtype=torch.float32, device=d),
                                          torch.zeros(1, self.rows + 2, hd, dtype=torch.float32, device=d))

    def _compressor_rows(self, comp, x):
        if not self.defer_ratio2_commit or comp.compress_ratio != 2:
            return super()._compressor_rows(comp, x)
        c = self._c
        dtype = x.dtype
        xf = x.float()
        kv, score = self._compressor_proj(comp, xf)                           # model.py:464-465 (parent's ops, T9b-8)
        pooled = pool_ratio2_rows(comp.kv_state[:1, :1], comp.score_state[:1, :1], kv, score, c.r2_seq())
        _, kvs, scs = self._r2[id(comp)]
        kvs[:, :2].copy_(comp.kv_state[:1])                                   # pre-step slots 0, 1
        kvs[:, 2:].copy_(kv)
        scs[:, :2].copy_(comp.score_state[:1])
        scs[:, 2:].copy_(score)
        self._r2_fresh = True
        return comp.norm(pooled.to(dtype))                                    # model.py:485

    def commit_ratio2(self, a_new: torch.Tensor, p0: torch.Tensor) -> None:
        """Slot s := the step's row of q_s (or the pre-step slot s), every ratio-2 compressor. Device ops only."""
        if not self._r2:
            return
        if not self.defer_ratio2_commit:
            raise RuntimeError("commit_ratio2 without defer_ratio2_commit (the step already committed n = 5)")
        src = ratio2_commit_src(a_new, p0, self._ar2)
        for comp, kvs, scs in self._r2.values():
            comp.kv_state[:1].copy_(kvs.index_select(1, src))
            comp.score_state[:1].copy_(scs.index_select(1, src))


# ------------------------------------------------------------------------------------------------ the loop
class DSparkLoop:
    """The DSpark k=5 loop over a Generator's model (module docstring). Builds the 6-row VerifyStep through the
    Generator (`gen.static_cls`, `gen.hidden_layers` = the mtp target layers, `gen.set_rows(6)`) and a DraftStep that
    aliases its main_hidden / t_out. gen.mode "graph" -> `capture()` then replays; "static" -> the same ops eagerly.
    max_new: buffer capacity (the per-call bound is a device tensor). poll_every / lookahead / tail_sync: the async poll
    schedule (module docstring, `PollPlan`; defaults 1 / 1 / True = overrun <= 1 step; 8 / 1 / False = the old 9-15).
    draft_impls: None = the verify's impls mapped by draft_step.draft_impls_for; "ref" = the reference draft; or a
    dict of DraftStep impl kwargs."""

    def __init__(self, gen, max_new: int = 256, *, eos_ids: Optional[Sequence[int]] = None, poll_every: int = 1,
                 lookahead: int = 1, tail_sync: bool = True, draft_impls=None, _allow_cpu: bool = False):
        from runtime.draft_step import DraftStep, draft_impls_for

        model = gen.model
        mt = getattr(model, "runtime_mtp", None)
        if mt is None or not getattr(model, "mtp", None):
            raise ValueError("DSparkLoop: the model has no mtp stages (build_and_load(..., load_mtp=True))")
        self.gen, self.model = gen, model
        self.hidden = tuple(int(i) for i in mt.target_layer_ids)
        self._saved = (gen.hidden_layers, gen.rows, gen.static_cls)
        gen.reset_static()
        gen.hidden_layers = self.hidden
        gen.set_rows(ROWS)
        gen.static_cls = VerifyStep
        with torch.inference_mode():
            st = gen.static_step()
        if not isinstance(st, VerifyStep):
            raise TypeError(f"DSparkLoop needs a VerifyStep (got {type(st).__name__}; a static_factory must build one)")
        if st.rows != ROWS or st.main_hidden is None:
            raise ValueError(f"verify step rows {st.rows}, main_hidden {st.main_hidden is not None}")
        st.defer_ratio2_commit = True
        self.st = st
        self.dev = st.dev
        self.is_cuda = self.dev.type == "cuda"
        if draft_impls is None:
            draft_impls = draft_impls_for(gen.moe_impl, gen.fp8_impl, gen.glue_impl, gen.attn_impl)
        elif draft_impls == "ref":
            draft_impls = {}
        self.draft_impls = dict(draft_impls)
        with torch.inference_mode():
            self.ds = DraftStep(model, main_hidden=st.main_hidden, tok=st.t_out, _allow_cpu=_allow_cpu,
                                **self.draft_impls)
        if self.ds.block != BLOCK or self.ds.C != ROWS:
            raise ValueError(f"draft block {self.ds.block} / ctx rows {self.ds.C}: the loop is k = {BLOCK}")
        self.max_pos = min(int(st.max_pos), int(self.ds.max_pos))
        self.eos_ids = list(gen.eos_ids if eos_ids is None else eos_ids)
        with torch.inference_mode():
            self.b = LoopBuffers(max_new, self.eos_ids, self.dev)
            self.force_on = torch.zeros(1, dtype=torch.long, device=self.dev)
            self.force_tok = torch.zeros(BLOCK, dtype=torch.long, device=self.dev)
        self.poll_every = max(1, int(poll_every))
        self.lookahead = max(0, int(lookahead))
        self.tail_sync = bool(tail_sync)
        self.graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.n_graphs = 0
        self.capture_info: Optional[dict] = None
        self.n_replays = 0
        self._hook_evs: List[tuple] = []   # A2-2: (start, end) CUDA events around the residency hook, one per timed step

    # ---------------------------------------------------------------- bodies
    def _verify_accept(self) -> None:
        """Graph A: verify (inside st._active()), accept/emit, the compressor commit, the draft's inputs, P0'."""
        st, b, ds = self.st, self.b, self.ds
        b.p0.copy_(st.pos)
        st._body(0)
        b.accept(st.tok, st.t_out)
        st.commit_ratio2(b.a_new, b.p0)
        ds.a_pos.copy_(b.a_new)
        ds.ctx_count.copy_(b.ctx)
        st.pos.copy_(b.a_new + 1)

    def _draft(self) -> None:
        """Graph B: the draft (outside the verify scope), then the next verify input when this step was live."""
        st, ds, b = self.st, self.ds, self.b
        ds._body()
        nxt = ds.out_ids.view(1, ROWS)
        drafts = torch.where(self.force_on > 0, self.force_tok.view(1, BLOCK), nxt[:, 1:])
        nxt = torch.cat([nxt[:, :1], drafts], dim=1)
        st.tok.copy_(torch.where(b.live.view(1, 1) > 0, nxt, st.tok))

    def _step_eager(self) -> None:
        with torch.inference_mode():
            with self.st._active():
                self._verify_accept()
            self._draft()

    def _first_draft(self) -> None:
        if "B" in self.graphs:
            self.graphs["B"].replay()
        else:
            with torch.inference_mode():
                self._draft()

    def step_once(self, events: Optional[tuple] = None) -> None:
        """One loop step (graph replay(s) or eager). events: (start, mid, end) CUDA events (mid unused at 1 graph)."""
        g = self.graphs
        if events is not None:
            events[0].record()
        if self.n_graphs == 2:
            g["A"].replay()
            if events is not None:
                events[1].record()
            g["B"].replay()
        elif self.n_graphs == 1:
            g["AB"].replay()
        else:
            if events is not None:
                with torch.inference_mode(), self.st._active():
                    self._verify_accept()
                events[1].record()
                with torch.inference_mode():
                    self._draft()
            else:
                self._step_eager()
        if events is not None:
            events[2].record()
        self.n_replays += 1
        rm = getattr(self.gen, "residency_mgr", None)
        if rm is not None:
            self._residency_hook(rm, events is not None)

    def _residency_hook(self, rm, timed: bool) -> None:
        """A2-2: the ResidencyManager's per-step call, between this step's replay and the next one, on the replay
        stream (runtime/residency.py: count the routed ids this replay left in st.ids_step, publish finished swaps,
        drain, one barrier; table writes are scalar fill_ kernels in stream order, so a graph enqueued later reads
        the new contents and one enqueued earlier is finished before a demoted slot is overwritten). Steps the loop
        issues after `done` are predicated no-ops: the device `live` flag keeps them out of the hit counters. The
        hook runs AFTER the step's end event, so the step/verify/draft ms stay graph-only; `generate` reports the
        manager's own GPU time per step as `mgr_ms` (a pair of events around the hook)."""
        if timed:
            h0 = torch.cuda.Event(enable_timing=True)
            h1 = torch.cuda.Event(enable_timing=True)
            h0.record()
        rm.after_step(live=self.b.live)
        if timed:
            h1.record()
            self._hook_evs.append((h0, h1))

    def set_force(self, drafts: Optional[Sequence[int]]) -> None:
        """Scripted drafts for the NEXT draft write (V1): d'1..d'5 := drafts; None = the draft step's own."""
        with torch.inference_mode():
            if drafts is None:
                self.force_on.zero_()
            else:
                d = [int(x) for x in drafts]
                if len(d) != BLOCK:
                    raise ValueError(f"{len(d)} forced drafts, want {BLOCK}")
                self.force_tok.copy_(torch.tensor(d, dtype=torch.long))
                self.force_on.fill_(1)

    # ---------------------------------------------------------------- capture
    def capture(self, n_graphs: int = 2, fills: bool = False, warmup: int = 2) -> dict:
        """Warm-ups (the StaticStep's and DraftStep's own capture paths: dummy prefill, JIT, GEMV tunes, DeepGEMM),
        their graphs released, then the loop graphs: n_graphs 2 = A (verify|accept|commit) + B (draft|next tok),
        1 = AB (one graph per step) + B (the first draft after a prefill). Must run before the real prefill (the
        warm-ups overwrite the caches with a dummy sequence)."""
        if not self.is_cuda:
            raise RuntimeError("capture needs CUDA (gen.mode 'static' runs the loop eagerly)")
        if n_graphs not in (1, 2):
            raise ValueError(f"n_graphs {n_graphs}: 1 or 2")
        from runtime.static_step import DUMMY_PROMPT

        st, ds = self.st, self.ds
        self.release_graphs()
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        st.defer_ratio2_commit = False                   # the StaticStep warm-ups are its own teacher-forced steps
        try:
            info_st = st.capture(fills=fills, warmup=warmup)
        finally:
            st.defer_ratio2_commit = True
        info_ds = ds.capture(fills=fills)
        st.release_graphs()
        ds.release_graph()
        torch.cuda.synchronize(self.dev)
        torch.cuda.empty_cache()
        det = getattr(torch.utils, "deterministic", None)
        prev_fill = getattr(det, "fill_uninitialized_memory", None)
        try:
            if det is not None and prev_fill is not None:
                det.fill_uninitialized_memory = bool(fills)
            with torch.inference_mode():
                p = len(DUMMY_PROMPT)
                self.b.reset(st.tok[0, :1], p, self.b.cap)
                st.pos.fill_(p)
                ds.a_pos.fill_(p - 1)
                ds.ctx_count.zero_()
                s = torch.cuda.Stream(device=self.dev)
                s.wait_stream(torch.cuda.current_stream(self.dev))
                with torch.cuda.stream(s):
                    self._draft()
                    for _ in range(warmup):
                        self._step_eager()
                torch.cuda.current_stream(self.dev).wait_stream(s)
                torch.cuda.synchronize(self.dev)
                t_warm = time.perf_counter() - t0
                pool = torch.cuda.graph_pool_handle()
                if n_graphs == 2:
                    gA = torch.cuda.CUDAGraph()
                    with st._active(), torch.cuda.graph(gA, pool=pool):
                        self._verify_accept()
                    self.graphs["A"] = gA
                else:
                    gAB = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gAB, pool=pool):
                        with st._active():
                            self._verify_accept()
                        self._draft()
                    self.graphs["AB"] = gAB
                gB = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gB, pool=pool):
                    self._draft()
                self.graphs["B"] = gB
                torch.cuda.synchronize(self.dev)
        finally:
            if det is not None and prev_fill is not None:
                det.fill_uninitialized_memory = prev_fill
            st._c = None
        sa = st.R.shared_attn
        sa.topk_idxs = sa.candidates = None              # in-graph tensors; the next forward re-publishes them
        with torch.inference_mode():
            st.cold_acc.zero_()
            st.uniq_acc.zero_()
            st.ovf_steps.zero_()
            st.ovf_total.zero_()
            self.force_on.zero_()
        st.n_counted = 0
        self.n_graphs = n_graphs
        self.n_replays = 0
        self.capture_info = {"n_graphs": n_graphs, "fills": bool(fills), "warmup_s": t_warm,
                             "capture_s": time.perf_counter() - t0 - t_warm, "static_step": info_st, "draft": info_ds}
        return self.capture_info

    def release_graphs(self) -> None:
        self.graphs = {}
        self.n_graphs = 0

    # ---------------------------------------------------------------- one sequence
    def check_bounds(self, L: int, max_new: int) -> None:
        last = L + max_new + SLACK - 2  # P0' <= L + max_new + 4, rows to +5, no-op steps stay there
        if last > self.max_pos:
            raise ValueError(f"prompt {L} + max_new {max_new}: positions up to {last} > max_pos {self.max_pos}")
        if L < 1:
            raise ValueError("empty prompt")

    @torch.inference_mode()
    def start(self, ids: Sequence[int], max_new: int, stop_on_eos: bool = True,
              keep_force: bool = False) -> torch.Tensor:
        """Prefill -> first draft (module docstring). Returns the prefill's fp32 logits [V] (device). keep_force: the
        scripted drafts set by `set_force` apply to the first draft too (tests); default off."""
        from runtime.draft_step import prefill_hidden, seed_draft_rings
        from runtime.prefill_fast import last_prefill as _last_prefill

        st, ds, b = self.st, self.ds, self.b
        L = len(ids)
        self.check_bounds(L, max_new)
        if self.n_graphs == 0 and self.gen.mode == "graph":
            raise RuntimeError("graph mode: capture() before the first prefill")
        if st.pool is not None and getattr(st.pool, "generation", None) != st._pool_gen:
            raise RuntimeError("the ExpertPool's storage moved after the verify step was built: rebuild the loop")
        b.set_eos(stop_on_eos)
        tokens = torch.tensor([list(ids)], dtype=torch.long, device=self.dev)
        out, logits, mh = prefill_hidden(self.model, tokens, self.hidden)  # the reference prefill (+ 256-ring sync)
        self.last_prefill = _last_prefill()                                # M4-P2: $DSV41_PREFILL mode + counters
        first = out.reshape(-1)[:1]
        seed_draft_rings(self.model, first, mh)                            # forward_spec at start_pos 0
        b.reset(first, L, max_new)
        st.t_out[:1].copy_(first)                                          # the bonus of ctx_count 0
        st.pos.fill_(L)
        st._synced = True
        ds.a_pos.fill_(L - 1)
        ds.ctx_count.zero_()
        if not keep_force:
            self.force_on.zero_()
        self._first_draft()                                                # A' = L-1, ctx 0 -> tok = [first, d1..5]
        return logits.reshape(-1, logits.size(-1))[-1]

    @torch.inference_mode()
    def generate(self, ids: Sequence[int], max_new: int, *, stop_on_eos: bool = True, timing: bool = True,
                 parity: bool = False, top_k: Optional[int] = None, tag: Optional[str] = None,
                 prompt: Optional[str] = None, profile: Optional[tuple] = None) -> dict:
        """One sequence through the loop -> {"token_ids", "finish_reason", "accepts" (n per live step), "steps",
        "steps_issued", "tokens_per_step", "prefill_ms", "wall_ms", per-step "step_ms"/"verify_ms"/"draft_ms" (CUDA
        events, live steps only; verify/draft only at 2 graphs or eager), ...}. parity: sync after every step and keep
        each emitted token's fp32 logits row -> a parity record in Generator.greedy's format (+ per-position sha256),
        else the async poll. profile = (first_step, n_steps): cudaProfilerStart/Stop around those steps (nsys)."""
        from runtime.generate import logprob_entry, make_record

        gen, b, st = self.gen, self.b, self.st
        top_k = gen.top_k if top_k is None else top_k
        cuda = self.is_cuda
        if cuda:
            torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        logits0 = self.start(ids, max_new, stop_on_eos)
        first = int(b.out_ids[0])                                          # the prefill is a sync point anyway
        if cuda:
            torch.cuda.synchronize(self.dev)
        prefill_ms = (time.perf_counter() - t0) * 1e3
        entries: List[dict] = []
        digests: List[str] = []
        self._hook_evs = []
        if parity:
            lc = logits0.detach().to("cpu", torch.float32)
            entries.append(logprob_entry(lc, first, top_k, gen.token_str))
            digests.append(hashlib.sha256(lc.numpy().tobytes()).hexdigest())
        eos = set(self.eos_ids) if stop_on_eos else set()
        cap_steps = max(0, max_new - 1) + (self.lookahead + 1) * self.poll_every
        evs: List[tuple] = []
        issued = 0
        poll = {"polls": 0, "drains": 0, "enqueue_us": 0.0, "wait_ms": 0.0}
        t1 = time.perf_counter()
        prof = profile is not None and cuda
        if not (max_new <= 1 or first in eos):
            mk = (lambda: tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))) if (timing and cuda) else None
            if parity or not cuda:
                while issued < cap_steps:
                    e = mk() if mk else None
                    self._prof(prof, profile, issued, True)
                    self.step_once(e)
                    self._prof(prof, profile, issued, False)
                    issued += 1
                    if e is not None:
                        evs.append(e)
                    if parity:
                        n = int(b.n_last)
                        if n >= 0:
                            lcs = st.logits[: n + 1].detach().to("cpu", torch.float32)
                            ts = st.t_out[: n + 1].tolist()
                            for i in range(n + 1):
                                entries.append(logprob_entry(lcs[i], ts[i], top_k, gen.token_str))
                                digests.append(hashlib.sha256(lcs[i].numpy().tobytes()).hexdigest())
                    if int(b.done):
                        break
            else:
                issued, poll = self._decode_async(max_new, cap_steps, mk, prof, profile, evs)
        if cuda:
            torch.cuda.synchronize(self.dev)
        if prof and getattr(self, "_prof_on", False):
            torch.cuda.profiler.stop()
            self._prof_on = False
        wall_ms = (time.perf_counter() - t0) * 1e3
        decode_wall_ms = (time.perf_counter() - t1) * 1e3
        r = b.read()
        accepts = r["accepts"]
        ids_out, finish = finalize_ids(r["ids"], max_new, self.eos_ids, stop_on_eos)
        live = len(accepts)
        res = {"token_ids": ids_out, "finish_reason": finish, "n_out_raw": r["n_out"], "done": r["done"],
               "accepts": accepts, "steps": live, "steps_issued": issued,
               "tokens_per_step": (sum(1 + n for n in accepts) / live) if live else None,
               "prefill_ms": prefill_ms, "wall_ms": wall_ms, "decode_wall_ms": decode_wall_ms,
               "n_graphs": self.n_graphs, "prompt_tokens": len(ids)}
        # LOOP-OVERRUN: steps issued after the sequence was done (predicated no-ops, a full graph replay each), their
        # GPU time (their own start/end events), and what the polls cost the host. Additive keys.
        res["overrun_steps"] = max(0, issued - live)
        res["poll"] = dict(poll, poll_every=self.poll_every, lookahead=self.lookahead, tail_sync=self.tail_sync)
        if evs:
            res["overrun_ms"] = [s.elapsed_time(e2) for s, _m, e2 in evs[live:]]
            res["decode_wall_ms_excl_overrun"] = decode_wall_ms - sum(res["overrun_ms"])
        if evs:
            ms = [(s.elapsed_time(e2), s.elapsed_time(m) if self.n_graphs != 1 else None,
                   m.elapsed_time(e2) if self.n_graphs != 1 else None) for s, m, e2 in evs[:live]]
            res["step_ms"] = [x[0] for x in ms]
            res["verify_ms"] = [x[1] for x in ms] if self.n_graphs != 1 else None
            res["draft_ms"] = [x[2] for x in ms] if self.n_graphs != 1 else None
            if self._hook_evs:   # A2-2: the residency manager's own GPU time per live step (count + table writes)
                res["mgr_ms"] = [h0.elapsed_time(h1) for h0, h1 in self._hook_evs[:live]]
        if parity:
            k = len(ids_out)
            entries, digests = entries[:k], digests[:k]
            text_ids = ids_out[:-1] if finish == "stop" else ids_out
            res["record"] = make_record(tag=tag, prompt=prompt, model_name=gen.model_name, prompt_tokens=len(ids),
                                        gen_ids=ids_out, entries=entries, finish_reason=finish,
                                        text=gen.decode(text_ids) if gen.tokenizer is not None else "")
            res["step_sha256"] = digests
            h = hashlib.sha256()
            for d_ in digests:
                h.update(d_.encode())
            res["logits_sha256"] = h.hexdigest()
        return res

    # ---------------------------------------------------------------- the async decode (LOOP-OVERRUN)
    def _poll_buffers(self, n: int) -> list:
        return [torch.zeros(3, dtype=torch.long, device="cpu", pin_memory=True) for _ in range(n)]

    def _poll_record(self, buf: torch.Tensor):
        """Enqueue the status D2H into `buf` and an event behind it (stream order: it sees the steps issued so far)."""
        buf.copy_(self.b.status, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return ev

    def _poll_wait(self, token) -> None:
        token.synchronize()

    def _decode_async(self, max_new: int, cap_steps: int, mk, prof, profile, evs: list) -> tuple:
        """Issue steps with no host sync except the poll waits (`PollPlan`: a poll after every `poll_every` steps and in
        the max_new tail, `lookahead` polls in flight, a drain before a step the tail cannot rule out). Returns
        (steps issued, poll stats). The poll primitives are methods so the schedule is testable without CUDA."""
        plan = PollPlan(max_new, self.poll_every, self.lookahead, self.tail_sync)
        nb = self.lookahead + 1
        pins = self._poll_buffers(nb)
        pending: collections.deque = collections.deque()      # (token, buf, steps issued when recorded)
        stats = {"polls": 0, "drains": 0, "enqueue_us": 0.0, "wait_ms": 0.0}
        pc = time.perf_counter
        issued = 0

        def record() -> None:
            t = pc()
            buf = pins[stats["polls"] % nb]
            pending.append((self._poll_record(buf), buf, issued))
            stats["polls"] += 1
            stats["enqueue_us"] += (pc() - t) * 1e6

        def wait_oldest() -> bool:
            tok, buf, step = pending.popleft()
            t = pc()
            self._poll_wait(tok)
            stats["wait_ms"] += (pc() - t) * 1e3
            n_out, done = int(buf[0]), bool(int(buf[1]))
            plan.observe(step, n_out, done)
            return done

        stop = False
        while issued < cap_steps and not stop:
            if plan.must_drain(issued):                       # the next step might be a no-op: wait for the exact state
                if not pending or pending[-1][2] != issued:
                    record()
                stats["drains"] += 1
                while pending and not stop:
                    stop = wait_oldest()
                if stop:
                    break
            e = mk() if mk else None
            self._prof(prof, profile, issued, True)
            self.step_once(e)
            self._prof(prof, profile, issued, False)
            issued += 1
            if e is not None:
                evs.append(e)
            if plan.poll_now(issued):
                record()
            while len(pending) > self.lookahead and not stop:
                stop = wait_oldest()
        return issued, stats

    def _prof(self, on: bool, profile, i: int, before: bool) -> None:
        if not on:
            return
        start, n = profile
        if before and i == start:
            torch.cuda.synchronize(self.dev)
            torch.cuda.profiler.start()
            self._prof_on = True
        if not before and i == start + n - 1 and getattr(self, "_prof_on", False):
            torch.cuda.synchronize(self.dev)
            torch.cuda.profiler.stop()
            self._prof_on = False

    # ---------------------------------------------------------------- misc
    def describe(self) -> dict:
        return {"rows": ROWS, "block": BLOCK, "hidden_layers": list(self.hidden), "max_pos": self.max_pos,
                "max_new_cap": self.b.cap, "poll_every": self.poll_every, "lookahead": self.lookahead,
                "tail_sync": self.tail_sync,
                "n_graphs": self.n_graphs, "ratio2_compressors": len(self.st._r2), "eos_ids": self.eos_ids,
                "draft_impls": dict(self.draft_impls), "verify": self.st.describe(), "draft": self.ds.describe()}

    def close(self) -> None:
        """Drop the graphs and the verify step; restore the Generator's rows / hidden_layers / static_cls."""
        self.release_graphs()
        self.ds.release_graph()
        gen = self.gen
        gen.reset_static()
        gen.hidden_layers, rows, gen.static_cls = self._saved
        gen.set_rows(rows)
