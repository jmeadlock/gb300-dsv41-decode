# M1 design — the 6-row DSpark speculative step in our runtime

Desk memo (no GPU, no box), 2026-11-06 22:25 CST, for `docs/HANDOFF-M1-DSPARK-DESIGN.md`. Every number is either cited
to a file in this repo or labelled **est.** `model.py` means `reference/deepseek-inference/model.py`; `vllm/` means
`reference/vllm-0.31.0/deepseek_v41/`.

## 0. Corrections to the handoff (read these first)

1. **The vendored `vllm/nvidia/dspark.py` is the draft *model* only** (weights, embed, three decoder layers, Markov and
   confidence heads, context-KV insert; dspark.py:72-363). The proposer, verification batch builder, rejection
   sampler and KV bookkeeping live in vLLM's core `v1/spec_decode` / scheduler, which are **not in our tree** (no
   `rejection`/`num_spec` code outside `deepseek_v41/`, checked). vLLM's acceptance rule and rollback therefore
   cannot be read here. What we *can* read is production's configuration: `method dspark, num_speculative_tokens 5
   (batch 1–4), draft_sample_method probabilistic, rejection_sample_method block` (`w0/results/ds41-args.json`,
   `--speculative-config`).
2. **The reference has no loop either:** "nothing calls forward_spec … the speculative-decoding loop itself is out of
   scope for this repo" (model.py:129-130). Its only driver is a smoke test that drafts after every 1-row step and
   never verifies (model.py:1306-1312). The loop below is our design, constrained by the reference's forward.
3. **"Rollback = write position is the accepted prefix end" is wrong for the 128-slot window ring** (§1.5). It holds for
   every position-indexed cache. It fails for the 128 ring even when every draft is accepted. The T9b-3 memo, written
   in parallel, found the same hazard independently and fixes it with a 256-slot ring
   (`docs/T9B3-ATTN-DESIGN.md:222-226`). This memo adopts that fix (§1.5).
4. **`T10` is taken.** M0-PORT-PLAN.md:221 already names T10 (M0 trace and verdict). The M1 tasks below are
   **M1-T10…M1-T13**. Rename one of the two series before either lands in LEDGER.
5. **`DSparkBlock` captures `args.temperature` at construction** (model.py:1113, used at 1156). Our `Generator` sets
   only `model.temperature = 0` (runtime/generate.py:301-302). Left alone, the drafts would be Gumbel-sampled
   (model.py:1293-1295) from a random stream. The emitted tokens would not change under exact-match acceptance, but
   acceptance lengths and self-replay would. Set `mtp[i].temperature = 0` too.

## 1. The algorithm as the checkpoint defines it

### 1.1 Shapes

| item | value | source |
|---|---|---|
| draft stages | 3 (`mtp.0..2`, layer ids 40–42, window-only attention, ratio 0) | config.json:6, 52; model.py:1210-1216 |
| block size = drafts per step | **5** | config.json:7; production `num_speculative_tokens 5` |
| draft MoE | 128 routed experts, top-3, + shared | config.json:11-12; model.py:142-149 |
| target layers feeding the draft | 37, 38, 39 (their *attention input*, hc-mean) | config.json:9; model.py:1267-1269 |
| Markov rank / noise token | 256 / 128799 | config.json:8, 10 |
| **verify rows per step** | **6** = 1 bonus token + 5 drafts | below |

### 1.2 The draft stage (`forward_spec`, model.py:1278-1285)

Inputs: `input_ids` [b] is the token just produced by the target for position `start_pos+1`. `main_hidden`
[b, 1, 3·5120] is the hc-mean of the inputs of layers 37–39 at position `start_pos`. `start_pos` is that position.

1. `forward_embed` (1131-1138): `main_x = main_norm(main_proj(main_hidden))` (one fp8 Linear 15360→5120). Block ids =
   `[input_id, noise, noise, noise, noise]`, embedded with the shared embedding and expanded to hc_mult = 4 copies.
2. Three `DSparkBlock`s, each an ordinary hyper-connection `Block` (970-996) whose attention is `DSparkAttention`
   (1034-1077). Per stage it runs these steps:
   - `main_kv = act_quant(rope(kv_norm(wkv(main_x))))` at position `start_pos`, written into the stage's own 128-ring
     at `start_pos % 128` (1041-1044, 1067);
   - the 5 block rows get q and kv at positions `start_pos+1 … start_pos+5` (1057-1064);
   - attention runs over `[ring(128) | block(5)]` with **every block row seeing all 5 block rows** (non-causal), plus
     the ring in slot order (1022-1031, 1068-1069). vLLM's docstring calls this "semi-autoregressive" and implements
     the non-causal part through sparse indices (dspark.py:3-11).
3. `forward_head` (1140-1159): collapse the hc copies with the last FFN pre-mix, then `logits = head(norm(x))` for all
   5 rows. This is the shared fp32 head. Then a **sequential Markov chain** (1152-1156): for i = 0..4,
   `logits[i] += markov_head(output_ids[i])`, a 256-rank bigram bias that is an embed then a [129280×256] head, and
   `output_ids[i+1] = sample(logits[i])`. Draft i+1 depends on draft i, which is the only autoregressive part.
   `confidence = proj(cat(x, markov_embed))`, one scalar per block row (1092-1100, 1158). vLLM applies a sigmoid and
   calls it the "per-position acceptance probability for each drafted token" (dspark.py:358-363). What production
   uses it for is not in our tree.
4. Returns `output_ids` [b, 6] = `[input_id, d1..d5]`, logits [b,5,V], confidence [b,5]. `d_i` is the draft for
   position `start_pos+1+i`.

**At `start_pos == 0`** (prefill) each stage only seeds its ring with the prompt's `main_kv` (1046-1054, 1126-1128),
and `forward_spec` returns `None` (1283-1284). Prefill produces no drafts.

### 1.3 One step of the loop (our design)

State entering a step: committed prefix through position `A`, bonus token `b` for `P0 = A+1`, drafts `d1..d5` for
`P0+1..P0+5`.

1. **Verify:** the target runs 6 rows `[b, d1, …, d5]` at positions `P0..P0+5`, each row causal over the committed
   prefix plus the earlier rows of this step. Out: `t_i = argmax(logits_i)` (the prediction for `P0+i+1`), and
   `main_hidden_i`.
2. **Accept (exact greedy match):** `n = max{m ≤ 5 : d_j == t_{j-1} for all j ≤ m}`. Emit `d1..dn, t_n`, which is
   n+1 tokens (1…6). The committed prefix becomes `A' = P0+n`. The new bonus is `t_n`.
3. **Commit:** the KV and state of rows 0..n become permanent, and rows n+1..5 are discarded (§1.5).
4. **Draft:** write the draft rings with `main_kv` of rows 0..n (positions `P0..P0+n`). This is vLLM's
   `precompute_and_store_context_kv`, which inserts context KV for all accepted positions from the same `main_x`
   through each layer's own `wkv`/`kv_norm`/RoPE/quant (dspark.py:166-197). Then run the block at positions
   `A'+1..A'+5` with ids `[t_n, noise×4]`, which gives `d'1..d'5`.

**"DSpark k=5" therefore means:** 6 verify rows, 0–5 accepted drafts, **1–6 tokens per step**. Production measured
**2.75 / 2.63 / 2.93 tokens/step** (1.63–1.93 accepted drafts + 1) at 7.746 / 7.747 / 7.724 ms/step, which is
**≈2.8 ms/token** (`w0/results/phase1/step-pg61.json`: one prompt, 512 tokens, 3 reps; a different instrument from
the 7.36 ms release number, W0-RESULTS.md:6). Our acceptance is unmeasured. Production drafts with
`probabilistic` sampling and we draft greedy, so 2.6–2.9 is a prior, not a prediction.

### 1.4 Sampling-free, so bit-identity stays defined

Every choice is an argmax: target `sample(…, 0)` (model.py:1291-1292) and draft argmax once fix 0.5 is applied. Under
exact-match acceptance, **the emitted stream is the target's greedy decode computed with the verify step's
numerics**. It equals our own 1-row greedy stream bit for bit iff verify row i produces the same logits as a 1-row
step at that prefix (§4). Drafts change only how many tokens a step emits, never which tokens. Confidence is
recorded and does not gate anything.

Is production exact-match at T=0? Not readable here (0.1). The indirect evidence is our *non-speculative* runtime
against Pg61's *speculative* fixture: 0 margin-weighted flips, with divergences only at near-ties
(T9B1-RESULT-20261106.md:13; FIRST-LIGHT-20261106.md:10). Acceptance that let a non-argmax draft through would show up
as flips at confident positions. **Measured consistency, not proof.**

### 1.5 KV and state under rejection, cache by cache

| cache | indexed by | spec rows past `A'` | rollback |
|---|---|---|---|
| main window ring, 40 layers, 128 slots (model.py:664-669, 719) | `pos % 128` | **unsafe** | 256-slot ring indexed `pos % 256`, after which overwrite is safe (below) |
| compressed KV, sources L2/8/14 (r=2), L20 (r=1) (670-680, 762) | `pos // r` | garbage rows for groups past `A'` | overwrite-safe: the next step rewrites them before any read, and every read is masked by the row's own `compress_lens` |
| indexer K cache (520-525, 547) | `pos // r` | same | overwrite-safe |
| ratio-2 compressor `kv_state`/`score_state` (450-456, 476-482) | `pos % 2` | must hold only committed tokens | commit: if `A'` is even, `state[0] := kv/score of row n` |
| Engram hash cache (engram.py:156, 167-173) | `pos` | written, then gathered at `≤ pos` | overwrite-safe |
| draft rings ×3 (model.py:1046-1067) | `pos % 128` | never written; step 4 writes only rows 0..n | nothing to undo |
| `shared_attn` (model.py:1169-1183) | per forward | — | — |

**Why the ring breaks.** A step writes positions `P0..P0+5`, which evicts `P0-128..P0-123`. Row j needs
`P0+j-127..P0+j`. With write-then-attend (the reference decode order, 719-720), rows 0–4 lose 5−j of their oldest
window entries **in the same step**, even when every draft is accepted. Under rejection it gets worse: a rejected row
at `P0+i` (i ≥ n+2) has evicted a position the next step still needs, and the overwrite never brings it back.
Concrete case: P0=1000, n=0. Row 0 of the next step (pos 1001) needs 874–877, and slots 106–109 now hold the
rejected 1002–1005.

**Fix (adopted, matching T9B3-ATTN-DESIGN.md:222-226): a ring of ≥ 128+5 slots; use 256, indexed `pos % 256`.** A
step's writes then evict only positions ≤ `P0-128`, which no current or future row reads. Rejected rows' slots
`P0+n+1..P0+5` are a subset of the next step's writes `P0'..P0'+5`, and every row reads only positions ≤ its own,
all rewritten earlier in the same step. So **the window needs no rollback**.

- Row j's 128-entry index list, oldest first, is the same 128 values in the same order as a 1-row step at `P0+j`. That
  matters because `sparse_attn` accumulates in index order (T8-T9A-DESIGN.md:61).
- Cost: after the reference prefill (which seeds mod 128, model.py:709-717), one re-seed copy per layer. The 1-row
  graph must move to `% 256` as well, so that both paths share one cache layout. That is a one-line change of
  `p_win`/`win_idx` (static_step.py:174-178), to be re-proven at L0.
- Considered and not taken: ring + 6 side rows with a commit kernel after acceptance. It keeps the reference layout,
  but it adds a 40-layer commit to every step.

The ratio-2 compressor state still needs its commit (table above). That runs on device after acceptance, three layers
of `[2,512]` fp32.

## 2. The static-shape version

### 2.1 Verify graph: what changes from today's `StaticStep`

| piece | today (1 row) | 6 rows |
|---|---|---|
| inputs | `tok[1,1]`, `pos[1]` (static_step.py:316-318) | `tok[1,6]`, device `P0`, `n_prev`; positions `P0+arange(6)` |
| RoPE rows | `index_select(pos)` (98-102) | `index_select(P0+arange(6))`; the latent row `pos+1-r` per completed group |
| window | `index_copy_` at `pos%128`, attend ring (472-474) | 256-ring: write 6 slots `(P0+i)%256`, then attend; per-row oldest-first lists `[1,6,128]` int32 built on device (§1.5) |
| compressed / index-K writes | `index_copy_` at `pos//r` (422, 458) | 6 rows (r=1); 3 groups (r=2), write rows from device positions |
| **ratio-2 pooling** | **two graphs by pos parity** (66, 448; T8 #12) | **one graph**: 6 rows always complete exactly 3 groups. Take the 7-row sequence `[state0, r0..r5]` and offset `o = 1 - (P0 % 2)`, then `seq[o:o+6].view(3,2,·)` pools the 3 groups. Prove it equal to the per-parity reference path (M1-T10 L0) |
| indexer | `compress_lens`, `unreach`, `last` scalars (183-187, 433-440) | per-row `[6,1]` tensors, so each row sees groups with end ≤ its own position; per-row top-k 512 over fixed width `W_r` |
| `sparse_attn` | grid (m=1, b) | grid (6, b): one CTA per query row (kernel.py:335), row-independent |
| MoE routing | sort 6 ids, gather 6 experts to `[6,N,K/2]` (232-249) | 36 (row, expert) pairs → stable sort by id → unique flags → slot = cumsum → **U ≤ 36 staging slots**, gather predicated on first-occurrence; `masked_m[u]` = rows per slot (≤ 6); row r's activations placed at `xq[u, rank]` |
| DG masked GEMM | `[6,128,K]`, `masked_m = ones(6)` (dg_moe.py:168-176) | `[36,128,K]` buffers; `masked_m` device-written per step; `expected_m` kept at 8 (same config as 1 row) |
| combine | `y += out[j]` in slot order (dg_moe.py:273-276) | per row, its 6 outputs in ascending expert id: the 1-row order |
| fp8 Linears | GEMV M=1 | GEMV M=6 (`gemv_mxfp8.py:1`: M ≤ 8; measured 3.47 vs 3.43 ms/step, gemv-bench-r2) |
| head | `x[:, -1]`, `logits[1,V]` | `full_logits`, `logits[6,V]` (model.py:1010-1014) |
| new outputs | — | `main_hidden[6,15360]` (3 hc-means, model.py:1268-1269); `t[6]` |
| loader | overlay `dspark_block_size=0`, targets `()`; `mtp.*` dropped (loader.py:59-64; placement.py:143-144); `StaticStep` refuses target layers (static_step.py:286-287) | load `mtp.*` into HBM (7.54 GiB, M0-PORT-PLAN.md:102, inside the 21.16 GiB slack at :105); keep `dspark_block_size=0` on the backbone module and build the draft as our own static module |

### 2.2 Acceptance, commit, emission: all on device

At the end of the verify graph:

- `match = (d[1:6] == t[0:5])`
- `n = cumprod(match).sum()`
- `bonus = t.gather(n)`
- emit through 6 predicated stores into a device `out_ids[max_new+6]` at `out_ids[n_out + i]` for `i ≤ n`
  (`d_{i+1}` for `i < n`, `bonus` at `i = n`); then `n_out += n+1`, `A' = P0+n`, `P0' = A'+1`
- the compressor-state commit (§1.5) runs under the same `n`

**Host sync is avoidable.** The host replays step graphs back to back and never needs `n`. Every N steps (est. 8–16)
it issues a non-blocking D2H of `(n_out, done)` into pinned memory and checks an event; `done` is set on device by
EOS or by `n_out ≥ max_new`. Steps after `done` become no-ops through predication (no emission, positions frozen).
The cost is ≤ N−1 wasted steps at the end of a sequence. The only syncs left are prefill and parity mode (per-step D2H
of the 6 rows' logits, for logprob records).

### 2.3 Draft graph

1. Context write: `main_x = main_norm(main_proj(main_hidden[6]))` (GEMV, new shape 5120×15360), then per stage
   `wkv`→`kv_norm`→RoPE at `P0+i`→`act_quant`→ring slot `(P0+i)%128`, predicated `i < ctx_count` (`ctx_count = n+1`).
   Stage MoE, mHC and so on run 5 rows: block positions `A'+1..A'+5`, the window `[ring(128) | block(5)]` with
   reference slot order (1022-1031).
2. Draft MoE: top-3 of 128 at 5 rows means ≤ 15 unique experts per stage, all HBM. Reuse the 6-row dedup path with
   k = 3.
3. Head on 5 rows, then the 5-step Markov chain (sequential argmax), then confidence. Write `d'[1..5]` into the verify
   graph's `tok[1,1:6]` and `t_n` into `tok[0]`.

**After prefill.** The `start_pos == 0` path seeds the rings (model.py:1126-1128). The first drafts come from the
*same* draft graph with `ctx_count = 0` and `A' = L-1`. That is the decode-path `forward_spec(first_token, ·, L-1)`
without re-writing slot `(L-1)%128`, which a reference call would recompute at decode numerics.

### 2.4 Graph count

The step graph is verify → accept → commit → draft, with no host decision anywhere, so **one CUDA graph per step**.
Capture it first as two graphs replayed back to back (verify, draft) so the instrument can attribute ms. The 1-row
parity graphs stay for the M0 path. Prefill stays eager.

## 3. Cost model at 6 rows, from measured 1-row numbers

Today's 1-row step is **32.16 ms** (T9B1-RESULT:15). Its split is partly measured and partly census estimate
(T9B1-RESULT:22).

| role | 1 row | basis | ×6 rows | 6 rows today (est.) |
|---|---|---|---|---|
| fp8 Linears, 290 GEMV | 3.43 | bench, measured | 3.47 measured | 3.5 |
| glue ~5,300 launches | ~5 | census est. | launch-bound, ×1.0–1.2 | 5–6 |
| `sparse_attn` + indexer | ~3 | census est. (sparse_attn alone 40×75 µs, anatomy) | 6 CTAs instead of 1; top-k per row, ×1.0–1.5 | 3–4.5 |
| act_quant ×450 | ~1.2 | census est. | grid over ceil(M/blk_m) (kernel.py:60), ×1.0–1.1 | 1.2–1.3 |
| head fp32 2.65 GB | ~0.4 | anatomy role | same bytes | 0.4–0.5 |
| Engram | ~0.4 | anatomy role | 6 small gathers | 0.4–0.6 |
| unattributed (aten mm ×187, graph node gaps, cat stopgap…) | ~14.8 | 32.16 minus the rows above | assume row-invariant, ×1.0–1.15 | 14.8–17 |
| **R, non-routed-MoE subtotal** | **25.8–28.2** | | | **≈ 26–32** |
| routed MoE | 4–6.4 | census ~4; bandwidth model 4.5–6.4 (below) | unique experts 6 → 22.45 | see next table |

MoE bytes per step use 18.80 MB per expert (M0-PORT-PLAN.md:86) and the unique experts per layer at T=6 rows from
production routing, **22.45** (22.09–23.39 across workload classes, `docs/data/unique_experts_vs_k.json`
`ALL["5"]`). Rates come from C2C-PROBE-20261106.md:8-9, 31: HBM→HBM copy 0.341 ms/GiB, C2C 366–390 GB/s by copy
engine, 230 GB/s by a naive SM reader. Our static map is cold for 24.6 % of the unique experts at T=6
(`grace_share_positional`), which agrees with the measured 56.5 of 240 = 23.5 % at 1 row
(first-light-gemv `cold_calls_per_step`).

| MoE design × residency, 6 rows | hot bytes | cold bytes (C2C) | ms/step (est.) |
|---|---|---|---|
| today's staging (copy every expert, then DG) × positional map | copy 12.7 GB → 4.0 | 4.15 GB → 10.6 (390) – 18.1 (230) | + DG read 16.9 GB ≈ 2.7 → **17–25** |
| in-place hot (DG contiguous over `[H+S]` or MegaMoE, T8-T9A §3.2) × positional | read 12.7 GB ≈ 2.0 | 4.15 GB → 10.6, at most ~2 ms hidden | **11–13** |
| in-place × Pg61-like map (1.6–2.06 cold/layer; peer3 `by_T["6"]`, C2C-PROBE:18) | ≈ 2–3 | 1.2–1.55 GB → 3.1–4.0 | **5–7** |
| in-place × usage-ranked map (1.4 % cold, `grace_share_usage_cold`; in-sample, optimistic) or adaptive ≤ 0.41/layer (C2C-PROBE:19) | Pg61 MegaMoE ≤ 79 µs/layer × 40 ≈ 3.2 (T8-T9A-DESIGN.md:179) | 0.24–0.31 GB → 0.6–0.8, overlappable (overlap probe) | **2–3.2** |

**Draft stage (est.).** Three window-only layers at 5 rows cost about a ratio-0 layer each: R/40 ≈ 0.7 ms plus a
≤ 15-expert all-HBM MoE ≈ 0.15. Add the head (0.4), the Markov chain (5 × a 132 MB fp32 GEMV + argmax, ≈ 0.2–0.3),
`main_proj` and the context writes (≈ 0.1). Today ≈ **2.5–4.5 ms**; after T9b ≈ **1.5–2.5**.

**Projected ms per 6-row step (est.)**

| scenario | R | MoE | draft | **step** | ms/token at 2.63–2.93 tok/step |
|---|---|---|---|---|---|
| A: today's kernels, today's map | 26–32 | 17–25 | 2.5–4.5 | **46–62** | 16–24 |
| B: + T9b-2/T9b-3 (the kernel-set floor of 15–20 ms at 1 row, T9B1-RESULT:22, minus ~4 MoE) | 11–18 | 17–25 | 1.5–2.5 | **30–46** | 10–17 |
| C: B + in-place hot MoE | 11–18 | 11–13 | 1.5–2.5 | **23–34** | 8–13 |
| D: C + usage-ranked / adaptive residency (A2) | 11–18 | 2–3.2 | 1.5–2.5 | **15–24** | 5–9 |
| Pg61 (measured) | | | | **7.36** release / 7.75 W0 | 2.8 |
| charter bar | | | | **≤ 5.0** | |

**What this says, plainly:**

- M1 beats our own 1-row runtime per token in every scenario: 32.2 ms/token today against 5–24.
- **M1 does not beat Pg61 in any scenario.** The best case, D, is 2–3× Pg61 per step.
- Once residency is fixed, the residual is **R**: the latency-bound, launch-bound non-MoE work. It costs about the
  same at 6 rows as at 1, which is why speculation helps, and it is 11–18 ms even after T9b. Reaching 5.0 needs R
  near 1–2 ms. That is M2's fused chain, not M1.
- The second gate is **residency (A2)**. Under today's positional map, cold C2C bytes alone (≥ 10.6 ms) exceed 7.36.
- Uncertainty: the largest term is the 14.8 ms of unattributed 1-row time. Assuming it is row-invariant is the bet
  in every row above. M1-T10's first timing settles it, and so would M0's T10 trace.

## 4. Parity instrument for M1

Acceptance is exact-match greedy (§1.3), so the spec runtime must reproduce **greedy decoding of the same target**.
It has two references and five levels.

| level | test | pass |
|---|---|---|
| V0 verify == sequential | teacher-forced: one 6-row verify on known tokens vs 6 consecutive 1-row static steps from the same state; per-row fp32 logits and every per-layer intermediate (hooked) | **`torch.equal` in row-invariant mode**. Fast mode: report the op-level invariance census (which ops differ M=1 vs M=6) and the mismatch rate |
| V1 rollback | scripted acceptor that forces n = 0..5 at chosen steps, plus random-draft fuzz (n ≈ 0); compare the next 32 steps' logits with an unspeculated run | `torch.equal` (row-invariant mode); proves the 256-ring overwrite argument and the compressor commit |
| V2 draft == reference | our draft graph vs reference `forward_spec` decode path on the same ring state, ref kernels; also prefill seeding then first draft | `torch.equal` on ids, logits and confidence |
| V3 spec == our greedy | 8 W0 prompts, 256 tokens: spec emitted stream vs the M0 1-row graph | byte-identical if V0 is strict; otherwise first divergences only at near-ties (the A3 rule), 0 margin-weighted flips |
| V4 vs Pg61 | same fixture and A3 bar as M0 (0 margin-weighted flips, mean \|Δlogprob\| ≤ 0.05); logprobs come from the verify row that emitted each token | A3. **The W0 fixture is already DSpark output** (Pg61 serves with the spec config), so no new capture is needed |
| V5 self-replay | 2 runs + relaunch | tokens, fp32 digests **and the per-step accept-length sequence** byte-identical (decision 14) |

**Row-invariant mode** is what makes V0, V1 and V3 strict.

- Known invariant by construction: `sparse_attn` (one CTA per row), act_quant (per-row blocks), elementwise ops.
- GEMV: invariant iff `numerics_key` matches at M=1 and M=6. One shape differs today: 2304×5120, `[18,9,1]` vs
  `[10,16,1]` (gemv-bench-r2). Pin one config.
- To test: DG with `masked_m > 1` vs `= 1`, torch row reductions (RMSNorm, softmax) at `[6,5120]` vs `[1,5120]`, and
  `topk` tie order on `[6,W]` vs `[1,W]`.
- Likely not invariant: the 187 cuBLAS `mm`/`bmm` (hc mixes, gate, `wo_a` einsum, fp32 compressor linears, indexer
  einsum, head). The mode runs those per row (6 M=1 calls). That is debug only.

The headline is graded by fast mode against the A3 bar. If the census shows only the cuBLAS ops break invariance,
replacing them with fixed-order kernels would make fast mode strict too. That is worth doing because it turns every
future M1 regression into a `torch.equal`.

**Headline instrument.** C1 per PROJECT.md:16: nsys `cudaProfilerApi` range over N step replays, with a marker kernel
per step. Report ms/step (verify, draft, total), tokens/step and ms/token on the W0 fixture plus the W0 step prompt,
next to Pg61's 7.746 ms / 2.75 tok on that same prompt.

## 5. Tasks, gates, order, risks

| task | scope | gate |
|---|---|---|
| **M1-T10** 6-row verify graph, teacher-forced, no drafts | §2.1 entire: positions, 256-ring (shared with the 1-row graph), single-graph ratio-2 pooling, per-row indexer, MoE dedup and `masked_m`, `logits[6,V]`, `main_hidden`; no `mtp` weights yet. **Box day 1:** time the 6-row graph against the 1-row graph, which is the R6/R1 number every projection rests on | V0 strict in row-invariant mode, plus the invariance census in fast mode; graph == static-eager; self-replay; **ms per 6-row verify reported** (no bar yet; the table predicts 43–57) |
| **M1-T11** draft graph | load `mtp.*` to HBM; static 3-stage DSpark at 5 rows; context write ≤ 6; Markov chain; confidence; temperature fix | V2 strict; ms per draft (est. 2.5–4.5) |
| **M1-T12** the loop | accept, commit kernel, emission buffer, async EOS poll, prefill → first draft; step graph(s) | V1, V3, V5; tokens/step reported |
| **M1-T13** headline + parity | C1 nsys instrument; V4; the long-horizon quality check that A3 deferred to M1 (PROJECT.md:95) | first honest ms/step vs 7.36, published with ms/token |

Order: T10 → T11 → T12 → T13. T10 can start on the current kernels without waiting for T9b-2/T9b-3. T11 can run in
parallel with T10 because it needs only the ring contents and `main_hidden`, which can be fed from the reference
eager path for V2. The in-place MoE and A2 residency are separate tracks that the headline depends on (§3).

Risks, ranked:

1. **Window ring hazard** (§1.5). Silent corruption that a short-context test would not catch. Mitigation: the
   256-ring, V1 forced rejections, and V0/V1 cases at positions ≥ 256 so the ring has wrapped.
2. **Row invariance.** It decides whether V0, V1 and V3 can be `torch.equal` or only drift-graded. The census comes
   first in T10.
3. **Indexer with 6 query rows.** Per-row `compress_lens`, the candidate-block `last` per row (L20), and top-k tie
   resolution changing with the batch shape (T8 #9/#11, now per row). The fixture never reaches the top-512 regime
   (M0-PORT-PLAN.md:191), so this needs the long-context check.
4. **DG `masked_m > 1` and the dedup mapping.** These are new invariants in `DGBuffers` (rows ≥ 1 were "never written
   again", dg_moe.py:144-145). An off-by-one in slot or rank corrupts silently. Test with a mapping round-trip on CPU
   and then `torch.equal` against the `ref` MoE at 6 rows.
5. **MoE bytes.** At the positional map, 6 rows move about 4.15 GB/step over C2C (est.). The M1 headline is hostage
   to A2 and to the in-place MoE.
6. **Graph count and parity.** The two parity graphs cannot survive data-dependent `P0`. The single-graph pooling
   must be proven equal to both reference parities.
7. **`start_pos == 0` behaviour.** No drafts from prefill. The first draft uses the decode path with `ctx_count = 0`
   (§2.3). Rewriting slot `L-1` at decode numerics would differ in bits from the prefill seed.
8. **Acceptance is ours to measure.** Production's probabilistic drafts and the possible use of the confidence head
   are not visible in our tree, so 2.6–2.9 tok/step may not transfer.
9. **HBM.** MTP 7.54 GiB, 36-slot staging 0.63 GiB, 36-group DG buffers (est. < 0.2 GiB): fits the 21.16 GiB slack
   (M0-PORT-PLAN.md:105), with less left for D3's hot-set growth.

M1-DESIGN: projected 46–62 ms per 6-row step on today's kernels and map (16–24 ms/token), 15–24 ms after T9b-2/3 + in-place MoE + adaptive residency — above 7.36 in every case, 5.0 needs M2; first task M1-T10 (teacher-forced 6-row verify graph on a 256-slot window ring, V0 invariance census, and the R6/R1 timing).
