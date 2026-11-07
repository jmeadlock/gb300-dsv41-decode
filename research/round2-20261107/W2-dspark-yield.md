# W2 — DSpark / MTP: production's loop and the token yield

Worker: W2 (Milo, Hermes milo profile, claude-opus-5-5), round 2, started 2026-11-07.
Status: complete (sections F1-F5, M, N, W, P, S).

## Verdict

Token yield cannot carry the 5.0-ms bar; it only converts step ms into token ms. Production (Pg61, vLLM 0.31 model
runner v2) verifies **6 rows** at batch 1 (bonus + 5 DSpark drafts). At T = 0 it accepts by **exact greedy match**:
`target_argmax == draft`, read from the v0.31.0 kernel. The configured `block`/`probabilistic` methods do nothing at
T = 0, and the confidence head is **off** (`enable_adaptive_verification` unset). Production gets 2.63-2.93 tok/step on
technical prose [M]. Published V4.1-Flash numbers put code/math/JSON at 3.7-4.3 and prose/chat at 2.2-2.9 [C].

More drafts do not help:
- The drafter is a fixed block-5 network with non-causal block attention. Running 7 query rows on a block-5 DSpark
  measured worse (tau 1.43-2.51 vs 3.03) [C].
- Rows are not free on our runtime: ~2.1 ms/row today [Mod from M], ~0.3 ms/row at the bytes floor [Mod].

The confidence head cannot skip a draft, because it is computed after the draft forward. What it can do is truncate
verify depth. Fixed depth 3-4 saves 1-8 % ms/token, and a perfect per-step depth bounds the gain at 17-29 % on prose
[Mod]. Done device-side through the MoE's `masked_m`, this is bit-identical.

The emitted stream equals our 1-row greedy stream exactly when every verify row is row-invariant: the GEMV pinned to
its M = 1 config, and the head fixed (W3/W4). Draft-side numerics never change emitted tokens.

Pricing:
- Today: (21 + ~2) / 2.77 ≈ 8.3 ms/token, against Pg61's 2.8.
- At the bar: ≈ 1.8 ms/token.
- The 6-row step itself must fall ~4.6x.

Legend: [M] measured (receipt cited), [Mod] modeled (arithmetic shown), [C] claimed by source (URL + date), [U] unverified.

## F1. The DSpark paper (DeepSeek-AI + PKU), arXiv 2607.05147v1, 2026-07-06

Source: https://arxiv.org/html/2607.05147v1 (fetched 2026-11-07).

- Architecture [C]: parallel backbone (DFlash-style: target hidden states from a few layers concatenated -> projection
  `W_c` -> RMSNorm -> injected as extra K/V "context" into every draft layer; all block positions attend
  bidirectionally to each other and to the context) + a lightweight sequential head. Modification vs DFlash: the
  anchor itself is the first prediction position, so gamma input tokens (anchor + gamma-1 masks) yield gamma
  draft logits. This is exactly our checkpoint: `dspark_block_size=5`, ids `[anchor, noise x4]` -> d1..d5
  (`reference/deepseek-inference/model.py:1131-1159`).
- Markov head [C]: first-order low-rank transition bias `B(x_{k-1}, .) = W1[x_{k-1}] W2`, r = 256 default
  (our `dspark_markov_rank=256`). RNN-head variant exists; "marginal additional gains", Markov is default and is what
  V4 ships.
- Confidence head [C]: `c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]])`, the *conditional* probability that draft k survives
  given all earlier drafts survived; supervised by the analytical acceptance `1 - 0.5*||p_d - p_t||_1` (i.e. trained
  for *sampled* rejection, not for greedy match); post-hoc "Sequential Temperature Scaling" calibration; raw
  ROC-AUC 0.81-0.90, ECE 3-8 % -> ~1 % after STS (Qwen3-4B, offline).
- Confidence-scheduled verification [C]: a hardware-aware prefix scheduler (Alg. 1) that greedily admits draft
  tokens across the *batch* in order of prefix survival `a_{r,j} = prod c`, maximizing `tau * SPS(B)` where SPS(B) is
  a profiled steps/s-vs-batch-tokens table. In production it runs asynchronously: the truncation length K uses
  confidence from two steps earlier (CUDA-graph / zero-overhead-scheduling compatible), ranking uses current
  confidence. Purpose is explicitly *throughput under concurrency*: "verifying extra tokens is nearly free under
  light loads". At low concurrency the scheduler expands verification "from MTP-1's static 2 tokens to roughly
  4-6 tokens per request" (Fig. 8, V4 production).
- V4 deployment [C]: draft = 3 MoE layers with mHC and sliding-window attention 128, max block gamma = 5, Markov head,
  STS-calibrated confidence head. This matches our checkpoint's `mtp.0..2` (window-only, 128-ring) exactly.
- Production results [C]: vs the prior MTP-1 production baseline, per-user speed +60-85 % (V4-Flash) at matched
  throughput; +51 % aggregate throughput at an 80 tok/s/user SLA. No batch-1 latency figure and no per-step ms
  is given. Note the V4 report says DSpark replaced MTP-1 "two weeks following the DeepSeek-V4-preview release".
- Offline accepted length tau (incl. bonus token), gamma = 7, T = 1.0 sampling, Qwen3 / Gemma targets, 5-layer drafter
  [C, Table 1]: math 4.9-6.2, code 4.5-5.6, chat 2.9-3.7. Qwen3-8B DSpark: GSM8K 6.17, MATH 5.78, AIME25 5.01, MBPP
  5.16, HumanEval 5.52, LCB 5.17, MT-Bench 3.72, Alpaca 3.58, Arena-Hard 3.21. These are small dense targets with a
  dedicated 5-layer drafter at block 7 — an upper-bound prior, not a V4.1-Flash number.
- Position-wise conditional acceptance [C, Fig. 2, Qwen3-4B]: DSpark starts ~0.93 at position 1 on math and stays
  "high and stable" across the block; DFlash code 0.87 -> 0.78, chat 0.72 -> 0.63. Only figure values quoted in text
  are available; the curves themselves are images (unverified beyond the quoted points).
- Block size scaling [C, Fig. 4]: drafts 4/8/12/16; DSpark's gain over DFlash grows with gamma (+16/15/18 % at 7, +30/26/22 %
  at 15). Draft-loop overhead +0.2-1.3 % of round latency at batch 128 (target-dominated regime — not ours).
- Limitation stated by the authors [C]: "DSpark still incurs a fixed draft-side cost to generate the initial
  gamma-token block ... For complex queries with inherently low acceptance rates, this upfront drafting compute is
  unrecoverable." They propose difficulty-aware early exit as *future work* — i.e. the confidence head is not
  used to skip drafting in the published system.
- Losslessness subtlety [C, App. A]: for *sampled* rejection, a truncation rule that peeks at c_{k+1} (which depends
  on x_k through the Markov embed) biases the output distribution. Under exact-greedy acceptance this issue does
  not exist: any truncation length emits the same greedy stream (see F5).

## F2. vLLM 0.31.0: where the loop lives and what it decides (source read at tag v0.31.0)

Our vendored tree has only the draft *model* (`reference/vllm-0.31.0/deepseek_v41/nvidia/dspark.py`). The loop is in
vLLM's "model runner v2" (`vllm/v1/worker/gpu/...`), fetched read-only from
https://github.com/vllm-project/vllm/tree/v0.31.0 on 2026-11-07 (raw.githubusercontent.com; files listed in Sources).
DSpark exists only in MRV2: `vllm/v1/worker/gpu/spec_decode/__init__.py:init_speculator` dispatches
`method == "dspark"` to `DSparkSpeculator`; there is no DSpark proposer in the legacy `vllm/v1/spec_decode/`.

What production (Pg61) runs [M, from `w0/results/ds41-args.json`]: `--speculative-config {"method":"dspark",
"num_speculative_tokens":5, "num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,32,3]],
"draft_sample_method":"probabilistic", "rejection_sample_method":"block"}`. Not set: `enable_adaptive_verification`
(default False, `config/speculative.py:552`) — **so production never uses the confidence head** (per its launch args;
Pg61's Garner-engine patches are not in our tree and were not read — unverified that they leave this path alone).
`DSparkSpeculator.load_draft_model` sets `use_confidence_head = enable_adaptive_verification and head present`
(`spec_decode/dspark/speculator.py:106-113`); with it False the head is loaded but not evaluated. Batch 1-4 -> 5 drafts
(6 verify rows); batch 5-32 -> 3 drafts.

1. **Acceptance at temperature 0 is exact greedy match — confirmed from code.** `_rejection_kernel`
   (`spec_decode/rejection_sampler_utils.py:485-668`): `is_greedy = temp == 0.0`; for greedy requests
   `accepted = target_argmax == draft_sampled`, the first mismatch stops verification and stores the *target argmax*
   as the corrected token (`:571-595`). The `block` branch (Sun et al. 2024 block verification, `:596-628`) and the
   standard Leviathan ratio test (`:629-666`) are only reached when `temp != 0`. `rejection_sample_method: block` and
   `draft_sample_method: probabilistic` are therefore **inert at T = 0**. The target argmax is a two-level
   argmax (per-vocab-block argmax, then argmax over block maxima, `_compute_global_target_argmax :95-118`) — first
   index wins within a block and the first maximal block wins, i.e. first-occurrence like `torch.argmax`, modulo NaN
   handling. Emitted stream = target greedy stream computed with the *verify* step's numerics.
2. **Drafts at T = 0 are argmax too.** `DSparkSpeculator._sample_logits` -> `gumbel_sample(...)` ->
   `gumbel_noised_argmax`, documented "plain argmax at temp 0" (`sample/gumbel.py:165-207`); the draft temperature is
   the request's (`prepare_dflash_inputs` copies `temperature` per request). So W0's "greedy" step cell (2.63-2.93
   tok/step) is an exact-greedy-acceptance number — the same rule our M1 design uses. Our reference sets
   `DSparkBlock.temperature` from args; M1 fix 0.5 (temperature 0 on `mtp[i]`) matches production behaviour.
3. **Rows per step.** Verify: the target runs `1 + N` rows per request (bonus + N drafts; `cu_num_logits` spans
   `num_draft_tokens + 1`, `rejection_sampler_utils.py:544-546`); N = 5 at batch 1 -> **6 verify rows**. Draft: exactly
   `N` query tokens per request (anchor + N-1 noise, "anchor-as-first-prediction", `dspark/speculator.py:10-17, 52-53`)
   -> **5 draft rows**, plus context-KV rows for the accepted target positions.
4. **Rejected positions' KV.** Target: paged KV; rejected rows' entries stay in the pages and are overwritten next step
   (standard vLLM: seq_len only advances by accepted+1; `get_num_sampled_and_rejected`). Draft: in
   `_prepare_dflash_inputs_kernel` (`dflash/speculator.py:570-622`) `valid_ctx_end = ctx_end - num_rejected`; only
   accepted rows get a context slot, rejected rows get `PAD_SLOT_ID` ("rejected suffix rows are invalid context ...
   may not write draft KV"). The 5 query rows *do* write the draft SWA cache at positions `last_valid_pos+1 ...`, and
   the next step's context insert overwrites them. Same commit rule as M1 design §1.3 step 4 (rows 0..n).
5. **Context KV** = `fused_wqa_wkv(main_x)` -> kv part -> `kv_norm` -> `fused_deepseek_v4_kv_rope_insert` per draft
   layer (vendored `dspark.py:166-197, 252-275`), `main_x = main_norm(main_proj(cat(aux hidden)))`.
6. **Draft graph.** One FULL CUDA graph covers the backbone forward *and* the sequential Markov loop
   (`dspark/speculator.py:22-23`; `DFlashCudaGraphManager`). Verify is the target's own FULL decode graph at 6 tokens.
   Rejection sampling, input prep (`prepare_dflash_inputs`), and the context-KV precompute for rows beyond the graph's
   capacity run as separate launches; MRV2 async scheduling hides the host. **No public per-role timing exists** (our
   W0 nsys attempt deadlocked, `w0/W0-RESULTS.md`).
7. **Markov head cost knob.** `dspark_draft_topk` (`config/speculative.py:609`; `_sample_sequential_topk`
   `dspark/speculator.py:201-248`): apply the Markov bias only to the top-k base-logit candidates instead of a full
   [V x 256] GEMV per position. Not set in production. Changes draft argmax only when the Markov-boosted winner lies
   outside the base top-k.
8. **Adaptive verification** (`spec_decode/adaptive_verification.py`, opt-in): profiles draft and verify cost curves
   per padded CUDA-graph size, scores every (request, step) slot by cumulative confidence and keeps the global top
   `draft_budget` (`_assign_draft_token_budget`). It needs varlen verify graphs. At batch 1 on a latency-bound engine
   the verify curve is flat, so it would admit all 5 — which is why nobody uses it at batch 1. (Our runtime's curve
   is *not* flat: R6/R1 = 2.0, see §M.)
9. **num_speculative_tokens is not tied to the checkpoint block size.** vLLM's own GSM8K smoke configs run
   `deepseek-ai/DeepSeek-V4-Flash-DSpark` (config `dspark_block_size: 5`, fetched from HF 2026-11-07) with
   `num_speculative_tokens: 7` (`tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-*.yaml`, `min_acceptance_length: 1.2`
   "very loose lower bound for smoke testing"). Because block rows attend to each other non-causally, running 7
   query rows changes d1..d5 too (out of the training distribution). No published acceptance numbers for N > 5 on V4.

## F3. SGLang's loop (for contrast)

- SGLang PR #29538 "[Spec] Add DSpark speculative decoding for DeepSeek-V4" (https://github.com/sgl-project/sglang/pull/29538,
  fetched 2026-11-07; merged around 2026-07-06 per the follow-up commit mention): native block drafter (3-stage MTP stack +
  Markov refine + confidence head), greedy verification is exact-match ("greedy losslessness" section), sampled requests use
  a target-only sampled verify; unit tests cover "shard-argmax pack exactness (forced ties, signed zeros, vocab padding)".
  Their numbers [C]: DeepSeek-V4-Flash-DSpark (V4, not V4.1), 8x B200, TP8/EP8, greedy, c = 1: accept length 3.72 (random
  256/512), 3.72 (ShareGPT), 3.58 (single stream 512/256 and 1024/1024); median TPOT 2.52 ms vs no-spec 5.98 ms (2.08x).
  Implied DSpark step = 3.72 x 2.52 = 9.4 ms vs 5.98 ms plain step, i.e. step(6 rows + draft) / step(1 row) = 1.57 [Mod from C].
- LMSYS blog "DSpark in SGLang" 2026-07-06 (https://www.lmsys.org/blog/2026-07-06-dspark-sglang): integration PR #30261;
  verify modes `static` (full block), `compact` (confidence-trimmed, ragged, full CUDA graph keyed on total tokens rounded
  to a captured tier), `cap-accept` (measurement). Claims [C]: "At batch size 1 the target verify does not slow down much
  with more tokens, so trimming saves little and the two arms tie" — the trim win is a high-batch effect. Mixed-traffic
  windows (block 6, cap-accept): gsm8k 5.24, arena-hard 3.78, poetry 2.91 tokens/step, 0.88-0.97 of the untrimmed ceiling.
  Batch-1 V4-Pro TP8 on B300: 383.7 tok/s at accept length ~5 -> 13.0 ms/step [Mod from C]; "things outside the target
  verify shrink by 1.7 ms, against a 7.3 ms verify" — draft + glue is a material share of a batch-1 step even there.
  Their additive step model `T(bs,K) = bias + alpha(bs) + theta(bs+K)`; only theta is recoverable by trimming.
- SGLang V4.1 cookbook (https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1, fetched 2026-11-07): "DSpark
  is DeepSeek-V4.1 Flash's own bundled draft — there is no EAGLE or MTP path"; `--speculative-dspark-block-size 5`; "the
  step cost does not move with accept length or with the draft block size, so tuning --speculative-dspark-block-size will
  not buy throughput"; acceptance "highest on maths and code, lower on open-ended chat, lower still on agentic traces".

## F4. Published acceptance for this family (all claimed unless marked M)

| source (date) | model / engine | sampling | depth | tokens per step (incl. bonus) |
|---|---|---|---|---|
| our W0, `w0/results/phase1/step-pg61.json` (2026-11-06) | **V4.1-Flash-1017, Pg61 vLLM, 1x GB300** | T=0, technical prose (STEP_PROMPT), 512 tok | 5 | **2.75 / 2.63 / 2.93 [M]** at 7.746 / 7.747 / 7.724 ms/step |
| dgpp model card (https://github.com/HawkBearPig/dgpp/blob/master/docs/model_cards/DeepSeek-V4.1-Flash-1017.md, 2026-10-18..20) | V4.1-Flash-1017, dgpp engine, 4x DGX Spark TP | T=0, 3x256 tok per class | fixed 1/2/3/4 | prose 1.66/2.02/2.16/2.24; code 1.83/2.55/3.27/3.75; JSON 1.95/2.87/3.54/4.25; math 1.92/2.80/3.40/3.98; chat 1.70/2.18/2.45/2.58 |
| same | same | T=0, schedule on | conditional acceptance per position | code 86/79/88/86 %; JSON 94/89/82/78 %; prose 63/46/24/67 % |
| allover326/deepseek-v4-cmp170hx RESULTS.md (https://github.com/allover326/deepseek-v4-cmp170hx/blob/main/RESULTS.md, fetched 2026-11-07) | V4-Flash-0731 (block 5), vLLM PP4 | T=0, tech/prose/code, 400 tok | 5 | 3.03; unconditional per-position 0.730/0.569/0.372/0.226/0.131 |
| same | same | T=0 | **7** (`num_speculative_tokens` 7 on a block-5 drafter) | **1.43-2.51**, 60.3 vs 98.1 tok/s at 5; "4 and below are rejected by vLLM for this checkpoint" |
| SGLang PR #29538 (2026-07) | V4-Flash-DSpark, 8x B200 | greedy, random / ShareGPT | 5 | 3.58-3.72 |
| InferenceX PR #4783 (https://github.com/SemiAnalysisAI/InferenceX/pull/4783, 2026-11-04) | V4.1-Flash, vLLM GB200 | synthetic | 5, adaptive off | "golden acceptance length (3.51, from the srt driver)" — an agentic-trace AL used for *simulated* acceptance, not a measurement of a run |
| DSpark paper Table 1 (2026-07-06) | Qwen3-4/8/14B, Gemma4-12B, 5-layer drafter, block 7 | T=1 | 7 | math 4.9-6.2, code 4.5-5.6, chat 2.9-3.7 |

Readings:
- The two V4.1-Flash-1017 sources agree on the shape: prose/chat 2.2-2.9, code/math/JSON 3.7-4.3 (depth 4) and ~4.0-4.6
  extrapolated to depth 5 [Mod]. Our own W0 prompt is the prose end. Note "agentic traces" are reported as lower still.
- Acceptance is not a free variable we can tune: the drafter is a fixed block-5 network; depth > 5 is out of
  distribution and measured worse (cmp170hx N=7). Depth < 5 is possible by *truncating verification* (not drafting).
- Determinism caveat [C]: cmp170hx reports vLLM DSpark output at T=0 "differs from non-speculative output on all 6 probe
  prompts, and differs between two runs of the same server". Our W0 reps agree: the same prompt took 186/195/175 steps
  at T=0 [M] — production's speculative decode is not run-to-run deterministic. Our runtime's exact-match + row-invariant
  design is what makes bit-identity achievable; production does not provide it.

## F5. What production's 7.36 ms step contains (what is known, what is not)

- Composition by construction (vLLM 0.31 MRV2 code, F2) [C-from-code]: (1) target verify FULL graph at 6 tokens (bonus +
  5 drafts), including the fp32/bf16 head on 6 rows; (2) `rejection_sample` kernels (per-vocab-block stats + argmax, a
  few Triton launches on 6x129280 logits); (3) `prepare_dflash_inputs` + `combine_hidden_states` (main_proj GEMV
  15360->5120 on accepted rows) + `precompute_and_store_context_kv` for the accepted rows that exceed the graph's
  pre-recorded context; (4) the draft FULL graph: 3 draft decoder layers on 5 query rows (window attention via sparse
  indices, 128-expert top-3 MoE), `hc_collapse`, head on 5 rows, the 5-step Markov loop (`markov_w1` embed +
  `markov_w2` [129280 x 256] GEMV + argmax each), no confidence head (adaptive off); (5) MRV2 async scheduling with
  no per-step host sync.
- Measured [M]: only the totals — 7.36 ms (release instrument) / 7.72-7.75 ms (W0 step cell), 2.63-2.93 tok/step at
  T=0 on technical prose -> 2.64-2.95 ms/token. No per-role split exists: W0 Phase 2/2b nsys captures deadlocked
  (`w0/W0-RESULTS.md`).
- Modeled split [Mod]: draft = 15-20 % of a plain 1-row step on two independent engines (dgpp: draft 5.1 ms vs plain
  31.1 ms = 16 %; SGLang V4-Pro: ~5.7 ms outside a 7.3 ms verify incl. scheduling, upper bound). For Pg61 that puts the
  draft + glue at ~1.0-1.8 ms and the 6-row verify at ~5.6-6.4 ms. **Unverified**; the cheapest way to verify is a
  Pg61 run with `num_speculative_tokens` 1 vs 5 (step-time slope = per-row cost) — not on our critical path.
- Calibration point [C]: DeepInfer discussion #224 (https://github.com/deepseek-ai/DeepInfer/discussions/224, 2026-10-24):
  V4.1-Flash-1017 on 8x B200 with vLLM/SGLang **TP4**, MTP/DSpark 5, golden AL 3.51: 665 tok/s/user at c = 1 -> 5.28
  ms/step [Mod from C]; "TP4 ... a whole step takes ~5.6 ms". Our bar (5.0 ms on one GB300) is below what four B200s
  achieve with tensor parallelism today.

## M. Pricing the yield on our runtime (all [Mod] unless marked)

Model: ms/token = (T(m) + X) / tau(d), m = d + 1 verify rows, d = drafts verified, T(m) = T1 + s (m - 1), X = draft cost
(paid whenever d >= 1), tau(d) = 1 + sum_{j<=d} a_j with a_j the *unconditional* prefix survival of draft j under exact
greedy match. Computed in Python (this session), not by hand.

Acceptance profiles (a_1..a_5):
- prose, W0-calibrated: cmp170hx shape scaled to Pg61's measured 2.77 -> 0.637 / 0.497 / 0.325 / 0.197 / 0.114, tau5 = 2.77
- mixed (cmp170hx, V4-0731, T=0) -> 0.730 / 0.569 / 0.372 / 0.226 / 0.131, tau5 = 3.03
- code (dgpp V4.1-1017 conditional 86/79/88/86 %, c5 := c4) -> 0.860 / 0.679 / 0.598 / 0.514 / 0.442, tau5 = 4.09

Cost scenarios:
- NOW: T1 = 10.52 [M], T6 = 21 (CONTEXT projection = 2.0 x T1; the 2.0 is the *slow* dg-masked path's R6/R1 [M]) ->
  s = 2.10 ms/row; X = 2.0 ms (range 1.5-3.0; derivation below).
- FAST6: T6 = 15 (in-place MoE reads 22.45 unique experts x 18.8 MB x 40 = 16.9 GB vs 4.5 GB at 1 row; +12.4 GB at the
  measured ~2.8-4 TB/s in-place rate = +3-4.4 ms, plus ~0.5 ms other) -> s = 0.9 ms/row, X = 2.0.
- BAR: whole step 5.0 ms at 6 rows with a bytes-floor row cost s = 0.3 ms (the same 12.4 GB at 8 TB/s / 5 rows) and
  X = 0.8 -> T1 = 2.7.

Draft cost X derivation: 3 window-only draft layers at 5 rows ~ 3 x (0.18-0.36) ms (our 1-row layer ~0.25 ms = 10.52 ms
/ 40 minus head; window-only layers skip compressor/indexer; x1.3-2.0 for 5 rows) = 0.55-1.1 ms; draft MoE <= 15 unique
of 128 experts per layer, all HBM; head on 5 rows over 2.65 GB fp32 ~0.33-0.42 ms; Markov chain 5 x (embed gather +
[129280 x 256] GEMV (66 MB bf16 in the checkpoint, `mtp.2.markov_head.embed.weight` BF16 per
`docs/checkpoint-1017-inventory.json`; 132 MB if widened to fp32) + argmax) ~0.15-0.3 ms; main_proj (78.6 MB fp8) +
3 context-KV writes ~0.1 ms; ~250-300 launches at ~3.7 us. Total 1.5-2.5 ms, central 2.0. Cross-check: draft = 16 % of
a plain step on dgpp, upper bound ~0.5x verify on SGLang V4-Pro -> 0.16-0.2 x 10.52 = 1.7-2.1 ms.

ms/token by verify depth d (d = 0 is the plain 1-row step; d = 5 is production's k = 5 / 6 rows):

| scenario | profile | d0 | d1 | d2 | d3 | d4 | d5 | best fixed d | oracle per-step depth |
|---|---|---|---|---|---|---|---|---|---|
| NOW | prose | 10.52 | 8.93 | 7.83 | **7.65** | 7.87 | 8.30 | 3 (-8 % vs d5) | 5.86 |
| NOW | mixed | 10.52 | 8.45 | 7.27 | **7.04** | 7.22 | 7.60 | 3 (-7 %) | 5.54 |
| NOW | code | 10.52 | 7.86 | 6.58 | 6.00 | 5.72 | **5.62** | 5 | 4.64 |
| FAST6 | prose | 10.52 | 8.19 | 6.71 | 6.18 | **6.06** | 6.14 | 4 (-1 %) | 5.09 |
| FAST6 | mixed | 10.52 | 7.75 | 6.22 | 5.69 | **5.56** | 5.61 | 4 (-1 %) | 4.73 |
| FAST6 | code | 10.52 | 7.21 | 5.64 | 4.85 | 4.41 | **4.15** | 5 | 3.74 |
| BAR | prose | 2.70 | 2.32 | 1.92 | 1.79 | **1.77** | 1.80 | 4 (-2 %) | 1.46 |
| BAR | mixed | 2.70 | 2.20 | 1.78 | 1.65 | **1.62** | 1.65 | 4 (-2 %) | 1.36 |
| BAR | code | 2.70 | 2.04 | 1.61 | 1.40 | 1.29 | **1.22** | 5 | 1.08 |
| Pg61 [M] | W0 prose | | | | | | 2.62-2.94 (7.72-7.75 ms / 2.63-2.93) | | |

"Oracle" = verify exactly the drafts that will be accepted (perfect confidence head) — an upper bound on what
confidence-gated depth can buy; a real head (ROC-AUC 0.81-0.90, trained for sampled TV acceptance, not greedy match)
captures some fraction of the gap. Unverified for our numerics.

Answers to the W2 questions:
1. **Realistic tokens/step at k = 5 (exact greedy):** 2.6-2.9 on prose like the W0 step prompt (prior = Pg61's measured
   2.63-2.93 under the identical acceptance rule), ~3.0 mixed, 3.7-4.3 code/math/JSON, ~3.5 the InferenceX agentic
   "golden" AL (synthetic). Ours could differ from Pg61's because our verify numerics differ at near-ties (A3 class) and
   our draft runs the reference arithmetic; expect within a few %, unmeasured.
2. **ms/token today:** (21 + 2) / 2.77 = **8.3 ms/token** on the W0 prompt (5.6 on code) vs Pg61's 2.8. At the FAST6
   point 6.1 (4.2 code). At the 5.0-ms bar, 1.8 ms/token on prose — 1.55x better than production per token.
3. **More drafts (k = 7, 9): no.** (a) The checkpoint's drafter is a block-5 network whose block rows attend to each
   other non-causally; running more query rows perturbs d1..d5 and is out of distribution. Measured on the V4-0731 block-5
   drafter: N = 7 gave mean acceptance 1.43-2.51 vs 3.03 at N = 5 and 60.3 vs 98.1 tok/s (cmp170hx). (b) Even a
   hypothetical drafter that held 86 % conditional acceptance to position 9 (would need new drafter weights: out of
   policy) moves ms/token by +1 % / +5 % (NOW, d7/d9 vs d5: 5.62 -> 5.66 -> 5.89, worse), -6 % (FAST6: 4.15 -> 3.87),
   -5 % (BAR: 1.22 -> 1.16), code profile only. (c) The rows-are-free premise is false on our runtime today: s = 2.1
   ms/row NOW, 0.9 FAST6, ~0.3 at the bytes floor — the extra rows' MoE experts are real bytes (unique experts per
   layer 6 -> 22.45 at 6 rows). Extra rows could alternatively go to a TreeSpark-style tree (siblings from the Markov
   head; claimed +15-25 % accepted drafts, +8-14 % single-request speed on Qwen3-4B where verify rows are ~free) — at
   s >= 0.3 ms/row and our 256-slot position-indexed window ring (two candidates per position collide) this is not worth
   building now.
4. **Confidence head:** (a) it cannot skip the *current* draft — it is computed from the draft backbone's hidden state and
   the Markov embedding, i.e. after the draft forward (`DSparkConfidenceHead`, model.py:1092-1100). The DSpark authors list
   draft early-exit as future work. Skipping drafting via a lagged signal only pays when a_1 < (s + X) / T1 =
   0.39 (NOW), 0.28 (FAST6), 0.41 (BAR); every measured profile has a_1 >= 0.64 on average, so only low-acceptance
   stretches (random text, some agentic/tool output) would qualify. If drafting is skipped, the draft rings must still
   receive the committed positions' context KV (3 x wkv/kv_norm/rope/quant per token) or later drafts degrade.
   (b) Its real use for us is **truncating verification depth**: best fixed depth is 3 (NOW) or 4 (FAST6/BAR) on prose
   for -1 to -8 % ms/token; a confidence-gated per-step depth sits between that and the oracle (NOW prose 7.65 -> <= 5.86).
   This only matters while s is large; at the bar it is worth ~2 % (fixed) to <= 18 % (oracle) on prose.
   (c) Under exact greedy, any depth rule — confidence-gated, lagged, or random — is lossless and leaves the emitted
   stream bit-identical (only the step count changes). The DSpark Appendix-A non-anticipation constraint applies to
   sampled acceptance only.
5. **Numerics of the instrument — confirmed from vLLM code:** greedy acceptance = `target_argmax == draft` with the target
   argmax emitted on the first mismatch (`rejection_sampler_utils.py:571-595`), so the emitted stream is the target's
   greedy stream *evaluated with the verify step's arithmetic*. vLLM itself is not bit-identical to its own 1-row greedy
   (batch-variant kernels; cmp170hx and our W0 rep spread are the evidence). Ours equals our 1-row greedy iff every verify
   row is row-invariant: per the M1-T10 census that requires (i) the M = 1 GEMV configs pinned at M = 6, (ii) a
   row-invariant head (cuBLAS fp32 `F.linear` is not — W3/W4 territory), (iii) first-index argmax ties on both paths.
   Draft-side numerics are free: anything done to the drafter (bf16 copy of the head for draft logits, top-k Markov,
   skipping the confidence head, lower-precision draft GEMVs) can change acceptance but never the emitted tokens.

## N. Numbers table

| quantity | value | class | source |
|---|---|---|---|
| Pg61 step, W0 step cell | 7.746 / 7.747 / 7.724 ms | measured | `w0/results/phase1/step-pg61.json` |
| Pg61 tokens/step, T=0 technical prose | 2.753 / 2.626 / 2.926 (175-195 steps for 512 tok) | measured | same |
| Pg61 ms/token | 2.64-2.95 | measured (derived) | same |
| Pg61 speculative config | dspark, N=5 at batch 1-4, probabilistic/block, adaptive verification off | measured (args) | `w0/results/ds41-args.json` |
| verify rows / draft rows per step at batch 1 | 6 / 5 (+ n+1 context rows) | claimed (code) | vLLM v0.31.0 `rejection_sampler_utils.py:544-546`, `dspark/speculator.py:52-53` |
| greedy acceptance rule | exact `target_argmax == draft`; block/probabilistic inert at T=0 | claimed (code) | `rejection_sampler_utils.py:549-595` |
| our 1-row step | 10.52 ms | measured | `first-light-t9b4-20261107.json` |
| R6/R1 (slow MoE path) | 2.008 | measured | `M1-T10-RESULT-20261107.md` |
| our 6-row verify, fast path | ~21 (CONTEXT) / ~15 (in-place MoE bytes model) ms | modeled | CONTEXT.md; §M |
| per-verify-row cost s | 2.1 (NOW) / 0.9 (FAST6) / ~0.3 (bytes floor) ms | modeled | §M |
| draft cost X | 1.5-2.5 ms, central 2.0 | modeled | §M |
| tokens/step at k=5, ours | 2.6-2.9 prose, ~3.0 mixed, 3.7-4.3 code/math/JSON | modeled (prior from measured/claimed) | F4 |
| ms/token, ours NOW (21+2) | 8.3 prose / 7.6 mixed / 5.6 code | modeled | §M |
| ms/token at the 5.0-ms bar | 1.80 prose / 1.65 mixed / 1.22 code | modeled | §M |
| best fixed verify depth NOW / FAST6 / BAR (prose) | 3 / 4 / 4, saving 8 / 1 / 2 % | modeled | §M |
| oracle depth bound NOW prose | 5.86 ms/token (-29 % vs d5) | modeled | §M |
| V4.1-1017 tokens/pass by depth 1-4, T=0 | prose 1.66-2.24, code 1.83-3.75, JSON 1.95-4.25, math 1.92-3.98, chat 1.70-2.58 | claimed | dgpp model card (2026-10-18) |
| V4.1-1017 conditional acceptance per position, T=0 | code 86/79/88/86 %, JSON 94/89/82/78 %, prose 63/46/24/67 % | claimed | dgpp model card (2026-10-19) |
| V4-0731 unconditional per-position acceptance, T=0 | 0.730/0.569/0.372/0.226/0.131, tau 3.03 | claimed | cmp170hx RESULTS.md |
| N = 7 on a block-5 drafter | tau 1.43-2.51 vs 3.03; 60.3 vs 98.1 tok/s | claimed | cmp170hx RESULTS.md |
| V4-Flash DSpark, 8x B200 TP8, c=1 | AL 3.58-3.72, TPOT 2.52 vs 5.98 ms no-spec | claimed | SGLang PR #29538 |
| V4.1-Flash-1017, 8x B200, vLLM/SGLang TP4, c=1 | 665 tok/s/user at golden AL 3.51 (= 5.28 ms/step); "~5.6 ms" step | claimed | DeepInfer #224 (2026-10-24) |
| DSpark V4 production vs MTP-1 | +60-85 % per-user speed at matched throughput (V4-Flash) | claimed | arXiv 2607.05147 |
| confidence head quality (Qwen3-4B, offline) | ROC-AUC 0.81-0.90, ECE 3-8 % raw, ~1 % after STS | claimed | arXiv 2607.05147 §4.3.3 |

## W. What this means for our runtime

1. **Keep N = 5 drafts, block-5 drafter, exact-greedy acceptance.** Nothing in the vLLM loop contradicts M1 design
   §1.3-§1.5; the context-KV commit for rows 0..n matches vLLM's `PAD_SLOT_ID` rule for rejected rows. Production
   drafts argmax at T = 0 too, so M1 fix 0.5 (`mtp[i].temperature = 0`) reproduces production's draft rule.
   Numerics: emitted stream bit-identical to our 1-row greedy given row-invariant verify rows.
2. **Do not build k > 5 or draft trees.** Measured worse for k = 7 on this drafter family; trees need per-position
   duplicate slots that break the 256-slot `pos % 256` ring (`runtime/static_step.py` window index) and buy <= 5 % even
   in the best hypothetical.
3. **Verify-depth truncation is the one yield lever with a positive sign while rows cost > 0.3 ms.** Cheapest form for
   us: one captured 6-row graph whose MoE routing ignores rows >= m (the DG masked GEMM already takes device-side
   `masked_m`, `runtime/dg_moe.py` / `runtime/moe_rows.py`; the dedup only counts active rows' (row, expert) pairs), so the
   unique-expert bytes — the dominant per-row cost — fall with m without recapture. GEMV, attention and glue still run
   6 rows (they are latency-bound and roughly row-flat). m chosen on device from the previous step's accepted length
   or the confidence head (2-steps-back relay like DeepSeek's). Expected: NOW prose 8.30 -> 7.65 ms/token at fixed
   d = 3 (step 23.0 -> 18.8 ms incl. draft) *if all of s is skippable*; masking only the MoE recovers only the MoE share
   of s (P1 measures it). FAST6 ~1 %, BAR ~2 %; oracle bound 29 % / 17 % / 19 %. Numerics: bit-identical
   (rows are independent; a skipped row only changes how many tokens the step emits). Headline remains the fixed 6-row step
   for comparability with Pg61; report ms/token beside it.
4. **Make the drafter cheap; its numerics are free.** Target X <= 2.0 ms now and <= 0.8 ms at the bar (16 % of a step).
   Levers that cannot change emitted tokens: a bf16 copy of the head for the 5 draft rows (the checkpoint stores the
   head in BF16, so these are the stock values without the reference's fp32 widening; 1.32 GB extra HBM; 2.65 ->
   1.32 GB read, ~-0.17 ms at HBM rate), top-k Markov bias (vLLM `dspark_draft_topk`; 5 x 66 MB -> gathered rows),
   skip the confidence head unless depth gating uses it, the draft MoE always in HBM. Each changes acceptance at most at
   draft near-ties — measure tau before/after.
5. **The bar is not reachable from the yield side.** Yield turns step ms into token ms; it cannot change the 6-row step.
   At 23 ms/step we would need tau ~ 8 to match Pg61 per token; the drafter tops out near 4-4.6 on code. The 6-row
   step must fall ~4.6x (fusion/launch-boundary work, W1 + M2 memo), with s driven toward the ~0.3 ms/row bytes floor.
6. **Instrumentation to add to M1-T12:** per-step accepted length (already decision 14), plus per-position
   conditional acceptance and the confidence-head outputs, so depth policies can be evaluated offline on recorded
   greedy runs (exact greedy makes counterfactual depth replay trivial: tokens never change).

## P. What must be measured on the box (each <= 30 min)

- **P1 T(m) curve.** Teacher-forced fast-path verify graph at m = 1, 2, 3, 4, 5, 6 (and 8) rows on the W0 positions
  (`bench/first_light.py --rows m --teacher-forced --mode graph`, in-place MoE once ported); fit s; add one nsys capture
  of m = 1 vs 6 to split the per-row cost into MoE vs everything else. Decides whether depth truncation is 8 % or 1 %.
- **P2 Draft cost X.** Once the M1-T11 draft graph exists: replay it alone at ctx_count 1..6, report ms; same with the
  bf16 draft-head copy and top-k Markov (k = 64) to price items W.4.
- **P3 Acceptance census on our numerics, no loop needed.** Reference/eager path on the 8 W0 fixture prompts + the W0
  step prompt: for each committed greedy position p, run `forward_spec(t_p, main_hidden_p, p)` and record d1..d5 vs the
  next five greedy tokens -> per-position conditional acceptance a_j and tau(d) per prompt class. Gives our own
  tokens/step before M1-T12 exists, and the class split (prose vs code) for the depth policy.
- **P4 Confidence head for greedy.** Same run: log `sigmoid(confidence)`; report ROC-AUC/ECE against *greedy-match*
  survival (the head was trained on TV-distance acceptance), the best fixed depth, and the gain of a threshold rule vs
  the oracle, with s from P1.
- **P5 (optional) N = 7 sanity.** In P3, also run the drafter with 7 query rows (anchor + 6 noise) and confirm d1..d5
  acceptance drops (out-of-distribution) — closes the k > 5 question on our checkpoint in 10 min.
- Not proposed: a Pg61 `num_speculative_tokens` 1-vs-5 run to split production's verify/draft — the lane is dark by
  charter and restoring it needs James's word; not on the critical path.

## S. Sources (all fetched 2026-11-07)

- DSpark paper, Cheng et al., arXiv 2607.05147v1, 2026-07-06 — https://arxiv.org/html/2607.05147v1
- DeepSeek-V4.1-Flash tech report, arXiv 2609.19969v1, 2026-09-17, §2.4.3 — https://arxiv.org/html/2609.19969
- vLLM v0.31.0 source (raw): `vllm/v1/worker/gpu/spec_decode/{__init__,dspark/speculator,dspark/utils,dflash/speculator,
  rejection_sampler,rejection_sampler_utils,adaptive_verification}.py`, `vllm/v1/worker/gpu/sample/gumbel.py`,
  `vllm/config/speculative.py`, `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-*.yaml`,
  `tests/v1/e2e/spec_decode/acceptance_rates/dspark/test_dspark.py` — https://github.com/vllm-project/vllm/tree/v0.31.0
  (local copies in `~/.hermes/profiles/milo/cache/scratch/w2src/`); vLLM main `dspark/speculator.py` (adds TP-sharded
  Markov sampling only).
- vLLM docs, DSpark speculator API — https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/spec_decode/dspark/speculator/
- vLLM recipes, DeepSeek-V4.1-Flash — https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash.md
- HF configs `deepseek-ai/DeepSeek-V4-Flash-DSpark`, `deepseek-ai/DeepSeek-V4-Pro-DSpark` (`dspark_block_size` 5) —
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/resolve/main/config.json
- SGLang PR #29538 (2026-07) — https://github.com/sgl-project/sglang/pull/29538
- LMSYS blog "DSpark in SGLang", 2026-07-06 — https://www.lmsys.org/blog/2026-07-06-dspark-sglang
- SGLang DeepSeek-V4.1 cookbook §3.3 — https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
- dgpp DeepSeek-V4.1-Flash-1017 model card (sections dated 2026-10-18..20) —
  https://github.com/HawkBearPig/dgpp/blob/master/docs/model_cards/DeepSeek-V4.1-Flash-1017.md
- allover326/deepseek-v4-cmp170hx RESULTS.md — https://github.com/allover326/deepseek-v4-cmp170hx/blob/main/RESULTS.md
- SemiAnalysis InferenceX PR #4783 (2026-11-04) — https://github.com/SemiAnalysisAI/InferenceX/pull/4783
- DeepInfer discussion #224 (2026-10-24) — https://github.com/deepseek-ai/DeepInfer/discussions/224
- TreeSpark, arXiv 2609.22098v1 — https://arxiv.org/html/2609.22098v1
- Repo receipts: `w0/results/ds41-args.json`, `w0/results/phase1/step-pg61.json`, `w0/W0-RESULTS.md`,
  `docs/M1-DSPARK-DESIGN.md`, `docs/M1-T10-RESULT-20261107.md`, `docs/checkpoint-1017-inventory.json`,
  `reference/deepseek-inference/{model.py,config.json}`, `reference/vllm-0.31.0/deepseek_v41/nvidia/dspark.py`.

W2: Yield cannot carry the bar — production verifies 6 rows by exact greedy match with the confidence head off and gets 2.6-2.9 tok/step on prose; the block-5 drafter can't usefully go past k=5, rows cost us ~2 ms each today, so the only yield lever is device-side verify-depth truncation (1-8 % ms/token, bit-identical) plus a <=2 ms drafter, while the 6-row step itself must fall ~4.6x.
