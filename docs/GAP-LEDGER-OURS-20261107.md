# Gap ledger -- ours vs Pg61 (2026-11-07 19:40:30 CST)

Tags: [M] measured, [D] derived (arithmetic below), [S] nsys static [static, diagnostic] -- eager kernel durations, not an additive accounting of the captured graph; `not measured` says why in the JSON.

| metric | ours: headline loop (own stream) | ours: 6-row teacher-forced on Pg61 tokens | Pg61: W0 step cell (STEP_PROMPT) | Pg61: parity-A (fixture, matched tokens) | Pg61: peer-tier-v3 sidecar counters |
|---|---|---|---|---|---|
| verify GPU ms/step | 17.400 [M] | not measured | not measured | not measured | not measured |
| draft GPU ms/step | 2.904 [M] | not measured | not measured | not measured | not measured |
| complete step ms (GPU events; Pg61: wall/steps) | 20.304 [M] | not measured | 7.740 [D] | not measured | not measured |
| tokens per step | 2.903 [M] | not measured | 2.763 [D] | not measured | not measured |
| GPU ms per emitted token (sum step ms / sum tokens) | 6.994 [M] | not measured | not measured | not measured | not measured |
| wall ms per emitted token, decode (total / total) | 8.103 [D] | not measured | 2.792 [D] | not measured | not measured |
| wall ms per emitted token, incl. prefill (total / total) | 29.048 [D] | not measured | 2.848 [D] | 2.909 [D] | not measured |
| routed MoE: expert GEMM (DeepGEMM fp8xfp4 / MegaMoE / ref fp4_gemm) | not measured | 4.449 [S] | not measured | not measured | not measured |
| MoE SwiGLU+quant (routed and shared share one kernel name) | not measured | 0.250 [S] | not measured | not measured | not measured |
| routed MoE: combine | not measured | 0.155 [S] | not measured | not measured | not measured |
| routed MoE: staging/prep (dedup, cold stage over C2C, SF pack, act quant bcast, gathers) | not measured | 4.645 [S] | not measured | not measured | not measured |
| shared expert (fp8 GEMV inside the moe range) | not measured | 0.881 [S] | not measured | not measured | not measured |
| router (gate fp32, plan, pre/post, top-k inside the moe range) | not measured | 0.811 [S] | not measured | not measured | not measured |
| dense projections (fp8 GEMV outside moe: q/kv/o + engram/compressor/indexer linears) | not measured | 2.009 [S] | not measured | not measured | not measured |
| head (fixed-order fp32 / bf16-stored head, cuBLAS head) | not measured | 1.018 [S] | not measured | not measured | not measured |
| fp8 GEMV, call site not resolvable (no NVTX role) | not measured | 0.000 [S] | not measured | not measured | not measured |
| attention (FlashMLA / sparse_attn) | not measured | 0.405 [S] | not measured | not measured | not measured |
| glue: norms + mHC (hc_norm, norm_row, sinkhorn, hc_post, mixes) | not measured | 2.501 [S] | not measured | not measured | not measured |
| glue: engram gate/out | not measured | 0.016 [S] | not measured | not measured | not measured |
| glue: indexer top-k/sort (outside the moe range) | not measured | 0.217 [S] | not measured | not measured | not measured |
| torch top-k/sort, router vs indexer not resolvable (no NVTX role) | not measured | 0.000 [S] | not measured | not measured | not measured |
| glue: ratio-2 compressor (T9b-8 fixed fp32) | not measured | 0.284 [S] | not measured | not measured | not measured |
| glue: act quant / rope / QDQ | not measured | 0.239 [S] | not measured | not measured | not measured |
| glue: torch residual (elementwise, reduce, cat, index, cub, asserts) | not measured | 1.375 [S] | not measured | not measured | not measured |
| residency manager kernels | not measured | 0.000 [S] | not measured | not measured | not measured |
| instrument (step marker) | not measured | 0.000 [S] | not measured | not measured | not measured |
| kernel names no rule matched | not measured | 0.000 [S] | not measured | not measured | not measured |
| sum of kernel ms/step | not measured | 19.254 [S] | not measured | not measured | not measured |
| unattributed = step - sum kernels (gaps/scheduling) | not measured | not measured | not measured | not measured | not measured |
| GPU busy (union) ms/step | not measured | 19.254 [S] | not measured | not measured | not measured |
| measured overlap = busy / wall | not measured | 0.096 [S] | not measured | not measured | not measured |
| sum kernel / busy | not measured | 1.000 [S] | not measured | not measured | not measured |
| routed expert bytes from HBM per verify step | not measured | not measured | not measured | not measured | not measured |
| cold expert bytes over C2C per verify step | not measured | not measured | not measured | not measured | not measured |
| cold share of routed unique-expert calls | not measured | not measured | not measured | not measured | not measured |
| cold experts per 6-row layer-call | not measured | not measured | not measured | not measured | 1.600 [M] |
| cold-expert work exposed on GB300 per step | not measured | not measured | not measured | not measured | 0.020 [D] |
| captured graph nodes | not measured | not measured | not measured | not measured | not measured |
| memcpy |  | {"DtoD": {"bytes_per_step": 3820316.25, "ms_per_step": 0.043 [S] |  |  |  |

## Conditions
- ours.headline condition: fixture 8 prompts, own greedy stream (DSpark k=5 loop), max_new 256, impls {'moe': 'dg-inplace', 'fp8': 'gemv', 'glue': 'fused', 'attn': 'flashmla', 'fp32': 'fixed', 'moe_rowmatch': 'on'}, map hot-v26ar-seed-295.json, spare 16, residency off
- ours.headline residency: seed map docs/maps/hot-v26ar-seed-295.json: 295 hot experts/layer in HBM, 89 cold in pinned Grace, cold read over C2C (UVA) into 16 spare slots/layer by the staging kernel; residency manager off
- ours.matched condition: None
- ours.matched residency: None
- pg61.step_cell condition: STEP_PROMPT (66 prompt tokens), 512 greedy tokens x 3 reps, thinking off; not the fixture
- pg61.step_cell residency: NOT all-HBM: Garner rowmap-mix-v1-h255 (255 hot/layer in GB300 HBM, 129 cold) with the cold experts COMPUTED on the RTX PRO 6000 by the peer-tier-v3 sidecar, not read over C2C (A2-RESIDENCY-DESIGN.md 0.2, w0/W0-RESULTS.md phase 2); confirm the map from the ds41 policy at the re-run. The two conditions cannot be equalised without changing one runtime; the ledger reports each side's cold work instead (bytes / exposure rows)
- pg61.fixture condition: fixture 8 prompts (prose,code,math,json,shell,tool,long,story), prompt tokens [42, 47, 54, 38, 34, 49, 46, 31], 1476 emitted, greedy T=0, top-20 logprobs, non-streamed
- pg61.fixture residency: NOT all-HBM: Garner rowmap-mix-v1-h255 (255 hot/layer in GB300 HBM, 129 cold) with the cold experts COMPUTED on the RTX PRO 6000 by the peer-tier-v3 sidecar, not read over C2C (A2-RESIDENCY-DESIGN.md 0.2, w0/W0-RESULTS.md phase 2); confirm the map from the ds41 policy at the re-run. The two conditions cannot be equalised without changing one runtime; the ledger reports each side's cold work instead (bytes / exposure rows)
- pg61.sidecar condition: Pg61 sidecar stats, all requests in its window, by_T['6'] = the 6-row layer-calls; not the fixture
- pg61.sidecar residency: NOT all-HBM: Garner rowmap-mix-v1-h255 (255 hot/layer in GB300 HBM, 129 cold) with the cold experts COMPUTED on the RTX PRO 6000 by the peer-tier-v3 sidecar, not read over C2C (A2-RESIDENCY-DESIGN.md 0.2, w0/W0-RESULTS.md phase 2); confirm the map from the ds41 policy at the re-run. The two conditions cannot be equalised without changing one runtime; the ledger reports each side's cold work instead (bytes / exposure rows)

## Gaps (ours - Pg61)
- step_gpu_ms: ours.headline 20.304 - pg61.step_cell 7.740 = +12.564 (NOT matched: different prompt, 512 tokens)
- tokens_per_step: ours.headline 2.903 - pg61.step_cell 2.763 = +0.141 (NOT matched: different prompt, 512 tokens)
- wall_ms_per_token_decode: ours.headline 8.103 - pg61.step_cell 2.792 = +5.311 (NOT matched: different prompt, 512 tokens)
- wall_ms_per_token_e2e: ours.headline 29.048 - pg61.fixture 2.909 = +26.140 (prompts matched; ours streamed fast reps incl. reference prefill vs Pg61 non-streamed with top-20 logprobs)

## Derivations
- ours.headline verify_gpu_ms: docs/headline-20261107j.json headline.two_graphs.ms_verify_graph.mean: CUDA events per step (live steps >= 32, 558 steps pooled over prompts x reps)
- ours.headline draft_gpu_ms: docs/headline-20261107j.json headline.two_graphs.ms_draft_graph.mean: CUDA events per step (live steps >= 32, 558 steps pooled over prompts x reps)
- ours.headline step_gpu_ms: docs/headline-20261107j.json headline.two_graphs.ms_per_step.mean: CUDA events per step (live steps >= 32, 558 steps pooled over prompts x reps)
- ours.headline tokens_per_step: docs/headline-20261107j.json two_graphs.tokens_per_step (sum tokens / sum steps, live steps >= 32, 558 steps pooled over prompts x reps)
- ours.headline gpu_ms_per_token: docs/headline-20261107j.json two_graphs.ms_per_token = sum step ms / sum tokens (live steps >= 32, 558 steps pooled over prompts x reps); all steps: 6.660
- ours.headline wall_ms_per_token_e2e: sum wall_ms_fast / sum (tokens x fast reps) = 85575.634 / 2946.000 = 29.048 ms (host wall incl. the reference prefill, 3857 ms/prompt)
- ours.headline wall_ms_per_token_decode: sum (wall_ms_fast - prefill_ms) / sum (tokens x fast reps) = 23871.320 / 2946.000 = 8.103 ms (host wall of the decode loop, all steps incl. the async poll)
- pg61.step_cell step_gpu_ms: wall-clock, not GPU: sum decode_wall_s / sum (steps - 1) = 1000 x 4.280 / 553.000 = 7.740 ms over reps [0, 1, 2] (SSE arrivals / metrics drafts; per rep [7.746, 7.747, 7.724])
- pg61.step_cell tokens_per_step: sum gen_tokens / sum steps = 1536.000 / 556.000 = 2.763
- pg61.step_cell wall_ms_per_token_decode: sum decode_wall_s / sum (gen_tokens - 1) = 1000 x 4.280 / 1533.000 = 2.792 ms (the first chunk = the prefill's token)
- pg61.step_cell wall_ms_per_token_e2e: sum wall_s / sum gen_tokens = 1000 x 4.374 / 1536.000 = 2.848 ms (incl. TTFT + HTTP)
- pg61.fixture wall_ms_per_token_e2e: sum wall_s / sum n_tokens = 1000 x 4.293 / 1476.000 = 2.909 ms (non-streamed: incl. prefill, HTTP and top-20 logprob work; one request each)
- pg61.sidecar cold_experts_per_layer_call: docs/data/peer3_stats.json by_T['6'].cold_exp_mean (distinct cold experts per 6-row layer-call, 14760 calls)
- pg61.sidecar cold_exposed_ms: GB300's exposed wait for the RTX's cold-expert work per 6-row step = 0.5 us x 40 layers / 1000 = 0.020 ms (by_T['6'].gb_wait_us_mean; RTX busy 37.0 us per cold call runs concurrently)

Pg61 cells resolvable from existing receipts: step_gpu_ms, tokens_per_step, wall_ms_per_token_decode, wall_ms_per_token_e2e, cold_experts_per_layer_call, cold_exposed_ms

## Pending columns
- ours.loop_static: box command 3 (optional): nsys + gap_ledger.py ours-loop-profile, then step_anatomy.py (draft families)
- pg61.matched: box command 4: gap_ledger.py pg61-drive on the box host -- needs Pg61 bound on :30006, i.e. James's word to boot it (lane is DARK by charter)
