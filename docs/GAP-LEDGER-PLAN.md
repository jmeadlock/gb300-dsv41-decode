# GAP-LEDGER plan — a matched-workload decomposition of our 6-row step vs Pg61 (2026-11-07 19:05 CST)

Worker: GAP-LEDGER (desk + code, main tree). Handoff `docs/HANDOFF-GAP-LEDGER.md`; reviewer ask (Astra, accepted): before any persistent-kernel bet, a ledger with matched prefixes/positions/shapes/residency that records verify, draft and step GPU time, wall ms per emitted token, per-family time, graph node/edge counts and measured overlap — with nsys static timings treated as diagnostic, not as an additive accounting of the graph.

Code: `bench/gap_ledger.py` (new: `build`, `ours-tf`, `ours-loop-profile`, `pg61-drive`), `bench/step_anatomy.py` (extended: kernel→family map over every kernel, phases, overlap, memcpy), `tests/test_gap_ledger_cpu.py` (20 tests). First build from existing receipts: `scratch/gap-ledger-ours.json` + `scratch/gap-ledger-ours.md`.

## 0. Findings that change the handoff's premises

1. **`w0/results/phase1/step-pg61.json` is not an nsys profile.** It is `w0_drive.py step`: 3 × 512-token greedy generations on `STEP_PROMPT` (not the fixture), steps from `/metrics` spec counters, wall from SSE arrival times (`"profile": false`). **No per-kernel profile of Pg61 exists.** Both nsys attempts (W0 phase 2 with `--cuda-graph-trace=node`, phase 2b with `=graph`) stalled the engine after 1–2 requests: CUPTI tracing and the peer-tier-v3 spin cannot coexist (`w0/W0-RESULTS.md`). So every Pg61 family, sum-of-kernels, unattributed, overlap and graph-census cell is **not measured**, and a matched re-run with the same harness cannot fill them either. The ledger says so cell by cell.
2. **Pg61's residency is not all-HBM.** Garner's `rowmap-mix-v1-h255`: 255 hot experts/layer in GB300 HBM, 129 cold, and the cold experts are *computed on the RTX PRO 6000* by the peer-tier-v3 sidecar, not read over C2C (`docs/A2-RESIDENCY-DESIGN.md` §0.2, §1.3). Ours: seed map `hot-v26ar-seed-295` (295 hot / 89 cold), cold experts read from pinned Grace over C2C by the staging kernel into 16 spare slots/layer, residency manager off. **The two residency conditions cannot be equalised without changing one runtime.** The ledger reports each side's cold work instead: ours = cold unique experts × 18,800,640 B per verify step (C2C bytes, from our counters); Pg61 = 1.60 distinct cold experts per 6-row layer-call and 0.5 µs exposed GB300 wait per layer-call → 0.020 ms/step (`docs/data/peer3_stats.json` `by_T["6"]`, measured by the sidecar).
3. **Pg61's own step boundaries cannot be replayed in our runtime as it stands.** Teacher-forcing keeps prefixes identical; a DSpark loop with acceptance keeps step boundaries. Both at once means forcing our accept count to Pg61's at positions where our argmax differs from Pg61's token (our stream matches Pg61's to 256 tokens on only 2/8 prompts, diverging at near-ties — A3 check, LEDGER 2026-11-07 15:54), which `DSparkLoop` has no hook for (`set_force` forces drafts, not row 0 or the accept). So the matched cell is **teacher-forced at stride 6**: every row a Pg61 token, every row accepted (`Generator.step_rows`, the M1-T10 cell, bit-equal to the multirow oracle). The 6-row shape is identical; what differs from Pg61's verify is the content of the rows Pg61 rejected (its rejected drafts are not recorded anywhere). The size of that bias on our side is measurable: our TF verify ms vs our headline verify ms (same runtime, same impls).
4. 7.36 ms/step is the 2026-10-29 production number (`PROJECT.md`), from an instrument not in this repo. On the W0 instrument Pg61 reads **7.740 ms/step** (pooled, below) — the ledger uses only receipts and labels the prompt mismatch.

## 1. The matched protocol

| axis | ours | Pg61 | matched? |
|---|---|---|---|
| prompts | the 8 fixture prompts (`w0/results/phase1/parity-A.jsonl`), checkpoint `encoding/` (chat mode, thinking off) — reproduces Pg61's `prompt_tokens` 8/8 (`docs/ENVIRONMENT.md` T0); `ours-tf` re-checks per prompt | same prompts, `w0_drive.chat_body` (T=0, thinking off) | yes |
| tokens / prefixes | **teacher-forced to Pg61's own emitted tokens** (parity-A `logprobs[].bytes` → ids via the checkpoint's byte-level `tokenizer.json`; decode == Pg61 text checked; a prompt with an unmapped token is skipped and reported) | its greedy stream; parity-A is its record; `pg61-drive` re-checks byte equality per prompt (A vs A2 was 7/8 identical, `code` diverged at 121) | yes, to the token, for every prompt whose re-run equals parity-A |
| positions | prompt length L, then L + 6s for step s (stride 6) | L + Σ emitted so far (its own acceptance) | no — see §0.3; `pg61-drive` records Pg61's per-step emissions so the difference is stated, not guessed |
| verify shape | 6 rows (1 + 5 drafts), DSpark k=5 | 6 rows (W2: production verifies 6 rows, exact greedy match) | yes |
| draft shape | 5 rows × 3 DSpark stages (the headline's draft graph; not in the TF cell) | its drafter, same checkpoint mtp weights | shape yes, prefixes no (draft numbers come from the free-running headline loop) |
| numerics/impls | headline j's: dg-inplace (contiguous16) + gemv (M=1 pin) + fused glue + FlashMLA + fixed fp32 + rowmatch on, fills off | Pg61 as deployed | n/a (that is the gap being measured) |
| residency | seed map 295/89, cold over C2C, spare 16, residency off | h255, cold on the RTX sidecar | **no** — §0.2; cold-work rows instead |

## 2. Which instrument gives which column

| column | ours | Pg61 |
|---|---|---|
| verify / draft / step GPU ms | CUDA events around graph replay: `bench/headline.py` two-graph cell (verify A, draft B; own stream) and `gap_ledger.py ours-tf --mode graph` (6-row verify-shaped step on Pg61 tokens) | **not measured** (no GPU timer; `/metrics` has no verify/draft split). Step **wall** ms = Σ decode wall ÷ Σ (steps − 1) from SSE arrivals + spec counters (W0 cell; `pg61-drive` on the fixture) — labelled wall-clock, not GPU |
| tokens/step | headline: Σ tokens ÷ Σ steps (window ≥ 32) | Σ gen ÷ Σ drafts (`/metrics`); per-step histogram from the logprobs pass |
| ms per emitted token | **total ÷ total**, never a mean of ratios: GPU Σ step ms ÷ Σ tokens; wall decode Σ(wall − prefill) ÷ Σ tokens; wall e2e Σ wall ÷ Σ tokens | wall decode Σ decode wall ÷ Σ(gen − first chunk); wall e2e Σ wall ÷ Σ gen (parity-A: non-streamed, incl. prefill + top-20 logprobs) |
| per-family ms | nsys of the **static (eager) path** of the same 6-row TF cell, `step_anatomy.py` → `family_of` over every kernel; tagged **[static, diagnostic]**. Draft families: `ours-loop-profile` (NVTX verify/draft phases), own stream | **not measured** (CUPTI deadlock, §0.1) |
| unattributed | graph event ms − Σ static kernel ms, derived with the arithmetic in the cell; a negative value means the graph ran its kernels faster than eager, not negative gaps | not measured |
| busy, overlap, concurrency | static profile: busy = union of kernel intervals; overlap = busy ÷ step wall; concurrency = Σ ÷ busy. The eager wall is host-launch-bound (~226 ms vs ~21 busy in the 6-row anatomy), so this overlap describes the static path, not the graph. The graph's own overlap is not measurable with nsys (graph-replay tracing stalls; `--cuda-graph-trace=node` was abandoned at T8) | not measured |
| node / edge counts, edge types | `graph_census` on the captured graphs (`--census-loop`: TF 6-row graph, loop A/B/AB): `cudaGraphGetNodes`, node types, `cudaGraphGetEdges_v2` edge kinds (default / programmatic), kernel node names via the driver → family counts and family-pair edge counts | not measured (vLLM's graphs live inside the engine) |
| expert bytes HBM / C2C | derived from the StaticStep's unique-expert counters around each TF cell: hot × 18.8 MB (read in place) and cold × 18.8 MB (over C2C, stream-ordered before the expert GEMM — no overlap with it by construction); CUPTI memcpy rows add any copy-engine HtoD | Pg61 does not read cold experts over C2C; cold work = sidecar counters (1.60 cold/layer-call, 0.020 ms exposed/step) |

Does Pg61 need a re-run? **Yes, but only for the step/yield/wall columns on matched prompts** (the existing step cell is a different prompt; parity-A has no step counts or arrival times). It cannot fill per-family, overlap or census columns. It needs Pg61 bound on :30006, i.e. **James's word** (the lane is DARK by charter; "no restore of Pg61 unless James says").

## 3. Ledger columns (definitions)

Rows of `scratch/gap-ledger-ours.md` (JSON: `columns[<source>][<metric>] = {value, kind, how}`, kinds measured / derived / static / not measured):

- `verify_gpu_ms`, `draft_gpu_ms`, `step_gpu_ms`, `tokens_per_step`, `gpu_ms_per_token`, `wall_ms_per_token_decode`, `wall_ms_per_token_e2e`
- families (`step_anatomy.FAMILIES`): routed MoE = `moe.routed.gemm` (DeepGEMM fp8×fp4 / MegaMoE), `moe.swiglu` (routed + shared: one kernel name), `moe.routed.combine`, `moe.routed.staging` (dedup, cold stage over C2C, SF pack, act-quant bcast); `moe.shared` (fp8 GEMV inside the `moe` NVTX range); `moe.router` (fp32 gate / plan / pre-post, top-k inside `moe`); dense = `dense.proj` (fp8 GEMV outside `moe`: q/kv/o and the engram/compressor/indexer linears — one kernel name, call sites not separable by name; per-launch grids are kept in `kernels_all` for a later shape split) and `dense.head`; `attn`; glue = `glue.norm_mhc`, `glue.engram`, `glue.indexer`, `glue.compressor` (T9b-8), `glue.quant_rope`, `glue.torch`; `residency`, `instrument`, `unmapped` (listed by name — extend the rules, never guess); `*.unsplit` when the profile has no `moe` NVTX ranges (and always in a graph census)
- `sum_kernels`, `unattributed`, `busy_ms`, `overlap`, `concurrency`
- `expert_bytes_hbm`, `expert_bytes_c2c`, `cold_share`, `cold_experts_per_layer_call`, `cold_exposed_ms`
- `graph_nodes <graph>`, `graph_edges <graph>`, `graph_kernel_families <graph>`
- gaps: ours − Pg61 for step ms, tokens/step and wall ms/token, each tagged matched / not matched

## 4. Box commands (orchestrator; after this lands on main and the box pulls it)

Common (one serialized script, one container at a time; receipts to `scratch/` only):

```bash
W=$HOME/dsv41/decode; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; S=$W/scratch
docker ps --format '{{.Names}}' | grep -q 'ds41\|mk-' && { echo refusing; exit 1; }
mkdir -p $S/prof
J=$HOME/dsv41/w0/jit-cache
D="docker run --rm --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang \
  -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm"
NSYS="--cap-add SYS_ADMIN -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro --entrypoint /nsys/target-linux-sbsa-armv8/nsys \
  vllm/vllm-openai:v0.31.0 profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true"
```

The `ours-tf` defaults are headline j's flags (dg-inplace, gemv, fused, flashmla, `--fp32-impl fixed`, `--moe-rowmatch on`, `--spare-slots 16`, seed map, GEMV M=1 pin, fills off).

1. **Matched events + census** (container `mk-gl-tf`, ~10–15 min: boot with mtp ~70 s, 8 × 2 reference prefills ~4 s each, 6-row capture, two loop captures):
   `$D --name mk-gl-tf --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/gap_ledger.py ours-tf --model-dir /ref --mode graph --reps 2 --census-loop --json scratch/gap-tf-graph.json > $S/gap-tf-graph.log 2>&1`
   Check: `[tf] targets ok 8/8`, per-prompt `agree` counts, `[census]` lines (nodes, edges, edge kinds), `overflow` 0.
2. **Matched static anatomy** (container `mk-gl-prof`, ~10 min + export): `$D --name mk-gl-prof $NSYS -o /work/scratch/prof/gl-tf python3 bench/gap_ledger.py ours-tf --model-dir /ref --mode static --reps 1 --profile 16:128 --nvtx-layers --json scratch/gap-tf-static.json`, then on the host `/usr/local/bin/nsys export --type sqlite -o $S/prof/gl-tf.sqlite $S/prof/gl-tf.nsys-rep && python3 bench/step_anatomy.py $S/prof/gl-tf.sqlite --out $S/gap-tf-anatomy.json > $S/gap-tf-anatomy.txt`. Check: `roles known`, `unmapped names` (none, or add rules), 128 `decode_step` ranges.
3. **(optional) Loop static profile — draft families** (container `mk-gl-loop`, ~10 min + export): `$D --name mk-gl-loop $NSYS -o /work/scratch/prof/gl-loop python3 bench/gap_ledger.py ours-loop-profile --model-dir /ref --tag prose --profile 16:32 --nvtx-layers --json scratch/gap-loop-static.json`, then export + `step_anatomy.py … --out $S/gap-loop-anatomy.json`. Own stream (not matched); the mtp stages have no `moe` NVTX hooks, so draft GEMVs land in `dense.proj`.
4. **Pg61 matched re-run — only on James's word** (box host, no container of ours; ~6 min boot + ~2 min drive + teardown). Boot: `cd ~/dsv41/garner && WORK=~/dsv41/garner/pg61-work MODEL_DIR=$M DS41_HOME=~/dsv41/garner/home launcher/launch-pg61.sh up` (W0-RESULTS "Box state at END"; LANE-OWNER line). Drive: `cd $W && python3 bench/gap_ledger.py pg61-drive --reps 3 --json scratch/pg61-drive.json > $S/pg61-drive.log 2>&1` (stdlib; uses `w0/w0_drive.py`'s bodies; per prompt a warm, a logprobs pass, 3 timing passes). Then down again as James directs (`launch-pg61.sh down`, CUTOVER.md step 3). Record the ds41 policy (rowmap file) in the receipt notes.
5. **Build** (Mac or box, stdlib, seconds): `.venv/bin/python bench/gap_ledger.py build --tf scratch/gap-tf-graph.json --anatomy scratch/gap-tf-anatomy.json [--loop-anatomy scratch/gap-loop-anatomy.json] [--pg61-drive scratch/pg61-drive.json]`.

Box commands for the orchestrator: **4** (1, 2 required; 3 optional; 4 gated on James), plus the build.

## 5. What the ledger says today (existing receipts only; `scratch/gap-ledger-ours.md`)

| | ours (headline j, own stream) | Pg61 W0 step cell (STEP_PROMPT) | Pg61 parity-A (fixture) | Pg61 sidecar |
|---|---|---|---|---|
| verify / draft GPU ms | 17.400 / 2.904 [M] | not measured | not measured | |
| step ms | 20.304 [M, GPU events] | 7.740 [D, wall: 1000 × 4.280 s / 553] | not measured | |
| tokens/step | 2.903 [M] | 2.763 [D: 1536 / 556] | not measured | |
| GPU ms/token | 6.994 [M] | not measured | not measured | |
| wall ms/token, decode | 8.103 [D: 23,871 ms / 2,946] | 2.792 [D: 4,280 ms / 1,533] | not measured | |
| wall ms/token, incl. prefill | 29.048 [D] (reference prefill ≈ 3.9 s/prompt dominates) | 2.848 [D] | 2.909 [D: 4,293 ms / 1,476] | |
| cold work | cold share recorded, bytes pending command 1 | | | 1.60 cold/layer-call [M]; 0.020 ms/step exposed [D] |
| families, unattributed, overlap, census | pending commands 1–2 | not measured (CUPTI) | not measured | |

Read: the step gap on unmatched prompts is 20.304 − 7.740 = 12.56 ms (ours GPU events vs Pg61 wall — not the same instrument either). Pg61's cold-expert work costs its GB300 0.02 ms per step because the RTX does it; ours is in `moe.routed.staging` and the C2C bytes row once command 1 lands.

## 6. Limits and follow-ups

- **Pg61 decomposition stays unmeasured** until a non-CUPTI instrument is tried (`nsys --gpu-metrics-devices` sampling: counters only, untested against the peer3 spin) or a peer3-off Grace-only arm is profiled (then it is not Pg61). Without one, the ledger can rank *our* families against our own step, and compare totals with Pg61 — not family against family.
- **Matched step boundaries** need a forced-accept hook in `DSparkLoop` (force row 0 and `a_new` from a device schedule = Pg61's per-step emissions from `pg61-drive`). That is a `runtime/` change for a coder, not this worker; with it, the matched cell becomes the real verify+draft loop on Pg61's exact positions.
- `dense.proj` mixes q/kv/o with the engram, compressor and indexer fp8 linears (one GEMV kernel name). `kernels_all[...].grids` records launch grids; a shape→site table (from `runtime/gemv_mxfp8.py`'s grid formula) would split it without new box work.
- The census names come from the driver (`cuFuncGetName`); if the driver refuses, counts and edge kinds still land and `names_error` says why.
