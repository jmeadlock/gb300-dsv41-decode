# GB300 DSV4.1 Decode — Project Charter

Status: **CHARTER, awaiting James's sign-off** (2026-11-06). Nothing on the lane has been touched.
Owner: James Meadlock. Handler: Milo. Settled by grill session 2026-11-06 (three rounds, tree empty).

## Goal

Make DeepSeek-V4.1-Flash-1017 (`1bae0e10`) run as fast as it can on one GB300 Station, measured as **single-user decode step time**.

Renamed from "GB300 DSV4.1 Megakernel" on 2026-11-06 (James): the technique was never the thesis. The project definition is *whatever makes the step shortest*; the first architecture bet is a fused multi-kernel chain with mailboxes + PDL (the sgl-nvidia-mega / Kimi-K3 "specialized decode" shape); one persistent kernel per decoder layer is the stretch, taken only if the measured boundary cost after the chain still justifies it. One kernel per forward is not a goal.

## The number

| | |
|---|---|
| Headline | C1 decode step time, ms/step, 6 rows (DSpark k=5), nsys on the `cudaProfilerApi` range, fixed prompt corpus |
| Baseline | Pg61: **7.36 ms/step** (production, 2026-10-29) |
| Bar | **Beat Pg61: < 7.36 ms per 6-row step, and publish ms/token alongside it** (restated by James 2026-11-07; the original bar was ≤ 5.0 ms/step) |
| Comparability (11/07) | Pg61's 7.36 is a two-GPU number: its 129 cold experts/layer are computed on the RTX PRO 6000 sidecar (GB300 pays ~0.02 ms/step). Recipe A is one GB300; cold staging over C2C costs us 4.55 ms/step. Bar unchanged; both framings published with every number against it. `docs/GAP-LEDGER-OURS-20261107.md` |
| Adoption rule (11/07, James) | *"The goal is better overall inference; every little bit matters if lots of people use our recipe."* A rung is adopted when it is **measured above the 0.02 ms floor, bit-identical (or a declared A3 class), and regresses nothing else** — no size threshold. Small wins merge; 1-row-only wins ship behind row-conditional defaults. The recipe (1-row, 6-row, prefill, long context) is the deliverable, not the headline alone. |
| Wall-clock round (11/08) | Measured first: no per-step host gap (blog §8.4 claim corrected). Overrun fix adopted (97 → 8 steps; host gap 0.02 ms/token). Grouped prefill on the reference kernels adopted, default: TTFT 3.9 s → 0.95 s, bit-equal 8/8. Recipe v1 published to J-M-Recipes (`deepseek-v4.1-flash-standalone-decode-runtime`, experimental). Open decision: A3-class grouped expert kernel for prefill (~0.1–0.2 s TTFT [E]). `docs/SUNDAY-20261108.md`. |
| Why restated | Measured 2026-11-07: a 6-row step moves ~29 GB (17 GB of routed experts at ~22 unique/layer); at the Station's measured HBM copy rate that is 4.6 ms of bytes alone, so 5.0 ms needs 92 % of HBM for the whole step — not reachable with stock weights at 6 rows under any fusion design (`docs/M2-FUSED-CHAIN-DESIGN.md` §5, `research/round2-20261107/REPORT.md`). 7.36 needs 3.9 TB/s and is reachable (fusion ladder models ~9, persistent per-layer ~7.3). ms/token (step ÷ accepted tokens) is the number a user feels and is reported from the same instrument (`bench/headline.py`). |
| Instrument | `bench/headline.py`: ms per 6-row DSpark step (cuda events, live steps ≥ 32, 8 fixture prompts), tokens/step, ms/token; first reading 2026-11-07 **22.3 ms/step, 2.97 tok/step, 7.5 ms/token** (`docs/HEADLINE-20261107.md`) |
| Secondary gates | quality gates as calibrated (margin-weighted flips vs reference, tool gauntlet 12/12, GSM8K-200 within control spread); C32 is a declared non-goal of the runtime; 1M context gates production candidacy only |

## Decisions (settled)

| # | Decision | Answer |
|---|---|---|
| 1 | Definition | banner = "shortest step"; chain first, per-layer persistent as stretch; no one-kernel-per-forward |
| 2 | Scope of the step | whole step incl. MoE section. **Two recipes:** (A) no-sidecar, Grace-only; (B) RTX PRO 6000 sidecar |
| 3 | Engine | **standalone decode runtime** — no vLLM, no SGLang in the serving process. vLLM/SGLang/Garner trees are reference and kernel donors |
| 4 | Constraints inherited | stock weights (no NVFP4/requant), no KV quantization, no REAP/pruning/router masking, quality gates. **Reopened:** weight *layout* (shuffle/retile) may change; values bit-identical to the pinned checkpoint |
| 5 | Forward ownership | runtime owns the whole forward (prefill, indexer, Engram, decode, sampler). Only the decode step is optimized; prefill speed is a non-goal until the headline is met |
| 6 | Context ladder | 32K → 128K → 1M; KV *format* fixed at day 1 (sliding-window MXFP8 528 B record, `fp8_ds_mla` compressed store, groups `[32,32,32,32,128,8]`, block 128) |
| 7 | Serving surface | bench harness first; minimal OpenAI `/v1/chat/completions` with vendored DSv4.1 parsers later (M4). Rows: C1–C8 designed in; C32 non-goal |
| 8 | Stack | Python host (torch for alloc/loading) + CuTe DSL kernels + Triton glue; CUDA C++ for the persistent scheduler/mailboxes. Own compile cache (image's `cute.compile` forces `no_cache`) |
| 9 | Borrow policy | M0 borrows every existing kernel as a library (DeepGEMM MegaMoE, FlashMLA mode-1, FlashInfer MXFP4 MoE/MXFP8 GEMM, vLLM Engram gather, Garner's draft samplers). Replace a piece only when the step trace names it and the replacement is measured against the borrowed one in the same window. sgl-nvidia-mega `sm103/` = reference source, not a dependency. Candidates from GrokMilo r3 (unverified): vLLM #64253 SWA metadata, #63921 `decode_topk` + cluster/DSMEM sparse MLA |
| 10 | Speculation | M0 = 1-row greedy (parity); DSpark k=5 is its own milestone (M1); kernels tiled for 6–8 rows from day 1 |
| 11 | GPU policy | **Station = dev box for the project's duration.** Production DSV4.1 moves to Sparks 1–4 SGLang `:8888`. Pg61 stays restorable on disk. Project outranks guest windows unless James overrides per request |
| 12 | Recipe order | **A (no-sidecar) first.** Thesis: a fused layer gathers cold expert rows directly from pinned Grace over C2C inside the kernel — no sidecar round trip, no `_finish2` spin. B = same runtime + GP1-style persistent cooperative RTX kernel. If A's C2C self-fetch loses to B on measurement, that is a published result |
| 13 | HBM/Grace split | static map (hot in HBM, cold in Grace, fixed at boot) for M0–M2; slot cache deferred; "everything in Grace" is one bounding *measurement* at M2, not a recipe |
| 14 | Determinism | **design requirement:** bit-identical across runs and relaunches (fixed-order reductions, atomics off the data). Parity vs our own replays = `torch.equal`; vs Pg61 = spread-based (Pg61 is not bit-stable) |
| 15 | People | Milo + subagents; Garner McCloud invited after M0 (repo public from first commit). Kernel-design workers on Anthropic; desk/grunt on Spark DSV4.1 (`custom[dsv41-spark]`) |
| 16 | Budget | soft lines **$200/mo each**: Anthropic, Nous, Grok; plus local inference. Overage tolerated; reported weekly, warn-at not cap |
| 17 | Kill rule | **M0 by 2026-12-18 (week 6)** or Pg61 is restored and the project parks as desk research. Park decision is James's. Milo flags a projected miss by week 4 (2026-12-04) |
| 18 | Repo | `github.com/jmeadlock/gb300-dsv41-decode`, Apache-2.0, local `~/hermes/gb300-dsv41-decode/`. Milo byline = session model |
| 19 | Cadence | Friday status: milestone table, spend per provider vs soft lines, box-time log, blockers. Launch messages end with a 12 h ETA; slips flagged. Blog only at measured milestones, teaching voice, corrections published |
| 20 | Verdict home | Obsidian `Ideas/Inference - Decode Fusion.md` under the verdict board; each milestone ends in a dated verdict there |

## Milestones

| M | Target | Done when |
|---|---|---|
| **W0** | first dark window (measurement only, ~1 day) | (1) Pg61 per-role nsys step breakdown at 7.36 ms; (2) per-boundary PDL gap histogram on Pg61 (kernel N end → N+1 first CTA → N+1 wait release); (3) 1-row greedy logit capture on the agent fixture = M0 parity reference. Taken while Pg61 still exists; then Pg61 down for good |
| **M0** | **2026-12-18** → *met 2026-11-07 under A3 (approved)* | single Python process, no vLLM; loads `1bae0e10` from `/models`; Engram pinned in host; experts split HBM/Grace per existing map; borrowed kernels end to end; greedy 1-row parity vs W0 capture (margin-weighted flips within calibrated bar) and `torch.equal` on own replays; 32K ctx; step trace with named boundaries + per-boundary gap histogram; step ≤ Pg61 + 10% at 1-row-equivalent |
| **M1** | Jan 2027 → **done 2026-11-07** | DSpark k=5 in the runtime, 6-row step, headline instrument live → first honest ms/step vs 7.36. *Done: 22.3 ms/step, 2.97 tok/step, 7.5 ms/token at the morning close; by day's end 20.3 ms/step, 6.99 ms/token, the 6-row step bit-identical to the 1-row step on 8/8 fixture prompts; verify and draft bit-equal to the reference (`docs/HEADLINE-20261107.md`).* |
| **M2** | Feb 2027 → *started 2026-11-07 (`docs/M2-FUSED-CHAIN-DESIGN.md`)* | recipe A fused chain: mHC / attention / indexer boundaries fused; C2C self-fetch of cold rows. The thesis test. Plus the "everything in Grace" bounding measurement *Round 1 (11/07 pm): L1 fork, L10 tail, L5 PDL, L3 MegaMoE, GEMV v3 all measured, none adopted at 6 rows (`docs/DAY2-REPORT-20261107.md`); matched gap ledger vs Pg61 before any persistent-kernel work (Astra review).* *Night 11/07→08: residency on, staging overlap, SF pool → SF-beside layout: **18.74 ms/step, 6.46 ms/token, 1-row 9.76**, all bit-identical (`docs/MORNING-20261108.md`). MegaMoE measured and declined (A3).* |
| **M3** | Mar 2027 | recipe B: persistent cooperative RTX kernel (GP1 shape) on the same runtime |
| **M4** | — | minimal OpenAI surface; 128K → 1M; production-candidacy review (C32/seat rules revisited then, not before) |
| stretch | — | per-layer persistent kernel, only if M2's trace still shows boundary cost worth it |

Month labels are targets, not promises.

## What this project is not

- Not a vLLM/SGLang fork or port channel (those stay: Pg61 on disk, knapcio on Sparks 1–4, Owen's spark-ds41f on 5–8).
- Not a quantization or pruning project.
- Not a multi-GPU/EP project. One Station, one GB300, optionally one RTX PRO 6000.
- Not a throughput project. C32 and seat economics are out of scope until M4.

## Evidence this charter stands on

- Lane skill `dsv41-gb300-serving` (step anatomy v27: 11.2 ms, 98.6% GPU-busy, 1,305 kernels/step, 40% of the step in sub-10 µs kernels; Pg61 7.36 ms; peer numbers; traps).
- `~/hermes/research/dsv41-catchup-6000-20261027/r6-outside-view/W1-megakernels.md` (fusion pool −0.7…−1.4 ms realistic; one-kernel-per-forward "do not bother"; the boundary-gap histogram as the prerequisite measurement).
- `~/hermes/research/dsv41-sidecar-release-20261031/` (Pg61 release round, Pg62 queue).
- Obsidian `Ideas/Inference Engineering.md` verdict board and lever notes.
- GrokMilo round 3 (2026-10-28, untrusted): determinism cause (b12x direct-routing atomics), #64253/#63921 as borrowable pieces, and the peer evidence that fusion alone does not move end-to-end — consistent with W1.

## Repo layout (to create on sign-off)

```
runtime/        Python host: loader, memory plan, scheduler, step loop
kernels/        CuTe DSL / Triton / CUDA C++ (persistent scheduler, mailboxes)
bench/          step profiler (nsys cudaProfilerApi), gap histogram, fixtures, harness
recipes/
  grace-only/   recipe A
  sidecar/      recipe B
third_party/    vendored kernels + licenses
docs/           lever ledger mirror, milestone verdicts, status
PROJECT.md      this charter
CUTOVER.md      Station → Sparks cutover checklist (reversible)
```

## Amendments (proposed 2026-11-06; **A1, A2, A3 approved by James 2026-11-07**) — see `docs/LEDGER.md`

- **A1 (decision 4 wording).** "No KV quantization" was written against Pg61's `--kv-cache-dtype fp8_ds_mla`. D1 (`docs/D1-KV-NUMERICS.md`) shows the *model definition* (DeepSeek's reference) attends FP4-QDQ'd compressed-KV and index-K values, and that Pg61 skips that QDQ and stores MXFP8 of the raw latent instead. Proposal: the runtime's default KV numerics = the reference's (E2M1/E4M3-16 QDQ of the latent, FP8/32 window, FP4/32 index-K), with a `--kv-numerics pg61` switch kept only for the parity instrument; decision 4 then reads "no quantization below the model definition's own".
- **A2 (decision 13).** Static map for M0 (parity) stays. For M2 (recipe A thesis test) the static map is replaced by adaptive residency (our v26ar seed map + swap policy, ported into the runtime): at the static map's cold share the C2C self-fetch costs ~3.9 ms/step on bandwidth alone (`docs/C2C-PROBE-20261106.md`), so the thesis cannot be tested with it. The M2 bar is restated as `cold bytes/step ÷ 390 GB/s` exposed time, with copies pipelined under hot-expert work (shown fully concurrent).
- **A3 (M0 parity bar).** First light (`docs/FIRST-LIGHT-20261106.md`) shows the plan's bar ("no worse than Pg61's own A/A2 spread") measures vLLM against itself; one quantization choice flipped inside our runtime perturbs the fixture as much as the whole gap to Pg61 (~0.02 nats). Proposed bar: self-replay bit-identical; 0 margin-weighted top-1 flips and mean |Δlogprob| ≤ 0.05 vs Pg61 over the shared prefix; tiered == all-HBM bit-exact; long-horizon quality check deferred to M1. All but the last are already met.

