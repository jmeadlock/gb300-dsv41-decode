# Port plan — our wins into Pg61 (Garner's recipe) and upstream vLLM

Decided by James 2026-11-08 ~15:45 CST: "let's port the wins into pg61/upstream." This supersedes the
runtime-ladder as the project's shipping vehicle. The standalone runtime stays as the parity oracle, the
measurement instrument and the donor of working code.

## The finding that shapes the port

**No single-GB300 vLLM path for DeepSeek-V4.1-Flash exists.** Stock vLLM 0.31 has no expert offload for
this model (only Engram `cpu_offload`). Garner's recipe (`reference/garner-8043a681/`, MIT) *requires* the RTX
PRO 6000: `coldskip` discards the 129 cold experts/layer at load, peer tier v2/v3 computes them on the RTX; README:
"the model does not fit without the sidecar." Our runtime is today the only thing that runs this checkpoint on a
stock Station. So the highest-value port is not a kernel — it is **a Grace cold-expert tier for vLLM**, built from
our loader + staging + residency, dropped into Garner's component framework at the seam peer3 already uses.

## Inventory — what we measured, and whether it transfers

| our win (receipt) | measured in our runtime | Pg61 / vLLM today | transfers? |
|---|---|---|---|
| Host tier loader: `cudaHostRegister` on `MADV_HUGEPAGE` chunks, UVA alias reads, chunk-wise copies (`runtime/loader.py`, T2T3) | 251 GiB registered in ~0.7 s; copy engine 390 GB/s, kernel reads ~230 GB/s | no expert host tier at all | **YES — P1 core** |
| Cold-expert staging on a forked capture stream under hot GEMM, per-projection launch shapes (M3-S1) | exposed cold work 4.55 → ~2 ms/step | sidecar: 0.02 ms exposed (two GPUs) | **YES — P1** (the single-GPU cold path) |
| SF layout 'beside' in the expert pool, no hot-SF copies (M3-S3) | −0.32 ms, 5.2 GiB HBM back | MegaMoE reads its own SF format | **P1 design question** (only if cold rows are staged into pool slots rather than computed from UVA) |
| Adaptive residency (v26ar seed + swap policy, `runtime/residency.py`) | −0.37 ms; cold share 1.00 → 0.65 % | static `rowmap-mix-v1-h255` | **YES — P1** (swap policy on top of the Grace tier) |
| Fixed-order fp32 head + mHC mixes, row-invariant by construction (T9b-5) | bit-identical R=1 vs R=6; cuBLAS differs 86–91 % of elements | cuBLAS fp32; not row-invariant | **YES — P2 determinism** |
| Fixed-order expert combine (our dg-inplace path; GP2 pre-registered on the lane) | self-replay 8/8 | sidecar bf16 atomics → 79 % of calls differ run-to-run | **YES — P2 determinism** |
| D1: model definition QDQs compressed-KV/index-K to FP4; `fp8_ds_mla` stores MXFP8 of the raw latent (`docs/D1-KV-NUMERICS.md`) | our default = reference numerics (A1) | vLLM skips the reference QDQ | **YES — P3 upstream fidelity issue / opt-in flag** |
| PDL + pre-wait L2 weight prefetch in Triton (L5 probe) | −5.07 µs/edge mixed chain, bit-identical | PDL everywhere already; pre-wait prefetch not | **maybe — P4**, after P1 |
| Grouped prefill on reference kernels (M4-P2) | TTFT 3.9 → 0.95 s | vLLM fused MoE prefill, TTFT ~30 ms | no |
| Loop overrun poll (LOOP-OVERRUN) | 97 → 8 overrun steps | vLLM scheduler owns this | no |
| Router plan-gemv fused (M2-T15) | R=6 40 → 19.6 µs/layer | Garner w4 router: one launch already | no |
| Fused glue / hc_norm split / merged-N GEMV (T9b-2/4) | glue 2.5 ms/step | Garner mHC TileLang 1.8 ms/step | no (theirs is faster) |
| bf16-stored head | −0.15 ms | v27 already bf16 head | no |
| FlashMLA packed records, DG in-place MoE | borrowed from vLLM | origin | no |

## Phases

| P | deliverable | gate | box time |
|---|---|---|---|
| **P1 Grace tier** | new Garner component `gracetier` (replaces `peer3`/`coldskip` when no RTX): cold experts pinned in Grace by our loader; w4 router's cold-row publish lands in a GB300-side consumer that computes cold experts from UVA weights (or stages them into spare pool slots) and adds them at the existing finish seam; residency swap policy on top. Result: **DeepSeek-V4.1-Flash on a stock GB300 Station in vLLM**, full surface, C1–C32, prefill, 1M KV. | card method (`dsv41-gb300-serving`): control = Pg61 with sidecar; arms = gracetier; TF flips within A/A2 spread; gauntlet 12/12; **C32 before any verdict**; prefill/TTFT; step ms/step + ms/token published both framings | design desk 0; mini-test window ~1 h; card windows ~2 h each |
| **P2 determinism** | fixed-order combine + fixed-order fp32 head/mixes as an opt-in (`DS41_DETERMINISTIC=1`): greedy bit-identical run-to-run and relaunch-to-relaunch on Pg61 | `torch.equal` across 2 runs × 8 prompts × 256 tokens; cost in ms/step reported; C32 | 1 card |
| **P3 KV fidelity upstream** | **WITHDRAWN 11/08 16:58** on re-reading D1: vLLM already ships the model definition's numerics as `nvfp4_ds_mla` (the SM100 default); `fp8_ds_mla` is Pg61's explicit opt-out (James's no-KV-quantization rule). Nothing to upstream; at most a docs note that `fp8_ds_mla` stores more bits than the model definition but attends different values. Our A1 finding stays a lane fact, not a vLLM bug. | — | none |
| **P4 PDL pre-wait prefetch** | apply to Garner's Triton fast paths where a weight image is small enough for L2 | bit-identical; ms/step | 1 card, after P1 |

## Upstream channels
1. Garner's `dgx-station-recipes` (MIT) — PR the component; Garner is the natural reviewer (invited after M0 per charter).
2. vLLM main — P3 issue/PR; P1 only if vLLM grows a generic expert host tier for this model family (desk check in the design round).
3. J-M-Recipes — a `deepseek-v4.1-flash-gb300-single-gpu` recipe once P1 has receipts.

## Rules carried over
Stock weights; no KV quantization below the model definition; every number from a receipt; production lane
`:30006` stays up except inside a pre-registered card window on James's word; restore Pg61 at every window end.
