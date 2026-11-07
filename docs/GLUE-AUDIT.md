# GLUE-AUDIT — torch glue in the 1-row decode step, and the T9b-2 fusion plan

T9b-1 desk part (docs/HANDOFF-T9B1-GEMV.md item 4). Written 2026-11-06 on the Mac; nothing here ran on a GPU.
Nothing in this file is implemented; it is the plan for T9b-2.

## Method

Two independent passes, which agree with each other.

1. **Reading the source.** I walked `model.py` `Block.forward` (970-996), `hc_mixes`, `hc_pre` and `hc_post`
   (950-968), `RMSNorm` (281-293), `apply_rotary_emb` (392-407), `Attention.forward` and `_window_kv`
   (701-791), `Gate`, `Expert` and `MoE` (811-906), and `Engram` (350-366). For the static path I also walked
   `runtime/static_step.py` (`_attn` 461-485, `_block` 487-502, `_body`) and `runtime/dg_moe.py`
   (`DGMaskedMoE.__call__` 243-277). The per-layer counts in the tables below come from this reading.
2. **Counting ops.** `bench/glue_census.py` (Mac, CPU, 4 s) records every aten op that one decode step dispatches,
   using a `TorchDispatchMode`. It runs on a synthetic checkpoint that has the real model's structure:
   - 40 layers with the real compress ratios (2×0, 18×2, 20×1);
   - kv sources at layers 2/8/14/20, index sources at 2/8/14/20/24/28/32/36, candidate source at 20;
   - Engram at layers 1 and 14;
   - 6 active routed experts, hc_mult 4, and 20 Sinkhorn iterations.

   Dims are tiny. That does not matter here: no op count on the decode path depends on dims.

   It covers three paths:
   - eager-ref: the reference `model(tok, pos)`, which is the anatomy's path;
   - static-ref: T8;
   - static-dg-gemv: T9a-1 + T9b-1, which is today's step plus the GEMV.

   Each reference, T8, DeepGEMM or GEMV kernel counts as one launch, plus the torch ops its kernel.py wrapper runs.
   `tests/test_glue_census_cpu.py` pins the results.

**Cross-check against the anatomy** (`docs/step-anatomy-m0-20261106.json`, eager, nsys):
- **Reference kernels match exactly.** The census sees, per step: `fp4_gemm` 720, `fp8_gemm` 290, `act_quant` 1090,
  `sparse_attn` 40, `hc_split_sinkhorn` 80 and `fp4_quant` 13. These are the same numbers nsys measured.
- **Non-reference launches do not match.** The census has 8,143 launching aten ops + 3,496 allocations (NaN-filled
  because the anatomy ran with deterministic fills on) = 11,639. nsys counted 18,316 non-reference kernels.

  The gap of ~6,700 sits in the reference MoE. nsys counts 14,480 MoE launches per step (362 per layer); the census
  counts ~8,760 (219 per layer). The attention role agrees within ~10/layer: nsys 76/layer, census ~67.

  The cause is ops the census sees as one op that launch several kernels on CUDA: 240 `nonzero` + 240
  deterministic `index_put_` per step in the per-expert loop (model.py:897-903). On CUDA these become
  DeviceCompactInit/SelectSweep/ReduceSingleTile/write_indices, a radix sort, indexing_backward and
  `_assert_async` — 240–486 each in `top_kernels`.

  T8 removed that loop. So the census is a lower bound for the eager path, and close to exact for the static
  paths: their only multi-kernel ops are `sort`/`topk` (48 + 49 per step).

**The "11,570 elementwise launches" number does not describe today's step.** That count was the eager reference
with fills on. Most of it was:
- the reference MoE's routing glue, which T8 replaced;
- the NaN-fill of 3,496 allocations. The T9a-1 step cell ran with fills off (`docs/first-light-dg-20261106.json`).

## The step today (static-dg-gemv, 1 row, per step; census)

| class | per step |
|---|---|
| launches: kernels + aten GEMMs + glue | **6,725** |
| reference/Triton/DeepGEMM kernels (GEMV 290, act_quant 450, gather 240, DG 120, sinkhorn 80, sparse_attn 40, fp4_act_quant 13) | 1,233 |
| aten GEMMs: mm/bmm (mHC mixes 80, gate 40, wo_a einsum 40, compressor/indexer fp32 linears, head) | 187 |
| **glue: elementwise, reduction, index, copy, cat, topk/sort** | **5,306** |
| allocations: no launch with fills off; +1,496 fills with fills on | 1,496 |
| static-ref (T8) for comparison: total / glue | 9,285 / 6,626 |
| eager-ref for comparison: total / glue | 10,376 / 7,956 (lower bound, see above) |

At the anatomy's measured ~1.0 µs per small elementwise kernel (11.74 ms / 11,570), 5,300 glue launches would be
**~5 ms of the 44.6 ms step**. That is an estimate. This step has no nsys trace under a graph. The first box action
for T9b-2 should be one, to time the clusters below before building any of them.

**Per layer** (glue + kernels + GEMMs, by layer type):

| type | layers | launches/layer |
|---|---|---|
| r1 / r2 (plain) | 30 | 157 |
| r0 (first two layers; also carry the per-step scalars) | 2 | 159.5 |
| r2 kv+index source | 3 | 200.3 |
| r1 index source, candidate consumer | 4 | 179 |
| r1 kv+index+candidate source (L20) | 1 | 225 |
| step-level (embed, Engram ×2, hash, head norm, sample) | – | 154 |

**One plain layer, 157 launches** (static-dg-gemv; by source, which agrees with the census):

| region | n | ops (source lines) |
|---|---|---|
| mHC mixes ×2 | 16 | `.float()`, `square`, `mean`, `+eps`, `rsqrt`, `F.linear`(mm), `*rsqrt`, sinkhorn kernel (950-957) |
| hc_pre ×2 | 8 | `.float()`, `*`, `sum(dim=2)`, `.to` (959-962) |
| attn_norm + ffn_norm | 16 | 8 each: `.float()`, `square`, `mean`, `+eps`, `rsqrt`, `*`, `weight*`, `.to` (288-293) |
| q_norm + kv_norm | 16 | same 8 each (static_step.py:466, 469) |
| fp8 Linears in attention | 8 | wq_a, wq_b, wkv, wo_b: act_quant + GEMV each (model.py:196-205) |
| RoPE ×3 (q, kv, inverse on o) | 10 | `.float()`, complex `*`, `copy_` each, + 1 `clone` from `freqs_cis.conj()` (392-407) |
| attention body | 9 | act_quant(kv)+QDQ `copy_`, `index_copy_` (ring slot), `cat` ×2, sparse_attn, act_quant(o)+QDQ `copy_`, einsum→bmm |
| hc_post ×2 | 10 | `*` ×2, `sum`, `+`, `type_as` (964-968) |
| gate | 13 | `.float()`, mm, `/temp`, `softplus`, `sqrt`, `+bias`, `topk`, `gather`, `sum`, `+1e-20`, `/=`, `*=` (811-829) |
| MoE (T9a-1 adapter + reference glue) | 38 | `sort`, `index`; 6 gather_rows; 8 `copy_` (load_act q/s ×2, convert_scales ×3, ids_step); act_quant ×2; DG ×3; `.float` ×2, `clamp` ×2, `silu`, `*` ×2, `.to`; `zeros_like`, `+=` ×6, `+= shared`, `type_as` (dg_moe.py:251-277) |
| shared expert | 13 | 3 × (act_quant + GEMV); `.float` ×2, `clamp` ×2, `silu`, `*`, `.to` (843-853) |

## Fusable clusters, ranked by launches saved (today's path, per step)

"Saved" is the census count of the ops replaced, minus the fused kernels that replace them. All counts are per
1-row step. Bit-identity risk is judged against the current static path, which is the L1/L2 equality gate:
- **none**: a pure data-movement or op-order-identical rewrite. Must be bit-identical, and is tested
  bitwise.
- **low**: elementwise math only. Bit-identical if written with the same fp32 op sequence and libdevice functions
  (not Triton's fast `exp`/`rsqrt`).
- **medium**: changes a reduction's order: the norm statistic, a dot product, a cross-copy sum. Expect 1-ulp fp32 →
  occasional bf16-ulp flips, the same class as the T9b-1 GEMV. Gate it with the GEMV rules: mismatch ≤0.5%,
  L1/L2 8/8 within the impl, and parity vs Pg61 in the A3 class.
- **high**: can change a discrete choice: topk ties, expert selection.

Every fused kernel stays behind a flag with `ref` as the default, as with `--moe-impl` and `--fp8-impl`.

| # | cluster | replaces (per instance) | instances | launches before → after | saved | risk |
|---|---|---|---|---|---|---|
| 1 | **hc_pre + RMSNorm + act_quant** (attn_norm, ffn_norm; head) | hc_pre 4 + norm 8 → 1 kernel writing the bf16 normed row and its MXFP8 quant once; this also removes the duplicate act_quants of that row (wq_a + wkv: 2 per layer; DG x + shared w1 + w3: 3 per layer) | 81 | 972 + 200 → 81 | **1,091** | medium (the norm's mean over 4096; the hc sum over 4); none for the act_quant dedupe |
| 2 | **SwiGLU + act_quant, routed (DG) and shared** | `.float` ×2, clamp ×2, silu, `*`(×2 routed), `.to`, act_quant(h) → 1 | 80 | 680 → 80 | **600** | low (silu via libdevice `expf`; clamp order kept) |
| 3 | **kv path: kv_norm + RoPE + QDQ + ring write** | norm 8 + rope 3 + act_quant + QDQ `copy_` + `index_copy_` → 1 | 40 | 560 → 40 | **520** | medium (norm); RoPE low if the complex multiply is written as torch does it (no FMA contraction) |
| 4 | **router: gate math + topk + sort/index** | `.float`, `/temp`, softplus, sqrt, `+bias`, topk, gather, sum, `+eps`, `/=`, `*=`, `sort`, `index` → 1 (mm kept) | 40 | 560 → 40 | **520** | **high** (topk tie order and softplus's threshold form decide which experts run). Gate on recorded real router logits, bitwise. |
| 5 | **mHC mixes → sinkhorn prologue** | `.float`, square, mean, `+eps`, rsqrt, `*rsqrt` → folded into `hc_split_sinkhorn` (mm kept) | 80 | 560 → 80 (+80 mm) | **480** | medium (the statistic over 16384) |
| 6 | **q path: q_norm + act_quant(wq_b)**, also writes qr bf16 for the indexer | norm 8 + act_quant → 1 | 40 | 360 → 40 | **320** | medium (norm) |
| 7 | **mHC post** | `*` ×2, sum over hc, `+`, `type_as` → 1 | 80 | 400 → 80 | **320** | low–medium (4-term fp32 sum; replicate the order) |
| 8 | **MoE combine** | `zeros_like`, 6 × `+=`, `+= shared`, `type_as` → 1 | 40 | 360 → 40 | **320** | none (sequential fp32 adds in slot order, exactly reproducible) |
| 9 | **DG adapter copies** | load_act q/s copies, 3 × convert_scales, ids_step copy → 0 (act_quant writes into DG's input buffers; e8m0→fp32 in the gather) | 40 | 320 → 0 | **320** | none (data movement) |
| 10 | **o path: inverse RoPE + QDQ** (+ hoist the conj table) | rope 3 + `clone` + act_quant + QDQ `copy_` → 1 | 40 | 240 → 40 | **200** | low |
| 11 | **one gather for all 6 staging tables** | 6 × gather_rows → 1 | 40 | 240 → 40 | **200** | none |
| 12 | **fp8 Linear merges on top of the GEMV**: wq_a‖wkv one launch (same input), shared w1‖w3 one launch, act_quant into the GEMV prologue for wo_b / indexer / Engram | | – | ~130 → 0 | **~130** | none for the merges (row-independent GEMV); low for prologue quant (a per-32 amax is order-free) |
| 13 | **q RoPE in the wq_b GEMV epilogue** (round to bf16 first, then rotate as torch does) | rope 3 → 0 | 40 | 120 → 0 | **120** | low |
| 14 | **Indexer** (8 source layers + 4 consumers): score `relu·w·sum`, masks, topk, its norm/rope/QDQ | ~180 → ~30 | 12 | ~180 → ~30 | **~150** | high (topk ties over candidates) |
| 15 | **Engram** (2 layers) + static hash | gate math 40/layer + hash 45 → ~3 | 2 | 125 → ~3 | **~122** | medium (two rsqrt statistics, signed sqrt) |
| 16 | **concat-free attention inputs**: window and compressed KV/idx as views of one buffer | `cat` ×2 | 38 | 76 → 0 | **76** | none |
| – | compressors, step scalars, sampling | small; leave alone | | ~60 | – | – |

**Total if all of it lands:** ~5,490 of the 6,725 launches, leaving ~1,240 per step: the GEMVs (~210 after
merges), DG, sparse_attn, sinkhorn, aten GEMMs and the fused kernels. These are estimates from the census, not
measurements.

**Recommended order.** First the no-risk set: #8, #9, #11, #12, #16. That is ~1,050 launches with bitwise equality
required and no new numerics class, so it can land in one change with the existing L1/L2/L0 gates.

Then the low-risk set: #2, #7, #10, #13 (~1,240). After that, the medium set: #1, #3, #5, #6 (~2,410, of which
#1's 200-launch act_quant dedupe is no-risk and can go with the first set). It has the
most launches, but all four share one hard part, the RMS statistic. Build one reduction helper and test it once
against torch's `mean` on real activations. If it is bitwise with torch, these four drop to "low". If it is not,
they enter the GEMV's mismatch class together.

Leave #4 and #14 for last, or never fuse their topk: a tie flip changes which experts or blocks run. Fusing only
their elementwise parts and keeping `torch.topk` saves most of #4 (~9 of 13 per layer) with no high-risk change.

## Notes for T9b-2

- **GEMV merges (#12) are free on the numerics side.** Each output row of the T9b-1 GEMV depends only on its own
  weight rows and scales. A concatenated [wq_a; wkv] or [w1; w3] weight gives the same bits as two launches. The
  loader would need a concatenated copy (extra HBM equal to those weights; size them from
  `checkpoint-1017-inventory.json`). The cheaper option passes two weight/scale pointer pairs to one launch.
- **The GEMV epilogue is the natural home for #2 and #13,** but only with BLOCK_N a multiple of 32 so one program
  owns whole act_quant blocks. The current tune candidates (BLOCK_N ∈ {8, 16, 32, 64}) would have to drop 8 and 16
  for those layers.
- **Triton numerics traps when matching torch bit-for-bit:**
  - `tl.exp`/`tl.sqrt`/`tl.rsqrt` may lower to approximate PTX. Use `tl.extra.cuda.libdevice` (`exp`, `sqrt_rn`,
    `rsqrt_rn`) and division-by-`div_rn`.
  - Fused multiply-add contraction changes rope/complex multiplies.
  - `.to(bf16)` must be round-to-nearest-even, as torch does.
- **Measure before building.** Two things to run on the box before any T9b-2 code: an nsys trace of the
  static-dg-gemv graph step, and the census's per-cluster counts mapped onto it. If glue really is ~5 ms of 44.6,
  T9b-2's ceiling is ~4 ms. The GEMV result (`bench/gemv_bench.py`) and sparse_attn/FlashMLA may be the larger
  rungs.
