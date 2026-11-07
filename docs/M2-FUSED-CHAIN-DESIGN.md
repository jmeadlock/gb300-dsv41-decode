# M2 design: the fused chain (recipe A) at 6 rows, with a cost model built from measured kernels

Desk memo, 2026-11-07 03:35 CST, for `docs/HANDOFF-M2-FUSED-CHAIN-DESIGN.md`. Nothing here ran on the box. The only
thing I ran was the existing CPU census, `bench/glue_census.py --paths static-dg-gemv-fused-r2,static-dg-gemv-fused`
on the Mac, with output to scratch outside the repo.

Labels:
- **[M]** measured, with the receipt cited.
- **[E]** my estimate, with the method stated.
- **[A]** an assumption that still needs measuring.

Cite prefixes: `model.py` and `kernel.py` are `reference/deepseek-inference/`; `rt/` is `runtime/`.

## 0. Corrections to the handoff

1. **The 10.52 ms step is not 2,827 kernels.** 2,827 is the count at 13.88 ms (`STEP-ANATOMY-R2-20261107.md:3`).
   - T9b-4 then removed 160 split-K reduces (250 → 90), 120 DeepGEMM SF converts, 80 GEMV launches by pairing, and the
     unfused router glue on dg-inplace (`rt/README.md:801-808, 843`).
   - The CPU census of today's path counts **1,742 ops/step** [M, census], i.e. 35 per plain layer. On the GPU it adds
     the 90 remaining reduces, cuBLAS's two-kernel mm, and the topk/sort sub-kernels.
   - So the 10.52 step is **≈ 2,000 kernels at ≈ 5 µs** [E]. Nobody has taken an anatomy at 10.52; that is M2-T13's
     first job.
2. **The bandwidth floor is not ~1.2 ms.** The R2 anatomy's "~9 GB" leaves out two large reads:
   - the routed experts: 6 × 18.80 MB × 40 = **4.5 GB at 1 row** (`A2-RESIDENCY-DESIGN.md:7`);
   - the bf16 `wo_a`: 64 MiB × 40 = 2.7 GB (`model.py:785-789`, loader dequantizes it).

   Corrected bytes [E, from shapes]:

   | step | GB | at the measured 6.3 TB/s HBM copy rate (`overlap-probe-gb300-20261106.json`) |
   |---|---|---|
   | 1 row | 15.9 | **2.5 ms** |
   | 6-row verify | 28.3 | 4.5 ms |

   About 8 ms of the 10.52 is therefore non-bandwidth time, ≈ 4 µs per launch [E]. That matches the handoff's
   "~3–4 µs per boundary".
3. **The boundary cost sits inside kernel durations, not between them.** The sum of kernel durations at 13.88
   (eager, CUPTI) is 13.91 ms, against a graph step of 13.88 (`step-anatomy-r2-20261107.json` `top_kernels`) [M].
   - So the graph adds nothing measurable between nodes.
   - What fusion removes is each kernel's fixed latency: CTA ramp, first DRAM round trip, tail, and one-program-per-row
     reductions.
   - A fused kernel keeps its internal dependencies. Each costs ~0.6–1.15 µs (grid sync or all-SM gather on sm_120,
     `W1-megakernels.md:13`; unmeasured on the two-die GB300).
   - So **a removed boundary saves ≈ 1.5–3 µs, not 4** [E].
4. **R6/R1 = 2.0 was measured on the slow path.** It used dg-masked (every unique expert copied) and ref glue
   (`M1-T10-RESULT-20261107.md:3-9`). The 6-row step still runs reference glue for every cluster except
   `swiglu_shared` and `hc_post` (`rt/fused_glue.py:115`).
   - Bottom-up, the fast path with row-generic glue is **13.2–15.4 ms** per 6-row verify [E, §1], not 21.
   - If the fast-path measurement comes in near 21, the excess is M1 port work: fused clusters at R rows, the dedup
     glue. It is not M2's.
5. **Cold traffic at 1 row on the seed map is 6.07 cold calls/step** (p50 2, p90 11.7, max 69) [M,
   `first-light-t9b4-20261107.json` `step_cell.cold`]. The pointer-table gather already self-fetches them from Grace
   inside the graph (`STEP-ANATOMY-M0-20261106.md:30, 40`).

## 1. Where the time is, as a launch-count problem

Each row is one chain segment of a plain (non-source) layer, in `model.py` order (`Block.forward` 983-996).

Column sources:
- **launches**: the census of today's path, plus the GPU-only sub-kernels.
- **µs @13.88**: each segment's kernel families at their R2-receipt mean µs [M means; attribution by launch count].
  Where cuBLAS families had to be assigned to call sites by count, I assigned them that way: `nvjet` 40 × 14.5 µs =
  `wo_a`; `dot`+`reduce_1Block` 80 × 5.5 µs = mHC mixes; `kernel` 40 × 4.8 µs = gate.
- **µs @10.52**: the 13.88 figure minus T9b-4's parts [E]. GEMV µs come from `gemv-bench-r2-20261106.json` [M].
- **6 rows**: [E] from the M1-T10 census (`first-light-rows6-20261107.json` `v0_census`). GEMV at M=6 is +0–5 %
  [M]. FlashMLA at b=6 ≈ b=1 [M, `flashmla-probe-20261106.json`]. Unique experts per layer go 6 → 22.45 [M,
  `docs/data/unique_experts_vs_k.json`].

| # | segment (model.py) | launches/layer (10.52) | µs/layer @13.88 | µs/layer @10.52 | µs/layer, 6 rows | what scales with rows |
|---|---|---|---|---|---|---|
| S1 | attn mix + hc_pre + attn_norm + act_quant (950-962, 984-986, 288-293) | 5: `hc_norm`, mixes mm (2 kernels), `*rsqrt`, sinkhorn | 32.7 (hc_norm 21.3) | **17.4** (hc_norm ~6, split) | 18–21 | none in time (per-row programs). cuBLAS fp32 mixes mm differs in *bits* at M=6 (91 % of outputs) [M] |
| S2 | q path (771-773) | 3: `wq_a‖wkv` pair, `q_norm`+quant, `wq_b`+RoPE | 27.3 | 28.7 | 29–30 | none (GEMV flat) |
| S3 | kv path (706-708, 719) | 1: `kv_norm`+RoPE+QDQ+ring | 12.2 | 3.4 | 3.5–4 | none |
| S4 | indexer + compressor (527-581, 458-485), **per step** | ~250/step | ~0.75 ms | ~0.75 ms | 0.8–1.0 ms | per-row scores and top-k |
| S5 | sparse attention (776-781) | 3: 2 × `cat`, FlashMLA | 13.7 | 13.7 | 14 | none [M] |
| S6 | o path + attn hc_post (782-790, 964-968) | 6: o RoPE+QDQ, `wo_a` bmm, act_quant, `wo_b`, reduce, hc_post | 40.7 | 40.7 | 41–43 | none |
| S7 | ffn mix + hc_pre + ffn_norm (991-993) | 5 | 32.7 | **17.4** | 18–21 | as S1 |
| S8 | router (811-829) | 5–6: gate mm, softplus, pre, `topk`, post | ~34 (unfused) | 18.5 | **35–40** | the 6-row dedup plan (`rt/moe_rows.py`) is unfused glue (~10 launches) [E] |
| S9 | routed MoE (891-906): stage, 3 × DG, SwiGLU, combine | 6 | 63.7 | **55.1** (DG 41.1) | **105–140** | **bytes**: 113 → 422 MB/layer; cold staging |
| S10 | shared expert (905, 843-853) | 4: pair, SwiGLU+quant, `w2`, reduce | 32.7 | 23 | 23–24 | none |
| S11 | ffn hc_post (995) | 1 | 2.1 | 2.1 | 2.2 | none |
| S12 | Engram L1/L14 + hash (1265-1266, 350-365), per step | ~80/step | ~0.25 ms | 0.25 | 0.3 | none |
| S13 | head hc_pre+norm, fp32 head, step glue (1271-1273), per step | ~15/step | 0.47 | 0.47 | 0.5 | none in time. Bits differ at M=6 [M] |
| | **plain layer** | **39–40** | 291.8 | **220** | 289–339 | |
| | **step** | ≈ 2,000 | 13.1 (measured 13.88) | **10.3** (measured 10.52) | **13.2–15.4** verify | |

**Reading.**
- At 6 rows, ~80 % of the step stays flat in time.
- Only S9 grows, by bytes: +50–85 µs/layer, 2.0–3.4 ms/step. S8 grows through glue that M1 or M2 removes.
- Fusion therefore pays at 6 rows exactly what it pays at 1 row, and the MoE has to be attacked as bandwidth, not as
  launches.

## 2. The fusion ladder

Shared rules for every rung:
- Each rung keeps the existing switch discipline: built behind a flag, self-checked at build, and turned off with a
  reason on any mismatch (`rt/fused_glue_r2.py:1-5`).
- "On-chip data" is what the fused kernel holds between the ops it absorbs. Every intermediate in this model is KB
  per row, so it fits in L2 and mostly in registers or SMEM:

  | intermediate | per row | at 6 rows |
  |---|---|---|
  | hc stream | 40 KB | 240 KB |
  | q / o | 64 KB | 384 KB |
  | qr | 2.5 KB | — |
  | gate scores | 1.5 KB | — |

  SMEM is ~228 KB/SM (sm_100 class) [A]. **The chain wins on launches and overlap, not on keeping data on chip.**

### 2.1 Bit-identity protocol per rung

**(a) Reductions we own.** Use the calibrated orders from R2.
- Torch's `square().mean(-1)` tree is (vec, min(512, last_pow2(w/4)), 4, dec), matched on 100 % of 100k rows at every
  width (`rt/fused_glue_r2.py:21-33`).
- `pick_order` is chosen at build per width **and** per row count.
- libdevice `rsqrt` equals torch; use `sqrt_rn`/`div_rn` (34-37).
- The hc_pre 4-term sum uses HC_SUM_MODES.

**(b) GEMV.**
- `numerics_key` (split partition, MMA) is pinned per shape. Pairing keeps each half's key; reduce-folding sums
  partials in split order (`rt/README.md:802-803`).
- At M=6, pin the M=1 configs [M: only `shared.w1` differs, `first-light-rows6` census].

**(c) cuBLAS ops** (mixes mm, gate mm, `wo_a` bmm, head).
- A fixed-order replacement is bit-identical only if calibration finds cuBLAS's order. That probe was never run
  (`rt/README.md:811-812`).
- Otherwise the replacement is **A3-class**, and it is **row-invariant by construction**.
- That is a parity gain: the fp32 cuBLAS mm/head are the only kernels that break V0 at 6 rows [M,
  `M1-T10-RESULT-20261107.md:17`].

**(d) Library ops stay in their accepted classes.**
- `torch.topk` stays unless tie-equivalence is proven on recorded inputs (`fused_glue_r2.py:38`).
- The TileLang sinkhorn (`kernel.py:406-475`) stays: its fragment reductions have a compiler-chosen order.
- DeepGEMM/MegaMoE and FlashMLA stay A3, as accepted in T9a-1 and T9b-3c.

### 2.2 The rungs, ordered by ms saved per engineering day at 6 rows

| rung | replaces | fused op sequence and rounding points | tool | numerics |
|---|---|---|---|---|
| **L1 mixes off the critical path** | S1/S7: mm(2) + `*rsqrt` + sinkhorn → a **forked graph branch**: 1 fused `mixes` kernel (fp32 x·hc_fn over 20,480, × rsqrt) + the reference sinkhorn | The mixes feed only `hc_post` after the sublayer and the *next* sublayer's hc_pre (`model.py:984-996`), so they have ≥ 1 sublayer of slack. The main path keeps `hc_norm` only. Data: x 80 KB/row fp32 (480 KB at 6 rows, in L2); hc_fn 1.9 MB streamed. Fork first with the current kernels (bit-identical), then fuse mm+mul | Triton (fp32 dot, no tensor cores); TileLang kept for sinkhorn | fork: bit-identical. Fused mm: A3 unless calibrated (row-invariant gain) |
| **L2 router + 6-row plan** | S8: gate mm, softplus, pre, topk, post, and at R rows the dedup sort/unique/cumsum/m_indices | One launch. Gate GEMV (bf16 weight widened in registers, which is exact; fp32 accumulate, fixed order) → last-CTA ticket: `softplus`, `sqrt`, `+bias`, top-6 of 384, gather, `sum+1e-20`, `/=`, `*= route_scale` (811-829), sort ids, slot/tier lookup, cold compaction → the MoE plan. Data: scores 1.5 KB/row | Triton | gate: A3 unless calibrated. topk: **high risk** — fused only if a defined tie rule equals `torch.topk` on 10⁵ recorded real router inputs, else torch.topk stays (−1 launch). softplus: a probe must show libdevice `log1p/exp` == torch (`fused_glue_r2.py:36-37`) |
| **L3 MegaMoE, world size 1** (`--moe-impl dg-mega`) | S9 + S10 + S11's combine: 3 DG + SwiGLU + combine + 3 shared GEMVs + reduce + combine → 1 persistent launch (+ stage when cold) | DeepGEMM `fp8_fp4_mega_moe` over `pool[0:H+S]`: dispatch, L1 over interleaved gate/up, bf16-round + clamp (cuh:1058-1066 = `model.py:845-849`), SwiGLU with `fast_math=False`, requant, L2, combine with top-k columns sorted by id, fused shared expert (`T8-T9A-DESIGN.md:109-119, 142-149`). Stub group, no network | borrowed library (decision 9) | **A3**: block-scaled UMMA plus the L2-input quantization taken from fp32 rather than bf16 (`T8-T9A-DESIGN.md:117, 121-124`). James sees the drift report |
| **L4 head** | S13 head (and the draft's shared head, `model.py:1215-1216`) | Own GEMV reading the checkpoint's **bf16** head bytes, widened exactly in registers, fp32 FMA in a fixed order. `model.py:1007` widens only for storage; the products are identical. 2.65 → 1.32 GB per head read; frees 2.47 GiB HBM (≈ 140 expert slots) | Triton | A3 vs cuBLAS (order only), row-invariant (§5.2) |
| **L5 PDL on owned kernels** | every boundary between our Triton kernels (~60 % of launches after L1–L4) | Producers trigger dependents right after issuing their last weight loads. Consumers issue their weight loads before `griddepcontrol.wait` (`W1-megakernels.md:36-37`) | Triton `gdc_*`; graph PDL edges [A: a box probe under torch capture] | bit-identical (scheduling only) |
| **L6 folds + concat-free + o-tail** | 90 remaining reduces folded into consumers (`wo_b`→hc_post, idx `wq_b`, Engram); 2 `cat`s → one kv buffer per kv-source group (`[layers × 256 ring | W_r compressed]`, per-layer index offsets; `T9B3-ATTN-DESIGN.md:154`); `wo_b`+reduce+hc_post in one launch (hc_post is column-local: per-tile ticket, no grid barrier) | data movement plus R2-order sums | Triton | bit-identical |
| **L7 cold overlap** (A2-3) | S9 stage, exposed | §4 | Triton gather on a forked branch | bit-identical (combine in id order) |
| **L8 indexer/compressor/Engram** | S4 ~250 → ~60 launches; S12 hash+gate | #14a/#14b score kernel, owner k path, compressor pooling + norm, #14s short-context fill (`T9B3-ATTN-DESIGN.md:155-158`); Engram hash + gate | Triton | #14s none; norms calibrated; score: medium-high past T/r > 512 |
| **L9 o-front** | S6: o RoPE+QDQ, `wo_a` bmm, act_quant → 1 | per group g: inverse RoPE on the 8 heads' last 64 dims, QDQ/32, then `wo_a` read as **fp8 + scales** (the loader's bf16 is fp8 × 2^k, exact in bf16), fp32 accumulate, bf16 → act_quant/32 in the epilogue (BLOCK_N a multiple of 32). Data: o group 8 KB; output 16 KB/row | Triton (GEMV class) | A3 vs cuBLAS bf16 bmm (order); −1.34 GB/step |
| **L10 q/kv tail** | S2/S3: pair + 2 norm_rows → 1; on the 8 index sources `wq_b ‖ idx wq_b` (same input qr: `T9B3-ATTN-DESIGN.md:151`) | pair GEMV, then per-output ticket: the last CTA sums partials in split order (`_t9b4_ld`), then q_norm+quant / kv_norm+RoPE+QDQ+ring in R2 orders | Triton | bit-identical |

**Tool choice.** Decision 8 puts CuTe DSL on tensor-core work. Every rung we own here is M ≤ 8 GEMV- or glue-class,
where the MMA is incidental and Triton's `tl.dot` GEMV is already accepted. The only tensor-core kernel with real
tiling is MegaMoE, which is borrowed. CuTe DSL / CUDA C++ enters at §3.

**Economics** at 6 rows, per step, from the §1 baselines:
- [M] marks the rate an estimate is anchored on.
- Each saving is the removed launches' measured µs minus the fused kernel's estimated phase cost (≈ 1–1.5 µs per
  internal dependency).

| rung | launches/step removed | ms saved, 6 rows [E] (central) | basis | eng. days | ms/day | risk |
|---|---|---|---|---|---|---|
| L1 | ~240 off the path | 0.55–0.91 (0.7) | 80 × 11.4 µs [M family means] hidden 60–100 % | 1.5 | 0.47 | SM contention: the branch competes with GEMVs that launch ≥ 2×SM programs [A] |
| L2 | ~400 | 1.0–1.2 (1.1) | S8 35–40 → ~10 µs | 3 | 0.37 | topk ties; softplus libdevice |
| L3 | ~360 | 1.6–2.6 (2.1) | S9+S10+S11 130–165 → stage + **81 µs MegaMoE (Pg61, 6 rows, measured on this box/model)** + 2 | 6 | 0.35 | ABI; in-place pool re-layout (SF ~12 GiB cannot be duplicated, `T8-T9A-DESIGN.md:208`); co-residency |
| L4 | 0 | 0.4 (2 heads) | 1.32 GB at cuBLAS's measured 6.3 TB/s | 1.5 | 0.27 | none beyond A3 |
| L5 | 0 | 0.35–1.0 (0.6) | ~700 owned edges × 0.5–1.4 µs (W1 §1.3, H100 evidence) | 3 | 0.20 | PDL in torch-captured graphs unproven here |
| L6 | ~200 | 0.35 | reduces 3.2 µs, cat 2.25 µs, hc_post 2.1 µs [M] | 2 | 0.18 | low |
| L7 | 0 | 0.3–0.6 (0.45) | in-domain 0.16–0.31 cold/layer × ~50 µs | 2.5 | 0.18 | §4 |
| L8 | ~230 | 0.5–0.6 (0.55) | ~2–2.5 µs per glue launch [M] | 3 | 0.18 | score ties at long context |
| L9 | ~80 | 0.3 | 19.2 → ~12 µs/layer | 2.5 | 0.12 | A3 |
| L10 | ~90 | 0.21 | norm_row 3.4 µs [M] | 2 | 0.10 | low |
| draft | — | 0.5 | the same kernels on 3 draft layers | (inside) | — | — |
| **total** | ~1,300 → **≈ 700 launches/step** (~16/layer, 13 on the critical path) | **≈ 7.3 (5.3–9.0)** | | **27** | | |

**Why ~16/layer, not 10.** Each remaining boundary is a real dependency:
- the norm needs the whole row;
- `wq_b` needs the whole normed qr;
- FlashMLA and MegaMoE are libraries;
- `wo_b` needs all 8,192 `wo_a` outputs;
- the router needs the whole normed row.

Collapsing those dependencies is §3's job.

## 3. The persistent per-layer kernel (decision 1 stretch)

**Shape.** After the ladder, each layer would be:

    P1 [attn front → q/kv GEMV → norms → wq_b (+idx wq_b)] → FlashMLA → P2 [o-front → wo_b → hc_post → ffn front →
    router → plan] → stage? → MegaMoE

- That is **4 launches per layer (+ stage when cold), ≈ 170 per step**.
- P1 has 2 grid barriers; P2 has 3. Everything else uses per-tile tickets.
- The next layer's P1 absorbs ffn hc_post: it is column-local, so it adds no barrier.

**Constraints.**
- **Co-residency.** Grid barriers need a cooperative grid of ≤ 152 × k CTAs. MegaMoE also grid-syncs, so persistent
  segments must be stream-ordered against it. Two co-resident grid-syncing kernels can deadlock
  (`T8-T9A-DESIGN.md:210`). That rules out running P2 concurrently with MegaMoE.
- **Per-phase working set.** Each phase's tiles (GEMV 32-K blocks ≈ 4–8 KB/stage) fit easily. Occupancy is not the
  limit. The barrier cost is ~0.6–1.15 µs on sm_120 [W1]; on GB300 it is unmeasured [A].

**Projection at 6 rows** [E]:

| segment | µs/layer | how |
|---|---|---|
| P1 | ≈ 18 | 49 MiB at ~5 TB/s, + 2 barriers, + tails |
| FlashMLA | 9–14 | [M] |
| P2 | ≈ 28 | 80 MiB, + 3 barriers, + topk/sinkhorn tails |
| stage | ≈ 3 | after L7 |
| MegaMoE | 81–87 | [M Pg61, scaled to 22.45 experts] |
| **layer total** | **≈ 140–150** | |

- Verify ≈ 6.4 ms; plus draft ≈ 0.9 → **≈ 7.3 ms/step (6.6–8.2)**.
- At 1 row ≈ 4.5–5.5 ms [E].
- Compared with ladder + PDL (≈ 9.0 central, §5) the gain is **≈ 1.0–1.7 ms**. That is larger than W1's "PDL = 97 % of
  persistent" (`W1-megakernels.md:36`).
- The reason: our dense phases stream at 0.3–3.5 TB/s [M, gemv-bench] because each is a short latency-bound launch. A
  persistent kernel streams the next phase's weights during the current phase's dependent tail. PDL can only prefetch
  across one edge.

**Reconciling with W1's "do not bother".** W1 sized the fused-layer pool on Pg61's step:
- 1,370 launches;
- PDL already on;
- the MoE hidden under a sidecar;
- result: −0.7…−1.4 ms realistic (`W1-megakernels.md:116`).

Our regime differs on every count:
- ≈ 2,000 launches, no PDL;
- Σ kernel durations equal the step (§0.3);
- ≈ 8 ms of the 10.52 ms 1-row step is non-bandwidth time (§0.2).

W1's "do not bother" was about **one kernel per forward**, and that still holds: FlashMLA, MegaMoE, the indexer's
top-k and Engram's host gather all cut the forward. The **per-layer** persistent kernel is a different question, and
here it is the only rung that projects under 7.36.

**Verdict.** Build the ladder first. It is bit-identical or A3, it has no deadlock class, and it produces the measured
boundary-gap histogram. Build P1/P2 (CUDA C++/CuTe DSL, decision 8) only if:
- the measured post-ladder 6-row step is within ~2 ms of 7.36; and
- the histogram shows ≥ 0.6 ms/step of inter-phase idle plus dense phases under 3.5 TB/s.

## 4. Recipe A self-fetch and the 6-row MoE

**Facts.**

| quantity | value | source |
|---|---|---|
| unique experts per layer at 6 rows | 22.45 | [M] |
| seed-map cold experts per layer, 6 rows, in domain | 0.16–0.31 | [E, `A2-RESIDENCY-DESIGN.md:92`] |
| same, under a traffic shift | 1.6 | [E, same table] |
| cold gather rate | 375 GB/s, ≈ 50 µs per expert | [M, `LEDGER.md:42`] |
| exposed cold time today | in-domain 0.3–0.6 ms/step; shift ≈ 3.2 ms/step | [E] |

Today the stage launch runs before DeepGEMM in stream order, so every cold byte is exposed.

**What self-fetch inside the fused kernel can add.**

1. **Overlap with the same layer's hot work.** This is defensible and is L7.
   - The cold set is known only after the router. The only work left to hide it under is the hot MoE.
   - On the contiguous path: hot groups' GEMM first; the cold gather on a forked branch; then a second DeepGEMM call on
     the H+j groups. The id-order combine makes the split numerically invisible (`T8-T9A-DESIGN.md:139`).
   - Hot work at 6 rows is 80–120 µs, against 8–80 µs of cold copy, so in-domain cold should be essentially hidden.
   - Under MegaMoE, staging must precede the single launch (`T8-T9A-DESIGN.md:149`), so cold is exposed again unless
     MegaMoE's TMA reads Grace directly. That needs a VMM-stitched VA (E4, `T8-T9A-DESIGN.md:148, 201`). It is the
     literal thesis and is unmeasured. Even then, the library's tile order decides whether the slow cold tiles start
     first.
2. **Removing the stage launch.** ≈ 9 µs/layer, 0.37 ms/step [M mean], when it is folded into the MoE prologue.
3. **C2C as a second memory channel.** C2C (390 GB/s) runs concurrently with HBM [M, overlap probe: HBM copy 0.341 →
   0.357 ms under H2D].
   - Deliberately reading ~1.5–2 experts/layer from Grace in parallel adds at most ~6 % to the expert stream: ≤ 0.2–0.3
     ms/step [E].
   - It only works if the fetch overlaps fully.
4. **Not defensible.**
   - Prefetching layer L+1's cold experts during L: L+1's routing depends on L's output.
   - Using draft-row routing: the draft's MoE is a different expert set (`model.py:142-149`; 128 routed, top-3 per
     `M1-DSPARK-DESIGN.md:38`), so it predicts nothing about the verify layers.
   - Cross-step locality is real (6 → 3.74 distinct per token over 6 tokens [M]), but residency (A2-2) captures it,
     not in-step prefetch.
   - Pre-gating (gate_{L+1}(h_L)) stays an **offline** hit-rate measurement on our traces (A2-3). Build it only if
     exposed cold is > 0.3 ms after (1).

**The deciding measurement** (A2 §4, unchanged in substance). Measure exposed cold =
t_step(adaptive + overlap) − t_step(timing control, every cold pointer redirected to one HBM slot) at 6 rows, on the
fixture plus the shift corpus.
- **Pass**: ≤ 0.3 ms/step.
- **Kill**: > 1.0 ms/step after overlap.

Add the charter's "everything in Grace" bound and a sweep of k forced-cold experts/layer (k = 0…3). If t_step stays
flat to k ≈ 1.5, recipe A adds bandwidth; if it rises at 50 µs per expert, A is only a capacity trick.

## 5. Cost model to 5.0 ms at 6 rows

### 5.1 Floors

These come first, because they decide the verdict. Bytes per 6-row step, using every legal byte cut (head read as
bf16, `wo_a` read as fp8, gate read as bf16) [E, shapes]:

| item | GB |
|---|---|
| verify | 25.4 |
| — of which routed experts (22.45 × 18.80 MB × 40) | 16.9 |
| draft (3 layers, ≤ 15 experts each, shared bf16 head, 5 Markov GEMVs over a 132 MB fp32 head) | 3.4 |
| **step** | **28.8** |

The floors that follow:

| floor | ms | what it means |
|---|---|---|
| 28.8 GB at the measured HBM copy rate, 6.3 TB/s | **4.6** | 5.0 needs a **5.76 TB/s average over the whole step**, i.e. 92 % of the copy rate, with ≈ 0.4 ms left for every latency-bound piece |
| at the best rate each byte class has shown on this box (experts 5.2 TB/s = Pg61 MegaMoE; dense 3.5 TB/s = our best GEMV; heads 6.3 TB/s = cuBLAS) [M rates] | **6.3** | zero latency |
| at 8 TB/s nominal | 3.6 | |

7.36 needs only 3.9 TB/s, which Pg61 already sustains with its sidecar.

### 5.2 Rung-by-rung projection

Columns:
- **bottom-up**: verify 14.3 (13.2–15.4, §1) + draft 2.0 (`M1-DSPARK-DESIGN.md:233`) = 16.3.
- **2.0×**: the handoff's 21 + 2.0 = 23.

All values are [E].

| after | bottom-up, ms/step | 2.0× start, ms/step | ms/token at 2.6–2.9 tok/step (bottom-up) |
|---|---|---|---|
| start (M1 fast path, row-generic glue) | 16.3 | 23.0 | 5.6–6.3 |
| L1 mixes branch | 15.6 | 22.3 | |
| L2 router + plan | 14.5 | 21.2 | |
| L3 MegaMoE | 12.4 | 19.1 | |
| L4 bf16 head ×2 | 12.0 | 18.7 | |
| L5 PDL | 11.4 | 18.1 | |
| L6 folds / concat-free | 11.1 | 17.8 | |
| L7 cold overlap | 10.6 | 17.3 | |
| L8 indexer / Engram | 10.1 | 16.8 | |
| L9 o-front | 9.8 | 16.5 | |
| L10 q/kv tail + draft | **9.0** (range 8.0–11.7) | **15.7** | 3.1–3.5 |
| §3 persistent P1/P2 | **7.3** (6.6–8.2) | 14.0 | 2.5–2.8 |
| Pg61, measured | 7.36 release / 7.75 W0 | | 2.8 |
| bar | 5.0 | | 1.7–1.9 |

**Crossings.**
- **7.36: the ladder alone does not cross it** at its central estimate. The best corner is ≈ 8.0. The persistent
  stretch crosses only in its optimistic half.
- **5.0: no path crosses it.** It sits 0.4 ms above the bandwidth floor, and 1.3 ms *below* the achieved-rate floor.
- A cross-check from the other direction: the bottom-up per-layer sum after the ladder is 177 µs × 40 + 0.9 = 8.0
  verify, plus draft 1.0–1.6, minus L5 → 8.3–9.0. That agrees with the table.

### 5.3 The fp32 head

The head is 0.42 ms [M], 8 % of the bar, and it is one read regardless of rows. It is read twice per step, once by the
verify and once by the draft (`model.py:1216`).

| option | effect | class |
|---|---|---|
| (a) cuBLAS fp32 | today | not row-invariant at M=6 [M] |
| (b) own fixed-order GEMV over the fp32 copy | 0 ms saved; makes V0 strict | A3 |
| **(c) own GEMV over the bf16 bytes, fp32 accumulate** | −0.2 ms per read, −0.4 per step; +2.47 GiB HBM | **A3**: the products are identical because the widening is exact; only the order changes. **Recommended** |
| (d) bf16 accumulate or bf16 logits | — | **A1**: changes the model's fp32 logits (`model.py:1007-1014`). Reject |
| (e) vocab pruning or a partial head | — | A1. Reject |

**Tokens/step.** The charter grades ms/step. At 2.6–2.9 accepted tokens/step (a prior: our greedy acceptance is
unmeasured, `M1-DSPARK-DESIGN.md:87-91`), 9.0 ms is 3.1–3.5 ms/token and 7.3 ms is 2.5–2.8, against Pg61's 2.8
[M W0]. The persistent stretch is what makes us competitive per token.

## 6. Plan

Prerequisites, all M1, already running or queued: fix-2 (dg-inplace at 6 rows), row-generic fused glue for the R1/R2
clusters, T11 draft graph, T12 loop. Without the row-generic glue, every 6-row number above is fiction.

| task | content | box window | gate |
|---|---|---|---|
| **M2-T13 measure** (first) | static-eager nsys anatomy at 10.52 and of the 6-row fast path; a per-segment in-graph timer (marker kernels stamping `%globaltimer`, so the boundary-gap histogram W1 calls the primary instrument); probes: fork concurrency (L1 branch + shared branch, timing only), PDL edges under torch capture with Triton `gdc_*`, libdevice softplus == torch, DeepGEMM contiguous time vs unique experts U = 6…36 | 60 min | §1's table refilled with [M]; each rung's estimate kept or cut |
| M2-T14 L1 (+ shared-expert fork as a stopgap until L3) | graph fork/join in `StaticStep.capture` | 20 min | output byte-identical; G/L 8/8; ≥ 0.5 ms at 1 row, else drop |
| M2-T15 L2 | router + plan kernel, 1 and 6 rows | 30 min | route bitwise on 10⁵ recorded router inputs (ids and weights); byte-identical output at 1 row (or A3 if gate mm uncalibrated) |
| M2-T16 L3 | `--moe-impl dg-mega`: E2 ABI read, in-place pool transform, H+S uniform, stub group | 2 × 45 min | per-layer mismatch vs dg-inplace reported; self-replay; **≤ 90 µs/layer at 6 rows**; A3 drift to James |
| M2-T17 L4 | bf16 head GEMV, verify + draft | 20 min | A3; V0 strict with M=1 GEMV pinned |
| M2-T18 L5 + L6 | PDL on owned kernels; folds, concat-free, o-tail | 30 min | bit-identical; L5 ≥ 0.3 ms or reverted |
| M2-T19 L7 (= A2-3) | cold overlap, timing control, k-forced-cold sweep, "everything in Grace" bound, offline pre-gate hit rate | 90 min | §4 pass/kill |
| M2-T20 L8–L10 | indexer/compressor/Engram, o-front, q/kv tail | 45 min | per-class gates as §2.1 |
| M2-T21 desk | persistent P1/P2 go/no-go from T13–T20 measurements | — | §3 rule |
| M2-T22 (conditional) | P1/P2 in CUDA C++/CuTe DSL | several | ≥ 1 ms below ladder at 6 rows |

**Order.** T13 → T14 → T15 → T16 → T17 → T18 → T19 → T20 → T21. That is ~27 engineering days of coding through T20
[E]; the box is serialized behind M1.

**Kill criterion for the charter, by 2026-12-04** (week-4 flag, decision 17):
- Once M1-T12 runs and T13–T17 have landed (est. ~Nov 28), measure the 6-row step (verify + draft).
- Compute **measured − (upper savings of the remaining rungs + the persistent stretch's upper 1.7 ms)**.
- If that exceeds **7.36**, recipe A cannot beat production on one GB300. Publish the result, and propose recipe B
  (sidecar bandwidth) as the path to the bar.
- Independently, **ask James now to restate the 5.0 bar.** On byte arithmetic it needs 92 % of the measured HBM copy
  rate across the entire step (§5.1). Keep it as a recipe-B stretch, and grade M2 on "beat 7.36 with recipe A".
- Report achieved TB/s (bytes/step ÷ ms/step) next to every ms/step from now on.

M2-DESIGN: projected 6-row step ≈ 9.0 ms after the ladder (8.0–11.7 from the bottom-up start; ≈ 15.7 if the measured 2.0× holds), ≈ 7.3 ms with the per-layer persistent stretch — 7.36 is crossed only by the stretch's optimistic half, and 5.0 is not reachable with this design (28.8 GB/step needs 5.76 TB/s ≈ 92 % of the measured 6.3 TB/s HBM rate; achieved-rate floor 6.3 ms).
