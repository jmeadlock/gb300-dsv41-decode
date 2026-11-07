# W3 — GEMV and head: the bandwidth-bound residual (round 2, 2026-11-07)

Status: complete (2026-11-07, W3, research only; no box access used).

## Verdict

The GEMV residual is real but it is a latency problem wearing a bandwidth costume. Our 290 MXFP8 GEMVs move 5.5 GB/step (0.69-0.79 ms at 7-8 TB/s) and take 3.43 ms because ~208 of them are 2.6-12 MB weights paying an ~8 us fixed cost each (measured fit), caused by a 32-byte-per-row-per-K-block load pattern that keeps too few bytes in flight plus a second split-K launch. Published single-launch kernels on B200/Rubin for exactly this byte class (FlashInfer TGV, TRT-LLM/FlashInfer tinygemm2; TRT-LLM #20634 on DeepSeek-V4.1's own wq_a/wkv) run 2.5-5 us and are flat to M=8-16. A rewrite with K-contiguous 256-512 B row chunks, a deep pipeline, an in-kernel deterministic (last-CTA, fixed split order) reduce, M-independent split plans and 2-3 same-input pointer groups models at ~1.4 ms (-2.0 ms at 6 rows, -1.2 to -1.6 ms at 1 row after T9b-4), bit-identical to today's GEMV per shape when the 32-block partial is formed identically, otherwise the existing 1-ulp class. The head should move to bf16 STORAGE (the checkpoint is bf16 and the reference only widens it; both operands are exact bf16 values, every product is exact in fp32) with our own fixed-order fp32-FMA kernel: 0.42 -> ~0.21 ms at 1-8 rows, row-invariant (removes the census's last non-invariant kernel), 1-ulp class vs the cuBLAS reference, and ~0.5-0.8 ms saved per DSpark cycle once the drafter's head and Markov heads are counted. vLLM's default V4.1 head produces bf16 logits (A1 for us -- do not copy; it may explain part of production's parity gap, unverified). Together G1+G2 are worth ~2.2-2.4 ms of a ~21 ms 6-row step: necessary, nowhere near sufficient for 5.0 ms.

## 0. Local facts read first (repo, read-only; all "measured" numbers are this project's receipts)

Per-shape GEMV, tuned, graph-timed on rotating weights (docs/gemv-bench-r2-20261106.json, 2026-11-07 03:17 UTC, GB300 152 SMs, Triton 3.7.1). us includes the split-K reduce launch when the plan splits. MB = fp8 weight bytes (N*K).

| N x K | calls/step | MB | us M=1 | GB/s M=1 | us M=6 | split_k (M=1) |
|---|---|---|---|---|---|---|
| 512 x 5120 (wkv) | 40 | 2.6 | 8.81 | 298 | 8.48 | 19 |
| 1280 x 5120 (wq_a) | 40 | 6.6 | 9.13 | 720 | 9.04 | 8 |
| 4096 x 1280 (indexer wq_b) | 8 | 5.2 | 6.35 | 828 | 6.68 | 5 |
| 2304 x 5120 (shared w1, w3) | 80 | 11.8 | 10.28 | 1149 | 10.40 | 9 |
| 5120 x 2304 (shared w2) | 40 | 11.8 | 9.56 | 1237 | 9.37 | 4 |
| 5120 x 8192 (wo_b) | 40 | 41.9 | 19.40 | 2165 | 20.06 | 8 |
| 32768 x 1280 (wq_b) | 40 | 41.9 | 14.84 | 2833 | 15.48 | 1 |
| 25600 x 6144 (engram wkv) | 2 | 157 | 44.95 | 3504 | 45.06 | 4 |
| per step (bench sum) | 290 | ~5,490 | 3,433 us | ~1.6 TB/s | | |

Modeled from that table (least-squares by eye, my arithmetic): on the four small shapes us ~ 8.4 + 0.16 us/MB, i.e. an ~8 us fixed cost per call and ~6 TB/s marginal; on the big shapes ~0.22 us/MB marginal (~4.5 TB/s). The step's GEMV problem is therefore ~200 of 290 calls paying ~8 us of fixed cost for 3-12 MB of bytes that need 0.3-1.5 us at 8 TB/s. M=6 costs the same as M=1 (measured, within 0.7 us on every shape): the kernel is row-free up to 8 rows, as expected for a bandwidth/latency-bound GEMV.

Kernel structure that explains the fixed cost (runtime/gemv_mxfp8.py:589-659, read): every program loads its weight tile as [BLOCK_N, 32] fp8 per 32-wide K block, i.e. ONE 32-byte sector per weight row per K block, KU=2 blocks unrolled, num_stages 3-4; split-K up to 19 ways with a SECOND launch (t9b_splitk_reduce, 250 launches/step, 3.2 us each, 0.80 ms/step measured in step-anatomy-r2). Bytes in flight per program per iteration are ~BLOCK_N*32*KU = 2-8 KB. By Little's law (section 2) an HBM3e part needs several MB in flight GPU-wide to reach peak; we keep roughly an order of magnitude less on the small shapes, so latency, not bandwidth, sets the time. The split-K exists to create programs, not to create bytes in flight; it then costs a second launch.

The head (reference/deepseek-inference/model.py:999-1018, read):
  * "bf16 in the checkpoint, kept as fp32 here so the logits come out in fp32 directly" -- weight [129280, 5120] widened to fp32 = 2.648 GB; logits = F.linear(x.float(), W_fp32).
  * x reaching the head is bf16: hc_pre returns y.to(x.dtype) (model.py:959-962) and RMSNorm returns its input dtype (model.py:288-293); generate.py:118 sets the default dtype to bf16. So x.float() and W.float() are both exactly bf16 values.
  * Consequence (arithmetic, not measured): every product x_k * w_k of two bf16 values has <= 16 significant bits and is EXACT in fp32. The fp32 head's only rounding is in the 5,120-term fp32 accumulation, whose order cuBLAS chooses (and changes with M -- the census's one non-row-invariant kernel, M1-T10-RESULT-20261107.md). Storing the head as the checkpoint's bf16 and widening in registers is therefore a LAYOUT change ("values may not change" holds: the fp32 operand values are identical bit-for-bit). It is not requantization and not A1.
  * vLLM 0.31.0 (reference/vllm-0.31.0/deepseek_v41/nvidia/model.py:1338-1345, 1374-1379, read): lm_head is a plain ParallelLMHead with no quant_config and logits go through LogitsProcessor. Upstream ParallelLMHead/UnquantizedEmbeddingMethod.apply is F.linear(x, W) in the model dtype (bf16) -- so production computes bf16 x bf16 with cuBLAS's fp32 accumulate and ROUNDS THE LOGITS TO bf16 before the sampler upcasts them (upstream code citation in section 1.5; the vendored copy does not include layers/). That bf16 rounding of logits (8 significand bits: ties within ~0.4 % of a logit's magnitude collapse) is an A1-class deviation from the reference's fp32 logits for us; we must not copy it. It is also a plausible contributor to the 6/8 non-identical production-vs-reference prompts in FIRST-LIGHT (unverified; it is a hypothesis W4/the A3 amendment can test cheaply).
  * The head is also called by the drafter (model.py:1141-1160): DSparkBlock.forward_head runs the SAME 2.65 GB head on block_size = 5 rows, plus 5 sequential DSparkMarkovHead calls (fp32 [129280, 256] = 132 MB each, 0.66 GB total). Per DSpark cycle the head family therefore reads ~2.65 (verify, M=6) + 2.65 (draft, M=5) + 0.66 = ~5.96 GB in fp32 (modeled) -- ~0.75 ms at the cuBLAS rate we measured -- vs ~2.98 GB / ~0.4 ms if both are read as bf16. W2 should price this; it is not in the 21 ms projection.

## 1. Findings with citations (web, fetched 2026-11-07)

### 1.1 What state-of-the-art skinny kernels reach on Blackwell (claimed-by-source unless noted)

* TRT-LLM issue #20634 "DeepSeek-V4.1 CSA2: BF16 wq_a / wkv projections at decode reach 0.9-2 TB/s on Vera Rubin; tinygemm2 is 1.6-3x faster" (opened 2026-10-19) -- OUR model, OUR shapes, but bf16 weights on Rubin (sm_107, not our sm_103): https://github.com/NVIDIA/TensorRT-LLM/issues/20634
  - CUDA-graph replay, weights rotated past L2, M=1 ("M = 4 and 6 behave the same"):
    wq_a 1280x5120: cuBLAS nvjet 6.64 us, trtllm::tinygemm2 3.85 us; wkv 512x5120: 5.86 vs 3.67 us; [wq_a; wkv] row-concatenated 1792x5120: 6.87 vs 4.67 us (the text says 4.13 us "for both" -- the issue is internally inconsistent by 0.5 us; I use the table).
  - E2E TP4 decode step 5.97 -> 5.70 ms from this change alone. Note the cuBLAS kernels it beats are the same nvjet 16x64/32x64 family our step-anatomy shows for the mHC/gate mm.
  - Caveat they raise: tinygemm2 triggers PDL launch-completion before its output stores (TRT-LLM PRs #15729/#15790). Anything we borrow from it must be launched without PDL or with the trigger moved after the stores.
* FlashInfer PR #3281 (CuTeDSL "cute_ext" TGV backend default for mm_bf16; opened 2026-05-10, merged): B200, bf16, L2-cold (~4x L2 rotating), CUDA-graph timing, us/kernel, TGV with PDL vs cuBLAS F.linear: (1,1280,2880) 3.76 vs 5.91 (5.02 without PDL); (1,2880,1024) 2.53 vs 5.01; (1,3072,1536) 2.96 vs 6.41; (1,7168,2048) 6.00 vs 7.44; "peaking near 5.3 TB/s DRAM BW (~66% of HBM3e)". M=1..8 flat (e.g. 1280x2880: 3.76 / 3.81 / 3.86 / 3.87 us at M=1/2/4/8). https://github.com/flashinfer-ai/flashinfer/pull/3281
* FlashInfer tgv_gemm_sm100 now also takes block-scaled operands: "MX scaling accepts FP4 or FP8 operands ... with FP8 E8M0 scales and K divisible by 128"; "Block-scaled inputs require M <= 8 and flattened 128x4 scale-factor layouts"; output bf16/fp16; SM100/SM103/SM107. https://docs.flashinfer.ai/generated/flashinfer.gemm.tgv_gemm_sm100.html (docs, fetched 2026-11-07; the release that added MX to TGV not pinned -- unverified whether it is in our image's flashinfer 0.7.0.post1). Also flashinfer.mm_fp8 backend "cutedsl_low_latency": "requires SM100/SM103, m <= 8, and k divisible by 128" (per-tensor scales, not our 32x32 blocks): https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp8.html
* FlashInfer PR #4274 (tinygemm2_sm100, CAKE-generated, opened 2026-07-30, merged): "bit-identical outputs" vs the TRT-LLM tinygemm2 port, B200 torch.profiler kernel-time medians (1,128,720) 2.24 us, (16,1024,1024) 2.50 us; "Small decode shapes are launch-bound"; verified on GB300 sm_103a with zero greedy token flips on gpt-oss. tinygemm_bf16 = TMA + mma.sync m16n8k16, BF16 in/out, FP32 accumulate, 384 threads (4 compute + 8 DMA warps), 16 pipeline stages, "ideally 1-8 rows". https://github.com/flashinfer-ai/flashinfer/pull/4274 , https://docs.flashinfer.ai/generated/flashinfer.gemm.tinygemm_bf16.html
* SGLang PR #27397 (JIT dsv3 fused-A GEMM = TRT-LLM dsv3FusedAGemm, mma.sync, weights prefetched to SMEM; opened 2026-06-05): SM100 bf16, K=6144 N=2112 (26 MB) M=1: 8.13 us CUDA (3.2 TB/s), torch 9.98; K=7168 N=2112 (30 MB): 9.18 us (3.3 TB/s), torch 10.91; flat to M=16. https://github.com/sgl-project/sglang/pull/27397
* TRT-LLM DeepSeek-R1 B200 min-latency blog: Fuse_A_GEMM "concatenates the weights for WDQ, WDKV, and WKR to reduce kernel launch overhead"; UQ_QR_GEMM concatenates WUQ and WQR "to reduce kernel launch overhead"; custom Fuse_A GEMM "prefetches the majority of its weights into shared memory (enabled by PDL ...)", wins for num_tokens < 16 (numbers only in a chart image -- not transcribed, unverified). https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html
* Hazy Research "Look Ma, No Bubbles!" (2025-05-27): H100 dummy kernel launch cost ~2.1 us on a stream, ~1.3 us in a CUDA graph; vLLM/SGLang "at most 50% of available GPU bandwidth" on Llama-1B bs1; megakernel 78% of H100 bandwidth; B200 forward 600-680 us, of which "250 microseconds are spent storing activations, awaiting consistency, and loading them". https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
* whirl-llm kernel guide (AMD RDNA, not NVIDIA; included because it is the cleanest published bit-identical grouped-GEMV write-up): "Launch head/tail costs about 2.8 us"; same-input GEMVs (q/k/v, gate/up, shared-expert gate/up, MTP copies) launched once with up to three (W, y, row_bytes, nrows) segments, block index rebased per segment, "Each row's kernel and float expression are unchanged -> bit-identical"; launches/token 740 -> 596, decode 25.66 -> 25.15 ms (-2.0 %), verify n=5 -2.0 %; codegen trap: pointer-select grew VGPRs 30-90 % and spilled -> separate "twin" entry points. https://github.com/tsaipifong/whirl-llm/blob/main/docs/guide/en/kernels.md
* PyTorch "Accelerating Llama3 FP8 Inference with Triton Kernels" (TK-GEMM, H100): SplitK for small M, "up to 1.94x over base Triton, 1.87x over cuBLAS FP8"; its split-K partials are summed "using an atomic reduction" -- order-nondeterministic, so not usable under our bit-identical policy. https://pytorch.org/blog/accelerating-llama3/
* Memory latency (for the Little's-law argument): Chips and Cheese B200 (2025): L2 latency to the near partition ~150 ns, far partition much higher; "B200 thus appears to have higher VRAM latency than the MI300X, as well as the older H100 and A100" (the VRAM number itself is only in a chart -- unverified value). arXiv 2512.02189 (Blackwell microbenchmarks) quotes Hopper global-memory miss latency ~1000 cycles. https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut , https://ar5iv.labs.arxiv.org/html/2512.02189
  - Modeled: at ~8 TB/s and ~0.7-1.0 us loaded DRAM latency, Little's law needs ~5.6-8 MB in flight GPU-wide (37-53 KB per SM at 152 SMs) to sit at peak. A 6.6 MB weight (wq_a fp8) is about ONE latency-window's worth of bytes: no kernel can stream it at 8 TB/s; the floor is ~(launch ramp) + (1 DRAM latency) + bytes/BW ~ 1.3 + 0.8 + 0.8 ~ 3 us. That is consistent with TGV's 2.5-3.8 us and tinygemm2's 3.7-3.9 us for 5-13 MB weights.

Reading of 1.1 (mine): for weights of 3-13 MB, the published single-launch state of the art on B200/Rubin is ~2.5-5 us per projection (with PDL at the low end), flat in M up to 8-16. We are at 8.5-10.3 us for the same byte class. The gap is ~2-2.5x per call and it is structural (one 32 B sector per row per K-block, split-K second launch), not tuning. For 40-160 MB weights published kernels reach ~5 TB/s (TGV peak 5.3 TB/s = 66 % of B200 HBM; our own cuBLAS head gemv does 2.65 GB in 420 us = 6.3 TB/s on GB300, measured), vs our 2.2-3.5 TB/s.

### 1.2 One launch per layer for several projections ("grouped GEMV")

Known pattern, three published forms:
  (a) concatenate weights that share an input into one matrix (TRT-LLM Fuse_A_GEMM WDQ||WDKV||WKR, UQ||QR; TRT-LLM #20634 [wq_a; wkv]) -- a loader/layout change; for us bit-identical only if each output column's K-order chain is unchanged (it is, if split partitions per half are kept, which is exactly what T9b-4's gemv_pair already does with separate pointers);
  (b) pointer-segment table in one kernel (whirl-llm GvArgs; Triton's official grouped-GEMM tutorial launches a fixed number of persistent CTAs over a device-side array of (A, B, C, M, N, K) pointers https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html ; CUTLASS ptr-array / grouped GEMM incl. Blackwell CuTe-DSL grouped_gemm example https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py and grouped schedulers https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html );
  (c) a persistent megakernel (Hazy, Mirage MPK https://github.com/mirage-project/mirage) -- that is M2 / fusion-ladder territory, not repeated here.
Published gain sizes for (a)/(b): TRT-LLM #20634 -8.4 us/layer for wq_a+wkv (Rubin); whirl-llm -2 % of step from -144 launches/token. NO published number for a grouped GEMV over projections that depend on each other in sequence (wq_a -> q_norm -> wq_b): those cannot share a launch without a grid-wide dependency, i.e. they are fusion, not grouping. In our layer only these sets are independent (same input): {wq_a, wkv} (+ indexer wk if it reads the same x -- check), {shared w1, w3}, {indexer wq_b, ...?}. So "all ~7 projections of a layer in one launch" is not a grouping problem; it is either a persistent kernel with in-kernel dependencies (M2) or 3-4 grouped launches per layer.

### 1.3 Deterministic split-K in ONE launch

Our 250 t9b_splitk_reduce launches (0.80 ms/step, measured) exist only to sum split partials in fixed order. The standard single-launch alternative is "serial split-K" / stream-K fix-up: each split CTA writes its fp32 partial, increments a per-tile arrival counter (atomic on an int, not on the data), and the LAST arriving CTA sums the partials in split order 0..NSPLIT-1 from +0.0 and stores bf16. The fp32 adds and their order are exactly t9b_splitk_reduce's, so the output is bit-identical to today's two-launch result by construction. CUTLASS ships this as split-K-serial / stream-K "fixup" with a deterministic reduction mode (Stream-K paper, Osama et al. 2023, https://arxiv.org/abs/2301.03598 ; Colfax CUTLASS tutorial: in stream-K's default mode each CTA waits for its predecessor and reduces into the workspace in order, with ReductionMode::Nondeterministic as the opt-in atomic alternative, https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/). Triton: implementable with tl.atomic_add on an int32 counter + tl.debug_barrier / memory fence semantics (Triton's atomics default to acq_rel sem since 2.x -- verify on 3.7.1). Cost model: the last CTA's extra read of NSPLIT x BLOCK_N fp32 partials from L2 (~150 ns latency class) instead of a ~3.2 us launch.

### 1.4 The head: who keeps it fp32, and what bf16 storage means

* Reference (DeepSeek's own inference code, vendored): bf16 checkpoint weight widened to fp32, F.linear in fp32 -> fp32 logits (section 0). The same ParallelHead class is used for the DSpark Markov head (fp32 [129280, 256]).
* vLLM default: lm_head in the model dtype, logits produced by LogitsProcessor._apply_head -> lm_head.quant_method.apply(...) in bf16 when head_dtype is unset ("Dtype of the lm_head projection. Defaults to the model dtype; an fp32 head (via --hf-overrides '{"head_dtype": "float32"}') is required for RL training-inference consistency"). With head_dtype=float32 on CUDA it runs torch.mm(flat, lm_head.weight.t(), out_dtype=torch.float32): "native bf16xbf16->fp32 ... avoids materializing a per-step fp32 copy of the lm_head weight". vLLM PR #48390, merged 2026-07-13: https://github.com/vllm-project/vllm/pull/48390 ; code: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/logits_processor/ . Our vendored V4.1 model file passes no head_dtype (nvidia/model.py:1338-1345) -- so production Pg61 (vLLM 0.31.0 + Garner patches) computes BF16 logits unless Pg61's launcher sets the override (unverified: the launcher is on the box; not inspected per the no-ssh rule).
* SGLang: PR #45631 (merged 2026-10-22) "Write the bf16 lm_head's fp32 logits straight into the logits buffer" -- i.e. SGLang has a bf16-weight / fp32-output head path, validated on GB300, measured on DeepSeek-V4.1-Flash DEP8 decode (step 22.11 -> 21.94 ms at 128 req/rank); isolated 1024 x 129280 x 4096 logits 0.890 -> 0.589 ms. Whether that path is SGLang's default for V4.1 or only for RL (test_fp32_lm_head) is unverified. https://github.com/sgl-project/sglang/pull/45631
* Numerics of the bf16-storage options (my analysis + one citation):
  - Storing the head as bf16 and widening in registers is value-identical (both operands are exact bf16 values; section 0). Every product is exact in fp32. Only the 5,120-term accumulation order can differ.
  - CUDA-core fp32 FMA in a fixed order (our own kernel): IEEE round-to-nearest per add; the result differs from cuBLAS's fp32 order by a few fp32 ulps at most on rare outputs -- "1-ulp class" in our taxonomy, and ROW-INVARIANT by construction (the order is ours, not a function of M). This also removes the census's last non-row-invariant kernel.
  - Tensor-core bf16 MMA with fp32 accumulate (torch.mm out_dtype=fp32, cuBLAS/nvjet, TGV): products exact, but tensor-core internal accumulation is not sequential IEEE fp32 -- Fasi, Higham, Mikaitis, Pranesh, "Numerical behavior of NVIDIA tensor cores" (PeerJ CS 2021, V100/T4/A100) found truncation and no normalization inside the block FMA: https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/ . Blackwell tcgen05 behaviour: unverified. Treat as A3 (drift-graded), and cuBLAS split-K by M makes it non-row-invariant unless pinned.
  - BF16 LOGITS (vLLM default): A1, rejected.

### 1.5 cuBLASLt small-M FP8 / MXFP8, Marlin-class

* cuBLASLt does block-scaled MXFP8 via CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 (1x32 scales on both operands; CUDA 13.4 adds experimental VEC32_MN_K4_UE8M0 layouts): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html , https://github.com/nvidia/cudalibrarysamples/issues/310 . No published M=1-8 TB/s for it found (unverified). The only published small-N cuBLAS decode numbers for our shapes are TRT-LLM #20634's nvjet bf16 0.9-2 TB/s -- no better than our Triton GEMV. cuBLAS heuristics also pick per-M tiles/split (our census already shows that for the fp32 head). Not a candidate.
* Marlin (Frantar et al., arXiv 2408.11743, 2024; not re-fetched today) is W4A16 and Ampere/Hopper-focused; the transferable lessons are its lock-ordered (deterministic) global reduce and streaming weight loads with evict-first hints -- the same two things 1.3 and section 3 prescribe. Not a kernel we can borrow for e4m3 x e4m3 with 32x32 e8m0 scales.

Correction/addition to 1.2 (checked in reference/deepseek-inference/model.py:514, 551, 736, 772): the indexer's wq_b (4096 x 1280) reads the SAME qr as attention's wq_b (32768 x 1280) on the 8 indexer layers -> a third same-input group {attn.wq_b, indexer.wq_b} (8 launches/step). The indexer's weights_proj (bf16, dim -> n_heads) reads the same x as wq_a/wkv but is bf16, not MXFP8 -- it cannot join an fp8 group without a mixed-dtype segment (whirl-llm's "mixed-type groups still launch separately").

## 2. Numbers table

Measured = this project's receipts (docs/). Claimed = source's own number. Modeled = my arithmetic, assumptions stated.

| item | value | class | source |
|---|---|---|---|
| fp8 GEMV bytes per step (290 calls, 9 shapes) | 5,495 MB | measured (shape x count from bench) | gemv-bench-r2 |
| GEMV sum per step, bench, M=1 / M=6 | 3,433 / 3,473 us | measured | gemv-bench-r2 |
| GEMV + reduce in the 13.9 ms step | 3.58 ms (2.78 + 0.80), 290 + 250 launches | measured | step-anatomy-r2 |
| GEMV fixed cost per call, small shapes (fit) | ~8.4 us + 0.16 us/MB | modeled from measured | section 0 |
| fit check: 8.4 us + 6 TB/s over all 290 calls | 3,352 us vs 3,433 measured | modeled | |
| head fp32 cuBLAS gemv2T, M=1 | 420 us, 2.648 GB -> 6.3 TB/s | measured | step-anatomy-r2 |
| head fp32 at M=6 | unmeasured (bytes floor 0.42 ms; modeled 0.42-0.6 ms) | modeled | |
| bf16 TGV (B200, L2-cold, graph) 1280x2880 / 2880x1024 / 7168x2048, M=1 | 3.76 / 2.53 / 6.00 us with PDL; 5.02 / 3.88 / 7.33 without | claimed | FlashInfer #3281 |
| TGV peak DRAM BW | ~5.3 TB/s (66 % of B200 HBM3e) | claimed | FlashInfer #3281 |
| tinygemm2 bf16 (Rubin) wq_a 1280x5120 / wkv 512x5120 / concat 1792x5120, M=1 | 3.85 / 3.67 / 4.67 us (cuBLAS 6.64 / 5.86 / 6.87) | claimed | TRT-LLM #20634 |
| fused-A bf16 (SM100) 2112x6144 / 2112x7168, M=1 | 8.13 / 9.18 us (~3.2-3.3 TB/s) | claimed | SGLang #27397 |
| tinygemm2_sm100 tiny shape kernel time | 2.24 us (1,128,720); "launch-bound" | claimed | FlashInfer #4274 |
| CUDA-graph launch cost, dummy kernel, H100 | ~1.3 us (2.1 on a stream) | claimed | Hazy no-bubbles |
| grouped same-input GEMV launches (RDNA) | 740 -> 596 launches/token, -2.0 % step | claimed | whirl-llm guide |
| B200 near-L2 latency | ~150 ns | claimed | Chips and Cheese |
| O1 (single-launch, t0 = 3.0 us, 7 TB/s marginal), 290 calls | 1,655 us | modeled | |
| O1 + same-input groups ({wq_a,wkv},{w1,w3}; 210 launches) | 1,415 us (+ {wq_b, idx wq_b}: ~1,395) | modeled | |
| O1 + groups + PDL (t0 = 2.0 us) | ~1,205 us | modeled (PDL is W1's lever) | |
| pure-bytes floor, 5,495 MB at 7 / 8 TB/s | 785 / 687 us | modeled | |
| head bf16 storage, 1.324 GB at 6.3 TB/s | 0.21 ms (M = 1..8) | modeled | |
| head fp32 FMA work at M = 6 / 8 | 3.97 / 5.30 G FMA -> 0.11-0.13 / 0.14-0.18 ms at 37.5-30 T FMA/s (75-60 TFLOPS fp32, spec-sheet range) | modeled; FP32 spec claimed by gpucompat.com/spheron (inconsistent sources) | |
| Markov head fp32 132 MB vs bf16 66 MB, per call | ~21 vs ~10.5 us + launch | modeled | |
| head family per DSpark cycle (verify M=6 + draft M=5 + 5 Markov) fp32 / bf16 | ~0.96-1.32 / ~0.49 ms | modeled | |

## 3. What this means for our runtime

Options are ranked by modeled ms; each says file/kernel, ms at 1 and 6 rows, numerics class.

G1. Rewrite the MXFP8 GEMV main loop for bytes-in-flight (runtime/gemv_mxfp8.py, new kernel next to t9b_mxfp8_gemv; Triton first, CuTe DSL if Triton cannot hold the pipeline):
  * per program a few weight rows (BLOCK_N 8-32) x a K-contiguous chunk of 256-512 B per row per iteration (8-16 K-blocks), 16-B vector loads (or TMA 2-D boxes with an mbarrier ring, as TGV/tinygemm2 do), 4-8 stages, evict-first hint on weights; keep the per-32-block partial -> exact pow2 scale -> fp32 add-in-K-order body unchanged, so for the SAME split partition the arithmetic is today's;
  * split-K reduce in the SAME launch: last-arriving CTA per tile (int32 arrival counter) sums the fp32 partials in split order from +0.0 -- t9b_splitk_reduce's exact order, i.e. bit-identical to today's two-launch output by construction (CUTLASS stream-K deterministic mode is the precedent: https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/ );
  * split partition keyed on (N, K) only, never on M (row-invariant for M <= 8 by construction; fixes the census's "tuned M=6 configs" item without the pin-M1 tax);
  * same-input groups via a 2-3 segment pointer table (extend t9b4_mxfp8_gemv2): {wq_a, wkv}, {w1, w3}, {attn wq_b, indexer wq_b}; and lift T9b-4's one-row guards (_rows(xq) == 1) so 6-row steps get the pair/fold too.
  Expected: GEMV 3.43 -> ~1.4 ms at both 1 and 6 rows on the bench basis (modeled, -2.0 ms). Against the live 10.52 ms 1-row step, where T9b-4 already took part of this (pair + fold), the remaining gain is ~1.2-1.6 ms (modeled; there is no per-kernel anatomy at 10.52 ms -- measure it). At 6 rows, where pair/fold are currently off, the full ~2.0 ms applies. Numerics: bit-identical to today's GEMV for unchanged split partitions IF the 32-wide block partial is formed the same way (tl.dot vs mma.sync m16n8k32 vs broadcast-sum may differ in the last bit -> must be asserted per shape with numerics_key; otherwise the existing "1-ulp class" vs the reference, unchanged).
G2. Head to bf16 storage + own fixed-order fp32 FMA kernel, M <= 8, optional fused argmax/top-1 epilogue (runtime: replace the loader's "widened_bf16_to_fp32: head.weight" -- first-light-t9b4 boot record -- and the head F.linear; same kernel serves the DSpark draft head (M=5) and the Markov head [129280, 256]):
  * 1 row: 0.42 -> ~0.21 ms (-0.21); 6 rows: unmeasured cuBLAS fp32 M=6 (0.42-0.6 modeled) -> ~0.21-0.25 ms (-0.2 to -0.35); per DSpark cycle including the drafter's head and Markov heads: ~0.96-1.32 -> ~0.49 ms (-0.5 to -0.8 ms, the biggest head number and not in anyone's 21 ms projection);
  * frees 1.32 GB HBM (the drafter's mtp.* needs 7.54 GiB of 29 free);
  * numerics: operands value-identical; accumulation order ours -> 1-ulp class (fp32 logits) vs the cuBLAS reference; row-invariant; bit-identical across 1/6 rows. Optionally calibrate our reduction order to cuBLAS gemv2T's (the project already did device-calibrated orders for 18 glue clusters) -> possibly bit-identical to the 1-row reference (unverified: depends on whether gemv2T's order is a simple strided+tree pattern).
  * Do NOT take vLLM's default (bf16 logits): A1. The vLLM/SGLang fp32-output variant (torch.mm out_dtype=fp32) is a fast fallback to try first (minutes to wire) but is tensor-core accumulation -> A3 and M-dependent.
G3. FlashInfer tgv_gemm_sm100 with MXFP8 operands as a borrowed alternative to G1 (M <= 8, K % 128 == 0 holds for all our K): requires re-laying our 32x32 weight scales as per-row 1x32 e8m0 in the 128x4 swizzle (+3.1 % bytes, values unchanged) and accepts tcgen05 block-scaled accumulation (1-ulp class at best, A3 until measured); expected per call ~ TGV bf16's per-byte numbers (2.5-6 us on our sizes); one launch per projection (no pointer groups), split behaviour unknown. Version in our image (flashinfer 0.7.0.post1) unverified for the MX path -> W4's upstream table.
G4. Not recommended: cuBLASLt MXFP8 at M <= 8 (no evidence of > 2 TB/s at small N, per-M heuristics); bf16-dequantized weights + tinygemm2 (value-exact dequant, but 2x bytes and +5.5 GB HBM); atomic split-K (nondeterministic).

Sum for the headline (modeled): G1 + G2 take ~2.2-2.4 ms out of a 6-row verify step (from the projected ~21 ms -> ~18.7 ms) and ~1.4-1.8 ms out of the 1-row step (10.5 -> ~8.9 ms), plus ~0.5-0.8 ms per DSpark cycle on the draft side. That is necessary work, not sufficient: even the pure-bytes floor of the GEMV + bf16 head is ~0.9-1.0 ms, so the remaining distance to 5.0 ms at 6 rows lives in launch count (W1, M2 fusion) and MoE, not here.

## 4. What must be measured on the box (each <= 30 min)

P1 (10 min) nsys anatomy of the CURRENT 10.52 ms 1-row step and the 6-row teacher-forced step (32 steps each, same instrument as step-anatomy-r2): GEMV main / reduce / pair launches and ms after T9b-4, and the cuBLAS fp32 head kernel name + us at M=6. Every G-number above is relative to the 13.9 ms anatomy until this exists.
P2 (20 min) bytes-in-flight microbench, no math: one Triton streaming-read kernel, CUDA-graph timed on rotating buffers (as autotune does), at 2.6 / 6.6 / 11.8 / 42 / 157 MB, two access patterns: (a) today's [BLOCK_N, 32 B] per K-block, (b) [BLOCK_N, 512 B] K-contiguous 16-B vectors, 4-8 stages. Output: achievable us per size on GB300 = the floor G1 can hit. Kill G1 if (b) is not >= 1.8x faster than (a) at 6.6-11.8 MB.
P3 (30 min) G1 prototype on 1280x5120 and 5120x8192 at M=1 and 6: K-chunked loads + in-kernel last-CTA fixed-order reduce. Gates: torch.equal vs t9b_mxfp8_gemv with the same (PER, NSPLIT, MMA) on 64 random + the real L0/L13 weights; us vs the bench table; one launch per call in the graph.
P4 (30 min) G2: bf16-stored head, Triton fixed-order fp32 FMA kernel, M in {1, 5, 6, 8}. Measure us vs cuBLAS fp32 F.linear at the same M (this also gives the missing M=6 head cost); on 1,024 real final hidden states from a first-light run: max |logit diff| in fp32 ulps vs the reference head, argmax agreement, and torch.equal of row r at M=6 vs M=1. Optional 4b: search reduction orders to match cuBLAS gemv2T bit-for-bit at M=1 (as done for the glue clusters); report match rate.
P5 (10 min) the fast fallback: torch.mm(x_bf16, W_bf16.t(), out_dtype=torch.float32) at M=1/6 -- us, ulp drift vs the fp32 reference, row invariance. Expect ~0.2 ms and A3-class drift; this number decides whether G2 needs a custom kernel at all.
P6 (20-30 min) does the image's flashinfer 0.7.0.post1 expose tgv_gemm_sm100 with MX (e8m0) operands? If yes: one shape (1280x5120) with re-laid 1x32 weight scales -- us at M=1/6 and drift vs our GEMV.
P7 (2 + 20 min) production head dtype: read Pg61's launcher for hf_overrides head_dtype (box worker; file under ~/dsv41/garner/). Then an A1 DIAGNOSTIC only: rerun FIRST-LIGHT's 8-prompt parity with our head's logits rounded to bf16 -- if it reproduces production's token divergences, the parity gap has a named cause.

## 5. Sources (all fetched 2026-11-07 unless noted; dates are the source's)

* TRT-LLM issue #20634 (2026-10-19), DSV4.1 wq_a/wkv tinygemm2 on Vera Rubin: https://github.com/NVIDIA/TensorRT-LLM/issues/20634
* FlashInfer PR #3281 (opened 2026-05-10, merged), CuTeDSL TGV backend, B200 numbers: https://github.com/flashinfer-ai/flashinfer/pull/3281
* FlashInfer tgv_gemm_sm100 docs (MX block-scaled, M <= 8): https://docs.flashinfer.ai/generated/flashinfer.gemm.tgv_gemm_sm100.html
* FlashInfer mm_fp8 docs (trtllm_low_latency, cutedsl_low_latency m <= 8): https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp8.html
* FlashInfer tinygemm_bf16 docs: https://docs.flashinfer.ai/generated/flashinfer.gemm.tinygemm_bf16.html
* FlashInfer PR #4274 (2026-07-30, merged), tinygemm2_sm100 bit-identical: https://github.com/flashinfer-ai/flashinfer/pull/4274
* FlashInfer gemm API list at 0a761d12 (router GEMMs mm_M1_16_*, tgv, grouped): https://github.com/flashinfer-ai/flashinfer/blob/0a761d12/docs/api/gemm.rst
* FlashInfer v0.7.2 release notes (prims_ts fused fp8 QKV/SwiGLU GEMMs): https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2
* SGLang PR #27397 (2026-06-05), JIT dsv3 fused-A GEMM, SM100/SM120 numbers: https://github.com/sgl-project/sglang/pull/27397
* SGLang PR #45631 (merged 2026-10-22), bf16 lm_head -> fp32 logits buffer, DSV4.1 on GB300: https://github.com/sgl-project/sglang/pull/45631
* vLLM PR #48390 (merged 2026-07-13), head_dtype for generation models: https://github.com/vllm-project/vllm/pull/48390
* vLLM LogitsProcessor (stable docs source): https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/logits_processor/
* vLLM v0.31.0 MXFP8 FlashInfer linear kernels (CuTe-DSL/CUTLASS/TRT-LLM wrappers): https://docs.vllm.ai/en/v0.31.0/api/vllm/model_executor/kernels/linear/mxfp8/flashinfer/
* TRT-LLM DeepSeek-R1 B200 min-latency blog (Fuse_A_GEMM, weight concatenation, PDL): https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html
* Hazy Research, "Look Ma, No Bubbles!" (2025-05-27): https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
* Mirage Persistent Kernel: https://github.com/mirage-project/mirage
* whirl-llm kernel guide (grouped same-input GEMVs, bit-identical): https://github.com/tsaipifong/whirl-llm/blob/main/docs/guide/en/kernels.md
* Triton grouped GEMM tutorial: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
* CUTLASS Blackwell CuTe-DSL grouped GEMM example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py ; grouped schedulers: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html
* Stream-K (Osama et al., 2023): https://arxiv.org/abs/2301.03598 ; Colfax CUTLASS stream-K tutorial (deterministic vs ReductionMode::Nondeterministic): https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
* PyTorch, "Accelerating Llama3 FP8 Inference with Triton Kernels" (SplitK, atomic reduce): https://pytorch.org/blog/accelerating-llama3/
* Chips and Cheese, "Nvidia's B200: Keeping the CUDA Juggernaut Rolling": https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut
* Jarmusch & Chandrasekaran, Blackwell microbenchmarks (arXiv 2512.02189): https://ar5iv.labs.arxiv.org/html/2512.02189
* Fasi, Higham, Mikaitis, Pranesh, "Numerical behavior of NVIDIA tensor cores" (PeerJ CS 2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/
* cuBLASLt block-scaling modes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html , https://github.com/nvidia/cudalibrarysamples/issues/310
* B300 FP32 spec (secondary, inconsistent 60-80 TFLOPS): https://gpucompat.com/gpus/nvidia-b300/ , https://nor-tech.com/wp-content/uploads/2026/03/blackwell-datasheet-B300.pdf
* Marlin, arXiv 2408.11743 (not re-fetched today).
* Local (read-only): runtime/gemv_mxfp8.py; docs/gemv-bench-r2-20261106.json; docs/step-anatomy-r2-20261107.json; docs/STEP-ANATOMY-R2-20261107.md; docs/M1-T10-RESULT-20261107.md; docs/first-light-t9b4-20261107.json; docs/LEDGER.md; docs/HANDOFF-T9B4-RESIDUAL.md; reference/deepseek-inference/model.py:281-293, 499-556, 945-962, 999-1018, 1080-1160, 1195-1275; reference/vllm-0.31.0/deepseek_v41/nvidia/model.py:1330-1385.


W3: GEMV is launch/latency-bound, not bandwidth-bound -- a bytes-in-flight rewrite with in-kernel deterministic split-K and same-input groups (~-2.0 ms at 6 rows) plus a bf16-stored fixed-order fp32 head (~-0.2 ms/step, ~-0.6 ms/DSpark cycle, row-invariant, 1-ulp class) is worth ~2.3 ms of the 6-row step: do it, but the 5.0 ms bar lives in launch count and MoE.
