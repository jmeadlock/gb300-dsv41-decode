# W4: Upstream watch and numerics at M>1 (research round 2, 2026-11-07)

Author: W4 (Milo / claude-opus-5-5). Research only: web, `gh` (read-only) and the repo's own `reference/` and `docs/`. No box access. Sources were read 2026-11-07, 03:37–05:00 CST.
Labels: **[M]** = measured by us (receipt in `docs/`), **[C]** = claimed by the cited source, **[Mod]** = modeled here, **[R]** = read from a source tree (code fact, not a run).

## 0. Verdict

**(b) Row invariance is the urgent part.** Two kernel sites are not row-invariant, not one. CONTEXT/M1-T10 lists only the fp32 head, but our own GPU census also marks the **mHC mixes** as row-variant: `F.linear(x, hc_fn)` in fp32, N=24, called twice per layer, so 80 times per step. The census shows 86.8 % of elements differing, up to 224 ulp [M] (`docs/rows-gpu-test-r2-20261107.log:76-78`). Together these two sites give **0/24 bit-equal positions and 1/24 argmax flips at the step level** [M]. That is a ~4 % per-position disagreement between 6-row verify and 1-row greedy, and DSpark's exact-match contract cannot ship with it.

Public evidence says cuBLAS fp32 GEMM is not batch-invariant under any env-var trick:
- vLLM's "split-K is the only source" comment is called stale by vLLM contributors.
- fp32 rows differ at every M>1 on H100 with vLLM's settings [C].

Two routes do give invariance for fp32:
1. **cuBLAS's new experimental batch-invariance mode.** It is SM10x only, covers fp32 with `CUBLAS_COMPUTE_32F`, and is documented in cuBLAS 13.4+/13.5 docs dated 2026-11-02. It is claimed present from `nvidia-cublas` 13.8.0.4. One tester reports 0/1,460,100 rows differing, fp32 included, on B200 with 13.8.2 [C]. Almost certainly not in our image's cuBLAS (to be verified).
2. **Our own fixed-order FFMA kernels** for the mixes and the head. The head has bf16 checkpoint weights widened exactly to fp32, so a bf16-weight read is product-exact and halves the head bytes.

Both routes are "1-ulp class" against today's reference unless the oracle process adopts the same kernel or flag. That is a policy decision for the A3 amendment. My recommendation is to adopt it in the oracle, so that 1-row == oracle and 6-row == 1-row both hold bit-exactly.

The vLLM, SGLang and Thinking Machines batch-invariant stacks give invariance, not equality with stock torch/cuBLAS at M=1. None of them covers the DeepSeek-V4.1 path:
- vLLM's V4.1 backends do not declare batch-invariance support.
- SGLang's `dsv4` backend is refused under `--enable-deterministic-inference`.

**(a) Upstream:** the most useful findings are about what is already in our image and what is broken in it, not new speed.
1. Our DeepGEMM is **not** `88965b07`. vLLM v0.31.0 builds the `vllm-project/DeepGEMM` fork at **`e1f418c2` (2026-09-19)**, 16 commits ahead [R]. It already has `set_pdl`, `use_deterministic_algorithms`, Mega MoE/Gate/mHC, and the 26/09 change that made determinism **opt-in**. Our runtime already sets it to True, and that call is load-bearing: with it off, bf16 GEMMs and einsums route to cuBLASLt, and mHC/Gate split-K varies with M.
2. The image's DeepGEMM **Mega mHC has a CUDA-graph-replay race**: `%gridid`-tagged split barriers (fixed upstream 2026-10-20, #478; JIT-header-only). The image's FlashMLA predates two fixes:
   - the SM100 sparse-decode head64 `rowwise_max_buf` race (fixed 2026-10-27/29);
   - `enable_batch_invariant` for sparse decode.

   Neither affects our current FlashMLA *prefill* path. Both matter if production Pg61 runs those kernels, and they matter for any move to decode kernels.
3. `fused_norm_rope_attn_rope_cast_fwd` is in our `.so` and takes our bf16 caches. It does Q-RoPE, sparse attention, O inverse-RoPE and the per-32 UE8M0 FP8 cast in one kernel, with Q_b/Wv weights permuted (layout only). It could absorb ~2–3 of our attention-path launches per layer: **~0.15–0.45 ms at 1 and at 6 rows [Mod]**. It is A3 class at best, the same class as our current FlashMLA adoption: inverse RoPE on the fp32 accumulator instead of the bf16-rounded O moves FP8 codes, so it is not bit-exact.
4. vLLM v0.31.0 also already ships a CuTe-DSL `FusedWoAKernel` (inverse RoPE + quant + wo_a + MXFP8 quant, 1–96 tokens, explicit `mul.rn`/`fma.rn` RoPE ordering). It is pure Python but imports vllm helpers, so it needs vendoring.

   vLLM merged an opt-in **GB300 latency-specialized V4.1 decode** (#61868, 2026-10-21; claimed 5,355 → 4,375 µs per spec step at batch 1, TP4, ~1,600 → fewer kernels). Its 1–8-row CuTe-DSL routed-expert kernels and low-latency MXFP8 GEMMs are the closest public analogue of our M2 work.

## 1. Pins: what our image actually contains

| component | doc'd pin | actual pin per v0.31.0 build files | date | evidence |
|---|---|---|---|---|
| vLLM image | v0.31.0 | v0.31.0 (released 2026-10-05) | 2026-10-05 | `gh release` [R] |
| DeepGEMM | "88965b07" (WORKERS.md) | **vllm-project/DeepGEMM `e1f418c2`**, `__version__ = '2.8.0'` | 2026-09-19 | `cmake/external_projects/deepgemm.cmake@v0.31.0` line 36; `tools/install_deepgemm.sh@v0.31.0` line 13; vendored `reference/vllm-0.31.0/deep_gemm/__init__.py` (has `set_pdl`, `use_deterministic_algorithms`, `mega_mhc`, `sm120_*` headers from fork #4) [R] |
| (88965b07 is) | | deepseek-ai/DeepGEMM "Multiple updates and refactorings (#347)" | 2026-06-01 | `e1f418c2` is 16 commits ahead of it (`gh api compare`) [R]. vLLM's `csrc/fused_mqa_topk/*` cites 88965b07 as "the last release before its SM100…" adaptation. That is probably where the number came from. |
| FlashMLA | "as vendored" | **vllm-project/FlashMLA `0eee43b1`** | 2026-09-15 | `cmake/external_projects/flashmla.cmake@v0.31.0` [R]. Includes #23 (caller FP8 out buffers, 09-14), which matches our `.so` schema (`out_fp8`, `out_sf`) in `flashmla-probe-20261106.json` [M]. |
| FlashInfer | 0.7.0.post1 | 0.7.0.post1 | 2026-09-29 | `requirements/cuda.txt@v0.31.0` [R] |
| Triton | 3.7.1 | 3.7.1 (2026-06-18); latest 3.9.0 (2026-10-27) | | `gh release` [R] |
| TileLang | 0.1.12 | 0.1.12 (2026-07-08); latest 0.1.15 (2026-09-30) | | reference pins 0.1.12, keep it [R] |
| CuTe DSL | 4.7.1 | 4.7.1 (2026-08-26); latest 4.8.1 (2026-10-16) | | PyPI [R] |
| cuBLAS | unknown | torch 2.13.0+cu130 wheel's cuBLAS: **unverified**, probably 13.0.x/13.1.x | | box probe P1 |

Later vLLM pins, for reference [R]:
- DeepGEMM: v0.32.0 → `1e1842a8`; v0.33.0rc3 → `08200307`; main → `82ca617e`.
- FlashMLA: v0.32.0 → `0eee43b1` (unchanged); v0.33.0rc3 → `e2afa5fd`; main → `847ef308`.

## 2. Upstream since our pins (findings, with dates)

"In image" means the change is inside the pins of section 1. "Overlay" says whether we could test it on our image without a rebuild:
- **JIT** = DeepGEMM `include/*.cuh`, compiled at runtime. Swap headers in a copy of the package dir.
- **py** = pure Python / CuTe DSL / Triton.
- **csrc** = needs a rebuilt `.so`.

### 2a. DeepGEMM (deepseek-ai upstream + vllm-project fork)

| change | merged | in image? | kind | relevance to us |
|---|---|---|---|---|
| 26/04 release #304: Mega MoE, FP8×FP4 GEMM, FP4 indexer, **PDL**, faster JIT | 2026-04-17 | yes | csrc+JIT | `deep_gemm.set_pdl()` is already in our image; see W1. |
| #347 (= 88965b07) / #364 multiple updates: unified MQA-logits kernel, BF16 Mega MoE, ring buffer | 06-01 / 06-23 | yes | csrc+JIT | none directly |
| #343 fix: in-flight tensormaps updated in place (race) | 07-06 | yes | csrc | correctness, already ours |
| 26/07 #377: MXFP8 indexer; Mega MoE +20 %, "11 kernels into 1", shared experts fused [C] | 07-15 | yes | csrc+JIT | Mega MoE is the 1-launch MoE (M2 memo's domain) |
| **26/09 #432**: Sparse Indexer, **Mega Gate** (routing GEMM + routing ops, 30–75 % faster [C]), **Mega mHC** (HC ops + RMSNorm in one kernel, 45–85 % faster [C]), DeepJIT. **Breaking: "Previously, all kernels were fully deterministic and batch-invariant. This release adds `use_deterministic_algorithms()` … defaults to False … Set True to restore deterministic, batch-invariant behavior."** | 2026-09-10 | **yes** (version 2.8.0) | csrc+JIT | See the box below. |
| fork #4/#9: SM120 ports and fixes; #11 Mega-Gate trap removed; #13 watchdog 300 s | 09-14…16 | yes | csrc | none |
| **#478 Fix `mega_mhc` split barriers under CUDA graph replay** | 2026-10-20 (fork: #48, 10-29) | **NO**: vendored `scheduler/mega_mhc.cuh:20-45` + `ptx/utils.cuh:25` use `%gridid` [R] | **JIT headers only** ("host-side barrier buffer and compiled extension are unchanged") | See the box below. |
| 26/09/30 #462: epilogue classes (FP8 cast, BF16 stochastic rounding), BF16 einsum fp32 out, padded-K packed UE8M0 SF, **Mega MoE combine reduction order fixed "for bitwise agreement with the shared-expert reference"**; breaking MQA-logits API changes, legacy `fp8_mqa_logits` removed, Mega Gate sqrtsoftplus only | 2026-09-30 | no | csrc+JIT | Only if we adopt Mega MoE. The API breaks mean a straight bump is not drop-in. |
| 2.8.2 #471: `FP8CastWithClamp` epilogue (replaces `FP8Quantization`) | 10-14 | no | csrc+JIT | none |
| fork #15/#21/#34: pybind11 → TORCH_LIBRARY, stable ABI (min torch 2.11) | 10-10…13 | no | csrc | v0.32+ wheels; irrelevant on our image |
| fork #40: FP8 einsum writes output SFs in 128×4 tiled layout | 10-17 | no | JIT | vLLM #60118 uses it so `wo_a` hands `wo_b` MXFP8 (0.4–2.0 µs/layer on GB200 [C]) |
| fork #17/#19/#22/#26/#27/#39/#43/#51/#44: NVFP4 MegaMoE, Muse Spark, SM90 FP8×FP4 MegaMoE | 09-23…11-06 | no | csrc+JIT | none (we are MXFP4) |

**#432 (2026-09-10), what the opt-in determinism means for us [R]**
- What the flag gates (deepseek-ai `csrc/apis/gemm.hpp:519`, `einsum.hpp:72,95`, `mega_mhc.hpp:16`, `heuristics/mega_gate.hpp:206`):
  - `bf16_gemm_*` and the bf16 `bhr,hdr->bhd` einsums go to **cuBLASLt** unless deterministic is on.
  - `mega_mhc` picks split-K from `num_m_blocks` (that is, from M) unless deterministic, where it uses a fixed 16.
  - `mega_gate` picks `num_split_k` unless deterministic, where it is 1.
- The fp8_fp4 GEMMs (our MoE path) are **not** gated by the flag.
- Our runtime already calls `use_deterministic_algorithms(True)` at build (`runtime/README.md:281`, `dg_moe.py:153`). Keep it, and assert it in the census process too.

**#478 (2026-10-20), the Mega mHC graph-replay race**
- Claimed race rate on B200, wrong replays [C]:

  | case | before fix | after fix |
  |---|---|---|
  | 33 tokens, SMs contended | 185/200 | 0/400 |
  | 512 tokens, idle GPU | 1/200 | 0/400 |

- **Do not adopt Mega mHC from the image without overlaying #478.**
- Possible production hazard if Pg61 (vLLM 0.31-based) uses `mega_mhc` inside graphs; that is unverified.

### 2b. FlashMLA (deepseek-ai upstream + vllm-project fork)

| change | merged | in image? | kind | relevance |
|---|---|---|---|---|
| #221 "Add kernels for DeepSeek v4.1": sparse decode head64 (+ `no_split` instantiations), sparse prefill, **fused norm+RoPE+attn+RoPE+cast** fwd/decode, `permute_q_b_proj`/`permute_wv_proj` | 2026-09-10 (fork sync #20, 09-10) | yes | csrc | §3a |
| fork #23: caller-provided FP8 output buffers for the fused API | 09-14 | yes | csrc | yes; this is what the fused path needs under graphs |
| fork #21: round NVFP4 tile scale to nearest; #18 NVFP4 KV for SM100 sparse decode | 08-27 / 09-14 | yes | csrc | production's compressed cache only |
| fork #24 (= pin `0eee43b1`): nvcc 13.0 `if constexpr` fix | 09-15 | yes (the pin) | csrc | none |
| upstream #229 / fork #30: 2026-09-30 release. V4.1 KV cache becomes **interleaved**; new **`enable_batch_invariant`** for sparse decode ("split-KV decoding is disabled so that the result does not depend on how the batch is partitioned"; NVFP4 and SM90 reject it) | 09-30 / 10-11 | **no** | csrc | Only if we move from `sparse_prefill_fwd` (unsplit, row-invariant: census [M]) to `sparse_decode_fwd`. That move would need this flag, or our own fixed `num_splits`, to stay row-invariant. |
| upstream #230: stable-ABI migration; +≤40 µs host overhead eager, erased in graphs [C] | 10-15 | no | csrc | none |
| **upstream #241 / fork #39: race on `rowwise_max_buf` before the `li` exchange in SM100 sparse decode head64** (row output and LSE silently wrong; "rare", reproduced only with a forced stall) [C] | 10-27 / 10-29 | **no** | csrc | Not our path: prefill uses a separate `rowwise_li_buf` per #241 [C]. Production's V4.1 fp8/fp4 sparse decode at h_q=64 reaches this kernel per #39's table. Whether the *fused* decode op shares the code is unverified. |
| fork #38: fused kernels take **int64** `token_positions` (removes one cast kernel per layer in vLLM) | 10-29 | no | csrc | Our image's op takes int32 positions (README example) [R]. |
| fork #41: SM90 dense decode PDL | 11-01 | no | csrc | none |

### 2c. vLLM / SGLang DeepSeek-V4.1 paths (what is upstream, what is already in our image)

| PR | merged | in v0.31.0 image? | what | notes for us |
|---|---|---|---|---|
| #56935 FlashMLA **mega attention** default on SM100, with NVFP4 compressed cache | ≤ v0.31.0 | yes | `fused_norm_rope_attn_rope_cast_decode` for decode | vLLM's own GB300 table, mega vs split-KV µs incl. all per-step ops [C]: **TP1 29/29 at s_q=1, 31/31 at 32, 31/40 at 128**. At one token the fused kernel does not beat vLLM's split path there; it wins at s_q ≥ 128 (`reference/vllm-0.31.0/deepseek_v41/nvidia/flash_mla_mega_attn.py:61-83`). |
| #58634 `FusedWoAKernel` (CuTe DSL): inverse RoPE + per-32 UE8M0 quant + MXFP8 `wo_a` (tcgen05, split-K by heads over DSMEM) + BF16 round + MXFP8 quant for `wo_b`; 1–32 tokens (1–96 at TP4/8). Replaces 3 kernels of 11–12 µs total on GB200 TP4 [C]. Explicit `mul.rn.f32`/`fma.rn.f32` "preserves the prior RoPE rounding order". | 2026-09-27 | **yes** (`reference/vllm-0.31.0/deepseek_v41/nvidia/ops/fused_wo_a.py`, 599 lines) [R] | py (CuTe DSL 4.7.1) | It imports `vllm.cute_utils` and the warmup helpers, so it needs vendoring (Apache-2.0) to respect "never import vllm". |
| #57643/#58586 fused TP all-reduce + mHC prep + MoE finalize; #57679 fused query RMSNorm + MXFP8 restored; #56266 Mega-Gate; #56254 DeepGEMM sparse MQA logits | ≤ v0.31.0 | yes | | |
| #60029 pad Q to 64, not 128, in FlashMLA sparse **prefill**; 1.8–2.0× kernel at 64 heads [C] | 2026-10-06 | no (v0.32) | py | **Already true for us**: we pass h_q=64 to `sparse_prefill_fwd` (`runtime/attn_flashmla.py`). |
| #60118 MegaAttn `wo_a` writes `wo_b`'s MXFP8 input; "outputs bitwise identical" [C] | 10-19 | no | py + DeepGEMM fork #40 | Production-only optimisation |
| **#61868 latency-specialized V4.1 decode on GB300** (opt-in `decode_backend: latency_specialized`). TP4 tray, DSpark k=5, up to 384 rows. Claimed spec step incl. draft: **5,355 → 4,375 µs at batch 1**, −13.7 % at batch 8, −11.3 % at 32. GSM8K/GPQA/AIME within a point; acceptance unchanged (3.93) [C]. | 2026-10-21 | no (v0.33rc) | py (CuTe DSL kernels + executor) | Measured on TP4 with an fp8 KV cache and the FlashInfer sparse-MLA backend, so it does not transfer 1:1 to TP1/our caches. Detail in the box below. |
| #61869 DSpark draft on the specialized decode | open | — | | |
| #56148 Triton split-K verify attention (multi-query ≤ 8) | 10-13 | no | py | Not our attention backend |
| #59468 DSpark no longer reserves draft slots (GB300 1373 → 3390 tok/s in PR bench [C]) | v0.33rc | no | py | W2's |

**#61868 detail [C]**
- "~1,600 kernels per step". One indexer-free layer is a serial chain of "~105 µs: ~50 µs attention + ~53 µs MoE".
- Routed-expert CuTe DSL kernels for 1–8 rows, µs at 1 / 6 rows:

  | kernel | 1 row | 6 rows |
  |---|---|---|
  | front | 3.04 | 4.06 |
  | route | 2.18 | 1.95 |
  | GEMM1 | 9.76 | 17.60 |
  | GEMM2 | 6.24 | 10.37 |

- FlashInfer low-latency CuTe-DSL MXFP8 GEMM with per-shape pinned tactics for `fused_wqa_wkv` / `wq_b` / `wo_b`.
- **LM head on FlashInfer's CuTe-DSL bf16 GEMM**, so production's head is bf16 (see W3).
- "no LoRA, KV transfer or **batch invariance**".

SGLang:
- Deterministic mode builds on the Thinking Machines ops and supports only the FlashInfer, FA3 and Triton attention backends (docs.sglang.io, read 2026-11-07).
- DeepSeek-V4's `dsv4` backend is refused under `--enable-deterministic-inference`. Issue sgl-project/sglang#36174 was opened 2026-08-24 and **closed for inactivity 2026-10-28** without a fix [R].
- I found no SGLang V4.1 fused-norm/rope kernel beyond FlashMLA's (unverified, not exhaustively searched).

### 2d. FlashInfer (MoE / GEMV / low-latency GEMM)

- **In our 0.7.0.post1 already** [R]: tree at tag `v0.7.0.post1` has
  - `flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py`
  - `dense_blockscaled_gemm_sm103.py`
  - `gemm_mm_mxfp8_cute_dsl.py`
  - TGV GEMM (`csrc/tgv_gemm.cu`)
  - `include/flashinfer/gemm/group_gemv.cuh`

  These are the kernels #61868 pins for the 1–8-row MXFP8 projections. The CuTe DSL ones are py/JIT, so they are overlay-testable today; this is input to W3.
- **0.7.3 (2026-11-04)**: `FrostLowLatencyConfig`, a cuDNN Frost low-latency MoE for **1–8 tokens**, covers MXFP8 activations and MXFP4 weights (#6626, #7184). It needs cuDNN Frost, so it is not overlay-trivial. It also has a "deterministic unfused finalize" for the CuTe DSL fused MoE (#6950, Rubin).
- 0.7.1/0.7.2: DSv4.1 dual-cache sparse MLA on SM12x, NVFP4 sparse-MLA decode on B200/B300 ("cake"), "round-6 DSv4 sparse-MLA Cake programs (exact numerics)" (#5700). No MXFP4 GEMV for SM10x found. 0.7.0.post1 has `gemv_bf16_fp4_sm12x.py`, which is SM12x only.

## 3. `fused_norm_rope_attn_rope_cast_{fwd,decode}`: what it computes and whether it can replace our q/kv/o path

### 3a. What it computes

Sources: deepseek-ai/FlashMLA README §"Fused norm + RoPE + attn + RoPE + cast kernel" (read 2026-11-07); vllm-project/FlashMLA@0eee43b1 `tests/test_fused_norm_rope_attn_rope_cast.py`; our vendored `flash_mla_mega_attn.py` [R].

One kernel does five things:
1. Optional Q RMSNorm. It is **off for V4.1**, which norms qr before `wq_b`.
2. Q RoPE, GPT-J (non-neox) style, rope_dim 64, from a `cos_sin_cache [pos, 32 cos | 32 sin]` and int32 positions in our image.
3. Sparse attention with sink and `topk_length`, the same contract as `sparse_prefill_fwd`.
4. **O inverse (conjugate) RoPE.**
5. The **FP8 e4m3 cast with per-32 packed UE8M0 scales**, written in DeepGEMM's TMA-aligned MN-major SF layout. The output is chunk-interleaved for a *permuted* `Wv`/`wo_a`, and the consumer is `deep_gemm.fp8_einsum("bhr,hdr->bhd", …, recipe=(1,1,32))`.

Two entry points:
- **`_fwd`** takes bf16 non-paged KV `[s_kv, h_kv, d_qk]` with RoPE already applied, which is exactly our cache form. vLLM uses it for prefill over a gathered bf16 workspace.
- **`_decode`** takes paged fp8 (528 B) / NVFP4 (288 B) records, which is production's form.

Requirements:
- `h_q ∈ {64, 128}` for FP8 decode. We have 64 heads.
- `n_heads / o_groups = 8` heads per `wo_a` group. We have 64/8 = 8 ✓.
- `wq_b` rows and `wo_a` columns are permuted once at load (`permute_q_b_proj`, `permute_wv_proj`; both ops are in our `.so`). This is layout only, which policy allows.

### 3b. Numerics against our reference

| step | reference (`model.py:777-787`) | fused kernel | class |
|---|---|---|---|
| Q RoPE | fp32 complex mul → `copy_` to bf16 | The test reference uses `fma(x0,cos,-(x1*sin))` / `fma(x0,sin,x1*cos)` → bf16. Our glue's ROPE_MODES calls this **mode 1** (`runtime/fused_glue.py:38-43`). | **Bit-identical iff** (i) the device-calibrated mode is 1 for both components, (ii) the kernel uses that form, and (iii) we pass a table holding exactly `freqs_cis.real/imag`, including YaRN rows. (ii) is unverified for the kernel itself; the test reference is not the kernel. |
| attention | TileLang `sparse_attn` | FlashMLA tiling, exp2 softmax | A3, the same as today's `--attn-impl flashmla` |
| O inverse RoPE | applied to the **bf16-rounded** `o` | The test reference applies it to `out` straight from the attention reference. The kernel almost certainly rotates the fp32/bf16 accumulator in registers (unverified). | A3: a different rounding point. FlashMLA's own test tolerates `abs 1.1e-3, rel 1/8` on dequantized `out` [R], so it does **not** claim bit-exactness. |
| QDQ | `act_quant(o, 32, ue8m0)` QDQ, bf16 in place | per-32 UE8M0 with `round_sf=True`; fp8 values out (not QDQ'd bf16) | The scale rule matches: power-of-two round-up of amax/448, the same as `fused_wo_a.py:_scale` uses `cvt.rp…ue8m0` [R]. Codes can flip where O differs by 1 ulp. |
| `wo_a` | bf16 einsum on the QDQ'd `o` with dequantized bf16 `wo_a` (cuBLAS bmm, row-invariant [M]) | Either (a) dequantize fp8 → bf16, which is exact (fp8 × 2^e fits bf16), and keep our einsum; or (b) DeepGEMM `fp8_einsum`, whose products are exact but whose accumulation order differs | (a) adds back a launch; (b) is 1-ulp class |

**Verdict on 3:** it cannot replace our path bit-exactly. The O-side rounding point differs, and FlashMLA's own test is tolerance-based. It is an **A3-class** option.

Its value is small at 1 row, because we already have 1-launch glue:
- `q_rope_epilogue` is folded into the `wq_b` GEMV, so it costs 0 launches.
- `o_rope_qdq` is 1 launch per layer (`fused_glue.py:11-21`) [R].
- So the fused kernel would save **≈ 1 launch × 40 layers ≈ 0.08–0.15 ms at 1 row [Mod]**, assuming 2–4 µs per sub-µs-work launch in-graph.

It is larger at 6 rows, because both glue clusters are **n/a at rows > 1** ("one position's freqs row per launch", `fused_glue.py:121-122`) and the reference op chains run instead. Those chains are ~6–8 launches per layer, which models to **0.5–1.0 ms [Mod]**.

The cheaper and **bit-identical** way to recover that same 6-row saving is to give our own `o_rope_qdq` / `q_rope_epilogue` a per-row freqs table. Do that first.

vLLM's own GB300 measurement shows the mega kernel tying its split path at s_q=1 (29 vs 29 µs per layer for the whole bundle) and winning only at s_q ≥ 128 [C]. Expect no kernel-time win at 6 rows beyond the launches it removes.

A related production kernel, **`FusedWoAKernel`** (vLLM #58634, in our image as py/CuTe DSL), fuses the *next* three steps: inverse RoPE + QDQ + `wo_a` + `wo_b`'s MXFP8 quant.
- Its RoPE uses explicit `mul.rn` + `fma.rn` ordering (`fused_wo_a.py:57-70`) [R]. That is our mode 1, so the RoPE part can be bit-exact.
- Its `wo_a` is a tcgen05 MXFP8 MMA with a fixed 8-way head split-K. Products are exact, the order differs, so it is 1-ulp class against the bf16 einsum.
- It is row-invariance-plausible (the split count is fixed by layout, not by M) but **unverified**.

## 4. Row invariance at M>1

### 4a. Our measured state

Source: `docs/rows-gpu-test-r2-20261107.log:55-90`, layer-0 shapes [M]. Every family below is listed in the log as equal (`yes`, 0 ulp) at 6 rows vs 1 row:
- `act_quant`
- ref `fp8_gemm` (`wq_a`, `wkv`)
- our fp8 GEMV, tuned M=6 **and** pinned M=1. Both show numerics_key `(4,2,1)` at M=1 and M=6 on these shapes, so on layer 0 the tuned and pinned configs are the same config.
- ref `fp4_gemm`
- DG masked (`masked_m` 6 vs 1)
- ref `sparse_attn` and FlashMLA with per-row lists
- RMSNorm (torch)
- **gate fp32 `F.linear` (N=384)**
- bf16 `F.linear` (indexer `weights_proj`)
- bf16 einsums (`wo_a`, indexer)
- `hc_split_sinkhorn`
- `torch.topk`

**Not invariant:**

| site | shape | calls/step | mismatch | max ulp |
|---|---|---|---|---|
| **cuBLAS fp32 `F.linear`, mHC mixes** (`hc_attn_fn`/`hc_ffn_fn`, `model.py:956`) | [M, 20480] × [24, 20480]ᵀ | 80 (2 per layer) | 86.8 % | 224 |
| **cuBLAS fp32 `F.linear`, head** (`model.py:1014`) | [M, 5120] × [129280, 5120]ᵀ | 1 | 69.2 % | 20,480 |

**Step level** (6-row row j vs 1-row at P0+j, 24 positions) [M]:

| impls | bit-equal | max \|Δlogit\| | argmax agree |
|---|---|---|---|
| ref | 0/24 | 11.99 | 23/24 |
| ref + `row_invariant_mode` (per-row cuBLAS) | **24/24** | 0 | 24/24 |
| fast, GEMV pinned | 0/24 | 11.5 | 22/24 |

The difference between the first two rows proves that these two cuBLAS fp32 sites are the whole V0 gap on the ref path. Two corrections follow:
- **M1-T10-RESULT's line "the only non-invariant kernel is the fp32 head" is wrong.** The mixes break invariance too, and they are the larger lever, because their error compounds through 40 layers before the head.
- The GEMV "tuned M=6 vs M=1" distinction in M1-T10-RESULT did not show up on layer 0's shapes. Other layers' shapes are not in the log; verify there.
- Open question: why the fast path is also 0/24 with the GEMV pinned. The same two cuBLAS fp32 sites should account for it, and the log does not rule out others. The `row_invariant_mode` rerun on the *fast* impls is missing (probe P0).

### 4b. cuBLAS / cuBLASLt determinism and split-K vs M

1. **Documented guarantee.** cuBLAS docs §2.1.4 "Results Reproducibility" (cuBLAS 13.5 docs, last updated 2026-11-02):

   > "By design, all cuBLAS API routines from a given toolkit version, generate the same bit-wise results at every run when executed on GPUs with the same architecture and the same number of SMs"

   This is run-to-run only, not across shapes. `CUBLAS_WORKSPACE_CONFIG=:16:8|:4096:8` is documented only as the multi-stream determinism fix. Our runs set `:4096:8` [M] (`flashmla-probe-20261106.json` meta).

2. **The env-var "no split-K" trick (vLLM).** vLLM's `enable_batch_invariant_mode()` on SM90/SM100 sets `CUBLAS_WORKSPACE_CONFIG=:16:8` + `CUBLASLT_WORKSPACE_SIZE=1`, with the comment "the only source of batch variance is split-k" (`vllm/model_executor/determinism/batch_invariant.py`, present at v0.31.0 lines 1073-1077 and on main) [R]. vLLM contributors call that comment wrong (vllm#65569, 2026-11-04: "The comment blaming split-K for all nondeterminism is just wrong. See #35404"). The evidence in the same thread [C]:
   - **fp32 `torch.mm` is not batch-invariant even after `init_batch_invariance()`**. H100 NVL, cuBLAS 13.1.1.3, K=N=4096: row 0 at M=1 vs M=2..64 differs in 3,606–3,972 of 4,096 elements (JonSnow1807, 2026-11-06).
   - The author's own data: "fp32 matches what I saw too (rows differ at every batch size above 1), so fp32 stays on Triton".
   - On B200 "with the pinned cuBLAS 13.1 … odd widths 97–127 fail".
   - On H100, narrow N (≤ 96) fails at some K, and an N=32 router fails at interleaved M values.

   Our head is N=129,280 (wide). It still fails [M], which agrees with "fp32 rows differ at every batch size above 1". Our mixes are N=24 (narrow), and those fail too [M].

3. **cuBLASLt algo pinning (the route WORKERS.md asked about).** The API exists:
   - `cublasLtMatmulAlgoInit` plus `CUBLASLT_ALGO_CONFIG_{ID,TILE_ID,STAGES_ID,SPLITK_NUM,REDUCTION_SCHEME}`;
   - `cublasLtMatmulAlgoCheck`;
   - DeepGEMM's own cuBLASLt wrapper restricts `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` to NONE|COMPUTE_TYPE (`smxx_cublaslt.hpp:83-89`) [R].

   One fixed tiled-GEMM algo with SPLITK=1 for both M=1 and M=6 is batch-position-invariant in the Thinking Machines sense *if* the kernel is data-parallel over M. **No public evidence found** that anyone has validated this for fp32 on SM10x, so treat it as unverified. It is also superseded by item 4, the official form of the same idea. It is reachable from Python only through ctypes or `nvmath-python`; whether `nvmath` is in the image is unverified.

4. **The official route: cuBLAS "Experimental: Batch Invariance"** (cuBLAS docs §3.1.5, dated 2026-11-02) [C]:
   - Setting: `CUBLASLT_MATMUL_DESC_BATCH_INVARIANCE_FLAGS = CUBLASLT_BATCH_INVARIANCE_{ALIGN_2B,4B,8B,16B,AUTO}`, or process-wide via the env var `CUBLAS_BATCH_INVARIANCE_FLAGS`.
   - What it promises: "computing any M or N sub-range of a problem produces results that are bitwise-identical to the corresponding part of the full problem", guaranteed when "the matmul descriptor, the K dimension, the workspace size, and the split-K configuration … are identical".
   - Limits:
     - **"Only GPUs with compute capabilities 10.x"**. GB300 is 10.3 ✓.
     - **"fp32 (with the CUBLAS_COMPUTE_32F compute type only), fp16, bf16, FP8, and FP4"**.
     - It "disables split-K", and some problems may return NOT_SUPPORTED.
     - **With the env var, unsupported combinations silently fall back to the non-invariant default.** Only the descriptor attribute returns an error.
   - Availability: per vllm#65569 it is in "CUDA 13.4.2 / cuBLAS 13.8.0.4" [C]. PyPI `nvidia-cublas` 13.8.0.4 was uploaded 2026-09-16; 13.8.2.12 on 10-27; 13.10.0.20 on 11-04 [R].
   - Test evidence: one tester on B200 with cuBLAS 13.8.2 and `CUBLAS_BATCH_INVARIANCE_FLAGS=8` (AUTO), batch sizes 1–8192, 25 shapes: "with PyTorch's default settings, 0 of 1,460,100 rows differ, **and fp32 is clean too**". There is one caveat: with vLLM's PyTorch flags (cuBLASLt preferred, reduced-precision reductions off), odd widths 97/127/129 were not invariant [C].
   - What our image has: torch 2.13.0+cu130 probably carries cuBLAS 13.0/13.1 (**unverified**, probe P1), so the mode is **not in our image**. It could be overlaid by preloading a newer `libcublas.so.13`/`libcublasLt.so.13`. Driver compatibility for a 13.8 cuBLAS on the box's driver is unverified.

### 4c. What vLLM, SGLang and Thinking Machines did

**Thinking Machines** ("Defeating Nondeterminism in LLM Inference", H. He et al., 2025-09-10; `thinking-machines-lab/batch_invariant_ops`) [C]:
- The diagnosis: **batch-size-dependent reduction strategy**, not concurrency or atomics. Three things cause it: split-K/split reductions, tensor-core instruction or tile changes at small M, and split-KV attention.
- The fix: substitute ops via `torch.Library` with fixed-strategy kernels, namely `mm`/`addmm` (one compiled Triton config for all shapes, no split-K), `log_softmax` and `mean`; RMSNorm is data-parallel per row; attention uses FlexAttention with a **fixed split size** (not a fixed number of splits) and the KV cache and page table updated before attention.
- Cost: "~20 % performance compared to cuBLAS" for the matmul; Qwen3-8B, 1000 sequences: 26 s → 55 s → 42 s with improved attention.
- **Equality with stock torch is not a goal:** these are different kernels, so M=1 results are not cuBLAS's.

**vLLM `VLLM_BATCH_INVARIANT=1`** (code on main, 2026-11-07; docs `features/batch_invariance.md`) [R]:
- Ops overridden:
  - `aten::mm`, `mm.dtype`, `addmm`, `matmul`, `linear` (Triton persistent matmul). The override is registered on SM80 and ROCm only; on SM90/SM100 `UnquantizedLinearMethod` calls `linear_batch_invariant` directly per vllm#65569.
  - `mean.dim`, `_log_softmax`, `softmax`, `bmm` (`allow_override=True`, because torch 2.12+ registers its own Triton bmm).
  - A Triton RMSNorm.
  - A fixed-rank-order all-reduce and reduce-scatter.
  - The NCCL pins (Tree, 1 channel, Simple).
  - `torch._inductor.config.batch_invariant=True` (#61118, 2026-10-17).
  - TF32 off in torch.
  - A deterministic split-K LoRA shrink.
  - The `GateLinear` fix (#59985).
  - Breakable CUDA graphs without torch.compile so tile configs follow the runtime M bucket (#57586, 2026-09-23).
- Tile configs vary by M bucket while staying invariant, because no split-K is used and tiles only partition M/N.
- **Not bit-identical to torch at M=1**, by design (different kernels). The tests compare BI bs=1 vs BI bs=N.
- **fp32 trap:** the persistent kernel uses `tl.dot(a, b, acc)` without `input_precision` (`batch_invariant.py:119,205,480`), and Triton's default for fp32×fp32 on NVIDIA tensor cores is `"tf32"` (triton-lang.org `tl.dot` docs, read 2026-11-07). vLLM's fp32 batch-invariant path is therefore TF32 unless `TRITON_F32_DEFAULT` is set; no such setting was found in the file. **For us that is A1-class: do not borrow it for the fp32 head or the mixes.**
- **DeepSeek-V4.1 is not covered.** The batch-invariant model list has V3 / V3.1 / R1 only. `supports_batch_invariance` is declared by neither `deepseek_v41/*` nor `flashmla_sparse` (gh code search 2026-11-07). #60271 (2026-10-08) makes models whose layers lack it **refuse to start**. #61868's specialized decode explicitly excludes batch invariance.

**SGLang `--enable-deterministic-inference`** (docs.sglang.io, read 2026-11-07) [R]:
- It builds on the Thinking Machines ops and supports only the FlashInfer, FA3 and Triton attention backends.
- DSV4 with `--attention-backend dsv4` raises at startup (sgl-project/sglang#36174, opened 2026-08-24, closed stale 2026-10-28). An open PR #34591 fixes "deterministic DSV4 top-k fallback on ROCm", so partial work exists.

**What follows for us:** no upstream stack gives the property we want, which is 1-row bit-equal to the reference *and* 6-row rows equal to 1-row. We get it only if the M=1 kernel *is* the reference's kernel, or if the oracle adopts the invariant kernel.

### 4d. Options for our two sites

| # | option | sites | row-invariant? | numerics vs today's reference (cuBLAS M=1) | modeled cost, 1 row / 6 rows | risk |
|---|---|---|---|---|---|---|
| **A** | **Own fixed-order fp32 FFMA kernels** (CUDA-core, `enable_fp_fusion` controlled, no `tl.dot`) for `mixes = F.linear(xf, hc_fn)` and for the head. The head reads the **bf16 checkpoint weights** (the fp32 head is an exact widening: `model.py:1006` "bf16 in the checkpoint, kept as fp32"; x is bf16-exact if the final norm outputs bf16, to verify), so every product is exact in fp32 and only the sum order differs. | both | yes, by construction (one program per output tile, sequential K, no M-dependent split) | **1-ulp class** (order only). **Bit-identical** if the oracle process calls the same kernel (policy amendment). | Head: 2.65 → 1.32 GB read, **≈0.19–0.25 ms at 1 and 6 rows** vs 0.42 ms measured at 1 row. Also frees 1.32 GB HBM if the fp32 copy is dropped. Mixes: 2 MB fp32 weight per call, ~2–3 µs/call; it can ride in the hc_norm launch, so ±0 launches at 1 row; at 6 rows it replaces 80 cuBLAS calls. Net **−0.15 to −0.25 ms** at both row counts [Mod]. | Low. It needs a Triton/CuTe kernel and a census entry. The FFMA throughput at 6 rows for the head (7.9 GFLOP) must not go compute-bound; probe. |
| **A′** | Same as A, but the kernel **reproduces cuBLAS's M=1 reduction order** (calibrated on device, the way our glue calibrates torch's sum orders and GEMV split partials, `fused_glue_r2.py:41-45`) | both | yes | **Bit-identical to today's reference** with no oracle change | as A | Unknown cuBLAS M=1 kernel internals: the head is "one cuBLAS gemv" [M], the mixes kernel is unknown. Feasibility is unverified; budget one probe before committing. |
| **B** | **cuBLAS batch-invariance mode.** `CUBLAS_BATCH_INVARIANCE_FLAGS=8` (AUTO) or `=4` (ALIGN_16B) set before cuBLAS init, in **both** the runtime and the oracle | both (plus any other cuBLAS GEMM) | yes [C, B200] | 1-ulp class vs today's oracle. Bit-identical if the oracle also runs with the flag. | Head ≈ today's 0.42 ms at 1 row, roughly the same at 6 (split-K off; N is wide). Mixes ≈ today. Net ±0 [Mod]. | Not in our cuBLAS (P1). It needs a cuBLAS ≥ 13.8 overlay and a compatible driver. **Silent fallback** for unsupported combos under the env var, so the census must re-verify. Experimental. Odd-width caveats [C]; N=24 is not odd. |
| **C** | **Fixed M**: call both sites with M padded to 8 (zeros) in the 1-row and 6-row graphs | both | Only if cuBLAS is batch-*position*-invariant within one shape (TM: true for most libraries, false for stream-K) | 1-ulp class vs today's oracle (M=1 now runs a different kernel than the reference's M=1) | Head: possibly *slower* at M=8 than the gemv; mixes ±0 | Unverified for fp32 on SM103; probe P3. Cheapest to try (no new kernel). |
| **D** | `row_invariant_mode` (per-row cuBLAS calls, already in `runtime/row_census.py`) | both | yes [M] (24/24) | **Bit-identical to the reference** | Head 6 × 0.42 = **2.5 ms at 6 rows**; mixes 80 × 5 extra calls ≈ 1.2–1.6 ms. **+3.5–4 ms at 6 rows** [Mod] | Debug only. It confirms the diagnosis and is not a serving option. |
| **E** | vLLM's Triton batch-invariant matmul for fp32 | both | yes | **A1 (TF32)** unless `input_precision="ieee"` | slower than cuBLAS [C] | Reject. |
| **F** | Accept the drift (A3 amendment: 6-row verify drift-graded) | — | no | A3 | 0 | 1/24 argmax disagreement on teacher-forced positions [M]. That breaks "emitted stream == 1-row greedy", which DSpark exact-match acceptance relies on (W2). Reject for the bit-exact instrument. |

**Recommendation:**
1. Run **A′** as a one-probe feasibility check.
2. Otherwise do **A** with the oracle adopting the same two kernels (a documented, reviewable policy amendment; the arithmetic stays IEEE fp32 dot products of exactly the reference's operands).
3. Keep **B** as the zero-code fallback once a ≥ 13.8 cuBLAS can be overlaid.

The head half of A is also a speed win (≈0.2 ms), which overlaps W3. W3 should treat "bf16-weight head with fp32 FFMA accumulation, fp32 logits" as 1-ulp class, **not** A1: the products are exact. A bf16 *output* or bf16 *accumulation* would be A1.

### 4e. Other row-invariance hazards found upstream

- **DeepGEMM determinism flag.** Keep `use_deterministic_algorithms(True)` and assert it in every process: runtime, oracle, census. With it off, DG's bf16 GEMM/einsum go to cuBLASLt and become batch-variant, and Mega mHC/Mega Gate split-K varies with M (§2a). The census covered `masked_m 6 vs 1` only. Our fast path is DG **contiguous in place** (dg-inplace), which the census does not list (probe P6).
- **FlashMLA sparse decode** (if we ever leave `sparse_prefill_fwd`) uses M-dependent split-KV. The upstream `enable_batch_invariant` (2026-09-30) is not in our `.so`, so we would need our own fixed `num_splits`.
- **Triton `tl.dot` on fp32 defaults to TF32** in any kernel we write or borrow. Use `input_precision="ieee"` or FFMA explicitly.
- **cuBLAS env vars are read at handle/workspace creation** (vllm-omni#5512 design note; vllm#65569 author: "whether they take effect may depend on when cuBLAS initializes"). Set them before `import torch` touches CUDA. This matters for option B.

## 5. Numbers table

| # | quantity | value | label | source |
|---|---|---|---|---|
| 1 | Step-level 6-row vs 1-row, ref impls | 0/24 bit-equal, max\|Δlogit\| 11.99, argmax 23/24 | M | `docs/rows-gpu-test-r2-20261107.log:86-90` |
| 2 | Same, with per-row cuBLAS (`row_invariant_mode`) | 24/24, 0, 24/24 | M | same |
| 3 | mHC mixes fp32 F.linear, 6 vs 1 | 86.8 % elements differ, max 224 ulp | M | same:76 |
| 4 | Head fp32 F.linear, 6 vs 1 | 69.2 %, max 20,480 ulp | M | same:78 |
| 5 | Head cuBLAS gemv at 1 row | 420 µs over 2.65 GB (≈ 6.3 TB/s) | M | `step-anatomy-r2-20261107.json` via CONTEXT |
| 6 | Head weight bytes, fp32 vs bf16 | 2.648 GB vs 1.324 GB (129,280 × 5,120) | arithmetic | `config.json` |
| 7 | Mixes per step | 80 calls, [M,20480]×[24,20480]ᵀ fp32, 1.97 MB weights each | arithmetic | `model.py:940-956`, `config.json` |
| 8 | fp32 cuBLAS mm batch variance on H100, K=N=4096, M=1 vs 2..64 | 3,606–3,972 of 4,096 elements differ | C | vllm#65569 (2026-11-06) |
| 9 | cuBLAS 13.8.2 BI mode, B200, 25 shapes, M 1–8192 | 0 of 1,460,100 rows differ, fp32 included | C | vllm#65569 (2026-11-06) |
| 10 | cuBLAS BI mode availability | CUDA 13.4.2 / cuBLAS 13.8.0.4 (wheel 2026-09-16) | C / R | vllm#65569; PyPI |
| 11 | vLLM BI-mode decode cost after #57586 | 1.04–1.07× vs BI off (Qwen3 1.7–8B, RTX PRO 6000) | C | vllm#57586 |
| 12 | TM batch-invariant matmul cost | ~20 % below cuBLAS; Qwen3-8B serving 26 s → 42 s | C | TM blog 2025-09-10 |
| 13 | DeepGEMM Mega mHC / Mega Gate | 45–85 % / 30–75 % faster than unfused | C | DeepGEMM#432 |
| 14 | Mega mHC replay race before #478 | 185/200 wrong replays (33 tok, contended); 1/200 (512 tok, idle) | C | DeepGEMM#478 |
| 15 | FlashMLA mega attn vs split-KV, GB300 TP1, per layer incl. all ops | 29/29 µs at s_q=1; 31/31 at 32; 31/40 at 128 | C | `flash_mla_mega_attn.py:72-79` (vendored v0.31.0) |
| 16 | FusedWoAKernel replaces 3 kernels of | 11–12 µs per layer (GB200, TP4, decode) | C | vllm#58634 |
| 17 | vLLM GB300 latency-specialized V4.1 decode, TP4 DSpark k=5, step incl. draft | 5,355 → 4,375 µs at batch 1; 10,598 → 9,151 at 8; 17,483 → 15,516 at 32 | C | vllm#61868 |
| 18 | Its routed-expert kernels at 1 / 6 rows | front 3.04/4.06, route 2.18/1.95, GEMM1 9.76/17.60, GEMM2 6.24/10.37 µs | C | vllm#61868 |
| 19 | Its production step size | "~1,600 kernels"; "~105 µs/layer serial (50 attn + 53 MoE)" | C | vllm#61868 |
| 20 | FlashMLA fused kernel saving for us | 0.08–0.15 ms at 1 row; 0.5–1.0 ms at 6 rows (the latter also reachable bit-exactly via multi-row glue) | Mod | §3b |
| 21 | Option A head saving | ≈ 0.2 ms at 1 and 6 rows (1.32 GB at ~6–7 TB/s) | Mod | §4d |
| 22 | Option D cost at 6 rows | +3.5–4 ms | Mod | §4d |

## 6. What this means for our runtime

1. **Fix the record.** Update M1-T10-RESULT's "only non-invariant kernel is the fp32 head" to **"mixes (80/step) + head"**. Recompute the R6/R1 and parity plans with both sites in scope. The DeepGEMM pin in WORKERS.md should read vllm-project/DeepGEMM `e1f418c2` (2.8.0), not `88965b07`.
2. **Mixes and head, the bit-exact 6-row instrument.** Do option A′/A (§4d):
   - New kernels: `runtime/kernels/` gets one fixed-order fp32 dot kernel parameterised for (N=24, K=20480) and (N=129280, K=5120, bf16 weights).
   - The mixes call sites move off `F.linear`: `fused_glue_r2.py:1516-1518` and the reference-op path used at rows > 1 (the static step's `hc_mixes`).
   - The head moves off `F.linear` at `static_step.py:1008-1015`.
   - Numerics class: **bit-identical** (A′, or A with the oracle amended), else **1-ulp class** (A).
   - Expected: head 0.42 → ~0.2 ms at 1 row and ~0.2–0.3 ms at 6 rows; mixes ±0 to −0.05 ms [Mod].
   - Gate: the census rows "mHC mixes" and "head" flip to `yes`, and the step-level row "fast, GEMV pinned" goes to 24/24.
3. **Keep DeepGEMM deterministic** and extend the census to the dg-inplace contiguous path (P6). Do **not** pull Mega mHC from the image without overlaying DeepGEMM #478. It is a JIT-header-only fix (3 files: `impls/sm100_mega_mhc.cuh`, `layout/mega_mhc.cuh`, `scheduler/mega_mhc.cuh`), so the overlay is a header copy. Mega mHC's numerics against the reference's fp32 path are unverified: its pre-norm GEMM family is `tf32_hc_prenorm_gemm`, so the A1 risk is unverified until probed.
4. **FlashMLA fused kernel: deprioritise.** It is A3 at best and worth ~0.1 ms at 1 row. First extend our own `o_rope_qdq` and `q_rope_epilogue` clusters to R rows with per-row freqs. That is bit-identical and recovers the 6-row ~0.5–1.0 ms the fused kernel would. Revisit only if the attention launch remains on the critical path after M2.
5. **Steal ideas, not code, from vllm#61868.** Its executor is the most direct public evidence of what a 1–8-row V4.1 decode on GB300 costs:
   - ~4.4 ms per DSpark step, but on TP4 with an all-reduce per boundary.
   - Its CuTe-DSL MoE kernels (≈21 µs per layer for 4 kernels at 1 row, TP4-sharded).
   - The FlashInfer low-latency MXFP8 GEMMs. These are already in our FlashInfer 0.7.0.post1 as py/CuTe DSL and are the candidate W3 should bench against our Triton GEMV.

   Its head is **bf16** on FlashInfer's CuTe-DSL GEMM. Production's logits are therefore not fp32-reference logits; expect near-tie flips between production and us that are no bug of ours.
6. **Production-parity caution.** Pg61 is built on the vLLM 0.31 image, which carries two kernel races fixed upstream later:
   - Mega mHC under graph replay (DeepGEMM#478);
   - the FlashMLA sparse-decode head64 `rowwise_max_buf` race (FlashMLA#241, vllm FlashMLA#39).

   Whether Pg61 executes either is unverified. Both are rare, but they are a reason not to treat a single production-vs-us divergence as evidence against our runtime without a rerun.

## 7. Box probes (each ≤ 30 min)

- **P0, fast-path diagnosis.** Run `tests/test_rows_gpu.py`'s step-level V0 on `fast, GEMV M=1 cfg pinned` **under `row_invariant_mode`**.
  - Pass if 24/24: then the mixes and head are the whole gap on the fast path too.
  - If not 24/24: bisect with `DSV41_GLUE_DISABLE`.
- **P1, cuBLAS version and BI mode.** Inside the image:
  1. Print the cuBLAS version via `ctypes` (`cublasLtGetVersion`) and `pip show nvidia-cublas` (or whatever wheel torch pulled).
  2. Rerun the census fp32 families with `CUBLAS_BATCH_INVARIANCE_FLAGS=8` and with `=4`, exported before Python starts.
  3. If the version is < 13.8, `pip download nvidia-cublas==13.8.2.12` into a scratch dir and `LD_PRELOAD` its `libcublasLt.so.13` and `libcublas.so.13`. Re-run, and record whether the driver accepts it.
  4. Time the head at M=1 and M=6 under BI.
  - Pass: mixes and head `yes`, head ≤ 0.45 ms.
- **P2, fixed-order FFMA kernels (option A/A′).**
  - Write a Triton kernel with `enable_fp_fusion=False` and explicit `fma` in a fixed K order: one program per (row-tile ≤ 8, N-tile), sequential K, no split.
  - Measure: (i) row invariance at M=1..8 and row-position invariance; (ii) ulp vs cuBLAS M=1, max ulp and mismatch %; (iii) graph-timed µs for the head (bf16 weights) and the mixes at 1 and 6 rows.
  - A′ add-on: try three candidate orders (sequential; 2-, 4- and 8-way K-split with pairwise combine; warp-shuffle tree over 32 lanes × K/32 chunks) against cuBLAS M=1 bits for the mixes. Report the best match rate.
- **P3, fixed-M padding (option C).** Run F.linear fp32 at M=8 with the real row at index 0, 3 and 7 (zeros elsewhere), against M=8 with 6 real rows.
  - Pass: all bit-equal per row, for both mixes and head.
  - Also time the head at M=8 vs M=1.
- **P4, FlashMLA fused fwd** (only if item 4 of §6 is revisited).
  - Permute `wq_b`/`wo_a` on one layer and call `fused_norm_rope_attn_rope_cast_fwd` on our bf16 cache with int32 positions and a `cos_sin_cache` built from `freqs_cis.real/imag`.
  - Compare the dequantized output vs our chain (FlashMLA `sparse_prefill_fwd` → `o_rope_qdq`): % of fp8 codes differing, max ulp.
  - Also check Q-RoPE bit-equality by setting `sm_scale` so attention is trivial, or compare lse.
  - Time it in a 40-call graph at 1 and 6 rows vs today's 9–15 µs + glue.
- **P5, image hazard check** (read-only `grep` in the image; also on the Pg61 image if the operator agrees):
  - `grep -c gridid …/deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh`: nonzero means #478 is not applied.
  - Whether Pg61's DSV4.1 path calls `deep_gemm.mega_mhc` and `sparse_decode_fwd` head64 under graphs: log-level evidence or `nsys` kernel names.
- **P6, DeepGEMM census completion.** Add `m_grouped_fp8_fp4_gemm_nt_contiguous` in-place (our dg-inplace shapes) at 1..8 tokens per expert to `row_census.py`, with `use_deterministic_algorithms(True)` asserted via `deep_gemm.get_…`. If no getter exists, assert the call happened at build. Run it with the flag **off** too, to see whether anything we call is affected.

## 8. Sources (read 2026-11-07 unless noted)

- vLLM build pins:
  - `cmake/external_projects/deepgemm.cmake` and `flashmla.cmake` at tags v0.31.0 / v0.32.0 / v0.33.0rc3 / main, and `tools/install_deepgemm.sh@v0.31.0`: https://github.com/vllm-project/vllm (via `gh api contents?ref=…`)
  - `requirements/cuda.txt@v0.31.0`
- vLLM releases v0.31.0 (2026-10-05), v0.32.0 (2026-10-18), v0.33.0rc3 (2026-11-02): https://github.com/vllm-project/vllm/releases
- vLLM PRs (merged dates in §2c/§4c):
  - #61868 https://github.com/vllm-project/vllm/pull/61868 (2026-10-21)
  - #58634 (09-27), #60029 (10-06), #60118 (10-19), #56148 (10-13), #60271 (10-08), #61118 (10-17), #59985 (10-13), #57586 (09-23)
- vLLM issue #65569 (opened 2026-11-04; comments through 11-06): https://github.com/vllm-project/vllm/issues/65569
- vLLM batch invariance code and docs: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/determinism/batch_invariant.py ; https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md
- DeepGEMM:
  - commits and PRs #304, #347, #364, #377, #432, #462, #471, #478: https://github.com/deepseek-ai/DeepGEMM (dates in §2a)
  - fork https://github.com/vllm-project/DeepGEMM (compare 88965b07…e1f418c2)
  - `csrc/apis/{gemm,einsum,mega_mhc,config}.hpp`, `csrc/jit_kernels/heuristics/{runtime,mega_gate}.hpp`, `csrc/jit_kernels/impls/smxx_cublaslt.hpp` at 66081d4c / 5081f851
- FlashMLA:
  - upstream PRs #221 (2026-09-10), #229 (09-30), #230 (10-15), #241 (10-27): https://github.com/deepseek-ai/FlashMLA
  - fork PRs #18, #21, #23, #24, #30, #38, #39, #41: https://github.com/vllm-project/FlashMLA
  - README §fused kernel; `tests/test_fused_norm_rope_attn_rope_cast.py@0eee43b1`; `flash_mla/flash_mla_interface.py@main` (`enable_batch_invariant`)
- FlashInfer releases v0.7.1, v0.7.2, v0.7.2.post1 (10-23), v0.7.3 (11-04), and the tree at v0.7.0.post1: https://github.com/flashinfer-ai/flashinfer
- cuBLAS documentation, cuBLAS 13.5 docs, "Last updated on Nov 02, 2026": §2.1.4 Results Reproducibility, §3.1.5 Experimental: Batch Invariance, §3.3.8 `cublasLtMatmulBatchInvarianceFlags_t`: https://docs.nvidia.com/cuda/cublas/index.html
- PyPI `nvidia-cublas` release history: https://pypi.org/project/nvidia-cublas/ ; `nvidia-cutlass-dsl`: https://pypi.org/project/nvidia-cutlass-dsl/
- Triton `tl.dot` (`input_precision` default `"tf32"`): https://triton-lang.org/main/python-api/generated/triton.language.dot.html ; releases https://github.com/triton-lang/triton/releases (3.8.0 2026-08-28, 3.9.0 2026-10-27)
- TileLang releases (0.1.13–0.1.15): https://github.com/tile-ai/tilelang/releases
- Thinking Machines, "Defeating Nondeterminism in LLM Inference", 2025-09-10: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ; https://github.com/thinking-machines-lab/batch_invariant_ops
- SGLang deterministic inference: https://docs.sglang.io/docs/advanced_features/deterministic_inference ; issue https://github.com/sgl-project/sglang/issues/36174 (2026-08-24 → closed 2026-10-28)
- Local, read-only:
  - `reference/vllm-0.31.0/{deep_gemm,flashmla,deepseek_v41,utils_deep_gemm.py}`
  - `reference/deepseek-inference/{model.py,config.json}`
  - `runtime/{attn_flashmla,fused_glue,fused_glue_r2,row_census,static_step}.py`
  - `docs/{ENVIRONMENT.md,M1-T10-RESULT-20261107.md,rows-gpu-test-r2-20261107.log,flashmla-probe-20261106.json}`

W4: Our 6-row step is non-invariant at two cuBLAS fp32 sites (80 mHC-mix calls + the head, not the head alone), and no upstream stack fixes that for V4.1. The fix is a fixed-order fp32 FFMA kernel for both, ideally calibrated to cuBLAS's M=1 order or adopted by the oracle; the head half also saves ~0.2 ms by reading the exact bf16 weights. cuBLAS 13.8's experimental batch-invariance mode is the zero-code fallback. Upstream offers no free 6-row speed: our image already has the useful DeepGEMM/FlashMLA/FlashInfer pieces, plus two races fixed upstream since (Mega mHC replay, FlashMLA head64 decode) to avoid or overlay.
