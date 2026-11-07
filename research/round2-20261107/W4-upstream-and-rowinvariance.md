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

It is larger at 6 rows, because both glue clusters are **n/a at rows > 1** ("one position's freqs row per launch", `fused_glue.py:117-120`) and the reference op chains run instead. Those chains are ~6–8 launches per layer, which models to **0.5–1.0 ms [Mod]**.

The cheaper and **bit-identical** way to recover that same 6-row saving is to give our own `o_rope_qdq` / `q_rope_epilogue` a per-row freqs table. Do that first.

vLLM's own GB300 measurement shows the mega kernel tying its split path at s_q=1 (29 vs 29 µs per layer for the whole bundle) and winning only at s_q ≥ 128 [C]. Expect no kernel-time win at 6 rows beyond the launches it removes.

A related production kernel, **`FusedWoAKernel`** (vLLM #58634, in our image as py/CuTe DSL), fuses the *next* three steps: inverse RoPE + QDQ + `wo_a` + `wo_b`'s MXFP8 quant.
- Its RoPE uses explicit `mul.rn` + `fma.rn` ordering (`fused_wo_a.py:57-70`) [R]. That is our mode 1, so the RoPE part can be bit-exact.
- Its `wo_a` is a tcgen05 MXFP8 MMA with a fixed 8-way head split-K. Products are exact, the order differs, so it is 1-ulp class against the bf16 einsum.
- It is row-invariance-plausible (the split count is fixed by layout, not by M) but **unverified**.

