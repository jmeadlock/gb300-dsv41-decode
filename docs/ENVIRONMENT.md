# Runtime environment (verified 2026-11-06)

Container: `vllm/vllm-openai:v0.31.0` used as a **library image only** (torch, TileLang, kernel libraries) — no vLLM engine in the process.

| component | version | note |
|---|---|---|
| torch | 2.13.0+cu130 | reference needs ≥2.10 |
| tilelang | 0.1.12 | exactly the reference's pin |
| safetensors | 0.8.0 | reference needs ≥0.7 |
| flashinfer | 0.7.0.post1 | borrowed-kernel donor |
| cutlass (CuTe DSL) | 4.7.1 | kernel language for the fused chain |
| triton | 3.7.1 | glue |
| transformers / tokenizers | 5.17.0 / 0.23.2 | tokenizer only |
| deep_gemm, flash_mla | not top-level modules | vendored inside vllm's tree; locate before borrowing |

GPU: NVIDIA GB300, compute capability **(10, 3)**, CDI device `nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55` (RTX PRO 6000 is `GPU-d07eb6e6-…`). Checkpoint: `/models/DeepSeek-V4.1-Flash-1017-1bae0e10…` on RAID0 NVMe.

## DeepSeek reference kernels on sm_103 — all 7 compile and run (`runtime/tools/kernel_smoke.py`, receipt `kernel-smoke-gb300-20261106.json`)

| kernel | result |
|---|---|
| `act_quant` (fp8, block 128, fp32 scales) | OK, dequant rel_err 0.026 (fp8 rounding) |
| `act_quant` ue8m0 scales | OK |
| `fp4_act_quant` (block 32, e8m0) | OK → `float4_e2m1fn_x2 [64,2560]` |
| `fp8_gemm` | OK, rel_err 0.037 vs fp32 torch (fp8 both operands) |
| `fp4_gemm` (fp8 act × packed fp4 weight, e8m0 scales) | OK, finite (structure test; numeric check needs real expert weights) |
| `hc_split_sinkhorn` | OK, rows/cols sum to 1 within 0 |
| `sparse_attn` (h=64, d=512, topk=512) | OK, rel_err 0.002 vs torch softmax reference |

First-compile cost: ~2–13 s per kernel; cached under `~/dsv41/w0/jit-cache/{tilelang,triton,nv}` on the box (mount into every runtime container). Two harness traps: `model.py` calls `set_dtype` so GEMM outputs allocate in the **default dtype** (set it to bf16 before calling); `fp4_gemm` wants e8m0 activation scales (`act_quant(..., block_size=32, scale_fmt="ue8m0", scale_dtype=float8_e8m0fnu)`), not the fp32 ones.

## T1 remainder — model-exact configs on real checkpoint bytes (`runtime/tools/kernel_smoke2.py`, receipt `kernel-smoke2-gb300-20261106.json`)

| case | result |
|---|---|
| `fp8_gemm` block-32 + E8M0 scales, real `layers.0.ffn.shared_experts.w1` [2304×5120], M=1 / M=6 | rel_err 0.0017 / 0.0016 vs fp32 dequant reference (= bf16 output rounding) |
| same on real `layers.1.engram.wkv` [25600×6144], M=1 | rel_err 0.0017 |
| `fp4_gemm` on **real routed expert** `layers.0.ffn.experts.0.w1` (I8 [2304,2560] viewed as `float4_e2m1fn_x2` + E8M0 [2304,160]), M=1 / M=6 | rel_err 0.0017 / 0.0017 vs an exact e2m1 nibble-unpack fp32 reference; scales span e8m0 119–122 (2⁻⁸…2⁻⁵) |
| in-place QDQ: window fp8/32 e8m0, index-K fp4/32 e8m0, latent fp4/16 e4m3 | all three run in place and change the tensor (max diffs 0.11 / 0.5 / 0.34 on N(0,1) input) |
| `F.embedding` / `index_select` on `float8_e4m3fn` and `float8_e8m0fnu` | both work natively on torch 2.13 — Engram's gather path needs no uint8 fallback |

## T0 — prompt rendering (`runtime/tools/render_fixture.py`, receipt `fixture-render-20261106.json`)
The checkpoint's own `encoding/encoding.py` in `thinking_mode="chat"` + `tokenizer.json` reproduces Pg61's `usage.prompt_tokens` **exactly on all 8 W0 fixture prompts** (42/47/54/38/34/49/46/31). Thinking mode adds a 26-token effort prefix. Pg61's overlay encoder and the checkpoint encoder agree for this fixture.
