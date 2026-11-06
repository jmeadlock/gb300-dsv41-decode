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
