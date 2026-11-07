# HANDOFF — T9b-2 round 1: fuse the zero/low-risk glue clusters, bit-identical to today's static path

Coding worker for GB300 DSV4.1 Decode. State: the 1-row step is one CUDA graph at **32.16 ms** (`docs/T9B1-RESULT-20261106.md`), path `--mode graph --moe-impl dg-masked --fp8-impl gemv`. The op census in `docs/GLUE-AUDIT.md` §"The step today" counts ~5,300 one-microsecond glue launches per step (~5 ms) and ranks 16 fusable clusters. **This round takes only the clusters whose bit-identity risk is "none" or "low"**: #2 SwiGLU + act_quant (routed and shared; 600 launches), #7 mHC post (320), #8 MoE combine (320), #9 DG adapter copies (320), #10 o path inverse RoPE + QDQ (200), #11 one gather for all six staging tables (200), #13 q RoPE in the wq_b GEMV epilogue (120), #16 concat-free attention inputs (76). Target ≈ 2,150 launches removed, ≈ 2 ms. Medium/high-risk clusters (#1, #3–#6, #12, #14, #15) are **not** in scope; do not touch norms, the router, the indexer or Engram.

Read first: `docs/GLUE-AUDIT.md` (the cluster table rows above, their source lines, and the §"Notes for T9b-2"), `runtime/static_step.py` (`StaticMoE`, the attention/o-path code, where each cluster's ops live today), `runtime/dg_moe.py` (the DG adapter: `load_act`, scale converts, `ids_step` copy — cluster #9), `runtime/gemv_mxfp8.py` (cluster #13 is an epilogue option on this kernel; cluster #12 is out of scope), `runtime/static_kernels.py` (the existing Triton kernels: gather, marker; add new ones here), `reference/deepseek-inference/model.py` lines the audit cites for each cluster (the exact op order you must reproduce), `bench/glue_census.py` (re-run it after; the count is your receipt), `bench/first_light.py` (`--parity-levels`).

## Rules of the fusion
- Each fused kernel reproduces the **exact fp32/bf16 op sequence and rounding points** of the ops it replaces (e.g. SwiGLU: `gate.float()`, `up.float()`, clamps with `swiglu_limit` in the reference's order, `silu(gate) * up`, `* weight`, `.to(bf16)`, then `act_quant` per-32 e8m0 — same `silu` formulation as torch for bf16/fp32 inputs; if `tl.sigmoid`/`exp` differ from torch's `silu` at the ulp level, say so and keep that cluster unfused rather than approximating). MoE combine: sequential fp32 adds in slot order. mHC post: the 4-term fp32 sum in the reference's order. RoPE: torch's complex multiply order.
- Flag `--glue-impl {ref,fused}` (default `ref` = today's static path, unchanged bytes), threaded like `--moe-impl`/`--fp8-impl`.
- **Gate for this round: fused == unfused static path, fp32-logit sha256 at every position, 32 steps × 8 prompts** (a new parity-levels row "fused glue vs ref glue, same moe/fp8 impl"), plus L1/L2 8/8 under the graph. A cluster that cannot be made bit-identical is left unfused and named in the summary with the reason — no approximations in this round.
- Graph-capturable: static buffers, no host sync, Triton JIT warmed before capture.

## Deliverables (only `runtime/`, `bench/`, `tests/`)
1. `runtime/fused_glue.py` (Triton kernels + thin wrappers) and the call-site switches in `runtime/static_step.py` / `runtime/dg_moe.py` / `runtime/gemv_mxfp8.py` (epilogue flag).
2. `bench/first_light.py`: the new flag, the new levels row, and the report records `glue_impl`.
3. `bench/glue_census.py`: run on the fused path; print launches removed per cluster vs the audit's prediction.
4. Tests: `tests/test_fused_glue_cpu.py` (torch reference of each fused op sequence vs the unfused ops on random inputs incl. clamp edges and e8m0 boundaries; use the `tests/triton_shim.py` approach for kernel bodies where feasible), `tests/gpu/test_fused_glue_gpu.py` (each kernel `torch.equal` vs its unfused ops on real shapes; mini checkpoint: fused static-eager == ref static-eager logits over 10 steps; graph == static; both parities).
5. `runtime/README.md`: flag + box commands.

## Gates (orchestrator)
- GPU test green; first light `--mode graph --moe-impl dg-masked --fp8-impl gemv --glue-impl fused --parity-levels`: new row 8/8, L1/L2 8/8, output JSONL byte-identical to the T9b-1 run (`w0/results/first-light-cand-gemv.jsonl`); census shows ≥ 1,800 launches removed; **step cell ≤ 30 ms**.

No ssh/docker/network/commit; never import vllm; `reference/` untouched; no invented numbers (Mac has no GPU). Summary ≤ 12 lines ending `T9B2-R1: <ready-for-box|blocked> — <reason>`, listing any cluster left unfused and why.
