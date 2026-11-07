# T9a-1 — DeepGEMM masked grouped GEMM over the staging buffer: ABI and numerics, measured (2026-11-06)

`bench/dg_masked_spike.py`, `bench/dg_control_spike.py`, DeepGEMM's own tests vendored at `reference/deepgemm-tests/`. Receipt `dg-masked-spike-20261106.json`. `import deep_gemm` from `vllm/third_party` with no `vllm` module loaded (decision 3 intact).

## The working call (everything else fails silently or with assertions)
```python
dg.m_grouped_fp8_fp4_gemm_nt_masked((xq, xs_f32), (W_int8, S_f32), d, masked_m, expected_m, recipe_a=(1, 32), recipe_b=(1, 32))
```
- `W_int8`: `[G, N, K/2]` int8, packed e2m1, **low nibble first — exactly the checkpoint's bytes** (DeepGEMM's own packer reproduces 100 % of scales and the nibble order on a real expert; the 11 % code differences are its round-half-even vs the checkpoint's rounding on ties, not layout).
- `S_f32`: `[G, N, K/32]` fp32 = `exp2(e8m0 − 127)` of the checkpoint's e8m0 scales (exact). No `transform_sf_into_required_layout` (that is for the legacy `recipe=` form and asserts on `is_sfa`).
- `xq`: `[G, max_m, K]` fp8 from the model's own `act_quant(x, 32, "ue8m0")`; `xs_f32` its scales as fp32. **`max_m ≥ 128`** (BLOCK_M): with 64 or 8 rows, groups ≥ 1 are silently corrupted — the trap that cost three spike rounds. `recipe_a=(1,32)` is accepted and correct (DeepGEMM's tests only exercise `(1,128)`; `(1,64)` is rejected).
- `recipe=(1,1,32)` (the MegaMoE form) gives garbage here; the masked API wants `recipe_a`/`recipe_b`.

## Numerics vs the reference `fp4_gemm` (real expert bytes `layers.0.ffn.experts.*.w1/w2`)
| activations | bf16 outputs differing | rel err | max |Δ| |
|---|---|---|---|
| N(0,1) blocks (scales coincide) | **0 / 0 / 0 / 0** (w1, w2 × M=1, 6) — bit-identical | 0 | 0 |
| varied block magnitudes 2⁻⁶…2⁶, per-32 scales (the model's) | 0.036 % / 0.035 % / 0.010 % / 0.012 % | 1e-5–1e-4 | 1 bf16 ulp |
| same, DeepGEMM's per-128 scales | 0.19–0.22 % | 5e-5–1e-4 | 1 ulp |

So the block-scaled UMMA accumulates in the reference's order except at wide dynamic range, where ~1 in 3,000–10,000 outputs moves one bf16 ulp. That is the whole T9a-1 drift: below the A3 class by orders of magnitude. The `ref` path stays for parity; T9a-1 is graded by the first-light instrument like everything else.

## Speed
3 masked calls (w1, w3, w2 for 6 experts) ≈ **70–80 µs per layer** at M=1 and M=6 (the kernel itself ~22 µs; the rest is this harness's per-call quant/convert), vs 18 reference launches ≈ 1.4 ms. Per step: 40 × ~0.08 = **~3 ms** for the routed GEMMs vs 56 ms. With the spike's gather and the reference glue around it, the design memo's ~19 ms/step estimate for T9a-1 stands; the glue and the shared expert are what remains.

## What this changes in the plan
- T9a-1 is a one-day task after T8: replace the 18 `fp4_gemm` launches in the static MoE with 3 masked calls over the same staging buffer (`[6,N,K/2]` + fp32 scales kept resident per layer; `max_m=128` activation buffer per layer = 128 × 5120 B, trivial). Everything else in the static MoE stays.
- Bit-identity: not exact under wide-range activations (1 ulp, ~0.01–0.04 %). `--moe-impl ref` remains the parity path, as the design memo specified.

## Addendum — the reference's fp8 GEMMs through DeepGEMM `fp8_gemm_nt` (`bench/dg_fp8_spike.py`, receipt `dg-fp8-spike-20261106.json`)

Works with `a=(fp8 [M,K], fp32 [M,K/32]) recipe_a=(1,32)`, `b=(fp8 [N,K], fp32 [N,K/32]) recipe_b=(1,32)` where the weight scale is the reference's 32×32 e8m0 block scale expanded per row (`repeat_interleave(32, 0)`, exact); no `max_m` padding needed for this non-grouped API. `recipe_b=(32,32)` also runs but is 2× slower. Drift vs reference `fp8_gemm`: 0–0.2 % of outputs 1 ulp. **Speed: only 1.3–1.8× (35–48 µs vs 44–83 µs per call)** on real shapes (2304×5120 … 25600×6144) at M=1/6. Both kernels are tile GEMMs; at M=1 they are launch/latency-bound and read weights at ~300 GB/s against an 8 TB/s HBM. 290 such calls/step ≈ 16 ms is better attacked with a **bandwidth-bound MXFP8 GEMV** (M ≤ 8 rows, per-row K-block loop, fp32 accumulate in the reference's block order) — ~2–4 µs per call at 3–6 TB/s → the 16 ms becomes ~1–2 ms. That is T9b-1.
