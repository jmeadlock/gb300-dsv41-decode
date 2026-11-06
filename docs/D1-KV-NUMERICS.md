# D1: compressed-KV and indexer numerics, reference vs vLLM `fp8_ds_mla`

T0 desk memo, 2026-11-06. This is a read of the source only: `reference/deepseek-inference/` (1bae0e10) and `reference/vllm-0.31.0/deepseek_v41/` (the stock v0.31.0 image). There was no box access, and nothing was executed except one local arithmetic check. Cite prefixes: `ref/` = reference/deepseek-inference, `v/` = reference/vllm-0.31.0/deepseek_v41.

## Answer

**Compressed KV.** vLLM does not apply the reference's FP4 QDQ to the compressed latent under `fp8_ds_mla`. It takes the bf16 latent, applies RoPE in fp32, and quantizes straight to MXFP8 (e4m3, one UE8M0 scale per 32 values). The result goes into the same 528 B record the window uses (v/attention.py:502-508; v/common/ops/fused_compress_quant_cache.py:354-405).

The reference does something else. It attends E2M1 values with one E4M3 scale per 16 (ref/model.py:759-762; ref/kernel.py:160-171). So the attended compressed values differ on every compressed layer (2–39). That includes the W0 fixture, where every compressed entry is attended.

**Window KV.** The transform is the same.

**Index K/Q.** These depend on `dsa_indexer_uses_fp4`, and its default is not in the vendored source. On the W0 fixture neither engine runs the scoring (both select every position), so index numerics cannot move the fixture.

| Tensor | Reference transform | vLLM `fp8_ds_mla` transform (Pg61 argv) | Equal attended values? | Cites |
|---|---|---|---|---|
| **Window KV** (all layers; 128-slot ring) | 1. `kv_norm(wkv(x))` in bf16. 2. RoPE on the last 64 dims, fp32 math written back to bf16. 3. FP8 QDQ over all 512 dims: block 32; scale = 2^ceil(log2(max(amax,1e-4)/448)) via the bit-exact `fast_log2_ceil`; clamp ±448; cast to e4m3; multiply by the scale back to bf16. The QDQ values are what gets attended. | 1. Fused RMSNorm. 2. C++ op `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(kv_mxfp8=True)` does RoPE + MXFP8 into the 528 B record (512 e4m3 + 16 UE8M0 per 32) on capability family 100. 3. Decode: FlashMLA reads the fp8 record. Prefill: dequant-gather to bf16. | **Yes** in format and transform. **Cannot tell** at the bit level: the C++ encoder is not vendored (csrc), so I can't see whether the RoPE output is rounded to bf16 before the scale is picked, or how ceil-log2 is lowered. | ref model.py:701-721, 396-405; kernel.py:22-37, 74-86 · v attention.py:119-126, 502-503, 1019-1048; cache_utils.py:43-44, 486-499; flashmla.py:230-246, 315-325 |
| **Compressed KV latent** (sources 2/8/14/20, read by layers 2–39) | 1. Compressor latent: ratio 2 = fp32 gated pool; ratio 1 = bf16 projection. 2. RMSNorm, to bf16. 3. RoPE on the last 64 dims at the group's first position, written back to bf16. 4. **FP4 QDQ**: E2M1, block 16; E4M3 scale = RN_e4m3(max(amax, 6·2⁻⁹)/6); x/s clamped ±6; multiply by the scale back into the bf16 cache. The QDQ values are what gets attended. | 1. `fused_save_compress_norm` produces the bf16 latent. 2. `rope_quant_insert` dispatches on record width to the 528 B kernel. 3. RoPE in fp32, with no bf16 round. 4. **MXFP8**: block 32; UE8M0 scale = 2^ceil(log2(max(amax,1e-4)/448)) via `tl.log2`; clamp ±448; cast to e4m3. There is no FP4 step on this path. | **No.** The 528 B record can't even hold the reference's values losslessly: 15 of the 56 E2M1×E4M3 significand products need more than e4m3's 4 bits. vLLM doesn't try. | ref model.py:458-485, 752-764, 776-781; kernel.py:155-171 · v compressor.py:273-291, 315-323; fused_compress_quant_cache.py:215-221, 254-259, 354-405; attention.py:506-508 |
| **Index K** (owners 2/8/14/20) | 1. `k_norm(wk(latent))` on the pre-RoPE, pre-QDQ latent, in bf16. 2. RoPE, to bf16. 3. FP4 QDQ: E2M1, block 32; E8M0 scale = 2^ceil(log2(max(amax, 6·2⁻¹²⁶)/6)), bit-exact; clamp ±6; result into the bf16 `k_cache`. | Same input (pre-RoPE latent, `wk`, `k_norm`). Then it branches on `use_fp4_kv = dsa_indexer_uses_fp4(...)`. **MXFP4:** the reference's steps (bf16 round after the norm and after RoPE, "for parity with the reference"; same amax floor; pow2 scale via `tl.log2`; multiply by the inverse), packed by `_fp32x2_to_fp4x2` into a 68 B row. **FP8:** one 128-wide block per token, pow2 fp32 scale, e4m3, 132 B row. | **Cannot tell.** The `indexer_kv_dtype` default lives in `vllm/v1/attention/backends/mla/indexer.py`, which is not in the vendored source. Pg61's argv sets no `--attention-config`. If MXFP4: same transform, bits unverifiable (`_fp32x2_to_fp4x2` is not vendored). If FP8: **no**. | ref model.py:537-547; kernel.py:164-171 · v attention.py:89-97, 1251-1255, 1349-1381; indexer_k_store.py:158-189, 210-255; common/ops/__init__.py:12-18; w0/results/ds41-args.json |
| **Index Q** (2/8/14/20/24–36) | 1. `wq_b` fp8_linear. 2. RoPE, to bf16. 3. FP4 QDQ, E2M1 with E8M0 scale per 32. 4. Scores in bf16: einsum, relu, weighted head sum. | 1. `fused_indexer_q_rope_quant(use_fp4=use_fp4_kv)`. 2. Scoring by `SparseAttnIndexer` / DeepGEMM. | **Cannot tell.** Neither the Q quant (`vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`) nor the scorer (`vllm/model_executor/layers/sparse_attn_indexer.py`) is vendored. | ref model.py:551-558, 579-581 · v attention.py:1327-1342, 1383-1412 |

## Other findings

1. **vLLM itself names the reference format, and Pg61 opts out of it.** v/nvidia/model.py:228-235 calls the 288 B NVFP4 record "the format the reference implementation itself stores". Mega-attention makes it the SM100 default (v/nvidia/flash_mla_mega_attn.py:193-197). Pg61 overrides that with an explicit `--kv-cache-dtype fp8_ds_mla`. The `nvfp4_ds_mla` writer (fused_compress_quant_cache.py:408-461) has the same structure as the reference's QDQ: E2M1, E4M3 per 16, scale = clamp(amax/6, 2⁻⁹, 448) (equivalent to the reference's 6·2⁻⁹ floor), and round-to-nearest division. Three differences remain:
   - its RoPE dims are not rounded to bf16 before quantization;
   - its E2M1 rounding is inside `_fp32x2_to_fp4x2`, which is not vendored;
   - how the reference's TileLang lowers `x / s` is not visible.
2. **Short-context bypass.** When `max_seq_len // ratio ≤ 512`, vLLM skips scoring and selects every compressed position (v/attention.py:1426-1448). The reference's top-`min(512, T/r)` does the same (ref/model.py:579). At ≤310 tokens the index K/Q path is dead in both engines. Layer 20's candidate pruning also keeps everything.
3. **Correction to M0-PORT-PLAN §5(d) and charter decision 6.** The plan says decision 6's records are "exact encodings of the reference's QDQ'd values … compressed 288 B … index K 68 B". That is true of `nvfp4_ds_mla` plus the MXFP4 indexer. It is not true of `fp8_ds_mla`, whose compressed record is 528 B MXFP8 (v/attention.py:506-508). Decision 6 as worded ("`fp8_ds_mla` compressed store") cannot carry the reference's numerics exactly.
4. **Upstream of D1, but it feeds the latent.** vLLM computes the compressor projection as a single fp32-output `mm` at both ratios (v/attention.py:916-921). The reference rounds the ratio-1 projection to bf16 before the norm (ref/model.py:446, 461-462). The difference is small but nonzero on layers 20–39.
5. **Pg61 is not stock vLLM.** It runs Garner's engine plus our patches; the vendored tree is the stock image. Whether those patches touch these writers is not in the vendored source. Two lines in Pg61's boot log settle the live state:
   - `Using DeepSeek's %s KV cache format.` (v/attention.py:158)
   - `Using %s indexer cache for Lightning Indexer.` (v/attention.py:1252-1255)
6. **`query_quant.py` is not the index-Q quant.** It quantizes the main attention's `qr` (the `wq_b` input) to MXFP8. It has no 1e-4 amax floor (query_quant.py:54-64), where the reference's `act_quant` does (kernel.py:76). This only differs on 32-blocks whose values are all near zero.

## Recommendation for M0

**(a) To match the reference (the model definition):**
- Window KV: FP8 QDQ as in the table.
- Compressed KV: E2M1 / E4M3-16 QDQ after a bf16 RoPE.
- Index K and Q: E2M1 / E8M0-32 QDQ, with bf16 index scores.

kernel.py:161 labels the compressed path "Training's compressed KV", so these are the numerics the weights were trained against.

**(b) To match Pg61 (the parity target):**
- Window KV: same as (a).
- Compressed KV: MXFP8/32 of the fp32-RoPE'd latent, with no FP4.
- Index K/Q: whatever the boot log says. This does not matter on the fixture.

**Can both be true?**
- Window KV: yes.
- Index K/Q: moot on the fixture. At long context it is possible, if Pg61's default turns out to be MXFP4.
- Compressed KV: **no**, by construction.

**Which one can the parity instrument (§7) tolerate?**
- Self-replay (m1) is indifferent to the choice.
- m2 and m3 grade against Pg61 with a spread bar. Pass means zero confident flips (p₁ ≥ 0.9, i.e. a top-2 logit gap ≥ ln 9 ≈ 2.2), and free-run divergences only where a margin is ≤ 0.25.
- Pg61-matching numerics are the certain pass, because by construction they give the smallest delta.
- Reference numerics pass only if the FP4-vs-MXFP8 difference on compressed entries never moves a confident position by ~2.2 nats. Per element, that difference is rounding of up to 20% (E2M1) against at most 5.9% (e4m3). Whether it ever adds up to 2.2 nats is an empirical question; the source cannot answer it.
- The instrument was built to tolerate exactly this kind of difference. Its margin classes exist so that a numerics change shows up as reported mid/near-tie flips rather than a gate failure. It stops tolerating the change at the first confident flip.

**Recommendation: make reference numerics M0's default and keep one switch.**

Why reference by default:
- The charter has M0 run the reference's math.
- Decision 4 forbids KV quantization as a lever. Swapping the model's FP4 QDQ for MXFP8 is exactly that kind of lever, pointed at Pg61.

The switch:
- `ckv_quant ∈ {ref_e2m1_e4m3_16, pg61_mxfp8_32}`.
- It applies at the compressed-KV write only: one kernel, and the bf16 cache stays as it is at M0 (§5(d)).
- At T6, run m2 under both settings.

How to read the T6 result:
- **Reference passes:** ship it. Amend decision 6 to the 288 B record for M2 packing.
- **Reference fails, Pg61 passes:** those confident flips measure Pg61 against the model, not a runtime bug. James picks.
- **Both fail:** the bug is somewhere else.

Before T6, read Pg61's two boot-log lines (finding 5) so the index K/Q row stops being "cannot tell".

D1-MEMO: different — under `fp8_ds_mla`, vLLM stores MXFP8 of the un-FP4'd latent, so Pg61 attends different compressed-KV values than the reference; window KV is the same; index K/Q is unknown from vendored source and moot on the W0 fixture.
