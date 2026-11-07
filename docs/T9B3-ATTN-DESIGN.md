# T9b-3 design: the attention + indexer path at 1 and 6 rows

Desk memo, 2026-11-06 22:24 CST, written on the Mac from source and receipts only; nothing here ran. Cite prefixes:
- `ref/` = reference/deepseek-inference
- `v/` = reference/vllm-0.31.0
- `rt/` = runtime at commit 80d0f29. T9b-2 R1 is editing `rt/static_step.py` concurrently, so line numbers there may shift.

**Measured** means a receipt in `docs/`. **Est.** is my arithmetic from those receipts. Real config: 64 heads × 512, rope 64, window 128, 40 layers, ratios 2×0 / 18×2 / 20×1, index top-k 512 over 32×128 index heads (`ref/config.json`).

## 0. Two corrections

1. **The main attention's q is not QDQ'd.**
   - q is the bf16 `wq_b` output plus a bf16 RoPE (`ref/model.py:771-773`). `sparse_attn` loads it as bf16 (`ref/kernel.py:329, 356`).
   - The block's only QDQs: window KV FP8/e8m0-32 (708), compressed latent E2M1/E4M3-16 (761), attention output FP8/e8m0-32 (787), and the *indexer's* FP4 q/k (547, 553).
   - Consequence for §2: a kernel that quantizes q adds a quantization.
2. **The 75 µs `sparse_attn` figure was taken at short context and does not describe today's static step.**
   - The anatomy (`STEP-ANATOMY-M0-20261106.md:13`) timed the eager reference. Its index width is `128 + min(512, T/r)` (`model.py:579, 778-779`), which on its prompt is 2–5 blocks of 64.
   - The static step always passes 640 indices (10 blocks) on compressed layers (`rt/static_step.py:428-440, 474-478`).
   - The kernel walks blocks serially in one CTA (`kernel.py:325, 358`). So today's cost is probably **100–150 µs/call (est.), 4–6 ms/step, not 3.0**.
   - Nobody has timed it under the graph; the T8 node trace was abandoned (LEDGER 19:41). It is the first number §5 measures.

## 1. What the block does per layer

### 1.1 Op list: compressed consumer layer, today's path (graph, dg-masked, gemv); M = 1 or 6 rows

| # | op (source) | shapes / dtypes | launches | µs/layer | bound |
|---|---|---|---|---|---|
| 1 | `wq_a`: act_quant(x) + GEMV (`model.py:771`) | x bf16 [M,5120] → bf16 [M,1280]; W fp8 6.2 MiB | 2 | 9.1 GEMV **meas.** (M=6: 9.0) + 2.6 | BW |
| 2 | `q_norm` (288-293) | fp32 stat over 1280 → bf16 qr | 8 | ~8 est. | latency |
| 3 | `wq_b`: act_quant(qr) + GEMV (772) | → q bf16 [M,64,512]; W 40 MiB | 2 | 14.8 **meas.** (M=6: 15.5) + 2.6 | BW |
| 4 | q RoPE, dims 448–511 (773, 392-406) | fp32 complex mul → bf16 | 3 | ~3 est. | latency |
| 5 | `wkv`: act_quant(x) + GEMV (706) | → kv bf16 [M,512]; W 2.5 MiB | 2 | 8.8 **meas.** + 2.6 | latency |
| 6 | `kv_norm` (706) | stat over 512 | 8 | ~8 est. | latency |
| 7 | kv RoPE (707) | | 3 | ~3 est. | latency |
| 8 | kv FP8 QDQ (708; `kernel.py:40-95`) | 16 blocks × 32, ue8m0, in place | 2 | ~3.6 est. | latency |
| 9 | ring write (719; `static_step.py:472`) | `index_copy_` slot pos % 128 | 1 | ~1 | latency |
| 10 | `cat` window ‖ compressed (778; `static_step.py:477`) | bf16 [128 + W_r, 512]; W_r = 2048/4096 at max_pos 4095 | 1 | ~1–3 est. (∝ max_pos) | BW |
| 11 | `cat` indices (779) | int32 [M,640] | 1 | ~1 | latency |
| 12 | `sparse_attn` (781; `kernel.py:310-403`) | q bf16 [M,64,512], kv bf16, idx int32 [M,640], sink fp32[64] → o bf16 | 1 | 75 **meas.** at 2–5 blocks; **100–150 est.** at 10 | latency: 1 CTA per row |
| 13 | o inverse RoPE + `conj` clone (782) | | 4 | ~4 est. | latency |
| 14 | o FP8 QDQ (786-787) | [M,8,4096], 128 blocks | 2 | ~3.6 est. | latency |
| 15 | `wo_a` einsum (788-789) | o bf16 × W **bf16** [8,1024,4096] = 64 MiB → [M,8,1024] | 1 | ~14 (anatomy: `nvjet_sm103…` 40 × 14.3 µs; attribution by launch count) | BW |
| 16 | `wo_b`: act_quant + GEMV (790) | [M,8192] → [M,5120]; W 40 MiB | 2 | 19.4 **meas.** (M=6: 20.1) + 2.6 | BW |
| | **total** | | **43** | **~190–240 est.** | |

GEMV times are from `gemv-bench-r2-20261106.json`; act_quant is the anatomy mean; glue is ~1 µs per launch (`GLUE-AUDIT.md:66`). Ratio-0 layers 0–1 drop rows 10–11 and attend 128 indices (2 blocks).

**Census reconciliation.**
- 43 launches/layer = the audit's norms 16 + Linears 8 + RoPE 10 + body 9 (`GLUE-AUDIT.md:87-91`).
- act_quant per step: 6 × 40 + 8 indexer + 120 shared expert + 80 DG + 2 Engram = **450**.
- fp4_act_quant per step: 8 indexer q + owner k and latent writes on 1 (even) or 4 (odd) layers each = 10 / 16, mean **13**.

Both match the census exactly.

**Block total (est.):** ≈ 2,000 of the step's 6,725 launches (1,720 per-layer + ~280 indexer/owner/compressor), and ≈ **8–10 ms of the 32.2 ms step**:

| item | ms/step |
|---|---|
| `sparse_attn` | 4–6 |
| GEMVs | 2.1 measured |
| `wo_a` | 0.6 |
| act_quant | 0.6 |
| glue | 1.3 |
| indexer | 0.4 |

**Bound by what.**
- Bandwidth-bound: only the weight reads (rows 1, 3, 5, 15, 16), the compressed-KV `cat`, and the indexer's index-K read (the last two ∝ max_pos). The weight reads stay flat to M=6 (bench: +0–5 %).
- Everything else is latency-bound.
  - `sparse_attn` runs one 256-thread CTA per (row, batch) (`kernel.py:335`); at 1 row, 151 of 152 SMs idle.
  - That CTA does all 64 heads and every 64-index block serially: gather 359-362, QK 365, online softmax 368-376, P→bf16 377, PV 380.
  - At 6 rows it launches 6 independent CTAs of identical code (m symbolic, 316), so the call is est. ~flat.

The 6-row block ≈ **1.0–1.2× the 1-row block (est.)**.

### 1.2 The indexer

- **Index sources:** 2, 8, 14, 20, 24, 28, 32, 36 (`config.json`).
- **k owners** (write index-K from the compressor latent, `model.py:500, 537-547`): 2, 8, 14 at ratio 2 (pool on odd positions only, 477), and 20 at ratio 1.
- **The 4 consumers** (24, 28, 32, 36) read L20's `k_cache` (549, 555) and mask to L20's candidate blocks (503, 574-576).
- **Every other layer** reuses its source's `topk_idxs` (726-727): 3–7 ← 2, …, 37–39 ← 36.

**Launches** (`rt/static_step.py:414-440`):
- per index source ~21: `wq_b` act_quant + GEMV (6.3 µs measured), RoPE 3, FP4 QDQ 2, `weights_proj` mm + scale 2, einsum 1, relu/×w/head-sum 3, mask 1, `topk` + `sort` 2+, `where`/`+offset`/`.int` ~4;
- 8 × 21 ≈ 170, matching the audit's ~180 (`GLUE-AUDIT.md:127`);
- owner k path: 15;
- L20 candidates: ~12.

**Data dependence.**
- Which 512 positions win depends on `relu(q·k)·w` summed over heads (557-558); how many, `min(512, T/r)`, on position (579).
- L20's 2048-block pruning (597-611) bites only past 16,384 compressed positions.
- Scores are **bf16**, so exact ties at the 512th place are plausible at long context, and `torch.topk`'s tie order is implementation-defined (T8 risk 6).

**Static padding** (docstring #9–#11): width W_r = (max_pos+1)/r (4096 / 2048), k = 512, unreachable = −inf via a device `compress_lens`. Two consequences:
- top-k cost scales with max_pos, not T (est. 10–20 µs/call, against the anatomy's 6.1 µs `gatherTopK` at short widths);
- **at T/r ≤ 512 the whole score/top-k path is dead.** The result is provably `[0…c−1, −1…]`. vLLM skips it there (`v/deepseek_v41/attention.py:1426-1448`), but not under graph capture (1428).

## 2. Borrow options for `sparse_attn`

**What the reference attends** (D1):
- window bf16 values = exactly e4m3 × 2^k (`kernel.py:82-86`);
- compressed bf16 values = exactly e2m1 × e4m3-scale (160-171);
- q bf16.

**What it computes:** fp32 scores from a bf16 MMA; online softmax per 64-block in index order, max initialised at −1e30 (352-355); P rounded to bf16 before PV (377); sink added to the denominator (382-383).

| option | KV it takes | holds the reference's values? | arithmetic vs reference | class |
|---|---|---|---|---|
| **A. reference TileLang, tuned** | bf16 caches as today | yes | identical if only the loop bound and data movement change | **bit-identical** |
| **B. FlashMLA sparse decode** `sparse_decode_fwd` (`v/flashmla/flash_mla_interface.py:57-194`) | window **528 B** = 512 e4m3 + 16 ue8m0/32; compressed as `extra_k_cache` **288 B** = 512 e2m1 (even index in the low nibble) + 32 e4m3/16 (113-117) | **yes, losslessly.** 528 B is exactly the window QDQ's code + exponent; 288 B exactly the latent QDQ's codes + E4M3 scales, the record D1 says vLLM calls "the format the reference implementation itself stores" | different: split-KV over SMs + merge; exp2 softmax; sink as `exp(lse)/(exp(lse)+exp(sink))` (93) | **A3-class drift if q stays bf16. Unknown whether the SM100 V4.1 kernel quantizes q to fp8 for a block-scaled MMA; if so, A1-class.** |
| C. FlashMLA sparse prefill `sparse_prefill_fwd` (197-234) | **bf16** [s_kv,1,512], our caches as-is | yes, no new writer | different; not tuned for s_q ≤ 6 | A3-class; speed unknown |
| D. FlashMLA mega (`v/…/flash_mla_mega_attn.py:361-381`) | 528 + 288 | yes | also absorbs q RoPE, inverse RoPE and the o FP8 cast; needs `wq_b`/`wo_a` permutes (232-247) and an fp8 `wo_a` einsum | A3 at best; deepest change. vLLM claims 29–31 µs at s_q 1–32, TP1, all-in (67-77; **claimed, not ours**) |
| E. FlashInfer TRTLLM-gen | SM10x: plain bf16 or per-tensor fp8; rejects `fp8_ds_mla` (`v/…/flashinfer_sparse.py:153-161`) | bf16: yes. Per-tensor fp8 at scale 1.0 (258-278): **no** | different | A3 with bf16; call ABI hidden behind an unvendored vLLM wrapper (29) |
| F. Pg61's `fp8_ds_mla` compressed (528 B MXFP8 of the raw latent) | — | **no** (D1) | — | **A1 deviation by construction. Reject.** |

### Import graph

- **The Python surface needs vllm.** `flash_mla_interface.py:9-10` does `import vllm._flashmla_C`, which runs `vllm/__init__`. The layer files import vllm config, forward context and workspace (`nvidia/flashmla.py:8-32`, `flash_mla_mega_attn.py:23-47`). There is no top-level `flash_mla` (`ENVIRONMENT.md:14`), unlike `deep_gemm` from `vllm/third_party` (`T9A-DEEPGEMM-ABI-20261106.md:3`).
- **The ops probably do not (unverified).** `torch.ops.load_library(<site-packages>/vllm/_flashmla_C*.so)`, found by glob, should register `sparse_decode_fwd` and `sparse_prefill_fwd` with no vLLM Python, provided the `.so`'s NEEDED list is only libtorch/c10/cudart.
  - We would own a ~40-line wrapper: `FlashMLASchedMeta` bookkeeping plus the op call, 124-194.
  - That keeps decision 3 literally, like DeepGEMM, while calling a vLLM-built binary. **James should confirm that reading.**
- **FlashInfer** 0.7.0.post1 imports standalone (`ENVIRONMENT.md:10`). Its DSV4 entry point is unknown here, and TRTLLM-gen cubins may download on first use.

### What B needs from us

- **Record writers.** The reference QDQ kernels with in-place off (`kernel.py:98-124, 184-204`) already emit the codes and scales. Pack them as data rows then scale rows per page (`v/…/cache_utils.py:481-484, 543-561`).
- **h_q = 64.** It needs h_q ∈ {64,128}, so ours fits exactly (`flashmla.py:66-74`).
- **Device per-row lengths.**
- **Graph capture.** One `FlashMLASchedMeta` per layer type, created inside the captured body, as vLLM does per step (`flashmla.py:212-228`); the in-kernel planner is then captured with the graph.
- **Six rows** as six batch entries, each with its own index list (`flashmla.py:200-203`).

### Recommendation

1. **Keep A as the default and make it cheaper bit-identically.**
   - A fully −1 block is already an exact no-op: max unchanged, exp(0) = 1, +0.0 adds (`kernel.py:363-380`; static docstring #9).
   - A per-row valid-block count read from the device turns 10 blocks back into 3–7 at fixture lengths: est. ~2× on today's call, zero cost at long context.
2. **Settle B with one box experiment (§5).** Adopt it as `--attn-impl flashmla` (A3-class, `ref` stays default) only if all four hold:
   - it loads without vllm;
   - its fp64-oracle error is bf16-class, and it is closer to the bf16-q oracle than to the MXFP8-q oracle;
   - it is ≥ 2× faster than tuned A under a graph at 1 and 6 rows;
   - replay is deterministic.
3. If B quantizes q, it is A1-class and goes to James.
4. C and E are fallbacks; D comes after B.

## 3. Fusions in this block

Ownership: #10, #13 and #16 are T9b-2 R1's (running now); T9b-3 inherits them. The orders below are the bit-identity contract either way.

| # | exact op order to reproduce (cites) | risk | launches saved/step |
|---|---|---|---|
| **3 kv path** | (1) `kv_norm`: `x.float()`, `x*x`, mean over 512, `+1e-20`, `rsqrt`, `x*r`, `w.float()*x`, `→bf16` RNE (`model.py:288-293`). (2) RoPE: `float()` of the bf16 values, pairs (2k, 2k+1) as (re, im) × (cos, sin) complex64, back to bf16 (392-406). (3) QDQ per 32: amax (exact), `max(amax, 1e-4)`, `s = 2^ceil(log2(amax·RN(1/448)))` via the bit trick (`kernel.py:22-37, 76-78`), `bf16(fp32(e4m3_rn(clamp(x/s, ±448)))·s)` (85); x/s exact since s = 2^k. (4) Store bf16 to the ring slot; optionally also the 528 B record for B. | medium (norm); RoPE low (FMA pattern, test); QDQ exact | 13 × 40 = **520** |
| **6 q path** | `q_norm` over 1280 (same 8 ops) → write qr bf16 (the indexer reads it, 527) **and** `act_quant(qr, 32, ue8m0)` for the `wq_b` GEMV. The 8 indexer `wq_b` calls reuse those bits (same input, same kernel). | medium (norm) | 8 × 40 + 8 = **328** |
| 10 o path (R1) | `freqs.conj()`; rotate `x·conj(f)` as torch's complex multiply (399, 404); `→bf16`; view [M,8,4096]; QDQ as #3 step (3). | low | 200 |
| 13 q RoPE (R1) | In the `wq_b` GEMV epilogue: round the fp32 accumulator to **bf16 first** (the reference's GEMM output is bf16, 772), then rotate as #3 step (2); the tile must own whole (2k, 2k+1) pairs. | low | 120 |
| 16 views (R1) | No `cat`s: either a two-pointer `sparse_attn` (`idx < 128` → window, else compressed[idx − 128]; addressing only), or one buffer per kv-source group holding each layer's ring plus the shared compressed rows, with a per-layer index offset. | none | 76 |
| **14a indexer q/k** | q: `wq_b` → bf16 → RoPE as #13 → FP4 QDQ/32: `max(amax, 6·2⁻¹²⁶)`, `s = 2^ceil(log2(amax·RN(1/6)))`, `bf16(fp32(e2m1(clamp(x/s, ±6)))·s)` (`kernel.py:165-171`). k (owners): `wk` bf16 mm → `k_norm` over 128 → RoPE → same QDQ → `index_copy_` at `pos//r`. | low (q); medium (k_norm) | ≈ **80** |
| **14b indexer score** | einsum q·k (bf16 out, fp32 accumulation over 128) → `relu` → `×w` (bf16 mm output × the scalar product, bf16) → sum over 32 heads (fp32, torch order) → bf16 → mask −inf. | **medium–high**: score bits change selection only at near-ties when T/r > 512 | ~48 |
| 14c topk/sort | keep `torch.topk` + `sort` | high | 0 |
| **14s short-context graphs** | When every row's `c = (pos+1)//r ≤ 512` (pos < 512), the indexer output is the integer constant `[0…c−1, −1…]` (vLLM's rule, `attention.py:101-116`). Replace score/top-k/candidates with a fill, keeping the owners' k writes. A second graph family; the host knows pos. | **none** | ~150 launches, est. 0.3–0.4 ms |

**The norm-statistics question** (#3, #6, k_norm, and T9b-2's #1/#5).

What is known:
- The reference statistic is `x.square().mean(-1)` (291).
- Torch's mean is a CUDA sum reduction projected as `sum × RN(num_out/numel)`: a multiply by RN(1/5120), not a division.
- The sum's order comes from `Reduce.cuh`'s launch configuration: vectorized per-thread accumulation, shuffle-down warp trees, then a shared-memory tree. That source is not vendored, so treat these details as **recollection to verify**.
- That configuration is chosen from the shape, **including the row count**.

Consequences:
- Triton can reproduce any fixed order if the reduction is written as explicit reshape-and-add steps (not `tl.sum`, whose tree is unspecified).
- Matching torch at M=1 means emulating torch 2.13's config for that shape, which is brittle.
- Torch itself may not be M-invariant (row i at M=6 ≠ M=1). If so, "6-row verify == six 1-row steps" is unattainable with torch glue regardless of fusion.
- `rsqrt`: torch's fp32 `rsqrt` is likely the math library's `rsqrtf`, which is not correctly rounded. The audit's `rsqrt_rn` suggestion (`GLUE-AUDIT.md:157-159`) would then *not* match.

**Box tests (no model; 10⁵ real activation rows at widths 128 / 512 / 1280 / 5120 / 16384):**
- (a) torch M-invariance of `square().mean(-1)` and `rsqrt`, M=1 vs 6;
- (b) which explicit Triton order matches torch at M=1;
- (c) the `rsqrt` match among libdevice `rsqrt`, `rsqrt_rn` and PTX approx.

If (b) finds a 100 % match, the medium rows become low. Otherwise build one fixed-order, one-program-per-row helper. Accept GEMV-class gates, and take **M-invariance by construction** as the payoff: M1 needs it more than M=1 bit-matching.

## 4. Six-row implications

Assumptions: rows i = 0…5 sit at p+i, and each row sees its own causal prefix; after n ≥ 1 accepts, the next step starts at p+n. The reference has no multi-token decode (`_window_kv`'s `squeeze(1)` 719, the one-token compressor branch 476-482, the scalar `compress_lens` 568). So the definition is: **row i == the 1-row step at p+i.**

| cache | writes per step | "write position = accepted prefix end" holds? | handling |
|---|---|---|---|
| window ring, 128 slots/layer | 6 | **no.** Writing `(p+i) % 128` evicts p−128…p−123, which rows 0–4 still need | **ring + 6-row side buffer, commit after accept.** This agrees with `M1-DSPARK-DESIGN.md` §1.5 (lines 117-131), which landed in parallel. Row j lists ring slots for p+j−127…p−1, then side rows 0…j: the same values in the same column order as the 1-row list (`static_step.py:90-95`), hence the same bits. Alternative: a ≥ 134-slot ring with no commit, at the cost of our own prefill seeding (709-717). |
| compressed KV / index-K at `pos // r` | 3 (r=2) or 6 (r=1) groups | **yes.** Group j is readable only from its completion position r·j+r−1 (564-568), and the completing row writes it before that layer's attention, so stale groups are rewritten before any read | per-row masks c_i = (p+i+1)//r |
| ratio-2 `kv_state`/`score_state` | slot `pos % 2` (479-480) | **no.** A rejected row can clobber the accepted partner | keep the 6 projections in a [6,512] staging buffer; pool row i from (row i−1 or the carried slot, row i) in slot order (even, odd) with the reference's `softmax(dim=1)`/`sum` (482); commit `staging[n−1]` as the carried slot when the next start is odd. The parity graphs remain: 2, or 4 with #14s |

**Indexer per row:** q [6,32,128] against one index-K; scores [6,W]; per-row c_i; L20 candidates per row (`select_candidate_blocks` broadcasts `compress_lens`, 605); top-k per row.
- Equality with the 1-row stream fails only at T/r > 512, via three things: shape-dependent bmm/head-sum reductions, `weights_proj` at M=6, and top-k ties.
- #14s makes fixture lengths exact.
- For long context, write #14b row-independent with a defined tie rule (equal score → lower position), and gate on "tie-explained differences only".

**`sparse_attn`:** M-invariant by construction (per-row CTA, symbolic m, no cross-row state; `kernel.py:316, 335`). FlashMLA with rows-as-batch is M-invariant within its own implementation.

**Rule for M1:** every new kernel here takes M rows, runs one fixed-order program per row, and is tested for **row i at M=6 bitwise equal to M=1**. That property makes the speculative runtime emit exactly its own greedy stream.

## 5. Plan

### T9b-3.0 — box measurement, no runtime change (~1 h, right after T9b-2 R1's window)

`bench/flashmla_probe.py` is the single FlashMLA experiment plus the timings the rest depends on.

1. **Load.**
   - Assert `'vllm' not in sys.modules`.
   - Glob `site-packages/vllm/_flashmla_C*.so` and record `readelf -d` NEEDED.
   - `torch.ops.load_library`, then assert vllm is still not loaded.
   - List `sparse_decode_fwd`, `sparse_prefill_fwd`, `fused_norm_rope_attn_rope_cast_decode`.
   - Read-only: the FlashInfer DSV4 entry point and whether its cubins are bundled.
2. **Values.**
   - Window from the reference `act_quant`, latent from `fp4_act_quant(…,16,e4m3)`, each run both in place and not.
   - Gate: dequant(codes, scales) == in-place, bitwise.
   - Pack 528 / 288 pages and gate: vLLM's dequant formula re-implemented in torch (`cache_utils.py:486-499, 543-566`) == the reference bf16 cache, bitwise. That proves the values are exact.
3. **Outputs.**
   - Indices: window oldest-first with leading −1s at p < 127; compressed sorted with a −1 tail.
   - Arms: A (reference), B (decode 528+288), C (prefill bf16), and E if callable.
   - Compare each to A bitwise (mismatch %, max ulp), and to fp64 oracles with bf16 q and with MXFP8-QDQ'd q. The nearer oracle settles the q question.
4. **Speed**, as 40 calls in one CUDA graph (2 ratio-0 + 38 compressed; b = 1 and 6; positions 300 and ≥ 512·r):
   - A at widths 128 / 278 / 428 / 640, which yields today's real cost (§0.2) and the per-block slope;
   - B and C;
   - replay equality ×2.
5. **Probes:** §3 (a)–(c), plus torch `softmax(dim=1)` over 2 slots and the indexer einsum, M=1 vs 6.

**Decision rule.**
- **Adopt B** if: it loads without vllm; its fp64 error is ≤ 2× A's and nearer the bf16-q oracle; it is ≥ 2× faster than A-with-skip at b = 1 and 6; replay is equal.
- **Park** if q is quantized (A1-class → James) or if it needs `import vllm`.

### Tasks

| task | content | parity gate | est. saving, 1 row |
|---|---|---|---|
| **T9b-3a** (bit-identical; `--attn-impl ref` bytes unchanged) | `sparse_attn` per-row valid-block count (TileLang loop guard from a device length; data-movement-only gather tweaks allowed); #14s short-context graph family; #16 if R1 did not land it | fp32-logit sha256 equal to the T9b-2 output at every position, 32 steps × 8 prompts; L1/L2 8/8, fills on/off | 2–3.5 ms (if §0.2 holds) + 0.3–0.4 |
| **T9b-3b** (medium; with T9b-2 R2's #1/#5, one shared RMS helper, one owner) | #3, #6 (+ dedupe), #14a | if §3(b) bit-matches: as 3a. Else GEMV class: ≤ 0.5 % output mismatch, L1/L2 8/8, A3 vs Pg61 (`PROJECT.md:95`) | ~0.9 (≈ 930 launches) |
| **T9b-3c** (A3-class `--attn-impl flashmla`; only if T9b-3.0 passes) | own wrapper over `torch.ops._flashmla_C`; 528/288 writes folded into #3 and the latent writer; per-type sched meta inside the graph | GEMV class + ≤ 35 µs/call at b = 1 and 6 | 1–4 vs tuned A |
| T9b-3d (optional, A3-class) | `wo_a` as an MXFP8 GEMV over the fp8 bytes the loader dequantizes today (`rt/loader.py:86`): same products, half the bytes | GEMV class | ~0.3 |
| T9b-3e (inside M1) | ring + side, per-row index lists and masks, compressor staging + commit, row-generic indexer | row i at 6 rows == 1-row step at p+i, bitwise, same impl, short and long context | — |

**Order:** T9b-2 R1 (running) → T9b-3.0 → T9b-3a → T9b-2 R2 + T9b-3b together → T9b-3c (decision) → M1 builds on row-generic kernels (T9b-3e). Nothing here blocks M1's design; 3a and #14s make its fixture-length verify exact.

**Saving:** ~3–7 ms off the 32.2 ms 1-row step (est.), mostly `sparse_attn`. Because the block runs at ~1.0–1.2× its 1-row cost at 6 rows, each ms saved here carries to the headline step.

T9B3: keep the reference TileLang `sparse_attn` as the bit-identical default and first make it skip its padded −1 blocks and the dead short-context indexer (exact). Run the one FlashMLA probe (load without vllm, 528+288 records hold the reference values exactly, fp64-oracle check for q quantization, graph timing at 1/6 rows). Adopt FlashMLA only as an A3-class `--attn-impl flashmla` if it is ≥ 2× faster and keeps q in bf16.
