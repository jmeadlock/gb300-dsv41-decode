# M2-L3 — DeepGEMM MegaMoE at world size 1: ABI findings (2026-11-07)

Desk findings from the vendored library (`reference/vllm-0.31.0/deep_gemm/`: `mega/__init__.py`,
`layout/mega_moe.cuh`, `impls/sm100_fp8_fp4_mega_moe.cuh`) and our runtime. **Nothing here ran on the box.** Every
number is either from a receipt (cited) or marked [E] (estimate, method given). The box measurements come from
`bench/dg_mega_spike.py` (real layer-0 bytes) and `tests/gpu/test_dg_mega_gpu.py` (mini gates).

## Verdict

The ABI fits our routing, our numerics contract and our pool's **w2** in place. It does **not** fit our pool's
**w1/w3**. MegaMoE wants one L1 tensor `[G, 2I, D/2]` with gate and up rows interleaved every 8 rows. Our pool keeps
w1 and w3 as two separate row-contiguous tensors. A whole-model copy of L1 is 136.7 GiB, which cannot fit. Landing
`--moe-impl dg-mega` on the real model therefore needs the pool itself re-laid out as interleaved L1 (option A below).
That change touches files outside M2-L3's ownership.

What M2-L3 delivers:
- **The spike.** It answers every ABI question on real bytes.
- **`runtime/dg_mega.py`.** A working dg-mega that uses an L1 **copy**. It is affordable on the mini checkpoints, so
  the row-invariance, replay, graph and drift gates can run on the box today. On the real model the HBM check refuses
  it, with a pointer to this file.
- **The `_build_moe` wiring.**

The decision to re-lay out the pool should wait for the spike's µs and drift numbers.

## What fits (read from source)

| item | what the library needs | where | ours |
|---|---|---|---|
| process group | `group.size()`, `.barrier()`, `.rank()` only at size 1; the buffer comes from `torch.empty`, the handle is a SimpleNamespace | `mega/__init__.py:51-67, 196` | `dg_mega.StubGroup`. No `dist.init`, no network |
| buffer | `SymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts)`. `num_experts` and the token count are **template constants** | `mega/__init__.py:26-84`; cuh template list | one buffer per distinct G = H+S. T = 6 token rows for both the 1-row and the 6-row step (see "rows") |
| experts per rank | ≤ 512 | `layout/mega_moe.cuh:56` | G = 295 + 16 = 311 |
| routing input | `topk_idx` int64 [T, k] (−1 = skip), `topk_weights` fp32 | cuh dispatch (`input_topk_idx_buffer`, `int64_t`), `layout/mega_moe.cuh:395-396` | our fp32 gate, top-6, ids ascending per row. The column holds the **pool group**: hot slot, H+c for cold, or −1 on overflow (counted) |
| combine order | fp32 accumulation over the top-k columns in column order, shared expert last, one bf16 rounding | cuh 1393-1490 (`__ffs` over the slot mask, `ptx::accumulate`) | ascending-id columns, which is the reference's `y += expert(x)` order (`model.py:897-906`) |
| SwiGLU | gate/up rounded to bf16, clamp in bf16 (gate max, up ±), `expf` with an IEEE divide when `fast_math=False`, `* weight` | cuh ~1058-1090 | `fast_math=False`, `activation_clamp=swiglu_limit`. Same order as `model.py:845-852` |
| L2 weights | one [G, D, I/2] int8 tensor. Expert e is rows `e·shape_n` of one 2-D TMA map | cuh:775 (`n_idx = local_expert_idx * shape_n + ...`) | **the pool's own `hbm_full[l]["w2"]`, in place** (the spike checks data_ptr identity) |
| SF | 4 UE8M0 codes per int32 (= our e8m0 bytes read as little-endian int32), per-row × 32-K, L1 rows interleaved, UTCCP 4×32 row transpose per 128 rows. The SF TMA's inner coordinate is n and its outer is `expert·K/128 + k`, so storage is **MN-major** `[G][K/128][N]` | `mega/__init__.py:137-172`; cuh:771-780 | `mega_l1_sf` / `mega_l2_sf`. Checked against the vendored functions on the Mac; checked against `transform_weights_for_mega_moe` by the spike (real bytes) and the GPU test. "k" storage is the spike's fallback |
| shared expert | fp8 weights + per-row SF. The 32×32 block scales repeat per row, which is exact | cuh `is_shared` path | `shared_weights()`, used only with `shared="fused"` |

## What blocks: the L1 gate/up interleave

- **What the kernel needs.** The weight TMA reads L1 as **one** 2-D tensor whose expert e starts at row `e·2I`
  (cuh:755-780). Within an expert, rows must be `[g0..7, u0..7, g8..15, u8..15, …]` (`_interleave_weights`,
  `mega/__init__.py:120-134`), because the epilogue pairs adjacent gate/up accumulator columns (cuh ~1058).
- **What our pool has.** w1 and w3 are two `[G, I, D/2]` tensors (`expert_pool.py`, grown in place by
  `add_spare_slots`, 107-144). Their rows are aliased row-contiguously by every other reader:
  - the reference `Expert` modules: eager prefill and `ref`. Their `w1.weight` *is* `pool.hbm[l]["w1"][0][slot]`, and
    `_build_moe` asserts it;
  - dg-inplace's contiguous GEMM, which needs each projection as one `[G, N, K/2]`;
  - the stage and gather kernels (whole-row copies);
  - the residency manager (rebinds expert params);
  - the host cold tier (checkpoint layout).
- **Why no view helps.** A view cannot turn two tensors into one uniformly strided tensor. A view of an interleaved
  tensor cannot give the Experts a contiguous `w1` back either: it would be 8-row pieces with a 16-row stride.
- **What a copy costs.** `dg_mega.hbm_bytes(40, [311]*40, 5120, 2304, "copy")` gives an L1 copy of 136.7 GiB (+ 12.8 GiB SF).
  HBM is 250 GiB, with ~5–17 GiB free after the 6-row builds (`LEDGER.md` 12:47 and 13:36; `dg_inplace.py:74-75`). One layer is 3.4 GiB, which the spike
  can afford.

### Options

**A. Re-lay out the pool in place (recommended if the spike pays).**
- The pool owns `L1[G, 2I, D/2]` interleaved, plus w2 as today. Converting is one layer at a time with a transient of
  one layer's w1+w3 (3.4 GiB [E, shapes]), the way `add_spare_slots` works.
- Every w1/w3 reader changes. Files: `expert_pool.py`, `patches.py` / the prefill MoE, `static_kernels.py`,
  `residency.py`, `host_tier.py` / `loader.py`, `dg_inplace.py`.
  1. Prefill and `ref` Experts. One fp4 GEMM over `L1[e]` with N = 2I, de-interleaved by a view of the output. It is
     bit-equal to the separate w1/w3 GEMMs only if the TileLang `fp4_gemm`'s per-element order does not depend on the
     row's position: a one-hour box probe. Otherwise, gather-de-interleave into the staging.
  2. dg-inplace. One contiguous GEMM over L1 instead of two. That is one launch fewer per layer, but its bit-equality
     needs the same probe.
  3. `stage_inplace` / `stage_rows` / the gather kernels. Interleaved destination copies (`l3_stage_mega` already does
     this).
  4. Residency swaps. The same interleaved copy.
  5. The cold tier. Interleave on the fly in staging (done), or convert the pinned host copy once.
- Estimate: 2–3 engineering days [E]. dg-mega's SF copy (12.8 GiB) replaces dg-inplace's "pool" SF store, which is
  the same bytes in another order; the two cannot coexist on the real model.

**B. CUDA VMM stitching.** Not possible: the interleave unit is 8 rows × 2,560 B = 20 KiB, far below the 2 MiB
mapping granularity.

**C. Patch DeepGEMM to take separate gate/up maps.** The tensor maps are built in host C++ (`csrc`, compiled into
`_C`, not vendored). This means rebuilding the library from source. That is outside the "borrowed library" decision.

**D. Gather the used experts into an interleaved staging every step** (dg-masked style). At 6 rows that copies
22.45 × 18.8 MB = 422 MB per layer, ≈ 80 µs at ~5 TB/s [E]. That cancels the gain.

## Shared expert

- **"gemv" (default; the handoff's rule).** The routed experts run in MegaMoE, whose output is bf16. The shared expert
  stays on our path, and the result is `bf16(fp32(y_routed_bf16) + fp32(shared))`. That is **one extra bf16
  rounding** compared with the reference's single fp32 sum. It is A3, and the drift numbers include it.
- **"fused".** The library's shared expert, combined last in fp32 by the kernel: a single rounding in the reference's
  order. It is computed by the block-scaled UMMA, so it cannot be bit-equal to our fixed-order GEMV. Under the
  handoff's rule it stays an option, measured (`fused_vs_gemv_mode`, oracle distance) and not adopted.

## Rows

- Every call is made with T = 6 token rows (`$DSV41_MEGA_MAX_TOKENS`). Rows ≥ R carry `topk_idx` −1 and zero x.
- So the 1-row and the 6-row steps launch **the same kernel instance with the same `num_tokens`**. Row invariance then
  rests on the kernel computing each token independently:
  - with swap-AB, tokens are UMMA columns;
  - amax is per token per 32;
  - combine is per token.
- The atomic dispatch order changes a token's column position, not its arithmetic. The GPU test and the spike check
  this bitwise.

## Expected cost per layer at 6 rows [E]

- **Basis.** Pg61's MegaMoE measured 81 µs/layer at 6 rows on this box and model (`M2-FUSED-CHAIN-DESIGN.md` §2.2 L3,
  economics table), scaled to 22.45 experts → 81–87.
- **Around the launch:**
  - route kernel ~3 µs (1 launch);
  - act_quant + x / x_sf writes ~6 µs (3 launches);
  - staging, only when cold, ~9 µs (`LEDGER`/design §4);
  - gemv-mode shared ≈ 23 µs (S10, design §1) + 1 add.
- **Totals [E]:**
  - fused shared: ≈ 90–100 µs/layer;
  - gemv shared: ≈ 115–125 µs/layer;
  - today: 130–165 µs/layer (handoff).
- **Against the ≤ 90 target:** fused is borderline; gemv misses. Getting to ≤ 90 needs `act_quant` writing straight
  into the SymmBuffer's x / x_sf (as T9b-2's `dg_act` does for dg-inplace), and the router kernel writing `topk_idx`.

## Gates (who measures what)

| gate | instrument | status |
|---|---|---|
| ABI: which call works (SF mn / k; shared 0 / 1), signatures, SymmBuffer views | spike `api`, `symm_buffer`, `first_call` | box |
| our layouts == the library's transform | spike `layouts` (real bytes); GPU test (random); Mac test (vendored source) | Mac PASS; box pending |
| mega vs dg-inplace, routed: equal / mismatch / max ulp at R = 1 and 6 | spike `numerics` | box |
| row-invariant R = 1 vs R = 6, replay-identical, graph == eager | spike (real bytes); GPU test (mini, every layer) | box |
| fp64-oracle distance ≤ today's | spike `oracle` (real weights, synthetic activations); GPU test (mini, asserted on relative L2) | box |
| margin-weighted top-1 flips, mini | GPU test end to end (printed) | box |
| margin-weighted top-1 flips, real fixtures | first_light parity | **blocked**: L1 copy does not fit on the real model (option A first) |
| µs per layer at 6 rows (target ≤ 90 incl. shared) | spike `speed` (real shapes, graph replay); GPU test (mini, informative) | box |
| launches before → after | GPU test (`torch.profiler` kernels per MoE call, dg-inplace vs dg-mega, R = 1 and 6) | box |

## To enable `--moe-impl dg-mega` (files outside M2-L3)

- `runtime/static_step.py` `MOE_IMPLS` (line 119) and `MOE_IMPLS_ROWS` (line 130): add `"dg-mega"`. `_build_moe`
  already handles it.
- `runtime/generate.py` `MOE_IMPLS` (line 105).
- `bench/first_light.py` `--moe-impl` choices.
- `tests/test_dg_moe_cpu.py` `test_contract_constants` (it pins the tuple) and its three "dg-mega is rejected"
  assertions.
- Knobs: `$DSV41_MEGA_SHARED` (gemv|fused), `$DSV41_MEGA_SF_MAJOR` (mn|k: set from the spike's `first_call`),
  `$DSV41_MEGA_SPARE`, `$DSV41_MEGA_MAX_TOKENS`, `$DSV41_MEGA_L1_STORE` (copy; inplace raises until option A).
- The fused-glue clusters `dg_act` / `swiglu_routed` / `moe_combine` turn themselves off for dg-mega (`fused_glue.py`
  `_applicable`: not dg-masked/dg-inplace). The router cluster is used.
- The residency manager does not know the L1 copy or the mega SF. A swap would leave them stale, so do not combine
  dg-mega with residency until option A gives one layout.
