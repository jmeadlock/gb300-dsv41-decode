# M3-N1 — can MegaMoE's numerics be made ours? (2026-11-07)

Bounded probe approved by James 2026-11-07 (`docs/HANDOFF-M3-N1-MEGAMOE-NUMERICS.md`). No pool re-layout;
`runtime/expert_pool.py` untouched. Instrument: `bench/dg_mega_numerics.py` (+ `runtime/dg_mega.MegaSubstitution`).
Labels: **[M box]** measured on the GB300 with the real checkpoint · **[M CPU synth]** measured on the Mac on
synthetic MXFP4 experts and synthetic activations (a model of the question, not the checkpoint) · **[S]** read from
source (file:line) · **[E]** estimate, method given · **not measured** = the box run below has not happened yet.
This worker had no box access: every real-checkpoint cell is **not measured**; the box commands are in section 7.

## 1. Answer so far

- **Where the drift comes from [S + M CPU synth].** One rounding point. MegaMoE quantizes the SwiGLU output to fp8
  straight from fp32. The reference first rounds it to bf16 (`model.py:853` `.to(dtype)`, then `act_quant`).
  The bf16 step moves a value by at most half a bf16 ulp. That is enough to land on the other side of an e4m3
  rounding midpoint, so about 3.5 % of the L2 inputs move by one e4m3 ulp. Every L2 output is a 2,304-long dot over
  those inputs, so 80–90 % of bf16 outputs change.
  - The synthetic model reproduces the spike's signature: routed rel-L2 to the reference-rounding oracle is
    **0.0161 (quant from fp32) vs 0.0017 (ours)**, and 89 % of routed elements differ. The spike measured
    0.0225 vs 0.0017 and 82 % [M box, layer 0, synthetic activations].
  - Everything else in the kernel matches the reference's rounding points [S]: bf16 gate/up, the bf16 clamp, the
    SwiGLU formula, and the power-of-two scale with its 1e-4 floor.
  - The second rounding in "gemv" shared mode is ours, not the kernel's. Alone it costs little: 0.0017 → 0.0020
    [M CPU synth].
- **It is not a worse numerics class [M CPU synth].** Against the exact result (fp64, no bf16 or fp8 rounding
  after the weights), the two arms are equally far: **ours 0.0268, quant-from-fp32 0.0266**.
  - The spike's fp64 oracle uses the reference's rounding points, so it measures fidelity to our convention, not
    accuracy. Against an oracle with MegaMoE's rounding points the roles swap: 0.0017 for the torch reproduction of
    MegaMoE, 0.0161 for ours.
  - The disposition (row 5) already says the "8× farther" gate was an operationalization; this is why.
- **Production uses MegaMoE (`docs/DAY2-REPORT-20261107.md:35`).** Under A3 (parity vs production), quant-from-fp32
  is therefore production's rounding point, and ours is the one that differs.
  - Two caveats. vLLM's DSV4 MoE call is not vendored (`docs/T8-T9A-DESIGN.md:13`), so we cannot see whether Pg61
    passes `fast_math=False`; the API default is `True` [S `mega/__init__.py:186`]. Pg61's cold experts also run on
    a sidecar GPU.
  - Plausible is not measured: the A3 arm (section 5) decides.
- **Can a variant close the gap?** The API has no activation-quant hook [S, section 2]. The torch reproduction with
  bf16 before the quant is **bitwise equal to ours** at the L2 input and in the routed sum, given the same GEMMs
  [M CPU synth].
  - So a one-line kernel patch should land in ours' class: final rel-L2 **≈ 0.0017–0.0020 vs 0.0134 unpatched**
    [E: synth model + the spike's accumulator class; the real patched kernel is **not measured**].
  - The patch exists and is mechanical: `patch-dg` applies it to a copy of the installed package, and the JIT
    recompiles it in a fresh cache, with no `_C` rebuild.
  - Whether to fork a borrowed library's kernel source is James's call (decision 9).

## 2. What the kernel does, read from source [S]

| stage | reference (`model.py:845-906`, `kernel.py:76-78`) | MegaMoE (`impls/sm100_fp8_fp4_mega_moe.cuh`) | equal? |
|---|---|---|---|
| L1 input | `act_quant(x)`: e4m3, scale `2^ceil(log2(max(amax,1e-4)·fp32(1/448)))` | the same bytes, written by us into `sym.x` / `x_sf` | yes (ours) |
| gate / up | fp4 GEMM → bf16 | UMMA fp32 acc → `__float22bfloat162_rn` (1058-1059) | accumulator order only (A3, as dg-inplace) |
| clamp | fp32 clamp of the bf16 values (gate max, up ±) | `__hmin2/__hmax2` in bf16 (1062-1066) | yes (10.0 exact in bf16) |
| SwiGLU | `F.silu(g)·u`, `·w` (CUDA silu = `x/(1+exp(-x))`) | `g/(1+expf(-g))·u·w`, IEEE divide when `fast_math=False` (1074-1089) | yes, if the JIT does not build with fast-math (the box measures it: `swiglu_formula`) |
| **L2 input** | **`.to(bf16)`**, then amax / scale / e4m3 | **amax and e4m3 cast from the fp32 values** (1092-1136) | **no: the drift** |
| L2 scale | `fast_round_scale` on `amax·(1/448)` | `get_ue8m0_sf_exp`: integer-carry ceiling, exponent floor 105 (`common/math.cuh:115-135`) | yes except fp32-product boundary cases (< 0.1 %, CPU test) |
| L2 → combine | bf16 per expert; fp32 `+=` in id order; + shared; one bf16 | bf16 per expert (1278); fp32 accumulate in column order (1449-1466); bf16 out | yes with ascending-id columns (ours) |
| shared, "gemv" | inside the fp32 sum | ours, added after the kernel's bf16 output | no: one extra rounding (our design) |

- **No quant knob.** The kernel's template constants are only clamp, alpha, beta and `fast_math`, and the `situ`
  variant does not change the requant.
  - `transform_weights_for_mega_moe` transforms weights only (`mega/__init__.py:154-172`).
  - `get_block_m_for_mega_moe` is a tile heuristic over six ints and a string (spike `api`).
  - No call argument reaches the L2-input quant. The L1 epilogue writes the fp8 L2 input into the SymmBuffer inside
    the same persistent launch, so GEMM2 cannot be fed a bf16-rounded input through the API.
- **What we can do, both in `layers`:**
  - Read the kernel's actual L2 input after the call. We zero `sym.l2_acts` first, then match each written row to
    our per-pair candidates. This needs no knowledge of the dispatch layout.
  - Patch the kernel (`patch-dg`).

## 3. Per-stage distances

Rel-L2 is pooled over all windows: √(Σ|d|²/Σ|ref|²). Mismatch is the fraction of bf16 (or fp8-byte) elements that
differ from the reference rounded to the same type. The reference is `oracle_ref` unless the row is marked
otherwise.

### 3a. Synthetic model [M CPU synth]

`bench/dg_mega_numerics.py model --windows 4`. Receipt: `scratch/m3n1-model-20261107.json`.

- Real shapes: 5120 × 2304, top-6, 6 rows.
- Experts: 17–21 unique per window, Gaussian weights quantized to MXFP4.
- Activations: N(0,1) times per-32 magnitudes 2^-6…2^6.
- GEMMs: fp32-emulated. There is no `mega` arm in this run.

| stage | ours | tm (quant from fp32) | tm_fix (bf16 before quant) |
|---|---|---|---|
| L1 gate/up (bf16) | 1.9e-5 (0.007 % differ) | = ours | = ours |
| L2 input, dequantized (bytes differing) | 1.1e-6 (0.0006 %) | **0.0159 (3.53 %; scales 0.41 %)** | 1.1e-6 (0.0006 %) |
| L2 output per expert (bf16) | 9.0e-6 | 0.0160 | 9.0e-6 |
| routed sum (bf16) | 0.00166 | **0.0161** (89 % of elements ≠ ours) | 0.00166 (bitwise = ours) |
| final, gemv-mode combine (2 roundings) | 0.00203 (`ours_2r`) | **0.0094** | 0.00203 |
| final, one rounding | **0.00167** | 0.0093 (`tm_1r`) | 0.00167 (`tm_fix_1r`, bitwise = ours) |
| routed vs `oracle_mega` (MegaMoE's rounding points) | 0.0161 | **0.00166** | 0.0161 |
| routed vs **exact** (no rounding after the weights) | **0.0268** | **0.0266** | 0.0268 |

The SwiGLU formulas were bitwise equal on all 331,776 CPU values. `oracle_mega` vs `oracle_ref` is 0.0160, the
same distance as tm vs ours.

### 3b. Real checkpoint, layers 0 / 10 / 20 / 39, fixture activations — **not measured**

Run with `layers`; the 1-row cells are each row alone through the kernel. The table below is `report`'s layout.

| layer | R | ours final | mega final (gemv) | tm | tm_fix | tm_fix_1r | mega bf16q | L2-input bytes ≠ oracle: ours / tm / mega-vs-tm (l2_acts) | mega vs tm routed mismatch |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| 0 | 1 | (row-independent: = R 6) | not measured | | | | not measured | row-invariant: not measured | not measured |
| 10 | 6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| 10 | 1 | | not measured | | | | not measured | not measured | not measured |
| 20 | 6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| 20 | 1 | | not measured | | | | not measured | not measured | not measured |
| 39 | 6 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| 39 | 1 | | not measured | | | | not measured | not measured | not measured |

The only real-checkpoint numbers so far are the spike's, on layer 0 with **synthetic** activations
(`docs/dg-mega-spike-20261107.json`) [M box]:

- final, gemv mode: mega 0.0134 vs ours 0.0017;
- routed: 0.0225 vs 0.0017; 82 % of elements differ, max_abs 8.5;
- row-invariant 6/6; replay-equal.

### How to read the box run

- **mega_vs_tm small** (the dg-inplace accumulator class, a few % of elements) **and l2_acts bytes match tm ≈ 1**
  (ours lower): the drift source is confirmed on the real kernel.
- **mega_vs_tm large:** something else differs. Candidates are fast-math in the JIT build (see
  `swiglu_formula`) or the L1 accumulators.
- **The patched run (`mega bf16q`):** its l2_acts bytes should match ours, and its final rel-L2 should fall to
  ours' class. If the l2_acts match does not move, the patch did not take (stale JIT, or the copy was not imported).

## 4. The bf16-before-quant variant

- **Kernel patch [S].** After `activation_values[i][k] = __fmul2_rn(__fmul2_rn(gated, up), weights);`
  (cuh:1088-1089), insert:

  ```
  activation_values[i][k] = __bfloat1622float2(__float22bfloat162_rn(activation_values[i][k]));
  ```

  amax and the e4m3 cast then see the reference's bf16 values. The anchor must occur exactly once, which the
  vendored source satisfies (CPU test).
- **How `patch-dg` delivers it:**
  - It copies the installed `deep_gemm` package (`_C` included) to `scratch/dg-bf16q/` and patches only that copy.
  - `layers --dg-path scratch/dg-bf16q --jit-cache scratch/dg-bf16q-jit` imports the copy in a fresh process. The
    JIT then compiles from the copy's include root into a fresh cache, so no unpatched kernel is reused.
  - If the copied `_C` fails to load, the fallback is the installed package re-initialised on the copy's root
    (`_C.init`). The receipt records which method ran.
- **Torch reproduction [M CPU synth].** With bf16 before the quant, the L2-input bytes and the routed sum are
  bitwise equal to ours. What remains:
  - The gemv-mode second rounding: final 0.0020 vs 0.0017.
  - The kernel's own accumulator order, which a CPU model cannot see. The box's mega_vs_tm measures it.
- **What a patch would buy [E].** Final rel-L2 ≈ sqrt(0.0020² + d_acc²), where d_acc is the UMMA-vs-contiguous
  accumulator class (dg-inplace vs oracle: 0.0017 [M box spike]). That gives ≈ 0.002–0.003, against 0.0134 today.
- **Fused shared mode is broken, not merely different.** The spike measured rel-L2 0.78, with 99.8 % of elements
  differing from gemv mode. The shared-weight layout or the library's shared path is wrong. That is a separate bug,
  not investigated here, and gemv mode is the only usable mode today.

## 5. Downstream: the A3 metric with MegaMoE on the four layers

- **Instrument: `downstream`.**
  - It boots the production 1-row config, with first_light's flags (dg-inplace, gemv, fused, flashmla, fp32 fixed,
    rowmatch off). It runs in static mode, which the L1 gate shows is bit-identical to graph mode, and which the
    capture hooks need.
  - Then, per arm:
    1. Teacher-forcing along **Pg61's own tokens** (parity-A, token bytes → ids). A flip is a position where our
       argmax is not Pg61's token, weighted by Pg61's top1 − top2 margin, and separately vs our unsubstituted
       argmax with our margin. The receipt also has |Δ logprob| of the forced token.
    2. Free-running greedy, then `bench/parity.py` (via first_light's `run_rep` / `run_parity`).
- **Why teacher-forcing [S `bench/parity.py:33-41`].** For two greedy streams, the weighted flips that `parity.py`
  computes are **zero by construction**. Flips are only counted before the first token divergence, where both argmaxes
  equal the shared token (exact ties aside). The "0.000 on 8/8" ledger lines (T9b-8) carry no information beyond the
  first-divergence position. Teacher-forcing makes the whole stream a shared prefix, which is what the A3 wording
  ("flips over shared prefixes") needs. Owner of `docs/A3-DEFINITION.md` (reviewer 10D): please note.
- **Substitution: `MegaSubstitution`.**
  - Per layer: an interleaved L1 copy of the pool's own bytes (G = the 1-row pool's 295 + 6 = 301), the SF copies,
    and the pool's w2 in place. Cold experts are staged into the pool's spare slots through the step's own pointer
    tables.
  - **3.3 GiB + 0.31 GiB SF per layer** [S `hbm_bytes`], plus 0.3 GB SymmBuffer. That is half the handoff's
    fresh-load of all 384 experts (7.2 GB/layer), from the same bytes.
  - The HBM check runs before any copy. With `--groups auto`, an arm that does not fit is halved: [0,10,20,39] →
    [0,10] and [20,39] → singles. The receipt says what fit.
  - The GPU test (`test_m3n1_substitution_*`) gates the substitution: equal to a full dg-mega step bitwise, and
    restore() is bit-identical.

| arm | layers | fit (free HBM before) | TF positions | flips vs Pg61 (weighted) | flips vs ours (weighted) | max abs dlogprob vs ours | free-run parity | free-run == ours |
|---|---|---|---|---|---|---|---|---|
| base (production 1-row) | none | — (free after build: not measured) | not measured | not measured | — | — | not measured | — |
| mega | 0, 10, 20, 39 | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| mega_bf16q (`downstream --dg-path`, patched kernel) | 0, 10, 20, 39 | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

## 6. µs per layer, hot, graph-replayed (the launch alone)

| layer | R = 1 | R = 6 | bf16q R = 1 | bf16q R = 6 |
|---|---|---|---|---|
| 0 | not measured (spike: 52.77, synthetic routing) | not measured (spike: 92.17) | not measured | not measured |
| 10 | not measured | not measured | not measured | not measured |
| 20 | not measured | not measured | not measured | not measured |
| 39 | not measured | not measured | not measured | not measured |

## 7. Box run (serialized, `mk-` named containers, in the worktree `~/dsv41/wt-m2-l3`)

Use the `dk` wrapper of `bench/box/m2-round1.sh` (`/ref`, `/work/scratch`, the DG JIT cache mounted):

```
dk $W m3n1t -- tests/gpu/test_dg_mega_gpu.py                                  # gates incl. test_m3n1_substitution_*
# ORDER: downstream FIRST (it writes scratch/m3n1-acts.pt, which layers reads), then layers, patch-dg, the patched runs.
dk $W m3n1d -- bench/dg_mega_numerics.py downstream --out scratch/m3n1-ds.json   # writes scratch/m3n1-acts.pt
dk $W m3n1l -- bench/dg_mega_numerics.py layers --out scratch/m3n1-layers.json   # needs the acts from downstream
dk $W m3n1p -- bench/dg_mega_numerics.py patch-dg --out scratch/dg-bf16q
dk $W m3n1q -- bench/dg_mega_numerics.py layers --dg-path scratch/dg-bf16q --jit-cache scratch/dg-bf16q-jit \
    --out scratch/m3n1-layers-bf16q.json
dk $W m3n1s -- bench/dg_mega_numerics.py downstream --dg-path scratch/dg-bf16q --jit-cache scratch/dg-bf16q-jit \
    --out scratch/m3n1-ds-bf16q.json                  # mega_bf16q arms; needs patch-dg first; does not rewrite the acts
python3 bench/dg_mega_numerics.py report --layers scratch/m3n1-layers.json --patched scratch/m3n1-layers-bf16q.json \
    --downstream scratch/m3n1-ds.json --downstream-patched scratch/m3n1-ds-bf16q.json --model scratch/m3n1-model-20261107.json
```

Expected wall time [E]:

- `downstream`: boot ~5 min, then per arm ≈ 2 × 1,476 static-eager steps. At ~0.1 s/step that is ≈ 5 min per arm;
  `auto` runs 1 to 7 arms.
- `layers`: ~1 min per layer.

## 8. Files

- `bench/dg_mega_numerics.py` (new): the instrument.
- `runtime/dg_mega.py`: `DGMegaBuffers(layers=)` (HBM plan over the served layers only), `DGMegaMoE(bufs=)`,
  `MegaSubstitution`.
- `tests/gpu/test_dg_mega_gpu.py`: the GPU substitution gate, plus the CPU tests M3-N1 owns.
- `bench/dg_mega_spike.py` is unchanged.
