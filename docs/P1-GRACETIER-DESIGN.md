# P1 — `gracetier`: a Grace cold-expert tier for Garner's vLLM recipe (design desk)

Handoff: `docs/HANDOFF-P1-GRACETIER-DESIGN.md`. Port plan row: `docs/PORT-PLAN.md:39`. Desk-only: no box, no network,
nothing imported. Written 2026-11-08 (CST).

Tags: **[M]** measured (receipt named), **[D]** derived from [M] by the arithmetic shown, **[E]** estimate / model
(inputs named), **[A]** assumption. "Not in tree" = the fact needed is not in this repository.
Short names: `B/` = `reference/garner-8043a681/fastpath/fastpath_boundary.py`, `P3/` = `.../fastpath/fastpath_peer3.py`,
`HK/` = `.../fastpath/hook/mega_peer_hook.py`, `PR/` = `.../fastpath/peer3/csrc/peer3_proto.cuh`,
`PG/` = `.../fastpath/peer3/csrc/peer3_gb300.cuh`, `G/` = `reference/garner-8043a681/`.

## Bottom line

- **The seam.** For T ≤ 16 the w4 router publishes each token's cold routes into the peer-v3 segment. The record
  holds the MXFP8 row, the fp32 route weights, and the cold-local ids. The finish then waits for a `done` tag and
  adds bf16 cold rows into the MegaMoE output. That add runs inside the next layer's attention seam (FUSE_FINISH),
  or in the post-only seam before an Engram layer / after the last layer (POST_FUSE), or as peer3's own finish kernel.
  For 17 ≤ T ≤ 64, peer3's own send + fold-in-MegaMoE does this job. For T > 64 (prefill, and C32 verify widths),
  the hook's v2 path to the RTX does it. The drafter has no cold tier.
- **Recommended: (b) stage-into-slots, in a "fixed-slot" form.** Every MoE layer's MegaMoE is built with the stock
  G = 384 groups. Each cold expert gets a fixed group H + c, where c is its cold-local id. Those cold groups are
  backed by **one HBM scratch shared by all 40 layers**, mapped behind each layer's hot block with CUDA VMM. The
  router gets a row_map that names a group for every expert, so it publishes nothing. One stage kernel, placed
  between the router and MegaMoE, copies the touched cold experts' weights from Grace into their groups. MegaMoE
  then computes hot and cold experts together.
  - No dynamic slot index, no overflow, any T.
  - **Bit-identical numerics to stock MegaMoE on all 384 experts**, unlike peer3, which Garner tags "changes
    numerics".
  - Graph-capturable with our measured staging kernel class.
  - About +7.6 GiB HBM, roughly −2.9 M KV tokens out of 11.4 M [D].
- **Cost.** At Garner's static h255 map, the cold bytes cross C2C serially before MegaMoE. Modeled exposed cold time
  is **3.24 ms/step at 6 rows and 3.60 ms/step at 1 row** [E, §2.3]. Against Pg61's 7.39 ms C1 step
  [M `G/README.md:126`] that is about +44 % [D].
  - (a) compute-from-UVA can only beat this by running *concurrently* with MegaMoE. That needs C2C bandwidth from
    the ≤ 8 SMs MegaMoE leaves free (`B/:100,111`). That rate is not in tree, and it is the one number the mini-test
    must produce before (a) is built.
  - Residency is the lever on the cost itself: at our runtime's measured 1.19 / 0.36 cold experts per layer-call,
    (b) costs 2.44 / 0.82 ms/step [D].
- **C32 is not modelable from the tree.** The distinct-cold count at T ≈ 128–256 is not in any receipt. Each 1.0
  distinct cold expert per layer-call costs 1.96 ms/step serial [D], against a 38.6 ms C32 step [D].

## 0. Corrections and inconsistencies in the inputs (read before the numbers)

1. **Which rowmap.** `HK/:3-6` says 295 hot / 89 cold, and so does `docs/C2C-PROBE-20261106.md` §3 ("Pg61's static
   rowmap (295 hot / 89 cold)"). The vendored rowmap `rowmap-mix-v1-h255.json` has `meta.n_hot = 255` and splits
   every one of the 40 layers 255 / 129 [M, read]. `G/docs/ARCHITECTURE.md:37-39` and `fastpath_coldskip.py:9`
   (129 cold, 90.3 GiB) agree with it. This design uses **255 / 129**.
   - Which map was live during the `peer3_stats.json` window: not in tree. The JSON does not record it.
2. **peer3_stats labels.** `docs/C2C-PROBE-20261106.md` §3 says "2.06 cold experts per layer-call at T=6 (1.6 at T=1)".
   The receipt `docs/data/peer3_stats.json` says otherwise [M]:

   | T | calls | distinct cold experts / call |
   |---|---|---|
   | 6 | 14,760 | 1.60 |
   | 8 | 4,840 | 2.11 |
   | 1 | 120 | 1.78 |

   The doc's labels look shifted, so the JSON is used here.
3. **Cold share per token depends on T.** Cold routes per token are 0.35 at T=6 (2.12 pairs / 6) and 0.41 at T=8,
   but 1.78 at T=1 and about 1.9–2.2 at every T ≥ 12. Examples: T=16 gives 32.79/16; T=64 gives 123.97/64 [D from
   `by_T`].
   - The high-volume single-stream verify widths therefore see about 5× fewer cold picks per token than any other
     width.
   - The T=1 row rests on **120 calls (3 steps)**.
   - The cause is not in tree. It could be the calibration mix of the h255 map (`meta.calibration`), padding rows, or
     workload. The 1-row figures below inherit this caveat.
4. **MIX_OFFPATH.** `G/docs/WHAT-DID-NOT-WORK.md:63-68` says moving the mix off the main path lost. `B/:97-102`
   ships `MIX_OFFPATH = True` with a measured −2.4 µs and MegaMoE at `kNumSMs = 144` (`B/:111`, `B/:508-519`).
   Gracetier must coexist with the shipped code, so it assumes 144/152 SMs belong to MegaMoE during the MoE call.
5. **Stripped binaries.** `libw4boundary.so`, the peer3 `.so` files and the MegaMoE fork's JIT outputs are stripped
   from the vendored tree (`VENDORED.md`, Garner entry).
   - Router behaviour beyond its Python ABI (`B/:171-181`, `B/:1044-1088`) is not in tree.
   - `DeepseekV4MegaMoEExperts` and vLLM's model loader are not in `reference/vllm-0.31.0/` either (only `deep_gemm`,
     `deepseek_v41`, `flashmla`, two utils). The attributes named below (`num_experts`, `get_symm_buffer()`,
     `_transformed_l1_weights`, …) are as Garner's Python uses them, not read from the class.

## 1. The seam, exactly

### 1.1 Which path covers which T

| T (tokens in the MoE call) | entry | cold publish | MegaMoE | cold add ("finish") | graph context |
|---|---|---|---|---|---|
| 1–16, target layers, router eligible | `DeepseekV4MoE.forward` → `B/_make_moe_forward` (`B/:1300-1389`); eligibility `B/:1151-1170` (`min_t` = 1 under v3, `B/:1157`; T ≤ `ROUTER_MAX_T` = 16, `B/:137,1159`) | **w4 router**, one launch: gate GEMM + top-6 + hot/cold split + MegaMoE staging + **peer-v3 publish** (`B/:1044-1088`, `B/:1359-1365`; `p3 = (seg.dptr, state)`) | `_mega_moe_call` at `kNumSMs` 144 (`B/:1375-1379`, `B/:508-519`) | **FUSE_FINISH** (`B/:70-74`): the finish is deferred (`B/:1382-1386`) and runs inside the next layer's attention-side seam (`mhc_shifted_post_pre`, `B/:982-1018` → `seam(p3fin=…)`, `B/:594-624`). For the layer before an Engram layer and the last layer, **POST_FUSE** runs it in the post-only seam (`B/:116-118`, `B/:809-833`). Membership in `_FUSE_LAYERS` comes from `B/:1419-1426`. Otherwise `_finish_now` = peer3's finish kernel (`B/:458-471`, `B/:1387-1388`). **COMBINE** (`B/:103-110`): for T ≤ 8 MegaMoE skips its combine (fork v11 skip word) and the seam sums the 7 combine slots before the cold add (`B/:1352-1357`, `B/:540-591`). | FULL CUDA graph of decode steps, main (current) stream (`B/:3`). FFN mix on vLLM's mHC side stream, not joined before MegaMoE (`B/:97-102`). |
| 1–16, router not eligible / no peer3 state yet | `B/:1313-1324` → stock forward → peer3's `forward` | peer3 `send` (grid T) | stock MegaMoE | as 17–64 | same |
| 17–64, target layers | stock gate (DeepGEMM mega gate) → `DeepseekV4MegaMoEExperts.forward` wrapped by peer3 (`P3/:164-197`) | `p3.ext.send(...)`: per-token MXFP8 quantize + v3 record (`P3/:191-192`) | stock MegaMoE (`P3/:133-134`) | **fold** (fork v5, default `PEER3_FOLD=1`, `P3/:19-23,40`): MegaMoE's combine waits for `done` and adds the cold rows itself (`P3/:96-125`). Else peer3's finish kernel as a PDL dependent of MegaMoE (`P3/:135-136`, `P3/:38`). | FULL or piecewise graph (the boundary checks `in_piecewise_cudagraph()`, `B/:992-995`); current stream |
| > 64 (prefill, wide verify) | `P3/:181-183` → hook forward | **peer v2**: separate pack/quantize (`HK/:222-227`) | stock MegaMoE (`HK/:228-229`) | v2 `peer2.finish` scatter-add (`HK/:230`) through the b12x sidecar. (`PEER3_DETERMINISTIC_PREFILL` = v3 in 64-token chunks, `P3/:140-161`, test mode.) | eager / piecewise (`B/:31`) |
| drafter (DSpark) layers, 2 ≤ T ≤ 64 | `B/:1304-1307` → `_draft_moe` (`B/:1253-1285`) | none — 128 experts, top-3, **all on the GB300** (`_mp is None`, `B/:1235`; `HK/:14`) | drafter's own MegaMoE | none | as the target layers |

`_finish2` is the **peer-v2** fused finish (`hook/peer_fused2.py:122`). The boundary calls it only when peer3 is
absent (`B/:1380-1381`). In the shipped recipe (peer3 + `PEER3_ON`, `B/:66-69`) it does not run. COPY_AHEAD is
**off** in the shipped file (`B/:124`). When on, it forks a copy chain after the publish, graph-capturable as
memop + memcpy nodes (`B/:416-433`). The chain is joined at the forward's end (`B/:1432-1441`) and at every
breakable-capture segment end (`B/:1446-1464`).

### 1.2 What the router publishes for cold rows (peer v3)

One pinned, host-mapped segment of `kSegBytes` (`PR/:24-54`). The router holds `(seg.dptr, state)`, and the offsets
are checked against the router library at first use (`B/:1128-1148`). Per token t < T ≤ `kMaxTokens` = 64
(`PR/:21`):

| field | where | type / shape | written |
|---|---|---|---|
| route weights | `rec[t] + 8` (`kOffRec` = 4096, 128 B records, `PR/:41-42`) | fp32 [6], top-k order | always |
| global ids (stats only) | `rec[t] + 32`, tag copy at +44 | int16 [6], −1 masked | optional (`PR/:55-58`) |
| MXFP8 activation row | `kOffXq` = 12288 + t·5120 (`PR/:43`) | fp8 e4m3 [5120] | only if the token has a cold pick (`PR/:6`) |
| its scales | `kOffXs` + t·160 (`PR/:44`) | e8m0 u8 [160] = 1×32 blocks (`PR/:22`). The quantizer is bit-identical to `route_send2`'s (`peer3/README.md:44,95`); its rounding rule is in `PR/:123-128`. | same |
| `word0` | `rec[t] + 0`, **release.sys** | u64 = tag(16) ‖ 6 × 8-bit **cold-local** expert id per top-k slot, 0xFF = not cold. tag = counter(4) ‖ layer(6) ‖ T−1(6) (`PR/:7-9`, `PR/:69-78`) | always (every call is acknowledged, `PR/:10-12`) |

The consumer's reply is `out[t][5120]` bf16 at `kOffOut` (`PR/:45`): the weighted sum of the token's cold experts in
top-k order. After it comes `done = tag` (u64 at 128, release.sys, `PR/:27`). Device-side outputs of the router are
listed below; the cold-row fields are unused in v3 (`cold_ids = None`, `cold_q/cold_sf = None`, `quant_cold = 0`,
`B/:1069,1075`).

- `topk_ids` int64 [T,6], `topk_w` f32 [T,6], `hot` int64 [T,6] (`B/:1056-1058`)
- `has` int8 [T], `pos` int32 [T], `rows` int32 [1] (`B/:1059-1061`)
- MegaMoE's staged inputs, written in place: `symm.x` fp8 [T,5120], `symm.x_sf` int32 [T,40], `shared_l1_acts_sf`,
  `symm.topk_idx[:T]` = hot slot or −1, `symm.topk_weights[:T]` (`B/:1359-1365`, `B/:1173-1181`)

### 1.3 The add, numerically

The add is `y[t] = bf16(float(y[t]) + float(out[t]))`, per bf16 pair (`PG/:171-182`, `PG/:184-203`), after
`wait_done` spins on `done` up to `PEER3_SPINS` (`PG/:148-169`, `P3/:36`). The same math runs in the finish kernel,
in the seam's fused path, and in the MegaMoE fold (`fastpath_megamoe.py:18-21`). Upstream of it, the RTX's `out`
rows come from b12x w4a8_mx with **bf16 accumulation and unordered atomics, "0.9 % rms between replays at M ≥ 2"**
(`G/docs/ARCHITECTURE.md:119-122`). That is why `peer3` is tagged "changes numerics" (`ARCHITECTURE.md:82`).

### 1.4 What gracetier replaces, and what it must not touch

- **Replaces:**
  - `coldskip`: the read filter, `fastpath_coldskip.py:59-75`
  - `peer3`: the expert-forward wrap and the segment, `P3/:164-197`
  - the RTX sidecar
  - the hook's v2 path for T > 64 (`HK/:222-231`)
- **Keeps:**
  - the w4 router and its MegaMoE staging
  - MegaMoE and the fork
  - the seams
  - the drafter path, untouched
  - `a12_retire` / `a12_memory_guard`, kept, with the geometry note in §6
- **Stays inside the hook mechanics:** a `@register("gracetier", …)` component in `components.py`'s list, in place
  of `fastpath_coldskip` + `fastpath_peer3` (`components.py:28-29`). It wraps `HK`'s `_init`, `_weight_loader`,
  `_finalize_weights` and `forward` the way peer3 wraps `forward` through the hook's globals (`P3/:167-172`). It
  needs the small boundary change listed in §2.2.

## 2. Two candidate consumers (both GB300-side)

### 2.0 Model inputs (shared)

| input | value | tag / receipt |
|---|---|---|
| MoE layers per step | 40 | [M] `fastpath_a12_retire.py:28`; rowmap 40 layers |
| bytes per cold expert | weights 17,694,720 B + e8m0 scales 1,105,920 B = **18,800,640 B** | [M] `docs/M3-S2-STAGING-BYTES.md:26-27`. MegaMoE layout per expert: L1 4608×2560 B + L1 SF 4608×40 int32 + L2 5120×1152 B + L2 SF 5120×18 int32 = the same total (`a12_retire.py:31`) [D] |
| distinct cold experts / layer-call | 6-row **1.598** (hist n = 14,760); 1-row **1.783** (n = 120) | [M] `docs/data/peer3_stats.json` `by_T['6']`, `by_T['1']` (§0.3 caveat) |
| Grace → HBM copy kernel | 2.82 µs at c = 0; 52.42 / 100.28 µs at c = 1 / 2 → **48.73 µs per expert = 385.8 GB/s** | [M] `docs/staging-bytes-probe-20261108.json` `serial[*]["m1_stage_rows/pool"]` (= `M3-S2:55` "48.9 µs, 362 GB/s" from the M3-S1 probe) |
| weights-only copy | 17,694,720 B / 385.8 GB/s = **45.86 µs** per expert | [D] |
| copy engine | 390 GB/s bulk; 52.5 µs/expert at expert granularity; concurrent with compute at full rate | [M] `docs/C2C-PROBE-20261106.md` table + overlap probe |
| naive kernel over UVA | 230 GB/s (the kernel is the limit: 361 GB/s on HBM) | [M] same |
| reference `fp4_gemm` over UVA | 102.8 / 102.9 µs per projection at M = 1 / 6 (91.3 / 91.1 from HBM), bit-identical | [M] same |
| MegaMoE (Garner fork), T = 6 | 64.6 / 68.7 / 83.0 µs at D = 15 / 16 / 21 experts → **+3.07 µs per extra expert**, intercept 18.6 µs at D = 0 | [M] `fastpath_megamoe.py:29-30`; slope / intercept [D] (2-point fit 15 → 21) |
| MegaMoE, T = 1 (our build, stock DeepGEMM, G = 311) | 50.98 µs | [M] `docs/m3n1-layers-20261107.json` `layers['10'].speed.mega_R1_hot_us` (proxy: Garner's fork at T = 1 not in tree) |
| side-stream fork/join + empty launch | 3.34 µs/layer | [M] probe `overlap[0]["kernel-overlap/pool"].exposed_us` |
| copy overlapped with a 137 µs DeepGEMM chain | exposed 3.34 / 26.79 / 47.47 µs at c = 0 / 1 / 2 | [M] probe `overlap[*]` (our dg-inplace chain, not MegaMoE) |
| Pg61 step, C1 | 7.39 ms (135.4 steps/s, fixed coding prompt); ~8.1 ms at 6 tokens | [M] `G/README.md:126`; `G/docs/ARCHITECTURE.md:187` |
| Pg61 step, C32 | 38.6 ms (79 tok/s per stream / 3.05 tokens per step) | [D] from `G/README.md:138` |

### 2.1 (a) compute-from-UVA

**Kernel choice.** The reference `fp4_gemm` is rejected: it is latency-bound at 102.9 µs per projection over UVA
[M], about 309 µs per cold expert [D]. A purpose-written kernel is avoidable. **The candidate is a second
`fp8_fp4_mega_moe` call** on a cold-only module: G_c = 129 groups, no shared expert. Its L1/L2 weight tensors are
**UVA views of the Grace cold pool**, stored in MegaMoE's per-expert layout:

- L1: gate/up rows interleaved at granularity 8
- L2: as is
- SF: packed int32, UTCCP-permuted, MN-major (`reference/vllm-0.31.0/deep_gemm/mega/__init__.py:120-172`;
  our pure equivalents `runtime/dg_mega.py:103-246`)

Its SF banks stay in HBM, 5.31 GiB for 40 × 129 experts [D]. The `topk_idx` is the cold-local id for cold picks and
−1 otherwise. This is the handoff's "DeepGEMM grouped fp8×fp4 kernel whose weight operand is the UVA alias". It
reads only the touched experts' tiles. **Whether MegaMoE's weight TMA accepts a host-registered UVA address is not in
tree.** Our only UVA kernel receipt is the reference `fp4_gemm` (C2C probe) plus our Triton staging kernels; DeepGEMM
with B on UVA was never run.

**Where it hooks.** It lands at the peer3 seam unchanged. gracetier owns a peer3-layout segment
(`peer3_seg.Segment`, `P3/:63`) and stamps `PROTO_VERSION` itself, since no sidecar exists (`P3/:76-86`). The router
publishes v3 records exactly as today. After the router, on a forked capture stream (the `_copy_ahead` pattern,
`B/:416-433`, joined like `join_copy`, `B/:435-455`), gracetier runs the cold MegaMoE. Its input is `symm.x/x_sf`
with the router's cold-local `topk_idx`, built by a 1-launch remap kernel from `topk_ids` + row_map. It then writes
the `out` rows and `done = tag` (stream-ordered). The seam's fused finish then waits, which returns immediately, and
adds. **No boundary edit, no seam edit.** The whole v3 consumer side moves from the RTX onto the GB300.

- Serial variant: the same work on the main stream after MegaMoE.
- Concurrent variant: hot MegaMoE at `set_num_sms(144)` (already the shipped setting) and the cold MegaMoE at
  `set_num_sms(8)`.
- The fork/join and the 8-SM split are exactly what `G/docs/WHAT-DID-NOT-WORK.md` §1 and §5 warn about (§6).

**What MegaMoE needs:** nothing new in the kernel. A second module instance with G_c = 129 is a new JIT
specialisation, since G is a template constant (`runtime/dg_mega.py:17`). It also needs TMA over UVA, which is
unverified.

**Graph capture.** Kernel reads of registered Grace memory are capturable; our staging runs inside captured graphs
today (`runtime/staging_overlap.py:9-18`). The fork must be joined inside every capture segment (`B/:1446-1464`
precedent). The segment's `done` word is written by a GB300 kernel, so there is no cross-device spin and no
`PEER3_SPINS` timeout path.

**Numerics.** Per-expert tile math is MegaMoE's: block-scaled UMMA, bf16 round + clamp, SwiGLU, fp8 requant from
fp32, combine in fp32 over top-k columns in column order (`runtime/dg_mega.py:10-14`). The cold sum is rounded to
bf16 on its own and then added in the seam: `bf16(float(y_hot) + float(y_cold))` (`PG/:171-182`). The result is
**one extra bf16 rounding vs stock** (class: our dg-mega "shared gemv" A3 case, `runtime/dg_mega.py:52-56`). It is
**deterministic** run-to-run, which peer3 is not (`ARCHITECTURE.md:121-122`). The component tag stays "changes
numerics" (vs stock); it is reproducible, unlike peer3.

**Bytes/time per step** (40 layers; weights only from Grace, SF resident; T = 6 window W = 68.7 µs, T = 1 window
W = 50.98 µs) [E, inputs §2.0]:

| variant | model per layer-call | 1-row ms/step | 6-row ms/step |
|---|---|---|---|
| serial, R = 385.8 GB/s | 18.6 + c·45.86 | 4.01 | 3.67 |
| serial, R = 230 | 18.6 + c·76.9 | 6.23 | 5.66 |
| concurrent, R = 385.8, ideal (no contention) | 3.34 + max(0, 18.6 + c·45.86 − W) | 2.32 | 1.64 |
| concurrent, R = 230 | same, 76.9 µs/expert | 4.54 | 3.58 |
| empirical overlap line (our probe, 137 µs GEMM window) | 3.34 + 22.07·c | 1.71 | 1.54 |

Bytes over C2C: 1.131 GB/step (6-row) / 1.262 GB/step (1-row) [D]. Cross-check: Garner books 1.4 GB/step of routed
cold experts at one stream (`ARCHITECTURE.md:16`) [M]. The concurrent rows assume the cold MegaMoE gets 230–386 GB/s
**from 8 SMs while a 144-SM MegaMoE runs**. That is not in tree, and it is the deciding measurement (§5, M7).

### 2.2 (b) stage-into-slots — the fixed-slot form (recommended)

**The idea.** A cold expert needs no *dynamic* slot index if every cold expert has a *fixed* group. What changes per
call is only which bytes sit behind the groups.

- Per target layer, MegaMoE is built with **G = 384** (stock geometry).
- Groups 0..H−1 hold the hot experts (HBM, as today). Groups H..383 hold cold expert c = cold-local id (`HK/:61-75`).
- The cold groups' weight bytes live in **one shared scratch** of 129 × 17.69 MB = 2.13 GiB [D]. CUDA VMM maps it at
  the VA right after *every* layer's hot block (`cuMemAddressReserve` + `cuMemMap` of the same physical allocation
  at 40 VAs). This is legal sharing because layers run strictly in stream order and a layer only reads cold groups
  the same call staged.
- Precedent: our dg-inplace already relies on "stale groups from earlier layers are never read" for its shared SF
  tensor (`runtime/dg_inplace.py:20-25`).
- The hot/cold boundary is not 2 MiB aligned (255 × 11,796,480 B is not a multiple of 2 MiB [D]). So the straddling
  page of each tensor stays a per-layer private page: ≤ 2 MiB × 2 weight tensors × 40 layers = 0.16 GiB [D].
- SF stay resident in HBM for all 384 groups. That means +5.31 GiB, and it keeps `a12_retire`'s banks static [D].

**Where it hooks:**

| T range | change | lines |
|---|---|---|
| ≤ 16 | The router runs with `p3 = None, peer = None` and gracetier's row_map (`row[e]` = hot slot, or H + c for cold; every entry ≥ 0). It sees no cold expert, writes every pick's group into `symm.topk_idx`, and publishes nothing (`has` = 0). | `B/:1051-1052` allows both None |
| ≤ 16 | Right after the router, on the current stream: `gt_stage` (one launch, §5 file 3). | after `B/:1365` |
| ≤ 16 | MegaMoE as today; **no finish, no fused finish, no COMBINE-in-seam** (the fork's own combine). | skip `B/:1380-1388` |
| boundary patch (small, in the Pg61 tree, never in `reference/`) | gracetier mode in `_moe_eligible`'s v2/v3 requirement | `B/:1158` |
| | the publish selection | `B/:1311-1339` |
| | the stage call | after `B/:1365` |
| | the finish branch | `B/:1380-1388` |
| 17–64 and > 64 | gracetier wraps `DeepseekV4MegaMoEExperts.forward` the way peer3 does: groups = row_map[topk_ids] → `gt_stage` → the hook's stock forward with groups as ids (`HK/:217`). | `P3/:174-194` pattern |
| drafter | untouched | |

**What MegaMoE has to accept:** nothing beyond stock. G = 384 is stock vLLM's geometry. The group index per expert is
fixed. The kernel reads weights only for groups named in `topk_idx`, so untouched cold groups (stale bytes) are never
read. This is the same property dg_mega uses (`runtime/dg_mega.py:37-45`). The VMM aliasing is invisible to
DeepGEMM: one contiguous VA per tensor, which MegaMoE requires (`runtime/dg_mega.py:18-21`).

**Graph capture.** `gt_stage` picks its sources on the device from the router's ids. That is a kernel copy, so it is
capturable; a copy-engine memcpy is not (`runtime/staging_overlap.py:28-39`; `docs/A2-RESIDENCY-DESIGN.md:159`). VMM
addresses are fixed before capture.

**Numerics: bit-identical to stock MegaMoE with all 384 experts in HBM.**

- Every group the kernel reads holds the checkpoint's bytes in the checkpoint's transform (P1 bytes,
  `docs/A2-RESIDENCY-DESIGN.md:172`).
- The combine is fp32 in top-k column order (`runtime/dg_mega.py:13-14`), which does not depend on the group index.
  This closes A2's open check (`A2-RESIDENCY-DESIGN.md:173`).
- The fork's schedule changes are bit-identical tile math (`fastpath_megamoe.py:3-4`).
- Expected class: **bit-identical** vs a 384-expert stock MegaMoE (gate: §5 M4).
- Compared with Pg61's control, outputs will differ, because peer3 is the side that "changes numerics".

**Bytes/time per step** [E, inputs §2.0]. Stage and MegaMoE are serial: MegaMoE must not start before the slots it
reads are written. Per layer-call: 2.82 µs launch + c·(45.86 µs copy + 3.07 µs MegaMoE per extra expert).

| map | c̄ | bytes/step over C2C | 1-row ms/step | 6-row ms/step |
|---|---|---|---|---|
| Garner static h255 | 1.783 (1-row) / 1.598 (6-row) | 1.262 / 1.131 GB | **3.60** | **3.24** |
| our residency, two_graphs cell (H = 295) | 1.19 | 0.84 GB | 2.44 | 2.44 |
| our residency, warm one_graph (H = 295) | 0.36 | 0.25 GB | 0.82 | 0.82 |

c̄ for the two residency rows: [M] `docs/M3-S2-STAGING-BYTES.md:50-52`, 6-row cells, our runtime with H = 295. With
Garner's H = 255 the c̄ is not in tree.

Marginal cost: each 1.0 distinct cold expert per layer-call = 40 × 48.93 µs = **1.96 ms/step** [D].

**Without VMM (fallback if §5 M1 fails).** Use per-layer spare groups G = H + S with compaction. This is our dg_mega
design: `l3_route_mega` maps cold picks to H + c over the call's unique cold experts and counts overflow
(`runtime/dg_mega.py:37-41`, `:428-470`). The costs:

- S × 0.70 GiB of HBM: S = 24 is 16.8 GiB, about −6.4 M KV tokens [D].
- A correctness fallback for c > S. Observed max per call: 12 at T=6, 22 at T=16, 31 at T=40 [M, `peer3_stats`
  hist]. T > 64 is unbounded.

The overflow path would have to be (a). This fallback is materially worse, and it is why M1 is the first item of the
mini-test.

### 2.3 Decision

| | (a) cold MegaMoE over UVA at the peer3 seam | (b) fixed-slot stage + one MegaMoE |
|---|---|---|
| exposed cold, 6-row (static map) | 3.67 serial; 1.5–3.6 concurrent, depending on the unmeasured 8-SM rate | **3.24** (serial, rate measured) |
| exposed cold, 1-row (static map) | 4.01 serial; 1.7–4.5 concurrent | **3.60** |
| numerics | +1 bf16 rounding, deterministic ("changes numerics") | **bit-identical to stock** |
| T coverage | ≤ 64 via the v3 segment; > 64 needs its own path | **any T**, one path |
| HBM | +5.31 GiB (cold SF) | +7.6 GiB (cold SF 5.31 + scratch 2.13 + straddle 0.16) |
| unverified platform facts | TMA over host-registered UVA; C2C rate from ≤ 8 SMs concurrent with MegaMoE | VMM aliasing under torch + DeepGEMM TMA; the router accepting an all-≥0 row_map with null publish pointers |
| SM co-residency hazard with MegaMoE's grid syncs (`B/:34`) | yes (concurrent variant) | none (serial on the main stream) |

**Recommend (b).** It is the only design here whose cost is built entirely from [M] rates. It keeps stock numerics,
covers every T including prefill and C32 widths with one code path, and adds no SM-sharing hazard next to a
grid-syncing kernel. (a) is not dead: its concurrent variant is the overlap upgrade. If M7 measures ≥ ~300 GB/s from
8 SMs beside the hot MegaMoE, the concurrent cold MegaMoE (on (b)'s Grace pool, same layout) would cut 6-row exposure
toward ~1.6 ms [E]. That becomes P1.1. The residency lever (§4) attacks c̄ itself and helps either design.

## 3. Loading: from "skip" to "read, then pin to Grace"

**Mechanism (no doubled host RAM):**

1. **Keep coldskip's filter for vLLM's iterator.** vLLM's safetensors path is the one with the 1.6× mmap peak
   (`docs/M0-PORT-PLAN.md:107`). The 30,960 cold tensors (`fastpath_coldskip.py:7-10`) stay out of it, so vLLM's peak
   does not grow by 90.3 GiB.
2. **gracetier reads the cold tensors itself.** It uses our loader's host path: `preadv` straight into the
   destination buffer, then `POSIX_FADV_DONTNEED` on each 64 MiB chunk (`runtime/ckpt.py:6-7,55,239-261`). No page
   cache is left behind and no transient copy lingers.
   - Destination: one `HostBuffer("cold_experts")` of exact size, hugepages (`runtime/host_tier.py:74-95`), allocated
     after `check_mem_available` (`runtime/host_tier.py:55`; our loader does the same sizing, `runtime/loader.py:393-400`).
   - Layout: per expert, MegaMoE's L1 rows interleaved + L2 rows (weights only, 17.69 MB). The interleave is a
     permutation of 8-row chunks, done by the reader on the CPU (`runtime/dg_mega.py:103-112`, pure).
   - The e8m0 scales go to the module's HBM SF banks at finalize, through the same transform the hot experts get.
   - Run it on a thread beside vLLM's own load and join before the first layer's finalize. `HK/_finalize_weights`
     already runs on the loader thread with a 600 s watchdog (`HK/:139-171`).
3. **Register** in ≤ 64 GiB chunks (`runtime/host_tier.py:31,35,158-178`): 2 chunks, about 13.3 s at the measured
   7.3 GB/s (90.3 GiB × 2^30 / 7.3e9) [D from the C2C probe registration row]. No copy may cross a chunk boundary
   (`runtime/host_tier.py:110-112`). Each expert must sit inside one chunk; pad at the 64 GiB boundary.
4. **Arming.** `coldskip`'s count check (`fastpath_coldskip.py:86-98`) is kept: gracetier asserts it *read* exactly
   the rowmap's 30,960 tensors.

**Host budget** (Station Grace; no swap):

| item | GiB | tag |
|---|---|---|
| Garner live server, total in use (Engram 2 × ~94 GiB pinned + 120 GB = 111.8 GiB host KV overflow + process) | 316.7 | [M] `G/README.md:61-63`, `ARCHITECTURE.md:41,174` |
| + cold experts (40 × 129 × 6 tensors) | 90.3 | [M] `fastpath_coldskip.py:9-10` |
| = gracetier, `DS41_OFFLOAD_GB` = 120 | **407.0** | [D] |
| MemAvailable on this host (nothing running) | 449.5 | [M] `docs/A2-RESIDENCY-DESIGN.md:133` (physical 494 GiB, `docs/HANDOFF-M0-PLAN.md:3`) |
| headroom | **42.5** | [D] — too thin for a swapless host under load |
| with `DS41_OFFLOAD_GB` = 60 (−55.9 GiB) | headroom 98.4 | [D] |

**What has to move: `DS41_OFFLOAD_GB` 120 → 60** (recommendation [E]). The port target is 1M KV
(`PORT-PLAN.md:39`). The GPU KV after (b) is about 8.5 M tokens: 11,372,409 − (7.6 GiB × ~380 K tokens/GiB)
(`G/docs/LIMITATIONS.md:68-69`) [D]. The host overflow is therefore not needed for capacity at the stated target. The
card verifies MemAvailable after boot. gracetier refuses to start below a stated floor; it does not trim.

**HBM budget for (b):** +5.31 (cold SF) + 2.13 (shared scratch) + 0.16 (straddle pages) = **+7.6 GiB**, about −2.9 M
KV tokens [D]. It must be allocated in `finalize_weights`, before vLLM's KV profiling, so `DS41_UTIL` sizing and
`a12_memory_guard` see it (`fastpath_a12_memory_guard.py:1-14`). The symm buffer at G = 384 vs 255 may grow too; its
size is not in tree, and §5 M2 prints it (`runtime/dg_mega.py:527`, `describe_symm`).

## 4. Residency

**Ship the static h255 map in v1.** The fixed-slot geometry already admits swaps without recapture.

- **Where the policy runs:** on the host, between steps. This is our `runtime/residency.py` model
  (`runtime/residency.py:10-31`):
  - the vendored v26ar.2 decision core (`runtime/residency_policy.py:1-10`)
  - counters drained every `drain_n` steps by async D2H
  - copies on a copy stream (copy engine allowed: the swap is host-chosen, 52.5 µs/expert [M C2C probe])
  - publish on the replay stream between two replays
  - In vLLM, "between steps" means a post-hook on `GPUModelRunner.execute_model` (the class `a12_retire` already
    hooks, `fastpath_a12_retire.py:14-15`).
  - Counters come free from `gt_stage`: it already scans the call's ids, and can bump `cnt[layer, e]` with integer
    atomics, which are order-independent (A2 §3.2's counter, `A2-RESIDENCY-DESIGN.md:145-147`).
- **What MegaMoE's pool must offer for a hot↔cold swap without re-capture.** All of these are in-place contents
  writes to fixed addresses, which a graph tolerates (`A2-RESIDENCY-DESIGN.md:161-167`).
  1. **row_map** (int32 [384], device, read by the router every call through a pointer). Promote e: `row[e]` ← its
     HBM group g. Demote e′: `row[e′]` ← H + c, where c is the scratch slot e vacated. Whether the router reads
     row_map at run time or caches it is not in tree (closed `.so`); §5 M2 checks it with two calls.
  2. **gracetier's source table** `src[layer][c]` = the Grace home of whoever owns cold slot c.
  3. **The bytes:** e's weights into group g (HBM, CE copy), and the two SF groups swapped inside the HBM banks. The
     banks are `a12_retire`'s compressible copies; the target pointers are read after retire (`a12_retire.py:1-12`).
  4. **A Grace home for e′** if it was boot-hot: one D2H into a demotion HostBuffer, in the same MegaMoE layout, so
     staging stays a plain copy (`runtime/residency.py:17-21` precedent).

  Order: publish e′ cold → wait an event behind the last replay that could read g → copy e into g → publish e hot.
  During the window e is staged on demand, which is correct by P1. With S_res = 0 spare hot groups, G stays 384 and
  `a12_retire`'s partition check is untouched (`a12_retire.py:48-56`). S_res > 0 needs that check patched.
- **The bigger lever is a larger hot set bought with KV.** Each extra hot expert per layer costs 0.70 GiB, about
  266 K KV tokens [D]. Going from H = 255 to our runtime's H = 295 condition costs about 28 GiB (≈ 10.6 M tokens), so
  H ≈ 280 is the realistic ceiling at a 1M-token floor [E]. Garner's "285-hot 1–2 % slower" (`ARCHITECTURE.md:45`)
  was measured *with* the RTX streaming the cold tail and does not apply here.

## 5. Mini-test plan (one ~1 h window, server not needed)

**Window.** Stopping `:30006` needs James's word and a pre-registered window. Pg61 is restored at the window end
(`PORT-PLAN.md:51`). The GB300 must be idle: Pg61 holds HBM at `DS41_UTIL` 0.95. No vLLM import anywhere: the probe
imports `deep_gemm` and `torch` only. The Garner router may be called through `fastpath_boundary.router()` from the
box's Pg61 copy, because `fastpath_hook` imports no vLLM (`fastpath_hook.py:6-12`).

**Files the coding worker creates (first build):**

1. `runtime/gracetier_vmm.py`
   - VA reserve
   - shared physical scratch mapped at N VAs, with private straddle pages
   - a torch view over a VA range (the `__cuda_array_interface__` pattern of `runtime/host_tier.py:66-72`)
   - release in reverse order
2. `runtime/gracetier_pool.py`
   - the Grace cold pool in MegaMoE per-expert layout (weights)
   - cold SF into the device banks
   - shard reads via `runtime/ckpt.py` (`read_tensor_into`, `ckpt.py:261`)
   - the source table and the row_map builder (`row[e]` = hot slot, or H + cold-local id)
3. `runtime/gracetier_stage.py`
   - Triton `gt_stage(groups, H, src_tab, dst_base, stride_l1, stride_l2, cnt)`: one launch, grid (129 slots ×
     NPROG). A program exits unless its slot is touched in this call's groups. It copies L1 + L2 weight blocks
     Grace → group H + c and bumps counters.
   - `stage_emulated` contract (the `runtime/dg_mega.py:336` pattern)
4. `bench/gracetier_minitest.py`: the box probe, subcommands `vmm | numerics | timing | r8`, one JSON receipt.
5. `bench/box/gracetier_minitest.sh`: box runner (image, lease, window guard), like `bench/box/m3n1.sh`.
6. `tests/test_gracetier_cpu.py`: CPU contracts. On mini shapes, the pool layout must equal the vendored
   `transform_weights_for_mega_moe`. Also: `stage_emulated` bytes, VMM offset and straddle math, row_map properties
   (a partition, every entry ≥ 0).
7. `tests/gpu/test_gracetier_gpu.py`: the GPU gate run in the window.

**Steps** (the [M] numbers to report are in bold):

| # | min | what | pass / report |
|---|---|---|---|
| M0 | 5 | preflight: `nvidia-smi` idle, MemAvailable, `date` | logged |
| M1 | 5 | VMM: one physical scratch mapped behind two "layer" VAs; write via one, read via the other; private straddle page; torch view; DeepGEMM MegaMoE (G = 384) on a VMM-backed L1/L2 vs the same bytes in a plain allocation | **`torch.equal`**; else stop and fall back to §2.2's compaction design |
| M2 | 10 | load layer 10: 129 cold experts (2.4 GB) from `/models` shards into a registered Grace pool (MegaMoE layout); 255 hot + all SF into HBM; also an **all-HBM 384-expert reference** for this one layer (≈ 6.8 GiB). Print `describe_symm` at G = 384 vs 255. Call `router()` twice with an all-≥0 row_map, changing one entry between the calls. | load/register s; symm bytes; router writes `topk_idx` = groups, `has` = 0, and reads row_map per call |
| M3 | 5 | activations: layer-10 MoE inputs from our runtime's reference forward on the 8-tag windows of `bench/dg_mega_numerics.py` (same checkpoint), plus synthetic routings that force c ∈ {0, 1, 2, 4, 8, 16} distinct cold at T ∈ {1, 6, 16, 64} | saved tensors |
| M4 | 10 | (b) numerics: MegaMoE(G = 384, VMM scratch filled by `gt_stage`) vs the all-HBM reference, every T and c | **`torch.equal` at all T** (gate); on failure, report rows exact / frac bytes equal the way `m3n1` does |
| M5 | 10 | (b) timing, graph replay: `gt_stage` µs vs c; MegaMoE G = 384 µs vs G = 255 at the same routing | **µs per cold expert (stage slope, MegaMoE marginal) at T = 1/6/16/64**; bar: stage slope ≤ 50 µs (model 45.86) |
| M6 | 10 | (a): cold MegaMoE (G_c = 129) with weights as **UVA views** of the Grace pool. First, does TMA over UVA run, and is it `torch.equal` to the HBM-weight copy? Then flip class of bf16(y_hot) + bf16(y_cold) vs the reference (max \|Δ\|, rows exact, frac bytes equal). Serial µs vs c at T = 1/6/16/64. | **µs per cold expert, flip class** |
| M7 | 5 | **R8**: a plain read kernel and `gt_stage` on {8, 16, 48, 152} SMs reading Grace: alone, and concurrent with the hot MegaMoE at `set_num_sms(144)`, T = 6; the hot MegaMoE's slowdown | **GB/s at 8 SMs under MegaMoE**; P1.1 bar: the concurrent (a) model with the measured rate ≤ 0.7 × (b)'s measured serial cost |

Build 2 (after M1 + M4 pass; next card) is `fastpath_gracetier.py` in the Pg61 tree. Our repo's port location is
not in tree; `PORT-PLAN.md` names Pg61 as the target. Build 2 contains:

- `@register("gracetier", "vllm.models.deepseek_v4.nvidia.model", defer_active=True)`
- the `HK` wraps (`_init` with G = 384 for MegaMoE, but loader params for H only, so HBM is not allocated for 384;
  `_weight_loader` cold branch = no-op, since the pool reader already holds the bytes; `_finalize_weights` = VMM
  tensors + hot transform copy-in + cold SF transform)
- the boundary patch of §2.2
- `components.py`: `fastpath_gracetier` replaces lines 28-29

## 6. Risks

1. **VMM aliasing is load-bearing for (b) and is not in tree.** Without it, (b) needs per-layer spare groups
   (16.8 GiB at S = 24) and an overflow path. M1 decides it first.
2. **The router is closed.** That it accepts row_map values ≥ n_hot, tolerates `p3 = peer = None`, and reads row_map
   at run time is inferred from its Python ABI (`B/:1051`, `HK/:61-75`), not from source (§0.5). M2 checks all three.
3. **C32.** At C32 the verify call is about 32 × 6–8 tokens, so T > 64. That is peer3's v2 regime (`P3/:181-183`;
   `G/docs/LIMITATIONS.md:73`), and no receipt in tree gives distinct cold experts per layer-call there.
   - Each 1.0 distinct cold costs 1.96 ms/step serial [D] against a 38.6 ms step [D]. At T = 33–64 the measured means
     are 4.6–12 (`peer3_stats`), which would cost 9–23 ms/step if they held at T ≈ 200 [D]. That is a 25–60 % C32
     loss at the static map.
   - Garner's own lesson applies: moving expert bytes onto a slower transfer path loses at multi-stream
     (`WHAT-DID-NOT-WORK.md:70-79`), and gracetier is the extreme case.
   - The C32 card comes **before** any verdict (`PORT-PLAN.md:39`). Residency (§4) is the mitigation.
   - The kernel shapes do not narrow with C: `gt_stage`'s grid is fixed at 129 × NPROG, and MegaMoE at G = 384 is
     stock. The narrow-kernel failure mode is therefore in the data volume, not in a shape cap.
4. **Prefill / TTFT.** At large T nearly every touched cold expert must cross C2C once per forward: ≤ 129 × 45.86 µs
   = 5.9 ms per layer, ≤ 237 ms per forward over 40 layers [D, upper bound]. Garner prefills at about 38–40 K tok/s
   (`G/README.md:48`), so an 8 K-token chunk could roughly double [D]. The actual touched count at T > 64 is not in
   tree.
5. **Graph capture of UVA reads.** Kernel reads are fine (our staging is captured). A copy-engine memcpy with a
   device-chosen source is not (`staging_overlap.py:28-39`). (a)'s fork must be joined in every breakable segment
   (`B/:1446-1464`). TMA over UVA (a only) is unverified (M6).
6. **`a12` guard and retire.**
   - The +7.6 GiB must exist before KV profiling, or the guard refuses after capture (`a12_memory_guard.py:5-11`).
     Allocate in `finalize_weights`.
   - `a12_retire` copies SF banks into compressible memory and asserts the originals released before capture
     (`a12_retire.py:1-12`). gracetier's cold-SF writes and later residency SF swaps must target the retired banks.
   - The partition check `n_hot + n_cold == 384` (`a12_retire.py:50-52`) holds with G = 384 (report n_hot = 384,
     n_cold = 0) and breaks for G > 384.
7. **Watchdogs.**
   - `HK`'s finalize watchdog kills at 600 s per layer (`HK/:152-158`); gracetier's per-layer finalize work is
     seconds [E].
   - Pg61's sidecar / p3 watchdog (launcher on the box; not in tree) must be disarmed or retargeted, or it will
     restart a healthy server that has no sidecar.
   - `PEER3_SPINS` timeouts disappear in (b); in (a) the GB300 writes `done` itself.
8. **WHAT-DID-NOT-WORK items that already killed similar ideas:**
   - §1 side-stream fork/join costs 2.3–4.3 µs/layer, and lines parked in L2 tax the next streaming kernel
     (`:13-24`). This applies to (a)-concurrent, not to (b).
   - §5 reserved SMs cost more than the overlap saved (`:63-68`). This applies to (a)-concurrent's 8-SM cold call.
   - §4 early TMA traffic raised the predecessor's grid-dependency wait from p90 0 to 4–5 µs (`:48-61`). (b)'s stage
     sits right before MegaMoE, but it is serial, so the 3.07 µs/expert marginal already absorbs it [E].
   - §6 transfer cost per byte eats the headroom at multi-stream (`:70-79`); see risk 3.
9. **Host memory.** 407 GiB pinned-or-used against 449.5 GiB available (§3) on a swapless host. Lower
   `DS41_OFFLOAD_GB` and assert MemAvailable at boot (`host_tier.check_mem_available`).
10. **Control comparability.** Pg61's peer3 deliberately changes numerics and gracetier (b) does not. The card's
    "TF flips within A/A2 spread" against a Pg61 control may fail for the right reason. Add a stock-numerics
    reference arm (all-HBM MegaMoE on a single layer is cheap; whole-model stock is impossible on 252 GiB).
11. **Input provenance.** The rowmap of the `peer3_stats` window is unknown. The 1-row c̄ rests on 120 calls. The
    T = 6/8 vs other-T cold-share gap is unexplained (§0.3). Every model number scales linearly with c̄.
12. **Link sharing.** Engram gathers (15.7 µs per 24 rows over UVA [M C2C probe]) and host-KV overflow traffic share
    C2C with staging. They are small at decode and unmeasured at C32.

## 7. Verdict

P1-DESIGN: b recommended; modeled exposed cold ms/step 1-row=3.60 6-row=3.24; first build=runtime/gracetier_vmm.py, runtime/gracetier_pool.py, runtime/gracetier_stage.py, bench/gracetier_minitest.py, bench/box/gracetier_minitest.sh, tests/test_gracetier_cpu.py, tests/gpu/test_gracetier_gpu.py
