"""M2-L3: the routed MoE (+ optionally the shared expert) as ONE DeepGEMM MegaMoE launch at world size 1
(`--moe-impl dg-mega`). Spec: docs/HANDOFF-M2-L3-MEGAMOE.md; design docs/M2-FUSED-CHAIN-DESIGN.md section 2.2 L3,
docs/T8-T9A-DESIGN.md 109-149; ABI findings docs/M2-L3-ABI-FINDINGS.md; spike bench/dg_mega_spike.py.

    st = StaticStep(model, pool, max_pos=4095, moe_impl="dg-mega")      # needs "dg-mega" in static_step.MOE_IMPLS

What the library wants (read from reference/vllm-0.31.0/deep_gemm: mega/__init__.py, layout/mega_moe.cuh,
impls/sm100_fp8_fp4_mega_moe.cuh), and how this module provides it from OUR pool:

  call     deep_gemm.fp8_fp4_mega_moe(y, (L1, L1_SF), (L2, L2_SF), sym_buffer, shared_l1, shared_l2,
           recipe=(1, 1, 32), activation="swiglu", activation_clamp=swiglu_limit, fast_math=False). One persistent
           launch with grid syncs: dispatch (per-expert token pools from topk_idx), L1, bf16 round + bf16 clamp,
           SwiGLU (`fast_math=False`: expf and an IEEE divide), * topk weight, fp8 requant FROM FP32 (amax per 32), L2,
           bf16, combine in fp32 over the top-k columns in column order (+ the shared expert last), bf16 out.
  group    world size 1: the Python touches only group.size(), .barrier(), .rank(); StubGroup below. No network.
  buffer   SymmBuffer(group, num_experts=G, num_max_tokens_per_rank=T, num_topk=k, hidden, intermediate_hidden,
           num_shared_experts) -- G (= H + S) and T are TEMPLATE constants of the kernel (one JIT per distinct G).
           Inputs are written into its views: x (fp8 [T, D]), x_sf (per-token UE8M0, layout printed by the spike),
           topk_idx (int64 [T, k], -1 = skip), topk_weights (fp32 [T, k]).
  L2       w2 as [G, D, I/2] int8 packed e2m1 = the POOL's own w2 tensor (pool.hbm_full[l]["w2"]), used IN PLACE.
           The kernel's weight TMA addresses expert e at row e * shape_n of ONE 2-D tensor (cuh:775), so it must be
           one contiguous [G, N, K/2] tensor -- which the pool's per-projection tensors are.
  L1       w1 and w3 as ONE [G, 2I, D/2] tensor whose rows are gate/up INTERLEAVED at granularity 8
           (mega/__init__.py _interleave_weights: rows [g0..7, u0..7, g8..15, u8..15, ...]). The pool holds w1 and
           w3 as two separate [G, I, D/2] tensors that the reference Experts, the eager prefill, `ref`, dg-inplace,
           the residency manager and the cold host tier all alias row-contiguously. No view of two tensors is one
           uniformly strided tensor, and no view of an interleaved tensor gives the Experts a contiguous w1 back:
           THIS IS THE ABI BLOCKER for the in-place build (l1_store "inplace": not implemented here, it needs the
           pool re-layout described in docs/M2-L3-ABI-FINDINGS.md). l1_store "copy" (default) builds an interleaved
           COPY of every layer's [G, 2I, D/2] (3.4 GiB/layer on the real model = 136.7 GiB: refused by the HBM check;
           trivially affordable on the mini checkpoints, which is what the GPU gates run on).
  SF       per-row x 32-K UE8M0 codes packed 4 per int32 (= the pool's e8m0 bytes read as little-endian int32,
           dg_inplace.pack_words), the L1 rows interleaved as the weights, then the UTCCP 4x32 row transpose inside
           every 128-row group (_transpose_sf_for_utccp), stored MN-major ([G][K/128][N], the SF TMA's inner
           coordinate is n: cuh:771-780) -- sf_major "k" (row-major words) is the alternative the spike also tries.
           A per-layer copy (12.8 GiB on the real model, the same bytes as dg-inplace's "pool" SF store it replaces).
  routing  ours (fp32 gate, top-6, ascending expert id per row = the reference's combine order, model.py:897).
           topk_idx[r, j] = the POOL GROUP of expert ids[r, j]: its hot slot, or H + c for a cold expert with compact
           cold index c (ascending id over the step's UNIQUE cold experts, c < S), or -1 if c >= S (overflow:
           counted in the owner's ovf_step, that token's MoE output wrong for the step, never silent). The kernel
           combines the top-k columns in column order, so ascending-id columns = the reference's fp32 sum order.
  stage    ONE Triton launch per layer (l3_stage_mega; contract stage_mega_emulated): the S staging slots' cold
           experts -> pool group H + c: w1/w3 rows interleaved into the L1 copy, w2 rows into the pool's w2 (in
           place), scales packed + interleaved + UTCCP-permuted into the SF copies. Cold sources are the same
           pointer tables as dg-inplace (HBM pool slot or UVA host address).
  rows     every call is made with T token rows (T = $DSV41_MEGA_MAX_TOKENS, default max(6, rows)): rows >= R carry
           topk_idx -1 and x 0 (skipped by dispatch, zero in combine), so the 1-row and the 6-row step launch the
           SAME kernel instance with the SAME num_tokens -- row invariance is then the kernel's per-token
           independence, which the GPU test and the spike measure.
  shared   "gemv" (default, the handoff's rule): MegaMoE runs the routed experts only and returns bf16; the shared
           expert stays on our path and is added as y = bf16(fp32(y_routed_bf16) + fp32(shared_bf16)) -- one extra
           bf16 rounding vs the reference's single fp32 sum (A3, reported by the drift numbers). "fused": the
           library's shared expert (fp8 weights, the 32x32 block scales expanded per row -- exact), combined last in
           fp32 by the kernel (single rounding, the reference's order) but computed by the block-scaled UMMA instead
           of our GEMV; adopted only if the spike shows it bit-equal to the GEMV (it will not be: different
           accumulation), else kept as a measured option.

Numerics class: A3 (block-scaled UMMA accumulators, as dg-inplace; plus the L2-input quantization taken from fp32
instead of bf16 -- T8-T9A-DESIGN.md:117, 121-124; plus, shared "gemv", the extra bf16 rounding above).
"""
from __future__ import annotations

import os
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # GPU only; the Mac imports the pure functions (tests/triton_shim.py runs the kernel body)
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - Mac
    triton = tl = None
    HAVE_TRITON = False

GRAN = 8                       # gate/up interleave granularity (mega/__init__.py _interleave_weights)
UTCCP = 128                    # UTCCP transpose group (4 x 32 rows, _transpose_sf_for_utccp)
SF_PACK = 4                    # e8m0 codes per int32 word
ACT_BLOCK = 32
MEGA_RECIPE = (1, 1, 32)       # per-row x 32-K for activations and weights (mega/__init__.py:183)
SHARED_MODES = ("gemv", "fused")
L1_STORES = ("copy", "inplace")
SF_MAJORS = ("mn", "k")
SHARED_ENV = "DSV41_MEGA_SHARED"
L1_STORE_ENV = "DSV41_MEGA_L1_STORE"
SF_MAJOR_ENV = "DSV41_MEGA_SF_MAJOR"
MAX_TOKENS_ENV = "DSV41_MEGA_MAX_TOKENS"
SPARE_ENV = "DSV41_MEGA_SPARE"
HEADROOM_ENV = "DSV41_MEGA_HEADROOM_GIB"
HEADROOM_GIB = 6.0
MAX_TOKENS_DEFAULT = 6         # the DSpark verify step's rows: the 1-row step launches the 6-row kernel instance
STAGE_NPROG = int(os.environ.get("DSV41_MEGA_STAGE_NPROG", "48"))
STAGE_BN = 64                  # scale rows per tile
STAGE_WARPS = 4
GiB = 2 ** 30
FINDINGS = "docs/M2-L3-ABI-FINDINGS.md"


# ------------------------------------------------------------------------------------------------ layout math (pure)
def l1_row(r: int, p: int, gran: int = GRAN) -> int:
    """Row of the interleaved L1 tensor that holds row r of projection p (0 = w1 / gate, 1 = w3 / up)."""
    return (r // gran) * 2 * gran + p * gran + r % gran


def l1_source(n: int, gran: int = GRAN) -> Tuple[int, int]:
    """Inverse of l1_row: interleaved row n -> (p, r)."""
    return (n // gran) % 2, (n // (2 * gran)) * gran + n % gran


def utccp_pos(n: int) -> int:
    """New row index of old row n under _transpose_sf_for_utccp (reshape(-1, 4, 32).transpose(.., 4) per 128 rows:
    old (a, i, j) = a*128 + i*32 + j -> new a*128 + j*4 + i)."""
    a, o = divmod(int(n), UTCCP)
    return a * UTCCP + (o % 32) * 4 + o // 32


def utccp_src(m: int) -> int:
    """Old row stored at new row m (inverse of utccp_pos)."""
    a, o = divmod(int(m), UTCCP)
    return a * UTCCP + (o % 4) * 32 + o // 4


def _with_group(t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if t.dim() not in (2, 3):
        raise ValueError(f"expected a 2-D or 3-D tensor, got {tuple(t.shape)}")
    return (t.unsqueeze(0), True) if t.dim() == 2 else (t, False)


def interleave_rows(t: torch.Tensor, gran: int = GRAN) -> torch.Tensor:
    """deep_gemm.mega._interleave_weights, re-implemented (same ops, same result incl. strides): [.., 2I, *rest] with
    rows [gate | up] -> [gate 0..7, up 0..7, gate 8..15, ...]. tests/test_dg_mega_cpu.py checks it against the
    vendored function's source."""
    t, sq = _with_group(t)
    g, n, *rest = t.shape
    if n % (2 * gran):
        raise ValueError(f"{n} rows: not a multiple of 2 x {gran}")
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    out = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return out.squeeze(0) if sq else out


def deinterleave_rows(t: torch.Tensor, gran: int = GRAN) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of interleave_rows -> (gate rows, up rows), new contiguous tensors."""
    t, sq = _with_group(t)
    g, n, *rest = t.shape
    v = t.reshape(g, n // (2 * gran), 2, gran, *rest)
    gate = v[:, :, 0].reshape(g, n // 2, *rest).contiguous()
    up = v[:, :, 1].reshape(g, n // 2, *rest).contiguous()
    return (gate.squeeze(0), up.squeeze(0)) if sq else (gate, up)


def utccp_rows(sf: torch.Tensor) -> torch.Tensor:
    """deep_gemm.mega._transpose_sf_for_utccp, re-implemented: rows permuted inside each 128-row group (4 x 32
    transpose); output keeps the input's strides (empty_like)."""
    sf, sq = _with_group(sf)
    g, mn, k = sf.shape
    if mn % UTCCP:
        raise ValueError(f"{mn} rows: not a multiple of {UTCCP} (UTCCP)")
    r = sf.reshape(g, -1, 4, 32, k).transpose(2, 3).reshape(g, mn, k)
    out = torch.empty_like(sf).copy_(r)
    return out.squeeze(0) if sq else out


def utccp_rows_inverse(sf: torch.Tensor) -> torch.Tensor:
    sf, sq = _with_group(sf)
    g, mn, k = sf.shape
    r = sf.reshape(g, -1, 32, 4, k).transpose(2, 3).reshape(g, mn, k)
    out = torch.empty_like(sf).copy_(r)
    return out.squeeze(0) if sq else out


def _u8(s: torch.Tensor) -> torch.Tensor:
    return s if s.dtype == torch.uint8 else s.view(torch.uint8)


def sf_words(codes: torch.Tensor) -> torch.Tensor:
    """e8m0 codes [..., N, KS] (uint8 / float8_e8m0fnu) -> int32 words [..., N, KS/4], row-major (K-major), the same
    bytes (dg_inplace.pack_words: DeepGEMM's pack_ue8m0_to_int)."""
    u = _u8(codes).contiguous()
    if u.size(-1) % SF_PACK:
        raise ValueError(f"{u.size(-1)} scales per row is not a multiple of {SF_PACK}")
    return u.view(torch.int32)


def words_to_codes(words: torch.Tensor) -> torch.Tensor:
    """int32 [..., N, C] (any strides) -> uint8 [..., N, 4C]."""
    p = torch.empty(words.shape, dtype=torch.int32, device=words.device).copy_(words)
    return p.view(torch.uint8)


def to_major(words: torch.Tensor, major: str) -> torch.Tensor:
    """K-major int32 words [..., N, C] -> a new tensor in the requested storage: "mn" = [..][C][N] storage viewed as
    [.., N, C] (strides (.., 1, N): DeepGEMM's MN-major packed SF), "k" = contiguous."""
    if major == "k":
        return words.contiguous()
    if major == "mn":
        return words.transpose(-1, -2).contiguous().transpose(-1, -2)
    raise ValueError(f"sf major {major!r} not in {SF_MAJORS}")


def mega_l1_weight(w1: torch.Tensor, w3: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L1 [G, 2I, K/2] interleaved from w1 [G, I, K/2] and w3 (any dtype; written chunk-wise into `out` without a
    [w1 | w3] temporary -- the same bytes as interleave_rows(cat([w1, w3], 1)))."""
    sq = w1.dim() == 2
    a, b = (w1.unsqueeze(0), w3.unsqueeze(0)) if sq else (w1, w3)
    G, I, *rest = a.shape
    if I % GRAN or tuple(b.shape) != tuple(a.shape):
        raise ValueError(f"w1 {tuple(a.shape)} / w3 {tuple(b.shape)}: equal shapes, rows a multiple of {GRAN}")
    if out is None:
        out = torch.empty(G, 2 * I, *rest, dtype=a.dtype, device=a.device)
        if sq:
            out = out.squeeze(0)
    o = out.unsqueeze(0) if out.dim() == len(rest) + 1 else out
    v = o.view(G, I // GRAN, 2, GRAN, *rest)
    v[:, :, 0].copy_(a.reshape(G, I // GRAN, GRAN, *rest))
    v[:, :, 1].copy_(b.reshape(G, I // GRAN, GRAN, *rest))
    return out


def mega_l1_sf(c1: torch.Tensor, c3: torch.Tensor, major: str = "mn") -> torch.Tensor:
    """L1 SF from the w1 / w3 e8m0 codes [G, I, K/32] (or 2-D): words, interleaved rows, UTCCP, `major` storage."""
    w = torch.cat([sf_words(c1), sf_words(c3)], dim=-2)
    return to_major(utccp_rows(interleave_rows(w)), major)


def mega_l2_sf(c2: torch.Tensor, major: str = "mn") -> torch.Tensor:
    return to_major(utccp_rows(sf_words(c2)), major)


def l1_sf_codes_of(sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of mega_l1_sf (any storage) -> (w1 codes, w3 codes) uint8 [G, I, K/32] (tests, the CPU fake)."""
    w = utccp_rows_inverse(words_to_codes(sf).view(torch.int32))
    g, u = deinterleave_rows(w)
    return words_to_codes(g), words_to_codes(u)


def l2_sf_codes_of(sf: torch.Tensor) -> torch.Tensor:
    return words_to_codes(utccp_rows_inverse(words_to_codes(sf).view(torch.int32)))


def block_codes_per_row(scale: torch.Tensor, n: int, block: int = 32) -> torch.Tensor:
    """The fp8 Linears' [ceil(N/32), K/32] block scale codes -> per-row codes [N, K/32] (repeat_interleave: exact,
    the T9a-1 fp8 addendum's re-tiling for recipe (1, 32))."""
    return _u8(scale).repeat_interleave(block, 0)[:n].contiguous()


def act_codes(s: torch.Tensor) -> torch.Tensor:
    """act_quant's scales (e8m0 / uint8, or fp32 powers of two) -> uint8 e8m0 codes, exact."""
    if s.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return _u8(s)
    if s.dtype == torch.float32:
        return ((s.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    raise TypeError(f"activation scales of dtype {s.dtype}")


def write_act_sf(view: torch.Tensor, codes: torch.Tensor, R: int) -> None:
    """codes uint8 [R, D/32] -> rows [:R] of the SymmBuffer's x_sf view in whatever dtype / packing it has (int32 =
    4 codes per word, little-endian; uint8 / e8m0 = one code per byte; fp32 = the exact value 2^(e-127))."""
    if view.dtype == torch.int32:
        view[:R].copy_(codes.contiguous().view(torch.int32).view(R, -1))
    elif view.dtype in (torch.uint8, torch.float8_e8m0fnu):
        view.view(torch.uint8)[:R].copy_(codes.view(R, -1))
    elif view.dtype == torch.float32:
        from runtime.dg_moe import E8M0_F32

        view[:R].copy_(E8M0_F32.to(codes.device)[codes.view(R, -1).long()])
    else:
        raise TypeError(f"x_sf view dtype {view.dtype} (shape {tuple(view.shape)}): unknown packing")


# ------------------------------------------------------------------------------------------------ routing (device)
def mega_route(ids: torch.Tensor, gtab: torch.Tensor, H: int, S: int, n_experts: int) -> Dict[str, torch.Tensor]:
    """ids int64 [R, k] (ascending per row) -> {"topk" int64 [R, k] pool groups (-1 = overflow), "stage" int64 [S]
    (the cold expert staged into spare slot c, -1 = none), "n_cold" / "n_over" int64 [1]}. Fixed shapes, no host
    sync, no atomics: a cold expert's compact index c = its rank among the step's UNIQUE cold experts by ascending
    id (the same expert in several rows shares its slot)."""
    dev = ids.device
    E = int(n_experts)
    flat = ids.reshape(-1)
    g = gtab.long()[flat]
    is_cold = g < 0
    ar = torch.arange(E, dtype=torch.long, device=dev)
    hit = (flat.view(-1, 1) == ar.view(1, -1)) & is_cold.view(-1, 1)            # [P, E]
    cold_e = hit.any(0)
    c_e = torch.cumsum(cold_e.long(), 0) - 1
    c = c_e[flat]
    over = is_cold & (c >= S)
    grp = torch.where(is_cold, c + int(H), g)
    topk = torch.where(over, torch.full_like(grp, -1), grp).view(ids.shape)
    js = torch.arange(int(S), dtype=torch.long, device=dev).view(-1, 1)
    eq = (c_e.view(1, -1) == js) & cold_e.view(1, -1)                          # [S, E]
    stage = torch.where(eq.any(1), (eq.long() * ar.view(1, -1)).sum(1), torch.full_like(js.view(-1), -1))
    n_cold = cold_e.long().sum().view(1)
    return {"topk": topk, "stage": stage, "n_cold": n_cold, "n_over": torch.clamp(n_cold - int(S), min=0)}


def mega_route_host(ids: Sequence[Sequence[int]], gtab: Sequence[int], H: int, S: int) -> dict:
    """Plain-Python mirror of mega_route (tests)."""
    cold = sorted({int(e) for r in ids for e in r if int(gtab[int(e)]) < 0})
    cidx = {e: i for i, e in enumerate(cold)}
    topk = []
    for r in ids:
        row = []
        for e in r:
            g = int(gtab[int(e)])
            if g >= 0:
                row.append(g)
            else:
                c = cidx[int(e)]
                row.append(H + c if c < S else -1)
        topk.append(row)
    stage = [cold[j] if j < len(cold) else -1 for j in range(S)]
    return {"topk": topk, "stage": stage, "n_cold": len(cold), "n_over": max(0, len(cold) - S)}


def unique_ids_record(ids: torch.Tensor, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The owner's per-layer counters at R rows: (sorted flattened ids, 1 where an id first occurs) padded to
    `width` (the R-row step's ids_step / uvalid_step width R*k)."""
    s = torch.sort(ids.reshape(-1)).values
    first = torch.ones_like(s)
    if s.numel() > 1:
        first[1:] = (s[1:] != s[:-1]).long()
    if s.numel() < width:
        pad = width - s.numel()
        s = torch.cat([s, s[-1:].expand(pad)])
        first = torch.cat([first, torch.zeros(pad, dtype=first.dtype, device=first.device)])
    return s[:width], first[:width]


# ------------------------------------------------------------------------------------------------ staging contract
def stage_mega_emulated(resolve, stage: torch.Tensor, ptab6: torch.Tensor, dsts: Sequence[torch.Tensor], H: int,
                        sf_major: str = "mn") -> None:
    """The contract of l3_stage_mega in torch through `resolve(address) -> uint8 tensor of the source row` (CPU tests,
    the CPU build, the GPU test's oracle). dsts = [L1 [G, 2I, K/2] int8, L2 [G, D, I/2] int8, SF1 int32 [G, 2I, C1],
    SF2 int32 [G, D, C2]] (SF in `sf_major` storage). For spare slot c with e = stage[c] >= 0: group H + c gets
    expert e's w1 / w3 rows interleaved into L1, its w2 rows into L2, its scales into SF1 / SF2 in the mega layout.
    Nothing else is written."""
    L1, L2, SF1, SF2 = dsts
    I2, KH1 = L1.shape[1:]
    I = I2 // 2
    D, KH2 = L2.shape[1:]
    for c, e in enumerate(int(v) for v in stage.tolist()):
        if e < 0:
            continue
        grp = H + c
        w = [resolve(int(ptab6[p, e])).view(-1) for p in range(3)]
        s = [resolve(int(ptab6[3 + p, e])).view(-1) for p in range(3)]
        mega_l1_weight(w[0][:I * KH1].view(torch.int8).view(I, KH1), w[1][:I * KH1].view(torch.int8).view(I, KH1),
                       out=L1[grp])
        L2[grp].view(torch.uint8).view(-1).copy_(w[2][:D * KH2])
        KS1 = SF1.size(-1) * SF_PACK
        KS2 = SF2.size(-1) * SF_PACK
        SF1[grp].copy_(mega_l1_sf(s[0][:I * KS1].view(I, KS1), s[1][:I * KS1].view(I, KS1), "k"))
        SF2[grp].copy_(mega_l2_sf(s[2][:D * KS2].view(D, KS2), "k"))


# ------------------------------------------------------------------------------------------------ staging kernel
if HAVE_TRITON:
    @triton.jit
    def l3_stage_mega(stage, ptab, dtab, E, H, N13, C13, N2, C2,
                      CHW: tl.constexpr, NCH: tl.constexpr, W2W: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr,
                      BC: tl.constexpr, NPROG: tl.constexpr, SF_MN: tl.constexpr):
        """M2-L3 staging, ONE launch per layer (contract: stage_mega_emulated). Program (c, q, p): e = stage[c]
        (-1: exit). Group grp = H + c. Weights (int64 words, 16-B vectors): p in {0, 1} (w1 / w3) copies 8-row chunks
        ch = q, q+NPROG, ... (CHW words each, NCH = I/8 chunks) of row ptab[p, e] to chunk 2*ch + p of L1 slot grp
        (the gate/up interleave); p = 2 copies the w2 row (W2W words) into L2 slot grp. Scales (int32 words of 4 e8m0
        codes): row r of projection p -> interleaved row (p < 2) -> UTCCP position n_d in a 128-row group -> SF1 / SF2
        group grp, MN-major ([G][C][N]) if SF_MN else row-major ([G][N][C]); scale tiles of BN rows, tile t by
        program q = t % NPROG. Each arm uses its own variable names (Triton unifies same-named variables)."""
        c = tl.program_id(0)
        q = tl.program_id(1)
        p = tl.program_id(2)
        e = tl.load(stage + c)
        if e >= 0:
            grp = (e * 0 + c + H).to(tl.int64)
            if p == 2:
                a_src = tl.multiple_of(tl.load(ptab + 2 * E + e).to(tl.pointer_type(tl.int64)), 16)
                a_dst = tl.multiple_of(tl.load(dtab + 1).to(tl.pointer_type(tl.int64)), 16) + grp * W2W
                for ita in range(0, ((W2W + BW - 1) // BW + NPROG - 1) // NPROG):
                    a_off = (q + ita * NPROG) * BW + tl.arange(0, BW)
                    a_m = a_off < W2W
                    tl.store(a_dst + a_off, tl.load(a_src + a_off, mask=a_m), mask=a_m)
            else:
                b_src = tl.multiple_of(tl.load(ptab + p * E + e).to(tl.pointer_type(tl.int64)), 16)
                b_dst = tl.multiple_of(tl.load(dtab + 0).to(tl.pointer_type(tl.int64)), 16) + grp * (2 * NCH * CHW)
                for itb in range(0, (NCH + NPROG - 1) // NPROG):
                    ch = q + itb * NPROG
                    if ch < NCH:
                        for bb in range(0, CHW // BW):
                            b_off = bb * BW + tl.arange(0, BW)
                            tl.store(b_dst + (2 * ch + p) * CHW + b_off, tl.load(b_src + ch * CHW + b_off))
            # scales
            is2 = p == 2
            Np = tl.where(is2, N2, N13)
            C = tl.where(is2, C2, C13)
            Nd = tl.where(is2, N2, 2 * N13)
            s_src = tl.load(ptab + (3 + p) * E + e).to(tl.pointer_type(tl.int32))
            s_dst = tl.load(dtab + tl.where(is2, 3, 2)).to(tl.pointer_type(tl.int32)) + grp * (C * Nd)
            cc = tl.arange(0, BC)
            cm = cc < C
            nb = (Np + BN - 1) // BN
            for its in range(0, (nb + NPROG - 1) // NPROG):
                r = (q + its * NPROG) * BN + tl.arange(0, BN)
                ni = tl.where(is2, r, (r // 8) * 16 + p * 8 + r % 8)
                nd = (ni // 128) * 128 + (ni % 32) * 4 + (ni % 128) // 32
                sm = (r < Np)[:, None] & cm[None, :]
                v = tl.load(s_src + r[:, None] * C + cc[None, :], mask=sm)
                if SF_MN:
                    tl.store(s_dst + cc[None, :] * Nd + nd[:, None], v, mask=sm)
                else:
                    tl.store(s_dst + nd[:, None] * C + cc[None, :], v, mask=sm)


def _pow2_divisor(n: int, cap: int) -> int:
    b = 1
    while b * 2 <= cap and n % (b * 2) == 0:
        b *= 2
    return b


# ------------------------------------------------------------------------------------------------ routing kernel
if HAVE_TRITON:
    @triton.jit
    def l3_route_mega(ids, w, gtab, topk_out, w_out, stage_out, ovf, H, S,
                      P: tl.constexpr, PB: tl.constexpr, EB: tl.constexpr, SB: tl.constexpr, HAS_OVF: tl.constexpr):
        """mega_route in ONE program (its torch form is the mirror): the P = R*k routed pairs (ids ascending per row,
        int64), their pool groups -> topk_out (the SymmBuffer's topk_idx rows [:R]), the weights -> w_out, the
        compact cold index over the UNIQUE cold experts (a [PB, EB] membership matrix, cumsum over experts), the
        staging list -> stage_out [S], HAS_OVF: one atomic add of max(n_cold - S, 0) to the step's counter."""
        pi = tl.arange(0, PB)
        pm = pi < P
        e = tl.load(ids + pi, mask=pm, other=0)
        g = tl.load(gtab + e, mask=pm, other=0).to(tl.int64)
        cold = (g < 0) & pm
        ei = tl.arange(0, EB)
        same = e[:, None] == ei[None, :]
        cold_e = tl.max((same & cold[:, None]).to(tl.int32), axis=0)
        c_e = tl.cumsum(cold_e, axis=0) - 1
        c = tl.sum(tl.where(same, c_e[None, :], 0), axis=1).to(tl.int64)
        over = cold & (c >= S)
        grp = tl.where(cold, c + H, g)
        tl.store(topk_out + pi, tl.where(over, grp * 0 - 1, grp), mask=pm)
        tl.store(w_out + pi, tl.load(w + pi, mask=pm, other=0.0), mask=pm)
        sj = tl.arange(0, SB)
        eqs = (c_e[None, :] == sj[:, None]) & (cold_e[None, :] > 0)
        has = tl.max(eqs.to(tl.int32), axis=1)
        ev = tl.sum(tl.where(eqs, ei[None, :], 0), axis=1)
        tl.store(stage_out + sj, tl.where(has > 0, ev, -1).to(tl.int64), mask=sj < S)
        if HAS_OVF:
            n_cold = tl.sum(cold_e, axis=0)
            tl.atomic_add(ovf, tl.maximum(n_cold - S, 0))


def route_mega(ids: torch.Tensor, w: torch.Tensor, gtab: torch.Tensor, topk_out: torch.Tensor,
               w_out: torch.Tensor, stage_out: torch.Tensor, H: int, S: int, n_experts: int,
               ovf: Optional[torch.Tensor] = None) -> None:
    """Launch l3_route_mega (contract: mega_route + the copies DGMegaBuffers.route does). ids / w [R, k] contiguous;
    topk_out int64 / w_out fp32 contiguous with >= R*k leading elements; stage_out int64 [S]."""
    P = ids.numel()
    if not (ids.is_contiguous() and w.is_contiguous() and topk_out.is_contiguous() and w_out.is_contiguous()):
        raise ValueError("route_mega: ids / w / topk_out / w_out must be contiguous")
    l3_route_mega[(1,)](ids, w, gtab, topk_out, w_out, stage_out, ovf if ovf is not None else stage_out, int(H),
                        int(S), P, triton.next_power_of_2(P), triton.next_power_of_2(int(n_experts)),
                        triton.next_power_of_2(int(S)), ovf is not None, num_warps=4)


def stage_mega(stage: torch.Tensor, ptab6: torch.Tensor, dtab: torch.Tensor, dsts: Sequence[torch.Tensor], H: int,
               sf_major: str = "mn", nprog: Optional[int] = None, block_words: int = 512,
               num_warps: Optional[int] = None) -> None:
    """Launch l3_stage_mega over the S staging slots (contract: stage_mega_emulated). Rows 16-B aligned (checked)."""
    L1, L2, SF1, SF2 = dsts
    I2, KH1 = L1.shape[1:]
    D, KH2 = L2.shape[1:]
    I = I2 // 2
    if (GRAN * KH1) % 8 or (D * KH2) % 8 or any(t.data_ptr() % 16 for t in (L1, L2)):
        raise ValueError("stage_mega: weight rows / destinations not 16-B aligned")
    chw = GRAN * KH1 // 8
    w2w = D * KH2 // 8
    bw = _pow2_divisor(chw, int(block_words))
    if bw < 2:
        raise ValueError(f"stage_mega: chunk of {chw} words has no power-of-two block >= 2")
    c13, c2 = SF1.size(-1), SF2.size(-1)
    bc = triton.next_power_of_2(max(int(c13), int(c2)))
    npg = int(nprog or STAGE_NPROG)
    l3_stage_mega[(stage.numel(), npg, 3)](stage, ptab6, dtab, ptab6.size(1), int(H), int(I), int(c13), int(D),
                                           int(c2), chw, I // GRAN, w2w, bw, STAGE_BN, bc, npg, sf_major == "mn",
                                           num_warps=int(num_warps or STAGE_WARPS))


# ------------------------------------------------------------------------------------------------ world size 1
class StubGroup:
    """The process-group surface deep_gemm.mega touches at world size 1: size(), rank(), barrier()
    (mega/__init__.py:52-67, 196). No torch.distributed init, no network."""

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        return None


def make_symm_buffer(dg, G: int, T: int, k: int, dim: int, inter: int, n_shared: int):
    """SymmBuffer at world size 1 (the vendored signature; the older get_symm_buffer_for_mega_moe as fallback)."""
    grp = StubGroup()
    if hasattr(dg, "SymmBuffer"):
        try:
            return dg.SymmBuffer(grp, int(G), int(T), int(k), int(dim), int(inter), num_shared_experts=int(n_shared),
                                 mma_type="fp8xfp4", activation="swiglu")
        except TypeError:
            return dg.SymmBuffer(grp, int(G), int(T), int(k), int(dim), int(inter), num_shared_experts=int(n_shared))
    return dg.get_symm_buffer_for_mega_moe(grp, int(G), int(T), int(k), int(dim), int(inter),
                                           num_shared_experts=int(n_shared))


SYMM_VIEWS = ("x", "x_sf", "topk_idx", "topk_weights", "shared_l1_acts", "shared_l1_acts_sf", "shared_l2_acts",
              "shared_l2_acts_sf", "l1_acts", "l1_acts_sf", "l2_acts", "l2_acts_sf")


def describe_symm(sym) -> dict:
    """shape / dtype / strides of every SymmBuffer view (the E2 question, T8-T9A-DESIGN.md:199)."""
    out = {"num_bytes": int(sym.buffer.numel()) if getattr(sym, "buffer", None) is not None else None}
    for n in SYMM_VIEWS:
        v = getattr(sym, n, None)
        out[n] = None if v is None else {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
    for n in ("num_experts", "num_max_tokens_per_rank", "num_topk", "hidden", "intermediate_hidden",
              "num_shared_experts"):
        out[n] = getattr(sym, n, None)
    return out


def hbm_bytes(n_layers: int, G: Sequence[int], dim: int, inter: int, l1_store: str) -> dict:
    """What a dg-mega build adds (bytes): the L1 copy (l1_store "copy"), the SF copies (always), per the shapes."""
    l1 = sum(int(g) * 2 * inter * dim // 2 for g in G) if l1_store == "copy" else 0
    sf = sum(int(g) * (2 * inter * dim // 32 + dim * inter // 32) for g in G)
    return {"l1_copy_B": l1, "sf_copy_B": sf, "total_B": l1 + sf, "l1_copy_GiB": round(l1 / GiB, 3),
            "sf_copy_GiB": round(sf / GiB, 3), "total_GiB": round((l1 + sf) / GiB, 3), "n_layers": n_layers}


# ------------------------------------------------------------------------------------------------ the buffers
class DGMegaBuffers:
    """Per-StaticStep MegaMoE state: per layer the L1 copy (or, someday, the in-place re-laid-out pool), the pool's
    w2 used in place, the SF copies, the routing tables; per distinct G one SymmBuffer; the output y [T, D]."""

    def __init__(self, owner, shapes, sshapes, *, pool, swiglu_limit: float, spare_slots: int,
                 max_tokens: Optional[int] = None, shared: Optional[str] = None, sf_major: Optional[str] = None,
                 l1_store: Optional[str] = None, deterministic: bool = True, dg=None):
        if pool is None:
            raise ValueError("dg-mega needs the ExpertPool (it reads the pool's w2 in place)")
        self.owner, self.pool, self.dev = owner, pool, owner.dev
        self.k = int(owner.topk)
        self.rows = int(getattr(owner, "rows", 1))
        self.N = {p: int(shapes[p][0]) for p in ("w1", "w3", "w2")}
        self.K = {p: int(shapes[p][1]) * 2 for p in ("w1", "w3", "w2")}
        if not (self.K["w1"] == self.K["w3"] == self.N["w2"] and self.N["w1"] == self.N["w3"] == self.K["w2"]):
            raise ValueError(f"w1/w3/w2 shapes do not chain: N {self.N} K {self.K}")
        for p in ("w1", "w3", "w2"):
            if tuple(sshapes[p]) != (self.N[p], self.K[p] // 32):
                raise ValueError(f"{p}: scale shape {sshapes[p]} != (N, K/32)")
        self.dim, self.inter = self.K["w1"], self.N["w1"]
        if self.inter % UTCCP or self.dim % UTCCP or self.inter % GRAN:
            raise ValueError(f"dim {self.dim} / inter {self.inter}: multiples of {UTCCP} needed (UTCCP SF groups)")
        self.swiglu_limit = float(swiglu_limit)
        self.shared = shared or os.environ.get(SHARED_ENV, "gemv")
        self.sf_major = sf_major or os.environ.get(SF_MAJOR_ENV, "mn")
        self.l1_store = l1_store or os.environ.get(L1_STORE_ENV, "copy")
        for name, v, ok in (("shared", self.shared, SHARED_MODES), ("sf_major", self.sf_major, SF_MAJORS),
                            ("l1_store", self.l1_store, L1_STORES)):
            if v not in ok:
                raise ValueError(f"dg-mega {name} {v!r} not in {ok}")
        if self.l1_store == "inplace":
            raise NotImplementedError(f"dg-mega l1_store 'inplace' needs the pool re-laid out with w1|w3 interleaved "
                                      f"(expert_pool / patches / residency / staging changes): see {FINDINGS}")
        self.T = int(max_tokens or os.environ.get(MAX_TOKENS_ENV, 0) or max(MAX_TOKENS_DEFAULT, self.rows))
        if self.T < self.rows:
            raise ValueError(f"max_tokens {self.T} < rows {self.rows}")
        self.S = int(spare_slots)
        if not 1 <= self.S <= pool.spare:
            raise ValueError(f"spare_slots {self.S}: the pool has {pool.spare} spare slots (add_spare_slots first)")
        self.dg = dg if dg is not None else _import_dg()
        self.deterministic = _set_deterministic(self.dg, deterministic)
        G = [int(pool.hbm_full[l]["w2"].size(0)) for l in range(pool.n_layers)]
        self.plan = hbm_bytes(pool.n_layers, G, self.dim, self.inter, self.l1_store)
        self._check_hbm()
        self.layers: Dict[int, dict] = {}
        self.syms: Dict[int, object] = {}
        with torch.inference_mode(False), torch.no_grad():
            self.y = torch.zeros(self.T, self.dim, dtype=torch.bfloat16, device=self.dev)
        self.ovf = getattr(owner, "ovf_step", None) if self.rows > 1 else None
        self.n_calls = 0
        self.spare_added_B = 0
        self.y_rows = "T"            # every call is made with T token rows (module docstring, "rows")
        self.symm_desc: Optional[dict] = None

    def _check_hbm(self) -> None:
        need = self.plan["total_B"]
        if self.dev.type != "cuda":
            return
        free, _ = torch.cuda.mem_get_info(self.dev)
        head = float(os.environ.get(HEADROOM_ENV, HEADROOM_GIB)) * GiB
        if free - need < head:
            raise RuntimeError(f"dg-mega l1_store {self.l1_store!r} needs {need / GiB:.2f} GiB (L1 copy "
                               f"{self.plan['l1_copy_GiB']} + SF {self.plan['sf_copy_GiB']}), {free / GiB:.2f} GiB "
                               f"free, headroom {head / GiB:.1f}: the L1 interleave cannot be done in place on this "
                               f"pool layout ({FINDINGS})")

    # ---------------------------------------------------------------- per layer (build time)
    def layer_tables(self, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor], moe=None) -> dict:
        pool, d = self.pool, self.dev
        H = len(pool.hot[layer])
        w = {p: pool.hbm_full[layer][p] for p in ("w1", "w3", "w2")}
        G = int(w["w2"].size(0))
        for p in w:
            if tuple(w[p].shape) != (G, self.N[p], self.K[p] // 2) or not w[p].is_contiguous():
                raise ValueError(f"L{layer} {p}: pool {tuple(w[p].shape)} is not a contiguous [G, N, K/2]")
        if G < H + self.S:
            raise ValueError(f"L{layer}: {G} pool slots < H {H} + spare {self.S}")
        with torch.inference_mode(False), torch.no_grad():
            l1 = mega_l1_weight(w["w1"], w["w3"])                                       # the copy (see FINDINGS)
            c = {p: torch.zeros(G, self.N[p], self.K[p] // 32, dtype=torch.uint8, device=d) for p in w}
            if H:
                for p in w:
                    c[p][:H].copy_(_u8(pool.hbm[layer][p][1]))
            sf1 = mega_l1_sf(c["w1"], c["w3"], self.sf_major)
            sf2 = mega_l2_sf(c["w2"], self.sf_major)
            del c
            gt = torch.full((pool.n_experts,), -1, dtype=torch.int32, device="cpu")
            for slot, e in enumerate(pool.hot[layer]):
                gt[e] = slot
            from runtime.dg_inplace import ORDER6

            dsts = [l1, w["w2"], sf1, sf2]
            t = {"layer": layer, "H": H, "G": G, "l1": l1, "l2": w["w2"], "sf1": sf1, "sf2": sf2,
                 "gtab": gt.to(d), "ptab6": torch.stack([ptabs[k_] for k_ in ORDER6]).contiguous(), "dsts": dsts,
                 "dtab": torch.tensor([x.data_ptr() for x in dsts], dtype=torch.int64, device="cpu").to(d),
                 "stage": torch.full((self.S,), -1, dtype=torch.long, device=d),
                 "shared_l1": None, "shared_l2": None, "resolve": None}
            if self.shared == "fused":
                if moe is None:
                    raise ValueError("shared 'fused' needs the layer's MoE module (shared expert weights)")
                t["shared_l1"], t["shared_l2"] = self.shared_weights(moe.shared_experts)
            if d.type != "cuda" and moe is not None:   # the CPU build's staging resolves addresses through params
                t["resolve"] = _param_resolver(moe)
        t["sym"] = self.symm(G)
        self.layers[layer] = t
        return t

    def shared_weights(self, ex) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """The shared Expert's fp8 Linears -> MegaMoE's shared weights: L1 = w1|w3 interleaved fp8 [2I, D] + SF, L2 =
        w2 fp8 [D, I] + SF; the 32x32 block scales expanded per row (exact), packed, UTCCP, sf_major storage."""
        w1, w3, w2 = ex.w1, ex.w3, ex.w2
        for lin in (w1, w3, w2):
            if lin.weight.dtype != torch.float8_e4m3fn:
                raise ValueError(f"shared 'fused' needs fp8 shared experts, got {lin.weight.dtype}")
        I, D = w1.weight.shape
        l1 = mega_l1_weight(w1.weight, w3.weight)
        sf1 = mega_l1_sf(block_codes_per_row(w1.scale, I), block_codes_per_row(w3.scale, I), self.sf_major)
        sf2 = mega_l2_sf(block_codes_per_row(w2.scale, D), self.sf_major)
        return (l1, sf1), (w2.weight.contiguous(), sf2)

    def symm(self, G: int):
        if G not in self.syms:
            n_sh = 1 if self.shared == "fused" else 0
            sym = make_symm_buffer(self.dg, G, self.T, self.k, self.dim, self.inter, n_sh)
            with torch.no_grad():
                sym.topk_idx.fill_(-1)
                sym.topk_weights.zero_()
                sym.x.view(torch.uint8).zero_()
            self.syms[G] = sym
            if self.symm_desc is None:
                self.symm_desc = describe_symm(sym)
        return self.syms[G]

    # ---------------------------------------------------------------- per step (graph-capturable)
    def route(self, t: dict, ids: torch.Tensor, w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ids int64 [R, k] ascending per row, w fp32 [R, k] -> the SymmBuffer's topk_idx / topk_weights rows [:R],
        this layer's staging list, the overflow count (R-row builds)."""
        R_ = ids.size(0)
        sym = t["sym"]
        if (self.dev.type == "cuda" and HAVE_TRITON and sym.topk_idx.is_contiguous()
                and sym.topk_weights.is_contiguous() and sym.topk_weights.dtype == torch.float32):
            route_mega(ids.contiguous(), w.float().contiguous(), t["gtab"], sym.topk_idx, sym.topk_weights,
                       t["stage"], t["H"], self.S, self.pool.n_experts, self.ovf)      # 1 launch
            return {"topk": sym.topk_idx[:R_], "stage": t["stage"]}
        m = mega_route(ids, t["gtab"], t["H"], self.S, self.pool.n_experts)
        sym.topk_idx[:R_].copy_(m["topk"])
        sym.topk_weights[:R_].copy_(w.float())
        t["stage"].copy_(m["stage"])
        if self.ovf is not None:
            self.ovf.add_(m["n_over"].to(self.ovf.dtype))
        return m

    def stage(self, t: dict) -> None:
        if t["resolve"] is not None:
            stage_mega_emulated(t["resolve"], t["stage"], t["ptab6"], t["dsts"], t["H"], self.sf_major)
        else:
            stage_mega(t["stage"], t["ptab6"], t["dtab"], t["dsts"], t["H"], self.sf_major)

    def act(self, t: dict, q: torch.Tensor, s: torch.Tensor) -> None:
        R_ = q.size(0)
        sym = t["sym"]
        sym.x[:R_].copy_(q)
        write_act_sf(sym.x_sf, act_codes(s), R_)

    def run(self, t: dict) -> torch.Tensor:
        """One MegaMoE launch with T token rows -> y [T, D] bf16 (rows >= R are zero)."""
        kw = {}
        if t["shared_l1"] is not None:
            kw = {"shared_l1_weights": t["shared_l1"], "shared_l2_weights": t["shared_l2"]}
        self.dg.fp8_fp4_mega_moe(self.y, (t["l1"], t["sf1"]), (t["l2"], t["sf2"]), t["sym"], recipe=MEGA_RECIPE,
                                 activation="swiglu",
                                 activation_clamp=self.swiglu_limit if self.swiglu_limit > 0 else None,
                                 fast_math=False, **kw)
        self.n_calls += 1
        return self.y

    def reset_rows(self) -> None:
        for sym in self.syms.values():
            sym.topk_idx.fill_(-1)
            sym.topk_weights.zero_()
            sym.x.view(torch.uint8).zero_()

    def warmup(self) -> None:
        """First-call JIT for every distinct G before any capture: one call with token 0 routed to the first k hot
        groups, weight 0 (output discarded); the routing rows are reset after."""
        seen = set()
        for l in sorted(self.layers):
            t = self.layers[l]
            if t["G"] in seen:
                continue
            seen.add(t["G"])
            sym = t["sym"]
            n = min(self.k, t["G"])
            sym.topk_idx[0, :n].copy_(torch.arange(n, dtype=torch.long, device=self.dev))
            self.run(t)
        self.reset_rows()

    def describe(self) -> dict:
        free = None
        if self.dev.type == "cuda":
            f, tot = torch.cuda.mem_get_info(self.dev)
            free = {"free_GiB": round(f / GiB, 3), "total_GiB": round(tot / GiB, 3)}
        return {"impl": "dg-mega", "deep_gemm": getattr(self.dg, "__file__", None),
                "deep_gemm_version": getattr(self.dg, "__version__", None), "recipe": list(MEGA_RECIPE),
                "shared": self.shared, "sf_major": self.sf_major, "l1_store": self.l1_store, "T": self.T,
                "y_rows": self.y_rows, "k": self.k, "dim": self.dim, "inter": self.inter, "spare": self.S,
                "fast_math": False, "activation_clamp": self.swiglu_limit if self.swiglu_limit > 0 else None,
                "deterministic": self.deterministic, "groups_per_layer": sorted({t["G"] for t in self.layers.values()}),
                "symm": self.symm_desc, "plan": self.plan, "hbm_after_build": free, "n_calls": self.n_calls,
                "spare_added_B": self.spare_added_B, "findings": FINDINGS}


def _import_dg():
    from runtime.dg_moe import import_deep_gemm

    return import_deep_gemm()


def _set_deterministic(dg, on: bool) -> dict:
    f = getattr(dg, "use_deterministic_algorithms", None)
    if not on or f is None:
        return {"requested": bool(on), "applied": False}
    try:
        f(True)
    except TypeError:
        f()
    return {"requested": True, "applied": True}


def _param_resolver(moe):
    """address -> uint8 view of the expert param at that address (the CPU build's staging: every ptab entry is an
    expert's weight / scale param, _build_moe asserts it)."""
    reg = {}
    for ex in moe.experts:
        if ex is None:
            continue
        for p in ("w1", "w3", "w2"):
            lin = getattr(ex, p)
            for t in (lin.weight, lin.scale):
                reg[t.data_ptr()] = t.detach().reshape(-1).view(torch.uint8)
    return reg.__getitem__


# ------------------------------------------------------------------------------------------------ the MoE
class DGMegaMoE:
    """One layer's MoE through MegaMoE. StaticMoE's interface (called by the patched MoE.forward while the owner
    runs a step); works for any R <= T rows (the 1-row and the R-row StaticStep use the same class)."""

    def __init__(self, owner, moe, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]):
        self.owner, self.moe, self.layer, self.ptabs = owner, moe, layer, ptabs
        self.t = owner.dg.layer_tables(layer, ptabs, moe)
        self.last: Dict[str, torch.Tensor] = {}

    @property
    def active(self) -> bool:
        return self.owner.active

    def _gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (ids int64 [R, k] ascending per row, weights fp32 [R, k])."""
        o, R_ = self.owner, x.size(0)
        g = o.glue
        if g is not None and g.on("router") and R_ == o.rows:
            if R_ == 1:
                ids, w = g.route(self.moe, x)
                return ids.view(1, -1).long(), w.view(1, -1)
            ids, w = g.route_rows(self.moe, x)
            return ids.long(), w
        weights, indices = self.moe.gate(x, None)                                     # model.py:894
        ids, order = torch.sort(indices, dim=-1)                                      # ascending id = model.py:897
        return ids.long(), torch.gather(weights, -1, order)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        o, moe, b, R = self.owner, self.moe, self.owner.dg, self.owner.R
        shape = x.size()
        x = x.view(-1, moe.dim)                                                       # model.py:893
        R_ = x.size(0)
        if R_ > b.T:
            raise ValueError(f"dg-mega: {R_} rows > max_tokens {b.T}")
        ids, w = self._gate(x)
        self.last = b.route(self.t, ids, w)
        if o.counters and R_ == o.rows:
            if R_ == 1:
                o.ids_step[self.layer].copy_(ids.view(-1))
            else:
                s, first = unique_ids_record(ids, o.ids_step.size(1))
                o.ids_step[self.layer].copy_(s)
                o.uvalid_step[self.layer].copy_(first)
        b.stage(self.t)                                                               # 1 launch: cold -> H + c
        q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)           # model.py:187
        b.act(self.t, q, s)
        y = b.run(self.t)[:R_]                                                        # 1 launch: the whole MoE
        if b.shared == "gemv":
            y = (y.float() + self._shared(x).float()).type_as(x)                      # model.py:905-906 (A3: 2nd rounding)
        else:
            y = y.type_as(x).clone()
        return y.view(shape)

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        glue = self.owner.glue
        if glue is not None and glue.on("swiglu_shared"):
            return glue.shared_expert(self.moe.shared_experts, x)
        return self.moe.shared_experts(x)
