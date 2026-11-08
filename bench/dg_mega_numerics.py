#!/usr/bin/env python3
"""dg_mega_numerics.py -- M3-N1: can MegaMoE's numerics be made ours? (docs/HANDOFF-M3-N1-MEGAMOE-NUMERICS.md;
results and method: docs/M2-L3-NUMERICS.md). A bounded probe: no pool re-layout, runtime/expert_pool.py untouched.

The arms (one routed MoE call; identical routing, activations, weights, shared-expert output in every arm):
  ours        today's dg-inplace chain = the reference's rounding points: DeepGEMM L1 -> bf16 gate/up -> SwiGLU in
              fp32 (torch's formula, model.py:845-852) -> bf16 -> act_quant per 32 -> DeepGEMM L2 -> bf16 per expert
              -> fp32 sum in ascending-id order + the shared expert -> ONE bf16 rounding (model.py:895-906).
  tm          the torch reproduction of MegaMoE: the same L1 GEMM -> SwiGLU in fp32 (the kernel's formula, cuh
              1058-1089, fast_math=False) -> fp8 quant FROM FP32 (amax of the fp32 values, get_ue8m0_sf_exp) -> the
              same L2 GEMM -> routed fp32 sum -> bf16 (the kernel's output) -> + shared -> bf16 (gemv mode: 2nd
              rounding).
  tm_fix      tm with the SwiGLU output rounded to bf16 before the quant (the "bf16-before-quant" variant = what a
              one-line kernel patch would compute), gemv-mode combine.
  tm_fix_1r   tm_fix with one final rounding (the shared expert inside the fp32 sum, as `ours`).
  tm_1r       tm with one final rounding (quant from fp32 alone, without gemv mode's 2nd rounding).
  ours_2r     ours with gemv mode's 2nd rounding (the 2nd rounding alone).
  mega        the real fp8_fp4_mega_moe (routed only) + the shared expert in gemv mode. With --dg-path pointing at
              `patch-dg`'s copy it is the PATCHED kernel (bf16 before quant) -- run in a fresh process.
Oracles (fp64 dots, exact dequantized operands): oracle_ref = the reference's rounding points (bf16 gate/up, bf16
SwiGLU output, act_quant, bf16 expert outputs; the spike's oracle_expert); oracle_mega = MegaMoE's rounding points
(the same but quant from the unrounded SwiGLU output). Every stage of every arm is compared with oracle_ref's
same stage: mismatch fraction, max ulp, relative L2 (pooled: sqrt(sum |d|^2 / sum |ref|^2) over all windows).

Subcommands (receipts under scratch/ by default):
  model       CPU (or GPU), NO checkpoint: the decomposition on SYNTHETIC MXFP4 experts (Gaussian weights quantized
              to e2m1 + per-32 UE8M0, the real shapes) and synthetic activations; the GEMMs are fp32 emulations and
              there is no `mega` arm. A MODEL of the box question, labelled synthetic in its receipt.
  downstream  box, the full checkpoint in the production 1-row config (first_light's flags; --mode static):
              (i) teacher-forced along Pg61's own tokens (parity-A token bytes -> ids, bench/gap_ledger.py), with
              the MoE inputs x (and the reference gate's ids / weights) captured at --capture-layers -> --acts (the
              `layers` input); (ii) free-running greedy + bench/parity.py (first_light.run_rep / run_parity: the
              A3 instrument); (iii) both again with MegaMoE substituted on groups of layers (runtime/dg_mega.py
              MegaSubstitution; HBM checked first, what fit is recorded) -> margin-weighted top-1 flips vs Pg61 and
              vs our own unsubstituted stream, |d logprob| of the forced token.
  layers      box, checkpoint only (no model boot): layers 0/10/20/39, the captured fixture activations in 6-row
              windows (R = 6) and each row alone (R = 1): every arm's stages vs the oracles, the real kernel's L2
              input bytes (SymmBuffer l2_acts, matched to tm / ours), row invariance, graph-replayed us per layer.
  patch-dg    box (in the container): copy the installed deep_gemm to --out with the one-line bf16-before-quant
              patch in impls/sm100_fp8_fp4_mega_moe.cuh; then
                layers --dg-path <out> --jit-cache <fresh dir> --out scratch/m3n1-layers-bf16q.json
              runs the real kernel with the reference's rounding point (a fresh process and a fresh JIT cache, so no
              unpatched kernel is reused; the l2-acts match says whether the patch took).
  report      receipts -> the tables of docs/M2-L3-NUMERICS.md (markdown on stdout).

Box order: downstream (writes the activations) -> layers -> patch-dg -> layers --dg-path -> downstream --dg-path
(mega_bf16q arms; the base arm is rerun for the flips-vs-ours columns, activations are not re-captured) -> report. deep_gemm is
imported from vllm/third_party with vllm never imported (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
FP8_MAX = 448.0
DIM, INTER, TOPK, N_EXPERTS = 5120, 2304, 6, 384
LIM, ROUTE_SCALE = 10.0, 1.5                       # config.json swiglu_limit / route_scale
LAYERS_DEFAULT = (0, 10, 20, 39)
ROWS = 6
SPIKE = {"mega_gemv_rel_l2": 0.013357026880999955, "ours_rel_l2": 0.001668479773879048,
         "mega_routed_rel_l2": 0.022494974367862772, "ours_routed_rel_l2": 0.0017287302562631309,
         "src": "docs/dg-mega-spike-20261107.json (layer 0, synthetic activations)"}
ARMS = ("ours", "ours_2r", "tm", "tm_1r", "tm_fix", "tm_fix_1r", "mega")
CUH = os.path.join("include", "deep_gemm", "impls", "sm100_fp8_fp4_mega_moe.cuh")
PATCH_ANCHOR = ("                            activation_values[i][k] = __fmul2_rn(\n"
                "                                __fmul2_rn(gated, up), weights);\n")
PATCH_LINES = ("                            // M3-N1 (bench/dg_mega_numerics.py patch-dg): round the SwiGLU output to bf16"
               " before the fp8\n"
               "                            // requant -- the reference's rounding point (model.py:853 .to(dtype), then"
               " act_quant)\n"
               "                            activation_values[i][k] = __bfloat1622float2(__float22bfloat162_rn("
               "activation_values[i][k]));\n")
PATCH_TAG = "M3-N1 (bench/dg_mega_numerics.py patch-dg)"


def cpu_gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


# ------------------------------------------------------------------------------------------------ quantizers (exact)
def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF


def scale_ref(amax: torch.Tensor) -> torch.Tensor:
    """The reference act_quant's UE8M0 scale (kernel.py:76-78 + fast_round_scale, round_scale=True): fp32
    s = 2^ceil(log2(max(amax, 1e-4) * (1/448))), the product rounded in fp32, the ceil from its bits."""
    p = torch.clamp(amax.float(), min=1e-4) * (1.0 / FP8_MAX)
    b = _bits(p)
    e = ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0).long()
    return torch.exp2(e.double()).float()


def scale_mega(amax: torch.Tensor) -> torch.Tensor:
    """MegaMoE's UE8M0 scale for an fp32 amax (math.cuh:115-135 get_ue8m0_sf_exp<e4m3, float>): the carry of
    bits + mantissa_mask - mantissa(1.75) is the exponent ceiling; exponent floor 105 (= amax floor 1e-4)."""
    b = _bits(amax)
    rounded = (b + 0x7FFFFF - 0x600000) >> 23
    e = torch.clamp(rounded, min=105 + 8) - 8
    return torch.exp2((e - 127).double()).float()


def quant32(h: torch.Tensor, scale_fn: Callable = scale_ref) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-32 blocks of the last dim: fp32 amax, power-of-two scale, e4m3 round-to-nearest-even of h / s (exact: s is a
    power of two; the kernel multiplies by 1/s, also exact) -> (q fp8 [..., K], s fp32 [..., K/32])."""
    z = h.float().unflatten(-1, (-1, 32))
    s = scale_fn(z.abs().amax(-1))
    q = (z / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8).flatten(-2)
    return q, s


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (q.double().unflatten(-1, (-1, 32)) * s.double().unsqueeze(-1)).flatten(-2)


def fp8_rows(q: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """q fp8 [M, K] rows idx (through uint8: index_select / stack are not implemented for float8 everywhere)."""
    return q.view(torch.uint8)[idx].view(FP8)


def fp8_stack(qs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([q.view(torch.uint8) for q in qs]).view(FP8)


# ------------------------------------------------------------------------------------------------ SwiGLU (fp32)
def swiglu_ref(g: torch.Tensor, u: torch.Tensor, w: torch.Tensor, lim: float = LIM) -> torch.Tensor:
    """model.py:845-852: gate/up = bf16 GEMM outputs .float(), clamp in fp32, F.silu(gate) * up, weights * x."""
    gate, up = g.float(), u.float()
    if lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    return w.float() * (F.silu(gate) * up)


def swiglu_mega(g: torch.Tensor, u: torch.Tensor, w: torch.Tensor, lim: float = LIM) -> torch.Tensor:
    """sm100_fp8_fp4_mega_moe.cuh:1058-1089 with fast_math=False: bf16 clamp (__hmin2 / __hmax2), then fp32
    gate / (1 + expf(-gate)) (IEEE divide), * up, * the top-k weight."""
    if lim > 0:
        g = torch.clamp(g, max=lim)
        u = torch.clamp(torch.clamp(u, min=-lim), max=lim)
    gate = g.float()
    gated = gate / (1.0 + torch.exp(-gate))
    return (gated * u.float()) * w.float()


# ------------------------------------------------------------------------------------------------ fp4 / fp8 dequant
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def deq_fp4(w_i8: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """int8 packed e2m1 [N, K/2] (element 2i = low nibble) + e8m0 codes [N, K/32] -> fp64 [N, K] (exact)."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device="cpu").to(w_i8.device)
    u = w_i8.view(torch.uint8).long()
    v = torch.stack([lut[u & 0xF], lut[u >> 4]], -1).flatten(-2)
    return v * torch.exp2(codes.view(torch.uint8).double() - 127).repeat_interleave(32, -1)


def deq_fp8_rows(w: torch.Tensor, codes_rows: torch.Tensor) -> torch.Tensor:
    return w.double() * torch.exp2(codes_rows.view(torch.uint8).double() - 127).repeat_interleave(32, -1)


def quant_fp4(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic MXFP4: fp32 [N, K] -> (int8 packed [N, K/2], uint8 e8m0 codes [N, K/32]); per-32 scale
    2^ceil(log2(amax / 6)), nearest e2m1 magnitude (the `model` subcommand's weights)."""
    N, K = W.shape
    blk = W.float().view(N, K // 32, 32)
    amax = blk.abs().amax(-1).clamp_min(2.0 ** -100)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    v = blk / torch.exp2(e).unsqueeze(-1)
    mags = torch.tensor(_E2M1[:8], dtype=torch.float32, device="cpu").to(W.device)
    idx = (v.abs().unsqueeze(-1) - mags).abs().argmin(-1)
    nib = (idx + 8 * (v < 0).long()).view(N, K)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed.contiguous(), (e + 127).to(torch.uint8).contiguous()


# ------------------------------------------------------------------------------------------------ oracle (fp64)
def swiglu64(g64: torch.Tensor, u64: torch.Tensor, w: float, lim: float = LIM) -> torch.Tensor:
    """The exact SwiGLU (no bf16 rounding anywhere), fp64."""
    gate, up = g64.double(), u64.double()
    if lim > 0:
        gate = torch.clamp(gate, max=lim)
        up = torch.clamp(up, min=-lim, max=lim)
    return gate * torch.sigmoid(gate) * up * float(w)


def oracle_stages(a64: torch.Tensor, W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, w: float,
                  lim: float = LIM, h_round: str = "bf16") -> Dict[str, torch.Tensor]:
    """One (token, expert) with fp64 dots on exact dequantized operands. Rounding points: bf16 gate/up (model.py:
    845-846), clamp, SwiGLU in fp64, then h_round "bf16" = the reference (bf16, act_quant: scale_ref) or "none" =
    MegaMoE's (quant from the unrounded value: scale_mega), W2, bf16 (model.py:853)."""
    g = (W1 @ a64).to(BF16)
    u = (W3 @ a64).to(BF16)
    gate, up = g.double(), u.double()
    if lim > 0:
        gate = torch.clamp(gate, max=lim)
        up = torch.clamp(up, min=-lim, max=lim)
    h = gate * torch.sigmoid(gate) * up * float(w)
    if h_round == "bf16":
        q, s = quant32(h.to(BF16).float(), scale_ref)
    elif h_round == "none":
        q, s = quant32(h.float(), scale_mega)
    else:
        raise ValueError(h_round)
    o = (W2 @ dequant(q, s)).to(BF16)
    return {"g": g, "u": u, "h": h, "q": q, "s": s, "o": o}


# ------------------------------------------------------------------------------------------------ pooled statistics
def _ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from runtime.dg_inplace import bf16_ulp

    return bf16_ulp(a.contiguous().to(BF16), b.contiguous().to(BF16))


def st_bf16(got: torch.Tensor, want64: torch.Tensor) -> dict:
    """got bf16 vs a reference (fp64 or bf16): mismatch vs the reference rounded to bf16, max ulp, |d|^2 sums."""
    want64 = want64.double()
    u = _ulp(got, want64.to(BF16)).reshape(-1)
    d = got.double() - want64
    return {"n": int(u.numel()), "n_mis": int((u != 0).sum()), "max_ulp": int(u.max()) if u.numel() else 0,
            "d2": float((d * d).sum()), "r2": float((want64 * want64).sum()),
            "max_abs": float(d.abs().max()) if d.numel() else 0.0}


def st_real(got: torch.Tensor, want64: torch.Tensor) -> dict:
    """fp32 / fp64 values vs a reference: relative-L2 sums only (no ulp)."""
    d = got.double() - want64.double()
    return {"n": int(d.numel()), "d2": float((d * d).sum()), "r2": float((want64.double() ** 2).sum()),
            "max_abs": float(d.abs().max()) if d.numel() else 0.0}


def st_fp8(q: torch.Tensor, s: torch.Tensor, q_ref: torch.Tensor, s_ref: torch.Tensor) -> dict:
    """fp8 L2 inputs: byte mismatch (values AND scales: a scale difference shows as a dequantized difference) and
    the dequantized relative-L2 sums."""
    a, b = dequant(q, s), dequant(q_ref, s_ref)
    d = a - b
    return {"n": int(q.numel()), "n_mis": int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum()),
            "n_scale_mis": int((s != s_ref).sum()), "n_scale": int(s.numel()),
            "d2": float((d * d).sum()), "r2": float((b * b).sum())}


def merge(a: Optional[dict], b: dict) -> dict:
    if a is None:
        return json.loads(json.dumps(b))
    if "d2" in b:
        out = dict(a)
        for k_, v in b.items():
            if k_ in ("max_ulp", "max_abs"):
                out[k_] = max(a.get(k_, 0), v)
            else:
                out[k_] = a.get(k_, 0) + v
        return out
    out = dict(a)
    for k_, v in b.items():
        out[k_] = merge(a.get(k_), v) if isinstance(v, dict) else v
    return out


def finalize(st: dict) -> dict:
    if "d2" in st:
        r = {"n": st["n"], "rel_l2": math.sqrt(st["d2"] / st["r2"]) if st.get("r2") else None}
        if "n_mis" in st:
            r["mismatch_frac"] = st["n_mis"] / st["n"] if st["n"] else None
        for k_ in ("max_ulp", "max_abs"):
            if k_ in st:
                r[k_] = st[k_]
        if "n_scale_mis" in st:
            r["scale_mismatch_frac"] = st["n_scale_mis"] / st["n_scale"] if st["n_scale"] else None
        return r
    return {k_: finalize(v) if isinstance(v, dict) else v for k_, v in st.items()}


# ------------------------------------------------------------------------------------------------ one window
def analyze_window(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, deq: Callable[[int], Tuple],
                   shared_bf16: torch.Tensor, gemm: Callable, mega: Optional[Callable] = None,
                   lim: float = LIM, want_raw: bool = False) -> dict:
    """Every arm and both oracles for one routed MoE call of R rows.

    x bf16 [R, D]; ids int64 [R, k] ascending per row; w fp32 [R, k]; deq(e) -> (W1, W3, W2) fp64 dequantized
    [N, K] on x's device; shared_bf16 [R, D] the shared expert's bf16 output (added in every arm and the oracle);
    gemm(proj, pairs, q fp8 [P, K], s fp32 [P, K/32]) -> bf16 [P, N] (pairs = [(r, j, e)]: the routed pairs in row
    / column order); mega() -> (routed bf16 [R, D], l2_acts snapshot uint8 [M, I] of the rows written, or None).
    -> {"stats": pooled-stat tree, "l2_match": ..., "swiglu_formula": ...} (+ "raw" tensors if want_raw)."""
    dev = x.device
    R, k = ids.shape
    D = x.size(-1)
    pairs = [(r, j, int(ids[r, j])) for r in range(R) for j in range(k)]
    P = len(pairs)
    rr = torch.tensor([p[0] for p in pairs], dtype=torch.long, device="cpu").to(dev)
    jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device="cpu").to(dev)
    wp = w.float()[rr, jj].view(P, 1)
    q1, s1 = quant32(x.float(), scale_ref)                                       # act_quant(x) (model.py:187)
    q1p, s1p = fp8_rows(q1, rr), s1[rr]
    g = gemm("w1", pairs, q1p, s1p)
    u = gemm("w3", pairs, q1p, s1p)
    h_ref = swiglu_ref(g, u, wp, lim)
    h_meg = swiglu_mega(g, u, wp, lim)
    q2 = {"ours": quant32(h_ref.to(BF16).float(), scale_ref),
          "tm": quant32(h_meg, scale_mega),
          "tm_fix": quant32(h_meg.to(BF16).float(), scale_mega)}
    o = {a_: gemm("w2", pairs, *q2[a_]) for a_ in q2}

    def comb(oo: torch.Tensor) -> torch.Tensor:                                  # fp32, ascending id per row
        acc = torch.zeros(R, D, dtype=torch.float32, device=dev)
        for i, (r, _, _) in enumerate(pairs):
            acc[r] += oo[i].float()
        return acc

    c = {a_: comb(o[a_]) for a_ in o}
    sh = shared_bf16.float()
    two = lambda cc: (cc.to(BF16).float() + sh).to(BF16)                          # noqa: E731 gemv mode
    one = lambda cc: (cc + sh).to(BF16)                                           # noqa: E731 one rounding
    y = {"ours": (c["ours"].to(BF16), one(c["ours"])), "ours_2r": (c["ours"].to(BF16), two(c["ours"])),
         "tm": (c["tm"].to(BF16), two(c["tm"])), "tm_1r": (c["tm"].to(BF16), one(c["tm"])),
         "tm_fix": (c["tm_fix"].to(BF16), two(c["tm_fix"])), "tm_fix_1r": (c["tm_fix"].to(BF16), one(c["tm_fix"]))}
    l2snap = None
    if mega is not None:
        ym, l2snap = mega()
        y["mega"] = (ym, (ym.float() + sh).to(BF16))

    # oracles, per unique expert (fp64 weights once each); "exact" = no intermediate rounding at all (fp64 gate/up,
    # SwiGLU and L2 on the unquantized SwiGLU output) given the quantized weights and the act_quant'ed x
    O = {"ref": [None] * P, "mega": [None] * P}
    o_exact: List[Optional[torch.Tensor]] = [None] * P
    a64 = dequant(q1, s1)
    for e in sorted({p[2] for p in pairs}):
        W1, W3, W2 = deq(e)
        for i, (r, j, ee) in enumerate(pairs):
            if ee == e:
                for kind, hr in (("ref", "bf16"), ("mega", "none")):
                    O[kind][i] = oracle_stages(a64[r], W1, W3, W2, float(w[r, j]), lim, hr)
                o_exact[i] = W2 @ swiglu64(W1 @ a64[r], W3 @ a64[r], float(w[r, j]), lim)
        del W1, W3, W2
    Os = {kind: {f: (fp8_stack if f == "q" else torch.stack)([d[f] for d in O[kind]])
                 for f in ("g", "u", "h", "q", "s", "o")} for kind in O}
    routed64 = {kind: torch.zeros(R, D, dtype=torch.float64, device=dev) for kind in O}
    for kind in O:
        for i, (r, _, _) in enumerate(pairs):
            routed64[kind][r] += Os[kind]["o"][i].double()
    final64 = routed64["ref"] + sh.double()
    final64_m = routed64["mega"] + sh.double()
    routed_ex = torch.zeros(R, D, dtype=torch.float64, device=dev)
    for i, (r, _, _) in enumerate(pairs):
        routed_ex[r] += o_exact[i]
    final_ex = routed_ex + sh.double()
    ref = Os["ref"]
    stats = {
        "l1_gate_up": {"ours=tm": st_bf16(torch.cat([g, u], -1), torch.cat([ref["g"], ref["u"]], -1).double())},
        "swiglu_out": {"ours": st_bf16(h_ref.to(BF16), ref["h"].to(BF16).double()),
                       "tm": st_real(h_meg, ref["h"].to(BF16).double()),
                       "oracle_ref_vs_exact": st_real(ref["h"].to(BF16).double(), ref["h"])},
        "l2_input": {a_: st_fp8(*q2[a_], ref["q"], ref["s"]) for a_ in q2},
        "l2_input_pairwise": {"tm_vs_ours": st_fp8(*q2["tm"], *q2["ours"]),
                              "tm_fix_vs_ours": st_fp8(*q2["tm_fix"], *q2["ours"]),
                              "oracle_mega_vs_oracle_ref": st_fp8(Os["mega"]["q"], Os["mega"]["s"], ref["q"], ref["s"])},
        "l2_out": {a_: st_bf16(o[a_], ref["o"].double()) for a_ in o},
        "routed": {a_: st_bf16(y[a_][0], routed64["ref"]) for a_ in y},
        "routed_vs_oracle_mega": {a_: st_bf16(y[a_][0], routed64["mega"]) for a_ in y},
        "final": {a_: st_bf16(y[a_][1], final64) for a_ in y},
        "final_vs_oracle_mega": {a_: st_bf16(y[a_][1], final64_m) for a_ in y},
        "routed_vs_exact": {a_: st_bf16(y[a_][0], routed_ex) for a_ in y},
        "final_vs_exact": {a_: st_bf16(y[a_][1], final_ex) for a_ in y},
        "oracle_mega_vs_oracle_ref": {"routed": st_real(routed64["mega"], routed64["ref"])},
        "oracles_vs_exact": {"oracle_ref": st_real(routed64["ref"], routed_ex),
                             "oracle_mega": st_real(routed64["mega"], routed_ex)},
        "pairwise_routed": {"tm_vs_ours": st_bf16(y["tm"][0], y["ours"][0].double()),
                            "tm_fix_vs_ours": st_bf16(y["tm_fix"][0], y["ours"][0].double())},
    }
    if "mega" in y:
        stats["pairwise_routed"]["mega_vs_ours"] = st_bf16(y["mega"][0], y["ours"][0].double())
        stats["pairwise_routed"]["mega_vs_tm"] = st_bf16(y["mega"][0], y["tm"][0].double())
        stats["pairwise_routed"]["mega_vs_tm_fix"] = st_bf16(y["mega"][0], y["tm_fix"][0].double())
    hb = (h_ref.contiguous().view(torch.int32) == h_meg.contiguous().view(torch.int32))
    out = {"stats": stats, "swiglu_formula": {"n": int(hb.numel()), "n_bitwise_equal": int(hb.sum())},
           "pairs": P, "unique_experts": len({p[2] for p in pairs})}
    if l2snap is not None:
        out["l2_match"] = match_l2_rows(l2snap, {a_: q2[a_][0] for a_ in q2})
    if want_raw:
        out["raw"] = {"y": {a_: (v[0].clone(), v[1].clone()) for a_, v in y.items()}, "q2": q2, "o": o, "g": g,
                      "u": u, "oracle_ref": ref, "routed64": routed64, "final64": final64}
    return out


def match_l2_rows(snap: torch.Tensor, cands: Dict[str, torch.Tensor], min_frac: float = 0.5) -> dict:
    """The real kernel's L2 inputs (SymmBuffer l2_acts rows it wrote this call, uint8 [M, I], any row order) vs our
    fp8 candidates per pair (name -> fp8 [P, I]): each row is assigned to the pair whose candidate bytes it matches
    best (over every name); then per name the fraction of equal bytes on the assigned rows. Layout-free."""
    snap = snap.to("cpu").view(torch.uint8)
    cu = {n: c.detach().to("cpu").contiguous().view(torch.uint8) for n, c in cands.items()}
    rows = snap[(snap != 0).any(1)] if snap.numel() else snap
    res = {"rows_written": int(rows.size(0)), "pairs": int(next(iter(cu.values())).size(0))}
    if rows.size(0) == 0:
        res["note"] = "no non-zero l2_acts row after the call: the view is not where the kernel keeps its L2 input"
        return res
    eq = {n: (rows.unsqueeze(1) == c.unsqueeze(0)).float().mean(-1) for n, c in cu.items()}     # [M, P]
    best = torch.stack(list(eq.values())).amax(0)                                               # [M, P]
    score, pair = best.max(1)
    ok = score >= min_frac
    res["rows_matched"] = int(ok.sum())
    res["pairs_covered"] = int(pair[ok].unique().numel())
    for n, e in eq.items():
        v = e[torch.arange(e.size(0), device=e.device), pair][ok]
        res[f"frac_bytes_eq_{n}"] = float(v.mean()) if v.numel() else None
        res[f"rows_exact_{n}"] = int((v == 1.0).sum())
    return res


# ------------------------------------------------------------------------------------------------ the kernel patch
def patch_bf16_before_quant(src: str) -> str:
    """The one-line bf16-before-quant patch of sm100_fp8_fp4_mega_moe.cuh: after the SwiGLU * weight product
    (cuh:1088-1089), round to bf16 and back, so amax and the e4m3 cast see the reference's bf16 values."""
    if PATCH_TAG in src:
        raise ValueError("already patched")
    n = src.count(PATCH_ANCHOR)
    if n != 1:
        raise ValueError(f"patch anchor found {n} times (expected exactly 1): the kernel source changed")
    return src.replace(PATCH_ANCHOR, PATCH_ANCHOR + PATCH_LINES)


def cmd_patch_dg(a) -> int:
    from runtime.dg_moe import DG_THIRD_PARTY

    src = a.src or os.path.join(os.environ.get("DSV41_DEEP_GEMM_PATH", DG_THIRD_PARTY), "deep_gemm")
    dst_root = os.path.abspath(a.out)
    dst = os.path.join(dst_root, "deep_gemm")
    if not os.path.isfile(os.path.join(src, CUH)):
        raise SystemExit(f"no {CUH} under {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst_root, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)
    f = os.path.join(dst, CUH)
    old = open(f).read()
    new = patch_bf16_before_quant(old)
    with open(f, "w") as fh:
        fh.write(new)
    man = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "src": src, "dst": dst, "file": CUH,
           "sha256_before": hashlib.sha256(old.encode()).hexdigest(),
           "sha256_after": hashlib.sha256(new.encode()).hexdigest(), "patch": PATCH_LINES,
           "has_C": sorted(n for n in os.listdir(dst) if n.startswith("_C"))}
    with open(os.path.join(dst_root, "patch-manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps(man, indent=1))
    print(f"next: python3 bench/dg_mega_numerics.py layers --dg-path {dst_root} --jit-cache {dst_root}-jit "
          f"--out scratch/m3n1-layers-bf16q.json")
    return 0


def import_dg(dg_path: Optional[str], jit_cache: Optional[str]) -> Tuple[object, dict]:
    """deep_gemm, optionally the patched copy at dg_path (then a fresh JIT cache is required). If the copied _C does
    not load from there, fall back to the installed package re-initialised on the copy's include root."""
    from runtime.dg_moe import import_deep_gemm

    info = {"dg_path": dg_path, "jit_cache": jit_cache, "method": "installed"}
    if jit_cache:
        os.makedirs(jit_cache, exist_ok=True)
        os.environ["DG_JIT_CACHE_DIR"] = os.path.abspath(jit_cache)
    if not dg_path:
        return import_deep_gemm(), info
    if not jit_cache:
        raise SystemExit("--dg-path needs --jit-cache <fresh dir>: a cached unpatched kernel would be reused")
    root = os.path.join(os.path.abspath(dg_path), "deep_gemm")
    with open(os.path.join(root, CUH)) as fh:
        info["patched_source"] = PATCH_TAG in fh.read()
    try:
        dg = import_deep_gemm(os.path.abspath(dg_path))
        info["method"] = "copy"
    except Exception as e:                                    # pragma: no cover - box only
        info["copy_import_error"] = repr(e)[:400]
        for m in [m for m in sys.modules if m == "deep_gemm" or m.startswith("deep_gemm.")]:
            del sys.modules[m]
        dg = import_deep_gemm()
        try:
            dg._C.init(root)
        except Exception as e2:
            raise SystemExit(f"patched deep_gemm: copy import failed ({info['copy_import_error']}) and re-init on "
                             f"{root} failed ({e2!r})")
        info["method"] = "re-init"
    info["file"] = getattr(dg, "__file__", None)
    if info["method"] == "copy" and not str(info["file"]).startswith(root):
        raise SystemExit(f"deep_gemm imported from {info['file']}, not the copy {root}")
    return dg, info


MEGA_ARM, MEGA_BF16Q_ARM = "mega", "mega_bf16q"


def mega_arm_name(dg_path: Optional[str]) -> str:
    """The receipt's name for the substituted arms: 'mega' (stock kernel) or 'mega_bf16q' (the patched copy)."""
    return MEGA_BF16Q_ARM if dg_path else MEGA_ARM


def activate_dg(a) -> Optional[dict]:
    """--dg-path / --jit-cache of `downstream`: exactly layers' mechanism (import_dg: DG_JIT_CACHE_DIR, then the
    patched package imported from the copy). Must run before anything imports deep_gemm in this process (the
    first_light / runtime imports, build_and_load, the Generator): import_deep_gemm returns an already-imported
    deep_gemm as is, which would silently run the stock kernel under the 'mega_bf16q' name. Returns the receipt's
    deep_gemm record, or None when neither option is given (stock, nothing touched)."""
    if not (a.dg_path or a.jit_cache):
        return None
    if "deep_gemm" in sys.modules and a.dg_path:
        raise SystemExit("--dg-path: deep_gemm is already imported in this process; the patched copy would not be used")
    _, info = import_dg(a.dg_path, a.jit_cache)
    if a.dg_path and not info.get("patched_source"):
        raise SystemExit(f"--dg-path {a.dg_path}: the copy's {CUH} lacks the patch ({PATCH_TAG}); run patch-dg")
    return info


# ------------------------------------------------------------------------------------------------ model (synthetic)
def synth_x(R: int, D: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(R, D, generator=g, device="cpu")
    m = torch.exp2(torch.randint(-6, 7, (R, D // 32, 1), generator=g, device="cpu").float())
    return (x.view(R, D // 32, 32) * m).view(R, D).to(BF16)


def synth_layer(n_exp: int, D: int, I: int, g: torch.Generator, x: torch.Tensor) -> dict:
    """n_exp synthetic MXFP4 experts (+ one shared, also fp4 here) sized so gate/up ~ N(0, 1.5^2) on x."""
    xr = float(x.float().pow(2).mean().sqrt())
    s1 = 1.5 / (math.sqrt(D) * xr)
    s2 = 1.0 / math.sqrt(I)
    ex = []
    for _ in range(n_exp + 1):
        ex.append({"w1": quant_fp4(torch.randn(I, D, generator=g, device="cpu") * s1),
                   "w3": quant_fp4(torch.randn(I, D, generator=g, device="cpu") * s1),
                   "w2": quant_fp4(torch.randn(D, I, generator=g, device="cpu") * s2)})
    return {"experts": ex[:n_exp], "shared": ex[n_exp]}


def emu_gemm_factory(layer: dict, dev):
    """fp32 emulation of the DeepGEMM fp8 x fp4 GEMM (exact operands, fp32 accumulation, bf16 out)."""
    cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def W(e, p):
        if (e, p) not in cache:
            wb, c = layer["experts"][e][p]
            cache[(e, p)] = deq_fp4(wb.to(dev), c.to(dev)).float()
        return cache[(e, p)]

    def gemm(p, pairs, q, s):
        a = dequant(q, s).float()
        return torch.stack([(W(e, p) @ a[i]).to(BF16) for i, (_, _, e) in enumerate(pairs)])

    return gemm


def cmd_model(a) -> int:
    dev = torch.device(a.device)
    g = cpu_gen(a.seed)
    D, I, k, R = a.dim, a.inter, TOPK, ROWS
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "label":
           "SYNTHETIC: Gaussian weights quantized to MXFP4, synthetic activations (N(0,1) x per-32 2^-6..2^6), "
           "fp32-emulated GEMMs; no mega arm (the real kernel is box-only). A model of the drift decomposition, "
           "not a measurement of the checkpoint.", "dims": [D, I], "windows": a.windows}
    acc = None
    misc = []
    t0 = time.time()
    for wdx in range(a.windows):
        x = synth_x(R, D, g)
        layer = synth_layer(a.experts, D, I, g, x)
        rng = random.Random(a.seed * 1000 + wdx)
        ids = torch.tensor([sorted(rng.sample(range(a.experts), k)) for _ in range(R)], dtype=torch.long,
                           device="cpu")
        wr = torch.rand(R, k, generator=g, device="cpu") + 0.05
        wr = wr / wr.sum(-1, keepdim=True) * ROUTE_SCALE
        x, ids, wr = x.to(dev), ids.to(dev), wr.to(dev)

        def deq(e, layer=layer):
            return tuple(deq_fp4(layer["experts"][e][p][0].to(dev), layer["experts"][e][p][1].to(dev))
                         for p in ("w1", "w3", "w2"))

        sw = tuple(deq_fp4(layer["shared"][p][0].to(dev), layer["shared"][p][1].to(dev)) for p in ("w1", "w3", "w2"))
        a64 = dequant(*quant32(x.float(), scale_ref))
        shared = torch.stack([oracle_stages(a64[r], *sw, 1.0, LIM, "bf16")["o"] for r in range(R)])
        res = analyze_window(x, ids, wr, deq, shared, emu_gemm_factory(layer, dev))
        acc = merge(acc, res["stats"])
        misc.append({"swiglu_formula": res["swiglu_formula"], "unique_experts": res["unique_experts"]})
        print(f"[model] window {wdx + 1}/{a.windows} ({time.time() - t0:.0f} s)", flush=True)
    out["stats"] = finalize(acc)
    out["windows_misc"] = misc
    out["summary"] = summarize_stats(out["stats"])
    path = a.out or os.path.join(REPO, "scratch", f"m3n1-model-{time.strftime('%Y%m%d')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for line in summary_lines(out["summary"], "model (synthetic)"):
        print(line)
    print("wrote", path)
    return 0


def summarize_stats(st: dict) -> dict:
    """The few numbers the verdict reads (all rel-L2 vs oracle_ref unless named)."""
    def rl(*path):
        v = st
        for p in path:
            v = (v or {}).get(p)
        return None if v is None else v.get("rel_l2")

    def mf(*path):
        v = st
        for p in path:
            v = (v or {}).get(p)
        return None if v is None else v.get("mismatch_frac")

    s = {"final": {a_: rl("final", a_) for a_ in ARMS}, "routed": {a_: rl("routed", a_) for a_ in ARMS},
         "routed_vs_oracle_mega": {a_: rl("routed_vs_oracle_mega", a_) for a_ in ARMS},
         "routed_vs_exact": {a_: rl("routed_vs_exact", a_) for a_ in ARMS},
         "final_vs_exact": {a_: rl("final_vs_exact", a_) for a_ in ARMS},
         "oracle_ref_vs_exact": rl("oracles_vs_exact", "oracle_ref"),
         "oracle_mega_vs_exact": rl("oracles_vs_exact", "oracle_mega"),
         "oracle_mega_vs_oracle_ref": rl("oracle_mega_vs_oracle_ref", "routed"),
         "l2_input_rel_l2": {a_: rl("l2_input", a_) for a_ in ("ours", "tm", "tm_fix")},
         "l2_input_byte_mismatch": {a_: mf("l2_input", a_) for a_ in ("ours", "tm", "tm_fix")},
         "l2_input_tm_vs_ours_byte_mismatch": mf("l2_input_pairwise", "tm_vs_ours"),
         "l1_gate_up_rel_l2": rl("l1_gate_up", "ours=tm"),
         "l2_out_rel_l2": {a_: rl("l2_out", a_) for a_ in ("ours", "tm", "tm_fix")},
         "pairwise_routed_mismatch": {k_: mf("pairwise_routed", k_) for k_ in
                                      ("tm_vs_ours", "tm_fix_vs_ours", "mega_vs_ours", "mega_vs_tm", "mega_vs_tm_fix")}}
    return s


def summary_lines(s: dict, label: str) -> List[str]:
    f = lambda v: "not measured" if v is None else f"{v:.4g}"  # noqa: E731
    L = [f"[{label}] rel-L2 to oracle_ref, final (+shared): " +
         ", ".join(f"{a_} {f(s['final'].get(a_))}" for a_ in ARMS),
         f"[{label}] routed: " + ", ".join(f"{a_} {f(s['routed'].get(a_))}" for a_ in ARMS),
         f"[{label}] stages: L1 gate/up {f(s['l1_gate_up_rel_l2'])}; L2 input " +
         ", ".join(f"{a_} {f(s['l2_input_rel_l2'][a_])} (bytes {f(s['l2_input_byte_mismatch'][a_])})"
                   for a_ in ("ours", "tm", "tm_fix")) +
         f"; tm vs ours L2-input bytes differ {f(s['l2_input_tm_vs_ours_byte_mismatch'])}",
         f"[{label}] L2 out: " + ", ".join(f"{a_} {f(s['l2_out_rel_l2'][a_])}" for a_ in ("ours", "tm", "tm_fix")),
         f"[{label}] routed mismatch pairs: " + ", ".join(f"{k_} {f(v)}" for k_, v in
                                                       s["pairwise_routed_mismatch"].items()),
         f"[{label}] routed vs oracle_mega (MegaMoE's rounding points): " +
         ", ".join(f"{a_} {f(s['routed_vs_oracle_mega'].get(a_))}" for a_ in ARMS) +
         f"; oracle_mega vs oracle_ref {f(s['oracle_mega_vs_oracle_ref'])}",
         f"[{label}] routed vs EXACT (no bf16 / fp8 rounding after the weights): " +
         ", ".join(f"{a_} {f(s['routed_vs_exact'].get(a_))}" for a_ in ARMS) +
         f"; oracle_ref {f(s['oracle_ref_vs_exact'])}, oracle_mega {f(s['oracle_mega_vs_exact'])}"]
    return L


# ------------------------------------------------------------------------------------------------ layers (box)
class LayerBox:
    """One layer, the spike's layout (G = H hot + S spare, the hot map's slots), from the checkpoint: the
    per-projection tensors + packed SF (today's DeepGEMM contiguous GEMM) and the mega layouts (interleaved L1 copy,
    UTCCP SF, w2 shared); cold experts placed into spare groups H + c by the routing (mega_route). Box: CUDA, the
    real checkpoint and deep_gemm; the CPU tests drive it with a fake deep_gemm and a synthetic checkpoint at small
    dims (`dims`)."""

    def __init__(self, dg, ck, L: int, hot: Sequence[int], S: int, dev, T: int = ROWS, sf_major: str = "mn",
                 dims: Tuple[int, int, int, int] = (DIM, INTER, N_EXPERTS, TOPK)):
        from runtime import dg_inplace as DI
        from runtime import dg_mega as DM

        self.DI, self.DM, self.dg, self.ck, self.L, self.dev = DI, DM, dg, ck, L, torch.device(dev)
        self.D, self.I, self.E, self.k = (int(v) for v in dims)
        self.hot, self.H, self.S = list(hot), len(hot), int(S)
        self.G = self.H + self.S
        self.sf_major = sf_major
        NK = {"w1": (self.I, self.D), "w3": (self.I, self.D), "w2": (self.D, self.I)}
        self.NK = NK
        t0 = time.time()
        self.full = {p: torch.zeros(self.G, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in NK}
        self.codes = {p: torch.zeros(self.G, NK[p][0], NK[p][1] // 32, dtype=torch.uint8, device=dev) for p in NK}
        for slot, e in enumerate(self.hot):
            for p in NK:
                w_, s_ = ck.expert(L, e, p)
                self.full[p][slot].copy_(w_)
                self.codes[p][slot].copy_(s_)
        self.sfp = {p: DI.packed_sf_zeros(self.G, NK[p][0], NK[p][1] // 128, dev) for p in NK}
        for p in NK:
            self.sfp[p][:self.H].copy_(DI.pack_words(self.codes[p][:self.H]))
        self.l1 = DM.mega_l1_weight(self.full["w1"], self.full["w3"])
        self.sf1 = DM.mega_l1_sf(self.codes["w1"], self.codes["w3"], sf_major)
        self.sf2 = DM.mega_l2_sf(self.codes["w2"], sf_major)
        self.shared = {}
        for p in NK:
            wt = ck.get(f"layers.{L}.ffn.shared_experts.{p}.weight").to(dev)
            sc = DM.act_codes(ck.get(f"layers.{L}.ffn.shared_experts.{p}.scale")).to(dev)
            self.shared[p] = deq_fp8_rows(wt, DM.block_codes_per_row(sc, wt.size(0)))
        gt = torch.full((self.E,), -1, dtype=torch.int32, device="cpu")
        for slot, e in enumerate(self.hot):
            gt[e] = slot
        self.gtab = gt.to(dev)
        self.placed: Dict[int, int] = {}
        self.sym = DM.make_symm_buffer(dg, self.G, T, self.k, self.D, self.I, 0)
        self.symm = DM.describe_symm(self.sym)
        self.y = torch.zeros(T, self.D, dtype=BF16, device=dev)
        self._sync()
        self.load_s = round(time.time() - t0, 1)

    def _sync(self) -> None:
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def place(self, e: int, grp: int) -> None:
        DM, DI = self.DM, self.DI
        for p in self.NK:
            w_, s_ = self.ck.expert(self.L, e, p)
            self.full[p][grp].copy_(w_)
            self.codes[p][grp].copy_(s_)
            self.sfp[p][grp:grp + 1].copy_(DI.pack_words(self.codes[p][grp:grp + 1]))
        DM.mega_l1_weight(self.full["w1"][grp], self.full["w3"][grp], out=self.l1[grp])
        self.sf1[grp:grp + 1].copy_(DM.mega_l1_sf(self.codes["w1"][grp:grp + 1], self.codes["w3"][grp:grp + 1], "k"))
        self.sf2[grp:grp + 1].copy_(DM.mega_l2_sf(self.codes["w2"][grp:grp + 1], "k"))
        self.placed[grp] = e

    def route(self, ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
        """-> (topk pool groups [R, k], {expert: group}); cold experts placed into their spare groups."""
        m = self.DM.mega_route(ids, self.gtab, self.H, self.S, self.E)
        if int(m["n_over"]) > 0:
            raise RuntimeError(f"L{self.L}: {int(m['n_cold'])} cold unique experts > {self.S} spare slots")
        for c, e in enumerate(m["stage"].tolist()):
            if e >= 0 and self.placed.get(self.H + c) != e:
                self.place(int(e), self.H + c)
        topk = m["topk"]
        gmap = {int(ids[r, j]): int(topk[r, j]) for r in range(ids.size(0)) for j in range(ids.size(1))}
        return topk, gmap

    def deq(self, gmap: Dict[int, int]):
        def f(e):
            grp = gmap[int(e)]
            return tuple(deq_fp4(self.full[p][grp], self.codes[p][grp]) for p in ("w1", "w3", "w2"))
        return f

    def gemm(self, gmap: Dict[int, int]):
        DI = self.DI

        def f(p, pairs, q, s):
            blocks = [[(i, gmap[e])] for i, (_, _, e) in enumerate(pairs)]
            acts = {i: (q[i], s[i].float()) for i in range(len(pairs))}
            o = DI.run_contiguous(self.dg, self.full[p], self.sfp[p], blocks, acts, DI.MK_ALIGN, len(blocks))
            return torch.stack([o[i] for i in range(len(pairs))])
        return f

    def set_rows(self, topk: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> None:
        DM, sym, R = self.DM, self.sym, x.size(0)
        sym.topk_idx.fill_(-1)
        sym.topk_weights.zero_()
        sym.x.view(torch.uint8).zero_()
        q, s = quant32(x.float(), scale_ref)
        sym.topk_idx[:R].copy_(topk)
        sym.topk_weights[:R].copy_(w.float())
        sym.x[:R].copy_(q)
        DM.write_act_sf(sym.x_sf, DM.act_codes(s), R)

    def call(self) -> torch.Tensor:
        self.dg.fp8_fp4_mega_moe(self.y, (self.l1, self.sf1), (self.full["w2"], self.sf2), self.sym,
                                 recipe=self.DM.MEGA_RECIPE, activation="swiglu", activation_clamp=LIM,
                                 fast_math=False)
        return self.y

    def mega(self, ids: torch.Tensor, w: torch.Tensor, x: torch.Tensor):
        """-> (routed bf16 [R, D], the l2_acts rows written by this call, uint8 on the CPU, or None)."""
        topk, _ = self.route(ids)
        self.set_rows(topk, w, x)
        l2 = getattr(self.sym, "l2_acts", None)
        if l2 is not None:
            l2.view(torch.uint8).zero_()
        self.call()
        self._sync()
        y = self.y[:x.size(0)].clone()
        snap = None
        if l2 is not None:
            u = l2.view(torch.uint8)
            snap = u[(u != 0).any(1)].cpu()
        return y, snap

    def shared_out(self, x: torch.Tensor) -> torch.Tensor:
        a64 = dequant(*quant32(x.float(), scale_ref))
        return torch.stack([oracle_stages(a64[r], self.shared["w1"], self.shared["w3"], self.shared["w2"], 1.0,
                                          LIM, "bf16")["o"] for r in range(x.size(0))])


def load_windows(acts: dict, L: int, n_windows: int, start: int, R: int = ROWS) -> List[dict]:
    """The captured fixture activations at layer L -> up to n_windows R-row windows (one per prompt, positions
    start .. start+R-1 of its teacher-forced decode)."""
    out = []
    for tag in sorted(acts["rows"]):
        per = acts["rows"][tag] or {}
        d = per.get(L) if L in per else per.get(str(L))
        if not d:
            continue
        pos = list(d["pos"])
        if start not in pos:
            continue
        i0 = pos.index(start)
        if i0 + R > len(pos) or pos[i0:i0 + R] != list(range(start, start + R)):
            continue
        out.append({"tag": tag, "pos": pos[i0:i0 + R], "x": d["x"][i0:i0 + R], "ids": d["ids"][i0:i0 + R],
                    "w": d["w"][i0:i0 + R]})
        if len(out) >= n_windows:
            break
    return out


def run_layer(box: LayerBox, wins: List[dict], timer: Optional[Callable] = None, iters: int = 200,
              log=print) -> dict:
    """Every window of one layer: R = 6 (analyze_window, the kernel's l2_acts matched), R = 1 (each row alone
    through the kernel: invariance + its distances), then `timer` (graph_us on the box) on window 0 at R = 1 / 6."""
    dev, R, k, D = box.dev, ROWS, box.k, box.D
    rec: Dict[str, object] = {"H": box.H, "S": box.S, "G": box.G, "load_s": box.load_s, "symm": box.symm,
                              "windows": []}
    acc6 = acc1 = None
    inv_all, l2m, sw = [], [], {"n": 0, "n_bitwise_equal": 0}
    for wdx, win in enumerate(wins):
        x = win["x"].to(dev).to(BF16).reshape(R, D)
        ids = win["ids"].to(dev).long().reshape(R, k)
        w = win["w"].to(dev).float().reshape(R, k)
        _, gmap = box.route(ids)
        sh = box.shared_out(x)
        res = analyze_window(x, ids, w, box.deq(gmap), sh, box.gemm(gmap),
                             mega=lambda: box.mega(ids, w, x), want_raw=True)
        acc6 = merge(acc6, res["stats"])
        if res.get("l2_match"):
            l2m.append(res["l2_match"])
        sw = {k_: sw[k_] + res["swiglu_formula"][k_] for k_ in sw}
        raw = res["raw"]
        y6 = raw["y"]["mega"][0]
        y1 = torch.cat([box.mega(ids[r:r + 1], w[r:r + 1], x[r:r + 1])[0] for r in range(R)])
        inv = [bool(torch.equal(y1[r].view(torch.int16), y6[r].view(torch.int16))) for r in range(R)]
        inv_all.append(inv)
        st1 = {"routed": {"mega": st_bf16(y1, raw["routed64"]["ref"])},
               "final": {"mega": st_bf16((y1.float() + sh.float()).to(BF16), raw["final64"])},
               "pairwise_routed": {"mega_vs_tm": st_bf16(y1, raw["y"]["tm"][0].double()),
                                   "mega_vs_ours": st_bf16(y1, raw["y"]["ours"][0].double())}}
        acc1 = merge(acc1, st1)
        rec["windows"].append({"tag": win["tag"], "pos": win["pos"], "unique_experts": res["unique_experts"],
                               "row_invariant": inv, "l2_match": res.get("l2_match"),
                               "mega_sha256": hashlib.sha256(y6.cpu().view(torch.int16).numpy().tobytes())
                               .hexdigest()})
        log(f"[L{box.L}] window {wdx + 1}/{len(wins)} {win['tag']}: {res['unique_experts']} experts, row-invariant "
            f"{all(inv)}, l2 rows matched {(res.get('l2_match') or {}).get('rows_matched')}", flush=True)
        del res, raw
    rec["R6"] = finalize(acc6) if acc6 else None
    rec["R1"] = finalize(acc1) if acc1 else None
    rec["summary_R6"] = summarize_stats(rec["R6"]) if rec["R6"] else None
    rec["row_invariant_all"] = bool(inv_all) and all(all(v) for v in inv_all)
    rec["swiglu_formula"] = sw
    if l2m:
        agg = {k_: sum(m.get(k_) or 0.0 for m in l2m) / len(l2m)
               for k_ in ("frac_bytes_eq_ours", "frac_bytes_eq_tm", "frac_bytes_eq_tm_fix")}
        for k_ in ("rows_matched", "rows_written", "pairs", "pairs_covered"):
            agg[k_] = sum(m.get(k_, 0) for m in l2m)
        rec["l2_match"] = agg
    if timer is not None and wins:
        win = wins[0]
        x = win["x"].to(dev).to(BF16).reshape(R, D)
        ids = win["ids"].to(dev).long().reshape(R, k)
        w = win["w"].to(dev).float().reshape(R, k)
        sp = {}
        for rr in (1, R):
            topk, _ = box.route(ids[:rr])
            box.set_rows(topk, w[:rr], x[:rr])
            sp[f"mega_R{rr}_hot_us"] = timer(box.call, iters=iters)
        rec["speed"] = sp
        log(f"[L{box.L}] speed {json.dumps(sp)}", flush=True)
    return rec


def cmd_layers(a) -> int:  # pragma: no cover - box only
    from dg_contiguous_spike import Ckpt, graph_us

    dg, dginfo = import_dg(a.dg_path, a.jit_cache)
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()
    dev = torch.device("cuda")
    try:
        acts = torch.load(a.acts, map_location="cpu", weights_only=False)     # our own capture file
    except TypeError:  # older torch: no weights_only
        acts = torch.load(a.acts, map_location="cpu")
    hot_map = json.load(open(a.map))
    ck = Ckpt(a.ckpt)
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "deep_gemm": dginfo,
           "deep_gemm_version": getattr(dg, "__version__", None), "acts": a.acts, "acts_meta": acts.get("meta"),
           "labels": {"activations": "fixture (captured MoE inputs, teacher-forced on Pg61 tokens)",
                      "routing": "the reference gate on the captured x", "expert_weights": "real layer bytes",
                      "shared": "oracle shared expert (fp64 dots, bf16 out), identical in every arm"},
           "kernel": "PATCHED bf16-before-quant" if dginfo.get("patched_source") else "installed",
           "layers": {}, "errors": []}
    path = a.out or os.path.join(REPO, "scratch", f"m3n1-layers-{time.strftime('%Y%m%d')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for L in [int(v) for v in a.layers.split(",")]:
        try:
            box = LayerBox(dg, ck, L, [int(e) for e in hot_map[str(L)]], a.spare, dev)
            wins = load_windows(acts, L, a.windows, a.win_start)
            out["layers"][str(L)] = run_layer(box, wins, timer=graph_us, iters=a.iters)
            del box
        except Exception:
            tb = traceback.format_exc()
            out["layers"][str(L)] = {"error": tb[-3000:]}
            out["errors"].append(f"L{L}")
            print(f"[L{L}] FAILED\n{tb}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    for L, rec in out["layers"].items():
        if rec.get("summary_R6"):
            for line in summary_lines(rec["summary_R6"], f"L{L} R6 {out['kernel']}"):
                print(line)
    print("wrote", path)
    return 0 if not out["errors"] else 2


# ------------------------------------------------------------------------------------------------ downstream (box)
class Capture:
    """forward-pre-hooks on layers[l].ffn: the MoE input x of every 1-row decode step at positions 1..steps of the
    current prompt, with the reference gate's (model.py:894) ids (ascending) / weights on it."""

    def __init__(self, model, layers: Sequence[int], steps: int):
        self.layers, self.steps = list(layers), int(steps)
        self.data: Dict[str, Dict[int, dict]] = {}
        self.tag: Optional[str] = None
        self.pos = 0
        self.handles = [model.layers[l].ffn.register_forward_pre_hook(self._hook(l)) for l in self.layers]

    def _hook(self, l: int):
        def fn(mod, args):
            x = args[0]
            if self.tag is None or not 1 <= self.pos <= self.steps or x.numel() != mod.dim:
                return None
            x2 = x.detach().reshape(1, mod.dim)
            wts, idx = mod.gate(x2, None)
            ids, order = torch.sort(idx, dim=-1)
            d = self.data.setdefault(self.tag, {}).setdefault(l, {"pos": [], "x": [], "ids": [], "w": []})
            d["pos"].append(self.pos)
            d["x"].append(x2.to("cpu"))
            d["ids"].append(ids.to("cpu"))
            d["w"].append(torch.gather(wts, -1, order).float().to("cpu"))
            return None
        return fn

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def save(self, path: str, meta: dict) -> dict:
        rows = {}
        for tag, per in self.data.items():
            rows[tag] = {l: {"pos": d["pos"], "x": torch.cat(d["x"]), "ids": torch.cat(d["ids"]),
                             "w": torch.cat(d["w"])} for l, d in per.items()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"rows": rows, "meta": meta}, path)
        return {"path": path, "prompts": len(rows), "rows_per_layer": {l: sum(len(rows[t][l]["pos"]) for t in rows
                                                                              if l in rows[t]) for l in self.layers}}


def pg61_targets(fixture_rows: List[dict], model_dir: str) -> Tuple[Dict[str, List[int]], dict]:
    from gap_ledger import pg61_ids, vocab_by_bytes

    with open(os.path.join(model_dir, "tokenizer.json")) as fh:
        table, added, amb = vocab_by_bytes(json.load(fh))
    out, bad = {}, {}
    for r in fixture_rows:
        ids, b = pg61_ids(r, table, added)
        out[r["tag"]] = ids
        if b:
            bad[r["tag"]] = b[:20]
    return out, {"unmapped_positions": bad, "ambiguous_byte_strings": len(amb)}


def pg61_margins(row: dict) -> Tuple[List[float], List[float]]:
    """Pg61's (top1 - top2 logprob, logprob of its own token) per position."""
    m, lp = [], []
    for e in row.get("logprobs") or []:
        t = sorted(e.get("top_logprobs") or [], key=lambda v: -v["logprob"])
        m.append(t[0]["logprob"] - t[1]["logprob"] if len(t) > 1 else 0.0)
        lp.append(float(e["logprob"]))
    return m, lp


@torch.inference_mode()
def tf_pass(gen, fixture_rows: List[dict], targets: Dict[str, List[int]], max_new: int, top_n: int = 5,
            cap: Optional[Capture] = None, log=print) -> Dict[str, dict]:
    """Teacher-forced 1-row decode along Pg61's tokens: position i's logits (prefill for i = 0, then the step that
    fed target i-1) -> argmax, top-n (ids, logprobs), the logprob of target i. Stops at an unmapped target."""
    out = {}
    for row in fixture_rows:
        tag = row["tag"]
        tg = list(targets.get(tag) or [])
        stop = next((i for i, t in enumerate(tg) if t < 0), len(tg))
        tg = tg[:min(stop, max_new)]
        ids = gen.encode([{"role": "user", "content": row["prompt"]}])
        L = len(ids)
        rec = {"n": len(tg), "prompt_tokens": L, "argmax": [], "lp_target": [], "top_ids": [], "top_lp": []}
        t0 = time.time()
        if cap is not None:
            cap.tag, cap.pos = tag, 0
        logits = gen.prefill(ids)
        for i, t in enumerate(tg):
            lc = logits.detach().to("cpu", torch.float32)
            lp = torch.log_softmax(lc, -1)
            v, ix = torch.topk(lp, top_n)
            rec["argmax"].append(int(torch.argmax(lc)))
            rec["lp_target"].append(float(lp[t]))
            rec["top_ids"].append(ix.tolist())
            rec["top_lp"].append([float(z) for z in v])
            if i == len(tg) - 1:
                break
            if cap is not None:
                cap.pos = i + 1
            logits = gen.step(int(t), L + i)
        if cap is not None:
            cap.tag = None
        rec["wall_s"] = round(time.time() - t0, 1)
        out[tag] = rec
        log(f"[tf] {tag:6s} {len(tg)} forced positions, {rec['wall_s']} s", flush=True)
    return out


def tf_compare(cand: Dict[str, dict], fixture_rows: List[dict], targets: Dict[str, List[int]],
               base: Optional[Dict[str, dict]] = None) -> dict:
    """The A3 metric over teacher-forced (= shared) prefixes: a flip at position i = the candidate's argmax is not
    the reference's top-1; weighted by the REFERENCE's margin (top1 - top2 logprob). vs Pg61: the reference top-1 is
    Pg61's token, the margin from its top-20; vs base: our unsubstituted run's argmax and margin. Plus
    |logprob(forced token) - reference's|."""
    by_tag = {r["tag"]: r for r in fixture_rows}
    per, tot = {}, {"positions": 0, "flips_pg61": 0, "flips_w_pg61": 0.0, "flips_base": 0, "flips_w_base": 0.0,
                    "max_abs_dlp_pg61": 0.0, "max_abs_dlp_base": 0.0, "sum_abs_dlp_base": 0.0}
    for tag, c in cand.items():
        n = c["n"]
        m_pg, lp_pg = pg61_margins(by_tag[tag])
        tg = targets[tag]
        fl = [i for i in range(n) if c["argmax"][i] != tg[i]]
        d = {"n": n, "pg61": {"flips": len(fl), "flips_weighted": round(sum(abs(m_pg[i]) for i in fl), 4),
                              "flip_positions": fl[:20],
                              "max_abs_dlogprob": round(max((abs(c["lp_target"][i] - lp_pg[i]) for i in range(n)),
                                                            default=0.0), 5)}}
        tot["positions"] += n
        tot["flips_pg61"] += len(fl)
        tot["flips_w_pg61"] += d["pg61"]["flips_weighted"]
        tot["max_abs_dlp_pg61"] = max(tot["max_abs_dlp_pg61"], d["pg61"]["max_abs_dlogprob"])
        if base is not None and tag in base:
            b = base[tag]
            nb = min(n, b["n"])
            mb = [b["top_lp"][i][0] - b["top_lp"][i][1] for i in range(nb)]
            fb = [i for i in range(nb) if c["argmax"][i] != b["argmax"][i]]
            dl = [abs(c["lp_target"][i] - b["lp_target"][i]) for i in range(nb)]
            d["base"] = {"flips": len(fb), "flips_weighted": round(sum(abs(mb[i]) for i in fb), 4),
                         "flip_positions": fb[:20], "flip_margins": [round(mb[i], 4) for i in fb[:20]],
                         "max_abs_dlogprob": round(max(dl, default=0.0), 5),
                         "mean_abs_dlogprob": round(sum(dl) / len(dl), 6) if dl else None}
            tot["flips_base"] += len(fb)
            tot["flips_w_base"] += d["base"]["flips_weighted"]
            tot["max_abs_dlp_base"] = max(tot["max_abs_dlp_base"], d["base"]["max_abs_dlogprob"])
            tot["sum_abs_dlp_base"] += sum(dl)
        per[tag] = d
    tot["mean_abs_dlp_base"] = tot.pop("sum_abs_dlp_base") / tot["positions"] if (base and tot["positions"]) \
        else None
    for k_ in ("flips_w_pg61", "flips_w_base"):
        tot[k_] = round(tot[k_], 4)
    return {"total": tot, "per_prompt": per}


def stream_vs_base(cand_path: str, base_path: str) -> dict:
    """Free-running streams: first divergence of the candidate from our unsubstituted stream, with the base's
    margin there (a near-tie divergence vs a real one)."""
    def load(p):
        with open(p) as fh:
            return {json.loads(l)["tag"]: json.loads(l) for l in fh if l.strip()}

    c, b = load(cand_path), load(base_path)
    per = {}
    for tag in sorted(b):
        if tag not in c:
            per[tag] = {"missing": True}
            continue
        bt = [e["token"] for e in b[tag]["logprobs"]]
        ct = [e["token"] for e in c[tag]["logprobs"]]
        n = min(len(bt), len(ct))
        first = next((i for i in range(n) if bt[i] != ct[i]), None)
        margin = None
        if first is not None:
            t = sorted(b[tag]["logprobs"][first]["top_logprobs"], key=lambda v: -v["logprob"])
            margin = round(t[0]["logprob"] - t[1]["logprob"], 4) if len(t) > 1 else None
        per[tag] = {"first_divergence": first, "base_margin_there": margin, "compared": n}
    return {"identical": sum(1 for v in per.values() if v.get("first_divergence") is None and not v.get("missing")),
            "n": len(per), "per_prompt": per}


def parse_groups(spec: str, layers: Sequence[int]) -> List[List[int]]:
    """'auto' -> [all]; 'a,b|c|d' -> [[a, b], [c], [d]]; 'each' -> one group per layer; 'all+each' -> both."""
    layers = [int(v) for v in layers]
    if spec in ("auto", "all"):
        return [layers]
    if spec == "each":
        return [[l] for l in layers]
    if spec == "all+each":
        return [layers] + [[l] for l in layers]
    return [[int(v) for v in g.split(",") if v.strip()] for g in spec.split("|") if g.strip()]


def split_group(g: List[int]) -> List[List[int]]:
    if len(g) <= 1:
        return []
    h = (len(g) + 1) // 2
    return [g[:h], g[h:]]


def cmd_downstream(a) -> int:  # pragma: no cover - box only
    dginfo = activate_dg(a)                       # before ANY import that could pull deep_gemm in
    from first_light import load_fixture, run_parity, run_rep
    from runtime import dg_mega as DM
    from runtime.fused_glue_r2 import FP32_IMPL_ENV
    from runtime.generate import Generator, set_determinism
    from runtime.loader import build_and_load

    from runtime import dg_inplace as DI

    os.environ[FP32_IMPL_ENV] = a.fp32_impl
    os.environ[DI.ROWMATCH_ENV] = a.moe_rowmatch
    date = time.strftime("%Y%m%d")
    stem = a.stem or os.path.join(REPO, "scratch", f"m3n1-ds-{date}")
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:], "errors": [], "arms": [],
           "config": {k_: getattr(a, k_) for k_ in ("model_dir", "map", "mode", "moe_impl", "fp8_impl", "glue_impl",
                                                    "attn_impl", "fp32_impl", "moe_rowmatch", "max_pos", "max_new",
                                                    "shared", "groups", "dg_path", "jit_cache")}}
    out["mega_arm"] = mega_arm_name(a.dg_path)
    out["deep_gemm"] = dginfo
    path = a.out or stem + ".json"

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, default=str)

    rows = load_fixture(a.fixture)
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    fixture2 = os.path.join(os.path.dirname(a.fixture), "parity-A2.jsonl")
    fixture2 = fixture2 if os.path.exists(fixture2) else None
    targets, tmeta = pg61_targets(rows, a.model_dir)
    out["targets"] = tmeta
    t0 = time.time()
    model = build_and_load(a.model_dir, a.map, "cuda", reserve_host_gib=a.reserve_host_gib)
    out["boot"] = {"wall_s": round(time.time() - t0, 1)}
    out["determinism"] = set_determinism(0, False if a.no_fill_uninit else None)
    gen = Generator(model, a.model_dir, mode=a.mode, max_pos=a.max_pos, moe_impl=a.moe_impl, fp8_impl=a.fp8_impl,
                    glue_impl=a.glue_impl, attn_impl=a.attn_impl)
    st = gen.static_step()
    if a.mode == "graph":
        gen.capture()
    f_, t_ = torch.cuda.mem_get_info()
    out["hbm_after_build_GiB"] = {"free": round(f_ / 2 ** 30, 2), "total": round(t_ / 2 ** 30, 2)}
    out["pool"] = {"spare": int(st.pool.spare), "G": int(st.pool.hbm_full[0]["w2"].size(0)), "topk": st.topk}
    print(f"[hbm] after the production 1-row build: {out['hbm_after_build_GiB']}; pool spare {st.pool.spare}",
          flush=True)
    layers = [int(v) for v in a.layers.split(",")]
    plan = DM.hbm_bytes(1, [out["pool"]["G"]], DIM, INTER, "copy")
    out["mega_plan_per_layer_GiB"] = plan["total_GiB"]
    save()

    def arm(label: str, sub_layers: Optional[List[int]]):
        rec = {"label": label, "layers": sub_layers}
        sub = None
        try:
            if sub_layers:
                f0 = torch.cuda.mem_get_info()[0]
                try:
                    sub = DM.MegaSubstitution(st, sub_layers, shared=a.shared)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:   # the HBM check (or the copy) refused
                    rec["fit"] = False
                    rec["free_before_GiB"] = round(f0 / 2 ** 30, 2)
                    rec["error"] = f"{type(e).__name__}: {str(e)[:600]}"
                    print(f"[arm {label}] did not fit: {rec['error']}", flush=True)
                    torch.cuda.empty_cache()
                    return rec
                rec["fit"] = True
                rec["substitution"] = {"plan_GiB": sub.bufs.plan["total_GiB"], "free_before_GiB": round(f0 / 2 ** 30, 2),
                                       "free_after_GiB": round(torch.cuda.mem_get_info()[0] / 2 ** 30, 2),
                                       "spare": sub.S, "T": sub.bufs.T, "G": sorted({t["G"] for t in
                                                                                    sub.bufs.layers.values()})}
                if a.mode == "graph":
                    st.release_graphs()
                    gen.capture()
            cap = Capture(model, [int(v) for v in a.capture_layers.split(",")], a.capture_steps) \
                if (sub_layers is None and a.mode == "static" and a.acts and not a.dg_path) else None
            tf = tf_pass(gen, rows, targets, a.max_new, cap=cap)
            if cap is not None:
                cap.close()
                out["capture"] = cap.save(a.acts, {"layers": cap.layers, "steps": cap.steps, "config": out["config"]})
                print(f"[capture] {out['capture']}", flush=True)
            rec["tf"] = tf
            if a.free_run:
                p = f"{stem}-{label}.jsonl"
                res = run_rep(gen, rows, a.max_new, p, 1)
                rec["free_run"] = {"path": p, "errors": [r["error"][-500:] for r in res if "error" in r]}
                pr = run_parity(p, a.fixture, fixture2, p + ".parity.json")
                rec["free_run"]["parity_verdict_line"] = pr.get("verdict_line")
                rec["free_run"]["parity_per_prompt"] = pr.get("per_prompt")
            if sub is not None:
                rec["mega_calls"] = sub.bufs.n_calls
        except Exception:
            rec["error"] = traceback.format_exc()[-3000:]
            out["errors"].append(label)
            print(f"[arm {label}] FAILED\n{rec['error']}", flush=True)
        finally:
            if sub is not None:
                sub.restore()
                if a.mode == "graph":
                    st.release_graphs()
                    gen.capture()
        return rec

    base = arm("base", None)
    out["arms"].append(base)
    save()
    if a.capture_only:
        print("wrote", path)
        return 0 if not out["errors"] else 2
    queue = parse_groups(a.groups, layers)
    while queue:
        g = queue.pop(0)
        rec = arm(out["mega_arm"] + "-L" + "-".join(str(l) for l in g), g)
        rec["arm"] = out["mega_arm"]
        out["arms"].append(rec)
        if rec.get("fit") is False and a.groups == "auto":
            queue = split_group(g) + queue
            print(f"[auto] {g} did not fit -> {split_group(g)}", flush=True)
        save()
    # metrics
    btf = base.get("tf")
    for rec in out["arms"]:
        if rec.get("tf"):
            rec["tf_metrics"] = tf_compare(rec["tf"], rows, targets, btf if rec is not base else None)
            if rec is not base and base.get("free_run") and rec.get("free_run"):
                rec["free_vs_base"] = stream_vs_base(rec["free_run"]["path"], base["free_run"]["path"])
        rec.pop("tf", None) if not a.keep_tf else None
    out["summary"] = [{"label": r["label"], "arm": r.get("arm", "base"), "layers": r["layers"], "fit": r.get("fit"),
                       "tf": (r.get("tf_metrics") or {}).get("total"),
                       "free_parity": (r.get("free_run") or {}).get("parity_verdict_line"),
                       "free_vs_base_identical": (r.get("free_vs_base") or {}).get("identical")}
                      for r in out["arms"]]
    save()
    for s in out["summary"]:
        print("[summary]", json.dumps(s))
    print("wrote", path)
    return 0 if not out["errors"] else 2


# ------------------------------------------------------------------------------------------------ report
def _f(v, d=4):
    if v is None:
        return "not measured"
    if isinstance(v, float):
        return f"{v:.{d}g}"
    return str(v)


def report_lines(layers: List[dict], patched: List[dict], ds: Optional[dict], model: Optional[dict],
                 ds_patched: Optional[dict] = None) -> List[str]:
    L: List[str] = []
    rows = {}
    for rc in layers:
        for l, rec in rc.get("layers", {}).items():
            rows.setdefault(l, {})["unpatched"] = rec
    for rc in patched:
        for l, rec in rc.get("layers", {}).items():
            rows.setdefault(l, {})["patched"] = rec
    L.append("| layer | R | ours final | mega final (gemv) | tm | tm_fix | tm_fix_1r | mega bf16q | "
             "L2-input bytes != oracle: ours / tm / mega-vs-tm (l2_acts) | mega vs tm routed mismatch |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for l in sorted(rows, key=int):
        u, p = rows[l].get("unpatched") or {}, rows[l].get("patched") or {}
        s = u.get("summary_R6") or {}
        sp = p.get("summary_R6") or {}
        fin = s.get("final") or {}
        l2 = u.get("l2_match") or {}
        L.append(f"| {l} | 6 | {_f(fin.get('ours'))} | {_f(fin.get('mega'))} | {_f(fin.get('tm'))} | "
                 f"{_f(fin.get('tm_fix'))} | {_f(fin.get('tm_fix_1r'))} | {_f((sp.get('final') or {}).get('mega'))} | "
                 f"{_f((s.get('l2_input_byte_mismatch') or {}).get('ours'))} / "
                 f"{_f((s.get('l2_input_byte_mismatch') or {}).get('tm'))} / "
                 f"{_f(None if not l2 else 1 - (l2.get('frac_bytes_eq_tm') or 0))} | "
                 f"{_f((s.get('pairwise_routed_mismatch') or {}).get('mega_vs_tm'))} |")
        r1 = (u.get("R1") or {})
        L.append(f"| {l} | 1 | (row-independent: = R 6) | {_f(((r1.get('final') or {}).get('mega') or {}).get('rel_l2'))}"
                 f" | | | | {_f((((p.get('R1') or {}).get('final') or {}).get('mega') or {}).get('rel_l2'))} | "
                 f"row-invariant {u.get('row_invariant_all')} | "
                 f"{_f(((r1.get('pairwise_routed') or {}).get('mega_vs_tm') or {}).get('mismatch_frac'))} |")
    L.append("")
    L.append("| layer | mega R1 hot us | mega R6 hot us | bf16q R1 | bf16q R6 |")
    L.append("|---|---|---|---|---|")
    for l in sorted(rows, key=int):
        su = (rows[l].get("unpatched") or {}).get("speed") or {}
        spp = (rows[l].get("patched") or {}).get("speed") or {}
        g = lambda d, R: (d.get(f"mega_R{R}_hot_us") or {}).get("us")  # noqa: E731
        L.append(f"| {l} | {_f(g(su, 1))} | {_f(g(su, 6))} | {_f(g(spp, 1))} | {_f(g(spp, 6))} |")
    if ds or ds_patched:
        L.append("")
        L.append("| arm | layers | fit | TF positions | flips vs Pg61 (weighted) | flips vs ours (weighted) | "
                 "max abs dlogprob vs ours | free-run parity | free-run == ours |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        # stock then patched; the patched receipt's base arm duplicates the stock one, so it is shown once
        summ = [(s, False) for s in (ds or {}).get("summary") or []]
        summ += [(s, True) for s in (ds_patched or {}).get("summary") or []
                 if not (ds and s.get("arm", "base") == "base")]
        for s, pat in summ:
            t = s.get("tf") or {}
            name = s["label"] if s.get("arm", "base") == "base" else \
                s.get("arm") or (MEGA_BF16Q_ARM if pat else MEGA_ARM)
            L.append(f"| {name}{'' if name == s['label'] else ' (' + s['label'] + ')'} | {s['layers']} | {s.get('fit')} | {_f(t.get('positions'))} | "
                     f"{_f(t.get('flips_pg61'))} ({_f(t.get('flips_w_pg61'))}) | {_f(t.get('flips_base'))} "
                     f"({_f(t.get('flips_w_base'))}) | {_f(t.get('max_abs_dlp_base'))} | {_f(s.get('free_parity'))} |"
                     f" {_f(s.get('free_vs_base_identical'))} |")
    if model:
        L.append("")
        L += summary_lines(model["summary"], "model (synthetic)")
    return L


def cmd_report(a) -> int:
    ld = lambda ps: [json.load(open(p)) for p in (ps or []) if p and os.path.exists(p)]  # noqa: E731
    layers, patched = ld(a.layers), ld(a.patched)
    ds = (ld([a.downstream]) or [None])[0]
    ds_patched = (ld([a.downstream_patched]) or [None])[0]
    model = (ld([a.model]) or [None])[0]
    for line in report_lines(layers, patched, ds, model, ds_patched):
        print(line)
    return 0


# ------------------------------------------------------------------------------------------------ cli
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="M3-N1 MegaMoE numerics probe")
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("model", help="synthetic decomposition (CPU ok, no checkpoint)")
    m.add_argument("--device", default="cpu")
    m.add_argument("--dim", type=int, default=DIM)
    m.add_argument("--inter", type=int, default=INTER)
    m.add_argument("--experts", type=int, default=22, help="synthetic experts per window (the 6x6 step's ~22)")
    m.add_argument("--windows", type=int, default=2)
    m.add_argument("--seed", type=int, default=20261107)
    m.add_argument("--out", default=None)
    d = sp.add_parser("downstream", help="box: full model, teacher-forced + free-run, MegaMoE on groups of layers")
    d.add_argument("--model-dir", default="/ref")
    d.add_argument("--map", default=os.path.join("docs", "maps", "hot-v26ar-seed-295.json"))
    d.add_argument("--fixture", default=os.path.join("w0", "results", "phase1", "parity-A.jsonl"))
    d.add_argument("--only", default=None)
    d.add_argument("--mode", choices=["static", "graph"], default="static",
                   help="static (default; the capture hooks need it; bit-identical to graph by the L1 gate) | graph")
    d.add_argument("--moe-impl", default="dg-inplace")
    d.add_argument("--fp8-impl", default="gemv")
    d.add_argument("--glue-impl", default="fused")
    d.add_argument("--attn-impl", default="flashmla")
    d.add_argument("--fp32-impl", default="fixed")
    d.add_argument("--moe-rowmatch", default="off", choices=["on", "align", "off"])
    d.add_argument("--max-pos", type=int, default=4095)
    d.add_argument("--max-new", type=int, default=256)
    d.add_argument("--reserve-host-gib", type=float, default=16.0)
    d.add_argument("--no-fill-uninit", action="store_true", default=True)
    d.add_argument("--layers", default=",".join(str(l) for l in LAYERS_DEFAULT))
    d.add_argument("--groups", default="auto",
                   help="auto (all layers in one arm, halved while the HBM check refuses) | each | all+each | "
                        "explicit 'a,b|c|d'")
    d.add_argument("--shared", default="gemv", choices=["gemv", "fused"])
    d.add_argument("--capture-layers", default=",".join(str(l) for l in LAYERS_DEFAULT))
    d.add_argument("--capture-steps", type=int, default=48)
    d.add_argument("--acts", default=os.path.join("scratch", "m3n1-acts.pt"))
    d.add_argument("--capture-only", action="store_true")
    d.add_argument("--free-run", action=argparse.BooleanOptionalAction, default=True)
    d.add_argument("--keep-tf", action="store_true", help="keep per-position TF records in the receipt")
    d.add_argument("--dg-path", default=None,
                   help="patch-dg's --out (the PATCHED kernel; arms are named mega_bf16q); needs --jit-cache")
    d.add_argument("--jit-cache", default=None, help="fresh JIT cache dir for the patched kernel")
    d.add_argument("--stem", default=None)
    d.add_argument("--out", default=None)
    y = sp.add_parser("layers", help="box: per-layer stage decomposition on fixture activations + speed")
    y.add_argument("--ckpt", default="/ref")
    y.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    y.add_argument("--acts", default=os.path.join("scratch", "m3n1-acts.pt"))
    y.add_argument("--layers", default=",".join(str(l) for l in LAYERS_DEFAULT))
    y.add_argument("--windows", type=int, default=8)
    y.add_argument("--win-start", type=int, default=1)
    y.add_argument("--spare", type=int, default=16)
    y.add_argument("--iters", type=int, default=200)
    y.add_argument("--dg-path", default=None, help="patch-dg's --out (the PATCHED kernel); needs --jit-cache")
    y.add_argument("--jit-cache", default=None)
    y.add_argument("--out", default=None)
    p = sp.add_parser("patch-dg", help="box: copy the installed deep_gemm with the bf16-before-quant patch")
    p.add_argument("--src", default=None, help="deep_gemm package dir (default: the installed one)")
    p.add_argument("--out", default=os.path.join("scratch", "dg-bf16q"))
    r = sp.add_parser("report", help="receipts -> markdown tables")
    r.add_argument("--layers", nargs="*", default=[])
    r.add_argument("--patched", nargs="*", default=[])
    r.add_argument("--downstream", default=None)
    r.add_argument("--downstream-patched", default=None,
                   help="downstream receipt of a run with --dg-path (mega_bf16q arms), shown beside --downstream")
    r.add_argument("--model", default=None)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"model": cmd_model, "downstream": cmd_downstream, "layers": cmd_layers, "patch-dg": cmd_patch_dg,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
