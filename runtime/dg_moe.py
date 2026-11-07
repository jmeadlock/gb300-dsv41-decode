"""T9a-1: the routed experts of the static MoE through DeepGEMM's masked grouped GEMM (`--moe-impl dg-masked`).

    st = StaticStep(model, pool, max_pos=4095, moe_impl="dg-masked")   # everything else as runtime/static_step.py

Spec: docs/HANDOFF-T9A1-DG-MOE.md; ABI and numerics measured in docs/T9A-DEEPGEMM-ABI-20261106.md with
bench/dg_masked_spike.py (the call below is that spike's call, copied, not re-derived); stage-by-stage numerics in
docs/T8-T9A-DESIGN.md section 3.1. `DGMaskedMoE` has `StaticMoE`'s interface and runs the same code up to and including
the pointer-table gather into the static [k,N,K/2] staging (runtime/static_step.py). From there, per layer:

  (a) `act_quant(x, 32, "ue8m0")` ONCE (model.py:187; the reference computes the same bytes for w1 and w3 of every
      expert, 2k times) -> row 0 of every group of the shared static activation buffer xq1[k, 128, DIM] fp8 /
      xs1[k, 128, DIM/32] fp32 (rows 1.. stay zero / one forever);
  (b) masked_m = ones(k) int32 (static), expected_m = 8;
  (c) w1 and w3: `m_grouped_fp8_fp4_gemm_nt_masked((xq1, xs1), (W_int8, S_f32), d, masked_m, 8, recipe_a=(1, 32),
      recipe_b=(1, 32))` over staging[("w1"|"w3", "weight")] (the checkpoint's packed e2m1 bytes, as-is) and the
      fp32 scale buffers -> d[k, 128, INTER] bf16 -> row 0 -> the reference glue of Expert.forward (model.py:845-853):
      `.float()` (the GEMM's bf16 rounding of w1.x / w3.x is kept), clamp(up, +-limit), clamp(gate, max=limit),
      silu(gate) * up, weights * that (fp32), `.to(bf16)`;
  (d) `act_quant` of the glue output ([k, INTER], per row = the reference's per-expert call) -> xq2/xs2 row 0 -> the
      w2 masked call -> d[k, 128, DIM] -> row 0 -> fp32 `y += out[j]` in slot order (= ascending expert id,
      model.py:897-902) -> + shared expert (the reference module, unchanged; model.py:905) -> `.type_as(x)`.

Scale conversion e8m0 -> fp32 (= exp2(e8m0 - 127), exact; 0 -> 2^-127, 255 -> NaN) is a device op recorded in the
graph: the gathered staging scales of each projection (one launch per projection) and the activation scales. The
method is chosen once at construction by an exhaustive check of all 256 codes on the device against an exact table
(`E8M0_F32`): "cast" = torch's own e8m0 -> fp32 conversion (`copy_` from the float8_e8m0fnu view, one launch), else
"exp2" = the spike's `exp2(uint8 - 127)` formula; neither exact -> refuse to build.

What breaks bit-identity vs `ref`, by design: the block-scaled UMMA accumulation inside DeepGEMM (measured: ~1 bf16 ulp
on 0.01-0.04 % of outputs with the model's per-32 scales; 0 with coinciding block scales). Everything around it is the
reference's arithmetic in the reference's order.

deep_gemm is imported from vllm's `third_party` directory with vllm itself never imported (charter decision 3):
`import_deep_gemm()` puts that directory on sys.path only for the import, then asserts no `vllm` module is loaded.
DeepGEMM JIT-compiles on first call (persist it: DG_JIT_CACHE_DIR, runtime/README.md); `StaticStep.capture()` calls
`DGBuffers.warmup()` before its warm-up steps so nothing compiles under capture.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

DG_THIRD_PARTY = "/usr/local/lib/python3.12/dist-packages/vllm/third_party"
MAX_M = 128          # DeepGEMM masked kernels: max_m >= BLOCK_M = 128 (64 / 8 silently corrupt groups >= 1; ABI memo)
EXPECTED_M = 8       # expected_m of the masked call (the spike's value at M = 1 and 6)
ACT_BLOCK = 32       # per-32 activation scales (the model's act_quant, model.py:187) -> recipe_a = (1, 32)
WEIGHT_BLOCK = 32    # per-32 e8m0 weight scales (fp4_block_size, model.py:28) -> recipe_b = (1, 32)
RECIPE_A = (1, ACT_BLOCK)
RECIPE_B = (1, WEIGHT_BLOCK)
PROJS = ("w1", "w3", "w2")
MOE_IMPLS = ("ref", "dg-masked")


# ------------------------------------------------------------------------------------------------ e8m0 -> fp32
def e8m0_table() -> torch.Tensor:
    """The exact fp32 value of every e8m0 code (OCP MX): 2^(e-127) for e = 0..254 (e = 0 is the fp32 subnormal
    2^-127, exactly representable), NaN for 255. Built from math.ldexp (exact in float64, exact in fp32)."""
    return torch.tensor([math.ldexp(1.0, e - 127) for e in range(255)] + [float("nan")], dtype=torch.float32)


E8M0_F32 = e8m0_table()


def _u8(src: torch.Tensor) -> torch.Tensor:
    return src if src.dtype == torch.uint8 else src.view(torch.uint8)


def e8m0_to_f32(src: torch.Tensor, out: torch.Tensor, method: str = "cast") -> torch.Tensor:
    """out <- fp32 value of the e8m0 codes in `src` (uint8 or float8_e8m0fnu storage; `out` may be a strided view and
    `src` may broadcast to it). method "cast": one conversion copy; "exp2": the spike's formula
    exp2(uint8 - 127) (dg_masked_spike.py:52, 55)."""
    if method == "cast":
        return out.copy_(_u8(src).view(torch.float8_e8m0fnu))
    if method == "exp2":
        return out.copy_(torch.exp2(_u8(src).to(torch.int32).float() - 127.0))
    raise ValueError(f"unknown e8m0 conversion method {method!r}")


def _bits_equal_table(got: torch.Tensor, n: int = 255) -> bool:
    """got[:255] bit-equal to the exact table, got[255] non-finite (NaN by the cast, inf by the spike's exp2 formula:
    an e8m0 255 is a NaN scale, never a real one, and either value poisons whatever it scales)."""
    want = E8M0_F32.to(got.device)
    return bool(torch.equal(got[:n].view(torch.int32), want[:n].view(torch.int32))) and not bool(torch.isfinite(got[255]))


def pick_e8m0_method(device) -> str:
    """Exhaustive check of all 256 codes on `device` (build time; syncs): "cast" if exact, else "exp2" if exact,
    else raise."""
    codes = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
    errs = {}
    for m in ("cast", "exp2"):
        out = torch.empty(256, dtype=torch.float32, device=device)
        try:
            e8m0_to_f32(codes, out, m)
            if _bits_equal_table(out):
                return m
            errs[m] = "inexact"
        except Exception as e:  # pragma: no cover - depends on the torch build
            errs[m] = repr(e)[:200]
    raise RuntimeError(f"no exact e8m0 -> fp32 device conversion on {device}: {errs}")


# ------------------------------------------------------------------------------------------------ deep_gemm import
def _vllm_loaded():
    return sorted(m for m in sys.modules if m == "vllm" or m.startswith("vllm."))


def import_deep_gemm(path: Optional[str] = None):
    """`import deep_gemm` from vllm's third_party directory WITHOUT importing vllm (design section 3; the spike's
    sys.path trick, dg_masked_spike.py:9-15). The directory is on sys.path only for the import. An already-imported
    `deep_gemm` (sys.modules) is returned as is. Refuses if any vllm module is loaded before or after."""
    bad = _vllm_loaded()
    if bad:
        raise RuntimeError(f"vllm is imported ({bad[:3]}...): charter decision 3 forbids it")
    mod = sys.modules.get("deep_gemm")
    if mod is not None:
        return mod
    path = path or os.environ.get("DSV41_DEEP_GEMM_PATH", DG_THIRD_PARTY)
    if not os.path.isdir(os.path.join(path, "deep_gemm")):
        raise ImportError(f"no deep_gemm package under {path} (set DSV41_DEEP_GEMM_PATH)")
    sys.path.insert(0, path)
    try:
        import deep_gemm as mod  # noqa: F811
    finally:
        try:
            sys.path.remove(path)
        except ValueError:  # pragma: no cover
            pass
    bad = _vllm_loaded()
    if bad:
        raise RuntimeError(f"importing deep_gemm pulled in vllm ({bad[:3]}...)")
    return mod


# ------------------------------------------------------------------------------------------------ static buffers
class DGBuffers:
    """The static buffers of the dg-masked MoE, allocated once per StaticStep and shared by every layer (layers run
    one after another inside one step, as the staging buffers are shared): fp32 weight-scale buffers per projection,
    the two activation buffers (w1/w3 input, w2 input), the three outputs, masked_m. All rows >= 1 of the activation
    buffers are written only here (zeros / ones) and never again."""

    def __init__(self, owner, shapes: Dict[str, Tuple[int, int]], sshapes: Dict[str, Tuple[int, int]], *,
                 swiglu_limit: float, deterministic: bool = True, dg=None):
        R, d, k = owner.R, owner.dev, owner.topk
        if (R.fp8_block_size, R.fp4_block_size, R.scale_fmt) != (ACT_BLOCK, WEIGHT_BLOCK, "ue8m0"):
            raise ValueError(f"dg-masked assumes per-32 ue8m0 activation and weight scales; the reference has "
                             f"fp8_block_size={R.fp8_block_size} fp4_block_size={R.fp4_block_size} {R.scale_fmt}")
        self.owner, self.k, self.dev = owner, k, d
        self.N = {p: shapes[p][0] for p in PROJS}
        self.K = {p: shapes[p][1] * 2 for p in PROJS}  # [N, K/2] packed e2m1
        for p in PROJS:
            if tuple(sshapes[p]) != (self.N[p], self.K[p] // WEIGHT_BLOCK):
                raise ValueError(f"{p}: scale shape {sshapes[p]} != (N, K/32) = ({self.N[p]}, {self.K[p] // 32})")
        if not (self.K["w1"] == self.K["w3"] == self.N["w2"] and self.N["w1"] == self.N["w3"] == self.K["w2"]):
            raise ValueError(f"w1/w3/w2 shapes do not chain: N {self.N} K {self.K}")
        self.dim, self.inter = self.K["w1"], self.N["w1"]
        self.swiglu_limit = float(swiglu_limit)
        self.dg = dg if dg is not None else import_deep_gemm()
        self.fn = self.dg.m_grouped_fp8_fp4_gemm_nt_masked
        self.deterministic = self._set_deterministic(deterministic)
        self.e8m0_method = pick_e8m0_method(d)
        f32, bf16, fp8 = torch.float32, torch.bfloat16, torch.float8_e4m3fn
        self.masked_m = torch.ones(k, dtype=torch.int32, device=d)
        self.expected_m = EXPECTED_M
        # activation buffers: [k, MAX_M, K] fp8 (row 0 = the token, rows 1.. zero), [k, MAX_M, K/32] fp32 scales
        self.xq = {1: torch.zeros(k, MAX_M, self.dim, dtype=fp8, device=d),
                   2: torch.zeros(k, MAX_M, self.inter, dtype=fp8, device=d)}
        self.xs = {1: torch.ones(k, MAX_M, self.dim // ACT_BLOCK, dtype=f32, device=d),
                   2: torch.ones(k, MAX_M, self.inter // ACT_BLOCK, dtype=f32, device=d)}
        self.sf32 = {p: torch.zeros(k, self.N[p], self.K[p] // WEIGHT_BLOCK, dtype=f32, device=d) for p in PROJS}
        self.d = {p: torch.zeros(k, MAX_M, self.N[p], dtype=bf16, device=d) for p in PROJS}
        self.n_calls = 0

    def _set_deterministic(self, on: bool):
        """deep_gemm.use_deterministic_algorithms (design section 4: self-replay with it on). Returns what happened."""
        f = getattr(self.dg, "use_deterministic_algorithms", None)
        if not on or f is None:
            return {"requested": bool(on), "applied": False, "note": None if on else "off"}
        try:
            f(True)
            return {"requested": True, "applied": True, "note": None}
        except TypeError:
            try:
                f()
                return {"requested": True, "applied": True, "note": "called without argument"}
            except Exception as e:  # pragma: no cover
                return {"requested": True, "applied": False, "note": repr(e)[:200]}
        except Exception as e:  # pragma: no cover
            return {"requested": True, "applied": False, "note": repr(e)[:200]}

    # ---------------------------------------------------------------- per-step ops (graph-capturable)
    def convert_scales(self, p: str, staged_scale: torch.Tensor) -> torch.Tensor:
        """sf32[p] <- fp32 of the gathered e8m0 scales staging[(p, "scale")] [k, N, K/32]: one device op."""
        return e8m0_to_f32(staged_scale, self.sf32[p], self.e8m0_method)

    def load_act(self, which: int, q: torch.Tensor, s: torch.Tensor) -> None:
        """Row 0 of every group <- the quantized activation: q [1 or k, K] fp8, s [1 or k, K/32] e8m0 (broadcast
        over the k groups when it is one row)."""
        self.xq[which][:, 0].copy_(q)
        e8m0_to_f32(s, self.xs[which][:, 0], self.e8m0_method)

    def gemm(self, p: str, weight: torch.Tensor, which: int) -> torch.Tensor:
        """The spike's call (dg_masked_spike.py:59): d[p] <- (xq, xs) x (W int8 [k,N,K/2], S fp32 [k,N,K/32])^T for
        the masked_m = 1 valid row of each group. Returns d[p] [k, MAX_M, N] bf16 (only row 0 is meaningful)."""
        self.fn((self.xq[which], self.xs[which]), (weight, self.sf32[p]), self.d[p], self.masked_m, self.expected_m,
                recipe_a=RECIPE_A, recipe_b=RECIPE_B)
        self.n_calls += 1
        return self.d[p]

    def warmup(self) -> None:
        """First-call JIT of both GEMM shapes (w1 = w3 shape, w2 shape) on the static buffers, before any capture."""
        st = self.owner.staging
        for p in PROJS:
            self.gemm(p, st[(p, "weight")], 2 if p == "w2" else 1)

    def describe(self) -> dict:
        return {"impl": "dg-masked", "deep_gemm": getattr(self.dg, "__file__", None),
                "deep_gemm_version": getattr(self.dg, "__version__", None), "max_m": MAX_M,
                "expected_m": self.expected_m, "recipe_a": list(RECIPE_A), "recipe_b": list(RECIPE_B),
                "e8m0_to_f32": self.e8m0_method, "deterministic": self.deterministic, "k": self.k,
                "dim": self.dim, "inter": self.inter,
                "buffer_bytes": sum(t.numel() * t.element_size() for t in
                                    [*self.xq.values(), *self.xs.values(), *self.sf32.values(), *self.d.values()])}


# ------------------------------------------------------------------------------------------------ the MoE
class DGMaskedMoE:
    """One layer's MoE with the routed GEMMs on DeepGEMM (module docstring). Same interface as StaticMoE: called by the
    patched `MoE.forward` (runtime/patches.py `static_moe`) only while its owner runs a step."""

    def __init__(self, owner, moe, layer: int, ptabs: Dict[Tuple[str, str], torch.Tensor]):
        self.owner, self.moe, self.layer, self.ptabs = owner, moe, layer, ptabs

    @property
    def active(self) -> bool:
        return self.owner.active

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from runtime.static_kernels import gather_rows

        o, moe, b = self.owner, self.moe, self.owner.dg
        R = o.R
        shape = x.size()
        x = x.view(-1, moe.dim)                                        # model.py:893
        weights, indices = moe.gate(x, None)                           # model.py:894 (text: no image mask)
        ids, order = torch.sort(indices[0])                            # ascending id = model.py:897 loop order
        w = weights[0, order]
        if o.counters:
            o.ids_step[self.layer].copy_(ids)
        for key, tab in self.ptabs.items():                            # the same gather as StaticMoE
            gather_rows(tab, ids, o.staging[key], o.row_bytes[key])
        for p in PROJS:                                                # e8m0 -> fp32, one op per projection
            b.convert_scales(p, o.staging[(p, "scale")])
        q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)       # model.py:187, once for w1 and w3
        b.load_act(1, q, s)
        gate = b.gemm("w1", o.staging[("w1", "weight")], 1)[:, 0].float()        # model.py:845
        up = b.gemm("w3", o.staging[("w3", "weight")], 1)[:, 0].float()          # model.py:846
        lim = b.swiglu_limit
        if lim > 0:                                                               # model.py:847-849
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up                                                     # model.py:850
        h = w.view(-1, 1) * h                                                     # model.py:852 (weights [1,1] each)
        h = h.to(x.dtype)                                                         # model.py:853 x.to(dtype)
        q2, s2 = R.act_quant(h, R.fp8_block_size, R.scale_fmt, R.scale_dtype)     # model.py:187 inside w2
        b.load_act(2, q2, s2)
        out = b.gemm("w2", o.staging[("w2", "weight")], 2)[:, 0]                 # [k, DIM] bf16
        y = torch.zeros_like(x, dtype=torch.float32)                              # model.py:895
        for j in range(b.k):                                                      # slot order = ascending id
            y += out[j]                                                           # model.py:902 (fp32 += bf16)
        y += moe.shared_experts(x)                                                # model.py:905
        return y.type_as(x).view(shape)                                           # model.py:906
