"""M2-T15 (docs/HANDOFF-M2-T15-ROUTER-PLAN.md; docs/M2-FUSED-CHAIN-DESIGN.md section 2.2 row L2): the router and the
R-row MoE plan in ONE launch per layer. Kernels: runtime/static_kernels.py `m2_router_rows` / `m2_gate_fp32` (section
"M2-T15"; round 1's `m2_router_plan` is kept for the bench).

    $DSV41_ROUTER_IMPL = fused (default) | plan | plan-gemv          (bench/router_plan_bench.py --router-impl)
    $DSV41_ROUTER_PLAN_LAUNCH = ticket (default) | split              (the R-row plan's launch form, M2-T15 fix)

  fused      today's path (T9b-2 R2 `route` / `route_rows` + M1-T10-MOE `dedup_fused`): the gate mm (cuBLAS fp32),
             F.softplus, t9b2r2_router_pre, torch.topk (gatherTopK + bitonicSort), t9b2r2_router_post, and at R rows
             m1_dedup_warp -- mm + 6 launches per layer at 6 rows, mm + 5 at 1 row.
  plan       the gate mm stays the same torch op on the same cached fp32 weight (so its bits are today's); ONE kernel
             does softplus, sqrt, + bias, the top-k (torch.topk's CUDA result, tie rule below), the weight
             normalisation (the calibrated sum order), the ascending sort and, at R rows, the plan (m1_dedup_warp
             called as a device function: the SAME code, so the same tensors). mm + 1 launch per layer.
             Numerics: bit-identical to `fused` when the build-time self-check passes (it gates the switch on the
             device: softplus probe, routing on random + tie-heavy scores vs torch.topk itself, plan tensors vs
             dedup_fused); a softplus probe mismatch keeps torch's F.softplus as a separate op (mm + 2).
  plan-gemv  the gate scores from our own GEMV (static_kernels.gate_dot = m2_gate_fp32: T9b-5's tuned fixed-order fp32
             dot on the bf16 weight, widened exactly; KB = gate_kb(D) lanes, a function of D only), then the same
             router kernel. 2 launches per layer (3 with F.softplus kept). Numerics: A3-class vs `fused` (cuBLAS's
             sum order is not ours, and the V0 census shows cuBLAS's M=6 gate differs from its M=1 gate on 84 % of
             elements -- docs/first-light-rows6-fast-20261107.json -- so no GEMV can be bit-equal to `fused` at both
             R); bit-equal to its own oracle (fused_glue_r2.fixed_dot_torch + the fused tail) and row-invariant by
             construction -- that is what the self-check gates; its route vs cuBLAS's is reported, not gated.

M2-T15 fix (docs/HANDOFF-M2-T15-FIX.md; round 1 had R=6 at 42.6 us vs fused 39.5: one CTA did all R rows and the plan
serially): the router kernel is now `m2_router_rows`, grid (R,), one program per row (the same per-row arithmetic --
bit-equal to round 1's single CTA by construction); the plan runs either in the LAST row program to arrive (`ticket`:
an acq_rel ticket, one launch) or as today's separate dedup launch (`split`: moe_rows.dedup_fused). The round-1 kernel
(m2_router_plan, with its in-kernel GEMV) stays for the bench's before / after only.

The 6-row plan reaches the MoE through a wrapper around moe_rows._RowsMoEBase._route (installed only when a glue
with a non-default router impl is built at R > 1; moe_rows.py itself is not edited -- it belongs to another worker
this round): when the plan is on and the dedup is the warp kernel, the wrapper returns the plan kernel's mapping
(the same dict `dedup_fused` returns, with the same buffers written); otherwise it calls the original.

THE TIE RULE (torch.topk on CUDA for a [R, 384] fp32 input, k = 6, sorted=True; ATen TensorTopK.cu + Sort.cu +
SortUtils.cuh + SortingRadixSelect.cuh, the latter two shipped as headers with every torch wheel):
  1. selection -- `launch_gather_topk_kernel` takes the single-block gatherTopK (sbtopk) at this size: the multi-block
     mbtopk is chosen by TensorTopK.cu's should_use_multiblock only for long slices (from our reading of the .cu,
     which the wheels do not ship: >= 20,000 elements at <= 20 slices; 384 x <= 8 slices is far below any branch),
     and topk's sort fallback (should_use_sort) only at >= 100,000 elements. radixSelect finds the k-th largest RADIX
     KEY (TopKTypeConfig<float>::convert: bits ^ 0x80000000
     for a clear sign bit, ~bits for a set one, 0xffffffff for any NaN -- so -0 < +0 and NaNs are the largest, all
     equal); gatherTopK writes every element with a key above it in ascending index order (block prefix scan), then
     "fills in the rest with actual == top-K values ... choose the first seen set": the LOWEST-index ties, also in
     ascending index order.
  2. order -- `sortKeyValueInplace(values, indices, dim, largest)` (stable = false, slice 6 <= 32) runs
     SmallBitonicSort: bitonicSortKVInPlace with sort size 32 (16 threads x 2 items), GTOp<float, handleNaN=true>
     (NaN above every number, NaNs equal, -0 == +0), bitonicSwap's `swap = (comp(kA, kB) && validA) || !validB;
     if (swap == dir) swap`, lanes >= 6 invalid (key 0.0, value 0). NOT stable: with ties among the 6 selected values
     the output order depends on the network (topk_cuda_torch below runs it; of the 6^6 = 46,656 value patterns of 6
     lanes, 6,988 come out stable, 15,392 reverse-stable, 24,276 neither -- tests/test_router_plan_cpu.py). Without a
     tie the result is the descending order. Rule used by the kernel: max-rounds for the set (step 1 exactly), the
     rank formula when the 6 selected values are pairwise distinct (GTOp keys), else the 32-lane network itself.
  Which ids run (step 1) and the weight sum's input order (step 2, the sum is order-sensitive) are both reproduced.
  The box decides: the self-check and tests/gpu/test_router_plan_gpu.py compare against torch.topk on the device
  with constructed ties (bench/router_plan_bench.py on 10^5 recorded inputs).

Softplus: ATen's CUDA softplus_kernel computes `(x * beta) > threshold ? x : log1p(exp(x * beta)) / beta` in fp32
with beta 1, threshold 20 (x * 1 and / 1 exact): the kernel uses libdevice log1p / exp; whether the device's
libdevice equals the CUDA math library torch was built with is the softplus probe (10^6 values + edges), at build
and in the GPU test. Round 1 on the box: 28,411 of 21M differ, every example an exp(x) below 2^-126 that Triton's
FTZ-linked libdevice flushes to +0 where torch keeps the subnormal. The fix adds two candidate forms for that range
only (SOFTPLUS_FORMS: exp in fp64, rounded once / via 24 bits); the probe picks the first exact form, else torch's
F.softplus stays a separate op.

Recording (the 10^5-input gate): $DSV41_ROUTER_DUMP=<dir> makes the FUSED path's `route` / `route_rows` (the
reference: today's ops) save every eager call's gate scores (raw [T, E] fp32, the mm's output), the layer, the call's
outputs (ids, w), the live weight-sum order and every 8th call's input row (x, bf16 values) to
<dir>/router-dump-<pid>-<n>.pt (+ the layers' biases once); never under graph capture (run first_light/headline with
--mode static, or the GPU test's mini model). bench/router_plan_bench.py --replay <dir> replays them.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROUTER_IMPL_ENV = "DSV41_ROUTER_IMPL"
ROUTER_IMPLS = ("fused", "plan", "plan-gemv")
ROUTER_IMPL_DEFAULT = "plan-gemv"      # until the box gate (HANDOFF-M2-T15)
PLAN_LAUNCH_ENV = "DSV41_ROUTER_PLAN_LAUNCH"
PLAN_LAUNCHES = ("ticket", "split")  # the R-row plan in the last row program / as today's separate dedup launch
DUMP_ENV = "DSV41_ROUTER_DUMP"
GATE_KB = 512                      # plan-gemv's fixed-order lanes (5120 = 10 tiles of 512; the mixes' KB)
SOFTPLUS_FORMS = {"torch": 0, "fused": 1, "fused-sub": 2, "fused-sub24": 3}   # -> the kernel's SOFTPLUS mode
NET = 32                           # SmallBitonicSort's sort size
EPS = 1e-20                        # model.py:827
FLT_MIN = np.float32(2.0 ** -126)
KERNELS_ON_CPU = False             # tests set it with SK = the shim-loaded static_kernels
SK = None                          # the kernel module (None: runtime.static_kernels, imported on first launch)


def router_impl() -> str:
    v = os.environ.get(ROUTER_IMPL_ENV, ROUTER_IMPL_DEFAULT) or ROUTER_IMPL_DEFAULT
    if v not in ROUTER_IMPLS:
        raise ValueError(f"{ROUTER_IMPL_ENV}={v!r} not in {ROUTER_IMPLS}")
    return v


def plan_launch() -> str:
    v = os.environ.get(PLAN_LAUNCH_ENV, "ticket") or "ticket"
    if v not in PLAN_LAUNCHES:
        raise ValueError(f"{PLAN_LAUNCH_ENV}={v!r} not in {PLAN_LAUNCHES}")
    return v


def gate_kb(D: int) -> int:
    """plan-gemv's KB at reduction width D: GATE_KB, or the largest power of two dividing D if smaller (mini
    checkpoints). A function of D only, never of R (fused_glue_r2.fixed_kb's rule)."""
    D = int(D)
    return max(1, min(GATE_KB, D & -D))


def _sk():
    global SK
    if SK is None:
        from runtime import static_kernels as _m

        SK = _m
    return SK


def _kern(t: torch.Tensor) -> bool:
    return t.is_cuda or KERNELS_ON_CPU


def _np2(n: int) -> int:
    return 1 if int(n) <= 1 else 1 << (int(n) - 1).bit_length()


# ------------------------------------------------------------------------------------------------ the tie rule (host)
def radix_key(v: np.ndarray) -> np.ndarray:
    """TopKTypeConfig<float>::convert as int64 (module docstring, step 1)."""
    v = np.ascontiguousarray(v, dtype=np.float32)
    b = v.view(np.uint32).astype(np.int64)
    k = np.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)
    return np.where(np.isnan(v), 0xFFFFFFFF, k)


def gt_key(v: np.ndarray) -> np.ndarray:
    """An int64 key whose `>` is GTOp<float, handleNaN=true> (module docstring, step 2)."""
    v = np.ascontiguousarray(v, dtype=np.float32)
    b = v.view(np.int32).astype(np.int64)
    k = np.where(b < 0, -(b & 0x7FFFFFFF), b)
    return np.where(np.isnan(v), 0x7FFFFFFF, k)


def bitonic_small(keys: np.ndarray, vals: np.ndarray, n: int, size: int = NET) -> np.ndarray:
    """SortUtils.cuh bitonicSort over `size` lanes (n valid, descending GTOp via gt_key keys), as written there:
    the lanes' values after the network (first n are the output)."""
    k = np.zeros(size, np.int64)
    v = np.zeros(size, np.int64)
    ok = np.zeros(size, bool)
    k[:n], v[:n], ok[:n] = keys[:n], vals[:n], True

    def stage(st: int, flag_of):
        for t in range(size // 2):
            p = 2 * t - (t & (st - 1))
            q = p + st
            sw = (bool(k[p] > k[q]) and bool(ok[p])) or (not ok[q])
            if sw == flag_of(t):
                k[p], k[q] = k[q], k[p]
                v[p], v[q] = v[q], v[p]
                ok[p], ok[q] = ok[q], ok[p]

    sz = 2
    while sz < size:
        st = sz // 2
        while st > 0:
            stage(st, lambda t, sz=sz: (t & (sz // 2)) != 0)
            st //= 2
        sz *= 2
    st = size // 2
    while st > 0:
        stage(st, lambda t: False)
        st //= 2
    return v


def topk_cuda_host(row: np.ndarray, k: int) -> np.ndarray:
    """torch.topk(row, k, sorted=True).indices on CUDA for a 1-D fp32 row (module docstring): the rule, in Python."""
    row = np.ascontiguousarray(row, dtype=np.float32)
    rk = radix_key(row)
    order = sorted(range(row.size), key=lambda i: (-int(rk[i]), i))
    kth = int(rk[order[k - 1]])
    above = [i for i in range(row.size) if rk[i] > kth]
    tied = [i for i in range(row.size) if rk[i] == kth][: k - len(above)]
    lay = np.array(above + tied, dtype=np.int64)                      # gatherTopK's output layout
    return bitonic_small(gt_key(row[lay]), lay, k)[:k]


def topk_cuda_torch(bsc: torch.Tensor, k: int) -> torch.Tensor:
    """topk_cuda_host per row of bsc [R, E] (or [E]) -> int64 indices [R, k] ([k]) on bsc's device."""
    one = bsc.dim() == 1
    a = bsc.detach().reshape(-1, bsc.size(-1)).float().cpu().numpy()
    out = np.stack([topk_cuda_host(r, k) for r in a])
    t = torch.from_numpy(out).to(bsc.device)
    return t.view(-1) if one else t


# ------------------------------------------------------------------------------------------------ mirrors
def softplus_np(x: torch.Tensor, mode: int = 1) -> torch.Tensor:
    """The kernel's softplus expression (SOFTPLUS mode 1 / 2 / 3) with numpy float32 exp / log1p and float64 exp (=
    tests/triton_shim.py's libdevice): the mirror the shim-run kernel body is compared against (NOT the device's bits;
    that is the softplus probe)."""
    a = x.detach().float().cpu().numpy()
    with np.errstate(over="ignore", under="ignore"):
        e = np.exp(a).astype(np.float32)
        lp = np.log1p(e).astype(np.float32)
        if int(mode) == 1:
            r = np.where(a > np.float32(20.0), a, lp)
        else:
            ed = np.exp(np.minimum(a, np.float32(-80.0)).astype(np.float64))
            if int(mode) == 2:
                es = ed.astype(np.float32)
            else:
                es = ((ed * 2.0 ** 64).astype(np.float32).astype(np.float64) * 2.0 ** -64).astype(np.float32)
            r = np.where(a > np.float32(20.0), a, np.where(e < FLT_MIN, es, lp))
    return torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32)).to(x.device)


def route_tail_torch(raw: torch.Tensor, bias: torch.Tensor, k: int, ordd: dict, rs: float, *, softplus=None,
                     topk=None, eps: float = EPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gate.forward after the mm (model.py:819-828) + the ascending sort, as the kernel computes it: raw [R, E] ->
    (ids [R, k] int64 ascending, w [R, k] fp32). softplus(raw) (default F.softplus), sqrt, + bias, topk(bsc, k)
    (default torch.topk on raw's device; pass topk_cuda_torch for the CUDA rule on the host), then
    fused_glue_r2.router_post_torch with the explicit order `ordd`."""
    from runtime.fused_glue_r2 import router_post_torch

    sp = (softplus or F.softplus)(raw.float())
    sc = sp.sqrt()
    bsc = sc + bias.float().view(1, -1)
    idx = topk(bsc, k) if topk is not None else bsc.topk(k, dim=-1)[1]
    return router_post_torch(idx.view(sc.size(0), k), sc.contiguous(), ordd, eps, rs)


# ------------------------------------------------------------------------------------------------ the plan's args
def dedup_args(buf, R_: int, k: int, block_rows: int, mm, ids_step, uval, inpl: Optional[dict], idx: torch.Tensor
               ) -> dict:
    """runtime/moe_rows._launch_warp's arguments to m1_dedup_warp (pointer args after idx / w, the int args from BR on,
    the constexprs), for the plan kernel's PLAN tail. Must stay in step with _launch_warp: the Mac test compares the
    plan kernel's outputs with dedup_fused's for every layout, and the build-time self-check does on the device."""
    from runtime import moe_rows as MR

    v2 = bool(inpl is not None and inpl.get("v2"))
    lay = inpl.get("layout", "contiguous") if v2 else "contiguous"
    mi = inpl.get("mi") if v2 else None
    has_in = inpl is not None
    dmy = inpl["dummy"] if has_in else idx
    tags_t, ts = (mi, int(block_rows)) if mi is not None else (buf.tags, R_)
    gl = inpl.get("gl") if v2 else None
    ovf = inpl.get("ovf") if v2 else None
    ptrs = (buf.uniq, buf.counts, buf.valid, buf.slot, buf.rank, buf.prow, buf.pw, buf.arow, buf.posmat,
            mm if mm is not None else buf.counts, ids_step if ids_step is not None else buf.uniq,
            uval if uval is not None else buf.valid, inpl["gtab"] if has_in else idx,
            buf.ids_stage, buf.sidx, tags_t, buf.wl, gl if gl is not None else buf.scr,
            ovf if ovf is not None else buf.scr, buf.scr, dmy.reshape(-1))
    ints = (int(block_rows), int(inpl["H"]) if has_in else 0, int(inpl["S"]) if has_in else 0, ts,
            int(inpl.get("G") or 0) if v2 else 0, int(inpl.get("gscr") or 0) if v2 else 0)
    cx = dict(DKP=buf.KP, NRP=_np2(R_), HAS_MM=mm is not None, HAS_CTR=ids_step is not None, INPL=has_in,
              HAS_DUMMY=bool(has_in and inpl["has_dummy"]), LAYOUT=MR.LAYOUT_IDS[lay] if v2 else 0,
              HAS_TAGS=has_in and (not v2 or lay == "contiguous"), HAS_WL=v2,
              STEP_SF=bool(v2 and inpl.get("step_sf")), HAS_OVF=ovf is not None)
    return {"ptrs": ptrs, "ints": ints, "cx": cx, "v2": v2, "lay": lay, "mi": mi}


def plan_out_dict(buf, mm, inpl: Optional[dict], a: dict) -> Dict[str, torch.Tensor]:
    """The dict moe_rows.dedup_fused returns for the same call (views of `buf`)."""
    out = {"uniq": buf.uniq, "counts": buf.counts, "valid": buf.valid, "slot": buf.slot, "rank": buf.rank,
           "prow": buf.prow, "pw": buf.pw, "arow": buf.arow, "posmat": buf.posmat, "mm_written": mm is not None}
    if inpl is not None:
        out.update(ids_stage=buf.ids_stage, sidx=buf.sidx)
        if a["mi"] is None:
            out["tags"] = buf.tags
    if a["v2"]:
        out.update(wl=buf.wl, layout_done=True)
    return out


# ------------------------------------------------------------------------------------------------ the moe_rows hook
_HOOK: Dict[str, object] = {"orig": None}


def install_rows_hook() -> None:
    """Wrap moe_rows._RowsMoEBase._route once (module docstring): the glue's RouterPlan answers when it can."""
    from runtime import moe_rows as MR

    if _HOOK["orig"] is not None and getattr(MR._RowsMoEBase._route, "_m2_t15", False):
        return
    orig = MR._RowsMoEBase._route

    def _route(self, x, block_rows, mm=None, inpl=None):
        g = self.owner.glue
        rp = getattr(g, "rplan", None) if g is not None else None
        if rp is not None:
            m = rp.route_plan_rows(self, x, block_rows, mm=mm, inpl=inpl)
            if m is not None:
                self.last = m
                return m
        return orig(self, x, block_rows, mm, inpl)

    _route._m2_t15 = True
    _route.__wrapped__ = orig
    _route.__doc__ = orig.__doc__
    MR._RowsMoEBase._route = _route
    _HOOK["orig"] = orig


def uninstall_rows_hook() -> None:
    from runtime import moe_rows as MR

    if _HOOK["orig"] is not None:
        MR._RowsMoEBase._route = _HOOK["orig"]
        _HOOK["orig"] = None


# ------------------------------------------------------------------------------------------------ recording
class _Recorder:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.n = 0
        self.part = 0
        self.rows: list = []
        self.bias_done: set = set()
        self.chunk = int(os.environ.get("DSV41_ROUTER_DUMP_CHUNK", "4096"))

    def add(self, layer: int, gate, raw: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, x: torch.Tensor,
            order=None) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if layer not in self.bias_done:
            torch.save({"layer": layer, "bias": gate.bias.detach().float().cpu(), "topk": gate.topk,
                        "route_scale": float(gate.route_scale), "gate_temp": float(gate.gate_temp)},
                       os.path.join(self.root, f"router-bias-{os.getpid()}-L{layer}.pt"))
            self.bias_done.add(layer)
        rec = {"layer": layer, "raw": raw.detach().float().cpu().clone(), "ids": ids.detach().cpu().clone(),
               "w": w.detach().float().cpu().clone()}
        if order is not None:          # the live weight-sum order (the replay re-runs the fused tail with it)
            rec["order"] = {"order": order.get("order"), "fin": order.get("fin", "mul"), "name": order.get("name")}
        if self.n % 8 == 0:
            rec["x"] = x.detach().to(torch.bfloat16).cpu().clone()
        self.rows.append(rec)
        self.n += 1
        if len(self.rows) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            torch.save(self.rows, os.path.join(self.root, f"router-dump-{os.getpid()}-{self.part:04d}.pt"))
            self.part += 1
            self.rows = []


_REC: Dict[str, Optional[_Recorder]] = {"r": None}


def recorder() -> Optional[_Recorder]:
    root = os.environ.get(DUMP_ENV)
    if not root:
        return None
    r = _REC["r"]
    if r is None or r.root != root:
        if r is not None:
            r.flush()
        r = _REC["r"] = _Recorder(root)
        import atexit

        atexit.register(r.flush)
    return r


def record(glue, moe, raw: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> None:
    """The dump hook (module docstring): called by fused_glue_r2.route / route_rows when $DSV41_ROUTER_DUMP is set."""
    r = recorder()
    if r is None:
        return
    layer = next((sm.layer for sm in glue.o.moes if sm.moe is moe), -1)
    T_ = raw.numel() // raw.size(-1)
    r.add(layer, moe.gate, raw.reshape(T_, -1), ids.reshape(T_, -1), w.reshape(T_, -1), x.reshape(T_, -1),
          order=getattr(glue, "orders", {}).get("router"))


# ------------------------------------------------------------------------------------------------ the switch
class RouterPlan:
    """The plan impl of one FusedGlue (glue.rplan): workspaces per gate, the self-check, the call sites."""

    def __init__(self, glue, impl: str):
        self.g = glue
        self.impl = impl
        self.status = "on"
        self.softplus = "fused"          # a SOFTPLUS_FORMS key: "fused*": in the kernel; "torch": F.softplus kept
        self.rows_plan = "on"            # the R-row plan with the launch (R > 1)
        self.launch_form = plan_launch()  # "ticket": the plan in the last row program; "split": + dedup_fused
        self.checks: Dict[str, object] = {}
        self.verified = False            # the device self-check passed (CUDA launches need it; CPU: the fused path)
        self._checking = False
        self.ws: Dict[int, dict] = {}
        o = glue.o
        dev = glue.dev
        R_ = max(1, int(glue.rows))
        for sm in o.moes:
            gate = sm.moe.gate
            E = int(gate.weight.size(0))
            k = int(gate.topk)
            f32 = dict(dtype=torch.float32, device=dev)
            ws = {"R": R_, "E": E, "k": k, "sc": torch.zeros(R_, E, **f32), "bsc": torch.zeros(R_, E, **f32),
                  "tk": torch.zeros(R_, k, dtype=torch.long, device=dev),
                  "net": torch.zeros(_np2(R_), NET, dtype=torch.long, device=dev),
                  "ticket": torch.zeros(1, dtype=torch.int32, device=dev)}
            if impl == "plan-gemv":
                ws["raw"] = torch.zeros(R_, E, **f32)
            self.ws[id(gate)] = ws
        why = self._applicable()
        if why:
            self.off(why)
        if R_ > 1 and self.on():
            install_rows_hook()

    # ---------------------------------------------------------------- status
    def on(self) -> bool:
        return self.status == "on"

    def off(self, why: str) -> None:
        if self.on():
            self.status = f"off: {why}"
            self.g.log(f"[router-plan] {self.impl} OFF: {why}")

    def _applicable(self) -> Optional[str]:
        g = self.g
        if not g.on("router"):
            return "the router cluster is off (the plan replaces its kernels)"
        gates = [sm.moe.gate for sm in g.o.moes]
        if not gates:
            return "no MoE layers"
        if any(gt.topk > NET or gt.weight.size(0) >= 65536 for gt in gates):
            return "topk > 32 or >= 65536 experts (the kernel's packing)"
        if int(g.rows) > 8:
            return "more than 8 rows"
        if self.impl == "plan-gemv":
            if any(gt.gate_temp != 1.0 for gt in gates):
                return "gate_temp != 1 (torch's scalar-division form is not reproduced in the GEMV)"
        return None

    def describe(self) -> dict:
        D = int(self.g.o.moes[0].moe.gate.weight.size(1)) if self.g.o.moes else 0
        return {"impl": self.impl, "status": self.status, "softplus": self.softplus, "rows_plan": self.rows_plan,
                "launch": self.launch_form, "kernel": "m2_router_rows (grid R)",
                "tie_rule": "gatherTopK set (radix key, lowest index) + SmallBitonicSort(32) order",
                "gate": ("cuBLAS F.linear on the cached fp32 weight (today's op)" if self.impl == "plan" else
                         f"m2_gate_fp32 (T9b-5 fixed-order fp32 dot), KB = {gate_kb(D)} (A3 vs cuBLAS)"),
                "launches_per_layer": self.launches_per_layer(),
                "checks": dict(self.checks), "hook": _HOOK["orig"] is not None}

    def launches_per_layer(self, rows: Optional[int] = None) -> str:
        """The launches of one layer's router (+ plan) under this configuration (for reports)."""
        R_ = int(self.g.rows) if rows is None else int(rows)
        n = 1 + (self.softplus == "torch") + (R_ > 1 and self.launch_form == "split" and self.rows_plan == "on")
        return f"mm + {n}" if self.impl == "plan" else str(n + 1)

    # ---------------------------------------------------------------- launches
    def _ready(self, gate, x: torch.Tensor, T_: int) -> Optional[dict]:
        if not self.on() or not _kern(x) or not (self.verified or self._checking):
            return None
        ws = self.ws.get(id(gate))
        if ws is None or T_ > ws["R"]:
            return None
        return ws

    def _launch(self, gate, xf2: Optional[torch.Tensor], T_: int, ids: torch.Tensor, w: torch.Tensor, plan=None,
                raw_in: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> None:
        """The gate scores of the T_ rows xf2 [T_, D] fp32 (plan: today's F.linear; plan-gemv: gate_dot into
        ws["raw"]) -- or raw_in [T_, E], the scores themselves (the self-check / bench entry) -- then (softplus
        "torch": F.softplus) m2_router_rows; plan = dedup_args -> the plan in its last row program."""
        from runtime.fused_glue_r2 import order_args

        g = self.g
        ws = self.ws[id(gate)]
        E, k = ws["E"], ws["k"]
        b = gate.bias if bias is None else bias
        if raw_in is not None:
            raw = raw_in
        elif self.impl == "plan-gemv":
            assert xf2 is not None
            raw = ws["raw"][:T_]
            _sk().gate_dot(xf2, gate.weight, raw, kb=gate_kb(xf2.size(-1)))
        else:
            assert xf2 is not None
            raw = F.linear(xf2, g.gate_w[id(gate)])
            if gate.gate_temp != 1.0:          # as route(): x / 1.0 skipped; else torch's own op
                raw = raw / gate.gate_temp
        if self.softplus == "torch":
            raw = F.softplus(raw)
        _sk().router_rows_launch(raw.contiguous(), b, ws, ids, w, R=T_, E=E, K=k, order=order_args(g.orders["router"]),
                                 rs=float(gate.route_scale), eps=EPS, softplus=SOFTPLUS_FORMS[self.softplus],
                                 plan=plan)

    def route(self, moe, x: torch.Tensor, rows: bool):
        """FusedGlue.route / route_rows under the plan impl: (ids, w) as theirs, or None (-> the fused path)."""
        gate = moe.gate
        D = x.size(-1)
        T_ = x.numel() // D
        ws = self._ready(gate, x, T_)
        if ws is None:
            return None
        ctx = self.g.ffn_pre(x)
        xf = ctx["xf"] if ctx is not None and ctx.get("xf") is not None else x.float()
        ids = torch.empty(T_, ws["k"], dtype=torch.long, device=x.device)
        w = torch.empty(T_, ws["k"], dtype=torch.float32, device=x.device)
        self._launch(gate, xf.reshape(T_, D), T_, ids, w)
        if not rows:
            return ids.view(-1), w.view(-1)
        return ids, w

    def plan_rows_ready(self, sm, x: torch.Tensor) -> bool:
        from runtime import moe_rows as MR

        g = self.g
        if self.rows_plan != "on" or not (g.on("router") and g.on("dedup")):
            return False
        D = x.size(-1)
        T_ = x.numel() // D
        if self._ready(sm.moe.gate, x, T_) is None or T_ < 2:
            return False
        P = T_ * int(sm.moe.gate.topk)
        if P != sm.dbuf.P or P > MR.DEDUP_MAX_PAIRS:
            return False
        return MR.dedup_impl() == "warp" and MR._WARP["error"] is None and (MR.HAVE_TRITON or KERNELS_ON_CPU)

    def route_plan_rows(self, sm, x: torch.Tensor, block_rows: int, mm=None, inpl: Optional[dict] = None,
                        raw_in: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
                        buf=None, ids_step=None, uval=None):
        """moe_rows._RowsMoEBase._route under the plan impl: the gate + route + dedup -> dedup_fused's dict (None: not
        applicable here -> the original _route). launch form "ticket": the plan in the router kernel's last row
        program (dedup_args); "split": the router kernel, then today's dedup launch (moe_rows.dedup_fused) on its
        ids / w. buf / ids_step / uval / raw_in / bias: the self-check's private copies."""
        from runtime import moe_rows as MR

        if not self.plan_rows_ready(sm, x):
            return None
        o = sm.owner
        gate = sm.moe.gate
        D = x.size(-1)
        T_ = x.numel() // D
        k = int(gate.topk)
        buf = sm.dbuf if buf is None else buf
        if ids_step is None and o.counters:
            ids_step, uval = o.ids_step[sm.layer], o.uvalid_step[sm.layer]
        ids = torch.empty(T_, k, dtype=torch.long, device=x.device)
        w = torch.empty(T_, k, dtype=torch.float32, device=x.device)
        if raw_in is None:
            ctx = self.g.ffn_pre(x)
            xf = ctx["xf"] if ctx is not None and ctx.get("xf") is not None else x.float()
            xf2 = xf.reshape(T_, D)
        else:
            xf2 = None
        if self.launch_form == "split":
            self._launch(gate, xf2, T_, ids, w, raw_in=raw_in, bias=bias)
            return MR.dedup_fused(buf, ids, w, block_rows, mm=mm, ids_step=ids_step, uval=uval, inpl=inpl,
                                  impl="warp")
        a = dedup_args(buf, T_, k, block_rows, mm, ids_step, uval, inpl, ids)
        if a["v2"]:
            MR._check_arow_range(inpl, block_rows, T_ * k)
        self._launch(gate, xf2, T_, ids, w, plan=a, raw_in=raw_in, bias=bias)
        return plan_out_dict(buf, mm, inpl, a)

    # ---------------------------------------------------------------- self-check (build time, eager, on device)
    def _fused_tail(self, gate, raw: torch.Tensor, bias: torch.Tensor):
        """route()'s / route_rows()'s ops after the mm, exactly (F.softplus, router_pre, torch.topk, router_post)."""
        from runtime.fused_glue_r2 import launch_router_post, launch_router_pre

        g = self.g
        T_, E = raw.shape
        dev = raw.device
        cuda = dev.type == "cuda"
        # on the device: the fused path's own ops (F.softplus, torch.topk); under the Mac shim (no CUDA torch.topk,
        # numpy libdevice): the kernel's documented mirrors (softplus_np of the chosen form when the kernel applies
        # softplus, and the tie rule's host emulation topk_cuda_torch)
        sp = (F.softplus(raw) if cuda or self.softplus == "torch" else
              softplus_np(raw, SOFTPLUS_FORMS[self.softplus])).contiguous()
        topk = (lambda t, k: t.topk(k, dim=-1)[1]) if cuda else topk_cuda_torch
        if T_ == 1:
            sc = torch.empty(E, dtype=torch.float32, device=dev)
            bsc = torch.empty(1, E, dtype=torch.float32, device=dev)
            launch_router_pre(sp.view(-1), bias, sc, bsc.view(-1))
            idx = topk(bsc, gate.topk).view(1, -1)
            ids = torch.empty(gate.topk, dtype=torch.long, device=dev)
            w = torch.empty(gate.topk, dtype=torch.float32, device=dev)
            launch_router_post(idx.view(-1), sc, ids, w, ordd=g.orders["router"], eps=EPS,
                               rs=float(gate.route_scale))
            return ids.view(1, -1), w.view(1, -1)
        sc = torch.empty(T_, E, dtype=torch.float32, device=dev)
        bsc = torch.empty(T_, E, dtype=torch.float32, device=dev)
        launch_router_pre(sp.view(-1), bias, sc.view(-1), bsc.view(-1))
        idx = topk(bsc, gate.topk)
        ids = torch.empty(T_, gate.topk, dtype=torch.long, device=dev)
        w = torch.empty(T_, gate.topk, dtype=torch.float32, device=dev)
        launch_router_post(idx.contiguous(), sc, ids, w, ordd=g.orders["router"], eps=EPS, rs=float(gate.route_scale))
        return ids, w

    @staticmethod
    def raw_draws(gen: torch.Generator, T_: int, E: int, n: int):
        """Self-check inputs [T_, E] fp32 (CPU, explicit dtypes: the runtime's default dtype is bf16): realistic
        scores, tie-heavy ones (few distinct raw values, a two-valued bias with exact zeros), "mixed" top-k (distinct
        values, ties, ties straddling the k-th place), all-equal rows, rows repeated across the step (heavy dedup),
        scores past the softplus threshold and deep in its underflow (sp = +0)."""
        f32 = dict(device="cpu", dtype=torch.float32)
        out = []
        for t in range(n):
            kind = t % 6
            if kind == 0:
                raw = torch.randn(T_, E, generator=gen, **f32) * 2.0
                bias = torch.randn(E, generator=gen, **f32) * 0.05
            elif kind == 1:
                raw = torch.randint(-8, 9, (T_, E), generator=gen, device="cpu").to(torch.float32) * 0.25
                bias = torch.where(torch.rand(E, generator=gen, **f32) < 0.5, torch.zeros(E, **f32),
                                   torch.full((E,), 0.25, **f32))
            elif kind == 2:
                raw = torch.full((T_, E), 0.5, **f32)
                raw[:, ::7] = 1.25
                bias = torch.zeros(E, **f32)
            elif kind == 3:
                raw = torch.randn(1, E, generator=gen, **f32).repeat(T_, 1) * 3.0
                raw[T_ // 2:] += torch.randn(T_ - T_ // 2, E, generator=gen, **f32) * 0.01
                bias = torch.randn(E, generator=gen, **f32) * 0.05
            elif kind == 4:
                raw = torch.full((T_, E), -5.0, **f32)
                pat = torch.tensor([3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5], **f32)
                m = min(E, pat.numel())
                for r in range(T_):
                    raw[r, torch.randperm(E, generator=gen, device="cpu")[:m]] = \
                        pat[:m][torch.randperm(m, generator=gen, device="cpu")]
                bias = torch.zeros(E, **f32)
            else:
                raw = torch.randn(T_, E, generator=gen, **f32) * 30.0
                raw[:, ::5] = -200.0
                bias = torch.where(torch.rand(E, generator=gen, **f32) < 0.5, torch.full((E,), -0.0, **f32),
                                   torch.zeros(E, **f32))
            out.append((raw.contiguous(), bias.contiguous()))
        return out

    def calibrate(self, gen: torch.Generator) -> None:
        """Softplus probe, routing == the fused path (and torch.topk) on the device, the R-row plan == dedup_fused.
        Any mismatch: the impl off (softplus: kept as a torch op instead), with the reason."""
        if not self.on():
            return
        g = self.g
        dev = g.dev
        if not (dev.type == "cuda" or KERNELS_ON_CPU):
            self.checks["device"] = "cpu: the plan impl runs the fused path (its CPU mirror); kernels unchecked here"
            return
        self._checking = True
        try:
            self._check_softplus(gen)
            if self.on():
                self._check_route(gen)
            if self.on() and g.rows > 1:
                self._check_plan_rows(gen)
        except Exception as e:  # a kernel that does not compile/run here: the impl off, with the reason
            self.off(f"self-check raised {type(e).__name__}: {str(e)[:300]}")
        finally:
            self._checking = False
        self.verified = self.on()

    def _check_softplus(self, gen) -> None:
        """The in-kernel softplus forms (SOFTPLUS_FORMS 1, 2, 3, in that order) vs F.softplus on the probe values (on
        the shim: vs each form's numpy mirror); the first exact form is used, none -> F.softplus as a torch op (both
        impls: plan-gemv keeps its GEMV and adds the op). NaN vs NaN counts as equal (payloads are not a contract)."""
        dev = self.g.dev
        x = softplus_probe_inputs(gen, 1 << 20).to(dev)
        counts: Dict[str, int] = {}
        chosen = None
        for form in ("fused", "fused-sub", "fused-sub24"):
            mode = SOFTPLUS_FORMS[form]
            got = _sk().softplus_probe(x, mode)
            ref = F.softplus(x) if dev.type == "cuda" else softplus_np(x, mode)   # shim: the body vs its mirror
            bad = (got.view(torch.int32) != ref.view(torch.int32)) & ~(torch.isnan(got) & torch.isnan(ref))
            counts[form] = int(bad.sum())
            if counts[form] == 0:
                chosen = form
                break
        self.checks["softplus"] = {"n": int(x.numel()), "mismatch": counts[chosen or "fused"], "forms": counts,
                                   "form": chosen or "torch"}
        self.softplus = chosen or "torch"
        if chosen is None:
            self.g.log(f"[router-plan] softplus: every in-kernel form differs from F.softplus ({counts} of "
                       f"{x.numel()} values) -> F.softplus kept as a torch op")
        elif chosen != "fused":
            self.g.log(f"[router-plan] softplus: in-kernel form {chosen!r} exact ({counts})")

    def _check_route(self, gen) -> None:
        from runtime.fused_glue_r2 import fixed_dot_torch

        g = self.g
        dev = g.dev
        Rr = max(1, int(g.rows))
        n_eq = 0
        for sm in g.o.moes[:2]:
            gate = sm.moe.gate
            E, D = gate.weight.shape
            ws = self.ws[id(gate)]
            for T_ in sorted({1, Rr}):
                # the router kernel on given scores (both impls run the same kernel after their gate): == the fused tail
                for raw, bias in self.raw_draws(gen, T_, E, 12):
                    raw, bias = raw.to(dev), bias.to(dev)
                    want = self._fused_tail(gate, raw, bias)
                    ids = torch.empty(T_, ws["k"], dtype=torch.long, device=dev)
                    w = torch.empty(T_, ws["k"], dtype=torch.float32, device=dev)
                    self._launch(gate, None, T_, ids, w, raw_in=raw, bias=bias)
                    if dev.type == "cuda":
                        ref_i = torch.sort((F.softplus(raw).sqrt() + bias).topk(gate.topk, dim=-1)[1], -1)[0]
                        if not torch.equal(ref_i, want[0]):
                            self.off("the fused path's ids are not torch.topk's (internal)")
                            return
                    if not (torch.equal(ids, want[0]) and torch.equal(w.view(torch.int32),
                                                                       want[1].view(torch.int32))):
                        self.off(f"route differs from the fused path (layer {sm.layer}, {T_} rows)")
                        return
                    n_eq += 1
                # the full launch from x (both impls): plan -> the fused route on the same x; plan-gemv -> its oracle
                for t in range(3):
                    x = (torch.randn(T_, D, generator=gen, device="cpu", dtype=torch.float32) * 0.5).to(
                        torch.bfloat16).float().to(dev)
                    ids = torch.empty(T_, ws["k"], dtype=torch.long, device=dev)
                    w = torch.empty(T_, ws["k"], dtype=torch.float32, device=dev)
                    self._launch(gate, x, T_, ids, w)
                    if self.impl == "plan":
                        raw = F.linear(x, g.gate_w[id(gate)])
                        if gate.gate_temp != 1.0:
                            raw = raw / gate.gate_temp
                    else:
                        raw = fixed_dot_torch(x, gate.weight, gate_kb(D))
                        if not torch.equal(ws["raw"][:T_].view(torch.int32), raw.view(torch.int32)):
                            self.off(f"gate GEMV differs from its fixed-order oracle (layer {sm.layer}, {T_} rows)")
                            return
                    want = self._fused_tail(gate, raw, gate.bias)
                    if not (torch.equal(ids, want[0]) and torch.equal(w.view(torch.int32), want[1].view(torch.int32))):
                        self.off(f"launch from x differs from {'the fused path' if self.impl == 'plan' else 'its oracle'}"
                                 f" (layer {sm.layer}, {T_} rows)")
                        return
                    n_eq += 1
        self.checks["route"] = f"equal on {n_eq} draws"

    def _check_plan_rows(self, gen) -> None:
        from runtime import moe_rows as MR

        g = self.g
        if not g.on("dedup"):
            self.rows_plan = "off: the dedup cluster is off"
            return
        o = g.o
        dev = g.dev
        T_ = int(g.rows)
        n_eq = 0
        for sm in o.moes[:2]:
            if not hasattr(sm, "dbuf"):
                self.rows_plan = "off: the R-row MoE has no dedup buffers"
                return
            gate = sm.moe.gate
            E = gate.weight.size(0)
            k = int(gate.topk)
            P = T_ * k
            kw = sm._route_kw() if hasattr(sm, "_route_kw") else {}
            br = sm._block_rows() if hasattr(sm, "_block_rows") else T_
            x = torch.zeros(T_, gate.weight.size(1), dtype=torch.float32, device=dev)
            if not self.plan_rows_ready(sm, x):
                self.rows_plan = "off: the dedup is not the warp kernel here (moe_rows.dedup_impl / fallback)"
                return
            for raw, bias in self.raw_draws(gen, T_, E, 12):
                raw, bias = raw.to(dev), bias.to(dev)
                res = []
                for mine in (False, True):
                    inpl = kw.get("inpl")
                    if inpl is not None and inpl.get("v2"):      # private m_indices / gl / counter (live ones stay)
                        lay = inpl.get("layout", "contiguous")
                        inpl = dict(inpl, mi=torch.full((P * br,), -1, dtype=torch.int32, device=dev)
                                    if lay == "contiguous" else None,
                                    gl=torch.zeros(max(int(inpl.get("G") or 0), 1), dtype=torch.int32, device=dev),
                                    ovf=torch.zeros(1, dtype=torch.int32, device=dev))
                    mm = torch.full((P,), -7, dtype=torch.int32, device=dev) if "mm" in kw else None
                    ids_s = torch.full((P,), -7, dtype=torch.long, device=dev)
                    uv = torch.full((P,), -7, dtype=torch.long, device=dev)
                    buf = MR.DedupBuffers(T_, k, dev)
                    if mine:
                        m = self.route_plan_rows(sm, x, br, mm=mm, inpl=inpl, raw_in=raw, bias=bias, buf=buf,
                                                 ids_step=ids_s, uval=uv)
                    else:
                        ids, w = self._fused_tail(gate, raw, bias)
                        m = MR.dedup_fused(buf, ids, w, br, mm=mm, ids_step=ids_s, uval=uv, inpl=inpl)
                    res.append((m, buf, mm, ids_s, uv, inpl))
                (m0, b0, mm0, i0, u0, p0), (m1, b1, mm1, i1, u1, p1) = res
                if sorted(m0) != sorted(m1):
                    self.rows_plan = f"off: plan dict keys {sorted(m1)} != dedup_fused's {sorted(m0)}"
                    return
                bad = [key for key in m0 if isinstance(m0[key], torch.Tensor) and not _bits_eq(m0[key], m1[key])]
                bad += [n for n, a_, b_ in (("mm", mm0, mm1), ("ids_step", i0, i1), ("uvalid", u0, u1))
                        if a_ is not None and not _bits_eq(a_, b_)]
                if p0 is not None and p0.get("v2"):
                    bad += [n for n in ("mi", "gl", "ovf") if p0.get(n) is not None and not _bits_eq(p0[n], p1[n])]
                if bad:
                    self.rows_plan = f"off: plan tensors {bad} differ from dedup_fused's (layer {sm.layer})"
                    return
                n_eq += 1
            # the whole launch from x (plan-gemv: GEMV + ticket + plan): ids / w vs the route check's reference
            for t in range(2):
                xr = (torch.randn(T_, gate.weight.size(1), generator=gen, device="cpu", dtype=torch.float32)
                      * 0.5).to(torch.bfloat16)
                xr = xr.float().to(dev)
                buf = MR.DedupBuffers(T_, k, dev)
                inpl = kw.get("inpl")
                if inpl is not None and inpl.get("v2"):
                    lay = inpl.get("layout", "contiguous")
                    inpl = dict(inpl, mi=torch.full((P * br,), -1, dtype=torch.int32, device=dev)
                                if lay == "contiguous" else None,
                                gl=torch.zeros(max(int(inpl.get("G") or 0), 1), dtype=torch.int32, device=dev),
                                ovf=torch.zeros(1, dtype=torch.int32, device=dev))
                mm = torch.full((P,), -7, dtype=torch.int32, device=dev) if "mm" in kw else None
                ids_s = torch.full((P,), -7, dtype=torch.long, device=dev)
                uv = torch.full((P,), -7, dtype=torch.long, device=dev)
                saved = self.g._ffn_ctx
                self.g._ffn_ctx = None
                try:
                    m = self.route_plan_rows(sm, xr, br, mm=mm, inpl=inpl, buf=buf, ids_step=ids_s, uval=uv)
                finally:
                    self.g._ffn_ctx = saved
                ids2 = torch.empty(T_, k, dtype=torch.long, device=dev)
                w2 = torch.empty(T_, k, dtype=torch.float32, device=dev)
                self._launch(gate, xr, T_, ids2, w2)
                ref = MR.dedup_routes(ids2, w2, br)
                if not all(torch.equal(m[key], ref[key]) for key in ("uniq", "counts", "slot", "rank", "prow")) \
                        or not _bits_eq(m["pw"], ref["pw"]):
                    self.rows_plan = f"off: the launch from x's plan differs from its own route (layer {sm.layer})"
                    return
                n_eq += 1
        self.checks["plan_rows"] = f"equal on {n_eq} draws"


def _bits_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        n = a.element_size()
        it = {2: torch.int16, 4: torch.int32, 8: torch.int64}[n]
        return torch.equal(a.contiguous().view(it), b.contiguous().view(it))
    return torch.equal(a, b)


def softplus_probe_inputs(gen: torch.Generator, n: int) -> torch.Tensor:
    """n fp32 values (CPU) for the softplus probe: N(0, s) at scales 2^-20 .. 2^7, uniform [-30, 30], a dense grid
    around the threshold (20 +- 2^-10 steps), 19.99 .. 20.01, large negatives (exp underflow), +-0, +-inf, NaN,
    denormals."""
    parts = []
    m = n // 4
    s = torch.exp2(torch.randint(-20, 8, (m,), generator=gen, device="cpu").float())
    parts.append(torch.randn(m, generator=gen, device="cpu", dtype=torch.float32) * s)
    parts.append(torch.rand(m, generator=gen, device="cpu", dtype=torch.float32) * 60.0 - 30.0)
    parts.append(20.0 + (torch.arange(m, device="cpu", dtype=torch.float32) - m / 2) * 2.0 ** -10)
    edge = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 20.0, 19.999998, 20.000002,
                         -87.3, -88.8, -103.9, -104.0, -200.0, 1e-45, -1e-45, 1.17e-38, 88.7, 89.0],
                        dtype=torch.float32, device="cpu")
    rest = n - 3 * m - edge.numel()
    parts.append(torch.randn(rest, generator=gen, device="cpu", dtype=torch.float32) * 8.0 - 4.0)
    parts.append(edge)
    return torch.cat(parts).contiguous()


def describe_default() -> dict:
    return {"impl": "fused", "env": ROUTER_IMPL_ENV, "impls": list(ROUTER_IMPLS)}
