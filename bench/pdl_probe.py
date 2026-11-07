#!/usr/bin/env python3
"""pdl_probe.py -- M2-L5 probe: does PDL between our Triton kernels work inside a torch-captured CUDA graph on this
box, is it bit-identical, and what does it save per owned edge?  (docs/HANDOFF-M2-L5-PDL-PROBE.md; W1 sec 1-3)

A chain of 40 (GEMV -> glue) pairs on real shapes, 6 rows, real data dependencies:
    pair i:  pdl_gemv   (bench/pdl_kernels.py, a copy of runtime t9b_mxfp8_gemv; W [N, K] fp8 + 32x32 e8m0 scales,
                         fixed-order split-K partials -> ws)
          -> pdl_norm_quant (a copy of runtime t9b2r2_norm_row, q-path form; folds the split-K reduce in its load,
                         RMSNorm, act_quant) writing the quantized row into the first N columns of the NEXT pair's
                         [R, K_next] activation buffer (the other columns are a static pre-quantized tail: in the
                         step they are other branches' activations).
    chains:  mixed = A,B,A,B,... (A = 1280 x 5120, B = 5120 x 8192; 20 of each; B -> A is an exact K match),
             A = 40 x 1280x5120,  B = 40 x 5120x8192.   Every pair has its own weight copy (HBM-streamed, > L2) and
             an L2 flush runs before each timed sample.
    variants (the same two kernels, only timing differs):
             a   plain                    launch_pdl=False, no gdc_* instruction
             b   PDL                      launch_pdl=True, gdc_wait() + gdc_launch_dependents() at kernel entry
             c   PDL + weight prefetch    b + prefetch.global.L2 of the GEMV's weight/scale slice BEFORE gdc_wait
             c2  PDL + bulk prefetch      b + cp.async.bulk.prefetch.L2.global per weight row before gdc_wait
    modes:   graph  one torch.cuda.CUDAGraph per variant (keep_graph=True so the captured graph can be walked),
                    replayed --replays times (cuda events, one sample per --batch replays, variants interleaved and
                    rotated per sample so clock drift cancels in the paired differences)
             eager  the same 80 launches issued eagerly: the launch-overhead baseline (GPU event time AND host time)
    checks:  (1) bit-equality of the whole intermediate state across variants, graph replay (arena compare), over
             --stress replays with a NEW input per replay (the TRT-LLM #20634 method), plus eager vs graph;
             (2) the captured graph's node count and edge types via cudart (cudaGraphGetEdges{,_v2}: default vs
             programmatic edges, ports) -- proves PDL attached under capture (a: 0 programmatic, b/c: kernels-1);
             (3) a variant that fails to compile / capture reports the exact error text and the run continues.
Numbers: mean / std / p50 / p95 / min per replay, us per pair (/40), paired differences vs a with a 95 % CI, and
us per owned edge (a chain of P pairs has 2P kernels and 2P-1 edges).

Box command (James runs it; no ssh/docker/network from the worker; GPU idle, the :30006 lane stopped or another GPU):
  W=$HOME/hermes/gb300-dsv41-decode-wtl5
  docker run --rm --device nvidia.com/gpu=<GPU-UUID> --ipc=host -v "$W":/work -w /work \\
    -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton --entrypoint python3 vllm/vllm-openai:v0.31.0 \\
    bench/pdl_probe.py --mode both --json scratch/pdl-probe-box.json [--gemv-cache scratch/gemv-tune.json]
Run it twice if the first run compiled kernels at load (the script warms up every variant eagerly before capture).
Mac: `.venv/bin/python bench/pdl_probe.py --plan` prints the plan; tests/test_pdl_probe_cpu.py runs both kernels'
bodies through tests/triton_shim.py. Modeled numbers are labelled MODEL; everything else printed is measured.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
VARIANTS = {
    "a": dict(pdl=False, prefetch=0, label="plain graph edges"),
    "b": dict(pdl=True, prefetch=0, label="PDL: launch_pdl + gdc_wait/launch_dependents at entry"),
    "c": dict(pdl=True, prefetch=1, label="PDL + prefetch.global.L2 of weight slice before gdc_wait"),
    "c2": dict(pdl=True, prefetch=2, label="PDL + cp.async.bulk.prefetch.L2 per weight row before gdc_wait"),
}
FLUSH_BYTES = 512 << 20       # > the ~126 MB L2
ALIGN = 256


def load_kernels():
    """bench/pdl_kernels.py (imported lazily: the Mac test loads its own copy under tests/triton_shim.py)."""
    from bench import pdl_kernels  # noqa: PLC0415

    return pdl_kernels


# ------------------------------------------------------------------------------------------------ pure helpers
def pctl(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]) of an ascending list."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


def summarize(vals: Sequence[float]) -> dict:
    """n / mean / std (ddof 1) / sem / min / p50 / p95 / max of a list of samples."""
    n = len(vals)
    if n == 0:
        return {"n": 0}
    v = sorted(float(x) for x in vals)
    mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "sem": std / math.sqrt(n) if n > 1 else 0.0, "min": v[0],
            "p50": pctl(v, 50), "p95": pctl(v, 95), "max": v[-1]}


def paired_diff(a: Sequence[float], b: Sequence[float]) -> dict:
    """Per-sample paired difference a - b (a slower than b -> positive): mean, sem, 95 % CI half width (1.96 sem),
    and the fraction of samples with a > b."""
    n = min(len(a), len(b))
    d = [float(a[i]) - float(b[i]) for i in range(n)]
    s = summarize(d)
    if not n:
        return {"n": 0}
    return {"n": n, "mean": s["mean"], "std": s["std"], "sem": s["sem"], "ci95": 1.96 * s["sem"],
            "frac_a_slower": sum(1 for x in d if x > 0) / n}


def chain_specs(kind: str, pairs: int, shapes: Dict[str, Tuple[int, int]]) -> List[Tuple[str, int, int]]:
    """[(shape name, N, K)] for a chain kind: 'mixed' (A,B alternating), a single shape name, or a comma list. Each
    pair's N must fit in the next pair's K (its output is written into that activation buffer's first N columns)."""
    if kind == "mixed":
        names = ["A" if i % 2 == 0 else "B" for i in range(pairs)]
    elif kind in shapes:
        names = [kind] * pairs
    else:
        names = [t.strip() for t in kind.split(",")]
        if not all(t in shapes for t in names):
            raise ValueError(f"chain kind {kind!r}: not mixed / {sorted(shapes)} / a comma list of those")
    specs = [(nm,) + tuple(shapes[nm]) for nm in names]
    for i in range(len(specs) - 1):
        if specs[i][1] > specs[i + 1][2]:
            raise ValueError(f"chain pair {i} ({specs[i][0]}): N = {specs[i][1]} does not fit K = {specs[i + 1][2]}")
    return specs


def parse_cfg(text: str):
    """'32,4,4,1,0,0' -> Cfg(block_n, ku, num_warps, num_stages, split_k, mma)."""
    K = load_kernels()
    v = [int(t) for t in text.split(",")]
    if len(v) != 6:
        raise ValueError(f"cfg {text!r}: want block_n,ku,num_warps,num_stages,split_k,mma")
    return K.Cfg(*v)


def load_cfg_cache(path: Optional[str]) -> dict:
    """runtime.gemv_mxfp8's tune cache ({"NxK/mB": {block_n, ku, num_warps, num_stages, split_k, mma}}) -> {key: Cfg}
    (read-only parse; unknown keys / fields ignored)."""
    if not path:
        return {}
    K = load_kernels()
    with open(path) as f:
        raw = json.load(f)
    raw = raw.get("configs", raw) if isinstance(raw, dict) else {}
    out = {}
    for k, v in raw.items():
        try:
            out[k] = K.Cfg(**{f: int(v[f]) for f in ("block_n", "ku", "num_warps", "num_stages", "split_k", "mma")})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def pick_cfgs(K, specs, R: int, sms: int, user: Optional[dict] = None, cache: Optional[dict] = None) -> list:
    """The GEMV config of each pair: the tune cache's entry for (N, K, block_m(R)), else the per-shape user override,
    else the runtime's untuned DEFAULT_CONFIG; split resolved for `sms`."""
    out = []
    for nm, N, Kd in specs:
        c = (cache or {}).get(f"{N}x{Kd}/m{K.block_m(R)}") or (user or {}).get(nm) or K.DEFAULT_CFG
        out.append(K.resolve(c, N, Kd, sms))
    return out


def edge_label(edge_type: int, from_port: int, to_port: int) -> str:
    """cudaGraphEdgeData -> a name. type 0 default / 1 programmatic; from_port 0 default, 1 programmatic (stream
    capture of a PDL launch), 2 launch completion."""
    if edge_type == 0:
        return "default" if from_port == 0 else f"default(from_port={from_port})"
    if edge_type == 1:
        return {1: "programmatic", 2: "programmatic(launch_completion)"}.get(from_port, f"programmatic(from_port={from_port})")
    return f"type{edge_type}(from_port={from_port},to_port={to_port})"


def tally_edges(edges: Sequence[Tuple[int, int, int]]) -> Dict[str, int]:
    """[(type, from_port, to_port)] -> {label: count}."""
    out: Dict[str, int] = {}
    for t, fp, tp in edges:
        k = edge_label(int(t), int(fp), int(tp))
        out[k] = out.get(k, 0) + 1
    return out


def scale_code_table(device: str) -> torch.Tensor:
    """uint8 [65536]: the e8m0 code the reference act_quant (block 32, ue8m0) gives a block whose amax has bf16 bits i
    (runtime.fused_glue.scale_code_table, from the formula: 2^ceil(log2(max(amax, 1e-4) / 448)))."""
    v = torch.arange(65536, dtype=torch.int32, device="cpu").to(torch.int16).view(torch.bfloat16).float()
    v = torch.where(torch.isfinite(v), v.abs(), torch.zeros_like(v))
    e = torch.ceil(torch.log2(v.clamp_min(1e-4) / 448.0))
    return (e + 127).clamp(0, 254).to(torch.uint8).to(device)


# ------------------------------------------------------------------------------------------------ chain data
def exact_fp8(shape, g: torch.Generator) -> torch.Tensor:
    """e4m3 with exponent field 6..8 (|v| in [0.5, 4)): finite, bounded sums."""
    r = torch.randint(0, 256, tuple(shape), generator=g, dtype=torch.uint8, device="cpu")
    return (((r & 0x80) | (0x30 + r % 24)).to(torch.uint8)).view(FP8)


def e8_codes(shape, lo: int, hi: int, g: torch.Generator) -> torch.Tensor:
    return torch.randint(lo, hi, tuple(shape), generator=g, dtype=torch.uint8, device="cpu").view(E8)


def to_dev(t: torch.Tensor, device: str) -> torch.Tensor:
    """Host -> device for the 1-byte float dtypes through their uint8 bytes (no dtype-conversion path involved)."""
    if t.element_size() == 1 and t.dtype != torch.uint8:
        return t.view(torch.uint8).to(device).view(t.dtype)
    return t.to(device)


class Weights:
    """Per-pair weights (fp8 [N, K] + e8m0 [N/32, K/32] + bf16 norm weight [N]); one random CPU template per shape,
    cloned per pair on the device (distinct HBM allocations, so a chain streams len(specs) x weight bytes)."""

    def __init__(self, specs, device: str, seed: int = 20261107):
        g = torch.Generator(device="cpu").manual_seed(seed)
        tmpl: Dict[Tuple[int, int], tuple] = {}
        self.w, self.s, self.nw = [], [], []
        for nm, N, Kd in specs:
            if (N, Kd) not in tmpl:
                tmpl[(N, Kd)] = (to_dev(exact_fp8((N, Kd), g), device), to_dev(e8_codes((N // 32, Kd // 32), 124, 131, g), device),
                                 (torch.rand(N, generator=g, dtype=torch.float32, device="cpu") + 0.5).to(torch.bfloat16).to(device))
            w0, s0, n0 = tmpl[(N, Kd)]
            self.w.append(w0.clone())
            self.s.append(s0.clone())
            self.nw.append(n0.clone())
        self.bytes = sum(w.numel() + s.numel() for w, s in zip(self.w, self.s))


class Inputs:
    """`count` first-pair activations (fp8 [R, K0] + e8m0 [R, K0/32]) and the static tails of every later pair's
    activation buffer (columns N_{i-1}..K_i), all CPU tensors."""

    def __init__(self, specs, R: int, count: int, seed: int = 7):
        g = torch.Generator(device="cpu").manual_seed(seed)
        K0 = specs[0][2]
        self.x = [(exact_fp8((R, K0), g), e8_codes((R, K0 // 32), 124, 131, g)) for _ in range(count)]
        self.tail = [None]
        for i in range(1, len(specs)):
            n_prev, k = specs[i - 1][1], specs[i][2]
            self.tail.append((exact_fp8((R, k - n_prev), g), e8_codes((R, (k - n_prev) // 32), 124, 131, g)))


class State:
    """All mutable buffers of one variant's chain. arena=True carves every tensor out of ONE uint8 arena (so a whole
    state compares with one kernel on the box); arena=False allocates separately (the CPU shim test: the shim writes
    back whole storages, which must not alias across dtypes)."""

    def __init__(self, specs, R: int, device: str, arena: bool = True):
        self.specs, self.R, self.device = specs, R, device
        P = len(specs)
        shapes = []  # (name, shape, dtype)
        for i, (_, N, Kd) in enumerate(specs):
            shapes += [(f"xq{i}", (R, Kd), FP8), (f"xs{i}", (R, Kd // 32), E8), (f"gout{i}", (R, N), torch.bfloat16),
                       (f"ob{i}", (R, N), torch.bfloat16)]
        Nl = specs[-1][1]
        shapes += [("qf", (R, Nl), FP8), ("sf", (R, Nl // 32), E8)]

        def nbytes(sh, dt):
            n = 1
            for d in sh:
                n *= d
            return n * (torch.empty(0, dtype=dt, device="cpu").element_size())

        total = sum(-(-nbytes(sh, dt) // ALIGN) * ALIGN for _, sh, dt in shapes)
        self.arena = torch.zeros(total, dtype=torch.uint8, device=device) if arena else None
        off = 0
        self.t: Dict[str, torch.Tensor] = {}
        for name, sh, dt in shapes:
            nb = nbytes(sh, dt)
            if arena:
                self.t[name] = self.arena[off:off + nb].view(dt).view(*sh)
                off += -(-nb // ALIGN) * ALIGN
            else:
                self.t[name] = torch.zeros(*sh, dtype=dt, device=device)
        self.ws: Dict[tuple, torch.Tensor] = {}
        self.plans: list = []
        self.P = P

    def workspace(self, nsplit: int, N: int) -> torch.Tensor:
        k = (self.R, N, nsplit)
        if k not in self.ws:
            self.ws[k] = torch.zeros(nsplit, self.R, N, dtype=torch.float32, device=self.device)
        return self.ws[k]

    def set_static(self, inputs: Inputs) -> None:
        for i in range(1, self.P):
            n_prev = self.specs[i - 1][1]
            tq, ts = inputs.tail[i]
            self.t[f"xq{i}"][:, n_prev:].view(torch.uint8).copy_(tq.view(torch.uint8))
            self.t[f"xs{i}"][:, n_prev // 32:].view(torch.uint8).copy_(ts.view(torch.uint8))

    def set_input(self, xq: torch.Tensor, xs: torch.Tensor) -> None:
        self.t["xq0"].view(torch.uint8).copy_(xq.view(torch.uint8))
        self.t["xs0"].view(torch.uint8).copy_(xs.view(torch.uint8))

    def mismatch(self, other: "State") -> torch.Tensor:
        """Number of differing BYTES between two states (device int64 scalar; no host sync)."""
        if self.arena is not None:
            return (self.arena != other.arena).sum()
        tot = torch.zeros((), dtype=torch.int64, device=self.device)
        for k, v in self.t.items():
            tot += (v.contiguous().view(torch.uint8) != other.t[k].contiguous().view(torch.uint8)).sum()
        return tot

    def snapshot(self) -> "State":
        c = State.__new__(State)
        c.specs, c.R, c.device, c.P = self.specs, self.R, self.device, self.P
        c.arena = self.arena.clone() if self.arena is not None else None
        c.t = {k: v.clone() for k, v in self.t.items()} if self.arena is None else {}
        return c


def build_plans(K, st: State, cfgs: list) -> None:
    st.plans = []
    for (nm, N, Kd), cfg in zip(st.specs, cfgs):
        p = K.gemv_plan(cfg, st.R, N, Kd)
        st.plans.append(p)
        if p["SPLIT"]:
            st.workspace(p["NSPLIT"], N)


def run_chain(K, st: State, wt: Weights, cfgs: list, tab: torch.Tensor, *, pdl: bool, prefetch: int) -> None:
    """The 2 x P launches of one chain replay (graph-capturable: no allocation, no host sync)."""
    R = st.R
    for i, (nm, N, Kd) in enumerate(st.specs):
        p = st.plans[i]
        xq, xs = st.t[f"xq{i}"], st.t[f"xs{i}"]
        gout = st.t[f"gout{i}"]
        ws = st.ws[(R, N, p["NSPLIT"])] if p["SPLIT"] else gout
        K.launch_gemv(xq, xs, wt.w[i], wt.s[i], gout, ws, cfgs[i], pdl=pdl, prefetch=prefetch)
        if i + 1 < st.P:
            q, s = st.t[f"xq{i + 1}"], st.t[f"xs{i + 1}"]
        else:
            q, s = st.t["qf"], st.t["sf"]
        if p["SPLIT"]:
            src, ns, xst = ws, p["NSPLIT"], R * N
        else:
            src, ns, xst = gout, 0, 0
        K.launch_norm_quant(src, wt.nw[i], st.t[f"ob{i}"], q, s, tab, N=N, R=R, ns=ns, xs_stride=xst,
                            qrs=q.stride(0), srs=s.stride(0), pdl=pdl)


# ------------------------------------------------------------------------------------------------ cudart graph census
def _cudart():
    import ctypes

    cands: List[str] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart" in line:
                    p = line.split()[-1]
                    if p not in cands:
                        cands.append(p)
    except OSError:
        pass
    cands += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    err = []
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError as e:
            err.append(str(e))
    raise RuntimeError("no libcudart: " + "; ".join(err[:3]))


def graph_census(g) -> dict:
    """Node count and edge types of a CUDAGraph made with keep_graph=True, via cudart (ctypes). {'error': text} if the
    walk fails (the probe's timing and bit checks do not depend on it)."""
    import ctypes as C

    try:
        raw = int(g.raw_cuda_graph())
        lib = _cudart()
        vp, st = C.c_void_p, C.c_size_t

        class EdgeData(C.Structure):
            _fields_ = [("from_port", C.c_ubyte), ("to_port", C.c_ubyte), ("type", C.c_ubyte),
                        ("reserved", C.c_ubyte * 5)]

        def chk(rc, what):
            if rc != 0:
                lib.cudaGetErrorString.restype = C.c_char_p
                raise RuntimeError(f"{what}: cudart error {rc}: {lib.cudaGetErrorString(rc).decode()}")

        ver = C.c_int(0)
        lib.cudaRuntimeGetVersion(C.byref(ver))
        n = st(0)
        lib.cudaGraphGetNodes.argtypes = [vp, C.c_void_p, C.POINTER(st)]
        chk(lib.cudaGraphGetNodes(vp(raw), None, C.byref(n)), "cudaGraphGetNodes(count)")
        nodes = (vp * max(1, n.value))()
        chk(lib.cudaGraphGetNodes(vp(raw), nodes, C.byref(n)), "cudaGraphGetNodes")
        lib.cudaGraphNodeGetType.argtypes = [vp, C.POINTER(C.c_int)]
        types: Dict[int, int] = {}
        for i in range(n.value):
            t = C.c_int(-1)
            chk(lib.cudaGraphNodeGetType(nodes[i], C.byref(t)), "cudaGraphNodeGetType")
            types[t.value] = types.get(t.value, 0) + 1
        fn = getattr(lib, "cudaGraphGetEdges_v2", None)
        used = "cudaGraphGetEdges_v2"
        if fn is None:
            fn, used = getattr(lib, "cudaGraphGetEdges", None), "cudaGraphGetEdges"
            if fn is None or ver.value < 13000:
                raise RuntimeError(f"no 5-argument cudaGraphGetEdges in this cudart (version {ver.value})")
        fn.argtypes = [vp, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(st)]
        ne = st(0)
        chk(fn(vp(raw), None, None, None, C.byref(ne)), used + "(count)")
        m = max(1, ne.value)
        frm, to, ed = (vp * m)(), (vp * m)(), (EdgeData * m)()
        chk(fn(vp(raw), frm, to, ed, C.byref(ne)), used)
        edges = [(ed[i].type, ed[i].from_port, ed[i].to_port) for i in range(ne.value)]
        return {"cudart_version": ver.value, "api": used, "nodes": n.value,
                "kernel_nodes": types.get(0, 0), "node_types": {str(k): v for k, v in sorted(types.items())},
                "edges": ne.value, "edge_kinds": tally_edges(edges)}
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------ capture / timing
def capture_graph(fn, keep_graph: bool = True, dot_path: Optional[str] = None):
    """Warm `fn` on a side stream, then capture it. Returns (graph, kept). keep_graph=True needs torch >= 2.13's
    CUDAGraph(keep_graph=...) + instantiate(); an older torch falls back to a plain graph (no census)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    kept = False
    g = None
    if keep_graph:
        try:
            g = torch.cuda.CUDAGraph(keep_graph=True)
            kept = True
        except TypeError:
            g = None
    if g is None:
        g = torch.cuda.CUDAGraph()
    if dot_path and hasattr(g, "enable_debug_mode"):
        g.enable_debug_mode()
    with torch.cuda.graph(g):
        fn()
    if kept:
        g.instantiate()
    if dot_path and hasattr(g, "debug_dump"):
        try:
            g.debug_dump(dot_path)
        except Exception:  # noqa: BLE001
            pass
    torch.cuda.synchronize()
    return g, kept


def make_flush(device: str):
    buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device)

    def flush():
        buf.fill_(1)

    return flush


def time_interleaved(fns: Dict[str, callable], reps: int, warmup: int, batch: int, flush, host: bool = False) -> dict:
    """Per sample, for each variant in an order rotated by the sample index: flush L2, record, run `batch` calls,
    record. Returns {name: {'us': [per-call us per sample], 'host_us': [...]}}; the lists are index-aligned across
    variants (sample r of every variant), which is what the paired differences need."""
    names = list(fns)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    torch.cuda.synchronize()
    ev: Dict[str, list] = {n: [] for n in names}
    hs: Dict[str, list] = {n: [] for n in names}
    for r in range(reps):
        k = r % len(names)
        for n in names[k:] + names[:k]:
            if flush is not None:
                flush()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            t0 = time.perf_counter()
            for _ in range(batch):
                fns[n]()
            t1 = time.perf_counter()
            b.record()
            ev[n].append((a, b))
            hs[n].append((t1 - t0) * 1e6 / batch)
    torch.cuda.synchronize()
    out = {}
    for n in names:
        out[n] = {"us": [a.elapsed_time(b) * 1e3 / batch for a, b in ev[n]]}
        if host:
            out[n]["host_us"] = hs[n]
    return out


def describe_timing(t: dict, pairs: int) -> dict:
    """{'us': stats of per-replay us, 'us_per_pair', 'host_us'?}."""
    s = summarize(t["us"])
    d = {"replay_us": s, "us_per_pair": {k: (v / pairs if k != "n" else v) for k, v in s.items()}}
    if "host_us" in t:
        d["host_us"] = summarize(t["host_us"])
    return d


def savings(tim: Dict[str, dict], pairs: int, base: str = "a") -> dict:
    """Paired differences base - variant per sample: us per replay / per pair / per owned edge (2P - 1), with CI."""
    edges = 2 * pairs - 1
    out = {}
    if base not in tim:
        base = next(iter(tim))
    names = [n for n in tim if n != base]
    ref = [(base, n) for n in names]
    if "b" in tim:
        ref += [(n, m) for n, m in (("b", "c"), ("b", "c2")) if n in tim and m in tim]
    for a, b in ref:
        d = paired_diff(tim[a]["us"], tim[b]["us"])
        if not d.get("n"):
            continue
        out[f"{a}-{b}"] = {"n": d["n"], "per_replay_us": d["mean"], "ci95_us": d["ci95"], "std_us": d["std"],
                           "per_pair_us": d["mean"] / pairs, "per_edge_us": d["mean"] / edges,
                           "per_edge_ci95_us": d["ci95"] / edges, "frac_first_slower": d["frac_a_slower"]}
    return out


# ------------------------------------------------------------------------------------------------ one chain
def run_one_chain(K, kind: str, args, device: str, sms: int, tab, cache, user_cfgs) -> dict:
    specs = chain_specs(kind, args.pairs, K.SHAPES)
    R = args.rows
    P = len(specs)
    cfgs = pick_cfgs(K, specs, R, sms, user_cfgs, cache)
    wt = Weights(specs, device)
    inputs = Inputs(specs, R, max(1, args.stress_inputs))
    xdev = [(to_dev(a, device), to_dev(b, device)) for a, b in inputs.x]
    names = [v for v in args.variants]
    res: dict = {"kind": kind, "pairs": P, "kernels": 2 * P, "edges_expected": 2 * P - 1, "rows": R,
                 "weight_bytes": wt.bytes, "shapes": [s[0] for s in specs],
                 "cfgs": {s[0]: c.as_dict() for s, c in zip(specs, cfgs)},
                 "plans": {}, "variants": {}}
    for nm, N, Kd in {(s[0], s[1], s[2]) for s in specs}:
        c = next(c for s, c in zip(specs, cfgs) if s[0] == nm)
        p = K.gemv_plan(c, R, N, Kd)
        res["plans"][nm] = {"grid": list(p["grid"]), "PER": p["PER"], "NSPLIT": p["NSPLIT"], "EVEN": p["EVEN"],
                            "prefetch_lines": K.prefetch_lines(p["PER"]), "prefetch_slines": K.prefetch_slines(p["PER"])}
    flush = None if args.no_flush else make_flush(device)

    states: Dict[str, State] = {}
    for n in names:
        st = State(specs, R, device, arena=not args.no_arena)
        st.set_static(inputs)
        build_plans(K, st, cfgs)
        st.set_input(*xdev[0])
        states[n] = st

    def runner(n):
        v = VARIANTS[n]
        return lambda: run_chain(K, states[n], wt, cfgs, tab, pdl=v["pdl"], prefetch=v["prefetch"])

    fns = {n: runner(n) for n in names}
    # ---- eager warm-up (compiles every kernel of every variant; a failure here is a recorded result, not a crash)
    ok = []
    for n in names:
        try:
            fns[n]()
            torch.cuda.synchronize()
            ok.append(n)
            res["variants"][n] = {"label": VARIANTS[n]["label"], "compile": "ok"}
        except Exception as e:  # noqa: BLE001
            res["variants"][n] = {"label": VARIANTS[n]["label"], "compile": "FAILED",
                                  "error": f"{type(e).__name__}: {e}", "traceback_tail": traceback.format_exc()[-1500:]}
    names = ok
    if not names:
        return res
    modes = ["graph", "eager"] if args.mode == "both" else [args.mode]
    refs: List[State] = []

    if "graph" in modes:
        graphs, gmodes = {}, {}
        for n in names:
            try:
                dot = os.path.join(args.dot_dir, f"{kind}-{n}.dot") if args.dot_dir else None
                g, kept = capture_graph(fns[n], keep_graph=True, dot_path=dot)
                graphs[n] = g
                res["variants"][n]["capture"] = "ok"
                res["variants"][n]["census"] = graph_census(g) if kept else {"error": "torch has no keep_graph"}
            except Exception as e:  # noqa: BLE001
                res["variants"][n]["capture"] = "FAILED"
                res["variants"][n]["capture_error"] = f"{type(e).__name__}: {e}"
                res["variants"][n]["traceback_tail"] = traceback.format_exc()[-1500:]
        gn = [n for n in names if n in graphs]
        res["graph"] = {"variants": gn}
        if gn:
            for n in gn:
                states[n].set_input(*xdev[0])
            # --- correctness: per-input reference from the first variant (a if present), then the stress loop
            base = gn[0]
            refs = []
            for i in range(len(xdev)):
                states[base].set_input(*xdev[i])
                graphs[base].replay()
                torch.cuda.synchronize()
                refs.append(states[base].snapshot())
            mism = {n: torch.zeros((), dtype=torch.int64, device=device) for n in gn}
            for rep in range(args.stress):
                i = rep % len(xdev)
                k = rep % len(gn)
                for n in gn[k:] + gn[:k]:
                    states[n].set_input(*xdev[i])
                    graphs[n].replay()
                    mism[n] += states[n].mismatch(refs[i])
            torch.cuda.synchronize()
            res["graph"]["stress"] = {"replays": args.stress, "inputs": len(xdev), "reference": base,
                                      "mismatched_bytes": {n: int(mism[n]) for n in gn},
                                      "bit_identical": all(int(mism[n]) == 0 for n in gn)}
            # --- timing (fixed input 0)
            for n in gn:
                states[n].set_input(*xdev[0])
            reps = max(1, args.replays // args.batch)
            tim = time_interleaved({n: graphs[n].replay for n in gn}, reps, args.warmup, args.batch, flush)
            res["graph"]["timing"] = {n: describe_timing(tim[n], P) for n in gn}
            res["graph"]["savings"] = savings(tim, P)
            res["graph"]["replays"] = reps * args.batch
            res["graph"]["batch"] = args.batch
            res["graph"]["_raw_us"] = {n: tim[n]["us"] for n in gn} if args.raw else None
            del graphs

    if "eager" in modes:
        for n in names:
            states[n].set_input(*xdev[0])
        reps = max(1, args.replays // args.batch)
        tim = time_interleaved(fns, reps, args.warmup, args.batch, flush, host=True)
        res["eager"] = {"variants": names, "timing": {n: describe_timing(tim[n], P) for n in names},
                        "savings": savings(tim, P), "replays": reps * args.batch, "batch": args.batch}
        # eager vs the graph reference (or, without one, vs the first eager variant) -- one input, bit for bit
        for n in names:
            states[n].set_input(*xdev[0])
            fns[n]()
        torch.cuda.synchronize()
        ref = refs[0] if refs else states[names[0]].snapshot()
        res["eager"]["bit_identical_vs_ref"] = {n: int(states[n].mismatch(ref)) == 0 for n in names}
    del wt, states, refs, fns
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------ verdict
def verdict(report: dict) -> List[str]:
    """Human summary lines from a report (pure; unit-tested on synthetic numbers)."""
    L: List[str] = []
    for ch in report.get("chains", []):
        k = ch.get("kind")
        vs = ch.get("variants", {})
        for n, v in vs.items():
            if v.get("compile") == "FAILED":
                L.append(f"[{k}] variant {n} COMPILE FAILED: {v.get('error')}")
            elif v.get("capture") == "FAILED":
                L.append(f"[{k}] variant {n} CAPTURE FAILED: {v.get('capture_error')}")
        g = ch.get("graph")
        if not g:
            continue
        st = g.get("stress")
        if st:
            L.append(f"[{k}] graph bit-identity over {st['replays']} replays x {st['inputs']} rotating inputs vs {st['reference']}: "
                     + ("ALL IDENTICAL" if st["bit_identical"] else f"MISMATCH {st['mismatched_bytes']}"))
        for n in g.get("variants", []):
            c = vs.get(n, {}).get("census", {})
            if "error" in c:
                L.append(f"[{k}] graph {n}: census unavailable ({c['error']})")
            elif c:
                pe = sum(v for kk, v in c["edge_kinds"].items() if kk.startswith("programmatic"))
                want = c["kernel_nodes"] - 1                      # one edge between each consecutive pair of kernels
                if not VARIANTS.get(n, {}).get("pdl"):
                    flag = "" if pe == 0 else f"  <-- UNEXPECTED: {pe} programmatic edges on a plain graph"
                elif pe == 0:
                    flag = "  <-- PDL NOT ATTACHED under capture (0 programmatic edges)"
                elif pe != want:
                    flag = f"  <-- programmatic on {pe} of {want} kernel-to-kernel edges"
                else:
                    flag = f"  (programmatic on all {want} kernel-to-kernel edges)"
                L.append(f"[{k}] graph {n}: {c['nodes']} nodes ({c['kernel_nodes']} kernel), {c['edges']} edges {c['edge_kinds']}{flag}")
        for key, s in (g.get("savings") or {}).items():
            L.append(f"[{k}] graph {key}: {s['per_replay_us']:+.2f} +- {s['ci95_us']:.2f} us/replay = "
                     f"{s['per_pair_us']:+.3f} us/pair = {s['per_edge_us']:+.3f} +- {s['per_edge_ci95_us']:.3f} us/edge "
                     f"(n={s['n']}, first slower in {100 * s['frac_first_slower']:.0f} % of samples)")
        e = ch.get("eager")
        if e:
            for n, t in e["timing"].items():
                L.append(f"[{k}] eager {n}: gpu {t['replay_us']['mean']:.1f} us/chain (p50 {t['replay_us']['p50']:.1f}), "
                         f"host launch {t.get('host_us', {}).get('mean', float('nan')):.1f} us/chain")
    return L


# ------------------------------------------------------------------------------------------------ main
def env_info() -> dict:
    d = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0]}
    try:
        import triton

        d["triton"] = triton.__version__
    except ImportError:
        d["triton"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update(device=p.name, cc=f"{p.major}.{p.minor}", sms=p.multi_processor_count,
                 hbm_gib=round(p.total_memory / 2 ** 30, 1))
    return d


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["graph", "eager", "both"], default="both")
    ap.add_argument("--chains", default="mixed;A;B", help="';'-separated chain kinds: mixed | A | B | a comma list like A,B,B")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--variants", default="a,b,c,c2", help="subset of a,b,c,c2 (a is the baseline)")
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1, help="replays per timed sample (1 = one event pair per replay)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--stress", type=int, default=400, help="bit-identity replays with a new input each")
    ap.add_argument("--stress-inputs", type=int, default=8)
    ap.add_argument("--no-flush", action="store_true", help="skip the 512 MiB L2 flush before each sample")
    ap.add_argument("--gemv-cache", default=None, help="runtime tune cache JSON ($DSV41_GEMV_CACHE) for the GEMV configs")
    ap.add_argument("--cfg-a", default=None, help="block_n,ku,warps,stages,split_k,mma for shape A")
    ap.add_argument("--cfg-b", default=None, help="same for shape B")
    ap.add_argument("--json", default=None, help="receipt path (default scratch/pdl-probe-<utc>.json)")
    ap.add_argument("--dot-dir", default=None, help="also debug_dump each captured graph here (best effort)")
    ap.add_argument("--no-arena", action="store_true", help="separate state tensors instead of one uint8 arena (debug)")
    ap.add_argument("--raw", action="store_true", help="keep the raw per-sample us in the receipt")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit (no GPU needed)")
    a = ap.parse_args(argv)
    a.variants = [v for v in a.variants.split(",") if v]
    for v in a.variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v!r}")
    a.chain_list = [c for c in a.chains.split(";") if c]
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    K = load_kernels()
    user = {}
    if args.cfg_a:
        user["A"] = parse_cfg(args.cfg_a)
    if args.cfg_b:
        user["B"] = parse_cfg(args.cfg_b)
    cache = load_cfg_cache(args.gemv_cache)
    if args.plan or not torch.cuda.is_available():
        sms = K.DEFAULT_SMS
        print(json.dumps(env_info()))
        for kind in args.chain_list:
            specs = chain_specs(kind, args.pairs, K.SHAPES)
            cfgs = pick_cfgs(K, specs, args.rows, sms, user, cache)
            print(f"chain {kind}: {len(specs)} pairs, {2 * len(specs)} kernels, {2 * len(specs) - 1} edges")
            for nm in sorted({s[0] for s in specs}):
                s = next(s for s in specs if s[0] == nm)
                c = next(c for sp, c in zip(specs, cfgs) if sp[0] == nm)
                p = K.gemv_plan(c, args.rows, s[1], s[2])
                print(f"  shape {nm} N={s[1]} K={s[2]}  cfg {c.as_dict()}  grid {p['grid']} PER {p['PER']} NSPLIT {p['NSPLIT']} "
                      f"prefetch lines {K.prefetch_lines(p['PER'])}/{K.prefetch_slines(p['PER'])}")
        if not args.plan:
            print("no CUDA device: plan only (run on the box)")
        return 0
    device = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tab = scale_code_table(device)
    report = {"probe": "M2-L5 PDL", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": env_info(),
              "args": {k: v for k, v in vars(args).items() if k != "chain_list"}, "chains": []}
    for kind in args.chain_list:
        print(f"== chain {kind} ==", flush=True)
        try:
            ch = run_one_chain(K, kind, args, device, sms, tab, cache, user)
        except Exception as e:  # noqa: BLE001
            ch = {"kind": kind, "error": f"{type(e).__name__}: {e}", "traceback_tail": traceback.format_exc()[-2000:]}
            print(f"chain {kind} FAILED: {ch['error']}", flush=True)
        report["chains"].append(ch)
        for line in verdict({"chains": [ch]}):
            print(line, flush=True)
    path = args.json or os.path.join(REPO, "scratch", f"pdl-probe-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"receipt: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
