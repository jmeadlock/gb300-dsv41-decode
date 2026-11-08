"""pdl.py -- M2-L5: Programmatic Dependent Launch (PDL) and pre-wait L2 weight prefetch on the OWNED Triton kernels.
docs/HANDOFF-M2-L5-PDL-IMPL.md; evidence: bench/pdl_probe.py + docs/pdl-probe-clean-20261107.json.

What the switch does (and does not do). PDL lets a kernel launch while its stream predecessor is still running; the
kernel's `griddepcontrol.wait` (tl.extra.cuda.gdc_wait) then blocks until every prerequisite grid has COMPLETED and
its memory is visible. So PDL moves timing only: every load/store a kernel issues after its wait sees exactly what it
saw before, and the arithmetic, tiling and launch grid of every kernel are untouched -- outputs are bit-identical by
construction (gated on the box: graph == eager, replay, 1 row == 6 rows per row).

    $DSV41_PDL           off (default, until the box gate) | on | prefetch
    $DSV41_PDL_TRIGGER   entry (default) | late         where gdc_launch_dependents() sits (below)
    $DSV41_PDL_EXCLUDE   comma list of kernel names and/or families: those launch exactly as with off
    $DSV41_PDL_ONLY      comma list of kernel names and/or families: when set, only those get PDL
    $DSV41_PDL_NOPF      comma list: PDL but no prefetch for those (prefetch mode)

  off       every launch is today's launch: no `launch_pdl`, the kernels' PDL constexprs stay 0 (no gdc_* instruction,
            no prefetch is compiled in).
  on        every owned launch gets `launch_pdl=True` (cudaLaunchAttributeProgrammaticStreamSerialization; under
            torch.cuda.graph capture this becomes a programmatic edge from the stream predecessor, verified by the
            probe: 79/79 programmatic) and the kernel runs gdc_wait() as its first device operation -- before any load
            or store (the only work above it is index arithmetic on program ids / scalar args) -- and
            gdc_launch_dependents() once per program.
  prefetch  on + the kernels that read WEIGHTS issue `prefetch.global.L2` (an L2 hint: no register result, no
            ordering, no fault on an in-bounds address) of the program's own weight slice BEFORE gdc_wait: the GEMV
            family (t9b_mxfp8_gemv w + s, t9b4_mxfp8_gemv2, the T9b-5 fixed fp32 dots head / mix / comp, m2_gate_fp32)
            and the glue kernels whose weights are norm weights or engram's per-copy weight (hc_norm, t9b4_hc_norm,
            norm_row: nw; engram_gate: wp). Nothing else precedes the wait: no load of any kind (the probe's alternative
            "plain-load" prefetch is NOT used), so a weight rewritten inside the step (residency swaps, staging) can
            never be read early -- a prefetch is a cache hint and L2 is the GPU's point of coherence.

  trigger   entry: gdc_launch_dependents() right after gdc_wait() -- the form the probe measured (-1.29 us/edge PDL,
            -5.07 us/edge with prefetch on the mixed chain). late: after the kernel's last weight load (the MXFP8 GEMV:
            after its K loop, before the epilogue stores; every other kernel: at the end of its body, where its last
            load is) -- the handoff's literal placement, UNMEASURED; kept as a box A/B. Either way the trigger comes
            AFTER the wait, so at most two grids overlap and a dependent's pre-wait prefetch never starts before its
            producer's producer has completed. Correctness does not depend on the placement (the wait does all the
            ordering).

Hazards checked (tests/test_pdl_impl_cpu.py runs every owned kernel through tests/triton_shim.py with PDL on and
asserts them per program; the shim fails any launch that breaks them):
  * nothing but prefetch hints precede gdc_wait: no load (of a producer's output OR of a weight), no store, no atomic.
    So a consumer that writes a buffer its producer still reads cannot race (its stores are after the wait), and a
    pre-wait read of a producer's output cannot happen.
  * exactly one gdc_wait and one gdc_launch_dependents per program, the trigger after the wait, neither inside a loop
    (Triton 3.7.1 pipeliner bug #11905): every gdc_* call sits at the kernel body's top level.
  * transitivity: a kernel's completion implies its wait returned, which implies its predecessor completed -- so the
    chain of completions stays total even though consecutive grids overlap. A borrowed kernel (DeepGEMM / FlashMLA /
    TileLang / torch) is launched without the attribute: it waits for our kernel's completion as today. An owned
    kernel after a borrowed one gets a programmatic edge too (the attribute is on the consumer); the borrowed
    producer never triggers early, so the implicit trigger at its exit applies and only our prologue (the prefetch)
    overlaps its drain. CAVEAT for the box: if a borrowed library launches ITSELF with PDL after one of our early-
    triggering kernels, it must wait before reading our outputs (DeepGEMM / FlashMLA wait at entry when they enable
    PDL); the bit gates (streams 8/8, self-replay) are the check, and the graph census lists any programmatic edge
    into a borrowed kernel.
  * co-residency: a dependent's CTAs are admitted only after EVERY primary CTA has triggered (or exited), i.e. all
    primary CTAs are already resident, so an early trigger never steals slots from the primary's own later waves and
    cannot deadlock (also with the ticket kernels m2_router_rows / m2_router_plan: their wait is above the ticket).
    What it can do is buy nothing: a primary whose grid fills every SM at full occupancy leaves no room for the
    dependent's CTAs until its tail -- graph_census() lists each owned kernel's CTA count against the SM count.
  * kernels also called as device functions (moe_rows.m1_dedup_warp inside m2_router_rows / m2_router_plan): the
    PDL constexprs default to 0, so the inlined body carries no gdc_* instruction.

`KERNELS` is the allow-list of owned kernels (name -> family, the weight arguments prefetched, the late trigger
point); test probes (t9b_pair_scale_probe, t9b2_rope_probe, m2_softplus_probe) are not in it and never get PDL.
Borrowed kernels (DeepGEMM, FlashMLA, TileLang, torch) are never touched -- our consumers may still prefetch after them.
"""
from __future__ import annotations

import functools
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ENV = "DSV41_PDL"
TRIGGER_ENV = "DSV41_PDL_TRIGGER"
EXCLUDE_ENV = "DSV41_PDL_EXCLUDE"
ONLY_ENV = "DSV41_PDL_ONLY"
NOPF_ENV = "DSV41_PDL_NOPF"
MODES = ("off", "on", "prefetch")
TRIGGERS = ("entry", "late")
DEFAULT_MODE = "off"          # until the box gate (headline with on, then prefetch)
DEFAULT_TRIGGER = "entry"     # the probe-measured form

# the kernels' PDL constexpr: 0 off, 1 wait + trigger at entry, 2 wait at entry + trigger late
PDL_OFF, PDL_ENTRY, PDL_LATE = 0, 1, 2

LINE = 128                    # bytes per L2 line / prefetch granule
PF_BLOCK = 2048               # prefetch points per chunk (a [rows, lines] block; bounds the prologue's registers)

# name -> (family, prefetched weight arguments, late trigger point)
KERNELS: Dict[str, Tuple[str, Tuple[str, ...], str]] = {
    # runtime/gemv_mxfp8.py
    "t9b_mxfp8_gemv": ("gemv", ("w", "s"), "after the K loop, before the epilogue stores"),
    "t9b_splitk_reduce": ("gemv", (), "end of body"),
    "t9b4_mxfp8_gemv2": ("gemv", ("wa", "sa", "wb", "sb"), "end of body (the tile helper is shared verbatim)"),
    # runtime/static_kernels.py
    "t9b5_head_fp32": ("fp32", ("w",), "end of body"),
    "t9b5_mix_fp32": ("fp32", ("w",), "end of body"),
    "t9b8_comp_fp32": ("fp32", ("w",), "end of body"),
    "m2_gate_fp32": ("router", ("w",), "end of body"),
    "m2_router_rows": ("router", (), "end of body"),
    "m2_router_plan": ("router", (), "end of body"),
    "t8_gather_rows": ("moe", (), "end of body"),
    "t8_gather_rows_u8": ("moe", (), "end of body"),
    "t9b2_gather_tables": ("moe", (), "end of body"),
    "t9b2_gather_tables_u8": ("moe", (), "end of body"),
    "t9a2_stage_inplace": ("moe", (), "end of body"),
    "m1_stage_rows": ("moe", (), "end of body"),
    "t8_step_marker": ("marker", (), "end of body"),
    # runtime/moe_rows.py
    "m1_dedup_routes": ("moe", (), "end of body"),
    "m1_dedup_warp": ("moe", (), "end of body"),
    # runtime/fused_glue.py
    "t9b2_act_quant_bcast": ("glue", (), "end of body"),
    "t9b2_swiglu_quant": ("glue", (), "end of body"),
    "t9b2_moe_combine": ("glue", (), "end of body"),
    "t9b2_hc_post": ("glue", (), "end of body"),
    "t9b2_o_rope_qdq": ("glue", (), "end of body"),
    # runtime/fused_glue_r2.py
    "t9b2r2_hc_norm": ("glue", ("nw",), "end of body"),
    "t9b4_hc_norm": ("glue", ("nw",), "end of body"),
    "t9b2r2_norm_row": ("glue", ("nw",), "end of body"),
    "t9b2r2_router_pre": ("glue", (), "end of body"),
    "t9b2r2_router_post": ("glue", (), "end of body"),
    "t9b2r2_engram_gate": ("glue", ("wp",), "end of body"),
    "t9b2r2_engram_out": ("glue", (), "end of body"),
    "t9b2r2_topk_post": ("glue", (), "end of body"),
}
FAMILIES = tuple(sorted({v[0] for v in KERNELS.values()}))
MODULES = {  # where each kernel lives (tests walk these)
    "runtime/gemv_mxfp8.py": ("t9b_mxfp8_gemv", "t9b_splitk_reduce", "t9b4_mxfp8_gemv2"),
    "runtime/static_kernels.py": ("t9b5_head_fp32", "t9b5_mix_fp32", "t9b8_comp_fp32", "m2_gate_fp32",
                                  "m2_router_rows", "m2_router_plan", "t8_gather_rows", "t8_gather_rows_u8",
                                  "t9b2_gather_tables", "t9b2_gather_tables_u8", "t9a2_stage_inplace",
                                  "m1_stage_rows", "t8_step_marker"),
    "runtime/moe_rows.py": ("m1_dedup_routes", "m1_dedup_warp"),
    "runtime/fused_glue.py": ("t9b2_act_quant_bcast", "t9b2_swiglu_quant", "t9b2_moe_combine", "t9b2_hc_post",
                              "t9b2_o_rope_qdq"),
    "runtime/fused_glue_r2.py": ("t9b2r2_hc_norm", "t9b4_hc_norm", "t9b2r2_norm_row", "t9b2r2_router_pre",
                                 "t9b2r2_router_post", "t9b2r2_engram_gate", "t9b2r2_engram_out", "t9b2r2_topk_post"),
}
GEMV_PF_FAMILIES = ("gemv", "fp32", "router")   # prefetch of MB-scale weight slices (the probe's shapes)


def prefetching(name: str) -> bool:
    """Whether kernel `name` has a pre-wait weight prefetch (in prefetch mode)."""
    return bool(KERNELS.get(name, ("", (), ""))[1])


# ------------------------------------------------------------------------------------------------ the switch
def _names(text: Optional[str]) -> frozenset:
    return frozenset(t.strip() for t in (text or "").split(",") if t.strip())


class Config:
    """The parsed environment (immutable; `config()` caches one per distinct environment)."""

    def __init__(self, mode: str, trigger: str, exclude: frozenset, only: frozenset, nopf: frozenset):
        for nm in exclude | only | nopf:
            if nm not in KERNELS and nm not in FAMILIES:
                raise ValueError(f"$DSV41_PDL_*: {nm!r} is neither an owned kernel ({sorted(KERNELS)}) nor a family "
                                 f"({FAMILIES})")
        self.mode, self.trigger = mode, trigger
        self.exclude, self.only, self.nopf = exclude, only, nopf
        self.pdl = PDL_ENTRY if trigger == "entry" else PDL_LATE

    @staticmethod
    def _hit(name: str, s: frozenset) -> bool:
        return name in s or KERNELS[name][0] in s

    def enabled(self, name: str) -> bool:
        if self.mode == "off" or name not in KERNELS:
            return False
        if self.only and not self._hit(name, self.only):
            return False
        return not self._hit(name, self.exclude)

    def prefetch(self, name: str) -> bool:
        return self.mode == "prefetch" and self.enabled(name) and prefetching(name) and not self._hit(name, self.nopf)

    def describe(self) -> dict:
        return {"mode": self.mode, "trigger": self.trigger, "exclude": sorted(self.exclude), "only": sorted(self.only),
                "nopf": sorted(self.nopf),
                "enabled": sorted(n for n in KERNELS if self.enabled(n)),
                "prefetch": sorted(n for n in KERNELS if self.prefetch(n))}


@functools.lru_cache(maxsize=16)
def _parse(mode: Optional[str], trigger: Optional[str], exclude: Optional[str], only: Optional[str],
           nopf: Optional[str]) -> Config:
    m = (mode or DEFAULT_MODE).strip().lower()
    if m not in MODES:
        raise ValueError(f"${ENV}={mode!r} not in {MODES}")
    t = (trigger or DEFAULT_TRIGGER).strip().lower()
    if t not in TRIGGERS:
        raise ValueError(f"${TRIGGER_ENV}={trigger!r} not in {TRIGGERS}")
    return Config(m, t, _names(exclude), _names(only), _names(nopf))


def config() -> Config:
    """The current switch (read at every launch -- an env lookup + a cached parse -- so a process can capture graphs
    under different modes; the value used for a captured graph is the one at capture time)."""
    e = os.environ
    return _parse(e.get(ENV), e.get(TRIGGER_ENV), e.get(EXCLUDE_ENV), e.get(ONLY_ENV), e.get(NOPF_ENV))


def mode() -> str:
    return config().mode


# host-side tally of launches that carried PDL / prefetch (eager calls + one per capture; not per replay)
TALLY: Dict[str, List[int]] = {}


def kw(name: str, pf=None) -> dict:
    """The extra launch keywords of owned kernel `name`: {} when PDL is off for it (today's launch, bit for bit the
    same compiled kernel); else {"PDL": 1|2, "launch_pdl": True}, plus -- prefetch mode, a prefetching kernel, `pf`
    given -- {"PDL_PF": 1, **pf()} (pf: a callable returning the kernel's prefetch constexprs; called only then)."""
    c = config()
    if not c.enabled(name):
        return {}
    d = {"PDL": c.pdl, "launch_pdl": True}
    p = 0
    if pf is not None and c.prefetch(name):
        d.update(pf())
        d["PDL_PF"] = 1
        p = 1
    t = TALLY.setdefault(name, [0, 0])
    t[0] += 1
    t[1] += p
    return d


def describe() -> dict:
    d = config().describe()
    d["host_launches"] = {k: {"pdl": v[0], "prefetch": v[1]} for k, v in sorted(TALLY.items())}
    return d


# ------------------------------------------------------------------------------------------------ prefetch geometry
def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def elems_per_line(elem_bytes: int) -> int:
    """Elements of `elem_bytes` per 128 B line (1 -> 128, 2 -> 64, 4 -> 32)."""
    if elem_bytes not in (1, 2, 4, 8):
        raise ValueError(f"prefetch: element size {elem_bytes}")
    return LINE // int(elem_bytes)


def lines(n_elems: int, epl: int) -> int:
    """Line points per row for a run of n_elems elements: ceil(n / epl) line starts + 1 clamped to the run's last
    element (an unaligned run touches up to ceil(L / 128) + 1 lines), rounded up to a power of two (tl.arange)."""
    return _np2(-(-max(1, int(n_elems)) // int(epl)) + 1)


def chunk(nl: int, rows: int) -> int:
    """Line points per chunk so a [rows, chunk] block stays <= PF_BLOCK points (powers of two: chunk divides nl)."""
    return max(1, min(int(nl), PF_BLOCK // max(1, _np2(rows))))


def pf_rows(nl_elems: int, elem_bytes: int, rows: int) -> dict:
    """The {PF_EPL, PF_NL, PF_CH} constexprs of a [rows, nl_elems] weight slice (the `_pdl_prefetch` helper)."""
    epl = elems_per_line(elem_bytes)
    nl = lines(nl_elems, epl)
    return {"PF_EPL": epl, "PF_NL": nl, "PF_CH": chunk(nl, rows)}


def prefetch_points(r0: int, nr: int, nrows: int, stride: int, c0: int, clen: int, epl: int, nl: int, ch: int) -> list:
    """The element offsets `_pdl_prefetch` touches (pure mirror of the device helper, for tests and the census)."""
    out = []
    for jc in range(nl // ch):
        for r in range(nr):
            rr = min(r0 + r, nrows - 1)
            for j in range(ch):
                jj = jc * ch + j
                out.append(rr * stride + c0 + min(jj * epl, clen - 1))
    return out


# ------------------------------------------------------------------------------------------------ edge census
# The probe's per-edge numbers (docs/pdl-probe-clean-20261107.json, paired graph-replay differences, 6 rows, L2 flushed
# before each sample). PDL alone: per owned->owned edge. Prefetch: the extra saving per edge INTO a prefetching GEMV
# (variant b - c per replay over the 39 glue->GEMV edges of a 40-pair chain); mixed is the headline, A (1280x5120) /
# B (5120x8192) the range.
PROBE_US = {
    "pdl_per_edge": {"mixed": 1.2914, "A": 0.5756, "B": 2.0575},
    "prefetch_extra_per_gemv_edge": {"mixed": 298.7454 / 39, "A": 89.9576 / 39, "B": 516.0717 / 39},
}
BORROWED_HINTS = ("dg_", "deep_gemm", "flash", "mla", "sparse_attn", "act_quant", "fp8_gemm", "fp4_gemm",
                  "hc_split_sinkhorn", "tilelang", "cutlass", "cublas", "gemm", "sm100", "sm90")


def owned_name(name: str) -> Optional[str]:
    """The owned kernel a node / event name refers to ('K:' prefixes and Triton's name suffixes stripped), or None."""
    n = name[2:] if name.startswith("K:") else name
    if n in KERNELS:
        return n
    for k in KERNELS:                          # Triton kernel symbols can carry a suffix (specialization id)
        if n.startswith(k + "_") and n[len(k) + 1:].isdigit():
            return k
    return None


def classify_edges(edges: Iterable[Tuple[str, str]], cfg: Optional[Config] = None) -> dict:
    """[(producer name, consumer name)] -> the census: per class and per consumer family. Classes:
      pdl_owned      owned -> owned with the consumer PDL-enabled: a full PDL edge (both sides cooperate)
      pdl_borrowed   borrowed / torch -> owned PDL consumer: programmatic edge, implicit trigger (prologue overlap)
      none           consumer borrowed / torch / PDL-disabled: unchanged
    and pf_owned / pf_borrowed: the subset whose consumer prefetches (prefetch mode), split GEMV family vs glue."""
    c = cfg or config()
    out = {"edges": 0, "pdl_owned": 0, "pdl_borrowed": 0, "none": 0, "pf_gemv_owned": 0, "pf_gemv_borrowed": 0,
           "pf_glue_owned": 0, "pf_glue_borrowed": 0, "by_consumer": {}, "into_borrowed_from_owned": 0}
    for a, b in edges:
        out["edges"] += 1
        oa, ob = owned_name(a), owned_name(b)
        if ob is None or not c.enabled(ob):
            out["none"] += 1
            if ob is None and oa is not None:
                out["into_borrowed_from_owned"] += 1
            continue
        k = "pdl_owned" if (oa is not None and c.enabled(oa)) else "pdl_borrowed"
        out[k] += 1
        fam = KERNELS[ob][0]
        bc = out["by_consumer"].setdefault(fam, {"pdl_owned": 0, "pdl_borrowed": 0, "prefetch": 0})
        bc[k] += 1
        if c.prefetch(ob):
            bc["prefetch"] += 1
            side = "owned" if k == "pdl_owned" else "borrowed"
            out[f"pf_{'gemv' if fam in GEMV_PF_FAMILIES else 'glue'}_{side}"] += 1
    return out


def model_ms(census: dict, mode_: str = "prefetch") -> dict:
    """MODELED ms per step from a census (edges x probe us). on: pdl_owned x PDL-alone us. prefetch: + pf_gemv_owned x
    the prefetch extra. Not modeled (listed): pdl_borrowed (implicit trigger: only launch latency / drain overlap),
    glue prefetches (KB-scale norm weights, far below the probe's MB slices), pf_gemv_borrowed (upper bound given)."""
    res = {}
    for shape in ("mixed", "A", "B"):
        on = census["pdl_owned"] * PROBE_US["pdl_per_edge"][shape] / 1e3
        pf = census["pf_gemv_owned"] * PROBE_US["prefetch_extra_per_gemv_edge"][shape] / 1e3
        ub = census["pf_gemv_borrowed"] * PROBE_US["prefetch_extra_per_gemv_edge"][shape] / 1e3
        res[shape] = {"on_ms": round(on, 3), "prefetch_ms": round(on + pf, 3),
                      "prefetch_borrowed_upper_ms": round(ub, 3)}
    res["mode"] = mode_
    res["label"] = "MODEL (edges x probe us/edge; not a measurement)"
    return res


# ------------------------------------------------------------------------------------------------ box graph census
def _lib(cands: Sequence[str], tag: str):
    import ctypes

    paths: List[str] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if tag in line:
                    p = line.split()[-1]
                    if p not in paths:
                        paths.append(p)
    except OSError:
        pass
    err = []
    for c in paths + list(cands):
        try:
            return ctypes.CDLL(c)
        except OSError as e:
            err.append(str(e))
    raise RuntimeError(f"no {tag}: " + "; ".join(err[:3]))


def graph_census(g, sms: Optional[int] = None, cfg: Optional[Config] = None) -> dict:
    """Node names / grids and edge types of a torch.cuda.CUDAGraph made with keep_graph=True (box only; ctypes into
    cudart for the topology -- bench/pdl_probe.py's walk -- and libcuda's cuGraphKernelNodeGetParams + cuFuncGetName
    for each kernel node's name and grid). Returns {'error': text} instead of raising. The classification uses the
    switch at call time (pass the cfg the graph was captured under)."""
    import ctypes as C

    try:
        raw = int(g.raw_cuda_graph())
        rt = _lib(["libcudart.so.13", "libcudart.so.12", "libcudart.so"], "libcudart")
        vp, st = C.c_void_p, C.c_size_t

        class EdgeData(C.Structure):
            _fields_ = [("from_port", C.c_ubyte), ("to_port", C.c_ubyte), ("type", C.c_ubyte),
                        ("reserved", C.c_ubyte * 5)]

        def chk(rc, what):
            if rc != 0:
                rt.cudaGetErrorString.restype = C.c_char_p
                raise RuntimeError(f"{what}: cudart error {rc}: {rt.cudaGetErrorString(rc).decode()}")

        n = st(0)
        rt.cudaGraphGetNodes.argtypes = [vp, C.c_void_p, C.POINTER(st)]
        chk(rt.cudaGraphGetNodes(vp(raw), None, C.byref(n)), "cudaGraphGetNodes(count)")
        nodes = (vp * max(1, n.value))()
        chk(rt.cudaGraphGetNodes(vp(raw), nodes, C.byref(n)), "cudaGraphGetNodes")
        rt.cudaGraphNodeGetType.argtypes = [vp, C.POINTER(C.c_int)]
        names: Dict[int, str] = {}
        grids: Dict[int, Tuple[int, int]] = {}
        name_err = None
        try:
            cu = _lib(["libcuda.so.1", "libcuda.so"], "libcuda.so")
            cu.cuGraphKernelNodeGetParams.argtypes = [vp, C.c_void_p]
            cu.cuFuncGetName.argtypes = [C.POINTER(C.c_char_p), vp]
        except Exception as e:  # noqa: BLE001
            cu, name_err = None, f"{type(e).__name__}: {e}"
        for i in range(n.value):
            t = C.c_int(-1)
            chk(rt.cudaGraphNodeGetType(nodes[i], C.byref(t)), "cudaGraphNodeGetType")
            key = int(nodes[i] or 0)
            if t.value != 0:
                names[key] = f"<node type {t.value}>"
                continue
            names[key] = "<kernel>"
            if cu is None:
                continue
            buf = (C.c_ubyte * 256)()
            rc = cu.cuGraphKernelNodeGetParams(nodes[i], buf)
            if rc != 0:
                name_err = name_err or f"cuGraphKernelNodeGetParams error {rc}"
                continue
            u32 = C.cast(buf, C.POINTER(C.c_uint32))
            func = C.cast(buf, C.POINTER(vp))[0]
            grids[key] = (int(u32[2]) * int(u32[3]) * int(u32[4]), int(u32[5]) * int(u32[6]) * int(u32[7]))
            nm = C.c_char_p()
            rc = cu.cuFuncGetName(C.byref(nm), vp(func))
            if rc == 0 and nm.value:
                names[key] = nm.value.decode(errors="replace")
            else:
                name_err = name_err or f"cuFuncGetName error {rc}"
        fn = getattr(rt, "cudaGraphGetEdges_v2", None) or getattr(rt, "cudaGraphGetEdges")
        fn.argtypes = [vp, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(st)]
        ne = st(0)
        chk(fn(vp(raw), None, None, None, C.byref(ne)), "cudaGraphGetEdges(count)")
        m = max(1, ne.value)
        frm, to, ed = (vp * m)(), (vp * m)(), (EdgeData * m)()
        chk(fn(vp(raw), frm, to, ed, C.byref(ne)), "cudaGraphGetEdges")
        kinds: Dict[str, int] = {}
        prog_pairs, all_pairs = [], []
        for i in range(ne.value):
            a, b = names.get(int(frm[i] or 0), "?"), names.get(int(to[i] or 0), "?")
            k = "programmatic" if ed[i].type == 1 else "default"
            kinds[k] = kinds.get(k, 0) + 1
            all_pairs.append((a, b))
            if ed[i].type == 1:
                prog_pairs.append((a, b))
        c = cfg or config()
        cls = classify_edges(all_pairs, c)
        prog_into_borrowed = sum(1 for a, b in prog_pairs if owned_name(b) is None)
        prog_into_owned = len(prog_pairs) - prog_into_borrowed
        expected = cls["pdl_owned"] + cls["pdl_borrowed"]
        per_kernel: Dict[str, dict] = {}
        for key, nm in names.items():
            o = owned_name(nm)
            if o is None:
                continue
            d = per_kernel.setdefault(o, {"nodes": 0, "max_ctas": 0})
            d["nodes"] += 1
            d["max_ctas"] = max(d["max_ctas"], grids.get(key, (0, 0))[0])
        multiwave = sorted(k for k, v in per_kernel.items() if sms and v["max_ctas"] > sms)
        return {"nodes": n.value, "kernel_nodes": sum(1 for v in names.values() if not v.startswith("<node")),
                "edges": ne.value, "edge_kinds": kinds, "names_resolved": name_err is None, "names_error": name_err,
                "classes": cls, "programmatic_into_owned": prog_into_owned,
                "programmatic_into_borrowed": prog_into_borrowed, "programmatic_expected": expected,
                "per_owned_kernel": per_kernel, "owned_grids_over_sms": multiwave, "sms": sms,
                "switch": c.describe()}
    except Exception as e:  # noqa: BLE001 - reported, never fatal
        return {"error": f"{type(e).__name__}: {e}"}


def census_lines(c: dict, tag: str = "") -> List[str]:
    """Human summary of a graph_census / classify_edges result."""
    if "error" in c:
        return [f"[pdl census{tag}] unavailable: {c['error']}"]
    cls = c.get("classes", c)
    L = [f"[pdl census{tag}] edges {cls['edges']}: owned->owned PDL {cls['pdl_owned']}, borrowed->owned PDL "
         f"{cls['pdl_borrowed']}, unchanged {cls['none']}; prefetch edges GEMV {cls['pf_gemv_owned']} (+"
         f"{cls['pf_gemv_borrowed']} after borrowed), glue {cls['pf_glue_owned']} (+{cls['pf_glue_borrowed']})"]
    if "programmatic_into_owned" in c:
        L.append(f"[pdl census{tag}] graph: {c['nodes']} nodes, {c['edges']} edges {c['edge_kinds']}; programmatic "
                 f"into owned {c['programmatic_into_owned']} (expected {c['programmatic_expected']}), into borrowed "
                 f"{c['programmatic_into_borrowed']}; owned grids over {c.get('sms')} SMs: "
                 f"{','.join(c['owned_grids_over_sms']) or 'none'}"
                 + ("" if c.get("names_resolved") else f"; names: {c.get('names_error')}"))
    for fam, v in sorted(cls.get("by_consumer", {}).items()):
        L.append(f"[pdl census{tag}]   consumer family {fam:7s} owned->owned {v['pdl_owned']:7g}  borrowed->owned "
                 f"{v['pdl_borrowed']:7g}  prefetching {v['prefetch']:7g}")
    return L


def print_census(c: dict, tag: str = "", file=None) -> None:
    for line in census_lines(c, tag):
        print(line, file=file or sys.stdout, flush=True)
