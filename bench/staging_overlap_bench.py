#!/usr/bin/env python3
"""staging_overlap_bench.py -- M3-S1 probe (docs/HANDOFF-M3-S1-STAGING-OVERLAP.md): can the 6-row step's cold-expert
staging (m1_stage_rows, 114 us/layer = 4.55 ms/step at ~263 GB/s, serially before the expert GEMMs) be hidden?

Subcommands
  model   [E] desk model only (CPU, no GPU): runtime/staging_overlap.model_step with the anatomy's kernel means and a
          Poisson cold-count distribution at the measured mean; prints the per-c table and ms/step per design.
  probe   (GPU, box) synthetic REAL shapes (w1/w3 [2304, 2560] int8 + [2304, 160] e8m0, w2 [5120, 1152] + [5120, 72];
          18.8 MB per expert), cold experts in a cudaHostRegister'd hugepage HostBuffer (as the runtime's cold pool;
          --host pinned = torch pin_memory), a [H + 16, N, K/2] HBM pool per projection with a packed SF copy (the
          runtime's sf_store "pool"), 6 rows x top-6 over 22 unique experts per layer-round, c of them cold.
          Every timed unit is ONE layer's work, unrolled over --rounds rounds with DISTINCT cold experts and hot sets
          per round (the cold footprint, rounds x 4 x 18.8 MB, exceeds L2: no replay reads cached host bytes), the
          rounds captured into one CUDA graph and replayed --reps times.
    A bandwidth  per c in 0..4 cold experts: m1_stage_rows (today), s1_stage_proj x 3, copy engine (one
                 cudaMemcpyAsync per expert x projection; 1, 2, 4 streams in flight) + the scale pack, CE weights
                 only. us/layer and GB/s (c x 18.8 MB / us). CE nodes in these graphs have FIXED addresses -- valid
                 here because each round's cold set is fixed by construction, i.e. this is CE's ceiling IF the
                 cold set were known at capture (it is not in the step: runtime/staging_overlap.py docstring).
    B overlap    per c: one layer = shared-expert stand-in (bf16 torch.mm of the shared expert's shapes on the main
                 stream: a STAND-IN, labeled) + the 3 DeepGEMM contiguous16 calls (w1, w3, w2; real kernel, real
                 shapes, the runtime's layout code), with the staging (i) none (GEMM-only floor), (ii) serial
                 m1_stage_rows before (today), (iii) kernel-overlap: forked s1_stage_proj per projection, main waits
                 on projection p's event before GEMM p, per --cfgs nprog:block_bytes pair, (iv) ce-overlap with fixed
                 addresses (the ceiling). exposed_us = variant - GEMM-only. Plus each GEMM's eager event duration
                 alone vs under the forked copy (contention), and the zero-cold cost (c = 0).
    C checks     after one round of each variant: spare slots + packed SF bytes and every GEMM output row bit-equal
                 to the serial variant's.
    D capability CUDA conditional-node APIs exposed by this torch / cuda-python (what a capturable CE would need).
    E model      the measured exposed_us per c (B) weighted by Poisson(cold mean 1.58) -> staging ms/step [E] per
                 variant and the modeled 6-row step against the 20.30 ms headline (4.55 ms of it serial staging).
  all     probe + model.

Run on the box (library image, repo at /work, DG JIT cache mounted; no checkpoint needed):
  python3 bench/staging_overlap_bench.py probe --out scratch/staging-overlap-probe-<date>.json
deep_gemm comes from vllm's third_party WITHOUT importing vllm (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime import staging_overlap as SO  # noqa: E402

PROJS = ("w1", "w3", "w2")
DIM, INTER, TOPK, ROWS = 5120, 2304, 6, 6
NK = {"w1": (INTER, DIM), "w3": (INTER, DIM), "w2": (DIM, INTER)}   # (N, K)
SPARE, UNIQUE, CMAX, ALIGN = 16, 22, 4, 16
RECIPE = dict(recipe_a=(1, 32), recipe_b=(1, 32))
FP8 = torch.float8_e4m3fn


def _per_proj_cfg(cfg, pj):
    """A --cfgs entry: (nprog, block_bytes) for all three projections, or ((n, b), (n, b), (n, b)) per projection."""
    if cfg is None:
        return None, None
    return cfg[pj] if isinstance(cfg[0], tuple) else cfg


def _cfg_tag(cfg) -> str:
    if isinstance(cfg[0], tuple):
        return "np" + "-".join(f"{n}b{b // 1024}k" for n, b in cfg)
    return f"np{cfg[0]}b{cfg[1] // 1024}k"


def parse_cfgs(spec: str):
    """'48:8192,16:32768' -> [(48, 8192), (16, 32768)]; a '/'-joined triple 'n:b/n:b/n:b' is one per-projection cfg
    (w1/w3/w2, the runtime's $DSV41_STAGE_OVERLAP_NPROG=48,16,16 / _BLOCK_BYTES=8192,32768,32768)."""
    out = []
    for v in str(spec).split(","):
        if not v:
            continue
        parts = [tuple(int(x) for x in q.split(":")) for q in v.split("/")]
        if any(len(q) != 2 for q in parts) or len(parts) not in (1, 3):
            raise ValueError(f"--cfgs entry {v!r}: nprog:block_bytes or three of them joined by '/'")
        out.append(parts[0] if len(parts) == 1 else tuple(parts))
    return out


def _key(v: str, cfg) -> str:
    return v if cfg is None else f"{v}/{_cfg_tag(cfg)}"


def _date() -> str:
    return time.strftime("%Y%m%d")


# ------------------------------------------------------------------------------------------------ model [E]
def cmd_model(a) -> dict:
    p = {"cold_mean": a.cold_mean, "kernel_GBps": a.kernel_gbps, "ce_GBps": a.ce_gbps, "contention": a.contention}
    r = SO.model_step(p)
    r["label"] = ("[E] desk model: per-layer timeline of runtime/staging_overlap.layer_exposed_us; kernel means [M] "
                  "from docs/gap-tf-anatomy-20261107.json; cold count per layer ~ Poisson(mean) [A]; the ce rows "
                  "assume a capturable data-dependent CE that does not exist (runtime/staging_overlap.py docstring)")
    print("[E] staging on the critical path, us per layer by cold count c:")
    print(f"  {'c':>2s} {'kernel':>8s} {'k-overlap':>10s} {'ce':>8s} {'ce-overlap':>11s}")
    for c, v in r["per_c_exposed_us"].items():
        print(f"  {c:>2d} {v['kernel']:8.1f} {v['kernel-overlap']:10.1f} {v['ce']:8.1f} {v['ce-overlap']:11.1f}")
    print("[E] ms/step (40 layers, Poisson cold mean %.2f):" % a.cold_mean)
    for k, v in r["staging_exposed_ms_per_step"].items():
        print(f"  {k:36s} staging {v:6.3f} ms -> step {r['step_ms_E'][k]:6.2f} ms")
    return r


# ------------------------------------------------------------------------------------------------ probe helpers
def _cover(rng, experts, k=TOPK, rows=ROWS):
    for _ in range(100000):
        rr = [sorted(rng.sample(experts, k)) for _ in range(rows)]
        if len({e for r in rr for e in r}) == len(experts):
            return rr
    raise RuntimeError("no routing cover")


def _graph_ms(body, reps: int, warm: int = 2) -> float:
    """ms per replay of a graph holding body()'s launches (body enqueues on the current stream; forks inside must
    use pre-created streams / events)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / reps
    del g
    return ms


def _capability() -> dict:
    cap = {"torch": torch.__version__, "torch_cuda": torch.version.cuda}
    try:
        cap["device"] = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None
    except Exception as e:  # pragma: no cover
        cap["device"] = repr(e)[:100]
    for name in ("begin_capture_to_if_node", "begin_capture_to_while_loop_node", "end_capture_to_conditional_node",
                 "set_conditional_handle"):
        cap[f"torch.cuda.CUDAGraph.{name}"] = hasattr(torch.cuda.CUDAGraph, name)
    try:
        from cuda.bindings import runtime as crt  # type: ignore

        cap["cuda.bindings"] = True
        nt = getattr(crt, "cudaGraphConditionalNodeType", None)
        cap["cudaGraphCondTypeSwitch"] = bool(nt is not None and hasattr(nt, "cudaGraphCondTypeSwitch"))
        cap["cudaGraphConditionalHandleCreate"] = hasattr(crt, "cudaGraphConditionalHandleCreate")
        cap["cudaStreamGetCaptureInfo"] = hasattr(crt, "cudaStreamGetCaptureInfo")
    except Exception as e:
        cap["cuda.bindings"] = f"not importable: {type(e).__name__}"
    cap["note"] = ("a capturable data-dependent CE needs a SWITCH conditional node per (layer, spare slot, projection) "
                   "with one memcpy case per possible cold expert and a device kernel calling cudaGraphSetConditional "
                   "after the dedup; torch's IF/WHILE helpers alone do not select a source address")
    return cap


class _Setup:
    """The synthetic layer: pools, packed SF, the cold host buffer, per-round routing / dedup / work lists."""

    def __init__(self, a, dev):
        from runtime import dg_inplace as DI
        from runtime import moe_rows as MR

        self.DI, self.MR = DI, MR
        self.dev = dev
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        rng = random.Random(a.seed)
        self.H, self.S, self.R = a.hot, SPARE, a.rounds
        H, S = self.H, self.S
        self.G = H + S
        NC = self.R * CMAX
        self.E = max(384, H + NC + 1)
        self.row_w = NK["w1"][0] * NK["w1"][1] // 2
        self.C = {p: DI.sf_words(NK[p][1] // 32) for p in PROJS}
        self.sf_dims = ((NK["w1"][0], self.C["w1"]), (NK["w2"][0], self.C["w2"]))
        t0 = time.time()
        # hot pools (random int8 bytes = fp4 pairs: e2m1 has no NaN) + e8m0 scales 2^-9..2^-3 (codes 118..124)
        self.full = {p: torch.randint(-128, 128, (self.G, NK[p][0], NK[p][1] // 2), dtype=torch.int8, device=dev)
                     for p in PROJS}
        for p in PROJS:
            self.full[p][H:].zero_()
        pool_s = {p: torch.randint(118, 125, (H, NK[p][0], NK[p][1] // 32), dtype=torch.uint8, device=dev)
                  for p in PROJS}
        self.sf = {}
        for p in PROJS:
            self.sf[p] = DI.packed_sf_zeros(self.G, NK[p][0], self.C[p], dev)
            self.sf[p][:H].copy_(DI.pack_words(pool_s[p]))
        # cold experts H .. H + NC - 1 in host memory: [weight w1|w3|w2, scale w1|w3|w2] per expert, 256-B aligned
        sizes = [self.row_w] * 3 + [NK[p][0] * NK[p][1] // 32 for p in PROJS]
        offs, o = [], 0
        for n in sizes:
            offs.append(o)
            o += (n + 255) // 256 * 256
        self.per_expert = o
        nbytes = NC * o
        self.host_kind = a.host
        self.hb = None
        if a.host == "hostbuffer":
            from runtime.host_tier import HostBuffer

            self.hb = HostBuffer(nbytes, "m3s1_cold", hugepages=True)
            cpu = self.hb.cpu
        else:
            cpu = torch.empty(nbytes, dtype=torch.uint8, device="cpu").pin_memory()
        for i in range(NC):
            base = i * o
            for j, n in enumerate(sizes):
                lo = base + offs[j]
                if j < 3:
                    cpu[lo:lo + n].copy_(torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g, device="cpu"))
                else:
                    cpu[lo:lo + n].copy_(torch.randint(118, 125, (n,), dtype=torch.uint8, generator=g, device="cpu"))
        if self.hb is not None:
            self.reg_s = self.hb.register()
            self.cold_u8 = self.hb.cuda_view(torch.uint8, (nbytes,), 0, dev)
        else:
            self.reg_s = None
            self.cold_u8 = torch.as_tensor(SO._CAI(cpu.data_ptr(), nbytes), device=dev)
        self._cpu_keep = cpu
        base_addr = self.cold_u8.data_ptr()
        # pointer / group tables
        dummy = torch.zeros(self.row_w, dtype=torch.int8, device=dev)
        dummy_s = torch.zeros(NK["w2"][0] * NK["w2"][1] // 32, dtype=torch.uint8, device=dev)
        self._dummies = (dummy, dummy_s)
        ptab = torch.zeros(6, self.E, dtype=torch.int64, device="cpu")
        gtab = torch.full((self.E,), -1, dtype=torch.int32, device="cpu")
        for e in range(self.E):
            for i, p in enumerate(PROJS):
                if e < H:
                    ptab[i, e], ptab[3 + i, e] = self.full[p][e].data_ptr(), pool_s[p][e].data_ptr()
                elif e < H + NC:
                    b = base_addr + (e - H) * o
                    ptab[i, e], ptab[3 + i, e] = b + offs[i], b + offs[3 + i]
                else:
                    ptab[i, e], ptab[3 + i, e] = dummy.data_ptr(), dummy_s.data_ptr()
            if e < H:
                gtab[e] = e
        self._pool_s = pool_s
        self.ptab, self.gtab = ptab.to(dev), gtab.to(dev)
        self.ptab_cpu = ptab
        self.dsts = [self.full[p] for p in PROJS] + [self.sf[p] for p in PROJS]
        self.dtab = torch.tensor([t.data_ptr() for t in self.dsts], dtype=torch.int64, device="cpu").to(dev)
        # rounds: per c, per round r -> routing over UNIQUE experts (UNIQUE - c hot + c cold), the dedup, tags
        self.rounds = {}
        for c in range(CMAX + 1):
            rs = []
            for r in range(self.R):
                cold = [H + r * CMAX + i for i in range(c)]
                hot = rng.sample(range(H), UNIQUE - c)
                idx = torch.tensor(_cover(rng, sorted(hot + cold)), dtype=torch.long, device="cpu")
                w = torch.rand(ROWS, TOPK, generator=g, device="cpu")
                idx_d, w_d = idx.to(dev), w.to(dev)
                m = MR.dedup_routes(idx_d, w_d, ALIGN)
                inpl = {"gtab": self.gtab, "H": H, "S": S, "dummy": torch.tensor(hot[0], device=dev),
                        "has_dummy": True, "v2": True, "layout": "contiguous", "G": self.G, "gscr": self.G,
                        "step_sf": False}
                ext = MR.rows_layout(m, inpl, ALIGN, ROWS, TOPK)
                P = ROWS * TOPK
                mi = torch.full((P * ALIGN,), -1, dtype=torch.int32, device=dev)
                mi.view(P, ALIGN)[:, :ROWS].copy_(ext["tags"])
                # the CE plan: (pool slot, host addresses) of this round's staged cold experts
                plan = [{"grp": H + i, "src": [int(ptab[j, e]) for j in range(3)], "e": e}
                        for i, e in enumerate(cold)]
                for it in plan:
                    it["views"] = [SO.address_bytes(a_, self.row_w, dev) for a_ in it["src"]]
                rs.append({"uniq": m["uniq"], "sidx": ext["sidx"], "wl": ext["wl"], "mi": mi, "arow": ext["arow"],
                           "plan": plan, "cold": cold})
            self.rounds[c] = rs
        # GEMM inputs (fp8 activations with power-of-two scales; one A shared by all rounds) + outputs
        P = ROWS * TOPK
        M = P * ALIGN
        self.M = M
        self.xq = {1: (torch.randn(M, DIM, generator=g, device="cpu") * 3).clamp(-448, 448).to(FP8).to(dev),
                   2: (torch.randn(M, INTER, generator=g, device="cpu") * 3).clamp(-448, 448).to(FP8).to(dev)}
        self.xs = {1: torch.exp2(torch.randint(-6, 7, (M, DIM // 32), generator=g, device="cpu").float()).to(dev),
                   2: torch.exp2(torch.randint(-6, 7, (M, INTER // 32), generator=g, device="cpu").float()).to(dev)}
        self.d = {p: torch.zeros(M, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}
        # shared-expert STAND-IN (bf16 torch.mm with the shared expert's shapes; the real one is fp8 GEMVs)
        self.sx = torch.randn(ROWS, DIM, generator=g, device="cpu").to(torch.bfloat16).to(dev)
        self.sw13 = (torch.randn(2 * INTER, DIM, generator=g, device="cpu") * 0.02).to(torch.bfloat16).to(dev)
        self.sw2 = (torch.randn(DIM, INTER, generator=g, device="cpu") * 0.02).to(torch.bfloat16).to(dev)
        self.load_s = round(time.time() - t0, 1)
        torch.cuda.synchronize()

    # ---- the units
    def stage_kernel(self, rd):
        from runtime import static_kernels as SK

        SK.stage_rows(self.ptab, rd["uniq"], self.gtab, self.dtab, self.dsts, rd["sidx"], rd["wl"], self.H,
                      self.row_w, self.sf_dims, self.S)

    def stage_proj(self, rd, pj, cfg=None, weights=True):
        nprog, bb = _per_proj_cfg(cfg, pj)
        SO.stage_proj(self.ptab, rd["uniq"], self.gtab, self.dtab, self.dsts, rd["sidx"], rd["wl"], self.H,
                      self.row_w, self.sf_dims, self.S, pj, weights=weights, nprog=nprog, block_bytes=bb)

    def ce_copies(self, rd, pj):
        w = self.full[PROJS[pj]]
        for it in rd["plan"]:
            w[it["grp"]].view(torch.uint8).view(-1).copy_(it["views"][pj], non_blocking=True)

    def shared(self):
        h = torch.mm(self.sx, self.sw13.t())
        return torch.mm(h[:, :INTER].contiguous(), self.sw2.t())

    def gemm(self, dg, rd, p):
        w_ = 2 if p == "w2" else 1
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous((self.xq[w_], self.xs[w_]), (self.full[p], self.sf[p]), self.d[p],
                                                rd["mi"], **RECIPE)


def cmd_probe(a) -> dict:
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": a.argv, "errors": [], "label": {
        "A": "[M] synthetic real shapes, graph replay, distinct cold experts per round",
        "B": "[M] real DeepGEMM contiguous16 + a bf16 torch.mm shared-expert STAND-IN",
        "E": "[E] measured per-c exposed time x Poisson cold distribution [A]"}}

    def step(name, fn):
        try:
            r = fn()
            out[name] = r
            print("OK  ", name, json.dumps(r, default=str)[:600], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            out[name] = {"error": tb[-2000:]}
            out["errors"].append(name)
            print("FAIL", name, tb[-2000:], flush=True)
            return None

    out["capability"] = _capability()
    if not torch.cuda.is_available():
        out["errors"].append("no CUDA")
        return out
    dev = torch.device("cuda")
    st = _Setup(a, dev)
    out["setup"] = {"H": st.H, "S": st.S, "rounds": st.R, "host": st.host_kind, "register_s": st.reg_s,
                    "expert_bytes": SO.EXPERT_BYTES, "row_w": st.row_w, "load_s": st.load_s,
                    "cold_footprint_GB": round(st.R * CMAX * st.per_expert / 1e9, 2)}
    reps = a.reps
    R = st.R
    streams = [torch.cuda.Stream() for _ in range(4)]
    side = torch.cuda.Stream()
    ev_r = [torch.cuda.Event() for _ in range(R)]
    ev_p = [[torch.cuda.Event() for _ in range(3)] for _ in range(R)]
    ev_s = [[torch.cuda.Event() for _ in range(4)] for _ in range(R)]
    nb = SO.EXPERT_BYTES

    def per_round_us(body_round):
        def body():
            for r in range(R):
                body_round(r)
        return _graph_ms(body, reps) * 1e3 / R

    # ---------------------------------------------------------------- A bandwidth
    def ce_multi(c, r, k, weights_only=False):
        rd = st.rounds[c][r]
        cur = torch.cuda.current_stream()
        ev_r[r].record(cur)
        jobs = [(pj, it) for pj in range(3) for it in rd["plan"]]
        n = max(1, min(k, len(jobs)))
        for i in range(n):
            streams[i].wait_event(ev_r[r])
            with torch.cuda.stream(streams[i]):
                for pj, it in jobs[i::n]:
                    w = st.full[PROJS[pj]]
                    w[it["grp"]].view(torch.uint8).view(-1).copy_(it["views"][pj], non_blocking=True)
                ev_s[r][i].record(streams[i])
        for i in range(n):
            cur.wait_event(ev_s[r][i])
        if not weights_only:
            for pj in range(3):
                st.stage_proj(rd, pj, weights=False)

    def bandwidth():
        res = {}
        for c in range(CMAX + 1):
            row = {}
            row["kernel_m1_stage_rows"] = per_round_us(lambda r, c=c: st.stage_kernel(st.rounds[c][r]))
            row["kernel_s1_proj_x3"] = per_round_us(lambda r, c=c: [st.stage_proj(st.rounds[c][r], pj)
                                                                       for pj in range(3)])
            for cfg in a.cfgs:
                row[_key("kernel_s1_proj_x3", cfg)] = per_round_us(
                    lambda r, c=c, cfg=cfg: [st.stage_proj(st.rounds[c][r], pj, cfg=cfg) for pj in range(3)])
            for k in (1, 2, 4):
                row[f"ce_{k}stream"] = per_round_us(lambda r, c=c, k=k: ce_multi(c, r, k))
            row["ce_weights_only_1stream"] = per_round_us(lambda r, c=c: ce_multi(c, r, 1, weights_only=True))
            row["scale_pack_x3"] = per_round_us(lambda r, c=c: [st.stage_proj(st.rounds[c][r], pj, weights=False)
                                                                   for pj in range(3)])
            res[c] = {k_: {"us": round(v, 2), "GBps": round(c * nb / (v * 1e3), 1) if c else None}
                      for k_, v in row.items()}
            print(f"  [A] c={c}: " + ", ".join(f"{k_} {v:.1f}us" for k_, v in row.items()), flush=True)
        return res

    step("bandwidth", bandwidth)

    # ---------------------------------------------------------------- B overlap with the real GEMM chain
    from runtime.dg_moe import import_deep_gemm

    dg = import_deep_gemm()
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()
    prev_align = dg.get_mk_alignment_for_contiguous_layout()
    dg.set_mk_alignment_for_contiguous_layout(ALIGN)

    def layer(c, r, variant, cfg=None, timing=None):
        rd = st.rounds[c][r]
        cur = torch.cuda.current_stream()
        forked = variant in ("kernel-overlap", "ce-overlap")
        if variant == "serial":
            st.stage_kernel(rd)
        if forked:
            ev_r[r].record(cur)
            side.wait_event(ev_r[r])
            with torch.cuda.stream(side):
                for pj in range(3):
                    if variant == "ce-overlap":
                        st.ce_copies(rd, pj)
                        st.stage_proj(rd, pj, weights=False)
                    else:
                        st.stage_proj(rd, pj, cfg=cfg)
                    ev_p[r][pj].record(side)
        st.shared()
        for pj, p in enumerate(PROJS):
            if forked:
                cur.wait_event(ev_p[r][pj])
            if timing is not None:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(cur)
                st.gemm(dg, rd, p)
                e1.record(cur)
                timing.append((p, e0, e1))
            else:
                st.gemm(dg, rd, p)

    variants = [("gemm-only", None), ("serial", None)] + [("kernel-overlap", cfg) for cfg in a.cfgs] + \
               [("ce-overlap", None)]

    def overlap():
        res = {}
        for c in range(CMAX + 1):
            row = {}
            for v, n in variants:
                key = _key(v, n)
                row[key] = per_round_us(lambda r, c=c, v=v, n=n: layer(c, r, v, n))
            base = row["gemm-only"]
            res[c] = {k_: {"us_per_layer": round(x, 2), "exposed_us": round(x - base, 2)} for k_, x in row.items()}
            print(f"  [B] c={c}: " + ", ".join(f"{k_} {x:.1f} (+{x - base:.1f})" for k_, x in row.items()), flush=True)
        return res

    ov = step("overlap", overlap)

    def contention():
        res = {}
        for c in (0, 2, 4):
            row = {}
            for v, n in [("gemm-only", None)] + [("kernel-overlap", cfg) for cfg in a.cfgs] + [("ce-overlap", None)]:
                key = _key(v, n)
                timing = []
                for _ in range(max(1, reps // 20)):
                    for r in range(R):
                        layer(c, r, v, n, timing)
                torch.cuda.synchronize()
                per = {p: [] for p in PROJS}
                for p, e0, e1 in timing:
                    per[p].append(e0.elapsed_time(e1) * 1e3)
                row[key] = {p: round(sum(x) / len(x), 2) for p, x in per.items()}
            res[c] = row
        return {"eager_gemm_us_per_call": res, "label": "[M] eager CUDA events around each GEMM on the main stream "
                "(after its wait): the GEMM's own duration alone vs with the forked staging running"}

    step("contention", contention)

    # ---------------------------------------------------------------- C bytes / outputs equal across variants
    def checks():
        res = {}
        c = CMAX
        ref = None
        for v, n in [("serial", None)] + [("kernel-overlap", cfg) for cfg in a.cfgs] + [("ce-overlap", None)]:
            for p in PROJS:
                st.full[p][st.H:].zero_()
                st.sf[p][st.H:].zero_()
            for x in st.d.values():
                x.zero_()
            layer(c, 0, v, n)
            torch.cuda.synchronize()
            got = {"slots": [st.full[p][st.H:].clone() for p in PROJS],
                   "sf": [st.sf[p][st.H:].clone() for p in PROJS],
                   "d": [st.d[p].index_select(0, st.rounds[c][0]["arow"]).clone() for p in PROJS]}
            if ref is None:
                ref = got
                # the staged bytes are the host bytes
                ok = True
                for i, it in enumerate(st.rounds[c][0]["plan"]):
                    for pj in range(3):
                        ok &= bool(torch.equal(st.full[PROJS[pj]][it["grp"]].view(torch.uint8).view(-1),
                                               it["views"][pj]))
                res["serial_staged_equal_host"] = ok
                continue
            key = _key(v, n)
            res[key] = {k_: all(torch.equal(x, y) for x, y in zip(got[k_], ref[k_])) for k_ in ("slots", "sf", "d")}
        return res

    step("checks", checks)
    dg.set_mk_alignment_for_contiguous_layout(prev_align)

    # ---------------------------------------------------------------- E model with the measured per-c costs
    if isinstance(ov, dict) and "error" not in ov:
        out["model_measured"] = model_from_measured(ov, a.cold_mean)
        for k_, v in out["model_measured"]["ms_per_step"].items():
            print(f"  [E] {k_:28s} staging exposed {v:6.3f} ms/step -> step {out['model_measured']['step_ms_E'][k_]:6.2f}")
    return out


def model_from_measured(ov: dict, cold_mean: float, layers: int = 40, step_ms: float = 20.304,
                        serial_ms: float = 4.552) -> dict:
    """[E] Poisson(cold_mean) over the measured exposed_us per c (c > CMAX: linear in c from the last two points)."""
    d = SO.poisson(cold_mean, 12)
    keys = [k for k in ov[0] if k != "gemm-only"]
    res, step_e = {}, {}
    for k in keys:
        pts = [float(ov[c][k]["exposed_us"]) for c in range(CMAX + 1)]
        slope = pts[-1] - pts[-2]
        val = lambda c: pts[c] if c <= CMAX else pts[-1] + slope * (c - CMAX)  # noqa: E731
        ms = layers * sum(pc * max(0.0, val(c)) for c, pc in enumerate(d)) / 1e3
        res[k] = round(ms, 3)
    if "serial" in res and res["serial"] > 0:
        for k, v in res.items():
            step_e[k] = round(step_ms - serial_ms + v * serial_ms / res["serial"], 3)
    return {"ms_per_step": res, "step_ms_E": step_e, "cold_mean": cold_mean, "dist": [round(x, 4) for x in d],
            "label": "[E] serial row is the synthetic probe's own; step_ms_E scales every row by the measured "
                     f"{serial_ms} ms / probe-serial so the ledger's staging line is the calibration"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("model", "probe", "all"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cold-mean", type=float, default=SO.MODEL_DEFAULTS["cold_mean"])
    ap.add_argument("--kernel-gbps", type=float, default=SO.MODEL_DEFAULTS["kernel_GBps"])
    ap.add_argument("--ce-gbps", type=float, default=SO.MODEL_DEFAULTS["ce_GBps"])
    ap.add_argument("--contention", type=float, default=0.0)
    ap.add_argument("--hot", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cfgs", default="48:8192,16:32768,8:65536,48:8192/16:32768/16:32768,"
                                      "144:8192/16:32768/16:32768",
                    help="kernel-overlap nprog:block_bytes pairs; 'n:b/n:b/n:b' = one per projection (w1/w3/w2)")
    ap.add_argument("--host", choices=("hostbuffer", "pinned"), default="hostbuffer")
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    a.cfgs = parse_cfgs(a.cfgs)
    out = {}
    if a.cmd in ("probe", "all"):
        out["probe"] = cmd_probe(a)
    if a.cmd in ("model", "all"):
        out["model"] = cmd_model(a)
    path = a.out or os.path.join(REPO, "scratch", f"staging-overlap-{a.cmd}-{_date()}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)
    errs = out.get("probe", {}).get("errors", [])
    return 2 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
