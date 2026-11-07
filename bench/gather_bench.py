#!/usr/bin/env python3
"""gather_bench.py -- T9b-3a Task B: the MoE staging gather at HBM rate. GPU only (the library image).

  python3 bench/gather_bench.py --model-dir /ref [--layer 0] [--hot 36] [--cold 12] [--iters 50] [--json out.json]
  python3 bench/gather_bench.py --synthetic ...      # random bytes at the real row sizes (no checkpoint)

One layer's real expert rows (`layers.<L>.ffn.experts.<e>.{w1,w3,w2}.{weight,scale}` read from the checkpoint; or
random bytes of the same sizes): `--hot` experts in HBM pool tensors, `--cold` in ONE cudaHostRegister'd host buffer
seen through UVA (runtime/host_tier.py, the expert pool's layout: 256-B aligned blocks), pointer tables as the static
step builds them. M = 6 routed ids per call (k of the real model), the hot case and the cold-over-C2C case separately;
the id set rotates over `--sets` disjoint-as-possible sets per iteration so the 113 MB a call moves is not L2-resident.

Arms (runtime/static_kernels.py): the T8/T9b-2 byte kernels (`u8`: 6 x t8_gather_rows_u8 = the ref glue path,
t9b2_gather_tables_u8 = R1's fused one-launch gather) vs the 16-B vector kernels (`vec`), scale rows as bytes and as
dg-masked's fp32 decode (`f32`); plus an HBM->HBM `copy_` of the same byte count (the copy-engine/SM ceiling,
C2C-PROBE: 0.341 ms/GiB = 3.15 TB/s copied). Every arm's staging is `torch.equal` to the u8 arm's for the same ids
before anything is timed. Rates are bytes COPIED (= bytes read from the source rows) per second; HBM traffic is 2x.
A sweep over vec block bytes x num_warps picks the best hot config. Gate printed (orchestrator): best hot one-launch
vec gather >= 3 TB/s. Nothing here touches numerics (data movement only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

PROJS = ("w1", "w3", "w2")  # gather_tables order: weights w1, w3, w2, then their scales
K_IDS = 6
GATE_TBPS = 3.0


def real_shapes(dim: int = 5120, inter: int = 2304) -> Dict[str, tuple]:
    """(weight [n, k/2] int8, scale [n, k/32] e8m0) per projection, DSV4.1-Flash (runtime/expert_pool.py proj_nk)."""
    nk = {"w1": (inter, dim), "w3": (inter, dim), "w2": (dim, inter)}
    return {p: ((n, k // 2), (n, k // 32)) for p, (n, k) in nk.items()}


def _cpu_default(fn):
    """CPU default device for the host-side builders (a test may run with the default device set to cuda)."""
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        with torch.device("cpu"):
            return fn(*a, **k)

    return w


class Tables:
    """Hot rows in HBM, cold rows in one registered host buffer, pointer tables, staging (static_step layout)."""

    @_cpu_default
    def __init__(self, shapes, hot: int, cold: int, device="cuda", index=None, layer: int = 0, seed: int = 0):
        from runtime.host_tier import HostBuffer

        self.shapes, self.hot, self.cold, self.dev = shapes, hot, cold, torch.device(device)
        E = hot + cold
        self.E = E
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.row_bytes = {(p, kind): shapes[p][0 if kind == "weight" else 1][0] * shapes[p][0 if kind == "weight"
                                                                                      else 1][1]
                          for p in PROJS for kind in ("weight", "scale")}
        rw = {self.row_bytes[(p, "weight")] for p in PROJS}
        rs = {self.row_bytes[(p, "scale")] for p in PROJS}
        assert len(rw) == 1 and len(rs) == 1, (rw, rs)  # one-launch gather needs equal row sizes (true for V4.1)
        self.row_w, self.row_s = rw.pop(), rs.pop()
        for k_, v in self.row_bytes.items():
            assert v % 16 == 0, (k_, v)  # 16-B vector gather precondition (handoff: assert at build)

        def src_name(e, p, kind):
            return f"layers.{layer}.ffn.experts.{e}.{p}.{kind}"

        def fill(dst_cpu: torch.Tensor, e, p, kind):  # dst_cpu: a uint8 CPU view of one expert row
            if index is not None:
                index.read_into(src_name(e, p, kind), dst_cpu)
            else:
                dst_cpu.copy_(torch.randint(0, 256, tuple(dst_cpu.shape), dtype=torch.uint8, generator=g))

        # hot: [hot, n, kh] per (proj, kind) on the device
        self.hbm = {}
        for p in PROJS:
            for kind in ("weight", "scale"):
                shp = shapes[p][0 if kind == "weight" else 1]
                t = torch.empty(hot, *shp, dtype=torch.uint8)
                for e in range(hot):
                    fill(t[e], e, p, kind)
                self.hbm[(p, kind)] = t.to(self.dev)
        # cold: one host buffer, 256-B aligned blocks (expert_pool._align)
        self.host = None
        self.cold_views = {}
        if cold:
            off, layout = 0, {}
            for p in PROJS:
                for kind in ("weight", "scale"):
                    layout[(p, kind)] = off
                    off = (off + cold * self.row_bytes[(p, kind)] + 255) // 256 * 256
            self.host = HostBuffer(off, "gather_bench_cold")
            self.host.register()
            for (p, kind), base in layout.items():
                shp = shapes[p][0 if kind == "weight" else 1]
                for c in range(cold):
                    o = base + c * self.row_bytes[(p, kind)]
                    fill(self.host.host_view(torch.uint8, shp, o), hot + c, p, kind)
                self.cold_views[(p, kind)] = [self.host.cuda_view(torch.uint8, shp,
                                                                  base + c * self.row_bytes[(p, kind)], self.dev)
                                              for c in range(cold)]
        # pointer tables: expert e -> its row address (hot slot e, cold slot e - hot)
        self.ptabs = {}
        for key in self.row_bytes:
            ptrs = [self.hbm[key][e].data_ptr() for e in range(hot)] + [v.data_ptr() for v in self.cold_views.get(
                key, [])]
            assert all(x % 16 == 0 for x in ptrs), key
            self.ptabs[key] = torch.tensor(ptrs, dtype=torch.int64, device=self.dev)
        self.ptab6 = torch.stack([self.ptabs[(p, "weight")] for p in PROJS] +
                                 [self.ptabs[(p, "scale")] for p in PROJS])
        # staging: as StaticStep (int8 weights / uint8 scales) + dg's fp32 scale buffers
        self.staging = {}
        for p in PROJS:
            self.staging[(p, "weight")] = torch.zeros(K_IDS, *shapes[p][0], dtype=torch.int8, device=self.dev)
            self.staging[(p, "scale")] = torch.zeros(K_IDS, *shapes[p][1], dtype=torch.uint8, device=self.dev)
        self.sf32 = {p: torch.zeros(K_IDS, *shapes[p][1], dtype=torch.float32, device=self.dev) for p in PROJS}
        self.dsts_u8 = [self.staging[(p, "weight")] for p in PROJS] + [self.staging[(p, "scale")] for p in PROJS]
        self.dsts_f32 = [self.staging[(p, "weight")] for p in PROJS] + [self.sf32[p] for p in PROJS]
        self.dtab_u8 = torch.tensor([t.data_ptr() for t in self.dsts_u8], dtype=torch.int64, device=self.dev)
        self.dtab_f32 = torch.tensor([t.data_ptr() for t in self.dsts_f32], dtype=torch.int64, device=self.dev)
        self.ids_out = torch.zeros(K_IDS, dtype=torch.long, device=self.dev)

    @property
    def bytes_per_call(self) -> int:
        return K_IDS * 3 * (self.row_w + self.row_s)

    @_cpu_default
    def id_sets(self, case: str, n_sets: int, seed: int = 1) -> List[torch.Tensor]:
        pool = list(range(self.hot)) if case == "hot" else list(range(self.hot, self.E))
        if len(pool) < K_IDS:
            raise ValueError(f"case {case}: {len(pool)} experts < {K_IDS} ids")
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = []
        for _ in range(n_sets):
            pick = torch.tensor(pool)[torch.randperm(len(pool), generator=g)[:K_IDS]].sort().values
            out.append(pick.to(self.dev))
        return out

    # ------------------------------------------------------------------ arms
    def run(self, arm: str, ids: torch.Tensor, block_bytes=None, num_warps=None) -> None:
        from runtime import static_kernels as SK

        if arm in ("rows-u8", "rows-vec"):
            impl = arm.split("-")[1]
            for key, tab in self.ptabs.items():
                SK.gather_rows(tab, ids, self.staging[key], self.row_bytes[key], impl=impl,
                               block_bytes=block_bytes, num_warps=num_warps)
        elif arm in ("tables-u8", "tables-vec", "tables-f32-u8", "tables-f32-vec"):
            f32 = "f32" in arm
            impl = arm.rsplit("-", 1)[1]
            SK.gather_tables(self.ptab6, ids, self.dtab_f32 if f32 else self.dtab_u8,
                             self.dsts_f32 if f32 else self.dsts_u8, self.row_w, self.row_s, f32, self.ids_out,
                             impl=impl, block_bytes=block_bytes, num_warps=num_warps)
        else:
            raise ValueError(arm)

    def snapshot(self, f32: bool) -> List[torch.Tensor]:
        return [t.clone() for t in (self.dsts_f32 if f32 else self.dsts_u8)]

    def zero(self) -> None:
        for t in self.dsts_u8 + self.dsts_f32:
            t.zero_()
        self.ids_out.zero_()

    def verify(self, ids_list, arms=("rows-vec", "tables-u8", "tables-vec"), f32_arms=("tables-f32-vec",),
               configs=((None, None),)) -> Dict[str, str]:
        """Every arm/config's staging (and ids_out) == the u8 reference arm's, bitwise, for every id set."""
        res = {}

        def eq(a, b):
            return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))

        for ids in ids_list:
            self.zero()
            self.run("rows-u8", ids)
            want = self.snapshot(False)
            self.zero()
            self.run("tables-f32-u8", ids)
            want32 = self.snapshot(True)
            for bb, nw in configs:
                for arm in arms:
                    self.zero()
                    self.run(arm, ids, bb, nw)
                    ok = eq(self.snapshot(False), want) and (arm.startswith("rows") or torch.equal(self.ids_out, ids))
                    tag = f"{arm}@{bb or 'def'}/{nw or 'def'}"
                    res[tag] = res.get(tag, "equal") if ok else "MISMATCH"
                for arm in f32_arms:
                    self.zero()
                    self.run(arm, ids, bb, nw)
                    ok = eq(self.snapshot(True), want32) and torch.equal(self.ids_out, ids)
                    tag = f"{arm}@{bb or 'def'}/{nw or 'def'}"
                    res[tag] = res.get(tag, "equal") if ok else "MISMATCH"
        torch.cuda.synchronize(self.dev)
        return res

    def time(self, arm, ids_list, iters, block_bytes=None, num_warps=None, warm=3) -> float:
        """Mean µs per layer-call (all six tables of one id set), cycling through ids_list."""
        for i in range(warm):
            self.run(arm, ids_list[i % len(ids_list)], block_bytes, num_warps)
        torch.cuda.synchronize(self.dev)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for i in range(iters):
            self.run(arm, ids_list[i % len(ids_list)], block_bytes, num_warps)
        b.record()
        torch.cuda.synchronize(self.dev)
        return a.elapsed_time(b) * 1e3 / iters

    def release(self) -> None:
        self.cold_views = {}
        self.ptabs = {}
        self.ptab6 = None
        if self.host is not None:
            torch.cuda.synchronize(self.dev)
            self.host.release()


def copy_ceiling_us(nbytes: int, iters: int, dev) -> float:
    """HBM->HBM copy_ of `nbytes` (rotating over 4 sources so the source is not L2-resident)."""
    srcs = [torch.empty(nbytes, dtype=torch.uint8, device=dev) for _ in range(4)]
    dst = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    for s in srcs:
        dst.copy_(s)
    torch.cuda.synchronize(dev)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters):
        dst.copy_(srcs[i % 4])
    b.record()
    torch.cuda.synchronize(dev)
    return a.elapsed_time(b) * 1e3 / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", default="/ref")
    ap.add_argument("--synthetic", action="store_true", help="random bytes at the real row sizes (no checkpoint)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--hot", type=int, default=36, help="experts in HBM (36 x 18.8 MB: the id sets miss L2)")
    ap.add_argument("--cold", type=int, default=12, help="experts in registered host memory (UVA over C2C)")
    ap.add_argument("--sets", type=int, default=6, help="id sets cycled per iteration")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--blocks", default="4096,8192,16384,32768", help="vec block bytes swept")
    ap.add_argument("--warps", default="4,8", help="vec num_warps swept")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("gather_bench: needs a CUDA device (the box)", file=sys.stderr)
        return 2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dev = torch.device("cuda")
    import triton

    index = None
    if not a.synthetic:
        from runtime.ckpt import ShardIndex

        index = ShardIndex.scan(a.model_dir)
    t0 = time.time()
    tb = Tables(real_shapes(), a.hot, a.cold, dev, index=index, layer=a.layer)
    load_s = time.time() - t0
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "device": torch.cuda.get_device_name(), "triton": triton.__version__, "torch": torch.__version__,
           "source": "synthetic" if a.synthetic else f"{a.model_dir} layer {a.layer}", "load_s": round(load_s, 1),
           "hot": a.hot, "cold": a.cold, "ids_per_call": K_IDS, "row_bytes": {"weight": tb.row_w, "scale": tb.row_s},
           "bytes_per_call": tb.bytes_per_call, "cases": {}}
    configs = [(int(b), int(w)) for b in a.blocks.split(",") for w in a.warps.split(",")]
    sets = {"hot": tb.id_sets("hot", a.sets)}
    if a.cold >= K_IDS:
        sets["cold"] = tb.id_sets("cold", a.sets, seed=2)
    out["verify"] = tb.verify([s for v in sets.values() for s in v[:2]], configs=[(None, None)] + configs)
    bad = [k for k, v in out["verify"].items() if v != "equal"]
    print(f"[verify] {len(out['verify']) - len(bad)}/{len(out['verify'])} arm/config staging == u8 bitwise"
          + (f"; MISMATCH: {bad}" if bad else ""), flush=True)
    gb = lambda us: tb.bytes_per_call / (us * 1e-6) / 1e9  # noqa: E731
    for case, ids_list in sets.items():
        res = {}
        for arm in ("rows-u8", "rows-vec", "tables-u8", "tables-vec", "tables-f32-u8", "tables-f32-vec"):
            us = tb.time(arm, ids_list, a.iters)
            res[arm] = {"us": round(us, 2), "GBps": round(gb(us), 1)}
        sweep = {}
        for bb, nw in configs:
            us = tb.time("tables-vec", ids_list, a.iters, bb, nw)
            sweep[f"{bb}/{nw}"] = {"us": round(us, 2), "GBps": round(gb(us), 1)}
        best = min(sweep, key=lambda k: sweep[k]["us"])
        res["tables-vec sweep (block bytes/warps)"] = sweep
        res["best"] = {"config": best, **sweep[best]}
        out["cases"][case] = res
        print(f"[{case}] µs per layer-call ({tb.bytes_per_call / 1e6:.1f} MB copied), GB/s copied:", flush=True)
        for arm in ("rows-u8", "rows-vec", "tables-u8", "tables-vec", "tables-f32-u8", "tables-f32-vec"):
            print(f"  {arm:16s} {res[arm]['us']:9.2f} µs  {res[arm]['GBps']:8.1f} GB/s", flush=True)
        print(f"  sweep tables-vec: " + ", ".join(f"{k} {v['GBps']:.0f}" for k, v in sweep.items())
              + f"  -> best {best} {sweep[best]['GBps']:.1f} GB/s", flush=True)
    us = copy_ceiling_us(tb.bytes_per_call, a.iters, dev)
    out["hbm_copy_ceiling"] = {"us": round(us, 2), "GBps": round(gb(us), 1)}
    print(f"[ceiling] HBM->HBM copy_ of the same bytes: {us:.2f} µs, {gb(us):.1f} GB/s copied", flush=True)
    hot_best = out["cases"]["hot"]["best"]["GBps"]
    out["gate"] = {"hot_best_GBps": hot_best, "bar_GBps": GATE_TBPS * 1e3,
                   "pass": hot_best >= GATE_TBPS * 1e3 and not bad}
    print(f"GATHER_BENCH hot best {hot_best:.1f} GB/s (bar {GATE_TBPS * 1e3:.0f}), verify "
          f"{'equal' if not bad else 'MISMATCH'}: {'PASS' if out['gate']['pass'] else 'FAIL'}", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")
    tb.release()
    if index is not None:
        index.close()
    return 0 if out["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
