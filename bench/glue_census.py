#!/usr/bin/env python3
"""glue_census.py -- T9b-1 glue audit instrument (CPU, Mac): every torch op one 1-row decode step dispatches, attributed
to layer type and code region, for the three decode paths. Feeds docs/GLUE-AUDIT.md (the T9b-2 fusion plan).

  .venv/bin/python bench/glue_census.py [--json scratch/glue-census.json] [--steps 2]

Model: a synthetic checkpoint with the REAL model's layer structure (40 layers; compress ratios 0/2/1 by layer;
kv sources 2,8,14,20; index sources 2,8,14,20,24,28,32,36; candidate source 20; Engram at 1 and 14; 6 of 8 routed
experts active; hc_mult 4, 20 Sinkhorn iterations; 16 heads so sparse_attn takes no head padding, like the real 64)
and tiny dims, built by tests/mini_ckpt.py and loaded by runtime/loader.py on CPU. Op counts per layer do not depend
on dims (no data-dependent loops on the decode path except the reference MoE's per-expert loop, which runs the k
selected experts either way).

Counting: a TorchDispatchMode records every aten op the step dispatches. The reference TileLang kernels, the T8 Triton
gather/marker, DeepGEMM and the T9b-1 GEMV are CPU emulations (tests/test_static_emulated_cpu.py and friends) whose
inner ops are NOT counted; each call counts as one kernel ("K:<name>") plus the torch ops its kernel.py wrapper really
runs (cited per kernel below: allocations, `x.copy_(y)` of the in-place QDQ, head padding). Classes:
  launch   an aten op that launches >= 1 CUDA kernel (elementwise, reduction, index, GEMM, copy, factory+fill, ...)
  kernel   a reference/Triton/DeepGEMM kernel call (one launch)
  alloc    empty/empty_like/new_empty: no launch, unless deterministic fill_uninitialized_memory is on (then 1 fill)
  view     metadata only (OpOverload.is_view and friends): no launch
  sync     a device->host scalar read
Caveats: aten-op -> CUDA-kernel is 1:1 except sort/topk/nonzero/bincount/index_put_(deterministic) which launch several
(listed separately as `multi`); CPU and CUDA dispatch the same aten ops on these paths. The anatomy
(docs/STEP-ANATOMY-M0-20261106.md) counted the eager reference path with fills ON.

Paths: eager-ref = the reference `model(tok, pos)` decode (T5, the anatomy's path); static-ref = T8's StaticStep
(moe_impl ref, fp8_impl ref); static-dg-gemv = T9a-1 + T9b-1 (moe_impl dg-masked, fp8_impl gemv), today's step;
static-dg-gemv-fused = the same with T9b-2's `glue_impl="fused"` (runtime/fused_glue.py; each fused kernel counts as
one launch). With both of the last two, the report adds `t9b2`: launches removed per cluster (by the code regions each
cluster replaces) next to docs/GLUE-AUDIT.md's prediction, and prints that table.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import os
import sys
import tempfile
import time
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for p in (REPO, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

REAL_RATIOS = [0, 0] + [2] * 18 + [1] * 20
ALLOC = {"empty", "empty_like", "new_empty", "empty_strided", "new_empty_strided"}
FREE = {"resize_", "lift_fresh", "lift_fresh_copy", "_unsafe_view", "set_", "sym_size", "sym_stride",
        "scalar_tensor"}  # scalar_tensor: a CPU 0-dim wrapper for a Python scalar (torch.where(c, -1, x)): no launch
SYNC = {"_local_scalar_dense", "item", "is_nonzero", "equal"}
MULTI = {"sort", "topk", "nonzero", "bincount", "index_put_", "index_put", "_index_put_impl_", "unique",
         "_unique2", "kthvalue", "median", "argsort", "msort"}
# T9b-2: which code regions each cluster replaces (bench/glue_census.py tags them; innermost region wins) and the
# audit's predicted saving per step (docs/GLUE-AUDIT.md, cluster table). "moe" (DGMaskedMoE.__call__ itself) holds
# the routed SwiGLU's two `.float()`s of the w1/w3 GEMM outputs; the rest of it (gate, sort, index) is not touched.
CLUSTER_REGIONS = {
    "#2 SwiGLU+act_quant (routed+shared)": ("moe.swiglu", "moe", "expert", "expert/linear"),
    "#7 mHC post": ("mhc.post",),
    "#8 MoE combine": ("moe.combine",),
    "#9 DG adapter copies": ("dg.load_act", "dg.convert", "moe.ids", "moe.act"),
    "#10 o path inverse RoPE + QDQ": ("attn.o_path", "attn.o_path/rope"),
    "#11 one gather for 6 tables": ("moe.gather",),
    "#13 q RoPE in wq_b GEMV epilogue": ("attn.q_proj", "attn.q_proj/linear", "attn.q_proj/rope"),
    "#16 concat-free attention inputs": (),
}
AUDIT_PREDICTED = {"#2 SwiGLU+act_quant (routed+shared)": 600, "#7 mHC post": 320, "#8 MoE combine": 320,
                   "#9 DG adapter copies": 320, "#10 o path inverse RoPE + QDQ": 200,
                   "#11 one gather for 6 tables": 200, "#13 q RoPE in wq_b GEMV epilogue": 120,
                   "#16 concat-free attention inputs": 76}


def real_structure_config():
    from mini_ckpt import MINI_CONFIG

    cfg = dict(MINI_CONFIG)
    cfg.update({
        "n_layers": 40, "n_mtp_layers": 1, "compress_ratios": REAL_RATIOS + [0],
        "kv_source_layers": [2, 8, 14, 20], "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8, "index_topk": 512,
        "n_routed_experts": 8, "n_activated_experts": 6, "n_heads": 16, "o_groups": 8, "o_lora_rank": 64,
        "head_dim": 128, "rope_head_dim": 64, "dim": 256, "q_lora_rank": 128, "moe_inter_dim": 128,
        "engram_layer_ids": [1, 14], "engram_num_embeddings": [1000, 1000], "engram_n_heads": 8,
        "engram_head_dim": 64, "engram_max_ngram_size": 4, "dspark_target_layer_ids": [37, 38, 39],
        "hc_mult": 4, "hc_sinkhorn_iters": 20,
    })
    hot = {str(L): [0, 2, 5, 7] for L in range(40)}
    return cfg, hot


# ------------------------------------------------------------------------------------------------ the census
class Census(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.events: List[tuple] = []
        self.paused = 0
        self.stack: List[str] = []
        self.on = False

    def add(self, cls: str, name: str):
        if self.on and not self.paused:
            self.events.append((tuple(self.stack), cls, name))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.on and not self.paused:
            name = func.overloadpacket.__name__
            if getattr(func, "is_view", False) or name in FREE:
                cls = "view"
            elif name in ALLOC:
                cls = "alloc"
            elif name in SYNC:
                cls = "sync"
            else:
                cls = "launch"
            self.events.append((tuple(self.stack), cls, name))
        return func(*args, **(kwargs or {}))


C = Census()


@contextlib.contextmanager
def region(name: str):
    C.stack.append(name)
    try:
        yield
    finally:
        C.stack.pop()


def tagged(fn, name):
    @functools.wraps(fn)
    def w(*a, **k):
        with region(name if not callable(name) else name(*a, **k)):
            return fn(*a, **k)
    return w


def kernel(fn, kname, wrapper_ops):
    """Count one kernel + the kernel.py wrapper's own ops, run the CPU emulation uncounted."""
    @functools.wraps(fn)
    def w(*a, **k):
        for cls, name in wrapper_ops(*a, **k):
            C.add(cls, name)
        C.add("kernel", "K:" + kname)
        C.paused += 1
        try:
            return fn(*a, **k)
        finally:
            C.paused -= 1
    return w


def _quant_ops(x, *a, **k):
    """kernel.py:98-124 / 184-204: z = x.contiguous(); y = empty_like; s = new_empty; kernel; inplace: x.copy_(y)."""
    inplace = k.get("inplace", a[-1] if a and isinstance(a[-1], bool) else False)
    ops = [] if x.is_contiguous() else [("launch", "clone")]
    ops += [("alloc", "empty_like"), ("alloc", "new_empty")]
    return ops + ([("launch", "copy_[inplace QDQ write-back]")] if inplace else [])


def _sparse_attn_ops(q, *a, **k):
    """kernel.py:392-403: pad heads to 16 (cat + new_zeros, twice) when h < 16; empty_like; kernel; narrow+contiguous."""
    pad = q.size(2) < 16
    ops = [("launch", "new_zeros"), ("launch", "cat"), ("launch", "new_zeros"), ("launch", "cat")] if pad else []
    return ops + [("alloc", "empty_like")] + ([("launch", "contiguous")] if pad else [])


def install(R):
    """Wrap the emulated kernels (counted as kernels) and tag regions; returns a restore function."""
    from test_static_emulated_cpu import KERNELS
    from test_dg_moe_cpu import _faithful_act_quant
    from runtime import gemv_mxfp8, static_step

    saved = []

    def setattr_saved(obj, attr, val):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)

    emu = dict(KERNELS)
    emu["act_quant"] = _faithful_act_quant
    setattr_saved(R, "act_quant", kernel(emu["act_quant"], "act_quant", _quant_ops))
    setattr_saved(R, "fp4_act_quant", kernel(emu["fp4_act_quant"], "fp4_act_quant", _quant_ops))
    setattr_saved(R, "fp8_gemm", kernel(emu["fp8_gemm"], "fp8_gemm", lambda *a, **k: [("alloc", "new_empty")]))
    setattr_saved(R, "fp4_gemm", kernel(emu["fp4_gemm"], "fp4_gemm", lambda *a, **k: [("alloc", "new_empty")]))
    setattr_saved(R, "hc_split_sinkhorn", kernel(emu["hc_split_sinkhorn"], "hc_split_sinkhorn",
                                                 lambda *a, **k: [("alloc", "new_empty")] * 3))
    setattr_saved(R, "sparse_attn", kernel(emu["sparse_attn"], "sparse_attn", _sparse_attn_ops))
    # T9b-1 GEMV: linear_fp8 = torch.empty + one launch (runtime/gemv_mxfp8.py)
    setattr_saved(gemv_mxfp8, "linear_fp8", kernel(gemv_mxfp8.linear_fp8, "mxfp8_gemv",
                                                   lambda *a, **k: [("alloc", "empty")]))
    # T9b-2 fused glue: one launch each (runtime/fused_glue.py launch_*; the GEMV with the RoPE epilogue)
    from runtime import fused_glue

    setattr_saved(gemv_mxfp8, "linear_fp8_rope", kernel(gemv_mxfp8.linear_fp8_rope, "mxfp8_gemv_rope",
                                                        lambda *a, **k: [("alloc", "empty")]))
    for fn, kname in (("launch_act_quant_bcast", "t9b2_act_quant_bcast"), ("launch_swiglu_quant", "t9b2_swiglu_quant"),
                      ("launch_moe_combine", "t9b2_moe_combine"), ("launch_hc_post", "t9b2_hc_post"),
                      ("launch_o_rope_qdq", "t9b2_o_rope_qdq")):
        setattr_saved(fused_glue, fn, kernel(getattr(fused_glue, fn), kname, lambda *a, **k: []))
    # regions (innermost wins in the report)
    for cls, attr, name in [
        (R.Block, "hc_mixes", "mhc.mixes"), (R.Block, "hc_pre", "mhc.pre"), (R.Block, "hc_post", "mhc.post"),
        (R.RMSNorm, "forward", "rmsnorm"), (R.Gate, "forward", "moe.gate"), (R.Expert, "forward", "expert"),
        (R.Engram, "forward", "engram"), (R.Compressor, "forward", "compressor"), (R.Indexer, "forward", "indexer"),
        (R.Attention, "forward", "attn"), (R.Attention, "_window_kv", "attn.window_kv"),
        (R.Attention, "_compress_kv", "attn.compress_kv"), (R.ParallelHead, "forward", "head"),
        (R.ParallelEmbedding, "forward", "embed"), (R.Linear, "forward", "linear"),
        (R.ColumnParallelLinear, "forward", "linear"), (R.RowParallelLinear, "forward", "linear"),
        (R.MoE, "forward", "moe"),
        (static_step.StaticStep, "_attn", "attn"), (static_step.StaticStep, "_compress_kv", "attn.compress_kv"),
        (static_step.StaticStep, "_indexer", "indexer"), (static_step.StaticMoE, "__call__", "moe"),
    ]:
        setattr_saved(cls, attr, tagged(getattr(cls, attr), name))
    import runtime.dg_moe as dgm
    setattr_saved(dgm.DGMaskedMoE, "__call__", tagged(dgm.DGMaskedMoE.__call__, "moe"))
    # T9b-2 attribution regions (both glue impls run through these methods)
    for cls, attr, name in [
        (dgm.DGMaskedMoE, "_gather", "moe.gather"), (dgm.DGMaskedMoE, "_gather_fused", "moe.gather"),
        (dgm.DGMaskedMoE, "_record_ids", "moe.ids"), (dgm.DGMaskedMoE, "_act", "moe.act"),
        (dgm.DGMaskedMoE, "_swiglu", "moe.swiglu"), (dgm.DGMaskedMoE, "_swiglu_fused", "moe.swiglu"),
        (dgm.DGMaskedMoE, "_combine", "moe.combine"), (dgm.DGBuffers, "convert_scales", "dg.convert"),
        (dgm.DGBuffers, "load_act", "dg.load_act"),
        (static_step.StaticMoE, "_gather", "moe.gather"), (static_step.StaticMoE, "_gather_fused", "moe.gather"),
        (static_step.StaticMoE, "_record_ids", "moe.ids"),
        (static_step.StaticStep, "_q_proj", "attn.q_proj"), (static_step.StaticStep, "_o_path", "attn.o_path"),
        (fused_glue.FusedGlue, "shared_expert", "expert"), (fused_glue.FusedGlue, "hc_post", "mhc.post"),
    ]:
        setattr_saved(cls, attr, tagged(getattr(cls, attr), name))
    setattr_saved(R, "apply_rotary_emb", tagged(R.apply_rotary_emb, "rope"))
    setattr_saved(static_step, "engram_hash_static", tagged(static_step.engram_hash_static, "engram.hash"))
    setattr_saved(static_step, "select_candidate_blocks_static",
                  tagged(static_step.select_candidate_blocks_static, "indexer.candidates"))
    setattr_saved(static_step, "indexer_topk_static", tagged(static_step.indexer_topk_static, "indexer.topk"))
    setattr_saved(R, "select_candidate_blocks", tagged(R.select_candidate_blocks, "indexer.candidates"))
    # layer-type tags: eager Block.forward(self=layer), static StaticStep._block(self, layer, ...)
    setattr_saved(R.Block, "forward", tagged(R.Block.forward, lambda layer, *a, **k: "L:" + layer_type(layer)))
    setattr_saved(static_step.StaticStep, "_block",
                  tagged(static_step.StaticStep._block, lambda st, layer, *a, **k: "L:" + layer_type(layer)))

    def restore():
        for obj, attr, val in reversed(saved):
            setattr(obj, attr, val)
    return restore


def layer_type(layer) -> str:
    a = layer.attn
    t = [f"r{a.compress_ratio}"]
    if a.is_kv_source:
        t.append("kvsrc")
    if a.is_index_source:
        t.append("idxsrc")
        if a.indexer is not None and a.indexer.is_candidate_source:
            t.append("candsrc")
        elif a.indexer is not None and a.indexer.uses_candidates:
            t.append("candcons")
    return "-".join(t)


# ------------------------------------------------------------------------------------------------ runs
def build(tmp):
    from mini_ckpt import make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference
    from test_static_emulated_cpu import _fake_static_kernels

    R = import_reference(cpu_stubs=True)
    import engram

    cfg, hot = real_structure_config()
    engram.build_compressed_token_map = lambda tok: ([i % 97 for i in range(cfg["vocab_size"])],
                                                     cfg["engram_compressed_vocab_size"])
    ck = make_mini_checkpoint(os.path.join(tmp, "ckpt"), cfg=cfg)
    model = build_and_load(ck["model_dir"], hot, "cpu", tokenizer=object(), cpu_stubs=True, io_threads=4,
                           log=lambda *a: None)
    pool = model.runtime_load.pool
    reg = {}
    for l in range(pool.n_layers):
        for p in ("w1", "w2", "w3"):
            w, s = pool.hbm[l][p]
            for t in list(w) + list(s):
                reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
            for e in pool.cold[l]:
                for kind_ in ("weight", "scale"):
                    t = pool.cold_view(l, e, p, kind_, cuda=False)
                    reg[t.data_ptr()] = t.reshape(-1).view(torch.uint8)
    fk = _fake_static_kernels(reg)
    gather, marker, tables = fk.gather_rows, fk.step_marker, fk.gather_tables
    fk.gather_rows = kernel(gather, "t8_gather_rows", lambda *a, **k: [])
    fk.step_marker = kernel(marker, "t8_step_marker", lambda *a, **k: [])
    fk.gather_tables = kernel(tables, "t9b2_gather_tables", lambda *a, **k: [])
    sys.modules["runtime.static_kernels"] = fk
    model.temperature = 0.0
    return R, model, pool, cfg


def summarize(events, n_steps: int, model) -> dict:
    per_class = collections.Counter(cls for _, cls, _ in events)
    by_type_layers = collections.Counter(layer_type(l) for l in model.layers)
    lt = collections.defaultdict(collections.Counter)      # layer type -> class counts (summed over its layers)
    region_ops = collections.defaultdict(collections.Counter)  # innermost region -> op name (launch/kernel) counts
    region_cls = collections.defaultdict(collections.Counter)
    multi = collections.Counter()
    lt_reg = collections.defaultdict(collections.Counter)    # (layer type, region) -> launch/kernel op counts
    for stack, cls, name in events:
        ltag = next((s[2:] for s in stack if s.startswith("L:")), "step")
        lt[ltag][cls] += 1
        inner = [s for s in stack if not s.startswith("L:")]
        reg = inner[-1] if inner else ("block" if ltag != "step" else "step")
        if inner and reg in ("rmsnorm", "linear", "rope") and len(inner) > 1:
            reg = f"{inner[-2]}/{reg}"
        region_cls[reg][cls] += 1
        if cls in ("launch", "kernel"):
            region_ops[reg][name] += 1
            lt_reg[(ltag, reg)][name] += 1
        if name in MULTI:
            multi[name] += 1
    f = lambda c: {k: round(v / n_steps, 2) for k, v in sorted(c.items())}  # noqa: E731
    out = {"per_step": f(per_class),
           "launches_per_step": round((per_class["launch"] + per_class["kernel"]) / n_steps, 1),
           "launches_per_step_fills_on": round((per_class["launch"] + per_class["kernel"] + per_class["alloc"])
                                               / n_steps, 1),
           "multi_kernel_ops_per_step": f(multi),
           "layer_types": {t: {"layers": by_type_layers.get(t, 0),
                               "per_layer": {k: round(v / n_steps / max(1, by_type_layers.get(t, 1)), 2)
                                             for k, v in sorted(c.items())} if t != "step" else f(c)}
                           for t, c in sorted(lt.items())},
           "layer_type_regions": {t: {r: {k: round(v / n_steps / max(1, by_type_layers.get(t, 1)), 2)
                                          for k, v in sorted(c.items())}
                                      for (t2, r), c in sorted(lt_reg.items()) if t2 == t}
                                  for t in sorted({t for t, _ in lt_reg})},
           "regions": {r: {"classes": f(region_cls[r]), "ops": f(region_ops[r])}
                       for r in sorted(region_cls, key=lambda r: -(region_cls[r]["launch"] + region_cls[r]["kernel"]))}}
    return out


def run_path(path: str, R, model, pool, ids: List[int], steps: int) -> dict:
    from runtime.static_step import StaticStep
    from runtime import patches

    st = None
    if path != "eager-ref":
        kw = {"static-ref": {}, "static-dg-gemv": {"moe_impl": "dg-masked", "fp8_impl": "gemv"},
              "static-dg-gemv-fused": {"moe_impl": "dg-masked", "fp8_impl": "gemv", "glue_impl": "fused"}}[path]
        st = StaticStep(model, pool, max_pos=127, _allow_cpu=True, **kw)  # (fused: self-check runs here, uncounted)
    import engram

    with torch.inference_mode():
        out_ids, logits, _ = model(torch.tensor([ids]), 0)
    tok = int(out_ids[0])
    C.events = []
    # Decode under no_grad, NOT inference_mode: under inference_mode composite ops (F.linear, einsum, .float(), type_as,
    # F.pad, ...) reach the dispatch mode undecomposed and same-dtype casts look like copies; under no_grad the mode
    # sees the primitive aten ops each composite really launches. The reference's @inference_mode decorators are
    # bypassed via __wrapped__ for the counted steps only.
    fwd = getattr(R.Transformer.forward, "__wrapped__", R.Transformer.forward)
    hash_fwd = engram.NgramHashState.forward
    engram.NgramHashState.forward = getattr(hash_fwd, "__wrapped__", hash_fwd)
    try:
        with torch.no_grad(), C:
            for k in range(steps):
                pos = len(ids) + k
                if st is None:
                    x = torch.tensor([[tok]])
                    C.on = True
                    try:
                        out_ids, logits, _ = fwd(model, x, pos)  # the reference decode call (generate.py:69-78)
                    finally:
                        C.on = False
                else:
                    st.set_inputs(tok, pos)                    # outside the graph in graph mode too
                    C.on = True
                    try:
                        with st._active():
                            st._body(pos % 2)                  # exactly what g_even / g_odd record
                    finally:
                        C.on = False
                    st._dev_pos = pos + 1
                    out_ids = st.tok.view(1)
                tok = int(out_ids.reshape(-1)[0])
    finally:
        engram.NgramHashState.forward = hash_fwd
    ev = list(C.events)
    glue = st.glue.describe() if st is not None and st.glue is not None else None
    if st is not None:
        st.detach()
    if patches.fp8_linear_gemv_enabled():
        patches.fp8_linear_gemv(False)
    out = summarize(ev, steps, model)
    if glue is not None:
        out["fused_glue"] = {k: glue[k] for k in ("clusters", "modes")}
    return out


def region_launches(path_report: dict, region: str) -> float:
    c = path_report["regions"].get(region, {}).get("classes", {})
    return c.get("launch", 0) + c.get("kernel", 0)


def t9b2_table(before: dict, after: dict) -> dict:
    """Launches removed per cluster (sum over the cluster's regions of before - after), vs the audit's prediction."""
    rows = {}
    for cl, regs in CLUSTER_REGIONS.items():
        b = sum(region_launches(before, r) for r in regs)
        a = sum(region_launches(after, r) for r in regs)
        rows[cl] = {"before": round(b, 1), "after": round(a, 1), "removed": round(b - a, 1),
                    "audit_predicted": AUDIT_PREDICTED[cl]}
    total = before["launches_per_step"] - after["launches_per_step"]
    return {"per_cluster": rows, "removed_total": round(total, 1),
            "removed_by_clusters": round(sum(r["removed"] for r in rows.values()), 1),
            "audit_predicted_total": sum(AUDIT_PREDICTED.values()),
            "launches_per_step": {"static-dg-gemv": before["launches_per_step"],
                                  "static-dg-gemv-fused": after["launches_per_step"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="per-step torch-op census of the decode paths (CPU)")
    ap.add_argument("--json", default=os.path.join(REPO, "scratch", "glue-census.json"))
    ap.add_argument("--steps", type=int, default=2, help="decode steps counted (2 = one per pos parity)")
    ap.add_argument("--paths", default="eager-ref,static-ref,static-dg-gemv,static-dg-gemv-fused")
    a = ap.parse_args(argv)
    import types

    from test_dg_moe_cpu import _fake_deep_gemm

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        R, model, pool, cfg = build(tmp)
        fake = _fake_deep_gemm()
        fn = fake.m_grouped_fp8_fp4_gemm_nt_masked
        fake.m_grouped_fp8_fp4_gemm_nt_masked = kernel(fn, "dg_masked_gemm", lambda *x, **y: [])
        sys.modules["deep_gemm"] = fake
        restore = install(R)
        prev = torch.get_default_dtype()
        try:
            ids = list(range(40, 56))
            res = {}
            for path in a.paths.split(","):
                res[path] = run_path(path, R, model, pool, ids, a.steps)
                print(f"{path:16s} launches/step {res[path]['launches_per_step']:8.1f} "
                      f"(+allocs with fills on: {res[path]['launches_per_step_fills_on']:8.1f}); "
                      f"classes {res[path]['per_step']}", flush=True)
        finally:
            restore()
            torch.set_default_dtype(prev)
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "config": {k: cfg[k] for k in (
        "n_layers", "kv_source_layers", "index_source_layers", "candidate_source_layer", "engram_layer_ids",
        "n_activated_experts", "hc_mult", "hc_sinkhorn_iters")}, "steps": a.steps, "paths": res,
        "wall_s": round(time.time() - t0, 1)}
    if "static-dg-gemv" in res and "static-dg-gemv-fused" in res:
        t = report["t9b2"] = t9b2_table(res["static-dg-gemv"], res["static-dg-gemv-fused"])
        print(f"T9b-2 fused glue: launches/step {t['launches_per_step']['static-dg-gemv']:.1f} -> "
              f"{t['launches_per_step']['static-dg-gemv-fused']:.1f}, removed {t['removed_total']:.1f} "
              f"(audit predicted {t['audit_predicted_total']})")
        print(f"  {'cluster':38s}{'before':>9s}{'after':>9s}{'removed':>9s}{'audit':>8s}")
        for cl, r in t["per_cluster"].items():
            print(f"  {cl:38s}{r['before']:9.1f}{r['after']:9.1f}{r['removed']:9.1f}{r['audit_predicted']:8d}")
        print(f"  {'sum over clusters':38s}{'':18s}{t['removed_by_clusters']:9.1f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1)
    print(f"report {a.json} ({report['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
