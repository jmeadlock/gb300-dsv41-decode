#!/usr/bin/env python3
"""step_anatomy.py <profile.sqlite> [--out anatomy.json]
Per-step, per-role anatomy of OUR runtime's decode step from an nsys sqlite export:
  NVTX ranges: decode_step (one per step), L{i} (per layer), engram/attn/moe/head (per module, nested in L{i}),
  and (GAP-LEDGER) verify/draft (the DSpark loop's two phases, bench/gap_ledger.py ours-loop-profile).
  Kernels: CUPTI_ACTIVITY_KIND_KERNEL; attributed to the innermost NVTX range that contains the kernel's *launch*
  (via the correlation id -> runtime API call start time), which is how nsys itself attributes.
Reports per step: wall, GPU-busy (union of kernel intervals), idle, kernel count; per role (sum over layers): GPU ms,
launches, mean kernel µs; the top kernels by total time; and the largest inter-kernel gaps with the NVTX context.
GAP-LEDGER additions (report keys, all [static, diagnostic] -- eager static-path durations, not an additive accounting
of the captured graph): `families` (EVERY kernel, not the top 40, mapped by `family_of` below: name rules + the NVTX
role for the names that serve several call sites), `families_by_phase` (verify / draft / other), `kernels_all` (every
kernel name: ms/step, launches/step, family, roles, grids), `sum_kernel_ms_per_step` (sum of durations),
`busy_ms_per_step` (union), `wall_ms_per_step`, `concurrency` (sum / busy: > 1 = kernels overlapped), `overlap`
(busy / wall), `memcpy` (CUPTI_ACTIVITY_KIND_MEMCPY per step by copy kind: bytes, ms -- HtoD from registered Grace is
the C2C copy-engine traffic; kernel-side UVA reads of Grace, e.g. t9a2_stage_inplace's cold rows, are NOT memcpys and
show up only as that kernel's duration), `unmapped` (kernel names no rule matched: extend FAMILY_RULES).
Stdlib only (sqlite3)."""
import bisect
import collections
import json
import re
import sqlite3
import statistics
import sys

# ------------------------------------------------------------------------------------------------ kernel -> family
# Ledger families (docs/GAP-LEDGER-PLAN.md section 4). Order matters: first match wins. Each rule is
# (family, regex over "short name | demangled name", role condition or None). role: the innermost NVTX module range
# ("moe" = layer.ffn's forward, generate.install_layer_nvtx); "!moe" = any kernel launched outside it. Kernels whose
# name serves several call sites (the fp8 GEMV, the torch top-k/sort, aten mm) are split by role; with no NVTX roles
# in the profile (or in a graph census, which has none) they land in the "*.unsplit" family instead of a guess.
FAMILIES = {
    "moe.routed.gemm": "routed MoE: expert GEMM (DeepGEMM fp8xfp4 / MegaMoE / ref fp4_gemm)",
    "moe.swiglu": "MoE SwiGLU+quant (routed and shared share one kernel name)",
    "moe.routed.combine": "routed MoE: combine",
    "moe.routed.staging": "routed MoE: staging/prep (dedup, cold stage over C2C, SF pack, act quant bcast, gathers)",
    "moe.shared": "shared expert (fp8 GEMV inside the moe range)",
    "moe.router": "router (gate fp32, plan, pre/post, top-k inside the moe range)",
    "dense.proj": "dense projections (fp8 GEMV outside moe: q/kv/o + engram/compressor/indexer linears)",
    "dense.head": "head (fixed-order fp32 / bf16-stored head, cuBLAS head)",
    "fp8_gemv.unsplit": "fp8 GEMV, call site not resolvable (no NVTX role)",
    "attn": "attention (FlashMLA / sparse_attn)",
    "glue.norm_mhc": "glue: norms + mHC (hc_norm, norm_row, sinkhorn, hc_post, mixes)",
    "glue.engram": "glue: engram gate/out",
    "glue.indexer": "glue: indexer top-k/sort (outside the moe range)",
    "topk.unsplit": "torch top-k/sort, router vs indexer not resolvable (no NVTX role)",
    "glue.compressor": "glue: ratio-2 compressor (T9b-8 fixed fp32)",
    "glue.quant_rope": "glue: act quant / rope / QDQ",
    "glue.torch": "glue: torch residual (elementwise, reduce, cat, index, cub, asserts)",
    "residency": "residency manager kernels",
    "instrument": "instrument (step marker)",
    "unmapped": "kernel names no rule matched",
}
FAMILY_RULES = [
    ("instrument", r"t8_step_marker", None),
    ("residency", r"residency", None),
    ("moe.routed.gemm", r"sm100_fp8_fp4_gemm|fp8_fp4_mega_moe|mega_moe|m_grouped_fp8_fp4|fp4_gemm_kernel", None),
    ("moe.swiglu", r"t9b2_swiglu_quant|swiglu", None),
    ("moe.routed.combine", r"t9b2_moe_combine|moe_combine", None),
    ("moe.routed.staging", r"t9a2_stage_inplace|m1_stage_rows|m1_dedup|t8_gather_rows|t9b2_gather_tables|"
                           r"transpose_and_pack_fp32_into_ue8m0|t9b2_act_quant_bcast|_t9a2_pack_scales", None),
    ("moe.router", r"m2_router|m2_gate_fp32|m2_softplus|t9b2r2_router", None),
    ("dense.head", r"t9b5_head_fp32|head_bf16|sgemm_largek|gemv2T_kernel", None),
    ("moe.shared", r"t9b_mxfp8_gemv|t9b4_mxfp8_gemv2|t9b_splitk_reduce|fp8_gemm_kernel", "moe"),
    ("dense.proj", r"t9b_mxfp8_gemv|t9b4_mxfp8_gemv2|t9b_splitk_reduce|fp8_gemm_kernel", "!moe"),
    ("fp8_gemv.unsplit", r"t9b_mxfp8_gemv|t9b4_mxfp8_gemv2|t9b_splitk_reduce|fp8_gemm_kernel", None),
    ("attn", r"sparse_attn|flash_fwd|flashmla|flash_mla|sparse_prefill|mla_|fused_norm_rope_attn|splitkv_mla", None),
    ("glue.compressor", r"t9b8_comp_fp32", None),
    ("glue.engram", r"t9b2r2_engram", None),
    ("glue.norm_mhc", r"t9b2r2_hc_norm|t9b4_hc_norm|t9b2r2_norm_row|hc_split_sinkhorn|t9b2_hc_post|t9b5_mix_fp32|"
                      r"hc_prenorm|mhc", None),
    # cuBLAS fp32 paths: Kernel2 = CUTLASS split-K fp32 (the mHC mixes at M=6, LEDGER 2026-11-07 06:39), its
    # splitKreduce, and aten mm (nvjet): inside moe = the router gate's cuBLAS matmul, outside = the mHC mixes
    ("moe.router", r"^Kernel2\||splitKreduce_kernel|nvjet", "moe"),
    ("glue.norm_mhc", r"^Kernel2\||splitKreduce_kernel|nvjet", "!moe"),
    ("glue.norm_mhc", r"^Kernel2\||splitKreduce_kernel|nvjet", None),
    ("glue.indexer", r"t9b2r2_topk_post", None),
    ("moe.router", r"gatherTopK|bitonicSort|radixSort|warpMergeSort|DeviceRadixSort|DeviceSelect|DeviceCompact", "moe"),
    ("glue.indexer", r"gatherTopK|bitonicSort|radixSort|warpMergeSort|DeviceRadixSort|DeviceSelect|DeviceCompact",
     "!moe"),
    ("topk.unsplit", r"gatherTopK|bitonicSort|radixSort|warpMergeSort|DeviceRadixSort|DeviceSelect|DeviceCompact", None),
    ("glue.quant_rope", r"t9b2_o_rope_qdq|t9b2_rope|act_quant_kernel|fp4_quant_kernel|q_rope", None),
    ("glue.torch", r"elementwise|reduce_kernel|reduce_1Block|CatArray|index|_scatter_gather|_assert_async|cunn_|"
                   r"DeviceReduce|write_indices|kernelHistogram|dot_kernel|gather_kernel|fill|copy", None),
]   # the bare short name "kernel" (46/step in every anatomy so far) is deliberately unmapped until identified
_COMPILED = [(f, re.compile(p), r) for f, p, r in FAMILY_RULES]

# ------------------------------------------------------------------------------------------------ M4-P1: prefill map
# The reference prefill (`model(tokens, 0)`, model.py:1245-1275) runs the reference's TileLang kernels + torch glue,
# not our decode kernels, so it gets its own family map. Roles are bench/prefill_anatomy.py's NVTX module ranges
# (innermost wins): "expert.hot" / "expert.cold" (one routed `Expert` call, model.py:902), "gate" (model.py:894),
# "shared" (model.py:905), "moe" (the rest of MoE.forward: zeros, bincount().tolist(), where(), y[idx] +=,
# type_as), "compressor" (model.py:748), "indexer" (model.py:736), "attn" (the rest of Attention.forward), "engram"
# (model.py:1266), "head" (model.py:1272). A rule's role condition is a set of roles the kernel's launch must sit in
# ("in") or must not ("not"); with no roles in the profile, role rules are skipped (the name rules still apply).
PREFILL_EXPERT_ROLES = frozenset({"expert.hot", "expert.cold"})
PREFILL_ROLES = ("expert.hot", "expert.cold", "gate", "shared", "moe", "compressor", "indexer", "attn", "engram",
                 "head")
PREFILL_FAMILIES = {
    "moe.routed.gemm": "routed experts: reference fp4_gemm, one launch per expert x projection (or a grouped GEMM)",
    "moe.routed.quant": "routed experts: act_quant of the expert's rows (3 per expert call)",
    "moe.routed.index": "routed MoE: where()/nonzero compaction, bincount, x[idx] / weights[idx, top] gathers",
    "moe.routed.combine": "routed MoE: y[idx] += (deterministic index_put: radix sort + indexing_backward)",
    "moe.routed.glue": "routed experts: float / clamp / silu / mul / weights / to(bf16) elementwise",
    "moe.router": "gate: fp32 F.linear (cuBLAS), softplus/sqrt, bias, topk, gather, normalise",
    "moe.shared": "shared expert (fp8_gemm x3 + glue)",
    "moe.other": "rest of MoE.forward (zeros, y += shared, type_as)",
    "attn.sparse": "attention kernel (reference sparse_attn / FlashMLA)",
    "attn.indexer": "indexer (wq_b, weights_proj, einsum scores, masks, topk + sort)",
    "attn.compressor": "compressor (wkv/wgate F.linear, softmax pooling, norm)",
    "attn.wo_a": "wo_a einsum (bf16 bmm) and other cuBLAS GEMMs inside the attention range",
    "dense.proj": "fp8 Linears outside the MoE (wq_a, wq_b, wkv, wo_b, engram wkv): reference fp8_gemm",
    "dense.head": "head (fp32 F.linear on the last row)",
    "engram": "engram (hash rows gather from Grace, wkv, gate)",
    "glue.mhc": "mHC: hc_mixes fp32 F.linear (cuBLAS) + hc_split_sinkhorn",
    "glue.quant_rope": "act_quant / fp4 QDQ outside the experts (KV, o, latent, index K/Q)",
    "glue.torch": "torch glue elsewhere (norms, RoPE, hc_pre/post, cat, index lists, copies, fills)",
    "instrument": "instrument (step marker)",
    "unmapped": "kernel names no rule matched",
}
_ANY = r"."
PREFILL_FAMILY_RULES = [
    ("instrument", r"t8_step_marker", None),
    ("moe.routed.gemm", r"fp4_gemm_kernel|sm100_fp8_fp4_gemm|m_grouped_fp8_fp4|fp8_fp4_mega_moe|mega_moe", None),
    ("moe.router", _ANY, ("in", frozenset({"gate"}))),
    ("moe.shared", _ANY, ("in", frozenset({"shared"}))),
    ("attn.compressor", _ANY, ("in", frozenset({"compressor"}))),
    ("attn.indexer", _ANY, ("in", frozenset({"indexer"}))),
    ("dense.head", _ANY, ("in", frozenset({"head"}))),
    ("engram", _ANY, ("in", frozenset({"engram"}))),
    ("moe.routed.quant", r"act_quant_kernel", ("in", PREFILL_EXPERT_ROLES)),
    ("moe.routed.index", r"DeviceSelect|DeviceCompact|write_indices|nonzero|kernelHistogram|index_elementwise|"
                         r"indexSelect|vectorized_gather|gather_kernel|_scatter_gather|elementwise_kernel_with_index",
     ("in", PREFILL_EXPERT_ROLES | {"moe"})),
    ("moe.routed.combine", r"indexing_backward|DeviceRadixSort|radixSort|index_put|DeviceScan",
     ("in", PREFILL_EXPERT_ROLES | {"moe"})),
    ("moe.routed.glue", _ANY, ("in", PREFILL_EXPERT_ROLES)),
    ("moe.other", _ANY, ("in", frozenset({"moe"}))),
    ("attn.sparse", r"sparse_attn|flash_fwd|flashmla|flash_mla|sparse_prefill|sparse_fwd|splitkv_mla", None),
    ("dense.proj", r"fp8_gemm_kernel|t9b_mxfp8_gemv|t9b4_mxfp8_gemv2|t9b_splitk_reduce", None),
    ("glue.mhc", r"hc_split_sinkhorn|hc_prenorm|mhc", None),
    ("attn.wo_a", r"nvjet|Kernel2|cutlass|gemm|splitKreduce|gemv", ("in", frozenset({"attn"}))),
    # cuBLAS fp32 outside attn / moe / head: Block.hc_mixes' F.linear (model.py:956) -- the only GEMM left there
    ("glue.mhc", r"nvjet|^Kernel2\||cutlass|sgemm|splitKreduce|gemv2T|gemm", None),
    ("glue.quant_rope", r"act_quant_kernel|fp4_quant_kernel", None),
    ("glue.torch", r"elementwise|reduce_kernel|reduce_1Block|CatArray|index|_scatter_gather|_assert_async|cunn_|"
                   r"DeviceReduce|write_indices|kernelHistogram|dot_kernel|gather|fill|copy|Device|radix|bitonic|"
                   r"warpMerge|gatherTopK|sort|topk", None),
]
_PREFILL_COMPILED = [(f, re.compile(p), c) for f, p, c in PREFILL_FAMILY_RULES]


def prefill_family_of(short: str, demangled: str = "", role=None, roles_known: bool = True) -> str:
    """Prefill kernel -> family (PREFILL_FAMILIES). `role`: the innermost bench/prefill_anatomy.py NVTX module range
    of the launch (PREFILL_ROLES or None). roles_known=False skips every role-conditioned rule."""
    text = f"{short or ''}|{demangled or ''}"
    for fam, rx, cond in _PREFILL_COMPILED:
        if cond is not None:
            if not roles_known:
                continue
            kind, roles = cond
            if (kind == "in") != (role in roles):
                continue
        if rx.search(text):
            return fam
    return "unmapped"


def family_of(short: str, demangled: str = "", role=None, roles_known: bool = True) -> str:
    """Kernel -> ledger family. `role`: innermost NVTX module range of the launch ("moe", "attn", ... or None).
    roles_known=False (no module ranges anywhere in the profile, or a graph census): role-conditioned rules are
    skipped, so multi-site names fall to their *.unsplit family rather than a wrong split."""
    text = f"{short or ''}|{demangled or ''}"
    for fam, rx, cond in _COMPILED:
        if cond is not None:
            if not roles_known:
                continue
            if cond == "moe" and role != "moe":
                continue
            if cond == "!moe" and role == "moe":
                continue
        if rx.search(text):
            return fam
    return "unmapped"


# CUPTI copyKind enum (CUpti_ActivityMemcpyKind)
COPY_KIND = {0: "unknown", 1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD",
             9: "HtoH", 10: "PtoP"}
PHASES = ("verify", "draft")
MODULE_ROLES = ("engram", "attn", "moe", "head")


def finder(ranges):
    starts = [r[0] for r in ranges]

    def f(t):
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            if ranges[i][0] <= t <= ranges[i][1]:
                return ranges[i]
            i -= 1
        return None
    return f


def analyze(db_path: str, log=print) -> dict:
    db = sqlite3.connect(db_path)
    q = lambda s, *a: db.execute(s, a).fetchall()
    tables = {r[0] for r in q("select name from sqlite_master where type='table'")}
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in tables, tables
    kcols = {r[1] for r in q("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    grid = ", k.gridX, k.gridY, k.gridZ" if {"gridX", "gridY", "gridZ"} <= kcols else ", null, null, null"

    # kernels with names and launch (runtime API) start time via correlationId
    # M1-T10-MOE: + the demangled name (a template wrapper's short name, e.g. CUTLASS's `Kernel2`, says nothing)
    kern = q(f"""select k.start, k.end, k.correlationId, coalesce(s.value, k.demangledName), k.streamId,
                       coalesce(d.value, ''){grid}
                from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id = k.shortName
                left join StringIds d on d.id = k.demangledName order by k.start""")
    rt = {}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        for s, e, cid in q("select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
            rt[cid] = s
    mcpy = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mcols = {r[1] for r in q("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        if {"start", "end", "bytes", "copyKind"} <= mcols:
            mcpy = q("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start")
    # nvtx ranges
    nvtx = []
    if "NVTX_EVENTS" in tables:
        for s, e, txt, tid in q("select start, end, coalesce(text, (select value from StringIds where id=textId)), "
                                "globalTid from NVTX_EVENTS where end is not null"):
            nvtx.append((s, e, txt or "", tid))
    steps = sorted([(s, e) for s, e, t, _ in nvtx if t == "decode_step"])
    if not steps:
        # graph replay: NVTX ranges do not survive; T8's per-step marker kernel (static_kernels.t8_step_marker) delimits steps
        marks = [k[0] for k in kern if "t8_step_marker" in (k[3] or "")]
        steps = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
        log(f"no decode_step NVTX; using {len(marks)} marker kernels -> {len(steps)} steps")
    layers = sorted([(s, e, t) for s, e, t, _ in nvtx if t.startswith("L") and t[1:].isdigit()])
    roles = sorted([(s, e, t) for s, e, t, _ in nvtx if t in MODULE_ROLES])
    phases = sorted([(s, e, t) for s, e, t, _ in nvtx if t in PHASES])
    roles_known = any(t == "moe" for _, _, t in roles)
    log(f"kernels {len(kern)}  runtime-api {len(rt)}  nvtx {len(nvtx)}  decode_step ranges {len(steps)}  layer ranges "
        f"{len(layers)}  role ranges {len(roles)}  phase ranges {len(phases)}  memcpy {len(mcpy)}")

    find_step, find_layer, find_role, find_phase = finder(steps), finder(layers), finder(roles), finder(phases)
    per_step = []
    role_t = collections.defaultdict(float); role_n = collections.Counter(); layer_t = collections.defaultdict(float)
    kname_t = collections.defaultdict(float); kname_n = collections.Counter()
    kname_full = {}; kname_role = collections.defaultdict(collections.Counter)   # M1-T10-MOE: what/who a name is
    kname_fam = collections.defaultdict(collections.Counter); kname_grid = collections.defaultdict(collections.Counter)
    fam_t = collections.defaultdict(float); fam_n = collections.Counter()
    ph_t = collections.defaultdict(lambda: collections.defaultdict(float))
    ph_n = collections.defaultdict(collections.Counter)
    gaps = []
    for si, (ss, se) in enumerate(steps):
        ks = [k for k in kern if ss <= k[0] < se]
        if not ks:
            continue
        busy = 0; last_end = None; n = 0; ksum = 0
        for s, e, cid, name, _, full, gx, gy, gz in ks:
            if last_end is None or s > last_end:
                busy += e - s
                if last_end is not None and s - last_end > 0:
                    gaps.append((s - last_end, si, name, (find_role(rt.get(cid, s)) or ("", "", "?"))[2],
                                 (find_layer(rt.get(cid, s)) or ("", "", "?"))[2]))
                last_end = e
            else:
                busy += max(0, e - last_end); last_end = max(last_end, e)
            n += 1
            ksum += e - s
            lt = rt.get(cid, s)
            r = find_role(lt); L = find_layer(lt); P = find_phase(lt)
            role = r[2] if r else ("step-other")
            phase = P[2] if P else "other"
            fam = family_of(name, full, r[2] if r else None, roles_known)
            ms = (e - s) / 1e6
            role_t[role] += ms; role_n[role] += 1
            if L:
                layer_t[L[2]] += ms
            kname_t[name] += ms; kname_n[name] += 1
            kname_full.setdefault(name, full); kname_role[name][role] += 1; kname_fam[name][fam] += 1
            if gx is not None:
                kname_grid[name][f"{gx}x{gy}x{gz}"] += 1
            fam_t[fam] += ms; fam_n[fam] += 1
            ph_t[phase][fam] += ms; ph_n[phase][fam] += 1
        mc = collections.defaultdict(lambda: [0, 0.0, 0])
        for s, e, nb, kind in mcpy:
            if ss <= s < se:
                m = mc[COPY_KIND.get(kind, str(kind))]
                m[0] += nb or 0; m[1] += (e - s) / 1e6; m[2] += 1
        per_step.append({"step": si, "wall_ms": (se - ss) / 1e6, "busy_ms": busy / 1e6, "kernels": n,
                         "sum_kernel_ms": ksum / 1e6,
                         "memcpy": {k: {"bytes": v[0], "ms": v[1], "n": v[2]} for k, v in mc.items()}})

    ns = len(per_step)
    rep = {"steps": per_step, "n_steps": ns, "roles_known": roles_known}
    if not ns:
        log("no steps with kernels")
        rep["error"] = "no steps with kernels"
        return rep
    rep["role_ms_per_step"] = {r: t / ns for r, t in role_t.items()}
    rep["role_launches_per_step"] = {r: c / ns for r, c in role_n.items()}
    rep["layer_ms_per_step"] = {L: t / ns for L, t in layer_t.items()}
    top = sorted(kname_t.items(), key=lambda x: -x[1])
    rep["top_kernels"] = [(n, t / ns, kname_n[n] / ns) for n, t in top[:40]]
    rep["kernel_names"] = {n: kname_full.get(n, "")[:400] for n, _ in top[:40]}
    rep["kernel_roles"] = {n: {r: c / ns for r, c in kname_role[n].items() if c} for n, _ in top[:40]}
    # GAP-LEDGER keys
    rep["kernels_all"] = {n: {"ms_per_step": t / ns, "launches_per_step": kname_n[n] / ns,
                              "families": {f: c / ns for f, c in kname_fam[n].items()},
                              "roles": {r: c / ns for r, c in kname_role[n].items()},
                              "grids": dict(kname_grid[n].most_common(6)), "demangled": kname_full.get(n, "")[:400]}
                          for n, t in top}
    rep["families"] = {f: {"ms_per_step": fam_t[f] / ns, "launches_per_step": fam_n[f] / ns} for f in fam_t}
    rep["families_by_phase"] = {p: {f: {"ms_per_step": ph_t[p][f] / ns, "launches_per_step": ph_n[p][f] / ns}
                                    for f in ph_t[p]} for p in ph_t}
    rep["unmapped"] = sorted(n for n in kname_t if "unmapped" in kname_fam[n])
    mean = lambda k: statistics.mean(p[k] for p in per_step)
    rep["wall_ms_per_step"] = mean("wall_ms")
    rep["busy_ms_per_step"] = mean("busy_ms")
    rep["sum_kernel_ms_per_step"] = mean("sum_kernel_ms")
    rep["kernels_per_step"] = mean("kernels")
    rep["concurrency"] = rep["sum_kernel_ms_per_step"] / rep["busy_ms_per_step"] if rep["busy_ms_per_step"] else None
    rep["overlap"] = rep["busy_ms_per_step"] / rep["wall_ms_per_step"] if rep["wall_ms_per_step"] else None
    mtot = collections.defaultdict(lambda: [0, 0.0, 0])
    for p in per_step:
        for k, v in p["memcpy"].items():
            mtot[k][0] += v["bytes"]; mtot[k][1] += v["ms"]; mtot[k][2] += v["n"]
    rep["memcpy"] = {k: {"bytes_per_step": v[0] / ns, "ms_per_step": v[1] / ns, "n_per_step": v[2] / ns}
                     for k, v in mtot.items()}
    gaps.sort(reverse=True)
    rep["idle_ms_per_step"] = sum(g[0] for g in gaps) / 1e6 / ns
    rep["_gaps_top"] = [(g / 1e3, si, name, role, L) for g, si, name, role, L in gaps[:10]]
    rep["family_labels"] = FAMILIES
    return rep


def report_lines(rep: dict) -> list:
    ps = rep.get("steps") or []
    ns = len(ps)
    if not ns:
        return ["no steps"]
    out = [f"\nsteps analysed {ns}: wall mean {statistics.mean(p['wall_ms'] for p in ps):.2f} ms  p50 "
           f"{statistics.median(p['wall_ms'] for p in ps):.2f}  busy mean {statistics.mean(p['busy_ms'] for p in ps):.2f} ms"
           f"  kernels/step {statistics.mean(p['kernels'] for p in ps):.0f}"]
    out.append("\nrole            GPU ms/step  launches/step  mean µs/kernel")
    for r, t in sorted(rep["role_ms_per_step"].items(), key=lambda x: -x[1]):
        n = rep["role_launches_per_step"][r]
        out.append(f"{r:15s} {t:10.2f} {n:14.0f} {1000 * t / n:14.1f}")
    out.append("\nlayer GPU ms/step (first 3, 14 (engram), last):")
    for L in ["L0", "L1", "L2", "L14", "L39"]:
        if L in rep["layer_ms_per_step"]:
            out.append(f"  {L:4s} {rep['layer_ms_per_step'][L]:7.2f}")
    out.append("\ntop kernels by GPU time (ms/step, launches/step, µs each):")
    for name, t, n in rep["top_kernels"][:18]:
        full = rep["kernel_names"].get(name, "")
        out.append(f"  {t:7.2f} {n:7.0f} {1000 * t / n:8.1f}  {name[:90]}"
                   + (f"  [{full[:120]}]" if full and full != name else ""))
    out.append(f"\ninter-kernel idle inside steps: {rep['idle_ms_per_step']:.2f} ms/step total; largest:")
    for g, si, name, role, L in rep["_gaps_top"]:
        out.append(f"  {g:8.1f} µs  step {si} {L}/{role} before {name[:70]}")
    # GAP-LEDGER
    out.append(f"\nfamilies [static, diagnostic] (roles {'known' if rep['roles_known'] else 'UNKNOWN: no moe NVTX'}): "
               f"sum {rep['sum_kernel_ms_per_step']:.3f} ms/step, busy {rep['busy_ms_per_step']:.3f}, wall "
               f"{rep['wall_ms_per_step']:.3f}, concurrency {rep['concurrency']:.3f}, overlap {rep['overlap']:.3f}")
    for f, v in sorted(rep["families"].items(), key=lambda x: -x[1]["ms_per_step"]):
        out.append(f"  {f:20s} {v['ms_per_step']:8.3f} ms {v['launches_per_step']:8.1f} launches")
    if rep["unmapped"]:
        out.append(f"  unmapped names: {', '.join(rep['unmapped'])[:300]}")
    for k, v in rep["memcpy"].items():
        out.append(f"  memcpy {k}: {v['bytes_per_step'] / 1e6:.1f} MB/step, {v['ms_per_step']:.3f} ms, {v['n_per_step']:.1f}/step")
    return out


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    rep = analyze(argv[1])
    if rep.get("error"):
        return 1
    for line in report_lines(rep):
        print(line)
    if out:
        slim = {k: v for k, v in rep.items() if not k.startswith("_")}
        with open(out, "w") as f:
            json.dump(slim, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
