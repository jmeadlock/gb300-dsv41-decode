"""M1-T11: load the checkpoint's DSpark draft stages (`mtp.*`) into HBM next to an already-loaded backbone.

    model = build_and_load(model_dir, map, "cuda", load_mtp=True)      # runtime/loader.py calls attach_mtp
    model.mtp            -> nn.ModuleList of reference `DSparkBlock`s (model.py:1103-1159), loaded, dense, in HBM
    model.runtime_mtp    -> MtpLoad (bytes, timings, the per-stage expert stacks DraftStep gathers from)

The backbone keeps the loader overlay (`dspark_block_size=0`, `dspark_target_layer_ids=()`, loader.py:59-65), so
`Transformer.__init__` built no draft stages (model.py:1212) and `forward` captures no main_hiddens (1268). This module
builds the stages afterwards, exactly as model.py:1210-1216 would have:

  * args = the checkpoint's own config with the overlay's non-dspark keys (vision off, batch 1, max_seq_len), so
    `DSparkBlock(n_layers + i, args)` gets `dspark_block_size`, `dspark_noise_token_id`, `dspark_target_layer_ids`,
    `dspark_markov_rank` and the draft MoE shape (`get_moe_config`, model.py:142-149: 128 routed, top-3) from it;
  * construction under the loader's meta-factory mode (loader.py `_ParamFactoriesToMeta`): parameters on meta, buffers
    (the stage's 128-slot `window_kv_cache`, its `freqs_cis`) real on the device;
  * `stage.embed = model.embed`, `stage.head = model.head` (model.py:1215-1216, shared with the backbone);
  * `stage.temperature = 0` -- DSparkBlock captures `args.temperature` at construction (model.py:1113) and samples the
    drafts with it (1156 -> `sample`, 1288-1295: Gumbel noise unless 0). Our Generator sets only the Transformer's
    (generate.py `model.temperature = 0.0`; model.py:1197 / 1273). Design memo M1-DSPARK-DESIGN.md section 0 item 5.
    With 0 both the reference `forward_spec` and runtime/draft_step.py take `logits.argmax(-1)` (1291-1292).

Placement: every `mtp.*` tensor in HBM, dense (no hot/cold map: 3 x 128 experts are 6.8 GB of the ~7.5 GiB,
M0-PORT-PLAN.md:102). The routed experts of a stage are stacked, one contiguous `[E, N, K/2]` int8 weight block and
`[E, N, K/32]` e8m0 scale block per projection; each reference `Expert`'s Linears are zero-copy views into the stack
(the same binding runtime/expert_pool.py does for the backbone), so the reference `forward_spec` and DraftStep's
static gather (`index_select` of whole expert rows into staging) read the same bytes.

Value paths: the loader's three (README "Value paths"), unchanged -- wo_a fp8 x e8m0 -> bf16 by convert.py's formula
(`loader.dequant_wo_a`), bf16 -> fp32 widening where model.py declares fp32 (here: `markov_head.head.weight`,
ParallelHead model.py:1007-1008; `confidence_head.proj.weight`, model.py:1095-1096), and I8 -> float4_e2m1fn_x2 views
of the expert weights. Anything else refuses the load with the missing / unexpected / mismatched lists.
Skipped: `mtp.*.ffn.gate.bias_vl` (vision off, model.py:809). `mtp.*.embed.weight` / `mtp.*.head.weight` never reach
the index (convert.py:109-110, runtime/ckpt.py `convert_name`).
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

MTP_EXPERT_RE = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
MTP_SKIP_SUFFIX = (".ffn.gate.bias_vl",)  # vision off (model.py:809)
PROJS = ("w1", "w3", "w2")
GiB = 1 << 30


@dataclass
class MtpLoad:
    n_stages: int
    target_layer_ids: Tuple[int, ...]
    block_size: int
    n_tensors: int = 0
    bytes: Dict[str, int] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    dequantized: List[str] = field(default_factory=list)
    widened: List[str] = field(default_factory=list)
    stacks: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    verify: Optional[dict] = None

    def summary(self) -> dict:
        return {"n_stages": self.n_stages, "target_layer_ids": list(self.target_layer_ids),
                "block_size": self.block_size, "n_tensors": self.n_tensors,
                "bytes_GiB": {k: round(v / GiB, 4) for k, v in self.bytes.items()}, "bytes": dict(self.bytes),
                "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
                "dequantized_wo_a": len(self.dequantized), "widened_bf16_to_fp32": list(self.widened),
                "verify": None if self.verify is None else {k: v for k, v in self.verify.items() if k != "per_tensor"}}


def mtp_config(config_path: str, overlay: Optional[dict] = None) -> dict:
    """The checkpoint's config with the backbone overlay's NON-dspark keys applied (vision off, batch 1, the same
    max_seq_len, so the stages' freqs_cis and window caches match the backbone's buffers)."""
    from runtime.loader import OVERLAY

    with open(config_path) as f:
        cfg = json.load(f)
    ov = OVERLAY if overlay is None else overlay
    cfg.update({k: v for k, v in ov.items() if not k.startswith("dspark_")})
    return cfg


def _skip(name: str) -> bool:
    return name.endswith(MTP_SKIP_SUFFIX)


def build_meta_stages(ref, args, device: torch.device) -> nn.ModuleList:
    """`DSparkBlock(n_layers + i, args)` for every draft stage (model.py:1213-1214): parameters meta, buffers real."""
    from runtime.loader import LoadRefused, _ParamFactoriesToMeta

    if not args.dspark_block_size or not args.n_mtp_layers:
        raise LoadRefused(f"config has no DSpark stages (dspark_block_size={args.dspark_block_size}, "
                          f"n_mtp_layers={args.n_mtp_layers})")
    if not args.dspark_target_layer_ids:
        raise LoadRefused("config has no dspark_target_layer_ids (model.py:1115)")
    with torch.device(device), _ParamFactoriesToMeta():
        stages = nn.ModuleList([ref.DSparkBlock(args.n_layers + i, args) for i in range(args.n_mtp_layers)])
    real = [n for n, p in stages.named_parameters() if not p.is_meta]
    if real:
        raise LoadRefused(f"draft parameters allocated during construction (expected meta): {real[:10]}")
    meta_bufs = [n for n, b in stages.named_buffers() if b.is_meta]
    if meta_bufs:
        raise LoadRefused(f"draft buffers on meta ({meta_bufs[:10]}); model.py changed?")
    for i, st in enumerate(stages):
        if st.attn.compress_ratio != 0:
            raise LoadRefused(f"mtp.{i}: compress ratio {st.attn.compress_ratio} (DSparkAttention asserts 0)")
    return stages


def check_mtp_keys(index, expected: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]) -> List[str]:
    """Refuse on missing / unexpected `mtp.*` keys, shape mismatch, or an undocumented dtype change (the loader's
    check_keys rules). Returns the names to place (wo_a.scale excluded: the dequant consumes it)."""
    from runtime.loader import LoadRefused

    provided = []
    for n in sorted(index.names()):
        if not n.startswith("mtp.") or _skip(n):
            continue
        if n.endswith(".attn.wo_a.scale") and n[: -len("scale")] + "weight" in index:
            continue
        provided.append(n)
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    for n in provided:
        if n.endswith(".attn.wo_a.weight") and n[: -len("weight")] + "scale" not in index:
            missing.append(n[: -len("weight")] + "scale")
    bad = []
    for n in sorted(set(provided) & set(expected)):
        e, (shape, dt) = index[n], expected[n]
        if tuple(e.shape) != tuple(shape):
            bad.append(f"{n}: shape {tuple(e.shape)} in checkpoint vs {tuple(shape)} in model")
            continue
        if e.dtype != dt:
            ok = ((n.endswith(".attn.wo_a.weight") and e.dtype == torch.float8_e4m3fn and dt == torch.bfloat16)
                  or (e.dtype == torch.bfloat16 and dt == torch.float32)
                  or (MTP_EXPERT_RE.match(n) and e.dtype == torch.int8 and dt == torch.float4_e2m1fn_x2))
            if not ok:
                bad.append(f"{n}: undocumented dtype change {e.dtype} -> {dt}")
    if missing or unexpected or bad:
        msg = [f"refusing mtp load: {len(missing)} missing, {len(unexpected)} unexpected, {len(bad)} mismatched",
               "MISSING (in the draft stages, not in checkpoint):"]
        msg += [f"  {n}" for n in missing] or ["  (none)"]
        msg.append("UNEXPECTED (mtp.* in checkpoint, not in the draft stages, not skipped):")
        msg += [f"  {n}" for n in unexpected] or ["  (none)"]
        msg.append("MISMATCHED:")
        msg += [f"  {b}" for b in bad] or ["  (none)"]
        text = "\n".join(msg)
        print(text, file=sys.stderr)
        raise LoadRefused(text)
    return provided


def attach_mtp(model: nn.Module, index, config_path: str, device, *, overlay: Optional[dict] = None,
               verify: bool = False, io_threads: int = 8, stage_bytes: int = 256 << 20, n_stage: int = 4,
               cpu_stubs: bool = False, log: Callable = print) -> MtpLoad:
    """Build the reference draft stages, load every `mtp.*` tensor into HBM (dense; experts stacked), bind them to
    `model.mtp` (module docstring). Returns the MtpLoad, also kept as `model.runtime_mtp`."""
    from runtime.loader import (LoadRefused, StagedUploader, _parallel, _set_param, dequant_wo_a)
    from runtime.refmodel import import_reference

    t_all = time.time()
    device = torch.device(device)
    ref = import_reference(cpu_stubs=cpu_stubs)
    if len(getattr(model, "mtp", ())) or getattr(model, "target_layer_ids", ()):
        raise LoadRefused("attach_mtp: the backbone already has draft stages / target layers (expected the loader "
                          "overlay dspark_block_size=0, dspark_target_layer_ids=())")
    cfg = mtp_config(config_path, overlay)
    args = ref.ModelArgs(**cfg)
    if args.n_layers != len(model.layers) or args.dim != model.embed.weight.size(1):
        raise LoadRefused(f"mtp config ({args.n_layers} layers, dim {args.dim}) does not match the backbone")
    if int(args.max_seq_len) != int(model.max_seq_len):
        raise LoadRefused(f"mtp max_seq_len {args.max_seq_len} != backbone {model.max_seq_len}")

    t0 = time.time()
    stages = build_meta_stages(ref, args, device)
    expected = {f"mtp.{n}": (tuple(p.shape), p.dtype) for n, p in stages.named_parameters()}
    names = check_mtp_keys(index, expected)
    timings = {"construct": time.time() - t0}

    # ---- allocate: expert stacks per (stage, proj), every other parameter on its own
    t0 = time.time()
    stacks: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for i, st in enumerate(stages):
        ex0 = st.ffn.experts[0]
        E = len(st.ffn.experts)
        stacks[i] = {}
        for p in PROJS:
            lin = getattr(ex0, p)
            w = torch.empty((E, *lin.weight.shape), dtype=torch.int8, device=device)
            s = torch.empty((E, *lin.scale.shape), dtype=torch.uint8, device=device)
            stacks[i][p] = (w, s)
    up = StagedUploader(index, device, stage_bytes, n_stage)
    params: Dict[str, torch.Tensor] = {}
    special: List[str] = []
    expert_views: Dict[str, Tuple[int, int, str, str]] = {}
    for n in names:
        m = MTP_EXPERT_RE.match(n)
        if m:
            i, e, p, kind = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            dst = stacks[i][p][0 if kind == "weight" else 1][e]
            up.add(n, dst)
            expert_views[n] = (i, e, p, kind)
            continue
        shape, dt = expected[n]
        t = torch.empty(shape, dtype=dt, device=device)
        params[n] = t
        if index[n].dtype == dt:
            up.add(n, t)
        else:
            special.append(n)
    timings["alloc"] = time.time() - t0
    t0 = time.time()
    up.run()
    dequantized, widened = [], []

    def do_special(n):
        t = params[n]
        src = index.read_tensor(n)
        if n.endswith(".attn.wo_a.weight"):
            val = dequant_wo_a(src, index.read_tensor(n[: -len("weight")] + "scale"))
            dequantized.append(n)
        else:
            val = src.to(t.dtype)  # bf16 -> fp32, exact
            widened.append(n)
        t.copy_(val)
        return src.numel() * src.element_size()

    special_bytes = sum(_parallel(do_special, special, min(io_threads, 4)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wo_a_scale_bytes = sum(index[n[: -len("weight")] + "scale"].nbytes for n in dequantized)
    timings["hbm_load"] = time.time() - t0

    # ---- bind
    for n, t in params.items():
        _set_param(stages, n[len("mtp."):], t)
    for n, (i, e, p, kind) in expert_views.items():
        if kind == "scale":
            continue
        lin = getattr(stages[i].ffn.experts[e], p)
        w = stacks[i][p][0][e].view(torch.float4_e2m1fn_x2)  # convert.py:181, zero-copy
        s = stacks[i][p][1][e].view(torch.float8_e8m0fnu)
        if tuple(w.shape) != tuple(lin.weight.shape) or tuple(s.shape) != tuple(lin.scale.shape):
            raise LoadRefused(f"{n}: stack view {tuple(w.shape)}/{tuple(s.shape)} vs model "
                              f"{tuple(lin.weight.shape)}/{tuple(lin.scale.shape)}")
        lin.weight = nn.Parameter(w, requires_grad=False)
        lin.scale = nn.Parameter(s, requires_grad=False)
    n_scale = 0
    for mod in stages.modules():
        if isinstance(mod, ref.Linear) and mod.scale is not None:
            mod.weight.scale = mod.scale  # model.py:222/227: linear() reads weight.scale
            n_scale += 1
    still_meta = [n for n, p in stages.named_parameters() if p.is_meta]
    if still_meta:
        raise LoadRefused(f"draft parameters left on meta after load: {still_meta[:20]}")
    for st in stages:
        st.temperature = 0.0  # model.py:1113 captured args.temperature; 1156 samples with it (design memo 0.5)
    placed = sum(t.numel() * t.element_size() for t in params.values())
    stack_bytes = sum(w.numel() * w.element_size() + s.numel() * s.element_size()
                      for d in stacks.values() for (w, s) in d.values())
    # shared with the backbone (model.py:1215-1216): assigned last, so the counts above are the draft's own
    for st in stages:
        st.embed = model.embed
        st.head = model.head
    model.mtp = stages

    res = MtpLoad(n_stages=len(stages), target_layer_ids=tuple(int(x) for x in args.dspark_target_layer_ids),
                  block_size=int(args.dspark_block_size), n_tensors=len(names), timings=timings,
                  dequantized=sorted(dequantized), widened=sorted(widened), stacks=stacks)
    res.bytes = {"read": up.bytes + special_bytes + wo_a_scale_bytes, "hbm_params": placed,
                 "hbm_expert_stacks": stack_bytes, "hbm_total": placed + stack_bytes}
    model.runtime_mtp = res
    rl = getattr(model, "runtime_load", None)
    if rl is not None:
        rl.bytes["mtp_HBM"] = placed + stack_bytes
    log(f"[load-mtp] {len(stages)} stages, {len(names)} tensors, HBM {(placed + stack_bytes) / GiB:.3f} GiB "
        f"(expert stacks {stack_bytes / GiB:.3f}, other {placed / GiB:.3f}); linears with scale {n_scale}; "
        f"wo_a dequantized {len(dequantized)}; widened {widened}")
    if verify:
        t0 = time.time()
        res.verify = verify_mtp(model, res, index, names, io_threads)
        timings["verify"] = time.time() - t0
        v = res.verify
        log(f"[verify-mtp] {v['ok']}/{v['n']} tensors byte-identical; mismatches: {v['mismatch'][:10]}")
        if v["mismatch"]:
            raise LoadRefused(f"mtp verify failed for {len(v['mismatch'])} tensors: {v['mismatch'][:20]}")
    timings["total"] = time.time() - t_all
    return res


def verify_mtp(model: nn.Module, res: MtpLoad, index, names: List[str], threads: int = 8) -> dict:
    """sha256(placed bytes) == sha256(shard bytes) for every placed mtp tensor (the loader's verify_placement rules:
    wo_a against convert.py's formula, widened tensors narrowed back, experts as stack rows)."""
    from runtime.loader import _parallel, dequant_wo_a, tensor_sha256

    stages = model.mtp

    def placed_tensor(n):
        m = MTP_EXPERT_RE.match(n)
        if m:
            i, e, p, kind = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            return res.stacks[i][p][0 if kind == "weight" else 1][e]
        mod_name, _, attr = n[len("mtp."):].rpartition(".")
        return getattr(stages.get_submodule(mod_name), attr)

    def one(n):
        e = index[n]
        t = placed_tensor(n).detach()
        if n.endswith(".attn.wo_a.weight"):
            ok = tensor_sha256(t) == tensor_sha256(dequant_wo_a(index.read_tensor(n),
                                                                index.read_tensor(n[:-6] + "scale")))
        elif t.dtype != e.dtype and not MTP_EXPERT_RE.match(n):
            ok = tensor_sha256(t.to(e.dtype)) == index.file_sha256(n)
        else:
            ok = tensor_sha256(t) == index.file_sha256(n)
        return n, ok

    out = _parallel(one, names, threads)
    mismatch = [n for n, ok in out if not ok]
    return {"n": len(out), "ok": len(out) - len(mismatch), "mismatch": mismatch}
