"""Build DeepSeek's reference `Transformer` at MP=1 on one GPU and stream the HF shards straight into their tiers.

    model = build_and_load(model_dir, map="hot.json", device="cuda")            # library use
    python -m runtime.loader --model-dir D --map M [--verify]                   # boot + report

Construction (M0-PORT-PLAN.md section 3: building on `cuda` OOMs at layer 14's Engram):
  * config = the reference config.json + overlay {vision_n_layers=0, dspark_block_size=0,
    dspark_target_layer_ids=(), max_batch_size=1, max_seq_len=32768};
  * default dtype bf16 (generate.py:118: GEMM outputs and the bf16 params/caches depend on it; left set);
  * the Transformer is built under `torch.device(device)` with a TorchFunctionMode that sends `torch.empty` and
    `torch.ones` to the meta device. In model.py those two factories create every Parameter (Linear 221-238,
    ParallelEmbedding 166, RMSNorm 286, ParallelEngramEmbedding 309-310, Engram 347-348, Attention 640,
    Gate 807-809, Block 943-948, ParallelHead 1008), while the buffers use torch.zeros/full/tensor and RoPE math
    (model.py:453-455, 521-525, 664-680, 689-699; engram.py:151-154) and so land on the real device with their
    real values. The one buffer made by torch.empty, `engram_hash.cache` (engram.py:155-157), is written before
    it is read (engram.py:167) and is re-allocated empty on the device. Afterwards every parameter must be meta and
    every other buffer real, or the build is refused -- so a model.py change that breaks this rule is caught.
    (Plain `with torch.device("meta")` would also make the buffers meta and poison precompute_freqs_cis's
    lru_cache with meta tensors.)

Materialization (each parameter exactly once, in its tier):
  HBM   torch.empty on the device + file bytes through pinned staging buffers (`StagedUploader`);
  HOST  Engram tables read straight into exact-size HostBuffers, registered, assigned as CUDA (UVA) views;
  pools routed experts via runtime.expert_pool (hot -> HBM slot pool, cold -> registered host pool);
and three documented value paths, all exact (no rounding): wo_a fp8 x E8M0 -> bf16 by convert.py's formula on CPU
(convert.py:157-173), bf16 -> fp32 widening where model.py declares fp32 (head model.py:1007-1008; compressor
wkv/wgate at ratio > 1 model.py:444-448 -- what safetensors.load_model's copy_ does in generate.py:133), and the
I8 -> float4_e2m1fn_x2 reinterpretation of expert weights (convert.py:181, zero-copy). Any other dtype or shape
difference, and any missing or unexpected key, refuses the load and prints both lists.

M2-T17 (`head_dtype="bf16"` / $DSV41_HEAD_DTYPE=bf16, default fp32 until the box gate passes): the head weight
(`head.weight`, and the draft's `mtp.*.markov_head.head.weight`) stays bf16 -- its checkpoint dtype -- instead of
being widened to fp32: half the bytes in HBM (1.32 GB instead of 2.65 GB for the 129280 x 5120 head), no widening path
for those tensors (the bytes go straight to HBM like any same-dtype tensor), and every use goes through the fixed-order
kernel that widens in registers (runtime/fused_glue_r2.py `install_bf16_head`; exact, bit-equal to the fp32-stored
kernel). The reference module's ParallelHead.forward is replaced on those instances (its fp32 F.linear cannot take a
bf16 weight); everything else is untouched.

`verify=True` hashes every placed tensor (sha256) against an independent read of its shard bytes; wo_a against
convert.py's formula applied to the re-read bytes; widened tensors after narrowing back (bf16 -> fp32 -> bf16 is
exact). Host-tier parameters are additionally checked to alias their HostBuffer (data_ptr identity).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from runtime.ckpt import ShardIndex
from runtime.expert_pool import PROJS, TIER_HBM, ExpertPool, pool_sf_layout
from runtime.host_tier import GiB, HostBuffer, check_mem_available, register_all
from runtime.placement import (EXPERT_RE, ENGRAM_TABLE_RE, HotMap, PlacementConfig, Tier, default_config_path,
                               make_plan, placed_dtype, role)
from runtime.refmodel import import_reference

OVERLAY = {
    "vision_n_layers": 0,          # no ViT / aligner / image_* / bias_vl (model.py:139-140, 809, 1219-1225)
    "dspark_block_size": 0,        # no DSpark blocks (model.py:1212)
    "dspark_target_layer_ids": (),  # no main_hiddens capture (model.py:1268)
    "max_batch_size": 1,           # KV/state buffers (model.py:51)
    "max_seq_len": 32768,          # KV/state buffers (model.py:52)
}
META_FACTORIES = (torch.empty, torch.ones)
META_BUFFER_OK = {"engram_hash.cache": "engram.py:155-157, written before read at engram.py:167"}
MiB = 1 << 20


class LoadRefused(RuntimeError):
    pass


class _ParamFactoriesToMeta(TorchFunctionMode):
    """torch.empty / torch.ones -> device='meta' (see module docstring for why exactly these two)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if func in META_FACTORIES:
            kwargs["device"] = "meta"
        return func(*args, **kwargs)


# ------------------------------------------------------------------------------------------------ value paths
def dequant_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """convert.py:157-173 verbatim: FP8 x E8M0 (32x32 or 128x128 blocks) -> bf16, on CPU."""
    assert weight.size(0) % scale.size(0) == 0
    assert weight.size(1) % scale.size(1) == 0
    out_block_size = weight.size(0) // scale.size(0)
    in_block_size = weight.size(1) // scale.size(1)
    assert (out_block_size, in_block_size) in ((32, 32), (128, 128)), (weight.shape, scale.shape)
    weight = (
        weight.unflatten(0, (-1, out_block_size)).unflatten(-1, (-1, in_block_size)).float()
        * scale[:, None, :, None].float()
    )
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def _flat_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def tensor_sha256(t: torch.Tensor, chunk: int = 256 * MiB) -> str:
    """sha256 of a tensor's bytes (any device). CUDA tensors are copied back in chunks."""
    b = _flat_bytes(t.contiguous())
    h = hashlib.sha256()
    if b.device.type == "cpu":
        import ctypes

        n = b.numel()
        if n:
            h.update(memoryview((ctypes.c_char * n).from_address(b.data_ptr())).cast("B"))
        return h.hexdigest()
    for i in range(0, b.numel(), chunk):
        part = b[i:i + chunk].cpu()
        h.update(part.numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ staged upload
class StagedUploader:
    """File bytes -> pinned staging -> destination (device) tensor, `n_stage` worker threads each owning one staging
    buffer. On a CPU 'device' the same path runs with plain host staging (that is what the Mac tests exercise)."""

    def __init__(self, index: ShardIndex, device: torch.device, stage_bytes: int = 256 * MiB, n_stage: int = 4):
        self.index, self.device, self.stage_bytes, self.n_stage = index, device, stage_bytes, n_stage
        self._tls = threading.local()
        self._jobs: List[Tuple[str, int, int, torch.Tensor]] = []
        self.bytes = 0

    def _stage(self) -> torch.Tensor:
        s = getattr(self._tls, "stage", None)
        if s is None:
            pin = self.device.type == "cuda"
            s = torch.empty(self.stage_bytes, dtype=torch.uint8, pin_memory=pin)
            self._tls.stage = s
        return s

    def add(self, name: str, dst: torch.Tensor) -> None:
        """Queue the whole tensor `name` into `dst` (same byte count; dst's storage dtype may differ, e.g. fp4)."""
        nbytes = self.index[name].nbytes
        if not dst.is_contiguous():
            raise ValueError(f"{name}: destination must be contiguous (a reshape would copy and drop the write)")
        flat = _flat_bytes(dst)
        if flat.numel() != nbytes:
            raise ValueError(f"{name}: destination {flat.numel()} B != shard {nbytes} B")
        for start in range(0, nbytes, self.stage_bytes):
            n = min(self.stage_bytes, nbytes - start)
            self._jobs.append((name, start, n, flat[start:start + n]))

    def _run(self, job) -> int:
        name, start, n, dst = job
        st = self._stage()[:n]
        self.index.read_into(name, st, start, n)
        dst.copy_(st, non_blocking=False)
        return n

    def run(self) -> int:
        jobs, self._jobs = self._jobs, []
        with ThreadPoolExecutor(max_workers=self.n_stage) as ex:
            for n in ex.map(self._run, jobs):
                self.bytes += n
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self.bytes


def _parallel(fn: Callable, items: Sequence, threads: int) -> list:
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------------------------------------ results
@dataclass
class LoadResult:
    model_dir: str
    config_path: str
    device: str
    plan: object
    pool: ExpertPool
    host_buffers: Dict[str, HostBuffer]
    index: ShardIndex
    timings: Dict[str, float] = field(default_factory=dict)
    bytes: Dict[str, int] = field(default_factory=dict)
    mem_log: List[dict] = field(default_factory=list)
    dequantized: List[str] = field(default_factory=list)
    widened: List[str] = field(default_factory=list)
    verify: Optional[dict] = None
    head_dtype: str = "fp32"      # M2-T17: the head weight's dtype in HBM ("bf16": kept as stored, fixed-order kernel)
    head_bf16: List[str] = field(default_factory=list)   # the tensors kept bf16 (head, + the Markov head with load_mtp)

    def summary(self) -> dict:
        return {"model_dir": self.model_dir, "config": self.config_path, "device": self.device,
                "map": self.plan.hot_map.source, "head_dtype": self.head_dtype, "head_bf16": list(self.head_bf16),
                "bytes_GiB": {k: round(v / GiB, 3) for k, v in self.bytes.items()},
                "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
                "mem_log": self.mem_log, "dequantized_wo_a": len(self.dequantized),
                "widened_bf16_to_fp32": self.widened,
                "host_buffers": {k: repr(v) for k, v in self.host_buffers.items()},
                "verify": None if self.verify is None else {k: v for k, v in self.verify.items() if k != "per_tensor"}}

    def release(self) -> None:
        """Unregister every host buffer. Only after the model (and every view) is gone."""
        for b in self.host_buffers.values():
            b.release()


# ------------------------------------------------------------------------------------------------ construction
def load_reference_config(config_path: str, overlay: Optional[dict] = None) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.update(OVERLAY if overlay is None else overlay)
    return cfg


def build_meta_model(ref, args, tokenizer, device: torch.device) -> nn.Module:
    """Reference Transformer with every parameter on meta and every buffer real on `device` (module docstring)."""
    torch.set_default_dtype(torch.bfloat16)  # generate.py:118
    ref.precompute_freqs_cis.cache_clear()   # never hand back a tensor cached on another device
    with torch.device(device), _ParamFactoriesToMeta():
        model = ref.Transformer(args, tokenizer)
    real_params = [n for n, p in model.named_parameters() if not p.is_meta]
    if real_params:
        raise LoadRefused(f"parameters allocated during construction (expected meta): {real_params}")
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        if name not in META_BUFFER_OK:
            raise LoadRefused(f"buffer {name} ended up on meta; model.py changed? ({tuple(buf.shape)} {buf.dtype})")
        mod_name, _, attr = name.rpartition(".")
        mod = model.get_submodule(mod_name) if mod_name else model
        mod._buffers[attr] = torch.empty(buf.shape, dtype=buf.dtype, device=device)
    return model


def load_tokenizer(model_dir: str):
    """generate.py:128's AutoTokenizer; falls back to PreTrainedTokenizerFast on tokenizer.json (the checkpoint's
    tokenizer_config.json names that class) if AutoTokenizer cannot resolve the model type."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:  # unknown model_type in config.json, etc.
        from transformers import PreTrainedTokenizerFast

        print(f"[load] AutoTokenizer failed ({type(e).__name__}: {str(e)[:120]}); using tokenizer.json directly",
              file=sys.stderr)
        return PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))


def _resolve_map(map_arg, pcfg: PlacementConfig, hot_per_layer: Optional[int]) -> HotMap:
    if isinstance(map_arg, HotMap):
        return map_arg
    if isinstance(map_arg, dict):
        return HotMap({int(k): v for k, v in map_arg.items()}, pcfg.n_layers, pcfg.n_routed_experts, "dict")
    if isinstance(map_arg, str):
        return HotMap.load(map_arg, pcfg)
    return HotMap.first_n(295 if hot_per_layer is None else hot_per_layer, pcfg)


def _set_param(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    mod_name, _, attr = name.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    if attr not in mod._parameters:
        raise LoadRefused(f"{name} is not a registered parameter")
    setattr(mod, attr, nn.Parameter(tensor, requires_grad=False))


HEAD_DTYPE_ENV = "DSV41_HEAD_DTYPE"
HEAD_DTYPES = ("fp32", "bf16")


def resolve_head_dtype(head_dtype: Optional[str] = None) -> str:
    """M2-T17: the head weight's dtype in HBM: the argument, else $DSV41_HEAD_DTYPE, else "fp32" (today's path)."""
    v = head_dtype or os.environ.get(HEAD_DTYPE_ENV) or "fp32"
    if v not in HEAD_DTYPES:
        raise ValueError(f"head_dtype {v!r} (or ${HEAD_DTYPE_ENV}) not in {HEAD_DTYPES}")
    return v


def check_keys(index: ShardIndex, plan, expected: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
               pcfg: PlacementConfig, head_dtype: str = "fp32") -> List[str]:
    """Refuse on missing / unexpected keys, shape mismatch, or an undocumented dtype change. Returns the names
    to place (non-SKIP, excluding wo_a.scale, which the wo_a dequant consumes)."""
    provided = {}
    for name, tier in plan.tiers.items():
        if tier is Tier.SKIP:
            continue
        if name.endswith(".attn.wo_a.scale"):
            if name[: -len("scale")] + "weight" not in index:
                provided[name] = tier  # orphan scale -> reported as unexpected
            continue
        provided[name] = tier
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    for name in provided:
        if name.endswith(".attn.wo_a.weight") and name[: -len("weight")] + "scale" not in index:
            missing.append(name[: -len("weight")] + "scale")
    bad = []
    for name in sorted(set(provided) & set(expected)):
        e, (shape, dt) = index[name], expected[name]
        if tuple(e.shape) != tuple(shape):
            bad.append(f"{name}: shape {tuple(e.shape)} in checkpoint vs {tuple(shape)} in model")
            continue
        pd = placed_dtype(name, e.dtype, pcfg)
        if name == "head.weight" and head_dtype == "bf16" and e.dtype == torch.bfloat16:
            pd = torch.bfloat16     # M2-T17: kept in its checkpoint dtype (the model's expected dtype is overridden too)
        if pd != dt:
            bad.append(f"{name}: checkpoint {e.dtype} -> placement {pd} but model wants {dt}")
            continue
        if e.dtype != dt:
            ok = ((name.endswith(".attn.wo_a.weight") and e.dtype == torch.float8_e4m3fn and dt == torch.bfloat16)
                  or (e.dtype == torch.bfloat16 and dt == torch.float32)
                  or (EXPERT_RE.match(name) and e.dtype == torch.int8 and dt == torch.float4_e2m1fn_x2))
            if not ok:
                bad.append(f"{name}: undocumented dtype change {e.dtype} -> {dt}")
    if missing or unexpected or bad:
        msg = [f"refusing load: {len(missing)} missing, {len(unexpected)} unexpected, {len(bad)} mismatched"]
        msg.append("MISSING (in model, not in checkpoint):")
        msg += [f"  {n}" for n in missing] or ["  (none)"]
        msg.append("UNEXPECTED (in checkpoint, not in model, not SKIP-placed):")
        msg += [f"  {n}" for n in unexpected] or ["  (none)"]
        msg.append("MISMATCHED:")
        msg += [f"  {b}" for b in bad] or ["  (none)"]
        text = "\n".join(msg)
        print(text, file=sys.stderr)
        raise LoadRefused(text)
    return sorted(provided)


# ------------------------------------------------------------------------------------------------ main entry
def build_and_load(model_dir: str, map=None, device: Union[str, torch.device] = "cuda", *,
                   hot_per_layer: Optional[int] = None, config_path: Optional[str] = None, tokenizer=None,
                   overlay: Optional[dict] = None, verify: bool = False, io_threads: int = 8,
                   stage_bytes: int = 256 * MiB, n_stage: int = 4, host_chunk_bytes: int = 256 * MiB,
                   reserve_host_gib: float = 16.0, register_host: Optional[bool] = None,
                   cpu_stubs: bool = False, install_counters: bool = True, load_mtp: bool = False,
                   head_dtype: Optional[str] = None, log: Callable = print) -> nn.Module:
    """Construct the reference Transformer and load `model_dir` into it, tier by tier. Returns the model with
    `model.runtime_load` (a LoadResult: plan, pools, host buffers, timings, verify results) attached.

    `load_mtp` (M1-T11): afterwards build the DSpark draft stages and load every `mtp.*` tensor into HBM
    (runtime/mtp_loader.py `attach_mtp`: `model.mtp`, `model.runtime_mtp`); the backbone keeps the overlay.

    `map`: path to `{"<layer>": [hot ids]}` JSON, a dict, a HotMap, or None for `hot_per_layer` (default 295).
    `head_dtype` (M2-T17): "fp32" (default: the loader widens head.weight and the Markov head, model.py:1008) or "bf16"
    (kept bf16 in HBM, read through the fixed-order kernel; module docstring); None -> $DSV41_HEAD_DTYPE -> "fp32".
    `register_host`: cudaHostRegister the host tier (default: device is CUDA). On a CPU device the host tier is
    plain host memory and host-tier params are CPU views of it (the Mac test path)."""
    t_all = time.time()
    head_dtype = resolve_head_dtype(head_dtype)
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    register_host = is_cuda if register_host is None else register_host
    ref = import_reference(cpu_stubs=cpu_stubs)

    t0 = time.time()
    index = ShardIndex.scan(model_dir)
    config_path = config_path or default_config_path(model_dir)
    cfg = load_reference_config(config_path, overlay)
    pcfg = PlacementConfig.from_dict(cfg)
    hot_map = _resolve_map(map, pcfg, hot_per_layer)
    plan = make_plan(index, pcfg, hot_map)
    timings = {"scan": time.time() - t0}
    log(f"[load] {len(index.entries)} tensors in {len(index.shards)} shards, map {hot_map.source}, config {config_path}")

    # ---- construct
    t0 = time.time()
    args = ref.ModelArgs(**cfg)
    if tokenizer is None and args.engram_layer_ids:
        tokenizer = load_tokenizer(model_dir)  # the Engram hash needs it (engram.py:145; asserts 99,092 classes)
    model = build_meta_model(ref, args, tokenizer, device)
    expected = {n: (tuple(p.shape), p.dtype) for n, p in model.named_parameters()}
    if head_dtype == "bf16":   # M2-T17: head.weight stays in its checkpoint dtype (a same-dtype tensor: no widening)
        hd = index["head.weight"].dtype if "head.weight" in index else None
        if "head.weight" not in expected or hd != torch.bfloat16:
            raise LoadRefused(f"head_dtype=bf16 needs head.weight stored as bf16 in the checkpoint (got {hd})")
        expected["head.weight"] = (expected["head.weight"][0], torch.bfloat16)
    timings["construct"] = time.time() - t0

    names = check_keys(index, plan, expected, pcfg, head_dtype)
    by_tier: Dict[str, int] = {"HBM": 0, "HOST": 0}

    # ---- host-tier sizing + MemAvailable before allocating anything big
    engram_names = [n for n in names if ENGRAM_TABLE_RE.match(n)]
    pool = None
    mem_log = []
    host_need = sum(index[n].nbytes for n in engram_names)
    # cold pool size is known only once the pool lays itself out; compute it the same way first
    n_cold = sum(len(hot_map.cold_ids(l)) for l in range(pcfg.n_layers))
    host_need += n_cold * pcfg.expert_bytes
    sf_beside = pool_sf_layout() == "beside"   # M3-S3: the hot experts' e8m0 scales go to Grace too
    if sf_beside:
        n_hot = sum(len(hot_map.hot_ids(l)) for l in range(pcfg.n_layers))
        host_need += n_hot * sum(n * (k // 32) for n, k in ((pcfg.moe_inter_dim, pcfg.dim),
                                                            (pcfg.moe_inter_dim, pcfg.dim),
                                                            (pcfg.dim, pcfg.moe_inter_dim)))
    mem_log.append(check_mem_available(host_need, int(reserve_host_gib * GiB), "before-host-alloc"))

    t0 = time.time()
    pool = ExpertPool(pcfg.n_layers, pcfg.n_routed_experts, pcfg.dim, pcfg.moe_inter_dim, hot_map, device)
    host_buffers: Dict[str, HostBuffer] = {}
    for n in engram_names:
        host_buffers[n] = HostBuffer(index[n].nbytes, n)
    if pool.host is not None:
        host_buffers["cold_experts"] = pool.host
    if pool.hot_sf_host is not None:   # M3-S3 beside: registered with the host tier, UVA views after bind
        host_buffers["hot_scales"] = pool.hot_sf_host
    timings["alloc"] = time.time() - t0

    # ---- host tier: read straight into the host buffers (parallel positional reads)
    t0 = time.time()
    jobs = []
    for n in engram_names:
        hb = host_buffers[n]
        for start in range(0, hb.nbytes, host_chunk_bytes):
            k = min(host_chunk_bytes, hb.nbytes - start)
            jobs.append((n, start, k, hb.memoryview(start, k)))
    expert_names = [n for n in names if EXPERT_RE.match(n)]
    hot_expert_names = []
    for n in expert_names:
        m = EXPERT_RE.match(n)
        l, e, p, kind = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        if pool.tier(l, e) == TIER_HBM and pool.sf_beside and kind == "scale":
            jobs.append((n, 0, index[n].nbytes, pool.hot_scale_memoryview(l, e, p)))   # M3-S3: to Grace
        elif pool.tier(l, e) == TIER_HBM:
            hot_expert_names.append((n, l, e, p, kind))
        else:
            jobs.append((n, 0, index[n].nbytes, pool.host_memoryview(l, e, p, kind)))
    _parallel(lambda j: index.read_into(j[0], j[3], j[1], j[2]), jobs, io_threads)
    by_tier["HOST"] = sum(j[2] for j in jobs)
    timings["host_read"] = time.time() - t0

    # ---- HBM: plain params, hot experts, wo_a, widened
    t0 = time.time()
    up = StagedUploader(index, device, stage_bytes, n_stage)
    hbm_params: Dict[str, torch.Tensor] = {}
    dequantized, widened = [], []
    special = []
    for n in names:
        if ENGRAM_TABLE_RE.match(n) or EXPERT_RE.match(n):
            continue
        shape, dt = expected[n]
        t = torch.empty(shape, dtype=dt, device=device)
        hbm_params[n] = t
        if index[n].dtype == dt:
            up.add(n, t)
        else:
            special.append(n)
    for n, l, e, p, kind in hot_expert_names:
        up.add(n, pool.hbm_dst(l, e, p, kind))
    up.run()

    def do_special(n):
        t = hbm_params[n]
        src = index.read_tensor(n)
        if n.endswith(".attn.wo_a.weight"):
            val = dequant_wo_a(src, index.read_tensor(n[: -len("weight")] + "scale"))
            dequantized.append(n)
        else:
            val = src.to(t.dtype)  # bf16 -> fp32, exact
            widened.append(n)
        t.copy_(val)
        return src.numel() * src.element_size()

    _parallel(do_special, special, min(io_threads, 4))
    if is_cuda:
        torch.cuda.synchronize(device)
    wo_a_scale_bytes = sum(index[n[: -len("weight")] + "scale"].nbytes for n in dequantized)
    by_tier["HBM"] = up.bytes + sum(index[n].nbytes for n in special) + wo_a_scale_bytes
    timings["hbm_load"] = time.time() - t0

    # ---- pin the host tier, then views
    t0 = time.time()
    if register_host and host_buffers:
        reg = register_all(list(host_buffers.values()), int(reserve_host_gib * GiB))
        mem_log += [reg["before"], reg["after"]]
        mem_log.append({"stage": "registered", "pinned_GiB": reg["pinned_GiB"], "register_s": reg["register_s"]})
    timings["register"] = time.time() - t0

    # ---- assign parameters
    t0 = time.time()
    for n, t in hbm_params.items():
        _set_param(model, n, t)
    if engram_names and is_cuda and not register_host:
        raise LoadRefused("host tier on a CUDA device requires register_host=True")
    for n in engram_names:
        hb = host_buffers[n]
        shape, dt = expected[n]
        view = hb.cuda_view(dt, shape, 0, device) if register_host else hb.host_view(dt, shape, 0)
        _set_param(model, n, view)
    pool.bind(model, cuda_views=register_host)
    if pool.sf_beside:   # M3-S3: the packed SF of every hot slot, from the host copy just read (no value change)
        t1 = time.time()
        pool.pack_hot_sf()
        timings["pack_hot_sf"] = time.time() - t1
    n_scale = 0
    for mod in model.modules():
        if isinstance(mod, ref.Linear) and mod.scale is not None:
            mod.weight.scale = mod.scale  # model.py:222/227: linear() reads weight.scale (model.py:192, 202)
            n_scale += 1
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if still_meta:
        raise LoadRefused(f"parameters left on meta after load: {still_meta[:20]} (+{max(0, len(still_meta) - 20)})")
    if install_counters:
        pool.install_counters(model)
    timings["assign"] = time.time() - t0

    result = LoadResult(model_dir=index.model_dir, config_path=config_path, device=str(device), plan=plan, pool=pool,
                        host_buffers=host_buffers, index=index, timings=timings, mem_log=mem_log,
                        dequantized=sorted(dequantized), widened=sorted(widened), head_dtype=head_dtype)
    result.bytes = {"read_HBM": by_tier["HBM"], "read_HOST": by_tier["HOST"],
                    "pool_HBM": pool.hbm_bytes(), "pool_HOST": pool.cold_bytes,
                    "pool_HOST_hot_scales": pool.hot_sf_bytes,
                    "engram_HOST": sum(host_buffers[n].nbytes for n in engram_names),
                    "hbm_params": sum(t.numel() * t.element_size() for t in hbm_params.values())}
    model.runtime_load = result
    if head_dtype == "bf16":   # M2-T17: the bf16 head runs the fixed-order kernel (widened in registers)
        from runtime.fused_glue_r2 import install_bf16_head

        install_bf16_head(model.head)
        result.head_bf16.append("head.weight")
        result.bytes["head_HBM"] = model.head.weight.numel() * model.head.weight.element_size()
    log(f"[load] linears with scale: {n_scale}; wo_a dequantized: {len(dequantized)}; widened: {widened}; "
        f"head {head_dtype}")
    if verify:
        t0 = time.time()
        result.verify = verify_placement(model, result, io_threads)
        timings["verify"] = time.time() - t0
        v = result.verify
        log(f"[verify] {v['ok']}/{v['n']} tensors byte-identical "
            f"({v['bytes'] / GiB:.2f} GiB hashed); mismatches: {v['mismatch'][:10]}")
        if v["mismatch"]:
            raise LoadRefused(f"verify failed for {len(v['mismatch'])} tensors: {v['mismatch'][:20]}")
    if load_mtp:  # M1-T11
        from runtime.mtp_loader import attach_mtp

        attach_mtp(model, index, config_path, device, overlay=overlay, verify=verify, io_threads=io_threads,
                   stage_bytes=stage_bytes, n_stage=n_stage, cpu_stubs=cpu_stubs, log=log)
        timings["mtp"] = model.runtime_mtp.timings.get("total", 0.0)
        if head_dtype == "bf16":
            keep_markov_bf16(model, result, log)
    timings["total"] = time.time() - t_all
    return model


def keep_markov_bf16(model: nn.Module, result: LoadResult, log: Callable = print) -> int:
    """M2-T17: the draft's Markov head (`mtp.<last>.markov_head.head.weight`, [V, rank] -- 132 MB as fp32 at rank 256,
    read once per draft position) kept bf16 like the backbone's head. attach_mtp loads it widened (and verifies the
    widened bytes against the shard); here it is narrowed back -- bf16 -> fp32 -> bf16 is exact, so the stored bytes
    are the checkpoint's -- and the module's forward is the fixed-order bf16 read. Returns the bytes freed."""
    from runtime.fused_glue_r2 import install_bf16_head

    freed = 0
    for si, st in enumerate(model.mtp):
        mk = getattr(st, "markov_head", None)
        if mk is None:
            continue
        w = mk.head.weight
        if w.dtype != torch.float32:
            continue
        nb = w.detach().to(torch.bfloat16)
        if not torch.equal(nb.float(), w.detach()):    # the widened values must be exactly bf16-representable
            raise LoadRefused(f"mtp.{si}.markov_head.head.weight is not bf16-representable; refusing to narrow")
        mk.head.weight = nn.Parameter(nb, requires_grad=False)
        install_bf16_head(mk.head)
        freed += w.numel() * (w.element_size() - nb.element_size())
        result.head_bf16.append(f"mtp.{si}.markov_head.head.weight")
        del w
    if freed:
        mr = getattr(model, "runtime_mtp", None)
        if mr is not None:
            mr.bytes["hbm_params"] -= freed
            mr.bytes["hbm_total"] -= freed
        result.bytes["mtp_HBM"] = result.bytes.get("mtp_HBM", 0) - freed
        if model.head.weight.is_cuda:
            torch.cuda.empty_cache()
        log(f"[load-mtp] Markov head kept bf16: {freed / (1 << 20):.1f} MiB freed")
    return freed


# ------------------------------------------------------------------------------------------------ verify
def verify_placement(model: nn.Module, result: LoadResult, threads: int = 8) -> dict:
    """sha256(placed bytes) == sha256(shard bytes) for every placed tensor (module docstring for the exceptions)."""
    index, pool = result.index, result.pool
    params = dict(model.named_parameters())
    names = [n for n, t in result.plan.tiers.items() if t is not Tier.SKIP and not n.endswith(".attn.wo_a.scale")]

    def one(n):
        e = index[n]
        p = params[n]
        m = EXPERT_RE.match(n)
        if n in result.host_buffers:
            hb = result.host_buffers[n]
            if p.data_ptr() != hb.ptr:
                return n, False, "host param does not alias its HostBuffer"
            placed = tensor_sha256(hb.cpu)
            ref_h = index.file_sha256(n)
        elif m and pool.tier(int(m.group(1)), int(m.group(2))) != TIER_HBM:
            off, dt, shape = pool.host_slice(int(m.group(1)), int(m.group(2)), m.group(3), m.group(4))
            if p.data_ptr() != pool.host.ptr + off:
                return n, False, "cold expert param does not alias the cold pool"
            placed = tensor_sha256(pool.host.host_view(dt, shape, off))
            ref_h = index.file_sha256(n)
        elif m and pool.sf_beside and m.group(4) == "scale":   # M3-S3: a hot expert's scales live on the host
            l_, e_, p_ = int(m.group(1)), int(m.group(2)), m.group(3)
            off = pool.hot_scale_offset(l_, e_, p_)
            assert pool.hot_sf_host is not None
            if p.data_ptr() != pool.hot_sf_host.ptr + off:
                return n, False, "hot expert scale param does not alias the hot-scale host buffer"
            placed = tensor_sha256(pool.hot_scale_host(l_, pool.slot[(l_, e_)][1], p_))
            ref_h = index.file_sha256(n)
        elif n.endswith(".attn.wo_a.weight"):
            placed = tensor_sha256(p)
            ref_h = tensor_sha256(dequant_wo_a(index.read_tensor(n), index.read_tensor(n[:-6] + "scale")))
        elif p.dtype != e.dtype and not m:
            placed = tensor_sha256(p.detach().to(e.dtype))  # narrowing back is exact for widened bf16
            ref_h = index.file_sha256(n)
        else:
            placed = tensor_sha256(p)
            ref_h = index.file_sha256(n)
        return n, placed == ref_h, placed

    out = _parallel(one, names, threads)
    mismatch = [n for n, ok, _ in out if not ok]
    return {"n": len(out), "ok": len(out) - len(mismatch), "mismatch": mismatch,
            "bytes": sum(index[n].nbytes for n in names), "per_tensor": {n: h for n, _, h in out}}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Boot the reference Transformer from HF shards into HBM + pinned host.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--hot-per-layer", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--io-threads", type=int, default=8)
    ap.add_argument("--reserve-host-gib", type=float, default=16.0)
    ap.add_argument("--json", default=None, help="write the load summary here")
    ap.add_argument("--load-mtp", action="store_true", help="M1-T11: also load the mtp.* draft stages into HBM")
    ap.add_argument("--head-dtype", choices=HEAD_DTYPES, default=None,
                    help="M2-T17: head weight dtype in HBM (default: $DSV41_HEAD_DTYPE, else fp32)")
    a = ap.parse_args(argv)
    model = build_and_load(a.model_dir, a.map, a.device, hot_per_layer=a.hot_per_layer, config_path=a.config,
                           verify=a.verify, io_threads=a.io_threads, reserve_host_gib=a.reserve_host_gib,
                           load_mtp=a.load_mtp, head_dtype=a.head_dtype)
    s = model.runtime_load.summary()
    if a.load_mtp:
        s["mtp"] = model.runtime_mtp.summary()
    if torch.cuda.is_available() and a.device.startswith("cuda"):
        s["cuda_allocated_GiB"] = round(torch.cuda.memory_allocated() / GiB, 3)
        s["cuda_reserved_GiB"] = round(torch.cuda.memory_reserved() / GiB, 3)
    print(json.dumps(s, indent=1, default=str))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
