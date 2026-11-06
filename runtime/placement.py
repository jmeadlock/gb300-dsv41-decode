"""Placement plan: every checkpoint tensor -> HBM | HOST (pinned Grace, CUDA-visible by UVA) | SKIP, and the dry-run
report of bytes per tier per role.

    python -m runtime.placement --model-dir /models/DeepSeek-V4.1-Flash-1017-... [--map hot.json | --hot-per-layer 295]

CPU only; reads the shard headers and nothing else (`runtime.ckpt.ShardIndex.scan`).

Tiers (M0, PROJECT.md decision 13, M0-PORT-PLAN.md section 4):
  HOST: the two Engram tables `layers.{1,14}.engram.embed.{weight,scale}` and the cold routed experts;
  SKIP: vision tower + aligner + image_* + gate.bias_vl (vision off at M0, model.py:139-140, 809, 1219-1225) and
        every `mtp.*` tensor (DSpark off at M0, model.py:1212);
  HBM : everything else.

Placed bytes differ from on-disk bytes in exactly three documented places, which the loader cross-checks against the
reference model's own parameter dtypes before reading a byte:
  wo_a.weight  fp8 -> bf16 (convert.py:157-173; model.py:646-650), wo_a.scale consumed by that dequant;
  head.weight  bf16 -> fp32 (model.py:1007-1008: "bf16 in the checkpoint, kept as fp32");
  Compressor wkv/wgate at compress ratio > 1: -> fp32 when stored narrower (model.py:444-448).
Routed experts keep their bytes: I8 [N, K/2] is a zero-copy view as float4_e2m1fn_x2 (convert.py:181).
"""
from __future__ import annotations

import argparse
import enum
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

GiB = 1024 ** 3
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "reference", "deepseek-inference", "config.json")

EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
ENGRAM_TABLE_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
COMPRESSOR_RE = re.compile(r"^layers\.(\d+)\.attn\.compressor\.(wkv|wgate)\.weight$")
LAYER_RE = re.compile(r"^layers\.(\d+)\.")


class Tier(str, enum.Enum):
    HBM = "HBM"
    HOST = "HOST"
    SKIP = "SKIP"


@dataclass(frozen=True)
class PlacementConfig:
    """The config fields placement needs (names = ModelArgs fields = config.json keys)."""

    n_layers: int
    n_routed_experts: int
    engram_layer_ids: Tuple[int, ...]
    compress_ratios: Tuple[int, ...]
    dim: int
    moe_inter_dim: int

    @classmethod
    def from_dict(cls, cfg: dict) -> "PlacementConfig":
        return cls(
            n_layers=int(cfg["n_layers"]),
            n_routed_experts=int(cfg["n_routed_experts"]),
            engram_layer_ids=tuple(cfg.get("engram_layer_ids", ())),
            compress_ratios=tuple(cfg.get("compress_ratios", ())),
            dim=int(cfg["dim"]),
            moe_inter_dim=int(cfg["moe_inter_dim"]),
        )

    @classmethod
    def from_file(cls, path: str) -> "PlacementConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def expert_bytes(self) -> int:
        """Bytes of one routed expert: w1, w3 [inter, dim/2] + [inter, dim/32]; w2 [dim, inter/2] + [dim, inter/32]."""
        d, i = self.dim, self.moe_inter_dim
        return 2 * (i * d // 2 + i * d // 32) + (d * i // 2 + d * i // 32)


class HotMap:
    """Per-layer hot expert ids (HBM); every other id of the layer is cold (HOST).

    File format (M0): JSON `{"<layer>": [hot expert ids...]}` covering every backbone layer. Fallback:
    `HotMap.first_n(N)` takes ids 0..N-1 of every layer. Ids are kept sorted, so the slot order (and with it the
    pool layout) does not depend on the JSON's ordering."""

    def __init__(self, hot: Dict[int, Sequence[int]], n_layers: int, n_experts: int, source: str):
        self.n_layers, self.n_experts, self.source = n_layers, n_experts, source
        self.hot: Dict[int, Tuple[int, ...]] = {}
        for layer in range(n_layers):
            if layer not in hot:
                raise ValueError(f"hot map {source}: layer {layer} missing")
            ids = [int(e) for e in hot[layer]]
            if len(set(ids)) != len(ids):
                raise ValueError(f"hot map {source}: duplicate ids in layer {layer}")
            bad = [e for e in ids if not 0 <= e < n_experts]
            if bad:
                raise ValueError(f"hot map {source}: layer {layer} ids out of range {bad[:5]}")
            self.hot[layer] = tuple(sorted(ids))
        extra = sorted(set(hot) - set(range(n_layers)))
        if extra:
            raise ValueError(f"hot map {source}: layers {extra} beyond n_layers={n_layers}")
        self._sets = {layer: frozenset(ids) for layer, ids in self.hot.items()}

    @classmethod
    def load(cls, path: str, cfg: PlacementConfig) -> "HotMap":
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "hot" in raw and isinstance(raw["hot"], dict):
            raw = raw["hot"]
        return cls({int(k): v for k, v in raw.items()}, cfg.n_layers, cfg.n_routed_experts, os.path.abspath(path))

    @classmethod
    def first_n(cls, n: int, cfg: PlacementConfig) -> "HotMap":
        if not 0 <= n <= cfg.n_routed_experts:
            raise ValueError(f"--hot-per-layer {n} outside [0, {cfg.n_routed_experts}]")
        return cls({layer: range(n) for layer in range(cfg.n_layers)}, cfg.n_layers, cfg.n_routed_experts,
                   f"first-{n}-per-layer")

    def is_hot(self, layer: int, expert: int) -> bool:
        return expert in self._sets[layer]

    def hot_ids(self, layer: int) -> Tuple[int, ...]:
        return self.hot[layer]

    def cold_ids(self, layer: int) -> Tuple[int, ...]:
        return tuple(e for e in range(self.n_experts) if e not in self._sets[layer])

    def to_json(self) -> dict:
        return {str(layer): list(ids) for layer, ids in self.hot.items()}


def skip_reason(name: str) -> Optional[str]:
    if name.startswith("vision.") or name.startswith("aligner.") or name.startswith("image_"):
        return "vision"  # vision_n_layers=0 overlay: no ViT/aligner/image_* params (model.py:1219-1225)
    if name.endswith(".gate.bias_vl"):
        return "vision"  # Gate.bias_vl only exists with vision on (model.py:809)
    if name.startswith("mtp."):
        return "mtp"  # dspark_block_size=0 overlay: no DSpark blocks (model.py:1212)
    return None


def place(name: str, shape: Sequence[int], dtype, cfg: PlacementConfig, hot_map: HotMap) -> Tier:
    """Tier of one tensor (runtime name, i.e. after `ckpt.convert_name`). `shape`/`dtype` are checked, not used to
    decide: a routed expert or Engram tensor with an unexpected shape is a checkpoint mismatch, not a placement."""
    if skip_reason(name):
        return Tier.SKIP
    m = EXPERT_RE.match(name)
    if m:
        layer, expert = int(m.group(1)), int(m.group(2))
        if layer >= cfg.n_layers:
            raise ValueError(f"{name}: routed expert outside the backbone")
        if len(shape) != 2:
            raise ValueError(f"{name}: expert tensor shape {tuple(shape)}")
        return Tier.HBM if hot_map.is_hot(layer, expert) else Tier.HOST
    m = ENGRAM_TABLE_RE.match(name)
    if m:
        if int(m.group(1)) not in cfg.engram_layer_ids:
            raise ValueError(f"{name}: Engram table on a layer not in engram_layer_ids {cfg.engram_layer_ids}")
        return Tier.HOST
    return Tier.HBM


def role(name: str) -> str:
    r = skip_reason(name)
    if r:
        return f"skip:{r}"
    if EXPERT_RE.match(name):
        return "routed_experts"
    if ENGRAM_TABLE_RE.match(name):
        return "engram_table"
    if name == "embed.weight":
        return "embed"
    if name == "head.weight":
        return "head"
    if ".engram." in name:
        return "engram_proj"
    if ".ffn.shared_experts." in name:
        return "shared_experts"
    if ".ffn.gate." in name:
        return "router"
    if ".attn.indexer." in name:
        return "indexer"
    if ".attn." in name:
        return "attention"
    if ".hc_" in name or name.endswith("norm.weight"):
        return "norm_mhc"
    return "other"


def placed_dtype(name: str, st_dtype: torch.dtype, cfg: PlacementConfig) -> Optional[torch.dtype]:
    """dtype the tensor has once placed (None = consumed, nothing placed). Encodes the three documented
    transforms (module docstring); the loader asserts these equal the reference parameters' dtypes."""
    if name.endswith(".attn.wo_a.weight"):
        return torch.bfloat16
    if name.endswith(".attn.wo_a.scale"):
        return None
    if name == "head.weight":
        return torch.float32
    m = COMPRESSOR_RE.match(name)
    if m and cfg.compress_ratios and cfg.compress_ratios[int(m.group(1))] > 1:
        return torch.float32
    if EXPERT_RE.match(name) and name.endswith(".weight") and st_dtype == torch.int8:
        return torch.float4_e2m1fn_x2
    return st_dtype


def _elsize(dt: torch.dtype) -> int:
    return torch.empty(0, dtype=dt).element_size()


def placed_nbytes(name: str, shape: Sequence[int], st_dtype: torch.dtype, cfg: PlacementConfig) -> int:
    dt = placed_dtype(name, st_dtype, cfg)
    if dt is None:
        return 0
    n = 1
    for s in shape:
        n *= s
    return n * _elsize(dt)


@dataclass
class Plan:
    """The placement of one checkpoint: tier per runtime name, plus the hot map it was made with."""

    tiers: Dict[str, Tier]
    hot_map: HotMap
    cfg: PlacementConfig

    def names(self, tier: Tier) -> List[str]:
        return [n for n, t in self.tiers.items() if t is tier]


def make_plan(index, cfg: PlacementConfig, hot_map: HotMap) -> Plan:
    tiers = {}
    for name, e in index.entries.items():
        tiers[name] = place(name, e.shape, e.dtype, cfg, hot_map)
    return Plan(tiers, hot_map, cfg)


def report(index, cfg: PlacementConfig, hot_map: HotMap) -> dict:
    """Bytes per (tier, role), on disk and placed, plus the plan section 4 checks."""
    plan = make_plan(index, cfg, hot_map)
    agg = defaultdict(lambda: {"n": 0, "disk": 0, "placed": 0})
    for name, e in index.entries.items():
        t = plan.tiers[name]
        a = agg[(t.value, role(name))]
        a["n"] += 1
        a["disk"] += e.nbytes
        a["placed"] += 0 if t is Tier.SKIP else placed_nbytes(name, e.shape, e.dtype, cfg)
    tiers = defaultdict(lambda: {"n": 0, "disk": 0, "placed": 0})
    for (t, _), a in agg.items():
        for k in a:
            tiers[t][k] += a[k]
    hot_counts = [len(hot_map.hot_ids(layer)) for layer in range(cfg.n_layers)]
    rows = [{"tier": t, "role": r, **a} for (t, r), a in sorted(agg.items())]
    get = lambda t, r: agg.get((t, r), {"placed": 0})["placed"]  # noqa: E731
    return {
        "model_dir": index.model_dir,
        "shards": len(index.shards),
        "tensors": len(index.entries),
        "dropped_by_convert": len(index.dropped),
        "map": hot_map.source,
        "hot_per_layer": {"min": min(hot_counts), "max": max(hot_counts), "total": sum(hot_counts)},
        "expert_bytes": cfg.expert_bytes,
        "rows": rows,
        "tiers": dict(tiers),
        "check": {
            "hbm_routed_experts_GiB": get("HBM", "routed_experts") / GiB,
            "host_routed_experts_GiB": get("HOST", "routed_experts") / GiB,
            "host_engram_GiB": get("HOST", "engram_table") / GiB,
            "host_pinned_GiB": tiers["HOST"]["placed"] / GiB,
            "hbm_weights_GiB": tiers["HBM"]["placed"] / GiB,
            "hbm_non_expert_GiB": (tiers["HBM"]["placed"] - get("HBM", "routed_experts")) / GiB,
        },
    }


def format_report(rep: dict, hbm_budget_gib: Optional[float] = None) -> str:
    out = []
    out.append(f"model_dir {rep['model_dir']}  shards={rep['shards']} tensors={rep['tensors']} "
               f"dropped_by_convert={rep['dropped_by_convert']}")
    h = rep["hot_per_layer"]
    out.append(f"map {rep['map']}  hot/layer min={h['min']} max={h['max']} total={h['total']}  "
               f"expert={rep['expert_bytes']:,} B")
    out.append(f"{'tier':5s} {'role':16s} {'n':>7s} {'disk GiB':>10s} {'placed GiB':>11s} {'placed bytes':>16s}")
    for r in rep["rows"]:
        out.append(f"{r['tier']:5s} {r['role']:16s} {r['n']:7d} {r['disk'] / GiB:10.3f} {r['placed'] / GiB:11.3f} "
                   f"{r['placed']:16,d}")
    for t in ("HBM", "HOST", "SKIP"):
        a = rep["tiers"].get(t, {"n": 0, "disk": 0, "placed": 0})
        out.append(f"{t:5s} {'TOTAL':16s} {a['n']:7d} {a['disk'] / GiB:10.3f} {a['placed'] / GiB:11.3f} "
                   f"{a['placed']:16,d}")
    c = rep["check"]
    out.append("plan section 4 check: HBM experts {:.2f} GiB | host experts {:.2f} | Engram {:.2f} | "
               "host pinned {:.2f} | HBM weights {:.2f} (non-expert {:.2f})".format(
                   c["hbm_routed_experts_GiB"], c["host_routed_experts_GiB"], c["host_engram_GiB"],
                   c["host_pinned_GiB"], c["hbm_weights_GiB"], c["hbm_non_expert_GiB"]))
    if hbm_budget_gib is not None:
        out.append(f"HBM budget {hbm_budget_gib:.2f} GiB - weights {c['hbm_weights_GiB']:.2f} = "
                   f"{hbm_budget_gib - c['hbm_weights_GiB']:.2f} GiB for KV, staging, activations, workspaces")
    return "\n".join(out)


def default_config_path(model_dir: str) -> str:
    cand = os.path.join(model_dir, "inference", "config.json")  # the checkpoint's own copy of the reference config
    return cand if os.path.exists(cand) else DEFAULT_CONFIG


def load_hot_map(cfg: PlacementConfig, map_path: Optional[str], hot_per_layer: Optional[int]) -> HotMap:
    if map_path:
        return HotMap.load(map_path, cfg)
    return HotMap.first_n(295 if hot_per_layer is None else hot_per_layer, cfg)


def main(argv: Optional[Iterable[str]] = None) -> int:
    from runtime.ckpt import ShardIndex

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--map", default=None, help='JSON {"<layer>": [hot expert ids]}')
    ap.add_argument("--hot-per-layer", type=int, default=None, help="fallback map: first N ids per layer (default 295)")
    ap.add_argument("--config", default=None, help="reference config.json (default: <model-dir>/inference/config.json, "
                                                  "else the vendored reference/deepseek-inference/config.json)")
    ap.add_argument("--hbm-budget-gib", type=float, default=249.81)
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--no-size-check", action="store_true", help="header-only shards (skip file-size check)")
    a = ap.parse_args(argv)
    cfg_path = a.config or default_config_path(a.model_dir)
    cfg = PlacementConfig.from_file(cfg_path)
    hot = load_hot_map(cfg, a.map, a.hot_per_layer)
    index = ShardIndex.scan(a.model_dir, check_sizes=not a.no_size_check)
    rep = report(index, cfg, hot)
    rep["config"] = cfg_path
    print(f"config {cfg_path}")
    print(format_report(rep, a.hbm_budget_gib))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
