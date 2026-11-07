"""Routed experts split across tiers, in reference layout, with zero-copy views handed to the unchanged `Expert`s.

Layout (M0-PORT-PLAN.md section 5(c)):
  hot  : per layer, per projection, one contiguous HBM pool `w [H, N, K/2]` (int8 bytes, viewed as
         float4_e2m1fn_x2) + `s [H, N, K/32]` (float8_e8m0fnu), H = hot experts of the layer;
  cold : one registered host pool (HostBuffer "cold_experts") holding, per layer and projection, the same
         `[C, N, K/2]` + `[C, N, K/32]` blocks, C = cold experts of the layer;
  loc  : int32 `[n_layers, n_experts, 2]` = (slot, tier) with tier 0 = HBM, 1 = HOST, on the model's device.

w1/w3 are Linear(dim, inter): N = inter, K = dim; w2 is Linear(inter, dim): N = dim, K = inter (model.py:838-840).
Each expert slice `w[slot]` / `s[slot]` is contiguous, which `fp4_gemm` asserts (kernel.py:573-582).

`bind()` gives every `Expert` of `model.layers[l].ffn.experts` parameter *views* into the pools: hot -> HBM slice,
cold -> CUDA (UVA) view of the host slice. `Expert.forward` / `MoE.forward` stay the reference's, so this is M0a's
zero-copy arm: the reference fp4_gemm reads cold weights over C2C (bench/c2c_probe.py: bit-identical, +11.5 us).

`install_counters()` adds a forward pre-hook to every routed Expert: host-side counters of calls and rows per layer
for hot and cold experts. `MoE.forward` already synced to the host to get its counts (model.py:896), so the hook
costs a Python increment per active expert and no GPU work.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from runtime.host_tier import HostBuffer
from runtime.placement import HotMap

PROJS = ("w1", "w2", "w3")
TIER_HBM, TIER_HOST = 0, 1
ALIGN = 256  # bytes; every host block starts on a 256-B boundary


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


class ExpertPool:
    def __init__(self, n_layers: int, n_experts: int, dim: int, inter: int, hot_map: HotMap, device,
                 hugepages: bool = True):
        self.n_layers, self.n_experts, self.dim, self.inter = n_layers, n_experts, dim, inter
        self.hot_map, self.device = hot_map, torch.device(device)
        self.proj_nk = {"w1": (inter, dim), "w3": (inter, dim), "w2": (dim, inter)}
        for p, (n, k) in self.proj_nk.items():
            if k % 32 or n <= 0:
                raise ValueError(f"{p}: K={k} must be a multiple of 32 (fp4 block)")
        self.hot: Dict[int, Tuple[int, ...]] = {l: hot_map.hot_ids(l) for l in range(n_layers)}
        self.cold: Dict[int, Tuple[int, ...]] = {l: hot_map.cold_ids(l) for l in range(n_layers)}
        self.slot: Dict[Tuple[int, int], Tuple[int, int]] = {}
        loc = torch.empty(n_layers, n_experts, 2, dtype=torch.int32)
        for l in range(n_layers):
            for s, e in enumerate(self.hot[l]):
                self.slot[(l, e)] = (TIER_HBM, s)
                loc[l, e, 0], loc[l, e, 1] = s, TIER_HBM
            for s, e in enumerate(self.cold[l]):
                self.slot[(l, e)] = (TIER_HOST, s)
                loc[l, e, 0], loc[l, e, 1] = s, TIER_HOST
        self.loc_cpu = loc
        self.loc = loc.to(self.device)

        # hot pools (HBM)
        self.hbm: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        for l in range(n_layers):
            h = len(self.hot[l])
            self.hbm[l] = {}
            for p, (n, k) in self.proj_nk.items():
                w = torch.empty(h, n, k // 2, dtype=torch.int8, device=self.device)
                s = torch.empty(h, n, k // 32, dtype=torch.float8_e8m0fnu, device=self.device)
                self.hbm[l][p] = (w, s)

        # cold pool (one host buffer)
        self.host_off: Dict[int, Dict[str, Tuple[int, int]]] = {}
        off = 0
        for l in range(n_layers):
            c = len(self.cold[l])
            self.host_off[l] = {}
            for p, (n, k) in self.proj_nk.items():
                w_off = off
                off = _align(off + c * n * (k // 2))
                s_off = off
                off = _align(off + c * n * (k // 32))
                self.host_off[l][p] = (w_off, s_off)
        self.cold_bytes = off
        self.host: Optional[HostBuffer] = HostBuffer(off, "cold_experts", hugepages) if off else None
        self.counters: Optional[dict] = None
        self._hooks: List = []
        # T9a-2 (runtime/dg_inplace.py): spare HBM slots appended after the hot slots of every layer's weight pools
        self.spare = 0
        self.hbm_full: Dict[int, Dict[str, torch.Tensor]] = {l: {p: self.hbm[l][p][0] for p in self.proj_nk}
                                                              for l in range(n_layers)}
        self.generation = 0  # bumped whenever pool storage moves (pointer tables built before are stale)
        self.residency = None  # A2-2: runtime/residency.py ResidencyManager, the single writer of the tables + loc

    # --------------------------------------------------------------- byte sizes
    @property
    def expert_bytes(self) -> int:
        return sum(n * (k // 2) + n * (k // 32) for n, k in self.proj_nk.values())

    def hbm_bytes(self) -> int:
        return sum(w.numel() * w.element_size() + s.numel() * s.element_size()
                   for l in self.hbm for w, s in self.hbm[l].values())

    # ------------------------------------------------------- T9a-2 spare slots
    def add_spare_slots(self, S: int, model: nn.Module) -> int:
        """Grow every layer's per-projection WEIGHT pool from [H, N, K/2] to [H+S, N, K/2] (hot slots unchanged, spare
        slots H..H+S-1 zeroed): runtime/dg_inplace.py stages cold experts there so DeepGEMM's contiguous grouped GEMM
        reads one [H+S, N, K/2] tensor per layer. Scale pools are untouched (dg-inplace keeps its scales elsewhere).
        Layer by layer: allocate, copy, rebind that layer's hot Experts' weight params to the new storage (so the old
        block is freed before the next layer grows; peak = one extra [H+S] block), release the cache. Idempotent for
        S <= self.spare. Bumps `generation`: pointer tables built before are stale (StaticStep builds after this).
        Returns the HBM bytes added."""
        S = int(S)
        if S <= self.spare:
            return 0
        rm = getattr(self, "residency", None)
        if rm is not None and rm.moved():  # A2-2: experts live in slots this copy would not carry over
            raise RuntimeError("add_spare_slots while runtime/residency.py has moved experts: restore_boot() first")
        added = 0
        with torch.inference_mode(False), torch.no_grad():
            for l in range(self.n_layers):
                experts = model.layers[l].ffn.experts
                for p in self.proj_nk:
                    w, s = self.hbm[l][p]
                    h = w.size(0)
                    full = torch.empty(h + S, *w.shape[1:], dtype=w.dtype, device=w.device)
                    full[:h].copy_(w)
                    full[h:].zero_()
                    added += (full.numel() - w.numel()) * full.element_size()
                    self.hbm_full[l][p] = full
                    self.hbm[l][p] = (full[:h], s)
                    for slot, e in enumerate(self.hot[l]):
                        lin = getattr(experts[e], p)
                        lin.weight = nn.Parameter(full[slot].view(torch.float4_e2m1fn_x2), requires_grad=False)
                        lin.weight.scale = lin.scale  # model.py:222
                    del w
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        self.spare = S
        self.generation += 1
        return added

    # ------------------------------------------------------------ destinations
    def tier(self, layer: int, expert: int) -> int:
        return self.slot[(layer, expert)][0]

    def hbm_dst(self, layer: int, expert: int, proj: str, kind: str) -> torch.Tensor:
        tier, s = self.slot[(layer, expert)]
        assert tier == TIER_HBM
        w, sc = self.hbm[layer][proj]
        return (w if kind == "weight" else sc)[s]

    def _host_block(self, layer: int, proj: str, kind: str) -> Tuple[int, torch.dtype, Tuple[int, int]]:
        n, k = self.proj_nk[proj]
        w_off, s_off = self.host_off[layer][proj]
        if kind == "weight":
            return w_off, torch.int8, (n, k // 2)
        return s_off, torch.float8_e8m0fnu, (n, k // 32)

    def host_slice(self, layer: int, expert: int, proj: str, kind: str) -> Tuple[int, torch.dtype, Tuple[int, int]]:
        """(byte offset in the cold HostBuffer, storage dtype, 2-D shape) of one cold expert tensor."""
        tier, s = self.slot[(layer, expert)]
        assert tier == TIER_HOST
        base, dt, shape = self._host_block(layer, proj, kind)
        nbytes = shape[0] * shape[1] * torch.empty(0, dtype=dt).element_size()
        return base + s * nbytes, dt, shape

    def host_memoryview(self, layer: int, expert: int, proj: str, kind: str) -> memoryview:
        off, dt, shape = self.host_slice(layer, expert, proj, kind)
        nbytes = shape[0] * shape[1] * torch.empty(0, dtype=dt).element_size()
        return self.host.memoryview(off, nbytes)

    def cold_view(self, layer: int, expert: int, proj: str, kind: str, cuda: bool) -> torch.Tensor:
        off, dt, shape = self.host_slice(layer, expert, proj, kind)
        return self.host.cuda_view(dt, shape, off, self.device) if cuda else self.host.host_view(dt, shape, off)

    # ------------------------------------------------------------------- bind
    def bind(self, model: nn.Module, cuda_views: bool) -> int:
        """Point every routed Expert's w1/w2/w3 weight+scale at its pool slice. Returns experts bound."""
        n = 0
        for l in range(self.n_layers):
            experts = model.layers[l].ffn.experts
            if len(experts) != self.n_experts:
                raise ValueError(f"layer {l}: model has {len(experts)} experts, pool {self.n_experts}")
            for e in range(self.n_experts):
                ex = experts[e]
                if ex is None:
                    raise ValueError(f"layer {l} expert {e} is None (world_size > 1?)")
                hot = self.tier(l, e) == TIER_HBM
                for p in PROJS:
                    lin = getattr(ex, p)
                    if hot:
                        w = self.hbm_dst(l, e, p, "weight")
                        s = self.hbm_dst(l, e, p, "scale")
                    else:
                        w = self.cold_view(l, e, p, "weight", cuda_views)
                        s = self.cold_view(l, e, p, "scale", cuda_views)
                    w = w.view(torch.float4_e2m1fn_x2)  # convert.py:181, zero-copy
                    if tuple(w.shape) != tuple(lin.weight.shape) or tuple(s.shape) != tuple(lin.scale.shape):
                        raise ValueError(f"layer {l} expert {e} {p}: pool {tuple(w.shape)}/{tuple(s.shape)} vs "
                                         f"model {tuple(lin.weight.shape)}/{tuple(lin.scale.shape)}")
                    if w.dtype != lin.weight.dtype or s.dtype != lin.scale.dtype:
                        raise ValueError(f"layer {l} expert {e} {p}: dtype {w.dtype}/{s.dtype} vs "
                                         f"{lin.weight.dtype}/{lin.scale.dtype}")
                    lin.weight = nn.Parameter(w, requires_grad=False)
                    lin.scale = nn.Parameter(s, requires_grad=False)
                    lin.weight.scale = lin.scale  # model.py:222, what linear() reads (model.py:192)
                n += 1
        return n

    # --------------------------------------------------------------- counters
    def install_counters(self, model: nn.Module) -> dict:
        """Per-layer host-side counters: calls/rows for hot and cold experts, plus per-(layer, expert) cold calls."""
        c = {"hot_calls": [0] * self.n_layers, "cold_calls": [0] * self.n_layers,
             "hot_rows": [0] * self.n_layers, "cold_rows": [0] * self.n_layers,
             "cold_by_expert": defaultdict(int)}
        self.counters = c
        for l in range(self.n_layers):
            for e, ex in enumerate(model.layers[l].ffn.experts):
                cold = self.tier(l, e) == TIER_HOST

                def hook(mod, inputs, l=l, e=e, cold=cold):
                    rows = int(inputs[0].shape[0]) if inputs and hasattr(inputs[0], "shape") else 0
                    if cold:
                        c["cold_calls"][l] += 1
                        c["cold_rows"][l] += rows
                        c["cold_by_expert"][(l, e)] += 1
                    else:
                        c["hot_calls"][l] += 1
                        c["hot_rows"][l] += rows

                self._hooks.append(ex.register_forward_pre_hook(hook))
        return c

    def reset_counters(self) -> None:
        if self.counters is None:
            return
        for k in ("hot_calls", "cold_calls", "hot_rows", "cold_rows"):
            self.counters[k] = [0] * self.n_layers
        self.counters["cold_by_expert"].clear()

    def counter_summary(self) -> dict:
        c = self.counters
        if c is None:
            return {}
        hot, cold = sum(c["hot_calls"]), sum(c["cold_calls"])
        return {"hot_calls": hot, "cold_calls": cold, "cold_share": cold / (hot + cold) if hot + cold else 0.0,
                "cold_calls_per_layer": list(c["cold_calls"])}

    def remove_counters(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.counters = None
