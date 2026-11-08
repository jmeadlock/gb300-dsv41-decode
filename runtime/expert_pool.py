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

M3-S3 (docs/M3-S3-POOL-SF-LAYOUT.md): where a slot's scale factors live, `$DSV41_POOL_SF` / `sf_layout=`:
  separate (default, today): `hbm[l][p] = (w, s)`, s = the K-major e8m0 [H, N, K/32] in HBM; DeepGEMM's packed
           MN-major copy belongs to runtime/dg_inplace.py (sf_store step / pool / resident).
  beside   the pool OWNS every slot's SF in the layout DeepGEMM's sm100_fp8_fp4_gemm_1d1d reads: `sfp[l][p]` =
           int32 [G, N, K/128] MN-major (strides (N*K/128, 1, N)), G = H + spare = the SAME slot / group index as
           `hbm_full[l][p]` (one m_indices entry addresses both: no second table). Hot groups are packed from the
           checkpoint's own bytes (`pack_hot_sf`: an int32 transpose, no value conversion); spare groups start zero
           and are written by whoever writes the slot's weights (the stage launch, a residency promotion). The K-major
           e8m0 copy of the hot scales moves to Grace (`hot_sf_host`, a HostBuffer registered with the host tier;
           `hbm[l][p][1]` is its UVA view after `bind(cuda_views=True)`): the reference prefill / ref / dg-masked read
           it over C2C, the dg-inplace decode step never does. Weights and SF physically adjacent in one slot is not
           addressable by the kernel: B and SFB are two 2-D TMA maps whose group strides are implicit (N x the B row
           pitch, K/128 x the SFB row pitch) -- the memo has the kernel lines.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from runtime.host_tier import HostBuffer
from runtime.placement import HotMap

PROJS = ("w1", "w2", "w3")
TIER_HBM, TIER_HOST = 0, 1
ALIGN = 256  # bytes; every host block starts on a 256-B boundary
POOL_SF_ENV = "DSV41_POOL_SF"
POOL_SF_LAYOUTS = ("separate", "beside")
POOL_SF_DEFAULT = "separate"
SF_PACK = 4  # e8m0 codes per int32 word (= runtime/dg_inplace.SF_PACK)


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def pool_sf_layout(v: Optional[str] = None) -> str:
    """The SF layout in force: the argument, else $DSV41_POOL_SF, else "separate"."""
    s = (v or os.environ.get(POOL_SF_ENV) or POOL_SF_DEFAULT).strip()
    if s not in POOL_SF_LAYOUTS:
        raise ValueError(f"pool SF layout {s!r} not in {POOL_SF_LAYOUTS} (${POOL_SF_ENV})")
    return s


def packed_sf_empty(G: int, N: int, C: int, device) -> torch.Tensor:
    """int32 [G, N, C] MN-major (strides (C*N, 1, N)) over zeroed storage [G][C][N] -- runtime/dg_inplace.py
    packed_sf_zeros (kept here so the pool does not import the dg module). Code 0 = 2^-127: finite."""
    if N % 4:
        raise ValueError(f"N={N}: MN-major packed SF needs N % 4 == 0 (TMA alignment)")
    return torch.zeros(G, C, N, dtype=torch.int32, device=device).transpose(1, 2)


def pack_words_e8m0(s: torch.Tensor) -> torch.Tensor:
    """e8m0 [..., N, KS] -> int32 [..., N, KS/4] over the SAME bytes (runtime/dg_inplace.py pack_words)."""
    u = s if s.dtype == torch.uint8 else s.view(torch.uint8)
    if u.stride(-1) != 1:
        u = u.contiguous()
    if u.size(-1) % SF_PACK:
        raise ValueError(f"{u.size(-1)} scales per row is not a multiple of {SF_PACK}")
    return u.view(torch.int32)


class ExpertPool:
    def __init__(self, n_layers: int, n_experts: int, dim: int, inter: int, hot_map: HotMap, device,
                 hugepages: bool = True, sf_layout: Optional[str] = None):
        self.n_layers, self.n_experts, self.dim, self.inter = n_layers, n_experts, dim, inter
        self.hot_map, self.device = hot_map, torch.device(device)
        self.sf_layout = pool_sf_layout(sf_layout)
        self.proj_nk = {"w1": (inter, dim), "w3": (inter, dim), "w2": (dim, inter)}
        for p, (n, k) in self.proj_nk.items():
            if k % 32 or n <= 0:
                raise ValueError(f"{p}: K={k} must be a multiple of 32 (fp4 block)")
            if self.sf_beside and ((k // 32) % SF_PACK or n % 4):
                raise ValueError(f"{p}: sf_layout beside needs K % 128 == 0 and N % 4 == 0 (packed MN-major SF)")
        self.hot: Dict[int, Tuple[int, ...]] = {l: hot_map.hot_ids(l) for l in range(n_layers)}
        self.cold: Dict[int, Tuple[int, ...]] = {l: hot_map.cold_ids(l) for l in range(n_layers)}
        self.slot: Dict[Tuple[int, int], Tuple[int, int]] = {}
        loc = torch.empty(n_layers, n_experts, 2, dtype=torch.int32, device="cpu")  # host mirror: explicit (cuda default device)
        for l in range(n_layers):
            for s, e in enumerate(self.hot[l]):
                self.slot[(l, e)] = (TIER_HBM, s)
                loc[l, e, 0], loc[l, e, 1] = s, TIER_HBM
            for s, e in enumerate(self.cold[l]):
                self.slot[(l, e)] = (TIER_HOST, s)
                loc[l, e, 0], loc[l, e, 1] = s, TIER_HOST
        self.loc_cpu = loc
        self.loc = loc.to(self.device)

        # M3-S3 beside: the hot experts' K-major e8m0 scales live in one host buffer (per layer, projection one
        # [H, N, K/32] block, 256-B aligned); the packed SF of every slot lives in HBM (sfp)
        self.hot_sf_off: Dict[int, Dict[str, int]] = {}
        self.hot_sf_bytes = 0
        self.hot_sf_host: Optional[HostBuffer] = None
        self.sfp: Dict[int, Dict[str, torch.Tensor]] = {}
        self.sf_packed: set = set()  # layers whose hot groups of sfp hold the packed hot scales
        if self.sf_beside:
            off = 0
            for l in range(n_layers):
                self.hot_sf_off[l] = {}
                for p, (n, k) in self.proj_nk.items():
                    self.hot_sf_off[l][p] = off
                    off = _align(off + len(self.hot[l]) * n * (k // 32))
            self.hot_sf_bytes = off
            self.hot_sf_host = HostBuffer(off, "hot_scales", hugepages) if off else None

        # hot pools (HBM)
        self.hbm: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        for l in range(n_layers):
            h = len(self.hot[l])
            self.hbm[l] = {}
            for p, (n, k) in self.proj_nk.items():
                w = torch.empty(h, n, k // 2, dtype=torch.int8, device=self.device)
                if self.sf_beside:
                    s = self._hot_sf_view(l, p, cuda=False)
                    self.sfp.setdefault(l, {})[p] = packed_sf_empty(h, n, k // 32 // SF_PACK, self.device)
                else:
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

    # --------------------------------------------------------------- M3-S3 SF layout
    @property
    def sf_beside(self) -> bool:
        return self.sf_layout == "beside"

    def _hot_sf_view(self, layer: int, proj: str, cuda: bool) -> torch.Tensor:
        """The K-major e8m0 [H, N, K/32] hot-scale block of (layer, proj) in hot_sf_host (beside)."""
        n, k = self.proj_nk[proj]
        shape = (len(self.hot[layer]), n, k // 32)
        if self.hot_sf_host is None:   # no hot expert anywhere: an empty tensor
            return torch.empty(shape, dtype=torch.float8_e8m0fnu, device="cpu")
        off = self.hot_sf_off[layer][proj]
        if cuda:
            return self.hot_sf_host.cuda_view(torch.float8_e8m0fnu, shape, off, self.device)
        return self.hot_sf_host.host_view(torch.float8_e8m0fnu, shape, off)

    def hot_scale_offset(self, layer: int, expert: int, proj: str) -> int:
        """beside: byte offset of a boot-hot expert's K-major scale block in hot_sf_host."""
        tier, s = self.slot[(layer, expert)]
        assert self.sf_beside and tier == TIER_HBM
        n, k = self.proj_nk[proj]
        return self.hot_sf_off[layer][proj] + s * n * (k // 32)

    def hot_scale_memoryview(self, layer: int, expert: int, proj: str) -> memoryview:
        """beside: the host bytes the loader reads a hot expert's scales into."""
        n, k = self.proj_nk[proj]
        assert self.hot_sf_host is not None
        return self.hot_sf_host.memoryview(self.hot_scale_offset(layer, expert, proj), n * (k // 32))

    def hot_scale_host(self, layer: int, slot: int, proj: str) -> torch.Tensor:
        """beside: CPU view of boot slot `slot`'s K-major scales (what residency digests / verify hash)."""
        return self._hot_sf_view(layer, proj, cuda=False)[slot]

    def pack_hot_sf(self, layers=None) -> int:
        """beside: hot groups of sfp[l][p] := the packed (int32 MN-major) bytes of the hot K-major scales, read from
        the HOST copy (fast CPU reads; one H2D per layer x projection). Idempotent per layer until storage moves.
        Returns the layers packed now."""
        if not self.sf_beside:
            raise RuntimeError("pack_hot_sf: the pool's SF layout is 'separate'")
        n_new = 0
        with torch.inference_mode(False), torch.no_grad():
            for l in (range(self.n_layers) if layers is None else layers):
                if l in self.sf_packed:
                    continue
                h = len(self.hot[l])
                if h:
                    for p in self.proj_nk:
                        src = pack_words_e8m0(self._hot_sf_view(l, p, cuda=False))
                        self.sfp[l][p][:h].copy_(src)
                self.sf_packed.add(l)
                n_new += 1
        return n_new

    def packed_sf(self, layer: int) -> Dict[str, torch.Tensor]:
        """beside: layer's pool-owned packed SF {proj: int32 [G, N, K/128] MN-major}, hot groups packed on first use."""
        if not self.sf_beside:
            raise RuntimeError("packed_sf: the pool's SF layout is 'separate'")
        if layer not in self.sf_packed:
            self.pack_hot_sf([layer])
        return self.sfp[layer]

    # --------------------------------------------------------------- byte sizes
    @property
    def expert_bytes(self) -> int:
        return sum(n * (k // 2) + n * (k // 32) for n, k in self.proj_nk.values())

    @property
    def weight_bytes(self) -> int:
        return sum(n * (k // 2) for n, k in self.proj_nk.values())

    @property
    def sf_group_bytes(self) -> int:
        """One slot's SF in any layout (e8m0 K-major = packed int32: the same bytes)."""
        return sum(n * (k // 32) for n, k in self.proj_nk.values())

    def hbm_bytes(self) -> int:
        """Pool bytes in HBM: separate = hot weights + hot e8m0 scales; beside = hot weights + the packed SF of every
        slot (hot and spare groups; the e8m0 copy is on the host: host_sf_bytes)."""
        if self.sf_beside:
            return (sum(w.numel() * w.element_size() for l in self.hbm for w, _ in self.hbm[l].values())
                    + sum(t.untyped_storage().nbytes() for l in self.sfp for t in self.sfp[l].values()))
        return sum(w.numel() * w.element_size() + s.numel() * s.element_size()
                   for l in self.hbm for w, s in self.hbm[l].values())

    # ------------------------------------------------------- T9a-2 spare slots
    def add_spare_slots(self, S: int, model: nn.Module) -> int:
        """Grow every layer's per-projection WEIGHT pool from [H, N, K/2] to [H+S, N, K/2] (hot slots unchanged, spare
        slots H..H+S-1 zeroed): runtime/dg_inplace.py stages cold experts there so DeepGEMM's contiguous grouped GEMM
        reads one [H+S, N, K/2] tensor per layer. Scale pools are untouched (dg-inplace keeps its scales elsewhere),
        except under the M3-S3 "beside" layout, where the pool-owned packed SF grows with the weights ([H+S] groups,
        hot groups carried over, spare groups zero). Layer by layer: allocate, copy, rebind that layer's hot Experts'
        weight params to the new storage (so the old block is freed before the next layer grows; peak = one extra
        [H+S] block), release the cache. Idempotent for S <= self.spare. Bumps `generation`: pointer tables built
        before are stale (StaticStep builds after this). Returns the HBM bytes added."""
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
                    old = self.hbm_full[l][p].size(0)  # H + the previous spare count (a re-growth adds only the rest)
                    full = torch.empty(h + S, *w.shape[1:], dtype=w.dtype, device=w.device)
                    full[:h].copy_(w)
                    full[h:].zero_()
                    added += (h + S - old) * full[0].numel() * full.element_size()
                    self.hbm_full[l][p] = full
                    self.hbm[l][p] = (full[:h], s)
                    for slot, e in enumerate(self.hot[l]):
                        lin = getattr(experts[e], p)
                        lin.weight = nn.Parameter(full[slot].view(torch.float4_e2m1fn_x2), requires_grad=False)
                        lin.weight.scale = lin.scale  # model.py:222
                    del w
                    if self.sf_beside:
                        sp = self.sfp[l][p]
                        n, c = sp.size(1), sp.size(2)
                        g = packed_sf_empty(h + S, n, c, sp.device)
                        g[:h].copy_(sp[:h])
                        added += (h + S - sp.size(0)) * n * c * 4
                        self.sfp[l][p] = g
                        del sp
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
        """Point every routed Expert's w1/w2/w3 weight+scale at its pool slice. Returns experts bound.
        beside: the hot scale views become UVA views of hot_sf_host first (cuda_views; it must be registered)."""
        if self.sf_beside:
            for l in range(self.n_layers):
                for p in self.proj_nk:
                    w, _ = self.hbm[l][p]
                    self.hbm[l][p] = (w, self._hot_sf_view(l, p, cuda=bool(cuda_views) and len(self.hot[l]) > 0))
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

    def describe_sf(self) -> dict:
        """M3-S3: the SF layout and where its bytes are."""
        GiB = 2 ** 30
        out = {"sf_layout": self.sf_layout, "hbm_pool_GiB": round(self.hbm_bytes() / GiB, 3)}
        if self.sf_beside:
            sfp = sum(t.untyped_storage().nbytes() for l in self.sfp for t in self.sfp[l].values())
            out.update({"packed_sf_hbm_GiB": round(sfp / GiB, 3), "hot_e8m0_host_GiB": round(self.hot_sf_bytes / GiB, 3),
                        "groups_per_layer": sorted({int(self.sfp[l]["w1"].size(0)) for l in self.sfp}),
                        "layers_packed": len(self.sf_packed)})
        return out

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
