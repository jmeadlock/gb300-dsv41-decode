"""P1 gracetier: the Grace cold-expert pool in MegaMoE's per-expert layout, the cold SF banks, the row_map, the source
table (docs/P1-GRACETIER-DESIGN.md sections 2.2, 3, 4; mini-test section 5 file 2).

Layout (weights only; e8m0 scales live in HBM for all G groups):

    per layer l, one region:  L1 block [C, 2I, D/2] int8   (gate/up rows interleaved at granularity 8, the bytes
                                                            deep_gemm's transform_weights_for_mega_moe produces)
                              L2 block [C, D, I/2]  int8   (w2 as is)
    C = the layer's cold count; cold-local id c = the expert's index in the rowmap's "cold" list.

Each block is ONE contiguous [C, N, K/2] tensor, so (a) can hand UVA views of it straight to a cold MegaMoE (G_c = C)
and (b)'s gt_stage copies group c from it (section 2.3: "(b)'s Grace pool, same layout"). A layer's region never
crosses a registration chunk boundary (runtime/host_tier.py: a memcpy must lie inside one registered chunk; kernels
may cross) -- the region is moved to the next chunk instead, so every expert sits inside one chunk.

row_map (the router's table, gracetier form): row[e] = hot slot of e (its index in "hot"), or H + c for cold
expert e with cold-local id c. Every entry >= 0 and the map is a permutation of 0..E-1 (a partition into hot and cold
groups). Garner's hook form (HK/:61-75) is row[e] = hot row, or -(c + 1); `row_map_from_hk` converts.

Reads: ShardIndex.read_into / read_tensor_into (runtime/ckpt.py:237-263): preadv into the destination +
POSIX_FADV_DONTNEED per chunk. w2 lands straight in its pool slot; w1 / w3 go through two reusable expert-sized CPU
buffers and are interleaved into the slot (mega_l1_weight, pure). Nothing here imports deep_gemm or vllm.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from runtime import dg_mega as DM
from runtime.host_tier import MAX_REGISTER_CHUNK, HostBuffer, check_mem_available

E_DEFAULT = 384
TOPK = 6
DIM, INTER = 5120, 2304
PROJ = ("w1", "w3", "w2")
TENSORS_PER_EXPERT = 6
REGION_ALIGN = 4096
GiB = 1024 ** 3


# ------------------------------------------------------------------------------------------------ rowmap / row_map
def load_rowmap(path: str, E: int = E_DEFAULT) -> Dict[int, Dict[str, List[int]]]:
    """{layer: {"hot": [...], "cold": [...]}} from a Garner rowmap JSON ({"layers": {...}, "meta": {...}} or the bare
    layer dict); every layer must partition 0..E-1."""
    with open(path) as f:
        raw = json.load(f)
    layers = raw["layers"] if "layers" in raw else raw
    out = {}
    for k, v in layers.items():
        hot, cold = [int(x) for x in v["hot"]], [int(x) for x in v["cold"]]
        if set(hot) | set(cold) != set(range(E)) or set(hot) & set(cold) or len(hot) + len(cold) != E:
            raise ValueError(f"rowmap {path} layer {k}: hot / cold is not a partition of 0..{E - 1}")
        out[int(k)] = {"hot": hot, "cold": cold}
    return out


def rowmap_meta(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return dict(raw.get("meta") or {})


def row_map(hot: Sequence[int], cold: Sequence[int], E: int = E_DEFAULT) -> List[int]:
    """gracetier row_map: row[e] = hot slot, or H + cold-local id."""
    H = len(hot)
    row = [-1] * E
    for s, e in enumerate(hot):
        row[int(e)] = s
    for c, e in enumerate(cold):
        row[int(e)] = H + c
    if min(row) < 0:
        raise ValueError("row_map: hot + cold do not cover every expert")
    return row


def hk_row_map(hot: Sequence[int], cold: Sequence[int], E: int = E_DEFAULT) -> List[int]:
    """Garner's ColdState.row (HK/:61-75): hot row, or -(c + 1)."""
    row = [0] * E
    for r, e in enumerate(hot):
        row[int(e)] = r
    for c, e in enumerate(cold):
        row[int(e)] = -(c + 1)
    return row


def row_map_from_hk(hk: Sequence[int], H: int) -> List[int]:
    return [int(r) if int(r) >= 0 else H + (-int(r) - 1) for r in hk]


def group_experts(row: Sequence[int]) -> List[int]:
    """Inverse of row_map: group g -> expert e."""
    inv = [-1] * len(row)
    for e, g in enumerate(row):
        inv[int(g)] = e
    return inv


def row_map_properties(row: Sequence[int], H: int, E: int = E_DEFAULT) -> dict:
    vals = [int(v) for v in row]
    return {"n": len(vals), "all_ge_0": min(vals) >= 0, "permutation": sorted(vals) == list(range(E)),
            "n_hot": sum(v < H for v in vals), "n_cold": sum(v >= H for v in vals), "H": H}


def cold_tensor_names(rowmap: Dict[int, Dict[str, List[int]]]) -> List[str]:
    """The tensors coldskip filters and gracetier reads itself (fastpath_coldskip.py:7-10, 86-98)."""
    out = []
    for l in sorted(rowmap):
        for e in rowmap[l]["cold"]:
            for p in PROJ:
                out += [f"layers.{l}.ffn.experts.{e}.{p}.weight", f"layers.{l}.ffn.experts.{e}.{p}.scale"]
    return out


def expert_names(layer: int, e: int, p: str) -> Tuple[str, str]:
    return f"layers.{layer}.ffn.experts.{e}.{p}.weight", f"layers.{layer}.ffn.experts.{e}.{p}.scale"


# ------------------------------------------------------------------------------------------------ pool layout
def l1_bytes(D: int, I: int) -> int:
    return 2 * I * D // 2


def l2_bytes(D: int, I: int) -> int:
    return D * I // 2


@dataclass
class PoolLayout:
    """Byte offsets of every layer's L1 / L2 block in one host buffer (pure)."""
    n_cold: Dict[int, int]
    D: int = DIM
    I: int = INTER
    chunk: int = MAX_REGISTER_CHUNK
    align: int = REGION_ALIGN
    l1_off: Dict[int, int] = field(default_factory=dict)
    l2_off: Dict[int, int] = field(default_factory=dict)
    total: int = 0
    pad_bytes: int = 0

    def __post_init__(self):
        if self.chunk % self.align:
            raise ValueError(f"chunk {self.chunk} is not a multiple of the region alignment {self.align}")
        b1, b2 = self.B1, self.B2
        if b1 % 16 or b2 % 16:
            raise ValueError(f"expert blocks {b1} / {b2} B are not 16-B aligned")
        off = 0
        for l in sorted(self.n_cold):
            C = int(self.n_cold[l])
            size = C * (b1 + b2)
            if size > self.chunk:
                raise ValueError(f"layer {l}: region {size} B > registration chunk {self.chunk} B")
            start = -(-off // self.align) * self.align
            if start // self.chunk != (start + max(size, 1) - 1) // self.chunk:     # would cross a chunk boundary
                start = (start // self.chunk + 1) * self.chunk
            self.pad_bytes += start - off
            self.l1_off[l] = start
            self.l2_off[l] = start + C * b1
            off = start + size
        self.total = max(off, 1)

    @property
    def B1(self) -> int:
        return l1_bytes(self.D, self.I)

    @property
    def B2(self) -> int:
        return l2_bytes(self.D, self.I)

    def expert_ranges(self, layer: int, c: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """((l1 start, l1 end), (l2 start, l2 end)) byte offsets of cold slot c of `layer`."""
        if not 0 <= c < self.n_cold[layer]:
            raise ValueError(f"layer {layer}: cold slot {c} outside [0, {self.n_cold[layer]})")
        a = self.l1_off[layer] + c * self.B1
        b = self.l2_off[layer] + c * self.B2
        return (a, a + self.B1), (b, b + self.B2)

    def chunk_of(self, off: int) -> int:
        return int(off) // self.chunk

    def describe(self) -> dict:
        return {"layers": len(self.n_cold), "D": self.D, "I": self.I, "B1": self.B1, "B2": self.B2,
                "total_B": self.total, "total_GiB": round(self.total / GiB, 3), "pad_B": self.pad_bytes,
                "chunk": self.chunk, "experts": sum(self.n_cold.values())}


class GracePool:
    """The cold pool in one HostBuffer (exact size, hugepages, cudaHostRegister'd in <= 64 GiB chunks)."""

    def __init__(self, layout: PoolLayout, name: str = "cold_experts", reserve_bytes: int = 0,
                 hugepages: bool = True):
        self.layout = layout
        self.mem = check_mem_available(layout.total, reserve_bytes, f"gracetier pool {name}")
        self.buf = HostBuffer(layout.total, name=name, hugepages=hugepages)
        self.tensors_read = 0
        self.bytes_read = 0
        self.read_s = 0.0
        self._tmp: Optional[Dict[str, torch.Tensor]] = None

    # ---------------------------------------------------------------- views
    def _shape1(self, layer: int) -> Tuple[int, int, int]:
        L = self.layout
        return (L.n_cold[layer], 2 * L.I, L.D // 2)

    def _shape2(self, layer: int) -> Tuple[int, int, int]:
        L = self.layout
        return (L.n_cold[layer], L.D, L.I // 2)

    def host_l1(self, layer: int) -> torch.Tensor:
        return self.buf.host_view(torch.int8, self._shape1(layer), self.layout.l1_off[layer])

    def host_l2(self, layer: int) -> torch.Tensor:
        return self.buf.host_view(torch.int8, self._shape2(layer), self.layout.l2_off[layer])

    def cuda_l1(self, layer: int, device=None) -> torch.Tensor:
        """UVA view (the pool must be registered)."""
        return self.buf.cuda_view(torch.int8, self._shape1(layer), self.layout.l1_off[layer], device)

    def cuda_l2(self, layer: int, device=None) -> torch.Tensor:
        return self.buf.cuda_view(torch.int8, self._shape2(layer), self.layout.l2_off[layer], device)

    def addr(self, layer: int, c: int) -> Tuple[int, int]:
        """(L1, L2) host addresses of cold slot c (= the device addresses under UVA once registered)."""
        (a, _), (b, _) = self.layout.expert_ranges(layer, c)
        return self.buf.ptr + a, self.buf.ptr + b

    def src_table(self, layer: int, device="cpu", slots: Optional[Sequence[int]] = None) -> torch.Tensor:
        """int64 [C, 2]: row c = the (L1, L2) source addresses of the expert that owns cold slot c. `slots[c]` = the
        pool slot to read for slot c (default: itself; residency / tests remap it)."""
        C = self.layout.n_cold[layer]
        sl = list(range(C)) if slots is None else [int(s) for s in slots]
        t = torch.tensor([list(self.addr(layer, s)) for s in sl], dtype=torch.int64, device="cpu")
        return t.to(device)

    # ---------------------------------------------------------------- fills
    def fill_expert(self, layer: int, c: int, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> None:
        """w1 / w3 [I, D/2], w2 [D, I/2] (int8 or uint8, CPU) -> slot c in the mega layout."""
        DM.mega_l1_weight(w1.view(torch.int8), w3.view(torch.int8), out=self.host_l1(layer)[c])
        self.host_l2(layer)[c].copy_(w2.view(torch.int8))

    def read_layer(self, idx, layer: int, cold: Sequence[int]) -> dict:
        """Read the cold experts' weights of `layer` from the checkpoint into their slots (cold-local id = position
        in `cold`). Returns {"experts", "tensors", "bytes", "s"}."""
        L = self.layout
        if len(cold) != L.n_cold[layer]:
            raise ValueError(f"layer {layer}: {len(cold)} cold ids for {L.n_cold[layer]} slots")
        if self._tmp is None:
            self._tmp = {p: torch.empty(L.I, L.D // 2, dtype=torch.int8, device="cpu") for p in ("w1", "w3")}
        t0 = time.time()
        nb = 0
        l2 = self.host_l2(layer)
        l1 = self.host_l1(layer)
        for c, e in enumerate(cold):
            for p in ("w1", "w3"):
                name = expert_names(layer, int(e), p)[0]
                _check_entry(idx, name, (L.I, L.D // 2))
                nb += idx.read_tensor_into(name, self._tmp[p])
            DM.mega_l1_weight(self._tmp["w1"], self._tmp["w3"], out=l1[c])
            name = expert_names(layer, int(e), "w2")[0]
            _check_entry(idx, name, (L.D, L.I // 2))
            nb += idx.read_tensor_into(name, l2[c])
            self.tensors_read += 3
        dt = time.time() - t0
        self.bytes_read += nb
        self.read_s += dt
        return {"experts": len(cold), "tensors": 3 * len(cold), "bytes": nb, "s": round(dt, 3)}

    def register(self) -> float:
        return self.buf.register()

    def release(self) -> None:
        self.buf.release()

    def describe(self) -> dict:
        return {"layout": self.layout.describe(), "hugepages": self.buf.hugepages,
                "registered_chunks": len(self.buf.chunks),
                "register_s": None if self.buf.register_s is None else round(self.buf.register_s, 3),
                "tensors_read": self.tensors_read, "bytes_read": self.bytes_read, "read_s": round(self.read_s, 3),
                "mem": self.mem}


def _check_entry(idx, name: str, shape: Tuple[int, int]) -> None:
    e = idx[name]
    if tuple(e.shape) != tuple(shape) or e.nbytes != shape[0] * shape[1]:
        raise ValueError(f"{name}: checkpoint shape {e.shape} ({e.st_dtype}, {e.nbytes} B), expected {shape} bytes")


# ------------------------------------------------------------------------------------------------ scales (HBM banks)
def read_codes(idx, layer: int, experts: Sequence[int], D: int = DIM, I: int = INTER) -> Dict[str, torch.Tensor]:
    """e8m0 codes uint8 {p: [n, N, K/32]} (CPU) of `experts` in the given order (3 tensors per expert)."""
    NK = {"w1": (I, D), "w3": (I, D), "w2": (D, I)}
    out = {p: torch.empty(len(experts), NK[p][0], NK[p][1] // 32, dtype=torch.uint8, device="cpu") for p in PROJ}
    for i, e in enumerate(experts):
        for p in PROJ:
            name = expert_names(layer, int(e), p)[1]
            _check_entry(idx, name, (NK[p][0], NK[p][1] // 32))
            idx.read_tensor_into(name, out[p][i])
    return out


def sf_banks(codes: Dict[str, torch.Tensor], major: str = "mn") -> Tuple[torch.Tensor, torch.Tensor]:
    """codes {p: [G, N, K/32]} in GROUP order -> (L1 SF, L2 SF) in MegaMoE's layout (packed, interleaved, UTCCP,
    `major` storage), the transform the hot experts get (runtime/dg_mega.py mega_l1_sf / mega_l2_sf)."""
    return DM.mega_l1_sf(codes["w1"], codes["w3"], major), DM.mega_l2_sf(codes["w2"], major)


def group_order_codes(hot_codes: Dict[str, torch.Tensor],
                      cold_codes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Hot slots 0..H-1 then cold slots H..G-1 (= row_map's group order)."""
    return {p: torch.cat([hot_codes[p], cold_codes[p].to(hot_codes[p].device)], 0) for p in PROJ}


def layer_bytes(G: int, D: int = DIM, I: int = INTER) -> dict:
    """Bytes of one layer at G groups: weights (L1 + L2) and SF (int32 words), the all-HBM reference's size."""
    w = G * (l1_bytes(D, I) + l2_bytes(D, I))
    sf = G * (2 * I * D // 32 + D * I // 32)
    return {"G": G, "weights_B": w, "sf_B": sf, "total_GiB": round((w + sf) / GiB, 3)}


def cold_expert_bytes(D: int = DIM, I: int = INTER) -> dict:
    """Per cold expert: weights (what gt_stage moves) and e8m0 scales (stay in HBM)."""
    w = l1_bytes(D, I) + l2_bytes(D, I)
    s = (2 * I * D + D * I) // 32
    return {"weights_B": w, "scales_B": s, "total_B": w + s}


def default_rowmap_path(repo: Optional[str] = None) -> str:
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "reference", "garner-8043a681", "fastpath", "hook", "rowmap-mix-v1-h255.json")
