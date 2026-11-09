"""P1 gracetier (b), fixed-slot form: CUDA VMM geometry for a [G, ...] MegaMoE weight tensor whose cold groups are
backed by ONE HBM scratch shared by every layer (docs/P1-GRACETIER-DESIGN.md section 2.2, mini-test section 5 file 1).

Per layer and per weight tensor (L1 = [G, 2I, D/2] int8, L2 = [G, D, I/2] int8), with B = bytes per group:

    VA  [0, align_up(H*B))            private physical allocation of this layer: the H hot groups, plus -- when H*B
                                      is not a multiple of the granularity -- the STRADDLE granule holding the tail
                                      of hot group H-1 and the head of cold group H (private, never shared)
    VA  [align_up(H*B), align_up(G*B)) the SHARED scratch (one physical allocation per tensor kind, mapped at this
                                      offset behind every layer's VA)

so every group g sits at VA g*B of one contiguous range (what MegaMoE's weight TMA needs: one 2-D tensor addressed
at row g * shape_n, runtime/dg_mega.py:20-22). Sharing is legal because layers run in stream order and a layer only
reads cold groups that the same call staged (gt_stage, runtime/gracetier_stage.py, right before MegaMoE).

Two layers of the plan below need not share H: the shared allocation is sized by the largest shared range.

What is pure (runs on the Mac, tests/test_gracetier_cpu.py): TensorPlan's offsets / straddle / byte budget, the
arena's bookkeeping and release order (driven by a fake driver in the tests). What needs the box: CuDriver (ctypes on
libcuda.so.1: cuMemGetAllocationGranularity, cuMemCreate, cuMemAddressReserve, cuMemMap, cuMemSetAccess, cuMemUnmap,
cuMemAddressFree, cuMemRelease) and the torch view (the `__cuda_array_interface__` pattern of runtime/host_tier.py).
Whether this aliasing is invisible to torch + DeepGEMM is exactly what the mini-test's M1 measures; this module claims
nothing about it.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

GiB = 1024 ** 3
MiB = 1024 ** 2

# CUDA driver enums (cuda.h)
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_HANDLE_TYPE_NONE = 0x0
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1


def align_up(n: int, g: int) -> int:
    if g <= 0:
        raise ValueError(f"granularity {g} must be > 0")
    return -(-int(n) // g) * g


def align_down(n: int, g: int) -> int:
    if g <= 0:
        raise ValueError(f"granularity {g} must be > 0")
    return (int(n) // g) * g


# ------------------------------------------------------------------------------------------------ the plan (pure)
@dataclass(frozen=True)
class TensorPlan:
    """VA geometry of one [G, ...] tensor: G groups of B bytes, the first H private (hot), the rest shared."""
    name: str
    G: int
    H: int
    B: int
    gran: int

    def __post_init__(self):
        if not (0 <= self.H <= self.G) or self.B <= 0 or self.gran <= 0:
            raise ValueError(f"{self.name}: G {self.G}, H {self.H}, B {self.B}, gran {self.gran}")

    @property
    def hot_bytes(self) -> int:
        return self.H * self.B

    @property
    def tensor_bytes(self) -> int:
        return self.G * self.B

    @property
    def va_bytes(self) -> int:
        return align_up(self.tensor_bytes, self.gran)

    @property
    def shared_va_off(self) -> int:
        """First VA byte backed by the shared scratch (= the private allocation's size)."""
        return min(align_up(self.hot_bytes, self.gran), self.va_bytes)

    @property
    def private_bytes(self) -> int:
        return self.shared_va_off

    @property
    def shared_bytes(self) -> int:
        return self.va_bytes - self.shared_va_off

    @property
    def straddle(self) -> bool:
        return self.hot_bytes % self.gran != 0 and self.H < self.G

    @property
    def straddle_off(self) -> Optional[int]:
        """VA offset of the straddle granule (private), or None when H*B is granule-aligned."""
        return align_down(self.hot_bytes, self.gran) if self.straddle else None

    @property
    def straddle_bytes(self) -> int:
        """Private bytes beyond the hot groups (the cold head inside the straddle granule)."""
        return self.private_bytes - self.hot_bytes if self.straddle else 0

    def where(self, va_off: int) -> Tuple[str, int]:
        """VA offset -> ("private" | "shared", offset inside that physical allocation)."""
        if not 0 <= va_off < self.va_bytes:
            raise ValueError(f"{self.name}: VA offset {va_off} outside [0, {self.va_bytes})")
        if va_off < self.shared_va_off:
            return "private", va_off
        return "shared", va_off - self.shared_va_off

    def group_pieces(self, g: int) -> List[Tuple[str, int, int]]:
        """Group g's bytes as (kind, physical offset, nbytes) pieces in VA order (1 piece, or 2 across the boundary)."""
        if not 0 <= g < self.G:
            raise ValueError(f"{self.name}: group {g} outside [0, {self.G})")
        lo, hi = g * self.B, (g + 1) * self.B
        out = []
        if lo < self.shared_va_off:
            out.append(("private", lo, min(hi, self.shared_va_off) - lo))
        if hi > self.shared_va_off:
            a = max(lo, self.shared_va_off)
            out.append(("shared", a - self.shared_va_off, hi - a))
        return out

    def straddle_groups(self) -> List[int]:
        """Groups with bytes in the straddle granule (hot tail and cold head)."""
        if not self.straddle:
            return []
        s0, s1 = align_down(self.hot_bytes, self.gran), self.shared_va_off
        return [g for g in range(self.G) if g * self.B < s1 and (g + 1) * self.B > s0]

    def describe(self) -> dict:
        return {"name": self.name, "G": self.G, "H": self.H, "B": self.B, "gran": self.gran,
                "va_bytes": self.va_bytes, "private_bytes": self.private_bytes, "shared_va_off": self.shared_va_off,
                "shared_bytes": self.shared_bytes, "straddle": self.straddle, "straddle_off": self.straddle_off,
                "straddle_bytes": self.straddle_bytes, "straddle_groups": self.straddle_groups()}


def l1_group_bytes(D: int, I: int) -> int:
    """Interleaved L1 weights of one expert: [2I, D/2] int8 (two e2m1 per byte)."""
    return 2 * I * D // 2


def l2_group_bytes(D: int, I: int) -> int:
    return D * I // 2


def plan_layer(G: int, H: int, D: int, I: int, gran: int) -> Dict[str, TensorPlan]:
    return {"l1": TensorPlan("l1", G, H, l1_group_bytes(D, I), gran),
            "l2": TensorPlan("l2", G, H, l2_group_bytes(D, I), gran)}


def shared_sizes(plans: Sequence[Dict[str, TensorPlan]]) -> Dict[str, int]:
    """One shared allocation per tensor kind, sized by the largest shared range over the layers."""
    out: Dict[str, int] = {}
    for p in plans:
        for k, tp in p.items():
            out[k] = max(out.get(k, 0), tp.shared_bytes)
    return out


def hbm_budget(hs: Sequence[int], G: int, D: int, I: int, gran: int) -> dict:
    """HBM bytes of the VMM weight geometry for layers with hot counts `hs` (pure arithmetic on the shapes; the
    resident SF banks of all G groups are counted separately, as in runtime/dg_mega.hbm_bytes)."""
    plans = [plan_layer(G, h, D, I, gran) for h in hs]
    sh = shared_sizes(plans)
    priv = sum(tp.private_bytes for p in plans for tp in p.values())
    strad = sum(tp.straddle_bytes for p in plans for tp in p.values())
    hot = sum(tp.hot_bytes for p in plans for tp in p.values())
    sf = len(hs) * G * (2 * I * D // 32 + D * I // 32)
    tot = priv + sum(sh.values())
    return {"layers": len(hs), "G": G, "gran": gran, "hot_B": hot, "private_B": priv, "straddle_B": strad,
            "shared_B": dict(sh), "shared_total_B": sum(sh.values()), "weights_total_B": tot, "sf_all_groups_B": sf,
            "weights_total_GiB": round(tot / GiB, 3), "shared_total_GiB": round(sum(sh.values()) / GiB, 3),
            "straddle_GiB": round(strad / GiB, 3), "sf_GiB": round(sf / GiB, 3)}


# ------------------------------------------------------------------------------------------------ driver (box)
class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", CUmemAllocFlags)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class CuDriver:
    """The eight VMM entry points of libcuda.so.1 through ctypes (Garner's _cuda_drv pattern, B/:382-388). Every
    call raises RuntimeError naming the function and the CUresult. Needs a current context: torch's primary context,
    made current by any CUDA op (`ensure_context`)."""

    def __init__(self, lib=None):
        self.lib = lib if lib is not None else ctypes.CDLL("libcuda.so.1")
        L, u64, sz = self.lib, ctypes.c_ulonglong, ctypes.c_size_t
        L.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(CUmemAllocationProp),
                                                    ctypes.c_int]
        L.cuMemCreate.argtypes = [ctypes.POINTER(u64), sz, ctypes.POINTER(CUmemAllocationProp), u64]
        L.cuMemAddressReserve.argtypes = [ctypes.POINTER(u64), sz, sz, u64, u64]
        L.cuMemMap.argtypes = [u64, sz, sz, u64, u64]
        L.cuMemSetAccess.argtypes = [u64, sz, ctypes.POINTER(CUmemAccessDesc), sz]
        L.cuMemUnmap.argtypes = [u64, sz]
        L.cuMemAddressFree.argtypes = [u64, sz]
        L.cuMemRelease.argtypes = [u64]
        for f in ("cuMemGetAllocationGranularity", "cuMemCreate", "cuMemAddressReserve", "cuMemMap",
                  "cuMemSetAccess", "cuMemUnmap", "cuMemAddressFree", "cuMemRelease"):
            getattr(L, f).restype = ctypes.c_int

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} returned CUresult {rc}")

    @staticmethod
    def _prop(device: int) -> CUmemAllocationProp:
        p = CUmemAllocationProp()
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED
        p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        p.location.id = int(device)
        return p

    def granularity(self, device: int, recommended: bool = True) -> int:
        g = ctypes.c_size_t(0)
        p = self._prop(device)
        opt = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED if recommended else CU_MEM_ALLOC_GRANULARITY_MINIMUM
        self._check(self.lib.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p), opt),
                    "cuMemGetAllocationGranularity")
        return int(g.value)

    def create(self, nbytes: int, device: int) -> int:
        h = ctypes.c_ulonglong(0)
        p = self._prop(device)
        self._check(self.lib.cuMemCreate(ctypes.byref(h), int(nbytes), ctypes.byref(p), 0), f"cuMemCreate({nbytes})")
        return int(h.value)

    def reserve(self, nbytes: int, alignment: int = 0) -> int:
        va = ctypes.c_ulonglong(0)
        self._check(self.lib.cuMemAddressReserve(ctypes.byref(va), int(nbytes), int(alignment), 0, 0),
                    f"cuMemAddressReserve({nbytes})")
        return int(va.value)

    def map(self, va: int, nbytes: int, offset: int, handle: int) -> None:
        self._check(self.lib.cuMemMap(int(va), int(nbytes), int(offset), int(handle), 0),
                    f"cuMemMap(va {va:#x}, {nbytes}, offset {offset})")

    def set_access(self, va: int, nbytes: int, device: int) -> None:
        d = CUmemAccessDesc()
        d.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        d.location.id = int(device)
        d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(self.lib.cuMemSetAccess(int(va), int(nbytes), ctypes.byref(d), 1), f"cuMemSetAccess({va:#x})")

    def unmap(self, va: int, nbytes: int) -> None:
        self._check(self.lib.cuMemUnmap(int(va), int(nbytes)), f"cuMemUnmap({va:#x}, {nbytes})")

    def address_free(self, va: int, nbytes: int) -> None:
        self._check(self.lib.cuMemAddressFree(int(va), int(nbytes)), f"cuMemAddressFree({va:#x}, {nbytes})")

    def release(self, handle: int) -> None:
        self._check(self.lib.cuMemRelease(int(handle)), f"cuMemRelease({handle:#x})")


def ensure_context(device: int = 0) -> None:
    """Make torch's primary context current on this thread (the driver calls need one)."""
    torch.zeros(1, device=torch.device("cuda", int(device)))
    torch.cuda.synchronize(int(device))


# ------------------------------------------------------------------------------------------------ the arena
@dataclass
class PhysAlloc:
    handle: int
    nbytes: int
    name: str = ""


class VmmArena:
    """Owns every handle / reservation / mapping it made; `release()` undoes them in reverse order (unmap all,
    then free the VA reservations, then release the physical handles), idempotently. `log` records each driver
    operation (tests check the order)."""

    def __init__(self, device: int = 0, driver=None, recommended: bool = True):
        self.device = int(device)
        self.drv = driver if driver is not None else CuDriver()
        self.gran = int(self.drv.granularity(self.device, recommended))
        self.allocs: List[PhysAlloc] = []
        self.reserved: List[Tuple[int, int]] = []
        self.maps: List[Tuple[int, int]] = []
        self.log: List[tuple] = []
        self.released = False

    def alloc(self, nbytes: int, name: str = "") -> PhysAlloc:
        n = align_up(nbytes, self.gran)
        h = self.drv.create(n, self.device)
        a = PhysAlloc(h, n, name)
        self.allocs.append(a)
        self.log.append(("create", name, n))
        return a

    def reserve(self, nbytes: int) -> int:
        n = align_up(nbytes, self.gran)
        va = self.drv.reserve(n, self.gran)
        self.reserved.append((va, n))
        self.log.append(("reserve", va, n))
        return va

    def map(self, va: int, nbytes: int, phys: PhysAlloc, phys_off: int = 0) -> None:
        if va % self.gran or nbytes % self.gran or phys_off % self.gran:
            raise ValueError(f"map: va {va:#x} / {nbytes} B / offset {phys_off} not multiples of {self.gran}")
        if phys_off + nbytes > phys.nbytes:
            raise ValueError(f"map: [{phys_off}, {phys_off + nbytes}) outside allocation {phys.name!r} "
                             f"({phys.nbytes} B)")
        if not any(r <= va and va + nbytes <= r + n for r, n in self.reserved):
            raise ValueError(f"map: [{va:#x}, +{nbytes}) is not inside a reservation of this arena")
        self.drv.map(va, nbytes, phys_off, phys.handle)
        self.maps.append((va, nbytes))
        self.log.append(("map", va, nbytes, phys.name, phys_off))
        self.drv.set_access(va, nbytes, self.device)
        self.log.append(("access", va, nbytes))

    def release(self) -> None:
        if self.released:
            return
        errors = []
        for va, n in reversed(self.maps):
            try:
                self.drv.unmap(va, n)
                self.log.append(("unmap", va, n))
            except Exception as e:  # keep going: leave as little behind as possible
                errors.append(repr(e))
        for va, n in reversed(self.reserved):
            try:
                self.drv.address_free(va, n)
                self.log.append(("free", va, n))
            except Exception as e:
                errors.append(repr(e))
        for a in reversed(self.allocs):
            try:
                self.drv.release(a.handle)
                self.log.append(("release", a.name, a.nbytes))
            except Exception as e:
                errors.append(repr(e))
        self.maps, self.reserved, self.allocs = [], [], []
        self.released = True
        if errors:
            raise RuntimeError(f"VmmArena.release: {errors}")

    def describe(self) -> dict:
        return {"device": self.device, "gran": self.gran, "allocs": [(a.name, a.nbytes) for a in self.allocs],
                "alloc_bytes": sum(a.nbytes for a in self.allocs), "reserved_bytes": sum(n for _, n in self.reserved),
                "maps": len(self.maps)}


class _CAI:
    """`__cuda_array_interface__` producer over device-addressable bytes; keeps `owner` (the arena) alive."""

    def __init__(self, owner, ptr: int, nbytes: int):
        self._owner = owner
        self.__cuda_array_interface__ = {"shape": (int(nbytes),), "typestr": "|u1", "data": (int(ptr), False),
                                         "version": 3}


def device_view(owner, ptr: int, nbytes: int, dtype: torch.dtype, shape: Sequence[int], device) -> torch.Tensor:
    """A CUDA tensor over [ptr, ptr + nbytes) (VMM-mapped VA): uint8 -> dtype -> shape, data_ptr checked."""
    dev = torch.device(device)
    with torch.cuda.device(dev):
        t = torch.as_tensor(_CAI(owner, ptr, nbytes), device=dev)
    if t.device.type != "cuda" or t.data_ptr() != int(ptr):
        raise RuntimeError(f"VMM view failed: device {t.device}, ptr {t.data_ptr():#x} != {int(ptr):#x}")
    n = 1
    for s in shape:
        n *= int(s)
    want = n * torch.empty(0, dtype=dtype, device="cpu").element_size()
    if want > nbytes:
        raise ValueError(f"view {tuple(shape)} {dtype} needs {want} B > {nbytes} B")
    return t[:want].view(dtype).view(tuple(int(s) for s in shape))


class GroupedVA:
    """One layer's [G, ...] tensor on VMM: reserve va_bytes; map the private allocation (hot + straddle) at 0 and the
    shared scratch at shared_va_off. map_hot=False maps only the straddle granule privately (a 'second layer' that
    never reads its hot groups -- M1 / M4 use one to dirty the shared scratch without another hot copy)."""

    def __init__(self, arena: VmmArena, plan: TensorPlan, shared: Optional[PhysAlloc], *, map_hot: bool = True,
                 private: Optional[PhysAlloc] = None, name: str = ""):
        if plan.gran != arena.gran:
            raise ValueError(f"plan granularity {plan.gran} != arena granularity {arena.gran}")
        self.arena, self.plan, self.shared, self.map_hot = arena, plan, shared, map_hot
        self.name = name or plan.name
        self.va = arena.reserve(plan.va_bytes)
        self.private: Optional[PhysAlloc] = None
        if plan.private_bytes:
            if map_hot:
                self.private = private or arena.alloc(plan.private_bytes, f"{self.name}.private")
                arena.map(self.va, plan.private_bytes, self.private, 0)
            elif plan.straddle:
                self.private = private or arena.alloc(arena.gran, f"{self.name}.straddle")
                arena.map(self.va + align_down(plan.hot_bytes, arena.gran), arena.gran, self.private, 0)
        if plan.shared_bytes:
            if shared is None or shared.nbytes < plan.shared_bytes:
                raise ValueError(f"{self.name}: shared scratch {None if shared is None else shared.nbytes} B < "
                                 f"{plan.shared_bytes} B")
            arena.map(self.va + plan.shared_va_off, plan.shared_bytes, shared, 0)

    @property
    def readable_from(self) -> int:
        """First VA offset that is mapped (0, or the straddle granule when the hot part is not mapped)."""
        if self.map_hot or not self.plan.private_bytes:
            return 0
        return align_down(self.plan.hot_bytes, self.arena.gran) if self.plan.straddle else self.plan.shared_va_off

    def tensor(self, dtype: torch.dtype, shape: Sequence[int], device=None) -> torch.Tensor:
        """The whole [G, ...] tensor (only valid to READ where mapped; see readable_from)."""
        dev = device if device is not None else torch.device("cuda", self.arena.device)
        return device_view(self.arena, self.va, self.plan.va_bytes, dtype, shape, dev)

    def bytes_view(self, va_off: int, nbytes: int, device=None) -> torch.Tensor:
        dev = device if device is not None else torch.device("cuda", self.arena.device)
        if va_off < self.readable_from or va_off + nbytes > self.plan.va_bytes:
            raise ValueError(f"{self.name}: [{va_off}, +{nbytes}) not mapped (readable from {self.readable_from})")
        return device_view(self.arena, self.va + va_off, nbytes, torch.uint8, (nbytes,), dev)


class LayerVMM:
    """Every layer's L1 / L2 on VMM with one shared scratch per tensor kind (section 2.2): build(), tensors(),
    release(). hs = hot count per layer (names are the caller's layer ids)."""

    def __init__(self, arena: VmmArena, layers: Dict[int, int], G: int, D: int, I: int,
                 map_hot: Optional[Dict[int, bool]] = None):
        self.arena, self.G, self.D, self.I = arena, int(G), int(D), int(I)
        self.plans = {l: plan_layer(G, h, D, I, arena.gran) for l, h in layers.items()}
        sizes = shared_sizes(list(self.plans.values()))
        self.shared = {k: arena.alloc(n, f"shared.{k}") for k, n in sizes.items() if n > 0}
        self.va: Dict[int, Dict[str, GroupedVA]] = {}
        for l, p in self.plans.items():
            mh = True if map_hot is None else bool(map_hot.get(l, True))
            self.va[l] = {k: GroupedVA(arena, tp, self.shared.get(k), map_hot=mh, name=f"L{l}.{k}")
                          for k, tp in p.items()}

    def l1(self, layer: int) -> torch.Tensor:
        return self.va[layer]["l1"].tensor(torch.int8, (self.G, 2 * self.I, self.D // 2))

    def l2(self, layer: int) -> torch.Tensor:
        return self.va[layer]["l2"].tensor(torch.int8, (self.G, self.D, self.I // 2))

    def describe(self) -> dict:
        return {"G": self.G, "D": self.D, "I": self.I, "gran": self.arena.gran,
                "shared_B": {k: a.nbytes for k, a in self.shared.items()},
                "plans": {str(l): {k: tp.describe() for k, tp in p.items()} for l, p in self.plans.items()},
                "arena": self.arena.describe()}
