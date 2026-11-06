"""Host tier: exact-size host buffers, cudaHostRegister'd in <= 64 GiB chunks, seen by the GPU through UVA.

Why not `torch.empty(..., pin_memory=True)`: torch's caching host allocator rounds each block up to a power of two,
so a 91.6 GiB Engram table would pin 128 GiB (M0-PORT-PLAN.md section 5(b)). A `HostBuffer` is an anonymous
private mmap of exactly `nbytes` (rounded only to the page by the kernel), `madvise(MADV_HUGEPAGE)` where available.

Registration follows the pattern of vLLM's `pin_mmap_region` (not imported, decision 3): register the region in
chunks of at most 64 GiB, and on any failure unregister the chunks already registered before raising, so a failed
boot leaves no pinned memory behind. Flags default to Portable|Mapped (= 3), the combination `bench/c2c_probe.py`
measured on the GB300 (bit-identical fp4_gemm through the UVA view).

CUDA view without vLLM's `_C.get_cuda_view_from_cpu_tensor`: under UVA the registered host address *is* the
device address, so `torch.as_tensor` on an object exposing `__cuda_array_interface__` with that pointer yields a
CUDA uint8 tensor over the same bytes; `.view(dtype).view(shape)` gives the parameter. The interface object holds
a reference to its HostBuffer, so the mapping outlives every view made from it.

Pure-Python parts (allocation, host views, MemAvailable parsing) run on the Mac; everything CUDA is gated on
`torch.cuda.is_available()` and exercised by tests/gpu/test_host_tier_gpu.py on the box.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

GiB = 1024 ** 3
MAX_REGISTER_CHUNK = 64 * GiB
CUDA_HOST_REGISTER_PORTABLE = 0x01
CUDA_HOST_REGISTER_MAPPED = 0x02
CUDA_HOST_REGISTER_READ_ONLY = 0x08
DEFAULT_FLAGS = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED


def _err(r) -> int:
    """torch's cudart binding returns a cudaError enum (pybind); vLLM reads `.value`, c2c_probe compares to 0."""
    return int(getattr(r, "value", r))


def mem_available_bytes() -> Optional[int]:
    """MemAvailable from /proc/meminfo, or None where there is no /proc (the Mac)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def check_mem_available(need_bytes: int, reserve_bytes: int, stage: str) -> dict:
    """Assert MemAvailable >= need + reserve. Returns the numbers for the boot log. No-op (logged) off Linux."""
    avail = mem_available_bytes()
    rec = {"stage": stage, "mem_available_GiB": None if avail is None else round(avail / GiB, 2),
           "need_GiB": round(need_bytes / GiB, 2), "reserve_GiB": round(reserve_bytes / GiB, 2)}
    if avail is not None and avail < need_bytes + reserve_bytes:
        raise MemoryError(f"[{stage}] MemAvailable {avail / GiB:.2f} GiB < need {need_bytes / GiB:.2f} GiB + "
                          f"reserve {reserve_bytes / GiB:.2f} GiB; refusing (494 GiB Grace, no swap)")
    return rec


class _CudaArrayInterface:
    """Minimal producer for torch.as_tensor(..., device='cuda'). Keeps the owning HostBuffer alive."""

    def __init__(self, owner: "HostBuffer", ptr: int, nbytes: int):
        self._owner = owner
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


class HostBuffer:
    """One exact-size anonymous host mapping. `cpu` is a zero-copy uint8 CPU tensor over it."""

    def __init__(self, nbytes: int, name: str = "", hugepages: bool = True):
        if nbytes <= 0:
            raise ValueError(f"HostBuffer {name!r}: nbytes must be > 0")
        self.name, self.nbytes = name, int(nbytes)
        self._mm = mmap.mmap(-1, self.nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.hugepages = False
        if hugepages and hasattr(mmap, "MADV_HUGEPAGE"):
            try:
                self._mm.madvise(mmap.MADV_HUGEPAGE)
                self.hugepages = True
            except OSError:
                pass
        self._cbuf = (ctypes.c_char * self.nbytes).from_buffer(self._mm)
        self.ptr = ctypes.addressof(self._cbuf)
        self.cpu = torch.frombuffer(self._mm, dtype=torch.uint8, count=self.nbytes)
        assert self.cpu.data_ptr() == self.ptr
        self.chunks: List[Tuple[int, int]] = []  # registered (ptr, nbytes)
        self.register_s: Optional[float] = None

    # ------------------------------------------------------------------ views
    def memoryview(self, offset: int = 0, nbytes: Optional[int] = None) -> memoryview:
        nbytes = self.nbytes - offset if nbytes is None else nbytes
        self._check_range(offset, nbytes)
        return memoryview(self._mm)[offset:offset + nbytes]

    def host_view(self, dtype: torch.dtype, shape: Sequence[int], offset: int = 0) -> torch.Tensor:
        """CPU tensor over [offset, offset + prod(shape) * itemsize)."""
        n = _nbytes(dtype, shape)
        self._check_range(offset, n)
        return self.cpu[offset:offset + n].view(dtype).view(tuple(shape))

    def cuda_view(self, dtype: torch.dtype, shape: Sequence[int], offset: int = 0, device=None) -> torch.Tensor:
        # NOTE: the alias is host memory the GPU can address. Read it on the GPU (kernels, device->device copy_).
        # Measured on GB300 (2026-11-06): kernels may address the whole alias across registration chunks (the VA
        # is contiguous), but a single cudaMemcpy/clone/copy_ must lie inside ONE registered chunk — a copy that
        # crosses a chunk boundary fails with cudaErrorInvalidValue. Use `copy_chunks()` for memcpy-style reads.
        # `alias.cpu()` / `.to("cpu")` issue cudaMemcpy DeviceToHost from a host pointer and fail with invalid argument;
        # the host-side data is already `self.cpu` / `host_view(...)`.
        """CUDA tensor over the same bytes (UVA alias). Requires the range to be registered."""
        n = _nbytes(dtype, shape)
        self._check_range(offset, n)
        if not self.is_registered(offset, n):
            raise RuntimeError(f"HostBuffer {self.name!r}: [{offset},{offset + n}) not registered")
        ptr = self.ptr + offset
        dev = torch.device("cuda") if device is None else torch.device(device)
        with torch.cuda.device(dev):
            t = torch.as_tensor(_CudaArrayInterface(self, ptr, n), device=dev)
        if t.device.type != "cuda" or t.data_ptr() != ptr:
            raise RuntimeError(f"UVA alias failed for {self.name!r}: device={t.device} ptr={t.data_ptr():#x} != {ptr:#x}")
        return t.view(dtype).view(tuple(shape))

    def copy_chunks(self, dst: torch.Tensor, offset: int = 0, nbytes: Optional[int] = None) -> torch.Tensor:
        """memcpy the registered range [offset, offset+nbytes) into device tensor `dst` (uint8, flat), one copy per
        registration chunk so no copy crosses a chunk boundary."""
        nbytes = self.nbytes - offset if nbytes is None else nbytes
        self._check_range(offset, nbytes)
        src = self.cuda_view(torch.uint8, (nbytes,), offset=offset)
        pos = 0
        for p, n in self.chunks:
            lo, hi = max(p, self.ptr + offset), min(p + n, self.ptr + offset + nbytes)
            if hi > lo:
                k = hi - lo
                dst[pos:pos + k].copy_(src[lo - (self.ptr + offset):hi - (self.ptr + offset)])
                pos += k
        assert pos == nbytes, (pos, nbytes)
        return dst

    def _check_range(self, offset: int, nbytes: int) -> None:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.nbytes:
            raise ValueError(f"HostBuffer {self.name!r}: range [{offset},{offset + nbytes}) outside {self.nbytes}")

    # ------------------------------------------------------------- registration
    def is_registered(self, offset: int = 0, nbytes: Optional[int] = None) -> bool:
        nbytes = self.nbytes - offset if nbytes is None else nbytes
        lo, hi = self.ptr + offset, self.ptr + offset + nbytes
        covered = lo
        for p, n in sorted(self.chunks):
            if p <= covered < p + n:
                covered = p + n
        return covered >= hi

    def register(self, flags: int = DEFAULT_FLAGS, chunk: int = MAX_REGISTER_CHUNK) -> float:
        """cudaHostRegister the whole buffer in <= `chunk` pieces; roll back on failure. Returns seconds."""
        if self.chunks:
            raise RuntimeError(f"HostBuffer {self.name!r} already registered")
        if chunk <= 0 or chunk % mmap.PAGESIZE:
            raise ValueError(f"chunk {chunk} must be a positive multiple of the page size {mmap.PAGESIZE}")
        cudart = torch.cuda.cudart()
        t0 = time.time()
        off = 0
        try:
            while off < self.nbytes:
                n = min(chunk, self.nbytes - off)
                r = _err(cudart.cudaHostRegister(self.ptr + off, n, flags))
                if r != 0:
                    raise RuntimeError(f"cudaHostRegister({self.name!r}, offset {off}, {n} B, flags {flags}) = {r}")
                self.chunks.append((self.ptr + off, n))
                off += n
        except BaseException:
            self.release()
            raise
        self.register_s = time.time() - t0
        return self.register_s

    def release(self) -> None:
        """cudaHostUnregister every registered chunk (reverse order). Views made by cuda_view must be dead first."""
        if not self.chunks:
            return
        cudart = torch.cuda.cudart()
        errors = []
        for p, n in reversed(self.chunks):
            r = _err(cudart.cudaHostUnregister(p))
            if r != 0:
                errors.append((p, n, r))
        self.chunks = []
        if errors:
            raise RuntimeError(f"cudaHostUnregister failed for {self.name!r}: {errors}")

    def close(self) -> None:
        """Unregister and unmap. Raises BufferError if CPU views are still alive (the mapping then stays)."""
        self.release()
        self.cpu = None
        self._cbuf = None
        self._mm.close()

    def __del__(self):
        # never let the mapping go away while CUDA still holds the registration: a later mmap can reuse the
        # address and every cudaHostRegister after that returns 712 (HostMemoryAlreadyRegistered).
        try:
            if getattr(self, "chunks", None):
                self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"HostBuffer({self.name!r}, {self.nbytes / GiB:.3f} GiB, ptr={self.ptr:#x}, "
                f"registered={len(self.chunks)} chunks, hugepages={self.hugepages})")


def _nbytes(dtype: torch.dtype, shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty(0, dtype=dtype).element_size()


def register_all(buffers: Sequence[HostBuffer], reserve_bytes: int, flags: int = DEFAULT_FLAGS,
                 chunk: int = MAX_REGISTER_CHUNK) -> Dict[str, object]:
    """Register every buffer (rolling back all of them if any fails) with MemAvailable checks before and after.

    Before: the buffers are already filled, so their pages are resident and counted out of MemAvailable; the check
    is that the box still has `reserve_bytes` free. After: registration must not have eaten the reserve either."""
    log: Dict[str, object] = {"before": check_mem_available(0, reserve_bytes, "before-pin")}
    done: List[HostBuffer] = []
    try:
        for b in buffers:
            b.register(flags=flags, chunk=chunk)
            done.append(b)
    except BaseException:
        for b in reversed(done):
            b.release()
        raise
    log["after"] = check_mem_available(0, reserve_bytes, "after-pin")
    log["pinned_GiB"] = round(sum(b.nbytes for b in buffers) / GiB, 3)
    log["register_s"] = {b.name: round(b.register_s or 0.0, 3) for b in buffers}
    return log
