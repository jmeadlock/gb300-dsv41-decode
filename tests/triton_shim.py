"""A tiny numpy stand-in for the subset of Triton that runtime/gemv_mxfp8.py's kernels use, so the Mac (no Triton,
no GPU) can execute the real `@triton.jit` function bodies program by program.

What it is for: the kernels' INDEX MATH and CONTROL FLOW -- pointer arithmetic, row/column clamping, masks, the
split-K K-block ranges, the KU unroll and its tail, the workspace layout, the reduce order, the constexpr branches.
What it is not: a compiler. It does not check Triton typing rules, layouts, tl.dot shape/dtype legality, register
pressure or speed, and its fp32 `tl.sum` order is numpy's (tests feed data whose 32-product block sums are exact in
fp32, so that order cannot matter). Every value is a numpy array; fp8 / bf16 / e8m0-byte tensors are loaded as their
exact float32 / int32 values; fp32 arithmetic is numpy float32 (one rounding per op, no FMA contraction).

    with triton_shim.installed():
        mod = load_module_under_shim("runtime/gemv_mxfp8.py")
    mod._launch(xq, xs, w, s, out, cfg, ws)     # CPU torch tensors in, results written back
"""
from __future__ import annotations

import contextlib
import importlib.util
import itertools
import sys
import types

import numpy as np
import torch

_PID = [0, 0, 0]


# ------------------------------------------------------------------------------------------------ values
def _v(x):
    return x.a if isinstance(x, T) else x


class T:
    """A block (or scalar) value: a numpy array with the operators the kernels use."""

    def __init__(self, a):
        self.a = np.asarray(a)

    # arithmetic / comparison (numpy broadcasting == Triton's for these shapes)
    def _bin(self, o, f):
        if isinstance(o, P):
            return NotImplemented
        return T(f(self.a, _v(o)))

    def __add__(self, o): return self._bin(o, lambda a, b: a + b)
    def __radd__(self, o): return T(_v(o) + self.a)
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b)
    def __rsub__(self, o): return T(_v(o) - self.a)
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b)
    def __rmul__(self, o): return T(_v(o) * self.a)
    def __floordiv__(self, o): return self._bin(o, lambda a, b: a // b)
    def __lshift__(self, o): return self._bin(o, lambda a, b: np.left_shift(a, b))
    def __rlshift__(self, o): return T(np.left_shift(_v(o), self.a))
    def __lt__(self, o): return self._bin(o, lambda a, b: a < b)
    def __le__(self, o): return self._bin(o, lambda a, b: a <= b)
    def __gt__(self, o): return self._bin(o, lambda a, b: a > b)
    def __ge__(self, o): return self._bin(o, lambda a, b: a >= b)
    def __eq__(self, o): return self._bin(o, lambda a, b: a == b)  # noqa: E704
    def __and__(self, o): return self._bin(o, lambda a, b: a & b)
    def __or__(self, o): return self._bin(o, lambda a, b: a | b)

    __hash__ = None

    def __getitem__(self, idx):
        return T(self.a[idx])

    @property
    def T(self):
        return T(self.a.T)

    @property
    def shape(self):
        return self.a.shape

    def to(self, dtype, bitcast=False):
        if bitcast:
            assert dtype is float32
            return T(self.a.astype(np.int32).view(np.float32))
        if dtype is bfloat16:
            f = torch.from_numpy(np.ascontiguousarray(self.a.astype(np.float32)))
            return T(f.to(torch.bfloat16).to(torch.float32).numpy())     # round to nearest even, as Triton
        return T(self.a.astype(dtype.np))


class P:
    """A block of pointers: a flat element buffer plus element offsets (any shape)."""

    def __init__(self, buf, off):
        self.buf, self.off = buf, np.asarray(off, dtype=np.int64)

    def __add__(self, o):
        return P(self.buf, self.off + np.asarray(_v(o)).astype(np.int64))

    __radd__ = __add__


class _DT:
    def __init__(self, name, npt):
        self.name, self.np = name, npt


int32, int64, float32, bfloat16 = _DT("int32", np.int32), _DT("int64", np.int64), _DT("float32", np.float32), \
    _DT("bfloat16", None)


class constexpr:  # noqa: N801 - Triton's name; only used as an annotation
    pass


# ------------------------------------------------------------------------------------------------ tl.*
def program_id(axis):
    return _PID[axis]


def arange(lo, hi):
    return T(np.arange(lo, hi, dtype=np.int32))


def zeros(shape, dtype):
    return T(np.zeros(shape, dtype=dtype.np))


def where(c, a, b):
    return T(np.where(_v(c), _v(a), _v(b)))


def minimum(a, b):
    r = np.minimum(_v(a), _v(b))
    return T(r) if isinstance(a, T) or isinstance(b, T) else r


def maximum(a, b):
    r = np.maximum(_v(a), _v(b))
    return T(r) if isinstance(a, T) or isinstance(b, T) else r


def sum(x, axis):  # noqa: A001 - Triton's name
    return T(np.sum(_v(x), axis=axis, dtype=np.float32))


def dot(a, b):
    a, b = _v(a), _v(b)
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0] and min(a.shape[0], b.shape[1], a.shape[1]) >= 16, \
        (a.shape, b.shape)  # tl.dot needs M, N, K >= 16
    # broadcast-multiply + sum rather than BLAS (macOS Accelerate raises spurious FP warnings); the tests' data makes
    # every 32-product sum exact in fp32, so the summation order is immaterial
    return T(np.sum(a.astype(np.float32)[:, None, :] * b.astype(np.float32).T[None, :, :], axis=2, dtype=np.float32))


def trans(x):
    return T(_v(x).T)


def static_range(n):
    return range(n)


def load(ptr, mask=None, other=None):
    vals = ptr.buf[np.clip(ptr.off, 0, ptr.buf.size - 1)]
    if mask is None:
        assert ((ptr.off >= 0) & (ptr.off < ptr.buf.size)).all(), "unmasked out-of-bounds load"
        return T(vals)
    m = np.broadcast_to(_v(mask), ptr.off.shape)
    assert ((ptr.off[m] >= 0) & (ptr.off[m] < ptr.buf.size)).all(), "masked-in out-of-bounds load"
    return T(np.where(m, vals, other if other is not None else 0))


def store(ptr, val, mask=None):
    v = np.broadcast_to(_v(val), ptr.off.shape)
    m = np.ones(ptr.off.shape, bool) if mask is None else np.broadcast_to(_v(mask), ptr.off.shape)
    off = ptr.off[m]
    assert ((off >= 0) & (off < ptr.buf.size)).all(), "out-of-bounds store"
    assert len(np.unique(off)) == len(off), "two lanes store to one address"
    ptr.buf[off] = v[m]
    STORES.append(off.copy())


STORES: list = []


# ------------------------------------------------------------------------------------------------ jit + launch
def _flat(t: torch.Tensor) -> torch.Tensor:
    """1-D view of t's storage from t's first element to the end (so a strided view's row stride indexes it the way
    the kernel's pointer arithmetic does)."""
    n = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
    return t.as_strided((n,), (1,), t.storage_offset())


def _buffer(t: torch.Tensor):
    f = _flat(t)
    if t.dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.float16):
        return f.float().numpy().copy()
    if t.dtype == torch.uint8:
        return f.to(torch.int32).numpy().copy()
    if t.dtype == torch.float32:
        return f.numpy().copy()
    raise TypeError(t.dtype)


class JitFunction:
    def __init__(self, fn):
        self.fn = fn
        self.launches = []

    def __call__(self, *a, **k):  # a @jit helper called from a kernel body
        return self.fn(*a, **k)

    def __getitem__(self, grid):
        def launch(*args, num_warps=None, num_stages=None, **constexprs):
            bufs = {}
            conv = []
            for x in args:
                if isinstance(x, torch.Tensor):
                    key = (x.data_ptr(), x.dtype)
                    if key not in bufs:
                        bufs[key] = (x, _buffer(x))
                    conv.append(P(bufs[key][1], 0))
                else:
                    conv.append(int(x))
            g = tuple(grid) + (1,) * (3 - len(grid))
            self.launches.append({"grid": tuple(grid), "constexprs": dict(constexprs)})
            for pid in itertools.product(*(range(n) for n in g)):
                _PID[:] = list(pid)
                self.fn(*conv, **constexprs)
            for t, b in bufs.values():  # write back (untouched elements round-trip exactly)
                if t.dtype in (torch.float32, torch.bfloat16):
                    _flat(t).copy_(torch.from_numpy(b).to(t.dtype))
        return launch


def jit(fn):
    return JitFunction(fn)


def cdiv(a, b):
    return -(-a // b)


def _modules():
    tl = types.ModuleType("triton.language")
    for name in ("T", "P", "int32", "int64", "float32", "bfloat16", "constexpr", "program_id", "arange", "zeros",
                 "where", "minimum", "maximum", "sum", "dot", "trans", "static_range", "load", "store"):
        setattr(tl, name, globals()[name])
    tr = types.ModuleType("triton")
    tr.jit, tr.cdiv, tr.language, tr.__version__ = jit, cdiv, tl, "numpy-shim"
    return tr, tl


@contextlib.contextmanager
def installed():
    """sys.modules['triton'] / ['triton.language'] = the shim for the duration (restored after)."""
    saved = {k: sys.modules.get(k) for k in ("triton", "triton.language")}
    tr, tl = _modules()
    sys.modules["triton"], sys.modules["triton.language"] = tr, tl
    try:
        yield tr
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load_module_under_shim(path: str, name: str = "gemv_mxfp8_under_shim"):
    """A private copy of the module at `path` imported with the shim as `triton` (the real runtime module is
    untouched)."""
    with installed():
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # dataclasses resolve string annotations through sys.modules[__module__]
        spec.loader.exec_module(mod)
    assert mod.HAVE_TRITON
    return mod
