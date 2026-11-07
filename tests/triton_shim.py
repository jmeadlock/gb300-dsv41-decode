"""A tiny numpy stand-in for the subset of Triton that runtime/gemv_mxfp8.py's and runtime/fused_glue.py's kernels
use, so the Mac (no Triton, no GPU) can execute the real `@triton.jit` function bodies program by program. (T9b-2
added: tl.max/abs/fma, the `mul.rn/add.rn/sub.rn.f32` inline-asm forms, reshape/split/join/broadcast_to, uint8 and
fp8e4nv casts (RNE, as `fp_downcast_rounding="rtne"`), `%`, `>>`, `^`, unary minus, fp32 scalar args, the
`enable_fp_fusion` launch option, and write-back of any stored-to fp8/uint8/int buffer.)

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
    # T9b-2 kernels (runtime/fused_glue.py, the GEMV's RoPE epilogue)
    def __rand__(self, o): return T(_v(o) & self.a)
    def __mod__(self, o): return self._bin(o, lambda a, b: a % b)
    def __rshift__(self, o): return self._bin(o, lambda a, b: np.right_shift(a, b))
    def __xor__(self, o): return self._bin(o, lambda a, b: a ^ b)
    def __ne__(self, o): return self._bin(o, lambda a, b: a != b)  # noqa: E704
    def __neg__(self): return T(-self.a)
    def __rfloordiv__(self, o): return T(_v(o) // self.a)

    __hash__ = None

    # T9a-2 (runtime/static_kernels.py t9a2_stage_inplace): runtime scalar branches / loop bounds on 0-d values
    def __bool__(self):
        assert self.a.size == 1, "truth value of a block"
        return bool(self.a.reshape(()))

    def __index__(self):
        assert self.a.size == 1
        return int(self.a.reshape(()))

    __int__ = __index__

    def __getitem__(self, idx):
        return T(self.a[idx])

    @property
    def T(self):
        return T(self.a.T)

    @property
    def shape(self):
        return self.a.shape

    def to(self, dtype, bitcast=False, fp_downcast_rounding=None):
        assert fp_downcast_rounding in (None, "rtne"), fp_downcast_rounding
        if isinstance(dtype, pointer_type):                              # T9a-2: an address read from a table
            return ADDRESSES.pointer(int(self), dtype.elem)
        if bitcast:   # (reshape: np.ascontiguousarray makes a 0-d scalar 1-d)
            if dtype is int32:                                           # fp32 bits as int32
                return T(np.ascontiguousarray(self.a.astype(np.float32)).view(np.int32).reshape(self.a.shape))
            assert dtype is float32
            return T(np.ascontiguousarray(self.a.astype(np.int32)).view(np.float32).reshape(self.a.shape))
        if dtype is bfloat16 or dtype is float8e4nv:
            f = torch.from_numpy(np.ascontiguousarray(self.a.astype(np.float32)))
            tdt = torch.bfloat16 if dtype is bfloat16 else torch.float8_e4m3fn
            return T(f.to(tdt).to(torch.float32).numpy())                # round to nearest even, as Triton
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
uint8, float8e4nv = _DT("uint8", np.uint8), _DT("float8e4nv", None)


class constexpr:  # noqa: N801 - Triton's name; only used as an annotation
    pass


# T9a-2: pointers read from address tables (t9a2_stage_inplace). `ADDRESSES.register(*tensors)` declares the CPU
# tensors whose data_ptr()s the tables hold; `.to(pointer_type(dt))` on such an address gives a P over a numpy view
# that SHARES the tensor's storage (stores land in the tensor directly, no write-back).
class pointer_type:  # noqa: N801 - Triton's name
    def __init__(self, elem):
        self.elem = elem


def multiple_of(x, n):
    return x


class _AddressSpace:
    def __init__(self):
        self.tensors = []

    def register(self, *tensors):
        self.tensors.extend(tensors)

    def clear(self):
        self.tensors = []

    def pointer(self, addr: int, elem) -> "P":
        es = np.dtype(elem.np).itemsize
        for t in self.tensors:
            st = t.untyped_storage()
            base, n = st.data_ptr(), st.nbytes()
            if base <= addr < base + n:
                assert (addr - base) % es == 0 and n % es == 0, (hex(addr), es)
                raw = torch.empty(0, dtype=torch.uint8).set_(st, 0, (n,), (1,))
                return P(raw.numpy().view(elem.np), (addr - base) // es)
        raise KeyError(f"address {addr:#x} is not in any registered tensor")


ADDRESSES = _AddressSpace()


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


# T9b-2 (runtime/fused_glue.py kernels, the GEMV RoPE epilogue)
def max(x, axis):  # noqa: A001 - Triton's name
    return T(np.max(_v(x), axis=axis))


def abs(x):  # noqa: A001 - Triton's name
    return T(np.abs(_v(x)))


def fma(a, b, c):
    """fp32 fma with one rounding, exactly (the a*b product is exact in float64; TwoSum keeps the sum's error)."""
    a, b, c = (np.asarray(_v(t), dtype=np.float32) for t in (a, b, c))
    a, b, c = np.broadcast_arrays(a, b, c)
    p = a.astype(np.float64) * b.astype(np.float64)
    cd = c.astype(np.float64)
    s = p + cd
    bb = s - p
    e = (p - (s - bb)) + (cd - bb)
    r = s.astype(np.float32)
    rd = r.astype(np.float64)
    n = np.nextafter(r, np.where(s > rd, np.float32(np.inf), np.float32(-np.inf))).astype(np.float32)
    mid = ((rd + n.astype(np.float64)) * 0.5 == s) & (rd != s)
    take_n = mid & (e != 0) & ((e > 0) == (s > rd))
    return T(np.where(take_n, n, r).astype(np.float32))


_ASM = {"mul.rn.f32": lambda a, b: a * b, "add.rn.f32": lambda a, b: a + b, "sub.rn.f32": lambda a, b: a - b}


def inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack):
    """Only the single-instruction fp32 forms runtime/fused_glue.py uses (`mul.rn.f32 $0, $1, $2;` etc.)."""
    op = asm.split()[0]
    assert op in _ASM and dtype is float32 and pack == 1, asm
    a, b = (np.asarray(_v(t), dtype=np.float32) for t in args)
    return T(_ASM[op](a, b).astype(np.float32))


def reshape(x, shape):
    return T(np.reshape(_v(x), tuple(shape)))


def split(x):
    a = _v(x)
    assert a.shape[-1] == 2
    return T(a[..., 0]), T(a[..., 1])


def join(a, b):
    return T(np.stack([_v(a), _v(b)], axis=-1))


def broadcast_to(x, shape):
    return T(np.broadcast_to(_v(x), tuple(shape)).copy())


def static_range(n):
    return range(n)


# T9b-2 R2 (runtime/fused_glue_r2.py): `triton.language.extra.libdevice` -- rsqrt is torch.rsqrt on fp32 (what the
# CPU mirrors call; on the device libdevice.rsqrt == torch.rsqrt was measured bit for bit, docs/flashmla-probe-
# 20261106.json probe (c)), sqrt_rn / div_rn are IEEE (numpy float32 is correctly rounded).
def _rsqrt(x):
    a = np.asarray(_v(x), dtype=np.float32)
    return T(torch.rsqrt(torch.from_numpy(np.ascontiguousarray(a).reshape(-1))).numpy().reshape(a.shape))


def _sqrt_rn(x):
    return T(np.sqrt(np.asarray(_v(x), dtype=np.float32)).astype(np.float32))


def _div_rn(a, b):
    return T((np.asarray(_v(a), dtype=np.float32) / np.asarray(_v(b), dtype=np.float32)).astype(np.float32))


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
    DIRTY.add(id(ptr.buf))


STORES: list = []
DIRTY: set = set()  # id() of the buffers a launch stored to (fp8 / uint8 / int64 ones are written back only then)


def atomic_add(ptr, val, mask=None):
    """M1-T10-FIX2 (t9a2_stage_inplace's overflow counter): scalar or block add at the pointer(s); returns the old
    value(s). Programs run one after another here, so the read-modify-write is trivially atomic."""
    off = np.atleast_1d(ptr.off)
    m = np.ones(off.shape, bool) if mask is None else np.broadcast_to(_v(mask), off.shape)
    assert ((off[m] >= 0) & (off[m] < ptr.buf.size)).all(), "out-of-bounds atomic"
    old = ptr.buf[off].copy()
    np.add.at(ptr.buf, off[m], np.broadcast_to(np.asarray(_v(val)), off.shape)[m].astype(ptr.buf.dtype))
    DIRTY.add(id(ptr.buf))
    return T(old.reshape(np.shape(ptr.off)))


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
    if t.dtype in (torch.int64, torch.int32):
        return f.numpy().copy()
    raise TypeError(t.dtype)


def _write_back(t: torch.Tensor, b) -> None:
    if t.dtype in (torch.float32, torch.bfloat16):
        _flat(t).copy_(torch.from_numpy(b).to(t.dtype))
    elif t.dtype in (torch.float8_e4m3fn, torch.float16):
        _flat(t).copy_(torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to(t.dtype))
    elif t.dtype == torch.uint8:
        assert ((b >= 0) & (b <= 255)).all()
        _flat(t).copy_(torch.from_numpy(np.ascontiguousarray(b).astype(np.int32)).to(torch.uint8))
    elif t.dtype in (torch.int64, torch.int32):
        _flat(t).copy_(torch.from_numpy(np.ascontiguousarray(b)).to(t.dtype))


class JitFunction:
    def __init__(self, fn):
        self.fn = fn
        self.launches = []

    def __call__(self, *a, **k):  # a @jit helper called from a kernel body
        return self.fn(*a, **k)

    def __getitem__(self, grid):
        def launch(*args, num_warps=None, num_stages=None, enable_fp_fusion=None, **constexprs):
            bufs = {}
            conv = []
            for x in args:
                if isinstance(x, torch.Tensor):
                    key = (x.data_ptr(), x.dtype)
                    if key not in bufs:
                        bufs[key] = (x, _buffer(x))
                    conv.append(P(bufs[key][1], 0))
                elif isinstance(x, float):
                    conv.append(np.float32(x))    # Triton passes a Python float as an fp32 scalar
                else:
                    conv.append(int(x))
            g = tuple(grid) + (1,) * (3 - len(grid))
            self.launches.append({"grid": tuple(grid), "constexprs": dict(constexprs),
                                  "enable_fp_fusion": enable_fp_fusion})
            DIRTY.clear()
            for pid in itertools.product(*(range(n) for n in g)):
                _PID[:] = list(pid)
                self.fn(*conv, **constexprs)
            for t, b in bufs.values():  # write back (untouched elements round-trip exactly)
                if t.dtype in (torch.float32, torch.bfloat16) or id(b) in DIRTY:
                    _write_back(t, b)
        return launch


def jit(fn):
    return JitFunction(fn)


def cdiv(a, b):
    return -(-a // b)


def _modules():
    tl = types.ModuleType("triton.language")
    for name in ("T", "P", "int32", "int64", "float32", "bfloat16", "constexpr", "program_id", "arange", "zeros",
                 "where", "minimum", "maximum", "sum", "dot", "trans", "static_range", "load", "store",
                 "uint8", "float8e4nv", "max", "abs", "fma", "inline_asm_elementwise", "reshape", "split", "join",
                 "broadcast_to", "pointer_type", "multiple_of", "atomic_add"):
        setattr(tl, name, globals()[name])
    tr = types.ModuleType("triton")
    tr.jit, tr.cdiv, tr.language, tr.__version__ = jit, cdiv, tl, "numpy-shim"
    ex = types.ModuleType("triton.language.extra")
    ld = types.ModuleType("triton.language.extra.libdevice")
    ld.rsqrt, ld.sqrt_rn, ld.div_rn = _rsqrt, _sqrt_rn, _div_rn
    ex.libdevice = ld
    tl.extra = ex
    tr.next_power_of_2 = lambda n: 1 if int(n) <= 1 else 1 << (int(n) - 1).bit_length()  # (this module shadows max)
    return tr, tl


_SHIM_KEYS = ("triton", "triton.language", "triton.language.extra", "triton.language.extra.libdevice")


@contextlib.contextmanager
def installed():
    """sys.modules['triton'] / ['triton.language'] (+ .extra.libdevice) = the shim for the duration (restored after)."""
    saved = {k: sys.modules.get(k) for k in _SHIM_KEYS}
    tr, tl = _modules()
    sys.modules["triton"], sys.modules["triton.language"] = tr, tl
    sys.modules["triton.language.extra"], sys.modules["triton.language.extra.libdevice"] = tl.extra, tl.extra.libdevice
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
