"""Import DeepSeek's reference `model.py` (unchanged) from reference/deepseek-inference/.

model.py imports `kernel` (tilelang), `engram`, `image_processor` (PIL) and `vision` as top-level modules, so the
reference directory goes on sys.path, exactly as generate.py runs it. Two concessions, both opt-in and both for
CPU-only construction (the Mac tests); the box path uses neither:

  * `cpu_stubs=True` installs a stand-in `kernel` module whose six functions raise if called (construction needs
    only the names), and, if PIL is missing, a stand-in `image_processor` exposing the five token-type constants
    model.py imports (image_processor.py:22-23). Building a model never calls a kernel; a forward would raise.
  * On Python < 3.10 (the Mac venv is 3.9) the reference sources are compiled with `from __future__ import
    annotations` so PEP 604 annotations (`int | None`, model.py:126 and many signatures) are not evaluated. The
    file contents are untouched; nothing in model.py introspects annotations except @dataclass, which accepts
    string annotations.

Bytecode writing is disabled while importing so nothing is written under reference/.
"""
from __future__ import annotations

import __future__ as _future
import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(REPO, "reference", "deepseek-inference")
REF_MODULES = ("kernel", "engram", "image_processor", "vision", "model", "convert")
KERNEL_FUNCS = ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn")


def tilelang_available() -> bool:
    return importlib.util.find_spec("tilelang") is not None


class _FutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):  # noqa: D401
        return compile(data, path, "exec", flags=_future.annotations.compiler_flag, dont_inherit=True,
                       optimize=_optimize)


class _RefFinder(importlib.abc.MetaPathFinder):
    """Finds the reference modules in REF_DIR and loads them with the future-annotations compiler flag."""

    def __init__(self, ref_dir: str):
        self.ref_dir = ref_dir

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in REF_MODULES:
            return None
        fn = os.path.join(self.ref_dir, fullname + ".py")
        if not os.path.exists(fn):
            return None
        return importlib.util.spec_from_file_location(fullname, fn, loader=_FutureAnnotationsLoader(fullname, fn))


def _stub_kernel() -> types.ModuleType:
    mod = types.ModuleType("kernel")
    mod.__doc__ = "CPU stand-in for reference kernel.py (tilelang): construction only, every call raises."
    mod.__runtime_stub__ = True

    def make(fname):
        def f(*a, **k):
            raise RuntimeError(f"kernel.{fname}: tilelang stub (CPU construction only); run on the GPU image")

        f.__name__ = fname
        return f

    for fname in KERNEL_FUNCS:
        setattr(mod, fname, make(fname))
    return mod


def _stub_image_processor() -> types.ModuleType:
    mod = types.ModuleType("image_processor")
    mod.__doc__ = "CPU stand-in for reference image_processor.py (PIL missing): token-type constants only."
    mod.__runtime_stub__ = True
    mod.TEXT = -1  # image_processor.py:22
    mod.IMAGE_START, mod.IMAGE, mod.IMAGE_NEW_LINE, mod.IMAGE_END = range(4)  # image_processor.py:23
    return mod


def import_reference(cpu_stubs: bool = False, ref_dir: str = REF_DIR) -> types.ModuleType:
    """Return the reference `model` module. Idempotent; refuses to mix a stubbed and a real kernel module."""
    loaded = sys.modules.get("model")
    if loaded is not None and getattr(loaded, "__file__", "").startswith(ref_dir):
        if not cpu_stubs and getattr(sys.modules.get("kernel"), "__runtime_stub__", False):
            raise RuntimeError("reference model was imported with the CPU kernel stub; a real run needs a fresh process")
        return loaded
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    if cpu_stubs:
        if "kernel" not in sys.modules and not tilelang_available():
            sys.modules["kernel"] = _stub_kernel()
        if "image_processor" not in sys.modules and importlib.util.find_spec("PIL") is None:
            sys.modules["image_processor"] = _stub_image_processor()
    elif not tilelang_available():
        raise ImportError("tilelang is not installed: the reference kernels need the GPU library image "
                          "(vllm/vllm-openai:v0.31.0). Pass cpu_stubs=True for CPU-only construction.")
    finder = None
    if sys.version_info < (3, 10):
        finder = _RefFinder(ref_dir)
        sys.meta_path.insert(0, finder)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import model  # noqa: F401  (the reference)
    finally:
        sys.dont_write_bytecode = prev
        if finder is not None:
            sys.meta_path.remove(finder)
    m = sys.modules["model"]
    if not m.__file__.startswith(ref_dir):
        raise ImportError(f"imported a foreign 'model' module: {m.__file__}")
    return m


@contextlib.contextmanager
def reference_device(device):
    """The reference forward's precondition, scoped: model.py builds its index tensors with bare factories on the
    DEFAULT device (get_window_topk_idxs, model.py:418-423: `torch.arange(...)` -> the `topk_idxs` that
    kernel.sparse_attn takes; 1027-1028 likewise), and reference generate.py:134 calls
    `torch.set_default_device("cuda")` before any forward. A caller that runs `model(tokens, start_pos)` without that
    (a test, the multirow oracle, StaticStep.capture's dummy prefill) hands the CUDA-targeted TileLang kernel a CPU
    `topk_idxs` -> "sparse_attn_kernel_ input topk_idxs device_type mismatch, expected cuda" (M1-T10 round 1).
    Inside this context the default device is `device` (innermost mode wins over a global set_default_device).
    model.py memoizes two of those index builders with `@lru_cache(1)` (get_window_topk_idxs 409, get_dspark_topk_idxs
    1025): an entry made under another default device would be replayed as is, so both caches are cleared on entry
    (they hold one entry each; the reference recomputes them on its next call, the values do not change)."""
    import sys

    import torch

    m = sys.modules.get("model")
    for name in ("get_window_topk_idxs", "get_dspark_topk_idxs"):
        fn = getattr(m, name, None) if m is not None else None
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    with torch.device(device):
        yield


def reference_forward(model, tokens, start_pos: int, **kw):
    """`model(tokens, start_pos)` (the reference Transformer.forward) under `reference_device(tokens.device)`."""
    with reference_device(tokens.device):
        return model(tokens, start_pos, **kw)


def import_convert(ref_dir: str = REF_DIR) -> types.ModuleType:
    """The reference convert.py as a module (tests run its main() on the mini checkpoint). It imports tqdm at top
    level; a pass-through stand-in is installed when tqdm is missing (it only wraps iterables)."""
    if "tqdm" not in sys.modules and importlib.util.find_spec("tqdm") is None:
        tq = types.ModuleType("tqdm")
        tq.tqdm = lambda it=None, *a, **k: it
        tq.trange = lambda *a, **k: range(*a)
        tq.__runtime_stub__ = True
        sys.modules["tqdm"] = tq
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    finder = None
    if sys.version_info < (3, 10):
        finder = _RefFinder(ref_dir)
        sys.meta_path.insert(0, finder)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import convert  # noqa: F401
    finally:
        sys.dont_write_bytecode = prev
        if finder is not None:
            sys.meta_path.remove(finder)
    return sys.modules["convert"]
