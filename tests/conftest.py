import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest


@pytest.fixture(autouse=True)
def _cpu_tests_stage_kernel(request, monkeypatch):
    """The runtime default is DSV41_STAGE=kernel-overlap (11/07). CPU tests that build a static step have no device
    staging kernel, so they run the plain 'kernel' path unless the test sets $DSV41_STAGE itself (the staging tests
    set it, or CPU_RESOLVE). GPU tests (tests/gpu/) keep the real default."""
    if "/gpu/" in str(request.node.fspath) or "DSV41_STAGE" in os.environ:
        return
    monkeypatch.setenv("DSV41_STAGE", "kernel")


@pytest.fixture(autouse=True)
def _restore_process_environ():
    """Runtime/bench code writes os.environ directly (gemv_mxfp8.apply_fp8_impl_arg -> DSV41_GEMV_IMPL, bench imports ->
    CUBLAS_WORKSPACE_CONFIG, bench mains -> DSV41_FP32_IMPL, dg activation -> DG_JIT_CACHE_DIR). Those writes are not
    monkeypatch-tracked and leaked into later test modules (order-dependent failures). Put os.environ back after every test."""
    saved = dict(os.environ)
    yield
    if dict(os.environ) != saved:
        os.environ.clear()
        os.environ.update(saved)
