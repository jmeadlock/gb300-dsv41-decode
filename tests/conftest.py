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
