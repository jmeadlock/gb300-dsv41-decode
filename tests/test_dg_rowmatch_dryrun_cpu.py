"""T9a-3: tests/gpu/test_dg_rowmatch_gpu.py's gate logic executed on the CPU emulations (DEV "cpu"; the fake deep_gemm
of tests/test_dg_inplace_cpu.py, row-wise emulated fp4/fp8 GEMMs of tests/test_static_rows_cpu.py): the box run then
meets only the real kernels, not harness bugs. The mini4 6-row dg-inplace step (contiguous16, S = 8) == the 1-row
dg-inplace step (rowmatch on and off) on every row's fp32 logits, both P0 parities, with every non-MoE op one row at a
time (row_invariant_mode); and the 1-row build's per-row check runs and passes."""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys

import pytest
import torch

from test_static_rows_cpu import emu, rows_dg  # noqa: F401 (fixtures)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gpu_module(monkeypatch, rows_dg_):  # noqa: F811
    for p in (os.path.join(REPO, "tests", "gpu"), os.path.join(REPO, "bench")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("gpu_rowmatch", os.path.join(REPO, "tests", "gpu",
                                                                               "test_dg_rowmatch_gpu.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    SR = sys.modules["test_static_rows_gpu"]
    monkeypatch.setattr(SR, "DEV", "cpu")
    monkeypatch.setattr(SR, "STATIC_KW", {"_allow_cpu": True})

    @contextlib.contextmanager
    def model_cpu(kind="mini4"):
        model, pool, _, _ = rows_dg_(kind)
        yield model, pool

    monkeypatch.setattr(G, "_model", model_cpu)
    monkeypatch.setattr(G, "_need_dg", lambda: None)
    return G


def test_gpu_rowmatch_gate_dry_run_on_cpu(rows_dg, monkeypatch, capsys):  # noqa: F811
    from runtime import dg_inplace as DI

    for v in (DI.ROWMATCH_ENV, DI.ROWS_GEMM_ENV, DI.ALIGN_ENV, DI.SF_STORE_ENV):
        monkeypatch.delenv(v, raising=False)
    G = _load_gpu_module(monkeypatch, rows_dg)
    G.test_mini4_static6_equals_static1_moe_isolated()
    G.test_build_check_mini4_real_deepgemm()
    out = capsys.readouterr().out
    assert out.count("rows bit-equal 24/24") == 4, out          # on and off, both parities
    assert "rowmatch on " in out and "align 16 x 24" in out and "-> on=True align 16 gemm_blocks 24" in out
    print(out)
    torch.set_default_device("cpu")
