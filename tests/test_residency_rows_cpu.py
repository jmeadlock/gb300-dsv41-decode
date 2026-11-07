"""Mac CPU tests of A2-2 at R rows per step (runtime/residency.py on runtime/moe_rows.py's R-row MoEs), on the row-wise
CPU emulations of tests/test_static_rows_cpu.py: the 6-row teacher-forced step with residency on (forced random
swaps every step + policy swaps from synthetic counters) == off, torch.equal on every row's fp32 logits, ref and
dg-inplace (n_stage = 6 * topk spare staging slots before the residency slots); the hit counter counts each used
unique slot once (uvalid_step); the audit passes. `.venv/bin/python -m pytest -q tests/`.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from runtime.residency import ResidencyConfig, ResidencyManager, golden_from_index, n_stage_for
from test_residency_cpu import _hot_cold_bias, _install
from test_static_rows_cpu import _assert_rows_equal, _prefill, _tf_tokens, emu  # noqa: F401


@pytest.fixture(autouse=True)
def _cpu_default_device():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


def _run6(model, st, ids, toks, n, rm=None, after=None):
    _prefill(model, ids)
    out = []
    with torch.inference_mode():
        for s in range(n):
            lg = st.step(toks[s * st.rows:(s + 1) * st.rows], len(ids) + s * st.rows, graph=False)
            out += [lg[j].clone() for j in range(st.rows)]
            if after is not None:
                after(s)
            if rm is not None:
                rm.after_step()
    return out


@pytest.mark.parametrize("impl", ["ref", "dg-inplace"])
def test_residency_rows6_on_equals_off(emu, monkeypatch, impl):
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")
    if impl == "dg-inplace":
        from test_dg_inplace_cpu import _fake_dg_contig
        from test_dg_moe_cpu import _faithful_act_quant
        from test_static_rows_cpu import _masked_rowwise

        fake = _fake_dg_contig()
        fake.m_grouped_fp8_fp4_gemm_nt_masked = _masked_rowwise(fake)
        monkeypatch.setitem(sys.modules, "deep_gemm", fake)
        monkeypatch.setattr(sys.modules["model"], "act_quant", _faithful_act_quant)
    _install(monkeypatch, pool)
    n = 8
    ids, toks = _tf_tokens(17, 6 * n, 29)
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl=impl, _allow_cpu=True)
    off = _run6(model, st, ids, toks, n)
    st.detach()
    k = model.layers[0].ffn.n_activated_experts
    cfg = ResidencyConfig(drain_n=2, min_abs=1.0, hyst=1.0, halflife=20.0, n_spare=2, n_demote=4, reserve_gib=0.0,
                          force_random=2)
    rm = ResidencyManager(pool, model, cfg, n_stage=n_stage_for(impl, 6, k))
    assert rm.base == [len(pool.hot[l]) + (6 * k if impl == "dg-inplace" else 0) for l in range(pool.n_layers)]
    st = StaticStep(model, pool, max_pos=255, rows=6, moe_impl=impl, _allow_cpu=True)
    rm.attach(st)
    rng = np.random.default_rng(2)
    hits = []

    def after(s):
        u = (st.ids_step.unsqueeze(-1) == torch.arange(rm.E)) & (st.uvalid_step.unsqueeze(-1) > 0)
        hits.append(u.sum(1).to(torch.int32))
        rm.inject_counts(_hot_cold_bias(pool, rng, 30))

    c0 = rm.cnt.clone()
    on = _run6(model, st, ids, toks, n, rm=rm, after=after)
    _assert_rows_equal(on, off, f"6-row {impl}: residency on vs off")
    assert rm.stats["published"] > 0 and rm.state.n_swaps > 0 and rm.stats["vetoed"] == 0
    rm.flush()
    a = rm.audit(golden=golden_from_index(model.runtime_load.index))
    assert a["ok"], a["errors"][:5]
    rm.close()
    st.detach()
    assert int(torch.stack(hits).sum()) > 0


def test_counter_counts_unique_used_slots_once(emu, monkeypatch):
    """At rows > 1 ids_step holds the U unique experts (slots >= U repeat the largest id with uvalid 0): the
    counter adds one per used unique expert per layer per step."""
    from runtime.static_step import StaticStep

    model, pool, _ = emu("mini4")
    _install(monkeypatch, pool)
    rm = ResidencyManager(pool, model, ResidencyConfig(frozen=True, n_spare=1, n_demote=1, reserve_gib=0.0),
                          n_stage=0)
    st = StaticStep(model, pool, max_pos=255, rows=6, _allow_cpu=True)
    rm.attach(st)
    ids, toks = _tf_tokens(17, 6, 31)
    _prefill(model, ids)
    with torch.inference_mode():
        st.step(toks[:6], len(ids), graph=False)
        rm.after_step()
    want = torch.zeros(rm.L, rm.E, dtype=torch.int32)
    for l in range(rm.L):
        for e, v in zip(st.ids_step[l].tolist(), st.uvalid_step[l].tolist()):
            if v:
                want[l, e] += 1
    assert torch.equal(rm.cnt, want) and int(want.max()) == 1 and int(want.sum()) == int(st.uvalid_step.sum())
    rm.close()
    st.detach()
