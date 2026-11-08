"""Mac CPU tests of SMALL-WINS (docs/HANDOFF-SMALL-WINS.md): the row-conditional defaults and the gemv3 selector.

  * $DSV41_MIXES default `auto` = fork when the step builds for 1 row, off for R > 1 (and for an unknown row count);
  * $DSV41_QKV_TAIL default `auto` = fused at 1 row, split for R > 1; explicit env values win at every R, bad values
    raise (`auto` itself is a legal explicit value);
  * a StaticStep built with no env resolves by its own rows (1 and 6): MixesFork.impl / .source, the glue's qkv_tail
    mode / source (the fused tail is self-checked on the CPU mirror at build, so 1 row really runs fused);
  * the report line (`runtime.fused_glue_r2.resolved_line`) and the benches' `--fp8-impl gemv|gemv3` (gemv3 = gemv +
    $DSV41_GEMV_IMPL=gemv3, default stays gemv; first_light / headline summary lines show it).
Bit-identity of fork / fused vs off / split is tests/test_mixes_fork_cpu.py and tests/test_qkv_tail_cpu.py's.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from runtime import fused_glue_r2 as R2
from runtime import gemv_mxfp8 as GM
from test_static_rows_cpu import _cpu_default_device, emu, rows_dg, rows_glue  # noqa: F401


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (R2.MIXES_ENV, R2.QKV_TAIL_ENV, "DSV41_MIXES_PRIORITY", "DSV41_GEMV_IMPL"):
        monkeypatch.delenv(k, raising=False)
    saved = GM._IMPL[0]
    GM._IMPL[0] = None
    yield
    GM._IMPL[0] = saved


# ------------------------------------------------------------------------------------------------ the resolution
def test_defaults_are_auto():
    assert R2.MIXES_DEFAULT == "auto" and R2.QKV_TAIL_DEFAULT == "auto"
    assert "auto" in R2.MIXES_CHOICES and "auto" not in R2.MIXES_IMPLS
    assert "auto" in R2.QKV_TAIL_CHOICES and "auto" not in R2.QKV_TAILS


@pytest.mark.parametrize("rows,mixes,qkv", [(1, "fork", "fused"), (6, "off", "split"), (2, "off", "split"),
                                            (8, "off", "split"), (None, "off", "split")])
def test_auto_resolution_by_rows(rows, mixes, qkv):
    assert R2.mixes_choice(rows) == (mixes, "auto") and R2.mixes_impl(rows) == mixes
    assert R2.qkv_tail_choice(rows) == (qkv, "auto") and R2.qkv_tail_mode(rows) == qkv


@pytest.mark.parametrize("rows", [None, 1, 6])
def test_explicit_env_wins_at_every_row_count(monkeypatch, rows):
    for v in R2.MIXES_IMPLS:
        monkeypatch.setenv(R2.MIXES_ENV, v)
        assert R2.mixes_choice(rows) == (v, "env")
    for v in R2.QKV_TAILS:
        monkeypatch.setenv(R2.QKV_TAIL_ENV, v)
        assert R2.qkv_tail_choice(rows) == (v, "env")
    monkeypatch.setenv(R2.MIXES_ENV, "auto")           # `auto` spelled out is the default, not an override
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "auto")
    assert R2.mixes_choice(rows)[1] == "auto" and R2.qkv_tail_choice(rows)[1] == "auto"
    assert R2.mixes_impl(1) == "fork" and R2.mixes_impl(6) == "off"
    assert R2.qkv_tail_mode(1) == "fused" and R2.qkv_tail_mode(6) == "split"


def test_empty_env_is_the_default(monkeypatch):
    monkeypatch.setenv(R2.MIXES_ENV, "")
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "")
    assert R2.mixes_choice(1) == ("fork", "auto") and R2.qkv_tail_choice(1) == ("fused", "auto")


def test_bad_values_raise(monkeypatch):
    monkeypatch.setenv(R2.MIXES_ENV, "sideways")
    with pytest.raises(ValueError, match="DSV41_MIXES"):
        R2.mixes_impl(1)
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "both")
    with pytest.raises(ValueError, match="DSV41_QKV_TAIL"):
        R2.qkv_tail_mode(1)


def test_mix_fixed_unchanged_by_auto(monkeypatch):
    """auto never resolves to `fused`, so the oracle's order only follows $DSV41_FP32_IMPL unless fused is explicit."""
    monkeypatch.setenv(R2.FP32_IMPL_ENV, "cublas")
    assert not R2.mix_fixed() and not R2.mix_fixed("fork") and R2.mix_fixed("fused")
    monkeypatch.setenv(R2.MIXES_ENV, "fused")
    assert R2.mix_fixed()


def test_resolved_choices_and_line(monkeypatch):
    rc = R2.resolved_choices((1, 6))["rows"]
    assert rc[1] == {"mixes": "fork", "mixes_src": "auto", "qkv_tail": "fused", "qkv_tail_src": "auto"}
    assert rc[6] == {"mixes": "off", "mixes_src": "auto", "qkv_tail": "split", "qkv_tail_src": "auto"}
    line = R2.resolved_line((1, 6), "gemv")
    assert line == ("[small-wins] R=1 mixes=fork(auto) qkv_tail=fused(auto); "
                    "R=6 mixes=off(auto) qkv_tail=split(auto); gemv=gemv"), line
    monkeypatch.setenv(R2.MIXES_ENV, "off")
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "split")
    line = R2.resolved_line((1,))
    assert "R=1 mixes=off(env) qkv_tail=split(env)" in line and "gemv=" not in line


# ------------------------------------------------------------------------------------------------ real builds
def _build(model, pool, rows):
    from runtime.static_step import StaticStep

    return StaticStep(model, pool, max_pos=255, rows=rows, _allow_cpu=True, moe_impl="dg-inplace", fp8_impl="gemv",
                      glue_impl="fused")


def test_step_build_resolves_by_its_rows(rows_glue):
    model, pool, _, _ = rows_glue("mini4")
    st1 = _build(model, pool, 1)
    try:
        assert st1.mix_fork.impl == "fork" and st1.mix_fork.source == "auto" and st1.mix_fork.on
        assert st1.mix_fork.describe()["source"] == "auto"
        q = st1.glue.describe()["r2"]["qkv_tail"]
        assert q["mode"] == "fused" and q["source"] == "auto" and q["rows"] == 1, q
        assert st1.glue.describe()["calibration"]["qkv_tail"].startswith("fused: equal"), st1.glue.describe()
    finally:
        st1.detach()
    st6 = _build(model, pool, 6)
    try:
        assert st6.mix_fork.impl == "off" and st6.mix_fork.source == "auto" and not st6.mix_fork.on
        q = st6.glue.describe()["r2"]["qkv_tail"]
        assert q["mode"] == "split" and q["source"] == "auto" and q["rows"] == 6, q
    finally:
        st6.detach()


def test_step_build_explicit_env_beats_auto(rows_glue, monkeypatch):
    model, pool, _, _ = rows_glue("mini4")
    monkeypatch.setenv(R2.MIXES_ENV, "off")
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "split")
    st1 = _build(model, pool, 1)
    try:
        assert st1.mix_fork.impl == "off" and st1.mix_fork.source == "env"
        assert st1.glue.describe()["r2"]["qkv_tail"]["mode"] == "split"
    finally:
        st1.detach()
    monkeypatch.setenv(R2.MIXES_ENV, "fork")
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "fused")
    st6 = _build(model, pool, 6)
    try:
        assert st6.mix_fork.impl == "fork" and st6.mix_fork.source == "env"
        q = st6.glue.describe()["r2"]["qkv_tail"]
        assert q["mode"] == "fused" and q["source"] == "env", q
    finally:
        st6.detach()


# ------------------------------------------------------------------------------------------------ gemv|gemv3
def test_fp8_impl_arg_gemv3(monkeypatch):
    assert GM.FP8_IMPL_ARGS == ("ref", "gemv", "gemv3")
    assert GM.gemv_impl() == "gemv"                    # the default stays gemv
    assert GM.apply_fp8_impl_arg("ref") == "ref" and GM.apply_fp8_impl_arg("gemv") == "gemv"
    assert "DSV41_GEMV_IMPL" not in os.environ and GM.gemv_impl() == "gemv"
    assert GM.apply_fp8_impl_arg("gemv3") == "gemv"    # what Generator / StaticStep get
    assert os.environ["DSV41_GEMV_IMPL"] == "gemv3" and GM.gemv_impl() == "gemv3"
    # runtime code (apply_fp8_impl_arg) wrote os.environ directly. monkeypatch.delenv here would RECORD "gemv3" as the
    # value to restore and re-install it at teardown (leaking gemv3 into every later test); pop it for real.
    os.environ.pop("DSV41_GEMV_IMPL")
    with pytest.raises(ValueError):
        GM.apply_fp8_impl_arg("gemv4")


def _bench(name):
    import importlib

    bench = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    return importlib.import_module(name)


def test_bench_parsers_take_gemv3():
    for name in ("first_light", "headline"):
        mod = _bench(name)
        p = mod.build_parser()
        assert p.parse_args(["--model-dir", "x"]).fp8_impl == "ref"
        assert p.parse_args(["--model-dir", "x", "--fp8-impl", "gemv"]).fp8_impl == "gemv"
        assert p.parse_args(["--model-dir", "x", "--fp8-impl", "gemv3"]).fp8_impl == "gemv3"
        with pytest.raises(SystemExit):
            p.parse_args(["--model-dir", "x", "--fp8-impl", "gemv4"])


def test_bench_report_lines_show_the_resolved_choices():
    fl, hl = _bench("first_light"), _bench("headline")
    sw = dict(R2.resolved_choices((1, 6)), fp8_impl_arg="gemv3", gemv_kernel="gemv3")
    r = {"date": "d", "env": {}, "errors": [], "generator": {"mode": "graph", "fp8_impl": "gemv"}, "small_wins": sw}
    lines = fl.summary_lines(r, "j.json")
    assert len(lines) == 12 and "fp8 gemv3" in lines[0], lines[0]
    assert "small-wins R1 mixes=fork qkv=fused, R6 mixes=off qkv=split" in lines[0], lines[0]
    r2 = dict(r, small_wins=dict(sw, gemv_kernel="gemv"))
    assert "fp8 gemv " in fl.summary_lines(r2, "j.json")[0] + " "
    plain = {"date": "d", "env": {}, "errors": [], "generator": {"mode": "graph", "fp8_impl": "gemv"}}
    assert len(fl.summary_lines(plain, "j.json")) == 12 and "small-wins" not in fl.summary_lines(plain, "j.json")[0]
    hs = hl.summary_lines({"small_wins": sw, "headline": {}, "prompts": {}, "streams": {}, "impls": {"fp8": "gemv3"}})
    assert hs[1].startswith("  small-wins: R=1 mixes=fork(auto) qkv_tail=fused(auto); R=6 mixes=off(auto) "
                            "qkv_tail=split(auto); fp8 gemv3 (kernel gemv3)"), hs[1]
    assert not any("small-wins" in l for l in hl.summary_lines({"headline": {}, "prompts": {}, "streams": {}}))
    assert sys.modules.get("vllm") is None
