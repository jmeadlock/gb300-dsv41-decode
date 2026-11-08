"""Mac CPU test of M2-L5: runtime/pdl.py (the $DSV41_PDL switch, the prefetch geometry, the edge census) and the PDL /
pre-wait prefetch additions to the owned Triton kernels. `.venv/bin/python -m pytest -q tests/test_pdl_impl_cpu.py`

What the Mac can check (Triton compilation, griddepcontrol itself, the L2 prefetch and every timing are box-only:
tests/gpu/test_pdl_gpu.py on the mini, bench/pdl_probe.py for the per-edge us, the orchestrator's headline gate):
  1. the switch: off (default) -> no launch keyword at all (today's launch); on / late / prefetch -> the PDL constexpr,
     launch_pdl=True and -- prefetching kernels only -- PDL_PF + the geometry; the exclude / only / no-prefetch lists by
     kernel name and family; bad values raise;
  2. the source structure of EVERY kernel in pdl.KERNELS (AST): PDL (and PDL_PF) constexprs defaulting to 0; exactly
     one gdc_wait and two gdc_launch_dependents (entry / late), each the body of a top-level `if PDL ...` -- never in
     a loop; nothing above the wait but index arithmetic on program ids / scalars and the prefetch helper; the late
     trigger the last statement (the MXFP8 GEMV: right after its K loop's `mask =`); every launch site of the kernel
     passes `**_pdl.kw(<its name>, ...)`; the prefetch helpers are verbatim copies across the three modules;
  3. dynamic, through tests/triton_shim.py (which asserts per program, for every launch_pdl launch: one wait, one
     trigger after it, NO load / store / atomic before the wait, every prefetch before it and in bounds): the GEMV,
     the one-launch pair, the fixed-order fp32 dots (head / mix / comp), the gate dot, norm_row, hc_norm (one and
     split forms) and engram_gate produce the SAME BITS with PDL off / on / prefetch (entry and late triggers), and
     each program's prefetch touches exactly the 128 B lines of its own weight slice (the weight / weight-scale /
     norm-weight arguments only -- never an activation);
  4. the edge census classification and the MODEL arithmetic on synthetic edges.
The rest of the owned kernels are covered dynamically by running the existing shim suites under $DSV41_PDL (receipts in
scratch/; opt-in here with DSV41_PDL_SLOW=1: test_existing_shim_suites_pass_under_pdl).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import triton_shim
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from runtime import gemv_mxfp8 as G
from runtime import pdl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
ENVS = (pdl.ENV, pdl.TRIGGER_ENV, pdl.EXCLUDE_ENV, pdl.ONLY_ENV, pdl.NOPF_ENV)
MODES = [("off", None), ("on", None), ("on", "late"), ("prefetch", None), ("prefetch", "late")]


def _bits(t):
    t = t.contiguous()
    n = t.element_size()
    return t.view(torch.uint8) if n == 1 else t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[n])


def _eq(a, b):
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


@pytest.fixture
def env(monkeypatch):
    """env(mode, trigger=None, exclude=None, only=None, nopf=None): set the switch for this test."""
    def set_(mode="off", trigger=None, exclude=None, only=None, nopf=None):
        for k, v in zip(ENVS, (mode, trigger, exclude, only, nopf)):
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
    for k in ENVS:
        monkeypatch.delenv(k, raising=False)
    return set_


@pytest.fixture(autouse=True)
def _keep_prefetch():
    triton_shim.KEEP_PREFETCH = True
    yield
    triton_shim.KEEP_PREFETCH = False


# ================================================================================ 1. the switch
def test_default_is_off_and_adds_no_launch_keyword(env):
    env()
    assert pdl.mode() == "off"
    for name in pdl.KERNELS:
        assert pdl.kw(name, pf=lambda: 1 / 0) == {}            # pf never evaluated
    assert pdl.config().describe()["enabled"] == []


@pytest.mark.parametrize("trigger,want", [(None, pdl.PDL_ENTRY), ("entry", pdl.PDL_ENTRY), ("late", pdl.PDL_LATE)])
def test_on_gives_every_owned_kernel_pdl_and_no_prefetch(env, trigger, want):
    env("on", trigger)
    for name in pdl.KERNELS:
        assert pdl.kw(name, pf=lambda: 1 / 0) == {"PDL": want, "launch_pdl": True}, name
    assert pdl.kw("t9b_pair_scale_probe") == {} and pdl.kw("m2_softplus_probe") == {}   # not owned step kernels


def test_prefetch_only_on_prefetching_kernels(env):
    env("prefetch")
    for name, (fam, wts, _) in pdl.KERNELS.items():
        d = pdl.kw(name, pf=lambda: {"PF_NL": 8})
        if wts:
            assert d == {"PDL": 1, "launch_pdl": True, "PDL_PF": 1, "PF_NL": 8}, name
        else:
            assert d == {"PDL": 1, "launch_pdl": True}, name
    assert pdl.kw("t9b_mxfp8_gemv") == {"PDL": 1, "launch_pdl": True}    # no pf callable: no prefetch


def test_exclude_only_nopf_by_name_and_family(env):
    env("prefetch", exclude="glue,t8_step_marker")
    assert pdl.kw("t9b2r2_norm_row", pf=dict) == {} and pdl.kw("t8_step_marker") == {}
    assert pdl.kw("t9b_mxfp8_gemv", pf=dict)["PDL_PF"] == 1
    env("on", only="gemv")
    assert {n for n in pdl.KERNELS if pdl.kw(n)} == {n for n, v in pdl.KERNELS.items() if v[0] == "gemv"}
    env("prefetch", only="gemv,t9b2r2_norm_row", nopf="t9b4_mxfp8_gemv2")
    assert "PDL_PF" not in pdl.kw("t9b4_mxfp8_gemv2", pf=dict) and pdl.kw("t9b4_mxfp8_gemv2", pf=dict)
    assert pdl.kw("t9b2r2_norm_row", pf=dict)["PDL_PF"] == 1 and pdl.kw("t9b2r2_hc_norm", pf=dict) == {}


@pytest.mark.parametrize("bad", [dict(mode="yes"), dict(mode="on", trigger="early"), dict(mode="on", exclude="nope"),
                                 dict(mode="on", only="t9b_mxfp8"), dict(mode="prefetch", nopf="gemm")])
def test_bad_values_raise(env, bad):
    env(**bad)
    with pytest.raises(ValueError):
        pdl.config()


def test_registry_is_complete_and_consistent():
    names = [n for ks in pdl.MODULES.values() for n in ks]
    assert sorted(names) == sorted(pdl.KERNELS) and len(names) == len(set(names))
    for name, (fam, wts, late) in pdl.KERNELS.items():
        assert fam in pdl.FAMILIES and isinstance(wts, tuple) and late


def test_prefetch_geometry():
    assert pdl.elems_per_line(1) == 128 and pdl.elems_per_line(2) == 64 and pdl.elems_per_line(4) == 32
    for n, epl in ((1, 128), (127, 128), (128, 128), (129, 128), (640, 128), (5120, 64), (28672, 32), (7, 32)):
        nl = pdl.lines(n, epl)
        assert nl & (nl - 1) == 0 and nl >= -(-n // epl) + 1
        for rows in (1, 2, 16, 32, 64):
            ch = pdl.chunk(nl, rows)
            assert nl % ch == 0 and ch * pdl._np2(rows) <= max(pdl.PF_BLOCK, pdl._np2(rows))
    # the pure mirror covers every line of an unaligned run and nothing outside it
    for base in (0, 5, 127, 128, 200):
        for clen in (1, 31, 128, 129, 300, 1000):
            nl = pdl.lines(clen, 128)
            pts = pdl.prefetch_points(0, 1, 1, 0, base, clen, 128, nl, pdl.chunk(nl, 1))
            assert {p // 128 for p in pts} == set(range(base // 128, (base + clen - 1) // 128 + 1))


# ================================================================================ 2. source structure (AST)
def _mod_tree(rel):
    path = os.path.join(REPO, rel)
    src = open(path).read()
    return src, ast.parse(src)


def _defs(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _gdc_calls(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
                "gdc_wait", "gdc_launch_dependents"):
            out.append(n)
    return out


def _pdl_if(stmt, op, val):
    """stmt is `if PDL <op> <val>:` (op 'gt' / 'eq')."""
    if not isinstance(stmt, ast.If) or not isinstance(stmt.test, ast.Compare):
        return False
    t = stmt.test
    return (isinstance(t.left, ast.Name) and t.left.id == "PDL" and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Gt if op == "gt" else ast.Eq)
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == val)


def _calls_in(stmt):
    return [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]


PRE_WAIT_OK = {"program_id", "minimum", "maximum", "arange"}


@pytest.mark.parametrize("rel", sorted(pdl.MODULES))
def test_every_owned_kernel_has_the_pdl_structure(rel):
    src, tree = _mod_tree(rel)
    defs = _defs(tree)
    for name in pdl.MODULES[rel]:
        fn = defs[name]
        args = fn.args.args
        defaults = dict(zip([a.arg for a in args[len(args) - len(fn.args.defaults):]], fn.args.defaults))
        assert "PDL" in defaults and defaults["PDL"].value == 0, name
        if pdl.prefetching(name):
            assert "PDL_PF" in defaults and defaults["PDL_PF"].value == 0, name
        calls = _gdc_calls(fn)
        assert sorted(c.func.attr for c in calls) == ["gdc_launch_dependents", "gdc_launch_dependents", "gdc_wait"], name
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        iw = next(i for i, s in enumerate(body) if _pdl_if(s, "gt", 0))
        assert _pdl_if(body[iw + 1], "eq", 1), name                        # entry trigger right after the wait
        assert [c.func.attr for c in _gdc_calls(body[iw])] == ["gdc_wait"]
        assert len(body[iw].body) == 1 and len(body[iw + 1].body) == 1
        il = next(i for i, s in enumerate(body) if _pdl_if(s, "eq", 2))
        assert [c.func.attr for c in _gdc_calls(body[il])] == ["gdc_launch_dependents"] and len(body[il].body) == 1
        # nothing above the wait but index arithmetic and the prefetch helper
        for s in body[:iw]:
            for c in _calls_in(s):
                f = c.func
                ok = (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tl"
                      and f.attr in PRE_WAIT_OK) or (isinstance(f, ast.Name) and f.id == "_pdl_prefetch") or \
                     (isinstance(f, ast.Attribute) and f.attr == "to")
                assert ok, (name, ast.unparse(s)[:120])
            assert not isinstance(s, (ast.For, ast.While)), name
        # the late trigger: the last statement, except the MXFP8 GEMV (after the K loop, before the stores)
        if name == "t9b_mxfp8_gemv":
            prev = body[il - 1]
            assert isinstance(prev, ast.Assign) and prev.targets[0].id == "mask"
            assert any(isinstance(s, ast.For) for s in body[iw:il]), "the K loop precedes the late trigger"
        else:
            assert il == len(body) - 1, (name, il, len(body))
        # gdc never inside a loop (Triton 3.7.1 pipeliner #11905): the three calls are in top-level ifs
        for loop in (n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))):
            assert not _gdc_calls(loop), name
        # prefetch only inside `if PDL_PF:` and only before the wait
        for i, s in enumerate(body):
            if any(isinstance(c.func, ast.Name) and c.func.id == "_pdl_prefetch" for c in _calls_in(s)):
                assert i < iw and isinstance(s, ast.If) and isinstance(s.test, ast.Name) and s.test.id == "PDL_PF"


def _launch_calls(tree):
    """(kernel expression text, Call) of every `<kernel>[grid](...)` launch in host code."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript):
            out.append((ast.unparse(n.func.value), n))
    return out


@pytest.mark.parametrize("rel", sorted(pdl.MODULES))
def test_every_launch_site_passes_the_pdl_keywords(rel):
    src, tree = _mod_tree(rel)
    owned = set(pdl.MODULES[rel])
    seen = set()
    for kexpr, call in _launch_calls(tree):
        if kexpr in owned or kexpr.startswith("T9B5_KERNELS["):
            kws = [k for k in call.keywords if k.arg is None and isinstance(k.value, ast.Call)
                   and ast.unparse(k.value.func) == "_pdl.kw"]
            assert len(kws) == 1, (rel, kexpr, ast.unparse(call)[:200])
            arg = kws[0].value.args[0]
            if kexpr.startswith("T9B5_KERNELS["):
                assert ast.unparse(arg) == "T9B5_PDL_NAMES[site]"
                seen |= {"t9b5_head_fp32", "t9b5_mix_fp32", "t9b8_comp_fp32"}
            else:
                assert isinstance(arg, ast.Constant) and arg.value == kexpr, (kexpr, ast.unparse(arg))
                seen.add(kexpr)
            if pdl.prefetching(kexpr if not kexpr.startswith("T9B5") else "t9b5_head_fp32"):
                assert any(k.arg == "pf" for k in kws[0].value.keywords), kexpr
    assert seen == owned, (rel, owned - seen)


def test_t9b5_names_match_the_kernels():
    with triton_shim.installed():
        pass
    src, tree = _mod_tree("runtime/static_kernels.py")
    assert 'T9B5_PDL_NAMES = {"head": "t9b5_head_fp32", "mix": "t9b5_mix_fp32", "comp": "t9b8_comp_fp32"}' in src


def test_prefetch_helpers_are_verbatim_copies():
    texts = {}
    for rel in ("runtime/gemv_mxfp8.py", "runtime/static_kernels.py", "runtime/fused_glue_r2.py"):
        src, tree = _mod_tree(rel)
        d = _defs(tree)
        lines = src.splitlines()
        texts[rel] = tuple(textwrap.dedent("\n".join(lines[d[n].lineno - 1:d[n].end_lineno]))
                           for n in ("_pdl_pf_l2", "_pdl_prefetch"))
    assert len(set(texts.values())) == 1, texts.keys()
    assert "prefetch.global.L2" in texts["runtime/gemv_mxfp8.py"][0]


# ================================================================================ 3. dynamic (shim)
@pytest.fixture(scope="module")
def KG():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "gemv_pdl_impl")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KS():
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"), "sk_pdl_impl")


@pytest.fixture(scope="module")
def KR():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue_r2.py"), "fg2_pdl_impl")
    mod.KERNELS_ON_CPU = True
    return mod


def _exact_fp8(*shape, g):
    e = torch.randint(6, 9, shape, generator=g, dtype=torch.int32)
    m = torch.randint(0, 8, shape, generator=g, dtype=torch.int32)
    sg = torch.randint(0, 2, shape, generator=g, dtype=torch.int32)
    return ((sg << 7) | (e << 3) | m).to(torch.uint8).view(FP8)


def _operands(M, N, K_, seed):
    g = torch.Generator().manual_seed(seed)
    xq, w = _exact_fp8(M, K_, g=g), _exact_fp8(N, K_, g=g)
    xs = torch.randint(110, 131, (M, K_ // 32), generator=g, dtype=torch.int32).to(torch.uint8).view(E8)
    s = torch.randint(118, 132, ((N + 31) // 32, K_ // 32), generator=g, dtype=torch.int32).to(torch.uint8).view(E8)
    return xq, xs, w, s


def _last(jf, kernel=None):
    recs = [r for r in jf.launches if kernel is None or r["kernel"] == kernel]
    return recs[-1]


def _pf_lines(rec, name, per_program=False):
    """Prefetched 128 B line indices of argument `name` (absolute addresses // 128), union or per program."""
    ptr, es = rec["tensors"][name]
    d = {}
    for pid, names, offs in rec["prefetch_offs"]:
        if name in names:
            d.setdefault(pid, set()).update(((ptr + offs.astype(np.int64) * es) // 128).tolist())
    if per_program:
        return d
    return set().union(*d.values()) if d else set()


def _run_lines(t: torch.Tensor, rows, c0_bytes: int, c1_bytes: int) -> set:
    """Lines of bytes [c0, c1) of the given rows of a row-major tensor."""
    rs = t.stride(0) * t.element_size()
    out = set()
    for r in rows:
        a = t.data_ptr() + r * rs
        out |= set(range((a + c0_bytes) // 128, (a + c1_bytes - 1) // 128 + 1))
    return out


def _check_record(rec, mode, trig, weights):
    assert rec["launch_pdl"] == (mode != "off")
    if mode == "off":
        assert "PDL" not in rec["constexprs"] and not rec["prefetch_offs"]
        return
    assert rec["constexprs"]["PDL"] == (pdl.PDL_LATE if trig == "late" else pdl.PDL_ENTRY)
    if mode == "prefetch" and weights:
        assert rec["constexprs"].get("PDL_PF") == 1 and rec["prefetch_offs"], rec["kernel"]
        # every prefetched buffer IS a weight argument (a buffer passed under several names -- e.g. norm_row's unused
        # freqs dummy is nw itself -- carries all of them; one must be a weight)
        for pid, names, _ in rec["prefetch_offs"]:
            assert names & set(weights), (rec["kernel"], pid, names)
    else:
        assert not rec["prefetch_offs"] and "PDL_PF" not in rec["constexprs"]


GEMV_CASES = [  # (M, N, K, GemvConfig(block_n, ku, warps, stages, split_k, mma))
    (1, 64, 256, G.GemvConfig(32, 4, 4, 2, 1, 0)),
    (1, 100, 1280, G.GemvConfig(32, 4, 4, 2, 7, 0)),
    (3, 96, 640, G.GemvConfig(64, 2, 4, 2, 4, 0)),
    (6, 40, 384, G.GemvConfig(32, 8, 8, 3, 5, 0)),
    (8, 40, 1024, G.GemvConfig(128, 2, 4, 2, 8, 1)),
]


@pytest.mark.parametrize("M,N,Kd,cfg", GEMV_CASES, ids=[f"M{c[0]}-N{c[1]}-K{c[2]}-s{c[3].split_k}" for c in GEMV_CASES])
def test_gemv_bits_and_prefetch_slice(KG, env, M, N, Kd, cfg):
    xq, xs, w, s = _operands(M, N, Kd, seed=M * 7 + N + Kd)
    plan = KG.launch_plan(cfg, M, N, Kd)
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        ws = torch.full((plan["NSPLIT"], M, N), float("nan"), dtype=torch.float32)
        out = torch.full((M, N), float("nan"), dtype=BF)
        KG._launch(xq, xs, w, s, out, cfg, ws if plan["SPLIT"] else out)
        rec = _last(KG.t9b_mxfp8_gemv)
        _check_record(rec, mode, trig, ("w", "s"))
        if plan["SPLIT"]:
            rr = _last(KG.t9b_splitk_reduce)
            assert rr["launch_pdl"] == (mode != "off") and not rr["prefetch_offs"]
        if mode == "prefetch":
            per = plan["PER"]
            got_w, got_s = _pf_lines(rec, "w", True), _pf_lines(rec, "s", True)
            for pn in range(plan["grid"][0]):
                rows = [r for r in range(pn * cfg.block_n, (pn + 1) * cfg.block_n) if r < N]
                srows = sorted({r // 32 for r in rows})
                for ks in range(plan["grid"][1]):
                    lo, hi = ks * per, min((ks + 1) * per, Kd // 32)
                    assert got_w[(pn, ks, 0)] == _run_lines(w, rows, lo * 32, hi * 32), (pn, ks)
                    assert got_s[(pn, ks, 0)] == _run_lines(s.view(torch.uint8), srows, lo, hi), (pn, ks)
        res[(mode, trig)] = (out.clone(), ws.clone())
    for k, (o, wsk) in res.items():
        assert _eq(o, res[("off", None)][0]), k
        if plan["SPLIT"]:
            assert _eq(wsk, res[("off", None)][1]), k


def test_gemv_pair_bits_and_prefetch_slices(KG, env):
    M, NA, NB, Kd = 6, 96, 40, 640
    xq, xs, wa, sa = _operands(M, NA, Kd, seed=1)
    _, _, wb, sb = _operands(M, NB, Kd, seed=2)
    cfgs = (G.GemvConfig(32, 4, 4, 2, 4, 0), G.GemvConfig(32, 4, 4, 2, 3, 0))
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        ra, rb = KG.gemv_pair(xq, xs, wa, sa, wb, sb, tags=("pdl_a", "pdl_b"), configs=cfgs)
        rec = _last(KG.t9b4_mxfp8_gemv2)
        _check_record(rec, mode, trig, ("wa", "sa", "wb", "sb"))
        res[(mode, trig)] = tuple((r.ws if hasattr(r, "ws") else r).clone() for r in (ra, rb))
        if mode == "prefetch":
            kb = Kd // 32
            assert _pf_lines(rec, "wa") == _run_lines(wa, range(NA), 0, kb * 32)
            assert _pf_lines(rec, "wb") == _run_lines(wb, range(NB), 0, kb * 32)
            assert _pf_lines(rec, "sa") == _run_lines(sa.view(torch.uint8), range((NA + 31) // 32), 0, kb)
            assert _pf_lines(rec, "sb") == _run_lines(sb.view(torch.uint8), range((NB + 31) // 32), 0, kb)
    for k, v in res.items():
        assert all(_eq(a, b) for a, b in zip(v, res[("off", None)])), k


@pytest.mark.parametrize("site,wdt", [("head", torch.float32), ("head", BF), ("mix", torch.float32),
                                      ("comp", torch.float32)])
@pytest.mark.parametrize("R", [1, 3])
def test_fixed_dots_bits_and_prefetch_rows(KS, env, site, wdt, R):
    g = torch.Generator().manual_seed(R + len(site))
    K_, N = 256, 40
    x = torch.randn(R, K_, generator=g)
    w = torch.randn(N, K_, generator=g).to(wdt)
    sc = torch.rand(R, generator=g) + 0.5 if site == "mix" else None
    name = {"head": "t9b5_head_fp32", "mix": "t9b5_mix_fp32", "comp": "t9b8_comp_fp32"}[site]
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        out = torch.full((R, N), float("nan"))
        KS.fixed_dot(x, w, out, kb=32, site=site, scale=sc)
        rec = _last(KS.T9B5_KERNELS[site])
        assert rec["kernel"] == name
        _check_record(rec, mode, trig, ("w",))
        if mode == "prefetch":
            bn = rec["constexprs"].get("BN") or KS.t9b5_config(site, R, w.dtype)[1]
            per = _pf_lines(rec, "w", True)
            for (pn, pr, _), lines in per.items():
                rows = [r for r in range(pn * bn, (pn + 1) * bn) if r < N]
                assert lines == _run_lines(w, rows, 0, K_ * w.element_size()), (pn, pr)
            assert len(per) == rec["programs"]
        res[(mode, trig)] = out.clone()
    assert all(_eq(v, res[("off", None)]) for v in res.values())


def test_gate_dot_bits_and_prefetch_rows(KS, env):
    g = torch.Generator().manual_seed(9)
    R, K_, E = 3, 256, 20
    x, w = torch.randn(R, K_, generator=g), torch.randn(E, K_, generator=g).to(BF)
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        out = torch.full((R, E), float("nan"))
        KS.gate_dot(x, w, out, kb=32)
        rec = _last(KS.m2_gate_fp32)
        _check_record(rec, mode, trig, ("w",))
        if mode == "prefetch":
            assert _pf_lines(rec, "w") == _run_lines(w, range(E), 0, K_ * 2)
        res[(mode, trig)] = out.clone()
    assert all(_eq(v, res[("off", None)]) for v in res.values())


@pytest.fixture(scope="module")
def tab():
    from test_fused_glue_r2_cpu import _R   # the reference act_quant stand-in the R2 suite uses
    return FG.scale_code_table(_R, "cpu")


def _od(order, fin="mul"):
    return {"order": order, "fin": fin, "exact": True, "name": R2.order_name(order, fin)}


@pytest.mark.parametrize("R_", [1, 3])
def test_norm_row_bits_and_prefetch_norm_weight(KR, env, tab, R_):
    N = 320
    g = torch.Generator().manual_seed(R_)
    x = FG._rand_act(g, R_, N, -4, 6)
    nw = (torch.randn(N, generator=g) * 0.3 + 1.0).to(BF)
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        out = torch.full((R_, N), 3.0, dtype=BF)
        q, s = torch.empty(R_, N, dtype=FP8), torch.empty(R_, N // 32, dtype=E8)
        KR.launch_norm_row(x if R_ > 1 else x.view(-1), nw, out if R_ > 1 else out.view(-1), eps=1e-6,
                           ordd=_od(R2.predicted_order(N), "mul"), code_tab=tab, q=q if R_ > 1 else q.view(-1),
                           s=s if R_ > 1 else s.view(-1))
        rec = _last(KR.t9b2r2_norm_row)
        _check_record(rec, mode, trig, ("nw",))
        if mode == "prefetch":
            per = _pf_lines(rec, "nw", True)
            assert len(per) == R_ and all(v == _run_lines(nw.view(1, N), [0], 0, 2 * N) for v in per.values())
        res[(mode, trig)] = (out.clone(), q.clone(), s.clone())
    for v in res.values():
        assert all(_eq(a, b) for a, b in zip(v, res[("off", None)]))


@pytest.mark.parametrize("split", [False, True])
def test_hc_norm_bits_and_prefetch_norm_weight(KR, env, tab, split):
    HC, D = 4, 512
    g = torch.Generator().manual_seed(5 + split)
    x = FG._rand_act(g, HC, D, -6, 7)
    pre = torch.rand(HC, generator=g) + 1e-3
    nw = (torch.randn(D, generator=g) * 0.5 + 1.0).to(BF)
    name = "t9b4_hc_norm" if split else "t9b2r2_hc_norm"
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        o = dict(y=torch.empty(D, dtype=BF), yf=torch.empty(D), xf=torch.empty(HC * D), mrs=torch.empty(1),
                 q=torch.empty(D, dtype=FP8), s=torch.empty(D // 32, dtype=E8))
        KR.launch_hc_norm(x, pre, nw, o["y"], eps=1e-6, norm_ord=_od(("vec", 64, 4, True)), hc_mode=1, code_tab=tab,
                          yf=o["yf"], xf=o["xf"], mrs=o["mrs"], mix_ord=_od(("vec", 128, 4, True)), mix_eps=1e-6,
                          q=o["q"], s=o["s"], split=split)
        rec = _last(getattr(KR, name))
        _check_record(rec, mode, trig, ("nw",))
        if mode == "prefetch":
            per = _pf_lines(rec, "nw", True)
            if split:   # program p < NAP: its own 32-blocks [p*BB, (p+1)*BB); the mix program: nothing
                bb = KR.HC_SPLIT_BB
                nap = -(-(D // 32) // bb)
                assert sorted(per) == [(p, 0, 0) for p in range(nap)]
                for (p, _, _), lines in per.items():
                    assert lines == _run_lines(nw.view(1, D), [0], 2 * p * bb * 32, 2 * min(D, (p + 1) * bb * 32))
            else:
                assert per[(0, 0, 0)] == _run_lines(nw.view(1, D), [0], 0, 2 * D)
        res[(mode, trig)] = {k: v.clone() for k, v in o.items()}
    for v in res.values():
        assert all(_eq(v[k], res[("off", None)][k]) for k in v)


def test_engram_gate_bits_and_prefetch_rows(KR, env):
    HC, D = 4, 160
    g = torch.Generator().manual_seed(11)
    x = FG._rand_act(g, HC, D, -6, 7)
    kv = FG._rand_act(g, 1, (HC + 1) * D, -4, 4).view(-1)
    wp = torch.randn(HC, D, generator=g).contiguous()
    c, cmin = R2.f32(D ** -0.5), R2.f32(1e-6)
    res = {}
    for mode, trig in MODES:
        env(mode, trig)
        z = torch.full((HC,), float("nan"))
        KR.launch_engram_gate(x, kv, wp, z, eps=1e-6, ordd=_od(("vec", 16, 4, True)), sordd=_od(("vec", 32, 2, True)),
                              c=c, cmin=cmin)
        rec = _last(KR.t9b2r2_engram_gate)
        _check_record(rec, mode, trig, ("wp",))
        if mode == "prefetch":
            per = _pf_lines(rec, "wp", True)
            for (i, _, _), lines in per.items():
                assert lines == _run_lines(wp, [i], 0, 4 * D), i
            assert len(per) == HC
        res[(mode, trig)] = z.clone()
    assert all(_eq(v, res[("off", None)]) for v in res.values())


def test_step_marker_under_pdl(KS, env):
    """t8_step_marker through the shim in every mode (the gather kernels t8_gather_rows(_u8) / t9b2_gather_tables(_u8)
    cast int64 row addresses to pointers, which the shim cannot run: their Mac coverage is the AST structure test,
    their dynamic check tests/gpu/test_pdl_gpu.py)."""
    for mode, trig in MODES:
        env(mode, trig)
        ctr = torch.tensor([41], dtype=torch.int64)
        KS.step_marker(ctr)
        rec = _last(KS.t8_step_marker)
        _check_record(rec, mode, trig, ())
        assert int(ctr[0]) == 42


def test_shim_rejects_a_load_before_the_wait():
    """The shim's own sensitivity: a kernel that loads before gdc_wait under launch_pdl fails the launch."""
    with triton_shim.installed():
        import triton
        import triton.language as tl

        @triton.jit
        def bad(x, y, PDL: tl.constexpr = 0):
            v = tl.load(x + tl.arange(0, 4))
            if PDL > 0:
                tl.extra.cuda.gdc_wait()
            if PDL == 1:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(y + tl.arange(0, 4), v)

        @triton.jit
        def no_trigger(x, y, PDL: tl.constexpr = 0):
            if PDL > 0:
                tl.extra.cuda.gdc_wait()
            tl.store(y + tl.arange(0, 4), tl.load(x + tl.arange(0, 4)))

        a, b = torch.arange(4, dtype=torch.float32), torch.zeros(4)
        bad[(1,)](a, b)                                            # no PDL: fine
        with pytest.raises(AssertionError, match="before gdc_wait"):
            bad[(1,)](a, b, PDL=1, launch_pdl=True)
        with pytest.raises(AssertionError, match="gdc_launch_dependents"):
            no_trigger[(1,)](a, b, PDL=1, launch_pdl=True)
        with pytest.raises(AssertionError, match="without launch_pdl"):
            no_trigger[(1,)](a, b, PDL=1)


# ================================================================================ 4. edge census / model
def test_classify_edges_and_model(env):
    env("prefetch")
    edges = [("K:t9b2r2_hc_norm", "K:t9b_mxfp8_gemv"),       # owned -> owned, GEMV prefetch
             ("K:t9b_mxfp8_gemv", "K:t9b2r2_norm_row"),      # owned -> owned, glue prefetch
             ("K:t9b2r2_norm_row", "K:sparse_attn"),         # owned -> borrowed: unchanged
             ("K:sparse_attn", "K:t9b2_o_rope_qdq"),         # borrowed -> owned
             ("mm", "K:t9b5_mix_fp32"),                      # torch -> owned GEMV-family prefetch
             ("K:dg_masked_gemm", "K:dg_masked_gemm")]       # borrowed -> borrowed
    c = pdl.classify_edges(edges)
    assert (c["edges"], c["pdl_owned"], c["pdl_borrowed"], c["none"]) == (6, 2, 2, 2)
    assert (c["pf_gemv_owned"], c["pf_glue_owned"], c["pf_gemv_borrowed"], c["pf_glue_borrowed"]) == (1, 1, 1, 0)
    assert c["into_borrowed_from_owned"] == 1
    m = pdl.model_ms(c)
    assert m["mixed"]["on_ms"] == round(2 * 1.2914e-3, 3)
    assert m["mixed"]["prefetch_ms"] == round(2 * 1.2914e-3 + 298.7454 / 39 * 1e-3, 3)
    assert "MODEL" in m["label"]
    env("on", exclude="gemv")
    c2 = pdl.classify_edges(edges)
    assert c2["pdl_owned"] == 0 and c2["pdl_borrowed"] == 3 and c2["pf_gemv_owned"] == 0   # a PDL-off producer = borrowed
    assert pdl.owned_name("t9b_mxfp8_gemv_0") == "t9b_mxfp8_gemv" and pdl.owned_name("fp8_gemm") is None
    assert any("owned->owned PDL 0, borrowed->owned PDL 3" in line for line in pdl.census_lines(c2))
    assert pdl.census_lines({"error": "x"})[0].endswith("x")


# ================================================================================ opt-in: existing suites under PDL
SHIM_SUITES = ["test_compressor_fixed_cpu.py", "test_dg_inplace_cpu.py", "test_fp32_fixed_cpu.py",
               "test_fused_glue_cpu.py", "test_fused_glue_r2_cpu.py", "test_fused_glue_t9b7_cpu.py",
               "test_gemv_kernel_shim_cpu.py", "test_head_bf16_cpu.py", "test_m1_port_cpu.py",
               "test_moe_rows_t10_cpu.py", "test_pdl_probe_cpu.py", "test_router_plan_cpu.py",
               "test_router_plan_fix_cpu.py", "test_router_plan_rows_cpu.py", "test_spare_compact_cpu.py",
               "test_t9b4_cpu.py"]


@pytest.mark.skipif(os.environ.get("DSV41_PDL_SLOW") != "1", reason="~20 min: DSV41_PDL_SLOW=1 to run")
@pytest.mark.parametrize("mode,trig", [("prefetch", "entry"), ("prefetch", "late")])
def test_existing_shim_suites_pass_under_pdl(mode, trig, tmp_path):
    """Every shim-run kernel test of the repo passes with PDL on (bits unchanged vs their oracles; the shim asserts
    the PDL invariants on every launch) and, together, they launch every owned kernel with launch_pdl."""
    cov = tmp_path / "cov.json"
    e = dict(os.environ, DSV41_PDL=mode, DSV41_PDL_TRIGGER=trig, PDL_COV_OUT=str(cov),
             PYTHONPATH=os.pathsep.join([os.path.join(REPO, "tests"), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "pdl_cov_plugin"]
                       + [os.path.join("tests", f) for f in SHIM_SUITES], cwd=REPO,   # suites open runtime/ relatively
                       env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]
    got = set(json.loads(cov.read_text())["pdl_kernels"]) | SHIM_DIRECT
    assert set(pdl.KERNELS) - got == set(), sorted(set(pdl.KERNELS) - got)


# launched through the shim by this file directly (step_marker) or not runnable under the shim at all (the gathers'
# int64 -> pointer casts; covered by the AST structure test here and by tests/gpu/test_pdl_gpu.py)
SHIM_DIRECT = {"t8_step_marker", "t8_gather_rows", "t8_gather_rows_u8", "t9b2_gather_tables", "t9b2_gather_tables_u8"}
