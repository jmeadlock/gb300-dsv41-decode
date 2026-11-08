"""Mac CPU test of the M2-L5 PDL probe (bench/pdl_probe.py + bench/pdl_kernels.py).
`.venv/bin/python -m pytest -q tests/test_pdl_probe_cpu.py`

What it checks (Triton compilation, PDL itself, the L2 prefetch asm and every timing are box-only):
  1. the copies are faithful: the helpers are byte-identical to the runtime's, the GEMV K loop is line-for-line the
     runtime's, and both kernel bodies -- executed program by program through tests/triton_shim.py -- equal the runtime's
     own kernels bit for bit in all four variants (a plain / b PDL / c line prefetch / c2 bulk prefetch);
  2. the PDL structure, per program, from an event log the shim stand-ins for `tl.extra.cuda.gdc_*` and the prefetch
     asm write: a: no gdc_*, launch_pdl=False; b/c/c2: exactly one gdc_wait and one gdc_launch_dependents per program,
     wait before trigger, every prefetch BEFORE the wait; source order: the wait precedes the first producer-data
     load and is never inside a loop;
  3. the prefetch covers exactly the program's own weight / weight-scale slice (every needed 128 B line, no line outside
     the slice, nothing from the activation), bulk prefetch rows 16 B aligned and a multiple of 16 bytes;
  4. the glue kernel writes the quantized row into the first N columns of a wider destination (the next GEMV's
     activation) and leaves the tail alone; the split-K fold and the unsplit form both equal the runtime kernel;
  5. a 3-pair chain (GEMV -> glue -> GEMV ...) on small shapes, all four variants bit-identical to each other and to
     the torch mirror (gemv_torch + the glue mirror); State arena compare; no host tensor touches the default device;
  6. the probe's pure helpers: statistics, paired differences, chain specs, config cache, edge labels, verdict text,
     scale-code table == the reference table.
"""
from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import os
import re
import sys
import types

import numpy as np
import pytest
import torch

import triton_shim
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from runtime import gemv_mxfp8 as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16


def _bits(t):
    t = t.contiguous()
    n = t.element_size()
    return t.view(torch.uint8) if n == 1 else t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[n])


def _eq(a, b):
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


# ================================================================================ fixtures
class _Proxy:
    """Stands in for a jit function in the kernel module: records each launch (grid, launch_pdl, USE_PDL, PREFETCH)
    and strips `launch_pdl` (a real-Triton launch option the shim's launcher does not know)."""

    def __init__(self, jf, name, log, launches):
        self.jf, self.name, self.log, self.launches = jf, name, log, launches

    def __getitem__(self, grid):
        inner = self.jf[grid]

        def launch(*a, launch_pdl=None, **k):
            self.launches.append({"kernel": self.name, "grid": tuple(grid), "launch_pdl": launch_pdl,
                                  "USE_PDL": k.get("USE_PDL"), "PREFETCH": k.get("PREFETCH")})
            self.log.append(("launch", self.name))
            return inner(*a, **k)
        return launch


@pytest.fixture(scope="module")
def KS():
    """bench/pdl_kernels.py loaded under the shim with `tl.extra.cuda.gdc_*` and the prefetch asm stood in for: both
    append to KS.log ((kind, pid, ...) tuples). A private copy -- sys.modules and runtime/ are untouched."""
    log, launches = [], []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(triton_shim.P, "to", lambda self, dtype: self, raising=False)   # ptr.to(tl.int64) is the identity here
        with triton_shim.installed():
            tl = sys.modules["triton.language"]
            tl.extra.cuda = types.SimpleNamespace(
                gdc_wait=lambda: log.append(("wait", tuple(triton_shim._PID))),
                gdc_launch_dependents=lambda: log.append(("trigger", tuple(triton_shim._PID))))
            orig = tl.inline_asm_elementwise

            def asm(asm_str, constraints, args, dtype, is_pure, pack):
                if "prefetch" in asm_str:
                    assert not is_pure and pack == 1 and dtype is tl.int32, asm_str   # a side effect, kept by the compiler
                    p = args[0]
                    if "cp.async.bulk.prefetch.L2.global" in asm_str:
                        log.append(("pf_bulk", tuple(triton_shim._PID), p.buf.size, p.off.copy(),
                                    np.broadcast_to(np.asarray(triton_shim._v(args[1])), p.off.shape).copy()))
                    else:
                        assert "prefetch.global.L2" in asm_str, asm_str
                        log.append(("pf_l2", tuple(triton_shim._PID), p.buf.size, p.off.copy()))
                    return triton_shim.T(np.zeros(np.shape(p.off), np.int32))
                return orig(asm_str, constraints, args, dtype, is_pure, pack)
            tl.inline_asm_elementwise = asm
            spec = importlib.util.spec_from_file_location("pdl_kernels_under_shim",
                                                          os.path.join(REPO, "bench", "pdl_kernels.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["pdl_kernels_under_shim"] = mod
            spec.loader.exec_module(mod)
        assert mod.HAVE_TRITON
        mod.pdl_gemv = _Proxy(mod.pdl_gemv, "gemv", log, launches)
        mod.pdl_norm_quant = _Proxy(mod.pdl_norm_quant, "norm", log, launches)
        mod.log, mod.launches = log, launches
        yield mod
    sys.modules.pop("pdl_kernels_under_shim", None)


@pytest.fixture(scope="module", autouse=True)
def _runtime_kernels_pdl_off():
    """M2-L5: the runtime kernels here are the probe's reference (today's launch); KS's module-wide `P.to` identity
    patch cannot carry the runtime's own prefetch (tests/test_pdl_impl_cpu.py covers that), so pin $DSV41_PDL off
    even when the suite runs under DSV41_PDL=on|prefetch."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DSV41_PDL", "off")
        yield


@pytest.fixture(scope="module")
def KG():
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "gemv_under_shim_pdl")


@pytest.fixture(scope="module")
def KR():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue_r2.py"), "fg2_under_shim_pdl")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("pdl_probe_under_test", os.path.join(REPO, "bench", "pdl_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pdl_probe_under_test"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("pdl_probe_under_test", None)


class _RefQ:
    fp8_block_size, scale_fmt, scale_dtype = 32, "ue8m0", E8

    @staticmethod
    def act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=E8, inplace=False):
        from test_static_emulated_cpu import act_quant

        q, s = act_quant(x, block_size, scale_fmt, scale_dtype, inplace)
        return q, s.to(E8)


@pytest.fixture(scope="module")
def tab(probe):
    return probe.scale_code_table("cpu")


def _src(fn):
    return inspect.getsource(fn.fn)


def _gcfg(c):
    return G.GemvConfig(c.block_n, c.ku, c.num_warps, c.num_stages, c.split_k, c.mma)


def _gemv_operands(M, N, K, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *sh: torch.randint(0, 256, sh, generator=g, dtype=torch.uint8, device="cpu")  # noqa: E731
    f8 = lambda *sh: (((r(*sh) & 0x80) | (0x30 + r(*sh) % 24)).to(torch.uint8)).view(FP8)  # noqa: E731
    xq, w = f8(M, K), f8(N, K)
    xs = torch.randint(122, 133, (M, K // 32), generator=g, dtype=torch.uint8, device="cpu").view(E8)
    s = torch.randint(122, 133, ((N + 31) // 32, K // 32), generator=g, dtype=torch.uint8, device="cpu").view(E8)
    return xq, xs, w, s


# ================================================================================ 1. the copies are faithful
def test_helpers_are_verbatim_copies(KS, KG, KR):
    assert _src(KS._t9b_pow2) == _src(KG._t9b_pow2)
    for name in ("_t9b2_code", "_t9b2_ipow2", "_t9b2_q8", "_t9b2_bf16", "_t9b4_ld", "_r2_offs", "_r2_col", "_r2_halve",
                 "_r2_pairs", "_r2_level", "_r2_finish", "_r2_fin"):
        assert _src(getattr(KS, name)) == _src(getattr(KR, name)), name


def _loop(path, start, stop):
    t = open(path).read()
    i = t.index(start)
    j = t.index(stop, i)
    return [ln.strip() for ln in t[i:j].splitlines() if ln.strip()]


def test_gemv_k_loop_and_stores_are_the_runtimes():
    a = _loop(os.path.join(REPO, "bench", "pdl_kernels.py"), "for kb0 in range(kb_lo, kb_hi, KU):", "mask = m_ok[:, None]")
    b = _loop(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "for kb0 in range(kb_lo, kb_hi, KU):", "mask = m_ok[:, None]")
    assert a == b and len(a) > 15
    pk = open(os.path.join(REPO, "bench", "pdl_kernels.py")).read()
    rg = open(os.path.join(REPO, "runtime", "gemv_mxfp8.py")).read()
    for line in ("mask = m_ok[:, None] & n_ok[None, :]",
                 "tl.store(ws + ks * M * N + rm[:, None] * N + rn[None, :], acc, mask=mask)",
                 "tl.store(out + rm[:, None] * stride_om + rn[None, :], acc.to(tl.bfloat16), mask=mask)"):
        assert line in pk and line in rg, line


def test_gemv_prologue_and_glue_body_are_the_runtimes():
    pk = open(os.path.join(REPO, "bench", "pdl_kernels.py")).read()
    rg = open(os.path.join(REPO, "runtime", "gemv_mxfp8.py")).read()
    for line in ("w_rows = w + rn_in.to(tl.int64)[:, None] * stride_wn + rj[None, :]",
                 "s_rows = s + (rn_in // 32).to(tl.int64) * stride_sn",
                 "acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)",
                 "sbf = _t9b_pow2(tl.load(s_rows + kbc).to(tl.int32))"):
        assert line in pk and line in rg, line
    rn = open(os.path.join(REPO, "runtime", "fused_glue_r2.py")).read()
    for mine, theirs in (
            ("r = libdevice.rsqrt(_r2_fin(_r2_finish(acc, T, V, DEC), fac, wf, DIV) + eps)",) * 2,
            ("code = _t9b2_code(tl.maximum(tl.max(tl.abs(a), axis=1), tl.max(tl.abs(b), axis=1)), tab)",
             "code = _t9b2_code(tl.maximum(tl.max(tl.abs(a2), axis=1), tl.max(tl.abs(b2), axis=1)), tab)"),
            ("v = _t9b4_ld(x, e, m, NS, XS)",) * 2,
            ("a = _t9b2_bf16(tl.load(nw + ce, mask=m, other=0.0).to(tl.float32)",) * 2,
            ("* (_t9b4_ld(x, ce + 1, m, NS, XS) * r))",) * 2,
            ("qa = _t9b2_q8(a, inv)", "qa = _t9b2_q8(a2, inv)"),
            ("tl.store(s + bi, code.to(tl.uint8), mask=bok)",) * 2):
        assert mine in pk and theirs in rn, (mine, theirs)


# ================================================================================ 2. PDL structure
def _by_pid(log):
    out = {}
    for ev in log:
        if ev[0] != "launch":
            out.setdefault(ev[1], []).append(ev)
    return out


GEMV_CASES = [  # (M, N, K, Cfg(block_n, ku, warps, stages, split_k, mma))
    (6, 100, 1280, (32, 4, 4, 2, 7, 0)),     # tail KU, short last split, N % 32 != 0
    (6, 96, 640, (64, 8, 4, 2, 3, 1)),       # MMA, PER 7 with KU 8
    (6, 64, 256, (32, 4, 4, 2, 1, 0)),       # no split: direct bf16 store
    (3, 160, 1024, (128, 2, 4, 2, 8, 1)),    # BLOCK_N > 32 rows of scales, EVEN split
]


@pytest.mark.parametrize("M,N,K_,cfg", GEMV_CASES, ids=[f"M{c[0]}-N{c[1]}-K{c[2]}-s{c[3][4]}" for c in GEMV_CASES])
@pytest.mark.parametrize("pdl,pf", [(False, 0), (True, 0), (True, 1), (True, 2)], ids=["a", "b", "c", "c2"])
def test_gemv_variants_equal_runtime_and_have_the_pdl_structure(KS, KG, M, N, K_, cfg, pdl, pf):
    cfg = KS.Cfg(*cfg)
    xq, xs, w, s = _gemv_operands(M, N, K_, seed=M + N + K_ + cfg.split_k)
    p = KS.gemv_plan(cfg, M, N, K_)
    lp = G.launch_plan(_gcfg(cfg), M, N, K_)
    assert {k: p[k] for k in ("grid", "PER", "NSPLIT", "BLOCK_M", "EVEN", "SPLIT")} == \
        {k: lp[k] for k in ("grid", "PER", "NSPLIT", "BLOCK_M", "EVEN", "SPLIT")}
    ws = torch.full((p["NSPLIT"], M, N), float("nan"), dtype=torch.float32, device="cpu")
    out = torch.full((M, N), float("nan"), dtype=BF, device="cpu")
    KS.log.clear()
    KS.launches.clear()
    KS.launch_gemv(xq, xs, w, s, out, ws if p["SPLIT"] else out, cfg, pdl=pdl, prefetch=pf)
    # the runtime's own kernel, same operands
    ws2 = torch.full((p["NSPLIT"], M, N), float("nan"), dtype=torch.float32, device="cpu")
    out2 = torch.full((M, N), float("nan"), dtype=BF, device="cpu")
    KG._launch(xq, xs, w, s, out2, _gcfg(cfg), ws2 if p["SPLIT"] else out2, reduce=False)
    if p["SPLIT"]:
        assert _eq(ws, ws2) and not torch.isnan(ws).any()
        got = torch.zeros(M, N, dtype=torch.float32, device="cpu")
        for j in range(p["NSPLIT"]):
            got = got + ws[j]
        assert torch.equal(got.to(BF), G.gemv_torch(xq, xs, w, s, split_k=cfg.split_k))   # the fold the glue does
    else:
        assert _eq(out, out2) and torch.equal(out, G.gemv_torch(xq, xs, w, s, split_k=1))
    # launch attribute + constexprs
    (ln,) = KS.launches
    assert (ln["launch_pdl"], ln["USE_PDL"], ln["PREFETCH"]) == (pdl, pdl, pf)
    # per-program event sequence
    seq = _by_pid(KS.log)
    nprog = p["grid"][0] * p["grid"][1]
    if not pdl:
        assert seq == {}
        return
    assert len(seq) == nprog
    for pid, evs in seq.items():
        kinds = [e[0] for e in evs]
        assert kinds.count("wait") == 1 and kinds.count("trigger") == 1, (pid, kinds)
        iw, it = kinds.index("wait"), kinds.index("trigger")
        assert iw < it
        assert all(k.startswith("pf") for k in kinds[:iw]) and kinds[iw + 1:] == ["trigger"]   # prefetch only BEFORE the wait
        assert [k for k in kinds[:iw]] == ([] if pf == 0 else ["pf_l2", "pf_l2" if pf == 1 else "pf_bulk"])
    if pf:
        _check_prefetch_cover(KS, seq, p, cfg, M, N, K_, w, s, pf)


def _check_prefetch_cover(KS, seq, p, cfg, M, N, K_, w, s, pf):
    kb, per = K_ // 32, p["PER"]
    for (pn, ks, _), evs in seq.items():
        lo, hi = ks * per, min((ks + 1) * per, kb)
        rows = [r for r in range(pn * cfg.block_n, (pn + 1) * cfg.block_n) if r < N]   # valid rows only
        # weights: the lines of this program's row slices
        need_w = {(r * K_ + b) // 128 for r in rows for b in range(lo * 32, hi * 32)}
        need_s = {((r // 32) * (K_ // 32) + b) // 128 for r in rows for b in range(lo, hi)}
        got_w, got_s, bulk = set(), set(), []
        for e in evs:
            if e[0] in ("wait", "trigger"):
                continue
            if e[0] == "pf_l2" and e[2] == w.numel():
                got_w |= {int(o) // 128 for o in e[3].reshape(-1)}
            elif e[0] == "pf_l2" and e[2] == s.numel():
                got_s |= {int(o) // 128 for o in e[3].reshape(-1)}
            elif e[0] == "pf_bulk":
                assert e[2] == w.numel()
                bulk.append(e)
            else:
                raise AssertionError(("a prefetch outside the weight / scale buffers", e[:3]))
        assert got_s == need_s, (pn, ks)                         # exact: every needed scale line, none outside
        if pf == 1:
            assert got_w == need_w, (pn, ks)                     # exact: every needed weight line, none outside
        else:
            (b,) = bulk
            offs, sizes = b[3].reshape(-1), b[4].reshape(-1)
            assert (offs % 16 == 0).all() and (sizes % 16 == 0).all() and (sizes == (hi - lo) * 32).all()
            assert {int(r) for r in rows} == {int(o) // K_ for o in offs}
            assert all(int(o) % K_ == lo * 32 for o in offs)
            cover = {(int(o) + i) // 128 for o in offs for i in range(int(sizes[0]))}
            assert cover == need_w


def test_pdl_is_outside_every_loop_and_before_the_first_producer_load():
    t = open(os.path.join(REPO, "bench", "pdl_kernels.py")).read()
    for kern, first_load, loop in (("pdl_gemv", "wt = tl.load(", "for kb0 in range"),
                                   ("pdl_norm_quant", "x = x + t * N", "for k in range")):
        body = t[t.index(f"def {kern}("):]
        body = body[:body.index("@triton.jit")] if "@triton.jit" in body[10:] else body
        assert body.count("tl.extra.cuda.gdc_wait()") == 1 and body.count("tl.extra.cuda.gdc_launch_dependents()") == 1
        iw = body.index("tl.extra.cuda.gdc_wait()")
        assert iw < body.index(first_load) and iw < body.index(loop)
        assert body.index("tl.extra.cuda.gdc_launch_dependents()") > iw
        line = [ln for ln in body.splitlines() if "tl.extra.cuda.gdc_wait()" in ln][0]
        assert line.startswith(" " * 12) and not line.startswith(" " * 16)      # directly under `if USE_PDL:`
    # prefetch happens strictly before the GEMV's wait and never touches xq / xs
    g = t[t.index("def pdl_gemv("):]
    pre = g[:g.index("tl.extra.cuda.gdc_wait()")]
    assert "_pf_l2(" in pre and "_pf_bulk(" in pre
    assert "xq" not in pre[pre.index("if PREFETCH > 0:"):] and "xs" not in pre[pre.index("if PREFETCH > 0:"):]


# ================================================================================ 3. the glue kernel
def _norm_inputs(R, N, ns, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if ns:
        x = (torch.randn(ns, R, N, generator=g, dtype=torch.float32, device="cpu") * 3.0).contiguous()
    else:
        x = (torch.randn(R, N, generator=g, dtype=torch.float32, device="cpu") * 3.0).to(BF).contiguous()
    nw = (torch.randn(N, generator=g, dtype=torch.float32, device="cpu") * 0.3 + 1.0).to(BF)
    return x, nw


@pytest.mark.parametrize("N,ns,pdl", [(160, 0, False), (160, 3, True), (96, 2, True), (320, 4, False)])
def test_norm_quant_equals_runtime_kernel_and_writes_into_a_wider_row(KS, KR, tab, N, ns, pdl):
    R = 6
    x, nw = _norm_inputs(R, N, ns, seed=N + ns)
    o = KS.norm_order(N)
    assert (o["T"], o["V"], o["U"], o["DEC"], o["DIV"]) == tuple(
        (lambda p: (p[1], p[2], False, bool(p[3]), False))(R2.predicted_order(N, 1)))
    assert KS.fac_of(N) == R2.fac_of(N)
    # runtime kernel, compact destination
    q2 = torch.zeros(R, N, dtype=FP8, device="cpu")
    s2 = torch.zeros(R, N // 32, dtype=E8, device="cpu")
    o2 = torch.zeros(R, N, dtype=BF, device="cpu")
    KR.t9b2r2_norm_row[(R,)](x, nw, o2, x, q2, s2.view(torch.uint8), nw, tab, 1e-6, R2.fac_of(N), float(N),
                             N=N, T=o["T"], V=o["V"], U=o["U"], DEC=o["DEC"], DIV=o["DIV"], HAS_SLOT=False, QQ=True,
                             HAS_Q=True, QDQ=False, RD=0, RE_MODE=0, IM_MODE=0, BB=min(16, KS._np2(N // 32)),
                             XS=R * N if ns else 0, NS=ns, num_warps=8, enable_fp_fusion=False)
    # the copy, destination 64 columns wider (the next GEMV's activation buffer) with a sentinel tail
    W = N + 64
    q = torch.full((R, W), 1.5, dtype=FP8, device="cpu")
    s = torch.full((R, W // 32), 127, dtype=torch.uint8, device="cpu").view(E8)
    ob = torch.zeros(R, N, dtype=BF, device="cpu")
    KS.log.clear()
    KS.launches.clear()
    KS.launch_norm_quant(x, nw, ob, q, s, tab, N=N, R=R, ns=ns, xs_stride=R * N if ns else 0, qrs=W, srs=W // 32,
                         pdl=pdl)
    assert _eq(ob, o2)
    assert _eq(q[:, :N], q2) and _eq(s[:, :N // 32], s2)
    assert (q[:, N:].float() == 1.5).all() and (s[:, N // 32:].view(torch.uint8) == 127).all()    # tail untouched
    (ln,) = KS.launches
    assert (ln["launch_pdl"], ln["USE_PDL"]) == (pdl, pdl)
    seq = _by_pid(KS.log)
    if pdl:
        assert len(seq) == R and all([e[0] for e in v] == ["wait", "trigger"] for v in seq.values())
    else:
        assert seq == {}


# ================================================================================ 4. a chain
SMALL = {"A": (96, 160), "B": (160, 256)}


def _make_chain(KS, probe, kind, cfg_user, arena=False):
    specs = probe.chain_specs(kind, 3, SMALL)
    cfgs = probe.pick_cfgs(KS, specs, 6, 148, cfg_user, None)
    wt = probe.Weights(specs, "cpu")
    inputs = probe.Inputs(specs, 6, 2)
    return specs, cfgs, wt, inputs


def _torch_chain(KS, probe, specs, cfgs, wt, inputs, tab, i_in):
    R = 6
    P = len(specs)
    xq = [None] * P
    xs = [None] * P
    for i, (_, N, Kd) in enumerate(specs):
        xq[i] = torch.zeros(R, Kd, dtype=FP8, device="cpu")
        xs[i] = torch.zeros(R, Kd // 32, dtype=E8, device="cpu")
        if i:
            n_prev = specs[i - 1][1]
            xq[i][:, n_prev:].view(torch.uint8).copy_(inputs.tail[i][0].view(torch.uint8))
            xs[i][:, n_prev // 32:].view(torch.uint8).copy_(inputs.tail[i][1].view(torch.uint8))
    xq[0].view(torch.uint8).copy_(inputs.x[i_in][0].view(torch.uint8))
    xs[0].view(torch.uint8).copy_(inputs.x[i_in][1].view(torch.uint8))
    outs = {}
    for i, (_, N, Kd) in enumerate(specs):
        g = G.gemv_torch(xq[i], xs[i], wt.w[i], wt.s[i], split_k=cfgs[i].split_k)
        o = KS.norm_order(N)
        ordd = {"order": ("vec", o["T"], o["V"], o["DEC"]), "fin": "mul", "exact": True,
                "name": R2.order_name(("vec", o["T"], o["V"], o["DEC"]), "mul")}
        ob = torch.zeros(R, N, dtype=BF, device="cpu")
        qt = torch.zeros(R, N, dtype=FP8, device="cpu")
        st = torch.zeros(R, N // 32, dtype=E8, device="cpu")
        R2.launch_norm_row(g.reshape(-1), wt.nw[i], ob, eps=1e-6, ordd=ordd, code_tab=tab, q=qt, s=st)
        outs[f"gout{i}"], outs[f"ob{i}"] = g, ob
        if i + 1 < P:
            xq[i + 1][:, :N].view(torch.uint8).copy_(qt.view(torch.uint8))
            xs[i + 1][:, :N // 32].view(torch.uint8).copy_(st.view(torch.uint8))
        else:
            outs["qf"], outs["sf"] = qt, st
    for i in range(P):
        outs[f"xq{i}"], outs[f"xs{i}"] = xq[i], xs[i]
    return outs


@pytest.mark.parametrize("kind,split", [("mixed", 2), ("mixed", 1), ("A", 2)])
def test_chain_all_variants_bit_identical_and_equal_to_the_torch_mirror(KS, probe, tab, kind, split):
    user = {"A": KS.Cfg(32, 2, 4, 1, split, 0), "B": KS.Cfg(32, 4, 4, 1, split, 0)}
    specs, cfgs, wt, inputs = _make_chain(KS, probe, kind, user)
    states = {}
    for n, v in probe.VARIANTS.items():
        st = probe.State(specs, 6, "cpu", arena=False)
        st.set_static(inputs)
        probe.build_plans(KS, st, cfgs)
        st.set_input(*inputs.x[1])
        KS.log.clear()
        KS.launches.clear()
        probe.run_chain(KS, st, wt, cfgs, tab, pdl=v["pdl"], prefetch=v["prefetch"])
        assert len(KS.launches) == 2 * len(specs)                                   # 2 launches per pair
        assert [l["kernel"] for l in KS.launches] == ["gemv", "norm"] * len(specs)
        assert all(l["launch_pdl"] == v["pdl"] and l["PREFETCH"] in (None, v["prefetch"]) for l in KS.launches)
        states[n] = st
    ref = _torch_chain(KS, probe, specs, cfgs, wt, inputs, tab, 1)
    for n, st in states.items():
        assert int(st.mismatch(states["a"])) == 0, n
        for k, want in ref.items():
            if k.startswith("gout") and st.plans[int(k[4:])]["SPLIT"]:
                continue                                                            # split plans keep fp32 partials in ws
            assert _eq(st.t[k], want), (n, k)
    snap = states["b"].snapshot()
    assert int(states["b"].mismatch(snap)) == 0
    states["b"].t["ob1"].view(torch.uint8)[0, 0] ^= 1
    assert int(states["b"].mismatch(snap)) == 1


def test_arena_state_aliases_one_buffer_and_compares_by_bytes(probe, KS):
    specs = probe.chain_specs("mixed", 3, SMALL)
    st = probe.State(specs, 6, "cpu", arena=True)
    other = probe.State(specs, 6, "cpu", arena=True)
    assert st.arena.numel() % probe.ALIGN == 0 and int(st.mismatch(other)) == 0
    st.t["ob1"][2, 3] = 1.0
    assert int(st.mismatch(other)) > 0 and st.arena.sum() > 0       # the bf16 view writes through to the arena
    snap = st.snapshot()
    assert int(st.mismatch(snap)) == 0
    for k, v in st.t.items():                                         # every view lies inside the arena
        base = v.data_ptr() - st.arena.data_ptr()
        assert 0 <= base and base + v.numel() * v.element_size() <= st.arena.numel(), k


def test_host_data_never_touches_the_default_device(probe, KS):
    """Weights / Inputs / the code table / norm constants build explicit-CPU tensors: under a meta default device they
    still work (the box's default device is cuda)."""
    with torch.device("meta"):
        specs = probe.chain_specs("mixed", 3, SMALL)
        wt = probe.Weights(specs, "cpu")
        inputs = probe.Inputs(specs, 6, 2)
        t = probe.scale_code_table("cpu")
        f = KS.fac_of(5120)
    assert wt.w[0].device.type == "cpu" and inputs.x[0][0].device.type == "cpu" and t.device.type == "cpu"
    assert f == R2.fac_of(5120)


# ================================================================================ 5. plans, configs, helpers
def test_plan_helpers_equal_the_runtimes(KS):
    for N, K_, bn, sms in ((1280, 5120, 32, 148), (5120, 8192, 64, 148), (96, 160, 32, 148), (4096, 4096, 128, 132)):
        assert KS.auto_split_k(N, K_, bn, sms) == G.auto_split_k(N, K_, bn, sms)
    for kb, sk in ((160, 8), (256, 2), (5, 2), (40, 7), (9, 9), (12, 5)):
        assert KS.split_plan(kb, sk) == G.split_plan(kb, sk)
    for m in (1, 2, 3, 6, 8):
        assert KS.block_m(m) == G.block_m(m)
    assert KS.DEFAULT_CFG.as_dict() == G.DEFAULT_CONFIG.as_dict()
    assert (KS.BLOCK, KS.MMA_BLOCK_M, KS.GEMV_MAX_M, KS.SPLIT_MIN_KB, KS.SPLIT_PROGRAMS_PER_SM) == \
        (G.BLOCK, G.MMA_BLOCK_M, G.GEMV_MAX_M, G.SPLIT_MIN_KB, G.SPLIT_PROGRAMS_PER_SM)
    assert KS.NORM_WARPS == R2.NORM_WARPS and KS.NORM_EPS == 1e-6
    for N in (96, 160, 1280, 5120, 7168):
        o = KS.norm_order(N)
        p = R2.predicted_order(N, 1)
        assert (o["T"], o["V"], o["DEC"]) == (p[1], p[2], p[3]) and not o["U"]


def test_prefetch_line_counts_cover_unaligned_runs(KS):
    for per in (1, 3, 4, 5, 20, 21, 128, 129):
        n = KS.prefetch_lines(per)
        assert n & (n - 1) == 0
        for start in range(0, 128, 32):              # a run begins on a 32 B boundary (kb_lo * 32)
            L = per * 32
            lines = {(start + min(j * 128, L - 1)) // 128 for j in range(n)}
            assert lines == set(range((start) // 128, (start + L - 1) // 128 + 1)), (per, start)
        ns = KS.prefetch_slines(per)
        for start in range(0, 128):
            lines = {(start + min(j * 128, per - 1)) // 128 for j in range(ns)}
            assert lines == set(range(start // 128, (start + per - 1) // 128 + 1)), (per, start)


def test_statistics(probe):
    s = probe.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5 and s["mean"] == 22.0 and s["p50"] == 3.0 and s["min"] == 1.0 and s["max"] == 100.0
    assert abs(s["std"] - np.std([1, 2, 3, 4, 100], ddof=1)) < 1e-9 and abs(s["sem"] - s["std"] / 5 ** 0.5) < 1e-12
    assert probe.summarize([])["n"] == 0 and probe.summarize([7.0])["std"] == 0.0
    assert probe.pctl([0.0, 10.0], 95) == 9.5
    d = probe.paired_diff([10.0, 12.0, 11.0, 13.0], [9.0, 10.0, 11.0, 10.0])
    assert d["mean"] == 1.5 and d["n"] == 4 and d["frac_a_slower"] == 0.75
    assert abs(d["ci95"] - 1.96 * d["sem"]) < 1e-12


def test_savings_per_pair_and_per_edge(probe):
    tim = {"a": {"us": [200.0, 202.0, 198.0, 200.0]}, "b": {"us": [190.0, 193.0, 187.0, 190.0]},
           "c": {"us": [180.0, 182.0, 178.0, 180.0]}}
    s = probe.savings(tim, pairs=40)
    assert s["a-b"]["per_replay_us"] == 10.0 and s["a-b"]["per_pair_us"] == 0.25
    assert abs(s["a-b"]["per_edge_us"] - 10.0 / 79) < 1e-12                         # 2P - 1 = 79 owned edges
    assert s["a-c"]["per_replay_us"] == 20.0 and s["b-c"]["per_replay_us"] == 10.0 and s["a-b"]["ci95_us"] > 0
    d = probe.describe_timing({"us": [200.0, 220.0], "host_us": [5.0, 7.0]}, 40)
    assert d["replay_us"]["mean"] == 210.0 and d["us_per_pair"]["mean"] == 210.0 / 40 and d["host_us"]["mean"] == 6.0


def test_chain_specs(probe, KS):
    sp = probe.chain_specs("mixed", 40, KS.SHAPES)
    assert len(sp) == 40 and [s[0] for s in sp[:4]] == ["A", "B", "A", "B"] and sp[0][1:] == (1280, 5120)
    assert sp[1][1:] == (5120, 8192) and sum(1 for s in sp if s[0] == "A") == 20
    assert [s[0] for s in probe.chain_specs("B", 3, KS.SHAPES)] == ["B"] * 3
    assert [s[0] for s in probe.chain_specs("A,B,B", 99, KS.SHAPES)] == ["A", "B", "B"]
    assert [s[0] for s in probe.chain_specs("B,A,B", 3, KS.SHAPES)] == ["B", "A", "B"]      # B -> A exact K match, A -> B fits
    with pytest.raises(ValueError):
        probe.chain_specs("A,Z", 3, KS.SHAPES)                                            # unknown shape name
    with pytest.raises(ValueError):
        probe.chain_specs("B,B", 3, {"B": (5120, 4096)})                                  # N = 5120 does not fit K = 4096


def test_cfg_cache_and_picking(probe, KS, tmp_path):
    cache = tmp_path / "gemv-tune.json"
    cache.write_text(json.dumps({"1280x5120/m8": {"block_n": 64, "ku": 8, "num_warps": 8, "num_stages": 3,
                                                  "split_k": 5, "mma": 0}, "bogus": {"block_n": 1}}))
    c = probe.load_cfg_cache(str(cache))
    assert list(c) == ["1280x5120/m8"] and c["1280x5120/m8"].block_n == 64
    specs = probe.chain_specs("mixed", 4, KS.SHAPES)
    cfgs = probe.pick_cfgs(KS, specs, 6, 148, {"B": KS.Cfg(32, 2, 4, 2, 3, 1)}, c)
    assert cfgs[0].split_k == 5 and cfgs[0].ku == 8                 # cache wins for A
    assert cfgs[1].split_k == 3 and cfgs[1].mma == 1                # user override for B
    d = probe.pick_cfgs(KS, specs, 6, 148, None, None)
    assert d[0].split_k > 0 and d[1].split_k > 0                    # untuned default, split resolved
    assert probe.parse_cfg("32,4,4,1,0,0").as_dict() == KS.Cfg(32, 4, 4, 1, 0, 0).as_dict()
    with pytest.raises(ValueError):
        probe.parse_cfg("1,2,3")
    assert probe.load_cfg_cache(None) == {}


def test_edge_labels_and_tally(probe):
    assert probe.edge_label(0, 0, 0) == "default" and probe.edge_label(1, 1, 0) == "programmatic"
    assert probe.edge_label(1, 2, 0) == "programmatic(launch_completion)"
    assert probe.tally_edges([(0, 0, 0)] * 3 + [(1, 1, 0)] * 79) == {"default": 3, "programmatic": 79}


def test_verdict_text(probe):
    rep = {"chains": [{"kind": "mixed", "variants": {
        "a": {"census": {"nodes": 82, "kernel_nodes": 80, "edges": 81, "edge_kinds": {"default": 81}}},
        "b": {"census": {"nodes": 82, "kernel_nodes": 80, "edges": 81, "edge_kinds": {"default": 2, "programmatic": 79}}},
        "c2": {"compile": "FAILED", "error": "PTXASError: bad asm"},
        "c": {"capture": "FAILED", "capture_error": "RuntimeError: boom"}},
        "graph": {"variants": ["a", "b"], "stress": {"replays": 400, "inputs": 8, "reference": "a",
                                                     "mismatched_bytes": {"a": 0, "b": 0}, "bit_identical": True},
                  "savings": {"a-b": {"per_replay_us": 30.0, "ci95_us": 1.0, "per_pair_us": 0.75, "per_edge_us": 0.38,
                                      "per_edge_ci95_us": 0.01, "n": 200, "frac_first_slower": 0.99}}},
        "eager": {"timing": {"a": {"replay_us": {"mean": 400.0, "p50": 399.0}, "host_us": {"mean": 900.0}}}}}]}
    text = "\n".join(probe.verdict(rep))
    for needle in ("ALL IDENTICAL", "programmatic': 79", "+0.380", "COMPILE FAILED: PTXASError", "CAPTURE FAILED: RuntimeError",
                   "eager a: gpu 400.0", "host launch 900.0"):
        assert needle in text, needle
    assert "programmatic on all 79 kernel-to-kernel edges" in text
    off = json.loads(json.dumps(rep))
    off["chains"][0]["variants"]["b"]["census"]["edge_kinds"] = {"default": 81}
    assert "PDL NOT ATTACHED under capture" in "\n".join(probe.verdict(off))
    part = json.loads(json.dumps(rep))
    part["chains"][0]["variants"]["b"]["census"]["edge_kinds"] = {"default": 40, "programmatic": 41}
    assert "programmatic on 41 of 79" in "\n".join(probe.verdict(part))
    bad = json.loads(json.dumps(rep))
    bad["chains"][0]["graph"]["stress"].update(bit_identical=False, mismatched_bytes={"a": 0, "b": 7})
    assert "MISMATCH" in "\n".join(probe.verdict(bad))


def test_scale_code_table_equals_the_reference_table(probe):
    assert torch.equal(probe.scale_code_table("cpu"), FG.scale_code_table(_RefQ, "cpu"))


def test_args_and_plan(probe, capsys):
    a = probe.parse_args([])
    assert (a.mode, a.pairs, a.rows, a.replays, a.batch, a.variants, a.chain_list) == \
        ("both", 40, 6, 200, 1, ["a", "b", "c", "c2"], ["mixed", "A", "B"])
    assert probe.parse_args(["--chains", "A,B,B;mixed", "--variants", "a,b"]).chain_list == ["A,B,B", "mixed"]
    assert probe.main(["--plan", "--chains", "mixed", "--pairs", "4"]) == 0
    out = capsys.readouterr().out
    assert "chain mixed: 4 pairs, 8 kernels, 7 edges" in out and "shape A N=1280 K=5120" in out


def test_files_obey_the_rules():
    for f in ("bench/pdl_probe.py", "bench/pdl_kernels.py"):
        t = open(os.path.join(REPO, f)).read()
        assert not re.search(r"^\s*(import|from)\s+vllm", t, re.M), f           # never import vllm
        assert not re.search(r"^\s*(import|from)\s+(runtime|reference)", t, re.M), f   # runtime/ is read, never imported here
        assert "subprocess" not in t and "import socket" not in t and "urllib" not in t and "requests" not in t, f
    t = open(os.path.join(REPO, "bench", "pdl_probe.py")).read()
    for m in re.finditer(r"torch\.(randint|rand|randn|zeros|ones|empty|arange|tensor|full)\(", t):
        window = t[m.start(): m.start() + 260]
        assert "device=" in window, window                                      # explicit device on every constructor
    assert "torch.Generator(device=\"cpu\")" in t and "torch.Generator()" not in t


# ================================================================================ 7. the whole probe flow, CUDA faked
class _FakeEvent:
    n = 0

    def __init__(self, enable_timing=False):
        _FakeEvent.n += 1
        self.t = 0.0

    def record(self, *a):
        _FakeEvent.n += 1
        self.t = _FakeEvent.n * 0.001

    def elapsed_time(self, other):
        return max(1e-3, other.t - self.t) * 100.0


class _FakeGraph:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        self.fn()


def test_run_one_chain_end_to_end_with_cuda_faked(probe, KS, tab, monkeypatch):
    """probe.run_one_chain on small shapes with torch.cuda stubbed and the graph = 'replay the captured function':
    exercises state building, the eager warm-up, the per-variant compile record, the stress comparison (a corrupted
    variant must be reported), the interleaved timing, the savings, the eager mode and the JSON receipt."""
    monkeypatch.setattr(KS, "SHAPES", SMALL)
    for name, fake in (("synchronize", lambda *a, **k: None), ("empty_cache", lambda: None), ("Event", _FakeEvent)):
        monkeypatch.setattr(torch.cuda, name, fake)
    monkeypatch.setattr(probe, "capture_graph", lambda fn, keep_graph=True, dot_path=None: (_FakeGraph(fn), True))
    monkeypatch.setattr(probe, "graph_census", lambda g: {"nodes": 8, "kernel_nodes": 6, "edges": 5,
                                                          "edge_kinds": {"default": 5}})
    args = probe.parse_args(["--pairs", "3", "--rows", "6", "--replays", "4", "--warmup", "1", "--stress", "5",
                             "--stress-inputs", "2", "--no-flush", "--no-arena", "--chains", "mixed"])
    cfgs_user = {"A": KS.Cfg(32, 2, 4, 1, 2, 0), "B": KS.Cfg(32, 4, 4, 1, 2, 0)}
    res = probe.run_one_chain(KS, "mixed", args, "cpu", 148, tab, {}, cfgs_user)
    assert res["pairs"] == 3 and res["kernels"] == 6 and res["edges_expected"] == 5
    assert set(res["variants"]) == {"a", "b", "c", "c2"} and all(v["compile"] == "ok" for v in res["variants"].values())
    g = res["graph"]
    assert g["stress"]["bit_identical"] and set(g["stress"]["mismatched_bytes"]) == {"a", "b", "c", "c2"}
    assert all(v == 0 for v in g["stress"]["mismatched_bytes"].values())
    assert g["replays"] == 4 and g["timing"]["a"]["replay_us"]["n"] == 4
    assert {"a-b", "a-c", "a-c2", "b-c", "b-c2"} <= set(g["savings"])
    e = res["eager"]
    assert e["bit_identical_vs_ref"] == {"a": True, "b": True, "c": True, "c2": True}
    assert e["timing"]["b"]["host_us"]["n"] == 4
    json.dumps(res)                                                   # the receipt serializes
    text = "\n".join(probe.verdict({"chains": [res]}))
    assert "ALL IDENTICAL" in text and "us/edge" in text and "eager a:" in text

    # a variant whose output differs is caught by the stress compare (the 'PDL changed a bit' failure mode)
    orig = probe.run_chain

    def corrupt(K, st, wt, cfgs, tab_, *, pdl, prefetch):
        orig(K, st, wt, cfgs, tab_, pdl=pdl, prefetch=prefetch)
        if pdl and prefetch == 1:
            st.t["ob1"].view(torch.uint8)[0, 0] ^= 1
    monkeypatch.setattr(probe, "run_chain", corrupt)
    bad = probe.run_one_chain(KS, "mixed", args, "cpu", 148, tab, {}, cfgs_user)
    mm = bad["graph"]["stress"]["mismatched_bytes"]
    assert mm["a"] == mm["b"] == mm["c2"] == 0 and mm["c"] > 0 and not bad["graph"]["stress"]["bit_identical"]
    assert "MISMATCH" in "\n".join(probe.verdict({"chains": [bad]}))
    assert bad["eager"]["bit_identical_vs_ref"]["c"] is False


def test_a_variant_that_fails_to_compile_is_reported_not_fatal(probe, KS, tab, monkeypatch):
    monkeypatch.setattr(KS, "SHAPES", SMALL)
    for name, fake in (("synchronize", lambda *a, **k: None), ("empty_cache", lambda: None), ("Event", _FakeEvent)):
        monkeypatch.setattr(torch.cuda, name, fake)
    monkeypatch.setattr(probe, "capture_graph", lambda fn, keep_graph=True, dot_path=None: (_FakeGraph(fn), True))
    monkeypatch.setattr(probe, "graph_census", lambda g: {"error": "no cudart"})
    orig = probe.run_chain

    def boom(K, st, wt, cfgs, tab_, *, pdl, prefetch):
        if prefetch == 2:
            raise RuntimeError("PTXAS error: Unknown modifier '.bulk'")
        orig(K, st, wt, cfgs, tab_, pdl=pdl, prefetch=prefetch)
    monkeypatch.setattr(probe, "run_chain", boom)
    args = probe.parse_args(["--pairs", "3", "--replays", "2", "--warmup", "0", "--stress", "2", "--stress-inputs", "2",
                             "--no-flush", "--no-arena", "--mode", "graph"])
    res = probe.run_one_chain(KS, "mixed", args, "cpu", 148, tab, {}, {"A": KS.Cfg(32, 2, 4, 1, 2, 0),
                                                                         "B": KS.Cfg(32, 4, 4, 1, 2, 0)})
    assert res["variants"]["c2"]["compile"] == "FAILED" and "Unknown modifier" in res["variants"]["c2"]["error"]
    assert res["graph"]["variants"] == ["a", "b", "c"] and "eager" not in res
    text = "\n".join(probe.verdict({"chains": [res]}))
    assert "variant c2 COMPILE FAILED: RuntimeError: PTXAS error" in text and "census unavailable (no cudart)" in text
