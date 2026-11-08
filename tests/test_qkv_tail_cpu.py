"""Mac CPU tests of M2-L10, the q/kv tail (docs/HANDOFF-M2-L10-QKV-TAIL.md): `.venv/bin/python -m pytest -q
tests/test_qkv_tail_cpu.py`

  1. module contract: gemv_mxfp8's copies of fused_glue_r2's helpers are verbatim; the ticket width agrees; the
     fence is static_kernels' fence; the switch ($DSV41_QKV_TAIL) parses, default split;
  2. the real `@triton.jit` bodies through tests/triton_shim.py: l10_qkv_tail (ONE launch: the pair GEMV + per-output
     ticket + both norm_row tails) == t9b4_mxfp8_gemv2 + t9b2r2_norm_row (q) + t9b2r2_norm_row (kv) on every output
     (qr, q, s, the whole ring) and on the GEMV results it hands back, for 1 / 3 / 6 rows, split and unsplit halves
     (ragged and EVEN splits), both block-partial forms, RoPE contraction forms, reduction-order families (vec /
     unroll, dec / inc, mul / div); the tickets are left at 0 (a second launch is identical); the launch grid is the
     pair kernel's; fused_glue_r2.launch_qkv_tail's kernel path == its CPU mirror;
  3. StaticStep on the emulated mini checkpoints (real split partials on CPU, gemv_mxfp8.CPU_SPLIT_K = 3), 1 and 6
     rows: qkv_tail fused == split == ref glue (torch.equal on logits, hidden, caches), one launch_qkv_tail per layer
     and step and no norm_row launch outside it, kv_path never re-writes the ring, the build-time self-check passes
     (and turns fused off with the reason when the tail is broken), kv_path refuses foreign ring arguments.
M2-L10-FIX (docs/HANDOFF-M2-L10-FIX.md): every kernel-body test runs both forms -- "stage" (l10_qkv_tail2: dynamic
CTA ids, per-tile fold into a bf16 stage buffer, one tail CTA per RP rows; the default) and "fold" (round 1) -- and
in shuffled program orders (a stage tail never waits on a later id: the shim fails a spin that is not released at
once); the fused launch takes gemv_pair's tuned config exactly (tile, KU, split-K, warps, stages); TAILS=False (the
bench's phase probe) writes no tail output and still leaves the tickets at 0.
Not here (box: tests/gpu/test_qkv_tail_gpu.py): Triton compilation, FMA contraction under the kernel's launch options,
the acq_rel tickets and the stage form's wait across real CTAs, tl.dot numerics, us.
"""
from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace

import pytest
import torch

import triton_shim
from runtime import fused_glue as FG
from runtime import fused_glue_r2 as R2
from runtime import gemv_mxfp8 as GM
from test_fused_glue_r2_cpu import _eq, _freqs, _od, tab  # noqa: F401 (fixture)
from test_gemv_cpu import _static_run as _static_run1, emu, gemv_emu  # noqa: F401 (fixtures)
from test_t9b4_cpu import _ops

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP8 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
BF = torch.bfloat16
VERBATIM = ("_t9b2_code", "_t9b2_pow2", "_t9b2_ipow2", "_t9b2_q8", "_t9b2_bf16", "_r2_offs", "_r2_col", "_r2_halve",
            "_r2_pairs", "_r2_level", "_r2_finish")


@pytest.fixture(autouse=True)
def _restore_dtype():
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(scope="module")
def KG():
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gemv_mxfp8.py"), "gemv_l10_shim")
    mod.KERNELS_ON_CPU = True
    return mod


@pytest.fixture(scope="module")
def KR(KG):
    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "fused_glue_r2.py"), "fg2_l10_shim")
    mod.KERNELS_ON_CPU = True
    mod.GEMV_MODULE = KG
    return mod


@pytest.fixture(scope="module")
def KS():
    return triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "static_kernels.py"), "sk_l10_shim")


def _n(fn):
    return len(fn.launches)


# ================================================================================ 1. module contract
def test_helpers_are_verbatim_copies(KG, KR):
    for name in VERBATIM:
        assert inspect.getsource(getattr(KG, name).fn) == inspect.getsource(getattr(KR, name).fn), name
    for name in ("_t9b2_mul_rn", "_t9b2_rope_re", "_t9b2_rope_im"):   # the GEMV module's own copies (T9b-2)
        assert inspect.getsource(getattr(KG, name).fn) == inspect.getsource(getattr(KR, name).fn), name


def test_fence_is_the_router_plan_fence(KG, KS):
    body = lambda f: " ".join(inspect.getsource(f.fn).split('"""')[-1].split())   # noqa: E731 (code after the doc)
    assert body(KG._l10_fence) == body(KS._m2_fence)
    for name in ("_m2_halve_rows", "_m2_pairs_rows", "_m2_level_rows", "_m2_finish_rows"):   # the rows tree
        norm = lambda f: " ".join(inspect.getsource(f.fn).split())   # noqa: E731 (indentation differs)
        assert norm(getattr(KG, name)) == norm(getattr(KS, name)), name


def test_tail_body_is_norm_row_with_two_explicit_products(KG, KR):
    """_l10_norm_rows's arithmetic == t9b2r2_norm_row's (QQ on) for every row at once: the same op forms with a
    leading row axis, except the loads (_l10_load, .cg) and the two products that feed an add (`v * v`,
    `sum * fac`), which are mul.rn -- the textual form of the bit-identity argument."""
    src = inspect.getsource(KG._l10_norm_pass.fn)
    assert "_l10_norm_pass(x, r0 + tl.arange(0, RP), M," in inspect.getsource(KG._l10_norm_rows.fn)
    ref = inspect.getsource(KR.t9b2r2_norm_row.fn)
    assert "acc = acc + _t9b2_mul_rn(v, v)" in src and "acc = acc + v * v" in ref
    assert "_l10_fin(_m2_finish_rows(acc, RP, T, V, DEC), fac, wf, DIV) + eps" in src
    assert "libdevice.rsqrt(_r2_fin(_r2_finish(acc, T, V, DEC), fac, wf, DIV) + eps)" in ref
    assert "_t9b2_mul_rn(ssum, fac)" in inspect.getsource(KG._l10_fin.fn)
    assert "ssum * fac" in inspect.getsource(KR._r2_fin.fn)
    for line in ("_t9b2_bf16(_t9b2_rope_re(a, b, c, d, RE_MODE)), a)",
                 "_t9b2_bf16(_t9b2_rope_im(a, b, c, d, IM_MODE)), b)",
                 "ce = bi[:, None] * 32 + 2 * tl.arange(0, 16)[None, :]", "e = _r2_offs(k, T, V, U)",
                 "(N - RD)) // 2, 0)", "qa = _t9b2_q8(a2, inv)", "qb = _t9b2_q8(b2, inv)",
                 '(qa.to(tl.float32) * sc).to(tl.bfloat16, fp_downcast_rounding="rtne")',
                 "a2.to(tl.bfloat16)", "code.to(tl.uint8)"):
        assert line in src and line in ref, line
    assert "_t9b2_code(tl.maximum(tl.max(tl.abs(a2), axis=2), tl.max(tl.abs(b2), axis=2)), tab)" in src
    assert "_t9b2_code(tl.maximum(tl.max(tl.abs(a2), axis=1), tl.max(tl.abs(b2), axis=1)), tab)" in ref
    for frm in ("tl.load(nw + ce, mask=", ".to(tl.float32)", "* (_l10_load(x, xo, m, NS, XS) * rb))"):
        assert frm in src, frm
    assert "* (_t9b4_ld(x, ce, m, NS, XS) * r))" in ref
    ld = inspect.getsource(KG._l10_load.fn)
    assert 'cache_modifier=".cg"' in ld and "acc = acc + tl.load(x + j * XS + e" in ld
    assert "acc = acc + tl.load(x + j * XS + e" in inspect.getsource(KR._t9b4_ld.fn)


def test_constants_and_switch(monkeypatch):
    assert R2.QKV_TICKETS == GM.L10_TICKET_WORDS == 256 and GM.L10_TICKETS == 2
    monkeypatch.delenv(GM.L10_FORM_ENV, raising=False)
    assert GM.qkv_tail_form() == "stage" == GM.L10_FORM_DEFAULT and GM.qkv_tail_form("fold") == "fold"
    monkeypatch.setenv(GM.L10_FORM_ENV, "fold")
    assert GM.qkv_tail_form() == "fold"
    monkeypatch.setenv(GM.L10_FORM_ENV, "per-row")
    with pytest.raises(ValueError):
        GM.qkv_tail_form()
    monkeypatch.delenv(GM.L10_FORM_ENV)
    monkeypatch.delenv(R2.QKV_TAIL_ENV, raising=False)
    # SMALL-WINS: the default is the row-conditional `auto` (fused at 1 row, split otherwise); test_small_wins_cpu.py
    assert R2.QKV_TAIL_DEFAULT == "auto" and R2.QKV_TAILS == ("split", "fused")
    assert R2.qkv_tail_mode(1) == "fused" and R2.qkv_tail_mode(6) == "split" and R2.qkv_tail_mode() == "split"
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "fused")
    assert R2.qkv_tail_mode() == "fused" == R2.qkv_tail_mode(6)
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "split")
    assert R2.qkv_tail_mode() == "split" == R2.qkv_tail_mode(1)
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "both")
    with pytest.raises(ValueError):
        R2.qkv_tail_mode()


def test_plan_metadata():
    pp = GM.pair_plan(GM.GemvConfig(32, 4, 4, 1, 3, 0), GM.GemvConfig(64, 2, 8, 3, 1, 0), 1, 96, 64, 320)
    tp = GM.qkv_tail_plan(pp, 6, 96, 64)
    assert tp["QNS"] == 3 and tp["KNS"] == 0 and tp["tickets"] == (3 * 3, 2 * 1) and tp["FT"] == 128
    assert tp["QBB"] == 4 and tp["KBB"] == 2 and tp["RP"] == GM.L10_TAIL_RP == 2      # the default rows per pass
    assert GM.qkv_tail_plan(pp, 6, 96, 64, rp=8)["RP"] == 8 and GM.qkv_tail_plan(pp, 6, 96, 64, rp=3)["RP"] == 4
    assert GM.qkv_tail_plan(pp, 1, 96, 64, rp=8)["RP"] == 1 and GM.qkv_tail_plan(pp, 3, 96, 64, rp=8)["RP"] == 4
    st = GM.qkv_tail_plan(pp, 6, 96, 64, form="stage")                       # M2-L10-FIX: one tail CTA per row
    assert st["RP"] == GM.L10_STAGE_RP == 1 and st["NTA"] == st["NTB"] == 6 and st["G0"] == 3 + 2   # TA 3, TB 2
    assert st["NWORK"] == 5 * 3 and st["grid"] == (5 * 3 + 12,) and st["ticket_words"] == 4 + 3 + 2
    st2 = GM.qkv_tail_plan(pp, 6, 96, 64, rp=4, form="stage")
    assert st2["RP"] == 4 and st2["NTA"] == 2 and st2["grid"] == (15 + 4,)


# ================================================================================ 2. kernel bodies (shim)
CASES = [  # (M, NA, NB, K, cfg_a, cfg_b, RD, modes, q order, kv order)
    (1, 128, 96, 320, GM.GemvConfig(32, 2, 4, 1, 3, 0), GM.GemvConfig(32, 4, 4, 1, 2, 0), 32, (1, 2),
     (("vec", 16, 4, True), "mul"), (("vec", 8, 4, True), "mul")),                    # both split, ragged
    (1, 96, 64, 256, GM.GemvConfig(32, 4, 4, 1, 1, 0), GM.GemvConfig(32, 2, 4, 1, 4, 0), 64, (0, 0),
     (("unroll", 8, 2, False), "div"), (("vec", 16, 1, False), "mul")),               # a unsplit, b EVEN split
    (3, 64, 128, 256, GM.GemvConfig(32, 4, 4, 1, 2, 1), GM.GemvConfig(64, 2, 4, 1, 1, 1), 32, (2, 1),
     (("vec", 4, 4, True), "mul"), (("unroll", 16, 2, True), "div")),                 # MMA, 3 rows, b unsplit
    (6, 128, 64, 320, GM.GemvConfig(32, 2, 4, 1, 5, 0), GM.GemvConfig(32, 4, 4, 1, 3, 0), 64, (1, 1),
     (("vec", 32, 4, True), "mul"), (("vec", 16, 4, True), "mul")),                   # 6 rows, both split
    (6, 96, 96, 256, GM.GemvConfig(32, 4, 4, 1, 1, 0), GM.GemvConfig(32, 4, 4, 1, 1, 0), 0, (0, 0),
     (("vec", 8, 4, False), "mul"), (("vec", 8, 4, False), "div")),                   # 6 rows, unsplit, no RoPE
]


def _tail_inputs(g, M, NA, NB, RD):
    qnw = (torch.randn(NA, generator=g, device="cpu") * 0.4 + 1.0).to(BF)
    kvnw = (torch.randn(NB, generator=g, device="cpu") * 0.4 + 1.0).to(BF)
    nslot = 16
    slot = torch.tensor([11, 2, 14, 3, 0, 7, 9, 5][:M], dtype=torch.long, device="cpu")
    fr = None
    if RD:
        fr = torch.stack([torch.view_as_real(_freqs(RD, 30 + 5 * t)).reshape(-1, 2) for t in range(M)]).contiguous()
        if M == 1:
            fr = fr[0].contiguous()
    return qnw, kvnw, nslot, slot, fr


def _split_path(KG, KR, tab, xq, xs, wa, sa, wb, sb, cfgs, qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot, tags):
    """Today's three launches: the pair (shim), q norm_row, kv norm_row (shim kernels)."""
    M, NA, NB = xq.size(0), wa.size(0), wb.size(0)
    ra, rb = KG.gemv_pair(xq, xs, wa, sa, wb, sb, tags=tags, configs=cfgs)
    ra = GM.SplitK(ra.ws.clone(), ra.nsplit) if hasattr(ra, "nsplit") else ra.clone()   # (the shared workspace)
    rb = GM.SplitK(rb.ws.clone(), rb.nsplit) if hasattr(rb, "nsplit") else rb.clone()
    qr, q = torch.empty(M, NA, dtype=BF, device="cpu"), torch.empty(M, NA, dtype=FP8, device="cpu")
    s = torch.empty(M, NA // 32, dtype=E8, device="cpu")
    KR.launch_norm_row(ra if hasattr(ra, "nsplit") else ra.view(M, NA), qnw, qr, eps=1e-6, ordd=qo, code_tab=tab,
                       q=q, s=s)
    ring = torch.full((1, nslot, NB), 3.0, dtype=BF, device="cpu")
    KR.launch_norm_row(rb if hasattr(rb, "nsplit") else rb.view(M, NB), kvnw, ring, eps=1e-6, ordd=ko, code_tab=tab,
                       slot=slot, qdq=True, rope=(fr, RD, modes) if RD else None)
    return dict(qr=qr, q=q, s=s, ring=ring), (ra, rb)


def _fused(KG, tab, xq, xs, wa, sa, wb, sb, cfgs, qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot, tags, ticket,
           rp=None, form=None, tails=True):
    M, NA, NB = xq.size(0), wa.size(0), wb.size(0)
    qr, q = torch.empty(M, NA, dtype=BF, device="cpu"), torch.empty(M, NA // 32 * 32, dtype=FP8, device="cpu")
    s = torch.empty(M, NA // 32, dtype=E8, device="cpu")
    ring = torch.full((1, nslot, NB), 3.0, dtype=BF, device="cpu")
    res = KG.qkv_pair_tail(
        xq, xs, wa, sa, wb, sb, ticket=ticket, tab=tab, tags=tags, configs=cfgs, rp=rp, form=form, tails=tails,
        q_tail=dict(nw=qnw, qr=qr, q=q, s=s, eps=1e-6, fac=R2.fac_of(NA), wf=float(NA), order=R2.order_args(qo)),
        kv_tail=dict(nw=kvnw, ring=ring, slot=slot, freqs=fr, FS=int(fr.stride(0)) if fr is not None and fr.dim() == 3
                     else 0, rd=RD, modes=modes, eps=1e-6, fac=R2.fac_of(NB), wf=float(NB), order=R2.order_args(ko)))
    return dict(qr=qr, q=q, s=s, ring=ring), res


def _kern_of(KG, form):
    return KG.l10_qkv_tail2 if form == "stage" else KG.l10_qkv_tail


def _ticket():
    return torch.zeros(GM.L10_TICKET_WORDS, dtype=torch.int32, device="cpu")


FORMS = ("stage", "fold")


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("rp", [None, 1, 8])   # fold: rows per pass (default 2); stage: rows per tail CTA (default 1)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_fused_launch_equals_pair_plus_two_norm_rows(KG, KR, tab, case, rp, form):
    M, NA, NB, K, ca, cb, RD, modes, qo_, ko_ = CASES[case]
    qo, ko = _od(*qo_), _od(*ko_)
    xq, xs, wa, sa = _ops(NA, K, M, 40 + case)
    _, _, wb, sb = _ops(NB, K, M, 50 + case)
    g = torch.Generator(device="cpu").manual_seed(60 + case)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(g, M, NA, NB, RD)
    tags = (f"l10a{case}", f"l10b{case}")
    want, (wa_res, wb_res) = _split_path(KG, KR, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD,
                                         modes, qo, ko, nslot, tags)
    ticket = _ticket()
    kern = _kern_of(KG, form)
    for rep in range(2):                                   # the second launch starts from the tickets the first left
        n0, n1, n2 = _n(kern), _n(KG.t9b4_mxfp8_gemv2), _n(_kern_of(KG, "fold" if form == "stage" else "stage"))
        got, res = _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot,
                          tags, ticket, rp, form)
        assert _n(kern) - n0 == 1 and _n(KG.t9b4_mxfp8_gemv2) == n1, "one launch, not the pair + tails"
        assert _n(_kern_of(KG, "fold" if form == "stage" else "stage")) == n2
        assert torch.equal(ticket, torch.zeros_like(ticket)), ticket
        for k in want:
            assert _eq(got[k], want[k]), (case, rep, k)
        for r, w in ((res[0], wa_res), (res[1], wb_res)):   # the GEMV results handed back (the fallback's inputs)
            assert hasattr(r, "nsplit") == hasattr(w, "nsplit")
            assert _eq(r.ws, w.ws) if hasattr(r, "nsplit") else _eq(r, w)
    pp = KG.pair_plan(KG.resolve_config(ca, NA, K), KG.resolve_config(cb, NB, K), M, NA, NB, K)
    ln = kern.launches[-1]
    assert ln["enable_fp_fusion"] is None                                    # the pair kernel's launch options
    if form == "fold":
        assert ln["grid"] == pp["grid"]
        assert ln["constexprs"]["QNS"] == (pp["NSA"] if pp["NSA"] > 1 else 0)
        assert ln["constexprs"]["KNS"] == (pp["NSB"] if pp["NSB"] > 1 else 0)
        assert ln["constexprs"]["RP"] == min(GM._np2(M), rp or GM.L10_TAIL_RP)
    else:
        rps = min(GM._np2(M), rp or GM.L10_STAGE_RP)
        nt = -(-M // rps)
        assert ln["grid"] == (pp["grid"][0] * pp["grid"][1] + 2 * nt,) and ln["constexprs"]["RP"] == rps
        assert ln["constexprs"]["QNS"] == max(1, pp["NSA"]) and ln["constexprs"]["KNS"] == max(1, pp["NSB"])
    assert (want["ring"][0, [r for r in range(nslot) if r not in slot.tolist()]].float() == 3.0).all()


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("case", [0, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tail_waits_for_its_half_in_any_program_order(KG, KR, tab, monkeypatch, seed, case, form):
    """The tails are gated by the tickets, not by grid order: the shim's programs run in a shuffled order (the CTA
    arrival order on the device is arbitrary) and every output still equals the split path; the tickets end at 0.
    Stage form: the work / tail roles come from the dynamic ids, so a tail never waits on a program that runs after
    it (the shim raises if one would)."""
    import itertools
    import random

    real = itertools.product

    def shuffled(*ranges):
        progs = list(real(*ranges))
        random.Random(seed).shuffle(progs)
        return iter(progs)
    monkeypatch.setattr(triton_shim.itertools, "product", shuffled)
    M, NA, NB, K, ca, cb, RD, modes, qo_, ko_ = CASES[case]
    qo, ko = _od(*qo_), _od(*ko_)
    xq, xs, wa, sa = _ops(NA, K, M, 90 + seed)
    _, _, wb, sb = _ops(NB, K, M, 95 + seed)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(torch.Generator(device="cpu").manual_seed(seed), M, NA, NB, RD)
    want, _ = _split_path(KG, KR, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, qo, ko,
                          nslot, ("sha", "shb"))
    ticket = _ticket()
    got, _ = _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot,
                    ("sha", "shb"), ticket, form=form)
    for k in want:
        assert _eq(got[k], want[k]), k
    assert torch.equal(ticket, torch.zeros_like(ticket))


@pytest.mark.parametrize("form", FORMS)
def test_rows_equal_one_row_launches(KG, KR, tab, form):
    """6 rows in one fused launch == each row alone in a 1-row fused launch (the row-invariance the 6-row step's
    8/8 rests on), at a config where both halves split."""
    M, NA, NB, K = 6, 128, 64, 320
    ca, cb = GM.GemvConfig(32, 2, 4, 1, 5, 0), GM.GemvConfig(32, 4, 4, 1, 3, 0)
    qo, ko = _od(("vec", 32, 4, True)), _od(("vec", 16, 4, True))
    xq, xs, wa, sa = _ops(NA, K, M, 7)
    _, _, wb, sb = _ops(NB, K, M, 8)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(torch.Generator(device="cpu").manual_seed(9), M, NA, NB, 64)
    ticket = _ticket()
    six, _ = _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, 64, (1, 2), qo, ko, nslot,
                    ("r6a", "r6b"), ticket, form=form)
    for t in range(M):
        one, _ = _fused(KG, tab, xq[t:t + 1].contiguous(), xs[t:t + 1].contiguous(), wa, sa, wb, sb, (ca, cb), qnw,
                        kvnw, slot[t:t + 1], fr[t].contiguous(), 64, (1, 2), qo, ko, nslot, ("r1a", "r1b"), ticket,
                        form=form)
        assert _eq(six["qr"][t], one["qr"][0]) and _eq(six["q"][t], one["q"][0]) and _eq(six["s"][t], one["s"][0])
        assert _eq(six["ring"][0, slot[t]], one["ring"][0, slot[t]])


def test_refuses_mixed_partial_forms(KG, tab):
    xq, xs, wa, sa = _ops(64, 256, 1, 1)
    _, _, wb, sb = _ops(64, 256, 1, 2)
    n0 = KG.L10_STATS["refused"]
    r = KG.qkv_pair_tail(xq, xs, wa, sa, wb, sb, ticket=torch.zeros(2, dtype=torch.int32, device="cpu"), tab=tab,
                         q_tail={}, kv_tail={}, configs=(GM.GemvConfig(32, 4, 4, 1, 2, 0),
                                                         GM.GemvConfig(32, 4, 4, 1, 2, 1)))
    assert r is None and KG.L10_STATS["refused"] == n0 + 1
    with pytest.raises(RuntimeError):                     # the real module on CPU: kernel path only
        GM.qkv_pair_tail(xq, xs, wa, sa, wb, sb, tab=tab, q_tail={},
                         ticket=torch.zeros(2, dtype=torch.int32, device="cpu"),
                         kv_tail={})


def _lin(w, s):
    w = w.clone()
    w.scale = s
    return SimpleNamespace(weight=w)


@pytest.mark.parametrize("M,RD", [(1, 32), (6, 64)])
def test_launch_wrapper_kernel_path_equals_cpu_mirror(KG, KR, tab, monkeypatch, M, RD):
    """fused_glue_r2.launch_qkv_tail: the kernel path (shim l10_qkv_tail) == the CPU mirror (gemv_pair + two
    launch_norm_row torch mirrors) at the same split partition (CPU_SPLIT_K = 3 on both modules)."""
    monkeypatch.setattr(GM, "CPU_SPLIT_K", 3)
    monkeypatch.setattr(KG, "CPU_SPLIT_K", 3)
    QL, HD, K = 128, 96, 320
    xq, xs, wa, sa = _ops(QL, K, M, 70 + M)
    _, _, wb, sb = _ops(HD, K, M, 80 + M)
    la, lb = _lin(wa, sa), _lin(wb, sb)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(torch.Generator(device="cpu").manual_seed(M), M, QL, HD, RD)
    qo, ko = _od(("vec", 16, 4, True)), _od(("unroll", 8, 4, False), "div")
    outs = []
    for mod, qord, kord in ((KR, qo, ko), (R2, qo, ko)):
        qr, q = torch.empty(1, M, QL, dtype=BF, device="cpu"), torch.empty(1, M, QL, dtype=FP8, device="cpu")
        s = torch.empty(1, M, QL // 32, dtype=E8, device="cpu")
        ring = torch.full((1, nslot, HD), 3.0, dtype=BF, device="cpu")
        n0 = _n(KG.l10_qkv_tail2)
        res = mod.launch_qkv_tail((xq.view(1, M, K), xs.view(1, M, K // 32)), la, lb, q_nw=qnw, q_eps=1e-6,
                                  q_ord=qord, qr=qr, q=q, s=s, kv_nw=kvnw, kv_eps=1e-6, kv_ord=kord, ring=ring,
                                  slot=slot, rope=(fr, RD, (2, 1)), code_tab=tab,
                                  ticket=torch.zeros(R2.QKV_TICKETS, dtype=torch.int32, device="cpu"),
                                  tags=(f"w{M}a", f"w{M}b"))
        assert res is not None and _n(KG.l10_qkv_tail2) - n0 == (1 if mod is KR else 0)      # the default form
        assert hasattr(res[0], "nsplit") and res[0].nsplit == 3
        outs.append((qr, q, s, ring))
    for a, b in zip(*outs):
        assert _eq(a, b)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("case", [0, 2, 3])
def test_fused_takes_gemv_pairs_config(KG, tab, case, form):
    """Cause (c) of the fix handoff: the fused launch's GEMV is gemv_pair's launch -- the same tile (BLOCK_M / N, KU),
    split-K partition, EVEN / MMA / SPLIT forms, num_warps and num_stages -- for the same configs (the tuned ones on
    the box: both resolve through gemv_mxfp8._resolved + pair_plan)."""
    M, NA, NB, K, ca, cb, RD, modes, qo_, ko_ = CASES[case]
    qo, ko = _od(*qo_), _od(*ko_)
    xq, xs, wa, sa = _ops(NA, K, M, 20 + case)
    _, _, wb, sb = _ops(NB, K, M, 25 + case)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(torch.Generator(device="cpu").manual_seed(case), M, NA, NB, RD)
    KG.gemv_pair(xq, xs, wa, sa, wb, sb, tags=("cfa", "cfb"), configs=(ca, cb))
    lp = KG.t9b4_mxfp8_gemv2.launches[-1]
    _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot, ("cfa", "cfb"),
           _ticket(), form=form)
    lf = _kern_of(KG, form).launches[-1]
    for k in ("BLOCK_M", "BLOCK_N", "KU", "EVENA", "EVENB", "MMA", "SPLITA", "SPLITB"):
        assert lf["constexprs"][k] == lp["constexprs"][k], k
    assert (lf["num_warps"], lf["num_stages"]) == (lp["num_warps"], lp["num_stages"]) == (ca.num_warps, ca.num_stages)
    if form == "fold":
        assert lf["grid"] == lp["grid"]
    else:
        assert lf["grid"][0] - lp["grid"][0] * lp["grid"][1] == 2 * M      # the pair's programs + 2 x M tails


@pytest.mark.parametrize("form", FORMS)
def test_no_tail_probe_writes_nothing(KG, KR, tab, form):
    """tails=False (bench/qkv_tail_bench.py's phase probe): the GEMV results are gemv_pair's, no tail output is
    touched, the tickets end at 0 (graph-replayable), and a real launch after it is still exact."""
    M, NA, NB, K, ca, cb, RD, modes, qo_, ko_ = CASES[3]
    qo, ko = _od(*qo_), _od(*ko_)
    xq, xs, wa, sa = _ops(NA, K, M, 33)
    _, _, wb, sb = _ops(NB, K, M, 34)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(torch.Generator(device="cpu").manual_seed(5), M, NA, NB, RD)
    want, (wa_res, wb_res) = _split_path(KG, KR, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD,
                                         modes, qo, ko, nslot, ("npa", "npb"))
    ticket = _ticket()
    M_, NA_, NB_ = xq.size(0), wa.size(0), wb.size(0)
    got, res = _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot,
                      ("npa", "npb"), ticket, form=form, tails=False)
    assert torch.equal(ticket, torch.zeros_like(ticket))
    assert (got["ring"].float() == 3.0).all()                                  # untouched
    assert _eq(res[0].ws, wa_res.ws) and _eq(res[1].ws, wb_res.ws)
    got2, _ = _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, qo, ko, nslot,
                     ("npa", "npb"), ticket, form=form)
    assert all(_eq(got2[k], want[k]) for k in want) and torch.equal(ticket, torch.zeros_like(ticket))


def test_shim_fails_an_unreleased_wait(KG, tab, monkeypatch):
    """The shim's spin guard (what makes the shuffled-order test meaningful for the stage form): a ticket that can
    never reach the target raises instead of hanging."""
    M, NA, NB, K, ca, cb, RD, modes, qo_, ko_ = CASES[3]
    xq, xs, wa, sa = _ops(NA, K, M, 1)
    _, _, wb, sb = _ops(NB, K, M, 2)
    qnw, kvnw, nslot, slot, fr = _tail_inputs(torch.Generator(device="cpu").manual_seed(1), M, NA, NB, RD)
    monkeypatch.setattr(triton_shim, "SPIN_LIMIT", 50)
    real = KG.pair_plan

    def one_more(*a, **k):                                # half b's tails wait for a tile that does not exist
        p = real(*a, **k)
        return dict(p, TB=p["TB"] + 1)
    monkeypatch.setattr(KG, "pair_plan", one_more)
    with pytest.raises(RuntimeError, match="never released"):
        _fused(KG, tab, xq, xs, wa, sa, wb, sb, (ca, cb), qnw, kvnw, slot, fr, RD, modes, _od(*qo_), _od(*ko_),
               nslot, ("spa", "spb"), _ticket(), form="stage")


@pytest.mark.parametrize("M,N,NS,BM,BN", [(6, 96, 5, 8, 32), (1, 64, 3, 1, 32), (3, 128, 4, 16, 64)])
def test_stage_fold_is_split_order_on_inexact_partials(KG, M, N, NS, BM, BN):
    """The stage form's per-tile fold (_l10_reduce_tile) == t9b_splitk_reduce's value (SplitK.reduce_torch: from
    +0.0, split 0, 1, ...; bf16 RNE) on partials whose sums are NOT exact in fp32 -- the GEMV tests' data cannot see
    the fold order (their sums are exact), this one can: the reversed order differs on this data."""
    @triton_shim.jit
    def fold(ws, st, M, N, NS: triton_shim.constexpr, BM: triton_shim.constexpr, BN: triton_shim.constexpr):
        KG._l10_reduce_tile(ws, st, M, N, triton_shim.program_id(0), NS, BM, BN)

    g = torch.Generator(device="cpu").manual_seed(M * 100 + N)
    ws = torch.randn(NS, M, N, generator=g, device="cpu") * torch.exp2(
        torch.randint(-12, 12, (NS, M, N), generator=g, device="cpu").float())
    big = torch.exp2(torch.full((M, N), 30.0, device="cpu")) * torch.randint(0, 2, (M, N), generator=g,
                                                                              device="cpu").float()
    ws[0] += big                                          # split 0 + big, split 1 - big: in split order they cancel
    ws[1] -= big                                          # first; reversed, the small splits are absorbed into them
    ws = ws.contiguous()
    st = torch.full((M, N), 7.0, dtype=BF, device="cpu")
    fold[(-(-N // BN),)](ws, st, M, N, NS=NS, BM=BM, BN=BN)
    want = GM.SplitK(ws, NS).reduce_torch()
    assert _eq(st, want)
    rev = GM.SplitK(ws.flip(0).contiguous(), NS).reduce_torch()
    assert not _eq(rev, want), "the data does not distinguish the fold order"


def test_bench_timing_bookkeeping_dry_run(KG, KR, tab, monkeypatch, capsys):
    """bench/qkv_tail_bench.py --timing's code end to end on CPU (shim kernels, tiny dims, one eager pass per phase,
    placeholder times): round 1's box run died in this bookkeeping (KeyError 'ms_saved_per_step_40_layers') after
    the 1-row block, so R = 6 was never timed. Every phase / variant / receipt key the handoff asks for is produced,
    every fused variant is bit-identical to split, and the summary line is printed."""
    monkeypatch.setattr(KG, "CPU_SPLIT_K", 3)
    sys.path.insert(0, os.path.join(REPO, "bench"))
    try:
        import qkv_tail_bench as QB
    finally:
        sys.path.pop(0)
    dry = {"GM": KG, "R2": KR, "dims": (256, 128, 96), "tab": tab}
    out = QB.timing([1, 6], 2, 1, (1, 1), (1, 2, 8), ("fold", "stage"), _dry=dry)
    assert out["dry_run"] is True
    for M in (1, 6):
        r = out[f"rows{M}"]
        for k in ("pair", "q_norm_row", "kv_norm_row", "split", "fold_notail", "stage_notail"):
            assert k in r["us_per_layer"], (M, k)
        rps = {1: [1], 6: [1, 2, 8]}[M]
        for form in ("fold", "stage"):
            for rp in rps:
                v = r["variants"][f"{form}_rp{rp}"]
                assert v["bit_identical"] and v["tickets_zero"], (M, form, rp, v)
        for k in ("phases_us", "best", "best_fused", "us_saved_per_layer", "ms_saved_per_step_40_layers",
                  "fused_beats_split", "bit_identical_all", "config", "pair_plan", "compiled"):
            assert k in r, (M, k)
        assert r["bit_identical_all"] and r["tickets_zero_all"]
    assert out["verdict"].startswith("R=1 split") and "R=6 split" in out["verdict"]
    assert "QKV-TAIL-TIMING: R=1" in capsys.readouterr().out


# ================================================================================ 3. StaticStep (emulated)
@pytest.fixture
def l10_emu(gemv_emu, monkeypatch):
    from test_dg_moe_cpu import _fake_deep_gemm

    monkeypatch.setitem(sys.modules, "deep_gemm", _fake_deep_gemm())
    monkeypatch.delenv(FG.DISABLE_ENV, raising=False)
    monkeypatch.setattr(GM, "CPU_SPLIT_K", 3)        # real split partials on CPU (the fold path)
    R2._ORDER_CACHE.clear()
    yield gemv_emu
    R2._ORDER_CACHE.clear()


def _counters(monkeypatch=None):
    """Counters of launch_qkv_tail and of the norm_row / gemv_pair calls made OUTSIDE it; patched directly on the
    modules -> (seen, restore)."""
    seen = {"tail": 0, "norm_row_outside": 0, "pair_outside": 0}
    real_t, real_n, real_p = R2.launch_qkv_tail, R2.launch_norm_row, GM.gemv_pair
    inside = [0]

    def tail(*a, **k):
        seen["tail"] += 1
        inside[0] += 1
        try:
            return real_t(*a, **k)
        finally:
            inside[0] -= 1

    def nr(*a, **k):
        seen["norm_row_outside"] += inside[0] == 0
        return real_n(*a, **k)

    def pr(*a, **k):
        seen["pair_outside"] += inside[0] == 0
        return real_p(*a, **k)
    R2.launch_qkv_tail, R2.launch_norm_row, GM.gemv_pair = tail, nr, pr

    def restore():
        R2.launch_qkv_tail, R2.launch_norm_row, GM.gemv_pair = real_t, real_n, real_p
    return seen, restore


def test_static1_fused_equals_split_and_ref(l10_emu, monkeypatch):
    from runtime.static_step import StaticStep

    model, pool, _ = l10_emu("mini4")
    ids = list(range(30, 47))
    kw = dict(max_pos=127, moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "fused")
    st = StaticStep(model, pool, glue_impl="fused", **kw)
    d = st.glue.describe()
    assert d["r2"]["qkv_tail"]["mode"] == "fused", d["r2"]["qkv_tail"]
    assert d["calibration"]["qkv_tail"].startswith("fused: equal"), d["calibration"]
    seen, restore = _counters()
    n_steps = 6
    try:
        got = _static_run1(model, st, ids, n_steps)
    finally:
        restore()
    L = len(model.layers)
    # one launch_qkv_tail per layer per step; the q / kv norm rows and the wq_a || wkv pair only inside it (the
    # shared expert's w1 || w3 pair stays: one per layer per step)
    assert seen == {"tail": L * n_steps, "norm_row_outside": 0, "pair_outside": L * n_steps}, seen
    assert st.glue.describe()["r2"]["qkv_tail"]["calls"]["fused"] >= L * n_steps
    st.detach()
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "split")
    split = StaticStep(model, pool, glue_impl="fused", **kw)
    assert split.glue.describe()["r2"]["qkv_tail"]["mode"] == "split"
    want = _static_run1(model, split, ids, n_steps)
    split.detach()
    monkeypatch.delenv(R2.QKV_TAIL_ENV)
    ref = StaticStep(model, pool, **kw)
    ref_out = _static_run1(model, ref, ids, n_steps)
    ref.detach()
    for a, b in ((got, want), (got, ref_out)):
        assert all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and a[1] == b[1]


def test_self_check_turns_a_broken_tail_off(l10_emu, monkeypatch):
    from runtime.static_step import StaticStep

    real = R2.launch_qkv_tail

    def bad(*a, **k):
        res = real(*a, **k)
        k["ring"].view(-1)[0:1].add_(1.0)                 # a stray write into the ring
        return res
    monkeypatch.setattr(R2, "launch_qkv_tail", bad)
    monkeypatch.setenv(R2.QKV_TAIL_ENV, "fused")
    model, pool, _ = l10_emu("mini4")
    st = StaticStep(model, pool, glue_impl="fused", max_pos=127, moe_impl="dg-masked", fp8_impl="gemv",
                    _allow_cpu=True)
    d = st.glue.describe()["r2"]["qkv_tail"]
    assert d["mode"] == "split" and d["why"].startswith("off: ['ring'] differ"), d
    st.detach()


def test_fused_needs_the_pair(l10_emu, monkeypatch):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(R2.QKV_TAIL_ENV, "fused")
    model, pool, _ = l10_emu("mini4")
    st = StaticStep(model, pool, glue_impl="fused", glue_opts={"disable": ["gemv_pair"]}, max_pos=127,
                    moe_impl="dg-masked", fp8_impl="gemv", _allow_cpu=True)
    d = st.glue.describe()["r2"]["qkv_tail"]
    assert d["mode"] == "split" and "gemv_pair" in d["why"], d
    st.detach()


def test_kv_path_refuses_other_ring_arguments(l10_emu, monkeypatch):
    from runtime.static_step import StaticStep

    monkeypatch.setenv(R2.QKV_TAIL_ENV, "fused")
    model, pool, _ = l10_emu("mini4")
    st = StaticStep(model, pool, glue_impl="fused", max_pos=127, moe_impl="dg-masked", fp8_impl="gemv",
                    _allow_cpu=True)
    glue = st.glue
    attn = model.layers[0].attn
    glue._kv_pend = (id(attn), "tail", object(), object(), object(), None)
    with pytest.raises(RuntimeError, match="M2-L10"):
        glue.kv_path(attn, None, None, torch.zeros(1, device="cpu"), torch.zeros(1, dtype=torch.long, device="cpu"),
                     ring=torch.zeros(1, device="cpu"))
    st.detach()


# 6 rows: the rows emulation (tests/test_static_rows_cpu.py fixtures)
from test_static_rows_cpu import (_assert_rows_equal, _cpu_default_device, _static_run, _tf_tokens,  # noqa: E402,F401
                                  rows_dg, rows_glue)


def test_static6_fused_equals_split(rows_glue, monkeypatch):
    from runtime.static_step import StaticStep

    monkeypatch.setattr(GM, "CPU_SPLIT_K", 3)
    model, pool, _, _ = rows_glue("mini4")
    hid = (len(model.layers) - 1,)
    ids, toks = _tf_tokens(18, 12, 51)
    kw = dict(max_pos=255, rows=6, hidden_layers=hid, moe_impl="dg-inplace", fp8_impl="gemv", glue_impl="fused",
              _allow_cpu=True)
    runs = {}
    for mode in ("fused", "split"):
        monkeypatch.setenv(R2.QKV_TAIL_ENV, mode)
        st = StaticStep(model, pool, **kw)
        d = st.glue.describe()
        assert d["r2"]["qkv_tail"]["mode"] == mode, d["r2"]["qkv_tail"]
        if mode == "fused":
            assert d["calibration"]["qkv_tail"].startswith("fused: equal"), d["calibration"]
            seen, restore = _counters()
            try:
                runs[mode] = _static_run(model, st, ids, toks, 2)
            finally:
                restore()
            L = len(model.layers)
            assert seen["tail"] == 2 * L and seen["norm_row_outside"] == 0, seen
        else:
            runs[mode] = _static_run(model, st, ids, toks, 2)
        st.detach()
    _assert_rows_equal(runs["fused"][0], runs["split"][0], "6-row qkv_tail fused vs split")
    assert all(torch.equal(x, y) for x, y in zip(runs["fused"][1], runs["split"][1]))
    for k_ in runs["split"][2]:
        assert torch.equal(runs["fused"][2][k_], runs["split"][2][k_]), k_
