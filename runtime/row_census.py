"""M1-T10 V0: the row-invariance census (docs/M1-DSPARK-DESIGN.md 4, risk 2) and the row-invariant debug mode.

    from runtime.row_census import census, census_table, row_invariant_mode
    rows = census(model, rows=6)                 # one record per kernel family; on the box (the real kernels)
    for line in census_table(rows): print(line)

Question per family: is row r of the op at 6 rows bit-equal to the op at 1 row on row r alone? The answer decides
whether the 6-row verify graph can be gated bitwise against the 1-row graph (V0 strict) or only drift-graded. Each
family runs on the model's OWN layer-0 shapes and weights (layer 0's attention / MoE / mHC / head) with synthetic
activations (CPU-generated, moved to the device), at 6 rows vs 6 separate 1-row calls:

  act_quant (TileLang, per-32 blocks); fp8 Linear `fp8_gemm` (ref) on wq_a; the Triton fp8 GEMV on layer 0's fp8
  shapes, tuned per-M configs vs the M=1 config pinned at M=6 ($DSV41_GEMV_PIN_M1, with the graph-timed us per call
  of both); the routed expert's `fp4_gemm` (ref); DeepGEMM masked (masked_m 6 vs 1) when deep_gemm imports;
  `sparse_attn` (ref TileLang) and FlashMLA (when it loads) with per-row index lists; RMSNorm (torch); cuBLAS F.linear
  fp32 (mHC mixes, gate, head) and bf16 (indexer weights_proj); the wo_a / indexer einsums; hc_split_sinkhorn; topk.

Per family: equal (torch.equal on every row), mismatch % of elements, max ulp (bf16 / fp32 sign-magnitude), max |d|.
A family that cannot run here (no CUDA, no deep_gemm, no FlashMLA .so) is recorded as "n/a" with the reason.

`row_invariant_mode()`: torch's F.linear and the reference's "bs..." einsums evaluated one query row at a time (the
design's debug mode for the cuBLAS ops, which are expected not to be row-invariant). On the Mac emulations this mode
makes the 6-row step equal the 1-row step bit for bit (tests/test_static_rows_cpu.py); on the box the GPU test runs
the 6-row static step under it to see which non-cuBLAS kernels still break V0.
"""
from __future__ import annotations

import contextlib
import os
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

PIN_ENV = "DSV41_GEMV_PIN_M1"


# ------------------------------------------------------------------------------------------------ debug mode
@contextlib.contextmanager
def row_invariant_mode():
    """F.linear and torch.einsum(\"bs...\", a, ...) one row at a time while open (process-global; restored on exit)."""
    lin0, ein0 = F.linear, torch.einsum

    def linear(x, w, b=None):
        K = x.size(-1)
        M = x.numel() // K
        if M <= 1:
            return lin0(x, w, b)
        xv = x.reshape(M, K)
        return torch.cat([lin0(xv[i:i + 1], w, b) for i in range(M)], 0).view(*x.shape[:-1], -1)

    def einsum(eq, *ops):
        a = ops[0] if ops else None
        if a is not None and eq.replace(" ", "").startswith("bs") and a.dim() >= 3 and a.size(1) > 1:
            return torch.cat([ein0(eq, a[:, i:i + 1], *ops[1:]) for i in range(a.size(1))], 1)
        return ein0(eq, *ops)

    F.linear, torch.einsum = linear, einsum
    try:
        yield
    finally:
        F.linear, torch.einsum = lin0, ein0


# ------------------------------------------------------------------------------------------------ comparison
def _ord_key(t: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude float bits -> a monotone integer line (+0 == -0)."""
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        i = t.contiguous().view(torch.int16).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    if t.dtype == torch.float32:
        i = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    return t.contiguous().view(torch.uint8).to(torch.int64) if t.element_size() == 1 else t.to(torch.int64)


def compare(full: torch.Tensor, parts: List[torch.Tensor]) -> dict:
    """full [R, ...] vs parts[r] [1, ...] (row r at one row) -> {equal, mismatch_pct, max_ulp, max_abs}."""
    a = full.reshape(len(parts), -1)
    b = torch.cat([p.reshape(1, -1) for p in parts], 0)
    if a.dtype != b.dtype or a.shape != b.shape:
        return {"equal": False, "error": f"shape/dtype {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"}
    if a.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8) or a.dtype == getattr(torch, "float8_e8m0fnu", None):
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    eq_el = (a == b) if not a.is_floating_point() else (_ord_key(a) == _ord_key(b))
    out = {"equal": bool(eq_el.all()), "mismatch_pct": float((~eq_el).float().mean()) * 100.0}
    if a.is_floating_point():
        u = (_ord_key(a) - _ord_key(b)).abs()
        fin = torch.isfinite(a.float()) & torch.isfinite(b.float())
        out["max_ulp"] = int(u.max()) if u.numel() else 0
        d = (a.float() - b.float()).abs()[fin]
        out["max_abs"] = float(d.max()) if d.numel() else 0.0
    else:
        out["max_ulp"] = int((a.to(torch.int64) - b.to(torch.int64)).abs().max()) if a.numel() else 0
        out["max_abs"] = float(out["max_ulp"])
    return out


def _rand(g, shape, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=g, device="cpu") * scale).to(dtype)


# ------------------------------------------------------------------------------------------------ census
def census(model, rows: int = 6, seed: int = 0, gemv_timing: bool = True, log: Optional[Callable] = None) -> List[dict]:
    """One record per family (module docstring): {"family", "op", "shape", "equal" | None, "mismatch_pct",
    "max_ulp", "max_abs", "note"}. Eager; syncs. Runs every family it can on the model's device."""
    from runtime.static_step import _ref

    R = _ref()
    dev = model.embed.weight.device
    cuda = dev.type == "cuda"
    g = torch.Generator(device="cpu").manual_seed(20261107 + seed)
    out: List[dict] = []
    L0 = model.layers[0]
    attn, moe = L0.attn, L0.ffn
    D = model.embed.weight.size(1)

    def rec(family, op, shape, fn):
        r = {"family": family, "op": op, "shape": shape, "equal": None, "note": ""}
        try:
            with torch.inference_mode():
                res = fn()
            if isinstance(res, str):
                r["note"] = res
            else:
                cmp_, note = res if isinstance(res, tuple) else (res, "")
                r.update(cmp_)
                r["note"] = note
        except Exception as e:  # recorded, never fatal: the census reports what it could not run
            r["note"] = f"n/a: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        out.append(r)
        if log:
            log(_line(r))
        return r

    x6 = _rand(g, (1, rows, D)).to(dev)

    def per_row(fn, x):
        return [fn(x[:, i:i + 1]) for i in range(x.size(1))]

    # act_quant (per-32 blocks)
    def f_aq():
        q, s = R.act_quant(x6, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
        parts = per_row(lambda xi: R.act_quant(xi, R.fp8_block_size, R.scale_fmt, R.scale_dtype), x6)
        cq, cs = compare(q, [p[0] for p in parts]), compare(s, [p[1] for p in parts])
        cq["equal"] = cq["equal"] and cs["equal"]
        return cq, f"scales equal {cs['equal']}"
    rec("act_quant", "act_quant(x, 32, ue8m0)", f"[1,{rows},{D}]", f_aq)

    # fp8 Linear, reference fp8_gemm
    def fp8_ref(lin, x):
        q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
        return R.fp8_gemm(q, s, lin.weight, lin.weight.scale, R.scale_dtype, block_size=R.fp8_block_size)
    for nm in ("wq_a", "wkv"):
        lin = getattr(attn, nm)
        rec("fp8 Linear (ref fp8_gemm)", f"L0.attn.{nm}", f"{tuple(lin.weight.shape)} M={rows}",
            lambda lin=lin: compare(fp8_ref(lin, x6), per_row(lambda xi: fp8_ref(lin, xi), x6)))

    # fp8 GEMV (Triton): tuned per-M configs vs the M=1 config pinned at M=6
    gemv_lins = [("attn.wq_a", attn.wq_a), ("attn.wq_b", attn.wq_b), ("attn.wkv", attn.wkv), ("attn.wo_b", attn.wo_b),
                 ("shared.w1", moe.shared_experts.w1), ("shared.w2", moe.shared_experts.w2)]
    for nm, lin in gemv_lins:
        rec("fp8 GEMV tuned (M=6 cfg)", f"L0.{nm}", f"{tuple(lin.weight.shape)}",
            lambda lin=lin: _gemv_case(R, lin, rows, g, dev, pin=False, timing=gemv_timing))
        rec("fp8 GEMV M=1 cfg pinned", f"L0.{nm}", f"{tuple(lin.weight.shape)}",
            lambda lin=lin: _gemv_case(R, lin, rows, g, dev, pin=True, timing=gemv_timing))

    # routed expert fp4_gemm (ref): the reference evaluates an expert at M = count rows
    ex = next(e for e in moe.experts if e is not None)
    def f_fp4():
        xx = x6.view(rows, D)
        q, s = R.act_quant(xx, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
        w = ex.w1
        full = R.fp4_gemm(q, s, w.weight, w.weight.scale, R.scale_dtype, act_block_size=R.fp8_block_size)
        parts = [R.fp4_gemm(q[i:i + 1], s[i:i + 1], w.weight, w.weight.scale, R.scale_dtype,
                            act_block_size=R.fp8_block_size) for i in range(rows)]
        sub = R.fp4_gemm(q[:2], s[:2], w.weight, w.weight.scale, R.scale_dtype, act_block_size=R.fp8_block_size)
        c = compare(full, parts)
        return c, f"M=2 vs M=6 rows 0-1 equal {bool(torch.equal(sub, full[:2]))}"
    rec("fp4 expert GEMM (ref fp4_gemm)", "expert.w1", f"{tuple(ex.w1.weight.shape)} M={rows}", f_fp4)

    # DeepGEMM masked: masked_m = 6 vs 1
    rec("DG masked grouped GEMM", "masked_m 6 vs 1", f"w1 {tuple(ex.w1.weight.shape)}",
        lambda: _dg_case(R, ex, rows, g, dev) if cuda else "n/a: no CUDA")

    # sparse_attn: reference TileLang, FlashMLA
    rec("sparse_attn (ref TileLang)", "m=6 vs m=1", f"h={attn.n_local_heads} d={attn.head_dim}",
        lambda: _attn_case(R, attn, rows, g, dev, "ref"))
    rec("sparse_attn FlashMLA", "m=6 vs m=1", f"h={attn.n_local_heads} d={attn.head_dim}",
        lambda: _attn_case(R, attn, rows, g, dev, "flashmla") if cuda else "n/a: no CUDA")

    # torch row reductions
    rec("RMSNorm (torch)", "L0.attn_norm", f"[1,{rows},{D}]",
        lambda: compare(L0.attn_norm(x6), per_row(L0.attn_norm, x6)))
    # cuBLAS F.linear fp32: mHC mixes, gate, head
    hc = L0.hc_mult
    xh = _rand(g, (1, rows, hc * D), dtype=torch.float32).to(dev)
    rec("cuBLAS fp32 F.linear", "mHC mixes (hc_attn_fn)", f"[{rows},{hc * D}]x{tuple(L0.hc_attn_fn.shape)}",
        lambda: compare(F.linear(xh, L0.hc_attn_fn), per_row(lambda xi: F.linear(xi, L0.hc_attn_fn), xh)))
    xg = x6.float().view(rows, D)
    rec("cuBLAS fp32 F.linear", "gate (x.float() @ W.float())", f"[{rows},{D}]",
        lambda: compare(F.linear(xg, moe.gate.weight.float()),
                        [F.linear(xg[i:i + 1], moe.gate.weight.float()) for i in range(rows)]))
    rec("cuBLAS fp32 F.linear", "head", f"[{rows},{D}]x{tuple(model.head.weight.shape)}",
        lambda: compare(F.linear(xg, model.head.weight), [F.linear(xg[i:i + 1], model.head.weight)
                                                          for i in range(rows)]))
    ix = next((l.attn.indexer for l in model.layers if l.attn.indexer is not None), None)
    if ix is not None:
        rec("cuBLAS bf16 F.linear", "indexer weights_proj", f"[{rows},{D}]",
            lambda: compare(ix.weights_proj(x6), per_row(ix.weights_proj, x6)))
    # einsums
    G_, orank = attn.n_local_groups, attn.o_lora_rank
    wo_a = attn.wo_a.weight.view(G_, orank, -1)
    o6 = _rand(g, (1, rows, G_, wo_a.size(-1))).to(dev)
    rec("einsum bf16 (cuBLAS bmm)", "wo_a bsgd,grd->bsgr", f"[1,{rows},{G_},{wo_a.size(-1)}]",
        lambda: compare(torch.einsum("bsgd,grd->bsgr", o6, wo_a),
                        per_row(lambda oi: torch.einsum("bsgd,grd->bsgr", oi, wo_a), o6)))
    if ix is not None:
        Wt = 256
        q6 = _rand(g, (1, rows, ix.n_local_heads, ix.index_head_dim)).to(dev)
        kk = _rand(g, (1, Wt, ix.index_head_dim)).to(dev)
        rec("einsum bf16 (cuBLAS bmm)", "indexer bshd,btd->bsht", f"[1,{rows},{ix.n_local_heads},{ix.index_head_dim}]"
            f"x[1,{Wt},.]", lambda: compare(torch.einsum("bshd,btd->bsht", q6, kk),
                                            per_row(lambda qi: torch.einsum("bshd,btd->bsht", qi, kk), q6)))
    # hc_split_sinkhorn (TileLang)
    mix = (2 + hc) * hc
    m6 = _rand(g, (1, rows, mix), dtype=torch.float32).to(dev)
    def f_sk():
        full = R.hc_split_sinkhorn(m6, L0.hc_attn_scale, L0.hc_attn_base, hc, L0.hc_sinkhorn_iters, L0.hc_eps)
        parts = per_row(lambda mi: R.hc_split_sinkhorn(mi, L0.hc_attn_scale, L0.hc_attn_base, hc,
                                                        L0.hc_sinkhorn_iters, L0.hc_eps), m6)
        cs = [compare(full[t], [p[t] for p in parts]) for t in range(3)]
        c = dict(max(cs, key=lambda x: (not x["equal"], x.get("max_ulp", 0))))  # the worst of the three
        c["equal"] = all(x["equal"] for x in cs)
        return c, "pre/post/comb equal " + "/".join(str(x["equal"]) for x in cs)
    rec("hc_split_sinkhorn (TileLang)", "mixes -> pre/post/comb", f"[1,{rows},{mix}]", f_sk)
    # topk on [6, W] vs [1, W] (indexer / router selection)
    def f_topk():
        s = torch.randint(0, 50, (rows, 2048), generator=g, device="cpu").float().to(dev)  # heavy ties
        full = s.topk(64, dim=-1, sorted=False).indices.sort(dim=-1).values
        parts = [s[i:i + 1].topk(64, dim=-1, sorted=False).indices.sort(dim=-1).values for i in range(rows)]
        return compare(full, parts), "ties: index sets compared"
    rec("torch.topk", "k=64 of 2048, heavy ties", f"[{rows},2048]", f_topk)
    return out


def _gemv_case(R, lin, rows, g, dev, pin: bool, timing: bool):
    if dev.type != "cuda":
        return "n/a: the GEMV is a CUDA kernel"
    from runtime import gemv_mxfp8 as GV

    K = lin.weight.size(1)
    x = _rand(g, (rows, K)).to(dev)
    q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
    prev = os.environ.get(PIN_ENV)
    os.environ[PIN_ENV] = "1" if pin else "0"
    try:
        full = GV.linear_fp8(q, s, lin.weight, lin.weight.scale)
        parts = [GV.linear_fp8(q[i:i + 1], s[i:i + 1], lin.weight, lin.weight.scale) for i in range(rows)]
        N = lin.weight.size(0)
        sms = GV.num_sms(dev)
        c6, c1 = GV.get_config(N, K, rows, dev), GV.get_config(N, K, 1, dev)
        k6, k1 = GV.numerics_key(c6, N, K, sms), GV.numerics_key(c1, N, K, sms)
        note = f"numerics_key M6 {k6} M1 {k1}"
        us = None
        if timing:
            us = GV.time_graph([lambda: GV.linear_fp8(q, s, lin.weight, lin.weight.scale)] * 20, reps=5)
            note += f"; {us:.2f} us/call at M={rows}"
        res = compare(full, parts)
        res["us_m6"] = us
        return res, note
    finally:
        if prev is None:
            os.environ.pop(PIN_ENV, None)
        else:
            os.environ[PIN_ENV] = prev


def _dg_case(R, ex, rows, g, dev):
    from runtime.dg_moe import EXPECTED_M, MAX_M, RECIPE_A, RECIPE_B, e8m0_to_f32, import_deep_gemm, pick_e8m0_method

    dg = import_deep_gemm()
    fn = dg.m_grouped_fp8_fp4_gemm_nt_masked
    meth = pick_e8m0_method(dev)
    w = ex.w1.weight.view(torch.int8).unsqueeze(0).contiguous()
    N, K = w.size(1), w.size(2) * 2
    sf = torch.empty(1, N, K // 32, dtype=torch.float32, device=dev)
    e8m0_to_f32(ex.w1.scale.unsqueeze(0), sf, meth)
    x = _rand(g, (rows, K)).to(dev)
    q, s = R.act_quant(x, R.fp8_block_size, R.scale_fmt, R.scale_dtype)
    s32 = torch.empty(rows, K // 32, dtype=torch.float32, device=dev)
    e8m0_to_f32(s, s32, meth)

    def run(qq, ss, m):
        xq = torch.zeros(1, MAX_M, K, dtype=torch.float8_e4m3fn, device=dev)
        xs = torch.ones(1, MAX_M, K // 32, dtype=torch.float32, device=dev)
        xq[0, :m] = qq
        xs[0, :m] = ss
        d = torch.zeros(1, MAX_M, N, dtype=torch.bfloat16, device=dev)
        fn((xq, xs), (w, sf), d, torch.tensor([m], dtype=torch.int32, device=dev), EXPECTED_M,
           recipe_a=RECIPE_A, recipe_b=RECIPE_B)
        return d[0, :m]

    full = run(q, s32, rows)
    parts = [run(q[i:i + 1], s32[i:i + 1], 1) for i in range(rows)]
    return compare(full, parts), f"expected_m {EXPECTED_M}"


def _attn_case(R, attn, rows, g, dev, impl):
    from runtime.kernels import sparse_attn_bounded as SAB

    win = attn.window_size
    k = 128
    kv_len = 256 + 512
    h, d = attn.n_local_heads, attn.head_dim
    q = _rand(g, (1, rows, h, d), 0.5).to(dev)
    kv = _rand(g, (1, kv_len, d)).to(dev)
    cases = SAB.check_cases(win, k, kv_len, 3)
    pick = [cases[(i * 7 + 3) % len(cases)] for i in range(rows)]
    idx = torch.stack([c["idx"] for c in pick]).to(dev).view(1, rows, -1)
    if impl == "flashmla":
        from runtime import attn_flashmla as AFM

        AFM.load_ops()
        tl = torch.tensor([win + c["c"] for c in pick], dtype=torch.int32, device=dev)
        full = AFM.sparse_attn_flashmla(q, kv, attn.attn_sink, idx, attn.softmax_scale, tl)
        parts = [AFM.sparse_attn_flashmla(q[:, i:i + 1].contiguous(), kv, attn.attn_sink,
                                          idx[:, i:i + 1].contiguous(), attn.softmax_scale, tl[i:i + 1])
                 for i in range(rows)]
    else:
        from runtime.patches import original_sparse_attn

        sa = original_sparse_attn()
        full = sa(q, kv, attn.attn_sink, idx, attn.softmax_scale)
        parts = [sa(q[:, i:i + 1].contiguous(), kv, attn.attn_sink, idx[:, i:i + 1].contiguous(), attn.softmax_scale)
                 for i in range(rows)]
    return compare(full, parts), f"per-row lists (c = {[c['c'] for c in pick]})"


def _line(r: dict) -> str:
    eq = "n/a" if r.get("equal") is None else ("yes" if r["equal"] else "NO")
    mu = r.get("max_ulp")
    mp = r.get("mismatch_pct")
    mu_s = "-" if mu is None else str(mu)
    mp_s = "-" if mp is None else "%.3f" % mp
    return (f"{r['family'][:32]:32s} {r['op'][:30]:30s} {eq:>4s}  {mu_s:>8}  {mp_s:>8}  {r.get('note', '')[:70]}")


def census_table(rows: List[dict]) -> List[str]:
    head = f"{'family':32s} {'op':30s} {'eq':>4s}  {'max_ulp':>8}  {'mism%':>8}  note"
    out = [f"V0 ROW-INVARIANCE CENSUS (row r at 6 rows vs the same row alone; torch.equal / max ulp)", head]
    out += [_line(r) for r in rows]
    yes = sorted({r["family"] for r in rows if r.get("equal") is True})
    no = sorted({r["family"] for r in rows if r.get("equal") is False})
    out.append(f"row-invariant: {', '.join(yes) or '-'}")
    out.append(f"NOT row-invariant: {', '.join(no) or '-'}")
    return out
