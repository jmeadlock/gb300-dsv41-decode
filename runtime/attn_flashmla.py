"""T9b-3c: the main sparse attention on FlashMLA's `sparse_prefill_fwd` over our bf16 caches (`--attn-impl flashmla`).

A3-class (not bit-identical to the reference TileLang `sparse_attn`, kernel.py:310-403): different tiling, exp2
softmax, sink applied as exp(lse) / (exp(lse) + exp(sink)). q stays bf16 -- the probe (bench/flashmla_probe.py, receipt
docs/flashmla-probe-20261106.json, arm C) found the output nearer the bf16-q fp64 oracle than the MXFP8-q one in 18/18
cases, with fp64 error <= 1.05x the reference kernel's own, 13.8-14.4 us per call in a 40-call graph at 1 and 6 rows
(reference 125-154 us). `ref` stays the default; this is opt-in.

Why arm C (prefill) and not B (decode over 528 / 288 B records): C reads our caches exactly as they are -- the static
step's `kv` is bf16 [1, n, 512] whose values ARE the reference's QDQ'd window / latent values (D1) -- so there is no new
cache writer, no record packing, no scheduler metadata, nothing to keep in sync with the reference's own cache writes.

Loading (`load_ops`): `<site-packages>/vllm/_flashmla_C*.so` is found by glob and loaded with
`torch.ops.load_library`; its NEEDED list is libtorch / CUDA / libc only (probe section 1). `vllm` must not be in
`sys.modules` before the load and must still not be after it -- if either fails the op is refused (never
`import vllm`; flash_mla_interface.py:9 does exactly that, so it is not used). Only `torch.ops._flashmla_C.
sparse_prefill_fwd` is called:

  sparse_prefill_fwd(q [s_q, h_q, d_qk] bf16, kv [s_kv, 1, d_qk] bf16, indices [s_q, 1, topk] int32 (-1 = skip),
                     sm_scale, d_v = 512, attn_sink [h_q] fp32 | None, topk_length [s_q] int32 | None, out | None)
      -> [out [s_q, h_q, d_v] bf16, max_logits, lse]            (flash_mla_interface.py:197-234; probe arm_C)

Mapping from the reference call `sparse_attn(q [b, m, h, d], kv [b, n, d], attn_sink [h], topk_idxs [b, m, topk],
softmax_scale) -> o [b, m, h, d] bf16` (`sparse_attn_flashmla`):
  * rows: s_q = b * m (the static step: b = 1, m = 1; M1: m = 6 rows of one sequence, one kv, each row its own index
    list -- the probe's b = 6 cases); b > 1 (distinct kv per batch) is flattened with a per-batch index offset;
  * heads: h_q must be 64 or 128 (vLLM pads to these, nvidia/flashmla.py:66-74); h < 64 is zero-padded (sink padded
    with 0) and the output narrowed back -- heads are independent, so the kept heads see the same inputs;
  * head dim: d_qk = d_v = 512 (V4.1). The real model has 512; a smaller d (mini checkpoints, 128) is zero-padded in
    q and kv: the extra products are exact zeros, so scores are unchanged, sm_scale is passed explicitly, and the
    padded output columns (zero V) are dropped;
  * topk: rounded up to a multiple of 128 with -1 (vLLM's `round_up(top_k + window, 128)`, flashmla.py:292); the real
    widths (128 window, 640 compressed) need no padding;
  * topk_length: on compressed layers the per-row count `win + min(k, compress_lens)` (`topk_length_static`, the same
    quantity T9b-3a's block count rounds up: `ceil(tlen / 64) == nblk_static`), computed on the device where the
    indices are built and published with them. Valid entries past it do not exist: the compressed part is a sorted
    prefix with a -1 tail (static_step #9), so the kernel's "only indices[:topk_length]" (flash_mla_interface.py:220)
    skips -1 slots only. Ratio-0 layers pass None (128 window slots, leading -1 at pos < 127, masked);
  * returns a contiguous [b, m, h, d] bf16 tensor, as kernel.py:398-403 does (the o path rotates it in place).

Graph capture: one kernel launch plus (only when padding is needed) pads/narrows; no host sync, no scheduler metadata;
the outputs come from the graph's private pool. The probe measured replay == replay and graph == eager.

T12-DIAG, row invariance and "num_splits": `sparse_prefill_fwd` takes NO num_splits / tile_scheduler_metadata. Those
belong to the decode entry `flash_mla_with_kvcache` -> `sparse_decode_fwd` / `dense_decode_fwd`
(flash_mla_interface.py:57-176: FlashMLASchedMeta.tile_scheduler_metadata / num_splits, split-KV whose partition
depends on the batch -- the M-dependent schedule research/round2-20261107/W4 4e warns about). The prefill kernel has
no split-KV combine: each query row's softmax / PV runs over that row's own index list in index order, so its
reduction order cannot depend on s_q (measured: row_census "sparse_attn FlashMLA m=6 vs m=1" equal on the real
checkpoint's h=64 d=512, first-light-rows6-*-20261107.json; self_check `rows_m_invariant` true). The closest thing to
"forcing the 1-row schedule" is to issue one s_q = 1 call per query row, which `$DSV41_FLASHMLA_ROW_SPLIT=1` (or
`row_split=True`) does: R launches instead of 1, outputs concatenated in row order -- a diagnostic switch
(bench/row_divergence_probe.py --flashmla-row-split); read at call time, i.e. at capture for a graph.

Routing: `StaticStep(attn_impl="flashmla")` calls `sparse_attn_flashmla` directly from its `_sparse_attn` (it owns the
call site), for every layer (ratio 0 and compressed), only inside its static step; the prefill and the eager reference
never reach it, so no monkeypatch on model.py is needed. A build-time `self_check` (eager, before any capture) runs it
against the reference `sparse_attn` and an fp64 oracle on the model's shapes; a failure (ops not loadable, vllm
pulled in, error > CHECK_MAX_RATIO x the reference's, non-finite) leaves the step on the reference and says why.

Mac: no .so; the pure functions run and the op can be replaced by a torch emulation (tests/test_attn_flashmla_cpu.py).
"""
from __future__ import annotations

import glob
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

ATTN_IMPL = "flashmla"
SITE_DIRS = ("/usr/local/lib/python3.12/dist-packages",)   # the library image (probe receipt load.so)
SO_GLOB = os.path.join("vllm", "_flashmla_C*.so")
OP_NS = "_flashmla_C"
OP = "sparse_prefill_fwd"
D_QK = 512                  # V4.1 sparse MLA: head_dim 512 (flash_mla_interface.py:111-112)
D_V = 512                   # "Can only be 512" (flash_mla_interface.py:215)
HQ = (64, 128)              # h_q the SM100 kernels take (nvidia/flashmla.py:66-74)
TOPK_ALIGN = 128            # nvidia/flashmla.py:292 round_up(top_k + window, 128)
CHECK_MAX_RATIO = 2.0       # build-time gross-error gate: fp64-oracle error <= 2x the reference kernel's (probe gate)
TOPK_LENGTH_ENV = "DSV41_FLASHMLA_TOPK_LENGTH"  # "0": pass no per-row length (the -1 slots are then masked instead)
ROW_SPLIT_ENV = "DSV41_FLASHMLA_ROW_SPLIT"      # "1": one s_q = 1 kernel call per query row (T12-DIAG, docstring)

_STATE: Dict[str, object] = {"ops": None, "so": None, "error": None}
_STATS = {"calls": 0, "padded": 0, "row_split_calls": 0}


def use_topk_length() -> bool:
    """Whether compressed-layer calls pass `topk_length` (default yes; `DSV41_FLASHMLA_TOPK_LENGTH=0` turns it off --
    a box knob in case the kernel mishandles lengths; the probe's arm C ran without them). Read at build/capture."""
    return os.environ.get(TOPK_LENGTH_ENV, "1") != "0"


def use_row_split() -> bool:
    """T12-DIAG: whether multi-row calls are issued one query row at a time ($DSV41_FLASHMLA_ROW_SPLIT=1; default
    off). Read at call time (= at capture for a graph)."""
    return os.environ.get(ROW_SPLIT_ENV, "0") == "1"


# ------------------------------------------------------------------------------------------------ loading
def vllm_modules() -> List[str]:
    return sorted(m for m in sys.modules if m == "vllm" or m.startswith("vllm."))


def find_so(roots: Optional[Sequence[str]] = None) -> List[str]:
    """Candidate `vllm/_flashmla_C*.so` paths under the image's dist-packages and every site/dist-packages on
    sys.path (no import, a glob)."""
    if roots is None:
        roots = list(SITE_DIRS) + [p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]
    out = []
    for r in roots:
        out += glob.glob(os.path.join(r, SO_GLOB))
    return sorted(set(out))


def load_ops(so: Optional[str] = None, roots: Optional[Sequence[str]] = None, _load=None):
    """torch.ops._flashmla_C with `sparse_prefill_fwd`, loaded once from the .so (cached). Raises RuntimeError if vllm
    is already imported, no .so is found, the load fails, the load pulled vllm Python in, or the op is missing.
    `_load` replaces torch.ops.load_library (tests)."""
    if _STATE["ops"] is not None:
        return _STATE["ops"]
    before = vllm_modules()
    if before:
        raise RuntimeError(f"vllm already imported before the FlashMLA load ({before[:5]}): refusing (decision 3)")
    if so is None:
        cands = find_so(roots)
        if not cands:
            raise RuntimeError(f"no {SO_GLOB} under {list(roots) if roots is not None else list(SITE_DIRS) + ['sys.path']}")
        so = cands[0]
    (_load or torch.ops.load_library)(so)
    after = vllm_modules()
    if after:
        raise RuntimeError(f"loading {so} imported vllm Python ({after[:5]}): refusing -- stop and report")
    ns = getattr(torch.ops, OP_NS)
    try:
        getattr(ns, OP)
    except (AttributeError, RuntimeError) as e:
        raise RuntimeError(f"{so} loaded but torch.ops.{OP_NS}.{OP} is missing: {e}") from e
    _STATE.update(ops=ns, so=so, error=None)
    return ns


def loaded() -> dict:
    return {"so": _STATE["so"], "loaded": _STATE["ops"] is not None, "vllm_modules": vllm_modules()}


def stats(reset: bool = False) -> dict:
    """Host-side call counts (eager calls and captures, not graph replays)."""
    out = dict(_STATS)
    if reset:
        for k in _STATS:
            _STATS[k] = 0
    return out


# ------------------------------------------------------------------------------------------------ pure shape/index math
def padded_heads(h: int) -> int:
    """h_q the kernel runs: 64 for h <= 64, 128 for h <= 128 (nvidia/flashmla.py:66-74)."""
    h = int(h)
    if h < 1 or h > HQ[-1]:
        raise ValueError(f"FlashMLA sparse attention takes at most {HQ[-1]} heads, got {h}")
    return HQ[0] if h <= HQ[0] else HQ[1]


def padded_topk(n: int, align: int = TOPK_ALIGN) -> int:
    return -(-int(n) // align) * align


def topk_length_static(clen: torch.Tensor, k: int, win: int) -> torch.Tensor:
    """Device per-row index count from the device compress_lens (`clen` int64 [rows] = (pos+1)//r): int32 [rows] =
    win + min(k, clen) -- the width the eager reference passes (model.py:579, 778-779). Two elementwise launches + a
    cast, no host sync. ceil(result / 64) == runtime.kernels.sparse_attn_bounded.nblk_static."""
    return (torch.clamp(clen, max=int(k)) + int(win)).int().reshape(-1)


def host_topk_length(pos: int, ratio: int, k: int, win: int = 128) -> Optional[int]:
    """Host mirror of `topk_length_static` (None at ratio 0: the window-only call passes no length)."""
    if not ratio:
        return None
    return int(win) + min(int(k), (int(pos) + 1) // int(ratio))


def reason(q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor, topk_length: Optional[torch.Tensor]) -> \
        Optional[str]:
    """None if (q, kv, topk_idxs, topk_length) can go to the kernel, else why not (metadata only)."""
    if q.dim() != 4 or q.dtype != torch.bfloat16:
        return "q"
    b, m, h, d = q.shape
    if h > HQ[-1] or d > D_QK:
        return "shape"
    if kv.dim() != 3 or kv.dtype != torch.bfloat16 or kv.size(0) != b or kv.size(-1) != d or kv.device != q.device:
        return "kv"
    if topk_idxs.dim() != 3 or topk_idxs.dtype != torch.int32 or tuple(topk_idxs.shape[:2]) != (b, m):
        return "idx"
    if topk_length is not None and (topk_length.dtype != torch.int32 or topk_length.numel() != b * m
                                    or topk_length.device != q.device):
        return "topk_length"
    return None


def prepare(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
            topk_length: Optional[torch.Tensor] = None):
    """The kernel's arguments for one reference-shaped call (module docstring): (q [s_q, hp, 512], kv [b*n, 1, 512],
    idx [s_q, 1, topk'] int32, sink [hp] fp32, tlen [s_q] int32 | None, (b, m, h, d), padded)."""
    b, m, h, d = q.shape
    n, topk = kv.size(1), topk_idxs.size(-1)
    hp = padded_heads(h)
    qf = q.reshape(b * m, h, d)
    if hp != h or d != D_QK:
        qf = F.pad(qf, (0, D_QK - d, 0, hp - h))
    kvf = kv if d == D_QK else F.pad(kv, (0, D_QK - d))
    kvf = kvf.reshape(b * n, 1, D_QK)
    idx = topk_idxs
    if b > 1:  # distinct kv per batch -> one flat kv with per-batch row offsets (-1 stays -1)
        off = (torch.arange(b, device=idx.device, dtype=idx.dtype) * n).view(b, 1, 1)
        idx = torch.where(idx >= 0, idx + off, idx)
    idx = idx.reshape(b * m, 1, topk)
    tk = padded_topk(topk)
    if tk != topk:
        idx = F.pad(idx, (0, tk - topk), value=-1)
    sink = attn_sink.detach().float()
    if hp != h:
        sink = F.pad(sink, (0, hp - h))
    tlen = None if topk_length is None else topk_length.reshape(-1)
    padded = hp != h or d != D_QK or tk != topk
    return qf.contiguous(), kvf.contiguous(), idx.contiguous(), sink.contiguous(), tlen, (b, m, h, d), padded


def sparse_attn_flashmla(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                         softmax_scale: float, topk_length: Optional[torch.Tensor] = None, ops=None,
                         row_split: Optional[bool] = None) -> torch.Tensor:
    """The reference `sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)` (kernel.py:392-403) on FlashMLA's
    sparse_prefill_fwd: same argument shapes and dtypes, returns a contiguous bf16 [b, m, h, d]. `topk_length`
    (int32, b*m entries) limits each row to its leading indices (all later ones must be -1). `ops` defaults to
    `load_ops()`. `row_split` (None = $DSV41_FLASHMLA_ROW_SPLIT): one s_q = 1 call per query row (T12-DIAG)."""
    why = reason(q, kv, topk_idxs, topk_length)
    if why is not None:
        raise ValueError(f"sparse_attn_flashmla: ineligible inputs ({why}): q {q.dtype} {tuple(q.shape)}, kv "
                         f"{kv.dtype} {tuple(kv.shape)}, idx {topk_idxs.dtype} {tuple(topk_idxs.shape)}")
    ops = ops if ops is not None else load_ops()
    qf, kvf, idx, sink, tlen, (b, m, h, d), padded = prepare(q, kv, attn_sink, topk_idxs, topk_length)
    _STATS["calls"] += 1
    _STATS["padded"] += int(padded)
    fn = getattr(ops, OP)
    split = use_row_split() if row_split is None else bool(row_split)
    if split and qf.size(0) > 1:
        # every slice is a contiguous leading-dim view; kv / sink are shared (the kernel only reads them)
        _STATS["row_split_calls"] += qf.size(0)
        o = torch.cat([fn(qf[i:i + 1], kvf, idx[i:i + 1], float(softmax_scale), D_V, sink,
                          None if tlen is None else tlen[i:i + 1], None)[0] for i in range(qf.size(0))], 0)
    else:
        o = fn(qf, kvf, idx, float(softmax_scale), D_V, sink, tlen, None)[0]
    if o.size(1) != h or o.size(-1) != d:
        o = o[:, :h, :d]
    return o.reshape(b, m, h, d).contiguous()


# ------------------------------------------------------------------------------------------------ oracle + self-check
def oracle64(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
             softmax_scale: float, topk_length: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fp64 softmax over each row's valid gathered rows with the sink in the denominator (kernel.py:363-385; the same
    math as FlashMLA's exp(lse)/(exp(lse)+exp(sink))) -> fp64 [b, m, h, d]. Index order does not matter here."""
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d, dtype=torch.float64, device=q.device)
    sink = attn_sink.detach().double()
    tl = None if topk_length is None else topk_length.reshape(b, m)
    for bi in range(b):
        kvb = kv[bi].double()
        for mi in range(m):
            ids = topk_idxs[bi, mi]
            if tl is not None:
                ids = ids[: int(tl[bi, mi])]
            sel = ids[ids >= 0].long()
            k = kvb[sel]
            s = (q[bi, mi].double() @ k.T) * float(softmax_scale)
            mx = torch.maximum(s.max(-1).values if sel.numel() else torch.full_like(sink, -math.inf), sink)
            e = torch.exp(s - mx.unsqueeze(-1))
            out[bi, mi] = (e @ k) / (e.sum(-1) + torch.exp(sink - mx)).unsqueeze(-1)
    return out


def _rel(o: torch.Tensor, ref64: torch.Tensor) -> float:
    n = ref64.norm().item()
    return (o.double() - ref64).norm().item() / n if n > 0 else 0.0


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16),
                                                                     b.contiguous().view(torch.int16))


def self_check(orig: Callable, device, heads: int, head_dim: int, softmax_scale: float, win: int, k: int,
               kv_len: int, attn_sink: Optional[torch.Tensor] = None, impl: Optional[Callable] = None,
               seed: int = 0, max_ratio: float = CHECK_MAX_RATIO) -> dict:
    """Build-time device check (eager, syncs; before any capture), on the model's shapes. Index lists in the static
    step's layout (runtime.kernels.sparse_attn_bounded.check_cases: window oldest-first with leading -1 at small pos,
    k compressed slots = c sorted positions + win, -1 tail), compressed calls with topk_length = win + c, plus the
    window-only (ratio-0) call per position with no length. Gates: finite; fp64-oracle error <= max_ratio x the
    reference kernel's on every case. Reported, not gated: max / mean ratio, mismatch % vs the reference, whether
    the length-limited call == the -1-masked call bitwise, whether 4 rows in one call == each row alone.
    -> {"ok", "reason", ...}."""
    from runtime.kernels import sparse_attn_bounded as SAB

    impl = impl or sparse_attn_flashmla
    dev = torch.device(device)
    q_all, kv, sink = SAB._check_inputs(heads, head_dim, kv_len, seed)       # CPU-built (@_cpu_default), then moved
    q_all, kv = q_all.to(dev), kv.to(dev)
    sink = attn_sink.detach().float().to(dev) if attn_sink is not None else sink.to(dev)
    cases = SAB.check_cases(win, k, kv_len, seed)
    rec = {"ok": False, "reason": None, "cases": 0, "heads": heads, "head_dim": head_dim, "win": win, "k": k,
           "kv_len": kv_len, "padded_heads": padded_heads(heads), "padded_head_dim": D_QK,
           "padded_topk": {"window": padded_topk(win), "compressed": padded_topk(win + k)},
           "topk_length": use_topk_length()}
    ratios, mism, tl_eq = [], [], True
    try:
        seen_pos = set()
        for ci, cs in enumerate(cases):
            q = q_all[:, ci % 4: ci % 4 + 1].contiguous()
            idx = cs["idx"].to(dev).view(1, 1, -1)
            tl0 = torch.tensor([win + cs["c"]], dtype=torch.int32, device=dev) if rec["topk_length"] else None
            calls: list = [(idx, tl0)]
            if cs["pos"] not in seen_pos:  # the ratio-0 call: window slots only, no length
                seen_pos.add(cs["pos"])
                calls.append((idx[..., :win].contiguous(), None))
            for ix, tl in calls:
                got = impl(q, kv, sink, ix, softmax_scale, tl)
                want = orig(q, kv, sink, ix, softmax_scale)
                o64 = oracle64(q, kv, sink, ix, softmax_scale)
                if not bool(torch.isfinite(got.float()).all()):
                    rec["reason"] = f"non-finite output: pos {cs['pos']} c {cs['c']} width {ix.size(-1)}"
                    return rec
                if got.shape != want.shape or got.dtype != want.dtype:
                    rec["reason"] = f"shape/dtype {got.dtype} {tuple(got.shape)} != reference {want.dtype} " \
                                    f"{tuple(want.shape)}"
                    return rec
                e_got, e_ref = _rel(got, o64), _rel(want, o64)
                ratios.append(e_got / max(e_ref, 1e-12))
                mism.append((got.view(torch.int16) != want.view(torch.int16)).float().mean().item() * 100.0)
                if tl is not None:
                    tl_eq = tl_eq and _bits_equal(got, impl(q, kv, sink, ix, softmax_scale, None))
                rec["cases"] += 1
        # four rows (different lengths) in one call vs each row alone
        pick = [cases[i] for i in (1, len(cases) // 3, len(cases) // 2, len(cases) - 1)]
        idx = torch.stack([c["idx"] for c in pick]).to(dev).view(1, len(pick), -1)
        tl = torch.tensor([win + c["c"] for c in pick], dtype=torch.int32, device=dev) if rec["topk_length"] else None
        got = impl(q_all, kv, sink, idx, softmax_scale, tl)
        rows_eq = all(_bits_equal(got[:, r:r + 1], impl(q_all[:, r:r + 1].contiguous(), kv, sink,
                                                        idx[:, r:r + 1].contiguous(), softmax_scale,
                                                        None if tl is None else tl[r:r + 1]))
                      for r in range(len(pick)))
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    except Exception as e:  # load / launch failure: record, the caller stays on the reference
        rec["reason"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        return rec
    rec.update(max_err_ratio=max(ratios), mean_err_ratio=sum(ratios) / len(ratios), max_mismatch_pct=max(mism),
               topk_length_eq_masked=tl_eq, rows_m_invariant=rows_eq)
    if max(ratios) > max_ratio:
        rec["reason"] = f"fp64-oracle error {max(ratios):.3f}x the reference kernel's (> {max_ratio})"
        return rec
    rec["ok"] = True
    return rec
