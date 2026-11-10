"""P1 gracetier (b): `gt_stage` -- ONE launch per layer-call that copies the touched cold experts' weights from the
Grace pool into their fixed MegaMoE groups H + c (docs/P1-GRACETIER-DESIGN.md section 2.2; mini-test section 5 file 3).

    gt_stage(groups, H, src_tab, dtab, cnt)       grid (C slots, NPROG)

  groups   int64 [P] = the call's MegaMoE topk_idx rows (T x k, flattened; -1 = skip). Read on the device, so the
           launch is graph-capturable: which experts move is decided by the router each call, the kernel picks its
           sources from the ids (a copy-engine memcpy could not: runtime/staging_overlap.py:28-39).
  program (c, q): scans groups for H + c (blocks of PB ids); if absent it exits. Else it copies 16-B-aligned int64
           blocks q, q + NPROG, ... of the slot's L1 block (W1 words) from src_tab[c, 0] to dtab[0] + (H + c) * W1,
           and of the L2 block (W2 words) from src_tab[c, 1] to dtab[1] + (H + c) * W2. Weights only: the e8m0
           scales of every group are resident in HBM (section 2.2).
  cnt      int32 [C] (COUNT): program (c, 0) adds the number of picks of slot c in this call (integer atomics:
           order-independent; the residency counter of section 4).

Contract: `stage_emulated` (torch; the Mac tests run the kernel body under tests/triton_shim.py against it, the box
GPU test runs the compiled kernel against it): for every slot c with hits > 0, group H + c of L1 / L2 gets the
source bytes and cnt[c] += hits; nothing else is written. The weight bytes are the pool's (the checkpoint's bytes in
MegaMoE's transform), so a staged group is byte-identical to an all-HBM 384-group build's group: that is the whole
numerics argument of (b); M4 measures it.

A3 (docs/HANDOFF-A3-RESIDENCY.md, card GT3) -- the per-expert residency counter, optional:

    gt_stage(groups, H, src_tab, dtab, cnt, b1, b2, g2e=g2e, ecnt=ecnt_row, count_ids_max=N)

  g2e      int32 [G]: group -> expert id of this layer (the inverse of the router's row_map; fixed address, the
           residency manager rewrites two entries per swap between steps).
  ecnt     int32 [E]: this layer's row of the [L, E] counter (fixed address). Program (0, 0) of the launch does
           atomic_add(ecnt + g2e[group], 1) for every id >= 0 of the call while it scans them, hot or cold (integer
           atomics: order-independent). The count is per EXPERT, so it stays attributed across swaps.
  counting is on only when the call has P <= count_ids_max ids (decode-sized: 256 tokens x top-k). P is a host-side
           constant of the launch, so each captured graph size bakes its own choice: a counted launch runs
           gt_stage_count_kernel (gt_stage_kernel's body + the count), an uncounted one runs gt_stage_kernel itself.
  Absent (the default): gt_stage_kernel with exactly the arguments of the first build -- nothing else changes.
`stage_emulated` takes the same optional g2e / ecnt and stays the contract of both kernels.

`gt_read` (M7, R8): a plain read kernel over the pool's L1 blocks with NPROG programs, xor-folding the words into
out[NPROG, BW] so the loads are kept. `gt_touch`: an empty-ish launch (one store) for the fork/join cost.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

try:  # GPU image only; the Mac runs the contract (and tests/triton_shim.py for the kernel body)
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on the environment
    triton = tl = None
    HAVE_TRITON = False

NPROG_DEFAULT = 48
BLOCK_WORDS_DEFAULT = 1024      # 8 KiB per program per trip (static_kernels' staging block)
SCAN_BLOCK = 512                # ids scanned per trip (T <= 64 x k = 6 fits in one)
WARPS_DEFAULT = 4
ALIGN = 16
COUNT_TOKENS_MAX = 256          # A3: the per-expert counter counts decode-sized calls only (v26ar's PIN_COUNT_TMAX)


# ------------------------------------------------------------------------------------------------ pure planning
def stage_plan(groups: Sequence[int], H: int, C: int) -> Dict[int, int]:
    """{cold slot c: picks} for the groups of one call (only touched slots)."""
    out: Dict[int, int] = {}
    for g in groups:
        g = int(g)
        if H <= g < H + C:
            out[g - H] = out.get(g - H, 0) + 1
    return out


def words(nbytes: int) -> int:
    if nbytes % ALIGN:
        raise ValueError(f"{nbytes} B is not a multiple of {ALIGN}")
    return nbytes // 8


# ------------------------------------------------------------------------------------------------ contract
def count_ids(tokens_max: int = COUNT_TOKENS_MAX, topk: int = 6) -> int:
    """The largest id count P = T x top-k of a counted call (A3)."""
    return int(tokens_max) * int(topk)


def count_plan(groups: Sequence[int], g2e: Sequence[int]) -> Dict[int, int]:
    """{expert id: picks} of one counted call: every id >= 0 counts once for the expert that owns its group."""
    out: Dict[int, int] = {}
    for g in groups:
        g = int(g)
        if g >= 0:
            e = int(g2e[g])
            out[e] = out.get(e, 0) + 1
    return out


def stage_emulated(groups: torch.Tensor, H: int, src_tab: torch.Tensor, dsts: Sequence[torch.Tensor],
                   cnt: Optional[torch.Tensor], resolve: Callable[[int], torch.Tensor],
                   g2e: Optional[torch.Tensor] = None, ecnt: Optional[torch.Tensor] = None,
                   count_ids_max: Optional[int] = None) -> Dict[int, int]:
    """The contract of gt_stage in torch. dsts = [L1 [G, ...], L2 [G, ...]] (any dtype; written as bytes);
    src_tab int64 [C, 2]; resolve(address) -> uint8 tensor starting at that address. Returns the plan.
    A3, optional (old behaviour when absent): with g2e int32 [G] and ecnt int32 [E], every id >= 0 of the call adds 1
    to ecnt[g2e[id]] -- unless the call has more than count_ids_max ids (default count_ids(), as gt_stage), which
    counts nothing."""
    if (g2e is None) != (ecnt is None):
        raise ValueError("stage_emulated: g2e and ecnt go together")
    if ecnt is not None and groups.numel() <= (count_ids() if count_ids_max is None else int(count_ids_max)):
        for e, n in sorted(count_plan(groups.reshape(-1).tolist(), g2e.reshape(-1).tolist()).items()):
            ecnt[e] += n
    L1, L2 = dsts
    C = int(src_tab.shape[0])
    b1 = L1[0].numel() * L1.element_size()
    b2 = L2[0].numel() * L2.element_size()
    plan = stage_plan(groups.reshape(-1).tolist(), H, C)
    st = src_tab.cpu().tolist()
    for c, hits in sorted(plan.items()):
        g = H + c
        L1[g].view(torch.uint8).view(-1).copy_(resolve(int(st[c][0])).reshape(-1)[:b1].to(L1.device))
        L2[g].view(torch.uint8).view(-1).copy_(resolve(int(st[c][1])).reshape(-1)[:b2].to(L2.device))
        if cnt is not None:
            cnt[c] += hits
    return plan


def read_emulated(src_tab: torch.Tensor, n_src: int, W: int, nprog: int, bw: int,
                  resolve: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Contract of gt_read: out[q, j] = xor over blocks b = q, q+NPROG, ... < n_src * ceil(W/bw) of word j of block
    b (block b = source b // nbw, words (b % nbw) * bw .. ; masked words read as 0)."""
    nbw = -(-W // bw)
    out = torch.zeros(nprog, bw, dtype=torch.int64, device="cpu")
    st = src_tab.cpu().tolist()
    for q in range(nprog):
        b = q
        while b < n_src * nbw:
            s, j = divmod(b, nbw)
            w = resolve(int(st[s][0])).reshape(-1)[:W * 8].view(torch.int64)
            blk = torch.zeros(bw, dtype=torch.int64, device="cpu")
            seg = w[j * bw:min((j + 1) * bw, W)]
            blk[:seg.numel()] = seg
            out[q] ^= blk
            b += nprog
    return out


# ------------------------------------------------------------------------------------------------ kernels
if HAVE_TRITON:
    @triton.jit
    def gt_stage_kernel(groups, P, H, src_tab, dtab, cnt, W1, W2,
                        PB: tl.constexpr, BW: tl.constexpr, NPROG: tl.constexpr, COUNT: tl.constexpr):
        """Program (c, q): hits = #(groups == H + c); exit if 0; copy L1 / L2 blocks q, q + NPROG, ... (int64 words,
        16-B aligned) of slot c's source rows into group H + c; COUNT and q == 0: cnt[c] += hits."""
        c = tl.program_id(0)
        q = tl.program_id(1)
        want = H + c
        acc = tl.zeros([PB], dtype=tl.int32)
        for it in range(0, (P + PB - 1) // PB):
            offs = it * PB + tl.arange(0, PB)
            gv = tl.load(groups + offs, mask=offs < P, other=-1)
            acc += (gv == want).to(tl.int32)
        hits = tl.sum(acc, axis=0)
        if hits > 0:
            if COUNT:
                if q == 0:
                    tl.atomic_add(cnt + c, hits)
            grp = (hits * 0 + c + H).to(tl.int64)
            a_src = tl.multiple_of(tl.load(src_tab + 2 * c).to(tl.pointer_type(tl.int64)), 16)
            a_dst = tl.multiple_of(tl.load(dtab + 0).to(tl.pointer_type(tl.int64)), 16) + grp * W1
            nb1 = (W1 + BW - 1) // BW
            for ia in range(0, (nb1 + NPROG - 1) // NPROG):
                a_off = (q + ia * NPROG) * BW + tl.arange(0, BW)
                a_m = a_off < W1
                tl.store(a_dst + a_off, tl.load(a_src + a_off, mask=a_m), mask=a_m)
            b_src = tl.multiple_of(tl.load(src_tab + 2 * c + 1).to(tl.pointer_type(tl.int64)), 16)
            b_dst = tl.multiple_of(tl.load(dtab + 1).to(tl.pointer_type(tl.int64)), 16) + grp * W2
            nb2 = (W2 + BW - 1) // BW
            for ib in range(0, (nb2 + NPROG - 1) // NPROG):
                b_off = (q + ib * NPROG) * BW + tl.arange(0, BW)
                b_m = b_off < W2
                tl.store(b_dst + b_off, tl.load(b_src + b_off, mask=b_m), mask=b_m)

    @triton.jit
    def gt_stage_count_kernel(groups, P, H, src_tab, dtab, cnt, W1, W2, g2e, ecnt,
                              PB: tl.constexpr, BW: tl.constexpr, NPROG: tl.constexpr, COUNT: tl.constexpr):
        """A3: gt_stage_kernel plus the per-expert counter. Program (0, 0), while it scans the call's ids, adds 1 to
        ecnt[g2e[id]] for every id >= 0 (hot and cold picks alike; one atomic per lane, so two lanes that name one
        expert both count). Everything else is gt_stage_kernel's body, statement for statement."""
        c = tl.program_id(0)
        q = tl.program_id(1)
        want = H + c
        acc = tl.zeros([PB], dtype=tl.int32)
        for it in range(0, (P + PB - 1) // PB):
            offs = it * PB + tl.arange(0, PB)
            gv = tl.load(groups + offs, mask=offs < P, other=-1)
            acc += (gv == want).to(tl.int32)
            if c == 0:
                if q == 0:
                    e_m = gv >= 0
                    e_id = tl.load(g2e + gv, mask=e_m, other=0)
                    # relaxed: the count feeds only the policy, and it is read between steps in stream order; the
                    # default acq_rel fence cost v26ar 25-78 us/step against 2-6 relaxed (its measured overhead)
                    tl.atomic_add(ecnt + e_id, e_m.to(tl.int32), mask=e_m, sem="relaxed")
        hits = tl.sum(acc, axis=0)
        if hits > 0:
            if COUNT:
                if q == 0:
                    tl.atomic_add(cnt + c, hits)
            grp = (hits * 0 + c + H).to(tl.int64)
            a_src = tl.multiple_of(tl.load(src_tab + 2 * c).to(tl.pointer_type(tl.int64)), 16)
            a_dst = tl.multiple_of(tl.load(dtab + 0).to(tl.pointer_type(tl.int64)), 16) + grp * W1
            nb1 = (W1 + BW - 1) // BW
            for ia in range(0, (nb1 + NPROG - 1) // NPROG):
                a_off = (q + ia * NPROG) * BW + tl.arange(0, BW)
                a_m = a_off < W1
                tl.store(a_dst + a_off, tl.load(a_src + a_off, mask=a_m), mask=a_m)
            b_src = tl.multiple_of(tl.load(src_tab + 2 * c + 1).to(tl.pointer_type(tl.int64)), 16)
            b_dst = tl.multiple_of(tl.load(dtab + 1).to(tl.pointer_type(tl.int64)), 16) + grp * W2
            nb2 = (W2 + BW - 1) // BW
            for ib in range(0, (nb2 + NPROG - 1) // NPROG):
                b_off = (q + ib * NPROG) * BW + tl.arange(0, BW)
                b_m = b_off < W2
                tl.store(b_dst + b_off, tl.load(b_src + b_off, mask=b_m), mask=b_m)

    @triton.jit
    def gt_read_kernel(src_tab, n_src, W, out, BW: tl.constexpr, NPROG: tl.constexpr):
        """Program q xor-folds words of blocks b = q, q + NPROG, ... of the n_src L1 rows (W words each)."""
        q = tl.program_id(0)
        nbw = (W + BW - 1) // BW
        nb = n_src * nbw
        acc = tl.zeros([BW], dtype=tl.int64)
        for it in range(0, (nb + NPROG - 1) // NPROG):
            b = q + it * NPROG
            if b < nb:
                s = b // nbw
                j = b % nbw
                r_src = tl.multiple_of(tl.load(src_tab + 2 * s).to(tl.pointer_type(tl.int64)), 16)
                r_off = j * BW + tl.arange(0, BW)
                acc = acc ^ tl.load(r_src + r_off, mask=r_off < W, other=0)
        tl.store(out + q * BW + tl.arange(0, BW), acc)

    @triton.jit
    def gt_touch_kernel(out):
        tl.store(out + tl.program_id(0), 1)


# ------------------------------------------------------------------------------------------------ launchers
def make_dtab(l1: torch.Tensor, l2: torch.Tensor, device=None) -> torch.Tensor:
    if l1.data_ptr() % ALIGN or l2.data_ptr() % ALIGN:
        raise ValueError("gt_stage: destinations not 16-B aligned")
    t = torch.tensor([l1.data_ptr(), l2.data_ptr()], dtype=torch.int64, device="cpu")
    return t.to(device if device is not None else l1.device)


def gt_stage(groups: torch.Tensor, H: int, src_tab: torch.Tensor, dtab: torch.Tensor, cnt: Optional[torch.Tensor],
             b1: int, b2: int, *, nprog: Optional[int] = None, block_words: Optional[int] = None,
             num_warps: Optional[int] = None, g2e: Optional[torch.Tensor] = None,
             ecnt: Optional[torch.Tensor] = None, count_ids_max: Optional[int] = None) -> Tuple[int, int]:
    """ONE launch of gt_stage_kernel over the C = src_tab.shape[0] cold slots. groups: contiguous int64 (any shape;
    flattened). b1 / b2: bytes of one L1 / L2 group. Returns the grid.
    A3, optional: g2e int32 [G] + ecnt int32 [E] (one layer's counter row) -> a call with at most count_ids_max ids
    (default count_ids()) launches gt_stage_count_kernel instead, which also counts every pick per expert. A larger
    call, or no g2e / ecnt, launches gt_stage_kernel with the first build's arguments."""
    if not HAVE_TRITON:
        raise RuntimeError("gt_stage needs triton (box) -- the Mac runs stage_emulated")
    if groups.dtype != torch.int64 or not groups.is_contiguous():
        raise ValueError("gt_stage: groups must be contiguous int64")
    if src_tab.dtype != torch.int64 or tuple(src_tab.shape[1:]) != (2,) or not src_tab.is_contiguous():
        raise ValueError("gt_stage: src_tab must be contiguous int64 [C, 2]")
    if not src_tab.is_cuda and bool(src_tab.remainder(ALIGN).any()):     # (device tables: checked by the caller)
        raise ValueError("gt_stage: source addresses not 16-B aligned")
    W1, W2 = words(b1), words(b2)
    C = int(src_tab.shape[0])
    npg = int(nprog or NPROG_DEFAULT)
    bw = int(block_words or BLOCK_WORDS_DEFAULT)
    if bw < 2 or bw & (bw - 1):
        raise ValueError(f"block_words {bw} must be a power of two >= 2")
    count = cnt is not None
    if (g2e is None) != (ecnt is None):
        raise ValueError("gt_stage: g2e and ecnt go together")
    if ecnt is not None:
        for t, what in ((g2e, "g2e"), (ecnt, "ecnt")):
            if t.dtype != torch.int32 or t.dim() != 1 or not t.is_contiguous():
                raise ValueError(f"gt_stage: {what} must be contiguous int32 [n]")
        if C < 1:
            raise ValueError("gt_stage: the per-expert counter needs program (0, 0): C >= 1")
        if groups.numel() <= (count_ids() if count_ids_max is None else int(count_ids_max)):
            gt_stage_count_kernel[(C, npg)](groups, groups.numel(), int(H), src_tab, dtab, cnt if count else groups,
                                            W1, W2, g2e, ecnt, SCAN_BLOCK, bw, npg, count,
                                            num_warps=int(num_warps or WARPS_DEFAULT))
            return C, npg
    gt_stage_kernel[(C, npg)](groups, groups.numel(), int(H), src_tab, dtab, cnt if count else groups, W1, W2,
                              SCAN_BLOCK, bw, npg, count, num_warps=int(num_warps or WARPS_DEFAULT))
    return C, npg


def gt_read(src_tab: torch.Tensor, n_src: int, nbytes: int, out: torch.Tensor, *, nprog: int,
            block_words: int = BLOCK_WORDS_DEFAULT, num_warps: int = WARPS_DEFAULT) -> None:
    """ONE launch of gt_read_kernel: nprog programs read n_src rows of `nbytes` (src_tab[:, 0]). out: int64
    [nprog * block_words]."""
    if not HAVE_TRITON:
        raise RuntimeError("gt_read needs triton (box) -- the Mac runs read_emulated")
    if out.dtype != torch.int64 or out.numel() < nprog * block_words:
        raise ValueError(f"gt_read: out must be int64 with >= {nprog * block_words} elements")
    gt_read_kernel[(int(nprog),)](src_tab, int(n_src), words(nbytes), out, int(block_words), int(nprog),
                                  num_warps=int(num_warps))


def gt_touch(out: torch.Tensor) -> None:
    gt_touch_kernel[(1,)](out, num_warps=1)
