"""A3 / card GT3: adaptive expert residency on gracetier's fixed slots -- the policy glue (docs/HANDOFF-A3-RESIDENCY.md).

gracetier's geometry does not change: G = 384 groups per layer, groups 0..H-1 private HBM, groups H..G-1 on the shared
scratch, Grace pool slot c <-> group H + c, SF banks resident for all G groups (runtime/gracetier_vmm.py,
runtime/gracetier_pool.py). What changes is WHICH expert sits in which group. A swap is a true exchange of a hot
expert e_out (group g < H) with a cold expert e_in (cold slot c), five steps enqueued on the model's own stream at a
barrier between two execute_model calls:

  1. D2D   HBM group g (L1, L2)          -> scratch group H + c   (the scratch is dead between steps: the staging row)
  2. H2D   Grace slot c (L1, L2)         -> HBM group g
  3. D2H   scratch group H + c (L1, L2)  -> Grace slot c
  4. SF    rows g and H + c exchanged in both SF banks (xor swap: three elementwise kernels per bank, no temporary)
  5. tables row_map[e_in] = g, row_map[e_out] = H + c, g2e[g] = e_in, g2e[H + c] = e_out

Every address is fixed (the source table never changes: slot c's Grace address is slot c's, only its contents change),
so nothing is re-captured, and stream order makes the swap atomic for every step: all five steps sit between the
kernels of one step and the kernels of the next. No extra host or HBM memory for the swap itself.

Copies. Steps 1-3 are `copy_` between contiguous byte views (cudaMemcpyAsync on the current stream). Two rules, both
from measurements on the GB300, not from documentation: a copy into or out of a cudaHostRegister'd buffer must lie
inside ONE registration chunk (runtime/host_tier.py:108-114; the pool keeps every expert inside one chunk,
runtime/gracetier_pool.py:13-14, and the wiring checks it); and a `copy_` whose destination is the UVA alias of a
registered host buffer writes the host bytes (runtime/residency.py `_copies` demotes experts this way in every box run
since 11/07). A copy across two VMM mappings of one reservation has NOT been measured (the mini-test's copies stay
inside one mapping), so LayerView takes the mapping boundaries (`cuts`) and no copy here crosses one.
Step 4 uses elementwise kernels on purpose: after Garner's a12_retire the SF banks live in a generic-compression
allocation, and its own loader forces SM stores after its copy ("a copy-engine transfer alone is insufficient
evidence that the compressed write path was populated", fastpath/native_scales.py:69-71).

Policy: runtime/residency_policy.py, the v26ar.2 decision core, vendored byte-identical and used unchanged. The mirror
is `residency_policy.State` with hot[l][r] = the expert in group r, cold[l][c] = the expert in cold slot c (group
H + c), so a policy item's (hot_row, cold_row) IS (g, c). Per barrier (= before every execute_model):
  * the drain: every `drain_n` barriers an async D2H of the device counter `cnt` int32 [L, E] (bumped per expert by
    gt_stage, runtime/gracetier_stage.py) into pinned memory; the next barrier that finds the copy complete polls the
    control file, runs `policy.drain` (EWMA, queue) and writes one telemetry line;
  * the barrier: `policy.take_barrier` / `is_stale` / `commit` in `apply_pending`'s order, actuating each swap.
Shipped v26 values (drain_n 64, swaps_per_drain 2, halflife 500, hyst 1.5, min_abs 8) and swaps_per_barrier 4.

Control (JSON file `<dir>/ctrl.json`, read at a drain when the file changed; write it by rename): `freeze`,
`baseline`, the policy knobs, `restore_map` (+ `restore_exact`, default true: the identical row table comes back, not
only the hot set), `dump` (tag -> `<dir>/dump-<tag>.{json,npz}`), `audit` ({"random": n per layer, "seed": s,
"moved": bool} or true), `seq` (echoed in the acknowledgement). `restore_map` first drops the swaps that are queued
but not applied (decisions only; nothing physical) so that no layer is skipped as busy. Telemetry `<dir>/
telemetry.jsonl`: one line per drain (host steps, cold share of the counted picks since the last drain, swaps
done / vetoed / stale, hot_moved, queue length), one per control document, one per audit.

Audit: at boot a digest of every expert at its home (four int64: L1 weights, L2 weights, L1 SF row, L2 SF row; each
the wrapping sum of the bytes' int64 words times fixed odd multipliers, so it is sensitive to position). `audit`
recomputes it for the experts that ever moved plus n random per layer, wherever they are now, and reports mismatches.
It is a device computation over fixed buffers (two word buffers of one L1 group, allocated at boot) and one sync.

Nothing here imports vllm; on the Mac every tensor is a CPU tensor and the copies are the same `copy_` calls
(tests/test_gracetier_res_cpu.py).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from runtime import residency_policy as P

MODES = ("adaptive", "off")
CTRL_KEYS = {"seq", "freeze", "baseline", "restore_map", "restore_exact", "dump", "audit", "count", "note"} | \
    {k for k, _, _ in P.CTRL_KNOBS}
DIGEST_SEED = 0x67743364
TELEMETRY_KEEP = 4096


# ------------------------------------------------------------------------------------------------ config
@dataclass
class ResConfig:
    """`gracetier-res.json`. Policy fields = residency_policy.Params at the shipped v26 values, swaps_per_barrier 4."""

    mode: str = "off"
    freeze: bool = True               # frozen at boot: count and drain, never queue (the card unfreezes by ctrl.json)
    drain_n: int = 64
    swaps_per_drain: int = 2
    halflife: float = 500.0
    hyst: float = 1.5
    min_abs: float = 8.0
    swaps_per_barrier: int = 4
    count_tokens_max: int = 256       # gt_stage counts calls with at most this many tokens (x top-k ids)
    dir: Optional[str] = "/prof/gracetier-res"   # ctrl.json / telemetry.jsonl / dumps; None = in memory only (tests)
    audit_random: int = 64            # `audit: true` checks the moved experts + this many random ones per layer
    seed: int = 0
    boot_digests: bool = True         # False: no boot digests, `audit` reports that it cannot run

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ResConfig":
        d = dict(d or {})
        if "halflife_steps" in d and "halflife" not in d:
            d["halflife"] = d.pop("halflife_steps")
        names = {f.name for f in fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"gracetier-res: unknown key(s) {bad}; known {sorted(names)}")
        c = cls(**d)
        if c.mode not in MODES:
            raise ValueError(f"gracetier-res: mode {c.mode!r} is not one of {MODES}")
        if int(c.drain_n) < 1 or int(c.swaps_per_barrier) < 0 or int(c.swaps_per_drain) < 0:
            raise ValueError(f"gracetier-res: drain_n {c.drain_n} / swaps_per_barrier {c.swaps_per_barrier} / "
                             f"swaps_per_drain {c.swaps_per_drain}")
        return c

    def params(self) -> P.Params:
        return P.Params(drain_n=int(self.drain_n), swaps_per_drain=int(self.swaps_per_drain),
                        halflife=float(self.halflife), hyst=float(self.hyst), min_abs=float(self.min_abs),
                        swaps_per_barrier=int(self.swaps_per_barrier), frozen_at_boot=bool(self.freeze))

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> Optional[ResConfig]:
    """The config at `path`, or None when the file is absent or says mode "off" (= the build without residency). A
    file that does not parse, or names an unknown key or mode, raises: a typo must not silently mean "off"."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        cfg = ResConfig.from_dict(json.load(f))
    return cfg if cfg.mode == "adaptive" else None


# ------------------------------------------------------------------------------------------------ tables (pure)
def g2e_list(row: Sequence[int]) -> List[int]:
    """Inverse of an all-resident row_map (expert -> group): group -> expert. Raises unless row is a permutation."""
    n = len(row)
    inv = [-1] * n
    for e, g in enumerate(row):
        g = int(g)
        if not 0 <= g < n or inv[g] != -1:
            raise ValueError("row_map is not a permutation of the groups")
        inv[g] = e
    return inv


def row_of(hot: Sequence[int], cold: Sequence[int]) -> List[int]:
    """The all-resident row_map of a (hot rows, cold slots) pair: hot row r -> r, cold slot c -> H + c."""
    H = len(hot)
    row = [-1] * (H + len(cold))
    for r, e in enumerate(hot):
        row[int(e)] = r
    for c, e in enumerate(cold):
        row[int(e)] = H + c
    if min(row) < 0:
        raise ValueError("hot + cold do not cover every expert")
    return row


def pieces(B: int, g: int, cuts: Sequence[int]) -> List[Tuple[int, int]]:
    """Group g's bytes [g*B, (g+1)*B) of a [G, B] tensor as (offset in the group, nbytes) runs, cut at every byte
    offset in `cuts` (offsets from the tensor's start: boundaries a single memcpy must not cross)."""
    lo, hi = g * B, (g + 1) * B
    pts = [lo] + sorted(int(b) for b in cuts if lo < int(b) < hi) + [hi]
    return [(a - lo, b - a) for a, b in zip(pts, pts[1:])]


def _bytes2d(t: torch.Tensor, what: str) -> torch.Tensor:
    """[n, B] uint8 view of a contiguous 1-byte [n, ...] tensor."""
    if t.element_size() != 1 or not t.is_contiguous() or t.dim() < 2:
        raise ValueError(f"{what}: need a contiguous 1-byte [n, ...] tensor, got {tuple(t.shape)} {t.dtype}")
    v = t.view(torch.uint8).view(t.shape[0], -1)
    if v.data_ptr() != t.data_ptr() or v.shape[1] % 16 or v.storage_offset() % 8:
        raise ValueError(f"{what}: byte view moved, a group is not a multiple of 16 B ({v.shape[1]}), or the tensor "
                         f"starts at an odd offset of its storage ({v.storage_offset()})")
    return v


def _sf2d(bank: torch.Tensor, what: str) -> torch.Tensor:
    """[G, n*q] int32 view of an MN-major SF bank (shape (G, n, q), strides (n*q, 1, n): a12_retire's layout,
    fastpath_a12_retire.py:71-77): row g = group g's n*q words, contiguous."""
    if bank.dtype != torch.int32 or bank.dim() != 3:
        raise ValueError(f"{what}: SF bank is {tuple(bank.shape)} {bank.dtype}, expected int32 (G, n, q)")
    G, n, q = bank.shape
    if tuple(bank.stride()) != (n * q, 1, n):
        raise ValueError(f"{what}: SF bank strides {tuple(bank.stride())} are not the MN-major ({n * q}, 1, {n})")
    t = bank.transpose(1, 2)
    if not t.is_contiguous() or (n * q) % 2 or bank.storage_offset() % 2:
        raise ValueError(f"{what}: SF bank rows are not contiguous int64-sized runs")
    v = t.view(G, n * q)
    if v.data_ptr() != bank.data_ptr():
        raise ValueError(f"{what}: SF row view moved")
    return v


class LayerView:
    """One gracetier layer's fixed-address state, as the manager swaps it. All tensors live where the step reads them
    (the box: device row_map / g2e, the VMM weight tensors, the SF banks MegaMoE reads, the pool's UVA alias; the Mac:
    CPU tensors).

      row_map  int32 [G]   expert -> group (the router's table)
      g2e      int32 [G]   group -> expert (gt_stage's counter index)
      l1, l2   1-byte [G, ...]  MegaMoE's weight tensors (groups H.. = the scratch)
      sf1, sf2 int32 (G, n, q) MN-major   the SF banks
      grace1, grace2  1-byte [C, ...]     the pool's L1 / L2 blocks of this layer (slot c = row c)
      cuts1, cuts2    byte offsets from the start of l1 / l2 that one memcpy must not cross (VMM mapping boundaries)
    """

    def __init__(self, idx: int, H: int, row_map: torch.Tensor, g2e: torch.Tensor, l1: torch.Tensor,
                 l2: torch.Tensor, sf1: torch.Tensor, sf2: torch.Tensor, grace1: torch.Tensor, grace2: torch.Tensor,
                 *, cuts1: Sequence[int] = (), cuts2: Sequence[int] = ()):
        self.idx, self.H = int(idx), int(H)
        w = f"gracetier-res layer {idx}"
        self.w = (_bytes2d(l1, f"{w} L1"), _bytes2d(l2, f"{w} L2"))
        self.gr = (_bytes2d(grace1, f"{w} Grace L1"), _bytes2d(grace2, f"{w} Grace L2"))
        self.sf = (_sf2d(sf1, f"{w} L1"), _sf2d(sf2, f"{w} L2"))
        self.G, self.C = int(self.w[0].shape[0]), int(self.gr[0].shape[0])
        self.B = (int(self.w[0].shape[1]), int(self.w[1].shape[1]))
        self.cuts = (tuple(int(b) for b in cuts1), tuple(int(b) for b in cuts2))
        self.row_map, self.g2e = row_map, g2e
        ok = (self.H + self.C == self.G and 0 < self.H < self.G
              and self.w[1].shape[0] == self.G and self.gr[1].shape[0] == self.C
              and self.gr[0].shape[1] == self.B[0] and self.gr[1].shape[1] == self.B[1]
              and self.sf[0].shape[0] == self.G and self.sf[1].shape[0] == self.G)
        if not ok:
            raise ValueError(f"{w}: geometry H {self.H} C {self.C} G {self.G}, group bytes {self.B}, Grace "
                             f"{tuple(self.gr[0].shape)} / {tuple(self.gr[1].shape)}, SF {tuple(self.sf[0].shape)} / "
                             f"{tuple(self.sf[1].shape)}")
        for t, name in ((row_map, "row_map"), (g2e, "g2e")):
            if t.dtype != torch.int32 or tuple(t.shape) != (self.G,) or not t.is_contiguous():
                raise ValueError(f"{w}: {name} must be contiguous int32 [{self.G}], got {tuple(t.shape)} {t.dtype}")

    def pieces(self, k: int, g: int) -> List[Tuple[int, int]]:
        return pieces(self.B[k], g, self.cuts[k])


# ------------------------------------------------------------------------------------------------ the manager
class GtResManager:
    """views {layer: LayerView}; hot / cold {layer: boot hot rows / boot cold slots} (the all-resident rowmap's
    lists); cnt int32 [L, E] = the device counter gt_stage bumps (row l = layer l)."""

    def __init__(self, views: Dict[int, LayerView], hot: Dict[int, Sequence[int]], cold: Dict[int, Sequence[int]],
                 cnt: torch.Tensor, cfg: Optional[ResConfig] = None, *, log: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or ResConfig(mode="adaptive")
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self.clock = clock
        self.views = {int(l): v for l, v in views.items()}
        if not self.views or set(self.views) != {int(l) for l in hot} or set(self.views) != {int(l) for l in cold}:
            raise ValueError("gracetier-res: views / hot / cold do not name the same layers")
        if cnt.dtype != torch.int32 or cnt.dim() != 2 or not cnt.is_contiguous():
            raise ValueError(f"gracetier-res: cnt must be contiguous int32 [L, E], got {tuple(cnt.shape)} {cnt.dtype}")
        self.cnt = cnt
        self.L, self.E = int(cnt.shape[0]), int(cnt.shape[1])
        self.dev = cnt.device
        self.cuda = self.dev.type == "cuda"
        if max(self.views) >= self.L or any(v.G != self.E for v in self.views.values()):
            raise ValueError(f"gracetier-res: cnt [{self.L}, {self.E}] does not cover layers {sorted(self.views)} "
                             f"with G {sorted({v.G for v in self.views.values()})}")
        self.params = self.cfg.params()
        rows = {l: {"hot": [int(e) for e in hot[l]], "cold": [int(e) for e in cold[l]]} for l in sorted(self.views)}
        self.state = P.State.from_rows(rows, self.E, self.L, self.params)
        self.state.freeze = bool(self.cfg.freeze)
        self.row: Dict[int, List[int]] = {}
        self.boot_row: Dict[int, List[int]] = {}
        for l, v in self.views.items():
            if len(rows[l]["hot"]) != v.H or len(rows[l]["cold"]) != v.C:
                raise ValueError(f"gracetier-res layer {l}: rowmap {len(rows[l]['hot'])} hot / "
                                 f"{len(rows[l]['cold'])} cold, view H {v.H} C {v.C}")
            want = row_of(rows[l]["hot"], rows[l]["cold"])
            # boot check (syncs; before serving): the device tables are the rowmap's -- the mirror starts from them
            if v.row_map.detach().cpu().tolist() != want:
                raise RuntimeError(f"gracetier-res layer {l}: the device row_map is not the all-resident rowmap")
            if v.g2e.detach().cpu().tolist() != g2e_list(want):
                raise RuntimeError(f"gracetier-res layer {l}: g2e is not the inverse of row_map")
            self.row[l], self.boot_row[l] = list(want), list(want)
        self.displaced = {l: 0 for l in self.views}            # experts not in their boot group, per layer
        self.ever_moved = {l: set() for l in self.views}
        with torch.inference_mode(False):
            self.cnt_host = torch.zeros(self.L, self.E, dtype=torch.int32, device="cpu", pin_memory=self.cuda)
        self.host_steps = 0
        self.stream: Optional[int] = None                      # the cuda_stream handle the barriers run on
        self._pending_read = None
        self._last_cur = np.zeros((self.L, self.E), dtype=np.int64)
        self._swaps_at_drain = 0
        self._ctrl_sig = None
        self._ctrl_seq = None
        self.stats = {"swaps": 0, "vetoed": 0, "drains": 0, "ctrl": 0, "audits": 0, "forced_swaps": 0,
                      "barriers_in_capture": 0, "restores": 0, "dropped_pending": 0, "copies": 0,
                      "stream_changes": 0, "telemetry_errors": 0}
        self.tele: deque = deque(maxlen=TELEMETRY_KEEP)
        self._tele_f = None
        self.dir = self.cfg.dir
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self._tele_f = open(os.path.join(self.dir, "telemetry.jsonl"), "a")
            self._ctrl_sig = self._sig(self.ctrl_path)         # a file left by an earlier boot is not a command
        # audit state: word buffers of the largest run (one L1 group), the boot digests
        self._words = max(max(v.B) for v in self.views.values()) // 8
        self.wts = self.tmp = self.boot_digest = self.boot_digest_dev = None
        self.digest_s = None
        if self.cfg.boot_digests:
            rng = np.random.default_rng(DIGEST_SEED)
            w = rng.integers(0, np.iinfo(np.int64).max, size=self._words, dtype=np.int64) | 1
            with torch.inference_mode(False):
                self.wts = torch.from_numpy(w).to(self.dev)
                self.tmp = torch.zeros(self._words, dtype=torch.int64, device=self.dev)
            t0 = self.clock()
            with torch.inference_mode():
                d = {l: self._digest_rows(l, range(self.E)) for l in sorted(self.views)}
                self.boot_digest_dev = d
                self.boot_digest = {l: t.detach().cpu() for l, t in d.items()}
            self.digest_s = self.clock() - t0
        P.baseline(self.state, np.zeros((self.L, self.E), dtype=np.int64), 0)
        self._emit({"event": "boot", "config": self.cfg.as_dict(), "params": self.params.as_dict(),
                    "layers": len(self.views), "H": sorted({v.H for v in self.views.values()}), "E": self.E,
                    "freeze": bool(self.state.freeze), "boot_digest_s": None if self.digest_s is None
                    else round(self.digest_s, 3), "audit_buffers_B": 0 if self.wts is None else 16 * self._words})
        self.log(self.describe_line())

    # ---------------------------------------------------------------- small helpers
    @property
    def ctrl_path(self) -> Optional[str]:
        return os.path.join(self.dir, "ctrl.json") if self.dir else None

    @staticmethod
    def _sig(path: Optional[str]):
        try:
            st = os.stat(path)
        except (OSError, TypeError):
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def _emit(self, rec: dict) -> dict:
        rec = {"t": round(self.clock(), 3), **rec}
        self.tele.append(rec)
        if self._tele_f is not None:
            try:
                self._tele_f.write(json.dumps(rec, sort_keys=True) + "\n")
                self._tele_f.flush()
            except OSError as e:       # a full disk must not stop serving: the records stay in memory (self.tele)
                self.stats["telemetry_errors"] += 1
                if self.stats["telemetry_errors"] == 1:
                    self.log(f"[gracetier-res] WARNING telemetry write failed ({e!r}); continuing without the file")
        return rec

    def at_boot_map(self) -> bool:
        return not any(self.displaced.values())

    def home(self, l: int, e: int) -> Tuple[str, int]:
        """("hbm", group) or ("grace", cold slot) of expert e now."""
        g = self.row[l][e]
        H = self.views[l].H
        return ("hbm", g) if g < H else ("grace", g - H)

    def weight_bytes(self, l: int, e: int, k: int) -> torch.Tensor:
        """uint8 [B] view of expert e's L1 (k = 0) / L2 (k = 1) weights at its current home."""
        v = self.views[l]
        where, i = self.home(l, e)
        return v.w[k][i] if where == "hbm" else v.gr[k][i]

    def sf_words(self, l: int, e: int, k: int) -> torch.Tensor:
        """int32 [n*q] view of expert e's SF row in bank k (the row of its current group)."""
        return self.views[l].sf[k][self.row[l][e]]

    def check_stream(self, what: str) -> None:
        """Work of the CURRENT execute_model call (an eager gt_stage, a CUDA graph replay) must be on the stream its
        barrier ran on: swaps are atomic for the step only because they are stream-ordered with it. The wiring calls
        this for the calls it can see; a different stream raises (docs/HANDOFF-A3-RESIDENCY.md: "a boot-time check
        that raises"). Before the first barrier there is nothing to compare with."""
        if not self.cuda or self.stream is None:
            return
        cur = int(torch.cuda.current_stream(self.dev).cuda_stream)
        if cur != self.stream:
            raise RuntimeError(f"gracetier-res: {what} runs on CUDA stream {cur:#x}, this step's barrier ran on "
                               f"{self.stream:#x}; swaps would not be stream-ordered with the step. Refusing.")

    def _enter_stream(self) -> None:
        """Record the barrier's stream. Two consecutive execute_model calls on different streams are not ordered with
        each other, so the first barrier on a new stream waits for the device before anything is enqueued."""
        cur = int(torch.cuda.current_stream(self.dev).cuda_stream)
        if self.stream is not None and cur != self.stream:
            torch.cuda.synchronize(self.dev)
            self.stats["stream_changes"] += 1
        self.stream = cur

    # ---------------------------------------------------------------- per barrier
    def barrier(self) -> int:
        """Call before every execute_model, on its thread and stream. Returns the swaps applied."""
        if self.cuda:
            if torch.cuda.is_current_stream_capturing():
                self.stats["barriers_in_capture"] += 1
                return 0
            self._enter_stream()
        self.host_steps += 1
        due = self.host_steps % max(1, int(self.params.drain_n)) == 0
        if self._pending_read is None and not self.state.pending and not due:
            return 0
        n = 0
        with torch.inference_mode():
            self._maybe_drain()
            if self.state.pending:
                n = self._apply()
            if due:
                self._read_counters()
        return n

    def _read_counters(self) -> None:
        if self._pending_read is not None:      # the previous read never completed in drain_n barriers: wait for it
            hs, ev = self._pending_read
            if ev is not None:
                ev.synchronize()
            self._drain_now(hs)
        if self.cuda:
            self.cnt_host.copy_(self.cnt, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev))
        else:
            self.cnt_host.copy_(self.cnt)
            ev = None
        self._pending_read = (self.host_steps, ev)

    def _maybe_drain(self) -> None:
        if self._pending_read is None:
            return
        hs, ev = self._pending_read
        if ev is None or ev.query():
            self._drain_now(hs)

    def _drain_now(self, hs: int) -> None:
        self._pending_read = None
        cur = self.cnt_host.numpy().astype(np.int64)
        self._last_cur = cur
        self._poll_ctrl(cur, hs)
        res = P.drain(self.state, self.params, cur, hs)
        self.stats["drains"] += 1
        done = self.stats["swaps"] - self._swaps_at_drain
        self._swaps_at_drain = self.stats["swaps"]
        self._emit({"event": "drain", "host_steps": hs, "steps": res.steps, "sel": res.sel, "cold": res.cold,
                    "cold_share": (res.cold / res.sel) if res.sel else None, "queued": res.queued,
                    "swaps": self.stats["swaps"], "swaps_since": done, "vetoed": self.stats["vetoed"],
                    "stale": self.state.stale, "hot_moved": P.hot_moved(self.state),
                    "pending": len(self.state.pending), "freeze": bool(self.state.freeze),
                    "at_boot": self.at_boot_map(), "drains": self.stats["drains"],
                    "stream_changes": self.stats["stream_changes"]})

    # ---------------------------------------------------------------- swaps
    def _apply(self) -> int:
        """policy.apply_pending's loop (take_barrier, layer check, is_stale, actuate, commit) with a veto."""
        n = 0
        for item in P.take_barrier(self.state, self.params):
            if item["layer"] not in self.state.hot:
                continue
            if P.is_stale(self.state, item):
                self.state.stale += 1
                continue
            if not self._actuate(item):
                self.stats["vetoed"] += 1
                continue
            P.commit(self.state, item)
            self._committed(item)
            n += 1
        return n

    def _actuate(self, item: dict) -> bool:
        l, g, c = int(item["layer"]), int(item["hot_row"]), int(item["cold_row"])
        e_in, e_out = int(item["in_id"]), int(item["out_id"])
        v = self.views.get(l)
        if v is None or not (0 <= g < v.H and 0 <= c < v.C):
            return False
        s = v.H + c
        if self.row[l][e_out] != g or self.row[l][e_in] != s:
            raise RuntimeError(f"gracetier-res layer {l}: mirror and tables disagree on swap (in {e_in}, out {e_out})")
        for k in (0, 1):
            w, gr = v.w[k], v.gr[k]
            ps, pg = v.pieces(k, s), v.pieces(k, g)
            for off, n in ps:                                               # 1. D2D  HBM g -> scratch H + c
                w[s, off:off + n].copy_(w[g, off:off + n], non_blocking=True)
            for off, n in pg:                                               # 2. H2D  Grace c -> HBM g
                w[g, off:off + n].copy_(gr[c, off:off + n], non_blocking=True)
            for off, n in ps:                                               # 3. D2H  scratch H + c -> Grace c
                gr[c, off:off + n].copy_(w[s, off:off + n], non_blocking=True)
            a, b = v.sf[k][g], v.sf[k][s]                                   # 4. SF rows g <-> H + c (xor swap)
            a.bitwise_xor_(b)
            b.bitwise_xor_(a)
            a.bitwise_xor_(b)
            self.stats["copies"] += 2 * len(ps) + len(pg)
        v.row_map[e_in].fill_(g)                                            # 5. tables
        v.row_map[e_out].fill_(s)
        v.g2e[g].fill_(e_in)
        v.g2e[s].fill_(e_out)
        return True

    def _committed(self, item: dict) -> None:
        l, g, c = int(item["layer"]), int(item["hot_row"]), int(item["cold_row"])
        s = self.views[l].H + c
        boot = self.boot_row[l]
        for e, new in ((int(item["in_id"]), g), (int(item["out_id"]), s)):
            self.displaced[l] += int(new != boot[e]) - int(self.row[l][e] != boot[e])
            self.row[l][e] = new
            self.ever_moved[l].add(e)
        self.stats["swaps"] += 1

    def force_swaps(self, swaps: Sequence[Tuple[int, int, int]]) -> int:
        """Queue explicit swaps (layer, in_id = a cold expert, out_id = a hot expert) on the policy's pending queue;
        they apply at the next barriers (stale ones are dropped there)."""
        n = 0
        for l, ein, eout in swaps:
            hot, cold = self.state.hot[l], self.state.cold[l]
            if ein not in cold or eout not in hot:
                raise ValueError(f"layer {l}: a forced swap needs a cold in_id and a hot out_id ({ein}, {eout})")
            self.state.pending.append({"layer": int(l), "in_id": int(ein), "out_id": int(eout),
                                       "hot_row": hot.index(eout), "cold_row": cold.index(ein), "ewma_in": -1.0,
                                       "ewma_out": -1.0, "ewma_coolest_hot": -1.0,
                                       "step": int(self.state.drain_steps), "forced": True})
            n += 1
        self.stats["forced_swaps"] += n
        return n

    def queue_random(self, k: int, rng) -> int:
        """k random (layer, cold, hot) swaps, no expert named twice among the queued ones (tests, the GPU gate)."""
        busy = {(p["layer"], p["in_id"]) for p in self.state.pending} | \
            {(p["layer"], p["out_id"]) for p in self.state.pending}
        layers = sorted(self.views)
        out = []
        for _ in range(int(k)):
            l = layers[int(rng.integers(len(layers)))]
            ein = int(rng.choice(self.state.cold[l]))
            eout = int(rng.choice(self.state.hot[l]))
            if (l, ein) in busy or (l, eout) in busy:
                continue
            busy |= {(l, ein), (l, eout)}
            out.append((l, ein, eout))
        return self.force_swaps(out)

    def flush(self) -> int:
        """Apply every queued swap now, ignoring the per-barrier budget (tests, the GPU gate; not the serving path)."""
        n = 0
        with torch.inference_mode():
            while self.state.pending:
                n += self._apply()
        return n

    # ---------------------------------------------------------------- control
    def _poll_ctrl(self, cur: np.ndarray, hs: int) -> None:
        path = self.ctrl_path
        sig = self._sig(path)
        if sig is None or sig == self._ctrl_sig:
            return
        self._ctrl_sig = sig
        try:
            with open(path) as f:
                doc = json.load(f)
            if not isinstance(doc, dict):
                raise ValueError("ctrl.json is not an object")
        except (OSError, ValueError) as e:
            self._emit({"event": "ctrl_error", "error": repr(e)[:300]})
            return
        if "seq" in doc and doc["seq"] == self._ctrl_seq:
            return
        self.apply_ctrl(doc, cur, hs)

    def apply_ctrl(self, doc: dict, cur: Optional[np.ndarray] = None, hs: Optional[int] = None) -> dict:
        """One control document, in the hook's order (freeze, knobs, baseline, restore_map, dump), then audit."""
        cur = self._last_cur if cur is None else cur
        hs = self.host_steps if hs is None else hs
        unknown = sorted(set(doc) - CTRL_KEYS)
        c = {k: v for k, v in doc.items() if k in CTRL_KEYS and k not in ("restore_map", "dump")}
        out = P.apply_ctrl(self.state, self.params, c, cur, hs)
        msg = list(out["msg"])
        if doc.get("baseline"):
            msg.append("baseline")
        if out["count"] is not None:
            msg.append("count ignored (the counter is part of gt_stage's launch)")
        rec = {"event": "ctrl", "seq": doc.get("seq"), "host_steps": hs, "unknown": unknown}
        if doc.get("restore_map"):
            dropped = len(self.state.pending)
            self.state.pending = []
            exact = bool(doc.get("restore_exact", True))
            queued = P.queue_restore(self.state, exact=exact)
            if self.params.restore_resets_ewma:
                self.state.ewma[:] = 0.0
            self.stats["restores"] += 1
            self.stats["dropped_pending"] += dropped
            msg.append(f"restore_map queued={queued} exact={int(exact)} dropped_pending={dropped}")
            rec.update({"restore_queued": queued, "restore_exact": exact, "dropped_pending": dropped})
        if doc.get("dump"):
            rec["dump"] = self.dump(str(doc["dump"]), cur)
            msg.append(f"dump {rec['dump']}")
        if doc.get("audit"):
            a = doc["audit"] if isinstance(doc["audit"], dict) else {}
            rec["audit"] = self.audit(random_per_layer=a.get("random"), seed=a.get("seed"),
                                      moved=bool(a.get("moved", True)), seq=doc.get("seq"))
            msg.append(f"audit mismatches={rec['audit'].get('mismatches')}")
        self._ctrl_seq = doc.get("seq", self._ctrl_seq)
        self.stats["ctrl"] += 1
        rec.update({"msg": msg, "freeze": bool(self.state.freeze), "pending": len(self.state.pending),
                    "params": self.params.as_dict(), "at_boot": self.at_boot_map(),
                    "hot_moved": P.hot_moved(self.state), "swaps": self.stats["swaps"]})
        return self._emit(rec)

    def dump(self, tag: str, cur: Optional[np.ndarray] = None) -> Optional[str]:
        """The current map (hot rows, cold slots, row_map per layer) as JSON and the last drained cumulative counter +
        EWMA as npz, under <dir>/dump-<tag>.*. Returns the JSON path (None without a directory)."""
        if not self.dir:
            return None
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:64] or "dump"
        base = os.path.join(self.dir, f"dump-{tag}")
        doc = {"tag": tag, "t": round(self.clock(), 3), "host_steps": self.host_steps, "E": self.E,
               "swaps": self.stats["swaps"], "hot_moved": P.hot_moved(self.state), "at_boot": self.at_boot_map(),
               "layers": {str(l): {"hot": list(self.state.hot[l]), "cold": list(self.state.cold[l]),
                                   "row": list(self.row[l])} for l in sorted(self.views)}}
        with open(base + ".json.tmp", "w") as f:
            json.dump(doc, f)
        os.replace(base + ".json.tmp", base + ".json")
        np.savez(base + ".npz", counts=self._last_cur if cur is None else cur, ewma=self.state.ewma)
        return base + ".json"

    # ---------------------------------------------------------------- audit
    def _digest1(self, words: torch.Tensor) -> torch.Tensor:
        """0-d int64: wrapping sum of words[i] * wts[i] (odd multipliers: any one changed word changes the sum, and
        two exchanged words do unless their difference cancels mod 2^64)."""
        n = words.numel()
        torch.mul(words, self.wts[:n], out=self.tmp[:n])
        return self.tmp[:n].sum()

    def _digest_rows(self, l: int, experts) -> torch.Tensor:
        """int64 [n, 4] on the manager's device: (L1 weights, L2 weights, L1 SF row, L2 SF row) of each expert at its
        current home."""
        out = []
        for e in experts:
            for k in (0, 1):
                out.append(self._digest1(self.weight_bytes(l, e, k).view(torch.int64)))
            for k in (0, 1):
                out.append(self._digest1(self.sf_words(l, e, k).view(torch.int64)))
        if not out:
            return torch.zeros(0, 4, dtype=torch.int64, device=self.dev)
        return torch.stack(out).view(-1, 4)

    def audit(self, random_per_layer: Optional[int] = None, seed: Optional[int] = None, moved: bool = True,
              experts: Optional[Dict[int, Sequence[int]]] = None, seq=None) -> dict:
        """Digest `experts` ({layer: ids}; default: every expert that ever moved + `random_per_layer` random ones per
        layer) at their current homes and compare with the boot digests. One sync. Emits and returns the record."""
        if self.boot_digest is None:
            return self._emit({"event": "audit", "seq": seq, "error": "no boot digests (boot_digests is false)",
                               "mismatches": None})
        t0 = self.clock()
        n_rand = int(self.cfg.audit_random if random_per_layer is None else random_per_layer)
        rng = np.random.default_rng(int(self.cfg.seed if seed is None else seed) + self.stats["audits"])
        picked, n_moved, n_random = {}, 0, 0
        for l in sorted(self.views):
            if experts is not None:
                ids = sorted({int(e) for e in experts.get(l, ())})
            else:
                mv = set(self.ever_moved[l]) if moved else set()
                rest = sorted(set(range(self.E)) - mv)
                k = min(n_rand, len(rest))
                rnd = set(int(x) for x in rng.choice(rest, size=k, replace=False)) if k else set()
                n_moved += len(mv)
                n_random += len(rnd)
                ids = sorted(mv | rnd)
            picked[l] = ids
        bad = []
        with torch.inference_mode():
            diffs = {}
            for l, ids in picked.items():
                if ids:
                    ix = torch.tensor(ids, dtype=torch.long, device="cpu").to(self.dev)
                    diffs[l] = self._digest_rows(l, ids) != self.boot_digest_dev[l][ix]
            for l, d in diffs.items():                                      # the one sync
                d = d.cpu()
                for i in d.any(1).nonzero().flatten().tolist():
                    e = picked[l][i]
                    bad.append({"layer": l, "expert": e, "home": list(self.home(l, e)),
                                "parts": [p for p, x in zip(("l1", "l2", "sf1", "sf2"), d[i].tolist()) if x]})
        self.stats["audits"] += 1
        checked = sum(len(v) for v in picked.values())
        return self._emit({"event": "audit", "seq": seq, "checked": checked, "moved": n_moved, "random": n_random,
                           "mismatches": len(bad), "first": bad[:16], "ms": round(1000 * (self.clock() - t0), 1),
                           "at_boot": self.at_boot_map(), "swaps": self.stats["swaps"]})

    # ---------------------------------------------------------------- report
    def describe_line(self) -> str:
        p = self.params
        return (f"[gracetier-res] adaptive residency on {len(self.views)} layers (H "
                f"{sorted({v.H for v in self.views.values()})}, G {self.E}): drain_n {p.drain_n} swaps/drain "
                f"{p.swaps_per_drain} cap/barrier {p.swaps_per_barrier} hyst {p.hyst} min_abs {p.min_abs} halflife "
                f"{p.halflife}; freeze={int(self.state.freeze)}; ctrl + telemetry {self.dir}; boot digests "
                f"{'off' if self.digest_s is None else f'{self.digest_s:.2f} s'}")

    def describe(self) -> dict:
        return {"config": self.cfg.as_dict(), "params": self.params.as_dict(), "host_steps": self.host_steps,
                "stats": dict(self.stats), "freeze": bool(self.state.freeze), "pending": len(self.state.pending),
                "policy": {"n_swaps": self.state.n_swaps, "stale": self.state.stale, "drains": self.state.drains,
                           "cum_sel": self.state.cum_sel, "cum_cold": self.state.cum_cold,
                           "hot_moved": P.hot_moved(self.state)},
                "at_boot": self.at_boot_map(), "stream": self.stream}

    def close(self) -> None:
        if self._tele_f is not None:
            self._tele_f.close()
            self._tele_f = None
