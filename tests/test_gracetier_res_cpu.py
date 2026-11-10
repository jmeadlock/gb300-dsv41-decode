"""Mac CPU contracts of A3 / card GT3, adaptive residency on gracetier's fixed slots (docs/HANDOFF-A3-RESIDENCY.md):
`.venv/bin/python -m pytest -q tests/test_gracetier_res_cpu.py`.

Mini shapes, CPU tensors, the manager's real code path (runtime/gracetier_res.py: the copies are the same `copy_`
calls the box enqueues on the model's stream). The scratch groups H.. of every layer are scribbled with random bytes
between steps, which is stricter than the box (there they alias one shared allocation and are dead between steps).

  * after N random swaps: every expert's bytes at its current home == golden; row_map is a permutation and g2e its
    inverse; SF rows follow their expert; the source table never changes; a staged step reads golden bytes
  * `restore_map` returns the boot map (the identical row table) and the boot bytes; the non-exact form the hot set
  * counters attribute picks to experts across swaps (stage_emulated's g2e / ecnt contract; the counting kernel body
    under tests/triton_shim.py == that contract); decode-sized calls only
  * the policy (the vendored v26ar.2 core, unchanged) promotes the traffic's cold experts; freeze / baseline / knobs /
    dump / audit through ctrl.json; the audit finds a corrupted byte and a corrupted SF word
  * `mode: off` (and no file) = the current build: gt_stage's arguments, the kernel, the component's tensors,
    GPUModelRunner.execute_model
  * the wiring in fastpath_gracetier.py on CPU tensors: finalize tables, the manager over the module's CURRENT SF
    banks, the execute_model wrap (barrier first), the checks that raise
  * the card's GT1_RES mode (text) and bench/gt_res_drive.py's pure parts (telemetry, T90, the verdict line)
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from runtime import dg_mega as DM
from runtime import gracetier_pool as GP
from runtime import gracetier_res as GR
from runtime import gracetier_stage as GS
from runtime import residency_policy as P

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "reference", "garner-8043a681")
PORT = os.path.join(REPO, "port", "garner")
CARD = os.path.join(REPO, "bench", "box", "card-gt1.sh")
for p in (os.path.join(REPO, "bench"),):
    if p not in sys.path:
        sys.path.insert(0, p)
import gt_res_drive as DRV  # noqa: E402

# sha256 of gt_stage_kernel's source at the commit before A3 (77ee156): "mode off = today's behaviour byte for byte"
GT_STAGE_KERNEL_SHA = "99995b85ae5c86a689634cf096bd10fd9cd1b2c5ebd2c655ea54f6ed532eff05"


def _g(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _i8(shape, g):
    return torch.randint(-128, 128, shape, generator=g, device="cpu", dtype=torch.int32).to(torch.int8)


def _u8(shape, g, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device="cpu", dtype=torch.int32).to(torch.uint8)


# ================================================================================================ the mini world
class World:
    """`layers` gracetier layers at mini shapes: per layer the golden bytes by expert id, MegaMoE's [G, ...] weight
    tensors (hot groups filled, scratch groups random), the SF banks (MN-major, group order), the router's row_map,
    g2e, and ONE Grace pool for all layers; a manager over them."""

    def __init__(self, layers=(0, 1, 3), E=12, H=7, D=256, I=128, seed=0, cuts=True, cfg=None, tmp=None,  # noqa: E741
                 manager=True):
        self.layers, self.E, self.H, self.C, self.D, self.I = tuple(layers), E, H, E - H, D, I
        self.pool = GP.GracePool(GP.PoolLayout({l: self.C for l in layers}, D, I))
        self.cnt = torch.zeros(max(layers) + 1, E, dtype=torch.int32, device="cpu")
        self.gen = _g(seed + 7)
        rng = random.Random(seed)
        self.gold, self.l1, self.l2, self.sf1, self.sf2 = {}, {}, {}, {}, {}
        self.row_map, self.g2e, self.hot, self.cold, self.src = {}, {}, {}, {}, {}
        for l in layers:
            g = _g(seed * 100 + l)
            w = {p: _i8((E, I, D // 2) if p != "w2" else (E, D, I // 2), g) for p in GP.PROJ}
            s = {p: _u8((E, I, D // 32) if p != "w2" else (E, D, I // 32), g, 100, 140) for p in GP.PROJ}
            g1 = DM.mega_l1_weight(w["w1"], w["w3"])                    # [E, 2I, D/2] by expert id
            g2 = w["w2"].clone()
            s1, s2 = GP.sf_banks(s, "mn")                               # (E, n, q) by expert id
            self.gold[l] = {"l1": g1, "l2": g2, "sf1": s1.contiguous(), "sf2": s2.contiguous()}
            ids = list(range(E))
            rng.shuffle(ids)
            hot, cold = ids[:H], ids[H:]
            row = GR.row_of(hot, cold)
            g2e = GR.g2e_list(row)
            l1 = _i8((E, 2 * I, D // 2), self.gen)
            l2 = _i8((E, D, I // 2), self.gen)
            l1[:H].copy_(g1[hot])
            l2[:H].copy_(g2[hot])
            for c, e in enumerate(cold):
                self.pool.fill_expert(l, c, w["w1"][e], w["w3"][e], w["w2"][e])
            banks = []
            for sx in (s1, s2):
                _, n, q = sx.shape
                b = torch.empty_strided((E, n, q), (n * q, 1, n), dtype=torch.int32, device="cpu")
                b.copy_(sx[torch.tensor(g2e, device="cpu")])
                banks.append(b)
            self.l1[l], self.l2[l], (self.sf1[l], self.sf2[l]) = l1, l2, banks
            self.row_map[l] = torch.tensor(row, dtype=torch.int32, device="cpu")
            self.g2e[l] = torch.tensor(g2e, dtype=torch.int32, device="cpu")
            self.hot[l], self.cold[l] = hot, cold
            self.src[l] = self.pool.src_table(l)
        self.boot = {"row": {l: self.row_map[l].clone() for l in layers},
                     "l1": {l: self.l1[l][:H].clone() for l in layers},
                     "l2": {l: self.l2[l][:H].clone() for l in layers},
                     "g1": {l: self.pool.host_l1(l).clone() for l in layers},
                     "g2": {l: self.pool.host_l2(l).clone() for l in layers},
                     "sf1": {l: self.sf1[l].clone() for l in layers}, "sf2": {l: self.sf2[l].clone() for l in layers},
                     "src": {l: self.src[l].clone() for l in layers}}
        b1, b2 = self.pool.layout.B1, self.pool.layout.B2
        # an emulated mapping boundary inside the first cold group (the box: the VMM private / shared boundary)
        self.cuts = ((H * b1 + 4096,), (H * b2 + 1024,)) if cuts else ((), ())
        self.cfg = cfg or GR.ResConfig(mode="adaptive", freeze=True, dir=None if tmp is None else str(tmp))
        self.logs = []
        self.mgr = self.manager() if manager else None

    def views(self):
        return {l: GR.LayerView(l, self.H, self.row_map[l], self.g2e[l], self.l1[l], self.l2[l], self.sf1[l],
                                self.sf2[l], self.pool.host_l1(l), self.pool.host_l2(l), cuts1=self.cuts[0],
                                cuts2=self.cuts[1]) for l in self.layers}

    def manager(self, cfg=None):
        return GR.GtResManager(self.views(), self.hot, self.cold, self.cnt, cfg or self.cfg, log=self.logs.append)

    def resolve(self, addr):
        return self.pool.buf.cpu[addr - self.pool.buf.ptr:]

    def dirty(self):
        """The scratch is dead between steps: scribble every layer's groups H.. (and nothing else)."""
        for l in self.layers:
            self.l1[l][self.H:].copy_(_i8(tuple(self.l1[l][self.H:].shape), self.gen))
            self.l2[l][self.H:].copy_(_i8(tuple(self.l2[l][self.H:].shape), self.gen))

    def step(self, l, ids, count=True):
        """One layer-call the way the server runs it: groups = row_map[ids] -> gt_stage (emulated, with the counter)
        -> the bytes MegaMoE would read for every routed expert are the golden ones. Returns the groups."""
        self.dirty()
        ids_t = torch.tensor(ids, dtype=torch.int64, device="cpu")
        groups = torch.where(ids_t >= 0, self.row_map[l][ids_t.clamp(min=0)].long(), ids_t)
        kw = {"g2e": self.g2e[l], "ecnt": self.cnt[l]} if count else {}
        GS.stage_emulated(groups, self.H, self.src[l], [self.l1[l], self.l2[l]], None, self.resolve, **kw)
        G = self.gold[l]
        for e, g in zip(ids_t.reshape(-1).tolist(), groups.reshape(-1).tolist()):
            if e < 0:
                continue
            assert torch.equal(self.l1[l][g], G["l1"][e]) and torch.equal(self.l2[l][g], G["l2"][e]), (l, e, g)
            assert torch.equal(self.sf1[l][g], G["sf1"][e]) and torch.equal(self.sf2[l][g], G["sf2"][e]), (l, e, g)
        return groups

    def check(self, mgr=None):
        """Every expert's bytes at its current home == golden; the tables are a permutation and its inverse; SF rows
        follow their expert; the mirror == the device tables; the source table is the boot one."""
        mgr = mgr or self.mgr
        E, H = self.E, self.H
        for l in self.layers:
            row = self.row_map[l].tolist()
            g2e = self.g2e[l].tolist()
            assert sorted(row) == list(range(E)), l
            assert all(g2e[row[e]] == e for e in range(E)) and all(row[g2e[g]] == g for g in range(E)), l
            G = self.gold[l]
            h1, h2 = self.pool.host_l1(l), self.pool.host_l2(l)
            for e in range(E):
                g = row[e]
                if g < H:
                    assert torch.equal(self.l1[l][g], G["l1"][e]) and torch.equal(self.l2[l][g], G["l2"][e]), (l, e)
                else:
                    assert torch.equal(h1[g - H], G["l1"][e]) and torch.equal(h2[g - H], G["l2"][e]), (l, e)
                assert torch.equal(self.sf1[l][g], G["sf1"][e]) and torch.equal(self.sf2[l][g], G["sf2"][e]), (l, e)
            assert torch.equal(self.src[l], self.boot["src"][l]) and torch.equal(self.pool.src_table(l), self.src[l])
            if mgr is not None:
                assert mgr.row[l] == row
                assert mgr.state.hot[l] == g2e[:H] and mgr.state.cold[l] == g2e[H:], l
                assert mgr.displaced[l] == sum(a != b for a, b in zip(row, self.boot["row"][l].tolist()))

    def at_boot(self):
        for l in self.layers:
            B = self.boot
            if not (torch.equal(self.row_map[l], B["row"][l]) and torch.equal(self.l1[l][:self.H], B["l1"][l])
                    and torch.equal(self.l2[l][:self.H], B["l2"][l])
                    and torch.equal(self.pool.host_l1(l), B["g1"][l]) and torch.equal(self.pool.host_l2(l), B["g2"][l])
                    and torch.equal(self.sf1[l], B["sf1"][l]) and torch.equal(self.sf2[l], B["sf2"][l])):
                return False
        return True


def _barriers(mgr, n):
    return sum(mgr.barrier() for _ in range(n))


def _swap_some(W, n, seed=1, per_barrier=3):
    """n random swaps through the queue + barriers (the serving path), a scratch scribble between barriers."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n:
        W.mgr.queue_random(per_barrier, rng)
        W.dirty()
        done += W.mgr.barrier()
    while W.mgr.state.pending:
        W.dirty()
        done += W.mgr.barrier()
    return done


# ================================================================================================ pure helpers
def test_tables_and_pieces():
    hot, cold = [5, 0, 3], [4, 1, 2]
    row = GR.row_of(hot, cold)
    assert row == [1, 4, 5, 2, 3, 0] and GR.g2e_list(row) == [5, 0, 3, 4, 1, 2] == hot + cold
    assert GR.row_of(hot, cold) == GP.row_map(hot, cold, 6) and GR.g2e_list(row) == GP.group_experts(row)
    with pytest.raises(ValueError):
        GR.g2e_list([0, 0, 1])
    with pytest.raises(ValueError):
        GR.row_of([0, 1], [1, 3])
    assert GR.pieces(40, 3, ()) == [(0, 40)]
    assert GR.pieces(40, 3, (128,)) == [(0, 8), (8, 32)]                # the boundary inside group 3: two memcpys
    assert GR.pieces(40, 3, (120, 160)) == [(0, 40)]                    # boundaries on the group's edges: one
    assert GR.pieces(40, 2, (128,)) == [(0, 40)] and GR.pieces(40, 3, (130, 125)) == [(0, 5), (5, 5), (10, 30)]
    assert sum(n for _, n in GR.pieces(48, 7, (340, 350, 383))) == 48


def test_config_file_modes(tmp_path):
    p = str(tmp_path / "gracetier-res.json")
    assert GR.load_config(p) is None                                    # absent = off
    json.dump({"mode": "off", "freeze": False}, open(p, "w"))
    assert GR.load_config(p) is None
    json.dump({"mode": "adaptive", "freeze": True, "halflife_steps": 250, "swaps_per_drain": 3}, open(p, "w"))
    c = GR.load_config(p)
    assert c.mode == "adaptive" and c.freeze and c.halflife == 250 and c.swaps_per_drain == 3
    pr = c.params()
    assert (pr.drain_n, pr.hyst, pr.min_abs, pr.swaps_per_barrier, pr.frozen_at_boot) == (64, 1.5, 8.0, 4, True)
    d = GR.ResConfig()                                                  # shipped v26 values, swaps_per_barrier 4
    assert (d.mode, d.drain_n, d.swaps_per_drain, d.halflife, d.hyst, d.min_abs, d.swaps_per_barrier,
            d.count_tokens_max, d.dir) == ("off", 64, 2, 500.0, 1.5, 8.0, 4, 256, "/prof/gracetier-res")
    for bad in ({"mode": "adaptve"}, {"mode": "adaptive", "hysteresis": 2}, {"mode": "adaptive", "drain_n": 0}):
        json.dump(bad, open(p, "w"))
        with pytest.raises(ValueError):
            GR.load_config(p)
    open(p, "w").write("{not json")
    with pytest.raises(ValueError):
        GR.load_config(p)


def test_layer_view_refuses_wrong_geometry():
    W = World(layers=(0,), manager=False)
    l = 0
    a = dict(idx=0, H=W.H, row_map=W.row_map[l], g2e=W.g2e[l], l1=W.l1[l], l2=W.l2[l], sf1=W.sf1[l], sf2=W.sf2[l],
             grace1=W.pool.host_l1(l), grace2=W.pool.host_l2(l))
    v = GR.LayerView(**a)
    assert (v.G, v.C, v.B) == (W.E, W.C, (W.pool.layout.B1, W.pool.layout.B2))
    assert v.w[0].data_ptr() == W.l1[l].data_ptr() and v.gr[0].data_ptr() == W.pool.host_l1(l).data_ptr()
    assert v.sf[0].data_ptr() == W.sf1[l].data_ptr() and torch.equal(v.sf[0][3].view(W.sf1[l].shape[2], -1).T,
                                                                     W.sf1[l][3])
    for k, bad in (("H", W.H + 1), ("l2", W.l1[l]), ("sf1", W.sf1[l].contiguous()), ("grace1", W.pool.host_l2(l)),
                   ("row_map", W.row_map[l].long()), ("g2e", W.g2e[l][:-1]), ("l1", W.l1[l].to(torch.int16))):
        with pytest.raises(ValueError):
            GR.LayerView(**dict(a, **{k: bad}))


def test_manager_refuses_tables_that_are_not_the_rowmap():
    W = World(layers=(0, 1), manager=False)
    W.manager()                                                         # the consistent world is accepted
    W.row_map[1][W.hot[1][0]] = W.H                                     # not the all-resident map of hot / cold
    with pytest.raises(RuntimeError, match="row_map"):
        W.manager()
    W = World(layers=(0, 1), manager=False)
    W.g2e[0][0], W.g2e[0][1] = int(W.g2e[0][1]), int(W.g2e[0][0])
    with pytest.raises(RuntimeError, match="g2e"):
        W.manager()
    W = World(layers=(0, 1), manager=False)
    with pytest.raises(ValueError):
        GR.GtResManager(W.views(), {0: W.hot[0]}, W.cold, W.cnt, W.cfg, log=W.logs.append)
    with pytest.raises(ValueError):
        GR.GtResManager(W.views(), W.hot, W.cold, W.cnt[:1], W.cfg, log=W.logs.append)


# ================================================================================================ swaps
def test_one_swap_is_a_true_exchange():
    W = World()
    W.check()
    l = W.layers[1]
    e_in, e_out = W.cold[l][2], W.hot[l][4]
    g, c = 4, 2
    before_src = W.src[l].clone()
    W.mgr.force_swaps([(l, e_in, e_out)])
    assert W.mgr.barrier() == 1
    row, g2e = W.row_map[l].tolist(), W.g2e[l].tolist()
    assert row[e_in] == g and row[e_out] == W.H + c and g2e[g] == e_in and g2e[W.H + c] == e_out   # step 5
    G = W.gold[l]
    assert torch.equal(W.l1[l][g], G["l1"][e_in]) and torch.equal(W.l2[l][g], G["l2"][e_in])       # step 2
    assert torch.equal(W.pool.host_l1(l)[c], G["l1"][e_out]) and torch.equal(W.pool.host_l2(l)[c], G["l2"][e_out])
    assert torch.equal(W.sf1[l][g], G["sf1"][e_in]) and torch.equal(W.sf1[l][W.H + c], G["sf1"][e_out])   # step 4
    assert torch.equal(W.sf2[l][g], G["sf2"][e_in]) and torch.equal(W.sf2[l][W.H + c], G["sf2"][e_out])
    assert torch.equal(W.src[l], before_src)                            # src_tab never changes
    assert W.mgr.state.hot[l][g] == e_in and W.mgr.state.cold[l][c] == e_out and W.mgr.state.n_swaps == 1
    assert W.mgr.home(l, e_in) == ("hbm", g) and W.mgr.home(l, e_out) == ("grace", c)
    assert W.mgr.stats["swaps"] == 1 and W.mgr.stats["copies"] == 2 * 3    # per tensor: steps 1-3, one piece each
    for other in W.layers:
        if other != l:
            assert torch.equal(W.row_map[other], W.boot["row"][other])
    W.check()
    assert not W.at_boot() and not W.mgr.at_boot_map()


def test_swap_of_the_group_on_the_mapping_boundary_is_cut_and_exact():
    """Cold slot 0 = group H holds the emulated private / shared boundary: its copies are two memcpys per tensor."""
    W = World(cuts=True)
    l = W.layers[0]
    v = W.mgr.views[l]
    assert len(v.pieces(0, W.H)) == 2 and len(v.pieces(1, W.H)) == 2 and len(v.pieces(0, W.H + 1)) == 1
    assert len(v.pieces(0, W.H - 1)) == 1                               # hot groups are never cut
    W.mgr.force_swaps([(l, W.cold[l][0], W.hot[l][0])])
    assert W.mgr.barrier() == 1
    assert W.mgr.stats["copies"] == 2 * (2 + 1 + 2)                     # steps 1 and 3 in two pieces, step 2 in one
    W.check()


@pytest.mark.parametrize("seed,n", [(0, 40), (1, 96)])
def test_random_swaps_keep_bytes_tables_and_sf(seed, n):
    W = World(seed=seed)
    done = _swap_some(W, n, seed=seed + 10)
    assert done >= n and W.mgr.stats["swaps"] == done == W.mgr.state.n_swaps
    W.check()
    assert any(W.mgr.displaced.values()) and not W.at_boot()
    rng = random.Random(seed)
    for l in W.layers:                                                  # a staged step reads golden bytes
        for _ in range(4):
            W.step(l, [rng.sample(range(W.E), 6) for _ in range(3)])
    W.check()
    a = W.mgr.audit(random_per_layer=W.E)
    assert a["mismatches"] == 0 and a["checked"] == len(W.layers) * W.E
    assert W.mgr.state.stale == 0 and W.mgr.stats["vetoed"] == 0


def test_barrier_budget_and_stale_swaps():
    W = World(cfg=GR.ResConfig(mode="adaptive", dir=None, swaps_per_barrier=4))
    l = W.layers[0]
    pairs = [(l, W.cold[l][i], W.hot[l][i]) for i in range(5)] + [(W.layers[1], W.cold[W.layers[1]][0],
                                                                    W.hot[W.layers[1]][0])]
    W.mgr.force_swaps(pairs)
    assert W.mgr.barrier() == 4 and len(W.mgr.state.pending) == 2       # <= swaps_per_barrier per barrier
    assert W.mgr.barrier() == 2 and W.mgr.barrier() == 0
    W.check()
    # the same decision queued twice: the second no longer matches the mirror -> dropped as stale, nothing moves
    e_in, e_out = W.mgr.state.cold[l][3], W.mgr.state.hot[l][6]
    W.mgr.force_swaps([(l, e_in, e_out)])
    item = dict(W.mgr.state.pending[-1])
    W.mgr.state.pending.append(item)
    assert W.mgr.barrier() == 1 and W.mgr.state.stale == 1
    W.check()
    with pytest.raises(ValueError):
        W.mgr.force_swaps([(l, e_in, e_out)])                           # e_in is hot now: not a cold in_id
    assert W.mgr.force_swaps([(l, e_out, e_in)]) == 1 and W.mgr.barrier() == 1   # the reverse is a valid swap
    W.check()


def test_mirror_table_disagreement_raises():
    W = World()
    l = W.layers[0]
    W.mgr.force_swaps([(l, W.cold[l][1], W.hot[l][1])])
    W.mgr.row[l][W.hot[l][1]] = 0                                       # corrupt the manager's own bookkeeping
    with pytest.raises(RuntimeError, match="disagree"):
        W.mgr.barrier()


# ================================================================================================ restore
def test_restore_map_returns_the_boot_map_and_bytes(tmp_path):
    W = World(tmp=tmp_path, cfg=GR.ResConfig(mode="adaptive", freeze=True, dir=str(tmp_path), drain_n=4))
    _swap_some(W, 60, seed=3)
    W.check()
    assert not W.at_boot()
    ctrl = tmp_path / "ctrl.json"
    ctrl.write_text(json.dumps({"seq": 11, "freeze": True, "restore_map": True}))
    n = 0
    while not (W.mgr.at_boot_map() and not W.mgr.state.pending):
        W.dirty()
        W.mgr.barrier()
        n += 1
        assert n < 2000
    ack = [r for r in W.mgr.tele if r["event"] == "ctrl"][-1]
    assert ack["seq"] == 11 and ack["restore_exact"] is True and ack["dropped_pending"] == 0 and ack["restore_queued"] > 0
    assert W.at_boot()                                                  # row_map, HBM groups, Grace pool, SF banks
    W.check()
    assert all(W.mgr.state.hot[l] == W.hot[l] and W.mgr.state.cold[l] == W.cold[l] for l in W.layers)
    assert P.hot_moved(W.mgr.state) == 0 and W.mgr.stats["restores"] == 1
    a = W.mgr.audit(random_per_layer=0)                                 # every expert that ever moved, back home
    assert a["mismatches"] == 0 and a["moved"] == a["checked"] > 0 and a["at_boot"] is True
    for l in W.layers:
        W.step(l, [list(range(6)), list(range(6, 12))])


def test_restore_map_hook_form_restores_the_hot_set():
    W = World()
    _swap_some(W, 40, seed=5)
    l0 = W.layers[0]
    W.mgr.force_swaps([(l0, W.mgr.state.cold[l0][0], W.mgr.state.hot[l0][0])])     # a decision still queued ...
    rec = W.mgr.apply_ctrl({"freeze": True, "restore_map": True, "restore_exact": False})
    assert rec["restore_exact"] is False and rec["dropped_pending"] == 1           # ... is dropped, not applied
    assert W.mgr.stats["dropped_pending"] == 1
    W.mgr.flush()
    W.check()
    assert all(set(W.mgr.state.hot[l]) == set(W.hot[l]) for l in W.layers) and P.hot_moved(W.mgr.state) == 0
    a = W.mgr.audit(random_per_layer=W.E)
    assert a["mismatches"] == 0


# ================================================================================================ counters
def test_counters_attribute_picks_to_experts_across_swaps():
    W = World()
    rng = random.Random(4)
    want = {l: [0] * W.E for l in W.layers}
    for it in range(12):
        for l in W.layers:
            ids = [rng.sample(range(W.E), 6) + [-1] for _ in range(rng.choice([1, 6]))]
            W.step(l, ids)
            for r in ids:
                for e in r:
                    if e >= 0:
                        want[l][e] += 1
        if it % 3 == 2:                                                 # swaps between steps move experts' groups
            _swap_some(W, 6, seed=it)
    assert any(W.mgr.displaced.values())
    for l in W.layers:
        assert W.cnt[l].tolist() == want[l], l                          # per EXPERT, wherever it sat when picked
    assert int(W.cnt.sum()) == sum(sum(v) for v in want.values())
    unused = [r for r in range(W.cnt.shape[0]) if r not in W.layers]
    assert not bool(W.cnt[unused].any())


def test_stage_emulated_counter_contract_and_gating():
    W = World(layers=(0,), manager=False)
    l, H = 0, W.H
    g2e, src = W.g2e[l], W.src[l]
    groups = torch.tensor([[0, H, H, -1, 3, H + 2], [H, 1, 3, 3, -1, -1]], dtype=torch.int64, device="cpu")
    assert GS.count_plan(groups.reshape(-1).tolist(), g2e.tolist()) == {
        int(g2e[0]): 1, int(g2e[H]): 3, int(g2e[3]): 3, int(g2e[H + 2]): 1, int(g2e[1]): 1}
    ecnt = torch.zeros(W.E, dtype=torch.int32, device="cpu")
    slot = torch.zeros(W.C, dtype=torch.int32, device="cpu")
    plan = GS.stage_emulated(groups, H, src, [W.l1[l], W.l2[l]], slot, W.resolve, g2e=g2e, ecnt=ecnt)
    assert plan == {0: 3, 2: 1} and slot.tolist() == [3, 0, 1, 0, 0]    # the per-slot counter of the first build
    assert int(ecnt.sum()) == 9 and int(ecnt[int(g2e[H])]) == 3 and int(ecnt[int(g2e[3])]) == 3
    # more ids than a counted call may have: staged as before, nothing counted
    before = ecnt.clone()
    GS.stage_emulated(groups, H, src, [W.l1[l], W.l2[l]], None, W.resolve, g2e=g2e, ecnt=ecnt, count_ids_max=11)
    assert torch.equal(ecnt, before)
    GS.stage_emulated(groups, H, src, [W.l1[l], W.l2[l]], None, W.resolve, g2e=g2e, ecnt=ecnt, count_ids_max=12)
    assert int(ecnt.sum()) == 18
    assert GS.count_ids() == 256 * 6 == 1536 and GS.count_ids(64, 6) == 384
    with pytest.raises(ValueError):
        GS.stage_emulated(groups, H, src, [W.l1[l], W.l2[l]], None, W.resolve, g2e=g2e)
    # old behaviour when the arguments are absent
    c2 = torch.zeros(W.C, dtype=torch.int32, device="cpu")
    assert GS.stage_emulated(groups, H, src, [W.l1[l], W.l2[l]], c2, W.resolve) == plan and c2.tolist() == slot.tolist()


def _stage_under_shim():
    import triton_shim

    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gracetier_stage.py"),
                                             "gracetier_stage_res_under_shim")
    return mod, triton_shim


@pytest.mark.parametrize("nprog,bw", [(1, 1024), (3, 512)])
def test_count_kernel_body_equals_contract(nprog, bw):
    """gt_stage_count_kernel program by program == stage_emulated with g2e / ecnt: the same staged bytes and slot
    counter as the first build's kernel, plus one count per pick per expert (duplicates, -1 skipped, hot and cold)."""
    mod, shim = _stage_under_shim()
    outs = []
    groups = None
    for use_kernel in (False, True):
        W = World(layers=(2,), E=7, H=3, seed=9, manager=False)
        l, H = 2, W.H
        groups = torch.tensor([[3, 0, 6, -1, 2, 5], [6, 1, 2, 0, 3, -1], [3, 3, 0, 0, -1, 6]], dtype=torch.int64,
                              device="cpu")
        cnt_slot = torch.zeros(W.C, dtype=torch.int32, device="cpu")
        dsts = [W.l1[l], W.l2[l]]
        if use_kernel:
            shim.ADDRESSES.clear()
            shim.ADDRESSES.register(W.pool.buf.cpu, *dsts)
            n_plain = len(mod.gt_stage_kernel.launches)
            try:
                grid = mod.gt_stage(groups.reshape(-1).contiguous(), H, W.src[l], GS.make_dtab(*dsts), cnt_slot,
                                    W.pool.layout.B1, W.pool.layout.B2, nprog=nprog, block_words=bw,
                                    g2e=W.g2e[l], ecnt=W.cnt[l])
            finally:
                shim.ADDRESSES.clear()
            assert grid == (W.C, nprog) and mod.gt_stage_count_kernel.launches[-1]["grid"] == (W.C, nprog)
            assert len(mod.gt_stage_kernel.launches) == n_plain         # ONE launch, the counting kernel
        else:
            GS.stage_emulated(groups, H, W.src[l], dsts, cnt_slot, W.resolve, g2e=W.g2e[l], ecnt=W.cnt[l])
        outs.append([t.clone() for t in dsts] + [cnt_slot.clone(), W.cnt.clone(), W.g2e[l].clone()])
    for i, (a, b) in enumerate(zip(*outs)):
        assert torch.equal(a, b), i
    g2e = outs[1][4].tolist()
    want = [0] * 7
    for g in groups.reshape(-1).tolist():
        if g >= 0:
            want[g2e[g]] += 1
    assert outs[1][3][2].tolist() == want and sum(want) == 15
    assert not bool(outs[1][3][:2].any())                               # only this layer's row of [L, E] is written


def test_count_kernel_gating_and_refusals():
    mod, shim = _stage_under_shim()
    W = World(layers=(0,), E=7, H=3, manager=False)
    l, H = 0, W.H
    dsts = [W.l1[l], W.l2[l]]
    dtab, b1, b2 = GS.make_dtab(*dsts), W.pool.layout.B1, W.pool.layout.B2
    ids = torch.tensor([3, 0, 4, -1, 2, 5, 6, 1], dtype=torch.int64, device="cpu")
    n_plain, n_count = len(mod.gt_stage_kernel.launches), len(mod.gt_stage_count_kernel.launches)
    shim.ADDRESSES.register(W.pool.buf.cpu, *dsts)
    try:
        # a call larger than a counted call: the first build's kernel, nothing counted
        mod.gt_stage(ids, H, W.src[l], dtab, None, b1, b2, nprog=2, g2e=W.g2e[l], ecnt=W.cnt[l], count_ids_max=7)
        assert len(mod.gt_stage_kernel.launches) == n_plain + 1 and not bool(W.cnt.any())
        assert len(mod.gt_stage_count_kernel.launches) == n_count
        mod.gt_stage(ids, H, W.src[l], dtab, None, b1, b2, nprog=2, g2e=W.g2e[l], ecnt=W.cnt[l], count_ids_max=8)
        assert len(mod.gt_stage_count_kernel.launches) == n_count + 1 and int(W.cnt.sum()) == 7
        # all ids -1 (the prewarm): nothing counted, nothing staged
        before = [t.clone() for t in dsts] + [W.cnt.clone()]
        mod.gt_stage(torch.full((6,), -1, dtype=torch.int64, device="cpu"), H, W.src[l], dtab, None, b1, b2,
                     nprog=1, g2e=W.g2e[l], ecnt=W.cnt[l])
        assert all(torch.equal(a, b) for a, b in zip(before, dsts + [W.cnt]))
    finally:
        shim.ADDRESSES.clear()
    for bad in (dict(g2e=W.g2e[l]), dict(g2e=W.g2e[l].long(), ecnt=W.cnt[l]), dict(g2e=W.g2e[l], ecnt=W.cnt)):
        with pytest.raises(ValueError):
            mod.gt_stage(ids, H, W.src[l], dtab, None, b1, b2, **bad)


# ================================================================================================ policy
def _traffic(W, popular, steps, rng, others=None):
    """`steps` x (barrier, then one layer-call per layer that routes 6 ids: 4 of that layer's popular experts + 2 of
    `others[l]`, default the experts that are not popular). The barrier runs BEFORE the step, as execute_model's wrap
    does, so a drain at barrier n has read the counts of n - 1 steps."""
    for _ in range(steps):
        W.mgr.barrier()
        for l in W.layers:
            rest = others[l] if others else [e for e in range(W.E) if e not in popular[l]]
            W.step(l, [rng.sample(popular[l], 4) + rng.sample(rest, 2)])


def test_policy_promotes_the_traffics_cold_experts(tmp_path):
    cfg = GR.ResConfig(mode="adaptive", freeze=False, dir=str(tmp_path), drain_n=8, min_abs=4.0)
    W = World(tmp=tmp_path, cfg=cfg)
    rng = random.Random(2)
    popular = {l: list(W.cold[l]) for l in W.layers}                    # the traffic: the 5 boot-cold experts ...
    keep = {l: W.hot[l][:2] for l in W.layers}                          # ... + 2 boot-hot ones = H = 7 experts in use
    _traffic(W, popular, 8 * 14, rng, others=keep)
    W.check()
    drains = [r for r in W.mgr.tele if r["event"] == "drain"]
    assert len(drains) >= 12
    for k in ("host_steps", "steps", "sel", "cold", "cold_share", "queued", "swaps", "swaps_since", "vetoed", "stale",
              "hot_moved", "pending", "freeze", "at_boot", "t"):
        assert k in drains[-1], k
    first = next(r for r in drains if r["sel"])
    assert first["cold_share"] == pytest.approx(4 / 6) and first["sel"] == (first["steps"] - 1) * 6 * len(W.layers)
    assert all(r["sel"] == r["steps"] * 6 * len(W.layers) for r in drains[1:])
    assert drains[-1]["cold_share"] == 0.0 and drains[-1]["cold"] == 0  # every expert in use now lives in HBM
    assert all(r["swaps_since"] <= cfg.swaps_per_barrier * r["steps"] for r in drains)
    for l in W.layers:
        assert set(W.mgr.state.hot[l]) == set(popular[l]) | set(keep[l]), l
    assert W.mgr.stats["swaps"] == sum(r["swaps_since"] for r in drains) > 0
    assert W.mgr.state.stale == 0 and P.hot_moved(W.mgr.state) == drains[-1]["hot_moved"]
    lines = [json.loads(x) for x in open(tmp_path / "telemetry.jsonl")]
    assert lines[0]["event"] == "boot" and [r for r in lines if r["event"] == "drain"] == drains
    assert W.mgr.audit(random_per_layer=W.E)["mismatches"] == 0
    # the policy core's own counters are the telemetry's
    assert W.mgr.state.cum_sel == sum(r["sel"] for r in drains) and W.mgr.state.drains == len(drains)


def test_policy_is_the_vendored_core_unchanged():
    """The manager's decisions are residency_policy's: same drain inputs -> the same pending queue as the bare core."""
    W = World(cfg=GR.ResConfig(mode="adaptive", freeze=False, dir=None, drain_n=4, min_abs=2.0))
    ref_state = P.State.from_rows({l: {"hot": list(W.hot[l]), "cold": list(W.cold[l])} for l in W.layers}, W.E,
                                  W.cnt.shape[0], W.mgr.params)
    ref_state.freeze = False
    rng = random.Random(8)
    for l in W.layers:
        for _ in range(6):
            W.step(l, [rng.sample(W.cold[l], 4) + rng.sample(W.hot[l], 2)])
    cur = W.cnt.numpy().astype(np.int64)
    _barriers(W.mgr, 4)                                                 # the 4th barrier reads the counter
    assert W.mgr.state.pending == []
    W.mgr.barrier()                                                     # the 5th drains and applies <= 4
    P.drain(ref_state, W.mgr.params, cur, 4)
    applied = P.apply_pending(ref_state, W.mgr.params)
    assert len(applied) == W.mgr.stats["swaps"] > 0
    assert ref_state.hot == W.mgr.state.hot and ref_state.cold == W.mgr.state.cold
    assert ref_state.pending == W.mgr.state.pending and np.array_equal(ref_state.ewma, W.mgr.state.ewma)
    W.check()
    src = open(os.path.join(REPO, "runtime", "residency_policy.py")).read()
    body = src.split("# ==== END-OF-HEADER ====\n", 1)[1]
    assert hashlib.sha256(body.encode()).hexdigest() == \
        "1aafea83557d09fd2d85089fd543c377e12f95dcdc082fd715d30bcc29494fec"     # the vendored file's own pin


def test_frozen_counts_and_drains_but_never_queues():
    W = World(cfg=GR.ResConfig(mode="adaptive", freeze=True, dir=None, drain_n=4, min_abs=1.0))
    rng = random.Random(3)
    _traffic(W, {l: list(W.cold[l]) for l in W.layers}, 40, rng)
    assert W.mgr.stats["drains"] >= 8 and W.mgr.stats["swaps"] == 0 and W.at_boot()
    assert float(W.mgr.state.ewma.sum()) > 0                            # the EWMA still follows the traffic
    drains = [r for r in W.mgr.tele if r["event"] == "drain"]
    assert all(r["freeze"] and r["queued"] == 0 and r["at_boot"] for r in drains)
    assert min(r["cold_share"] for r in drains if r["sel"]) > 0.6


# ================================================================================================ control file
def _ctrl(tmp_path, doc):
    tmp = tmp_path / ".ctrl.tmp"
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, tmp_path / "ctrl.json")


def _until_ack(W, seq, rng, popular, limit=400):
    for _ in range(limit):
        _traffic(W, popular, 1, rng)
        ack = DRV.find_ack(list(W.mgr.tele), seq)
        if ack:
            return ack
    raise AssertionError(f"ctrl seq {seq} never acknowledged")


def test_ctrl_file_is_polled_at_drains_and_acknowledged(tmp_path):
    (tmp_path / "ctrl.json").write_text(json.dumps({"seq": 1, "freeze": False}))    # left by an earlier boot
    cfg = GR.ResConfig(mode="adaptive", freeze=True, dir=str(tmp_path), drain_n=4, min_abs=2.0)
    W = World(tmp=tmp_path, cfg=cfg)
    rng = random.Random(6)
    popular = {l: list(W.cold[l]) for l in W.layers}
    _traffic(W, popular, 24, rng)
    assert W.mgr.state.freeze and W.mgr.stats["ctrl"] == 0 and W.mgr.stats["swaps"] == 0   # the stale file is ignored
    # unfreeze + baseline: acknowledged at a drain, the EWMA starts from zero there, swaps follow
    _ctrl(tmp_path, {"seq": 2, "freeze": False, "baseline": True, "swaps_per_drain": 1, "bogus": 3})
    ack = _until_ack(W, 2, rng, popular)
    assert ack["freeze"] is False and ack["unknown"] == ["bogus"] and ack["params"]["swaps_per_drain"] == 1
    assert "freeze=0" in ack["msg"] and "baseline" in ack["msg"]
    after = [r for r in W.mgr.tele if r["event"] == "drain" and r["t"] >= ack["t"]]
    assert after[0]["sel"] == 0 and after[0]["cold_share"] is None      # the baseline drain counts nothing
    _traffic(W, popular, 40, rng)
    assert W.mgr.stats["swaps"] > 0
    drains = [r for r in W.mgr.tele if r["event"] == "drain" and r["t"] >= ack["t"]]
    assert all(r["queued"] <= len(W.layers) for r in drains)            # swaps_per_drain 1, per layer
    # the same seq again (the file is rewritten): not applied twice
    n_ctrl = W.mgr.stats["ctrl"]
    _ctrl(tmp_path, {"seq": 2, "freeze": True})
    _traffic(W, popular, 12, rng)
    assert W.mgr.stats["ctrl"] == n_ctrl and W.mgr.state.freeze is False
    # freeze + dump + audit in one document
    _ctrl(tmp_path, {"seq": 3, "freeze": True, "dump": "r3 map/x", "audit": {"random": 4, "seed": 1}})
    ack = _until_ack(W, 3, rng, popular)
    assert ack["freeze"] is True and ack["audit"]["mismatches"] == 0
    assert ack["audit"]["random"] <= 4 * len(W.layers) and ack["audit"]["moved"] > 0
    d = json.load(open(ack["dump"]))
    assert ack["dump"].endswith("dump-r3_map_x.json") and d["at_boot"] is False
    for l in W.layers:
        assert d["layers"][str(l)]["row"] == W.row_map[l].tolist() and d["layers"][str(l)]["hot"] == W.mgr.state.hot[l]
    z = np.load(ack["dump"][:-5] + ".npz")
    assert z["counts"].shape == (W.cnt.shape[0], W.E) and z["ewma"].shape == z["counts"].shape
    # a broken file is reported once and does not stop the barriers
    (tmp_path / "ctrl.json").write_text("{broken")
    _traffic(W, popular, 12, rng)
    assert [r for r in W.mgr.tele if r["event"] == "ctrl_error"] and W.mgr.state.freeze is True
    W.check()
    W.mgr.close()
    assert [json.loads(x)["event"] for x in open(tmp_path / "telemetry.jsonl")].count("ctrl") == 2


def test_audit_finds_a_corrupted_byte_and_sf_word():
    W = World()
    _swap_some(W, 30, seed=7)
    assert W.mgr.audit(random_per_layer=W.E)["mismatches"] == 0
    l = W.layers[1]
    e_hot, e_cold = W.mgr.state.hot[l][2], W.mgr.state.cold[l][1]
    W.l2[l][2].view(-1)[5] += 1                                         # one byte of a hot expert's L2 in HBM
    W.pool.host_l1(l)[1].view(-1)[-1] ^= 1                              # one bit of a cold expert's L1 in Grace
    e_sf = W.mgr.state.hot[l][0]
    W.sf1[l][0, 0, 0] += 1                                              # one SF word of group 0
    a = W.mgr.audit(random_per_layer=W.E)
    got = {(b["layer"], b["expert"]): b["parts"] for b in a["first"]}
    assert a["mismatches"] == 3 and got == {(l, e_hot): ["l2"], (l, e_cold): ["l1"], (l, e_sf): ["sf1"]}
    assert [b["home"] for b in a["first"] if b["expert"] == e_cold] == [["grace", 1]]
    # exchanged words inside one expert are seen too (the digest is position-weighted)
    W2 = World(layers=(0,))
    row = W2.l1[0][1].view(-1)
    a8, b8 = row[:8].clone(), row[8:16].clone()
    assert not torch.equal(a8, b8)
    row[:8], row[8:16] = b8, a8
    assert W2.mgr.audit(random_per_layer=W2.E)["mismatches"] == 1
    # explicit experts; no boot digests -> the audit says so instead of passing
    assert W2.mgr.audit(experts={0: [W2.hot[0][0]]})["checked"] == 1
    W3 = World(layers=(0,), cfg=GR.ResConfig(mode="adaptive", dir=None, boot_digests=False))
    r = W3.mgr.audit()
    assert r["mismatches"] is None and "no boot digests" in r["error"]


def test_swapped_bytes_without_tables_would_be_caught():
    """Mutation: an exchange that forgets the SF rows (step 4) is found by check() and by the audit."""
    W = World(layers=(0,))
    l = 0
    v = W.mgr.views[l]
    a, b = v.sf[0][2].clone(), v.sf[0][W.H + 1].clone()
    W.mgr.force_swaps([(l, W.cold[l][1], W.hot[l][2])])
    W.mgr.barrier()
    v.sf[0][2].copy_(a)                                                 # undo step 4 for bank 1
    v.sf[0][W.H + 1].copy_(b)
    with pytest.raises(AssertionError):
        W.check()
    assert W.mgr.audit(random_per_layer=0)["mismatches"] == 2


# ================================================================================================ the component
def _component():
    if "fastpath_gracetier" in sys.modules:
        return sys.modules["fastpath_gracetier"]
    for p in (os.path.join(REF, "fastpath"), os.path.join(PORT, "fastpath")):
        if p not in sys.path:
            sys.path.append(p)
    return importlib.import_module("fastpath_gracetier")


@pytest.fixture
def comp(monkeypatch):
    """fastpath_gracetier with a fresh process state (restored after the test)."""
    G = _component()
    monkeypatch.setattr(G, "_S", G._State())
    return G


def test_res_doc_modes(comp, tmp_path):
    G = comp
    p = str(tmp_path / "gracetier-res.json")
    assert G.res_doc(p) is None
    json.dump({"mode": "off"}, open(p, "w"))
    assert G.res_doc(p) is None
    json.dump({"mode": "adaptive", "freeze": True}, open(p, "w"))
    assert G.res_doc(p) == {"mode": "adaptive", "freeze": True}
    for bad in ({"mode": "on"}, {"freeze": True}, ["adaptive"]):
        json.dump(bad, open(p, "w"))
        with pytest.raises(RuntimeError):
            G.res_doc(p)
    assert os.path.basename(G.RES_CONFIG) == "gracetier-res.json"
    assert os.path.dirname(G.RES_CONFIG) == os.path.dirname(os.path.abspath(G.__file__))   # = /fastpath on the box
    assert not os.path.exists(G.RES_CONFIG)                             # the repo ships no config: off by default


def _fake_runner():
    calls = []

    class GPUModelRunner:
        def load_model(self, *a, **k):
            calls.append("load")

        def execute_model(self, scheduler_output, x=1):
            """the original"""
            calls.append(("exec", scheduler_output, x))
            return "out"

    return types.SimpleNamespace(GPUModelRunner=GPUModelRunner), calls


def test_mode_off_is_the_current_build(comp, monkeypatch):
    """No config file (and mode off): gt_stage is launched with the first build's seven arguments and no keyword, the
    kernel is the first build's source, finalize builds no residency tensor, execute_model is not touched."""
    G = comp
    assert G._res_doc() is None and G._res_cfg() is None
    calls = []
    fake_gs = types.SimpleNamespace(gt_stage=lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(G._S, "rt", types.SimpleNamespace(GS=fake_gs))
    hot, cold = list(range(255)), list(range(255, 384))
    st = G.GtState(4, hot, cold, GP.row_map(hot, cold))
    G._res_finalize(st, "cpu")                                          # off: nothing is built
    assert st.g2e is None and st.ecnt is None and st.count_ids is None and G._S.res_cnt is None
    assert G._S.modules == {} and G._S.res is None
    st.src_tab, st.dtab, st.cnt, st.b1, st.b2 = torch.zeros(129, 2, dtype=torch.int64), "dtab", "cnt", 11, 22
    G.stage(st, torch.arange(18, dtype=torch.int64))
    (args, kw), = calls
    assert len(args) == 7 and kw == {} and args[1:] == (255, st.src_tab, "dtab", "cnt", 11, 22)
    # the runner: load_model wrapped as before, execute_model the very same function object
    mod, _ = _fake_runner()
    orig_exec = mod.GPUModelRunner.execute_model
    G.patch_runner(mod)
    assert mod.GPUModelRunner.execute_model is orig_exec
    assert getattr(mod.GPUModelRunner.load_model, "_ds41_gracetier", False)
    G._res_create()                                                     # off: returns at once
    assert G._S.res is None
    assert not getattr(torch.cuda.CUDAGraph.replay, "_ds41_gracetier", False)
    # the kernel the off mode launches is the first build's, byte for byte, and the launcher reaches it unchanged
    src = open(os.path.join(REPO, "runtime", "gracetier_stage.py")).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "gt_stage_kernel")
    assert hashlib.sha256(ast.get_source_segment(src, fn).encode()).hexdigest() == GT_STAGE_KERNEL_SHA
    smod, shim = _stage_under_shim()
    W = World(layers=(0,), E=7, H=3, manager=False)
    dsts = [W.l1[0], W.l2[0]]
    n_count = len(smod.gt_stage_count_kernel.launches)
    shim.ADDRESSES.register(W.pool.buf.cpu, *dsts)
    try:
        smod.gt_stage(torch.tensor([3, 0, 5], dtype=torch.int64), W.H, W.src[0], GS.make_dtab(*dsts), None,
                      W.pool.layout.B1, W.pool.layout.B2, nprog=2)
    finally:
        shim.ADDRESSES.clear()
    assert len(smod.gt_stage_count_kernel.launches) == n_count
    assert smod.gt_stage_kernel.launches[-1]["constexprs"] == {} and smod.gt_stage_kernel.launches[-1]["grid"] == (4, 2)
    assert torch.equal(dsts[0][3], W.pool.host_l1(0)[0]) and torch.equal(dsts[1][5], W.pool.host_l2(0)[2])


class _FakeModule:
    pass


def _wired(G, monkeypatch, tmp_path, layers=(0, 1), doc=None):
    """The component's residency wiring on CPU tensors: E = 384 / H = 255 (the component's constants), mini D / I."""
    W = World(layers=layers, E=384, H=255, D=128, I=128, seed=3, manager=False, cuts=False)
    doc = doc or {"mode": "adaptive", "freeze": True, "dir": str(tmp_path), "drain_n": 4}
    monkeypatch.setattr(G._S, "res_doc", doc)
    calls = []
    gs = types.SimpleNamespace(gt_stage=lambda *a, **k: calls.append((a, k)), count_ids=GS.count_ids)
    monkeypatch.setattr(G._S, "rt", types.SimpleNamespace(GS=gs, GR=GR, GP=GP))
    monkeypatch.setattr(G._S, "allres", {l: {"hot": W.hot[l], "cold": W.cold[l], "row": W.row_map[l].tolist()}
                                         for l in layers})
    monkeypatch.setattr(G._S, "pool", W.pool)
    mods = {}
    for l in layers:
        st = G.GtState(l, W.hot[l], W.cold[l], W.row_map[l].tolist())
        st.row_map = W.row_map[l]
        st.src_tab, st.dtab, st.cnt = W.src[l], "dtab", "cnt"
        st.b1, st.b2 = W.pool.layout.B1, W.pool.layout.B2
        G._res_finalize(st, torch.device("cpu"))
        m = _FakeModule()
        m._mp = st
        # finalize's own banks (what gracetier built) ...
        m._transformed_l1_weights = (W.l1[l], W.sf1[l].clone())
        m._transformed_l2_weights = (W.l2[l], W.sf2[l].clone())
        # ... then replaced, as a12_retire does inside its finalize wrap, by the banks MegaMoE reads from then on
        m._transformed_l1_weights = (m._transformed_l1_weights[0], W.sf1[l])
        m._transformed_l2_weights = (m._transformed_l2_weights[0], W.sf2[l])
        m._a12_retire_owners = ("o1", "o2")
        G._S.modules[l] = m
        mods[l] = m
        W.g2e[l] = st.g2e
    W.cnt = G._S.res_cnt
    return W, mods, calls


def test_wiring_adaptive_finalize_manager_and_stage(comp, monkeypatch, tmp_path):
    G = comp
    W, mods, calls = _wired(G, monkeypatch, tmp_path)
    assert tuple(G._S.res_cnt.shape) == (2, 384) and G._S.res_cnt.dtype == torch.int32
    for l, m in mods.items():
        st = m._mp
        assert st.g2e.dtype == torch.int32 and st.g2e.tolist() == GR.g2e_list(st.row_allres)
        assert st.ecnt.data_ptr() == G._S.res_cnt[l].data_ptr() and st.count_ids == 1536
    G._res_create()
    mgr = G._S.res
    assert isinstance(mgr, GR.GtResManager) and sorted(mgr.views) == [0, 1] and mgr.state.freeze is True
    for l, m in mods.items():                                           # the manager swaps the banks the step reads
        assert mgr.views[l].sf[0].data_ptr() == m._transformed_l1_weights[1].data_ptr() == W.sf1[l].data_ptr()
        assert mgr.views[l].w[0].data_ptr() == W.l1[l].data_ptr()
        assert mgr.views[l].gr[0].data_ptr() == W.pool.host_l1(l).data_ptr()
    # stage: the same launch, now with the counter arguments
    st = mods[1]._mp
    G.stage(st, torch.arange(12, dtype=torch.int64))
    (args, kw), = calls
    assert len(args) == 7 and args[1:] == (255, st.src_tab, "dtab", "cnt", st.b1, st.b2)
    assert set(kw) == {"g2e", "ecnt", "count_ids_max"} and kw["g2e"] is st.g2e and kw["count_ids_max"] == 1536
    assert kw["ecnt"].data_ptr() == G._S.res_cnt[1].data_ptr()
    # a swap through the wired manager keeps the world golden
    W.mgr = mgr
    mgr.force_swaps([(1, W.cold[1][0], W.hot[1][0]), (0, W.cold[0][7], W.hot[0][100])])
    assert mgr.barrier() == 2
    W.check()
    assert mgr.audit(random_per_layer=8)["mismatches"] == 0
    assert (tmp_path / "telemetry.jsonl").exists()


def test_wiring_execute_model_wrap_runs_the_barrier_first(comp, monkeypatch, tmp_path):
    G = comp
    W, mods, _ = _wired(G, monkeypatch, tmp_path)
    mod, calls = _fake_runner()
    G.patch_runner(mod)
    cls = mod.GPUModelRunner
    assert getattr(cls.execute_model, "_ds41_gracetier", False) and cls.execute_model.__doc__ == "the original"
    r = cls()
    assert r.execute_model("s0") == "out" and calls == [("exec", "s0", 1)]      # before load end: a plain passthrough
    G._res_create()
    mgr = G._S.res
    order = []
    real_barrier = mgr.barrier
    monkeypatch.setattr(mgr, "barrier", lambda: (order.append(("barrier", G._S.res_in_exec)), real_barrier())[1])
    calls.clear()
    orig = cls.execute_model.__wrapped__
    monkeypatch.setattr(G, "_res_check_eager", lambda: order.append("eager-check"))

    def inner(self, scheduler_output, x=1):
        order.append(("exec", G._S.res_in_exec))
        G.stage(mods[0]._mp, torch.arange(6, dtype=torch.int64))        # an eager gt_stage call inside the step
        return orig(self, scheduler_output, x)

    cls2 = type("GPUModelRunner", (), {"execute_model": inner, "load_model": lambda self: None})
    G._wrap_execute_model(cls2)
    assert cls2().execute_model("s1", x=5) == "out"
    assert order == [("barrier", False), ("exec", True), "eager-check"] and calls == [("exec", "s1", 5)]
    assert G._S.res_in_exec is False and G._S.res_rechecked is True and mgr.host_steps == 1
    cls3 = type("GPUModelRunner", (), {"execute_model": lambda self: 1 / 0})
    G._wrap_execute_model(cls3)
    with pytest.raises(ZeroDivisionError):                              # the flag is cleared on an exception too
        cls3().execute_model()
    assert G._S.res_in_exec is False and mgr.host_steps == 2            # (the barrier ran before the failing step)
    # the checks that raise
    with pytest.raises(RuntimeError, match="execute_model"):
        G._wrap_execute_model(type("NoExec", (), {}))

    async def coro(self):
        return 1

    with pytest.raises(RuntimeError, match="coroutine"):
        G._wrap_execute_model(type("Async", (), {"execute_model": coro}))


def test_wiring_refuses_replaced_tensors_and_missing_layers(comp, monkeypatch, tmp_path):
    G = comp
    W, mods, _ = _wired(G, monkeypatch, tmp_path)
    G._res_create()
    mods[1]._transformed_l1_weights = (W.l1[1], W.sf1[1].clone())       # a bank replaced after the manager was built
    G._S.res_rechecked = False
    with pytest.raises(RuntimeError, match="replaced after load_model"):
        G._res_recheck()
    assert G._S.res_rechecked is False
    G2 = comp
    monkeypatch.setattr(G2, "_S", G2._State())
    W, mods, _ = _wired(G2, monkeypatch, tmp_path)
    del G2._S.modules[1]
    with pytest.raises(RuntimeError, match="finalize"):
        G2._res_create()
    monkeypatch.setattr(G2, "_S", G2._State())
    with pytest.raises(ValueError, match="unknown key"):                # a typo in the config is not "off"
        _wired(G2, monkeypatch, tmp_path, doc={"mode": "adaptive", "freez": True})


def test_component_still_registers_four_targets_and_imports_nothing_heavy():
    G = _component()
    import fastpath_hook as FH

    assert sorted(t for t, _ in FH._COMPONENTS["gracetier"]) == sorted([G.TARGET, G.WEIGHT_UTILS, G.DEFAULT_LOADER,
                                                                         G.RUNNER])
    src = open(os.path.join(PORT, "fastpath", "fastpath_gracetier.py")).read()
    assert not re.search(r"^\s*(import vllm|from vllm)", src, re.M)
    assert not re.search(r"^(import torch|from torch|import numpy|from runtime)", src, re.M)
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    for f in ("runtime/gracetier_res.py", "bench/gt_res_drive.py", "tests/test_gracetier_res_cpu.py"):
        for n in ast.walk(ast.parse(open(os.path.join(REPO, f)).read())):
            if isinstance(n, ast.Import):
                assert not any(a.name.split(".")[0] == "vllm" for a in n.names), f
            elif isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "vllm", f
    drv = ast.parse(open(os.path.join(REPO, "bench", "gt_res_drive.py")).read())    # the box's host python: stdlib
    mods = {a.name.split(".")[0] for n in ast.walk(drv) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"argparse", "json", "os", "statistics", "sys", "time", "w0_drive"}, mods


# ================================================================================================ the driver (pure)
def _drain(t, sel, cold, swaps=0, pending=0, at_boot=False):
    return {"event": "drain", "t": t, "sel": sel, "cold": cold, "swaps_since": swaps, "pending": pending,
            "at_boot": at_boot, "cold_share": (cold / sel) if sel else None}


def test_driver_t90_and_series():
    t0 = 1000.0
    shares = [0.20, 0.12, 0.06, 0.03, 0.012, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010]
    dr = [_drain(t0, 0, 0)] + [_drain(t0 + 0.7 * (i + 1), 10000, int(s * 10000), swaps=4) for i, s in enumerate(shares)]
    series = DRV.cold_share_series(dr)
    assert len(series) == 12 and series[0] == (t0 + 0.7, 0.20)          # the baseline drain (no picks) is not a point
    c = DRV.t90(series, t0)
    assert c["a"] == 0.20 and abs(c["b"] - 0.010) < 1e-12 and abs(c["threshold"] - 0.029) < 1e-12
    assert c["t90_s"] == 3.5 and c["plateau_drains"] == 3 and c["t_first_drain_s"] == 0.7
    flat = DRV.t90([(t0 + i, 0.01) for i in range(8)], t0)
    assert flat["t90_s"] is None and "no reduction" in flat["note"]
    few = DRV.t90(series[:3], t0)
    assert few["t90_s"] is None and few["a"] is None and "too few" in few["note"]
    s = DRV.swaps_per_drain(dr)
    assert s["total"] == 48 and s["max"] == 4 and s["drains"] == 13
    assert DRV.drains_after(dr + [{"event": "ctrl", "t": t0 + 1}], t0 + 0.7, t0 + 1.5) == dr[1:3]


def test_driver_ack_settled_and_ctrl_file(tmp_path):
    seq = DRV.write_ctrl(str(tmp_path), {"freeze": True})
    doc = json.load(open(tmp_path / "ctrl.json"))
    assert doc == {"freeze": True, "seq": seq} and not list(tmp_path.glob(".ctrl.*"))
    rows = [{"event": "drain", "t": 5.0, "pending": 0}, {"event": "ctrl", "seq": seq, "t": 10.0, "pending": 3},
            _drain(11.0, 10, 1, pending=2)]
    ack = DRV.find_ack(rows, seq)
    assert ack["t"] == 10.0 and DRV.find_ack(rows, seq + 1) is None
    assert DRV.settled(rows, ack, "ack") and not DRV.settled(rows, ack, "idle")
    rows.append(_drain(12.0, 10, 1, pending=0))
    assert DRV.settled(rows, ack, "idle") and not DRV.settled(rows, ack, "boot")
    rows.append(_drain(13.0, 10, 1, pending=0, at_boot=True))
    assert DRV.settled(rows, ack, "boot") and DRV.last_state(rows, 10.0)["t"] == 13.0
    tele = tmp_path / "telemetry.jsonl"
    tele.write_text("\n".join(json.dumps(r) for r in rows) + "\n{\"event\": \"drain\", \"t\": 14")   # a line in flight
    assert DRV.read_telemetry(str(tele)) == rows and DRV.read_telemetry(str(tmp_path / "none")) == []
    a = DRV.build_parser().parse_args(["ctrl", "--dir", "d", "--set", "{}", "--wait", "boot", "--nudge", "step"])
    assert (a.cmd, a.wait, a.nudge, a.timeout) == ("ctrl", "boot", "step", 300.0)
    assert DRV.nudge_prompts("step") == [DRV.w0().STEP_PROMPT] and len(DRV.nudge_prompts("fixture")) == 8
    assert DRV.UNFREEZE == {"freeze": False, "baseline": True}


def _step_receipt(path, ms, tps=2.75):
    json.dump({"reps": [{"rep": i, "step_ms_from_metrics": m, "tokens_per_step": tps} for i, m in enumerate(ms)]},
              open(path, "w"))


def _capture(path, streams):
    with open(path, "w") as f:
        for tag, toks in streams.items():
            f.write(json.dumps({"tag": tag, "text": "".join(toks), "logprobs": [{"token": t} for t in toks]}) + "\n")


def test_driver_summary_verdict_and_adopt_bar(tmp_path):
    S, st = str(tmp_path), "20261109-200000"
    p = lambda n: os.path.join(S, f"gt1-res-{n}-{st}")  # noqa: E731
    empty = DRV.build_summary(S, st)                                    # no receipts: nothing is invented
    assert empty["verdict"] == ("GT3: R0 not measured | R2a not measured | R2b not measured | T90 not measured | "
                                "cold share not measured \u2192 not measured | audit not measured mismatches | "
                                "C32 not measured")
    assert empty["adopt"]["adopt"] is False and len(empty["adopt"]["not_measured"]) == 5
    _step_receipt(p("r0-step") + ".json", [10.40, 10.50, 10.45])
    _step_receipt(p("r2a-step") + ".json", [9.90, 9.80, 9.85])
    for i, base in enumerate([9.6, 9.2, 9.0, 8.9, 8.8], 1):
        _step_receipt(p(f"r2b-step{i}") + ".json", [base, base + 0.02, base - 0.02])
    _step_receipt(p("r4-step") + ".json", [10.50, 10.52, 10.48])
    json.dump({"convergence": {"a": 0.0412, "b": 0.0061, "t90_s": 11.4, "note": None},
               "swaps_per_drain": {"total": 310, "drains": 120, "max": 14}, "requests": [1, 2], "tok_s_median": 270.0},
              open(p("r1-c1") + ".json", "w"))
    json.dump({"ack": {"audit": {"mismatches": 0, "checked": 3000}}}, open(p("r4-audit1") + ".json", "w"))
    json.dump({"ack": {"audit": {"mismatches": 0, "checked": 2900}}}, open(p("r4-audit2") + ".json", "w"))
    json.dump({"ack": {"seq": 1}, "settled": True}, open(p("r4-restore") + ".json", "w"))
    json.dump({"summary": {"pass": True, "ok": 32, "n": 32, "server_alive_after": True}}, open(p("r5-c32") + ".json", "w"))
    base = {t: ["a", "b", t] for t in ("prose", "code", "math", "json", "shell", "tool", "long", "story")}
    _capture(p("r0-parity") + ".jsonl", base)
    _capture(p("r3-parity-a") + ".jsonl", dict(base, code=["a", "x", "code"]))
    _capture(p("r3-parity-b") + ".jsonl", dict(base, code=["a", "x", "code"]))
    s = DRV.build_summary(S, st)
    assert s["verdict"] == ("GT3: R0 10.450 | R2a 9.850 | R2b 8.800 | T90 11.4 | cold share 0.0412 \u2192 0.0061 | "
                            "audit 0 mismatches | C32 pass")
    assert s["R2b"]["step_ms_reps"] == pytest.approx([8.8, 8.82, 8.78])                # the fifth cell's three reps
    assert s["ratios"] == {"R2a_over_R0": 0.9426, "R2b_over_R0": 0.8421, "R4_over_R0": 1.0048}
    assert s["R3"]["self"] == {"identical": 8, "prompts": 8, "differ": []}
    assert s["R3"]["vs_r0_map"] == {"identical": 7, "prompts": 8, "differ": ["code"]}
    assert s["adopt"]["adopt"] is True and all(v is True for v in s["adopt"]["criteria"].values())
    assert s["R0"]["ms_per_token"] == round(10.45 / 2.75, 3)
    # each criterion can fail on its own
    _step_receipt(p("r4-step") + ".json", [10.80, 10.80, 10.80])        # 3.3 % off R0
    json.dump({"ack": {"audit": {"mismatches": 2}}}, open(p("r4-audit2") + ".json", "w"))
    s = DRV.build_summary(S, st)
    assert s["adopt"]["adopt"] is False and s["adopt"]["criteria"]["R4 within 2 % of R0"] is False
    assert s["adopt"]["criteria"]["audits 0"] is False and "audit 2 mismatches" in s["verdict"]
    assert DRV.main(["summary", "--scratch", S, "--stamp", st, "--out", p("summary") + ".json"]) == 0
    assert json.load(open(p("summary") + ".json"))["verdict"] == s["verdict"]
    assert DRV.main(["compare", "--a", p("r3-parity-a") + ".jsonl", "--b", p("r3-parity-b") + ".jsonl", "--ref",
                     p("r0-parity") + ".jsonl", "--out", p("r3-compare") + ".json"]) == 0
    assert json.load(open(p("r3-compare") + ".json"))["vs_ref"]["identical"] == 7


def test_driver_reads_the_managers_real_telemetry(tmp_path):
    """The driver's acknowledgement / T90 reading against telemetry the manager itself wrote."""
    clock = {"t": 100.0}

    def tick():
        clock["t"] += 0.01
        return clock["t"]

    cfg = GR.ResConfig(mode="adaptive", freeze=True, dir=str(tmp_path), drain_n=4, min_abs=2.0)
    W = World(tmp=tmp_path, cfg=cfg, manager=False)
    W.mgr = GR.GtResManager(W.views(), W.hot, W.cold, W.cnt, cfg, log=W.logs.append, clock=tick)
    rng = random.Random(1)
    popular = {l: list(W.cold[l]) for l in W.layers}
    _traffic(W, popular, 12, rng)
    seq = DRV.write_ctrl(str(tmp_path), dict(DRV.UNFREEZE))
    tele = str(tmp_path / "telemetry.jsonl")
    for _ in range(200):
        _traffic(W, popular, 1, rng)
    rows = DRV.read_telemetry(tele)
    ack = DRV.find_ack(rows, seq)
    assert ack is not None and ack["freeze"] is False
    series = DRV.cold_share_series(DRV.drains_after(rows, ack["t"]))
    c = DRV.t90(series, ack["t"])
    # plateau of this traffic: the 5 popular experts are in HBM, 2 picks go to the 7 others of which 5 are now cold
    assert c["a"] > 0.6 and abs(c["b"] - (2 * 5 / 7) / 6) < 0.06
    assert c["t90_s"] is not None and 0 < c["t90_s"] < series[-1][0] - ack["t"]
    W.check()


# ================================================================================================ the card
def test_card_res_mode_text():
    s = open(CARD).read()
    assert subprocess.run(["bash", "-n", CARD]).returncode == 0
    # the key redaction, in every mode, before the up log is read and again at finish
    assert "sed -E -i 's/(API key \\(save it\\): )[0-9a-f]+/\\1<redacted>/' \"$S/gt1-up-$STAMP.log\"" in s
    i_up = s.index('bash "$GT/launch-gt1.sh" up')
    assert s.index("redact_key() {") < i_up < s.index("\nredact_key   #") < s.index('say "up rc=$urc')
    fin = s[s.index("finish() {"):s.index("trap finish EXIT")]
    assert "redact_key" in fin
    # the build copies the two runtime files and writes the config only in the residency mode
    assert 'RES_RT="gracetier_res residency_policy"' in s and "ckpt $RES_RT; do" in s
    assert 'printf \'%s\\n\' "$GT1_RES_CONFIG" > "$GT/tree/fastpath/gracetier-res.json"' in s
    assert "GT1_RES_CONFIG=${GT1_RES_CONFIG:-'{\"mode\": \"adaptive\", \"freeze\": true}'}" in s
    assert 'RESDIR=$GT/home/state/prof/gracetier-res' in s
    # the residency block: after G1 and the nsys mode, before G2; it exits
    i_g1 = s.index("# ---------------------------------------------------------------------------------------------- G1")
    i_a2 = s.index('if [ "${GT1_NSYS:-0}" = 1 ]; then\n  say "== A2 nsys anatomy')
    i_res = s.index('if [ "${GT1_RES:-0}" = 1 ]; then\n  say "== card GT3')
    i_g2 = s.index('say "== G2 greedy fixture parity"')
    assert i_g1 < i_a2 < i_res < i_g2
    blk = s[i_res:i_g2]
    assert blk.rstrip().endswith("G3=res-ran\n  exit 0\nfi\n\n# ---------------------------------------------------------------------------------------------- G2".rstrip()) \
        or "G3=res-ran\n  exit 0\nfi" in blk
    cells = [blk.index(f'say "== {c} ') for c in ("R0", "R1", "R2a", "R2b", "R3", "R4", "R5")]
    assert cells == sorted(cells)
    for needle in ("stepcell r0-step", "capture r0-parity", "c1 --dir \"$RESDIR\" --seconds 90 --tokens 256",
                   "ctl r2a-ctrl '{\"freeze\": true}' fixture idle", "stepcell r2a-step",
                   "ctl r2b-ctrl '{\"freeze\": false}' step ack", 'for i in 1 2 3 4 5; do stepcell "r2b-step$i"; done',
                   "ctl r3-ctrl '{\"freeze\": true}' step idle", "capture r3-parity-a", "capture r3-parity-b",
                   'ctl r4-audit1 "$AUDIT" step ack', "ctl r4-restore '{\"freeze\": true, \"restore_map\": true}' step boot 900",
                   'ctl r4-audit2 "$AUDIT" step ack', "stepcell r4-step", "ctl r5-ctrl '{\"freeze\": false}' fixture ack",
                   "bench/c32_smoke.py --base \"$BASE\" --model \"$SERVED\" --n 32 --tokens 128",
                   "bench/gt_res_drive.py summary --scratch \"$S\" --stamp \"$STAMP\"",
                   "AUDIT='{\"audit\": {\"moved\": true, \"random\": 64}}'", "gracetier: residency: manager over"):
        assert needle in blk, needle
    assert json.loads(re.search(r"AUDIT='(.*)'", blk).group(1)) == {"audit": {"moved": True, "random": 64}}
    # the plain gates and the nsys mode are untouched text
    for keep in ('timeout 1800 python3 w0/w0_drive.py parity "$S/gt1-parity-$STAMP.jsonl"',
                 'timeout 1800 python3 w0/w0_drive.py step "$S/gt1-step-$STAMP.json"',
                 'timeout 300 python3 w0/w0_drive.py step "$S/gt1-a2-step-$STAMP.json" --profile',
                 "06-nsys-wrap-serve.patch", "say \"== G5 prefill / TTFT"):
        assert keep in s, keep
    # the file names the card writes are the ones the summary reads
    names = set(re.findall(r'(?:stepcell|capture|ctl) "?(r[0-9][a-z0-9$-]*)', blk)) | {"r1-c1", "r5-c32"}
    assert {"r0-step", "r0-parity", "r2a-step", "r2b-step$i", "r3-parity-a", "r3-parity-b", "r4-audit1", "r4-restore",
            "r4-audit2", "r4-step", "r1-c1", "r5-c32"} <= names
    drv = open(os.path.join(REPO, "bench", "gt_res_drive.py")).read()
    for n in ("r0-step", "r2a-step", "r4-step", "r1-c1", "r4-audit1", "r4-audit2", "r4-restore", "r5-c32", "r0-parity",
              "r3-parity-a", "r3-parity-b"):
        assert f'"{n}"' in drv, n
    assert 'f"r2b-step{i}"' in drv


def test_card_default_config_is_accepted_by_the_component_and_the_manager(tmp_path):
    s = open(CARD).read()
    cfg_txt = re.search(r"GT1_RES_CONFIG=\$\{GT1_RES_CONFIG:-'(.*)'\}", s).group(1)
    p = tmp_path / "gracetier-res.json"
    p.write_text(cfg_txt + "\n")
    G = _component()
    doc = G.res_doc(str(p))
    assert doc == {"mode": "adaptive", "freeze": True}
    c = GR.load_config(str(p))
    assert c.freeze is True and c.dir == "/prof/gracetier-res" and c.swaps_per_barrier == 4
    # the preflight's own check of the config (run as the card runs it)
    code = re.search(r"python3 -c '(import ast, json, os, sys\n.*?)' \\\n", s, re.S).group(1)
    r = subprocess.run([sys.executable, "-c", code, os.path.join(REPO, "bench", "gt_res_drive.py"),
                        os.path.join(REPO, "runtime", "gracetier_res.py")], env=dict(os.environ, RESCFG=cfg_txt),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-c", code, os.path.join(REPO, "bench", "gt_res_drive.py"),
                        os.path.join(REPO, "runtime", "gracetier_res.py")],
                       env=dict(os.environ, RESCFG='{"mode": "off"}'), capture_output=True, text=True)
    assert r.returncode != 0
