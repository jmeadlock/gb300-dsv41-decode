"""Mac CPU contracts of the P1 gracetier first build (docs/P1-GRACETIER-DESIGN.md section 5 file 6):
`.venv/bin/python -m pytest -q tests/test_gracetier_cpu.py`.

  * VMM geometry (runtime/gracetier_vmm.py): offsets, straddle granule, shared range, the HBM budget at real shapes;
    the arena + GroupedVA + LayerVMM driven by a byte-emulating fake driver (write through one layer's VA, read
    through the other's; private straddle; reverse release order); the ctypes structs' sizes (cuda.h layout).
  * row_map (runtime/gracetier_pool.py): the vendored h255 rowmap is 40 layers x 255 / 129; row_map is a partition
    (a permutation of 0..383), every entry >= 0; Garner's hook form converts; 30,960 cold tensors.
  * the Grace pool layout == the vendored deep_gemm transform_weights_for_mega_moe on mini shapes (executed from the
    vendored source, no deep_gemm import), the SF banks too; chunk padding; reads through runtime/ckpt.py.
  * gt_stage (runtime/gracetier_stage.py): stage_emulated's bytes (only touched groups, counters) and the Triton
    kernel bodies (gt_stage_kernel, gt_read_kernel) executed under tests/triton_shim.py == their contracts.
  * the probe's pure parts (bench/gracetier_minitest.py): synthetic routings with exactly c distinct cold experts,
    bit comparisons, fits, the P1.1 bar, the receipt merge / verdict line, the guarded router import, the parser,
    no vllm import anywhere; the box runner's guards (text).
"""
from __future__ import annotations

import ast
import ctypes
import json
import os
import random
import sys

import pytest
import torch

from runtime import dg_mega as DM
from runtime import gracetier_pool as GP
from runtime import gracetier_stage as GS
from runtime import gracetier_vmm as GV

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
import gracetier_minitest as GT  # noqa: E402

VENDORED = os.path.join(REPO, "reference", "vllm-0.31.0", "deep_gemm", "mega", "__init__.py")
ROWMAP = GP.default_rowmap_path(REPO)
MiB = 1024 ** 2


def _g(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _i8(shape, g):
    return torch.randint(-128, 128, shape, generator=g, device="cpu", dtype=torch.int32).to(torch.int8)


def _u8(shape, g, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device="cpu", dtype=torch.int32).to(torch.uint8)


# ================================================================================================ VMM geometry
def test_plan_offsets_straddle_and_pieces():
    p = GV.TensorPlan("t", G=8, H=3, B=40, gran=64)
    assert (p.hot_bytes, p.va_bytes, p.shared_va_off, p.private_bytes, p.shared_bytes) == (120, 320, 128, 128, 192)
    assert p.straddle and p.straddle_off == 64 and p.straddle_bytes == 8
    assert p.straddle_groups() == [1, 2, 3]
    assert p.where(127) == ("private", 127) and p.where(128) == ("shared", 0) and p.where(319) == ("shared", 191)
    with pytest.raises(ValueError):
        p.where(320)
    assert p.group_pieces(3) == [("private", 120, 8), ("shared", 0, 32)]
    assert p.group_pieces(2) == [("private", 80, 40)] and p.group_pieces(4) == [("shared", 32, 40)]
    cover = []
    for g in range(p.G):
        pcs = p.group_pieces(g)
        assert sum(n for _, _, n in pcs) == p.B
        cover += [(k, o, n) for k, o, n in pcs]
    priv = sorted((o, n) for k, o, n in cover if k == "private")
    shar = sorted((o, n) for k, o, n in cover if k == "shared")
    assert priv[0][0] == 0 and all(a + n == b for (a, n), (b, _) in zip(priv, priv[1:]))
    assert shar[0][0] == 0 and all(a + n == b for (a, n), (b, _) in zip(shar, shar[1:]))
    assert priv[-1][0] + priv[-1][1] == p.hot_bytes + p.straddle_bytes


def test_plan_aligned_and_edge_cases():
    a = GV.TensorPlan("a", G=6, H=2, B=64, gran=64)
    assert not a.straddle and a.straddle_off is None and a.straddle_groups() == [] and a.shared_va_off == 128
    z = GV.TensorPlan("z", G=4, H=0, B=48, gran=64)
    assert z.private_bytes == 0 and z.shared_va_off == 0 and z.shared_bytes == z.va_bytes == 192
    f = GV.TensorPlan("f", G=4, H=4, B=48, gran=64)
    assert f.shared_bytes == 0 and not f.straddle and f.private_bytes == f.va_bytes
    for bad in (dict(G=4, H=5, B=8, gran=8), dict(G=4, H=1, B=0, gran=8), dict(G=4, H=1, B=8, gran=0)):
        with pytest.raises(ValueError):
            GV.TensorPlan("bad", **bad)


def test_real_shapes_geometry_and_budget():
    """H = 255 / G = 384 at the real shapes with 2 MiB granules: the hot / cold boundary straddles a granule for L1
    (255 x 11,796,480 B is not a multiple of 2 MiB, design section 2.2); the shared range holds the 129 cold groups
    (+ < 1 granule); the 40-layer straddle pages <= 2 granules x 2 tensors x 40."""
    gran = 2 * MiB
    p = GV.plan_layer(384, 255, 5120, 2304, gran)
    assert p["l1"].B == 11_796_480 and p["l2"].B == 5_898_240 and p["l1"].B + p["l2"].B == 17_694_720
    assert p["l1"].straddle
    for tp in p.values():
        cold = (tp.G - tp.H) * tp.B
        assert cold - gran < tp.shared_bytes <= cold
        assert tp.shared_bytes + tp.straddle_bytes >= cold
        assert tp.straddle_bytes < gran
    b = GV.hbm_budget([255] * 40, 384, 5120, 2304, gran)
    assert b["straddle_B"] <= 2 * gran * 2 * 40
    assert b["shared_total_B"] <= 129 * 17_694_720
    assert b["sf_all_groups_B"] == 40 * 384 * (2 * 2304 * 5120 // 32 + 5120 * 2304 // 32)
    mixed = GV.shared_sizes([GV.plan_layer(384, h, 5120, 2304, gran) for h in (255, 280)])
    assert mixed["l1"] == GV.plan_layer(384, 255, 5120, 2304, gran)["l1"].shared_bytes   # the largest range


def test_ctypes_structs_match_cuda_h_layout():
    assert ctypes.sizeof(GV.CUmemLocation) == 8
    assert ctypes.sizeof(GV.CUmemAllocationProp) == 32 and GV.CUmemAllocationProp.win32HandleMetaData.offset == 16
    assert ctypes.sizeof(GV.CUmemAccessDesc) == 12


class FakeDriver:
    """VMM semantics in host memory: physical allocations are bytearrays, the VA space a list of mappings."""

    def __init__(self, gran=64):
        self.gran, self.next_va, self.next_h = gran, 1 << 20, 1
        self.phys, self.maps, self.reserved, self.calls = {}, [], [], []

    def granularity(self, device, recommended=True):
        return self.gran

    def create(self, n, device):
        assert n % self.gran == 0
        h = self.next_h
        self.next_h += 1
        self.phys[h] = bytearray(n)
        self.calls.append(("create", h))
        return h

    def reserve(self, n, alignment=0):
        va = -(-self.next_va // self.gran) * self.gran
        self.next_va = va + n + self.gran            # a hole between reservations
        self.reserved.append((va, n))
        self.calls.append(("reserve", va))
        return va

    def map(self, va, n, off, h):
        assert all(va + n <= a or a + m <= va for a, m, _, _ in self.maps), "overlapping map"
        assert off + n <= len(self.phys[h])
        self.maps.append((va, n, off, h))
        self.calls.append(("map", va))

    def set_access(self, va, n, device):
        self.calls.append(("access", va))

    def unmap(self, va, n):
        m = next(x for x in self.maps if x[0] == va and x[1] == n)
        self.maps.remove(m)
        self.calls.append(("unmap", va))

    def address_free(self, va, n):
        assert not any(va <= a < va + n for a, _, _, _ in self.maps), "free of a reservation still mapped"
        self.reserved.remove((va, n))
        self.calls.append(("free", va))

    def release(self, h):
        assert not any(x[3] == h for x in self.maps), "release of a mapped handle"
        del self.phys[h]
        self.calls.append(("release", h))

    def _at(self, va):
        for a, n, off, h in self.maps:
            if a <= va < a + n:
                return h, off + va - a
        raise KeyError(f"VA {va:#x} not mapped")

    def write(self, va, data: bytes):
        for i, b in enumerate(data):
            h, o = self._at(va + i)
            self.phys[h][o] = b

    def read(self, va, n) -> bytes:
        out = bytearray(n)
        for i in range(n):
            h, o = self._at(va + i)
            out[i] = self.phys[h][o]
        return bytes(out)


def test_two_layers_share_the_scratch_with_private_straddle():
    drv = FakeDriver(64)
    ar = GV.VmmArena(0, drv)
    plan = GV.TensorPlan("l1", G=8, H=3, B=40, gran=64)
    shared = ar.alloc(plan.shared_bytes, "shared")
    A = GV.GroupedVA(ar, plan, shared, map_hot=True, name="A")
    B = GV.GroupedVA(ar, plan, shared, map_hot=False, name="B")
    assert B.readable_from == plan.straddle_off
    # a cold group written through A is read back through B (groups 4..7 are wholly in the shared range)
    for g in range(4, 8):
        data = bytes((g * 7 + i) % 256 for i in range(plan.B))
        drv.write(A.va + g * plan.B, data)
        assert drv.read(B.va + g * plan.B, plan.B) == data
    # the straddle granule is per layer: group 3's head (in it) written via A is NOT seen via B, the tail is
    drv.write(B.va + 3 * plan.B, b"\xbb" * plan.B)
    drv.write(A.va + 3 * plan.B, b"\xaa" * plan.B)
    head = plan.shared_va_off - 3 * plan.B
    assert drv.read(B.va + 3 * plan.B, head) == b"\xbb" * head
    assert drv.read(B.va + plan.shared_va_off, plan.B - head) == b"\xaa" * (plan.B - head)
    with pytest.raises(KeyError):                              # B's hot groups are not mapped
        drv.read(B.va, 1)
    with pytest.raises(ValueError):
        B.bytes_view(0, 8)


def test_layer_vmm_groups_land_where_the_plan_says_and_release_reverses():
    drv = FakeDriver(64)
    ar = GV.VmmArena(0, drv)
    D, I, G = 32, 8, 12                                         # L1 group = 2*8*16 = 256 B, L2 = 32*4 = 128 B
    lv = GV.LayerVMM(ar, {0: 5, 1: 7}, G, D, I)
    assert set(lv.shared) == {"l1", "l2"}
    for k in ("l1", "l2"):
        assert lv.shared[k].nbytes == max(lv.plans[l][k].shared_bytes for l in (0, 1))
    for l in (0, 1):
        for k, tp in lv.plans[l].items():
            va = lv.va[l][k].va
            for g in range(G):
                data = bytes([(l * 31 + g) % 256]) * tp.B
                drv.write(va + g * tp.B, data)
                for kind, off, n in tp.group_pieces(g):
                    h = lv.va[l][k].private.handle if kind == "private" else lv.shared[k].handle
                    assert bytes(drv.phys[h][off:off + n]) == data[:n]
                    break
    n_maps = len(drv.maps)
    assert n_maps == 2 * 2 * 2
    ar.release()
    ops = [c[0] for c in drv.calls]
    i_unmap, i_free, i_rel = ops.index("unmap"), ops.index("free"), ops.index("release")
    assert i_unmap < i_free < i_rel and "map" not in ops[i_unmap:] and "unmap" not in ops[i_free:]
    assert not drv.maps and not drv.reserved and not drv.phys
    maps_order = [c[1] for c in drv.calls if c[0] == "map"]
    unmaps_order = [c[1] for c in drv.calls if c[0] == "unmap"]
    assert unmaps_order == list(reversed(maps_order))
    ar.release()                                                # idempotent
    assert ops == [c[0] for c in drv.calls]


def test_arena_refuses_misaligned_and_foreign_maps():
    drv = FakeDriver(64)
    ar = GV.VmmArena(0, drv)
    a = ar.alloc(100, "a")
    assert a.nbytes == 128
    va = ar.reserve(128)
    with pytest.raises(ValueError):
        ar.map(va + 8, 64, a)
    with pytest.raises(ValueError):
        ar.map(va, 192, a)
    with pytest.raises(ValueError):
        ar.map(va + (1 << 30), 64, a)
    with pytest.raises(ValueError):                              # shared scratch too small for the plan
        GV.GroupedVA(ar, GV.TensorPlan("x", G=8, H=3, B=40, gran=64), a)
    ar.release()


# ================================================================================================ row_map
def test_vendored_rowmap_and_row_map_properties():
    rm = GP.load_rowmap(ROWMAP)
    assert sorted(rm) == list(range(40)) and GP.rowmap_meta(ROWMAP).get("n_hot") == 255
    for l, v in rm.items():
        assert len(v["hot"]) == 255 and len(v["cold"]) == 129
        row = GP.row_map(v["hot"], v["cold"])
        props = GP.row_map_properties(row, 255)
        assert props == {"n": 384, "all_ge_0": True, "permutation": True, "n_hot": 255, "n_cold": 129, "H": 255}
        assert all(row[e] == s for s, e in enumerate(v["hot"]))
        assert all(row[e] == 255 + c for c, e in enumerate(v["cold"]))
        assert GP.row_map_from_hk(GP.hk_row_map(v["hot"], v["cold"]), 255) == row
        inv = GP.group_experts(row)
        assert all(row[inv[g]] == g for g in range(384))
    assert len(GP.cold_tensor_names(rm)) == 30_960                 # fastpath_coldskip.py:9 (40 x 129 x 6)


def test_load_rowmap_refuses_a_non_partition(tmp_path):
    bad = {"layers": {"0": {"hot": list(range(200)), "cold": list(range(199, 384))}}}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="partition"):
        GP.load_rowmap(str(p))
    with pytest.raises(ValueError):
        GP.row_map(list(range(10)), list(range(10, 20)), E=24)


# ================================================================================================ pool layout
def test_pool_layout_chunk_padding_and_ranges():
    D, I = 256, 128
    b = GP.l1_bytes(D, I) + GP.l2_bytes(D, I)                    # 32768 + 16384
    L = GP.PoolLayout({0: 3, 1: 3, 2: 1}, D, I, chunk=4 * b, align=4096)
    assert L.l1_off[0] == 0 and L.l2_off[0] == 3 * L.B1
    assert L.l1_off[1] == 4 * b                                  # 3 + 3 experts would cross the chunk: moved
    for l, C in L.n_cold.items():
        lo, hi = L.l1_off[l], L.l1_off[l] + C * b
        assert L.chunk_of(lo) == L.chunk_of(hi - 1)
        for c in range(C):
            (a0, a1), (b0, b1) = L.expert_ranges(l, c)
            assert a1 - a0 == L.B1 and b1 - b0 == L.B2 and L.chunk_of(a0) == L.chunk_of(b1 - 1)
    assert L.pad_bytes == L.l1_off[1] - 3 * b + (L.l1_off[2] - (L.l1_off[1] + 3 * b))
    with pytest.raises(ValueError):
        GP.PoolLayout({0: 5}, D, I, chunk=4 * b, align=4096)


def test_pool_layout_real_shapes_fit_the_64gib_chunks():
    L = GP.PoolLayout({l: 129 for l in range(40)})
    per = 129 * 17_694_720
    assert L.total >= 40 * per and L.pad_bytes < L.chunk
    for l in range(40):
        assert L.chunk_of(L.l1_off[l]) == L.chunk_of(L.l2_off[l] + 129 * L.B2 - 1)


def _vendored(*names):
    tree = ast.parse(open(VENDORED).read())
    from typing import Optional, Tuple, Union

    ns = {"torch": torch, "Tuple": Tuple, "Union": Union, "Optional": Optional}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module([n], []), VENDORED, "exec"), ns)
        elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_SUPPORTED_ACTIVATIONS" for t in n.targets):
            exec(compile(ast.Module([n], []), VENDORED, "exec"), ns)
    return [ns[n] for n in names]


def _mini_experts(C, D, I, seed):
    g = _g(seed)
    w = {p: _i8((C, I, D // 2) if p != "w2" else (C, D, I // 2), g) for p in GP.PROJ}
    s = {p: _u8((C, I, D // 32) if p != "w2" else (C, D, I // 32), g, 100, 140) for p in GP.PROJ}
    return w, s


def test_pool_layout_equals_vendored_transform_weights_for_mega_moe():
    (transform,) = _vendored("_validate_activation", "_interleave_weights", "_transpose_sf_for_utccp",
                             "transform_weights_for_mega_moe")[-1:]
    D, I, C = 256, 128, 3
    w, s = _mini_experts(C, D, I, 1)
    pool = GP.GracePool(GP.PoolLayout({7: C}, D, I))
    for c in range(C):
        pool.fill_expert(7, c, w["w1"][c], w["w3"][c], w["w2"][c])
    sf13 = torch.cat([DM.sf_words(s["w1"]), DM.sf_words(s["w3"])], 1)
    (tl1, tsf1), (tl2, tsf2) = transform((torch.cat([w["w1"], w["w3"]], 1), sf13), (w["w2"], DM.sf_words(s["w2"])))
    assert torch.equal(pool.host_l1(7), tl1) and torch.equal(pool.host_l2(7), tl2)
    for major in ("k", "mn"):
        sf1, sf2 = GP.sf_banks(s, major)
        assert torch.equal(sf1.contiguous(), tsf1) and torch.equal(sf2.contiguous(), tsf2)
    sf1, _ = GP.sf_banks(s, "mn")
    assert sf1.stride() == (2 * I * D // 128, 1, 2 * I)          # MN-major storage, as runtime/dg_mega builds it
    order = GP.group_order_codes({p: v[:1] for p, v in s.items()}, {p: v[1:] for p, v in s.items()})
    assert all(torch.equal(order[p], s[p]) for p in GP.PROJ)


def test_src_table_addresses_and_slot_remap():
    D, I, C = 256, 128, 4
    pool = GP.GracePool(GP.PoolLayout({0: 2, 3: C}, D, I))
    t = pool.src_table(3)
    assert t.dtype == torch.int64 and tuple(t.shape) == (C, 2)
    for c in range(C):
        (a, _), (b, _) = pool.layout.expert_ranges(3, c)
        assert t[c].tolist() == [pool.buf.ptr + a, pool.buf.ptr + b]
        assert int(t[c, 0]) % 16 == 0 and int(t[c, 1]) % 16 == 0
    r = pool.src_table(3, slots=[1, 2, 3, 0])
    assert torch.equal(r, t[[1, 2, 3, 0]])
    assert pool.host_l1(3).data_ptr() == pool.buf.ptr + pool.layout.l1_off[3]


def _write_ckpt(path, L, experts, w, s):
    from safetensors.torch import save_file

    t = {}
    for i, e in enumerate(experts):
        for p in GP.PROJ:
            wn, sn = GP.expert_names(L, e, p)
            t[wn] = w[p][i].contiguous()
            t[sn] = s[p][i].contiguous()
    save_file(t, os.path.join(path, "model-00001-of-00001.safetensors"))


def test_pool_reads_through_shardindex_equal_fill(tmp_path):
    from runtime.ckpt import ShardIndex

    D, I, L = 256, 128, 5
    cold = [11, 3, 7]
    w, s = _mini_experts(len(cold), D, I, 2)
    _write_ckpt(str(tmp_path), L, cold, w, s)
    idx = ShardIndex.scan(str(tmp_path))
    pool = GP.GracePool(GP.PoolLayout({L: len(cold)}, D, I))
    r = pool.read_layer(idx, L, cold)
    assert r["experts"] == 3 and r["tensors"] == 9 and r["bytes"] == 3 * (pool.layout.B1 + pool.layout.B2)
    ref = GP.GracePool(GP.PoolLayout({L: len(cold)}, D, I))
    for c in range(len(cold)):
        ref.fill_expert(L, c, w["w1"][c], w["w3"][c], w["w2"][c])
    assert torch.equal(pool.host_l1(L), ref.host_l1(L)) and torch.equal(pool.host_l2(L), ref.host_l2(L))
    codes = GP.read_codes(idx, L, cold, D, I)
    assert all(torch.equal(codes[p], s[p]) for p in GP.PROJ)
    with pytest.raises(ValueError):
        pool.read_layer(idx, L, cold[:2])
    idx.close()


# ================================================================================================ gt_stage
def _stage_world(D, I, C, H, seed):
    """A pool of C cold experts, destinations [G = H + C] with random bytes, a resolver over the pool."""
    w, _ = _mini_experts(C, D, I, seed)
    pool = GP.GracePool(GP.PoolLayout({0: C}, D, I))
    for c in range(C):
        pool.fill_expert(0, c, w["w1"][c], w["w3"][c], w["w2"][c])
    g = _g(seed + 1)
    G = H + C
    dsts = [_i8((G, 2 * I, D // 2), g), _i8((G, D, I // 2), g)]
    base = pool.buf.ptr

    def resolve(addr):
        return pool.buf.cpu[addr - base:]

    return pool, dsts, resolve


def test_stage_plan_and_emulated_contract():
    D, I, C, H = 256, 128, 5, 4
    pool, dsts, resolve = _stage_world(D, I, C, H, 3)
    before = [t.clone() for t in dsts]
    groups = torch.tensor([[0, 4, 6, -1], [4, 3, 8, 1]], dtype=torch.int64, device="cpu")
    assert GS.stage_plan(groups.reshape(-1).tolist(), H, C) == {0: 2, 2: 1, 4: 1}
    cnt = torch.zeros(C, dtype=torch.int32, device="cpu")
    plan = GS.stage_emulated(groups, H, pool.src_table(0), dsts, cnt, resolve)
    assert plan == {0: 2, 2: 1, 4: 1} and cnt.tolist() == [2, 0, 1, 0, 1]
    for c in range(C):
        g_ = H + c
        if c in plan:
            assert torch.equal(dsts[0][g_], pool.host_l1(0)[c]) and torch.equal(dsts[1][g_], pool.host_l2(0)[c])
        else:
            assert torch.equal(dsts[0][g_], before[0][g_]) and torch.equal(dsts[1][g_], before[1][g_])
    assert torch.equal(dsts[0][:H], before[0][:H]) and torch.equal(dsts[1][:H], before[1][:H])


def _stage_under_shim():
    import triton_shim

    mod = triton_shim.load_module_under_shim(os.path.join(REPO, "runtime", "gracetier_stage.py"),
                                             "gracetier_stage_under_shim")
    return mod, triton_shim


@pytest.mark.parametrize("I,D,nprog,bw", [(128, 256, 1, 1024), (128, 256, 3, 512), (128, 384, 2, 4096)])
def test_stage_kernel_body_equals_contract(I, D, nprog, bw):
    """gt_stage_kernel program by program == stage_emulated: exits for untouched slots, full and partial blocks,
    extra trips masked, both projections, the counters."""
    mod, shim = _stage_under_shim()
    C, H = 4, 3
    groups = torch.tensor([[3, 0, 6, -1, 2, 5], [6, 1, 2, 0, 3, -1]], dtype=torch.int64, device="cpu")
    outs = []
    for use_kernel in (False, True):
        pool, dsts, resolve = _stage_world(D, I, C, H, 9)
        src = pool.src_table(0)
        cnt = torch.zeros(C, dtype=torch.int32, device="cpu")
        if use_kernel:
            shim.ADDRESSES.clear()
            shim.ADDRESSES.register(pool.buf.cpu, *dsts)
            try:
                grid = mod.gt_stage(groups.reshape(-1).contiguous(), H, src, GS.make_dtab(*dsts), cnt,
                                    pool.layout.B1, pool.layout.B2, nprog=nprog, block_words=bw)
            finally:
                shim.ADDRESSES.clear()
            assert grid == (C, nprog) and mod.gt_stage_kernel.launches[-1]["grid"] == (C, nprog)
        else:
            GS.stage_emulated(groups, H, src, dsts, cnt, resolve)
        outs.append([t.clone() for t in dsts] + [cnt.clone()])
    for i, (a, b) in enumerate(zip(*outs)):
        assert torch.equal(a, b), i
    assert outs[1][2].tolist() == [2, 0, 1, 2]                    # groups 3 x2, 5 x1, 6 x2 (H = 3)


def test_stage_kernel_without_counter_and_refusals():
    mod, shim = _stage_under_shim()
    D, I, C, H = 256, 128, 2, 2
    pool, dsts, _ = _stage_world(D, I, C, H, 4)
    src, dtab, b1, b2 = pool.src_table(0), GS.make_dtab(*dsts), pool.layout.B1, pool.layout.B2

    def ids(v, dt=torch.int64):
        return torch.tensor(v, dtype=dt, device="cpu")

    shim.ADDRESSES.register(pool.buf.cpu, *dsts)
    try:
        mod.gt_stage(ids([2, 3]), H, src, dtab, None, b1, b2, nprog=2)
    finally:
        shim.ADDRESSES.clear()
    assert torch.equal(dsts[0][2], pool.host_l1(0)[0]) and torch.equal(dsts[1][3], pool.host_l2(0)[1])
    with pytest.raises(ValueError):
        mod.gt_stage(ids([2], torch.int32), H, src, dtab, None, b1, b2)
    with pytest.raises(ValueError):
        mod.gt_stage(ids([2]), H, src, dtab, None, b1, b2, block_words=3)
    with pytest.raises(ValueError):
        GS.words(24)
    with pytest.raises(RuntimeError, match="triton"):
        if GS.HAVE_TRITON:
            raise RuntimeError("triton present: refusal not applicable")
        GS.gt_stage(ids([2]), H, src, dtab, None, 16, 16)


@pytest.mark.parametrize("nprog,bw,n_src", [(1, 512, 2), (3, 1024, 3), (5, 4096, 1)])
def test_read_kernel_body_equals_contract(nprog, bw, n_src):
    mod, shim = _stage_under_shim()
    D, I, C = 384, 128, 3
    pool, _, resolve = _stage_world(D, I, C, 1, 5)
    src = pool.src_table(0)
    out = torch.zeros(nprog * bw, dtype=torch.int64, device="cpu")
    shim.ADDRESSES.register(pool.buf.cpu)
    try:
        mod.gt_read(src, n_src, pool.layout.B1, out, nprog=nprog, block_words=bw)
    finally:
        shim.ADDRESSES.clear()
    want = GS.read_emulated(src, n_src, pool.layout.B1 // 8, nprog, bw, resolve)
    assert torch.equal(out.view(nprog, bw), want)
    assert bool(want.ne(0).any())


# ============================================================================================ the probe's pure parts
@pytest.mark.parametrize("T", GT.TS)
def test_synth_routing_forces_exactly_c_distinct_cold(T):
    rm = GP.load_rowmap(ROWMAP)[10]
    row = GP.row_map(rm["hot"], rm["cold"])
    rng = random.Random(T)
    for c in GT.CS:
        ids = GT.synth_routing(T, c, rm["hot"], rm["cold"], 6, rng)
        if c > T * 6:
            assert ids is None
            continue
        assert len(ids) == T and all(len(r) == 6 and len(set(r)) == 6 and r == sorted(r) for r in ids)
        assert GT.distinct_cold(ids, row, 255) == c
        groups = [[row[e] for e in r] for r in ids]
        assert len({g for r in groups for g in r if g >= 255}) == c
    w = GT.synth_weights(T, 6, rng)
    assert all(abs(sum(r) - GT.ROUTE_SCALE) < 1e-9 and min(r) > 0 for r in w)


def test_compare_bits_and_fit_line():
    a = torch.randn(4, 64, generator=_g(0), dtype=torch.float32, device="cpu").to(torch.bfloat16)
    assert GT.compare_bits(a, a.clone()) == {"equal": True, "rows": 4, "rows_exact": 4, "frac_bytes_equal": 1.0,
                                             "max_abs": 0.0, "nonfinite": 0}
    b = a.clone()
    b[1, 3] = b[1, 3] * 2 + 1
    r = GT.compare_bits(a, b)
    assert not r["equal"] and r["rows_exact"] == 3 and r["frac_bytes_equal"] == round(1 - 1 / 256, 6)
    assert r["max_abs"] > 0
    nan = a.clone()
    nan[0, 0] = float("nan")
    assert GT.compare_bits(nan, nan.clone())["equal"] and GT.compare_bits(nan, nan)["nonfinite"] == 2  # bits
    f = GT.fit_line([0, 1, 2, 4], [3.0, 5.0, 7.0, 11.0])
    assert f == {"slope": 2.0, "intercept": 3.0, "n": 4}
    assert GT.fit_line([1, 1], [2.0, 3.0]) is None and GT.fit_line([0, 1], [None, 1.0]) is None
    assert GT.gbps(1e6, 10.0) == 100.0 and GT.gbps(1e6, None) is None


def test_m5_summary_bar_and_b_serial():
    table = [{"T": 6, "c": c, "stage_us": 3.0 + 46.0 * c, "mega384_us": 60.0 + 3.0 * c,
              "both_us": 63.0 + 49.0 * c} for c in (0, 1, 2, 4)]
    table += [{"T": 1, "c": c, "stage_us": 3.0 + 55.0 * c, "mega384_us": 50.0, "both_us": 53.0 + 55.0 * c}
              for c in (0, 1, 2)]
    s = GT.m5_summary(table, (1, 6, 16))
    assert s["6"]["stage_slope_us"] == 46.0 and s["6"]["bar_pass"] is True and s["6"]["mega_marginal_us"] == 3.0
    assert s["1"]["bar_pass"] is False and s["16"]["stage"] is None and s["16"]["bar_pass"] is None
    assert GT.b_serial_from_m5(table, 6) == {1: 52.0, 2: 101.0, 4: 199.0}
    assert GT.b_serial_from_m5(table, 16) == {}


def test_p11_bar_uses_only_given_measurements():
    r = GT.p11_bar(3.0, 18.0, 300.0, 60.0, {1: 50.0, 2: 100.0, 4: 200.0}, 17_694_720)
    a = {c: 3.0 + max(0.0, 18.0 + c * 17_694_720 / 300e3 - 60.0) for c in (1, 2, 4)}
    assert r["per_c"][1]["a_model_us"] == round(a[1], 2) and r["per_c"][4]["a_model_us"] == round(a[4], 2)
    assert r["status"] == ("pass" if all(a[c] <= 0.7 * b for c, b in ((1, 50.0), (2, 100.0), (4, 200.0)))
                           else "fail")
    miss = GT.p11_bar(None, 18.0, None, 60.0, {1: 50.0}, 1)
    assert miss["status"] == GT.NOT_MEASURED and set(miss["missing"]) == {"fork_join_us", "rate_GBps"}
    assert GT.p11_bar(1.0, 1.0, 1.0, 1.0, {}, 1)["status"] == GT.NOT_MEASURED
    timing = {"phases": {"M5": {"table": [{"T": 6, "c": 0, "mega255_us": 55.0}], "b_serial_us_T6": {"1": 50.0}},
                         "M6": {"fits": {"6": {"intercept_us": 18.0}}}}}
    m7 = {"fork_join_empty_us": 3.0, "headline": {"read_GBps_8_under_mega": 200.0, "stage_GBps_8_under_mega": None}}
    p = GT.p11_from(timing, m7, 17_694_720)
    assert p["status"] in ("pass", "fail")
    assert p["inputs"]["w_hot_us"] == 55.0 and p["inputs"]["cold_intercept_us"] == 18.0
    assert p["with_gt_stage_rate"]["status"] == GT.NOT_MEASURED
    assert GT.p11_from(None, m7, 1)["status"] == GT.NOT_MEASURED


def test_merge_and_verdict_line():
    parts = {"vmm": {"phases": {"M0": {}, "M1": {"gate": True}}, "errors": []},
             "numerics": {"phases": {"M2": {}, "M3": {}, "M4": {"gate": True}}, "errors": ["x"]},
             "timing": None, "r8": None}
    m = GT.merge_children(parts)
    assert set(m["phases"]) == {"M0", "M1", "M2", "M3", "M4"} and m["errors"] == ["numerics: x"]
    v = GT.verdict_line(m["phases"])
    assert v.startswith("GRACETIER-MINITEST: M1 torch.equal=True; M4 torch.equal all T=True")
    assert v.count(GT.NOT_MEASURED) >= 4
    v2 = GT.verdict_line({})
    assert "M1 torch.equal=not measured" in v2


def test_guarded_router_import_reports_not_run(tmp_path, monkeypatch):
    fn, why = GT.import_garner_router(str(tmp_path / "missing"))
    assert fn is None and why.startswith("not run:")
    d = tmp_path / "fp_broken"
    d.mkdir()
    (d / "fastpath_boundary.py").write_text("raise ImportError('no fastpath_hook here')\n")
    monkeypatch.delitem(sys.modules, "fastpath_boundary", raising=False)
    fn, why = GT.import_garner_router(str(d))
    assert fn is None and "import fastpath_boundary failed" in why and str(d) not in sys.path
    e = tmp_path / "fp_nolib"
    e.mkdir()
    (e / "fastpath_boundary.py").write_text("def router(*a, **k):\n    pass\n\ndef lib():\n"
                                            "    raise OSError('libw4boundary.so: cannot open shared object file')\n")
    monkeypatch.delitem(sys.modules, "fastpath_boundary", raising=False)
    fn, why = GT.import_garner_router(str(e))
    assert fn is None and why.startswith("not run: libw4boundary.so")
    ok = tmp_path / "fp_ok"
    ok.mkdir()
    (ok / "fastpath_boundary.py").write_text("def router(*a, **k):\n    return 'r'\n\ndef lib():\n    return 1\n")
    monkeypatch.delitem(sys.modules, "fastpath_boundary", raising=False)
    fn, mod = GT.import_garner_router(str(ok))
    assert fn() == "r"
    monkeypatch.delitem(sys.modules, "fastpath_boundary", raising=False)


def test_parser_defaults_and_child_argv_round_trip():
    ap = GT.build_parser()
    for sub in GT.SUBS + ("all",):
        a = ap.parse_args([sub])
        assert a.layer == 10 and a.json is None and a.fast_math == "both" and a.model_dir == "/ref"
        assert a.garner == "/garner/fastpath" and a.rowmap == ROWMAP and a.max_tokens == GT.MAX_TOKENS
    assert GT.default_json().startswith(os.path.join("scratch", "gracetier-minitest-"))
    a = ap.parse_args(["all", "--layer", "12", "--json", "scratch/x.json", "--iters", "50", "--ts", "1,6",
                       "--allow-busy"])
    c = ap.parse_args(GT.child_argv(a, "timing", "scratch/x.timing.json", ["--gate-json", "scratch/x.vmm.json"]))
    assert (c.cmd, c.layer, c.json, c.iters, c.ts, c.allow_busy, c.gate_json) == \
        ("timing", 12, "scratch/x.timing.json", 50, "1,6", True, "scratch/x.vmm.json")
    assert GT._ints("1, 6,16") == [1, 6, 16]
    assert GT.smi_used_mib("NVIDIA GB300, GPU-1c18, 1234 MiB, 254000 MiB, 0 %, 1965 MHz") == 1234.0
    assert GT.smi_used_mib("error: x") is None


def test_receipt_records_a_missing_deep_gemm_as_an_error(tmp_path, monkeypatch):
    """On the Mac `vmm` cannot import deep_gemm: the probe writes its receipt with the error and exits 2 (never a
    crash, never an invented number)."""
    monkeypatch.setenv("DSV41_DEEP_GEMM_PATH", str(tmp_path / "nowhere"))
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    out = tmp_path / "r.json"
    rc = GT.main(["vmm", "--json", str(out)])
    d = json.load(open(out))
    assert rc == GT.EXIT_ERRORS and d["exit"] == GT.EXIT_ERRORS and d["errors"] and d["phases"] == {}
    assert d["verdict"].startswith("GRACETIER-MINITEST: M1 torch.equal=not measured")


def test_no_vllm_import_in_the_new_files():
    files = [os.path.join(REPO, "runtime", f) for f in ("gracetier_vmm.py", "gracetier_pool.py",
                                                        "gracetier_stage.py")]
    files += [os.path.join(REPO, "bench", "gracetier_minitest.py"),
              os.path.join(REPO, "tests", "gpu", "test_gracetier_gpu.py")]
    for f in files:
        for n in ast.walk(ast.parse(open(f).read())):
            if isinstance(n, ast.Import):
                assert not any(a.name.split(".")[0] == "vllm" for a in n.names), f
            elif isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "vllm", f


def test_box_runner_guards():
    s = open(os.path.join(REPO, "bench", "box", "gracetier_minitest.sh")).read()
    assert 'source "$(dirname "$0")/lease.sh"' in s and "GRACETIER_WINDOW" in s
    assert s.index("GRACETIER_WINDOW:-") < s.index("lease.sh\"")          # refused before queueing on the lease
    assert "--network none" in s and "vllm/vllm-openai:v0.31.0" in s and ":/garner/fastpath:ro" in s
    assert "--layer" in s and "--json" in s and "LAYER=10" in s and "scratch/*)" in s
    assert "bench/gracetier_minitest.py all" in s and "tests/gpu/test_gracetier_gpu.py" in s
    for verb in ("docker stop", "docker start", "docker restart", "systemctl"):
        assert verb not in s, verb
    assert 'JSON="scratch/gracetier-minitest-' in s                          # receipts under scratch/, never docs/
