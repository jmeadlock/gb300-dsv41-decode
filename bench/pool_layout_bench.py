#!/usr/bin/env python3
"""pool_layout_bench.py -- M3-S3 (docs/HANDOFF-M3-S3-POOL-SF-LAYOUT.md, results docs/M3-S3-POOL-SF-LAYOUT.md): the
expert pool's scale factors "beside" their weights ($DSV41_POOL_SF=beside, runtime/expert_pool.py) vs today's
"separate" layout -- slot bytes, what DeepGEMM's SF addressing allows, the HBM it frees, the stage's per-layer-call
cost with and without the hot experts' SF, and the expected headline.

Subcommands (CPU unless marked; numbers are labeled [M] measured, [D] derived from receipts, [E] modeled, [A]
assumed):
  bytes     slot bytes per projection at the real shapes, alignment / pitch facts, the HBM accounting of the SF
            bytes today (headline arm C: K-major hot e8m0 + residency e8m0 + partial pool 17/40 layers + step SF) vs
            "beside" (one packed group per slot, e8m0 on Grace) at several --pool-spare values [D].
  estimate  the expected ms/step of the box arms from tonight's arms A / B / C (docs/headline-s2-*-20261108.json,
            re-read when present) and from the M3-S2 per-layer model [E]; the arithmetic is printed.
  contract  rows the staging contract reads per layer-call on a tiny pool in both layouts (runtime/dg_inplace.py
            stage_rows_emulated through a counting resolver; separate = the "step" store's lists, beside = the pool
            store's): beside never reads a hot scale row [M, CPU].
  probe     (GPU, box) A build: a real-shape ExpertPool (--layers layers x --hot hot experts, random bytes) in both
            layouts -- construct + fill + pack time, pool.hbm_bytes, the HBM the process holds (mem_get_info deltas);
            B gemm: DeepGEMM's contiguous grouped GEMM on beside's pool-owned packed SF == on a copy packed from the
            separate pool's e8m0 (bit-identical, w1 and w2 shapes); C stage: per-layer-call us of the staging launch
            with the hot SF listed (today's step layers) vs cold only (every layer under beside) -- reuses
            bench/staging_bytes_bench.py probe's serial + overlap sections [M].
  all       bytes + estimate + contract (+ probe with CUDA).

Box (library image, repo at /work, DG JIT cache mounted; no checkpoint):
  python3 bench/pool_layout_bench.py probe --out scratch/pool-layout-probe-<date>.json
deep_gemm comes from vllm's third_party WITHOUT importing vllm (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

import staging_bytes_bench as SB  # noqa: E402  (real shapes + receipts, one source)
from runtime import dg_inplace as DI  # noqa: E402

GiB = 2 ** 30
PROJS = ("w1", "w3", "w2")
L, H = SB.LAYERS, SB.HOT_SEED                          # 40 layers, 295 hot per layer (seed map)
E_W, E_S = SB.EXPERT_W, SB.EXPERT_S                    # 17,694,720 / 1,105,920 B per expert
# ---- tonight's arms (docs/headline-s2-{A,B,C}-20261108.json, two_graphs ms_per_step mean) [M]
R_ARMS = {"A": {"ms": 19.306, "pool_spare": 24, "res": 8, "pooled": 0, "free_after_build": 4.186},
          "B": {"ms": 19.216, "pool_spare": 24, "res": 8, "pooled": 7, "free_after_build": 1.777},
          "C": {"ms": 19.064, "pool_spare": 20, "res": 4, "pooled": 17, "free_after_build": 1.541}}
STAGE = 16                                             # staging slots at 6 rows (--spare-slots 16)
R_RES_SLOT_MS = (0.0, 0.02)   # two_graphs cell: ms/step per residency slot (M3-S2 (2): 0.02 average) [D]
M3S2_US_PER_LAYER = (33.1, 36.6)   # M3-S2's per-pooled-layer model (serial-derived, overlap residual) [D]


def _date() -> str:
    return time.strftime("%Y%m%d")


def arms() -> dict:
    """R_ARMS, re-read from docs/ when the receipts are there."""
    out = json.loads(json.dumps(R_ARMS))
    for k in out:
        d = SB._load(f"docs/headline-s2-{k}-20261108.json")
        if d:
            try:
                out[k]["ms"] = round(d["headline"]["two_graphs"]["ms_per_step"]["mean"], 3)
                out[k]["free_after_build"] = d["hbm_after_build_GiB"]["free"]
                out[k]["source"] = f"docs/headline-s2-{k}-20261108.json"
            except Exception:
                pass
    return out


# ------------------------------------------------------------------------------------------------ bytes
def slot_bytes() -> dict:
    """Per projection: weight bytes, e8m0 bytes (= packed int32 bytes), row pitches, DeepGEMM's group strides."""
    out = {}
    for p in PROJS:
        n, k = SB.NK[p]
        w_row, s_row = k // 2, k // 32
        out[p] = {"N": n, "K": k, "weight_B": n * w_row, "e8m0_B": n * s_row, "packed_B": n * (k // 128) * 4,
                  "B_row_pitch_B": w_row, "B_group_stride_B": n * w_row,
                  "SFB_row_pitch_B": n * 4, "SFB_rows_per_group": k // 128, "SFB_group_stride_B": (k // 128) * n * 4,
                  "sf_as_extra_B_rows": n * s_row / w_row,
                  "row_pitch_if_sf_per_row_B": w_row + s_row, "pitch_16B_ok": (w_row + s_row) % 16 == 0}
    return out


def sf_hbm_today(pool_spare: int, res: int, pooled: int) -> float:
    """[D] SF bytes in HBM under "separate" + the partial pool: hot K-major e8m0 (pool) + residency slots' e8m0
    (ResidencyManager.sspare) + `pooled` layers' pool-owned packed copy (H + pool_spare groups) + the step store's
    shared copy (one layer's groups, allocated while any layer is "step")."""
    G = H + pool_spare
    b = L * H * E_S + L * res * E_S + pooled * G * E_S + (G * E_S if pooled < L else 0)
    return b / GiB


def sf_hbm_beside(pool_spare: int) -> float:
    """[D] SF bytes in HBM under "beside": one packed group per slot (hot + spare), nothing else."""
    return L * (H + pool_spare) * E_S / GiB


def cmd_bytes(a) -> dict:
    sb = slot_bytes()
    ar = arms()
    c = ar["C"]
    today = sf_hbm_today(c["pool_spare"], c["res"], c["pooled"])
    rows = []
    for ps in (20, 24, 26, 27, 28):
        bes = sf_hbm_beside(ps)
        d_w = L * (ps - c["pool_spare"]) * E_W / GiB          # more weight slots than C
        delta = d_w + bes - today                              # HBM vs arm C (negative = freed)
        rows.append({"pool_spare": ps, "residency_slots": ps - STAGE, "sf_hbm_GiB": round(bes, 3),
                     "weights_vs_C_GiB": round(d_w, 3), "hbm_vs_C_GiB": round(delta, 3),
                     "free_after_build_GiB_E": round(c["free_after_build"] - delta, 2),
                     "host_e8m0_GiB": round((L * H + L * (ps - STAGE)) * E_S / GiB, 3)})
    out = {"slot": sb, "expert_weight_B": E_W, "expert_sf_B": E_S,
           "slot_today": {"hot": f"{E_W:,} B weights (HBM) + {E_S:,} B e8m0 (HBM); its packed SF re-made each step "
                                 f"(step layers) or a 2nd {E_S:,} B copy (pooled layers)",
                          "spare": f"{E_W:,} B weights (HBM); residency slots + {E_S:,} B e8m0 (HBM); packed SF in "
                                   f"the step / pool copy"},
           "slot_beside": f"{E_W:,} B weights + {E_S:,} B packed SF, both HBM, both at slot index g (two tensors: "
                          f"hbm_full[l][p][g], sfp[l][p][g]); K-major e8m0 of hot slots on Grace",
           "today_C": {"pool_spare": c["pool_spare"], "res": c["res"], "pooled": c["pooled"],
                       "sf_hbm_GiB": round(today, 3),
                       "parts_GiB": {"hot_e8m0": round(L * H * E_S / GiB, 3),
                                     "residency_e8m0": round(L * c["res"] * E_S / GiB, 3),
                                     "partial_pool": round(c["pooled"] * (H + c["pool_spare"]) * E_S / GiB, 3),
                                     "step_sf": round((H + c["pool_spare"]) * E_S / GiB, 3)},
                       "free_after_build_GiB_M": c["free_after_build"]},
           "beside": rows, "residency_slot_GiB": round(L * (E_W + E_S) / GiB, 4),
           "label": "[D] real shapes x the seed map (295 hot / layer); today = arm C's configuration; free after "
                    "build [E] = arm C's measured free minus the HBM delta"}
    print(f"[D] per expert slot: weights {E_W:,} B + SF {E_S:,} B (e8m0 = packed int32, same bytes)")
    for p, r in sb.items():
        print(f"  {p}: N {r['N']} K {r['K']}: B row pitch {r['B_row_pitch_B']} B, group stride {r['B_group_stride_B']:,}"
              f" B; SFB {r['SFB_rows_per_group']} rows x {r['SFB_row_pitch_B']} B = {r['SFB_group_stride_B']:,} B/group;"
              f" SF as extra B rows {r['sf_as_extra_B_rows']:.0f}; SF-per-row pitch {r['row_pitch_if_sf_per_row_B']} B"
              f" (16-B ok {r['pitch_16B_ok']})")
    t = out["today_C"]
    print(f"[D] today (arm C, pool-spare 20, 4 residency, 17/40 pooled): SF in HBM {t['sf_hbm_GiB']} GiB = "
          f"{t['parts_GiB']}")
    print("  beside pool-spare res  SF GiB  weights vs C  HBM vs C  free after build [E]  host e8m0 GiB")
    for r in rows:
        print(f"  {r['pool_spare']:17d} {r['residency_slots']:3d} {r['sf_hbm_GiB']:7.3f} {r['weights_vs_C_GiB']:13.3f}"
              f" {r['hbm_vs_C_GiB']:9.3f} {r['free_after_build_GiB_E']:21.2f} {r['host_e8m0_GiB']:14.3f}")
    return out


# ------------------------------------------------------------------------------------------------ estimate
def cmd_estimate(a) -> dict:
    ar = arms()
    A, B, C = ar["A"], ar["B"], ar["C"]
    per_layer = {"A->B": (A["ms"] - B["ms"]) / (B["pooled"] - A["pooled"]),
                 "A->C (res loss 0)": (A["ms"] - C["ms"]) / (C["pooled"] - A["pooled"]),
                 "B->C (res loss 0)": (B["ms"] - C["ms"]) / (C["pooled"] - B["pooled"]),
                 "B->C (res loss 0.02/slot)": (B["ms"] - C["ms"] + (B["res"] - C["res"]) * R_RES_SLOT_MS[1])
                 / (C["pooled"] - B["pooled"])}
    rest = L - C["pooled"]
    lo, hi = min(per_layer.values()), max(per_layer.values())
    central = per_layer["A->C (res loss 0)"]
    arm20 = {"central": C["ms"] - rest * central, "range": [C["ms"] - rest * hi, C["ms"] - rest * lo]}
    res_back = [n * (24 - STAGE - C["res"]) for n in R_RES_SLOT_MS]
    arm24 = {"central": arm20["central"] - sum(res_back) / 2,
             "range": [arm20["range"][0] - res_back[1], arm20["range"][1] - res_back[0]]}
    model = [C["ms"] - rest * u / 1e3 for u in reversed(M3S2_US_PER_LAYER)]
    out = {"arms_M": ar, "ms_per_pooled_layer_D": {k: round(v, 4) for k, v in per_layer.items()},
           "layers_left": rest,
           "beside_pool_spare_20_E": {"central": round(arm20["central"], 2), "range": [round(x, 2) for x in arm20["range"]]},
           "beside_pool_spare_24_E": {"central": round(arm24["central"], 2), "range": [round(x, 2) for x in arm24["range"]]},
           "m3s2_model_ceiling_E": [round(x, 2) for x in model],
           "arithmetic": (f"per pooled layer from the arms: A->B ({A['ms']} - {B['ms']}) / 7 = {per_layer['A->B']:.4f}; "
                          f"A->C ({A['ms']} - {C['ms']}) / 17 = {central:.4f}; B->C ({B['ms']} - {C['ms']}) / 10 = "
                          f"{per_layer['B->C (res loss 0)']:.4f} (+4 residency slots x 0.02 -> "
                          f"{per_layer['B->C (res loss 0.02/slot)']:.4f}); the remaining {rest} layers x those = "
                          f"{rest * lo:.2f}..{rest * hi:.2f} ms off {C['ms']}; residency 4 -> 8 slots (pool-spare "
                          f"24) {res_back[0]:.2f}..{res_back[1]:.2f} ms more; M3-S2's model "
                          f"({M3S2_US_PER_LAYER[0]}-{M3S2_US_PER_LAYER[1]} us/layer) x {rest} = "
                          f"{rest * M3S2_US_PER_LAYER[0] / 1e3:.2f}..{rest * M3S2_US_PER_LAYER[1] / 1e3:.2f} ms "
                          f"(the measured per-layer gain was ~40 % of that model: the ceiling, not the forecast)"),
           "label": "[E] linear in pooled layers (each arm's gain / its layer count); beside pools all 40"}
    print("[E]", out["arithmetic"])
    print(f"[E] beside @ pool-spare 20: {out['beside_pool_spare_20_E']}; @ 24 (8 residency): "
          f"{out['beside_pool_spare_24_E']}; M3-S2 model ceiling {out['m3s2_model_ceiling_E']}")
    return out


# ------------------------------------------------------------------------------------------------ contract (CPU)
def tiny_pool(layout: str, *, E=8, hot=(0, 1, 3, 5, 6), S=4, dim=128, inter=128, seed=7, device="cpu"):
    """A one-layer ExpertPool in `layout` with random checkpoint-like bytes (hot in the pool, cold in its host
    buffer), S spare slots, bound to nothing (no model). Returns (pool, src) with src[(p, kind, e)] = the bytes."""
    from runtime.expert_pool import ExpertPool, TIER_HBM
    from runtime.placement import HotMap

    g = torch.Generator(device="cpu").manual_seed(seed)
    hm = HotMap({0: list(hot)}, 1, E, "tiny")
    pool = ExpertPool(1, E, dim, inter, hm, device, hugepages=False, sf_layout=layout)
    src = {}
    for e in range(E):
        for p, (n, k) in pool.proj_nk.items():
            w = torch.randint(-128, 128, (n, k // 2), generator=g, device="cpu").to(torch.int8)
            s = torch.randint(110, 135, (n, k // 32), generator=g, device="cpu").to(torch.uint8)
            src[(p, "weight", e)], src[(p, "scale", e)] = w, s
            if pool.tier(0, e) == TIER_HBM:
                pool.hbm_dst(0, e, p, "weight").copy_(w)
                pool.hbm_dst(0, e, p, "scale").view(torch.uint8).copy_(s)
            else:
                pool.cold_view(0, e, p, "weight", cuda=False).copy_(w)
                pool.cold_view(0, e, p, "scale", cuda=False).view(torch.uint8).copy_(s)

    _grow(pool, S)
    if layout == "beside":
        pool.pack_hot_sf()
    return pool, src


def _grow(pool, S):
    """add_spare_slots without rebinding Experts (tiny pools have no model)."""
    class _Lin:
        scale = None

    class _Ex:
        w1 = _Lin()
        w2 = _Lin()
        w3 = _Lin()

    class _M:
        layers = [type("L", (), {"ffn": type("F", (), {"experts": [_Ex() for _ in range(pool.n_experts)]})()})()
                  for _ in range(pool.n_layers)]

    pool.add_spare_slots(S, _M())


def layer_tables(pool, layer=0):
    """The dg-inplace staging tables of a pool layer, the way DGInplaceBuffers.layer_tables builds them: ptab6 (hot
    -> pool weight slot / the slot's e8m0 home, cold -> its host bytes), gtab, the six destinations (weights +
    packed SF; separate: a fresh packed copy as the "step" store's, beside: the pool's own)."""
    from runtime.expert_pool import TIER_HBM

    G = int(pool.hbm_full[layer]["w1"].size(0))
    Hh = len(pool.hot[layer])
    rows = []
    for p, kind in DI.ORDER6:
        r = []
        for e in range(pool.n_experts):
            t, s = pool.slot[(layer, e)]
            if t == TIER_HBM:
                v = pool.hbm_full[layer][p][s] if kind == "weight" else pool.hbm[layer][p][1][s]
            else:
                v = pool.cold_view(layer, e, p, kind, cuda=False)
            r.append(v.data_ptr())
        rows.append(r)
    ptab6 = torch.tensor(rows, dtype=torch.int64, device="cpu")
    gtab = torch.full((pool.n_experts,), -1, dtype=torch.int32, device="cpu")
    for s, e in enumerate(pool.hot[layer]):
        gtab[e] = s
    if pool.sf_beside:
        sf = pool.packed_sf(layer)
    else:
        sf = {}
        for p in PROJS:
            n, k = pool.proj_nk[p]
            sf[p] = DI.packed_sf_zeros(G, n, k // 128, "cpu")
    dsts = [pool.hbm_full[layer][p] for p in PROJS] + [sf[p] for p in PROJS]
    dtab = torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64, device="cpu")
    nk = pool.proj_nk
    sf_dims = ((nk["w1"][0], nk["w1"][1] // 128), (nk["w2"][0], nk["w2"][1] // 128))
    return {"ptab6": ptab6, "gtab": gtab, "dsts": dsts, "dtab": dtab, "H": Hh, "G": G, "sf": sf,
            "row_w": nk["w1"][0] * nk["w1"][1] // 2, "sf_dims": sf_dims}


def route_lists(pool, t, idx, step_sf: bool, S: int):
    """moe_rows' dedup + work list for routing idx [R, k] (the runtime's own functions)."""
    from runtime import moe_rows as MR

    R_, k = idx.shape
    br = 8                                              # block rows >= R (a power of two, the contiguous ALIGN)
    m = MR.dedup_routes(idx, torch.rand(R_, k, generator=torch.Generator().manual_seed(1)), br)
    inpl = {"gtab": t["gtab"], "H": t["H"], "S": S, "dummy": torch.tensor(int(pool.hot[0][-1])), "has_dummy": True,
            "v2": True, "layout": "contiguous", "G": t["G"], "gscr": 0, "step_sf": step_sf}
    ext = MR.rows_layout(m, inpl, br, R_, k)
    return m, ext


def address_kinds(pool, layer=0) -> dict:
    """data_ptr -> (kind, uint8 bytes) for every address a ptab6 entry can name (the counting resolver)."""
    from runtime.expert_pool import TIER_HBM

    out = {}
    for e in range(pool.n_experts):
        t, s = pool.slot[(layer, e)]
        for p, kind in DI.ORDER6:
            if t == TIER_HBM:
                v = pool.hbm_full[layer][p][s] if kind == "weight" else pool.hbm[layer][p][1][s]
                tag = f"hot.{kind}"
            else:
                v = pool.cold_view(layer, e, p, kind, cuda=False)
                tag = f"cold.{kind}"
            out[v.data_ptr()] = (tag, v.reshape(-1).view(torch.uint8))
    return out


def cmd_contract(a) -> dict:
    """Rows the staging contract reads per layer-call: separate + "step" lists vs beside + pool lists."""
    idx = torch.tensor([[0, 2, 4], [1, 2, 7], [0, 4, 7], [3, 5, 6]])   # hot 0 1 3 5 6, cold 2 4 7
    out = {}
    for layout, step_sf in (("separate", True), ("beside", False)):
        pool, _ = tiny_pool(layout)
        t = layer_tables(pool)
        m, ext = route_lists(pool, t, idx, step_sf, 4)
        kinds = address_kinds(pool)
        cnt = {}

        def resolve(addr):
            k_, v = kinds[int(addr)]
            cnt[k_] = cnt.get(k_, 0) + 1
            return v

        hot_before = [x[: t["H"]].clone() for x in t["dsts"][3:]]
        n_work = int(idx.numel()) if step_sf else 4
        DI.stage_rows_emulated(resolve, t["ptab6"], m["uniq"], t["gtab"], t["dtab"], t["dsts"], ext["sidx"],
                               ext["wl"], t["H"], t["row_w"], t["sf_dims"], n_work)
        hot_unchanged = all(torch.equal(x[: t["H"]], y) for x, y in zip(t["dsts"][3:], hot_before))
        out[layout] = {"rows_read": dict(sorted(cnt.items())), "n_work": n_work,
                       "hot_sf_groups_untouched": hot_unchanged}
    print(f"[M] stage contract rows read: {json.dumps(out)}")
    return out


# ------------------------------------------------------------------------------------------------ probe (GPU)
def cmd_probe(a) -> dict:
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": a.argv, "errors": [], "label": {
        "build": "[M] real-shape ExpertPool, random bytes, both layouts",
        "gemm": "[M] DeepGEMM contiguous on beside's pool SF vs on a copy packed from separate's e8m0",
        "stage": "[M] bench/staging_bytes_bench.py probe serial + overlap (step lists = with hot SF, pool lists = "
                 "without)"}}

    def step(name, fn):
        try:
            r = fn()
            out[name] = r
            print("OK  ", name, json.dumps(r, default=str)[:800], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            out[name] = {"error": tb[-2000:]}
            out["errors"].append(name)
            print("FAIL", name, tb[-2000:], flush=True)
            return None

    if not torch.cuda.is_available():
        out["errors"].append("no CUDA")
        return out
    dev = torch.device("cuda")
    pools = {}

    def build():
        from runtime.expert_pool import ExpertPool
        from runtime.placement import HotMap

        res = {}
        for layout in ("separate", "beside"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            f0 = torch.cuda.mem_get_info()[0]
            t0 = time.time()
            hm = HotMap({l: list(range(a.hot)) for l in range(a.layers)}, a.layers, SB.EXPERTS, "probe")
            pool = ExpertPool(a.layers, SB.EXPERTS, SB.DIM, SB.INTER, hm, dev, sf_layout=layout)
            t_con = time.time() - t0
            g = torch.Generator(device=dev).manual_seed(a.seed)
            t0 = time.time()
            for l in range(a.layers):
                for p in PROJS:
                    w, s = pool.hbm[l][p]
                    w.copy_(torch.randint(-128, 128, w.shape, generator=g, device=dev, dtype=torch.int8))
                    codes = torch.randint(110, 135, s.shape, generator=g, device=dev, dtype=torch.uint8)
                    if layout == "separate":
                        s.view(torch.uint8).copy_(codes)
                    else:
                        s.view(torch.uint8).copy_(codes.cpu())          # the host copy (the loader's read)
            torch.cuda.synchronize()
            t_fill = time.time() - t0
            t_reg = None
            if pool.hot_sf_host is not None:
                t_reg = pool.hot_sf_host.register()
            t0 = time.time()
            if pool.sf_beside:
                pool.pack_hot_sf()
            torch.cuda.synchronize()
            t_pack = time.time() - t0
            t0 = time.time()
            _grow(pool, a.spare)
            torch.cuda.synchronize()
            t_grow = time.time() - t0
            f1 = torch.cuda.mem_get_info()[0]
            pools[layout] = pool
            res[layout] = {"construct_s": round(t_con, 3), "fill_s": round(t_fill, 3), "register_s": t_reg,
                           "pack_s": round(t_pack, 3), "grow_s": round(t_grow, 3),
                           "pool_hbm_GiB": round(pool.hbm_bytes() / GiB, 3),
                           "process_hbm_delta_GiB": round((f0 - f1) / GiB, 3),
                           "host_e8m0_GiB": round(pool.hot_sf_bytes / GiB, 3),
                           "per_layer_hbm_GiB": round(pool.hbm_bytes() / GiB / a.layers, 4)}
        # the separate layout's packed SF (a "pool" store) would be another copy on top: per layer
        res["separate_plus_pool_sf_per_layer_GiB"] = round(
            res["separate"]["per_layer_hbm_GiB"] + (a.hot + a.spare) * E_S / GiB, 4)
        return res

    step("build", build)

    def gemm():
        from runtime.dg_moe import import_deep_gemm

        dg = import_deep_gemm()
        assert not any(m_ == "vllm" or m_.startswith("vllm.") for m_ in sys.modules)
        sep, bes = pools["separate"], pools["beside"]
        res = {}
        gen = torch.Generator(device="cpu").manual_seed(a.seed + 1)
        for p, which in (("w1", 1), ("w2", 2)):
            n, k = SB.NK[p]
            G = int(bes.hbm_full[0][p].size(0))
            # both pools were filled from the same seeded generator: identical weights and e8m0 codes; beside's
            # packed SF came from ITS host copy (pack_hot_sf), the reference is packed from separate's HBM e8m0
            ref_sf = DI.packed_sf_zeros(G, n, k // 128, dev)
            ref_sf[: a.hot].copy_(DI.pack_words(sep.hbm[0][p][1]))
            assert torch.equal(bes.hbm_full[0][p][: a.hot], sep.hbm_full[0][p][: a.hot]), (p, "weights differ")
            res[f"{p}_packed_equal"] = bool(torch.equal(bes.sfp[0][p], ref_sf))
            groups = sorted(torch.randperm(a.hot, generator=gen)[:22].tolist())
            keys = [(r, gq) for r, gq in enumerate(groups)]
            q = (torch.randn(len(keys), k, generator=gen) * 2).clamp(-448, 448).to(torch.float8_e4m3fn).to(dev)
            s = torch.exp2(torch.randint(-3, 4, (len(keys), k // 32), generator=gen).float()).to(dev)
            acts: dict = {kk: (q[i], s[i]) for i, kk in enumerate(keys)}
            blocks = [[(kk, kk[1])] for kk in keys]
            o_ref = DI.run_contiguous(dg, sep.hbm_full[0][p], ref_sf, blocks, acts, 16, len(blocks))
            o_bes = DI.run_contiguous(dg, bes.hbm_full[0][p], bes.sfp[0][p], blocks, acts, 16, len(blocks))
            cat = lambda m_: torch.stack([m_[kk] for kk in keys])  # noqa: E731
            res[p] = DI.compare_bf16(cat(o_bes), cat(o_ref))
        res["equal"] = all(v["equal"] for v in res.values() if isinstance(v, dict)) and \
            all(v for k_, v in res.items() if k_.endswith("_packed_equal"))
        return res

    if "build" not in out["errors"]:
        step("gemm", gemm)
    pools.clear()
    torch.cuda.empty_cache()

    def stage():
        sa = ["probe", "--rounds", str(a.rounds), "--reps", str(a.reps), "--hot", str(min(a.hot, 128)),
              "--out", os.path.join(REPO, "scratch", f"pool-layout-stage-{_date()}.json")]
        rc = SB.main(sa)
        rep = json.load(open(sa[-1]))["probe"]
        res = {"rc": rc, "errors": rep.get("errors")}
        for c, v in (rep.get("serial") or {}).items():
            res[f"serial_c{c}"] = {"with_hot_sf_us": v.get("s1_proj_x3/step"), "without_us": v.get("s1_proj_x3/pool"),
                                   "hot_sf_term_us": v.get("hot_sf_term_s1_us")}
        for c, v in (rep.get("overlap") or {}).items():
            res[f"overlap_c{c}"] = {"exposed_with_hot_sf_us": v["kernel-overlap/step"]["exposed_us"],
                                    "exposed_without_us": v["kernel-overlap/pool"]["exposed_us"],
                                    "beside_layer_saves_us": v["pooled_layer_saves_us"]}
        res["checks"] = rep.get("checks")
        return res

    step("stage", stage)
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("cmd", choices=("bytes", "estimate", "contract", "probe", "all"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, default=2, help="probe: real-shape layers built per layout")
    ap.add_argument("--hot", type=int, default=H, help="probe: hot experts per layer")
    ap.add_argument("--spare", type=int, default=20, help="probe: spare slots per layer (the headline's pool-spare)")
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    out = {}
    if a.cmd in ("bytes", "all"):
        out["bytes"] = cmd_bytes(a)
    if a.cmd in ("estimate", "all"):
        out["estimate"] = cmd_estimate(a)
    if a.cmd in ("contract", "all"):
        out["contract"] = cmd_contract(a)
    if a.cmd == "probe" or (a.cmd == "all" and torch.cuda.is_available()):
        out["probe"] = cmd_probe(a)
    path = a.out or os.path.join(REPO, "scratch", f"pool-layout-{a.cmd}-{_date()}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)
    errs = out.get("probe", {}).get("errors", []) if isinstance(out.get("probe"), dict) else []
    return 2 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
