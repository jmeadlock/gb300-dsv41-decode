#!/usr/bin/env python3
"""dg_rowmatch_spike.py -- T9a-3 item 1 (box): does a token's routed-expert output depend on the DeepGEMM call shape it
rides in? On layer 0's REAL pool bytes ([H+S, N, K/2] int8 + packed SF, cold experts in the spare slots), for
--sets routed sets of R = 6 rows x k = 6 experts (each set routes >= 1 cold expert), every row's outputs of w1, w3, w2
under these call shapes, each fed the SAME activations (fp8 + power-of-two per-32 scales; w1/w3 one row per token,
w2 one row per (token, expert) pair):

  r1_c128        today's 1-row call (T9a-2): contiguous, ALIGN 128, k blocks, the token in row 0 of each block
  r1_c128_pk     the same with the packed activation SF (T9b-4 dg_sf_pack: what the 1-row step runs)
  r1_c16         ALIGN 16 (the 6-row step's), k blocks                      -- --moe-rowmatch align
  r1_c16_pad     ALIGN 16 and R*k blocks (the 6-row call's M), the token's k blocks tagged, the rest -1
                                                                              -- --moe-rowmatch on
  r1_c16_pad_pk  the same with packed activation SF
  r6_c16         the 6-row step: contiguous, ALIGN 16, R*k blocks, one block per unique expert with its rows
                 ascending (runtime/moe_rows.py's layout; fp32 activation SF = headline run 4, dg_sf_pack off at R)
  r6_c16_pk      the same with packed activation SF
  r6_c128        the 6-row step's former call (ALIGN 128)
  r1_c16_pad_g16 r1_c16_pad over the same pool plus 16 zero groups (num_groups G -> G + 16: a 1-row step built
                 before the pool grew to the 6-row step's spare count calls DeepGEMM with a smaller G)
  r1_masked      masked grouped GEMM over the pool's G groups, masked_m = 1 on the token's groups, expected_m 1
  r6_masked      the same with all R rows, masked_m = rows per group, expected_m R

-> for every pair of shapes: rows (of sets x R) whose 3 x k outputs are bitwise equal (the handoff's question: does a
row's result depend on its group's M / the kernel config?), max ulp; graph-timed us per layer (3 GEMMs) of the 1-row
shapes on one token and the projected 1-row step = --base-ms + --layers x (r1_c16_pad_pk - r1_c128_pk) / 1000
(the gate: within +0.2 ms of 10.52). Receipt: scratch/dg-rowmatch-spike-<date>.json (never docs/).

Run on the box like bench/dg_rows_spike.py (library image, checkpoint at /ref, repo at /work, DG JIT cache):
  python3 bench/dg_rowmatch_spike.py [--sets 8] [--rows-align 16] [--out scratch/dg-rowmatch-spike-<date>.json]
deep_gemm is imported from vllm/third_party with vllm never imported (runtime.dg_moe.import_deep_gemm).
"""
from __future__ import annotations

import argparse
import json
import os
import random
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

from runtime import dg_inplace as DI  # noqa: E402

FP8 = torch.float8_e4m3fn
PROJS = ("w1", "w3", "w2")
DIM, INTER, TOPK, E = 5120, 2304, 6, 384
NK = {"w1": (INTER, DIM), "w3": (INTER, DIM), "w2": (DIM, INTER)}   # (N, K)
RECIPE = dict(recipe_a=(1, 32), recipe_b=(1, 32))
MASKED_M = 128
GPAD = 16              # extra zero groups behind the pool for the num_groups arm (r1_c16_pad_g16)
BASE_MS = 10.52        # 1-row step, T9b-4 receipt (docs/LEDGER.md 2026-11-07 02:05; regression 02:39)
GATE_DELTA_MS = 0.2    # the handoff's gate: 1-row within +0.2 ms of 10.52


# ------------------------------------------------------------------------------------------------ pure helpers
def make_sets(hot, cold, n_sets, rows, topk, seed, n_cand=14):
    """n_sets routings of `rows` rows x `topk` distinct experts; set s routes 1 + s % 3 cold experts (each cold
    expert in >= 1 row) out of a candidate set of n_cand hot + the cold ones (experts shared across rows, as in the
    6-row step). -> [[sorted expert ids per row], ...] per set."""
    rng = random.Random(seed)
    out = []
    for s in range(n_sets):
        n_cold = 1 + s % 3
        cs = rng.sample(list(cold), n_cold)
        cand = rng.sample(list(hot), min(n_cand, len(hot))) + cs
        rexp = [rng.sample(cand, topk) for _ in range(rows)]
        for c in cs:
            if not any(c in r for r in rexp):
                r = rng.randrange(rows)
                j = next(i for i, e in enumerate(rexp[r]) if e not in cs)
                rexp[r][j] = c
        out.append([sorted(r) for r in rexp])
    return out


def keys_of(rexp):
    return [(r, e) for r, ex in enumerate(rexp) for e in ex]


def rows_equal(a, b, rexp):
    """rows r whose outputs (every projection, every routed expert) are bitwise equal in a and b."""
    n = 0
    for r, ex in enumerate(rexp):
        n += all(torch.equal(a[p][(r, e)].view(torch.int16), b[p][(r, e)].view(torch.int16)) for p in PROJS
                 for e in ex)
    return n


def max_ulp(a, b, rexp):
    ks = keys_of(rexp)
    return max(int(DI.bf16_ulp(torch.stack([a[p][k] for k in ks]), torch.stack([b[p][k] for k in ks])).max())
               for p in PROJS)


class Contig:
    """One contiguous call shape on persistent buffers (timing + outputs): blocks via DI.contiguous_blocks_layout."""

    def __init__(self, dg, W, SF, blocks, acts, align, n_blocks, packed, dev):
        self.dg, self.W, self.SF, self.align = dg, W, SF, int(align)
        self.rows, mi = DI.contiguous_blocks_layout(blocks, align, n_blocks)
        self.keys = list(self.rows)
        K = int(acts[self.keys[0]][0].numel())
        M = int(n_blocks) * int(align)
        self.A = torch.zeros(M, K, dtype=FP8, device=dev)
        S = torch.ones(M, K // 32, dtype=torch.float32, device=dev)
        self.idx = torch.tensor([self.rows[k] for k in self.keys], dtype=torch.long, device="cpu").to(dev)
        self.A.view(torch.uint8).index_copy_(0, self.idx, torch.stack([acts[k][0].view(torch.uint8) for k in self.keys]))
        S.index_copy_(0, self.idx, torch.stack([acts[k][1] for k in self.keys]))
        self.S = DI.pack_act_sf_from_f32(S) if packed else S
        self.mi = torch.tensor(mi, dtype=torch.int32, device="cpu").to(dev)
        self.D = torch.zeros(M, W.size(1), dtype=torch.bfloat16, device=dev)

    def __call__(self):
        self.dg.set_mk_alignment_for_contiguous_layout(self.align)
        self.dg.m_grouped_fp8_fp4_gemm_nt_contiguous((self.A, self.S), (self.W, self.SF), self.D, self.mi, **RECIPE)

    def out(self):
        o = self.D.index_select(0, self.idx)
        return {k: o[i].clone() for i, k in enumerate(self.keys)}


class Masked:
    """The masked grouped GEMM with the pool as B: block g = pool group g, masked_m[g] = its rows."""

    def __init__(self, dg, W, SF, entries, acts, expected_m, bufs, dev):
        self.dg, self.W, self.SF, self.em = dg, W, SF, int(expected_m)
        A, S, D = bufs
        G = W.size(0)
        self.A, self.S, self.D = A[:G], S[:G], D[:G]
        self.A.view(torch.uint8).zero_()
        self.S.fill_(1.0)
        mm = [0] * G
        self.pos = {}
        for key, g in entries:
            self.pos[key] = (g, mm[g])
            mm[g] += 1
        if max(mm) > MASKED_M:
            raise ValueError("masked: > 128 rows in a group")
        for key, (g, t) in self.pos.items():
            self.A[g, t].view(torch.uint8).copy_(acts[key][0].view(torch.uint8))
            self.S[g, t].copy_(acts[key][1])
        self.mm = torch.tensor(mm, dtype=torch.int32, device="cpu").to(dev)

    def __call__(self):
        self.dg.m_grouped_fp8_fp4_gemm_nt_masked((self.A, self.S), (self.W, self.SF), self.D, self.mm, self.em, **RECIPE)

    def out(self):
        return {k: self.D[g, t].clone() for k, (g, t) in self.pos.items()}


# ------------------------------------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="/ref")
    ap.add_argument("--map", default=os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sets", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--rows-align", type=int, default=16, help="the 6-row step's ALIGN (--rows-moe-gemm contiguousNN)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--layers", type=int, default=40, help="MoE layers per step (projection)")
    ap.add_argument("--base-ms", type=float, default=BASE_MS)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--no-masked", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu: the Mac dry run on a fake deep_gemm (no timing)")
    ap.add_argument("--out", default=None, help="default scratch/dg-rowmatch-spike-<date>.json")
    return ap


def main(argv=None):
    from dg_contiguous_spike import Ckpt, graph_us
    from runtime.dg_moe import import_deep_gemm

    a = build_parser().parse_args(argv)
    out_path = a.out or os.path.join(REPO, "scratch", f"dg-rowmatch-spike-{time.strftime('%Y%m%d')}.json")
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv[1:] if argv is None else argv,
           "errors": [], "arms": {}, "pairs": {}, "timing": {}}

    def step(name, fn, into):
        try:
            r = fn()
            into[name] = r
            print("OK  ", name, json.dumps(r, default=str)[:300], flush=True)
            return r
        except Exception:
            tb = traceback.format_exc()
            into[name] = {"error": tb[-1500:]}
            rep["errors"].append(name)
            print("FAIL", name, tb[-1500:], flush=True)
            return None

    dev = torch.device(a.device)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
    R_, k, A_ = a.rows, TOPK, int(a.rows_align)
    dg = import_deep_gemm()
    det = getattr(dg, "use_deterministic_algorithms", None)
    if det is not None:
        try:
            det(True)
        except TypeError:
            det()
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    rep["deep_gemm"] = {"file": getattr(dg, "__file__", None), "version": getattr(dg, "__version__", None),
                        "alignment_default": dg.get_mk_alignment_for_contiguous_layout()}
    al0 = dg.get_mk_alignment_for_contiguous_layout()

    # ---------------------------------------------------------------- layer pool (hot + every set's cold experts)
    L = a.layer
    hot = [int(e) for e in json.load(open(a.map))[str(L)]]
    hs = set(hot)
    cold_all = [e for e in range(E) if e not in hs]
    sets = make_sets(hot, cold_all, a.sets, R_, k, a.seed)
    cold_used = sorted({e for s in sets for r in s for e in r if e not in hs})
    H, S = len(hot), len(cold_used)
    G = H + S
    gmap = {e: i for i, e in enumerate(hot)}
    gmap.update({e: H + c for c, e in enumerate(cold_used)})
    ck = Ckpt(a.ckpt)
    t0 = time.time()
    C = {p: DI.sf_words(NK[p][1] // 32) for p in PROJS}
    # storage G + GPAD groups (the tail stays zero); every arm but *_g16 sees the G-group prefix views
    full_g = {p: torch.zeros(G + GPAD, NK[p][0], NK[p][1] // 2, dtype=torch.int8, device=dev) for p in PROJS}
    sf_g = {p: DI.packed_sf_zeros(G + GPAD, NK[p][0], C[p], dev) for p in PROJS}
    full = {p: t[:G] for p, t in full_g.items()}
    sf = {p: t[:G] for p, t in sf_g.items()}
    for e, g_ in gmap.items():
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            full[p][g_].copy_(w)
            sf[p][g_].copy_(DI.pack_words(s.contiguous().view(NK[p][0], -1)).to(dev))
    sync()
    ok = True
    for c, e in enumerate(cold_used):
        for p in PROJS:
            w, s = ck.expert(L, e, p)
            ok &= bool(torch.equal(full[p][H + c].cpu(), w)) and bool(torch.equal(
                DI.unpack_sf(sf[p][H + c:H + c + 1]).cpu()[0], s.contiguous().view(NK[p][0], -1)))
    rep["pool"] = {"layer": L, "H": H, "S": S, "G": G, "cold_used": cold_used, "cold_bytes_equal_checkpoint": ok,
                   "sets": sets, "cold_per_set": [sum(e not in hs for e in {x for r in s_ for x in r}) for s_ in sets],
                   "load_s": round(time.time() - t0, 1)}
    print(f"[pool] layer {L}: {H} hot + {S} cold in spare slots; {a.sets} sets of {R_}x{k}", flush=True)

    # ---------------------------------------------------------------- activations per set (CPU generator)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    def acts_for(rexp):
        out = {}
        for which, K in ((1, DIM), (2, INTER)):
            ks = keys_of(rexp)
            n = R_ if which == 1 else len(ks)
            q = (torch.randn(n, K, generator=gen, device="cpu") * 3).clamp(-448, 448).to(FP8).to(dev)
            s = torch.exp2(torch.randint(-6, 7, (n, K // 32), generator=gen, device="cpu").float()).to(dev)
            out[which] = {kk: ((q[kk[0]], s[kk[0]]) if which == 1 else (q[i], s[i])) for i, kk in enumerate(ks)}
        return out

    set_acts = [acts_for(s_) for s_ in sets]
    mbufs = {}
    if not a.no_masked:
        for which, K in ((1, DIM), (2, INTER)):
            mbufs[which] = (torch.zeros(G, MASKED_M, K, dtype=FP8, device=dev),
                            torch.ones(G, MASKED_M, K // 32, dtype=torch.float32, device=dev))
        mbufs["d"] = {p: torch.zeros(G, MASKED_M, NK[p][0], dtype=torch.bfloat16, device=dev) for p in PROJS}

    # name -> (kind, ALIGN, A blocks, packed act SF, pool with GPAD extra groups)
    ARMS = {"r1_c128": ("one", 128, k, False, False), "r1_c128_pk": ("one", 128, k, True, False),
            "r1_c16": ("one", A_, k, False, False),
            "r1_c16_pad": ("one", A_, R_ * k, False, False), "r1_c16_pad_pk": ("one", A_, R_ * k, True, False),
            "r1_c16_pad_g16": ("one", A_, R_ * k, False, True),
            "r6_c16": ("all", A_, R_ * k, False, False), "r6_c16_pk": ("all", A_, R_ * k, True, False),
            "r6_c128": ("all", 128, R_ * k, False, False)}
    if not a.no_masked:
        ARMS.update({"r1_masked": ("one_m", None, None, None, False), "r6_masked": ("all_m", None, None, None, False)})

    def arm_out(name, si):
        kind, al, nb, pk, gp = ARMS[name]
        Wd, Sd = (full_g, sf_g) if gp else (full, sf)
        rexp, acts = sets[si], set_acts[si]
        res = {}
        for p in PROJS:
            which = 2 if p == "w2" else 1
            got = {}
            if kind in ("one", "all"):
                groups = ([DI.one_row_blocks(r, rexp[r], gmap) for r in range(R_)] if kind == "one"
                          else [DI.r_row_blocks(rexp, gmap)])
                for bl in groups:
                    c = Contig(dg, Wd[p], Sd[p], bl, acts[which], al, nb, pk, dev)
                    c()
                    got.update(c.out())
            else:
                bufs = (mbufs[which][0], mbufs[which][1], mbufs["d"][p])
                if kind == "one_m":
                    for r in range(R_):
                        m = Masked(dg, full[p], sf[p], [((r, e), gmap[e]) for e in rexp[r]], acts[which], 1, bufs, dev)
                        m()
                        got.update(m.out())
                else:
                    ents = [((r, e), gmap[e]) for e in sorted({x for rr in rexp for x in rr})
                            for r in range(R_) if e in rexp[r]]
                    m = Masked(dg, full[p], sf[p], ents, acts[which], R_, bufs, dev)
                    m()
                    got.update(m.out())
            res[p] = got
        sync()
        dg.set_mk_alignment_for_contiguous_layout(al0)
        return res

    outs = {}
    for name in ARMS:
        def run(name=name):
            outs[name] = [arm_out(name, si) for si in range(len(sets))]
            return {"rows": len(sets) * R_}
        step(name, run, rep["arms"])

    # ---------------------------------------------------------------- per-row equality matrix
    names = [n for n in ARMS if n in outs]
    tot = len(sets) * R_
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            n_eq = sum(rows_equal(outs[x][si], outs[y][si], sets[si]) for si in range(len(sets)))
            mu = max(max_ulp(outs[x][si], outs[y][si], sets[si]) for si in range(len(sets)))
            rep["pairs"][f"{x}|{y}"] = {"rows_equal": n_eq, "of": tot, "max_ulp": mu}
    for key, v in rep["pairs"].items():
        print(f"  {key:30s} rows bit-equal {v['rows_equal']:3d}/{v['of']}  max ulp {v['max_ulp']}")

    # ---------------------------------------------------------------- timing (one token, 3 GEMMs = one layer)
    def timing():
        if dev.type != "cuda":
            raise RuntimeError("timing needs CUDA (dry run)")
        r = {}
        rexp, acts = sets[0], set_acts[0]
        for name in ("r1_c128", "r1_c128_pk", "r1_c16", "r1_c16_pad", "r1_c16_pad_pk", "r6_c16", "r6_c16_pk"):
            kind, al, nb, pk, _ = ARMS[name]
            bl = DI.one_row_blocks(0, rexp[0], gmap) if kind == "one" else DI.r_row_blocks(rexp, gmap)
            cs = [Contig(dg, full[p], sf[p], bl, acts[2 if p == "w2" else 1], al, nb, pk, dev) for p in PROJS]
            r[name] = graph_us(lambda cs=cs: [c() for c in cs], iters=a.iters)["us"]
        dg.set_mk_alignment_for_contiguous_layout(al0)
        d_ = r["r1_c16_pad_pk"] - r["r1_c128_pk"]
        r["delta_us_per_layer_on_vs_today"] = round(d_, 2)
        r["delta_us_per_layer_align_vs_today"] = round(r["r1_c16"] - r["r1_c128"], 2)
        r["projected_1row_ms_on"] = round(a.base_ms + a.layers * d_ / 1000.0, 3)
        r["projected_1row_ms_align"] = round(a.base_ms + a.layers * (r["r1_c16"] - r["r1_c128"]) / 1000.0, 3)
        r["note"] = (f"projection = {a.base_ms} ms + {a.layers} layers x the per-layer delta of 3 isolated GEMMs "
                     f"(graph replay); first_light 1-row is the measurement")
        return r

    step("us_per_layer", timing, rep["timing"])

    # ---------------------------------------------------------------- verdict
    def eq(x, y):
        v = rep["pairs"].get(f"{x}|{y}") or rep["pairs"].get(f"{y}|{x}")
        return None if v is None else v["rows_equal"] == v["of"]

    t = rep["timing"].get("us_per_layer") or {}
    on_ok = eq("r1_c16_pad", "r6_c16")
    rep["verdict"] = {
        "rows": tot,
        "rowmatch_on_equal_r6": on_ok,                       # --moe-rowmatch on vs the 6-row step, per row
        "rowmatch_on_packed_equal_r6": eq("r1_c16_pad_pk", "r6_c16"),
        "rowmatch_align_equal_r6": eq("r1_c16", "r6_c16"),
        "today_equal_r6": eq("r1_c128_pk", "r6_c16"),       # was the DG path ever the 1-vs-6 difference?
        "today_fp32_equal_r6": eq("r1_c128", "r6_c16"),
        "r6_packed_equal_fp32": eq("r6_c16", "r6_c16_pk"),
        "r6_c128_equal_c16": eq("r6_c128", "r6_c16"),
        "num_groups_invariant": eq("r1_c16_pad_g16", "r1_c16_pad"),   # G vs G + 16 groups, same call otherwise
        "masked_1_equal_6": eq("r1_masked", "r6_masked"),
        "masked_equal_contiguous": eq("r6_masked", "r6_c16"),
        "all_equal": all(v["rows_equal"] == v["of"] for v in rep["pairs"].values()) if rep["pairs"] else None,
        "not_equal_pairs": [kk for kk, v in rep["pairs"].items() if v["rows_equal"] != v["of"]],
        "projected_1row_ms_on": t.get("projected_1row_ms_on"),
        "delta_us_per_layer_on": t.get("delta_us_per_layer_on_vs_today"),
        "gate_1row_ms": round(a.base_ms + GATE_DELTA_MS, 2),
        "failed": [kk for kk, v in rep["arms"].items() if isinstance(v, dict) and "error" in v],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print("wrote", out_path)
    print("VERDICT", json.dumps(rep["verdict"]))
    return 0 if not rep["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
