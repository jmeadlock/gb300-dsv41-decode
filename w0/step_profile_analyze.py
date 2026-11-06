#!/usr/bin/env python3
"""step_profile_analyze.py <nsys.sqlite> [--rej NAME] [--moe SUBSTR] [--steps A:B]

Per-decode-step kernel profile from an nsys sqlite export (see references/step-profile-nsys.md).
Stdlib only. Prints: step wall, GPU-busy union, idle histogram, exclusive-attribution table,
and the per-kernel top-40, so a fusion or kernel-selection question can be answered with numbers.

Steps are delimited by the kernel that fires exactly once per spec step (default: _rejection_kernel).
"""
import collections
import sqlite3
import statistics as st
import sys


def load(db):
    q = ("select k.start, k.end, s.value, d.value from CUPTI_ACTIVITY_KIND_KERNEL k "
         "join StringIds s on k.shortName = s.id join StringIds d on k.demangledName = d.id order by k.start")
    return sqlite3.connect(db).execute(q).fetchall()


def role(name):
    n = name
    if "mega_moe" in n:
        return "moe_hot"
    if "_finish2" in n:
        return "sidecar_wait"
    if "mhc" in n or "hc_prenorm" in n:
        return "mhc"
    if "fused_norm_rope_attn" in n or "attn" in n.lower():
        return "attention"
    if "LowLatency" in n or "gemm" in n.lower() or "nvjet" in n or "Gemm" in n:
        return "dense_gemm"
    if "route" in n:
        return "route_send"
    return "other"


def main(argv):
    db = argv[1]
    rej = "_rejection_kernel"
    if "--rej" in argv:
        rej = argv[argv.index("--rej") + 1]
    K = load(db)
    if not K:
        print("no kernels")
        return 1
    bounds = [i for i, k in enumerate(K) if k[2] == rej]
    if len(bounds) < 4:
        # fall back: the name carrying the fewest-launch kernel that divides the MoE count
        cnt = collections.Counter(k[2] for k in K)
        moe = sum(v for n, v in cnt.items() if "mega_moe" in n)
        for name, v in sorted(cnt.items(), key=lambda x: -x[1]):
            if v > 3 and moe % v == 0 and moe // v >= 2 and "mega_moe" not in name:
                bounds = [i for i, k in enumerate(K) if k[2] == name]
                print(f"# step boundary inferred: {name} ({v} launches)")
                break
    if len(bounds) < 4:
        print("could not find a per-step boundary; pass --rej")
        return 2
    moe = [i for i, k in enumerate(K) if "mega_moe" in k[2]]
    per = collections.Counter(sum(1 for m in moe if a <= m < b) for a, b in zip(bounds, bounds[1:]))
    print(f"kernels={len(K)}  window_ms={(K[-1][1]-K[0][0])/1e6:.2f}  MegaMoE per step={dict(per)}")

    steps = []
    for a, b in zip(bounds, bounds[1:]):
        seg = K[a:b]
        if not seg:
            continue
        t0, t1 = seg[0][0], K[b][0]
        iv = sorted((max(s, t0), min(e, t1)) for s, e, _, _ in seg)
        cur_s, cur_e = iv[0]
        idle = []
        for s, e in iv[1:]:
            if s > cur_e:
                idle.append(s - cur_e)
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if t1 > cur_e:
            idle.append(t1 - cur_e)
        busy = (t1 - t0) - sum(idle)
        ev = []
        for s, e, _, dn in seg:
            r = role(dn)
            ev.append((max(s, t0), 1, r))
            ev.append((min(e, t1), -1, r))
        ev.sort()
        live, excl, last = collections.Counter(), collections.Counter(), t0
        for t, d, r in ev:
            n_live = sum(1 for v in live.values() if v > 0)
            if t > last and n_live:
                for k, v in live.items():
                    if v > 0:
                        excl[k] += (t - last) / n_live
            last = t
            live[r] += d
        steps.append({"wall": t1 - t0, "busy": busy, "n": len(seg), "idle": idle, "excl": excl,
                      "streams": len({k[2] for k in seg})})
    steps = steps[1:-1] or steps
    n = len(steps)
    wall = st.median(s["wall"] for s in steps) / 1e6
    print(f"steps={n}  wall_median_ms={wall:.3f}  busy={st.median(s['busy'] for s in steps)/1e6:.3f} "
          f"({100*st.median(s['busy']/s['wall'] for s in steps):.1f}%)  kernels/step={st.median(s['n'] for s in steps):.0f}"
          f"  idle_ms={sum(g for s in steps for g in s['idle'])/n/1e6:.3f}")
    print(f"{'role':<14}{'excl ms/step':>14}")
    roles = sorted({r for s in steps for r in s["excl"]})
    for r in roles:
        print(f"{r:<14}{st.mean(s['excl'][r] for s in steps)/1e6:>14.3f}")
    durs = [e - s for s, e, _, _ in K[bounds[1]:bounds[-2]]]
    for lo, hi in ((0, 3e3), (3e3, 6e3), (6e3, 10e3), (10e3, 20e3), (20e3, 50e3), (50e3, 1e9)):
        sel = [d for d in durs if lo <= d < hi]
        print(f"  {lo/1e3:5.0f}-{min(hi,1e8)/1e3:5.0f} us: {len(sel)/n:7.1f}/step  {sum(sel)/n/1e6:6.3f} ms/step")
    tot = collections.Counter()
    cnt = collections.Counter()
    for s, e, sn, dn in K[bounds[1]:bounds[-2]]:
        key = dn[:90] if sn in ("fwd_kernel", "kernel") else sn
        tot[key] += e - s
        cnt[key] += 1
    print(f"{'ms/step':>9}{'n/step':>8}{'avg_us':>8}  kernel")
    for k, v in tot.most_common(40):
        print(f"{v/n/1e6:9.3f}{cnt[k]/n:8.1f}{v/cnt[k]/1e3:8.1f}  {k[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
