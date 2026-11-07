#!/usr/bin/env python3
"""step_anatomy.py <profile.sqlite> [--out anatomy.json]
Per-step, per-role anatomy of OUR runtime's decode step from an nsys sqlite export:
  NVTX ranges: decode_step (one per step), L{i} (per layer), engram/attn/moe/head (per module, nested in L{i}).
  Kernels: CUPTI_ACTIVITY_KIND_KERNEL; attributed to the innermost NVTX range that contains the kernel's *launch*
  (via the correlation id -> runtime API call start time), which is how nsys itself attributes.
Reports per step: wall, GPU-busy (union of kernel intervals), idle, kernel count; per role (sum over layers): GPU ms,
launches, mean kernel µs; the top kernels by total time; and the largest inter-kernel gaps with the NVTX context.
Stdlib only (sqlite3)."""
import sqlite3, sys, json, collections, statistics

db = sqlite3.connect(sys.argv[1]); out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
q = lambda s, *a: db.execute(s, a).fetchall()
tables = {r[0] for r in q("select name from sqlite_master where type='table'")}
assert "CUPTI_ACTIVITY_KIND_KERNEL" in tables, tables

# kernels with names and launch (runtime API) start time via correlationId
# M1-T10-MOE: + the demangled name (a template wrapper's short name, e.g. CUTLASS's `Kernel2`, says nothing)
kern = q("""select k.start, k.end, k.correlationId, coalesce(s.value, k.demangledName), k.streamId,
                   coalesce(d.value, '')
            from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id = k.shortName
            left join StringIds d on d.id = k.demangledName order by k.start""")
rt = {}
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
    for s, e, cid in q("select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        rt[cid] = s
# nvtx ranges
nvtx = []
if "NVTX_EVENTS" in tables:
    for s, e, txt, tid in q("select start, end, coalesce(text, (select value from StringIds where id=textId)), globalTid from NVTX_EVENTS where end is not null"):
        nvtx.append((s, e, txt or "", tid))
steps = sorted([(s, e) for s, e, t, _ in nvtx if t == "decode_step"])
if not steps:
    # graph replay: NVTX ranges do not survive; T8's per-step marker kernel (static_kernels.t8_step_marker) delimits steps
    marks = [k[0] for k in kern if "t8_step_marker" in (k[3] or "")]
    steps = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    print(f"no decode_step NVTX; using {len(marks)} marker kernels -> {len(steps)} steps")
layers = sorted([(s, e, t) for s, e, t, _ in nvtx if t.startswith("L") and t[1:].isdigit()])
roles = sorted([(s, e, t) for s, e, t, _ in nvtx if t in ("engram", "attn", "moe", "head")])
print(f"kernels {len(kern)}  runtime-api {len(rt)}  nvtx {len(nvtx)}  decode_step ranges {len(steps)}  layer ranges {len(layers)}  role ranges {len(roles)}")

import bisect
def finder(ranges):
    starts = [r[0] for r in ranges]
    def f(t):
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            if ranges[i][0] <= t <= ranges[i][1]: return ranges[i]
            i -= 1
        return None
    return f
find_step, find_layer, find_role = finder(steps), finder(layers), finder(roles)

per_step = []
role_t = collections.defaultdict(float); role_n = collections.Counter(); layer_t = collections.defaultdict(float)
kname_t = collections.defaultdict(float); kname_n = collections.Counter()
kname_full = {}; kname_role = collections.defaultdict(collections.Counter)   # M1-T10-MOE: what/who a name is
gaps = []
for si, (ss, se) in enumerate(steps):
    ks = [k for k in kern if ss <= k[0] < se]
    if not ks: continue
    busy = 0; last_end = None; n = 0
    for s, e, cid, name, _, full in ks:
        if last_end is None or s > last_end:
            busy += e - s
            if last_end is not None and s - last_end > 0:
                gaps.append((s - last_end, si, name, (find_role(rt.get(cid, s)) or ("", "", "?"))[2], (find_layer(rt.get(cid, s)) or ("", "", "?"))[2]))
            last_end = e
        else:
            busy += max(0, e - last_end); last_end = max(last_end, e)
        n += 1
        lt = rt.get(cid, s)
        r = find_role(lt); L = find_layer(lt)
        role = r[2] if r else ("step-other")
        role_t[role] += (e - s) / 1e6; role_n[role] += 1
        if L: layer_t[L[2]] += (e - s) / 1e6
        kname_t[name] += (e - s) / 1e6; kname_n[name] += 1
        kname_full.setdefault(name, full); kname_role[name][role] += 1
    per_step.append({"step": si, "wall_ms": (se - ss) / 1e6, "busy_ms": busy / 1e6, "kernels": n})

ns = len(per_step)
print(f"\nsteps analysed {ns}: wall mean {statistics.mean(p['wall_ms'] for p in per_step):.2f} ms  p50 {statistics.median(p['wall_ms'] for p in per_step):.2f}  "
      f"busy mean {statistics.mean(p['busy_ms'] for p in per_step):.2f} ms  kernels/step {statistics.mean(p['kernels'] for p in per_step):.0f}")
print("\nrole            GPU ms/step  launches/step  mean µs/kernel")
for r, t in sorted(role_t.items(), key=lambda x: -x[1]):
    print(f"{r:15s} {t / ns:10.2f} {role_n[r] / ns:14.0f} {1000 * t / role_n[r]:14.1f}")
print("\nlayer GPU ms/step (first 3, 14 (engram), last):")
for L in ["L0", "L1", "L2", "L14", "L39"]:
    if L in layer_t: print(f"  {L:4s} {layer_t[L] / ns:7.2f}")
print("\ntop kernels by GPU time (ms/step, launches/step, µs each):")
for name, t in sorted(kname_t.items(), key=lambda x: -x[1])[:18]:
    full = kname_full.get(name, "")
    print(f"  {t / ns:7.2f} {kname_n[name] / ns:7.0f} {1000 * t / kname_n[name]:8.1f}  {name[:90]}"
          + (f"  [{full[:120]}]" if full and full != name else ""))
gaps.sort(reverse=True)
tot_gap = sum(g[0] for g in gaps) / 1e6 / ns
print(f"\ninter-kernel idle inside steps: {tot_gap:.2f} ms/step total; largest:")
for g, si, name, role, L in gaps[:10]:
    print(f"  {g / 1e3:8.1f} µs  step {si} {L}/{role} before {name[:70]}")
if out:
    json.dump({"steps": per_step, "role_ms_per_step": {r: t / ns for r, t in role_t.items()}, "role_launches_per_step": {r: c / ns for r, c in role_n.items()},
               "layer_ms_per_step": {L: t / ns for L, t in layer_t.items()}, "top_kernels": [(n, t / ns, kname_n[n] / ns) for n, t in sorted(kname_t.items(), key=lambda x: -x[1])[:40]],
               "idle_ms_per_step": tot_gap,
               "kernel_names": {n: kname_full.get(n, "")[:400] for n, _ in sorted(kname_t.items(), key=lambda x: -x[1])[:40]},
               "kernel_roles": {n: {r: c / ns for r, c in kname_role[n].items() if c} for n, _ in sorted(kname_t.items(), key=lambda x: -x[1])[:40]}},
              open(out, "w"), indent=1)
