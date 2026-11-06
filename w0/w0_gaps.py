#!/usr/bin/env python3
"""w0_gaps.py <nsys.sqlite> [--out gaps.json]

Per-boundary gap histogram for the decode step (W1 2026-10-28 named this the prerequisite measurement for any
fusion/PDL work). Stdlib only. Two views:

  (1) serial boundaries: consecutive kernels in global start order on the GPU: gap = start[i+1] - end[i].
      Negative gap = the next kernel started before the previous ended (overlap: another stream, or PDL early launch).
  (2) union idle: gaps between busy intervals of the union of all kernels (what the GPU actually sat idle).

Prints histograms in microseconds and totals per 1 ms of busy time, and writes JSON.
"""
import json, sqlite3, sys, collections

db = sys.argv[1]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
con = sqlite3.connect(db)
rows = con.execute("select start, end, streamId, deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
if not rows:
    print("no kernels"); sys.exit(1)
devs = collections.Counter(r[3] for r in rows)
dev = devs.most_common(1)[0][0]
rows = [r for r in rows if r[3] == dev]
t0, t1 = rows[0][0], max(r[1] for r in rows)
span_ms = (t1 - t0) / 1e6

# (1) serial boundary gaps
gaps = [(rows[i + 1][0] - rows[i][1]) / 1e3 for i in range(len(rows) - 1)]  # µs
bins = [(-1e9, -5), (-5, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 50), (50, 1e9)]
hist = collections.OrderedDict()
for lo, hi in bins:
    sel = [g for g in gaps if lo <= g < hi]
    hist[f"[{lo:g},{hi:g})"] = {"n": len(sel), "sum_us": round(sum(sel), 1)}
pos = [g for g in gaps if g > 0]

# (2) union idle
busy = []
cs, ce = rows[0][0], rows[0][1]
for s, e, *_ in rows[1:]:
    if s <= ce:
        ce = max(ce, e)
    else:
        busy.append((cs, ce)); cs, ce = s, e
busy.append((cs, ce))
busy_ms = sum(e - s for s, e in busy) / 1e6
idle = [(busy[i + 1][0] - busy[i][1]) / 1e3 for i in range(len(busy) - 1)]
ihist = collections.OrderedDict()
for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 50), (50, 1e9)]:
    sel = [g for g in idle if lo <= g < hi]
    ihist[f"[{lo:g},{hi:g})"] = {"n": len(sel), "sum_us": round(sum(sel), 1)}

res = {"db": db, "device": dev, "kernels": len(rows), "span_ms": round(span_ms, 3), "busy_ms": round(busy_ms, 3),
       "busy_frac": round(busy_ms / span_ms, 4), "idle_ms": round(span_ms - busy_ms, 3),
       "serial_gap_hist_us": hist, "serial_gap_pos_sum_us": round(sum(pos), 1), "serial_gap_pos_n": len(pos),
       "serial_gap_neg_n": len(gaps) - len(pos), "union_idle_hist_us": ihist, "union_idle_segments": len(idle)}
print(json.dumps(res, indent=1))
if out:
    json.dump(res, open(out, "w"), indent=1)
