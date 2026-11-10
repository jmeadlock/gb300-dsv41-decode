#!/usr/bin/env python3
"""gt_res_drive.py -- card GT3 driver (docs/HANDOFF-A3-RESIDENCY.md): control, traffic and receipts for the adaptive
residency cells R0..R5 of `GT1_RES=1 bench/box/card-gt1.sh`. Stdlib only (it runs on the box's host python, like
w0/w0_drive.py, whose fixture, step prompt and stream client it uses).

  ctrl    --dir D --set JSON [--nudge fixture|step] [--wait ack|idle|boot] [--timeout S] [--out F]
          Write D/ctrl.json (by rename, with a fresh "seq"), then drive short generations until the server's
          telemetry acknowledges that seq (the manager reads the file at a counter drain, i.e. only under traffic).
          --wait idle: keep driving until a later drain reports an empty swap queue; --wait boot: until it also
          reports the boot map (after restore_map). Exit 0 = acknowledged (and idle / at boot), 1 = timed out.
  c1      --dir D --seconds N --out F [--ctrl JSON] [--tokens 256]
          The C1 traffic of cell R1: the 8 fixture prompts x --tokens, one stream, looped until N seconds after the
          control document (default: unfreeze + baseline) is acknowledged. Receipt: per request rows, the telemetry
          drains of the window, cold share first -> plateau, swaps per drain, T90.
  compare --a X.jsonl --b Y.jsonl [--ref R.jsonl] --out F
          Greedy fixture captures (w0_drive.py parity): prompts whose token streams are identical, a vs b and a vs ref.
  summary --scratch S --stamp STAMP --out F
          Read the cells' receipts (S/gt1-res-<cell>-STAMP.*), print the verdict line
            GT3: R0 <ms> | R2a <ms> | R2b <ms> | T90 <s> | cold share <a> -> <b> | audit <n> mismatches | C32 <pass>
          and the pre-registered adopt bar: R2b <= 0.90 x R0, T90 <= 20 s, audits 0, C32 alive, R4 within 2 % of R0.

Definitions (fixed before the card runs):
  cold share   per drain, from the server's telemetry: picks of experts that were cold at the drain / all counted
               picks since the previous drain (decode-sized calls only).
  a -> b       a = the first drain with counted picks after the unfreeze is acknowledged (the boot map on this
               traffic: swaps are queued at that drain and applied after it); b = the mean of the last quarter of the
               window's drains (at least 3).
  T90          seconds from the acknowledgement to the first drain whose cold share is <= b + 0.1 x (a - b). "no
               reduction" when a <= b. This is v26ar's telemetry T90 (the lane skill gates convergence on it, never on
               raw tok/s windows over rotating prompts).
  ms/step      w0_drive.py step: decode wall / (spec drafts - 1), median of the cell's reps; R2b = the reps of the
               fifth (last) cell.
Every number comes from this run's receipts; a missing receipt prints "not measured".
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NOT_MEASURED = "not measured"
ADOPT = {"r2b_over_r0_max": 0.90, "t90_s_max": 20.0, "audit_mismatches_max": 0, "r4_within": 0.02}
NUDGE_TOKENS = 128
UNFREEZE = {"freeze": False, "baseline": True}


def w0():
    """w0/w0_drive.py (stdlib): FIXTURE, STEP_PROMPT, one_stream, env W0_BASE / W0_MODEL."""
    p = os.path.join(REPO, "w0")
    if p not in sys.path:
        sys.path.insert(0, p)
    import w0_drive

    return w0_drive


# ================================================================================================ pure
def read_telemetry(path: str) -> List[dict]:
    """The telemetry lines that parse (a line being written is skipped, not an error)."""
    rows = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
    except OSError:
        return []
    return rows


def find_ack(rows: List[dict], seq) -> Optional[dict]:
    for r in rows:
        if r.get("event") == "ctrl" and r.get("seq") == seq:
            return r
    return None


def drains_after(rows: List[dict], t0: float, t1: Optional[float] = None) -> List[dict]:
    return [r for r in rows if r.get("event") == "drain" and r.get("t", 0) >= t0 and (t1 is None or r["t"] <= t1)]


def last_state(rows: List[dict], t0: float) -> Optional[dict]:
    """The newest drain / ctrl record at or after t0 (they carry pending, at_boot, freeze)."""
    for r in reversed(rows):
        if r.get("event") in ("drain", "ctrl") and r.get("t", 0) >= t0:
            return r
    return None


def settled(rows: List[dict], ack: dict, wait: str) -> bool:
    if wait == "ack":
        return True
    s = last_state(rows, ack.get("t", 0))
    if s is None or s.get("pending") != 0:
        return False
    return wait == "idle" or bool(s.get("at_boot"))


def cold_share_series(drains: List[dict]) -> List[tuple]:
    """[(t, cold share)] of the drains that counted picks."""
    return [(float(r["t"]), float(r["cold"]) / float(r["sel"])) for r in drains if r.get("sel")]


def t90(series: List[tuple], t0: float) -> dict:
    """a, b, the 90 % threshold and T90 (module docstring) of a [(t, cold share)] series that starts at the
    acknowledgement time t0."""
    out = {"n_drains": len(series), "a": None, "b": None, "threshold": None, "t90_s": None, "note": None}
    if len(series) < 4:
        out["note"] = f"{len(series)} drains with counted picks: too few for a plateau"
        return out
    k = max(3, len(series) // 4)
    a = series[0][1]
    b = statistics.mean(s for _, s in series[-k:])
    out.update({"a": a, "b": b, "plateau_drains": k, "t_first_drain_s": round(series[0][0] - t0, 3)})
    if a <= b:
        out["note"] = "no reduction (first drain <= plateau)"
        return out
    thr = b + 0.1 * (a - b)
    out["threshold"] = thr
    for t, s in series:
        if s <= thr:
            out["t90_s"] = round(t - t0, 3)
            break
    return out


def swaps_per_drain(drains: List[dict]) -> dict:
    xs = [int(r.get("swaps_since") or 0) for r in drains]
    return {"drains": len(xs), "total": sum(xs), "max": max(xs) if xs else None,
            "mean": round(sum(xs) / len(xs), 3) if xs else None, "series": xs}


def step_cell(path: str) -> Optional[dict]:
    """Medians of one w0_drive.py step receipt."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    reps = [r for r in d.get("reps", []) if r.get("step_ms_from_metrics") and r.get("tokens_per_step")]
    if not reps:
        return None
    ms = [r["step_ms_from_metrics"] for r in reps]
    tps = [r["tokens_per_step"] for r in reps]
    return {"reps": len(reps), "step_ms": round(statistics.median(ms), 3), "step_ms_reps": ms,
            "tokens_per_step": round(statistics.median(tps), 3),
            "ms_per_token": round(statistics.median(m / t for m, t in zip(ms, tps)), 3), "receipt": path}


def load_capture(path: str) -> Optional[dict]:
    """{tag: token list (or the text when the capture has no logprobs)} of a w0_drive.py parity capture."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    toks = [c.get("token") for c in (r.get("logprobs") or [])]
                    out[r["tag"]] = toks if toks else r.get("text")
    except (OSError, ValueError, KeyError):
        return None
    return out or None


def identical(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    if not a or not b:
        return None
    tags = sorted(set(a) & set(b))
    same = [t for t in tags if a[t] == b[t]]
    return {"identical": len(same), "prompts": len(tags), "differ": [t for t in tags if t not in same]}


def _load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def build_summary(S: str, stamp: str) -> dict:
    """Everything the verdict needs, from the cells' receipts under S (None where a receipt is missing)."""
    p = lambda name: os.path.join(S, f"gt1-res-{name}-{stamp}")  # noqa: E731
    r0, r2a, r4 = step_cell(p("r0-step") + ".json"), step_cell(p("r2a-step") + ".json"), step_cell(p("r4-step") + ".json")
    r2b_cells = [step_cell(p(f"r2b-step{i}") + ".json") for i in range(1, 6)]
    r2b = r2b_cells[-1]
    r1 = _load(p("r1-c1") + ".json") or {}
    conv = r1.get("convergence") or {}
    audits = []
    for name in ("r4-audit1", "r4-audit2"):
        a = ((_load(p(name) + ".json") or {}).get("ack") or {}).get("audit")
        audits.append(None if not a else a.get("mismatches"))
    restore = (_load(p("r4-restore") + ".json") or {})
    c32 = ((_load(p("r5-c32") + ".json") or {}).get("summary") or {})
    cap0, capa, capb = (load_capture(p(n) + ".jsonl") for n in ("r0-parity", "r3-parity-a", "r3-parity-b"))
    ms = lambda c: None if c is None else c["step_ms"]  # noqa: E731
    ratio = lambda x, y: round(x / y, 4) if x and y else None  # noqa: E731
    n_audit = None if any(a is None for a in audits) else sum(audits)
    s = {"stamp": stamp, "R0": r0, "R2a": r2a, "R2b": r2b, "R2b_cells": r2b_cells, "R4": r4,
         "R1": {"a": conv.get("a"), "b": conv.get("b"), "t90_s": conv.get("t90_s"), "note": conv.get("note"),
                "swaps_per_drain": r1.get("swaps_per_drain"), "requests": len(r1.get("requests") or []),
                "tok_s_median": r1.get("tok_s_median")},
         "R3": {"self": identical(capa, capb), "vs_r0_map": identical(capa, cap0)},
         "audit": {"before_restore": audits[0], "after_restore": audits[1], "mismatches": n_audit},
         "restore": {"settled": restore.get("settled"), "ack": bool(restore.get("ack"))},
         "C32": {"pass": c32.get("pass"), "ok": c32.get("ok"), "n": c32.get("n"),
                 "server_alive_after": c32.get("server_alive_after")},
         "ratios": {"R2a_over_R0": ratio(ms(r2a), ms(r0)), "R2b_over_R0": ratio(ms(r2b), ms(r0)),
                    "R4_over_R0": ratio(ms(r4), ms(r0))}}
    s["adopt"] = adopt(s)
    s["verdict"] = verdict_line(s)
    return s


def adopt(s: dict) -> dict:
    """The pre-registered bar. Each criterion True / False, or None when its inputs were not measured."""
    rr, t, n = s["ratios"]["R2b_over_R0"], s["R1"]["t90_s"], s["audit"]["mismatches"]
    r4, c32 = s["ratios"]["R4_over_R0"], s["C32"]["pass"]
    c = {"R2b <= 0.90 x R0": None if rr is None else rr <= ADOPT["r2b_over_r0_max"],
         "T90 <= 20 s": None if t is None else t <= ADOPT["t90_s_max"],
         "audits 0": None if n is None else n <= ADOPT["audit_mismatches_max"],
         "C32 alive": None if c32 is None else bool(c32),
         "R4 within 2 % of R0": None if r4 is None else abs(r4 - 1.0) <= ADOPT["r4_within"]}
    return {"criteria": c, "bar": dict(ADOPT), "adopt": all(v is True for v in c.values()),
            "not_measured": [k for k, v in c.items() if v is None]}


def _f(x, nd=3):
    return NOT_MEASURED if x is None else f"{x:.{nd}f}"


def verdict_line(s: dict) -> str:
    ms = lambda c: None if c is None else c["step_ms"]  # noqa: E731
    c32 = s["C32"]["pass"]
    n = s["audit"]["mismatches"]
    return (f"GT3: R0 {_f(ms(s['R0']))} | R2a {_f(ms(s['R2a']))} | R2b {_f(ms(s['R2b']))} | "
            f"T90 {_f(s['R1']['t90_s'], 1)} | cold share {_f(s['R1']['a'], 4)} \u2192 {_f(s['R1']['b'], 4)} | "
            f"audit {NOT_MEASURED if n is None else n} mismatches | "
            f"C32 {NOT_MEASURED if c32 is None else ('pass' if c32 else 'FAIL')}")


# ================================================================================================ box side
def write_ctrl(d: str, doc: dict) -> int:
    """D/ctrl.json by rename, with a fresh seq (ms clock). Returns the seq."""
    seq = int(time.time() * 1000)
    body = dict(doc, seq=seq)
    tmp = os.path.join(d, f".ctrl.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(body, f)
    os.replace(tmp, os.path.join(d, "ctrl.json"))
    return seq


def nudge_prompts(kind: str) -> List[str]:
    W = w0()
    return [W.STEP_PROMPT] if kind == "step" else [p for _, p in W.FIXTURE]


def drive_until(d: str, seq: int, wait: str, nudge: str, timeout: float) -> dict:
    """Short generations until telemetry acknowledges `seq` and the wait condition holds."""
    W = w0()
    tele = os.path.join(d, "telemetry.jsonl")
    prompts = nudge_prompts(nudge)
    t0 = time.time()
    n = 0
    ack = None
    errors = []
    while time.time() - t0 < timeout:
        rows = read_telemetry(tele)
        ack = ack or find_ack(rows, seq)
        if ack is not None and settled(rows, ack, wait):
            return {"ack": ack, "settled": True, "nudges": n, "wait_s": round(time.time() - t0, 3),
                    "state": last_state(rows, ack.get("t", 0)), "errors": errors}
        try:
            W.one_stream(prompts[n % len(prompts)], NUDGE_TOKENS)
        except Exception as e:  # noqa: BLE001 - reported, and the loop times out if the server is gone
            errors.append(repr(e)[:200])
            time.sleep(2)
        n += 1
    rows = read_telemetry(tele)
    return {"ack": ack or find_ack(rows, seq), "settled": False, "nudges": n, "wait_s": round(time.time() - t0, 3),
            "state": None if ack is None else last_state(rows, ack.get("t", 0)), "errors": errors}


def cmd_ctrl(a) -> int:
    doc = json.loads(a.set)
    seq = write_ctrl(a.dir, doc)
    res = drive_until(a.dir, seq, a.wait, a.nudge, a.timeout)
    res.update({"seq": seq, "ctrl": doc, "wait": a.wait, "nudge": a.nudge, "date": time.strftime("%F %T %Z")})
    ack = res["ack"] or {}
    print(f"ctrl {json.dumps(doc)} seq {seq}: ack {'yes' if res['ack'] else 'NO'} after {res['wait_s']} s / "
          f"{res['nudges']} nudges; {ack.get('msg')}; settled({a.wait}) {res['settled']}; "
          f"state {json.dumps({k: (res['state'] or {}).get(k) for k in ('pending', 'at_boot', 'freeze', 'hot_moved', 'swaps')})}"
          + (f"; audit {json.dumps({k: ack['audit'].get(k) for k in ('checked', 'moved', 'random', 'mismatches', 'ms')})}"
             if isinstance(ack.get("audit"), dict) else ""), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["ack"] and res["settled"] else 1


def cmd_c1(a) -> int:
    W = w0()
    tele = os.path.join(a.dir, "telemetry.jsonl")
    doc = json.loads(a.ctrl) if a.ctrl else dict(UNFREEZE)
    seq = write_ctrl(a.dir, doc)
    prompts = list(W.FIXTURE)
    rows, ack, i = [], None, 0
    t_start = time.time()
    while True:
        now = time.time()
        if ack is None:
            ack = find_ack(read_telemetry(tele), seq)
            if ack is None and now - t_start > a.ack_timeout:
                break
        if ack is not None and now - ack["t"] >= a.seconds:
            break
        tag, prompt = prompts[i % len(prompts)]
        t0 = time.time()
        try:
            t_first, t_last, n, usage, _ = W.one_stream(prompt, a.tokens)
            ct = (usage or {}).get("completion_tokens")
            dec = (t_last - t_first) if t_first and t_last else None
            rows.append({"i": i, "tag": tag, "t0": round(t0, 3), "wall_s": round(time.time() - t0, 3),
                         "ttft_s": round((t_first or time.time()) - t0, 3), "completion_tokens": ct, "chunks": n,
                         "decode_s": None if dec is None else round(dec, 3),
                         "tok_s": round(ct / dec, 2) if ct and dec else None})
        except Exception as e:  # noqa: BLE001
            rows.append({"i": i, "tag": tag, "t0": round(t0, 3), "error": repr(e)[:200]})
            time.sleep(1)
        i += 1
    t_end = time.time()
    tel = read_telemetry(tele)
    out = {"date": time.strftime("%F %T %Z"), "ctrl": doc, "seq": seq, "ack": ack, "seconds": a.seconds,
           "tokens": a.tokens, "requests": rows, "errors": sum(1 for r in rows if "error" in r)}
    if ack is not None:
        dr = drains_after(tel, ack["t"], t_end)
        series = cold_share_series(dr)
        out.update({"drains": dr, "convergence": t90(series, ack["t"]), "swaps_per_drain": swaps_per_drain(dr),
                    "cold_share_series": [[round(t - ack["t"], 3), round(s, 6)] for t, s in series]})
        late = [r["tok_s"] for r in rows if r.get("tok_s") and r["t0"] >= ack["t"]]
        out["tok_s_median"] = round(statistics.median(late), 2) if late else None
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    c = out.get("convergence") or {}
    print(f"c1: {len(rows)} requests ({out['errors']} errors), ack {'yes' if ack else 'NO'}; cold share "
          f"{_f(c.get('a'), 4)} -> {_f(c.get('b'), 4)}; T90 {_f(c.get('t90_s'), 1)} s ({c.get('note')}); swaps "
          f"{(out.get('swaps_per_drain') or {}).get('total')} over {(out.get('swaps_per_drain') or {}).get('drains')} "
          f"drains (max {(out.get('swaps_per_drain') or {}).get('max')}/drain); tok/s median {out.get('tok_s_median')}",
          flush=True)
    return 0 if ack is not None and not out["errors"] else 1


def cmd_compare(a) -> int:
    A, B = load_capture(a.a), load_capture(a.b)
    ref = load_capture(a.ref) if a.ref else None
    out = {"a": a.a, "b": a.b, "ref": a.ref, "self": identical(A, B), "vs_ref": identical(A, ref)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    s, r = out["self"] or {}, out["vs_ref"] or {}
    print(f"compare: self-identical {s.get('identical')}/{s.get('prompts')} (differ {s.get('differ')}); vs the R0-map "
          f"capture {r.get('identical')}/{r.get('prompts')} (differ {r.get('differ')})", flush=True)
    return 0 if out["self"] else 1


def cmd_summary(a) -> int:
    s = build_summary(a.scratch, a.stamp)
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
    print(s["verdict"], flush=True)
    ad = s["adopt"]
    print("GT3 adopt bar: " + "; ".join(f"{k}: {NOT_MEASURED if v is None else v}" for k, v in ad["criteria"].items())
          + f" -> {'ADOPT' if ad['adopt'] else 'not adopted'}", flush=True)
    r3 = s["R3"]
    print(f"GT3 detail: tok/step R0 {(s['R0'] or {}).get('tokens_per_step')} R2b {(s['R2b'] or {}).get('tokens_per_step')}; "
          f"R4 {_f((s['R4'] or {}).get('step_ms'))} ms/step (x{s['ratios']['R4_over_R0']} of R0); R3 self-identical "
          f"{(r3['self'] or {}).get('identical')}/{(r3['self'] or {}).get('prompts')}, vs R0 map "
          f"{(r3['vs_r0_map'] or {}).get('identical')}/{(r3['vs_r0_map'] or {}).get('prompts')}; audits "
          f"{s['audit']['before_restore']} / {s['audit']['after_restore']}; R1 swaps "
          f"{(s['R1']['swaps_per_drain'] or {}).get('total')}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="card GT3 driver (adaptive residency cells)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ctrl")
    c.add_argument("--dir", required=True)
    c.add_argument("--set", required=True, help="the control document (JSON object)")
    c.add_argument("--nudge", choices=["fixture", "step"], default="fixture")
    c.add_argument("--wait", choices=["ack", "idle", "boot"], default="ack")
    c.add_argument("--timeout", type=float, default=300.0)
    c.add_argument("--out", default=None)
    c1 = sub.add_parser("c1")
    c1.add_argument("--dir", required=True)
    c1.add_argument("--seconds", type=float, default=90.0)
    c1.add_argument("--tokens", type=int, default=256)
    c1.add_argument("--ctrl", default=None, help="control document written at the start (default: unfreeze + baseline)")
    c1.add_argument("--ack-timeout", type=float, default=120.0)
    c1.add_argument("--out", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    cp.add_argument("--ref", default=None)
    cp.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--scratch", required=True)
    s.add_argument("--stamp", required=True)
    s.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"ctrl": cmd_ctrl, "c1": cmd_c1, "compare": cmd_compare, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
