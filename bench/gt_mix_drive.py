#!/usr/bin/env python3
"""Card GT4: mixed-traffic step cells for gracetier + adaptive residency (Milo, 2026-11-09). Stdlib only.

GT3's adaptive number (8.20 ms/step) is one prompt repeated; its held-out number (9.60) is a frozen map fitted to other
traffic. This driver measures the case in between: many different prompts, one after another, C1, greedy.

  pass    --set A|B --tokens 256 --out <json>     each of the set's 20 prompts once, in order; per prompt the step
                                                  count from /metrics (spec drafts) and the stream's decode wall.
                                                  Reported: ms/step = sum(decode wall) / sum(steps - 1), tok/step,
                                                  ms/token, per-prompt rows. No warm-up request (a warm-up would be
                                                  traffic the residency policy sees).
  summary --scratch <dir> --stamp <stamp> --out <json>   the GT4 table and the verdict line from the pass receipts.

Sets A and B are disjoint, 20 prompts each, the same ten task kinds in both, none of them the w0 fixture or step
prompt. Env: W0_BASE / W0_MODEL (w0/w0_drive.py).

Cells (bench/box/card-gt1.sh, GT1_RES=1 GT1_RES_CELLS=mix):
  M0a / M0b  frozen at the boot map: pass A, pass B            = same-boot static controls
  M1a..c     unfreeze; pass A three times                      = adapting, then adapted to A
  M2         pass B (first sight, policy live)                 = new traffic on a map fitted to other traffic
  M3         pass A again                                      = did B displace A (thrash)?
  M4         pass B again                                      = B once seen
  M5         freeze, audit, restore_map, audit, pass A         = back at M0a within 2 %, 0 mismatches
Pre-registered reading (reported, no adopt bar: GT3 already adopted): mixed-traffic gain = M2 / M0b (first sight) and
(M3 + M4) / (M0a + M0b) (seen, interleaved). Thrash = M3 / M1c > 1.03.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SETS = {
    "A": [
        ("code-py", "Write a Python class `LRUCache` with get and put in O(1) using only the standard library. Include type hints and a short usage example."),
        ("code-rs", "Write a Rust function that parses a comma-separated line of signed integers into a Vec<i64>, returning a Result with a custom error enum. Add two unit tests."),
        ("sql", "Given tables orders(id, customer_id, total, created_at) and customers(id, name, country), write a SQL query for the top 5 customers by total spend in 2024 per country, and explain the window function you used."),
        ("json", "Return ONLY a JSON array of five objects, each with keys city, country, population_millions and timezone, for five large Asian cities. No prose."),
        ("shell", "Explain step by step what this command does and one risk of running it: find . -name '*.log' -mtime +30 -print0 | xargs -0 gzip -9"),
        ("math", "A tank holds 2,400 litres and drains at 37.5 litres per minute while being filled at 12 litres per minute. How long until it is one quarter full? Show each step."),
        ("history", "Describe the main causes and consequences of the 1848 revolutions in Europe in about 250 words."),
        ("cooking", "Give a recipe for a weeknight chicken and rice dish for four people, with quantities in grams, timing for each step, and two substitutions."),
        ("translate", "Translate into Spanish, then into French: 'The meeting has been moved to Thursday afternoon because the supplier could not deliver the parts on time.' Then explain one grammatical difference between the two translations."),
        ("story", "Write a 200-word story about a lighthouse keeper who finds a message in a bottle addressed to her by name."),
        ("code-js", "Write a JavaScript debounce function with a leading-edge option and explain when you would use leading versus trailing."),
        ("legal", "Explain in plain language the difference between a warranty and an indemnity in a commercial contract, with one example of each."),
        ("bio", "Explain how mRNA vaccines cause cells to produce an antigen, from injection to antibody response, for a high-school student."),
        ("email", "Write a polite but firm email to a landlord asking for a broken heating system to be repaired within seven days, citing two previous unanswered requests."),
        ("table", "Compare electric, hybrid and diesel cars in a markdown table with rows for purchase cost, running cost, range, refuelling time and maintenance, then give a one-sentence summary."),
        ("poem", "Write a sonnet about the first frost of autumn in a mountain town."),
        ("physics", "Explain why the sky is blue and sunsets are red, including the wavelength dependence of Rayleigh scattering."),
        ("tool", "You have a tool `get_weather(city, unit)`. The user asks: should I bring an umbrella in Lisbon tomorrow? Say exactly which tool call you would make as a JSON object with keys name and arguments, then stop."),
        ("finance", "Explain the difference between a traditional IRA and a Roth IRA, and who typically benefits more from each, in about 200 words."),
        ("debug", "This Python code raises 'RuntimeError: dictionary changed size during iteration'. Explain why and give two fixes:\nfor k in d:\n    if d[k] is None:\n        del d[k]"),
    ],
    "B": [
        ("code-py", "Write a Python generator that yields prime numbers using an incremental sieve, with a docstring and an example that prints the first twenty."),
        ("code-go", "Write a Go function that fetches a list of URLs concurrently with a limit of four in flight, returning the status code of each. Explain how you bound the concurrency."),
        ("sql", "Given a table events(user_id, event_type, ts), write a SQL query that finds users whose first 'purchase' came within 7 days of their first 'signup', and explain the join."),
        ("json", "Return ONLY a JSON object describing a weekly gym plan: keys are the seven weekday names, each value an object with keys focus, exercises (array of three strings) and minutes. No prose."),
        ("shell", "Write a bash script that backs up a directory given as its first argument to a dated tar.gz under /backup, keeps only the seven newest archives, and exits non-zero on any failure. Comment each section."),
        ("math", "A loan of 18,000 at 6 percent annual interest, compounded monthly, is repaid over 5 years. Derive the monthly payment formula and compute the payment."),
        ("history", "Describe how the printing press changed European society between 1450 and 1600 in about 250 words."),
        ("cooking", "Explain how to make sourdough bread from an active starter, with a timeline across two days, hydration percentage, and three common mistakes."),
        ("translate", "Translate into German, then into Italian: 'Please keep the receipt, because without it we cannot exchange the shoes or refund your money.' Then explain one difference in word order between the two."),
        ("story", "Write a 200-word story about two rival chess players who meet again as old men on a train."),
        ("code-c", "Write a C function that reverses a singly linked list in place and explain the pointer updates line by line."),
        ("legal", "Explain in plain language what 'force majeure' means in a contract and when a party can usually rely on it, with one example."),
        ("bio", "Explain how the kidneys filter blood and regulate water balance, naming the parts of the nephron, for a high-school student."),
        ("email", "Write a short email declining a job offer politely while leaving the door open for future roles, mentioning that the decision was about location."),
        ("table", "Compare PostgreSQL, SQLite and MongoDB in a markdown table with rows for data model, concurrency, typical use, operational effort and licensing, then give a one-sentence recommendation for a small web app."),
        ("poem", "Write a poem in free verse about a city waking up before dawn, around sixteen lines."),
        ("physics", "Explain how a heat pump can deliver more heat energy than the electrical energy it consumes, including what the coefficient of performance means."),
        ("tool", "You have a tool `search_flights(origin, destination, date)`. The user asks: find me a flight from Denver to Boston on the 3rd of March. Say exactly which tool call you would make as a JSON object with keys name and arguments, then stop."),
        ("finance", "Explain what an index fund is, how its fees compare with an actively managed fund, and one drawback, in about 200 words."),
        ("debug", "This JavaScript prints 3 three times instead of 0, 1, 2. Explain why and give two fixes:\nfor (var i = 0; i < 3; i++) { setTimeout(function () { console.log(i); }, 0); }"),
    ],
}


def w0():
    p = os.path.join(REPO, "w0")
    if p not in sys.path:
        sys.path.insert(0, p)
    import w0_drive

    return w0_drive


def aggregate(rows):
    ok = [r for r in rows if r.get("steps") and r["steps"] > 1 and r.get("decode_wall_s")]
    steps = sum(r["steps"] - 1 for r in ok)
    wall = sum(r["decode_wall_s"] for r in ok)
    gen = sum(r["gen_tokens"] for r in ok)
    all_steps = sum(r["steps"] for r in ok)
    tk = [r for r in rows if r.get("completion_tokens") and r.get("decode_wall_s")]
    ctok = sum(r["completion_tokens"] for r in tk)
    cwall = sum(r["decode_wall_s"] for r in tk)
    return {"prompts": len(rows), "prompts_measured": len(ok), "errors": sum(1 for r in rows if r.get("error")),
            "prompts_timed": len(tk), "completion_tokens": ctok,
            # client-side decode rate: completion tokens over first-to-last content chunk, every server reports it
            "decode_tok_s": round(ctok / cwall, 2) if cwall else None,
            "steps": all_steps, "gen_tokens": gen,
            "step_ms": round(1000 * wall / steps, 3) if steps else None,
            "tokens_per_step": round(gen / all_steps, 3) if all_steps else None,
            "ms_per_token": round(1000 * wall / steps / (gen / all_steps), 3) if steps and gen else None}


def cmd_pass(a) -> int:
    W = w0()
    rows = []
    t_start = time.time()
    for kind, prompt in SETS[a.set]:
        rec = {"kind": kind}
        try:
            m0 = W.metrics()
            t0 = time.time()
            t_first, t_last, n, usage, _ = W.one_stream(prompt, a.tokens)
            m1 = W.metrics()
            have = "num_drafts_total" in m0 and "num_drafts_total" in m1   # absent on a server without spec counters
            steps = (m1["num_drafts_total"] - m0["num_drafts_total"]) if have else None
            rec.update({"ttft_s": round((t_first or t0) - t0, 3),
                        "decode_wall_s": round(t_last - t_first, 4) if t_first and t_last else None, "chunks": n,
                        "steps": steps,
                        "gen_tokens": (m1["generation_tokens_total"] - m0["generation_tokens_total"])
                        if "generation_tokens_total" in m0 and "generation_tokens_total" in m1 else None,
                        "completion_tokens": (usage or {}).get("completion_tokens")})
            if steps and steps > 1 and rec["decode_wall_s"]:
                rec["step_ms"] = round(1000 * rec["decode_wall_s"] / (steps - 1), 3)
        except Exception as e:  # reported, never hidden
            rec["error"] = repr(e)[:200]
        rows.append(rec)
    res = {"set": a.set, "tokens": a.tokens, "base": W.BASE, "model": W.MODEL, "t_start": t_start, "t_end": time.time(),
           "agg": aggregate(rows), "rows": rows}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    g = res["agg"]
    print(f"pass {a.set}: {g['prompts_measured']}/{g['prompts']} prompts, errors {g['errors']}; decode tok/s "
          f"{g['decode_tok_s']} ({g['prompts_timed']} timed); ms/step {g['step_ms']} "
          f"tok/step {g['tokens_per_step']} ms/token {g['ms_per_token']}; per-prompt ms/step min "
          f"{min((r['step_ms'] for r in rows if r.get('step_ms')), default=None)} max "
          f"{max((r['step_ms'] for r in rows if r.get('step_ms')), default=None)}", flush=True)
    return 0 if g["errors"] == 0 and g["prompts_timed"] == g["prompts"] else 1


def cmd_conc(a) -> int:
    """N concurrent streams. Requests = the 40 prompts of A + B, cycled, each with a salt in front (same salts for
    every arm, never repeated inside an arm, so no arm gets a prefix-cache hit another did not). Reported: aggregate
    tok/s = completion tokens / wall of the whole cell; per-stream decode tok/s; TTFT."""
    import threading

    W = w0()
    prompts = [p for _, p in SETS["A"] + SETS["B"]]
    total = a.requests or max(40, 2 * a.n)
    jobs = [(i, f"(request {a.tag}-{i}) " + prompts[i % len(prompts)]) for i in range(total)]
    rows, lock, nxt = [], threading.Lock(), [0]

    def worker():
        while True:
            with lock:
                if nxt[0] >= len(jobs):
                    return
                i, prompt = jobs[nxt[0]]
                nxt[0] += 1
            rec = {"i": i}
            t0 = time.time()
            try:
                t_first, t_last, n, usage, _ = W.one_stream(prompt, a.tokens)
                ct = (usage or {}).get("completion_tokens")
                rec.update({"t0": t0, "t_end": time.time(), "ttft_s": round((t_first or t0) - t0, 3),
                            "decode_wall_s": round(t_last - t_first, 4) if t_first and t_last else None,
                            "completion_tokens": ct})
                if ct and rec["decode_wall_s"]:
                    rec["decode_tok_s"] = round(ct / rec["decode_wall_s"], 2)
            except Exception as e:
                rec["error"] = repr(e)[:200]
            with lock:
                rows.append(rec)

    t_start = time.time()
    th = [threading.Thread(target=worker, daemon=True) for _ in range(a.n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t_start
    ok = [r for r in rows if r.get("completion_tokens")]
    ctok = sum(r["completion_tokens"] for r in ok)
    ds = sorted(r["decode_tok_s"] for r in ok if r.get("decode_tok_s"))
    tt = sorted(r["ttft_s"] for r in ok)
    med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    alive = True
    try:
        W.metrics()
    except Exception:
        alive = False
    agg = {"n": a.n, "requests": total, "ok": len(ok), "errors": len(rows) - len(ok), "wall_s": round(wall, 2),
           "completion_tokens": ctok, "agg_tok_s": round(ctok / wall, 1) if wall else None,
           "stream_decode_tok_s_p50": med(ds), "stream_decode_tok_s_min": ds[0] if ds else None,
           "ttft_s_p50": med(tt), "ttft_s_max": tt[-1] if tt else None, "server_alive_after": alive}
    with open(a.out, "w") as f:
        json.dump({"tag": a.tag, "tokens": a.tokens, "base": W.BASE, "model": W.MODEL, "agg": agg, "rows": rows}, f, indent=1)
    print(f"conc C{a.n}: {agg['ok']}/{total} ok, errors {agg['errors']}; aggregate {agg['agg_tok_s']} tok/s over "
          f"{agg['wall_s']} s; per-stream decode p50 {agg['stream_decode_tok_s_p50']} min {agg['stream_decode_tok_s_min']} "
          f"tok/s; TTFT p50 {agg['ttft_s_p50']} max {agg['ttft_s_max']} s; alive {alive}", flush=True)
    return 0 if agg["errors"] == 0 and alive else 1


WORDS = ("time year people way day man thing woman life child world school state family student group country problem "
         "hand part place case week company system program question work government number night point home water room "
         "mother area money story fact month lot right study book eye job word business issue side kind head house "
         "service friend father power hour game line end member law car city community name president team minute idea "
         "kid body information back parent face others level office door health person art war history party result "
         "change morning reason research girl guy moment air teacher force education").split()


def cmd_ttft(a) -> int:
    """Time to first token by prompt length: max_tokens 1, non-streamed wall time, `usage.prompt_tokens` as the length.
    Each run's prompt starts with its own salt and is seeded pseudo-random common words, so no run can hit the prefix
    cache of another."""
    import random

    W = w0()
    out = []
    for size in [int(x) for x in a.sizes.split(",")]:
        rec = {"target_tokens": size, "runs": []}
        for r in range(a.reps):
            rng = random.Random(f"{size}-{r}-{a.salt}")
            n = max(4, int(size / 1.05) - 24)
            text = f"(ttft {a.salt} {size} {r}) " + " ".join(rng.choice(WORDS) for _ in range(n)) + "\nReply with the single word OK."
            t0 = time.time()
            try:
                d = json.loads(W.post("/v1/chat/completions", W.chat_body(text, 1), timeout=1800))
                rec["runs"].append({"s": round(time.time() - t0, 3), "prompt_tokens": d["usage"]["prompt_tokens"]})
            except Exception as e:
                rec["runs"].append({"error": repr(e)[:200]})
        ok = sorted(x["s"] for x in rec["runs"] if "s" in x)
        rec["median_s"] = ok[len(ok) // 2] if ok else None
        rec["prompt_tokens"] = next((x["prompt_tokens"] for x in rec["runs"] if "prompt_tokens" in x), None)
        if rec["median_s"] and rec["prompt_tokens"]:
            rec["prefill_tok_s"] = round(rec["prompt_tokens"] / rec["median_s"])
        out.append(rec)
    with open(a.out, "w") as f:
        json.dump({"base": W.BASE, "model": W.MODEL, "salt": a.salt, "sizes": out}, f, indent=1)
    print("ttft: " + "; ".join(f"{x['prompt_tokens']} tok {x['median_s']} s" for x in out)
          + f"; errors {sum(1 for x in out for y in x['runs'] if 'error' in y)}", flush=True)
    return 0


CELLS = ["m0a", "m0b", "m1a", "m1b", "m1c", "m2", "m3", "m4", "m6a", "m6b", "m5"]


def cmd_summary(a) -> int:
    s = {"stamp": a.stamp, "cells": {}}
    for c in CELLS:
        p = glob.glob(os.path.join(a.scratch, f"gt1-res-{c}-pass-{a.stamp}.json"))
        if p:
            d = json.load(open(p[0]))
            s["cells"][c] = dict(d["agg"], set=d["set"])
    g = lambda c: (s["cells"].get(c) or {}).get("step_ms")  # noqa: E731
    r = lambda x, y: round(x / y, 4) if x and y else None  # noqa: E731
    s["ratios"] = {
        "first_sight_M2_over_M0b": r(g("m2"), g("m0b")),
        "adapted_A_M1c_over_M0a": r(g("m1c"), g("m0a")),
        "seen_interleaved_(M3+M4)_over_(M0a+M0b)": r((g("m3") or 0) + (g("m4") or 0), (g("m0a") or 0) + (g("m0b") or 0))
        if g("m3") and g("m4") and g("m0a") and g("m0b") else None,
        "thrash_M3_over_M1c": r(g("m3"), g("m1c")),
        "restore_M5_over_M0a": r(g("m5"), g("m0a")),
    }
    audits = []
    for p in sorted(glob.glob(os.path.join(a.scratch, f"gt1-res-m5-audit*-{a.stamp}.json"))):
        try:
            d = json.load(open(p))
            audits.append((((d.get("ack") or {}).get("audit")) or d.get("audit") or {}).get("mismatches"))
        except Exception as e:
            audits.append(repr(e)[:80])
    s["audit_mismatches"] = audits
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
    f3 = lambda x: "not measured" if x is None else f"{x:.3f}"  # noqa: E731
    print("GT4: " + " | ".join(f"{c.upper()}({(s['cells'].get(c) or {}).get('set', '?')}) {f3(g(c))}" for c in CELLS))
    print("GT4 ratios: " + "; ".join(f"{k} {f3(v)}" for k, v in s["ratios"].items()) + f"; audits {audits}")
    print("GT4 tok/step: " + " ".join(f"{c} {(s['cells'].get(c) or {}).get('tokens_per_step')}" for c in CELLS))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pass")
    p.add_argument("--set", choices=sorted(SETS), required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--scratch", required=True)
    s.add_argument("--stamp", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("conc")
    c.add_argument("--n", type=int, required=True)
    c.add_argument("--requests", type=int, default=0, help="default max(40, 2 x n)")
    c.add_argument("--tokens", type=int, default=256)
    c.add_argument("--tag", required=True, help="salt tag: use the same tag for the same cell in every arm")
    c.add_argument("--out", required=True)
    t = sub.add_parser("ttft")
    t.add_argument("--sizes", default="32,300,1000,8000,64000")
    t.add_argument("--reps", type=int, default=3)
    t.add_argument("--salt", default=str(int(time.time())))
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"pass": cmd_pass, "summary": cmd_summary, "conc": cmd_conc, "ttft": cmd_ttft}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
