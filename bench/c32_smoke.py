#!/usr/bin/env python3
"""c32_smoke.py -- C32 smoke for an OpenAI-compatible server (card GT1 gate G4): N concurrent /v1/completions
requests, each forced to >= M generated tokens; the gate is "the server does not crash and every request returns its
tokens". No existing C32 driver lives in bench/ or w0/ (searched 2026-11-09), so this one was written for GT1.

  python3 bench/c32_smoke.py --base http://127.0.0.1:30006 --model dsv41-flash-gracetier --n 32 --tokens 128 \
      --json scratch/gt1-c32.json

Each request: /v1/completions, temperature 0, max_tokens = min_tokens = --tokens, ignore_eos (vLLM extra params, so a
short natural answer cannot end the request early), a distinct prompt per request (the 8 W0 fixture prompts cycled,
each prefixed with its request index so prefix caching cannot collapse them). All N are released together from a
barrier. Client: the `openai` package when importable (the handoff's choice), else stdlib urllib -- the receipt says
which. After the burst: GET /v1/models must still answer.

Receipt (--json): per request {i, ok, http/error, completion_tokens, prompt_tokens, latency_s, finish_reason}; summary
{n, ok, min_completion_tokens, all_tokens, wall_s, aggregate_tok_s (= sum completion tokens / burst wall),
latency_s p50/max, server_alive_after, pass}. Every number is measured in this run. Exit 0 = pass, 1 = fail, 2 = usage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

FIXTURE_FALLBACK = ["Explain why a mixture-of-experts model needs fast memory for its routed experts."]


def fixture_prompts() -> List[str]:
    """The 8 W0 fixture prompts (w0/w0_drive.py FIXTURE), read from the file without importing anything else."""
    try:
        sys.path.insert(0, os.path.join(REPO, "w0"))
        import w0_drive  # stdlib-only module

        return [p for _, p in w0_drive.FIXTURE]
    except Exception:  # noqa: BLE001
        return list(FIXTURE_FALLBACK)
    finally:
        if sys.path and sys.path[0] == os.path.join(REPO, "w0"):
            sys.path.pop(0)


def build_prompts(n: int, base: Optional[List[str]] = None) -> List[str]:
    base = base or fixture_prompts()
    return [f"[request {i} of {n}] {base[i % len(base)]}" for i in range(n)]


def request_body(model: str, prompt: str, tokens: int) -> dict:
    return {"model": model, "prompt": prompt, "max_tokens": int(tokens), "min_tokens": int(tokens), "temperature": 0,
            "ignore_eos": True, "seed": 1}


def pct(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 3)


def summarize(rows: List[dict], n: int, tokens: int, wall_s: float, alive: bool) -> dict:
    ok = [r for r in rows if r.get("ok")]
    comp = [int(r.get("completion_tokens") or 0) for r in ok]
    lat = [float(r["latency_s"]) for r in ok if r.get("latency_s") is not None]
    all_tokens = all(c >= tokens for c in comp) and len(ok) == n
    total = sum(comp)
    return {"n": n, "ok": len(ok), "failed": n - len(ok), "tokens_requested": tokens,
            "min_completion_tokens": min(comp) if comp else None, "all_tokens": all_tokens,
            "completion_tokens_total": total, "wall_s": round(wall_s, 3),
            "aggregate_tok_s": round(total / wall_s, 1) if wall_s > 0 else None,
            "latency_s_p50": pct(lat, 0.5), "latency_s_max": max(lat) if lat else None,
            "server_alive_after": bool(alive), "pass": bool(all_tokens and alive)}


def _client(base: str, timeout: float):
    try:
        import openai  # noqa: F401

        c = openai.OpenAI(base_url=base.rstrip("/") + "/v1", api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                          timeout=timeout, max_retries=0)
        return c, f"openai {getattr(openai, '__version__', '?')}"
    except Exception as e:  # noqa: BLE001
        return None, f"urllib (openai not importable: {type(e).__name__})"


def one(i: int, base: str, body: dict, client, timeout: float) -> dict:
    t0 = time.time()
    try:
        if client is not None:
            extra = {k: body[k] for k in ("min_tokens", "ignore_eos")}
            r = client.completions.create(model=body["model"], prompt=body["prompt"], max_tokens=body["max_tokens"],
                                          temperature=0, seed=body["seed"], extra_body=extra)
            u = r.usage
            return {"i": i, "ok": True, "latency_s": round(time.time() - t0, 3),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "finish_reason": r.choices[0].finish_reason if r.choices else None}
        req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read())
        u = d.get("usage") or {}
        ch = (d.get("choices") or [{}])[0]
        return {"i": i, "ok": True, "latency_s": round(time.time() - t0, 3), "http": 200,
                "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
                "finish_reason": ch.get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"i": i, "ok": False, "http": e.code, "error": e.read()[:300].decode(errors="replace"),
                "latency_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "latency_s": round(time.time() - t0, 3)}


def alive(base: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=10) as r:
            return model in r.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C32 smoke: N concurrent /v1/completions x >= M tokens")
    ap.add_argument("--base", default=os.environ.get("W0_BASE", "http://127.0.0.1:30006"))
    ap.add_argument("--model", default=os.environ.get("W0_MODEL", "dsv41-flash-uva"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.n < 1 or a.tokens < 1:
        return 2
    client, how = _client(a.base, a.timeout)
    prompts = build_prompts(a.n)
    rows: List[Optional[dict]] = [None] * a.n
    bar = threading.Barrier(a.n + 1)

    def run(i):
        bar.wait()
        rows[i] = one(i, a.base, request_body(a.model, prompts[i], a.tokens), client, a.timeout)

    ths = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(a.n)]
    for t in ths:
        t.start()
    date = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    bar.wait()
    t0 = time.time()
    for t in ths:
        t.join(a.timeout + 60)
    wall = time.time() - t0
    done: List[dict] = [r if r is not None else {"i": i, "ok": False, "error": "thread did not finish"}
                        for i, r in enumerate(rows)]
    live = alive(a.base, a.model)
    s = summarize(done, a.n, a.tokens, wall, live)
    rec = {"schema": "c32-smoke/1", "date": date, "base": a.base, "model": a.model, "client": how, "summary": s,
           "requests": done}
    print(f"C32 {date}: {s['ok']}/{a.n} ok, min tokens {s['min_completion_tokens']} (>= {a.tokens}: {s['all_tokens']}), "
          f"{s['completion_tokens_total']} tokens in {s['wall_s']} s = {s['aggregate_tok_s']} tok/s aggregate, "
          f"latency p50 {s['latency_s_p50']} s max {s['latency_s_max']} s, server alive after: {live}, client {how} "
          f"-> {'PASS' if s['pass'] else 'FAIL'}", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1)
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
