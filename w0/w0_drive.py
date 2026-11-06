#!/usr/bin/env python3
"""w0_drive.py — W0 request driver for the GB300 DSV4.1 Decode project (Milo, 2026-11-06). Stdlib only.

  parity <out.jsonl>            greedy T=0 chat completions with top-20 logprobs on the fixture prompts (M0 parity reference)
  step   <out.json> [--profile] warm, then one 512-token greedy generation; step time from /metrics spec counters + stream
                                arrival times. --profile wraps the generation in POST /start_profile .. /stop_profile
                                (the nsys-wrapped server then exits on purpose: --capture-range-end=stop-shutdown).
Env: W0_BASE (default http://127.0.0.1:30006), W0_MODEL (default dsv41-flash-uva).
"""
import json, os, sys, time, urllib.request, urllib.error

BASE = os.environ.get("W0_BASE", "http://127.0.0.1:30006")
MODEL = os.environ.get("W0_MODEL", "dsv41-flash-uva")

FIXTURE = [
    ("prose", "Explain in three paragraphs why a mixture-of-experts language model needs its routed experts to be read "
              "from fast memory, and what happens to decode speed when some of them live in host RAM."),
    ("code", "Write a Python function `topk_rows(scores, k)` that returns the indices of the k largest values per row "
             "of a 2-D list without using numpy. Include a short docstring and two asserts."),
    ("math", "A GPU has 7.2 TB/s of memory bandwidth. A decode step must read 448 MiB of expert weights plus 60 MiB of "
             "attention weights. What is the bandwidth-bound floor for the step in microseconds? Show the arithmetic."),
    ("json", "Return ONLY a JSON object with keys name, port, replicas describing a service called 'ds41' on port 30006 "
             "with 1 replica. No prose."),
    ("shell", "Give the exact bash one-liner that lists the five largest files under /var/log, newest first, and explain "
              "each flag in one line each."),
    ("tool", "You have a tool `read_file(path)`. The user asks: what is in /etc/hostname? Say exactly which tool call you "
             "would make, as a JSON object with keys name and arguments, then stop."),
    ("long", "Summarize the trade-offs between tensor parallelism and expert parallelism for serving a 550B MoE model on "
             "one GPU versus four, in a table with four rows and three columns, then one sentence of recommendation."),
    ("story", "Write a 150-word story about a kernel engineer who discovers that the GPU was waiting on a PCIe round trip "
              "the whole time."),
]

STEP_PROMPT = ("Write a detailed technical explanation, about 600 words, of how speculative decoding with a draft head "
               "works in a mixture-of-experts transformer, including why each verify step touches more unique experts "
               "than a plain decode step. Then write a Python function that simulates acceptance length for a given "
               "per-position acceptance probability vector.")


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def post_raw(path, timeout=30):
    req = urllib.request.Request(BASE + path, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()[:200].decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:200].decode(errors="replace")
    except Exception as e:  # server exiting after stop_profile is expected
        return -1, repr(e)[:200]


def metrics():
    want = {"vllm:spec_decode_num_drafts_total": "num_drafts_total",
            "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens_total",
            "vllm:generation_tokens_total": "generation_tokens_total",
            "vllm:num_preemptions_total": "num_preemptions_total"}
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            for w, k in want.items():
                if line.startswith(w + "{"):
                    out[k] = float(line.rsplit(" ", 1)[1])
    return out


def chat_body(prompt, max_tokens, stream=False, logprobs=False):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1,
         "max_tokens": max_tokens, "seed": 1, "chat_template_kwargs": {"thinking": False}, "stream": stream}
    if logprobs:
        b["logprobs"] = True
        b["top_logprobs"] = 20
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


def parity(out):
    n = 0
    with open(out, "w") as f:
        for tag, prompt in FIXTURE:
            t0 = time.time()
            raw = post("/v1/chat/completions", chat_body(prompt, 256, logprobs=True))
            d = json.loads(raw)
            ch = d["choices"][0]
            toks = [c["token"] for c in (ch.get("logprobs") or {}).get("content") or []]
            rec = {"tag": tag, "prompt": prompt, "model": d.get("model"), "wall_s": round(time.time() - t0, 3),
                   "finish_reason": ch.get("finish_reason"), "n_tokens": len(toks),
                   "text": ch["message"]["content"], "logprobs": (ch.get("logprobs") or {}).get("content"),
                   "usage": d.get("usage")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            print(f"parity {tag}: {len(toks)} tokens, {rec['wall_s']} s, finish={rec['finish_reason']}", flush=True)
    print(f"parity: {n} prompts -> {out}", flush=True)


def one_stream(prompt, max_tokens):
    """Return (t_first, t_last, n_chunks, usage) for a streamed greedy generation."""
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(chat_body(prompt, max_tokens, stream=True)).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    t_first = t_last = None; n = 0; usage = None; arrivals = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage") and not d.get("choices"):
                usage = d["usage"]; continue
            ch = (d.get("choices") or [{}])[0]
            if ch.get("delta", {}).get("content"):
                t = time.time(); arrivals.append(t)
                if t_first is None: t_first = t
                t_last = t; n += 1
            if d.get("usage"):
                usage = d["usage"]
    return t_first, t_last, n, usage, arrivals


def step(out, profile):
    # warm: two short generations so prefix cache + graphs are hot
    for _ in range(2):
        one_stream(STEP_PROMPT, 32)
    res = {"base": BASE, "model": MODEL, "profile": profile, "reps": []}
    reps = 1 if profile else 3
    for i in range(reps):
        m0 = metrics(); t0 = time.time()
        if profile:
            res["start_profile"] = post_raw("/start_profile")
        t_first, t_last, n, usage, arrivals = one_stream(STEP_PROMPT, 512)
        t1 = time.time()
        rec = {"rep": i, "wall_s": round(t1 - t0, 3), "ttft_s": round((t_first or t1) - t0, 3),
               "decode_wall_s": round((t_last - t_first), 3) if t_first and t_last else None, "chunks": n, "usage": usage}
        if profile:
            rec["stop_profile"] = post_raw("/stop_profile")
            # the server is expected to exit now; /metrics may fail
            try:
                m1 = metrics()
            except Exception as e:
                m1 = {"error": repr(e)[:120]}
        else:
            m1 = metrics()
        if "num_drafts_total" in m1 and "num_drafts_total" in m0:
            steps = m1["num_drafts_total"] - m0["num_drafts_total"]
            gen = m1["generation_tokens_total"] - m0["generation_tokens_total"]
            acc = m1["num_accepted_tokens_total"] - m0["num_accepted_tokens_total"]
            rec.update({"steps": steps, "gen_tokens": gen, "accepted": acc,
                        "tokens_per_step": round(gen / steps, 3) if steps else None,
                        "step_ms_from_metrics": round(1000 * rec["decode_wall_s"] / (steps - 1), 3) if steps and steps > 1 and rec["decode_wall_s"] else None,
                        "preemptions": m1.get("num_preemptions_total", 0) - m0.get("num_preemptions_total", 0)})
        if arrivals and len(arrivals) > 2:
            gaps = sorted((b - a) * 1000 for a, b in zip(arrivals, arrivals[1:]))
            rec["arrival_gap_ms_p50"] = round(gaps[len(gaps) // 2], 3)
            rec["arrival_gap_ms_mean"] = round(sum(gaps) / len(gaps), 3)
        res["reps"].append(rec)
        print(f"step rep{i}: {json.dumps({k: v for k, v in rec.items() if k != 'usage'})}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    print(f"step: -> {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "parity":
        parity(sys.argv[2])
    elif sys.argv[1] == "step":
        step(sys.argv[2], "--profile" in sys.argv[3:])
    else:
        print(__doc__); sys.exit(2)
