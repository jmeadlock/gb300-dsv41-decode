#!/usr/bin/env python3
"""parity.py — the M0 parity instrument (plan §7).

Compares a candidate greedy run against the Pg61 reference captures (w0/results/phase1/parity-A.jsonl, parity-A2.jsonl).
Candidate format = the same JSONL the W0 driver wrote (tag, text, logprobs[{token, logprob, top_logprobs[...]}], n_tokens),
produced by the runtime's own `bench/w0_drive.py parity`-compatible path, or by `--from-runtime <jsonl>`.

Metrics per prompt:
  - first_divergence: first position where the candidate token != reference token (None if identical over min length)
  - top1_flips_weighted: sum over positions (before divergence) of |margin| where candidate's argmax != reference's argmax,
    margin = ref top1 logprob − ref top2 logprob (small-margin flips are cheap; large-margin flips are real errors)
  - max_abs_dlogprob: max |cand.logprob(tok) − ref.logprob(tok)| over the shared prefix (bf16-level noise ≈ 1e-2 is expected)
Pass rule (M0): the candidate vs reference A must be no worse than reference A2 vs A on every metric:
  identical prompts ≥ 7/8; where A2 diverged (code@121) the candidate may diverge at ≥ the same position;
  weighted flips ≤ A2's; max_abs_dlogprob ≤ max(A2's, 0.05).
Usage: python bench/parity.py --ref w0/results/phase1/parity-A.jsonl --ref2 w0/results/phase1/parity-A2.jsonl --cand <jsonl>
       python bench/parity.py --self-test   (A2 vs A: prints the spread the bar is graded against)
"""
import argparse, json, sys

def load(p):
    return {json.loads(l)["tag"]: json.loads(l) for l in open(p) if l.strip()}

def toks(r):
    return [lp["token"] for lp in r["logprobs"]]

def compare(ref, cand):
    rt, ct = toks(ref), toks(cand)
    n = min(len(rt), len(ct)); first = None
    for i in range(n):
        if rt[i] != ct[i]:
            first = i; break
    end = n if first is None else first
    flips = 0.0; n_flips = 0; maxd = 0.0
    for i in range(end):
        rl, cl = ref["logprobs"][i], cand["logprobs"][i]
        rtop = sorted(rl["top_logprobs"], key=lambda t: -t["logprob"])
        margin = (rtop[0]["logprob"] - rtop[1]["logprob"]) if len(rtop) > 1 else 0.0
        ctop = max(cl["top_logprobs"], key=lambda t: t["logprob"])["token"] if cl.get("top_logprobs") else cl["token"]
        if ctop != rtop[0]["token"]:
            flips += abs(margin); n_flips += 1
        maxd = max(maxd, abs(rl["logprob"] - cl["logprob"]))
    return {"len_ref": len(rt), "len_cand": len(ct), "first_divergence": first, "shared_prefix": end,
            "top1_flips": n_flips, "top1_flips_weighted": round(flips, 4), "max_abs_dlogprob": round(maxd, 5)}

def grade(ref, ref2, cand):
    rows = []; ok_all = True
    for tag in ref:
        a2 = compare(ref[tag], ref2[tag]) if ref2 and tag in ref2 else None
        c = compare(ref[tag], cand[tag]) if tag in cand else None
        if c is None:
            rows.append((tag, a2, None, "MISSING")); ok_all = False; continue
        ok = True
        if a2 is not None:
            if a2["first_divergence"] is None and c["first_divergence"] is not None: ok = False
            if a2["first_divergence"] is not None and c["first_divergence"] is not None and c["first_divergence"] < a2["first_divergence"]: ok = False
            if c["top1_flips_weighted"] > a2["top1_flips_weighted"] + 1e-9: ok = False
            if c["max_abs_dlogprob"] > max(a2["max_abs_dlogprob"], 0.05): ok = False
        rows.append((tag, a2, c, "ok" if ok else "FAIL")); ok_all &= ok
    return rows, ok_all

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="w0/results/phase1/parity-A.jsonl")
    ap.add_argument("--ref2", default="w0/results/phase1/parity-A2.jsonl"); ap.add_argument("--cand"); ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", help="write per-prompt metrics here")
    a = ap.parse_args()
    ref = load(a.ref); ref2 = load(a.ref2) if a.ref2 else None
    if a.self_test:
        print(f"{'tag':10s} first_div  flips  flips_w   max|dlogp|")
        for tag in ref:
            assert ref2 is not None, "--self-test needs --ref2"
            m = compare(ref[tag], ref2[tag]); print(f"{tag:10s} {str(m['first_divergence']):9s} {m['top1_flips']:5d} {m['top1_flips_weighted']:8.4f} {m['max_abs_dlogprob']:10.5f}")
        return 0
    cand = load(a.cand); rows, ok = grade(ref, ref2, cand)
    print(f"{'tag':10s} {'A2 first/flips_w/maxd':28s} {'cand first/flips_w/maxd':28s} verdict")
    for tag, a2, c, v in rows:
        f = lambda m: "-" if m is None else f"{str(m['first_divergence']):5s} {m['top1_flips_weighted']:7.3f} {m['max_abs_dlogprob']:7.4f}"
        print(f"{tag:10s} {f(a2):28s} {f(c):28s} {v}")
    n_id = sum(1 for _, _, c, _ in rows if c and c["first_divergence"] is None)
    print(f"identical {n_id}/{len(rows)}; PARITY: {'PASS' if ok and n_id >= 7 else 'FAIL'}")
    if a.json: json.dump({t: {"a2": a2, "cand": c, "verdict": v} for t, a2, c, v in rows}, open(a.json, "w"), indent=1)
    return 0 if ok and n_id >= 7 else 1

if __name__ == "__main__":
    sys.exit(main())
