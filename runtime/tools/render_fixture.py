#!/usr/bin/env python3
"""render_fixture.py — render the W0 fixture prompts with the checkpoint's own encoding.py + tokenizer.json and
compare token counts with Pg61's usage.prompt_tokens. Run inside the library image with /ref (checkpoint root) mounted."""
import json, sys
sys.path.insert(0, "/ref/encoding")
from encoding import encode_messages  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file("/ref/tokenizer.json")
rows = [json.loads(l) for l in open("/fix/parity-A.jsonl")]
out = []
for r in rows:
    msgs = [{"role": "user", "content": r["prompt"]}]
    res = {}
    for mode in ("chat", "thinking"):
        kw = {"thinking_mode": mode}
        if mode == "thinking":
            kw["reasoning_effort"] = 75
        prompt = encode_messages(msgs, **kw)
        if isinstance(prompt, tuple):
            prompt = prompt[0]
        ids = tok.encode(prompt, add_special_tokens=False).ids
        res[mode] = {"n": len(ids), "head": prompt[:80], "tail": prompt[-60:], "ids_head": ids[:6]}
    want = r["usage"]["prompt_tokens"]
    match = {m: v["n"] == want for m, v in res.items()}
    out.append({"tag": r["tag"], "pg61_prompt_tokens": want, "rendered": res, "match": match})
    print(r["tag"], "pg61", want, {m: v["n"] for m, v in res.items()}, match, flush=True)
json.dump(out, open("/out/fixture-render.json", "w"), indent=1, ensure_ascii=False)
