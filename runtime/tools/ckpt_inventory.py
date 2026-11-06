#!/usr/bin/env python3
"""ckpt_inventory.py <model_dir> [--out inventory.json]
Stdlib-only inventory of a safetensors checkpoint: reads only each shard's header (8-byte length + JSON), never the data.
Groups tensors by (layer, role) and reports dtype/shape/bytes so the runtime's loader and memory plan are built from
the checkpoint, not from a paper. Read-only.
"""
import json, os, re, struct, sys, collections

d = sys.argv[1]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
cfg = json.load(open(os.path.join(d, "config.json")))
files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
DT = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1, "F4": 0.5, "U4": 0.5, "F6_E2M3": 0.75, "F6_E3M2": 0.75}
tensors = {}
for f in files:
    with open(os.path.join(d, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        b0, b1 = v["data_offsets"]; tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "bytes": b1 - b0, "file": f}

def role(name):
    m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", name)
    layer = int(m.group(1)) if m else None; rest = m.group(2) if m else name
    if "experts" in rest and "shared" not in rest: r = "routed_experts"
    elif "shared_experts" in rest: r = "shared_experts"
    elif "engram" in rest.lower(): r = "engram"
    elif "indexer" in rest: r = "indexer"
    elif "self_attn" in rest or "attn" in rest: r = "attention"
    elif "mlp" in rest: r = "dense_mlp"
    elif "gate" in rest and "experts" not in rest: r = "router"
    elif "mhc" in rest.lower() or "norm" in rest: r = "norm_mhc"
    elif layer is None and ("embed" in rest or "lm_head" in rest): r = "embed_head"
    elif layer is None and ("mtp" in rest.lower() or "draft" in rest.lower()): r = "mtp_draft"
    else: r = "other"
    return layer, r, rest

by = collections.defaultdict(lambda: {"n": 0, "bytes": 0, "dtypes": collections.Counter(), "examples": []})
per_layer = collections.defaultdict(lambda: collections.defaultdict(int))
for k, v in tensors.items():
    layer, r, rest = role(k)
    g = by[r]; g["n"] += 1; g["bytes"] += v["bytes"]; g["dtypes"][v["dtype"]] += 1
    if len(g["examples"]) < 6: g["examples"].append((k, v["dtype"], v["shape"], v["bytes"]))
    if layer is not None: per_layer[layer][r] += v["bytes"]
GiB = 1024 ** 3
total = sum(v["bytes"] for v in tensors.values())
summary = {"model_dir": d, "shards": len(files), "tensors": len(tensors), "total_GiB": round(total / GiB, 2),
           "config_keys": {k: cfg[k] for k in sorted(cfg) if not isinstance(cfg[k], (dict, list)) or k in ("quantization_config",)},
           "roles": {r: {"n": g["n"], "GiB": round(g["bytes"] / GiB, 3), "dtypes": dict(g["dtypes"]), "examples": g["examples"]} for r, g in sorted(by.items())},
           "per_layer_GiB": {str(l): {r: round(b / GiB, 3) for r, b in sorted(rs.items())} for l, rs in sorted(per_layer.items())}}
print(json.dumps({k: v for k, v in summary.items() if k not in ("per_layer_GiB","roles")}, indent=1, default=str)[:6000])
for r, g in summary["roles"].items():
    print(f"ROLE {r:16s} n={g['n']:5d} GiB={g['GiB']:8.3f} dtypes={g['dtypes']}")
    for e in g["examples"][:3]: print("     ", e)
print("layers with weights:", len(per_layer), "| per-layer roles of layer 3:", summary["per_layer_GiB"].get("3"))
print("layer 1 (engram?):", summary["per_layer_GiB"].get("1")); print("layer 14:", summary["per_layer_GiB"].get("14"))
if out: json.dump(summary, open(out, "w"), indent=1, default=str)
