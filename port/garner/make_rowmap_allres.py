#!/usr/bin/env python3
"""make_rowmap_allres.py -- the all-resident row_map for the `gracetier` component (card GT1;
docs/P1-GRACETIER-DESIGN.md section 2.2, docs/HANDOFF-P1-BUILD2.md deliverable 3). Stdlib only.

Garner's hook rowmap (`fastpath/hook/rowmap-mix-v1-h255.json`) splits every MoE layer's 384 experts into `hot` (HBM)
and `cold` (the RTX sidecar). His hook's ColdState.row is `hot slot` or `-(cold-local id + 1)` (HK/:61-75), and the w4
router publishes every negative entry as a cold route. gracetier keeps all 384 experts resident in one G = 384 MegaMoE
module (hot groups 0..H-1 in HBM, cold groups H..383 on the shared VMM scratch, staged from Grace each call), so the
router must see a map with NO negative entry:

    row[e] = hot slot of e        (its index in `hot`)          for e in hot      -> 0 .. H-1
    row[e] = H + cold-local id    (its index in `cold`)         for e in cold     -> H .. 383

The router accepting such a map (all entries >= 0, `has` = 0, topk_idx = row[ids], read per call) was measured in the
mini-test M2 (docs/gracetier-minitest-L10-gt-mt-1108.json, phases.M2.router); nothing else about the closed router is
assumed here.

Output JSON (one file, read by fastpath_gracetier from the hook dir, mounted at /w in the container):
    {"meta": {schema, E, form, n_hot (uniform H or null), n_hot_per_layer, source_rowmap, source_sha256,
              source_meta, hot_map, hot_map_sha256},
     "layers": {"<l>": {"hot": [...], "cold": [...], "row": [384 ints]}}}

Hot / cold order: with the rowmap alone (H = 255 first, the card's default) both lists are the rowmap's own, in its
order -- the hot slot r of expert e is the same r Garner's hook gives it. With --hot-map (e.g.
docs/maps/hot-v26ar-seed-295.json = {"<l>": [hot ids]}) the hot list is the map's (in the map's order) and the cold
list is the remaining ids ascending.

Usage:
    python3 port/garner/make_rowmap_allres.py --rowmap reference/garner-8043a681/fastpath/hook/rowmap-mix-v1-h255.json \
        --out <hook dir>/rowmap-allres.json [--hot-map docs/maps/hot-v26ar-seed-295.json]
Prints one summary line; exit 0, or 2 with the reason if any layer is not a partition of 0..E-1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

E_DEFAULT = 384
SCHEMA = "gracetier-rowmap-allres/1"
FORM = "row[e] = hot slot (index in hot) or H + cold-local id (index in cold); every entry >= 0"
DEFAULT_NAME = "rowmap-allres.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_split(path: str) -> Dict[int, Dict[str, List[int]]]:
    """{layer: {"hot": [...], "cold": [...]}} from a Garner rowmap ({"layers": ..., "meta": ...} or bare)."""
    with open(path) as f:
        raw = json.load(f)
    layers = raw["layers"] if "layers" in raw else raw
    return {int(k): {"hot": [int(x) for x in v["hot"]], "cold": [int(x) for x in v["cold"]]} for k, v in layers.items()}


def load_hot_map(path: str) -> Dict[int, List[int]]:
    """{layer: [hot ids]} from a hot map ({"<l>": [...]}, or {"layers": {"<l>": [...] | {"hot": [...]}}})."""
    with open(path) as f:
        raw = json.load(f)
    layers = raw["layers"] if isinstance(raw, dict) and "layers" in raw else raw
    out = {}
    for k, v in layers.items():
        ids = v["hot"] if isinstance(v, dict) else v
        out[int(k)] = [int(x) for x in ids]
    return out


def is_partition(hot: Sequence[int], cold: Sequence[int], E: int = E_DEFAULT) -> bool:
    return (len(hot) + len(cold) == E and len(set(hot)) == len(hot) and len(set(cold)) == len(cold)
            and set(hot) | set(cold) == set(range(E)) and not (set(hot) & set(cold)))


def allres_row(hot: Sequence[int], cold: Sequence[int], E: int = E_DEFAULT) -> List[int]:
    """row[e] = hot slot, or H + cold-local id (the gracetier form; same as runtime/gracetier_pool.row_map)."""
    if not is_partition(hot, cold, E):
        raise ValueError(f"hot ({len(hot)}) / cold ({len(cold)}) is not a partition of 0..{E - 1}")
    H = len(hot)
    row = [-1] * E
    for s, e in enumerate(hot):
        row[int(e)] = s
    for c, e in enumerate(cold):
        row[int(e)] = H + c
    return row


def build_allres(split: Dict[int, Dict[str, List[int]]], hot_map: Optional[Dict[int, List[int]]] = None,
                 E: int = E_DEFAULT) -> Dict[str, dict]:
    """The "layers" part: {"<l>": {"hot", "cold", "row"}}. hot_map replaces each layer's hot set (cold = the rest,
    ascending); its layers must equal the rowmap's."""
    if hot_map is not None and set(hot_map) != set(split):
        raise ValueError(f"hot map layers {sorted(hot_map)[:5]}.. != rowmap layers {sorted(split)[:5]}..")
    out = {}
    for l in sorted(split):
        if hot_map is None:
            hot, cold = list(split[l]["hot"]), list(split[l]["cold"])
        else:
            hot = list(hot_map[l])
            hs = set(hot)
            if len(hs) != len(hot) or not hs <= set(range(E)):
                raise ValueError(f"hot map layer {l}: duplicate or out-of-range ids")
            cold = [e for e in range(E) if e not in hs]
        out[str(l)] = {"hot": hot, "cold": cold, "row": allres_row(hot, cold, E)}
    return out


def validate_allres(doc: dict, E: int = E_DEFAULT) -> dict:
    """Check a whole all-resident document; returns per-property booleans + counts (raises nothing).

    partition: hot / cold partition 0..E-1 per layer; row_ok: row == allres_row(hot, cold); all_ge_0; permutation:
    every layer's row is a permutation of 0..E-1; hot_low: every hot expert maps into 0..H-1 and every cold expert
    into H..E-1."""
    layers = doc.get("layers") or {}
    res = {"layers": len(layers), "partition": True, "row_ok": True, "all_ge_0": True, "permutation": True,
           "hot_low": True, "n_hot": sorted({len(v.get("hot", [])) for v in layers.values()}), "bad": []}
    for k, v in layers.items():
        hot, cold, row = v.get("hot", []), v.get("cold", []), v.get("row", [])
        part = is_partition(hot, cold, E)
        ok_row = part and len(row) == E and list(row) == allres_row(hot, cold, E)
        ge0 = len(row) == E and min(row) >= 0
        perm = sorted(row) == list(range(E))
        H = len(hot)
        low = (len(row) == E and all(0 <= row[e] < H for e in hot) and all(H <= row[e] < E for e in cold))
        for name, val in (("partition", part), ("row_ok", ok_row), ("all_ge_0", ge0), ("permutation", perm),
                          ("hot_low", low)):
            if not val:
                res[name] = False
                res["bad"].append(f"layer {k}: {name}")
    res["ok"] = bool(layers) and all(res[n] for n in ("partition", "row_ok", "all_ge_0", "permutation", "hot_low"))
    return res


def make_doc(rowmap_path: str, hot_map_path: Optional[str] = None, E: int = E_DEFAULT) -> dict:
    split = load_split(rowmap_path)
    with open(rowmap_path) as f:
        raw = json.load(f)
    src_meta = dict(raw.get("meta") or {}) if isinstance(raw, dict) else {}
    hm = load_hot_map(hot_map_path) if hot_map_path else None
    layers = build_allres(split, hm, E)
    nh = {k: len(v["hot"]) for k, v in layers.items()}
    meta = {"schema": SCHEMA, "E": E, "form": FORM,
            "n_hot": (next(iter(set(nh.values()))) if len(set(nh.values())) == 1 else None),
            "n_hot_per_layer": nh,
            "source_rowmap": os.path.basename(rowmap_path), "source_sha256": sha256_file(rowmap_path),
            "source_meta": src_meta,
            "hot_map": os.path.basename(hot_map_path) if hot_map_path else None,
            "hot_map_sha256": sha256_file(hot_map_path) if hot_map_path else None}
    return {"meta": meta, "layers": layers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--rowmap", required=True, help="Garner hook rowmap (hot/cold per layer), e.g. rowmap-mix-v1-h255.json")
    ap.add_argument("--hot-map", default=None, help="optional {layer: [hot ids]} map (docs/maps/hot-v26ar-seed-*.json)")
    ap.add_argument("--out", required=True, help=f"output JSON (the card writes <hook dir>/{DEFAULT_NAME})")
    ap.add_argument("--experts", type=int, default=E_DEFAULT)
    a = ap.parse_args(argv)
    try:
        doc = make_doc(a.rowmap, a.hot_map, a.experts)
    except (ValueError, KeyError) as e:
        print(f"make_rowmap_allres: REFUSED: {e}", file=sys.stderr)
        return 2
    v = validate_allres(doc, a.experts)
    if not v["ok"]:
        print(f"make_rowmap_allres: REFUSED: {v['bad'][:5]}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, a.out)
    print(f"make_rowmap_allres: {v['layers']} layers, H={doc['meta']['n_hot']} (per-layer {v['n_hot']}), "
          f"source {doc['meta']['source_rowmap']} sha256 {doc['meta']['source_sha256'][:16]}, "
          f"hot_map {doc['meta']['hot_map']}, all entries >= 0, permutation per layer -> {a.out} "
          f"sha256 {sha256_file(a.out)[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
