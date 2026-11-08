#!/usr/bin/env python3
"""Build hot-expert maps with N hot experts per layer from the decode counts the seed map was fitted from.

The seed map docs/maps/hot-v26ar-seed-295.json ({"<layer>": [295 sorted ids]}) is m3's rowmap-mixed-ar-v1 hot sets,
fitted by research/dsv41-prefetch-20261023/ar/fit_mixed_rowmap.py: top-295 per layer by cumulative decode count from
counts-post-20261024-020438.npz (`cnt` int64 [40, 384], AR1+AR2), ties (equal counts) -> the existing static-map
membership, then expert id. This tool applies the same ranking for any N:

    rank key per layer:  (-count, not in tie-set, expert id)          (ascending)
    hot_N[layer]       = base[layer]  U  the best (N - |base[layer]|) experts outside base by that key

so hot_N is ALWAYS a superset of the base map (residency and replay evidence are built on the base map: it is never
shrunk). When the base map is itself the top-|base| of the same ranking (true for the 295 seed on its own counts),
hot_N is exactly the top-N of the ranking -- the first |base| ranks are the base, the next ranks are the extras.

Self-test (always run first, unless --skip-self-test): fit the top-|base| from the counts with NO base information in
the ranking beyond the tie-set and require it to equal the base map on every layer. On the box this proves the counts
file is the one the seed was fitted from. Exit 2 on mismatch, before anything is written.

The npz layout is inspected defensively: every array is printed (name, shape, dtype); the per-(layer, expert) count
array is the ONE integer array of shape [n_layers, n_experts] (or, with --snapshot, the one [S, n_layers, n_experts]);
zero or several candidates -> exit 2 asking for --key. Floats (ewma) and [L, 295]/[L, 89] hot/cold lists never qualify.

Usage (box container; numpy only):
  python3 bench/hot_map_from_counts.py --counts /ar/run/counts-post-20261024-020438.npz \\
      --base-map docs/maps/hot-v26ar-seed-295.json --n 301 304 307 --out-dir docs/maps
  python3 bench/hot_map_from_counts.py --counts ... --base-map ... --self-test-only
Output format = the base map's: json.dumps({"0": [sorted ids], ...}), no indent, no trailing newline (so the
N = |base| rebuild is byte-identical to the checked-in file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

N_EXPERTS = 384


class MapToolError(Exception):
    """Fatal, user-facing (exit 2)."""


# ------------------------------------------------------------------------------------------------ npz inspection
def describe_arrays(arrays: Dict[str, np.ndarray]) -> List[Tuple[str, Tuple[int, ...], str]]:
    return [(k, tuple(int(s) for s in v.shape), str(v.dtype)) for k, v in arrays.items()]


def pick_counts(arrays: Dict[str, np.ndarray], n_layers: int, n_experts: int = N_EXPERTS, key: Optional[str] = None,
                snapshot: Optional[int] = None, first_layers: Optional[int] = None) -> Tuple[str, np.ndarray]:
    """The per-(layer, expert) decode-count array as int64 [n_layers, n_experts], plus the key it came from.
    Ambiguity or absence is an error, never a guess."""
    if key is not None:
        if key not in arrays:
            raise MapToolError(f"--key {key!r} not in the npz (arrays: {sorted(arrays)})")
        names = [key]
    else:
        names = []
        for k, v in arrays.items():
            if v.dtype.kind not in "iu":
                continue
            if v.ndim == 2 and v.shape[1] == n_experts and v.shape[0] >= n_layers:
                names.append(k)
            elif v.ndim == 3 and v.shape[2] == n_experts and v.shape[1] >= n_layers:
                names.append(k)
        if len(names) != 1:
            raise MapToolError(
                f"cannot pick the per-(layer, expert) count array: {len(names)} candidates {names} (integer, "
                f"[>= {n_layers}, {n_experts}] or [S, >= {n_layers}, {n_experts}]); arrays: "
                f"{describe_arrays(arrays)}; pass --key (and --snapshot for a [S, L, E] stack)")
    k = names[0]
    a = arrays[k]
    if a.dtype.kind not in "iu":
        raise MapToolError(f"array {k!r} is {a.dtype}, counts must be an integer dtype")
    if a.ndim == 3:
        if snapshot is None:
            raise MapToolError(f"array {k!r} has shape {tuple(a.shape)} (a stack of snapshots): pass --snapshot "
                               f"<index> (cumulative counters: -1 = the last)")
        try:
            a = a[snapshot]
        except IndexError:
            raise MapToolError(f"--snapshot {snapshot} outside [0, {arrays[k].shape[0]})")
    elif snapshot is not None:
        raise MapToolError(f"--snapshot given but {k!r} is 2-D {tuple(a.shape)}")
    if a.ndim != 2 or a.shape[1] != n_experts:
        raise MapToolError(f"array {k!r} has shape {tuple(a.shape)}, want [n_layers, {n_experts}]")
    if a.shape[0] != n_layers:
        if a.shape[0] > n_layers and first_layers == n_layers:
            a = a[:n_layers]   # e.g. pinhot dumps: 40 main MoE modules + 3 DSpark nextn rows
        else:
            raise MapToolError(f"array {k!r} has {a.shape[0]} rows, the base map has {n_layers} layers "
                               f"(more rows: pass --first-layers {n_layers} to take the first {n_layers})")
    a = a.astype(np.int64)
    if (a < 0).any():
        raise MapToolError(f"array {k!r} has negative counts")
    return k, a


# ------------------------------------------------------------------------------------------------ the maps
def load_map(path: str) -> Dict[int, List[int]]:
    """{layer: sorted ids} from `{"<l>": [...]}`, `{"hot": {...}}` or m3 rowmap-v0 `{"layers": {l: {hot, cold}}}`."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("hot"), dict):
        raw = raw["hot"]
    if isinstance(raw, dict) and isinstance(raw.get("layers"), dict):
        raw = {k: v["hot"] for k, v in raw["layers"].items()}
    m = {int(k): sorted(int(e) for e in v) for k, v in raw.items()}
    return validate_map(m, path)


def validate_map(m: Dict[int, List[int]], what: str = "map", n_experts: int = N_EXPERTS) -> Dict[int, List[int]]:
    if sorted(m) != list(range(len(m))) or not m:
        raise MapToolError(f"{what}: layers must be 0..L-1 contiguous, got {sorted(m)[:5]}..")
    for l, ids in m.items():
        if len(set(ids)) != len(ids):
            raise MapToolError(f"{what}: duplicate ids in layer {l}")
        if ids and not (0 <= min(ids) and max(ids) < n_experts):
            raise MapToolError(f"{what}: layer {l} ids outside [0, {n_experts})")
    return m


def rank_experts(counts_row: np.ndarray, tie_set: frozenset = frozenset()) -> List[int]:
    """All experts best-first: count descending, then tie-set members first, then lower id. (The original fit's key.)"""
    return sorted(range(len(counts_row)), key=lambda e: (-int(counts_row[e]), e not in tie_set, e))


def fit_top_n(counts: np.ndarray, n: int, tie: Optional[Mapping[int, Sequence[int]]] = None) -> Dict[int, List[int]]:
    """Pure fit: the top-n of every layer's ranking, sorted ids. No base map involved."""
    out = {}
    for l in range(counts.shape[0]):
        ts = frozenset(tie[l]) if tie and l in tie else frozenset()
        out[l] = sorted(rank_experts(counts[l], ts)[:n])
    return out


def extend_map(base: Dict[int, List[int]], counts: np.ndarray, n: int,
               tie: Optional[Mapping[int, Sequence[int]]] = None) -> Dict[int, List[int]]:
    """base U the best (n - |base[l]|) experts outside base by the ranking. Superset of base by construction."""
    out = {}
    for l, b in base.items():
        if n < len(b):
            raise MapToolError(f"N={n} < |base[{l}]|={len(b)}: this tool never drops a base hot expert")
        if n > counts.shape[1]:
            raise MapToolError(f"N={n} > {counts.shape[1]} experts")
        bs = frozenset(b)
        ts = frozenset(tie[l]) if tie and l in tie else bs
        extra = [e for e in rank_experts(counts[l], ts) if e not in bs][: n - len(b)]
        ids = sorted(list(b) + extra)
        assert len(ids) == n and len(set(ids)) == n and bs <= set(ids)
        out[l] = ids
    return out


def self_test(base: Dict[int, List[int]], counts: np.ndarray,
              tie: Optional[Mapping[int, Sequence[int]]] = None) -> List[int]:
    """Layers where the pure top-|base| fit differs from the base map ([] = reproduced exactly). The ranking's
    tie-set is `tie` (the static-v1 map) if given, else the base map's own membership (what 'ties -> existing
    membership' means when no static map is at hand)."""
    n = len(next(iter(base.values())))
    if any(len(v) != n for v in base.values()):
        raise MapToolError("base map is not uniform; the self-test needs one N")
    fit = fit_top_n(counts, n, tie if tie is not None else base)
    return [l for l in sorted(base) if fit[l] != sorted(base[l])]


def to_json_text(m: Dict[int, List[int]]) -> str:
    return json.dumps({str(l): [int(e) for e in sorted(m[l])] for l in sorted(m)})


def map_stats(m: Dict[int, List[int]], counts: np.ndarray) -> dict:
    tot = int(counts.sum())
    hot = sum(int(counts[l, m[l]].sum()) for l in m)
    n = len(next(iter(m.values())))
    tie_layers = 0
    for l in m:
        c = np.sort(counts[l])[::-1]
        if n < len(c) and c[n - 1] == c[n]:
            tie_layers += 1
    return {"hot_per_layer": n, "decode_selections": tot, "in_sample_cold_share": round(1 - hot / tot, 6) if tot else None,
            "boundary_tie_layers": tie_layers}


# ------------------------------------------------------------------------------------------------ cli
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--counts", required=True, help="counts-post-*.npz (the file the seed map was fitted from)")
    ap.add_argument("--base-map", required=True, help="docs/maps/hot-v26ar-seed-295.json: every output is a superset")
    ap.add_argument("--n", type=int, nargs="*", default=[], help="hot experts per layer to emit (>= |base|), e.g. 301 304 307")
    ap.add_argument("--out-dir", default="docs/maps")
    ap.add_argument("--prefix", default="hot-v26ar-seed", help="output file: <out-dir>/<prefix>-<N>.json")
    ap.add_argument("--key", default=None, help="npz array holding the counts (default: the only integer [L, 384])")
    ap.add_argument("--snapshot", type=int, default=None, help="index into a [S, L, E] stack of cumulative snapshots")
    ap.add_argument("--first-layers", type=int, default=None, help="take the first N rows of a taller array (43 -> 40)")
    ap.add_argument("--tie-map", default=None, help="optional static map for tie-breaking (rowmap-static-v1.json); "
                                                       "default: the base map's membership")
    ap.add_argument("--n-experts", type=int, default=N_EXPERTS)
    ap.add_argument("--self-test-only", action="store_true", help="inspect + self-test, write nothing")
    ap.add_argument("--skip-self-test", action="store_true", help="exploratory counts that are NOT the seed's source")
    ap.add_argument("--force", action="store_true", help="overwrite an output file whose content differs")
    ap.add_argument("--report", default=None, help="write the summary JSON here too")
    a = ap.parse_args(argv)
    try:
        return _run(a)
    except MapToolError as e:
        print(f"[hot-map] FAIL: {e}", file=sys.stderr)
        return 2


def _run(a) -> int:
    base = load_map(a.base_map)
    L = len(base)
    sizes = {len(v) for v in base.values()}
    if len(sizes) != 1:
        raise MapToolError(f"base map is not uniform: per-layer sizes {sorted(sizes)}")
    n_base = sizes.pop()
    tie = load_map(a.tie_map) if a.tie_map else None
    if not os.path.isfile(a.counts):
        raise MapToolError(f"counts file {a.counts} not found")
    with np.load(a.counts, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    print(f"[hot-map] {a.counts}: {len(arrays)} arrays")
    for name, shape, dt in describe_arrays(arrays):
        print(f"[hot-map]   {name:<12} shape={shape} dtype={dt}")
    key, counts = pick_counts(arrays, L, a.n_experts, a.key, a.snapshot, a.first_layers)
    print(f"[hot-map] counts array: {key!r} -> int64 {tuple(counts.shape)}, total {int(counts.sum())} decode selections")
    print(f"[hot-map] base map {a.base_map}: {L} layers x {n_base} hot (sha256 {_sha256(open(a.base_map, 'rb').read())[:16]})")

    bad = self_test(base, counts, tie)
    if bad and not a.skip_self_test:
        raise MapToolError(f"SELF-TEST FAILED: the top-{n_base} fit of these counts differs from the base map in "
                           f"layers {bad} -- not the counts the seed was fitted from (or not the same tie rule). "
                           f"Nothing written. (--skip-self-test to proceed anyway; outputs stay supersets of the base.)")
    print(f"[hot-map] self-test {'SKIPPED (mismatch: ' + str(bad) + ')' if bad else 'PASS'}: top-{n_base} fit "
          f"{'!=' if bad else '=='} base map on {L - len(bad)}/{L} layers")
    report = {"counts": a.counts, "counts_key": key, "base_map": a.base_map, "base_hot": n_base, "layers": L,
              "self_test": "skipped-mismatch" if bad else "pass", "base": map_stats(base, counts), "outputs": {}}
    if a.self_test_only:
        print(json.dumps(report, indent=1))
        return 0
    if not a.n:
        raise MapToolError("no --n given (use --self-test-only to just verify)")
    for n in a.n:
        m = extend_map(base, counts, n, tie)
        for l in range(L):
            assert set(base[l]) <= set(m[l]) and len(m[l]) == n and m[l] == sorted(m[l])
        pure = fit_top_n(counts, n, tie if tie is not None else base)
        text = to_json_text(m)
        path = os.path.join(a.out_dir, f"{a.prefix}-{n}.json")
        os.makedirs(a.out_dir, exist_ok=True)
        if os.path.exists(path):
            old = open(path).read()
            if old != text and not a.force:
                raise MapToolError(f"{path} exists with different content (--force to overwrite)")
        with open(path, "w") as f:
            f.write(text)
        st = map_stats(m, counts)
        st.update(path=path, sha256=_sha256(text.encode()), extra_per_layer=n - n_base,
                  equals_pure_top_n=all(pure[l] == m[l] for l in m))
        report["outputs"][str(n)] = st
        print(f"[hot-map] N={n}: {path}  +{n - n_base}/layer  in-sample cold share {st['in_sample_cold_share']}  "
              f"boundary-tie layers {st['boundary_tie_layers']}  superset OK  pure-top-N {st['equals_pure_top_n']}  "
              f"sha256 {st['sha256'][:16]}")
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
