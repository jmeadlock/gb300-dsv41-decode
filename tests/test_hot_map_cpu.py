"""Mac CPU tests of bench/hot_map_from_counts.py (docs/HOT-MAP.md): `.venv/bin/python -m pytest tests/test_hot_map_cpu.py -q`.

Synthetic counts only (the real npz lives on the box / in research/); the properties the box self-test relies on:
  * N = |base| reproduces the base map exactly when the base IS the top-N of the counts (incl. ties -> membership);
  * N > |base| is a superset of the base on every layer, even when the base is NOT the top-N of the counts;
  * sorted ids, 40 layers, no duplicates, ids in range; the file format is the base map's, byte for byte;
  * the npz inspection picks the one integer [L, 384] array and fails loudly on ambiguity;
  * the checked-in docs/maps/hot-v26ar-seed-<N>.json are supersets of the 295 seed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from bench import hot_map_from_counts as H

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, E, NB = 40, 384, 295


def _counts(seed=0, ties=True):
    rng = np.random.default_rng(seed)
    c = (rng.zipf(1.3, size=(L, E)).clip(max=10_000)).astype(np.int64)
    if ties:
        c[:, :] = (c // 3) * 3   # plenty of equal counts, including at the rank-295 boundary
    return c


def _base_from(counts, tie=None):
    return H.fit_top_n(counts, NB, tie)


def test_reproduces_base_at_n_equal_base():
    c = _counts()
    base = _base_from(c)                       # no tie info: id breaks ties
    assert H.self_test(base, c, tie=base) == []   # tie-set = base membership: same answer
    assert H.fit_top_n(c, NB) == base


def test_reproduction_with_static_tie_membership():
    c = _counts(1)
    rng = np.random.default_rng(5)
    static = {l: sorted(rng.choice(E, NB, replace=False).tolist()) for l in range(L)}
    base = _base_from(c, static)               # ties -> static membership, as the original fit
    assert H.self_test(base, c, tie=static) == []
    # the tie rule matters: with a boundary tie, ignoring the static set picks differently on some layer
    assert any(H.fit_top_n(c, NB)[l] != base[l] for l in range(L))
    assert H.self_test(base, c, tie=None) == []   # tie=None -> base membership as the tie-set, still reproduces


def test_self_test_flags_wrong_counts():
    c = _counts(2)
    base = _base_from(c)
    bad = _counts(3)
    assert H.self_test(base, bad) != []


@pytest.mark.parametrize("n", [295, 301, 304, 307])
def test_superset_sorted_forty_layers(n):
    c = _counts(4)
    base = _base_from(c)
    m = H.extend_map(base, c, n)
    assert sorted(m) == list(range(40))
    for l in range(L):
        assert m[l] == sorted(m[l]) and len(m[l]) == n == len(set(m[l]))
        assert set(base[l]) <= set(m[l])
        assert 0 <= min(m[l]) and max(m[l]) < E
    if n == NB:
        assert m == base
    # nested: 301 ⊂ 304 ⊂ 307
    m2 = H.extend_map(base, c, n + 3, None)
    assert all(set(m[l]) <= set(m2[l]) for l in range(L))


def test_superset_even_if_base_is_not_top_n():
    """A base that is NOT the top-295 of these counts (e.g. first-295): outputs still never drop a base expert."""
    c = _counts(5)
    base = {l: list(range(NB)) for l in range(L)}
    assert H.self_test(base, c) != []
    m = H.extend_map(base, c, 304)
    for l in range(L):
        assert set(base[l]) <= set(m[l]) and len(m[l]) == 304
        # the 9 extras are the 9 hottest outside the base
        outside = sorted(range(NB, E), key=lambda e: (-int(c[l, e]), e))[:9]
        assert sorted(set(m[l]) - set(base[l])) == sorted(outside)


def test_extras_are_the_best_outside_base_and_ties_by_id():
    c = np.zeros((L, E), dtype=np.int64)
    c[:, :NB] = 10                                  # base: counts 10
    c[:, NB:] = 3                                   # rest tie at 3 -> lowest ids first
    c[:, 380] = 7                                   # one clearly hotter outsider
    base = {l: list(range(NB)) for l in range(L)}
    m = H.extend_map(base, c, 298)
    assert all(m[l] == sorted(list(range(NB)) + [380, NB, NB + 1]) for l in range(L))


def test_refuses_to_drop_and_to_exceed():
    c = _counts(6)
    base = _base_from(c)
    with pytest.raises(H.MapToolError):
        H.extend_map(base, c, 290)
    with pytest.raises(H.MapToolError):
        H.extend_map(base, c, 385)


def test_json_text_matches_base_file_format():
    p = os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json")
    raw = open(p).read()
    assert H.to_json_text(H.load_map(p)) == raw      # byte-identical regeneration of the checked-in 295 map


# ------------------------------------------------------------------------------------------------ npz inspection
def test_pick_counts_the_one_integer_array():
    arrays = {"cnt": _counts(), "ewma": _counts().astype(float), "hot": np.zeros((L, 295), np.int32),
              "cold": np.zeros((L, 89), np.int32), "host_steps": np.array(5, np.int64)}
    k, a = H.pick_counts(arrays, L)
    assert k == "cnt" and a.shape == (L, E) and a.dtype == np.int64


def test_pick_counts_ambiguous_or_missing_fails_loudly():
    c = _counts()
    with pytest.raises(H.MapToolError, match="candidates"):
        H.pick_counts({"cnt_dec": c, "cnt_all": c}, L)
    with pytest.raises(H.MapToolError, match="candidates"):
        H.pick_counts({"ewma": c.astype(float)}, L)
    assert H.pick_counts({"cnt_dec": c, "cnt_all": c}, L, key="cnt_all")[0] == "cnt_all"
    with pytest.raises(H.MapToolError):
        H.pick_counts({"cnt": c}, L, key="nope")
    with pytest.raises(H.MapToolError, match="negative"):
        H.pick_counts({"cnt": -c}, L)


def test_pick_counts_rows_and_stacks():
    c = _counts()
    tall = np.concatenate([c, c[:3]])                # pinhot-style: 40 main + 3 nextn rows
    with pytest.raises(H.MapToolError, match="first-layers"):
        H.pick_counts({"cnt": tall}, L)
    assert H.pick_counts({"cnt": tall}, L, first_layers=L)[1].shape == (L, E)
    stack = np.stack([c, c * 2])
    with pytest.raises(H.MapToolError, match="snapshot"):
        H.pick_counts({"snap_cnt": stack}, L)
    assert np.array_equal(H.pick_counts({"snap_cnt": stack}, L, snapshot=-1)[1], c * 2)


# ------------------------------------------------------------------------------------------------ CLI end to end
def _write_inputs(tmp_path, c, base):
    npz = tmp_path / "counts.npz"
    np.savez(npz, cnt=c, ewma=c.astype(float), hot=np.zeros((L, NB), np.int32), host_steps=np.array(1))
    bm = tmp_path / "base.json"
    bm.write_text(H.to_json_text(base))
    return str(npz), str(bm)


def test_cli_writes_supersets_and_refuses_on_selftest_mismatch(tmp_path, capsys):
    c = _counts(7)
    base = _base_from(c)
    npz, bm = _write_inputs(tmp_path, c, base)
    out = tmp_path / "maps"
    assert H.main(["--counts", npz, "--base-map", bm, "--n", "295", "301", "304", "307", "--out-dir", str(out)]) == 0
    assert (out / "hot-v26ar-seed-295.json").read_text() == (tmp_path / "base.json").read_text()
    for n in (301, 304, 307):
        m = H.load_map(str(out / f"hot-v26ar-seed-{n}.json"))
        assert len(m) == 40 and all(len(v) == n and set(base[l]) <= set(v) and v == sorted(v) for l, v in m.items())
    # counts that are not the seed's source: self-test fails, exit 2, nothing written
    npz2, _ = _write_inputs(tmp_path, _counts(8), base)
    out2 = tmp_path / "maps2"
    assert H.main(["--counts", npz2, "--base-map", bm, "--n", "301", "--out-dir", str(out2)]) == 2
    assert not out2.exists()
    # --skip-self-test still gives a superset
    assert H.main(["--counts", npz2, "--base-map", bm, "--n", "301", "--out-dir", str(out2), "--skip-self-test"]) == 0
    m = H.load_map(str(out2 / "hot-v26ar-seed-301.json"))
    assert all(set(base[l]) <= set(m[l]) for l in range(L))
    # an existing output with different content is not clobbered
    (out2 / "hot-v26ar-seed-301.json").write_text("{}")
    assert H.main(["--counts", npz2, "--base-map", bm, "--n", "301", "--out-dir", str(out2), "--skip-self-test"]) == 2
    capsys.readouterr()


def test_cli_self_test_only_and_prints_arrays(tmp_path, capsys):
    c = _counts(9)
    npz, bm = _write_inputs(tmp_path, c, _base_from(c))
    assert H.main(["--counts", npz, "--base-map", bm, "--self-test-only"]) == 0
    out = capsys.readouterr().out
    assert "shape=(40, 384)" in out and "self-test PASS" in out and "'cnt'" in out


# ------------------------------------------------------------------------------------------------ the checked-in maps
@pytest.mark.parametrize("n", [301, 304, 307])
def test_checked_in_maps_are_supersets_of_295(n):
    d = os.path.join(REPO, "docs", "maps")
    p = os.path.join(d, f"hot-v26ar-seed-{n}.json")
    if not os.path.exists(p):
        pytest.skip(f"{p} not built")
    base, m = H.load_map(os.path.join(d, "hot-v26ar-seed-295.json")), H.load_map(p)
    assert len(m) == 40
    for l in range(40):
        assert len(m[l]) == n and m[l] == sorted(m[l]) and set(base[l]) <= set(m[l])
    assert open(p).read() == H.to_json_text(m)
