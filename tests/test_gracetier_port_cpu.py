"""Mac CPU contracts of P1 Build 2, the `gracetier` port into Garner's recipe (docs/HANDOFF-P1-BUILD2.md deliverable 6):
`.venv/bin/python -m pytest -q tests/test_gracetier_port_cpu.py`.

  * patches: 0001-boundary-gracetier.patch and 0002-components-gracetier.patch apply to a copy of
    reference/garner-8043a681/fastpath with `patch --dry-run` (and strictly, `git apply --check`) with the expected hunk
    counts; the patched boundary carries gracetier at the four sites of design section 2.2 and still parses;
  * serve-gracetier.patch applies on top of the Pg61 lane's serve.sh (the pin + launcher/patches/01-policy-hooks) and
    leaves every line outside its hunks byte-identical (line counts and a line diff);
  * the all-resident row_map (port/garner/make_rowmap_allres.py) is, per layer, a partition of 0..383 with every entry
    >= 0, hot ids -> 0..H-1, cold ids -> H..383; the component's loader accepts it and refuses a corrupted one;
  * the host-RAM budget function's arithmetic and the --kv-offloading-size lookup;
  * the component imports without vllm, registers its four hook targets, and its per-call glue (stage_symm, the forward
    wrap's group mapping) does what the boundary patch relies on;
  * the card runner's launcher / policy clone edits apply to the reference launch-pg61.sh and pg61-policy.env.
"""
from __future__ import annotations

import difflib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types

import pytest
import torch

from runtime import gracetier_pool as GP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "reference", "garner-8043a681")
PORT = os.path.join(REPO, "port", "garner")
P0001 = os.path.join(PORT, "patches", "0001-boundary-gracetier.patch")
P0002 = os.path.join(PORT, "patches", "0002-components-gracetier.patch")
PSERVE = os.path.join(PORT, "serve-gracetier.patch")
P01 = os.path.join(REF, "launcher", "patches", "01-policy-hooks-serve.patch")
H255 = os.path.join(REF, "fastpath", "hook", "rowmap-mix-v1-h255.json")
HOT295 = os.path.join(REPO, "docs", "maps", "hot-v26ar-seed-295.json")
CARD = os.path.join(REPO, "bench", "box", "card-gt1.sh")
E = 384

for p in (PORT, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
import make_rowmap_allres as MRA  # noqa: E402


# ================================================================================================ helpers
def hunks(path: str) -> dict:
    """{file: number of @@ hunks} parsed from a unified / git diff."""
    out, cur = {}, None
    for line in open(path):
        if line.startswith("+++ "):
            cur = line[4:].strip()
            cur = cur[2:] if cur.startswith("b/") else cur
            out[cur] = 0
        elif line.startswith("@@ ") and cur is not None:
            out[cur] += 1
    return out


def plus_minus(path: str) -> tuple:
    plus, minus = [], []
    for line in open(path):
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            plus.append(line[1:].rstrip("\n"))
        elif line.startswith("-"):
            minus.append(line[1:].rstrip("\n"))
    return plus, minus


def run_patch(tree: str, patch: str, dry: bool) -> subprocess.CompletedProcess:
    args = ["patch", "-p1", "-d", tree, "--forward"] + (["--dry-run"] if dry else [])
    with open(patch) as f:
        return subprocess.run(args, stdin=f, capture_output=True, text=True)


def tree_copy(tmp_path, rels) -> str:
    root = str(tmp_path / "tree")
    for rel in rels:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(REF, rel), dst)
    return root


def fastpath_copy(tmp_path) -> str:
    """A copy of reference/garner-8043a681/fastpath (python + hook files only; the binaries are not vendored)."""
    root = str(tmp_path / "tree")
    shutil.copytree(os.path.join(REF, "fastpath"), os.path.join(root, "fastpath"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    return root


# ================================================================================================ patches 0001 / 0002
def test_patch_files_are_git_diffs_against_the_pin_paths():
    for p, rel in ((P0001, "fastpath/fastpath_boundary.py"), (P0002, "fastpath/components.py")):
        txt = open(p).read()
        assert txt.startswith(f"diff --git a/{rel} b/{rel}\n"), p
        assert f"--- a/{rel}\n+++ b/{rel}\n" in txt
    assert hunks(P0001) == {"fastpath/fastpath_boundary.py": 5}
    assert hunks(P0002) == {"fastpath/components.py": 2}


def test_boundary_and_components_patches_apply_dry_run_to_the_reference_copy(tmp_path):
    root = fastpath_copy(tmp_path)
    before = {rel: open(os.path.join(root, rel)).read()
              for rel in ("fastpath/fastpath_boundary.py", "fastpath/components.py")}
    for p, n in ((P0001, 5), (P0002, 2)):
        r = run_patch(root, p, dry=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert sum(hunks(p).values()) == n
        assert "FAILED" not in r.stdout + r.stderr
        assert not re.search(r"fuzz|offset", r.stdout + r.stderr, re.I)
    for rel, txt in before.items():   # a dry run changes nothing
        assert open(os.path.join(root, rel)).read() == txt
    # strict (no offset, no fuzz): git apply --check
    for p in (P0001, P0002):
        r = subprocess.run(["git", "apply", "--check", p], cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_patched_boundary_has_gracetier_at_the_four_sites(tmp_path):
    root = fastpath_copy(tmp_path)
    for p in (P0001, P0002):
        r = run_patch(root, p, dry=False)
        assert r.returncode == 0, r.stdout + r.stderr
    src = open(os.path.join(root, "fastpath", "fastpath_boundary.py")).read()
    compile(src, "fastpath_boundary.py", "exec")
    # site 1: _moe_eligible's v2/v3 requirement
    assert "def _moe_eligible(moe, hidden_states, input_ids, g, v3: bool, gracetier: bool = False) -> bool:" in src
    assert "min_t = ROUTER_MIN_T if v3 or gracetier or not _peer3_installed(exp) else 5" in src
    assert 'return ((v3 or gracetier or (g is not None and g.get("PEER2")' in src
    # site 2: the publish selection (no peer v2, no peer v3 for a gracetier layer)
    assert "want_v3 = ROUTER_ON and PEER3_ON and gt is None and _peer3_installed(self.experts)" in src
    assert "if p3 is None and gt is None:\n            peer = g[\"_peer_tier2\"]" in src
    # site 3: stage right after the router, before MegaMoE
    fwd = src[src.index("def _make_moe_forward"):]
    i_router = fwd.index("_, _, _, has, pos, rows = router(")
    i_stage = fwd.index("gt.stage_symm(exp, symm, T)")
    i_mega = fwd.index("_mega_moe_call(dg, y, exp._transformed_l1_weights")
    assert i_router < i_stage < i_mega
    # site 4: no finish for gracetier
    assert "if gt is not None:\n            pass  # gracetier" in fwd and "elif p3 is None:\n            pf2.finish2" in fwd
    comp = open(os.path.join(root, "fastpath", "components.py")).read()
    assert "import fastpath_gracetier" in comp and "    fastpath_gracetier,\n" in comp
    assert "import fastpath_peer3" in comp and "import fastpath_coldskip" in comp   # the control stays bootable


def test_patched_boundary_helper_routes_to_the_component(tmp_path):
    """The patched module's _gracetier() returns fastpath_gracetier exactly for a gracetier layer (imports on the Mac:
    torch is lazy in fastpath_boundary, the .so is loaded only by lib())."""
    root = fastpath_copy(tmp_path)
    assert run_patch(root, P0001, dry=False).returncode == 0
    fp = os.path.join(REF, "fastpath")
    sys.path.insert(0, fp)
    try:
        spec = importlib.util.spec_from_file_location("gt_patched_boundary",
                                                      os.path.join(root, "fastpath", "fastpath_boundary.py"))
        B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(B)
    finally:
        sys.path.remove(fp)
    G = _component()
    st = G.GtState(3, list(range(255)), list(range(255, 384)), MRA.allres_row(list(range(255)), list(range(255, 384))))
    gt_layer = types.SimpleNamespace(_mp=st)
    other = types.SimpleNamespace(_mp=None)
    assert B._gracetier(gt_layer) is G
    assert B._gracetier(other) is None
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)


# ================================================================================================ serve.sh
def _pg61_serve(tmp_path) -> str:
    root = tree_copy(tmp_path, ["serve/serve.sh"])
    r = run_patch(root, P01, dry=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return root


def test_serve_patch_applies_on_the_pg61_lane_serve_sh(tmp_path):
    assert hunks(PSERVE) == {"serve/serve.sh": 4}
    root = _pg61_serve(tmp_path)
    r = run_patch(root, PSERVE, dry=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(["git", "apply", "--check", PSERVE], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_serve_patch_leaves_everything_outside_its_hunks_byte_identical(tmp_path):
    root = _pg61_serve(tmp_path)
    path = os.path.join(root, "serve", "serve.sh")
    before = open(path).read().splitlines()
    assert run_patch(root, PSERVE, dry=False).returncode == 0
    after = open(path).read().splitlines()
    plus, minus = plus_minus(PSERVE)
    assert (len(plus), len(minus)) == (5, 3)
    assert len(after) == len(before) + len(plus) - len(minus)
    added, removed = [], []
    for line in difflib.ndiff(before, after):
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])
    assert added == plus and removed == minus
    # every unchanged line is byte-identical and in order
    keep_b = [l for l in before if l not in minus]
    keep_a = [l for l in after if l not in plus]
    assert keep_b == keep_a
    txt = "\n".join(after)
    # the four skips, each conditional on gracetier in the component list
    assert 'gracetier_on() { case ",${DS41_FASTPATH_COMPONENTS:-}," in *,gracetier,*) return 0;; esac; return 1; }' in txt
    assert '[ -d "$DS41_FASTPATH_DIR/peer3" ] || gracetier_on || die "missing peer3 tree' in txt
    assert '[ -n "$RTX_UUID" ] || gracetier_on || die "found only the GB300' in txt
    assert 'if gracetier_on; then log "gracetier: no RTX sidecar (cold experts in Grace)"; else start_sidecar; fi' in txt
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_serve_gracetier_on_matches_only_the_whole_component_name(tmp_path):
    fn = 'gracetier_on() { case ",${DS41_FASTPATH_COMPONENTS:-}," in *,gracetier,*) return 0;; esac; return 1; }'
    for comps, want in (("boundary,gracetier,megamoe", 0), ("gracetier", 0), ("coldskip,peer3,megamoe", 1),
                        ("gracetier_x,megamoe", 1), ("", 1)):
        r = subprocess.run(["bash", "-c", f'{fn}\nDS41_FASTPATH_COMPONENTS="{comps}"; gracetier_on'])
        assert r.returncode == want, comps


# ================================================================================================ row_map
@pytest.fixture(scope="module")
def allres_doc():
    return MRA.make_doc(H255)


def test_allres_h255_is_a_partition_with_every_entry_ge_0(allres_doc):
    v = MRA.validate_allres(allres_doc)
    assert v["ok"] and v["layers"] == 40 and v["n_hot"] == [255], v["bad"][:3]
    split = GP.load_rowmap(H255)
    for k, lay in allres_doc["layers"].items():
        l = int(k)
        hot, cold, row = lay["hot"], lay["cold"], lay["row"]
        assert hot == split[l]["hot"] and cold == split[l]["cold"]       # Garner's own order: same hot slots
        assert len(row) == E and min(row) >= 0 and sorted(row) == list(range(E))
        assert sorted(hot + cold) == list(range(E)) and not set(hot) & set(cold)
        assert all(row[e] == s for s, e in enumerate(hot))               # hot id -> its slot in 0..H-1
        assert all(0 <= row[e] < 255 for e in hot) and all(255 <= row[e] < E for e in cold)
        assert row == GP.row_map(hot, cold, E)                           # == the mini-test's row_map (M2/M4)
    assert allres_doc["meta"]["n_hot"] == 255 and allres_doc["meta"]["source_sha256"] == MRA.sha256_file(H255)


def test_allres_hot_map_option_and_cli(tmp_path):
    doc = MRA.make_doc(H255, HOT295)
    v = MRA.validate_allres(doc)
    assert v["ok"] and v["n_hot"] == [295]
    hm = MRA.load_hot_map(HOT295)
    for k, lay in doc["layers"].items():
        assert lay["hot"] == hm[int(k)] and lay["cold"] == sorted(set(range(E)) - set(hm[int(k)]))
    out = str(tmp_path / "rowmap-allres.json")
    assert MRA.main(["--rowmap", H255, "--out", out]) == 0
    assert json.load(open(out)) == json.loads(json.dumps(MRA.make_doc(H255)))
    bad = str(tmp_path / "bad.json")
    raw = json.load(open(H255))
    raw["layers"]["7"]["cold"] = raw["layers"]["7"]["cold"][:-1]
    json.dump(raw, open(bad, "w"))
    assert MRA.main(["--rowmap", bad, "--out", str(tmp_path / "x.json")]) == 2
    assert not os.path.exists(tmp_path / "x.json")


def test_component_loads_the_allres_file_and_refuses_a_corrupted_one(tmp_path, allres_doc):
    G = _component()
    p = str(tmp_path / "ra.json")
    json.dump(allres_doc, open(p, "w"))
    a = G.load_allres(p)
    assert sorted(a) == list(range(40)) and all(min(v["row"]) >= 0 for v in a.values())
    assert G.expected_cold_tensors(a) == 40 * 129 * 6 == 30960          # = fastpath_coldskip.py:9 for h255
    for corrupt in ("negative", "swap"):
        d = json.loads(json.dumps(allres_doc))
        row = d["layers"]["5"]["row"]
        if corrupt == "negative":
            row[d["layers"]["5"]["cold"][0]] = -1                        # Garner's cold form must not reach the router
        else:
            h0, c0 = d["layers"]["5"]["hot"][0], d["layers"]["5"]["cold"][0]
            row[h0], row[c0] = row[c0], row[h0]                          # a permutation, but not hot -> 0..H-1
        json.dump(d, open(p, "w"))
        with pytest.raises(RuntimeError):
            G.load_allres(p)


def test_gtstate_forms_match_the_hook_and_a12_retire(allres_doc):
    G = _component()
    lay = allres_doc["layers"]["10"]
    st = G.GtState(10, lay["hot"], lay["cold"], lay["row"])
    assert (st.H, st.C) == (255, 129)
    assert st.n_hot + st.n_cold == E and st.n_hot == E                   # a12_retire: banks hold all 384 groups
    assert st.row == GP.hk_row_map(lay["hot"], lay["cold"], E)           # the hook's _weight_loader form
    assert GP.row_map_from_hk(st.row, st.H) == st.row_allres == lay["row"]
    assert sum(1 for r in st.row if r < 0) == 129 and min(st.row_allres) >= 0


def test_cold_name_filter_matches_coldskip(allres_doc):
    G = _component()
    cold = G.cold_sets({int(k): v for k, v in allres_doc["layers"].items()})
    e_c, e_h = allres_doc["layers"]["3"]["cold"][0], allres_doc["layers"]["3"]["hot"][0]
    assert G.is_cold_name(f"layers.3.ffn.experts.{e_c}.w1.weight", cold)
    assert G.is_cold_name(f"layers.3.ffn.experts.{e_c}.w2.scale", cold)
    assert not G.is_cold_name(f"layers.3.ffn.experts.{e_h}.w1.weight", cold)
    assert not G.is_cold_name("layers.3.ffn.shared_experts.w1.weight", cold)
    assert not G.is_cold_name(f"mtp.0.ffn.experts.{e_c}.w1.weight", cold)
    assert not G.is_cold_name("layers.1.engram.embed.weight", cold)


# ================================================================================================ host-RAM budget
GIB = 1024 ** 3


def test_host_ram_budget_arithmetic():
    G = _component()
    mem, pool, engram, kv, floor = 300 * GIB, 90 * GIB, 183 * GIB, 48 * GIB, 32 * GIB
    post = G.host_ram_budget(mem, pool, engram, kv, floor, resident=("cold_pool", "engram"))
    assert post["pending"] == ["kv_offload"] and post["pending_B"] == kv
    assert post["headroom_B"] == mem - kv == 252 * GIB and post["fits"] is True
    assert post["host_total_B"] == pool + engram + kv
    assert post["GiB"]["headroom"] == 252.0 and post["GiB"]["host_total"] == 321.0
    pre = G.host_ram_budget(mem, pool, engram, kv, floor, resident=("engram", "kv_offload"))
    assert pre["pending"] == ["cold_pool"] and pre["headroom_B"] == mem - pool and pre["fits"]
    cold = G.host_ram_budget(mem, pool, engram, kv, floor, resident=())
    assert cold["pending_B"] == pool + engram + kv and cold["headroom_B"] == mem - pool - engram - kv
    assert cold["headroom_B"] == -21 * GIB and cold["fits"] is False
    # the floor is inclusive
    edge = G.host_ram_budget(kv + floor, pool, engram, kv, floor)
    assert edge["headroom_B"] == floor and edge["fits"] is True
    assert G.host_ram_budget(kv + floor - 1, pool, engram, kv, floor)["fits"] is False
    with pytest.raises(ValueError):
        G.host_ram_budget(mem, pool, engram, kv, floor, resident=("swap",))
    with pytest.raises(ValueError):
        G.host_ram_budget(None, pool, engram, kv, floor)
    line = G.budget_line("post-load", post)
    assert line.startswith("HOST_RAM_BUDGET [post-load] engram 183.0 GiB (resident) + cold_pool 90.0 GiB (resident) + "
                           "kv_offload 48.0 GiB (pending) = 321.0 GiB; MemAvailable 300.0 GiB - pending 48.0 GiB = "
                           "headroom 252.0 GiB vs floor 32.0 GiB -> fits ")
    assert json.loads(re.search(r"(\{.*\})\s*$", line).group(1))["fits"] is True   # what card G1 parses


def test_kv_offload_lookup():
    G = _component()
    assert G.kv_offload_bytes(None, [], {"GRACETIER_KV_OFFLOAD_GB": "48"}) == (48 * GIB, "env GRACETIER_KV_OFFLOAD_GB")
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(kv_offloading_size=48.0))
    assert G.kv_offload_bytes(cfg, [], {})[0] == 48 * GIB
    argv = ["vllm", "serve", "--model", "/model", "--kv-offloading-size", "48", "--port", "30006"]
    assert G.kv_offload_bytes(None, argv, {}) == (48 * GIB, "argv --kv-offloading-size")
    assert G.kv_offload_bytes(None, ["vllm", "serve", "--kv-offloading-size=12.5"], {})[0] == int(12.5 * GIB)
    assert G.kv_offload_bytes(None, ["vllm", "serve", "--model", "/model"], {})[0] == 0
    with pytest.raises(RuntimeError):
        G.kv_offload_bytes(None, ["python3", "x.py"], {})


# ================================================================================================ the component
def _component():
    """Import port/garner/fastpath/fastpath_gracetier.py the way components.py does (fastpath_hook on the path)."""
    if "fastpath_gracetier" in sys.modules:
        return sys.modules["fastpath_gracetier"]
    for p in (os.path.join(REF, "fastpath"), os.path.join(PORT, "fastpath")):
        if p not in sys.path:
            sys.path.append(p)
    return importlib.import_module("fastpath_gracetier")


def test_component_imports_without_vllm_and_registers_its_hooks():
    G = _component()
    import fastpath_hook as FH

    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
    assert "gracetier" in FH._DEFERRED
    targets = sorted(t for t, _ in FH._COMPONENTS["gracetier"])
    assert targets == sorted([G.TARGET, G.WEIGHT_UTILS, G.DEFAULT_LOADER, G.RUNNER])
    assert G.TARGET == "vllm.models.deepseek_v4.nvidia.model"           # = peer3's and the hook's target
    for name in ("is_gracetier", "stage_symm", "stage"):                 # what the boundary patch calls
        assert callable(getattr(G, name))
    src = open(os.path.join(PORT, "fastpath", "fastpath_gracetier.py")).read()
    assert not re.search(r"^\s*(import vllm|from vllm)", src, re.M)
    assert not re.search(r"^(import torch|from torch)", src, re.M)      # torch only inside the hooks
    assert G.sf_strides(4608, 40) == (4608 * 40, 1, 4608)                # a12_retire.source_spec's layout


def test_stage_symm_passes_the_routers_groups_to_gt_stage(monkeypatch):
    G = _component()
    calls = []
    fake_gs = types.SimpleNamespace(gt_stage=lambda *a, **k: calls.append(a))
    monkeypatch.setattr(G._S, "rt", types.SimpleNamespace(GS=fake_gs))
    hot, cold = list(range(255)), list(range(255, 384))
    st = G.GtState(4, hot, cold, MRA.allres_row(hot, cold))
    with pytest.raises(RuntimeError):
        G.stage(st, torch.zeros(6, dtype=torch.int64))                   # before finalize
    st.src_tab, st.dtab, st.cnt, st.b1, st.b2 = torch.zeros(129, 2, dtype=torch.int64), "dtab", "cnt", 11, 22
    symm = types.SimpleNamespace(topk_idx=torch.arange(64 * 6, dtype=torch.int64).view(64, 6))
    G.stage_symm(types.SimpleNamespace(_mp=st), symm, 3)
    groups, H, src, dtab, cnt, b1, b2 = calls[-1]
    assert torch.equal(groups, torch.arange(18, dtype=torch.int64)) and groups.is_contiguous()
    assert (H, dtab, cnt, b1, b2) == (255, "dtab", "cnt", 11, 22) and src is st.src_tab


def test_forward_wrap_maps_ids_to_groups_and_stages_first(monkeypatch):
    G = _component()
    order = []
    hot = list(range(129, 384))
    cold = list(range(0, 129))
    st = G.GtState(9, hot, cold, MRA.allres_row(hot, cold))
    st.row_map = torch.tensor(st.row_allres, dtype=torch.int32)
    st.src_tab = torch.zeros(129, 2, dtype=torch.int64)
    monkeypatch.setattr(G, "stage", lambda s, g: order.append(("stage", g.clone())))

    def stock(self, hs, w, ids, *, activation_clamp, fast_math=True):
        order.append(("mega", ids.clone()))
        return hs

    monkeypatch.setattr(G._S, "hk", {"forward": stock})
    monkeypatch.setattr(G._S, "fwd_checked", True)
    mod = types.SimpleNamespace(_mp=st)
    ids = torch.tensor([[0, 128, 129, 383, -1, 5]], dtype=torch.int64)
    G._forward(mod, torch.zeros(1, 8), torch.ones(1, 6), ids, activation_clamp=None)
    assert [k for k, _ in order] == ["stage", "mega"]
    want = torch.tensor([[255 + 0, 255 + 128, 0, 254, -1, 255 + 5]], dtype=torch.int64)
    assert torch.equal(order[1][1], want)                                # cold -> H + c, hot -> slot, -1 kept
    assert torch.equal(order[0][1], want.view(-1))


def test_parity_ok_rule():
    G = _component()
    assert G.parity_ok({"l1_weight": True, "l2_weight": True, "l1_sf": True, "l2_sf": "not checked: x"})
    assert not G.parity_ok({"l1_weight": True, "l2_weight": False, "l1_sf": True, "l2_sf": True})
    assert not G.parity_ok({"l1_weight": True, "l2_weight": True, "l1_sf": False, "l2_sf": True})


# ================================================================================================ the card
def _heredoc(text: str, tag: str) -> str:
    m = re.search(rf"<<'{tag}'[^\n]*\n(.*?)\n{tag}\n", text, re.S)
    assert m, tag
    return m.group(1)


def test_card_guards_order_and_receipts():
    s = open(CARD).read()
    assert subprocess.run(["bash", "-n", CARD]).returncode == 0
    assert '[ -n "${GT1_WINDOW:-}" ] || refuse' in s
    i_down = s.index("bash launcher/launch-pg61.sh down")
    i_build = s.index("build > \"$S/gt1-build-$STAMP.log\"")
    i_up = s.index('bash "$GT/launch-gt1.sh" up')
    assert s.index("note \"CARD GT1 START") < i_down < i_build < i_up
    for g in ("G1", "G2", "G3", "G4", "G5"):
        assert re.search(rf"^# -+ {g}$", s, re.M), g
    assert "trap finish EXIT" in s and "restore-pg61-after-gt-mt-1108.sh restore-pg61-after-gt1.sh" in s
    assert 'S=$DEC/scratch' in s and "gt1-g1-$STAMP.json" in s and "gt1-summary-" in s
    assert "LANE-OWNER" in s and "MAINTENANCE" in s
    for p in ("patches/0001-boundary-gracetier.patch", "patches/0002-components-gracetier.patch",
              "serve-gracetier.patch", "make_rowmap_allres.py", "fastpath_gracetier.py", "bench/c32_smoke.py",
              "w0/w0_drive.py parity", "w0/w0_drive.py step", "bench/parity.py"):
        assert p in s, p
    assert "git pull" not in s.replace("does NOT git pull", "")


def test_card_launcher_and_policy_clones_apply_to_the_reference(tmp_path):
    s = open(CARD).read()
    launcher = str(tmp_path / "launch-gt1.sh")
    r = subprocess.run([sys.executable, "-", os.path.join(REF, "launcher", "launch-pg61.sh"), launcher],
                       input=_heredoc(s, "PY_LAUNCHER"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    L = open(launcher).read()
    assert subprocess.run(["bash", "-n", launcher]).returncode == 0
    assert "SERVED=${SERVED:-dsv41-flash-gracetier}" in L
    comps = re.search(r"^COMPONENTS=(\S+)$", L, re.M).group(1).split(",")
    pg = re.search(r"^COMPONENTS=(\S+)$", open(os.path.join(REF, "launcher", "launch-pg61.sh")).read(), re.M)
    pg = pg.group(1).split(",")
    assert "gracetier" in comps and "coldskip" not in comps and "peer3" not in comps
    assert [c for c in pg if c not in ("coldskip", "peer3")] == [c for c in comps if c != "gracetier"]
    assert comps.index("gracetier") == pg.index("coldskip")              # same place, before megamoe / a12_*
    assert "sidecar_watchdog_p3.py --server ds41 --sidecar ds41-sidecar --mode enforce" not in L
    assert 'docker update --restart no "$c"' not in L and "docker update --restart no ds41 >/dev/null" in L
    assert "cmd_down" in L and "cmd_verify" in L                          # the rest is Pg61's
    diff = [l for l in difflib.ndiff(open(os.path.join(REF, "launcher", "launch-pg61.sh")).read().splitlines(),
                                     L.splitlines()) if l[:2] in ("+ ", "- ")]
    assert len(diff) <= 20, diff

    pol = str(tmp_path / "policy.env")
    r = subprocess.run([sys.executable, "-", os.path.join(REF, "pg61-policy.env"), pol, "/home/milo/dsv41/gracetier/overlay-vllm"],
                       input=_heredoc(s, "PY_POLICY"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    a = open(os.path.join(REF, "pg61-policy.env")).read().splitlines()
    b = open(pol).read().splitlines()
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and [x.split("=")[0] for x, _ in changed] == [
        "DS41_SERVED", "DS41_FASTPATH_COMPONENTS_OVERRIDE", "DS41_OVERLAY_58132_DIR"]
    assert "DS41_OFFLOAD_GB=48" in b and "DS41_KV=fp8_ds_mla" in b and "DS41_MAXLEN=1048576" in b and "DS41_PORT=30006" in b


# ================================================================================================ c32 smoke (pure)
def test_c32_smoke_pure_parts():
    import c32_smoke as C

    ps = C.build_prompts(32)
    assert len(ps) == 32 and len(set(ps)) == 32
    b = C.request_body("m", "p", 128)
    assert b["max_tokens"] == b["min_tokens"] == 128 and b["ignore_eos"] is True and b["temperature"] == 0
    rows = [{"i": i, "ok": True, "completion_tokens": 128, "latency_s": 1.0 + i} for i in range(32)]
    s = C.summarize(rows, 32, 128, 10.0, True)
    assert s["pass"] and s["ok"] == 32 and s["completion_tokens_total"] == 4096 and s["aggregate_tok_s"] == 409.6
    rows[3] = {"i": 3, "ok": False, "error": "x"}
    assert not C.summarize(rows, 32, 128, 10.0, True)["pass"]
    rows[3] = {"i": 3, "ok": True, "completion_tokens": 127, "latency_s": 1.0}
    assert not C.summarize(rows, 32, 128, 10.0, True)["pass"]
    assert not C.summarize([dict(r, completion_tokens=128) for r in rows], 32, 128, 10.0, False)["pass"]
