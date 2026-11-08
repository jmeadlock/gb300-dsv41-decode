"""LOOSE-ENDS-2 (docs/HANDOFF-LOOSE-ENDS-2-1107.md): CPU tests for bench/receipt.py (receipt stamping), its call sites in
bench/headline.py and bench/first_light.py, headline's total-GPU-ms / total-tokens line, bench/box/lease.sh (with a
python flock shim: macOS has no flock(1)), and docs/A3-DEFINITION.md's worked example. Stdlib + pytest only; every
expected value is recomputed here from the file/command it came from, never typed in (except where a test says so)."""
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench")
for p in (REPO, BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

import receipt as R  # noqa: E402

LEASE = os.path.join(BENCH, "box", "lease.sh")
A3DOC = os.path.join(REPO, "docs", "A3-DEFINITION.md")
PARITY_A = os.path.join(REPO, "w0", "results", "phase1", "parity-A.jsonl")
PARITY_A2 = os.path.join(REPO, "w0", "results", "phase1", "parity-A2.jsonl")
CAND_NIGHT = os.path.join(REPO, "w0", "results", "first-light-cand-night-20261107.jsonl")
need = lambda *ps: pytest.mark.skipif(not all(os.path.exists(p) for p in ps), reason="receipt inputs not in this checkout")  # noqa: E731


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    (d / "a.txt").write_text("one\n")
    (d / "b.bin").write_bytes(b"\x00\x01\x02")
    _git(d, "add", ".")
    _git(d, "commit", "-q", "-m", "c1")
    return d


# ------------------------------------------------------------------------------------------------ git
def test_git_state_clean_and_dirty(repo):
    s = R.git_state(str(repo))
    assert s["commit"] == _git(repo, "rev-parse", "HEAD").strip() and s["source"] == "git"
    assert s["dirty"] is False and s["dirty_sha256"] is None and s["dirty_files"] == [] and s["untracked"] == 0
    (repo / "a.txt").write_text("two\n")
    (repo / "b.bin").write_bytes(b"\x00\xff")
    (repo / "new.txt").write_text("untracked\n")
    s = R.git_state(str(repo))
    want = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--binary"], capture_output=True, check=True).stdout
    assert s["dirty"] is True and s["dirty_sha256"] == hashlib.sha256(want).hexdigest()
    assert sorted(s["dirty_files"]) == ["a.txt", "b.bin"] and s["untracked"] == 1
    # staged changes count too (diff HEAD), and the hash changes with the content
    _git(repo, "add", "a.txt")
    assert R.git_state(str(repo))["dirty_sha256"] == s["dirty_sha256"]
    (repo / "a.txt").write_text("three\n")
    assert R.git_state(str(repo))["dirty_sha256"] != s["dirty_sha256"]


def test_git_state_env_fallback_when_git_cannot_run(tmp_path):
    env = {"RECEIPT_GIT_COMMIT": "abc123", "RECEIPT_GIT_BRANCH": "main", "RECEIPT_GIT_DIRTY_SHA256": "dead",
           "RECEIPT_GIT_DIRTY_FILES": "x.py,y.py", "RECEIPT_GIT_UNTRACKED": "7"}
    s = R.git_state_env(env)
    assert s == {"commit": "abc123", "branch": "main", "dirty": True, "dirty_sha256": "dead",
                 "dirty_files": ["x.py", "y.py"], "untracked": 7, "source": "env"}
    with pytest.raises(RuntimeError):
        R.git_state_env({})
    rc = R.begin(repo=str(tmp_path), env=env)                  # tmp_path is not a repo
    assert rc["git_commit"] == "abc123" and rc["git_source"] == "env" and rc["git_dirty"] is True
    assert any(e.startswith("git:") for e in rc["receipt_errors"])


# ------------------------------------------------------------------------------------------------ begin / finish
def test_begin_contract_and_never_raises(tmp_path, repo):
    assert len(set(R.RECEIPT_FIELDS)) == len(R.RECEIPT_FIELDS)
    cfgdir = tmp_path / "ckpt"
    cfgdir.mkdir()
    (cfgdir / "config.json").write_text('{"a": 1}')
    fx = tmp_path / "fx.jsonl"
    fx.write_text('{"tag": "t"}\n')
    env = {"DSV41_FP32_IMPL": "fixed", "DSV41_GEMV_PIN_M1": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HOME": "/x", "PATH": "/y"}
    rc = R.begin(argv=["--a", "1"], impls={"moe": "dg-inplace"}, model_dir=str(cfgdir), fixture=str(fx),
                 residency=None, map_path="m.json", repo=str(repo), env=env)
    assert set(rc) == set(R.RECEIPT_FIELDS)
    assert rc["schema"] == R.SCHEMA and rc["argv"] == ["--a", "1"] and rc["impls"] == {"moe": "dg-inplace"}
    assert rc["env_switches"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "DSV41_FP32_IMPL": "fixed", "DSV41_GEMV_PIN_M1": "1"}
    assert rc["checkpoint_path"] == str(cfgdir) and rc["checkpoint_config_path"] == str(cfgdir / "config.json")
    assert rc["checkpoint_config_sha256"] == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert rc["fixture_sha256"] == hashlib.sha256(b'{"tag": "t"}\n').hexdigest() and rc["fixture_path"] == str(fx)
    assert rc["git_commit"] == _git(repo, "rev-parse", "HEAD").strip() and rc["git_dirty"] is False
    assert rc["residency_condition"] == "off (static map m.json)"
    assert rc["start_wall"] <= time.time() and rc["end_wall"] is None
    # a missing model dir / fixture / non-repo / no GPU tools: still a dict, the failures are listed, nothing raised
    rc2 = R.begin(model_dir="/nonexistent/ckpt", fixture="/nonexistent/fx.jsonl", repo="/nonexistent", env={})
    assert set(rc2) == set(R.RECEIPT_FIELDS) and rc2["checkpoint_config_sha256"] is None and rc2["fixture_sha256"] is None
    errs = " | ".join(rc2["receipt_errors"])
    assert "checkpoint_config:" in errs and "fixture:" in errs and "git:" in errs
    json.dumps(rc, default=str)


def test_smi_fields_clocks_and_finish(monkeypatch, repo):
    calls = []

    def fake_run(cmd, cwd=None, timeout=20.0):
        calls.append(list(cmd))
        if cmd[0] == "nvidia-smi":
            q = cmd[1]
            if q == "--query-gpu=driver_version,uuid,name":
                return b"580.82.07, GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55, NVIDIA GB300\n"
            if q == "--query-gpu=clocks.sm,clocks.mem":
                n = sum(1 for c in calls if c[0] == "nvidia-smi" and c[1] == q)
                return (b"1920, 3996\n" if n == 1 else b"1650, 3996\n")
        return subprocess.run(list(cmd), cwd=cwd, capture_output=True, check=True, timeout=timeout).stdout

    monkeypatch.setattr(R, "_run", fake_run)
    rc = R.begin(repo=str(repo), env={})
    assert (rc["driver_version"], rc["gpu_uuid"], rc["gpu_name"]) == ("580.82.07", "GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55", "NVIDIA GB300")
    assert (rc["clocks_sm_mhz_start"], rc["clocks_mem_mhz_start"]) == (1920.0, 3996.0)
    assert ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader,nounits"] in calls
    out = R.finish(rc)
    assert out is rc and rc["clocks_sm_mhz_end"] == 1650.0 and rc["clocks_mem_mhz_end"] == 3996.0
    assert rc["end_wall"] >= rc["start_wall"] and rc["wall_s"] >= 0 and rc["end_iso"]
    assert R.fields_present(rc) >= 25 and "receipt:" in R.summary_line(rc) and "1920.0->1650.0" in R.summary_line(rc)
    assert R.finish(None) is None and R.summary_line(None) == "receipt: none"


def test_finish_rereads_env_and_survives_no_gpu(monkeypatch):
    monkeypatch.setattr(R, "_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no nvidia-smi")))
    rc = R.begin(repo="/nonexistent", env={"DSV41_A": "1"})
    monkeypatch.setenv("DSV41_SET_LATER", "x")
    R.finish(rc)
    assert rc["env_switches"]["DSV41_SET_LATER"] == "x" and rc["env_switches"]["DSV41_A"] == "1"
    assert rc["clocks_sm_mhz_end"] is None and any(e.startswith("clocks_end:") for e in rc["receipt_errors"])


def test_pure_helpers():
    assert R.parse_smi("1920, 3996\n\n", 2) == [["1920", "3996"]]
    assert R.parse_smi("a, b, c\n", 2) == [["a", "b"]] and R.parse_smi("x\n", 2) == []
    assert R.env_switches({"DSV41_X": "1", "OTHER": "2", "CUBLAS_WORKSPACE_CONFIG": "c"}) == {"CUBLAS_WORKSPACE_CONFIG": "c", "DSV41_X": "1"}
    assert R.residency_condition(None, None) == "off (static map)"
    assert R.residency_condition({"stage_keep": 8, "drain": 64}, "m.json") == "on drain=64, stage_keep=8 (map m.json)"

    class Cfg:
        def __init__(self):
            self.stage_keep = 8
    assert R.residency_condition(Cfg()) == "on stage_keep=8"


# ------------------------------------------------------------------------------------------------ call sites
def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(BENCH, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_headline_totals_are_explicit_and_labeled():
    B = _load("headline")
    runs = [{"step_ms": [10.0] * 3 + [20.0, 30.0], "verify_ms": [6.0] * 3 + [12.0, 18.0],
             "draft_ms": [4.0] * 3 + [8.0, 12.0], "accepts": [0, 1, 2, 3, 4], "mgr_ms": [1.0, 1.0, 1.0, 0.5, 0.5]},
            {"step_ms": [5.0, 5.0], "verify_ms": [3.0, 3.0], "draft_ms": [2.0, 2.0], "accepts": [5, 5]}]
    w = B.window(runs, 3)
    assert w["total_gpu_ms"] == 20.0 + 30.0 and w["total_emitted_tokens"] == 4 + 5
    assert w["ms_per_token"] == w["total_gpu_ms"] / w["total_emitted_tokens"] == 50.0 / 9
    assert w["total_manager_ms"] == 1.0 and w["ms_per_token_definition"].startswith("total GPU ms / total emitted tokens")
    assert w["all_total_gpu_ms"] == 10 * 3 + 20 + 30 + 5 + 5 and w["all_total_emitted_tokens"] == 1 + 2 + 3 + 4 + 5 + 6 + 6
    e = B.window([], 32)
    assert e["total_gpu_ms"] is None and e["total_emitted_tokens"] is None
    rep = {"date": "d", "mode": "graph", "impls": {}, "max_new": 8, "n_prompts": 0, "headline": {"two_graphs": w},
           "streams": {}, "prompts": {}, "overflow": None}
    lines = B.summary_lines(rep)
    tot = [l for l in lines if "total GPU ms / total emitted tokens" in l]
    assert len(tot) == 1 and "= 50.0 ms / 9 tokens = 5.556" in tot[0] and "[measured:" in tot[0], lines


def test_headline_real_receipt_totals_consistent():
    """On the repo's real headline receipt, the pooled ms/token is total/total of the stored per-step data (derived here)."""
    path = os.path.join(REPO, "docs", "headline-20261107j.json")
    if not os.path.exists(path):
        pytest.skip("headline j not in this checkout")
    w = json.load(open(path))["headline"]["two_graphs"]
    assert w["steps"] == 558 and abs(w["ms_per_step"]["mean"] * w["steps"] / (w["tokens_per_step"] * w["steps"]) - w["ms_per_token"]) < 1e-9


def test_call_sites_stamp_before_every_json_dump():
    hl = open(os.path.join(BENCH, "headline.py")).read()
    fl = open(os.path.join(BENCH, "first_light.py")).read()
    assert hl.count("_receipt.begin(") == 1 and hl.count("_receipt.finish(") == 1
    assert hl.index("_receipt.finish(") < hl.index("json.dump(slim, f")
    assert fl.count("_receipt.begin(") == 1 and fl.count("_receipt.finish(") == 2
    assert len(re.findall(r"_receipt\.finish\(report\.get\(\"receipt\"\)\)\n    with open\(json_path", fl)) == 2
    # env that changes numerics is set before the receipt is begun
    assert fl.index("os.environ[FP32_IMPL_ENV] = a.fp32_impl") < fl.index("_receipt.begin(") and fl.index("rows_moe_env(a)") < fl.index("_receipt.begin(")
    assert hl.index('os.environ["DSV41_MOE_ROWMATCH"]') < hl.index('rep["receipt"] = _begin_receipt(')
    assert 'report["receipt"]' in fl and 'rep["receipt"]' in hl


def test_receipt_import_is_light():
    code = ("import sys; sys.path.insert(0, %r); import receipt; "
            "bad = [m for m in ('torch', 'vllm', 'triton') if m in sys.modules]; assert not bad, bad" % BENCH)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------------ lease.sh
def _shim_flock(bindir):
    bindir.mkdir(exist_ok=True)
    shim = bindir / "flock"
    shim.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, fcntl, time
        a = sys.argv[1:]; nb = False; wait = 0.0; fd = None; i = 0
        while i < len(a):
            if a[i] == "-n": nb = True
            elif a[i] == "-w": wait = float(a[i + 1]); i += 1
            else: fd = int(a[i])
            i += 1
        end = time.time() + (0 if nb else wait)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); sys.exit(0)
            except OSError:
                if time.time() >= end: sys.exit(1)
                time.sleep(0.05)
        """))
    shim.chmod(0o755)
    return bindir


def _bash(script, env, **kw):
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60, **kw)


@pytest.fixture
def leaseenv(tmp_path, repo):
    bindir = _shim_flock(tmp_path / "bin")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RECEIPT_", "BOX_", "DSV41_"))}
    env.update(PATH=f"{bindir}:{os.environ['PATH']}", HOME=str(tmp_path), DSV41_BOX_LOCK=str(tmp_path / "d" / "box.lock"),
               BOX_REPO=str(repo), DSV41_BOX_LEASE_WAIT="1")
    return env


def test_lease_syntax_and_documented_contract():
    r = subprocess.run(["bash", "-n", LEASE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = open(LEASE).read()
    assert "36000" in src and "box.lock" in src and "exit 75" in src and "BOX_LEASE_HELD" in src


def test_lease_acquire_exports_receipt_env_matching_receipt_py(leaseenv, repo):
    (repo / "a.txt").write_text("dirty\n")
    shasum = subprocess.run("command -v sha256sum || command -v shasum", shell=True, capture_output=True, text=True).stdout.strip()
    if not shasum:
        pytest.skip("no sha256 tool")
    r = _bash(f'set -u; source {LEASE}; echo "H=$BOX_LEASE_HELD"; echo "C=$RECEIPT_GIT_COMMIT"; echo "S=$RECEIPT_GIT_DIRTY_SHA256"; '
              f'echo "F=$RECEIPT_GIT_DIRTY_FILES"; echo "U=$RECEIPT_GIT_UNTRACKED"; echo "L=$RECEIPT_LEASE"; box_lease_docker_env', leaseenv)
    assert r.returncode == 0, r.stderr
    kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if re.match(r"^[HCSFUL]=", l))
    want = R.git_state(str(repo))
    assert kv["H"] == "1" and kv["C"] == want["commit"] and kv["S"] == want["dirty_sha256"]
    assert kv["F"] == ",".join(want["dirty_files"]) and kv["U"] == str(want["untracked"])
    assert kv["L"].startswith(leaseenv["DSV41_BOX_LOCK"] + "|") and " " not in kv["L"]
    assert f"-e RECEIPT_GIT_COMMIT={want['commit']}" in r.stdout.splitlines()[-1] and "RECEIPT_LEASE=" in r.stdout.splitlines()[-1]
    assert "box lease taken" in r.stderr
    holder = open(leaseenv["DSV41_BOX_LOCK"] + ".holder").read()
    assert "pid=" not in holder and " pid " in holder           # the human line, informational
    # begin() inside the "container": git cannot run there, the env record is used and agrees with the host's
    cenv = {"RECEIPT_GIT_COMMIT": kv["C"], "RECEIPT_GIT_DIRTY_SHA256": kv["S"], "RECEIPT_GIT_DIRTY_FILES": kv["F"],
            "RECEIPT_GIT_UNTRACKED": kv["U"], "RECEIPT_LEASE": kv["L"]}
    rc = R.begin(repo="/nonexistent", env=cenv)
    assert rc["git_source"] == "env" and rc["git_dirty_sha256"] == want["dirty_sha256"] and rc["lease"] == kv["L"]


def test_lease_clean_tree_reports_no_dirty_sha(leaseenv):
    r = _bash(f'source {LEASE}; echo "S=[$RECEIPT_GIT_DIRTY_SHA256]"', leaseenv)
    assert r.returncode == 0 and "S=[]" in r.stdout, (r.stdout, r.stderr)


def test_lease_blocks_second_holder_and_dead_holder_releases(leaseenv):
    holder = subprocess.Popen(["bash", "-c", f'source {LEASE}; echo ready; exec sleep 60'], env=leaseenv,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        t0 = time.time()
        r2 = _bash(f'source {LEASE}; echo ACQUIRED', leaseenv)
        assert r2.returncode == 75 and "ACQUIRED" not in r2.stdout and "gave up after 1s" in r2.stderr, (r2.stdout, r2.stderr)
        assert "box lease held" in r2.stderr and time.time() - t0 >= 0.9           # it really waited
        holder.kill()                                                               # SIGKILL: the kernel releases the flock
        holder.wait()
        r3 = _bash(f'source {LEASE}; echo ACQUIRED', leaseenv)
        assert r3.returncode == 0 and "ACQUIRED" in r3.stdout, (r3.stdout, r3.stderr)
    finally:
        holder.kill()


def test_lease_waiter_gets_it_when_holder_exits(leaseenv):
    env = dict(leaseenv, DSV41_BOX_LEASE_WAIT="20")
    holder = subprocess.Popen(["bash", "-c", f'source {LEASE}; echo ready; sleep 1.5'], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "ready"
    r = _bash(f'source {LEASE}; echo ACQUIRED', env)
    holder.wait()
    assert r.returncode == 0 and "ACQUIRED" in r.stdout and "waiting up to 20s" in r.stderr, (r.stdout, r.stderr)


def test_lease_reentrant_and_fails_closed_without_flock(leaseenv, tmp_path):
    r = _bash(f'source {LEASE}; bash -c "source {LEASE}; echo CHILD_OK \\$BOX_LEASE_HELD"', leaseenv)    # child under a holder
    assert r.returncode == 0 and "CHILD_OK 1" in r.stdout and r.stderr.count("box lease taken") == 1, (r.stdout, r.stderr)
    empty = tmp_path / "empty"
    empty.mkdir()
    env = dict(leaseenv, PATH=str(empty))
    r = subprocess.run(["/bin/bash", "-c", f"source {LEASE}; echo RAN_UNLEASED"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 75 and "RAN_UNLEASED" not in r.stdout and "no flock binary" in r.stderr


# ------------------------------------------------------------------------------------------------ A3 doc
@need(PARITY_A, PARITY_A2, CAND_NIGHT, A3DOC)
def test_a3_doc_numbers_recompute():
    import parity as P

    ref, a2, cand = P.load(PARITY_A), P.load(PARITY_A2), P.load(CAND_NIGHT)
    doc = open(A3DOC).read()
    # structural facts: top-1 flips cannot occur in the compared prefix
    for d, n in ((ref, 1476), (a2, 1444), (cand, 1501)):
        pos = [lp for r in d.values() for lp in r["logprobs"]]
        assert len(pos) == n and f"{n:,}" in doc
        assert all(max(lp["top_logprobs"], key=lambda t: t["logprob"])["token"] == lp["token"] for lp in pos)
    rows = {t: P.compare(ref[t], cand[t]) for t in ref}
    assert all(c["top1_flips"] == 0 and c["top1_flips_weighted"] == 0.0 for c in rows.values())
    code = rows["code"]
    assert code["first_divergence"] == 114 and code["max_abs_dlogprob"] == 0.31493 and "0.31493" in doc
    assert P.compare(ref["code"], a2["code"])["first_divergence"] == 121 and P.compare(ref["code"], a2["code"])["max_abs_dlogprob"] == 0.12061
    tot = n_pos = 0
    for t, c in rows.items():
        e = c["shared_prefix"]
        ds = [abs(ref[t]["logprobs"][i]["logprob"] - cand[t]["logprobs"][i]["logprob"]) for i in range(e)]
        assert f"{sum(ds) / len(ds):.5f}" in doc, t                        # the derived mean |dlogp| column
        assert f"{c['max_abs_dlogprob']:.5f}" in doc, t
        tot, n_pos = tot + sum(ds), n_pos + e
    assert n_pos == 333 and f"{tot:.4f}" in doc and f"{tot / n_pos:.5f}" in doc
    assert sum(1 for c in rows.values() if c["first_divergence"] is None) == 2 and "identical 2/8" in doc
    ties = sum(1 for r in ref.values() for lp in r["logprobs"]
               if sorted((x["logprob"] for x in lp["top_logprobs"]), reverse=True)[:2][0] == sorted((x["logprob"] for x in lp["top_logprobs"]), reverse=True)[:2][1])
    assert ties == 18 and "18 of 1,476" in doc
    assert sum(len(r["logprobs"]) for r in ref.values()) == 1476
    # the stored first-light tables carry the same cand numbers
    fl = os.path.join(REPO, "docs", "first-light-t9b4-20261107.json")
    if os.path.exists(fl):
        pp = json.load(open(fl))["parity"]["1"]["per_prompt"]
        assert pp["code"]["cand"]["max_abs_dlogprob"] == code["max_abs_dlogprob"] and pp["code"]["cand"]["first_divergence"] == 114


@need(os.path.join(REPO, "docs", "gap-tf-static-20261107.json"), A3DOC)
def test_a3_doc_teacher_forced_receipt_numbers():
    d = json.load(open(os.path.join(REPO, "docs", "gap-tf-static-20261107.json")))
    cells = d["reps"][0]["prompts"]
    agree = sum(c["argmax_agree_with_pg61"] for c in cells)
    comp = sum(c["argmax_compared"] for c in cells)
    doc = open(A3DOC).read()
    assert (agree, comp, comp - agree) == (1391, 1446, 55) and "1,391 / 1,446" in doc and "55 disagreements" in doc
    assert f"{(comp - agree) / comp * 100:.1f} %" in doc


def test_a3_doc_has_teacher_forced_spec_not_implemented():
    doc = open(A3DOC).read()
    assert "teacher-forced fixed-prefix arm" in doc and "SPEC — not implemented" in doc and "Pre-register" in doc
