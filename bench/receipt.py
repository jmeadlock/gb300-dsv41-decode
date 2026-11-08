#!/usr/bin/env python3
"""receipt.py -- LOOSE-ENDS-2 (Astra review, accepted): every box receipt says exactly what produced it.

  from receipt import begin, finish          # bench/headline.py, bench/first_light.py
  rep["receipt"] = begin(argv=..., impls=..., model_dir=..., fixture=..., residency=..., map_path=...)
  ...                                        # the run
  finish(rep["receipt"])                     # end wall time + clocks, just before the JSON is written

Every field is MEASURED at run time (a command's output, a file's hash, a library's __version__), never typed in;
a probe that fails stores None and appends '<field>: <error>' to `receipt_errors` -- stamping never raises and never
changes a run. RECEIPT_FIELDS is the contract (tests/test_receipt_cpu.py); the count is len(RECEIPT_FIELDS).

Where a field comes from:
  git_*              `git rev-parse HEAD`, `git rev-parse --abbrev-ref HEAD`, `git diff HEAD` (tracked files, staged +
                     unstaged, binary-safe) -> sha256 of that diff + the changed file names. Untracked files are
                     counted, not hashed (scratch/ receipts and dispatch logs live there). When git cannot run (the
                     vllm container has no git / 'dubious ownership'), the RECEIPT_GIT_* environment variables that
                     bench/box/lease.sh exports into the container are used and `git_source` says 'env'.
  env_switches       every DSV41_* variable + CUBLAS_WORKSPACE_CONFIG (the numerics-relevant ones), as set at begin()
                     plus any set during the run (re-read at finish()).
  checkpoint_*       <model_dir>/config.json (or --config) path + sha256; the weights themselves are NOT hashed (too
                     big); the checkpoint directory name carries the commit hash of the snapshot.
  fixture_*          the --fixture file path + sha256.
  torch/triton/cuda  torch.__version__, triton.__version__, torch.version.cuda (None if not importable).
  driver / gpu       nvidia-smi --query-gpu=driver_version,uuid,name (torch's device properties as a fallback for the uuid).
  clocks_*           nvidia-smi --query-gpu=clocks.sm,clocks.mem (MHz) at begin() and at finish(): the receipt for
                     'was the GPU throttled / idle-clocked during this number'.
  residency          one line: the A2-2 residency condition ('off, static map <path>' or the config).
  lease              RECEIPT_LEASE (exported by lease.sh when this process runs under the box lock; None otherwise).
  start/end          wall-clock time.time() + ISO local time; wall_s = end - start.
"""
from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

SCHEMA = "dsv41-receipt/1"
ENV_PREFIXES = ("DSV41_", "CUBLAS_WORKSPACE_CONFIG")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECEIPT_FIELDS = (
    "schema", "argv", "host", "python",
    "git_commit", "git_branch", "git_dirty", "git_dirty_sha256", "git_dirty_files", "git_untracked_count", "git_source",
    "env_switches", "impls",
    "checkpoint_path", "checkpoint_config_path", "checkpoint_config_sha256",
    "fixture_path", "fixture_sha256",
    "torch_version", "triton_version", "cuda_version",
    "driver_version", "gpu_name", "gpu_uuid",
    "clocks_sm_mhz_start", "clocks_mem_mhz_start", "clocks_sm_mhz_end", "clocks_mem_mhz_end",
    "residency_condition", "lease",
    "start_wall", "start_iso", "end_wall", "end_iso", "wall_s",
    "receipt_errors",
)


# ------------------------------------------------------------------------------------------------ probes (all total)
def _run(cmd: Sequence[str], cwd: Optional[str] = None, timeout: float = 20.0) -> bytes:
    p = subprocess.run(list(cmd), cwd=cwd, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} rc={p.returncode}: {p.stderr.decode(errors='replace').strip()[:160]}")
    return p.stdout


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: str = REPO) -> Dict[str, Any]:
    """{'commit','branch','dirty','dirty_sha256','dirty_files','untracked','source'} -- 'dirty_sha256' is the sha256
    of `git diff HEAD --binary` (tracked files only; None when the tree is clean); raises if git cannot run."""
    g = ["git", "-c", "safe.directory=*", "-C", repo]
    commit = _run(g + ["rev-parse", "HEAD"]).decode().strip()
    branch = _run(g + ["rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    diff = _run(g + ["diff", "HEAD", "--binary"], timeout=60.0)
    names = [x for x in _run(g + ["diff", "HEAD", "--name-only"]).decode().splitlines() if x]
    untracked = [x for x in _run(g + ["ls-files", "--others", "--exclude-standard"]).decode().splitlines() if x]
    return {"commit": commit, "branch": branch, "dirty": bool(diff), "dirty_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
            "dirty_files": names, "untracked": len(untracked), "source": "git"}


def git_state_env(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The same record from RECEIPT_GIT_* (exported by bench/box/lease.sh on the host, passed into the container)."""
    e = os.environ if env is None else env
    if not e.get("RECEIPT_GIT_COMMIT"):
        raise RuntimeError("no RECEIPT_GIT_COMMIT in the environment")
    files = [x for x in e.get("RECEIPT_GIT_DIRTY_FILES", "").split(",") if x]
    sha = e.get("RECEIPT_GIT_DIRTY_SHA256") or None
    return {"commit": e["RECEIPT_GIT_COMMIT"], "branch": e.get("RECEIPT_GIT_BRANCH") or None, "dirty": bool(sha),
            "dirty_sha256": sha, "dirty_files": files, "untracked": int(e.get("RECEIPT_GIT_UNTRACKED") or 0), "source": "env"}


def env_switches(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    e = os.environ if env is None else env
    return {k: e[k] for k in sorted(e) if k.startswith(ENV_PREFIXES)}


def parse_smi(text: str, n: int) -> List[List[str]]:
    """nvidia-smi --format=csv,noheader,nounits output -> rows of n stripped cells (blank lines dropped)."""
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= n and any(cells):
            rows.append(cells[:n])
    return rows


def smi_query(fields: str) -> List[List[str]]:
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]).decode()
    return parse_smi(out, fields.count(",") + 1)


def clocks() -> Dict[str, Optional[float]]:
    """{'sm': MHz, 'mem': MHz} of the first GPU (`nvidia-smi --query-gpu=clocks.sm,clocks.mem`)."""
    row = smi_query("clocks.sm,clocks.mem")[0]
    return {"sm": float(row[0]), "mem": float(row[1])}


def residency_condition(residency: Any = None, map_path: Optional[str] = None) -> str:
    """One line for the A2-2 residency state: 'off (static map <path>)' or 'on <config>'. `residency` is the
    runtime.residency config (a dict / dataclass / None)."""
    if not residency:
        return f"off (static map {map_path})" if map_path else "off (static map)"
    cfg = residency if isinstance(residency, dict) else getattr(residency, "__dict__", None) or {"config": str(residency)}
    return "on " + ", ".join(f"{k}={cfg[k]}" for k in sorted(cfg)) + (f" (map {map_path})" if map_path else "")


# ------------------------------------------------------------------------------------------------ begin / finish
def begin(argv: Optional[Sequence[str]] = None, impls: Optional[dict] = None, model_dir: Optional[str] = None,
          fixture: Optional[str] = None, residency: Any = None, map_path: Optional[str] = None,
          config_path: Optional[str] = None, repo: str = REPO, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The receipt as of the start of a run (call right after argument parsing, before the build). Never raises."""
    rc: Dict[str, Any] = {k: None for k in RECEIPT_FIELDS}
    errs: List[str] = []

    def probe(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - a receipt must never break a run
            errs.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
            return None

    now = time.time()
    rc.update(schema=SCHEMA, argv=list(argv) if argv is not None else list(sys.argv[1:]), host=socket.gethostname(),
              python=f"{platform.python_version()} {platform.machine()}", start_wall=now,
              start_iso=time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(now)))
    g = probe("git", lambda: git_state(repo))
    if g is None:
        g = probe("git_env", lambda: git_state_env(env))
    if g:
        rc.update(git_commit=g["commit"], git_branch=g["branch"], git_dirty=g["dirty"], git_dirty_sha256=g["dirty_sha256"],
                  git_dirty_files=g["dirty_files"], git_untracked_count=g["untracked"], git_source=g["source"])
    rc["env_switches"] = env_switches(env)
    rc["impls"] = dict(impls) if impls else None
    if model_dir:
        rc["checkpoint_path"] = model_dir
        cfg = config_path or os.path.join(model_dir, "config.json")
        rc["checkpoint_config_path"] = cfg
        rc["checkpoint_config_sha256"] = probe("checkpoint_config", lambda: sha256_file(cfg))
    if fixture:
        rc["fixture_path"] = fixture
        rc["fixture_sha256"] = probe("fixture", lambda: sha256_file(fixture))
    for name, mod in (("torch_version", "torch"), ("triton_version", "triton")):
        rc[name] = probe(name, lambda m=mod: __import__(m).__version__)
    rc["cuda_version"] = probe("cuda_version", lambda: __import__("torch").version.cuda)
    smi = probe("nvidia-smi", lambda: smi_query("driver_version,uuid,name")[0])
    if smi:
        rc.update(driver_version=smi[0], gpu_uuid=smi[1], gpu_name=smi[2])
    else:
        rc["gpu_uuid"] = probe("gpu_uuid_torch", lambda: str(__import__("torch").cuda.get_device_properties(0).uuid))
        rc["gpu_name"] = probe("gpu_name_torch", lambda: __import__("torch").cuda.get_device_name())
    c = probe("clocks_start", clocks)
    if c:
        rc.update(clocks_sm_mhz_start=c["sm"], clocks_mem_mhz_start=c["mem"])
    rc["residency_condition"] = residency_condition(residency, map_path)
    rc["lease"] = (env if env is not None else os.environ).get("RECEIPT_LEASE")
    rc["receipt_errors"] = errs
    return rc


def finish(rc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add the end wall time, wall_s and the end clocks (in place; returns rc). Call just before the JSON write.
    Total; a None receipt is returned as is."""
    if not isinstance(rc, dict):
        return rc
    now = time.time()
    rc["end_wall"] = now
    rc["end_iso"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(now))
    if rc.get("start_wall") is not None:
        rc["wall_s"] = round(now - rc["start_wall"], 3)
    if isinstance(rc.get("env_switches"), dict):     # switches a module set during the run (e.g. a GEMV pin) join the record
        rc["env_switches"] = {**rc["env_switches"], **env_switches()}
    try:
        c = clocks()
        rc["clocks_sm_mhz_end"], rc["clocks_mem_mhz_end"] = c["sm"], c["mem"]
    except Exception as e:  # noqa: BLE001
        rc.setdefault("receipt_errors", []).append(f"clocks_end: {type(e).__name__}: {str(e)[:200]}")
    return rc


def fields_present(rc: dict) -> int:
    """How many RECEIPT_FIELDS carry a value (not None / empty-list-for-receipt_errors does not count)."""
    return sum(1 for k in RECEIPT_FIELDS if k != "receipt_errors" and rc.get(k) is not None)


def summary_line(rc: Optional[dict]) -> str:
    if not rc:
        return "receipt: none"
    return (f"receipt: {str(rc.get('git_commit'))[:10]}{'+dirty ' + str(rc.get('git_dirty_sha256'))[:8] if rc.get('git_dirty') else ''} | "
            f"torch {rc.get('torch_version')} triton {rc.get('triton_version')} cuda {rc.get('cuda_version')} drv {rc.get('driver_version')} | "
            f"{rc.get('gpu_uuid')} sm {rc.get('clocks_sm_mhz_start')}->{rc.get('clocks_sm_mhz_end')} MHz | "
            f"{fields_present(rc)}/{len(RECEIPT_FIELDS) - 1} fields, {len(rc.get('receipt_errors') or [])} probe errors")


if __name__ == "__main__":      # python3 bench/receipt.py -> print the receipt this checkout/box would stamp now
    import json

    print(json.dumps(finish(begin()), indent=1, default=str))
