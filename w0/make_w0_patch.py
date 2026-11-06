#!/usr/bin/env python3
"""make_w0_patch.py — build ~/dsv41/w0/patches/ = Pg61 patches + 06-nsys-wrap-serve.patch, and launch-w0.sh.
Run on the box. CPU only; touches nothing under ~/dsv41/garner/.
"""
import os, shutil, subprocess, sys, tempfile

GR = os.path.expanduser("~/dsv41/garner")
W0 = os.path.expanduser("~/dsv41/w0")
SRC_SERVE = f"{GR}/pg61-work/tree/serve/serve.sh"   # Pg61's serve.sh = Garner's + patches 01..05 already applied
os.makedirs(f"{W0}/patches", exist_ok=True)

# 1) patches dir = Pg61's, plus 06
for name in sorted(os.listdir(f"{GR}/launcher/patches")):
    s, d = f"{GR}/launcher/patches/{name}", f"{W0}/patches/{name}"
    if os.path.isdir(s):
        if os.path.exists(d): shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

orig = open(SRC_SERVE).read()
s = orig
a = """  # Every variable below is read by vLLM or by the mounted hook/fastpath; the fastpath's tuning values are code
  # constants in its components, not container env.
"""
b = """  # W0 (decode-runtime project, 2026-11-06): optional nsys wrap of the server, a no-op unless DS41_NSYS_WRAP=1.
  local nsysmnt=() nsyspfx=() nsysprof=()
  if [ "${DS41_NSYS_WRAP:-0}" = 1 ]; then
    mkdir -p "$DS41_CACHE_DIR/prof/${DS41_NSYS_TAG:-W0}"
    nsysmnt=(-v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro)
    nsyspfx=(/nsys/target-linux-sbsa-armv8/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --force-overwrite=true -o "/prof/${DS41_NSYS_TAG:-W0}/${DS41_NSYS_TAG:-W0}")
    nsysprof=(--profiler-config.profiler=cuda)
  fi
""" + a
assert s.count(a) == 1; s = s.replace(a, b)
a = """    "${keyargs[@]}" "${mounts[@]}" "${envs[@]}" --entrypoint /usr/bin/numactl "$DS41_IMAGE" --membind=0 \\
    vllm serve --model /model"""
b = """    "${keyargs[@]}" "${mounts[@]}" "${nsysmnt[@]}" "${envs[@]}" --entrypoint /usr/bin/numactl "$DS41_IMAGE" --membind=0 \\
    "${nsyspfx[@]}" vllm serve --model /model"""
assert s.count(a) == 1; s = s.replace(a, b)
a = """    "${effort[@]}" "${offload[@]}" \\
    --host 0.0.0.0 --port "$DS41_PORT" >/dev/null"""
b = """    "${effort[@]}" "${offload[@]}" "${nsysprof[@]}" \\
    --host 0.0.0.0 --port "$DS41_PORT" >/dev/null"""
assert s.count(a) == 1; s = s.replace(a, b)

with tempfile.TemporaryDirectory() as td:
    os.makedirs(f"{td}/a/serve"); os.makedirs(f"{td}/b/serve")
    open(f"{td}/a/serve/serve.sh", "w").write(orig); open(f"{td}/b/serve/serve.sh", "w").write(s)
    p = subprocess.run(["diff", "-u", "a/serve/serve.sh", "b/serve/serve.sh"], cwd=td, capture_output=True, text=True)
    assert p.returncode == 1, p
    patch = p.stdout.replace("--- a/serve/serve.sh", "--- a/serve/serve.sh").replace("+++ b/serve/serve.sh", "+++ b/serve/serve.sh")
    # strip timestamps diff adds after the filenames
    lines = patch.splitlines()
    lines[0] = "--- a/serve/serve.sh"; lines[1] = "+++ b/serve/serve.sh"
    open(f"{W0}/patches/06-nsys-wrap-serve.patch", "w").write("\n".join(lines) + "\n")
print("06-nsys-wrap-serve.patch:", sum(1 for l in open(f"{W0}/patches/06-nsys-wrap-serve.patch") if l.startswith("+") and not l.startswith("+++")), "added lines")

# 2) launch-w0.sh = launch-pg61.sh + two policy lines (nsys wrap baked into policy.env at build time)
L = open(f"{GR}/launcher/launch-pg61.sh").read()
a = "DS41_PEER3_SPINS=1000000000000\nEOF\n"
b = "DS41_PEER3_SPINS=1000000000000\nDS41_NSYS_WRAP=${DS41_NSYS_WRAP:-0}\nDS41_NSYS_TAG=${DS41_NSYS_TAG:-W0}\nEOF\n"
assert L.count(a) == 1; L = L.replace(a, b)
open(f"{W0}/launch-w0.sh", "w").write(L); os.chmod(f"{W0}/launch-w0.sh", 0o755)
d = subprocess.run(["diff", f"{GR}/launcher/launch-pg61.sh", f"{W0}/launch-w0.sh"], capture_output=True, text=True).stdout
print("launch-w0.sh diff vs launch-pg61.sh:\n" + d)
assert d.count("\n> ") == 2 and d.count("\n< ") == 0
subprocess.run(["bash", "-n", f"{W0}/launch-w0.sh"], check=True)
# dry-apply the patch set on a scratch copy of Garner's pinned tree to prove 06 applies after 01..05
src = f"{GR}/src-git/deepseek-v4.1-flash"
with tempfile.TemporaryDirectory() as td:
    shutil.copytree(src, f"{td}/tree")
    for name in sorted(os.listdir(f"{W0}/patches")):
        if not name.endswith(".patch"): continue
        r = subprocess.run(["patch", "-d", f"{td}/tree", "-p1", "--forward", "--no-backup-if-mismatch", "-i", f"{W0}/patches/{name}"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stdout, r.stderr)
    subprocess.run(["bash", "-n", f"{td}/tree/serve/serve.sh"], check=True)
    t = open(f"{td}/tree/serve/serve.sh").read()
    assert t.count("nsyspfx") == 3 and t.count("profiler-config.profiler=cuda") == 1
print("patch set applies cleanly on the pinned tree; serve.sh bash -n OK")
