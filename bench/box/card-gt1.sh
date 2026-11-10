#!/bin/bash
# card-gt1.sh -- card GT1: Pg61 down -> gracetier (cold experts in Grace, no RTX sidecar) built + booted on :30006 ->
# gates G1..G5 -> down -> Pg61 restored. Receipts to $DEC/scratch/gt1-* only; everything timestamped.
# Design docs/P1-GRACETIER-DESIGN.md; handoff docs/HANDOFF-P1-BUILD2.md; precedent bench/box/window-gt-mt-1108.sh.
#
#   GT1_WINDOW=<pre-registered window id> bash bench/box/card-gt1.sh
#
# Needs James's go and a pre-registered window (stopping :30006). Run from the box's decode checkout, already at the
# commit to test (this script does NOT git pull: bash reads a running script incrementally).
# Optional env: GT1_HOT_MAP=<repo-relative hot map, e.g. docs/maps/hot-v26ar-seed-295.json> (default: the h255 split),
#   GT1_SEED_CACHE=1 (copy Pg61's JIT kernel caches -- jit/ and flashinfer/ only, never vLLM's compile cache),
#   GT1_NSYS=1 (A2 anatomy: nsys-wrapped server, one profiled cell after G1, then exit),
#   GT1_RES=1 (card GT3, docs/HANDOFF-A3-RESIDENCY.md: adaptive residency. The build also copies runtime/gracetier_res.py
#     + residency_policy.py into gracetier_rt/ and writes fastpath/gracetier-res.json = $GT1_RES_CONFIG, default
#     {"mode": "adaptive", "freeze": true}; after G1 the cells R0..R5 below run instead of G2..G5, then exit),
#   MODEL_DIR, GR, GT, DEC (box paths below).
#
# The tree: a copy of the Pg61 built tree ($GR/pg61-work/tree = Garner's clone at 8043a681 + our lane patches 01-05 +
# libw6mk.so, as launch-pg61.sh build made it), checked against the pin for the two files 0001/0002 touch, then
#   port/garner/patches/0001-boundary-gracetier.patch, 0002-components-gracetier.patch, port/garner/serve-gracetier.patch
#   + fastpath_gracetier.py + fastpath/gracetier_rt/runtime/ (our runtime modules it imports) + hook/rowmap-allres.json.
# Boot: launch-gt1.sh = launch-pg61.sh cloned (served name dsv41-flash-gracetier, coldskip,peer3 -> gracetier, no
# sidecar watchdog); policy.env = Pg61's with coldskip,peer3 -> gracetier and the served name (DS41_OFFLOAD_GB=48,
# fp8_ds_mla KV, 1M maxlen, port 30006 kept and asserted).
# Gates (each to scratch/gt1-<gate>-$STAMP.{json,log}):
#   G1 boot log: `DS41_FASTPATH ACTIVE gracetier` + the post-load HOST_RAM_BUDGET line (fits)      -- fail stops G2..G5
#   G2 w0/w0_drive.py parity (greedy fixture, top-20 logprobs) -> bench/parity.py vs w0/results/phase1/parity-A.jsonl
#      (A2 = Pg61's own spread): first divergence, top-1 flips / margin-weighted / max|dlogprob| per prompt. A3 class
#      expected (peer3 is the side that changes numerics); reported, not gated.
#   G3 w0/w0_drive.py step: 1 stream, 3 reps -> ms/step (spec counters) and ms/token; Pg61's phase-1 receipt beside it
#   G4 bench/c32_smoke.py: 32 concurrent /v1/completions x >= 128 tokens; must not crash (all ok, server alive, no restart)
#   G5 prefill / TTFT on the 8 fixture prompts (streamed, salted so the prefix cache cannot serve them), 2 reps
# Card GT3 cells (GT1_RES=1; each to scratch/gt1-res-<cell>-$STAMP.*; control + telemetry in $GT/home/state/prof/gracetier-res,
# the container's /prof/gracetier-res; bench/gt_res_drive.py writes ctrl.json and drives traffic until the server's
# telemetry acknowledges it -- the manager reads the file at a counter drain, every 64 execute_model calls):
#   R0  frozen at the boot map: w0_drive.py step (3 reps) = the same-boot static control; + a fixture capture (R3's reference)
#   R1  unfreeze + baseline; C1 = the 8 fixture prompts x 256 tokens, looped 90 s -> cold share, swaps per drain, T90
#   R2a freeze (map fitted to the fixture); step cell (the step prompt is not in the fixture) -> held-out ms/step
#   R2b unfreeze; step cell x 5 -> adaptive ms/step = the last cell's 3 reps
#   R3  freeze; fixture capture twice -> self-identical N/8; vs the R0-map capture N/8 (reported, not gated)
#   R4  audit (every moved expert + 64 random per layer); restore_map; audit again; step cell -> mismatches (must be 0),
#       ms/step back at R0 within 2 %
#   R5  unfreeze; bench/c32_smoke.py; server alive
#   Verdict: GT3: R0 <ms> | R2a <ms> | R2b <ms> | T90 <s> | cold share <a> -> <b> | audit <n> mismatches | C32 <pass>
#   Adopt bar (pre-registered): R2b <= 0.90 x R0, T90 <= 20 s, audits 0, C32 alive, R4 within 2 % of R0.
# Exit: 0 = ran (verdict line in the log + scratch/gt1-summary-$STAMP.json), 2 = preflight refused (nothing touched).
set -u
GR=${GR:-/home/milo/dsv41/garner}
GT=${GT:-/home/milo/dsv41/gracetier}
DEC=${DEC:-/home/milo/dsv41/decode}
PG61_WORK=$GR/pg61-work; PG61_HOME=$GR/home; SRC=$GR/src-git/deepseek-v4.1-flash
MODEL_DIR=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
PIN=8043a6811a5feda70fdfb6e827eebb2ec25c65d1
PORT=30006; SERVED=dsv41-flash-gracetier; BASE=http://127.0.0.1:$PORT
IMAGE=vllm/vllm-openai:v0.31.0
LO1=/home/milo/dsv41/m3/LANE-OWNER.txt; LO2=/home/milo/dsv41/m3/logs/LANE-OWNER.txt
MF=/home/milo/dsv41/m3/logs/MAINTENANCE
P=$DEC/port/garner
STAMP=$(date +%Y%m%d-%H%M%S)
S=$DEC/scratch
LOG=$S/gt1-window-$STAMP.log
BUSY_USED_MIB=8192
ts() { date '+%F %T %Z'; }
say() { echo "$(ts) [gt1] $*"; }
note() { echo "- $(ts) $*" | tee -a "$LO1" >> "$LO2"; }
mkdir -p "$S"
exec > >(tee -a "$LOG") 2>&1

# ================================================================================================ preflight (no changes)
say "== card GT1 preflight (window ${GT1_WINDOW:-UNSET}) log $LOG"
export STAMP
refuse() {
  say "PREFLIGHT REFUSED: $*"
  REASON="$*" python3 -c 'import json, os, sys; json.dump({"stamp": os.environ["STAMP"], "refused": os.environ["REASON"]}, open(sys.argv[1], "w"))' "$S/gt1-preflight-$STAMP.json"
  exit 2
}
[ -n "${GT1_WINDOW:-}" ] || refuse "GT1_WINDOW unset (James's go + a pre-registered window id)"
command -v flock >/dev/null || refuse "missing flock"
mkdir -p "$(dirname "$GT")"
exec 9>"$GT.card-gt1.lock"
flock -n 9 || refuse "another card-gt1 run holds $GT.card-gt1.lock"
command -v docker >/dev/null && command -v nvidia-smi >/dev/null && command -v python3 >/dev/null \
  && command -v rsync >/dev/null && command -v patch >/dev/null || refuse "missing docker / nvidia-smi / python3 / rsync / patch"
others=$(docker ps --format '{{.Names}}' | grep -vxE 'ds41|ds41-sidecar' || true)
[ -z "$others" ] || refuse "containers other than Pg61's are running: $(echo $others)"
runners=$(pgrep -af "window-gt|gracetier_minitest|w0_runner|restore-pg61|rollback-pg61|headline\.py|first_light\.py" | grep -v "^$$ " || true)
[ -z "$runners" ] || refuse "another runner is active: $runners"
for f in "$P/fastpath/fastpath_gracetier.py" "$P/patches/0001-boundary-gracetier.patch" \
         "$P/patches/0002-components-gracetier.patch" "$P/serve-gracetier.patch" "$P/make_rowmap_allres.py" \
         "$DEC/bench/c32_smoke.py" "$DEC/bench/parity.py" "$DEC/w0/w0_drive.py" \
         "$DEC/w0/results/phase1/parity-A.jsonl" "$DEC/w0/results/phase1/parity-A2.jsonl" \
         "$DEC/w0/results/phase1/step-pg61.json" "$GR/launcher/launch-pg61.sh" "$PG61_WORK/policy.env" \
         "$GR/promote/restore-pg61-after-gt-mt-1108.sh"; do
  [ -f "$f" ] || refuse "missing $f"
done
for m in __init__ gracetier_vmm gracetier_pool gracetier_stage dg_mega dg_moe host_tier ckpt; do
  [ -f "$DEC/runtime/$m.py" ] || refuse "missing runtime/$m.py"
done
RES_RT=""; RESDIR=$GT/home/state/prof/gracetier-res
if [ "${GT1_RES:-0}" = 1 ]; then   # card GT3: the residency manager, its vendored policy, the driver, a valid config
  RES_RT="gracetier_res residency_policy"
  for f in "$DEC/runtime/gracetier_res.py" "$DEC/runtime/residency_policy.py" "$DEC/bench/gt_res_drive.py"; do
    [ -f "$f" ] || refuse "GT1_RES=1: missing $f"
  done
  [ "${GT1_NSYS:-0}" != 1 ] || refuse "GT1_RES=1 and GT1_NSYS=1 are separate cards (one mode per window)"
  GT1_RES_CONFIG=${GT1_RES_CONFIG:-'{"mode": "adaptive", "freeze": true}'}
  RESCFG="$GT1_RES_CONFIG" python3 -c 'import ast, json, os, sys
c = json.loads(os.environ["RESCFG"]); assert isinstance(c, dict) and c.get("mode") == "adaptive", c
ast.parse(open(sys.argv[1]).read()); ast.parse(open(sys.argv[2]).read())' \
    "$DEC/bench/gt_res_drive.py" "$DEC/runtime/gracetier_res.py" \
    || refuse "GT1_RES=1: GT1_RES_CONFIG is not a JSON object with mode adaptive, or python syntax"
fi
for d in "$PG61_WORK/tree" "$PG61_WORK/overlay-vllm" "$SRC" "$MODEL_DIR"; do [ -d "$d" ] || refuse "missing dir $d"; done
[ "$(git -C "$GR/src-git" rev-parse HEAD 2>/dev/null)" = "$PIN" ] || refuse "Garner's clone $GR/src-git is not at $PIN"
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$P/fastpath/fastpath_gracetier.py" \
  "$P/make_rowmap_allres.py" "$DEC/bench/c32_smoke.py" "$DEC/w0/w0_drive.py" "$DEC/bench/parity.py" \
  || refuse "python syntax"
[ -z "${GT1_HOT_MAP:-}" ] || [ -f "$DEC/$GT1_HOT_MAP" ] || refuse "GT1_HOT_MAP $GT1_HOT_MAP not found"
SMI=$(nvidia-smi --query-gpu=name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -1)
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | paste -sd';' -)
MEMAV=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
PG61_BOUND=$(curl -sf -m 3 "$BASE/v1/models" | grep -o '"id":"[^"]*"' | paste -sd, - || true)
if ! docker ps --format '{{.Names}}' | grep -qx ds41; then
  used=$(echo "$SMI" | awk -F', ' '{print $3}' | awk '{print $1}')
  [ -z "$APPS" ] && [ "${used:-99999}" -lt "$BUSY_USED_MIB" ] || refuse "GPU not idle with Pg61 down: $SMI apps=$APPS"
fi
COMMIT=$(git -C "$DEC" log --oneline -1 2>/dev/null)
DIRTY=$(git -C "$DEC" status --short port runtime bench w0 2>/dev/null | paste -sd';' -)
RESTORE_PG61=0
case "${GT1_RESTORE:-auto}" in
  always) RESTORE_PG61=1 ;;
  auto) echo "$PG61_BOUND" | grep -q '"id":"dsv41-flash-uva"' && RESTORE_PG61=1 ;;
  never) RESTORE_PG61=0 ;;
  *) refuse "GT1_RESTORE must be auto|always|never" ;;
esac
export GT1_WINDOW COMMIT DIRTY SMI APPS MEMAV PG61_BOUND MODEL_DIR RESTORE_PG61
python3 - "$S/gt1-preflight-$STAMP.json" <<'PY'
import json, os, sys
e = os.environ
json.dump({"stamp": e["STAMP"], "window": e["GT1_WINDOW"], "commit": e["COMMIT"], "dirty": e["DIRTY"],
           "nvidia_smi": e["SMI"], "compute_apps": e["APPS"], "MemAvailable_GiB": float(e["MEMAV"] or "nan"),
           "pg61_models": e["PG61_BOUND"], "restore_pg61_at_end": e["RESTORE_PG61"] == "1",
           "model_dir": e["MODEL_DIR"], "hot_map": e.get("GT1_HOT_MAP", ""), "refused": None},
          open(sys.argv[1], "w"), indent=1)
PY
say "preflight OK: commit $COMMIT; dirty [$DIRTY]; GPU $SMI; apps [$APPS]; MemAvailable ${MEMAV} GiB; :$PORT serves [$PG61_BOUND]; restore Pg61 at end: $RESTORE_PG61 (GT1_RESTORE=${GT1_RESTORE:-auto})"

# ================================================================================================ window
PG61_DOWN=0; GT_UP=0; FINISHED=0
G1=skip; G2=skip; G3=skip; G4=skip; G5=skip
gpu_idle() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  say "GPU memory.used ${used} MiB, compute apps ${apps}"
  [ "${used:-99999}" -lt "$BUSY_USED_MIB" ] && [ "${apps:-1}" -eq 0 ]
}
gt_down() {
  [ -f "$GT/launch-gt1.sh" ] || return 0
  ( cd "$GT" && WORK=$GT DS41_HOME=$GT/home bash "$GT/launch-gt1.sh" down ) > "$S/gt1-down-$STAMP.log" 2>&1
  local rc=$?; say "gracetier down rc=$rc ($(tail -1 "$S/gt1-down-$STAMP.log" | cut -c1-160))"
}
redact_key() {   # idempotent; the up log is the receipt that carries the launcher's "API key (save it): <hex>" line
  [ -f "$S/gt1-up-$STAMP.log" ] && sed -E -i 's/(API key \(save it\): )[0-9a-f]+/\1<redacted>/' "$S/gt1-up-$STAMP.log"
  return 0
}
finish() {
  [ "$FINISHED" = 1 ] && return; FINISHED=1
  say "== finish (G1=$G1 G2=$G2 G3=$G3 G4=$G4 G5=$G5)"
  redact_key
  if docker ps -a --format '{{.Names}}' | grep -qx ds41 && [ "$GT_UP" = 1 ]; then
    docker logs ds41 > "$S/gt1-server-$STAMP.log" 2>&1 || true
  fi
  [ "$GT_UP" = 1 ] && gt_down
  docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && say "WARN: ds41 containers still up after down"
  local rrc=skipped
  if [ "$PG61_DOWN" = 1 ] && [ "$RESTORE_PG61" = 0 ]; then
    say "Pg61 was not serving at preflight (GT1_RESTORE=${GT1_RESTORE:-auto}): not restored, lane left as found"
  elif [ "$PG61_DOWN" = 1 ]; then
    say "== restore Pg61"
    (
      cd "$GR/promote" || exit 9
      [ -f restore-pg61-after-gt1.sh ] || cp restore-pg61-after-gt-mt-1108.sh restore-pg61-after-gt1.sh || exit 9
      python3 - restore-pg61-after-gt1.sh <<'PY' || exit 9
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('the gracetier mini-test window gt-mt-1108', 'the gracetier card GT1 window').replace('gt-mt-1108', 'gt1')
open(p, 'w').write(s)
PY
      bash -n restore-pg61-after-gt1.sh || exit 9
      say "restore clone: $(diff restore-pg61-after-gt-mt-1108.sh restore-pg61-after-gt1.sh | grep -c '^[<>]') changed lines"
      bash restore-pg61-after-gt1.sh 9>&-   # the p3 watchdog it re-arms must not inherit the card lock (fd 9)
    ) > "$S/gt1-restore-$STAMP.log" 2>&1
    rrc=$?
    say "restore rc=$rrc ($(tail -2 "$S/gt1-restore-$STAMP.log" | tr '\n' ' ' | cut -c1-200))"
  fi
  if [ "$rrc" = 0 ] || [ "$rrc" = skipped ]; then   # window over: our MAINTENANCE flag goes; a prior one comes back
    [ -f "$MF" ] && mv "$MF" "$MF.stale-gt1-$STAMP"
    [ -f "$MF.pre-gt1-$STAMP" ] && [ "$rrc" = skipped ] && mv "$MF.pre-gt1-$STAMP" "$MF"
  fi
  python3 - "$S" "$STAMP" "$G1" "$G2" "$G3" "$G4" "$G5" "$rrc" <<'PY'
import glob, json, os, sys
S, stamp, g1, g2, g3, g4, g5, rrc = sys.argv[1:9]
def rd(n):
    p = os.path.join(S, f"gt1-{n}-{stamp}.json")
    try:
        return json.load(open(p))
    except Exception:
        return None
out = {"stamp": stamp, "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}, "restore_rc": rrc,
       "g1": rd("g1"), "g3": (rd("g3") or {}).get("summary"), "g4": (rd("g4") or {}).get("summary"),
       "g5": (rd("g5") or {}).get("summary"), "g2": (rd("g2s") or {})}
json.dump(out, open(os.path.join(S, f"gt1-summary-{stamp}.json"), "w"), indent=1)
g3s, g4s, g5s, g2s = out["g3"] or {}, out["g4"] or {}, out["g5"] or {}, out["g2"] or {}
print(f"GT1 VERDICT {stamp}: G1 ACTIVE+budget={g1}; G2 parity vs Pg61 A: identical {g2s.get('identical')}/"
      f"{g2s.get('prompts')}, flips {g2s.get('top1_flips')}, weighted {g2s.get('top1_flips_weighted')}, "
      f"max|dlogp| {g2s.get('max_abs_dlogprob')} ({g2}); G3 ms/step {g3s.get('step_ms_median')} ms/token "
      f"{g3s.get('ms_per_token_median')} (Pg61 {g3s.get('pg61_step_ms_median')} / {g3s.get('pg61_ms_per_token_median')}) "
      f"({g3}); G4 C32 {g4s.get('ok')}/{g4s.get('n')} >= {g4s.get('tokens_requested')} tok, alive "
      f"{g4s.get('server_alive_after')} ({g4}); G5 TTFT mean {g5s.get('ttft_ms_mean')} ms ({g5}); restore rc {rrc}")
PY
  note "CARD GT1 END (Milo, window $GT1_WINDOW): G1=$G1 G2=$G2 G3=$G3 G4=$G4 G5=$G5; restore rc=$rrc; receipts $S/gt1-*-$STAMP.*; log $LOG"
  say "== END"
}
trap finish EXIT
trap 'say "TERM/INT received"; exit 143' TERM INT

note "CARD GT1 START (Milo, window $GT1_WINDOW, James go): Pg61 down; gracetier (cold experts in Grace, no RTX) boot on :$PORT + gates G1..G5; then Pg61 restore. log $LOG"
[ -f "$MF" ] && cp "$MF" "$MF.pre-gt1-$STAMP"
echo "card GT1 window $GT1_WINDOW $(ts) -- gracetier on :$PORT; no restore until this runner's END line or James says" > "$MF"

say "== Pg61 down"
( cd "$GR" && WORK=$PG61_WORK DS41_HOME=$PG61_HOME bash launcher/launch-pg61.sh down ) > "$S/gt1-pg61-down-$STAMP.log" 2>&1
drc=$?
say "Pg61 down rc=$drc ($(tail -1 "$S/gt1-pg61-down-$STAMP.log" | cut -c1-160))"
PG61_DOWN=1
sleep 5
docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && { say "FAIL: ds41 containers still up after Pg61 down"; exit 0; }
gpu_idle || { say "FAIL: GPU not idle after Pg61 down"; exit 0; }

# ================================================================================================ build the tree
say "== build $GT (copy of $PG61_WORK/tree + gracetier)"
build() {
  mkdir -p "$GT" "$GT/home"
  rsync -a --delete --exclude __pycache__ "$PG61_WORK/tree/" "$GT/tree/" || return 1
  rsync -a --delete "$PG61_WORK/overlay-vllm/" "$GT/overlay-vllm/" || return 1
  # Pg61's built tree is frozen read-only (BUILD-SHA256SUMS); the copy must be writable for the patches.
  chmod -R u+rwX "$GT/tree" "$GT/overlay-vllm" || return 1
  local f
  for f in fastpath/fastpath_boundary.py fastpath/components.py; do
    cmp -s "$GT/tree/$f" "$SRC/$f" || { say "FAIL: $f in the Pg61 tree differs from the pin $PIN"; return 1; }
  done
  for f in fastpath/fastpath_boundary.py fastpath/components.py serve/serve.sh; do
    cmp -s "$GT/tree/$f" "$PG61_WORK/tree/$f" || { say "FAIL: rsync left $f modified"; return 1; }
  done
  for f in patches/0001-boundary-gracetier.patch patches/0002-components-gracetier.patch serve-gracetier.patch; do
    # strict first: git apply --check takes no fuzz and no offset (patch(1) would silently fuzz a drifted context)
    ( cd "$GT/tree" && GIT_CEILING_DIRECTORIES="$GT" git apply --check "$P/$f" ) \
      || { say "FAIL: $f does not apply exactly (git apply --check)"; return 1; }
    out=$(patch -d "$GT/tree" -p1 --forward --dry-run < "$P/$f" 2>&1) || { say "FAIL: $f dry run: $out"; return 1; }
    echo "$out" | grep -qiE "fuzz|offset|FAILED|reversed|ignored" && { say "FAIL: $f dry run not clean: $out"; return 1; }
    patch -d "$GT/tree" -p1 --forward --no-backup-if-mismatch < "$P/$f" > /dev/null || { say "FAIL: patch $f"; return 1; }
    say "applied $f ($(grep -c '^@@ ' "$P/$f") hunks, clean)"
  done
  cp "$P/fastpath/fastpath_gracetier.py" "$GT/tree/fastpath/" || return 1
  mkdir -p "$GT/tree/fastpath/gracetier_rt/runtime"
  for f in __init__ gracetier_vmm gracetier_pool gracetier_stage dg_mega dg_moe host_tier ckpt $RES_RT; do
    cp "$DEC/runtime/$f.py" "$GT/tree/fastpath/gracetier_rt/runtime/" || return 1
  done
  if [ "${GT1_RES:-0}" = 1 ]; then   # card GT3: the component reads this file (the container forwards no arbitrary env)
    printf '%s\n' "$GT1_RES_CONFIG" > "$GT/tree/fastpath/gracetier-res.json" || return 1
    say "residency: gracetier-res.json = $(cat "$GT/tree/fastpath/gracetier-res.json"); gracetier_rt + $RES_RT"
  fi
  [ "${GT1_RES:-0}" = 1 ] || rm -f "$GT/tree/fastpath/gracetier-res.json"   # the plain / nsys modes: residency off
  python3 "$P/make_rowmap_allres.py" --rowmap "$GT/tree/fastpath/hook/rowmap-mix-v1-h255.json" \
    --out "$GT/tree/fastpath/hook/rowmap-allres.json" ${GT1_HOT_MAP:+--hot-map "$DEC/$GT1_HOT_MAP"} || return 1
  # policy.env = Pg61's with coldskip,peer3 -> gracetier and the served name; overlay -> our copy
  python3 - "$PG61_WORK/policy.env" "$GT/policy.env" "$GT/overlay-vllm" <<'PY_POLICY' || return 1
import re, sys
src, dst, ovl = sys.argv[1:4]
s = open(src).read()
def one(old, new, text):
    n = text.count(old)
    assert n == 1, f"policy.env: {old!r} occurs {n} times"
    return text.replace(old, new)
s = one(",coldskip,peer3,", ",gracetier,", s)
s = one("DS41_SERVED=dsv41-flash-uva\n", "DS41_SERVED=dsv41-flash-gracetier\n", s)
s, n = re.subn(r"^DS41_OVERLAY_58132_DIR=.*$", "DS41_OVERLAY_58132_DIR=" + ovl, s, flags=re.M)
assert n == 1, f"policy.env: DS41_OVERLAY_58132_DIR lines {n}"
for want in ("DS41_PORT=30006\n", "DS41_KV=fp8_ds_mla\n", "DS41_MAXLEN=1048576\n", "DS41_OFFLOAD_GB=48\n"):
    assert want in s, f"policy.env lacks {want.strip()}"
assert "coldskip" not in s and "peer3," not in s and ",gracetier," in s
open(dst, "w").write(s)
print("policy.env: " + " ".join(l for l in s.splitlines() if not l.startswith("DS41_FASTPATH_COMPONENTS")))
PY_POLICY
  if [ "${GT1_NSYS:-0}" = 1 ]; then   # A2 anatomy: W0's optional nsys wrap of the server (no-op unless DS41_NSYS_WRAP=1)
    patch -p1 -d "$GT/tree" --no-backup-if-mismatch < "$HOME/dsv41/w0/patches/06-nsys-wrap-serve.patch" || return 1
    [ "$(grep -c nsyspfx "$GT/tree/serve/serve.sh")" = 3 ] || { echo "nsys wrap: unexpected hunk result"; return 1; }
    printf 'DS41_NSYS_WRAP=1\nDS41_NSYS_TAG=GTA2\n' >> "$GT/policy.env"
    rm -rf "$GT/home/state/prof/GTA2"
    echo "nsys wrap applied (tag GTA2, $(grep -o 'cuda-graph-trace=[^ "]*' "$GT/tree/serve/serve.sh" | head -1))"
  fi
  # launch-gt1.sh = launch-pg61.sh with the served name, the component list, no sidecar restart policy, no p3 watchdog
  python3 - "$GR/launcher/launch-pg61.sh" "$GT/launch-gt1.sh" <<'PY_LAUNCHER' || return 1
import sys
src, dst = sys.argv[1:3]
s = open(src).read()
def one(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"launch-pg61.sh: {old[:70]!r} occurs {n} times"
    s = s.replace(old, new)
one("# launch-pg61.sh \u2014", "# launch-gt1.sh: card GT1 clone of launch-pg61.sh (gracetier: cold experts in Grace, no RTX sidecar,\n"
    "# no p3 watchdog; the tree is built by bench/box/card-gt1.sh). The original header follows.\n# launch-pg61.sh \u2014")
one("SERVED=${SERVED:-dsv41-flash-uva}", "SERVED=${SERVED:-dsv41-flash-gracetier}")
one(",fastload,coldskip,peer3,megamoe,", ",fastload,gracetier,megamoe,")
one('  for c in ds41 ds41-sidecar; do docker update --restart no "$c" >/dev/null; done\n',
    '  docker update --restart no ds41 >/dev/null\n')
a = s.index('  pgrep -f "sidecar_watchdog_p3.py --server ds41" >/dev/null && { log "watchdog already running"; return 0; }\n')
b = s.index('  log "UP on :$PORT; watchdog ENFORCE armed (status $WORK/wdp3.status.json)"\n')
end = b + len('  log "UP on :$PORT; watchdog ENFORCE armed (status $WORK/wdp3.status.json)"\n')
s = s[:a] + '  log "UP on :$PORT (gracetier: no RTX sidecar, no p3 watchdog)"\n' + s[end:]
one("  build) cmd_build ;;", '  build) die "launch-gt1.sh: the tree is built by bench/box/card-gt1.sh" ;;')
open(dst, "w").write(s)
PY_LAUNCHER
  chmod +x "$GT/launch-gt1.sh"
  bash -n "$GT/launch-gt1.sh" || return 1
  say "launch-gt1.sh: $(diff "$GR/launcher/launch-pg61.sh" "$GT/launch-gt1.sh" | grep -c '^[<>]') changed lines vs launch-pg61.sh"
  cp -n "$PG61_HOME/config.env" "$GT/home/config.env" 2>/dev/null || true
  if [ "${GT1_SEED_CACHE:-1}" = 1 ]; then   # content-keyed kernel caches only; vLLM's compile cache is not copied
    for d in jit flashinfer; do
      [ -d "$PG61_HOME/state/$d" ] && [ ! -d "$GT/home/state/$d" ] && mkdir -p "$GT/home/state" && \
        rsync -a "$PG61_HOME/state/$d/" "$GT/home/state/$d/" && say "seeded $GT/home/state/$d from Pg61"
    done
  fi
  ( cd "$GT" && find tree/fastpath tree/serve tree/config overlay-vllm policy.env launch-gt1.sh -type f \
      ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) > "$GT/BUILD-SHA256SUMS"
  cp "$GT/BUILD-SHA256SUMS" "$S/gt1-build-sha256-$STAMP.txt"
  say "built: $(wc -l < "$GT/BUILD-SHA256SUMS") files hashed; component $(sha256sum "$GT/tree/fastpath/fastpath_gracetier.py" | cut -c1-16); rowmap-allres $(sha256sum "$GT/tree/fastpath/hook/rowmap-allres.json" | cut -c1-16)"
}
build > "$S/gt1-build-$STAMP.log" 2>&1; brc=$?
cat "$S/gt1-build-$STAMP.log"
[ "$brc" = 0 ] || { say "FAIL: build rc=$brc"; exit 0; }

# ================================================================================================ boot
say "== boot gracetier on :$PORT ($SERVED)"
GT_UP=1
if [ "${GT1_RES:-0}" = 1 ]; then   # card GT3: this user owns the control dir (the container writes telemetry into it)
  mkdir -p "$RESDIR" && rm -f "$RESDIR"/ctrl.json "$RESDIR"/.ctrl.*.tmp "$RESDIR"/telemetry.jsonl "$RESDIR"/dump-* \
    || { say "FAIL: cannot prepare $RESDIR"; exit 0; }
fi
( cd "$GT" && WORK=$GT MODEL_DIR=$MODEL_DIR DS41_HOME=$GT/home bash "$GT/launch-gt1.sh" up ) > "$S/gt1-up-$STAMP.log" 2>&1
urc=$?
redact_key   # the launcher prints the generated API key: never into a receipt (a key reached the public repo once)
say "up rc=$urc ($(tail -2 "$S/gt1-up-$STAMP.log" | tr '\n' ' ' | cut -c1-200))"
if docker ps -a --format '{{.Names}}' | grep -qx ds41; then
  docker logs ds41 > "$S/gt1-boot-$STAMP.log" 2>&1
else   # serve.sh removed a failed container; it saved the startup log under its state dir
  lastlog=$(ls -t "$GT"/home/state/logs/*-ds41.log 2>/dev/null | head -1)
  [ -n "$lastlog" ] && cp "$lastlog" "$S/gt1-boot-$STAMP.log"
fi

# ---------------------------------------------------------------------------------------------- G1
python3 - "$S/gt1-boot-$STAMP.log" "$S/gt1-g1-$STAMP.json" "$urc" <<'PY_G1'
import json, re, sys
path, out, urc = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    lines = open(path, errors="replace").read().splitlines()
except OSError:
    lines = []
pick = lambda pat: [l[:2000] for l in lines if re.search(pat, l)]
active = pick(r"DS41_FASTPATH ACTIVE gracetier\b")
budget = pick(r"gracetier: HOST_RAM_BUDGET")
post = [l for l in budget if "[post-load" in l]
fits = None
if post:
    m = re.search(r"(\{.*\})\s*$", post[-1])
    try:
        fits = bool(json.loads(m.group(1))["fits"]) if m else None
    except Exception:
        fits = None
rec = {"up_rc": urc, "boot_log": path, "lines": len(lines), "active_line": active[:1], "budget_lines": budget,
       "post_load_fits": fits, "engaged": pick(r"gracetier: ENGAGED")[:1], "parity_slot0": pick(r"gracetier: parity slot0"),
       "symm": pick(r"gracetier: SymmBuffer"), "vmm": pick(r"gracetier: VMM:"), "first_forward": pick(r"first eager stock-forward"),
       "reader": pick(r"gracetier: cold pool"), "all_active": sorted(set(re.findall(r"DS41_FASTPATH ACTIVE (\S+)", "\n".join(lines)))),
       "kv_cache": pick(r"GPU KV cache size")[:2], "a12_memory": pick(r"A12_INSTALL_MEMORY")[:1],
       "fatal": pick(r"FATAL|Traceback|RuntimeError|MemoryError")[:20]}
rec["pass"] = bool(urc == 0 and active and fits)
json.dump(rec, open(out, "w"), indent=1)
print(f"G1: up rc {urc}; ACTIVE gracetier {'yes' if active else 'NO'}; post-load budget fits {fits}; "
      f"components ACTIVE {len(rec['all_active'])}; fatal lines {len(rec['fatal'])} -> {'PASS' if rec['pass'] else 'FAIL'}")
for l in (rec["budget_lines"] + rec["engaged"] + rec["parity_slot0"] + rec["kv_cache"])[:8]:
    print("   ", l[:400])
sys.exit(0 if rec["pass"] else 1)
PY_G1
if [ $? = 0 ]; then G1=pass; else G1=FAIL; say "G1 failed: G2..G5 skipped"; exit 0; fi

cd "$DEC" || exit 0
export W0_BASE=$BASE W0_MODEL=$SERVED
alive() { curl -sf -m 5 "$BASE/v1/models" | grep -q "$SERVED" && \
          [ "$(docker inspect -f '{{.State.Running}} {{.RestartCount}}' ds41 2>/dev/null)" = "true 0" ]; }

# ---------------------------------------------------------------------------------------------- A2 (GT1_NSYS=1)
if [ "${GT1_NSYS:-0}" = 1 ]; then
  say "== A2 nsys anatomy: warm cell under the wrap, then one profiled 512-token greedy cell (the server exits on /stop_profile)"
  PROF=$GT/home/state/prof/GTA2; NSYS=/usr/local/bin/nsys
  docker inspect -f '{{json .Args}}' ds41 | grep -q '/nsys","profile"' || { say "FAIL: ds41 args carry no nsys wrap"; exit 0; }
  timeout 600 python3 w0/w0_drive.py step "$S/gt1-a2-warm-$STAMP.json" > "$S/gt1-a2-warm-$STAMP.log" 2>&1
  say "warm cell rc=$? $(grep 'step rep' "$S/gt1-a2-warm-$STAMP.log" | tail -1 | cut -c1-230)"
  timeout 300 python3 w0/w0_drive.py step "$S/gt1-a2-step-$STAMP.json" --profile > "$S/gt1-a2-step-$STAMP.log" 2>&1
  say "profiled cell rc=$? $(grep 'step rep' "$S/gt1-a2-step-$STAMP.log" | tail -1 | cut -c1-230)"
  curl -s -m 20 -X POST "$BASE/stop_profile" -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do docker ps --format '{{.Names}}' | grep -qx ds41 || break; sleep 5; done
  docker ps --format '{{.Names}}' | grep -qx ds41 && { say "WARN: ds41 still up 5 min after stop_profile; stopping"; docker stop -t 60 ds41 >/dev/null 2>&1 || true; }
  docker logs ds41 2>&1 | grep -iE "nsys|profil|Generat|report" | tail -20 > "$S/gt1-a2-nsys-lines-$STAMP.log"
  rep=$(ls -t "$PROF"/*.nsys-rep 2>/dev/null | head -1)
  say "nsys report: ${rep:-MISSING} $( [ -n "$rep" ] && stat -c %s "$rep" ) bytes"
  if [ -n "$rep" ]; then
    timeout 1200 $NSYS export --type sqlite --force-overwrite=true -o "$S/gt1-a2-$STAMP.sqlite" "$rep" > "$S/gt1-a2-export-$STAMP.log" 2>&1
    say "sqlite $(stat -c %s "$S/gt1-a2-$STAMP.sqlite" 2>/dev/null) bytes"
    timeout 900 $NSYS stats --report cuda_gpu_kern_sum --format csv "$rep" > "$S/gt1-a2-kern-$STAMP.csv" 2> "$S/gt1-a2-stats-$STAMP.err"
    say "kern csv lines=$(wc -l < "$S/gt1-a2-kern-$STAMP.csv")"
    timeout 900 python3 w0/step_profile_analyze.py "$S/gt1-a2-$STAMP.sqlite" > "$S/gt1-a2-analyze-$STAMP.log" 2>&1
    say "analyze rc=$? $(head -3 "$S/gt1-a2-analyze-$STAMP.log" | tr '\n' ' ' | cut -c1-220)"
    G3=a2-profiled
  else
    G3=a2-no-report
  fi
  exit 0
fi

# ---------------------------------------------------------------------------------------------- GT3 (GT1_RES=1)
if [ "${GT1_RES:-0}" = 1 ]; then
  say "== card GT3: adaptive residency cells R0..R5 (control + telemetry $RESDIR)"
  R() { echo "$S/gt1-res-$1-$STAMP"; }
  drv() { local t=$1; shift; timeout "$t" python3 bench/gt_res_drive.py "$@"; }
  stepcell() {   # $1 = cell name: w0_drive.py step (2 warm generations + 3 reps of 512 greedy tokens, 1 stream)
    timeout 1800 python3 w0/w0_drive.py step "$(R "$1").json" > "$(R "$1").log" 2>&1
    say "$1 rc=$? $(grep 'step rep' "$(R "$1").log" | tail -1 | cut -c1-220)"
  }
  capture() {    # $1 = cell name: the greedy fixture capture (8 prompts x 256 tokens, top-20 logprobs)
    timeout 1800 python3 w0/w0_drive.py parity "$(R "$1").jsonl" > "$(R "$1").log" 2>&1
    say "$1 rc=$? $(tail -1 "$(R "$1").log" | cut -c1-160)"
  }
  ctl() {        # $1 = cell name, $2 = control document, $3 = nudge traffic, $4 = wait (ack|idle|boot), $5 = timeout s
    drv $(( ${5:-300} + 60 )) ctrl --dir "$RESDIR" --set "$2" --nudge "$3" --wait "$4" --timeout "${5:-300}" \
      --out "$(R "$1").json" > "$(R "$1").log" 2>&1
    local rc=$?; say "$1 rc=$rc $(tail -1 "$(R "$1").log" | cut -c1-400)"; return $rc
  }
  grep -E "gracetier: (residency|\[gracetier-res\]|stream check|GPUModelRunner\.execute_model wrapped)" \
    "$S/gt1-boot-$STAMP.log" | cut -c1-600 > "$(R boot).log"
  cat "$(R boot).log" | cut -c1-300
  grep -q "gracetier: residency: manager over" "$(R boot).log" \
    || { say "FAIL: the boot log has no residency manager line (gracetier-res.json not read?): R0..R5 skipped"; G3=res-not-armed; exit 0; }
  [ -f "$RESDIR/telemetry.jsonl" ] || { say "FAIL: no $RESDIR/telemetry.jsonl: R0..R5 skipped"; G3=res-no-telemetry; exit 0; }
  AUDIT='{"audit": {"moved": true, "random": 64}}'

  say "== R0 frozen at the boot map: step cell (3 reps) + the fixture capture R3 compares with"
  stepcell r0-step
  capture r0-parity

  say "== R1 unfreeze + baseline; C1 = the 8 fixture prompts x 256 tokens, looped 90 s"
  if alive; then
    drv 600 c1 --dir "$RESDIR" --seconds 90 --tokens 256 --out "$(R r1-c1).json" > "$(R r1-c1).log" 2>&1
    say "R1 rc=$? $(tail -1 "$(R r1-c1).log" | cut -c1-400)"
  else say "R1 skipped: server not alive"; fi

  say "== R2a freeze (the map is now fitted to the fixture); step cell on the held-out step prompt"
  if alive && ctl r2a-ctrl '{"freeze": true}' fixture idle; then stepcell r2a-step; else say "R2a skipped"; fi

  say "== R2b unfreeze; step cell x 5 (reported: the fifth cell's 3 reps)"
  if alive && ctl r2b-ctrl '{"freeze": false}' step ack; then
    for i in 1 2 3 4 5; do stepcell "r2b-step$i"; done
  else say "R2b skipped"; fi

  say "== R3 freeze; fixture capture twice (self-identical N/8; vs the R0-map capture N/8, reported, not gated)"
  if alive && ctl r3-ctrl '{"freeze": true}' step idle; then
    capture r3-parity-a
    capture r3-parity-b
    drv 120 compare --a "$(R r3-parity-a).jsonl" --b "$(R r3-parity-b).jsonl" --ref "$(R r0-parity).jsonl" \
      --out "$(R r3-compare).json" > "$(R r3-compare).log" 2>&1
    say "R3 $(tail -1 "$(R r3-compare).log" | cut -c1-300)"
    # the A3 instrument, for the record: column A2 = this map against itself, column cand = the R0-map capture
    python3 bench/parity.py --ref "$(R r3-parity-a).jsonl" --ref2 "$(R r3-parity-b).jsonl" --cand "$(R r0-parity).jsonl" \
      --json "$(R r3-parity).json" > "$(R r3-parity).log" 2>&1 || true
  else say "R3 skipped"; fi

  say "== R4 audit (every moved expert + 64 random per layer); restore_map; audit; step cell"
  if alive; then
    ctl r4-audit1 "$AUDIT" step ack
    if ctl r4-restore '{"freeze": true, "restore_map": true}' step boot 900; then
      ctl r4-audit2 "$AUDIT" step ack
      stepcell r4-step
    else say "R4: the boot map did not come back within the timeout: audit 2 and the step cell skipped"; fi
  else say "R4 skipped: server not alive"; fi

  say "== R5 unfreeze; C32 smoke (32 x >= 128 tokens); server alive"
  if alive && ctl r5-ctrl '{"freeze": false}' fixture ack; then
    timeout 1800 python3 bench/c32_smoke.py --base "$BASE" --model "$SERVED" --n 32 --tokens 128 \
      --json "$(R r5-c32).json" > "$(R r5-c32).log" 2>&1
    crc=$?
    tail -1 "$(R r5-c32).log"
    say "R5 c32 rc=$crc; container $(docker inspect -f '{{.State.Status}} restarts={{.RestartCount}}' ds41 2>/dev/null); alive $(alive && echo yes || echo NO)"
  else say "R5 skipped"; fi

  cp "$RESDIR/telemetry.jsonl" "$(R telemetry).jsonl" 2>/dev/null || say "WARN: no telemetry.jsonl to copy"
  docker logs ds41 2>&1 | grep -E "gracetier: (residency|\[gracetier-res\]|stream check)" | cut -c1-600 > "$(R server-lines).log" || true
  say "stream checks: $(grep -c 'stream check' "$(R server-lines).log") line(s) $(grep 'stream check' "$(R server-lines).log" | cut -c1-200 | tr '\n' ';')"
  PYTHONIOENCODING=utf-8 python3 bench/gt_res_drive.py summary --scratch "$S" --stamp "$STAMP" --out "$(R summary).json" \
    2>&1 | tee "$(R summary).log"
  G3=res-ran
  exit 0
fi

# ---------------------------------------------------------------------------------------------- G2
say "== G2 greedy fixture parity"
timeout 1800 python3 w0/w0_drive.py parity "$S/gt1-parity-$STAMP.jsonl" > "$S/gt1-g2-$STAMP.log" 2>&1
prc=$?
python3 bench/parity.py --ref w0/results/phase1/parity-A.jsonl --ref2 w0/results/phase1/parity-A2.jsonl \
  --cand "$S/gt1-parity-$STAMP.jsonl" --json "$S/gt1-g2-$STAMP.json" >> "$S/gt1-g2-$STAMP.log" 2>&1
python3 - "$S/gt1-g2-$STAMP.json" "$S/gt1-g2s-$STAMP.json" "$prc" <<'PY_G2'
import json, sys
src, out, prc = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = json.load(open(src))
except Exception as e:
    json.dump({"error": repr(e), "drive_rc": prc}, open(out, "w")); print(f"G2: no parity table ({e!r})"); sys.exit(1)
c = {t: v["cand"] for t, v in d.items() if v.get("cand")}
a2 = {t: v["a2"] for t, v in d.items() if v.get("a2")}
s = {"drive_rc": prc, "prompts": len(d), "identical": sum(1 for v in c.values() if v["first_divergence"] is None),
     "top1_flips": sum(v["top1_flips"] for v in c.values()),
     "top1_flips_weighted": round(sum(v["top1_flips_weighted"] for v in c.values()), 4),
     "max_abs_dlogprob": max((v["max_abs_dlogprob"] for v in c.values()), default=None),
     "first_divergence": {t: v["first_divergence"] for t, v in c.items()},
     "a2_first_divergence": {t: v["first_divergence"] for t, v in a2.items()},
     "a2_max_abs_dlogprob": max((v["max_abs_dlogprob"] for v in a2.values()), default=None),
     "note": "bench/parity.py (A3 instrument): flips are counted before the first divergence (docs/A3-DEFINITION.md); "
             "peer3 in the Pg61 control changes numerics, gracetier (b) does not -- an A3-class difference is expected"}
json.dump(s, open(out, "w"), indent=1)
print(f"G2: identical {s['identical']}/{s['prompts']}; flips {s['top1_flips']} weighted {s['top1_flips_weighted']}; "
      f"max|dlogp| {s['max_abs_dlogprob']} (A2 spread {s['a2_max_abs_dlogprob']}); first divergence {s['first_divergence']}")
PY_G2
[ $prc = 0 ] && G2=ran || G2="drive-rc$prc"
cat "$S/gt1-g2-$STAMP.log" | tail -14

# ---------------------------------------------------------------------------------------------- G3
if alive; then
  say "== G3 step cell (1 stream, 3 reps)"
  timeout 1800 python3 w0/w0_drive.py step "$S/gt1-step-$STAMP.json" > "$S/gt1-g3-$STAMP.log" 2>&1
  src=$?
  python3 - "$S/gt1-step-$STAMP.json" w0/results/phase1/step-pg61.json "$S/gt1-g3-$STAMP.json" <<'PY_G3'
import json, statistics, sys
cand, ref, out = sys.argv[1:4]
def cell(p):
    try:
        d = json.load(open(p))
    except Exception as e:
        return None, repr(e)
    reps = []
    for r in d.get("reps", []):
        sm, tps = r.get("step_ms_from_metrics"), r.get("tokens_per_step")
        reps.append({"rep": r.get("rep"), "step_ms": sm, "tokens_per_step": tps,
                     "ms_per_token": round(sm / tps, 3) if sm and tps else None, "steps": r.get("steps"),
                     "ttft_s": r.get("ttft_s"), "decode_wall_s": r.get("decode_wall_s"),
                     "decode_wall_ms_per_chunk": round(1000 * r["decode_wall_s"] / r["chunks"], 3)
                     if r.get("decode_wall_s") and r.get("chunks") else None})
    return reps, None
c, ce = cell(cand)
p, pe = cell(ref)
med = lambda rs, k: statistics.median([r[k] for r in rs if r.get(k) is not None]) if rs and any(r.get(k) is not None for r in rs) else None
s = {"reps": len(c or []), "step_ms_median": med(c, "step_ms"), "ms_per_token_median": med(c, "ms_per_token"),
     "tokens_per_step_median": med(c, "tokens_per_step"), "pg61_step_ms_median": med(p, "step_ms"),
     "pg61_ms_per_token_median": med(p, "ms_per_token"), "pg61_receipt": ref,
     "framings": "ms/step = decode wall / (spec drafts - 1) from /metrics; ms/token = ms/step / tokens per step"}
json.dump({"summary": s, "gracetier": c, "pg61_phase1": p, "errors": [ce, pe]}, open(out, "w"), indent=1)
print(f"G3: gracetier ms/step {s['step_ms_median']} ms/token {s['ms_per_token_median']} tok/step "
      f"{s['tokens_per_step_median']} ({s['reps']} reps) | Pg61 phase-1 receipt {s['pg61_step_ms_median']} / "
      f"{s['pg61_ms_per_token_median']}")
PY_G3
  [ $src = 0 ] && G3=ran || G3="step-rc$src"
else
  say "G3 skipped: server not alive"; G3=dead
fi

# ---------------------------------------------------------------------------------------------- G4
if alive; then
  say "== G4 C32 smoke (32 x >= 128 tokens)"
  timeout 1800 python3 bench/c32_smoke.py --base "$BASE" --model "$SERVED" --n 32 --tokens 128 \
    --json "$S/gt1-g4-$STAMP.json" > "$S/gt1-g4-$STAMP.log" 2>&1
  crc=$?
  tail -1 "$S/gt1-g4-$STAMP.log"
  if [ $crc = 0 ] && alive; then G4=pass; else G4=FAIL; fi
  say "G4 $G4 (c32 rc=$crc; container $(docker inspect -f '{{.State.Status}} restarts={{.RestartCount}}' ds41 2>/dev/null))"
else
  say "G4 skipped: server not alive"; G4=dead
fi

# ---------------------------------------------------------------------------------------------- G5
if alive; then
  say "== G5 prefill / TTFT (8 fixture prompts, salted, 2 reps)"
  timeout 1200 python3 - "$S/gt1-g5-$STAMP.json" "$STAMP" <<'PY_G5' > "$S/gt1-g5-$STAMP.log" 2>&1
import json, statistics, sys, time
sys.path.insert(0, "w0")
import w0_drive as W
out, stamp = sys.argv[1:3]
rows = []
for rep in range(2):
    for i, (tag, prompt) in enumerate(W.FIXTURE):
        p = f"(gt1 ttft {stamp} rep {rep} #{i}) {prompt}"   # unique prefix: the prefix cache cannot serve it
        t0 = time.time()
        try:
            t_first, t_last, n, usage, _ = W.one_stream(p, 16)
            rows.append({"rep": rep, "tag": tag, "ttft_ms": round(1000 * ((t_first or time.time()) - t0), 1),
                         "prompt_tokens": (usage or {}).get("prompt_tokens"), "chunks": n})
        except Exception as e:  # noqa: BLE001
            rows.append({"rep": rep, "tag": tag, "error": repr(e)[:200]})
        print(json.dumps(rows[-1]), flush=True)
ok = [r["ttft_ms"] for r in rows if "ttft_ms" in r]
s = {"requests": len(rows), "ok": len(ok), "ttft_ms_mean": round(statistics.mean(ok), 1) if ok else None,
     "ttft_ms_median": round(statistics.median(ok), 1) if ok else None, "ttft_ms_max": max(ok) if ok else None,
     "prompt_tokens": sorted({r.get("prompt_tokens") for r in rows if r.get("prompt_tokens")}),
     "how": "chat stream, max_tokens 16, TTFT = first content chunk - request start (client side, includes HTTP)"}
json.dump({"summary": s, "rows": rows}, open(out, "w"), indent=1)
print(f"G5: TTFT mean {s['ttft_ms_mean']} ms median {s['ttft_ms_median']} max {s['ttft_ms_max']} over {s['ok']}/{s['requests']}")
PY_G5
  g5rc=$?
  tail -1 "$S/gt1-g5-$STAMP.log"
  [ $g5rc = 0 ] && G5=ran || G5="rc$g5rc"
else
  say "G5 skipped: server not alive"; G5=dead
fi
exit 0
