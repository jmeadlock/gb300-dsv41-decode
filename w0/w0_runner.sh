#!/usr/bin/env bash
# w0_runner.sh — W0 measurement window for the GB300 DSV4.1 Megakernel project (Milo, 2026-11-06).
#   Phase 1 (Pg61 as-is, consumers already moved off :30006): parity capture ×2 (A, A2 → Pg61's own spread),
#            clean C1 step cell ×3 from /metrics spec counters.
#   Phase 2: Pg61 down (production launcher) → build + boot the nsys-wrapped clone (launch-w0.sh, patch 06) →
#            warm → /start_profile … 512-token greedy generation … /stop_profile → nsys exits the server on purpose →
#            nsys stats + sqlite export → step_profile_analyze.py + w0_gaps.py.
#   Phase 3: launch-w0.sh down. Lane stays DARK by charter (Station = dev box). No restore.
# Launch: ( cd ~/dsv41/w0 && setsid nohup bash w0_runner.sh > /dev/null 2>&1 < /dev/null & )
set -uo pipefail
W=$HOME/dsv41/w0; GR=$HOME/dsv41/garner
R=$W/results-$(date +%Y%m%d-%H%M); mkdir -p "$R"/{phase1,phase2}
OWNER=$HOME/dsv41/m3/logs/LANE-OWNER.txt
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai:v0.31.0
NSYS=/usr/local/bin/nsys
PROF=$GR/home/state/prof/W0
exec 3>>"$R/runner.log"
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ echo "- $(date '+%F %T %Z') $*" >> "$OWNER"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
bound(){ curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; }
kill_wd(){ local p; for p in $(pgrep -f "[s]idecar_watchdog_p3.py --server ds41" || true); do kill "$p" 2>/dev/null || true; done; }
w0_down(){ ( cd "$W" && WORK=$W/work DS41_HOME=$GR/home bash launch-w0.sh down ) >> "$R/phase2/down.log" 2>&1 || true; }
finish(){ # every exit path: make sure nothing is left serving, say so, END
  kill_wd; w0_down
  docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && log "WARN: ds41 containers still up at END"
  own "W0 END: lane DARK by charter (megakernel dev box). Receipts $R. No restore until James says."
  log "===== W0 END (lane = DARK, by charter) ====="
}
echo $$ > "$W/w0_runner.pid"
log "===== W0 START pid $$ R=$R ====="

# ---------- preflight
SKIP_P1=${SKIP_P1:-0}
if [ "$SKIP_P1" = 1 ]; then
  docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && { log "ABORT: SKIP_P1=1 but ds41 containers are up"; exit 5; }
  log "SKIP_P1=1: phase 1 receipts taken in an earlier run; Pg61 already down"
else
  bound || { log "ABORT: Pg61 not bound on :30006 (phase 1 needs it)"; exit 5; }
  docker ps --format '{{.Names}}' | grep -qx ds41 || { log "ABORT: ds41 not running"; exit 5; }
fi
[ -f "$HOME/dsv41/m3/logs/MAINTENANCE" ] || log "NOTE: no MAINTENANCE flag (expected one from the cutover)"
other=$(pgrep -af "card[A-Za-z0-9]*_runner.sh|restore-pg61|rollback-pg61" | grep -v "^$$ " | grep -v w0_runner || true)
[ -z "$other" ] || { log "ABORT: other runner present: $other"; exit 5; }
free_gib=$(df -Pk / | awk 'NR==2{print int($4/1048576)}'); [ "$free_gib" -ge 300 ] || { log "ABORT: root free ${free_gib} GiB < 300"; exit 5; }
[ -x "$NSYS" ] || { log "ABORT: nsys missing"; exit 5; }
[ -d /opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8 ] || { log "ABORT: nsys target dir missing"; exit 5; }
for f in w0_drive.py w0_gaps.py step_profile_analyze.py launch-w0.sh patches/06-nsys-wrap-serve.patch; do [ -f "$W/$f" ] || { log "ABORT: missing $W/$f"; exit 5; }; done
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$W/w0_drive.py" "$W/w0_gaps.py" "$W/step_profile_analyze.py" || { log "ABORT: driver syntax"; exit 5; }
bash -n "$W/launch-w0.sh" || { log "ABORT: launch-w0.sh syntax"; exit 5; }
{ echo "nsys $($NSYS --version | head -1)"; for f in "$0" "$W/w0_drive.py" "$W/w0_gaps.py" "$W/step_profile_analyze.py" "$W/launch-w0.sh" "$W/patches/06-nsys-wrap-serve.patch"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done
  docker inspect -f '{{json .Args}}' ds41; docker inspect -f '{{json .Config.Env}}' ds41 | tr ',' '\n' | grep -v API_KEY | tr '\n' ' '; echo; } > "$R/instruments.txt"
ulimit -c 0
log "preflight OK; root ${free_gib} GiB free; Pg61 bound"
own "W0 runner pid $$ START: phase 1 read-only on Pg61 (parity ×2, step ×3), then Pg61 DOWN for one nsys-wrapped boot of the same config; lane stays DARK after. Receipts $R"
trap 'log "TERM received"; finish; exit 143' TERM
trap 'log "INT received"; finish; exit 130' INT

# ---------- phase 1: Pg61 as-is
cd "$W"
if [ "$SKIP_P1" != 1 ]; then
for arm in A A2; do
  timeout 1800 python3 w0_drive.py parity "$R/phase1/parity-$arm.jsonl" > "$R/phase1/parity-$arm.log" 2>&1; log "P1 parity $arm rc=$? $(tail -1 "$R/phase1/parity-$arm.log")"
done
timeout 1800 python3 w0_drive.py step "$R/phase1/step-pg61.json" > "$R/phase1/step-pg61.log" 2>&1; log "P1 step rc=$? $(grep -c 'step rep' "$R/phase1/step-pg61.log") reps"
python3 - "$R/phase1/step-pg61.json" >> "$R/runner.log" 2>&1 <<'PY'
import json,sys; d=json.load(open(sys.argv[1]))
for r in d["reps"]: print("  P1 rep%d step_ms=%s tok/step=%s steps=%s ttft=%s" % (r["rep"], r.get("step_ms_from_metrics"), r.get("tokens_per_step"), r.get("steps"), r.get("ttft_s")))
PY
python3 - "$R/phase1/parity-A.jsonl" "$R/phase1/parity-A2.jsonl" >> "$R/runner.log" 2>&1 <<'PY'
import json,sys
A=[json.loads(l) for l in open(sys.argv[1])]; B=[json.loads(l) for l in open(sys.argv[2])]
for a,b in zip(A,B):
    ta=[c["token"] for c in a["logprobs"] or []]; tb=[c["token"] for c in b["logprobs"] or []]
    n=min(len(ta),len(tb)); first=next((i for i in range(n) if ta[i]!=tb[i]), None)
    print("  P1 parity A vs A2 %-6s len %d/%d first_divergence=%s" % (a["tag"], len(ta), len(tb), first))
PY
fi

# ---------- phase 2: Pg61 down, nsys-wrapped boot
if [ "$SKIP_P1" != 1 ]; then
t_down0=$(date +%s)
( cd "$GR" && WORK=$GR/pg61-work DS41_HOME=$GR/home bash launcher/launch-pg61.sh down ) > "$R/phase2/pg61-down.log" 2>&1; log "P2 pg61 down rc=$? ($(( $(date +%s) - t_down0 )) s)"
docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && { log "ABORT: ds41 still up after down"; finish; exit 6; }
own "W0: Pg61 stopped (launch-pg61.sh down) for the nsys-wrapped boot; restore = launch-pg61.sh up (~6 min) ONLY on James's word"
fi
drop_caches; sleep 20
if [ ! -f "$W/work/policy.env" ]; then
  t_b0=$(date +%s)
  ( cd "$W" && DS41_NSYS_WRAP=1 DS41_NSYS_TAG=W0 GARNER_SRC=$GR/src-git WORK=$W/work PATCHES=$W/patches bash launch-w0.sh build ) > "$R/phase2/build.log" 2>&1; rc=$?
  log "P2 build rc=$rc ($(( $(date +%s) - t_b0 )) s) $(tail -1 "$R/phase2/build.log" | cut -c1-160)"
  [ $rc -eq 0 ] || { finish; exit 7; }
fi
grep -q '^DS41_NSYS_WRAP=1$' "$W/work/policy.env" || { log "ABORT: policy.env lacks DS41_NSYS_WRAP=1"; finish; exit 7; }
# the frozen-hash check in verify compares libw6mk against the W0 tree (same nvcc, same source → same cubin as Pg61; record both)
echo "libw6mk pg61 $(sha256sum "$GR/pg61-work/tree/fastpath/libw6mk.so" | cut -c1-16) w0 $(sha256sum "$W/work/tree/fastpath/libw6mk.so" | cut -c1-16)" >> "$R/instruments.txt"
mkdir -p "$PROF"; rm -f "$PROF"/W0.nsys-rep "$PROF"/W0.sqlite
t_up0=$(date +%s)
( cd "$W" && WORK=$W/work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash launch-w0.sh up ) > "$R/phase2/up.log" 2>&1; rc=$?
log "P2 up rc=$rc ($(( $(date +%s) - t_up0 )) s) $(tail -1 "$R/phase2/up.log" | cut -c1-200)"
[ $rc -eq 0 ] && bound || { log "ABORT: nsys-wrapped boot not bound"; docker logs ds41 2>&1 | tail -40 > "$R/phase2/ds41-tail.log"; finish; exit 8; }
docker inspect -f '{{json .Args}}' ds41 > "$R/phase2/ds41-args.json"
grep -q '/nsys","profile"' "$R/phase2/ds41-args.json" || { log "ABORT: ds41 args carry no nsys wrap"; finish; exit 8; }
grep -q 'profiler-config.profiler=cuda' "$R/phase2/ds41-args.json" || { log "ABORT: ds41 args carry no --profiler-config.profiler=cuda"; finish; exit 8; }
kill_wd; log "P2 watchdog stopped for the capture (nsys ends the server on /stop_profile by design)"
docker update --restart no ds41 ds41-sidecar >/dev/null 2>&1 || true
timeout 1800 python3 w0_drive.py step "$R/phase2/step-nsys.json" --profile > "$R/phase2/step-nsys.log" 2>&1; log "P2 profiled step rc=$? $(grep 'step rep' "$R/phase2/step-nsys.log" | cut -c1-220)"
# wait for nsys to finalize: ds41 exits
for i in $(seq 1 60); do docker ps --format '{{.Names}}' | grep -qx ds41 || break; sleep 5; done
docker ps --format '{{.Names}}' | grep -qx ds41 && { log "WARN: ds41 still running 5 min after stop_profile; stopping it"; docker stop -t 30 ds41 >/dev/null 2>&1 || true; }
docker logs ds41 2>&1 | grep -iE "nsys|profil|Generating|report" | tail -20 > "$R/phase2/ds41-nsys-lines.log"
rep=$(ls -t "$PROF"/*.nsys-rep 2>/dev/null | head -1); log "P2 nsys report: ${rep:-MISSING} $( [ -n "$rep" ] && stat -c %s "$rep" ) bytes"
if [ -n "$rep" ]; then
  timeout 900 $NSYS stats --report cuda_gpu_kern_sum --format csv "$rep" > "$R/phase2/nsys-kern.csv" 2> "$R/phase2/nsys-stats.err"; log "P2 kern csv lines=$(wc -l < "$R/phase2/nsys-kern.csv")"
  timeout 900 $NSYS export --type sqlite --force-overwrite=true -o "$PROF/W0.sqlite" "$rep" > "$R/phase2/nsys-export.log" 2>&1; log "P2 sqlite $(stat -c %s "$PROF/W0.sqlite" 2>/dev/null) bytes"
  timeout 900 python3 step_profile_analyze.py "$PROF/W0.sqlite" > "$R/phase2/step-profile.txt" 2>&1; log "P2 analyze rc=$? $(grep -m1 -iE 'step|wall' "$R/phase2/step-profile.txt" | cut -c1-160)"
  timeout 600 python3 w0_gaps.py "$PROF/W0.sqlite" --out "$R/phase2/gaps.json" > "$R/phase2/gaps.txt" 2>&1; log "P2 gaps rc=$? $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print("kernels",d["kernels"],"busy_frac",d["busy_frac"],"idle_ms",d["idle_ms"],"pos_gap_sum_us",d["serial_gap_pos_sum_us"])' "$R/phase2/gaps.json" 2>&1)"
  cp "$rep" "$R/phase2/" 2>/dev/null || true
fi
finish
exit 0
