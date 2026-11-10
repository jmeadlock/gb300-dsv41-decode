#!/bin/bash
# Card A5, v28-prob arm (Milo, 2026-11-09): the October single-GPU recipe (container dsv41-vllm-1017-v28-prob-BOUND-REF,
# stopped-and-kept, started unchanged with docker start) on the A5 cells, then Pg61 restored. Refuses unless Pg61 is
# serving and nothing else runs. Every exit path after Pg61 is down stops v28 and restores Pg61.
set -u
: "${A5_STAMP:?}"
GR=/home/milo/dsv41/garner; DEC=/home/milo/dsv41/decode; S=$DEC/scratch
V28=dsv41-vllm-1017-v28-prob-BOUND-REF
LO=/home/milo/dsv41/m3/LANE-OWNER.txt; MF=/home/milo/dsv41/m3/logs/MAINTENANCE
say() { echo "$(date '+%F %T %Z') [a5-v28] $*"; }
[ "$(docker ps --format '{{.Names}}' | sort | tr '\n' ' ')" = "ds41 ds41-sidecar " ] || { say "REFUSED: containers are not exactly Pg61's"; exit 2; }
curl -sf -m 5 localhost:30006/v1/models | grep -q dsv41-flash-uva || { say "REFUSED: Pg61 not serving"; exit 2; }
docker inspect "$V28" >/dev/null 2>&1 || { say "REFUSED: $V28 missing"; exit 2; }
pgrep -af "[c]ard-gt1|[r]estore-pg61|[r]ollback-pg61" >/dev/null && { say "REFUSED: another runner is active"; exit 2; }
[ "$(df -Pk / | awk 'NR==2{print int($4/1048576)}')" -ge 200 ] || { say "REFUSED: < 200 GiB free on /"; exit 2; }
DOWN=0; DONE=0
restore() {
  [ "$DONE" = 1 ] && return; DONE=1
  [ "$DOWN" = 1 ] || return
  docker logs "$V28" --since "$T0" > "$S/a5-v28-server-$A5_STAMP.log" 2>&1 || true
  docker stop -t 60 "$V28" >/dev/null 2>&1; sleep 10
  docker ps --format '{{.Names}}' | grep -q . && { say "FAIL: containers still running, not restoring: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; echo "- $(date '+%F %T %Z') A5 v28 arm: restore NOT run, containers still up; lane DARK, needs hands" >> $LO; return; }
  say "== restore Pg61"
  bash "$GR/promote/restore-pg61-after-gt1.sh"; rrc=$?
  say "restore rc=$rrc; $(docker ps --format '{{.Names}} {{.Status}}' | tr '\n' ';')"
  if [ "$rrc" = 0 ]; then mv "$MF" "$MF.stale-a5-v28-$A5_STAMP" 2>/dev/null; fi
  echo "- $(date '+%F %T %Z') CARD A5 v28 arm END (Milo, $A5_STAMP): restore rc=$rrc; receipts $S/a5-v28-$A5_STAMP-*" >> $LO
}
trap restore EXIT
trap 'say "signal"; exit 1' INT TERM HUP
echo "A5 v28-prob arm $A5_STAMP (Milo; James: publish recipe overnight); Pg61 comes back at the end" > "$MF"
echo "- $(date '+%F %T %Z') CARD A5 v28 arm START (Milo, $A5_STAMP): Pg61 down -> docker start $V28 -> cells -> Pg61 restore" >> $LO
say "== Pg61 down"
( cd "$GR" && WORK=$GR/pg61-work DS41_HOME=$GR/home bash launcher/launch-pg61.sh down ) > "$S/a5-v28-pg61-down-$A5_STAMP.log" 2>&1
say "Pg61 down rc=$? ($(tail -1 "$S/a5-v28-pg61-down-$A5_STAMP.log" | cut -c1-140))"
DOWN=1; sleep 5
docker ps --format '{{.Names}}' | grep -q . && { say "FAIL: containers still up after Pg61 down"; exit 1; }
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
free -g | sed -n 2p
T0=$(date -u +%FT%TZ)
say "== docker start $V28"
docker start "$V28" >/dev/null || { say "FAIL: docker start"; exit 1; }
ok=0
for i in $(seq 1 180); do   # up to 45 min (a cold autotune would be longer: then the arm is skipped)
  sleep 15
  [ "$(docker inspect -f '{{.State.Running}}' "$V28" 2>/dev/null)" = true ] || { say "FAIL: $V28 exited during boot: $(docker logs --since "$T0" "$V28" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-400)"; exit 1; }
  if curl -sf -m 5 localhost:30006/v1/models | grep -q dsv41-flash-uva; then ok=1; break; fi
done
[ "$ok" = 1 ] || { say "FAIL: not bound after 45 min"; exit 1; }
say "bound after $((i*15)) s; $(docker logs --since "$T0" "$V28" 2>&1 | grep -E "GPU KV cache size" | tail -1 | cut -c40-200)"
export W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva
curl -s -m 120 localhost:30006/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"thinking":false}}' | cut -c1-200; echo
A5_OUT="$S/a5-v28-$A5_STAMP" A5_ARM=v28 A5_WARM=1 A5_GAUNTLET=0 bash "$DEC/bench/box/a5-cells.sh"
say "server alive after cells: $(curl -sf -m 5 localhost:30006/v1/models | grep -q dsv41-flash-uva && echo yes || echo NO)"
exit 0
