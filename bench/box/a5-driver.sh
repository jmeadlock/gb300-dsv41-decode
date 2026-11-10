#!/bin/bash
# Card A5 driver (Milo, 2026-11-09): same-harness three-way on the mixed sets at C1 / C8 / C32.
#   1. gracetier (card-gt1.sh GT1_RES_CELLS=a5; H 255, adaptive residency, hyst 3.0) -> Pg61 restored by the card
#   2. v28-prob (a5-v28.sh) -> Pg61 restored by the arm
#   3. Pg61 (the live lane, no window)
# Each next step runs only if Pg61 is serving again. Usage: A5_STAMP=<stamp> bash bench/box/a5-driver.sh
set -u
: "${A5_STAMP:?}"
DEC=/home/milo/dsv41/decode; S=$DEC/scratch; cd "$DEC" || exit 9
say() { echo "$(date '+%F %T %Z') [a5] $*"; }
serving() { curl -sf -m 5 localhost:30006/v1/models | grep -q dsv41-flash-uva && [ "$(docker ps --format '{{.Names}}' | sort | tr '\n' ' ')" = "ds41 ds41-sidecar " ]; }
serving || { say "REFUSED: Pg61 not serving at start"; exit 2; }
say "== 1/3 gracetier arm"
GT1_WINDOW=a5-$A5_STAMP GT1_RES=1 GT1_RES_CELLS=a5 GT1_SEED_CACHE=1 \
  GT1_RES_CONFIG='{"mode": "adaptive", "freeze": false, "hyst": 3.0}' bash bench/box/card-gt1.sh > "$S/a5-card-gracetier-$A5_STAMP.log" 2>&1
say "gracetier card rc=$?; $(grep -E "restore rc" "$S/a5-card-gracetier-$A5_STAMP.log" | tail -1 | cut -c1-160)"
grep -E "\[a5 gracetier\]|FAIL|REFUSED|a5-audit|G1:" "$S/a5-card-gracetier-$A5_STAMP.log" | cut -c1-330
sleep 60
if serving; then
  say "== 2/3 v28-prob arm"
  bash bench/box/a5-v28.sh > "$S/a5-v28-runner-$A5_STAMP.log" 2>&1
  say "v28 arm rc=$?"
  grep -E "\[a5 v28\]|\[a5-v28\]" "$S/a5-v28-runner-$A5_STAMP.log" | cut -c1-330
  sleep 60
else say "SKIP v28 arm: Pg61 not serving after the gracetier card"; fi
if serving; then
  say "== 3/3 Pg61 arm (live lane)"
  W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva A5_OUT="$S/a5-pg61-$A5_STAMP" A5_ARM=pg61 A5_WARM=1 A5_GAUNTLET=1 bash bench/box/a5-cells.sh
else say "SKIP Pg61 arm: Pg61 NOT SERVING -- lane dark, needs attention"; fi
say "END; $(docker ps --format '{{.Names}} {{.Status}}' | tr '\n' ';') serving=$(serving && echo yes || echo NO)"
