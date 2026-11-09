#!/bin/bash
# Window gt-mt-1108: Pg61 down -> gracetier mini-test (M0..M7) -> Pg61 restore. James "go" 2026-11-08 16:5x CST.
set -u
GR=/home/milo/dsv41/garner
LO1=/home/milo/dsv41/m3/LANE-OWNER.txt; LO2=/home/milo/dsv41/m3/logs/LANE-OWNER.txt
MF=/home/milo/dsv41/m3/logs/MAINTENANCE
LOG=/home/milo/dsv41/decode/scratch/window-gt-mt-1108-$(date +%Y%m%d-%H%M).log
ts() { date '+%F %T %Z'; }
note() { echo "- $(ts) $*" | tee -a "$LO1" >> "$LO2"; }
exec > >(tee -a "$LOG") 2>&1
echo "== window gt-mt-1108 start $(ts)"
note "WINDOW gt-mt-1108 START (Milo, James go): Pg61 down for the gracetier mini-test (~1 h), then restore. log $LOG"
echo "gracetier mini-test window gt-mt-1108 $(ts) -- no restore until this runner's END line or James says" > "$MF"

cd "$GR" && WORK=$GR/pg61-work DS41_HOME=$GR/home bash launcher/launch-pg61.sh down; echo "down rc=$?"
sleep 5; docker ps --format '{{.Names}}'; nvidia-smi --query-gpu=memory.used --format=csv,noheader

cd /home/milo/dsv41/decode && git pull -q && git log --oneline -1
GRACETIER_WINDOW=gt-mt-1108 PG61_FASTPATH=$GR/src-git/deepseek-v4.1-flash/fastpath \
  timeout 5400 bash bench/box/gracetier_minitest.sh --layer 10 --json scratch/gracetier-minitest-L10-gt-mt-1108.json
echo "minitest rc=$?  $(ts)"
docker ps -q | xargs -r docker stop >/dev/null 2>&1

echo "== restore $(ts)"
cd $GR/promote && cp -n restore-pg61-after-decode1108.sh restore-pg61-after-gt-mt-1108.sh && \
  python3 - <<'PY'
p='/home/milo/dsv41/garner/promote/restore-pg61-after-gt-mt-1108.sh'; s=open(p).read()
s=s.replace('decode1108','gt-mt-1108').replace('the decode-project windows (11/06-11/08)','the gracetier mini-test window gt-mt-1108')
open(p,'w').write(s)
PY
bash -n restore-pg61-after-gt-mt-1108.sh && echo "diff hunks: $(diff restore-pg61-after-decode1108.sh restore-pg61-after-gt-mt-1108.sh | grep -c '^[<>]')"
bash restore-pg61-after-gt-mt-1108.sh; rc=$?
echo "restore rc=$rc $(ts)"
if [ $rc -eq 0 ]; then mv "$MF" "$MF.stale-gt-mt-1108"; fi
note "WINDOW gt-mt-1108 END (Milo): mini-test done; restore rc=$rc; receipts ~/dsv41/decode/scratch/gracetier-*gt-mt-1108*; log $LOG"
echo "== END $(ts)"
