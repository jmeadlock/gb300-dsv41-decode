#!/bin/bash
# Card A5 cells, identical for every arm (Milo, 2026-11-09). Env: A5_OUT (file prefix), A5_ARM (tag), W0_BASE, W0_MODEL,
# A5_WARM=1 (two unmeasured passes first: lets an adaptive-residency arm fit, and warms every arm the same way),
# A5_GAUNTLET=1 (tool gauntlet, pin-hot-experts/round7). Cells: C1 = mixed pass A + pass B (20 prompts x 256 tokens each),
# C8 = 40 salted requests on 8 streams, C32 = 64 salted requests on 32 streams. Prints one line per cell; never exits non-zero
# for a failed cell (the line says so).
set -u
: "${A5_OUT:?}" "${A5_ARM:?}"
DEC=$(cd "$(dirname "$0")/../.." && pwd); cd "$DEC" || exit 9
say() { echo "$(date '+%F %T %Z') [a5 $A5_ARM] $*"; }
run() { local name=$1; shift; timeout 1800 python3 bench/gt_mix_drive.py "$@" --out "$A5_OUT-$name.json" > "$A5_OUT-$name.log" 2>&1
        say "$name rc=$? $(tail -1 "$A5_OUT-$name.log" | cut -c1-300)"; }
if [ "${A5_SKIP_SPEED:-0}" != 1 ]; then
if [ "${A5_WARM:-0}" = 1 ]; then run warm-a pass --set A; run warm-b pass --set B; fi
run c1-a pass --set A
run c1-b pass --set B
run c8 conc --n 8 --tag c8
run c32 conc --n 32 --tag c32
run c1-a2 pass --set A
fi
# A5_TTFT=1: time to first token by prompt length (max_tokens 1, 3 runs each, a new salt per run: never a prefix hit)
if [ "${A5_TTFT:-0}" = 1 ]; then run ttft ttft --sizes 32,300,1000,8000,64000 --reps 3; fi
if [ "${A5_GAUNTLET:-0}" = 1 ]; then
  GDIR=$HOME/pin-hot-experts/round7/gauntlet; GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
  rm -rf "$A5_OUT-gauntlet"   # run_suite.py creates its output dir itself and refuses an existing one
  ( cd "$GDIR" && timeout 1500 python3 run_suite.py --base-url "$W0_BASE/v1" --model "$W0_MODEL" --output "$A5_OUT-gauntlet" \
      --tag "a5-$A5_ARM" --cache-salt "a5-$A5_ARM" --sandbox-image "$GIMG" --workers 4 ) > "$A5_OUT-gauntlet.log" 2>&1
  say "gauntlet rc=$? $(tail -2 "$A5_OUT-gauntlet.log" | tr '\n' ' ' | cut -c1-300)"
fi
say "cells done"
