#!/usr/bin/env bash
# launch-pg61.sh — DeepSeek-V4.1-Flash on DGX Station GB300 + RTX PRO 6000, built on Garner McCloud's recipe.
#
#   Upstream (the engine of this lane): https://github.com/garnermccloud/dgx-station-recipes  (deepseek-v4.1-flash/)
#   pinned at 8043a6811a5feda70fdfb6e827eebb2ec25c65d1, image vllm/vllm-openai:v0.31.0, launched by his serve/serve.sh.
#   Everything that makes this lane fast is his: the 29-component DS41_FASTPATH, peer tier v3, the MegaMoE fork,
#   the rowmap, the overlays. This script only applies a small policy + correctness patch set on top:
#     01/02  six policy hooks in serve.sh / recipe.env, each a no-op when its variable is unset
#            (no API key, attention-config, generation override, reasoning-effort default, component list, PEER3_SPINS)
#     03-05  vLLM #61161 spec_step_seed in his fused draft samplers (w5_markov, w5_markov_gemv, w6mk.cu -> libw6mk.so)
#     overlay-vllm  #61161 gumbel.py + rejection_sampler_utils.py with the #60945 guard, layered on his 58132 overlay
#     sidecar_watchdog_p3.py  fail-closed watchdog (a peer-tier-v3 timeout skips cold-expert adds silently otherwise)
#   Policy: stock weights, fp8_ds_mla KV, 1M context, 48 GB host KV, no sampling override, 24 of his 29 components
#   (drops ds41_sampling, ds41_stop, w8oproj, lowb_aux, tail_drafthead_mxfp8 — the last three change numerics).
#
# Usage:
#   GARNER_SRC=<checkout> WORK=<dir> launch-pg61.sh build      # tree + overlay + rebuilt libw6mk.so (CPU only)
#   WORK=<dir> MODEL_DIR=<weights> DS41_HOME=<state> launch-pg61.sh up     # serve.sh start + hashes + watchdog
#   WORK=<dir> DS41_HOME=<state> launch-pg61.sh down | verify | render
# Env (defaults): PORT=30006 SERVED=dsv41-flash-uva ALIASES=deepseek-v4.1-flash IMAGE=vllm/vllm-openai:v0.31.0
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PATCHES=${PATCHES:-$HERE/patches}
PIN=8043a6811a5feda70fdfb6e827eebb2ec25c65d1
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}
PORT=${PORT:-30006}; SERVED=${SERVED:-dsv41-flash-uva}; ALIASES=${ALIASES:-deepseek-v4.1-flash}
WORK=${WORK:?set WORK}
TREE=$WORK/tree; OVL=$WORK/overlay-vllm
COMPONENTS=tail_markov_fused,tail_swa_meta,tail_dflash_prep,tail_ctxkv,tail_replay_skip,tail_embed,attn,preattn,preattn_cand,preattn_index,preattn_compress,tail_draft_embed,boundary,suffix,topp_fast,tail_proj_side,engram_side,fastload,coldskip,peer3,megamoe,quant_producer,a12_retire,a12_memory_guard
WANT_GUMBEL=6eac3e1348339d157ba0e2184114859aca207df1
WANT_RSU=edc8f8f4c425f51a962840fc2e0e7aa32059f57a
VP=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu
log(){ printf '%s [pg61] %s\n' "$(date '+%F %T %Z')" "$*"; }
die(){ log "FAIL: $*"; exit 1; }

policy_env(){ cat <<EOF
DS41_PORT=$PORT
DS41_SERVED=$SERVED
DS41_ALIASES=$ALIASES
DS41_GPU_MODE=cdi
DS41_NO_KEY=1
DS41_MAXLEN=1048576
DS41_KV=fp8_ds_mla
DS41_OFFLOAD_GB=48
DS41_EFFORT_DEFAULT=0
DS41_FASTPATH_COMPONENTS_OVERRIDE=$COMPONENTS
DS41_OVERLAY_58132_DIR=$OVL
DS41_PEER3_SPINS=1000000000000
EOF
}

cmd_build(){
  local src=${GARNER_SRC:?set GARNER_SRC to a checkout of garnermccloud/dgx-station-recipes}
  [ "$(git -C "$src" rev-parse HEAD)" = "$PIN" ] || die "GARNER_SRC is not at $PIN"
  [ -e "$TREE" ] && [ "${FORCE:-0}" != 1 ] && die "$TREE exists (FORCE=1 to rebuild)"
  rm -rf "$TREE" "$OVL"; mkdir -p "$WORK"
  cp -a "$src/deepseek-v4.1-flash" "$TREE"
  local p; for p in "$PATCHES"/0*.patch; do patch -d "$TREE" -p1 --forward --no-backup-if-mismatch < "$p" >/dev/null || die "patch $(basename "$p")"; log "applied $(basename "$p")"; done
  log "building libw6mk.so (nvcc sm_103a in $IMAGE, CPU only)"
  docker run --rm --network none -v "$TREE/kernels/w6-moe/markov:/w" -w /w --entrypoint bash "$IMAGE" -c \
    "nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared w6mk.cu -o /w/libw6mk.so 2>/dev/null"
  mv "$TREE/kernels/w6-moe/markov/libw6mk.so" "$TREE/fastpath/libw6mk.so"
  cp -a "$TREE/docker/overlay-58132/vllm" "$OVL"
  cp -a "$PATCHES/overlay-vllm/." "$OVL/"
  [ "$(sha1sum < "$OVL/v1/worker/gpu/sample/gumbel.py" | cut -c1-40)" = $WANT_GUMBEL ] || die "overlay gumbel.py hash"
  [ "$(sha1sum < "$OVL/v1/worker/gpu/spec_decode/rejection_sampler_utils.py" | cut -c1-40)" = $WANT_RSU ] || die "overlay rsu hash"
  cp "$PATCHES/sidecar_watchdog_p3.py" "$WORK/"
  policy_env > "$WORK/policy.env"
  ( cd "$WORK" && find tree/fastpath tree/serve tree/config overlay-vllm sidecar_watchdog_p3.py policy.env -type f \
      ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) > "$WORK/BUILD-SHA256SUMS"
  log "built: $(wc -l < "$WORK/BUILD-SHA256SUMS") files hashed; libw6mk $(sha256sum "$TREE/fastpath/libw6mk.so" | cut -c1-16)"
}

cmd_render(){ policy_env | sed 's/^/env /'; echo "bash $TREE/serve/serve.sh start"; }

cmd_up(){
  [ -f "$WORK/policy.env" ] || die "run build first"
  : "${MODEL_DIR:?set MODEL_DIR}" "${DS41_HOME:?set DS41_HOME}"
  docker ps --format '{{.Names}}' | grep -qx ds41 && die "ds41 already running"
  # 2026-10-31 (H1): a killed server or a watchdog trip leaves /dev/shm/vllm_peer_tier{2,3} with published != done;
  # the --once check below would then read a stale-tag trip and refuse. With no ds41/ds41-sidecar running the
  # segments are dead state, so remove them here exactly as cmd_down does (and say so).
  if ls /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null 2>&1; then
    docker ps --format '{{.Names}}' | grep -qx ds41-sidecar && die "stale peer segment with ds41-sidecar running: run 'launch-pg61.sh down' first"
    docker run --rm -v /dev/shm:/dev/shm --entrypoint /bin/rm "$IMAGE" -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3
    ls /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null 2>&1 && die "could not remove stale peer segments"
    log "removed stale peer segments left by a previous run (nothing was running)"
  fi
  local envs; mapfile -t envs < "$WORK/policy.env"
  ( cd "$TREE" && env "${envs[@]}" DS41_HOME="$DS41_HOME" DS41_MODEL_DIR="$MODEL_DIR" bash serve/serve.sh start )
  for c in ds41 ds41-sidecar; do docker update --restart no "$c" >/dev/null; done
  local i; for i in $(seq 1 180); do curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED" && break; sleep 5; done
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$SERVED" || die "not bound on :$PORT"
  cmd_verify
  pgrep -f "sidecar_watchdog_p3.py --server ds41" >/dev/null && { log "watchdog already running"; return 0; }
  python3 "$WORK/sidecar_watchdog_p3.py" --server ds41 --sidecar ds41-sidecar --mode report --once \
     --status "$WORK/wdp3.once.json" --log "$WORK/wdp3.log" >/dev/null || die "watchdog --once refused (see $WORK/wdp3.log)"
  grep -q '"trip": null' "$WORK/wdp3.once.json" || die "watchdog --once reports a trip: $(cat "$WORK/wdp3.once.json")"
  ( setsid nohup python3 "$WORK/sidecar_watchdog_p3.py" --server ds41 --sidecar ds41-sidecar --mode enforce \
      --status "$WORK/wdp3.status.json" --log "$WORK/wdp3.log" >/dev/null 2>&1 </dev/null & )
  sleep 3; grep -q '"trip": null' "$WORK/wdp3.status.json" || die "watchdog not armed"
  log "UP on :$PORT; watchdog ENFORCE armed (status $WORK/wdp3.status.json)"
  # 2026-10-31 (A8): soak clock. Every up starts a new soak; the file is the receipt the recipe quotes.
  printf '{"soak_start_utc": "%s", "soak_start_local": "%s", "epoch": %s}\n' "$(date -u +%FT%TZ)" "$(date '+%F %T %Z')" "$(date +%s)" > "$WORK/soak-start.json"
  log "soak clock started: $(cat "$WORK/soak-start.json")"
}

cmd_verify(){
  local g r l s
  g=$(docker exec ds41 sha1sum $VP/sample/gumbel.py | cut -c1-40)
  r=$(docker exec ds41 sha1sum $VP/spec_decode/rejection_sampler_utils.py | cut -c1-40)
  l=$(docker exec ds41 sha256sum /fastpath/libw6mk.so | cut -c1-64)
  s=$(docker exec ds41 printenv PEER3_SPINS || true)
  [ "$g" = $WANT_GUMBEL ] && [ "$r" = $WANT_RSU ] && [ "$l" = "$(sha256sum < "$TREE/fastpath/libw6mk.so" | cut -c1-64)" ] \
    && [ "$s" = 1000000000000 ] || die "in-container check: gumbel $g rsu $r libw6mk ${l:0:16} PEER3_SPINS=$s"
  log "verify OK: gumbel ${g:0:12} rsu ${r:0:12} libw6mk ${l:0:16} PEER3_SPINS=$s"
}

cmd_down(){
  local p; for p in $(pgrep -f "sidecar_watchdog_p3.py --server ds41" || true); do kill "$p"; done
  docker unpause ds41-sidecar >/dev/null 2>&1 || true
  ( cd "$TREE" && DS41_HOME="${DS41_HOME:?set DS41_HOME}" bash serve/serve.sh down ) || true
  docker run --rm -v /dev/shm:/dev/shm --entrypoint /bin/rm "$IMAGE" -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3
  # 2026-10-31 (H1/H3): verify, do not assume.
  ls /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null 2>&1 && die "down: peer segments still present"
  docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && die "down: ds41 containers still running"
  pgrep -f "sidecar_watchdog_p3.py --server ds41" >/dev/null && die "down: p3 watchdog still running"
  log "down; peer shm removed (verified absent), containers stopped, watchdog stopped"
}

case "${1:-}" in
  build) cmd_build ;; up) cmd_up ;; down) cmd_down ;; verify) cmd_verify ;; render) cmd_render ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
