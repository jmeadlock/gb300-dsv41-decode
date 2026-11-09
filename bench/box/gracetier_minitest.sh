#!/bin/bash
# P1 gracetier mini-test (docs/P1-GRACETIER-DESIGN.md section 5; probe bench/gracetier_minitest.py). Lease-serialized,
# window-guarded. Receipts to scratch/ (never docs/).
#
#   GRACETIER_WINDOW=<pre-registered window id> bench/box/gracetier_minitest.sh [--layer 10] [--json scratch/x.json] \
#       [-- <extra probe args, e.g. --iters 100>]
#
# The window: stopping :30006 (Pg61) needs James's word and a pre-registered window (PORT-PLAN.md:51). This script
# NEVER stops, starts or restores the server; the window owner stops Pg61 before and restores it at the window end.
# It refuses (exit 64/65/66) unless GRACETIER_WINDOW is set, no container is running and the GPU holds no compute
# process / little memory. Env: PG61_FASTPATH = the box's Pg61 fastpath dir (Garner's router; mounted read-only at
# /garner/fastpath; unset or missing -> the probe's router check reports `not run`), GT_WORKTREE (default the decode
# checkout), DSV41_MODEL_DIR_HOST (the checkpoint, default the 1017 snapshot on /models).
set -u
LAYER=10
JSON=""
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --layer) LAYER="$2"; shift 2 ;;
        --json) JSON="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "usage: $0 [--layer N] [--json scratch/<name>.json] [-- probe args]" >&2; exit 64 ;;
    esac
done
STAMP="$(date +%Y%m%d-%H%M%S)"
[ -n "$JSON" ] || JSON="scratch/gracetier-minitest-L${LAYER}-${STAMP}.json"
case "$JSON" in
    scratch/*) ;;
    *) echo "refusing: --json must be under scratch/ (got $JSON)" >&2; exit 64 ;;
esac

# ---- window guard (before the lease: a refused run must not queue)
if [ -z "${GRACETIER_WINDOW:-}" ]; then
    echo "refusing: GRACETIER_WINDOW is unset. Set it to the pre-registered window id once James has said go and" \
         "the window owner has stopped :30006." >&2
    exit 64
fi

source "$(dirname "$0")/lease.sh"

RUNNING="$(docker ps --format '{{.Names}}' | tr '\n' ' ')"
if [ -n "${RUNNING// /}" ]; then
    echo "refusing: containers running ($RUNNING) -- the window needs an idle GB300 (Pg61 stopped)" >&2
    exit 65
fi
APPS="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
if [ -n "$APPS" ] || [ "${USED:-0}" -gt 8192 ]; then
    echo "refusing: GPU not idle (compute apps: ${APPS:-none}; memory.used ${USED:-?} MiB)" >&2
    exit 66
fi

S=$HOME/dsv41/decode/scratch
M=${DSV41_MODEL_DIR_HOST:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
W=${GT_WORKTREE:-$HOME/dsv41/decode}
J=$HOME/dsv41/w0/jit-cache
GARNER_MOUNT=()
if [ -n "${PG61_FASTPATH:-}" ] && [ -d "${PG61_FASTPATH}" ]; then
    GARNER_MOUNT=(-v "${PG61_FASTPATH}:/garner/fastpath:ro")
else
    echo "note: PG61_FASTPATH unset or missing -- M2's router check will report 'not run'" >&2
fi
mkdir -p "$S"
cd "$W" || exit 70
git log --oneline -1 | cut -c1-60
echo "window ${GRACETIER_WINDOW}; layer ${LAYER}; receipt ${JSON}; $(date '+%Y-%m-%d %H:%M %Z')"

dk() { N=$1; shift; docker run --rm --name gt-$N --network none \
    --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    --cap-add IPC_LOCK -v "$W":/work -w /work -v "$M":/ref:ro -v "$S":/work/scratch ${GARNER_MOUNT[@]+"${GARNER_MOUNT[@]}"} \
    $(box_lease_docker_env) -e GRACETIER_WINDOW="${GRACETIER_WINDOW}" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv \
    -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm --entrypoint python3 vllm/vllm-openai:v0.31.0 "$@"; }

echo "########## t GPU gate tests ($(date +%H:%M))"
dk gtt tests/gpu/test_gracetier_gpu.py > "$S/gracetier-gpu-${STAMP}.log" 2>&1
echo "rc=$?"; grep -E "^(FAIL|PASS|SKIP|[0-9]+/[0-9]+)" "$S/gracetier-gpu-${STAMP}.log" | cut -c1-160 | tail -12

echo "########## p probe M0..M7 ($(date +%H:%M))"
dk gtp bench/gracetier_minitest.py all --layer "$LAYER" --json "$JSON" --model-dir /ref \
    --garner /garner/fastpath ${EXTRA[@]+"${EXTRA[@]}"} > "$S/gracetier-minitest-${STAMP}.log" 2>&1
RC=$?
echo "rc=$RC (0 ok | 2 phase error | 3 M1 gate failed: stop, compaction fallback | 4 M4 gate failed)"
grep -E "^\[M[0-7]\]|^GRACETIER-MINITEST|^receipt|GATE" "$S/gracetier-minitest-${STAMP}.log" | cut -c1-220 | tail -60
echo "########## done ($(date +%H:%M)) -- window owner: restore Pg61 on :30006 now (PORT-PLAN.md:51)"
exit $RC
