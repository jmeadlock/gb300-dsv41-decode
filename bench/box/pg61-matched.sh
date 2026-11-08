#!/bin/bash
# Pg61 matched re-run (James: yes, 11/08): boot Pg61 (sidecar config, its real policy), drive the 8 fixture prompts, down, build the ledger.
# Then residency-slot arms on main (pool 24 = 8 slots, 28 = 12). Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; W=$HOME/dsv41/decode
cd $W && git checkout -q main && git pull -q; git log --oneline -1 | cut -c1-60
echo "########## pg61 up ($(date +%H:%M))"
( cd ~/dsv41/garner && WORK=~/dsv41/garner/pg61-work MODEL_DIR=$M DS41_HOME=~/dsv41/garner/home launcher/launch-pg61.sh up ) > $S/pg61-up.log 2>&1; echo "up rc=$?"; tail -3 $S/pg61-up.log | cut -c1-200
for i in $(seq 1 60); do curl -s -m 5 http://127.0.0.1:30006/v1/models | grep -q dsv41 && break; sleep 10; done; curl -s -m 5 http://127.0.0.1:30006/v1/models | cut -c1-120
grep -h "rowmap\|policy" ~/dsv41/garner/pg61-work/*.env ~/dsv41/garner/home/*.json 2>/dev/null | head -3 | cut -c1-160
echo "########## pg61 drive ($(date +%H:%M))"
python3 bench/gap_ledger.py pg61-drive --reps 3 --json scratch/pg61-drive.json > $S/pg61-drive.log 2>&1; echo "drive rc=$?"; grep -vE "^\s*$" $S/pg61-drive.log | grep -iE "prompt|ms/step|tok|wall|total|Error" | cut -c1-200 | tail -14
echo "########## pg61 down ($(date +%H:%M))"
( cd ~/dsv41/garner && WORK=~/dsv41/garner/pg61-work MODEL_DIR=$M DS41_HOME=~/dsv41/garner/home launcher/launch-pg61.sh down ) > $S/pg61-down.log 2>&1; echo "down rc=$?"; docker ps --format "{{.Names}}"
while docker ps --format "{{.Names}}" | grep -q .; do sleep 10; done
echo "########## build ($(date +%H:%M))"
python3 bench/gap_ledger.py build --tf scratch/gap-tf-graph.json --anatomy scratch/gap-tf-anatomy.json --pg61-drive scratch/pg61-drive.json > $S/gap-ledger-build2.log 2>&1; echo "build rc=$?"; grep -E "^\| (complete step|tokens per|GPU ms per|wall ms per)|^- step_gpu|Gaps" -A0 $S/gap-ledger-ours.md | cut -c1-220 | head -8
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency-drain 64 --max-new 256"
rep() { grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $1 | grep -iE "two_graphs:|streams|cold share" | cut -c1-200 | head -3; }
for ps in 24 28; do echo "########## residency slots: pool-spare $ps ($(date +%H:%M))"; dk rs$ps --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --pool-spare $ps --json scratch/headline-rs$ps.json > $S/headline-rs$ps.log 2>&1; echo rc=$?; rep $S/headline-rs$ps.log; done
echo "########## done ($(date +%H:%M))"'
