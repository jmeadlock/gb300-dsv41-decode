#!/bin/bash
# M3-S2 arms: gate, bytes probe, A baseline, B partial SF pool, hist, C pool-spare 20, D spare 8/pool 16 (read overflow!). Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; W=$HOME/dsv41/wt-m3s1
cd $W && git pull -q origin m3-s1; git log --oneline -1 | cut -c1-60
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 24 --residency-drain 64 --max-new 256"
rep() { grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $1 | grep -iE "two_graphs|streams|sf_store|overflow" | cut -c1-220 | head -4; }
echo "########## t gate ($(date +%H:%M))"; dk s2t --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py > $S/s2-gpu.log 2>&1; echo rc=$?; grep -E "^(FAIL|[0-9]+/[0-9]+)" $S/s2-gpu.log | cut -c1-160 | tail -3
echo "########## p bytes probe ($(date +%H:%M))"; dk s2p --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/staging_bytes_bench.py probe --out scratch/staging-bytes-probe.json > $S/s2-probe.log 2>&1; echo rc=$?; grep -vE "Warning|transformers" $S/s2-probe.log | grep -iE "MB|us|bytes|term" | cut -c1-200 | tail -10
echo "########## A baseline ($(date +%H:%M))"; dk s2a --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s2-A.json > $S/headline-s2-A.log 2>&1; echo rc=$?; rep $S/headline-s2-A.log
echo "########## B partial SF pool ($(date +%H:%M))"; dk s2b -e DSV41_SF_POOL_LAYERS=auto -e DSV41_SF_POOL_HEADROOM_GIB=2.0 --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s2-B.json > $S/headline-s2-B.log 2>&1; echo rc=$?; rep $S/headline-s2-B.log
echo "########## H histogram ($(date +%H:%M))"; dk s2h -e DSV41_STAGE_HIST=scratch/stage-hist-s2.json --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s2-hist.json > $S/headline-s2-hist.log 2>&1; echo rc=$?; python3 -c "
import json; h=json.load(open(\"$S/stage-hist-s2.json\")); print(str(h)[:600])" 2>&1 | head -3
echo "########## C pool-spare 20 ($(date +%H:%M))"; dk s2c -e DSV41_SF_POOL_LAYERS=auto -e DSV41_SF_POOL_HEADROOM_GIB=2.0 --entrypoint python3 vllm/vllm-openai:v0.31.0 ${HL/--pool-spare 24/--pool-spare 20} --json scratch/headline-s2-C.json > $S/headline-s2-C.log 2>&1; echo rc=$?; rep $S/headline-s2-C.log
echo "########## D spare 8 / pool 16 ($(date +%H:%M))"; dk s2d -e DSV41_SF_POOL_LAYERS=auto -e DSV41_SF_POOL_HEADROOM_GIB=2.0 --entrypoint python3 vllm/vllm-openai:v0.31.0 ${HL/--spare-slots 16 --residency on --pool-spare 24/--spare-slots 8 --residency on --pool-spare 16} --json scratch/headline-s2-D.json > $S/headline-s2-D.log 2>&1; echo rc=$?; rep $S/headline-s2-D.log
echo "########## done ($(date +%H:%M))"'
