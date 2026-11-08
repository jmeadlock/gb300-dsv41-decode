#!/bin/bash
# M3-S1-FIX gate: GPU test (20x mini) -> headline per-projection shapes -> none-diag ceiling (wrong outputs, diagnostic only). Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; W=$HOME/dsv41/wt-m3s1
cd $W && git pull -q origin m3-s1; git log --oneline -1 | cut -c1-60
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 24 --residency-drain 64 --max-new 256"
echo "########## t gpu test ($(date +%H:%M))"; dk s1t2 --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py > $S/s1-gpu2.log 2>&1; echo rc=$?; grep -E "^(FAIL|PASS|[0-9]+/[0-9]+)" $S/s1-gpu2.log | cut -c1-160 | tail -8
echo "########## h headline shapes 48,16,16 ($(date +%H:%M))"; dk s1hs -e DSV41_STAGE=kernel-overlap -e DSV41_STAGE_OVERLAP_NPROG=48,16,16 -e DSV41_STAGE_OVERLAP_BLOCK_BYTES=8192,32768,32768 --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s1-shapes.json > $S/headline-s1-shapes.log 2>&1; echo rc=$?; grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $S/headline-s1-shapes.log | grep -iE "two_graphs|streams" | cut -c1-220 | head -3
echo "########## d none-diag ceiling (WRONG OUTPUTS, diagnostic) ($(date +%H:%M))"; dk s1nd -e DSV41_STAGE=none-diag -e DSV41_STAGE_DIAG=wrong-outputs-ok --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s1-nonediag.json > $S/headline-s1-nonediag.log 2>&1; echo rc=$?; grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $S/headline-s1-nonediag.log | grep -iE "two_graphs|streams" | cut -c1-220 | head -3
echo "########## done ($(date +%H:%M))"'
