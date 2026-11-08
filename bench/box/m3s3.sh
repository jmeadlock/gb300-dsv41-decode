#!/bin/bash
# M3-S3 arms: gates, probe, A baseline (arm C config), B beside, C beside + 8 residency slots, 1-row sep vs beside. Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; W=$HOME/dsv41/wt-m3s1
cd $W && git pull -q origin m3-s1; git log --oneline -1 | cut -c1-60
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 20 --residency-drain 64 --max-new 256"
FL="bench/first_light.py --model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit"
rep() { grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $1 | grep -iE "two_graphs:|streams|sf_store|hbm_after|boot " | cut -c1-200 | head -5; }
echo "########## t gates ($(date +%H:%M))"; dk s3t --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_pool_layout_gpu.py > $S/s3-gpu.log 2>&1; echo rc=$?; grep -E "^(FAIL|[0-9]+/[0-9]+)" $S/s3-gpu.log | cut -c1-160 | tail -4; dk s3u --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py > $S/s3-gpu-so.log 2>&1; echo rc=$?; grep -E "^(FAIL|[0-9]+/[0-9]+)" $S/s3-gpu-so.log | cut -c1-160 | tail -2
echo "########## p probe ($(date +%H:%M))"; dk s3p --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/pool_layout_bench.py probe --out scratch/pool-layout-probe.json > $S/s3-probe.log 2>&1; echo rc=$?; grep -vE "Warning|transformers" $S/s3-probe.log | grep -iE "GiB|us|equal|stage" | cut -c1-200 | tail -8
echo "########## A baseline ($(date +%H:%M))"; dk s3a --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s3-A.json > $S/headline-s3-A.log 2>&1; echo rc=$?; rep $S/headline-s3-A.log
echo "########## B beside ($(date +%H:%M))"; dk s3b -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s3-B.json > $S/headline-s3-B.log 2>&1; echo rc=$?; rep $S/headline-s3-B.log; grep -oE "[A-Za-z]*Error: .{0,160}" $S/headline-s3-B.log | head -2
echo "########## C beside + pool 24 ($(date +%H:%M))"; dk s3c -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 ${HL/--pool-spare 20/--pool-spare 24} --json scratch/headline-s3-C.json > $S/headline-s3-C.log 2>&1; echo rc=$?; rep $S/headline-s3-C.log
echo "########## f 1-row sep vs beside ($(date +%H:%M))"; dk s3f0 --entrypoint python3 vllm/vllm-openai:v0.31.0 $FL --out scratch/fl-s3-sep.jsonl > $S/fl-s3-sep.log 2>&1; dk s3f1 -e DSV41_POOL_SF=beside --entrypoint python3 vllm/vllm-openai:v0.31.0 $FL --out scratch/fl-s3-bes.jsonl > $S/fl-s3-bes.log 2>&1; grep -E "step cell \(cuda" $S/fl-s3-sep.log $S/fl-s3-bes.log | cut -c1-150; cmp -s $S/fl-s3-sep.jsonl $S/fl-s3-bes.jsonl && echo "1-row stream byte-identical" || echo "1-row stream DIFFERS"
echo "########## done ($(date +%H:%M))"'
