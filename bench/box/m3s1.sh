#!/bin/bash
# M3-S1 staging overlap: GPU test -> probe bench -> 1-row byte-identity -> 6-row headline (kernel-overlap, two NPROG settings). Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; W=$HOME/dsv41/wt-m3s1
git -C $HOME/dsv41/decode fetch -q origin m3-s1:m3-s1 2>/dev/null || git -C $HOME/dsv41/decode fetch -q origin
[ -d $W ] || git -C $HOME/dsv41/decode worktree add -q $W m3-s1
cd $W && git pull -q origin m3-s1; ln -sfn $S scratch; git log --oneline -1 | cut -c1-60
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency on --pool-spare 24 --residency-drain 64 --max-new 256"
FL="bench/first_light.py --model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit"
echo "########## t gpu test ($(date +%H:%M))"; dk s1t --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_staging_overlap_gpu.py > $S/s1-gpu.log 2>&1; echo rc=$?; grep -E "^(FAIL|PASS|[0-9]+/[0-9]+)|overlap|us/layer" $S/s1-gpu.log | cut -c1-180 | tail -14; grep -oE "[A-Za-z]*Error: .{0,160}" $S/s1-gpu.log | sort | uniq -c | head -3
echo "########## p probe ($(date +%H:%M))"; dk s1p --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/staging_overlap_bench.py all --out scratch/s1-bench.json > $S/s1-bench.log 2>&1; echo rc=$?; grep -vE "Warning|transformers" $S/s1-bench.log | grep -iE "GB/s|us|overlap|nprog|model" | cut -c1-200 | tail -16
echo "########## f 1-row kernel-overlap ($(date +%H:%M))"; dk s1f -e DSV41_STAGE=kernel-overlap --entrypoint python3 vllm/vllm-openai:v0.31.0 $FL --out scratch/fl-s1.jsonl > $S/fl-s1.log 2>&1; echo rc=$?; grep -E "step cell \(cuda|self-replay rep1 vs rep2:|stage" $S/fl-s1.log | cut -c1-180 | head -4; cmp -s $S/fl-s1.jsonl $S/fl-sw-gemv.jsonl && echo "1-row stream byte-identical to small-wins run" || echo "1-row stream DIFFERS"
for np in 48 16; do echo "########## h headline kernel-overlap NPROG=$np ($(date +%H:%M))"; dk s1h$np -e DSV41_STAGE=kernel-overlap -e DSV41_STAGE_OVERLAP_NPROG=$np --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-s1-$np.json > $S/headline-s1-$np.log 2>&1; echo rc=$?; grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $S/headline-s1-$np.log | grep -iE "two_graphs|streams|stage" | cut -c1-220 | head -4; done
echo "########## done ($(date +%H:%M))"'
