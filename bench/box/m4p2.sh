#!/bin/bash
# M4-P2 grouped prefill: GPU test -> bench (streams 1,4,8, equality to reference) -> headline with grouped prefill (stream byte-identity via tokens_sha256). Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
W=$HOME/dsv41/wt-m4p2; S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
git -C $HOME/dsv41/decode fetch -q origin m4-p2:m4-p2 2>/dev/null || git -C $HOME/dsv41/decode fetch -q origin
[ -d $W ] || git -C $HOME/dsv41/decode worktree add -q $W m4-p2
cd $W && git pull -q origin m4-p2; ln -sfn $S scratch; git log --oneline -1 | cut -c1-60
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm "$@"; }
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency-drain 64 --max-new 256"
echo "########## t gpu test ($(date +%H:%M))"; dk p2t --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_prefill_fast_gpu.py > $S/m4p2-gpu-test.log 2>&1; echo rc=$?; grep -E "^(FAIL|PASS|[0-9]+/[0-9]+)" $S/m4p2-gpu-test.log | cut -c1-160 | tail -8; grep -oE "[A-Za-z]*Error: .{0,160}" $S/m4p2-gpu-test.log | sort | uniq -c | head -3
echo "########## b bench ($(date +%H:%M))"; dk p2b --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/prefill_bench.py run --model-dir /ref --only all --reps 3 --streams 1,4,8 --kineto --json /work/scratch/prefill-bench-m4p2.json > $S/prefill-bench-m4p2.log 2>&1; echo rc=$?; grep -vE "Warning|^\[load\]|transformers" $S/prefill-bench-m4p2.log | grep -iE "prefill|equal|streams|ttft|launch|Error" | cut -c1-220 | tail -24
echo "########## h headline grouped prefill ($(date +%H:%M))"; dk p2h -e DSV41_PREFILL=grouped -e DSV41_PREFILL_STREAMS=4 --entrypoint python3 vllm/vllm-openai:v0.31.0 $HL --json scratch/headline-m4p2.json > $S/headline-m4p2.log 2>&1; echo rc=$?; grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $S/headline-m4p2.log | grep -iE "two_graphs:|streams exact|ttft|prefill|Error" | cut -c1-220 | head -6
python3 - <<PY
import json
a=json.load(open("$S/headline-main-ov2.json")); b=json.load(open("$S/headline-m4p2.json"))
same=[k for k in a["prompts"] if a["prompts"][k]["tokens_sha256"]==b["prompts"][k]["tokens_sha256"]]
print("tokens_sha256 equal to reference-prefill run:", len(same), "/ 8")
PY
echo "########## done ($(date +%H:%M))"'
