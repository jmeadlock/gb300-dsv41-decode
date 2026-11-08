#!/bin/bash
# Gap ledger, our side: matched teacher-forced cell (events + census) and static anatomy. Serialized; receipts to scratch/.
set -u
W=$HOME/dsv41/decode; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; S=$W/scratch
while docker ps --format '{{.Names}}' | grep -q .; do sleep 20; done
cd $W && git checkout -q main && git pull -q
mkdir -p $S/prof
J=$HOME/dsv41/w0/jit-cache
D="docker run --rm --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm"
NSYS="--cap-add SYS_ADMIN -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro --entrypoint /nsys/target-linux-sbsa-armv8/nsys vllm/vllm-openai:v0.31.0 profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true"
echo "########## 1 matched events + census ($(date +%H:%M))"
$D --name mk-gl-tf --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/gap_ledger.py ours-tf --model-dir /ref --mode graph --reps 2 --census-loop --json scratch/gap-tf-graph.json > $S/gap-tf-graph.log 2>&1; echo "rc=$?"
grep -vE "Warning|^\[load\]|transformers" $S/gap-tf-graph.log | grep -E "\[tf\]|\[census\]|agree|overflow|ms/step|Error" | cut -c1-220 | head -24
echo "########## 2 matched static anatomy ($(date +%H:%M))"
$D --name mk-gl-prof $NSYS -o /work/scratch/prof/gl-tf python3 bench/gap_ledger.py ours-tf --model-dir /ref --mode static --reps 1 --profile 16:128 --nvtx-layers --json scratch/gap-tf-static.json > $S/gap-tf-static.log 2>&1; echo "rc=$?"
grep -oE "[A-Za-z]*Error: .{0,200}" $S/gap-tf-static.log | sort | uniq -c | head -3
/usr/local/bin/nsys export --type sqlite -o $S/prof/gl-tf.sqlite $S/prof/gl-tf.nsys-rep > /dev/null 2>&1; echo "export rc=$?"
python3 bench/step_anatomy.py $S/prof/gl-tf.sqlite --out $S/gap-tf-anatomy.json > $S/gap-tf-anatomy.txt 2>&1; echo "anatomy rc=$?"; head -30 $S/gap-tf-anatomy.txt | cut -c1-200
echo "########## 5 build ($(date +%H:%M))"
python3 bench/gap_ledger.py build --tf scratch/gap-tf-graph.json --anatomy scratch/gap-tf-anatomy.json > $S/gap-ledger-build.log 2>&1; echo "build rc=$?"; cat $S/gap-ledger-ours.md 2>/dev/null | head -60 | cut -c1-220
echo "########## done ($(date +%H:%M))"
