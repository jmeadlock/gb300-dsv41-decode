#!/bin/bash
# M4-P1 box: prefill anatomy run (timing, fills, census, MoE probe, rows census) + nsys of one prefill + analyze. Lease-serialized.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
W=$HOME/dsv41/decode; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; S=$W/scratch
cd $W && git pull -q; git log --oneline -1 | cut -c1-60; mkdir -p $S/prof
D="docker run --rm --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v $HOME/dsv41/w0/jit-cache/tilelang:/root/.tilelang -v $HOME/dsv41/w0/jit-cache/triton:/root/.triton -v $HOME/dsv41/w0/jit-cache/nv:/root/.nv -v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref"
echo "########## 1 run ($(date +%H:%M))"; $D --name mk-prefill --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/prefill_anatomy.py run --model-dir /ref --only all --reps 3 --fills both --probe-moe 0,1,2,14,20,39 --rows-probe 42,64 --json /work/scratch/prefill-anatomy-all.json > $S/prefill-anatomy-all.log 2>&1; echo rc=$?
grep -vE "Warning|^\[load\]|transformers" $S/prefill-anatomy-all.log | grep -iE "prefill|fills|equal|census|moe_probe|rows|ms|Error" | cut -c1-200 | tail -30
echo "########## 2 nsys ($(date +%H:%M))"; $D --name mk-prefill-nsys --cap-add SYS_ADMIN -v /opt/nvidia/nsight-systems/2025.6.3:/nsys:ro --entrypoint /nsys/target-linux-sbsa-armv8/nsys vllm/vllm-openai:v0.31.0 profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o /work/scratch/prof/prefill-prose python3 bench/prefill_anatomy.py run --model-dir /ref --only prose --reps 0 --fills on --profile --json /work/scratch/prefill-anatomy-prose-nsys.json > $S/prefill-anatomy-prose-nsys.log 2>&1; echo rc=$?
/usr/local/bin/nsys export --type sqlite -o $S/prof/prefill-prose.sqlite $S/prof/prefill-prose.nsys-rep > /dev/null 2>&1; echo "export rc=$?"
python3 $W/bench/prefill_anatomy.py analyze $S/prof/prefill-prose.sqlite --out $S/prefill-anatomy-prose-nsys-analysis.json > $S/prefill-anatomy-prose-analysis.log 2>&1; echo "analyze rc=$?"; head -40 $S/prefill-anatomy-prose-analysis.log | cut -c1-200
echo "########## done ($(date +%H:%M))"'
