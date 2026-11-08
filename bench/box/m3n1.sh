#!/bin/bash
# M3-N1 MegaMoE numerics, serialized under the box lease. Receipts to scratch/.
set -u
flock -w 36000 ~/dsv41/box.lock bash -c '
while docker ps --format "{{.Names}}" | grep -q .; do sleep 20; done
S=$HOME/dsv41/decode/scratch; M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170; W=$HOME/dsv41/wt-m2-l3
cd $W && git pull -q origin m2-l3; git log --oneline -1 | cut -c1-60
J=$HOME/dsv41/w0/jit-cache
dk() { N=$1; shift; docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref -v $J/tilelang:/root/.tilelang -v $J/triton:/root/.triton -v $J/nv:/root/.nv -v $J/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm --entrypoint python3 vllm/vllm-openai:v0.31.0 "$@"; }
echo "########## t gates ($(date +%H:%M))"; dk m3n1t tests/gpu/test_dg_mega_gpu.py > $S/m3n1-gpu.log 2>&1; echo rc=$?; grep -E "^(FAIL|PASS|[0-9]+/[0-9]+)" $S/m3n1-gpu.log | cut -c1-160 | tail -12
echo "########## l layers ($(date +%H:%M))"; dk m3n1l bench/dg_mega_numerics.py layers --out scratch/m3n1-layers.json > $S/m3n1-layers.log 2>&1; echo rc=$?; grep -vE "Warning|transformers" $S/m3n1-layers.log | grep -iE "layer|rel|ulp|us" | cut -c1-200 | tail -12
echo "########## p patch-dg ($(date +%H:%M))"; dk m3n1p bench/dg_mega_numerics.py patch-dg --out scratch/dg-bf16q > $S/m3n1-patch.log 2>&1; echo rc=$?; tail -3 $S/m3n1-patch.log | cut -c1-200
echo "########## q layers bf16q ($(date +%H:%M))"; dk m3n1q bench/dg_mega_numerics.py layers --dg-path scratch/dg-bf16q --jit-cache scratch/dg-bf16q-jit --out scratch/m3n1-layers-bf16q.json > $S/m3n1-layers-bf16q.log 2>&1; echo rc=$?; grep -vE "Warning|transformers" $S/m3n1-layers-bf16q.log | grep -iE "layer|rel|ulp|us" | cut -c1-200 | tail -12
echo "########## d downstream ($(date +%H:%M))"; dk m3n1d bench/dg_mega_numerics.py downstream --out scratch/m3n1-ds.json > $S/m3n1-ds.log 2>&1; echo rc=$?; grep -vE "Warning|^\[load\]|transformers" $S/m3n1-ds.log | grep -iE "arm|flips|margin|A3|fit|HBM" | cut -c1-200 | tail -14
echo "########## r report ($(date +%H:%M))"; dk m3n1r bench/dg_mega_numerics.py report --layers scratch/m3n1-layers.json --patched scratch/m3n1-layers-bf16q.json --downstream scratch/m3n1-ds.json --model scratch/m3n1-model-20261107.json > $S/m3n1-report.log 2>&1; echo rc=$?; grep -v Warning $S/m3n1-report.log | head -40 | cut -c1-220
echo "########## done ($(date +%H:%M))"'
