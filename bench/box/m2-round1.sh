#!/bin/bash
# Serialized box run for the M2 round (one process, one container at a time, per-branch worktrees).
set -u
M=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
S=$HOME/dsv41/decode/scratch
img=vllm/vllm-openai:v0.31.0
dk() { # dk <worktree> <name> [extra docker args...] -- <python args...>
  local W=$1 N=$2; shift 2; local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  docker run --rm --name mk-$N --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
    -v $W:/work -w /work -v $M:/ref:ro -v $S:/work/scratch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json -e DSV41_MODEL_DIR=/ref \
    -v $HOME/dsv41/w0/jit-cache/tilelang:/root/.tilelang -v $HOME/dsv41/w0/jit-cache/triton:/root/.triton -v $HOME/dsv41/w0/jit-cache/nv:/root/.nv -v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
    "${extra[@]}" --entrypoint python3 $img "$@"
}
HL="bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --max-new 256"
FL="bench/first_light.py --model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla --fp32-impl fixed --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit"
hl_summary() { grep -vE "Warning|^\[load\]|transformers|^\[r1\]|^\[loop" $1 | grep -iE "two_graphs|streams|Error" | cut -c1-240 | head -3; }
while docker ps --format '{{.Names}}' | grep -q .; do sleep 20; done
echo "########## L10 ($(date +%H:%M))"; W=$HOME/dsv41/wt-m2-l10; cd $W && git pull -q origin m2-l10 2>/dev/null
dk $W l10t -- tests/gpu/test_qkv_tail_gpu.py > $S/qkv-gpu.log 2>&1; grep -E "^(FAIL|PASS|[0-9]+/[0-9]+)" $S/qkv-gpu.log | cut -c1-140 | tail -6; grep -oE "[A-Za-z]*Error: .{0,200}" $S/qkv-gpu.log | sort | uniq -c | head -3
dk $W l10b -- bench/qkv_tail_bench.py --timing --json scratch/qkv-bench.json > $S/qkv-bench.log 2>&1; echo "bench rc=$?"; grep -vE "Warning|transformers" $S/qkv-bench.log | grep -iE "us|rows|split|fused" | cut -c1-200 | tail -8
dk $W l10h -e DSV41_QKV_TAIL=fused -- $HL --json scratch/headline-qkv.json > $S/headline-qkv.log 2>&1; echo "headline rc=$?"; hl_summary $S/headline-qkv.log
echo "########## L1 ($(date +%H:%M))"; W=$HOME/dsv41/wt-m2-l1; cd $W && git pull -q origin m2-l1 2>/dev/null
dk $W l1e -e DSV41_MIXES=fork -- -c 'import os; from runtime.fused_glue_r2 import mixes_impl; print("env", os.environ.get("DSV41_MIXES"), "impl", mixes_impl())'
dk $W l1f -e DSV41_MIXES=fork -- $FL --out scratch/fl-mixfork.jsonl > $S/fl-mixfork.log 2>&1; echo "1-row rc=$?"; grep -vE "Warning|^\[load\]|transformers" $S/fl-mixfork.log | grep -iE "step cell|self-replay|mixes" | cut -c1-200 | head -4; cmp -s $S/fl-mixfork.jsonl $S/fl-t9b8.jsonl && echo "1-row stream byte-identical to T9b-8 run" || echo "1-row stream DIFFERS from T9b-8 run"
for mx in fork fused; do dk $W l1h$mx -e DSV41_MIXES=$mx -- $HL --json scratch/headline-mixes-$mx.json > $S/headline-mixes-$mx.log 2>&1; echo "headline $mx rc=$?"; hl_summary $S/headline-mixes-$mx.log; done
echo "########## PDL probe clean ($(date +%H:%M))"; W=$HOME/dsv41/wt-m2-l5; cd $W && git pull -q origin m2-l5 2>/dev/null
dk $W l5 -- bench/pdl_probe.py --json scratch/pdl-probe-clean.json > $S/pdl-probe-clean.log 2>&1; echo "probe rc=$?"; grep -vE "Warning|transformers" $S/pdl-probe-clean.log | grep -E "graph a-c:|graph a-b:|bit-ident" | cut -c1-200
echo "########## L3 spike debug ($(date +%H:%M))"; W=$HOME/dsv41/wt-m2-l3; cd $W && git pull -q origin m2-l3 2>/dev/null
dk $W l3i -- -c 'import runtime.dg_mega as DM; print("dg_mega import ok", DM.__file__)' 2>&1 | tail -3 | cut -c1-200
dk $W l3s -- bench/dg_mega_spike.py --ckpt /ref --layer 0 --out scratch/dg-mega-spike.json > $S/dg-mega-spike.log 2>&1; echo "spike rc=$?"; grep -vE "Warning|^\[load\]|transformers" $S/dg-mega-spike.log | grep -iE "us|ulp|equal|flip|rows|verdict|mega|Error" | cut -c1-220 | tail -16
echo "########## done ($(date +%H:%M))"
