# runtime/ — M0 loader, placement, host tier, expert pools (T2+T3)

Standalone: no vllm import anywhere (charter decision 3). The reference `model.py` is imported unchanged from
`reference/deepseek-inference/` (`runtime/refmodel.py`); every behavior change is a flag-gated monkeypatch in
`patches.py` (default off) citing the model.py line it replaces.

| file | what |
|---|---|
| `ckpt.py` | shard index (headers scanned once), convert.py's rename map (`convert_name`, line-for-line from convert.py:104-126), `read_tensor_into` / `read_into` = `preadv` into the caller's buffer + `posix_fadvise(DONTNEED)` per 64 MiB chunk |
| `placement.py` | `place(name, shape, dtype, cfg, map) -> Tier.HBM / HOST / SKIP`, hot maps (JSON `{layer: [hot ids]}` or `--hot-per-layer N`), the dry-run report CLI |
| `host_tier.py` | `HostBuffer`: exact-size anonymous mmap (+ MADV_HUGEPAGE), `cudaHostRegister` in chunks of 64 GiB or less with rollback, a UVA CUDA view built from `__cuda_array_interface__`, `release()`, MemAvailable checks |
| `expert_pool.py` | hot experts in per-layer HBM slot pools `[H,N,K/2]` + `[H,N,K/32]`; cold experts in one registered host pool, same layout; `loc[L][E] = (slot, tier)` int32 on device; binds the reference `Expert`s to zero-copy views (M0a zero-copy arm); host-side cold-hit counters |
| `loader.py` | `build_and_load(model_dir, map, device)` and a CLI: meta construction, tiered streaming load, refusal on any key/shape/dtype mismatch, `--verify` sha256 of every placed tensor |
| `refmodel.py` | imports the reference modules; `cpu_stubs=True` (Mac tests only) stands in for tilelang/PIL; Python < 3.10 compiles them with `from __future__ import annotations` (the file is untouched) |
| `generate.py` | T5 driver: `Generator(model, tokenizer_dir)` -- `encode`/`decode` (checkpoint `encoding.py` chat mode + `tokenizer.json`), `prefill`/`step` (generate.py's call pattern), `greedy` -> one parity JSONL record (fp32 log-softmax top-20), NVTX `decode_step`, `arm_profile` (cudaProfilerApi window), `set_determinism`; T8 `mode="eager"|"static"|"graph"`, `marker=True` |
| `static_step.py` | T8: `StaticStep(model, pool, max_pos=)` -- the fixed-shape 1-row decode step (design table 2.1 #1-#15 row by row in its docstring), `forward_static(parity)` eager, `capture()` g_even/g_odd, `replay(pos)` |
| `static_kernels.py` | T8 Triton kernels (GPU only): the spike's pointer-table gather `t8_gather_rows`, the per-step `t8_step_marker` |
| `dg_moe.py` | T9a-1: `DGMaskedMoE` (`--moe-impl dg-masked`): the static MoE's routed GEMMs as 3 DeepGEMM masked grouped GEMMs over the same staging; `import_deep_gemm` (no vllm import); exact e8m0 → fp32 |
| `patches.py` | flag-gated monkeypatches: `kv_numerics_pg61` (model.py:761), `static_moe` (model.py:891-906, design #14) |

## Value paths (bit-identity rule)

Every placed tensor keeps its shard bytes except three transforms. All three are exact, and the loader checks
each one against the reference parameter's dtype before it reads a byte:

1. **wo_a**: fp8 × E8M0 becomes bf16 by convert.py:157-173's formula, on the CPU. Its `.scale` is used up by the dequant.
2. **bf16 → fp32 widening** where model.py declares a parameter fp32 and the checkpoint stores it bf16:
   - `head.weight` (model.py:1007-1008);
   - Compressor `wkv`/`wgate` at compress ratio > 1 (model.py:444-448).

   This is what `safetensors.load_model`'s `copy_` does for the reference (generate.py:133). The handoff named
   only wo_a, but the reference cannot be matched without this widening. Plan §4 already budgets the fp32 head
   (2.47 GiB). `--verify` narrows the value back to bf16 and compares bytes.
3. **Expert weights**: `I8 [N, K/2]` is viewed as `float4_e2m1fn_x2` (convert.py:181). This is zero-copy; the bytes do not change.

Anything else refuses the load and prints the missing, unexpected and mismatched lists. SKIP covers:
- vision, aligner, `image_*` and `gate.bias_vl` (the `vision_n_layers=0` overlay);
- `mtp.*` (the `dspark_block_size=0` overlay).

## Construction without allocating on the GPU

The plan (§3) says the reference build OOMs at layer 14's Engram when built under `torch.device("cuda")`.

**How the loader avoids it.** It builds under the target device inside a `TorchFunctionMode` that sends
`torch.empty` and `torch.ones` to `meta`:
- In model.py, those two factories create every Parameter.
- Buffers come from zeros, full, tensor and RoPE math, so they are real from the start.

**Why not plain `torch.device("meta")`.** That would also put the buffers on meta, and it would poison
`precompute_freqs_cis`'s `lru_cache` with meta tensors.

**One exception.** `engram_hash.cache` is a `torch.empty` buffer. It is written before it is read, so the loader re-allocates it.

**Checks after construction.**
- Every parameter must be meta.
- Every other buffer must be real.

**Load order (engram is the HOST tier).**
1. HBM params: `torch.empty` on the device, then pinned staging.
2. HOST tier: engram tables and the cold pool are read straight into their `HostBuffer`s.
3. `register_all`, with MemAvailable checks before and after.
4. UVA views are assigned.
5. `weight.scale = scale` is re-set on every quantized Linear (model.py:222).

**Process-global side effects.**
- `build_and_load` leaves `torch.set_default_dtype(torch.bfloat16)` set, as generate.py:118 does.
- The driver must still call `torch.set_default_device("cuda")` before a forward (generate.py:134), because the
  forward builds index tensors on the default device.

## Run it

**Mac (CPU) tests.** These cover the reference `Transformer` built and loaded on CPU and compared byte-for-byte
with convert.py's own MP=1 output:

    .venv/bin/python -m pytest -q tests/

**Placement dry-run.** CPU only; it reads the shard headers and nothing else:

    python -m runtime.placement --model-dir /models/DeepSeek-V4.1-Flash-1017-1bae0e10... --hot-per-layer 295
    python -m runtime.placement --model-dir D --map hot_map.json --json report.json

On the box, inside the image, from the repo checkout (no GPU needed):

    docker run --rm -v "$HOME/hermes/gb300-dsv41-decode":/work -w /work \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 -m runtime.placement --model-dir /ref --hot-per-layer 295

The config defaults to `<model-dir>/inference/config.json` (the checkpoint's own copy of the reference config).
If that file is missing, it falls back to the vendored `reference/deepseek-inference/config.json`.

**GPU tests (box).** The exact `docker run` line is in the docstring of `tests/gpu/test_host_tier_gpu.py`. It runs
`python3 tests/gpu/test_host_tier_gpu.py`, which needs no pytest. The tests:
1. register and the UVA view, plus rollback on failure;
2. the Engram gather, UVA vs HBM, `torch.equal` on 10⁶ random ids (real bytes if `DSV41_MODEL_DIR` is set);
3. hot and cold pool views feeding the reference `fp4_gemm` / `Expert`, `torch.equal` vs HBM copies;
4. mini checkpoint `build_and_load` on cuda with `--verify`;
5. mini forward, tiered vs all-HBM, `torch.equal`.

**Real boot + verify (box, T2's done-when).**

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$HOME/hermes/gb300-dsv41-decode":/work -w /work \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      -m runtime.loader --model-dir /ref --hot-per-layer 295 --verify --json /work/scratch/t2-boot.json

To use the static rowmap, pass `--map <rowmap-static-v1 as {layer: [hot ids]}.json>` instead of
`--hot-per-layer 295`. That map file is not in this repo. `--hot-per-layer 295` takes ids 0..294 (the fallback).
`--verify` re-reads all ~475 GiB and hashes them, so expect minutes.

## Memory numbers

**Synthetic mini checkpoint** (`tests/mini_ckpt.py`, printed by the dry-run on the Mac). Layout: 2 layers, dim 256,
8 experts (4 hot / 4 cold per layer), Engram 2 × 1,000 rows.

    tier  role                   n   disk GiB  placed GiB     placed bytes
    HBM   attention             29      0.001       0.001          919,200
    HBM   embed                  1      0.001       0.001        1,048,576
    HBM   engram_proj            8      0.004       0.004        3,944,192
    HBM   head                   1      0.001       0.002        2,097,152
    HBM   indexer                5      0.000       0.000           51,360
    HBM   norm_mhc              17      0.000       0.000          396,208
    HBM   routed_experts        48      0.001       0.001          835,584
    HBM   router                 4      0.000       0.000            8,256
    HBM   shared_experts        12      0.000       0.000          393,600
    HOST  engram_table           4      0.000       0.000          528,000
    HOST  routed_experts        48      0.001       0.001          835,584
    SKIP  skip:mtp               8      0.000       0.000                0
    SKIP  skip:vision           12      0.000       0.000                0
    HBM   TOTAL                125      0.008       0.009        9,694,128
    HOST  TOTAL                 52      0.001       0.001        1,363,584
    SKIP  TOTAL                 20      0.001       0.000                0

**Real 1017 shapes, experts + Engram only.** These come from a header-free virtual index built from config.json,
cross-checked against `docs/checkpoint-1017-inventory.json` in `test_placement_report_reproduces_plan_section4`:

    HBM   routed_experts     70800    206.612     206.612  221,847,552,000
    HOST  engram_table           4    188.833     188.833  202,758,032,400
    HOST  routed_experts     21360     62.334      62.334   66,930,278,400
    plan section 4 check: HBM experts 206.61 GiB | host experts 62.33 | Engram 188.83 | host pinned 251.17

That matches M0-PORT-PLAN.md §4 (206.61 / 62.33 / 188.83; 251.17 pinned). At 325 hot per layer the test also
reproduces 227.62 / 41.32 / 230.16.

**Real checkpoint dry-run (headers of /models).** The orchestrator pastes it here:

    (pending: python -m runtime.placement --model-dir /ref --hot-per-layer 295)

## First light (T5 + T6)

`runtime/generate.py` drives the reference forward exactly as `generate.py` does: `model(tokens[:, 0:L], 0)` to
prefill, then one token at its own position. There is no cache reset between prompts; a fresh prompt overwrites
from position 0 and nothing stale is read (the module docstring has the line cites). Greedy is the model's own
`argmax` (`model.temperature = 0`). The record is the W0 parity JSONL record minus `wall_s`, so two replays are
byte-identical files. EOS is kept in `logprobs`/`n_tokens` and dropped from `text`, as in the W0 capture.

**Mac tests** (CPU): `.venv/bin/python -m pytest -q tests/`. `tests/test_generate_cpu.py` reproduces the 8 fixture
prompt-token counts and first ids with the vendored `encoding.py`. It needs a DSv4.1 `tokenizer.json` on the Mac:
`$DSV41_TOKENIZER_DIR`, or the local `~/code/deepseek-recipe/static/tokenizers/v41` copy (a modified V4-Flash-Vision-
Exp file; the box test with the checkpoint's own file is first light itself). It also re-creates every Pg61 token
string by single-id decode and runs first_light's whole post-boot path on a fake model.

**GPU test (box), mini checkpoint:** prefill + 10 greedy steps, self-replay equal (with a longer prompt in between),
profile/NVTX hooks, and the CUDA-event step cell:

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$HOME/hermes/gb300-dsv41-megakernel":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_generate_mini_gpu.py

**First light (box), real checkpoint.** Mounts: the T2 boot's two (repo at `/work`, checkpoint at `/ref`), plus
`$W/w0` (the fixture), the JIT caches, `IPC_LOCK`, and the cuBLAS workspace setting:

    W=$HOME/hermes/gb300-dsv41-megakernel
    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --hot-per-layer 295 \
        --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-cand.jsonl \
        --max-new 256 --reps 2 --step-cell 512

What it writes:
- `scratch/first-light-cand.jsonl` (rep 1) and `scratch/first-light-cand-replay.jsonl` (rep 2);
- `<jsonl>.parity.json` for each rep (`bench/parity.py --json`);
- `docs/first-light-<YYYYMMDD>.json` with boot, determinism, per-prompt meta (fp32-logit sha256, token ids, prefill,
  cold hits), parity, self-replay, the step cell (per-step ms, cold calls per step and per layer), and errors.

It prints the parity tables, any self-replay divergence, and a 12-line summary. The exit code is 0, or 2 if any
stage logged an error. The run continues past a failing prompt.

Options:
- `--map hot.json` replaces `--hot-per-layer 295`.
- `--only prose,json` runs a subset.
- `--no-fill-uninit` turns off torch's NaN fill of `torch.empty` under deterministic mode. That fill is on by default
  and shows in the step time.
- For an nsys capture of the step cell, add `--skip-parity --profile --profile-warm 32 --profile-steps 64
  [--nvtx-layers]` and wrap the command in `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.
  Where nsys lives (host `/usr/local/bin/nsys` in W0) and how it is mounted into the container is not settled here;
  the hooks are tested only without nsys attached.

## T8: one CUDA graph per decode step (`static_step.py`)

Spec: `docs/T8-T9A-DESIGN.md` section 2. `Generator(..., mode="graph")` captures `g_even`/`g_odd` on its first
prefill and replays them per step; `mode="static"` runs the same code eagerly; prefill is always the reference's.
Every replacement keeps the reference arithmetic; the one known limit is design risk 6: once a ratio-1 layer can
reach more than 512 compressed positions (pos > 512), or a candidate source more than 2,048 blocks, the selected set
depends on score bits and topk tie order at a different width than the reference's, and is identical only when there
is no tie at the k-th boundary (tests/test_static_cpu.py pins exactly that property). The W0 fixture (<= 310 tokens)
never gets there; the 512-step step cell does after ~470 steps.

**Mac:** `.venv/bin/python -m pytest -q tests/` -- `test_static_cpu.py` (index math vs the reference's host
functions, padded-vs-dynamic topk with ties, parity selection, Engram hash, Generator modes, first_light's
`--parity-levels` table) and `test_static_emulated_cpu.py` (the whole `StaticStep` vs the reference forward on the
mini checkpoints with CPU emulations of the six kernels, `torch.equal` every position).

**Box, GPU test** (mini checkpoints; static-eager == reference, graph == static-eager, no host sync, fills-on
recapture, relaunch, marker, step cell):

    W=$HOME/hermes/gb300-dsv41-decode
    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_static_mini_gpu.py

**Box, first light in graph mode** (L0/L1/L2 table first, then the reps graded by bench/parity.py, then the step cell
in graph mode; same mounts as T6's first light):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --hot-per-layer 295 --mode graph --parity-levels --max-pos 4095 \
        --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-graph.jsonl \
        --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit --json docs/first-light-graph-<date>.json

`--no-fill-uninit` applies to the reps/step cell (timed runs fills off, design 1(b)); `--parity-levels` runs its own
fills-off and fills-on passes regardless. Add `--levels-ref <earlier report>` on a second run for the cross-process
relaunch rows; add `--marker` (+ `--skip-parity --profile`) under nsys to find step boundaries after replay.

## T9a-1: routed experts on DeepGEMM (`--moe-impl dg-masked`, `dg_moe.py`)

Spec: `docs/HANDOFF-T9A1-DG-MOE.md`; ABI and numerics: `docs/T9A-DEEPGEMM-ABI-20261106.md` (the call is
`bench/dg_masked_spike.py`'s, copied). `StaticStep(..., moe_impl="dg-masked")` / `Generator(..., moe_impl=...)` /
`first_light.py --moe-impl dg-masked` keep the T8 step and its gate, sort and pointer-table gather unchanged, then
per layer:
1. one `act_quant(x, 32, "ue8m0")` written to row 0 of every group of a shared `xq[k,128,DIM]` fp8 / `xs[k,128,DIM/32]`
   fp32 buffer (rows 1.. stay zero/one; `max_m >= 128` is required, smaller silently corrupts groups >= 1);
2. e8m0 → fp32 of the gathered scales, one device op per projection inside the graph (torch's e8m0 cast, or the
   spike's `exp2(e8m0-127)`; whichever is bit-exact over all 256 codes on the device, checked at build);
3. w1 and w3 as `m_grouped_fp8_fp4_gemm_nt_masked(..., masked_m=ones(k), expected_m=8, recipe_a=(1,32),
   recipe_b=(1,32))` over `staging[(w1|w3, weight)]` (checkpoint bytes as-is) → row 0 → the reference glue
   (`.float()`, clamps, `silu(gate)*up*weight` in fp32, `.to(bf16)`);
4. `act_quant` of that → the w2 masked call → row 0 → fp32 sum in slot order (ascending id) → + shared expert.

`ref` (the default) is the T8 path, untouched and bit-identical; `dg-masked` is not bit-identical by design (DeepGEMM's
block-scaled UMMA: ~1 bf16 ulp on 0.01–0.04 % of GEMM outputs in the spike). DeepGEMM comes from the image's
`/usr/local/lib/python3.12/dist-packages/vllm/third_party` (override: `DSV41_DEEP_GEMM_PATH`) with vllm never imported.
It JIT-compiles on first call: `capture()` warms both GEMM shapes before capturing, and the cache must be mounted
(`-v $HOME/dsv41/w0/jit-cache/dg:/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm`).
`deep_gemm.use_deterministic_algorithms(True)` is applied at build (design §4); the report's `dg_moe` records what
happened, the e8m0 method, and the deep_gemm path/version.

**Mac:** `.venv/bin/python -m pytest -q tests/` — `tests/test_dg_moe_cpu.py`: e8m0 table exactness, the call-contract
constants and buffer shapes, the deep_gemm import guard, and the whole dg-masked `StaticStep` == the reference
forward (`torch.equal`, every position) with the reference kernels and DeepGEMM both emulated by the same math (a
contract-checking fake that NaN-poisons padding rows), plus `--moe-impl` / the parity-levels drift row on the fake.

**Box, GPU test** (mini checkpoints: dg static-eager runs and self-replays, no host sync, graph == static-eager, replay
x2, fills-on recapture, relaunch; top-1 stream == ref over 10 steps; per-layer MoE bf16 mismatch vs ref printed):

    W=$HOME/hermes/gb300-dsv41-decode
    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_dg_moe_gpu.py

**Box, first light** (`--parity-levels`: L1/L2 gated as for T8; L0 static-eager vs reference-eager reported, not
gated; extra rows "ref static-eager vs reference-eager" (gated) and "dg-masked vs ref static-eager" (first token
divergence + mean |Δlogit|, reported); then the reps graded vs Pg61 by bench/parity.py, then the graph step cell):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --hot-per-layer 295 --mode graph --moe-impl dg-masked --parity-levels \
        --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-dg.jsonl \
        --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit --json docs/first-light-dg-<date>.json

`--levels-ref` x-proc rows compare runs with the same label, so point it at a report made with the same `--moe-impl`.
