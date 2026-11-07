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
| `static_kernels.py` | T8 Triton kernels (GPU only): the pointer-table gather `t8_gather_rows`, the per-step `t8_step_marker`; T9b-2's one-launch six-table gather `t9b2_gather_tables`; T9b-3a: both gathers move 16-B vectors per lane (the byte kernels kept as `*_u8`, `DSV41_GATHER_IMPL=u8`) |
| `kernels/sparse_attn_bounded.py` | T9b-3a (`--attn-impl bounded`): a copy of the reference TileLang `sparse_attn` (kernel.py:310-403) whose 64-index block loop is bounded by a device per-row valid-block count; three loop spellings, build-time self-check vs the reference; the count math (`nblk_static`, `valid_blocks`, `ref_width`) |
| `attn_flashmla.py` | T9b-3c (`--attn-impl flashmla`, A3-class): every layer's `sparse_attn` on FlashMLA's `sparse_prefill_fwd` over our bf16 caches; the op loaded from vllm's `_flashmla_C` .so with `torch.ops.load_library` (no vllm Python, asserted before and after); head / head-dim / topk padding for small shapes; per-row `topk_length`; fp64 oracle and build-time self-check |
| `dg_moe.py` | T9a-1: `DGMaskedMoE` (`--moe-impl dg-masked`): the static MoE's routed GEMMs as 3 DeepGEMM masked grouped GEMMs over the same staging; `import_deep_gemm` (no vllm import); exact e8m0 → fp32 |
| `fused_glue.py` | T9b-2 round 1 (`--glue-impl fused`): Triton kernels for the zero/low-risk glue clusters (gather, DG act, SwiGLU+act_quant, MoE combine, mHC post, o-path RoPE+QDQ, q RoPE in the GEMV store), table-driven act_quant/silu, build-time self-check, per-cluster switches; torch mirrors on CPU |
| `gemv_mxfp8.py` | T9b-1: Triton MXFP8 GEMV `mxfp8_gemv(xq, xs, w, s, out)` for the fp8 Linears at <= 8 rows (`--fp8-impl gemv`); exact e8m0 pair scale; per-(N,K,M) tuning cache; `gemv_torch` = the same algorithm in torch (CPU) |
| `patches.py` | flag-gated monkeypatches: `kv_numerics_pg61` (model.py:761), `static_moe` (model.py:891-906, design #14), `fp8_linear_gemv` (model.py:196-205, design #16), `sparse_attn_bounded` (model.py:781 / kernel.py:392-403, design #18) |

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

## T9b-1: fp8 Linear layers on a Triton MXFP8 GEMV (`--fp8-impl gemv`, `gemv_mxfp8.py`)

Spec: `docs/HANDOFF-T9B1-GEMV.md`. Every `Linear` whose weight is fp8 with 32x32 e8m0 block scales goes through
`linear()`'s fp8 branch (model.py:196-205): `act_quant(x, 32, "ue8m0")`, then the reference `fp8_gemm` -- a 32x128 tile
GEMM that reads weights at ~0.3 TB/s at M = 1. At 1 row that is 290 calls/step (40 x {wq_a, wq_b, wkv, wo_b, shared
w1, w3, w2} + 8 indexer wq_b + 2 Engram wkv), ~16 ms in the eager anatomy. With `fp8_impl="gemv"` the same act_quant
runs, then one Triton launch: each program streams BLOCK_N weight rows once (KU 32-byte K blocks per row per
iteration), forms each 32-wide block's fp32 partial, scales it by the exact power of two 2^(sa+sb-254) built from its
bits, and adds it into an fp32 accumulator in K-block order -- the reference's order and scales; only the sum *inside*
one 32-wide block is ordered differently (tensor core there, `tl.sum` here), which can move a bf16 output by one ulp.

Flags: `StaticStep(..., fp8_impl="gemv")` / `Generator(..., fp8_impl=...)` / `first_light.py --fp8-impl gemv`
(combines with `--moe-impl dg-masked`). Routing is scoped: only inside the StaticStep's `_active()` (eager static
steps, warm-ups, capture), only fp8 + e8m0 weights, only 1..8 rows (`x.numel() // K` -- not `x.size(0)`, which is 1
for a whole prefill too), only the reference's MXFP8 globals and a bf16 default dtype; everything else (prefill,
eager reference, routed experts, bf16/fp32 Linears) calls the original `linear`. `fp8_impl="ref"` is T8, bit-identical.

Tuning: (BLOCK_N, KU, num_warps, num_stages) only, timed once per (N, K, BLOCK_M) under a CUDA graph on rotating
synthetic weights (HBM-streamed), in the eager warm-up (never under capture: an untuned shape there is an error).
Kept for the process (relaunch reuses it) and persisted to `$DSV41_GEMV_CACHE` when set; `DSV41_GEMV_TUNE=0` uses
`DEFAULT_CONFIG`. The candidates do not change the per-output K order; the box test asserts they are bit-identical.
First real-model capture with tuning on compiles ~9 configs x 8 shape keys (Triton cache mounted: once). Pre-tune with
the bench (`--tune-cache scratch/gemv-tune.json`) and pass `-e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json`.

**Mac:** `.venv/bin/python -m pytest -q tests/` -- `tests/test_gemv_cpu.py`: `pair_scale` vs the fp32 product of torch's
e8m0 decodes on all 65,536 code pairs and vs the reference's `(C*sa)*sb`; scale-row indexing n // 32 and K-block
pairing; the GEMV algorithm == a block-faithful emulation of the reference fp8_gemm bit for bit and within bf16
rounding of float64; rows independent for M = 1..8; the operand contract; the tuning cache; the patch's routing
conditions; the whole `fp8_impl="gemv"` StaticStep (alone and with dg-masked) == the reference forward under the same
GEMM math, with exactly 7/layer + indexer + Engram calls routed and the prefill/eager reference untouched; Generator /
first_light `--fp8-impl` plumbing and the "<impl> vs ref static-eager" drift row. Triton itself is untested on the Mac.

**Box, microbench** (real weights, every distinct fp8 shape, M = 1 and 6: µs, GB/s, mismatch vs the reference, config
invariance, gates; writes `scratch/gemv-bench.json` and the tuned configs `scratch/gemv-tune.json`):

    W=$HOME/hermes/gb300-dsv41-decode
    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$W":/work -w /work \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/gemv_bench.py --model-dir /ref

**Box, GPU test** (device pair scale on all code pairs; GEMV vs reference `fp8_gemm` on every real/mini/odd shape at
M in {1,2,6,8}, mismatch <= 0.5 %; all candidate configs bit-identical; graph replay == eager, no host sync, untuned
shape refused under capture; mini: gemv static self-replay + routed count + no sync, graph == static-eager / replay x2
/ fills-on recapture / relaunch, top-1 stream == ref, tuned capture + cache reload; informative step cells):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_gemv_gpu.py

**Box, first light** (gate: L1/L2 8/8, parity vs Pg61 in the A3 class, step cell <= 32 ms; the drift row is
"dg-masked+gemv vs ref static-eager"; run the bench first so the capture reuses its tuned configs):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --hot-per-layer 295 --mode graph --moe-impl dg-masked --fp8-impl gemv \
        --parity-levels --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-gemv.jsonl \
        --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit --json docs/first-light-gemv-<date>.json

`--levels-ref` x-proc rows compare runs with the same label: point it at a report made with the same `--moe-impl`
and `--fp8-impl`, and keep `DSV41_GEMV_CACHE` the same so both processes use the same configs.

**Glue census (Mac, CPU):** `.venv/bin/python bench/glue_census.py` -- every torch op a 1-row decode step dispatches on
a real-structure synthetic model (40 layers, real ratios/sources/Engram), by layer type and code region, for the
eager reference, T8, T9a-1+T9b-1 and T9b-2 fused-glue paths (`scratch/glue-census.json`); the numbers behind
`docs/GLUE-AUDIT.md`. With both static-dg-gemv paths it prints launches removed per cluster vs the audit.

## T9b-2 round 1: the zero/low-risk glue clusters as fused kernels (`--glue-impl fused`, `fused_glue.py`)

Spec: `docs/HANDOFF-T9B2-GLUE-R1.md`; cluster numbers from `docs/GLUE-AUDIT.md`. `glue_impl="fused"` replaces, per
layer, the glue between the GEMMs with one Triton launch each, bit-identical to `glue_impl="ref"` (the default: the
static path above, unchanged) with the same moe/fp8 impls:

| cluster (`describe()` name) | audit | replaces | applies to |
|---|---|---|---|
| `gather` | #11 + #9 | 6 x `t8_gather_rows` (+ dg: 3 x e8m0->fp32 `convert_scales`, the `ids_step` copy) -> `t9b2_gather_tables` | both moe impls |
| `dg_act` | #9 | act_quant(x) + `load_act(1)` copies -> act_quant straight into row 0 of every DeepGEMM group | dg-masked |
| `swiglu_routed` | #2 + #9 | `.float` x2, clamps, silu, `*`, `* weights`, `.to(bf16)`, act_quant, `load_act(2)` | dg-masked |
| `swiglu_shared` | #2 | the shared expert's SwiGLU + bf16 + w2's act_quant (w1/w3/w2 keep their GEMM path) | fp8 shared experts |
| `moe_combine` | #8 | `zeros_like`, k x `+=`, `+= shared`, `type_as` | dg-masked |
| `hc_post` | #7 | `Block.hc_post` | always |
| `o_rope_qdq` | #10 | inverse RoPE + act_quant QDQ on the attention output (in place) | always |
| `q_rope_epilogue` | #13 | q RoPE moved into the wq_b GEMV's store (`gemv_mxfp8.linear_fp8_rope`; split-K: in the reduce) | fp8 gemv |
| `concat_free` | #16 | not done in round 1 (window ring per layer vs compressed caches/topk shared per group) | -- |

How it stays exact: the act_quant scale code is read from a 65,536-row table the reference act_quant produced for
every bf16 amax; silu is read from a table of torch's own `F.silu` over every bf16 value (+ the limit); the rest are
the reference's fp32 ops in the reference's order (`enable_fp_fusion=False`; the RoPE helpers are explicit
`mul.rn`/`fma.rn`). Where torch's order depends on how the device compiled it -- the complex multiply's FMA form,
the 4-term `torch.sum` in `hc_post` -- the build-time self-check (`FusedGlue.calibrate`, eager, before any capture)
picks the form that matches torch on random and cancellation-crafted data, then verifies every enabled cluster
end to end against its reference ops at the model's shapes. A cluster that matches no form or fails a check is
**off** (reason in `st.glue.describe()`, recorded in the first-light report as `fused_glue`) and runs today's ops;
nothing is approximated. `DSV41_GLUE_DISABLE=gather,hc_post` forces clusters off (bisecting a mismatch).

Flags: `StaticStep(..., glue_impl="fused")` / `Generator(..., glue_impl=...)` / `first_light.py --glue-impl fused`
(threaded like `--moe-impl`/`--fp8-impl`). With `--parity-levels` one extra static-eager run with glue `ref` and the
same moe/fp8 impls gives the gated row **"fused glue vs ref glue (same moe/fp8 impl)"** (level G); the drift
reference stays pure T8. Census receipt (Mac, real structure): static-dg-gemv 6,725 -> 4,645 launches/step, 2,080
removed; each implemented cluster removes exactly the audit's count (#16's 76 deferred).

**Mac:** `tests/test_fused_glue_cpu.py` -- the torch mirrors vs the unfused op sequences (clamp edges incl. the bf16
neighbours of the limit, e8m0 boundaries, every finite bf16 value through act_quant/QDQ, an exact fp32 fma vs
Fraction arithmetic, the three RoPE forms and hc orders distinguishable); the real kernel bodies of all six
fused_glue kernels and the GEMV/reduce RoPE epilogue through `tests/triton_shim.py` == the mirrors bit for bit
(`FG_SHIM_MODULE=<mutated copy>` for sensitivity checks); fused == ref StaticStep logits on both mini checkpoints for
all four moe/fp8 pairs with the same GEMV routing counts; `DSV41_GLUE_DISABLE`; the self-check switching broken
clusters off; Generator / first_light / census plumbing. The gather kernel's address loads are GPU-only (emulated).

**Box, GPU test** (kernels vs reference at real shapes, the RoPE forms / hc order that match on the device printed;
one-launch gather vs 6 gathers on real pointer tables; mini/mini4 x ref/ref and dg-masked/gemv: every applicable
cluster on, fused == ref static-eager over prefill + 10 steps both parities, no host sync, graph == static / replay
x2 / fills-on / relaunch; informative step cells):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_fused_glue_gpu.py

**Box, first light** (gates: row G 8/8, L1/L2 8/8, output byte-identical to the T9b-1 run, step cell <= 30 ms; same
flags as T9b-1's `docs/first-light-gemv-20261106.json` plus `--glue-impl fused`, same tuned GEMV configs -- copy
them to scratch first, the cache file is written back: `cp docs/gemv-tune-gb300-20261106.json scratch/gemv-tune.json`):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --hot-per-layer 295 --mode graph --moe-impl dg-masked --fp8-impl gemv \
        --glue-impl fused --parity-levels --max-pos 4095 --fixture w0/results/phase1/parity-A.jsonl \
        --out scratch/first-light-fused.jsonl --max-new 256 --reps 2 --step-cell 512 --no-fill-uninit \
        --json docs/first-light-fused-<date>.json
    cmp scratch/first-light-fused.jsonl w0/results/first-light-cand-gemv.jsonl && echo BYTE-IDENTICAL

Then check `fused_glue.clusters` in the report: any cluster "off" is named with its reason (it ran today's ops).


## T9b-3a: `sparse_attn` bounded to its valid blocks (`--attn-impl bounded`) + the gather at HBM rate

Spec: `docs/HANDOFF-T9B3A-ATTN-BOUND.md` (memo `docs/T9B3-ATTN-DESIGN.md` sections 0.2 / 5). Both bit-identical.

**Task A.** The static step passes 640 indices (128 window + 512 compressed) to `sparse_attn` on every compressed
layer; at fixture lengths only 128 + min(512, T/r) of them are valid (the -1 tail of static_step #9), and the kernel
walks all 10 blocks of 64 serially in one CTA. `attn_impl="bounded"` computes, where the indices are built
(`_compress_kv`, once per (ratio, k) per step, device ops only), `nblk = ceil((128 + min(k, compress_lens)) / 64)`
and runs `runtime/kernels/sparse_attn_bounded.py`: kernel.py's `sparse_attn_kernel` text with the block loop bounded
by `nblk[b, m]`. Why that is exact: a block whose 64 indices are all -1 adds `-inf` scores, so the running max is
unchanged (reduce_max with clear=False), the rescale is exp(+0) = 1, the probabilities are exp(-inf) = +0, and the
sum, the bf16 P and the PV accumulate change nothing (walked op by op in the module docstring, kernel.py:358-380);
the static step already depends on this today (640-wide static == narrow eager reference, T8 L0). Valid entries
are a sorted prefix, so the count never cuts one (a candidate mask can only make it an over-estimate). Ratio-0
layers keep the reference call. Long context (T/r >= 512) gives 10 = the full width: correct, no saving.
TileLang 0.1.12 must compile a loop bound read from a tensor; three spellings are tried in order at StaticStep build
(`extent`: the Pipelined extent, `serial`: a T.serial loop, `guard`: the fixed Pipelined loop with an `if`), each
self-checked on the model's shapes against the reference at the full AND the narrow width (`torch.equal`, 40 index
lists incl. widths 128/278/428/640, leading -1 windows, a 4-row launch); the first that passes is used, none -> the
step stays on the reference and says so (`st.attn_describe()`, report `attn_bound`, summary `attn bounded(OFF)`).
Kernel name in nsys: `sparse_attn_bounded_kernel_<variant>_` (2 ratio-0 layers keep `sparse_attn_kernel_`).
Expected (memo, est.): 128 µs -> ~30-60 µs per call at fixture length, ~2.5-3.5 ms/step.

**Task B.** `t8_gather_rows` / `t9b2_gather_tables` loaded uint8 through a row address read from memory (alignment
unknown to Triton: one byte per lane, ~1.3 TB/s). They now copy int64 words through addresses declared 16-B
aligned (`tl.multiple_of`): 16-B vectors per lane, unmasked when a row is a whole number of 8 KiB blocks (every real
table). Same bytes, so nothing numeric changes; the old kernels stay as `t8_gather_rows_u8` / `t9b2_gather_tables_u8`
(GPU test reference, bench baseline, `DSV41_GATHER_IMPL=u8`). Row bytes and destination alignment are checked per
call (metadata), source row addresses at StaticStep build (the pool's 256-B host blocks and torch's allocations
always satisfy it). Tuning knobs: `DSV41_GATHER_BLOCK_BYTES` (8192), `DSV41_GATHER_WARPS` (4).

Flags: `StaticStep(..., attn_impl="bounded")` / `Generator(..., attn_impl=...)` / `first_light.py --attn-impl bounded`
(threaded like `--glue-impl`). With `--parity-levels` one extra static-eager run with attn `ref` and the same
moe/fp8/glue impls gives the gated row **"bounded attn vs ref attn (same moe/fp8/glue)"** (level A); L0 stays gated
for ref moe/fp8 (bounded is not a variant).

**Mac:** `tests/test_attn_bound_cpu.py` -- the count vs the reference width 128 + min(512, T/r) for every position
1..4,096 at both ratios and the rounding edges; the count vs the last valid block of the static step's real index
lists (ties, candidate masks); the kernel's algorithm in torch bounded == full == narrow width; the self-check's
pick and rejection; the patch's routing; bounded StaticStep == ref StaticStep == the reference forward on the
emulated mini checkpoints (incl. mini4 at index_topk 256, where blocks are really skipped) with every bounded call's
count checked against the host formula; the build-time gather alignment check; Generator / first_light plumbing.
TileLang and the Triton gathers themselves only run on the box.

**Box, GPU test** (bounded vs reference kernel at real shapes, every spelling reported; vector vs byte gather on
real-size HBM + UVA rows and on mini4's tables; mini / mini4 / mini4w bounded == ref static-eager, graph == static,
replay x2, fills-on, relaunch, no host sync; mini4w with dg-masked/gemv/fused too; informative step cells):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      --cap-add IPC_LOCK -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_attn_bound_gpu.py

**Box, gather bench** (real layer-0 expert rows: 36 hot in HBM, 12 cold in registered host memory; M = 6 ids, hot and
cold cases separately; byte vs vector kernels, a block/warps sweep, the HBM->HBM copy ceiling; every arm verified
`torch.equal` first; gate line: best hot one-launch gather >= 3 TB/s copied):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v "$W":/work -w /work \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/gather_bench.py --model-dir /ref --layer 0 --json docs/gather-bench-<date>.json

If the best sweep config beats the default, rerun first light with `-e DSV41_GATHER_BLOCK_BYTES=<b> -e
DSV41_GATHER_WARPS=<w>` (the gather config is not part of the numerics).

**Box, first light** (gates: row A 8/8, L1/L2 8/8, output byte-identical to the R1 run with the same flags, step
cell <= 26 ms; the R1 flags plus `--attn-impl bounded`; same tuned GEMV configs copied to scratch first):

    cp docs/gemv-tune-gb300-20261106.json scratch/gemv-tune.json
    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph \
        --moe-impl dg-masked --fp8-impl gemv --glue-impl fused --attn-impl bounded --parity-levels --max-pos 4095 \
        --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-attn.jsonl --max-new 256 --reps 2 \
        --step-cell 512 --no-fill-uninit --json docs/first-light-attn-<date>.json
    cmp scratch/first-light-attn.jsonl <the R1 run's rep-1 JSONL, same map/flags> && echo BYTE-IDENTICAL

Then check `attn_bound` in the report (`on: true`, the spelling, every self-check row "equal") and `gather`
(`impl: vec`, `vec_error: null`). Any spelling that failed to compile is listed with its error.


## T9b-3c: the main sparse attention on FlashMLA (`--attn-impl flashmla`, A3-class)

Spec: `docs/HANDOFF-T9B3C-FLASHMLA.md`; the adoption receipt is the probe (`bench/flashmla_probe.py`,
`docs/flashmla-probe-20261106.json`: ADOPT; loads with no vllm import; arm C fp64 error <= 1.05x the reference
kernel's, nearer the bf16-q oracle in 18/18 cases, 13.8-14.4 us/call vs 125-154 in a 40-call graph, replay-equal).
`ref` stays the default.

**What runs.** `attn_impl="flashmla"` sends EVERY layer's `sparse_attn` (2 ratio-0 + 38 compressed) to
`runtime/attn_flashmla.py` `sparse_attn_flashmla`, i.e. `torch.ops._flashmla_C.sparse_prefill_fwd` (the probe's arm C)
over the static step's bf16 kv / int32 indices / fp32 sink / softmax_scale exactly as they are -- no new cache writer,
no 528/288 B records, no scheduler metadata. Compressed layers pass `topk_length = 128 + min(k, compress_lens)` per row
(the eager reference's own width; computed where the indices are built and published with them, like T9b-3a's block
count, of which it is the unrounded form); the compressed indices are a sorted prefix with a -1 tail, so the length
drops only -1 slots. Ratio-0 layers pass no length. The real shapes (64 heads, 512, widths 128 / 640) go through with
no copy; smaller shapes (the mini checkpoints) are zero-padded to h_q 64, d 512 and topk % 128, which leaves the
kept heads' values unchanged. Not bit-identical to the reference (different tiling, exp2 softmax, sink as
exp(lse)/(exp(lse)+exp(sink))); q stays bf16, so the class is A3 (as the GEMV), not A1.

**Loading.** `<dist-packages>/vllm/_flashmla_C*.so` by glob, `torch.ops.load_library`; 'vllm' must be absent from
`sys.modules` before and after (else refused: "stop and say so"). Nothing imports `flash_mla_interface.py` (it does
`import vllm._flashmla_C`). This is the sanctioned reading of decision 3 for this binary (handoff rules).

**Routing / fallback.** No monkeypatch: the static step owns the call site (`StaticStep._sparse_attn`), so the prefill
and the eager reference never reach FlashMLA. `DSV41_FLASHMLA_TOPK_LENGTH=0` passes no per-row length (the -1 slots
are masked instead, as the probe's arm C ran) -- a knob in case the kernel mishandles lengths; the GPU test reports
whether length == masking bitwise. At StaticStep build: load, then a self-check on the model's shapes (the
T9b-3a index lists: 40 compressed lists with lengths + the window-only lists, a 4-row call) against the reference
kernel and an fp64 oracle -- gates finite and error <= 2x the reference's; reports the max/mean ratio, mismatch %,
length == -1-masking, rows independent. A failed load or check leaves the reference in force and says so
(`st.attn_describe()`, report `attn_bound`, summary `attn flashmla(OFF)`).

Flags: `StaticStep(..., attn_impl="flashmla")` / `Generator(..., attn_impl="flashmla")` / `first_light.py --attn-impl
flashmla`. With `--parity-levels` the step is a variant (label gains `flashmla`: L0 static rows reported, not gated;
the gated "ref static-eager vs reference-eager" row; the "<label> vs ref static-eager" drift row), and the attn-`ref`
static-eager run with the same moe/fp8/glue impls gives the drift row **"flashmla vs ref attn (same other impls)"**
(level A, not gated; first token divergence + mean |delta logit|; `parity_levels.attn_drift`).

**Mac:** `tests/test_attn_flashmla_cpu.py` -- the length vs the reference width for every position 1..4,096 at both
ratios (and vs the T9b-3a block count); exactness on the static step's real index lists; argument construction (no
copy at real shapes, padding at mini shapes, batch offsets); the wrapper through a torch emulation of the op's contract
== the fp64 oracle, length == masking, rows independent; the load guards (vllm already imported, no .so, a load that
pulls vllm in, op missing); the self-check's accept / reject; StaticStep routing on the emulated minis (every layer,
exact lengths) and the load-failure fallback; Generator / first_light plumbing. The op itself only runs on the box.

**Box, GPU test** (load without vllm; flashmla vs the reference kernel at real shapes, positions 300 / 1100 / 4000,
1 and 6 rows, ratio 0/2/1: mismatch %, max |delta|, fp64-oracle ratio <= 1.1x; a 40-call graph replay x2 == eager at
b = 1/6 and p = 300/1100/4000 with us/call for flashmla vs the reference and, at 300 and 4000, FlashMLA's decode
kernel over 528/288 B records (the B-vs-C question: does C's cost grow with context where B's does not); mini / mini4
/ mini4w (+ dg-masked/gemv/fused) static-eager runs, graph == static, replay x2, fills-on, relaunch, no host sync,
top-1 stream == attn ref over 10 steps; informative step cells):

    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
      --cap-add IPC_LOCK -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_attn_flashmla_gpu.py

**Box, first light** (gates: L1/L2 8/8, self-replay 8/8, drift row "flashmla vs ref attn" <= 0.05 mean, parity vs
Pg61 within the A3 class (2/8 identical, 0 weighted flips), step cell <= 22 ms; the default path's flags plus
`--attn-impl flashmla`):

    cp docs/gemv-tune-gb300-20261106.json scratch/gemv-tune.json
    docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 \
      --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
      -v "$W":/work -w /work -v "$W/w0":/work/w0 \
      -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \
      -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
      -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
      -v "$HOME/dsv41/w0/jit-cache/dg":/root/.deep_gemm -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json \
      --entrypoint python3 vllm/vllm-openai:v0.31.0 \
      bench/first_light.py --model-dir /ref --map docs/maps/hot-v26ar-seed-295.json --mode graph \
        --moe-impl dg-masked --fp8-impl gemv --glue-impl fused --attn-impl flashmla --parity-levels --max-pos 4095 \
        --fixture w0/results/phase1/parity-A.jsonl --out scratch/first-light-flashmla.jsonl --max-new 256 --reps 2 \
        --step-cell 512 --no-fill-uninit --json docs/first-light-flashmla-<date>.json

Then check `attn_bound` in the report (`impl: flashmla`, `on: true`, `load.vllm_modules: []`, the self-check's
`max_err_ratio`), the level-A drift row (`parity_levels.attn_drift.mean_abs_dlogit`), and the parity table vs Pg61.
Kernel name in nsys: FlashMLA's sparse prefill kernel (40 per step); no `sparse_attn_kernel_` should remain.
