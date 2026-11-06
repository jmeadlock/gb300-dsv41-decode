# runtime/ — M0 loader, placement, host tier, expert pools (T2+T3)

Standalone: no vllm import anywhere (charter decision 3). The reference `model.py` is imported unchanged from
`reference/deepseek-inference/` (`runtime/refmodel.py`); nothing is monkeypatched, so there is no `patches.py`.

| file | what |
|---|---|
| `ckpt.py` | shard index (headers scanned once), convert.py's rename map (`convert_name`, line-for-line from convert.py:104-126), `read_tensor_into` / `read_into` = `preadv` into the caller's buffer + `posix_fadvise(DONTNEED)` per 64 MiB chunk |
| `placement.py` | `place(name, shape, dtype, cfg, map) -> Tier.HBM / HOST / SKIP`, hot maps (JSON `{layer: [hot ids]}` or `--hot-per-layer N`), the dry-run report CLI |
| `host_tier.py` | `HostBuffer`: exact-size anonymous mmap (+ MADV_HUGEPAGE), `cudaHostRegister` in chunks of 64 GiB or less with rollback, a UVA CUDA view built from `__cuda_array_interface__`, `release()`, MemAvailable checks |
| `expert_pool.py` | hot experts in per-layer HBM slot pools `[H,N,K/2]` + `[H,N,K/32]`; cold experts in one registered host pool, same layout; `loc[L][E] = (slot, tier)` int32 on device; binds the reference `Expert`s to zero-copy views (M0a zero-copy arm); host-side cold-hit counters |
| `loader.py` | `build_and_load(model_dir, map, device)` and a CLI: meta construction, tiered streaming load, refusal on any key/shape/dtype mismatch, `--verify` sha256 of every placed tensor |
| `refmodel.py` | imports the reference modules; `cpu_stubs=True` (Mac tests only) stands in for tilelang/PIL; Python < 3.10 compiles them with `from __future__ import annotations` (the file is untouched) |

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

    docker run --rm -v "$HOME/hermes/gb300-dsv41-megakernel":/work -w /work \
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
      -v "$HOME/hermes/gb300-dsv41-megakernel":/work -w /work \
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
