# HANDOFF — P1 mini-test coding worker (gracetier first build)

Read in full, in order: `docs/PORT-PLAN.md`; `docs/P1-GRACETIER-DESIGN.md` (all of it — §2.2 is the design you
build, §5 is the test plan you implement, §0 lists input inconsistencies you must not paper over); then the files §5
names as patterns: `runtime/host_tier.py` (host registration, `__cuda_array_interface__` view), `runtime/ckpt.py`
(`read_tensor_into`), `runtime/dg_mega.py` (MegaMoE call forms, `transform_weights_for_mega_moe`, `stage_emulated`
contract), `runtime/staging_overlap.py` and `runtime/static_kernels.py` (our staging kernel), `bench/dg_mega_numerics.py`
(8-tag activation windows), `bench/box/m3n1.sh` (box runner shape: lease, window guard, image, JSON receipt),
`tests/gpu/` (how GPU tests are skipped on the Mac), `reference/garner-8043a681/fastpath/fastpath_boundary.py`
(`router()` interface; read-only), `references` in the skill are not available to you — everything is in the tree.

## Build exactly the §5 file list
`runtime/gracetier_vmm.py`, `runtime/gracetier_pool.py`, `runtime/gracetier_stage.py`, `bench/gracetier_minitest.py`
(subcommands `vmm | numerics | timing | r8` + `all`, one JSON receipt at a path given by `--json`, default under
`scratch/`), `bench/box/gracetier_minitest.sh`, `tests/test_gracetier_cpu.py`, `tests/gpu/test_gracetier_gpu.py`.
Steps M0–M7 of §5 are the probe's phases; each phase prints its [M] numbers and writes them to the receipt. M1 is the
gate: if `torch.equal` fails there the probe must stop and say so (exit code 3), not continue into M4.

## Rules
- Mac has no GPU: every host-side constructor pins `device="cpu"` and fp32 explicitly; use `.venv/bin/python -m pytest tests/ -q`
  and make the suite green. Mark GPU tests with the existing skip pattern.
- `deep_gemm` and `torch` only on the box path; **never import vllm**. The Garner `router()` may be imported from
  `/garner/fastpath` (mounted read-only by the runner) only inside M2's router check, guarded so a missing module
  reports `not run`, never a crash.
- Receipts to `scratch/` on the box, never `docs/`. The runner takes `--layer` (default 10) and `--json`.
- No ssh, no docker, no network, no git commit, `reference/` untouched. Behaviour changes to existing runtime files:
  none — new files only (plus a one-line import in `tests/gpu/conftest.py` if needed).
- No invented numbers in docstrings or output; the probe measures or says `not measured`.
- Finish with: a list of created files, the pytest count, and the line `P1-MINITEST: code ready; cpu tests <n> pass; gate=M1 torch.equal; phases=M0..M7`.
