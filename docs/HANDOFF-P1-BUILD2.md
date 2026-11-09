# HANDOFF — Build 2 coding worker: `gracetier` component for Garner's recipe (card GT1)

Read in full, in order: `docs/PORT-PLAN.md`; `docs/P1-GRACETIER-DESIGN.md` §1, §2.2, §3, §4, §5 "Build 2", §6;
`docs/LEDGER.md` tail (the mini-test results of 2026-11-08: M1 VMM aliasing torch.equal, M4 (b) bit-identical,
stage 46–50 µs/expert, (a) dead); the mini-test code you build on: `runtime/gracetier_vmm.py`,
`runtime/gracetier_pool.py`, `runtime/gracetier_stage.py`, `bench/gracetier_minitest.py`; then Garner's tree
(read-only): `reference/garner-8043a681/fastpath/components.py`, `fastpath_hook.py`, `fastpath_peer3.py`,
`fastpath_coldskip.py`, `fastpath_megamoe.py`, `fastpath_boundary.py` (§2.2's line cites: `_moe_eligible` ~1158,
publish selection 1311–1339, router call ~1365, finish 1380–1388), `hook/mega_peer_hook.py` (`_init`,
`_weight_loader`, `_finalize_weights`, `_forward`), `fastpath/sitecustomize.py`, `serve/serve.sh` (how the fastpath
dir and rowmap are mounted; the RTX requirement at line ~124 and the peer3 tree check at ~83),
`launcher/launch-pg61.sh` + `pg61-policy.env` (how our lane launches it: `DS41_FASTPATH_COMPONENTS_OVERRIDE`,
`DS41_OFFLOAD_GB`, `DS41_KV=fp8_ds_mla`, `DS41_MAXLEN`).

## Deliverables (new files only; nothing under `reference/` changes)
Everything lands under `port/garner/` in this repo, applied on the box to a **copy** of Garner's tree
(`~/dsv41/gracetier/tree`, never the production clone):

1. `port/garner/fastpath/fastpath_gracetier.py` — the component (`@register("gracetier", ...)`, `defer_active=True`
   like peer3). Responsibilities per §2.2/§3: (i) MegaMoE built at G = 384 per layer, hot groups 0..H−1 in HBM,
   cold groups H..383 on the shared VMM scratch (one physical allocation mapped after every layer's hot block,
   private straddle pages), SF resident for all 384; (ii) loading: cold tensors are read from the shards into the
   registered Grace pool (MegaMoE layout) instead of skipped — do the host-RAM budget in a boot log line
   (Engram + cold pool + KV overflow vs MemAvailable) and refuse to boot if it does not fit; (iii) the all-resident
   row_map (`row[e]` = hot slot or H + cold-local id) handed to the router; (iv) `gt_stage` on the current stream
   right after the router and before MegaMoE for T ≤ 16 (boundary path) and inside the MegaMoE forward wrap for
   T > 16 (peer3's wrap pattern); no finish/combine seam. Capturable (kernel copy, fixed VAs).
2. `port/garner/patches/0001-boundary-gracetier.patch` and `0002-components-gracetier.patch` — minimal diffs
   against 8043a681 (`git diff` format): boundary gracetier mode at the four sites §2.2 names; `components.py`
   registers `fastpath_gracetier`. A unit test applies them to a copy of `reference/garner-8043a681/fastpath` with
   `patch --dry-run` and asserts the hunk count.
3. `port/garner/make_rowmap_allres.py` — writes the all-resident row_map JSON from `hook/rowmap-mix-v1-h255.json`
   (H = 255 first; `--hot-map docs/maps/hot-v26ar-seed-295.json` as an option for later).
4. `port/garner/serve-gracetier.patch` — the smallest change to `serve/serve.sh` that lets it boot with **no RTX**
   when `DS41_FASTPATH_COMPONENTS_OVERRIDE` contains `gracetier` (skip the RTX requirement, the peer3 tree check,
   the sidecar start/wait, the peer shm cleanup). Everything else byte-identical.
5. `bench/box/card-gt1.sh` — the card runner, modeled on `bench/box/window-gt-mt-1108.sh` + `launch-pg61.sh`:
   preflight (no containers, GPU idle, `GT1_WINDOW` set); LANE-OWNER + MAINTENANCE lines; `launch-pg61.sh down`;
   build the gracetier tree copy (rsync Garner's clone → `~/dsv41/gracetier/tree`, apply the three patches, copy the
   component, write the rowmap); boot via a cloned launcher `launch-gt1.sh` (policy.env = Pg61's with `coldskip,peer3`
   → `gracetier`, `DS41_OFFLOAD_GB=48` kept, same KV/maxlen, port 30006, served name `dsv41-flash-gracetier`);
   wait for `/v1/models`; gates in order, each to `scratch/gt1-*.json/log`: (G1) boot log shows the gracetier
   ACTIVE line + host-RAM budget; (G2) greedy fixture `w0/w0_drive.py parity` → TF flips vs `w0/results/phase1/parity-A.jsonl`
   (report flips/weighted/max|Δ|; A3 class is expected since peer3 is the side that changes numerics); (G3)
   `w0/w0_drive.py` step cell → ms/step and ms/token at 1 stream, 3 reps; (G4) C32 (32 reqs × ≥128 tokens, our
   `bench/` or `w0/` driver that already does it — find it; if none exists, write `bench/c32_smoke.py` with the
   OpenAI client against `/v1/completions`) — must not crash; (G5) prefill/TTFT on the 8 fixture prompts; then
   `down`, restore Pg61 via a clone of `restore-pg61-after-gt-mt-1108.sh` (`gt-mt-1108` → `gt1`), archive
   MAINTENANCE, LANE-OWNER END line. Everything timestamped; receipts to `scratch/` only.
6. `tests/test_gracetier_port_cpu.py` — CPU contracts: patches apply dry-run with the expected hunk counts; row_map
   is a partition of 0..383 per layer with every entry ≥ 0 and hot ids mapping to 0..H−1; the host-RAM budget
   function's arithmetic; `serve-gracetier.patch` leaves serve.sh byte-identical outside its hunks (count lines).

## Rules
Mac has no GPU; `device="cpu"` and fp32 on every host-side constructor; `.venv/bin/python -m pytest tests/ -q`
green. **Never import vllm on the Mac**; the component imports vllm only inside the hook functions that run in the
container (follow peer3's import discipline). No ssh, no docker, no network, no git commit. `reference/` untouched.
No invented numbers. Where Garner's behaviour is inferred rather than read (the router's acceptance of an
all-≥0 map was confirmed in M2; anything else), say so in a comment.
Finish with: file list, pytest count, and `P1-BUILD2: code ready; cpu tests <n> pass; card=bench/box/card-gt1.sh; gates=G1..G5`.
