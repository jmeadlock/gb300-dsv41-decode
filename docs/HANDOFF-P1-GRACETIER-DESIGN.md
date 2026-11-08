# HANDOFF — P1 design desk: a Grace cold-expert tier for Garner's vLLM recipe

Desk worker. Read in this order, in full: `docs/PORT-PLAN.md`; `reference/garner-8043a681/README.md`,
`docs/ARCHITECTURE.md`, `docs/ENGINEERING-RULES.md`, `docs/WHAT-DID-NOT-WORK.md`; `fastpath/components.py`,
`fastpath/fastpath_peer3.py`, `fastpath/fastpath_coldskip.py`, `fastpath/fastpath_boundary.py` (the w4 router: gate
GEMM + top-k + hot/cold split + cold-row publish + finish seam), `fastpath/fastpath_megamoe.py`,
`fastpath/hook/mega_peer_hook.py`, `fastpath/hook/peer_tier2.py`, `fastpath/hook/peer_fused2.py`, and whatever in
`fastpath/peer3/` is Python. Then ours: `runtime/loader.py` (host tier: cudaHostRegister chunks, UVA alias),
`runtime/staging.py` or wherever `m1_stage_rows` / the kernel-overlap stage lives (grep `DSV41_STAGE`),
`runtime/residency.py`, `runtime/moe_dg.py` or the dg-inplace path (grep `dg-inplace`), and the receipts
`docs/M3-S3-POOL-SF-LAYOUT.md`, `docs/C2C-PROBE-20261106.md`, `docs/A2-RESIDENCY-DESIGN.md`.

## The question
Garner's recipe computes the 129 cold experts/layer on an RTX PRO 6000 (peer tier v3) and `coldskip` never
loads them on the GB300. Design a component **`gracetier`** that lets the same vLLM server run on a Station with
**no RTX**: cold experts pinned in Grace memory, read over C2C, added at the seam peer3 uses today.

## Write only `docs/P1-GRACETIER-DESIGN.md`, with these sections
1. **The seam, exactly.** What the w4 router publishes for cold rows (tensor shapes, dtype — MXFP8 rows + SF?, expert
   ids, row ids), where and when the finish adds the cold outputs back (`_finish2` vs peer3 finish, fused-finish,
   POST_FUSE, COPY_AHEAD), which T ranges each path covers (1–16, 17–64, >64 incl. prefill), and how the drafter's
   layers are treated. Cite `file:line`. Name the stream/graph-capture context each seam runs in.
2. **Two candidate consumers**, both GB300-side, both landing at that seam:
   (a) **compute-from-UVA**: a GB300 kernel (DeepGEMM masked grouped fp8×fp4 or the reference `fp4_gemm`) whose
   weight operand is the UVA alias of the pinned cold expert — our measured kernel read rate from Grace is ~230 GB/s,
   copy engine ~390 GB/s; per cold expert 18.8 MB (w1/w3/w2 + SF); (b) **stage-into-slots**: our kernel-overlap
   staging copies the cold expert into spare slots of a pool and MegaMoE/DeepGEMM computes it with the hot ones
   (our runtime's dg-inplace shape; SF 'beside' layout). For each: where it hooks, what MegaMoE needs to accept
   (dynamic slot index per cold expert?), graph-capture compatibility, numerics class vs peer3 (peer3 accumulates
   bf16 on the RTX — note Garner's own "changes numerics" tag), and a bytes/time model per 6-row step using
   Pg61's measured 1.6 distinct cold experts per 6-row layer-call (`docs/data/peer3_stats.json` by_T['6']) and per
   1-row step. State the model's inputs as [M] measured or [E] estimated, with the receipt.
3. **Loading.** How `coldskip`'s read filter becomes "read, then pin to Grace" without doubling host RAM: Engram
   tables already pin ~190 GiB; cold experts are ~90 GiB; host is 494 GiB with a 120 GB KV overflow — do the
   budget, say what has to move (e.g. `DS41_OFFLOAD_GB`).
4. **Residency.** Where our swap policy (`runtime/residency.py`) would run (host thread? between steps?) and what
   it needs from MegaMoE's pool to swap a hot↔cold pair without re-capturing graphs. If that is impossible in the
   first version, say so and ship the static map.
5. **Mini-test plan for the box** (one ~1 h window, server not needed): load 1 layer's cold experts to Grace,
   run the candidate consumer against the reference expert output on captured activations, report µs per cold
   expert at T=1/6/16/64 and bit-equality or flip class. Name the exact files the coding worker will create.
6. **Risks**: what breaks C32 (the recipe's own lesson: narrow kernels that pass C≤8 die at C32), graph capture of
   UVA reads, Garner's `a12` memory guard, the watchdog, anything in `WHAT-DID-NOT-WORK.md` that already killed a
   similar idea.
7. **Verdict line**: `P1-DESIGN: <a|b|both> recommended; modeled exposed cold ms/step 1-row=<x> 6-row=<y>; first build=<file list>`.

## Rules
No ssh, no docker, no network, no other edits, no git commit, never import vllm. `reference/` is read-only.
No invented numbers: every figure carries [M] + receipt or [E] + the arithmetic. If a fact is not in the files,
write "not in tree" rather than guessing. Garner's component system is the integration target, not a rewrite of
vLLM; stay inside its hook mechanics.
