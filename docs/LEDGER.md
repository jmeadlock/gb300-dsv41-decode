# Ledger (append-only; orchestrator = Milo on anthropic/claude-fable-5-1 unless stated)

- 2026-11-06 10:45 CST — Charter signed; cutover executed (consumers → Spark `:8888`; Echo gateway restart by James 12:43). `CUTOVER.md`.
- 2026-11-06 11:27 — **W0 phase 1**: Pg61 on our driver 7.72–7.75 ms/step (3 × 512 greedy, 2.6–2.9 tok/step); parity A vs A2 7/8 byte-identical, `code` diverges at token 121. `w0/W0-RESULTS.md`.
- 2026-11-06 11:27 — **W0 phase 2 FAILED**: nsys-wrapped Pg61 stalls after `/start_profile` (GPU 0 %, EngineCore 163 % CPU). One boot lost to my own preflight grep.
- 2026-11-06 12:30 — Blog live and listed: https://al-engr.com/gb300-dsv41-megakernel.html (writer: one headless session, anthropic/claude-fable-5.1 via Nous). Repo public 12:45; post updated 12:46.
- 2026-11-06 13:00 — **W0b FAILED the same way** with `--cuda-graph-trace=graph`, stalling *before* `/start_profile` → the stall is nsys + peer-tier-v3 spin, not the capture range. **Pg61 anatomy PARKED** (not on M0's critical path). Lane DARK.
- 2026-11-06 13:07 — M0 port plan written by desk worker (anthropic/claude-opus-5-5, Max OAuth): `docs/M0-PORT-PLAN.md`, verdict `M0-PLAN: ready`. Base = DeepSeek's reference `inference/` shipped in the checkpoint (MIT), vendored.
- 2026-11-06 13:10 — T1 env verified on the box: vLLM v0.31.0 image as library image; 7 reference TileLang kernels on sm_103; model-exact configs on real bytes (real-expert `fp4_gemm` rel_err 0.0017 at M=1/6); `python model.py` 80 s. `docs/ENVIRONMENT.md`.
- 2026-11-06 13:13 — T0: checkpoint `encoding/` reproduces Pg61 prompt tokens 8/8.
- 2026-11-06 13:18 — **D1 memo** (desk, opus-5-5): Pg61's `fp8_ds_mla` stores MXFP8 of the *un-FP4'd* latent; the reference attends E2M1/E4M3-16 QDQ (= vLLM's own "reference" 288 B NVFP4 record). Runtime default = reference numerics with a switch. Reopens decision-4 wording for James. `docs/D1-KV-NUMERICS.md`.
- 2026-11-06 13:22 — **C2C probe** on real expert bytes: UVA path bit-identical; copy engine 390 GB/s; register 7.3 GB/s; Pg61 static map = 2.06 cold experts/layer at 6 rows → ~3.9 ms/step over C2C alone. 13:28 overlap probe: H2D fully concurrent with compute (0 % GEMM slowdown). **Recipe A needs adaptive residency → proposed amendment to decision 13.** `docs/C2C-PROBE-20261106.md`.
- 2026-11-06 13:36 — T2+T3 loader worker dispatched (opus-5-5; first dispatch died on my `git commit` guard, re-dispatched). Handoff `docs/HANDOFF-T2T3-LOADER.md`.
