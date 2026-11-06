# W0 — first measurement window (2026-11-06, 10:59–11:27 CST)

Receipts: `results/` (copied from the box `~/dsv41/w0/results-20261106-{1059,1107}/`).

## Phase 1 — Pg61 as-is (consumers already moved) — DONE
- Step cell (`w0_drive.py step`, 512-token greedy generation, thinking off, 3 reps; steps from `/metrics` spec counters, wall from SSE arrival times): **7.746 / 7.747 / 7.724 ms/step**, 2.75 / 2.63 / 2.93 tokens/step, 186 / 195 / 175 steps, TTFT 0.03 s. Different prompt and instrument from the 7.36 ms release number — not comparable as a regression.
- Parity reference (`parity-A.jsonl`, `parity-A2.jsonl`): 8 fixture prompts, greedy T=0, top-20 logprobs, 256 max tokens. A vs A2: **7/8 byte-identical; `code` diverged at token 121** (224 vs 256 tokens). This is Pg61's own run-to-run spread; M0's parity bar is graded against it, not against zero.

## Phase 2 — nsys-wrapped Pg61 boot — FAILED (finding)
- Clone: `launch-w0.sh` (= launch-pg61.sh + 2 policy lines), patch `06-nsys-wrap-serve.patch` (serve.sh: `-v nsight-systems:/nsys:ro`, `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node … vllm serve … --profiler-config.profiler=cuda`). Build 6 s, boot 339 s, bound, hash verify OK, args carried the wrap (`ds41-args.json`).
- Two warm requests answered (64 tokens). `POST /start_profile` → "Profiler started". The profiled 512-token request was accepted (200) and **never ran**: `num_requests_running 0`, `generation_tokens_total` frozen at 64, GPU util 0 %, `VLLM::EngineCore` at 163 % CPU for 10 min. `POST /stop_profile` logged "Stopping profiler..." and never completed. Stopped after 6 min; GPUs released cleanly (22 MiB / 120 MiB), not wedged.
- Reading: once CUPTI kernel tracing opened, the engine stopped making progress. Pg61's peer tier v3 (persistent cooperative kernel on the RTX + GB300 spin on host-mapped shm, `PEER3_SPINS=1e12`) and CUPTI tracing do not coexist. The v27-lane capture (Card I1b) that produced the 11.216 ms anatomy ran peer v2, not v3.
- First attempt (10:59) aborted on my own preflight grep (`nsys profile` vs JSON `"nsys","profile"`) — cost one 6-min boot; fixed.

## Phase 2b — second attempt, `--cuda-graph-trace=graph` (12:47–13:00 CST) — FAILED the same way, and the finding sharpened
- Same clone, only the trace mode changed; progress watchdog added (kill after 120 s without token progress). Boot 339 s, bound.
- Stalled on the **second warm request, before `/start_profile` was ever posted** (`generation_tokens_total` frozen at 32; first attempt froze at 64 after two warms + the profiled request). EngineCore at 271 % CPU, both GPUs 0 % util, sidecar log healthy (`serving`, 2,880 peer3 calls, 0 lost records), `/stop_profile` never completed. `deadlock-diag-W0b.txt`.
- So the stall is not the CUPTI capture range and not graph-node tracing: **an nsys-wrapped Pg61 engine stalls after 1–2 requests regardless of capture mode**, in a host spin consistent with the peer-tier-v3 wait loop (`PEER3_SPINS=1e12`) never seeing the sidecar's done flag. Unknown whether nsys's CUDA-API interception alters the host-mapped shm path or just the timing; the Card I1b capture that produced the v27 anatomy ran peer v2, not v3.
- **Decision (Milo, 13:00 CST): stop spending boots on a Pg61 per-role anatomy.** It is not on M0's critical path (M0 is graded on `/metrics` step time + parity), and the runtime will own its own sidecar protocol, so the next per-role trace is the runtime's. Reopen only with a Grace-only arm (no peer3) or a non-CUPTI instrument (`nsys --gpu-metrics-devices` sampling).

## Open
- Pg61 per-role anatomy still missing (parked, see 2b). Candidates if reopened, cheapest first: (a) `--cuda-graph-trace=graph` (no per-node CUPTI records; may not deadlock; gives graph-level timing only); (b) `nsys --gpu-metrics-devices` sampling (no CUPTI injection; counters only, `gpu_metrics_decode_windows.py`); (c) capture the Grace-only arm (peer3 off) — not Pg61, but the shape recipe A inherits; (d) profile with the sidecar in `coldskip`/self-answer mode. Needs a window; Station is dark by charter so cost is only box time.
- Per-boundary PDL gap histogram: same dependency.

## Box state at END
Lane DARK by charter. `ds41`/`ds41-sidecar` removed (`launch-w0.sh down`), peer shm absent, no watchdog. Pg61 restore = `launch-pg61.sh up` with `WORK=~/dsv41/garner/pg61-work MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e10… DS41_HOME=~/dsv41/garner/home` (~6 min), only on James's word. W0 clone tree at `~/dsv41/w0/work` (keep; reuse for the next capture attempt).
