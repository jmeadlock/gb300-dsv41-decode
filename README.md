# GB300 DSV4.1 Megakernel

Can one NVIDIA DGX Station GB300 run DeepSeek-V4.1-Flash's decode step in ≤ 5.0 ms (today: 7.36 ms on vLLM + a peer's recipe)?

A standalone decode runtime for DeepSeek-V4.1-Flash-1017 on a single GB300 (optionally + one RTX PRO 6000), built to find out. "Megakernel" is the banner; the project definition is *whatever makes the single-user decode step shortest*, under stock weights, all 384 experts, no KV quantization, bit-identical output by design.

- Charter, decisions, milestones, kill rule: [`PROJECT.md`](PROJECT.md)
- Intro post: https://al-engr.com/gb300-dsv41-megakernel.html
- First measurement window (W0) receipts and finding: [`w0/W0-RESULTS.md`](w0/W0-RESULTS.md)

Status (2026-11-06): chartered; production moved to the 4-Spark lane; Station is the dev box; no runtime code yet. Milestone 0 (re-hosted borrowed-kernel chain, greedy parity) is due 2026-12-18.

Layout (planned): `runtime/` Python host · `kernels/` CuTe DSL / Triton / CUDA C++ · `bench/` step profiler + fixtures · `recipes/{grace-only,sidecar}/` · `third_party/` vendored kernels with licenses · `docs/` ledgers and verdicts.

License: Apache-2.0. Authors: James Meadlock, with Milo (a Hermes agent; the model behind each commit is named in the commit or document byline).
