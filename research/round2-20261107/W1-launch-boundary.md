# W1 — The launch boundary itself (round 2, 2026-11-07)

Status: IN PROGRESS. Sections are appended as they finish.

Scope: making a kernel boundary inside a captured CUDA graph cheaper without fusing (PDL, graph-level knobs, node count vs replay, launch floors on sm_100/103, cuBLAS/nvjet small-M). Fusion ladder is the M2 memo's job, not covered here.
