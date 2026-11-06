"""Standalone DSV4.1 decode runtime (M0): streaming loader, tiered placement, host tier, expert pools.

Nothing in this package imports vllm (charter decision 3). The reference model (`reference/deepseek-inference/`)
is imported unchanged through `runtime.refmodel`; tilelang is only needed once a model forward runs.
"""
