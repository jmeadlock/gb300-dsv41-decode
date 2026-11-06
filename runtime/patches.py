"""Monkeypatches on the unmodified reference model.py, each gated by an explicit flag. None active by default.

kv_numerics_pg61(enable):
    Replace the compressed-KV latent's FP4 QDQ (model.py:761, `fp4_act_quant(latent, 16, True, scale_dtype=e4m3)`)
    with the MXFP8 QDQ Pg61/vLLM `fp8_ds_mla` applies instead (D1 memo §Compressed KV: e4m3 values, block 32, UE8M0 pow2
    scale, no FP4 step). Window KV (model.py:708) already is FP8/32 UE8M0 in both; index K/Q (546/553) are left as the
    reference — at <= 310 tokens the indexer selects every position in both engines (D1 §2), so they are inert for the
    W0 fixture. Purpose: a discriminating experiment, not a product mode — if first-light drift vs Pg61 collapses under
    this flag, the gap is "model definition vs Pg61's KV format", not a runtime bug.
"""
import sys


def kv_numerics_pg61(enable: bool = True) -> None:
    ref = sys.modules.get("model") or __import__("model")
    kernel = sys.modules.get("kernel") or __import__("kernel")
    if not hasattr(ref, "_orig_fp4_act_quant"):
        ref._orig_fp4_act_quant = ref.fp4_act_quant

    def patched(x, block_size, inplace=False, scale_dtype=None):
        import torch
        if inplace and block_size == 16 and scale_dtype is torch.float8_e4m3fn:
            # the compressed-KV latent call site (model.py:761): MXFP8 in place instead of FP4
            kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True)
            return None
        return ref._orig_fp4_act_quant(x, block_size, inplace, scale_dtype) if scale_dtype is not None \
            else ref._orig_fp4_act_quant(x, block_size, inplace)

    ref.fp4_act_quant = patched if enable else ref._orig_fp4_act_quant
