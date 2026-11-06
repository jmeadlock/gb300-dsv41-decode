"""Synthetic mini checkpoint with the 1017 checkpoint's tensor names, dtypes and packing (2 layers, dim 256,
8 routed experts, two Engram tables of 1,000 rows), written as 2 safetensors shards + index.json, plus the reference
config (`inference/config.json`, where the real checkpoint keeps it) and a hot map.

The tensor schema below is written from model.py by hand (cites per line), not derived from the model, so that
`test_meta_model_matches_schema` is a real check of it against the reference `Transformer`. Naming follows the real
checkpoint (already convert.py-spelled: `layers.0.ffn.experts.0.w1.scale`, docs/checkpoint-1017-inventory.json);
`hf_names=True` writes the pre-convert spelling (`model.` prefix, self_attn, mlp, weight_scale_inv,
e_score_correction_bias, plus an `mtp.0.embed.weight` that convert.py drops) for the rename round trip.
Values are random but valid: e4m3 from N(0, 0.5), E8M0 exponents 118..130, I8 over the full byte range
(every e2m1 nibble), bf16/f32 from N(0, 1).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file

MINI_CONFIG = {
    "vocab_size": 2048, "dim": 256, "moe_inter_dim": 256, "n_layers": 2, "n_mtp_layers": 1,
    "dspark_block_size": 5, "dspark_noise_token_id": 2000, "dspark_target_layer_ids": [1], "dspark_markov_rank": 64,
    "dspark_n_routed_experts": 4, "dspark_n_activated_experts": 2,
    "n_heads": 4, "n_routed_experts": 8, "n_shared_experts": 1, "n_activated_experts": 2,
    "score_func": "sqrtsoftplus", "route_scale": 1.5, "swiglu_limit": 10.0,
    "q_lora_rank": 128, "head_dim": 128, "rope_head_dim": 32, "norm_eps": 1e-20, "o_groups": 2, "o_lora_rank": 128,
    "window_size": 128, "kv_source_layers": [1], "index_source_layers": [1], "original_seq_len": 65536,
    "rope_theta": 10000, "rope_factor": 16, "beta_fast": 32, "beta_slow": 1,
    "index_n_heads": 4, "index_head_dim": 64, "index_topk": 16,
    "candidate_source_layer": -1, "candidate_topk_blocks": 0, "candidate_block_size": 0,
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
    "engram_layer_ids": [0, 1], "engram_vocab_size": 100, "engram_num_embeddings": [1000, 1000],
    "engram_max_ngram_size": 4, "engram_pad_id": 2, "engram_compressed_vocab_size": 99092,
    "engram_n_heads": 2, "engram_head_dim": 256,
    "dtype": "fp8", "expert_dtype": "fp4", "compress_rope_theta": 160000,
    "compress_ratios": [0, 2, 0],
    "vision_n_layers": 1, "vision_dim": 64, "vision_n_heads": 2, "vision_inter_dim": 128, "vision_patch_size": 14,
    "image_token_id": 2047,
}
# Hot ids per layer (the rest are cold): exercises both tiers and a non-prefix map.
MINI_HOT_MAP = {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6]}


def _fp8_scale_shape(out_f: int, in_f: int) -> Tuple[int, int]:
    return ((out_f + 31) // 32, (in_f + 31) // 32)  # model.py:228-232


def schema(cfg: dict) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """(name, safetensors dtype, shape) of every tensor the model loads, in the checkpoint's own dtypes."""
    D, V, I = cfg["dim"], cfg["vocab_size"], cfg["moe_inter_dim"]
    H, hd, ql = cfg["n_heads"], cfg["head_dim"], cfg["q_lora_rank"]
    G, orank, E = cfg["o_groups"], cfg["o_lora_rank"], cfg["n_routed_experts"]
    hc = cfg["hc_mult"]
    mix = (2 + hc) * hc
    out: List[Tuple[str, str, Tuple[int, ...]]] = []

    def fp8(name, out_f, in_f):  # Linear with fp8 weights (model.py:225-233)
        out.append((f"{name}.weight", "F8_E4M3", (out_f, in_f)))
        out.append((f"{name}.scale", "F8_E8M0", _fp8_scale_shape(out_f, in_f)))

    def fp4(name, out_f, in_f):  # Linear with fp4 weights: I8 [out, in/2] + E8M0 [out, in/32] (model.py:219-224)
        out.append((f"{name}.weight", "I8", (out_f, in_f // 2)))
        out.append((f"{name}.scale", "F8_E8M0", (out_f, in_f // 32)))

    out.append(("embed.weight", "BF16", (V, D)))                       # model.py:1204, 166
    for L in range(cfg["n_layers"]):
        p = f"layers.{L}"
        for k in ("attn", "ffn"):                                       # model.py:943-948
            out.append((f"{p}.hc_{k}_fn", "F32", (mix, hc * D)))
            out.append((f"{p}.hc_{k}_base", "F32", (mix,)))
            out.append((f"{p}.hc_{k}_scale", "F32", (3,)))
        out.append((f"{p}.attn_norm.weight", "BF16", (D,)))             # model.py:935-936
        out.append((f"{p}.ffn_norm.weight", "BF16", (D,)))
        a = f"{p}.attn"
        out.append((f"{a}.attn_sink", "F32", (H,)))                     # model.py:640
        fp8(f"{a}.wq_a", ql, D)                                         # model.py:641
        out.append((f"{a}.q_norm.weight", "BF16", (ql,)))               # model.py:642
        fp8(f"{a}.wq_b", H * hd, ql)                                    # model.py:643
        fp8(f"{a}.wkv", hd, D)                                          # model.py:644
        out.append((f"{a}.kv_norm.weight", "BF16", (hd,)))              # model.py:645
        fp8(f"{a}.wo_a", G * orank, H * hd // G)                        # stored fp8; model wants bf16 (646-650)
        fp8(f"{a}.wo_b", D, G * orank)                                  # model.py:651
        ratio = cfg["compress_ratios"][L]
        if L in cfg["kv_source_layers"]:                                # model.py:655-660, 437-456
            out.append((f"{a}.compressor.norm.weight", "BF16", (hd,)))
            out.append((f"{a}.compressor.wkv.weight", "BF16", (hd, D)))  # fp32 in the model at ratio > 1
            if ratio > 1:
                out.append((f"{a}.compressor.wgate.weight", "BF16", (hd, D)))
        if L in cfg["index_source_layers"]:                             # model.py:656-662, 496-525
            ih, ihd = cfg["index_n_heads"], cfg["index_head_dim"]
            fp8(f"{a}.indexer.wq_b", ih * ihd, ql)
            out.append((f"{a}.indexer.weights_proj.weight", "BF16", (ih, D)))
            if L in cfg["kv_source_layers"]:
                out.append((f"{a}.indexer.wk.weight", "BF16", (ihd, hd)))
                out.append((f"{a}.indexer.k_norm.weight", "BF16", (ihd,)))
        f = f"{p}.ffn"
        out.append((f"{f}.gate.weight", "BF16", (E, D)))                # model.py:807
        out.append((f"{f}.gate.bias", "F32", (E,)))                     # model.py:808
        out.append((f"{f}.gate.bias_vl", "F32", (E,)))                  # model.py:809 (vision only -> SKIP)
        for e in range(E):                                              # model.py:875-887, 838-840
            fp4(f"{f}.experts.{e}.w1", I, D)
            fp4(f"{f}.experts.{e}.w2", D, I)
            fp4(f"{f}.experts.{e}.w3", I, D)
        fp8(f"{f}.shared_experts.w1", I, D)                             # model.py:889
        fp8(f"{f}.shared_experts.w2", D, I)
        fp8(f"{f}.shared_experts.w3", I, D)
        if L in cfg["engram_layer_ids"]:                                # model.py:933-934, 343-348, 309-310
            rows = cfg["engram_num_embeddings"][cfg["engram_layer_ids"].index(L)]
            ehd = cfg["engram_head_dim"]
            ncols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
            out.append((f"{p}.engram.embed.weight", "F8_E4M3", (rows, ehd)))
            out.append((f"{p}.engram.embed.scale", "F8_E8M0", (rows, ehd // 32)))
            fp8(f"{p}.engram.wkv", D * (hc + 1), ncols * ehd)
            out.append((f"{p}.engram.q_weight", "BF16", (hc, D)))
            out.append((f"{p}.engram.k_weight", "BF16", (hc, D)))
    out.append(("norm.weight", "BF16", (D,)))                           # model.py:1208
    out.append(("head.weight", "BF16", (V, D)))                         # model.py:1209, bf16 on disk
    return out


def skip_extras(cfg: dict) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """Tensors the real checkpoint carries that M0 does not load (vision, aligner, image_*, mtp.*)."""
    D, vd = cfg["dim"], cfg["vision_dim"]
    return [
        ("vision.blocks.0.attn.wqkv.weight", "BF16", (3 * vd, vd)),
        ("vision.blocks.0.attn.wqkv.bias", "BF16", (3 * vd,)),
        ("vision.blocks.0.mlp.w1.weight", "BF16", (2 * cfg["vision_inter_dim"], vd)),
        ("vision.blocks.0.norm1.weight", "BF16", (vd,)),
        ("vision.patch_embed.proj.weight", "BF16", (vd, 588)),
        ("aligner.w1.weight", "BF16", (D, 9 * vd)),
        ("aligner.w1.bias", "BF16", (D,)),
        ("image_start", "BF16", (D,)), ("image_end", "BF16", (D,)), ("image_newline", "BF16", (D,)),
        ("mtp.0.hc_ffn_fn", "F32", ((2 + cfg["hc_mult"]) * cfg["hc_mult"], cfg["hc_mult"] * D)),
        ("mtp.0.main_proj.weight", "F8_E4M3", (D, D)),
        ("mtp.0.main_proj.scale", "F8_E8M0", _fp8_scale_shape(D, D)),
        ("mtp.0.attn.wo_a.weight", "F8_E4M3", (256, 256)),
        ("mtp.0.attn.wo_a.scale", "F8_E8M0", (8, 8)),
        ("mtp.0.ffn.experts.0.w1.weight", "I8", (cfg["moe_inter_dim"], D // 2)),
        ("mtp.0.ffn.experts.0.w1.scale", "F8_E8M0", (cfg["moe_inter_dim"], D // 32)),
        ("mtp.0.markov_head.embed.weight", "BF16", (cfg["vocab_size"], cfg["dspark_markov_rank"])),
    ]


def hf_name(name: str) -> str:
    """Pre-convert spelling of a runtime name (inverse of convert.py:104-115 for the names we generate)."""
    n = name
    if not n.startswith("vision."):
        n = n.replace(".ffn.", ".mlp.")
    n = n.replace(".attn.", ".self_attn.")
    if n.endswith(".scale"):
        n = n[: -len(".scale")] + ".weight_scale_inv"
    if n.endswith(".gate.bias"):
        n = n[: -len(".gate.bias")] + ".gate.e_score_correction_bias"
    return "model." + n


def _value(st: str, shape: Tuple[int, ...], g: torch.Generator) -> torch.Tensor:
    if st == "F8_E4M3":
        return (torch.randn(shape, generator=g, dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    if st == "F8_E8M0":
        return torch.randint(118, 131, shape, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    if st == "I8":
        return torch.randint(-128, 128, shape, generator=g, dtype=torch.int8)
    if st == "BF16":
        return torch.randn(shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    if st == "F32":
        return torch.randn(shape, generator=g, dtype=torch.float32)  # explicit: the loader sets default bf16
    raise ValueError(st)


def make_mini_checkpoint(out_dir: str, hf_names: bool = False, seed: int = 0, drop: Tuple[str, ...] = (),
                         extra: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (), cfg: dict = None) -> Dict[str, str]:
    """Write the mini checkpoint. Returns {"model_dir", "config", "map"}."""
    cfg = dict(MINI_CONFIG if cfg is None else cfg)
    os.makedirs(os.path.join(out_dir, "inference"), exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    items = [t for t in schema(cfg) + skip_extras(cfg) if t[0] not in drop] + list(extra)
    tensors = {}
    for name, st, shape in items:
        tensors[hf_name(name) if hf_names else name] = _value(st, shape, g).contiguous()
    if hf_names:
        tensors["model.mtp.0.embed.weight"] = _value("BF16", (cfg["vocab_size"], cfg["dim"]), g)  # convert drops it
    names = sorted(tensors)
    # two shards: layer 0 + top-level + skip extras in shard 1, the rest in shard 2
    shard_of = {n: (2 if ".layers.1." in "." + n.replace("model.", "", 1) else 1) for n in names}
    weight_map = {}
    for k in (1, 2):
        fn = f"model-0000{k}-of-00002.safetensors"
        part = {n: tensors[n] for n in names if shard_of[n] == k}
        save_file(part, os.path.join(out_dir, fn), metadata={"format": "pt"})
        weight_map.update({n: fn for n in part})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    cfg_path = os.path.join(out_dir, "inference", "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    map_path = os.path.join(out_dir, "hot_map.json")
    with open(map_path, "w") as f:
        json.dump(MINI_HOT_MAP, f)
    return {"model_dir": out_dir, "config": cfg_path, "map": map_path}
