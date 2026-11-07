"""M1-T11: mini checkpoints WITH the full `mtp.*` draft schema (tests/mini_ckpt.py writes only a few mtp tensors as
SKIP extras). The schema is written from model.py by hand (cites per line) so that runtime/mtp_loader.py's key check is
a real check of it against the reference DSparkBlock.

  ck = make_mini_checkpoint_mtp(dir, cfg=mtp_config_kind("mini3"))      # -> {"model_dir", "config", "map"}
"""
from __future__ import annotations

from typing import List, Tuple

from mini_ckpt import MINI_CONFIG, MINI_HOT_MAP, _fp8_scale_shape, make_mini_checkpoint, skip_extras


def mtp_schema(cfg: dict) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """(name, safetensors dtype, shape) of every mtp.* tensor DSparkBlock loads (+ one vision-only gate.bias_vl per
    stage, which the mtp loader skips)."""
    D, V, I = cfg["dim"], cfg["vocab_size"], cfg["moe_inter_dim"]
    H, hd, ql = cfg["n_heads"], cfg["head_dim"], cfg["q_lora_rank"]
    G, orank = cfg["o_groups"], cfg["o_lora_rank"]
    E = cfg.get("dspark_n_routed_experts") or cfg["n_routed_experts"]   # model.py:142-149
    hc = cfg["hc_mult"]
    mix = (2 + hc) * hc
    rank = cfg["dspark_markov_rank"]
    n_t = len(cfg["dspark_target_layer_ids"])
    out: List[Tuple[str, str, Tuple[int, ...]]] = []

    def fp8(name, out_f, in_f):
        out.append((f"{name}.weight", "F8_E4M3", (out_f, in_f)))
        out.append((f"{name}.scale", "F8_E8M0", _fp8_scale_shape(out_f, in_f)))

    def fp4(name, out_f, in_f):
        out.append((f"{name}.weight", "I8", (out_f, in_f // 2)))
        out.append((f"{name}.scale", "F8_E8M0", (out_f, in_f // 32)))

    n = cfg["n_mtp_layers"]
    for i in range(n):
        p = f"mtp.{i}"
        for k in ("attn", "ffn"):                                       # model.py:943-948
            out.append((f"{p}.hc_{k}_fn", "F32", (mix, hc * D)))
            out.append((f"{p}.hc_{k}_base", "F32", (mix,)))
            out.append((f"{p}.hc_{k}_scale", "F32", (3,)))
        out.append((f"{p}.attn_norm.weight", "BF16", (D,)))             # model.py:935-936
        out.append((f"{p}.ffn_norm.weight", "BF16", (D,)))
        a = f"{p}.attn"                                                 # model.py:640-651 (no compressor/indexer:
        out.append((f"{a}.attn_sink", "F32", (H,)))                     # not a backbone layer, 654-662)
        fp8(f"{a}.wq_a", ql, D)
        out.append((f"{a}.q_norm.weight", "BF16", (ql,)))
        fp8(f"{a}.wq_b", H * hd, ql)
        fp8(f"{a}.wkv", hd, D)
        out.append((f"{a}.kv_norm.weight", "BF16", (hd,)))
        fp8(f"{a}.wo_a", G * orank, H * hd // G)                        # fp8 on disk, bf16 in the model
        fp8(f"{a}.wo_b", D, G * orank)
        f = f"{p}.ffn"
        out.append((f"{f}.gate.weight", "BF16", (E, D)))                # model.py:807-809
        out.append((f"{f}.gate.bias", "F32", (E,)))
        out.append((f"{f}.gate.bias_vl", "F32", (E,)))                  # vision only -> skipped
        for e in range(E):
            fp4(f"{f}.experts.{e}.w1", I, D)
            fp4(f"{f}.experts.{e}.w2", D, I)
            fp4(f"{f}.experts.{e}.w3", I, D)
        fp8(f"{f}.shared_experts.w1", I, D)
        fp8(f"{f}.shared_experts.w2", D, I)
        fp8(f"{f}.shared_experts.w3", I, D)
        if i == 0:                                                      # model.py:1114-1117
            fp8(f"{p}.main_proj", D, D * n_t)
            out.append((f"{p}.main_norm.weight", "BF16", (D,)))
        if i == n - 1:                                                  # model.py:1118-1121, 1080-1100
            out.append((f"{p}.norm.weight", "BF16", (D,)))
            out.append((f"{p}.markov_head.embed.weight", "BF16", (V, rank)))
            out.append((f"{p}.markov_head.head.weight", "BF16", (V, rank)))       # fp32 in the model
            out.append((f"{p}.confidence_head.proj.weight", "BF16", (1, D + rank)))  # fp32 in the model
    return out


def mtp_config_kind(kind: str) -> dict:
    """mini (1 stage, 1 target layer), mini3 (3 stages, 2 target layers: the real model's shape of the problem),
    mini3w (mini3 with 4 experts top-3: duplicate experts across the 5 block rows are likely)."""
    cfg = dict(MINI_CONFIG)
    if kind == "mini":
        return cfg
    if kind in ("mini3", "mini3w"):
        cfg.update({"n_mtp_layers": 3, "compress_ratios": [0, 2, 0, 0, 0], "dspark_target_layer_ids": [0, 1]})
        if kind == "mini3w":
            cfg.update({"dspark_n_routed_experts": 4, "dspark_n_activated_experts": 3})
        return cfg
    raise ValueError(kind)


def make_mini_checkpoint_mtp(out_dir: str, cfg: dict = None, seed: int = 0, drop: Tuple[str, ...] = (),
                             extra=()) -> dict:
    cfg = dict(MINI_CONFIG if cfg is None else cfg)
    old = tuple(n for n, _, _ in skip_extras(cfg) if n.startswith("mtp."))
    mtp = tuple(t for t in mtp_schema(cfg) if t[0] not in drop)  # make_mini_checkpoint does not filter `extra`
    return make_mini_checkpoint(out_dir, seed=seed, drop=old + tuple(drop), extra=mtp + tuple(extra), cfg=cfg)


__all__ = ["mtp_schema", "mtp_config_kind", "make_mini_checkpoint_mtp", "MINI_HOT_MAP"]
