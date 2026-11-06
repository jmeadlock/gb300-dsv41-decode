"""CPU tests of the T2+T3 loader on a synthetic mini checkpoint (Mac: `.venv/bin/python -m pytest tests/`).

What runs here: rename map vs convert.py itself, placement report sums (mini + the real 1017 shapes from config +
inventory), read_tensor_into byte-exactness for every tensor, wo_a dequant vs a naive per-block computation and vs
convert.py's output, the fp4 zero-copy view, and a full `build_and_load(device="cpu")` of the *reference*
Transformer (model.py imported unchanged; tilelang stubbed because nothing here runs a kernel) whose every
parameter is compared byte-for-byte with convert.py's MP=1 output. GPU parts: tests/gpu/test_host_tier_gpu.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import redirect_stdout

import pytest
import torch
from safetensors import safe_open

from mini_ckpt import MINI_CONFIG, MINI_HOT_MAP, hf_name, make_mini_checkpoint, schema, skip_extras
from runtime import ckpt as ckpt_mod
from runtime.ckpt import ShardIndex, TensorEntry, convert_name, torch_dtype
from runtime.expert_pool import TIER_HBM, TIER_HOST, ExpertPool
from runtime.host_tier import GiB, HostBuffer, check_mem_available, mem_available_bytes
from runtime.loader import LoadRefused, build_and_load, build_meta_model, dequant_wo_a, load_reference_config
from runtime.placement import (DEFAULT_CONFIG, HotMap, PlacementConfig, Tier, format_report, place, placed_dtype,
                               report)
from runtime.placement import main as placement_main
from runtime.refmodel import import_convert, import_reference

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVENTORY = os.path.join(REPO, "docs", "checkpoint-1017-inventory.json")


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def mini(tmp_path_factory):
    return make_mini_checkpoint(str(tmp_path_factory.mktemp("mini")))


@pytest.fixture(scope="session")
def mini_hf(tmp_path_factory):
    return make_mini_checkpoint(str(tmp_path_factory.mktemp("mini_hf")), hf_names=True)


@pytest.fixture(scope="session")
def convert_out(mini):
    """convert.py's MP=1 state dict for the mini checkpoint (its save_file captured instead of writing)."""
    return _run_convert(mini["model_dir"])


def _run_convert(model_dir):
    conv = import_convert()
    captured = {}
    orig = conv.save_file
    conv.save_file = lambda sd, path: captured.update(sd)
    try:
        out = os.path.join(model_dir, "_convert_out")
        with redirect_stdout(io.StringIO()):
            conv.main(model_dir, out, 1, None)
    finally:
        conv.save_file = orig
    return captured


@pytest.fixture
def ref_cpu(monkeypatch):
    """Reference model module with the CPU kernel stub; the Engram tokenizer map stubbed (the hash is not
    exercised by loading; on the box the real tokenizer is used)."""
    ref = import_reference(cpu_stubs=True)
    import engram

    monkeypatch.setattr(engram, "build_compressed_token_map",
                        lambda tok: (list(range(MINI_CONFIG["vocab_size"])), MINI_CONFIG["engram_compressed_vocab_size"]))
    prev = torch.get_default_dtype()
    yield ref
    torch.set_default_dtype(prev)  # build_and_load leaves bf16 set, as generate.py:118 does


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _shard_tensors(model_dir):
    out = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, f), "pt", device="cpu") as fh:
                for k in fh.keys():
                    out[k] = fh.get_tensor(k)
    return out


# ----------------------------------------------------------------------------------------------- rename map
def test_rename_identity_on_converted_names():
    for name, _, _ in schema(MINI_CONFIG) + skip_extras(MINI_CONFIG):
        assert convert_name(name) == name


def test_rename_round_trip_hf_spelling():
    for name, _, _ in schema(MINI_CONFIG) + skip_extras(MINI_CONFIG):
        assert convert_name(hf_name(name)) == name, (hf_name(name), convert_name(hf_name(name)))
    assert convert_name("model.mtp.0.embed.weight") is None
    assert convert_name("mtp.2.head.weight") is None
    assert convert_name("mtp.2.markov_head.embed.weight") == "mtp.2.markov_head.embed.weight"
    assert convert_name("vision.blocks.3.mlp.w1.weight") == "vision.blocks.3.mlp.w1.weight"  # convert.py:112
    assert convert_name("model.layers.7.mlp.gate.e_score_correction_bias") == "layers.7.ffn.gate.bias"


def test_rename_matches_convert_py(mini, mini_hf, convert_out):
    """Our index names == the keys convert.py itself writes (wo_a.scale popped by its dequant; mtp embed dropped)."""
    for ck, conv in ((mini, convert_out), (mini_hf, _run_convert(mini_hf["model_dir"]))):
        idx = ShardIndex.scan(ck["model_dir"])
        ours = {n for n in idx.names() if not n.endswith("wo_a.scale")}
        assert ours == set(conv), (sorted(ours - set(conv))[:5], sorted(set(conv) - ours)[:5])
    assert ShardIndex.scan(mini_hf["model_dir"]).dropped == ["model.mtp.0.embed.weight"]


def test_index_json_mismatch_refused(tmp_path):
    ck = make_mini_checkpoint(str(tmp_path))
    p = os.path.join(ck["model_dir"], "model.safetensors.index.json")
    j = json.load(open(p))
    j["weight_map"]["layers.0.ghost.weight"] = "model-00001-of-00002.safetensors"
    json.dump(j, open(p, "w"))
    with pytest.raises(ValueError, match="incomplete"):
        ShardIndex.scan(ck["model_dir"])


# ----------------------------------------------------------------------------------------------- reads
def test_read_tensor_into_byte_exact_every_tensor(mini, monkeypatch):
    """sha256(placed bytes) == sha256(shard bytes) for every tensor, three destination kinds, chunked reads."""
    monkeypatch.setattr(ckpt_mod, "READ_CHUNK", 4096)  # force the multi-chunk path on small tensors
    idx = ShardIndex.scan(mini["model_dir"])
    shard = _shard_tensors(mini["model_dir"])
    hb = HostBuffer(max(e.nbytes for e in idx.entries.values()), "test")
    for name, e in idx.entries.items():
        want = _sha(shard[e.source_name])
        t = torch.empty(e.shape, dtype=e.dtype)
        idx.read_tensor_into(name, t)
        assert _sha(t) == want, name
        ba = bytearray(e.nbytes)
        idx.read_tensor_into(name, ba)
        assert hashlib.sha256(ba).hexdigest() == want, name
        idx.read_tensor_into(name, hb.memoryview(0, e.nbytes))
        assert hashlib.sha256(hb.memoryview(0, e.nbytes)).hexdigest() == want, name
        assert idx.file_sha256(name) == want, name
    # byte ranges compose
    e = idx["layers.0.engram.embed.weight"]
    parts = bytearray()
    for start in range(0, e.nbytes, 70001):
        n = min(70001, e.nbytes - start)
        b = bytearray(n)
        idx.read_into(e.name, b, start, n)
        parts += b
    assert hashlib.sha256(parts).hexdigest() == _sha(shard[e.source_name])
    with pytest.raises(ValueError):
        idx.read_tensor_into(e.name, bytearray(e.nbytes - 1))
    idx.close()


# ----------------------------------------------------------------------------------------------- placement
def test_placement_tiers_mini(mini):
    cfg = PlacementConfig.from_file(mini["config"])
    hot = HotMap.load(mini["map"], cfg)
    idx = ShardIndex.scan(mini["model_dir"])
    for name, e in idx.entries.items():
        t = place(name, e.shape, e.dtype, cfg, hot)
        if name.startswith(("vision.", "aligner.", "image_", "mtp.")) or name.endswith("bias_vl"):
            assert t is Tier.SKIP, name
        elif ".engram.embed." in name:
            assert t is Tier.HOST, name
        elif ".ffn.experts." in name:
            layer, ex = int(name.split(".")[1]), int(name.split(".")[4])
            assert t is (Tier.HBM if ex in MINI_HOT_MAP[str(layer)] else Tier.HOST), name
        else:
            assert t is Tier.HBM, name


def test_placement_report_sums_mini(mini):
    cfg = PlacementConfig.from_file(mini["config"])
    hot = HotMap.load(mini["map"], cfg)
    idx = ShardIndex.scan(mini["model_dir"])
    rep = report(idx, cfg, hot)
    assert sum(r["n"] for r in rep["rows"]) == len(idx.entries)
    assert sum(r["disk"] for r in rep["rows"]) == idx.total_bytes()
    for t, a in rep["tiers"].items():
        assert a["disk"] == sum(r["disk"] for r in rep["rows"] if r["tier"] == t)
        assert a["placed"] == sum(r["placed"] for r in rep["rows"] if r["tier"] == t)
    eb = cfg.expert_bytes
    assert eb == 3 * (256 * 128 + 256 * 8)
    c = rep["check"]
    assert round(c["hbm_routed_experts_GiB"] * GiB) == 2 * 4 * eb
    assert round(c["host_routed_experts_GiB"] * GiB) == 2 * 4 * eb
    assert round(c["host_engram_GiB"] * GiB) == 2 * 1000 * (256 + 8)
    rows = {(r["tier"], r["role"]): r for r in rep["rows"]}
    # placed vs disk: wo_a fp8+scale -> bf16, head bf16 -> fp32, ratio-2 compressor bf16 -> fp32
    att = rows[("HBM", "attention")]
    wo_a_disk = 2 * (256 * 256 + 8 * 8)
    comp = 2 * 128 * 256 * 2
    assert att["placed"] - att["disk"] == 2 * 256 * 256 * 2 - wo_a_disk + comp
    assert rows[("HBM", "head")]["placed"] == 2 * rows[("HBM", "head")]["disk"]
    assert rep["tiers"]["SKIP"]["placed"] == 0


def _virtual_index(cfg_path):
    """A ShardIndex of the real 1017 routed experts + Engram tables, shapes from config.json (cross-checked
    against the inventory below). No shards needed: placement only reads headers."""
    cfg = json.load(open(cfg_path))
    D, I, E, L = cfg["dim"], cfg["moe_inter_dim"], cfg["n_routed_experts"], cfg["n_layers"]
    idx = ShardIndex(model_dir="<virtual 1017>", shards=["<virtual>"])
    off = 0

    def add(name, st, shape):
        nonlocal off
        n = 1
        for s in shape:
            n *= s
        nb = n * ckpt_mod.ST_DTYPES[st][1]
        idx.entries[name] = TensorEntry(name, name, "<virtual>", off, nb, st, tuple(shape))
        off += nb

    for l in range(L):
        for e in range(E):
            for p, (n, k) in (("w1", (I, D)), ("w2", (D, I)), ("w3", (I, D))):
                add(f"layers.{l}.ffn.experts.{e}.{p}.weight", "I8", (n, k // 2))
                add(f"layers.{l}.ffn.experts.{e}.{p}.scale", "F8_E8M0", (n, k // 32))
    for l, rows in zip(cfg["engram_layer_ids"], cfg["engram_num_embeddings"]):
        add(f"layers.{l}.engram.embed.weight", "F8_E4M3", (rows, cfg["engram_head_dim"]))
        add(f"layers.{l}.engram.embed.scale", "F8_E8M0", (rows, cfg["engram_head_dim"] // 32))
    return idx


def test_placement_report_reproduces_plan_section4():
    """M0-PORT-PLAN.md section 4: HBM experts 206.61 GiB at 295/layer, host experts 62.33, Engram 188.83
    (and the 325/layer fit max: 227.62 / 41.32)."""
    inv = json.load(open(INVENTORY))
    cfg = PlacementConfig.from_file(DEFAULT_CONFIG)
    assert cfg.expert_bytes == 18_800_640
    # the expert size and Engram shapes agree with the checkpoint inventory
    assert round((40 * 384 + 3 * 128) * cfg.expert_bytes / GiB, 3) == inv["roles"]["routed_experts"]["GiB"]
    ex = {e[0]: e for e in inv["roles"]["routed_experts"]["examples"] + inv["roles"]["engram"]["examples"]}
    assert ex["layers.0.ffn.experts.0.w1.scale"][2] == [2304, 160]
    assert ex["layers.0.ffn.experts.0.w2.scale"][2] == [5120, 72]
    assert ex["layers.1.engram.embed.weight"][3] == 384006168 * 256
    assert ex["layers.1.engram.embed.scale"][3] == 384006168 * 8
    idx = _virtual_index(DEFAULT_CONFIG)
    rep = report(idx, cfg, HotMap.first_n(295, cfg))
    c = rep["check"]
    assert round(c["hbm_routed_experts_GiB"], 2) == 206.61
    assert round(c["host_routed_experts_GiB"], 2) == 62.33
    assert round(c["host_engram_GiB"], 2) == 188.83
    assert round(c["host_pinned_GiB"], 2) == 251.17  # plan section 4: "Grace pins 251.17 GiB at 295"
    rep = report(idx, cfg, HotMap.first_n(325, cfg))
    assert round(rep["check"]["hbm_routed_experts_GiB"], 2) == 227.62
    assert round(rep["check"]["host_routed_experts_GiB"], 2) == 41.32
    assert round(rep["check"]["host_pinned_GiB"], 2) == 230.16


def test_placement_cli_mini(mini, capsys):
    assert placement_main(["--model-dir", mini["model_dir"], "--map", mini["map"]]) == 0
    out = capsys.readouterr().out
    assert "plan section 4 check" in out and "HOST  TOTAL" in out


def test_hot_map_validation():
    cfg = PlacementConfig.from_dict(MINI_CONFIG)
    with pytest.raises(ValueError, match="layer 1 missing"):
        HotMap({0: [1]}, cfg.n_layers, cfg.n_routed_experts, "t")
    with pytest.raises(ValueError, match="duplicate"):
        HotMap({0: [1, 1], 1: []}, cfg.n_layers, cfg.n_routed_experts, "t")
    with pytest.raises(ValueError, match="out of range"):
        HotMap({0: [8], 1: []}, cfg.n_layers, cfg.n_routed_experts, "t")
    h = HotMap.first_n(3, cfg)
    assert h.hot_ids(1) == (0, 1, 2) and h.cold_ids(1) == (3, 4, 5, 6, 7)
    assert HotMap({0: [7, 1], 1: [2]}, 2, 8, "t").hot_ids(0) == (1, 7)  # slot order independent of JSON order


# ----------------------------------------------------------------------------------------------- value paths
def test_wo_a_dequant_vs_naive_and_convert(mini, convert_out):
    idx = ShardIndex.scan(mini["model_dir"])
    for name in [n for n in idx.names() if n.endswith(".attn.wo_a.weight") and not n.startswith("mtp.")]:
        w = idx.read_tensor(name)
        s = idx.read_tensor(name[:-6] + "scale")
        got = dequant_wo_a(w, s)
        # naive: every 32x32 block times 2^(e - 127), then bf16
        naive = torch.empty(w.shape, dtype=torch.float32)
        e8 = s.view(torch.uint8).to(torch.int32)
        for bi in range(s.shape[0]):
            for bj in range(s.shape[1]):
                blk = w[bi * 32:(bi + 1) * 32, bj * 32:(bj + 1) * 32].float()
                naive[bi * 32:(bi + 1) * 32, bj * 32:(bj + 1) * 32] = blk * (2.0 ** (int(e8[bi, bj]) - 127))
        assert torch.equal(got, naive.to(torch.bfloat16)), name
        assert torch.equal(got, convert_out[name]), name
        # exact: e4m3 x 2^k fits bf16, so widening back loses nothing
        assert torch.equal(got.float(), naive), name


def test_fp4_view_shape_and_zero_copy(mini):
    idx = ShardIndex.scan(mini["model_dir"])
    e = idx["layers.0.ffn.experts.3.w2.weight"]
    assert e.st_dtype == "I8" and e.shape == (256, 128)
    t = idx.read_tensor(e.name)
    v = t.view(torch.float4_e2m1fn_x2)
    assert v.dtype == torch.float4_e2m1fn_x2 and tuple(v.shape) == (256, 128)  # [N, K/2]: two nibbles per byte
    assert v.data_ptr() == t.data_ptr()
    assert placed_dtype(e.name, e.dtype, PlacementConfig.from_dict(MINI_CONFIG)) == torch.float4_e2m1fn_x2


# ----------------------------------------------------------------------------------------------- host tier (pure)
def test_host_buffer_exact_size_and_views():
    n = 3 * 4096 + 17
    hb = HostBuffer(n, "t")
    assert hb.nbytes == n and hb.cpu.numel() == n and hb.cpu.data_ptr() == hb.ptr
    hb.cpu[:] = torch.arange(n, dtype=torch.int64).to(torch.uint8)
    v = hb.host_view(torch.float8_e4m3fn, (4, 8), offset=16)
    assert v.data_ptr() == hb.ptr + 16 and tuple(v.shape) == (4, 8)
    assert bytes(hb.memoryview(16, 4)) == bytes([16, 17, 18, 19])
    with pytest.raises(ValueError):
        hb.host_view(torch.uint8, (n + 1,))
    assert not hb.is_registered()
    with pytest.raises(RuntimeError, match="not registered"):
        hb.cuda_view(torch.uint8, (4,))
    del v
    hb.close()


def test_mem_available_check():
    avail = mem_available_bytes()
    if avail is None:  # the Mac: no /proc/meminfo, check is a logged no-op
        assert check_mem_available(1 << 60, 0, "t")["mem_available_GiB"] is None
    else:
        with pytest.raises(MemoryError):
            check_mem_available(avail + GiB, 0, "t")


# ----------------------------------------------------------------------------------------------- full load (CPU)
def test_meta_model_matches_schema(ref_cpu, mini):
    """The hand-written mini schema is exactly what the reference Transformer declares (shapes; dtypes after the
    documented transforms), so the mini checkpoint really has the real naming/packing."""
    cfg = load_reference_config(mini["config"])
    model = build_meta_model(ref_cpu, ref_cpu.ModelArgs(**cfg), object(), torch.device("cpu"))
    params = {n: p for n, p in model.named_parameters()}
    assert all(p.is_meta for p in params.values())
    assert all(not b.is_meta for b in model.buffers())
    pcfg = PlacementConfig.from_dict(cfg)
    sch = {n: (st, s) for n, st, s in schema(MINI_CONFIG)}
    loaded = {n for n in sch if not n.endswith("bias_vl") and not n.endswith("wo_a.scale")}
    assert loaded == set(params), (sorted(loaded - set(params)), sorted(set(params) - loaded))
    for n in loaded:
        st, shape = sch[n]
        assert tuple(params[n].shape) == shape, n
        assert placed_dtype(n, torch_dtype(st), pcfg) == params[n].dtype, n


def test_build_and_load_cpu_matches_convert(ref_cpu, mini, convert_out):
    model = build_and_load(mini["model_dir"], mini["map"], "cpu", tokenizer=object(), cpu_stubs=True, verify=True,
                           io_threads=4, stage_bytes=1 << 16, n_stage=3, host_chunk_bytes=50_000, log=lambda *a: None)
    res = model.runtime_load
    assert res.verify["mismatch"] == [] and res.verify["ok"] == res.verify["n"] > 0
    params = dict(model.named_parameters())
    assert set(params) == set(convert_out) - {n for n in convert_out
                                              if n.startswith(("vision.", "aligner.", "image_", "mtp."))
                                              or n.endswith("bias_vl")}
    for n, p in params.items():
        c = convert_out[n]
        if p.dtype == torch.float32 and c.dtype == torch.bfloat16:  # head, ratio-2 compressor: load_model's copy_
            assert torch.equal(p, c.float()), n
        else:
            assert p.dtype == c.dtype and p.shape == c.shape, (n, p.dtype, c.dtype)
            assert torch.equal(p.reshape(-1).view(torch.uint8), c.reshape(-1).view(torch.uint8)), n
    assert sorted(res.widened) == ["head.weight", "layers.1.attn.compressor.wgate.weight",
                                   "layers.1.attn.compressor.wkv.weight"]
    assert len(res.dequantized) == 2
    # every quantized Linear exposes weight.scale as the reference's linear() expects (model.py:192, 202, 222)
    for mod in model.modules():
        if isinstance(mod, ref_cpu.Linear) and mod.scale is not None:
            assert mod.weight.scale is mod.scale
    # tiers: Engram params alias their host buffers; experts are pool views
    for n in ("layers.0.engram.embed.weight", "layers.1.engram.embed.scale"):
        assert params[n].data_ptr() == res.host_buffers[n].ptr
    pool = res.pool
    for l in range(2):
        for e in range(8):
            w = model.layers[l].ffn.experts[e].w1.weight
            assert w.dtype == torch.float4_e2m1fn_x2 and tuple(w.shape) == (256, 128) and w.is_contiguous()
            tier, slot = int(pool.loc[l, e, 1]), int(pool.loc[l, e, 0])
            if e in MINI_HOT_MAP[str(l)]:
                assert tier == TIER_HBM and w.data_ptr() == pool.hbm[l]["w1"][0][slot].data_ptr()
            else:
                assert tier == TIER_HOST
                assert pool.host.ptr <= w.data_ptr() < pool.host.ptr + pool.host.nbytes
    # cold-hit counter fires (host-side) before the expert's first kernel call (the CPU stub raises there)
    cold_e = [e for e in range(8) if e not in MINI_HOT_MAP["0"]][0]
    with pytest.raises(RuntimeError, match="tilelang stub"):
        model.layers[0].ffn.experts[cold_e](torch.zeros(3, 256, dtype=torch.bfloat16))
    assert pool.counters["cold_calls"][0] == 1 and pool.counters["cold_rows"][0] == 3
    assert pool.counter_summary()["cold_share"] == 1.0
    # buffers are real (KV caches, RoPE tables, Engram hash state)
    assert all(not b.is_meta for b in model.buffers())


def test_refuses_missing_and_unexpected(ref_cpu, tmp_path):
    ck = make_mini_checkpoint(str(tmp_path), drop=("layers.1.attn.kv_norm.weight",),
                              extra=(("layers.0.attn.mystery.weight", "BF16", (4, 4)),))
    with pytest.raises(LoadRefused) as ei:
        build_and_load(ck["model_dir"], ck["map"], "cpu", tokenizer=object(), cpu_stubs=True, log=lambda *a: None)
    msg = str(ei.value)
    assert "MISSING" in msg and "layers.1.attn.kv_norm.weight" in msg
    assert "UNEXPECTED" in msg and "layers.0.attn.mystery.weight" in msg


def test_refuses_shape_mismatch(ref_cpu, tmp_path):
    bad = [("layers.0.attn.q_norm.weight", "BF16", (64,))]
    ck = make_mini_checkpoint(str(tmp_path), drop=("layers.0.attn.q_norm.weight",), extra=tuple(bad))
    with pytest.raises(LoadRefused, match="shape"):
        build_and_load(ck["model_dir"], ck["map"], "cpu", tokenizer=object(), cpu_stubs=True, log=lambda *a: None)
