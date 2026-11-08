"""GPU tests of M4-P2 (runtime/prefill_fast.py: the grouped, bit-identical prefill). Skipped without CUDA/tilelang (Mac);
tests/test_prefill_fast_cpu.py dry-runs this file's gate functions on the CPU kernel emulations (DEV "cpu").

Gates (docs/HANDOFF-M4-P2-FAST-PREFILL.md "Gate on the mini then the box"), mini checkpoints (tests/mini_mtp.py "mini3":
3 draft stages, 2 target layers, a ratio-2 compressor layer, top-2 of 8; tests/test_static_emulated_cpu.py "mini4":
4 layers, ratios 0/2/1/1, top-4 of 8 -- the fp32 combine order is observable at k >= 3), real kernels, hot + cold
tiers (cold = UVA views of pinned host memory):
  * MoE per layer: the MoE input of every layer captured on a reference prefill (bench/prefill_anatomy.py
    PrefillProbe), then the reference MoE.forward vs `grouped_moe_forward` at streams 1 and 4: torch.equal.
  * prefill: prefill_hidden mode "reference" with fills ON (today's headline state) vs mode "grouped" with fills OFF
    (the new default): ids, fp32 logits, main_hidden, and EVERY buffer of every layer after the prefill (window KV,
    compressed KV, indexer K cache, ratio-2 compressor kv_state / score_state) torch.equal, tensor by tensor; the
    grouped prefill twice (self-replay); the counters (expert calls = distinct routed experts per layer, GEMM
    launches = 3 x calls, cold calls = the pool's cold experts among them).
  * stream: the DSpark loop (static eager; on CUDA also graph mode, 2 graphs) after a reference prefill vs after a
    grouped prefill: emitted token ids, accept-length sequence and every emitted position's fp32-logits sha256 equal.
Real checkpoint: bench/prefill_bench.py `run` (the 8 fixture prompts, logits / main_hidden / caches, timing, launches)
and bench/headline.py with DSV41_PREFILL=grouped --residency-ref <today's receipt> (the 256-token streams); the box
lines are `python3 bench/prefill_bench.py box`.

Run on the box (orchestrator), repo at $W:
  docker run --rm --network none --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host \\
    --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v "$W":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv --entrypoint python3 vllm/vllm-openai:v0.31.0 \\
    tests/gpu/test_prefill_fast_gpu.py
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime.refmodel import tilelang_available  # noqa: E402

DEV = "cuda"       # tests/test_prefill_fast_cpu.py dry-runs the gates on the CPU emulations (DEV "cpu")
STEP_KW = {}       # ... with {"_allow_cpu": True}
LOOP_KW = {}       # ... with {"_allow_cpu": True}
MAX_POS = 511
COUNTS = {"gates": 0, "layers_equal": 0, "buffers_equal": 0, "tokens": 0, "logit_rows": 0}
REPORT = []


def _need_tilelang():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def _mini4_cfg():
    from mini_ckpt import MINI_CONFIG

    cfg = dict(MINI_CONFIG)
    cfg.update({"n_layers": 4, "compress_ratios": [0, 2, 1, 1, 0], "kv_source_layers": [1, 2],
                "index_source_layers": [1, 2, 3], "candidate_source_layer": 2, "candidate_topk_blocks": 64,
                "candidate_block_size": 8, "index_topk": 64, "dspark_target_layer_ids": [3],
                "n_activated_experts": 4})
    return cfg, {"0": [0, 2, 5, 7], "1": [1, 3, 4, 6], "2": [0, 1, 2, 3], "3": [4, 5, 6, 7]}


@contextlib.contextmanager
def _model(kind: str = "mini3"):
    """Build + load a mini checkpoint (mtp schema, load_mtp=True) on cuda -> (model, pool)."""
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    import engram
    from mini_ckpt import MINI_CONFIG
    from mini_mtp import make_mini_checkpoint_mtp, mtp_config_kind

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    tmp = tempfile.TemporaryDirectory()
    if kind == "mini4":                       # no mtp stages (the stream gate uses mini3)
        from mini_ckpt import make_mini_checkpoint

        cfg, hot = _mini4_cfg()
        ck = make_mini_checkpoint(os.path.join(tmp.name, "ckpt"), cfg=cfg)
        mtp = False
    else:
        ck = make_mini_checkpoint_mtp(os.path.join(tmp.name, "ckpt"), cfg=mtp_config_kind(kind))
        hot, mtp = ck["map"], True
    model = build_and_load(ck["model_dir"], hot, "cuda", tokenizer=object(), io_threads=4, stage_bytes=1 << 20,
                           load_mtp=mtp, log=lambda *a: None)
    model.temperature = 0.0
    res = model.runtime_load
    try:
        yield model, res.pool
    finally:
        torch.set_default_device("cpu")
        del model
        torch.cuda.synchronize()
        res.release()
        tmp.cleanup()


def _toks(n, seed, hi=2000):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, hi, (n,), generator=g, device="cpu").tolist()


@contextlib.contextmanager
def _env(**kv):
    prev = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _determinism():
    """The headline's state on CUDA (set_determinism: deterministic algorithms, warn_only); restored after."""
    if DEV != "cuda":
        yield
        return
    from runtime.generate import set_determinism

    was = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    det = getattr(torch.utils, "deterministic", None)
    fill = getattr(det, "fill_uninitialized_memory", None)
    set_determinism(0)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
        if det is not None and fill is not None:
            det.fill_uninitialized_memory = fill


def layer_buffers(model):
    """Every buffer of every main layer (KV caches, compressor states, indexer caches, freqs), cloned."""
    return {n: b.detach().clone() for n, b in model.layers.named_buffers()}


def buffers_diff(a, b):
    return [n for n in a if not torch.equal(a[n], b[n])]


def _sha(t) -> str:
    return hashlib.sha256(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ gates
def _gate_moe_layers(kind="mini4", L=29, streams=(1, 4)):
    """Per layer: reference MoE.forward == grouped_moe_forward on the captured prefill input."""
    import prefill_anatomy as PA
    from runtime.prefill_fast import grouped_moe_forward

    with _model(kind) as (model, pool), _determinism():
        ids = _toks(L, 40 + L)
        probe = PA.PrefillProbe(model, pool, capture_moe=range(len(model.layers)))
        with probe, _env(DSV41_PREFILL="reference"):
            PA.prefill_once(model, ids, (), seed=False, device=DEV)
        n = 0
        with torch.inference_mode():
            for l in range(len(model.layers)):
                moe, x = model.layers[l].ffn, probe.moe_inputs[l]
                ref = moe(x)
                for s in streams:
                    st: dict = {}
                    got = grouped_moe_forward(moe, x, streams=s, stats=st, pool=pool)
                    _sync()
                    assert torch.equal(ref, got), f"{kind} L={L} layer {l} streams {s}: grouped MoE != reference"
                    assert st["gemm_launches"] == 3 * st["expert_calls"], st
                    n += 1
        COUNTS["gates"] += 1
        COUNTS["layers_equal"] += n
        print(f"  {kind} L={L}: {len(model.layers)} layers x streams {list(streams)}: grouped MoE == reference "
              f"(torch.equal, {n} comparisons)")
        return n


def _gate_prefill(kind="mini4", L=37, streams=4):
    """prefill_hidden reference (fills on) vs grouped (fills off): outputs + every layer buffer; replay; counters."""
    import prefill_anatomy as PA
    from runtime.draft_step import prefill_hidden
    from runtime.prefill_fast import last_prefill

    with _model(kind) as (model, pool), _determinism():
        ids = _toks(L, 70 + L)
        tok = torch.tensor([ids], dtype=torch.long, device=DEV)
        hidden = tuple(int(i) for i in getattr(getattr(model, "runtime_mtp", None), "target_layer_ids", ()) or ())
        runs = []
        for mode, fills in (("reference", "on"), ("grouped", "off"), ("grouped", "off")):
            with _env(DSV41_PREFILL_FILLS=fills, DSV41_PREFILL_STREAMS=streams):
                out, logits, mh = prefill_hidden(model, tok, hidden, mode=mode)
            _sync()
            runs.append({"mode": mode, "out": out.clone(), "logits": logits.clone(),
                         "mh": None if mh is None else mh.clone(), "bufs": layer_buffers(model),
                         "stats": last_prefill()})
        a, b, c = runs
        for r in (b, c):
            assert torch.equal(a["out"], r["out"]), f"{kind} L={L}: first token differs"
            assert torch.equal(a["logits"], r["logits"]), f"{kind} L={L}: prefill logits differ"
            assert (a["mh"] is None) == (r["mh"] is None) and (a["mh"] is None or torch.equal(a["mh"], r["mh"])), \
                f"{kind} L={L}: main_hidden differs"
            d = buffers_diff(a["bufs"], r["bufs"])
            assert not d, f"{kind} L={L}: caches differ after the grouped prefill: {d}"
        st = b["stats"]
        assert st["mode"] == "grouped" and st["fills"] == "off" and st.get("layers") == len(model.layers), st
        # the census: calls per layer = distinct routed experts (the reference's Expert calls), cold by pool tier
        probe = PA.PrefillProbe(model, pool)
        with probe, _env(DSV41_PREFILL="reference"):
            PA.prefill_once(model, ids, (), seed=False, device=DEV)
        cen = probe.census(L, int(model.layers[0].ffn.n_activated_experts))
        assert st["calls_per_layer"] == cen["calls_per_layer"], (st["calls_per_layer"], cen["calls_per_layer"])
        assert st["expert_calls"] == cen["expert_calls"] and st["cold_calls"] == cen["cold_calls"], (st, cen)
        assert st["gemm_launches"] == 3 * cen["expert_calls"] and st["pairs"] == cen["rows"], (st, cen)
        assert st["padded_rows"] % 32 == 0 and st["padded_rows"] >= st["pairs"]
        COUNTS["gates"] += 1
        COUNTS["buffers_equal"] += len(a["bufs"])
        print(f"  {kind} L={L}: prefill grouped (fills off, streams {streams}) == reference (fills on): first, logits "
              f"{_sha(a['logits'].reshape(-1, a['logits'].size(-1))[-1])[:12]}, main_hidden, {len(a['bufs'])} layer "
              f"buffers; replay equal; {st['expert_calls']} expert calls ({st['cold_calls']} cold) -> "
              f"{st['gemm_launches']} GEMM launches (reference: {cen['expert_calls']} Expert calls)")
        return st


def _gate_stream(kind="mini3", L=17, max_new=48, mode="static", n_graphs=None, streams=4):
    """The DSpark loop after a reference prefill vs after a grouped prefill: tokens, accepts, per-position logits."""
    from runtime.dspark_loop import DSparkLoop
    from runtime.generate import Generator
    from tiny_tok import make_wordlevel_tokenizer

    with _model(kind) as (model, pool), _determinism():
        V = model.head.weight.size(0)
        gen = Generator(model, tokenizer=make_wordlevel_tokenizer(V), eos_ids=[1], device=DEV, mode=mode,
                        max_pos=MAX_POS)
        gen.static_kw = dict(STEP_KW)
        ids = _toks(L, 500 + L, min(2000, V))
        loop = DSparkLoop(gen, max_new, **LOOP_KW)
        if mode == "graph":
            loop.capture(n_graphs=n_graphs or 2)
        res = {}
        try:
            for pm in ("reference", "grouped", "grouped"):
                with _env(DSV41_PREFILL=pm, DSV41_PREFILL_STREAMS=streams):
                    r = loop.generate(ids, max_new, stop_on_eos=False, parity=True)
                res.setdefault(pm, []).append({"ids": list(r["token_ids"]), "acc": list(r["accepts"]),
                                               "sha": list(r["step_sha256"]), "lp": dict(loop.last_prefill),
                                               "prefill_ms": r["prefill_ms"]})
        finally:
            loop.close()
        a = res["reference"][0]
        for g in res["grouped"]:
            assert g["lp"]["mode"] == "grouped", g["lp"]
            assert g["ids"] == a["ids"], f"{kind} L={L} {mode}: stream differs after the grouped prefill"
            assert g["acc"] == a["acc"], f"{kind} L={L} {mode}: accept lengths differ"
            assert g["sha"] == a["sha"], (f"{kind} L={L} {mode}: logits rows differ at "
                                          f"{[i for i, (x, y) in enumerate(zip(a['sha'], g['sha'])) if x != y][:8]}")
        assert a["lp"]["mode"] == "reference"
        COUNTS["gates"] += 1
        COUNTS["tokens"] += len(a["ids"])
        COUNTS["logit_rows"] += len(a["sha"])
        tag = f"{kind} L={L} mode={mode}{'/' + str(n_graphs or 2) + 'g' if mode == 'graph' else ''}"
        print(f"  {tag}: loop after grouped prefill == after reference prefill ({len(a['ids'])} tokens, accepts "
              f"{a['acc'][:6]}..., {len(a['sha'])} logits rows); prefill ms ref {a['prefill_ms']:.1f} grouped "
              f"{res['grouped'][0]['prefill_ms']:.1f} (mini: not a speed gate)")
        REPORT.append(f"{tag}: tokens sha "
                      f"{hashlib.sha256(str(a['ids']).encode()).hexdigest()[:12]} equal under both prefills")
        return res


# ------------------------------------------------------------------------------------------------ tests
def test_moe_layers_grouped_equal():
    _need_tilelang()
    _gate_moe_layers("mini4", 29)
    _gate_moe_layers("mini3", 61)


def test_prefill_grouped_equal_caches_and_replay():
    _need_tilelang()
    _gate_prefill("mini4", 37)
    _gate_prefill("mini3", 130, streams=1)      # past the 128-window: the window ring wraps inside the prefill


def test_loop_stream_after_grouped_prefill_static():
    _need_tilelang()
    _gate_stream("mini3", 17, 48, mode="static")
    _gate_stream("mini3", 18, 48, mode="static", streams=1)


def test_loop_stream_after_grouped_prefill_graph():
    _need_tilelang()
    _gate_stream("mini3", 17, 48, mode="graph")


TESTS = [test_moe_layers_grouped_equal, test_prefill_grouped_equal_caches_and_replay,
         test_loop_stream_after_grouped_prefill_static, test_loop_stream_after_grouped_prefill_graph]

if __name__ == "__main__":
    import traceback

    results = {}
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            results[t.__name__] = "PASS"
        except unittest.SkipTest as e:
            results[t.__name__] = f"SKIP ({e})"
        except Exception:
            traceback.print_exc()
            results[t.__name__] = "FAIL"
        print(f"{results[t.__name__]:4s} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    for line in REPORT:
        print(f"  note: {line}")
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"comparisons executed: {COUNTS}")
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
