"""GPU half of the T5 driver tests, on the synthetic mini checkpoint (tests/mini_ckpt.py: 2 layers, 8 experts with 4
cold per layer in registered host memory, 2 Engram tables over UVA): prefill + 10 greedy decode steps through the
reference forward, self-replay byte-identical (records + fp32-logit digests), a longer prompt in between does not
leak into a replay (the "fresh prompt overwrites from pos 0" claim in runtime/generate.py), the --profile hooks and
per-layer NVTX hooks do not crash or change numerics, and bench/first_light.py's CUDA-event step cell agrees with
greedy token for token and sees the cold-hit counters. Skipped without CUDA/tilelang (Mac).

Run on the box (orchestrator), from the repo checkout, inside the library image with the GB300 by CDI:

  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \
    -v "$HOME/hermes/gb300-dsv41-megakernel":/work -w /work -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \
    --entrypoint python3 vllm/vllm-openai:v0.31.0 tests/gpu/test_generate_mini_gpu.py

(`python3 tests/gpu/test_generate_mini_gpu.py` runs every test without pytest. No checkpoint mount is needed: the
mini checkpoint and its WordLevel tokenizer are written to a temp dir.)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from runtime.refmodel import tilelang_available  # noqa: E402


def _need_tilelang():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA (Mac): GPU tests run on the box")
    if not tilelang_available():
        raise unittest.SkipTest("tilelang missing: run inside vllm/vllm-openai:v0.31.0")


def _first_light():
    spec = importlib.util.spec_from_file_location("bench_first_light", os.path.join(REPO, "bench", "first_light.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_mini(tmp):
    from mini_ckpt import MINI_CONFIG, make_mini_checkpoint
    from runtime.loader import build_and_load
    from runtime.refmodel import import_reference

    import_reference()
    import engram

    engram.build_compressed_token_map = lambda tok: (list(range(MINI_CONFIG["vocab_size"])),
                                                     MINI_CONFIG["engram_compressed_vocab_size"])
    ck = make_mini_checkpoint(tmp)
    return build_and_load(ck["model_dir"], ck["map"], "cuda", tokenizer=object(), io_threads=4, stage_bytes=1 << 20,
                          log=lambda *a: None)


def test_generate_mini_greedy_replay_profile_stepcell():
    _need_tilelang()
    from runtime.generate import Generator, dump_record, install_layer_nvtx, set_determinism
    from tiny_tok import make_wordlevel_tokenizer

    det = set_determinism(0)
    fl = _first_light()
    with tempfile.TemporaryDirectory() as tmp:
        model = _load_mini(os.path.join(tmp, "ckpt"))
        tokdir = os.path.join(tmp, "tok")
        make_wordlevel_tokenizer(2048, out_dir=tokdir)
        try:
            g = torch.Generator().manual_seed(5)  # CPU draws before Generator flips the default device to cuda
            ids = torch.randint(2, 2000, (16,), generator=g).tolist()
            long_ids = torch.randint(2, 2000, (150,), generator=g).tolist()  # > window 128: wraps the ring
            gen = Generator(model, tokdir)
            assert gen.eos_ids == [1] and gen.is_cuda and model.temperature == 0.0

            # prefill + 10 greedy decode steps
            n0 = gen.n_steps
            rec = gen.greedy(ids, 11, tag="mini", prompt="mini", stop_on_eos=False)
            meta = dict(gen.last_meta)
            assert gen.n_steps - n0 == 10 and rec["n_tokens"] == 11 and rec["finish_reason"] == "length"
            assert all(len(e["top_logprobs"]) == 20 for e in rec["logprobs"])
            assert all(e["logprob"] == e["logprob"] and e["logprob"] <= 0.0 for e in rec["logprobs"])  # finite, <= 0
            assert [e["token"] for e in rec["logprobs"]] == [gen.token_str(i) for i in meta["token_ids"]]
            # the first token is the argmax of a direct reference forward on the same prompt
            logits0 = model(torch.tensor([ids], device="cuda"), 0)[1][0]
            assert int(logits0.argmax()) == meta["token_ids"][0]
            cold = meta.get("cold")
            # counters count expert *calls* (one per active expert per layer per forward): 10 decode steps x 2 layers
            # x 2 active = 40, plus the prefill's distinct experts per layer (2..8 each, 2 layers)
            assert cold is not None and 40 + 4 <= cold["hot_calls"] + cold["cold_calls"] <= 40 + 16, cold

            # self-replay, with a longer prompt in between (stale caches must not leak)
            gen.greedy(long_ids, 8, stop_on_eos=False)
            rec2 = gen.greedy(ids, 11, tag="mini", prompt="mini", stop_on_eos=False)
            assert dump_record(rec2) == dump_record(rec), "self-replay records differ"
            assert gen.last_meta["logits_sha256"] == meta["logits_sha256"], "self-replay fp32 logits differ"

            # --profile hooks (cudaProfilerStart/Stop without nsys attached are no-ops) and per-layer NVTX
            w = gen.arm_profile(warm=2, steps=3)
            hooks = install_layer_nvtx(model)
            try:
                rec3 = gen.greedy(ids, 11, tag="mini", prompt="mini", stop_on_eos=False)
            finally:
                for h in hooks:
                    h.remove()
            assert w["started"] is not None and w["stopped"] == w["started"] + 3, w
            assert dump_record(rec3) == dump_record(rec) and gen.last_meta["logits_sha256"] == meta["logits_sha256"]

            # first_light's step cell: CUDA events, same tokens as greedy, counters per step
            sc = fl.step_cell(gen, ids, 20, stats_from=4)
            assert sc["timer"] == "cuda_events" and len(sc["step_ms"]) == 20 and sc["ms_per_step"]["n"] == 17
            assert all(x > 0 for x in sc["step_ms"])
            assert sc["token_ids"][:11] == meta["token_ids"], (sc["token_ids"][:11], meta["token_ids"])
            c = sc["cold"]
            assert c["routed_calls_per_step"]["min"] == c["routed_calls_per_step"]["max"] == 2 * 2, c
            assert len(c["cold_calls_per_layer"]) == 2
            print(f"  mini greedy 11 tok ok; replay equal; profile window {w}; step cell p50 "
                  f"{sc['ms_per_step']['p50']:.2f} ms, cold share/step mean {c['cold_share_per_step']['mean']:.3f}; "
                  f"determinism {det}")
        finally:
            torch.set_default_device("cpu")
            res = model.runtime_load
            gen = None  # noqa: F841  (drop every reference to the model's UVA views before unregistering)
            del model
            torch.cuda.synchronize()
            res.release()


TESTS = [test_generate_mini_greedy_replay_profile_stepcell]

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
    n_fail = sum(v == "FAIL" for v in results.values())
    print(f"{len(results) - n_fail}/{len(results)} not failed; {n_fail} failed")
    sys.exit(1 if n_fail else 0)
