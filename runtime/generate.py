"""T5 decode driver: greedy 1-row generation on the reference `Transformer`, emitting the W0 parity JSONL record.

    gen = Generator(model, "/ref")                 # model = runtime.loader.build_and_load(...)
    ids = gen.encode([{"role": "user", "content": "hi"}])
    rec = gen.greedy(ids, max_new=256)             # one parity-A.jsonl record (minus tag/prompt, which the caller sets)

Call pattern = reference generate.py, exactly (M0-PORT-PLAN.md section 5(e)):
  * prefill: `model(tokens[:, 0:L], 0)` (generate.py:64-74, first loop iteration has prev_pos 0);
  * decode:  `model(tokens[:, p:p+1], p)` with p = the position of the token being fed (generate.py:67-78: prev_pos
    becomes cur_pos after each forward), so the first decode step feeds the prefill's argmax at position L;
  * the forward returns `(output_ids, logits, main_hidden)` (model.py:1271-1275); generate.py keeps `[0]`
    (generate.py:74). `logits` is fp32 `[b, vocab]` for the last position only (model.py:1010-1014);
  * greedy = `sample(logits, 0)` = `logits.argmax(-1)` (model.py:1288-1292). generate.py sets `args.temperature`
    from the CLI (generate.py:123) and the model reads it at construction (model.py:1197); the loader builds from
    config.json, which carries no temperature, so ModelArgs' default 1 applies (model.py:53) and `Generator` sets
    `model.temperature = 0` -- the same attribute generate.py controls, not a code change;
  * generate.py runs `max_new` forwards in total (prefill + max_new-1 decode steps; loop bound generate.py:50, 67),
    stops on EOS (generate.py:77-80). Here EOS is *kept* in the record (`n_tokens` and `logprobs` include it, `text`
    does not), because that is what the W0 server capture shows (parity-A.jsonl json/tool/story rows).

KV caches and reset between prompts: the caches are model-owned buffers sized `max_batch_size` x ... (window ring
model.py:665-668, compressed KV 672-680, compressor state 452-456, index K 521-525, Engram hash cache
engram.py:155-157); 1 row = batch slot 0. The reference never resets them: every `generate()` call simply starts at
prev_pos 0 (generate.py:64), and interactive mode re-prefills the whole conversation from position 0 into the same
model (generate.py:137-160). A fresh prompt overwrites from pos 0 and nothing stale is ever read:
  * window ring: prefill writes slots [:seqlen] (model.py:709-711); decode masks every slot > start_pos with -1
    (model.py:424), so slots left by a longer previous prompt are invisible until overwritten;
  * compressor state: slots [:remainder] written at prefill (model.py:466-472), decode fills slot start_pos % ratio
    and pools only when the group completes (model.py:477-482) -- every pooled slot is from this sequence;
  * compressed KV / index K: written at [start_pos//ratio ...] before being read as [:end_pos // ratio]
    (model.py:547, 555, 762-764);
  * Engram hash cache: written at [start_pos:start_pos+seqlen] (engram.py:167), gathered only at positions <= the
    current one, look-back before 0 blocked (engram.py:172-173);
  * `SharedAttentionRuntime`: "nothing needs resetting between forwards" (model.py:1170-1172).
So: **no reset; a fresh prompt overwrites from position 0** (generate.py:64).

Logprobs = `log_softmax` over the full vocab in fp32 of the model's fp32 logits (on CPU, after one D2H copy, so the
same code is what the Mac tests exercise), top-k by a *stable* descending sort (ties -> lower id first, matching
argmax's first-max rule). Token strings = single-id decode with special tokens kept (what the W0 capture shows:
`<｜end▁of▁sentence｜>` appears as a token; `bytes` = its UTF-8). Every step's fp32 logits are also hashed
(`last_meta["logits_sha256"]`): self-replay compares that digest, i.e. all logits bitwise, not just the top-20.

Profiling (M0-PORT-PLAN.md section 5(f)): NVTX range "decode_step" around every `step` (and "prefill"); optional
per-layer ranges (`install_layer_nvtx`); `arm_profile(warm, steps)` calls `torch.cuda.profiler.start()` before the
(warm+1)-th following `step` and `stop()` after `warm+steps` of them -- for `nsys --capture-range=cudaProfilerApi`.

Determinism (`set_determinism`): `torch.manual_seed(0)`, `torch.use_deterministic_algorithms(True, warn_only=True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (must be in the environment before cuBLAS creates its handle; first_light sets it
before importing torch and the docker line passes it too). With deterministic algorithms on, torch also fills
`torch.empty` outputs with NaN (`torch.utils.deterministic.fill_uninitialized_memory`), which surfaces any read of
uninitialized memory but costs a fill kernel per allocation; it is reported and switchable. Everything is greedy, so
sampling never matters.

Process-global side effects: `Generator` calls `torch.set_default_device(device)` (generate.py:134: the forward builds
index tensors on the default device).

Modes (T8, docs/T8-T9A-DESIGN.md section 2): `mode="eager"` (default) is the reference call above; `"static"` runs the
fixed-shape step of runtime/static_step.py eagerly (`StaticStep.forward_static`, graph off); `"graph"` replays its two
parity CUDA graphs (`StaticStep.replay`). Prefill is the reference's in every mode. In `"graph"` mode the graphs are
captured on the first prefill (capture runs its own dummy prefill + warm-ups, which the real prefill then overwrites
from position 0). In static/graph modes `step` returns the StaticStep's static logits buffer (valid until the next
step) and `next_token_tensor()` its static `tok`; the record format of `greedy` is unchanged, and `last_meta` gains
`step_sha256` (one fp32-logit digest per position) and `mode`. `marker=True` launches the fixed-name Triton kernel
`t8_step_marker` once per decode step (inside the graph in graph mode) so nsys finds step boundaries after replay.
T9a-1: `moe_impl="ref"|"dg-masked"` selects the static step's routed-MoE implementation (runtime/dg_moe.py; `ref` is
the bit-identical T8 path); `last_meta["moe_impl"]` records it; `logits_sink(i, lc)` (default None) sees every
position's fp32 CPU logits in `greedy` (first_light's dg-vs-ref drift row).
T9b-1: `fp8_impl="ref"|"gemv"` selects the static step's fp8 Linear implementation (runtime/gemv_mxfp8.py; `ref` is
the bit-identical reference `fp8_gemm`), with the same rebuild-on-change rule as `moe_impl`; `last_meta["fp8_impl"]`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED_ENCODING = os.path.join(REPO, "reference", "deepseek-inference", "encoding", "encoding.py")
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MODES = ("eager", "static", "graph")
MOE_IMPLS = ("ref", "dg-masked")  # = runtime.static_step.MOE_IMPLS (not imported: that module needs the reference)
FP8_IMPLS = ("ref", "gemv")       # = runtime.static_step.FP8_IMPLS (T9b-1)
RECORD_KEYS = ("tag", "prompt", "model", "finish_reason", "n_tokens", "text", "logprobs", "usage")


# ------------------------------------------------------------------------------------------------ determinism
def set_determinism(seed: int = 0, fill_uninitialized_memory: Optional[bool] = None) -> dict:
    """The determinism knobs (plan section 5(e)). Returns what is in force, for the report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    det = getattr(torch.utils, "deterministic", None)
    if fill_uninitialized_memory is not None and det is not None and hasattr(det, "fill_uninitialized_memory"):
        det.fill_uninitialized_memory = bool(fill_uninitialized_memory)
    return {
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "fill_uninitialized_memory": getattr(det, "fill_uninitialized_memory", None) if det is not None else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


# ------------------------------------------------------------------------------------------------ tokenizer side
def load_encoding(tokenizer_dir: Optional[str] = None):
    """The checkpoint's own `encoding/encoding.py` (`<tokenizer_dir>/encoding/encoding.py`, where render_fixture.py
    found it on the box), else the vendored copy of the same file. Imported by path under a private module name so
    it cannot collide with anything called `encoding`. Returns (module, path)."""
    cands = []
    if tokenizer_dir:
        cands.append(os.path.join(tokenizer_dir, "encoding", "encoding.py"))
    cands.append(VENDORED_ENCODING)
    for path in cands:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("_dsv41_encoding", path)
            mod = importlib.util.module_from_spec(spec)
            prev = sys.dont_write_bytecode
            sys.dont_write_bytecode = True  # never write under reference/ or /ref
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.dont_write_bytecode = prev
            return mod, path
    raise FileNotFoundError(f"no encoding.py in {cands}")


def load_tokenizer(tokenizer_dir: str):
    """`tokenizers.Tokenizer` from `<dir>/tokenizer.json` (render_fixture.py:9)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))


def _token_content(t) -> Optional[str]:
    if isinstance(t, dict):
        return t.get("content")
    return t if isinstance(t, str) else None


def eos_ids_from(tokenizer_dir: Optional[str], tokenizer) -> List[int]:
    """EOS id(s): `eos_token` of tokenizer_config.json, plus `eos_token_id` of generation_config.json / config.json
    when present (int or list). Sorted, unique."""
    ids = set()
    if tokenizer_dir:
        tc = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if os.path.exists(tc):
            with open(tc) as f:
                content = _token_content(json.load(f).get("eos_token"))
            if content is not None and tokenizer is not None:
                i = tokenizer.token_to_id(content)
                if i is not None:
                    ids.add(int(i))
        for fn in ("generation_config.json", "config.json"):
            p = os.path.join(tokenizer_dir, fn)
            if os.path.exists(p):
                with open(p) as f:
                    v = json.load(f).get("eos_token_id")
                if isinstance(v, int):
                    ids.add(v)
                elif isinstance(v, list):
                    ids.update(int(x) for x in v)
    return sorted(ids)


def encode_messages_to_ids(enc, tokenizer, messages, thinking_mode: str = "chat") -> List[int]:
    """The exact call that reproduced Pg61's prompt token counts 8/8 (runtime/tools/render_fixture.py:19-22):
    `encode_messages(messages, thinking_mode="chat")` (BOS is in the rendered text, encoding.py:810-815), then
    `tokenizer.encode(prompt, add_special_tokens=False)`."""
    prompt = enc.encode_messages(messages, thinking_mode=thinking_mode)
    if isinstance(prompt, tuple):
        prompt = prompt[0]
    return list(tokenizer.encode(prompt, add_special_tokens=False).ids)


# ------------------------------------------------------------------------------------------------ record shaping
def logprob_entry(logits: torch.Tensor, token_id: int, top_k: int, token_str: Callable[[int], str]) -> dict:
    """One `logprobs[i]` element of the parity record from one position's logits ([vocab], any float dtype, any
    device): fp32 log_softmax over the full vocab; the chosen token's logprob; the top-k alternatives by a stable
    descending sort (ties -> lower id first)."""
    lp = torch.log_softmax(logits.detach().to("cpu", torch.float32).reshape(-1), dim=-1)
    order = torch.sort(lp, descending=True, stable=True).indices[:top_k].tolist()

    def ent(i: int) -> dict:
        s = token_str(i)
        return {"token": s, "logprob": float(lp[i]), "bytes": list(s.encode("utf-8"))}

    e = ent(int(token_id))
    e["top_logprobs"] = [ent(i) for i in order]
    return e


def make_record(*, tag: Optional[str], prompt: Optional[str], model_name: str, prompt_tokens: int,
                gen_ids: Sequence[int], entries: List[dict], finish_reason: str, text: str) -> dict:
    """The parity JSONL record (w0/results/phase1/parity-A.jsonl's fields, minus the volatile `wall_s`, so two
    replays are byte-identical files)."""
    n = len(gen_ids)
    return {"tag": tag, "prompt": prompt, "model": model_name, "finish_reason": finish_reason, "n_tokens": n,
            "text": text, "logprobs": entries,
            "usage": {"prompt_tokens": int(prompt_tokens), "total_tokens": int(prompt_tokens) + n,
                      "completion_tokens": n}}


def dump_record(rec: dict) -> str:
    """One JSONL line. Deterministic (fixed key order, repr-exact floats)."""
    return json.dumps(rec, ensure_ascii=True)


def record_divergence(a: dict, b: dict) -> Optional[dict]:
    """First difference between two records of the same prompt, or None when byte-identical as JSONL lines.
    Reports the first token index whose id/string differs, else the first position whose logprob entry differs,
    else the first differing top-level field."""
    if dump_record(a) == dump_record(b):
        return None
    la, lb = a.get("logprobs") or [], b.get("logprobs") or []
    for i in range(min(len(la), len(lb))):
        if la[i].get("token") != lb[i].get("token"):
            return {"kind": "token", "index": i, "a": la[i].get("token"), "b": lb[i].get("token")}
    for i in range(min(len(la), len(lb))):
        if la[i] != lb[i]:
            d = abs(float(la[i].get("logprob", 0.0)) - float(lb[i].get("logprob", 0.0)))
            return {"kind": "logprob", "index": i, "abs_dlogprob_chosen": d}
    if len(la) != len(lb):
        return {"kind": "length", "index": min(len(la), len(lb)), "a": len(la), "b": len(lb)}
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            return {"kind": "field", "field": k}
    return {"kind": "serialization"}


# ------------------------------------------------------------------------------------------------ NVTX
def _nvtx():
    try:
        return torch.cuda.nvtx
    except Exception:  # pragma: no cover
        return None


def install_layer_nvtx(model) -> List:
    """NVTX ranges L{i} -> {engram, attn, ffn} as forward pre/post hooks (plan section 5(f)). Hooks only; the model
    is untouched. Returns the hook handles (call .remove() on each to uninstall)."""
    nv = _nvtx()
    handles = []

    def wrap(mod, name):
        # hooks MUST return None: a forward_pre_hook's non-None return replaces the module's inputs, and
        # nvtx.range_push returns the range depth (an int) — that silently fed `(depth,)` to Engram.forward.
        def pre(m, i, n=name):
            nv.range_push(n)
        def post(m, i, o):
            nv.range_pop()
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    for i, layer in enumerate(model.layers):
        wrap(layer, f"L{i}")
        if getattr(layer, "engram", None) is not None:
            wrap(layer.engram, "engram")  # runs before the block, outside L{i} (model.py:1265-1266)
        wrap(layer.attn, "attn")
        wrap(layer.ffn, "moe")
    wrap(model.head, "head")
    return handles


# ------------------------------------------------------------------------------------------------ the driver
class Generator:
    """Greedy 1-row driver over the reference forward. `model` = what `build_and_load` returns (or any object with
    the reference's `__call__(tokens, start_pos) -> (ids, logits, hidden)` and `max_seq_len`).

    tokenizer_dir: checkpoint root (tokenizer.json, tokenizer_config.json, encoding/); `tokenizer` / `eos_ids`
    override what is read from it (the mini-checkpoint GPU test and the Mac tests use small in-memory tokenizers)."""

    def __init__(self, model, tokenizer_dir: Optional[str] = None, *, tokenizer=None,
                 eos_ids: Optional[Iterable[int]] = None, device=None, top_k: int = 20, nvtx: bool = True,
                 model_name: str = "m0-reference-runtime", profiler=None, set_default_device: bool = True,
                 mode: str = "eager", max_pos: Optional[int] = None, marker: bool = False,
                 static_counters: bool = True, static_factory: Optional[Callable] = None, moe_impl: str = "ref",
                 fp8_impl: str = "ref"):
        self.model = model
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = tokenizer if tokenizer is not None else (load_tokenizer(tokenizer_dir) if tokenizer_dir else None)
        self.eos_ids = sorted(set(int(i) for i in eos_ids)) if eos_ids is not None else eos_ids_from(tokenizer_dir, self.tokenizer)
        self._enc = None
        self.encoding_path = None
        if device is None:
            p = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
            device = p.device if p is not None else torch.device("cpu")
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if set_default_device:
            torch.set_default_device(self.device)  # generate.py:134
        if hasattr(model, "temperature"):
            model.temperature = 0.0  # greedy: sample() -> argmax (model.py:1291-1292); generate.py:123 sets the same
        self.top_k = top_k
        self.nvtx = nvtx and self.is_cuda
        self.model_name = model_name
        self.max_seq_len = int(getattr(model, "max_seq_len", 1 << 30))
        self._tok_cache: Dict[int, str] = {}
        self._last_ids: Optional[torch.Tensor] = None
        self.n_steps = 0  # decode steps run by this Generator (profile window counts these)
        self._profiler = profiler if profiler is not None else (torch.cuda.profiler if self.is_cuda else None)
        self.profile: Optional[dict] = None
        self.pool = getattr(getattr(model, "runtime_load", None), "pool", None)
        self.last_meta: dict = {}
        # T8 modes
        self.marker = bool(marker)
        self.static_counters = bool(static_counters)
        self.static_factory = static_factory
        self.static = None  # runtime.static_step.StaticStep (or the factory's stand-in), built on first use
        self._static_max_pos = max_pos
        self._marker_ctr = None
        self.moe_impl = "ref"
        self.set_moe_impl(moe_impl)
        self.fp8_impl = "ref"
        self.set_fp8_impl(fp8_impl)
        self.logits_sink: Optional[Callable] = None  # greedy calls sink(i, fp32 CPU logits) per position (drift rows)
        self.mode = "eager"
        self.set_mode(mode)

    # ---------------------------------------------------------------- modes (T8)
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        self.mode = mode

    def set_moe_impl(self, impl: str) -> None:
        """T9a-1: the static step's routed-MoE implementation ("ref" | "dg-masked"). A change drops the StaticStep (and
        its graphs); the next static/graph use rebuilds it with the new impl. Eager mode never uses it."""
        if impl not in MOE_IMPLS:
            raise ValueError(f"moe_impl {impl!r} not in {MOE_IMPLS}")
        if impl != self.moe_impl and self.static is not None:
            self.reset_static()
        self.moe_impl = impl

    def set_fp8_impl(self, impl: str) -> None:
        """T9b-1: the static step's fp8 Linear implementation ("ref" | "gemv"). A change drops the StaticStep (and its
        graphs); the next static/graph use rebuilds it. Eager mode never uses it."""
        if impl not in FP8_IMPLS:
            raise ValueError(f"fp8_impl {impl!r} not in {FP8_IMPLS}")
        if impl != self.fp8_impl and self.static is not None:
            self.reset_static()
        self.fp8_impl = impl

    @property
    def max_pos(self) -> int:
        """Largest decode position this Generator can serve in its current mode."""
        if self.mode == "eager":
            return self.max_seq_len - 1
        mp = self._static_max_pos
        return self.max_seq_len - 1 if mp is None else int(mp)

    def static_step(self):
        """The StaticStep (built on first call: pointer tables, staging, static buffers; no capture)."""
        if self.static is None:
            if self.static_factory is not None:
                # factories predating T9a-1/T9b-1 take only the model; an impl is passed only when it is not "ref"
                kw = {} if self.moe_impl == "ref" else {"moe_impl": self.moe_impl}
                if self.fp8_impl != "ref":
                    kw["fp8_impl"] = self.fp8_impl
                self.static = self.static_factory(self.model, **kw)
            else:
                from runtime.static_step import StaticStep

                self.static = StaticStep(self.model, self.pool, max_pos=self._static_max_pos, marker=self.marker,
                                         counters=self.static_counters, moe_impl=self.moe_impl,
                                         fp8_impl=self.fp8_impl)
        return self.static

    def reset_static(self) -> None:
        """Drop the StaticStep and its graphs (a "relaunch" of the static path; the next use rebuilds it)."""
        if self.static is not None and hasattr(self.static, "detach"):
            self.static.detach()
        self.static = None
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def capture(self, fills: bool = False) -> dict:
        """(Re)capture g_even/g_odd. Must run before a real prefill (it prefills a dummy prompt itself)."""
        return self.static_step().capture(fills=fills)

    def _ensure_captured(self) -> None:
        if self.mode == "graph" and not self.static_step().captured:
            self.capture()  # before the real prefill: capture prefills a dummy prompt and runs warm-up steps

    def _marker_eager(self) -> None:
        if not (self.marker and self.is_cuda):
            return
        from runtime.static_kernels import step_marker

        if self._marker_ctr is None:
            self._marker_ctr = torch.zeros(1, dtype=torch.int32, device=self.device)
        step_marker(self._marker_ctr)

    # ---------------------------------------------------------------- text
    @property
    def enc(self):
        if self._enc is None:
            self._enc, self.encoding_path = load_encoding(self.tokenizer_dir)
        return self._enc

    def encode(self, messages, thinking_mode: str = "chat") -> List[int]:
        return encode_messages_to_ids(self.enc, self.tokenizer, messages, thinking_mode)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_str(self, i: int) -> str:
        """Single-id decode, special tokens kept (how the W0 server capture renders a logprob token)."""
        s = self._tok_cache.get(i)
        if s is None:
            s = self.tokenizer.decode([int(i)], skip_special_tokens=False)
            self._tok_cache[i] = s = "" if s is None else s
        return s

    # ---------------------------------------------------------------- profiling
    def arm_profile(self, warm: int, steps: int) -> dict:
        """cudaProfilerStart before the (warm+1)-th decode step from now, cudaProfilerStop after warm+steps."""
        self.profile = {"start_at": self.n_steps + warm, "stop_at": self.n_steps + warm + steps,
                        "started": None, "stopped": None}
        return self.profile

    def _profile_before(self) -> None:
        p = self.profile
        if p and p["started"] is None and self.n_steps == p["start_at"] and self._profiler is not None:
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
            self._profiler.start()
            p["started"] = self.n_steps

    def _profile_after(self) -> None:
        p = self.profile
        if p and p["started"] is not None and p["stopped"] is None and self.n_steps == p["stop_at"]:
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
            self._profiler.stop()
            p["stopped"] = self.n_steps

    # ---------------------------------------------------------------- forward
    @torch.inference_mode()
    def prefill(self, ids: Sequence[int]) -> torch.Tensor:
        """`model(tokens[:, 0:L], 0)` (generate.py:69-74). Returns fp32 logits [vocab] of the last position (on the
        model's device); the model's own greedy id is kept for `next_token()`."""
        if len(ids) == 0:
            raise ValueError("empty prompt")
        if len(ids) > self.max_seq_len:
            raise ValueError(f"prompt {len(ids)} > max_seq_len {self.max_seq_len}")
        self._ensure_captured()
        tokens = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        nv = _nvtx() if self.nvtx else None
        if nv:
            nv.range_push("prefill")
        try:
            out_ids, logits, _ = self.model(tokens, 0)
        finally:
            if nv:
                nv.range_pop()
        self._last_ids = out_ids
        return logits[0]

    @torch.inference_mode()
    def step(self, token, pos: int) -> torch.Tensor:
        """`model(tokens[:, pos:pos+1], pos)` (generate.py:69-78): feed `token` (int, or a device tensor holding one
        id -- no host sync) at position `pos`. NVTX "decode_step"; profiler window hooks. In "static"/"graph" mode the
        StaticStep runs instead (eagerly / as a graph replay); the returned logits are then its static buffer."""
        if pos >= self.max_seq_len:
            raise ValueError(f"position {pos} >= max_seq_len {self.max_seq_len}")
        static = self.mode != "eager"
        if not static:
            if isinstance(token, torch.Tensor):
                tokens = token.reshape(1, 1).to(device=self.device, dtype=torch.long)
            else:
                tokens = torch.tensor([[int(token)]], dtype=torch.long, device=self.device)
        else:
            st = self.static_step()
            if self.mode == "graph" and not st.captured:
                raise RuntimeError("graph mode: capture() must run before the prefill (prefill() does it)")
        self._profile_before()
        nv = _nvtx() if self.nvtx else None
        if nv:
            nv.range_push("decode_step")
        try:
            if static:
                logits = st.step(token, pos, graph=self.mode == "graph")
                out_ids = st.tok.view(1)
            else:
                self._marker_eager()
                out_ids, logits, _ = self.model(tokens, pos)
        finally:
            if nv:
                nv.range_pop()
        self.n_steps += 1
        self._profile_after()
        self._last_ids = out_ids
        return logits[0]

    def next_token_tensor(self) -> torch.Tensor:
        """The model's greedy id from the last forward, still on the device (`[1]`)."""
        return self._last_ids

    # ---------------------------------------------------------------- counters
    def cold_totals(self) -> Optional[Tuple[int, int, List[int], List[int]]]:
        """(hot calls, cold calls, hot per layer, cold per layer) from expert_pool's counters, or None. In static/graph
        mode the decode steps' routed calls come from the StaticStep's device counters (the reference experts are not
        called, so their hooks do not fire); the prefill's still come from the hooks. Reading them syncs."""
        c = getattr(self.pool, "counters", None) if self.pool is not None else None
        if not c:
            return None
        hot_l, cold_l = list(c["hot_calls"]), list(c["cold_calls"])
        cc = self.static.cold_counts() if self.static is not None and hasattr(self.static, "cold_counts") else None
        if cc is not None:
            cold_dev, n, k = cc
            hot_l = [h + n * k - cd for h, cd in zip(hot_l, cold_dev)]
            cold_l = [x + cd for x, cd in zip(cold_l, cold_dev)]
        return sum(hot_l), sum(cold_l), hot_l, cold_l

    # ---------------------------------------------------------------- greedy
    @torch.inference_mode()
    def greedy(self, ids: Sequence[int], max_new: int, top_k: Optional[int] = None, *, tag: Optional[str] = None,
               prompt: Optional[str] = None, stop_on_eos: bool = True) -> dict:
        """Greedy generation of up to `max_new` tokens -> the parity JSONL record. `last_meta` gets the fp32-logits
        digest, timings and cold-hit deltas of this call."""
        top_k = self.top_k if top_k is None else top_k
        L = len(ids)
        if L + max_new > self.max_seq_len:
            raise ValueError(f"prompt {L} + max_new {max_new} > max_seq_len {self.max_seq_len}")
        if self.mode != "eager" and L + max_new - 2 > self.max_pos:
            raise ValueError(f"prompt {L} + max_new {max_new}: last decode position {L + max_new - 2} > max_pos "
                             f"{self.max_pos} of the static step")
        self._ensure_captured()
        eos = set(self.eos_ids)
        digest = hashlib.sha256()
        step_digests: List[str] = []
        c0 = self.cold_totals()
        t0 = time.perf_counter()
        logits = self.prefill(ids)
        gen: List[int] = []
        entries: List[dict] = []
        finish = "length"
        t_prefill = None
        pos = L
        for i in range(max_new):
            lc = logits.detach().to("cpu", torch.float32)  # one D2H copy; syncs
            if t_prefill is None:
                t_prefill = time.perf_counter() - t0
            if not bool(torch.isfinite(lc).all()):
                bad = int((~torch.isfinite(lc)).sum())
                raise FloatingPointError(f"position {pos}: {bad} non-finite logits (try --no-fill-uninit to see if "
                                         f"an uninitialized read is the source)")
            tok = int(self._last_ids.reshape(-1)[0])
            # the model's own argmax (model.py:1292) and ours on the copied logits must agree (first max wins)
            amax = int(torch.argmax(lc))
            if amax != tok:
                raise RuntimeError(f"position {pos}: model argmax {tok} != argmax of returned logits {amax}")
            digest.update(lc.numpy().tobytes())
            step_digests.append(hashlib.sha256(lc.numpy().tobytes()).hexdigest())
            if self.logits_sink is not None:
                self.logits_sink(i, lc)
            entries.append(logprob_entry(lc, tok, top_k, self.token_str))
            gen.append(tok)
            if stop_on_eos and tok in eos:
                finish = "stop"
                break
            if i == max_new - 1:
                break
            logits = self.step(tok, pos)
            pos += 1
        wall = time.perf_counter() - t0
        text_ids = gen[:-1] if finish == "stop" else gen
        rec = make_record(tag=tag, prompt=prompt, model_name=self.model_name, prompt_tokens=L, gen_ids=gen,
                          entries=entries, finish_reason=finish, text=self.decode(text_ids))
        c1 = self.cold_totals()
        meta = {"logits_sha256": digest.hexdigest(), "token_ids": gen, "prompt_tokens": L, "wall_s": wall,
                "prefill_s": t_prefill, "n_forward": len(gen), "mode": self.mode, "step_sha256": step_digests,
                "moe_impl": self.moe_impl, "fp8_impl": self.fp8_impl,
                "decode_ms_per_token": (wall - t_prefill) / max(1, len(gen) - 1) * 1e3 if len(gen) > 1 else None}
        if c0 is not None and c1 is not None:
            hot, cold = c1[0] - c0[0], c1[1] - c0[1]
            meta["cold"] = {"hot_calls": hot, "cold_calls": cold, "cold_share": cold / (hot + cold) if hot + cold else 0.0,
                            "cold_calls_per_layer": [b - a for a, b in zip(c0[3], c1[3])]}
        self.last_meta = meta
        return rec
