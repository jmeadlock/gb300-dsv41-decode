#!/usr/bin/env python3
"""flashmla_probe.py -- T9b-3.0 (docs/T9B3-ATTN-DESIGN.md section 5): the one FlashMLA experiment plus the attention
timings the rest of T9b-3 depends on. No runtime change; nothing here touches runtime/.

Sections (each one try/except'd; the JSON receipt is rewritten after every section, so a crash never loses it):

  1 load     'vllm' not in sys.modules; glob <site-packages>/vllm/_flashmla_C*.so; its ELF NEEDED list (readelf -d,
             or a built-in ELF parser); torch.ops.load_library(.so); 'vllm' still not in sys.modules; the
             _flashmla_C ops registered (schemas of sparse_decode_fwd / sparse_prefill_fwd /
             fused_norm_rope_attn_rope_cast_decode). Read-only: FlashInfer's DSV4 sparse-MLA entry point (source grep,
             no import of the hit) and whether its cubins are bundled.
  2 values   Window KV through the reference act_quant (FP8 e4m3, ue8m0/32) and the compressed latent through
             fp4_act_quant(.., 16, e4m3), each in place AND not in place. Gates: torch dequant(codes, scales) == in-place
             bf16, bitwise; 528 B / 288 B paged records (data rows, then scale rows, per page) read back with vLLM's own
             dequant formulas (cache_utils.py:486-499 MXFP8, 543-566 NVFP4) re-implemented in torch == the reference
             bf16 cache, bitwise.
  3 outputs  Indices as the static step builds them (window oldest-first, leading -1 at p < 127; compressed sorted with
             a -1 tail; c > 512 -> 512 sorted picks as a top-k stand-in). Arms: A = reference TileLang sparse_attn
             (static 640 wide, and exact width = the T9b-3a skip, which must be bitwise equal), B = FlashMLA
             sparse_decode_fwd over 528 + 288 records (with per-row lengths as vLLM passes them; B0 = no lengths, -1
             masked), C = sparse_prefill_fwd over the bf16 caches, E = FlashInfer if callable. Each vs A bitwise
             (mismatch %, max bf16 ulp) and vs two fp64 oracles: q bf16, and q MXFP8-QDQ'd (act_quant 32/ue8m0). The
             nearer oracle settles whether the kernel quantizes q.
  4 speed    40 calls in one CUDA graph (2 ratio-0 + 18 ratio-2 + 20 ratio-1, per-layer window caches, one compressed
             cache per kv-source group), b = 1 and 6, positions 300 and 1100 (>= 512 r): A static (today), A exact
             (proxy for A-with-skip), B, B0, C; plus A at uniform widths 128/278/428/640 (per-64-block slope). Replay
             equality x2 and graph == eager for every graph. FlashMLA sched meta: one per layer type, created inside the
             captured body (flashmla.py:212-228).
  5 probes   section 3 (a)-(c) on 1e5 rows per width 128/512/1280/5120/16384 (real-weight projections of synthetic
             hidden states where the checkpoint has the weight; else synthetic, labelled): (a) torch square().mean(-1)
             and rsqrt M=1 vs M=6 vs M=N; (b) which explicit fp32 summation order (emulated with exact elementwise
             adds = what a fixed-order Triton kernel computes; plus Triton tl.sum) matches torch at M=1; (c) torch.rsqrt
             vs libdevice rsqrt / rsqrt_rn / PTX rsqrt.approx(.ftz) / div_rn(1, sqrt_rn). Plus the ratio-2 compressor's
             softmax(dim=1) over 2 slots, and the indexer einsum / weights_proj / head-sum, M=1 vs 6.

Every gate prints PASS/FAIL with its numbers. Decision (design section 5): ADOPT B iff it loads without vllm, its fp64
error is <= 2x A's and nearer the bf16-q oracle, it is >= 2x faster than A-exact at b = 1 and 6, and replay is equal;
PARK if q is quantized (A1-class -> James) or it needs vllm.

Box command (same image and mounts as the T9a/T9b runs; checkpoint read-only at /ref; receipt in $W/scratch):

  W=$HOME/hermes/gb300-dsv41-decode
  docker run --rm --device nvidia.com/gpu=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55 --ipc=host --ulimit memlock=-1:-1 \\
    -v "$W":/work -w /work \\
    -v /models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170:/ref:ro \\
    -v "$HOME/dsv41/w0/jit-cache/tilelang":/root/.tilelang -v "$HOME/dsv41/w0/jit-cache/triton":/root/.triton \\
    -v "$HOME/dsv41/w0/jit-cache/nv":/root/.nv \\
    --entrypoint python3 vllm/vllm-openai:v0.31.0 bench/flashmla_probe.py --model-dir /ref \\
      --json /work/scratch/flashmla-probe.json

Options: --sections load,values,outputs,speed,probes (default all); --probe-rows 100000; --page 64; --reps 50;
--arm-e (attempt the FlashInfer call even when its cubins are not bundled -- may trigger a cubin download).
Never imports vllm; the FlashMLA ops come from the .so via torch.ops.load_library only.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

SITE = "/usr/local/lib/python3.12/dist-packages"
REF_INFERENCE = "/ref/inference"

H, D, RD, WIN = 64, 512, 64, 128
SCALE = D ** -0.5
TOPK = 512
E8 = torch.float8_e8m0fnu
E4 = torch.float8_e4m3fn
BF = torch.bfloat16
U8 = torch.uint8
TYPES = [0] * 2 + [2] * 18 + [1] * 20           # compress_ratios[0:40] (config.json)
BASE_LAYER = {0: 0, 2: 2, 1: 20}                # real-weight donor layer per layer type
MAX_POS = 4095
WR = {2: (MAX_POS + 1) // 2, 1: (MAX_POS + 1) // 1}   # static compressed width W_r (rt/static_step.py:337)
POS_CORRECT = (60, 300, 1100)
POS_SPEED = (300, 1100)
BS = (1, 6)
A_WIDTHS = (128, 278, 428, 640)


def group_of(layer):
    return 2 if layer < 8 else 8 if layer < 14 else 14 if layer < 20 else 20


def vllm_mods():
    return sorted(m for m in sys.modules if m == "vllm" or m.startswith("vllm."))


# ------------------------------------------------------------------------------------------------ receipt plumbing
OUT = {"meta": {}, "load": {}, "values": {}, "outputs": {}, "speed": {}, "probes": {}, "gates": {}, "decision": {},
       "errors": {}}
JSON_PATH = ["/work/scratch/flashmla-probe.json"]
ST = {}  # cross-section state: OPS, K, DATA, ...


def dump():
    try:
        d = os.path.dirname(JSON_PATH[0])
        if d:
            os.makedirs(d, exist_ok=True)
        with open(JSON_PATH[0], "w") as fh:
            json.dump(OUT, fh, indent=1, default=str)
    except Exception as e:  # noqa: BLE001
        print("WARN could not write receipt:", repr(e), flush=True)


def gate(name, ok, **nums):
    OUT["gates"][name] = dict(nums, pass_=bool(ok))
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(nums, default=str), flush=True)
    return bool(ok)


def info(name, **nums):
    print("INFO " + name + " " + json.dumps(nums, default=str), flush=True)


def section(name, fn):
    t0 = time.time()
    print("=" * 30, name, "=" * 30, flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        OUT["errors"][name] = traceback.format_exc()[-4000:]
        print("ERROR in section", name, repr(e)[:800], flush=True)
    OUT["meta"].setdefault("section_s", {})[name] = round(time.time() - t0, 1)
    dump()


def attempt(store, key, fn):
    """Run one arm/probe; record its result or its error; never raise."""
    try:
        r = fn()
        store[key] = r
        return r
    except Exception as e:  # noqa: BLE001
        store[key] = {"err": repr(e)[:1500]}
        print("FAIL(err) " + key + " " + repr(e)[:600], flush=True)
        return None


def sync():
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ numerics helpers
def bits16(x):
    return x.contiguous().view(torch.int16).to(torch.int32)


def bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 2:
        return bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))
    if a.element_size() == 4:
        return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))
    return bool(torch.equal(a.contiguous().view(U8), b.contiguous().view(U8)))


def bf16_cmp(a, b):
    """mismatch % of bf16 bit patterns and max ulp distance (sign-magnitude ordered, +0 == -0 is 0 ulp)."""
    ai, bi = bits16(a.to(BF)), bits16(b.to(BF))
    mism = (ai != bi).float().mean().item() * 100.0
    oa = torch.where(ai < 0, -(ai & 0x7FFF), ai)
    ob = torch.where(bi < 0, -(bi & 0x7FFF), bi)
    ulp = (oa - ob).abs().max().item() if ai.numel() else 0
    return {"mismatch_pct": round(mism, 5), "max_ulp": int(ulp)}


def err64(o, ref):
    d = o.double() - ref
    n = ref.norm().item()
    return {"rel": d.norm().item() / n if n > 0 else float("nan"), "max_abs": d.abs().max().item(),
            "mean_abs": d.abs().mean().item()}


# ------------------------------------------------------------------------------------------------ reference math (literal copies; rt/static_step.py and model.py cite)
def rms(x, w, eps=1e-20):
    """model.py:288-293."""
    dt = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (w * x).to(dt)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """model.py:368-389."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_(x, f):
    """model.py:392-406 for x [..., rd] with f broadcastable complex [..., rd/2]; in place."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    x.copy_(torch.view_as_real(xc * f).flatten(-2))
    return x


def lin_fp8(x, w, s):
    """model.py:196-205: act_quant(x, 32, ue8m0) + fp8_gemm(block 32)."""
    K = ST["K"]
    xq, xs = K.act_quant(x.contiguous(), 32, "ue8m0", E8)
    return K.fp8_gemm(xq, xs, w, s, E8, block_size=32)


def hidden(n, seed, width=5120):
    """Synthetic residual-stream-like rows: N(0,1) x a fixed per-channel log-normal scale, 16 outlier channels x10."""
    f32 = torch.float32
    g = torch.Generator(device="cuda")
    g.manual_seed(1234 + width)
    ch = torch.exp(0.5 * torch.randn(width, generator=g, device="cuda", dtype=f32))
    ch[torch.randperm(width, generator=g, device="cuda")[:16]] *= 10.0
    g.manual_seed(int(seed))
    out = torch.empty(n, width, dtype=BF, device="cuda")
    for i in range(0, n, 16384):
        m = min(16384, n - i)
        out[i:i + m] = (torch.randn(m, width, generator=g, device="cuda", dtype=f32) * ch).to(BF)
    return out


def blocky(n, width, seed, block=32):
    """Fallback values: N(0,1) with per-block magnitudes 2^-4..2^4 (exercises the QDQ scales)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed))
    x = torch.randn(n, width // block, block, generator=g, device="cuda", dtype=torch.float32)
    x = x * torch.exp2(torch.randint(-4, 5, (n, width // block, 1), generator=g, device="cuda").float())
    return x.view(n, width).to(BF)


# ------------------------------------------------------------------------------------------------ checkpoint
class Ckpt:
    def __init__(self, d):
        self.d, self.idx, self.fh = d, None, {}

    def _index(self):
        self.idx = {}
        for f in sorted(x for x in os.listdir(self.d) if x.endswith(".safetensors")):
            with open(os.path.join(self.d, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k in hdr:
                if k != "__metadata__":
                    self.idx[k] = f

    def get(self, name):
        """CPU tensor or None (missing name / no checkpoint)."""
        try:
            if self.idx is None:
                self._index()
            f = self.idx.get(name)
            if f is None:
                return None
            from safetensors import safe_open
            if f not in self.fh:
                self.fh[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
            return self.fh[f].get_tensor(name)
        except Exception as e:  # noqa: BLE001
            OUT["meta"].setdefault("ckpt_errors", []).append(f"{name}: {e!r}"[:300])
            return None


def gpu(t):
    return None if t is None else t.cuda()


def fp8w(ck, name):
    w, s = ck.get(name + ".weight"), ck.get(name + ".scale")
    if w is None or s is None:
        return None
    return w.cuda().view(E4), s.cuda().view(U8).view(E8)


# ------------------------------------------------------------------------------------------------ 1. load
def elf_needed_py(path):
    """DT_NEEDED / DT_RUNPATH / DT_RPATH of an ELF64 LE shared object, from its section headers."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian file")
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        secs.append(struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize))
    needed, paths = [], []
    for sh in secs:
        if sh[1] != 6:  # SHT_DYNAMIC
            continue
        stroff = secs[sh[6]][4]
        off, size = sh[4], sh[5]
        for j in range(0, size, 16):
            tag, val = struct.unpack_from("<qQ", data, off + j)
            if tag == 0:
                break
            if tag in (1, 15, 29):
                end = data.index(b"\x00", stroff + val)
                s = data[stroff + val:end].decode(errors="replace")
                (needed if tag == 1 else paths).append(s)
    return needed, paths


def sec_load(args):
    L = OUT["load"]
    before = vllm_mods()
    gate("load.no_vllm_before", not before, vllm_modules=before)
    cands = []
    for root in [SITE] + [p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]:
        cands += glob.glob(os.path.join(root, "vllm", "_flashmla_C*.so"))
    cands = sorted(set(cands))
    L["so_candidates"] = cands
    if not gate("load.so_found", bool(cands), n=len(cands), paths=cands):
        return
    so = cands[0]
    L["so"] = so
    L["so_bytes"] = os.path.getsize(so)
    # NEEDED
    try:
        if shutil.which("readelf"):
            txt = subprocess.run(["readelf", "-d", so], capture_output=True, text=True, timeout=60).stdout
            needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", txt)
            paths = re.findall(r"\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]+)\]", txt)
            L["needed_method"] = "readelf -d"
        else:
            needed, paths = elf_needed_py(so)
            L["needed_method"] = "builtin ELF parser"
        L["needed"], L["runpath"] = needed, paths
        ok_pat = re.compile(r"^(libtorch\w*|libc10\w*|libcudart|libcuda|libc|libm|libdl|libpthread|librt|libstdc\+\+|"
                            r"libgcc_s|ld-linux[\w-]*)\.so")
        foreign = [n for n in needed if not ok_pat.match(n)]
        gate("load.needed_only_torch_cuda_libc", not foreign, needed=needed, foreign=foreign)
    except Exception as e:  # noqa: BLE001
        L["needed_err"] = repr(e)
        print("FAIL(err) load.needed", repr(e), flush=True)
    # load
    try:
        torch.ops.load_library(so)
        L["load_library"] = "ok"
    except Exception as e:  # noqa: BLE001
        L["load_library"] = repr(e)[:1500]
        gate("load.load_library", False, err=repr(e)[:300])
        return
    after = vllm_mods()
    gate("load.no_vllm_after_load", not after, vllm_modules=after)
    ns = torch.ops._flashmla_C
    ST["OPS"] = ns
    try:
        names = sorted(n.split("::", 1)[1] for n in torch._C._dispatch_get_all_op_names()
                       if n.startswith("_flashmla_C::"))
    except Exception as e:  # noqa: BLE001
        names = ["<listing failed: %r>" % (e,)]
    L["registered_ops"] = names
    schemas = {}
    for n in ("sparse_decode_fwd", "sparse_prefill_fwd", "fused_norm_rope_attn_rope_cast_decode",
              "fused_norm_rope_attn_rope_cast_fwd", "dense_decode_fwd", "permute_q_b_proj"):
        try:
            op = getattr(ns, n)
            schemas[n] = [str(getattr(op, ov)._schema) for ov in op.overloads()]
        except Exception as e:  # noqa: BLE001
            schemas[n] = "absent: %r" % (e,)
    L["schemas"] = schemas
    for n in ("sparse_decode_fwd", "sparse_prefill_fwd", "fused_norm_rope_attn_rope_cast_decode"):
        gate("load.op." + n, not str(schemas.get(n, "absent")).startswith("absent"), schema=schemas.get(n))
    # FlashInfer, read-only
    fi = {}
    try:
        spec = importlib.util.find_spec("flashinfer")
        fi["found"] = spec is not None
        if spec is not None:
            root = list(spec.submodule_search_locations)[0]
            fi["root"] = root
            hits = []
            for dp, _, files in os.walk(root):
                for f in files:
                    if not f.endswith(".py"):
                        continue
                    p = os.path.join(dp, f)
                    try:
                        text = open(p, errors="replace").read()
                    except OSError:
                        continue
                    for m in re.finditer(r"^\s*def\s+(\w*(?:sparse_mla|dsv4|ds_v4|deepseek_v4)\w*)\s*\(", text,
                                         re.M | re.I):
                        hits.append({"file": os.path.relpath(p, root), "def": m.group(1),
                                     "line": text.count("\n", 0, m.start()) + 1})
            fi["dsv4_defs"] = hits
            fi["cubins_in_pkg"] = len(glob.glob(os.path.join(root, "**", "*.cubin"), recursive=True))
        for pk in ("flashinfer_cubin", "flashinfer_jit_cache"):
            s = importlib.util.find_spec(pk)
            if s is None:
                fi[pk] = None
            else:
                r = list(s.submodule_search_locations)[0]
                fi[pk] = {"root": r,
                          "cubins": len(glob.glob(os.path.join(r, "**", "*.cubin"), recursive=True)),
                          "sos": len(glob.glob(os.path.join(r, "**", "*.so"), recursive=True))}
        cache = os.path.expanduser("~/.cache/flashinfer")
        fi["user_cache_exists"] = os.path.isdir(cache)
        fi["cubins_bundled"] = bool(fi.get("cubins_in_pkg")) or bool((fi.get("flashinfer_cubin") or {}).get("cubins"))
    except Exception as e:  # noqa: BLE001
        fi["err"] = repr(e)[:600]
    L["flashinfer"] = fi
    info("load.flashinfer", found=fi.get("found"), dsv4_defs=[h["def"] for h in fi.get("dsv4_defs", [])][:12],
         cubins_bundled=fi.get("cubins_bundled"))
    gate("load.no_vllm_end_of_load", not vllm_mods(), vllm_modules=vllm_mods())


# ------------------------------------------------------------------------------------------------ data (shared by 2-5)
def build_data(args):
    K = ST["K"]
    ck = Ckpt(args.model_dir) if os.path.isdir(args.model_dir) else None
    ST["CK"] = ck
    src = {}
    DATA = {}
    T = max(POS_CORRECT + POS_SPEED) + max(BS) + 1
    for t, Lyr in BASE_LAYER.items():
        d = {}
        pre = "layers.%d.attn." % Lyr
        freqs = precompute_freqs_cis(RD, 8192, 65536 if t else 0, 160000 if t else 10000, 16, 32, 1)
        d["freqs"] = freqs
        sink = gpu(ck.get(pre + "attn_sink")) if ck else None
        d["sink"] = (sink.float().contiguous() if sink is not None
                     else torch.linspace(-2.0, 2.0, H, device="cuda", dtype=torch.float32))
        src["L%d.attn_sink" % Lyr] = "ckpt" if sink is not None else "synthetic"
        # window rows for positions 0..T-1: kv_norm(wkv(x)) + RoPE (model.py:706-707), pre-QDQ
        wkv = fp8w(ck, pre + "wkv") if ck else None
        kvn = gpu(ck.get(pre + "kv_norm.weight")) if ck else None
        if wkv is not None and kvn is not None:
            kv = rms(lin_fp8(hidden(T, 100 + t), *wkv), kvn)
            src["L%d.window" % Lyr] = "ckpt wkv/kv_norm on synthetic hidden"
        else:
            kv = blocky(T, D, 100 + t)
            src["L%d.window" % Lyr] = "synthetic blocky"
        rope_(kv[:, -RD:], freqs[:T])
        d["win_pre"] = kv.contiguous()
        # q per position (model.py:771-773)
        d["wq_a"] = fp8w(ck, pre + "wq_a") if ck else None
        d["q_norm"] = gpu(ck.get(pre + "q_norm.weight")) if ck else None
        d["wq_b"] = fp8w(ck, pre + "wq_b") if ck else None
        src["L%d.q" % Lyr] = ("ckpt wq_a/q_norm/wq_b on synthetic hidden"
                             if d["wq_a"] is not None and d["q_norm"] is not None and d["wq_b"] is not None
                             else "synthetic")
        if t:
            r, Wr = t, WR[t]
            cw = gpu(ck.get(pre + "compressor.wkv.weight")) if ck else None
            cn = gpu(ck.get(pre + "compressor.norm.weight")) if ck else None
            cg = gpu(ck.get(pre + "compressor.wgate.weight")) if (ck and r > 1) else None
            if cw is not None and cn is not None and (r == 1 or cg is not None):
                if r == 1:   # model.py:461-462: bf16 projection + norm
                    lat = rms(F.linear(hidden(Wr, 200 + t), cw.to(BF)), cn)
                else:        # model.py:464-485 at decode: fp32 gated pool over 2 slots, then norm
                    x = hidden(2 * Wr, 200 + t).float()
                    kvf = F.linear(x, cw.float())
                    sc = F.linear(x, cg.float())
                    lat = rms((kvf.view(Wr, 2, D) * sc.view(Wr, 2, D).softmax(dim=1)).sum(dim=1).to(BF), cn)
                    del x, kvf, sc
                src["L%d.latent" % Lyr] = "ckpt compressor on synthetic hidden"
            else:
                lat = blocky(Wr, D, 300 + t, block=16)
                src["L%d.latent" % Lyr] = "synthetic blocky"
            rope_(lat[:, -RD:], freqs[torch.arange(Wr, device="cuda") * r])   # group j at position j*r (757)
            d["lat_pre"] = lat.contiguous()
        DATA[t] = d
    # reference QDQ (in place = what the model attends) + codes/scales (not in place) for the records
    for t, d in DATA.items():
        d["win_ref"] = K.act_quant(d["win_pre"].clone(), 32, "ue8m0", E8, True)
        c, s = K.act_quant(d["win_pre"].clone(), 32, "ue8m0", E8)
        d["win_codes"], d["win_sc"] = c.view(U8), s.view(U8)
        if t:
            d["lat_ref"] = K.fp4_act_quant(d["lat_pre"].clone(), 16, True, scale_dtype=E4)
            c, s = K.fp4_act_quant(d["lat_pre"].clone(), 16, False, scale_dtype=E4)
            d["lat_codes"], d["lat_sc"] = c.view(U8), s.view(U8)
            d["lat_pages"] = pack_pages(d["lat_codes"], d["lat_sc"], args.page)
    sync()
    OUT["meta"]["data_source"] = src
    ST["DATA"] = DATA


# ------------------------------------------------------------------------------------------------ records (528 / 288)
def deq_mxfp8(codes, sc):
    """torch MXFP8 dequant: e4m3 * 2^(e8m0 - 127) per 32 -> bf16 (cache_utils.py:486-499)."""
    n = codes.size(0)
    v = codes.view(E4).float().view(n, 16, 32)
    s = torch.exp2(sc.float() - 127.0).view(n, 16, 1)
    return (v * s).view(n, D).to(BF)


def deq_nvfp4(packed, sc, low_even=True):
    """torch NVFP4 dequant: e2m1 (even index in the low nibble) * e4m3 per 16 -> bf16 (cache_utils.py:543-566)."""
    n = packed.size(0)
    lo, hi = (packed & 0xF).to(torch.int32), (packed >> 4).to(torch.int32)
    codes = torch.stack([lo, hi] if low_even else [hi, lo], -1).view(n, D)
    mc = codes & 7
    e = (mc >> 1).float()
    m = (mc & 1).float()
    mag = torch.where(mc < 2, m * 0.5, (1.0 + m * 0.5) * torch.exp2(e - 1.0))
    vals = torch.where(codes >= 8, -mag, mag)
    s = sc.view(E4).float().view(n, 32, 1)
    return (vals.view(n, 32, 16) * s).view(n, D).to(BF)


def pack_pages(data_u8, scale_u8, P):
    """vLLM's V4.1 page: P data rows then P scale rows (cache_utils.py:481-484). -> uint8 [nb, P, 1, row_bytes]."""
    T, bd = data_u8.shape
    bs = scale_u8.size(1)
    nb = -(-T // P)
    if nb * P != T:
        data_u8 = torch.cat([data_u8, data_u8.new_zeros(nb * P - T, bd)])
        scale_u8 = torch.cat([scale_u8, scale_u8.new_full((nb * P - T, bs), 127)])
    page = torch.empty(nb, P * (bd + bs), dtype=U8, device=data_u8.device)
    page[:, :P * bd] = data_u8.reshape(nb, P * bd)
    page[:, P * bd:] = scale_u8.reshape(nb, P * bs)
    return page.view(nb, P, 1, bd + bs)


def read_pages(pages, T, bd, bs, P):
    """vLLM's addressing, independent of pack_pages: data at page + off*bd, scales at page + P*bd + off*bs."""
    flat = pages.reshape(pages.size(0), -1)
    pos = torch.arange(T, device=pages.device)
    blk, off = (pos // P).unsqueeze(1), (pos % P).unsqueeze(1)
    data = flat[blk, off * bd + torch.arange(bd, device=pages.device)]
    sc = flat[blk, P * bd + off * bs + torch.arange(bs, device=pages.device)]
    return data, sc


def sec_values(args):
    V = OUT["values"]
    DATA = ST["DATA"]
    P = args.page
    for t, d in DATA.items():
        tag = "t%d" % t
        n = d["win_ref"].size(0)
        dq = deq_mxfp8(d["win_codes"], d["win_sc"])
        eq = bits_equal(dq, d["win_ref"])
        V[tag + ".window_dequant"] = dict(rows=n, eq=eq, **bf16_cmp(dq, d["win_ref"]),
                                          changed_by_qdq_pct=round(bf16_cmp(d["win_pre"], d["win_ref"])["mismatch_pct"], 3))
        gate("values.%s.window_dequant_eq_inplace" % tag, eq, rows=n, **bf16_cmp(dq, d["win_ref"]))
        pages = pack_pages(d["win_codes"], d["win_sc"], P)
        dd, ss = read_pages(pages, n, D, 16, P)
        dq2 = deq_mxfp8(dd, ss)
        eq2 = bits_equal(dq2, d["win_ref"])
        gate("values.%s.window_528_pages_eq_ref" % tag, eq2, rows=n, page=P, record_bytes=pages.size(-1),
             **bf16_cmp(dq2, d["win_ref"]))
        if t:
            n = d["lat_ref"].size(0)
            dq = deq_nvfp4(d["lat_codes"], d["lat_sc"], True)
            eq = bits_equal(dq, d["lat_ref"])
            alt = None
            if not eq:
                alt = bits_equal(deq_nvfp4(d["lat_codes"], d["lat_sc"], False), d["lat_ref"])
            V[tag + ".latent_dequant"] = dict(rows=n, eq=eq, high_nibble_even_eq=alt, **bf16_cmp(dq, d["lat_ref"]),
                                              zero_scales=int((d["lat_sc"] == 0).sum().item()))
            gate("values.%s.latent_dequant_eq_inplace" % tag, eq, rows=n, high_nibble_even_eq=alt,
                 **bf16_cmp(dq, d["lat_ref"]))
            pages = d["lat_pages"]
            dd, ss = read_pages(pages, n, D // 2, 32, P)
            dq2 = deq_nvfp4(dd, ss, True)
            eq2 = bits_equal(dq2, d["lat_ref"])
            gate("values.%s.latent_288_pages_eq_ref" % tag, eq2, rows=n, page=P, record_bytes=pages.size(-1),
                 **bf16_cmp(dq2, d["lat_ref"]))


# ------------------------------------------------------------------------------------------------ indices + case inputs
def win_idx(pos):
    """rt/static_step.py:97-102 (= model.py:422-424): slots oldest-first, -1 where slot > pos."""
    oldest = pos % WIN + 1
    return [s if s <= pos else -1 for s in ((oldest + j) % WIN for j in range(WIN))]


def comp_valid(pos, r):
    """Compressed positions row `pos` attends: c = (pos+1)//r; all if c <= 512 else 512 sorted picks (top-k stand-in)."""
    c = (pos + 1) // r
    if c <= TOPK:
        return list(range(c))
    rnd = random.Random(1000003 * pos + r)
    return sorted(rnd.sample(range(c), TOPK))


def make_q(t, pos_list):
    d = ST["DATA"][t]
    n = len(pos_list)
    seed = 7000 + 131 * t + pos_list[0]
    if d["wq_a"] is not None and d["q_norm"] is not None and d["wq_b"] is not None:
        qr = rms(lin_fp8(hidden(n, seed), *d["wq_a"]), d["q_norm"])
        q = lin_fp8(qr, *d["wq_b"]).view(n, H, D)
    else:
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        q = torch.randn(n, H, D, generator=g, device="cuda").to(BF)
    f = d["freqs"][torch.tensor(pos_list, device="cuda")].unsqueeze(1)
    rope_(q[..., -RD:], f)
    return q.contiguous()


def case_inputs(t, p, b, P):
    """Every arm's inputs for one (layer type, start position, rows) case. Rows i = 0..b-1 at p+i share one ring
    written up to p+b-1 (at p >= 127 rows < b-1 then also see later positions in their oldest slots: identical inputs
    for every arm, so arm comparisons are unaffected)."""
    d = ST["DATA"][t]
    pos = [p + i for i in range(b)]
    p_last = pos[-1]
    ring_pos = [p_last - ((p_last - s) % WIN) for s in range(WIN)]
    rp = torch.tensor([max(x, 0) for x in ring_pos], device="cuda")
    empty = torch.tensor([x < 0 for x in ring_pos], device="cuda")
    ring = d["win_ref"][rp].clone()
    ring[empty] = 0
    rcodes = d["win_codes"][rp].clone()
    rcodes[empty] = 0
    rsc = d["win_sc"][rp].clone()
    rsc[empty] = 127
    wl = [win_idx(x) for x in pos]
    q = make_q(t, pos)
    ci = {"t": t, "p": p, "b": b, "pos": pos, "q": q, "sink": d["sink"], "ring": ring, "rcodes": rcodes, "rsc": rsc}
    wvalid = [[s for s in w if s >= 0] for w in wl]
    ci["B_widx"] = torch.tensor([[v + [-1] * (WIN - len(v))] for v in wvalid], dtype=torch.int32, device="cuda")
    ci["B_wlen"] = torch.tensor([len(v) for v in wvalid], dtype=torch.int32, device="cuda")
    ci["B0_widx"] = torch.tensor([[w] for w in wl], dtype=torch.int32, device="cuda")
    ci["wpages"] = pack_pages(rcodes, rsc, P)
    if t == 0:
        ci["kvA"] = ring.unsqueeze(0).contiguous()
        ci["idxA"] = torch.tensor([wl], dtype=torch.int32, device="cuda")
        ci["idxA_exact"] = ci["idxA"]
        ci["width_exact"] = WIN
    else:
        r, Wr = t, WR[t]
        cv = [comp_valid(x, r) for x in pos]
        ci["kvA"] = torch.cat([ring, d["lat_ref"][:Wr]], 0).unsqueeze(0).contiguous()
        stat = [w + [v + WIN for v in c] + [-1] * (TOPK - len(c)) for w, c in zip(wl, cv)]
        we = max(len(c) for c in cv)
        exact = [w + [v + WIN for v in c] + [-1] * (we - len(c)) for w, c in zip(wl, cv)]
        ci["idxA"] = torch.tensor([stat], dtype=torch.int32, device="cuda")
        ci["idxA_exact"] = torch.tensor([exact], dtype=torch.int32, device="cuda")
        ci["width_exact"] = WIN + we
        ci["cpages"] = d["lat_pages"]
        ci["B_cidx"] = torch.tensor([[c + [-1] * (TOPK - len(c))] for c in cv], dtype=torch.int32, device="cuda")
        ci["B_clen"] = torch.tensor([len(c) for c in cv], dtype=torch.int32, device="cuda")
    ci["qA"] = q.unsqueeze(0)                                  # [1, b, 64, 512]
    ci["qB"] = q.unsqueeze(1)                                  # [b, 1, 64, 512] rows as batch (flashmla.py:200-203)
    ci["kvC"] = ci["kvA"][0].unsqueeze(1)                      # [n, 1, 512]
    ci["idxC"] = ci["idxA"][0].unsqueeze(1).contiguous()       # [b, 1, topk]
    return ci


# ------------------------------------------------------------------------------------------------ arms
class Sched:
    """FlashMLASchedMeta's two tensors (flash_mla_interface.py:124-194): None on the first call -> the in-kernel
    planner fills them; later same-type calls reuse them."""

    def __init__(self):
        self.tsm = None
        self.ns = None


def arm_A(ci, exact=False):
    K = ST["K"]
    return K.sparse_attn(ci["qA"], ci["kvA"], ci["sink"], ci["idxA_exact"] if exact else ci["idxA"], SCALE)[0]


def _cache_view(x):
    return x if ST.get("B_DTYPE", "uint8") == "uint8" else x.view(E4)


def arm_B(ci, sched=None, lens=True):
    ops = ST["OPS"]
    sched = sched or Sched()
    t = ci["t"]
    kc = _cache_view(ci["wpages"])
    idx = ci["B_widx"] if lens else ci["B0_widx"]
    tl_ = ci["B_wlen"] if lens else None
    ekc = _cache_view(ci["cpages"]) if t else None
    eidx = ci["B_cidx"] if t else None
    etl = ci["B_clen"] if (t and lens) else None
    out, lse, tsm, ns = ops.sparse_decode_fwd(ci["qB"], kc, idx, tl_, ci["sink"], sched.tsm, sched.ns,
                                              ekc, eidx, etl, D, SCALE, None)
    sched.tsm, sched.ns = tsm, ns
    return out[:, 0]


def arm_C(ci):
    ops = ST["OPS"]
    res = ops.sparse_prefill_fwd(ci["q"], ci["kvC"], ci["idxC"], SCALE, D, ci["sink"], None, None)
    return res[0]


def resolve_B_dtype(ci):
    """FlashMLA's cache dtype: try uint8 (vLLM's fp8_ds_mla storage), else a float8_e4m3fn view."""
    errs = {}
    for dt in ("uint8", "float8_e4m3fn"):
        ST["B_DTYPE"] = dt
        try:
            o = arm_B(ci)
            sync()
            OUT["outputs"]["B_cache_dtype"] = dt
            return o
        except Exception as e:  # noqa: BLE001
            errs[dt] = repr(e)[:600]
    OUT["outputs"]["B_cache_dtype_errors"] = errs
    ST["B_DTYPE"] = "uint8"
    raise RuntimeError("sparse_decode_fwd rejected both cache dtypes: %s" % errs)


def resolve_E(args):
    """Best-effort FlashInfer DSV4 decode: pick a source hit, import it, map its parameters by name. Returns
    (fn, sig) or None with OUT['outputs']['E'] explaining why."""
    fi = OUT["load"].get("flashinfer", {})
    hits = [h for h in fi.get("dsv4_defs", []) if "decode" in h["def"].lower()]
    E = OUT["outputs"].setdefault("E", {})
    if not hits:
        E["status"] = "not callable: no FlashInfer dsv4 decode def found"
        return None
    if not (fi.get("cubins_bundled") or args.arm_e):
        E["status"] = "not attempted: cubins not bundled (pass --arm-e to allow a first-use download)"
        E["candidates"] = hits
        return None
    import flashinfer  # noqa: F401  (standalone; checked below that it did not pull vllm)
    for h in hits:
        mod = "flashinfer." + h["file"][:-3].replace(os.sep, ".")
        if mod.endswith(".__init__"):
            mod = mod[:-9]
        try:
            m = importlib.import_module(mod)
            fn = getattr(m, h["def"])
            sig = inspect.signature(fn)
            E.setdefault("signatures", {})[mod + "." + h["def"]] = str(sig)
            if vllm_mods():
                E["status"] = "tainted: importing flashinfer loaded vllm"
                return None
            return fn, sig
        except Exception as e:  # noqa: BLE001
            E.setdefault("import_errors", {})[mod + "." + h["def"]] = repr(e)[:300]
    E["status"] = "not callable: imports failed"
    return None


def arm_E(ci, fn, sig, P):
    """bf16 paged caches [nb, P, 512], indices as storage offsets; parameters mapped by name (vLLM's call sites,
    flashinfer_sparse.py:546-561, 815-828)."""
    b, t = ci["b"], ci["t"]
    swa = ci["ring"].view(-1, P, D)
    comp = ST["DATA"][t]["lat_ref"][:WR[t]].view(-1, P, D) if t else None
    out = torch.empty(b, H, D, dtype=BF, device="cuda")
    ws = ST.setdefault("E_WS", torch.zeros(128 << 20, dtype=U8, device="cuda"))
    have = {"query": ci["q"], "q": ci["q"], "swa_kv_cache": swa, "kv_cache": swa, "workspace_buffer": ws,
            "sparse_indices": ci["B_widx"][:, 0], "compressed_kv_cache": comp, "out": out, "bmm1_scale": SCALE,
            "bmm2_scale": 1.0, "sinks": ci["sink"], "kv_layout": "NHD", "swa_topk_lens": ci["B_wlen"],
            "sparse_topk_lens": ci["B_wlen"], "seq_lens": torch.tensor(ci["pos"], dtype=torch.int32,
                                                                       device="cuda") + 1,
            "max_q_len": 1, "cum_seq_lens_q": torch.arange(b + 1, dtype=torch.int32, device="cuda"),
            "sparse_indices_are_storage_offsets": True,
            "extra_sparse_indices": ci["B_cidx"] if t else None, "extra_sparse_topk_lens": ci["B_clen"] if t else None}
    kw, missing = {}, []
    for name, prm in sig.parameters.items():
        if name in have:
            kw[name] = have[name]
        elif prm.default is inspect.Parameter.empty and prm.kind not in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            missing.append(name)
    if missing:
        raise RuntimeError("unmapped required parameters %s" % missing)
    r = fn(**kw)
    o = r[0] if isinstance(r, (tuple, list)) else (r if isinstance(r, torch.Tensor) else out)
    return o.view(b, H, D)


def oracle(ci, qv):
    """fp64: softmax over the gathered rows with the sink in the denominator (kernel.py:363-385; same math as
    FlashMLA's exp(lse)/(exp(lse)+exp(sink)))."""
    kv = ci["kvA"][0].double()
    idx = ci["idxA"][0]
    sink = ci["sink"].double()
    q = qv.double()
    out = torch.zeros(ci["b"], H, D, dtype=torch.float64, device="cuda")
    for i in range(ci["b"]):
        sel = idx[i][idx[i] >= 0].long()
        k = kv[sel]
        s = (q[i] @ k.T) * SCALE
        m = torch.maximum(s.max(-1).values, sink)
        e = torch.exp(s - m.unsqueeze(-1))
        out[i] = (e @ k) / (e.sum(-1) + torch.exp(sink - m)).unsqueeze(-1)
    return out


def sec_outputs(args):
    O = OUT["outputs"]
    K = ST["K"]
    P = args.page
    have_ops = ST.get("OPS") is not None
    efn = None
    try:
        efn = resolve_E(args)
    except Exception as e:  # noqa: BLE001
        O.setdefault("E", {})["status"] = "resolve failed: %r" % (e,)
    cases = O.setdefault("cases", {})
    first_B = True
    for t in (0, 2, 1):
        for p in POS_CORRECT:
            for b in BS:
                key = "t%d_p%d_b%d" % (t, p, b)
                c = cases.setdefault(key, {})
                try:
                    ci = case_inputs(t, p, b, P)
                except Exception as e:  # noqa: BLE001
                    c["err"] = "inputs: %r" % (e,)
                    print("FAIL(err) case", key, repr(e)[:400], flush=True)
                    continue
                c["width_static"] = int(ci["idxA"].size(-1))
                c["width_exact"] = int(ci["width_exact"])
                ref_b = oracle(ci, ci["q"])
                qm = K.act_quant(ci["q"].clone(), 32, "ue8m0", E8, True)
                ref_m = oracle(ci, qm)
                c["oracle_sep_rel"] = err64(ref_m.to(torch.float64), ref_b)["rel"]
                outs = {}
                outs["A"] = attempt(c, "A_run", lambda: arm_A(ci))
                outs["A_exact"] = attempt(c, "A_exact_run", lambda: arm_A(ci, exact=True))
                if have_ops:
                    if first_B:
                        outs["B"] = attempt(c, "B_run", lambda: resolve_B_dtype(ci))
                        first_B = False
                    else:
                        outs["B"] = attempt(c, "B_run", lambda: arm_B(ci))
                    outs["B0"] = attempt(c, "B0_run", lambda: arm_B(ci, lens=False))
                    outs["C"] = attempt(c, "C_run", lambda: arm_C(ci))
                if efn is not None:
                    outs["E"] = attempt(c, "E_run", lambda: arm_E(ci, efn[0], efn[1], P))
                sync()
                for arm, o in outs.items():
                    if o is None or not isinstance(o, torch.Tensor):
                        continue
                    o = o.view(b, H, D)
                    eb, em = err64(o, ref_b), err64(o, ref_m)
                    rec = {"vs_bf16q": eb, "vs_mxfp8q": em, "nearer": "bf16q" if eb["rel"] <= em["rel"] else "mxfp8q",
                           "finite": bool(torch.isfinite(o.float()).all().item())}
                    if outs.get("A") is not None and arm != "A":
                        rec["vs_A"] = bf16_cmp(o, outs["A"].view(b, H, D))
                    c[arm] = rec
                    c.pop(arm + "_run", None)
                summ = {a: (round(c[a]["vs_bf16q"]["rel"], 7), c[a]["nearer"], c[a].get("vs_A", {}).get("mismatch_pct"))
                        for a in outs if isinstance(c.get(a), dict) and "vs_bf16q" in c[a]}
                print("CASE", key, "w", c["width_static"], "/", c["width_exact"], "sep %.2e" % c["oracle_sep_rel"],
                      json.dumps(summ), flush=True)
                del ci
        dump()
    # gates over cases
    def coll(arm, f):
        return [f(v[arm]) for v in cases.values() if isinstance(v.get(arm), dict) and "vs_bf16q" in v[arm]]
    n_cases = len(cases)
    a_ex = [v["A_exact"]["vs_A"]["mismatch_pct"] == 0.0 for v in cases.values()
            if isinstance(v.get("A_exact"), dict) and "vs_A" in v["A_exact"]]
    gate("outputs.A_exact_eq_A_static_bitwise", bool(a_ex) and all(a_ex), cases=len(a_ex), of=n_cases,
         note="whole -1 blocks are exact no-ops (design 2, rec 1)")
    for arm in ("B", "B0", "C", "E"):
        got = [k for k, v in cases.items() if isinstance(v.get(arm), dict) and "vs_bf16q" in v[arm]]
        if not got:
            info("outputs.%s" % arm, ran=0, of=n_cases)
            continue
        ratios = []
        for k in got:
            a = cases[k].get("A", {}).get("vs_bf16q", {}).get("rel")
            if a:
                ratios.append(cases[k][arm]["vs_bf16q"]["rel"] / a)
        near = coll(arm, lambda r: r["nearer"] == "bf16q")
        fin = coll(arm, lambda r: r["finite"])
        mm = coll(arm, lambda r: r.get("vs_A", {}).get("mismatch_pct", float("nan")))
        ul = coll(arm, lambda r: r.get("vs_A", {}).get("max_ulp", -1))
        O["summary_" + arm] = {"ran": len(got), "of": n_cases, "max_err_ratio_vs_A": max(ratios) if ratios else None,
                               "nearer_bf16q": sum(near), "finite": sum(fin), "max_mismatch_pct_vs_A": max(mm),
                               "max_ulp_vs_A": max(ul)}
        gate("outputs.%s.ran_all_cases" % arm, len(got) == n_cases, ran=len(got), of=n_cases)
        gate("outputs.%s.finite" % arm, all(fin), finite=sum(fin), of=len(fin))
        gate("outputs.%s.err_le_2x_A" % arm, bool(ratios) and max(ratios) <= 2.0,
             max_ratio=round(max(ratios), 4) if ratios else None,
             mean_ratio=round(sum(ratios) / len(ratios), 4) if ratios else None)
        gate("outputs.%s.nearer_bf16q_oracle" % arm, all(near), nearer_bf16q=sum(near), of=len(near))
        info("outputs.%s.vs_A" % arm, max_mismatch_pct=max(mm), max_ulp=max(ul))
    a_rel = coll("A", lambda r: r["vs_bf16q"]["rel"])
    a_near = coll("A", lambda r: r["nearer"] == "bf16q")
    info("outputs.A.vs_oracles", max_rel_bf16q=max(a_rel) if a_rel else None, nearer_bf16q=sum(a_near),
         of=len(a_near))


# ------------------------------------------------------------------------------------------------ 4. speed
def time_graph(body, reps, n_calls):
    """Eager warm-up x2 on a side stream, capture one graph of the body, replay x2 (equality), then `reps` replays
    timed with CUDA events."""
    r = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
        eager = body()
    torch.cuda.current_stream().wait_stream(s)
    sync()
    eager = [o.clone() for o in eager]
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(5):
        body()
    ev1.record()
    sync()
    r["eager_us_per_call"] = round(ev0.elapsed_time(ev1) * 1000.0 / 5 / n_calls, 2)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = body()
    g.replay()
    sync()
    r1 = [o.clone() for o in outs]
    g.replay()
    sync()
    r["replay_equal"] = all(bits_equal(a, b) for a, b in zip(r1, outs))
    r["graph_eq_eager"] = all(bits_equal(a, b) for a, b in zip(r1, eager))
    for _ in range(3):
        g.replay()
    ev0.record()
    for _ in range(reps):
        g.replay()
    ev1.record()
    sync()
    ms = ev0.elapsed_time(ev1) / reps
    r["us_per_graph"] = round(ms * 1000.0, 2)
    r["us_per_call"] = round(ms * 1000.0 / n_calls, 3)
    r["calls"] = n_calls
    del g
    return r


def step_layers(b, p, P):
    """40 layers' inputs: per-layer window caches (bf16 ring and 528 B pages), one compressed cache per kv-source
    group (2, 8, 14 at ratio 2; 20 at ratio 1); A/C get the static step's per-layer cat (rt/static_step.py:512)."""
    per_type = {t: case_inputs(t, p, b, P) for t in (0, 2, 1)}
    compA, compB = {}, {}
    layers = []
    for l, t in enumerate(TYPES):
        ci = per_type[t]
        L = dict(ci)
        ring = ci["ring"].clone()
        L["wpages"] = ci["wpages"].clone()
        if t:
            g = group_of(l)
            if g not in compA:
                compA[g] = ST["DATA"][t]["lat_ref"][:WR[t]].clone()
                compB[g] = ci["cpages"].clone()
            L["kvA"] = torch.cat([ring, compA[g]], 0).unsqueeze(0).contiguous()
            L["cpages"] = compB[g]
        else:
            L["kvA"] = ring.unsqueeze(0).contiguous()
        L["kvC"] = L["kvA"][0].unsqueeze(1)
        layers.append(L)
    return layers


def sec_speed(args):
    S = OUT["speed"]
    K = ST["K"]
    P, reps = args.page, args.reps
    have_ops = ST.get("OPS") is not None
    for b in BS:
        for p in POS_SPEED:
            key = "b%d_p%d" % (b, p)
            R = S.setdefault(key, {})
            try:
                layers = step_layers(b, p, P)
            except Exception as e:  # noqa: BLE001
                R["err"] = "inputs: %r" % (e,)
                continue
            R["widths"] = {"static": [int(L["idxA"].size(-1)) for L in layers[:3]] + ["..."],
                           "exact": sorted({int(L["width_exact"]) for L in layers})}

            def bA(exact):
                return lambda: [K.sparse_attn(L["qA"], L["kvA"], L["sink"], L["idxA_exact"] if exact else L["idxA"],
                                              SCALE) for L in layers]

            def bB(lens):
                def body():
                    sch = {0: Sched(), 1: Sched(), 2: Sched()}   # one per layer type, inside the captured body
                    return [arm_B(L, sch[L["t"]], lens=lens) for L in layers]
                return body

            def bC():
                return [arm_C(L) for L in layers]

            arms = [("A_static", bA(False)), ("A_exact", bA(True))]
            if have_ops:
                arms += [("B", bB(True)), ("B0", bB(False)), ("C", bC)]
            for name, body in arms:
                r = attempt(R, name, lambda body=body: time_graph(body, reps, len(TYPES)))
                if r:
                    print("SPEED %s %-8s %8.2f us/call  %9.1f us/40  replay_eq=%s graph_eq_eager=%s eager %.2f us/call"
                          % (key, name, r["us_per_call"], r["us_per_graph"], r["replay_equal"], r["graph_eq_eager"],
                             r["eager_us_per_call"]), flush=True)
            del layers
            torch.cuda.empty_cache()
            dump()
    # A at uniform widths (all 40 calls the same width; ratio-1 data at p=1100 so every listed index is valid)
    U = S.setdefault("A_uniform", {})
    for b in BS:
        try:
            ci = case_inputs(1, 1100, b, P)
            kvs = [ci["kvA"].clone() for _ in range(len(TYPES))]
            full = ci["idxA"]
            for w in A_WIDTHS:
                idx = full[..., :w].contiguous()
                kk = "b%d_w%d" % (b, w)
                r = attempt(U, kk, lambda idx=idx: time_graph(
                    lambda: [K.sparse_attn(ci["qA"], kv, ci["sink"], idx, SCALE) for kv in kvs], reps, len(TYPES)))
                if r:
                    print("SPEED A_uniform %s %8.2f us/call (%d blocks of 64) replay_eq=%s"
                          % (kk, r["us_per_call"], -(-w // 64), r["replay_equal"]), flush=True)
            xs = [-(-w // 64) for w in A_WIDTHS]
            ys = [U.get("b%d_w%d" % (b, w), {}).get("us_per_call") for w in A_WIDTHS]
            if all(isinstance(y, (int, float)) for y in ys):
                n = len(xs)
                mx, my = sum(xs) / n, sum(ys) / n
                sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
                U["b%d_fit" % b] = {"us_per_block64": round(sl, 3), "us_intercept": round(my - sl * mx, 3)}
                info("speed.A_uniform.b%d.slope" % b, us_per_block64=round(sl, 3), intercept_us=round(my - sl * mx, 3))
            del kvs
        except Exception as e:  # noqa: BLE001
            U["b%d_err" % b] = repr(e)[:800]
        dump()
    # gates
    rep_ok, all_rep = True, []
    for k, R in S.items():
        if not isinstance(R, dict):
            continue
        for a, r in R.items():
            if isinstance(r, dict) and "replay_equal" in r:
                all_rep.append((k, a, r["replay_equal"], r["graph_eq_eager"]))
    bad = [x for x in all_rep if not x[2]]
    gate("speed.replay_equal_all_graphs", bool(all_rep) and not bad, graphs=len(all_rep), unequal=bad)
    info("speed.graph_eq_eager", unequal=[x[:2] for x in all_rep if not x[3]])
    for b in BS:
        for p in POS_SPEED:
            R = S.get("b%d_p%d" % (b, p), {})
            a = R.get("A_exact", {}).get("us_per_graph")
            a0 = R.get("A_static", {}).get("us_per_graph")
            bb = R.get("B", {}).get("us_per_graph")
            cc = R.get("C", {}).get("us_per_graph")
            info("speed.b%d_p%d" % (b, p), A_static_us=a0, A_exact_us=a, B_us=bb, B0_us=R.get("B0", {}).get(
                "us_per_graph"), C_us=cc, ms_per_step_A_static=None if a0 is None else round(a0 / 1000.0, 3))
            ok = isinstance(a, (int, float)) and isinstance(bb, (int, float)) and bb * 2.0 <= a
            gate("speed.B_ge_2x_A_exact.b%d_p%d" % (b, p), ok, A_exact_us=a, B_us=bb,
                 speedup=round(a / bb, 3) if (isinstance(a, (int, float)) and bb) else None)


# ------------------------------------------------------------------------------------------------ 5. probes
def probe_rows(args):
    """name -> (bf16 [N, w] rows, source label)."""
    N = args.probe_rows
    ck = ST.get("CK")
    DATA = ST.get("DATA", {})
    rows = {}
    x = hidden(N, 9001)
    rows[5120] = (x, "synthetic hidden (heavy-tailed channels)")
    d2 = DATA.get(2, {})
    wkv = fp8w(ck, "layers.2.attn.wkv") if ck else None
    rows[512] = ((lin_fp8(x, *wkv), "ckpt L2 wkv(x), pre kv_norm") if wkv is not None
                 else (blocky(N, 512, 9002), "synthetic blocky"))
    wqa = d2.get("wq_a")
    rows[1280] = ((lin_fp8(x, *wqa), "ckpt L2 wq_a(x), pre q_norm") if wqa is not None
                  else (blocky(N, 1280, 9003), "synthetic blocky"))
    wk = gpu(ck.get("layers.2.attn.indexer.wk.weight")) if ck else None
    if wk is not None:
        lat = rms(rows[512][0], torch.ones(512, device="cuda", dtype=BF)) if wkv is not None else blocky(N, 512, 9004)
        rows[128] = (F.linear(lat, wk.to(BF)), "ckpt L2 indexer wk(normed kv-like rows), pre k_norm")
    else:
        rows[128] = (blocky(N, 128, 9005), "synthetic blocky")
    rows[16384] = (hidden(N, 9006, 16384), "synthetic heavy-tailed rows (no 16384-wide norm input in the ckpt)")
    return rows


def emu_sum(sq, T, V, family, warp_inc):
    """Explicit fp32 summation orders (exact elementwise adds; a fixed-order Triton kernel computes the same bits).
    Thread t, slot v: family 'vec' takes elements ((k*T + t)*V + v) (vectorized loads), 'unroll' ((k*V + v)*T + t);
    each (t, v) sums its k sequentially; slots fold left; threads: shared-memory halving T/2..32, then a warp
    shfl_down tree at offsets 1..16 (warp_inc) or 16..1. Zero-pads to a multiple of T*V."""
    N, w = sq.shape
    tile = T * V
    Kn = -(-w // tile)
    if Kn * tile != w:
        sq = F.pad(sq, (0, Kn * tile - w))
    a = sq.view(N, Kn, T, V) if family == "vec" else sq.view(N, Kn, V, T).transpose(2, 3)
    acc = a[:, 0]
    for k in range(1, Kn):
        acc = acc + a[:, k]
    s = acc[..., 0]
    for v in range(1, V):
        s = s + acc[..., v]
    off = T // 2
    while off >= 32:
        s = s[:, :off] + s[:, off:2 * off]
        off //= 2
    n = s.size(1)
    for o in ([1, 2, 4, 8, 16] if warp_inc else [16, 8, 4, 2, 1]):
        if o < n:
            s = s + torch.roll(s, -o, dims=1)
    return s[:, 0]


def simple_orders(sq):
    N, w = sq.shape
    out = {}
    acc = sq[:, 0].clone()
    for j in range(1, w):
        acc = acc + sq[:, j]
    out["sequential"] = acc
    p2 = 1 << (w - 1).bit_length()
    s = F.pad(sq, (0, p2 - w))
    while s.size(1) > 1:
        h = s.size(1) // 2
        s = s[:, :h] + s[:, h:]
    out["pairwise_halves"] = s[:, 0]
    s = F.pad(sq, (0, p2 - w))
    while s.size(1) > 1:
        s = s[:, 0::2] + s[:, 1::2]
    out["pairwise_adjacent"] = s[:, 0]
    return out


def triton_kernels():
    if TRITON_ERR is not None:
        raise RuntimeError(TRITON_ERR)
    return triton, _sumsq_k, _rsqrt_k


# Module-level jit kernels (Triton resolves globals, not closure variables).
try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001
        from triton.language.extra.cuda import libdevice

    @triton.jit
    def _sumsq_k(X, Y, w, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        x = tl.load(X + row * stride + offs, mask=offs < w, other=0.0).to(tl.float32)
        tl.store(Y + row, tl.sum(x * x, axis=0))

    @triton.jit
    def _rsqrt_k(X, Y, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(X + offs, mask=m, other=1.0)
        if MODE == 0:
            y = libdevice.rsqrt(x)
        elif MODE == 1:
            y = libdevice.rsqrt_rn(x)
        elif MODE == 2:
            y = tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32,
                                          is_pure=True, pack=1)
        elif MODE == 3:
            y = tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32,
                                          is_pure=True, pack=1)
        elif MODE == 4:
            y = libdevice.div_rn(1.0, libdevice.sqrt_rn(x))
        else:
            y = tl.rsqrt(x)
        tl.store(Y + offs, y, mask=m)

    TRITON_ERR = None
except Exception as _e:  # noqa: BLE001
    triton = None
    TRITON_ERR = repr(_e)[:600]


def bits32_eq_frac(a, b):
    return (a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).float().mean().item()


def sec_probes(args):
    Pr = OUT["probes"]
    eps = 1e-20
    rows = probe_rows(args)
    Pr["row_sources"] = {w: s for w, (_, s) in rows.items()}
    try:
        triton, sumsq_k, rsqrt_k = triton_kernels()
    except Exception as e:  # noqa: BLE001
        triton = None
        Pr["triton_err"] = repr(e)[:600]
    m1rows = min(args.probe_rows, args.m1_rows)
    all_var = []
    for w in (128, 512, 1280, 5120, 16384):
        x = rows[w][0]
        N = x.size(0)
        R = Pr.setdefault("w%d" % w, {"rows": N, "m1_rows": m1rows, "source": rows[w][1]})
        try:
            t0 = time.time()
            vN = x.float().square().mean(-1)
            v1 = torch.empty(m1rows, dtype=torch.float32, device="cuda")
            for i in range(m1rows):
                v1[i:i + 1] = x[i:i + 1].float().square().mean(-1)
            v6 = torch.empty(N - N % 6, dtype=torch.float32, device="cuda")
            for i in range(0, N - N % 6, 6):
                v6[i:i + 6] = x[i:i + 6].float().square().mean(-1)
            sync()
            n6 = min(m1rows, v6.numel())
            R["a_var_M1_vs_M6_mismatch_rows"] = int((v1[:n6].view(torch.int32) != v6[:n6].view(torch.int32)).sum())
            R["a_var_M1_vs_MN_mismatch_rows"] = int((v1.view(torch.int32) != vN[:m1rows].view(torch.int32)).sum())
            R["a_var_M6_vs_MN_mismatch_rows"] = int((v6.view(torch.int32) != vN[:v6.numel()].view(torch.int32)).sum())
            r1, r6, rN = torch.rsqrt(v1 + eps), torch.rsqrt(v6 + eps), torch.rsqrt(vN + eps)
            R["a_rsqrt_M1_vs_M6_mismatch_rows"] = int((r1[:n6].view(torch.int32) != r6[:n6].view(torch.int32)).sum())
            R["a_rsqrt_M1_vs_MN_mismatch_rows"] = int((r1.view(torch.int32) != rN[:m1rows].view(torch.int32)).sum())
            R["a_s"] = round(time.time() - t0, 1)
            gate("probes.a.w%d.torch_var_M_invariant_1_vs_6" % w, R["a_var_M1_vs_M6_mismatch_rows"] == 0,
                 mismatch_rows=R["a_var_M1_vs_M6_mismatch_rows"], of=n6,
                 M1_vs_MN=R["a_var_M1_vs_MN_mismatch_rows"], M6_vs_MN=R["a_var_M6_vs_MN_mismatch_rows"])
            all_var.append(v1 + eps)
        except Exception as e:  # noqa: BLE001
            R["a_err"] = repr(e)[:800]
            print("FAIL(err) probes.a.w%d" % w, repr(e)[:300], flush=True)
            continue
        # (b) explicit orders vs torch at M=1
        try:
            sq = x[:m1rows].float().square()
            fac = torch.tensor(1.0, dtype=torch.float32, device="cuda") / torch.tensor(float(w), dtype=torch.float32,
                                                                                       device="cuda")
            cands = {}
            for nm, s in simple_orders(sq).items():
                cands[nm] = s
            for fam in ("vec", "unroll"):
                for T in (32, 64, 128, 256, 512):
                    for V in (1, 2, 4, 8):
                        if T * V > 4 * w:
                            continue
                        for inc in (True, False):
                            cands["%s_T%d_V%d_%s" % (fam, T, V, "inc" if inc else "dec")] = emu_sum(sq, T, V, fam, inc)
            if triton is not None:
                try:
                    y = torch.empty(m1rows, dtype=torch.float32, device="cuda")
                    xc = x[:m1rows].contiguous()
                    sumsq_k[(m1rows,)](xc, y, w, xc.stride(0), BLOCK=triton.next_power_of_2(w))
                    cands["triton_tl_sum"] = y
                except Exception as e:  # noqa: BLE001
                    R["b_triton_err"] = repr(e)[:400]
            res = {}
            for nm, s in cands.items():
                res[nm + "|*RN(1/w)"] = bits32_eq_frac(s * fac, v1)
                res[nm + "|/w"] = bits32_eq_frac(s / float(w), v1)
            best = sorted(res.items(), key=lambda kv: -kv[1])[:8]
            R["b_best"] = best
            R["b_n_candidates"] = len(res)
            R["b_full_matches"] = [k for k, v in res.items() if v == 1.0]
            gate("probes.b.w%d.explicit_order_matches_torch_M1" % w, bool(R["b_full_matches"]),
                 n_full=len(R["b_full_matches"]), best=[(k, round(v, 6)) for k, v in best[:3]],
                 candidates=len(res))
            del sq, cands
        except Exception as e:  # noqa: BLE001
            R["b_err"] = repr(e)[:800]
            print("FAIL(err) probes.b.w%d" % w, repr(e)[:300], flush=True)
        dump()
    # (c) rsqrt
    try:
        C = Pr.setdefault("c_rsqrt", {})
        g = torch.Generator(device="cuda")
        g.manual_seed(77)
        sweep = torch.exp2(torch.rand(1 << 20, generator=g, device="cuda", dtype=torch.float32) * 200.0 - 100.0)
        for nm, v in (("var_eps_rows", torch.cat(all_var) if all_var else None), ("logsweep_2^-100..2^100", sweep)):
            if v is None:
                continue
            v = v.contiguous()
            ref = torch.rsqrt(v)
            crn = (1.0 / torch.sqrt(v.double())).float()
            Cn = C.setdefault(nm, {"n": v.numel(), "torch_vs_fp64_rn": bits32_eq_frac(ref, crn)})
            if triton is None:
                continue
            for mode, label in ((0, "libdevice.rsqrt"), (1, "libdevice.rsqrt_rn"), (2, "ptx rsqrt.approx.f32"),
                                (3, "ptx rsqrt.approx.ftz.f32"), (4, "div_rn(1,sqrt_rn)"), (5, "tl.rsqrt")):
                try:
                    y = torch.empty_like(v)
                    rsqrt_k[(triton.cdiv(v.numel(), 1024),)](v, y, v.numel(), MODE=mode, BLOCK=1024)
                    sync()
                    Cn[label] = {"match_torch": bits32_eq_frac(y, ref), "match_fp64_rn": bits32_eq_frac(y, crn)}
                except Exception as e:  # noqa: BLE001
                    Cn[label] = {"err": repr(e)[:300]}
            full = [k for k, r in Cn.items() if isinstance(r, dict) and r.get("match_torch") == 1.0]
            gate("probes.c.%s.rsqrt_variant_matches_torch" % nm, bool(full), full_matches=full,
                 torch_vs_fp64_rn=round(Cn["torch_vs_fp64_rn"], 6),
                 rates={k: round(r["match_torch"], 6) for k, r in Cn.items() if isinstance(r, dict) and "match_torch" in r})
    except Exception as e:  # noqa: BLE001
        Pr["c_err"] = repr(e)[:800]
    del rows
    torch.cuda.empty_cache()
    dump()
    # ratio-2 compressor softmax(dim=1) over 2 slots (model.py:482), M=1 vs 6
    try:
        Sx = Pr.setdefault("softmax2", {})
        n = args.m1_rows - args.m1_rows % 6
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        f32 = torch.float32
        kv = torch.randn(n, 2, D, generator=g, device="cuda", dtype=f32) * 2.0     # kv_state/score_state are fp32
        sc = torch.randn(n, 2, D, generator=g, device="cuda", dtype=f32) * 3.0
        sm1, po1 = torch.empty_like(sc), torch.empty(n, 1, D, device="cuda", dtype=f32)
        for i in range(n):
            s = sc[i:i + 1].softmax(dim=1)
            sm1[i:i + 1] = s
            po1[i:i + 1] = (kv[i:i + 1] * s).sum(dim=1, keepdim=True)
        sm6, po6 = torch.empty_like(sc), torch.empty(n, 1, D, device="cuda", dtype=f32)
        for i in range(0, n, 6):
            s = sc[i:i + 6].softmax(dim=1)
            sm6[i:i + 6] = s
            po6[i:i + 6] = (kv[i:i + 6] * s).sum(dim=1, keepdim=True)
        Sx["rows"] = n
        Sx["softmax_mismatch_rows"] = int((sm1.view(torch.int32) != sm6.view(torch.int32)).flatten(1).any(1).sum())
        Sx["pooled_mismatch_rows"] = int((po1.view(torch.int32) != po6.view(torch.int32)).flatten(1).any(1).sum())
        gate("probes.softmax2.M_invariant_1_vs_6", Sx["softmax_mismatch_rows"] == 0 and Sx["pooled_mismatch_rows"] == 0,
             **Sx)
    except Exception as e:  # noqa: BLE001
        Pr["softmax2_err"] = repr(e)[:800]
    # indexer einsum / weights_proj / head-sum (model.py:556-558), M=1 vs 6
    try:
        K = ST["K"]
        ck = ST.get("CK")
        Ix = Pr.setdefault("indexer", {})
        wp = gpu(ck.get("layers.2.attn.indexer.weights_proj.weight")) if ck else None
        Ix["weights_proj_source"] = "ckpt L2" if wp is not None else "synthetic"
        if wp is None:
            wp = (torch.randn(32, 5120, device="cuda") * 0.02).to(BF)
        wp = wp.to(BF)
        g = torch.Generator(device="cuda")
        for W in (2048, 4096):
            g.manual_seed(W)
            kk = K.fp4_act_quant(torch.randn(1, W, 128, generator=g, device="cuda").to(BF), 32, True)
            bad = {"einsum": 0, "weights_proj": 0, "head_sum": 0}
            trials = args.indexer_trials
            for tr in range(trials):
                q6 = K.fp4_act_quant(torch.randn(1, 6, 32, 128, generator=g, device="cuda").to(BF), 32, True)
                x6 = hidden(6, 40000 + tr).view(1, 6, 5120)

                def path(q, x):
                    w = F.linear(x, wp) * (128 ** -0.5 * 32 ** -0.5)
                    e = torch.einsum("bshd,btd->bsht", q, kk)
                    return e, w, (e.clone().relu_() * w.unsqueeze(-1)).sum(dim=2)
                e6, w6, s6 = path(q6, x6)
                for i in range(6):
                    e1, w1, s1 = path(q6[:, i:i + 1].contiguous(), x6[:, i:i + 1].contiguous())
                    bad["einsum"] += int(not bits_equal(e1[0, 0], e6[0, i]))
                    bad["weights_proj"] += int(not bits_equal(w1[0, 0], w6[0, i]))
                    bad["head_sum"] += int(not bits_equal(s1[0, 0], s6[0, i]))
            Ix["W%d" % W] = dict(rows=6 * trials, **bad)
            gate("probes.indexer.W%d.M_invariant_1_vs_6" % W, not any(bad.values()), rows=6 * trials, **bad)
    except Exception as e:  # noqa: BLE001
        Pr["indexer_err"] = repr(e)[:800]


# ------------------------------------------------------------------------------------------------ decision
def decide():
    g = OUT["gates"]

    def p(name):
        return bool(g.get(name, {}).get("pass_"))
    loads = (p("load.so_found") and p("load.op.sparse_decode_fwd") and p("load.no_vllm_after_load")
             and p("load.no_vllm_end_of_load"))
    err_ok = p("outputs.B.err_le_2x_A")
    near = p("outputs.B.nearer_bf16q_oracle")
    sumB = OUT["outputs"].get("summary_B", {})
    q_quant = bool(sumB) and sumB.get("nearer_bf16q", 0) * 2 < sumB.get("ran", 0)
    speed = all(p("speed.B_ge_2x_A_exact.b%d_p%d" % (b, pp)) for b in BS for pp in POS_SPEED)
    replay = p("speed.replay_equal_all_graphs")
    vals = [v.get("pass_") for k, v in g.items() if k.startswith("values.")]
    values = bool(vals) and all(vals)
    reasons = []
    if not loads:
        reasons.append("does not load without vllm (or op missing)")
    if not values:
        reasons.append("528/288 records do not hold the reference values bitwise")
    if q_quant:
        reasons.append("B is nearer the MXFP8-q oracle: q quantized -> A1-class, James decides")
    elif not near:
        reasons.append("B not nearer the bf16-q oracle in every case")
    if not err_ok:
        reasons.append("B fp64 error > 2x A's")
    if not speed:
        reasons.append("B not >= 2x faster than A-exact at every b/position")
    if not replay:
        reasons.append("some graph replay unequal")
    verdict = "ADOPT" if not reasons else "PARK"
    OUT["decision"] = {"verdict": verdict, "loads_without_vllm": loads, "values_exact": values,
                       "fp64_err_le_2x_A": err_ok, "nearer_bf16q": near, "q_quantized": q_quant,
                       "ge_2x_faster": speed, "replay_equal": replay, "reasons": reasons,
                       "note": "A-exact (exact index width) stands in for A-with-skip (T9b-3a); ADOPT means "
                               "'--attn-impl flashmla' as A3-class, ref stays default"}
    gate("decision.adopt_B", verdict == "ADOPT", reasons=reasons)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default="/ref")
    ap.add_argument("--json", default="/work/scratch/flashmla-probe.json")
    ap.add_argument("--sections", default="load,values,outputs,speed,probes")
    ap.add_argument("--page", type=int, default=64, help="tokens per FlashMLA page block (vLLM SM100: 64 at r=2)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--probe-rows", type=int, default=100000)
    ap.add_argument("--m1-rows", type=int, default=100000, help="rows run one at a time (M=1) in the probes")
    ap.add_argument("--indexer-trials", type=int, default=50)
    ap.add_argument("--arm-e", action="store_true")
    args = ap.parse_args()
    JSON_PATH[0] = args.json
    secs = set(s.strip() for s in args.sections.split(","))
    OUT["meta"].update({"argv": sys.argv, "t_start": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "page": args.page, "reps": args.reps, "positions_correct": POS_CORRECT,
                        "positions_speed": POS_SPEED, "rows": BS, "layer_types": TYPES})
    try:
        OUT["meta"]["device"] = torch.cuda.get_device_name(0)
        OUT["meta"]["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:  # noqa: BLE001
        OUT["meta"]["device_err"] = repr(e)
    torch.set_default_dtype(BF)
    torch.set_default_device("cuda")
    torch.set_grad_enabled(False)
    dump()
    if "load" in secs:
        section("load", lambda: sec_load(args))
    else:
        gate("load.no_vllm_before", not vllm_mods(), vllm_modules=vllm_mods())

    def k_and_data():
        if REF_INFERENCE not in sys.path:
            sys.path.insert(0, REF_INFERENCE)
        import kernel as K  # noqa: E402  (the checkpoint's own reference kernels)
        ST["K"] = K
        build_data(args)
        gate("data.no_vllm_after_reference_import", not vllm_mods(), vllm_modules=vllm_mods())
    section("data", k_and_data)
    if "values" in secs:
        section("values", lambda: sec_values(args))
    if "outputs" in secs:
        section("outputs", lambda: sec_outputs(args))
    if "speed" in secs:
        section("speed", lambda: sec_speed(args))
    if "probes" in secs:
        section("probes", lambda: sec_probes(args))
    section("decision", decide)
    OUT["meta"]["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    OUT["meta"]["vllm_modules_at_exit"] = vllm_mods()
    dump()
    n_pass = sum(1 for v in OUT["gates"].values() if v.get("pass_"))
    print("T9B3-PROBE: %d/%d gates PASS; decision %s %s; errors in %s; receipt %s"
          % (n_pass, len(OUT["gates"]), OUT["decision"].get("verdict"), OUT["decision"].get("reasons"),
             sorted(OUT["errors"]), JSON_PATH[0]), flush=True)


if __name__ == "__main__":
    main()
