"""ds41 fastpath component `gracetier`: the cold experts live in Grace, every MoE layer computes all 384 in one MegaMoE.

Card GT1 / docs/P1-GRACETIER-DESIGN.md sections 2.2, 3, 4 and 5 "Build 2"; mini-test receipts
docs/gracetier-minitest-L10-gt-mt-1108*.json (M1 VMM aliasing torch.equal; M4 stage-into-slots == all-HBM G = 384
MegaMoE bit-identical at T = 1/6/16/64; stage 46-50 us per cold expert). Replaces `coldskip` + `peer3` + the RTX
sidecar on a stock DGX Station (one GB300, no RTX PRO 6000). Arm it with "gracetier" in DS41_FASTPATH in place of
"coldskip,peer3"; it needs the boundary patch (patches/0001-boundary-gracetier.patch) for T <= 16 and the hook
(mega_peer_hook) installed first, like peer3.

What it does, per target MoE layer l (40 layers; H_l hot, C_l = 384 - H_l cold; H = 255 with the h255 rowmap):
  (i)   geometry  MegaMoE runs at G = 384 groups. Groups 0..H-1 hold the hot experts (private HBM, the bytes vLLM's own
                  finalize produced), groups H..383 hold cold expert c = H + cold-local id. The cold groups of EVERY
                  layer are backed by ONE shared HBM scratch per weight tensor (CUDA VMM: one physical allocation mapped
                  behind each layer's hot block; the granule straddling the hot/cold boundary is private per layer;
                  runtime/gracetier_vmm.py). Sharing is legal because layers run in stream order and a layer reads only
                  the cold groups its own call staged. The e8m0 scale banks are resident in HBM for all 384 groups.
  (ii)  loading   the cold tensors are filtered out of vLLM's safetensors iterator (as coldskip did: vLLM's 1.6x mmap
                  peak stays where it was) and read by this component instead -- preadv straight into ONE registered
                  Grace buffer in MegaMoE's per-expert layout (runtime/gracetier_pool.py), on a thread beside vLLM's
                  own load; each layer's finalize waits for its layer. The host-RAM budget (Engram + cold pool + KV
                  overflow vs MemAvailable) is logged as one HOST_RAM_BUDGET line and the boot is refused if it does
                  not fit (checked twice: before the pool is allocated, for the pool itself; after load_model, when
                  Engram and the pool are resident and only the KV overflow is still to come).
  (iii) row_map   the w4 router gets the all-resident map (row[e] = hot slot, or H + cold-local id; every entry >= 0;
                  port/garner/make_rowmap_allres.py), so it publishes nothing and writes every pick's GROUP into
                  MegaMoE's topk_idx. (The router accepting an all->=0 map with p3 = peer = None, has = 0, and reading
                  the map per call: measured in mini-test M2, phases.M2.router. Not read from source: it is closed.)
  (iv)  staging   gt_stage (one Triton launch, grid C x NPROG; runtime/gracetier_stage.py) copies the touched cold
                  experts' weights Grace -> their groups on the CURRENT stream, right before MegaMoE:
                    T <= 16, boundary-eligible: right after the router (fastpath_boundary, patched: stage_symm below);
                    every other call (T > 16, prefill, fallbacks): inside the DeepseekV4MegaMoEExperts.forward wrap
                    (peer3's wrap pattern, P3/:164-197): groups = row_map[topk_ids] -> gt_stage -> vLLM's stock
                    forward with the groups as ids.
                  No finish, no fused finish, no combine-in-seam: MegaMoE's own combine sums all six routes.
                  Capturable: a kernel copy whose sources are picked on the device from the ids; every address (VMM
                  VAs, the Grace pool, the tables) is fixed before capture.

Numerics: every group MegaMoE reads holds the checkpoint's bytes in the checkpoint's transform, so the layer output is
bit-identical to a stock 384-group MegaMoE (M4). Versus the Pg61 control the outputs differ, because peer3 (bf16 RTX
rows, unordered atomics) is the side that changes numerics (G/docs/ARCHITECTURE.md:82, 119-122).

READ vs INFERRED (vLLM's DeepseekV4MegaMoEExperts and the w4 router are not in this tree; docs/P1-GRACETIER-DESIGN.md
section 0.5). Read in Garner's tree and used as read: the hook's _init / _weight_loader / _finalize_weights / _forward
and its `_orig` dict (hook/mega_peer_hook.py:91-232); `_transformed_l1_weights` / `_transformed_l2_weights` as
(weights, SF) pairs and `get_symm_buffer()` (fastpath_boundary.py:1340, 1375-1378; fastpath_peer3.py:106); the SF bank
layout int32 (E, N, K/128) with strides (N*K/128, 1, N) on a dedicated storage (fastpath_a12_retire.py:31, 74-86);
a12_retire reading `_mp.n_hot + _mp.n_cold == 384` as the number of experts in the module's banks (:48-56).
INFERRED, each checked at boot where possible (a failed check raises; nothing is trimmed or skipped silently):
  * vLLM's finalize produces the hot transform as [H, 2I, D/2] / [H, D, I/2] 1-byte weights + the SF banks above --
    shapes asserted; slot 0 is cross-checked against our own transform of the loader's raw tensors when the loader
    parameters (w13_weight / w2_weight + their e8m0 scales) are visible before finalize ("parity slot0" log line);
  * get_symm_buffer() is how both the boundary and vLLM's stock forward reach the module's SymmBuffer; gracetier
    builds ONE G = 384 SymmBuffer (same class, group, token count, top-k, shapes, shared-expert count as vLLM's own)
    and binds it per instance. The first eager stock-forward call checks the buffer's topk_idx holds our groups;
  * num_experts / num_local_experts are what the boundary's get_block_m_for_mega_moe and the stock forward read for
    G (B/:1344): set to 384 after finalize;
  * vLLM calls finalize_weights per layer before KV profiling, on its loader thread (HK/:139-171 says so for the
    thread); the VMM HBM is therefore seen by the KV sizing and by a12_memory_guard;
  * --kv-offloading-size is read from vllm_config.cache_config.kv_offloading_size, else sys.argv; its unit is counted
    as GiB (the larger reading: the budget errs toward refusing);
  * the MegaMoE fork (component `megamoe`) at G = 384 with zeroed fold / C11 control blocks behaves as stock MegaMoE
    (the drafter's buffers run it that way); the mini-test measured stock DeepGEMM, not the fork, at G = 384.

Env (container): GRACETIER_ROWMAP (default: rowmap-allres.json next to MEGA_ROWMAP, i.e. /w/rowmap-allres.json),
GRACETIER_RT_DIR (default: gracetier_rt/ next to this file: a copy of our runtime/ package's gracetier_*, dg_mega,
dg_moe, host_tier, ckpt modules), GRACETIER_HOST_FLOOR_GIB (default 32: [A] a policy floor, not a measurement),
GRACETIER_KV_OFFLOAD_GB (overrides the --kv-offloading-size lookup), GRACETIER_FINALIZE_S (600), GRACETIER_JOIN_S (1800).
This module imports neither torch nor vllm at import time (components.py is imported by sitecustomize in every
interpreter); vllm is never imported here at all, torch and the runtime package only inside the hooks.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from typing import Any

from fastpath_hook import activate as _hook_activate, bounded, register

TARGET = "vllm.models.deepseek_v4.nvidia.model"
WEIGHT_UTILS = "vllm.model_executor.model_loader.weight_utils"
DEFAULT_LOADER = "vllm.model_executor.model_loader.default_loader"
RUNNER = "vllm.v1.worker.gpu.model_runner"
NAME = "gracetier"

E = 384                 # routed experts per target layer = MegaMoE groups (stock geometry)
TOPK = 6
HIDDEN, INTER = 5120, 2304
SF_BLOCK = 32           # e8m0 codes per 32 K-elements; 4 codes per int32 word -> K/128 words
TENSORS_PER_EXPERT = 6  # w1/w2/w3 weight + scale (fastpath_coldskip.py:30)
GiB = 1024 ** 3
_NAME = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")   # = fastpath_coldskip._NAME
_PREFIX_DEFAULT = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.ffn\.experts$")   # = mega_peer_hook._PREFIX

_DIR = os.path.dirname(os.path.abspath(__file__))
RT_DIR = os.environ.get("GRACETIER_RT_DIR", os.path.join(_DIR, "gracetier_rt"))
ROWMAP = os.environ.get("GRACETIER_ROWMAP") or os.path.join(
    os.path.dirname(os.environ.get("MEGA_ROWMAP", "/w/rowmap-mix-v1-h255.json")), "rowmap-allres.json")
HOST_FLOOR_GIB = float(os.environ.get("GRACETIER_HOST_FLOOR_GIB", "32"))
FINALIZE_S = float(os.environ.get("GRACETIER_FINALIZE_S", "600"))   # = the hook's per-layer finalize watchdog
JOIN_S = float(os.environ.get("GRACETIER_JOIN_S", "1800"))
PREWARM_P = (TOPK, 8 * TOPK)   # gt_stage's id count P = T * 6: one value not divisible by 16, one divisible


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH gracetier: {msg}\n")
    sys.stderr.flush()


# ================================================================================================ pure (Mac-tested)
def load_allres(path: str, n_experts: int = E) -> dict:
    """{layer: {"hot", "cold", "row"}} from make_rowmap_allres.py's JSON; raises unless every layer's hot / cold is a
    partition of 0..E-1 and row is the all-resident map of it (every entry >= 0, hot -> 0..H-1, cold -> H..E-1)."""
    with open(path) as f:
        doc = json.load(f)
    layers = doc.get("layers") if isinstance(doc, dict) else None
    if not layers:
        raise RuntimeError(f"gracetier: {path} has no 'layers'")
    out = {}
    for k, v in layers.items():
        hot, cold, row = [int(x) for x in v["hot"]], [int(x) for x in v["cold"]], [int(x) for x in v["row"]]
        H = len(hot)
        part = (len(hot) + len(cold) == n_experts and set(hot) | set(cold) == set(range(n_experts))
                and not set(hot) & set(cold) and len(set(hot)) == len(hot))
        want = [-1] * n_experts
        for s, e in enumerate(hot):
            want[e] = s
        for c, e in enumerate(cold):
            want[e] = H + c
        if not part or row != want:
            raise RuntimeError(f"gracetier: {path} layer {k}: hot/cold is not a partition of 0..{n_experts - 1} "
                               f"or row is not the all-resident map (row[e] = hot slot or H + cold-local id)")
        out[int(k)] = {"hot": hot, "cold": cold, "row": row}
    return out


def hk_row(hot, cold, n_experts: int = E) -> list:
    """Garner's ColdState.row form (HK/:61-75): hot slot, or -(cold-local id + 1) -- what the hook's _weight_loader
    reads to place hot experts and drop cold ones."""
    row = [0] * n_experts
    for r, e in enumerate(hot):
        row[int(e)] = r
    for c, e in enumerate(cold):
        row[int(e)] = -(c + 1)
    return row


def cold_sets(allres: dict) -> dict:
    return {int(l): set(int(e) for e in v["cold"]) for l, v in allres.items()}


def expected_cold_tensors(allres: dict) -> int:
    """Tensors filtered from vLLM's iterator and read by gracetier: 6 per cold expert (30,960 for h255)."""
    return sum(len(v["cold"]) for v in allres.values()) * TENSORS_PER_EXPERT


def is_cold_name(name: str, cold: dict) -> bool:
    m = _NAME.match(name)
    return m is not None and int(m.group(2)) in cold.get(int(m.group(1)), ())


def host_ram_budget(mem_available_B: int, cold_pool_B: int, engram_B: int, kv_offload_B: int, floor_B: int,
                    resident=("cold_pool", "engram")) -> dict:
    """The host-RAM budget on a swapless Grace: the terms already resident are inside MemAvailable's complement;
    the pending ones must still fit with `floor_B` left. headroom = MemAvailable - sum(pending); fits = headroom >=
    floor. All inputs in bytes; GiB copies rounded to 2 decimals for the log line."""
    terms = {"engram": int(engram_B), "cold_pool": int(cold_pool_B), "kv_offload": int(kv_offload_B)}
    bad = [k for k in resident if k not in terms]
    if bad:
        raise ValueError(f"unknown budget terms {bad}; known {sorted(terms)}")
    if mem_available_B is None:
        raise ValueError("MemAvailable unknown (no /proc/meminfo): cannot budget host RAM")
    pending = [k for k in terms if k not in resident]
    pend_B = sum(terms[k] for k in pending)
    headroom = int(mem_available_B) - pend_B
    g = lambda b: round(b / GiB, 2)  # noqa: E731
    return {"MemAvailable_B": int(mem_available_B), "terms_B": terms, "resident": [k for k in terms if k in resident],
            "pending": pending, "pending_B": pend_B, "host_total_B": sum(terms.values()), "headroom_B": headroom,
            "floor_B": int(floor_B), "fits": headroom >= int(floor_B),
            "GiB": {"MemAvailable": g(mem_available_B), "engram": g(terms["engram"]),
                    "cold_pool": g(terms["cold_pool"]), "kv_offload": g(terms["kv_offload"]),
                    "pending": g(pend_B), "headroom": g(headroom), "floor": g(floor_B),
                    "host_total": g(sum(terms.values()))}}


def budget_line(stage: str, b: dict) -> str:
    G = b["GiB"]
    parts = " + ".join(f"{k} {G[k]} GiB ({'resident' if k in b['resident'] else 'pending'})"
                       for k in ("engram", "cold_pool", "kv_offload"))
    return (f"HOST_RAM_BUDGET [{stage}] {parts} = {G['host_total']} GiB; MemAvailable {G['MemAvailable']} GiB - "
            f"pending {G['pending']} GiB = headroom {G['headroom']} GiB vs floor {G['floor']} GiB -> "
            f"{'fits' if b['fits'] else 'DOES NOT FIT'} " + json.dumps(b, sort_keys=True))


def parse_kv_offload_gb(argv) -> tuple:
    """(value, saw_serve): --kv-offloading-size N / --kv-offloading-size=N from a `vllm serve` argv."""
    argv = [str(a) for a in (argv or [])]
    saw = any(a == "serve" or a.endswith("vllm") for a in argv) or "--model" in argv
    for i, a in enumerate(argv):
        if a == "--kv-offloading-size" and i + 1 < len(argv):
            return float(argv[i + 1]), saw
        if a.startswith("--kv-offloading-size="):
            return float(a.split("=", 1)[1]), saw
    return None, saw


def kv_offload_bytes(vllm_config=None, argv=None, env=None) -> tuple:
    """(bytes, source) of vLLM's host KV overflow (--kv-offloading-size). The unit is not read from vLLM: counted as
    GiB, the larger of GB / GiB, so the budget errs toward refusing. Raises when no source says anything."""
    env = os.environ if env is None else env
    v = env.get("GRACETIER_KV_OFFLOAD_GB")
    if v not in (None, ""):
        return int(float(v) * GiB), "env GRACETIER_KV_OFFLOAD_GB"
    cc = getattr(vllm_config, "cache_config", None)
    for attr in ("kv_offloading_size",):   # [inferred] vLLM CacheConfig field behind --kv-offloading-size
        val = getattr(cc, attr, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return int(float(val) * GiB), f"vllm_config.cache_config.{attr}"
    val, saw = parse_kv_offload_gb(sys.argv if argv is None else argv)
    if val is not None:
        return int(val * GiB), "argv --kv-offloading-size"
    if saw:
        return 0, "argv: vllm serve without --kv-offloading-size"
    raise RuntimeError("gracetier: cannot determine --kv-offloading-size (no env, no cache_config field, no serve argv)"
                       "; refusing to boot without a host-RAM budget")


def sf_strides(n: int, q: int) -> tuple:
    """a12_retire's MN-major SF layout for one group: shape (n, q), strides (1, n) inside a (E, n, q) bank."""
    return (n * q, 1, n)


# ================================================================================================ per-layer state
class GtState:
    """The hook's per-layer `_mp` for a gracetier layer (the boundary reads .idx and .row_map; the hook's
    _weight_loader reads .row; a12_retire reads .idx, .n_hot, .n_cold).

    n_hot = 384, n_cold = 0: a12_retire reads n_hot as "experts in this module's scale banks", and all 384 are
    resident here (design section 6.6). The real split is .H / .C."""

    gracetier = True

    def __init__(self, idx: int, hot, cold, row_allres):
        self.idx = int(idx)
        self.hot, self.cold = [int(e) for e in hot], [int(e) for e in cold]
        self.H, self.C = len(self.hot), len(self.cold)
        self.n_hot, self.n_cold = E, 0
        self.row = hk_row(self.hot, self.cold)          # hook form: _weight_loader places hot rows, drops cold
        self.row_allres = [int(r) for r in row_allres]  # gracetier form: every entry >= 0
        self.row_map = None                             # int32 [E] device (row_allres), built at finalize
        self.src_tab = self.dtab = self.cnt = None
        self.b1 = self.b2 = 0
        self.vas = None


# ================================================================================================ process state
class _State:
    def __init__(self):
        self.allres: Any = None
        self.cold: Any = None
        self.expected: Any = 0
        self.rt: Any = None
        self.hk: Any = None              # the hook's _orig dict: vLLM's original methods
        self.hk_forward: Any = None      # the hook's own forward (non-gracetier layers: drafter, non-target)
        self.prefix: Any = _PREFIX_DEFAULT
        self.vllm_config: Any = None
        self.skipped: Any = 0
        self.loads: Any = 0
        self.started: Any = False
        self.device: Any = None
        self.idx: Any = None
        self.pool: Any = None
        self.layout: Any = None
        self.engram_B: Any = 0
        self.mem_at_start: Any = None
        self.events: Any = {}
        self.cold_sf: Any = {}
        self.tensors_read: Any = 0
        self.reader: Any = None
        self.reader_error: Any = None
        self.registered: Any = threading.Event()
        self.read_s: Any = None
        self.register_s: Any = None
        self.arena: Any = None
        self.shared: Any = None
        self.symm: Any = None
        self.symm_desc: Any = None
        self.layers_done: Any = set()
        self.prewarmed: Any = set()
        self.fwd_checked: Any = False
        self.parity: Any = {}
        self.lock: Any = threading.Lock()


_S = _State()


def _rt():
    """Our runtime package from RT_DIR (gracetier_*, dg_mega, dg_moe, host_tier, ckpt; no vllm)."""
    if _S.rt is None:
        if not os.path.isdir(os.path.join(RT_DIR, "runtime")):
            raise RuntimeError(f"gracetier: runtime package missing at {RT_DIR}/runtime (the card copies it)")
        if RT_DIR not in sys.path:
            sys.path.insert(0, RT_DIR)
        import types

        import runtime
        from runtime import ckpt as CK
        from runtime import dg_mega as DM
        from runtime import gracetier_pool as GP
        from runtime import gracetier_stage as GS
        from runtime import gracetier_vmm as GV
        from runtime import host_tier as HT

        here = os.path.dirname(os.path.abspath(runtime.__file__))
        if os.path.realpath(here) != os.path.realpath(os.path.join(RT_DIR, "runtime")):
            raise RuntimeError(f"gracetier: `runtime` resolved to {here}, not {RT_DIR}/runtime")
        if not GS.HAVE_TRITON:
            raise RuntimeError("gracetier: triton is not importable; gt_stage cannot run")
        _S.rt = types.SimpleNamespace(CK=CK, DM=DM, GP=GP, GS=GS, GV=GV, HT=HT)
    return _S.rt


def _allres() -> dict:
    if _S.allres is None:
        _S.allres = load_allres(ROWMAP)
        _S.cold = cold_sets(_S.allres)
        _S.expected = expected_cold_tensors(_S.allres)
        hs = sorted({len(v["hot"]) for v in _S.allres.values()})
        _log(f"all-resident rowmap {ROWMAP}: {len(_S.allres)} layers, H {hs}, {_S.expected} cold tensors read by "
             f"gracetier (filtered from vLLM's iterator), G = {E}")
    return _S.allres


# ================================================================================================ the Grace pool
def _start_reader(model_dir: str, device) -> None:
    """Scan the checkpoint headers, budget + allocate the pool (raises here, on the loader's thread, if it does not
    fit), then read every layer's cold experts on a thread (weights into the pool, e8m0 codes -> SF banks on the CPU);
    register the pool once all layers are read."""
    import torch

    R = _rt()
    allres = _allres()
    t0 = time.time()
    _S.idx = R.CK.ShardIndex.scan(model_dir)
    _S.engram_B = sum(_S.idx[n].nbytes for n in _S.idx.names() if ".engram." in n)
    _S.layout = R.GP.PoolLayout({l: len(v["cold"]) for l, v in allres.items()}, HIDDEN, INTER)
    _S.mem_at_start = R.HT.mem_available_bytes()
    kv_B, kv_src = kv_offload_bytes(_S.vllm_config)
    floor_B = int(HOST_FLOOR_GIB * GiB)
    pre = host_ram_budget(_S.mem_at_start, _S.layout.total, _S.engram_B, kv_B, floor_B,
                          resident=("engram", "kv_offload"))
    _log(budget_line("pre-alloc: the cold pool must fit now; Engram / KV residency is checked after load_model",
                     pre) + f" kv_source={kv_src!r} scan_s={time.time() - t0:.2f}")
    if not pre["fits"]:
        raise RuntimeError(f"gracetier: host RAM: the cold pool ({pre['GiB']['cold_pool']} GiB) leaves "
                           f"{pre['GiB']['headroom']} GiB < floor {pre['GiB']['floor']} GiB; refusing to boot")
    _S.pool = R.GP.GracePool(_S.layout, name="gracetier_cold", reserve_bytes=floor_B)
    _S.events = {l: threading.Event() for l in allres}
    _S.device = device
    _S.started = True

    def _main():
        try:
            t1 = time.time()
            for l in sorted(allres):
                cold = allres[l]["cold"]
                _S.pool.read_layer(_S.idx, l, cold)
                codes = R.GP.read_codes(_S.idx, l, cold, HIDDEN, INTER)
                sf1, sf2 = R.GP.sf_banks(codes, "mn")
                with _S.lock:
                    _S.cold_sf[l] = (sf1, sf2)
                    _S.tensors_read += 3 * len(cold)
                _S.events[l].set()
            _S.tensors_read += _S.pool.tensors_read
            _S.read_s = time.time() - t1
            torch.cuda.set_device(device)
            _S.register_s = _S.pool.register()
            _log(f"cold pool read: {_S.tensors_read} tensors, {_S.pool.bytes_read / 1e9:.2f} GB weights in "
                 f"{_S.read_s:.1f} s; registered {len(_S.pool.buf.chunks)} chunk(s) in {_S.register_s:.2f} s "
                 f"({_S.layout.total / GiB:.2f} GiB, hugepages {_S.pool.buf.hugepages})")
        except BaseException:  # noqa: BLE001 - surfaced by every waiter
            _S.reader_error = traceback.format_exc()
            _log(f"FATAL cold pool reader: {_S.reader_error[-1500:]}")
        finally:
            for ev in _S.events.values():
                ev.set()
            _S.registered.set()

    _S.reader = threading.Thread(target=_main, name="gracetier-reader", daemon=True)
    _S.reader.start()
    _log(f"cold pool reader started: {sum(len(v['cold']) for v in allres.values())} experts, "
         f"{_S.layout.total / GiB:.2f} GiB pool, model {model_dir}")


def _cold_sf_of(idx: int):
    ev = _S.events.get(idx)
    if ev is None:
        raise RuntimeError(f"gracetier layer {idx}: finalize before the cold pool reader started (no target load?)")
    if not ev.wait(FINALIZE_S):
        raise RuntimeError(f"gracetier layer {idx}: cold pool reader did not reach this layer in {FINALIZE_S:g} s")
    if _S.reader_error:
        raise RuntimeError(f"gracetier: cold pool reader failed:\n{_S.reader_error[-2000:]}")
    with _S.lock:
        return _S.cold_sf.pop(idx)


# ================================================================================================ finalize
def _vmm(dev):
    """The process's VMM arena and the shared scratch (one physical allocation per weight tensor kind, sized by the
    largest cold range over all layers)."""
    if _S.arena is None:
        R = _rt()
        R.GV.ensure_context(dev.index)
        _S.arena = R.GV.VmmArena(dev.index)
        plans = [R.GV.plan_layer(E, len(v["hot"]), HIDDEN, INTER, _S.arena.gran) for v in _S.allres.values()]
        sizes = R.GV.shared_sizes(plans)
        _S.shared = {k: _S.arena.alloc(n, f"shared.{k}") for k, n in sizes.items() if n > 0}
        bud = R.GV.hbm_budget([len(v["hot"]) for v in _S.allres.values()], E, HIDDEN, INTER, _S.arena.gran)
        sh = {k: round(a.nbytes / GiB, 3) for k, a in _S.shared.items()}
        _log(f"VMM: granularity {_S.arena.gran} B; shared scratch {sh} GiB; plan for all layers: "
             f"weights {bud['weights_total_GiB']} GiB (shared {bud['shared_total_GiB']}, straddle "
             f"{bud['straddle_GiB']}), SF all groups {bud['sf_GiB']} GiB")
    return _S.arena, _S.shared


def _symm_for(module):
    """ONE G = 384 SymmBuffer for every gracetier layer: vLLM's own buffer's class / group / token count / top-k /
    shapes / shared-expert count, at num_experts = 384. [inferred] get_symm_buffer() is the module's accessor."""
    if _S.symm is None:
        sb = type(module).get_symm_buffer(module)
        if int(getattr(sb, "num_experts", -1)) == E:
            _S.symm = sb
            how = "vLLM's own (already G = 384)"
        else:
            kw = {"num_shared_experts": int(getattr(sb, "num_shared_experts", 0))}
            for k in ("mma_type", "activation"):
                if hasattr(sb, k):
                    kw[k] = getattr(sb, k)
            _S.symm = type(sb)(sb.group, E, int(sb.num_max_tokens_per_rank), int(sb.num_topk), int(sb.hidden),
                               int(sb.intermediate_hidden), **kw)
            how = f"new (vLLM's has G = {getattr(sb, 'num_experts', '?')}, {int(sb.buffer.numel())} B)"
        _S.symm_desc = {"num_experts": int(_S.symm.num_experts), "bytes": int(_S.symm.buffer.numel()),
                        "num_max_tokens_per_rank": int(_S.symm.num_max_tokens_per_rank),
                        "num_shared_experts": int(getattr(_S.symm, "num_shared_experts", 0)), "how": how}
        _log(f"SymmBuffer G = {E}: {json.dumps(_S.symm_desc, sort_keys=True)}")
    return _S.symm


def _first_param(module, names, shape):
    import torch

    for n in names:
        t = getattr(module, n, None)
        if isinstance(t, torch.Tensor) and tuple(t.shape) == tuple(shape) and t.element_size() == 1:
            return n, t
    return None, None


def _slot0_snapshot(module, st):
    """Hot slot 0's raw loader tensors (CPU copies), taken before vLLM's finalize drops them, or (None, why). The
    parameter names are [inferred] from the hook (`w13_` / `w2_` in weight_name, HK/:119-126)."""
    I, D = INTER, HIDDEN
    n13, w13 = _first_param(module, ("w13_weight",), (st.H, 2 * I, D // 2))
    n2, w2 = _first_param(module, ("w2_weight",), (st.H, D, I // 2))
    s_names = ("w13_weight_scale", "w13_weight_scale_inv", "w13_scale")
    ns13, s13 = _first_param(module, s_names, (st.H, 2 * I, D // SF_BLOCK))
    ns2, s2 = _first_param(module, tuple(s.replace("w13", "w2") for s in s_names), (st.H, D, I // SF_BLOCK))
    if w13 is None or w2 is None:
        return None, "loader weight parameters not visible as w13_weight / w2_weight [H, ...] (1-byte)"
    snap = {"w13": w13[0].detach().cpu().clone(), "w2": w2[0].detach().cpu().clone(),
            "s13": None if s13 is None else s13[0].detach().cpu().clone(),
            "s2": None if s2 is None else s2[0].detach().cpu().clone(), "names": [n13, n2, ns13, ns2]}
    return snap, None


def _parity_slot0(st, snap, w1, b1, w2, b2) -> dict:
    """Our transform of hot slot 0's raw tensors == vLLM's finalized group 0 (weights; SF when the scales were seen).
    This is the check that our cold transform (the same functions) matches vLLM's hot transform."""
    import torch

    DM = _rt().DM
    I = INTER
    out = {"names": snap["names"]}
    a = DM.mega_l1_weight(snap["w13"][:I].view(torch.int8), snap["w13"][I:].view(torch.int8))
    out["l1_weight"] = bool(torch.equal(a.view(torch.uint8), w1[0].detach().cpu().view(torch.uint8)))
    out["l2_weight"] = bool(torch.equal(snap["w2"].view(torch.uint8), w2[0].detach().cpu().view(torch.uint8)))
    if snap["s13"] is not None:
        s = DM.mega_l1_sf(snap["s13"][:I], snap["s13"][I:], "mn")
        out["l1_sf"] = bool(torch.equal(s.contiguous(), b1[0].detach().cpu().contiguous()))
    else:
        out["l1_sf"] = "not checked: no w13 e8m0 scale parameter visible"
    if snap["s2"] is not None:
        s = DM.mega_l2_sf(snap["s2"], "mn")
        out["l2_sf"] = bool(torch.equal(s.contiguous(), b2[0].detach().cpu().contiguous()))
    else:
        out["l2_sf"] = "not checked: no w2 e8m0 scale parameter visible"
    out["ok"] = parity_ok(out)
    return out


def parity_ok(par: dict) -> bool:
    """Weights must be checked and equal; an SF entry that could not be checked (a string) is not a failure."""
    return (par.get("l1_weight") is True and par.get("l2_weight") is True
            and all(par.get(k) is True or isinstance(par.get(k), str) for k in ("l1_sf", "l2_sf")))


def _sf_bank(dev, hot_sf, cold_sf, n: int, q: int, H: int, C: int, what: str):
    """A dedicated (E, n, q) int32 MN-major bank (a12_retire's layout): groups 0..H-1 from vLLM, H..E-1 from ours."""
    import torch

    if tuple(hot_sf.shape) != (H, n, q) or hot_sf.dtype != torch.int32:
        raise RuntimeError(f"gracetier: vLLM's {what} SF is {tuple(hot_sf.shape)} {hot_sf.dtype}, expected "
                           f"({H}, {n}, {q}) int32")
    if tuple(cold_sf.shape) != (C, n, q) or cold_sf.dtype != torch.int32:
        raise RuntimeError(f"gracetier: cold {what} SF is {tuple(cold_sf.shape)} {cold_sf.dtype}, expected ({C}, {n}, {q})")
    bank = torch.empty_strided((E, n, q), sf_strides(n, q), dtype=torch.int32, device=dev)
    bank[:H].copy_(hot_sf)
    bank[H:].copy_(cold_sf.to(dev))
    return bank


def _prewarm(st, dev) -> None:
    """Compile gt_stage's specializations eagerly (P divisible by 16 or not), touching nothing (ids all -1)."""
    import torch

    key = (st.H, st.C)
    if key in _S.prewarmed:
        return
    GS = _rt().GS
    for P in PREWARM_P:
        g = torch.full((P,), -1, dtype=torch.int64, device=dev)
        GS.gt_stage(g, st.H, st.src_tab, st.dtab, st.cnt, st.b1, st.b2)
    torch.cuda.synchronize(dev)
    _S.prewarmed.add(key)


def _finalize_layer(module, st, dev, snap, why_no_snap) -> None:
    import torch

    R = _rt()
    I, D, H, C = INTER, HIDDEN, st.H, st.C
    l1, sf1 = module._transformed_l1_weights
    l2, sf2 = module._transformed_l2_weights
    for t, shp, what in ((l1, (H, 2 * I, D // 2), "L1"), (l2, (H, D, I // 2), "L2")):
        if tuple(t.shape) != shp or t.element_size() != 1:
            raise RuntimeError(f"gracetier layer {st.idx}: vLLM's {what} weights are {tuple(t.shape)} {t.dtype}, "
                               f"expected {shp} 1-byte")
    csf1, csf2 = _cold_sf_of(st.idx)
    arena, shared = _vmm(dev)
    plans = R.GV.plan_layer(E, H, D, I, arena.gran)
    va1 = R.GV.GroupedVA(arena, plans["l1"], shared.get("l1"), map_hot=True, name=f"L{st.idx}.l1")
    va2 = R.GV.GroupedVA(arena, plans["l2"], shared.get("l2"), map_hot=True, name=f"L{st.idx}.l2")
    w1 = va1.tensor(l1.dtype, (E, 2 * I, D // 2), dev)
    w2 = va2.tensor(l2.dtype, (E, D, I // 2), dev)
    w1[:H].copy_(l1)
    w2[:H].copy_(l2)
    b1 = _sf_bank(dev, sf1, csf1, 2 * I, D // (4 * SF_BLOCK), H, C, "L1")
    b2 = _sf_bank(dev, sf2, csf2, D, I // (4 * SF_BLOCK), H, C, "L2")
    if snap is not None:
        par = _parity_slot0(st, snap, w1, b1, w2, b2)
        _S.parity[st.idx] = par
        if st.idx == min(_S.allres) or not par["ok"]:
            _log(f"parity slot0 layer {st.idx} (our transform of the loader's raw hot slot 0 vs vLLM's finalized "
                 f"group 0): {json.dumps(par, sort_keys=True)}")
        if not par["ok"]:
            raise RuntimeError(f"gracetier layer {st.idx}: our MegaMoE transform differs from vLLM's on hot slot 0 "
                               f"({par}); the cold groups would not be the checkpoint's bytes. Refusing.")
    elif st.idx == min(_S.allres):
        _log(f"parity slot0: not checked ({why_no_snap})")
    module._transformed_l1_weights = (w1, b1)
    module._transformed_l2_weights = (w2, b2)
    del l1, sf1, l2, sf2
    st.row_map = torch.tensor(st.row_allres, dtype=torch.int32, device="cpu").to(dev)
    st.src_tab = _S.pool.src_table(st.idx, dev)
    st.dtab = R.GS.make_dtab(w1, w2)
    st.cnt = torch.zeros(C, dtype=torch.int32, device=dev)
    st.b1, st.b2 = R.GP.l1_bytes(D, I), R.GP.l2_bytes(D, I)
    st.vas = (va1, va2)
    # [inferred] what the boundary's get_block_m_for_mega_moe (B/:1344) and vLLM's stock forward read for G
    module.num_experts = E
    module.num_local_experts = E
    sb = _symm_for(module)
    module.get_symm_buffer = (lambda _sb=sb: _sb)
    _prewarm(st, dev)
    torch.cuda.synchronize(dev)
    _S.layers_done.add(st.idx)
    if st.idx in (min(_S.allres), max(_S.allres)):
        _log(f"finalize layer {st.idx}: G = {E} (hot 0..{H - 1} private HBM, cold {H}..{E - 1} shared VMM scratch), "
             f"SF banks {tuple(b1.shape)} / {tuple(b2.shape)}, row_map all >= 0, {C} cold sources in Grace; "
             f"VMM arena {json.dumps(arena.describe(), sort_keys=True)}")


# ================================================================================================ per call
def stage(st, groups) -> None:
    """gt_stage on the current stream: the cold groups named in `groups` (contiguous int64, any shape; -1 = skip)
    get their Grace bytes. One launch; capturable."""
    if st.src_tab is None:
        raise RuntimeError(f"gracetier layer {st.idx}: forward before finalize")
    _S.rt.GS.gt_stage(groups, st.H, st.src_tab, st.dtab, st.cnt, st.b1, st.b2)


def stage_symm(experts, symm, T: int) -> None:
    """The boundary's T <= 16 entry (patches/0001): right after the router wrote the groups into symm.topk_idx."""
    stage(experts._mp, symm.topk_idx[:T].view(-1))


def is_gracetier(experts) -> bool:
    return isinstance(getattr(experts, "_mp", None), GtState)


# ================================================================================================ the hook wraps
def _target_layer(prefix: str, num_experts: int, num_local_experts: int):
    m = _S.prefix.search(prefix or "")
    if m is None or num_experts != E or num_local_experts != E:
        return None
    idx = int(m.group(1))
    return idx if idx in _allres() else None


def _init(self, vllm_config, *, num_experts, num_local_experts, experts_start_idx, **kw):
    """The hook's _init (HK/:91-106) with the all-resident split: vLLM's experts module is built for the H hot
    experts (loader parameters and transform for H only, so HBM is not allocated for 384)."""
    idx = _target_layer(kw.get("prefix", ""), num_experts, num_local_experts)
    if idx is None:
        _S.hk["init"](self, vllm_config, num_experts=num_experts, num_local_experts=num_local_experts,
                      experts_start_idx=experts_start_idx, **kw)
        self._mp = None
        return
    sp = _allres()[idx]
    H = len(sp["hot"])
    kw["num_logical_experts"] = E
    _S.hk["init"](self, vllm_config, num_experts=H, num_local_experts=H, experts_start_idx=0, **kw)
    self._mp = GtState(idx, sp["hot"], sp["cold"], sp["row"])
    if _S.vllm_config is None:
        _S.vllm_config = vllm_config
    if idx == min(_S.allres):
        _log(f"layer {idx} ({kw.get('prefix')}): {H} hot experts in HBM groups 0..{H - 1}, {E - H} cold in Grace "
             f"(groups {H}..{E - 1} on the shared VMM scratch), MegaMoE G = {E}")


def _finalize_weights(self, shared_experts=None):
    st = getattr(self, "_mp", None)
    if not isinstance(st, GtState):
        return _S.hk["finalize_weights"](self, shared_experts)
    import torch

    w = getattr(self, "w13_weight", None)
    dev = w.device if w is not None else torch.device("cuda", torch.cuda.current_device())   # as HK/:149
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    with bounded(f"gracetier finalize_weights layer {st.idx}", FINALIZE_S):
        with torch.cuda.device(dev):
            snap, why = _slot0_snapshot(self, st)
            _S.hk["finalize_weights"](self, shared_experts)
            _finalize_layer(self, st, dev, snap, why)
            torch.cuda.empty_cache()
    return None


def _forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
    """T > 16 and every non-boundary call: groups = row_map[ids] -> gt_stage -> vLLM's stock MegaMoE forward with the
    groups as ids (peer3's wrap pattern, P3/:164-197; the hook's forward for non-gracetier layers)."""
    st = getattr(self, "_mp", None)
    if not isinstance(st, GtState):
        return _S.hk_forward(self, hidden_states, topk_weights, topk_ids, activation_clamp=activation_clamp,
                             fast_math=fast_math)
    import torch

    if st.row_map is None:
        raise RuntimeError(f"gracetier layer {st.idx}: row_map not finalized")
    ids = topk_ids.contiguous()
    groups = torch.where(ids >= 0, st.row_map[ids.clamp(min=0).long()].to(ids.dtype), ids)
    if ids.numel():
        stage(st, groups.reshape(-1).to(torch.int64).contiguous())
    y = _S.hk["forward"](self, hidden_states, topk_weights, groups, activation_clamp=activation_clamp,
                         fast_math=fast_math)
    if not _S.fwd_checked and not torch.cuda.is_current_stream_capturing():
        # MegaMoE leaves topk_idx in place after the call (Garner's seam combine reads symm.topk_idx after MegaMoE,
        # B/:1385), so this reads what the stock forward staged for THIS call.
        sb = self.get_symm_buffer()
        T = ids.shape[0]
        if 0 < T <= int(sb.topk_idx.shape[0]):
            ok = bool(torch.equal(sb.topk_idx[:T].long(), groups.long()))
            _log(f"first eager stock-forward call (layer {st.idx}, T {T}): G = {E} SymmBuffer topk_idx == our groups: "
                 f"{ok}")
            if not ok:
                raise RuntimeError("gracetier: vLLM's stock MegaMoE forward did not read the G = 384 SymmBuffer bound "
                                   "by get_symm_buffer(); the cold groups would be wrong. Refusing.")
            _S.fwd_checked = True
    return y


# ================================================================================================ registrations
@register(NAME, TARGET, defer_active=True)
def patch_model(module) -> None:
    """Install the wraps over mega_peer_hook's methods (the hook must be installed first, as for peer3)."""
    cls = module.DeepseekV4MegaMoEExperts
    if getattr(cls, "_gracetier", False):
        return
    hook_forward = cls.forward
    g = getattr(hook_forward, "__globals__", {})
    orig = g.get("_orig") or {}
    if not all(k in orig for k in ("init", "weight_loader", "finalize_weights", "forward")):
        raise RuntimeError("gracetier needs the MegaMoE hook (mega_peer_hook) installed first")
    if getattr(hook_forward, "__module__", "") == "fastpath_peer3":
        raise RuntimeError("gracetier and peer3 are exclusive: disarm peer3 (and coldskip)")
    _allres()
    _S.hk = orig
    _S.hk_forward = hook_forward
    if isinstance(g.get("_PREFIX"), re.Pattern):
        _S.prefix = g["_PREFIX"]
    cls.__init__ = _init
    cls.finalize_weights = _finalize_weights
    cls.forward = _forward
    cls._gracetier = True
    _log(f"{TARGET}.DeepseekV4MegaMoEExperts: __init__ / finalize_weights / forward wrapped (weight_loader stays the "
         f"hook's: cold rows are dropped there, and never reach it)")


@register(NAME, WEIGHT_UTILS, defer_active=True)
def patch_weight_utils(module) -> None:
    """Filter the cold-expert names before vLLM's safetensors iterator reads them (coldskip's filter, B/coldskip:59)."""
    orig = module.should_skip_weight
    if getattr(orig, "_ds41_gracetier", False):
        return
    cold = cold_sets(_allres())

    def should_skip_weight(weight_name, local_expert_ids):
        if is_cold_name(weight_name, cold):
            _S.skipped += 1
            return True
        return orig(weight_name, local_expert_ids)

    should_skip_weight._ds41_gracetier = True
    module.should_skip_weight = should_skip_weight


@register(NAME, DEFAULT_LOADER, defer_active=True)
def patch_default_loader(module) -> None:
    """Target load: start the cold pool reader first, then check the skip count (coldskip's check, :86-98)."""
    cls = module.DefaultModelLoader
    orig = cls.load_weights
    if getattr(orig, "_ds41_gracetier", False):
        return

    def load_weights(self, model, model_config):
        import torch

        first = _S.loads == 0
        if first:
            _start_reader(str(getattr(model_config, "model", "/model")),
                          torch.device("cuda", torch.cuda.current_device()))
        before = _S.skipped
        out = orig(self, model, model_config)
        got = _S.skipped - before
        _S.loads += 1
        if first and got != _S.expected:
            raise RuntimeError(f"gracetier: target load skipped {got} cold-expert tensors; the all-resident rowmap "
                               f"implies {_S.expected}")
        _log(f"load {_S.loads}: skipped {got} cold-expert tensors before reading (gracetier reads them into Grace)")
        return out

    load_weights._ds41_gracetier = True
    cls.load_weights = load_weights


@register(NAME, RUNNER, defer_active=True)
def patch_runner(module) -> None:
    """After load_model: join the reader, check every target layer engaged, the host-RAM budget; then ACTIVE."""
    cls = module.GPUModelRunner
    orig = cls.load_model
    if getattr(orig, "_ds41_gracetier", False):
        return

    def load_model(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if not _S.started:
            raise RuntimeError("gracetier armed but the target load never started the cold pool reader")
        with bounded("gracetier cold pool read + registration", JOIN_S):
            _S.registered.wait()
        if _S.reader_error:
            raise RuntimeError(f"gracetier: cold pool reader failed:\n{_S.reader_error[-2000:]}")
        want = set(_S.allres)
        if _S.layers_done != want:
            raise RuntimeError(f"gracetier: layers finalized {sorted(_S.layers_done)}; expected all of {sorted(want)}")
        if _S.tensors_read != _S.expected:
            raise RuntimeError(f"gracetier: read {_S.tensors_read} cold tensors into Grace; expected {_S.expected}")
        if not _S.pool.buf.is_registered():
            raise RuntimeError("gracetier: the cold pool is not registered")
        R = _rt()
        kv_B, kv_src = kv_offload_bytes(_S.vllm_config)
        b = host_ram_budget(R.HT.mem_available_bytes(), _S.layout.total, _S.engram_B, kv_B,
                            int(HOST_FLOOR_GIB * GiB), resident=("cold_pool", "engram"))
        b["MemAvailable_at_load_start_B"] = _S.mem_at_start
        b["kv_source"] = kv_src
        _log(budget_line("post-load: Engram + cold pool resident, KV overflow pending", b))
        if not b["fits"]:
            raise RuntimeError(f"gracetier: host RAM budget does not fit (headroom {b['GiB']['headroom']} GiB < floor "
                               f"{b['GiB']['floor']} GiB); refusing to boot")
        _hook_activate(NAME)
        hs = sorted({len(v["hot"]) for v in _S.allres.values()})
        _log(f"ENGAGED: {len(_S.layers_done)} layers, H {hs}, G = {E}, {_S.tensors_read} cold tensors in Grace "
             f"({_S.layout.total / GiB:.2f} GiB, read {_S.read_s:.1f} s, registered {_S.register_s:.2f} s); VMM "
             f"{json.dumps(_S.arena.describe(), sort_keys=True)}; SymmBuffer {json.dumps(_S.symm_desc, sort_keys=True)}; "
             f"parity slot0 checked on {sum(1 for p in _S.parity.values() if p.get('ok'))}/{len(_S.layers_done)} "
             f"layers")
        return result

    load_model._ds41_gracetier = True
    cls.load_model = load_model
