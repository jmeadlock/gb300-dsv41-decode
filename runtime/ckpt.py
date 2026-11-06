"""Shard index over the HF safetensors checkpoint: headers scanned once, convert.py's renames applied, and
byte-range reads straight into caller-owned buffers (no page-cache doubling).

Why not `safetensors.safe_open`: it mmaps the shard, so every byte read lands in the page cache and stays there
until memory pressure evicts it. On a 494 GiB swapless Grace that will pin ~251 GiB, the page cache would double
the host tier during boot. Here every read is `preadv` into the destination buffer followed by
`posix_fadvise(POSIX_FADV_DONTNEED)` on the range just read.

The rename map is a line-for-line copy of `reference/deepseek-inference/convert.py:104-126` (`convert_name`),
checked against convert.py itself in `tests/test_loader_mini.py::test_rename_matches_convert_py`.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

# safetensors dtype tag -> (torch dtype attribute name, bytes per element). F4/U4/F6 are listed by the inventory tool
# but the 1017 checkpoint stores experts as I8 (two nibbles per byte), so sub-byte tags are refused rather than guessed.
ST_DTYPES: Dict[str, Tuple[str, int]] = {
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "F8_E8M0": ("float8_e8m0fnu", 1),
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I8": ("int8", 1),
    "U8": ("uint8", 1),
    "I16": ("int16", 2),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
    "BOOL": ("bool", 1),
}

# convert.py:56-64. Every entry maps a key to itself at MP=1 (the dim only matters for MP>1 sharding), so the
# rename step at convert.py:122-126 is the identity; it is kept literally so a future checkpoint that changes the
# table changes us too.
CONVERT_MAPPING = {
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}

READ_CHUNK = 64 << 20  # bytes per preadv + fadvise step


def torch_dtype(st_dtype: str) -> torch.dtype:
    if st_dtype not in ST_DTYPES:
        raise ValueError(f"unsupported safetensors dtype {st_dtype!r}")
    return getattr(torch, ST_DTYPES[st_dtype][0])


def convert_name(source_name: str) -> Optional[str]:
    """The name convert.py gives `source_name`, or None when convert.py drops it.

    Line-for-line from convert.py:104-126 (verified against the file, not against the handoff's summary):
      104-106 strip `model.`;  109-110 drop `mtp.*.embed.weight` / `mtp.*.head.weight` (tied to the backbone);
      111 `self_attn`->`attn`;  112-113 `mlp`->`ffn` except under `vision.`;  114 `weight_scale_inv`->`scale`;
      115 `e_score_correction_bias`->`bias`;  116-126 the `mapping` key rename (identity at MP=1, see above).
    The 1017 checkpoint already uses the converted spelling (`layers.0.ffn.experts.0.w1.scale`, inventory examples),
    so on it this is the identity except for the mtp drop; HF-spelled names are tested on the mini checkpoint.
    """
    name = source_name
    if name.startswith("model."):
        name = name[len("model."):]
    if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        return None
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_"]):  # without .weight
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    new_key = CONVERT_MAPPING[key][0] if key in CONVERT_MAPPING else key
    name = name.replace(key, new_key)
    return name


@dataclass(frozen=True)
class TensorEntry:
    name: str          # runtime (convert.py) name
    source_name: str   # name in the shard header
    path: str          # absolute shard path
    offset: int        # absolute byte offset of the tensor data in the shard
    nbytes: int
    st_dtype: str
    shape: Tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        return torch_dtype(self.st_dtype)

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


def read_header(path: str) -> Tuple[int, dict]:
    """(data_start, header dict) of one safetensors file; reads only the 8-byte length and the JSON."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        n = struct.unpack("<Q", raw)[0]
        hdr = json.loads(fh.read(n))
    return 8 + n, hdr


def _fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    # Linux: drop the clean pages we just read. macOS has no posix_fadvise; the Mac only runs the tiny test shards.
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def writable_bytes(dst) -> memoryview:
    """A flat writable byte view of `dst`: a CPU torch tensor (contiguous, any dtype incl. fp8/fp4/bf16),
    an mmap, a bytearray, a numpy array or a memoryview."""
    if isinstance(dst, torch.Tensor):
        if dst.device.type != "cpu":
            raise ValueError(f"read target must be host memory, got {dst.device}")
        if not dst.is_contiguous():
            raise ValueError("read target tensor must be contiguous")
        nbytes = dst.numel() * dst.element_size()
        if nbytes == 0:
            return memoryview(bytearray(0))
        import ctypes

        return memoryview((ctypes.c_char * nbytes).from_address(dst.data_ptr())).cast("B")
    mv = memoryview(dst)
    if mv.readonly:
        raise ValueError("read target is read-only")
    return mv.cast("B") if mv.format != "B" or mv.ndim != 1 else mv


@dataclass
class ShardIndex:
    model_dir: str
    entries: Dict[str, TensorEntry] = field(default_factory=dict)      # by runtime name
    dropped: List[str] = field(default_factory=list)                    # source names convert.py drops
    shards: List[str] = field(default_factory=list)
    _fds: Dict[str, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------------------------------------------------------------- scan
    @classmethod
    def scan(cls, model_dir: str, check_sizes: bool = True) -> "ShardIndex":
        """Scan every `*.safetensors` header once. Refuses on: duplicate source names, two source names converting
        to one runtime name, byte ranges that disagree with dtype x shape, data past end of file (`check_sizes`),
        and a `model.safetensors.index.json` whose name set differs from the shards' (convert.py:148-152)."""
        model_dir = os.path.abspath(model_dir)
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {model_dir}")
        idx = cls(model_dir=model_dir, shards=[os.path.join(model_dir, f) for f in files])
        seen_source: Dict[str, str] = {}
        for path in idx.shards:
            data_start, hdr = read_header(path)
            fsize = os.path.getsize(path)
            for src, v in hdr.items():
                if src == "__metadata__":
                    continue
                if src in seen_source:
                    raise ValueError(f"tensor {src} in both {seen_source[src]} and {path}")
                seen_source[src] = path
                b0, b1 = v["data_offsets"]
                shape = tuple(int(s) for s in v["shape"])
                st = v["dtype"]
                if st not in ST_DTYPES:
                    raise ValueError(f"{src}: unsupported dtype {st}")
                numel = 1
                for s in shape:
                    numel *= s
                if b1 - b0 != numel * ST_DTYPES[st][1]:
                    raise ValueError(f"{src}: {b1 - b0} bytes != {numel} x {st}")
                if check_sizes and data_start + b1 > fsize:
                    raise ValueError(f"{src}: data ends at {data_start + b1} > file size {fsize} ({path})")
                name = convert_name(src)
                if name is None:
                    idx.dropped.append(src)
                    continue
                if name in idx.entries:
                    raise ValueError(f"{src} and {idx.entries[name].source_name} both convert to {name}")
                idx.entries[name] = TensorEntry(name, src, path, data_start + b0, b1 - b0, st, shape)
        index_json = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_json):
            expected = set(json.load(open(index_json))["weight_map"])
            got = set(seen_source)
            if expected != got:
                raise ValueError(
                    f"checkpoint shards incomplete: {len(expected - got)} tensors missing, "
                    f"{len(got - expected)} unexpected vs model.safetensors.index.json "
                    f"(e.g. missing {sorted(expected - got)[:5]}, unexpected {sorted(got - expected)[:5]})"
                )
        return idx

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __getitem__(self, name: str) -> TensorEntry:
        return self.entries[name]

    def names(self) -> Iterable[str]:
        return self.entries.keys()

    # ---------------------------------------------------------------- reads
    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            return fd

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()

    def read_into(self, name: str, dst, start: int = 0, nbytes: Optional[int] = None, fadvise: bool = True) -> int:
        """Read bytes [start, start+nbytes) of tensor `name` into `dst` (exactly nbytes long). Thread-safe
        (positional reads). Each chunk is followed by POSIX_FADV_DONTNEED on the range just read."""
        e = self.entries[name]
        if nbytes is None:
            nbytes = e.nbytes - start
        if start < 0 or start + nbytes > e.nbytes:
            raise ValueError(f"{name}: range [{start},{start + nbytes}) outside {e.nbytes} bytes")
        mv = writable_bytes(dst)
        if len(mv) != nbytes:
            raise ValueError(f"{name}: destination holds {len(mv)} bytes, range is {nbytes}")
        fd = self._fd(e.path)
        done = 0
        while done < nbytes:
            n = min(READ_CHUNK, nbytes - done)
            off = e.offset + start + done
            got = os.preadv(fd, [mv[done:done + n]], off)
            if got <= 0:
                raise IOError(f"{name}: short read at {off} ({e.path})")
            if fadvise:
                _fadvise_dontneed(fd, off, got)
            done += got
        return nbytes

    def read_tensor_into(self, name: str, dst_buffer) -> int:
        """Read the whole tensor's bytes into `dst_buffer` (a host buffer of exactly `nbytes`)."""
        return self.read_into(name, dst_buffer)

    def read_tensor(self, name: str) -> torch.Tensor:
        """Allocate a CPU tensor in the checkpoint dtype/shape and read it (small tensors: wo_a, widening)."""
        e = self.entries[name]
        t = torch.empty(e.shape, dtype=e.dtype)
        self.read_into(name, t)
        return t

    def file_sha256(self, name: str, chunk: int = READ_CHUNK) -> str:
        """sha256 of the tensor's shard bytes from an independent positional read pass (the verify reference)."""
        e = self.entries[name]
        fd = self._fd(e.path)
        h = hashlib.sha256()
        done = 0
        while done < e.nbytes:
            n = min(chunk, e.nbytes - done)
            b = os.pread(fd, n, e.offset + done)
            if not b:
                raise IOError(f"{name}: short read")
            _fadvise_dontneed(fd, e.offset + done, len(b))
            h.update(b)
            done += len(b)
        return h.hexdigest()

    def total_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries.values())
