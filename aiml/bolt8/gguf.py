from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

# GGUF v3
GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_F8E4M3 = 24  # llama.cpp; may vary by version
GGML_TYPE_F8E5M2 = 25

_VALUE_READERS = {
    0: ("B", 1),   # uint8
    1: ("b", 1),   # int8
    2: ("H", 2),   # uint16
    3: ("h", 2),   # int16
    4: ("I", 4),   # uint32
    5: ("i", 4),   # int32
    6: ("f", 4),   # float32
    7: ("Q", 8),   # bool stored as 1 byte in v3 — handled separately
    10: ("Q", 8),  # uint64
    11: ("q", 8),  # int64
    12: ("d", 8),  # float64
}


class GgufError(RuntimeError):
    pass


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _read_value(f, typ: int):
    if typ == 7:  # bool
        return bool(f.read(1)[0])
    if typ == 8:  # string
        return _read_str(f)
    if typ == 9:  # array
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, et) for _ in range(n)]
    spec = _VALUE_READERS.get(typ)
    if spec is None:
        raise GgufError(f"unsupported metadata type {typ}")
    fmt, sz = spec
    return struct.unpack("<" + fmt, f.read(sz))[0]


def parse_gguf_header(path: str | Path) -> dict:
    """Read GGUF metadata and tensor table. Does not mmap weights."""
    path = Path(path)
    tensors = []
    kv = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise GgufError("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_kv):
            key = _read_str(f)
            (typ,) = struct.unpack("<I", f.read(4))
            kv[key] = _read_value(f, typ)
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims)))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
        data_start = f.tell()
        # align to 32 bytes
        if data_start % 32:
            data_start += 32 - (data_start % 32)
    return {
        "path": str(path),
        "version": version,
        "kv": kv,
        "tensors": tensors,
        "data_start": data_start,
    }


def iter_fp8_tensors(info: dict) -> Iterator[dict]:
    for t in info["tensors"]:
        if t["type"] in (GGML_TYPE_F8E4M3, GGML_TYPE_F8E5M2, GGML_TYPE_F16, GGML_TYPE_Q8_0):
            yield t


def read_tensor_bytes(path: str, info: dict, tensor: dict) -> bytes:
    """Copy one tensor into process RAM (no mmap)."""
    start = info["data_start"] + tensor["offset"]
    dims = tensor["dims"]
    n = 1
    for d in dims:
        n *= d
    typ = tensor["type"]
    if typ in (GGML_TYPE_F8E4M3, GGML_TYPE_F8E5M2):
        nbytes = n
    elif typ == GGML_TYPE_F16:
        nbytes = n * 2
    elif typ == GGML_TYPE_Q8_0:
        # super-blocks of 34 bytes (2 scale + 32 qs) typical
        block = 34
        nbytes = ((n + 31) // 32) * block
    else:
        raise GgufError(f"unsupported tensor type {typ}")
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(nbytes)
