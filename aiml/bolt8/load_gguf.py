from __future__ import annotations

import re

import numpy as np
import torch

from bolt8.gguf import (
    GGML_TYPE_F16,
    GGML_TYPE_F8E4M3,
    GGML_TYPE_F8E5M2,
    GGML_TYPE_Q8_0,
    parse_gguf_header,
    read_tensor_bytes,
)
from bolt8.gptq import compute_hessian, gptq_pack_blockwise
from bolt8.model import Bolt8Model
from bolt8.pack import pack_1bit_blockwise, pack_2bit_blockwise


_EXPERT = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight")
_MAT_SLOT = {"gate": 0, "up": 1, "down": 2}


def dequant_gguf_weight(raw: bytes, typ: int, out_f: int, in_f: int) -> torch.Tensor:
    """Dequantize one 2D GGUF tensor to fp16 [out_f, in_f] (GGUF dims are reversed).

    F16 and FP8 are exact; Q8_0 uses the real 34-byte block layout (fp16 block
    scale + 32 int8 quants per 32 input weights).
    """
    n = out_f * in_f
    if typ == GGML_TYPE_F16:
        if len(raw) != n * 2:
            raise ValueError("F16 buffer size mismatch")
        return torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().view(out_f, in_f)
    if typ in (GGML_TYPE_F8E4M3, GGML_TYPE_F8E5M2):
        if len(raw) != n:
            raise ValueError("FP8 buffer size mismatch")
        dt = torch.float8_e4m3fn if typ == GGML_TYPE_F8E4M3 else torch.float8_e5m2
        u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        return u8.view(dt).to(torch.float16).view(out_f, in_f)
    if typ == GGML_TYPE_Q8_0:
        if len(raw) % 34 or in_f % 32:
            raise ValueError("unsupported Q8_0 layout")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
        scales = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)  # [n_blocks]
        qs = arr[:, 2:].copy().view(np.int8).astype(np.float32)  # [n_blocks, 32]
        flat = (qs * scales[:, None]).reshape(-1).astype(np.float16)
        return torch.from_numpy(flat.reshape(out_f, in_f))
    raise ValueError(f"unsupported tensor type {typ}")


def _layer_hessians(calib, cfg) -> dict[int, torch.Tensor]:
    """calib: None | Tensor [n, d_model] shared by all layers | {layer: Tensor}."""
    if calib is None:
        return {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(calib, torch.Tensor):
        H = compute_hessian(calib).to(dev)
        return {layer: H for layer in range(cfg.n_layers)}
    return {int(layer): compute_hessian(x).to(dev) for layer, x in calib.items()}


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def load_gguf_into_model(
    model: Bolt8Model,
    path: str,
    calib: torch.Tensor | dict | None = None,
    gptq_chunk: int = 32,
) -> int:
    """Copy GGUF expert tensors into the RAM/VRAM arenas (read(), not mmap).

    Assembles all three matrices (gate, up, down) of each expert and writes each
    expert slot exactly once. Hot experts (GPU-resident, default routing) are
    packed at cfg.hot_bits; cold experts at 1 bit. With `calib` (expert-input
    activations: [n, d_model] shared by all layers, or {layer: [n, d_model]}),
    gate and up matrices are packed with GPTQ-style Hessian error compensation --
    identical packed format, so decode speed and memory are unchanged. `down`
    keeps plain packing: its stored layout scales along the output axis, where
    the objective is separable and input-axis compensation does not apply.
    Returns expert matrices stored.
    """
    info = parse_gguf_header(path)
    if model.store is None:
        return 0
    cfg = model.cfg
    d_ff, d_model = cfg.d_ff, cfg.d_model
    store = model.store
    # pass 1: index expert tensors by (layer, expert, kind); metadata only
    found: dict[tuple[int, int, str], dict] = {}
    for tensor in info["tensors"]:
        m = _EXPERT.match(tensor["name"])
        if not m:
            continue
        layer, kind, expert = int(m.group(1)), m.group(2), int(m.group(3))
        if layer >= cfg.n_layers or expert >= cfg.n_experts:
            continue
        dims = tensor["dims"]
        if len(dims) != 2:
            continue
        out_f, in_f = int(dims[1]), int(dims[0])
        if kind in ("gate", "up"):
            if (out_f, in_f) != (d_ff, d_model):
                continue
        elif (out_f, in_f) != (d_model, d_ff):  # down is stored [d_model, d_ff]
            continue
        found[(layer, expert, kind)] = tensor
    # pass 2a: GPTQ-pack gate/up (all experts of a layer stacked per call;
    # rows are independent given the shared Hessian, so chunking is exact)
    hess = _layer_hessians(calib, cfg)
    npl = store.n_gpu_per_layer
    parts: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    by_group: dict[tuple[int, str], list[int]] = {}
    for layer, expert, kind in found:
        by_group.setdefault((layer, kind), []).append(expert)
    for (layer, kind), experts in by_group.items():
        if kind not in ("gate", "up") or layer not in hess:
            continue
        if store.hot_bits == 1:
            subsets = [(1, sorted(experts))]
        else:
            subsets = [
                (2, sorted(e for e in experts if e < npl)),
                (1, sorted(e for e in experts if e >= npl)),
            ]
        for nbits, subset in subsets:
            for chunk in _chunks(subset, gptq_chunk):
                stack = []
                for e in chunk:
                    t = found[(layer, e, kind)]
                    w = dequant_gguf_weight(
                        read_tensor_bytes(path, info, t),
                        t["type"],
                        int(t["dims"][1]),
                        int(t["dims"][0]),
                    )
                    stack.append(w.float().to(hess[layer].device))
                W = torch.cat(stack, dim=0)
                packed, scales = gptq_pack_blockwise(
                    W, hess[layer], cfg.scale_block, nbits=nbits
                )
                for i, e in enumerate(chunk):
                    parts[(layer, e, kind)] = (
                        packed[i * d_ff : (i + 1) * d_ff],
                        scales[i * d_ff : (i + 1) * d_ff],
                    )
    # pass 2b: plain-pack everything not already GPTQ-packed
    for key, tensor in found.items():
        if key in parts:
            continue
        layer, expert, kind = key
        out_f, in_f = int(tensor["dims"][1]), int(tensor["dims"][0])
        raw = read_tensor_bytes(path, info, tensor)
        w = dequant_gguf_weight(raw, tensor["type"], out_f, in_f)
        if kind == "down":
            w = w.t().contiguous()  # -> [d_ff, d_model] to match the store layout
        hot = expert < store.n_gpu_per_layer and store.hot_bits == 2
        packer = pack_2bit_blockwise if hot else pack_1bit_blockwise
        parts[key] = packer(w, cfg.scale_block)
    # pass 2c: assemble each expert slot in one write
    stored = 0
    slots: dict[tuple[int, int], list[str]] = {}
    for layer, expert, kind in found:
        slots.setdefault((layer, expert), []).append(kind)
    for (layer, expert), kinds in slots.items():
        hot = expert < store.n_gpu_per_layer
        slot_cols = store.hot_packed_cols if hot else store.packed_cols
        full_p = torch.zeros(store.rows, slot_cols, dtype=torch.uint8)
        full_s = torch.zeros(store.rows, store.n_blocks, dtype=torch.float16)
        nbits = store.hot_bits if hot else 1
        for kind in kinds:
            packed, scale = parts[(layer, expert, kind)]
            r0 = _MAT_SLOT[kind] * d_ff
            full_p[r0 : r0 + d_ff].copy_(packed)
            full_s[r0 : r0 + d_ff].copy_(scale)
        store.load_expert_packed(layer, expert, full_p, full_s, nbits=nbits)
        stored += len(kinds)
    return stored
