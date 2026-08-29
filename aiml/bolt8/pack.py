from __future__ import annotations

import numpy as np
import torch


def pack_codes(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack a [rows, cols] uint8 code matrix (values < 2**nbits) LSB-first.

    nbits=1 -> 8 codes per byte ([rows, ceil(cols/8)]); nbits=2 -> 4 codes
    per byte ([rows, ceil(cols/4)]).
    """
    if nbits not in (1, 2):
        raise ValueError("nbits must be 1 or 2")
    per = 8 // nbits
    rows, cols = codes.shape
    if codes.dtype != torch.uint8:
        codes = codes.to(torch.uint8)
    codes = codes.contiguous()
    pad = (-cols) % per
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    codes = codes.view(rows, -1, per)
    shifts = torch.arange(per, device=codes.device, dtype=torch.uint8) * nbits
    return (codes << shifts).sum(dim=-1, dtype=torch.uint8).contiguous()


def pack_codes_scales(
    codes: torch.Tensor, scales: torch.Tensor, block: int, nbits: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn precomputed quantization codes + block scales into packed format.

    codes: [rows, cols] uint8 (nbits=1: 0/1 signs; nbits=2: 0..3 levels);
    scales: [rows, n_blocks] float with n_blocks = ceil(cols / block).
    Returns (packed uint8, fp8 scales) for PackedExpertStore slots of `nbits`.
    """
    rows, cols = codes.shape
    n_blocks = (cols + block - 1) // block
    if tuple(scales.shape) != (rows, n_blocks):
        raise ValueError(f"scales must have shape [{rows}, {n_blocks}]")
    packed = pack_codes(codes, nbits)
    return packed.cpu(), scales.clamp_min(0.002).to(torch.float8_e4m3fn).cpu()


def pack_1bit_blockwise(
    weight: torch.Tensor, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack sign(w) as bits with one FP8 E4M3 scale per block of `block` inputs.

    scale = mean(|w|) per block: the MSE-optimal scale for w ~= s * sign(w).
    Cost is 1 + 8/block bits per weight (block=128 -> ~1.06 bpw, ~+6% bytes).
    Returns (packed [rows, ceil(cols/8)] uint8, scales [rows, ceil(cols/block)] fp8).
    """
    w = weight.detach().float()
    if w.ndim != 2:
        raise ValueError("expected 2D weight")
    if block < 8:
        raise ValueError("block must be >= 8")
    rows, cols = w.shape
    n_blocks = (cols + block - 1) // block
    signs = (w >= 0).to(torch.uint8).contiguous()
    # exact per-block mean |w| (the last block can be partial)
    abs_pad = torch.nn.functional.pad(w.abs(), (0, (-cols) % block))
    sums = abs_pad.view(rows, n_blocks, -1).sum(dim=-1)
    counts = torch.full((n_blocks,), float(block))
    counts[-1] = float(cols - (n_blocks - 1) * block)
    scale = (sums / counts.to(w.device).unsqueeze(0)).clamp_min(0.002)
    return pack_codes_scales(signs, scale, block, 1)


_SHIFTS: dict[str, torch.Tensor] = {}


def unpack_1bit(packed: torch.Tensor, scale: torch.Tensor, cols: int) -> torch.Tensor:
    """GPU unpack: bits -> {-1, +1} * per-block scale.

    packed: [rows, packed_cols] uint8; scale: [rows, n_blocks] fp8 (or fp16),
    with n_blocks = ceil(cols / block); the block size is inferred from scale.
    """
    rows, packed_cols = packed.shape
    n_blocks = scale.shape[-1]
    key = str(packed.device)
    shifts = _SHIFTS.get(key)
    if shifts is None:
        shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
        _SHIFTS[key] = shifts
    expanded = packed.to(torch.int32).unsqueeze(-1)
    bits = (expanded >> shifts) & 1
    signs = bits.reshape(rows, packed_cols * 8)[:, :cols].to(dtype=torch.float16)
    signs.mul_(2).sub_(1)
    block = (cols + n_blocks - 1) // n_blocks
    s = scale.to(device=packed.device, dtype=torch.float16).unsqueeze(2)
    pad = n_blocks * block - cols
    if pad:
        signs = torch.nn.functional.pad(signs, (0, pad))
    out = signs.reshape(rows, n_blocks, block) * s
    if pad:
        return out.reshape(rows, n_blocks * block)[:, :cols]
    return out.reshape(rows, cols)


_SHIFTS2: dict[str, torch.Tensor] = {}


def unpack_2bit(packed: torch.Tensor, scale: torch.Tensor, cols: int) -> torch.Tensor:
    """GPU unpack: 2-bit codes -> {+-0.5, +-1.5} * per-block scale.

    packed: [rows, ceil(cols/4)] uint8; scale: [rows, n_blocks] fp8 (or fp16),
    with n_blocks = ceil(cols / block); the block size is inferred from scale.
    Codes stay uint8 until the final decode: {0,1,2,3} -> {-0.5,-1.5,+0.5,+1.5}
    via where(c >= 2, c - 1.5, -0.5 - c), then x block scale.
    """
    rows, packed_cols = packed.shape
    n_blocks = scale.shape[-1]
    key = str(packed.device)
    shifts = _SHIFTS2.get(key)
    if shifts is None:
        shifts = torch.arange(4, device=packed.device, dtype=torch.uint8) * 2
        _SHIFTS2[key] = shifts
    expanded = packed.unsqueeze(-1)  # uint8 view; avoids a 4x int32 blow-up
    codes = (expanded >> shifts) & 3  # uint8 [rows, packed_cols, 4]
    codes = codes.reshape(rows, packed_cols * 4)[:, :cols].to(torch.float16)
    signs = torch.where(codes >= 2, codes - 1.5, -0.5 - codes)
    block = (cols + n_blocks - 1) // n_blocks
    s = scale.to(device=packed.device, dtype=torch.float16).unsqueeze(2)
    pad = n_blocks * block - cols
    if pad:
        signs = torch.nn.functional.pad(signs, (0, pad))
    out = signs.reshape(rows, n_blocks, block) * s
    if pad:
        return out.reshape(rows, n_blocks * block)[:, :cols]
    return out.reshape(rows, cols)


def _levels2_scale(w_pad: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Near-LS-optimal 4-level block scale via one Lloyd step from mean|w|.

    w_pad: [rows, n_blocks, block]; counts: [1, n_blocks] real element counts.
    Returns scales [rows, n_blocks].
    """
    s0 = (w_pad.abs().sum(dim=-1) / counts).clamp_min(0.002)
    big = w_pad.abs() >= s0.unsqueeze(2)
    l0 = torch.where(big, 1.5, 0.5) * torch.where(w_pad >= 0, 1.0, -1.0)
    den = (l0 * l0).sum(dim=-1).clamp_min(1e-8)
    return ((l0 * w_pad).sum(dim=-1) / den).clamp(0.002, 448.0)


def pack_2bit_blockwise(
    weight: torch.Tensor, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack w as 2-bit codes {+-0.5, +-1.5} x one FP8 scale per input block.

    ~2.06 bits/weight: roughly 2-bit-uniform quality, a large step up from
    1-bit signs. Returns (packed [rows, ceil(cols/4)] uint8,
    scales [rows, n_blocks] fp8) -- the hot-expert slot format.
    """
    w = weight.detach().float()
    if w.ndim != 2:
        raise ValueError("expected 2D weight")
    if block < 8:
        raise ValueError("block must be >= 8")
    rows, cols = w.shape
    n_blocks = (cols + block - 1) // block
    w_pad = torch.nn.functional.pad(w, (0, (-cols) % block)).view(rows, n_blocks, -1)
    counts = torch.full((n_blocks,), float(block))
    counts[-1] = float(cols - (n_blocks - 1) * block)
    counts = counts.to(w.device).unsqueeze(0)
    s = _levels2_scale(w_pad, counts)
    codes = (((w_pad >= 0).to(torch.uint8) << 1)
             | (w_pad.abs() >= s.unsqueeze(2)).to(torch.uint8))
    codes = codes.reshape(rows, n_blocks * block)[:, :cols].contiguous()
    return pack_codes_scales(codes, s, block, 2)


def fill_packed_random(packed: torch.Tensor, generator: torch.Generator | None = None) -> None:
    packed.random_(0, 256, generator=generator)


def row_scales_ones(rows: int, device: torch.device) -> torch.Tensor:
    return torch.ones(rows, device=device, dtype=torch.float16)


def fp8_to_fp16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float8_e4m3fn:
        return x.to(torch.float16)
    return x.half() if x.dtype != torch.float16 else x


def pack_numpy_fp8_e4m3(data: bytes, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Interpret raw FP8 E4M3 bytes, pack to 1-bit signs + row scales."""
    rows, cols = shape
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size < rows * cols:
        raise ValueError("FP8 buffer too small")
    raw = raw[: rows * cols].reshape(rows, cols)
    # E4M3: sign in bit 7. Approximate abs via decode through torch if available.
    signs = (raw < 128).astype(np.uint8)
    pad = (-cols) % 8
    if pad:
        signs = np.pad(signs, ((0, 0), (0, pad)))
    packed = np.packbits(signs, axis=-1)
    # crude scale: use a constant; real GGUF path can pass fp16 scales
    scale = np.ones(rows, dtype=np.float16)
    return packed, scale
