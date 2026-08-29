from __future__ import annotations

import torch
from bolt8.pack import unpack_1bit, unpack_2bit


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (x32.to(orig) * weight).to(orig)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RoPECache:
    def __init__(self, head_dim: int, max_seq: int, theta: float, device, dtype):
        half = head_dim // 2
        freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(max_seq, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freq)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        self.cos = torch.cat([cos, cos], dim=-1)
        self.sin = torch.cat([sin, sin], dim=-1)

    def at(self, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[pos], self.sin[pos]


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def expert_swiglu(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    d_model: int,
    d_ff: int,
    nbits: int = 1,
) -> torch.Tensor:
    """x: [B, d_model], packed: [3*d_ff, packed_cols] of stacked [gate, up, down^T layout].

    Layout: W_all = [W_gate; W_up; W_down] each [d_ff, d_model] (down stored as [d_ff, d_model]
    and applied as x_ff @ W_down.T later by viewing down as [d_ff, d_model]).
    scale: [3*d_ff, n_blocks] fp8 block scales. nbits: 1 = cold experts (sign
    bits, packed_cols = ceil(d_model/8)); 2 = hot experts (4-level codes,
    packed_cols = ceil(d_model/4)).
    """
    unpack = unpack_2bit if nbits == 2 else unpack_1bit
    w = unpack(packed, scale, d_model)  # [3*d_ff, d_model]
    gate_w, up_w, down_w = w.split(d_ff, dim=0)
    # x @ W.T  ==  F.linear(x, W)
    gated = silu(torch.nn.functional.linear(x, gate_w)) * torch.nn.functional.linear(x, up_w)
    return torch.nn.functional.linear(gated, down_w.t().contiguous())


def expert_swiglu_fp16(x: torch.Tensor, stacked: torch.Tensor, d_ff: int) -> torch.Tensor:
    gate_w, up_w, down_w = stacked.split(d_ff, dim=0)
    gated = silu(torch.nn.functional.linear(x, gate_w)) * torch.nn.functional.linear(x, up_w)
    return torch.nn.functional.linear(gated, down_w.t().contiguous())
