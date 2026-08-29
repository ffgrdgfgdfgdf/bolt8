from __future__ import annotations

import torch

from bolt8.pack import pack_codes_scales


def compute_hessian(activations: torch.Tensor) -> torch.Tensor:
    """Second-moment matrix H = X^T X of calibration activations X [n, in].

    Pass the hidden states that feed the layer being packed. Only the relative
    structure matters; damping is applied inside the packer.
    """
    X = activations.detach().float()
    if X.ndim != 2:
        raise ValueError("activations must be [n_samples, in_features]")
    return X.t() @ X


def _quant_1bit(w: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    return torch.where(w >= 0, gs, -gs)


def _quant_2bit(w: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    mag = torch.where(w.abs() >= gs, 1.5, 0.5) * gs
    return torch.where(w >= 0, mag, -mag)


def _group_scale(Wb: torch.Tensor, nbits: int) -> torch.Tensor:
    """Per-row group scale for the current (compensated) block values."""
    if nbits == 1:
        # provably MSE-optimal for w ~= s * sign(w)
        return Wb.abs().mean(dim=1).clamp_min(0.002)
    s0 = Wb.abs().mean(dim=1).clamp_min(0.002)
    big = Wb.abs() >= s0.unsqueeze(1)
    l0 = torch.where(big, 1.5, 0.5) * torch.where(Wb >= 0, 1.0, -1.0)
    den = (l0 * l0).sum(dim=1).clamp_min(1e-8)
    return ((l0 * Wb).sum(dim=1) / den).clamp(0.002, 448.0)


def gptq_pack_blockwise(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    block: int = 128,
    damp: float = 0.01,
    nbits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ-style error-compensated packing (1-bit or 2-bit) with block scales.

    Sequentially quantizes `block`-column groups of `weight` [rows, cols]. After
    each column, its quantization error is pushed into the remaining columns,
    Hessian-weighted, so the total layer OUTPUT error
    trace((W - Q) H (W - Q)^T) is minimized instead of the raw weight error.

    nbits=1: sign x scale; nbits=2: {+-0.5, +-1.5} x scale (Lloyd-refined).
    All rows are independent given the shared Hessian, so callers can stack many
    weight matrices (e.g. every expert of a layer) into `weight` and pay the
    sequential column loop only once.

    Returns exactly what pack.pack_1bit_blockwise / pack_2bit_blockwise return
    for that nbits: identical memory, identical decode speed.
    """
    if nbits not in (1, 2):
        raise ValueError("nbits must be 1 or 2")
    W = weight.detach().float().clone()
    H = hessian.detach().float().clone().to(W.device)
    rows, cols = W.shape
    if tuple(H.shape) != (cols, cols):
        raise ValueError(f"hessian must be [{cols}, {cols}], got {tuple(H.shape)}")
    if block < 8:
        raise ValueError("block must be >= 8")
    # columns with no calibration signal: unit diagonal (GPTQ convention)
    dead = H.diagonal() <= 0
    if dead.any():
        H[dead, dead] = 1.0
    # relative damping keeps the Cholesky stable for near-singular H
    H = H + damp * H.diagonal().mean() * torch.eye(cols, dtype=H.dtype, device=H.device)
    # H -> H^-1 -> upper-triangular factor (GPTQ's Hinv)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    n_blocks = (cols + block - 1) // block
    codes = torch.empty((rows, cols), dtype=torch.uint8, device=W.device)
    scales = torch.empty((rows, n_blocks), dtype=torch.float32, device=W.device)
    quant = _quant_1bit if nbits == 1 else _quant_2bit
    for b in range(n_blocks):
        i1 = b * block
        i2 = min(i1 + block, cols)
        bw = i2 - i1
        Wb = W[:, i1:i2].clone()
        Hb = Hinv[i1:i2, i1:i2]
        gs = _group_scale(Wb, nbits)
        Qb = torch.empty_like(Wb)
        Err = torch.empty_like(Wb)
        for i in range(bw):
            w = Wb[:, i]
            q = quant(w, gs)
            Qb[:, i] = q
            if nbits == 1:
                codes[:, i1 + i] = (w >= 0).to(torch.uint8)
            else:
                codes[:, i1 + i] = ((w >= 0).to(torch.uint8) << 1) | (
                    w.abs() >= gs
                ).to(torch.uint8)
            err = (w - q) / Hb[i, i]
            if i + 1 < bw:
                Wb[:, i + 1 :] -= err.unsqueeze(1) * Hb[i, i + 1 :].unsqueeze(0)
            Err[:, i] = err
        scales[:, b] = gs
        if i2 < cols:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return pack_codes_scales(codes, scales, block, nbits)