from __future__ import annotations

import torch

from bolt8.config import ModelConfig


def _fill_fp8(t: torch.Tensor, value: float) -> torch.Tensor:
    # torch.full()/fill_ are not reliable for float8 dtypes across versions;
    # a broadcast copy from a small fp16 -> fp8 scalar always works.
    t.copy_(torch.tensor([value], dtype=torch.float16).to(t.dtype))
    return t


class PackedExpertStore:
    """1-bit expert weights in VRAM + host RAM. Decode never touches disk."""

    def __init__(self, cfg: ModelConfig, device: torch.device, allocate_cpu: bool = True):
        self.cfg = cfg
        self.device = device
        if cfg.hot_bits not in (1, 2):
            raise ValueError("hot_bits must be 1 or 2")
        rows = 3 * cfg.d_ff
        cols = cfg.d_model
        packed_cols = (cols + 7) // 8  # 1-bit codes (cold experts)
        hot_packed_cols = (cols * cfg.hot_bits + 7) // 8  # hot experts
        n_blocks = (cols + cfg.scale_block - 1) // cfg.scale_block
        npl = max(cfg.top_k + 1, int(cfg.n_experts * cfg.vram_pack_fraction))
        npl = min(npl, cfg.n_experts)
        n_cpu = cfg.n_experts - npl
        self.rows = rows
        self.cols = cols
        self.packed_cols = packed_cols
        self.hot_packed_cols = hot_packed_cols
        self.n_blocks = n_blocks
        self.hot_bits = cfg.hot_bits
        self.n_gpu_per_layer = npl
        self.n_cpu_per_layer = n_cpu if allocate_cpu else 0

        self.gpu_packed = torch.empty(
            (cfg.n_layers, npl, rows, hot_packed_cols), device=device, dtype=torch.uint8
        )
        self.gpu_scale = _fill_fp8(
            torch.empty(
                (cfg.n_layers, npl, rows, n_blocks), device=device, dtype=torch.float8_e4m3fn
            ),
            0.02,
        )
        self.cpu_packed = None
        self.cpu_scale = None
        if allocate_cpu and n_cpu:
            try:
                self.cpu_packed = torch.empty(
                    (cfg.n_layers, n_cpu, rows, packed_cols), dtype=torch.uint8
                )
                self.cpu_scale = _fill_fp8(
                    torch.empty(
                        (cfg.n_layers, n_cpu, rows, n_blocks), dtype=torch.float8_e4m3fn
                    ),
                    0.02,
                )
            except (RuntimeError, MemoryError):
                self.cpu_packed = None
                self.cpu_scale = None
                self.n_cpu_per_layer = 0
        self._stage = torch.empty((rows, packed_cols), device=device, dtype=torch.uint8)
        self._stage_scale = torch.empty(
            (rows, n_blocks), device=device, dtype=torch.float8_e4m3fn
        )

    def bytes_resident(self) -> tuple[int, int]:
        # fp8 scales are 1 byte each; packed bits are 1 byte per 8 weights
        gpu = self.gpu_packed.numel() + self.gpu_scale.numel()
        cpu = 0 if self.cpu_packed is None else (
            self.cpu_packed.numel() + self.cpu_scale.numel()
        )
        return int(gpu), int(cpu)

    def gather_hot(self, layer: int, expert_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """GPU-only gather (hot experts, hot_bits wide). No CPU sync."""
        return self.gpu_packed[layer, expert_idx], self.gpu_scale[layer, expert_idx], self.hot_bits

    def gather(
        self, layer: int, expert_idx: torch.Tensor, hot_only: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (packed, scale, nbits): hot experts are hot_bits wide,
        staged cold experts are 1-bit."""
        if hot_only or self.cpu_packed is None:
            return self.gather_hot(layer, expert_idx)
        e = int(expert_idx.item())
        if e < self.n_gpu_per_layer:
            return self.gpu_packed[layer, e], self.gpu_scale[layer, e], self.hot_bits
        self._stage.copy_(self.cpu_packed[layer, e - self.n_gpu_per_layer], non_blocking=True)
        self._stage_scale.copy_(self.cpu_scale[layer, e - self.n_gpu_per_layer], non_blocking=True)
        return self._stage, self._stage_scale, 1

    def load_expert_packed(
        self,
        layer: int,
        expert: int,
        packed: torch.Tensor,
        scale: torch.Tensor,
        nbits: int = 1,
    ) -> None:
        scale = scale.view(self.rows, self.n_blocks).to(torch.float8_e4m3fn)
        if expert < self.n_gpu_per_layer:
            # hot slot: hot_bits-wide packed codes
            if nbits != self.hot_bits:
                raise ValueError(f"hot expert slots need nbits={self.hot_bits}, got {nbits}")
            packed = packed.view(self.rows, self.hot_packed_cols).to(torch.uint8)
            self.gpu_packed[layer, expert].copy_(packed.to(self.device, non_blocking=True))
            self.gpu_scale[layer, expert].copy_(scale.to(self.device, non_blocking=True))
        else:
            # cold slot: 1-bit signs
            if nbits != 1:
                raise ValueError("cold expert slots need nbits=1")
            packed = packed.view(self.rows, self.packed_cols).to(torch.uint8)
            if self.cpu_packed is not None:
                self.cpu_packed[layer, expert - self.n_gpu_per_layer].copy_(packed.cpu())
                self.cpu_scale[layer, expert - self.n_gpu_per_layer].copy_(scale.cpu())
