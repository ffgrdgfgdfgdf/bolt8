from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """200B-class MoE shaped to fit 16GB VRAM + 32GB RAM when packed.

    Dense FP8 200B is ~200GB and cannot reside in 48GB total memory.
    Bolt8 stores experts as packed low-bit tensors split across VRAM
    (2-bit hot, default-routed) and pinned RAM (1-bit cold; never mmap /
    disk during decode). Only top-k experts are dequantized to FP8/FP16
    for compute. Active parameter traffic is what sets tokens/s, not the
    200B nominal size.
    """

    name: str = "bolt8-200b"
    vocab_size: int = 32000
    n_layers: int = 48
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 1280
    n_experts: int = 264
    n_shared_experts: int = 1
    top_k: int = 2
    rope_theta: float = 100000.0
    rms_eps: float = 1e-5
    max_seq: int = 4096
    # Packed experts (RAM/VRAM resident). Shared + attn stay FP16 on GPU.
    expert_bits: int = 1
    # Bits per weight for HOT (GPU-resident) experts: default routing only
    # ever picks these, so the whole active expert path runs at this
    # precision. 2 -> codes {+-0.5, +-1.5} x block scale (~2.06 bpw).
    # Cold (RAM, --full-route only) experts stay expert_bits=1.
    hot_bits: int = 2
    # Input weights per expert scale. Block-wise scales track local weight
    # magnitude and cut quantization error a lot for a few % more bytes.
    scale_block: int = 128
    compute_dtype: str = "float16"
    # Fraction of experts per layer to keep in VRAM (hot, hot_bits wide).
    # 0.15 x 264 = 39 experts/layer at 2 bits ~ 7.6GB VRAM on the 200B profile.
    vram_pack_fraction: float = 0.15

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def expert_params(self) -> int:
        # SwiGLU: gate, up, down
        return 3 * self.d_model * self.d_ff

    @property
    def total_experts(self) -> int:
        return self.n_layers * self.n_experts

    @property
    def moe_params(self) -> int:
        return self.n_layers * self.n_experts * self.expert_params

    @property
    def attn_params(self) -> int:
        # q, kv (GQA), o
        kv = 2 * self.n_kv_heads * self.head_dim * self.d_model
        qo = 2 * self.d_model * self.d_model
        return self.n_layers * (qo + kv)

    @property
    def shared_ffn_params(self) -> int:
        return self.n_layers * self.n_shared_experts * self.expert_params

    @property
    def other_params(self) -> int:
        embed = self.vocab_size * self.d_model
        norms = self.n_layers * 2 * self.d_model + self.d_model
        router = self.n_layers * self.d_model * self.n_experts
        lm = self.vocab_size * self.d_model
        return embed + norms + router + lm

    @property
    def total_params(self) -> int:
        return self.moe_params + self.attn_params + self.shared_ffn_params + self.other_params

    @property
    def active_params(self) -> int:
        per_layer = (
            (self.top_k + self.n_shared_experts) * self.expert_params
            + (2 * self.d_model * self.d_model + 2 * self.n_kv_heads * self.head_dim * self.d_model)
            + self.d_model * self.n_experts
        )
        return self.n_layers * per_layer + 2 * self.vocab_size * self.d_model

    @property
    def packed_expert_bytes(self) -> int:
        # mixed precision: hot experts at hot_bits, cold at expert_bits
        n_blocks = (self.d_model + self.scale_block - 1) // self.scale_block
        rows = 3 * self.d_ff
        scale_bytes = rows * n_blocks
        npl = min(
            max(self.top_k + 1, int(self.n_experts * self.vram_pack_fraction)),
            self.n_experts,
        )
        per_hot = rows * ((self.d_model * self.hot_bits + 7) // 8) + scale_bytes
        per_cold = rows * ((self.d_model * self.expert_bits + 7) // 8) + scale_bytes
        hot = self.n_layers * npl * per_hot
        cold = self.n_layers * (self.n_experts - npl) * per_cold
        return hot + cold

    def summary(self) -> str:
        tp = self.total_params / 1e9
        ap = self.active_params / 1e9
        pb = self.packed_expert_bytes / 1e9
        return (
            f"{self.name}: {tp:.1f}B total, {ap:.2f}B active/token, "
            f"{pb:.2f}GB packed experts ({self.hot_bits}b hot / {self.expert_bits}b cold), "
            f"{self.n_experts} experts x {self.n_layers} layers"
        )


PROFILES: dict[str, ModelConfig] = {
    "200b": ModelConfig(name="bolt8-200b"),
    "debug": ModelConfig(
        name="bolt8-debug",
        vocab_size=256,
        n_layers=4,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        d_ff=256,
        n_experts=8,
        top_k=2,
        max_seq=256,
        vram_pack_fraction=0.5,
    ),
    "speed": ModelConfig(
        name="bolt8-speed",
        vocab_size=32000,
        n_layers=48,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        d_ff=1280,
        n_experts=264,
        top_k=2,
        vram_pack_fraction=0.15,
    ),
}


def params_match_200b(cfg: ModelConfig) -> bool:
    return cfg.total_params >= 180_000_000_000
