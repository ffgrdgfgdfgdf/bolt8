from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch

from bolt8.config import ModelConfig, PROFILES
from bolt8.model import Bolt8Model, KVCache


@dataclass
class BenchResult:
    tokens: int
    seconds: float
    tps: float
    total_params_b: float
    active_params_b: float
    packed_gpu_gb: float
    packed_cpu_gb: float
    vram_gb: float

    def as_text(self) -> str:
        return (
            f"{self.tps:.2f} tok/s  ({self.tokens} tokens in {self.seconds:.2f}s)\n"
            f"model {self.total_params_b:.1f}B total / {self.active_params_b:.2f}B active\n"
            f"packed experts  GPU {self.packed_gpu_gb:.2f}GB  RAM {self.packed_cpu_gb:.2f}GB\n"
            f"torch VRAM allocated {self.vram_gb:.2f}GB"
        )


class Bolt8Engine:
    def __init__(
        self,
        cfg: ModelConfig | str = "200b",
        device: str | None = None,
        allocate_experts: bool = True,
        gpu_experts_only: bool = True,
        allocate_cpu: bool = True,
    ):
        if isinstance(cfg, str):
            if cfg not in PROFILES:
                raise KeyError(f"unknown profile {cfg}; choose from {list(PROFILES)}")
            cfg = PROFILES[cfg]
        # Copy so VRAM-fraction retries in _build never mutate shared PROFILES.
        self.cfg = replace(cfg)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("Bolt8 needs a CUDA/ROCm GPU")
        self._allocate_cpu = allocate_cpu
        self.model = self._build(allocate_experts)
        self.model.gpu_experts_only = gpu_experts_only
        self.model.eval()

    def _build(self, allocate_experts: bool) -> Bolt8Model:
        last_err: Exception | None = None
        fractions = [self.cfg.vram_pack_fraction, 0.12, 0.10, 0.08, 0.05]
        seen = []
        for frac in fractions:
            if frac in seen:
                continue
            seen.append(frac)
            self.cfg.vram_pack_fraction = frac
            try:
                torch.cuda.empty_cache()
                return Bolt8Model(
                    self.cfg,
                    self.device,
                    allocate_experts=allocate_experts,
                    allocate_cpu=self._allocate_cpu,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                last_err = e
                if "out of memory" not in str(e).lower() and "OOM" not in str(e):
                    raise
                torch.cuda.empty_cache()
        raise RuntimeError(f"could not fit model in VRAM: {last_err}")

    @torch.inference_mode()
    def generate_ids(
        self,
        prompt_ids: list[int],
        max_new: int = 64,
        temperature: float = 0.0,
    ) -> list[int]:
        cache = self.model.new_cache()
        ids = list(prompt_ids)
        x = torch.tensor([ids[0]], device=self.device, dtype=torch.long)
        for i, tok in enumerate(ids):
            x[0] = tok
            logits = self.model.forward_tokens(x, cache)
        out = []
        for _ in range(max_new):
            if cache.pos >= self.cfg.max_seq - 1:
                break
            if temperature and temperature > 0:
                probs = torch.softmax(logits[0].float() / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = torch.argmax(logits[0], dim=-1, keepdim=True)
            token = int(nxt.item())
            out.append(token)
            ids.append(token)
            x[0] = token
            logits = self.model.forward_tokens(x, cache)
        return out

    @torch.inference_mode()
    def bench(self, tokens: int = 64, warmup: int = 8) -> BenchResult:
        cache = self.model.new_cache()
        x = torch.randint(0, min(256, self.cfg.vocab_size), (1,), device=self.device)
        for _ in range(warmup):
            if cache.pos >= self.cfg.max_seq - 2:
                cache.reset()
            self.model.forward_tokens(x, cache)
        torch.cuda.synchronize()
        cache.reset()
        t0 = time.perf_counter()
        for _ in range(tokens):
            self.model.forward_tokens(x, cache)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gpu_b, cpu_b = (0, 0)
        if self.model.store is not None:
            gpu_b, cpu_b = self.model.store.bytes_resident()
        return BenchResult(
            tokens=tokens,
            seconds=dt,
            tps=tokens / dt if dt > 0 else 0.0,
            total_params_b=self.cfg.total_params / 1e9,
            active_params_b=self.cfg.active_params / 1e9,
            packed_gpu_gb=gpu_b / 1e9,
            packed_cpu_gb=cpu_b / 1e9,
            vram_gb=torch.cuda.memory_allocated() / 1e9,
        )


def encode_bytes(text: str, vocab: int) -> list[int]:
    ids = [1]
    ids.extend(min(b, vocab - 1) for b in text.encode("utf-8", errors="replace")[:128])
    return ids or [1]


def decode_bytes(ids: list[int]) -> str:
    raw = bytes(i & 255 for i in ids)
    return raw.decode("utf-8", errors="replace")
