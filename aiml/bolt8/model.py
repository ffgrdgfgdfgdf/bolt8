from __future__ import annotations

import math

import torch
from torch import nn

from bolt8.config import ModelConfig
from bolt8.kernels import RoPECache, apply_rope, expert_swiglu, rmsnorm, silu
from bolt8.store import PackedExpertStore


class KVCache:
    def __init__(self, cfg: ModelConfig, device: torch.device, dtype: torch.dtype):
        self.k = torch.zeros(
            cfg.n_layers, cfg.n_kv_heads, cfg.max_seq, cfg.head_dim,
            device=device, dtype=dtype,
        )
        self.v = torch.zeros_like(self.k)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0


class Bolt8Model(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        device: torch.device,
        allocate_experts: bool = True,
        allocate_cpu: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        dt = torch.float16
        self.dt = dt
        d, n_kv, hd = cfg.d_model, cfg.n_kv_heads, cfg.head_dim

        self.embed = nn.Embedding(cfg.vocab_size, d, device=device, dtype=dt)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False, device=device, dtype=dt)
        self.final_norm = nn.Parameter(torch.ones(d, device=device, dtype=dt))

        self.wq = nn.Parameter(torch.empty(cfg.n_layers, d, d, device=device, dtype=dt))
        self.wk = nn.Parameter(torch.empty(cfg.n_layers, n_kv * hd, d, device=device, dtype=dt))
        self.wv = nn.Parameter(torch.empty(cfg.n_layers, n_kv * hd, d, device=device, dtype=dt))
        self.wo = nn.Parameter(torch.empty(cfg.n_layers, d, d, device=device, dtype=dt))
        self.attn_n1 = nn.Parameter(torch.ones(cfg.n_layers, d, device=device, dtype=dt))
        self.ffn_n = nn.Parameter(torch.ones(cfg.n_layers, d, device=device, dtype=dt))
        self.router = nn.Parameter(
            torch.empty(cfg.n_layers, cfg.n_experts, d, device=device, dtype=dt)
        )
        stacked = 3 * cfg.d_ff
        self.shared = nn.Parameter(
            torch.empty(cfg.n_layers, stacked, d, device=device, dtype=dt)
        )

        with torch.no_grad():
            for p in (self.wq, self.wk, self.wv, self.wo, self.router, self.shared):
                p.normal_(0.0, 0.02)
            self.embed.weight.normal_(0.0, 0.02)
            self.lm_head.weight.normal_(0.0, 0.02)

        self.rope = RoPECache(hd, cfg.max_seq, cfg.rope_theta, device, dt)
        self.store = (
            PackedExpertStore(cfg, device, allocate_cpu=allocate_cpu)
            if allocate_experts
            else None
        )
        self.scale = 1.0 / math.sqrt(hd)
        self.gpu_experts_only = True

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg, self.device, self.dt)

    def _attn(self, layer: int, x: torch.Tensor, cache: KVCache) -> torch.Tensor:
        cfg = self.cfg
        b, d = x.shape
        h = rmsnorm(x, self.attn_n1[layer], cfg.rms_eps)
        q = torch.nn.functional.linear(h, self.wq[layer]).view(b, cfg.n_heads, cfg.head_dim)
        k = torch.nn.functional.linear(h, self.wk[layer]).view(b, cfg.n_kv_heads, cfg.head_dim)
        v = torch.nn.functional.linear(h, self.wv[layer]).view(b, cfg.n_kv_heads, cfg.head_dim)
        pos = cache.pos
        cos, sin = self.rope.at(pos)
        q, k = apply_rope(q, k, cos, sin)
        cache.k[layer, :, pos, :].copy_(k[0])
        cache.v[layer, :, pos, :].copy_(v[0])
        sl = pos + 1
        k_all = cache.k[layer, :, :sl, :]
        v_all = cache.v[layer, :, :sl, :]
        rep = cfg.n_heads // cfg.n_kv_heads
        if rep > 1:
            k_all = k_all.repeat_interleave(rep, dim=0)
            v_all = v_all.repeat_interleave(rep, dim=0)
        att = torch.einsum("bhd,hsd->bhs", q, k_all) * self.scale
        att = torch.softmax(att.float(), dim=-1).type_as(q)
        out = torch.einsum("bhs,hsd->bhd", att, v_all).reshape(b, d)
        return torch.nn.functional.linear(out, self.wo[layer])

    def _moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = rmsnorm(x, self.ffn_n[layer], cfg.rms_eps)
        logits = torch.nn.functional.linear(h, self.router[layer])
        # Mask experts that are not actually resident: GPU-only routing, or
        # full routing when the CPU arena was skipped / failed to allocate.
        if self.store is not None and (self.gpu_experts_only or self.store.cpu_packed is None):
            logits[:, self.store.n_gpu_per_layer :] = -float("inf")
        weights, idx = torch.topk(logits, cfg.top_k, dim=-1)
        weights = torch.softmax(weights.float(), dim=-1).type_as(h)
        gate_w, up_w, down_w = self.shared[layer].split(cfg.d_ff, dim=0)
        y = silu(torch.nn.functional.linear(h, gate_w)) * torch.nn.functional.linear(h, up_w)
        y = torch.nn.functional.linear(y, down_w.t())
        if self.store is None:
            return y
        for t in range(cfg.top_k):
            packed, scale, nbits = self.store.gather(
                layer, idx[0, t], hot_only=self.gpu_experts_only
            )
            ye = expert_swiglu(h, packed, scale, cfg.d_model, cfg.d_ff, nbits=nbits)
            y = y + ye * weights[:, t : t + 1]
        return y

    def forward_tokens(self, token_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        x = self.embed(token_ids)
        for layer in range(self.cfg.n_layers):
            x = x + self._attn(layer, x, cache)
            x = x + self._moe(layer, x)
        cache.pos += 1
        x = rmsnorm(x, self.final_norm, self.cfg.rms_eps)
        return self.lm_head(x)
