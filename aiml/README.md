Bolt8 is a Python inference runtime for huge MoE checkpoints on one consumer GPU.

A dense 200B FP8 GGUF is ~200GB. This machine class has 16GB VRAM + 32GB RAM, so Bolt8 does not stream weights from disk. It:

1. Keeps MoE experts as packed tensors: 2-bit in VRAM (hot, default-routed), 1-bit in RAM (cold)
2. Routes top-k experts per token
3. Dequantizes only those experts to FP16/FP8-style GEMM on the GPU
4. Keeps attention + a shared expert in VRAM as FP16

That is the only way 200B-class models hit 5–20 tok/s here: **active** weights per token are a few billion parameters, not 200B.

## Expert quantization

Experts are stored at two precisions, both with one FP8 E4M3 scale per 128
consecutive input weights:

- **Hot experts** (GPU-resident; default routing only ever picks these) are
  **2-bit codes `{±0.5, ±1.5} × block scale`** (~2.06 bits/weight) — roughly
  2-bit-uniform quality, a large step up from 1-bit signs.
- **Cold experts** (RAM, only used with `--full-route`) are **1-bit signs ×
  block scale** (~1.06 bits/weight).

On the 200B profile: 39 hot experts/layer at 2-bit (~7.6GB VRAM) + 225 cold at
1-bit (~22.6GB RAM). Attention, shared experts, router, embeddings and the LM
head stay FP16.

The GGUF loader stores **all three** expert matrices (gate/up/down) per expert and
dequantizes F16 and FP8 exactly, plus real per-block Q8_0 (fp16 block scales).

## Calibration-aware packing (GPTQ)

`load_gguf_into_model(model, path, calib=...)` optionally takes calibration
activations — the hidden states that feed the MoE layers: `[n, d_model]` (shared
by all layers) or `{layer: [n, d_model]}`. With calib, gate/up matrices are packed
with GPTQ-style sequential error compensation (2-bit for hot experts, 1-bit for
cold): each 128-column group's quantization
error is pushed into the remaining columns, Hessian-weighted, minimizing layer
**output** error instead of weight error. Measured on structured synthetic data:
~40% lower output error, and ~82% lower on outlier-heavy activations (the regime
that dominates real LLM damage) — at **identical memory and identical decode
speed**, because the packed format is unchanged. All experts of a layer are
stacked into each GPTQ call (rows are independent given the shared Hessian), so
the sequential loop runs once per layer in 32-expert chunks: ~2.4s per chunk on
an RX 7800 XT, roughly half an hour to pack a full 200B checkpoint offline.
`down` keeps plain packing: its stored layout scales along the output axis, where
the objective is separable and input-axis compensation does not apply.

## Run

```text
python -m bolt8 info --profile 200b
python -m bolt8 bench --profile debug --tokens 16
python -m bolt8 bench --profile 200b --tokens 16
python -m bolt8 generate --profile debug --prompt Hello
python -m bolt8 info --gguf path\to\model.gguf
```

`--no-host-experts` skips the RAM-side packed arena if the OS cannot hold it.

`--full-route` also pulls cold experts from RAM (slower; more of the 200B participates).

## Hardware this was built against

AMD Radeon RX 7800 XT 16GB (ROCm PyTorch) + ~32GB system RAM. RDNA3 has no native FP8 tensor cores; compute is FP16 after unpack.
