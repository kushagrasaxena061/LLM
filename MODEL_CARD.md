# MODEL_CARD.md: MiniGPT-151M

## Model Details
- **Architecture:** Custom Decoder-Only Transformer
- **Parameter Count:** 151,862,784 parameters
- **Hidden Dimension (d_model):** 768
- **Transformer Layers:** 12
- **Attention Heads:** 12
- **Positional Encoding:** Rotary Position Embeddings (RoPE)
- **Normalization:** Pre-RMSNorm
- **Activation Function:** SwiGLU
- **Precision:** FP32 Baseline, with Post-Training Dynamic INT8 Quantization (2.65x memory reduction)

## Intended Use
- Educational, research, and edge-device deployment exploration.
- Serving as a sandbox for Hybrid RAG, LoRA parameter-efficient fine-tuning, and adversarial security evaluation.

## Performance Characteristics
- **Time-To-First-Token (TTFT):** ~336 ms (Apple Silicon MPS / CPU)
- **Inter-Token Latency (ITL):** ~5.8 ms
- **Throughput:** ~35.8 tokens/sec (Autoregressive KV-Cached Decoding)
