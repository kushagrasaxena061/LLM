# MiniGPT-151M Model Card
## Architecture
- **Parameters:** 151,862,784
- **Hidden Dimension (d_model):** 768
- **Transformer Layers:** 12
- **Attention Heads:** 12 (64 dim/head)
- **Positional Encoding:** Rotary Position Embeddings (RoPE)
- **Normalization:** Pre-RMSNorm
- **Activation Function:** SwiGLU
- **Weight Tying:** Enabled

## Infrastructure Capabilities
- Custom BPE Tokenizer (Lossless UTF-8)
- $O(N)$ KV-Cache Autoregressive Generation
- Hybrid RAG (Dense + BM25 + Reciprocal Rank Fusion + CrossEncoder Reranking)
- LoRA Parameter-Efficient Fine-Tuning
- INT8 Post-Training Quantization
- Multimodal Vision-Language Projection Adapter
- OWASP Security Guardrails & PII Sanitization
