# Architecture Decision Log (ADR)

| Decision Point | Chosen Approach | Alternative Considered | Justification / Trade-off |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Rotary Position Embedding (RoPE) | Absolute Learned / ALiBi | Better context extrapolation and relative position modeling for transformer self-attention. |
| **Normalization** | RMSNorm | LayerNorm | Computationally lighter by eliminating mean-centering calculations while preserving training stability. |
| **Activation Function** | SwiGLU | GELU / ReLU | Enhanced representation capacity and performance at the cost of slight compute overhead. |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation) | Full Fine-Tuning | Dramatically reduces VRAM consumption by freezing base weights and training only ~2% parameter matrices. |
| **Vector Retrieval** | Hybrid (Dense Vector + BM25) | Dense Vector Only | Combines semantic intent matching with exact keyword accuracy via Reciprocal Rank Fusion (RRF). |
