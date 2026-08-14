# Model Card: Custom Decoder-Only Transformer (151M Base)

## Model Overview
* **Architecture:** Decoder-only Transformer (GPT-style) implemented from scratch in PyTorch.
* **Parameters:** ~151.8M ($d_{model} = 768$, $n_{layers} = 12$, $n_{heads} = 12$).
* **Tokenization:** Custom Byte-Pair Encoding (BPE) tokenizer.
* **Key Enhancements:** Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activation functions, and LoRA instruction fine-tuning.

## Intended Use
* **Primary Use Cases:** Research, educational demonstration of transformer mechanics, lightweight domain-specific text generation, and Retrieval-Augmented Generation (RAG) prototyping.
* **Out-of-Scope:** High-stakes decision making, medical diagnosis, financial advisory, or autonomous safety-critical deployment without strict third-party verification.

## Training Data & Methodology
* **Pretraining Corpus:** Tokenized subsets of open-source datasets (e.g., FineWeb / Shakespeare corpus).
* **Fine-Tuning:** Supervised Fine-Tuning (SFT) using Parameter-Efficient Fine-Tuning (PEFT) via custom Low-Rank Adaptation (LoRA) matrices (Rank $r=4$, $\alpha=16$).

## Evaluation & Metrics
* **Validation Loss:** ~11.15
* **Perplexity:** ~69,753 (on initial educational training runs)
* **Inference Speedup:** Optimized via KV-Cache and dynamic Post-Training INT8 Quantization (2.65x memory reduction).

## Limitations & Biases
* The model has a limited context window (1024 tokens) and a compact parameter scale (151M), meaning it may exhibit hallucination, repetition loops, or lack advanced world knowledge compared to billion-scale foundation models.
* Guardrails include heuristic prompt injection detection and PII scrubbing middleware, but adversarial robustness remains bounded by local regex and pattern matching.
