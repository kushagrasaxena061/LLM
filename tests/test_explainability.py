# tests/test_explainability.py
"""Unit tests for the Transformer Debugger and Attention extraction."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from explainability.visualizer import AttentionVisualizer

def test_attention_extraction():
    """Verifies that the attention matrix is correctly captured and dimensionally accurate."""
    config = GPTConfig(vocab_size=260, context_length=64, d_model=32, n_layers=2, n_heads=4)
    model = GPT(config).to(env_config.device)
    
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps.")
    
    visualizer = AttentionVisualizer(model, tokenizer, env_config.device)
    
    test_phrase = "The quick brown fox"
    tokens, attn_matrix = visualizer.extract_attention(test_phrase)
    
    seq_len = len(tokens)
    
    # Assert the internal tensor was captured successfully
    assert attn_matrix is not None, "Attention weights were not captured!"
    
    # Assert shape: (Batch=1, Heads=4, SeqLen, SeqLen)
    expected_shape = (1, config.n_heads, seq_len, seq_len)
    assert attn_matrix.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {attn_matrix.shape}"
    
    print(f"\n✅ Explainability Test Passed!")
    print(f"   - Extracted Tokens: {tokens}")
    print(f"   - Attention Matrix Captured: {list(attn_matrix.shape)}")
