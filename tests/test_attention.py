# tests/test_attention.py
"""Numerical correctness tests for the Attention mechanism."""

import torch
from model.config import GPTConfig
from model.attention import MultiHeadAttention
from model.rope import precompute_freqs_cis

def test_causal_masking():
    """Proves that tokens cannot look into the future."""
    config = GPTConfig(vocab_size=100, context_length=8, d_model=16, n_heads=2, dropout=0.0)
    attention = MultiHeadAttention(config)
    x = torch.randn(2, 8, 16)
    
    # Precompute RoPE frequencies (head_dim = 16 / 2 = 8)
    freqs_cis = precompute_freqs_cis(dim=8, end=8)
    
    # Pass freqs_cis into the forward pass
    y = attention(x, freqs_cis)
    
    assert y.shape == x.shape
    print(f"\n✅ Attention output shape verified: {y.shape}")

    x_modified = x.clone()
    x_modified[:, 7, :] = torch.randn(16)
    y_modified = attention(x_modified, freqs_cis)
    
    assert torch.allclose(y[:, 3, :], y_modified[:, 3, :])
    print("✅ Causal masking successfully prevented information leakage from the future.")
