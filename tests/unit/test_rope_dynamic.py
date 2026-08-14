# tests/unit/test_rope_dynamic.py
"""Unit tests verifying dynamic RoPE sequence expansion beyond default context."""

import torch
from model.config import GPTConfig
from model.transformer import GPT

def test_rope_dynamic_expansion_long_sequence():
    config = GPTConfig(vocab_size=100, context_length=64, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config)
    
    long_seq_len = 150
    idx = torch.randint(0, 100, (1, long_seq_len))
    
    # Forward pass on sequence longer than context_length
    logits, _, _ = model(idx, targets=idx)
    assert logits.shape == (1, long_seq_len, 100)
    assert model.freqs_cis.shape[0] >= long_seq_len
