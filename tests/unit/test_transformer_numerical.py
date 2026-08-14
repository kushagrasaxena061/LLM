# tests/unit/test_transformer_numerical.py
"""Numerical correctness and mathematical invariant tests for Transformer modules."""

import torch
import torch.nn as nn
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from model.norm import RMSNorm

def test_causal_attention_masking_invariant():
    """
    CRITICAL NUMERICAL TEST:
    Proves that token at position i CANNOT attend to token at position i + k.
    If we change future tokens, the logits at all previous positions must remain bitwise identical.
    """
    config = GPTConfig(vocab_size=100, context_length=16, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    # Input sequence A: [10, 20, 30, 40, 50]
    seq_a = torch.tensor([[10, 20, 30, 40, 50]], device=env_config.device)
    
    # Input sequence B: [10, 20, 30, 99, 88] (positions 0, 1, 2 are identical, positions 3, 4 are modified)
    seq_b = torch.tensor([[10, 20, 30, 99, 88]], device=env_config.device)
    
    with torch.no_grad():
        # FIX: Pass dummy targets to bypass the [-1] inference mode optimization slice
        logits_a, _, _ = model(seq_a, targets=seq_a)
        logits_b, _, _ = model(seq_b, targets=seq_b)
        
    # Logits at positions 0, 1, 2 MUST be bitwise identical
    diff_pos0_2 = (logits_a[:, :3, :] - logits_b[:, :3, :]).abs().max().item()
    assert diff_pos0_2 < 1e-5, f"Causal mask failure! Modifying future tokens altered past logits by {diff_pos0_2}"
    
    # Logits at positions 3, 4 MUST be different
    diff_pos3_4 = (logits_a[:, 3:, :] - logits_b[:, 3:, :]).abs().max().item()
    assert diff_pos3_4 > 1e-3, "Perturbing positions 3 & 4 failed to change output logits at positions 3 & 4"


def test_rmsnorm_formula_correctness():
    """Verifies custom RMSNorm against explicit mathematical formula."""
    dim = 64
    # FIX: Pass dimension positionally
    norm = RMSNorm(dim, eps=1e-6)
    x = torch.randn(2, 8, dim)
    
    out = norm(x)
    
    # Mathematical reference: x * rsqrt(mean(x^2) + eps) * weight
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
    expected = (x / rms) * norm.weight
    
    assert torch.allclose(out, expected, atol=1e-5), "RMSNorm output diverged from mathematical ground truth"


def test_transformer_gradient_flow():
    """Verifies that gradients propagate through all layers without exploding or vanishing to NaN."""
    config = GPTConfig(vocab_size=100, context_length=16, d_model=32, n_layers=4, n_heads=4)
    model = GPT(config).to(env_config.device)
    model.train()
    
    x = torch.randint(0, 100, (2, 8), device=env_config.device)
    y = torch.randint(0, 100, (2, 8), device=env_config.device)
    
    logits, loss, _ = model(x, targets=y)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} gradient contains Inf!"
