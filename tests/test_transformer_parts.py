# tests/test_transformer_parts.py
"""Unit tests for the advanced transformer components."""

import torch

from model.config import GPTConfig
from model.ffn import SwiGLUFFN
from model.norm import RMSNorm
from model.rope import apply_rotary_emb, precompute_freqs_cis


def test_rmsnorm():
    """Verifies that RMSNorm correctly scales data without crashing."""
    dim = 768
    x = torch.randn(2, 10, dim) * 10.0  # Deliberately large numbers
    
    norm = RMSNorm(dim)
    out = norm(x)
    
    # Ensure shapes remain identical
    assert out.shape == x.shape, "RMSNorm changed the tensor shape!"
    print(f"\n✅ RMSNorm Output shape verified: {out.shape}")

def test_swiglu():
    """Verifies the SwiGLU FFN processes data correctly."""
    config = GPTConfig(d_model=16)
    ffn = SwiGLUFFN(config)
    
    x = torch.randn(2, 5, 16) # Batch=2, Seq=5, Dim=16
    out = ffn(x)
    
    # FFN should always return the exact same shape it received
    assert out.shape == x.shape, f"SwiGLU shape mismatch: {out.shape} != {x.shape}"
    print(f"✅ SwiGLU Output shape verified: {out.shape}")

def test_rope():
    """Verifies the complex geometric rotations execute successfully."""
    # Dummy Q and K tensors: Batch=1, SeqLen=4, Heads=2, HeadDim=8
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    
    # Precompute the frequencies for sequence length 4, head dimension 8
    freqs_cis = precompute_freqs_cis(dim=8, end=4)
    
    # Apply the rotation
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    
    # The rotations must not alter the tensor dimensions
    assert q_rot.shape == q.shape, "RoPE altered Query shape!"
    assert k_rot.shape == k.shape, "RoPE altered Key shape!"
    print(f"✅ RoPE rotation applied successfully. Output shape: {q_rot.shape}")
