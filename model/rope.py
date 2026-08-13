# model/rope.py
"""Rotary Positional Embeddings (RoPE)."""

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the cosine and sine frequencies for the rotations.
    We do this once during initialization to save massive compute during training.
    """
    # 1. Calculate the frequency intervals
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. Create the position indices (0, 1, 2, ..., end)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # 3. Calculate the outer product of positions and frequencies
    freqs = torch.outer(t, freqs)
    
    # 4. Convert to polar/complex coordinates (Euler's formula)
    # This creates a tensor of complex numbers where real=cos and imag=sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies the mathematical rotation to the Query and Key tensors.
    """
    # 1. Reshape Q and K to treat pairs of dimensions as complex numbers
    # Shape becomes: (Batch, Sequence, Heads, Head_Dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Ensure the frequencies match the input shape for broadcasting
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    
    # 3. Multiply the complex numbers (this performs the geometric rotation!)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # 4. Return to the original datatype
    return xq_out.type_as(xq), xk_out.type_as(xk)
