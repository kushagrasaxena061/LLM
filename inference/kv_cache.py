# inference/kv_cache.py
"""KV-Cache state management for efficient autoregressive generation."""

import torch
from typing import Optional

class KVCache:
    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates new key/value tensors with the cached historical tensors.
        
        Args:
            k_new: New key tensor of shape (Batch, Heads, 1, HeadDim)
            v_new: New value tensor of shape (Batch, Heads, 1, HeadDim)
        """
        if self.k_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            # Concatenate along the sequence dimension (dim=2)
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
            
        return self.k_cache, self.v_cache

    def reset(self):
        """Clears the cache between independent generation prompts."""
        self.k_cache = None
        self.v_cache = None
