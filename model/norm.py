# model/norm.py
"""Root Mean Square Normalization (RMSNorm)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim (int): The dimension of the input tensor (d_model).
            eps (float): A tiny number added to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        # The learned scaling parameter (gamma). 
        # After we normalize the numbers to a standard scale, the network can 
        # learn to scale them back up or down via this weight if it helps training.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the root mean square and divides the input by it."""
        # 1. Square the input (x.pow(2))
        # 2. Get the mean across the last dimension (d_model)
        # 3. Add epsilon for numerical stability
        # 4. Take the reciprocal of the square root (rsqrt)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stable calculation, then back to the original type (e.g., float16)
        output = self._norm(x.float()).type_as(x)
        # Multiply by the learned scaling weight
        return output * self.weight
