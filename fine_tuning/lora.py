# fine_tuning/lora.py
"""Low-Rank Adaptation (LoRA) module for parameter-efficient fine-tuning."""

import math
import torch
import torch.nn as nn
from typing import Optional

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 16):
        """
        Wraps an existing linear layer with LoRA A and B matrices.
        
        Args:
            original_linear: The frozen base layer (e.g., Q or V projection).
            rank (int): The low rank dimension (r).
            alpha (float): Scaling hyperparameter.
        """
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 1. Freeze the original base weights
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False
            
        # 2. Initialize LoRA matrices A and B
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrix A is initialized with a normal distribution
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * (1.0 / math.sqrt(rank)))
        # Matrix B is initialized to zeros so the initial adaptation delta (BA) is zero
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes: W_0(x) + (B @ A(x)) * scaling
        """
        # Base model forward path (frozen weights)
        base_out = nn.functional.linear(x, self.weight, self.bias)
        
        # LoRA adaptation path (trainable low-rank matrices)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        
        return base_out + lora_out * self.scaling
