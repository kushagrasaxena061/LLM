# model/ffn.py
"""SwiGLU Feed-Forward Network."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig



class SwiGLUFFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # In a standard FFN, hidden dimension is 4 * d_model.
        # Because SwiGLU uses an extra gating layer, we historically use 
        # a slightly different multiplier (e.g., 8/3) to keep parameter counts equal.
        # For simplicity in this project, we will use an expansion factor of 4.
        hidden_dim = config.d_model * 4
        
        # The 'gate' projection (creates the values used to gate the information)
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        # The 'up' projection (creates the actual information)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=False)
        # The 'down' projection (compresses it back to d_model size)
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The SwiGLU formula: W3( SiLU(W1(x)) * W2(x) )
        """
        # 1. Calculate the gate and apply the SiLU (Swish) activation function
        gate = F.silu(self.w1(x))
        
        # 2. Calculate the information payload
        info = self.w2(x)
        
        # 3. Multiply them together (element-wise gating)
        gated_hidden = gate * info
        
        # 4. Project back to the model dimension
        return self.w3(gated_hidden)
