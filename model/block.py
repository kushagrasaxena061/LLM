# model/block.py
"""A single Transformer Decoder Block."""

import torch
import torch.nn as nn
from model.config import GPTConfig
from model.attention import MultiHeadAttention
from model.ffn import SwiGLUFFN
from model.norm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Pre-Norm layer for Attention
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)

        # Pre-Norm layer for the Feed-Forward Network
        self.ln_2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass utilizing Pre-Norm and Residual connections.
        """
        # 1. First residual block (Attention)
        # We normalize `x` BEFORE passing it to attention,
        # and then add the result back to the original un-normalized `x`.
        x = x + self.attn(self.ln_1(x), freqs_cis)

        # 2. Second residual block (Feed-Forward)
        x = x + self.ffn(self.ln_2(x))

        return x
