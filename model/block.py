# model/block.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.config import GPTConfig
from model.attention import MultiHeadAttention
from model.ffn import SwiGLUFFN
from model.norm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        attn_out, present = self.attn(self.ln_1(x), freqs_cis, layer_past, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, present
