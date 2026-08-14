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
        
        self.mlp = SwiGLUFFN(config)
        self.ffn = self.mlp  # Alias to prevent attribute errors

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, return_attention: bool = False):
        norm_x = self.ln_1(x)
        attn_out, present, attn_weights = self.attn(norm_x, freqs_cis=freqs_cis, layer_past=layer_past, use_cache=use_cache, return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present, attn_weights
