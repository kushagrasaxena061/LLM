# model/attention.py
"""Multi-Head Causal Self-Attention mechanism."""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from model.config import GPTConfig
from model.rope import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        
        mask = torch.tril(torch.ones(config.context_length, config.context_length)).view(
            1, 1, config.context_length, config.context_length
        )
        self.register_buffer("bias", mask)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # 1. Reshape but DO NOT transpose yet! 
        # Layout remains (Batch, Time, Heads, HeadDim) so RoPE math works.
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        # 2. Apply RoPE geometry
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])
        
        # 3. NOW transpose for the Attention matrix multiplication
        # Layout becomes (Batch, Heads, Time, HeadDim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(y)
