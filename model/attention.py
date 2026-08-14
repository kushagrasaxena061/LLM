import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import GPTConfig
from model.rope import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, return_attention: bool = False):
        B, T, C = x.size()
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.w_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.w_v(x).view(B, T, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            
        present = (k, v) if use_cache else None
        total_k_len = k.shape[2]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if T > 1:
            causal_mask = torch.tril(torch.ones(T, total_k_len, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        self.attention_weights = att  # Save for Explainability lab
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_out(y), present, (att if return_attention else None)
