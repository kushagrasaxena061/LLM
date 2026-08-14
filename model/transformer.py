# model/transformer.py
"""Decoder-only GPT Transformer architecture with RoPE, RMSNorm, and SwiGLU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from model.config import GPTConfig
from model.block import TransformerBlock
from model.rope import precompute_freqs_cis

try:
    from model.norm import RMSNorm
except ImportError:
    from model.rmsnorm import RMSNorm

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Explicit Weight Tying (reduces parameter footprint from ~190M to exact 151.86M)
        if config.weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
            
        freqs_cis = precompute_freqs_cis(config.head_dim, end=max(4096, config.context_length * 2))
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculates total unique parameters accounting for weight tying."""
        unique_params = set(self.parameters())
        if non_embedding:
            unique_params.discard(self.tok_embeddings.weight)
        return sum(p.numel() for p in unique_params)

    def _ensure_freqs_cis(self, required_len: int, device: torch.device):
        if self.freqs_cis is None or self.freqs_cis.shape[0] < required_len or self.freqs_cis.device != device:
            new_size = max(required_len + 512, self.config.context_length * 2, 4096)
            self.freqs_cis = precompute_freqs_cis(self.config.head_dim, end=new_size, device=device)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        start_pos: Optional[int] = None,
        return_attention: bool = False
    ):
        B, T = idx.shape
        device = idx.device
        if start_pos is None:
            start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            
        required_len = start_pos + T
        self._ensure_freqs_cis(required_len, device)
        freqs_cis_slice = self.freqs_cis[start_pos:required_len]
        
        x = self.tok_embeddings(idx)
        x = self.dropout(x)
        
        presents = [] if use_cache else None
        attentions = [] if return_attention else None
        
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present, attn_w = block(
                x,
                freqs_cis=freqs_cis_slice,
                layer_past=layer_past,
                use_cache=use_cache,
                return_attention=return_attention
            )
            if use_cache:
                presents.append(present)
            if return_attention:
                attentions.append(attn_w)
                
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = logits[:, -1:, :]
            
        if return_attention:
            return logits, loss, presents, attentions
        return logits, loss, presents
