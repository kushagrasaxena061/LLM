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
        idx: torch.Tensor = None,
        targets: torch.Tensor = None,
        past_key_values: list = None,
        use_cache: bool = False,
        inputs_embeds: torch.Tensor = None,
        return_attention: bool = False,
        **kwargs
    ):
        # 1. Route between Text Tokens (idx) or Multimodal Embeddings (inputs_embeds)
        if inputs_embeds is not None:
            x = inputs_embeds
            b, t, _ = x.shape
        elif idx is not None:
            b, t = idx.size()
            x = self.tok_embeddings(idx)
        else:
            raise ValueError("Must provide either idx or inputs_embeds")

        x = self.dropout(x)

        # 2. Dynamic RoPE Positional Offset
        seq_offset = 0
        if past_key_values is not None:
            seq_offset = past_key_values[0][0].shape[2]

        if hasattr(self, 'freqs_cis'):
            freqs_cis = self.freqs_cis[seq_offset : seq_offset + t].to(x.device)
        else:
            freqs_cis = None

        presents = [] if use_cache else None
        all_attentions = [] if return_attention else None

        # 3. Indestructible Block Routing
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            
            block_ret = block(x, freqs_cis, layer_past=past, use_cache=use_cache, return_attention=return_attention)
            if return_attention:
                x, present_i, attn_w = block_ret
                if use_cache and present_i is not None:
                    presents.append(present_i)
                all_attentions.append(attn_w)
            elif isinstance(block_ret, tuple):
                x = block_ret[0]
                if use_cache:
                    presents.append(block_ret[1])
            else:
                x = block_ret

        # Dynamically map to the correct normalization layer (norm or ln_f)
        x = getattr(self, "norm", getattr(self, "ln_f", lambda x: x))(x)

        # 4. Loss & Logit Calculation
        if targets is not None:
            logits = self.lm_head(x)
            import torch.nn.functional as F
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if return_attention:
                return logits, loss, presents, all_attentions
            return logits, loss, presents
        else:
            logits = self.lm_head(x[:, [-1], :] if not return_attention and x.size(1) > 1 else x)
            if return_attention:
                return logits, None, presents, all_attentions
            return logits, None, presents
