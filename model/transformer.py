# model/transformer.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, List
from model.config import GPTConfig
from model.norm import RMSNorm
from model.block import TransformerBlock
from model.rope import precompute_freqs_cis
from utils.logger import get_logger

logger = get_logger(__name__)

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight
        
        freqs_cis = precompute_freqs_cis(dim=config.head_dim, end=config.context_length)
        self.register_buffer("freqs_cis", freqs_cis)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        
        b, t = idx.size()
        
        # If we have past_key_values, we are generating the Nth token. 
        # We only pass the last generated token through the model.
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
            freqs_cis = self.freqs_cis[past_length : past_length + t]
        else:
            freqs_cis = self.freqs_cis[:t]

        x = self.tok_embeddings(idx)
        x = self.dropout(x)

        presents = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, freqs_cis, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, presents
