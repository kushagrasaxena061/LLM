# model/transformer.py
"""Full GPT Decoder-Only Language Model implementation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.block import TransformerBlock
from model.config import GPTConfig
from model.norm import RMSNorm
from model.rope import precompute_freqs_cis
from utils.logger import get_logger

logger = get_logger(__name__)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 1. Token Embedding Layer: Maps integer IDs to d_model vectors
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # 2. Stack of N Transformer Decoder Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # 3. Final Normalization layer right before projection
        self.norm = RMSNorm(config.d_model)

        # 4. Language Model Head: Projects d_model vectors to vocabulary size logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # OPTIONAL: Weight Tying (Shares weights between embedding and output layer)
        # Standard in GPT-2/3 to save memory and improve performance on smaller models.
        self.tok_embeddings.weight = self.lm_head.weight

        # Precompute RoPE frequencies for max context length during initialization
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.context_length
        )
        self.register_buffer("freqs_cis", freqs_cis)

        # Initialize neural network weights using standard scaling
        self.apply(self._init_weights)

        logger.info(
            "GPT Model Initialized",
            params=self.get_num_params(),
            layers=config.n_layers,
            d_model=config.d_model,
        )

    def _init_weights(self, module: nn.Module):
        """Initializes weights with standard normal distribution for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Calculates total trainable parameters in the network."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the full language model.

        Args:
            idx (torch.Tensor): Tensor of shape (Batch, Seq_Len) with token IDs.
            targets (Optional[torch.Tensor]): Target token IDs of shape (Batch, Seq_Len) for loss.

        Returns:
            Tuple containing:
            - logits: Raw prediction scores of shape (Batch, Seq_Len, vocab_size)
            - loss: Cross-Entropy loss tensor (or None if no targets provided)
        """
        b, t = idx.size()

        if t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {t}, context length is {self.config.context_length}"
            )

        # Step 1: Convert token IDs to dense vectors
        x = self.tok_embeddings(idx)
        x = self.dropout(x)

        # Step 2: Extract RoPE frequencies for current sequence length t
        freqs_cis = self.freqs_cis[:t]

        # Step 3: Pass sequentially through all N Transformer Blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Step 4: Apply final normalization
        x = self.norm(x)

        # Step 5: Project to logits
        if targets is not None:
            # Training mode: Calculate logits and Cross-Entropy loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference mode optimization: Only calculate logits for the last token position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
