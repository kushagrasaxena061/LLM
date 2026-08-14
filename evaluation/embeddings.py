# evaluation/embeddings.py
"""Embedding extraction, vector similarity, and PCA 2D projections."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from model.transformer import GPT

class EmbeddingEngine:
    def __init__(self, model: GPT):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def extract_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Extracts token embedding vectors directly from the model embedding table."""
        # token_ids shape: [batch_size, seq_len]
        return self.model.tok_embeddings(token_ids)

    @torch.no_grad()
    def extract_sequence_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Extracts sequence-level pooled representation from the final Transformer block."""
        x = self.model.tok_embeddings(token_ids)
        x = self.model.dropout(x)
        seq_len = token_ids.shape[1]
        freqs_cis = self.model.freqs_cis[:seq_len].to(token_ids.device)
        
        for block in self.model.blocks:
            x, _, _ = block(x, freqs_cis, use_cache=False)
            
        x = self.model.norm(x)
        # Mean pooling across the sequence dimension
        pooled = torch.mean(x, dim=1)
        return F.normalize(pooled, p=2, dim=-1)

    @staticmethod
    def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        """Computes pairwise cosine similarity between normalized vectors."""
        norm_emb = F.normalize(embeddings, p=2, dim=-1)
        return torch.mm(norm_emb, norm_emb.t())

    @staticmethod
    def compute_pca_2d(embeddings: torch.Tensor) -> List[Dict[str, float]]:
        """Projects high-dimensional vectors to 2D coordinates using SVD PCA."""
        if embeddings.shape[0] < 2:
            return [{"x": 0.0, "y": 0.0}] * embeddings.shape[0]
            
        centered = embeddings - torch.mean(embeddings, dim=0, keepdim=True)
        # SVD decomposition
        U, S, V = torch.pca_lowrank(centered, q=2)
        projected = torch.mm(centered, V[:, :2])
        
        return [{"x": float(row[0]), "y": float(row[1])} for row in projected]
