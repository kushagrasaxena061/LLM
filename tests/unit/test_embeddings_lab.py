# tests/unit/test_embeddings_lab.py
"""Unit tests for embedding extraction, similarity matrix, and PCA projection."""

import torch
from model.config import GPTConfig
from model.transformer import GPT
from evaluation.embeddings import EmbeddingEngine

def test_embedding_and_pca_math():
    config = GPTConfig(vocab_size=100, context_length=32, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config)
    engine = EmbeddingEngine(model)
    
    token_ids = torch.randint(0, 100, (4, 8))
    seq_emb = engine.extract_sequence_embedding(token_ids)
    assert seq_emb.shape == (4, 32)
    
    sim_matrix = engine.compute_similarity_matrix(seq_emb)
    assert sim_matrix.shape == (4, 4)
    assert torch.allclose(torch.diag(sim_matrix), torch.ones(4), atol=1e-4)
    
    pca_pts = engine.compute_pca_2d(seq_emb)
    assert len(pca_pts) == 4
    assert "x" in pca_pts[0] and "y" in pca_pts[0]
