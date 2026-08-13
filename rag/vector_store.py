# rag/vector_store.py
"""Lightweight vector database and semantic retrieval engine for RAG."""

import torch
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class SimpleVectorStore:
    def __init__(self, embedding_dim: int):
        """
        Initializes an in-memory vector store for document chunk retrieval.
        
        Args:
            embedding_dim (int): Dimensionality of the embedding vectors.
        """
        self.embedding_dim = embedding_dim
        self.documents: List[str] = []
        self.embeddings: torch.Tensor = torch.empty((0, embedding_dim))
        
        logger.info("VectorStore initialized", embedding_dim=embedding_dim)

    def add_texts(self, texts: List[str], embeddings: torch.Tensor):
        """
        Adds text chunks and their corresponding embedding vectors to the store.
        """
        assert embeddings.shape[1] == self.embedding_dim, "Embedding dimension mismatch!"
        
        self.documents.extend(texts)
        if self.embeddings.numel() == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
            
        logger.info("Documents added to VectorStore", total_docs=len(self.documents))

    def similarity_search(self, query_embedding: torch.Tensor, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Performs cosine similarity search to retrieve the top-k most relevant document chunks.
        """
        if self.embeddings.numel() == 0:
            return []
            
        # Ensure stored embeddings match the query device to prevent device mismatch errors
        embeddings = self.embeddings.to(query_embedding.device)
            
        # Normalize vectors for cosine similarity computation
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        doc_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Matrix multiplication computes cosine similarity scores across all documents
        scores = torch.matmul(doc_norm, query_norm.unsqueeze(-1)).squeeze(-1)
        
        # Get top-k highest scores
        top_k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append((self.documents[idx.item()], score.item()))
            
        return results
