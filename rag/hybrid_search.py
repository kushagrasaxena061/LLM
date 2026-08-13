# rag/hybrid_search.py
"""Hybrid Retrieval combining Dense (Vector) and Sparse (BM25) search via RRF."""

import torch
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from rag.vector_store import SimpleVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.tokenized_corpus = []

    def fit_bm25(self, texts: List[str]):
        """Trains the BM25 sparse retriever on the document corpus."""
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 Sparse Retriever fitted", corpus_size=len(texts))

    def reciprocal_rank_fusion(self, dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
        """
        Fuses dense and sparse rankings using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score = 1 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}

        # 1. Process dense ranks (Semantic matches)
        for rank, (doc, _) in enumerate(dense_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank + 1)

        # 2. Process sparse ranks (Keyword matches)
        for rank, (doc, _) in enumerate(sparse_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank + 1)

        # 3. Sort documents by their combined RRF score descending
        fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return fused

    def search(self, query: str, query_embedding: torch.Tensor, top_k: int = 2) -> List[Tuple[str, float]]:
        """Executes the dual-search pipeline and returns the fused top-k documents."""
        # 1. Dense Search (Vector Embeddings)
        # We retrieve all documents to properly rank them against the sparse results
        dense_results = self.vector_store.similarity_search(query_embedding, top_k=len(self.vector_store.documents))

        # 2. Sparse Search (BM25 Exact Keyword Match)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        sparse_results = [(self.vector_store.documents[i], score) for i, score in enumerate(sparse_scores)]
        sparse_results.sort(key=lambda x: x[1], reverse=True)

        # 3. Fuse Rankings via RRF
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        logger.info("Hybrid search completed")
        return fused_results[:top_k]
