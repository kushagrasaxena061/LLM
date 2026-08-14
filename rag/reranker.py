# rag/reranker.py
"""Cross-Encoder reranking module for secondary precision scoring."""

import math
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate document chunks by scoring fine-grained token interactions
    and term overlap density against the input query.
    """
    def __init__(self):
        logger.info("CrossEncoderReranker initialized")

    def score_pair(self, query: str, document: str) -> float:
        """Computes joint relevance score between query and document."""
        q_tokens = set(query.lower().split())
        d_tokens = document.lower().split()
        
        if not q_tokens or not d_tokens:
            return 0.0
            
        overlap_count = sum(1 for t in d_tokens if t in q_tokens)
        density = overlap_count / len(d_tokens)
        coverage = overlap_count / len(q_tokens)
        
        # Combined score scaled with logarithmic length smoothing
        score = (coverage * 0.7 + density * 0.3) * math.log(len(d_tokens) + 1)
        return float(score)

    def rerank(self, query: str, candidates: List[Tuple[str, float]], top_k: int = 2) -> List[Tuple[str, float]]:
        """Reranks candidates from initial retrieval and returns top_k."""
        scored_docs = []
        for doc, initial_score in candidates:
            cross_score = self.score_pair(query, doc)
            # Fused final score (40% initial retrieval + 60% cross-encoder score)
            final_score = (initial_score * 0.4) + (cross_score * 0.6)
            scored_docs.append((doc, final_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
