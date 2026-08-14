# evaluation/rag_eval.py
"""Information Retrieval (IR) and generation evaluation metrics."""

import math
from typing import List, Set

class RAGEvaluator:
    @staticmethod
    def recall_at_k(retrieved: List[str], ground_truth: Set[str], k: int) -> float:
        """Recall@K = (Relevant docs retrieved in top K) / (Total relevant docs)."""
        if not ground_truth:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in ground_truth)
        return hits / len(ground_truth)

    @staticmethod
    def precision_at_k(retrieved: List[str], ground_truth: Set[str], k: int) -> float:
        """Precision@K = (Relevant docs retrieved in top K) / K."""
        if k == 0:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in ground_truth)
        return hits / k

    @staticmethod
    def mrr(retrieved: List[str], ground_truth: Set[str]) -> float:
        """Mean Reciprocal Rank (MRR) of the first relevant document."""
        for rank, doc in enumerate(retrieved, 1):
            if doc in ground_truth:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], ground_truth: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain (NDCG@K)."""
        top_k = retrieved[:k]
        dcg = 0.0
        for i, doc in enumerate(top_k):
            rel = 1.0 if doc in ground_truth else 0.0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG calculation
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
        return (dcg / idcg) if idcg > 0.0 else 0.0

    @staticmethod
    def calculate_faithfulness(context: str, generated_answer: str) -> float:
        """Calculates factual overlap between context and generated answer."""
        context_tokens = set(context.lower().split())
        answer_tokens = generated_answer.lower().split()
        if not answer_tokens:
            return 0.0
        overlap = sum(1 for t in answer_tokens if t in context_tokens)
        return overlap / len(answer_tokens)
