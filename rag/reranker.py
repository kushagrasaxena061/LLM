import math
from typing import List, Tuple, Any

class HeuristicLexicalReranker:
    def score_pair(self, query: str, document: str) -> float:
        q_tokens = set(query.lower().split())
        d_tokens = document.lower().split()
        if not q_tokens or not d_tokens: return 0.0
        overlap = sum(1 for t in d_tokens if t in q_tokens)
        density = overlap / len(d_tokens)
        coverage = overlap / len(q_tokens)
        return (coverage * 0.7 + density * 0.3) * math.log(len(d_tokens) + 1)

    def rerank_documents(self, query: str, candidates: List[Tuple[str, float, Any]], top_k: int = 2) -> List[Any]:
        scored_docs = []
        for text, initial_score, doc_obj in candidates:
            cross_score = self.score_pair(query, text)
            final_score = (initial_score * 0.4) + (cross_score * 0.6)
            scored_docs.append((final_score, doc_obj))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc_obj for score, doc_obj in scored_docs[:top_k]]
