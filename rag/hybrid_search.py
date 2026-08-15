import torch
import math
from collections import Counter
from typing import List, Tuple
from rag.vector_store import SimpleVectorStore, Document

class HybridRetriever:
    def __init__(self, vector_store: SimpleVectorStore, embedding_engine=None):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.doc_freqs = Counter()
        self.doc_lengths = []
        self.avgdl = 0.0

    def fit_bm25(self, docs: List[Document]):
        self.doc_lengths = [len(((d.text if hasattr(d, 'text') else d) if hasattr(d, 'text') else d).split()) for d in docs]
        self.avgdl = sum(self.doc_lengths) / len(docs) if docs else 1.0
        for d in docs:
            self.doc_freqs.update(set(((d.text if hasattr(d, 'text') else d) if hasattr(d, 'text') else d).lower().split()))

    def search(self, query: str, query_embedding: torch.Tensor, top_k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vector_store.documents: return []
        if len(self.doc_lengths) != len(self.vector_store.documents):
            self.fit_bm25(self.vector_store.documents)
        
        dense_results = self.vector_store.similarity_search(query_embedding, top_k=len(self.vector_store.documents))
        
        q_terms = query.lower().split()
        bm25_scores = []
        N = len(self.vector_store.documents)
        for i, doc in enumerate(self.vector_store.documents):
            score = 0.0
            d_terms = (doc.text if hasattr(doc, 'text') else doc).lower().split()
            term_counts = Counter(d_terms)
            for term in q_terms:
                if term not in term_counts: continue
                tf = term_counts[term]
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * self.doc_lengths[i] / self.avgdl))
            bm25_scores.append((doc, score))

        rrf_scores = {doc.id: 0.0 for doc in self.vector_store.documents}
        for rank, (doc, _) in enumerate(dense_results):
            rrf_scores[doc.id] += 1.0 / (60 + rank + 1)
        
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, _) in enumerate(bm25_scores):
            rrf_scores[doc.id] += 1.0 / (60 + rank + 1)

        fused = sorted(self.vector_store.documents, key=lambda d: rrf_scores[d.id], reverse=True)
        return [(doc, rrf_scores[doc.id]) for doc in fused[:top_k]]
