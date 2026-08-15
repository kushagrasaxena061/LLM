# tests/test_hybrid_rag.py
"""Unit tests for Hybrid RAG (Dense + BM25) and Reciprocal Rank Fusion."""

import torch
from rag.vector_store import SimpleVectorStore
from rag.hybrid_search import HybridRetriever

def test_hybrid_retrieval_rrf():
    """Verifies that RRF successfully merges semantic and keyword search scores."""
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a fast, modern programming language for AI.",
        "FastAPI is a modern, fast web framework for building APIs."
    ]
    
    # 1. Setup Vector Store with dummy embeddings
    dim = 16
    store = SimpleVectorStore(embedding_dim=dim)
    torch.manual_seed(42)
    embeddings = torch.randn(len(docs), dim)
    store.add_texts(docs, embeddings)
    
    # 2. Initialize and train Hybrid Retriever
    hybrid = HybridRetriever(store)
    hybrid.fit_bm25(docs)
    
    # 3. Query prioritizing specific keywords
    query = "fast modern web framework"
    query_embedding = torch.randn(dim) # Random semantic vector
    
    results = hybrid.search(query, query_embedding, top_k=1)
    
    assert len(results) == 1, "Hybrid search failed to return results!"
    top_doc, rrf_score = results[0]
    
    # The BM25 algorithm should force the FastAPI document to the top due to keyword overlaps
    assert "FastAPI" in getattr(top_doc, "text", top_doc), "Hybrid search failed to boost the BM25 exact keyword match!"
    
    print(f"\n✅ Advanced RAG (Hybrid Search + RRF) Test Passed!")
    print(f"   - Query: '{query}'")
    print(f"   - Top Retrieved Document: '{top_doc}'")
    print(f"   - RRF Score: {rrf_score:.4f}")
