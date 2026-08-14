# tests/unit/test_rag_hybrid.py
"""Unit tests to verify Reciprocal Rank Fusion (RRF) algorithm correctness."""

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Fuses two ranked lists using RRF. 
    Score = 1 / (k + rank)
    """
    fused_scores = {}
    
    # Process dense ranks
    for rank, (doc_id, _) in enumerate(dense_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Process sparse ranks
    for rank, (doc_id, _) in enumerate(sparse_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort by descending score
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def test_rrf_ranking_boost():
    """
    CRITICAL RAG TEST:
    Proves that a document appearing in BOTH dense and sparse results 
    gets boosted above a document that only appears highly in one.
    """
    # Format: [(doc_id, score), ...] - assuming already sorted by score
    dense_results = [("doc_A", 0.9), ("doc_B", 0.8), ("doc_C", 0.7)]
    sparse_results = [("doc_C", 10.5), ("doc_D", 9.2), ("doc_A", 8.1)]
    
    # In Dense: doc_A is 1st.
    # In Sparse: doc_A is 3rd. 
    # doc_C is 3rd in Dense, 1st in Sparse.
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    
    # Extract just the sorted doc_ids
    ranked_docs = [doc_id for doc_id, score in fused]
    
    assert "doc_A" in ranked_docs, "Missing document in fused results!"
    assert "doc_C" in ranked_docs, "Missing document in fused results!"
    
    print(f"\n✅ RRF Hybrid Fusion Verified!")
    print(f"   - Fused Ranking: {ranked_docs}")
    # doc_A (1st + 3rd) and doc_C (3rd + 1st) should tie or be at the top over doc_B (2nd + None)
    assert ranked_docs.index("doc_B") > ranked_docs.index("doc_A"), "RRF failed to prioritize multi-modal matches!"
