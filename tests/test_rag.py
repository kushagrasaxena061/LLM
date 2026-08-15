# tests/test_rag.py
"""Unit tests for the RAG vector store and semantic search retrieval."""

import torch
from rag.vector_store import SimpleVectorStore

def test_vector_store_retrieval():
    """Verifies that vector similarity search successfully retrieves the correct document."""
    dim = 16
    store = SimpleVectorStore(embedding_dim=dim)
    
    # Create sample documents and deterministic mock embeddings
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum computing utilizes qubits for parallel computation.",
        "Transformers are deep learning architectures based on attention mechanisms."
    ]
    
    # Generate mock embeddings (normalized random vectors)
    torch.manual_seed(42)
    embeddings = torch.randn(len(docs), dim)
    
    store.add_texts(docs, embeddings)
    
    # Query using a vector close to the first document's embedding
    query_vector = embeddings[0] + torch.randn(dim) * 0.01
    results = store.similarity_search(query_vector, top_k=1)
    
    assert len(results) == 1, "Retrieval did not return the requested top_k results!"
    retrieved_text, score = results[0]
    
    assert getattr(retrieved_text, 'text', retrieved_text) == docs[0], "Retrieved wrong document chunk!"
    print(f"\n✅ RAG Vector Retrieval Passed! Top match score: {score:.4f}")
    print(f"   - Retrieved Chunk: '{retrieved_text}'")
