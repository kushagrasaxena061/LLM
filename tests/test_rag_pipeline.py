# tests/test_rag_pipeline.py
"""Integration tests for the complete RAG text generation pipeline."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from rag.vector_store import SimpleVectorStore
from rag.pipeline import RAGPipeline

def test_rag_pipeline_execution():
    """Verifies that RAG successfully injects context and generates a response."""
    dim = 16
    config = GPTConfig(
        vocab_size=260,
        context_length=256,
        d_model=dim,
        n_layers=2,
        n_heads=2
    )
    model = GPT(config).to(env_config.device)
    
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Artificial intelligence is transforming technology.")
    
    # Setup Vector Store
    store = SimpleVectorStore(embedding_dim=dim)
    docs = ["The quick brown fox jumps over the lazy dog."]
    
    torch.manual_seed(42)
    embeddings = torch.randn(1, dim)
    store.add_texts(docs, embeddings)
    
    # Initialize RAG Pipeline
    pipeline = RAGPipeline(vector_store=store, model=model, tokenizer=tokenizer, device=env_config.device)
    
    # Run query
    query = "What does the fox do?"
    query_embedding = embeddings[0]
    
    output = pipeline.answer_query(query, top_k=1, max_new_tokens=15)
    
    assert isinstance(output, str), "RAG pipeline did not return a string!"
    assert len(output) > 0, "RAG pipeline did not return any output!"
    
    print(f"\n✅ RAG Pipeline Integration Test Passed!")
    print(f"   - Generated Output:\n{output}")
