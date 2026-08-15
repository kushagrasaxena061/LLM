import torch
import pytest
from unittest.mock import patch
from model.config import tiny_test_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from evaluation.embeddings import EmbeddingEngine
from rag.vector_store import SimpleVectorStore, Document
from rag.hybrid_search import HybridRetriever
from rag.reranker import HeuristicLexicalReranker
from rag.pipeline import RAGPipeline

def test_production_rag_no_random_embeddings():
    device = "cpu"
    model = GPT(tiny_test_config).to(device)
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Python is great.")
    
    embedding_engine = EmbeddingEngine(model, tokenizer)
    store = SimpleVectorStore(embedding_dim=tiny_test_config.d_model)
    retriever = HybridRetriever(store, embedding_engine)
    reranker = HeuristicLexicalReranker()
    
    pipeline = RAGPipeline(retriever, reranker, model, tokenizer, embedding_engine, device)
    
    docs = [
        Document(id="1", text="FastAPI is a modern web framework.", metadata={"source": "docs/fastapi.txt"}),
        Document(id="2", text="PyTorch is a deep learning library.", metadata={"source": "docs/pytorch.txt"})
    ]
    
    embeds = []
    for d in docs:
        ids = torch.tensor([tokenizer.encode(d.text)], dtype=torch.long, device=device)
        vec = embedding_engine.extract_sequence_embedding(ids)[0].detach()
        embeds.append(vec)
        
    store.add_documents(docs, torch.stack(embeds))
    retriever.fit_bm25(docs)

    original_randn = torch.randn
    def mock_randn(*args, **kwargs):
        raise RuntimeError("CRITICAL FAILURE: torch.randn was called in production RAG path!")
        
    with patch("torch.randn", side_effect=mock_randn):
        answer = pipeline.answer_query("Tell me about FastAPI.", top_k=1)
        
    assert "[1] docs/fastapi.txt" in answer, "Citation metadata did not survive the pipeline!"
    assert "docs/fastapi.txt" in answer, "Source metadata missing!"
    
    candidates = retriever.search("FastAPI", embeds[0], top_k=2)
    assert candidates[0][0].id == "1", "RRF/BM25 failed to retrieve the correct document!"

def test_reranker_accurate_naming():
    reranker = HeuristicLexicalReranker()
    assert "Heuristic Reranker" not in reranker.__class__.__name__, "Reranker is falsely named!"
