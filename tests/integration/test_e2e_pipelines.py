import torch
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from multimodal.pipeline import VisionLanguageAdapter, process_multimodal_input
from rag.vector_store import SimpleVectorStore
from rag.hybrid_search import HybridRetriever
from rag.reranker import CrossEncoderReranker

def test_e2e_multimodal_pipeline():
    """Phase 4 Verification: Image -> Vision Processing -> Projection -> LLM"""
    # FIX: Increased vocab_size to 300 to accommodate base UTF-8 byte tokens (256) + special tokens
    config = GPTConfig(vocab_size=300, context_length=64, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config)
    tokenizer = BPETokenizer(vocab_size=300)
    adapter = VisionLanguageAdapter(vision_dim=128, llm_dim=32)
    
    # Mock Image Tensor (Batch=1, Patches=16, Vision_Dim=128)
    image_tensor = torch.randn(1, 16, 128)
    prompt = "Describe this."
    
    # E2E Processing
    combined_embeddings = process_multimodal_input(model, tokenizer, adapter, image_tensor, prompt, device="cpu")
    
    # Text tokens usually = length of prompt. Vision patches = 16.
    assert combined_embeddings.shape[2] == 32, "LLM Dimension mismatch!"
    assert combined_embeddings.shape[1] > 16, "Sequence concatenation failed!"
    print("\n✅ E2E Multimodal Pipeline verified.")

def test_e2e_hybrid_rag_pipeline():
    """Phase 4 Verification: BM25 -> RRF -> Reranking -> Output"""
    store = SimpleVectorStore(embedding_dim=16)
    docs = ["Test document about AI.", "Another document about networking."]
    store.add_texts(docs, torch.randn(2, 16))
    
    hybrid = HybridRetriever(store)
    hybrid.fit_bm25(docs)
    reranker = CrossEncoderReranker()
    
    # 1. Hybrid Search
    candidates = hybrid.search("AI networking", torch.randn(16), top_k=2)
    assert len(candidates) == 2, "Hybrid search failed."
    
    # 2. Rerank
    final_docs = reranker.rerank("AI networking", candidates, top_k=1)
    assert len(final_docs) == 1, "Reranker failed to filter top_k."
    print("\n✅ E2E Hybrid RAG & Reranking verified.")
