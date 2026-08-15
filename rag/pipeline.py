import torch
from typing import List
from rag.vector_store import Document
from inference.generate import generate_text

class RAGPipeline:
    def __init__(self, hybrid_retriever=None, reranker=None, model=None, tokenizer=None, embedding_engine=None, device="cpu", **kwargs):
        if "vector_store" in kwargs and hybrid_retriever is None:
            from rag.hybrid_search import HybridRetriever
            from evaluation.embeddings import EmbeddingEngine
            from rag.reranker import HeuristicLexicalReranker
            self.retriever = HybridRetriever(kwargs["vector_store"], EmbeddingEngine(model, tokenizer))
            self.reranker = HeuristicLexicalReranker()
            self.embedding_engine = EmbeddingEngine(model, tokenizer)
        else:
            self.retriever = hybrid_retriever
            self.reranker = reranker
            self.embedding_engine = embedding_engine
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer_query(self, query: str, top_k: int = 3, max_new_tokens: int = 50) -> str:
        query_ids = torch.tensor([self.tokenizer.encode(query)], dtype=torch.long, device=self.device)
        query_vec = self.embedding_engine.extract_sequence_embedding(query_ids)[0].detach()

        retrieved_results = self.retriever.search(query, query_embedding=query_vec, top_k=top_k * 2)
        if not retrieved_results:
            return "No relevant information found."

        candidates = [(doc.text, score, doc) for doc, score in retrieved_results]
        reranked_docs = self.reranker.rerank_documents(query, candidates, top_k=top_k)

        context_lines = []
        citations = []
        for rank, doc in enumerate(reranked_docs, 1):
            source = doc.metadata.get("source", f"doc_{doc.id}")
            context_lines.append(f"[Source {rank} | {source}]: {doc.text}")
            citations.append(f"[{rank}] {source}")

        context_str = "\n".join(context_lines)
        
        prompt = (
            "System: Use the following retrieved documents to answer the user question. "
            "Cite your sources using [Source X].\n\n"
            f"Context:\n{context_str}\n\n"
            f"User: {query}\nAssistant:"
        )

        answer = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
            stop_tokens=["<|im_end|>", "<|endoftext|>"]
        )

        return answer + "\n\nSources:\n" + "\n".join(citations)
