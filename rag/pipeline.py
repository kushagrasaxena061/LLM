# rag/pipeline.py
"""Retrieval-Augmented Generation (RAG) pipeline combining vector search with prompt injection."""

import torch
from typing import List
from rag.vector_store import SimpleVectorStore
from tokenizer.base import BaseTokenizer
from model.transformer import GPT
from inference.generate import generate_text
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self, vector_store: SimpleVectorStore, model: GPT, tokenizer: BaseTokenizer, device: str):
        """
        Initializes the RAG generation pipeline.
        
        Args:
            vector_store: The populated SimpleVectorStore instance.
            model: The trained GPT language model.
            tokenizer: The BPE tokenizer.
            device: Hardware device ('mps', 'cuda', 'cpu').
        """
        self.vector_store = vector_store
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        logger.info("RAGPipeline initialized successfully")

    def answer_query(self, query: str, query_embedding: torch.Tensor, top_k: int = 1, max_new_tokens: int = 20) -> str:
        """
        Retrieves relevant context, constructs an augmented prompt, and generates a response.
        """
        # 1. Retrieve relevant document chunks via vector similarity search
        retrieved_results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        if retrieved_results:
            context_chunk, score = retrieved_results[0]
            logger.info("Retrieved RAG context chunk", score=f"{score:.4f}", chunk=context_chunk[:30])
        else:
            context_chunk = "No relevant context found."

        # 2. Construct the augmented prompt template
        augmented_prompt = (
            f"### Context:\n{context_chunk}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:\n"
        )
        
        # 3. Generate response autoregressively using our model and inference engine
        response = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=augmented_prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
            temperature=0.7
        )
        
        return response
