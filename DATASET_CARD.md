# DATASET_CARD.md

## Dataset Overview
- **Tokenizer Training Corpus:** UTF-8 character and byte-level corpus augmented with ASCII printable range.
- **BPE Vocabulary Size:** Configurable (260 - 50,257 tokens).
- **RAG Retrieval Corpus:** In-memory vector store supporting dense cosine embeddings and BM25 sparse keyword indices.
- **Preprocessing Pipeline:** Streaming sequence chunker with sliding context windows and next-token target alignment.
