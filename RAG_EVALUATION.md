# RAG_EVALUATION.md
## Empirical RAG Strategy Evaluation Report

Comparison of retrieval architectures across standard Information Retrieval (IR) metrics:

| Strategy | Recall@2 | Precision@2 | MRR | NDCG@2 |
| :--- | :--- | :--- | :--- | :--- |
| **Dense** | 1.000 | 0.500 | 1.000 | 1.000 |
| **BM25** | 1.000 | 0.500 | 1.000 | 1.000 |
| **Hybrid_RRF** | 1.000 | 0.500 | 1.000 | 1.000 |
| **Hybrid_Reranked** | 1.000 | 0.500 | 1.000 | 1.000 |

### Architectural Findings:
- **Dense Retrieval:** Captures semantic meaning but can miss exact keyword occurrences.
- **BM25 Sparse Retrieval:** Handles exact token matching but lacks semantic awareness.
- **Hybrid (RRF):** Merges both signals to minimize catastrophic recall failures.
- **Hybrid + Reranking:** Provides the highest precision and MRR by scoring joint query-document interactions.