import torch
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]

class SimpleVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.embeddings: torch.Tensor = torch.empty((0, embedding_dim))

    def add_documents(self, docs: List[Document], embeddings: torch.Tensor):
        self.documents.extend(docs)
        if self.embeddings.numel() == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)

    def similarity_search(self, query_embedding: torch.Tensor, top_k: int = 2) -> List[Tuple[Document, float]]:
        if self.embeddings.numel() == 0: return []
        query_embedding = query_embedding.to(self.embeddings.device)
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        doc_norm = torch.nn.functional.normalize(self.embeddings, p=2, dim=-1)
        scores = torch.matmul(doc_norm, query_norm.unsqueeze(-1)).squeeze(-1)
        top_k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append((self.documents[idx.item()], score.item()))
        return results

    def add_texts(self, texts: list, embeddings: torch.Tensor):
        import uuid
        docs = [Document(id=str(uuid.uuid4()), text=t, metadata={"source": "legacy"}) for t in texts]
        self.add_documents(docs, embeddings)
