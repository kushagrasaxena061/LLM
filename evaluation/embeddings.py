import torch
import torch.nn as nn

class EmbeddingEngine:
    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        try: self.device = next(model.parameters()).device
        except Exception: self.device = "cpu"

    @torch.no_grad()
    def extract_sequence_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        b, seq_len = input_ids.shape
        x = self.model.tok_embeddings(input_ids)
        freqs_cis = None
        if hasattr(self.model, "freqs_cis"):
            freqs_cis = self.model.freqs_cis[:seq_len].to(self.device)
        for block in getattr(self.model, "blocks", []):
            x, _, _ = block(x, freqs_cis=freqs_cis, use_cache=False)
        ln_f = getattr(self.model, "ln_f", getattr(self.model, "norm", None))
        if ln_f: x = ln_f(x)
        embeddings = x.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        if was_training: self.model.train()
        return embeddings

    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        normed = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return torch.matmul(normed, normed.T)

    def compute_pca_2d(self, embeddings: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(embeddings, dim=0)
        centered = embeddings - mean
        U, S, V = torch.svd(centered)
        proj = torch.matmul(centered, V[:, :2])
        return [{"x": p[0].item(), "y": p[1].item()} for p in proj]
