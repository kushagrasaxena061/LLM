# explainability/visualizer.py
"""Transformer Explainability Engine for rendering Attention Heatmaps."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from model.transformer import GPT
from tokenizer.base import BaseTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

class AttentionVisualizer:
    def __init__(self, model: GPT, tokenizer: BaseTokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        logger.info("AttentionVisualizer initialized")

    def extract_attention(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """
        Runs a forward pass and extracts the attention matrix from the first transformer layer.
        """
        input_ids = self.tokenizer.encode(text)
        idx = torch.tensor([input_ids], device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            self.model(idx)
            
        # Extract from Layer 0 (the first Transformer Block)
        # Shape: (Batch, Heads, SeqLen, SeqLen)
        attn_matrix = self.model.blocks[0].attn.attention_weights
        
        # Decode individual tokens for the heatmap labels
        tokens = [self.tokenizer.decode([i]).strip() for i in input_ids]
        
        return tokens, attn_matrix

    def plot_attention_heatmap(self, tokens: List[str], attn_matrix: torch.Tensor, head_idx: int = 0):
        """
        Generates a Matplotlib figure visualizing the attention scores.
        """
        # Move tensor to CPU and convert to NumPy for Seaborn
        matrix = attn_matrix[0, head_idx].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", ax=ax)
        
        ax.set_title(f"Attention Heatmap (Layer 0, Head {head_idx})", pad=20)
        ax.set_xlabel("Key (Token Being Attended To)", labelpad=10)
        ax.set_ylabel("Query (Token Focusing)", labelpad=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
