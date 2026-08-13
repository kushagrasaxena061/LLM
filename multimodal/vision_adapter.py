# multimodal/vision_adapter.py
"""Vision-Language Adapter for Multimodal LLM Integration."""

import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)

class VisionLanguageAdapter(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        """
        Projects output from a Vision Encoder (like CLIP) into the LLM's embedding space.
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        
        # Linear projection aligns the visual features with text semantics
        self.projection = nn.Linear(vision_dim, llm_dim, bias=False)
        logger.info("VisionLanguageAdapter initialized", vision_dim=vision_dim, llm_dim=llm_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: Tensor of shape (Batch, Num_Patches, Vision_Dim)
        Returns:
            Projected embeddings of shape (Batch, Num_Patches, LLM_Dim)
        """
        return self.projection(image_features)

def combine_embeddings(text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Prepends projected image embeddings to text embeddings to form a single multimodal sequence.
    """
    # image_embeddings: (B, Num_Patches, LLM_Dim)
    # text_embeddings: (B, Seq_Len, LLM_Dim)
    # combined: (B, Num_Patches + Seq_Len, LLM_Dim)
    return torch.cat([image_embeddings, text_embeddings], dim=1)
