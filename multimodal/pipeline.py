# multimodal/pipeline.py
"""End-to-End Multimodal Pipeline: Image -> Patch Extraction -> GPT -> Text."""

import torch
import torch.nn as nn
from model.transformer import GPT

class VisionLanguageAdapter(nn.Module):
    """Projects vision embeddings into the LLM's hidden dimension."""
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.projection = nn.Linear(vision_dim, llm_dim, bias=False)
        
    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(vision_embeddings)

def process_multimodal_input(
    model: GPT, 
    tokenizer, 
    adapter: VisionLanguageAdapter, 
    image_tensor: torch.Tensor, 
    prompt: str, 
    device: str
) -> torch.Tensor:
    """
    Phase 10 Integration:
    Takes a raw image tensor (simulated vision encoder output) and text prompt,
    projects the image patches, and concatenates them with text token embeddings.
    """
    # 1. Project vision features to LLM dimension
    vision_embeddings = adapter(image_tensor.to(device)) # Shape: (Batch, Num_Patches, LLM_Dim)
    
    # 2. Embed text tokens
    token_ids = tokenizer.encode(prompt)
    text_tensor = torch.tensor([token_ids], device=device)
    text_embeddings = model.tok_embeddings(text_tensor) # Shape: (Batch, Seq_Len, LLM_Dim)
    
    # 3. Concatenate [Image Embeddings | Text Embeddings]
    combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
    
    return combined_embeddings
