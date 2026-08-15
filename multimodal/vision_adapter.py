import torch
import torch.nn as nn
from PIL import Image

class VisionPatchExtractor(nn.Module):
    """Deterministically extracts real visual features from image pixels."""
    def __init__(self, patch_size=16, in_channels=3, vision_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, vision_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def preprocess_image(image: Image.Image, size=224) -> torch.Tensor:
    image = image.resize((size, size)).convert('RGB')
    import numpy as np
    img_arr = (np.array(image, dtype=np.float32) / 127.5) - 1.0
    tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

class VisionLanguageAdapter(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.projection = nn.Linear(vision_dim, llm_dim, bias=False)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(vision_embeddings)
