import torch
import torch.nn as nn
from PIL import Image

try:
    import torchvision.transforms as transforms
except ImportError:
    transforms = None

class VisionPatchEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, vision_dim: int = 512, patch_size: int = 16):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, vision_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # img_tensor: [B, C, H, W] -> [B, vision_dim, H/patch, W/patch]
        x = self.patch_embed(img_tensor)
        # Flatten the spatial dimensions and transpose -> [B, num_patches, vision_dim]
        return x.flatten(2).transpose(1, 2)

class VisionLanguageAdapter(nn.Module):
    def __init__(self, vision_dim: int = 512, llm_dim: int = 768):
        super().__init__()
        self.encoder = VisionPatchEncoder(vision_dim=vision_dim)
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        vision_features = self.encoder(img_tensor)
        projected = self.projection(vision_features)
        return projected

def preprocess_image(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    if transforms is not None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image.convert("RGB")).unsqueeze(0)
    else:
        # Safe fallback if torchvision is missing
        import numpy as np
        img = image.convert("RGB").resize((img_size, img_size))
        arr = np.array(img).transpose(2, 0, 1) / 255.0
        tensor = torch.tensor(arr, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

def combine_embeddings(text_embeddings: torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cat([vision_embeddings, text_embeddings], dim=1)
