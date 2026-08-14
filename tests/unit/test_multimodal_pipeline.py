import torch
from multimodal.vision_adapter import VisionLanguageAdapter
def test_multimodal_projection():
    adapter = VisionLanguageAdapter(vision_dim=512, llm_dim=32)
    # Expected real image tensor input: [Batch, Channels, Height, Width]
    img_tensor = torch.randn(1, 3, 224, 224)
    projected = adapter(img_tensor)
    # 224/16 = 14 patches per side -> 196 patches
    assert projected.shape == (1, 196, 32)
