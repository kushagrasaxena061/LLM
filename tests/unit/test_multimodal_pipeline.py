import torch
from multimodal.vision_adapter import VisionLanguageAdapter, VisionPatchExtractor

def test_multimodal_projection():
    adapter = VisionLanguageAdapter(vision_dim=512, llm_dim=32)
    extractor = VisionPatchExtractor(patch_size=16, in_channels=3, vision_dim=512)
    
    # Expected real image tensor input: [Batch, Channels, Height, Width]
    img_tensor = torch.randn(1, 3, 224, 224)
    projected = adapter(extractor(img_tensor))
    
    # Verify the shape: Batch=1, Patches=196 (for 224x224 img with 16x16 patches), LLM_dim=32
    assert projected.shape == (1, 196, 32)
