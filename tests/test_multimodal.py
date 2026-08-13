# tests/test_multimodal.py
"""Unit tests for Multimodal projection and embedding concatenation."""

import torch
from multimodal.vision_adapter import VisionLanguageAdapter, combine_embeddings

def test_multimodal_projection():
    """Verifies that vision embeddings are correctly projected and concatenated with text."""
    batch_size = 2
    num_patches = 16 # e.g., a 4x4 image grid from a Vision Transformer
    seq_len = 10     # 10 text tokens in the user's prompt
    vision_dim = 512 # Standard CLIP ViT output dimension
    llm_dim = 32     # Our dummy LLM's embedding dimension (d_model)

    # 1. Mock output from a Vision Encoder (simulating an image of a dog)
    mock_image_features = torch.randn(batch_size, num_patches, vision_dim)

    # 2. Mock output from our LLM Token Embeddings (simulating "What is in this image?")
    mock_text_embeddings = torch.randn(batch_size, seq_len, llm_dim)

    # 3. Project image features into the LLM's dimension
    adapter = VisionLanguageAdapter(vision_dim, llm_dim)
    projected_images = adapter(mock_image_features)

    assert projected_images.shape == (batch_size, num_patches, llm_dim), "Projection shape mismatch!"

    # 4. Combine embeddings into one continuous sequence for the Transformer
    combined = combine_embeddings(mock_text_embeddings, projected_images)

    expected_seq_length = num_patches + seq_len
    assert combined.shape == (batch_size, expected_seq_length, llm_dim), "Combined shape mismatch!"

    print(f"\n✅ Multimodal Adapter Test Passed!")
    print(f"   - Original Vision Shape: {list(mock_image_features.shape)} (Vision Dim: {vision_dim})")
    print(f"   - Projected Vision Shape: {list(projected_images.shape)} (LLM Dim: {llm_dim})")
    print(f"   - Final Multimodal Sequence Shape: {list(combined.shape)}")
