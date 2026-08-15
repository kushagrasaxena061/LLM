import torch
import pytest
from model.config import tiny_test_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.multimodal import generate_multimodal_text

def test_multimodal_generation_end_to_end():
    """
    Proves that a raw image tensor passes through the vision extractor, 
    the language adapter, fuses with text embeddings, and successfully 
    generates text tokens autoregressively.
    """
    device = "cpu"
    model = GPT(tiny_test_config).to(device)
    
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train("The quick brown fox jumps over the lazy dog. A picture is worth a thousand words.")
    
    # Mocking Extractor and Adapter explicitly mapping to tiny_test_config.d_model
    import torch.nn as nn
    extractor = lambda x: torch.randn(x.shape[0], 4, 16).to(device)
    adapter = nn.Linear(16, tiny_test_config.d_model).to(device)
    
    dummy_image = torch.randn(1, 3, 224, 224)
    prompt = "What is this?"
    
    output = generate_multimodal_text(
        model=model,
        tokenizer=tokenizer,
        vision_extractor=extractor,
        vision_adapter=adapter,
        image_tensor=dummy_image,
        prompt=prompt,
        max_new_tokens=5,
        device=device
    )
    
    assert isinstance(output, str), "Failed to generate string from multimodal pipeline."
    assert len(output) > 0, "Generated multimodal text is completely empty."
    print(f"\n✅ Multimodal Pipeline Generated: {output}")
