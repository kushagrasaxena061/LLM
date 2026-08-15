import torch
import pytest
from model.config import tiny_test_config
from model.transformer import GPT

def test_multimodal_e2e_gpt_connection():
    """
    Proves that projected vision features can seamlessly prepend to language 
    embeddings and flow cleanly through the entire GPT forward pass.
    """
    device = "cpu"
    model = GPT(tiny_test_config).to(device)
    
    # Simulate a 768-dim vision encoder output projected down to GPT's d_model (32 for tiny test)
    # Batch size 2, 4 image patches, d_model features
    simulated_vision_projection = torch.randn(2, 4, tiny_test_config.d_model, device=device)
    
    # Simulate Text Prompt: "What is this?" (Batch 2, 5 tokens)
    text_ids = torch.randint(0, 300, (2, 5), device=device)
    text_embeddings = model.tok_embeddings(text_ids)
    
    # Multimodal Concatenation (Vision + Text)
    combined_embeddings = torch.cat([simulated_vision_projection, text_embeddings], dim=1)
    
    assert combined_embeddings.shape == (2, 9, tiny_test_config.d_model), "Multimodal concatenation failed."
    
    # Pass directly into the Transformer blocks bypassing the standard token embedding layer
    x = combined_embeddings
    for block in model.blocks:
        freqs_cis = model.freqs_cis[:x.shape[1]].to(device) if hasattr(model, 'freqs_cis') else None
        x, _, _ = block(x, freqs_cis=freqs_cis, use_cache=False)
    
    final_norm = getattr(model, 'norm', getattr(model, 'ln_f', None))
    if final_norm is not None:
        x = final_norm(x)
    logits = model.lm_head(x)
    
    # Output logits must match the vocab size for generation
    assert logits.shape == (2, 9, tiny_test_config.vocab_size), "Multimodal GPT forward pass crashed or output wrong shape."
