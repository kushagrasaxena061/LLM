import torch
from model.config import tiny_test_config
from model.transformer import GPT

def test_multimodal_e2e_gpt_connection():
    device = "cpu"
    model = GPT(tiny_test_config).to(device)
    
    simulated_vision_projection = torch.randn(2, 4, tiny_test_config.d_model, device=device)
    text_ids = torch.randint(0, 300, (2, 5), device=device)
    text_embeddings = model.tok_embeddings(text_ids)
    
    combined_embeddings = torch.cat([simulated_vision_projection, text_embeddings], dim=1)
    
    # Pass directly into transformer natively without manually unpacking blocks
    logits, loss, _ = model(idx=None, inputs_embeds=combined_embeddings, use_cache=False)
    
    assert logits.shape == (2, 1, tiny_test_config.vocab_size)
    print("\n✅ E2E Multimodal GPT Forward Pass Verified.")
