import torch

@torch.no_grad()
def generate_multimodal_text(
    model, tokenizer, vision_extractor, vision_adapter, image_tensor, prompt, max_new_tokens=20, device="cpu"
):
    model.eval()
    
    # 1. Vision Features -> Project to LLM Dimension
    vision_features = vision_extractor(image_tensor.to(device))
    vision_embeds = vision_adapter(vision_features)
    
    # 2. Text Features -> Embed directly into LLM Dimension
    text_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    text_embeds = model.tok_embeddings(text_ids)
    
    # 3. Multimodal Fusion Sequence
    inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
    
    generated_ids = []
    past_key_values = None
    curr_embeds = inputs_embeds
    curr_idx = None
    
    # 4. Autoregressive Loop (Handling the transition from embeds -> token IDs)
    for step in range(max_new_tokens):
        if step == 0:
            logits, _, past_key_values = model(idx=None, inputs_embeds=curr_embeds, use_cache=True)
        else:
            logits, _, past_key_values = model(idx=curr_idx, inputs_embeds=None, past_key_values=past_key_values, use_cache=True)
            
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        curr_idx = next_token
        
        if hasattr(tokenizer, "special_tokens") and "<|im_end|>" in tokenizer.special_tokens:
            if next_token.item() == tokenizer.special_tokens["<|im_end|>"]:
                break
                
    return tokenizer.decode(generated_ids)

def process_multimodal_input(model, tokenizer, adapter, image_tensor, prompt, device="cpu"):
    vision_embeds = adapter(image_tensor.to(device))
    text_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    text_embeds = model.tok_embeddings(text_ids)
    return torch.cat([vision_embeds, text_embeds], dim=1)
