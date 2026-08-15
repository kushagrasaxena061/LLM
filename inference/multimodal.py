"""
Educational Multimodal Generation Pipeline.

NOTE: This proves the computational graph correctly fuses 
Vision features -> LLM Embeddings -> Transformer Generation.
Visual understanding is dummy/random unless a pre-trained VLM adapter is loaded.
"""
import torch

def generate_multimodal_text(model, tokenizer, vision_extractor, vision_adapter, image_tensor, prompt, max_new_tokens=20, device="cpu"):
    model.eval()
    was_training = model.training
    
    with torch.no_grad():
        # 1. Vision Pathway (Patch Extractor -> Feature Adapter)
        patches = vision_extractor(image_tensor.to(device))
        vision_embeds = vision_adapter(patches)
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0)
            
        # 2. Text Pathway
        text_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        text_embeds = model.tok_embeddings(text_ids)
        
        # 3. Multimodal Fusion
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # 4. Auto-regressive Generation Loop
        generated_ids = []
        for _ in range(max_new_tokens):
            # Pass continuous fused embeddings through standard transformer forward
            logits, _, _ = model(idx=None, inputs_embeds=inputs_embeds)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated_ids.append(next_token.item())
            
            # Stop generation if EOS or Special token hit
            if next_token.item() in tokenizer.inverse_special:
                if next_token.item() in [0, 2]: # <|endoftext|> or <|im_end|>
                    break
                    
            # Auto-regressive append
            next_embed = model.tok_embeddings(next_token.unsqueeze(0))
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            
    if was_training:
        model.train()
        
    return tokenizer.decode(generated_ids)
