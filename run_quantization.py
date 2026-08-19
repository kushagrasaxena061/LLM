import torch
from model.config import canonical_151m_config
from model.transformer import GPT
from quantization.quantize import quantize_model_to_int8

print("⚙️ Loading and Merging fine-tuned model for INT8 Quantization...")
model = GPT(canonical_151m_config)
state = torch.load("production_checkpoints/final_assistant_model.pt", map_location="cpu", weights_only=False)
state_dict = state["model_state"] if "model_state" in state else state

# 1. Merge LoRA weights permanently back into the base model weights
merged_state = {}
for key in list(state_dict.keys()):
    # Identify LoRA A matrices
    if key.endswith('.lora_A'):
        prefix = key.replace('.lora_A', '')
        lora_A = state_dict[prefix + '.lora_A']
        lora_B = state_dict[prefix + '.lora_B']
        
        # Retrieve the original frozen weight (accommodating different naming conventions)
        if prefix + '.original.weight' in state_dict:
            base_weight = state_dict[prefix + '.original.weight']
        else:
            base_weight = state_dict[prefix + '.weight']
        
        # Mathematically merge: W = W + (B @ A) * (alpha / rank)
        # We used rank=4 and alpha=16 during training, making our scaling factor 4.0
        merged_weight = base_weight + (lora_B @ lora_A) * 4.0
        merged_state[prefix + '.weight'] = merged_weight
        
    # Skip LoRA B and original weights since we just mathematically combined them
    elif key.endswith('.lora_B') or key.endswith('.original.weight'):
        continue
    elif key.endswith('.weight') and key.replace('.weight', '.lora_A') in state_dict:
        continue
    else:
        # Keep all other standard base model parameters untouched
        merged_state[key] = state_dict[key]

# 2. Load the cleanly merged FP32 state dict into the standard GPT base model
model.load_state_dict(merged_state)
print("✅ LoRA weights successfully baked into base architecture!")

# 3. Quantize linear layers to INT8
quantized_model = quantize_model_to_int8(model)

# 4. Save quantized artifact
torch.save(quantized_model.state_dict(), "production_checkpoints/quantized_assistant_int8.pt")
print("✅ INT8 Quantized model saved to: production_checkpoints/quantized_assistant_int8.pt")
