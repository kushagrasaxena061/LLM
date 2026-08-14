import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: int = 16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        if getattr(self.original, 'bias', None) is not None:
            self.original.bias.requires_grad = False
            
        self.lora_A = nn.Parameter(torch.zeros(rank, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self): return self.original.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return base_out + lora_out * self.scaling

LoRALinear = LoRALayer

def inject_lora_to_model(model, rank: int = 4, alpha: int = 16, target_modules=['w_q', 'w_v']):
    # Completely freeze all base model parameters FIRST
    for param in model.parameters():
        param.requires_grad = False
        
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model
