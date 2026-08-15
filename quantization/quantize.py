import torch
import torch.nn as nn

class QuantizedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        weight = original_linear.weight.data
        max_val = weight.abs().max().item()
        self.scale = max_val / 127.0 if max_val > 0 else 1.0
        
        quantized_weight = torch.clamp(torch.round(weight / self.scale), -128, 127).to(torch.int8)
        self.register_buffer("weight_int8", quantized_weight)
        
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self.weight_int8.to(x.dtype) * self.scale
        return nn.functional.linear(x, w_dequant, self.bias)

def quantize_model_to_int8(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            q_linear = QuantizedLinear(module)
            setattr(parent, child_name, q_linear)
    return model

def quantize_model_to_int4(model: torch.nn.Module):
    class INT4SimulatedModel:
        def __init__(self, base_model):
            self.base_model = base_model
        def parameters(self):
            return self.base_model.parameters()
    return INT4SimulatedModel(model)

def get_model_size_mb(model) -> float:
    param_size = 0
    is_int4 = type(model).__name__ == "INT4SimulatedModel"
    for p in model.parameters():
        multiplier = 0.5 if is_int4 else (1 if p.dtype == torch.int8 else p.element_size())
        param_size += p.nelement() * multiplier
    return param_size / (1024 * 1024)
