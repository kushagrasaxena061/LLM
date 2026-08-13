# quantization/quantize.py
"""Post-training quantization utilities for model compression."""

import torch
import torch.nn as nn
from model.transformer import GPT
from utils.logger import get_logger

logger = get_logger(__name__)

class QuantizedLinear(nn.Module):
    """Custom INT8 quantized linear layer for cross-platform compatibility."""
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        weight = original_linear.weight.data
        # Compute per-tensor scale for INT8 quantization [-127, 127]
        max_val = weight.abs().max().item()
        self.scale = max_val / 127.0 if max_val > 0 else 1.0
        
        # Quantize weight to INT8 (1 byte per parameter)
        quantized_weight = torch.clamp(torch.round(weight / self.scale), -128, 127).to(torch.int8)
        self.register_buffer("weight_int8", quantized_weight)
        
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly for accurate computation
        w_dequant = self.weight_int8.to(x.dtype) * self.scale
        return nn.functional.linear(x, w_dequant, self.bias)

def quantize_model_to_int8(model: GPT) -> nn.Module:
    """
    Applies custom post-training INT8 weight quantization to linear layers,
    slashing memory footprint by 4x across all platforms (CPU, MPS, CUDA).
    """
    logger.info("Initiating Post-Training INT8 Weight Quantization...")
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            q_linear = QuantizedLinear(module)
            setattr(parent, child_name, q_linear)
            
    logger.info("Quantization complete successfully.")
    return model

def get_model_size_mb(model: nn.Module) -> float:
    """Calculates the physical memory size of a model's state dictionary in Megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb
