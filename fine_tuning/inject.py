# fine_tuning/inject.py
"""Utility to inject LoRA layers into an existing GPT model architecture."""

import torch.nn as nn
from model.transformer import GPT
from fine_tuning.lora import LoRALinear
from utils.logger import get_logger

logger = get_logger(__name__)

def inject_lora_to_model(model: GPT, rank: int = 4, alpha: float = 16, target_modules: list = ["qkv_proj", "out_proj"]):
    """
    Recursively replaces target linear layers in the GPT model with LoRALinear wrappers
    and freezes all other base model parameters.
    
    Args:
        model (GPT): The pretrained base GPT model.
        rank (int): LoRA rank dimension.
        alpha (float): LoRA scaling factor.
        target_modules (list): Names of linear layers to adapt.
    """
    # 1. Freeze ALL parameters in the base model first
    for param in model.parameters():
        param.requires_grad = False

    injected_count = 0
    
    # 2. Recursively iterate through all named modules in the model
    for name, module in model.named_modules():
        # Check if the module is a Linear layer and matches our target names
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            # Find the parent module so we can replace the attribute
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Wrap the original linear layer with LoRA
            lora_wrapped = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_wrapped)
            injected_count += 1
            
    logger.info("LoRA injected successfully", injected_layers=injected_count, rank=rank)
    return model
