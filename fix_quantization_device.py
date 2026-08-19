import os

quant_path = "quantization/quantize.py"
if os.path.exists(quant_path):
    with open(quant_path, "r") as f:
        code = f.read()
        
    # Wrap quantization to safely handle MPS/CUDA devices by moving to CPU first
    new_quant_code = """"""\"Post-Training Dynamic Quantization for INT8 Model Compression.\"\"\"
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)

def quantize_model_to_int8(model: nn.Module) -> nn.Module:
    \"\"\"
    Applies dynamic INT8 quantization to Linear layers.
    Note: PyTorch quantization requires weights to reside on CPU during pre-packing.
    \"\"\"
    logger.info("Initiating Post-Training Dynamic Quantization (FP32 -> INT8)...")
    
    # Store original device to restore later if needed
    original_device = next(model.parameters()).device
    model.to("cpu")
    model.eval()
    
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    quantized_model.to(original_device)
    logger.info("Quantization complete successfully.")
    return quantized_model
"""
    with open(quant_path, "w") as f:
        f.write(new_quant_code)
    print("✅ Successfully patched quantization/quantize.py for CPU/MPS device safety!")
else:
    print("⚠️ quantization/quantize.py not found.")
