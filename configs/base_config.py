# 2. configs/base_config.py

# Where it goes: Inside the configs/ folder.
# Why it exists: We need a single, central place to define hardware settings.
# Hardcoding "cpu" or "cuda" everywhere causes massive bugs. 
# We use a library called pydantic to enforce strict data types.

# configs/base_config.py
"""Global environment and hardware configurations."""

import torch
from pydantic import BaseModel, Field

# Pydantic is a data validation library. It ensures our configs are strict and type-safe.


def get_device() -> str:
    """
    Automatically detects the best hardware available on the machine.
    """
    # 1. Check for NVIDIA GPUs (Windows/Linux)
    if torch.cuda.is_available():
        return "cuda"
    # 2. Check for Apple Silicon GPUs (M1/M2/M3 MacBooks)
    # MPS stands for Metal Performance Shaders. It allows PyTorch to use your Mac's GPU.
    elif torch.backends.mps.is_available():
        return "mps"
    # 3. Fallback to standard CPU if no GPU is found
    return "cpu"

class EnvironmentConfig(BaseModel):
    """
    Defines the global settings for our entire LLM project.
    By inheriting from BaseModel, Pydantic will automatically validate these types.
    """
    # We use 42 as a standard global seed. Changing this changes the random numbers generated.
    seed: int = Field(default=42, description="Global random seed for reproducibility")
    
    # We dynamically assign the device using the function we wrote above.
    device: str = Field(
        default_factory=get_device,
        description="Target hardware device"
    )
    
    # Mixed precision uses 16-bit math instead of 32-bit math to save memory.
    mixed_precision: bool = Field(
        default=True, description="Enable Automatic Mixed Precision (AMP)"
    )

# We create a single 'global instance' of this config. 
# Any other file in our project will import 'env_config' to check the hardware status.
env_config = EnvironmentConfig()
