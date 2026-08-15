import os
import torch
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_full_checkpoint(filepath: str, model, optimizer, scheduler=None, scaler=None, step: int = 0, epoch: int = 0):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    mps_rng = None
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try: mps_rng = torch.mps.get_rng_state()
        except Exception: pass

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_mps": mps_rng
    }
    tmp_filepath = filepath + ".tmp"
    torch.save(state, tmp_filepath)
    os.replace(tmp_filepath, filepath)
    logger.info(f"Checkpoint safely saved: {filepath} at step {step}")

def load_full_checkpoint(filepath: str, model, optimizer, scheduler=None, scaler=None, device="cpu"):
    if not os.path.exists(filepath): return 0, 0
    state = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"): scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"): scaler.load_state_dict(state["scaler"])
    
    if "rng_python" in state: random.setstate(state["rng_python"])
    if "rng_numpy" in state: np.random.set_state(state["rng_numpy"])
    if "rng_torch" in state: torch.set_rng_state(state["rng_torch"].cpu().to(torch.uint8))
    
    if torch.cuda.is_available() and state.get("rng_cuda"):
        try: torch.cuda.set_rng_state_all(state["rng_cuda"])
        except Exception: pass
            
    if hasattr(torch, "mps") and torch.backends.mps.is_available() and state.get("rng_mps") is not None:
        try: torch.mps.set_rng_state(state["rng_mps"])
        except Exception: pass
            
    logger.info(f"Checkpoint loaded: {filepath}")
    return state.get("step", 0), state.get("epoch", 0)

def save_checkpoint(model, optimizer, step=0, loss=0.0, filepath="checkpoint.pt", **kwargs):
    return save_full_checkpoint(filepath, model, optimizer, step=step)

def load_checkpoint(filepath, model, optimizer, device="cpu"):
    step, _ = load_full_checkpoint(filepath, model, optimizer, device=device)
    return step
