import torch
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def save_checkpoint(model, optimizer, step: int, loss: float, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved: {filepath} at step {step}")

def load_checkpoint(filepath: str, model, optimizer=None, device='cpu'):
    if not os.path.exists(filepath): return 0
    state = torch.load(filepath, map_location=device)
    model.load_state_dict(state['model_state'])
    if optimizer and 'optimizer_state' in state:
        optimizer.load_state_dict(state['optimizer_state'])
    logger.info(f"Checkpoint loaded: {filepath}")
    return state.get('step', 0)
