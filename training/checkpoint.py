import os
import torch
import random
import numpy as np

def save_checkpoint(model, optimizer=None, step=0, filepath="checkpoint.pt", scheduler=None, epoch=0, scaler=None, loss=None):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    state = {
        'model_state': model.state_dict(),
        'global_step': step,
        'epoch': epoch,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
    }
    if optimizer:
        state['optimizer_state'] = optimizer.state_dict()
    if scheduler:
        state['scheduler_state'] = scheduler.state_dict()
    if scaler:
        state['scaler_state'] = scaler.state_dict()
    if torch.cuda.is_available():
        state['rng_cuda'] = torch.cuda.get_rng_state_all()
    
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None, map_location='cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    try:
        state = torch.load(filepath, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load corrupt checkpoint: {filepath}. Error: {e}")
    
    model.load_state_dict(state['model_state'])
    
    if optimizer and 'optimizer_state' in state:
        optimizer.load_state_dict(state['optimizer_state'])
    if scheduler and 'scheduler_state' in state:
        scheduler.load_state_dict(state['scheduler_state'])
    if scaler and 'scaler_state' in state:
        scaler.load_state_dict(state['scaler_state'])
        
    if 'rng_python' in state:
        random.setstate(state['rng_python'])
    if 'rng_numpy' in state:
        np.random.set_state(state['rng_numpy'])
    if 'rng_torch' in state:
        torch.set_rng_state(state['rng_torch'])
    if 'rng_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        
    return state.get('global_step', 0)
