import os
import torch
import pytest
from model.config import GPTConfig
from model.transformer import GPT
from training.checkpoint import save_checkpoint, load_checkpoint

def get_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y

def test_checkpoint_determinism():
    config = GPTConfig(vocab_size=300, context_length=32, d_model=32, n_layers=2, n_heads=2)
    device = "cpu"
    batch_size, seq_len = 2, 16
    
    # ==========================================
    # 1. CONTINUOUS RUN (Ground Truth)
    # ==========================================
    torch.manual_seed(42)
    model_c = GPT(config).to(device)
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)
    
    continuous_losses = []
    for _ in range(5):
        x, y = get_batch(batch_size, seq_len, config.vocab_size, device)
        _, loss, _ = model_c(x, targets=y)
        loss.backward()
        opt_c.step()
        opt_c.zero_grad()
        continuous_losses.append(loss.item())
        
    # ==========================================
    # 2. INTERRUPTED RUN
    # ==========================================
    torch.manual_seed(42) # Reset exactly to the same initial state
    model_i = GPT(config).to(device)
    opt_i = torch.optim.AdamW(model_i.parameters(), lr=1e-3)
    
    interrupted_losses = []
    for _ in range(3):
        x, y = get_batch(batch_size, seq_len, config.vocab_size, device)
        _, loss, _ = model_i(x, targets=y)
        loss.backward()
        opt_i.step()
        opt_i.zero_grad()
        interrupted_losses.append(loss.item())
        
    # Save Checkpoint at Step 3
    ckpt_path = "temp_determinism_ckpt.pt"
    save_checkpoint(model_i, opt_i, step=3, filepath=ckpt_path)
    
    # ==========================================
    # 3. RESUMED RUN
    # ==========================================
    # Force a completely different unseeded state to prove checkpoint restoration works
    torch.manual_seed(999) 
    model_r = GPT(config).to(device)
    opt_r = torch.optim.AdamW(model_r.parameters(), lr=1e-3)
    
    step = load_checkpoint(ckpt_path, model_r, opt_r, map_location=device)
    assert step == 3
    
    # Continue for remaining 2 steps
    for _ in range(3, 5):
        x, y = get_batch(batch_size, seq_len, config.vocab_size, device)
        _, loss, _ = model_r(x, targets=y)
        loss.backward()
        opt_r.step()
        opt_r.zero_grad()
        interrupted_losses.append(loss.item())
        
    # ==========================================
    # 4. VERIFY EXACT MATHEMATICAL MATCH
    # ==========================================
    for i, (c, r) in enumerate(zip(continuous_losses, interrupted_losses)):
        assert abs(c - r) < 1e-6, f"Mismatch at step {i}: Continuous={c}, Resumed={r}"
        
    # 5. Verify Error Handling Edge Cases
    with pytest.raises(FileNotFoundError):
        load_checkpoint("missing_file_xyz.pt", model_r)
        
    with open("corrupt_file_xyz.pt", "w") as f:
        f.write("corrupt data")
    with pytest.raises(RuntimeError):
        load_checkpoint("corrupt_file_xyz.pt", model_r)
        
    # Cleanup
    os.remove(ckpt_path)
    os.remove("corrupt_file_xyz.pt")
    print("\n✅ Checkpoint determinism verified! RNG states, model weights, and optimizer safely resumed.")

if __name__ == "__main__":
    test_checkpoint_determinism()
