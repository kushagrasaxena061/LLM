import os
import pytest
import torch
from model.config import tiny_test_config
from model.transformer import GPT
from training.checkpointing import save_full_checkpoint, load_full_checkpoint

def test_checkpoint_resume_parity(tmp_path):
    device = "cpu"
    ckpt_path = str(tmp_path / "resume_ckpt.pt")
    
    torch.manual_seed(42)
    model1 = GPT(tiny_test_config).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=10)
    
    x = torch.randint(0, 300, (2, 16), device=device)
    y = torch.randint(0, 300, (2, 16), device=device)
    
    for step in range(3):
        optimizer1.zero_grad()
        _, loss1, _ = model1(x, targets=y, use_cache=False)
        loss1.backward()
        optimizer1.step()
        scheduler1.step()
        
    save_full_checkpoint(ckpt_path, model1, optimizer1, scheduler1, step=3, epoch=1)
    
    # FIX: Capture the exact learning rate right when the checkpoint is saved
    expected_lr_at_resume = optimizer1.param_groups[0]["lr"]
    
    optimizer1.zero_grad()
    _, loss_step4_target, _ = model1(x, targets=y, use_cache=False)
    loss_step4_target.backward()
    optimizer1.step()
    scheduler1.step()
    target_lr = optimizer1.param_groups[0]["lr"]
    
    model2 = GPT(tiny_test_config).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
    
    loaded_step, loaded_epoch = load_full_checkpoint(
        ckpt_path, model2, optimizer2, scheduler2, device=device
    )
    
    assert loaded_step == 3
    assert loaded_epoch == 1
    # Check against the captured LR, not the advanced scheduler!
    assert optimizer2.param_groups[0]["lr"] == expected_lr_at_resume
    
    optimizer2.zero_grad()
    _, loss_step4_resumed, _ = model2(x, targets=y, use_cache=False)
    loss_step4_resumed.backward()
    optimizer2.step()
    scheduler2.step()
    resumed_lr = optimizer2.param_groups[0]["lr"]
    
    assert torch.allclose(loss_step4_target, loss_step4_resumed, atol=1e-6), "Loss diverged! Resumed model did not match continuous model."
    assert target_lr == resumed_lr, "Scheduler failed to restore correct LR state."
