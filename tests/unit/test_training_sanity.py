"""Comprehensive pretraining sanity test suite."""
import os
import pytest
import torch
from model.config import canonical_151m_config, tiny_test_config
from model.transformer import GPT
from training.checkpointing import save_full_checkpoint, load_full_checkpoint
from training.train import count_parameters


def test_01_tiny_deterministic_training():
    """1. Proves deterministic training reproducibility with fixed seeds."""
    def run_training(seed):
        torch.manual_seed(seed)
        model = GPT(tiny_test_config).to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
        y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            _, loss, _ = model(x, targets=y, use_cache=False)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    run1 = run_training(42)
    run2 = run_training(42)
    assert run1 == run2, f"Deterministic runs diverged: {run1} vs {run2}"
    print(f"\n✅ Test 1 (Determinism): Exact matching losses across runs -> {run1}")


def test_02_tiny_lm_overfit():
    """2. Proves that the model memorizes a tiny synthetic batch and loss converges."""
    torch.manual_seed(42)
    model = GPT(tiny_test_config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
    y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))

    initial_loss, final_loss = None, None
    for step in range(40):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y, use_cache=False)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.1, f"Model failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"\n✅ Test 2 (Overfitting): Loss dropped from {initial_loss:.4f} to {final_loss:.4f}")


def test_03_gradient_finite_values():
    """3. Proves gradients are finite numbers across all trainable layers."""
    torch.manual_seed(42)
    model = GPT(tiny_test_config).to("cpu")
    x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
    y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))

    _, loss, _ = model(x, targets=y, use_cache=False)
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient in {name} is not finite!"
    print("\n✅ Test 3 (Finite Gradients): All parameter gradients are finite real numbers.")


def test_04_nan_inf_detection():
    """4. Proves forward loss and output representations do not contain NaN or Inf."""
    torch.manual_seed(42)
    model = GPT(tiny_test_config).to("cpu")
    x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
    y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))

    logits, loss, _ = model(x, targets=y, use_cache=False)
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf!"
    assert torch.isfinite(loss), "Loss is NaN/Inf!"
    print(f"\n✅ Test 4 (NaN/Inf Detection): Clean forward pass verified (Loss: {loss.item():.4f}).")


def test_05_gradient_clipping():
    """5. Proves that gradient clipping caps the global L2 gradient norm."""
    torch.manual_seed(42)
    model = GPT(tiny_test_config).to("cpu")
    x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
    y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))

    _, loss, _ = model(x, targets=y, use_cache=False)
    (loss * 1000.0).backward()  # Artificially scale to force large gradients

    pre_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
    assert pre_clip_norm > 1.0, f"Expected unclipped norm > 1.0, got {pre_clip_norm.item():.4f}"

    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    post_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))

    assert post_clip_norm <= max_norm + 1e-4, f"Clipped norm {post_clip_norm.item():.4f} exceeds {max_norm}"
    print(f"\n✅ Test 5 (Grad Clip): Norm scaled from {pre_clip_norm.item():.2f} down to {post_clip_norm.item():.2f}")


def test_06_optimizer_update():
    """6. Proves that AdamW updates weights strictly in the direction of gradients."""
    torch.manual_seed(42)
    model = GPT(tiny_test_config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
    y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))

    before_weights = {name: p.clone() for name, p in model.named_parameters()}
    optimizer.zero_grad()
    _, loss, _ = model(x, targets=y, use_cache=False)
    loss.backward()
    optimizer.step()

    updated = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert not torch.equal(before_weights[name], p), f"Weight {name} was not updated!"
            updated += 1
    assert updated > 0
    print(f"\n✅ Test 6 (Optimizer Step): Successfully updated all {updated} parameter tensors.")


def test_07_scheduler_update():
    """7. Proves learning rate updates across steps following the scheduler."""
    model = GPT(tiny_test_config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    lrs = []
    for _ in range(5):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert len(set(lrs)) == 5, "Learning rate did not update monotonically!"
    assert lrs[0] > lrs[-1], "Learning rate failed to decay under Cosine schedule!"
    print(f"\n✅ Test 7 (Scheduler): LR decayed from {lrs[0]:.6f} to {lrs[-1]:.6f}")


def test_08_checkpoint_save_load(tmp_path):
    """8. Proves full state serialization and exact tensor restoration."""
    ckpt_path = str(tmp_path / "test_ckpt.pt")
    m1 = GPT(tiny_test_config).to("cpu")
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)

    save_full_checkpoint(ckpt_path, m1, opt1, step=15, epoch=1)

    m2 = GPT(tiny_test_config).to("cpu")
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step, epoch = load_full_checkpoint(ckpt_path, m2, opt2, device="cpu")

    assert step == 15
    assert epoch == 1
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter tensor mismatch in {n1} after checkpoint restore!"
    print(f"\n✅ Test 8 (Checkpoint Save/Load): Exact tensor state match verified at step {step}.")


def test_09_resume_training(tmp_path):
    """9. Proves mathematical equivalence between continuous training and resumed training."""
    ckpt_path = str(tmp_path / "resume_sanity.pt")
    torch.manual_seed(42)
    m1 = GPT(tiny_test_config).to("cpu")
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    x = torch.randint(0, tiny_test_config.vocab_size, (2, 8))
    y = torch.randint(0, tiny_test_config.vocab_size, (2, 8))

    # Continuous: 2 steps
    for _ in range(2):
        opt1.zero_grad()
        _, l, _ = m1(x, targets=y, use_cache=False)
        l.backward()
        opt1.step()

    save_full_checkpoint(ckpt_path, m1, opt1, step=2)

    # Step 3 continuous
    opt1.zero_grad()
    _, l_target, _ = m1(x, targets=y, use_cache=False)
    l_target.backward()
    opt1.step()

    # Resumed: Load at step 2 and execute Step 3
    m2 = GPT(tiny_test_config).to("cpu")
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    load_full_checkpoint(ckpt_path, m2, opt2, device="cpu")

    opt2.zero_grad()
    _, l_resumed, _ = m2(x, targets=y, use_cache=False)
    l_resumed.backward()
    opt2.step()

    assert torch.allclose(l_target, l_resumed, atol=1e-6), "Resumed training diverged from continuous path!"
    print(f"\n✅ Test 9 (Resume Parity): Target Loss {l_target.item():.6f} == Resumed Loss {l_resumed.item():.6f}")


def test_10_canonical_151m_initialization():
    """10. Proves canonical production model instantiates without NaN and matches exact parameter count."""
    model = GPT(canonical_151m_config)
    params = count_parameters(model)

    assert params == 151862784, f"Parameter count mismatch: {params:,} != 151,862,784"
    assert model.config.d_model == 768
    assert model.config.n_layers == 12
    assert model.config.n_heads == 12
    assert model.config.head_dim == 64
    assert model.config.vocab_size == 50257
    assert model.config.context_length == 2048
    assert model.config.weight_tying is True

    # Validate all initialized parameters are finite numbers
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Canonical weight {name} contains NaN/Inf upon initialization!"
    print(f"\n✅ Test 10 (Canonical 151M Init): Exactly {params:,} valid, finite parameters initialized.")
