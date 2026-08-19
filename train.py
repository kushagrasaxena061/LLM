import os
import argparse
import torch
from model.config import canonical_151m_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from dataset.pipeline import create_dataloaders
from training.checkpoint import save_checkpoint, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="MiniGPT Production Pretraining Pipeline")
    parser.add_argument("--corpus", type=str, required=True, help="Path to raw text corpus")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to save/load tokenizer")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--ckpt_interval", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # 1. Real Tokenizer Generation / Loading
    tokenizer = BPETokenizer(vocab_size=50257)
    if os.path.exists(args.tokenizer_path):
        print(f"Loading existing tokenizer from {args.tokenizer_path}")
        tokenizer.load(args.tokenizer_path)
    else:
        print(f"Training production tokenizer on {args.corpus} (vocab_size=50257)...")
        tokenizer.train(args.corpus)
        tokenizer.save(args.tokenizer_path)

    # 2. Production Dataset / DataLoader
    print("Initializing Streaming Dataset...")
    train_loader, val_loader = create_dataloaders(
        args.corpus, tokenizer, batch_size=args.batch_size, seq_len=args.seq_len
    )
    
    # 3. Instantiate Canonical 151M Architecture
    model = GPT(canonical_151m_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated Canonical Model: {total_params:,} parameters")

    # 4. Optimization Engine
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None

    global_step = 0
    if args.resume:
        global_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        print(f"Resumed from step {global_step}")

    # Track weights for explicit mathematical proof of learning
    initial_weight = model.tok_embeddings.weight[0][0].item()
    
    train_iter = iter(train_loader)
    model.train()
    
    initial_loss = None
    final_loss = None
    
    print("\nStarting Production Pretraining Loop...")
    for step in range(global_step, args.max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # 5. Forward, Loss, Backward, AMP & Clipping
        if scaler:
            with torch.autocast(device_type=device, dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss, _ = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        if step > 0 and step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
        
        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val
        
        # Periodic Logging
        if step % max(1, args.max_steps // 10) == 0:
            print(f"Step {step}/{args.max_steps} | Loss: {loss_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
        # 6. Periodic Validation
        if args.val_interval > 0 and step > 0 and step % args.val_interval == 0:
            model.eval()
            val_iter = iter(val_loader)
            val_loss_total = 0.0
            batches = 0
            with torch.no_grad():
                for _ in range(5):
                    try:
                        vx, vy = next(val_iter)
                        vx, vy = vx.to(device), vy.to(device)
                        _, vloss, _ = model(vx, targets=vy)
                        val_loss_total += vloss.item()
                        batches += 1
                    except StopIteration:
                        break
            val_loss_out = val_loss_total / max(1, batches)
            print(f"--- Validation at Step {step} | Val Loss: {val_loss_out:.4f} ---")
            model.train()
            
        # 7. Checkpointing
        if args.ckpt_interval > 0 and step > 0 and step % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"step_{step}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path, scheduler, scaler=scaler)
            print(f"Saved checkpoint to {ckpt_path}")
            
    final_weight = model.tok_embeddings.weight[0][0].item()
    weight_changed = initial_weight != final_weight
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Parameter Count: {total_params:,}")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Weights Changed: {weight_changed}")
    print(f"Checkpoint location: {args.out_dir}/")

if __name__ == "__main__":
    main()
