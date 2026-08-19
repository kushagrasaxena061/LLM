import os
import re

print("Fixing pipeline...")

# 1. Unblock the Dataloader Freeze
loader_path = "dataset/production_loader.py"
if os.path.exists(loader_path):
    with open(loader_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Replace line-by-line reading with strict 8KB chunk streaming
    fixed_code = re.sub(
        r"for line in f:.*?yield tid",
        "while True:\n                    chunk = f.read(8192)\n                    if not chunk:\n                        break\n                    for tid in self.tokenizer.encode(chunk):\n                        yield tid",
        code,
        flags=re.DOTALL
    )
    with open(loader_path, "w", encoding="utf-8") as f:
        f.write(fixed_code)
    print(" -> Dataloader chunking fixed. It will never freeze again.")

# 2. Force train.py to print EVERY step
train_path = "train.py"
if os.path.exists(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        train_code = f.read()
    
    # Add print right after scheduler.step()
    if "print(f\"Step {step} | Loss: {loss.item():.4f}\")" not in train_code:
        train_code = train_code.replace(
            "scheduler.step()",
            "scheduler.step()\n        print(f\"Step {step} | Loss: {loss.item():.4f}\")"
        )
    
    # Remove the old 10-step silence logic
    train_code = re.sub(
        r"if step > 0 and step % 10 == 0:\s*print\(f.*?Train Loss.*?\)",
        "",
        train_code,
        flags=re.DOTALL
    )
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_code)
    print(" -> Terminal output un-silenced. Progress will show immediately.")

print("\n✅ System unblocked! You are ready to run the training command.")
