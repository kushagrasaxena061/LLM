import os
import glob
import torch
import subprocess
import sys

# Ensure datasets library is available
try:
    import datasets
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets

from model.config import canonical_151m_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from fine_tuning.sft import run_sft_training

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Initializing LoRA Instruction Fine-Tuning on {device}...")

# 1. Load Tokenizer
tokenizer = BPETokenizer(vocab_size=canonical_151m_config.vocab_size)
tokenizer.load("production_151m_bpe.json")

# 2. Instantiate Model and Load Latest 75,000-Step Base Weights
model = GPT(canonical_151m_config).to(device)
ckpts = sorted(glob.glob("production_checkpoints/step_*.pt"), key=os.path.getmtime)
if not ckpts:
    raise FileNotFoundError("No checkpoint found in production_checkpoints/!")

latest_ckpt = ckpts[-1]
print(f"📦 Loading base pretraining weights from: {latest_ckpt}")
state = torch.load(latest_ckpt, map_location=device, weights_only=False)
model.load_state_dict(state["model_state"] if "model_state" in state else state)

# 3. Stream Instruction Dataset (Alpaca)
print("📥 Streaming Instruction Dataset (Alpaca)...")
ds = datasets.load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
sft_samples = []

for i, row in enumerate(ds):
    if i >= 3000:  # 3,000 pairs provides strong instruction alignment in ~20-30 min
        break
    prompt = row["instruction"]
    if row.get("input"):
        prompt += f"\n{row['input']}"
    sft_samples.append({
        "prompt": prompt,
        "response": row["output"]
    })

# 4. Execute SFT with LoRA and Loss Masking
print(f"🎯 Starting LoRA Fine-Tuning on {len(sft_samples)} conversations...")
tuned_model = run_sft_training(model, tokenizer, sft_samples, device=device, epochs=2)

# 5. Save the Final Assistant Checkpoint
final_path = "production_checkpoints/final_assistant_model.pt"
torch.save({"model_state": tuned_model.state_dict()}, final_path)
print(f"\n🎉 Successfully saved fine-tuned chat model to: {final_path}")
