# train.py
"""Master execution script for end-to-end LLM pretraining."""

import os
import torch
from configs.base_config import env_config
from utils.seed import set_seed
from utils.logger import get_logger

from tokenizer.bpe import BPETokenizer
from data.dataset import create_dataloader
from model.config import GPTConfig
from model.transformer import GPT
from training.trainer import LLMTrainer
from training.scheduler import CosineWarmupScheduler
from training.loop import train_model
from inference.generate import generate_text

logger = get_logger("train_pipeline")

def main():
    # 1. Initialize hardware and reproducibility
    set_seed(env_config.seed)
    device = env_config.device
    logger.info("Starting Pretraining Pipeline", device=device)

    # 2. Prepare the Data and Tokenizer
    data_path = "data/shakespeare.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Please download the dataset to {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # We use a vocab of 512 for this small dataset to ensure high token efficiency
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(raw_text)

    # 3. Define the Model Architecture
    # This is a "Nano" config (~10M parameters) tailored for the 1MB Shakespeare text.
    # To scale to 100M parameters, simply increase d_model=768, n_layers=12, n_heads=12.
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1
    )
    
    model = GPT(config)

    # 4. Create the Data Stream
    dataloader = create_dataloader(
        text_path=data_path,
        tokenizer=tokenizer,
        context_length=config.context_length,
        batch_size=32,
        shuffle=True
    )

    # 5. Initialize the Training Infrastructure
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # We will train for exactly 500 steps so you can see results in a few minutes
    max_steps = 500 
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=50, # 10% warmup
        max_steps=max_steps,
        max_lr=1e-3,
        min_lr=1e-4
    )
    
    trainer = LLMTrainer(model=model, optimizer=optimizer, device=device)

    # 6. View Untrained Output (Garbage)
    logger.info("--- UNTRAINED MODEL GENERATION ---")
    untrained_text = generate_text(model, tokenizer, "O Romeo, ", max_new_tokens=20, device=device)
    print(f"\n{untrained_text}\n")

    # 7. EXECUTE TRAINING
    logger.info("--- BEGINNING PRETRAINING ---")
    train_model(
        trainer=trainer,
        dataloader=dataloader,
        scheduler=scheduler,
        max_steps=max_steps,
        eval_interval=250, # Save checkpoint twice
        checkpoint_dir="checkpoints/nano_shakespeare"
    )

    # 8. View Trained Output (Coherent)
    logger.info("--- TRAINED MODEL GENERATION ---")
    trained_text = generate_text(model, tokenizer, "O Romeo, ", max_new_tokens=50, device=device, temperature=0.8)
    print(f"\n{trained_text}\n")

if __name__ == "__main__":
    main()
