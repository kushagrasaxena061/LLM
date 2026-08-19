# fine_tuning/sft.py
"""Supervised Fine-Tuning (SFT) pipeline with prompt loss masking."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from tokenizer.base import BaseTokenizer
from model.transformer import GPT
from fine_tuning.inject import inject_lora_to_model
from utils.logger import get_logger

logger = get_logger(__name__)

class InstructionDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: BaseTokenizer, context_length: int):
        """
        Prepares prompt-response pairs for instruction tuning with loss masking.
        
        Args:
            data: List of dicts with 'prompt' and 'response' keys.
            tokenizer: The BPE tokenizer.
            context_length: Maximum context length window.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples = []
        
        for item in data:
            prompt_text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n"
            response_text = f"{item['response']}\n<|endoftext|>"
            
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(response_text)
            
            # Combine input sequence
            input_ids = prompt_ids + response_ids
            
            # Create targets: We mask out the prompt tokens using -100 so the model 
            # only learns to generate the response tokens!
            targets = [-100] * len(prompt_ids) + response_ids
            
            # Truncate or pad to context length
            if len(input_ids) < context_length:
                padding_len = context_length - len(input_ids)
                input_ids = input_ids + [0] * padding_len
                targets = targets + [-100] * padding_len
            else:
                input_ids = input_ids[:context_length]
                targets = targets[:context_length]
                
            self.examples.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.long)
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]['input_ids'], self.examples[idx]['targets']


def run_sft_training(model: GPT, tokenizer: BaseTokenizer, dataset_samples: List[Dict[str, str]], device: str, epochs: int = 3):
    """
    Executes LoRA fine-tuning on the injected GPT model.
    """
    logger.info("Injecting LoRA for Supervised Fine-Tuning...")
    model = inject_lora_to_model(model, rank=4, alpha=16)
    model.to(device)
    
    # FIX: Use model.config.context_length instead of hardcoding 128
    dataset = InstructionDataset(dataset_samples, tokenizer, context_length=model.config.context_length)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Only optimize parameters that require gradients (our LoRA weights!)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3.0e-4)
    
    model.train()
    logger.info("Beginning SFT Training Loop...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: model cross_entropy handles -100 masking automatically
            logits, loss, _ = model(x, targets=y)
            
            loss.backward()
            optimizer.step()
            if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
            total_loss += loss.item()
            if step % 5 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
        logger.info(f"Epoch {epoch+1}/{epochs} completed", avg_loss=f"{total_loss / len(dataloader):.4f}")
        
    return model
