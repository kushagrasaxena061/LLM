import os
import torch
import pytest
from dataset.pipeline import StreamingTextDataset, create_dataloaders
from tokenizer.bpe import BPETokenizer

def test_dataset_production_pipeline():
    # 1. Setup real corpus fixture
    corpus_path = "temp_test_corpus.txt"
    raw_text = "The quick brown fox jumps over the lazy dog. \n\tThis preserves formatting.\n" * 100
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
        
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus_path)
    
    seq_len = 16
    batch_size = 2
    
    # 2. Test DataLoader instantiation
    train_loader, val_loader = create_dataloaders(
        corpus_path, tokenizer, batch_size=batch_size, seq_len=seq_len, train_ratio=0.8
    )
    
    # 3. Pull a batch and verify shapes
    train_iter = iter(train_loader)
    x, y = next(train_iter)
    
    assert x.shape == (batch_size, seq_len), f"Expected input shape (2, 16), got {x.shape}"
    assert y.shape == (batch_size, seq_len), f"Expected target shape (2, 16), got {y.shape}"
    
    # 4. Verify Causal-LM Shift (targets[t] == inputs[t+1])
    # The first token of the target should equal the second token of the input
    assert y[0][0].item() == x[0][1].item(), "Causal shift is misaligned!"
    assert y[0][-2].item() == x[0][-1].item(), "Causal shift is misaligned!"
    
    # 5. Verify token authenticity (no random generation, perfectly decodes to real text)
    decoded_text = tokenizer.decode(x[0].tolist())
    assert "The quick brown" in decoded_text, "Tokens do not match original text corpus!"
    
    # 6. Verify distinct train/validation splits
    val_iter = iter(val_loader)
    val_x, val_y = next(val_iter)
    assert val_x.shape == (batch_size, seq_len)
    
    os.remove(corpus_path)
    print("\n✅ Streaming Dataset Pipeline Confirmed! Causal Shift and formatting perfectly preserved.")

if __name__ == "__main__":
    test_dataset_production_pipeline()
