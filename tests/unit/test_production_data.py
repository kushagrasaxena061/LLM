"""Unit tests for production sharded streaming dataset and sequence packing."""

import os
import pytest
import torch
from tokenizer.bpe import BPETokenizer
from dataset.production_loader import ShardedTextStreamingDataset, MemoryMappedDataset
from dataset.splitter import deterministic_hash_split

@pytest.fixture
def dummy_tokenizer():
    tok = BPETokenizer(vocab_size=300)
    tok.train("The quick brown fox jumps over the lazy dog.")
    return tok

@pytest.fixture
def temp_shards(tmp_path):
    shard1 = tmp_path / "shard_001.txt"
    shard2 = tmp_path / "shard_002.txt"
    shard1.write_text("Hello world from shard one. Pretraining streaming dataset test.\n", encoding="utf-8")
    shard2.write_text("Hello world from shard two. Scalable token packing verification.\n", encoding="utf-8")
    return [str(shard1), str(shard2)]

def test_tiny_dataset_compatibility(dummy_tokenizer):
    texts = ["Small educational sample text for testing unit compatibility."]
    dataset = MemoryMappedDataset(texts, dummy_tokenizer, context_length=8)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.shape == (8,)

def test_sequence_packing_and_alignment(dummy_tokenizer, temp_shards):
    dataset = ShardedTextStreamingDataset(
        shard_paths=temp_shards,
        tokenizer=dummy_tokenizer,
        context_length=10,
        shuffle_shards=False
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch_x, batch_y = next(iter(loader))
    
    assert batch_x.shape == (2, 10)
    assert batch_y.shape == (2, 10)
    assert torch.equal(batch_y[0, :-1], batch_x[0, 1:])

def test_deterministic_shard_iteration(dummy_tokenizer, temp_shards):
    ds1 = ShardedTextStreamingDataset(temp_shards, dummy_tokenizer, context_length=5, seed=123)
    ds2 = ShardedTextStreamingDataset(temp_shards, dummy_tokenizer, context_length=5, seed=123)
    
    samples1 = list(ds1)
    samples2 = list(ds2)
    
    assert len(samples1) == len(samples2)
    for (x1, y1), (x2, y2) in zip(samples1, samples2):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

def test_train_val_shard_separation(temp_shards):
    train_shards, val_shards = deterministic_hash_split(temp_shards, val_ratio=0.5, salt="pretrain_test")
    
    intersection = set(train_shards).intersection(set(val_shards))
    assert len(intersection) == 0, "Shard leakage detected between train and val splits!"
    assert len(train_shards) + len(val_shards) == len(temp_shards)

def test_memory_efficient_large_file_behavior(tmp_path, dummy_tokenizer):
    large_file = tmp_path / "large_shard.txt"
    with open(large_file, "w", encoding="utf-8") as f:
        for i in range(500):
            f.write(f"Line number {i} with repeatable text tokens for validation.\n")
            
    dataset = ShardedTextStreamingDataset([str(large_file)], dummy_tokenizer, context_length=16)
    count = 0
    for x, y in dataset:
        count += 1
        if count >= 5:
            break
    assert count == 5
