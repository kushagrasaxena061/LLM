"""Unit tests for deterministic corpus train/validation splitting."""
import pytest
from dataset.splitter import deterministic_hash_split

def test_deterministic_split():
    corpus = [f"document_{i}.txt" for i in range(1000)]
    train1, val1 = deterministic_hash_split(corpus, val_ratio=0.1)
    train2, val2 = deterministic_hash_split(corpus, val_ratio=0.1)
    assert train1 == train2, "Split is not deterministic across runs!"
    assert val1 == val2, "Split is not deterministic across runs!"

def test_no_overlap_leakage():
    corpus = [f"document_{i}.txt" for i in range(1000)]
    train, val = deterministic_hash_split(corpus, val_ratio=0.1)
    intersection = set(train).intersection(set(val))
    assert len(intersection) == 0, f"CRITICAL LEAKAGE DETECTED! Overlap: {intersection}"

def test_correct_approximate_ratio():
    corpus = [f"document_{i}.txt" for i in range(10000)]
    train, val = deterministic_hash_split(corpus, val_ratio=0.15)
    ratio = len(val) / len(corpus)
    assert 0.135 <= ratio <= 0.165, f"Split ratio {ratio} deviated too far from target 0.15"

def test_tiny_dataset_works():
    corpus = ["tiny_doc.txt"]
    train, val = deterministic_hash_split(corpus, val_ratio=0.5)
    assert len(train) + len(val) == 1, "Data was lost or duplicated during tiny split!"
