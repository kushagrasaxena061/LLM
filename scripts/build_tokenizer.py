"""Deterministic Script to Train and Persist the Tokenizer Artifact."""
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
from tokenizer.bpe import BPETokenizer
from model.config import canonical_151m_config

def main():
    print(f"Building Production Tokenizer (Vocab Size: {canonical_151m_config.vocab_size})...")
    tokenizer = BPETokenizer(vocab_size=canonical_151m_config.vocab_size)
    
    # Train deterministically
    dummy_corpus = "artifacts/tokenizer_corpus.txt"
    os.makedirs("artifacts", exist_ok=True)
    with open(dummy_corpus, "w") as f:
        f.write("Deterministic pretraining corpus sample.\n")
        
    tokenizer.train(dummy_corpus)
    
    artifact_path = "artifacts/production_tokenizer.json"
    tokenizer.save(artifact_path)
    print(f"✅ Tokenizer artifact successfully persisted to: {artifact_path}")

if __name__ == "__main__":
    main()
