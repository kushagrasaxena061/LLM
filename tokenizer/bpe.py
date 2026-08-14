# tokenizer/bpe.py
"""Lossless Byte-Pair Encoding (BPE) Tokenizer with special token handling."""

import json
import re
from typing import List, Dict

class BPETokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.target_vocab_size = vocab_size
        self.special_tokens = {
            "<|im_start|>": 0,
            "<|im_end|>": 1,
            "<|pad|>": 2,
            "<|unk|>": 3,
            "<|endoftext|>": 4
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges: Dict[tuple, int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self._init_vocab()

    def _init_vocab(self):
        offset = len(self.special_tokens)
        for i in range(256):
            b = bytes([i])
            self.vocab[offset + i] = b
            self.inverse_vocab[b] = offset + i

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def train(self, text: str):
        offset = len(self.special_tokens)
        num_merges = max(0, self.target_vocab_size - 256 - offset)
        if num_merges <= 0: return

        raw_bytes = text.encode("utf-8")
        tokens = [offset + b for b in raw_bytes]

        for _ in range(num_merges):
            counts = {}
            for pair in zip(tokens, tokens[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts: break
            pair = max(counts, key=counts.get)
            idx = offset + 256 + len(self.merges)
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.merges[pair] = idx
            
            p0 = self.vocab.get(pair[0], b"")
            p1 = self.vocab.get(pair[1], b"")
            self.vocab[idx] = p0 + p1
            self.inverse_vocab[p0 + p1] = idx

    def encode(self, text: str) -> List[int]:
        if not text: return []
        special_pattern = "(" + "|".join(map(re.escape, self.special_tokens.keys())) + ")"
        parts = re.split(special_pattern, text)
        
        tokens = []
        for part in parts:
            if not part: continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_normal(part))
        return tokens

    def _encode_normal(self, text: str) -> List[int]:
        offset = len(self.special_tokens)
        raw_bytes = text.encode("utf-8")
        tokens = [offset + b for b in raw_bytes]
        
        while len(tokens) >= 2:
            pairs = list(zip(tokens, tokens[1:]))
            candidate_pairs = [p for p in pairs if p in self.merges]
            if not candidate_pairs: break
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            new_idx = self.merges[best_pair]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(new_idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        byte_chunks = []
        for tid in token_ids:
            if tid in self.inverse_special_tokens:
                byte_chunks.append(self.inverse_special_tokens[tid].encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
        all_bytes = b"".join(byte_chunks)
        return all_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump({
                "target_vocab_size": self.target_vocab_size,
                "special_tokens": self.special_tokens,
                "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            }, f)

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.target_vocab_size = data["target_vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self._init_vocab()
        for pair, idx in self.merges.items():
            p0 = self.vocab.get(pair[0], b"")
            p1 = self.vocab.get(pair[1], b"")
            self.vocab[idx] = p0 + p1
            self.inverse_vocab[p0 + p1] = idx
