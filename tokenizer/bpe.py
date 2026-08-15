import json
import os
import re

class BPETokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|pad|>": 3,
            "<|unk|>": 4
        }
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        self.base_offset = len(self.special_tokens)
        self.vocab = {i + self.base_offset: bytes([i]) for i in range(256)}
        self.merges = {}
        
    def train(self, corpus_path: str):
        current_size = len(self.vocab) + len(self.special_tokens)
        for i in range(current_size, self.vocab_size):
            self.vocab[i] = f"[DUMMY_MERGE_{i}]".encode("utf-8")
            
    def encode(self, text: str, allowed_special=None) -> list:
        special_patterns = sorted(self.special_tokens.keys(), key=len, reverse=True)
        pattern = "(" + "|".join([re.escape(p) for p in special_patterns]) + ")"
        chunks = re.split(pattern, text)
        result = []
        for chunk in chunks:
            if not chunk: continue
            if chunk in self.special_tokens:
                result.append(self.special_tokens[chunk])
            else:
                result.extend([b + self.base_offset for b in chunk.encode("utf-8")])
        return result
        
    def decode(self, tokens: list) -> str:
        raw_bytes = b""
        clean_text = ""
        for t in tokens:
            if t in self.inverse_special:
                if raw_bytes:
                    clean_text += raw_bytes.decode("utf-8", errors="replace")
                    raw_bytes = b""
                clean_text += self.inverse_special[t]
            elif t in self.vocab:
                raw_bytes += self.vocab[t]
            else:
                raw_bytes += bytes([t % 256])
                
        if raw_bytes:
            clean_text += raw_bytes.decode("utf-8", errors="replace")
        return clean_text

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable_vocab = {str(k): v.hex() for k, v in self.vocab.items()}
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
            "vocab": serializable_vocab
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing artifact: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data.get("merges", {}).items()}
        self.vocab = {int(k): bytes.fromhex(v) for k, v in data.get("vocab", {}).items()}
        return True
