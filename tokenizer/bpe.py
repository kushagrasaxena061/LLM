import os
import json
import re

class BPETokenizer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        # Strictly assign special tokens to the highest indices of the 50,257 vocabulary
        self.special_tokens = {
            "<|unk|>": self.vocab_size - 5,
            "<|pad|>": self.vocab_size - 4,
            "<|im_end|>": self.vocab_size - 3,
            "<|im_start|>": self.vocab_size - 2,
            "<|endoftext|>": self.vocab_size - 1,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        self.merges = {}
        # Base vocabulary is the 256 raw bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
        
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text_or_path: str):
        """Trains genuine BPE merges deterministically."""
        # Accept a real corpus path or fallback to direct text for tiny tests
        if os.path.exists(text_or_path):
            with open(text_or_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_or_path
            
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # Calculate exactly how many merges we are allowed to make
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            
            # Deterministic tie-breaking: highest frequency, then alphabetical by byte
            top_pair = max(stats.keys(), key=lambda p: (stats[p], p))
            
            # Stop if there are no more repeating pairs
            if stats[top_pair] < 2:
                break
                
            idx = 256 + i
            ids = self._merge(ids, top_pair, idx)
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
    def encode(self, text: str) -> list[int]:
        """Encodes text to BPE token IDs, handling special tokens natively."""
        pattern = "|".join(re.escape(k) for k in self.special_tokens.keys())
        if pattern:
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]
            
        out = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                out.append(self.special_tokens[chunk])
            elif chunk:
                # Encode chunk via real BPE merges
                chunk_bytes = chunk.encode("utf-8")
                ids = list(chunk_bytes)
                while len(ids) >= 2:
                    stats = self._get_stats(ids)
                    # Find the pair with the lowest merge index (oldest merge)
                    pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
                    if pair not in self.merges:
                        break # No more valid merges exist
                    ids = self._merge(ids, pair, self.merges[pair])
                out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        """Decodes BPE token IDs safely without fracturing multi-byte UTF-8 sequences."""
        text_bytes = bytearray()
        for idx in ids:
            if idx in self.inverse_special_tokens:
                text_bytes.extend(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                text_bytes.extend(self.vocab[idx])
            else:
                # Fallback for corrupted/unmapped IDs to prevent catastrophic crashes
                text_bytes.extend(b'?')
                
        # decode with errors="replace" guarantees it will never crash on fractional emojis
        return text_bytes.decode("utf-8", errors="replace")
        
    def save(self, file_path: str):
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab_size": self.vocab_size,
            "merges": merges_str
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
            
    def load(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.special_tokens = {
            "<|unk|>": self.vocab_size - 5,
            "<|pad|>": self.vocab_size - 4,
            "<|im_end|>": self.vocab_size - 3,
            "<|im_start|>": self.vocab_size - 2,
            "<|endoftext|>": self.vocab_size - 1,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        for k_str, v in data["merges"].items():
            p0, p1 = map(int, k_str.split(","))
            self.merges[(p0, p1)] = v
            self.vocab[v] = self.vocab[p0] + self.vocab[p1]
