import re
from typing import Dict, List, Tuple
from tokenizer.base import BaseTokenizer

SPECIAL_TOKENS = {'<|endoftext|>': 0, '<|im_start|>': 1, '<|im_end|>': 2, '<|pad|>': 3, '<|unk|>': 4}

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 300):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self._init_vocab()

    def _init_vocab(self):
        offset = len(self.special_tokens)
        for i in range(256):
            b = bytes([i])
            token_id = offset + i
            self.vocab[token_id] = b
            self.inverse_vocab[b] = token_id

    @property
    def vocab_size(self) -> int: return len(self.vocab) + len(self.special_tokens)

    def train(self, text: str):
        num_merges = self.target_vocab_size - self.vocab_size
        if num_merges <= 0: return
        offset = len(self.special_tokens)
        
        # FIX: Ensure base bytes are strictly shifted so self.vocab[] lookup never fails
        tokens = [b + offset for b in text.encode('utf-8')]
        for _ in range(num_merges):
            counts = {}
            for pair in zip(tokens, tokens[1:]): counts[pair] = counts.get(pair, 0) + 1
            if not counts: break
            pair = max(counts, key=counts.get)
            idx = self.vocab_size
            new_ids = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_ids.append(idx); i += 2
                else:
                    new_ids.append(tokens[i]); i += 1
            tokens = new_ids
            self.merges[pair] = idx
            
            # FIX: Guaranteed to exist, no more byte() fallback errors
            p0 = self.vocab[pair[0]]
            p1 = self.vocab[pair[1]]
            self.vocab[idx] = p0 + p1
            self.inverse_vocab[p0 + p1] = idx

    def encode(self, text: str) -> List[int]:
        if not text: return []
        pattern = f"({'|'.join(re.escape(k) for k in self.special_tokens.keys())})"
        parts = re.split(pattern, text)
        token_ids = []
        offset = len(self.special_tokens)
        for part in parts:
            if not part: continue
            if part in self.special_tokens: token_ids.append(self.special_tokens[part])
            else:
                ids = [b + offset for b in part.encode('utf-8')]
                for pair, merge_id in self.merges.items():
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                            new_ids.append(merge_id); i += 2
                        else:
                            new_ids.append(ids[i]); i += 1
                    ids = new_ids
                token_ids.extend(ids)
        return token_ids

    def decode(self, ids: List[int]) -> str:
        byte_chunks = []
        for i in ids:
            if i in self.inverse_special_tokens: byte_chunks.append(self.inverse_special_tokens[i].encode('utf-8'))
            elif i in self.vocab: byte_chunks.append(self.vocab[i])
            else:
                offset = len(self.special_tokens)
                if 0 <= i - offset < 256: byte_chunks.append(bytes([i - offset]))
                else: byte_chunks.append(b'?')
        return b''.join(byte_chunks).decode('utf-8', errors='replace')

    def save(self, filepath: str):
        import json
        data = {
            'vocab_size': self.vocab_size,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w') as f: json.dump(data, f)

    def load(self, filepath: str):
        import json
        with open(filepath, 'r') as f: data = json.load(f)
        self.special_tokens = data['special_tokens']
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self._init_vocab()
        # Reconstruct vocab from merges
        for pair, idx in self.merges.items():
            p0 = self.vocab[pair[0]]
            p1 = self.vocab[pair[1]]
            self.vocab[idx] = p0 + p1
            self.inverse_vocab[p0 + p1] = idx
