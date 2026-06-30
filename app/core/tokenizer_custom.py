"""
BPE 토크나이저 — 추론 전용(로드/인코딩/디코딩).
stage1_from_scratch/src/tokenizer.py 의 BPETokenizer.save() 형식과 호환.
학습 로직은 제외해 서빙 레이어를 가볍게 유지.
"""
import json


class BPETokenizer:
    def __init__(self):
        self.vocab = {}          # token(str) -> id(int)
        self.id_to_token = {}    # id(int)  -> token(str)
        self.merges = []         # [(a, b), ...] 순서 중요
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        self._encode_cache = {}

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.special_tokens = data["special_tokens"]
        self.id_to_token = {int(idx): tok for tok, idx in self.vocab.items()}
        self._encode_cache = {}

    def _tokenize_word(self, word):
        symbols = list(word) + ["</w>"]
        for a, b in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text):
        unk = self.vocab["<unk>"]
        ids = []
        for word in text.strip().split():
            if word not in self._encode_cache:
                self._encode_cache[word] = [
                    self.vocab.get(tok, unk) for tok in self._tokenize_word(word)
                ]
            ids.extend(self._encode_cache[word])
        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
