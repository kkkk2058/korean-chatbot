"""
HF ByteLevel BPE 토크나이저 — 추론 전용 어댑터.
train_model_v2.ipynb 에서 학습/저장한 tokenizer.json (HuggingFace `tokenizers` 형식) 로드.
engine.py 가 기대하는 인터페이스(vocab / encode / decode)를 그대로 제공해 서빙 코드를 가볍게 유지.

※ 직접 구현한 v1 BPE 토크나이저(vocab/merges/special_tokens 형식)는 tokenizer_custom.py 에 보존.
"""
from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self):
        self.tk = None
        self.vocab = {}          # token(str) -> id(int), 특수토큰 포함

    def load(self, path):
        self.tk = Tokenizer.from_file(path)
        self.vocab = self.tk.get_vocab()

    def encode(self, text):
        return self.tk.encode(text).ids          # list[int]

    def decode(self, ids):
        return self.tk.decode(ids)               # 특수토큰(<pad>/<s>/</s>) 자동 제외
