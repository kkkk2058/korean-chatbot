from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self):
        self.tokenizer = None

    def train(self, paths: list[str], vocab_size: int):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        )
        self.tokenizer.train(paths, trainer)  # 파일에서 읽으면서 학습

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        self.tokenizer.save(path)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        # self.tokenizer.decoder = ByteLevelDecoder()

    def tokenize(self, text: str) -> list[str]:
        # 글자를 숫자 ID가 아니라, 쪼개진 글자 조각(Token) 문자열 형태로 반환
        return self.tokenizer.encode(text).tokens
