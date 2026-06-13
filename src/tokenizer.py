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
        self.tokenizer.decoder = ByteLevelDecoder()

    def tokenize(self, text: str) -> list[str]:
        # 글자를 숫자 ID가 아니라, 쪼개진 글자 조각(Token) 문자열 형태로 반환
        return self.tokenizer.encode(text).tokens

# import json


# class BPETokenizer:
#     def __init__(self):
#         self.token_to_id = {
#             "[PAD]": 0,
#             "[BOS]": 1,
#             "[EOS]": 2,
#             "[UNK]": 3,
#         }
#         self.id_to_token = {v: k for k, v in self.token_to_id.items()}

#     def encode(self, text: str) -> list[int]:
#         ids = []
#         ids.append(self.token_to_id["[BOS]"])
#         sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)

#         i = 0
#         while i < len(text):
#             matched = False
#             for token in sorted_vocab:
#                 if text[i:i+len(token)] == token:
#                     ids.append(self.token_to_id[token])
#                     i += len(token)
#                     matched = True
#                     break
#             if not matched:
#                 ids.append(self.token_to_id["[UNK]"])
#                 i += 1

#         ids.append(self.token_to_id["[EOS]"])
#         return ids



#     def decode(self, ids: list[int]) -> str:
#         tokens = []
#         for id in ids:
#             if id in [0,1,2,3]:
#                 continue
#             tokens.append(self.id_to_token[id])
#         return "".join(tokens)


#     def train(self, texts: list[str], vocab_size: int):
#         corpus = []
#         for text in texts:
#             corpus.append(list(text))

#         while len(self.token_to_id) < vocab_size:
#             pairs ={}
#             for word in corpus:
#                 for i in range(len(word)-1):
#                     pair = (word[i], word[i+1])
#                     pairs[pair] = pairs.get(pair, 0) + 1

#             if not pairs:
#                 break


#             best_pair = max(pairs, key = lambda x:pairs[x])

#             new_token = best_pair[0] + best_pair[1]
#             new_id = len(self.token_to_id)           # 현재 vocab 크기가 다음 id
#             self.token_to_id[new_token] = new_id
#             self.id_to_token[new_id] = new_token



#             # corpus 업데이트
#             new_corpus = []

#             for word in corpus:
#                 new_word = []
#                 i = 0
#                 while i < len(word):
#                     if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
#                         new_word.append(new_token)
#                         i += 2
#                     else:
#                         new_word.append(word[i])
#                         i += 1
#                 new_corpus.append(new_word)
#             corpus = new_corpus





#     def save(self, path: str):
#         # token_to_id를 json 파일로 저장
#         with open(path, 'w', encoding= 'utf-8') as f :
#             json.dump({'token_to_id': self.token_to_id}, f)

#     def load(self, path: str):
#         with open(path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             self.token_to_id = data['token_to_id']
#             self.id_to_token = {v: k for k, v in self.token_to_id.items()}