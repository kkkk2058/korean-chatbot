# import json
# from tqdm import tqdm
# import collections


# class BPETokenizer:
#     def __init__(self):
#         ## Greedy (sorted) 방식
#         # 사전 만들기 토큰을 id로
#         # 특수 토큰에 공백을 대체할 특수 문자 추가 (SentencePiece 방식)
#         self.SPACE_TOKEN = " "  # (특수 공백 문자)
#         # 기본 토큰 넣기
#         self.token_to_id = {
#             "[PAD]": 0, # 문장 길이 맞추는 패딩
#             "[BOS]": 1, # 문장 시작
#             "[EOS]": 2, # 문장 끝
#             "[UNK]": 3, # 모르는 단어
#         }
#         # 뒤집어서 id를 토큰으로
#         self.id_to_token = {v: k for k, v in self.token_to_id.items()}
#         self.sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)




#     def encode(self, text: str) -> list[int]:

#         text = text.replace(" ", self.SPACE_TOKEN)
#         ids = []
#         # 기본 토큰 추가
#         ids.append(self.token_to_id["[BOS]"])
        
#         i = 0
#         while i < len(text):
#             matched = False
#             for token in self.sorted_vocab:
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
#         # 특수토큰일 때 무시하고 사전에 해당하는 id를 토큰으로 바꿈
#         for id in ids:
#             if id in [0,1,2,3]:
#                 continue
#             # id 없을 때
#             if id not in self.id_to_token:
#                 continue
#             else:
#                 tokens.append(self.id_to_token[id])
#         # 합쳐서 리턴 + 골백문자 띄어쓰기로 변경
#         return "".join(tokens).replace(self.SPACE_TOKEN, " ")


#     def train(self, texts: list[str], vocab_size: int):

#         processed_texts = [text.replace(" ", self.SPACE_TOKEN) for text in texts]
#         unique_chars = set("".join(processed_texts))
        
#         # 사전 학습 학습 효율(시간)이 너무 떨어진다.  (가~힣,아스키 제외)
#         for char in unique_chars:
#             if char not in self.token_to_id:
#                 new_id = len(self.token_to_id)
#                 self.token_to_id[char] = new_id
#                 self.id_to_token[new_id] = char
        
#         # corpus = []
#         # for text in processed_texts :
#         #     corpus.append(list(text))
#         corpus = collections.Counter()

#         for text in texts:
#             words = text.split(" ")
#             if not words:
#                 continue
        
#             passed_words = [words[0]] + [self.SPACE_TOKEN + w for w in words[1:]]

#             for word in passed_words:
#                 corpus[tuple(word)] += 1

#         pairs = collections.Counter()

#         for word_tuple, freq in corpus.items():
#             for i in range(len(word_tuple)-1):
#                 pair = (word_tuple[i], word_tuple[i+1])
#                 pairs[pair] += freq


#         initial_size = len(self.token_to_id)
#         pbar = tqdm(total=vocab_size, initial=initial_size, desc=" BPE 사전 진화 중")
        
#         while len(self.token_to_id) < vocab_size:
#             if not pairs:
#                 break

#             #best_pair = max(pairs, key = lambda x:pairs[x])
#             best_pair = max(pairs, key=pairs.get)
            
#             new_token = best_pair[0] + best_pair[1]

#             if new_token not in self.token_to_id:
#                 new_id = len(self.token_to_id)   
#                 self.token_to_id[new_token] = new_id
#                 self.id_to_token[new_id] = new_token

            
                
            
#             # corpus 업데이트
#             new_corpus = {}

#             for word_tuple, freq in corpus.items():

#                 if best_pair[0] not in word_tuple:
#                     new_corpus[word_tuple] = freq
#                     continue
                
#                 for i in range(len(word_tuple) - 1):
#                     p = (word_tuple[i], word_tuple[i+1])
#                     pairs[p] -= freq
#                     if pairs[p] <= 0:
#                         del pairs[p] # 사전에 0인 찌꺼기가 쌓이지 않게 삭제 (max 탐색 속도 극대화)
#                 # -------------------------------------------------------------
                
                
#                 new_word = []
#                 i = 0
#                 while i < len(word_tuple):
#                     if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == best_pair:
#                         new_word.append(new_token)
#                         i += 2
#                     else:
#                         new_word.append(word_tuple[i])
#                         i += 1
#                 new_word_tuple = tuple(new_word)
#                 new_corpus[new_word_tuple] = freq

#                 for i in range(len(new_word_tuple) - 1):
#                     p = (new_word_tuple[i], new_word_tuple[i+1])
#                     pairs[p] += freq
#             corpus = new_corpus   
#             pbar.n = len(self.token_to_id)
#             pbar.refresh() # 화면에 즉시 반영
            
#         self.sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)
 



#     def save(self, path: str):
#         # token_to_id를 json 파일로 저장
#         with open(path, 'w', encoding= 'utf-8') as f :
#             json.dump({'token_to_id': self.token_to_id}, f)

#     def load(self, path: str):
#         with open(path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             self.token_to_id = data['token_to_id']
#             self.id_to_token = {v: k for k, v in self.token_to_id.items()}
#             self.sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)



"""
tokenizer.py
기본 BPE(Byte Pair Encoding) 토크나이저 - 라이브러리 없이 순수 파이썬 구현
한국어 챗봇용 (KoAlpaca 데이터셋 학습 헬퍼 포함) + Jupyter Notebook 시각화 최적화
"""
import json
from collections import defaultdict
from tqdm.notebook import tqdm  # 주피터 노트북 전용 tqdm으로 변경


class BPETokenizer:
    def __init__(self):
        self.vocab = {}          # token(str) -> id(int)
        self.id_to_token = {}    # id(int)  -> token(str)
        self.merges = []         # 학습된 merge 규칙 (순서 중요): [(a, b), ...]
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        self._encode_cache = {}  # word -> [id, ...] 캐시

    # ----------------------- 학습 -----------------------
    def _get_word_freqs(self, corpus):
        word_freqs = defaultdict(int)
        for line in corpus:
            for word in line.strip().split():
                word_freqs[word] += 1
        return word_freqs

    def _build_initial_splits(self, word_freqs):
        return {word: list(word) + ["</w>"] for word in word_freqs}

    def _count_pairs(self, splits, word_freqs):
        pairs = defaultdict(int)
        for word, symbols in splits.items():
            freq = word_freqs[word]
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _build_pair_index(self, splits):
        """pair → 그 pair가 등장하는 단어들의 집합 (역인덱스)"""
        index = defaultdict(set)
        for word, symbols in splits.items():
            for i in range(len(symbols) - 1):
                index[(symbols[i], symbols[i + 1])].add(word)
        return index

    def _merge_incremental(self, pair, splits, word_freqs, pairs, pair_index):
        """pair가 포함된 단어만 순회해 O(전체단어) → O(해당pair등장단어) 로 단축."""
        a, b = pair
        merged = a + b

        affected_words = list(pair_index.get(pair, []))

        for word in affected_words:
            symbols = splits[word]
            freq = word_freqs[word]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    # 왼쪽 이웃 pair 갱신
                    if new_symbols:
                        left = new_symbols[-1]
                        pairs[(left, a)] -= freq
                        if pairs[(left, a)] <= 0:
                            del pairs[(left, a)]
                        pair_index[(left, a)].discard(word)
                        pairs[(left, merged)] = pairs.get((left, merged), 0) + freq
                        pair_index[(left, merged)].add(word)
                    # 오른쪽 이웃 pair 갱신
                    if i + 2 < len(symbols):
                        right = symbols[i + 2]
                        pairs[(b, right)] -= freq
                        if pairs[(b, right)] <= 0:
                            del pairs[(b, right)]
                        pair_index[(b, right)].discard(word)
                        pairs[(merged, right)] = pairs.get((merged, right), 0) + freq
                        pair_index[(merged, right)].add(word)
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            splits[word] = new_symbols

        # 병합된 pair 자체 제거
        if pair in pairs:
            del pairs[pair]
        if pair in pair_index:
            del pair_index[pair]

    def train(self, corpus, vocab_size=8000, max_corpus_lines=None, verbose=True):
        if max_corpus_lines:
            corpus = corpus[:max_corpus_lines]

        print("1. 말뭉치 빈도 분석 및 초기 분할 생성 중...")
        word_freqs = self._get_word_freqs(corpus)
        splits = self._build_initial_splits(word_freqs)

        base_vocab = set()
        for symbols in splits.values():
            base_vocab.update(symbols)
        tokens = list(self.special_tokens) + sorted(base_vocab)

        num_merges = vocab_size - len(tokens)
        if num_merges <= 0:
            print(f"이미 목표 vocab_size({vocab_size})를 넘는 기초 토큰({len(tokens)})이 존재합니다.")
            num_merges = 0

        print(f"2. BPE 토큰 병합(Merge) 시작 (목표 횟수: {num_merges}회)...")

        pairs = self._count_pairs(splits, word_freqs)
        pair_index = self._build_pair_index(splits)

        pbar = tqdm(range(num_merges), desc="BPE 학습 진행률", disable=not verbose)
        for i in pbar:
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self._merge_incremental(best, splits, word_freqs, pairs, pair_index)
            self.merges.append(best)
            tokens.append(best[0] + best[1])

            if verbose and (i + 1) % 100 == 0:
                pbar.set_description(f"BPE 학습 중 [ '{best[0]}' + '{best[1]}' ]")

        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        if verbose:
            print(f"✨ 학습 완료! 최종 어휘 사전 크기(Vocab Size) = {len(self.vocab)}")

    # ----------------------- 인코딩 / 디코딩 -----------------------
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
        """문자열 -> id 리스트 (단어 단위 영구 캐싱으로 반복 단어 재계산 방지)"""
        unk = self.vocab["<unk>"]
        ids = []
        for word in text.strip().split():
            if word not in self._encode_cache:
                self._encode_cache[word] = [self.vocab.get(tok, unk) for tok in self._tokenize_word(word)]
            ids.extend(self._encode_cache[word])
        return ids

    def decode(self, ids):
        """id 리스트 -> 문자열"""
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    # ----------------------- 저장 / 불러오기 -----------------------
    def save(self, path):
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.special_tokens = data["special_tokens"]
        self.id_to_token = {int(idx): tok for tok, idx in self.vocab.items()}
        self._encode_cache = {}


def load_koalpaca_corpus(split="train"):
    """KoAlpaca 데이터셋에서 텍스트를 모아 코퍼스(문장 리스트)로 반환"""
    from datasets import load_dataset

    print("Hugging Face에서 KoAlpaca 데이터셋 로드 중...")
    ds = load_dataset("beomi/KoAlpaca-v1.1a", split=split)
    text_fields = ["instruction", "input", "output"]
    corpus = []
    
    # 데이터셋 파싱 구간 주피터 전용 tqdm 적용
    for row in tqdm(ds, desc="데이터셋 코퍼스 변환 중"):
        for f in text_fields:
            v = row.get(f)
            if v:
                corpus.append(v)
    return corpus


def train_on_koalpaca(vocab_size=8000, split="train", save_path="ko_bpe.json", max_corpus_lines=None):
    """KoAlpaca로 BPE 토크나이저를 학습하고 저장"""
    corpus = load_koalpaca_corpus(split=split)
    print(f"코퍼스 수집 완료. 총 문장 수: {len(corpus)}")

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size, max_corpus_lines=max_corpus_lines)
    tokenizer.save(save_path)
    print(f"💾 파일 저장 완료 -> {save_path}")
    return tokenizer