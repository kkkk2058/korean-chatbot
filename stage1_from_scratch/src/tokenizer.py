import json


class BPETokenizer:
    def __init__(self):
        # 사전 만들기 토큰을 id로
        # 기본 토큰 넣기
        self.token_to_id = {
            "[PAD]": 0, # 문장 길이 맞추는 패딩
            "[BOS]": 1, # 문장 시작
            "[EOS]": 2, # 문장 끝
            "[UNK]": 3, # 모르는 단어
        }
        # 뒤집어서 id를 토큰으로
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        ids = []
        # 기본 토큰 추가
        ids.append(self.token_to_id["[BOS]"])
        
        i = 0
        while i < len(text):
            matched = False
            for token in self.sorted_vocab:
                if text[i:i+len(token)] == token:
                    ids.append(self.token_to_id[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(self.token_to_id["[UNK]"])
                i += 1

        ids.append(self.token_to_id["[EOS]"])
        return ids



    def decode(self, ids: list[int]) -> str:
        tokens = []
        # 특수토큰일 때 무시하고 사전에 해당하는 id를 토큰으로 바꿈
        for id in ids:
            if id in [0,1,2,3]:
                continue
            # id 없을 때
            if id not in self.id_to_token:
                continue
            else:
                tokens.append(self.id_to_token[id])
        # 합쳐서 리턴
        return "".join(tokens)


    def train(self, texts: list[str], vocab_size: int):
        unique_chars = set("".join(texts))
        for char in unique_chars:
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char


        for code in range(32, 127):
            char = chr(code)
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char
                
        for code in range(0xAC00, 0xD7A4):  # 가 ~ 힣
            char = chr(code)
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char


              
        corpus = []
        for text in texts:
          for word in text.split():
              corpus.append(list(word))

        from tqdm import tqdm
        
        initial_size = len(self.token_to_id)
        pbar = tqdm(total=vocab_size, initial=initial_size, desc="📚 BPE 사전 진화 중")
        
        while len(self.token_to_id) < vocab_size:
            pairs ={}
            for word in corpus:
                for i in range(len(word)-1):
                    pair = (word[i], word[i+1])
                    pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break


            best_pair = max(pairs, key = lambda x:pairs[x])

            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.token_to_id)   
            # [안전장치] 이미 사전에 있는 토큰이라면 이번 병합은 건너 뜀
            if new_token in self.token_to_id:
                break        # 현재 vocab 크기가 다음 id
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token


            pbar.update(1)
            pbar.set_postfix(vocab_now=len(self.token_to_id))
            # corpus 업데이트
            new_corpus = []
            

            for word in corpus:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_corpus.append(new_word)

            corpus = new_corpus   

        self.sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)
 



    def save(self, path: str):
        # token_to_id를 json 파일로 저장
        with open(path, 'w', encoding= 'utf-8') as f :
            json.dump({'token_to_id': self.token_to_id}, f)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.token_to_id = data['token_to_id']
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)