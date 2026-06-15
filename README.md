# Korean Chatbot 프로젝트

한국어 언어모델을 **바닥부터 단계적으로 구현**하는 학습 프로젝트입니다.  
토크나이저와 모델 모두 직접 구현 → HuggingFace 라이브러리 활용 → Pretrained 파인튜닝 순서로 발전합니다.


![alt text](image.png)
---

## 프로젝트 구조

```
korean-chatbot/
├── main.py                  # FastAPI 서버 진입점
├── src/
│   ├── tokenizer_0.py       # BPE 토크나이저 직접 구현
│   ├── tokenizer_bpe.py     # HuggingFace tokenizers 래퍼
│   ├── tokenizer.py         # KoGPT2 pretrained 토크나이저
│   ├── model_0.py           # Transformer 직접 구현
│   ├── model_gpt2.py        # GPT2Config 기반 모델
│   ├── model.py             # KoGPT2 pretrained 파인튜닝
│   └── inference.py         # 추론 엔진
├── web/
│   ├── index.html           # 챗봇 UI
│   ├── style.css            # 다크테마 스타일
│   └── chat.js              # 채팅 로직 (API 통신)
├── notebooks/
│   ├── prepare_data.ipynb       # 나무위키 데이터 수집·정제
│   ├── prepare_data_k.ipynb     # KoAlpaca 데이터 준비
│   ├── train_tokenizer.ipynb    # 토크나이저 학습
│   ├── train_model.ipynb        # 사전학습 (나무위키)
│   ├── fine_tunning_model.ipynb # 파인튜닝 (KoAlpaca)
│   └── inference.ipynb          # 추론 및 대화 루프
├── data/
│   ├── namuwiki.txt         # 나무위키 정제 텍스트 (10만 문장)
│   └── train.txt            # KoAlpaca 질문/답변 포맷
└── models/
    ├── vocab.json            # 학습된 BPE 토크나이저
    └── model.pt              # 학습된 모델 가중치
```

---

## 전체 파이프라인

```
나무위키 / KoAlpaca (HuggingFace)
    ↓ prepare_data*.ipynb
namuwiki.txt / train.txt  →  Google Drive 백업
    ↓ train_tokenizer.ipynb
vocab.json (8K BPE)       →  Google Drive 백업
    ↓ train_model.ipynb
model.pt (사전학습)        →  Google Drive 백업
    ↓ fine_tunning_model.ipynb
model.pt (파인튜닝)        →  Google Drive 백업
    ↓ inference.ipynb
챗봇 대화 루프
```

---

## Stage 1 — 바닥부터 직접 구현 (학습용)

### 토크나이저: `tokenizer_0.py`

`BPETokenizer` 클래스를 순수 Python으로 구현합니다.

- 특수 토큰 4종 내장: `[PAD](0)`, `[BOS](1)`, `[EOS](2)`, `[UNK](3)`
- 학습 텍스트에서 등장한 문자 + ASCII(32~126) + 한글 전체(가\~힣, 11,172자)를 기본 vocab에 추가
- BPE 알고리즘으로 자주 등장하는 문자 쌍을 반복 병합해 vocab 확장
- `encode` / `decode` / `save` / `load` 인터페이스 제공

```python
tok = BPETokenizer()
tok.train(texts, vocab_size=8000)
tok.save("models/vocab.json")
```

### 모델: `model_0.py`

`Transformer` (Decoder-only) 를 PyTorch 기본 연산만으로 구현합니다.

- `nn.Embedding` 으로 토큰 임베딩 + 위치 임베딩
- `MultiHeadAttention`: Q/K/V 분리 → Scaled Dot-Product → 인과 마스크(causal mask) 적용 → 헤드 합치기
- `DecoderBlock`: Pre-LayerNorm 구조 (Attention → Add → FFN → Add)
- FFN 활성화 함수: GELU

```python
model = Transformer(vocab_size=8000, d_model=256, n_heads=4, n_layers=4, max_seq_len=512)
```

---

## Stage 2 — HuggingFace 라이브러리 활용

### 토크나이저: `tokenizer_bpe.py`

HuggingFace `tokenizers` 라이브러리의 `BPE` 모델을 래핑합니다.

- `ByteLevel` 전처리 + `ByteLevelDecoder` 로 UTF-8 문자를 안전하게 처리
- **파일 기반 학습**: 텍스트를 메모리에 전부 올리지 않고 스트리밍으로 학습
- `vocab.json` 하나로 저장·로드 가능

```python
tok = BPETokenizer()
tok.train(["data/namuwiki.txt"], vocab_size=8000)
tok.save("models/vocab.json")
```

### 모델: `model_gpt2.py`

`GPT2Config`로 하이퍼파라미터를 지정하고 `GPT2LMHeadModel`을 초기화합니다.  
내부 구조는 Stage 1과 동일하지만 구현을 라이브러리에 위임합니다.

- 패딩 토큰(`id=0`)은 loss 계산에서 `-100`으로 마스킹해 학습 대상에서 제외
- `generate()` 에서 top-k(50) + top-p(0.92) 샘플링 + no_repeat_ngram_size(3) 앵무새 방지

```python
model = Transformer(vocab_size=8000, d_model=256, n_heads=4, n_layers=4, max_seq_len=512)
```

---

## Stage 3 — Pretrained 파인튜닝

### 토크나이저: `tokenizer.py`

`skt/kogpt2-base-v2` 의 `PreTrainedTokenizerFast`를 그대로 로드합니다.  
별도 학습 없이 vocab size 51,200의 한국어 특화 토크나이저를 바로 사용합니다.

### 모델: `model.py`

`skt/kogpt2-base-v2` 가중치를 불러와 파인튜닝합니다.

- `GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")`
- 학습률 `3e-5` (사전학습의 `3e-4`보다 10배 작게) — 기존 지식이 덮이지 않도록
- `generate()`는 tokenizer의 `bos/eos/pad_token_id`를 직접 참조

---

## 추론 엔진: `inference.py`

`InferenceEngine` 클래스 하나로 토크나이저 + 모델 로드 및 추론을 통합 관리합니다.

```python
engine = InferenceEngine(
    model_path="models/model.pt",
    tokenizer_path="models/vocab.json"
)
result = engine.generate("안녕하세요", max_new_tokens=100, temperature=0.8)
```

- temperature로 생성 다양성 조절 (낮을수록 보수적, 높을수록 창의적)
- `[BOS]` 토큰 자동 삽입, `[EOS]` 토큰 만나면 생성 종료
- `torch.amp.autocast` 로 bfloat16 추론 가속

---

## 노트북 실행 가이드 (Google Colab)

모든 노트북은 실행 첫 셀에서 Google Drive 마운트 + GitHub 레포 클론/풀을 자동으로 처리합니다.

### 1. 데이터 준비

**나무위키 (사전학습용)**
```
prepare_data.ipynb 실행
→ data/namuwiki.txt 생성 (10만 문장, 정제 완료)
→ Google Drive 자동 백업
```

정제 내용: `[[링크]]` → 텍스트만 추출, URL 제거, `== 제목 ==` 제거, 연속 공백 축소

**KoAlpaca (파인튜닝용)**
```
prepare_data_k.ipynb 실행
→ data/train.txt 생성
→ Google Drive 자동 백업
```

저장 포맷:
```
### 질문: {instruction}
### 답변: {output}
```

### 2. 토크나이저 학습

```
train_tokenizer.ipynb 실행
→ namuwiki.txt로 BPE 학습 (vocab_size=8000)
→ models/vocab.json 저장
→ Google Drive 자동 백업
```

### 3. 모델 사전학습

```
train_model.ipynb 실행
→ vocab.json + namuwiki.txt 로드
→ Transformer(GPT2 구조) 학습
→ models/model.pt 저장
→ Google Drive 자동 백업
```

학습 설정:

| 항목 | 값 |
|---|---|
| batch_size | 64 |
| optimizer | AdamW, lr=3e-4 |
| epochs | 10 |
| 정밀도 | bfloat16 AMP |
| GPU 최적화 | A100 TF32 활성화 |
| max_seq_len | 512 |

### 4. 파인튜닝

```
fine_tunning_model.ipynb 실행
→ KoGPT2 pretrained + KoAlpaca 학습
→ models/model.pt 덮어쓰기
→ Google Drive 자동 백업
```

학습 설정:

| 항목 | 값 |
|---|---|
| batch_size | 16 |
| optimizer | AdamW, lr=3e-5 |
| epochs | 3 |
| 데이터셋 | beomi/KoAlpaca-v1.1a |
| 검증 | Perplexity 측정 포함 |

### 5. 추론

```
inference.ipynb 실행
→ vocab.json + model.pt 로드
→ 배치 테스트 또는 대화 루프 실행
```

대화 루프 종료: `q` 입력

---

## 구현 단계별 비교

| 구분 | Stage 1 (직접 구현) | Stage 2 (HF 라이브러리) | Stage 3 (Pretrained) |
|---|---|---|---|
| 토크나이저 | `tokenizer_0.py` | `tokenizer_bpe.py` | `tokenizer.py` |
| 모델 | `model_0.py` | `model_gpt2.py` | `model.py` |
| vocab size | 자유 설정 | 8,000 (나무위키) | 51,200 (KoGPT2) |
| 학습 데이터 | 커스텀 | 나무위키 | KoAlpaca (파인튜닝) |
| 목적 | 내부 원리 이해 | 실용적 구현 | 성능 극대화 |

---

## 데이터셋

| 데이터셋 | 출처 | 용도 |
|---|---|---|
| 나무위키 | `heegyu/namuwiki-extracted` | 토크나이저 학습 + 사전학습 |
| KoAlpaca v1.1a | `beomi/KoAlpaca-v1.1a` | 파인튜닝 (질문-답변) |

---

## 서버 실행: `main.py`

학습이 완료된 모델을 FastAPI 서버로 서빙합니다.

```
POST /chat
Body: { "message": "안녕하세요" }
Response: { "response": "안녕하세요! 무엇을 도와드릴까요?" }
```

서버 시작 시 `src/tokenizer.py`(KoGPT2 토크나이저)와 `src/model.py`(파인튜닝된 KoGPT2)를 로드하고, `models/model_1.pt` 가중치를 주입합니다. 입력 프롬프트는 `### 질문: / ### 답변:` 포맷으로 감싸서 모델에 전달하고, 응답에서 `### 답변:` 이후 텍스트만 추출해 반환합니다.

`web/` 폴더는 `/` 경로에 정적 파일로 마운트되어, 서버 하나로 API와 UI를 함께 제공합니다.

```bash
uvicorn main:app --reload
# → http://localhost:8000 에서 챗봇 UI 접속
```


![alt text](image-1.png)

---

## 웹 UI: `web/`

ChatGPT 스타일의 다크테마 채팅 인터페이스입니다.

| 파일 | 역할 |
|---|---|
| `index.html` | 사이드바 + 채팅 영역 레이아웃 |
| `style.css` | 다크테마 (`#212121` 배경), 말풍선, 타이핑 애니메이션 |
| `chat.js` | `POST /chat` API 호출, 메시지 렌더링, 대화 히스토리 사이드바 |

주요 기능:

- `Enter` 전송 / `Shift+Enter` 줄바꿈
- 봇 응답 대기 중 타이핑 애니메이션 (점 3개 bounce)
- 새 대화 버튼으로 채팅 초기화
- 사이드바에 최근 입력 기록 자동 추가

---

## 환경 설정

```bash
git clone https://github.com/kkkk2058/korean-chatbot.git
cd korean-chatbot
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload
```

주요 의존성: `torch`, `transformers`, `tokenizers`, `datasets`, `tqdm`, `fastapi`, `uvicorn`


---

## KoGPT2 pretrained + KoAlpaca fine-tuning 후 챗봇 답변 예시
==================================================
🤖 챗봇이 깨어났습니다! (종료하려면 '종료' 또는 'exit' 입력)
==================================================

👤 유저: 안녕
🤖 챗봇: 안녕하세요! 한국어의 어원은 무엇인가요?
저는 초등학교 2학년 학생입니다. 한국어 선생님께서 "알겠습니다~"라고 말했는데, 그 뜻은 알겠습니다. 그러나 이 말의 정확한 의미는 무엇일까요? 또한 '알아서' 대신 '그대로'를 사용해야 할 것 같은데요.

👤 유저: 대한민국에 대해서 설명해
🤖 챗봇: 대한민국이란 이름은 중국의 이름으로 쓰였으며, 한국이 중화민국과 합병되어 탄생한 나라입니다. 이 때 중국에서 국호를 '대한민국'으로 고쳤습니다. 따라서 "대한"로 바뀌었습니다.

👤 유저: 한국에 대해 설명
🤖 챗봇: 한국에는 '한국' 이라는 지명이 존재합니다. 이 지명은 서울이라는 지명에서 비롯되었으며, '서울'이라는 명칭은 한국 역사에서 중요한 역할을 했습니다. 이러한 지명을 가진 나라는 일본이었으며, 그 중에서도 특히 임진왜란 이후 일본으로 이주한 조선인들이 많이 살고 있습니다. 그리고 일본과 중국의 전쟁시기에 한국을 지키기 위해 노력하였던 것도 '한국'이라는 이름이 우리나라에 처음으로 등장한 이유 중 하나입니다.

👤 유저: 북한에 대한 설명
🤖 챗봇: 북한은 공산주의와 자본주의, 그리고 사회주의적 가치를 모두 가지고 있습니다. 북한의 사회주의는 다양하며, 그 중에서 가장 큰 특징은 체제나 조직 등 다양한 요소들이 결합한다는 것입니다. 이에 따라 북한에서는 사회주의적 가치가 매우 강하다는 특징이 있습니다.

북한의 정치사상에서 보면, 김일성은 '사회주의'를 강조하여 강조하였고, 이는 노동계급의 이익을 최우선으로 여겨야 한다는 생각을 담고 있었습니다. 이러한 사상과 사상이 결합하여 북한이 공산주의 국가로 발전하게 된 것입니다.

하지만 이 외에도 다른 요소들이 있으며, 이를 종합적으로 고려하면 북한에서의 사회주의는 크게 두 가지로 분류할 수 있을 것으로 보입니다. 첫째, 경제적으로는 민주화를 이룩하였으며, 이는 남한의 경제성장을 촉진시켰으며, 이를 통해 남한은 세계에서 제일 강력한 국가임을 보여줍니다.
둘째, 정치, 문화, 종교 등에서는 북한을 사회주의국가로 발전시키기 위한 노력을 하였습니다.

👤 유저: 요즘 핸드폰 추천
🤖 챗봇: 핸드폰은 주로 휴대폰과 연결이 잘 안되는 기능을 가지고 있습니다. 이러한 기능은 대부분 스마트폰에서 많이 사용됩니다. 일반적으로 SKT와 KT는 핸드폰으로 통화할 수 있으며, LG U+나 삼성전자도 마찬가지입니다. 하지만 대부분의 경우 통신사별로 다른 기능이 있는 경우가 많습니다. 예를 들어, KT의 경우, 011, 017, 019, 2000 등 전화번호가 등록되어 있지 않은 경우에만 010을 사용할 수 있고, SK텔레콤에서는 016 등의 전화를 할 때 필요한 음성통화와 문자 메시지를 이용할 수 있습니다. 또한, 일부 업체들은 서비스 출시일이 정해진 경우도 있지만, 이는 매우 적은 숫자일 뿐입니다.

👤 유저: 요즘 스마트폰 추천
🤖 챗봇: 안녕하세요! 

저는 최근에 휴대전화 자주 사용하는 것을 고민 중입니다. 최근에 스마트폰을 많이 사용하면서 불편함을 느끼고 있습니다. 하지만 제가 사용할 때는 항상 깔끔하고 유용한 제품을 찾고 싶습니다. 어떤 것이 좋을까요? 또한, 좋은 제품이 있다면 언제 어디서든 언제든지 추천 부탁드립니다.

👤 유저: q
🤖 챗봇: q는 영어 약자입니다. 'q'는 영어의 'quilli'와 마찬가지로, 명사 뒤에 붙는 '-'의 의미를 가지고 있습니다. 따라서, q가 있는 문장에서는 "q"와 같이 쓰이거나 "q"는 "q"의 의미로 사용됩니다.




==================================================
🤖 챗봇 시작! (종료: '종료' / 'exit' / 'quit')
==================================================

👤 유저: 안녕
🤖 챗봇: - #
- # (feat.naver.com/written_mixx=7)
- https://term.blog.me/LongPostView&volumeNoWSHOUSEXIEGRANCYBERA1FJECTMEDICKZ8qa102289123134140680568)

👤 유저: 넌 누구야?
🤖 챗봇: 네, 안녕하세요, 가수 싸이더스입니다. 네, 네. 그리고 최근 앨범 [Space]를 발매했습니다. 이 곡에서 전작과 달리 큰 키와 매력적인 외모를 지닌 멤버들의 조합으로 더욱 유명해졌습니다.

싸이의 이번 정규앨범 [True High Start]은 'Third With This Love'의 약자로, 한 마디로 "내일은 너의 일상이다!"라는 의미를 담고 있습니다. 또한, 싸이는 지난 2016년 12월 22일 자신의 트위터에 “너를 잊지 마십시오”라는 글을 남겨 팬들과 함께 감사의 마음을 표현하기도 했습니다. 이와 같은 이유로,

👤 유저: 오늘 날씨 덥다
🤖 챗봇: 날씨는 구름이 많은 날씨가 예상됩니다. 특히 새벽에는 흐리고 비가 오는 곳이 많습니다. 이는 고기압의 영향을 받는 북태평양고기압이 우리나라로 확장하면서 만들어진 비구름이기 때문입니다. 따라서 이날은 구름 없이 맑은 날이 많고, 일사에 의해 기온이 오르면서 추워질 것으로 생각됩니다.

따라서, 기상청에서는 "날씨 예보"를 통해 날씨 예측을 하고 있습니다. 
※ 답변 출처: http://www.kma1.go.kr/articleView-201612/2106&blogs_story2.asp?docId=847

👤 유저: 핸드폰 추천해줘
🤖 챗봇: 네, 안녕하세요. 스마트폰으로 영화와 TV를 시청하면서 음악을 감상하는 것은 어떨까요? 이는 하나의 문화 현상 중 하나입니다. 휴대폰에서 음악 재생이나 인터넷 검색을 하면 그 음악이 바로 검색되는 것과 같은 원리로 음원 사이트나 스트리밍 사이트 등에서 다양한 기능을 사용할 수 있습니다. 또한, 간단한 정보나 노래를 듣고 싶은 욕구가 있다면 뮤직비디오도 볼 수도 있고, 자신의 취향에 맞는 영화를 선택하여 즐길 수도 있습니다.

한편, 음악은 여러 가지 용도로 활용될 수 있으며, 특히 디지털 음악의 경우 음향 효과(디지털 피아노의 사운드, 모션 센서 등의 기능)가 뛰어나기 때문에 많은 이들이 이용합니다. 따라서, 집에서 음악과

👤 유저: 핸드폰 기종 추천
🤖 챗봇: 휴대폰이 보급되면서 다양한 제품들이 출시되고 있습니다. 그 중에서도 TGI(Thin Global Inchange)라는 제품이 가장 인기 있는 폰 중 하나입니다. 이 제품은 듀얼 폴더와 슬라이딩을 지원하며, 슬림형 디자인으로 휴대성과 편의성을 높였습니다. 또한 무선 이어폰과 와이파이(Wi-Fi) 기능을 제공합니다.

👤 유저: 나 힘들어
🤖 챗봇: 네, 안녕하세요. 안녕히 주무십시오. 

1. 네, 괜찮습니다. 제가 좋아하던 제 애인입니다.
2. 제가 좋아하는 사람입니다. 다른 여자 친구들하고 다툰다는 것이 싫어서 이젠 혼자 고민하지 않으셔도 됩니다.
3. 오늘은 너무 편하게 놀러와주세요.
4. 저는 친구의 남자친구인 예쁘고 멋진 모습을 보고 싶습니다.
5. 아주 잘 지내실 겁니다.
6. 더 좋은 친구가 되기 위해서 노력해보겠습니다.
7. 앞으로 살아가기 위해 노력하는 동안 많은 것을 배울 수 있을 것입니다.
8. 이제부터 모든 일이 즐겁게 해결될 것입니다. 기분이 좋으니까 걱정 안

👤 유저: 나 화나 진정시켜줘
🤖 챗봇: 네, 안녕하세요! "화났어요." 라는 말은 '폭탄' 또는 '비리'라는 뜻으로 사용됩니다. 이 말의 어원은 분명하지 않지만, 폭탄과 관련된 표현 중 하나인 '폭설'이라는 단어에서 유래된 것으로 추측되고 있습니다. 

- 비리에 대한 개념: 형법 제385조에 따라 처벌될 수 있는 범죄가 있다면 그 범죄를 저지른 사람이 누구인지 정확히 알고 있어야 합니다. 따라서 경찰관이나 검찰관 등이 해당 범죄의 범인을 파악하고 있을 가능성이 높습니다. 따라서, 폭력이나 부조리를 저지르는 경우에는 이러한 처벌을 피하기 어려울 것입니다.

👤 유저: 배고파 뭐먹지
🤖 챗봇: 안녕하세요, 배달해주신 배부르다에요! 
배달드립니다. 배 고프다는 말씀이시군요. 하지만 요즘은 음식점에서도 다양한 종류의 요리가 나오면서 고기 종류도 다양해졌습니다. 또한 최근에는 라면이나 생수 등 일부 제품도 인기를 끌고 있습니다. 이러한 트렌드는 어느 정도 예상된 일로, 최근 몇 년간 외식업계에서는 새로운 메뉴를 출시하고 새로운 마케팅 전략을 선보이고 있는 것입니다.
