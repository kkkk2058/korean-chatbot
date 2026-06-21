# Korean Chatbot 프로젝트

한국어 언어모델을 **바닥부터 단계적으로 구현**하는 학습 프로젝트입니다.
토크나이저와 Transformer를 라이브러리 없이 직접 만드는 것에서 시작해,
HuggingFace 라이브러리 활용, KoGPT2 파인튜닝까지 3단계로 발전시킵니다.

**현재 서버는 Stage 1(직접 구현) 모델을 서빙합니다.**

![챗봇 UI](image-2.png)

---

## 프로젝트 구조

```
korean-chatbot/
├── app/                          # 서빙 레이어 (FastAPI)
│   ├── config.py                 # 경로·생성 파라미터
│   ├── main.py                   # FastAPI 진입점 (POST /chat + web UI)
│   └── core/
│       ├── model.py              # Decoder-only Transformer
│       ├── tokenizer.py          # BPE 토크나이저 (추론 전용)
│       └── engine.py             # InferenceEngine (모델 로드 + 생성)
├── web/                          # 프론트엔드 (정적)
│   ├── index.html
│   ├── style.css
│   └── chat.js
├── stage1_from_scratch/          # ① 바닥부터 직접 구현 (현재 서빙)
│   ├── config.py                 # 토크나이저·모델·학습 하이퍼파라미터
│   ├── src/
│   │   ├── model.py              # Transformer (순수 PyTorch)
│   │   └── tokenizer.py          # BPETokenizer (순수 Python)
│   ├── tokenizer/tokenizer.json  # 학습된 BPE 토크나이저 (vocab 16k)
│   └── colab/
│       └── train_model.ipynb     # 위키 사전학습 → KoAlpaca 파인튜닝 → 생성
├── stage2_with_library/          # ② HuggingFace 라이브러리 활용
│   ├── config.py
│   ├── src/{model.py, tokenizer.py}
│   └── colab/{prepare_data, train_tokenizer, train_model, inference}.ipynb
├── stage3_fine_tuning/           # ③ KoGPT2 파인튜닝
│   ├── config.py
│   └── colab/{prepare_data_koAlpaca, fine_tunning_koAlpaca}.ipynb
├── models/                       # 학습된 가중치 (git 제외)
│   └── model_scratch.pt          # ← 현재 서빙 중인 Stage 1 모델
├── data/                         # 학습 데이터 (git 제외)
├── Makefile
└── requirements.txt
```

---

## 구현 단계별 비교

| 구분 | Stage 1 (직접 구현) | Stage 2 (HF 라이브러리) | Stage 3 (파인튜닝) |
|---|---|---|---|
| 토크나이저 | 순수 Python BPE | `tokenizers` 라이브러리 | KoGPT2 pretrained |
| 모델 | 순수 PyTorch Transformer | GPT2Config 기반 | skt/kogpt2-base-v2 |
| vocab size | 16,000 (위키피디아) | 8,000 (나무위키) | 51,200 (KoGPT2) |
| 학습 데이터 | 위키 사전학습 + KoAlpaca 파인튜닝 | 나무위키 | KoAlpaca |
| 목적 | 내부 원리 이해 | 실용적 구현 | 성능 극대화 |

---

## Stage 1 — 바닥부터 직접 구현 (현재 서빙 버전)

라이브러리 없이 BPE 토크나이저와 Transformer를 직접 구현하고,
**한국어 위키피디아로 사전학습 → KoAlpaca로 파인튜닝**하는 전체 파이프라인입니다.
모든 과정은 `stage1_from_scratch/colab/train_model.ipynb` 한 노트북에서 실행되며,
산출물(토크나이저·체크포인트·변환 캐시)은 Google Drive에 저장돼 런타임 재시작에도 복원됩니다.

### 학습 흐름

```
위키 30k문서 → BPE(vocab 16k) 학습 → tokenizer.json
                    ↓
위키 100k줄 → 사전학습 (FP16, eff.batch 64, lr 3e-4) → 언어 이해
                    ↓ (가중치 이어받기)
KoAlpaca 21k → 파인튜닝 (답변만 -100 마스킹 학습, lr 3e-4) → QA 형식
                    ↓
생성 (temperature 0.8, top_k 30, repetition_penalty 1.3)
```

### 토크나이저: `stage1_from_scratch/src/tokenizer.py`

`BPETokenizer` 클래스를 순수 Python으로 구현합니다.

- 특수 토큰 4종: `<pad>`, `<unk>`, `<s>`, `</s>`
- BPE 알고리즘으로 자주 등장하는 문자 쌍을 반복 병합해 vocab(16,000) 확장
- 속도 최적화: **증분 pair 카운트 업데이트 + pair 역인덱스 + 단어 단위 인코딩 캐시** (5시간 → 수 분)
- `encode` / `decode` / `save` / `load` 인터페이스 제공

```python
tok = BPETokenizer()
tok.train(wiki_texts, vocab_size=16000)
tok.save("tokenizer.json")
```

### 모델: `stage1_from_scratch/src/model.py`

Decoder-only Transformer를 PyTorch 기본 연산만으로 구현합니다.

- `nn.Embedding`으로 토큰 임베딩 + 위치 임베딩
- `MultiHeadAttention`: Q/K/V 분리 → Scaled Dot-Product → 인과 마스크(causal mask)
- `DecoderBlock`: Pre-LayerNorm 구조 (Attention → Add → FFN → Add), GELU 활성화

### 파라미터: `stage1_from_scratch/config.py`

| 구분 | 항목 | 값 |
|---|---|---|
| 토크나이저 | `VOCAB_SIZE` | 16,000 |
| 모델 | `D_MODEL` / `N_HEADS` / `N_LAYERS` | 512 / 8 / 12 |
| 모델 | `MAX_SEQ_LEN` / `DROPOUT` | 512 / 0.1 |
| 사전학습 | `PRETRAIN_BATCH_SIZE` × `GRAD_ACCUM` | 16 × 4 (실질 64) |
| 사전학습 | `PRETRAIN_LR` / `EPOCHS` | 3e-4 / 3+ |
| 파인튜닝 | `FINETUNE_BATCH_SIZE` × `GRAD_ACCUM` | 16 × 4 (실질 64) |
| 파인튜닝 | LR / `EPOCHS` | 3e-4 / 10 |

> 파라미터 수 약 52M. 모델 가중치만 저장 시 약 210MB (`model_scratch.pt`).

### 사전학습 (위키피디아)

- 위키피디아 30,000 문서를 `streaming`으로 받아 약 100,000줄을 학습 (순수 Python BPE라 전체 변환은 비현실적)
- **FP16 mixed precision + gradient accumulation**으로 A100 40GB 내 학습
- 변환 결과를 Drive에 캐시(`pretrain_ids.pt`)해 재실행 시 즉시 로드
- 체크포인트에 **scheduler/scaler state까지 저장** → 이어학습 시 LR 폭등 방지

### 파인튜닝 (KoAlpaca)

- 프롬프트 형식: `질문: {q}\n답변: {a}`
- **답변 토큰만 loss 계산** (질문 부분은 `-100`으로 마스킹) → "답변하는 모델"로 전환되는 핵심
- 사전학습 가중치를 이어받아 lr 3e-4로 파인튜닝

### 생성

| 파라미터 | 값 | 역할 |
|---|---|---|
| `temperature` | 0.8 | 다양성 |
| `top_k` | 30 | 상위 토큰만 샘플링 |
| `repetition_penalty` | 1.3 | 반복 루프 억제 |
| `max_new_tokens` | 100 | 깨지기 전 종료 |

> **한계**: 약 52M 파라미터의 소형 모델이라 문장 단위 일관성은 되지만 사실 정확성·긴 문맥은 약합니다.
> 개선하려면 사전학습 데이터(`MAX_TRAIN_LINES`)를 늘려 사전학습 loss를 더 낮추는 것이 핵심입니다.

---

## Stage 2 — HuggingFace 라이브러리 활용

### 토크나이저: `stage2_with_library/src/tokenizer.py`

HuggingFace `tokenizers` 라이브러리의 BPE 모델을 활용합니다.

- `ByteLevel` 전처리 + `ByteLevelDecoder`로 UTF-8 문자를 안전하게 처리
- 파일 기반 스트리밍 학습으로 대용량 데이터도 메모리 효율적으로 처리
- `vocab.json` 단일 파일로 저장·로드

### 모델: `stage2_with_library/src/model.py`

`GPT2Config`로 하이퍼파라미터를 지정하고 `GPT2LMHeadModel`을 초기화합니다.
내부 구조는 Stage 1과 동일하지만 구현을 라이브러리에 위임합니다.

### 파라미터: `stage2_with_library/config.py`

```python
from config import ModelConfig, TrainConfig, TokenizerConfig

model_cfg = ModelConfig()          # vocab_size=8000, d_model=256, n_heads=4, n_layers=4
train_cfg = TrainConfig()          # batch_size=64, lr=3e-4, epochs=10, bf16=True
tok_cfg = TokenizerConfig()        # vocab_size=8000, data_path, save_path
```

---

## Stage 3 — KoGPT2 파인튜닝

`skt/kogpt2-base-v2` 사전학습 모델을 KoAlpaca 질문/답변 데이터로 파인튜닝합니다.

### 학습 설정

| 항목 | 값 |
|---|---|
| 베이스 모델 | skt/kogpt2-base-v2 |
| 데이터셋 | beomi/KoAlpaca-v1.1a |
| batch_size | 32 (gradient_accumulation×2 → 실질 64) |
| learning_rate | 3e-5 (cosine 스케줄러, warmup 10%) |
| epochs | 최대 5 (EarlyStopping patience=2) |
| 정밀도 | bf16 (A100) |
| GPU | Google Colab Pro A100 |

### 학습 프롬프트 포맷

```
<usr>질문 내용<bot>답변 내용</s>
```

답변(`<bot>` 이후)만 loss 계산에 포함하고 질문 부분은 `-100`으로 마스킹합니다.
(Stage 1의 답변-마스킹 파인튜닝과 동일한 원리)

---

## Colab 실행 가이드

### Stage 1

`train_model.ipynb` 한 노트북을 위에서 아래로 실행합니다. (A100 권장)

```
0. 환경 설정        → Drive 마운트 + 레포 클론/풀
1. 토크나이저       → tokenizer.json (없으면 위키로 학습, 있으면 로드)
2. 모델 초기화
3. 사전학습         → pretrain_checkpoint.pt (위키 100k줄, 이어학습 지원)
4. 파인튜닝         → finetune_checkpoint.pt (KoAlpaca, 답변만 마스킹 학습)
5. 생성 테스트
6. Drive 저장
```

### Stage 2

```
1. prepare_data.ipynb       → data/namuwiki.txt 생성 (나무위키 10만 문장)
2. train_tokenizer.ipynb    → models/vocab.json 생성 (BPE vocab_size=8000)
3. train_model.ipynb        → models/model.pt 생성 (사전학습)
4. inference.ipynb          → 추론 테스트
```

### Stage 3

```
1. prepare_data_koAlpaca.ipynb   → data/train.txt 생성 (KoAlpaca QA 포맷)
2. fine_tunning_koAlpaca.ipynb   → models/model_fine_tuning.pt 생성
```

---

## 데이터셋

| 데이터셋 | 출처 | 용도 |
|---|---|---|
| 위키피디아 | `wikimedia/wikipedia` (20231101.ko) | Stage 1 토크나이저 학습 + 사전학습 |
| 나무위키 | `heegyu/namuwiki-extracted` | Stage 2 토크나이저 학습 + 사전학습 |
| KoAlpaca v1.1a | `beomi/KoAlpaca-v1.1a` | 파인튜닝 (질문-답변) |

---

## 서버 실행

`app/`은 **Stage 1 from-scratch 모델**(`models/model_scratch.pt`)을 서빙합니다.

```bash
git clone https://github.com/kkkk2058/korean-chatbot.git
cd korean-chatbot
make install   # pip install -r requirements.txt

make run       # uvicorn app.main:app --reload
# → http://localhost:8000  (브라우저로 열면 채팅 UI)
```

### 모델 파일 준비

서버는 아래 두 파일이 필요하며, **반드시 같은 학습 세션의 짝**이어야 합니다 (토크나이저 ↔ 모델 불일치 시 출력이 깨짐):

```
models/model_scratch.pt                         # Stage 1 모델 가중치 (state_dict)
stage1_from_scratch/tokenizer/tokenizer.json    # 학습에 쓴 BPE 토크나이저
```

경로는 환경변수 `MODEL_PATH` / `TOKENIZER_PATH`로 override 가능합니다.
모델 아키텍처(vocab·d_model·n_layers·max_seq_len)는 체크포인트 shape에서 **자동 추론**되며,
`n_heads`만 학습 때 값과 동일하게 `app/config.py`(또는 `N_HEADS` 환경변수)에 맞추면 됩니다.

> 짝이 맞는지 확인: 한국어 문장의 LM loss가 랜덤(≈9.7)보다 충분히 낮으면 정상.
> loss가 랜덤보다 높으면 토크나이저-모델 불일치입니다.

### API 스펙

```
POST /chat
Body:     { "message": "안녕하세요" }
Response: { "response": "..." }
```

`web/` 폴더는 `/` 경로에 정적 파일로 마운트되어, 서버 하나로 API와 UI를 함께 제공합니다.

### `app/` 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 경로 + 생성 파라미터(temperature/top_k/repetition_penalty) |
| `core/model.py` | Decoder-only Transformer |
| `core/tokenizer.py` | BPE 토크나이저 (추론 전용 load/encode/decode) |
| `core/engine.py` | `InferenceEngine` — 체크포인트 자동 추론 로드 + 생성 |
| `main.py` | FastAPI `POST /chat` + `web/` 정적 UI 서빙 |

---

## 웹 UI

ChatGPT 스타일의 다크테마 채팅 인터페이스입니다.

| 파일 | 역할 |
|---|---|
| `index.html` | 사이드바 + 채팅 영역 레이아웃 |
| `style.css` | 다크테마 (`#212121` 배경), 말풍선, 타이핑 애니메이션 |
| `chat.js` | `POST /chat` API 호출, 메시지 렌더링, 대화 히스토리 사이드바 |

- `Enter` 전송 / `Shift+Enter` 줄바꿈
- 봇 응답 대기 중 타이핑 애니메이션
- 새 대화 버튼으로 채팅 초기화

![서버 실행 화면](image-1.png)

---

## 챗봇 답변 예시 (Stage 1 from-scratch)

약 52M 파라미터의 소형 모델이라 완벽하진 않지만, 한국어 문장 구조와 답변 형식을 학습했습니다.

```
👤 한국의 수도는 어디인가요?
🤖 서울에서 10월 2013년 11월 ... 지역인 '북대행'을 가리키는 기념일을 의미합니다.

👤 날씨가 덥다
🤖 우리 몸의 눈이나 꽃잎의 색소를 가지고 있습니다. 이러한 현상은 우리가 말할 때
   발생하는 현상입니다. ...

👤 안녕?
🤖 대한민국 국민으로 인정되며, 그 중 하나인 한국의 ... 국가입니다.
```

> 더 자연스러운 답변을 원하면 Stage 3(KoGPT2 파인튜닝) 모델을 서빙하도록
> `MODEL_PATH`를 교체하면 됩니다.
