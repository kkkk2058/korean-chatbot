# Korean Chatbot 프로젝트

한국어 언어모델을 **바닥부터 단계적으로 구현**하는 학습 프로젝트입니다.
토크나이저와 Transformer를 라이브러리 없이 직접 만드는 것에서 시작해,
HuggingFace 라이브러리 활용, KoGPT2 파인튜닝까지 3단계로 발전시킵니다.

**현재 서버는 Stage 1(직접 구현) 모델을 서빙합니다.** (토크나이저는 v2 — HuggingFace ByteLevel BPE)

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
│       ├── tokenizer.py          # (서빙) HF tokenizers 어댑터 — load/encode/decode
│       ├── tokenizer_custom.py   # v1 자작 BPE 로더 (보존)
│       └── engine.py             # InferenceEngine (모델 로드 + 생성)
├── web/                          # 프론트엔드 (정적)
│   ├── index.html · style.css · chat.js
├── stage1_from_scratch/          # ① 바닥부터 직접 구현 (현재 서빙)
│   ├── config.py                 # 토크나이저·모델·학습 하이퍼파라미터
│   ├── src/
│   │   ├── model.py              # Transformer (순수 PyTorch)
│   │   └── tokenizer.py          # BPETokenizer (순수 Python, v1)
│   ├── tokenizer/                # 학습된 토크나이저 (vocab 16k)
│   └── colab/
│       ├── train_tokenizer.ipynb # (v1) KoWiki로 자작 BPE 학습 → tokenizer.json (1회성)
│       ├── train_model.ipynb     # (v1) 자작 토크나이저 + 사전학습 → 파인튜닝
│       └── train_model_v2.ipynb  # (v2) HF ByteLevel 토크나이저, 그 외 v1과 동일
├── stage2_with_library/          # ② HuggingFace 라이브러리 활용
├── stage3_fine_tuning/           # ③ KoGPT2 파인튜닝
├── models/                       # 학습된 가중치 (git 제외)
│   └── model_scratch.pt          # ← 현재 서빙 중인 Stage 1 모델 (v2 파인튜닝본)
├── data/                         # 학습 데이터 (git 제외)
├── Makefile · requirements.txt
```

---

## 구현 단계별 비교

| 구분 | Stage 1 (직접 구현) | Stage 2 (HF 라이브러리) | Stage 3 (파인튜닝) |
|---|---|---|---|
| 토크나이저 | 순수 Python BPE (v1) / HF ByteLevel (v2) | `tokenizers` 라이브러리 | KoGPT2 pretrained |
| 모델 | 순수 PyTorch Transformer | GPT2Config 기반 | skt/kogpt2-base-v2 |
| vocab size | 16,000 (위키피디아) | 8,000 (나무위키) | 51,200 (KoGPT2) |
| 학습 데이터 | 위키 사전학습 + KoAlpaca 파인튜닝 | 나무위키 | KoAlpaca |
| 목적 | 내부 원리 이해 | 실용적 구현 | 성능 극대화 |

---

## Stage 1 — 바닥부터 직접 구현 (현재 서빙 버전)

라이브러리 없이 BPE 토크나이저와 Transformer를 직접 구현하고,
**한국어 위키피디아로 사전학습 → KoAlpaca로 파인튜닝**하는 전체 파이프라인입니다.

### 두 가지 버전 (토크나이저 통제 비교)

같은 데이터·같은 모델·같은 학습 파이프라인을 쓰되 **토크나이저만 다른** 두 노트북을 둡니다.
→ `train_model.ipynb`(v1) ↔ `train_model_v2.ipynb`(v2) 비교 = **순수 토크나이저 효과** 측정.

| | v1 (`train_model.ipynb`) | v2 (`train_model_v2.ipynb`) |
|---|---|---|
| 토크나이저 | 자작 순수 Python BPE (음절 위주) | HF `tokenizers` ByteLevel BPE (subword) |
| 토크나이저 학습 | `train_tokenizer.ipynb`에서 사전 학습 (느림, 1회성) | 노트북 내장 (Rust, 빠름) |
| 데이터/모델/학습 | **동일** | **동일** |
| 산출물 접미사 | `_custom` (v2와 분리 저장) | 기본 |

> 토크나이저는 한 번 학습 후 **고정(freeze)**. 사전학습·파인튜닝·추론이 같은 토크나이저를 재사용합니다.

### 학습 흐름

```
[v1] KoWiki → 자작 BPE(vocab 16k) 학습 → tokenizer.json
[v2] KoWiki → HF ByteLevel BPE(vocab 16k) (노트북 내장)
                    ↓
KoWiki 전체 → 블록 청킹(512 토큰, 패딩 0%) → 사전학습 15 epoch
              (FP16, eff.batch 64, lr 3e-4 cosine, best 체크포인트)
                    ↓ (가중치 이어받기)
KoAlpaca 21k → 파인튜닝 5 epoch (답변만 -100 마스킹, lr 1e-4, best-val 저장)
                    ↓
생성 (temperature 0.8, top_k 30, repetition_penalty 1.3, no_repeat_ngram 옵션)
```

### 토크나이저: `stage1_from_scratch/src/tokenizer.py`

`BPETokenizer`를 순수 Python으로 구현 (v1).

- 특수 토큰 4종: `<pad>`, `<unk>`, `<s>`, `</s>`
- BPE로 자주 등장하는 쌍을 반복 병합해 vocab(16,000) 확장
- 속도 최적화: **증분 pair 카운트 + pair 역인덱스 + 단어 단위 인코딩 캐시**
- 한국어 완성형 음절(약 11k)이 기초 토큰을 차지 → 실제 merge는 ~4.3k(음절 위주 분해)

> v2는 HF `tokenizers`의 ByteLevel BPE를 사용. 바이트 단위라 공백·줄바꿈을 손실 없이 복원하고,
> 기초 토큰이 256바이트뿐이라 merge 예산을 전부 subword에 쓸 수 있습니다.

### 모델: `stage1_from_scratch/src/model.py`

Decoder-only Transformer를 PyTorch 기본 연산만으로 구현.

- 토큰 임베딩 + 위치 임베딩(`nn.Embedding`)
- `MultiHeadAttention`: Q/K/V 분리 → Scaled Dot-Product → 인과 마스크
- `DecoderBlock`: Pre-LayerNorm (Attention → Add → FFN → Add), GELU

### 파라미터: `stage1_from_scratch/config.py`

| 구분 | 항목 | 값 |
|---|---|---|
| 토크나이저 | `VOCAB_SIZE` | 16,000 |
| 모델 | `D_MODEL` / `N_HEADS` / `N_LAYERS` | 512 / 8 / 12 |
| 모델 | `MAX_SEQ_LEN` / `DROPOUT` | 512 / 0.1 |
| 사전학습 | batch × grad_accum | 16 × 4 (실질 64) |
| 사전학습 | `PRETRAIN_LR` / `EPOCHS` | 3e-4 / **15** |
| 파인튜닝 | batch × grad_accum | 16 × 4 (실질 64) |
| 파인튜닝 | `FINETUNE_LR` / `EPOCHS` | **1e-4** / **5** |

> 파라미터 수 약 52M.

### 사전학습 (위키피디아) — 데이터 파이프라인

- **블록 청킹**: 줄 단위+패딩 대신, 전체 텍스트를 하나의 토큰 스트림으로 이어붙여 `MAX_SEQ_LEN(512)` 블록으로 분할
  → **패딩 0%, 문맥 100% 활용** (nanoGPT/GPT-2 방식). 문서 경계는 `<s>`/`</s>`로 표시.
- **데이터 캡 해제**: 모은 위키 전체를 학습 (`MAX_TRAIN_LINES=None`)
- 블록 캐시(`pretrain_blocks.pt` / v1은 `_custom`)를 Drive에 저장 → 재실행 시 즉시 로드
- **FP16 + gradient accumulation**으로 A100 내 학습
- 체크포인트에 scheduler/scaler state까지 저장 → 이어학습 시 LR 폭등 방지
- 매 epoch **val perplexity** 로깅

> ⚠️ v1(자작 토크나이저)은 순수 Python `encode`라 전체 위키 토큰화가 매우 느립니다(수 시간).
> 빠른 실험은 `WIKI_DOCS`를 줄이거나, `_tokenize_word`를 rank 기반으로 최적화하세요.

### 파인튜닝 (KoAlpaca)

- 프롬프트 형식: `질문: {q}\n답변: {a}`
- **답변 토큰만 loss 계산** (질문은 `-100` 마스킹) → "답변하는 모델"로 전환되는 핵심
- 사전학습 가중치를 이어받아 **lr 1e-4**(사전학습보다 낮게, 망각 방지)로 5 epoch
- **best-val 체크포인트만 저장** → overfit된 마지막 epoch 대신 검증 최저점 보존

### 생성

| 파라미터 | 값 | 역할 |
|---|---|---|
| `temperature` | 0.8 | 다양성 |
| `top_k` | 30 | 상위 토큰만 샘플링 |
| `repetition_penalty` | 1.3 | 반복 억제 |
| `max_new_tokens` | 100 | 길이 제한 |
| `no_repeat_ngram` | 0(기본) | 같은 n-gram 반복 차단 (루프 시 3으로) |

---

## 정성적 결과 (시연)

사전학습을 충분히 늘리자 **"유창한 헛소리 → 질문에 붙는 답변"**으로 개선됐습니다.

**사전학습 epoch에 따른 변화 (v2 기준)**

| 사전학습 | val perplexity | "한국의 수도는?" 답변 |
|---|---|---|
| 3 epoch | ~134 | 호주·뉴질랜드 등 **완전 무관** (유창하나 헛소리) |
| 15 epoch | ~40 | "**서울**을 수도로 정한 이유는…" (질문에 붙음) |

```
👤 한국의 수도는 어디인가요?
🤖 (v2) 한국은 '서울의 강남'이라는 뜻으로, 서울을 수도로 정한 이유는 …
👤 안녕?
🤖 (v2) 안녕하세요! …
👤 핸드폰 추천해줘
🤖 (v2) 핸드폰 추천해봐요? 1. …
```

> 문법·주제 인식·답변 형식은 학습됨. 사실 정확성·긴 문맥 일관성은 여전히 약함(아래 한계점).

---

## 비교 평가 시스템 (토크나이저 v1 ↔ v2)

### 핵심 발견: perplexity는 토크나이저가 다르면 비교 불가

같은 데이터·모델·파이프라인에서 토크나이저만 바꿔 비교한 결과:

| | v1 (자작 BPE) | v2 (HF ByteLevel) |
|---|---|---|
| 사전학습 val ppl | **~32 (더 낮음)** | ~40 |
| 파인튜닝 val ppl | **~24 (더 낮음)** | ~32 |
| "안녕?" 인사 인식 | ❌ (2회 연속 실패) | ✅ |
| "핸드폰 추천" 요청 인식 | ❌ | ✅ |
| 질문에 붙는 답변 | 약함 | 상대적으로 강함 |

**v1이 perplexity는 더 낮지만 생성 품질은 더 나쁩니다.** 이유:
- v1은 거의 음절 단위라 단어 내부 음절 예측이 거의 결정적 → loss가 **거저** 낮아짐(정보량당으론 불리)
- 평균 loss를 쉬운 음절 토큰이 지배 → 정작 중요한 내용 토큰은 묻혀 모델이 얕은 통계만 학습
- 토큰 수가 많아 "질문 의도"가 더 긴 시퀀스를 견뎌야 함 → 생성 중 드리프트

→ **교훈: 토크나이저가 다르면 ppl로 비교하면 안 됨.** 글자당 손실(bits-per-char) 또는 정성 평가로 판단.
좋은 subword 토크나이저(v2)는 토큰 효율이 높아 실제 응답 품질을 끌어올림.

### 평가 데이터셋 (질문 + 정답, 12문항)

| # | 카테고리 | 질문 | 정답(핵심) |
|---|---|---|---|
| 1 | 사실 | 한국의 수도는 어디인가요? | 서울 |
| 2 | 사실 | 물은 어떤 원소로 이루어져 있나요? | 수소와 산소 (H₂O) |
| 3 | 사실 | 태양계에서 가장 큰 행성은 무엇인가요? | 목성 |
| 4 | 사실 | 한글을 만든 사람은 누구인가요? | 세종대왕 |
| 5 | 사실 | 1년은 며칠인가요? | 365일 (윤년 366일) |
| 6 | 정의 | 광합성이란 무엇인가요? | 식물이 빛으로 CO₂·물에서 양분과 산소를 만드는 과정 |
| 7 | 정의 | 중력이란 무엇인가요? | 질량이 있는 물체끼리 끌어당기는 힘 |
| 8 | 추론 | 얼음이 녹으면 무엇이 되나요? | 물 |
| 9 | 추론 | 비가 올 때 무엇을 챙기면 좋을까요? | 우산 |
| 10 | 계산 | 2 더하기 3은 얼마인가요? | 5 |
| 11 | 대화 | 안녕하세요? | 인사 응대 |
| 12 | 대화 | 고마워 | 천만에요/별말씀을 류 |

### 채점 기준

| 정확도 | 점수 | | 유창성 | 점수 |
|---|---|---|---|---|
| 정답·핵심 포함 | 2 | | 자연스러운 한국어 | 2 |
| 부분/모호 | 1 | | 비문 일부 | 1 |
| 오답·헛소리 | 0 | | 깨짐/반복 | 0 |

> 항목당 정확도(0~2)+유창성(0~2), 12문항 합산. 카테고리별 평균도 함께 보고.
> 생성은 확률 샘플링이라 **문항당 2~3회** 돌려 집계하고, 토크나이저 비교 시 ppl 대신 이 점수(또는 bpc)를 사용.

---

## 현재 한계점

| 한계 | 원인 |
|---|---|
| 사실 정확도 낮음 | 위키≠QA 지식 형태 + 약 52M 소형 모델 |
| 데이터 부족(data-starved) | 50M 모델 대비 토큰 수 부족 → 지식 용량 한계 |
| 단일턴만 가능 | KoAlpaca가 단일턴 Q&A |
| 긴 답변 일관성·꼬리 깨짐 | 소형 모델 + 샘플링; 긴 생성일수록 드리프트 |

해결된 것: 토크나이저 깨짐/`<unk>` 없음, 공백·줄바꿈 보존(v2), 학습 파이프라인 안정성, 반복 루프(no-repeat-ngram).

---

## 앞으로의 방향성

1. **데이터 확대·다양화** (가장 큰 레버) — 위키 외 뉴스·교과서 등 추가
2. **측정하며 스케일업** — val ppl(또는 bpc)을 추적하며 모델 크기↑ (데이터가 받쳐줄 때만)
3. **멀티턴 + 도구 호출** instruction 튜닝
4. **구조 개선** — weight tying, LR warmup

> 데이터 먼저, 모델은 따라온다 — 데이터를 키우기 전에 모델만 키우면 overfit만 심해짐.

---

## Stage 2 — HuggingFace 라이브러리 활용

### 토크나이저: `stage2_with_library/src/tokenizer.py`

HuggingFace `tokenizers`의 BPE 모델 활용.

- `ByteLevel` 전처리 + `ByteLevelDecoder`로 UTF-8 안전 처리
- 파일 기반 스트리밍 학습으로 대용량도 메모리 효율적
- `vocab.json` 단일 파일로 저장·로드

### 모델: `stage2_with_library/src/model.py`

`GPT2Config`로 하이퍼파라미터를 지정하고 `GPT2LMHeadModel` 초기화.
내부 구조는 Stage 1과 동일하지만 구현을 라이브러리에 위임.

```python
from config import ModelConfig, TrainConfig, TokenizerConfig
model_cfg = ModelConfig()   # vocab_size=8000, d_model=256, n_heads=4, n_layers=4
train_cfg = TrainConfig()   # batch_size=64, lr=3e-4, epochs=10, bf16=True
tok_cfg = TokenizerConfig() # vocab_size=8000, data_path, save_path
```

---

## Stage 3 — KoGPT2 파인튜닝

`skt/kogpt2-base-v2` 사전학습 모델을 KoAlpaca로 파인튜닝.

| 항목 | 값 |
|---|---|
| 베이스 모델 | skt/kogpt2-base-v2 |
| 데이터셋 | beomi/KoAlpaca-v1.1a |
| batch_size | 32 (grad_accum×2 → 실질 64) |
| learning_rate | 3e-5 (cosine, warmup 10%) |
| epochs | 최대 5 (EarlyStopping patience=2) |
| 정밀도 | bf16 (A100) |

학습 프롬프트 포맷:

```
<usr>질문 내용<bot>답변 내용</s>
```

답변(`<bot>` 이후)만 loss 계산, 질문은 `-100` 마스킹 (Stage 1과 동일 원리).

---

## Colab 실행 가이드

### Stage 1

```
train_tokenizer.ipynb   → (v1) KoWiki 자작 BPE 학습 → tokenizer.json (1회성, 느림)
train_model.ipynb       → (v1) 토크나이저 로드 → 사전학습 → 파인튜닝 → 생성
train_model_v2.ipynb    → (v2) HF 토크나이저 내장 학습 → 이하 동일
```
각 노트북을 위에서 아래로 실행 (A100 권장). 체크포인트·캐시는 Drive에 저장돼 이어학습 지원.
처음부터 다시 학습하려면 옛 `pretrain_checkpoint*.pt`를 삭제 후 실행 (LR 스케줄 재설정).

### Stage 2 / Stage 3

```
[Stage2] prepare_data → train_tokenizer → train_model → inference
[Stage3] prepare_data_koAlpaca → fine_tunning_koAlpaca
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

`app/`은 **Stage 1 from-scratch 모델**(`models/model_scratch.pt`, v2 파인튜닝본)을 서빙합니다.

```bash
git clone https://github.com/kkkk2058/korean-chatbot.git
cd korean-chatbot
make install   # pip install -r requirements.txt

make run       # uvicorn app.main:app --reload
# → http://localhost:8000  (브라우저로 열면 채팅 UI)
```

### 모델 파일 준비

서버는 아래 두 파일이 필요하며, **반드시 같은 학습 세션의 짝**이어야 합니다 (토크나이저↔모델 불일치 시 출력이 한자·기호로 깨짐):

```
models/model_scratch.pt                         # Stage 1 모델 가중치
stage1_from_scratch/tokenizer/tokenizer.json    # 모델과 짝인 토크나이저 (서빙: HF ByteLevel)
```

- 현재 서빙 모델은 **v2(HF ByteLevel)** 이므로 토크나이저도 HF 포맷이어야 합니다.
- 앱 토크나이저 로더는 `app/core/tokenizer.py`(HF 어댑터). v1 자작 BPE 로더는 `tokenizer_custom.py`에 보존.
- 경로는 환경변수 `MODEL_PATH` / `TOKENIZER_PATH`로 override 가능.
- 모델 아키텍처(vocab·d_model·n_layers·max_seq_len)는 체크포인트 shape에서 **자동 추론**, `n_heads`만 학습 때 값과 일치시키면 됩니다(`N_HEADS`).

> 짝 확인: 생성이 한국어로 일관되게 나오면 정상. 한자·외국어 토큰이 무작위로 섞이면 토크나이저-모델 불일치.

### API 스펙

```
POST /chat
Body:     { "message": "안녕하세요" }
Response: { "response": "..." }
```

`web/` 폴더는 `/` 경로에 정적 마운트되어 서버 하나로 API와 UI를 함께 제공합니다.

### `app/` 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 경로 + 생성 파라미터(temperature/top_k/repetition_penalty/no_repeat_ngram) |
| `core/model.py` | Decoder-only Transformer |
| `core/tokenizer.py` | HF tokenizers 어댑터 (추론 전용 load/encode/decode) |
| `core/tokenizer_custom.py` | v1 자작 BPE 로더 (보존) |
| `core/engine.py` | `InferenceEngine` — 체크포인트 자동 추론 로드 + 생성 |
| `main.py` | FastAPI `POST /chat` + `web/` 정적 UI 서빙 |

---

## 웹 UI

ChatGPT 스타일의 다크테마 채팅 인터페이스.

| 파일 | 역할 |
|---|---|
| `index.html` | 사이드바 + 채팅 영역 레이아웃 |
| `style.css` | 다크테마(`#212121`), 말풍선, 타이핑 애니메이션 |
| `chat.js` | `POST /chat` 호출, 메시지 렌더링, 히스토리 사이드바 |

- `Enter` 전송 / `Shift+Enter` 줄바꿈
- 봇 응답 대기 중 타이핑 애니메이션
- 새 대화 버튼으로 초기화

![서버 실행 화면](image-1.png)
