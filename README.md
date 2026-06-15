# 🚀 3-Step Korean LLM Scratch Project

한국어 언어모델(LLM)을 밑바닥부터 단계별로 구현하며 학습하는 프로젝트입니다. 
토크나이저와 모델 아키텍처를 직접 손으로 구현하는 것부터 시작하여, HuggingFace 라이브러리 활용, 최종적으로 Pretrained 모델을 로드하여 파인튜닝하는 것까지 총 3단계의 커리큘럼으로 구성되어 있습니다.

---

## 📌 프로젝트 핵심 흐름
> **"직접 짜며 이해하고, 라이브러리로 효율을 높이고, 사전학습 모델로 실전 적용하기"**
1. **[Step 0] Scratch 구현:** 외부 라이브러리를 최소화하고 토크나이저와 Transformer(GPT-2) 아키텍처를 바닥부터 직접 구현합니다.
2. **[Step 1] Library 최적화:** HuggingFace의 라이브러리들을 활용해 직접 구현했던 로직을 간결하고 강력하게 리팩토링합니다.
3. **[Step 2] Pretrained & Fine-tuning:** 이미 대규모 데이터로 학습된 KoGPT2 모델과 토크나이저를 가져와 한국어 생성 성능을 극대화합니다.

---

## 🛠️ 컴포넌트별 상세 요약

### 1. 토크나이저 (Tokenizer)
| 파일명 | 방식 | 주요 특징 |
| :--- | :--- | :--- |
| `tokenizer_0.py` | **BPE 직접 구현** | 한글 전체(가~힣) 및 ASCII 추가, `tqdm`을 활용한 진행 상황 시각화 |
| `tokenizer_bpe.py` | **HuggingFace 래퍼** | `tokenizers` 라이브러리의 ByteLevel BPE 활용, 파일 기반 학습 지원 |
| `tokenizer.py` | **Pretrained 로드** | KoGPT2의 `PreTrainedTokenizerFast`를 그대로 가져와 사용 |

### 2. 모델 (Model)
| 파일명 | 방식 | 주요 특징 |
| :--- | :--- | :--- |
| `model_0.py` | **Transformer 직접 구현** | `MultiHeadAttention`, `DecoderBlock`, 인과 마스크(Causal Mask) 등 핵심 로직 수동 작성 |
| `model_gpt2.py` | **GPT2 아키텍처 활용** | `GPT2Config` + `GPT2LMHeadModel` 조합, 하이퍼파라미터(`vocab_size`, `d_model` 등) 직접 지정 및 초기화 |
| `model.py` | **KoGPT2 Fine-tuning** | 사전학습된 `skt/kogpt2-base-v2`를 로드하여 한국어 다운스트림 태스크에 맞게 파인튜닝 |

### 3. 추론 엔진 (Inference Engine)
* **`inference.py`**
  * `tokenizer_bpe.py`(라이브러리 기반 토크나이저)와 `model_gpt2.py`(GPT2 구조 모델)를 조합하여 작동합니다.
  * **Temperature 기반 샘플링** 알고리즘이 적용되어 있어, 텍스트 생성의 창의성과 다양성을 조절할 수 있는 `InferenceEngine`을 제공합니다.

---

## 📂 디렉토리 구조
```bash
.
├── tokenizer_0.py      # Step 0: 순수 Python/PyTorch 기반 BPE 구현
├── tokenizer_bpe.py    # Step 1: HuggingFace Tokenizers 래퍼
├── tokenizer.py        # Step 2: KoGPT2 Pretrained 토크나이저
├── model_0.py          # Step 0: Transformer Decoder 바닥부터 구현
├── model_gpt2.py       # Step 1: GPT2 구조 직접 지정 및 초기화
├── model.py            # Step 2: KoGPT2 Pretrained 모델 로드
└── inference.py        # 🚀 통합 추론 엔진 (Temperature 샘플링 지원)