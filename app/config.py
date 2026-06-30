"""
서빙 설정 — 경로, 모델 아키텍처 보조값, 생성 파라미터를 한 곳에서 관리.
환경변수로 override 가능 (배포 시 MODEL_PATH 등 주입).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # 레포 루트


class Settings:
    # --- 경로 ---
    # from-scratch 모델 체크포인트 (Colab에서 학습 후 Drive → 여기로 복사)
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "model_scratch.pt"),
    )
    # 커스텀 BPE 토크나이저 (BPETokenizer.save() 형식: vocab/merges/special_tokens)
    TOKENIZER_PATH = os.getenv(
        "TOKENIZER_PATH",
        str(BASE_DIR / "stage1_from_scratch" / "tokenizer" / "tokenizer.json"),
    )
    WEB_DIR = str(BASE_DIR / "web")

    # --- 모델 아키텍처 ---
    # vocab_size / d_model / n_layers / max_seq_len 은 체크포인트에서 자동 추론.
    # n_heads 만 shape 로 추론 불가하므로 명시 (학습 때 config.N_HEADS 와 동일해야 함).
    N_HEADS = int(os.getenv("N_HEADS", "8"))

    # --- 생성 파라미터 (Colab generate 와 동일) ---
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
    TOP_K = int(os.getenv("TOP_K", "30"))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.3"))
    NO_REPEAT_NGRAM = int(os.getenv("NO_REPEAT_NGRAM", "0"))  # 0=끔(Colab과 동일), 루프 재발 시 3으로 켜는 안전장치

    # 파인튜닝 때 사용한 프롬프트 형식과 반드시 일치해야 함
    PROMPT_FMT = "질문: {q}\n답변:"


settings = Settings()
