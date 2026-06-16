from dataclasses import dataclass


@dataclass
class TrainConfig:
    # 데이터
    max_length: int = 256
    test_size: float = 0.05
    seed: int = 42

    # 학습
    batch_size: int = 32
    gradient_accumulation_steps: int = 2   # 실질 배치: batch_size * gradient_accumulation_steps
    learning_rate: float = 3e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    epochs: int = 5
    early_stopping_patience: int = 2
    grad_clip: float = 1.0
    bf16: bool = True                      # A100 환경

    # 체크포인트
    output_dir: str = "checkpoints"
    save_total_limit: int = 2

    # 저장 경로
    model_save_path: str = "models/model_fine_tuning.pt"
    tokenizer_save_path: str = "./my_tokenizer"


@dataclass
class GenerateConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3
