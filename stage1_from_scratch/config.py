# from dataclasses import dataclass


# @dataclass
# class ModelConfig:
#     vocab_size: int = 8000
#     d_model: int = 256
#     n_heads: int = 4
#     n_layers: int = 4
#     max_seq_len: int = 512


# @dataclass
# class TrainConfig:
#     batch_size: int = 64
#     learning_rate: float = 3e-4
#     epochs: int = 10
#     grad_clip: float = 1.0

# 토크나이저
VOCAB_SIZE   = 8_000
TOKENIZER_PATH = "tokenizer.json"

# 모델
D_MODEL      = 256
N_HEADS      = 8
N_LAYERS     = 6
MAX_SEQ_LEN  = 512
DROPOUT      = 0.1

# 학습
BATCH_SIZE   = 32
LR           = 3e-4
EPOCHS       = 10