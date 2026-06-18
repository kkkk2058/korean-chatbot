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
VOCAB_SIZE   = 16_000
TOKENIZER_PATH = "tokenizer.json"

# 모델
D_MODEL      = 512
N_HEADS      = 8
N_LAYERS     = 12
MAX_SEQ_LEN  = 512
DROPOUT      = 0.1

# 사전학습
PRETRAIN_BATCH_SIZE   = 16
PRETRAIN_GRAD_ACCUM   = 4       # effective batch = 64
PRETRAIN_LR           = 3e-4
PRETRAIN_EPOCHS       = 3

# 파인튜닝
FINETUNE_BATCH_SIZE   = 16
FINETUNE_GRAD_ACCUM   = 4
FINETUNE_LR           = 1e-4    # 사전학습보다 낮게
FINETUNE_EPOCHS       = 10