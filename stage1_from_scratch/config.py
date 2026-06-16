from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 10
    grad_clip: float = 1.0
