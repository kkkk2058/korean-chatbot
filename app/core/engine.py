"""
InferenceEngine — 토크나이저 + from-scratch Transformer 로드 후 텍스트 생성.
체크포인트 state_dict 의 shape 로 아키텍처(vocab/d_model/n_layers/max_seq_len)를
자동 추론하므로, 학습 모델이 바뀌어도 코드 수정 없이 로드된다.
"""
import os

import torch
import torch.nn.functional as F

from app.config import settings
from app.core.model import Transformer
from app.core.tokenizer import BPETokenizer


class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 토크나이저 ---
        if not os.path.exists(settings.TOKENIZER_PATH):
            raise FileNotFoundError(
                f"토크나이저 파일이 없습니다: {settings.TOKENIZER_PATH}\n"
                "Colab에서 학습한 tokenizer.json 을 이 경로에 복사하세요."
            )
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(settings.TOKENIZER_PATH)

        self.pad_id = self.tokenizer.vocab["<pad>"]
        self.bos_id = self.tokenizer.vocab["<s>"]
        self.eos_id = self.tokenizer.vocab["</s>"]

        # --- 모델 ---
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(
                f"모델 체크포인트가 없습니다: {settings.MODEL_PATH}\n"
                "Colab에서 학습한 finetune_checkpoint.pt 를 이 경로에 복사하세요."
            )
        ckpt = torch.load(settings.MODEL_PATH, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # 체크포인트 shape 에서 아키텍처 자동 추론
        vocab_size, d_model = state["embedding.weight"].shape
        max_seq_len = state["positional.weight"].shape[0]
        n_layers = 1 + max(
            int(k.split(".")[1]) for k in state if k.startswith("layers.")
        )

        self.max_seq_len = max_seq_len
        self.model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=settings.N_HEADS,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        print(
            f"[InferenceEngine] device={self.device} "
            f"vocab={vocab_size} d_model={d_model} n_layers={n_layers} "
            f"n_heads={settings.N_HEADS} max_seq_len={max_seq_len}"
        )

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        text = settings.PROMPT_FMT.format(q=prompt.strip())
        ids = [self.bos_id] + self.tokenizer.encode(text)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(settings.MAX_NEW_TOKENS):
            if x.size(1) >= self.max_seq_len:
                break
            logits = self.model(x)[:, -1, :]

            # 반복 페널티: 이미 등장한 토큰 확률 낮춤
            for tok in set(x[0].tolist()):
                logits[0, tok] /= settings.REPETITION_PENALTY

            logits = logits / settings.TEMPERATURE
            top_k = min(settings.TOP_K, logits.size(-1))
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == self.eos_id:
                break
            x = torch.cat([x, next_id], dim=1)

        return self.tokenizer.decode(x[0].tolist()[len(ids):])
