from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, max_seq_len=512):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=d_model,
            n_layer=n_layers,
            n_head=n_heads,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, x):
        output = self.model(input_ids=x)
        return output.logits

    def loss(self, x):
        # [이전 팁 반영] 패딩 토큰(0)은 학습 대상에서 빼서 뇌가 멍청해지는 것을 방지합니다.
        labels = x.clone()
        labels[labels == 0] = -100
        
        output = self.model(input_ids=x, labels=labels)
        return output.loss

    def generate(self, text: str, tokenizer, max_length: int = 100):
        # 1. 텍스트 → 토큰 ID (커스텀 토크나이저 구조에 맞게 ids 추출 안전장치)
        input_ids_list = tokenizer.encode(text)

        # 2. 리스트 → 텐서 변환 및 ★모델과 동일한 장치(CPU/GPU)로 이동★
        device = next(self.parameters()).device # 모델이 cuda에 있으면 cuda를 가져옴
        input_ids = torch.tensor([input_ids_list]).to(device)

        # 3. GPT2 generate 호출 (앵무새 반복 방지 및 부드러운 문장 생성 옵션 추가)
        output_ids = self.model.generate(
            input_ids, 
            max_length=max_length,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            do_sample=True,         # 랜덤 샘플링을 켜야 자연스러운 문장이 나옵니다.
            top_k=50,               # 확률 상위 50개 단어 중 선택
            top_p=0.92,             # 누적 확률 92% 안에서 선택
            no_repeat_ngram_size=3  # 동일한 3단어 조합이 반복되면 강제로 막아줌 (앵무새 방지)
        )

        # 4. 토큰 ID → 텍스트로 복원
        return tokenizer.decode(output_ids[0].tolist())

