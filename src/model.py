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



# import math
# import torch
# import torch.nn as nn

# class Transformer(nn.Module):
#     def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.positional = nn.Embedding(max_seq_len, d_model)

#         self.layers = nn.ModuleList([
#             DecoderBlock(d_model, n_heads)
#             for _ in range(n_layers)
#         ])

#         self.norm = nn.LayerNorm(d_model)
#         self.linear = nn.Linear(d_model, vocab_size)

#     def forward(self, x):
#         # x: 토큰 ID 시퀀스 (batch_size, seq_len)
#         batch_size, seq_len = x.size()

#         # 1. 위치 인덱스 만들기
#         positions = torch.arange(0, seq_len, device=x.device)
#         # 2. embedding + positional 더하기
#         x = self.embedding(x) + self.positional(positions)

#         mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()

#         # 3. layers 통과
#         for layer in self.layers:
#             x = layer(x,mask=mask)
#         # 4. norm
#         x= self.norm(x)
#         # 5. linear
#         x = self.linear(x)

#         return x

# class DecoderBlock(nn.Module):
#     def __init__(self, d_model, n_heads):
#         super().__init__()
#         self.attention = MultiHeadAttention(d_model, n_heads)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(), # 요즘 LLM 표준
#             nn.Linear(d_model * 4, d_model),
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, x, mask=None):
#         # 1. Attention + Add & Norm
#         x = self.norm1(x + self.attention(x, mask))

#         # 2. FeedForward + Add & Norm
#         x = self.norm2(x + self.ff(x))

#         return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads):
#         super().__init__()
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads  # 헤드 하나당 차원

#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
#         self.W_o = nn.Linear(d_model, d_model)

#     def forward(self, x, mask=None):
#         batch_size, seq_len, d_model = x.size()

#         # 1. Q, K, V 만들기
#         Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
#         K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
#         V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

#         # 3. Attention 계산
#         # QK^T
#         scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, n_heads, seq, seq)

#         # √d_k 로 나누기
#         scores = scores / math.sqrt(self.d_head)

#         # 4. 마스크 적용 (미래 토큰 가리기)
#         # 마스크가 있으면 가려야 할 위치를 -∞로 채워 Softmax에서 0이 되게 함
#         if mask is not None:
#             mask = mask.unsqueeze(0).unsqueeze(1) # 차원맞추기 브로드캐스팅사용
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         # softmax 점수를 확률(참고 비율)로 변환
#         attn = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)

#         # * V
#         x = torch.matmul(attn, V)  # (batch, n_heads, seq, d_head)

#         # 5. 헤드 합치기
#         # (batch, n_heads, seq, d_head) → (batch, seq, d_model)
#         x = x.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)

#         # 6. 출력 Linear
#         x = self.W_o(x)
#         return x