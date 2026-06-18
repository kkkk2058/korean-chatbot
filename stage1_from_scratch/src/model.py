import math
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
				
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: 토큰 ID 시퀀스 (batch_size, seq_len)
        batch_size, seq_len = x.size()

        # 1. 위치 인덱스 만들기
        # positions = torch.arange(0, seq_len, device=x.device)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        # 2. embedding + positional 더하기
        # x = self.embedding(x) + self.positional(positions)
        x = self.embedding(x) * math.sqrt(self.d_model) + self.positional(positions)
        
        x = self.dropout(x)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(1) # 미리 4차원 [1, 1, S, S]로 만들어 둡니다.
        # 3. layers 통과
        for layer in self.layers:
            x = layer(x,mask=mask)
        # 4. norm
        x= self.norm(x)
        # 5. linear
        x = self.linear(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)


        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), # 요즘 LLM 표준
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Attention + Add & Norm
        # x = self.norm1(x + self.attention(x, mask))
        x = x + self.dropout1(self.attention(self.norm1(x), mask))
        # 2. FeedForward + Add & Norm
        # x = self.norm2(x + self.ff(x))
        x = x + self.dropout2(self.ff(self.norm2(x)))

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 헤드 하나당 차원

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


        self.attn_dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 1. Q, K, V 만들기
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Attention 계산
        # QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, n_heads, seq, seq)

        # √d_k 로 나누기
        scores = scores / math.sqrt(self.d_head)

        # 4. 마스크 적용 (미래 토큰 가리기)
        if mask is not None:
        # 마스크가 있으면 가려야 할 위치를 -∞로 채워 Softmax에서 0이 되게 함
            # scores = scores.masked_fill(mask == 0, float('-inf'))
            # FP16/FP32 환경 모두에서 100% 안전한 상숫값 사용
            scores = scores.masked_fill(~mask, -1e4) 

        # softmax 점수를 확률(참고 비율)로 변환
        attn = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)

        attn = self.attn_dropout(attn)

        # * V
        x = torch.matmul(attn, V)  # (batch, n_heads, seq, d_head)

        # 5. 헤드 합치기
        # (batch, n_heads, seq, d_head) → (batch, seq, d_model)
        x = x.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)

        # 6. 출력 Linear
        x = self.W_o(x)
        return x