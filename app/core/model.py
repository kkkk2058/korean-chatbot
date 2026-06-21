"""
Decoder-only Transformer (from-scratch).
stage1_from_scratch/src/model.py 의 추론용 사본 — 서빙 레이어를 독립적으로 유지.
구조가 동일해야 학습된 가중치를 그대로 로드할 수 있음.
"""
import math

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = (
            torch.arange(0, seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        x = self.embedding(x) * math.sqrt(self.d_model) + self.positional(positions)
        x = self.dropout(x)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return self.linear(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.attention(self.norm1(x), mask))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(x)
