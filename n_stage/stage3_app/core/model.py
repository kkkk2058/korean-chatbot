import torch.nn as nn
from transformers import GPT2LMHeadModel


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    def forward(self, x):
        return self.model(input_ids=x).logits

    def loss(self, x, pad_token_id):
        labels = x.clone()
        labels[labels == pad_token_id] = -100
        return self.model(input_ids=x, labels=labels).loss
