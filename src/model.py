from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # KoGPT2 불러오기 (바닥부터 학습 X)
        self.model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    def forward(self, x):
        output = self.model(input_ids=x)
        return output.logits

    def loss(self, x, pad_token_id):
        labels = x.clone()
        labels[labels == pad_token_id] = -100  # 패딩 무시
        output = self.model(input_ids=x, labels=labels)
        return output.loss

    def generate(self, text: str, tokenizer, max_length: int = 100):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            no_repeat_ngram_size=3
        )

        return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)