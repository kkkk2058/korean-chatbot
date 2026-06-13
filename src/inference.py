from tokenizers import Tokenizer
from model import Transformer
import torch


class InferenceEngine:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 토크나이저 로드
        self.tokenizer_obj = Tokenizer.from_file(tokenizer_path)
        vocab_size = self.tokenizer_obj.get_vocab_size()
        
        # 모델 로드
        self.model = Transformer(vocab_size=vocab_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
        ids = self.tokenizer_obj.encode(prompt).ids
        input_tensor = torch.tensor([ids]).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_tensor)
                
                # 마지막 토큰의 logits
                next_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # EOS면 종료
                if next_token.item() == 2:
                    break
                
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=-1)
        
        generated_ids = input_tensor[0].tolist()
        return self.tokenizer_obj.decode(generated_ids)