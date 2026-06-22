import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from n_stage.stage3_app.config import MODEL_PATH, TOKENIZER_PATH
from n_stage.stage3_fine_tuning.config import GenerateConfig


class InferenceEngine:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_cfg = GenerateConfig()

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

        self.model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        enc = self.tokenizer(f"<usr>{prompt}<bot>", return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=self.gen_cfg.temperature,
            top_p=self.gen_cfg.top_p,
            repetition_penalty=self.gen_cfg.repetition_penalty,
            no_repeat_ngram_size=self.gen_cfg.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        if "<bot>" in result:
            result = result.split("<bot>")[-1]
        return result.replace("</s>", "").strip()
