from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model_fine_tuning.pt"
TOKENIZER_PATH = BASE_DIR / "my_tokenizer"
WEB_DIR = BASE_DIR / "web"
