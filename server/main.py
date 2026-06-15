from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import sys
from src.tokenizer import load_tokenizer
from src.model import Transformer


app = FastAPI()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = load_tokenizer()
model = Transformer()
model.load_state_dict(torch.load("models/model.pt", map_location=device))
model.to(device)
model.eval()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = f"### 질문: {req.message}\n### 답변:"
    response = model.generate(prompt, tok)
    
    if "### 답변:" in response:
        response = response.split("### 답변:")[-1].strip()
    
    return {"response": response}

# 맨 마지막에 추가
app.mount("/", StaticFiles(directory="web", html=True), name="web")