from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from n_stage.stage3_app.config import WEB_DIR
from n_stage.stage3_app.core.engine import InferenceEngine

app = FastAPI()
engine = InferenceEngine()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return ChatResponse(response=engine.generate(req.message))


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
