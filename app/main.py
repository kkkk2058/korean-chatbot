"""
FastAPI 진입점 — POST /chat API + web/ 정적 UI 서빙.
서버 하나로 API와 채팅 UI를 함께 제공한다.
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.core.engine import InferenceEngine

engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine()  # 서버 기동 시 모델 1회 로드
    yield


app = FastAPI(title="Korean Chatbot (from-scratch)", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = engine.generate(req.message)
    return ChatResponse(response=answer or "(응답을 생성하지 못했습니다)")


# index.html 직접 라우팅 후, 나머지 정적 파일은 / 에 마운트
@app.get("/")
def index():
    return FileResponse(f"{settings.WEB_DIR}/index.html")


app.mount("/", StaticFiles(directory=settings.WEB_DIR), name="web")
