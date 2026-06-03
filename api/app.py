"""
api/app.py
==========
FastAPI REST API للمشروع.

Endpoints المتاحة:
- POST /chat          → محادثة عادية مع الـ LLM
- POST /rag/query     → سؤال على الـ RAG
- POST /rag/add-text  → إضافة نص للـ knowledge base
- POST /analyze       → تحليل مشاعر
- POST /summarize     → تلخيص نص
- GET  /health        → التحقق من حالة الـ API
- GET  /stats         → إحصائيات الـ system

طريقة التشغيل:
    uvicorn api.app:app --reload
    أو
    python main.py --mode api
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings, LLMProvider
from core.openai_llm import OpenAILLM
from core.ollama_llm import OllamaLLM
from core.chat_manager import ChatManager
from rag.rag_pipeline import RAGPipeline
from llm.structured_output import StructuredOutputGenerator
from llm.schemas import ChatResponse, RAGResponse, SentimentResponse, SummaryResponse


# ==========================================
# Request/Response Models للـ API
# ==========================================

class ChatRequest(BaseModel):
    """طلب المحادثة."""
    message: str = Field(..., min_length=1, max_length=10000)
    system_prompt: Optional[str] = None
    stream: bool = False


class RAGQueryRequest(BaseModel):
    """طلب السؤال على الـ RAG."""
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0)


class AddTextRequest(BaseModel):
    """طلب إضافة نص للـ knowledge base."""
    text: str = Field(..., min_length=10)
    source_name: str = Field(default="api_input")


class AnalyzeRequest(BaseModel):
    """طلب تحليل نص."""
    text: str = Field(..., min_length=1, max_length=5000)


# ==========================================
# تهيئة الـ App (Startup / Shutdown)
# ==========================================

# Global objects
llm = None
chat_manager = None
rag_pipeline = None
structured_output = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager للـ FastAPI.
    بيشغّل كود عند بدء التطبيق وعند إيقافه.
    """
    global llm, chat_manager, rag_pipeline, structured_output

    print("🚀 بدء تشغيل الـ API...")

    # إنشاء الـ LLM المناسب
    if settings.llm_provider == LLMProvider.OPENAI:
        llm = OpenAILLM()
        print("✅ OpenAI LLM جاهز")
    else:
        llm = OllamaLLM()
        print("✅ Ollama LLM جاهز")

    # إنشاء باقي الـ components
    chat_manager = ChatManager(llm=llm)
    rag_pipeline = RAGPipeline(llm=llm)
    structured_output = StructuredOutputGenerator(llm=llm)

    print("✅ كل الـ components جاهزة - الـ API شغّال!")

    yield  # هنا الـ app بيشتغل

    # Cleanup عند الإيقاف
    print("👋 إيقاف الـ API...")


# ==========================================
# إنشاء الـ FastAPI App
# ==========================================

app = FastAPI(
    title="🤖 AI Project API",
    description="API لمشروع AI متكامل مع LLM + RAG + Structured Outputs",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - يسمح للـ frontend يتكلم مع الـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # في production غيّر ده لـ domain محدد
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# API Endpoints
# ==========================================

@app.get("/health")
async def health_check():
    """التحقق من حالة الـ API والـ LLM."""
    llm_healthy = await llm.health_check() if llm else False
    return {
        "status": "healthy" if llm_healthy else "degraded",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.openai_model if settings.llm_provider == LLMProvider.OPENAI else settings.ollama_model,
        "llm_status": "online" if llm_healthy else "offline"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    محادثة عادية مع الـ LLM.
    بيحتفظ بـ history المحادثة في الـ session.
    """
    if not chat_manager:
        raise HTTPException(status_code=503, detail="الـ LLM مش جاهز")

    start_time = time.time()

    try:
        response = await chat_manager.chat(request.message)
        return ChatResponse(
            answer=response.content,
            model_used=response.model,
            tokens_used=response.total_tokens,
            response_time_ms=round((time.time() - start_time) * 1000, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGQueryRequest):
    """
    سؤال على الـ RAG - بيبحث في الـ knowledge base ويجيب إجابة.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="الـ RAG مش جاهز")

    try:
        return await rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            min_relevance=request.min_relevance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/add-text")
async def add_text_to_rag(request: AddTextRequest, background_tasks: BackgroundTasks):
    """
    إضافة نص للـ knowledge base.
    التحويل لـ embeddings بيحصل في الـ background.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="الـ RAG مش جاهز")

    # إضافة في الـ background عشان مانتظرش
    def add_task():
        count = rag_pipeline.add_text(request.text, request.source_name)
        print(f"✅ تم إضافة {count} chunk من: {request.source_name}")

    background_tasks.add_task(add_task)

    return {
        "status": "accepted",
        "message": f"جاري إضافة النص من: {request.source_name}"
    }


@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: AnalyzeRequest):
    """تحليل مشاعر نص."""
    if not structured_output:
        raise HTTPException(status_code=503, detail="الـ structured output مش جاهز")

    try:
        return await structured_output.analyze_sentiment(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/summarize", response_model=SummaryResponse)
async def summarize_text(request: AnalyzeRequest):
    """تلخيص نص."""
    if not structured_output:
        raise HTTPException(status_code=503, detail="الـ structured output مش جاهز")

    try:
        return await structured_output.summarize(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """إحصائيات الـ system الكاملة."""
    stats = {
        "llm": {
            "provider": settings.llm_provider,
            "model": llm.model if llm else None,
            "temperature": settings.llm_temperature,
        }
    }

    if rag_pipeline:
        stats["rag"] = rag_pipeline.get_stats()

    if chat_manager:
        stats["chat"] = chat_manager.get_stats()

    return stats
