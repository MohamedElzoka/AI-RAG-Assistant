"""
llm/schemas.py
==============
كل الـ Pydantic Schemas للمخرجات المنظمة.

الفكرة من Pydantic Outputs:
- بدل ما الـ LLM يرد بنص حر، بيرد بـ JSON محدد الشكل
- Pydantic بيتحقق إن الرد صح وكامل ومن النوع الصح
- ده بيخلي المخرجات يمكن التعامل معاها برمجياً بسهولة
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


# ==========================================
# Base Response - كل الردود بتورث منه
# ==========================================
class BaseResponse(BaseModel):
    """
    الـ Base class لكل الردود.
    بيضيف metadata مشتركة.
    """
    success: bool = Field(
        default=True,
        description="هل العملية نجحت؟"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="رسالة الخطأ لو فيه خطأ"
    )


# ==========================================
# Chat Response - رد المحادثة العادية
# ==========================================
class ChatResponse(BaseResponse):
    """
    رد المحادثة العادية مع metadata.
    """
    answer: str = Field(
        description="إجابة الـ LLM"
    )
    model_used: str = Field(
        description="اسم النموذج اللي أجاب"
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="عدد التوكنز المستخدمة"
    )
    response_time_ms: Optional[float] = Field(
        default=None,
        description="وقت الاستجابة بالميلي ثانية"
    )


# ==========================================
# RAG Response - رد الـ RAG مع المصادر
# ==========================================
class DocumentSource(BaseModel):
    """
    معلومات عن مصدر معلومة من الـ RAG.
    """
    content: str = Field(description="محتوى الـ chunk المسترجع")
    source: str = Field(description="اسم المصدر/الملف")
    relevance_score: float = Field(
        description="درجة التشابه مع السؤال (0-1)",
        ge=0.0,
        le=1.0
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="رقم الـ chunk في الملف الأصلي"
    )


class RAGResponse(BaseResponse):
    """
    رد الـ RAG مع المصادر المستخدمة في الإجابة.
    """
    answer: str = Field(description="الإجابة المولدة")
    sources: List[DocumentSource] = Field(
        default_factory=list,
        description="المصادر المستخدمة في الإجابة"
    )
    model_used: str = Field(description="النموذج المستخدم")
    sources_count: int = Field(
        description="عدد المصادر المسترجعة"
    )

    @field_validator("sources_count", mode="before")
    @classmethod
    def calculate_sources_count(cls, v, info):
        """
        بيحسب عدد المصادر تلقائياً من الـ sources list.
        """
        if "sources" in info.data:
            return len(info.data["sources"])
        return v


# ==========================================
# Sentiment Analysis - تحليل المشاعر
# ==========================================
class SentimentLabel(str, Enum):
    """أنواع المشاعر الممكنة."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentResponse(BaseResponse):
    """
    نتيجة تحليل المشاعر لنص ما.
    مثال على Structured Output للمهام المتخصصة.
    """
    text: str = Field(description="النص المحلل")
    sentiment: SentimentLabel = Field(description="نوع المشاعر")
    confidence: float = Field(
        description="نسبة الثقة في التحليل (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="سبب التصنيف")


# ==========================================
# Text Summary - تلخيص النصوص
# ==========================================
class SummaryResponse(BaseResponse):
    """
    نتيجة تلخيص نص ما.
    """
    original_length: int = Field(description="طول النص الأصلي بالكلمات")
    summary: str = Field(description="الملخص")
    summary_length: int = Field(description="طول الملخص بالكلمات")
    key_points: List[str] = Field(
        description="أهم النقاط في النص",
        min_length=1
    )
    compression_ratio: float = Field(
        description="نسبة الضغط (كلما قلت كلما كان الملخص أقصر)",
        ge=0.0,
        le=1.0
    )

    @field_validator("compression_ratio", mode="before")
    @classmethod
    def calculate_ratio(cls, v, info):
        """بيحسب نسبة الضغط تلقائياً."""
        data = info.data
        if "original_length" in data and "summary_length" in data:
            original = data["original_length"]
            if original > 0:
                return round(data["summary_length"] / original, 2)
        return v


# ==========================================
# Information Extraction - استخراج معلومات
# ==========================================
class ExtractedEntity(BaseModel):
    """
    كيان مستخرج من النص.
    مثلاً: أسماء أشخاص، أماكن، تواريخ، إلخ.
    """
    text: str = Field(description="النص الأصلي للكيان")
    entity_type: str = Field(
        description="نوع الكيان: PERSON, LOCATION, DATE, ORG, etc."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResponse(BaseResponse):
    """
    نتيجة استخراج المعلومات من نص.
    """
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="الكيانات المستخرجة"
    )
    summary: str = Field(description="ملخص المعلومات المستخرجة")


# ==========================================
# Document Metadata - معلومات المستند
# ==========================================
class DocumentMetadata(BaseModel):
    """
    Metadata لكل مستند في الـ vector store.
    """
    source: str = Field(description="مسار/اسم الملف المصدر")
    doc_type: str = Field(
        default="text",
        description="نوع الملف: pdf, txt, md, docx"
    )
    chunk_index: int = Field(
        default=0,
        description="رقم هذا الـ chunk في الملف"
    )
    total_chunks: int = Field(
        default=1,
        description="إجمالي الـ chunks في الملف"
    )
    char_count: int = Field(
        default=0,
        description="عدد الأحرف في هذا الـ chunk"
    )
