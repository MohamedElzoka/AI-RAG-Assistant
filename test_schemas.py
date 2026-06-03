"""
tests/test_schemas.py
=====================
اختبارات الـ Pydantic Schemas والـ Structured Outputs.
"""

import pytest
from pydantic import ValidationError

from llm.schemas import (
    ChatResponse, RAGResponse, DocumentSource,
    SentimentResponse, SentimentLabel,
    SummaryResponse, DocumentMetadata
)


# ==========================================
# Tests للـ Pydantic Validation
# ==========================================

def test_document_source_valid():
    """اختبار إنشاء DocumentSource صحيح."""
    source = DocumentSource(
        content="محتوى النص",
        source="test.txt",
        relevance_score=0.85
    )
    assert source.content == "محتوى النص"
    assert source.relevance_score == 0.85


def test_document_source_invalid_score():
    """اختبار إن الـ relevance_score خارج النطاق يفشل."""
    with pytest.raises(ValidationError):
        DocumentSource(
            content="test",
            source="test.txt",
            relevance_score=1.5  # أكبر من 1
        )


def test_sentiment_response():
    """اختبار SentimentResponse."""
    result = SentimentResponse(
        text="نص سعيد",
        sentiment=SentimentLabel.POSITIVE,
        confidence=0.9,
        reasoning="الكلمات الإيجابية"
    )
    assert result.sentiment == SentimentLabel.POSITIVE
    assert result.confidence == 0.9
    assert result.success is True  # default value


def test_sentiment_invalid_confidence():
    """اختبار إن confidence خارج 0-1 يفشل."""
    with pytest.raises(ValidationError):
        SentimentResponse(
            text="test",
            sentiment=SentimentLabel.NEUTRAL,
            confidence=1.5,  # خطأ
            reasoning="test"
        )


def test_rag_response_sources_count():
    """اختبار إن sources_count يُحسب تلقائياً."""
    sources = [
        DocumentSource(content="chunk 1", source="doc.txt", relevance_score=0.9),
        DocumentSource(content="chunk 2", source="doc.txt", relevance_score=0.8),
    ]
    response = RAGResponse(
        answer="إجابة",
        sources=sources,
        model_used="test-model",
        sources_count=0  # هيتحسب تلقائياً
    )
    assert response.sources_count == 2


def test_chat_response_defaults():
    """اختبار القيم الافتراضية للـ ChatResponse."""
    response = ChatResponse(
        answer="مرحباً",
        model_used="gpt-4o-mini"
    )
    assert response.success is True
    assert response.tokens_used is None
    assert response.error_message is None


def test_summary_response():
    """اختبار SummaryResponse."""
    result = SummaryResponse(
        original_length=100,
        summary="ملخص قصير",
        summary_length=10,
        key_points=["نقطة 1", "نقطة 2"],
        compression_ratio=0.1
    )
    assert len(result.key_points) == 2
    assert result.compression_ratio == 0.1


def test_base_response_error():
    """اختبار BaseResponse مع رسالة خطأ."""
    from llm.schemas import BaseResponse

    error_response = BaseResponse(
        success=False,
        error_message="حدث خطأ ما"
    )
    assert error_response.success is False
    assert error_response.error_message == "حدث خطأ ما"
