"""
tests/test_llm.py
=================
اختبارات الـ LLM components.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.base_llm import Message, LLMResponse
from core.chat_manager import ChatManager


# ==========================================
# Mock LLM للاختبارات
# ==========================================

class MockLLM:
    """
    LLM وهمي للاختبارات.
    بيقلد الـ LLM الحقيقي بدون ما يحتاج API key.
    """
    model = "mock-model"
    temperature = 0.7
    max_tokens = 100

    async def chat(self, messages, system_prompt=None):
        return LLMResponse(
            content="هذا رد تجريبي من الـ Mock LLM",
            model=self.model,
            total_tokens=50
        )

    async def stream_chat(self, messages, system_prompt=None):
        words = ["هذا", " رد", " تجريبي", " streaming"]
        for word in words:
            yield word

    async def health_check(self):
        return True


# ==========================================
# Tests
# ==========================================

@pytest.mark.asyncio
async def test_chat_manager_basic():
    """اختبار المحادثة الأساسية."""
    llm = MockLLM()
    manager = ChatManager(llm=llm)

    response = await manager.chat("مرحباً")

    assert response.content == "هذا رد تجريبي من الـ Mock LLM"
    assert len(manager.history) == 1
    assert manager.history[0].user_message == "مرحباً"


@pytest.mark.asyncio
async def test_chat_manager_history():
    """اختبار حفظ تاريخ المحادثة."""
    llm = MockLLM()
    manager = ChatManager(llm=llm, max_history_turns=3)

    # إرسال 5 رسائل
    for i in range(5):
        await manager.chat(f"رسالة {i}")

    # التحقق من إن الـ history محدود بـ max_history_turns
    assert len(manager.history) == 3


@pytest.mark.asyncio
async def test_chat_manager_clear():
    """اختبار مسح تاريخ المحادثة."""
    llm = MockLLM()
    manager = ChatManager(llm=llm)

    await manager.chat("رسالة 1")
    manager.clear_history()

    assert len(manager.history) == 0
    assert len(manager.messages) == 0


@pytest.mark.asyncio
async def test_chat_manager_streaming():
    """اختبار الـ streaming."""
    llm = MockLLM()
    manager = ChatManager(llm=llm)

    chunks = []
    async for chunk in manager.stream_chat("مرحباً"):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0


def test_message_to_dict():
    """اختبار تحويل Message لـ dict."""
    msg = Message(role="user", content="مرحباً")
    d = msg.to_dict()

    assert d["role"] == "user"
    assert d["content"] == "مرحباً"


def test_chat_manager_stats():
    """اختبار إحصائيات المحادثة."""
    llm = MockLLM()
    manager = ChatManager(llm=llm)

    stats = manager.get_stats()

    assert "total_turns" in stats
    assert "model" in stats
    assert stats["model"] == "mock-model"
