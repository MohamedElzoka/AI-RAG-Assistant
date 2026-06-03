"""
core/chat_manager.py
====================
إدارة تاريخ المحادثات والـ context window.

المشكلة اللي بيحلها:
- الـ LLMs مش عندهم memory تلقائية
- كل request مستقل بذاته
- عشان الـ LLM "يتذكر" المحادثة، لازم نبعتله كل الـ history في كل request

التحديات:
- الـ context window محدود (مثلاً 128k token)
- الـ history الطويلة بيأثر على التكلفة والسرعة
- الحل: نحتفظ بعدد محدود من الرسائل الأخيرة
"""

import json
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from .base_llm import BaseLLM, Message, LLMResponse
from config.prompts import BASE_SYSTEM_PROMPT, SUMMARY_PROMPT


@dataclass
class ConversationTurn:
    """
    دورة محادثة واحدة: سؤال + رد.
    """
    user_message: str
    assistant_message: str
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    response_time_ms: float = 0.0


class ChatManager:
    """
    إدارة محادثة كاملة مع الـ LLM.

    المميزات:
    - حفظ تاريخ المحادثة
    - إدارة الـ context window تلقائياً
    - دعم streaming
    - تلخيص المحادثات الطويلة
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = BASE_SYSTEM_PROMPT,
        max_history_turns: int = 10   # الحد الأقصى لعدد الأدوار في الـ history
    ):
        """
        Parameters:
            llm: الـ LLM المستخدم في المحادثة
            system_prompt: الشخصية/الهوية الخاصة بالـ assistant
            max_history_turns: الحد الأقصى لعدد الأدوار المحفوظة
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns

        # تاريخ المحادثة الكامل
        self.history: List[ConversationTurn] = []

        # الرسائل الحالية (للـ API)
        self.messages: List[Message] = []

    async def chat(self, user_message: str) -> LLMResponse:
        """
        إرسال رسالة والحصول على رد مع الحفاظ على الـ history.

        الخطوات:
        1. إضافة رسالة المستخدم للـ messages
        2. إرسال كل الـ messages للـ LLM
        3. إضافة رد الـ LLM للـ messages
        4. حفظ الـ turn في الـ history
        """
        # إضافة رسالة المستخدم
        self.messages.append(Message(role="user", content=user_message))

        # إرسال الطلب مع كل الـ history
        response = await self.llm.chat(
            messages=self.messages,
            system_prompt=self.system_prompt
        )

        # إضافة رد الـ assistant للـ messages
        self.messages.append(
            Message(role="assistant", content=response.content)
        )

        # حفظ الـ turn في الـ history
        self.history.append(ConversationTurn(
            user_message=user_message,
            assistant_message=response.content,
            tokens_used=response.total_tokens,
            response_time_ms=response.response_time_ms
        ))

        # تنظيف الـ history لو تعدى الحد
        self._trim_history()

        return response

    async def stream_chat(self, user_message: str):
        """
        إرسال رسالة والحصول على الرد بشكل streaming.

        الاستخدام:
            full_response = ""
            async for chunk in manager.stream_chat("مرحبا"):
                print(chunk, end="")
                full_response += chunk
        """
        # إضافة رسالة المستخدم
        self.messages.append(Message(role="user", content=user_message))

        # جمع الرد الكامل أثناء الـ streaming
        full_response = ""

        async for chunk in self.llm.stream_chat(
            messages=self.messages,
            system_prompt=self.system_prompt
        ):
            full_response += chunk
            yield chunk  # نبعت كل chunk للـ caller

        # بعد ما الـ streaming خلص، نضيف الرد للـ history
        self.messages.append(
            Message(role="assistant", content=full_response)
        )

        self.history.append(ConversationTurn(
            user_message=user_message,
            assistant_message=full_response
        ))

        self._trim_history()

    def _trim_history(self):
        """
        تنظيف الـ history القديم لو تعدى الحد الأقصى.

        بنحتفظ بـ:
        1. الـ system prompt (ثابت)
        2. آخر N دورة من المحادثة
        """
        if len(self.history) > self.max_history_turns:
            # حذف أقدم الرسائل من الـ messages
            # بنحذف دورتين (user + assistant) في كل مرة
            excess = len(self.history) - self.max_history_turns
            self.messages = self.messages[excess * 2:]  # كل دورة = رسالتين
            self.history = self.history[excess:]

    def clear_history(self):
        """مسح كل تاريخ المحادثة والبدء من أول."""
        self.history = []
        self.messages = []

    def get_stats(self) -> Dict:
        """إحصائيات المحادثة الحالية."""
        total_tokens = sum(turn.tokens_used for turn in self.history)
        avg_response_time = (
            sum(turn.response_time_ms for turn in self.history) / len(self.history)
            if self.history else 0
        )

        return {
            "total_turns": len(self.history),
            "total_tokens_used": total_tokens,
            "average_response_time_ms": round(avg_response_time, 2),
            "model": self.llm.model,
        }

    def export_history(self) -> List[Dict]:
        """
        تصدير تاريخ المحادثة كـ list من dicts.
        مفيد للحفظ في قاعدة بيانات أو ملف.
        """
        return [
            {
                "user": turn.user_message,
                "assistant": turn.assistant_message,
                "timestamp": turn.timestamp,
                "tokens": turn.tokens_used,
            }
            for turn in self.history
        ]
