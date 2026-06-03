"""
core/base_llm.py
================
الـ Abstract Base Class لكل الـ LLM implementations.

الفكرة من Design Pattern ده:
- بنعرف interface موحد لكل الـ LLMs
- سواء كان OpenAI أو Ollama أو غيرهم، نفس الميثودز
- الكود اللي بيستخدم الـ LLM مش محتاج يعرف هو أي نوع
  (Dependency Inversion Principle)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class Message:
    """
    رسالة في المحادثة.
    role: "system" | "user" | "assistant"
    content: نص الرسالة
    """
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """تحويل الرسالة لـ dict عشان نبعتها للـ API."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """
    الرد الكامل من الـ LLM مع metadata.
    """
    content: str                          # نص الرد
    model: str                            # اسم النموذج
    prompt_tokens: int = 0                # توكنز السؤال
    completion_tokens: int = 0            # توكنز الرد
    total_tokens: int = 0                 # إجمالي التوكنز
    finish_reason: str = "stop"           # سبب الإيقاف
    response_time_ms: float = 0.0         # وقت الاستجابة


class BaseLLM(ABC):
    """
    الـ Abstract Base Class لكل الـ LLMs.

    أي LLM جديد هيورث من الكلاس ده ويطبق الميثودز الـ abstract.
    ده بيضمن إن كل الـ LLMs عندهم نفس الـ interface.
    """

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        """
        Parameters:
            model: اسم النموذج
            temperature: إبداعية الإجابات (0=محدد, 1=إبداعي)
            max_tokens: الحد الأقصى للتوكنز في الرد
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        إرسال محادثة والحصول على رد.
        يجب تطبيق هذه الميثود في كل subclass.
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        إرسال محادثة والحصول على رد بشكل streaming (كلمة كلمة).
        يجب تطبيق هذه الميثود في كل subclass.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        التحقق من أن الـ LLM شغال وجاهز.
        """
        pass

    def _build_messages(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        Helper: بناء قائمة الرسائل مع إضافة الـ system prompt.
        الـ system prompt بيكون دايماً أول رسالة في القائمة.
        """
        result = []

        # إضافة الـ system prompt كأول رسالة
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        # إضافة باقي الرسائل
        result.extend([msg.to_dict() for msg in messages])

        return result
