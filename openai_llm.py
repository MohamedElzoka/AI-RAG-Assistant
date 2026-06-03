"""
core/openai_llm.py
==================
تطبيق الـ LLM باستخدام OpenAI API.

المميزات:
- دعم GPT-4o, GPT-4o-mini, وغيرهم
- Streaming للردود الطويلة
- حساب التوكنز تلقائياً
- Retry logic عند الفشل
"""

import time
import asyncio
from typing import List, Optional, AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_llm import BaseLLM, Message, LLMResponse
from config.settings import settings


class OpenAILLM(BaseLLM):
    """
    تطبيق الـ LLM باستخدام OpenAI API.
    بيورث من BaseLLM ويطبق كل الـ abstract methods.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        إنشاء OpenAI LLM instance.

        Parameters:
            model: النموذج (default: من الـ settings)
            temperature: الإبداعية (default: من الـ settings)
            max_tokens: الحد الأقصى (default: من الـ settings)
            api_key: مفتاح الـ API (default: من الـ settings)
        """
        # استخدام قيم الـ settings كـ defaults
        super().__init__(
            model=model or settings.openai_model,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens
        )

        if not OPENAI_AVAILABLE:
            raise ImportError("مكتبة openai مش مثبتة. شغّل: pip install openai")

        # إنشاء الـ client مع الـ API key
        self.client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key
        )

    @retry(
        stop=stop_after_attempt(3),          # حاول 3 مرات
        wait=wait_exponential(min=1, max=10)  # انتظر بشكل تدريجي بين المحاولات
    )
    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        إرسال رسائل للـ OpenAI والحصول على رد كامل.

        الـ @retry decorator بيعيد المحاولة لو فيه خطأ مؤقت.
        """
        start_time = time.time()

        # بناء قائمة الرسائل مع الـ system prompt
        formatted_messages = self._build_messages(messages, system_prompt)

        # إرسال الطلب للـ API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # حساب وقت الاستجابة
        response_time = (time.time() - start_time) * 1000

        # استخراج الرد والمعلومات منه
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            response_time_ms=response_time
        )

    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        إرسال رسائل والحصول على الرد بشكل streaming.
        بيستخدم AsyncGenerator عشان يولد النص تدريجياً.

        الاستخدام:
            async for chunk in llm.stream_chat(messages):
                print(chunk, end="", flush=True)
        """
        formatted_messages = self._build_messages(messages, system_prompt)

        # stream=True هو السر في الـ streaming
        async with self.client.chat.completions.stream(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ) as stream:
            async for event in stream:
                # استخراج النص من كل chunk
                if (event.choices and
                    event.choices[0].delta and
                    event.choices[0].delta.content):
                    yield event.choices[0].delta.content

    async def health_check(self) -> bool:
        """
        التحقق من أن الـ OpenAI API شغال ومتاح.
        بيجرب رسالة بسيطة ويشوف لو الرد جه صح.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return bool(response.choices)
        except Exception:
            return False
