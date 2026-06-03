"""
core/ollama_llm.py
==================
تطبيق الـ LLM باستخدام Ollama - تشغيل النماذج محلياً.

إيه هو Ollama؟
- برنامج مجاني بيخليك تشغّل LLMs على جهازك
- مش محتاج API key أو اشتراك مدفوع
- يدعم Llama, Mistral, Gemma, وغيرهم كتير
- نزّله من: https://ollama.com

المميزات:
- مجاني تماماً
- خصوصية كاملة (مش بيبعت بياناتك لأي سيرفر)
- يشتغل بدون إنترنت
- دعم streaming
"""

import json
import time
import httpx
from typing import List, Optional, AsyncGenerator

from .base_llm import BaseLLM, Message, LLMResponse
from config.settings import settings


class OllamaLLM(BaseLLM):
    """
    تطبيق الـ LLM باستخدام Ollama API المحلية.
    Ollama بيشغّل server محلي على port 11434.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None
    ):
        """
        Parameters:
            model: اسم النموذج (مثلاً: llama3.2, mistral, gemma2)
            temperature: إبداعية الإجابات
            max_tokens: الحد الأقصى للتوكنز
            base_url: عنوان سيرفر Ollama (default: localhost)
        """
        super().__init__(
            model=model or settings.ollama_model,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens
        )

        self.base_url = base_url or settings.ollama_base_url
        self.api_url = f"{self.base_url}/api/chat"

    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        إرسال رسائل لـ Ollama والحصول على رد كامل.

        Ollama بيستخدم نفس format الـ OpenAI تقريباً،
        بس بيتفاعل مع سيرفر محلي بدل cloud.
        """
        start_time = time.time()

        # بناء الرسائل
        formatted_messages = self._build_messages(messages, system_prompt)

        # إعداد الـ request body
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,   # مش streaming في الميثود دي
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,  # Ollama بيستخدم num_predict بدل max_tokens
            }
        }

        # إرسال الطلب للـ Ollama server
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()

        response_time = (time.time() - start_time) * 1000

        # استخراج الرد من الـ response
        content = data.get("message", {}).get("content", "")

        # Ollama بيرد بعدد التوكنز في بعض الـ responses
        eval_count = data.get("eval_count", 0)       # completion tokens
        prompt_eval = data.get("prompt_eval_count", 0)  # prompt tokens

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            total_tokens=prompt_eval + eval_count,
            finish_reason=data.get("done_reason", "stop"),
            response_time_ms=response_time
        )

    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        إرسال رسائل والحصول على الرد بشكل streaming.

        Ollama بيبعت كل كلمة/token في سطر JSON منفصل،
        فبنقرأهم تدريجياً.
        """
        formatted_messages = self._build_messages(messages, system_prompt)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,    # streaming mode
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        # استخدام streaming HTTP request
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self.api_url, json=payload) as response:
                response.raise_for_status()

                # كل سطر هو JSON object بيحتوي على chunk من النص
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        # استخراج النص من الـ chunk
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content

                        # لو done=True معناه انتهى الرد
                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        # تجاهل الـ lines اللي مش JSON صحيح
                        continue

    async def health_check(self) -> bool:
        """
        التحقق من أن Ollama server شغال ومتاح.
        بيتحقق من الـ models endpoint المتاحة.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list:
        """
        قائمة بكل النماذج المثبتة على Ollama.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def pull_model(self, model_name: str) -> bool:
        """
        تحميل نموذج جديد لـ Ollama (مثل: ollama pull llama3.2).
        """
        try:
            payload = {"name": model_name, "stream": False}
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 دقائق timeout
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json=payload
                )
                return response.status_code == 200
        except Exception:
            return False
