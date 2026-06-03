"""
llm/structured_output.py
========================
توليد مخرجات منظمة (Structured Outputs) من الـ LLM.

المشكلة:
- الـ LLMs بطبعهم بيردوا بنص حر
- النص الحر صعب التعامل معه برمجياً
- مثلاً لو طلبنا تحليل مشاعر، عايزين نرجع sentiment + confidence كـ fields

الحل:
1. نطلب من الـ LLM يرد بـ JSON محدد الـ schema
2. Pydantic يتحقق من الـ JSON ده ويحوله لـ object
3. لو الـ JSON مش صح، بنحاول مرة تانية

مثال على الاستخدام:
    result = await structured_output.analyze_sentiment("النص هنا")
    print(result.sentiment)     # "positive"
    print(result.confidence)    # 0.95
"""

import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

from core.base_llm import BaseLLM, Message
from config.prompts import STRUCTURED_OUTPUT_SYSTEM_PROMPT
from .schemas import (
    SentimentResponse, SentimentLabel,
    SummaryResponse,
    ExtractionResponse
)

# Generic type للـ Pydantic models
T = TypeVar("T", bound=BaseModel)


class StructuredOutputGenerator:
    """
    توليد مخرجات منظمة من الـ LLM باستخدام Pydantic.

    الطريقة:
    1. بنبني prompt بيطلب من الـ LLM يرد بـ JSON بشكل محدد
    2. بنحلل الرد ونتحقق منه بـ Pydantic
    3. لو فيه خطأ، بنحاول مرة تانية (retry)
    """

    def __init__(self, llm: BaseLLM, max_retries: int = 2):
        """
        Parameters:
            llm: الـ LLM المستخدم
            max_retries: عدد مرات إعادة المحاولة لو الـ JSON غلط
        """
        self.llm = llm
        self.max_retries = max_retries

    async def generate(
        self,
        prompt: str,
        schema: Type[T],
        system_context: str = ""
    ) -> T:
        """
        توليد مخرج منظم بناءً على Pydantic schema.

        Parameters:
            prompt: السؤال/الطلب
            schema: الـ Pydantic model المطلوب
            system_context: سياق إضافي للـ system prompt

        Returns:
            Instance من الـ schema بالقيم المُولَّدة
        """
        # بناء الـ prompt مع schema info
        schema_prompt = self._build_schema_prompt(prompt, schema)
        system = f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n{system_context}"

        last_error = None

        # محاولة الـ parsing بعدد محدد من المرات
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.llm.chat(
                    messages=[Message(role="user", content=schema_prompt)],
                    system_prompt=system
                )

                # استخراج JSON من الرد
                json_data = self._extract_json(response.content)

                # التحقق من البيانات بـ Pydantic
                return schema(**json_data)

            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < self.max_retries:
                    # إضافة رسالة خطأ للـ retry
                    schema_prompt += f"\n\nملاحظة: الرد السابق كان خاطئاً ({str(e)}). حاول مرة أخرى بـ JSON صحيح."

        raise ValueError(f"فشل توليد المخرج بعد {self.max_retries + 1} محاولة: {last_error}")

    def _build_schema_prompt(self, prompt: str, schema: Type[T]) -> str:
        """
        بناء الـ prompt مع شرح الـ JSON schema المطلوب.
        بنستخدم Pydantic's model_json_schema لتوليد الـ schema تلقائياً.
        """
        schema_info = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)

        return f"""{prompt}

رد بـ JSON فقط بدون أي نص إضافي، يتبع الـ schema التالي:
{schema_info}

تأكد إن الـ JSON:
1. صحيح تماماً (valid JSON)
2. يحتوي على كل الحقول المطلوبة
3. القيم من النوع الصح
4. بدون أي نص قبله أو بعده"""

    def _extract_json(self, text: str) -> dict:
        """
        استخراج JSON من النص.
        بعض الـ LLMs بيحطوا الـ JSON داخل markdown code blocks.
        """
        # تنظيف markdown code blocks
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # محاولة parse مباشرة
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # محاولة إيجاد أي JSON object في النص
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        raise ValueError(f"مفيش JSON صالح في الرد: {text[:200]}")

    # ==========================================
    # Convenience Methods - مهام جاهزة
    # ==========================================

    async def analyze_sentiment(self, text: str) -> SentimentResponse:
        """تحليل مشاعر نص."""
        prompt = f"""حلّل مشاعر النص التالي:
"{text}"

حدد:
1. هل المشاعر إيجابية أو سلبية أو محايدة
2. درجة الثقة في التحليل (0 إلى 1)
3. سبب التصنيف"""

        result = await self.generate(prompt, SentimentResponse)
        result.text = text
        return result

    async def summarize(self, text: str) -> SummaryResponse:
        """تلخيص نص."""
        word_count = len(text.split())
        prompt = f"""لخّص النص التالي ({word_count} كلمة):
"{text}"

قدّم:
1. ملخصاً مختصراً
2. أهم 3-5 نقاط"""

        result = await self.generate(prompt, SummaryResponse)
        result.original_length = word_count
        if not result.summary_length:
            result.summary_length = len(result.summary.split())
        return result

    async def extract_info(self, text: str) -> ExtractionResponse:
        """استخراج المعلومات والكيانات من نص."""
        prompt = f"""استخرج الكيانات المهمة (أشخاص، أماكن، تواريخ، منظمات) من:
"{text}"

وقدم ملخصاً للمعلومات المستخرجة."""

        return await self.generate(prompt, ExtractionResponse)
