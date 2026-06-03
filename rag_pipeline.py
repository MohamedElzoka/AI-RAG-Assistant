"""
rag/rag_pipeline.py
===================
الـ RAG Pipeline الكامل - بيجمع كل الأجزاء مع بعض.

إيه هو الـ RAG؟
RAG = Retrieval-Augmented Generation
= استرجاع + توليد مُعزَّز

الفكرة:
1. عندنا مجموعة مستندات (knowledge base)
2. المستخدم بيسأل سؤال
3. بنبحث في المستندات عن المعلومات الأقرب للسؤال
4. بنديها للـ LLM مع السؤال
5. الـ LLM بيولّد إجابة مبنية على المعلومات دي

ليه RAG؟
- الـ LLMs بيبقوا outdated (training data قديم)
- RAG بيخليهم يجاوبوا بناءً على معلوماتك الخاصة
- بيقلل الـ hallucination (اختراع معلومات)
- أرخص من fine-tuning النموذج
"""

from typing import List, Optional, AsyncGenerator

from core.base_llm import BaseLLM, Message
from llm.schemas import RAGResponse, DocumentSource
from config.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from .vector_store import VectorStore
from .document_loader import DocumentLoader, Document


class RAGPipeline:
    """
    الـ RAG Pipeline الكامل.

    المسؤوليات:
    1. إدارة إضافة المستندات للـ knowledge base
    2. البحث عن المعلومات ذات الصلة بالسؤال
    3. بناء الـ prompt المناسب مع المعلومات المسترجعة
    4. إرسال الـ prompt للـ LLM والحصول على إجابة
    """

    def __init__(
        self,
        llm: BaseLLM,
        vector_store: Optional[VectorStore] = None,
        document_loader: Optional[DocumentLoader] = None,
        system_prompt: str = RAG_SYSTEM_PROMPT
    ):
        """
        Parameters:
            llm: الـ LLM المستخدم للإجابة
            vector_store: مخزن الـ vectors (يُنشأ تلقائياً لو مش موجود)
            document_loader: محمل المستندات (يُنشأ تلقائياً لو مش موجود)
            system_prompt: الـ prompt الخاص بالـ RAG
        """
        self.llm = llm
        self.vector_store = vector_store or VectorStore()
        self.document_loader = document_loader or DocumentLoader()
        self.system_prompt = system_prompt

    # ==========================================
    # إضافة المستندات
    # ==========================================

    def add_file(self, file_path: str) -> int:
        """إضافة ملف واحد للـ knowledge base."""
        documents = self.document_loader.load_file(file_path)
        return self.vector_store.add_documents(documents)

    def add_directory(self, dir_path: str) -> int:
        """إضافة كل ملفات مجلد للـ knowledge base."""
        documents = self.document_loader.load_directory(dir_path)
        return self.vector_store.add_documents(documents)

    def add_text(self, text: str, source_name: str = "manual") -> int:
        """إضافة نص مباشرة للـ knowledge base."""
        documents = self.document_loader.load_text(text, source_name)
        return self.vector_store.add_documents(documents)

    # ==========================================
    # البحث والإجابة
    # ==========================================

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_relevance: float = 0.3  # الحد الأدنى للتشابه (0-1)
    ) -> RAGResponse:
        """
        السؤال الرئيسي للـ RAG Pipeline.

        الخطوات:
        1. البحث عن المعلومات ذات الصلة
        2. فلترة النتائج الضعيفة
        3. بناء الـ prompt مع المعلومات المسترجعة
        4. الحصول على إجابة من الـ LLM
        """
        # الخطوة 1: البحث في الـ vector store
        sources = self.vector_store.search(question, top_k=top_k)

        # الخطوة 2: فلترة النتائج الضعيفة
        relevant_sources = [
            s for s in sources
            if s.relevance_score >= min_relevance
        ]

        # الخطوة 3: بناء الـ context من المصادر
        if relevant_sources:
            context = self._build_context(relevant_sources)
        else:
            context = "لا توجد معلومات ذات صلة في قاعدة المعرفة."

        # الخطوة 4: بناء الـ prompt مع الـ context
        user_message = RAG_USER_TEMPLATE.format(
            context=context,
            question=question
        )

        # الخطوة 5: إرسال السؤال للـ LLM
        response = await self.llm.chat(
            messages=[Message(role="user", content=user_message)],
            system_prompt=self.system_prompt
        )

        return RAGResponse(
            answer=response.content,
            sources=relevant_sources,
            model_used=response.model,
            sources_count=len(relevant_sources)
        )

    async def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_relevance: float = 0.3
    ) -> AsyncGenerator[str, None]:
        """
        نفس query بس الرد بيجي streaming.
        """
        sources = self.vector_store.search(question, top_k=top_k)
        relevant_sources = [s for s in sources if s.relevance_score >= min_relevance]

        context = (
            self._build_context(relevant_sources)
            if relevant_sources
            else "لا توجد معلومات ذات صلة."
        )

        user_message = RAG_USER_TEMPLATE.format(
            context=context,
            question=question
        )

        async for chunk in self.llm.stream_chat(
            messages=[Message(role="user", content=user_message)],
            system_prompt=self.system_prompt
        ):
            yield chunk

    def _build_context(self, sources: List[DocumentSource]) -> str:
        """
        بناء الـ context من قائمة المصادر.
        بنرتبهم حسب التشابه وبنضيف معلومات المصدر.
        """
        context_parts = []

        for i, source in enumerate(sources, 1):
            # إضافة معلومات المصدر مع النص
            context_parts.append(
                f"[مصدر {i}: {source.source} | تشابه: {source.relevance_score:.0%}]\n"
                f"{source.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> dict:
        """إحصائيات الـ pipeline."""
        vs_stats = self.vector_store.get_stats()
        return {
            "knowledge_base": vs_stats,
            "llm_model": self.llm.model,
        }
