"""
rag/vector_store.py
===================
تخزين واسترجاع الـ embeddings باستخدام ChromaDB.

إيه هو الـ Vector Store؟
- قاعدة بيانات متخصصة في تخزين الـ vectors
- بتعمل semantic search (بحث بالمعنى مش بالكلمات بالضبط)
- مثلاً لو سألت عن "كيفية حل المشكلة" هيلاقيلك "طرق التعامل مع الأخطاء"

ليه ChromaDB؟
- مفتوح المصدر ومجاني
- سريع جداً
- يعمل محلياً بدون سيرفر خارجي
- أسهل بكتير من Pinecone أو Weaviate للمشاريع الصغيرة
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from .embeddings import EmbeddingsManager
from .document_loader import Document
from llm.schemas import DocumentMetadata, DocumentSource
from config.settings import settings


class VectorStore:
    """
    Vector Store محلي باستخدام ChromaDB.

    العمليات الأساسية:
    1. add_documents: إضافة مستندات مع embeddings
    2. search: البحث بالمعنى
    3. delete: حذف مستندات
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_path: Optional[str] = None,
        embeddings_manager: Optional[EmbeddingsManager] = None
    ):
        """
        Parameters:
            collection_name: اسم المجموعة (collection) في ChromaDB
            persist_path: مكان حفظ البيانات على الـ disk
            embeddings_manager: الـ manager المسؤول عن توليد الـ embeddings
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb مش مثبتة. شغّل: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_path = persist_path or settings.vector_db_path

        # إنشاء مجلد الحفظ لو مش موجود
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        # إنشاء ChromaDB client مع حفظ على الـ disk
        self.client = chromadb.PersistentClient(
            path=self.persist_path
        )

        # إنشاء أو فتح الـ collection
        # get_or_create يعني لو موجودة يفتحها، لو لأ ينشئها
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )

        # الـ embeddings manager
        self.embeddings = embeddings_manager or EmbeddingsManager()

        print(f"✅ Vector Store جاهز: {self.collection.count()} مستند محفوظ")

    def add_documents(self, documents: List[Document]) -> int:
        """
        إضافة مستندات للـ vector store.

        الخطوات:
        1. تحويل كل مستند لـ embedding
        2. حفظ الـ embedding + النص + الـ metadata في ChromaDB

        Returns:
            عدد المستندات المضافة
        """
        if not documents:
            return 0

        # استخراج البيانات من الـ documents
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata.model_dump() for doc in documents]

        # توليد IDs فريدة لكل document
        ids = [
            f"{meta['source']}_{meta['chunk_index']}"
            for meta in metadatas
        ]

        # توليد الـ embeddings دفعة واحدة (أسرع)
        print(f"⏳ جاري توليد embeddings لـ {len(texts)} مستند...")
        embeddings = self.embeddings.embed_texts(texts)

        # التحقق من وجود مستندات بنفس الـ IDs وحذفها (تحديث)
        existing_ids = set(self.collection.get()["ids"])
        new_ids = [id for id in ids if id not in existing_ids]

        if len(new_ids) < len(ids):
            duplicate_count = len(ids) - len(new_ids)
            print(f"ℹ️ تم تخطي {duplicate_count} مستند موجود مسبقاً")

        # إضافة المستندات الجديدة فقط
        new_indices = [i for i, id in enumerate(ids) if id in new_ids]

        if not new_indices:
            return 0

        self.collection.add(
            ids=[ids[i] for i in new_indices],
            embeddings=[embeddings[i] for i in new_indices],
            documents=[texts[i] for i in new_indices],
            metadatas=[metadatas[i] for i in new_indices]
        )

        added = len(new_indices)
        print(f"✅ تم إضافة {added} مستند للـ vector store")
        return added

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None
    ) -> List[DocumentSource]:
        """
        البحث بالمعنى في الـ vector store.

        Parameters:
            query: سؤال/نص البحث
            top_k: عدد النتائج المطلوبة
            filter_source: لو عايز تبحث في مصدر معين بس

        Returns:
            قائمة من DocumentSource مرتبة حسب التشابه
        """
        if self.collection.count() == 0:
            return []

        top_k = top_k or settings.top_k_results

        # تحويل السؤال لـ embedding
        query_embedding = self.embeddings.embed_text(query)

        # إعداد الـ filter لو محدد
        where_filter = None
        if filter_source:
            where_filter = {"source": {"$eq": filter_source}}

        # البحث في ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # تحويل النتائج لـ DocumentSource objects
        sources = []
        if results["documents"] and results["documents"][0]:
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # ChromaDB بيرجع distance (كلما قل كلما كان أقرب)
                # بنحوله لـ similarity score (0-1)
                # مع normalized embeddings: similarity = 1 - distance/2
                similarity = max(0, 1 - distance / 2)

                sources.append(DocumentSource(
                    content=text,
                    source=metadata.get("source", "unknown"),
                    relevance_score=round(similarity, 3),
                    chunk_index=metadata.get("chunk_index")
                ))

        return sources

    def delete_source(self, source_name: str) -> int:
        """
        حذف كل الـ chunks من مصدر معين.
        مفيد لتحديث مستند معين.
        """
        results = self.collection.get(
            where={"source": {"$eq": source_name}}
        )

        if not results["ids"]:
            return 0

        self.collection.delete(ids=results["ids"])
        deleted = len(results["ids"])
        print(f"🗑️ تم حذف {deleted} chunk من مصدر: {source_name}")
        return deleted

    def get_stats(self) -> Dict:
        """إحصائيات الـ vector store."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_path": self.persist_path,
            "embedding_model": self.embeddings.model_name,
            "embedding_dimensions": self.embeddings.embedding_dim,
        }

    def clear(self):
        """مسح كل البيانات من الـ collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        print("🧹 تم مسح الـ vector store")
