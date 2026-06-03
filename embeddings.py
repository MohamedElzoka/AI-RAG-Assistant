"""
rag/embeddings.py
=================
تحويل النصوص إلى Embeddings (Vectors).

إيه هي الـ Embeddings؟
- هي تحويل النص لقائمة من الأرقام (vector)
- النصوص المتشابهة في المعنى بيكون الـ vector بتاعها متقارب
- مثلاً: "القطة" و"الهرة" هيبقوا vectors متقاربين
- ده اللي بيخلي الـ semantic search يشتغل صح

إزاي بنستخدمها في الـ RAG؟
1. عند الإضافة: نحول النص لـ vector ونحفظه في الـ vector store
2. عند البحث: نحول السؤال لـ vector ونبحث عن أقرب vectors
"""

from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from config.settings import settings


class EmbeddingsManager:
    """
    إدارة الـ embeddings باستخدام sentence-transformers.

    المميزات:
    - مجاني ومحلي (مش محتاج API)
    - يدعم أكتر من 50 لغة
    - سريع وكفء
    - النموذج all-MiniLM-L6-v2 صغير وسريع وجيد
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Parameters:
            model_name: اسم نموذج الـ embeddings
                - all-MiniLM-L6-v2: صغير وسريع (84MB) - الأفضل للبداية
                - all-mpnet-base-v2: أدق بس أبطأ (420MB)
                - paraphrase-multilingual-MiniLM-L12-v2: متعدد اللغات
        """
        if not ST_AVAILABLE:
            raise ImportError(
                "مكتبة sentence-transformers مش مثبتة.\n"
                "شغّل: pip install sentence-transformers"
            )

        self.model_name = model_name or settings.embedding_model
        # تحميل النموذج (مرة واحدة بس)
        # أول مرة هيحمّل النموذج من الإنترنت، بعدها هيتحفظ محلياً
        print(f"📦 تحميل نموذج الـ embeddings: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ النموذج جاهز! أبعاد الـ embedding: {self.embedding_dim}")

    def embed_text(self, text: str) -> List[float]:
        """
        تحويل نص واحد لـ embedding vector.

        Returns:
            قائمة من الأرقام (float) تمثل معنى النص
        """
        # normalize_embeddings=True بيخلي المقارنة بـ cosine similarity أدق
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        تحويل قائمة من النصوص لـ embeddings دفعة واحدة.
        أسرع بكتير من تحويلهم واحد واحد لأن الـ model بيعالجهم batch.

        Parameters:
            texts: قائمة النصوص

        Returns:
            قائمة من الـ embeddings
        """
        if not texts:
            return []

        # batch_size=32 يعني بيعالج 32 نص في كل مرة
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,  # يوري progress bar لو أكتر من 10
            batch_size=32
        )
        return embeddings.tolist()

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        حساب التشابه بين vectorين.

        بنستخدم Cosine Similarity:
        - 1.0 = متطابق تماماً
        - 0.0 = مش مترابطين
        - -1.0 = معاكسين (نادر مع normalize)
        """
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)

        # Cosine similarity = dot product لأن الـ vectors normalized
        return float(np.dot(v1, v2))

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        إيجاد أقرب الـ embeddings لـ query معين.

        Returns:
            قائمة من (index, similarity_score) مرتبة تنازلياً
        """
        if not candidate_embeddings:
            return []

        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        # حساب التشابه مع كل الـ candidates دفعة واحدة (أسرع)
        similarities = np.dot(candidates, query)

        # ترتيب النتائج تنازلياً
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]
