"""
config/settings.py
==================
إدارة كل إعدادات المشروع باستخدام Pydantic Settings.

الفكرة:
- كل الإعدادات بتتحمل من ملف .env تلقائياً
- لو مفيش قيمة، بيستخدم القيمة الافتراضية
- Pydantic بيتحقق من نوع كل قيمة تلقائياً
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum


class LLMProvider(str, Enum):
    """
    الـ Providers المتاحة للـ LLM.
    بنستخدم Enum عشان نضمن إن القيمة صح.
    """
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """
    كل إعدادات المشروع في مكان واحد.
    Pydantic بيحمل القيم من .env file تلقائياً.
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # اقرأ من ملف .env
        env_file_encoding="utf-8",
        case_sensitive=False,      # OPENAI_API_KEY = openai_api_key
        extra="ignore"             # تجاهل أي متغيرات زيادة
    )

    # ==========================================
    # إعدادات OpenAI
    # ==========================================
    openai_api_key: str = Field(
        default="",
        description="مفتاح API الخاص بـ OpenAI"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="النموذج المستخدم من OpenAI"
    )

    # ==========================================
    # إعدادات Ollama (Local LLM)
    # ==========================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="عنوان سيرفر Ollama المحلي"
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="النموذج المحلي المستخدم في Ollama"
    )

    # ==========================================
    # إعدادات الـ LLM العامة
    # ==========================================
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="هل نستخدم OpenAI ولا Ollama المحلي؟"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,       # أكبر من أو يساوي 0
        le=2.0,       # أصغر من أو يساوي 2
        description="إبداعية الإجابات: 0 = محدد، 1 = إبداعي"
    )
    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="الحد الأقصى للتوكنز في الرد"
    )

    # ==========================================
    # إعدادات الـ RAG
    # ==========================================
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="حجم الـ chunk بالأحرف عند تقسيم النصوص"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="التداخل بين الـ chunks لضمان الترابط"
    )
    top_k_results: int = Field(
        default=5,
        gt=0,
        description="كم نتيجة نسترجع من الـ vector store"
    )

    # ==========================================
    # إعدادات الـ Embeddings
    # ==========================================
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="نموذج الـ embeddings - بيشتغل محلياً ومجاناً"
    )
    vector_db_path: str = Field(
        default="./data/vectordb",
        description="المسار اللي هيتحفظ فيه الـ vector database"
    )

    # ==========================================
    # إعدادات الـ API
    # ==========================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, lt=65536)
    api_debug: bool = Field(default=False)

    # ==========================================
    # Logging
    # ==========================================
    log_level: str = Field(default="INFO")


# ==========================================
# Singleton - نعمل instance واحد بس للـ settings
# ==========================================
settings = Settings()
