# 🤖 AI Project - Full LLM + RAG + Local LLM System

## 📚 المشروع ده بيغطي المفاهيم دي كلها:

| المفهوم | الوصف |
|---|---|
| **LLM** | Large Language Models - النماذج اللغوية الكبيرة |
| **RAG** | Retrieval-Augmented Generation - استرجاع المعلومات وتوليد الإجابات |
| **Local LLMs** | تشغيل النماذج محلياً عن طريق Ollama |
| **Pydantic Outputs** | مخرجات منظمة ومحققة بـ Pydantic |
| **Embeddings** | تحويل النصوص لـ vectors رقمية |
| **Vector Store** | تخزين واسترجاع الـ embeddings |
| **Prompt Engineering** | هندسة الـ prompts |
| **Chat History** | إدارة تاريخ المحادثات |
| **Streaming** | استقبال الردود بشكل streaming |

## 🗂️ هيكل المشروع

```
ai_project/
├── config/
│   ├── settings.py          # كل الإعدادات والـ configs
│   └── prompts.py           # الـ prompts المنظمة
├── core/
│   ├── base_llm.py          # Abstract base class للـ LLMs
│   ├── openai_llm.py        # OpenAI implementation
│   ├── ollama_llm.py        # Local LLM عن طريق Ollama
│   └── chat_manager.py      # إدارة المحادثات والـ history
├── rag/
│   ├── embeddings.py        # توليد الـ embeddings
│   ├── vector_store.py      # تخزين واسترجاع الـ vectors
│   ├── document_loader.py   # تحميل وتقسيم المستندات
│   └── rag_pipeline.py      # Pipeline كامل للـ RAG
├── llm/
│   ├── schemas.py           # Pydantic schemas للمخرجات
│   └── structured_output.py # توليد مخرجات منظمة
├── api/
│   └── app.py               # FastAPI REST API
├── cli/
│   └── chat_cli.py          # واجهة سطر الأوامر
├── data/
│   └── sample_docs/         # مستندات تجريبية
├── tests/
│   ├── test_llm.py
│   ├── test_rag.py
│   └── test_schemas.py
├── main.py                  # نقطة الدخول الرئيسية
├── requirements.txt         # المكتبات المطلوبة
└── .env.example             # مثال على المتغيرات البيئية
```

## 🚀 طريقة التشغيل

### 1. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. إعداد المتغيرات البيئية
```bash
cp .env.example .env
# عدّل الـ .env وحط الـ API keys
```

### 3. تشغيل النموذج المحلي (اختياري)
```bash
# تثبيت Ollama من ollama.com
ollama pull llama3.2
```

### 4. تشغيل المشروع
```bash
# CLI mode
python main.py --mode cli

# API mode
python main.py --mode api

# Demo mode
python main.py --mode demo
```
