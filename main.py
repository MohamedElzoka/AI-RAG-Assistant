"""
main.py
=======
نقطة الدخول الرئيسية للمشروع.

طرق التشغيل:
    python main.py --mode demo     → Demo يعرض كل المميزات
    python main.py --mode cli      → محادثة تفاعلية
    python main.py --mode rag      → وضع RAG
    python main.py --mode api      → REST API
"""

import asyncio
import argparse
import sys
from pathlib import Path


def print_header():
    """طباعة header جميل للمشروع."""
    header = """
╔══════════════════════════════════════════════════╗
║          🤖 AI Project - Full LLM System         ║
║     LLM + RAG + Local LLMs + Pydantic Outputs    ║
╚══════════════════════════════════════════════════╝
"""
    print(header)


def create_sample_docs():
    """إنشاء مستندات تجريبية للاختبار."""
    sample_dir = Path("data/sample_docs")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ملف معلومات عن الـ AI
    ai_doc = sample_dir / "ai_basics.txt"
    if not ai_doc.exists():
        ai_doc.write_text("""
الذكاء الاصطناعي وتعلم الآلة

ما هو الذكاء الاصطناعي؟
الذكاء الاصطناعي (AI) هو فرع من علوم الحاسوب يهدف إلى محاكاة الذكاء البشري في الآلات.
يشمل الذكاء الاصطناعي عدة مجالات منها: تعلم الآلة، معالجة اللغات الطبيعية، والرؤية الحاسوبية.

ما هي النماذج اللغوية الكبيرة (LLMs)؟
النماذج اللغوية الكبيرة هي نماذج ذكاء اصطناعي تم تدريبها على كميات ضخمة من النصوص.
تستطيع هذه النماذج فهم اللغة الطبيعية وتوليدها بشكل طبيعي.
أمثلة: GPT-4, Claude, Llama, Gemma.

ما هو RAG؟
RAG (Retrieval-Augmented Generation) هو أسلوب يجمع بين استرجاع المعلومات وتوليد النصوص.
يساعد RAG في تحسين دقة الإجابات من خلال الاعتماد على مصادر موثوقة.
يقلل RAG من ظاهرة الهلوسة في النماذج اللغوية.

ما هو Ollama؟
Ollama هو برنامج يتيح تشغيل النماذج اللغوية محلياً على جهازك.
مجاني تماماً ويحافظ على خصوصيتك لأن البيانات لا تغادر جهازك.
يدعم نماذج مثل Llama 3, Mistral, Gemma, وغيرها.
""", encoding="utf-8")

    print(f"✅ تم إنشاء مستندات تجريبية في: {sample_dir}")
    return str(sample_dir)


async def main():
    """الدالة الرئيسية."""
    print_header()

    # الـ argument parser
    parser = argparse.ArgumentParser(
        description="AI Project - Full LLM System"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "cli", "rag", "api"],
        default="demo",
        help="وضع التشغيل: demo, cli, rag, api"
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default=None,
        help="مسار مجلد المستندات للـ RAG mode"
    )

    args = parser.parse_args()

    print(f"🚀 وضع التشغيل: {args.mode}\n")

    if args.mode == "demo":
        # Demo يعرض كل المميزات
        from cli.chat_cli import demo_mode
        await demo_mode()

    elif args.mode == "cli":
        # محادثة تفاعلية
        from cli.chat_cli import chat_mode
        await chat_mode()

    elif args.mode == "rag":
        # وضع RAG
        docs_path = args.docs_path
        if not docs_path:
            docs_path = create_sample_docs()

        from cli.chat_cli import rag_mode
        await rag_mode(knowledge_base_path=docs_path)

    elif args.mode == "api":
        # تشغيل الـ API
        try:
            import uvicorn
            print(f"🌐 تشغيل الـ API على: http://{settings.api_host}:{settings.api_port}")
            print(f"📖 API Docs: http://localhost:{settings.api_port}/docs\n")

            uvicorn.run(
                "api.app:app",
                host=settings.api_host,
                port=settings.api_port,
                reload=settings.api_debug
            )
        except ImportError:
            print("❌ uvicorn مش مثبتة. شغّل: pip install uvicorn")
            sys.exit(1)


if __name__ == "__main__":
    # Import هنا عشان نتجنب circular imports
    from config.settings import settings
    asyncio.run(main())
