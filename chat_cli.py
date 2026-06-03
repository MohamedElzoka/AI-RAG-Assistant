"""
cli/chat_cli.py
===============
واجهة سطر الأوامر (CLI) للتفاعل مع الـ AI.

المميزات:
- محادثة تفاعلية مع الـ LLM
- وضع RAG للبحث في المستندات
- تحليل المشاعر والتلخيص
- واجهة ملونة وجميلة باستخدام rich
"""

import asyncio
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from config.settings import settings, LLMProvider
from core.chat_manager import ChatManager
from core.openai_llm import OpenAILLM
from core.ollama_llm import OllamaLLM
from rag.rag_pipeline import RAGPipeline
from llm.structured_output import StructuredOutputGenerator


# إنشاء console لـ rich
console = Console() if RICH_AVAILABLE else None


def print_styled(text: str, style: str = ""):
    """طباعة نص بـ style معين."""
    if RICH_AVAILABLE and console:
        console.print(text, style=style)
    else:
        print(text)


def print_panel(content: str, title: str = "", style: str = "blue"):
    """طباعة نص داخل box."""
    if RICH_AVAILABLE and console:
        console.print(Panel(content, title=title, border_style=style))
    else:
        print(f"\n{'='*50}")
        if title:
            print(f"  {title}")
            print(f"{'='*50}")
        print(content)
        print(f"{'='*50}\n")


async def get_llm():
    """إنشاء الـ LLM المناسب بناءً على الـ settings."""
    if settings.llm_provider == LLMProvider.OPENAI:
        llm = OpenAILLM()
        print_styled("🤖 استخدام OpenAI GPT", "green")
    else:
        llm = OllamaLLM()
        print_styled(f"🖥️ استخدام Ollama (محلي): {settings.ollama_model}", "green")

    # التحقق من الاتصال
    is_healthy = await llm.health_check()
    if not is_healthy:
        print_styled("⚠️ تحذير: الـ LLM مش متاح. تأكد من إعداداتك.", "yellow")

    return llm


async def chat_mode():
    """
    وضع المحادثة التفاعلية.
    """
    print_panel(
        "محادثة تفاعلية مع الـ AI\n"
        "اكتب رسالتك ثم اضغط Enter\n"
        "اكتب 'خروج' أو 'quit' للخروج\n"
        "اكتب 'مسح' لمسح تاريخ المحادثة",
        title="💬 وضع المحادثة",
        style="blue"
    )

    llm = await get_llm()
    manager = ChatManager(llm=llm)

    while True:
        try:
            # طلب الإدخال من المستخدم
            if RICH_AVAILABLE:
                user_input = Prompt.ask("\n[bold cyan]أنت[/bold cyan]")
            else:
                user_input = input("\nأنت: ")

            # التحقق من الأوامر الخاصة
            if user_input.lower() in ["خروج", "quit", "exit"]:
                print_styled("\n👋 وداعاً!", "yellow")
                break

            if user_input in ["مسح", "clear"]:
                manager.clear_history()
                print_styled("✅ تم مسح تاريخ المحادثة", "green")
                continue

            if not user_input.strip():
                continue

            # الحصول على الرد بشكل streaming
            print_styled("\n🤖 المساعد:", "bold green")
            full_response = ""

            async for chunk in manager.stream_chat(user_input):
                print(chunk, end="", flush=True)
                full_response += chunk

            print()  # سطر جديد بعد الرد

            # عرض معلومات الجلسة
            stats = manager.get_stats()
            print_styled(
                f"\n[dim]📊 الجلسة: {stats['total_turns']} دورة | "
                f"التوكنز: {stats['total_tokens_used']}[/dim]",
                "dim"
            )

        except KeyboardInterrupt:
            print_styled("\n\n👋 وداعاً!", "yellow")
            break
        except Exception as e:
            print_styled(f"\n❌ خطأ: {e}", "red")


async def rag_mode(knowledge_base_path: Optional[str] = None):
    """
    وضع RAG - البحث في المستندات.
    """
    print_panel(
        "البحث والإجابة من المستندات\n"
        "اكتب سؤالك وسيتم البحث في قاعدة المعرفة",
        title="📚 وضع RAG",
        style="green"
    )

    llm = await get_llm()
    pipeline = RAGPipeline(llm=llm)

    # تحميل المستندات لو محدد مجلد
    if knowledge_base_path:
        print_styled(f"📂 جاري تحميل المستندات من: {knowledge_base_path}", "yellow")
        count = pipeline.add_directory(knowledge_base_path)
        print_styled(f"✅ تم تحميل {count} chunk", "green")
    else:
        # إضافة نص تجريبي
        sample_text = """
        الذكاء الاصطناعي (AI) هو محاكاة الذكاء البشري في الآلات.
        يشمل التعلم الآلي، معالجة اللغات الطبيعية، والرؤية الحاسوبية.
        GPT و Claude هي نماذج لغوية كبيرة تستخدم في الإجابة على الأسئلة.
        RAG يساعد في تحسين دقة الإجابات من خلال استرجاع معلومات من قاعدة بيانات.
        """
        pipeline.add_text(sample_text, "sample_knowledge")
        print_styled("✅ تم إضافة معلومات تجريبية", "green")

    while True:
        try:
            if RICH_AVAILABLE:
                question = Prompt.ask("\n[bold cyan]سؤالك[/bold cyan]")
            else:
                question = input("\nسؤالك: ")

            if question.lower() in ["خروج", "quit", "exit"]:
                break

            if not question.strip():
                continue

            # البحث والإجابة
            print_styled("\n🔍 جاري البحث...", "yellow")
            result = await pipeline.query(question)

            # عرض الإجابة
            print_panel(result.answer, title="📝 الإجابة", style="green")

            # عرض المصادر
            if result.sources:
                print_styled("\n📌 المصادر المستخدمة:", "bold blue")
                for i, source in enumerate(result.sources, 1):
                    print_styled(
                        f"  {i}. [{source.source}] تشابه: {source.relevance_score:.0%}",
                        "dim"
                    )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print_styled(f"\n❌ خطأ: {e}", "red")


async def demo_mode():
    """
    Demo يعرض كل مميزات المشروع.
    """
    print_panel(
        "🎯 Demo شامل لكل مميزات المشروع",
        title="🤖 AI Project Demo",
        style="magenta"
    )

    llm = await get_llm()

    # 1. محادثة عادية
    print_styled("\n1️⃣ محادثة عادية مع الـ LLM:", "bold yellow")
    from core.base_llm import Message
    response = await llm.chat(
        messages=[Message(role="user", content="مرحباً! عرّف نفسك في جملة واحدة.")],
        system_prompt="أنت مساعد ذكي ومختصر."
    )
    print_styled(f"🤖 الرد: {response.content}", "green")

    # 2. Structured Output
    print_styled("\n2️⃣ Structured Output - تحليل المشاعر:", "bold yellow")
    so = StructuredOutputGenerator(llm=llm)
    sentiment = await so.analyze_sentiment("أنا سعيد جداً بهذا المشروع الرائع!")
    print_styled(f"✅ المشاعر: {sentiment.sentiment.value} (ثقة: {sentiment.confidence:.0%})", "green")
    print_styled(f"   السبب: {sentiment.reasoning}", "dim")

    # 3. RAG
    print_styled("\n3️⃣ RAG - البحث في المستندات:", "bold yellow")
    pipeline = RAGPipeline(llm=llm)
    pipeline.add_text(
        "Python هي لغة برمجة عالية المستوى تُستخدم في الذكاء الاصطناعي وتحليل البيانات.",
        "python_info"
    )
    rag_result = await pipeline.query("ما هي Python؟")
    print_styled(f"✅ الإجابة: {rag_result.answer[:200]}...", "green")

    print_styled("\n✅ Demo انتهى! كل الأجزاء شغّالة.", "bold green")
