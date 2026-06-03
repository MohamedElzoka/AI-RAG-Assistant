"""
tests/test_rag.py
=================
اختبارات الـ RAG components.
"""

import pytest
from pathlib import Path
import tempfile

from rag.document_loader import DocumentLoader
from llm.schemas import DocumentMetadata


# ==========================================
# Tests للـ DocumentLoader
# ==========================================

def test_load_text_basic():
    """اختبار تحميل نص بسيط."""
    loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
    docs = loader.load_text("هذا نص تجريبي قصير.", source_name="test")

    assert len(docs) >= 1
    assert docs[0].content == "هذا نص تجريبي قصير."
    assert docs[0].metadata.source == "test"


def test_load_text_chunking():
    """اختبار تقسيم النص لـ chunks."""
    # نص طويل
    long_text = "كلمة " * 200  # 1000 حرف تقريباً
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    docs = loader.load_text(long_text, source_name="long_test")

    # لازم يتقسم لأكتر من chunk
    assert len(docs) > 1


def test_chunk_metadata():
    """اختبار صحة الـ metadata."""
    loader = DocumentLoader(chunk_size=50, chunk_overlap=5)
    long_text = "نص تجريبي طويل " * 50
    docs = loader.load_text(long_text)

    # التحقق من الـ metadata
    for i, doc in enumerate(docs):
        assert doc.metadata.chunk_index == i
        assert doc.metadata.total_chunks == len(docs)


def test_load_txt_file():
    """اختبار تحميل ملف .txt."""
    loader = DocumentLoader()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("محتوى ملف النص التجريبي")
        tmp_path = f.name

    docs = loader.load_file(tmp_path)
    assert len(docs) >= 1
    assert "محتوى ملف النص التجريبي" in docs[0].content

    # حذف الملف المؤقت
    Path(tmp_path).unlink()


def test_load_nonexistent_file():
    """اختبار محاولة تحميل ملف غير موجود."""
    loader = DocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_file("ملف_غير_موجود.txt")


def test_load_unsupported_file():
    """اختبار محاولة تحميل نوع ملف غير مدعوم."""
    loader = DocumentLoader()

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        tmp_path = f.name

    with pytest.raises(ValueError, match="نوع الملف غير مدعوم"):
        loader.load_file(tmp_path)

    Path(tmp_path).unlink()


# ==========================================
# Tests للـ Pydantic Schemas
# ==========================================

def test_document_metadata_validation():
    """اختبار Pydantic validation للـ metadata."""
    metadata = DocumentMetadata(
        source="test.txt",
        doc_type="txt",
        chunk_index=0,
        total_chunks=5,
        char_count=100
    )

    assert metadata.source == "test.txt"
    assert metadata.chunk_index == 0


def test_document_metadata_defaults():
    """اختبار القيم الافتراضية للـ metadata."""
    metadata = DocumentMetadata(source="test.txt")

    assert metadata.doc_type == "text"
    assert metadata.chunk_index == 0
    assert metadata.total_chunks == 1
