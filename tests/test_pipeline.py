"""Тесты для RAGPipeline."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.pipeline import RAGPipeline


@pytest.fixture
def mock_pipeline():
    """Pipeline с мок-зависимостями."""
    vector_store = MagicMock()
    llm_client = AsyncMock()
    pipeline = RAGPipeline(vector_store, llm_client)
    return pipeline


@pytest.mark.asyncio
async def test_answer_no_results(mock_pipeline):
    mock_pipeline.vector_store.query.return_value = []
    answer = await mock_pipeline.answer(1, "test question")
    assert "не нашёл" in answer


@pytest.mark.asyncio
async def test_answer_with_results(mock_pipeline):
    mock_pipeline.vector_store.query.return_value = [
        {"text": "Python is great", "source": "doc.txt", "distance": 0.1}
    ]
    mock_pipeline.llm_client.generate.return_value = "Python is indeed great!"
    answer = await mock_pipeline.answer(1, "what is python?")
    assert answer == "Python is indeed great!"


@pytest.mark.asyncio
async def test_answer_uses_cache(mock_pipeline):
    mock_pipeline.vector_store.query.return_value = [
        {"text": "data", "source": "f.txt", "distance": 0.1}
    ]
    mock_pipeline.llm_client.generate.return_value = "cached answer"

    # Первый вызов
    await mock_pipeline.answer(1, "question")
    # Второй вызов — должен вернуть из кеша
    result = await mock_pipeline.answer(1, "question")
    assert result == "cached answer"
    # LLM вызвана только 1 раз
    assert mock_pipeline.llm_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_ingest(mock_pipeline):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write("This is a test document with some content for ingestion testing purposes. " * 10)
        f.flush()
        count = await mock_pipeline.ingest(1, f.name, "test.txt")
    assert count > 0
    mock_pipeline.vector_store.add_documents.assert_called_once()
    Path(f.name).unlink()


@pytest.mark.asyncio
async def test_summarize_empty(mock_pipeline):
    mock_pipeline.vector_store.get_all_text.return_value = ""
    result = await mock_pipeline.summarize(1)
    assert "нет документов" in result


@pytest.mark.asyncio
async def test_summarize(mock_pipeline):
    mock_pipeline.vector_store.get_all_text.return_value = "Some document text"
    mock_pipeline.llm_client.generate.return_value = "Summary of text"
    result = await mock_pipeline.summarize(1)
    assert result == "Summary of text"


@pytest.mark.asyncio
async def test_web_answer_no_results(mock_pipeline):
    mock_pipeline.web_search.search = AsyncMock(return_value=[])
    result = await mock_pipeline.web_answer("test")
    assert "Не удалось" in result


@pytest.mark.asyncio
async def test_analyze_image(mock_pipeline):
    mock_pipeline.llm_client.generate_vision.return_value = "Image shows a cat"
    result = await mock_pipeline.analyze_image("describe", b"fake-image")
    assert result == "Image shows a cat"
