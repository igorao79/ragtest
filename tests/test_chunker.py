"""Тесты для TextChunker."""

from rag.chunker import TextChunker


def test_empty_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_short_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    result = chunker.chunk("Hello, this is a short text for testing.")
    assert len(result) == 1
    assert result[0] == "Hello, this is a short text for testing."


def test_chunking_with_overlap():
    text = "A" * 200
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


def test_filters_short_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    text = "Short. " + "A" * 150
    chunks = chunker.chunk(text)
    for chunk in chunks:
        assert len(chunk) >= 20


def test_sentence_boundary():
    text = (
        "First sentence here. Second sentence here. Third sentence here. "
        "Fourth sentence here. Fifth sentence here. Sixth sentence here. "
        "Seventh sentence here. Eighth sentence here. Ninth sentence. "
        "Tenth sentence here. Eleventh sentence. Twelfth sentence here."
    )
    chunker = TextChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Чанки должны быть стрипнуты
    for chunk in chunks:
        assert chunk == chunk.strip()


def test_chunk_size_respected():
    text = "Word " * 500
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(text)
    for chunk in chunks:
        # Чанки могут быть чуть длиннее из-за поиска границ предложений
        assert len(chunk) <= 150
