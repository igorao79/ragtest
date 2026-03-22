"""Тесты для VectorStore."""

import tempfile

import pytest

from rag.vector_store import VectorStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield VectorStore(tmpdir)


def test_add_and_query(store):
    chunks = ["Python is a programming language", "JavaScript is used for web"]
    metadata = [{"source": "test.txt", "chunk_index": 0}, {"source": "test.txt", "chunk_index": 1}]
    store.add_documents(1, chunks, metadata)

    results = store.query(1, "programming language", top_k=2)
    assert len(results) > 0
    assert results[0]["source"] == "test.txt"


def test_empty_query(store):
    results = store.query(1, "anything")
    assert results == []


def test_doc_count(store):
    assert store.get_doc_count(1) == 0
    chunks = ["text one", "text two", "text three"]
    metadata = [{"source": "f.txt", "chunk_index": i} for i in range(3)]
    store.add_documents(1, chunks, metadata)
    assert store.get_doc_count(1) == 3


def test_delete_collection(store):
    chunks = ["data"]
    metadata = [{"source": "f.txt", "chunk_index": 0}]
    store.add_documents(1, chunks, metadata)
    assert store.get_doc_count(1) == 1
    store.delete_collection(1)
    assert store.get_doc_count(1) == 0


def test_file_list(store):
    chunks = ["a", "b", "c"]
    metadata = [
        {"source": "file1.pdf", "chunk_index": 0},
        {"source": "file1.pdf", "chunk_index": 1},
        {"source": "file2.txt", "chunk_index": 0},
    ]
    store.add_documents(1, chunks, metadata)
    files = store.get_file_list(1)
    assert len(files) == 2
    names = {f["name"] for f in files}
    assert "file1.pdf" in names
    assert "file2.txt" in names


def test_file_list_empty(store):
    files = store.get_file_list(1)
    assert files == []


def test_delete_file(store):
    chunks = ["a", "b", "c"]
    metadata = [
        {"source": "keep.txt", "chunk_index": 0},
        {"source": "delete.txt", "chunk_index": 0},
        {"source": "delete.txt", "chunk_index": 1},
    ]
    store.add_documents(1, chunks, metadata)
    assert store.get_doc_count(1) == 3

    deleted = store.delete_file(1, "delete.txt")
    assert deleted == 2
    assert store.get_doc_count(1) == 1


def test_delete_file_not_found(store):
    deleted = store.delete_file(1, "nonexistent.txt")
    assert deleted == 0


def test_get_all_text(store):
    chunks = ["hello world", "foo bar", "test data"]
    metadata = [{"source": "f.txt", "chunk_index": i} for i in range(3)]
    store.add_documents(1, chunks, metadata)
    text = store.get_all_text(1)
    assert "hello world" in text


def test_get_all_text_empty(store):
    text = store.get_all_text(1)
    assert text == ""


def test_upsert_idempotent(store):
    chunks = ["version 1"]
    metadata = [{"source": "f.txt", "chunk_index": 0}]
    store.add_documents(1, chunks, metadata)
    assert store.get_doc_count(1) == 1

    # Повторная загрузка с тем же ID — не дублирует
    chunks = ["version 2"]
    store.add_documents(1, chunks, metadata)
    assert store.get_doc_count(1) == 1


def test_user_isolation(store):
    store.add_documents(1, ["user1 data"], [{"source": "f.txt", "chunk_index": 0}])
    store.add_documents(2, ["user2 data"], [{"source": "f.txt", "chunk_index": 0}])

    r1 = store.query(1, "data")
    r2 = store.query(2, "data")
    assert r1[0]["text"] == "user1 data"
    assert r2[0]["text"] == "user2 data"
