"""Тесты для мульти-коллекций VectorStore."""

import tempfile
import shutil

import pytest

from rag.vector_store import VectorStore


@pytest.fixture
def store():
    """Создать временное хранилище."""
    tmpdir = tempfile.mkdtemp()
    vs = VectorStore(tmpdir)
    yield vs
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestMultiCollections:
    def test_default_collection(self, store: VectorStore):
        store.add_documents(
            1,
            ["chunk1", "chunk2"],
            [{"source": "f.txt", "chunk_index": 0}, {"source": "f.txt", "chunk_index": 1}],
        )
        assert store.get_doc_count(1) == 2

    def test_named_collection(self, store: VectorStore):
        store.add_documents(
            1,
            ["chunk1"],
            [{"source": "f.txt", "chunk_index": 0}],
            collection_name="work",
        )
        assert store.get_doc_count(1, "work") == 1
        assert store.get_doc_count(1) == 0  # default is empty

    def test_query_named_collection(self, store: VectorStore):
        store.add_documents(
            1,
            ["Python is a programming language"],
            [{"source": "f.txt", "chunk_index": 0}],
            collection_name="study",
        )
        results = store.query(1, "programming", top_k=1, collection_name="study")
        assert len(results) == 1
        assert "Python" in results[0]["text"]

    def test_separate_collections_isolated(self, store: VectorStore):
        store.add_documents(
            1,
            ["work data"],
            [{"source": "w.txt", "chunk_index": 0}],
            collection_name="work",
        )
        store.add_documents(
            1,
            ["study data"],
            [{"source": "s.txt", "chunk_index": 0}],
            collection_name="study",
        )
        assert store.get_doc_count(1, "work") == 1
        assert store.get_doc_count(1, "study") == 1
        assert store.get_doc_count(1) == 0

    def test_list_user_collections(self, store: VectorStore):
        store.get_or_create_collection(1)
        store.get_or_create_collection(1, "work")
        store.get_or_create_collection(1, "study")

        cols = store.list_user_collections(1)
        assert "default" in cols
        assert "work" in cols
        assert "study" in cols

    def test_delete_named_collection(self, store: VectorStore):
        store.add_documents(
            1,
            ["data"],
            [{"source": "f.txt", "chunk_index": 0}],
            collection_name="temp",
        )
        assert store.get_doc_count(1, "temp") == 1
        store.delete_collection(1, "temp")
        assert store.get_doc_count(1, "temp") == 0

    def test_get_file_list_named(self, store: VectorStore):
        store.add_documents(
            1,
            ["data1", "data2"],
            [{"source": "a.pdf", "chunk_index": 0}, {"source": "b.pdf", "chunk_index": 0}],
            collection_name="docs",
        )
        files = store.get_file_list(1, "docs")
        names = [f["name"] for f in files]
        assert "a.pdf" in names
        assert "b.pdf" in names

    def test_delete_file_named(self, store: VectorStore):
        store.add_documents(
            1,
            ["data"],
            [{"source": "rm.txt", "chunk_index": 0}],
            collection_name="misc",
        )
        deleted = store.delete_file(1, "rm.txt", "misc")
        assert deleted == 1
        assert store.get_doc_count(1, "misc") == 0
