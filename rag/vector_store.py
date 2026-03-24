"""Обёртка над ChromaDB: добавление, поиск и удаление документов.

Поддерживает мульти-коллекции — каждый пользователь может иметь
несколько "папок" знаний (работа, учёба и т.д.).

Использует nomic-embed-text через Ollama для качественных эмбеддингов.
"""

import logging

import chromadb
from chromadb import Collection

logger = logging.getLogger(__name__)


def _create_embedding_function(
    ollama_url: str = "http://localhost:11434",
    model_name: str = "nomic-embed-text",
):
    """Создать embedding function через Ollama.

    Если Ollama недоступна — fallback на дефолтные эмбеддинги ChromaDB.
    """
    try:
        from chromadb.utils.embedding_functions.ollama_embedding_function import (
            OllamaEmbeddingFunction,
        )
        ef = OllamaEmbeddingFunction(
            url=ollama_url,
            model_name=model_name,
        )
        # Проверяем что Ollama отвечает
        ef(["test"])
        logger.info("Эмбеддинги: %s через Ollama", model_name)
        return ef
    except Exception as e:
        logger.warning(
            "Ollama эмбеддинги недоступны (%s), используем дефолтные ChromaDB: %s",
            model_name, e,
        )
        return None


class VectorStore:
    """Управляет коллекциями ChromaDB — по одной+ на пользователя Telegram."""

    def __init__(
        self, persist_dir: str,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = _create_embedding_function(ollama_url, embedding_model)
        logger.info("ChromaDB инициализирована: %s", persist_dir)

    def _collection_name(self, user_id: int, collection_name: str | None = None) -> str:
        """Сформировать имя коллекции ChromaDB."""
        if collection_name:
            return f"user_{user_id}_{collection_name}"
        return f"user_{user_id}"

    def get_or_create_collection(
        self, user_id: int, collection_name: str | None = None
    ) -> Collection:
        """Получить или создать коллекцию для пользователя."""
        name = self._collection_name(user_id, collection_name)
        kwargs = {"name": name}
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn
        return self.client.get_or_create_collection(**kwargs)

    def add_documents(
        self, user_id: int, chunks: list[str], metadata: list[dict],
        collection_name: str | None = None,
    ) -> None:
        """Добавить чанки в коллекцию пользователя."""
        collection = self.get_or_create_collection(user_id, collection_name)
        ids = [f"{meta['source']}_{meta['chunk_index']}" for meta in metadata]
        collection.upsert(documents=chunks, metadatas=metadata, ids=ids)
        logger.info(
            "Добавлено %d чанков в коллекцию %s",
            len(chunks), collection.name,
        )

    def query(
        self, user_id: int, question: str, top_k: int = 4,
        collection_name: str | None = None,
    ) -> list[dict]:
        """Найти ближайшие чанки по вопросу."""
        collection = self.get_or_create_collection(user_id, collection_name)
        if collection.count() == 0:
            return []

        results = collection.query(query_texts=[question], n_results=top_k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def delete_collection(
        self, user_id: int, collection_name: str | None = None
    ) -> None:
        """Удалить коллекцию пользователя."""
        name = self._collection_name(user_id, collection_name)
        try:
            self.client.delete_collection(name)
            logger.info("Коллекция %s удалена", name)
        except ValueError:
            logger.warning("Коллекция %s не найдена для удаления", name)

    def get_doc_count(
        self, user_id: int, collection_name: str | None = None
    ) -> int:
        """Получить количество чанков в коллекции пользователя."""
        collection = self.get_or_create_collection(user_id, collection_name)
        return collection.count()

    def get_file_list(
        self, user_id: int, collection_name: str | None = None
    ) -> list[dict]:
        """Получить список загруженных файлов с количеством чанков."""
        collection = self.get_or_create_collection(user_id, collection_name)
        count = collection.count()
        if count == 0:
            return []

        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        file_counts: dict[str, int] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            file_counts[source] = file_counts.get(source, 0) + 1

        return [
            {"name": name, "chunks": cnt}
            for name, cnt in sorted(file_counts.items())
        ]

    def delete_file(
        self, user_id: int, filename: str, collection_name: str | None = None
    ) -> int:
        """Удалить конкретный файл из коллекции."""
        collection = self.get_or_create_collection(user_id, collection_name)
        result = collection.get(
            where={"source": filename},
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            return 0

        collection.delete(ids=ids)
        logger.info("Удалено %d чанков файла %s из %s", len(ids), filename, collection.name)
        return len(ids)

    def get_all_text(
        self, user_id: int, max_chars: int = 5000,
        collection_name: str | None = None,
    ) -> str:
        """Получить весь текст из коллекции для суммаризации."""
        collection = self.get_or_create_collection(user_id, collection_name)
        count = collection.count()
        if count == 0:
            return ""

        result = collection.get(include=["documents"])
        documents = result.get("documents", [])

        text_parts: list[str] = []
        total = 0
        for doc in documents:
            if total + len(doc) > max_chars:
                remaining = max_chars - total
                if remaining > 50:
                    text_parts.append(doc[:remaining])
                break
            text_parts.append(doc)
            total += len(doc)

        return "\n\n".join(text_parts)

    def list_user_collections(self, user_id: int) -> list[str]:
        """Получить список коллекций пользователя.

        Returns:
            Список имён коллекций (без префикса user_ID_).
        """
        prefix = f"user_{user_id}"
        all_collections = self.client.list_collections()
        user_collections: list[str] = []
        for col in all_collections:
            name = col.name if hasattr(col, "name") else str(col)
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix == "":
                    user_collections.append("default")
                elif suffix.startswith("_"):
                    user_collections.append(suffix[1:])
        return sorted(user_collections)
