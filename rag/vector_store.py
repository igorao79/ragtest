"""Обёртка над ChromaDB: добавление, поиск и удаление документов."""

import logging

import chromadb
from chromadb import Collection

logger = logging.getLogger(__name__)


class VectorStore:
    """Управляет коллекциями ChromaDB — по одной на пользователя Telegram."""

    def __init__(self, persist_dir: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB инициализирована: %s", persist_dir)

    def get_or_create_collection(self, user_id: int) -> Collection:
        """Получить или создать коллекцию для пользователя."""
        name = f"user_{user_id}"
        return self.client.get_or_create_collection(name=name)

    def add_documents(
        self, user_id: int, chunks: list[str], metadata: list[dict]
    ) -> None:
        """Добавить чанки в коллекцию пользователя.

        Args:
            user_id: ID пользователя Telegram.
            chunks: Список текстовых чанков.
            metadata: Метаданные каждого чанка (source, chunk_index).
        """
        collection = self.get_or_create_collection(user_id)
        ids = [f"{meta['source']}_{meta['chunk_index']}" for meta in metadata]
        # upsert для идемпотентности — повторная загрузка перезаписывает
        collection.upsert(documents=chunks, metadatas=metadata, ids=ids)
        logger.info(
            "Добавлено %d чанков в коллекцию user_%d", len(chunks), user_id
        )

    def query(
        self, user_id: int, question: str, top_k: int = 4
    ) -> list[dict]:
        """Найти ближайшие чанки по вопросу.

        Args:
            user_id: ID пользователя Telegram.
            question: Вопрос для поиска.
            top_k: Количество результатов.

        Returns:
            Список словарей с ключами text, source, distance.
        """
        collection = self.get_or_create_collection(user_id)
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

    def delete_collection(self, user_id: int) -> None:
        """Удалить коллекцию пользователя."""
        name = f"user_{user_id}"
        try:
            self.client.delete_collection(name)
            logger.info("Коллекция %s удалена", name)
        except ValueError:
            logger.warning("Коллекция %s не найдена для удаления", name)

    def get_doc_count(self, user_id: int) -> int:
        """Получить количество чанков в коллекции пользователя."""
        collection = self.get_or_create_collection(user_id)
        return collection.count()
