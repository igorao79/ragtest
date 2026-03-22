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
        """Добавить чанки в коллекцию пользователя."""
        collection = self.get_or_create_collection(user_id)
        ids = [f"{meta['source']}_{meta['chunk_index']}" for meta in metadata]
        collection.upsert(documents=chunks, metadatas=metadata, ids=ids)
        logger.info(
            "Добавлено %d чанков в коллекцию user_%d", len(chunks), user_id
        )

    def query(
        self, user_id: int, question: str, top_k: int = 4
    ) -> list[dict]:
        """Найти ближайшие чанки по вопросу."""
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

    def get_file_list(self, user_id: int) -> list[dict]:
        """Получить список загруженных файлов с количеством чанков.

        Returns:
            Список словарей {name, chunks}.
        """
        collection = self.get_or_create_collection(user_id)
        count = collection.count()
        if count == 0:
            return []

        # Получаем все метаданные
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

    def delete_file(self, user_id: int, filename: str) -> int:
        """Удалить конкретный файл из коллекции.

        Args:
            user_id: ID пользователя.
            filename: Имя файла для удаления.

        Returns:
            Количество удалённых чанков.
        """
        collection = self.get_or_create_collection(user_id)
        # Находим все ID с этим source
        result = collection.get(
            where={"source": filename},
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            return 0

        collection.delete(ids=ids)
        logger.info("Удалено %d чанков файла %s для user_%d", len(ids), filename, user_id)
        return len(ids)

    def get_all_text(self, user_id: int, max_chars: int = 5000) -> str:
        """Получить весь текст из коллекции для суммаризации."""
        collection = self.get_or_create_collection(user_id)
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
