"""RAG-пайплайн: retrieve -> build_prompt -> generate."""

import logging

from bot.config import (
    MAX_CONTEXT_LENGTH,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    TOP_K,
)
from rag.chunker import TextChunker
from rag.document_loader import DocumentLoader
from rag.llm_client import OllamaClient
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_NO_RESULTS_MESSAGE = (
    "Я не нашёл релевантной информации в загруженных документах. "
    "Попробуйте загрузить нужные файлы командой или переформулировать вопрос."
)


class RAGPipeline:
    """Оркестрирует загрузку документов и генерацию ответов."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: OllamaClient,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)

    async def answer(self, user_id: int, question: str) -> str:
        """Найти релевантные чанки и сгенерировать ответ.

        Args:
            user_id: ID пользователя Telegram.
            question: Вопрос пользователя.

        Returns:
            Сгенерированный ответ.
        """
        # 1. Retrieve
        results = self.vector_store.query(user_id, question, TOP_K)

        if not results:
            return _NO_RESULTS_MESSAGE

        # 2. Build context
        context_parts: list[str] = []
        total_length = 0
        for r in results:
            text = r["text"]
            if total_length + len(text) > MAX_CONTEXT_LENGTH:
                remaining = MAX_CONTEXT_LENGTH - total_length
                if remaining > 50:
                    context_parts.append(text[:remaining])
                break
            context_parts.append(text)
            total_length += len(text)

        context = "\n\n".join(context_parts)

        # 3. Build prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 4. Generate
        logger.info("Генерация ответа для user_%d, контекст: %d символов", user_id, len(context))
        answer = await self.llm_client.generate(prompt, system=SYSTEM_PROMPT)
        return answer.strip()

    async def ingest(self, user_id: int, file_path: str, filename: str) -> int:
        """Загрузить документ в векторное хранилище.

        Args:
            user_id: ID пользователя Telegram.
            file_path: Путь к скачанному файлу.
            filename: Имя оригинального файла.

        Returns:
            Количество добавленных чанков.
        """
        # 1. Load
        text = self.loader.load(file_path)

        # 2. Chunk
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        # 3. Store
        metadata = [
            {"source": filename, "chunk_index": i} for i in range(len(chunks))
        ]
        self.vector_store.add_documents(user_id, chunks, metadata)

        logger.info(
            "Загружено %d чанков из %s для user_%d", len(chunks), filename, user_id
        )
        return len(chunks)
