"""RAG-пайплайн: retrieve -> build_prompt -> generate."""

import logging

from bot.config import (
    COMBINED_PROMPT_TEMPLATE,
    COMBINED_SYSTEM_PROMPT,
    MAX_CONTEXT_LENGTH,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    TOP_K,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_PROMPT_TEMPLATE,
    WEB_SEARCH_SYSTEM_PROMPT,
)
from rag.chunker import TextChunker
from rag.document_loader import DocumentLoader
from rag.llm_client import OllamaClient
from rag.vector_store import VectorStore
from rag.web_search import WebSearchClient

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
        self.web_search = WebSearchClient(max_results=WEB_SEARCH_MAX_RESULTS)

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

    async def web_answer(self, question: str) -> str:
        """Поиск в интернете и генерация ответа.

        Args:
            question: Вопрос пользователя.

        Returns:
            Сгенерированный ответ на основе веб-поиска.
        """
        results = await self.web_search.search(question)
        if not results:
            return "Не удалось найти результаты в интернете. Попробуйте другой запрос."

        web_context = WebSearchClient.format_results(results)
        # Обрезаем контекст
        if len(web_context) > MAX_CONTEXT_LENGTH:
            web_context = web_context[:MAX_CONTEXT_LENGTH]

        prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
            web_context=web_context, question=question
        )

        logger.info("Веб-поиск + генерация, контекст: %d символов", len(web_context))
        answer = await self.llm_client.generate(prompt, system=WEB_SEARCH_SYSTEM_PROMPT)

        # Добавляем ссылки в конец
        links = WebSearchClient.format_results_short(results)
        return f"{answer.strip()}\n\n📎 Источники:\n{links}"

    async def combined_answer(self, user_id: int, question: str) -> str:
        """Ответ на основе документов + веб-поиска.

        Args:
            user_id: ID пользователя Telegram.
            question: Вопрос пользователя.

        Returns:
            Сгенерированный ответ из обоих источников.
        """
        # 1. RAG из документов
        doc_results = self.vector_store.query(user_id, question, TOP_K)
        doc_context = ""
        if doc_results:
            parts: list[str] = []
            total = 0
            for r in doc_results:
                text = r["text"]
                if total + len(text) > MAX_CONTEXT_LENGTH // 2:
                    break
                parts.append(text)
                total += len(text)
            doc_context = "\n\n".join(parts)

        # 2. Веб-поиск
        web_results = await self.web_search.search(question)
        web_context = WebSearchClient.format_results(web_results)
        if len(web_context) > MAX_CONTEXT_LENGTH // 2:
            web_context = web_context[: MAX_CONTEXT_LENGTH // 2]

        if not doc_context and not web_context:
            return "Не удалось найти информацию ни в документах, ни в интернете."

        # 3. Генерация
        if doc_context and web_context:
            prompt = COMBINED_PROMPT_TEMPLATE.format(
                doc_context=doc_context, web_context=web_context, question=question
            )
            system = COMBINED_SYSTEM_PROMPT
        elif web_context:
            prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
                web_context=web_context, question=question
            )
            system = WEB_SEARCH_SYSTEM_PROMPT
        else:
            prompt = PROMPT_TEMPLATE.format(context=doc_context, question=question)
            system = SYSTEM_PROMPT

        logger.info(
            "Комбинированный ответ для user_%d: doc=%d, web=%d символов",
            user_id, len(doc_context), len(web_context),
        )
        answer = await self.llm_client.generate(prompt, system=system)

        # Добавляем ссылки
        result = answer.strip()
        if web_results:
            links = WebSearchClient.format_results_short(web_results)
            result += f"\n\n📎 Источники из интернета:\n{links}"
        return result
