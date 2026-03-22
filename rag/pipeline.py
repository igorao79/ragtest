"""RAG-пайплайн: retrieve -> build_prompt -> generate."""

import logging
from collections.abc import AsyncIterator

from bot.config import (
    CACHE_MAX_SIZE,
    CACHE_TTL,
    COMBINED_PROMPT_TEMPLATE,
    COMBINED_SYSTEM_PROMPT,
    MAX_CONTEXT_LENGTH,
    PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOP_K,
    VISION_SYSTEM_PROMPT,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_PROMPT_TEMPLATE,
    WEB_SEARCH_SYSTEM_PROMPT,
)
from rag.cache import ResponseCache
from rag.chunker import TextChunker
from rag.document_loader import DocumentLoader
from rag.llm_client import OllamaClient
from rag.vector_store import VectorStore
from rag.web_search import WebSearchClient

logger = logging.getLogger(__name__)

_NO_RESULTS_MESSAGE = (
    "Я не нашёл релевантной информации в загруженных документах. "
    "Попробуйте загрузить нужные файлы или переформулировать вопрос."
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
        self.cache = ResponseCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)

    def _build_context(self, results: list[dict], max_length: int | None = None) -> str:
        """Собрать контекст из результатов поиска."""
        limit = max_length or MAX_CONTEXT_LENGTH
        parts: list[str] = []
        total = 0
        for r in results:
            text = r["text"]
            if total + len(text) > limit:
                remaining = limit - total
                if remaining > 50:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    async def answer(self, user_id: int, question: str) -> str:
        """Найти релевантные чанки и сгенерировать ответ."""
        # Проверяем кеш
        cached = self.cache.get(user_id, question)
        if cached:
            return cached

        results = self.vector_store.query(user_id, question, TOP_K)
        if not results:
            return _NO_RESULTS_MESSAGE

        context = self._build_context(results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        logger.info("Генерация ответа для user_%d, контекст: %d символов", user_id, len(context))
        answer = await self.llm_client.generate(prompt, system=SYSTEM_PROMPT)
        answer = answer.strip()

        self.cache.put(user_id, question, answer)
        return answer

    async def answer_stream(self, user_id: int, question: str) -> AsyncIterator[str]:
        """Стриминг ответа по частям."""
        results = self.vector_store.query(user_id, question, TOP_K)
        if not results:
            yield _NO_RESULTS_MESSAGE
            return

        context = self._build_context(results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        logger.info("Стриминг ответа для user_%d", user_id)
        full_answer = []
        async for token in self.llm_client.generate_stream(prompt, system=SYSTEM_PROMPT):
            full_answer.append(token)
            yield token

        # Кешируем полный ответ
        self.cache.put(user_id, question, "".join(full_answer).strip())

    async def ingest(self, user_id: int, file_path: str, filename: str) -> int:
        """Загрузить документ в векторное хранилище."""
        text = self.loader.load(file_path)
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        metadata = [
            {"source": filename, "chunk_index": i} for i in range(len(chunks))
        ]
        self.vector_store.add_documents(user_id, chunks, metadata)
        # Инвалидируем кеш — данные изменились
        self.cache.invalidate_user(user_id)

        logger.info("Загружено %d чанков из %s для user_%d", len(chunks), filename, user_id)
        return len(chunks)

    async def ingest_url(self, user_id: int, url: str) -> int:
        """Загрузить веб-страницу в базу знаний."""
        text = self.loader.load_url(url)
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        # Используем домен как имя файла
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or url[:50]
        filename = f"url_{domain}"

        metadata = [
            {"source": filename, "chunk_index": i} for i in range(len(chunks))
        ]
        self.vector_store.add_documents(user_id, chunks, metadata)
        self.cache.invalidate_user(user_id)

        logger.info("Загружено %d чанков из URL %s для user_%d", len(chunks), url, user_id)
        return len(chunks)

    async def summarize(self, user_id: int) -> str:
        """Суммаризация всех документов пользователя."""
        text = self.vector_store.get_all_text(user_id, max_chars=MAX_CONTEXT_LENGTH)
        if not text:
            return "В вашей базе нет документов для суммаризации."

        prompt = SUMMARY_PROMPT_TEMPLATE.format(context=text)
        logger.info("Суммаризация для user_%d, текст: %d символов", user_id, len(text))
        answer = await self.llm_client.generate(prompt, system=SUMMARY_SYSTEM_PROMPT)
        return answer.strip()

    async def analyze_image(
        self, prompt: str, image_data: bytes, vision_model: str | None = None
    ) -> str:
        """Анализ изображения через vision-модель."""
        logger.info("Анализ изображения, %d байт", len(image_data))
        answer = await self.llm_client.generate_vision(
            prompt, image_data, model=vision_model
        )
        return answer.strip()

    async def web_answer(self, question: str) -> str:
        """Поиск в интернете и генерация ответа."""
        results = await self.web_search.search(question)
        if not results:
            return "Не удалось найти результаты в интернете. Попробуйте другой запрос."

        web_context = WebSearchClient.format_results(results)
        if len(web_context) > MAX_CONTEXT_LENGTH:
            web_context = web_context[:MAX_CONTEXT_LENGTH]

        prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
            web_context=web_context, question=question
        )

        logger.info("Веб-поиск + генерация, контекст: %d символов", len(web_context))
        answer = await self.llm_client.generate(prompt, system=WEB_SEARCH_SYSTEM_PROMPT)

        links = WebSearchClient.format_results_short(results)
        return f"{answer.strip()}\n\n📎 Источники:\n{links}"

    async def combined_answer(self, user_id: int, question: str) -> str:
        """Ответ на основе документов + веб-поиска."""
        doc_results = self.vector_store.query(user_id, question, TOP_K)
        doc_context = self._build_context(doc_results, MAX_CONTEXT_LENGTH // 2) if doc_results else ""

        # Улучшаем веб-запрос: если есть документы, извлекаем ключевые слова
        web_query = question
        if doc_results:
            # Добавляем ключевые слова из топ-чанка к запросу
            top_text = doc_results[0]["text"][:200]
            # Берём первые значимые слова из документа для контекста поиска
            doc_words = [w for w in top_text.split() if len(w) > 4][:5]
            if doc_words:
                web_query = f"{question} {' '.join(doc_words)}"

        web_results = await self.web_search.search(web_query)
        web_context = WebSearchClient.format_results(web_results)
        if len(web_context) > MAX_CONTEXT_LENGTH // 2:
            web_context = web_context[: MAX_CONTEXT_LENGTH // 2]

        if not doc_context and not web_context:
            return "Не удалось найти информацию ни в документах, ни в интернете."

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

        result = answer.strip()
        if web_results:
            links = WebSearchClient.format_results_short(web_results)
            result += f"\n\n📎 Источники из интернета:\n{links}"
        return result
