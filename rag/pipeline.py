"""RAG-пайплайн: retrieve -> rerank -> build_prompt -> generate."""

import logging
from collections.abc import AsyncIterator

from bot.config import (
    CACHE_MAX_SIZE,
    CACHE_TTL,
    COMBINED_PROMPT_TEMPLATE,
    COMBINED_SYSTEM_PROMPT,
    CONVERSATION_MAX_MESSAGES,
    CONVERSATION_TTL,
    MAX_CONTEXT_LENGTH,
    PROMPT_TEMPLATE,
    RERANK_ENABLED,
    RERANK_FETCH_K,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOP_K,
    VISION_SYSTEM_PROMPT,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_PROMPT_TEMPLATE,
    WEB_SEARCH_SYSTEM_PROMPT,
    WHISPER_MODEL,
)
from rag.cache import ResponseCache
from rag.chunker import TextChunker
from rag.conversation import ConversationMemory
from rag.document_loader import DocumentLoader
from rag.llm_client import OllamaClient
from rag.reranker import LLMReranker
from rag.vector_store import VectorStore
from rag.web_search import WebSearchClient
from rag.whisper_client import WhisperClient

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
        self.conversation = ConversationMemory(
            max_messages=CONVERSATION_MAX_MESSAGES, ttl=CONVERSATION_TTL
        )
        self.reranker = LLMReranker(llm_client) if RERANK_ENABLED else None
        self.whisper = WhisperClient(model_size=WHISPER_MODEL)

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

    async def _retrieve_and_rerank(
        self, user_id: int, question: str, collection_name: str | None = None
    ) -> list[dict]:
        """Retrieve + optional rerank."""
        fetch_k = RERANK_FETCH_K if self.reranker else TOP_K
        results = self.vector_store.query(
            user_id, question, fetch_k, collection_name=collection_name
        )
        if not results:
            return []

        if self.reranker and len(results) > TOP_K:
            results = await self.reranker.rerank(question, results, top_k=TOP_K)

        return results

    def _get_history_block(self, user_id: int) -> str:
        """Получить блок истории для промпта."""
        history = self.conversation.get_context_string(user_id)
        if history:
            return f"История диалога:\n{history}\n\n"
        return ""

    async def answer(
        self, user_id: int, question: str, collection_name: str | None = None
    ) -> str:
        """Найти релевантные чанки и сгенерировать ответ."""
        cached = self.cache.get(user_id, question)
        if cached:
            return cached

        # Добавляем вопрос в историю
        self.conversation.add_user_message(user_id, question)

        results = await self._retrieve_and_rerank(user_id, question, collection_name)
        if not results:
            return _NO_RESULTS_MESSAGE

        context = self._build_context(results)
        history = self._get_history_block(user_id)
        prompt = PROMPT_TEMPLATE.format(
            context=context, question=question, history=history
        )

        logger.info("Генерация ответа для user_%d, контекст: %d символов", user_id, len(context))
        answer = await self.llm_client.generate(prompt, system=SYSTEM_PROMPT)
        answer = answer.strip()

        # Сохраняем ответ в историю и кеш
        self.conversation.add_assistant_message(user_id, answer)
        self.cache.put(user_id, question, answer)
        return answer

    async def answer_stream(
        self, user_id: int, question: str, collection_name: str | None = None
    ) -> AsyncIterator[str]:
        """Стриминг ответа по частям."""
        self.conversation.add_user_message(user_id, question)

        results = await self._retrieve_and_rerank(user_id, question, collection_name)
        if not results:
            yield _NO_RESULTS_MESSAGE
            return

        context = self._build_context(results)
        history = self._get_history_block(user_id)
        prompt = PROMPT_TEMPLATE.format(
            context=context, question=question, history=history
        )

        logger.info("Стриминг ответа для user_%d", user_id)
        full_answer = []
        async for token in self.llm_client.generate_stream(prompt, system=SYSTEM_PROMPT):
            full_answer.append(token)
            yield token

        final = "".join(full_answer).strip()
        self.conversation.add_assistant_message(user_id, final)
        self.cache.put(user_id, question, final)

    async def ingest(
        self, user_id: int, file_path: str, filename: str,
        collection_name: str | None = None,
    ) -> int:
        """Загрузить документ в векторное хранилище."""
        text = self.loader.load(file_path)
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        metadata = [
            {"source": filename, "chunk_index": i} for i in range(len(chunks))
        ]
        self.vector_store.add_documents(
            user_id, chunks, metadata, collection_name=collection_name
        )
        self.cache.invalidate_user(user_id)

        logger.info("Загружено %d чанков из %s для user_%d", len(chunks), filename, user_id)
        return len(chunks)

    async def ingest_url(
        self, user_id: int, url: str, collection_name: str | None = None
    ) -> int:
        """Загрузить веб-страницу в базу знаний."""
        text = self.loader.load_url(url)
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        from urllib.parse import urlparse
        domain = urlparse(url).netloc or url[:50]
        filename = f"url_{domain}"

        metadata = [
            {"source": filename, "chunk_index": i} for i in range(len(chunks))
        ]
        self.vector_store.add_documents(
            user_id, chunks, metadata, collection_name=collection_name
        )
        self.cache.invalidate_user(user_id)

        logger.info("Загружено %d чанков из URL %s для user_%d", len(chunks), url, user_id)
        return len(chunks)

    async def summarize(
        self, user_id: int, collection_name: str | None = None
    ) -> str:
        """Суммаризация всех документов пользователя."""
        text = self.vector_store.get_all_text(
            user_id, max_chars=MAX_CONTEXT_LENGTH, collection_name=collection_name
        )
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

    async def transcribe_voice(self, audio_path: str) -> str:
        """Транскрибировать голосовое сообщение."""
        return await self.whisper.transcribe(audio_path)

    async def _make_search_query(self, question: str) -> str:
        """Попросить LLM сформулировать поисковый запрос."""
        try:
            result = await self.llm_client.generate(
                prompt=question,
                system=(
                    "Переформулируй вопрос пользователя в короткий поисковый запрос "
                    "для поисковой системы (3-6 слов). Убери разговорные слова. "
                    "Оставь только ключевые слова и имена. "
                    "Ответь ТОЛЬКО поисковым запросом, без пояснений.\n"
                    "Примеры:\n"
                    "Вопрос: а кто правил после николая второго?\n"
                    "Запрос: правитель России после Николая II\n"
                    "Вопрос: что еще интересного можешь сказать о Николае втором?\n"
                    "Запрос: Николай II интересные факты\n"
                    "Вопрос: какая столица Франции?\n"
                    "Запрос: столица Франции"
                ),
            )
            query = result.strip().strip('"').strip("'").split("\n")[0].strip()
            if len(query) < 3:
                return question
            logger.info("LLM поисковый запрос: '%s' -> '%s'", question, query)
            return query
        except Exception as e:
            logger.warning("Не удалось сгенерировать поисковый запрос: %s", e)
            return question

    async def web_answer(self, question: str) -> str:
        """Поиск в интернете и генерация ответа."""
        search_query = await self._make_search_query(question)
        results = await self.web_search.search(search_query)
        if not results:
            return "Не удалось найти результаты в интернете."

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
        doc_results = await self._retrieve_and_rerank(user_id, question)
        doc_context = self._build_context(doc_results, MAX_CONTEXT_LENGTH // 2) if doc_results else ""

        search_query = await self._make_search_query(question)
        web_results = await self.web_search.search(search_query)
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
            prompt = PROMPT_TEMPLATE.format(
                context=doc_context, question=question, history=""
            )
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
