"""Веб-поиск через DuckDuckGo (бесплатный, без API-ключа)."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска."""

    title: str
    url: str
    snippet: str


class WebSearchClient:
    """Клиент для поиска в интернете через DuckDuckGo."""

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Выполнить поиск в интернете.

        Args:
            query: Поисковый запрос.
            max_results: Количество результатов (по умолчанию self.max_results).

        Returns:
            Список результатов поиска.
        """
        import asyncio
        from functools import partial

        limit = max_results or self.max_results
        # duckduckgo_search — синхронная, запускаем в потоке
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, partial(self._search_sync, query, limit)
        )
        return results

    def _search_sync(self, query: str, max_results: int) -> list[SearchResult]:
        """Синхронный поиск (выполняется в thread executor)."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in raw
            ]
            logger.info("Веб-поиск '%s': найдено %d результатов", query, len(results))
            return results

        except Exception as e:
            logger.error("Ошибка веб-поиска: %s", e)
            return []

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        """Форматировать результаты поиска в текст для контекста.

        Args:
            results: Список результатов.

        Returns:
            Отформатированный текст.
        """
        if not results:
            return ""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}\n{r.snippet}\nИсточник: {r.url}")
        return "\n\n".join(parts)

    @staticmethod
    def format_results_short(results: list[SearchResult]) -> str:
        """Краткий формат для отображения пользователю.

        Args:
            results: Список результатов.

        Returns:
            Краткий текст со ссылками.
        """
        if not results:
            return "Ничего не найдено."
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. [{r.title}]({r.url})")
        return "\n".join(parts)
