"""Веб-поиск через DuckDuckGo (бесплатный, без API-ключа)."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


def _detect_russian(text: str) -> bool:
    """Определить, содержит ли текст кириллицу."""
    return bool(_CYRILLIC_RE.search(text))


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
        """Выполнить поиск в интернете."""
        import asyncio
        from functools import partial

        limit = max_results or self.max_results
        region = "ru-ru" if _detect_russian(query) else "wt-wt"

        logger.info("Веб-поиск: '%s', region=%s", query, region)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, partial(self._search_sync, query, limit, region)
        )

        # Fallback без региона
        if not results and region != "wt-wt":
            logger.info("Fallback без региона: '%s'", query)
            results = await loop.run_in_executor(
                None, partial(self._search_sync, query, limit, "wt-wt")
            )

        return results

    def _search_sync(
        self, query: str, max_results: int, region: str = "wt-wt"
    ) -> list[SearchResult]:
        """Синхронный поиск через ddgs."""
        try:
            from ddgs import DDGS

            raw = DDGS().text(
                keywords=query,
                region=region,
                safesearch="moderate",
                max_results=max_results,
            )

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in raw
            ]

            results = self._filter_nsfw(results)

            logger.info(
                "Веб-поиск '%s' (region=%s): %d результатов",
                query, region, len(results),
            )
            return results

        except Exception as e:
            logger.error("Ошибка веб-поиска: %s", e)
            return []

    @staticmethod
    def _filter_nsfw(results: list[SearchResult]) -> list[SearchResult]:
        """Отфильтровать NSFW-результаты."""
        blocked_domains = {
            "pornhub", "xvideos", "xnxx", "xhamster", "redtube",
            "youporn", "tube8", "spankbang", "brazzers", "onlyfans",
            "chaturbate", "stripchat", "livejasmin", "cam4",
        }
        filtered = []
        for r in results:
            url_lower = r.url.lower()
            if any(domain in url_lower for domain in blocked_domains):
                continue
            title_lower = r.title.lower()
            if any(word in title_lower for word in ("porn", "xxx", "sex", "nude", "nsfw")):
                continue
            filtered.append(r)
        return filtered

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        """Форматировать результаты поиска в текст для контекста."""
        if not results:
            return ""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}\n{r.snippet}\nИсточник: {r.url}")
        return "\n\n".join(parts)

    @staticmethod
    def format_results_short(results: list[SearchResult]) -> str:
        """Краткий формат для отображения пользователю."""
        if not results:
            return "Ничего не найдено."
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. [{r.title}]({r.url})")
        return "\n".join(parts)
