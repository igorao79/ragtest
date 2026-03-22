"""Веб-поиск через DuckDuckGo (бесплатный, без API-ключа)."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Слова-команды боту — не несут поисковой ценности
_BOT_COMMANDS = {
    "расскажи", "скажи", "найди", "покажи", "подскажи", "объясни",
    "можешь", "пожалуйста", "интересного", "интересное",
    "мне", "ещё", "еще", "вообще",
    "интернете", "документе", "документах", "файле", "файлах",
    # English
    "tell", "find", "show", "explain", "please", "interesting",
    "more", "also",
}

# Кириллица — для определения языка запроса
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


def _detect_russian(text: str) -> bool:
    """Определить, содержит ли текст кириллицу."""
    return bool(_CYRILLIC_RE.search(text))


def clean_search_query(query: str) -> str:
    """Мягкая очистка запроса: убирает только команды боту."""
    cleaned = re.sub(r"[^\w\s-]", " ", query)
    words = cleaned.split()
    result_words = [w for w in words if w.lower() not in _BOT_COMMANDS]
    result = " ".join(result_words).strip()
    if len(result) < 3:
        return " ".join(words)
    return result


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
        cleaned = clean_search_query(query)
        logger.info("Веб-поиск: '%s' -> '%s'", query, cleaned)

        loop = asyncio.get_event_loop()

        # Определяем регион по языку запроса
        region = "ru-ru" if _detect_russian(cleaned) else "wt-wt"

        results = await loop.run_in_executor(
            None, partial(self._search_sync, cleaned, limit, region)
        )

        # Fallback: если 0 результатов и запрос был очищен — пробуем оригинал
        if not results and cleaned != query:
            logger.info("Повторный поиск: '%s'", query)
            results = await loop.run_in_executor(
                None, partial(self._search_sync, query, limit, region)
            )

        # Fallback 2: если region не помог — пробуем wt-wt
        if not results and region != "wt-wt":
            logger.info("Повторный поиск без региона: '%s'", cleaned)
            results = await loop.run_in_executor(
                None, partial(self._search_sync, cleaned, limit, "wt-wt")
            )

        return results

    def _search_sync(
        self, query: str, max_results: int, region: str = "wt-wt"
    ) -> list[SearchResult]:
        """Синхронный поиск (выполняется в thread executor)."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw = list(ddgs.text(
                    keywords=query,
                    region=region,
                    safesearch="moderate",
                    max_results=max_results,
                    backend="html",
                ))

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
