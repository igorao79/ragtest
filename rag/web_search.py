"""Веб-поиск через DuckDuckGo (бесплатный, без API-ключа)."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Стоп-слова для очистки поисковых запросов (русские и английские)
_STOP_WORDS = {
    # Русские
    "что", "кто", "как", "где", "когда", "почему", "зачем", "какой", "какая",
    "какое", "какие", "чем", "чего", "кого", "кому", "чему", "ещё", "еще",
    "расскажи", "скажи", "найди", "покажи", "подскажи", "объясни",
    "можешь", "можно", "пожалуйста", "интересного", "интересное",
    "мне", "про", "для", "это", "этот", "эта", "эти", "этого",
    "тоже", "также", "ещё", "вот", "так", "вообще", "очень",
    "бы", "же", "ли", "ну", "да", "нет", "не", "ни",
    "а", "и", "о", "в", "на", "из", "за", "от", "до", "по", "с", "к", "у",
    "об", "со", "во", "ко",
    "интернете", "документе", "документах", "файле", "файлах",
    # Английские
    "what", "who", "how", "where", "when", "why", "which",
    "tell", "me", "about", "find", "show", "explain", "can", "you",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "for", "of", "with", "from",
    "more", "also", "please", "interesting",
}


def clean_search_query(query: str) -> str:
    """Очистить пользовательский запрос от стоп-слов для поиска.

    Оставляет ключевые слова — имена собственные, термины, числа.
    """
    # Убираем знаки пунктуации кроме дефисов
    cleaned = re.sub(r"[^\w\s-]", " ", query)
    words = cleaned.split()

    # Фильтруем стоп-слова, но сохраняем слова с заглавной буквы (имена)
    important = []
    for word in words:
        lower = word.lower()
        if lower not in _STOP_WORDS:
            important.append(word)
        elif word[0].isupper() and len(word) > 2:
            # Имена собственные сохраняем даже если они в стоп-словах
            important.append(word)

    result = " ".join(important)

    # Если всё отфильтровалось — возвращаем исходный запрос без пунктуации
    if len(result.strip()) < 3:
        return " ".join(words)

    return result.strip()


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

        # Очищаем запрос от стоп-слов
        cleaned_query = clean_search_query(query)
        logger.info("Поиск: '%s' → '%s'", query, cleaned_query)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, partial(self._search_sync, cleaned_query, limit)
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
