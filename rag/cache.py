"""Простой TTL-кеш для ответов."""

import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ResponseCache:
    """LRU-кеш с TTL для ответов LLM."""

    def __init__(self, max_size: int = 100, ttl: int = 300) -> None:
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl

    def _make_key(self, user_id: int, question: str) -> str:
        """Создать ключ кеша."""
        return f"{user_id}:{question.lower().strip()}"

    def get(self, user_id: int, question: str) -> str | None:
        """Получить ответ из кеша."""
        key = self._make_key(user_id, question)
        if key not in self._cache:
            return None

        answer, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None

        # Перемещаем в конец (LRU)
        self._cache.move_to_end(key)
        logger.debug("Кеш-хит для %s", key[:50])
        return answer

    def put(self, user_id: int, question: str, answer: str) -> None:
        """Сохранить ответ в кеш."""
        key = self._make_key(user_id, question)
        self._cache[key] = (answer, time.time())
        self._cache.move_to_end(key)

        # Удаляем старые, если превышен лимит
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate_user(self, user_id: int) -> None:
        """Очистить кеш для пользователя."""
        keys_to_delete = [k for k in self._cache if k.startswith(f"{user_id}:")]
        for k in keys_to_delete:
            del self._cache[k]

    def clear(self) -> None:
        """Очистить весь кеш."""
        self._cache.clear()
