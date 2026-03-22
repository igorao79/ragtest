"""Rate limiter для ограничения запросов пользователей."""

import time


class RateLimiter:
    """Скользящее окно для ограничения частоты запросов."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[int, list[float]] = {}

    def is_allowed(self, user_id: int) -> bool:
        """Проверить, может ли пользователь сделать запрос."""
        now = time.time()
        if user_id not in self._requests:
            self._requests[user_id] = []

        # Очищаем старые запросы
        self._requests[user_id] = [
            t for t in self._requests[user_id]
            if now - t < self.window_seconds
        ]

        if len(self._requests[user_id]) >= self.max_requests:
            return False

        self._requests[user_id].append(now)
        return True

    def remaining(self, user_id: int) -> int:
        """Сколько запросов осталось."""
        now = time.time()
        if user_id not in self._requests:
            return self.max_requests
        active = [t for t in self._requests[user_id] if now - t < self.window_seconds]
        return max(0, self.max_requests - len(active))

    def retry_after(self, user_id: int) -> int:
        """Через сколько секунд можно повторить."""
        if user_id not in self._requests or not self._requests[user_id]:
            return 0
        oldest = min(
            t for t in self._requests[user_id]
            if time.time() - t < self.window_seconds
        )
        return max(0, int(self.window_seconds - (time.time() - oldest)))
