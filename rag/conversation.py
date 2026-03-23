"""Диалоговая память — хранение последних N сообщений пользователя."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Одно сообщение в истории."""
    role: str  # "user" или "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Хранит историю диалогов для каждого пользователя.

    Позволяет боту понимать контекст беседы:
    "а расскажи подробнее", "а что насчёт второго пункта" и т.д.
    """

    def __init__(self, max_messages: int = 10, ttl: int = 1800) -> None:
        """
        Args:
            max_messages: Максимальное количество сообщений на пользователя.
            ttl: Время жизни истории в секундах (30 мин по умолчанию).
        """
        self.max_messages = max_messages
        self.ttl = ttl
        self._history: dict[int, list[Message]] = defaultdict(list)

    def add_user_message(self, user_id: int, text: str) -> None:
        """Добавить сообщение пользователя."""
        self._cleanup_old(user_id)
        self._history[user_id].append(Message(role="user", text=text))
        self._trim(user_id)

    def add_assistant_message(self, user_id: int, text: str) -> None:
        """Добавить ответ ассистента."""
        self._cleanup_old(user_id)
        self._history[user_id].append(Message(role="assistant", text=text))
        self._trim(user_id)

    def get_history(self, user_id: int) -> list[Message]:
        """Получить историю диалога."""
        self._cleanup_old(user_id)
        return list(self._history[user_id])

    def get_context_string(self, user_id: int, max_chars: int = 1500) -> str:
        """Получить историю как строку для контекста промпта.

        Args:
            user_id: ID пользователя.
            max_chars: Максимальная длина контекста.

        Returns:
            Форматированная история диалога.
        """
        history = self.get_history(user_id)
        if not history:
            return ""

        # Берём последние сообщения (кроме текущего — оно ещё не добавлено)
        # Идём с конца, собирая пока не превысим лимит
        parts: list[str] = []
        total = 0
        for msg in reversed(history):
            role_label = "Пользователь" if msg.role == "user" else "Ассистент"
            line = f"{role_label}: {msg.text}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)

        if not parts:
            return ""

        parts.reverse()
        return "\n".join(parts)

    def clear(self, user_id: int) -> None:
        """Очистить историю пользователя."""
        self._history.pop(user_id, None)

    def _cleanup_old(self, user_id: int) -> None:
        """Удалить сообщения старше TTL."""
        if user_id not in self._history:
            return
        now = time.time()
        self._history[user_id] = [
            m for m in self._history[user_id]
            if now - m.timestamp < self.ttl
        ]

    def _trim(self, user_id: int) -> None:
        """Обрезать историю до max_messages."""
        if len(self._history[user_id]) > self.max_messages:
            self._history[user_id] = self._history[user_id][-self.max_messages:]
