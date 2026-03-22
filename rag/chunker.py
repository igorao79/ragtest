"""Разбиение текста на чанки с overlap и учётом границ предложений."""

import logging
import re

logger = logging.getLogger(__name__)

# Паттерн конца предложения
_SENTENCE_END = re.compile(r"[.!?]\s+")
_MIN_CHUNK_LENGTH = 20


class TextChunker:
    """Разбивает текст на чанки фиксированного размера с перекрытием."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Разбить текст на чанки.

        Args:
            text: Исходный текст.

        Returns:
            Список чанков.
        """
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)

            # Попытка найти конец предложения вблизи границы
            if end < length:
                boundary = self._find_sentence_boundary(text, start, end)
                if boundary is not None:
                    end = boundary

            chunk = text[start:end].strip()
            if len(chunk) >= _MIN_CHUNK_LENGTH:
                chunks.append(chunk)

            # Следующее начало с учётом overlap
            if end >= length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        logger.info("Текст разбит на %d чанков", len(chunks))
        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int | None:
        """Найти ближайший конец предложения к позиции end."""
        # Ищем в последних 20% чанка
        search_start = start + int(self.chunk_size * 0.8)
        search_region = text[search_start:end]

        matches = list(_SENTENCE_END.finditer(search_region))
        if matches:
            # Берём последнее совпадение — ближе к границе чанка
            return search_start + matches[-1].end()
        return None
