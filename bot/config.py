"""Конфигурация бота. Загрузка параметров из переменных окружения."""

import os
from pathlib import Path


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Получить переменную окружения."""
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(f"Переменная окружения {key} обязательна, но не задана")
    return value  # type: ignore[return-value]


# Telegram
TELEGRAM_BOT_TOKEN: str = _get_env("TELEGRAM_BOT_TOKEN", required=True)

# Ollama
OLLAMA_BASE_URL: str = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _get_env("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_VISION_MODEL: str = _get_env("OLLAMA_VISION_MODEL", "qwen3-vl:8b")

# ChromaDB
CHROMA_PERSIST_DIR: str = _get_env(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "chroma_db"),
)

# Chunking
CHUNK_SIZE: int = int(_get_env("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(_get_env("CHUNK_OVERLAP", "64"))

# Retrieval
TOP_K: int = int(_get_env("TOP_K", "4"))
MAX_CONTEXT_LENGTH: int = int(_get_env("MAX_CONTEXT_LENGTH", "3000"))

# File upload limits
MAX_FILE_SIZE_MB: int = int(_get_env("MAX_FILE_SIZE_MB", "20"))
ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"]

# Rate limiting
RATE_LIMIT_MESSAGES: int = int(_get_env("RATE_LIMIT_MESSAGES", "10"))
RATE_LIMIT_WINDOW: int = int(_get_env("RATE_LIMIT_WINDOW", "60"))

# Cache
CACHE_TTL: int = int(_get_env("CACHE_TTL", "300"))
CACHE_MAX_SIZE: int = int(_get_env("CACHE_MAX_SIZE", "100"))

# Conversation memory
CONVERSATION_MAX_MESSAGES: int = int(_get_env("CONVERSATION_MAX_MESSAGES", "10"))
CONVERSATION_TTL: int = int(_get_env("CONVERSATION_TTL", "1800"))

# Whisper
WHISPER_MODEL: str = _get_env("WHISPER_MODEL", "base")

# Reranking
RERANK_ENABLED: bool = _get_env("RERANK_ENABLED", "true").lower() == "true"
RERANK_FETCH_K: int = int(_get_env("RERANK_FETCH_K", "8"))

# System prompt
SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент. Отвечай на вопросы пользователя "
    "ТОЛЬКО на основе предоставленного контекста. "
    "Если в контексте нет информации для ответа — честно скажи, что не знаешь. "
    "Отвечай на том же языке, на котором задан вопрос. "
    "Будь кратким и точным. Ссылайся на источник, если это возможно. "
    "Используй Markdown для форматирования: **жирный**, *курсив*, `код`, списки."
)

PROMPT_TEMPLATE: str = "Контекст:\n---\n{context}\n---\n\n{history}Вопрос: {question}\n\nОтвет:"

SUMMARY_SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент. Сделай краткий структурированный пересказ "
    "предоставленного текста. Используй Markdown: заголовки, списки, выделение. "
    "Отвечай на том же языке, что и текст."
)

SUMMARY_PROMPT_TEMPLATE: str = (
    "Текст для суммаризации:\n---\n{context}\n---\n\n"
    "Сделай краткий структурированный пересказ:"
)

# Web search
WEB_SEARCH_MAX_RESULTS: int = int(_get_env("WEB_SEARCH_MAX_RESULTS", "5"))

WEB_SEARCH_SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент. Отвечай на вопросы пользователя "
    "на основе результатов поиска в интернете. "
    "Ссылайся на источники, указывая номер в квадратных скобках [1], [2] и т.д. "
    "Отвечай на том же языке, на котором задан вопрос. "
    "Будь кратким и точным. Используй Markdown для форматирования."
)

WEB_SEARCH_PROMPT_TEMPLATE: str = (
    "Результаты поиска в интернете:\n---\n{web_context}\n---\n\n"
    "Вопрос: {question}\n\nОтвет:"
)

COMBINED_SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент. Отвечай на вопросы пользователя, используя "
    "ОБА источника: загруженные документы И результаты поиска в интернете. "
    "Сначала используй информацию из документов, затем дополни из интернета. "
    "Ссылайся на источники. Отвечай на том же языке, на котором задан вопрос. "
    "Будь кратким и точным. Используй Markdown для форматирования."
)

COMBINED_PROMPT_TEMPLATE: str = (
    "Контекст из документов:\n---\n{doc_context}\n---\n\n"
    "Результаты поиска в интернете:\n---\n{web_context}\n---\n\n"
    "Вопрос: {question}\n\nОтвет:"
)

VISION_SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент с возможностью анализировать изображения. "
    "Опиши содержимое изображения подробно. Если на изображении есть текст — "
    "извлеки его. Отвечай на том же языке, на котором задан вопрос."
)
