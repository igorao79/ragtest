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
OLLAMA_MODEL: str = _get_env("OLLAMA_MODEL", "llama3.2:3b")

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
ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]

# System prompt
SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент. Отвечай на вопросы пользователя "
    "ТОЛЬКО на основе предоставленного контекста. "
    "Если в контексте нет информации для ответа — честно скажи, что не знаешь. "
    "Отвечай на том же языке, на котором задан вопрос. "
    "Будь кратким и точным. Ссылайся на источник, если это возможно."
)

PROMPT_TEMPLATE: str = "Контекст:\n---\n{context}\n---\n\nВопрос: {question}\n\nОтвет:"
