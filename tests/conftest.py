"""Fixtures и настройки для тестов."""

import os

# Устанавливаем env до импорта модулей
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token-for-tests")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "gemma3:4b")
