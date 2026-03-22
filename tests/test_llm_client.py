"""Тесты для OllamaClient."""

import pytest

from rag.llm_client import OllamaClient, OllamaConnectionError


@pytest.mark.asyncio
async def test_connection_error():
    """Тест что ошибка подключения поднимается корректно."""
    client = OllamaClient("http://localhost:99999", "test-model")
    with pytest.raises(OllamaConnectionError):
        await client.generate("test")
    await client.close()


@pytest.mark.asyncio
async def test_is_available_offline():
    """Тест is_available когда Ollama недоступна."""
    client = OllamaClient("http://localhost:99999", "test-model")
    assert await client.is_available() is False
    await client.close()


@pytest.mark.asyncio
async def test_generate_vision_connection_error():
    """Тест ошибки подключения для vision."""
    client = OllamaClient("http://localhost:99999", "test-model")
    with pytest.raises(OllamaConnectionError):
        await client.generate_vision("test", b"fake-image-data")
    await client.close()
