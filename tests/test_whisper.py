"""Тесты для модуля распознавания речи."""

from rag.whisper_client import WhisperClient, WhisperError
import pytest


class TestWhisperClient:
    def test_init(self):
        client = WhisperClient(model_size="base")
        assert client.model_size == "base"

    def test_is_available_without_whisper(self):
        """Проверяем что is_available не падает."""
        client = WhisperClient()
        # Результат зависит от того, установлен ли whisper
        result = client.is_available()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_transcribe_nonexistent_file(self):
        client = WhisperClient()
        if not client.is_available():
            pytest.skip("whisper not installed")
        with pytest.raises((WhisperError, Exception)):
            await client.transcribe("/nonexistent/audio.ogg")
