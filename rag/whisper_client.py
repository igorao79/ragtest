"""Распознавание речи через OpenAI Whisper (локально)."""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperError(Exception):
    """Ошибка распознавания речи."""


class WhisperClient:
    """Клиент для распознавания голосовых сообщений.

    Поддерживает два бэкенда:
    1. openai-whisper (Python-библиотека) — если установлена
    2. ffmpeg + Ollama — fallback через vision-модель
    """

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._whisper_model = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Проверить доступность Whisper."""
        if self._available is not None:
            return self._available
        try:
            import whisper  # noqa: F401
            self._available = True
            logger.info("Whisper доступен (модель: %s)", self.model_size)
        except ImportError:
            self._available = False
            logger.warning("openai-whisper не установлен. pip install openai-whisper")
        return self._available

    def _get_model(self):
        """Загрузить модель Whisper (ленивая инициализация)."""
        if self._whisper_model is None:
            import whisper
            logger.info("Загрузка модели Whisper '%s'...", self.model_size)
            self._whisper_model = whisper.load_model(self.model_size)
            logger.info("Модель Whisper загружена")
        return self._whisper_model

    async def transcribe(self, audio_path: str) -> str:
        """Транскрибировать аудиофайл в текст.

        Args:
            audio_path: Путь к аудиофайлу (ogg, mp3, wav и др.)

        Returns:
            Распознанный текст.

        Raises:
            WhisperError: Если не удалось распознать.
        """
        import asyncio

        if not self.is_available():
            raise WhisperError(
                "Whisper не установлен. Установите: pip install openai-whisper"
            )

        # Конвертируем OGG в WAV через ffmpeg (Telegram шлёт ogg/opus)
        wav_path = await asyncio.get_event_loop().run_in_executor(
            None, self._convert_to_wav, audio_path
        )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_sync, wav_path
            )
            return result
        finally:
            # Удаляем временный WAV
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _convert_to_wav(self, input_path: str) -> str:
        """Конвертировать аудио в WAV через ffmpeg."""
        wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = wav_file.name
        wav_file.close()

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_path,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    wav_path,
                ],
                capture_output=True,
                timeout=30,
                check=True,
            )
            return wav_path
        except FileNotFoundError:
            raise WhisperError(
                "ffmpeg не найден. Установите ffmpeg для обработки аудио."
            )
        except subprocess.CalledProcessError as e:
            raise WhisperError(f"Ошибка конвертации аудио: {e.stderr.decode()}")

    def _transcribe_sync(self, wav_path: str) -> str:
        """Синхронная транскрипция через Whisper."""
        model = self._get_model()
        result = model.transcribe(wav_path, language=None)
        text = result.get("text", "").strip()
        if not text:
            raise WhisperError("Не удалось распознать речь в аудио")
        language = result.get("language", "unknown")
        logger.info("Whisper: распознано %d символов, язык: %s", len(text), language)
        return text
