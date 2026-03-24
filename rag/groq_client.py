"""Groq API клиент с автоматическим fallback на Ollama."""

import logging
from collections.abc import AsyncIterator

from groq import AsyncGroq, APIError, RateLimitError, APIConnectionError

from rag.llm_client import OllamaClient, OllamaConnectionError, OllamaTimeoutError

logger = logging.getLogger(__name__)


class GroqClient:
    """Клиент Groq API, совместимый по интерфейсу с OllamaClient.

    При ошибках (rate limit, баланс, сеть) автоматически
    переключается на Ollama как fallback.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        fallback: OllamaClient | None = None,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self.model = model
        self._fallback = fallback
        self._use_fallback = False

    async def generate(self, prompt: str, system: str = "") -> str:
        """Сгенерировать ответ через Groq, fallback на Ollama."""
        if self._use_fallback and self._fallback:
            return await self._fallback.generate(prompt, system)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""

        except RateLimitError as e:
            logger.warning("Groq rate limit, fallback на Ollama: %s", e)
            return await self._do_fallback(prompt, system)

        except (APIError, APIConnectionError) as e:
            logger.error("Groq API ошибка, fallback на Ollama: %s", e)
            return await self._do_fallback(prompt, system)

        except Exception as e:
            logger.error("Groq неизвестная ошибка: %s", e)
            return await self._do_fallback(prompt, system)

    async def generate_stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Стриминг через Groq, fallback на Ollama."""
        if self._use_fallback and self._fallback:
            async for token in self._fallback.generate_stream(prompt, system):
                yield token
            return

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self._client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except (RateLimitError, APIError, APIConnectionError) as e:
            logger.warning("Groq stream ошибка, fallback: %s", e)
            if self._fallback:
                async for token in self._fallback.generate_stream(prompt, system):
                    yield token

    async def generate_vision(
        self, prompt: str, image_data: bytes, model: str | None = None
    ) -> str:
        """Vision — всегда через Ollama (Groq не поддерживает vision бесплатно)."""
        if self._fallback:
            return await self._fallback.generate_vision(prompt, image_data, model)
        raise OllamaConnectionError("Vision не поддерживается без Ollama fallback")

    async def is_available(self) -> bool:
        """Проверить доступность Groq API."""
        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model=self.model,
                max_tokens=5,
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning("Groq недоступен: %s", e)
            return False

    async def _do_fallback(self, prompt: str, system: str) -> str:
        """Выполнить запрос через fallback (Ollama)."""
        if self._fallback:
            logger.info("Используем Ollama fallback")
            return await self._fallback.generate(prompt, system)
        raise OllamaConnectionError(
            "Groq API недоступен и нет fallback. Проверьте API ключ."
        )

    async def close(self) -> None:
        """Закрыть клиент."""
        await self._client.close()
        if self._fallback:
            await self._fallback.close()
