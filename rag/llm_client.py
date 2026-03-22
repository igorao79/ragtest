"""HTTP-клиент для Ollama API (/api/generate, /api/chat)."""

import base64
import logging
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Не удалось подключиться к Ollama."""


class OllamaTimeoutError(Exception):
    """Превышено время ожидания ответа от Ollama."""


class OllamaClient:
    """Асинхронный клиент для Ollama API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, prompt: str, system: str = "") -> str:
        """Сгенерировать ответ через Ollama."""
        url = f"{self.base_url}/api/generate"
        body = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "num_predict": 1024,
            },
        }
        try:
            response = await self._client.post(url, json=body)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Не удалось подключиться к Ollama ({self.base_url}): {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Таймаут при генерации ответа: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(
                f"Ошибка HTTP от Ollama: {e.response.status_code}"
            ) from e

    async def generate_stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Стриминг ответа по частям."""
        url = f"{self.base_url}/api/generate"
        body = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "num_predict": 1024,
            },
        }
        try:
            async with self._client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    import json
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Не удалось подключиться к Ollama ({self.base_url}): {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Таймаут при генерации ответа: {e}"
            ) from e

    async def generate_vision(
        self, prompt: str, image_data: bytes, model: str | None = None
    ) -> str:
        """Анализ изображения через vision-модель.

        Args:
            prompt: Текст запроса.
            image_data: Байты изображения.
            model: Модель для vision (по умолчанию self.model).

        Returns:
            Ответ модели.
        """
        url = f"{self.base_url}/api/generate"
        b64_image = base64.b64encode(image_data).decode("utf-8")
        body = {
            "model": model or self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {
                "num_predict": 1024,
            },
        }
        try:
            response = await self._client.post(url, json=body)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Не удалось подключиться к Ollama ({self.base_url}): {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Таймаут при анализе изображения: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(
                f"Ошибка HTTP от Ollama: {e.response.status_code}"
            ) from e

    async def is_available(self) -> bool:
        """Проверить доступность Ollama и наличие модели."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            for name in models:
                if name == self.model or name.startswith(f"{self.model}:"):
                    return True
                if self.model in name:
                    return True
            logger.warning(
                "Модель %s не найдена. Доступные: %s", self.model, models
            )
            return False
        except Exception as e:
            logger.error("Ollama недоступна: %s", e)
            return False

    async def close(self) -> None:
        """Закрыть HTTP-клиент."""
        await self._client.aclose()
