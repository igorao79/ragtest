"""HTTP-клиент для Ollama API (/api/generate)."""

import logging

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
        """Сгенерировать ответ через Ollama.

        Args:
            prompt: Текст промпта.
            system: Системный промпт.

        Returns:
            Сгенерированный текст.

        Raises:
            OllamaConnectionError: Если не удалось подключиться.
            OllamaTimeoutError: Если превышен таймаут.
        """
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

    async def is_available(self) -> bool:
        """Проверить доступность Ollama и наличие модели.

        Returns:
            True если Ollama доступна и модель загружена.
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Проверяем точное совпадение или совпадение без тега
            for name in models:
                if name == self.model or name.startswith(f"{self.model}:"):
                    return True
                # llama3.2:3b может быть в списке как llama3.2:3b-instruct-...
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
