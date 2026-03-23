"""Агентский роутер — мозг, решающий какой инструмент вызвать."""

import json
import logging
import re
from typing import Any

from agent.tools import ToolRegistry, ToolResult
from rag.llm_client import OllamaClient

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 3  # Максимум цепочек вызовов

_ROUTER_SYSTEM_PROMPT = """Ты — умный ассистент с доступом к инструментам.

Доступные инструменты:
{tools_description}

ПРАВИЛА:
1. Если для ответа нужен инструмент — ответь ТОЛЬКО JSON-объектом:
   {{"tool": "имя_инструмента", "args": {{"параметр": "значение"}}}}
2. Если инструмент НЕ нужен — ответь обычным текстом.
3. Используй calculator для любых математических вычислений.
4. Используй web_search для вопросов о текущих событиях, датах, фактах.
5. Используй rag_query когда вопрос связан с загруженными документами.
6. Используй weather для вопросов о погоде.
7. Используй python для сложных вычислений и генерации данных.
8. НИКОГДА не оборачивай JSON в markdown-блоки.
9. Отвечай на том же языке, на котором задан вопрос."""

_TOOL_RESULT_PROMPT = """Результат инструмента "{tool_name}":
---
{result}
---

На основе этого результата ответь пользователю на его вопрос.
Отвечай кратко и по делу. Не упоминай инструменты.
Исходный вопрос: {question}"""

# Паттерн для извлечения JSON из ответа LLM
_JSON_PATTERN = re.compile(
    r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}[^{}]*\}',
    re.DOTALL,
)


class AgentRouter:
    """Маршрутизатор агента — определяет нужен ли инструмент и вызывает его."""

    def __init__(self, llm_client: OllamaClient, registry: ToolRegistry) -> None:
        self.llm_client = llm_client
        self.registry = registry

    def _build_system_prompt(self) -> str:
        """Построить системный промпт с описанием инструментов."""
        return _ROUTER_SYSTEM_PROMPT.format(
            tools_description=self.registry.get_tools_prompt()
        )

    def _parse_tool_call(self, response: str) -> dict | None:
        """Извлечь вызов инструмента из ответа LLM.

        Returns:
            {"tool": str, "args": dict} или None если это обычный текст.
        """
        text = response.strip()

        # Убираем markdown-обёртки если LLM добавила
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Пробуем распарсить как JSON напрямую
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data and "args" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Ищем JSON-паттерн в тексте
        match = _JSON_PATTERN.search(text)
        if match:
            try:
                data = json.loads(match.group())
                if "tool" in data and "args" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    async def route(
        self,
        question: str,
        user_id: int = 0,
        collection: str | None = None,
        conversation_context: str = "",
    ) -> str:
        """Обработать вопрос пользователя через агентский цикл.

        1. Отправляем вопрос в LLM с описанием инструментов
        2. Если LLM вернула tool call — выполняем инструмент
        3. Результат отправляем обратно в LLM для формирования ответа
        4. Повторяем до MAX_TOOL_ROUNDS или пока LLM не ответит текстом

        Args:
            question: Вопрос пользователя.
            user_id: ID пользователя (для RAG).
            collection: Активная коллекция (для RAG).
            conversation_context: История диалога.

        Returns:
            Финальный текстовый ответ.
        """
        system = self._build_system_prompt()

        # Добавляем контекст диалога если есть
        prompt = question
        if conversation_context:
            prompt = f"История диалога:\n{conversation_context}\n\nВопрос: {question}"

        for round_num in range(_MAX_TOOL_ROUNDS):
            logger.info("Агент: раунд %d, вопрос: %s", round_num + 1, prompt[:100])

            response = await self.llm_client.generate(prompt, system=system)
            response = response.strip()

            # Пробуем распарсить как tool call
            tool_call = self._parse_tool_call(response)

            if tool_call is None:
                # LLM ответила обычным текстом — готово
                logger.info("Агент: прямой ответ (без инструмента)")
                return response

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            logger.info("Агент: вызов инструмента %s(%s)", tool_name, tool_args)

            # Находим и выполняем инструмент
            tool = self.registry.get(tool_name)
            if tool is None:
                logger.warning("Агент: инструмент %s не найден", tool_name)
                # Просим LLM ответить без инструмента
                prompt = (
                    f"Инструмент '{tool_name}' не найден. "
                    f"Ответь на вопрос без инструментов: {question}"
                )
                continue

            # Добавляем системные параметры
            tool_args["_user_id"] = user_id
            tool_args["_collection"] = collection

            result = await tool.execute(**tool_args)

            if result.success:
                # Отправляем результат обратно в LLM
                prompt = _TOOL_RESULT_PROMPT.format(
                    tool_name=tool_name,
                    result=result.data,
                    question=question,
                )
            else:
                prompt = (
                    f"Инструмент {tool_name} вернул ошибку: {result.error}\n"
                    f"Попробуй ответить на вопрос другим способом: {question}"
                )

        # Если исчерпали раунды — возвращаем последний ответ
        logger.warning("Агент: исчерпаны раунды, возвращаем последний ответ")
        return response

    async def route_stream(
        self,
        question: str,
        user_id: int = 0,
        collection: str | None = None,
        conversation_context: str = "",
    ):
        """Стриминг версия route — для обычных ответов стримит, для tool calls нет."""
        system = self._build_system_prompt()

        prompt = question
        if conversation_context:
            prompt = f"История диалога:\n{conversation_context}\n\nВопрос: {question}"

        # Первый запрос — без стриминга, чтобы определить tool call
        response = await self.llm_client.generate(prompt, system=system)
        response = response.strip()

        tool_call = self._parse_tool_call(response)

        if tool_call is None:
            # Обычный ответ — стримим его
            async for token in self.llm_client.generate_stream(prompt, system=system):
                yield token
            return

        # Tool call — выполняем и стримим финальный ответ
        tool_name = tool_call["tool"]
        tool_args = tool_call.get("args", {})
        tool_args["_user_id"] = user_id
        tool_args["_collection"] = collection

        tool = self.registry.get(tool_name)
        if tool is None:
            yield f"Инструмент '{tool_name}' не найден."
            return

        logger.info("Агент (стрим): вызов %s(%s)", tool_name, tool_args)
        result = await tool.execute(**tool_args)

        if result.success:
            final_prompt = _TOOL_RESULT_PROMPT.format(
                tool_name=tool_name,
                result=result.data,
                question=question,
            )
        else:
            final_prompt = (
                f"Инструмент {tool_name} вернул ошибку: {result.error}\n"
                f"Ответь на вопрос: {question}"
            )

        async for token in self.llm_client.generate_stream(final_prompt, system=system):
            yield token
