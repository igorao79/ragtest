"""Агентский роутер — мозг, решающий какой инструмент вызвать."""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from agent.tools import ToolRegistry, ToolResult
from rag.llm_client import OllamaClient

# Тип callback-функции для уведомления о вызове инструмента
# Принимает (tool_name, tool_description) и возвращает awaitable
ToolNotifyCallback = Callable[[str, str], Awaitable[Any]]

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 3  # Максимум цепочек вызовов

_ROUTER_SYSTEM_PROMPT = """Ты — умный ассистент с доступом к инструментам.

ГЛАВНОЕ ПРАВИЛО: Если ты УВЕРЕН в ответе — отвечай СРАЗУ текстом, БЕЗ инструментов.
Ты уже знаешь общеизвестные факты: столицы, президенты, история, наука, математика.
Инструменты нужны ТОЛЬКО когда ты НЕ знаешь ответ или нужны свежие/точные данные.

Доступные инструменты:
{tools_description}

КОГДА использовать инструменты:
- web_search: ТОЛЬКО для свежих новостей, текущих цен, событий последних дней, или когда ты НЕ уверен в ответе
- rag_query: когда пользователь спрашивает о СВОИХ загруженных документах/файлах
- calculator: когда нужно ТОЧНОЕ вычисление (большие числа, формулы)
- python: сложный код, генерация данных, обработка
- weather: текущая погода в городе

КОГДА НЕ использовать:
- Общеизвестные факты (сколько президентов, столица страны, кто написал книгу)
- Простые вопросы по истории, географии, науке
- Объяснения концепций, советы
- Простая арифметика (2+2, 10*5)

Формат вызова инструмента (ТОЛЬКО когда нужен):
{{"tool": "имя", "args": {{"параметр": "значение"}}}}

Отвечай на том же языке, на котором задан вопрос."""

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


# Красивые имена инструментов для отображения пользователю
TOOL_DISPLAY = {
    "rag_query":  ("📄", "Поиск по документам"),
    "web_search": ("🌐", "Поиск в интернете"),
    "calculator": ("🔢", "Калькулятор"),
    "python":     ("🐍", "Выполнение кода"),
    "weather":    ("🌤", "Проверка погоды"),
}


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
        on_tool_call: ToolNotifyCallback | None = None,
    ) -> str:
        """Обработать вопрос пользователя через агентский цикл.

        Args:
            question: Вопрос пользователя.
            user_id: ID пользователя (для RAG).
            collection: Активная коллекция (для RAG).
            conversation_context: История диалога.
            on_tool_call: Callback вызываемый при использовании инструмента.
                          Принимает (tool_name, display_text).

        Returns:
            Финальный текстовый ответ.
        """
        system = self._build_system_prompt()

        prompt = question
        if conversation_context:
            prompt = f"История диалога:\n{conversation_context}\n\nВопрос: {question}"

        for round_num in range(_MAX_TOOL_ROUNDS):
            logger.info("Агент: раунд %d, вопрос: %s", round_num + 1, prompt[:100])

            response = await self.llm_client.generate(prompt, system=system)
            response = response.strip()

            tool_call = self._parse_tool_call(response)

            if tool_call is None:
                logger.info("Агент: прямой ответ (без инструмента)")
                return response

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            logger.info("Агент: вызов инструмента %s(%s)", tool_name, tool_args)

            # Уведомляем о вызове инструмента
            if on_tool_call:
                icon, label = TOOL_DISPLAY.get(tool_name, ("🔧", tool_name))
                display = f"{icon} *{label}*..."
                await on_tool_call(tool_name, display)

            tool = self.registry.get(tool_name)
            if tool is None:
                logger.warning("Агент: инструмент %s не найден", tool_name)
                prompt = (
                    f"Инструмент '{tool_name}' не найден. "
                    f"Ответь на вопрос без инструментов: {question}"
                )
                continue

            tool_args["_user_id"] = user_id
            tool_args["_collection"] = collection

            result = await tool.execute(**tool_args)

            if result.success:
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
