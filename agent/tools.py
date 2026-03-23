"""Базовый класс и реестр инструментов агента."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolParam:
    """Описание параметра инструмента."""
    name: str
    description: str
    type: str = "string"
    required: bool = True


@dataclass
class ToolResult:
    """Результат выполнения инструмента."""
    success: bool
    data: str
    error: str | None = None


class BaseTool(ABC):
    """Базовый класс для всех инструментов."""

    name: str = ""
    description: str = ""
    params: list[ToolParam] = []

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Выполнить инструмент."""
        ...

    def to_schema(self) -> dict:
        """Сериализовать описание инструмента для промпта LLM."""
        params_schema = {}
        for p in self.params:
            params_schema[p.name] = {
                "type": p.type,
                "description": p.description,
                "required": p.required,
            }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params_schema,
        }


class ToolRegistry:
    """Реестр всех доступных инструментов."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Зарегистрировать инструмент."""
        self._tools[tool.name] = tool
        logger.info("Инструмент зарегистрирован: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        """Получить инструмент по имени."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Список всех инструментов."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Получить схемы всех инструментов для промпта."""
        return [t.to_schema() for t in self._tools.values()]

    def get_tools_prompt(self) -> str:
        """Сформировать описание инструментов для системного промпта."""
        lines = []
        for tool in self._tools.values():
            params_desc = ", ".join(
                f"{p.name}: {p.description}" for p in tool.params
            )
            lines.append(f"- {tool.name}: {tool.description}")
            if params_desc:
                lines.append(f"  Параметры: {params_desc}")
        return "\n".join(lines)
