"""Инструмент: поиск в интернете."""

from typing import Any

from agent.tools import BaseTool, ToolParam, ToolResult


class WebSearchTool(BaseTool):
    """Поиск информации в интернете через DuckDuckGo."""

    name = "web_search"
    description = (
        "Поиск актуальной информации в интернете. "
        "Используй для вопросов о текущих событиях, фактах, погоде, "
        "новостях или чего нет в документах пользователя."
    )
    params = [
        ToolParam("query", "Поисковый запрос (3-6 ключевых слов)", "string", True),
    ]

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, data="", error="Пустой запрос")

        try:
            answer = await self._pipeline.web_answer(query)
            return ToolResult(success=True, data=answer)
        except Exception as e:
            return ToolResult(success=False, data="", error=str(e))
