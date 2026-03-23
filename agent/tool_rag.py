"""Инструмент: поиск по загруженным документам (RAG)."""

from typing import Any

from agent.tools import BaseTool, ToolParam, ToolResult


class RAGQueryTool(BaseTool):
    """Поиск ответа по загруженным документам пользователя."""

    name = "rag_query"
    description = (
        "Поиск по загруженным документам пользователя. "
        "Используй когда вопрос касается содержимого файлов, "
        "которые пользователь загрузил ранее."
    )
    params = [
        ToolParam("query", "Поисковый запрос к документам", "string", True),
    ]

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        user_id = kwargs.get("_user_id", 0)
        collection = kwargs.get("_collection")

        if not query:
            return ToolResult(success=False, data="", error="Пустой запрос")

        try:
            answer = await self._pipeline.answer(user_id, query, collection)
            return ToolResult(success=True, data=answer)
        except Exception as e:
            return ToolResult(success=False, data="", error=str(e))
