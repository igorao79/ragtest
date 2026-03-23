"""Инструмент: выполнение Python-кода в песочнице."""

import asyncio
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent.tools import BaseTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# Лимиты безопасности
_TIMEOUT = 10  # секунд
_MAX_OUTPUT = 2000  # символов

# Запрещённые импорты
_BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "importlib", "ctypes", "signal", "multiprocessing",
    "threading", "pickle", "shelve", "sqlite3",
}


def _check_code_safety(code: str) -> str | None:
    """Проверить код на опасные конструкции. Возвращает ошибку или None."""
    for module in _BLOCKED_MODULES:
        if f"import {module}" in code or f"from {module}" in code:
            return f"Запрещён импорт модуля: {module}"
    if "open(" in code and ("w" in code or "a" in code):
        return "Запись в файлы запрещена"
    if "exec(" in code or "eval(" in code:
        return "exec/eval запрещены"
    if "__import__" in code:
        return "__import__ запрещён"
    return None


class PythonExecTool(BaseTool):
    """Выполнение Python-кода для вычислений и обработки данных."""

    name = "python"
    description = (
        "Выполнить Python-код для математических вычислений, "
        "обработки данных, генерации таблиц. Используй для точных "
        "расчётов вместо приблизительных ответов. "
        "Код выполняется в песочнице с ограничениями."
    )
    params = [
        ToolParam("code", "Python-код для выполнения. Результат print() будет возвращён.", "string", True),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        if not code.strip():
            return ToolResult(success=False, data="", error="Пустой код")

        # Проверка безопасности
        safety_error = _check_code_safety(code)
        if safety_error:
            return ToolResult(success=False, data="", error=safety_error)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_code, code
            )
            return result
        except Exception as e:
            return ToolResult(success=False, data="", error=str(e))

    @staticmethod
    def _run_code(code: str) -> ToolResult:
        """Запустить код в отдельном процессе."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                cwd=tempfile.gettempdir(),
            )

            stdout = result.stdout[:_MAX_OUTPUT] if result.stdout else ""
            stderr = result.stderr[:_MAX_OUTPUT] if result.stderr else ""

            if result.returncode == 0:
                output = stdout.strip() or "(код выполнен, но ничего не напечатано)"
                return ToolResult(success=True, data=output)
            else:
                return ToolResult(
                    success=False,
                    data=stdout.strip(),
                    error=stderr.strip() or f"Код завершился с ошибкой (код {result.returncode})",
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, data="",
                error=f"Превышен таймаут ({_TIMEOUT} сек)",
            )
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
