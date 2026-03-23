"""Инструмент: калькулятор для точных вычислений."""

import ast
import math
import operator
from typing import Any

from agent.tools import BaseTool, ToolParam, ToolResult

# Разрешённые операции
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Разрешённые математические функции
_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(expr: str) -> float | int:
    """Безопасное вычисление математического выражения."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Недопустимая константа: {node.value}")
        elif isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Недопустимая операция: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            # Защита от огромных степеней
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and right > 1000:
                raise ValueError("Степень слишком большая (макс 1000)")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Недопустимая операция: {type(node.op).__name__}")
            return op(_eval(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                func = _SAFE_FUNCS.get(func_name)
                if func is None:
                    raise ValueError(f"Недопустимая функция: {func_name}")
                args = [_eval(a) for a in node.args]
                if callable(func):
                    return func(*args)
                return func  # константы (pi, e)
            raise ValueError("Вложенные вызовы не поддерживаются")
        elif isinstance(node, ast.Name):
            val = _SAFE_FUNCS.get(node.id)
            if val is None:
                raise ValueError(f"Неизвестная переменная: {node.id}")
            return val
        else:
            raise ValueError(f"Недопустимый узел: {type(node).__name__}")

    return _eval(tree)


class CalculatorTool(BaseTool):
    """Калькулятор для точных математических вычислений."""

    name = "calculator"
    description = (
        "Точный калькулятор для математических вычислений. "
        "Поддерживает +, -, *, /, **, sqrt, sin, cos, log, factorial и др. "
        "Используй для любых числовых расчётов, чтобы не галлюцинировать."
    )
    params = [
        ToolParam(
            "expression",
            "Математическое выражение, например: sqrt(144) + 2**10 или factorial(20)",
            "string",
            True,
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get("expression", "")
        if not expression.strip():
            return ToolResult(success=False, data="", error="Пустое выражение")

        try:
            result = _safe_eval(expression.strip())
            return ToolResult(success=True, data=str(result))
        except Exception as e:
            return ToolResult(success=False, data="", error=f"Ошибка вычисления: {e}")
