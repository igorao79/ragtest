"""Тесты для агентского слоя."""

import pytest

from agent.tool_calc import CalculatorTool, _safe_eval
from agent.tool_python import PythonExecTool, _check_code_safety
from agent.tools import ToolParam, ToolRegistry, ToolResult


# === Calculator Tests ===


class TestSafeEval:
    def test_basic_arithmetic(self):
        assert _safe_eval("2 + 3") == 5
        assert _safe_eval("10 - 4") == 6
        assert _safe_eval("6 * 7") == 42
        assert _safe_eval("15 / 3") == 5.0

    def test_power(self):
        assert _safe_eval("2 ** 10") == 1024

    def test_functions(self):
        assert _safe_eval("abs(-5)") == 5
        assert _safe_eval("round(3.7)") == 4
        assert _safe_eval("max(1, 5, 3)") == 5

    def test_math_functions(self):
        import math
        assert _safe_eval("sqrt(144)") == 12.0
        assert abs(_safe_eval("sin(0)")) < 0.001
        assert _safe_eval("factorial(5)") == 120

    def test_constants(self):
        import math
        assert abs(_safe_eval("pi") - math.pi) < 0.001
        assert abs(_safe_eval("e") - math.e) < 0.001

    def test_complex_expression(self):
        result = _safe_eval("sqrt(144) + 2 ** 3")
        assert result == 20.0

    def test_reject_huge_power(self):
        with pytest.raises(ValueError, match="слишком большая"):
            _safe_eval("2 ** 10000")

    def test_reject_unknown_function(self):
        with pytest.raises(ValueError):
            _safe_eval("exec('print(1)')")


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 2")
        assert result.success
        assert result.data == "4"

    @pytest.mark.asyncio
    async def test_empty(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_invalid(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="not_a_number")
        assert not result.success

    def test_schema(self):
        tool = CalculatorTool()
        schema = tool.to_schema()
        assert schema["name"] == "calculator"
        assert "expression" in schema["parameters"]


# === Python Exec Tests ===


class TestCodeSafety:
    def test_safe_code(self):
        assert _check_code_safety("print(2 + 2)") is None
        assert _check_code_safety("x = [1,2,3]\nprint(sum(x))") is None

    def test_blocked_import_os(self):
        err = _check_code_safety("import os")
        assert err is not None
        assert "os" in err

    def test_blocked_import_subprocess(self):
        err = _check_code_safety("import subprocess")
        assert err is not None

    def test_blocked_exec(self):
        err = _check_code_safety("exec('print(1)')")
        assert err is not None

    def test_blocked_dunder_import(self):
        err = _check_code_safety("__import__('os')")
        assert err is not None

    def test_allowed_math(self):
        assert _check_code_safety("import math\nprint(math.factorial(10))") is None


class TestPythonExecTool:
    @pytest.mark.asyncio
    async def test_simple_print(self):
        tool = PythonExecTool()
        result = await tool.execute(code="print(2 + 2)")
        assert result.success
        assert "4" in result.data

    @pytest.mark.asyncio
    async def test_empty_code(self):
        tool = PythonExecTool()
        result = await tool.execute(code="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocked_import(self):
        tool = PythonExecTool()
        result = await tool.execute(code="import os\nprint(os.getcwd())")
        assert not result.success

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        tool = PythonExecTool()
        result = await tool.execute(code="print(")
        assert not result.success


# === Tool Registry Tests ===


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.get("calculator") is tool
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(PythonExecTool())
        assert len(registry.list_tools()) == 2

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "calculator"

    def test_get_tools_prompt(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        prompt = registry.get_tools_prompt()
        assert "calculator" in prompt


# === Router Parse Tests ===


class TestRouterParsing:
    def _make_router(self):
        from agent.router import AgentRouter
        from agent.tool_calc import CalculatorTool
        from agent.tool_web import WebSearchTool
        from agent.tool_weather import WeatherTool
        from agent.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        # WebSearchTool and WeatherTool need pipeline, use mock
        class FakeWebSearch(WebSearchTool):
            def __init__(self):
                self.name = "web_search"
                self.description = "search"
                self.params = WebSearchTool.params
        class FakeWeather(WeatherTool):
            def __init__(self):
                self.name = "weather"
                self.description = "weather"
                self.params = WeatherTool.params

        registry.register(FakeWebSearch())
        registry.register(FakeWeather())

        router = AgentRouter.__new__(AgentRouter)
        router.registry = registry
        return router

    def test_parse_json_tool_call(self):
        router = self._make_router()
        result = router._parse_tool_call('{"tool": "calculator", "args": {"expression": "2+2"}}')
        assert result is not None
        assert result["tool"] == "calculator"
        assert result["args"]["expression"] == "2+2"

    def test_parse_plain_text(self):
        router = self._make_router()
        result = router._parse_tool_call("Это обычный текст, без инструмента.")
        assert result is None

    def test_parse_json_in_text(self):
        router = self._make_router()
        result = router._parse_tool_call(
            'Я думаю нужно посчитать: {"tool": "calculator", "args": {"expression": "5*5"}}'
        )
        assert result is not None
        assert result["tool"] == "calculator"

    def test_parse_markdown_wrapped(self):
        router = self._make_router()
        result = router._parse_tool_call(
            '```json\n{"tool": "weather", "args": {"city": "Moscow"}}\n```'
        )
        assert result is not None
        assert result["tool"] == "weather"

    def test_parse_empty(self):
        router = self._make_router()
        assert router._parse_tool_call("") is None

    def test_parse_colon_format(self):
        """LLM пишет web_search: "запрос" вместо JSON."""
        router = self._make_router()
        result = router._parse_tool_call('web_search: "выборы губернатора Тула 2026"')
        assert result is not None
        assert result["tool"] == "web_search"
        assert result["args"]["query"] == "выборы губернатора Тула 2026"

    def test_parse_paren_format(self):
        """LLM пишет calculator("2+2") вместо JSON."""
        router = self._make_router()
        result = router._parse_tool_call('calculator("2+2")')
        assert result is not None
        assert result["tool"] == "calculator"
        assert result["args"]["expression"] == "2+2"

    def test_parse_colon_single_quotes(self):
        router = self._make_router()
        result = router._parse_tool_call("weather: 'Moscow'")
        assert result is not None
        assert result["tool"] == "weather"
        assert result["args"]["city"] == "Moscow"
