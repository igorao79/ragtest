"""Инструмент: погода через бесплатный API Open-Meteo."""

import logging
from typing import Any

import httpx

from agent.tools import BaseTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# Open-Meteo — полностью бесплатный, без API-ключа
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "ясно ☀️", 1: "преимущественно ясно 🌤", 2: "переменная облачность ⛅",
    3: "пасмурно ☁️", 45: "туман 🌫", 48: "туман с инеем 🌫",
    51: "лёгкая морось 🌦", 53: "морось 🌧", 55: "сильная морось 🌧",
    61: "небольшой дождь 🌦", 63: "дождь 🌧", 65: "сильный дождь 🌧",
    71: "небольшой снег 🌨", 73: "снег ❄️", 75: "сильный снег ❄️",
    80: "ливень 🌧", 81: "сильный ливень 🌧", 82: "экстремальный ливень ⛈",
    95: "гроза ⛈", 96: "гроза с градом ⛈", 99: "гроза с крупным градом ⛈",
}


class WeatherTool(BaseTool):
    """Получение текущей погоды в городе."""

    name = "weather"
    description = (
        "Узнать текущую погоду в указанном городе. "
        "Используй когда пользователь спрашивает о погоде, температуре, осадках."
    )
    params = [
        ToolParam("city", "Название города (на любом языке)", "string", True),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        city = kwargs.get("city", "")
        if not city.strip():
            return ToolResult(success=False, data="", error="Город не указан")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 1. Геокодинг — получаем координаты
                geo_resp = await client.get(
                    _GEOCODE_URL,
                    params={"name": city.strip(), "count": 1, "language": "ru"},
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

                results = geo_data.get("results", [])
                if not results:
                    return ToolResult(
                        success=False, data="",
                        error=f"Город «{city}» не найден",
                    )

                loc = results[0]
                lat, lon = loc["latitude"], loc["longitude"]
                city_name = loc.get("name", city)
                country = loc.get("country", "")

                # 2. Погода
                weather_resp = await client.get(
                    _WEATHER_URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                        "timezone": "auto",
                    },
                )
                weather_resp.raise_for_status()
                weather_data = weather_resp.json()

                current = weather_data.get("current", {})
                temp = current.get("temperature_2m", "?")
                humidity = current.get("relative_humidity_2m", "?")
                wind = current.get("wind_speed_10m", "?")
                code = current.get("weather_code", 0)
                condition = _WMO_CODES.get(code, f"код {code}")

                data = (
                    f"Погода в {city_name}, {country}:\n"
                    f"🌡 Температура: {temp}°C\n"
                    f"💧 Влажность: {humidity}%\n"
                    f"💨 Ветер: {wind} км/ч\n"
                    f"🌤 Условия: {condition}"
                )
                return ToolResult(success=True, data=data)

        except httpx.HTTPError as e:
            return ToolResult(success=False, data="", error=f"Ошибка API: {e}")
        except Exception as e:
            logger.error("Ошибка weather tool: %s", e, exc_info=True)
            return ToolResult(success=False, data="", error=str(e))
