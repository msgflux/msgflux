from os import getenv
from typing import Any

from msgflux.data.retrievers.retriever import Retriever


class Weather:
    name = "weather"
    display_name = "Weather"
    engine_env_var = "MSGFLUX_TOOL_WEATHER_ENGINE"
    default_engine = "open_meteo"
    description = """
    Get current, forecast, or historical weather data for a location.

    Use `location` as a simple city or place name, such as "Fortaleza".
    Coordinates like "-3.71722,-38.54306" are also accepted when the user
    provides them. Use `when` as "now", a relative time like "+6h" or "-3d",
    or an ISO datetime.
    """

    def __init__(self, engine: str | None = None, **params: Any):
        self.engine = engine or getenv(self.engine_env_var) or self.default_engine
        self.params = dict(params)
        self.retriever = Retriever.weather(self.engine, **self.params)

    def __call__(self, location: str, when: str = "now") -> dict[str, Any]:
        return self.retriever(location=location, when=when)

    async def acall(self, location: str, when: str = "now") -> dict[str, Any]:
        return await self.retriever.acall(location=location, when=when)
