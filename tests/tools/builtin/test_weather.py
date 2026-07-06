"""Unit tests for msgflux.tools.builtin.weather."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.providers.open_meteo import ResolvedWhen
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.builtin.weather import WeatherTool


def response(payload):
    mock = MagicMock()
    mock.json.return_value = payload
    return mock


def forecast_payload():
    return {
        "utc_offset_seconds": -10800,
        "current": {
            "time": "2026-05-05T09:00",
            "temperature_2m": 28.0,
            "relative_humidity_2m": 75,
            "apparent_temperature": 31.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "weather_code": 2,
            "cloud_cover": 30,
            "wind_speed_10m": 12.0,
            "wind_direction_10m": 90,
        },
        "current_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": [
                "2026-05-05T09:00",
                "2026-05-05T10:00",
                "2026-05-05T11:00",
            ],
            "temperature_2m": [28.0, 29.0, 30.0],
            "relative_humidity_2m": [75, 72, 70],
            "apparent_temperature": [31.0, 32.0, 33.0],
            "precipitation": [0.0, 0.2, 0.0],
            "rain": [0.0, 0.2, 0.0],
            "weather_code": [2, 61, 1],
            "cloud_cover": [30, 60, 20],
            "wind_speed_10m": [12.0, 13.0, 11.0],
            "wind_direction_10m": [90, 95, 100],
        },
        "hourly_units": {"wind_speed_10m": "km/h"},
    }


def archive_payload():
    return {
        "utc_offset_seconds": -10800,
        "hourly": {
            "time": ["2026-05-04T08:00", "2026-05-04T09:00"],
            "temperature_2m": [25.0, 26.0],
            "relative_humidity_2m": [80, 78],
            "apparent_temperature": [27.0, 28.0],
            "precipitation": [1.2, 0.0],
            "rain": [1.2, 0.0],
            "weather_code": [61, 3],
            "cloud_cover": [90, 85],
            "wind_speed_10m": [10.0, 9.0],
            "wind_direction_10m": [120, 130],
        },
    }


class TestWeatherToolInit:
    def test_name_attribute(self):
        assert WeatherTool.name == "weather"

    def test_defaults(self):
        tool = WeatherTool()

        assert tool.engine == "open_meteo"
        assert tool.retriever.max_future_days == 7
        assert tool.retriever.max_past_days == 90
        assert tool.retriever.forecast_hours_when_now == 6
        assert tool.retriever.include_forecast_when_now is True
        assert tool.retriever.timeout == 15.0

    def test_engine_can_be_loaded_from_env(self, mocker):
        mock_weather = MagicMock()
        factory = mocker.patch(
            "msgflux.tools.builtin.weather.Retriever.weather",
            return_value=mock_weather,
        )
        mocker.patch.dict(
            "os.environ",
            {"MSGFLUX_TOOL_WEATHER_ENGINE": "custom_weather"},
            clear=True,
        )

        tool = WeatherTool(timeout=3)

        assert tool.engine == "custom_weather"
        assert tool.retriever is mock_weather
        factory.assert_called_once_with("custom_weather", timeout=3)

    def test_init_engine_takes_precedence_over_env(self, mocker):
        mock_weather = MagicMock()
        factory = mocker.patch(
            "msgflux.tools.builtin.weather.Retriever.weather",
            return_value=mock_weather,
        )
        mocker.patch.dict(
            "os.environ",
            {"MSGFLUX_TOOL_WEATHER_ENGINE": "env_weather"},
            clear=True,
        )

        tool = WeatherTool(engine="init_weather")

        assert tool.engine == "init_weather"
        factory.assert_called_once_with("init_weather")


class TestWeatherToolCall:
    def test_current_weather_for_city_returns_structured_result(self, mocker):
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            response(
                {
                    "results": [
                        {
                            "name": "Fortaleza",
                            "admin1": "Ceara",
                            "country": "Brazil",
                            "latitude": -3.73,
                            "longitude": -38.52,
                        }
                    ]
                }
            ),
            response(forecast_payload()),
        ]
        mocker.patch(
            "msgflux.data.retrievers.providers.open_meteo.httpx.Client",
            return_value=mock_client,
        )

        result = WeatherTool(forecast_hours_when_now=2)("Fortaleza")

        assert isinstance(result, dotdict)
        assert result["location"] == {
            "input": "Fortaleza",
            "name": "Fortaleza, Ceara, Brazil",
            "latitude": -3.73,
            "longitude": -38.52,
            "source": "geocoding",
        }
        assert result.location.name == "Fortaleza, Ceara, Brazil"
        assert result["when"]["kind"] == "now"
        assert result.when.kind == "now"
        assert result["weather"]["temperature_c"] == 28.0
        assert result.weather.temperature_c == 28.0
        assert result["weather"]["condition"] == "partly cloudy"
        assert result["weather"]["is_raining"] is False
        assert result["forecast"][0]["time"] == "2026-05-05T10:00"
        assert result["forecast"][0]["is_raining"] is True
        assert result["source"] == {"provider": "open-meteo", "endpoint": "forecast"}

    def test_coordinates_skip_geocoding(self, mocker):
        mock_client = MagicMock()
        mock_client.get.return_value = response(forecast_payload())
        mocker.patch(
            "msgflux.data.retrievers.providers.open_meteo.httpx.Client",
            return_value=mock_client,
        )

        result = WeatherTool()("-3.71722;-38.54306")

        assert result["location"]["source"] == "coordinates"
        assert result["location"]["latitude"] == -3.71722
        assert result["location"]["longitude"] == -38.54306
        assert mock_client.get.call_count == 1

    def test_past_weather_uses_archive_endpoint(self, mocker):
        mock_client = MagicMock()
        mock_client.get.return_value = response(archive_payload())
        mocker.patch(
            "msgflux.data.retrievers.providers.open_meteo.httpx.Client",
            return_value=mock_client,
        )

        target = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
        mocker.patch.object(
            type(WeatherTool().retriever),
            "_resolve_when",
            return_value=ResolvedWhen("-1d", target, "past"),
        )

        result = WeatherTool()("-3.71722,-38.54306", when="-1d")

        assert result["source"]["endpoint"] == "archive"
        assert result["weather"]["time"] == "2026-05-04T09:00"
        assert result["weather"]["condition"] == "overcast"
        assert "forecast" not in result
        assert mock_client.get.call_args.kwargs["params"]["start_date"] == "2026-05-04"

    def test_future_weather_uses_closest_hourly_item(self, mocker):
        mock_client = MagicMock()
        mock_client.get.return_value = response(forecast_payload())
        mocker.patch(
            "msgflux.data.retrievers.providers.open_meteo.httpx.Client",
            return_value=mock_client,
        )

        target = datetime(2026, 5, 5, 14, 0, tzinfo=timezone.utc)
        mocker.patch.object(
            type(WeatherTool().retriever),
            "_resolve_when",
            return_value=ResolvedWhen("+2h", target, "future"),
        )

        result = WeatherTool()("-3.71722,-38.54306", when="+2h")

        assert result["weather"]["time"] == "2026-05-05T11:00"
        assert result["weather"]["temperature_c"] == 30.0
        assert result["source"]["endpoint"] == "forecast"

    def test_location_resolution_errors_have_context(self, mocker):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("timeout")
        mocker.patch(
            "msgflux.data.retrievers.providers.open_meteo.httpx.Client",
            return_value=mock_client,
        )

        with pytest.raises(RuntimeError, match="Failed to resolve location"):
            WeatherTool()("Fortaleza")


class TestWeatherToolValidation:
    def test_invalid_location_raises(self):
        with pytest.raises(ValueError, match="location"):
            WeatherTool()("")

    def test_invalid_when_raises(self):
        with pytest.raises(ValueError, match="when must be"):
            WeatherTool().retriever._resolve_when("tomorrow morning")

    def test_out_of_range_coordinates_raise(self):
        with pytest.raises(ValueError, match="Latitude"):
            WeatherTool()("91,0")


class TestWeatherToolLibraryIntegration:
    def test_schema_exposes_location_and_when(self):
        tool = WeatherTool()
        library = ToolLibrary(name="weather", tools=[tool])

        schema = library.get_tool_json_schemas()[0]["function"]
        props = schema["parameters"]["properties"]

        assert schema["name"] == "weather"
        assert "current, forecast, or historical weather" in schema["description"]
        assert "location" in props
        assert "when" in props
        assert "output_mode" not in props

    def test_schema_can_be_generated_from_class(self):
        library = ToolLibrary(name="weather", tools=[WeatherTool])

        schema = library.get_tool_json_schemas()[0]["function"]

        assert schema["name"] == "weather"
        assert "current, forecast, or historical weather" in schema["description"]


@pytest.mark.asyncio
async def test_weather_async_call_with_coordinates(mocker):
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response(forecast_payload()))
    mocker.patch(
        "msgflux.data.retrievers.providers.open_meteo.httpx.AsyncClient",
        return_value=mock_client,
    )

    result = await WeatherTool().acall("-3.71722,-38.54306")

    assert result["location"]["source"] == "coordinates"
    assert result["weather"]["temperature_c"] == 28.0
