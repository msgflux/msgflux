# Weather Retrievers

## ✦₊⁺ Overview

Weather retrievers fetch current, forecast, or historical weather data through
`mf.Retriever.weather(...)`. The built-in `open_meteo` provider uses Open-Meteo
for geocoding, forecast data, and historical observations.

Use the built-in `Weather` tool when an agent needs weather access. Use the
retriever directly when you want structured weather data in application code.

---

## 1. **Open-Meteo Weather**

The `open_meteo` retriever accepts a simple city/place name or coordinates and
returns a structured `dotdict`.

```python
import msgflux as mf

retriever = mf.Retriever.weather("open_meteo")

current = retriever("Fortaleza")
forecast = retriever("Fortaleza", when="+6h")
historical = retriever("-3.71722,-38.54306", when="-3d")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_future_days` | `7` | Furthest future forecast allowed before clamping |
| `max_past_days` | `90` | Furthest historical observation allowed before clamping |
| `forecast_hours_when_now` | `6` | Number of upcoming hourly items included for current/future calls |
| `include_forecast_when_now` | `True` | Whether current weather calls include next-hour forecast data |
| `enable_cache` | `True` | Cache geocoding, forecast, and historical responses |
| `timeout` | `15.0` | HTTP timeout in seconds |

### Tool Arguments

The retriever call accepts:

- **`location`**: A simple city/place name such as `"Fortaleza"`. Coordinates
  like `"-3.71722,-38.54306"` are also accepted when already available.
- **`when`**: `"now"`, a relative time like `"+6h"` or `"-3d"`, or an ISO
  datetime.

---

## 2. **Response Format**

Weather retrievers return a structured `dotdict`, so fields can be accessed by
key or attribute:

```python
{
    "location": {
        "input": "Fortaleza",
        "name": "Fortaleza, Ceará, Brazil",
        "latitude": -3.71722,
        "longitude": -38.54306,
        "source": "geocoding",
    },
    "when": {
        "input": "now",
        "target": "2026-05-06T12:30:00+00:00",
        "kind": "now",
        "clamped": False,
    },
    "weather": {
        "time": "2026-05-06T09:30",
        "temperature_c": 30.6,
        "apparent_temperature_c": 34.9,
        "relative_humidity_percent": 58,
        "condition": "overcast",
        "weather_code": 3,
        "precipitation_mm": 0.0,
        "rain_mm": 0.0,
        "is_raining": False,
        "cloud_cover_percent": 100,
        "wind_speed_kmh": 13.8,
        "wind_direction_degrees": 123,
    },
    "forecast": [...],
    "source": {
        "provider": "open-meteo",
        "endpoint": "forecast",
    },
    "units": {...},
}
```

---

## 3. **Examples**

=== "Current"

    ```python
    import msgflux as mf

    weather = mf.Retriever.weather("open_meteo")

    response = weather("Fortaleza")

    print(response["weather"]["temperature_c"])
    print(response.weather.temperature_c)
    print(response["weather"]["is_raining"])
    ```

=== "Forecast"

    ```python
    import msgflux as mf

    weather = mf.Retriever.weather(
        "open_meteo",
        max_future_days=7,
        forecast_hours_when_now=3,
    )

    response = weather("Fortaleza", when="+6h")

    print(response["weather"]["condition"])
    print(response["forecast"])
    ```

=== "Historical"

    ```python
    import msgflux as mf

    weather = mf.Retriever.weather("open_meteo", max_past_days=90)

    response = weather("Fortaleza", when="-3d")

    print(response["source"]["endpoint"])  # "archive"
    print(response["weather"]["temperature_c"])
    ```

=== "Coordinates"

    ```python
    import msgflux as mf

    weather = mf.Retriever.weather("open_meteo")

    response = weather("-3.71722,-38.54306")

    print(response["location"]["source"])  # "coordinates"
    ```

=== "Async"

    ```python
    import msgflux as mf

    weather = mf.Retriever.weather("open_meteo")

    response = await weather.acall("Fortaleza", when="+6h")

    print(response["weather"]["temperature_c"])
    ```

---

## 4. **Agent Tool**

The built-in `Weather` tool is a thin wrapper around this retriever:

```python
from msgflux.tools.builtin import Weather

tool = Weather(engine="open_meteo")
```

The tool exposes only `location` and `when` to the agent, while provider
configuration stays in the constructor.

If `engine` is not passed, `Weather` reads `MSGFLUX_TOOL_WEATHER_ENGINE`.
When neither is set, it defaults to `open_meteo`.
