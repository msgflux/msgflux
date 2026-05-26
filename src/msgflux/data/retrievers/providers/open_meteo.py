import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import httpx

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WeatherRetriever

WhenKind = Literal["now", "future", "past"]
LocationSource = Literal["coordinates", "geocoding"]


@dataclass(frozen=True)
class ResolvedLocation:
    input: str
    latitude: float
    longitude: float
    name: str | None
    source: LocationSource


@dataclass(frozen=True)
class ResolvedWhen:
    input: str
    target: datetime
    kind: WhenKind
    clamped: bool = False


@dataclass
class CacheEntry:
    value: Any
    expires_at: datetime


class TTLCache:
    def __init__(self, *, default_ttl_seconds: int = 3600):
        self.default_ttl_seconds = default_ttl_seconds
        self._data: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None

        if entry.expires_at <= datetime.now(timezone.utc):
            self._data.pop(key, None)
            return None

        return entry.value

    def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._data[key] = CacheEntry(
            value=value,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
        )

    def clear(self) -> None:
        self._data.clear()


@register_retriever
class OpenMeteoWeatherRetriever(BaseRetriever, WeatherRetriever):
    provider = "open_meteo"
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    coord_re = re.compile(
        r"^\s*(?P<lat>-?\d+(?:\.\d+)?)\s*[,;]\s*(?P<lon>-?\d+(?:\.\d+)?)\s*$"
    )
    relative_re = re.compile(r"^(?P<sign>[+-])(?P<value>\d+)(?P<unit>h|d)$")

    weather_code_map = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snow fall",
        73: "moderate snow fall",
        75: "heavy snow fall",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }

    current_fields = (
        "temperature_2m,relative_humidity_2m,apparent_temperature,"
        "precipitation,rain,weather_code,cloud_cover,wind_speed_10m,"
        "wind_direction_10m"
    )
    hourly_fields = current_fields

    def __init__(
        self,
        *,
        max_future_days: int = 7,
        max_past_days: int = 90,
        forecast_hours_when_now: int = 6,
        include_forecast_when_now: bool = True,
        enable_cache: bool = True,
        geocoding_cache_ttl_seconds: int = 60 * 60 * 24 * 30,
        historical_cache_ttl_seconds: int = 60 * 60 * 24 * 30,
        forecast_cache_ttl_seconds: int = 60 * 5,
        timeout: float = 15.0,
    ):
        self.max_future_days = max_future_days
        self.max_past_days = max_past_days
        self.forecast_hours_when_now = forecast_hours_when_now
        self.include_forecast_when_now = include_forecast_when_now
        self.enable_cache = enable_cache
        self.geocoding_cache_ttl_seconds = geocoding_cache_ttl_seconds
        self.historical_cache_ttl_seconds = historical_cache_ttl_seconds
        self.forecast_cache_ttl_seconds = forecast_cache_ttl_seconds
        self.timeout = timeout
        self._cache = TTLCache()
        self.client = httpx.Client(timeout=self.timeout)
        self.async_client = httpx.AsyncClient(timeout=self.timeout)

    def __call__(self, location: str, when: str = "now") -> dict[str, Any]:
        return self._call_with_client(self.client, location=location, when=when)

    async def acall(self, location: str, when: str = "now") -> dict[str, Any]:
        loc = await self._aresolve_location(self.async_client, location)
        resolved_when = self._resolve_when(when)
        if resolved_when.kind in {"now", "future"}:
            raw = await self._afetch_forecast(self.async_client, loc, resolved_when)
        else:
            raw = await self._afetch_historical(self.async_client, loc, resolved_when)
        return self._result(raw, loc, resolved_when)

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.async_client.aclose()

    def _call_with_client(
        self,
        client: httpx.Client,
        *,
        location: str,
        when: str,
    ) -> dict[str, Any]:
        loc = self._resolve_location(client, location)
        resolved_when = self._resolve_when(when)
        if resolved_when.kind in {"now", "future"}:
            raw = self._fetch_forecast(client, loc, resolved_when)
        else:
            raw = self._fetch_historical(client, loc, resolved_when)
        return self._result(raw, loc, resolved_when)

    def _resolve_location(
        self, client: httpx.Client, location: str
    ) -> ResolvedLocation:
        location = self._validate_text(location, "location")
        coord = self.coord_re.match(location)
        if coord:
            return self._coordinates_location(location, coord)

        cached = self._cache_get(f"geocode:{location.casefold()}")
        if cached is not None:
            return cached

        try:
            response = client.get(
                self.geocoding_url,
                params={
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to resolve location {location!r}: {exc}"
            ) from exc

        return self._location_from_geocoding(location, response.json())

    async def _aresolve_location(
        self,
        client: httpx.AsyncClient,
        location: str,
    ) -> ResolvedLocation:
        location = self._validate_text(location, "location")
        coord = self.coord_re.match(location)
        if coord:
            return self._coordinates_location(location, coord)

        cached = self._cache_get(f"geocode:{location.casefold()}")
        if cached is not None:
            return cached

        try:
            response = await client.get(
                self.geocoding_url,
                params={
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to resolve location {location!r}: {exc}"
            ) from exc

        return self._location_from_geocoding(location, response.json())

    def _coordinates_location(
        self, location: str, coord: re.Match[str]
    ) -> ResolvedLocation:
        lat = float(coord.group("lat"))
        lon = float(coord.group("lon"))
        if not -90 <= lat <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not -180 <= lon <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        return ResolvedLocation(
            input=location,
            latitude=lat,
            longitude=lon,
            name=None,
            source="coordinates",
        )

    def _location_from_geocoding(
        self,
        location: str,
        data: dict[str, Any],
    ) -> ResolvedLocation:
        results = data.get("results") or []
        if not results:
            raise ValueError(f"Could not resolve location: {location!r}")

        item = results[0]
        name = ", ".join(
            part
            for part in [item.get("name"), item.get("admin1"), item.get("country")]
            if part
        )
        resolved = ResolvedLocation(
            input=location,
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            name=name or None,
            source="geocoding",
        )
        self._cache_set(
            f"geocode:{location.casefold()}",
            resolved,
            ttl_seconds=self.geocoding_cache_ttl_seconds,
        )
        return resolved

    def _resolve_when(self, when: str) -> ResolvedWhen:
        when = self._validate_text(when, "when")
        now = datetime.now(timezone.utc)
        if when == "now":
            return ResolvedWhen(input=when, target=now, kind="now")

        relative = self.relative_re.match(when)
        if relative:
            sign = relative.group("sign")
            value = int(relative.group("value"))
            unit = relative.group("unit")
            delta = timedelta(hours=value) if unit == "h" else timedelta(days=value)
            return self._clamped_when(
                when, now + delta if sign == "+" else now - delta, now
            )

        try:
            parsed = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "when must be 'now', a relative expression like '+6h' or '-7d', "
                "or an ISO datetime."
            ) from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return self._clamped_when(when, parsed.astimezone(timezone.utc), now)

    def _clamped_when(self, raw: str, target: datetime, now: datetime) -> ResolvedWhen:
        if target >= now:
            max_target = now + timedelta(days=self.max_future_days)
            return ResolvedWhen(
                input=raw,
                target=min(target, max_target),
                kind="future",
                clamped=target > max_target,
            )

        min_target = now - timedelta(days=self.max_past_days)
        return ResolvedWhen(
            input=raw,
            target=max(target, min_target),
            kind="past",
            clamped=target < min_target,
        )

    def _fetch_forecast(
        self,
        client: httpx.Client,
        loc: ResolvedLocation,
        when: ResolvedWhen,
    ) -> dict[str, Any]:
        cache_key = self._forecast_cache_key(loc, when)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            response = client.get(self.forecast_url, params=self._forecast_params(loc))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch weather forecast: {exc}") from exc

        data = response.json()
        data["_tool"] = {"endpoint": "forecast", "cache_key": cache_key}
        self._cache_set(cache_key, data, ttl_seconds=self.forecast_cache_ttl_seconds)
        return data

    async def _afetch_forecast(
        self,
        client: httpx.AsyncClient,
        loc: ResolvedLocation,
        when: ResolvedWhen,
    ) -> dict[str, Any]:
        cache_key = self._forecast_cache_key(loc, when)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            response = await client.get(
                self.forecast_url, params=self._forecast_params(loc)
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch weather forecast: {exc}") from exc

        data = response.json()
        data["_tool"] = {"endpoint": "forecast", "cache_key": cache_key}
        self._cache_set(cache_key, data, ttl_seconds=self.forecast_cache_ttl_seconds)
        return data

    def _fetch_historical(
        self,
        client: httpx.Client,
        loc: ResolvedLocation,
        when: ResolvedWhen,
    ) -> dict[str, Any]:
        cache_key = self._historical_cache_key(loc, when)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            response = client.get(
                self.archive_url, params=self._historical_params(loc, when)
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch historical weather: {exc}") from exc

        data = response.json()
        data["_tool"] = {"endpoint": "archive", "cache_key": cache_key}
        self._cache_set(cache_key, data, ttl_seconds=self.historical_cache_ttl_seconds)
        return data

    async def _afetch_historical(
        self,
        client: httpx.AsyncClient,
        loc: ResolvedLocation,
        when: ResolvedWhen,
    ) -> dict[str, Any]:
        cache_key = self._historical_cache_key(loc, when)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            response = await client.get(
                self.archive_url,
                params=self._historical_params(loc, when),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch historical weather: {exc}") from exc

        data = response.json()
        data["_tool"] = {"endpoint": "archive", "cache_key": cache_key}
        self._cache_set(cache_key, data, ttl_seconds=self.historical_cache_ttl_seconds)
        return data

    def _forecast_params(self, loc: ResolvedLocation) -> dict[str, Any]:
        return {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "timezone": "auto",
            "current": self.current_fields,
            "hourly": self.hourly_fields,
            "forecast_days": self.max_future_days,
        }

    def _historical_params(
        self,
        loc: ResolvedLocation,
        when: ResolvedWhen,
    ) -> dict[str, Any]:
        date = when.target.date().isoformat()
        return {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "timezone": "auto",
            "start_date": date,
            "end_date": date,
            "hourly": self.hourly_fields,
        }

    def _result(
        self,
        raw: dict[str, Any],
        loc: ResolvedLocation,
        when: ResolvedWhen,
    ) -> dotdict:
        if raw["_tool"]["endpoint"] == "forecast" and when.kind == "now":
            weather = self._weather_item(raw.get("current") or {})
        else:
            weather = self._closest_hourly(raw, when.target)

        result: dict[str, Any] = {
            "location": self._location_dict(loc),
            "when": self._when_dict(when),
            "weather": weather,
            "source": {
                "provider": "open-meteo",
                "endpoint": raw["_tool"]["endpoint"],
            },
            "units": self._units(raw),
        }

        if raw["_tool"]["endpoint"] == "forecast" and (
            when.kind == "future" or self.include_forecast_when_now
        ):
            result["forecast"] = self._next_hours(
                raw,
                hours=self.forecast_hours_when_now,
                after=weather.get("time"),
            )

        return dotdict(result)

    def _weather_item(self, item: dict[str, Any]) -> dict[str, Any]:
        code = item.get("weather_code")
        return {
            "time": item.get("time"),
            "temperature_c": item.get("temperature_2m"),
            "apparent_temperature_c": item.get("apparent_temperature"),
            "relative_humidity_percent": item.get("relative_humidity_2m"),
            "condition": self._condition_name(code),
            "weather_code": code,
            "precipitation_mm": item.get("precipitation"),
            "rain_mm": item.get("rain"),
            "is_raining": self._is_raining(item),
            "cloud_cover_percent": item.get("cloud_cover"),
            "wind_speed_kmh": item.get("wind_speed_10m"),
            "wind_direction_degrees": item.get("wind_direction_10m"),
        }

    def _closest_hourly(self, raw: dict[str, Any], target: datetime) -> dict[str, Any]:
        hourly = raw.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return self._weather_item({})

        target_local = self._to_response_local_time(raw, target)
        index = min(
            range(len(times)),
            key=lambda i: abs(datetime.fromisoformat(times[i]) - target_local),
        )
        return self._weather_item(self._hourly_at(hourly, times, index))

    def _next_hours(
        self,
        raw: dict[str, Any],
        *,
        hours: int,
        after: str | None,
    ) -> list[dict[str, Any]]:
        hourly = raw.get("hourly") or {}
        times = hourly.get("time") or []
        if not times or hours <= 0:
            return []

        start = 0
        if after:
            after_dt = datetime.fromisoformat(after)
            start = next(
                (
                    i
                    for i, value in enumerate(times)
                    if datetime.fromisoformat(value) > after_dt
                ),
                len(times),
            )

        end = min(start + hours, len(times))
        return [
            self._weather_item(self._hourly_at(hourly, times, i))
            for i in range(start, end)
        ]

    def _hourly_at(
        self,
        hourly: dict[str, Any],
        times: list[str],
        index: int,
    ) -> dict[str, Any]:
        return {
            "time": times[index],
            "temperature_2m": self._safe_get(hourly, "temperature_2m", index),
            "relative_humidity_2m": self._safe_get(
                hourly, "relative_humidity_2m", index
            ),
            "apparent_temperature": self._safe_get(
                hourly, "apparent_temperature", index
            ),
            "precipitation": self._safe_get(hourly, "precipitation", index),
            "rain": self._safe_get(hourly, "rain", index),
            "weather_code": self._safe_get(hourly, "weather_code", index),
            "cloud_cover": self._safe_get(hourly, "cloud_cover", index),
            "wind_speed_10m": self._safe_get(hourly, "wind_speed_10m", index),
            "wind_direction_10m": self._safe_get(hourly, "wind_direction_10m", index),
        }

    def _to_response_local_time(
        self, raw: dict[str, Any], target: datetime
    ) -> datetime:
        offset_seconds = raw.get("utc_offset_seconds")
        if isinstance(offset_seconds, int):
            return (target + timedelta(seconds=offset_seconds)).replace(tzinfo=None)
        return target.replace(tzinfo=None)

    def _units(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "current": raw.get("current_units") or {},
            "hourly": raw.get("hourly_units") or {},
        }

    def _is_raining(self, item: dict[str, Any]) -> bool | None:
        precipitation = item.get("precipitation")
        rain = item.get("rain")
        if precipitation is None and rain is None:
            return None
        return (precipitation or 0) > 0 or (rain or 0) > 0

    def _condition_name(self, code: Any) -> str | None:
        if code is None:
            return None
        try:
            return self.weather_code_map.get(int(code), f"weather code {code}")
        except (TypeError, ValueError):
            return f"weather code {code}"

    def _location_dict(self, loc: ResolvedLocation) -> dict[str, Any]:
        return {
            "input": loc.input,
            "name": loc.name,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "source": loc.source,
        }

    def _when_dict(self, when: ResolvedWhen) -> dict[str, Any]:
        return {
            "input": when.input,
            "target": when.target.isoformat(),
            "kind": when.kind,
            "clamped": when.clamped,
        }

    def _forecast_cache_key(self, loc: ResolvedLocation, when: ResolvedWhen) -> str:
        return (
            f"forecast:{loc.latitude:.5f}:{loc.longitude:.5f}:"
            f"{when.target.strftime('%Y-%m-%dT%H')}:"
            f"{self.max_future_days}:{self.forecast_hours_when_now}"
        )

    def _historical_cache_key(self, loc: ResolvedLocation, when: ResolvedWhen) -> str:
        return f"historical:{loc.latitude:.5f}:{loc.longitude:.5f}:{when.target.date()}"

    def _safe_get(self, data: dict[str, Any], key: str, index: int) -> Any:
        values = data.get(key)
        if values is None or index >= len(values):
            return None
        return values[index]

    def _cache_get(self, key: str) -> Any | None:
        if not self.enable_cache:
            return None
        return self._cache.get(key)

    def _cache_set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        if self.enable_cache:
            self._cache.set(key, value, ttl_seconds=ttl_seconds)

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _validate_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{name}` must be a non-empty string.")
        return value.strip()
