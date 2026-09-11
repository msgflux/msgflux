"""Provider-neutral HTTP transport for model requests."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from os import getenv
from typing import Any

import httpx2

from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.model_credentials import ResolvedModelCredentials

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, *range(500, 600)}


class HTTPTransport:
    """Send model HTTP requests with shared lifecycle and failure semantics."""

    def __init__(
        self,
        *,
        client: Any = None,
        async_client: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_retry_delay: float = 8.0,
    ) -> None:
        self._client = client
        self._async_client = async_client
        self._owns_client = client is None
        self._owns_async_client = async_client is None
        self._async_client_loop = None
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_delay = max_retry_delay

    def request(
        self,
        owner: Any,
        endpoint: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
    ) -> Any:
        client = self._get_client()
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            owner._raise_if_aborted()
            try:
                credentials = owner.credential_resolver.resolve(owner)
                response = client.request(
                    method,
                    self._url(owner, endpoint, credentials),
                    **self._request_kwargs(
                        headers=merge_headers(headers or {}, credentials.headers),
                        json=json,
                        data=data,
                        files=files,
                    ),
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    _raise_for_status(response, owner)
                    return response
                if attempt == attempts:
                    _raise_for_status(response, owner)
                    return response
                self._wait(owner, self._retry_delay(attempt, response.headers))
            except httpx2.HTTPStatusError:
                raise
            except httpx2.RequestError:
                if attempt == attempts:
                    raise
                self._wait(owner, self._retry_delay(attempt))
        raise RuntimeError("HTTP retry loop ended without a response")

    async def arequest(
        self,
        owner: Any,
        endpoint: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
    ) -> Any:
        client = self._get_async_client()
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            owner._raise_if_aborted()
            try:
                credentials = await owner.credential_resolver.aresolve(owner)
                response = await client.request(
                    method,
                    self._url(owner, endpoint, credentials),
                    **self._request_kwargs(
                        headers=merge_headers(headers or {}, credentials.headers),
                        json=json,
                        data=data,
                        files=files,
                    ),
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    _raise_for_status(response, owner)
                    return response
                if attempt == attempts:
                    _raise_for_status(response, owner)
                    return response
                await self._await(owner, self._retry_delay(attempt, response.headers))
            except httpx2.HTTPStatusError:
                raise
            except httpx2.RequestError:
                if attempt == attempts:
                    raise
                await self._await(owner, self._retry_delay(attempt))
        raise RuntimeError("HTTP retry loop ended without a response")

    def stream(
        self,
        owner: Any,
        endpoint: str,
        *,
        iterate: Callable[[Any], Iterator[Any]],
        method: str = "POST",
        headers: dict[str, str] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
    ) -> Iterator[Any]:
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            emitted = False
            owner._raise_if_aborted()
            try:
                credentials = owner.credential_resolver.resolve(owner)
                with self._get_client().stream(
                    method,
                    self._url(owner, endpoint, credentials),
                    **self._request_kwargs(
                        headers=merge_headers(headers or {}, credentials.headers),
                        json=json,
                        data=data,
                        files=files,
                    ),
                ) as response:
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and attempt < attempts
                    ):
                        self._wait(
                            owner,
                            self._retry_delay(attempt, response.headers),
                        )
                        continue
                    _raise_for_status(response, owner)
                    for item in iterate(response):
                        owner._raise_if_aborted()
                        emitted = True
                        yield item
                    return
            except httpx2.HTTPStatusError:
                raise
            except httpx2.RequestError:
                if emitted or attempt == attempts:
                    raise
                self._wait(owner, self._retry_delay(attempt))

    async def astream(
        self,
        owner: Any,
        endpoint: str,
        *,
        iterate: Callable[[Any], AsyncIterator[Any]],
        method: str = "POST",
        headers: dict[str, str] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
    ) -> AsyncIterator[Any]:
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            emitted = False
            owner._raise_if_aborted()
            try:
                credentials = await owner.credential_resolver.aresolve(owner)
                async with self._get_async_client().stream(
                    method,
                    self._url(owner, endpoint, credentials),
                    **self._request_kwargs(
                        headers=merge_headers(headers or {}, credentials.headers),
                        json=json,
                        data=data,
                        files=files,
                    ),
                ) as response:
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and attempt < attempts
                    ):
                        await self._await(
                            owner,
                            self._retry_delay(attempt, response.headers),
                        )
                        continue
                    _raise_for_status(response, owner)
                    async for item in iterate(response):
                        owner._raise_if_aborted()
                        emitted = True
                        yield item
                    return
            except httpx2.HTTPStatusError:
                raise
            except httpx2.RequestError:
                if emitted or attempt == attempts:
                    raise
                await self._await(owner, self._retry_delay(attempt))

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None and self._owns_async_client:
            if self._async_client_loop is not asyncio.get_running_loop():
                self._async_client = None
                self._async_client_loop = None
                return
            await self._async_client.aclose()
            self._async_client = None
            self._async_client_loop = None

    @staticmethod
    def _request_kwargs(
        *,
        headers: dict[str, str],
        json: Any,
        data: Any,
        files: Any,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"headers": headers}
        if json is not None:
            kwargs["json"] = json
        if data is not None:
            kwargs["data"] = data
        if files is not None:
            kwargs["files"] = files
        return kwargs

    @staticmethod
    def _url(
        owner: Any,
        endpoint: str,
        credentials: ResolvedModelCredentials,
    ) -> str:
        base_url = credentials.base_url or owner.sampling_params.get("base_url")
        if not base_url:
            base_url = "https://api.openai.com/v1"
        return f"{str(base_url).rstrip('/')}/{endpoint.lstrip('/')}"

    def _get_client(self):
        if self._client is None:
            self._client = httpx2.Client(
                timeout=self._timeout(),
                limits=httpx2.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
                verify=self._verify_ssl(),
            )
        return self._client

    def _get_async_client(self):
        current_loop = asyncio.get_running_loop()
        if self._async_client is None or (
            self._owns_async_client and self._async_client_loop is not current_loop
        ):
            self._async_client = httpx2.AsyncClient(
                timeout=self._timeout(),
                limits=httpx2.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
                verify=self._verify_ssl(),
            )
            self._async_client_loop = current_loop
        return self._async_client

    def _timeout(self) -> float | None:
        if self.timeout is not None:
            return self.timeout
        value = getenv("OPENAI_TIMEOUT")
        return float(value) if value else None

    def _max_retries(self) -> int:
        if self.max_retries is not None:
            return max(0, self.max_retries)
        value = getenv("OPENAI_MAX_RETRIES")
        return max(0, int(value)) if value else 2

    @staticmethod
    def _verify_ssl() -> bool:
        return getenv("OPENAI_SSL_VERIFY", "true").lower() not in {
            "0",
            "false",
            "no",
        }

    def _retry_delay(self, attempt: int, headers: Any = None) -> float:
        retry_after = _retry_after_seconds(headers)
        if retry_after is not None:
            return min(retry_after, self.max_retry_delay)
        base = min(0.25 * (2**attempt), self.max_retry_delay)
        jitter = random.uniform(0, base * 0.25)  # noqa: S311 - retry jitter
        return min(base + jitter, self.max_retry_delay)

    @staticmethod
    def _wait(owner: Any, delay: float) -> None:
        deadline = time.monotonic() + delay
        while True:
            owner._raise_if_aborted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))

    @staticmethod
    async def _await(owner: Any, delay: float) -> None:
        deadline = time.monotonic() + delay
        while True:
            owner._raise_if_aborted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.05))


def merge_headers(*values: dict[str, str]) -> dict[str, str]:
    """Merge headers case-insensitively, giving later values precedence."""
    merged: dict[str, str] = {}
    names: dict[str, str] = {}
    for value in values:
        for key, item in value.items():
            normalized = key.casefold()
            previous = names.get(normalized)
            if previous is not None:
                merged.pop(previous)
            names[normalized] = key
            merged[key] = item
    return merged


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _raise_for_status(response: Any, owner: Any) -> None:
    try:
        response.raise_for_status()
    except httpx2.HTTPStatusError as exc:
        error = _structured_error(response)
        raise ModelProviderHTTPError(
            status_code=response.status_code,
            description=error["description"],
            provider=getattr(owner, "provider", None),
            model_id=getattr(owner, "model_id", None),
            code=error.get("code"),
            error_type=error.get("type"),
            param=error.get("param"),
            request_id=_request_id(response.headers),
            response=response,
        ) from exc


def _structured_error(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        description = error.get("message") or error.get("detail")
        code = error.get("code")
        error_type = error.get("type") or error.get("status")
        param = error.get("param")
    elif isinstance(error, str):
        description = error
        code = None
        error_type = None
        param = None
    elif isinstance(payload, dict):
        description = payload.get("message") or payload.get("detail")
        code = payload.get("code")
        error_type = payload.get("type") or payload.get("status")
        param = payload.get("param")
    else:
        description = None
        code = None
        error_type = None
        param = None
    if not isinstance(description, str) or not description:
        description = getattr(response, "reason_phrase", None) or "Request failed"
    return {
        "description": description[:4000],
        "code": code,
        "type": error_type if isinstance(error_type, str) else None,
        "param": param if isinstance(param, str) else None,
    }


def _request_id(headers: Any) -> str | None:
    for name in ("x-request-id", "request-id", "x-groq-request-id", "cf-ray"):
        value = headers.get(name)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = ["HTTPTransport", "merge_headers"]
