"""Direct HTTP transport for provider chat APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from msgflux.models.chat_api import ChatTransport, PreparedChatRequest
from msgflux.models.http_transport import HTTPTransport, merge_headers
from msgflux.models.sse import aiter_sse_json, iter_sse_json


class HTTPChatTransport(ChatTransport):
    """Adapt prepared chat requests to the shared model HTTP transport."""

    def __init__(
        self,
        *,
        client: Any = None,
        async_client: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_retry_delay: float = 8.0,
    ) -> None:
        self.http = HTTPTransport(
            client=client,
            async_client=async_client,
            timeout=timeout,
            max_retries=max_retries,
            max_retry_delay=max_retry_delay,
        )

    def create(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("stream"):
            return self._stream(owner, request)
        response = self.http.request(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
        )
        return owner.api_adapter.decode_response(response.json())

    async def acreate(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("stream"):
            return self._astream(owner, request)
        response = await self.http.arequest(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
        )
        return owner.api_adapter.decode_response(response.json())

    def close(self, _owner: Any) -> None:
        self.http.close()

    async def aclose(self, _owner: Any) -> None:
        await self.http.aclose()

    def _stream(self, owner: Any, request: PreparedChatRequest) -> Iterator[Any]:
        for payload in self.http.stream(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
            iterate=lambda response: iter_sse_json(response.iter_lines()),
        ):
            yield owner.api_adapter.decode_stream_event(payload)

    async def _astream(
        self,
        owner: Any,
        request: PreparedChatRequest,
    ) -> AsyncIterator[Any]:
        async for payload in self.http.astream(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
            iterate=lambda response: aiter_sse_json(response.aiter_lines()),
        ):
            yield owner.api_adapter.decode_stream_event(payload)

    @staticmethod
    def _headers(request: PreparedChatRequest) -> dict[str, str]:
        return merge_headers(
            {"accept": "application/json", "content-type": "application/json"},
            request.headers,
        )
