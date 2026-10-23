"""Security regression: owned service transport must not use ambient proxies."""

import asyncio
import json

import httpx2
import pytest

from msgflux.runtime.service.http.client import AgentServiceClient


class _Recorder:
    """Loopback-only HTTP endpoint that records request headers."""

    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode()
        self.requests: list[dict[str, str]] = []
        self._server: asyncio.AbstractServer | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        socket = (self._server.sockets or ())[0]
        return f"http://127.0.0.1:{socket.getsockname()[1]}"

    async def _handle(self, reader, writer) -> None:
        try:
            request_line = (
                (await asyncio.wait_for(reader.readline(), 2))
                .decode("latin-1")
                .rstrip("\r\n")
            )
            headers = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), 2)
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                headers[name.strip().lower()] = value.strip()
            self.requests.append({"request_line": request_line, **headers})
            response_head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            response_tail = (
                f"Content-Length: {len(self.body)}\r\nConnection: close\r\n\r\n"
            ).encode()
            writer.write(response_head + response_tail + self.body)
            await asyncio.wait_for(writer.drain(), 2)
        except (TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 2)
            except (TimeoutError, ConnectionError):
                pass

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc_info):
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()


def _use_proxy(monkeypatch, url: str) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, url)


@pytest.mark.asyncio
async def test_owned_client_ignores_ambient_proxy_and_authenticates_directly(
    monkeypatch,
):
    token = "fake-service-token"
    async with _Recorder({"instance_id": "local", "version": 1}) as service:
        async with _Recorder({"instance_id": "proxy", "version": 1}) as proxy:
            _use_proxy(monkeypatch, proxy.url)
            client = AgentServiceClient(service.url, token=token, timeout=1)
            try:
                health = await asyncio.wait_for(client.health(), 3)
            finally:
                await asyncio.wait_for(client.aclose(), 3)

            assert health.instance_id == "local"
            assert proxy.requests == []
            assert len(service.requests) == 1
            assert service.requests[0]["authorization"] == f"Bearer {token}"
            assert client._client.is_closed


@pytest.mark.asyncio
async def test_injected_client_keeps_explicit_proxy_and_borrowed_lifecycle(monkeypatch):
    token = "fake-service-token"
    async with _Recorder({"instance_id": "proxy", "version": 1}) as proxy:
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        async with httpx2.AsyncClient(proxy=proxy.url, trust_env=False) as http:
            client = AgentServiceClient(proxy.url, token=token, client=http)
            assert (await client.health()).instance_id == "proxy"
            await client.aclose()
            assert not http.is_closed

        assert len(proxy.requests) == 1
        assert proxy.requests[0]["authorization"] == f"Bearer {token}"
