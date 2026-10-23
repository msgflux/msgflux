"""Regression tests for BearerTokenAuth explicit zero expiry."""

import asyncio

import pytest

from msgflux.protocols.mcp.auth import BearerTokenAuth
from msgflux.protocols.mcp.transports import HTTPTransport


@pytest.mark.asyncio
async def test_zero_expiry_refreshes_before_first_request():
    calls = []

    async def refresh():
        calls.append("refresh")
        return "new-token"

    auth = BearerTokenAuth(
        token="old-token",
        expires_in=0,
        refresh_callback=refresh,
    )

    assert auth.is_expired()

    assert await auth.refresh_if_needed() is True
    assert calls == ["refresh"]
    assert auth.apply_auth({})["Authorization"] == "Bearer new-token"


def test_update_token_with_zero_expiry_expires_immediately():
    auth = BearerTokenAuth(token="old-token")

    auth.update_token("new-token", expires_in=0)

    assert auth.is_expired()
    assert auth.apply_auth({})["Authorization"] == "Bearer new-token"


@pytest.mark.asyncio
async def test_successful_refresh_clears_stale_deadline():
    calls = 0

    async def refresh():
        nonlocal calls
        calls += 1
        return f"new-token-{calls}"

    auth = BearerTokenAuth(
        token="old-token",
        expires_in=0,
        refresh_callback=refresh,
    )
    transport = HTTPTransport("http://127.0.0.1:1/mcp", auth=auth)

    first = await transport._get_headers()
    second = await transport._get_headers()

    assert calls == 1
    assert first["Authorization"] == "Bearer new-token-1"
    assert second["Authorization"] == "Bearer new-token-1"
    assert auth._expires_at is None
    assert not auth.is_expired()


@pytest.mark.asyncio
async def test_concurrent_header_builds_share_one_refresh():
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def refresh():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return f"new-token-{calls}"

    auth = BearerTokenAuth(
        token="old-token",
        expires_in=0,
        refresh_callback=refresh,
    )
    transport = HTTPTransport("http://127.0.0.1:1/mcp", auth=auth)

    tasks = [asyncio.create_task(transport._get_headers()) for _ in range(8)]
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        headers = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert calls == 1
    assert {header["Authorization"] for header in headers} == {"Bearer new-token-1"}
    assert not auth.is_expired()
