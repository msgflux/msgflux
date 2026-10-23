"""Regression tests for MCP OAuth2 token expiry and refresh coordination."""

import asyncio

import pytest

from msgflux.protocols.mcp.auth import OAuth2Auth
from msgflux.protocols.mcp.transports import HTTPTransport


def _expired_auth(refresh_callback) -> OAuth2Auth:
    return OAuth2Auth(
        access_token="expired-access",
        refresh_token="refresh-1",
        expires_in=-1,
        refresh_callback=refresh_callback,
    )


@pytest.mark.asyncio
async def test_zero_expiry_refreshes_before_first_request():
    calls = []

    async def refresh(refresh_token):
        calls.append(refresh_token)
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }

    auth = OAuth2Auth(
        access_token="expired-access",
        refresh_token="refresh-1",
        expires_in=0,
        refresh_callback=refresh,
    )
    transport = HTTPTransport("http://127.0.0.1:1/mcp", auth=auth)

    assert auth.is_expired()

    headers = await transport._get_headers()

    assert calls == ["refresh-1"]
    assert headers["Authorization"] == "Bearer new-access"
    assert not auth.is_expired()


@pytest.mark.asyncio
async def test_concurrent_header_builds_share_one_refresh():
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def refresh(refresh_token):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {
            "access_token": f"access-{calls}",
            "refresh_token": refresh_token,
            "expires_in": 3600,
        }

    auth = _expired_auth(refresh)
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
    assert {header["Authorization"] for header in headers} == {"Bearer access-1"}


@pytest.mark.asyncio
async def test_refresh_without_expiry_does_not_refresh_every_request():
    calls = 0

    async def refresh(refresh_token):
        nonlocal calls
        calls += 1
        return {"access_token": f"access-{calls}"}

    auth = _expired_auth(refresh)
    transport = HTTPTransport("http://127.0.0.1:1/mcp", auth=auth)

    first = await transport._get_headers()
    second = await transport._get_headers()

    assert calls == 1
    assert first["Authorization"] == "Bearer access-1"
    assert second["Authorization"] == "Bearer access-1"
    assert not auth.is_expired()


@pytest.mark.asyncio
async def test_refresh_with_null_expiry_succeeds_with_unknown_lifetime():
    calls = 0

    async def refresh(refresh_token):
        nonlocal calls
        calls += 1
        return {"access_token": f"access-{calls}", "expires_in": None}

    auth = _expired_auth(refresh)

    assert await auth.refresh() is True
    assert auth._access_token == "access-1"
    assert not auth.is_expired()
    assert await auth.refresh_if_needed() is False
    assert calls == 1


def test_update_tokens_with_zero_expiry_expires_immediately():
    auth = OAuth2Auth(access_token="old-access")

    auth.update_tokens(access_token="new-access", expires_in=0)

    assert auth.is_expired()


@pytest.mark.asyncio
async def test_cancelled_refresh_owner_releases_lock_for_waiter():
    calls = 0
    started = asyncio.Event()
    cancelled = asyncio.Event()
    waiter_ready = asyncio.Event()

    async def refresh(refresh_token):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return {"access_token": f"access-{calls}", "expires_in": 3600}

    async def wait_for_refresh():
        waiter_ready.set()
        return await auth.refresh_if_needed()

    auth = _expired_auth(refresh)
    owner = asyncio.create_task(auth.refresh_if_needed())
    waiter = None
    try:
        await asyncio.wait_for(started.wait(), timeout=1)

        waiter = asyncio.create_task(wait_for_refresh())
        await asyncio.wait_for(waiter_ready.wait(), timeout=1)

        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner

        assert cancelled.is_set()
        assert await asyncio.wait_for(waiter, timeout=1) is True
        assert calls == 2
        assert auth._access_token == "access-2"
        assert not auth.is_expired()
    finally:
        for task in (owner, waiter):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (owner, waiter) if task is not None),
            return_exceptions=True,
        )
