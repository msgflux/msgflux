"""Negotiated HTTP response compression through the public service app.

Every test drives the root ``create_service_app`` ASGI application with
``httpx2.ASGITransport`` and an explicit ``Accept-Encoding`` header so
negotiation is deterministic; real-socket streaming behaviour is covered by the
native-SSE integration suite.
"""

import json
import os
import tomllib

import httpx2
import pytest

pytest.importorskip("litestar")

from msgflux.runtime.service import AgentSession, AgentService, SQLiteServiceStore
from msgflux.runtime.service.http import ServiceHTTPConfig, create_service_app
from msgflux.runtime.service.local.manifest import load_service_manifest
from tests.test_agent_service import _agent

TOKEN = "secret"
THREADS = 24
_MANIFEST_PREFIX = (
    '[defaults]\nagent = "custom"\n[agents.custom]\nfactory = "package:factory"\n'
)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
@pytest.mark.parametrize("exclusion", ["identity;q=0", "*;q=0"])
async def test_small_response_respects_identity_exclusion(encoding, exclusion):
    service, app = await _populated_app(compression=True)
    try:
        response = await _get(
            app,
            "/v1/health",
            accept_encoding=f"{encoding}, {exclusion}",
        )
        assert response.status_code == 200
        assert len(response.content) < 500
        assert response.headers["content-encoding"] == encoding
        assert isinstance(json.loads(response.content), dict)
    finally:
        await service.aclose()
        service.store.close()


async def _populated_app(*, compression: bool | None = None):
    """Service whose thread listing is comfortably above the minimum size."""
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(_agent()))
    for _ in range(THREADS):
        await service.open_thread("main")
    app = create_service_app(
        service,
        token=TOKEN,
        http_config=(
            None if compression is None else ServiceHTTPConfig(compression=compression)
        ),
    )
    return service, app


async def _get(app, path="/v1/threads", *, accept_encoding, authorized=True):
    headers = {"Accept-Encoding": accept_encoding}
    if authorized:
        headers["Authorization"] = f"Bearer {TOKEN}"
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://service"
    ) as client:
        return await client.get(path, headers=headers)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["zstd", "gzip", "identity"])
async def test_default_server_negotiates_compression(encoding):
    service, app = await _populated_app()
    try:
        response = await _get(app, accept_encoding=encoding)
        assert response.status_code == 200
        assert response.headers.get("content-encoding") == (
            None if encoding == "identity" else encoding
        )
        assert len(response.json()["threads"]) == THREADS
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accept_encoding,expected",
    [
        ("zstd", "zstd"),
        ("gzip", "gzip"),
        ("identity", None),
        ("*", "zstd"),
        ("gzip;q=0.9, zstd;q=0.5", "gzip"),
        ("zstd;q=0, gzip;q=1", "gzip"),
        ("zstd;q=0, gzip;q=0", None),
        ("br", None),
        ("", None),
        ("zstd;q=0, *;q=1", "gzip"),
        ("gzip;q=0.2, identity;q=0.9", None),
        ("ZSTD;q=0.8, gzip;q=0.2", "zstd"),
    ],
)
async def test_json_response_is_negotiated_and_decompressible(
    accept_encoding, expected
):
    service, app = await _populated_app(compression=True)
    try:
        response = await _get(app, accept_encoding=accept_encoding)
        assert response.status_code == 200
        if expected is None:
            assert "content-encoding" not in response.headers
            assert int(response.headers["content-length"]) == len(response.content)
        else:
            assert response.headers["content-encoding"] == expected
            # A negotiated coding really shrank the wire payload.
            assert int(response.headers["content-length"]) < len(response.content)
        assert "Accept-Encoding" in response.headers["vary"]
        payload = json.loads(response.content)
        assert [thread["agent_id"] for thread in payload["threads"]] == [
            "main"
        ] * THREADS
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("accept_encoding", ["identity;q=0", "*;q=0"])
async def test_no_acceptable_encoding_returns_406_without_body(accept_encoding):
    service, app = await _populated_app(compression=True)
    try:
        response = await _get(app, accept_encoding=accept_encoding)
        assert response.status_code == 406
        assert response.content == b""
        assert "content-encoding" not in response.headers
        assert response.headers["vary"] == "Accept-Encoding"
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accept_encoding",
    ["zstd;q=bogus", "zstd;q=", "zstd;q", "zstd;q=2", "gzip;q=-1", ";q=0.5"],
)
async def test_malformed_quality_values_never_fail_with_server_error(
    accept_encoding,
):
    service, app = await _populated_app(compression=True)
    try:
        response = await _get(app, accept_encoding=accept_encoding)
        # Unparsable or out-of-range qualities degrade to a negotiation
        # answer; they must never surface as a server error.
        assert response.status_code in {200, 406}
        assert "Accept-Encoding" in response.headers["vary"]
        if response.status_code == 200:
            assert len(json.loads(response.content)["threads"]) == THREADS
        else:
            assert response.content == b""
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accept_encoding", ["zstd", "gzip", "*", "identity", "*;q=0", "identity;q=0"]
)
async def test_disabled_compression_ignores_negotiation(accept_encoding):
    service, app = await _populated_app(compression=False)
    try:
        response = await _get(app, accept_encoding=accept_encoding)
        assert response.status_code == 200
        assert "content-encoding" not in response.headers
        assert "vary" not in response.headers
        assert int(response.headers["content-length"]) == len(response.content)
        assert len(json.loads(response.content)["threads"]) == THREADS
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_error_responses_keep_the_negotiation_contract():
    service, app = await _populated_app(compression=True)
    try:
        unauthorized = await _get(app, accept_encoding="zstd", authorized=False)
        assert unauthorized.status_code == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"
        # Bodies below the minimum size stay identity-coded but readable.
        assert "content-encoding" not in unauthorized.headers
        assert json.loads(unauthorized.content)["code"] == "unauthorized"

        missing = await _get(
            app, f"/v1/agents/{'a' * 700}/inspection", accept_encoding="zstd"
        )
        assert missing.status_code == 404
        assert missing.headers["content-encoding"] == "zstd"
        assert "Accept-Encoding" in missing.headers["vary"]
        assert int(missing.headers["content-length"]) < len(missing.content)
        assert json.loads(missing.content)["code"] == "agent_not_found"

        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            invalid = await client.post(
                "/v1/threads",
                headers={
                    "Accept-Encoding": "gzip",
                    "Authorization": f"Bearer {TOKEN}",
                },
                json={"unexpected": True},
            )
        assert invalid.status_code == 422
        assert "content-encoding" not in invalid.headers
        assert json.loads(invalid.content)["code"] == "invalid_request"
    finally:
        await service.aclose()
        service.store.close()


def test_manifest_compression_is_a_strict_boolean(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(_MANIFEST_PREFIX)
    assert load_service_manifest(config).http.compression is True
    config.write_text(_MANIFEST_PREFIX + "[http]\ncors_allow_origins = []\n")
    assert load_service_manifest(config).http.compression is True
    config.write_text(_MANIFEST_PREFIX + "[http]\ncompression = true\n")
    assert load_service_manifest(config).http == ServiceHTTPConfig(compression=True)
    config.write_text(_MANIFEST_PREFIX + "[http]\ncompression = false\n")
    assert load_service_manifest(config).http == ServiceHTTPConfig(compression=False)
    for invalid in ('"yes"', "1", "[]"):
        config.write_text(
            _MANIFEST_PREFIX
            + f'[http]\ncors_allow_origins = ["*"]\ncompression = {invalid}\n'
        )
        with pytest.raises(ValueError, match="Expected `bool`"):
            load_service_manifest(config)


def test_invalid_boolean_cannot_hide_behind_wildcard_origins():
    with pytest.raises(TypeError, match="compression must be a bool"):
        ServiceHTTPConfig(("*",), compression="yes")
    with pytest.raises(TypeError, match="compression must be a bool"):
        ServiceHTTPConfig(("*",), compression=1)
    wildcard = ServiceHTTPConfig(("*",), compression=True)
    assert wildcard.cors_allow_origins == ("*",) and wildcard.compression is True


@pytest.mark.skipif(os.name != "posix", reason="Config editing uses POSIX locks")
def test_cors_mutation_preserves_compression_setting(tmp_path):
    pytest.importorskip("tomlkit")
    from msgflux.runtime.service.local.cors_config import mutate_cors_config

    path = tmp_path / "service.toml"
    path.write_text(
        _MANIFEST_PREFIX + "[http]\ncors_allow_origins = []\ncompression = true\n"
    )
    result = mutate_cors_config(path, "https://example.com", operation="add")
    assert result.changed and result.restart_required
    assert result.cors_allow_origins == ("https://example.com",)
    assert load_service_manifest(path).http == ServiceHTTPConfig(
        ("https://example.com",), compression=True
    )
    assert tomllib.loads(path.read_text())["http"]["compression"] is True
