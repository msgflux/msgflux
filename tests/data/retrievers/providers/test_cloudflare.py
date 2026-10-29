import asyncio
import json
from unittest.mock import patch

import httpx2
import pytest

from msgflux.data.retrievers.providers.cloudflare import (
    CLOUDFLARE_API_BASE,
    CloudflareWebFetchRetriever,
)
from msgflux.data.retrievers.retriever import Retriever

from ._httpx2_mock import install_httpx2_mock, json_response


def _ok(request: httpx2.Request) -> httpx2.Response:
    return json_response(
        request,
        200,
        {"success": True, "result": "# Cloudflare Markdown"},
    )


ENV = {"CLOUDFLARE_ACCOUNT_ID": "account-1", "CLOUDFLARE_API_TOKEN": "token-1"}


def test_init_defaults(mocker):
    install_httpx2_mock(mocker, _ok)
    with patch.dict("os.environ", ENV, clear=True):
        retriever = CloudflareWebFetchRetriever()

    assert retriever.provider == "cloudflare"
    assert retriever.retriever_type == "web_fetch"
    assert retriever.account_id is None
    assert retriever.account_id_env == "CLOUDFLARE_ACCOUNT_ID"
    assert retriever.api_token_env == "CLOUDFLARE_API_TOKEN"
    assert retriever.base_url == CLOUDFLARE_API_BASE
    assert retriever.timeout == 30.0
    assert not hasattr(retriever, "api_token")

    retriever.close()


def test_sync_fetch_uses_account_and_bearer_token(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    with patch.dict("os.environ", ENV, clear=True):
        retriever = CloudflareWebFetchRetriever()
        result = retriever("example.com/page?q=1")

    assert result.response_type == "web_fetch"
    assert result.data[0]["data"] == {
        "url": "https://example.com/page?q=1",
        "content": "# Cloudflare Markdown",
    }
    request = mock.requests[0]
    assert request.method == "POST"
    assert (
        str(request.url) == "https://api.cloudflare.com/client/v4/accounts/account-1/"
        "browser-run/markdown"
    )
    assert request.headers["authorization"] == "Bearer token-1"
    assert json.loads(request.content) == {"url": "https://example.com/page?q=1"}

    retriever.close()


def test_custom_account_and_token_env(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    with patch.dict(
        "os.environ",
        {"MY_ACCOUNT": "account-2", "MY_TOKEN": "token-2"},
        clear=True,
    ):
        retriever = CloudflareWebFetchRetriever(
            account_id="account-2",
            account_id_env="MY_ACCOUNT",
            api_token_env="MY_TOKEN",
            base_url="https://cf.example/client/v4",
            timeout=4,
        )
        retriever("https://example.com")

    assert str(mock.requests[0].url) == (
        "https://cf.example/client/v4/accounts/account-2/browser-run/markdown"
    )
    assert mock.requests[0].headers["authorization"] == "Bearer token-2"
    assert retriever.timeout == 4

    retriever.close()


def test_reads_token_per_request(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    with patch.dict(
        "os.environ",
        {"CLOUDFLARE_ACCOUNT_ID": "account-1", "CLOUDFLARE_API_TOKEN": "token-1"},
        clear=True,
    ):
        retriever = CloudflareWebFetchRetriever()
        retriever("https://example.com")

    with patch.dict(
        "os.environ",
        {"CLOUDFLARE_ACCOUNT_ID": "account-1", "CLOUDFLARE_API_TOKEN": "rotated"},
        clear=True,
    ):
        retriever("https://example.com")

    assert mock.requests[-1].headers["authorization"] == "Bearer rotated"

    retriever.close()


def test_missing_account_is_surfaced(mocker):
    install_httpx2_mock(mocker, _ok)
    with patch.dict("os.environ", {"CLOUDFLARE_API_TOKEN": "token-1"}, clear=True):
        retriever = CloudflareWebFetchRetriever()

        with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
            retriever("https://example.com")

    retriever.close()


def test_missing_token_is_surfaced(mocker):
    install_httpx2_mock(mocker, _ok)
    with patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "account-1"}, clear=True):
        retriever = CloudflareWebFetchRetriever()

        with pytest.raises(ValueError, match="CLOUDFLARE_API_TOKEN"):
            retriever("https://example.com")

    retriever.close()


def test_http_error_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 500, {"success": False, "errors": ["boom"]}
        ),
    )
    with patch.dict("os.environ", {"CLOUDFLARE_API_TOKEN": "token-1"}, clear=True):
        retriever = CloudflareWebFetchRetriever(account_id="account-1")

        with pytest.raises(RuntimeError, match=r"500.*boom"):
            retriever("https://example.com")

    retriever.close()


def test_success_false_body_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request,
            200,
            {"success": False, "errors": [{"message": "quota exceeded"}]},
        ),
    )
    with patch.dict("os.environ", {"CLOUDFLARE_API_TOKEN": "token-1"}, clear=True):
        retriever = CloudflareWebFetchRetriever(account_id="account-1")

        with pytest.raises(RuntimeError, match="quota exceeded"):
            retriever("https://example.com")

    retriever.close()


@pytest.mark.asyncio
async def test_async_fetch(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    with patch.dict("os.environ", ENV, clear=True):
        retriever = CloudflareWebFetchRetriever()
        result = await retriever.acall("https://example.com")

    assert result.data[0]["data"]["content"] == "# Cloudflare Markdown"
    assert str(mock.requests[0].url).endswith("/browser-run/markdown")

    await retriever.aclose()


@pytest.mark.asyncio
async def test_async_cancellation_propagates(mocker):
    async def cancelled(request):
        raise asyncio.CancelledError

    install_httpx2_mock(mocker, cancelled)
    with patch.dict("os.environ", ENV, clear=True):
        retriever = CloudflareWebFetchRetriever()

        with pytest.raises(asyncio.CancelledError):
            await retriever.acall("https://example.com")

    await retriever.aclose()


@pytest.mark.asyncio
async def test_serialization_roundtrip_rebuilds_clients(mocker):
    install_httpx2_mock(mocker, _ok)
    with patch.dict("os.environ", ENV, clear=True):
        retriever = CloudflareWebFetchRetriever(timeout=6)

    serialized = retriever.serialize()

    assert "client" not in serialized["state"]
    assert "aclient" not in serialized["state"]
    assert serialized["state"]["timeout"] == 6
    assert serialized["state"]["account_id_env"] == "CLOUDFLARE_ACCOUNT_ID"

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert restored.timeout == 6
    assert restored.api_token_env == "CLOUDFLARE_API_TOKEN"
    assert restored.client is not retriever.client

    await retriever.aclose()
    await restored.aclose()
