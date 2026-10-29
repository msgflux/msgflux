import asyncio
import json

import httpx2
import pytest

from msgflux.data.retrievers.providers.markdown import (
    DEFAULT_MARKDOWN_ENDPOINT,
    MarkdownWebFetchRetriever,
)
from msgflux.data.retrievers.retriever import Retriever

from ._httpx2_mock import install_httpx2_mock, json_response


def _ok(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        text="# Markdown body",
        headers={"content-type": "text/markdown; charset=utf-8"},
        request=request,
    )


def test_retriever_web_fetch_factory(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = Retriever.web_fetch("markdown", method="ai", timeout=3)

    assert isinstance(retriever, MarkdownWebFetchRetriever)
    assert retriever.retriever_type == "web_fetch"
    assert retriever.method == "ai"
    assert retriever.timeout == 3

    retriever.close()


def test_init_defaults(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever()

    assert retriever.provider == "markdown"
    assert retriever.retriever_type == "web_fetch"
    assert retriever.endpoint == DEFAULT_MARKDOWN_ENDPOINT
    assert retriever.method == "auto"
    assert retriever.retain_images is False
    assert retriever.timeout == 30.0
    assert retriever.client is not None
    assert retriever.aclient is not None

    retriever.close()


def test_custom_configuration(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever(
        endpoint="https://parser.example",
        method="browser",
        retain_images=True,
        default_headers={"X-Test": "yes"},
        timeout=5,
    )

    assert retriever.endpoint == "https://parser.example"
    assert retriever.method == "browser"
    assert retriever.retain_images is True
    assert retriever.default_headers == {"X-Test": "yes"}
    assert retriever.timeout == 5

    retriever.close()


def test_rejects_invalid_method(mocker):
    install_httpx2_mock(mocker, _ok)

    with pytest.raises(ValueError, match="auto, ai, browser"):
        MarkdownWebFetchRetriever(method="javascript")


def test_sync_fetch_posts_json_and_returns_standard_shape(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever(default_headers={"X-Test": "yes"})

    result = retriever("example.com/page?q=1")

    assert result.response_type == "web_fetch"
    assert result.data[0]["data"] == {
        "url": "https://example.com/page?q=1",
        "content": "# Markdown body",
    }
    request = mock.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://markdown.new/"
    assert request.headers["x-test"] == "yes"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "url": "https://example.com/page?q=1",
        "method": "auto",
    }

    retriever.close()


def test_sync_fetch_sends_browser_and_images(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever(method="browser", retain_images=True)

    retriever("https://example.com")

    assert json.loads(mock.requests[0].content) == {
        "url": "https://example.com",
        "method": "browser",
        "retain_images": True,
    }

    retriever.close()


def test_fetch_preserves_url_order(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever()

    result = retriever(["https://one.example", "https://two.example"])

    assert [item["data"]["url"] for item in result.data] == [
        "https://one.example",
        "https://two.example",
    ]

    retriever.close()


def test_rejects_non_http_schemes(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever()

    with pytest.raises(ValueError, match="only http and https"):
        retriever("file:///etc/passwd")

    retriever.close()


def test_http_error_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 429, {"error": "daily conversion limit reached"}
        ),
    )
    retriever = MarkdownWebFetchRetriever()

    with pytest.raises(RuntimeError, match="429"):
        retriever("https://example.com")

    retriever.close()


def test_empty_response_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: httpx2.Response(200, text=" ", request=request),
    )
    retriever = MarkdownWebFetchRetriever()

    with pytest.raises(RuntimeError, match="empty response"):
        retriever("https://example.com")

    retriever.close()


@pytest.mark.asyncio
async def test_async_fetch_posts_json(mocker):
    mock = install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever()

    result = await retriever.acall(["example.com"])

    assert result.response_type == "web_fetch"
    assert result.data[0]["data"]["content"] == "# Markdown body"
    assert str(mock.requests[0].url) == "https://markdown.new/"

    await retriever.aclose()


@pytest.mark.asyncio
async def test_async_cancellation_propagates(mocker):
    async def cancelled(request):
        raise asyncio.CancelledError

    install_httpx2_mock(mocker, cancelled)
    retriever = MarkdownWebFetchRetriever()

    with pytest.raises(asyncio.CancelledError):
        await retriever.acall("https://example.com")

    await retriever.aclose()


@pytest.mark.asyncio
async def test_serialization_roundtrip_rebuilds_clients(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever(
        endpoint="https://parser.example",
        method="ai",
        retain_images=True,
        default_headers={"X-Test": "yes"},
        timeout=7,
    )

    serialized = retriever.serialize()

    assert "client" not in serialized["state"]
    assert "aclient" not in serialized["state"]
    assert serialized["retriever_type"] == "web_fetch"
    assert serialized["state"]["endpoint"] == "https://parser.example"
    assert serialized["state"]["method"] == "ai"
    assert serialized["state"]["retain_images"] is True
    assert serialized["state"]["timeout"] == 7

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert restored.endpoint == "https://parser.example"
    assert restored.method == "ai"
    assert restored.retain_images is True
    assert restored.default_headers == {"X-Test": "yes"}
    assert restored.client is not retriever.client
    assert restored.aclient is not retriever.aclient

    await retriever.aclose()
    await restored.aclose()


def test_close_is_idempotent(mocker):
    install_httpx2_mock(mocker, _ok)
    retriever = MarkdownWebFetchRetriever()

    retriever.close()
    retriever.close()

    assert retriever.client.is_closed is True
