import asyncio

import httpx2
import pytest

from msgflux.data.retrievers.providers.http import HttpWebFetchRetriever
from msgflux.data.retrievers.retriever import Retriever

from ._httpx2_mock import install_httpx2_mock, json_response


def _html(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        text="<html><body><p>Hello world</p></body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
        request=request,
    )


def test_init_defaults(mocker):
    install_httpx2_mock(mocker, _html)
    retriever = HttpWebFetchRetriever()

    assert retriever.provider == "http"
    assert retriever.retriever_type == "web_fetch"
    assert retriever.default_headers == {}
    assert retriever.timeout == 30.0

    retriever.close()


def test_sync_html_fetch_uses_html_to_text(mocker):
    mock = install_httpx2_mock(mocker, _html)
    html_to_text = mocker.patch(
        "msgflux.data.retrievers.providers.http.html_to_text",
        return_value="Hello world",
    )
    retriever = HttpWebFetchRetriever(default_headers={"X-Test": "yes"})

    result = retriever("example.com/page?q=1")

    assert result.response_type == "web_fetch"
    assert result.data[0]["data"] == {
        "url": "https://example.com/page?q=1",
        "content": "Hello world",
    }
    html_to_text.assert_called_once_with("<html><body><p>Hello world</p></body></html>")
    request = mock.requests[0]
    assert request.method == "GET"
    assert str(request.url) == "https://example.com/page?q=1"
    assert request.headers["x-test"] == "yes"

    retriever.close()


def test_markdown_content_type_is_respected(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: httpx2.Response(
            200,
            text="# Already Markdown",
            headers={"content-type": "text/markdown; charset=utf-8"},
            request=request,
        ),
    )
    html_to_text = mocker.patch("msgflux.data.retrievers.providers.http.html_to_text")
    retriever = HttpWebFetchRetriever()

    result = retriever("https://example.com/readme")

    assert result.data[0]["data"]["content"] == "# Already Markdown"
    html_to_text.assert_not_called()

    retriever.close()


def test_markdown_path_is_respected_without_content_type(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: httpx2.Response(
            200,
            text="# Markdown file",
            headers={"content-type": "text/plain"},
            request=request,
        ),
    )
    html_to_text = mocker.patch("msgflux.data.retrievers.providers.http.html_to_text")
    retriever = HttpWebFetchRetriever()

    result = retriever("https://example.com/guide.md")

    assert result.data[0]["data"]["content"] == "# Markdown file"
    html_to_text.assert_not_called()

    retriever.close()


def test_host_port_without_scheme_is_normalized(mocker):
    mock = install_httpx2_mock(mocker, _html)
    mocker.patch(
        "msgflux.data.retrievers.providers.http.html_to_text",
        return_value="Hello world",
    )
    retriever = HttpWebFetchRetriever()

    result = retriever("example.com:8080/path?q=1")

    assert result.data[0]["data"]["url"] == "https://example.com:8080/path?q=1"
    assert str(mock.requests[0].url) == "https://example.com:8080/path?q=1"

    retriever.close()


def test_rejects_non_http_scheme_without_request(mocker):
    mock = install_httpx2_mock(mocker, _html)
    retriever = HttpWebFetchRetriever()

    with pytest.raises(ValueError, match="only http and https"):
        retriever("ftp://example.com/file")

    assert mock.requests == []
    retriever.close()


def test_http_error_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: httpx2.Response(
            404,
            text="not found",
            request=request,
        ),
    )
    retriever = HttpWebFetchRetriever()

    with pytest.raises(RuntimeError, match="404"):
        retriever("https://example.com/missing")

    retriever.close()


def test_empty_response_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: httpx2.Response(
            200,
            text=" ",
            headers={"content-type": "text/markdown"},
            request=request,
        ),
    )
    retriever = HttpWebFetchRetriever()

    with pytest.raises(RuntimeError, match="empty response"):
        retriever("https://example.com")

    retriever.close()


def test_json_response_is_html_converted(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"ok": True}),
    )
    html_to_text = mocker.patch(
        "msgflux.data.retrievers.providers.http.html_to_text",
        return_value='{"ok": true}',
    )
    retriever = HttpWebFetchRetriever()

    result = retriever("https://example.com/data")

    assert result.data[0]["data"]["content"] == '{"ok": true}'
    html_to_text.assert_called_once_with('{"ok":true}')

    retriever.close()


@pytest.mark.asyncio
async def test_async_fetch(mocker):
    mock = install_httpx2_mock(mocker, _html)
    mocker.patch(
        "msgflux.data.retrievers.providers.http.html_to_text",
        return_value="Hello world",
    )
    retriever = HttpWebFetchRetriever()

    result = await retriever.acall(["example.com", "https://two.example"])

    assert [item["data"]["url"] for item in result.data] == [
        "https://example.com",
        "https://two.example",
    ]
    assert len(mock.requests) == 2

    await retriever.aclose()


@pytest.mark.asyncio
async def test_async_cancellation_propagates(mocker):
    async def cancelled(request):
        raise asyncio.CancelledError

    install_httpx2_mock(mocker, cancelled)
    retriever = HttpWebFetchRetriever()

    with pytest.raises(asyncio.CancelledError):
        await retriever.acall("https://example.com")

    await retriever.aclose()


@pytest.mark.asyncio
async def test_serialization_roundtrip_rebuilds_clients(mocker):
    install_httpx2_mock(mocker, _html)
    retriever = HttpWebFetchRetriever(
        default_headers={"X-Test": "yes"},
        timeout=8,
    )

    serialized = retriever.serialize()

    assert "client" not in serialized["state"]
    assert "aclient" not in serialized["state"]
    assert serialized["state"]["default_headers"] == {"X-Test": "yes"}
    assert serialized["state"]["timeout"] == 8

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert restored.default_headers == {"X-Test": "yes"}
    assert restored.timeout == 8
    assert restored.client is not retriever.client

    await retriever.aclose()
    await restored.aclose()


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "example.com/path?next=https://other.example/x",
            "https://example.com/path?next=https://other.example/x",
        ),
        (
            "example.com/path#https://other.example/x",
            "https://example.com/path#https://other.example/x",
        ),
        (
            "example.com/http://other.example",
            "https://example.com/http://other.example",
        ),
        (
            "localhost:8080/path?next=https://other.example",
            "https://localhost:8080/path?next=https://other.example",
        ),
        (
            "//example.com/path?next=https://other.example",
            "https://example.com/path?next=https://other.example",
        ),
    ],
)
def test_normalize_schemeless_url_with_nested_url(url, expected):
    from msgflux.data.retrievers.providers.http import normalize_url

    assert normalize_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "mailto:user@example.com",
        "file:/etc/passwd",
        "urn:user@example.com",
        "http:example.com",
        "https:/example.com",
    ],
)
def test_non_http_uri_is_not_reinterpreted_as_http_credentials(url):
    from msgflux.data.retrievers.providers.http import normalize_url

    with pytest.raises(ValueError):
        normalize_url(url)
