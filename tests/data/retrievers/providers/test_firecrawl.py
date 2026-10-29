import asyncio
import json
from unittest.mock import patch

import httpx2
import pytest

from msgflux.data.retrievers.providers.firecrawl import (
    FIRECRAWL_API_URL,
    FirecrawlWebFetchRetriever,
    FirecrawlWebRetriever,
)
from msgflux.data.retrievers.retriever import Retriever
from msgflux.nn.modules.searcher import Searcher

from ._httpx2_mock import install_httpx2_mock, json_response


def _scrape_ok(request: httpx2.Request) -> httpx2.Response:
    return json_response(
        request,
        200,
        {
            "success": True,
            "data": {
                "markdown": "# Firecrawl Markdown",
                "metadata": {"url": "https://example.com"},
            },
        },
    )


def _search_ok(request: httpx2.Request) -> httpx2.Response:
    return json_response(
        request,
        200,
        {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "First",
                        "url": "https://first.example",
                        "description": "First description",
                        "markdown": "# First",
                    },
                    {
                        "title": "Second",
                        "url": "https://second.example",
                        "description": "Second description",
                        "markdown": None,
                    },
                ]
            },
        },
    )


def test_fetch_init_defaults(mocker):
    install_httpx2_mock(mocker, _scrape_ok)
    retriever = FirecrawlWebFetchRetriever()

    assert retriever.provider == "firecrawl"
    assert retriever.retriever_type == "web_fetch"
    assert retriever.base_url == FIRECRAWL_API_URL
    assert retriever.timeout == 30.0
    assert not hasattr(retriever, "api_key")

    retriever.close()


def test_search_registration_uses_web_search_type(mocker):
    install_httpx2_mock(mocker, _search_ok)
    retriever = FirecrawlWebRetriever()

    assert retriever.provider == "firecrawl"
    assert retriever.retriever_type == "web_search"
    assert "firecrawl" in Retriever.providers()["web_search"]
    assert "web" not in Retriever.retriever_types()

    retriever.close()


def test_sync_scrape_keyless_has_no_authorization(mocker):
    mock = install_httpx2_mock(mocker, _scrape_ok)
    retriever = FirecrawlWebFetchRetriever()

    with patch.dict("os.environ", {}, clear=True):
        result = retriever("example.com/page?q=1")

    assert result.response_type == "web_fetch"
    assert result.data[0]["data"] == {
        "url": "https://example.com/page?q=1",
        "content": "# Firecrawl Markdown",
    }
    request = mock.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{FIRECRAWL_API_URL}/scrape"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "url": "https://example.com/page?q=1",
        "formats": ["markdown"],
    }

    retriever.close()


def test_sync_scrape_reads_api_key_per_request(mocker):
    mock = install_httpx2_mock(mocker, _scrape_ok)
    retriever = FirecrawlWebFetchRetriever()

    with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "first"}, clear=True):
        retriever("https://example.com")
    with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "rotated"}, clear=True):
        retriever("https://example.com")

    assert mock.requests[0].headers["authorization"] == "Bearer first"
    assert mock.requests[1].headers["authorization"] == "Bearer rotated"

    retriever.close()


def test_scrape_http_429_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 429, {"error": "Request rate limit exceeded"}
        ),
    )
    retriever = FirecrawlWebFetchRetriever()

    with pytest.raises(RuntimeError, match=r"429.*rate limit"):
        retriever("https://example.com")

    retriever.close()


def test_scrape_success_false_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 200, {"success": False, "error": "daily keyless quota reached"}
        ),
    )
    retriever = FirecrawlWebFetchRetriever()

    with pytest.raises(RuntimeError, match="daily keyless quota reached"):
        retriever("https://example.com")

    retriever.close()


def test_scrape_empty_markdown_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 200, {"success": True, "data": {"markdown": " "}}
        ),
    )
    retriever = FirecrawlWebFetchRetriever()

    with pytest.raises(RuntimeError, match="empty markdown"):
        retriever("https://example.com")

    retriever.close()


def test_sync_search_parses_markdown_and_descriptions(mocker):
    mock = install_httpx2_mock(mocker, _search_ok)
    retriever = FirecrawlWebRetriever(country="US")

    with patch.dict("os.environ", {}, clear=True):
        result = retriever("search query", top_k=2)

    assert result.response_type == "web_search"
    assert len(result.data) == 1
    entries = result.data[0].results
    assert entries[0]["data"] == {
        "title": "First",
        "content": "# First",
        "url": "https://first.example",
    }
    assert entries[1]["data"]["content"] == "Second description"
    request = mock.requests[0]
    assert str(request.url) == f"{FIRECRAWL_API_URL}/search"
    assert "authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload["query"] == "search query"
    assert payload["limit"] == 2
    assert payload["country"] == "US"
    assert payload["scrapeOptions"] == {"formats": [{"type": "markdown"}]}

    retriever.close()


def test_search_sends_optional_api_key(mocker):
    mock = install_httpx2_mock(mocker, _search_ok)
    retriever = FirecrawlWebRetriever()

    with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "search-key"}, clear=True):
        retriever("search query", top_k=1)

    assert mock.requests[0].headers["authorization"] == "Bearer search-key"

    retriever.close()


def test_search_http_429_is_not_silently_empty(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 429, {"error": "Request rate limit exceeded"}
        ),
    )
    retriever = FirecrawlWebRetriever()

    with pytest.raises(RuntimeError, match="rate limit"):
        retriever("search query")

    retriever.close()


def test_search_success_false_is_not_silently_empty(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(
            request, 200, {"success": False, "error": "keyless quota exhausted"}
        ),
    )
    retriever = FirecrawlWebRetriever()

    with pytest.raises(RuntimeError, match="keyless quota exhausted"):
        retriever("search query")

    retriever.close()


@pytest.mark.asyncio
async def test_async_scrape_and_search(mocker):
    def handler(request):
        if request.url.path.endswith("/search"):
            return _search_ok(request)
        return _scrape_ok(request)

    mock = install_httpx2_mock(mocker, handler)
    fetch = FirecrawlWebFetchRetriever()
    search = FirecrawlWebRetriever()

    fetched = await fetch.acall("https://example.com")
    searched = await search.acall("search query", top_k=1)

    assert fetched.data[0]["data"]["content"] == "# Firecrawl Markdown"
    assert searched.data[0].results[0]["data"]["title"] == "First"
    assert len(mock.requests) == 2

    await fetch.aclose()
    await search.aclose()


@pytest.mark.asyncio
async def test_async_cancellation_propagates(mocker):
    async def cancelled(request):
        raise asyncio.CancelledError

    install_httpx2_mock(mocker, cancelled)
    retriever = FirecrawlWebFetchRetriever()

    with pytest.raises(asyncio.CancelledError):
        await retriever.acall("https://example.com")

    await retriever.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [Retriever.web_search, Retriever.web])
@pytest.mark.parametrize("serialized_type", ["web_search", "web"])
async def test_search_serialization_roundtrip_rebuilds_clients(
    mocker, factory, serialized_type
):
    install_httpx2_mock(mocker, _search_ok)
    retriever = factory(
        "firecrawl",
        include_domains=["example.com"],
        tbs="qdr:w",
        country="BR",
        timeout=11,
    )

    serialized = retriever.serialize()

    assert "client" not in serialized["state"]
    assert "aclient" not in serialized["state"]
    assert serialized["retriever_type"] == "web_search"
    assert serialized["state"]["include_domains"] == ["example.com"]
    assert serialized["state"]["tbs"] == "qdr:w"
    assert serialized["state"]["country"] == "BR"
    assert serialized["state"]["timeout"] == 11

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized_type,
        serialized["state"],
    )

    assert restored.retriever_type == "web_search"
    assert Searcher(retriever=restored).retriever is restored
    assert restored.include_domains == ["example.com"]
    assert restored.tbs == "qdr:w"
    assert restored.country == "BR"
    assert restored.timeout == 11
    assert restored.client is not retriever.client
    result = await restored.acall("Python documentation", top_k=1)
    assert result.response_type == "web_search"
    assert result.data[0].results[0].data.url == "https://first.example"

    await retriever.aclose()
    await restored.aclose()


@pytest.mark.asyncio
async def test_serialization_roundtrip_rebuilds_clients(mocker):
    install_httpx2_mock(mocker, _scrape_ok)
    retriever = FirecrawlWebFetchRetriever(timeout=9)

    serialized = retriever.serialize()

    assert "client" not in serialized["state"]
    assert "aclient" not in serialized["state"]
    assert serialized["state"]["base_url"] == FIRECRAWL_API_URL
    assert serialized["state"]["timeout"] == 9

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert restored.base_url == FIRECRAWL_API_URL
    assert restored.timeout == 9
    assert restored.client is not retriever.client

    await retriever.aclose()
    await restored.aclose()
