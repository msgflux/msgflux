from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import msgflux.data.retrievers.providers.searxng as searxng_provider
from msgflux.data.retrievers.providers.searxng import SearXNGWebRetriever


@pytest.fixture
def mock_httpx_clients():
    response = MagicMock()
    response.json.return_value = {}
    response.raise_for_status.return_value = None

    sync_client = MagicMock()
    sync_client.get.return_value = response

    async_client = MagicMock()
    async_client.get = AsyncMock(return_value=response)
    async_client.aclose = AsyncMock(return_value=None)

    client_cls = MagicMock(return_value=sync_client)
    async_client_cls = MagicMock(return_value=async_client)
    fake_httpx = SimpleNamespace(Client=client_cls, AsyncClient=async_client_cls)

    with patch.object(searxng_provider, "httpx", fake_httpx):
        yield SimpleNamespace(
            response=response,
            client_cls=client_cls,
            sync_client=sync_client,
            async_client_cls=async_client_cls,
            async_client=async_client,
        )


@pytest.fixture
def retriever(mock_httpx_clients):
    with patch.dict("os.environ", {}, clear=True):
        yield SearXNGWebRetriever()


def test_init_defaults(retriever):
    assert retriever.base_url == "http://localhost:8080"
    assert retriever.search_url == "http://localhost:8080/search"
    assert retriever.categories is None
    assert retriever.engines is None
    assert retriever.language is None
    assert retriever.time_range is None
    assert retriever.safesearch is None
    assert retriever.pageno is None
    assert retriever.timeout == 30.0
    assert retriever.client is not None
    assert retriever.aclient is not None


def test_init_custom_params(mock_httpx_clients):
    retriever = SearXNGWebRetriever(
        base_url="http://localhost:8888/",
        categories="general,news",
        engines="duckduckgo,wikipedia",
        language="en",
        time_range="month",
        safesearch=1,
        pageno=2,
        timeout=5.0,
    )

    assert retriever.base_url == "http://localhost:8888"
    assert retriever.search_url == "http://localhost:8888/search"
    assert retriever.categories == "general,news"
    assert retriever.engines == "duckduckgo,wikipedia"
    assert retriever.language == "en"
    assert retriever.time_range == "month"
    assert retriever.safesearch == 1
    assert retriever.pageno == 2
    assert retriever.timeout == 5.0


def test_init_reads_base_url_from_env(mock_httpx_clients):
    with patch.dict(
        "os.environ",
        {"SEARXNG_BASE_URL": "http://searxng.local:8080/"},
        clear=True,
    ):
        retriever = SearXNGWebRetriever()

    assert retriever.base_url == "http://searxng.local:8080"
    assert retriever.search_url == "http://searxng.local:8080/search"


def test_init_raises_without_httpx():
    with (
        patch.object(searxng_provider, "httpx", None),
        pytest.raises(ImportError),
    ):
        SearXNGWebRetriever()


def test_build_search_params_with_filters(mock_httpx_clients):
    retriever = SearXNGWebRetriever(
        categories="general,news",
        engines="duckduckgo,wikipedia",
        language="en",
        time_range="year",
        safesearch=2,
        pageno=3,
    )

    params = retriever._build_search_params("python news")

    assert params == {
        "q": "python news",
        "format": "json",
        "categories": "general,news",
        "engines": "duckduckgo,wikipedia",
        "language": "en",
        "time_range": "year",
        "safesearch": 2,
        "pageno": 3,
    }


def test_sync_search_uses_direct_httpx_request(mock_httpx_clients):
    retriever = SearXNGWebRetriever(
        base_url="http://localhost:8888",
        categories="general",
        language="en",
    )
    mock_httpx_clients.response.json.return_value = {"results": []}

    retriever("local query", top_k=5)

    mock_httpx_clients.client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.sync_client.get.call_args
    assert call_args[0][0] == "http://localhost:8888/search"
    params = call_args[1]["params"]
    assert params["q"] == "local query"
    assert params["format"] == "json"
    assert params["categories"] == "general"
    assert params["language"] == "en"


@pytest.mark.asyncio
async def test_async_search_uses_direct_httpx_request(mock_httpx_clients):
    retriever = SearXNGWebRetriever(
        base_url="http://localhost:8888",
        engines="duckduckgo",
        time_range="month",
    )
    mock_httpx_clients.response.json.return_value = {"results": []}

    await retriever.acall("local query", top_k=5)

    mock_httpx_clients.async_client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.async_client.get.call_args
    assert call_args[0][0] == "http://localhost:8888/search"
    params = call_args[1]["params"]
    assert params["q"] == "local query"
    assert params["format"] == "json"
    assert params["engines"] == "duckduckgo"
    assert params["time_range"] == "month"


@pytest.mark.asyncio
async def test_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "query": "test query",
        "results": [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "content": "Test content",
                "engine": "duckduckgo",
                "category": "general",
            }
        ],
    }

    results = await retriever.acall("test query", top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert len(results.data[0].results) == 1

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Test Title"
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "Test content"
    assert result_data["engine"] == "duckduckgo"
    assert result_data["category"] == "general"


def test_image_search_result_includes_images(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "title": "Image Title",
                "url": "https://example.com/page",
                "content": "Image content",
                "thumbnail": "https://example.com/thumb.jpg",
            }
        ]
    }

    results = retriever("image query", top_k=1)

    result = results.data[0].results[0]
    assert result["data"]["title"] == "Image Title"
    assert result["images"] == ["https://example.com/thumb.jpg"]


def test_results_are_limited_to_top_k(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {"title": "First", "url": "https://example.com/1"},
            {"title": "Second", "url": "https://example.com/2"},
        ]
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_sync_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "title": "Sync Test",
                "url": "https://sync.com",
                "content": "Sync content",
            }
        ]
    }

    results = retriever(["sync query"], top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_close_closes_sync_client(retriever):
    retriever.close()

    retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_async_client(retriever):
    await retriever.aclose()

    retriever.aclient.aclose.assert_awaited_once()
