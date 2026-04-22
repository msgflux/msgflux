from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import msgflux.data.retrievers.providers.serpapi as serpapi_provider
from msgflux.data.retrievers.providers.serpapi import SerpApiWebRetriever


@pytest.fixture
def mock_httpx_clients():
    response = MagicMock()
    response.json.return_value = {}
    response.raise_for_status.return_value = None

    sync_client = MagicMock()
    sync_client.get.return_value = response
    sync_context = MagicMock()
    sync_context.__enter__.return_value = sync_client
    sync_context.__exit__.return_value = None

    async_client = MagicMock()
    async_client.get = AsyncMock(return_value=response)
    async_context = MagicMock()
    async_context.__aenter__ = AsyncMock(return_value=async_client)
    async_context.__aexit__ = AsyncMock(return_value=None)

    client_cls = MagicMock(return_value=sync_context)
    async_client_cls = MagicMock(return_value=async_context)
    fake_httpx = SimpleNamespace(Client=client_cls, AsyncClient=async_client_cls)

    with patch.object(serpapi_provider, "httpx", fake_httpx):
        yield SimpleNamespace(
            response=response,
            client_cls=client_cls,
            sync_client=sync_client,
            async_client_cls=async_client_cls,
            async_client=async_client,
        )


@pytest.fixture
def retriever(mock_httpx_clients):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        return SerpApiWebRetriever()


def test_init_defaults(retriever):
    assert retriever.engine == "google"
    assert retriever.location is None
    assert retriever.gl is None
    assert retriever.hl is None
    assert retriever.safe is None
    assert retriever.tbm is None


def test_init_custom_params(mock_httpx_clients):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(
            engine="google",
            location="Austin,Texas",
            gl="us",
            hl="en",
            safe="active",
            tbm="nws",
        )
        assert retriever.location == "Austin,Texas"
        assert retriever.gl == "us"
        assert retriever.hl == "en"
        assert retriever.safe == "active"
        assert retriever.tbm == "nws"


def test_init_accepts_legacy_env_names(mock_httpx_clients):
    with patch.dict("os.environ", {"SERP_API_KEY": "test_key"}, clear=True):
        retriever = SerpApiWebRetriever()

    assert retriever.api_key == "test_key"


def test_init_raises_without_httpx():
    with (
        patch.object(serpapi_provider, "httpx", None),
        patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}),
        pytest.raises(ImportError),
    ):
        SerpApiWebRetriever()


def test_sync_search_uses_direct_httpx_request(mock_httpx_clients):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(
            engine="bing",
            location="Austin,Texas",
            gl="us",
        )

    mock_httpx_clients.response.json.return_value = {"organic_results": []}

    retriever("local query", top_k=5)

    call_args = mock_httpx_clients.sync_client.get.call_args
    assert call_args[0][0] == serpapi_provider.SERPAPI_SEARCH_URL
    params = call_args[1]["params"]
    assert params["api_key"] == "test_key"
    assert params["engine"] == "bing"
    assert params["q"] == "local query"
    assert params["location"] == "Austin,Texas"
    assert params["gl"] == "us"
    assert params["num"] == 5


@pytest.mark.asyncio
async def test_async_search_uses_direct_httpx_request(mock_httpx_clients):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(
            engine="bing",
            location="Austin,Texas",
            gl="us",
        )

    mock_httpx_clients.response.json.return_value = {"organic_results": []}

    await retriever.acall("local query", top_k=5)

    call_args = mock_httpx_clients.async_client.get.call_args
    assert call_args[0][0] == serpapi_provider.SERPAPI_SEARCH_URL
    params = call_args[1]["params"]
    assert params["api_key"] == "test_key"
    assert params["engine"] == "bing"
    assert params["q"] == "local query"
    assert params["location"] == "Austin,Texas"
    assert params["gl"] == "us"
    assert params["num"] == 5


@pytest.mark.asyncio
async def test_organic_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "organic_results": [
            {
                "title": "Test Title",
                "link": "https://example.com",
                "snippet": "Test snippet content",
                "date": "2024-01-15",
            }
        ]
    }

    results = await retriever.acall("test query", top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert len(results.data[0].results) == 1

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Test Title"
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "Test snippet content"
    assert result_data["date"] == "2024-01-15"


@pytest.mark.asyncio
async def test_news_search(mock_httpx_clients):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(tbm="nws")

    mock_httpx_clients.response.json.return_value = {
        "news_results": [
            {
                "title": "News Title",
                "link": "https://news.com",
                "snippet": "News snippet",
                "date": "2 hours ago",
            }
        ]
    }

    results = await retriever.acall("news query", top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data[0].results) == 1
    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "News Title"


def test_search_uses_engine_specific_query_param(mock_httpx_clients):
    with patch.dict("os.environ", {"SERPAPI_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(engine="yahoo")

    params = retriever._build_search_params("query", top_k=1)

    assert params["api_key"] == "test_key"
    assert params["engine"] == "yahoo"
    assert params["p"] == "query"
    assert "q" not in params


def test_image_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "images_results": [
            {
                "title": "Image Title",
                "link": "https://example.com/page",
                "original": "https://example.com/image.jpg",
                "thumbnail": "https://example.com/thumb.jpg",
                "source": "Example",
            }
        ]
    }

    results = retriever("image query", top_k=1)

    result = results.data[0].results[0]
    assert result["data"]["title"] == "Image Title"
    assert result["data"]["url"] == "https://example.com/page"
    assert result["images"] == ["https://example.com/thumb.jpg"]


def test_shopping_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "shopping_results": [
            {
                "title": "Product Title",
                "link": "https://shop.example/product",
                "source": "Shop",
                "price": "$10.00",
            }
        ]
    }

    results = retriever("shopping query", top_k=1)

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Product Title"
    assert result_data["url"] == "https://shop.example/product"
    assert result_data["price"] == "$10.00"


def test_results_are_limited_to_top_k(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "organic_results": [
            {"title": "First", "link": "https://example.com/1"},
            {"title": "Second", "link": "https://example.com/2"},
        ]
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_sync_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "organic_results": [
            {
                "title": "Sync Test",
                "link": "https://sync.com",
                "snippet": "Sync content",
            }
        ]
    }

    results = retriever(["sync query"], top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_init_raises_without_api_key(mock_httpx_clients):
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            SerpApiWebRetriever()
