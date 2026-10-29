from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import msgflux.data.retrievers.http as http_module
import msgflux.data.retrievers.providers.brave as brave_provider
from msgflux.data.retrievers.providers.brave import BraveWebRetriever


@pytest.fixture(autouse=True)
def _api_key():
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}, clear=True):
        yield


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

    with patch.object(http_module, "httpx2", fake_httpx):
        yield SimpleNamespace(
            response=response,
            client_cls=client_cls,
            sync_client=sync_client,
            async_client_cls=async_client_cls,
            async_client=async_client,
        )


@pytest.fixture
def retriever(mock_httpx_clients):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}, clear=True):
        yield BraveWebRetriever()


def test_init_defaults(retriever):
    assert retriever.mode == "search"
    assert retriever.return_images is False
    assert retriever.base_url == brave_provider.DEFAULT_BASE_URL
    assert retriever.timeout == 30.0
    assert retriever.client is not None
    assert retriever.aclient is not None
    assert not hasattr(retriever, "api_key")


def test_init_custom_params(mock_httpx_clients):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}, clear=True):
        retriever = BraveWebRetriever(
            mode="news",
            return_images=True,
            base_url="https://brave.example.com/",
            timeout=5.0,
        )

    assert retriever.mode == "news"
    assert retriever.return_images is True
    assert retriever.base_url == "https://brave.example.com"
    assert retriever.timeout == 5.0


def test_sync_web_search_uses_direct_httpx_request(retriever, mock_httpx_clients):
    retriever.return_images = True
    mock_httpx_clients.response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "Test Title",
                    "description": "Test Content",
                    "url": "http://test.com",
                    "thumbnail": {"src": "img.jpg"},
                }
            ]
        }
    }

    results = retriever("query", top_k=1)

    mock_httpx_clients.client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.sync_client.get.call_args
    assert call_args[0][0] == "https://api.search.brave.com/res/v1/web/search"
    assert call_args[1]["headers"] == {
        "Accept": "application/json",
        "X-Subscription-Token": "test_key",
    }
    assert call_args[1]["params"] == {"q": "query", "count": 1}

    result = results.data[0].results[0]
    assert result["data"]["title"] == "Test Title"
    assert result["data"]["content"] == "Test Content"
    assert result["data"]["url"] == "http://test.com"
    assert result["images"] == ["img.jpg"]
    mock_httpx_clients.async_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_async_news_search_uses_direct_httpx_request(
    retriever, mock_httpx_clients
):
    retriever.mode = "news"
    retriever.return_images = True
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "title": "News Title",
                "description": "News Content",
                "url": "http://news.com",
                "age": "2h",
                "thumbnail": {"src": "news.jpg"},
            }
        ]
    }

    results = await retriever.acall("news query", top_k=2)

    mock_httpx_clients.async_client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.async_client.get.call_args
    assert call_args[0][0] == "https://api.search.brave.com/res/v1/news/search"
    assert call_args[1]["headers"]["X-Subscription-Token"] == "test_key"
    assert call_args[1]["params"] == {"q": "news query", "count": 2}

    result = results.data[0].results[0]
    assert result["data"]["title"] == "News Title"
    assert result["data"]["date"] == "2h"
    assert result["images"] == ["news.jpg"]
    mock_httpx_clients.sync_client.get.assert_not_called()


def test_sync_image_search_uses_direct_httpx_request(mock_httpx_clients):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}, clear=True):
        retriever = BraveWebRetriever(mode="image")

    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "title": "Image Title",
                "url": "http://img.com",
                "thumbnail": {"src": "http://thumbnail.com"},
            }
        ]
    }

    results = retriever("image query", top_k=3)

    call_args = mock_httpx_clients.sync_client.get.call_args
    assert call_args[0][0] == "https://api.search.brave.com/res/v1/images/search"
    assert call_args[1]["params"] == {"q": "image query", "count": 3}

    result = results.data[0].results[0]
    assert result["data"]["title"] == "Image Title"
    assert result["data"]["url"] == "http://img.com"
    assert result["images"] == ["http://img.com"]


def test_sync_search_reads_api_key_per_call(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {"web": {"results": []}}

    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "rotated_key"}, clear=True):
        retriever("query", top_k=1)

    headers = mock_httpx_clients.sync_client.get.call_args[1]["headers"]
    assert headers["X-Subscription-Token"] == "rotated_key"


def test_results_are_limited_to_top_k(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "web": {
            "results": [
                {"title": "First", "url": "https://example.com/1"},
                {"title": "Second", "url": "https://example.com/2"},
            ]
        }
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_invalid_mode_returns_empty_results(retriever, mock_httpx_clients):
    retriever.mode = "videos"

    with patch.object(brave_provider.logger, "warning") as warning:
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    warning.assert_called_once()
    assert "Brave search failed" in warning.call_args[0][0]
    mock_httpx_clients.sync_client.get.assert_not_called()


def test_search_returns_empty_results_on_http_error(retriever, mock_httpx_clients):
    mock_httpx_clients.response.raise_for_status.side_effect = RuntimeError("boom")

    with patch.object(brave_provider.logger, "warning") as warning:
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    warning.assert_called_once()
    assert "Brave search failed" in warning.call_args[0][0]


def test_search_returns_empty_results_without_api_key(mock_httpx_clients):
    with patch.dict("os.environ", {}, clear=True):
        retriever = BraveWebRetriever()
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    mock_httpx_clients.sync_client.get.assert_not_called()


def test_close_closes_sync_client(retriever):
    retriever.close()

    retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_both_clients(retriever):
    await retriever.aclose()

    retriever.aclient.aclose.assert_awaited_once()
    retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_sync_client_when_async_close_fails(retriever):
    retriever.aclient.aclose.side_effect = RuntimeError("close boom")

    with pytest.raises(RuntimeError, match="close boom"):
        await retriever.aclose()

    retriever.client.close.assert_called_once()


def test_serialize_excludes_clients_and_retains_config(retriever):
    serialized = retriever.serialize()

    state = serialized["state"]
    assert "client" not in state
    assert "aclient" not in state
    assert state["mode"] == "search"
    assert state["return_images"] is False
    assert state["timeout"] == 30.0


def test_from_serialized_rebuilds_clients(mock_httpx_clients):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}, clear=True):
        retriever = BraveWebRetriever(mode="news", timeout=5.0)

    restored = BraveWebRetriever.from_serialized(retriever.serialize()["state"])

    assert restored.mode == "news"
    assert restored.timeout == 5.0
    assert restored.client is mock_httpx_clients.sync_client
    assert restored.aclient is mock_httpx_clients.async_client


def test_initialize_closes_sync_client_when_async_client_fails(mock_httpx_clients):
    sync_client = MagicMock()
    mock_httpx_clients.client_cls.return_value = sync_client
    mock_httpx_clients.async_client_cls.side_effect = RuntimeError("async boom")

    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}, clear=True):
        with pytest.raises(RuntimeError, match="async boom"):
            BraveWebRetriever()

    sync_client.close.assert_called_once()
