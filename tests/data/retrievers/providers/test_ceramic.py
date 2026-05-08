from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import msgflux.data.retrievers.providers.ceramic as ceramic_provider
from msgflux.data.retrievers.providers.ceramic import CeramicWebRetriever


@pytest.fixture
def mock_httpx_clients():
    response = MagicMock()
    response.json.return_value = {}
    response.raise_for_status.return_value = None

    sync_client = MagicMock()
    sync_client.post.return_value = response

    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=response)
    async_client.aclose = AsyncMock(return_value=None)

    client_cls = MagicMock(return_value=sync_client)
    async_client_cls = MagicMock(return_value=async_client)
    fake_httpx = SimpleNamespace(Client=client_cls, AsyncClient=async_client_cls)

    with patch.object(ceramic_provider, "httpx", fake_httpx):
        yield SimpleNamespace(
            response=response,
            client_cls=client_cls,
            sync_client=sync_client,
            async_client_cls=async_client_cls,
            async_client=async_client,
        )


@pytest.fixture
def retriever(mock_httpx_clients):
    with patch.dict("os.environ", {"CERAMIC_API_KEY": "test_key"}):
        yield CeramicWebRetriever()


def test_init_defaults(retriever):
    assert not hasattr(retriever, "api_key")
    assert retriever.timeout == 30.0
    assert retriever.client is not None
    assert retriever.aclient is not None


def test_init_custom_timeout(mock_httpx_clients):
    with patch.dict("os.environ", {"CERAMIC_API_KEY": "test_key"}):
        retriever = CeramicWebRetriever(timeout=5.0)

    assert retriever.timeout == 5.0


def test_reads_ceramic_api_key(mock_httpx_clients):
    with patch.dict("os.environ", {"CERAMIC_API_KEY": "test_key"}, clear=True):
        retriever = CeramicWebRetriever()
        assert retriever._get_api_key() == "test_key"


def test_init_raises_without_httpx():
    with (
        patch.object(ceramic_provider, "httpx", None),
        patch.dict("os.environ", {"CERAMIC_API_KEY": "test_key"}),
        pytest.raises(ImportError),
    ):
        CeramicWebRetriever()


def test_sync_search_uses_direct_httpx_request(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {"result": {"results": []}}

    retriever("California rental laws", top_k=5)

    mock_httpx_clients.client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.sync_client.post.call_args
    assert call_args[0][0] == ceramic_provider.CERAMIC_SEARCH_URL
    assert call_args[1]["headers"] == {
        "Authorization": "Bearer test_key",
        "Content-Type": "application/json",
    }
    assert call_args[1]["json"] == {"query": "California rental laws"}


def test_sync_search_reads_api_key_per_call(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {"result": {"results": []}}

    with patch.dict("os.environ", {"CERAMIC_API_KEY": "rotated_key"}):
        retriever("California rental laws", top_k=1)

    call_args = mock_httpx_clients.sync_client.post.call_args
    assert call_args[1]["headers"]["Authorization"] == "Bearer rotated_key"


@pytest.mark.asyncio
async def test_async_search_uses_direct_httpx_request(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {"result": {"results": []}}

    await retriever.acall("California rental laws", top_k=5)

    mock_httpx_clients.async_client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.async_client.post.call_args
    assert call_args[0][0] == ceramic_provider.CERAMIC_SEARCH_URL
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"
    assert call_args[1]["json"] == {"query": "California rental laws"}


@pytest.mark.asyncio
async def test_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "requestId": "request-id",
        "result": {
            "results": [
                {
                    "title": "California Tenant Rights Guide",
                    "url": "https://example.com/tenant-rights",
                    "description": "Comprehensive guide to California rental laws.",
                }
            ],
            "searchMetadata": {"executionTime": 0.097},
            "totalResults": 10,
        },
    }

    results = await retriever.acall("California rental laws", top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert len(results.data[0].results) == 1

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "California Tenant Rights Guide"
    assert result_data["url"] == "https://example.com/tenant-rights"
    assert result_data["content"] == "Comprehensive guide to California rental laws."


def test_results_are_limited_to_top_k(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "result": {
            "results": [
                {
                    "title": "First",
                    "url": "https://example.com/1",
                    "description": "First result.",
                },
                {
                    "title": "Second",
                    "url": "https://example.com/2",
                    "description": "Second result.",
                },
            ]
        }
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_sync_search(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "result": {
            "results": [
                {
                    "title": "Sync Test",
                    "url": "https://sync.com",
                    "description": "Sync content",
                }
            ]
        }
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


def test_search_raises_without_api_key_at_call_time(mock_httpx_clients):
    with patch.dict("os.environ", {}, clear=True):
        retriever = CeramicWebRetriever()

    assert retriever("query").data[0].results == []
