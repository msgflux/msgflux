from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import msgflux.data.retrievers.http as http_module
import msgflux.data.retrievers.providers.exa as exa_provider
from msgflux.data.retrievers.providers.exa import ExaWebRetriever


@pytest.fixture(autouse=True)
def _api_key():
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        yield


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
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        yield ExaWebRetriever()


def test_init_defaults(retriever):
    assert retriever.search_type == "auto"
    assert retriever.include_text is True
    assert retriever.include_domains is None
    assert retriever.exclude_domains is None
    assert retriever.max_characters is None
    assert retriever.base_url == exa_provider.DEFAULT_BASE_URL
    assert retriever.search_url == "https://api.exa.ai/search"
    assert retriever.timeout == 30.0
    assert retriever.client is not None
    assert retriever.aclient is not None
    assert not hasattr(retriever, "api_key")


def test_init_custom_params(mock_httpx_clients):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        retriever = ExaWebRetriever(
            search_type="neural",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            include_text=True,
            max_characters=500,
            base_url="https://exa.example.com/",
            timeout=5.0,
        )

    assert retriever.search_type == "neural"
    assert retriever.include_domains == ["example.com"]
    assert retriever.exclude_domains == ["spam.com"]
    assert retriever.start_published_date == "2024-01-01"
    assert retriever.end_published_date == "2024-12-31"
    assert retriever.max_characters == 500
    assert retriever.base_url == "https://exa.example.com"
    assert retriever.search_url == "https://exa.example.com/search"
    assert retriever.timeout == 5.0


def test_sync_search_uses_direct_httpx_request(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "text": "Test content text",
                "publishedDate": "2024-01-15",
            }
        ]
    }

    results = retriever("test query", top_k=2)

    mock_httpx_clients.client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.sync_client.post.call_args
    assert call_args[0][0] == "https://api.exa.ai/search"
    assert call_args[1]["headers"] == {
        "x-api-key": "test_key",
        "Content-Type": "application/json",
    }
    assert call_args[1]["json"] == {
        "query": "test query",
        "numResults": 2,
        "type": "auto",
        "contents": {"text": True},
    }

    assert results.response_type == "web_search"
    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Test Title"
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "Test content text"
    assert result_data["date"] == "2024-01-15"
    mock_httpx_clients.async_client.post.assert_not_called()


def test_sync_search_preserves_filters_and_text_options(mock_httpx_clients):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        retriever = ExaWebRetriever(
            search_type="deep",
            include_domains=["cnn.com", "bbc.com"],
            exclude_domains=["spam.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            max_characters=500,
        )

    mock_httpx_clients.response.json.return_value = {"results": []}

    retriever("news query", top_k=7)

    payload = mock_httpx_clients.sync_client.post.call_args[1]["json"]
    assert payload == {
        "query": "news query",
        "numResults": 7,
        "type": "deep",
        "includeDomains": ["cnn.com", "bbc.com"],
        "excludeDomains": ["spam.com"],
        "startPublishedDate": "2024-01-01",
        "endPublishedDate": "2024-12-31",
        "contents": {"text": {"maxCharacters": 500}},
    }


def test_sync_search_without_text_omits_contents(mock_httpx_clients):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        retriever = ExaWebRetriever(include_text=False)

    mock_httpx_clients.response.json.return_value = {"results": []}

    retriever("query", top_k=1)

    payload = mock_httpx_clients.sync_client.post.call_args[1]["json"]
    assert "contents" not in payload


def test_sync_search_reads_api_key_per_call(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {"results": []}

    with patch.dict("os.environ", {"EXA_API_KEY": "rotated_key"}, clear=True):
        retriever("query", top_k=1)

    headers = mock_httpx_clients.sync_client.post.call_args[1]["headers"]
    assert headers["x-api-key"] == "rotated_key"


@pytest.mark.asyncio
async def test_async_search_uses_direct_httpx_request(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "title": "Async Title",
                "url": "https://async.example.com",
                "text": "Async content",
            }
        ]
    }

    results = await retriever.acall("async query", top_k=1)

    mock_httpx_clients.async_client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.async_client.post.call_args
    assert call_args[0][0] == "https://api.exa.ai/search"
    assert call_args[1]["headers"]["x-api-key"] == "test_key"
    assert call_args[1]["json"]["query"] == "async query"
    assert call_args[1]["json"]["numResults"] == 1
    assert results.data[0].results[0]["data"]["title"] == "Async Title"
    mock_httpx_clients.sync_client.post.assert_not_called()


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


def test_search_returns_empty_results_on_http_error(retriever, mock_httpx_clients):
    mock_httpx_clients.response.raise_for_status.side_effect = RuntimeError("boom")

    with patch.object(exa_provider.logger, "warning") as warning:
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    warning.assert_called_once()
    assert "Exa search failed" in warning.call_args[0][0]


def test_search_returns_empty_results_without_api_key(mock_httpx_clients):
    with patch.dict("os.environ", {}, clear=True):
        retriever = ExaWebRetriever()
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    mock_httpx_clients.sync_client.post.assert_not_called()


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
    assert state["search_type"] == "auto"
    assert state["search_url"] == "https://api.exa.ai/search"
    assert state["timeout"] == 30.0


def test_from_serialized_rebuilds_clients(mock_httpx_clients):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        retriever = ExaWebRetriever(timeout=5.0, include_domains=["example.com"])

    restored = ExaWebRetriever.from_serialized(retriever.serialize()["state"])

    assert restored.timeout == 5.0
    assert restored.include_domains == ["example.com"]
    assert restored.client is mock_httpx_clients.sync_client
    assert restored.aclient is mock_httpx_clients.async_client


def test_initialize_closes_sync_client_when_async_client_fails(mock_httpx_clients):
    sync_client = MagicMock()
    mock_httpx_clients.client_cls.return_value = sync_client
    mock_httpx_clients.async_client_cls.side_effect = RuntimeError("async boom")

    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}, clear=True):
        with pytest.raises(RuntimeError, match="async boom"):
            ExaWebRetriever()

    sync_client.close.assert_called_once()
