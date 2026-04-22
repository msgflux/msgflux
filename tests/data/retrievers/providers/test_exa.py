import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the module before import
mock_exa_module = MagicMock()
sys.modules["exa_py"] = mock_exa_module

import msgflux.data.retrievers.providers.exa as exa_provider
from msgflux.data.retrievers.providers.exa import ExaWebRetriever


@pytest.fixture
def mock_exa_clients():
    sync_client = MagicMock()
    async_client = MagicMock()
    async_client.search = AsyncMock()
    async_client.search_and_contents = AsyncMock()

    sync_factory = MagicMock(return_value=sync_client)
    async_factory = MagicMock(return_value=async_client)

    with (
        patch.object(exa_provider, "Exa", sync_factory),
        patch.object(exa_provider, "AsyncExa", async_factory),
    ):
        yield sync_client, async_client


@pytest.fixture
def mock_exa_client(mock_exa_clients):
    return mock_exa_clients[0]


@pytest.fixture
def mock_async_exa_client(mock_exa_clients):
    return mock_exa_clients[1]


@pytest.fixture
def retriever(mock_exa_clients):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        return ExaWebRetriever()


def test_init_defaults(retriever):
    assert retriever.search_type == "auto"
    assert retriever.include_text is True
    assert retriever.include_domains is None
    assert retriever.exclude_domains is None
    assert retriever.max_characters is None


def test_init_custom_params(mock_exa_clients):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(
            search_type="neural",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
            include_text=True,
            max_characters=500,
        )
        assert retriever.search_type == "neural"
        assert retriever.include_domains == ["example.com"]
        assert retriever.exclude_domains == ["spam.com"]
        assert retriever.max_characters == 500


@pytest.mark.asyncio
async def test_search_with_text(retriever, mock_exa_client, mock_async_exa_client):
    mock_result = SimpleNamespace(
        title="Test Title",
        url="https://example.com",
        text="Test content text",
        published_date="2024-01-15",
    )
    mock_response = SimpleNamespace(results=[mock_result])

    mock_async_exa_client.search_and_contents.return_value = mock_response

    results = await retriever.acall("test query", top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert len(results.data[0].results) == 1

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Test Title"
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "Test content text"
    assert result_data["date"] == "2024-01-15"

    mock_async_exa_client.search_and_contents.assert_called_once()
    mock_exa_client.search_and_contents.assert_not_called()


@pytest.mark.asyncio
async def test_search_without_text(mock_exa_clients):
    sync_client, async_client = mock_exa_clients
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(include_text=False)

        mock_result = SimpleNamespace(
            title="Test Title",
            url="https://example.com",
        )
        mock_response = SimpleNamespace(results=[mock_result])

        async_client.search.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data[0].results) == 1

        async_client.search.assert_called_once()
        async_client.search_and_contents.assert_not_called()
        sync_client.search.assert_not_called()


@pytest.mark.asyncio
async def test_search_with_domain_filters(mock_async_exa_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(
            include_domains=["cnn.com", "bbc.com"],
            exclude_domains=["spam.com"],
        )

        mock_response = SimpleNamespace(results=[])
        mock_async_exa_client.search_and_contents.return_value = mock_response

        await retriever.acall("news query", top_k=5)

        call_kwargs = mock_async_exa_client.search_and_contents.call_args[1]
        assert call_kwargs["include_domains"] == ["cnn.com", "bbc.com"]
        assert call_kwargs["exclude_domains"] == ["spam.com"]


@pytest.mark.asyncio
async def test_search_with_date_filters(mock_async_exa_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
        )

        mock_response = SimpleNamespace(results=[])
        mock_async_exa_client.search_and_contents.return_value = mock_response

        await retriever.acall("recent news", top_k=3)

        call_kwargs = mock_async_exa_client.search_and_contents.call_args[1]
        assert call_kwargs["start_published_date"] == "2024-01-01"
        assert call_kwargs["end_published_date"] == "2024-12-31"


def test_sync_search(retriever, mock_exa_client):
    mock_result = SimpleNamespace(
        title="Sync Test",
        url="https://sync.com",
        text="Sync content",
    )
    mock_response = SimpleNamespace(results=[mock_result])

    mock_exa_client.search_and_contents.return_value = mock_response

    # Use sync __call__
    results = retriever(["sync query"], top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert results.data[0].results[0]["data"]["title"] == "Sync Test"
