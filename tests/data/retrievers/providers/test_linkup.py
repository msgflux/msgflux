import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the module before import
mock_linkup_module = MagicMock()
mock_linkup_module.LinkupClient = MagicMock()
sys.modules["linkup"] = mock_linkup_module

from msgflux.data.retrievers.providers.linkup import LinkupWebRetriever


@pytest.fixture
def mock_linkup_client():
    client_instance = MagicMock()
    client_instance.async_search = AsyncMock()
    mock_linkup_module.LinkupClient.return_value = client_instance
    return client_instance


@pytest.fixture
def retriever(mock_linkup_client):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}):
        return LinkupWebRetriever()


def test_init_defaults(retriever):
    assert retriever.depth == "standard"
    assert retriever.output_type == "searchResults"
    assert retriever.include_images is False
    assert retriever.include_domains is None
    assert retriever.exclude_domains is None


def test_init_custom_params(mock_linkup_client):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}):
        retriever = LinkupWebRetriever(
            depth="deep",
            output_type="sourcedAnswer",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
            include_images=True,
        )
        assert retriever.depth == "deep"
        assert retriever.output_type == "sourcedAnswer"
        assert retriever.include_domains == ["example.com"]
        assert retriever.exclude_domains == ["spam.com"]
        assert retriever.include_images is True


@pytest.mark.asyncio
async def test_search_results(mock_linkup_client):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}):
        retriever = LinkupWebRetriever(output_type="searchResults")

        # Mock response for searchResults
        mock_result = MagicMock()
        mock_result.name = "Test Title"
        mock_result.url = "https://example.com"
        mock_result.content = "Test content"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_linkup_client.async_search.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert len(results.data[0].results) == 1

        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "Test Title"
        assert result_data["url"] == "https://example.com"
        assert result_data["content"] == "Test content"


@pytest.mark.asyncio
async def test_sourced_answer(mock_linkup_client):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}):
        retriever = LinkupWebRetriever(output_type="sourcedAnswer")

        # Mock response for sourcedAnswer
        mock_source = MagicMock()
        mock_source.name = "Source Name"
        mock_source.url = "https://source.com"
        mock_source.snippet = "Source snippet"

        mock_response = MagicMock()
        mock_response.sources = [mock_source]

        mock_linkup_client.async_search.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data[0].results) == 1
        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "Source Name"
        assert result_data["content"] == "Source snippet"


@pytest.mark.asyncio
async def test_search_with_filters(mock_linkup_client):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}):
        retriever = LinkupWebRetriever(
            depth="deep",
            include_domains=["cnn.com", "bbc.com"],
        )

        mock_response = MagicMock()
        mock_response.results = []
        mock_linkup_client.async_search.return_value = mock_response

        await retriever.acall("news query", top_k=5)

        call_kwargs = mock_linkup_client.async_search.call_args[1]
        assert call_kwargs["depth"] == "deep"
        assert call_kwargs["include_domains"] == ["cnn.com", "bbc.com"]
        assert call_kwargs["max_results"] == 5


def test_sync_search(mock_linkup_client):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}):
        retriever = LinkupWebRetriever()

        mock_result = MagicMock()
        mock_result.name = "Sync Test"
        mock_result.url = "https://sync.com"
        mock_result.content = "Sync content"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_linkup_client.search.return_value = mock_response

        results = retriever(["sync query"], top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            LinkupWebRetriever()
