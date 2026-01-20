import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the module before import
mock_tavily_module = MagicMock()
mock_tavily_module.TavilyClient = MagicMock()
mock_tavily_module.AsyncTavilyClient = MagicMock()
sys.modules["tavily"] = mock_tavily_module

from msgflux.data.retrievers.providers.tavily import TavilyWebRetriever


@pytest.fixture
def mock_tavily_client():
    sync_client = MagicMock()
    async_client = AsyncMock()
    mock_tavily_module.TavilyClient.return_value = sync_client
    mock_tavily_module.AsyncTavilyClient.return_value = async_client
    return sync_client, async_client


@pytest.fixture
def retriever(mock_tavily_client):
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        return TavilyWebRetriever()


def test_init_defaults(retriever):
    assert retriever.search_depth == "basic"
    assert retriever.topic == "general"
    assert retriever.include_answer is False
    assert retriever.include_images is False
    assert retriever.include_raw_content is False


def test_init_custom_params(mock_tavily_client):
    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        retriever = TavilyWebRetriever(
            search_depth="advanced",
            topic="news",
            time_range="week",
            include_domains=["cnn.com"],
            exclude_domains=["spam.com"],
            include_answer=True,
            include_images=True,
        )
        assert retriever.search_depth == "advanced"
        assert retriever.topic == "news"
        assert retriever.time_range == "week"
        assert retriever.include_domains == ["cnn.com"]
        assert retriever.exclude_domains == ["spam.com"]
        assert retriever.include_answer is True
        assert retriever.include_images is True


@pytest.mark.asyncio
async def test_search(mock_tavily_client):
    sync_client, async_client = mock_tavily_client

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        retriever = TavilyWebRetriever()

        # Mock response
        mock_response = {
            "results": [
                {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "content": "Test content",
                    "score": 0.95,
                    "published_date": "2024-01-15",
                }
            ]
        }

        async_client.search.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert len(results.data[0].results) == 1

        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "Test Title"
        assert result_data["url"] == "https://example.com"
        assert result_data["content"] == "Test content"
        assert result_data["date"] == "2024-01-15"
        assert results.data[0].results[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_search_with_filters(mock_tavily_client):
    sync_client, async_client = mock_tavily_client

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        retriever = TavilyWebRetriever(
            search_depth="advanced",
            topic="news",
            include_domains=["cnn.com", "bbc.com"],
        )

        async_client.search.return_value = {"results": []}

        await retriever.acall("news query", top_k=5)

        # Verify search was called with correct params
        call_kwargs = async_client.search.call_args[1]
        assert call_kwargs["search_depth"] == "advanced"
        assert call_kwargs["topic"] == "news"
        assert call_kwargs["include_domains"] == ["cnn.com", "bbc.com"]
        assert call_kwargs["max_results"] == 5


def test_sync_search(mock_tavily_client):
    sync_client, async_client = mock_tavily_client

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
        retriever = TavilyWebRetriever()

        mock_response = {
            "results": [
                {
                    "title": "Sync Test",
                    "url": "https://sync.com",
                    "content": "Sync content",
                    "score": 0.90,
                }
            ]
        }

        sync_client.search.return_value = mock_response

        results = retriever(["sync query"], top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            TavilyWebRetriever()
