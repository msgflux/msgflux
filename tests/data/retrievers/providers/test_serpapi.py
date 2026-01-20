import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock the module before import
mock_serpapi_module = MagicMock()
mock_google_search = MagicMock()
mock_serpapi_module.GoogleSearch = mock_google_search
sys.modules["serpapi"] = mock_serpapi_module

from msgflux.data.retrievers.providers.serpapi import SerpApiWebRetriever


@pytest.fixture
def mock_google_search_instance():
    search_instance = MagicMock()
    mock_google_search.return_value = search_instance
    return search_instance


@pytest.fixture
def retriever(mock_google_search_instance):
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "test_key"}):
        return SerpApiWebRetriever()


def test_init_defaults(retriever):
    assert retriever.engine == "google"
    assert retriever.location is None
    assert retriever.gl is None
    assert retriever.hl is None
    assert retriever.safe is None
    assert retriever.tbm is None


def test_init_custom_params(mock_google_search_instance):
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "test_key"}):
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


@pytest.mark.asyncio
async def test_organic_search(mock_google_search_instance):
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "test_key"}):
        retriever = SerpApiWebRetriever()

        # Mock response
        mock_response = {
            "organic_results": [
                {
                    "title": "Test Title",
                    "link": "https://example.com",
                    "snippet": "Test snippet content",
                    "date": "2024-01-15",
                }
            ]
        }

        mock_google_search_instance.get_dict.return_value = mock_response

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
async def test_news_search(mock_google_search_instance):
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(tbm="nws")

        mock_response = {
            "news_results": [
                {
                    "title": "News Title",
                    "link": "https://news.com",
                    "snippet": "News snippet",
                    "date": "2 hours ago",
                }
            ]
        }

        mock_google_search_instance.get_dict.return_value = mock_response

        results = await retriever.acall("news query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data[0].results) == 1
        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "News Title"


@pytest.mark.asyncio
async def test_search_with_location(mock_google_search_instance):
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "test_key"}):
        retriever = SerpApiWebRetriever(
            location="Austin,Texas",
            gl="us",
        )

        mock_response = {"organic_results": []}
        mock_google_search_instance.get_dict.return_value = mock_response

        await retriever.acall("local query", top_k=5)

        # Verify GoogleSearch was called with correct params
        call_args = mock_google_search.call_args[0][0]
        assert call_args["location"] == "Austin,Texas"
        assert call_args["gl"] == "us"
        assert call_args["num"] == 5


def test_sync_search(mock_google_search_instance):
    with patch.dict("os.environ", {"SERPAPI_API_KEY": "test_key"}):
        retriever = SerpApiWebRetriever()

        mock_response = {
            "organic_results": [
                {
                    "title": "Sync Test",
                    "link": "https://sync.com",
                    "snippet": "Sync content",
                }
            ]
        }

        mock_google_search_instance.get_dict.return_value = mock_response

        results = retriever(["sync query"], top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            SerpApiWebRetriever()
