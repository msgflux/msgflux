import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock the module before import
mock_exa_module = MagicMock()
sys.modules["exa_py"] = mock_exa_module

from msgflux.data.retrievers.providers.exa import ExaWebRetriever


@pytest.fixture
def mock_exa_client():
    client_instance = MagicMock()
    mock_exa_module.Exa.return_value = client_instance
    return client_instance


@pytest.fixture
def retriever(mock_exa_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        return ExaWebRetriever()


def test_init_defaults(retriever):
    assert retriever.search_type == "auto"
    assert retriever.include_text is True
    assert retriever.include_domains is None
    assert retriever.exclude_domains is None
    assert retriever.max_characters is None


def test_init_custom_params(mock_exa_client):
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
async def test_search_with_text(retriever, mock_exa_client):
    # Mock response
    mock_result = MagicMock()
    mock_result.title = "Test Title"
    mock_result.url = "https://example.com"
    mock_result.text = "Test content text"
    mock_result.published_date = "2024-01-15"

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    mock_exa_client.search_and_contents.return_value = mock_response

    results = await retriever.acall("test query", top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert len(results.data[0].results) == 1

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Test Title"
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "Test content text"
    assert result_data["date"] == "2024-01-15"

    mock_exa_client.search_and_contents.assert_called_once()


@pytest.mark.asyncio
async def test_search_without_text(mock_exa_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(include_text=False)

        mock_result = MagicMock()
        mock_result.title = "Test Title"
        mock_result.url = "https://example.com"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_exa_client.search.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data[0].results) == 1

        # Should use search, not search_and_contents
        mock_exa_client.search.assert_called_once()
        mock_exa_client.search_and_contents.assert_not_called()


@pytest.mark.asyncio
async def test_search_with_domain_filters(mock_exa_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(
            include_domains=["cnn.com", "bbc.com"],
            exclude_domains=["spam.com"],
        )

        mock_response = MagicMock()
        mock_response.results = []
        mock_exa_client.search_and_contents.return_value = mock_response

        await retriever.acall("news query", top_k=5)

        # Check that domain filters were passed
        call_kwargs = mock_exa_client.search_and_contents.call_args[1]
        assert call_kwargs["include_domains"] == ["cnn.com", "bbc.com"]
        assert call_kwargs["exclude_domains"] == ["spam.com"]


@pytest.mark.asyncio
async def test_search_with_date_filters(mock_exa_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        retriever = ExaWebRetriever(
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
        )

        mock_response = MagicMock()
        mock_response.results = []
        mock_exa_client.search_and_contents.return_value = mock_response

        await retriever.acall("recent news", top_k=3)

        call_kwargs = mock_exa_client.search_and_contents.call_args[1]
        assert call_kwargs["start_published_date"] == "2024-01-01"
        assert call_kwargs["end_published_date"] == "2024-12-31"


def test_sync_search(retriever, mock_exa_client):
    mock_result = MagicMock()
    mock_result.title = "Sync Test"
    mock_result.url = "https://sync.com"
    mock_result.text = "Sync content"

    mock_response = MagicMock()
    mock_response.results = [mock_result]

    mock_exa_client.search_and_contents.return_value = mock_response

    # Use sync __call__
    results = retriever(["sync query"], top_k=1)

    assert results.response_type == "web_search"
    assert len(results.data) == 1
    assert results.data[0].results[0]["data"]["title"] == "Sync Test"
