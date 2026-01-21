import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock httpx and trafilatura BEFORE importing the retriever
mock_httpx = MagicMock()
mock_trafilatura = MagicMock()
sys.modules["httpx"] = mock_httpx
sys.modules["trafilatura"] = mock_trafilatura

# Reload the retriever module to pick up the mock
if "msgflux.data.retrievers.providers.trafilatura" in sys.modules:
    importlib.reload(sys.modules["msgflux.data.retrievers.providers.trafilatura"])

from msgflux.data.retrievers.providers.trafilatura import TrafilaturaWebRetriever


@pytest.fixture
def mock_metadata():
    metadata = MagicMock()
    metadata.title = "Test Page"
    metadata.author = "Test Author"
    metadata.date = "2026-01-20"
    return metadata


@pytest.fixture
def retriever():
    return TrafilaturaWebRetriever()


def test_init_defaults(retriever):
    assert retriever.include_comments is False
    assert retriever.include_tables is True
    assert retriever.output_format == "txt"
    assert retriever.timeout == 30.0
    assert retriever.follow_redirects is True


def test_init_custom_params():
    retriever = TrafilaturaWebRetriever(
        include_comments=True,
        include_tables=False,
        output_format="markdown",
        timeout=60.0,
        follow_redirects=False,
    )
    assert retriever.include_comments is True
    assert retriever.include_tables is False
    assert retriever.output_format == "markdown"
    assert retriever.timeout == 60.0
    assert retriever.follow_redirects is False


def test_extract_content(retriever, mock_metadata):
    mock_trafilatura.extract.return_value = "Extracted text content"
    mock_trafilatura.extract_metadata.return_value = mock_metadata

    result = retriever._extract_content("<html>test</html>", "https://example.com")

    assert result["url"] == "https://example.com"
    assert result["content"] == "Extracted text content"
    assert result["title"] == "Test Page"
    assert result["author"] == "Test Author"
    assert result["date"] == "2026-01-20"
    assert result["success"] is True


def test_sync_fetch(retriever, mock_metadata):
    mock_response = MagicMock()
    mock_response.text = "<html>test</html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get.return_value = mock_response

    mock_httpx.Client.return_value = mock_client
    mock_trafilatura.extract.return_value = "Content"
    mock_trafilatura.extract_metadata.return_value = mock_metadata

    results = retriever(["https://example.com"])

    assert results.response_type == "web_fetch"
    assert len(results.data) == 1


@pytest.mark.asyncio
async def test_async_fetch(retriever, mock_metadata):
    mock_response = MagicMock()
    mock_response.text = "<html>test</html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)

    mock_httpx.AsyncClient.return_value = mock_client
    mock_trafilatura.extract.return_value = "Async content"
    mock_trafilatura.extract_metadata.return_value = mock_metadata

    results = await retriever.acall(["https://example.com"])

    assert results.response_type == "web_fetch"
    assert len(results.data) == 1


@pytest.mark.asyncio
async def test_fetch_multiple_urls(retriever, mock_metadata):
    mock_response = MagicMock()
    mock_response.text = "<html>test</html>"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)

    mock_httpx.AsyncClient.return_value = mock_client
    mock_trafilatura.extract.return_value = "Content"
    mock_trafilatura.extract_metadata.return_value = mock_metadata

    results = await retriever.acall(
        ["https://example.com", "https://python.org"]
    )

    assert results.response_type == "web_fetch"
    assert len(results.data) == 2
