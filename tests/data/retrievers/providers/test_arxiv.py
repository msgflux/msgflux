"""Tests for ArxivWebRetriever."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestArxivWebRetriever:
    """Tests for ArxivWebRetriever."""

    @pytest.fixture
    def mock_arxiv(self):
        """Mock the arxiv module."""
        mock_module = MagicMock()

        # Mock SortCriterion enum
        mock_module.SortCriterion = MagicMock()
        mock_module.SortCriterion.Relevance = "relevance"
        mock_module.SortCriterion.LastUpdatedDate = "lastUpdatedDate"
        mock_module.SortCriterion.SubmittedDate = "submittedDate"

        # Mock SortOrder enum
        mock_module.SortOrder = MagicMock()
        mock_module.SortOrder.Ascending = "ascending"
        mock_module.SortOrder.Descending = "descending"

        # Mock result object
        mock_result = MagicMock()
        mock_result.title = "Deep Learning for Natural Language Processing"
        mock_result.summary = (
            "This paper presents a novel approach to NLP using deep learning."
        )
        mock_author = MagicMock()
        mock_author.name = "John Doe"
        mock_result.authors = [mock_author]
        mock_result.published = datetime(2024, 1, 15, 12, 0, 0)
        mock_result.updated = datetime(2024, 1, 20, 12, 0, 0)
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.12345.pdf"
        mock_result.entry_id = "http://arxiv.org/abs/2401.12345v1"
        mock_result.categories = ["cs.CL", "cs.LG"]

        # Mock Search class
        mock_module.Search = MagicMock()

        # Mock Client class
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_module.Client.return_value = mock_client

        with patch.dict("sys.modules", {"arxiv": mock_module}):
            # Import to get the mocked version
            from msgflux.data.retrievers.providers.arxiv import ArxivWebRetriever

            yield ArxivWebRetriever

    def test_init_without_library_raises_error(self):
        """Test that initialization fails without arxiv."""
        with patch.dict("sys.modules", {"arxiv": None}):
            # Force reimport
            if "msgflux.data.retrievers.providers.arxiv" in sys.modules:
                del sys.modules["msgflux.data.retrievers.providers.arxiv"]

            from msgflux.data.retrievers.providers.arxiv import ArxivWebRetriever

            with pytest.raises(ImportError) as exc_info:
                ArxivWebRetriever()

            assert "arxiv" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

    def test_init_with_defaults(self, mock_arxiv):
        """Test initialization with default parameters."""
        retriever = mock_arxiv()
        assert retriever.max_results == 10
        assert retriever.sort_by == "relevance"
        assert retriever.sort_order == "descending"

    def test_init_with_custom_params(self, mock_arxiv):
        """Test initialization with custom parameters."""
        retriever = mock_arxiv(
            max_results=5, sort_by="submittedDate", sort_order="ascending"
        )
        assert retriever.max_results == 5
        assert retriever.sort_by == "submittedDate"
        assert retriever.sort_order == "ascending"

    def test_init_with_none_applies_defaults(self, mock_arxiv):
        """Test that None values are replaced with defaults."""
        retriever = mock_arxiv(max_results=None, sort_by=None, sort_order=None)
        assert retriever.max_results == 10
        assert retriever.sort_by == "relevance"
        assert retriever.sort_order == "descending"

    def test_invalid_sort_by_raises_error(self, mock_arxiv):
        """Test that invalid sort_by value raises error."""
        with pytest.raises(ValueError) as exc_info:
            mock_arxiv(sort_by="invalid")

        assert "invalid sort_by" in str(exc_info.value).lower()

    def test_invalid_sort_order_raises_error(self, mock_arxiv):
        """Test that invalid sort_order value raises error."""
        with pytest.raises(ValueError) as exc_info:
            mock_arxiv(sort_order="invalid")

        assert "invalid sort_order" in str(exc_info.value).lower()

    def test_format_result(self, mock_arxiv):
        """Test result formatting."""
        retriever = mock_arxiv()

        # Create a mock result
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.summary = "Test summary"
        mock_author = MagicMock()
        mock_author.name = "Author Name"
        mock_result.authors = [mock_author]
        mock_result.published = datetime(2024, 1, 15)
        mock_result.updated = datetime(2024, 1, 20)
        mock_result.pdf_url = "https://arxiv.org/pdf/test.pdf"
        mock_result.entry_id = "http://arxiv.org/abs/test"
        mock_result.categories = ["cs.AI"]

        formatted = retriever._format_result(mock_result)

        assert "data" in formatted
        assert formatted["data"]["title"] == "Test Paper"
        assert formatted["data"]["summary"] == "Test summary"
        assert formatted["data"]["authors"] == ["Author Name"]
        assert formatted["data"]["pdf_url"] == "https://arxiv.org/pdf/test.pdf"
        assert formatted["data"]["categories"] == ["cs.AI"]

    def test_search_single_query(self, mock_arxiv):
        """Test search with a single query."""
        retriever = mock_arxiv()
        results = retriever(["machine learning"], top_k=1)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) >= 0

    def test_search_multiple_queries(self, mock_arxiv):
        """Test search with multiple queries."""
        retriever = mock_arxiv()
        results = retriever(["machine learning", "deep learning"], top_k=1)

        assert len(results.data) == 2

    @pytest.mark.asyncio
    async def test_acall_single_query(self, mock_arxiv):
        """Test async search with a single query."""
        retriever = mock_arxiv()
        results = await retriever.acall(["machine learning"], top_k=1)

        assert len(results.data) == 1

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, mock_arxiv):
        """Test async search applies defaults correctly."""
        retriever = mock_arxiv()
        results = await retriever.acall("machine learning")

        assert len(results.data) == 1

    @pytest.mark.asyncio
    async def test_acall_multiple_queries(self, mock_arxiv):
        """Test async search with multiple queries."""
        retriever = mock_arxiv()
        results = await retriever.acall(["neural networks", "transformers"], top_k=2)

        assert len(results.data) == 2
