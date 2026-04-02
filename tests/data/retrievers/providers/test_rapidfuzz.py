"""Tests for RapidFuzzFuzzyRetriever."""

import sys
from unittest.mock import MagicMock, patch

import pytest

import msgflux.data.retrievers.providers.rapidfuzz as rapidfuzz_module
from msgflux.data.retrievers.providers.rapidfuzz import RapidFuzzFuzzyRetriever


class TestRapidFuzzFuzzyRetriever:
    """Tests for RapidFuzzFuzzyRetriever."""

    @pytest.fixture
    def mock_fuzz(self):
        """Mock the rapidfuzz.fuzz module-level variable."""
        mock = MagicMock()
        with patch.object(rapidfuzz_module, "fuzz", mock):
            yield mock

    @pytest.fixture
    def mock_process(self, mock_fuzz):
        """Mock rapidfuzz.process with two default match results."""
        mock = MagicMock()
        mock.extract.return_value = [
            ("João da Silva", 90.0, 0),
            ("Maria Oliveira", 75.0, 1),
        ]
        with patch.object(rapidfuzz_module, "process", mock):
            yield mock

    # ── Init ──────────────────────────────────────────────────────────────────

    def test_init_without_library_raises_error(self):
        """Test that initialization fails when rapidfuzz is not installed."""
        with patch.object(rapidfuzz_module, "process", None):
            with pytest.raises(ImportError) as exc_info:
                RapidFuzzFuzzyRetriever()
            assert "rapidfuzz" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

    def test_init_without_library_sys_modules(self):
        """Test ImportError path via sys.modules patch (module reimport)."""
        with patch.dict("sys.modules", {"rapidfuzz": None}):
            if "msgflux.data.retrievers.providers.rapidfuzz" in sys.modules:
                del sys.modules["msgflux.data.retrievers.providers.rapidfuzz"]

            from msgflux.data.retrievers.providers.rapidfuzz import (
                RapidFuzzFuzzyRetriever as Cls,
            )

            with pytest.raises(ImportError):
                Cls()

    def test_init_with_defaults(self, mock_process):
        """Test initialization with default parameters."""
        retriever = RapidFuzzFuzzyRetriever()
        assert retriever.scorer == "WRatio"
        assert retriever.documents == []

    def test_init_with_custom_scorer(self, mock_process):
        """Test initialization stores the given scorer."""
        retriever = RapidFuzzFuzzyRetriever(scorer="partial_ratio")
        assert retriever.scorer == "partial_ratio"

    def test_init_with_none_applies_default(self, mock_process):
        """Test that None scorer is replaced with the default WRatio."""
        retriever = RapidFuzzFuzzyRetriever(scorer=None)
        assert retriever.scorer == "WRatio"

    def test_init_with_invalid_scorer_raises_error(self, mock_process):
        """Test that an unknown scorer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scorer"):
            RapidFuzzFuzzyRetriever(scorer="invalid_scorer")

    # ── add ───────────────────────────────────────────────────────────────────

    def test_add_documents(self, mock_process):
        """Test adding documents stores them in the index."""
        retriever = RapidFuzzFuzzyRetriever()
        documents = ["hello world", "python programming", "machine learning"]
        retriever.add(documents)
        assert retriever.documents == documents

    def test_add_documents_incremental(self, mock_process):
        """Test that successive add calls accumulate documents."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["doc one"])
        retriever.add(["doc two"])
        assert len(retriever.documents) == 2
        assert retriever.documents[1] == "doc two"

    # ── __call__ ──────────────────────────────────────────────────────────────

    def test_search_returns_fuzzy_response_type(self, mock_process):
        """Test that search returns the 'fuzzy_search' response type."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva", "Maria Oliveira"])
        result = retriever("joao")
        assert result.response_type == "fuzzy_search"

    def test_search_single_query_string(self, mock_process):
        """Test search with a single query string (auto-wrapped to list)."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva", "Maria Oliveira"])
        result = retriever("joao", top_k=2)
        assert len(result.data) == 1
        assert len(result.data[0]) == 2

    def test_search_multiple_queries(self, mock_process):
        """Test search with a list of queries produces one result per query."""
        mock_process.extract.return_value = [("doc one", 80.0, 0)]
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["doc one", "doc two"])
        result = retriever(["query one", "query two"], top_k=1)
        assert len(result.data) == 2

    def test_search_with_return_score(self, mock_process):
        """Test that scores are present when return_score=True."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva"])
        result = retriever("joao", return_score=True)
        first = result.data[0][0]
        assert hasattr(first, "score")
        assert isinstance(first.score, float)
        assert first.score == 90.0

    def test_search_without_return_score(self, mock_process):
        """Test that scores are absent when return_score=False (default)."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva"])
        result = retriever("joao", return_score=False)
        first = result.data[0][0]
        assert not hasattr(first, "score")

    def test_search_passes_score_cutoff_to_extract(self, mock_process):
        """Test that threshold is forwarded to process.extract as score_cutoff."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva"])
        retriever("joao", threshold=60.0)
        kwargs = mock_process.extract.call_args.kwargs
        assert kwargs["score_cutoff"] == 60.0

    def test_search_passes_limit_to_extract(self, mock_process):
        """Test that top_k is forwarded to process.extract as limit."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["doc one", "doc two", "doc three"])
        retriever("doc", top_k=2)
        kwargs = mock_process.extract.call_args.kwargs
        assert kwargs["limit"] == 2

    def test_search_empty_corpus_returns_empty(self, mock_process):
        """Test that searching an empty corpus returns empty results."""
        retriever = RapidFuzzFuzzyRetriever()
        result = retriever("query", top_k=5)
        assert result.data == [[]]

    def test_search_result_data_matches_document(self, mock_process):
        """Test that each result's .data field contains the document text."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva", "Maria Oliveira"])
        result = retriever("joao", top_k=2)
        assert result.data[0][0].data == "João da Silva"
        assert result.data[0][1].data == "Maria Oliveira"

    # ── acall ─────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_acall_single_query(self, mock_process):
        """Test async search with a single query string."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva", "Maria Oliveira"])
        result = await retriever.acall("joao", top_k=2)
        assert len(result.data) == 1
        assert len(result.data[0]) == 2

    @pytest.mark.asyncio
    async def test_acall_multiple_queries(self, mock_process):
        """Test async search with multiple queries."""
        mock_process.extract.return_value = [("doc", 80.0, 0)]
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["doc one", "doc two"])
        result = await retriever.acall(["query one", "query two"], top_k=1)
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, mock_process):
        """Test that acall applies the same defaults as __call__."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["a", "b", "c"])
        result = await retriever.acall("x")
        assert result.response_type == "fuzzy_search"
        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_acall_delegates_to_call(self, mock_process):
        """Test that acall result matches __call__ result for the same input."""
        retriever = RapidFuzzFuzzyRetriever()
        retriever.add(["João da Silva"])
        sync_result = retriever("joao", top_k=2, return_score=True)
        async_result = await retriever.acall("joao", top_k=2, return_score=True)
        assert sync_result.response_type == async_result.response_type
        assert len(sync_result.data) == len(async_result.data)
