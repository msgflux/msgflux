"""Tests for BM25LexicalRetriever."""

import pytest

from msgflux.data.retrievers.providers.bm25 import BM25LexicalRetriever
from msgflux.data.retrievers.retriever import Retriever


class TestBM25LexicalRetriever:
    """Tests for BM25LexicalRetriever."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        retriever = BM25LexicalRetriever()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.documents == []

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        retriever = BM25LexicalRetriever(k1=2.0, b=0.5)
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5

    def test_init_with_none_applies_defaults(self):
        """Test that None values are replaced with defaults."""
        retriever = BM25LexicalRetriever(k1=None, b=None)
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75

    def test_add_documents(self):
        """Test adding documents to the index."""
        retriever = BM25LexicalRetriever()
        documents = ["hello world", "python programming", "machine learning"]
        retriever.add(documents)

        assert len(retriever.documents) == 3
        assert retriever.documents[0] == "hello world"
        assert len(retriever.doc_lengths) == 3
        assert retriever.avg_doc_length > 0

    def test_tokenize(self):
        """Test tokenization."""
        retriever = BM25LexicalRetriever()
        tokens = retriever._tokenize("Hello World Python")

        assert tokens == ["hello", "world", "python"]
        assert all(isinstance(token, str) for token in tokens)

    def test_search_single_query(self):
        """Test search with a single query."""
        retriever = BM25LexicalRetriever()
        documents = [
            "python programming language",
            "java programming language",
            "machine learning with python",
        ]
        retriever.add(documents)

        results = retriever(["python"], top_k=2)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) <= 2

    def test_search_multiple_queries(self):
        """Test search with multiple queries."""
        retriever = BM25LexicalRetriever()
        documents = [
            "python programming",
            "java programming",
            "machine learning",
        ]
        retriever.add(documents)

        results = retriever(["python", "java"], top_k=1)

        assert len(results.data) == 2

    def test_search_with_threshold(self):
        """Test search with score threshold."""
        retriever = BM25LexicalRetriever()
        documents = [
            "python programming",
            "java programming",
            "completely different text",
        ]
        retriever.add(documents)

        # High threshold should filter out irrelevant docs
        results = retriever(["python"], top_k=10, threshold=1.0, return_score=True)

        assert len(results.data) == 1
        # Should filter some results
        if len(results.data[0]["results"]) > 0:
            assert all(result.score >= 1.0 for result in results.data[0]["results"])

    def test_search_with_return_score(self):
        """Test that scores are returned when requested."""
        retriever = BM25LexicalRetriever()
        documents = ["python programming", "java programming"]
        retriever.add(documents)

        results = retriever(["python"], top_k=2, return_score=True)

        assert len(results.data[0]["results"]) > 0
        first_result = results.data[0]["results"][0]
        assert hasattr(first_result, "score")
        assert isinstance(first_result.score, float)

    def test_get_score_statistics(self):
        """Test score statistics calculation."""
        retriever = BM25LexicalRetriever()
        documents = [
            "python programming language",
            "java programming language",
            "machine learning with python",
        ]
        retriever.add(documents)

        stats = retriever.get_score_statistics("python")

        assert stats is not None
        assert "min_score" in stats
        assert "max_score" in stats
        assert "mean_score" in stats
        assert "median_score" in stats
        assert "std_score" in stats
        assert stats["min_score"] <= stats["max_score"]

    def test_get_score_statistics_empty_corpus(self):
        """Test score statistics with empty corpus."""
        retriever = BM25LexicalRetriever()
        stats = retriever.get_score_statistics("python")

        assert stats is None

    @pytest.mark.asyncio
    async def test_acall_single_query(self):
        """Test async search with a single query."""
        retriever = BM25LexicalRetriever()
        documents = ["python programming", "java programming"]
        retriever.add(documents)

        results = await retriever.acall(["python"], top_k=1)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) <= 1

    @pytest.mark.asyncio
    async def test_acall_multiple_queries(self):
        """Test async search with multiple queries."""
        retriever = BM25LexicalRetriever()
        documents = ["python programming", "java programming", "machine learning"]
        retriever.add(documents)

        results = await retriever.acall(["python", "java"], top_k=1)

        assert len(results.data) == 2

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self):
        """Test async search applies defaults correctly."""
        retriever = BM25LexicalRetriever()
        documents = ["python", "java", "rust", "go", "c++"]
        retriever.add(documents)

        # Call with no parameters to test defaults
        results = await retriever.acall("python")

        assert len(results.data) == 1
        # Should apply default top_k=5
        assert len(results.data[0]["results"]) <= 5

    def test_serialized_roundtrip_restores_configuration_and_requires_corpus(self):
        """Restore configuration, but keep the corpus as a runtime resource."""
        retriever = BM25LexicalRetriever(k1=1.2, b=0.4)
        retriever.add(
            [
                "python programming language",
                "machine learning with python",
                "java programming language",
            ]
        )
        serialized = retriever.serialize()

        assert "documents" not in serialized["state"]
        restored = Retriever.from_serialized(
            serialized["provider"],
            serialized["retriever_type"],
            serialized["state"],
        )

        assert restored.k1 == 1.2
        assert restored.b == 0.4
        assert restored.documents == []

        # The restored configuration is usable once the caller supplies the
        # corpus again; serialization intentionally does not persist it.
        restored.add(
            [
                "python programming language",
                "machine learning with python",
                "java programming language",
            ]
        )
        response = restored("machine learning", top_k=1)

        assert response.data[0].results[0].data == "machine learning with python"
