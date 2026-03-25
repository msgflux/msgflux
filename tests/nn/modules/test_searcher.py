"""Tests for msgflux.nn.modules.searcher module."""

import pytest
from unittest.mock import Mock

from msgflux.nn.modules.searcher import Searcher
from msgflux.nn.modules.embedder import Embedder
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse
from msgflux.data.retrievers.types import LexicalRetriever, SemanticRetriever


class MockEmbedderModel(BaseModel):
    """Mock embedder model for testing."""

    model_type = "text_embedder"
    provider = "mock"
    batch_support = True

    def __init__(self):
        self.model_id = "test-embedder"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        data = kwargs.get("data", [])
        if isinstance(data, list):
            embeddings = [[0.1, 0.2, 0.3] for _ in data]
        else:
            embeddings = [[0.1, 0.2, 0.3]]
        response = ModelResponse()
        response.set_response_type("embeddings")
        response.add(embeddings)
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


class MockLexicalRetriever(LexicalRetriever):
    """Mock lexical retriever using the real dotdict response format."""

    def __call__(self, queries, **kwargs):
        """Return mock results in the same format as BaseLexical.__call__."""
        data = []
        for query in queries:
            results = [
                dotdict({"data": f"Result 1 for {query}", "score": 0.95}),
                dotdict({"data": f"Result 2 for {query}", "score": 0.85}),
            ]
            data.append(dotdict({"results": results}))
        return dotdict({"response_type": "lexical_search", "data": data})


class MockSemanticRetriever(SemanticRetriever):
    """Mock semantic retriever using the real dotdict response format."""

    def __call__(self, queries, **kwargs):
        """Return mock results in the same format as real retrievers."""
        data = []
        for _ in queries:
            results = [
                dotdict({"data": "Semantic result 1", "score": 0.92}),
                dotdict({"data": "Semantic result 2", "score": 0.88}),
            ]
            data.append(dotdict({"results": results}))
        return dotdict({"response_type": "semantic_search", "data": data})


class TestSearcher:
    """Test suite for Searcher module."""

    def test_initialization(self):
        """Test Searcher basic initialization."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        assert ret.retriever is mock_retriever
        assert ret.embedder is None

    def test_initialization_with_config(self):
        """Test Searcher initialization with configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 5, "threshold": 0.7}
        ret = Searcher(retriever=mock_retriever, config=config)

        assert ret.retriever is mock_retriever
        assert ret._buffers["config"]["top_k"] == 5
        assert ret._buffers["config"]["threshold"] == 0.7

    def test_inheritance_from_module(self):
        """Test that Searcher inherits from Module."""
        from msgflux.nn.modules.module import Module

        assert issubclass(Searcher, Module)

    def test_with_top_k(self):
        """Test Searcher with top_k configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 3}
        ret = Searcher(retriever=mock_retriever, config=config)

        assert ret._buffers["config"]["top_k"] == 3

    def test_with_embedder_model(self):
        """Test Searcher with embedder model."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Searcher(retriever=mock_retriever, model=mock_model)

        assert ret.embedder is not None
        assert isinstance(ret.embedder, Embedder)
        assert ret.model is mock_model

    def test_with_embedder_instance(self):
        """Test Searcher with Embedder instance."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)
        ret = Searcher(retriever=mock_retriever, model=embedder)

        assert ret.embedder is embedder

    def test_forward_single_string(self):
        """Test Searcher forward with single string query."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        result = ret("test query")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["query"] == "test query"
        assert len(result[0]["results"]) == 2

    def test_forward_list_of_strings(self):
        """Test Searcher forward with list of string queries."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        result = ret(["query1", "query2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["query"] == "query1"
        assert result[1]["query"] == "query2"

    def test_with_embedder(self):
        """Test Searcher with embedder for semantic search."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Searcher(retriever=mock_retriever, model=mock_model)

        result = ret("semantic query")

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]["results"]) == 2

    def test_forward_with_message(self):
        """Test Searcher forward with Message object."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(
            retriever=mock_retriever, message_fields={"task_inputs": "query"}
        )

        msg = Message()
        msg.query = "message query"
        result = ret(msg)

        assert isinstance(result, list)

    def test_forward_list_of_dicts(self):
        """Test Searcher forward with list of dictionaries."""
        mock_retriever = MockLexicalRetriever()
        config = {"dict_key": "text"}
        ret = Searcher(retriever=mock_retriever, config=config)

        queries = [{"text": "query1"}, {"text": "query2"}]
        result = ret(queries)

        assert len(result) == 2

    def test_list_of_dicts_missing_dict_key(self):
        """Test Searcher raises error when dict_key is missing."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        queries = [{"text": "query1"}]
        with pytest.raises(AttributeError, match="require a `dict_key`"):
            ret(queries)

    @pytest.mark.asyncio
    async def test_aforward_single_string(self):
        """Test Searcher async forward with single string."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        result = await ret.aforward("async query")

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_aforward_with_embedder(self):
        """Test Searcher async forward with embedder."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Searcher(retriever=mock_retriever, model=mock_model)

        result = await ret.aforward("async semantic query")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_config_top_k(self):
        """Test Searcher with top_k configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 10}
        ret = Searcher(retriever=mock_retriever, config=config)

        params = ret._prepare_retriever_execution(["test"])
        assert params["top_k"] == 10

    def test_config_return_score(self):
        """Test Searcher with return_score configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"return_score": True}
        ret = Searcher(retriever=mock_retriever, config=config)

        params = ret._prepare_retriever_execution(["test"])
        assert params["return_score"] is True

    def test_config_threshold(self):
        """Test Searcher with threshold configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"threshold": 0.8}
        ret = Searcher(retriever=mock_retriever, config=config)

        params = ret._prepare_retriever_execution(["test"])
        assert params["threshold"] == 0.8

    def test_config_no_threshold(self):
        """Test Searcher without threshold configuration."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        params = ret._prepare_retriever_execution(["test"])
        assert "threshold" not in params

    def test_config_invalid_type(self):
        """Test Searcher raises TypeError for invalid config type."""
        mock_retriever = MockLexicalRetriever()

        with pytest.raises(TypeError, match="`config` must be a dict or None"):
            Searcher(retriever=mock_retriever, config="invalid")

    def test_invalid_retriever_type(self):
        """Test Searcher raises TypeError for invalid retriever type."""
        mock_retriever = Mock()

        with pytest.raises(TypeError, match="`retriever` requires"):
            Searcher(retriever=mock_retriever)

    def test_model_property_getter(self):
        """Test Searcher model property getter."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Searcher(retriever=mock_retriever, model=mock_model)

        assert ret.model is mock_model

    def test_model_property_getter_no_embedder(self):
        """Test Searcher model property getter with no embedder."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        assert ret.model is None

    def test_model_property_setter(self):
        """Test Searcher model property setter."""
        mock_retriever = MockSemanticRetriever()
        ret = Searcher(retriever=mock_retriever)

        mock_model = MockEmbedderModel()
        ret.model = mock_model

        assert ret.model is mock_model
        assert isinstance(ret.embedder, Embedder)

    def test_with_templates(self):
        """Test Searcher with templates."""
        mock_retriever = MockLexicalRetriever()
        templates = {"response": "Results: {{ results }}"}
        ret = Searcher(retriever=mock_retriever, templates=templates)

        assert ret.templates == templates

    def test_with_response_mode(self):
        """Test Searcher with custom response_mode."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever, response_mode="custom.field")

        assert ret.response_mode == "custom.field"

    def test_inspect_embedder_params(self):
        """Test inspect_embedder_params method."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Searcher(retriever=mock_retriever, model=mock_model)

        params = ret.inspect_embedder_params("test query")

        assert params["queries"] == ["test query"]

    def test_inspect_embedder_params_no_embedder(self):
        """Test inspect_embedder_params returns empty dict without embedder."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        params = ret.inspect_embedder_params("test query")

        assert params == {}

    def test_result_formatting(self):
        """Test Searcher formats results correctly."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        result = ret("test")

        assert "results" in result[0]
        assert "query" in result[0]
        assert "data" in result[0]["results"][0]
        assert "score" in result[0]["results"][0]

    def test_with_name(self):
        """Test Searcher initialization with name."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever, name="my_searcher")

        assert ret.name == "my_searcher"

    def test_semantic_type(self):
        """Test Searcher with SemanticRetriever type."""
        mock_retriever = MockSemanticRetriever()
        ret = Searcher(retriever=mock_retriever)

        assert isinstance(ret.retriever, SemanticRetriever)

    def test_lexical_type(self):
        """Test Searcher with LexicalRetriever type."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        assert isinstance(ret.retriever, LexicalRetriever)


class TestSearcherTemplates:
    """Test Searcher template rendering."""

    def test_single_result_with_template(self):
        """Test template rendering for a single query."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(
            retriever=mock_retriever,
            templates={
                "response": "Query: {{ query }}\n"
                "{% for r in results %}{{ r.data }}\n{% endfor %}"
            },
        )

        result = ret("test query")

        assert isinstance(result, str)
        assert "Query: test query" in result
        assert "Result 1 for test query" in result

    def test_multi_result_with_template(self):
        """Test template rendering for multiple queries."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(
            retriever=mock_retriever,
            templates={"response": "{{ query }}: {{ results|length }} results"},
        )

        result = ret(["q1", "q2"])

        assert isinstance(result, str)
        assert "q1: 2 results" in result
        assert "q2: 2 results" in result

    def test_no_template_returns_list(self):
        """Test that without template, result is a list."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever)

        result = ret("test")

        assert isinstance(result, list)


class TestSearcherResponseMode:
    """Test Searcher response_mode behavior."""

    def test_response_mode_writes_to_message(self):
        """Test response_mode writes results to message path."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(
            retriever=mock_retriever,
            message_fields={"task_inputs": "question"},
            response_mode="context",
        )

        msg = Message()
        msg.question = "test query"
        result = ret(msg)

        assert result is None
        assert msg.context is not None

    def test_response_mode_none_returns_directly(self):
        """Test response_mode=None returns results directly."""
        mock_retriever = MockLexicalRetriever()
        ret = Searcher(retriever=mock_retriever, response_mode=None)

        result = ret("test")

        assert isinstance(result, list)


class TestSearcherWithRealBM25:
    """Integration tests with real BM25 retriever."""

    def _make_bm25(self, documents=None):
        from msgflux.data.retrievers.providers.bm25 import BM25LexicalRetriever

        bm25 = BM25LexicalRetriever()
        if documents is None:
            documents = [
                "Python is a programming language created by Guido van Rossum",
                "JavaScript runs in the browser and on the server with Node.js",
                "Rust is fast and memory safe, created by Mozilla",
                "Go was designed at Google by Robert Griesemer",
            ]
        bm25.add(documents)
        return bm25

    def test_single_query(self):
        """Test single query with real BM25."""
        bm25 = self._make_bm25()
        searcher = Searcher(retriever=bm25, config={"top_k": 2, "return_score": True})

        result = searcher("Python programming")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["query"] == "Python programming"
        assert len(result[0]["results"]) == 2
        assert "Python" in result[0]["results"][0]["data"]
        assert result[0]["results"][0]["score"] > 0

    def test_multi_query(self):
        """Test multiple queries with real BM25."""
        bm25 = self._make_bm25()
        searcher = Searcher(retriever=bm25, config={"top_k": 1, "return_score": True})

        result = searcher(["Python", "Rust"])

        assert len(result) == 2
        assert result[0]["query"] == "Python"
        assert result[1]["query"] == "Rust"
        assert "Python" in result[0]["results"][0]["data"]
        assert "Rust" in result[1]["results"][0]["data"]

    def test_with_template(self):
        """Test real BM25 with template formatting."""
        bm25 = self._make_bm25()
        searcher = Searcher(
            retriever=bm25,
            config={"top_k": 1, "return_score": True},
            templates={
                "response": "{% for r in results %}{{ r.data }}{% endfor %}"
            },
        )

        result = searcher("Google")

        assert isinstance(result, str)
        assert "Google" in result

    def test_without_scores(self):
        """Test real BM25 without return_score returns None scores."""
        bm25 = self._make_bm25()
        searcher = Searcher(retriever=bm25, config={"top_k": 1})

        result = searcher("Python")

        assert result[0]["results"][0]["score"] is None

    def test_with_factory_method(self):
        """Test creating BM25 via Retriever.lexical() factory."""
        from msgflux.data.retrievers.retriever import Retriever

        bm25 = Retriever.lexical("bm25")
        bm25.add(["Python is great", "Rust is fast"])

        searcher = Searcher(retriever=bm25, config={"top_k": 1})
        result = searcher("Python")

        assert len(result) == 1
        assert "Python" in result[0]["results"][0]["data"]

    @pytest.mark.asyncio
    async def test_async_with_real_bm25(self):
        """Test async forward with real BM25."""
        bm25 = self._make_bm25()
        searcher = Searcher(retriever=bm25, config={"top_k": 1})

        result = await searcher.aforward("Python")

        assert len(result) == 1
        assert "Python" in result[0]["results"][0]["data"]

    def test_response_mode_with_message(self):
        """Test response_mode writes results into message."""
        bm25 = self._make_bm25()
        searcher = Searcher(
            retriever=bm25,
            message_fields={"task_inputs": "question"},
            response_mode="context",
            config={"top_k": 1},
        )

        msg = Message()
        msg.question = "Rust memory"
        result = searcher(msg)

        assert result is None
        assert msg.context is not None
        assert len(msg.context) == 1


class TestSearcherAutoParams:
    """Test AutoParams integration (description, name, annotations)."""

    def test_docstring_as_description(self):
        """Test that class docstring becomes description via AutoParams."""
        mock_retriever = MockLexicalRetriever()

        class MySearcher(Searcher):
            """Search the knowledge base for relevant documents."""
            retriever = mock_retriever

        s = MySearcher()
        assert s.description == "Search the knowledge base for relevant documents."

    def test_classname_as_name(self):
        """Test that class name becomes name via AutoParams."""
        mock_retriever = MockLexicalRetriever()

        class ProductSearcher(Searcher):
            retriever = mock_retriever

        s = ProductSearcher()
        assert s.name == "ProductSearcher"

    def test_default_annotations(self):
        """Test default annotations for tool compatibility."""
        mock_retriever = MockLexicalRetriever()
        s = Searcher(retriever=mock_retriever)

        assert s.annotations == {"query": str, "return": str}

    def test_custom_annotations(self):
        """Test custom annotations override."""
        mock_retriever = MockLexicalRetriever()
        s = Searcher(
            retriever=mock_retriever,
            annotations={"query": str, "limit": int, "return": str},
        )

        assert s.annotations["limit"] is int

    def test_explicit_description_overrides_docstring(self):
        """Test that explicit description takes precedence over docstring."""
        mock_retriever = MockLexicalRetriever()

        class MySearcher(Searcher):
            """This docstring is ignored."""
            retriever = mock_retriever
            description = "Explicit description wins."

        s = MySearcher()
        assert s.description == "Explicit description wins."

    def test_converts_to_tool(self):
        """Test that Searcher with description converts to nn.Tool."""
        from msgflux.nn.modules.tool import _convert_module_to_nn_tool

        mock_retriever = MockLexicalRetriever()

        class KBSearcher(Searcher):
            """Search the company knowledge base."""
            retriever = mock_retriever
            config = {"top_k": 2}

        s = KBSearcher()
        tool = _convert_module_to_nn_tool(s)

        assert tool.name == "KBSearcher"
        assert tool.description == "Search the company knowledge base."
        assert "query" in tool.annotations


class TestSearcherNormalization:
    """Test _normalize_retriever_response method."""

    def test_normalize_dotdict_response(self):
        """Test normalization of standard dotdict retriever response."""
        mock_retriever = MockLexicalRetriever()
        searcher = Searcher(retriever=mock_retriever)

        response = dotdict(
            {
                "response_type": "lexical_search",
                "data": [
                    dotdict(
                        {
                            "results": [
                                dotdict({"data": "doc1", "score": 0.9}),
                            ]
                        }
                    ),
                ],
            }
        )

        normalized = searcher._normalize_retriever_response(response)

        assert isinstance(normalized, list)
        assert len(normalized) == 1
        assert normalized[0][0]["data"] == "doc1"

    def test_normalize_plain_list(self):
        """Test normalization passes through plain lists."""
        mock_retriever = MockLexicalRetriever()
        searcher = Searcher(retriever=mock_retriever)

        response = [[{"data": "doc1", "score": 0.9}]]

        normalized = searcher._normalize_retriever_response(response)

        assert normalized is response
