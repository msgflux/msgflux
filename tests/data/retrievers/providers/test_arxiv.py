"""Tests for ArxivWebRetriever."""

from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pytest

import msgflux.data.retrievers.http as http_module
import msgflux.data.retrievers.providers.arxiv as arxiv_provider
from msgflux.data.retrievers.providers.arxiv import ArxivWebRetriever

_ATOM_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <opensearch:totalResults>2</opensearch:totalResults>
"""

_ATOM_FOOTER = "</feed>"

_ENTRY_ONE = """
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-01-20T12:00:00Z</updated>
    <published>2024-01-15T12:00:00Z</published>
    <title>Deep Learning for Natural Language
      Processing</title>
    <summary>This paper presents a novel approach
      to NLP using deep learning.</summary>
    <author>
      <name>John Doe</name>
      <arxiv:affiliation>Example University</arxiv:affiliation>
    </author>
    <author>
      <name>Jane Smith</name>
    </author>
    <link href="http://arxiv.org/abs/2401.12345v2"
          rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2"
          rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"
          scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
"""

_ENTRY_TWO = """
  <entry>
    <id>http://arxiv.org/abs/2402.67890v1</id>
    <updated>2024-02-05T09:30:00-05:00</updated>
    <published>2024-02-05T09:30:00-05:00</published>
    <title>Graph Neural Networks in Practice</title>
    <summary>A practical survey of graph neural networks.</summary>
    <author><name>Alex Kim</name></author>
    <link href="http://arxiv.org/abs/2402.67890v1"
          rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2402.67890v1"
          rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
"""

_ENTRY_THREE = """
  <entry>
    <id>http://arxiv.org/abs/2403.11111v1</id>
    <updated>2024-03-01T00:00:00Z</updated>
    <published>2024-03-01T00:00:00Z</published>
    <title>Transformers for Vision</title>
    <summary>An overview of vision transformers.</summary>
    <author><name>Sam Lee</name></author>
    <link href="http://arxiv.org/abs/2403.11111v1"
          rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2403.11111v1"
          rel="related" type="application/pdf"/>
    <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
"""

_ERROR_FEED = (
    _ATOM_HEADER
    + """
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_1234.12345</id>
    <title>Error</title>
    <summary>incorrect id format for 1234.12345</summary>
    <updated>2007-10-12T00:00:00-04:00</updated>
    <link href="http://arxiv.org/api/errors#incorrect_id_format_for_1234.12345"
          rel="alternate" type="text/html"/>
  </entry>
"""
    + _ATOM_FOOTER
)


def _atom_response(entries: str = "", *, request=None) -> httpx2.Response:
    return httpx2.Response(
        200,
        text=_ATOM_HEADER + entries + _ATOM_FOOTER,
        headers={"Content-Type": "application/atom+xml"},
        request=request,
    )


class _FakeAPI:
    def __init__(self):
        self.requests = []
        self.responder = None

    def __call__(self, request):
        self.requests.append(request)
        if self.responder is None:
            raise AssertionError("No responder configured")
        return self.responder(request)


@pytest.fixture
def api(monkeypatch):
    api = _FakeAPI()
    transport = httpx2.MockTransport(api)
    real_client = httpx2.Client
    real_async_client = httpx2.AsyncClient

    def make_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    def make_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    fake_httpx = SimpleNamespace(Client=make_client, AsyncClient=make_async_client)
    monkeypatch.setattr(http_module, "httpx2", fake_httpx)
    return api


@pytest.fixture
def retriever(api):
    instance = ArxivWebRetriever()
    yield instance
    instance.client.close()


class TestArxivWebRetriever:
    """Tests for ArxivWebRetriever."""

    def test_init_with_defaults(self, retriever):
        """Test initialization with default parameters."""
        assert retriever.max_results == 10
        assert retriever.sort_by == "relevance"
        assert retriever.sort_order == "descending"
        assert retriever.timeout == 30.0
        assert retriever.client is not None
        assert retriever.aclient is not None

    def test_init_with_custom_params(self, api):
        """Test initialization with custom parameters."""
        retriever = ArxivWebRetriever(
            max_results=5,
            sort_by="submittedDate",
            sort_order="ascending",
            timeout=5.0,
        )
        try:
            assert retriever.max_results == 5
            assert retriever.sort_by == "submittedDate"
            assert retriever.sort_order == "ascending"
            assert retriever.timeout == 5.0
        finally:
            retriever.client.close()

    def test_init_with_none_applies_defaults(self, retriever):
        """Test that None values are replaced with defaults."""
        retriever = ArxivWebRetriever(
            max_results=None, sort_by=None, sort_order=None, timeout=None
        )
        try:
            assert retriever.max_results == 10
            assert retriever.sort_by == "relevance"
            assert retriever.sort_order == "descending"
            assert retriever.timeout == 30.0
        finally:
            retriever.client.close()

    def test_invalid_sort_by_raises_error(self, api):
        """Test that invalid sort_by value raises error."""
        with pytest.raises(ValueError, match="Invalid sort_by value"):
            ArxivWebRetriever(sort_by="invalid")

    def test_invalid_sort_order_raises_error(self, api):
        """Test that invalid sort_order value raises error."""
        with pytest.raises(ValueError, match="Invalid sort_order value"):
            ArxivWebRetriever(sort_order="invalid")

    def test_sync_search_formats_all_fields(self, retriever, api):
        """Test sync requests and full Atom field formatting."""
        api.responder = lambda request: _atom_response(
            _ENTRY_ONE + _ENTRY_TWO, request=request
        )

        results = retriever("deep learning", top_k=2)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert len(results.data[0]["results"]) == 2

        first = results.data[0]["results"][0]["data"]
        assert first["title"] == "Deep Learning for Natural Language Processing"
        assert first["summary"] == (
            "This paper presents a novel approach to NLP using deep learning."
        )
        assert first["authors"] == ["John Doe", "Jane Smith"]
        assert first["published"] == "2024-01-15T12:00:00+00:00"
        assert first["updated"] == "2024-01-20T12:00:00+00:00"
        assert first["pdf_url"] == "http://arxiv.org/pdf/2401.12345v2"
        assert first["entry_id"] == "http://arxiv.org/abs/2401.12345v2"
        assert first["categories"] == ["cs.CL", "cs.LG"]

        second = results.data[0]["results"][1]["data"]
        assert second["title"] == "Graph Neural Networks in Practice"
        assert second["authors"] == ["Alex Kim"]
        assert second["published"] == "2024-02-05T09:30:00-05:00"

        request = api.requests[0]
        assert str(request.url).split("?")[0] == arxiv_provider.ARXIV_API_URL
        assert request.url.params["search_query"] == "deep learning"
        assert request.url.params["start"] == "0"
        assert request.url.params["max_results"] == "2"
        assert request.url.params["sortBy"] == "relevance"
        assert request.url.params["sortOrder"] == "descending"
        assert request.headers["Accept"] == "application/atom+xml"
        assert request.headers["User-Agent"] == arxiv_provider.USER_AGENT

    def test_sync_search_uses_custom_sort(self, retriever, api):
        """Test that custom sort settings are sent to the Atom API."""
        retriever = ArxivWebRetriever(sort_by="submittedDate", sort_order="ascending")
        try:
            api.responder = lambda request: _atom_response(request=request)

            retriever("query", top_k=1)

            params = api.requests[0].url.params
            assert params["sortBy"] == "submittedDate"
            assert params["sortOrder"] == "ascending"
        finally:
            retriever.client.close()

    def test_search_limits_results_to_top_k(self, retriever, api):
        """Test that more Atom entries than requested are not returned."""
        api.responder = lambda request: _atom_response(
            _ENTRY_ONE + _ENTRY_TWO + _ENTRY_THREE, request=request
        )

        results = retriever("query", top_k=1)

        assert len(results.data[0]["results"]) == 1
        assert results.data[0]["results"][0]["data"]["title"] == (
            "Deep Learning for Natural Language Processing"
        )
        assert api.requests[0].url.params["max_results"] == "1"

    def test_search_deduplicates_entries(self, retriever, api):
        """Test that repeated Atom entries are returned once."""
        api.responder = lambda request: (
            _atom_response(_ENTRY_ONE + _ENTRY_ONE + _ENTRY_TWO, request=request)
            if request.url.params["start"] == "0"
            else _atom_response(request=request)
        )

        results = retriever("query", top_k=3)

        titles = [item["data"]["title"] for item in results.data[0]["results"]]
        assert titles == [
            "Deep Learning for Natural Language Processing",
            "Graph Neural Networks in Practice",
        ]

    def test_search_paginates_until_top_k(self, retriever, api, monkeypatch):
        """Test that large result sets are fetched in pages."""
        monkeypatch.setattr(arxiv_provider, "ARXIV_PAGE_SIZE", 1)
        entries = [_ENTRY_ONE, _ENTRY_TWO]

        def responder(request):
            start = int(request.url.params["start"])
            return _atom_response(entries[start], request=request)

        api.responder = responder

        results = retriever("query", top_k=2)

        assert len(results.data[0]["results"]) == 2
        assert [request.url.params["start"] for request in api.requests] == [
            "0",
            "1",
        ]
        assert all(request.url.params["max_results"] == "1" for request in api.requests)

    def test_search_with_multiple_queries(self, retriever, api):
        """Test that parallel queries keep results in input order."""
        entries_by_query = {
            "machine learning": _ENTRY_ONE,
            "deep learning": _ENTRY_TWO,
        }

        def responder(request):
            return _atom_response(
                entries_by_query[request.url.params["search_query"]], request=request
            )

        api.responder = responder

        results = retriever(["machine learning", "deep learning"], top_k=1)

        assert len(results.data) == 2
        assert [group["results"][0]["data"]["title"] for group in results.data] == [
            "Deep Learning for Natural Language Processing",
            "Graph Neural Networks in Practice",
        ]
        assert sorted(
            request.url.params["search_query"] for request in api.requests
        ) == ["deep learning", "machine learning"]

    def test_search_returns_empty_results_on_http_error(self, retriever, api):
        """Test that HTTP failures are logged and return no results."""
        api.responder = lambda request: httpx2.Response(
            503, text="unavailable", request=request
        )

        with patch.object(arxiv_provider.logger, "warning") as warning:
            results = retriever("query", top_k=1)

        assert results.data[0]["results"] == []
        warning.assert_called_once()
        assert "arXiv search failed" in warning.call_args[0][0]

    def test_search_returns_empty_results_on_malformed_xml(self, retriever, api):
        """Test that malformed Atom payloads return no results."""
        api.responder = lambda request: httpx2.Response(
            200, text="<feed><entry>", request=request
        )

        results = retriever("query", top_k=1)

        assert results.data[0]["results"] == []

    def test_search_returns_empty_results_for_api_error_feed(self, retriever, api):
        """Test that arXiv error feeds are not returned as papers."""
        api.responder = lambda request: httpx2.Response(
            200, text=_ERROR_FEED, request=request
        )

        results = retriever("query", top_k=1)

        assert results.data[0]["results"] == []

    @pytest.mark.asyncio
    async def test_acall_single_query(self, retriever, api):
        """Test async search with a single query."""
        api.responder = lambda request: _atom_response(_ENTRY_ONE, request=request)

        results = await retriever.acall(["machine learning"], top_k=1)

        assert len(results.data) == 1
        assert results.data[0]["results"][0]["data"]["title"] == (
            "Deep Learning for Natural Language Processing"
        )

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, retriever, api):
        """Test async search applies defaults correctly."""
        api.responder = lambda request: _atom_response(_ENTRY_ONE, request=request)

        results = await retriever.acall("machine learning")

        assert len(results.data) == 1
        assert api.requests[0].url.params["max_results"] == "1"

    @pytest.mark.asyncio
    async def test_acall_multiple_queries(self, retriever, api):
        """Test async search with multiple queries."""
        api.responder = lambda request: _atom_response(_ENTRY_TWO, request=request)

        results = await retriever.acall(["neural networks", "transformers"], top_k=2)

        assert len(results.data) == 2
        assert len(api.requests) == 2
        assert results.data[0]["results"][0]["data"]["title"] == (
            "Graph Neural Networks in Practice"
        )

    def test_close_closes_sync_client(self, retriever):
        """Test that close() closes the reusable sync client."""
        retriever.close()

        assert retriever.client.is_closed is True

    @pytest.mark.asyncio
    async def test_aclose_closes_both_clients(self, retriever):
        """Test that aclose() closes both reusable clients."""
        await retriever.aclose()

        assert retriever.aclient.is_closed is True
        assert retriever.client.is_closed is True

    def test_serialize_excludes_clients_and_retains_config(self, retriever):
        """Test serialization keeps configuration and drops runtime clients."""
        retriever = ArxivWebRetriever(
            max_results=5,
            sort_by="submittedDate",
            sort_order="ascending",
            timeout=7.0,
        )
        try:
            serialized = retriever.serialize()
        finally:
            retriever.client.close()

        state = serialized["state"]
        assert "client" not in state
        assert "aclient" not in state
        assert state["max_results"] == 5
        assert state["sort_by"] == "submittedDate"
        assert state["sort_order"] == "ascending"
        assert state["timeout"] == 7.0

    def test_from_serialized_rebuilds_clients(self, api):
        """Test that deserialization rebuilds clients from timeout."""
        retriever = ArxivWebRetriever(
            max_results=5,
            sort_by="submittedDate",
            sort_order="ascending",
            timeout=7.0,
        )
        try:
            serialized = retriever.serialize()
        finally:
            retriever.client.close()

        restored = ArxivWebRetriever.from_serialized(serialized["state"])

        assert restored.max_results == 5
        assert restored.sort_by == "submittedDate"
        assert restored.sort_order == "ascending"
        assert restored.timeout == 7.0
        assert restored.client is not retriever.client
        assert restored.aclient is not retriever.aclient
        restored.client.close()
