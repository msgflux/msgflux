"""Tests for WikipediaWebRetriever."""

from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pytest

import msgflux.data.retrievers.http as http_module
import msgflux.data.retrievers.providers.wikipedia as wikipedia_provider
from msgflux.data.retrievers.providers.wikipedia import WikipediaWebRetriever

_PYTHON_TITLE = "Python (programming language)"
_PYTHON_EXTRACT = (
    "Python is a high-level programming language. "
    "It is widely used for web development. "
    "Python emphasizes code readability."
)


def _json_response(payload, *, request=None) -> httpx2.Response:
    return httpx2.Response(200, json=payload, request=request)


def _search_payload(titles):
    return {
        "batchcomplete": True,
        "query": {
            "searchinfo": {"totalhits": len(titles)},
            "search": [
                {"ns": 0, "title": title, "pageid": index + 1}
                for index, title in enumerate(titles)
            ],
        },
    }


def _missing_page_payload(title):
    return {
        "batchcomplete": True,
        "query": {"pages": [{"ns": 0, "title": title, "missing": True}]},
    }


def _page_payload(
    title,
    extract,
    *,
    pageid=1,
    links=None,
    disambiguation=False,
):
    page = {
        "pageid": pageid,
        "ns": 0,
        "title": title,
        "extract": extract,
        "pageprops": {"disambiguation": ""} if disambiguation else {},
    }
    if links:
        page["links"] = [{"ns": 0, "title": link} for link in links]

    return {"batchcomplete": True, "query": {"pages": [page]}}


def _imageinfo_payload(images):
    pages = [
        {
            "ns": 6,
            "title": title,
            "imageinfo": [{"url": url}],
        }
        for title, url in images
    ]
    return {"batchcomplete": True, "query": {"pages": pages}}


def _wiki_handler(*, search, pages, images=None):
    images = images or []

    def handler(request):
        params = request.url.params
        if params.get("list") == "search":
            return _json_response(_search_payload(search), request=request)

        if params.get("generator") == "images":
            return _json_response(_imageinfo_payload(images), request=request)

        title = params.get("titles")
        if title in pages:
            return _json_response(pages[title], request=request)
        return _json_response(_missing_page_payload(title), request=request)

    return handler


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
    instance = WikipediaWebRetriever()
    yield instance
    instance.client.close()


class TestWikipediaWebRetriever:
    """Tests for WikipediaWebRetriever."""

    def test_init_with_defaults(self, retriever):
        """Test initialization with default parameters."""
        assert retriever.language == "en"
        assert retriever.return_images is False
        assert retriever.max_return_images == 5
        assert retriever.summary is None
        assert retriever.timeout == 30.0
        assert retriever.client is not None
        assert retriever.aclient is not None

    def test_init_with_custom_params(self, api):
        """Test initialization with custom parameters."""
        retriever = WikipediaWebRetriever(
            language="pt",
            return_images=True,
            max_return_images=3,
            summary=2,
            timeout=5.0,
        )
        try:
            assert retriever.language == "pt"
            assert retriever.return_images is True
            assert retriever.max_return_images == 3
            assert retriever.summary == 2
            assert retriever.timeout == 5.0
        finally:
            retriever.client.close()

    def test_init_with_none_applies_defaults(self, retriever):
        """Test that None values are replaced with defaults."""
        retriever = WikipediaWebRetriever(
            language=None,
            return_images=None,
            max_return_images=None,
            timeout=None,
        )
        try:
            assert retriever.language == "en"
            assert retriever.return_images is False
            assert retriever.max_return_images == 5
            assert retriever.timeout == 30.0
        finally:
            retriever.client.close()

    def test_set_language_is_instance_scoped(self, retriever):
        """Test that setting one language does not mutate other instances."""
        retriever = WikipediaWebRetriever()
        portuguese = WikipediaWebRetriever(language="pt")
        try:
            retriever.set_language("fr")

            assert retriever.language == "fr"
            assert portuguese.language == "pt"
            assert retriever._get_api_url() == ("https://fr.wikipedia.org/w/api.php")
            assert portuguese._get_api_url() == ("https://pt.wikipedia.org/w/api.php")
        finally:
            retriever.client.close()
            portuguese.client.close()

    def test_extract_sentences(self, retriever):
        """Test sentence extraction."""
        text = "This is sentence one. This is sentence two. Short."
        sentences = retriever._extract_sentences(text)

        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_process_content_with_summary(self, retriever):
        """Test content processing with summary."""
        retriever = WikipediaWebRetriever(summary=1)
        try:
            result = retriever._process_content(
                "First sentence. Second sentence. Third sentence.", "Test Title"
            )

            assert "Test Title" in result
            assert "First sentence" in result
            assert "Second sentence" not in result
        finally:
            retriever.client.close()

    def test_process_content_without_summary(self, retriever):
        """Test content processing without summary."""
        result = retriever._process_content("Full content here.", "Test Title")

        assert result == "Test Title\n\nFull content here."

    def test_sync_search_formats_page_content(self, retriever, api):
        """Test sync search requests and page extraction formatting."""
        api.responder = _wiki_handler(
            search=[_PYTHON_TITLE],
            pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
        )

        results = retriever("Python programming", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        result = results.data[0]["results"][0]
        assert result["data"] == {
            "title": _PYTHON_TITLE,
            "content": f"{_PYTHON_TITLE}\n\n{_PYTHON_EXTRACT}",
        }
        assert "images" not in result

        search_request = api.requests[0]
        assert search_request.url.host == "en.wikipedia.org"
        assert search_request.url.path == "/w/api.php"
        params = search_request.url.params
        assert params["action"] == "query"
        assert params["list"] == "search"
        assert params["srsearch"] == "Python programming"
        assert params["srlimit"] == "1"
        assert params["srnamespace"] == "0"
        assert params["formatversion"] == "2"
        assert search_request.headers["Accept"] == "application/json"
        assert search_request.headers["User-Agent"] == wikipedia_provider.USER_AGENT

        page_request = api.requests[1]
        page_params = page_request.url.params
        assert page_params["titles"] == _PYTHON_TITLE
        assert page_params["prop"] == "extracts|pageprops|links"
        assert page_params["explaintext"] == "1"
        assert page_params["redirects"] == "1"
        assert page_params["pllimit"] == "1"

    def test_sync_search_applies_summary_limit(self, api):
        """Test that the summary parameter limits extracted sentences."""
        retriever = WikipediaWebRetriever(summary=2)
        try:
            api.responder = _wiki_handler(
                search=[_PYTHON_TITLE],
                pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
            )

            results = retriever("Python programming", top_k=1)

            content = results.data[0]["results"][0]["data"]["content"]
            assert "Python is a high-level programming language." in content
            assert "It is widely used for web development." in content
            assert "Python emphasizes code readability." not in content
        finally:
            retriever.client.close()

    def test_search_includes_images_when_requested(self, api):
        """Test image URL extraction, filtering, and bounding."""
        retriever = WikipediaWebRetriever(return_images=True, max_return_images=2)
        try:
            api.responder = _wiki_handler(
                search=[_PYTHON_TITLE],
                pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
                images=[
                    (
                        "File:Python logo.svg",
                        "https://upload.wikimedia.org/Python_logo.svg",
                    ),
                    (
                        "File:Python add5.png",
                        "https://upload.wikimedia.org/Python_add5.png",
                    ),
                    (
                        "File:Commons-logo.svg",
                        "https://upload.wikimedia.org/Commons-logo.svg",
                    ),
                    (
                        "File:Example.jpg",
                        "https://upload.wikimedia.org/Example.jpg",
                    ),
                    (
                        "File:Extra.gif",
                        "https://upload.wikimedia.org/Extra.gif",
                    ),
                ],
            )

            results = retriever("Python programming", top_k=1)

            result = results.data[0]["results"][0]
            assert result["images"] == [
                "https://upload.wikimedia.org/Python_add5.png",
                "https://upload.wikimedia.org/Example.jpg",
            ]

            image_request = api.requests[2]
            params = image_request.url.params
            assert params["generator"] == "images"
            assert params["titles"] == _PYTHON_TITLE
            assert params["gimlimit"] == "10"
            assert params["prop"] == "imageinfo"
            assert params["iiprop"] == "url"
        finally:
            retriever.client.close()

    def test_search_skips_images_when_disabled(self, retriever, api):
        """Test that image requests are not made unless requested."""
        api.responder = _wiki_handler(
            search=[_PYTHON_TITLE],
            pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
        )

        results = retriever("Python programming", top_k=1)

        assert "images" not in results.data[0]["results"][0]
        assert len(api.requests) == 2
        assert all(
            request.url.params.get("generator") != "images" for request in api.requests
        )

    def test_search_limits_and_deduplicates_titles(self, retriever, api):
        """Test that duplicate titles are removed and top_k bounds page fetches."""
        pages = {
            "Alpha": _page_payload("Alpha", "Alpha is the first letter."),
            "Beta": _page_payload("Beta", "Beta is the second letter."),
            "Gamma": _page_payload("Gamma", "Gamma is the third letter."),
        }
        api.responder = _wiki_handler(
            search=["Alpha", "Alpha", "Beta", "Gamma"],
            pages=pages,
        )

        results = retriever("Greek letters", top_k=2)

        titles = [item["data"]["title"] for item in results.data[0]["results"]]
        assert titles == ["Alpha", "Beta"]
        page_requests = [
            request
            for request in api.requests
            if request.url.params.get("titles") in pages
        ]
        assert [request.url.params["titles"] for request in page_requests] == [
            "Alpha",
            "Beta",
        ]

    def test_search_follows_disambiguation_link(self, retriever, api):
        """Test that disambiguation pages fall back to their first option."""
        pages = {
            "Mercury": _page_payload(
                "Mercury",
                "Mercury may refer to:",
                disambiguation=True,
                links=["Mercury (planet)"],
            ),
            "Mercury (planet)": _page_payload(
                "Mercury (planet)",
                "Mercury is the smallest planet in the Solar System.",
            ),
        }
        api.responder = _wiki_handler(search=["Mercury"], pages=pages)

        results = retriever("Mercury", top_k=1)

        result = results.data[0]["results"][0]
        assert result["data"]["title"] == "Mercury (planet)"
        assert "smallest planet" in result["data"]["content"]

    def test_search_skips_missing_pages(self, retriever, api):
        """Test that missing pages are skipped."""
        api.responder = _wiki_handler(search=["Missing page"], pages={})

        results = retriever("Missing page", top_k=1)

        assert results.data[0]["results"] == []

    def test_search_returns_empty_results_on_api_error(self, retriever, api):
        """Test that MediaWiki API errors are logged and return no results."""
        api.responder = lambda request: _json_response(
            {"error": {"code": "search-error", "info": "search failed"}},
            request=request,
        )

        with patch.object(wikipedia_provider.logger, "warning") as warning:
            results = retriever("query", top_k=1)

        assert results.data[0]["results"] == []
        warning.assert_called_once()
        assert "Wikipedia search failed" in warning.call_args[0][0]

    def test_search_returns_empty_results_on_http_error(self, retriever, api):
        """Test that HTTP failures return no results."""
        api.responder = lambda request: httpx2.Response(
            503, text="unavailable", request=request
        )

        results = retriever("query", top_k=1)

        assert results.data[0]["results"] == []

    def test_search_uses_language_host_per_instance(self, api):
        """Test that each instance sends requests to its own language host."""
        api.responder = _wiki_handler(search=[], pages={})
        english = WikipediaWebRetriever()
        portuguese = WikipediaWebRetriever(language="pt")
        try:
            english.set_language("fr")

            english("query", top_k=1)
            portuguese("query", top_k=1)

            assert [request.url.host for request in api.requests] == [
                "fr.wikipedia.org",
                "pt.wikipedia.org",
            ]
        finally:
            english.client.close()
            portuguese.client.close()

    @pytest.mark.asyncio
    async def test_acall_single_query(self, retriever, api):
        """Test async search with a single query."""
        api.responder = _wiki_handler(
            search=[_PYTHON_TITLE],
            pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
        )

        results = await retriever.acall(["Python programming"], top_k=1)

        assert len(results.data) == 1
        assert results.data[0]["results"][0]["data"]["title"] == _PYTHON_TITLE

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, retriever, api):
        """Test async search applies defaults correctly."""
        api.responder = _wiki_handler(
            search=[_PYTHON_TITLE],
            pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
        )

        results = await retriever.acall("Python")

        assert len(results.data) == 1
        assert api.requests[0].url.params["srlimit"] == "1"

    @pytest.mark.asyncio
    async def test_acall_multiple_queries_and_images(self, api):
        """Test async search over multiple queries with image extraction."""
        retriever = WikipediaWebRetriever(return_images=True, max_return_images=1)
        try:
            api.responder = _wiki_handler(
                search=[_PYTHON_TITLE],
                pages={_PYTHON_TITLE: _page_payload(_PYTHON_TITLE, _PYTHON_EXTRACT)},
                images=[
                    (
                        "File:Python add5.png",
                        "https://upload.wikimedia.org/Python_add5.png",
                    )
                ],
            )

            results = await retriever.acall(
                ["Python programming", "Python language"], top_k=1
            )

            assert len(results.data) == 2
            assert len(api.requests) == 6
            assert results.data[0]["results"][0]["images"] == [
                "https://upload.wikimedia.org/Python_add5.png"
            ]
        finally:
            retriever.client.close()

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
        retriever = WikipediaWebRetriever(
            language="pt",
            summary=2,
            return_images=True,
            max_return_images=3,
            timeout=7.0,
        )
        try:
            serialized = retriever.serialize()
        finally:
            retriever.client.close()

        state = serialized["state"]
        assert "client" not in state
        assert "aclient" not in state
        assert state["language"] == "pt"
        assert state["summary"] == 2
        assert state["return_images"] is True
        assert state["max_return_images"] == 3
        assert state["timeout"] == 7.0

    def test_from_serialized_rebuilds_clients(self, api):
        """Test that deserialization rebuilds clients from timeout."""
        retriever = WikipediaWebRetriever(
            language="pt",
            summary=2,
            return_images=True,
            max_return_images=3,
            timeout=7.0,
        )
        try:
            serialized = retriever.serialize()
        finally:
            retriever.client.close()

        restored = WikipediaWebRetriever.from_serialized(serialized["state"])

        assert restored.language == "pt"
        assert restored.summary == 2
        assert restored.return_images is True
        assert restored.max_return_images == 3
        assert restored.timeout == 7.0
        assert restored.client is not retriever.client
        assert restored.aclient is not retriever.aclient
        restored.client.close()
