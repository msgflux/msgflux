import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

import msgflux.data.retrievers.http as http_module
import msgflux.data.retrievers.providers.linkup as linkup_provider
from msgflux.data.retrievers.providers.linkup import (
    LinkupWebFetchRetriever,
    LinkupWebRetriever,
)
from msgflux.data.retrievers.retriever import Retriever

from ._httpx2_mock import install_httpx2_mock, json_response


@pytest.fixture(autouse=True)
def _api_key():
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        yield


@pytest.fixture
def mock_httpx_clients():
    response = MagicMock()
    response.json.return_value = {}
    response.raise_for_status.return_value = None

    sync_client = MagicMock()
    sync_client.post.return_value = response

    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=response)
    async_client.aclose = AsyncMock(return_value=None)

    client_cls = MagicMock(return_value=sync_client)
    async_client_cls = MagicMock(return_value=async_client)
    fake_httpx = SimpleNamespace(Client=client_cls, AsyncClient=async_client_cls)

    with patch.object(http_module, "httpx2", fake_httpx):
        yield SimpleNamespace(
            response=response,
            client_cls=client_cls,
            sync_client=sync_client,
            async_client_cls=async_client_cls,
            async_client=async_client,
        )


@pytest.fixture
def retriever(mock_httpx_clients):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        yield LinkupWebRetriever()


@pytest.fixture
def fetch_retriever(mock_httpx_clients):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        yield LinkupWebFetchRetriever()


def test_init_defaults(retriever):
    assert retriever.depth == "standard"
    assert retriever.output_type == "searchResults"
    assert retriever.include_images is False
    assert retriever.include_domains is None
    assert retriever.exclude_domains is None
    assert retriever.base_url == linkup_provider.DEFAULT_BASE_URL
    assert retriever.search_url == "https://api.linkup.so/v1/search"
    assert retriever.timeout == 30.0
    assert retriever.client is not None
    assert retriever.aclient is not None
    assert not hasattr(retriever, "api_key")


def test_init_custom_params(mock_httpx_clients):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        retriever = LinkupWebRetriever(
            depth="deep",
            output_type="sourcedAnswer",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
            include_images=True,
            base_url="https://linkup.example.com/",
            timeout=5.0,
        )

    assert retriever.depth == "deep"
    assert retriever.output_type == "sourcedAnswer"
    assert retriever.include_domains == ["example.com"]
    assert retriever.exclude_domains == ["spam.com"]
    assert retriever.include_images is True
    assert retriever.base_url == "https://linkup.example.com"
    assert retriever.search_url == "https://linkup.example.com/v1/search"
    assert retriever.timeout == 5.0


def test_sync_search_results_uses_direct_httpx_request(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {
                "name": "Test Title",
                "url": "https://example.com",
                "content": "Test content",
                "type": "text",
            }
        ]
    }

    results = retriever("test query", top_k=2)

    mock_httpx_clients.client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.sync_client.post.call_args
    assert call_args[0][0] == "https://api.linkup.so/v1/search"
    assert call_args[1]["headers"] == {
        "Authorization": "Bearer test_key",
        "Content-Type": "application/json",
    }
    assert call_args[1]["json"] == {
        "q": "test query",
        "depth": "standard",
        "outputType": "searchResults",
        "maxResults": 2,
        "includeImages": False,
    }

    assert results.response_type == "web_search"
    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Test Title"
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "Test content"
    mock_httpx_clients.async_client.post.assert_not_called()


def test_sync_search_preserves_filters(mock_httpx_clients):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        retriever = LinkupWebRetriever(
            depth="deep",
            include_domains=["cnn.com", "bbc.com"],
            exclude_domains=["spam.com"],
            include_images=True,
        )

    mock_httpx_clients.response.json.return_value = {"results": []}

    retriever("news query", top_k=5)

    payload = mock_httpx_clients.sync_client.post.call_args[1]["json"]
    assert payload == {
        "q": "news query",
        "depth": "deep",
        "outputType": "searchResults",
        "maxResults": 5,
        "includeImages": True,
        "includeDomains": ["cnn.com", "bbc.com"],
        "excludeDomains": ["spam.com"],
    }


@pytest.mark.asyncio
async def test_async_sourced_answer_uses_direct_httpx_request(
    mock_httpx_clients,
):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        retriever = LinkupWebRetriever(output_type="sourcedAnswer")

    mock_httpx_clients.response.json.return_value = {
        "answer": "An answer",
        "sources": [
            {
                "name": "Source Name",
                "url": "https://source.com",
                "snippet": "Source snippet",
                "favicon": "",
            }
        ],
    }

    results = await retriever.acall("test query", top_k=1)

    mock_httpx_clients.async_client_cls.assert_called_once_with(timeout=30.0)
    call_args = mock_httpx_clients.async_client.post.call_args
    assert call_args[0][0] == "https://api.linkup.so/v1/search"
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"
    assert call_args[1]["json"]["q"] == "test query"
    assert call_args[1]["json"]["outputType"] == "sourcedAnswer"

    result_data = results.data[0].results[0]["data"]
    assert result_data["title"] == "Source Name"
    assert result_data["content"] == "Source snippet"
    assert result_data["url"] == "https://source.com"
    mock_httpx_clients.sync_client.post.assert_not_called()


def test_sync_search_reads_api_key_per_call(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {"results": []}

    with patch.dict("os.environ", {"LINKUP_API_KEY": "rotated_key"}, clear=True):
        retriever("query", top_k=1)

    headers = mock_httpx_clients.sync_client.post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer rotated_key"


def test_results_are_limited_to_top_k(retriever, mock_httpx_clients):
    mock_httpx_clients.response.json.return_value = {
        "results": [
            {"name": "First", "url": "https://example.com/1", "content": "First"},
            {"name": "Second", "url": "https://example.com/2", "content": "Second"},
        ]
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_sources_are_limited_to_top_k(mock_httpx_clients):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        retriever = LinkupWebRetriever(output_type="sourcedAnswer")

    mock_httpx_clients.response.json.return_value = {
        "sources": [
            {"name": "First", "url": "https://example.com/1", "snippet": "First"},
            {"name": "Second", "url": "https://example.com/2", "snippet": "Second"},
        ]
    }

    results = retriever("query", top_k=1)

    assert len(results.data[0].results) == 1
    assert results.data[0].results[0]["data"]["title"] == "First"


def test_search_returns_empty_results_on_http_error(retriever, mock_httpx_clients):
    mock_httpx_clients.response.raise_for_status.side_effect = RuntimeError("boom")

    with patch.object(linkup_provider.logger, "warning") as warning:
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    warning.assert_called_once()
    assert "Linkup search failed" in warning.call_args[0][0]


def test_search_returns_empty_results_without_api_key(mock_httpx_clients):
    with patch.dict("os.environ", {}, clear=True):
        retriever = LinkupWebRetriever()
        results = retriever("query", top_k=1)

    assert results.data[0].results == []
    mock_httpx_clients.sync_client.post.assert_not_called()


def test_close_closes_sync_client(retriever):
    retriever.close()

    retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_both_clients(retriever):
    await retriever.aclose()

    retriever.aclient.aclose.assert_awaited_once()
    retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_sync_client_when_async_close_fails(retriever):
    retriever.aclient.aclose.side_effect = RuntimeError("close boom")

    with pytest.raises(RuntimeError, match="close boom"):
        await retriever.aclose()

    retriever.client.close.assert_called_once()


def test_serialize_excludes_clients_and_retains_config(retriever):
    serialized = retriever.serialize()

    state = serialized["state"]
    assert "client" not in state
    assert "aclient" not in state
    assert state["depth"] == "standard"
    assert state["output_type"] == "searchResults"
    assert state["search_url"] == "https://api.linkup.so/v1/search"
    assert state["timeout"] == 30.0


def test_from_serialized_rebuilds_clients(mock_httpx_clients):
    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        retriever = LinkupWebRetriever(timeout=5.0, include_domains=["example.com"])

    restored = LinkupWebRetriever.from_serialized(retriever.serialize()["state"])

    assert restored.timeout == 5.0
    assert restored.include_domains == ["example.com"]
    assert restored.client is mock_httpx_clients.sync_client
    assert restored.aclient is mock_httpx_clients.async_client


def test_initialize_closes_sync_client_when_async_client_fails(mock_httpx_clients):
    sync_client = MagicMock()
    mock_httpx_clients.client_cls.return_value = sync_client
    mock_httpx_clients.async_client_cls.side_effect = RuntimeError("async boom")

    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        with pytest.raises(RuntimeError, match="async boom"):
            LinkupWebRetriever()

    sync_client.close.assert_called_once()


def test_fetch_retriever_factory(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Page"}),
    )

    retriever = Retriever.web_fetch("linkup", render_js=True)

    assert isinstance(retriever, LinkupWebFetchRetriever)
    assert retriever.provider == "linkup"
    assert retriever.retriever_type == "web_fetch"
    assert retriever.render_js is True

    retriever.close()


def test_fetch_init_defaults(fetch_retriever):
    assert fetch_retriever.mode == "standard"
    assert fetch_retriever.render_js is False
    assert fetch_retriever.include_raw_content is False
    assert fetch_retriever.extract_images is False
    assert fetch_retriever.base_url == linkup_provider.DEFAULT_BASE_URL
    assert fetch_retriever.fetch_url == "https://api.linkup.so/v1/fetch"
    assert fetch_retriever.timeout == 30.0
    assert fetch_retriever.client is not None
    assert fetch_retriever.aclient is not None
    assert not hasattr(fetch_retriever, "api_key")


def test_fetch_init_custom_params():
    retriever = LinkupWebFetchRetriever(
        mode="pro",
        render_js=True,
        include_raw_content=True,
        extract_images=True,
        base_url="https://linkup.example.com/",
        timeout=7.0,
    )

    assert retriever.mode == "pro"
    assert retriever.render_js is True
    assert retriever.include_raw_content is True
    assert retriever.extract_images is True
    assert retriever.base_url == "https://linkup.example.com"
    assert retriever.fetch_url == "https://linkup.example.com/v1/fetch"
    assert retriever.timeout == 7.0

    retriever.close()


def test_fetch_sync_uses_real_mock_transport(mocker):
    mock = install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Page"}),
    )
    retriever = LinkupWebFetchRetriever()

    result = retriever("example.com/page?q=1")

    request = mock.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.linkup.so/v1/fetch"
    assert request.headers["authorization"] == "Bearer test_key"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "url": "https://example.com/page?q=1",
        "mode": "standard",
        "renderJs": False,
    }
    assert result.response_type == "web_fetch"
    assert result.data[0]["data"] == {
        "url": "https://example.com/page?q=1",
        "content": "# Page",
    }

    retriever.close()


def test_fetch_sends_documented_rendering_options(mocker):
    mock = install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Page"}),
    )
    retriever = LinkupWebFetchRetriever(
        mode="pro",
        render_js=True,
        include_raw_content=True,
        extract_images=True,
    )

    retriever("https://example.com")

    request = mock.requests[0]
    assert json.loads(request.content) == {
        "url": "https://example.com",
        "mode": "pro",
        "renderJs": True,
        "includeRawContent": True,
        "extractImages": True,
    }

    retriever.close()


def test_fetch_batch_preserves_url_order(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Page"}),
    )
    retriever = LinkupWebFetchRetriever()

    result = retriever(["https://one.example", "https://two.example"])

    assert [entry["data"]["url"] for entry in result.data] == [
        "https://one.example",
        "https://two.example",
    ]

    retriever.close()


@pytest.mark.asyncio
async def test_fetch_async_uses_real_mock_transport(mocker):
    mock = install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Async Page"}),
    )
    retriever = LinkupWebFetchRetriever(render_js=True)

    result = await retriever.acall("https://example.com")

    request = mock.requests[0]
    assert str(request.url) == "https://api.linkup.so/v1/fetch"
    assert request.headers["authorization"] == "Bearer test_key"
    assert json.loads(request.content)["renderJs"] is True
    assert result.data[0]["data"]["content"] == "# Async Page"

    await retriever.aclose()


def test_fetch_reads_api_key_per_request(mocker):
    mock = install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Page"}),
    )
    retriever = LinkupWebFetchRetriever()

    with patch.dict("os.environ", {"LINKUP_API_KEY": "first"}, clear=True):
        retriever("https://example.com")
    with patch.dict("os.environ", {"LINKUP_API_KEY": "rotated"}, clear=True):
        retriever("https://example.com")

    assert mock.requests[0].headers["authorization"] == "Bearer first"
    assert mock.requests[1].headers["authorization"] == "Bearer rotated"

    retriever.close()


def test_fetch_empty_markdown_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": " "}),
    )
    retriever = LinkupWebFetchRetriever()

    with pytest.raises(RuntimeError, match="empty markdown"):
        retriever("https://example.com")

    retriever.close()


def test_fetch_http_error_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 429, {"error": "rate limit exceeded"}),
    )
    retriever = LinkupWebFetchRetriever()

    with pytest.raises(RuntimeError, match=r"429.*rate limit"):
        retriever("https://example.com")

    retriever.close()


@pytest.mark.asyncio
async def test_fetch_async_http_error_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 429, {"error": "rate limit exceeded"}),
    )
    retriever = LinkupWebFetchRetriever()

    with pytest.raises(RuntimeError, match=r"429.*rate limit"):
        await retriever.acall("https://example.com")

    await retriever.aclose()


def test_fetch_invalid_json_is_surfaced(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: httpx2.Response(200, text="not json", request=request),
    )
    retriever = LinkupWebFetchRetriever()

    with pytest.raises(RuntimeError, match="invalid JSON"):
        retriever("https://example.com")

    retriever.close()


def test_fetch_requires_api_key(mocker):
    mock = install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Page"}),
    )

    with patch.dict("os.environ", {}, clear=True):
        retriever = LinkupWebFetchRetriever()
        with pytest.raises(ValueError, match="LINKUP_API_KEY"):
            retriever("https://example.com")

    assert mock.requests == []
    retriever.close()


def test_fetch_close_closes_sync_client(fetch_retriever):
    fetch_retriever.close()

    fetch_retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_aclose_closes_both_clients(fetch_retriever):
    await fetch_retriever.aclose()

    fetch_retriever.aclient.aclose.assert_awaited_once()
    fetch_retriever.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_aclose_closes_sync_client_when_async_close_fails(fetch_retriever):
    fetch_retriever.aclient.aclose.side_effect = RuntimeError("close boom")

    with pytest.raises(RuntimeError, match="close boom"):
        await fetch_retriever.aclose()

    fetch_retriever.client.close.assert_called_once()


def test_fetch_serialization_roundtrip_rebuilds_clients(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 200, {"markdown": "# Restored"}),
    )
    retriever = LinkupWebFetchRetriever(
        mode="pro",
        render_js=True,
        include_raw_content=True,
        extract_images=True,
        timeout=7.0,
    )

    serialized = retriever.serialize()

    assert serialized["retriever_type"] == "web_fetch"
    assert serialized["provider"] == "linkup"
    assert "client" not in serialized["state"]
    assert "aclient" not in serialized["state"]
    assert serialized["state"]["mode"] == "pro"
    assert serialized["state"]["render_js"] is True
    assert serialized["state"]["include_raw_content"] is True
    assert serialized["state"]["extract_images"] is True
    assert serialized["state"]["fetch_url"] == "https://api.linkup.so/v1/fetch"
    assert serialized["state"]["timeout"] == 7.0

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert isinstance(restored, LinkupWebFetchRetriever)
    assert restored.mode == "pro"
    assert restored.render_js is True
    assert restored.include_raw_content is True
    assert restored.extract_images is True
    assert restored.base_url == "https://api.linkup.so"
    assert restored.fetch_url == "https://api.linkup.so/v1/fetch"
    assert restored.timeout == 7.0
    assert restored.client is not retriever.client

    result = restored("https://example.com")

    assert result.data[0]["data"]["content"] == "# Restored"

    retriever.close()
    restored.close()


def test_fetch_initialize_closes_sync_client_when_async_client_fails(
    mock_httpx_clients,
):
    sync_client = MagicMock()
    mock_httpx_clients.client_cls.return_value = sync_client
    mock_httpx_clients.async_client_cls.side_effect = RuntimeError("async boom")

    with patch.dict("os.environ", {"LINKUP_API_KEY": "test_key"}, clear=True):
        with pytest.raises(RuntimeError, match="async boom"):
            LinkupWebFetchRetriever()

    sync_client.close.assert_called_once()
