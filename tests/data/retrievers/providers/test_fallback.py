import asyncio

import httpx2
import pytest

from msgflux.data.retrievers.providers.fallback import FallbackWebFetchRetriever
from msgflux.data.retrievers.providers.firecrawl import FirecrawlWebRetriever
from msgflux.data.retrievers.providers.markdown import MarkdownWebFetchRetriever
from msgflux.data.retrievers.retriever import Retriever

from ._httpx2_mock import install_httpx2_mock, json_response


def _firecrawl_ok(request: httpx2.Request) -> httpx2.Response:
    return json_response(
        request,
        200,
        {
            "success": True,
            "data": {"markdown": "# From Firecrawl"},
        },
    )


def _markdown_ok(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        text="# From markdown.new",
        headers={"content-type": "text/markdown"},
        request=request,
    )


def _recovering_handler(request: httpx2.Request) -> httpx2.Response:
    if request.url.host == "markdown.new":
        return json_response(request, 500, {"error": "markdown down"})
    return _firecrawl_ok(request)


def test_default_chain_uses_markdown_then_firecrawl(mocker):
    mock = install_httpx2_mock(mocker, _recovering_handler)
    retriever = FallbackWebFetchRetriever()

    result = retriever("https://example.com/page")

    assert result.response_type == "web_fetch"
    assert result.data[0]["data"] == {
        "url": "https://example.com/page",
        "content": "# From Firecrawl",
    }
    assert [request.url.host for request in mock.requests] == [
        "markdown.new",
        "api.firecrawl.dev",
    ]

    retriever.close()


def test_empty_markdown_falls_back_to_firecrawl(mocker):
    def handler(request):
        if request.url.host == "markdown.new":
            return httpx2.Response(
                200,
                text=" ",
                headers={"content-type": "text/markdown"},
                request=request,
            )
        return _firecrawl_ok(request)

    mock = install_httpx2_mock(mocker, handler)
    retriever = FallbackWebFetchRetriever()

    result = retriever("https://example.com/page")

    assert result.data[0]["data"]["content"] == "# From Firecrawl"
    assert len(mock.requests) == 2

    retriever.close()


def test_both_failures_are_reported_together(mocker):
    install_httpx2_mock(
        mocker,
        lambda request: json_response(request, 503, {"error": "maintenance"}),
    )
    retriever = FallbackWebFetchRetriever()

    with pytest.raises(RuntimeError) as excinfo:
        retriever("https://example.com/page")

    message = str(excinfo.value)
    assert "Failed to fetch https://example.com/page" in message
    assert "markdown" in message
    assert "firecrawl" in message
    assert "maintenance" in message

    retriever.close()


def test_primary_success_does_not_call_fallback(mocker):
    mock = install_httpx2_mock(mocker, _markdown_ok)
    retriever = FallbackWebFetchRetriever()

    result = retriever("https://example.com/page")

    assert result.data[0]["data"]["content"] == "# From markdown.new"
    assert [request.url.host for request in mock.requests] == ["markdown.new"]

    retriever.close()


def test_preserves_url_order(mocker):
    install_httpx2_mock(mocker, _markdown_ok)
    retriever = FallbackWebFetchRetriever()

    result = retriever(["https://one.example", "https://two.example"])

    assert [item["data"]["url"] for item in result.data] == [
        "https://one.example",
        "https://two.example",
    ]
    assert all(item["data"]["content"] == "# From markdown.new" for item in result.data)

    retriever.close()


def test_invalid_url_raises_value_error_without_network(mocker):
    mock = install_httpx2_mock(mocker, _markdown_ok)
    retriever = FallbackWebFetchRetriever()

    with pytest.raises(ValueError, match="only http and https"):
        retriever("file:///etc/passwd")

    assert mock.requests == []
    retriever.close()


def test_accepts_injected_child_retrievers(mocker):
    mock = install_httpx2_mock(mocker, _markdown_ok)
    child = MarkdownWebFetchRetriever(method="ai")
    retriever = FallbackWebFetchRetriever(
        retrievers=[child],
        timeout=5,
    )

    result = retriever("https://example.com")

    assert result.data[0]["data"]["content"] == "# From markdown.new"
    assert [request.url.host for request in mock.requests] == ["markdown.new"]
    assert retriever.children[0] is child
    assert retriever.children[0].method == "ai"
    assert retriever.children[0].client is child.client

    retriever.close()


def test_default_construction_reuses_children_without_double_clients(mocker):
    client_cls = mocker.patch(
        "msgflux.data.retrievers.http.httpx2.Client",
        wraps=httpx2.Client,
    )
    async_client_cls = mocker.patch(
        "msgflux.data.retrievers.http.httpx2.AsyncClient",
        wraps=httpx2.AsyncClient,
    )

    retriever = FallbackWebFetchRetriever()
    children = list(retriever.children)

    assert [type(child).__name__ for child in children] == [
        "MarkdownWebFetchRetriever",
        "FirecrawlWebFetchRetriever",
    ]
    assert client_cls.call_count == 2
    assert async_client_cls.call_count == 2

    asyncio.run(retriever.aclose())

    for child in children:
        assert child.client.is_closed is True
        assert child.aclient.is_closed is True


def test_injected_child_is_not_cloned_or_allocated_twice(mocker):
    client_cls = mocker.patch(
        "msgflux.data.retrievers.http.httpx2.Client",
        wraps=httpx2.Client,
    )
    async_client_cls = mocker.patch(
        "msgflux.data.retrievers.http.httpx2.AsyncClient",
        wraps=httpx2.AsyncClient,
    )
    child = MarkdownWebFetchRetriever(method="ai")

    assert client_cls.call_count == 1
    assert async_client_cls.call_count == 1

    retriever = FallbackWebFetchRetriever(retrievers=[child])

    assert retriever.children == [child]
    assert retriever.children[0] is child
    assert client_cls.call_count == 1
    assert async_client_cls.call_count == 1

    asyncio.run(retriever.aclose())


def test_provider_name_config_uses_constructor_defaults_and_rebuilds(mocker):
    install_httpx2_mock(mocker, _markdown_ok)
    retriever = FallbackWebFetchRetriever(retrievers=["markdown", "http"])

    assert [type(child).__name__ for child in retriever.children] == [
        "MarkdownWebFetchRetriever",
        "HttpWebFetchRetriever",
    ]
    specs = retriever.serialize()["state"]["retrievers"]
    assert [spec["provider"] for spec in specs] == ["markdown", "http"]
    assert specs[0]["state"]["endpoint"] == "https://markdown.new/"
    assert specs[0]["state"]["method"] == "auto"
    assert specs[0]["state"]["timeout"] == 30.0
    assert specs[1]["state"]["default_headers"] == {}
    assert specs[1]["state"]["timeout"] == 30.0

    result = retriever("https://example.com")
    assert result.data[0]["data"]["content"] == "# From markdown.new"

    serialized = retriever.serialize()
    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert [type(child).__name__ for child in restored.children] == [
        "MarkdownWebFetchRetriever",
        "HttpWebFetchRetriever",
    ]
    assert restored.children[0].endpoint == "https://markdown.new/"
    assert restored.children[1].default_headers == {}

    restored_result = restored("https://two.example")
    assert restored_result.data[0]["data"]["content"] == "# From markdown.new"

    retriever.close()
    restored.close()


@pytest.mark.parametrize(
    "spec",
    [
        {"provider": "markdown", "retriever_type": "web_fetch"},
        {"provider": "markdown", "retriever_type": "web_fetch", "state": {}},
        {"provider": "", "retriever_type": "web_fetch", "state": {"timeout": 30}},
        {"provider": "markdown", "retriever_type": "web", "state": {"timeout": 30}},
        {"provider": "markdown", "retriever_type": "web_fetch", "state": []},
    ],
)
def test_rejects_malformed_serialized_specs(spec):
    with pytest.raises(ValueError):
        FallbackWebFetchRetriever(retrievers=[spec])


def test_rejects_search_retriever_instance(mocker):
    install_httpx2_mock(mocker, _firecrawl_ok)
    child = FirecrawlWebRetriever()

    with pytest.raises(ValueError, match="web_fetch"):
        FallbackWebFetchRetriever(retrievers=[child])

    child.close()


def test_partial_construction_closes_built_child(mocker):
    install_httpx2_mock(mocker, _markdown_ok)
    child = MarkdownWebFetchRetriever()

    with pytest.raises(ValueError, match="not registered"):
        FallbackWebFetchRetriever(retrievers=[child, "not-a-provider"])

    assert child.client.is_closed is True
    asyncio.run(child.aclose())


def test_rejects_empty_retriever_list(mocker):
    install_httpx2_mock(mocker, _markdown_ok)

    with pytest.raises(ValueError, match="at least one"):
        FallbackWebFetchRetriever(retrievers=[])


@pytest.mark.asyncio
async def test_provider_name_config_async_and_rebuild(mocker):
    install_httpx2_mock(mocker, _recovering_handler)
    retriever = FallbackWebFetchRetriever(retrievers=["markdown", "firecrawl"])

    result = await retriever.acall("https://example.com/page")
    assert result.data[0]["data"]["content"] == "# From Firecrawl"

    serialized = retriever.serialize()
    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    restored_result = await restored.acall("https://two.example/page")
    assert restored_result.data[0]["data"]["content"] == "# From Firecrawl"

    await retriever.aclose()
    await restored.aclose()


@pytest.mark.asyncio
async def test_async_fallback_recovery(mocker):
    mock = install_httpx2_mock(mocker, _recovering_handler)
    retriever = FallbackWebFetchRetriever()

    result = await retriever.acall("https://example.com/page")

    assert result.data[0]["data"]["content"] == "# From Firecrawl"
    assert [request.url.host for request in mock.requests] == [
        "markdown.new",
        "api.firecrawl.dev",
    ]

    await retriever.aclose()


@pytest.mark.asyncio
async def test_async_cancellation_propagates(mocker):
    async def cancelled(request):
        raise asyncio.CancelledError

    install_httpx2_mock(mocker, cancelled)
    retriever = FallbackWebFetchRetriever()

    with pytest.raises(asyncio.CancelledError):
        await retriever.acall("https://example.com")

    await retriever.aclose()


@pytest.mark.asyncio
async def test_serialization_rebuilds_children_and_closes_all_clients(mocker):
    install_httpx2_mock(mocker, _markdown_ok)
    retriever = FallbackWebFetchRetriever(timeout=4)

    serialized = retriever.serialize()

    assert "children" not in serialized["state"]
    assert [spec["provider"] for spec in serialized["state"]["retrievers"]] == [
        "markdown",
        "firecrawl",
    ]
    assert serialized["state"]["retrievers"][0]["state"]["timeout"] == 4

    restored = Retriever.from_serialized(
        serialized["provider"],
        serialized["retriever_type"],
        serialized["state"],
    )

    assert [type(child).__name__ for child in restored.children] == [
        "MarkdownWebFetchRetriever",
        "FirecrawlWebFetchRetriever",
    ]
    assert restored.children[0].client is not retriever.children[0].client
    assert restored.children[1].client is not retriever.children[1].client

    await retriever.aclose()
    await restored.aclose()

    for child in (*retriever.children, *restored.children):
        assert child.client.is_closed is True
        assert child.aclient.is_closed is True
