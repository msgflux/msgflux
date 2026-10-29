"""Unit tests for msgflux.tools.builtin.web_fetch."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.builtin.web_fetch import WebFetchTool


def _fetch_result(url="https://example.com", content="# Markdown content"):
    return dotdict(
        {
            "response_type": "web_fetch",
            "data": [{"data": {"url": url, "content": content}}],
        }
    )


class TestWebFetchToolInit:
    """Tests for WebFetchTool.__init__ and class attributes."""

    def test_name_attribute(self):
        assert WebFetchTool.name == "web_fetch"
        assert WebFetchTool.display_name == "Web Fetch"

    def test_defaults_build_fallback_retriever(self, mocker):
        sentinel = object()
        factory = mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=sentinel,
        )

        tool = WebFetchTool()

        assert tool.web_parser == "https://markdown.new/"
        assert tool.default_headers == {}
        assert tool.timeout == 30
        assert tool.retriever is sentinel
        factory.assert_called_once_with(
            "fallback",
            web_parser="https://markdown.new/",
            default_headers={},
            timeout=30,
        )

    def test_custom_parser_url_adds_trailing_slash(self, mocker):
        factory = mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=MagicMock(),
        )

        tool = WebFetchTool(web_parser="https://custom.parser")

        assert tool.web_parser == "https://custom.parser/"
        assert factory.call_args.kwargs["web_parser"] == "https://custom.parser/"

    def test_custom_parser_url_keeps_single_trailing_slash(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=MagicMock(),
        )

        tool = WebFetchTool(web_parser="https://custom.parser/")

        assert tool.web_parser == "https://custom.parser/"

    def test_custom_headers_are_copied_and_forwarded(self, mocker):
        factory = mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=MagicMock(),
        )
        headers = {"Authorization": "Bearer token"}

        tool = WebFetchTool(default_headers=headers)

        assert tool.default_headers == headers
        assert tool.default_headers is not headers
        assert factory.call_args.kwargs["default_headers"] == headers

    def test_none_headers_becomes_empty_dict(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=MagicMock(),
        )

        tool = WebFetchTool(default_headers=None)

        assert tool.default_headers == {}

    def test_custom_timeout_is_forwarded(self, mocker):
        factory = mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=MagicMock(),
        )

        tool = WebFetchTool(timeout=5)

        assert tool.timeout == 5
        assert factory.call_args.kwargs["timeout"] == 5

    def test_injected_retriever_bypasses_factory(self, mocker):
        factory = mocker.patch("msgflux.tools.builtin.web_fetch.Retriever.web_fetch")
        retriever = MagicMock()

        tool = WebFetchTool(retriever=retriever)

        assert tool.retriever is retriever
        factory.assert_not_called()

    def test_tool_library_uses_builtin_display_name(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.Retriever.web_fetch",
            return_value=MagicMock(),
        )
        library = ToolLibrary(name="web", tools=[WebFetchTool])

        assert library.get_tool_display_names()["web_fetch"] == "Web Fetch"


class TestWebFetchToolCall:
    """Tests for WebFetchTool.__call__ (sync)."""

    def test_returns_content_from_standard_result(self):
        retriever = MagicMock(return_value=_fetch_result())

        tool = WebFetchTool(retriever=retriever)
        result = tool("https://example.com")

        assert result == "# Markdown content"
        retriever.assert_called_once_with("https://example.com")

    def test_accepts_url_without_scheme(self):
        retriever = MagicMock(return_value=_fetch_result("https://example.com/page"))

        tool = WebFetchTool(retriever=retriever)
        result = tool("example.com/page")

        assert result == "# Markdown content"
        retriever.assert_called_once_with("example.com/page")

    def test_accepts_plain_string_result(self):
        retriever = MagicMock(return_value="# direct text")

        tool = WebFetchTool(retriever=retriever)

        assert tool("https://example.com") == "# direct text"

    def test_missing_data_raises_clear_error(self):
        retriever = MagicMock(
            return_value=dotdict({"response_type": "web_fetch", "data": []})
        )

        tool = WebFetchTool(retriever=retriever)

        with pytest.raises(RuntimeError, match="empty retriever response"):
            tool("https://example.com")

    def test_missing_content_raises_clear_error(self):
        retriever = MagicMock(
            return_value=dotdict(
                {
                    "response_type": "web_fetch",
                    "data": [{"data": {"url": "https://example.com"}}],
                }
            )
        )

        tool = WebFetchTool(retriever=retriever)

        with pytest.raises(RuntimeError, match="missing fetched content"):
            tool("https://example.com")


class TestWebFetchToolAcall:
    """Tests for WebFetchTool.acall (async)."""

    @pytest.mark.asyncio
    async def test_returns_content_from_standard_result(self):
        retriever = MagicMock()
        retriever.acall = AsyncMock(return_value=_fetch_result())

        tool = WebFetchTool(retriever=retriever)
        result = await tool.acall("https://example.com")

        assert result == "# Markdown content"
        retriever.acall.assert_awaited_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_accepts_plain_string_result(self):
        retriever = MagicMock()
        retriever.acall = AsyncMock(return_value="# async text")

        tool = WebFetchTool(retriever=retriever)

        assert await tool.acall("https://example.com") == "# async text"


class TestWebFetchToolLifecycle:
    def test_close_delegates_to_retriever(self):
        retriever = MagicMock()

        WebFetchTool(retriever=retriever).close()

        retriever.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_aclose_delegates_to_retriever(self):
        retriever = MagicMock()
        retriever.aclose = AsyncMock()

        await WebFetchTool(retriever=retriever).aclose()

        retriever.aclose.assert_awaited_once_with()
