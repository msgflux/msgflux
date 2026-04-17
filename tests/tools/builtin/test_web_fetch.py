"""Unit tests for msgflux.tools.builtin.web_fetch."""

import httpx
import pytest

import msgflux.tools.builtin.web_fetch as web_fetch_module
from msgflux.tools.builtin.web_fetch import WebFetch


class TestWebFetchInit:
    """Tests for WebFetch.__init__ and class attributes."""

    def test_name_attribute(self):
        assert WebFetch.name == "web_fetch"

    def test_defaults(self):
        tool = WebFetch()
        assert tool.web_parser == "https://markdown.new/"
        assert tool.default_headers == {}
        assert tool.timeout == 1

    def test_custom_parser_url_adds_trailing_slash(self):
        tool = WebFetch(web_parser="https://custom.parser")
        assert tool.web_parser == "https://custom.parser/"

    def test_custom_parser_url_keeps_single_trailing_slash(self):
        tool = WebFetch(web_parser="https://custom.parser/")
        assert tool.web_parser == "https://custom.parser/"

    def test_custom_headers(self):
        headers = {"Authorization": "Bearer token"}
        tool = WebFetch(default_headers=headers)
        assert tool.default_headers == headers

    def test_none_headers_becomes_empty_dict(self):
        tool = WebFetch(default_headers=None)
        assert tool.default_headers == {}

    def test_custom_timeout(self):
        tool = WebFetch(timeout=5)
        assert tool.timeout == 5


class TestBuildUrl:
    """Tests for WebFetch._build_url."""

    def test_url_with_https_scheme(self):
        tool = WebFetch()
        assert (
            tool._build_url("https://example.com")
            == "https://markdown.new/https://example.com"
        )

    def test_url_with_http_scheme(self):
        tool = WebFetch()
        assert (
            tool._build_url("http://example.com")
            == "https://markdown.new/http://example.com"
        )

    def test_url_without_scheme_adds_https(self):
        tool = WebFetch()
        assert (
            tool._build_url("example.com") == "https://markdown.new/https://example.com"
        )

    def test_url_with_path(self):
        tool = WebFetch()
        result = tool._build_url("https://example.com/some/path?q=1")
        assert result == "https://markdown.new/https://example.com/some/path?q=1"

    def test_custom_parser(self):
        tool = WebFetch(web_parser="https://r.jina.ai/")
        result = tool._build_url("https://example.com")
        assert result == "https://r.jina.ai/https://example.com"


class TestWebFetchCall:
    """Tests for WebFetch.__call__ (sync)."""

    def test_success_returns_text(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "# Markdown content"
        mock_get = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            return_value=mock_response,
        )

        tool = WebFetch()
        result = tool("https://example.com")

        assert result == "# Markdown content"
        mock_get.assert_called_once_with(
            "https://markdown.new/https://example.com",
            headers={},
            timeout=1,
        )

    def test_passes_default_headers(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "content"
        mock_get = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            return_value=mock_response,
        )

        headers = {"X-Custom": "value"}
        tool = WebFetch(default_headers=headers)
        tool("https://example.com")

        mock_get.assert_called_once_with(
            "https://markdown.new/https://example.com",
            headers=headers,
            timeout=1,
        )

    def test_passes_custom_timeout(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "content"
        mock_get = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            return_value=mock_response,
        )

        tool = WebFetch(timeout=5)
        tool("https://example.com")

        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 5

    def test_url_without_scheme_is_normalized(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "content"
        mock_get = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            return_value=mock_response,
        )

        tool = WebFetch()
        tool("example.com/page")

        args, _ = mock_get.call_args
        assert args[0] == "https://markdown.new/https://example.com/page"

    def test_http_error_raises_runtime_error(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            side_effect=httpx.HTTPError("connection refused"),
        )

        tool = WebFetch()
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            tool("https://example.com")

    def test_runtime_error_message_contains_url(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            side_effect=httpx.HTTPError("timeout"),
        )

        tool = WebFetch()
        with pytest.raises(RuntimeError, match="https://example.com"):
            tool("https://example.com")

    def test_raises_import_error_when_httpx_unavailable(self, mocker):
        mocker.patch.object(web_fetch_module, "httpx", None)

        tool = WebFetch()
        with pytest.raises(ImportError, match="httpx is required"):
            tool("https://example.com")

    def test_import_error_message_contains_install_hint(self, mocker):
        mocker.patch.object(web_fetch_module, "httpx", None)

        tool = WebFetch()
        with pytest.raises(ImportError, match="pip install msgflux"):
            tool("https://example.com")


class TestWebFetchFallback:
    """Tests for WebFetch fallback: parser fails → raw HTML + html_to_text."""

    def test_fallback_called_on_parser_failure(self, mocker):
        raw_html = "<html><body><p>Hello</p></body></html>"
        mock_response = mocker.Mock()
        mock_response.text = raw_html
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            side_effect=[httpx.HTTPError("parser down"), mock_response],
        )
        mock_html_to_text = mocker.patch(
            "msgflux.tools.builtin.web_fetch.html_to_text",
            return_value="Hello",
        )

        tool = WebFetch()
        result = tool("https://example.com")

        assert result == "Hello"
        mock_html_to_text.assert_called_once_with(raw_html)

    def test_fallback_fetches_raw_url_not_parser_url(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "<html></html>"
        mock_get = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            side_effect=[httpx.HTTPError("parser down"), mock_response],
        )
        mocker.patch("msgflux.tools.builtin.web_fetch.html_to_text", return_value="ok")

        tool = WebFetch()
        tool("https://example.com")

        first_url = mock_get.call_args_list[0][0][0]
        second_url = mock_get.call_args_list[1][0][0]
        assert first_url == "https://markdown.new/https://example.com"
        assert second_url == "https://example.com"

    def test_fallback_normalizes_url_without_scheme(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "<html></html>"
        mock_get = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            side_effect=[httpx.HTTPError("parser down"), mock_response],
        )
        mocker.patch("msgflux.tools.builtin.web_fetch.html_to_text", return_value="ok")

        tool = WebFetch()
        tool("example.com")

        second_url = mock_get.call_args_list[1][0][0]
        assert second_url == "https://example.com"

    def test_fallback_failure_raises_runtime_error(self, mocker):
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.get",
            side_effect=httpx.HTTPError("all down"),
        )

        tool = WebFetch()
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            tool("https://example.com")

    @pytest.mark.asyncio
    async def test_async_fallback_called_on_parser_failure(self, mocker):
        raw_html = "<html><body><p>Hello</p></body></html>"
        ok_response = mocker.Mock()
        ok_response.text = raw_html

        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(
            side_effect=[httpx.HTTPError("parser down"), ok_response]
        )
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )
        mock_html_to_text = mocker.patch(
            "msgflux.tools.builtin.web_fetch.html_to_text",
            return_value="Hello",
        )

        tool = WebFetch()
        result = await tool.acall("https://example.com")

        assert result == "Hello"
        mock_html_to_text.assert_called_once_with(raw_html)

    @pytest.mark.asyncio
    async def test_async_fallback_fetches_raw_url(self, mocker):
        ok_response = mocker.Mock()
        ok_response.text = "<html></html>"

        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(
            side_effect=[httpx.HTTPError("parser down"), ok_response]
        )
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )
        mocker.patch("msgflux.tools.builtin.web_fetch.html_to_text", return_value="ok")

        tool = WebFetch()
        await tool.acall("https://example.com")

        first_url = mock_client.get.call_args_list[0][0][0]
        second_url = mock_client.get.call_args_list[1][0][0]
        assert first_url == "https://markdown.new/https://example.com"
        assert second_url == "https://example.com"

    @pytest.mark.asyncio
    async def test_async_fallback_failure_raises_runtime_error(self, mocker):
        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(side_effect=httpx.HTTPError("all down"))
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )

        tool = WebFetch()
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            await tool.acall("https://example.com")


class TestWebFetchAcall:
    """Tests for WebFetch.acall (async)."""

    @pytest.mark.asyncio
    async def test_success_returns_text(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "# Async Markdown"

        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )

        tool = WebFetch()
        result = await tool.acall("https://example.com")

        assert result == "# Async Markdown"
        mock_client.get.assert_called_once_with(
            "https://markdown.new/https://example.com",
            headers={},
        )

    @pytest.mark.asyncio
    async def test_passes_default_headers(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "content"

        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )

        headers = {"X-Custom": "value"}
        tool = WebFetch(default_headers=headers)
        await tool.acall("https://example.com")

        mock_client.get.assert_called_once_with(
            "https://markdown.new/https://example.com",
            headers=headers,
        )

    @pytest.mark.asyncio
    async def test_uses_timeout_for_async_client(self, mocker):
        mock_response = mocker.Mock()
        mock_response.text = "content"

        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_async_client_cls = mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )

        tool = WebFetch(timeout=3)
        await tool.acall("https://example.com")

        mock_async_client_cls.assert_called_once_with(timeout=3)

    @pytest.mark.asyncio
    async def test_http_error_raises_runtime_error(self, mocker):
        mock_client = mocker.AsyncMock()
        mock_client.get = mocker.AsyncMock(
            side_effect=httpx.HTTPError("connection refused")
        )
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch(
            "msgflux.tools.builtin.web_fetch.httpx.AsyncClient",
            return_value=mock_client,
        )

        tool = WebFetch()
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            await tool.acall("https://example.com")

    @pytest.mark.asyncio
    async def test_raises_import_error_when_httpx_unavailable(self, mocker):
        mocker.patch.object(web_fetch_module, "httpx", None)

        tool = WebFetch()
        with pytest.raises(ImportError, match="httpx is required"):
            await tool.acall("https://example.com")
