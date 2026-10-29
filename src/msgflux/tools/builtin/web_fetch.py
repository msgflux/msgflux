"""Fetch a known URL through the configured web fetch retriever."""

from collections.abc import Mapping
from typing import Any, Optional

from msgflux.data.retrievers.retriever import Retriever

DEFAULT_WEB_PARSER = "https://markdown.new/"


class WebFetchTool:
    """Fetch a known URL and return the page as text."""

    name = "web_fetch"
    display_name = "Web Fetch"
    default_usage_guidance = (
        "Use when the user provides a URL or asks about a specific web page. "
        "Fetch the page before answering questions about its contents."
    )

    def __init__(
        self,
        web_parser: Optional[str] = DEFAULT_WEB_PARSER,
        default_headers: Optional[dict] = None,
        timeout: Optional[float] = 30,
        *,
        retriever: Any = None,
    ):
        """Args:
        web_parser: markdown parser endpoint URL used by the default chain.
        default_headers: Headers applied to default parser requests.
        timeout: Request timeout in seconds. Defaults to 30.
        retriever: Optional retriever instance; overrides the default chain.
        """
        self.web_parser = (web_parser or DEFAULT_WEB_PARSER).rstrip("/") + "/"
        self.default_headers = dict(default_headers or {})
        self.timeout = timeout

        if retriever is None:
            self.retriever = Retriever.web_fetch(
                "fallback",
                web_parser=self.web_parser,
                default_headers=self.default_headers,
                timeout=self.timeout,
            )
        else:
            self.retriever = retriever

    @staticmethod
    def _content_from_result(url: str, result: Any) -> str:
        if isinstance(result, str):
            return result

        data = getattr(result, "data", None)
        if data is None and isinstance(result, Mapping):
            data = result.get("data")
        if not data:
            raise RuntimeError(f"Failed to fetch {url}: empty retriever response")

        entry = data[0]
        if isinstance(entry, str):
            return entry
        if isinstance(entry, Mapping):
            payload = entry.get("data", entry)
            if isinstance(payload, Mapping) and payload.get("content") is not None:
                return payload["content"]

        raise RuntimeError(f"Failed to fetch {url}: missing fetched content")

    def __call__(self, url: str) -> str:
        """Fetch a URL and return parsed page text.

        Args:
            url: Page URL, with or without an ``https://`` scheme.
        """
        result = self.retriever(url)
        return self._content_from_result(url, result)

    async def acall(self, url: str) -> str:
        """Fetch a URL asynchronously and return parsed page text.

        Args:
            url: Page URL, with or without an ``https://`` scheme.
        """
        result = await self.retriever.acall(url)
        return self._content_from_result(url, result)

    def close(self) -> None:
        """Close the retriever's synchronous HTTP clients."""
        close = getattr(self.retriever, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        """Close the retriever's async and synchronous HTTP clients."""
        aclose = getattr(self.retriever, "aclose", None)
        if callable(aclose):
            await aclose()
            return
        self.close()
