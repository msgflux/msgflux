from urllib.parse import urlparse

from msgflux.utils.html import html_to_text

try:
    import httpx
except ImportError:
    httpx = None


class WebFetchTool:
    """Fetches web pages as Markdown via a remote parser endpoint."""

    name = "web_fetch"
    display_name = "Web Fetch"

    def __init__(
        self,
        web_parser: str = "https://markdown.new/",
        default_headers: dict | None = None,
        timeout: int = 1,
    ):
        """Args:
        web_parser: Parser endpoint URL (must include scheme).
        default_headers: Headers applied to every request.
        timeout: Request timeout in seconds.
        """
        self.web_parser = web_parser.rstrip("/") + "/"
        self.default_headers = default_headers or {}
        self.timeout = timeout

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        return url if parsed.scheme else f"https://{url}"

    def _build_url(self, url: str) -> str:
        return f"{self.web_parser}{self._normalize_url(url)}"

    def __call__(self, url: str) -> str:
        """Fetch and return web content as a string.

        Tries the parser endpoint first. On failure, fetches raw HTML
        and extracts clean text via semantic chunking as fallback.

        Args:
            url: Target website URL.

        Returns:
            Parsed response body as a string.

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: On network or HTTP failure.
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for WebFetchTool."
                " Install it with: pip install msgflux[httpx]"
            )

        final_url = self._build_url(url)
        try:
            response = httpx.get(
                final_url,
                headers=self.default_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            pass

        target_url = self._normalize_url(url)
        try:
            response = httpx.get(
                target_url,
                headers=self.default_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return html_to_text(response.text)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    async def acall(self, url: str) -> str:
        """Asynchronously fetch and return web content as a string.

        Tries the parser endpoint first. On failure, fetches raw HTML
        and extracts clean text via semantic chunking as fallback.

        Args:
            url: Target website URL.

        Returns:
            Parsed response body as a string.

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: On network or HTTP failure.
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for WebFetchTool."
                " Install it with: pip install msgflux[httpx]"
            )

        final_url = self._build_url(url)
        target_url = self._normalize_url(url)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(final_url, headers=self.default_headers)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError:
                pass

            try:
                response = await client.get(target_url, headers=self.default_headers)
                response.raise_for_status()
                return html_to_text(response.text)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
