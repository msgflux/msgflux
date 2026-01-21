import asyncio
from typing import List

try:
    import httpx
except ImportError:
    httpx = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class TrafilaturaWebRetriever(BaseWebFetch, BaseRetriever, WebRetriever):
    """A lightweight web fetcher using httpx + trafilatura for text extraction.

    This class fetches web pages with httpx and uses trafilatura to extract
    clean, readable text content from HTML.
    """

    provider = "trafilatura"

    def __init__(
        self,
        *,
        include_comments: bool = False,
        include_tables: bool = True,
        output_format: str = "txt",
        timeout: float = 30.0,
        follow_redirects: bool = True,
    ):
        """Initialize TrafilaturaWebRetriever.

        Args:
            include_comments:
                Whether to include comments in extracted text.
                Defaults to False.
            include_tables:
                Whether to include tables in extracted text.
                Defaults to True.
            output_format:
                Output format. Can be "txt", "markdown", "xml", "json".
                Defaults to "txt".
            timeout:
                HTTP request timeout in seconds.
                Defaults to 30.0.
            follow_redirects:
                Whether to follow HTTP redirects.
                Defaults to True.

        !!! example

            ```python
            retriever = TrafilaturaWebRetriever(output_format="markdown")
            results = retriever(["https://example.com"])
            print(results)
            ```
        """
        if httpx is None:
            raise ImportError(
                "The 'httpx' package is not installed. "
                "Please install it via pip: pip install httpx"
            )
        if trafilatura is None:
            raise ImportError(
                "The 'trafilatura' package is not installed. "
                "Please install it via pip: pip install trafilatura"
            )

        self.include_comments = include_comments
        self.include_tables = include_tables
        self.output_format = output_format
        self.timeout = timeout
        self.follow_redirects = follow_redirects

    def _extract_content(self, html: str, url: str) -> dict:
        """Extract content from HTML using trafilatura."""
        try:
            content = trafilatura.extract(
                html,
                include_comments=self.include_comments,
                include_tables=self.include_tables,
                output_format=self.output_format,
                url=url,
            )

            # Get metadata
            metadata = trafilatura.extract_metadata(html)

            return {
                "url": url,
                "content": content,
                "title": metadata.title if metadata else None,
                "author": metadata.author if metadata else None,
                "date": metadata.date if metadata else None,
                "success": content is not None,
            }
        except Exception as e:
            logger.warning("Trafilatura extraction failed for '%s': %s", url, e)
            return {
                "url": url,
                "content": None,
                "title": None,
                "success": False,
                "error": str(e),
            }

    def _single_fetch(self, url: str) -> dict:
        """Synchronously fetch a single URL."""
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                html = response.text
                return {"data": self._extract_content(html, url)}
        except Exception as e:
            logger.warning("HTTP fetch failed for '%s': %s", url, e)
            return {
                "data": {
                    "url": url,
                    "content": None,
                    "title": None,
                    "success": False,
                    "error": str(e),
                }
            }

    async def _asingle_fetch(self, url: str) -> dict:
        """Async fetch a single URL."""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
                return {"data": self._extract_content(html, url)}
        except Exception as e:
            logger.warning("HTTP fetch failed for '%s': %s", url, e)
            return {
                "data": {
                    "url": url,
                    "content": None,
                    "title": None,
                    "success": False,
                    "error": str(e),
                }
            }

    def _fetch(self, urls: List[str]) -> List[dotdict]:
        """Synchronous fetch for multiple URLs."""
        results = []
        for url in urls:
            result = self._single_fetch(url)
            results.append(dotdict({"results": [result]}))
        return results

    async def _afetch(self, urls: List[str]) -> List[dotdict]:
        """Async fetch that runs multiple URLs in parallel."""
        tasks = [self._asingle_fetch(url) for url in urls]
        fetch_results = await asyncio.gather(*tasks)
        results = []
        for result in fetch_results:
            results.append(dotdict({"results": [result]}))
        return results
