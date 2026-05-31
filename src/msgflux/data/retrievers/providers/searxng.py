import asyncio
from os import getenv
from typing import Any, List, Mapping, Optional
from urllib.parse import urljoin

try:
    import httpx
except ImportError:
    httpx = None

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.logger import init_logger

logger = init_logger(__name__)

BASE_URL_ENV_NAME = "SEARXNG_BASE_URL"
DEFAULT_BASE_URL = "http://localhost:8080"


@register_retriever
class SearXNGWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A SearXNG client for local or self-hosted metasearch retrieval."""

    provider = "searxng"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        categories: Optional[str] = None,
        engines: Optional[str] = None,
        language: Optional[str] = None,
        time_range: Optional[str] = None,
        safesearch: Optional[int] = None,
        pageno: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize SearXNGWebRetriever.

        Args:
            base_url:
                Base URL for a SearXNG instance. Defaults to `SEARXNG_BASE_URL`
                or `http://localhost:8080`.
            categories:
                Comma-separated active search categories.
            engines:
                Comma-separated active search engines.
            language:
                Search language code.
            time_range:
                Time range filter: "day", "month", or "year".
            safesearch:
                Safe search level: 0 (none), 1 (moderate), or 2 (strict).
            pageno:
                Search page number.
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        self._ensure_httpx()

        self.base_url = (
            base_url or getenv(BASE_URL_ENV_NAME) or DEFAULT_BASE_URL
        ).rstrip("/")
        self.search_url = urljoin(f"{self.base_url}/", "search")
        self.categories = categories
        self.engines = engines
        self.language = language
        self.time_range = time_range
        self.safesearch = safesearch
        self.pageno = pageno
        self.timeout = timeout or 30.0
        self.client = httpx.Client(timeout=self.timeout)
        self.aclient = httpx.AsyncClient(timeout=self.timeout)

    def _ensure_httpx(self) -> None:
        if httpx is None:
            raise ImportError(
                "The 'httpx' package is not installed. "
                "Please install it via pip: pip install httpx"
            )

    def _build_search_params(self, query: str) -> dict:
        params = {
            "q": query,
            "format": "json",
        }

        if self.categories:
            params["categories"] = self.categories
        if self.engines:
            params["engines"] = self.engines
        if self.language:
            params["language"] = self.language
        if self.time_range:
            params["time_range"] = self.time_range
        if self.safesearch is not None:
            params["safesearch"] = self.safesearch
        if self.pageno is not None:
            params["pageno"] = self.pageno

        return params

    def _format_result(self, result: Mapping[str, Any]) -> dict:
        data = {
            "title": result.get("title"),
            "content": result.get("content"),
            "url": result.get("url"),
        }

        for field in ("engine", "category", "publishedDate"):
            if result.get(field):
                data[field] = result[field]

        item = {"data": data}
        image_url = result.get("thumbnail") or result.get("img_src")
        if image_url:
            item["images"] = [image_url]

        return item

    def _parse_results(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        results = []
        for result in response.get("results", []):
            results.append(self._format_result(result))
            if len(results) >= top_k:
                return results
        return results

    def _parse_response_data(
        self, data: Mapping[str, Any], query: str, top_k: int
    ) -> List[dict]:
        if data.get("error"):
            logger.warning(
                "SearXNG search failed for query '%s': %s",
                query,
                data["error"],
            )
            return []

        return self._parse_results(data, top_k)

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search SearXNG for a single query."""
        try:
            response = self.client.get(
                self.search_url,
                params=self._build_search_params(query),
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_response_data(data, query, top_k)
        except Exception as e:
            logger.warning("SearXNG search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search SearXNG for a single query."""
        try:
            response = await self.aclient.get(
                self.search_url,
                params=self._build_search_params(query),
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_response_data(data, query, top_k)
        except Exception as e:
            logger.warning("SearXNG search failed for query '%s': %s", query, e)
            return []

    def close(self) -> None:
        """Close the reusable synchronous HTTP client."""
        self.client.close()

    async def aclose(self) -> None:
        """Close the reusable asynchronous HTTP client."""
        await self.aclient.aclose()

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Synchronous search for multiple queries."""
        results = []
        for query in queries:
            query_results = self._single_search(query, top_k)
            results.append(dotdict({"results": query_results}))
        return results

    async def _asearch(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Async search that runs multiple queries in parallel."""
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results
