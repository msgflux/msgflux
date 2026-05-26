import asyncio
from os import getenv
from typing import Any, List, Mapping, Optional

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

API_KEY_ENV_NAME = "CERAMIC_API_KEY"
CERAMIC_SEARCH_URL = "https://api.ceramic.ai/search"


@register_retriever
class CeramicWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A Ceramic client for lexical web search retrieval.

    Ceramic Search is a web search provider based on lexical query matching.
    """

    provider = "ceramic"

    def __init__(
        self,
        *,
        timeout: Optional[float] = None,
    ):
        """Initialize CeramicWebRetriever.

        Requires the `CERAMIC_API_KEY` environment variable to be set.

        Args:
            timeout:
                Request timeout in seconds. Defaults to 30.
        """
        self._ensure_httpx()
        self.timeout = timeout or 30.0
        self.client = httpx.Client(timeout=self.timeout)
        self.aclient = httpx.AsyncClient(timeout=self.timeout)

    def _ensure_httpx(self) -> None:
        if httpx is None:
            raise ImportError(
                "The 'httpx' package is not installed. "
                "Please install it via pip: pip install httpx"
            )

    def _get_api_key(self) -> str:
        api_key = getenv(API_KEY_ENV_NAME)
        if api_key:
            return api_key

        raise ValueError(
            f"The Ceramic API key is not available. Please set `{API_KEY_ENV_NAME}`"
        )

    def _build_headers(self) -> dict:
        api_key = self._get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, query: str) -> dict:
        return {"query": query}

    def _format_result(self, result: Mapping[str, Any]) -> dict:
        data = {
            "title": result.get("title"),
            "content": result.get("description"),
            "url": result.get("url"),
        }
        return {"data": data}

    def _parse_results(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        result = response.get("result") or {}
        results = result.get("results") or []
        return [self._format_result(item) for item in results[:top_k]]

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Ceramic for a single query."""
        try:
            response = self.client.post(
                CERAMIC_SEARCH_URL,
                headers=self._build_headers(),
                json=self._build_payload(query),
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_results(data, top_k)
        except Exception as e:
            logger.warning("Ceramic search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Ceramic for a single query."""
        try:
            response = await self.aclient.post(
                CERAMIC_SEARCH_URL,
                headers=self._build_headers(),
                json=self._build_payload(query),
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_results(data, top_k)
        except Exception as e:
            logger.warning("Ceramic search failed for query '%s': %s", query, e)
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
