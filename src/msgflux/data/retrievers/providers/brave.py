import asyncio
from os import getenv
from typing import Any, List, Mapping, Optional

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebSearchRetriever
from msgflux.logger import init_logger

logger = init_logger(__name__)

API_KEY_ENV_NAME = "BRAVE_SEARCH_API_KEY"
DEFAULT_BASE_URL = "https://api.search.brave.com"
SEARCH_PATH_BY_MODE = {
    "search": "/res/v1/web/search",
    "news": "/res/v1/news/search",
    "image": "/res/v1/images/search",
}


@register_retriever
class BraveWebRetriever(
    HTTPRetrieverMixin, BaseWebSearch, BaseRetriever, WebSearchRetriever
):
    """A Brave Search client for retrieving web, image, and news content.

    This class interfaces with the Brave Search API to provide grounded search results.
    """

    provider = "brave"

    def __init__(
        self,
        *,
        mode: str = "search",
        return_images: Optional[bool] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize BraveWebRetriever.

        Requires the `BRAVE_SEARCH_API_KEY` environment variable to be set when
        a request is made.

        Args:
            mode:
                Search mode. Can be "search", "image", or "news". Defaults to "search".
            return_images:
                Whether to include images in the results. Defaults to False.
            base_url:
                Brave Search API base URL.
                Defaults to "https://api.search.brave.com".
            timeout:
                Request timeout in seconds. Defaults to 30.

        !!! example

            ```python
            retriever = BraveWebRetriever(mode="search")
            results = retriever(["latest AI news"], top_k=3)
            print(results)
            ```
        """
        self.mode = mode
        self.return_images = return_images or False
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or 30.0
        self._initialize()

    def _get_api_key(self) -> str:
        api_key = getenv(API_KEY_ENV_NAME)
        if api_key:
            return api_key

        raise ValueError(
            f"The Brave Search API key is not available. "
            f"Please set `{API_KEY_ENV_NAME}`"
        )

    def _build_headers(self) -> dict:
        api_key = self._get_api_key()
        return {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }

    def _get_search_url(self) -> str:
        path = SEARCH_PATH_BY_MODE.get(self.mode)
        if path is None:
            raise ValueError(f"Invalid mode: {self.mode}")
        return f"{self.base_url}{path}"

    def _build_search_params(self, query: str, top_k: int) -> dict:
        return {"q": query, "count": top_k}

    def _format_web_result(self, result: Mapping[str, Any]) -> dict:
        data = {
            "title": result.get("title"),
            "content": result.get("description"),
            "url": result.get("url"),
        }
        item = {"data": data}
        thumbnail = result.get("thumbnail") or {}
        image_url = thumbnail.get("src")
        if self.return_images and image_url:
            item["images"] = [image_url]
        return item

    def _format_news_result(self, result: Mapping[str, Any]) -> dict:
        data = {
            "title": result.get("title"),
            "content": result.get("description"),
            "url": result.get("url"),
            "date": result.get("age"),
        }
        item = {"data": data}
        thumbnail = result.get("thumbnail") or {}
        image_url = thumbnail.get("src")
        if self.return_images and image_url:
            item["images"] = [image_url]
        return item

    def _format_image_result(self, result: Mapping[str, Any]) -> dict:
        data = {"title": result.get("title"), "url": result.get("url")}
        # For image search, the item itself is the image
        item = {"data": data}
        image_url = result.get("url")
        if image_url:
            item["images"] = [image_url]
        return item

    def _parse_results(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        if self.mode == "search":
            raw_results = (response.get("web") or {}).get("results") or []
            formatter = self._format_web_result
        elif self.mode == "news":
            raw_results = response.get("results") or []
            formatter = self._format_news_result
        elif self.mode == "image":
            raw_results = response.get("results") or []
            formatter = self._format_image_result
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return [formatter(result) for result in raw_results[:top_k]]

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Brave for a single query."""
        try:
            response = self.client.get(
                self._get_search_url(),
                headers=self._build_headers(),
                params=self._build_search_params(query, top_k),
            )
            response.raise_for_status()

            return self._parse_results(response.json(), top_k)
        except Exception as e:
            logger.warning("Brave search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Brave for a single query."""
        try:
            response = await self.aclient.get(
                self._get_search_url(),
                headers=self._build_headers(),
                params=self._build_search_params(query, top_k),
            )
            response.raise_for_status()

            return self._parse_results(response.json(), top_k)
        except Exception as e:
            logger.warning("Brave search failed for query '%s': %s", query, e)
            return []

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Synchronous search for multiple queries."""
        results = []
        for query in queries:
            query_results = self._single_search(query, top_k)
            results.append(dotdict({"results": query_results}))
        return results

    async def _asearch(self, queries: List[str], top_k: int) -> List[dotdict]:
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results
