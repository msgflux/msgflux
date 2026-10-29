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

API_KEY_ENV_NAME = "TAVILY_API_KEY"
DEFAULT_BASE_URL = "https://api.tavily.com"


@register_retriever
class TavilyWebRetriever(
    HTTPRetrieverMixin, BaseWebSearch, BaseRetriever, WebSearchRetriever
):
    """A Tavily client for retrieving web content with AI-powered search.

    This class interfaces with the Tavily API to provide search results
    optimized for AI applications.
    """

    provider = "tavily"

    def __init__(
        self,
        *,
        search_depth: Optional[str] = None,
        topic: Optional[str] = None,
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Optional[bool] = None,
        include_images: Optional[bool] = None,
        include_raw_content: Optional[bool] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize TavilyWebRetriever.

        Requires the `TAVILY_API_KEY` environment variable to be set when a
        request is made.

        Args:
            search_depth:
                Search depth. Can be "basic" or "advanced". Defaults to "basic".
            topic:
                Topic category. Can be "general", "news", or "finance".
                Defaults to "general".
            time_range:
                Time range filter. Can be "day", "week", "month", "year" or
                shortcuts "d", "w", "m", "y". Defaults to None.
            include_domains:
                List of domains to restrict search to. Defaults to None.
            exclude_domains:
                List of domains to exclude from search. Defaults to None.
            include_answer:
                Whether to include AI-generated answer. Defaults to False.
            include_images:
                Whether to include images in results. Defaults to False.
            include_raw_content:
                Whether to include raw page content. Defaults to False.
            base_url:
                Tavily API base URL. Defaults to "https://api.tavily.com".
            timeout:
                Request timeout in seconds. Defaults to 30.

        !!! example

            ```python
            retriever = TavilyWebRetriever(search_depth="advanced", topic="news")
            results = retriever(["latest AI news"], top_k=5)
            print(results)
            ```
        """
        self.search_depth = search_depth or "basic"
        self.topic = topic or "general"
        self.time_range = time_range
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self.include_answer = include_answer or False
        self.include_images = include_images or False
        self.include_raw_content = include_raw_content or False
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.search_url = f"{self.base_url}/search"
        self.timeout = timeout or 30.0
        self._initialize()

    def _get_api_key(self) -> str:
        api_key = getenv(API_KEY_ENV_NAME)
        if api_key:
            return api_key

        raise ValueError(
            f"The Tavily API key is not available. Please set `{API_KEY_ENV_NAME}`"
        )

    def _build_headers(self) -> dict:
        api_key = self._get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_search_kwargs(self, top_k: int) -> dict:
        """Build REST payload fields for Tavily search requests."""
        kwargs = {
            "max_results": top_k,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "include_answer": self.include_answer,
            "include_images": self.include_images,
            "include_raw_content": self.include_raw_content,
        }

        if self.time_range:
            kwargs["time_range"] = self.time_range
        if self.include_domains:
            kwargs["include_domains"] = self.include_domains
        if self.exclude_domains:
            kwargs["exclude_domains"] = self.exclude_domains

        return kwargs

    def _build_payload(self, query: str, top_k: int) -> dict:
        return {"query": query, **self._build_search_kwargs(top_k)}

    def _parse_results(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        """Parse Tavily response into standard format."""
        results = []
        for result in response.get("results", [])[:top_k]:
            data = {
                "title": result.get("title"),
                "content": result.get("content"),
                "url": result.get("url"),
            }

            # Add published date if available
            if result.get("published_date"):
                data["date"] = result["published_date"]

            # Add raw content if available
            if result.get("raw_content"):
                data["raw_content"] = result["raw_content"]

            item = {"data": data}

            # Add score if available
            if result.get("score"):
                item["score"] = result["score"]

            results.append(item)

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Tavily for a single query."""
        try:
            response = self.client.post(
                self.search_url,
                headers=self._build_headers(),
                json=self._build_payload(query, top_k),
            )
            response.raise_for_status()

            return self._parse_results(response.json(), top_k)
        except Exception as e:
            logger.warning("Tavily search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Tavily for a single query."""
        try:
            response = await self.aclient.post(
                self.search_url,
                headers=self._build_headers(),
                json=self._build_payload(query, top_k),
            )
            response.raise_for_status()

            return self._parse_results(response.json(), top_k)
        except Exception as e:
            logger.warning("Tavily search failed for query '%s': %s", query, e)
            return []

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
