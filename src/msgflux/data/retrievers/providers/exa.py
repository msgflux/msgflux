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

API_KEY_ENV_NAME = "EXA_API_KEY"
DEFAULT_BASE_URL = "https://api.exa.ai"


@register_retriever
class ExaWebRetriever(
    HTTPRetrieverMixin, BaseWebSearch, BaseRetriever, WebSearchRetriever
):
    """An Exa.ai client for retrieving web content with semantic search.

    This class interfaces with the Exa API to provide semantic search results
    with optional content retrieval.
    """

    provider = "exa"

    def __init__(
        self,
        *,
        search_type: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[bool] = None,
        max_characters: Optional[int] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize ExaWebRetriever.

        Requires the `EXA_API_KEY` environment variable to be set when a
        request is made.

        Args:
            search_type:
                The type of search. Can be "auto" (default), "neural", "fast",
                or "deep". Auto intelligently combines methods, neural uses
                embeddings, fast is streamlined, deep provides comprehensive
                results with query expansion.
            include_domains:
                List of domains to restrict search to. Defaults to None.
            exclude_domains:
                List of domains to exclude from search. Defaults to None.
            start_published_date:
                Filter results published after this date (ISO 8601 format).
                Defaults to None.
            end_published_date:
                Filter results published before this date (ISO 8601 format).
                Defaults to None.
            include_text:
                Whether to include text content in results. Defaults to True.
            max_characters:
                Maximum characters of text to return per result.
                Defaults to None (full content).
            base_url:
                Exa API base URL. Defaults to "https://api.exa.ai".
            timeout:
                Request timeout in seconds. Defaults to 30.

        !!! example

            ```python
            retriever = ExaWebRetriever(include_text=True, max_characters=1000)
            results = retriever(["latest AI news"], top_k=3)
            print(results)
            ```
        """
        self.search_type = search_type or "auto"
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.start_published_date = start_published_date
        self.end_published_date = end_published_date
        self.include_text = include_text if include_text is not None else True
        self.max_characters = max_characters
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.search_url = f"{self.base_url}/search"
        self.timeout = timeout or 30.0
        self._initialize()

    def _get_api_key(self) -> str:
        api_key = getenv(API_KEY_ENV_NAME)
        if api_key:
            return api_key

        raise ValueError(
            f"The Exa API key is not available. Please set `{API_KEY_ENV_NAME}`"
        )

    def _build_headers(self) -> dict:
        api_key = self._get_api_key()
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _build_search_kwargs(self, top_k: int) -> dict:
        """Build REST payload fields for Exa search requests."""
        kwargs = {
            "numResults": top_k,
            "type": self.search_type,
        }

        if self.include_domains:
            kwargs["includeDomains"] = self.include_domains
        if self.exclude_domains:
            kwargs["excludeDomains"] = self.exclude_domains
        if self.start_published_date:
            kwargs["startPublishedDate"] = self.start_published_date
        if self.end_published_date:
            kwargs["endPublishedDate"] = self.end_published_date

        return kwargs

    def _build_text_options(self) -> dict:
        """Build content retrieval options for Exa."""
        if not self.include_text:
            return {}

        text_opts = {}
        if self.max_characters:
            text_opts["maxCharacters"] = self.max_characters

        return {"contents": {"text": text_opts if text_opts else True}}

    def _build_payload(self, query: str, top_k: int) -> dict:
        payload = {"query": query, **self._build_search_kwargs(top_k)}
        payload.update(self._build_text_options())
        return payload

    def _parse_response(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        """Parse an Exa response into msgflux web search result items."""
        results = []
        for result in response.get("results", [])[:top_k]:
            data = {
                "title": result.get("title"),
                "url": result.get("url"),
            }

            if result.get("text"):
                data["content"] = result["text"]

            published_date = result.get("publishedDate") or result.get("published_date")
            if published_date:
                data["date"] = published_date

            results.append({"data": data})

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Exa for a single query."""
        try:
            response = self.client.post(
                self.search_url,
                headers=self._build_headers(),
                json=self._build_payload(query, top_k),
            )
            response.raise_for_status()

            return self._parse_response(response.json(), top_k)

        except Exception as e:
            logger.warning("Exa search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Exa for a single query."""
        try:
            response = await self.aclient.post(
                self.search_url,
                headers=self._build_headers(),
                json=self._build_payload(query, top_k),
            )
            response.raise_for_status()

            return self._parse_response(response.json(), top_k)

        except Exception as e:
            logger.warning("Exa async search failed for query '%s': %s", query, e)
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
