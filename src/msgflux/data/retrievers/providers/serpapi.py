import asyncio
from os import getenv
from typing import List, Optional

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class SerpApiWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A SerpAPI client for retrieving web search results.

    This class interfaces with the SerpAPI to provide Google search results
    and other search engines.
    """

    provider = "serpapi"

    def __init__(
        self,
        *,
        engine: Optional[str] = None,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        safe: Optional[str] = None,
        tbm: Optional[str] = None,
    ):
        """Initialize SerpApiWebRetriever.

        Requires the `SERPAPI_API_KEY` environment variable to be set.

        Args:
            engine:
                Search engine to use. Can be "google", "bing", "yahoo", etc.
                Defaults to "google".
            location:
                Location for localized results (e.g., "Austin,Texas").
                Defaults to None.
            gl:
                Country code for Google (e.g., "us", "br").
                Defaults to None.
            hl:
                Language code for Google UI (e.g., "en", "pt").
                Defaults to None.
            safe:
                Safe search mode. Can be "active" or "off".
                Defaults to None.
            tbm:
                Type of search. Can be "nws" (news), "isch" (images),
                "shop" (shopping). Defaults to None (web search).

        !!! example

            ```python
            retriever = SerpApiWebRetriever(location="Austin,Texas", gl="us")
            results = retriever(["latest AI news"], top_k=5)
            print(results)
            ```
        """
        if GoogleSearch is None:
            raise ImportError(
                "The 'google-search-results' package is not installed. "
                "Please install it via pip: pip install google-search-results"
            )

        api_key = getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "The SerpAPI API key is not available. Please set `SERPAPI_API_KEY`"
            )

        self.engine = engine or "google"
        self.location = location
        self.gl = gl
        self.hl = hl
        self.safe = safe
        self.tbm = tbm

    def _build_search_params(self, query: str, top_k: int) -> dict:
        """Build params for SerpAPI search."""
        api_key = getenv("SERPAPI_API_KEY")
        params = {
            "q": query,
            "api_key": api_key,
            "num": top_k,
        }

        if self.location:
            params["location"] = self.location
        if self.gl:
            params["gl"] = self.gl
        if self.hl:
            params["hl"] = self.hl
        if self.safe:
            params["safe"] = self.safe
        if self.tbm:
            params["tbm"] = self.tbm

        return params

    def _parse_results(self, response: dict) -> List[dict]:
        """Parse SerpAPI response into standard format."""
        results = []

        # Handle organic results (web search)
        organic_results = response.get("organic_results", [])
        for result in organic_results:
            data = {
                "title": result.get("title"),
                "content": result.get("snippet"),
                "url": result.get("link"),
            }

            # Add date if available
            if result.get("date"):
                data["date"] = result["date"]

            results.append({"data": data})

        # Handle news results if tbm=nws
        news_results = response.get("news_results", [])
        for result in news_results:
            data = {
                "title": result.get("title"),
                "content": result.get("snippet"),
                "url": result.get("link"),
            }

            if result.get("date"):
                data["date"] = result["date"]

            results.append({"data": data})

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search SerpAPI for a single query."""
        try:
            params = self._build_search_params(query, top_k)
            search = GoogleSearch(params)
            response = search.get_dict()
            return self._parse_results(response)
        except Exception as e:
            logger.warning("SerpAPI search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search SerpAPI for a single query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._single_search(query, top_k)
        )

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
