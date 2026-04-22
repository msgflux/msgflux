import asyncio
from os import getenv
from typing import Any, List, Mapping, Optional

try:
    import serpapi
except ImportError:
    serpapi = None

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.logger import init_logger

logger = init_logger(__name__)

API_KEY_ENV_NAMES = ("SERPAPI_KEY", "SERPAPI_API_KEY", "SERP_API_KEY")
QUERY_PARAM_BY_ENGINE = {
    "apple_app_store": "term",
    "ebay": "_nkw",
    "naver": "query",
    "walmart": "query",
    "yahoo": "p",
    "youtube": "search_query",
}


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

        Requires the `SERPAPI_KEY` environment variable to be set.

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
        if serpapi is None:
            raise ImportError(
                "The 'serpapi' package is not installed. "
                "Please install it via pip: pip install serpapi"
            )

        self.engine = engine or "google"
        self.location = location
        self.gl = gl
        self.hl = hl
        self.safe = safe
        self.tbm = tbm
        self.client = serpapi.Client(api_key=self._get_api_key())

    def _get_api_key(self) -> str:
        for env_name in API_KEY_ENV_NAMES:
            api_key = getenv(env_name)
            if api_key:
                return api_key

        raise ValueError(
            "The SerpApi API key is not available. Please set `SERPAPI_KEY`"
        )

    def _get_query_param(self) -> str:
        return QUERY_PARAM_BY_ENGINE.get(self.engine, "q")

    def _build_search_params(self, query: str, top_k: int) -> dict:
        """Build params for SerpAPI search."""
        params = {
            "engine": self.engine,
            self._get_query_param(): query,
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

    def _format_result(self, result: Mapping[str, Any]) -> dict:
        data = {
            "title": result.get("title"),
            "content": result.get("snippet") or result.get("source"),
            "url": result.get("link") or result.get("url") or result.get("original"),
        }

        for field in ("date", "price", "source"):
            if result.get(field):
                data[field] = result[field]

        item = {"data": data}
        image_url = result.get("thumbnail") or result.get("original")
        if image_url:
            item["images"] = [image_url]

        return item

    def _parse_results(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        """Parse SerpAPI response into standard format."""
        results = []

        for collection in (
            "organic_results",
            "news_results",
            "images_results",
            "shopping_results",
        ):
            for result in response.get(collection, []):
                results.append(self._format_result(result))
                if len(results) >= top_k:
                    return results

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search SerpAPI for a single query."""
        try:
            params = self._build_search_params(query, top_k)
            response = self.client.search(params)
            return self._parse_results(response, top_k)
        except Exception as e:
            logger.warning("SerpAPI search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search SerpAPI for a single query."""
        loop = asyncio.get_running_loop()
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
