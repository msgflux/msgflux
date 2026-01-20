import asyncio
from os import getenv
from typing import List, Optional

try:
    from linkup import LinkupClient
except ImportError:
    LinkupClient = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class LinkupWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A Linkup client for retrieving web content with AI-powered search.

    This class interfaces with the Linkup API to provide search results
    with support for standard and deep agentic search.
    """

    provider = "linkup"

    def __init__(
        self,
        *,
        depth: Optional[str] = None,
        output_type: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_images: Optional[bool] = None,
    ):
        """Initialize LinkupWebRetriever.

        Requires the `LINKUP_API_KEY` environment variable to be set.

        Args:
            depth:
                Search depth. Can be "standard" (fast) or "deep" (agentic).
                Defaults to "standard".
            output_type:
                Output type. Can be "searchResults" or "sourcedAnswer".
                Defaults to "searchResults".
            include_domains:
                List of domains to restrict search to. Defaults to None.
            exclude_domains:
                List of domains to exclude from search. Defaults to None.
            include_images:
                Whether to include images in results. Defaults to False.

        !!! example

            ```python
            retriever = LinkupWebRetriever(depth="deep", output_type="sourcedAnswer")
            results = retriever(["latest AI news"], top_k=5)
            print(results)
            ```
        """
        if LinkupClient is None:
            raise ImportError(
                "The 'linkup-sdk' package is not installed. "
                "Please install it via pip: pip install linkup-sdk"
            )

        api_key = getenv("LINKUP_API_KEY")
        if not api_key:
            raise ValueError(
                "The Linkup API key is not available. Please set `LINKUP_API_KEY`"
            )

        self.client = LinkupClient(api_key=api_key)
        self.depth = depth or "standard"
        self.output_type = output_type or "searchResults"
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.include_images = include_images or False

    def _build_search_kwargs(self, top_k: int) -> dict:
        """Build kwargs for Linkup search methods."""
        kwargs = {
            "depth": self.depth,
            "output_type": self.output_type,
            "max_results": top_k,
            "include_images": self.include_images,
        }

        if self.include_domains:
            kwargs["include_domains"] = self.include_domains
        if self.exclude_domains:
            kwargs["exclude_domains"] = self.exclude_domains

        return kwargs

    def _parse_results(self, response) -> List[dict]:
        """Parse Linkup response into standard format."""
        results = []

        # Handle sourcedAnswer output type
        if self.output_type == "sourcedAnswer":
            if hasattr(response, "sources") and response.sources:
                for source in response.sources:
                    data = {
                        "title": getattr(source, "name", None),
                        "content": getattr(source, "snippet", None),
                        "url": getattr(source, "url", None),
                    }
                    results.append({"data": data})
            return results

        # Handle searchResults output type
        if hasattr(response, "results"):
            for result in response.results:
                data = {
                    "title": getattr(result, "name", None),
                    "content": getattr(result, "content", None),
                    "url": getattr(result, "url", None),
                }
                results.append({"data": data})

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Linkup for a single query."""
        try:
            kwargs = self._build_search_kwargs(top_k)
            response = self.client.search(query, **kwargs)
            return self._parse_results(response)
        except Exception as e:
            logger.warning("Linkup search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Linkup for a single query."""
        try:
            kwargs = self._build_search_kwargs(top_k)
            response = await self.client.async_search(query, **kwargs)
            return self._parse_results(response)
        except Exception as e:
            logger.warning("Linkup search failed for query '%s': %s", query, e)
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
