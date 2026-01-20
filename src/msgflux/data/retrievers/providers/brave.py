import asyncio
from typing import List, Optional

try:
    from brave_search_python_client import (
        BraveSearch,
        ImagesSearchRequest,
        NewsSearchRequest,
        WebSearchRequest,
    )
except ImportError:
    BraveSearch = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class BraveWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A Brave Search client for retrieving web, image, and news content.

    This class interfaces with the Brave Search API to provide grounded search results.
    """

    provider = "brave"

    def __init__(
        self,
        *,
        mode: str = "search",
        return_images: Optional[bool] = None,
    ):
        """Initialize BraveWebRetriever.

        Requires the `BRAVE_SEARCH_PYTHON_CLIENT_API_KEY` environment variable
        to be set.

        Args:
            mode:
                Search mode. Can be "search", "image", or "news". Defaults to "search".
            return_images:
                Whether to include images in the results. Defaults to False.

        !!! example

            ```python
            retriever = BraveWebRetriever(mode="search")
            results = retriever(["latest AI news"], top_k=3)
            print(results)
            ```
        """
        if BraveSearch is None:
            raise ImportError(
                "The 'brave-search-python-client' package is not installed. "
                "Please install it via pip: pip install brave-search-python-client"
            )

        self.client = BraveSearch()
        self.mode = mode
        self.return_images = return_images or False

    async def _asingle_search(self, query: str, top_k: int) -> List[dotdict]:
        """Async internal method to search Brave for a single query."""
        try:
            if self.mode == "search":
                return await self._search_web(query, top_k)
            elif self.mode == "news":
                return await self._search_news(query, top_k)
            elif self.mode == "image":
                return await self._search_images(query, top_k)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        except Exception as e:
            logger.warning("Brave search failed for query '%s': %s", query, e)
            return []

    async def _search_web(self, query: str, top_k: int) -> List[dotdict]:
        req = WebSearchRequest(q=query, count=top_k)
        response = await self.client.web(req)
        web_results = response.web.results if response.web else []
        results = []
        for result in web_results:
            data = {
                "title": result.title,
                "content": result.description,
                "url": result.url,
            }
            item = {"data": data}
            if self.return_images and result.thumbnail:
                src = getattr(result.thumbnail, "src", None)
                if src:
                    item["images"] = [src]
            results.append(item)
        return results

    async def _search_news(self, query: str, top_k: int) -> List[dotdict]:
        req = NewsSearchRequest(q=query, count=top_k)
        response = await self.client.news(req)
        news_results = response.results if response.results else []
        results = []
        for result in news_results:
            data = {
                "title": result.title,
                "content": result.description,
                "url": result.url,
                "date": getattr(result, "age", None),
            }
            item = {"data": data}
            if self.return_images and getattr(result, "thumbnail", None):
                src = getattr(result.thumbnail, "src", None)
                if src:
                    item["images"] = [src]
            results.append(item)
        return results

    async def _search_images(self, query: str, top_k: int) -> List[dotdict]:
        req = ImagesSearchRequest(q=query, count=top_k)
        response = await self.client.images(req)
        image_results = response.results if response.results else []
        results = []
        for result in image_results:
            data = {"title": result.title, "url": result.url}
            # For image search, the item itself is the image
            item = {"data": data, "images": [result.url]}
            results.append(item)
        return results

    async def _asearch(self, queries: List[str], top_k: int) -> List[dotdict]:
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Synchronous search wrapper."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Standard asyncio run call
        return asyncio.run(self._asearch(queries, top_k))
