import asyncio
from os import getenv
from typing import List, Optional, Union

try:
    from exa_py import Exa
except ImportError:
    Exa = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class ExaWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
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
    ):
        """Initialize ExaWebRetriever.

        Requires the `EXA_API_KEY` environment variable to be set.

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

        !!! example

            ```python
            retriever = ExaWebRetriever(include_text=True, max_characters=1000)
            results = retriever(["latest AI news"], top_k=3)
            print(results)
            ```
        """
        if Exa is None:
            raise ImportError(
                "The 'exa-py' package is not installed. "
                "Please install it via pip: pip install exa-py"
            )

        api_key = getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError(
                "The Exa API key is not available. Please set `EXA_API_KEY`"
            )

        self.client = Exa(api_key=api_key)
        self.search_type = search_type or "auto"
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.start_published_date = start_published_date
        self.end_published_date = end_published_date
        self.include_text = include_text if include_text is not None else True
        self.max_characters = max_characters

    def _build_search_kwargs(self, top_k: int) -> dict:
        """Build kwargs for Exa search methods."""
        kwargs = {
            "num_results": top_k,
            "type": self.search_type,
        }

        if self.include_domains:
            kwargs["include_domains"] = self.include_domains
        if self.exclude_domains:
            kwargs["exclude_domains"] = self.exclude_domains
        if self.start_published_date:
            kwargs["start_published_date"] = self.start_published_date
        if self.end_published_date:
            kwargs["end_published_date"] = self.end_published_date

        return kwargs

    def _build_text_options(self) -> dict:
        """Build text options for content retrieval."""
        if not self.include_text:
            return {}

        text_opts = {}
        if self.max_characters:
            text_opts["max_characters"] = self.max_characters

        return {"text": text_opts if text_opts else True}

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Exa for a single query."""
        try:
            kwargs = self._build_search_kwargs(top_k)

            if self.include_text:
                text_opts = self._build_text_options()
                kwargs.update(text_opts)
                response = self.client.search_and_contents(query, **kwargs)
            else:
                response = self.client.search(query, **kwargs)

            results = []
            for result in response.results:
                data = {
                    "title": result.title,
                    "url": result.url,
                }

                # Add text content if available
                if hasattr(result, "text") and result.text:
                    data["content"] = result.text

                # Add published date if available
                if hasattr(result, "published_date") and result.published_date:
                    data["date"] = result.published_date

                results.append({"data": data})

            return results

        except Exception as e:
            logger.warning("Exa search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Exa for a single query."""
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

    async def acall(
        self, queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> dotdict:
        """Async version of __call__ for searching Exa.

        Args:
            queries:
                Single query string or list of queries.
            top_k:
                Number of results to return per query. Defaults to 1.

        Returns:
            dotdict containing search results.

        !!! example

            ```python
            retriever = ExaWebRetriever(include_text=True)
            results = await retriever.acall(
                ["Python programming", "Machine learning"], top_k=2
            )
            print(results)
            ```
        """
        if isinstance(queries, str):
            queries = [queries]
        if top_k is None:
            top_k = 1

        results = await self._asearch(queries, top_k)
        return dotdict({"response_type": "web_search", "data": results})
