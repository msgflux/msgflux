import asyncio
from typing import Any, List, Mapping, Optional, Union

try:
    import arxiv
except ImportError:
    arxiv = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import logger
from msgflux.nn import functional as F


@register_retriever
class ArxivWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A customizable arXiv client for searching and retrieving academic papers.

    This class provides a flexible interface to search arXiv articles and retrieve
    metadata including titles, summaries, authors, and PDF URLs.
    """

    provider = "arxiv"

    def __init__(
        self,
        *,
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
        """Args:
            max_results:
                Maximum number of results to fetch per query. Defaults to 10.
            sort_by:
                Sort criterion for results. One of "relevance", "lastUpdatedDate",
                or "submittedDate". Defaults to "relevance".
            sort_order:
                Sort order for results. One of "ascending" or "descending".
                Defaults to "descending".

        !!! example

            ```python
            retriever = ArxivWebRetriever(
                max_results=5, sort_by="submittedDate", sort_order="descending"
            )
            results = retriever(
                ["machine learning", "neural networks"], top_k=2
            )
            print(results)
            ```
        """
        if arxiv is None:
            raise ImportError(
                "The 'arxiv' package is not installed. "
                "Please install it via pip: pip install arxiv"
            )

        # Apply defaults
        if max_results is None:
            max_results = 10
        if sort_by is None:
            sort_by = "relevance"
        if sort_order is None:
            sort_order = "descending"

        self.max_results = max_results
        self.sort_by = self._get_sort_criterion(sort_by)
        self.sort_order = self._get_sort_order(sort_order)
        self.client = arxiv.Client()

    def _get_sort_criterion(self, sort_by: str) -> "arxiv.SortCriterion":
        """Convert string sort criterion to arxiv.SortCriterion enum."""
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        if sort_by not in sort_map:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. "
                f"Must be one of: {list(sort_map.keys())}"
            )
        return sort_map[sort_by]

    def _get_sort_order(self, sort_order: str) -> "arxiv.SortOrder":
        """Convert string sort order to arxiv.SortOrder enum."""
        order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }
        if sort_order not in order_map:
            raise ValueError(
                f"Invalid sort_order value: {sort_order}. "
                f"Must be one of: {list(order_map.keys())}"
            )
        return order_map[sort_order]

    def _format_result(self, result: "arxiv.Result") -> Mapping[str, Any]:
        """Format an arxiv.Result into the standard response format."""
        return {
            "data": {
                "title": result.title,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat() if result.published else None,
                "updated": result.updated.isoformat() if result.updated else None,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "categories": result.categories,
            }
        }

    def _single_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Internal method to search arXiv for a single query."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=top_k,
                sort_by=self.sort_by,
                sort_order=self.sort_order,
            )

            results = []
            for result in self.client.results(search):
                try:
                    formatted = self._format_result(result)
                    results.append(formatted)
                except Exception as e:
                    logger.debug(f"Error formatting result: {e}")
                    continue

            return results

        except Exception as e:
            logger.debug(f"Error searching arXiv: {e}")
            return []

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Search arXiv for multiple queries in parallel."""
        args_list = [(query,) for query in queries]
        kwargs_list = [{"top_k": top_k} for _ in queries]
        query_results = F.map_gather(
            self._single_search, args_list=args_list, kwargs_list=kwargs_list
        )
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results

    async def _asingle_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Async internal method to search arXiv for a single query.

        Uses asyncio.run_in_executor to run synchronous arxiv calls
        in a thread pool.
        """
        loop = asyncio.get_event_loop()

        try:
            search = arxiv.Search(
                query=query,
                max_results=top_k,
                sort_by=self.sort_by,
                sort_order=self.sort_order,
            )

            # Run the synchronous client.results in executor
            results_list = await loop.run_in_executor(
                None, lambda: list(self.client.results(search))
            )

            results = []
            for result in results_list:
                try:
                    formatted = self._format_result(result)
                    results.append(formatted)
                except Exception as e:
                    logger.debug(f"Error formatting result: {e}")
                    continue

            return results

        except Exception as e:
            logger.debug(f"Error searching arXiv: {e}")
            return []

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
        """Async version of __call__ for searching arXiv.

        Args:
            queries:
                Single query string or list of queries.
            top_k:
                Number of results to return per query. Defaults to 1.

        Returns:
            dotdict containing search results.

        !!! example

            ```python
            retriever = ArxivWebRetriever(sort_by="submittedDate")
            results = await retriever.acall(
                ["machine learning", "deep learning"], top_k=3
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
