from typing import List, Mapping, Optional, Union

from msgflux._private.client import BaseClient
from msgflux.core.dotdict import dotdict


class BaseRetriever(BaseClient):
    msgflux_type = "retriever"
    to_ignore = ["client", "documents"]

    def instance_type(self) -> Mapping[str, str]:
        return {"retriever_type": self.retriever_type}


class BaseLexical:
    def __call__(
        self,
        queries: Union[str, List[str]],
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        return_score: Optional[bool] = None,
    ) -> dotdict:
        """Retrieve the most relevant documents for one or multiple queries
        using BM25 ranking.

        Args:
            queries:
                A single query string or a list of query strings to search for.
            top_k:
                The maximum number of documents to return for each query.
                Defaults to 5.
            threshold:
                Minimum BM25 score a document must have to be included in the
                results. Defaults to 0.0.
            return_score:
                If True, includes the BM25 score in the returned results.
                Defaults to False.

        Returns:
            dotdict:
                A response object containing the search results for each query.
                Each result
                includes the document text, and optionally the BM25 score if
                `return_score`
                is True.

        Raises:
            ValueError:
                If `queries` is empty or contains invalid types.
        """
        if isinstance(queries, str):
            queries = [queries]
        if top_k is None:
            top_k = 5
        if threshold is None:
            threshold = 0.0
        if return_score is None:
            return_score = False
        results = self._search(queries, top_k, threshold, return_score=return_score)
        return dotdict({"response_type": "lexical_search", "data": results})


class BaseFuzzy:
    def __call__(
        self,
        queries: Union[str, List[str]],
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        return_score: Optional[bool] = None,
    ) -> dotdict:
        """Retrieve the most similar documents for one or multiple queries
        using fuzzy string matching.

        Args:
            queries:
                A single query string or a list of query strings to search for.
            top_k:
                The maximum number of documents to return for each query.
                Defaults to 5.
            threshold:
                Minimum similarity score (0-100) a document must have to be
                included in the results. Defaults to 0.0.
            return_score:
                If True, includes the similarity score in the returned results.
                Defaults to False.

        Returns:
            dotdict:
                A response object containing the search results for each query.
                Each result includes the document text, and optionally the
                similarity score if `return_score` is True.

        Raises:
            ValueError:
                If `queries` is empty or contains invalid types.
        """
        if isinstance(queries, str):
            queries = [queries]
        if top_k is None:
            top_k = 5
        if threshold is None:
            threshold = 0.0
        if return_score is None:
            return_score = False
        results = self._search(queries, top_k, threshold, return_score=return_score)
        return dotdict({"response_type": "fuzzy_search", "data": results})


class BaseWebSearch:
    def __call__(
        self, queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> dotdict:
        """Search web and retrieve results for given queries.

        Args:
            queries:
                Single query string or list of queries.
            top_k:
                Number of results to return per query. Defaults to 1.

        Returns:
            List of retriever results containing `data` and
            `images` (if return_images=True).
        """
        if isinstance(queries, str):
            queries = [queries]
        if top_k is None:
            top_k = 1
        results = self._search(queries, top_k)
        return dotdict({"response_type": "web_search", "data": results})

    async def acall(
        self, queries: Union[str, List[str]], top_k: Optional[int] = None
    ) -> dotdict:
        """Async version for searching the web.

        Args:
            queries:
                Single query string or list of queries.
            top_k:
                Number of results to return per query. Defaults to 1.

        Returns:
            dotdict containing search results.
        """
        if isinstance(queries, str):
            queries = [queries]
        if top_k is None:
            top_k = 1
        results = await self._asearch(queries, top_k)
        return dotdict({"response_type": "web_search", "data": results})


class BaseWebFetch:
    def __call__(self, urls: Union[str, List[str]]) -> dotdict:
        """Fetch web pages and extract content.

        Args:
            urls:
                Single URL string or list of URLs to fetch.

        Returns:
            dotdict containing fetched page content as Markdown.
        """
        if isinstance(urls, str):
            urls = [urls]
        results = self._fetch(urls)
        return dotdict({"response_type": "web_fetch", "data": results})

    async def acall(self, urls: Union[str, List[str]]) -> dotdict:
        """Async version for fetching web pages.

        Args:
            urls:
                Single URL string or list of URLs to fetch.

        Returns:
            dotdict containing fetched page content as Markdown.
        """
        if isinstance(urls, str):
            urls = [urls]
        results = await self._afetch(urls)
        return dotdict({"response_type": "web_fetch", "data": results})
