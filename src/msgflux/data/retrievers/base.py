from typing import List, Mapping, Optional, Union

from msgflux._private.client import BaseClient
from msgflux.data.retrievers.response import RetrieverResponse


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
        top_k: Optional[int] = 5,
        threshold: Optional[float] = 0.0,
        return_score: Optional[bool] = False,
    ) -> RetrieverResponse:
        """Retrieve the most relevant documents for one or multiple queries using BM25 ranking.

        Args:
            queries:
                A single query string or a list of query strings to search for.
            top_k:
                The maximum number of documents to return for each query.
            threshold:
                Minimum BM25 score a document must have to be included in the results.
            return_score:
                If True, includes the BM25 score in the returned results.

        Returns:
            RetrieverResponse:
                A response object containing the search results for each query. Each result
                includes the document text, and optionally the BM25 score if `return_score`
                is True.

        Raises:
            ValueError:
                If `queries` is empty or contains invalid types.
        """
        if isinstance(queries, str):
            queries = [queries]
        results = self._search(queries, top_k, threshold, return_score)
        response = RetrieverResponse()
        response.set_response_type("lexical_search")
        response.add(results)
        return response


class BaseWebSearch:
    def __call__(
        self,
        queries: Union[str, List[str]],
        top_k: Optional[int] = 1
    ) -> RetrieverResponse:
        """Search web and retrieve results for given queries.

        Args:
            queries:
                Single query string or list of queries.
            top_k:
                Number of results to return per query.

        Returns:
            List of retriever results containing `data` and
            `images` (if return_images=True).
        """
        if isinstance(queries, str):
            queries = [queries]
        results = self._search(queries, top_k)
        response = RetrieverResponse()
        response.set_response_type("web_search")
        response.add(results)
        return response
