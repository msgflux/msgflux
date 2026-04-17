from typing import Any, Dict, List, Mapping, Optional, Union

try:
    import bm25s
    import numpy as np
except ImportError:
    bm25s = None
    np = None

import msgflux.nn.functional as F
from msgflux.data.retrievers.base import BaseLexical, BaseRetriever
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import LexicalRetriever
from msgflux.core.dotdict import dotdict


@register_retriever
class BM25SLexicalRetriever(BaseLexical, BaseRetriever, LexicalRetriever):
    """BM25S - Fast BM25 implementation using Scipy sparse matrices.

    This retriever uses the bm25s library which provides a high-performance
    implementation of BM25 with support for multiple variants (Lucene, Robertson,
    Atire, BM25L, BM25+).
    """

    provider = "bm25s"

    def __init__(
        self,
        *,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        method: Optional[str] = None,
        stopwords: Optional[Union[str, List[str]]] = None,
    ):
        """Args:
        k1:
            Tuning parameter for term frequency. Defaults to 1.5.
        b:
            Tuning parameter for document length. Defaults to 0.75.
        method:
            BM25 variant to use. Options: "lucene" (default), "robertson",
            "atire", "bm25l", "bm25+". Defaults to "lucene".
        stopwords:
            Language code (e.g., "en") or list of stopwords to filter.
            Defaults to None (no filtering).
        """
        if bm25s is None:
            raise ImportError(
                "The 'bm25s' package is not installed. "
                "Please install it via pip: pip install bm25s"
            )

        # Apply defaults
        if k1 is None:
            k1 = 1.5
        if b is None:
            b = 0.75
        if method is None:
            method = "lucene"

        self.k1 = k1
        self.b = b
        self.method = method
        self.stopwords = stopwords
        self._initialize()

    def _initialize(self):
        self.documents: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[object] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words using bm25s tokenizer."""
        # Use bm25s.tokenize for single text (returns list of tokens)
        tokens = bm25s.tokenize([text], stopwords=self.stopwords, return_ids=False)
        # bm25s.tokenize returns list of token lists, get first element
        return tokens[0] if tokens else []

    def add(self, documents: List[str]):
        """Add documents to the BM25S index."""
        self.documents.extend(documents)

        # Tokenize all documents at once for efficiency
        new_tokens = bm25s.tokenize(
            documents, stopwords=self.stopwords, return_ids=False
        )
        self.tokenized_corpus.extend(new_tokens)

        # Rebuild index with all documents
        self.bm25 = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self.bm25.index(self.tokenized_corpus)

    def _search_single(
        self, query: str, top_k: int, threshold: float, *, return_score: bool
    ) -> List[Mapping[str, Any]]:
        """Search for a single query."""
        if not self.bm25:
            return []

        # Tokenize query
        query_tokens = bm25s.tokenize(
            [query], stopwords=self.stopwords, return_ids=False
        )

        # Retrieve results (returns tuple of doc_ids and scores)
        doc_ids, scores = self.bm25.retrieve(query_tokens, k=len(self.documents))

        # doc_ids and scores are 2D arrays (n_queries, k)
        # Get first query results
        doc_ids_array = doc_ids[0] if len(doc_ids) > 0 else []
        scores_array = scores[0] if len(scores) > 0 else []

        # Filter by threshold and build results
        results = []
        for doc_id, score in zip(doc_ids_array, scores_array):
            if score >= threshold:
                result = dotdict({"data": self.documents[doc_id]})
                if return_score:
                    result.score = float(score)
                results.append(result)

        # Return top_k results (already sorted by score in descending order)
        return results[:top_k]

    def _search(
        self, queries: List[str], top_k: int, threshold: float, *, return_score: bool
    ) -> List[List[Mapping[str, Any]]]:
        """Finds the top_k most similar documents for multiple queries.

        Args:
            queries:
                Query string or list of strings.
            top_k:
                Number of results to return.
            threshold:
                Minimum score to include a document in the results.
            return_score:
                If True, returns the score along with the document.

        Returns:
            List of results for each query.
        """
        if not self.bm25:
            return [[] for _ in queries]

        args_list = [(query, top_k, threshold) for query in queries]
        kwargs_list = [{"return_score": return_score} for _ in queries]
        query_results = F.map_gather(
            self._search_single, args_list=args_list, kwargs_list=kwargs_list
        )
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results

    def get_score_statistics(self, query: str) -> Dict[str, float]:
        """Calculate score statistics for a query across all documents.

        Args:
            query: Query string.

        Returns:
            Dictionary with min, max, mean, median, and std scores.
        """
        if not self.bm25:
            return None

        # Tokenize query
        query_tokens = bm25s.tokenize(
            [query], stopwords=self.stopwords, return_ids=False
        )

        # Get scores for all documents
        _, scores = self.bm25.retrieve(query_tokens, k=len(self.documents))
        scores_array = np.array(scores[0], dtype=np.float64)

        mean_score = np.mean(scores_array)
        median_score = np.median(scores_array)
        std_score = np.std(scores_array)

        return {
            "min_score": float(np.min(scores_array)),
            "max_score": float(np.max(scores_array)),
            "mean_score": float(mean_score),
            "median_score": float(median_score),
            "std_score": float(std_score),
        }

    async def acall(
        self,
        queries: Union[str, List[str]],
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        return_score: Optional[bool] = None,
    ):
        """Async version of __call__ for BM25S retrieval.

        BM25S is pure in-memory CPU work — no I/O to await.
        Delegates directly to __call__ to avoid unnecessary executor overhead.

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
            dotdict containing search results.
        """
        return self(
            queries, top_k=top_k, threshold=threshold, return_score=return_score
        )
