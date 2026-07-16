from typing import Any, Dict, List, Mapping, Optional, Union

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    np = None

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseLexical, BaseRetriever
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import LexicalRetriever


@register_retriever
class RankBM25LexicalRetriever(BaseLexical, BaseRetriever, LexicalRetriever):
    """Rank Okapi BM25 - Best Matching 25 Lexical Retriever."""

    provider = "rank_bm25"

    def __init__(self, *, k1: Optional[float] = None, b: Optional[float] = None):
        """Args:
        k1:
            Tuning parameter for term frequency. Defaults to 1.5.
        b:
            Tuning parameter for document length. Defaults to 0.75.
        """
        if BM25Okapi is None:
            raise ImportError(
                "The 'rank_bm25' package is not installed. "
                "Please install it via pip: pip install rank-bm25"
            )

        # Apply defaults
        if k1 is None:
            k1 = 1.5
        if b is None:
            b = 0.75

        self.k1 = k1
        self.b = b
        self._initialize()

    def _initialize(self):
        self.documents: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None  # type: ignore[name-defined]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()

    def add(self, documents: List[str]):
        """Add documents to the BM25 index."""
        self.documents.extend(documents)
        self.tokenized_corpus.extend(self._tokenize(doc) for doc in documents)
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def _search_single(
        self, query: str, top_k: int, threshold: float, *, return_score: bool
    ) -> List[Mapping[str, Any]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Filter by threshold
        filtered_doc_scores = [
            (doc_id, score) for doc_id, score in enumerate(scores) if score >= threshold
        ]

        # Order by score
        filtered_doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top_k
        results = []
        for doc_id, score in filtered_doc_scores[:top_k]:
            result = dotdict({"data": self.documents[doc_id]})
            if return_score:
                result.score = float(score)
            results.append(result)
        return results

    def _search(
        self, queries: List[str], top_k: int, threshold: float, *, return_score: bool
    ) -> List[List[Mapping[str, Any]]]:
        if not self.bm25:
            return [[] for _ in queries]
        args_list = [(query, top_k, threshold) for query in queries]
        kwargs_list = [{"return_score": return_score} for _ in queries]
        results = list(
            F.map_gather(
                self._search_single, args_list=args_list, kwargs_list=kwargs_list
            )
        )
        return results

    def get_score_statistics(self, query: str) -> Dict[str, float]:
        if not self.bm25:
            return None

        tokenized_query = self._tokenize(query)
        scores = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float64)

        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)

        return {
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
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
        """Async version of __call__ for Rank BM25 retrieval.

        BM25 is pure in-memory CPU work — no I/O to await.
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
