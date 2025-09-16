from typing import Any, Dict, List, Mapping, Optional, Union

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    np = None

from msgflux.data.retrievers.base import BaseLexical, BaseRetriever
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import LexicalRetriever
from msgflux.dotdict import dotdict
from msgflux.nn import functional as F


#@register_retriever
class RankBM25LexicalRetriever(BaseLexical, BaseRetriever, LexicalRetriever):
    """Rank Okapi BM25 - Best Matching 25 Lexical Retriever."""

    provider = "rank_bm25"

    def __init__(self, *, k1: Optional[float] = 1.5, b: Optional[float] = 0.75):
        """Args:
        k1:
            Tuning parameter for term frequency.
        b:
            Tuning parameter for document length.
        """
        if BM25Okapi is None:
            raise ImportError(
                "`rank_bm25` client is not available. "
                "Install with `pip install rank_bm25`."
            )
        self.k1 = k1
        self.b = b
        self._initialize()

    def _initialize(self):
        self.documents: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()

    def add(self, documents: List[str]):
        """Add documents to the BM25 index."""
        self.documents.extend(documents)
        self.tokenized_corpus.extend(self._tokenize(doc) for doc in documents)
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def _search_single(self, query: str) -> List[Mapping[str, Any]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Filtra por threshold
        filtered_doc_scores = [
            (doc_id, score)
            for doc_id, score in enumerate(scores)
            if score >= threshold
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
        self, queries: List[str], top_k: int, threshold: float, return_score: bool
    ) -> List[List[Mapping[str, Any]]]:
        if not self.bm25:
            return [[] for _ in queries]
        results = list(F.map_gather(self._search_single, args_list=queries))
        return results

    def get_score_statistics(self, query: str) -> Dict[str, float]:
        if not self.bm25:
            return None

        tokenized_query = self._tokenize(query)
        scores = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float64)

        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)  # já calcula a raiz

        return {
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_score": float(mean_score),
            "median_score": float(median_score),
            "std_score": float(std_score),
        }
