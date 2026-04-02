from typing import Any, List, Mapping, Optional, Union

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseFuzzy, BaseRetriever
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import FuzzyRetriever

_SCORERS = {
    "WRatio": lambda: fuzz.WRatio,
    "ratio": lambda: fuzz.ratio,
    "partial_ratio": lambda: fuzz.partial_ratio,
    "token_sort_ratio": lambda: fuzz.token_sort_ratio,
    "token_set_ratio": lambda: fuzz.token_set_ratio,
}


@register_retriever
class RapidFuzzFuzzyRetriever(BaseFuzzy, BaseRetriever, FuzzyRetriever):
    """RapidFuzz - Approximate String Matching Fuzzy Retriever."""

    provider = "rapidfuzz"

    def __init__(self, *, scorer: Optional[str] = None):
        """Args:
        scorer:
            Scoring function to use for fuzzy matching. One of ``"WRatio"``,
            ``"ratio"``, ``"partial_ratio"``, ``"token_sort_ratio"``, or
            ``"token_set_ratio"``. Defaults to ``"WRatio"``.
        """
        if process is None:
            raise ImportError(
                "The 'rapidfuzz' package is not installed. "
                "Please install it via pip: pip install rapidfuzz"
            )

        if scorer is None:
            scorer = "WRatio"

        if scorer not in _SCORERS:
            raise ValueError(
                f"Unknown scorer '{scorer}'. Choose one of: {list(_SCORERS.keys())}"
            )

        self.scorer = scorer
        self._initialize()

    def _initialize(self):
        self.documents: List[str] = []

    def add(self, documents: List[str]):
        """Add documents to the fuzzy index."""
        self.documents.extend(documents)

    def _search_single(
        self, query: str, top_k: int, threshold: float, *, return_score: bool
    ) -> List[Mapping[str, Any]]:
        scorer_fn = _SCORERS[self.scorer]()
        matches = process.extract(
            query,
            self.documents,
            scorer=scorer_fn,
            score_cutoff=threshold,
            limit=top_k,
        )
        results = []
        for document, score, _ in matches:
            result = dotdict({"data": document})
            if return_score:
                result.score = float(score)
            results.append(result)
        return results

    def _search(
        self, queries: List[str], top_k: int, threshold: float, *, return_score: bool
    ) -> List[List[Mapping[str, Any]]]:
        if not self.documents:
            return [[] for _ in queries]
        args_list = [(query, top_k, threshold) for query in queries]
        kwargs_list = [{"return_score": return_score} for _ in queries]
        results = list(
            F.map_gather(
                self._search_single, args_list=args_list, kwargs_list=kwargs_list
            )
        )
        return results

    async def acall(
        self,
        queries: Union[str, List[str]],
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        return_score: Optional[bool] = None,
    ):
        """Async version of __call__ for RapidFuzz retrieval.

        RapidFuzz is pure in-memory CPU work — no I/O to await.
        Delegates directly to __call__ to avoid unnecessary executor overhead.

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
            dotdict containing search results.
        """
        return self(
            queries, top_k=top_k, threshold=threshold, return_score=return_score
        )
