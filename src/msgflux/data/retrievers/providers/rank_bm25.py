import math
from collections import Counter
from typing import Dict, List, Optional, Union

from gevent.pool import Pool

try:
    from rank_bm25 import BM25Okapi
except:
    raise ImportError("pip install rank_bm25")  # TODO
from msgflux.data.retrievers.base import BaseRetriever
from msgflux.data.retrievers.types import LexicalRetriever


class RankBM25LexicalRetriever(BaseRetriever, LexicalRetriever):
    """Rank Okapi BM25 - Best Matching 25.

    Args:
        k1: Tuning parameter for term frequency (default 1.5)
        b: Tuning parameter for document length (default 0.75)
        max_workers: Maximum number of workers for parallel queries
            (default: None, uses the number of CPUs)
    """

    def __init__(
        self,
        k1: Optional[float] = 1.5,
        b: Optional[float] = 0.75,
        max_workers: Optional[int] = None,
    ):
        self.b = b
        self.k1 = k1
        self.max_workers = max_workers
        self._initialize()

    def _initialize(self):
        self.avg_doc_length = 0
        self.documents = []
        self.doc_lengths = []
        self.idf = {}
        self.term_document_matrix = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()

    def add(self, documents: List[str]):
        start_index = len(self.documents)
        self.documents.extend(documents)

        new_doc_lengths = [len(self._tokenize(doc)) for doc in documents]
        self.doc_lengths.extend(new_doc_lengths)

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        # Update inverted index and IDF
        for doc_id in range(start_index, len(self.documents)):
            tokens = self._tokenize(self.documents[doc_id])
            token_counts = Counter(tokens)

            for token, freq in token_counts.items():
                if token not in self.term_document_matrix:
                    self.term_document_matrix[token] = []
                self.term_document_matrix[token].append((doc_id, freq))

        # Calculate IDF for all terms
        total_docs = len(self.documents)
        self.idf = {
            term: math.log(
                1 + (total_docs - len(doc_list) + 0.5) / (len(doc_list) + 0.5)
            )
            for term, doc_list in self.term_document_matrix.items()
        }

    def _calculate_bm25_score(self, query, doc_id):
        """Calculates the BM25 score for a specific document.

        Args:
            query: list of query tokens
            doc_id: ID of the document to be scored

        Returns:
            BM25 score for the document
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_id]

        for term in set(query):
            if term not in self.term_document_matrix:
                continue

            # Find the frequency of the term in the document
            term_freq = next(
                (
                    freq
                    for did, freq in self.term_document_matrix[term]
                    if did == doc_id
                ),
                0,
            )

            # BM25 score calculation
            idf = self.idf.get(term, 0)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def _search(self, queries, top_k, threshold, return_score):
        """Finds the top_k most similar documents for multiple queries.

        Args:
            queries: query string or list of strings
            top_k: number of results to return
            threshold: minimum score to include a document in the results
            return_score: if True, returns the score along with the document

        Returns:
            List of results for each query
        """
        pool = Pool(size=self.max_workers)

        def process_query(query):
            query_tokens = self._tokenize(query)

            # Calculate scores for all documents
            doc_scores = [
                (doc_id, self._calculate_bm25_score(query_tokens, doc_id))
                for doc_id in range(len(self.documents))
            ]

            # Filter documents by threshold
            filtered_doc_scores = [
                (doc_id, score) for doc_id, score in doc_scores if score >= threshold
            ]

            # Sort documents by score in descending order
            filtered_doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Returns the K best results
            results = []
            for doc_id, score in filtered_doc_scores[:top_k]:
                result = {"data": self.documents[doc_id]}
                if return_score:
                    result["score"] = score
                results.append(result)

            return results

        results = pool.map(process_query, queries)
        return results

    def __call__(
        self,
        queries: Union[str, List[str]],
        top_k: Optional[int] = 5,
        threshold: Optional[float] = 0.0,
        return_score: Optional[bool] = False,
    ):
        if isinstance(queries, str):
            queries = [queries]
        return self._search(queries, top_k, threshold, return_score)

    def get_score_statistics(self, query: str) -> Dict[str, float]:
        if not self.documents:
            return None

        query_tokens = self._tokenize(query)

        doc_scores = [
            self._calculate_bm25_score(query_tokens, doc_id)
            for doc_id in range(len(self.documents))
        ]

        mean_score = sum(doc_scores) / len(doc_scores)

        sorted_scores = sorted(doc_scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            median_score = sorted_scores[n // 2]

        mean = mean_score
        variance = sum((x - mean) ** 2 for x in doc_scores) / len(doc_scores)
        std_score = math.sqrt(variance)

        return {
            "min_score": min(doc_scores),
            "max_score": max(doc_scores),
            "mean_score": mean_score,
            "median_score": median_score,
            "std_score": std_score,
        }
