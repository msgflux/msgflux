"""Native HTTP client for the arXiv Atom search API."""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, List, Mapping, Optional, Union

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebSearchRetriever
from msgflux.logger import init_logger
from msgflux.nn import functional as F

logger = init_logger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PAGE_SIZE = 100
USER_AGENT = "msgflux/1.0 (https://github.com/msgflux/msgflux)"

ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
_NAMESPACES = {"atom": ATOM_NAMESPACE, "arxiv": ARXIV_NAMESPACE}
_ERROR_ID_PREFIX = "http://arxiv.org/api/errors#"

_SORT_BY_VALUES = ("relevance", "lastUpdatedDate", "submittedDate")
_SORT_ORDER_VALUES = ("ascending", "descending")


@register_retriever
class ArxivWebRetriever(
    HTTPRetrieverMixin, BaseWebSearch, BaseRetriever, WebSearchRetriever
):
    """A customizable arXiv client for searching and retrieving academic papers.

    This class provides a flexible interface to search arXiv articles and retrieve
    metadata including titles, summaries, authors, and PDF URLs.
    """

    provider = "arxiv"
    to_ignore = ["client", "aclient", "documents"]

    def __init__(
        self,
        *,
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        timeout: Optional[float] = None,
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
            timeout:
                Request timeout in seconds. Defaults to 30.

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
        self.timeout = timeout or 30.0
        self._initialize()

    def _get_sort_criterion(self, sort_by: str) -> str:
        """Validate and return a string sort criterion."""
        if sort_by not in _SORT_BY_VALUES:
            raise ValueError(
                f"Invalid sort_by value: {sort_by}. "
                f"Must be one of: {list(_SORT_BY_VALUES)}"
            )
        return sort_by

    def _get_sort_order(self, sort_order: str) -> str:
        """Validate and return a string sort order."""
        if sort_order not in _SORT_ORDER_VALUES:
            raise ValueError(
                f"Invalid sort_order value: {sort_order}. "
                f"Must be one of: {list(_SORT_ORDER_VALUES)}"
            )
        return sort_order

    def _build_headers(self) -> dict:
        """Build request headers for the arXiv API."""
        return {
            "Accept": "application/atom+xml",
            "User-Agent": USER_AGENT,
        }

    def _build_params(self, query: str, limit: int, start: int) -> dict:
        """Build query parameters for one page of arXiv results."""
        return {
            "search_query": query,
            "start": start,
            "max_results": limit,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order,
        }

    @staticmethod
    def _get_text(entry: ET.Element, path: str) -> str:
        """Read and normalize text from an Atom entry."""
        node = entry.find(path, _NAMESPACES)
        if node is None or not node.text:
            return ""
        return " ".join(node.text.split())

    @staticmethod
    def _format_datetime(value: Optional[str]) -> Optional[str]:
        """Format an Atom timestamp using the previous SDK's ISO representation."""
        if not value:
            return None

        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"

        try:
            return datetime.fromisoformat(normalized).isoformat()
        except ValueError:
            return normalized

    @staticmethod
    def _is_error_entry(entry: ET.Element) -> bool:
        """Return whether an Atom entry describes an arXiv API error."""
        entry_id = entry.findtext("atom:id", default="", namespaces=_NAMESPACES)
        title = entry.findtext("atom:title", default="", namespaces=_NAMESPACES)
        return entry_id.startswith(_ERROR_ID_PREFIX) or title.strip().lower() == "error"

    def _format_result(self, entry: ET.Element) -> Mapping[str, Any]:
        """Format an Atom entry into the standard response format."""
        pdf_url = None
        for link in entry.findall("atom:link", _NAMESPACES):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        authors = []
        for author in entry.findall("atom:author", _NAMESPACES):
            name = self._get_text(author, "atom:name")
            if name:
                authors.append(name)

        categories = []
        for category in entry.findall("atom:category", _NAMESPACES):
            term = category.get("term")
            if term:
                categories.append(term)

        return {
            "data": {
                "title": self._get_text(entry, "atom:title"),
                "summary": self._get_text(entry, "atom:summary"),
                "authors": authors,
                "published": self._format_datetime(
                    entry.findtext(
                        "atom:published", default=None, namespaces=_NAMESPACES
                    )
                ),
                "updated": self._format_datetime(
                    entry.findtext("atom:updated", default=None, namespaces=_NAMESPACES)
                ),
                "pdf_url": pdf_url,
                "entry_id": self._get_text(entry, "atom:id") or None,
                "categories": categories,
            }
        }

    def _parse_entries(self, payload: str) -> List[ET.Element]:
        """Parse an Atom payload into entry elements."""
        root = ET.fromstring(payload)  # noqa: S314
        return root.findall("atom:entry", _NAMESPACES)

    def _append_entries(
        self,
        entries: List[ET.Element],
        results: List[Mapping[str, Any]],
        seen: set,
        top_k: int,
    ) -> int:
        """Append unique formatted entries and return how many were added."""
        added = 0
        for entry in entries:
            if self._is_error_entry(entry):
                summary = self._get_text(entry, "atom:summary")
                raise ValueError(f"arXiv API error: {summary or 'unknown error'}")

            try:
                formatted = self._format_result(entry)
            except Exception as e:
                logger.debug(f"Error formatting result: {e}")
                continue

            data = formatted["data"]
            key = data.get("entry_id") or data.get("title")
            if key in seen:
                continue

            seen.add(key)
            results.append(formatted)
            added += 1
            if len(results) >= top_k:
                break

        return added

    def _single_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Internal method to search arXiv for a single query."""
        try:
            results: List[Mapping[str, Any]] = []
            seen = set()
            start = 0

            while len(results) < top_k:
                limit = min(top_k - len(results), ARXIV_PAGE_SIZE)
                response = self.client.get(
                    ARXIV_API_URL,
                    headers=self._build_headers(),
                    params=self._build_params(query, limit, start),
                )
                response.raise_for_status()

                entries = self._parse_entries(response.text)
                if not entries:
                    break

                added = self._append_entries(entries, results, seen, top_k)
                if len(entries) < limit or added == 0:
                    break

                start += len(entries)

            return results

        except Exception as e:
            logger.warning("arXiv search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Async internal method to search arXiv for a single query."""
        try:
            results: List[Mapping[str, Any]] = []
            seen = set()
            start = 0

            while len(results) < top_k:
                limit = min(top_k - len(results), ARXIV_PAGE_SIZE)
                response = await self.aclient.get(
                    ARXIV_API_URL,
                    headers=self._build_headers(),
                    params=self._build_params(query, limit, start),
                )
                response.raise_for_status()

                entries = self._parse_entries(response.text)
                if not entries:
                    break

                added = self._append_entries(entries, results, seen, top_k)
                if len(entries) < limit or added == 0:
                    break

                start += len(entries)

            return results

        except Exception as e:
            logger.warning("arXiv async search failed for query '%s': %s", query, e)
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
