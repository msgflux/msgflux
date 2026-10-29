"""Native HTTP client for the MediaWiki Action API."""

import asyncio
import re
from typing import Any, List, Mapping, Optional

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebSearchRetriever
from msgflux.logger import init_logger

logger = init_logger(__name__)

WIKIPEDIA_API_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"
USER_AGENT = "msgflux/1.0 (https://github.com/msgflux/msgflux)"
MAX_SEARCH_RESULTS = 500
MAX_IMAGES_PER_REQUEST = 500

_VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")
_SKIP_IMAGE_MARKERS = ("commons-logo", "wikimedia-logo", "edit-icon")


@register_retriever
class WikipediaWebRetriever(
    HTTPRetrieverMixin, BaseWebSearch, BaseRetriever, WebSearchRetriever
):
    """A customizable Wikipedia client for searching and retrieving Wikipedia content.

    This class provides a flexible interface to search Wikipedia articles, retrieve
    content with optional summaries, and extract images from pages.
    """

    provider = "wikipedia"
    to_ignore = ["client", "aclient", "documents"]

    def __init__(
        self,
        *,
        language: Optional[str] = None,
        summary: Optional[int] = None,
        return_images: Optional[bool] = None,
        max_return_images: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """Args:
            language:
                The language code for Wikipedia searches. Defaults to "en".
            summary:
                Number of sentences to include in summary.
                Defaults to None (full content).
            return_images:
                Whether to include images in the results. Defaults to False.
            max_return_images:
                Maximum number of images returned. Defaults to 5.
            timeout:
                Request timeout in seconds. Defaults to 30.

        !!! example

            ```python
            retriever = WikipediaWebRetriever(
                language="pt", return_images=True, summary=3
            )
            results = retriever(
                ["Python programming", "Machine learning"], top_k=2
            )
            print(results)
            ```
        """
        # Apply defaults
        if language is None:
            language = "en"
        if return_images is None:
            return_images = False
        if max_return_images is None:
            max_return_images = 5

        self.language = language
        self.max_return_images = max_return_images
        self.return_images = return_images
        self.summary = summary
        self.timeout = timeout or 30.0
        self._initialize()

    def set_language(self, language: str) -> None:
        """Change the Wikipedia language setting for this instance."""
        self.language = language

    def _get_api_url(self) -> str:
        """Return the MediaWiki Action API endpoint for this language."""
        return WIKIPEDIA_API_TEMPLATE.format(language=self.language)

    def _build_headers(self) -> dict:
        """Build request headers for the MediaWiki API."""
        return {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _build_search_params(self, query: str, top_k: int) -> dict:
        """Build query parameters for a MediaWiki search request."""
        return {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": min(top_k, MAX_SEARCH_RESULTS),
            "srnamespace": 0,
            "format": "json",
            "formatversion": 2,
        }

    def _build_page_params(self, title: str) -> dict:
        """Build query parameters for one page's extraction and links."""
        return {
            "action": "query",
            "prop": "extracts|pageprops|links",
            "titles": title,
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "plnamespace": 0,
            "pllimit": 1,
            "format": "json",
            "formatversion": 2,
        }

    def _build_image_params(self, title: str) -> dict:
        """Build query parameters for a page's embedded image URLs."""
        image_limit = min(max(self.max_return_images * 5, 10), MAX_IMAGES_PER_REQUEST)
        return {
            "action": "query",
            "generator": "images",
            "titles": title,
            "gimlimit": image_limit,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
            "formatversion": 2,
        }

    @staticmethod
    def _collect_pages(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """Return MediaWiki query pages for both format versions."""
        pages = (data.get("query") or {}).get("pages") or []
        if isinstance(pages, Mapping):
            return list(pages.values())
        return pages

    def _get_json(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute a MediaWiki API request and validate its response."""
        response = self.client.get(
            self._get_api_url(),
            headers=self._build_headers(),
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        self._raise_api_error(data)
        return data

    async def _aget_json(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute an async MediaWiki API request and validate its response."""
        response = await self.aclient.get(
            self._get_api_url(),
            headers=self._build_headers(),
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        self._raise_api_error(data)
        return data

    @staticmethod
    def _raise_api_error(data: Mapping[str, Any]) -> None:
        """Raise when MediaWiki returns an API-level error payload."""
        if not isinstance(data, Mapping):
            return

        error = data.get("error")
        if not error:
            return

        info = error.get("info") if isinstance(error, Mapping) else error
        raise ValueError(f"MediaWiki API error: {info}")

    def _parse_search_titles(self, data: Mapping[str, Any], top_k: int) -> List[str]:
        """Extract unique result titles from a MediaWiki search response."""
        search_results = (data.get("query") or {}).get("search") or []
        titles = []
        seen = set()

        for result in search_results:
            title = result.get("title")
            if not title:
                continue

            key = title.casefold()
            if key in seen:
                continue

            seen.add(key)
            titles.append(title)
            if len(titles) >= top_k:
                break

        return titles

    def _first_page(self, data: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Return the first page from a MediaWiki query response."""
        pages = self._collect_pages(data)
        if not pages:
            return None
        return pages[0]

    def _fetch_page(
        self, title: str, visited: Optional[set] = None
    ) -> Optional[Mapping[str, Any]]:
        """Fetch a page, following the first link for disambiguation pages."""
        visited = visited or set()
        key = title.casefold()
        if key in visited:
            return None
        visited.add(key)

        page = self._first_page(self._get_json(self._build_page_params(title)))
        if page is None or page.get("missing"):
            return None

        pageprops = page.get("pageprops") or {}
        if "disambiguation" in pageprops:
            links = page.get("links") or []
            if links and links[0].get("title"):
                return self._fetch_page(links[0]["title"], visited)
            return None

        return page

    async def _afetch_page(
        self, title: str, visited: Optional[set] = None
    ) -> Optional[Mapping[str, Any]]:
        """Async version of ``_fetch_page``."""
        visited = visited or set()
        key = title.casefold()
        if key in visited:
            return None
        visited.add(key)

        page = self._first_page(await self._aget_json(self._build_page_params(title)))
        if page is None or page.get("missing"):
            return None

        pageprops = page.get("pageprops") or {}
        if "disambiguation" in pageprops:
            links = page.get("links") or []
            if links and links[0].get("title"):
                return await self._afetch_page(links[0]["title"], visited)
            return None

        return page

    def _process_content(self, content: str, title: str) -> str:
        """Process page content based on summary parameter.

        Args:
            content:
                Raw page content.
            title:
                Page title.

        Returns:
            str: Processed content (title + summary or title + full content).
        """
        if self.summary is not None:
            # Extract specified number of sentences
            sentences = self._extract_sentences(content)
            summary_text = " ".join(sentences[: self.summary])
            return f"{title}\n\n{summary_text}"
        else:
            return f"{title}\n\n{content}"

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using regex.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        # Clean text and split into sentences
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Split by sentence endings, considering abbreviations
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Filter out very short sentences and clean them
        clean_sentences = []
        for raw_sentence in sentences:
            sentence = raw_sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                clean_sentences.append(sentence)

        return clean_sentences

    @staticmethod
    def _is_valid_image_url(url: str) -> bool:
        """Return whether an image URL matches the previous provider filters."""
        lower_url = url.lower()
        if not any(ext in lower_url for ext in _VALID_IMAGE_EXTENSIONS):
            return False
        return not any(skip in lower_url for skip in _SKIP_IMAGE_MARKERS)

    def _parse_image_urls(self, data: Mapping[str, Any]) -> List[str]:
        """Extract valid image URLs from a MediaWiki imageinfo response."""
        urls = []
        for page in self._collect_pages(data):
            imageinfo = page.get("imageinfo") or []
            if not imageinfo:
                continue

            url = imageinfo[0].get("url")
            if url and self._is_valid_image_url(url):
                urls.append(url)

        return urls[: self.max_return_images]

    def _get_images(self, page: Mapping[str, Any]) -> List[str]:
        """Extract image URLs from a Wikipedia page.

        Args:
            page: MediaWiki page mapping.

        Returns:
            List[str]: List of image URLs.
        """
        title = page.get("title")
        if not title:
            return []

        try:
            data = self._get_json(self._build_image_params(title))
            return self._parse_image_urls(data)
        except Exception as e:
            logger.debug("Error retrieving Wikipedia images: %s", e)
            return []

    async def _aget_images(self, page: Mapping[str, Any]) -> List[str]:
        """Async version of ``_get_images``."""
        title = page.get("title")
        if not title:
            return []

        try:
            data = await self._aget_json(self._build_image_params(title))
            return self._parse_image_urls(data)
        except Exception as e:
            logger.debug("Error retrieving Wikipedia images: %s", e)
            return []

    def _format_page(self, page: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Convert a MediaWiki page into a standard web search result."""
        extract = page.get("extract")
        title = page.get("title")
        if not isinstance(extract, str) or not title:
            return None

        content = self._process_content(extract, title)
        return {"data": {"title": title, "content": content}}

    def _single_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Internal method to search Wikipedia for a single query."""
        if top_k <= 0:
            return []

        try:
            data = self._get_json(self._build_search_params(query, top_k))
        except Exception as e:
            logger.warning("Wikipedia search failed for query '%s': %s", query, e)
            return []

        results = []
        for title in self._parse_search_titles(data, top_k):
            try:
                page = self._fetch_page(title)
                if page is None:
                    continue

                result = self._format_page(page)
                if result is None:
                    continue

                if self.return_images:
                    result["images"] = self._get_images(page)

                results.append(result)
            except Exception as e:
                logger.debug(str(e))
                continue

        return results

    async def _asingle_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Async internal method to search Wikipedia for a single query."""
        if top_k <= 0:
            return []

        try:
            data = await self._aget_json(self._build_search_params(query, top_k))
        except Exception as e:
            logger.warning("Wikipedia async search failed for query '%s': %s", query, e)
            return []

        results = []
        for title in self._parse_search_titles(data, top_k):
            try:
                page = await self._afetch_page(title)
                if page is None:
                    continue

                result = self._format_page(page)
                if result is None:
                    continue

                if self.return_images:
                    result["images"] = await self._aget_images(page)

                results.append(result)
            except Exception as e:
                logger.debug(str(e))
                continue

        return results

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        args_list = [(query,) for query in queries]
        kwargs_list = [{"top_k": top_k} for _ in queries]
        query_results = F.map_gather(
            self._single_search, args_list=args_list, kwargs_list=kwargs_list
        )
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results

    async def _asearch(self, queries: List[str], top_k: int):
        """Async search that runs multiple queries in parallel."""
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results
