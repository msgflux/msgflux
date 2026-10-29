import asyncio
from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Union

import httpx2

from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch, BaseWebSearch
from msgflux.data.retrievers.http import HTTPRetrieverMixin
from msgflux.data.retrievers.providers.http import build_fetch_entry, normalize_url
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebFetchRetriever, WebSearchRetriever
from msgflux.logger import init_logger

logger = init_logger(__name__)

API_KEY_ENV_NAME = "LINKUP_API_KEY"
DEFAULT_BASE_URL = "https://api.linkup.so"


def _response_detail(response: Optional[httpx2.Response]) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:300]
    if isinstance(payload, Mapping):
        for key in ("error", "message", "detail", "code"):
            if payload.get(key):
                return str(payload[key])
    return response.text.strip()[:300]


def _http_failure_detail(exc: httpx2.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    detail = f"HTTP {response.status_code}"
    body = _response_detail(response)
    return f"{detail}: {body}" if body else detail


class _LinkupAuthMixin:
    """Shared Linkup API-key lookup and request headers."""

    def _get_api_key(self) -> str:
        api_key = getenv(API_KEY_ENV_NAME)
        if api_key:
            return api_key

        raise ValueError(
            f"The Linkup API key is not available. Please set `{API_KEY_ENV_NAME}`"
        )

    def _build_headers(self) -> dict:
        api_key = self._get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


@register_retriever
class LinkupWebRetriever(
    _LinkupAuthMixin,
    HTTPRetrieverMixin,
    BaseWebSearch,
    BaseRetriever,
    WebSearchRetriever,
):
    """A Linkup client for retrieving web content with AI-powered search.

    This class interfaces with the Linkup API to provide search results
    with support for standard and deep agentic search.
    """

    provider = "linkup"

    def __init__(
        self,
        *,
        depth: Optional[str] = None,
        output_type: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_images: Optional[bool] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize LinkupWebRetriever.

        Requires the `LINKUP_API_KEY` environment variable to be set when a
        request is made.

        Args:
            depth:
                Search depth. Can be "standard" (fast) or "deep" (agentic).
                Defaults to "standard".
            output_type:
                Output type. Can be "searchResults" or "sourcedAnswer".
                Defaults to "searchResults".
            include_domains:
                List of domains to restrict search to. Defaults to None.
            exclude_domains:
                List of domains to exclude from search. Defaults to None.
            include_images:
                Whether to include images in results. Defaults to False.
            base_url:
                Linkup API base URL. Defaults to "https://api.linkup.so".
            timeout:
                Request timeout in seconds. Defaults to 30.

        !!! example

            ```python
            retriever = LinkupWebRetriever(depth="deep", output_type="sourcedAnswer")
            results = retriever(["latest AI news"], top_k=5)
            print(results)
            ```
        """
        self.depth = depth or "standard"
        self.output_type = output_type or "searchResults"
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.include_images = include_images or False
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.search_url = f"{self.base_url}/v1/search"
        self.timeout = timeout or 30.0
        self._initialize()

    def _build_search_kwargs(self, top_k: int) -> dict:
        """Build REST payload fields for Linkup search requests."""
        kwargs = {
            "depth": self.depth,
            "outputType": self.output_type,
            "maxResults": top_k,
            "includeImages": self.include_images,
        }

        if self.include_domains:
            kwargs["includeDomains"] = self.include_domains
        if self.exclude_domains:
            kwargs["excludeDomains"] = self.exclude_domains

        return kwargs

    def _build_payload(self, query: str, top_k: int) -> dict:
        return {"q": query, **self._build_search_kwargs(top_k)}

    def _parse_results(self, response: Mapping[str, Any], top_k: int) -> List[dict]:
        """Parse Linkup response into standard format."""
        results = []

        # Handle sourcedAnswer output type
        if self.output_type == "sourcedAnswer":
            for source in response.get("sources") or []:
                data = {
                    "title": source.get("name"),
                    "content": source.get("snippet"),
                    "url": source.get("url"),
                }
                results.append({"data": data})
                if len(results) >= top_k:
                    break
            return results

        # Handle searchResults output type
        for result in (response.get("results") or [])[:top_k]:
            data = {
                "title": result.get("name"),
                "content": result.get("content"),
                "url": result.get("url"),
            }
            results.append({"data": data})

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Linkup for a single query."""
        try:
            response = self.client.post(
                self.search_url,
                headers=self._build_headers(),
                json=self._build_payload(query, top_k),
            )
            response.raise_for_status()

            return self._parse_results(response.json(), top_k)
        except Exception as e:
            logger.warning("Linkup search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Linkup for a single query."""
        try:
            response = await self.aclient.post(
                self.search_url,
                headers=self._build_headers(),
                json=self._build_payload(query, top_k),
            )
            response.raise_for_status()

            return self._parse_results(response.json(), top_k)
        except Exception as e:
            logger.warning("Linkup search failed for query '%s': %s", query, e)
            return []

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Synchronous search for multiple queries."""
        results = []
        for query in queries:
            query_results = self._single_search(query, top_k)
            results.append(dotdict({"results": query_results}))
        return results

    async def _asearch(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Async search that runs multiple queries in parallel."""
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results


@register_retriever
class LinkupWebFetchRetriever(
    BaseWebFetch,
    _LinkupAuthMixin,
    HTTPRetrieverMixin,
    BaseRetriever,
    WebFetchRetriever,
):
    """Fetch a single webpage as Markdown with Linkup."""

    provider = "linkup"

    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        render_js: Optional[bool] = None,
        include_raw_content: Optional[bool] = None,
        extract_images: Optional[bool] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize LinkupWebFetchRetriever.

        Requires the `LINKUP_API_KEY` environment variable to be set when a
        request is made.

        Args:
            mode:
                Fetch strategy. Can be "standard" or "pro". Defaults to
                "standard".
            render_js:
                Whether Linkup should render JavaScript before extraction.
                Defaults to False.
            include_raw_content:
                Whether to ask Linkup to include raw page content alongside
                Markdown. Defaults to False.
            extract_images:
                Whether to ask Linkup to extract image URLs alongside Markdown.
                Defaults to False.
            base_url:
                Linkup API base URL. Defaults to "https://api.linkup.so".
            timeout:
                Request timeout in seconds. Defaults to 30.

        !!! example

            ```python
            retriever = LinkupWebFetchRetriever(render_js=True)
            results = retriever("https://docs.linkup.so")
            print(results)
            ```
        """
        self.mode = mode or "standard"
        self.render_js = bool(render_js)
        self.include_raw_content = bool(include_raw_content)
        self.extract_images = bool(extract_images)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.fetch_url = f"{self.base_url}/v1/fetch"
        self.timeout = timeout or 30.0
        self._initialize()

    def _build_payload(self, url: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "url": url,
            "mode": self.mode,
            "renderJs": self.render_js,
        }
        if self.include_raw_content:
            payload["includeRawContent"] = True
        if self.extract_images:
            payload["extractImages"] = True
        return payload

    def _parse_payload(
        self, url: str, payload: Mapping[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        markdown = payload.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            detail = payload.get("error") or payload.get("message")
            reason = f": {detail}" if detail else ": empty markdown response"
            raise RuntimeError(f"Linkup fetch failed for {url}{reason}")
        return build_fetch_entry(normalize_url(url), markdown)

    def _fetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        try:
            response = self.client.post(
                self.fetch_url,
                headers=self._build_headers(),
                json=self._build_payload(target_url),
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            detail = _http_failure_detail(exc)
            raise RuntimeError(f"Linkup fetch failed for {url}: {detail}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Linkup fetch failed for {url}: invalid JSON response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Linkup fetch failed for {url}: invalid response body")
        return self._parse_payload(target_url, payload)

    async def _afetch_one(self, url: str) -> Dict[str, Dict[str, str]]:
        target_url = normalize_url(url)
        try:
            response = await self.aclient.post(
                self.fetch_url,
                headers=self._build_headers(),
                json=self._build_payload(target_url),
            )
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            detail = _http_failure_detail(exc)
            raise RuntimeError(f"Linkup fetch failed for {url}: {detail}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Linkup fetch failed for {url}: invalid JSON response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Linkup fetch failed for {url}: invalid response body")
        return self._parse_payload(target_url, payload)

    def _fetch(self, urls: Union[str, List[str]]) -> List[Dict[str, Dict[str, str]]]:
        return [self._fetch_one(url) for url in urls]

    async def _afetch(
        self, urls: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, str]]]:
        results = []
        for url in urls:
            results.append(await self._afetch_one(url))
        return results
